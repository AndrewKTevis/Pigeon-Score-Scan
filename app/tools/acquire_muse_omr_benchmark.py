#!/usr/bin/env python3
"""Acquire a pinned, leakage-isolated subset of the Muse OMR benchmark.

The official benchmark pairs scan-degraded PDFs with MuseScore ground truth.
It is deliberately stored under ``external/benchmarks`` and must not be used
for model fitting.  A stable hash-ranked subset makes incremental acquisition
reproducible while retaining the option to acquire the complete 1077 pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from app.tools.muse_omr_contract import SCAN_DEGRADED_IMAGE_ORIGIN


REPOSITORY = "musegroup/omr_benchmark"
REVISION = "e27f6a8634e80ad0997af8a806c8dc00e45c4a07"
LICENSE = "CC0-1.0"
PAIR_COUNT = 1077
MANIFEST_PATH = "benchmark_dataset.json"
MANIFEST_BYTES = 123_759
MANIFEST_SHA256 = "e61a961611ba095c34dd2f12e0399a52f3bdc5c3f4c8f2bb2d937d3c27e00957"
DEFAULT_HOLDOUT_PAIR_COUNT = 420
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "training_data"
    / "external"
    / "benchmarks"
    / f"muse_omr_{REVISION[:10]}"
)
DEFAULT_WORK_CATALOG = (
    Path(__file__).resolve().parents[2]
    / "training_data"
    / "external"
    / "catalogs"
    / f"muse_omr_work_catalog_{REVISION[:10]}"
    / "work-catalog.json"
)
USER_AGENT = "ScoreScan-Muse-OMR-benchmark-acquirer/1"
_PAIR_PATH = re.compile(r"^(pdf|mscz)/score_file_(\d+)\.(pdf|mscz)$")


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    sha256: str | None


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(value: str) -> Path:
    posix = PurePosixPath(value)
    if (
        posix.is_absolute()
        or ".." in posix.parts
        or "\\" in value
        or any(":" in part or "\x00" in part for part in posix.parts)
    ):
        raise ValueError(f"unsafe repository path: {value!r}")
    if not posix.parts or any(part in {"", "."} for part in posix.parts):
        raise ValueError(f"invalid repository path: {value!r}")
    return Path(*posix.parts)


def parse_pair_manifest(payload: bytes) -> dict[int, tuple[str, str]]:
    if len(payload) != MANIFEST_BYTES or sha256_bytes(payload) != MANIFEST_SHA256:
        raise ValueError("pinned Muse OMR manifest size or SHA-256 mismatch")
    raw = json.loads(payload)
    if not isinstance(raw, dict) or len(raw) != PAIR_COUNT:
        raise ValueError("unexpected Muse OMR pair manifest shape")
    pairs: dict[int, tuple[str, str]] = {}
    seen_paths: set[str] = set()
    for raw_key, row in raw.items():
        if not isinstance(row, dict):
            raise ValueError(f"invalid pair row {raw_key!r}")
        try:
            pair_id = int(raw_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid pair id {raw_key!r}") from exc
        if str(pair_id) != str(raw_key) or not 0 <= pair_id < PAIR_COUNT:
            raise ValueError(f"out-of-range or non-canonical pair id {raw_key!r}")
        score = str(row.get("score", ""))
        pdf = str(row.get("pdf_image", ""))
        for path, expected_kind, expected_suffix in (
            (score, "mscz", "mscz"),
            (pdf, "pdf", "pdf"),
        ):
            safe_relative_path(path)
            match = _PAIR_PATH.fullmatch(path)
            if (
                match is None
                or match.group(1) != expected_kind
                or match.group(3) != expected_suffix
                or int(match.group(2)) != pair_id
            ):
                raise ValueError(f"pair {pair_id} has inconsistent path {path!r}")
            if path in seen_paths:
                raise ValueError(f"duplicate pair path: {path}")
            seen_paths.add(path)
        pairs[pair_id] = (score, pdf)
    if set(pairs) != set(range(PAIR_COUNT)):
        raise ValueError("Muse OMR pair ids are not contiguous")
    return pairs


def stable_pair_ids(
    available: Iterable[int], *, limit: int | None, seed: int
) -> list[int]:
    ids = sorted(set(int(value) for value in available))
    if limit is None:
        return ids
    if limit <= 0:
        raise ValueError("pair limit must be positive")
    ranked = sorted(
        ids,
        key=lambda pair_id: hashlib.sha256(
            f"{seed}\0{pair_id}".encode("ascii")
        ).hexdigest(),
    )
    return sorted(ranked[:limit])


def augmented_pair_ids(
    available: Iterable[int],
    *,
    limit: int | None,
    seed: int,
    required_pair_ids: Iterable[int] = (),
) -> tuple[list[int], list[int]]:
    """Add annotation-stratified evidence without replacing the frozen base."""

    available_ids = set(int(value) for value in available)
    required = sorted(set(int(value) for value in required_pair_ids))
    unknown = sorted(set(required) - available_ids)
    if unknown:
        raise ValueError(f"required benchmark pair ids are unavailable: {unknown}")
    base = stable_pair_ids(available_ids, limit=limit, seed=seed)
    return sorted(set(base) | set(required)), required


def annotation_stratified_work_fingerprints(
    base_pair_ids: Iterable[int],
    required_pair_ids: Iterable[int],
    work_by_pair: dict[int, str],
) -> list[str]:
    """Reject variant-level additions that do not add independent works."""

    base_works = {work_by_pair[int(pair_id)] for pair_id in base_pair_ids}
    required_rows = [
        (int(pair_id), work_by_pair[int(pair_id)])
        for pair_id in required_pair_ids
    ]
    overlapping = [
        pair_id
        for pair_id, fingerprint in required_rows
        if fingerprint in base_works
    ]
    fingerprints = [fingerprint for _pair_id, fingerprint in required_rows]
    if overlapping:
        raise ValueError(
            "annotation-stratified pairs duplicate base holdout works: "
            f"{overlapping}"
        )
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("annotation-stratified pairs contain duplicate works")
    return sorted(fingerprints)


def next_link(value: str | None) -> str | None:
    if not value:
        return None
    for item in value.split(","):
        match = re.fullmatch(r'\s*<([^>]+)>\s*;\s*rel="([^"]+)"\s*', item)
        if match and match.group(2) == "next":
            return match.group(1)
    return None


def _request_bytes(url: str, *, timeout: float) -> tuple[bytes, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
        return payload, response.headers


def fetch_manifest(*, timeout: float) -> bytes:
    encoded_path = urllib.parse.quote(MANIFEST_PATH, safe="/")
    url = (
        f"https://huggingface.co/datasets/{REPOSITORY}/resolve/"
        f"{REVISION}/{encoded_path}?download=true"
    )
    payload, _headers = _request_bytes(url, timeout=timeout)
    # parse_pair_manifest performs the immutable byte-level verification.
    parse_pair_manifest(payload)
    return payload


def fetch_remote_file_index(*, timeout: float) -> dict[str, RemoteFile]:
    url: str | None = (
        f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}"
        "?recursive=true&expand=true&limit=100"
    )
    files: dict[str, RemoteFile] = {}
    while url is not None:
        payload, headers = _request_bytes(url, timeout=timeout)
        rows = json.loads(payload)
        if not isinstance(rows, list):
            raise ValueError("unexpected Hugging Face tree response")
        for row in rows:
            if not isinstance(row, dict) or row.get("type") != "file":
                continue
            path = str(row.get("path", ""))
            safe_relative_path(path)
            size = int(row.get("size", -1))
            if size < 0:
                raise ValueError(f"missing remote size for {path}")
            lfs = row.get("lfs")
            digest = (
                str(lfs.get("oid"))
                if isinstance(lfs, dict) and lfs.get("oid")
                else None
            )
            if digest is not None and not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"invalid remote SHA-256 for {path}")
            if path in files:
                raise ValueError(f"duplicate remote path: {path}")
            files[path] = RemoteFile(path=path, size=size, sha256=digest)
        url = next_link(headers.get("Link"))
    return files


def selected_remote_files(
    pairs: dict[int, tuple[str, str]],
    pair_ids: Iterable[int],
    remote_index: dict[str, RemoteFile],
) -> list[RemoteFile]:
    result: list[RemoteFile] = []
    for pair_id in pair_ids:
        if pair_id not in pairs:
            raise ValueError(f"unknown pair id: {pair_id}")
        for path in pairs[pair_id]:
            remote = remote_index.get(path)
            if remote is None:
                raise ValueError(f"remote benchmark file is missing: {path}")
            result.append(remote)
    return sorted(result, key=lambda item: item.path)


def _download_once(
    remote: RemoteFile,
    destination: Path,
    *,
    timeout: float,
) -> dict[str, Any]:
    relative = safe_relative_path(remote.path)
    if destination.name != relative.name:
        raise ValueError("download destination does not match repository path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != remote.size:
            raise ValueError(f"existing benchmark file has wrong size: {destination}")
        digest = sha256_file(destination)
        if remote.sha256 is not None and digest != remote.sha256:
            raise ValueError(f"existing benchmark file has wrong SHA-256: {destination}")
        return {"status": "already_present", "sha256": digest}

    temporary = destination.with_suffix(destination.suffix + ".part")
    existing = temporary.stat().st_size if temporary.exists() else 0
    if existing > remote.size:
        raise ValueError(f"partial benchmark file is too large: {temporary}")
    headers = {"User-Agent": USER_AGENT}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    encoded_path = urllib.parse.quote(remote.path, safe="/")
    url = (
        f"https://huggingface.co/datasets/{REPOSITORY}/resolve/"
        f"{REVISION}/{encoded_path}?download=true"
    )
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = int(getattr(response, "status", response.getcode()) or 0)
        append = existing > 0 and status == 206
        with temporary.open("ab" if append else "wb") as output:
            while chunk := response.read(4 * 1024 * 1024):
                output.write(chunk)
    if temporary.stat().st_size != remote.size:
        raise ValueError(
            f"download size mismatch for {remote.path}: "
            f"expected {remote.size}, got {temporary.stat().st_size}"
        )
    digest = sha256_file(temporary)
    if remote.sha256 is not None and digest != remote.sha256:
        raise ValueError(f"download SHA-256 mismatch for {remote.path}")
    os.replace(temporary, destination)
    return {"status": "downloaded", "sha256": digest}


def download_file(
    remote: RemoteFile,
    output_dir: Path,
    *,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    destination = output_dir / safe_relative_path(remote.path)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            result = _download_once(remote, destination, timeout=timeout)
            return {
                **asdict(remote),
                "local_path": str(destination.resolve()),
                **result,
            }
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {remote.path}") from last_error


def reuse_or_download_file(
    remote: RemoteFile,
    output_dir: Path,
    *,
    reuse_dirs: tuple[Path, ...],
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    destination = output_dir / safe_relative_path(remote.path)
    if destination.exists():
        return download_file(
            remote,
            output_dir,
            timeout=timeout,
            retries=retries,
        )
    for reuse_dir in reuse_dirs:
        source = reuse_dir / safe_relative_path(remote.path)
        if not source.is_file() or source.is_symlink():
            continue
        if source.stat().st_size != remote.size:
            raise ValueError(f"reuse candidate has wrong size: {source}")
        digest = sha256_file(source)
        if remote.sha256 is not None and digest != remote.sha256:
            raise ValueError(f"reuse candidate has wrong SHA-256: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
            status = "hardlinked"
        except OSError:
            shutil.copy2(source, destination)
            status = "copied"
        return {
            **asdict(remote),
            "local_path": str(destination.resolve()),
            "status": status,
            "sha256": digest,
        }
    return download_file(
        remote,
        output_dir,
        timeout=timeout,
        retries=retries,
    )


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _pdf_coverage(
    output_dir: Path,
    pairs: dict[int, tuple[str, str]],
    selected_ids: list[int],
    work_by_pair: dict[int, str],
) -> dict[str, Any]:
    import fitz

    rows: list[dict[str, Any]] = []
    independent_rows: list[dict[str, Any]] = []
    seen_works: set[str] = set()
    for pair_id in selected_ids:
        pdf_path = output_dir / safe_relative_path(pairs[pair_id][1])
        with fitz.open(pdf_path) as document:
            pages = int(document.page_count)
        if pages <= 0:
            raise ValueError(f"benchmark PDF has no pages: {pdf_path}")
        fingerprint = work_by_pair[pair_id]
        row = {
            "pair_id": pair_id,
            "work_fingerprint": fingerprint,
            "pages": pages,
        }
        rows.append(row)
        if fingerprint not in seen_works:
            seen_works.add(fingerprint)
            independent_rows.append(row)
    if len(seen_works) != len(set(work_by_pair[pair_id] for pair_id in selected_ids)):
        raise RuntimeError("independent PDF coverage accounting failed")
    return {
        "selected_pdf_page_count": sum(int(row["pages"]) for row in rows),
        "selected_independent_work_pdf_page_count": sum(
            int(row["pages"]) for row in independent_rows
        ),
        "pair_pdf_page_counts": rows,
        "independent_work_pdf_representatives": independent_rows,
    }


def acquire(
    *,
    output_dir: Path,
    work_catalog: Path,
    limit: int | None,
    seed: int,
    workers: int,
    timeout: float,
    retries: int,
    maximum_bytes: int,
    required_pair_ids: Iterable[int] = (),
    reuse_dirs: Iterable[Path] = (),
) -> dict[str, Any]:
    if workers <= 0 or retries <= 0 or maximum_bytes <= 0:
        raise ValueError("workers, retries, and maximum bytes must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = fetch_manifest(timeout=timeout)
    pairs = parse_pair_manifest(manifest)
    selected_ids, required_ids = augmented_pair_ids(
        pairs,
        limit=limit,
        seed=seed,
        required_pair_ids=required_pair_ids,
    )
    from app.tools.build_muse_omr_work_catalog import load_work_catalog

    work_by_pair = load_work_catalog(work_catalog)
    base_ids = stable_pair_ids(pairs, limit=limit, seed=seed)
    required_work_fingerprints = (
        annotation_stratified_work_fingerprints(
            base_ids,
            required_ids,
            work_by_pair,
        )
    )
    selected_work_fingerprints = sorted(
        {work_by_pair[pair_id] for pair_id in selected_ids}
    )
    remote_index = fetch_remote_file_index(timeout=timeout)
    files = selected_remote_files(pairs, selected_ids, remote_index)
    expected_bytes = sum(item.size for item in files)
    if expected_bytes > maximum_bytes:
        raise ValueError(
            f"selected benchmark bytes exceed limit: {expected_bytes} > {maximum_bytes}"
        )
    (output_dir / MANIFEST_PATH).write_bytes(manifest)
    selection = {
        "format": 1,
        "repository": REPOSITORY,
        "revision": REVISION,
        "role": "external_scan_degraded_development_benchmark_not_training",
        "source_image_origin": SCAN_DEGRADED_IMAGE_ORIGIN,
        "production_evidence_eligible": False,
        "license": LICENSE,
        "selection_seed": seed,
        "base_pair_limit": limit,
        "annotation_stratified_required_pair_ids": required_ids,
        "annotation_stratified_required_work_fingerprints": sorted(
            required_work_fingerprints
        ),
        "selected_pair_count": len(selected_ids),
        "selected_pair_ids": selected_ids,
        "selected_work_count": len(selected_work_fingerprints),
        "selected_work_fingerprints": selected_work_fingerprints,
        "pair_work_fingerprints": [
            {
                "pair_id": pair_id,
                "work_fingerprint": work_by_pair[pair_id],
            }
            for pair_id in selected_ids
        ],
        "work_catalog_sha256": sha256_file(work_catalog),
        "expected_download_bytes": expected_bytes,
    }
    atomic_json(output_dir / "selection.json", selection)

    downloaded: list[dict[str, Any]] = []
    verified_reuse_dirs = tuple(
        path.resolve()
        for path in reuse_dirs
        if path.resolve() != output_dir.resolve()
    ) + (work_catalog.parent.resolve(),)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                reuse_or_download_file,
                remote,
                output_dir,
                reuse_dirs=verified_reuse_dirs,
                timeout=timeout,
                retries=retries,
            ): remote.path
            for remote in files
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            downloaded.append(row)
            if (
                row["status"] != "already_present"
                or completed % 50 == 0
                or completed == len(files)
            ):
                print(
                    f"[{completed}/{len(files)}] "
                    f"{row['status']}: {row['path']}",
                    flush=True,
                )

    coverage = _pdf_coverage(
        output_dir,
        pairs,
        selected_ids,
        work_by_pair,
    )
    report = {
        **selection,
        **coverage,
        "downloaded_bytes": sum(int(row["size"]) for row in downloaded),
        "files": sorted(downloaded, key=lambda row: str(row["path"])),
    }
    atomic_json(output_dir / "provenance.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--work-catalog",
        type=Path,
        default=DEFAULT_WORK_CATALOG,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_HOLDOUT_PAIR_COUNT,
        help="stable number of pairs to acquire; use 0 for all pairs",
    )
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--include-pair-id",
        type=int,
        action="append",
        default=[],
        help=(
            "add a pinned annotation-stratified pair without replacing the "
            "stable base sample; may be repeated"
        ),
    )
    parser.add_argument(
        "--reuse-dir",
        type=Path,
        action="append",
        default=[],
        help="verified existing benchmark directory used for hardlink reuse",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--max-download-gb", type=float, default=20.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    if args.max_download_gb <= 0:
        raise SystemExit("--max-download-gb must be positive")
    report = acquire(
        output_dir=args.output_dir.resolve(),
        work_catalog=args.work_catalog.resolve(),
        limit=None if args.limit == 0 else args.limit,
        seed=args.seed,
        workers=args.workers,
        timeout=args.timeout_seconds,
        retries=args.retries,
        maximum_bytes=int(args.max_download_gb * 1024**3),
        required_pair_ids=args.include_pair_id,
        reuse_dirs=args.reuse_dir,
    )
    print(
        json.dumps(
            {
                "selected_pair_count": report["selected_pair_count"],
                "selected_work_count": report["selected_work_count"],
                "files": len(report["files"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
