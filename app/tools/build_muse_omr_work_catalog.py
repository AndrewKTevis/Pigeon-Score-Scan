#!/usr/bin/env python3
"""Build a pinned Muse OMR work catalog for leakage-safe grouped splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from lxml import etree

from app.tools.acquire_muse_omr_benchmark import (
    LICENSE,
    PAIR_COUNT,
    REPOSITORY,
    REVISION,
    RemoteFile,
    atomic_json,
    download_file,
    fetch_manifest,
    fetch_remote_file_index,
    parse_pair_manifest,
    sha256_file,
)


FINGERPRINT_VERSION = "mscx-c14n-without-eid-v1"
MAXIMUM_MSCX_BYTES = 128 * 1024 * 1024
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "training_data"
    / "external"
    / "catalogs"
    / f"muse_omr_work_catalog_{REVISION[:10]}"
)
DEFAULT_REUSE_DIRS = (
    PROJECT_ROOT
    / "training_data"
    / "external"
    / "benchmarks"
    / f"muse_omr_{REVISION[:10]}",
    PROJECT_ROOT
    / "training_data"
    / "external"
    / "training"
    / f"muse_omr_scan_train_{REVISION[:10]}",
)


def mscx_payload_fingerprint(payload: bytes) -> str:
    if not payload or len(payload) > MAXIMUM_MSCX_BYTES:
        raise ValueError("unsafe MSCX payload size")
    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        remove_blank_text=False,
        huge_tree=False,
    )
    root = etree.fromstring(payload, parser=parser)
    for element in root.xpath("//*[local-name()='eid']"):
        parent = element.getparent()
        if parent is not None:
            parent.remove(element)
    canonical = etree.tostring(
        root,
        method="c14n",
        exclusive=False,
        with_comments=False,
    )
    return hashlib.sha256(canonical).hexdigest()


def work_fingerprint(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    with ZipFile(path) as archive:
        candidates = [
            item
            for item in archive.infolist()
            if (
                not item.is_dir()
                and item.filename.lower().endswith(".mscx")
                and "/" not in item.filename.replace("\\", "/")
            )
        ]
        if len(candidates) != 1:
            raise ValueError(f"expected exactly one MSCX score in {path}")
        score = candidates[0]
        if score.file_size <= 0 or score.file_size > MAXIMUM_MSCX_BYTES:
            raise ValueError(f"unsafe MSCX size in {path}")
        payload = archive.read(score)
    return mscx_payload_fingerprint(payload)


def load_work_catalog(path: Path) -> dict[int, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = report.get("pair_work_fingerprints")
    if (
        report.get("name") != "scorescan-muse-omr-work-catalog-v1"
        or report.get("repository") != REPOSITORY
        or report.get("revision") != REVISION
        or report.get("fingerprint_version") != FINGERPRINT_VERSION
        or int(report.get("pair_count", -1)) != PAIR_COUNT
        or not isinstance(rows, list)
        or len(rows) != PAIR_COUNT
    ):
        raise ValueError("Muse OMR work catalog contract failed")
    mapping: dict[int, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid Muse OMR work catalog row")
        pair_id = int(row.get("pair_id", -1))
        fingerprint = str(row.get("work_fingerprint", ""))
        if (
            pair_id in mapping
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise ValueError("invalid Muse OMR work fingerprint")
        mapping[pair_id] = fingerprint
    if set(mapping) != set(range(PAIR_COUNT)):
        raise ValueError("Muse OMR work catalog pair ids are incomplete")
    if int(report.get("work_count", -1)) != len(set(mapping.values())):
        raise ValueError("Muse OMR work catalog work count is inconsistent")
    return mapping


def _reuse_or_download(
    remote: RemoteFile,
    *,
    output_dir: Path,
    reuse_dirs: tuple[Path, ...],
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    destination = output_dir / remote.path
    if destination.is_file():
        return download_file(
            remote,
            output_dir,
            timeout=timeout,
            retries=retries,
        )
    for reuse_dir in reuse_dirs:
        source = reuse_dir / remote.path
        if not source.is_file() or source.is_symlink():
            continue
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
            "path": remote.path,
            "size": destination.stat().st_size,
            "sha256": digest,
            "status": status,
        }
    return download_file(
        remote,
        output_dir,
        timeout=timeout,
        retries=retries,
    )


def build_catalog(
    *,
    output_dir: Path,
    reuse_dirs: tuple[Path, ...],
    workers: int,
    timeout: float,
    retries: int,
    maximum_bytes: int,
) -> dict[str, Any]:
    if workers <= 0 or retries <= 0 or maximum_bytes <= 0:
        raise ValueError("workers, retries, and maximum bytes must be positive")
    manifest = fetch_manifest(timeout=timeout)
    pairs = parse_pair_manifest(manifest)
    remote_index = fetch_remote_file_index(timeout=timeout)
    score_files = [
        remote_index[pairs[pair_id][0]]
        for pair_id in sorted(pairs)
    ]
    expected_bytes = sum(item.size for item in score_files)
    if expected_bytes > maximum_bytes:
        raise ValueError(
            f"Muse OMR score catalog exceeds byte limit: "
            f"{expected_bytes} > {maximum_bytes}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _reuse_or_download,
                remote,
                output_dir=output_dir,
                reuse_dirs=reuse_dirs,
                timeout=timeout,
                retries=retries,
            ): remote.path
            for remote in score_files
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            downloaded.append(row)
            if (
                row["status"] != "already_present"
                or completed % 50 == 0
                or completed == len(score_files)
            ):
                print(
                    f"[{completed}/{len(score_files)}] "
                    f"{row['status']}: {row['path']}",
                    flush=True,
                )

    pair_rows: list[dict[str, Any]] = []
    works: dict[str, list[int]] = {}
    for pair_id in sorted(pairs):
        path = output_dir / pairs[pair_id][0]
        fingerprint = work_fingerprint(path)
        pair_rows.append(
            {
                "pair_id": pair_id,
                "mscz_path": pairs[pair_id][0],
                "mscz_sha256": sha256_file(path),
                "work_fingerprint": fingerprint,
            }
        )
        works.setdefault(fingerprint, []).append(pair_id)
    report = {
        "format": 1,
        "name": "scorescan-muse-omr-work-catalog-v1",
        "repository": REPOSITORY,
        "revision": REVISION,
        "license": LICENSE,
        "role": "metadata_catalog_not_training_or_evaluation",
        "fingerprint_version": FINGERPRINT_VERSION,
        "pair_count": len(pair_rows),
        "work_count": len(works),
        "expected_mscz_bytes": expected_bytes,
        "pair_work_fingerprints": pair_rows,
        "works": [
            {
                "work_fingerprint": fingerprint,
                "pair_ids": pair_ids,
            }
            for fingerprint, pair_ids in sorted(works.items())
        ],
        "files": sorted(downloaded, key=lambda row: str(row["path"])),
    }
    atomic_json(output_dir / "work-catalog.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reuse-dir",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--max-download-gb", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reuse_dirs = tuple(
        path.resolve()
        for path in (
            args.reuse_dir
            if args.reuse_dir is not None
            else DEFAULT_REUSE_DIRS
        )
        if path.is_dir()
    )
    report = build_catalog(
        output_dir=args.output_dir.resolve(),
        reuse_dirs=reuse_dirs,
        workers=args.workers,
        timeout=args.timeout_seconds,
        retries=args.retries,
        maximum_bytes=int(args.max_download_gb * 1024**3),
    )
    print(
        json.dumps(
            {
                "pair_count": report["pair_count"],
                "work_count": report["work_count"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
