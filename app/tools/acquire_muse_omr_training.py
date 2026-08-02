#!/usr/bin/env python3
"""Acquire a pinned Muse OMR training split disjoint from the dev holdout.

The public CC0 corpus contains paired scan-degraded PDFs and MuseScore truth.  This
tool deliberately refuses every pair reserved by the external benchmark selection,
so scan-domain fitting cannot contaminate ScoreScan's independent development gate.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from app.tools.acquire_muse_omr_benchmark import (
    LICENSE,
    MANIFEST_PATH,
    REPOSITORY,
    REVISION,
    atomic_json,
    fetch_manifest,
    fetch_remote_file_index,
    reuse_or_download_file,
    parse_pair_manifest,
    selected_remote_files,
    sha256_file,
    stable_pair_ids,
)
from app.tools.muse_omr_contract import SCAN_DEGRADED_IMAGE_ORIGIN

DEFAULT_TRAINING_PAIR_COUNT = 384


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HOLDOUT_SELECTION = (
    PROJECT_ROOT
    / "training_data"
    / "external"
    / "benchmarks"
    / f"muse_omr_{REVISION[:10]}"
    / "selection.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "training_data"
    / "external"
    / "training"
    / f"muse_omr_scan_train_{REVISION[:10]}"
)
DEFAULT_WORK_CATALOG = (
    PROJECT_ROOT
    / "training_data"
    / "external"
    / "catalogs"
    / f"muse_omr_work_catalog_{REVISION[:10]}"
    / "work-catalog.json"
)


def load_reserved_selection(
    path: Path,
) -> tuple[tuple[int, ...], frozenset[str], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("repository") != REPOSITORY
        or payload.get("revision") != REVISION
        or payload.get("role")
        != "external_scan_degraded_development_benchmark_not_training"
        or payload.get("source_image_origin")
        != SCAN_DEGRADED_IMAGE_ORIGIN
        or payload.get("production_evidence_eligible") is not False
    ):
        raise ValueError("holdout selection does not identify the pinned benchmark")
    raw_ids = payload.get("selected_pair_ids")
    raw_works = payload.get("selected_work_fingerprints")
    pair_rows = payload.get("pair_work_fingerprints")
    work_catalog_sha256 = str(payload.get("work_catalog_sha256", ""))
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError("holdout selection has no reserved pair ids")
    if (
        not isinstance(raw_works, list)
        or not raw_works
        or not isinstance(pair_rows, list)
        or len(pair_rows) != len(raw_ids)
        or len(work_catalog_sha256) != 64
    ):
        raise ValueError("holdout selection has no work-level isolation data")
    ids = tuple(sorted(int(value) for value in raw_ids))
    if len(ids) != len(set(ids)):
        raise ValueError("holdout selection contains duplicate pair ids")
    works = frozenset(str(value) for value in raw_works)
    row_mapping = {
        int(row["pair_id"]): str(row["work_fingerprint"])
        for row in pair_rows
        if isinstance(row, dict)
    }
    if (
        len(works) != len(raw_works)
        or set(row_mapping) != set(ids)
        or set(row_mapping.values()) != set(works)
        or int(payload.get("selected_work_count", -1)) != len(works)
    ):
        raise ValueError("holdout work-level isolation data is inconsistent")
    return ids, works, work_catalog_sha256


def load_reserved_pair_ids(path: Path) -> tuple[int, ...]:
    ids, _works, _catalog_hash = load_reserved_selection(path)
    return ids


def training_pair_ids(
    available: set[int],
    *,
    reserved: set[int],
    reserved_works: frozenset[str],
    work_by_pair: dict[int, str],
    limit: int | None,
    seed: int,
) -> list[int]:
    if not reserved <= available:
        raise ValueError("reserved holdout contains ids outside the pinned corpus")
    if set(work_by_pair) != available or not reserved_works:
        raise ValueError("work catalog does not cover the pinned corpus")
    candidates = {
        pair_id
        for pair_id in available - reserved
        if work_by_pair[pair_id] not in reserved_works
    }
    if not candidates:
        raise ValueError("no pairs remain after holdout exclusion")
    selected = stable_pair_ids(candidates, limit=limit, seed=seed)
    if set(selected) & reserved:
        raise RuntimeError("training/holdout pair leakage detected")
    if {work_by_pair[pair_id] for pair_id in selected} & reserved_works:
        raise RuntimeError("training/holdout work leakage detected")
    return selected


def acquire_training(
    *,
    output_dir: Path,
    holdout_selection: Path,
    work_catalog: Path,
    limit: int | None,
    seed: int,
    workers: int,
    timeout: float,
    retries: int,
    maximum_bytes: int,
) -> dict[str, Any]:
    if workers <= 0 or retries <= 0 or maximum_bytes <= 0:
        raise ValueError("workers, retries, and maximum bytes must be positive")
    reserved_ids, reserved_works, holdout_catalog_hash = (
        load_reserved_selection(holdout_selection)
    )
    reserved = set(reserved_ids)
    from app.tools.build_muse_omr_work_catalog import load_work_catalog

    work_by_pair = load_work_catalog(work_catalog)
    work_catalog_hash = sha256_file(work_catalog)
    if holdout_catalog_hash != work_catalog_hash:
        raise ValueError("holdout selection uses a different work catalog")
    manifest = fetch_manifest(timeout=timeout)
    pairs = parse_pair_manifest(manifest)
    selected_ids = training_pair_ids(
        set(pairs),
        reserved=reserved,
        reserved_works=reserved_works,
        work_by_pair=work_by_pair,
        limit=limit,
        seed=seed,
    )
    selected_works = sorted(
        {work_by_pair[pair_id] for pair_id in selected_ids}
    )
    remote_index = fetch_remote_file_index(timeout=timeout)
    files = selected_remote_files(pairs, selected_ids, remote_index)
    expected_bytes = sum(item.size for item in files)
    if expected_bytes > maximum_bytes:
        raise ValueError(
            f"selected training bytes exceed limit: {expected_bytes} > {maximum_bytes}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / MANIFEST_PATH).write_bytes(manifest)
    selection = {
        "format": 1,
        "repository": REPOSITORY,
        "revision": REVISION,
        "license": LICENSE,
        "role": "external_scan_degraded_training_only",
        "source_image_origin": SCAN_DEGRADED_IMAGE_ORIGIN,
        "production_evidence_eligible": False,
        "selection_seed": seed,
        "selected_pair_count": len(selected_ids),
        "selected_pair_ids": selected_ids,
        "selected_work_count": len(selected_works),
        "selected_work_fingerprints": selected_works,
        "pair_work_fingerprints": [
            {
                "pair_id": pair_id,
                "work_fingerprint": work_by_pair[pair_id],
            }
            for pair_id in selected_ids
        ],
        "work_catalog_sha256": work_catalog_hash,
        "reserved_holdout_pair_count": len(reserved),
        "reserved_holdout_pair_ids": sorted(reserved),
        "reserved_holdout_work_count": len(reserved_works),
        "reserved_holdout_work_fingerprints": sorted(reserved_works),
        "training_holdout_overlap": [],
        "training_holdout_work_overlap": [],
        "expected_download_bytes": expected_bytes,
    }
    atomic_json(output_dir / "selection.json", selection)

    downloaded: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                reuse_or_download_file,
                remote,
                output_dir,
                reuse_dirs=(work_catalog.parent,),
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

    report = {
        **selection,
        "downloaded_bytes": sum(int(row["size"]) for row in downloaded),
        "files": sorted(downloaded, key=lambda row: str(row["path"])),
    }
    atomic_json(output_dir / "provenance.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--holdout-selection",
        type=Path,
        default=DEFAULT_HOLDOUT_SELECTION,
    )
    parser.add_argument(
        "--work-catalog",
        type=Path,
        default=DEFAULT_WORK_CATALOG,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_TRAINING_PAIR_COUNT,
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--max-download-gb", type=float, default=12.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    if args.max_download_gb <= 0:
        raise SystemExit("--max-download-gb must be positive")
    report = acquire_training(
        output_dir=args.output_dir.resolve(),
        holdout_selection=args.holdout_selection.resolve(),
        work_catalog=args.work_catalog.resolve(),
        limit=None if args.limit == 0 else args.limit,
        seed=args.seed,
        workers=args.workers,
        timeout=args.timeout_seconds,
        retries=args.retries,
        maximum_bytes=int(args.max_download_gb * 1024**3),
    )
    print(
        json.dumps(
            {
                "selected_pair_count": report["selected_pair_count"],
                "selected_work_count": report["selected_work_count"],
                "reserved_holdout_work_count": report[
                    "reserved_holdout_work_count"
                ],
                "files": len(report["files"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
