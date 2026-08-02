from __future__ import annotations

"""Acquire exact scan-degraded pairs from the work-disjoint dev supplement."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from app.tools.acquire_muse_omr_benchmark import (
    LICENSE,
    MANIFEST_PATH,
    REPOSITORY,
    REVISION,
    _pdf_coverage,
    atomic_json,
    fetch_manifest,
    fetch_remote_file_index,
    parse_pair_manifest,
    reuse_or_download_file,
    selected_remote_files,
    sha256_file,
)
from app.tools.build_muse_omr_work_catalog import load_work_catalog
from app.tools.catalog_muse_omr_boundary_supplement import _selection_works
from app.tools.muse_omr_contract import SCAN_DEGRADED_IMAGE_ORIGIN
from scorescan.product_scope import PRODUCTION_BOUNDARY_CONTRACT_VERSION


DEFAULT_SCORE_SHAPES = (
    "single_staff_solo",
    "keyboard_plus_single_staff_ensemble",
)


def select_supplement_pairs(
    catalog: dict[str, object],
    *,
    score_shapes: set[str],
    work_by_pair: dict[int, str],
    forbidden_works: set[str],
) -> list[tuple[int, str, str]]:
    if (
        catalog.get("role") != "candidate_catalog_not_training_or_evaluation"
        or catalog.get("repository") != REPOSITORY
        or catalog.get("revision") != REVISION
        or catalog.get("boundary_contract_version")
        != PRODUCTION_BOUNDARY_CONTRACT_VERSION
        or catalog.get("reserved_work_overlap") != []
    ):
        raise ValueError("invalid supplement candidate catalog")
    raw_cases = catalog.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("supplement candidate catalog has no cases")
    selected: list[tuple[int, str, str]] = []
    selected_works: set[str] = set()
    for case in raw_cases:
        if not isinstance(case, dict):
            raise ValueError("invalid supplement candidate row")
        boundary = case.get("boundary")
        if not isinstance(boundary, dict):
            raise ValueError("supplement candidate has no boundary record")
        shape = str(boundary.get("score_shape", ""))
        if boundary.get("accepted") is not True or shape not in score_shapes:
            continue
        pair_id = int(case.get("pair_id", -1))
        fingerprint = str(case.get("work_fingerprint", ""))
        if (
            work_by_pair.get(pair_id) != fingerprint
            or fingerprint in forbidden_works
            or fingerprint in selected_works
        ):
            raise ValueError(
                "supplement candidate violates work-level isolation"
            )
        selected_works.add(fingerprint)
        selected.append((pair_id, fingerprint, shape))
    if not selected:
        raise ValueError("no accepted supplement pairs match the target shapes")
    return sorted(selected)


def acquire_supplement(
    *,
    supplement_catalog: Path,
    output_dir: Path,
    work_catalog: Path,
    training_selection: Path,
    holdout_selection: Path,
    score_shapes: Iterable[str],
    reuse_dirs: Iterable[Path] = (),
    workers: int = 6,
    timeout: float = 120.0,
    retries: int = 4,
    maximum_bytes: int = 20 * 1024**3,
) -> dict[str, Any]:
    if workers <= 0 or retries <= 0 or maximum_bytes <= 0:
        raise ValueError("workers, retries, and maximum bytes must be positive")
    work_by_pair = load_work_catalog(work_catalog)
    work_catalog_sha256 = sha256_file(work_catalog)
    training_works = _selection_works(
        training_selection,
        expected_catalog_sha256=work_catalog_sha256,
        work_by_pair=work_by_pair,
    )
    holdout_works = _selection_works(
        holdout_selection,
        expected_catalog_sha256=work_catalog_sha256,
        work_by_pair=work_by_pair,
    )
    if training_works & holdout_works:
        raise ValueError("training and holdout selections overlap by work")
    catalog = json.loads(supplement_catalog.read_text(encoding="utf-8"))
    if (
        catalog.get("work_catalog_sha256") != work_catalog_sha256
        or catalog.get("training_selection_sha256")
        != sha256_file(training_selection)
        or catalog.get("holdout_selection_sha256")
        != sha256_file(holdout_selection)
    ):
        raise ValueError("supplement catalog provenance is stale")
    target_shapes = {str(value) for value in score_shapes}
    if not target_shapes:
        raise ValueError("at least one target score shape is required")
    selected = select_supplement_pairs(
        catalog,
        score_shapes=target_shapes,
        work_by_pair=work_by_pair,
        forbidden_works=training_works | holdout_works,
    )
    selected_ids = [row[0] for row in selected]
    selected_works = sorted(row[1] for row in selected)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = fetch_manifest(timeout=timeout)
    pairs = parse_pair_manifest(manifest)
    missing = sorted(set(selected_ids) - set(pairs))
    if missing:
        raise ValueError(f"supplement pairs are unavailable: {missing}")
    remote_index = fetch_remote_file_index(timeout=timeout)
    files = selected_remote_files(pairs, selected_ids, remote_index)
    expected_bytes = sum(item.size for item in files)
    if expected_bytes > maximum_bytes:
        raise ValueError(
            f"selected supplement bytes exceed limit: "
            f"{expected_bytes} > {maximum_bytes}"
        )
    (output_dir / MANIFEST_PATH).write_bytes(manifest)
    selection = {
        "format": 1,
        "boundary_contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
        "repository": REPOSITORY,
        "revision": REVISION,
        "role": "external_scan_degraded_development_benchmark_not_training",
        "source_image_origin": SCAN_DEGRADED_IMAGE_ORIGIN,
        "production_evidence_eligible": False,
        "license": LICENSE,
        "selection_mode": "exact_work_disjoint_boundary_supplement",
        "target_score_shapes": sorted(target_shapes),
        "selected_pair_count": len(selected_ids),
        "selected_pair_ids": selected_ids,
        "selected_work_count": len(selected_works),
        "selected_work_fingerprints": selected_works,
        "pair_work_fingerprints": [
            {
                "pair_id": pair_id,
                "work_fingerprint": fingerprint,
                "score_shape": shape,
            }
            for pair_id, fingerprint, shape in selected
        ],
        "work_catalog_sha256": work_catalog_sha256,
        "training_selection_sha256": sha256_file(training_selection),
        "holdout_selection_sha256": sha256_file(holdout_selection),
        "supplement_catalog_sha256": sha256_file(supplement_catalog),
        "training_work_overlap": [],
        "holdout_work_overlap": [],
        "expected_download_bytes": expected_bytes,
    }
    atomic_json(output_dir / "selection.json", selection)

    verified_reuse_dirs = tuple(path.resolve() for path in reuse_dirs)
    downloaded: list[dict[str, Any]] = []
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("supplement_catalog", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--work-catalog", type=Path, required=True)
    parser.add_argument("--training-selection", type=Path, required=True)
    parser.add_argument("--holdout-selection", type=Path, required=True)
    parser.add_argument(
        "--score-shape",
        action="append",
        default=[],
        choices=(
            "single_staff_solo",
            "keyboard",
            "single_staff_ensemble",
            "keyboard_plus_single_staff_ensemble",
        ),
    )
    parser.add_argument("--reuse-dir", type=Path, action="append", default=[])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--max-download-gb", type=float, default=20.0)
    args = parser.parse_args()
    report = acquire_supplement(
        supplement_catalog=args.supplement_catalog.resolve(),
        output_dir=args.output_dir.resolve(),
        work_catalog=args.work_catalog.resolve(),
        training_selection=args.training_selection.resolve(),
        holdout_selection=args.holdout_selection.resolve(),
        score_shapes=args.score_shape or DEFAULT_SCORE_SHAPES,
        reuse_dirs=[path.resolve() for path in args.reuse_dir],
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
                "selected_independent_work_pdf_page_count": report[
                    "selected_independent_work_pdf_page_count"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
