from __future__ import annotations

"""Classify Muse OMR works unused by both training and frozen holdout splits.

This stage only exports semantic references.  It does not make the works part of
evaluation and never counts pages before their paired scans have been acquired.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.util import atomic_write_json, sha256_file, utc_now_iso  # noqa: E402
from scorescan.product_scope import (  # noqa: E402
    PRODUCTION_BOUNDARY_CONTRACT_VERSION,
)
from app.tools.acquire_muse_omr_benchmark import (  # noqa: E402
    REPOSITORY,
    REVISION,
)
from app.tools.build_muse_omr_work_catalog import (  # noqa: E402
    load_work_catalog,
)
from app.tools.prepare_muse_omr_benchmark import (  # noqa: E402
    _export_musicxml,
    analyze_reference_boundary,
)


WORK_FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{64}")
ALLOWED_SELECTION_ROLES = {
    "external_scan_degraded_development_benchmark_not_training",
    "external_scan_degraded_training_only",
}


def _selection_works(
    path: Path,
    *,
    expected_catalog_sha256: str,
    work_by_pair: dict[int, str],
) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("repository") != REPOSITORY
        or payload.get("revision") != REVISION
        or payload.get("role") not in ALLOWED_SELECTION_ROLES
        or payload.get("work_catalog_sha256") != expected_catalog_sha256
    ):
        raise ValueError(f"selection does not match the pinned catalog: {path}")
    raw_ids = payload.get("selected_pair_ids")
    raw_works = payload.get("selected_work_fingerprints")
    raw_rows = payload.get("pair_work_fingerprints")
    if (
        not isinstance(raw_ids, list)
        or not isinstance(raw_works, list)
        or not isinstance(raw_rows, list)
    ):
        raise ValueError(f"selection has no work-level provenance: {path}")
    pair_ids = [int(value) for value in raw_ids]
    works = [str(value) for value in raw_works]
    rows = {
        int(row["pair_id"]): str(row["work_fingerprint"])
        for row in raw_rows
        if isinstance(row, dict)
    }
    if (
        len(pair_ids) != len(set(pair_ids))
        or len(works) != len(set(works))
        or set(rows) != set(pair_ids)
        or set(rows.values()) != set(works)
        or any(work_by_pair.get(pair_id) != fingerprint for pair_id, fingerprint in rows.items())
        or int(payload.get("selected_work_count", -1)) != len(works)
    ):
        raise ValueError(f"selection work-level provenance is inconsistent: {path}")
    return set(works)


def unused_work_representatives(
    work_by_pair: dict[int, str],
    reserved_works: set[str],
) -> list[tuple[int, str]]:
    """Return the smallest pair id for every work outside both frozen splits."""

    catalog_works = set(work_by_pair.values())
    if any(
        WORK_FINGERPRINT_PATTERN.fullmatch(fingerprint) is None
        for fingerprint in catalog_works
    ):
        raise ValueError("work catalog contains an invalid fingerprint")
    missing = reserved_works - catalog_works
    if missing:
        raise ValueError("reserved selections contain works outside the catalog")
    representative_by_work: dict[str, int] = {}
    for pair_id, fingerprint in sorted(work_by_pair.items()):
        if fingerprint not in reserved_works:
            representative_by_work.setdefault(fingerprint, pair_id)
    return sorted(
        (
            (pair_id, fingerprint)
            for fingerprint, pair_id in representative_by_work.items()
        ),
        key=lambda row: row[0],
    )


def catalog_supplement(
    *,
    work_catalog: Path,
    training_selection: Path,
    holdout_selection: Path,
    mscz_root: Path,
    output_dir: Path,
    musescore: Path,
    timeout_seconds: int = 180,
    force: bool = False,
) -> dict[str, object]:
    if not musescore.is_file():
        raise FileNotFoundError(musescore)
    work_by_pair = load_work_catalog(work_catalog)
    catalog_sha256 = sha256_file(work_catalog)
    training_works = _selection_works(
        training_selection,
        expected_catalog_sha256=catalog_sha256,
        work_by_pair=work_by_pair,
    )
    holdout_works = _selection_works(
        holdout_selection,
        expected_catalog_sha256=catalog_sha256,
        work_by_pair=work_by_pair,
    )
    overlap = training_works & holdout_works
    if overlap:
        raise ValueError("training and holdout selections overlap by work")
    representatives = unused_work_representatives(
        work_by_pair,
        training_works | holdout_works,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = output_dir / "references"
    rows: list[dict[str, object]] = []
    for position, (pair_id, fingerprint) in enumerate(representatives, start=1):
        source = mscz_root / f"score_file_{pair_id}.mscz"
        if not source.is_file():
            raise FileNotFoundError(source)
        reference = reference_dir / f"score_file_{pair_id}.musicxml"
        if force or not reference.is_file():
            _export_musicxml(
                source,
                reference,
                musescore,
                timeout_seconds=timeout_seconds,
            )
        boundary = analyze_reference_boundary(reference)
        rows.append(
            {
                "pair_id": pair_id,
                "work_fingerprint": fingerprint,
                "source_mscz": str(source.resolve()),
                "source_mscz_sha256": sha256_file(source),
                "reference": str(reference.relative_to(output_dir)),
                "reference_sha256": sha256_file(reference),
                "boundary": boundary,
            }
        )
        state = "accepted" if boundary["accepted"] else "rejected"
        print(
            f"[{position}/{len(representatives)}] pair {pair_id}: "
            f"{state} {boundary['score_shape']}",
            flush=True,
        )

    accepted = [row for row in rows if row["boundary"]["accepted"]]
    shapes = Counter(str(row["boundary"]["score_shape"]) for row in accepted)
    rejected_reasons = Counter(
        str(reason)
        for row in rows
        for reason in row["boundary"]["reasons"]
    )
    report = {
        "format": 1,
        "boundary_contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
        "created_at": utc_now_iso(),
        "name": "Muse OMR unused-work boundary supplement catalog",
        "role": "candidate_catalog_not_training_or_evaluation",
        "repository": REPOSITORY,
        "revision": REVISION,
        "work_catalog_sha256": catalog_sha256,
        "training_selection_sha256": sha256_file(training_selection),
        "holdout_selection_sha256": sha256_file(holdout_selection),
        "training_work_count": len(training_works),
        "holdout_work_count": len(holdout_works),
        "reserved_work_overlap": [],
        "unused_independent_work_count": len(representatives),
        "accepted_independent_work_count": len(accepted),
        "accepted_work_count_by_score_shape": dict(sorted(shapes.items())),
        "rejected_reason_counts": dict(sorted(rejected_reasons.items())),
        "cases": rows,
    }
    atomic_write_json(output_dir / "supplement_catalog.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("work_catalog", type=Path)
    parser.add_argument("training_selection", type=Path)
    parser.add_argument("holdout_selection", type=Path)
    parser.add_argument("mscz_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--musescore", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = catalog_supplement(
        work_catalog=args.work_catalog.resolve(),
        training_selection=args.training_selection.resolve(),
        holdout_selection=args.holdout_selection.resolve(),
        mscz_root=args.mscz_root.resolve(),
        output_dir=args.output_dir.resolve(),
        musescore=args.musescore.resolve(),
        timeout_seconds=args.timeout_seconds,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "unused_independent_work_count": report[
                    "unused_independent_work_count"
                ],
                "accepted_independent_work_count": report[
                    "accepted_independent_work_count"
                ],
                "accepted_work_count_by_score_shape": report[
                    "accepted_work_count_by_score_shape"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
