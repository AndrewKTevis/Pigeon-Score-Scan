from __future__ import annotations

"""Merge work-disjoint Muse OMR boundary sources with exact page accounting."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.util import atomic_write_json, sha256_file, utc_now_iso  # noqa: E402
from scorescan.product_scope import (  # noqa: E402
    PRODUCTION_BOUNDARY_CONTRACT_VERSION,
)
from app.tools.evaluate_release_dataset import (  # noqa: E402
    PRODUCTION_RELEASE_GATES_V2,
    PRODUCTION_SCORE_CONFIGURATIONS,
)
from app.tools.prepare_muse_omr_benchmark import (  # noqa: E402
    production_page_coverage,
    unique_work_cases,
)
from app.tools.muse_omr_contract import (  # noqa: E402
    SCAN_DEGRADED_IMAGE_ORIGIN,
)


def _resolved_cases(
    manifest_path: Path,
    payload: dict[str, object],
) -> tuple[list[dict[str, object]], set[str]]:
    raw_cases = payload.get("cases")
    if (
        int(payload.get("format", 0) or 0) != 1
        or payload.get("role")
        != "external_scan_degraded_development_benchmark_not_training"
        or payload.get("boundary_contract_version")
        != PRODUCTION_BOUNDARY_CONTRACT_VERSION
        or payload.get("source_image_origin")
        != SCAN_DEGRADED_IMAGE_ORIGIN
        or payload.get("production_evidence_eligible") is not False
        or not isinstance(raw_cases, list)
        or not raw_cases
    ):
        raise ValueError(f"invalid boundary manifest: {manifest_path}")
    cases: list[dict[str, object]] = []
    works: set[str] = set()
    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise ValueError(f"invalid boundary case: {manifest_path}")
        case = dict(raw)
        boundary = case.get("boundary")
        if (
            not isinstance(boundary, dict)
            or boundary.get("contract_version")
            != PRODUCTION_BOUNDARY_CONTRACT_VERSION
        ):
            raise ValueError(
                f"boundary case uses a stale contract: {case.get('id')}"
            )
        reference = Path(str(case.get("reference", "")))
        if not reference.is_absolute():
            reference = manifest_path.parent / reference
        reference = reference.resolve()
        input_pdf = Path(str(case.get("input_pdf", ""))).resolve()
        if not reference.is_file() or not input_pdf.is_file():
            raise FileNotFoundError(
                f"boundary case files are incomplete: {case.get('id')}"
            )
        case["reference"] = str(reference)
        case["input_pdf"] = str(input_pdf)
        fingerprint = str(case.get("work_fingerprint", ""))
        unique_work_cases([case])
        works.add(fingerprint)
        cases.append(case)
    return cases, works


def merge_boundary_manifests(
    manifest_paths: list[Path],
    output_path: Path,
) -> dict[str, object]:
    if len(manifest_paths) < 2:
        raise ValueError("at least two boundary manifests are required")
    all_cases: list[dict[str, object]] = []
    all_works: set[str] = set()
    seen_ids: set[str] = set()
    sources: list[dict[str, object]] = []
    for manifest_path in manifest_paths:
        manifest_path = manifest_path.resolve(strict=True)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"invalid boundary manifest: {manifest_path}")
        cases, works = _resolved_cases(manifest_path, payload)
        overlap = sorted(all_works & works)
        if overlap:
            raise ValueError(
                "boundary manifests overlap by independent work: "
                + ", ".join(overlap[:5])
            )
        ids = {str(case.get("id", "")) for case in cases}
        if "" in ids or seen_ids & ids:
            raise ValueError("boundary manifests contain duplicate/empty case ids")
        all_works.update(works)
        seen_ids.update(ids)
        all_cases.extend(cases)
        sources.append(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "case_count": len(cases),
                "work_count": len(works),
            }
        )

    accepted_cases = [
        case
        for case in all_cases
        if isinstance(case.get("boundary"), dict)
        and case["boundary"].get("accepted") is True
    ]
    accepted_work_cases = unique_work_cases(accepted_cases)
    accepted_work_fingerprints = sorted(
        str(case["work_fingerprint"]) for case in accepted_work_cases
    )
    total_pages, pages_by_configuration = production_page_coverage(
        accepted_work_cases
    )
    minimum = PRODUCTION_RELEASE_GATES_V2["minimum"]
    coverage_gaps = {
        name: max(
            0,
            int(minimum[f"{name}_page_count"])
            - pages_by_configuration[name],
        )
        for name in PRODUCTION_SCORE_CONFIGURATIONS
    }
    coverage_gaps["submitted_scan_page_count"] = max(
        0,
        int(minimum["submitted_scan_page_count"]) - total_pages,
    )
    coverage_gaps["source_group_count"] = max(
        0,
        int(minimum["source_group_count"])
        - len(accepted_work_fingerprints),
    )
    report = {
        "format": 1,
        "boundary_contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
        "created_at": utc_now_iso(),
        "name": "Muse OMR work-disjoint combined ScoreScan boundary benchmark",
        "role": "external_scan_degraded_development_benchmark_not_training",
        "source_image_origin": SCAN_DEGRADED_IMAGE_ORIGIN,
        "production_evidence_eligible": False,
        "production_evidence_blockers": [
            "source images are generated renders with simulated scan degradation",
            "production-v2 requires uniquely identified physical scan pages",
            "development references are not double-annotated frozen release truth",
        ],
        "source_manifests": sources,
        "case_count": len(all_cases),
        "work_count": len(all_works),
        "accepted_case_count": len(accepted_cases),
        "accepted_submitted_document_count": len(accepted_work_cases),
        "accepted_work_count": len(accepted_work_fingerprints),
        "accepted_work_fingerprints": accepted_work_fingerprints,
        "accepted_input_page_count": total_pages,
        "accepted_input_pages_by_score_configuration": (
            pages_by_configuration
        ),
        "development_coverage_against_production_shape_minimum": (
            coverage_gaps
        ),
        "development_shape_coverage_complete": all(
            gap == 0 for gap in coverage_gaps.values()
        ),
        "production_scope_coverage_complete": False,
        "rejected_case_count": len(all_cases) - len(accepted_cases),
        "cases": all_cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = merge_boundary_manifests(
        [path.resolve() for path in args.manifest],
        args.output.resolve(),
    )
    print(
        json.dumps(
            {
                "accepted_work_count": report["accepted_work_count"],
                "accepted_input_page_count": report[
                    "accepted_input_page_count"
                ],
                "accepted_input_pages_by_score_configuration": report[
                    "accepted_input_pages_by_score_configuration"
                ],
                "development_coverage_against_production_shape_minimum": report[
                    "development_coverage_against_production_shape_minimum"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
