from __future__ import annotations

"""Build a coverage-stratified Muse OMR split without consulting model output.

The original random holdout is kept in full. Only boundary-accepted works from
the old training split may be promoted into a development evaluation split.
This closes rare-configuration development coverage gaps without moving a
holdout work into training or selecting examples based on ScoreScan accuracy.
It does not create physical-scan release evidence.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.tools.evaluate_release_dataset import (  # noqa: E402
    PRODUCTION_RELEASE_GATES_V2,
    PRODUCTION_SCORE_CONFIGURATIONS,
    SCORE_CONFIGURATION_BY_SHAPE,
)
from app.tools.muse_omr_contract import (  # noqa: E402
    BENCHMARK_SELECTION_ROLE,
    SCAN_DEGRADED_IMAGE_ORIGIN,
    TRAINING_SELECTION_ROLE,
)
from app.tools.prepare_muse_omr_benchmark import (  # noqa: E402
    TRAINING_BOUNDARY_CLASSIFICATION_ROLE,
    production_page_coverage,
    unique_work_cases,
)
from scorescan.product_scope import (  # noqa: E402
    PRODUCTION_BOUNDARY_CONTRACT_VERSION,
)
from scorescan.util import (  # noqa: E402
    atomic_write_json,
    sha256_file,
    utc_now_iso,
)


PARTITION_CONTRACT_VERSION = "muse-omr-boundary-promotion@1"
BOUNDARY_FILTER_CONTRACT_VERSION = "muse-omr-boundary-filter@1"


def _resolved_accepted_cases(
    manifest_path: Path,
    *,
    allowed_role: str,
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or int(payload.get("format", 0)) != 1
        or payload.get("role") != allowed_role
        or payload.get("boundary_contract_version")
        != PRODUCTION_BOUNDARY_CONTRACT_VERSION
        or payload.get("source_image_origin")
        != SCAN_DEGRADED_IMAGE_ORIGIN
        or payload.get("production_evidence_eligible") is not False
        or not isinstance(raw_cases, list)
    ):
        raise ValueError(f"invalid source boundary manifest: {manifest_path}")
    accepted: list[dict[str, object]] = []
    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise ValueError(f"invalid source boundary case: {manifest_path}")
        boundary = raw.get("boundary")
        if (
            not isinstance(boundary, dict)
            or boundary.get("contract_version")
            != PRODUCTION_BOUNDARY_CONTRACT_VERSION
        ):
            raise ValueError("source boundary case uses a stale contract")
        if boundary.get("accepted") is not True:
            continue
        case = dict(raw)
        reference = Path(str(case.get("reference", "")))
        if not reference.is_absolute():
            reference = manifest_path.parent / reference
        input_pdf = Path(str(case.get("input_pdf", "")))
        if not input_pdf.is_absolute():
            input_pdf = manifest_path.parent / input_pdf
        reference = reference.resolve(strict=True)
        input_pdf = input_pdf.resolve(strict=True)
        if sha256_file(input_pdf) != case.get("input_pdf_sha256"):
            raise ValueError(f"source PDF hash mismatch: {case.get('id')}")
        case["reference"] = str(reference)
        case["input_pdf"] = str(input_pdf)
        accepted.append(case)
    return payload, unique_work_cases(accepted)


def _configuration(case: dict[str, object]) -> str:
    boundary = case.get("boundary")
    if not isinstance(boundary, dict):
        raise ValueError("case has no boundary")
    try:
        return SCORE_CONFIGURATION_BY_SHAPE[str(boundary["score_shape"])]
    except KeyError as exc:
        raise ValueError("case has an unsupported score configuration") from exc


def _rank(case: dict[str, object], seed: int) -> str:
    material = (
        f"{seed}\0{case.get('work_fingerprint')}\0"
        f"{case.get('input_pdf_sha256')}"
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def select_promotions(
    base_cases: list[dict[str, object]],
    training_cases: list[dict[str, object]],
    *,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    base_cases = unique_work_cases(base_cases)
    training_cases = unique_work_cases(training_cases)
    base_works = {str(case["work_fingerprint"]) for case in base_cases}
    training_works = {
        str(case["work_fingerprint"]) for case in training_cases
    }
    overlap = sorted(base_works & training_works)
    if overlap:
        raise ValueError(
            "base evaluation and training classification overlap by work"
        )
    _total, base_pages = production_page_coverage(base_cases)
    minimum = PRODUCTION_RELEASE_GATES_V2["minimum"]
    promotions: list[dict[str, object]] = []
    selected_works: set[str] = set()
    promotion_pages = {
        name: 0 for name in PRODUCTION_SCORE_CONFIGURATIONS
    }
    for configuration in PRODUCTION_SCORE_CONFIGURATIONS:
        target = int(minimum[f"{configuration}_page_count"])
        needed = max(0, target - base_pages[configuration])
        candidates = sorted(
            (
                case
                for case in training_cases
                if _configuration(case) == configuration
            ),
            key=lambda case: _rank(case, seed),
        )
        for case in candidates:
            if promotion_pages[configuration] >= needed:
                break
            fingerprint = str(case["work_fingerprint"])
            if fingerprint in selected_works:
                continue
            promotions.append(case)
            selected_works.add(fingerprint)
            promotion_pages[configuration] += int(
                case["input_pdf_pages"]
            )
        if promotion_pages[configuration] < needed:
            raise ValueError(
                "training partition cannot close boundary coverage: "
                f"{configuration} needs {needed} pages but only "
                f"{promotion_pages[configuration]} are available"
            )

    combined = [*base_cases, *promotions]
    total_pages, pages_by_configuration = production_page_coverage(combined)
    gaps = {
        name: max(
            0,
            int(minimum[f"{name}_page_count"])
            - pages_by_configuration[name],
        )
        for name in PRODUCTION_SCORE_CONFIGURATIONS
    }
    gaps["submitted_scan_page_count"] = max(
        0,
        int(minimum["submitted_scan_page_count"]) - total_pages,
    )
    gaps["source_group_count"] = max(
        0,
        int(minimum["source_group_count"]) - len(combined),
    )
    if any(gaps.values()):
        raise ValueError(f"promoted boundary still has coverage gaps: {gaps}")
    return promotions, {
        "selection_method": (
            "configuration_deficit_then_seeded_content_hash"
        ),
        "selection_seed": seed,
        "model_outputs_used_for_selection": False,
        "base_input_page_count": sum(base_pages.values()),
        "base_pages_by_score_configuration": base_pages,
        "promoted_work_count": len(promotions),
        "promoted_pages_by_score_configuration": promotion_pages,
        "result_input_page_count": total_pages,
        "result_pages_by_score_configuration": pages_by_configuration,
        "coverage_gaps": gaps,
    }


def _selection_pair_work_map(
    selection: dict[str, Any],
) -> dict[int, str]:
    rows = selection.get("pair_work_fingerprints")
    pair_ids = selection.get("selected_pair_ids")
    works = selection.get("selected_work_fingerprints")
    if (
        not isinstance(rows, list)
        or not isinstance(pair_ids, list)
        or not isinstance(works, list)
    ):
        raise ValueError("selection has no work provenance")
    mapping = {
        int(row["pair_id"]): str(row["work_fingerprint"])
        for row in rows
        if isinstance(row, dict)
    }
    if (
        set(mapping) != {int(value) for value in pair_ids}
        or set(mapping.values()) != {str(value) for value in works}
    ):
        raise ValueError("selection work provenance is inconsistent")
    return mapping


def _link_verified(
    source: Path,
    destination: Path,
) -> tuple[str, int, str]:
    expected_hash = sha256_file(source)
    size = source.stat().st_size
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if (
            destination.stat().st_size != size
            or sha256_file(destination) != expected_hash
        ):
            raise FileExistsError(
                f"existing partition artifact differs: {destination}"
            )
        return expected_hash, size, "already_present"
    try:
        os.link(source, destination)
        status = "hardlinked"
    except OSError:
        shutil.copy2(source, destination)
        status = "copied"
    if (
        destination.stat().st_size != size
        or sha256_file(destination) != expected_hash
    ):
        raise RuntimeError(f"partition artifact verification failed: {source}")
    return expected_hash, size, status


def materialize_training_partition(
    *,
    source_root: Path,
    output_root: Path,
    promoted_works: set[str],
    eligible_training_works: set[str],
    eligible_training_pair_ids: set[int],
    frozen_selections: list[Path],
    partition_plan: Path,
    partition_contract_version: str = PARTITION_CONTRACT_VERSION,
) -> dict[str, object]:
    source_selection_path = source_root / "selection.json"
    source_dataset_path = source_root / "benchmark_dataset.json"
    source_selection = json.loads(
        source_selection_path.read_text(encoding="utf-8")
    )
    source_dataset = json.loads(
        source_dataset_path.read_text(encoding="utf-8")
    )
    if (
        source_selection.get("role") != TRAINING_SELECTION_ROLE
        or not isinstance(source_dataset, dict)
    ):
        raise ValueError("source training partition is invalid")
    pair_work = _selection_pair_work_map(source_selection)
    remaining_pair_ids = [
        pair_id
        for pair_id in [int(value) for value in source_selection["selected_pair_ids"]]
        if (
            pair_id in eligible_training_pair_ids
            and
            pair_work[pair_id] in eligible_training_works
            and pair_work[pair_id] not in promoted_works
        )
    ]
    remaining_rows = [
        {
            "pair_id": pair_id,
            "work_fingerprint": pair_work[pair_id],
        }
        for pair_id in remaining_pair_ids
    ]
    remaining_works = sorted({row["work_fingerprint"] for row in remaining_rows})
    if (
        not remaining_pair_ids
        or promoted_works & set(remaining_works)
        or not set(remaining_works) <= eligible_training_works
    ):
        raise ValueError("promoted works were not removed from training")
    excluded_pair_ids = {
        pair_id
        for pair_id, fingerprint in pair_work.items()
        if (
            pair_id not in eligible_training_pair_ids
            or fingerprint not in eligible_training_works
        )
    }
    excluded_works = {
        pair_work[pair_id] for pair_id in excluded_pair_ids
    } - set(remaining_works) - promoted_works

    reserved_pair_ids: set[int] = set()
    reserved_works: set[str] = set()
    for selection_path in frozen_selections:
        frozen = json.loads(selection_path.read_text(encoding="utf-8"))
        if frozen.get("role") != BENCHMARK_SELECTION_ROLE:
            raise ValueError(
                f"frozen selection is not evaluation-only: {selection_path}"
            )
        frozen_map = _selection_pair_work_map(frozen)
        pair_overlap = sorted(reserved_pair_ids & set(frozen_map))
        work_overlap = sorted(reserved_works & set(frozen_map.values()))
        if pair_overlap or work_overlap:
            raise ValueError(
                "frozen evaluation selections overlap: "
                f"pairs={pair_overlap}, works={work_overlap}"
            )
        reserved_pair_ids.update(frozen_map)
        reserved_works.update(frozen_map.values())
    source_frozen_pair_overlap = sorted(set(pair_work) & reserved_pair_ids)
    source_frozen_work_overlap = sorted(
        set(pair_work.values()) & reserved_works
    )
    if source_frozen_pair_overlap or source_frozen_work_overlap:
        raise ValueError(
            "source training and frozen evaluation already overlap: "
            f"pairs={source_frozen_pair_overlap}, "
            f"works={source_frozen_work_overlap}"
        )
    promoted_pair_ids = {
        pair_id
        for pair_id, fingerprint in pair_work.items()
        if fingerprint in promoted_works
    }
    reserved_pair_ids.update(promoted_pair_ids)
    reserved_works.update(promoted_works)
    if set(remaining_pair_ids) & reserved_pair_ids:
        raise ValueError("new training/evaluation pair overlap")
    if set(remaining_works) & reserved_works:
        raise ValueError("new training/evaluation work overlap")

    output_root.mkdir(parents=True, exist_ok=True)
    output_dataset: dict[str, object] = {}
    files: list[dict[str, object]] = []
    expected_bytes = 0
    for pair_id in remaining_pair_ids:
        source_info = source_dataset.get(str(pair_id))
        if not isinstance(source_info, dict):
            raise ValueError(f"training pair is absent: {pair_id}")
        output_info: dict[str, str] = {}
        for field in ("score", "pdf_image"):
            relative = Path(str(source_info[field]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("training dataset path escapes its root")
            source = (source_root / relative).resolve(strict=True)
            destination = output_root / relative
            file_hash, size, status = _link_verified(source, destination)
            expected_bytes += size
            output_info[field] = relative.as_posix()
            files.append(
                {
                    "path": relative.as_posix(),
                    "local_path": str(destination.resolve()),
                    "size": size,
                    "sha256": file_hash,
                    "status": status,
                }
            )
        output_dataset[str(pair_id)] = output_info

    selection = {
        "format": 1,
        "partition_contract_version": partition_contract_version,
        "partition_plan_sha256": sha256_file(partition_plan),
        "source_selection_sha256": sha256_file(source_selection_path),
        "license": source_selection.get("license"),
        "repository": source_selection.get("repository"),
        "revision": source_selection.get("revision"),
        "role": TRAINING_SELECTION_ROLE,
        "work_catalog_sha256": source_selection.get("work_catalog_sha256"),
        "selection_seed": source_selection.get("selection_seed"),
        "selected_pair_count": len(remaining_pair_ids),
        "selected_pair_ids": remaining_pair_ids,
        "selected_work_count": len(remaining_works),
        "selected_work_fingerprints": remaining_works,
        "pair_work_fingerprints": remaining_rows,
        "reserved_holdout_pair_count": len(reserved_pair_ids),
        "reserved_holdout_pair_ids": sorted(reserved_pair_ids),
        "reserved_holdout_work_count": len(reserved_works),
        "reserved_holdout_work_fingerprints": sorted(reserved_works),
        "training_holdout_overlap": [],
        "training_holdout_work_overlap": [],
        "promoted_evaluation_pair_ids": sorted(promoted_pair_ids),
        "promoted_evaluation_work_fingerprints": sorted(promoted_works),
        "excluded_out_of_boundary_pair_ids": sorted(excluded_pair_ids),
        "excluded_fully_out_of_boundary_work_fingerprints": sorted(
            excluded_works
        ),
        "expected_download_bytes": expected_bytes,
    }
    atomic_write_json(output_root / "benchmark_dataset.json", output_dataset)
    atomic_write_json(output_root / "selection.json", selection)
    provenance = {
        **selection,
        "downloaded_bytes": expected_bytes,
        "files": files,
    }
    atomic_write_json(output_root / "provenance.json", provenance)
    return selection


def build_promoted_evaluation_manifest(
    *,
    base_manifest: Path,
    training_manifest: Path,
    promotions: list[dict[str, object]],
    promotion_report: dict[str, object],
    partition_plan: Path,
    training_selection: Path,
    output_path: Path,
) -> dict[str, object]:
    _base_payload, base_cases = _resolved_accepted_cases(
        base_manifest,
        allowed_role=BENCHMARK_SELECTION_ROLE,
    )
    promoted_evaluation_cases = [
        {
            **case,
            "role": "external_test_only",
            "promoted_from_training_partition": True,
        }
        for case in promotions
    ]
    cases = [*base_cases, *promoted_evaluation_cases]
    total_pages, pages = production_page_coverage(cases)
    works = sorted(str(case["work_fingerprint"]) for case in cases)
    training = json.loads(training_selection.read_text(encoding="utf-8"))
    training_works = {
        str(value)
        for value in training.get("selected_work_fingerprints", [])
    }
    overlap = sorted(training_works & set(works))
    if overlap:
        raise ValueError("promoted evaluation overlaps materialized training")
    report = {
        "format": 1,
        "boundary_contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
        "partition_contract_version": PARTITION_CONTRACT_VERSION,
        "created_at": utc_now_iso(),
        "name": (
            "Muse OMR coverage-stratified ScoreScan development benchmark"
        ),
        "role": BENCHMARK_SELECTION_ROLE,
        "source_image_origin": SCAN_DEGRADED_IMAGE_ORIGIN,
        "production_evidence_eligible": False,
        "production_evidence_blockers": [
            "source images are generated renders with simulated scan degradation",
            "production-v2 requires uniquely identified physical scan pages",
            "development references are not double-annotated frozen release truth",
        ],
        "source_manifests": [
            {
                "manifest": str(base_manifest.resolve()),
                "manifest_sha256": sha256_file(base_manifest),
            },
            {
                "manifest": str(training_manifest.resolve()),
                "manifest_sha256": sha256_file(training_manifest),
                "usage": "promoted_cases_only",
            },
        ],
        "partition_plan": str(partition_plan.resolve()),
        "partition_plan_sha256": sha256_file(partition_plan),
        "training_selection": str(training_selection.resolve()),
        "training_selection_sha256": sha256_file(training_selection),
        "training_evaluation_work_overlap": [],
        "selection_used_model_outputs": False,
        "case_count": len(cases),
        "work_count": len(works),
        "accepted_case_count": len(cases),
        "accepted_submitted_document_count": len(cases),
        "accepted_work_count": len(works),
        "accepted_work_fingerprints": works,
        "accepted_input_page_count": total_pages,
        "accepted_input_pages_by_score_configuration": pages,
        "development_coverage_against_production_shape_minimum": (
            promotion_report["coverage_gaps"]
        ),
        "development_shape_coverage_complete": all(
            int(value) == 0
            for value in promotion_report["coverage_gaps"].values()
        ),
        "production_scope_coverage_complete": False,
        "rejected_case_count": 0,
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument(
        "--training-classification-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--source-training-root", type=Path, required=True)
    parser.add_argument("--output-training-root", type=Path, required=True)
    parser.add_argument(
        "--frozen-selection",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output-evaluation-manifest", type=Path, required=True)
    parser.add_argument("--output-partition-plan", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    base_manifest = args.base_manifest.resolve()
    training_manifest = args.training_classification_manifest.resolve()
    _base_payload, base_cases = _resolved_accepted_cases(
        base_manifest,
        allowed_role=BENCHMARK_SELECTION_ROLE,
    )
    training_payload, training_cases = _resolved_accepted_cases(
        training_manifest,
        allowed_role=TRAINING_BOUNDARY_CLASSIFICATION_ROLE,
    )
    promotions, promotion_report = select_promotions(
        base_cases,
        training_cases,
        seed=args.seed,
    )
    plan = {
        "format": 1,
        "partition_contract_version": PARTITION_CONTRACT_VERSION,
        "created_at": utc_now_iso(),
        "base_manifest_sha256": sha256_file(base_manifest),
        "training_classification_manifest_sha256": sha256_file(
            training_manifest
        ),
        **promotion_report,
        "promoted_cases": [
            {
                "id": case["id"],
                "pair_id": case["pair_id"],
                "work_fingerprint": case["work_fingerprint"],
                "input_pdf_sha256": case["input_pdf_sha256"],
                "input_pdf_pages": case["input_pdf_pages"],
                "score_configuration": _configuration(case),
                "selection_rank": _rank(case, args.seed),
            }
            for case in promotions
        ],
    }
    partition_plan = args.output_partition_plan.resolve()
    partition_plan.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(partition_plan, plan)
    selection = materialize_training_partition(
        source_root=args.source_training_root.resolve(),
        output_root=args.output_training_root.resolve(),
        promoted_works={
            str(case["work_fingerprint"]) for case in promotions
        },
        eligible_training_works={
            str(case["work_fingerprint"]) for case in training_cases
        },
        eligible_training_pair_ids={
            int(case["pair_id"])
            for case in training_payload["cases"]
            if (
                isinstance(case, dict)
                and isinstance(case.get("boundary"), dict)
                and case["boundary"].get("accepted") is True
            )
        },
        frozen_selections=[
            path.resolve() for path in args.frozen_selection
        ],
        partition_plan=partition_plan,
    )
    evaluation = build_promoted_evaluation_manifest(
        base_manifest=base_manifest,
        training_manifest=training_manifest,
        promotions=promotions,
        promotion_report=promotion_report,
        partition_plan=partition_plan,
        training_selection=(
            args.output_training_root.resolve() / "selection.json"
        ),
        output_path=args.output_evaluation_manifest.resolve(),
    )
    print(
        json.dumps(
            {
                "promoted_work_count": len(promotions),
                "training_work_count": selection["selected_work_count"],
                "evaluation_work_count": evaluation["accepted_work_count"],
                "evaluation_input_page_count": evaluation[
                    "accepted_input_page_count"
                ],
                "evaluation_pages_by_score_configuration": evaluation[
                    "accepted_input_pages_by_score_configuration"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
