from __future__ import annotations

"""Materialize a model-independent, boundary-only Muse training partition."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.tools.muse_omr_contract import (  # noqa: E402
    BENCHMARK_SELECTION_ROLE,
)
from app.tools.prepare_muse_omr_benchmark import (  # noqa: E402
    TRAINING_BOUNDARY_CLASSIFICATION_ROLE,
)
from app.tools.promote_muse_training_cases_to_boundary import (  # noqa: E402
    BOUNDARY_FILTER_CONTRACT_VERSION,
    _resolved_accepted_cases,
    materialize_training_partition,
)
from scorescan.util import atomic_write_json, sha256_file, utc_now_iso  # noqa: E402


def build_boundary_filtered_training(
    *,
    training_classification_manifest: Path,
    source_training_root: Path,
    output_training_root: Path,
    frozen_selections: list[Path],
    output_partition_plan: Path,
) -> dict[str, object]:
    payload, accepted_cases = _resolved_accepted_cases(
        training_classification_manifest,
        allowed_role=TRAINING_BOUNDARY_CLASSIFICATION_ROLE,
    )
    eligible_pair_ids = {
        int(case["pair_id"])
        for case in payload["cases"]
        if (
            isinstance(case, dict)
            and isinstance(case.get("boundary"), dict)
            and case["boundary"].get("accepted") is True
        )
    }
    eligible_works = {
        str(case["work_fingerprint"]) for case in accepted_cases
    }
    if len(eligible_works) < 200 or len(eligible_pair_ids) < 200:
        raise ValueError(
            "boundary-filtered Muse training has fewer than 200 independent "
            "works or pairs"
        )
    plan = {
        "format": 1,
        "partition_contract_version": BOUNDARY_FILTER_CONTRACT_VERSION,
        "created_at": utc_now_iso(),
        "role": "training_boundary_filter_only_not_release_evaluation",
        "training_classification_manifest": str(
            training_classification_manifest.resolve()
        ),
        "training_classification_manifest_sha256": sha256_file(
            training_classification_manifest
        ),
        "source_training_selection_sha256": sha256_file(
            source_training_root / "selection.json"
        ),
        "eligible_pair_count": len(eligible_pair_ids),
        "eligible_work_count": len(eligible_works),
        "rejected_case_count": int(payload["rejected_case_count"]),
        "model_outputs_used_for_selection": False,
        "release_evaluation_authorized": False,
        "frozen_selections": [
            {
                "selection": str(path.resolve()),
                "selection_sha256": sha256_file(path),
                "role": BENCHMARK_SELECTION_ROLE,
            }
            for path in frozen_selections
        ],
    }
    output_partition_plan.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_partition_plan, plan)
    selection = materialize_training_partition(
        source_root=source_training_root,
        output_root=output_training_root,
        promoted_works=set(),
        eligible_training_works=eligible_works,
        eligible_training_pair_ids=eligible_pair_ids,
        frozen_selections=frozen_selections,
        partition_plan=output_partition_plan,
        partition_contract_version=BOUNDARY_FILTER_CONTRACT_VERSION,
    )
    if (
        int(selection["selected_pair_count"]) != len(eligible_pair_ids)
        or int(selection["selected_work_count"]) != len(eligible_works)
        or selection["promoted_evaluation_pair_ids"]
        or selection["promoted_evaluation_work_fingerprints"]
    ):
        raise RuntimeError("boundary-filtered training materialization drifted")
    return selection


def main() -> None:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--output-partition-plan", type=Path, required=True)
    args = parser.parse_args()
    selection = build_boundary_filtered_training(
        training_classification_manifest=(
            args.training_classification_manifest.resolve()
        ),
        source_training_root=args.source_training_root.resolve(),
        output_training_root=args.output_training_root.resolve(),
        frozen_selections=[
            path.resolve() for path in args.frozen_selection
        ],
        output_partition_plan=args.output_partition_plan.resolve(),
    )
    print(
        json.dumps(
            {
                "selected_pair_count": selection["selected_pair_count"],
                "selected_work_count": selection["selected_work_count"],
                "excluded_out_of_boundary_pair_count": len(
                    selection["excluded_out_of_boundary_pair_ids"]
                ),
                "excluded_fully_out_of_boundary_work_count": len(
                    selection[
                        "excluded_fully_out_of_boundary_work_fingerprints"
                    ]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
