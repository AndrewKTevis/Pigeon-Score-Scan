from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP / "src"))
sys.path.insert(0, str(APP / "tools"))
sys.path.insert(0, str(HERE))

from scorescan.policy import DEFAULT_POLICY  # noqa: E402
from scorescan.selection_risk import FEATURE_NAMES  # noqa: E402
from scorescan.util import atomic_write_json  # noqa: E402
from train_selection_risk_v5 import _build_dataset, _partition_indices  # noqa: E402
from tree_export import deployed_forest_probabilities  # noqa: E402


def _metrics(labels: np.ndarray, accepted: np.ndarray, gains: np.ndarray) -> dict[str, object]:
    count = int(np.sum(accepted))
    false_accepts = int(np.sum(accepted & (labels == 0)))
    true_accepts = int(np.sum(accepted & (labels == 1)))
    return {
        "accepted": count,
        "true_accepts": true_accepts,
        "false_accepts": false_accepts,
        "coverage": float(np.mean(accepted)),
        "precision": float(np.mean(labels[accepted])) if count else 1.0,
        "total_semantic_gain": float(np.sum(gains[accepted])),
        "mean_semantic_gain": float(np.mean(gains[accepted])) if count else 0.0,
    }


def _old_exact_guard(values: np.ndarray) -> np.ndarray:
    index = {name: position for position, name in enumerate(FEATURE_NAMES)}
    family_count = np.maximum(
        1,
        np.rint(values[:, index["eligible_family_count_scaled"]] * 5.0).astype(np.int64),
    )
    exact_family_count = np.rint(
        values[:, index["exact_family_support_ratio"]] * family_count
    ).astype(np.int64)
    family_ratio = exact_family_count / family_count
    independently_corroborated = (
        (exact_family_count >= 3)
        | (
            (exact_family_count >= 2)
            & (values[:, index["selected_visual_probability"]] >= 0.65)
            & (values[:, index["selected_vs_template_visual_probability"]] >= 0.08)
        )
    )
    return (
        (values[:, index["selection_exact_majority"]] > 0.5)
        & (values[:, index["strict_majority"]] > 0.5)
        & (values[:, index["page_valid"]] > 0.5)
        & independently_corroborated
        & (family_ratio >= (2.0 / 3.0))
        & (values[:, index["exact_support_ratio"]] >= (2.0 / 3.0))
        & (values[:, index["missing_ratio"]] <= 0.05)
        & (values[:, index["selected_ensemble_probability"]] >= 0.88)
        & (values[:, index["selected_event_probability"]] >= 0.72)
        & (values[:, index["selected_measure_probability"]] >= 0.50)
        & (values[:, index["selected_vs_template_ensemble_probability"]] >= 0.15)
        & (values[:, index["selected_vs_template_event_probability"]] >= 0.12)
        & (values[:, index["selected_vs_template_measure_probability"]] >= 0.08)
    )


def _evaluate(
    payload: dict[str, object],
    values: np.ndarray,
    labels: np.ndarray,
    metadata: list[dict[str, object]],
) -> dict[str, object]:
    probabilities = deployed_forest_probabilities(payload, values)
    kinds = np.asarray([str(item["selection_kind"]) for item in metadata], dtype=object)
    exact_guard = np.asarray(
        [bool(item["corroborated_exact_majority"]) for item in metadata], dtype=bool
    )
    semantic_guard = np.asarray(
        [bool(item["corroborated_semantic_consensus"]) for item in metadata], dtype=bool
    )
    gains = np.asarray([float(item["semantic_gain"]) for item in metadata], dtype=np.float64)
    thresholds = payload.get("auto_replace_thresholds", {})
    if not isinstance(thresholds, dict):
        thresholds = {}
    stored = float(payload.get("auto_replace_threshold", 1.0))
    stored_exact = float(thresholds.get("exact_majority", stored))
    stored_semantic = float(thresholds.get("semantic_consensus", stored))
    verified_exact = float(DEFAULT_POLICY.replacement_selection_risk_floor)

    new_policy = np.where(
        kinds == "exact_majority",
        exact_guard & (probabilities >= verified_exact),
        semantic_guard & (probabilities >= stored_semantic),
    )
    stored_exact_policy = np.where(
        kinds == "exact_majority",
        exact_guard & (probabilities >= stored_exact),
        semantic_guard & (probabilities >= stored_semantic),
    )
    old_policy = np.where(
        kinds == "exact_majority",
        _old_exact_guard(values) & (probabilities >= verified_exact),
        probabilities >= stored_semantic,
    )

    result: dict[str, object] = {
        "samples": int(len(labels)),
        "thresholds": {
            "verified_exact_majority": verified_exact,
            "stored_exact_majority": stored_exact,
            "stored_semantic_consensus": stored_semantic,
        },
        "policy_0_23": _metrics(labels, new_policy, gains),
        "stored_exact_threshold_control": _metrics(labels, stored_exact_policy, gains),
        "policy_0_22_control": _metrics(labels, old_policy, gains),
        "by_selection_kind": {},
        "by_scenario": {},
    }
    for kind in ("exact_majority", "semantic_consensus"):
        mask = kinds == kind
        result["by_selection_kind"][kind] = {
            "policy_0_23": _metrics(labels[mask], new_policy[mask], gains[mask]),
            "stored_exact_threshold_control": _metrics(
                labels[mask], stored_exact_policy[mask], gains[mask]
            ),
            "policy_0_22_control": _metrics(labels[mask], old_policy[mask], gains[mask]),
        }
    scenarios = sorted({str(item["scenario"]) for item in metadata})
    for scenario in scenarios:
        mask = np.asarray(
            [str(item["scenario"]) == scenario for item in metadata], dtype=bool
        )
        result["by_scenario"][scenario] = {
            "policy_0_23": _metrics(labels[mask], new_policy[mask], gains[mask]),
            "policy_0_22_control": _metrics(labels[mask], old_policy[mask], gains[mask]),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=APP / "src" / "scorescan" / "resources" / "selection_risk.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "training" / "selection_risk_policy_audit_v5.json",
    )
    parser.add_argument("--seed", type=int, default=20270501)
    parser.add_argument("--groups", type=int, default=6000)
    parser.add_argument("--safety-seed", type=int, default=20270503)
    parser.add_argument("--safety-groups", type=int, default=4000)
    parser.add_argument("--confirmation-seed", type=int, default=20270703)
    parser.add_argument("--confirmation-groups", type=int, default=3000)
    args = parser.parse_args()

    payload = json.loads(args.model.read_text(encoding="utf-8"))
    values, labels, group_ids, metadata = _build_dataset(
        seed=args.seed, groups=args.groups, decisions_per_group=3
    )
    *_unused, test = _partition_indices(group_ids, args.seed)
    frozen = _evaluate(
        payload,
        values[test],
        labels[test],
        [metadata[int(index)] for index in test],
    )

    safety_values, safety_labels, _safety_ids, safety_metadata = _build_dataset(
        seed=args.safety_seed, groups=args.safety_groups, decisions_per_group=3
    )
    safety = _evaluate(payload, safety_values, safety_labels, safety_metadata)

    confirmation_values, confirmation_labels, _confirmation_ids, confirmation_metadata = (
        _build_dataset(
            seed=args.confirmation_seed,
            groups=args.confirmation_groups,
            decisions_per_group=3,
        )
    )
    confirmation = _evaluate(
        payload, confirmation_values, confirmation_labels, confirmation_metadata
    )

    report = {
        "model_version": payload.get("model_version"),
        "policy_version": DEFAULT_POLICY.version,
        "workflow_version": "single-staff-printed-scan@51",
        "dataset": {
            "generator": "selection-risk-v5 programmatic grouped evidence",
            "split_unit": "programmatic score-family identity",
            "training_seed": args.seed,
            "training_groups": args.groups,
            "safety_seed": args.safety_seed,
            "safety_groups": args.safety_groups,
            "confirmation_seed": args.confirmation_seed,
            "confirmation_groups": args.confirmation_groups,
        },
        "frozen_test": frozen,
        "safety_corpus": safety,
        "independent_confirmation": confirmation,
        "release_assertions": {
            "zero_false_accepts": all(
                section["policy_0_23"]["false_accepts"] == 0
                for section in (frozen, safety, confirmation)
            ),
            "invalid_family_trap_zero_accepts": all(
                section["by_scenario"]["exact-invalid-family-trap"]["policy_0_23"][
                    "accepted"
                ]
                == 0
                for section in (frozen, safety, confirmation)
            ),
            "cross_family_trap_zero_accepts": all(
                section["by_scenario"]["exact-cross-family-trap"]["policy_0_23"][
                    "accepted"
                ]
                == 0
                for section in (frozen, safety, confirmation)
            ),
            "verified_floor_adds_true_accepts_without_harm": all(
                section["policy_0_23"]["true_accepts"]
                >= section["stored_exact_threshold_control"]["true_accepts"]
                and section["policy_0_23"]["false_accepts"] == 0
                for section in (frozen, safety, confirmation)
            ),
        },
        "scope": "programmatic replacement-risk policy audit; not end-to-end OMR accuracy",
        "limitations": [
            "Programmatic grouped evidence does not replace a frozen real-scan benchmark.",
            "The verifier can only veto a selected replacement and cannot create notation.",
        ],
    }
    atomic_write_json(args.output, report)
    if not all(bool(value) for value in report["release_assertions"].values()):
        raise RuntimeError(json.dumps(report["release_assertions"], sort_keys=True))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
