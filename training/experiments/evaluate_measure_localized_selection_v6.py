from __future__ import annotations

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

from scorescan.config import WORKFLOW_VERSION
from scorescan.policy import DEFAULT_POLICY
from scorescan.util import atomic_write_json
from tree_export import deployed_forest_probabilities
from train_selection_risk_measure_rescue_v6 import (
    _baseline_probabilities,
    _build_dataset,
    _partition_indices,
)

SCENARIOS = (
    "measure-localized-clear-gain",
    "measure-localized-common-error",
    "measure-localized-partial-trap",
)


def _accepted(payload: dict[str, object], values: np.ndarray, metadata: list[dict[str, object]], *, baseline: bool) -> np.ndarray:
    probabilities = (
        _baseline_probabilities(payload, values)
        if baseline
        else deployed_forest_probabilities(payload, values)
    )
    thresholds = payload.get("auto_replace_thresholds", {})
    if not isinstance(thresholds, dict):
        thresholds = {}
    default = float(payload.get("auto_replace_threshold", 1.0))
    exact_threshold = float(thresholds.get("exact_majority", default))
    semantic_threshold = float(thresholds.get("semantic_consensus", default))
    kinds = np.asarray([str(item["selection_kind"]) for item in metadata], dtype=object)
    exact_guard = np.asarray([bool(item["corroborated_exact_majority"]) for item in metadata], dtype=bool)
    semantic_guard = np.asarray([bool(item["corroborated_semantic_consensus"]) for item in metadata], dtype=bool)
    return np.where(
        kinds == "exact_majority",
        exact_guard & (probabilities >= exact_threshold),
        semantic_guard & (probabilities >= semantic_threshold),
    )


def _metrics(labels: np.ndarray, accepted: np.ndarray, metadata: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {
        "samples": int(len(labels)),
        "accepted": int(np.sum(accepted)),
        "true_accepts": int(np.sum(accepted & (labels == 1))),
        "false_accepts": int(np.sum(accepted & (labels == 0))),
        "coverage": float(np.mean(accepted)),
        "by_scenario": {},
    }
    for scenario in SCENARIOS:
        mask = np.asarray([str(item["scenario"]) == scenario for item in metadata], dtype=bool)
        scenario_accepted = accepted[mask]
        scenario_labels = labels[mask]
        result["by_scenario"][scenario] = {
            "samples": int(np.sum(mask)),
            "accepted": int(np.sum(scenario_accepted)),
            "true_accepts": int(np.sum(scenario_accepted & (scenario_labels == 1))),
            "false_accepts": int(np.sum(scenario_accepted & (scenario_labels == 0))),
        }
    return result


def _dataset(seed: int, groups: int, frozen: bool) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    values, labels, group_ids, metadata = _build_dataset(seed=seed, groups=groups, decisions_per_group=3)
    if not frozen:
        return values, labels, metadata
    *_parts, test = _partition_indices(group_ids, seed)
    return values[test], labels[test], [metadata[int(index)] for index in test]


def main() -> None:
    production = json.loads((ROOT / "training" / "baselines" / "selection_risk_v4.json").read_text(encoding="utf-8"))
    candidate = json.loads((ROOT / "training" / "experiments" / "candidate_selection_risk_v6_model.json").read_text(encoding="utf-8"))
    datasets = {
        "frozen_test": _dataset(20270501, 6000, True),
        "safety_corpus": _dataset(20270503, 4000, False),
        "independent_confirmation": _dataset(20270611, 3000, False),
    }
    sections: dict[str, object] = {}
    for name, (values, labels, metadata) in datasets.items():
        v4_accepted = _accepted(production, values, metadata, baseline=True)
        v6_accepted = _accepted(candidate, values, metadata, baseline=False)
        sections[name] = {
            "production_v4": _metrics(labels, v4_accepted, metadata),
            "candidate_v6": _metrics(labels, v6_accepted, metadata),
        }
    assertions = {
        "production_v4_zero_false_accepts": all(
            section["production_v4"]["false_accepts"] == 0 for section in sections.values()
        ),
        "production_v4_accepts_valid_measure_rescues": all(
            section["production_v4"]["by_scenario"]["measure-localized-clear-gain"]["true_accepts"] > 0
            for section in sections.values()
        ),
        "production_v4_rejects_measure_rescue_traps": all(
            section["production_v4"]["by_scenario"][scenario]["accepted"] == 0
            for section in sections.values()
            for scenario in ("measure-localized-common-error", "measure-localized-partial-trap")
        ),
        "candidate_v6_has_no_independent_advantage": (
            sections["independent_confirmation"]["candidate_v6"]["false_accepts"] > 0
            or sections["independent_confirmation"]["candidate_v6"]["true_accepts"]
            <= sections["independent_confirmation"]["production_v4"]["true_accepts"]
        ),
    }
    report = {
        "format": 1,
        "production_model": production.get("model_version"),
        "candidate_model": candidate.get("model_version"),
        "policy_version": DEFAULT_POLICY.version,
        "workflow_version": WORKFLOW_VERSION,
        "datasets": {
            "generator": "programmatic grouped replacement-risk evidence with measure-localised rescue scenarios",
            "split_unit": "programmatic score-family identity",
            "frozen_seed": 20270501,
            "safety_seed": 20270503,
            "confirmation_seed": 20270611,
        },
        "sections": sections,
        "release_assertions": assertions,
        "decision": "retain production v4; reject v6",
        "scope": "CPU replacement-risk and measure-localised rescue policy audit; not end-to-end OMR accuracy",
    }
    atomic_write_json(ROOT / "training" / "measure_localized_rescue_policy_audit_v1.json", report)
    if not all(assertions.values()):
        raise RuntimeError(json.dumps(assertions, sort_keys=True))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
