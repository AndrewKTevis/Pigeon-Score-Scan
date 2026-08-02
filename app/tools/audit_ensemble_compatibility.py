from __future__ import annotations

"""Audit the released ensemble calibrator against its frozen grouped corpus.

This tool does not train or tune the released ensemble model. It rebuilds the strictly
grouped frozen decision corpus with the currently bundled measure, visual, event and
context calibrators, including the independent system-localized family, then verifies
that the deployed bounded ensemble remains compatible with those components.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ensemble_training_data import EnsembleDataset, build_dataset  # noqa: E402
from scorescan.context_calibration import ContextCalibrator  # noqa: E402
from scorescan.ensemble_calibration import EnsembleCalibrator  # noqa: E402
from scorescan.event_calibration import EventCalibrator  # noqa: E402
from scorescan.measure_calibration import MeasureCalibrator  # noqa: E402
from scorescan.policy import DEFAULT_POLICY  # noqa: E402
from scorescan.util import atomic_write_json, read_json  # noqa: E402
from scorescan.visual_evidence import VisualMeasureCalibrator  # noqa: E402

AUDIT_VERSION = "scorescan-ensemble-compatibility-audit-3"


def _frozen_indices(group_ids: np.ndarray, seed: int) -> np.ndarray:
    groups = sorted(set(int(value) for value in group_ids.tolist()))
    random.Random(seed).shuffle(groups)
    frozen = set(groups[int(round(len(groups) * 0.90)) :])
    return np.flatnonzero(np.isin(group_ids, np.asarray(sorted(frozen), dtype=group_ids.dtype)))


def _sample_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    predictions = probabilities >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_accepts": int(np.sum(predictions & (labels == 0))),
        "false_rejects": int(np.sum((~predictions) & (labels == 1))),
    }


def _group_top1(
    probabilities: np.ndarray,
    dataset: EnsembleDataset,
    indices: np.ndarray,
    *,
    scenario: str | None = None,
) -> dict[str, float | int]:
    position = {int(index): offset for offset, index in enumerate(indices)}
    correct = total = 0
    for group_id, decision in enumerate(dataset.decision_groups):
        if scenario is not None and dataset.scenarios[group_id] != scenario:
            continue
        if not decision or not all(index in position for index in decision):
            continue
        selected = max(decision, key=lambda index: (probabilities[position[index]], -index))
        correct += int(dataset.labels[selected] == 1)
        total += 1
    return {"groups": total, "correct": correct, "top1_accuracy": correct / max(total, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent / "training" / "ensemble_component_compatibility_audit_v3.json",
    )
    parser.add_argument(
        "--reference-report",
        type=Path,
        default=ROOT.parent / "training" / "ensemble_calibrator_report_v3.json",
    )
    parser.add_argument("--seed", type=int, default=20270115)
    parser.add_argument("--groups", type=int, default=2400)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    dataset = build_dataset(args.seed, args.groups, max(1, args.workers))
    frozen = _frozen_indices(dataset.groups, args.seed)
    model = EnsembleCalibrator()
    probabilities = np.asarray(
        [
            (model.forest.predict(row) if model.forest.enabled else model.legacy.predict(row))
            for row in dataset.features[frozen]
        ],
        dtype=np.float64,
    )
    scenarios = sorted(set(dataset.scenarios))
    reference = read_json(args.reference_report, {})
    reference_frozen = reference.get("frozen_test", {}) if isinstance(reference, dict) else {}
    reference_decision = reference_frozen.get("decision", {}) if isinstance(reference_frozen, dict) else {}
    reference_scenarios = reference_frozen.get("by_scenario", {}) if isinstance(reference_frozen, dict) else {}

    decision = _group_top1(probabilities, dataset, frozen)
    by_scenario = {
        scenario: _group_top1(probabilities, dataset, frozen, scenario=scenario)
        for scenario in scenarios
    }
    current_top1 = float(decision["top1_accuracy"])
    reference_top1 = float(reference_decision.get("top1_accuracy", 0.0) or 0.0)
    clean_majority = float(by_scenario.get("clean-majority", {}).get("top1_accuracy", 0.0) or 0.0)
    clean_agreement = float(by_scenario.get("clean-agreement", {}).get("top1_accuracy", 0.0) or 0.0)
    cross_family = float(by_scenario.get("cross-family-fuzzy-trap", {}).get("top1_accuracy", 0.0) or 0.0)
    reference_cross = float(
        reference_scenarios.get("cross-family-fuzzy-trap", {}).get("top1_accuracy", 0.0) or 0.0
    )
    localized_rescue = float(by_scenario.get("localized-rescue", {}).get("top1_accuracy", 0.0) or 0.0)
    localized_isolation = float(by_scenario.get("localized-isolation-trap", {}).get("top1_accuracy", 0.0) or 0.0)
    localized_partial = float(by_scenario.get("localized-partial-trap", {}).get("top1_accuracy", 0.0) or 0.0)
    checks = {
        "model_version_preserved": model.model_version == "scorescan-ensemble-forest-3",
        "measure_v3_active": MeasureCalibrator().model_version == "scorescan-measure-forest-3",
        "visual_v4_active": VisualMeasureCalibrator().model_version
        == "scorescan-visual-measure-calibrator-4",
        "top1_floor": current_top1 >= 0.87,
        "top1_regression_bounded": current_top1 >= reference_top1 - 0.01,
        "clean_majority_preserved": clean_majority >= 0.99,
        "clean_agreement_preserved": clean_agreement >= 0.99,
        "cross_family_floor": cross_family >= 0.55,
        "cross_family_regression_bounded": cross_family >= reference_cross - 0.01,
        "localized_rescue_floor": localized_rescue >= 0.80,
        "localized_isolation_trap_floor": localized_isolation >= 0.80,
        "localized_partial_trap_floor": localized_partial >= 0.80,
    }
    report = {
        "audit_version": AUDIT_VERSION,
        "seed": args.seed,
        "groups": args.groups,
        "samples": int(len(dataset.labels)),
        "frozen_samples": int(len(frozen)),
        "split_unit": "complete source decision; ensemble v3 frozen partition is rebuilt without tuning",
        "current_models": {
            "ensemble": model.model_version,
            "measure": MeasureCalibrator().model_version,
            "visual": VisualMeasureCalibrator().model_version,
            "event": EventCalibrator().model_version,
            "context": ContextCalibrator().model_version,
        },
        "frozen_test": {
            "sample": _sample_metrics(dataset.labels[frozen], probabilities, 0.5),
            "policy_gate": _sample_metrics(
                dataset.labels[frozen],
                probabilities,
                DEFAULT_POLICY.replacement_ensemble_probability_floor,
            ),
            "decision": decision,
            "by_scenario": by_scenario,
        },
        "reference_v3_training_report": {
            "model_version": reference.get("model_version") if isinstance(reference, dict) else None,
            "decision": reference_decision,
            "cross_family_fuzzy_trap": reference_scenarios.get("cross-family-fuzzy-trap", {}),
        },
        "checks": checks,
        "passed": all(checks.values()),
        "scope": (
            "compatibility audit for ensemble v3 across whole-page and system-localized "
            "candidate families; not model training or end-to-end OMR"
        ),
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
