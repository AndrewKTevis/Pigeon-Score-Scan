from __future__ import annotations

"""Train the bounded ensemble meta-calibrator on current production evidence.

The training set is built from complete three-to-eight-variant decisions.  Measure, visual,
event and context probabilities are produced by the bundled production models rather
than by hand-written proxy equations.  Source-measure identity remains the split unit,
and independent probability calibration, model selection and frozen testing are kept
strictly separate.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ensemble_training_data import EnsembleDataset, build_dataset  # noqa: E402
from scorescan.context_calibration import ContextCalibrator  # noqa: E402
from scorescan.ensemble_calibration import FEATURE_NAMES  # noqa: E402
from scorescan.event_calibration import EventCalibrator  # noqa: E402
from scorescan.linear_model import StandardizedLogisticModel  # noqa: E402
from scorescan.measure_calibration import MeasureCalibrator  # noqa: E402
from scorescan.policy import DEFAULT_POLICY  # noqa: E402
from scorescan.util import atomic_write_json  # noqa: E402
from scorescan.visual_evidence import VisualMeasureCalibrator  # noqa: E402
from tree_export import deployed_forest_probabilities, serialize_probability_forest  # noqa: E402

MODEL_VERSION = "scorescan-ensemble-forest-3"
MODEL_CONFIGS = (
    {"n_estimators": 24, "max_depth": 6, "min_samples_leaf": 8},
    {"n_estimators": 32, "max_depth": 7, "min_samples_leaf": 6},
    {"n_estimators": 48, "max_depth": 8, "min_samples_leaf": 6},
    {"n_estimators": 64, "max_depth": 10, "min_samples_leaf": 4},
)


def _split_groups(
    group_ids: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    groups = sorted(set(int(value) for value in group_ids.tolist()))
    random.Random(seed).shuffle(groups)
    count = len(groups)
    train_cut = int(round(count * 0.70))
    calibration_cut = int(round(count * 0.80))
    audit_cut = int(round(count * 0.90))
    partitions = (
        set(groups[:train_cut]),
        set(groups[train_cut:calibration_cut]),
        set(groups[calibration_cut:audit_cut]),
        set(groups[audit_cut:]),
    )
    return tuple(
        np.flatnonzero(np.isin(group_ids, np.asarray(sorted(partition), dtype=group_ids.dtype)))
        for partition in partitions
    )  # type: ignore[return-value]


def _fit_forest(
    features: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    calibration: np.ndarray,
    seed: int,
    config: dict[str, int],
) -> tuple[RandomForestClassifier, LogisticRegression, dict[str, object]]:
    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=1,
        random_state=seed,
    )
    model.fit(features[train], labels[train])
    raw_calibration = model.predict_proba(features[calibration])[:, 1]
    calibrator = LogisticRegression(C=1000.0, max_iter=2000, random_state=seed)
    calibrator.fit(raw_calibration.reshape(-1, 1), labels[calibration])
    payload: dict[str, object] = {
        "model_version": MODEL_VERSION,
        "model_type": "random_forest",
        "feature_names": list(FEATURE_NAMES),
        "trees": serialize_probability_forest(model),
        "calibration_intercept": float(calibrator.intercept_[0]),
        "calibration_slope": float(calibrator.coef_[0, 0]),
        "training_seed": seed,
        "model_config": dict(config),
        "training_distribution": (
            "complete three-to-eight-variant difficult decisions across five independent candidate families, scored by current production component models"
        ),
    }
    return model, calibrator, payload


def _calibrated_probabilities(
    model: RandomForestClassifier,
    calibrator: LogisticRegression,
    values: np.ndarray,
) -> np.ndarray:
    raw = model.predict_proba(values)[:, 1]
    return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


def _sample_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
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
        selected = max(
            decision,
            key=lambda index: (probabilities[position[index]], -index),
        )
        correct += int(dataset.labels[selected] == 1)
        total += 1
    return {
        "groups": total,
        "correct": correct,
        "top1_accuracy": correct / max(total, 1),
    }


def _fit_linear_ablation(
    dataset: EnsembleDataset,
    train: np.ndarray,
    test: np.ndarray,
    seed: int,
) -> np.ndarray:
    scaler = StandardScaler().fit(dataset.features[train])
    model = LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(scaler.transform(dataset.features[train]), dataset.labels[train])
    return model.predict_proba(scaler.transform(dataset.features[test]))[:, 1]


def _baseline_probabilities(
    model_path: Path,
    features: np.ndarray,
) -> tuple[np.ndarray, str]:
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    version = str(payload.get("model_version", "unknown"))
    baseline_features = np.asarray(features, dtype=np.float64).copy()
    if version == "scorescan-ensemble-forest-2":
        # v2 encoded candidate counts 1..7 with denominator six.  Reconstruct the
        # integer count before scoring it on the v3 1..8 dataset so the ablation is
        # a genuine old-model comparison rather than an accidental schema shift.
        column = FEATURE_NAMES.index("candidate_count_scaled")
        candidate_count = np.rint(baseline_features[:, column] * 7.0).astype(np.int64) + 1
        baseline_features[:, column] = np.minimum(1.0, np.maximum(0, candidate_count - 1) / 6.0)
    if str(payload.get("model_type", "")) == "random_forest":
        return deployed_forest_probabilities(payload, baseline_features), version
    legacy = StandardizedLogisticModel.load(
        model_path,
        "ensemble_candidate_calibration",
        FEATURE_NAMES,
    )
    if not legacy.enabled:
        raise RuntimeError("ensemble baseline model is invalid")
    return (
        np.asarray([legacy.predict(row) for row in baseline_features], dtype=np.float64),
        legacy.model_version,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "src" / "scorescan" / "resources" / "ensemble_calibrator.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT.parent / "training" / "ensemble_calibrator_report_v3.json",
    )
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=ROOT.parent / "training" / "baselines" / "ensemble_calibrator_v2.json",
    )
    parser.add_argument("--seed", type=int, default=20270115)
    parser.add_argument("--groups", type=int, default=2400)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    dataset = build_dataset(args.seed, args.groups, max(1, args.workers))
    train, calibration, audit, test = _split_groups(dataset.groups, args.seed)

    audit_rows: list[dict[str, object]] = []
    trained: list[tuple[RandomForestClassifier, LogisticRegression, dict[str, object]]] = []
    for config in MODEL_CONFIGS:
        model, calibrator, payload = _fit_forest(
            dataset.features,
            dataset.labels,
            train,
            calibration,
            args.seed,
            config,
        )
        probabilities = _calibrated_probabilities(
            model,
            calibrator,
            dataset.features[audit],
        )
        audit_rows.append(
            {
                "config": dict(config),
                "sample": _sample_metrics(dataset.labels[audit], probabilities),
                "decision": _group_top1(probabilities, dataset, audit),
            }
        )
        trained.append((model, calibrator, payload))

    best_top1 = max(float(row["decision"]["top1_accuracy"]) for row in audit_rows)
    best_log_loss = min(float(row["sample"]["log_loss"]) for row in audit_rows)
    # The meta-calibrator is a bounded tie-breaker, so a large resource is not justified
    # by a marginal synthetic-audit gain. Prefer the smallest forest within 2.5
    # percentage points of the best group Top-1 and 0.02 log-loss of the best model.
    compact = [
        index
        for index, row in enumerate(audit_rows)
        if float(row["decision"]["top1_accuracy"]) >= best_top1 - 0.025
        and float(row["sample"]["log_loss"]) <= best_log_loss + 0.02
    ]
    selected_index = min(
        compact,
        key=lambda index: (
            int(audit_rows[index]["config"]["n_estimators"]),
            int(audit_rows[index]["config"]["max_depth"]),
            int(audit_rows[index]["config"]["min_samples_leaf"]),
        ),
    )
    model, calibrator, payload = trained[selected_index]
    selected_config = dict(audit_rows[selected_index]["config"])
    payload["training_groups"] = args.groups
    payload["selected_on"] = "independent grouped model-selection audit"
    payload["selection_rule"] = (
        "smallest forest within 0.025 group Top-1 and 0.02 log-loss of the independent-audit best"
    )
    payload["component_models"] = {
        "measure": MeasureCalibrator().model_version,
        "visual": VisualMeasureCalibrator().model_version,
        "event": EventCalibrator().model_version,
        "context": ContextCalibrator().model_version,
    }

    sklearn_test = _calibrated_probabilities(model, calibrator, dataset.features[test])
    deployed_test = deployed_forest_probabilities(payload, dataset.features[test])
    deployment_delta = float(np.max(np.abs(sklearn_test - deployed_test), initial=0.0))
    if deployment_delta > 1e-10:
        raise RuntimeError(f"deployment prediction mismatch: {deployment_delta}")

    baseline_test, baseline_version = _baseline_probabilities(
        args.baseline_model,
        dataset.features[test],
    )
    linear_test = _fit_linear_ablation(dataset, train, test, args.seed)

    scenarios = sorted(set(dataset.scenarios))
    policy_threshold = DEFAULT_POLICY.replacement_ensemble_probability_floor
    report = {
        "model_version": MODEL_VERSION,
        "seed": args.seed,
        "groups": args.groups,
        "samples": int(len(dataset.labels)),
        "positive_rate": float(np.mean(dataset.labels)),
        "split_unit": (
            "complete source decision with three to eight candidates across up to five independent families; correlated preprocessing families never cross partitions"
        ),
        "samples_by_partition": {
            "train": len(train),
            "probability_calibration": len(calibration),
            "model_selection_audit": len(audit),
            "frozen_test": len(test),
        },
        "component_models": payload["component_models"],
        "model_selection_audit": audit_rows,
        "selected_config": selected_config,
        "selection_rule": payload["selection_rule"],
        "frozen_test": {
            "sample": _sample_metrics(dataset.labels[test], deployed_test),
            "policy_gate": _sample_metrics(dataset.labels[test], deployed_test, policy_threshold),
            "decision": _group_top1(deployed_test, dataset, test),
            "by_scenario": {
                scenario: _group_top1(deployed_test, dataset, test, scenario=scenario)
                for scenario in scenarios
            },
        },
        "baseline_v2_same_frozen_test": {
            "model_version": baseline_version,
            "sample": _sample_metrics(dataset.labels[test], baseline_test),
            "policy_gate": _sample_metrics(dataset.labels[test], baseline_test, policy_threshold),
            "decision": _group_top1(baseline_test, dataset, test),
            "by_scenario": {
                scenario: _group_top1(baseline_test, dataset, test, scenario=scenario)
                for scenario in scenarios
            },
        },
        "linear_model_ablation": {
            "description": "retrained standardized logistic regression using all current features",
            "sample": _sample_metrics(dataset.labels[test], linear_test),
            "decision": _group_top1(linear_test, dataset, test),
        },
        "deployment_parity": {
            "max_absolute_probability_delta": deployment_delta,
            "tolerance": 1e-10,
        },
        "feature_names": list(FEATURE_NAMES),
        "scope": (
            "programmatic difficult measure-candidate decisions scored by current component models; not end-to-end OMR"
        ),
        "limitations": [
            "No large frozen real-scan candidate ensemble corpus is bundled.",
            "The model is a bounded prior and cannot create consensus or bypass hard validation.",
            "Cross-family fuzzy errors remain difficult when several independent evidence layers share the same mistake.",
        ],
    }

    atomic_write_json(args.output, payload)
    atomic_write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
