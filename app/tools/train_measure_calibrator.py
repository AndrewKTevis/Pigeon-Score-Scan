from __future__ import annotations

"""Train the bounded CPU measure-structure calibrator.

Source-measure identities are split into fitting, forest probability calibration,
compact-model selection and frozen test partitions.  The released model embeds the
bundled v2 logistic prior and combines it with a structural forest through a transparent
legacy-preserving gate.  This lets new boundary and XML-integrity evidence rescue or
veto close decisions without invalidating the probability scale already consumed by the
ensemble layer.

Metrics describe candidate selection inside procedural MusicXML ensembles, not
end-to-end OMR accuracy.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from measure_training_data import MeasureDataset, build_dataset  # noqa: E402
from scorescan.linear_model import StandardizedLogisticModel  # noqa: E402
from scorescan.measure_calibration import FEATURE_NAMES, LEGACY_FEATURE_NAMES  # noqa: E402
from scorescan.util import atomic_write_json  # noqa: E402
from tree_export import deployed_forest_probabilities, serialize_probability_forest  # noqa: E402

MODEL_VERSION = "scorescan-measure-forest-3"
MODEL_CONFIGS = (
    {"n_estimators": 24, "max_depth": 6, "min_samples_leaf": 8},
    {"n_estimators": 32, "max_depth": 7, "min_samples_leaf": 6},
    {"n_estimators": 48, "max_depth": 8, "min_samples_leaf": 6},
)
LEGACY_PRESERVATION_FLOORS = (0.50, 0.55, 0.60, 0.65)


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
        "format": 2,
        "model_version": MODEL_VERSION,
        "model_type": "random_forest",
        "feature_names": list(FEATURE_NAMES),
        "trees": serialize_probability_forest(model),
        "calibration_intercept": float(calibrator.intercept_[0]),
        "calibration_slope": float(calibrator.coef_[0][0]),
        "training_seed": seed,
    }
    return model, calibrator, payload


def _forest_probabilities(
    model: RandomForestClassifier,
    calibrator: LogisticRegression,
    features: np.ndarray,
) -> np.ndarray:
    raw = model.predict_proba(features)[:, 1]
    return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


def _baseline_probabilities(
    baseline_model: StandardizedLogisticModel,
    features: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [baseline_model.predict(row[: len(LEGACY_FEATURE_NAMES)]) for row in features],
        dtype=np.float64,
    )


def _hybrid_probabilities(
    baseline: np.ndarray,
    forest: np.ndarray,
    floor: float,
) -> np.ndarray:
    preserved = baseline * (floor + (1.0 - floor) * forest)
    return np.maximum(forest, preserved)


def _sample_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    predictions = probabilities >= 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    return {
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
    dataset: MeasureDataset,
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


def _fit_legacy_ablation(
    dataset: MeasureDataset,
    train: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
    seed: int,
    config: dict[str, int],
    baseline_model: StandardizedLogisticModel,
    floor: float,
) -> np.ndarray:
    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=1,
        random_state=seed,
    )
    model.fit(dataset.legacy_features[train], dataset.labels[train])
    raw_calibration = model.predict_proba(dataset.legacy_features[calibration])[:, 1]
    calibrator = LogisticRegression(C=1000.0, max_iter=2000, random_state=seed)
    calibrator.fit(raw_calibration.reshape(-1, 1), dataset.labels[calibration])
    raw_test = model.predict_proba(dataset.legacy_features[test])[:, 1]
    forest_test = calibrator.predict_proba(raw_test.reshape(-1, 1))[:, 1]
    baseline_test = np.asarray(
        [baseline_model.predict(row) for row in dataset.legacy_features[test]],
        dtype=np.float64,
    )
    return _hybrid_probabilities(baseline_test, forest_test, floor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "src" / "scorescan" / "resources" / "measure_calibrator.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT.parent / "training" / "measure_calibrator_report_v3.json",
    )
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=ROOT.parent / "training" / "baselines" / "measure_calibrator_v2.json",
    )
    parser.add_argument("--seed", type=int, default=20270201)
    parser.add_argument("--groups", type=int, default=2400)
    args = parser.parse_args()

    dataset = build_dataset(args.seed, args.groups)
    train, calibration, audit, test = _split_groups(dataset.groups, args.seed)
    baseline_payload = json.loads(args.baseline_model.read_text(encoding="utf-8"))
    baseline_model = StandardizedLogisticModel.from_payload(
        baseline_payload,
        LEGACY_FEATURE_NAMES,
        status="bundled_baseline",
    )
    if not baseline_model.enabled:
        raise RuntimeError("measure v2 baseline model is invalid")

    baseline_audit = _baseline_probabilities(baseline_model, dataset.features[audit])
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
        forest_audit = _forest_probabilities(model, calibrator, dataset.features[audit])
        trained.append((model, calibrator, payload))
        for floor in LEGACY_PRESERVATION_FLOORS:
            audit_probabilities = _hybrid_probabilities(baseline_audit, forest_audit, floor)
            audit_rows.append(
                {
                    "config": dict(config),
                    "legacy_preservation_floor": floor,
                    "sample": _sample_metrics(dataset.labels[audit], audit_probabilities),
                    "decision": _group_top1(audit_probabilities, dataset, audit),
                }
            )

    best_top1 = max(float(row["decision"]["top1_accuracy"]) for row in audit_rows)
    tied = [
        index
        for index, row in enumerate(audit_rows)
        if float(row["decision"]["top1_accuracy"]) == best_top1
    ]
    best_loss = min(float(audit_rows[index]["sample"]["log_loss"]) for index in tied)
    compact = [
        index
        for index in tied
        if float(audit_rows[index]["sample"]["log_loss"]) <= best_loss + 0.01
    ]
    selected_row_index = min(
        compact,
        key=lambda index: (
            int(audit_rows[index]["config"]["n_estimators"]),
            int(audit_rows[index]["config"]["max_depth"]),
            -float(audit_rows[index]["legacy_preservation_floor"]),
        ),
    )
    selected_row = audit_rows[selected_row_index]
    selected_config = dict(selected_row["config"])
    selected_floor = float(selected_row["legacy_preservation_floor"])
    config_index = next(index for index, config in enumerate(MODEL_CONFIGS) if config == selected_config)
    model, calibrator, payload = trained[config_index]
    payload["training_groups"] = args.groups
    payload["selected_on"] = "independent grouped model-selection audit"
    payload["legacy_model"] = baseline_payload
    payload["legacy_preservation_floor"] = selected_floor

    forest_test_sklearn = _forest_probabilities(model, calibrator, dataset.features[test])
    forest_test_deployed = deployed_forest_probabilities(payload, dataset.features[test])
    baseline_test = _baseline_probabilities(baseline_model, dataset.features[test])
    sklearn_test = _hybrid_probabilities(baseline_test, forest_test_sklearn, selected_floor)
    deployed_test = _hybrid_probabilities(baseline_test, forest_test_deployed, selected_floor)
    deployment_delta = float(np.max(np.abs(sklearn_test - deployed_test), initial=0.0))
    if deployment_delta > 1e-10:
        raise RuntimeError(f"deployment prediction mismatch: {deployment_delta}")

    ablation_test = _fit_legacy_ablation(
        dataset,
        train,
        calibration,
        test,
        args.seed,
        selected_config,
        baseline_model,
        selected_floor,
    )

    scenarios = sorted(set(dataset.scenarios))
    report = {
        "model_version": MODEL_VERSION,
        "seed": args.seed,
        "groups": args.groups,
        "samples": int(len(dataset.labels)),
        "positive_rate": float(np.mean(dataset.labels)),
        "split_unit": "source measure identity; all three-to-seven correlated candidates remain together",
        "samples_by_partition": {
            "train": len(train),
            "forest_probability_calibration": len(calibration),
            "model_selection_audit": len(audit),
            "frozen_test": len(test),
        },
        "hybrid_rule": (
            "max(forest_probability, legacy_probability * "
            "(legacy_preservation_floor + (1 - legacy_preservation_floor) * forest_probability))"
        ),
        "model_selection_audit": audit_rows,
        "selected_config": selected_config,
        "selected_legacy_preservation_floor": selected_floor,
        "frozen_test": {
            "sample": _sample_metrics(dataset.labels[test], deployed_test),
            "decision": _group_top1(deployed_test, dataset, test),
            "by_scenario": {
                scenario: _group_top1(deployed_test, dataset, test, scenario=scenario)
                for scenario in scenarios
            },
        },
        "baseline_v2_same_frozen_test": {
            "model_version": baseline_model.model_version,
            "sample": _sample_metrics(dataset.labels[test], baseline_test),
            "decision": _group_top1(baseline_test, dataset, test),
            "by_scenario": {
                scenario: _group_top1(baseline_test, dataset, test, scenario=scenario)
                for scenario in scenarios
            },
        },
        "new_feature_ablation": {
            "description": (
                "same selected forest, preservation floor and split using only the twenty-eight v2 features"
            ),
            "sample": _sample_metrics(dataset.labels[test], ablation_test),
            "decision": _group_top1(ablation_test, dataset, test),
            "by_scenario": {
                scenario: _group_top1(ablation_test, dataset, test, scenario=scenario)
                for scenario in scenarios
            },
        },
        "deployment_parity": {"max_absolute_probability_delta": deployment_delta},
        "limitations": [
            "Procedural immutable MusicXML measure candidates; not an end-to-end scan OMR metric.",
            "Pitch-only errors remain primarily the responsibility of event, context and ensemble evidence.",
            "Boundary features distinguish likely pickup/final partial measures but do not infer missing notes.",
            "The embedded v2 prior intentionally preserves the probability scale expected by downstream gates.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, payload)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
