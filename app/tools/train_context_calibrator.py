from __future__ import annotations

"""Train the family-balanced CPU cross-measure context calibrator.

Complete seven-variant ensembles are split by source-segment identity into training,
independent probability calibration, model-selection audit, and frozen test partitions.
The model remains a narrow prior: it cannot edit notes, create consensus, or bypass
MusicXML and rhythm validation.
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

from context_training_data import ContextDataset, build_dataset  # noqa: E402
from scorescan.context_calibration import FEATURE_NAMES, LEGACY_FEATURE_NAMES  # noqa: E402
from scorescan.tree_model import VerifiedGradientBoostingModel  # noqa: E402
from scorescan.util import atomic_write_json  # noqa: E402
from tree_export import deployed_forest_probabilities, serialize_probability_forest  # noqa: E402

MODEL_VERSION = "scorescan-context-forest-2"
MODEL_CONFIGS = (
    {"n_estimators": 24, "max_depth": 6, "min_samples_leaf": 8},
    {"n_estimators": 32, "max_depth": 7, "min_samples_leaf": 6},
    {"n_estimators": 48, "max_depth": 8, "min_samples_leaf": 6},
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
        np.flatnonzero(
            np.isin(
                group_ids,
                np.asarray(sorted(partition), dtype=group_ids.dtype),
            )
        )
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
    calibrator = LogisticRegression(
        C=1000.0,
        max_iter=2000,
        random_state=seed,
    )
    calibrator.fit(raw_calibration.reshape(-1, 1), labels[calibration])
    payload: dict[str, object] = {
        "model_version": MODEL_VERSION,
        "model_type": "random_forest",
        "feature_names": list(FEATURE_NAMES),
        "trees": serialize_probability_forest(model),
        "calibration_intercept": float(calibrator.intercept_[0]),
        "calibration_slope": float(calibrator.coef_[0][0]),
        "training_seed": seed,
    }
    return model, calibrator, payload


def _calibrated_probabilities(
    model: RandomForestClassifier,
    calibrator: LogisticRegression,
    features: np.ndarray,
) -> np.ndarray:
    raw = model.predict_proba(features)[:, 1]
    return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


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
    dataset: ContextDataset,
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


def _fit_legacy_ablation(
    dataset: ContextDataset,
    train: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
    seed: int,
    config: dict[str, int],
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
    return calibrator.predict_proba(raw_test.reshape(-1, 1))[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "src" / "scorescan" / "resources" / "context_calibrator.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT.parent / "training" / "context_calibrator_report_v2.json",
    )
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=ROOT.parent / "training" / "baselines" / "context_calibrator_v1.json",
    )
    parser.add_argument("--seed", type=int, default=20261201)
    parser.add_argument("--groups", type=int, default=2400)
    args = parser.parse_args()

    dataset = build_dataset(args.seed, args.groups)
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
    top1_tied = [
        index
        for index, row in enumerate(audit_rows)
        if float(row["decision"]["top1_accuracy"]) == best_top1
    ]
    best_log_loss = min(float(audit_rows[index]["sample"]["log_loss"]) for index in top1_tied)
    compact_equivalents = [
        index
        for index in top1_tied
        if float(audit_rows[index]["sample"]["log_loss"]) <= best_log_loss + 0.01
    ]
    selected_index = min(
        compact_equivalents,
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

    sklearn_test = _calibrated_probabilities(
        model,
        calibrator,
        dataset.features[test],
    )
    deployed_test = deployed_forest_probabilities(payload, dataset.features[test])
    deployment_delta = float(
        np.max(np.abs(sklearn_test - deployed_test), initial=0.0)
    )
    if deployment_delta > 1e-10:
        raise RuntimeError(f"deployment prediction mismatch: {deployment_delta}")

    baseline_model = VerifiedGradientBoostingModel.load(
        args.baseline_model,
        "context_candidate_calibration",
        LEGACY_FEATURE_NAMES,
    )
    if not baseline_model.enabled:
        raise RuntimeError("context v1 baseline model is invalid")
    baseline_test = np.asarray(
        [baseline_model.predict(row) for row in dataset.legacy_features[test]],
        dtype=np.float64,
    )
    ablation_test = _fit_legacy_ablation(
        dataset,
        train,
        calibration,
        test,
        args.seed,
        selected_config,
    )

    scenarios = sorted(set(dataset.scenarios))
    report = {
        "model_version": MODEL_VERSION,
        "seed": args.seed,
        "groups": args.groups,
        "samples": int(len(dataset.labels)),
        "positive_rate": float(np.mean(dataset.labels)),
        "split_unit": (
            "source three-measure segment identity; all seven correlated preprocessing variants remain together"
        ),
        "samples_by_partition": {
            "train": len(train),
            "probability_calibration": len(calibration),
            "model_selection_audit": len(audit),
            "frozen_test": len(test),
        },
        "model_selection_audit": audit_rows,
        "selected_config": selected_config,
        "frozen_test": {
            "sample": _sample_metrics(dataset.labels[test], deployed_test),
            "decision": _group_top1(deployed_test, dataset, test),
            "by_scenario": {
                scenario: _group_top1(
                    deployed_test,
                    dataset,
                    test,
                    scenario=scenario,
                )
                for scenario in scenarios
            },
        },
        "baseline_v1_same_frozen_test": {
            "model_version": baseline_model.model_version,
            "sample": _sample_metrics(dataset.labels[test], baseline_test),
            "decision": _group_top1(baseline_test, dataset, test),
            "by_scenario": {
                scenario: _group_top1(
                    baseline_test,
                    dataset,
                    test,
                    scenario=scenario,
                )
                for scenario in scenarios
            },
        },
        "family_feature_ablation": {
            "description": (
                "same forest configuration and split using only the twenty-four same-variant context features"
            ),
            "sample": _sample_metrics(dataset.labels[test], ablation_test),
            "decision": _group_top1(ablation_test, dataset, test),
            "by_scenario": {
                scenario: _group_top1(
                    ablation_test,
                    dataset,
                    test,
                    scenario=scenario,
                )
                for scenario in scenarios
            },
        },
        "deployment_parity": {
            "max_absolute_probability_delta": deployment_delta,
            "implementation": "dependency-free verified random-forest JSON runtime",
        },
        "scope": (
            "grouped procedural three-measure candidate selection with preprocessing-family correlations; "
            "not end-to-end scan or MusicXML accuracy"
        ),
        "limitations": [
            "No large frozen real-scan candidate sequence corpus is bundled.",
            "Context is a bounded prior and cannot override event consensus or hard validation.",
            "Context-neutral internal note errors remain the responsibility of event and visual evidence.",
        ],
    }
    atomic_write_json(args.output, payload)
    atomic_write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
