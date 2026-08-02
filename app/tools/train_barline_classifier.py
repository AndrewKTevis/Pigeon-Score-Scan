from __future__ import annotations

"""Train the CPU local-barline classifier from production proposal distributions."""

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

from barline_training_data import flatten_examples, render_groups  # noqa: E402
from scorescan.barline_classifier import FEATURE_NAMES  # noqa: E402
from scorescan.linear_model import StandardizedLogisticModel  # noqa: E402
from scorescan.util import atomic_write_json  # noqa: E402
from tree_export import deployed_forest_probabilities, serialize_probability_forest  # noqa: E402

MODEL_VERSION = "scorescan-barline-forest-2"
LEGACY_FEATURE_NAMES = FEATURE_NAMES[:14]


def _split_groups(group_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    groups = sorted(set(int(value) for value in group_ids.tolist()))
    random.Random(seed).shuffle(groups)
    count = len(groups)
    train_cut = int(round(count * 0.70))
    calibration_cut = int(round(count * 0.80))
    threshold_cut = int(round(count * 0.90))
    partitions = (
        set(groups[:train_cut]),
        set(groups[train_cut:calibration_cut]),
        set(groups[calibration_cut:threshold_cut]),
        set(groups[threshold_cut:]),
    )
    return tuple(
        np.flatnonzero(np.isin(group_ids, np.asarray(sorted(partition), dtype=group_ids.dtype)))
        for partition in partitions
    )  # type: ignore[return-value]


def _threshold(labels: np.ndarray, probabilities: np.ndarray, minimum_precision: float = 0.995) -> float:
    order = np.argsort(-probabilities, kind="stable")
    sorted_labels = labels[order]
    sorted_probabilities = probabilities[order]
    true_positives = np.cumsum(sorted_labels == 1)
    false_positives = np.cumsum(sorted_labels == 0)
    precision = true_positives / np.maximum(true_positives + false_positives, 1)
    recall = true_positives / max(int(np.sum(labels == 1)), 1)
    valid = np.flatnonzero(precision >= minimum_precision)
    if valid.size == 0:
        return 0.999
    best = int(valid[np.argmax(recall[valid])])
    return float(sorted_probabilities[best])


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = probabilities >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predicted,
        average="binary",
        zero_division=0,
    )
    false_accepts = int(np.sum(predicted & (labels == 0)))
    false_rejects = int(np.sum((~predicted) & (labels == 1)))
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
        "accepted": int(np.sum(predicted)),
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
    }


def _legacy_probabilities(path: Path, values: np.ndarray) -> tuple[str, np.ndarray]:
    model = StandardizedLogisticModel.load(path, "barline_classification", LEGACY_FEATURE_NAMES)
    probabilities = np.asarray(
        [model.predict(row[: len(LEGACY_FEATURE_NAMES)], neutral=0.5) for row in values],
        dtype=np.float64,
    )
    return model.model_version, probabilities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "src" / "scorescan" / "resources" / "barline_classifier.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT.parent / "training" / "barline_classifier_report_v2.json",
    )
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=ROOT.parent / "training" / "baselines" / "barline_classifier_v1.json",
    )
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--groups", type=int, default=400)
    parser.add_argument("--variants-per-group", type=int, default=2)
    args = parser.parse_args()

    systems = render_groups(args.seed, args.groups, args.variants_per_group)
    examples = flatten_examples(systems)
    x = np.asarray([item.features.vector() for item in examples], dtype=np.float64)
    y = np.asarray([item.label for item in examples], dtype=np.int32)
    groups = np.asarray([item.group for item in examples], dtype=np.int32)
    train, calibration, threshold_indices, test = _split_groups(groups, args.seed)

    model = RandomForestClassifier(
        n_estimators=32,
        max_depth=10,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=1,
        random_state=args.seed,
    )
    model.fit(x[train], y[train])

    calibration_raw = model.predict_proba(x[calibration])[:, 1]
    calibrator = LogisticRegression(C=1000.0, max_iter=1000, random_state=args.seed)
    calibrator.fit(calibration_raw.reshape(-1, 1), y[calibration])
    payload: dict[str, object] = {
        "model_version": MODEL_VERSION,
        "model_type": "random_forest",
        "feature_names": list(FEATURE_NAMES),
        "trees": serialize_probability_forest(model),
        "calibration_intercept": float(calibrator.intercept_[0]),
        "calibration_slope": float(calibrator.coef_[0, 0]),
        "training_seed": args.seed,
        "training_groups": args.groups,
        "variants_per_group": args.variants_per_group,
        "training_distribution": "complete rendered systems passed through production vertical proposals",
    }

    sklearn_test = calibrator.predict_proba(model.predict_proba(x[test])[:, 1].reshape(-1, 1))[:, 1]
    deployed_test = deployed_forest_probabilities(payload, x[test])
    deployment_delta = float(np.max(np.abs(sklearn_test - deployed_test), initial=0.0))
    if deployment_delta > 1e-10:
        raise RuntimeError(f"deployment prediction mismatch: {deployment_delta}")

    threshold_probabilities = deployed_forest_probabilities(payload, x[threshold_indices])
    recommended_threshold = _threshold(y[threshold_indices], threshold_probabilities)
    payload["recommended_threshold"] = recommended_threshold

    legacy_version, legacy_test = _legacy_probabilities(args.baseline_model, x[test])
    report = {
        "model_version": MODEL_VERSION,
        "seed": args.seed,
        "groups": args.groups,
        "variants_per_group": args.variants_per_group,
        "rendered_systems": len(systems),
        "proposal_samples": len(examples),
        "positive_rate": float(np.mean(y)),
        "split_unit": "rendered staff-system identity; degraded variants never cross partitions",
        "samples": {
            "train": len(train),
            "probability_calibration": len(calibration),
            "threshold_selection": len(threshold_indices),
            "frozen_test": len(test),
        },
        "recommended_threshold": recommended_threshold,
        "threshold_selection": _metrics(y[threshold_indices], threshold_probabilities, recommended_threshold),
        "frozen_test": _metrics(y[test], deployed_test, 0.5),
        "frozen_test_at_recommended_threshold": _metrics(y[test], deployed_test, recommended_threshold),
        "baseline_same_frozen_test": {
            "model_version": legacy_version,
            "at_0_5": _metrics(y[test], legacy_test, 0.5),
            "at_rc5_runtime_threshold": _metrics(y[test], legacy_test, 0.68),
        },
        "deployment_parity": {
            "max_absolute_probability_delta": deployment_delta,
            "implementation": "dependency-free production JSON runtime",
        },
        "scope": "grouped rendered proposal classification; not end-to-end OMR accuracy",
        "limitations": [
            "No large frozen real-scan proposal corpus is bundled.",
            "The generator targets deskewed single-staff printed notation.",
        ],
    }
    atomic_write_json(args.output, payload)
    atomic_write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
