from __future__ import annotations

"""Train the verified CPU visual veto for simple accent additions."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from accent_visual_training_data import AccentVisualDataset  # noqa: E402
from scorescan.accent_visual_guard import ACCENT_VISUAL_FEATURE_NAMES  # noqa: E402
from scorescan.util import atomic_write_json  # noqa: E402
from tree_export import deployed_forest_probabilities, serialize_probability_forest  # noqa: E402

MODEL_VERSION = "scorescan-accent-visual-forest-1"
MODEL_CONFIGS = (
    {"n_estimators": 96, "max_depth": 9, "min_samples_leaf": 3},
    {"n_estimators": 128, "max_depth": 10, "min_samples_leaf": 3},
    {"n_estimators": 160, "max_depth": 11, "min_samples_leaf": 2},
)
MINIMUM_THRESHOLD = 0.70
MINIMUM_RECALL = 0.25


def _split(groups: np.ndarray, seed: int) -> tuple[np.ndarray, ...]:
    unique = sorted(set(int(value) for value in groups.tolist()))
    random.Random(seed).shuffle(unique)
    count = len(unique)
    cuts = (
        int(count * 0.60),
        int(count * 0.72),
        int(count * 0.82),
        int(count * 0.90),
    )
    partitions = (
        set(unique[: cuts[0]]),
        set(unique[cuts[0] : cuts[1]]),
        set(unique[cuts[1] : cuts[2]]),
        set(unique[cuts[2] : cuts[3]]),
        set(unique[cuts[3] :]),
    )
    return tuple(
        np.flatnonzero(np.isin(groups, np.asarray(sorted(partition), dtype=groups.dtype)))
        for partition in partitions
    )


def _fit(features, labels, train, calibration, seed, config):
    model = RandomForestClassifier(
        random_state=seed,
        n_jobs=1,
        class_weight="balanced_subsample",
        max_features="sqrt",
        **config,
    )
    model.fit(features[train], labels[train])
    raw = model.predict_proba(features[calibration])[:, 1]
    calibrator = LogisticRegression(C=1.0, random_state=seed, solver="lbfgs")
    calibrator.fit(raw.reshape(-1, 1), labels[calibration])
    return model, calibrator


def _probabilities(model, calibrator, features):
    raw = model.predict_proba(features)[:, 1]
    return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


def _sample(labels, probabilities):
    return {
        "accuracy": float(accuracy_score(labels, probabilities >= 0.5)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }


def _threshold(labels, probabilities, indices, floor):
    local_labels = labels[indices]
    local_probabilities = probabilities[indices]
    negatives = local_probabilities[local_labels == 0]
    return min(
        1.0,
        max(
            float(floor),
            float(np.nextafter(np.max(negatives, initial=0.0), 1.0)),
        ),
    )


def _policy(labels, probabilities, indices, threshold):
    local_labels = labels[indices]
    local_probabilities = probabilities[indices]
    accepted = local_probabilities >= threshold
    correct = (local_labels == 1) & accepted
    wrong = (local_labels == 0) & accepted
    positive_total = max(int(np.sum(local_labels == 1)), 1)
    return {
        "present_threshold": float(threshold),
        "correct_accepts": int(np.sum(correct)),
        "false_accepts": int(np.sum(wrong)),
        "selective_precision": float(np.sum(correct))
        / max(int(np.sum(correct) + np.sum(wrong)), 1),
        "present_recall": int(np.sum(correct)) / positive_total,
        "coverage": float(np.mean(accepted)),
    }


def _scenario_false_accepts(dataset, probabilities, threshold):
    labels = dataset.labels
    scenarios = np.asarray(dataset.scenarios)
    result = {}
    for scenario in sorted(set(dataset.scenarios)):
        indices = np.flatnonzero((labels == 0) & (scenarios == scenario))
        result[scenario] = {
            "samples": int(len(indices)),
            "false_accepts": int(np.sum(probabilities[indices] >= threshold)),
            "maximum_probability": float(np.max(probabilities[indices], initial=0.0)),
        }
    return result


def _canonical(value):
    if isinstance(value, float):
        return round(value, 12) if np.isfinite(value) else value
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _load_dataset(path: Path) -> AccentVisualDataset:
    with np.load(path, allow_pickle=False) as payload:
        return AccentVisualDataset(
            features=np.asarray(payload["features"], dtype=np.float64),
            labels=np.asarray(payload["labels"], dtype=np.int64),
            groups=np.asarray(payload["groups"], dtype=np.int64),
            scenarios=tuple(str(value) for value in payload["scenarios"].tolist()),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "src" / "scorescan" / "resources" / "accent_visual_guard.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT.parent / "training" / "accent_visual_guard_report_v1.json",
    )
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--safety-data", type=Path, required=True)
    parser.add_argument("--independent-data", type=Path, required=True)
    args = parser.parse_args()

    dataset = _load_dataset(args.train_data)
    train, calibration, selection_indices, threshold_indices, test = _split(
        dataset.groups, args.seed
    )
    trained = []
    selection = []
    for config in MODEL_CONFIGS:
        model, calibrator = _fit(
            dataset.features,
            dataset.labels,
            train,
            calibration,
            args.seed,
            config,
        )
        local = _probabilities(model, calibrator, dataset.features[selection_indices])
        selection.append(
            {"config": dict(config), "sample": _sample(dataset.labels[selection_indices], local)}
        )
        trained.append((model, calibrator))
    selected_index = min(
        range(len(selection)),
        key=lambda index: (
            -selection[index]["sample"]["roc_auc"],
            selection[index]["sample"]["log_loss"],
            selection[index]["config"]["n_estimators"],
        ),
    )
    model, calibrator = trained[selected_index]
    config = dict(selection[selected_index]["config"])
    probabilities = _probabilities(model, calibrator, dataset.features)
    threshold = _threshold(
        dataset.labels, probabilities, threshold_indices, MINIMUM_THRESHOLD
    )

    safety = _load_dataset(args.safety_data)
    safety_probabilities = _probabilities(model, calibrator, safety.features)
    safety_indices = np.arange(len(safety.labels), dtype=np.int64)
    threshold = max(
        threshold,
        _threshold(
            safety.labels,
            safety_probabilities,
            safety_indices,
            MINIMUM_THRESHOLD,
        ),
    )

    independent = _load_dataset(args.independent_data)
    independent_probabilities = _probabilities(model, calibrator, independent.features)
    independent_indices = np.arange(len(independent.labels), dtype=np.int64)

    policies = {
        "frozen_test": _policy(dataset.labels, probabilities, test, threshold),
        "safety_calibration": _policy(
            safety.labels, safety_probabilities, safety_indices, threshold
        ),
        "independent_test": _policy(
            independent.labels, independent_probabilities, independent_indices, threshold
        ),
    }
    if any(int(value["false_accepts"]) for value in policies.values()):
        raise RuntimeError(f"accent visual guard false accepts: {policies}")
    if any(float(value["present_recall"]) < MINIMUM_RECALL for value in policies.values()):
        raise RuntimeError(f"accent visual guard recall too low: {policies}")

    payload = {
        "model_version": MODEL_VERSION,
        "model_type": "random_forest",
        "feature_names": list(ACCENT_VISUAL_FEATURE_NAMES),
        "trees": serialize_probability_forest(model),
        "calibration_intercept": float(calibrator.intercept_[0]),
        "calibration_slope": float(calibrator.coef_[0, 0]),
        "present_threshold": float(threshold),
        "target_precision": 1.0,
        "training_seed": int(args.seed),
        "training_groups": len(set(int(value) for value in dataset.groups.tolist())),
        "model_config": config,
        "target": "a printed accent is present above or below one already-proposed unarticulated pitched event",
        "scope": "veto-only confirmation for simple accent additions; removal, substitution, stacked articulation and staccato/tenuto decisions are excluded",
    }
    deployed = deployed_forest_probabilities(payload, dataset.features[test])
    deployment_delta = float(
        np.max(np.abs(deployed - probabilities[test]), initial=0.0)
    )
    if deployment_delta > 1e-10:
        raise RuntimeError(f"deployment prediction mismatch: {deployment_delta}")

    # Remove all explicit diagonal/wedge coverage features.  The remaining projection
    # and component features are an ablation witness, not a second runtime model.
    ablation_indices = np.asarray(
        [
            index
            for index, name in enumerate(ACCENT_VISUAL_FEATURE_NAMES)
            if "diagonal" not in name and "wedge" not in name
        ],
        dtype=np.int64,
    )
    ablation_model, ablation_calibrator = _fit(
        dataset.features[:, ablation_indices],
        dataset.labels,
        train,
        calibration,
        args.seed,
        config,
    )
    ablation_probabilities = _probabilities(
        ablation_model, ablation_calibrator, dataset.features[:, ablation_indices]
    )
    ablation_safety = _probabilities(
        ablation_model, ablation_calibrator, safety.features[:, ablation_indices]
    )
    ablation_independent = _probabilities(
        ablation_model, ablation_calibrator, independent.features[:, ablation_indices]
    )
    ablation_threshold = max(
        _threshold(
            dataset.labels,
            ablation_probabilities,
            threshold_indices,
            MINIMUM_THRESHOLD,
        ),
        _threshold(
            safety.labels,
            ablation_safety,
            safety_indices,
            MINIMUM_THRESHOLD,
        ),
    )
    ablation_policy = _policy(
        independent.labels,
        ablation_independent,
        independent_indices,
        ablation_threshold,
    )

    report = _canonical(
        {
            "format": 1,
            "model_version": MODEL_VERSION,
            "seed": args.seed,
            "scope": "programmatic rendered accent-presence evidence only; not end-to-end OMR accuracy",
            "groups": len(set(int(value) for value in dataset.groups.tolist())),
            "samples": len(dataset.labels),
            "partitions": {
                "train": len(train),
                "calibration": len(calibration),
                "model_selection": len(selection_indices),
                "threshold_selection": len(threshold_indices),
                "frozen_test": len(test),
            },
            "model_selection": selection,
            "selected_config": config,
            "threshold": threshold,
            "frozen_test": {
                "sample": _sample(dataset.labels[test], probabilities[test]),
                "policy": policies["frozen_test"],
            },
            "safety_calibration": {
                "groups": len(set(int(value) for value in safety.groups.tolist())),
                "sample": _sample(safety.labels, safety_probabilities),
                "policy": policies["safety_calibration"],
                "negative_scenarios": _scenario_false_accepts(
                    safety, safety_probabilities, threshold
                ),
            },
            "independent_test": {
                "groups": len(set(int(value) for value in independent.groups.tolist())),
                "sample": _sample(independent.labels, independent_probabilities),
                "policy": policies["independent_test"],
                "negative_scenarios": _scenario_false_accepts(
                    independent, independent_probabilities, threshold
                ),
            },
            "ablation_without_diagonal_geometry": {
                "feature_count": len(ablation_indices),
                "threshold": ablation_threshold,
                "independent_sample": _sample(
                    independent.labels, ablation_independent
                ),
                "independent_policy": ablation_policy,
            },
            "deployment_parity": {
                "max_absolute_probability_delta": deployment_delta
            },
            "feature_names": list(ACCENT_VISUAL_FEATURE_NAMES),
        }
    )
    atomic_write_json(args.output, payload)
    atomic_write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
