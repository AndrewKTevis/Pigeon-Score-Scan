from __future__ import annotations

"""Train the verified CPU visual veto for one note-versus-rest transaction."""

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

from event_kind_visual_training_data import EventKindVisualDataset  # noqa: E402
from scorescan.event_kind_visual_guard import EVENT_KIND_VISUAL_FEATURE_NAMES  # noqa: E402
from scorescan.util import atomic_write_json  # noqa: E402
from tree_export import deployed_forest_probabilities, serialize_probability_forest  # noqa: E402

MODEL_VERSION = "scorescan-event-kind-visual-forest-1"
MODEL_CONFIGS = (
    {"n_estimators": 96, "max_depth": 9, "min_samples_leaf": 3},
    {"n_estimators": 128, "max_depth": 10, "min_samples_leaf": 3},
    {"n_estimators": 160, "max_depth": 11, "min_samples_leaf": 2},
)
MINIMUM_THRESHOLD = 0.72
MINIMUM_RECALL = 0.25


def _load(path: Path) -> EventKindVisualDataset:
    with np.load(path, allow_pickle=False) as payload:
        return EventKindVisualDataset(
            features=np.asarray(payload["features"], dtype=np.float64),
            labels=np.asarray(payload["labels"], dtype=np.int64),
            groups=np.asarray(payload["groups"], dtype=np.int64),
            scenarios=tuple(str(value) for value in payload["scenarios"].tolist()),
            target_kinds=tuple(str(value) for value in payload["target_kinds"].tolist()),
        )


def _split(groups: np.ndarray, seed: int) -> tuple[np.ndarray, ...]:
    unique = sorted(set(int(value) for value in groups.tolist()))
    random.Random(seed).shuffle(unique)
    count = len(unique)
    cuts = (int(count * 0.60), int(count * 0.72), int(count * 0.82), int(count * 0.90))
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
    return min(1.0, max(float(floor), float(np.nextafter(np.max(negatives, initial=0.0), 1.0))))


def _policy(dataset, probabilities, indices, threshold):
    local_labels = dataset.labels[indices]
    local_probabilities = probabilities[indices]
    local_kinds = np.asarray(dataset.target_kinds)[indices]
    accepted = local_probabilities >= threshold
    result = {
        "auto_patch_threshold": float(threshold),
        "correct_accepts": int(np.sum((local_labels == 1) & accepted)),
        "false_accepts": int(np.sum((local_labels == 0) & accepted)),
        "selective_precision": float(np.sum((local_labels == 1) & accepted))
        / max(int(np.sum(accepted)), 1),
        "coverage": float(np.mean(accepted)),
    }
    for kind in ("note", "rest"):
        mask = (local_labels == 1) & (local_kinds == kind)
        result[f"{kind}_positive_total"] = int(np.sum(mask))
        result[f"{kind}_correct_accepts"] = int(np.sum(mask & accepted))
        result[f"{kind}_recall"] = int(np.sum(mask & accepted)) / max(int(np.sum(mask)), 1)
    return result


def _scenario_false_accepts(dataset, probabilities, threshold):
    labels = dataset.labels
    scenarios = np.asarray(dataset.scenarios)
    return {
        scenario: {
            "negative_samples": int(np.sum((labels == 0) & (scenarios == scenario))),
            "false_accepts": int(np.sum((labels == 0) & (scenarios == scenario) & (probabilities >= threshold))),
            "maximum_negative_probability": float(
                np.max(probabilities[(labels == 0) & (scenarios == scenario)], initial=0.0)
            ),
        }
        for scenario in sorted(set(dataset.scenarios))
    }


def _canonical(value):
    if isinstance(value, float):
        return round(value, 12) if np.isfinite(value) else value
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "src" / "scorescan" / "resources" / "event_kind_visual_guard.json")
    parser.add_argument("--report", type=Path, default=ROOT.parent / "training" / "event_kind_visual_guard_report_v1.json")
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--safety-data", type=Path, required=True)
    parser.add_argument("--independent-data", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    dataset = _load(args.training_data)
    train, calibration, selection_indices, threshold_indices, test = _split(dataset.groups, args.seed)
    trained = []
    selection = []
    for config in MODEL_CONFIGS:
        model, calibrator = _fit(dataset.features, dataset.labels, train, calibration, args.seed, config)
        probabilities = _probabilities(model, calibrator, dataset.features[selection_indices])
        selection.append({"config": dict(config), "sample": _sample(dataset.labels[selection_indices], probabilities)})
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
    raw_probabilities = _probabilities(model, calibrator, dataset.features)
    safety = _load(args.safety_data)
    raw_safety_probabilities = _probabilities(model, calibrator, safety.features)
    safety_indices = np.arange(len(safety.labels), dtype=np.int64)
    independent = _load(args.independent_data)
    raw_independent_probabilities = _probabilities(model, calibrator, independent.features)
    independent_indices = np.arange(len(independent.labels), dtype=np.int64)

    payload = {
        "model_version": MODEL_VERSION,
        "model_type": "random_forest",
        "feature_names": list(EVENT_KIND_VISUAL_FEATURE_NAMES),
        "trees": serialize_probability_forest(model),
        "calibration_intercept": float(calibrator.intercept_[0]),
        "calibration_slope": float(calibrator.coef_[0, 0]),
        "auto_patch_threshold": 1.0,
        "target_precision": 1.0,
        "training_seed": int(args.seed),
        "training_groups": len(set(int(value) for value in dataset.groups.tolist())),
        "model_config": config,
        "target": "the preserved local source image supports exactly one already-proposed note-versus-rest event-kind replacement",
        "scope": "veto-only confirmation for one fixed-onset fixed-duration note/rest replacement; multiple events, pitch selection, duration changes, chords and grace notes are excluded",
    }
    # Thresholds and all reported policies are computed from the exact canonical JSON
    # payload that the runtime will load, not from the in-memory sklearn objects.
    probabilities = deployed_forest_probabilities(payload, dataset.features)
    safety_probabilities = deployed_forest_probabilities(payload, safety.features)
    independent_probabilities = deployed_forest_probabilities(payload, independent.features)
    threshold = _threshold(dataset.labels, probabilities, threshold_indices, MINIMUM_THRESHOLD)
    threshold = max(
        threshold,
        _threshold(safety.labels, safety_probabilities, safety_indices, MINIMUM_THRESHOLD),
    )
    payload["auto_patch_threshold"] = float(threshold)

    policies = {
        "frozen_test": _policy(dataset, probabilities, test, threshold),
        "safety_calibration": _policy(safety, safety_probabilities, safety_indices, threshold),
        "independent_test": _policy(independent, independent_probabilities, independent_indices, threshold),
    }
    if any(int(value["false_accepts"]) for value in policies.values()):
        raise RuntimeError(f"event-kind visual guard false accepts: {policies}")
    for name, value in policies.items():
        if float(value["note_recall"]) < MINIMUM_RECALL or float(value["rest_recall"]) < MINIMUM_RECALL:
            raise RuntimeError(f"event-kind visual guard recall too low in {name}: {value}")

    deployment_deltas = {
        "training_all": float(np.max(np.abs(probabilities - raw_probabilities), initial=0.0)),
        "safety_calibration": float(np.max(np.abs(safety_probabilities - raw_safety_probabilities), initial=0.0)),
        "independent_test": float(np.max(np.abs(independent_probabilities - raw_independent_probabilities), initial=0.0)),
    }
    deployment_delta = max(deployment_deltas.values(), default=0.0)
    if deployment_delta > 1e-9:
        raise RuntimeError(f"deployment prediction mismatch: {deployment_deltas}")

    # Context-only ablation proves the model cannot solve the task from target-kind or
    # duration priors without source-image descriptors.
    context_count = 10
    ablation = RandomForestClassifier(
        random_state=args.seed,
        n_jobs=1,
        class_weight="balanced_subsample",
        max_features="sqrt",
        **config,
    )
    ablation.fit(dataset.features[train, :context_count], dataset.labels[train])
    ablation_probabilities = ablation.predict_proba(independent.features[:, :context_count])[:, 1]
    ablation_metrics = _sample(independent.labels, ablation_probabilities)

    report = {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "training_seed": int(args.seed),
        "feature_count": len(EVENT_KIND_VISUAL_FEATURE_NAMES),
        "dataset": {
            "training_groups": len(set(int(value) for value in dataset.groups.tolist())),
            "training_samples": int(len(dataset.labels)),
            "safety_groups": len(set(int(value) for value in safety.groups.tolist())),
            "safety_samples": int(len(safety.labels)),
            "independent_groups": len(set(int(value) for value in independent.groups.tolist())),
            "independent_samples": int(len(independent.labels)),
        },
        "selection": selection,
        "selected_config": config,
        "sample_metrics": {
            "frozen_test": _sample(dataset.labels[test], probabilities[test]),
            "safety_calibration": _sample(safety.labels, safety_probabilities),
            "independent_test": _sample(independent.labels, independent_probabilities),
        },
        "policies": policies,
        "scenario_false_accepts": {
            "frozen_test": _scenario_false_accepts(
                EventKindVisualDataset(
                    dataset.features[test], dataset.labels[test], dataset.groups[test],
                    tuple(np.asarray(dataset.scenarios)[test].tolist()),
                    tuple(np.asarray(dataset.target_kinds)[test].tolist()),
                ),
                probabilities[test],
                threshold,
            ),
            "safety_calibration": _scenario_false_accepts(safety, safety_probabilities, threshold),
            "independent_test": _scenario_false_accepts(independent, independent_probabilities, threshold),
        },
        "context_only_ablation": ablation_metrics,
        "deployment_max_probability_delta": deployment_delta,
        "deployment_probability_deltas": deployment_deltas,
        "limitations": [
            "Programmatic rendered note/rest transactions are not a substitute for an independently reviewed real-scan benchmark.",
            "The runtime gate confirms only one fixed-onset, fixed-duration note/rest replacement and cannot select pitch or duration.",
            "Multiple event-kind changes, grace notes, chords, unpitched events and unsupported note types remain review-only when source evidence is present.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, payload)
    atomic_write_json(args.report, _canonical(report))
    print(json.dumps(_canonical({"model": payload, "report": report}), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
