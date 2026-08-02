from __future__ import annotations

"""Train the verified CPU pairwise rhythm-symbol compatibility guard."""

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

from rhythm_symbol_training_data import build_rendered_rhythm_symbol_dataset  # noqa: E402
from scorescan.rhythm_symbol_guard import (  # noqa: E402
    RHYTHM_SYMBOL_FEATURE_NAMES,
    RHYTHM_SYMBOL_OBSERVATION_FEATURE_NAMES,
)
from scorescan.util import atomic_write_json  # noqa: E402
from tree_export import deployed_forest_probabilities, serialize_probability_forest  # noqa: E402

MODEL_VERSION = "scorescan-rhythm-symbol-forest-1"
MODEL_CONFIGS = (
    {"n_estimators": 64, "max_depth": 9, "min_samples_leaf": 4},
    {"n_estimators": 80, "max_depth": 10, "min_samples_leaf": 4},
    {"n_estimators": 96, "max_depth": 11, "min_samples_leaf": 3},
)
MINIMUM_PUBLICATION_THRESHOLD = 0.98
MINIMUM_POSITIVE_RECALL = 0.60


def _split(groups: np.ndarray, seed: int) -> tuple[np.ndarray, ...]:
    unique = sorted(set(int(value) for value in groups.tolist()))
    random.Random(seed).shuffle(unique)
    count = len(unique)
    cuts = (int(count * 0.65), int(count * 0.75), int(count * 0.85), int(count * 0.90))
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
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(features[train], labels[train])
    raw = model.predict_proba(features[calibration])[:, 1]
    calibrator = LogisticRegression(C=1000.0, max_iter=3000, random_state=seed)
    calibrator.fit(raw.reshape(-1, 1), labels[calibration])
    return model, calibrator


def _payload(model, calibrator, seed, config):
    return {
        "model_version": MODEL_VERSION,
        "model_type": "random_forest",
        "feature_names": list(RHYTHM_SYMBOL_FEATURE_NAMES),
        "trees": serialize_probability_forest(model),
        "calibration_intercept": float(calibrator.intercept_[0]),
        "calibration_slope": float(calibrator.coef_[0, 0]),
        "training_seed": seed,
        "model_config": dict(config),
        "target": "the complete proposed rhythm transaction is more compatible with the local source crop than the template transaction",
        "scope": "pairwise measure-transaction veto aggregated over every changed event after strict meter-complete independent-family rhythm consensus",
    }


def _probabilities(model, calibrator, values):
    raw = model.predict_proba(values)[:, 1]
    return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


def _confidence(probabilities: np.ndarray, indices: np.ndarray) -> np.ndarray:
    reverse = np.bitwise_xor(indices, 1)
    return np.minimum(probabilities[indices], 1.0 - probabilities[reverse])


def _policy(labels, probabilities, indices, threshold):
    confidence = _confidence(probabilities, indices)
    local_labels = labels[indices]
    accepted = confidence >= threshold
    true_accepts = int(np.sum(accepted & (local_labels == 1)))
    false_accepts = int(np.sum(accepted & (local_labels == 0)))
    return {
        "threshold": float(threshold),
        "accepted": int(np.sum(accepted)),
        "true_accepts": true_accepts,
        "false_accepts": false_accepts,
        "precision": true_accepts / max(true_accepts + false_accepts, 1),
        "coverage": float(np.mean(accepted)),
        "positive_recall": true_accepts / max(int(np.sum(local_labels == 1)), 1),
        "maximum_false_direction_confidence": float(
            np.max(confidence[local_labels == 0], initial=0.0)
        ),
    }


def _sample(labels, probabilities):
    return {
        "accuracy": float(accuracy_score(labels, probabilities >= 0.5)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }


def _select_threshold(labels, probabilities, indices):
    confidence = _confidence(probabilities, indices)
    local_labels = labels[indices]
    candidates = {
        MINIMUM_PUBLICATION_THRESHOLD,
        0.985,
        0.99,
        0.995,
        0.999,
    }
    candidates.update(
        float(value)
        for value in confidence
        if math_is_finite(value) and float(value) >= MINIMUM_PUBLICATION_THRESHOLD
    )
    valid = []
    for threshold in sorted(candidates):
        metrics = _policy(labels, probabilities, indices, threshold)
        if metrics["accepted"] and metrics["false_accepts"] == 0:
            valid.append((threshold, metrics))
    if not valid:
        return 1.0, _policy(labels, probabilities, indices, 1.0)
    # Maximum safe coverage, then prefer the stricter threshold on a tie.
    return max(
        valid,
        key=lambda item: (
            item[1]["positive_recall"],
            item[1]["coverage"],
            item[0],
        ),
    )


def math_is_finite(value: float) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError, OverflowError):
        return False


def _canonical_report(value):
    """Remove non-semantic BLAS reduction noise from persisted evaluation JSON."""
    if isinstance(value, float):
        if not math_is_finite(value):
            return value
        return round(value, 12)
    if isinstance(value, dict):
        return {key: _canonical_report(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonical_report(item) for item in value]
    if isinstance(value, tuple):
        return [_canonical_report(item) for item in value]
    return value


def _scenario_metrics(scenarios, indices, labels, probabilities, threshold):
    values = np.asarray(scenarios)
    kinds = np.asarray([value.split(":", 1)[0] for value in values])
    result = {}
    for kind in sorted(set(kinds[indices].tolist())):
        local_indices = indices[kinds[indices] == kind]
        result[kind] = _policy(labels, probabilities, local_indices, threshold)
    return result


def _candidate_only_indices() -> np.ndarray:
    observation_count = len(RHYTHM_SYMBOL_OBSERVATION_FEATURE_NAMES)
    event_count = observation_count * 3
    visual_count = 23
    candidate_count = 15
    result: list[int] = []
    for aggregate_offset in (0, event_count, event_count * 2):
        for observation_offset in (0, observation_count, observation_count * 2):
            start = aggregate_offset + observation_offset + visual_count
            result.extend(range(start, start + candidate_count))
    return np.asarray(result, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "src" / "scorescan" / "resources" / "rhythm_symbol_guard.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT.parent / "training" / "rhythm_symbol_guard_report_v1.json",
    )
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--groups", type=int, default=800)
    parser.add_argument("--confirmation-groups", type=int, default=300)
    parser.add_argument("--stress-groups", type=int, default=0)
    args = parser.parse_args()

    dataset = build_rendered_rhythm_symbol_dataset(args.seed, args.groups)
    train, calibration, audit, threshold_indices, test = _split(dataset.groups, args.seed)
    trained = []
    selection = []
    for config in MODEL_CONFIGS:
        model, calibrator = _fit(
            dataset.features, dataset.labels, train, calibration, args.seed, config
        )
        probabilities = _probabilities(model, calibrator, dataset.features[audit])
        selection.append({"config": dict(config), "sample": _sample(dataset.labels[audit], probabilities)})
        trained.append((model, calibrator))
    selected_index = min(
        range(len(selection)),
        key=lambda index: (
            -float(selection[index]["sample"]["roc_auc"]),
            float(selection[index]["sample"]["log_loss"]),
            int(selection[index]["config"]["n_estimators"]),
        ),
    )
    model, calibrator = trained[selected_index]
    selected_config = dict(selection[selected_index]["config"])
    probabilities = _probabilities(model, calibrator, dataset.features)
    threshold, threshold_policy = _select_threshold(
        dataset.labels, probabilities, threshold_indices
    )
    payload = _payload(model, calibrator, args.seed, selected_config)
    payload.update({
        "training_groups": args.groups,
        "selected_config": selected_config,
        "selected_on": "independent group-isolated rendered audit",
        "auto_patch_threshold": threshold,
        "reverse_probability_ceiling": 1.0 - threshold,
        "target_precision": 1.0,
    })

    test_probabilities = probabilities[test]
    deployed = deployed_forest_probabilities(payload, dataset.features[test])
    deployment_delta = float(np.max(np.abs(test_probabilities - deployed), initial=0.0))
    if deployment_delta > 1e-10:
        raise RuntimeError(f"deployment prediction mismatch: {deployment_delta}")

    confirmation = build_rendered_rhythm_symbol_dataset(args.seed + 4, args.confirmation_groups)
    confirmation_probabilities = _probabilities(model, calibrator, confirmation.features)
    frozen_policy = _policy(dataset.labels, probabilities, test, threshold)
    confirmation_indices = np.arange(len(confirmation.labels), dtype=np.int64)
    confirmation_policy = _policy(
        confirmation.labels, confirmation_probabilities, confirmation_indices, threshold
    )
    stress = None
    stress_probabilities = None
    stress_policy = None
    if args.stress_groups > 0:
        stress = build_rendered_rhythm_symbol_dataset(args.seed + 112, args.stress_groups)
        stress_probabilities = _probabilities(model, calibrator, stress.features)
        stress_indices = np.arange(len(stress.labels), dtype=np.int64)
        stress_policy = _policy(stress.labels, stress_probabilities, stress_indices, threshold)
    false_accepts = int(frozen_policy["false_accepts"]) + int(confirmation_policy["false_accepts"])
    if stress_policy is not None:
        false_accepts += int(stress_policy["false_accepts"])
    if false_accepts:
        suffix = (
            f", stress={stress_policy['false_accepts']}"
            if stress_policy is not None
            else ""
        )
        raise RuntimeError(
            "rhythm symbol guard failed zero-error-accept publication gate: "
            f"frozen={frozen_policy['false_accepts']}, "
            f"confirmation={confirmation_policy['false_accepts']}{suffix}"
        )
    recalls = [
        float(frozen_policy["positive_recall"]),
        float(confirmation_policy["positive_recall"]),
    ]
    if stress_policy is not None:
        recalls.append(float(stress_policy["positive_recall"]))
    if min(recalls) < MINIMUM_POSITIVE_RECALL:
        suffix = (
            f", stress={stress_policy['positive_recall']:.4f}"
            if stress_policy is not None
            else ""
        )
        raise RuntimeError(
            "rhythm symbol guard coverage is too low for deployment: "
            f"frozen={frozen_policy['positive_recall']:.4f}, "
            f"confirmation={confirmation_policy['positive_recall']:.4f}{suffix}"
        )

    candidate_indices = _candidate_only_indices()
    ablation_features = dataset.features[:, candidate_indices]
    ablation_model, ablation_calibrator = _fit(
        ablation_features, dataset.labels, train, calibration, args.seed, selected_config
    )
    ablation_probabilities = _probabilities(
        ablation_model, ablation_calibrator, ablation_features[test]
    )

    atomic_write_json(args.output, payload)
    report = {
        "model_version": MODEL_VERSION,
        "seed": args.seed,
        "groups": args.groups,
        "samples": len(dataset.labels),
        "scope": "rendered pairwise rhythm-transaction compatibility aggregated over changed events; not end-to-end OMR accuracy",
        "partitions": {
            "train": len(train),
            "calibration": len(calibration),
            "model_selection_audit": len(audit),
            "threshold_selection": len(threshold_indices),
            "frozen_test": len(test),
        },
        "model_selection": selection,
        "selected_config": selected_config,
        "selected_threshold": threshold_policy,
        "frozen_test": {
            "sample": _sample(dataset.labels[test], test_probabilities),
            "policy": frozen_policy,
            "scenarios": _scenario_metrics(
                dataset.scenarios, test, dataset.labels, probabilities, threshold
            ),
            "unguarded_accept_all": {
                "true_accepts": int(np.sum(dataset.labels[test] == 1)),
                "false_accepts": int(np.sum(dataset.labels[test] == 0)),
                "precision": 0.5,
                "positive_recall": 1.0,
            },
        },
        "independent_confirmation": {
            "seed": args.seed + 4,
            "groups": args.confirmation_groups,
            "sample": _sample(confirmation.labels, confirmation_probabilities),
            "policy": confirmation_policy,
        },
        "ablation_candidate_signatures_only": {
            "feature_count": int(ablation_features.shape[1]),
            "frozen_test_sample": _sample(dataset.labels[test], ablation_probabilities),
        },
        "deployment_parity": {"max_absolute_probability_delta": deployment_delta},
        "feature_names": list(RHYTHM_SYMBOL_FEATURE_NAMES),
        "model_bytes": args.output.stat().st_size,
    }
    if stress is not None and stress_probabilities is not None and stress_policy is not None:
        report["independent_stress_audit"] = {
            "seed": args.seed + 112,
            "groups": args.stress_groups,
            "sample": _sample(stress.labels, stress_probabilities),
            "policy": stress_policy,
        }
    report = _canonical_report(report)
    atomic_write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
