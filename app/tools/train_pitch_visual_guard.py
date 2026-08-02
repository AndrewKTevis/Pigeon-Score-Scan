from __future__ import annotations

"""Train the fail-closed CPU guard for staff-position pitch repairs.

The model sees only direct source-crop notehead geometry and transaction size.  It is
not allowed to create a pitch or a family majority.  A positive label means replacing
the template pitch grid with the proposal moves the semantic noteheads toward the
rendered source; the paired negative row performs the inverse regression on the same
crop and remains in the same split group.
"""

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

from pitch_visual_training_data import build_rendered_pitch_dataset  # noqa: E402
from scorescan.pitch_consensus import (  # noqa: E402
    PITCH_VISUAL_FEATURE_INDICES,
    PITCH_VISUAL_FEATURE_NAMES,
)
from scorescan.util import atomic_write_json  # noqa: E402
from tree_export import deployed_forest_probabilities, serialize_probability_forest  # noqa: E402

MODEL_VERSION = "scorescan-pitch-visual-guard-2"
MODEL_CONFIGS = (
    {"n_estimators": 64, "max_depth": 6, "min_samples_leaf": 8},
    {"n_estimators": 96, "max_depth": 7, "min_samples_leaf": 6},
    {"n_estimators": 128, "max_depth": 8, "min_samples_leaf": 5},
)


def _split(groups: np.ndarray, seed: int) -> tuple[np.ndarray, ...]:
    unique = sorted(set(int(value) for value in groups.tolist()))
    random.Random(seed).shuffle(unique)
    count = len(unique)
    cuts = (int(count * 0.50), int(count * 0.65), int(count * 0.75), int(count * 0.90))
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


def _fit(
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
    raw = model.predict_proba(features[calibration])[:, 1]
    calibrator = LogisticRegression(C=1000.0, max_iter=3000, random_state=seed)
    calibrator.fit(raw.reshape(-1, 1), labels[calibration])
    payload: dict[str, object] = {
        "model_version": MODEL_VERSION,
        "model_type": "random_forest",
        "feature_names": list(PITCH_VISUAL_FEATURE_NAMES),
        "trees": serialize_probability_forest(model),
        "calibration_intercept": float(calibrator.intercept_[0]),
        "calibration_slope": float(calibrator.coef_[0, 0]),
        "training_seed": seed,
        "model_config": dict(config),
        "target": "paired inclusive and staff-removed notehead evidence supports a local staff-position pitch transaction",
        "scope": "veto-only guard after exact MusicXML preview and independent-family pitch majority",
    }
    return model, calibrator, payload


def _probabilities(
    model: RandomForestClassifier,
    calibrator: LogisticRegression,
    values: np.ndarray,
) -> np.ndarray:
    raw = model.predict_proba(values)[:, 1]
    return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


def _sample(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, probabilities >= 0.5)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }


def _policy(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    accepted = probabilities >= threshold
    true_accepts = int(np.sum(accepted & (labels == 1)))
    false_accepts = int(np.sum(accepted & (labels == 0)))
    return {
        "threshold": float(threshold),
        "accepted": int(np.sum(accepted)),
        "true_accepts": true_accepts,
        "false_accepts": false_accepts,
        "precision": true_accepts / max(true_accepts + false_accepts, 1),
        "coverage": float(np.mean(accepted)),
        "positive_recall": true_accepts / max(int(np.sum(labels == 1)), 1),
    }


def _threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, float | int]]:
    negative = probabilities[labels == 0]
    if not len(negative):
        return 1.0, _policy(labels, probabilities, 1.0)
    # Strictly exceed every observed regression probability in the dedicated threshold
    # partition.  The frozen and independent sets remain untouched; any later false
    # acceptance blocks model publication rather than being hidden by retuning.
    selected = min(1.0, float(np.max(negative)) + 1e-12)
    return selected, _policy(labels, probabilities, selected)


def _scenario_metrics(
    scenarios: tuple[str, ...],
    indices: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for scenario in sorted(set(scenarios[int(index)] for index in indices)):
        local = np.asarray(
            [offset for offset, index in enumerate(indices) if scenarios[int(index)] == scenario],
            dtype=np.int64,
        )
        result[scenario] = _policy(labels[local], probabilities[local], threshold)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "src" / "scorescan" / "resources" / "pitch_visual_guard.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT.parent / "training" / "pitch_visual_guard_report_v2.json",
    )
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--groups", type=int, default=1600)
    parser.add_argument("--confirmation-groups", type=int, default=600)
    args = parser.parse_args()

    dataset = build_rendered_pitch_dataset(args.seed, args.groups)
    features = dataset.features[:, PITCH_VISUAL_FEATURE_INDICES]
    train, calibration, audit, threshold_indices, test = _split(dataset.groups, args.seed)

    trained: list[tuple[RandomForestClassifier, LogisticRegression, dict[str, object]]] = []
    selection: list[dict[str, object]] = []
    for config in MODEL_CONFIGS:
        model, calibrator, payload = _fit(
            features, dataset.labels, train, calibration, args.seed, config
        )
        probabilities = _probabilities(model, calibrator, features[audit])
        selection.append({"config": dict(config), "sample": _sample(dataset.labels[audit], probabilities)})
        trained.append((model, calibrator, payload))
    selected_index = min(
        range(len(selection)),
        key=lambda index: (
            -float(selection[index]["sample"]["roc_auc"]),
            float(selection[index]["sample"]["log_loss"]),
            int(selection[index]["config"]["n_estimators"]),
        ),
    )
    model, calibrator, payload = trained[selected_index]
    selected_config = dict(selection[selected_index]["config"])
    threshold_probabilities = _probabilities(model, calibrator, features[threshold_indices])
    threshold, threshold_policy = _threshold(
        dataset.labels[threshold_indices], threshold_probabilities
    )
    payload.update({
        "training_groups": args.groups,
        "selected_config": selected_config,
        "selected_on": "independent group-isolated rendered audit",
        "auto_patch_threshold": threshold,
        "target_precision": 1.0,
    })

    test_probabilities = _probabilities(model, calibrator, features[test])
    deployed = deployed_forest_probabilities(payload, features[test])
    deployment_delta = float(np.max(np.abs(test_probabilities - deployed), initial=0.0))
    if deployment_delta > 1e-10:
        raise RuntimeError(f"deployment prediction mismatch: {deployment_delta}")

    confirmation = build_rendered_pitch_dataset(args.seed + 4, args.confirmation_groups)
    confirmation_features = confirmation.features[:, PITCH_VISUAL_FEATURE_INDICES]
    confirmation_probabilities = _probabilities(model, calibrator, confirmation_features)
    frozen_policy = _policy(dataset.labels[test], test_probabilities, threshold)
    confirmation_policy = _policy(
        confirmation.labels, confirmation_probabilities, threshold
    )
    if int(frozen_policy["false_accepts"]) or int(confirmation_policy["false_accepts"]):
        raise RuntimeError(
            "pitch visual guard failed zero-error-accept publication gate: "
            f"frozen={frozen_policy['false_accepts']}, "
            f"confirmation={confirmation_policy['false_accepts']}"
        )

    atomic_write_json(args.output, payload)
    report = {
        "model_version": MODEL_VERSION,
        "seed": args.seed,
        "groups": args.groups,
        "samples": len(dataset.labels),
        "scope": "rendered pitch transaction safety only; not end-to-end OMR accuracy",
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
                dataset.scenarios,
                test,
                dataset.labels[test],
                test_probabilities,
                threshold,
            ),
            "accept_all": _policy(
                dataset.labels[test], np.ones(len(test), dtype=np.float64), 0.5
            ),
        },
        "independent_confirmation": {
            "seed": args.seed + 4,
            "groups": args.confirmation_groups,
            "sample": _sample(confirmation.labels, confirmation_probabilities),
            "policy": confirmation_policy,
            "scenarios": _scenario_metrics(
                confirmation.scenarios,
                np.arange(len(confirmation.labels), dtype=np.int64),
                confirmation.labels,
                confirmation_probabilities,
                threshold,
            ),
        },
        "deployment_parity": {"max_absolute_probability_delta": deployment_delta},
        "feature_names": list(PITCH_VISUAL_FEATURE_NAMES),
        "model_bytes": args.output.stat().st_size,
    }
    atomic_write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
