from __future__ import annotations

"""Train the veto-only CPU calibrator for simple grace-note repair."""

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scorescan.grace_consensus import GracePatchInput, FEATURE_NAMES  # noqa: E402
from scorescan.policy import DEFAULT_POLICY  # noqa: E402
from scorescan.util import atomic_write_json, sha256_file  # noqa: E402
from tree_export import deployed_forest_probabilities, serialize_probability_forest  # noqa: E402

MODEL_VERSION = "scorescan-grace-patch-forest-1"
TARGET_PRECISION = 0.9995
SAMPLES_PER_GROUP = 3
MODEL_CONFIGS = (
    {"n_estimators": 64, "max_depth": 8, "min_samples_leaf": 8},
    {"n_estimators": 80, "max_depth": 9, "min_samples_leaf": 7},
    {"n_estimators": 96, "max_depth": 10, "min_samples_leaf": 6},
)
SCENARIO_WEIGHTS = {
    "remove-false-regular": 0.16,
    "restore-missing-regular": 0.15,
    "two-grace-corrections": 0.08,
    "context-supported": 0.08,
    "visual-supported": 0.08,
    "weak-family-margin": 0.08,
    "content-family-split": 0.08,
    "template-quality-superior": 0.08,
    "poor-meter-improvement": 0.08,
    "incomplete-family": 0.07,
    "high-change-burden": 0.07,
    "common-mode-grace-confusion": 0.07,
}
POSITIVE_SCENARIOS = {
    "remove-false-regular",
    "restore-missing-regular",
    "two-grace-corrections",
    "context-supported",
    "visual-supported",
}


@dataclass(frozen=True)
class Dataset:
    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    scenarios: tuple[str, ...]


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _quality(rng: random.Random, centre: float, spread: float = 0.07) -> float:
    return _clip(rng.gauss(centre, spread), 0.01, 0.995)


def _choose_scenario(rng: random.Random) -> str:
    point = rng.random()
    total = 0.0
    for name, weight in SCENARIO_WEIGHTS.items():
        total += weight
        if point <= total:
            return name
    return next(reversed(SCENARIO_WEIGHTS))


def _row(seed: int, group_id: int, sample_id: int) -> tuple[GracePatchInput, int, str]:
    group_seed = (seed << 23) ^ (group_id * 0x9E3779B1)
    scenario = _choose_scenario(random.Random(group_seed))
    rng = random.Random(group_seed ^ ((sample_id + 1) * 0x85EBCA77))
    positive = scenario in POSITIVE_SCENARIOS

    candidate_count = rng.randint(4, 7)
    eligible = rng.choice((3, 4))
    voting = eligible
    changed = 1
    total_events = rng.randint(4, 16)
    added = 1
    removed = 0
    winner = 4 if eligible == 4 and rng.random() < (0.55 if positive else 0.12) else 3
    winner = min(winner, voting)
    runner_up = max(0, voting - winner)
    template = runner_up
    incomplete = 0
    content_min = winner
    content_mean = float(winner) - rng.uniform(0.0, 0.12)
    content_margin = max(1, winner - runner_up)
    base_error = rng.choice((0.25, 0.5, 1.0))
    patched_error = 0.0
    quality = 0.93 if positive else 0.64
    visual_centre = 0.88 if positive else 0.52
    context_centre = 0.88 if positive else 0.53
    ensemble_delta = rng.uniform(-0.03, 0.20) if positive else rng.uniform(-0.28, 0.04)
    score_margin = rng.uniform(-10.0, 32.0) if positive else rng.uniform(-48.0, 8.0)

    if scenario == "restore-missing-regular":
        added, removed = 0, 1
    elif scenario == "two-grace-corrections":
        changed = 2
        added = rng.choice((0, 1, 2))
        removed = changed - added
        total_events = rng.randint(8, 18)
        base_error = rng.choice((0.5, 1.0, 1.5))
        quality = 0.89
    elif scenario == "context-supported":
        visual_centre = 0.65
        context_centre = 0.98
        quality = 0.86
    elif scenario == "visual-supported":
        visual_centre = 0.98
        context_centre = 0.66
        quality = 0.87
    elif scenario == "weak-family-margin":
        winner = 3
        runner_up = 1 if voting == 4 else 0
        template = runner_up
        content_min = winner
        content_mean = float(winner) - 0.2
        content_margin = max(0, winner - runner_up)
        quality = 0.60
        visual_centre = 0.54
        context_centre = 0.55
    elif scenario == "content-family-split":
        content_min = 2
        content_mean = rng.uniform(2.0, 2.5)
        content_margin = 1
        incomplete = 1
        quality = 0.66
    elif scenario == "template-quality-superior":
        quality = 0.72
        ensemble_delta = rng.uniform(-0.36, -0.05)
        score_margin = rng.uniform(-55.0, -7.0)
        visual_centre = 0.62
        context_centre = 0.61
    elif scenario == "poor-meter-improvement":
        base_error = rng.uniform(0.02, 0.12)
        patched_error = rng.uniform(0.01, base_error)
        quality = 0.82
        ensemble_delta = rng.uniform(-0.10, 0.05)
    elif scenario == "incomplete-family":
        incomplete = rng.choice((1, 2))
        voting = max(3, eligible - 1)
        winner = min(3, voting)
        runner_up = max(0, voting - winner)
        template = runner_up
        content_min = winner
        quality = 0.67
    elif scenario == "high-change-burden":
        changed = rng.randint(3, 5)
        total_events = rng.randint(6, 10)
        added = rng.randint(1, changed)
        removed = changed - added
        base_error = rng.uniform(1.0, 3.0)
        quality = 0.60
        visual_centre = 0.43
        context_centre = 0.45
    elif scenario == "common-mode-grace-confusion":
        quality = 0.79
        visual_centre = 0.20
        context_centre = 0.36
        ensemble_delta = rng.uniform(-0.18, 0.01)

    page = _quality(rng, quality)
    measure = _quality(rng, quality + (0.02 if positive else 0.0))
    visual = _quality(rng, visual_centre, 0.09)
    event = _quality(rng, quality + (0.05 if positive else 0.0))
    context = _quality(rng, context_centre, 0.09)
    ensemble = _quality(rng, quality + (0.06 if positive else 0.0))
    minimum_ensemble = _clip(ensemble - rng.uniform(0.02, 0.18))

    item = GracePatchInput(
        candidate_count=candidate_count,
        eligible_family_count=eligible,
        voting_family_count=voting,
        changed_event_count=changed,
        total_event_count=total_events,
        added_grace_count=added,
        removed_grace_count=removed,
        winner_family_count=winner,
        winner_margin_count=max(0, winner - runner_up),
        template_family_count=template,
        incomplete_family_count=incomplete,
        minimum_content_family_count=content_min,
        mean_content_family_count=content_mean,
        minimum_content_margin_count=content_margin,
        base_duration_error=base_error,
        patched_duration_error=patched_error,
        duration_error_improvement=base_error - patched_error,
        mean_support_page_probability=page,
        mean_support_measure_probability=measure,
        mean_support_visual_probability=visual,
        mean_support_event_probability=event,
        mean_support_context_probability=context,
        mean_support_ensemble_probability=ensemble,
        minimum_support_ensemble_probability=minimum_ensemble,
        mean_support_page_score_margin=score_margin,
        mean_support_vs_template_ensemble_probability=ensemble_delta,
    )
    return item, int(positive), scenario


def build_dataset(seed: int, groups: int) -> Dataset:
    rows: list[list[float]] = []
    labels: list[int] = []
    group_ids: list[int] = []
    scenarios: list[str] = []
    for group_id in range(groups):
        for sample_id in range(SAMPLES_PER_GROUP):
            item, label, scenario = _row(seed, group_id, sample_id)
            rows.append(item.feature_vector())
            labels.append(label)
            group_ids.append(group_id)
            scenarios.append(scenario)
    return Dataset(
        np.asarray(rows, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(group_ids, dtype=np.int64),
        tuple(scenarios),
    )


def _dataset_fingerprint(dataset: Dataset) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(dataset.features, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(dataset.labels, dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(dataset.groups, dtype="<i8").tobytes(order="C"))
    for scenario in dataset.scenarios:
        value = scenario.encode("utf-8")
        digest.update(len(value).to_bytes(4, "little"))
        digest.update(value)
    return digest.hexdigest()


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


def _fit(dataset: Dataset, train: np.ndarray, calibration: np.ndarray, seed: int, config: dict[str, int]):
    model = RandomForestClassifier(
        **config,
        random_state=seed,
        class_weight="balanced_subsample",
        n_jobs=1,
    )
    model.fit(dataset.features[train], dataset.labels[train])
    raw = model.predict_proba(dataset.features[calibration])[:, 1]
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=seed)
    calibrator.fit(raw.reshape(-1, 1), dataset.labels[calibration])
    payload: dict[str, object] = {
        "model_version": MODEL_VERSION,
        "model_type": "random_forest",
        "feature_names": list(FEATURE_NAMES),
        "trees": serialize_probability_forest(model),
        "calibration_intercept": float(calibrator.intercept_[0]),
        "calibration_slope": float(calibrator.coef_[0, 0]),
        "training_seed": seed,
        "model_config": dict(config),
        "target": "simple grace-state repair restores exact meter without changing the pitched event sequence",
        "scope": "empty attribute-free grace elements and owned duration/type/dot fields after strict family and XML guards",
    }
    return model, calibrator, payload


def _probabilities(model, calibrator, features: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(features)[:, 1]
    return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    accepted = probabilities >= threshold
    true_accepts = int(np.sum(accepted & (labels == 1)))
    false_accepts = int(np.sum(accepted & (labels == 0)))
    accepts = int(np.sum(accepted))
    return {
        "threshold": float(threshold),
        "accepted": accepts,
        "true_accepts": true_accepts,
        "false_accepts": false_accepts,
        "precision": true_accepts / max(accepts, 1),
        "coverage": accepts / max(len(labels), 1),
        "positive_recall": true_accepts / max(int(np.sum(labels == 1)), 1),
    }


def _sample_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities >= 0.5
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }


def _select_threshold(labels: np.ndarray, probabilities: np.ndarray):
    candidates = sorted(set(float(value) for value in probabilities))
    candidates.extend([0.97, 0.98, 0.99, 0.995, 0.997, 0.999, 0.9995])
    valid = []
    for threshold in sorted(set(candidates)):
        metrics = _metrics(labels, probabilities, threshold)
        if metrics["accepted"] and float(metrics["precision"]) >= TARGET_PRECISION:
            valid.append((threshold, metrics))
    if not valid:
        return 1.0, _metrics(labels, probabilities, 1.0)
    return max(valid, key=lambda item: (float(item[1]["coverage"]), float(item[1]["positive_recall"]), item[0]))


def _scenario_metrics(dataset: Dataset, indices: np.ndarray, probabilities: np.ndarray, threshold: float):
    result = {}
    for scenario in sorted(set(dataset.scenarios[index] for index in indices)):
        local = np.asarray([offset for offset, index in enumerate(indices) if dataset.scenarios[index] == scenario])
        result[scenario] = _metrics(dataset.labels[indices][local], probabilities[local], threshold)
    return result


def _group_leakage_audit(dataset: Dataset, partitions: tuple[np.ndarray, ...]) -> dict[str, object]:
    names = ("train", "calibration", "model_selection_audit", "threshold_selection", "frozen_test")
    sets = {name: set(int(value) for value in dataset.groups[index].tolist()) for name, index in zip(names, partitions, strict=True)}
    overlaps: dict[str, int] = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlaps[f"{left}__{right}"] = len(sets[left] & sets[right])
    return {
        "groups_per_partition": {name: len(sets[name]) for name in names},
        "pairwise_group_overlap": overlaps,
        "leakage_detected": any(overlaps.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "src/scorescan/resources/grace_patch_calibrator.json")
    parser.add_argument("--report", type=Path, default=ROOT.parent / "training/grace_patch_calibrator_report_v1.json")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--groups", type=int, default=9000)
    parser.add_argument("--confirmation-groups", type=int, default=3000)
    args = parser.parse_args()

    dataset = build_dataset(args.seed, args.groups)
    partitions = _split(dataset.groups, args.seed)
    train, calibration, audit, threshold_indices, test = partitions
    leakage_audit = _group_leakage_audit(dataset, partitions)
    if leakage_audit["leakage_detected"]:
        raise RuntimeError("group leakage detected between training partitions")

    trained = []
    audit_rows = []
    for config in MODEL_CONFIGS:
        model, calibrator, payload = _fit(dataset, train, calibration, args.seed, config)
        probabilities = _probabilities(model, calibrator, dataset.features[audit])
        audit_rows.append({"config": dict(config), "sample": _sample_metrics(dataset.labels[audit], probabilities)})
        trained.append((model, calibrator, payload))
    best_auc = max(float(row["sample"]["roc_auc"]) for row in audit_rows)
    best_log_loss = min(float(row["sample"]["log_loss"]) for row in audit_rows)
    eligible = [
        index for index, row in enumerate(audit_rows)
        if best_auc - float(row["sample"]["roc_auc"]) <= 2e-5
        and float(row["sample"]["log_loss"]) <= best_log_loss + 0.002
    ]
    best_index = min(
        eligible,
        key=lambda index: (
            int(audit_rows[index]["config"]["n_estimators"]),
            int(audit_rows[index]["config"]["max_depth"]),
            -int(audit_rows[index]["config"]["min_samples_leaf"]),
            float(audit_rows[index]["sample"]["log_loss"]),
        ),
    )
    model, calibrator, payload = trained[best_index]
    selected_config = dict(audit_rows[best_index]["config"])
    threshold_probabilities = _probabilities(model, calibrator, dataset.features[threshold_indices])
    raw_threshold, raw_threshold_metrics = _select_threshold(dataset.labels[threshold_indices], threshold_probabilities)
    threshold = max(float(DEFAULT_POLICY.grace_patch_probability_floor), raw_threshold)
    threshold_metrics = _metrics(dataset.labels[threshold_indices], threshold_probabilities, threshold)
    payload.update({
        "training_groups": args.groups,
        "samples_per_group": SAMPLES_PER_GROUP,
        "selected_on": "independent grouped model-selection audit",
        "selected_config": selected_config,
        "auto_patch_threshold": threshold,
        "target_precision": TARGET_PRECISION,
    })

    test_probabilities = _probabilities(model, calibrator, dataset.features[test])
    deployed = deployed_forest_probabilities(payload, dataset.features[test])
    deployment_delta = float(np.max(np.abs(test_probabilities - deployed), initial=0.0))
    if deployment_delta > 1e-10:
        raise RuntimeError(f"deployment prediction mismatch: {deployment_delta}")
    confirmation = build_dataset(args.seed + 2, args.confirmation_groups)
    confirmation_probabilities = _probabilities(model, calibrator, confirmation.features)
    frozen_policy = _metrics(dataset.labels[test], test_probabilities, threshold)
    confirmation_policy = _metrics(confirmation.labels, confirmation_probabilities, threshold)
    if int(frozen_policy["false_accepts"]) or int(confirmation_policy["false_accepts"]):
        raise RuntimeError("release threshold produced a harmful automatic grace repair")
    baseline_test = _metrics(dataset.labels[test], np.ones(len(test), dtype=np.float64), 0.5)
    baseline_confirmation = _metrics(confirmation.labels, np.ones(len(confirmation.labels), dtype=np.float64), 0.5)

    atomic_write_json(args.output, payload)
    report = {
        "model_version": MODEL_VERSION,
        "seed": args.seed,
        "groups": args.groups,
        "samples_per_group": SAMPLES_PER_GROUP,
        "samples": len(dataset.labels),
        "dataset_fingerprint": _dataset_fingerprint(dataset),
        "group_leakage_audit": leakage_audit,
        "partitions": {
            "train": len(train),
            "calibration": len(calibration),
            "model_selection_audit": len(audit),
            "threshold_selection": len(threshold_indices),
            "frozen_test": len(test),
        },
        "model_selection": audit_rows,
        "selected_config": selected_config,
        "raw_precision_threshold": raw_threshold_metrics,
        "policy_probability_floor": DEFAULT_POLICY.grace_patch_probability_floor,
        "selected_threshold": threshold_metrics,
        "frozen_test": {
            "sample": _sample_metrics(dataset.labels[test], test_probabilities),
            "policy": frozen_policy,
            "accept_all_deterministic_proposals": baseline_test,
            "scenarios": _scenario_metrics(dataset, test, test_probabilities, threshold),
        },
        "independent_confirmation": {
            "seed": args.seed + 2,
            "groups": args.confirmation_groups,
            "samples": len(confirmation.labels),
            "dataset_fingerprint": _dataset_fingerprint(confirmation),
            "policy": confirmation_policy,
            "accept_all_deterministic_proposals": baseline_confirmation,
        },
        "deployment_parity": {"max_absolute_probability_delta": deployment_delta},
        "feature_names": list(FEATURE_NAMES),
        "model_bytes": args.output.stat().st_size,
        "model_sha256": sha256_file(args.output),
    }
    atomic_write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
