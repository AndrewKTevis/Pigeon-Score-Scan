from __future__ import annotations

"""Train the CPU veto for conservative cross-measure tie repairs."""

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

APP_ROOT = Path(__file__).resolve().parents[1]
ROOT = APP_ROOT
sys.path.insert(0, str(APP_ROOT / "src"))

from scorescan.cross_tie_consensus import FEATURE_NAMES, CrossTiePatchInput  # noqa: E402
from scorescan.policy import DEFAULT_POLICY  # noqa: E402
from scorescan.tree_model import VerifiedRandomForestModel  # noqa: E402
from scorescan.util import atomic_write_json, sha256_file  # noqa: E402
from tree_export import deployed_forest_probabilities, serialize_probability_forest  # noqa: E402

MODEL_VERSION = "scorescan-cross-tie-patch-forest-2"
TARGET_PRECISION = 0.9995
SAMPLES_PER_GROUP = 3
MODEL_CONFIGS = (
    {"n_estimators": 48, "max_depth": 8, "min_samples_leaf": 10},
    {"n_estimators": 64, "max_depth": 9, "min_samples_leaf": 8},
    {"n_estimators": 80, "max_depth": 10, "min_samples_leaf": 7},
)
SCENARIO_WEIGHTS = {
    "missing-boundary-strong-support": 0.13,
    "spurious-boundary-strong-support": 0.12,
    "continuation-chain": 0.07,
    "context-supported-neutral": 0.07,
    "low-visual-valid": 0.06,
    "template-near-tie-valid": 0.05,
    "common-mode-false-boundary": 0.10,
    "high-confidence-common-error": 0.09,
    "weak-family-margin": 0.07,
    "alignment-weak": 0.06,
    "visual-context-double-conflict": 0.07,
    "template-quality-superior": 0.06,
    "incomplete-family-trap": 0.05,
}
POSITIVE_SCENARIOS = {
    "missing-boundary-strong-support",
    "spurious-boundary-strong-support",
    "continuation-chain",
    "context-supported-neutral",
    "low-visual-valid",
    "template-near-tie-valid",
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


def _row(seed: int, group_id: int, sample_id: int) -> tuple[CrossTiePatchInput, int, str]:
    group_seed = (seed << 23) ^ (group_id * 0x9E3779B1)
    scenario = _choose_scenario(random.Random(group_seed))
    rng = random.Random(group_seed ^ ((sample_id + 1) * 0x85EBCA77))
    positive = scenario in POSITIVE_SCENARIOS

    candidate_count = rng.randint(4, 7)
    eligible_families = rng.choice((3, 4))
    voting_families = eligible_families
    incomplete = 0 if rng.random() < 0.88 else 1
    winner = 4 if eligible_families == 4 and rng.random() < (0.52 if positive else 0.18) else 3
    winner = min(winner, voting_families)
    runner_up = max(0, voting_families - winner)
    template_support = runner_up
    winner_present = rng.random() < 0.56
    changed_endpoints = 2 if rng.random() < 0.92 else 1
    added = changed_endpoints if winner_present else 0
    removed = changed_endpoints - added
    left_duration = rng.choice((Fraction(1, 4), Fraction(1, 2), Fraction(1, 1), Fraction(2, 1)))
    right_duration = rng.choice((Fraction(1, 4), Fraction(1, 2), Fraction(1, 1), Fraction(2, 1)))

    quality = 0.92 if positive else 0.66
    visual_centre = 0.87 if positive else 0.57
    context_centre = 0.92 if positive else 0.55
    event_centre = quality + (0.05 if positive else 0.0)
    alignment = 0.96 if positive else 0.76
    ensemble_delta = rng.uniform(0.12, 0.31) if positive else rng.uniform(-0.20, 0.08)
    score_margin = rng.uniform(7.0, 40.0) if positive else rng.uniform(-38.0, 12.0)

    if scenario == "spurious-boundary-strong-support":
        winner_present = False
        added, removed = 0, changed_endpoints
    elif scenario == "continuation-chain":
        winner_present = True
        quality = 0.89
        context_centre = 0.96
        left_duration = rng.choice((Fraction(1, 2), Fraction(1, 1)))
        right_duration = rng.choice((Fraction(1, 2), Fraction(1, 1)))
    elif scenario == "context-supported-neutral":
        visual_centre = 0.69
        context_centre = 0.98
        quality = 0.86
    elif scenario == "low-visual-valid":
        visual_centre = 0.42
        context_centre = 0.97
        quality = 0.90
        ensemble_delta = rng.uniform(0.08, 0.24)
        score_margin = rng.uniform(2.0, 24.0)
    elif scenario == "template-near-tie-valid":
        visual_centre = 0.78
        context_centre = 0.91
        quality = 0.90
        ensemble_delta = rng.uniform(0.015, 0.10)
        score_margin = rng.uniform(-2.0, 12.0)
    elif scenario == "common-mode-false-boundary":
        quality = 0.83
        visual_centre = 0.39
        context_centre = 0.43
        ensemble_delta = rng.uniform(-0.08, 0.06)
    elif scenario == "high-confidence-common-error":
        quality = 0.91
        visual_centre = 0.46
        context_centre = 0.48
        alignment = 0.94
        ensemble_delta = rng.uniform(-0.05, 0.05)
        score_margin = rng.uniform(-4.0, 18.0)
        winner = min(voting_families, 3)
        runner_up = max(0, voting_families - winner)
        template_support = runner_up
    elif scenario == "weak-family-margin":
        winner = 3
        runner_up = 1 if voting_families == 4 else 0
        template_support = runner_up
        quality = 0.69
        visual_centre = 0.59
        context_centre = 0.58
        ensemble_delta = rng.uniform(-0.11, 0.06)
    elif scenario == "alignment-weak":
        quality = 0.80
        alignment = rng.uniform(0.42, 0.68)
        context_centre = 0.52
        ensemble_delta = rng.uniform(-0.15, 0.04)
        incomplete = max(incomplete, 1)
    elif scenario == "visual-context-double-conflict":
        quality = 0.86
        visual_centre = 0.18
        context_centre = 0.20
        event_centre = 0.52
        ensemble_delta = rng.uniform(-0.10, 0.04)
        score_margin = rng.uniform(-12.0, 8.0)
    elif scenario == "template-quality-superior":
        quality = 0.72
        visual_centre = 0.66
        context_centre = 0.67
        alignment = 0.88
        ensemble_delta = rng.uniform(-0.27, -0.04)
        score_margin = rng.uniform(-45.0, -4.0)
    elif scenario == "incomplete-family-trap":
        quality = 0.89
        visual_centre = 0.70
        context_centre = 0.70
        event_centre = 0.75
        incomplete = max(incomplete, 1)
        winner = min(voting_families, 3)
        runner_up = max(0, voting_families - winner)
        template_support = runner_up
        ensemble_delta = rng.uniform(-0.06, 0.07)
        score_margin = rng.uniform(-8.0, 15.0)

    page = _quality(rng, quality)
    measure = _quality(rng, quality + (0.03 if positive else 0.0))
    visual = _quality(rng, visual_centre, 0.09)
    event = _quality(rng, event_centre)
    context = _quality(rng, context_centre, 0.09)
    ensemble = _quality(rng, quality + (0.06 if positive else 0.0))
    minimum_ensemble = _clip(ensemble - rng.uniform(0.02, 0.20))
    alignment = _quality(rng, alignment, 0.05)

    if rng.random() < 0.22:
        visual = _clip(visual + rng.gauss(0.0, 0.12))
        context = _clip(context + rng.gauss(0.0, 0.12))
        alignment = _clip(alignment + rng.gauss(0.0, 0.08))
        ensemble_delta = _clip(ensemble_delta + rng.gauss(0.0, 0.07), -1.0, 1.0)
        score_margin += rng.gauss(0.0, 8.0)

    item = CrossTiePatchInput(
        candidate_count=candidate_count,
        eligible_family_count=eligible_families,
        voting_family_count=voting_families,
        winner_family_count=winner,
        runner_up_family_count=runner_up,
        template_family_count=template_support,
        incomplete_family_count=incomplete,
        winner_boundary_present=winner_present,
        changed_endpoint_count=changed_endpoints,
        added_endpoint_count=added,
        removed_endpoint_count=removed,
        left_boundary_duration=left_duration,
        right_boundary_duration=right_duration,
        mean_support_page_probability=page,
        mean_support_measure_probability=measure,
        mean_support_visual_probability=visual,
        mean_support_event_probability=event,
        mean_support_context_probability=context,
        mean_support_ensemble_probability=ensemble,
        minimum_support_ensemble_probability=minimum_ensemble,
        mean_support_alignment_similarity=alignment,
        mean_support_page_score_margin=score_margin,
        mean_support_vs_template_measure_probability=ensemble_delta + rng.gauss(0.0, 0.04),
        mean_support_vs_template_visual_probability=ensemble_delta + rng.gauss(0.0, 0.05),
        mean_support_vs_template_event_probability=ensemble_delta + rng.gauss(0.0, 0.04),
        mean_support_vs_template_context_probability=ensemble_delta + rng.gauss(0.0, 0.05),
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
        encoded = scenario.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
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
        n_jobs=1,
        class_weight="balanced_subsample",
        max_features="sqrt",
    )
    model.fit(dataset.features[train], dataset.labels[train])
    raw = model.predict_proba(dataset.features[calibration])[:, 1]
    calibrator = LogisticRegression(C=1000.0, max_iter=3000, random_state=seed)
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
        "target": "cross-measure tie endpoint repair improves one fixed adjacent boundary",
        "scope": "aligned monophonic complete-measure boundaries already passing independent-family and deterministic guards",
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
    candidates.extend([0.97, 0.98, 0.99, 0.995, 0.999, 0.9995])
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


def _zero_false_accept_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    minimum: float,
) -> tuple[float, dict[str, float | int]]:
    negatives = probabilities[labels == 0]
    threshold = float(minimum)
    if len(negatives):
        maximum_negative = float(np.max(negatives))
        threshold = max(threshold, float(np.nextafter(maximum_negative, 1.0)))
    threshold = min(1.0, threshold)
    metrics = _metrics(labels, probabilities, threshold)
    if int(metrics["false_accepts"]) != 0:
        raise RuntimeError("safety threshold still accepts a negative decision")
    return threshold, metrics


def _baseline_probabilities(features: np.ndarray) -> np.ndarray:
    path = APP_ROOT.parent / "training/baselines/cross_tie_patch_calibrator_v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = VerifiedRandomForestModel.from_payload(
        payload, FEATURE_NAMES, verified=True, status="baseline"
    )
    if not model.enabled:
        raise RuntimeError("cross-tie v1 baseline model is unavailable")
    return np.asarray([model.predict(row) for row in features], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=APP_ROOT / "src/scorescan/resources/cross_tie_patch_calibrator.json")
    parser.add_argument("--report", type=Path, default=APP_ROOT.parent / "training/cross_tie_patch_calibrator_report_v2.json")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--groups", type=int, default=9000)
    parser.add_argument("--safety-groups", type=int, default=4000)
    parser.add_argument("--confirmation-groups", type=int, default=3000)
    parser.add_argument("--secondary-confirmation-groups", type=int, default=3000)
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
    best_index = min(
        range(len(audit_rows)),
        key=lambda index: (
            -float(audit_rows[index]["sample"]["roc_auc"]),
            float(audit_rows[index]["sample"]["log_loss"]),
            int(audit_rows[index]["config"]["n_estimators"]),
        ),
    )
    model, calibrator, payload = trained[best_index]
    selected_config = dict(audit_rows[best_index]["config"])
    threshold_probabilities = _probabilities(model, calibrator, dataset.features[threshold_indices])
    raw_threshold, raw_threshold_metrics = _select_threshold(dataset.labels[threshold_indices], threshold_probabilities)
    initial_threshold = max(float(DEFAULT_POLICY.cross_tie_patch_probability_floor), raw_threshold)
    threshold_metrics = _metrics(dataset.labels[threshold_indices], threshold_probabilities, initial_threshold)

    safety = build_dataset(args.seed + 1, args.safety_groups)
    safety_probabilities = _probabilities(model, calibrator, safety.features)
    threshold, safety_metrics = _zero_false_accept_threshold(
        safety.labels, safety_probabilities, initial_threshold
    )
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
    secondary_confirmation = build_dataset(
        args.seed + 3, args.secondary_confirmation_groups
    )
    secondary_confirmation_probabilities = _probabilities(
        model, calibrator, secondary_confirmation.features
    )
    baseline_test = _metrics(dataset.labels[test], np.ones(len(test), dtype=np.float64), 0.5)
    baseline_confirmation = _metrics(confirmation.labels, np.ones(len(confirmation.labels), dtype=np.float64), 0.5)
    baseline_secondary_confirmation = _metrics(
        secondary_confirmation.labels,
        np.ones(len(secondary_confirmation.labels), dtype=np.float64),
        0.5,
    )
    v1_test_probabilities = _baseline_probabilities(dataset.features[test])
    v1_confirmation_probabilities = _baseline_probabilities(confirmation.features)
    v1_secondary_confirmation_probabilities = _baseline_probabilities(
        secondary_confirmation.features
    )
    v1_threshold = float(DEFAULT_POLICY.cross_tie_patch_probability_floor)

    atomic_write_json(args.output, payload)
    report = {
        "model_version": MODEL_VERSION,
        "seed": args.seed,
        "groups": args.groups,
        "samples_per_group": SAMPLES_PER_GROUP,
        "samples": len(dataset.labels),
        "safety_groups": args.safety_groups,
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
        "policy_probability_floor": DEFAULT_POLICY.cross_tie_patch_probability_floor,
        "threshold_selection": {
            "initial_policy": threshold_metrics,
            "raw_precision_threshold": raw_threshold_metrics,
        },
        "safety_calibration": {
            "seed": args.seed + 1,
            "groups": args.safety_groups,
            "samples": len(safety.labels),
            "dataset_fingerprint": _dataset_fingerprint(safety),
            "policy": safety_metrics,
        },
        "selected_threshold": _metrics(dataset.labels[threshold_indices], threshold_probabilities, threshold),
        "frozen_test": {
            "sample": _sample_metrics(dataset.labels[test], test_probabilities),
            "policy": _metrics(dataset.labels[test], test_probabilities, threshold),
            "accept_all_deterministic_proposals": baseline_test,
            "scenarios": _scenario_metrics(dataset, test, test_probabilities, threshold),
            "v1_same_data_policy": _metrics(dataset.labels[test], v1_test_probabilities, v1_threshold),
            "v1_same_data_scenarios": _scenario_metrics(dataset, test, v1_test_probabilities, v1_threshold),
        },
        "independent_confirmation": {
            "seed": args.seed + 2,
            "groups": args.confirmation_groups,
            "samples": len(confirmation.labels),
            "dataset_fingerprint": _dataset_fingerprint(confirmation),
            "policy": _metrics(confirmation.labels, confirmation_probabilities, threshold),
            "accept_all_deterministic_proposals": baseline_confirmation,
            "v1_same_data_policy": _metrics(confirmation.labels, v1_confirmation_probabilities, v1_threshold),
        },
        "secondary_confirmation": {
            "seed": args.seed + 3,
            "groups": args.secondary_confirmation_groups,
            "samples": len(secondary_confirmation.labels),
            "dataset_fingerprint": _dataset_fingerprint(secondary_confirmation),
            "policy": _metrics(
                secondary_confirmation.labels,
                secondary_confirmation_probabilities,
                threshold,
            ),
            "accept_all_deterministic_proposals": baseline_secondary_confirmation,
            "v1_same_data_policy": _metrics(
                secondary_confirmation.labels,
                v1_secondary_confirmation_probabilities,
                v1_threshold,
            ),
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
