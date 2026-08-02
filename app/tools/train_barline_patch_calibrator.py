from __future__ import annotations

"""Train the CPU veto for conservative repeat/barline repairs."""

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
sys.path.insert(0, str(ROOT / "src"))

from scorescan.barline_consensus import BarlinePatchInput, FEATURE_NAMES  # noqa: E402
from scorescan.policy import DEFAULT_POLICY  # noqa: E402
from scorescan.util import atomic_write_json, sha256_file  # noqa: E402
from tree_export import deployed_forest_probabilities, serialize_probability_forest  # noqa: E402

MODEL_VERSION = "scorescan-barline-patch-forest-1"
TARGET_PRECISION = 0.9995
SAMPLES_PER_GROUP = 3
MODEL_CONFIGS = (
    {"n_estimators": 64, "max_depth": 8, "min_samples_leaf": 8},
    {"n_estimators": 80, "max_depth": 9, "min_samples_leaf": 7},
    {"n_estimators": 96, "max_depth": 10, "min_samples_leaf": 6},
)
SCENARIO_WEIGHTS = {
    "true-repeat-strong-support": 0.16,
    "remove-spurious-repeat": 0.13,
    "terminal-style-correction": 0.10,
    "double-boundary-correction": 0.07,
    "context-supported-repeat": 0.08,
    "common-mode-false-repeat": 0.13,
    "weak-family-margin": 0.08,
    "visual-conflict": 0.07,
    "context-conflict": 0.06,
    "template-quality-superior": 0.06,
    "complex-navigation-conflict": 0.06,
}
POSITIVE_SCENARIOS = {
    "true-repeat-strong-support",
    "remove-spurious-repeat",
    "terminal-style-correction",
    "double-boundary-correction",
    "context-supported-repeat",
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


def _row(seed: int, group_id: int, sample_id: int) -> tuple[BarlinePatchInput, int, str]:
    group_seed = (seed << 23) ^ (group_id * 0x9E3779B1)
    scenario = _choose_scenario(random.Random(group_seed))
    rng = random.Random(group_seed ^ ((sample_id + 1) * 0x85EBCA77))
    positive = scenario in POSITIVE_SCENARIOS

    candidate_count = rng.randint(4, 7)
    eligible_families = rng.choice((3, 4))
    voting_families = eligible_families
    incomplete = 0 if rng.random() < 0.90 else 1
    winner = 4 if eligible_families == 4 and rng.random() < (0.55 if positive else 0.15) else 3
    winner = min(winner, voting_families)
    runner_up = max(0, voting_families - winner)
    template_support = runner_up
    changed_locations = 1
    added = 1 if rng.random() < 0.46 else 0
    removed = 0 if added else (1 if rng.random() < 0.38 else 0)
    repeat_changes = 1
    style_changes = 1 if rng.random() < 0.52 else 0
    winner_barlines = rng.choice((1, 1, 2))

    quality = 0.93 if positive else 0.64
    visual_centre = 0.88 if positive else 0.52
    context_centre = 0.91 if positive else 0.52
    # The complete-page template is often the highest-scoring candidate even when
    # independent lower-ranked variants agree on the repeat sign.  Positive examples
    # therefore include neutral and mildly negative template-relative margins; visual,
    # context and family evidence must carry the decision rather than page rank alone.
    ensemble_delta = rng.uniform(-0.04, 0.20) if positive else rng.uniform(-0.22, 0.06)
    score_margin = rng.uniform(-14.0, 32.0) if positive else rng.uniform(-42.0, 10.0)

    if scenario == "remove-spurious-repeat":
        added, removed = 0, 1
        style_changes = rng.choice((0, 1))
    elif scenario == "terminal-style-correction":
        repeat_changes = 0
        style_changes = 1
        added = removed = 0
        quality = 0.90
    elif scenario == "double-boundary-correction":
        changed_locations = 2
        repeat_changes = rng.choice((1, 2))
        style_changes = rng.choice((1, 2))
        winner_barlines = 2
        quality = 0.88
    elif scenario == "context-supported-repeat":
        visual_centre = 0.70
        context_centre = 0.98
        quality = 0.86
    elif scenario == "common-mode-false-repeat":
        quality = 0.83
        visual_centre = 0.30
        context_centre = 0.38
        ensemble_delta = rng.uniform(-0.11, 0.04)
    elif scenario == "weak-family-margin":
        winner = 3
        runner_up = 1 if voting_families == 4 else 0
        template_support = runner_up
        quality = 0.67
        visual_centre = 0.56
        context_centre = 0.57
        ensemble_delta = rng.uniform(-0.13, 0.05)
    elif scenario == "visual-conflict":
        quality = 0.80
        visual_centre = 0.10
        context_centre = 0.79
        ensemble_delta = rng.uniform(-0.14, 0.02)
    elif scenario == "context-conflict":
        quality = 0.80
        visual_centre = 0.78
        context_centre = 0.10
        ensemble_delta = rng.uniform(-0.14, 0.02)
    elif scenario == "template-quality-superior":
        quality = 0.72
        visual_centre = 0.63
        context_centre = 0.65
        ensemble_delta = rng.uniform(-0.30, -0.04)
        score_margin = rng.uniform(-48.0, -5.0)
    elif scenario == "complex-navigation-conflict":
        changed_locations = 2
        repeat_changes = 2
        style_changes = 2
        winner_barlines = 2
        incomplete = max(incomplete, 1)
        quality = 0.50
        visual_centre = 0.37
        context_centre = 0.37
        ensemble_delta = rng.uniform(-0.24, 0.0)

    page = _quality(rng, quality)
    measure = _quality(rng, quality + (0.03 if positive else 0.0))
    visual = _quality(rng, visual_centre, 0.09)
    event = _quality(rng, quality + (0.04 if positive else 0.0))
    context = _quality(rng, context_centre, 0.09)
    ensemble = _quality(rng, quality + (0.06 if positive else 0.0))
    minimum_ensemble = _clip(ensemble - rng.uniform(0.02, 0.20))
    if rng.random() < 0.25:
        visual = _clip(visual + rng.gauss(0.0, 0.12))
        context = _clip(context + rng.gauss(0.0, 0.12))
        ensemble_delta = _clip(ensemble_delta + rng.gauss(0.0, 0.07), -1.0, 1.0)
        score_margin += rng.gauss(0.0, 8.0)

    item = BarlinePatchInput(
        candidate_count=candidate_count,
        eligible_family_count=eligible_families,
        voting_family_count=voting_families,
        changed_location_count=changed_locations,
        added_barline_count=added,
        removed_barline_count=removed,
        repeat_change_count=repeat_changes,
        style_change_count=style_changes,
        winner_family_count=winner,
        runner_up_family_count=runner_up,
        template_family_count=template_support,
        incomplete_family_count=incomplete,
        winner_barline_count=winner_barlines,
        mean_support_page_probability=page,
        mean_support_measure_probability=measure,
        mean_support_visual_probability=visual,
        mean_support_event_probability=event,
        mean_support_context_probability=context,
        mean_support_ensemble_probability=ensemble,
        minimum_support_ensemble_probability=minimum_ensemble,
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
        "target": "simple repeat/barline repair improves an aligned measure without changing musical events",
        "scope": "left/right bar styles and forward/backward repeats already passing independent-family and deterministic XML guards",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "src/scorescan/resources/barline_patch_calibrator.json")
    parser.add_argument("--report", type=Path, default=ROOT.parent / "training/barline_patch_calibrator_report_v1.json")
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
    # Prefer the smallest forest inside a narrow near-optimal audit envelope.
    # A microscopic AUC gain is not worth a materially larger bundled resource on
    # ordinary Windows machines. The final threshold must still pass both the
    # frozen and independently seeded zero-harm gates below.
    best_auc = max(float(row["sample"]["roc_auc"]) for row in audit_rows)
    best_log_loss = min(float(row["sample"]["log_loss"]) for row in audit_rows)
    eligible = [
        index
        for index, row in enumerate(audit_rows)
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
    threshold = max(float(DEFAULT_POLICY.barline_patch_probability_floor), raw_threshold)
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
        raise RuntimeError("release threshold produced a harmful automatic barline repair")
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
        "policy_probability_floor": DEFAULT_POLICY.barline_patch_probability_floor,
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
