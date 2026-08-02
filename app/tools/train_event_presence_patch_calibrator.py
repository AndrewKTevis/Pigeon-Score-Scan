from __future__ import annotations

"""Train the CPU veto for conservative one-event insertion/deletion repairs."""

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

from scorescan.event_presence_consensus import FEATURE_NAMES, EventPresencePatchInput  # noqa: E402
from scorescan.policy import DEFAULT_POLICY  # noqa: E402
from scorescan.util import atomic_write_json, sha256_file  # noqa: E402
from tree_export import deployed_forest_probabilities, serialize_probability_forest  # noqa: E402

MODEL_VERSION = "scorescan-event-presence-patch-forest-1"
TARGET_PRECISION = 0.9995
SAMPLES_PER_GROUP = 3
MODEL_CONFIGS = (
    {"n_estimators": 64, "max_depth": 8, "min_samples_leaf": 8},
    {"n_estimators": 80, "max_depth": 9, "min_samples_leaf": 7},
    {"n_estimators": 96, "max_depth": 10, "min_samples_leaf": 6},
)
SCENARIO_WEIGHTS = {
    "clear-missing-event": 0.18,
    "clear-extra-event": 0.16,
    "rest-insertion": 0.08,
    "strong-context-recovery": 0.08,
    "common-mode-hallucination": 0.12,
    "meter-boundary-conflict": 0.10,
    "visual-conflict": 0.08,
    "context-conflict": 0.07,
    "template-quality-superior": 0.07,
    "weak-majority": 0.06,
}
POSITIVE_SCENARIOS = {
    "clear-missing-event",
    "clear-extra-event",
    "rest-insertion",
    "strong-context-recovery",
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


def _row(seed: int, group_id: int, sample_id: int) -> tuple[EventPresencePatchInput, int, str]:
    group_seed = (seed << 23) ^ (group_id * 0x9E3779B1)
    scenario = _choose_scenario(random.Random(group_seed))
    rng = random.Random(group_seed ^ ((sample_id + 1) * 0x85EBCA77))
    positive = scenario in POSITIVE_SCENARIOS
    operation = "insert" if scenario in {"clear-missing-event", "rest-insertion", "strong-context-recovery"} else "delete"
    if scenario not in POSITIVE_SCENARIOS:
        operation = rng.choice(("insert", "delete"))

    candidate_count = rng.randint(4, 7)
    eligible_families = rng.choice((3, 4))
    incomplete = 0 if rng.random() < 0.85 else 1
    total_families = eligible_families + incomplete
    winning = 4 if total_families >= 4 and rng.random() < (0.45 if positive else 0.18) else 3
    winning = min(winning, eligible_families)
    runner_up = max(0, eligible_families - winning)
    template_count = rng.randint(3, 11)
    winner_count = template_count + (1 if operation == "insert" else -1)
    winner_count = max(2, winner_count)
    edit_index = rng.randint(1, max(1, winner_count - 2)) if winner_count > 2 else 0
    anchor_match = rng.uniform(0.82, 1.0) if positive else rng.uniform(0.66, 0.96)
    anchor_margin = rng.uniform(0.25, 1.0) if positive else rng.uniform(0.02, 0.48)
    inserted_content = winning if operation == "insert" else 0
    inserted_runner_up = 0
    inserted_rest = operation == "insert" and scenario == "rest-insertion"
    expected_duration = rng.choice((3.0, 4.0, 6.0))
    inserted_duration = rng.choice((0.25, 0.5, 1.0, 2.0)) if operation == "insert" else 0.0
    inserted_duration = min(inserted_duration, expected_duration / 2.0)
    template_error = inserted_duration if operation == "insert" else rng.choice((0.5, 1.0, 2.0))
    if operation == "delete":
        inserted_duration = 0.0
    patched_error = 0.0

    quality = 0.89 if positive else 0.65
    visual_centre = 0.88 if positive else 0.55
    context_centre = 0.87 if positive else 0.56
    ensemble_delta = rng.uniform(0.10, 0.30) if positive else rng.uniform(-0.18, 0.08)
    score_margin = rng.uniform(6.0, 38.0) if positive else rng.uniform(-32.0, 12.0)

    if scenario == "strong-context-recovery":
        visual_centre = 0.72
        context_centre = 0.96
        quality = 0.84
    elif scenario == "common-mode-hallucination":
        quality = 0.83
        visual_centre = 0.48
        context_centre = 0.50
        ensemble_delta = rng.uniform(-0.04, 0.08)
        anchor_margin = rng.uniform(0.04, 0.30)
    elif scenario == "meter-boundary-conflict":
        quality = 0.78
        visual_centre = 0.67
        context_centre = 0.42
        ensemble_delta = rng.uniform(-0.10, 0.04)
        anchor_match = rng.uniform(0.68, 0.88)
        inserted_duration = expected_duration * rng.uniform(0.28, 0.50) if operation == "insert" else 0.0
    elif scenario == "visual-conflict":
        quality = 0.84
        visual_centre = 0.16
        context_centre = 0.72
        ensemble_delta = rng.uniform(-0.08, 0.05)
    elif scenario == "context-conflict":
        quality = 0.82
        visual_centre = 0.76
        context_centre = 0.18
        ensemble_delta = rng.uniform(-0.10, 0.05)
    elif scenario == "template-quality-superior":
        quality = 0.72
        visual_centre = 0.66
        context_centre = 0.70
        ensemble_delta = rng.uniform(-0.24, -0.02)
        score_margin = rng.uniform(-38.0, -2.0)
    elif scenario == "weak-majority":
        quality = 0.34
        visual_centre = 0.30
        context_centre = 0.33
        ensemble_delta = rng.uniform(-0.25, -0.04)
        score_margin = rng.uniform(-46.0, -10.0)
        incomplete = max(incomplete, 1)

    page = _quality(rng, quality)
    measure = _quality(rng, quality + (0.03 if positive else 0.0))
    visual = _quality(rng, visual_centre, 0.09)
    event = _quality(rng, quality + (0.05 if positive else 0.0))
    context = _quality(rng, context_centre, 0.09)
    ensemble = _quality(rng, quality + (0.06 if positive else 0.0))
    minimum_ensemble = _clip(ensemble - rng.uniform(0.02, 0.20))

    # Add overlap so the forest cannot solve the task from a single synthetic field.
    if rng.random() < 0.18:
        visual = _clip(visual + rng.gauss(0.0, 0.10))
        context = _clip(context + rng.gauss(0.0, 0.10))
        ensemble_delta = _clip(ensemble_delta + rng.gauss(0.0, 0.06), -1.0, 1.0)
        anchor_margin = _clip(anchor_margin + rng.gauss(0.0, 0.08))

    item = EventPresencePatchInput(
        candidate_count=candidate_count,
        eligible_family_count=eligible_families,
        winning_family_count=winning,
        runner_up_family_count=runner_up,
        incomplete_family_count=incomplete,
        operation=operation,
        template_event_count=template_count,
        winner_event_count=winner_count,
        edit_index=edit_index,
        anchor_match_ratio=anchor_match,
        anchor_margin_ratio=anchor_margin,
        inserted_content_family_count=inserted_content,
        inserted_content_runner_up_count=inserted_runner_up,
        inserted_event_is_rest=inserted_rest,
        inserted_event_duration=inserted_duration,
        expected_measure_duration=expected_duration,
        template_duration_error=template_error,
        patched_duration_error=patched_error,
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
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=1,
        random_state=seed,
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
        "target": "single-event insertion/deletion improves a uniquely anchored simple measure",
        "scope": "interior meter-complete monophonic proposals already passing deterministic family/XML guards",
    }
    return model, calibrator, payload


def _probabilities(model, calibrator, values: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(values)[:, 1]
    return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    accepted = probabilities >= threshold
    true_accepts = int(np.sum(accepted & (labels == 1)))
    false_accepts = int(np.sum(accepted & (labels == 0)))
    accepts = true_accepts + false_accepts
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
    candidates.extend([0.975, 0.985, 0.99, 0.995, 0.999, 0.9995])
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
    parser.add_argument("--output", type=Path, default=ROOT / "src/scorescan/resources/event_presence_patch_calibrator.json")
    parser.add_argument("--report", type=Path, default=ROOT.parent / "training/event_presence_patch_calibrator_report_v1.json")
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--groups", type=int, default=8000)
    parser.add_argument("--confirmation-groups", type=int, default=2600)
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
    threshold = max(float(DEFAULT_POLICY.event_presence_patch_probability_floor), raw_threshold)
    threshold_metrics = _metrics(dataset.labels[threshold_indices], threshold_probabilities, threshold)
    payload.update({
        "training_groups": args.groups,
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
        "policy_probability_floor": DEFAULT_POLICY.event_presence_patch_probability_floor,
        "selected_threshold": threshold_metrics,
        "frozen_test": {
            "sample": _sample_metrics(dataset.labels[test], test_probabilities),
            "policy": _metrics(dataset.labels[test], test_probabilities, threshold),
            "accept_all_deterministic_proposals": baseline_test,
            "scenarios": _scenario_metrics(dataset, test, test_probabilities, threshold),
        },
        "independent_confirmation": {
            "seed": args.seed + 2,
            "groups": args.confirmation_groups,
            "samples": len(confirmation.labels),
            "dataset_fingerprint": _dataset_fingerprint(confirmation),
            "policy": _metrics(confirmation.labels, confirmation_probabilities, threshold),
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
