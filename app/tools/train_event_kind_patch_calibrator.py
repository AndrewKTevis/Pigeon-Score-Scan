from __future__ import annotations

"""Train the CPU veto for conservative rest-versus-pitched-note repairs."""

import argparse
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

from scorescan.event_kind_consensus import FEATURE_NAMES, EventKindPatchInput  # noqa: E402
from scorescan.util import atomic_write_json  # noqa: E402
from tree_export import deployed_forest_probabilities, serialize_probability_forest  # noqa: E402

MODEL_VERSION = "scorescan-event-kind-patch-forest-1"
TARGET_PRECISION = 0.999
SCENARIO_WEIGHTS = {
    "false-rest-repair": 0.18,
    "false-note-repair": 0.16,
    "complementary-kind-errors": 0.12,
    "high-visual-corroboration": 0.08,
    "correlated-family-error": 0.13,
    "template-better": 0.09,
    "visual-conflict": 0.08,
    "context-conflict": 0.06,
    "common-mode-kind-error": 0.05,
    "weak-quality-majority": 0.05,
}
MODEL_CONFIGS = (
    {"n_estimators": 48, "max_depth": 7, "min_samples_leaf": 8},
    {"n_estimators": 64, "max_depth": 8, "min_samples_leaf": 7},
    {"n_estimators": 80, "max_depth": 9, "min_samples_leaf": 6},
)


@dataclass(frozen=True)
class Dataset:
    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    scenarios: tuple[str, ...]


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _quality(rng: random.Random, centre: float, spread: float = 0.08) -> float:
    return _clip(rng.gauss(centre, spread), 0.02, 0.995)


def _choose_scenario(rng: random.Random) -> str:
    value = rng.random()
    total = 0.0
    for name, weight in SCENARIO_WEIGHTS.items():
        total += weight
        if value <= total:
            return name
    return next(reversed(SCENARIO_WEIGHTS))


def _row(seed: int, group_id: int) -> tuple[EventKindPatchInput, int, str]:
    rng = random.Random((seed << 21) ^ (group_id * 0x9E3779B1))
    scenario = _choose_scenario(rng)
    positive = scenario in {
        "false-rest-repair",
        "false-note-repair",
        "complementary-kind-errors",
        "high-visual-corroboration",
    }

    candidate_count = rng.randint(4, 7)
    family_count = rng.choice((3, 4))
    voting_families = family_count if rng.random() < 0.84 else max(3, family_count - 1)
    winner_count = voting_families if rng.random() < 0.35 else max(3, voting_families - 1)
    if family_count == 4 and scenario in {
        "correlated-family-error", "common-mode-kind-error", "visual-conflict", "context-conflict"
    }:
        voting_families = 4
        winner_count = 3
    runner_up = max(0, voting_families - winner_count)
    changed_events = rng.randint(1, 2 if positive else 3)
    total_events = rng.randint(max(4, changed_events + 1), 14)
    support_ratio = winner_count / family_count
    margin_ratio = max(1, winner_count - runner_up) / family_count
    template_ratio = runner_up / family_count
    abstention = (family_count - voting_families) / family_count + rng.uniform(0.0, 0.05)

    if scenario == "false-rest-repair":
        pitched_winners = changed_events
    elif scenario == "false-note-repair":
        pitched_winners = 0
    else:
        pitched_winners = rng.randint(0, changed_events)
    rest_winners = changed_events - pitched_winners
    pitched_support_min = support_ratio if pitched_winners else 1.0
    pitched_support_mean = _clip(pitched_support_min + rng.uniform(-0.02, 0.03))

    quality_centre = 0.88 if positive else 0.62
    visual_centre = 0.88 if positive else 0.55
    context_centre = 0.84 if positive else 0.56
    ensemble_delta = rng.uniform(0.10, 0.30) if positive else rng.uniform(-0.18, 0.08)
    visual_delta = rng.uniform(0.08, 0.28) if positive else rng.uniform(-0.25, 0.08)
    event_delta = rng.uniform(0.08, 0.25) if positive else rng.uniform(-0.16, 0.08)
    context_delta = rng.uniform(0.04, 0.22) if positive else rng.uniform(-0.18, 0.08)
    measure_delta = rng.uniform(0.04, 0.20) if positive else rng.uniform(-0.14, 0.08)
    score_margin = rng.uniform(5.0, 36.0) if positive else rng.uniform(-30.0, 14.0)

    if scenario == "correlated-family-error":
        quality_centre = 0.82
        visual_centre = 0.48
        context_centre = 0.55
        ensemble_delta = rng.uniform(-0.04, 0.08)
        visual_delta = rng.uniform(-0.18, 0.02)
        event_delta = rng.uniform(-0.05, 0.08)
    elif scenario == "template-better":
        quality_centre = 0.74
        visual_centre = 0.70
        context_centre = 0.72
        ensemble_delta = rng.uniform(-0.20, 0.0)
        visual_delta = rng.uniform(-0.16, 0.02)
        event_delta = rng.uniform(-0.14, 0.02)
        context_delta = rng.uniform(-0.12, 0.02)
        measure_delta = rng.uniform(-0.12, 0.02)
        score_margin = rng.uniform(-34.0, 2.0)
    elif scenario == "visual-conflict":
        quality_centre = 0.84
        visual_centre = 0.16
        context_centre = 0.68
        ensemble_delta = rng.uniform(-0.05, 0.06)
        visual_delta = rng.uniform(-0.42, -0.15)
    elif scenario == "context-conflict":
        quality_centre = 0.82
        visual_centre = 0.76
        context_centre = 0.20
        context_delta = rng.uniform(-0.38, -0.12)
        ensemble_delta = rng.uniform(-0.08, 0.06)
    elif scenario == "common-mode-kind-error":
        quality_centre = 0.86
        visual_centre = 0.52
        context_centre = 0.50
        ensemble_delta = rng.uniform(-0.02, 0.09)
        visual_delta = rng.uniform(-0.12, 0.06)
        event_delta = rng.uniform(-0.06, 0.08)
        context_delta = rng.uniform(-0.10, 0.05)
    elif scenario == "weak-quality-majority":
        quality_centre = 0.34
        visual_centre = 0.32
        context_centre = 0.36
        ensemble_delta = rng.uniform(-0.22, -0.03)
        visual_delta = rng.uniform(-0.25, -0.04)
        event_delta = rng.uniform(-0.20, -0.02)
        context_delta = rng.uniform(-0.20, -0.02)
        measure_delta = rng.uniform(-0.18, -0.02)
        score_margin = rng.uniform(-42.0, -8.0)
    elif scenario == "high-visual-corroboration":
        quality_centre = 0.78
        visual_centre = 0.95
        context_centre = 0.76
        visual_delta = rng.uniform(0.24, 0.45)

    page = _quality(rng, quality_centre)
    measure = _quality(rng, quality_centre + (0.03 if positive else 0.0))
    visual = _quality(rng, visual_centre, 0.10)
    event = _quality(rng, quality_centre + (0.05 if positive else 0.0), 0.08)
    context = _quality(rng, context_centre, 0.10)
    ensemble = _quality(rng, quality_centre + (0.06 if positive else 0.0), 0.07)
    minimum_ensemble = _clip(ensemble - rng.uniform(0.02, 0.22))

    if rng.random() < 0.14:
        perturb = rng.gauss(0.0, 0.08)
        visual = _clip(visual + perturb)
        context = _clip(context - perturb / 2)
        ensemble_delta = _clip(ensemble_delta + rng.gauss(0.0, 0.05), -1.0, 1.0)
        event_delta = _clip(event_delta + rng.gauss(0.0, 0.05), -1.0, 1.0)

    item = EventKindPatchInput(
        candidate_count=candidate_count,
        eligible_family_count=family_count,
        voting_family_count=voting_families,
        changed_event_count=changed_events,
        total_event_count=total_events,
        minimum_winner_family_support_ratio=_clip(support_ratio - rng.uniform(0.0, 0.03)),
        mean_winner_family_support_ratio=_clip(support_ratio + rng.uniform(-0.02, 0.02)),
        minimum_winner_margin_ratio=_clip(margin_ratio - rng.uniform(0.0, 0.04)),
        mean_winner_margin_ratio=_clip(margin_ratio + rng.uniform(-0.02, 0.03)),
        maximum_template_family_support_ratio=_clip(template_ratio + rng.uniform(0.0, 0.04)),
        family_abstention_ratio=_clip(abstention),
        pitched_winner_count=pitched_winners,
        rest_winner_count=rest_winners,
        minimum_pitched_winner_support_ratio=_clip(pitched_support_min),
        mean_pitched_winner_support_ratio=_clip(pitched_support_mean),
        mean_support_page_probability=page,
        mean_support_measure_probability=measure,
        mean_support_visual_probability=visual,
        mean_support_event_probability=event,
        mean_support_context_probability=context,
        mean_support_ensemble_probability=ensemble,
        minimum_support_ensemble_probability=minimum_ensemble,
        mean_support_page_score_margin=score_margin,
        mean_support_vs_template_measure_probability=measure_delta,
        mean_support_vs_template_visual_probability=visual_delta,
        mean_support_vs_template_event_probability=event_delta,
        mean_support_vs_template_context_probability=context_delta,
        mean_support_vs_template_ensemble_probability=ensemble_delta,
    )
    return item, int(positive), scenario


def build_dataset(seed: int, groups: int) -> Dataset:
    rows: list[list[float]] = []
    labels: list[int] = []
    group_ids: list[int] = []
    scenarios: list[str] = []
    for group_id in range(groups):
        item, label, scenario = _row(seed, group_id)
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
        "target": "rest-versus-pitched-note patch improves the template without changing a correct event kind",
        "scope": "fixed simple event lattices already passing strict independent-family and XML guards",
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
    candidates.extend([0.95, 0.975, 0.99, 0.995, 0.999, 0.9995])
    valid = []
    for threshold in sorted(set(candidates)):
        metrics = _metrics(labels, probabilities, threshold)
        if metrics["accepted"] and float(metrics["precision"]) >= TARGET_PRECISION:
            valid.append((threshold, metrics))
    if not valid:
        return 1.0, _metrics(labels, probabilities, 1.0)
    return max(valid, key=lambda item: (float(item[1]["coverage"]), float(item[1]["positive_recall"]), item[0]))


def _scenario_metrics(dataset: Dataset, indices: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, object]:
    result: dict[str, object] = {}
    for scenario in sorted(set(dataset.scenarios[index] for index in indices)):
        local = np.asarray([offset for offset, index in enumerate(indices) if dataset.scenarios[index] == scenario], dtype=np.int64)
        result[scenario] = _metrics(dataset.labels[indices][local], probabilities[local], threshold)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "src" / "scorescan" / "resources" / "event_kind_patch_calibrator.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT.parent / "training" / "event_kind_patch_calibrator_report_v1.json",
    )
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--groups", type=int, default=7000)
    parser.add_argument("--confirmation-groups", type=int, default=2200)
    args = parser.parse_args()

    dataset = build_dataset(args.seed, args.groups)
    train, calibration, audit, threshold_indices, test = _split(dataset.groups, args.seed)
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
    threshold, threshold_metrics = _select_threshold(dataset.labels[threshold_indices], threshold_probabilities)
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, payload)
    report = {
        "model_version": MODEL_VERSION,
        "seed": args.seed,
        "groups": args.groups,
        "samples": len(dataset.labels),
        "partitions": {
            "train": len(train),
            "calibration": len(calibration),
            "model_selection_audit": len(audit),
            "threshold_selection": len(threshold_indices),
            "frozen_test": len(test),
        },
        "model_selection": audit_rows,
        "selected_config": selected_config,
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
            "policy": _metrics(confirmation.labels, confirmation_probabilities, threshold),
            "accept_all_deterministic_proposals": baseline_confirmation,
        },
        "deployment_parity": {"max_absolute_probability_delta": deployment_delta},
        "feature_names": list(FEATURE_NAMES),
        "model_bytes": args.output.stat().st_size,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
