from __future__ import annotations

"""Train the CPU veto for conservative event-level rhythm repairs.

Rows model proposals which already passed deterministic deployment guards: one voice,
no chords/grace/tuplets, complete family votes, a strict independent-family majority,
and a meter-complete/type-consistent proposed sequence.  The label asks whether the
proposal improves the reference rhythm without introducing a wrong duration or note
type.  The model cannot create or select rhythm; it can only reject a proposal.
"""

import argparse
import json
import math
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

from scorescan.rhythm_consensus import FEATURE_NAMES, RhythmPatchInput  # noqa: E402
from scorescan.util import atomic_write_json  # noqa: E402
from tree_export import deployed_forest_probabilities, serialize_probability_forest  # noqa: E402

MODEL_VERSION = "scorescan-rhythm-patch-forest-1"
TARGET_PRECISION = 0.9975
SCENARIO_WEIGHTS = {
    "clear-meter-repair": 0.18,
    "complementary-errors": 0.18,
    "type-duration-repair": 0.12,
    "subtle-rhythm-repair": 0.08,
    "correlated-family-error": 0.14,
    "template-better": 0.09,
    "pitch-identity-ambiguity": 0.08,
    "visual-conflict": 0.06,
    "pickup-meter-overfit": 0.04,
    "quality-conflict": 0.03,
}
MODEL_CONFIGS = (
    {"n_estimators": 40, "max_depth": 7, "min_samples_leaf": 8},
    {"n_estimators": 56, "max_depth": 8, "min_samples_leaf": 7},
    {"n_estimators": 72, "max_depth": 9, "min_samples_leaf": 6},
)


@dataclass(frozen=True)
class Dataset:
    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    scenarios: tuple[str, ...]


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _sigmoid(value: float) -> float:
    value = max(-30.0, min(30.0, value))
    return 1.0 / (1.0 + math.exp(-value))


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


def _row(seed: int, group_id: int) -> tuple[RhythmPatchInput, int, str]:
    rng = random.Random((seed << 23) ^ (group_id * 0x9E3779B1))
    scenario = _choose_scenario(rng)
    positive = scenario in {
        "clear-meter-repair",
        "complementary-errors",
        "type-duration-repair",
        "subtle-rhythm-repair",
    }

    candidate_count = rng.randint(4, 7)
    family_count = rng.choice((3, 4))
    voting_families = family_count if rng.random() < 0.80 else max(3, family_count - 1)
    changed_events = rng.randint(1, 3 if positive else 4)
    total_events = rng.randint(max(4, changed_events + 1), 14)
    support_count = rng.randint(3, voting_families)
    if scenario in {"correlated-family-error", "pitch-identity-ambiguity"}:
        support_count = 3
        voting_families = 4
    support_ratio = support_count / voting_families
    runner_up = max(0, voting_families - support_count)
    margin_ratio = max(1, support_count - runner_up) / voting_families
    template_support = runner_up / voting_families
    abstention = rng.uniform(0.0, 0.08 if positive else 0.20)

    template_error = rng.choice((0.25, 0.5, 1.0, 1.5)) if positive else rng.choice((0.0, 0.25, 0.5, 1.0))
    template_mismatch = rng.uniform(0.08, 0.40) if positive else rng.uniform(0.0, 0.28)
    if scenario == "type-duration-repair":
        template_error = rng.choice((0.0, 0.25))
        template_mismatch = rng.uniform(0.20, 0.60)
    elif scenario == "subtle-rhythm-repair":
        template_error = rng.choice((0.0, 0.25))
        template_mismatch = rng.uniform(0.06, 0.20)
    elif scenario == "template-better":
        template_error = 0.0
        template_mismatch = 0.0
    elif scenario == "pickup-meter-overfit":
        template_error = rng.uniform(0.5, 1.5)
        template_mismatch = rng.uniform(0.0, 0.08)
    patched_error = 0.0
    patched_mismatch = 0.0
    improvement = template_error - patched_error

    pitch_min = rng.uniform(0.74, 1.0)
    pitch_mean = rng.uniform(max(pitch_min, 0.82), 1.0)
    quality_centre = 0.88 if positive else 0.62
    visual_centre = 0.84 if positive else 0.60
    context_centre = 0.84 if positive else 0.60
    ensemble_delta = rng.uniform(0.08, 0.28) if positive else rng.uniform(-0.16, 0.10)
    score_margin = rng.uniform(4.0, 34.0) if positive else rng.uniform(-25.0, 16.0)

    if scenario == "correlated-family-error":
        quality_centre = 0.88
        visual_centre = 0.38
        context_centre = 0.55
        pitch_min = rng.uniform(0.52, 0.72)
        pitch_mean = rng.uniform(pitch_min, 0.78)
        ensemble_delta = rng.uniform(-0.02, 0.10)
        score_margin = rng.uniform(-4.0, 20.0)
    elif scenario == "pitch-identity-ambiguity":
        quality_centre = 0.82
        visual_centre = 0.56
        context_centre = 0.48
        pitch_min = rng.uniform(0.50, 0.61)
        pitch_mean = rng.uniform(0.55, 0.70)
        ensemble_delta = rng.uniform(-0.08, 0.07)
    elif scenario == "visual-conflict":
        quality_centre = 0.86
        visual_centre = 0.20
        context_centre = 0.62
        ensemble_delta = rng.uniform(-0.04, 0.08)
    elif scenario == "pickup-meter-overfit":
        quality_centre = 0.75
        visual_centre = 0.45
        context_centre = 0.22
        ensemble_delta = rng.uniform(-0.12, 0.02)
        pitch_min = rng.uniform(0.62, 0.80)
    elif scenario == "quality-conflict":
        quality_centre = 0.40
        visual_centre = 0.44
        context_centre = 0.42
        ensemble_delta = rng.uniform(-0.18, 0.0)
        score_margin = rng.uniform(-35.0, -5.0)
    elif scenario == "template-better":
        quality_centre = 0.72
        visual_centre = 0.58
        context_centre = 0.55
        ensemble_delta = rng.uniform(-0.15, 0.04)
        pitch_min = rng.uniform(0.68, 0.90)

    page = _quality(rng, quality_centre)
    measure = _quality(rng, quality_centre + (0.03 if positive else 0.0))
    visual = _quality(rng, visual_centre, 0.10)
    event = _quality(rng, quality_centre + 0.04, 0.07)
    context = _quality(rng, context_centre, 0.10)
    ensemble = _quality(rng, quality_centre + 0.06, 0.06)
    minimum_ensemble = _clip(ensemble - rng.uniform(0.02, 0.20))

    # A small overlap region prevents a trivial one-feature separator and exercises the
    # calibrated veto near the public-release threshold.
    if rng.random() < 0.10:
        perturb = rng.gauss(0.0, 0.08)
        visual = _clip(visual + perturb)
        context = _clip(context - perturb / 2)
        ensemble_delta = _clip(ensemble_delta + rng.gauss(0.0, 0.05), -1.0, 1.0)

    item = RhythmPatchInput(
        candidate_count=candidate_count,
        eligible_family_count=family_count,
        voting_family_count=voting_families,
        changed_event_count=changed_events,
        total_event_count=total_events,
        minimum_winner_family_support_ratio=_clip(support_ratio - rng.uniform(0.0, 0.04)),
        mean_winner_family_support_ratio=_clip(support_ratio + rng.uniform(-0.02, 0.02)),
        minimum_winner_margin_ratio=_clip(margin_ratio - rng.uniform(0.0, 0.05)),
        mean_winner_margin_ratio=_clip(margin_ratio + rng.uniform(-0.02, 0.03)),
        maximum_template_family_support_ratio=_clip(template_support + rng.uniform(0.0, 0.04)),
        family_abstention_ratio=_clip(abstention),
        minimum_pitch_coherence_ratio=_clip(pitch_min),
        mean_pitch_coherence_ratio=_clip(pitch_mean),
        template_duration_error=template_error,
        patched_duration_error=patched_error,
        duration_error_improvement=improvement,
        template_type_mismatch_ratio=_clip(template_mismatch),
        patched_type_mismatch_ratio=patched_mismatch,
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


def _fit(
    dataset: Dataset,
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
        "target": "rhythm-only patch improves template without introducing a wrong duration or note type",
        "scope": "meter-complete monophonic proposals already passing deterministic family and XML guards",
    }
    return model, calibrator, payload


def _probabilities(model: RandomForestClassifier, calibrator: LogisticRegression, values: np.ndarray) -> np.ndarray:
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


def _select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, float | int]]:
    candidates = sorted(set(float(value) for value in probabilities))
    candidates.extend([0.92, 0.95, 0.975, 0.99, 0.995, 0.999])
    valid: list[tuple[float, dict[str, float | int]]] = []
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
        default=ROOT / "src" / "scorescan" / "resources" / "rhythm_patch_calibrator.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT.parent / "training" / "rhythm_patch_calibrator_report_v1.json",
    )
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--groups", type=int, default=6000)
    parser.add_argument("--confirmation-groups", type=int, default=2000)
    args = parser.parse_args()

    dataset = build_dataset(args.seed, args.groups)
    train, calibration, audit, threshold_indices, test = _split(dataset.groups, args.seed)
    trained: list[tuple[RandomForestClassifier, LogisticRegression, dict[str, object]]] = []
    audit_rows: list[dict[str, object]] = []
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
    atomic_write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
