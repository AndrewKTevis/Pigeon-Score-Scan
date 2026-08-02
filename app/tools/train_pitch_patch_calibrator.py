from __future__ import annotations

"""Train the CPU gate for event-level pitch-only consensus repairs.

The model is a veto layer.  Training rows already satisfy the deterministic runtime
preconditions: identical non-pitch event skeletons, at least three independent family
votes, and strict family majorities for every disputed pitch.  The target is whether
the proposed patch reduces pitch error without introducing any new wrong pitch.
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

from scorescan.pitch_consensus import FEATURE_NAMES, PitchPatchInput  # noqa: E402
from scorescan.util import atomic_write_json  # noqa: E402
from pitch_visual_training_data import build_rendered_pitch_dataset  # noqa: E402
from tree_export import deployed_forest_probabilities, serialize_probability_forest  # noqa: E402

MODEL_VERSION = "scorescan-pitch-patch-forest-4"
TARGET_PRECISION = 0.999
NO_VISUAL_PROBABILITY_FLOOR = 0.70
SCENARIO_WEIGHTS = {
    "clear-independent-gain": 0.19,
    "complementary-errors": 0.17,
    "subtle-gain": 0.09,
    "accidental-only-gain": 0.06,
    "correlated-family-error": 0.15,
    "template-better": 0.09,
    "visual-conflict": 0.08,
    "quality-conflict": 0.06,
    "hidden-octave-shift": 0.06,
    "local-pitch-conflict": 0.05,
}
MODEL_CONFIGS = (
    {"n_estimators": 32, "max_depth": 7, "min_samples_leaf": 8},
    {"n_estimators": 48, "max_depth": 8, "min_samples_leaf": 7},
    {"n_estimators": 64, "max_depth": 9, "min_samples_leaf": 6},
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


def _probability(
    *,
    correct: bool,
    strength: float,
    style: float,
    noise: float,
    rng: random.Random,
) -> float:
    direction = 1.0 if correct else -1.0
    return _clip(_sigmoid(direction * strength + style + rng.gauss(0.0, noise)), 0.01, 0.995)


def _row(seed: int, group_id: int) -> tuple[PitchPatchInput, int, str]:
    rng = random.Random(seed * 1_000_003 + group_id * 97_409)
    scenarios = tuple(SCENARIO_WEIGHTS)
    scenario = rng.choices(scenarios, weights=tuple(SCENARIO_WEIGHTS.values()), k=1)[0]
    positive = scenario in {
        "clear-independent-gain",
        "complementary-errors",
        "subtle-gain",
        "accidental-only-gain",
    }

    family_count = rng.choice((3, 4, 4, 4))
    candidate_count = family_count + rng.randint(1, 3)
    total_events = rng.randint(2, 16)
    if scenario == "complementary-errors":
        changed_events = rng.randint(max(2, total_events // 4), max(2, total_events // 2))
    elif scenario in {"clear-independent-gain", "correlated-family-error", "local-pitch-conflict"}:
        changed_events = rng.randint(1, min(5, total_events))
    else:
        changed_events = rng.randint(1, min(3, total_events))

    if family_count == 3:
        support_ratio = 1.0
        margin_ratio = rng.uniform(0.66, 1.0)
        template_support = 0.0
    else:
        support_ratio = rng.choices((0.75, 1.0), weights=(0.78, 0.22), k=1)[0]
        margin_ratio = 0.50 if support_ratio == 0.75 else rng.uniform(0.75, 1.0)
        template_support = rng.uniform(0.0, 0.25 if support_ratio == 0.75 else 0.08)

    abstention = _clip(rng.betavariate(1.1, 12.0) * 0.35, 0.0, 0.28)
    if scenario == "quality-conflict":
        abstention = rng.uniform(0.08, 0.25)
    voting_families = max(3, round(family_count * (1.0 - abstention)))

    style_page = rng.gauss(0.0, 0.20)
    style_visual = rng.gauss(0.0, 0.22)
    style_event = rng.gauss(0.0, 0.18)
    style_context = rng.gauss(0.0, 0.20)

    if scenario == "clear-independent-gain":
        strengths = (1.55, 1.85, 1.30, 1.80, 1.20, 1.95)
        deltas = (0.20, 0.24, 0.18, 0.24, 0.16, 0.28)
    elif scenario == "complementary-errors":
        strengths = (1.25, 1.55, 1.10, 1.60, 1.10, 1.70)
        deltas = (0.12, 0.17, 0.12, 0.18, 0.11, 0.20)
    elif scenario == "subtle-gain":
        strengths = (0.85, 1.05, 0.72, 1.08, 0.78, 1.18)
        deltas = (0.04, 0.06, 0.04, 0.07, 0.04, 0.08)
    elif scenario == "accidental-only-gain":
        strengths = (1.05, 1.25, 1.10, 1.20, 0.88, 1.32)
        deltas = (0.08, 0.11, 0.10, 0.10, 0.07, 0.13)
    elif scenario == "correlated-family-error":
        strengths = (1.15, 1.35, 0.30, 1.18, 0.65, 1.30)
        deltas = (0.08, 0.10, -0.12, 0.05, -0.03, 0.04)
    elif scenario == "hidden-octave-shift":
        strengths = (1.35, 1.55, -0.15, 1.35, 0.75, 1.42)
        deltas = (0.12, 0.15, -0.24, 0.10, -0.04, 0.08)
    elif scenario in {"visual-conflict", "local-pitch-conflict"}:
        strengths = (0.95, 1.15, -0.65, 1.00, 0.55, 1.05)
        deltas = (0.04, 0.06, -0.30, 0.02, -0.06, 0.00)
    elif scenario == "quality-conflict":
        strengths = (0.35, 0.55, 0.20, 0.42, 0.18, 0.38)
        deltas = (-0.08, -0.05, -0.04, -0.08, -0.06, -0.10)
    else:  # template-better
        strengths = (-0.55, -0.35, -0.28, -0.45, -0.30, -0.38)
        deltas = (-0.18, -0.15, -0.12, -0.18, -0.13, -0.20)

    page_strength, measure_strength, visual_strength, event_strength, context_strength, ensemble_strength = strengths
    page = _probability(correct=positive, strength=abs(page_strength), style=style_page, noise=0.55, rng=rng)
    measure = _probability(correct=positive, strength=abs(measure_strength), style=0.0, noise=0.52, rng=rng)
    visual = _probability(correct=positive, strength=abs(visual_strength), style=style_visual, noise=0.62, rng=rng)
    event = _probability(correct=positive, strength=abs(event_strength), style=style_event, noise=0.52, rng=rng)
    context = _probability(correct=positive, strength=abs(context_strength), style=style_context, noise=0.60, rng=rng)
    ensemble = _probability(correct=positive, strength=abs(ensemble_strength), style=0.0, noise=0.46, rng=rng)

    # Hard negatives may fool page/measure/event layers; visual and template-relative
    # deltas remain the independent evidence expected at deployment.
    if scenario in {
        "correlated-family-error",
        "hidden-octave-shift",
        "visual-conflict",
        "local-pitch-conflict",
    }:
        page = rng.uniform(0.70, 0.96)
        measure = rng.uniform(0.72, 0.97)
        event = rng.uniform(0.68, 0.96)
        ensemble = rng.uniform(0.72, 0.97)
        context = rng.uniform(0.40, 0.88)
        visual = rng.uniform(0.08, 0.52)

    page_delta, measure_delta, visual_delta, event_delta, context_delta, ensemble_delta = deltas
    jitter = lambda scale=0.035: rng.gauss(0.0, scale)
    visual_available = rng.random() >= 0.12
    accidental_only_ratio = 1.0 if scenario == "accidental-only-gain" else 0.0
    if scenario in {"clear-independent-gain", "complementary-errors"} and rng.random() < 0.12:
        accidental_only_ratio = rng.uniform(0.35, 0.75)
    staff_position_ratio = 0.0 if accidental_only_ratio >= 0.99 else rng.uniform(0.55, 1.0)
    maximum_staff_delta = 0.0
    if staff_position_ratio > 0.0:
        maximum_staff_delta = rng.choice((1, 1, 2, 2, 3))
        if scenario == "hidden-octave-shift":
            maximum_staff_delta = rng.choice((7, 7, 8, 9))

    if positive and scenario != "accidental-only-gain":
        direct_centre = {
            "clear-independent-gain": 0.24,
            "complementary-errors": 0.17,
            "subtle-gain": 0.07,
        }[scenario]
    elif scenario == "accidental-only-gain":
        direct_centre = 0.0
    elif scenario in {"visual-conflict", "hidden-octave-shift", "local-pitch-conflict"}:
        direct_centre = -0.26 if scenario != "hidden-octave-shift" else -0.38
    elif scenario == "correlated-family-error":
        direct_centre = -0.10
    else:
        direct_centre = -0.04

    if not visual_available:
        template_gaps = [0.5] * 7
        proposal_gaps = [0.5] * 7
        direct = [0.0] * 7
        strict_template_gaps = [0.5] * 7
        strict_proposal_gaps = [0.5] * 7
        strict_direct = [0.0] * 7
    else:
        scales = (1.0, 1.05, 0.90, 1.10, 0.65, 0.90, 0.72)
        requested_direct = [
            _clip(direct_centre * scale + rng.gauss(0.0, 0.055), -1.0, 1.0)
            for scale in scales
        ]
        # Direct notehead features must remain neutral for accidental-only changes.
        if accidental_only_ratio >= 0.99:
            requested_direct = [rng.gauss(0.0, 0.015) for _ in requested_direct]
        template_gaps = [rng.uniform(0.16, 0.82) for _ in requested_direct]
        proposal_gaps = [
            _clip(before - improvement)
            for before, improvement in zip(template_gaps, requested_direct, strict=True)
        ]
        direct = [
            before - after
            for before, after in zip(template_gaps, proposal_gaps, strict=True)
        ]
        # The strict detector observes the same transaction through staff-removed ink.
        # It is correlated but not duplicated: weak or joined heads may disappear,
        # while staff-intersection artefacts are less likely to survive.
        strict_requested = [
            _clip(value * rng.uniform(0.72, 1.05) + rng.gauss(0.0, 0.045), -1.0, 1.0)
            for value in requested_direct
        ]
        if accidental_only_ratio >= 0.99:
            strict_requested = [rng.gauss(0.0, 0.012) for _ in strict_requested]
        strict_template_gaps = [rng.uniform(0.16, 0.82) for _ in strict_requested]
        strict_proposal_gaps = [
            _clip(before - improvement)
            for before, improvement in zip(
                strict_template_gaps, strict_requested, strict=True
            )
        ]
        strict_direct = [
            before - after
            for before, after in zip(
                strict_template_gaps, strict_proposal_gaps, strict=True
            )
        ]

    input_row = PitchPatchInput(
        candidate_count=candidate_count,
        eligible_family_count=family_count,
        voting_family_count=voting_families,
        changed_event_count=changed_events,
        total_event_count=total_events,
        minimum_winner_family_support_ratio=_clip(support_ratio - rng.uniform(0.0, 0.04)),
        mean_winner_family_support_ratio=_clip(support_ratio + rng.uniform(-0.02, 0.02)),
        minimum_winner_margin_ratio=_clip(margin_ratio - rng.uniform(0.0, 0.06)),
        mean_winner_margin_ratio=_clip(margin_ratio + rng.uniform(-0.03, 0.03)),
        maximum_template_family_support_ratio=_clip(template_support + rng.uniform(0.0, 0.04)),
        family_abstention_ratio=abstention,
        mean_support_page_probability=page,
        mean_support_measure_probability=measure,
        mean_support_visual_probability=visual,
        mean_support_event_probability=event,
        mean_support_context_probability=context,
        mean_support_ensemble_probability=ensemble,
        minimum_support_ensemble_probability=_clip(ensemble - rng.uniform(0.02, 0.22)),
        mean_support_page_score_margin=(page_delta + jitter(0.05)) * 100.0,
        mean_support_vs_template_measure_probability=measure_delta + jitter(),
        mean_support_vs_template_visual_probability=visual_delta + jitter(0.045),
        mean_support_vs_template_event_probability=event_delta + jitter(),
        mean_support_vs_template_context_probability=context_delta + jitter(0.04),
        mean_support_vs_template_ensemble_probability=ensemble_delta + jitter(),
        visual_evidence_available=visual_available,
        changed_staff_position_ratio=staff_position_ratio,
        maximum_staff_position_delta=maximum_staff_delta,
        accidental_only_change_ratio=accidental_only_ratio,
        notehead_exact_cell_improvement=direct[0],
        notehead_near_cell_improvement=direct[1],
        notehead_vertical_chamfer_improvement=direct[2],
        notehead_severe_vertical_improvement=direct[3],
        notehead_visual_unmatched_improvement=direct[4],
        notehead_column_centroid_improvement=direct[5],
        notehead_column_order_improvement=direct[6],
        template_notehead_exact_cell_gap=template_gaps[0],
        template_notehead_near_cell_gap=template_gaps[1],
        template_notehead_vertical_chamfer_gap=template_gaps[2],
        template_notehead_severe_vertical_gap=template_gaps[3],
        template_notehead_visual_unmatched_gap=template_gaps[4],
        template_notehead_column_centroid_gap=template_gaps[5],
        template_notehead_column_order_gap=template_gaps[6],
        proposal_notehead_exact_cell_gap=proposal_gaps[0],
        proposal_notehead_near_cell_gap=proposal_gaps[1],
        proposal_notehead_vertical_chamfer_gap=proposal_gaps[2],
        proposal_notehead_severe_vertical_gap=proposal_gaps[3],
        proposal_notehead_visual_unmatched_gap=proposal_gaps[4],
        proposal_notehead_column_centroid_gap=proposal_gaps[5],
        proposal_notehead_column_order_gap=proposal_gaps[6],
        strict_notehead_exact_cell_improvement=strict_direct[0],
        strict_notehead_near_cell_improvement=strict_direct[1],
        strict_notehead_vertical_chamfer_improvement=strict_direct[2],
        strict_notehead_severe_vertical_improvement=strict_direct[3],
        strict_notehead_visual_unmatched_improvement=strict_direct[4],
        strict_notehead_column_centroid_improvement=strict_direct[5],
        strict_notehead_column_order_improvement=strict_direct[6],
        template_strict_notehead_exact_cell_gap=strict_template_gaps[0],
        template_strict_notehead_near_cell_gap=strict_template_gaps[1],
        template_strict_notehead_vertical_chamfer_gap=strict_template_gaps[2],
        template_strict_notehead_severe_vertical_gap=strict_template_gaps[3],
        template_strict_notehead_visual_unmatched_gap=strict_template_gaps[4],
        template_strict_notehead_column_centroid_gap=strict_template_gaps[5],
        template_strict_notehead_column_order_gap=strict_template_gaps[6],
        proposal_strict_notehead_exact_cell_gap=strict_proposal_gaps[0],
        proposal_strict_notehead_near_cell_gap=strict_proposal_gaps[1],
        proposal_strict_notehead_vertical_chamfer_gap=strict_proposal_gaps[2],
        proposal_strict_notehead_severe_vertical_gap=strict_proposal_gaps[3],
        proposal_strict_notehead_visual_unmatched_gap=strict_proposal_gaps[4],
        proposal_strict_notehead_column_centroid_gap=strict_proposal_gaps[5],
        proposal_strict_notehead_column_order_gap=strict_proposal_gaps[6],
    )
    return input_row, int(positive), scenario


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


def _combine_datasets(*datasets: object) -> Dataset:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    scenarios: list[str] = []
    next_group = 0
    for dataset in datasets:
        local_features = np.asarray(getattr(dataset, "features"), dtype=np.float64)
        local_labels = np.asarray(getattr(dataset, "labels"), dtype=np.int64)
        local_groups = np.asarray(getattr(dataset, "groups"), dtype=np.int64)
        local_scenarios = tuple(str(value) for value in getattr(dataset, "scenarios"))
        if len(local_features) != len(local_labels) or len(local_labels) != len(local_groups):
            raise ValueError("dataset row counts differ")
        unique = sorted(set(int(value) for value in local_groups.tolist()))
        remap = {value: next_group + index for index, value in enumerate(unique)}
        features.append(local_features)
        labels.append(local_labels)
        groups.append(np.asarray([remap[int(value)] for value in local_groups], dtype=np.int64))
        scenarios.extend(local_scenarios)
        next_group += len(unique)
    return Dataset(
        np.concatenate(features, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(groups, axis=0),
        tuple(scenarios),
    )


def _rendered_mask(dataset: Dataset, indices: np.ndarray) -> np.ndarray:
    return np.asarray(
        [dataset.scenarios[int(index)].startswith("rendered-") for index in indices],
        dtype=bool,
    )


def _no_visual_mask(dataset: Dataset, indices: np.ndarray) -> np.ndarray:
    feature_index = FEATURE_NAMES.index("visual_evidence_available")
    return np.asarray(dataset.features[indices, feature_index] < 0.5, dtype=bool)


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
        "target": "pitch-only patch improves template without introducing a wrong pitch",
        "scope": "grouped synthetic plus rendered source-crop transactions after deterministic skeleton and family-majority guards",
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


def _select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    rendered_mask: np.ndarray | None = None,
) -> tuple[float, dict[str, float | int]]:
    candidates = sorted(set(float(value) for value in probabilities), reverse=True)
    candidates.extend([0.90, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9999])
    valid: list[tuple[float, dict[str, float | int]]] = []
    for threshold in sorted(set(candidates)):
        metrics = _metrics(labels, probabilities, threshold)
        rendered_false_accepts = 0
        if rendered_mask is not None and bool(np.any(rendered_mask)):
            rendered_false_accepts = int(
                _metrics(labels[rendered_mask], probabilities[rendered_mask], threshold)[
                    "false_accepts"
                ]
            )
        metrics = {**metrics, "rendered_false_accepts": rendered_false_accepts}
        if (
            metrics["accepted"]
            and float(metrics["precision"]) >= TARGET_PRECISION
            and rendered_false_accepts == 0
        ):
            valid.append((threshold, metrics))
    if not valid:
        fallback = _metrics(labels, probabilities, 1.0)
        return 1.0, {**fallback, "rendered_false_accepts": 0}
    return max(
        valid,
        key=lambda item: (
            float(item[1]["coverage"]),
            float(item[1]["positive_recall"]),
            item[0],
        ),
    )


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
        default=ROOT / "src" / "scorescan" / "resources" / "pitch_patch_calibrator.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT.parent / "training" / "pitch_patch_calibrator_report_v4.json",
    )
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=ROOT.parent / "training" / "baselines" / "pitch_patch_calibrator_v2.json",
    )
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--groups", type=int, default=9000)
    parser.add_argument("--confirmation-groups", type=int, default=3000)
    parser.add_argument("--rendered-seed", type=int, default=20260727)
    parser.add_argument("--rendered-groups", type=int, default=1600)
    parser.add_argument("--rendered-confirmation-groups", type=int, default=600)
    parser.add_argument("--no-visual-audit-seed", type=int, default=20260743)
    parser.add_argument("--no-visual-audit-groups", type=int, default=30000)
    args = parser.parse_args()

    synthetic = build_dataset(args.seed, args.groups)
    rendered = build_rendered_pitch_dataset(args.rendered_seed, args.rendered_groups)
    dataset = _combine_datasets(synthetic, rendered)
    train, calibration, audit, threshold_indices, test = _split(dataset.groups, args.seed)

    trained: list[tuple[RandomForestClassifier, LogisticRegression, dict[str, object]]] = []
    audit_rows: list[dict[str, object]] = []
    for config in MODEL_CONFIGS:
        model, calibrator, payload = _fit(dataset, train, calibration, args.seed, config)
        probabilities = _probabilities(model, calibrator, dataset.features[audit])
        rendered_audit = _rendered_mask(dataset, audit)
        synthetic_audit = ~rendered_audit
        audit_rows.append({
            "config": dict(config),
            "sample": _sample_metrics(dataset.labels[audit], probabilities),
            "synthetic": _sample_metrics(
                dataset.labels[audit][synthetic_audit], probabilities[synthetic_audit]
            ),
            "rendered": _sample_metrics(
                dataset.labels[audit][rendered_audit], probabilities[rendered_audit]
            ),
        })
        trained.append((model, calibrator, payload))

    best_index = min(
        range(len(audit_rows)),
        key=lambda index: (
            -min(
                float(audit_rows[index]["synthetic"]["roc_auc"]),
                float(audit_rows[index]["rendered"]["roc_auc"]),
            ),
            -float(audit_rows[index]["rendered"]["roc_auc"]),
            float(audit_rows[index]["sample"]["log_loss"]),
            int(audit_rows[index]["config"]["n_estimators"]),
        ),
    )
    model, calibrator, payload = trained[best_index]
    selected_config = dict(audit_rows[best_index]["config"])
    threshold_probabilities = _probabilities(model, calibrator, dataset.features[threshold_indices])
    threshold_rendered = _rendered_mask(dataset, threshold_indices)
    threshold, threshold_metrics = _select_threshold(
        dataset.labels[threshold_indices],
        threshold_probabilities,
        rendered_mask=threshold_rendered,
    )
    no_visual_threshold_mask = _no_visual_mask(dataset, threshold_indices)
    selected_no_visual_threshold, _ = _select_threshold(
        dataset.labels[threshold_indices][no_visual_threshold_mask],
        threshold_probabilities[no_visual_threshold_mask],
    )
    no_visual_threshold = max(
        NO_VISUAL_PROBABILITY_FLOOR, selected_no_visual_threshold
    )
    no_visual_threshold_metrics = _metrics(
        dataset.labels[threshold_indices][no_visual_threshold_mask],
        threshold_probabilities[no_visual_threshold_mask],
        no_visual_threshold,
    )
    payload.update({
        "training_groups": args.groups,
        "training_rendered_groups": args.rendered_groups,
        "selected_on": "independent grouped model-selection audit",
        "selected_config": selected_config,
        "auto_patch_threshold": threshold,
        "no_visual_auto_patch_threshold": no_visual_threshold,
        "target_precision": TARGET_PRECISION,
    })

    test_probabilities = _probabilities(model, calibrator, dataset.features[test])
    deployed = deployed_forest_probabilities(payload, dataset.features[test])
    deployment_delta = float(np.max(np.abs(test_probabilities - deployed), initial=0.0))
    if deployment_delta > 1e-10:
        raise RuntimeError(f"deployment prediction mismatch: {deployment_delta}")

    confirmation = _combine_datasets(
        build_dataset(args.seed + 2, args.confirmation_groups),
        build_rendered_pitch_dataset(
            args.rendered_seed + 2, args.rendered_confirmation_groups
        ),
    )
    confirmation_probabilities = _probabilities(model, calibrator, confirmation.features)

    test_no_visual = _no_visual_mask(dataset, test)
    confirmation_indices = np.arange(len(confirmation.labels), dtype=np.int64)
    confirmation_no_visual = _no_visual_mask(confirmation, confirmation_indices)
    no_visual_frozen_metrics = _metrics(
        dataset.labels[test][test_no_visual],
        test_probabilities[test_no_visual],
        no_visual_threshold,
    )
    no_visual_confirmation_metrics = _metrics(
        confirmation.labels[confirmation_no_visual],
        confirmation_probabilities[confirmation_no_visual],
        no_visual_threshold,
    )
    no_visual_audit = build_dataset(
        args.no_visual_audit_seed, args.no_visual_audit_groups
    )
    no_visual_audit_probabilities = deployed_forest_probabilities(
        payload, no_visual_audit.features
    )
    no_visual_audit_indices = np.arange(
        len(no_visual_audit.labels), dtype=np.int64
    )
    no_visual_audit_mask = _no_visual_mask(
        no_visual_audit, no_visual_audit_indices
    )
    no_visual_audit_metrics = _metrics(
        no_visual_audit.labels[no_visual_audit_mask],
        no_visual_audit_probabilities[no_visual_audit_mask],
        no_visual_threshold,
    )
    if (
        int(no_visual_threshold_metrics["false_accepts"])
        or int(no_visual_frozen_metrics["false_accepts"])
        or int(no_visual_confirmation_metrics["false_accepts"])
        or int(no_visual_audit_metrics["false_accepts"])
    ):
        raise RuntimeError("no-visual threshold admitted a negative proposal")

    # Baseline means accepting every deterministic family-majority proposal.  It
    # quantifies the correlated-family failure the learned veto is intended to remove.
    baseline_test = _metrics(dataset.labels[test], np.ones(len(test), dtype=np.float64), 0.5)
    baseline_confirmation = _metrics(
        confirmation.labels,
        np.ones(len(confirmation.labels), dtype=np.float64),
        0.5,
    )

    baseline_payload = json.loads(args.baseline_model.read_text(encoding="utf-8"))
    baseline_feature_count = len(baseline_payload.get("feature_names", ()))
    baseline_test_probabilities = deployed_forest_probabilities(
        baseline_payload,
        dataset.features[test, :baseline_feature_count],
    )
    baseline_confirmation_probabilities = deployed_forest_probabilities(
        baseline_payload,
        confirmation.features[:, :baseline_feature_count],
    )
    try:
        baseline_threshold = float(baseline_payload.get("auto_patch_threshold", 1.0))
    except (TypeError, ValueError):
        baseline_threshold = 1.0

    atomic_write_json(args.output, payload)
    report = {
        "model_version": MODEL_VERSION,
        "seed": args.seed,
        "synthetic_groups": args.groups,
        "rendered_groups": args.rendered_groups,
        "samples": len(dataset.labels),
        "partitions": {
            "train": len(train),
            "calibration": len(calibration),
            "model_selection_audit": len(audit),
            "threshold_selection": len(threshold_indices),
            "frozen_test": len(test),
            "rendered_train_rows": int(np.sum(_rendered_mask(dataset, train))),
            "rendered_calibration_rows": int(np.sum(_rendered_mask(dataset, calibration))),
            "rendered_model_selection_rows": int(np.sum(_rendered_mask(dataset, audit))),
            "rendered_threshold_rows": int(np.sum(_rendered_mask(dataset, threshold_indices))),
            "rendered_frozen_test_rows": int(np.sum(_rendered_mask(dataset, test))),
        },
        "model_selection": audit_rows,
        "selected_config": selected_config,
        "selected_threshold": threshold_metrics,
        "selected_no_visual_threshold": no_visual_threshold_metrics,
        "frozen_test": {
            "sample": _sample_metrics(dataset.labels[test], test_probabilities),
            "policy": _metrics(dataset.labels[test], test_probabilities, threshold),
            "accept_all_deterministic_proposals": baseline_test,
            "synthetic_policy": _metrics(
                dataset.labels[test][~_rendered_mask(dataset, test)],
                test_probabilities[~_rendered_mask(dataset, test)],
                threshold,
            ),
            "rendered_policy": _metrics(
                dataset.labels[test][_rendered_mask(dataset, test)],
                test_probabilities[_rendered_mask(dataset, test)],
                threshold,
            ),
            "no_visual_policy": no_visual_frozen_metrics,
            "scenarios": _scenario_metrics(dataset, test, test_probabilities, threshold),
            "v2_same_test": {
                "policy": _metrics(dataset.labels[test], baseline_test_probabilities, baseline_threshold),
                "sample": _sample_metrics(dataset.labels[test], baseline_test_probabilities),
                "scenarios": _scenario_metrics(
                    dataset, test, baseline_test_probabilities, baseline_threshold
                ),
            },
        },
        "independent_confirmation": {
            "seed": args.seed + 2,
            "synthetic_groups": args.confirmation_groups,
            "rendered_groups": args.rendered_confirmation_groups,
            "policy": _metrics(confirmation.labels, confirmation_probabilities, threshold),
            "synthetic_policy": _metrics(
                confirmation.labels[np.asarray([not value.startswith("rendered-") for value in confirmation.scenarios])],
                confirmation_probabilities[np.asarray([not value.startswith("rendered-") for value in confirmation.scenarios])],
                threshold,
            ),
            "rendered_policy": _metrics(
                confirmation.labels[np.asarray([value.startswith("rendered-") for value in confirmation.scenarios])],
                confirmation_probabilities[np.asarray([value.startswith("rendered-") for value in confirmation.scenarios])],
                threshold,
            ),
            "no_visual_policy": no_visual_confirmation_metrics,
            "accept_all_deterministic_proposals": baseline_confirmation,
            "v2_same_test": _metrics(
                confirmation.labels,
                baseline_confirmation_probabilities,
                baseline_threshold,
            ),
        },
        "large_no_visual_confirmation": {
            "seed": args.no_visual_audit_seed,
            "groups": args.no_visual_audit_groups,
            "eligible_rows": int(np.sum(no_visual_audit_mask)),
            "negative_rows": int(
                np.sum(no_visual_audit.labels[no_visual_audit_mask] == 0)
            ),
            "positive_rows": int(
                np.sum(no_visual_audit.labels[no_visual_audit_mask] == 1)
            ),
            "policy": no_visual_audit_metrics,
        },
        "deployment_parity": {"max_absolute_probability_delta": deployment_delta},
        "feature_names": list(FEATURE_NAMES),
        "model_bytes": args.output.stat().st_size,
        "baseline_model_version": str(baseline_payload.get("model_version", "unknown")),
    }
    atomic_write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
