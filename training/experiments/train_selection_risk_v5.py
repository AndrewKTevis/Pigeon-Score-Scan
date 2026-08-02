from __future__ import annotations

"""Train the CPU replacement-verification gate.

The verifier is the last learned veto before ScoreScan replaces a page-template
measure.  Unlike v2, it covers both exact-majority and fuzzy semantic-consensus
replacements.  It never creates consensus, chooses another candidate, or edits
MusicXML; it can only retain the template and request review.

Five disjoint score-family partitions are used for fitting, probability calibration,
model selection, threshold selection, and a frozen final test.  Related evidence
conditions never cross partitions.  Thresholds are selected independently for exact
majority and semantic consensus at very high selective precision.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.selection_risk import (  # noqa: E402
    FEATURE_NAMES,
    SelectionRiskInput,
    corroborated_exact_majority,
    corroborated_semantic_consensus,
)
from scorescan.tree_model import gradient_boosting_probability  # noqa: E402
from scorescan.util import atomic_write_json  # noqa: E402
from tree_export import deployed_forest_probabilities, serialize_probability_forest  # noqa: E402

MODEL_VERSION = "scorescan-selection-risk-forest-5"
FAMILY_FEATURE_COUNT = 5
SCENARIO_WEIGHTS = {
    "semantic-clear-gain": 0.14,
    "semantic-subtle-gain": 0.10,
    "semantic-false-consensus": 0.12,
    "exact-clear-gain": 0.16,
    "exact-family-redundant-trap": 0.08,
    "exact-invalid-family-trap": 0.07,
    "exact-invalid-family-clear-gain": 0.05,
    "exact-cross-family-trap": 0.07,
    "template-better": 0.07,
    "invalid-selected": 0.04,
    "evidence-conflict": 0.04,
    "localized-clear-gain": 0.07,
    "localized-confident-trap": 0.05,
    "localized-partial-trap": 0.03,
}
MODEL_CONFIGS = (
    {"n_estimators": 32, "max_depth": 7, "min_samples_leaf": 8},
    {"n_estimators": 48, "max_depth": 8, "min_samples_leaf": 7},
    {"n_estimators": 64, "max_depth": 9, "min_samples_leaf": 6},
)


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _sigmoid(value: float) -> float:
    value = max(-30.0, min(30.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def _scenario_distances(kind: str, rng: random.Random) -> tuple[float, float]:
    if kind in {"semantic-clear-gain", "exact-clear-gain", "exact-invalid-family-clear-gain", "localized-clear-gain"}:
        template = rng.uniform(0.10, 0.36)
        selected = max(0.0, template - rng.uniform(0.06, 0.24))
    elif kind == "semantic-subtle-gain":
        template = rng.uniform(0.05, 0.22)
        selected = max(0.0, template - rng.uniform(0.012, 0.060))
    elif kind in {"semantic-false-consensus", "exact-family-redundant-trap", "exact-invalid-family-trap", "exact-cross-family-trap", "template-better", "localized-confident-trap", "localized-partial-trap"}:
        template = rng.uniform(0.0, 0.13)
        selected = min(0.55, template + rng.uniform(0.02, 0.23))
    elif kind == "invalid-selected":
        template = rng.uniform(0.04, 0.20)
        selected = max(0.0, template - rng.uniform(0.02, 0.12))
    else:
        template = rng.uniform(0.03, 0.24)
        selected = _clip(template + rng.uniform(-0.11, 0.11), 0.0, 0.50)
    return selected, template


def _probability(distance: float, bias: float, sensitivity: float, style: float, noise: float, rng: random.Random) -> float:
    return _clip(_sigmoid(bias - sensitivity * distance + style + rng.gauss(0.0, noise)), 0.01, 0.995)


def _build_dataset(*, seed: int, groups: int, decisions_per_group: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    scenario_names = tuple(SCENARIO_WEIGHTS)
    scenario_weights = tuple(SCENARIO_WEIGHTS.values())
    rows: list[list[float]] = []
    labels: list[int] = []
    group_ids: list[int] = []
    metadata: list[dict[str, object]] = []

    for group_id in range(groups):
        rng = random.Random(seed * 1_000_003 + group_id * 97_409)
        style_page = rng.gauss(0.0, 0.18)
        style_visual = rng.gauss(0.0, 0.22)
        style_event = rng.gauss(0.0, 0.18)
        style_context = rng.gauss(0.0, 0.20)
        difficulty = rng.random()

        for decision_id in range(decisions_per_group):
            kind = rng.choices(scenario_names, weights=scenario_weights, k=1)[0]
            selection_kind = "exact_majority" if kind.startswith("exact-") else "semantic_consensus"
            selected_distance, template_distance = _scenario_distances(kind, rng)
            selected_valid = kind not in {"invalid-selected", "localized-partial-trap"} and rng.random() > 0.015 + 0.10 * selected_distance
            replacement_improves = int(selected_valid and selected_distance + 0.008 < template_distance)

            candidate_count = rng.randint(3, 8)
            if kind in {"exact-family-redundant-trap", "exact-invalid-family-trap", "exact-invalid-family-clear-gain"}:
                candidate_count = rng.randint(6, 8)
            elif kind.startswith("localized-"):
                candidate_count = rng.randint(5, 8)
            eligible_family_count = min(5, candidate_count)
            if kind == "exact-invalid-family-trap":
                # Several raw candidates appear to agree, but invalid siblings make
                # their preprocessing families abstain in production.  Only one or
                # two complete healthy families remain, so automatic replacement must
                # be rejected even when candidate-count support looks overwhelming.
                eligible_family_count = rng.randint(1, 2)
            elif kind == "exact-invalid-family-clear-gain":
                eligible_family_count = rng.randint(3, 4)
            if selection_kind == "exact_majority":
                exact_support_count = rng.randint(candidate_count // 2 + 1, candidate_count - 1)
                if kind in {"exact-family-redundant-trap", "exact-invalid-family-trap"}:
                    exact_family_support_count = min(2, eligible_family_count)
                elif kind == "exact-invalid-family-clear-gain":
                    exact_family_support_count = rng.randint(3, eligible_family_count)
                elif kind == "exact-cross-family-trap":
                    exact_family_support_count = min(3, eligible_family_count)
                else:
                    minimum = min(3, eligible_family_count)
                    exact_family_support_count = rng.randint(minimum, eligible_family_count)
                semantic_family_support_count = exact_family_support_count
                exact_support_ratio = exact_support_count / candidate_count
                semantic_support_ratio = _clip(exact_support_ratio + rng.uniform(0.0, 0.12))
                mean_cluster_distance = rng.uniform(0.0, 0.012)
                strict_majority = True
            else:
                exact_support_count = rng.randint(1, max(1, candidate_count // 2))
                exact_family_support_count = min(exact_support_count, eligible_family_count)
                minimum_family_support = 3 if kind.startswith("localized-") else min(2, eligible_family_count)
                semantic_family_support_count = rng.randint(
                    min(minimum_family_support, eligible_family_count),
                    eligible_family_count,
                )
                exact_support_ratio = exact_support_count / candidate_count
                semantic_support_ratio = _clip(rng.uniform(0.74, 0.96) - 0.04 * difficulty, 0.70, 0.99)
                mean_cluster_distance = _clip(rng.uniform(0.008, 0.043) + 0.012 * difficulty, 0.0, 0.075)
                strict_majority = False

            signature_support_ratio = exact_support_ratio
            missing_ratio = _clip(rng.betavariate(1.2, 9.0) * 0.28, 0.0, 0.25)
            selected_template_distance = _clip(abs(selected_distance - template_distance) + rng.uniform(0.012, 0.10), 0.009, 0.50)
            selected_alignment = _clip(0.97 - 0.36 * selected_distance + style_page * 0.04 + rng.gauss(0.0, 0.035), 0.42, 0.997)
            template_alignment = _clip(0.97 - 0.36 * template_distance + style_page * 0.04 + rng.gauss(0.0, 0.035), 0.42, 0.997)

            selected_page_probability = _probability(selected_distance, 1.55, 4.6, style_page, 0.48, rng)
            template_page_probability = _probability(template_distance, 1.80, 4.0, style_page, 0.44, rng)
            selected_measure_probability = _probability(selected_distance, 1.35 + 0.65 * semantic_support_ratio, 6.0, 0.0, 0.52, rng)
            template_measure_probability = _probability(template_distance, 1.55, 5.7, 0.0, 0.52, rng)
            selected_visual_probability = _probability(selected_distance, 0.60, 2.5, style_visual, 0.66, rng)
            template_visual_probability = _probability(template_distance, 0.62, 2.5, style_visual, 0.66, rng)
            selected_event_probability = _probability(selected_distance, 1.05, 5.4, style_event, 0.56, rng)
            template_event_probability = _probability(template_distance, 1.05, 5.4, style_event, 0.56, rng)
            selected_context_probability = _probability(selected_distance, 0.80, 3.8, style_context, 0.65, rng)
            template_context_probability = _probability(template_distance, 0.80, 3.8, style_context, 0.65, rng)

            consensus_boost = 0.0
            if kind in {"semantic-false-consensus", "exact-family-redundant-trap", "exact-invalid-family-trap", "localized-confident-trap", "localized-partial-trap"}:
                consensus_boost = 0.85
            elif kind == "exact-cross-family-trap":
                consensus_boost = 0.55
            selected_ensemble_probability = _clip(_sigmoid(
                -0.65
                + 1.0 * selected_page_probability
                + 1.2 * selected_measure_probability
                + 0.55 * selected_visual_probability
                + 1.0 * selected_event_probability
                + 0.75 * selected_context_probability
                + 1.15 * semantic_support_ratio
                + consensus_boost
                + rng.gauss(0.0, 0.38)
            ), 0.01, 0.995)
            template_ensemble_probability = _clip(_sigmoid(
                -0.45
                + 1.0 * template_page_probability
                + 1.2 * template_measure_probability
                + 0.55 * template_visual_probability
                + 1.0 * template_event_probability
                + 0.75 * template_context_probability
                + rng.gauss(0.0, 0.38)
            ), 0.01, 0.995)

            # Correlated false-consensus scenarios deliberately boost raw support while
            # leaving independent visual/event/template comparisons informative.
            if kind in {"exact-family-redundant-trap", "exact-invalid-family-trap"}:
                # One correlated preprocessing family can be confidently wrong across
                # all local evidence layers.  Raw candidate support therefore looks
                # like a clean majority; only independent-family support reveals that
                # the evidence is duplicated rather than corroborated.
                selected_page_probability = rng.uniform(0.82, 0.96)
                selected_measure_probability = rng.uniform(0.82, 0.98)
                selected_visual_probability = rng.uniform(0.68, 0.93)
                selected_event_probability = rng.uniform(0.80, 0.97)
                selected_context_probability = rng.uniform(0.72, 0.94)
                selected_ensemble_probability = rng.uniform(0.88, 0.995)
                template_page_probability = rng.uniform(0.55, 0.78)
                template_measure_probability = rng.uniform(0.50, 0.76)
                template_visual_probability = rng.uniform(0.45, 0.70)
                template_event_probability = rng.uniform(0.50, 0.74)
                template_context_probability = rng.uniform(0.48, 0.72)
                template_ensemble_probability = rng.uniform(0.55, 0.79)
            elif kind == "exact-cross-family-trap":
                selected_event_probability = _clip(selected_event_probability - 0.18)
                selected_visual_probability = _clip(selected_visual_probability - 0.12)
            elif kind == "evidence-conflict":
                selected_ensemble_probability = _clip(selected_ensemble_probability + rng.uniform(-0.12, 0.18))
                selected_event_probability = _clip(selected_event_probability + rng.uniform(-0.20, 0.20))
            elif kind == "localized-clear-gain":
                # The stitched candidate can have a modest page prior while independent
                # event, measure and family evidence confirms the local rescue.
                selected_page_probability = _clip(selected_page_probability - rng.uniform(0.08, 0.20))
                selected_measure_probability = _clip(selected_measure_probability + rng.uniform(0.08, 0.18))
                selected_event_probability = _clip(selected_event_probability + rng.uniform(0.08, 0.18))
                selected_ensemble_probability = _clip(selected_ensemble_probability + rng.uniform(0.04, 0.12))
            elif kind in {"localized-confident-trap", "localized-partial-trap"}:
                # Clean system crops may produce overconfident page/alignment signals.
                # Direct template and event/visual comparisons must still veto the
                # semantically wrong or incomplete stitch.
                selected_page_probability = rng.uniform(0.86, 0.98)
                selected_measure_probability = rng.uniform(0.72, 0.92)
                selected_ensemble_probability = rng.uniform(0.88, 0.995)
                selected_visual_probability = rng.uniform(0.30, 0.60)
                selected_event_probability = rng.uniform(0.32, 0.62)
                selected_context_probability = rng.uniform(0.38, 0.68)
                template_page_probability = rng.uniform(0.62, 0.82)
                template_measure_probability = rng.uniform(0.58, 0.82)
                template_ensemble_probability = rng.uniform(0.60, 0.84)
                template_visual_probability = rng.uniform(0.58, 0.82)
                template_event_probability = rng.uniform(0.60, 0.86)
                template_context_probability = rng.uniform(0.55, 0.80)
                selected_alignment = rng.uniform(0.91, 0.995)

            page_score = 820.0 + 190.0 * selected_page_probability + rng.gauss(0.0, 22.0)
            best_other_page_probability = max(template_page_probability, _clip(template_page_probability + rng.gauss(-0.02, 0.07)))
            page_score_margin = (selected_page_probability - best_other_page_probability) * 115.0 + rng.gauss(-5.0, 12.0)
            distinct_signature_count = rng.randint(2, candidate_count)
            runner_up = max(1, exact_support_count - rng.randint(1, max(1, exact_support_count)))
            top_signature_margin = max(0.0, (exact_support_count - runner_up) / candidate_count)
            template_in_cluster = rng.random() < (0.08 if replacement_improves else 0.62)
            template_exact = rng.random() < (0.05 if replacement_improves else 0.50)

            item = SelectionRiskInput(
                selection_kind=selection_kind,
                selected_page_score=page_score,
                selected_page_probability=selected_page_probability,
                selected_ensemble_probability=selected_ensemble_probability,
                ensemble_probability_margin=selected_ensemble_probability - max(template_ensemble_probability, 0.5),
                selected_measure_probability=selected_measure_probability,
                measure_probability_margin=selected_measure_probability - max(template_measure_probability, 0.5),
                selected_visual_probability=selected_visual_probability,
                visual_probability_margin=selected_visual_probability - max(template_visual_probability, 0.5),
                selected_event_probability=selected_event_probability,
                event_probability_margin=selected_event_probability - max(template_event_probability, 0.5),
                selected_context_probability=selected_context_probability,
                context_probability_margin=selected_context_probability - max(template_context_probability, 0.5),
                exact_support_ratio=exact_support_ratio,
                semantic_support_ratio=semantic_support_ratio,
                signature_support_ratio=signature_support_ratio,
                missing_ratio=missing_ratio,
                mean_cluster_distance=mean_cluster_distance,
                template_distance=selected_template_distance,
                alignment_similarity=selected_alignment,
                alignment_margin=selected_alignment - max(selected_alignment, template_alignment),
                selected_distance_to_medoid=(
                    0.0
                    if selection_kind == "semantic_consensus"
                    else _clip(rng.uniform(0.0, 0.02) + 0.06 * difficulty)
                ),
                selected_mean_peer_distance=_clip(mean_cluster_distance + rng.uniform(0.0, 0.03)),
                page_score_margin=page_score_margin,
                candidate_count=candidate_count,
                exact_support_count=exact_support_count,
                distinct_signature_count=distinct_signature_count,
                top_signature_margin=top_signature_margin,
                unanimous=False,
                strict_majority=strict_majority,
                selected_is_template=False,
                selected_is_exact_signature=True,
                selected_in_initial_cluster=True,
                page_valid=selected_valid,
                selected_vs_template_page_probability=selected_page_probability - template_page_probability,
                selected_vs_template_ensemble_probability=selected_ensemble_probability - template_ensemble_probability,
                selected_vs_template_measure_probability=selected_measure_probability - template_measure_probability,
                selected_vs_template_visual_probability=selected_visual_probability - template_visual_probability,
                selected_vs_template_event_probability=selected_event_probability - template_event_probability,
                selected_vs_template_context_probability=selected_context_probability - template_context_probability,
                selected_vs_template_alignment_similarity=selected_alignment - template_alignment,
                template_page_valid=True,
                template_in_initial_cluster=template_in_cluster,
                template_is_exact_signature=template_exact,
                eligible_family_count=eligible_family_count,
                exact_family_support_count=exact_family_support_count,
                semantic_family_support_count=semantic_family_support_count,
            )
            rows.append(item.feature_vector())
            labels.append(replacement_improves)
            group_ids.append(group_id)
            metadata.append({
                "group": group_id,
                "decision": decision_id,
                "scenario": kind,
                "selection_kind": selection_kind,
                "label": replacement_improves,
                "semantic_gain": template_distance - selected_distance,
                "corroborated_exact_majority": corroborated_exact_majority(item),
                "corroborated_semantic_consensus": corroborated_semantic_consensus(item),
            })

    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(labels, dtype=np.int32),
        np.asarray(group_ids, dtype=np.int32),
        metadata,
    )


def _partition_indices(group_ids: np.ndarray, seed: int) -> tuple[np.ndarray, ...]:
    groups = sorted(set(int(value) for value in group_ids))
    rng = random.Random(seed + 91)
    rng.shuffle(groups)
    count = len(groups)
    boundaries = (int(count * 0.65), int(count * 0.75), int(count * 0.85), int(count * 0.90))
    partitions = (
        set(groups[: boundaries[0]]),
        set(groups[boundaries[0] : boundaries[1]]),
        set(groups[boundaries[1] : boundaries[2]]),
        set(groups[boundaries[2] : boundaries[3]]),
        set(groups[boundaries[3] :]),
    )
    return tuple(np.asarray([i for i, group in enumerate(group_ids) if int(group) in part], dtype=np.int64) for part in partitions)


def _fit_forest(values: np.ndarray, labels: np.ndarray, train: np.ndarray, calibration: np.ndarray, seed: int, config: dict[str, int]) -> tuple[RandomForestClassifier, LogisticRegression, dict[str, object]]:
    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features="sqrt",
        class_weight="balanced_subsample",
        bootstrap=True,
        n_jobs=1,
        random_state=seed,
    )
    model.fit(values[train], labels[train])
    raw = model.predict_proba(values[calibration])[:, 1]
    calibrator = LogisticRegression(C=1000.0, max_iter=2000, random_state=seed)
    calibrator.fit(raw.reshape(-1, 1), labels[calibration])
    payload: dict[str, object] = {
        "model_version": MODEL_VERSION,
        "model_type": "random_forest",
        "feature_names": list(FEATURE_NAMES),
        "trees": serialize_probability_forest(model),
        "calibration_intercept": float(calibrator.intercept_[0]),
        "calibration_slope": float(calibrator.coef_[0, 0]),
        "model_config": dict(config),
    }
    return model, calibrator, payload


def _probabilities(model: RandomForestClassifier, calibrator: LogisticRegression, values: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(values)[:, 1]
    return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


def _classification(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities >= 0.5
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }


def _selective(labels: np.ndarray, probabilities: np.ndarray, accepted: np.ndarray, gains: np.ndarray | None = None) -> dict[str, float | int]:
    count = int(np.sum(accepted))
    result: dict[str, float | int] = {
        "accepted": count,
        "coverage": float(np.mean(accepted)),
        "precision": float(np.mean(labels[accepted])) if count else 1.0,
        "true_accepts": int(np.sum(accepted & (labels == 1))),
        "false_accepts": int(np.sum(accepted & (labels == 0))),
    }
    if gains is not None:
        result["total_semantic_gain"] = float(np.sum(gains[accepted]))
        result["mean_semantic_gain"] = float(np.mean(gains[accepted])) if count else 0.0
    return result


def _choose_threshold(labels: np.ndarray, probabilities: np.ndarray, mask: np.ndarray, target_precision: float, minimum_coverage: float, floor: float) -> float:
    subset_probabilities = probabilities[mask]
    subset_labels = labels[mask]
    best = 1.0
    best_coverage = -1.0
    for threshold in sorted(set(float(value) for value in subset_probabilities), reverse=True):
        threshold = max(floor, threshold)
        accepted = subset_probabilities >= threshold
        coverage = float(np.mean(accepted))
        if coverage < minimum_coverage:
            continue
        precision = float(np.mean(subset_labels[accepted])) if np.any(accepted) else 1.0
        if precision >= target_precision and coverage > best_coverage:
            best = threshold
            best_coverage = coverage
    if best_coverage < 0.0:
        raise RuntimeError("no threshold satisfies selective precision and coverage")
    return best


def _optional_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    mask: np.ndarray,
    target_precision: float,
    minimum_coverage: float,
    floor: float,
) -> tuple[float, bool]:
    try:
        return (
            _choose_threshold(
                labels,
                probabilities,
                mask,
                target_precision,
                minimum_coverage,
                floor,
            ),
            True,
        )
    except RuntimeError:
        return 1.0, False


def _baseline_probabilities(payload: dict[str, object], values: np.ndarray) -> np.ndarray:
    baseline_values = np.asarray(values, dtype=np.float64).copy()
    if str(payload.get("model_version", "")) == "scorescan-selection-risk-forest-3":
        # v3 saturated the independent-family count at four.  Convert the v4
        # five-family encoding back to the exact v3 representation for same-test
        # comparison.
        column = FEATURE_NAMES.index("eligible_family_count_scaled")
        family_count = np.rint(baseline_values[:, column] * 5.0).astype(np.int64)
        baseline_values[:, column] = np.minimum(1.0, np.maximum(1, family_count) / 4.0)
    if str(payload.get("model_type", "")) == "random_forest":
        return deployed_forest_probabilities(payload, baseline_values)
    feature_names = payload.get("feature_names", [])
    count = len(feature_names) if isinstance(feature_names, list) else 0
    return np.asarray([
        gradient_boosting_probability(
            row[:count],
            intercept=float(payload.get("intercept", 0.0)),
            learning_rate=float(payload.get("learning_rate", 0.0)),
            trees=payload.get("trees", []),
            calibration_intercept=float(payload.get("calibration_intercept", 0.0)),
            calibration_slope=float(payload.get("calibration_slope", 1.0)),
        )
        for row in baseline_values
    ], dtype=np.float64)


def _policy_metrics(labels: np.ndarray, metadata: list[dict[str, object]], indices: np.ndarray, accepted: np.ndarray, gains: np.ndarray) -> dict[str, object]:
    result: dict[str, object] = _selective(labels[indices], np.ones(len(indices)), accepted, gains[indices])
    by_kind: dict[str, object] = {}
    for kind in ("exact_majority", "semantic_consensus"):
        mask = np.asarray([metadata[int(index)]["selection_kind"] == kind for index in indices], dtype=bool)
        by_kind[kind] = _selective(labels[indices][mask], np.ones(int(np.sum(mask))), accepted[mask], gains[indices][mask])
    result["by_selection_kind"] = by_kind
    return result


def _zero_false_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    mask: np.ndarray,
    current: float,
) -> float:
    negatives = probabilities[mask & (labels == 0)]
    if negatives.size == 0:
        return float(current)
    return min(1.0, max(float(current), float(np.max(negatives)) + 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "src" / "scorescan" / "resources" / "selection_risk.json")
    parser.add_argument("--report", type=Path, default=ROOT.parent / "training" / "selection_risk_report_v5.json")
    parser.add_argument("--baseline-model", type=Path, default=ROOT.parent / "training" / "baselines" / "selection_risk_v4.json")
    parser.add_argument("--seed", type=int, default=20270501)
    parser.add_argument("--groups", type=int, default=6000)
    parser.add_argument("--decisions-per-group", type=int, default=3)
    parser.add_argument("--target-precision", type=float, default=1.0)
    parser.add_argument("--exact-minimum-coverage", type=float, default=0.15)
    parser.add_argument("--semantic-minimum-coverage", type=float, default=0.015)
    parser.add_argument("--minimum-threshold", type=float, default=0.90)
    parser.add_argument("--exact-minimum-threshold", type=float, default=0.986)
    parser.add_argument("--semantic-minimum-threshold", type=float, default=0.992)
    parser.add_argument("--safety-seed", type=int, default=20270503)
    parser.add_argument("--safety-groups", type=int, default=4000)
    parser.add_argument("--confirmation-seed", type=int, default=20270611)
    parser.add_argument("--confirmation-groups", type=int, default=3000)
    args = parser.parse_args()

    baseline_payload = json.loads(args.baseline_model.read_text(encoding="utf-8"))
    values, labels, group_ids, metadata = _build_dataset(seed=args.seed, groups=args.groups, decisions_per_group=args.decisions_per_group)
    train, calibration, model_audit, threshold_audit, test = _partition_indices(group_ids, args.seed)

    fitted: list[tuple[RandomForestClassifier, LogisticRegression, dict[str, object]]] = []
    audit_rows: list[dict[str, object]] = []
    for config in MODEL_CONFIGS:
        model, calibrator, payload = _fit_forest(values, labels, train, calibration, args.seed, config)
        probabilities = _probabilities(model, calibrator, values[model_audit])
        audit_rows.append({"config": config, "metrics": _classification(labels[model_audit], probabilities)})
        fitted.append((model, calibrator, payload))
    best_auc = max(float(row["metrics"]["roc_auc"]) for row in audit_rows)
    best_loss = min(float(row["metrics"]["log_loss"]) for row in audit_rows)
    compact = [i for i, row in enumerate(audit_rows) if float(row["metrics"]["roc_auc"]) >= best_auc - 0.004 and float(row["metrics"]["log_loss"]) <= best_loss + 0.015]
    selected_index = min(compact, key=lambda i: (MODEL_CONFIGS[i]["n_estimators"], MODEL_CONFIGS[i]["max_depth"]))
    model, calibrator, payload = fitted[selected_index]

    threshold_probabilities = _probabilities(model, calibrator, values[threshold_audit])
    threshold_kinds = np.asarray([metadata[int(index)]["selection_kind"] for index in threshold_audit])
    threshold_corroborated = np.asarray(
        [bool(metadata[int(index)]["corroborated_exact_majority"]) for index in threshold_audit],
        dtype=bool,
    )
    exact_threshold = _choose_threshold(
        labels[threshold_audit], threshold_probabilities,
        (threshold_kinds == "exact_majority") & threshold_corroborated, args.target_precision,
        args.exact_minimum_coverage,
        max(args.minimum_threshold, args.exact_minimum_threshold),
    )
    threshold_semantic_corroborated = np.asarray(
        [bool(metadata[int(index)]["corroborated_semantic_consensus"]) for index in threshold_audit],
        dtype=bool,
    )
    semantic_threshold = _choose_threshold(
        labels[threshold_audit], threshold_probabilities,
        (threshold_kinds == "semantic_consensus") & threshold_semantic_corroborated,
        args.target_precision,
        args.semantic_minimum_coverage,
        max(args.minimum_threshold, args.semantic_minimum_threshold),
    )

    # A disjoint safety-calibration corpus may only raise thresholds.  The final
    # confirmation seed below remains untouched until model, calibration and thresholds
    # are fixed.  This gives the release policy a reproducible zero-false-accept target
    # without tuning on the frozen test or final confirmation partitions.
    safety_values, safety_labels, _safety_groups, safety_metadata = _build_dataset(
        seed=args.safety_seed,
        groups=args.safety_groups,
        decisions_per_group=args.decisions_per_group,
    )
    safety_probabilities = _probabilities(model, calibrator, safety_values)
    safety_kinds = np.asarray([item["selection_kind"] for item in safety_metadata], dtype=object)
    safety_exact_corroborated = np.asarray(
        [bool(item["corroborated_exact_majority"]) for item in safety_metadata], dtype=bool
    )
    safety_semantic_corroborated = np.asarray(
        [bool(item["corroborated_semantic_consensus"]) for item in safety_metadata], dtype=bool
    )
    exact_threshold = _zero_false_threshold(
        safety_labels,
        safety_probabilities,
        (safety_kinds == "exact_majority") & safety_exact_corroborated,
        exact_threshold,
    )
    semantic_threshold = _zero_false_threshold(
        safety_labels,
        safety_probabilities,
        (safety_kinds == "semantic_consensus") & safety_semantic_corroborated,
        semantic_threshold,
    )
    safety_accepted = np.where(
        safety_kinds == "exact_majority",
        safety_exact_corroborated & (safety_probabilities >= exact_threshold),
        safety_semantic_corroborated & (safety_probabilities >= semantic_threshold),
    )
    for kind, minimum in (("exact_majority", args.exact_minimum_coverage), ("semantic_consensus", args.semantic_minimum_coverage)):
        kind_mask = safety_kinds == kind
        coverage = float(np.mean(safety_accepted[kind_mask])) if np.any(kind_mask) else 0.0
        if coverage < float(minimum):
            raise RuntimeError(f"zero-false safety threshold loses required {kind} coverage: {coverage}")

    payload.update({
        "auto_replace_threshold": min(exact_threshold, semantic_threshold),
        "auto_replace_thresholds": {"exact_majority": exact_threshold, "semantic_consensus": semantic_threshold},
        "target_precision": args.target_precision,
        "training": {
            "seed": args.seed,
            "groups": args.groups,
            "decisions_per_group": args.decisions_per_group,
            "samples": len(values),
            "split_unit": "programmatic score-family identity",
            "target_definition": "selected replacement is page-valid and improves reference distance over the retained template by more than 0.008",
        },
    })

    sklearn_test = _probabilities(model, calibrator, values[test])
    deployed_test = deployed_forest_probabilities(payload, values[test])
    deployment_delta = float(np.max(np.abs(sklearn_test - deployed_test), initial=0.0))
    if deployment_delta > 1e-10:
        raise RuntimeError(f"deployment prediction mismatch: {deployment_delta}")

    test_kinds = np.asarray([metadata[int(index)]["selection_kind"] for index in test])
    test_exact_corroborated = np.asarray(
        [bool(metadata[int(index)]["corroborated_exact_majority"]) for index in test], dtype=bool
    )
    test_semantic_corroborated = np.asarray(
        [bool(metadata[int(index)]["corroborated_semantic_consensus"]) for index in test], dtype=bool
    )
    v5_accepted = np.where(
        test_kinds == "exact_majority",
        test_exact_corroborated & (deployed_test >= exact_threshold),
        test_semantic_corroborated & (deployed_test >= semantic_threshold),
    )
    baseline_all = _baseline_probabilities(baseline_payload, values)
    baseline_thresholds = baseline_payload.get("auto_replace_thresholds", {})
    if not isinstance(baseline_thresholds, dict):
        baseline_thresholds = {}
    baseline_default = float(baseline_payload.get("auto_replace_threshold", 1.0))
    baseline_exact_threshold = float(baseline_thresholds.get("exact_majority", baseline_default))
    baseline_semantic_threshold = float(baseline_thresholds.get("semantic_consensus", baseline_default))
    baseline_accepted = np.where(
        test_kinds == "exact_majority",
        test_exact_corroborated & (baseline_all[test] >= baseline_exact_threshold),
        test_semantic_corroborated & (baseline_all[test] >= baseline_semantic_threshold),
    )
    gains = np.asarray([float(item["semantic_gain"]) for item in metadata], dtype=np.float64)

    # Family-feature ablation uses the selected forest complexity and identical splits.
    ablation_values = values[:, :-FAMILY_FEATURE_COUNT]
    ablation_model, ablation_calibrator, _ = _fit_forest(ablation_values, labels, train, calibration, args.seed, MODEL_CONFIGS[selected_index])
    ablation_threshold = _probabilities(ablation_model, ablation_calibrator, ablation_values[threshold_audit])
    ablation_exact_threshold, ablation_exact_feasible = _optional_threshold(
        labels[threshold_audit], ablation_threshold,
        (threshold_kinds == "exact_majority") & threshold_corroborated, args.target_precision,
        args.exact_minimum_coverage,
        max(args.minimum_threshold, args.exact_minimum_threshold),
    )
    ablation_semantic_threshold, ablation_semantic_feasible = _optional_threshold(
        labels[threshold_audit], ablation_threshold,
        (threshold_kinds == "semantic_consensus") & threshold_semantic_corroborated,
        args.target_precision,
        args.semantic_minimum_coverage,
        max(args.minimum_threshold, args.semantic_minimum_threshold),
    )
    ablation_test = _probabilities(ablation_model, ablation_calibrator, ablation_values[test])
    ablation_accepted = np.where(
        test_kinds == "exact_majority",
        test_exact_corroborated & (ablation_test >= ablation_exact_threshold),
        test_semantic_corroborated & (ablation_test >= ablation_semantic_threshold),
    )

    scenarios = sorted(SCENARIO_WEIGHTS)
    scenario_metrics: dict[str, object] = {}
    for scenario in scenarios:
        mask = np.asarray([metadata[int(index)]["scenario"] == scenario for index in test], dtype=bool)
        scenario_metrics[scenario] = _selective(labels[test][mask], deployed_test[mask], v5_accepted[mask], gains[test][mask])

    # A completely independent confirmation corpus is generated only after model and
    # policy thresholds are fixed.  It is not used for fitting, calibration, model
    # selection, or threshold selection.
    confirmation_values, confirmation_labels, _confirmation_groups, confirmation_metadata = _build_dataset(
        seed=args.confirmation_seed,
        groups=args.confirmation_groups,
        decisions_per_group=args.decisions_per_group,
    )
    confirmation_probabilities = deployed_forest_probabilities(payload, confirmation_values)
    confirmation_kinds = np.asarray(
        [item["selection_kind"] for item in confirmation_metadata],
        dtype=object,
    )
    confirmation_exact_corroborated = np.asarray(
        [bool(item["corroborated_exact_majority"]) for item in confirmation_metadata], dtype=bool
    )
    confirmation_semantic_corroborated = np.asarray(
        [bool(item["corroborated_semantic_consensus"]) for item in confirmation_metadata], dtype=bool
    )
    confirmation_accepted = np.where(
        confirmation_kinds == "exact_majority",
        confirmation_exact_corroborated & (confirmation_probabilities >= exact_threshold),
        confirmation_semantic_corroborated & (confirmation_probabilities >= semantic_threshold),
    )
    confirmation_gains = np.asarray(
        [float(item["semantic_gain"]) for item in confirmation_metadata],
        dtype=np.float64,
    )
    confirmation_baseline_probabilities = _baseline_probabilities(
        baseline_payload,
        confirmation_values,
    )
    confirmation_baseline_accepted = np.where(
        confirmation_kinds == "exact_majority",
        confirmation_exact_corroborated & (confirmation_baseline_probabilities >= baseline_exact_threshold),
        confirmation_semantic_corroborated & (confirmation_baseline_probabilities >= baseline_semantic_threshold),
    )

    report = {
        "model_version": MODEL_VERSION,
        "seed": args.seed,
        "groups": args.groups,
        "samples": len(values),
        "feature_names": list(FEATURE_NAMES),
        "splits": {
            "train": len(train),
            "probability_calibration": len(calibration),
            "model_selection_audit": len(model_audit),
            "threshold_selection_audit": len(threshold_audit),
            "frozen_test": len(test),
        },
        "model_selection_audit": audit_rows,
        "selected_config": MODEL_CONFIGS[selected_index],
        "thresholds": {
            "target_precision": args.target_precision,
            "exact_majority": exact_threshold,
            "semantic_consensus": semantic_threshold,
        },
        "frozen_test": {
            "classification": _classification(labels[test], deployed_test),
            "v5_policy": _policy_metrics(labels, metadata, test, v5_accepted, gains),
            "baseline_v4_policy_same_test": _policy_metrics(labels, metadata, test, baseline_accepted, gains),
            "by_scenario": scenario_metrics,
        },
        "safety_threshold_calibration": {
            "seed": args.safety_seed,
            "groups": args.safety_groups,
            "samples": int(len(safety_values)),
            "policy": _policy_metrics(
                safety_labels,
                safety_metadata,
                np.arange(len(safety_labels), dtype=np.int64),
                safety_accepted,
                np.asarray([float(item["semantic_gain"]) for item in safety_metadata], dtype=np.float64),
            ),
        },
        "independent_confirmation": {
            "seed": args.confirmation_seed,
            "groups": args.confirmation_groups,
            "samples": int(len(confirmation_values)),
            "v5_policy": _policy_metrics(
                confirmation_labels,
                confirmation_metadata,
                np.arange(len(confirmation_labels), dtype=np.int64),
                confirmation_accepted,
                confirmation_gains,
            ),
            "baseline_v4_policy_same_confirmation": _policy_metrics(
                confirmation_labels,
                confirmation_metadata,
                np.arange(len(confirmation_labels), dtype=np.int64),
                confirmation_baseline_accepted,
                confirmation_gains,
            ),
        },
        "family_feature_ablation": {
            "feature_count": len(FEATURE_NAMES) - FAMILY_FEATURE_COUNT,
            "classification": _classification(labels[test], ablation_test),
            "thresholds": {
                "exact_majority": ablation_exact_threshold,
                "semantic_consensus": ablation_semantic_threshold,
                "exact_majority_feasible": ablation_exact_feasible,
                "semantic_consensus_feasible": ablation_semantic_feasible,
            },
            "policy": _policy_metrics(labels, metadata, test, ablation_accepted, gains),
            "exact_family_redundant_trap": _selective(
                labels[test][np.asarray([metadata[int(index)]["scenario"] == "exact-family-redundant-trap" for index in test], dtype=bool)],
                ablation_test[np.asarray([metadata[int(index)]["scenario"] == "exact-family-redundant-trap" for index in test], dtype=bool)],
                ablation_accepted[np.asarray([metadata[int(index)]["scenario"] == "exact-family-redundant-trap" for index in test], dtype=bool)],
                gains[test][np.asarray([metadata[int(index)]["scenario"] == "exact-family-redundant-trap" for index in test], dtype=bool)],
            ),
        },
        "deployment_parity": {"max_absolute_probability_delta": deployment_delta, "tolerance": 1e-10},
        "scope": "programmatic exact-majority and fuzzy-consensus replacement-benefit verification; not end-to-end OMR accuracy",
        "limitations": [
            "The verifier is trained on programmatic evidence distributions and still requires a large frozen real-scan corpus.",
            "It can only veto an automatic replacement; it cannot repair a measure or create consensus.",
            "Retaining the template can preserve a pre-existing template error, so rejected decisions remain visible for review.",
        ],
    }

    atomic_write_json(args.output, payload)
    atomic_write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
