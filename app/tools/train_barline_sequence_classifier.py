from __future__ import annotations

"""Train the conservative CPU barline-sequence classifier.

The local visual classifier can accept full-height note stems.  This trainer generates
deterministic staff systems with uneven engraving, pickups, compressed endings, missing
boundaries, duplicate strokes, and inserted stem-like candidates.  Related variants of
one system stay in the same split.  The deployed gradient-boosting model is probability
calibrated on a separate partition and evaluated through the dependency-free JSON
runtime used by ScoreScan.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.barline_sequence import (  # noqa: E402
    FEATURE_NAMES,
    LEGACY_FEATURE_NAMES,
    BarlineSequenceFeatures,
    extract_sequence_features,
)
from scorescan.model_registry import build_manifest  # noqa: E402
from scorescan.policy import DEFAULT_POLICY  # noqa: E402
from scorescan.tree_model import gradient_boosting_probability  # noqa: E402
from scorescan.util import atomic_write_json  # noqa: E402

MODEL_VERSION = "scorescan-barline-sequence-gbdt-2"
OLD_RUNTIME_POLICY = {
    "probability_floor": 0.30,
    "hard_reject_floor": 0.04,
    "local_override": 0.94,
    "short_gap_ratio": 0.72,
    "merged_gap_deviation": 0.28,
    "edge_distance_floor": 0.35,
    "candidate_density_floor": 1.10,
    "probability_margin_ceiling": -0.10,
}

System = tuple[int, int, float, list[tuple[float, int]], dict[int, int]]
GroupedSystem = tuple[int, System]
Predictor = Callable[[BarlineSequenceFeatures], float]


def _system(rng: random.Random) -> System:
    spacing = rng.uniform(8.0, 24.0)
    measure_count = rng.randint(3, 10)
    typical = rng.uniform(spacing * 12.0, spacing * 22.0)
    widths = [typical * rng.uniform(0.76, 1.25) for _ in range(measure_count)]
    # Genuine narrow or broad interior measures prevent the model from equating all
    # irregular spacing with a false split.
    if measure_count >= 5 and rng.random() < 0.12:
        interior = rng.randrange(1, measure_count - 1)
        widths[interior] *= rng.choice((rng.uniform(0.58, 0.76), rng.uniform(1.30, 1.58)))
    if rng.random() < 0.28:
        widths[0] *= rng.uniform(0.46, 0.78)
    if rng.random() < 0.25:
        widths[-1] *= rng.uniform(0.56, 0.88)

    left = rng.randint(40, 120)
    true_positions: list[int] = []
    cursor = float(left)
    for width in widths[:-1]:
        cursor += width
        true_positions.append(int(round(cursor)))
    right = int(round(left + sum(widths)))

    candidates: list[tuple[float, int]] = []
    labels: dict[int, int] = {}
    true_probabilities: dict[int, float] = {}
    for x in true_positions:
        probability = rng.uniform(0.62, 0.86) if rng.random() < 0.18 else rng.uniform(0.78, 0.997)
        candidates.append((probability, x))
        labels[x] = 1
        true_probabilities[x] = probability

    edges = [left, *true_positions, right]
    occupied = set(true_positions)
    # Known real-scan failure mode: a full-height opening stem can split the first
    # measure while retaining moderately high local confidence.  Generate this as a
    # family of grouped hard negatives rather than special-casing any page or x value.
    if true_positions and rng.random() < 0.32:
        first_boundary = true_positions[0]
        first_width = first_boundary - left
        fraction = rng.uniform(0.43, 0.61)
        x = int(round(left + first_width * fraction))
        if x not in occupied and left + spacing < x < first_boundary - spacing:
            neighbour_probability = true_probabilities[first_boundary]
            probability = max(0.58, min(0.92, neighbour_probability - rng.uniform(0.045, 0.16)))
            candidates.append((probability, x))
            labels[x] = 0
            occupied.add(x)

    false_count = 0 if rng.random() < 0.18 else rng.randint(1, max(2, measure_count // 2 + 2))
    for _ in range(false_count):
        interval = rng.randrange(measure_count)
        a, b = edges[interval], edges[interval + 1]
        if b - a < spacing * 5.0:
            continue
        kind = rng.choices(
            ("split", "random", "near_true"),
            weights=(0.58, 0.20, 0.22),
            k=1,
        )[0]
        if kind == "split":
            fraction = rng.uniform(0.22, 0.78)
            x = int(round(a + (b - a) * fraction))
        elif kind == "near_true" and true_positions:
            anchor = rng.choice(true_positions)
            x = int(round(anchor + rng.choice((-1, 1)) * rng.uniform(spacing * 0.75, spacing * 2.5)))
        else:
            margin = max(2, int(spacing))
            x = rng.randint(a + margin, b - margin)
        if x in occupied or not (left + spacing < x < right - spacing):
            continue
        occupied.add(x)
        probability = rng.uniform(0.82, 0.98) if rng.random() < 0.22 else rng.uniform(0.54, 0.93)
        candidates.append((probability, x))
        labels[x] = 0

    # The sequence layer cannot invent missing boundaries.  Such cases remain in the
    # dataset so surviving true candidates are not over-pruned in incomplete sequences.
    if true_positions and rng.random() < 0.18:
        removed = rng.choice(true_positions)
        candidates = [item for item in candidates if item[1] != removed]
        labels.pop(removed, None)

    candidates.sort(key=lambda item: item[1])
    return left, right, spacing, candidates, labels


def build_systems(seed: int, groups: int, variants_per_group: int) -> list[GroupedSystem]:
    master = random.Random(seed)
    result: list[GroupedSystem] = []
    for group in range(groups):
        group_seed = master.randrange(1 << 30)
        for variant in range(variants_per_group):
            result.append((group, _system(random.Random(group_seed + variant * 8191))))
    return result


def build_dataset(systems: list[GroupedSystem]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[tuple[float, ...]] = []
    labels: list[int] = []
    group_ids: list[int] = []
    for group, (left, right, spacing, candidates, truth) in systems:
        for index, (_probability, x) in enumerate(candidates):
            features = extract_sequence_features(
                left=left,
                right=right,
                spacing=spacing,
                candidates=candidates,
                index=index,
            )
            rows.append(features.vector())
            labels.append(truth[x])
            group_ids.append(group)
    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(group_ids, dtype=np.int64),
    )


def _serialize_trees(model: GradientBoostingClassifier) -> list[dict[str, object]]:
    trees: list[dict[str, object]] = []
    for estimator in model.estimators_.ravel():
        tree = estimator.tree_
        nodes: list[dict[str, object]] = []
        for node_index in range(tree.node_count):
            nodes.append(
                {
                    "feature": int(tree.feature[node_index]),
                    "threshold": float(tree.threshold[node_index]),
                    "left": int(tree.children_left[node_index]),
                    "right": int(tree.children_right[node_index]),
                    "value": float(tree.value[node_index].ravel()[0]),
                }
            )
        trees.append({"nodes": nodes})
    return trees


def _payload_predictor(payload: dict[str, object]) -> Predictor:
    return lambda features: gradient_boosting_probability(
        features.vector(),
        intercept=float(payload.get("intercept", 0.0)),
        learning_rate=float(payload.get("learning_rate", 0.0)),
        trees=payload.get("trees", ()),
        calibration_intercept=float(payload.get("calibration_intercept", 0.0)),
        calibration_slope=float(payload.get("calibration_slope", 1.0)),
    )


def _legacy_predictor(payload: dict[str, object] | None) -> Predictor | None:
    if not payload or tuple(payload.get("feature_names", ())) != LEGACY_FEATURE_NAMES:
        return None
    coefficients = tuple(float(value) for value in payload.get("coefficients", ()))
    means = tuple(float(value) for value in payload.get("means", ()))
    scales = tuple(max(float(value), 1e-9) for value in payload.get("scales", ()))
    if not (len(coefficients) == len(means) == len(scales) == len(LEGACY_FEATURE_NAMES)):
        return None
    intercept = float(payload.get("intercept", 0.0))

    def predict(features: BarlineSequenceFeatures) -> float:
        standardized = [
            (value - mean) / scale
            for value, mean, scale in zip(features.legacy_vector(), means, scales, strict=True)
        ]
        score = intercept + sum(
            coefficient * value
            for coefficient, value in zip(coefficients, standardized, strict=True)
        )
        if score >= 0:
            return 1.0 / (1.0 + math.exp(-min(score, 40.0)))
        exp_score = math.exp(max(score, -40.0))
        return exp_score / (1.0 + exp_score)

    return predict


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predicted = probabilities >= 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predicted,
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _split_like(features: BarlineSequenceFeatures, policy: dict[str, float]) -> bool:
    return (
        features.left_gap_ratio <= policy["short_gap_ratio"]
        and features.right_gap_ratio <= policy["short_gap_ratio"]
        and features.merged_gap_deviation <= policy["merged_gap_deviation"]
        and features.edge_distance_ratio >= policy["edge_distance_floor"]
        and features.candidate_density_ratio >= policy["candidate_density_floor"]
        and features.probability_margin <= policy["probability_margin_ceiling"]
    )


def _refine(
    system: System,
    predictor: Predictor,
    policy: dict[str, float],
    *,
    iterative: bool,
    max_iterations: int = 8,
) -> set[int]:
    left, right, spacing, candidates, _truth = system
    current = list(candidates)
    limit = max_iterations if iterative else 1
    for _ in range(limit):
        proposals: list[tuple[float, int, int]] = []
        for index, (local_probability, x) in enumerate(current):
            features = extract_sequence_features(
                left=left,
                right=right,
                spacing=spacing,
                candidates=current,
                index=index,
            )
            probability = predictor(features)
            split_like = _split_like(features, policy)
            reject = probability < policy["probability_floor"]
            local_allows = (
                local_probability < policy["local_override"]
                or probability < policy["hard_reject_floor"]
            )
            opening_reject = (
                "opening_probability_floor" in policy
                and index == 0
                and features.normalised_position <= policy["opening_position_ceiling"]
                and features.left_gap_ratio <= policy["opening_gap_ratio_ceiling"]
                and features.right_gap_ratio <= policy["opening_gap_ratio_ceiling"]
                and features.merged_gap_deviation <= policy["opening_merged_gap_deviation"]
                and features.probability_margin <= policy["opening_margin_ceiling"]
                and features.removal_regularity_gain >= policy["opening_regularity_gain_floor"]
                and local_probability < policy["opening_local_ceiling"]
                and probability < policy["opening_probability_floor"]
            )
            if split_like and ((reject and local_allows) or opening_reject):
                priority = probability + 0.15 * features.merged_gap_deviation + 0.04 * local_probability
                proposals.append((priority, x, index))
        if not proposals:
            break
        if iterative:
            _priority, _x, index = min(proposals)
            current.pop(index)
        else:
            rejected = {item[1] for item in proposals}
            current = [item for item in current if item[1] not in rejected]
            break
    return {x for _probability, x in current}


def _sequence_metrics(
    systems: list[GroupedSystem],
    predictor: Predictor,
    policy: dict[str, float],
    *,
    iterative: bool,
) -> dict[str, float | int]:
    baseline_exact = refined_exact = 0
    true_total = true_retained = false_total = false_removed = 0
    helped = harmed = 0
    for _group, system in systems:
        _left, _right, _spacing, candidates, truth = system
        true_count = sum(truth.values()) + 1
        baseline_ok = len(candidates) + 1 == true_count
        baseline_exact += int(baseline_ok)
        retained = _refine(system, predictor, policy, iterative=iterative)
        refined_ok = len(retained) + 1 == true_count
        refined_exact += int(refined_ok)
        helped += int(not baseline_ok and refined_ok)
        harmed += int(baseline_ok and not refined_ok)
        for x, label in truth.items():
            if label:
                true_total += 1
                true_retained += int(x in retained)
            else:
                false_total += 1
                false_removed += int(x not in retained)
    count = max(len(systems), 1)
    return {
        "systems": len(systems),
        "baseline_exact_measure_count_rate": baseline_exact / count,
        "refined_exact_measure_count_rate": refined_exact / count,
        "true_boundary_retention": true_retained / max(true_total, 1),
        "false_candidate_removal": false_removed / max(false_total, 1),
        "helped_system_rate": helped / count,
        "harmed_system_rate": harmed / count,
        "harmed_systems": harmed,
    }


def _runtime_policy() -> dict[str, float]:
    return {
        "probability_floor": DEFAULT_POLICY.barline_sequence_probability_floor,
        "hard_reject_floor": DEFAULT_POLICY.barline_sequence_hard_reject_floor,
        "local_override": DEFAULT_POLICY.barline_sequence_local_override,
        "short_gap_ratio": DEFAULT_POLICY.barline_sequence_short_gap_ratio,
        "merged_gap_deviation": DEFAULT_POLICY.barline_sequence_merged_gap_deviation,
        "edge_distance_floor": DEFAULT_POLICY.barline_sequence_edge_distance_floor,
        "candidate_density_floor": DEFAULT_POLICY.barline_sequence_candidate_density_floor,
        "probability_margin_ceiling": DEFAULT_POLICY.barline_sequence_probability_margin_ceiling,
        "opening_probability_floor": DEFAULT_POLICY.barline_sequence_opening_probability_floor,
        "opening_position_ceiling": DEFAULT_POLICY.barline_sequence_opening_position_ceiling,
        "opening_local_ceiling": DEFAULT_POLICY.barline_sequence_opening_local_ceiling,
        "opening_gap_ratio_ceiling": DEFAULT_POLICY.barline_sequence_opening_gap_ratio_ceiling,
        "opening_margin_ceiling": DEFAULT_POLICY.barline_sequence_opening_margin_ceiling,
        "opening_regularity_gain_floor": DEFAULT_POLICY.barline_sequence_opening_regularity_gain_floor,
        "opening_merged_gap_deviation": DEFAULT_POLICY.barline_sequence_opening_merged_gap_deviation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "src" / "scorescan" / "resources" / "barline_sequence_classifier.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT.parent / "training" / "barline_sequence_classifier_report_v2.json",
    )
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=ROOT.parent / "training" / "baselines" / "barline_sequence_classifier_v1.json",
    )
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--groups", type=int, default=2200)
    parser.add_argument("--variants-per-group", type=int, default=2)
    args = parser.parse_args()

    baseline_path = args.baseline_model
    try:
        baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        if not isinstance(baseline_payload, dict):
            baseline_payload = None
    except (OSError, json.JSONDecodeError):
        baseline_payload = None

    systems = build_systems(args.seed, args.groups, args.variants_per_group)
    x, y, group_ids = build_dataset(systems)
    first = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=args.seed)
    train_indices, holdout_indices = next(first.split(x, y, group_ids))
    second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=args.seed + 1)
    calibration_local, test_local = next(
        second.split(x[holdout_indices], y[holdout_indices], group_ids[holdout_indices])
    )
    calibration_indices = holdout_indices[calibration_local]
    test_indices = holdout_indices[test_local]
    calibration_groups = set(int(value) for value in group_ids[calibration_indices])
    test_groups = set(int(value) for value in group_ids[test_indices])
    calibration_systems = [item for item in systems if item[0] in calibration_groups]
    test_systems = [item for item in systems if item[0] in test_groups]

    model = GradientBoostingClassifier(
        n_estimators=60,
        learning_rate=0.055,
        max_depth=2,
        min_samples_leaf=18,
        subsample=0.90,
        random_state=args.seed,
    )
    train_x = x[train_indices]
    train_y = y[train_indices]
    sample_weight = np.where(train_y == 0, 1.25, 1.0)
    model.fit(train_x, train_y, sample_weight=sample_weight)

    raw_calibration = model.decision_function(x[calibration_indices]).reshape(-1, 1)
    calibrator = LogisticRegression(C=1000.0, max_iter=1000, random_state=args.seed)
    calibrator.fit(raw_calibration, y[calibration_indices])

    initial_probability = float(model.init_.predict_proba(x[:1])[0, 1])
    raw_intercept = math.log(
        max(initial_probability, 1e-12) / max(1.0 - initial_probability, 1e-12)
    )
    payload: dict[str, object] = {
        "model_version": MODEL_VERSION,
        "model_type": "gradient_boosting",
        "feature_names": list(FEATURE_NAMES),
        "intercept": raw_intercept,
        "learning_rate": float(model.learning_rate),
        "trees": _serialize_trees(model),
        "calibration_intercept": float(calibrator.intercept_[0]),
        "calibration_slope": float(calibrator.coef_[0, 0]),
        "training_seed": args.seed,
        "training_groups": args.groups,
        "variants_per_group": args.variants_per_group,
        "target_definition": "true staff-system boundary versus inserted stem/noise candidate",
    }

    sklearn_probabilities = calibrator.predict_proba(
        model.decision_function(x[test_indices]).reshape(-1, 1)
    )[:, 1]
    deployed_probabilities = np.asarray(
        [
            gradient_boosting_probability(
                row,
                intercept=raw_intercept,
                learning_rate=float(model.learning_rate),
                trees=payload["trees"],
                calibration_intercept=float(calibrator.intercept_[0]),
                calibration_slope=float(calibrator.coef_[0, 0]),
            )
            for row in x[test_indices]
        ],
        dtype=np.float64,
    )
    deployment_delta = float(np.max(np.abs(sklearn_probabilities - deployed_probabilities)))
    if deployment_delta > 1e-10:
        raise RuntimeError(f"部署推理与训练推理不一致：{deployment_delta}")

    predictor = _payload_predictor(payload)
    baseline_predictor = _legacy_predictor(baseline_payload)
    runtime_policy = _runtime_policy()
    report: dict[str, object] = {
        "model_version": MODEL_VERSION,
        "seed": args.seed,
        "groups": args.groups,
        "variants_per_group": args.variants_per_group,
        "samples": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "train_samples": int(len(train_indices)),
        "calibration_samples": int(len(calibration_indices)),
        "test_samples": int(len(test_indices)),
        "feature_names": list(FEATURE_NAMES),
        "test": _metrics(y[test_indices], deployed_probabilities),
        "deployment_parity": {
            "max_absolute_probability_delta": deployment_delta,
            "samples": int(len(test_indices)),
        },
        "runtime_policy": runtime_policy,
        "sequence_test": _sequence_metrics(
            test_systems,
            predictor,
            runtime_policy,
            iterative=True,
        ),
        "sequence_calibration": _sequence_metrics(
            calibration_systems,
            predictor,
            runtime_policy,
            iterative=True,
        ),
        "scope": "grouped synthetic staff-system boundary refinement; not end-to-end OMR accuracy",
    }
    if baseline_predictor is not None:
        baseline_probabilities = np.asarray(
            [
                baseline_predictor(
                    extract_sequence_features(
                        left=system[0],
                        right=system[1],
                        spacing=system[2],
                        candidates=system[3],
                        index=index,
                    )
                )
                for _group, system in test_systems
                for index in range(len(system[3]))
            ],
            dtype=np.float64,
        )
        baseline_labels = np.asarray(
            [
                system[4][x]
                for _group, system in test_systems
                for _probability, x in system[3]
            ],
            dtype=np.int64,
        )
        report["baseline_v1_on_same_test"] = {
            "model_version": str(baseline_payload.get("model_version", "unknown")),
            "test": _metrics(baseline_labels, baseline_probabilities),
            "sequence_test": _sequence_metrics(
                test_systems,
                baseline_predictor,
                OLD_RUNTIME_POLICY,
                iterative=False,
            ),
        }

    atomic_write_json(args.output, payload)
    atomic_write_json(args.output.parent / "model_manifest.json", build_manifest(args.output.parent))
    atomic_write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
