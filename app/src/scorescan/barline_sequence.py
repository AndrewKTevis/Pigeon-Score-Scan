from __future__ import annotations

"""Conservative global refinement for staff-spanning barline candidates.

The local classifier asks whether one vertical stroke resembles a barline.  This module
asks whether that stroke belongs in the complete boundary sequence across a staff
system.  Full-height note stems and scan artefacts often create two short intervals
inside one otherwise plausible measure.  A compact CPU model ranks those candidates,
while deterministic geometry remains an independent removal gate.

Refinement is iterative: at most one candidate is removed per pass and all neighbourhood
features are then recomputed.  This resolves clusters of false stems without making a
single stale decision against the original over-segmented sequence.  Missing boundaries
are never invented.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .policy import DEFAULT_POLICY
from .tree_model import VerifiedGradientBoostingModel


LEGACY_FEATURE_NAMES = (
    "local_probability",
    "normalised_position",
    "edge_distance_ratio",
    "left_gap_ratio",
    "right_gap_ratio",
    "minimum_gap_ratio",
    "maximum_gap_ratio",
    "merged_gap_ratio",
    "merged_gap_deviation",
    "gap_balance",
    "left_neighbour_probability",
    "right_neighbour_probability",
    "probability_margin",
    "candidate_density_ratio",
    "both_adjacent_short",
    "near_system_edge",
)

SHORT_GAP_FEATURE_THRESHOLD = 0.72

FEATURE_NAMES = LEGACY_FEATURE_NAMES + (
    "split_fraction",
    "left_outer_gap_ratio",
    "right_outer_gap_ratio",
    "merged_outer_deviation",
    "removal_regularity_gain",
    "local_probability_percentile",
    "neighbour_probability_min",
    "neighbour_probability_max",
    "candidate_count_scaled",
    "typical_gap_staff_spaces",
)


@dataclass(frozen=True)
class BarlineSequenceFeatures:
    local_probability: float
    normalised_position: float
    edge_distance_ratio: float
    left_gap_ratio: float
    right_gap_ratio: float
    minimum_gap_ratio: float
    maximum_gap_ratio: float
    merged_gap_ratio: float
    merged_gap_deviation: float
    gap_balance: float
    left_neighbour_probability: float
    right_neighbour_probability: float
    probability_margin: float
    candidate_density_ratio: float
    both_adjacent_short: float
    near_system_edge: float
    split_fraction: float
    left_outer_gap_ratio: float
    right_outer_gap_ratio: float
    merged_outer_deviation: float
    removal_regularity_gain: float
    local_probability_percentile: float
    neighbour_probability_min: float
    neighbour_probability_max: float
    candidate_count_scaled: float
    typical_gap_staff_spaces: float

    def vector(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, name)) for name in FEATURE_NAMES)

    def legacy_vector(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, name)) for name in LEGACY_FEATURE_NAMES)


@dataclass(frozen=True)
class RefinedBarline:
    x: int
    local_probability: float
    sequence_probability: float
    final_probability: float
    retained: bool
    split_like: bool


def _clip(value: float, lower: float = 0.0, upper: float = 4.0) -> float:
    return max(lower, min(upper, float(value)))


def _typical_gap(left: int, right: int, candidates: list[tuple[float, int]], spacing: float) -> float:
    positions = [int(left), *(int(x) for _probability, x in candidates), int(right)]
    gaps = np.diff(np.asarray(positions, dtype=np.float64))
    minimum = max(1.0, float(spacing) * 3.8)
    usable = np.sort(gaps[gaps >= minimum])
    if usable.size == 0:
        return max(minimum, (right - left) / max(len(candidates) + 1, 1))
    # Inserted stems create short split intervals.  The upper half is more stable than
    # the ordinary median while still allowing pickups and compressed final measures.
    upper = usable[usable.size // 2 :]
    return max(minimum, float(np.median(upper)))


def extract_sequence_features(
    *,
    left: int,
    right: int,
    spacing: float,
    candidates: Iterable[tuple[float, int]],
    index: int,
) -> BarlineSequenceFeatures:
    ordered = sorted(((float(probability), int(x)) for probability, x in candidates), key=lambda item: item[1])
    if not ordered or not (0 <= index < len(ordered)):
        raise IndexError("barline sequence candidate index out of range")
    width = max(1.0, float(right - left))
    typical = _typical_gap(left, right, ordered, spacing)
    probability, x = ordered[index]
    previous_x = int(left) if index == 0 else ordered[index - 1][1]
    next_x = int(right) if index + 1 == len(ordered) else ordered[index + 1][1]
    previous_previous_x = int(left) if index <= 1 else ordered[index - 2][1]
    next_next_x = int(right) if index + 2 >= len(ordered) else ordered[index + 2][1]

    left_gap = max(1.0, float(x - previous_x))
    right_gap = max(1.0, float(next_x - x))
    left_ratio = left_gap / typical
    right_ratio = right_gap / typical
    merged_gap = left_gap + right_gap
    merged_ratio = merged_gap / typical
    minimum_ratio = min(left_ratio, right_ratio)
    maximum_ratio = max(left_ratio, right_ratio)
    left_probability = 1.0 if index == 0 else ordered[index - 1][0]
    right_probability = 1.0 if index + 1 == len(ordered) else ordered[index + 1][0]
    neighbour_mean = 0.5 * (left_probability + right_probability)
    normalised_position = (x - left) / width
    expected_candidate_count = max(1.0, width / typical - 1.0)
    candidate_density = len(ordered) / expected_candidate_count

    left_outer_gap = typical if index == 0 else max(1.0, float(previous_x - previous_previous_x))
    right_outer_gap = typical if index + 1 == len(ordered) else max(1.0, float(next_next_x - next_x))
    outer_reference = float(np.median((left_outer_gap, right_outer_gap, typical)))
    regularity_before = abs(left_ratio - 1.0) + abs(right_ratio - 1.0)
    regularity_after = abs(merged_ratio - 1.0)
    probabilities = np.asarray([item[0] for item in ordered], dtype=np.float64)
    percentile = (
        float(np.count_nonzero(probabilities < probability))
        + 0.5 * float(np.count_nonzero(probabilities == probability))
    ) / max(len(ordered), 1)

    return BarlineSequenceFeatures(
        local_probability=_clip(probability, 0.0, 1.0),
        normalised_position=_clip(normalised_position, 0.0, 1.0),
        edge_distance_ratio=_clip(min(x - left, right - x) / typical),
        left_gap_ratio=_clip(left_ratio),
        right_gap_ratio=_clip(right_ratio),
        minimum_gap_ratio=_clip(minimum_ratio),
        maximum_gap_ratio=_clip(maximum_ratio),
        merged_gap_ratio=_clip(merged_ratio),
        merged_gap_deviation=_clip(abs(merged_ratio - 1.0)),
        gap_balance=_clip(minimum_ratio / max(maximum_ratio, 1e-6), 0.0, 1.0),
        left_neighbour_probability=_clip(left_probability, 0.0, 1.0),
        right_neighbour_probability=_clip(right_probability, 0.0, 1.0),
        probability_margin=_clip(probability - neighbour_mean, -1.0, 1.0),
        candidate_density_ratio=_clip(candidate_density),
        both_adjacent_short=float(
            left_ratio <= SHORT_GAP_FEATURE_THRESHOLD
            and right_ratio <= SHORT_GAP_FEATURE_THRESHOLD
        ),
        near_system_edge=float(normalised_position <= 0.035 or normalised_position >= 0.965),
        split_fraction=_clip(min(left_gap, right_gap) / max(merged_gap, 1.0), 0.0, 0.5),
        left_outer_gap_ratio=_clip(left_outer_gap / typical),
        right_outer_gap_ratio=_clip(right_outer_gap / typical),
        merged_outer_deviation=_clip(abs(merged_gap / max(outer_reference, 1.0) - 1.0)),
        removal_regularity_gain=_clip(regularity_before - regularity_after, -4.0, 4.0),
        local_probability_percentile=_clip(percentile, 0.0, 1.0),
        neighbour_probability_min=_clip(min(left_probability, right_probability), 0.0, 1.0),
        neighbour_probability_max=_clip(max(left_probability, right_probability), 0.0, 1.0),
        candidate_count_scaled=_clip(len(ordered) / 12.0, 0.0, 2.0),
        typical_gap_staff_spaces=_clip(typical / max(float(spacing), 1.0) / 40.0, 0.0, 1.0),
    )


class BarlineSequenceClassifier:
    def __init__(self, model_path: Path | None = None) -> None:
        path = model_path or Path(__file__).resolve().parent / "resources" / "barline_sequence_classifier.json"
        self.model = VerifiedGradientBoostingModel.load(path, "barline_sequence_classification", FEATURE_NAMES)

    @property
    def enabled(self) -> bool:
        return self.model.enabled

    @property
    def model_version(self) -> str:
        return self.model.model_version

    @property
    def status(self) -> str:
        return self.model.status

    def predict(self, features: BarlineSequenceFeatures) -> float:
        return self.model.predict(features.vector(), neutral=0.5)


def _split_like(features: BarlineSequenceFeatures) -> bool:
    return (
        features.left_gap_ratio <= DEFAULT_POLICY.barline_sequence_short_gap_ratio
        and features.right_gap_ratio <= DEFAULT_POLICY.barline_sequence_short_gap_ratio
        and features.merged_gap_deviation <= DEFAULT_POLICY.barline_sequence_merged_gap_deviation
        and features.edge_distance_ratio >= DEFAULT_POLICY.barline_sequence_edge_distance_floor
        and features.candidate_density_ratio >= DEFAULT_POLICY.barline_sequence_candidate_density_floor
        and features.probability_margin <= DEFAULT_POLICY.barline_sequence_probability_margin_ceiling
    )


def _opening_split_like(
    features: BarlineSequenceFeatures,
    *,
    index: int,
    local_probability: float,
    sequence_probability: float,
) -> bool:
    """Return whether a candidate matches the conservative opening-stem signature."""
    return (
        index == 0
        and features.normalised_position <= DEFAULT_POLICY.barline_sequence_opening_position_ceiling
        and features.left_gap_ratio <= DEFAULT_POLICY.barline_sequence_opening_gap_ratio_ceiling
        and features.right_gap_ratio <= DEFAULT_POLICY.barline_sequence_opening_gap_ratio_ceiling
        and features.merged_gap_deviation <= DEFAULT_POLICY.barline_sequence_opening_merged_gap_deviation
        and features.probability_margin <= DEFAULT_POLICY.barline_sequence_opening_margin_ceiling
        and features.removal_regularity_gain >= DEFAULT_POLICY.barline_sequence_opening_regularity_gain_floor
        and local_probability < DEFAULT_POLICY.barline_sequence_opening_local_ceiling
        and sequence_probability < DEFAULT_POLICY.barline_sequence_opening_probability_floor
    )


def refine_barline_sequence(
    *,
    left: int,
    right: int,
    spacing: float,
    candidates: Iterable[tuple[float, int]],
    classifier: BarlineSequenceClassifier | None = None,
) -> tuple[RefinedBarline, ...]:
    """Return globally refined candidates without inventing missing boundaries.

    Removal is two-key: the calibrated probability must be low and deterministic
    geometry must show that deleting the candidate merges two short intervals into one
    plausible measure.  Only the strongest safe rejection is applied per iteration;
    neighbourhood features are then recomputed before any further deletion.
    """

    original = sorted(((float(probability), int(x)) for probability, x in candidates), key=lambda item: item[1])
    if not original:
        return ()
    model = classifier or BarlineSequenceClassifier()
    current = list(original)
    removed: dict[int, tuple[float, bool]] = {}

    for _iteration in range(max(1, DEFAULT_POLICY.barline_sequence_max_iterations)):
        if not current or not model.enabled:
            break
        proposals: list[tuple[float, int, int, float]] = []
        for index, (local_probability, x) in enumerate(current):
            features = extract_sequence_features(
                left=left,
                right=right,
                spacing=spacing,
                candidates=current,
                index=index,
            )
            sequence_probability = model.predict(features)
            split_like = _split_like(features)
            learned_reject = sequence_probability < DEFAULT_POLICY.barline_sequence_probability_floor
            overwhelming_reject = sequence_probability < DEFAULT_POLICY.barline_sequence_hard_reject_floor
            local_allows_reject = (
                local_probability < DEFAULT_POLICY.barline_sequence_local_override
                or overwhelming_reject
            )
            opening_reject = _opening_split_like(
                features,
                index=index,
                local_probability=local_probability,
                sequence_probability=sequence_probability,
            )
            if split_like and ((learned_reject and local_allows_reject) or opening_reject):
                # Stable deterministic ordering.  Lower model probability dominates;
                # geometry and local confidence only break close ties.
                priority = (
                    sequence_probability
                    + 0.15 * features.merged_gap_deviation
                    + 0.04 * local_probability
                )
                proposals.append((priority, x, index, sequence_probability))
        if not proposals:
            break
        _priority, x, index, sequence_probability = min(proposals)
        removed[x] = (sequence_probability, True)
        current.pop(index)

    retained_evidence: dict[int, tuple[float, bool]] = {}
    for index, (_local_probability, x) in enumerate(current):
        features = extract_sequence_features(
            left=left,
            right=right,
            spacing=spacing,
            candidates=current,
            index=index,
        )
        retained_evidence[x] = (model.predict(features) if model.enabled else 0.5, _split_like(features))

    original_features = {
        x: extract_sequence_features(
            left=left,
            right=right,
            spacing=spacing,
            candidates=original,
            index=index,
        )
        for index, (_local_probability, x) in enumerate(original)
    }
    results: list[RefinedBarline] = []
    for local_probability, x in original:
        retained = x not in removed
        sequence_probability, split_like = (
            retained_evidence.get(x, (0.5, False)) if retained else removed[x]
        )
        initial_features = original_features[x]
        final_probability = (
            local_probability
            if initial_features.near_system_edge > 0.5
            else 0.72 * local_probability + 0.28 * sequence_probability
        )
        results.append(
            RefinedBarline(
                x=x,
                local_probability=round(local_probability, 6),
                sequence_probability=round(sequence_probability, 6),
                final_probability=round(max(0.0, min(1.0, final_probability)), 6),
                retained=retained,
                split_like=split_like,
            )
        )
    return tuple(results)
