from __future__ import annotations

"""Deterministic grouped data for visual-to-MusicXML direction anchoring.

Each source group is one synthetic engraved staff system.  Related text marks and all
candidate target measures stay in the same train/calibration/audit/test partition.
The product feature extractor is used directly; this module does not duplicate runtime
geometry logic.
"""

import random
from dataclasses import dataclass

import numpy as np

from scorescan.direction_anchor import (
    ANCHOR_FEATURE_NAMES,
    anchor_candidate_feature_vector,
    measure_anchor_candidate_indices,
)
from scorescan.layout import StaffSystem, anchor_x_to_measure

KINDS = ("dynamic", "metronome", "direction", "text")
PLACEMENTS = ("above", "below")
SCENARIOS = ("missing_boundary", "extra_boundary", "mixed_boundaries")


@dataclass(frozen=True)
class AnchorDecision:
    candidate_rows: tuple[int, ...]
    candidate_indices: tuple[int, ...]
    baseline_index: int
    true_index: int
    source_group: int
    scenario: str
    kind: str
    placement: str


@dataclass(frozen=True)
class DirectionMeasureAnchorDataset:
    features: np.ndarray
    labels: np.ndarray
    group_ids: np.ndarray
    decisions: tuple[AnchorDecision, ...]


def _true_widths(rng: random.Random, count: int) -> list[float]:
    base = rng.uniform(175.0, 285.0)
    widths = [base * rng.uniform(0.68, 1.34) for _ in range(count)]
    if count >= 3 and rng.random() < 0.58:
        widths[0] *= rng.uniform(0.42, 0.76)  # pickup or compact opening
    if count >= 3 and rng.random() < 0.46:
        widths[-1] *= rng.uniform(0.50, 0.82)  # compact final measure
    if count >= 5 and rng.random() < 0.42:
        index = rng.randrange(1, count - 1)
        widths[index] *= rng.uniform(1.28, 1.68)  # direction/ornament-rich measure
    return [max(90.0, value) for value in widths]


def _boundaries(left: float, widths: list[float]) -> list[float]:
    result = [left]
    for width in widths:
        result.append(result[-1] + width)
    return result


def _corrupt_boundaries(
    rng: random.Random,
    true_boundaries: list[float],
    spacing: float,
    scenario: str,
) -> tuple[list[float], set[int], set[int]]:
    """Return observed internal boundaries plus affected true measures.

    Missing boundary index ``b`` separates true measures ``b-1`` and ``b``.  Extra
    boundaries are inserted inside one true measure.  The affected sets are used only
    to sample informative marks; labels are always derived from ground truth geometry.
    """

    target_count = len(true_boundaries) - 1
    internal = list(true_boundaries[1:-1])
    missing_measures: set[int] = set()
    extra_measures: set[int] = set()

    if scenario in {"missing_boundary", "mixed_boundaries"} and internal:
        remove_count = 1 + int(target_count >= 8 and rng.random() < 0.28)
        available = list(range(1, target_count))
        rng.shuffle(available)
        removed = sorted(available[: min(remove_count, len(available))], reverse=True)
        for boundary_index in removed:
            del internal[boundary_index - 1]
            missing_measures.update({boundary_index - 1, boundary_index})

    if scenario in {"extra_boundary", "mixed_boundaries"}:
        add_count = 1 + int(target_count >= 7 and rng.random() < 0.30)
        measures = list(range(target_count))
        rng.shuffle(measures)
        for measure_index in measures[:add_count]:
            left = true_boundaries[measure_index]
            right = true_boundaries[measure_index + 1]
            ratio = rng.uniform(0.28, 0.72)
            value = left + ratio * (right - left)
            # Keep the proposal semantically distinct from a thick true boundary.
            if min(value - left, right - value) >= max(spacing * 1.6, 18.0):
                internal.append(value)
                extra_measures.add(measure_index)

    jitter = max(0.4, spacing * 0.12)
    observed = [value + rng.uniform(-jitter, jitter) for value in internal]
    observed.sort()
    return observed, missing_measures, extra_measures


def _mark_ratio(rng: random.Random, kind: str) -> float:
    if kind == "metronome":
        return rng.betavariate(1.5, 5.2)
    if kind == "direction":
        return rng.betavariate(1.7, 3.9)
    if kind == "dynamic":
        return rng.betavariate(2.0, 2.8)
    return rng.betavariate(2.1, 2.2)


def _sample_system(group_id: int, seed: int) -> tuple[StaffSystem, list[float], str, set[int]]:
    rng = random.Random(seed)
    target_count = rng.randint(3, 12)
    spacing = rng.uniform(10.0, 18.0)
    left = rng.uniform(70.0, 150.0)
    widths = _true_widths(rng, target_count)
    true_boundaries = _boundaries(left, widths)
    scenario = rng.choice(SCENARIOS)
    observed, missing, extra = _corrupt_boundaries(rng, true_boundaries, spacing, scenario)
    # Extremely rare failed extra insertion must still produce a mismatch.
    if len(observed) + 1 == target_count:
        measure = rng.randrange(target_count)
        lft, rgt = true_boundaries[measure], true_boundaries[measure + 1]
        observed.append(lft + 0.5 * (rgt - lft))
        observed.sort()
        extra.add(measure)
        scenario = "extra_boundary" if not missing else "mixed_boundaries"
    system = StaffSystem(
        index=group_id + 1,
        line_y=[300.0 + spacing * index for index in range(5)],
        top=int(300.0 - spacing * 4.0),
        bottom=int(300.0 + spacing * 8.0),
        left=int(round(left)),
        right=int(round(true_boundaries[-1])),
        spacing=spacing,
        barlines=[int(round(value)) for value in observed],
        measure_count=len(observed) + 1,
    )
    affected = missing | extra
    return system, true_boundaries, scenario, affected


def build_dataset(seed: int = 20270112, system_groups: int = 3200) -> DirectionMeasureAnchorDataset:
    rows: list[tuple[float, ...]] = []
    labels: list[int] = []
    groups: list[int] = []
    decisions: list[AnchorDecision] = []
    master = random.Random(seed)

    for group_id in range(system_groups):
        group_seed = master.randrange(1 << 62)
        system, true_boundaries, scenario, affected = _sample_system(group_id, group_seed)
        rng = random.Random(group_seed ^ 0x5A17C0DE)
        target_count = len(true_boundaries) - 1
        mark_count = rng.randint(5, 9)
        for mark_number in range(mark_count):
            kind = rng.choice(KINDS)
            placement = rng.choice(PLACEMENTS)
            if affected and mark_number < max(2, mark_count // 2):
                true_index = rng.choice(sorted(affected))
            else:
                true_index = rng.randrange(target_count)
            ratio = _mark_ratio(rng, kind)
            # Include boundary-adjacent directions because these are the most costly
            # ownership ambiguities in MusicXML.
            if rng.random() < 0.18:
                ratio = rng.choice((rng.uniform(0.01, 0.08), rng.uniform(0.90, 0.985)))
            x = true_boundaries[true_index] + ratio * (
                true_boundaries[true_index + 1] - true_boundaries[true_index]
            )
            x += rng.uniform(-0.20, 0.20) * system.spacing
            x = max(float(system.left) + 1e-3, min(float(system.right) - 1e-3, x))
            baseline = anchor_x_to_measure(system, x, target_count)
            candidate_rows: list[int] = []
            candidate_indices = measure_anchor_candidate_indices(system, x, target_count)
            for candidate_index in candidate_indices:
                candidate_rows.append(len(rows))
                rows.append(
                    anchor_candidate_feature_vector(
                        system,
                        x,
                        target_count,
                        candidate_index,
                        kind=kind,
                        placement=placement,
                    )
                )
                labels.append(int(candidate_index == true_index))
                groups.append(group_id)
            if not candidate_rows:
                continue
            decisions.append(
                AnchorDecision(
                    candidate_rows=tuple(candidate_rows),
                    candidate_indices=tuple(candidate_indices),
                    baseline_index=baseline.local_index,
                    true_index=true_index,
                    source_group=group_id,
                    scenario=scenario,
                    kind=kind,
                    placement=placement,
                )
            )

    features = np.asarray(rows, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != len(ANCHOR_FEATURE_NAMES):
        raise RuntimeError("direction measure anchor feature shape mismatch")
    return DirectionMeasureAnchorDataset(
        features=features,
        labels=np.asarray(labels, dtype=np.int64),
        group_ids=np.asarray(groups, dtype=np.int64),
        decisions=tuple(decisions),
    )
