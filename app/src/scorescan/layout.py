from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .barline_classifier import BarlineClassifier, BarlineFeatures, extract_barline_features
from .barline_sequence import (
    BarlineSequenceClassifier,
    RefinedBarline,
    extract_sequence_features,
    refine_barline_sequence,
)
from .models import PageInfo
from .policy import DEFAULT_POLICY
from .util import atomic_write_json


@dataclass
class StaffSystem:
    index: int
    line_y: list[float]
    top: int
    bottom: int
    left: int
    right: int
    spacing: float
    barlines: list[int] = field(default_factory=list)
    measure_count: int = 1
    barline_confidences: list[float] = field(default_factory=list)
    barline_sequence_confidences: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "line_y": [round(value, 2) for value in self.line_y],
            "top": self.top,
            "bottom": self.bottom,
            "left": self.left,
            "right": self.right,
            "spacing": round(self.spacing, 3),
            "barlines": self.barlines,
            "barline_confidences": [round(value, 6) for value in self.barline_confidences],
            "barline_sequence_confidences": [round(value, 6) for value in self.barline_sequence_confidences],
            "measure_count": self.measure_count,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StaffSystem":
        return cls(
            index=int(payload.get("index", 0)),
            line_y=[float(value) for value in payload.get("line_y", [])],
            top=int(payload.get("top", 0)),
            bottom=int(payload.get("bottom", 0)),
            left=int(payload.get("left", 0)),
            right=int(payload.get("right", 0)),
            spacing=float(payload.get("spacing", 0.0)),
            barlines=[int(value) for value in payload.get("barlines", [])],
            barline_confidences=[float(value) for value in payload.get("barline_confidences", [])],
            barline_sequence_confidences=[float(value) for value in payload.get("barline_sequence_confidences", [])],
            measure_count=int(payload.get("measure_count", 1)),
        )


@dataclass
class ScoreSystemLayout:
    """One horizontal system containing one or more physical five-line staves."""

    index: int
    staff_indices: list[int]
    top: int
    bottom: int
    left: int
    right: int
    spacing: float
    barlines: list[int] = field(default_factory=list)
    measure_count: int = 1
    grouping_confidence: float = 0.0
    grouping_method: str = "vertical_gap"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "staff_indices": self.staff_indices,
            "top": self.top,
            "bottom": self.bottom,
            "left": self.left,
            "right": self.right,
            "spacing": round(self.spacing, 3),
            "barlines": self.barlines,
            "measure_count": self.measure_count,
            "grouping_confidence": round(self.grouping_confidence, 6),
            "grouping_method": self.grouping_method,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScoreSystemLayout":
        return cls(
            index=int(payload.get("index", 0)),
            staff_indices=[int(value) for value in payload.get("staff_indices", [])],
            top=int(payload.get("top", 0)),
            bottom=int(payload.get("bottom", 0)),
            left=int(payload.get("left", 0)),
            right=int(payload.get("right", 0)),
            spacing=float(payload.get("spacing", 0.0)),
            barlines=[int(value) for value in payload.get("barlines", [])],
            measure_count=max(1, int(payload.get("measure_count", 1))),
            grouping_confidence=float(payload.get("grouping_confidence", 0.0)),
            grouping_method=str(payload.get("grouping_method", "vertical_gap")),
        )


@dataclass
class PartStaffGroup:
    """A source-evidenced instrument grouping inside a score system.

    Empty ``part_groups`` means unresolved, not "one instrument per staff". This
    prevents spacing heuristics from silently splitting a piano grand staff or
    merging two independent instruments.
    """

    system_index: int
    part_index: int
    staff_indices: list[int]
    role: str = "unresolved"
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_index": self.system_index,
            "part_index": self.part_index,
            "staff_indices": self.staff_indices,
            "role": self.role,
            "confidence": round(self.confidence, 6),
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PartStaffGroup":
        return cls(
            system_index=int(payload.get("system_index", 0)),
            part_index=int(payload.get("part_index", 0)),
            staff_indices=[int(value) for value in payload.get("staff_indices", [])],
            role=str(payload.get("role", "unresolved")),
            confidence=float(payload.get("confidence", 0.0)),
            evidence=[str(value) for value in payload.get("evidence", [])],
        )


@dataclass(frozen=True)
class CandidateLayoutExpectation:
    """Layout evidence conditioned on a candidate's simultaneous staff topology."""

    simultaneous_staff_count: int
    physical_staff_appearance_count: int
    score_system_count: int
    measure_count: int
    incomplete_staff_count: int
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "simultaneous_staff_count": self.simultaneous_staff_count,
            "physical_staff_appearance_count": self.physical_staff_appearance_count,
            "score_system_count": self.score_system_count,
            "measure_count": self.measure_count,
            "incomplete_staff_count": self.incomplete_staff_count,
            "confidence": round(self.confidence, 6),
        }


@dataclass
class PageLayout:
    width: int
    height: int
    systems: list[StaffSystem]
    confidence: float
    warnings: list[str] = field(default_factory=list)
    model_versions: dict[str, str] = field(default_factory=dict)
    model_statuses: dict[str, str] = field(default_factory=dict)
    score_systems: list[ScoreSystemLayout] = field(default_factory=list)
    part_groups: list[PartStaffGroup] = field(default_factory=list)

    @property
    def effective_score_systems(self) -> list[ScoreSystemLayout]:
        return self.score_systems or infer_score_systems(self.systems)

    @property
    def estimated_measure_count(self) -> int:
        return sum(system.measure_count for system in self.effective_score_systems)

    def expectation_for_staff_topology(
        self,
        simultaneous_staff_count: int,
    ) -> CandidateLayoutExpectation:
        """Interpret repeated staff bands using a candidate's score topology."""

        ordered = sorted(
            self.systems,
            key=lambda item: (
                item.line_y[0] if item.line_y else item.top,
                item.index,
            ),
        )
        simultaneous = max(1, int(simultaneous_staff_count or 1))
        if not ordered:
            return CandidateLayoutExpectation(
                simultaneous,
                0,
                0,
                0,
                simultaneous,
                0.0,
            )
        groups = [
            ordered[start:start + simultaneous]
            for start in range(0, len(ordered), simultaneous)
        ]
        measure_count = sum(
            max(
                1,
                int(round(float(np.median([staff.measure_count for staff in group])))),
            )
            for group in groups
            if group
        )
        remainder = len(ordered) % simultaneous
        incomplete = 0 if remainder == 0 else simultaneous - remainder
        confidence = max(0.0, min(1.0, float(self.confidence)))
        if incomplete:
            confidence *= max(0.0, 1.0 - incomplete / simultaneous)
        return CandidateLayoutExpectation(
            simultaneous,
            len(ordered),
            len(groups),
            measure_count,
            incomplete,
            confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            # Kept for readers of the previous artifact schema.
            "system_count": len(self.systems),
            "physical_staff_count": len(self.systems),
            "score_system_count": len(self.effective_score_systems),
            "estimated_measure_count": self.estimated_measure_count,
            "measure_count_hypotheses": sorted(
                {
                    self.expectation_for_staff_topology(staff_count).measure_count
                    for staff_count in range(1, min(16, len(self.systems)) + 1)
                    if len(self.systems) % staff_count == 0
                }
            ),
            "confidence": round(self.confidence, 4),
            "warnings": self.warnings,
            "model_versions": dict(sorted(self.model_versions.items())),
            "model_statuses": dict(sorted(self.model_statuses.items())),
            "systems": [system.to_dict() for system in self.systems],
            "score_systems": [system.to_dict() for system in self.effective_score_systems],
            "part_groups": [group.to_dict() for group in self.part_groups],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PageLayout":
        return cls(
            width=int(payload.get("width", 0)),
            height=int(payload.get("height", 0)),
            systems=[StaffSystem.from_dict(item) for item in payload.get("systems", [])],
            confidence=float(payload.get("confidence", 0.0)),
            warnings=[str(item) for item in payload.get("warnings", [])],
            model_versions={str(key): str(value) for key, value in dict(payload.get("model_versions") or {}).items()},
            model_statuses={str(key): str(value) for key, value in dict(payload.get("model_statuses") or {}).items()},
            score_systems=[
                ScoreSystemLayout.from_dict(item)
                for item in payload.get("score_systems", [])
            ],
            part_groups=[
                PartStaffGroup.from_dict(item)
                for item in payload.get("part_groups", [])
            ],
        )


def _consensus_barlines(staffs: list[StaffSystem], spacing: float) -> list[int]:
    if not staffs:
        return []
    tolerance = max(4.0, spacing * 0.9)
    candidates = sorted(
        (float(x), staff.index)
        for staff in staffs
        for x in staff.barlines
    )
    clusters: list[list[tuple[float, int]]] = []
    for candidate in candidates:
        if not clusters:
            clusters.append([candidate])
            continue
        center = float(np.median([item[0] for item in clusters[-1]]))
        if candidate[0] - center <= tolerance:
            clusters[-1].append(candidate)
        else:
            clusters.append([candidate])
    minimum_support = max(1, int(np.ceil(len(staffs) * 0.5)))
    return [
        int(round(float(np.median([item[0] for item in cluster]))))
        for cluster in clusters
        if len({item[1] for item in cluster}) >= minimum_support
    ]


def _score_system_measure_barlines(
    staffs: list[StaffSystem],
    spacing: float,
    measure_count: int,
) -> list[int]:
    """Recover semantic score-system boundaries from uneven physical staves.

    Multi-staff scans often expose the opening barline on only one staff while
    internal barlines are shared by most staves.  Conversely, braces and instrument
    labels can extend the detected physical staff extent far to the left.  Start with
    majority-supported barlines, then recover one missing opening boundary from the
    union when the expected measure count proves that it is needed.
    """

    consensus = _consensus_barlines(staffs, spacing)
    if not staffs:
        return consensus
    expected_boundary_count = max(2, int(measure_count) + 1)
    if len(consensus) >= expected_boundary_count:
        return consensus

    tolerance = max(4.0, spacing * 0.9)
    union = sorted(
        {
            int(round(float(value)))
            for staff in staffs
            for value in staff.barlines
        }
    )
    if consensus and len(consensus) == expected_boundary_count - 1:
        preceding = [
            value
            for value in union
            if value < float(consensus[0]) - tolerance
        ]
        if preceding:
            interior_widths = [
                float(right - left)
                for left, right in zip(consensus, consensus[1:], strict=False)
                if right - left > tolerance
            ]
            median_width = (
                float(np.median(interior_widths))
                if interior_widths
                else max(float(consensus[0] - preceding[-1]), 1.0)
            )
            plausible = [
                value
                for value in preceding
                if 0.35
                <= (float(consensus[0]) - float(value)) / max(median_width, 1.0)
                <= 3.0
            ]
            opening = min(
                plausible or preceding,
                key=lambda value: abs(
                    (float(consensus[0]) - float(value))
                    / max(median_width, 1.0)
                    - 1.0
                ),
            )
            return [opening, *consensus]
    return consensus


def infer_score_systems(staffs: list[StaffSystem]) -> list[ScoreSystemLayout]:
    """Infer vertical score-system groups without inferring instrument identity.

    This stage uses only page geometry. Instrument/grand-staff grouping is kept
    unresolved until brace/bracket and recognizer evidence is available.
    """

    ordered = sorted(staffs, key=lambda item: (item.line_y[0] if item.line_y else item.top, item.index))
    if not ordered:
        return []
    if len(ordered) == 1:
        staff = ordered[0]
        barlines = _score_system_measure_barlines(
            [staff],
            staff.spacing,
            staff.measure_count,
        )
        return [
            ScoreSystemLayout(
                1,
                [staff.index],
                staff.top,
                staff.bottom,
                barlines[0] if len(barlines) >= staff.measure_count + 1 else staff.left,
                staff.right,
                staff.spacing,
                barlines,
                staff.measure_count,
                1.0,
                "single_physical_staff",
            )
        ]

    gap_ratios: list[float] = []
    for upper, lower in zip(ordered, ordered[1:], strict=False):
        upper_bottom_line = upper.line_y[-1] if upper.line_y else upper.bottom
        lower_top_line = lower.line_y[0] if lower.line_y else lower.top
        spacing = max(1.0, float(np.median([upper.spacing, lower.spacing])))
        gap_ratios.append(max(0.0, float(lower_top_line - upper_bottom_line)) / spacing)

    method = "absolute_vertical_gap"
    threshold = 9.5
    confidence = 0.72
    sorted_ratios = sorted(gap_ratios)
    if min(gap_ratios) >= 9.5:
        # Typical solo-instrument page: every five-line staff is a separate
        # horizontal system.
        threshold = 9.0
        method = "all_gaps_separate"
        confidence = min(0.99, 0.82 + (min(gap_ratios) - threshold) * 0.025)
    elif len(sorted_ratios) >= 2:
        jumps = [
            (sorted_ratios[index + 1] - sorted_ratios[index], index)
            for index in range(len(sorted_ratios) - 1)
        ]
        largest_jump, jump_index = max(jumps)
        lower = sorted_ratios[jump_index]
        upper = sorted_ratios[jump_index + 1]
        if largest_jump >= 2.5 and upper >= 8.0:
            threshold = (lower + upper) / 2.0
            method = "adaptive_vertical_gap"
            confidence = min(0.98, 0.74 + largest_jump * 0.035)
        elif len(gap_ratios) >= 5:
            # Dense piano and ensemble pages can have two stable gap bands even
            # when no single sorted gap jump is large.  A raw two-cluster split
            # is not sufficient: a smooth sequence of unrelated gaps would also
            # form two clusters.  Accept it only when the larger-gap boundaries
            # produce a repeated physical-staff topology across the page.
            low_center = min(gap_ratios)
            high_center = max(gap_ratios)
            low_cluster: list[float] = []
            high_cluster: list[float] = []
            for _iteration in range(32):
                low_cluster = []
                high_cluster = []
                for value in gap_ratios:
                    if abs(value - low_center) <= abs(value - high_center):
                        low_cluster.append(value)
                    else:
                        high_cluster.append(value)
                if not low_cluster or not high_cluster:
                    break
                next_low = float(np.mean(low_cluster))
                next_high = float(np.mean(high_cluster))
                if (
                    abs(next_low - low_center) <= 1e-9
                    and abs(next_high - high_center) <= 1e-9
                ):
                    break
                low_center = next_low
                high_center = next_high

            if low_cluster and high_cluster:
                low_max = max(low_cluster)
                high_min = min(high_cluster)
                candidate_threshold = (low_max + high_min) / 2.0
                split_after = [
                    index + 1
                    for index, value in enumerate(gap_ratios)
                    if value >= candidate_threshold
                ]
                boundaries = [0, *split_after, len(ordered)]
                group_sizes = [
                    right - left
                    for left, right in zip(boundaries, boundaries[1:], strict=False)
                ]
                size_counts = {
                    size: group_sizes.count(size)
                    for size in set(group_sizes)
                    if size >= 2
                }
                dominant_size, dominant_count = max(
                    size_counts.items(),
                    key=lambda item: (item[1], item[0]),
                    default=(0, 0),
                )
                separation = high_min - low_max
                center_ratio = high_center / max(low_center, 1e-6)
                repeated_topology = (
                    len(split_after) >= 2
                    and dominant_size >= 2
                    and dominant_count >= len(group_sizes) - 1
                )
                if (
                    repeated_topology
                    and separation >= 0.70
                    and high_center >= 7.0
                    and center_ratio >= 1.30
                ):
                    threshold = candidate_threshold
                    method = "periodic_bimodal_vertical_gap"
                    consistency = dominant_count / max(len(group_sizes), 1)
                    confidence = min(
                        0.96,
                        0.78
                        + min(0.08, separation * 0.025)
                        + min(0.08, consistency * 0.08),
                    )

    groups: list[list[StaffSystem]] = [[ordered[0]]]
    for gap_ratio, staff in zip(gap_ratios, ordered[1:], strict=True):
        if gap_ratio >= threshold:
            groups.append([staff])
        else:
            groups[-1].append(staff)

    result: list[ScoreSystemLayout] = []
    for index, group in enumerate(groups, start=1):
        spacing = float(np.median([staff.spacing for staff in group]))
        measure_count = max(
            1,
            int(round(float(np.median([staff.measure_count for staff in group])))),
        )
        barlines = _score_system_measure_barlines(group, spacing, measure_count)
        music_left = (
            barlines[0]
            if len(barlines) >= measure_count + 1
            else min(staff.left for staff in group)
        )
        result.append(
            ScoreSystemLayout(
                index=index,
                staff_indices=[staff.index for staff in group],
                top=min(staff.top for staff in group),
                bottom=max(staff.bottom for staff in group),
                left=music_left,
                right=max(staff.right for staff in group),
                spacing=spacing,
                barlines=barlines,
                measure_count=measure_count,
                grouping_confidence=confidence,
                grouping_method=method,
            )
        )
    return result


@dataclass(frozen=True)
class MeasureAnchor:
    local_index: int
    offset_ratio: float
    confidence: float
    method: str


SystemMeasureGeometry = StaffSystem | ScoreSystemLayout


def system_measure_boundaries(system: SystemMeasureGeometry) -> list[int]:
    """Return stable source-image measure boundaries for one staff system.

    Detected barlines are preferred, with the staff extent acting as implicit outer
    boundaries.  Near-duplicate thick/double strokes collapse to one semantic boundary.
    """

    tolerance = max(5, int(system.spacing * 0.9))
    boundaries = [int(system.left)]
    for x in sorted(int(value) for value in system.barlines):
        if x <= system.left or x >= system.right:
            continue
        if x - boundaries[-1] > tolerance:
            boundaries.append(x)
    if int(system.right) - boundaries[-1] > tolerance:
        boundaries.append(int(system.right))
    elif len(boundaries) == 1:
        boundaries.append(int(system.right))
    return boundaries


def system_measure_bounds(system: SystemMeasureGeometry) -> list[tuple[int, int]]:
    boundaries = system_measure_boundaries(system)
    return [
        (boundaries[index], boundaries[index + 1])
        for index in range(max(0, len(boundaries) - 1))
    ]


def anchor_x_to_measure(
    system: SystemMeasureGeometry,
    x: float,
    target_measure_count: int,
) -> MeasureAnchor:
    """Map a source x-coordinate to a MusicXML measure and within-measure offset.

    When detected visual intervals and MusicXML measures agree, the mapping is exact.
    If one side contains a small count error, the piecewise visual coordinate is
    proportionally rescaled instead of falling back to equal-width page bins.  This
    preserves irregular engraving and pickup-measure spacing while remaining bounded.
    """

    target_count = max(1, int(target_measure_count))
    bounds = system_measure_bounds(system)
    if not bounds:
        width = max(1.0, float(system.right - system.left))
        measure_float = max(0.0, min(float(target_count) - 1e-6, (float(x) - system.left) / width * target_count))
        index = min(target_count - 1, int(measure_float))
        return MeasureAnchor(index, measure_float - index, 0.35, "uniform_fallback")

    clamped_x = max(float(system.left), min(float(system.right) - 1e-6, float(x)))
    source_index = len(bounds) - 1
    source_offset = 0.999999
    for index, (left, right) in enumerate(bounds):
        if clamped_x < right or index == len(bounds) - 1:
            source_index = index
            source_offset = (clamped_x - left) / max(float(right - left), 1.0)
            source_offset = max(0.0, min(0.999999, source_offset))
            break

    source_count = len(bounds)
    source_coordinate = source_index + source_offset
    if source_count == target_count:
        local_index = min(target_count - 1, source_index)
        confidence = 0.92
        method = "barline_exact"
        offset = source_offset
    else:
        mapped = source_coordinate / max(float(source_count), 1.0) * target_count
        mapped = max(0.0, min(float(target_count) - 1e-6, mapped))
        local_index = min(target_count - 1, int(mapped))
        offset = mapped - local_index
        mismatch = abs(source_count - target_count) / max(source_count, target_count, 1)
        confidence = max(0.40, 0.82 - mismatch * 0.90)
        method = "barline_rescaled"

    return MeasureAnchor(local_index, float(offset), float(confidence), method)


def _group_indices(values: np.ndarray, max_gap: int = 2) -> list[tuple[int, int]]:
    if values.size == 0:
        return []
    groups: list[tuple[int, int]] = []
    start = previous = int(values[0])
    for raw in values[1:]:
        value = int(raw)
        if value - previous > max_gap:
            groups.append((start, previous))
            start = value
        previous = value
    groups.append((start, previous))
    return groups


def _projection_staff_line_centers(binary: np.ndarray) -> list[float]:
    height, width = binary.shape
    projection = np.count_nonzero(binary, axis=1)
    threshold = max(80, int(width * 0.32))
    rows = np.where(projection >= threshold)[0]
    bands = _group_indices(rows, max_gap=max(1, height // 1800))
    centers = [(start + end) / 2 for start, end in bands if end - start <= max(8, height // 500)]
    if len(centers) >= 5:
        return centers

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(60, int(width * 0.28)), 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    projection = np.count_nonzero(horizontal, axis=1)
    rows = np.where(projection >= max(40, int(width * 0.20)))[0]
    return [(a + b) / 2 for a, b in _group_indices(rows, max_gap=2)]


def _staff_group_shared_ink(
    binary: np.ndarray,
    lines: list[float],
) -> tuple[float, int]:
    profiles: list[np.ndarray] = []
    for line in lines:
        y = int(round(line))
        profiles.append(
            np.any(
                binary[
                    max(0, y - 1):min(binary.shape[0], y + 2),
                    :,
                ]
                > 0,
                axis=0,
            )
        )
    if len(profiles) != 5:
        return 0.0, 0
    shared = (np.sum(np.stack(profiles), axis=0) >= 3).astype(np.uint8)
    spacing = max(1.0, float(np.median(np.diff(lines))))
    closed = cv2.morphologyEx(
        shared.reshape(1, -1),
        cv2.MORPH_CLOSE,
        np.ones((1, max(5, int(round(spacing * 3.0)))), np.uint8),
    ).reshape(-1)
    runs = _group_indices(np.flatnonzero(closed), max_gap=1)
    longest = max((end - start + 1 for start, end in runs), default=0)
    return float(np.mean(shared)), int(longest)


def _comb_staff_groups(
    binary: np.ndarray,
    *,
    seed_groups: list[list[float]],
) -> list[list[float]]:
    """Recover complete five-line staffs from locally prominent row combs.

    A page-wide projection is precise on clean scans, but one dark or intact
    system can satisfy its early threshold while lighter systems disappear.
    This bounded second pass removes a local row baseline, searches five
    near-equidistant peaks, and requires their ink to overlap horizontally.
    Texture folds and text can create strong rows, but rarely five rows with
    consistent staff spacing and shared horizontal support.
    """

    height, width = binary.shape
    projection = np.count_nonzero(binary, axis=1).astype(np.float32)
    background = cv2.GaussianBlur(
        projection.reshape(-1, 1),
        (1, 31),
        0,
    ).reshape(-1)
    residual = np.maximum(0.0, projection - background)
    peak = cv2.dilate(
        residual.reshape(-1, 1),
        np.ones((5, 1), np.uint8),
    ).reshape(-1)
    strong = max(45.0, float(np.quantile(residual, 0.95)))
    weak = max(20.0, float(np.quantile(residual, 0.90)))
    maximum_spacing = min(55, max(8, int(height * 0.027)))
    candidates: list[tuple[float, list[float], float, float]] = []
    for spacing in range(8, maximum_spacing + 1):
        count = height - 4 * spacing
        if count <= 5:
            continue
        values = np.stack(
            [
                peak[offset * spacing:offset * spacing + count]
                for offset in range(5)
            ],
            axis=1,
        )
        weak_count = np.sum(values >= weak, axis=1)
        mask = (
            (
                np.sum(values >= strong, axis=1) >= 3
            )
            & (weak_count >= 4)
        ) | (weak_count == 5)
        for start in np.flatnonzero(mask):
            actual: list[float] = []
            for offset in range(5):
                target = int(start + offset * spacing)
                low = max(0, target - 2)
                high = min(height, target + 3)
                actual.append(
                    float(low + int(np.argmax(residual[low:high])))
                )
            diffs = np.diff(actual)
            actual_spacing = float(np.median(diffs))
            if (
                max(abs(float(value) - actual_spacing) for value in diffs)
                > max(2.5, 0.18 * actual_spacing)
            ):
                continue
            shared, longest_shared_run = _staff_group_shared_ink(
                binary,
                actual,
            )
            if (
                shared < 0.08
                or longest_shared_run
                < max(
                    int(round(actual_spacing * 10.0)),
                    int(round(width * 0.08)),
                )
            ):
                continue
            score = float(
                np.sum(np.minimum(values[start], strong * 2.5))
                / (strong * 5.0)
            )
            score += 1.6 * min(
                1.0,
                longest_shared_run / max(width * 0.25, 1.0),
            )
            candidates.append(
                (score + 0.35 * shared, actual, actual_spacing, shared)
            )
    if not candidates:
        return seed_groups

    seed_spacings = [
        float(np.median(np.diff(group)))
        for group in seed_groups
        if len(group) == 5
    ]
    if len(seed_spacings) >= 2:
        modal_spacing = float(np.median(seed_spacings))
    else:
        weighted_bins: dict[int, float] = {}
        for score, _lines, spacing, shared in candidates:
            key = int(round(spacing))
            weighted_bins[key] = weighted_bins.get(key, 0.0) + score * shared
        modal_spacing = float(
            max(weighted_bins, key=lambda key: (weighted_bins[key], -key))
        )
    tolerance = max(2.5, modal_spacing * 0.15)
    compatible = [
        item
        for item in candidates
        if (
            item[0] >= 1.85
            and abs(item[2] - modal_spacing) <= tolerance
        )
    ]
    selected = [
        (100.0, list(group))
        for group in seed_groups
        if abs(float(np.median(np.diff(group))) - modal_spacing) <= tolerance
    ]
    minimum_center_gap = modal_spacing * 3.4
    for score, lines, _spacing, _shared in sorted(
        compatible,
        key=lambda item: (-item[0], item[1][0]),
    ):
        centre = float(np.mean(lines))
        if any(
            abs(centre - float(np.mean(other))) < minimum_center_gap
            for _other_score, other in selected
        ):
            continue
        selected.append((score, lines))
    return [
        lines
        for _score, lines in sorted(
            selected,
            key=lambda item: float(np.mean(item[1])),
        )
    ]


def _faint_short_staff_groups(
    binary: np.ndarray,
    *,
    seed_groups: list[list[float]],
) -> list[list[float]]:
    """Recover a trailing short system whose faint lines lose the global vote.

    A short final system can occupy less than one sixth of a scanned page.  If two
    of its five lines are also faded, neither the page projection nor the regular
    comb pass sees all five rows.  Once other staves establish the page's spacing,
    five locally aligned rows plus shared horizontal ink are strong enough evidence
    to recover that system without accepting isolated text or page-fold strokes.
    This pass is deliberately trailing-only: inserting a partial faint hypothesis
    between established systems can disrupt the later five-row grouping and is
    better handled by candidate-conditioned topology recovery.
    """

    if not seed_groups:
        return seed_groups
    height, width = binary.shape
    seed_spacings = [
        float(np.median(np.diff(group)))
        for group in seed_groups
        if len(group) == 5
    ]
    if not seed_spacings:
        return seed_groups
    modal_spacing = float(np.median(seed_spacings))
    if not 3.0 <= modal_spacing <= max(55.0, height * 0.027):
        return seed_groups

    projection = np.count_nonzero(binary, axis=1).astype(np.float32)
    background = cv2.GaussianBlur(
        projection.reshape(-1, 1),
        (1, 31),
        0,
    ).reshape(-1)
    residual = np.maximum(0.0, projection - background)
    peak = cv2.dilate(
        residual.reshape(-1, 1),
        np.ones((5, 1), np.uint8),
    ).reshape(-1)
    strong = max(14.0, float(np.quantile(residual, 0.80)))
    weak = max(6.0, float(np.quantile(residual, 0.58)))
    spacing_radius = max(2, int(round(modal_spacing * 0.12)))
    candidates: list[tuple[float, list[float]]] = []
    for spacing in range(
        max(3, int(round(modal_spacing)) - spacing_radius),
        int(round(modal_spacing)) + spacing_radius + 1,
    ):
        count = height - 4 * spacing
        if count <= 5:
            continue
        values = np.stack(
            [
                peak[offset * spacing:offset * spacing + count]
                for offset in range(5)
            ],
            axis=1,
        )
        mask = (
            (np.sum(values >= strong, axis=1) >= 3)
            & np.all(values >= weak, axis=1)
        )
        for start in np.flatnonzero(mask):
            actual: list[float] = []
            for offset in range(5):
                target = int(start + offset * spacing)
                low = max(0, target - 2)
                high = min(height, target + 3)
                actual.append(
                    float(low + int(np.argmax(residual[low:high])))
                )
            diffs = np.diff(actual)
            actual_spacing = float(np.median(diffs))
            if (
                abs(actual_spacing - modal_spacing)
                > max(2.5, 0.15 * modal_spacing)
                or max(
                    abs(float(value) - actual_spacing)
                    for value in diffs
                )
                > max(2.5, 0.18 * actual_spacing)
            ):
                continue
            shared, longest_shared_run = _staff_group_shared_ink(
                binary,
                actual,
            )
            if (
                shared < 0.07
                or longest_shared_run
                < max(
                    int(round(actual_spacing * 8.0)),
                    int(round(width * 0.055)),
                )
            ):
                continue
            sampled = np.array(
                [
                    residual[int(round(value))]
                    for value in actual
                ],
                dtype=np.float32,
            )
            y1 = max(0, int(round(actual[0])))
            y2 = min(height, int(round(actual[-1])) + 1)
            vertical_support = (
                int(
                    np.count_nonzero(
                        np.count_nonzero(binary[y1:y2, :], axis=0)
                        / max(1, y2 - y1)
                        >= 0.75
                    )
                )
                if y2 > y1
                else 0
            )
            score = float(
                np.sum(np.minimum(sampled / max(strong, 1.0), 2.5))
                + 8.0 * shared
                + min(
                    2.0,
                    longest_shared_run / max(width * 0.08, 1.0),
                )
                + 0.15 * min(60, vertical_support)
            )
            candidates.append((score, actual))

    selected = [(100.0, list(group)) for group in seed_groups]
    last_seed_bottom = max(float(group[-1]) for group in seed_groups)
    minimum_center_gap = modal_spacing * 3.4
    for score, lines in sorted(
        candidates,
        key=lambda item: (-item[0], item[1][0]),
    ):
        if float(lines[0]) <= last_seed_bottom + 1.8 * modal_spacing:
            continue
        center = float(np.mean(lines))
        if any(
            abs(line - existing_line) <= max(3.0, 0.22 * modal_spacing)
            for line in lines
            for _other_score, other in selected
            for existing_line in other
        ):
            continue
        if any(
            (
                max(
                    0.0,
                    max(float(lines[0]), float(other[0]))
                    - min(float(lines[-1]), float(other[-1])),
                )
                if (
                    float(lines[-1]) < float(other[0])
                    or float(other[-1]) < float(lines[0])
                )
                else 0.0
            )
            < 1.8 * modal_spacing
            for _other_score, other in selected
        ):
            continue
        if any(
            abs(center - float(np.mean(other))) < minimum_center_gap
            for _other_score, other in selected
        ):
            continue
        selected.append((score, lines))
    return [
        lines
        for _score, lines in sorted(
            selected,
            key=lambda item: float(np.mean(item[1])),
        )
    ]


def _merge_staff_group_hypotheses(
    primary_groups: list[list[float]],
    supplemental_groups: list[list[float]],
    *,
    gray: np.ndarray | None = None,
) -> list[list[float]]:
    """Add complete faint staffs without moving strict-threshold geometry.

    Relaxing the image threshold is useful for faded scans, but it also makes
    beams, text baselines, and page stains look more line-like.  Strict groups
    therefore own every overlapping staff band.  A relaxed group may only fill
    an otherwise empty band when it has five regular lines and its spacing
    agrees with the established strict-page spacing.
    """

    primary = [
        [float(line) for line in group]
        for group in primary_groups
        if len(group) == 5
    ]
    supplemental = [
        [float(line) for line in group]
        for group in supplemental_groups
        if len(group) == 5
    ]
    if not primary:
        return sorted(supplemental, key=lambda group: float(np.mean(group)))
    primary_spacings = [
        float(np.median(np.diff(group)))
        for group in primary
    ]
    modal_spacing = float(np.median(primary_spacings))
    if modal_spacing <= 0.0:
        return sorted(primary, key=lambda group: float(np.mean(group)))

    spacing_tolerance = max(2.5, modal_spacing * 0.15)
    minimum_center_gap = modal_spacing * 3.4
    maximum_line_shift = max(3.0, modal_spacing * 0.22)
    selected = list(primary)
    for group in sorted(supplemental, key=lambda item: float(np.mean(item))):
        diffs = np.diff(group)
        spacing = float(np.median(diffs))
        if (
            spacing <= 0.0
            or abs(spacing - modal_spacing) > spacing_tolerance
            or max(abs(float(value) - spacing) for value in diffs)
            > max(2.5, 0.18 * spacing)
        ):
            continue
        if gray is not None:
            line_rows: list[np.ndarray] = []
            interline_rows: list[np.ndarray] = []
            for line in group:
                y = int(round(line))
                line_rows.append(
                    gray[max(0, y - 1):min(gray.shape[0], y + 2), :]
                )
            for upper, lower in zip(group, group[1:], strict=False):
                y = int(round(0.5 * (upper + lower)))
                interline_rows.append(
                    gray[max(0, y - 1):min(gray.shape[0], y + 2), :]
                )
            line_mean = float(
                np.mean(np.concatenate(line_rows, axis=0))
            )
            interline_mean = float(
                np.mean(np.concatenate(interline_rows, axis=0))
            )
            if interline_mean - line_mean < 4.0:
                continue
        center = float(np.mean(group))
        if any(
            abs(center - float(np.mean(existing))) < minimum_center_gap
            for existing in selected
        ):
            continue
        if any(
            abs(line - existing_line) <= maximum_line_shift
            for line in group
            for existing in selected
            for existing_line in existing
        ):
            continue
        selected.append(group)
    return sorted(selected, key=lambda group: float(np.mean(group)))


def _candidate_staff_lines(
    binary: np.ndarray,
    gray: np.ndarray | None = None,
) -> list[float]:
    """Find staff lines without dropping antialiased light-grey scan rows.

    A global Otsu threshold can cut alternating staff lines when the page has a
    grey paper/background gradient.  We compare that strict hypothesis with one
    bounded relaxed threshold and keep the hypothesis forming more complete
    five-line staffs.
    """

    strict = _projection_staff_line_centers(binary)
    if gray is None:
        return strict
    otsu, _unused = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    relaxed_threshold = int(max(80, min(225, round(float(otsu)) + 45)))
    relaxed = np.where(gray <= relaxed_threshold, 255, 0).astype(np.uint8)
    relaxed_centers = _projection_staff_line_centers(relaxed)
    strict_groups = _group_staffs(strict, gray.shape[0])
    relaxed_groups = _group_staffs(relaxed_centers, gray.shape[0])
    strict_recovered = _comb_staff_groups(
        binary,
        seed_groups=strict_groups,
    )
    relaxed_recovered = _comb_staff_groups(
        relaxed,
        seed_groups=relaxed_groups,
    )
    recovered = _merge_staff_group_hypotheses(
        strict_recovered,
        relaxed_recovered,
        gray=gray,
    )
    recovered = _faint_short_staff_groups(
        binary,
        seed_groups=recovered,
    )
    return [
        float(line)
        for group in recovered
        for line in group
    ]


def _group_staffs(centers: list[float], height: int) -> list[list[float]]:
    results: list[list[float]] = []
    index = 0
    while index + 4 < len(centers):
        candidate = centers[index:index + 5]
        diffs = np.diff(candidate)
        median = float(np.median(diffs))
        tolerance = max(2.2, median * 0.32)
        if 3.0 <= median <= max(55.0, height * 0.027) and np.all(np.abs(diffs - median) <= tolerance):
            # Avoid duplicate overlapping groups caused by thick lines.
            if not results or candidate[0] - results[-1][-1] > median * 2.5:
                results.append(candidate)
            index += 5
        else:
            index += 1
    return results


def _staff_horizontal_extent(binary: np.ndarray, lines: list[float], spacing: float) -> tuple[int, int]:
    height, width = binary.shape
    ys: list[int] = []
    for line in lines:
        y = int(round(line))
        ys.extend(range(max(0, y - 1), min(height, y + 2)))
    profile = np.count_nonzero(binary[ys, :], axis=0)
    columns = np.where(profile >= max(2, int(len(ys) * 0.20)))[0]
    if columns.size == 0:
        return int(width * 0.05), int(width * 0.95)
    # Ignore isolated title/text pixels by using long horizontal continuity.
    mask = np.zeros(width, np.uint8)
    mask[columns] = 255
    kernel = np.ones(max(5, int(spacing * 3)), np.uint8)
    closed = cv2.morphologyEx(mask.reshape(1, -1), cv2.MORPH_CLOSE, kernel.reshape(1, -1)).ravel()
    columns = np.where(closed > 0)[0]
    return int(columns[0]), int(columns[-1])


def _single_line_staffs(
    binary: np.ndarray,
    five_line_staffs: list[StaffSystem],
) -> list[StaffSystem]:
    """Recover isolated one-line percussion staves.

    A one-line staff has no five-line comb, but its row support is comparable to
    the printed lines of the surrounding instruments.  Requiring more than half
    a page of horizontal ink and strong isolation from every known staff line
    keeps this path specific to full-width ensemble percussion staves.
    """

    if not five_line_staffs:
        return []
    height, width = binary.shape
    spacing = float(np.median([staff.spacing for staff in five_line_staffs]))
    projection = np.count_nonzero(binary, axis=1)
    rows = np.where(projection >= max(160, int(round(width * 0.52))))[0]
    bands = _group_indices(rows, max_gap=max(1, int(round(spacing * 0.12))))
    known_lines = [
        float(line)
        for staff in five_line_staffs
        for line in staff.line_y
    ]
    recovered: list[StaffSystem] = []
    for start, end in bands:
        if (
            end - start > max(8, int(round(spacing * 0.40)))
            or start < height * 0.03
            or end > height * 0.97
        ):
            continue
        center = 0.5 * (start + end)
        if any(
            abs(center - line) < max(60.0, spacing * 2.8)
            for line in known_lines
        ):
            continue
        row_slice = binary[
            max(0, start - 1):min(height, end + 2),
            :,
        ]
        profile = np.any(row_slice > 0, axis=0).astype(np.uint8)
        closed = cv2.morphologyEx(
            profile.reshape(1, -1),
            cv2.MORPH_CLOSE,
            np.ones(
                (1, max(9, int(round(spacing * 5.0)))),
                np.uint8,
            ),
        ).reshape(-1)
        runs = _group_indices(np.flatnonzero(closed), max_gap=1)
        if not runs:
            continue
        left, right = max(
            runs,
            key=lambda item: item[1] - item[0],
        )
        if right - left < max(width * 0.45, height * 1.5):
            continue
        recovered.append(
            StaffSystem(
                index=0,
                line_y=[center],
                top=max(0, int(round(center - spacing * 4.2))),
                bottom=min(
                    height - 1,
                    int(round(center + spacing * 4.2)),
                ),
                left=int(left),
                right=int(right),
                spacing=spacing,
                barlines=[],
                measure_count=1,
            )
        )
    return recovered


def _recover_five_line_staffs_from_long_line_pairs(
    binary: np.ndarray,
    gray: np.ndarray,
    five_line_staffs: list[StaffSystem],
    single_line_staffs: list[StaffSystem],
) -> tuple[list[list[float]], list[StaffSystem]]:
    """Turn two compatible long-line remnants back into one faded staff.

    On very wide ensemble scans, noteheads and scan seams can break three staff
    lines below the page-wide projection floor while two lines remain long
    enough to resemble independent one-line percussion staves.  A pair is
    promoted only when it determines a five-line comb at the established page
    spacing and that complete comb has shared horizontal ink plus grayscale
    line/interline contrast.  A genuine isolated one-line staff is therefore
    left untouched.
    """

    if len(five_line_staffs) < 2 or len(single_line_staffs) < 2:
        return [], single_line_staffs
    height, width = binary.shape
    spacings = [
        float(staff.spacing)
        for staff in five_line_staffs
        if math.isfinite(float(staff.spacing)) and float(staff.spacing) > 0.0
    ]
    if len(spacings) < 2:
        return [], single_line_staffs
    modal_spacing = float(np.median(spacings))
    spacing_tolerance = max(2.5, modal_spacing * 0.15)
    maximum_anchor_shift = max(3.0, modal_spacing * 0.22)
    known_centres = [
        float(np.mean(staff.line_y))
        for staff in five_line_staffs
        if len(staff.line_y) == 5
    ]
    candidates: list[
        tuple[float, int, int, list[float]]
    ] = []
    ordered = sorted(
        enumerate(single_line_staffs),
        key=lambda item: float(item[1].line_y[0]),
    )
    for ordered_left, (left_index, left_staff) in enumerate(ordered):
        left_center = float(left_staff.line_y[0])
        for right_index, right_staff in ordered[ordered_left + 1:]:
            right_center = float(right_staff.line_y[0])
            delta = right_center - left_center
            if delta <= 0.0 or delta > modal_spacing * 4.5:
                continue
            common_left = max(left_staff.left, right_staff.left)
            common_right = min(left_staff.right, right_staff.right)
            common_width = common_right - common_left + 1
            if common_width < max(width * 0.25, modal_spacing * 20.0):
                continue
            for gap_count in range(1, 5):
                spacing = delta / gap_count
                if abs(spacing - modal_spacing) > spacing_tolerance:
                    continue
                for left_line_index in range(0, 5 - gap_count):
                    first = left_center - left_line_index * spacing
                    lines = [
                        first + offset * spacing
                        for offset in range(5)
                    ]
                    if lines[0] < 0.0 or lines[-1] >= height:
                        continue
                    centre = float(np.mean(lines))
                    if any(
                        abs(centre - known) < modal_spacing * 3.4
                        for known in known_centres
                    ):
                        continue
                    if (
                        min(abs(left_center - line) for line in lines)
                        > maximum_anchor_shift
                        or min(abs(right_center - line) for line in lines)
                        > maximum_anchor_shift
                    ):
                        continue
                    shared, longest_shared_run = _staff_group_shared_ink(
                        binary[:, common_left:common_right + 1],
                        lines,
                    )
                    if (
                        shared < 0.06
                        or longest_shared_run
                        < max(
                            int(round(spacing * 8.0)),
                            int(round(common_width * 0.05)),
                        )
                    ):
                        continue
                    line_rows = [
                        gray[
                            max(0, int(round(line)) - 1):
                            min(height, int(round(line)) + 2),
                            common_left:common_right + 1,
                        ]
                        for line in lines
                    ]
                    interline_rows = [
                        gray[
                            max(0, int(round(0.5 * (upper + lower))) - 1):
                            min(
                                height,
                                int(round(0.5 * (upper + lower))) + 2,
                            ),
                            common_left:common_right + 1,
                        ]
                        for upper, lower in zip(
                            lines,
                            lines[1:],
                            strict=False,
                        )
                    ]
                    line_mean = float(
                        np.mean(np.concatenate(line_rows, axis=0))
                    )
                    interline_mean = float(
                        np.mean(np.concatenate(interline_rows, axis=0))
                    )
                    contrast = interline_mean - line_mean
                    if contrast < 6.0:
                        continue
                    score = (
                        contrast
                        + 12.0 * shared
                        + min(
                            4.0,
                            longest_shared_run / max(
                                modal_spacing * 10.0,
                                1.0,
                            ),
                        )
                        - abs(spacing - modal_spacing)
                    )
                    candidates.append(
                        (score, left_index, right_index, lines)
                    )

    recovered: list[list[float]] = []
    consumed: set[int] = set()
    recovered_centres: list[float] = []
    for _score, left_index, right_index, lines in sorted(
        candidates,
        key=lambda item: (-item[0], item[3][0]),
    ):
        if left_index in consumed or right_index in consumed:
            continue
        centre = float(np.mean(lines))
        if any(
            abs(centre - existing) < modal_spacing * 3.4
            for existing in recovered_centres
        ):
            continue
        recovered.append(lines)
        recovered_centres.append(centre)
        consumed.update((left_index, right_index))
    remaining = [
        staff
        for index, staff in enumerate(single_line_staffs)
        if index not in consumed
    ]
    return (
        sorted(recovered, key=lambda lines: float(np.mean(lines))),
        remaining,
    )


def _vertical_proposals(crop: np.ndarray, spacing: float) -> list[tuple[int, int]]:
    """Generate permissive vertical-stroke bands for the visual classifier.

    The proposal stage is intentionally recall-oriented.  A shorter opening kernel and
    a lower raw-coverage floor recover broken, thin, and partially connected barlines;
    learned and deterministic gates perform rejection later.  Widening proposals is
    safe only because the complete proposal pipeline is frozen and benchmarked.
    """

    vertical_height = max(7, int(round(spacing * 2.30)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_height))
    vertical = cv2.morphologyEx(crop, cv2.MORPH_OPEN, kernel)
    opened_profile = np.count_nonzero(vertical, axis=0)
    raw_profile = np.count_nonzero(crop, axis=0)
    opened = np.where(opened_profile >= max(4, int(round(spacing * 1.75))))[0]
    raw = np.where(raw_profile >= max(5, int(round(crop.shape[0] * 0.38))))[0]
    indices = np.unique(np.concatenate((opened, raw)))
    bands = _group_indices(indices, max_gap=max(2, int(round(spacing * 0.24))))
    return [
        (start, end)
        for start, end in bands
        if end - start <= max(10, int(round(spacing * 1.60)))
    ]


_BARLINE_CLASSIFIER = BarlineClassifier()
_BARLINE_SEQUENCE_CLASSIFIER = BarlineSequenceClassifier()


@dataclass(frozen=True)
class BarlineProposalEvidence:
    """Immutable image and model evidence for one vertical proposal."""

    x: int
    absolute_start: int
    absolute_end: int
    probability: float
    features: BarlineFeatures
    top_hit: bool
    bottom_hit: bool
    relative_x: float
    model_enabled: bool


def _extract_barline_proposal_evidence(
    binary: np.ndarray,
    system: StaffSystem,
    *,
    classifier: BarlineClassifier | None = None,
) -> tuple[BarlineProposalEvidence, ...]:
    """Extract proposal evidence once, without making sequence-level decisions."""

    classifier = classifier or _BARLINE_CLASSIFIER
    height, width = binary.shape
    y1 = max(0, int(round(system.line_y[0] - system.spacing * 1.35)))
    y2 = min(height, int(round(system.line_y[-1] + system.spacing * 1.35)))
    crop = binary[y1:y2, system.left:system.right + 1]
    if crop.size == 0:
        return ()

    endpoint_window = max(2, int(round(system.spacing * 0.25)))
    staff_top = max(0, int(round(system.line_y[0] - 1)))
    staff_bottom = min(height, int(round(system.line_y[-1] + 2)))
    evidence: list[BarlineProposalEvidence] = []
    for start, end in _vertical_proposals(crop, system.spacing):
        absolute_start = start + system.left
        absolute_end = end + system.left
        x = int(round((absolute_start + absolute_end) / 2))
        features = extract_barline_features(
            binary,
            x_start=absolute_start,
            x_end=absolute_end,
            line_y=system.line_y,
            spacing=system.spacing,
        )
        classification = classifier.classify(features)
        strip_half = max(1, int(round(system.spacing * 0.20)))
        strip = binary[
            staff_top:staff_bottom,
            max(0, x - strip_half):min(width, x + strip_half + 1),
        ]
        if strip.size == 0:
            continue
        row_presence = np.any(strip > 0, axis=1)
        top_hit = bool(np.any(row_presence[:endpoint_window]))
        bottom_hit = bool(np.any(row_presence[-endpoint_window:]))
        evidence.append(
            BarlineProposalEvidence(
                x=x,
                absolute_start=absolute_start,
                absolute_end=absolute_end,
                probability=float(classification.probability),
                features=features,
                top_hit=top_hit,
                bottom_hit=bottom_hit,
                relative_x=(x - system.left) / max(system.right - system.left, 1),
                model_enabled=classifier.enabled,
            )
        )
    return tuple(evidence)


def _high_density_connected_boundary(item: BarlineProposalEvidence) -> bool:
    features = item.features
    return (
        item.probability >= DEFAULT_POLICY.barline_connected_probability_floor
        and features.interline_mean_coverage >= DEFAULT_POLICY.barline_connected_interline_mean_floor
        and features.longest_vertical_run >= DEFAULT_POLICY.barline_connected_run_floor
        and features.staff_line_intersection_ratio
        >= DEFAULT_POLICY.barline_connected_staff_intersection_floor
    )


def _local_barline_accept(item: BarlineProposalEvidence, *, probability_floor: float) -> bool:
    features = item.features
    endpoint_accept = item.top_hit and item.bottom_hit
    geometry_accept = (
        features.row_coverage >= 0.52
        and features.longest_vertical_run >= 0.46
        and endpoint_accept
        and features.staff_line_intersection_ratio >= 0.80
    )
    high_confidence_broken = (
        item.probability >= 0.93
        and features.row_coverage >= 0.90
        and features.staff_line_intersection_ratio >= 0.80
        and features.side_density <= 0.56
    )
    attachment_accept = (
        features.side_density <= 0.56
        or _high_density_connected_boundary(item)
        or (item.relative_x <= 0.15 and item.probability >= 0.72)
        or (item.relative_x >= 0.94 and item.probability >= 0.80)
    )
    structural_accept = (
        endpoint_accept
        and features.row_coverage >= 0.90
        and features.longest_vertical_run >= 0.90
        and features.staff_line_intersection_ratio >= 1.0
        and features.local_vertical_dominance >= 0.90
        and features.interline_min_coverage >= 0.90
        and features.side_density <= 0.45
    )
    complete_interline_boundary = (
        # A thin printed barline can receive a low learned score when a nearby
        # note or staff-line junction resembles training negatives.  Full
        # uninterrupted coverage of every staff gap is stronger independent
        # evidence than the local probability.  The extension and attachment
        # ceilings keep ordinary stems and text strokes outside this rescue.
        endpoint_accept
        and features.row_coverage >= 0.92
        and features.longest_vertical_run >= 0.92
        and features.staff_line_intersection_ratio >= 1.0
        and features.interline_min_coverage >= 0.98
        and features.side_density <= 0.30
        and features.mid_horizontal_attachment <= 0.50
        and features.above_extension <= 0.25
        and features.below_extension <= 0.25
    )
    model_accept = (
        item.model_enabled
        and (
            item.probability >= probability_floor
            or structural_accept
            or complete_interline_boundary
        )
        and features.row_coverage >= 0.72
        and features.longest_vertical_run >= 0.36
        and features.staff_line_intersection_ratio >= 0.80
        and attachment_accept
        and (endpoint_accept or high_confidence_broken)
    )
    return model_accept or (not item.model_enabled and geometry_accept)


def _opening_dual_model_reject(system: StaffSystem, item: RefinedBarline) -> bool:
    """Reject only an opening proposal that both learned layers consider weak."""
    x = int(getattr(item, "x"))
    local_probability = float(getattr(item, "local_probability"))
    sequence_probability = float(getattr(item, "sequence_probability"))
    relative_x = (x - system.left) / max(system.right - system.left, 1)
    return (
        relative_x <= DEFAULT_POLICY.barline_opening_post_position_ceiling
        and local_probability <= DEFAULT_POLICY.barline_opening_post_local_ceiling
        and sequence_probability <= DEFAULT_POLICY.barline_opening_post_sequence_ceiling
    )


def _post_refine_barlines(
    system: StaffSystem,
    sequence: tuple[RefinedBarline, ...],
) -> list[RefinedBarline]:
    """Apply narrow structural safety rules after learned sequence refinement.

    These rules do not regularise all measure widths. They cover only an opening split
    hidden by an explicit left-edge proposal and a close strong/weak duplicate pair.
    """

    retained = [item for item in sequence if item.retained]
    if not retained:
        return retained

    # Ignore proposals at the implicit system edges when locating the first musical
    # interior candidate. A left border must not disable the opening split guard.
    edge_tolerance = max(5, int(round(system.spacing * 1.1)))
    interior = [
        item
        for item in retained
        if item.x - system.left > edge_tolerance and system.right - item.x > edge_tolerance
    ]
    if len(interior) >= 2:
        candidate = interior[0]
        ordered = [(item.local_probability, item.x) for item in retained]
        index = next(i for i, (_probability, x) in enumerate(ordered) if x == candidate.x)
        features = extract_sequence_features(
            left=system.left,
            right=system.right,
            spacing=system.spacing,
            candidates=ordered,
            index=index,
        )
        if (
            features.normalised_position <= DEFAULT_POLICY.barline_opening_post_position_ceiling
            and candidate.local_probability <= DEFAULT_POLICY.barline_opening_split_local_ceiling
            and candidate.sequence_probability <= DEFAULT_POLICY.barline_opening_split_sequence_ceiling
            and features.left_gap_ratio <= DEFAULT_POLICY.barline_opening_split_gap_ratio_ceiling
            and features.right_gap_ratio <= DEFAULT_POLICY.barline_opening_split_gap_ratio_ceiling
            and features.merged_gap_deviation <= DEFAULT_POLICY.barline_opening_split_merged_deviation
        ):
            retained = [item for item in retained if item.x != candidate.x]

    # A proposal just beside a substantially stronger boundary is usually the second
    # side of a thick stroke or a connected stem. This is deliberately pair-local and
    # does not assume globally regular measure widths.
    changed = True
    while changed and len(retained) >= 2:
        changed = False
        retained.sort(key=lambda item: item.x)
        for left, right in zip(retained, retained[1:]):
            if right.x - left.x > system.spacing * DEFAULT_POLICY.barline_dominated_pair_spacing_ratio:
                continue
            stronger, weaker = (
                (left, right)
                if left.local_probability >= right.local_probability
                else (right, left)
            )
            if (
                stronger.local_probability >= DEFAULT_POLICY.barline_dominated_pair_strong_floor
                and weaker.local_probability <= DEFAULT_POLICY.barline_dominated_pair_weak_ceiling
                and stronger.local_probability - weaker.local_probability
                >= DEFAULT_POLICY.barline_dominated_pair_margin_floor
            ):
                retained = [item for item in retained if item.x != weaker.x]
                changed = True
                break
    return retained


def _select_barlines_from_evidence(
    system: StaffSystem,
    evidence: tuple[BarlineProposalEvidence, ...] | list[BarlineProposalEvidence],
    *,
    probability_floor: float | None = None,
    sequence_classifier: BarlineSequenceClassifier | None = None,
) -> tuple[list[int], list[float], list[float]]:
    """Apply local, geometric, non-maximum, and global sequence decisions."""

    floor = DEFAULT_POLICY.barline_probability_floor if probability_floor is None else float(probability_floor)
    ranked = [
        (item.probability if item.model_enabled else 0.75, item.x)
        for item in evidence
        if _local_barline_accept(item, probability_floor=floor)
    ]

    selected: list[tuple[float, int]] = []
    minimum_gap = max(5, int(round(system.spacing * DEFAULT_POLICY.barline_min_spacing_ratio)))
    for probability, x in sorted(ranked, key=lambda value: (-value[0], value[1])):
        if any(abs(x - other_x) <= minimum_gap for _other_probability, other_x in selected):
            continue
        selected.append((probability, x))
    selected.sort(key=lambda value: value[1])

    sequence = refine_barline_sequence(
        left=system.left,
        right=system.right,
        spacing=system.spacing,
        candidates=selected,
        classifier=sequence_classifier or _BARLINE_SEQUENCE_CLASSIFIER,
    )
    retained = [
        item
        for item in _post_refine_barlines(system, sequence)
        if not _opening_dual_model_reject(system, item)
    ]
    selected = [(item.final_probability, item.x) for item in retained]
    sequence_confidences = [item.sequence_probability for item in retained]

    plausible_limit = min(
        DEFAULT_POLICY.barline_max_candidates_per_system,
        max(12, int((system.right - system.left) / max(system.spacing * 7.0, 1.0))),
    )
    if len(selected) > plausible_limit:
        return [], [], []
    return (
        [x for _probability, x in selected],
        [probability for probability, _x in selected],
        sequence_confidences,
    )


def _detect_barlines(binary: np.ndarray, system: StaffSystem) -> tuple[list[int], list[float], list[float]]:
    evidence = _extract_barline_proposal_evidence(binary, system)
    return _select_barlines_from_evidence(system, evidence)

def _measure_count(system: StaffSystem) -> int:
    boundaries = list(system.barlines)
    edge_tolerance = max(10, int(system.spacing * 3.0))
    if not boundaries or abs(boundaries[0] - system.left) > edge_tolerance:
        boundaries.insert(0, system.left)
    if abs(boundaries[-1] - system.right) > edge_tolerance:
        boundaries.append(system.right)
    # Remove nearly duplicate thin/thick double-line detections.
    merged: list[int] = []
    for x in boundaries:
        if not merged or x - merged[-1] > max(5, int(system.spacing * 0.9)):
            merged.append(x)
    return max(1, len(merged) - 1)


def analyze_layout(image_path: Path) -> PageLayout:
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return PageLayout(0, 0, [], 0.0, ["页面无法读取"])
    height, width = gray.shape
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    centers = _candidate_staff_lines(binary, gray)
    staff_groups = _group_staffs(centers, height)
    systems: list[StaffSystem] = []
    warnings: list[str] = []
    model_confidence_factor = 1.0
    if not _BARLINE_CLASSIFIER.enabled:
        warnings.append("小节线局部分类模型不可用，已使用几何回退")
        model_confidence_factor *= 0.92
    if not _BARLINE_SEQUENCE_CLASSIFIER.enabled:
        warnings.append("小节线序列模型不可用，已停用全局候选精炼")
        model_confidence_factor *= 0.96
    for index, lines in enumerate(staff_groups, start=1):
        spacing = float(np.median(np.diff(lines)))
        top = max(0, int(round(lines[0] - spacing * 4.2)))
        bottom = min(height - 1, int(round(lines[-1] + spacing * 4.2)))
        left, right = _staff_horizontal_extent(binary, lines, spacing)
        system = StaffSystem(index, list(lines), top, bottom, left, right, spacing)
        (
            system.barlines,
            system.barline_confidences,
            system.barline_sequence_confidences,
        ) = _detect_barlines(binary, system)
        system.measure_count = _measure_count(system)
        systems.append(system)

    single_line_staffs = _single_line_staffs(binary, systems)
    recovered_groups, single_line_staffs = (
        _recover_five_line_staffs_from_long_line_pairs(
            binary,
            gray,
            systems,
            single_line_staffs,
        )
    )
    for lines in recovered_groups:
        spacing = float(np.median(np.diff(lines)))
        top = max(0, int(round(lines[0] - spacing * 4.2)))
        bottom = min(
            height - 1,
            int(round(lines[-1] + spacing * 4.2)),
        )
        left, right = _staff_horizontal_extent(binary, lines, spacing)
        system = StaffSystem(
            0,
            list(lines),
            top,
            bottom,
            left,
            right,
            spacing,
        )
        (
            system.barlines,
            system.barline_confidences,
            system.barline_sequence_confidences,
        ) = _detect_barlines(binary, system)
        system.measure_count = _measure_count(system)
        systems.append(system)
    if single_line_staffs:
        systems.extend(single_line_staffs)
    if recovered_groups or single_line_staffs:
        systems.sort(
            key=lambda item: (
                float(np.mean(item.line_y)),
                item.left,
            )
        )
        for index, system in enumerate(systems, start=1):
            system.index = index

    score_systems = infer_score_systems(systems)
    warnings.extend(_layout_boundary_warnings(systems, score_systems))
    spacing_variation = 0.0
    if systems:
        spacings = np.array([item.spacing for item in systems])
        spacing_variation = float(np.std(spacings) / max(np.mean(spacings), 1e-6))
        if spacing_variation > 0.18:
            warnings.append("不同系统的谱线间距变化较大")
    confidence = (
        min(1.0, len(systems) / 3.0)
        * max(0.0, 1.0 - spacing_variation * 1.8)
        * model_confidence_factor
    )
    return PageLayout(
        width,
        height,
        systems,
        confidence,
        warnings,
        model_versions={
            "barline_local": _BARLINE_CLASSIFIER.model_version,
            "barline_sequence": _BARLINE_SEQUENCE_CLASSIFIER.model_version,
        },
        model_statuses={
            "barline_local": _BARLINE_CLASSIFIER.status,
            "barline_sequence": _BARLINE_SEQUENCE_CLASSIFIER.status,
        },
        score_systems=score_systems,
    )


def _layout_boundary_warnings(
    systems: list[StaffSystem],
    score_systems: list[ScoreSystemLayout],
) -> list[str]:
    """Apply product limits to simultaneous systems, not whole-page repeats."""
    if not systems:
        return ["没有可靠检测到五线谱系统"]
    warnings = []
    if len(score_systems) > 16:
        warnings.append("检测到异常多的总谱系统，可能包含非目标内容")
    if any(len(system.staff_indices) > 16 for system in score_systems):
        warnings.append(
            "单个总谱系统超过 16 行物理谱表，超出高准确度边界"
        )
    return warnings


def write_layout_artifacts(page: PageInfo, output_dir: Path) -> PageLayout:
    image_path = Path(page.normalized_path or page.image_path)
    layout = analyze_layout(image_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"page_{page.index:04d}.json"
    overlay_path = output_dir / f"page_{page.index:04d}_overlay.png"
    atomic_write_json(json_path, layout.to_dict())

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is not None:
        for score_system in layout.effective_score_systems:
            cv2.rectangle(
                image,
                (score_system.left, score_system.top),
                (score_system.right, score_system.bottom),
                (180, 80, 200),
                2,
            )
            cv2.putText(
                image,
                f"SYS{score_system.index} ST={len(score_system.staff_indices)} M~{score_system.measure_count}",
                (score_system.left, max(20, score_system.top - 28)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (160, 40, 170),
                2,
                cv2.LINE_AA,
            )
        for system in layout.systems:
            cv2.rectangle(image, (system.left, system.top), (system.right, system.bottom), (0, 120, 255), 3)
            for y in system.line_y:
                cv2.line(image, (system.left, int(round(y))), (system.right, int(round(y))), (0, 180, 0), 1)
            for x in system.barlines:
                cv2.line(image, (x, system.top), (x, system.bottom), (255, 80, 0), 2)
            cv2.putText(
                image,
                f"S{system.index} M~{system.measure_count}",
                (system.left, max(20, system.top - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 80, 200),
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(overlay_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    page.layout_path = str(json_path)
    page.overlay_path = str(overlay_path)
    page.physical_staff_count = len(layout.systems)
    page.score_system_count = len(layout.effective_score_systems)
    page.staff_system_count = page.score_system_count
    page.estimated_measure_count = layout.estimated_measure_count
    page.quality_notes.extend(item for item in layout.warnings if item not in page.quality_notes)
    return layout
