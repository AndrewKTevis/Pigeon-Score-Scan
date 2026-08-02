from __future__ import annotations

"""Independent source-image inventory for high-risk notation objects.

The OMR engine is one semantic observer. This module deliberately derives visual
evidence from the scan itself so unanimous omissions by all preprocessing variants
cannot be mislabeled as high confidence. It is an audit layer: geometric candidates
are never injected into MusicXML without a separate relation/transaction decision.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from lxml import etree

from .layout import PageLayout, StaffSystem
from .score_ir import score_from_tree


@dataclass(frozen=True)
class VisualNotationCandidate:
    kind: str
    staff_index: int
    placement: str
    bbox: tuple[int, int, int, int]
    confidence: float
    geometry: tuple[tuple[str, float], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "staff_index": self.staff_index,
            "placement": self.placement,
            "bbox": list(self.bbox),
            "confidence": round(self.confidence, 6),
            "geometry": {key: round(value, 6) for key, value in self.geometry},
        }


@dataclass(frozen=True)
class NotationCoverageKind:
    kind: str
    confident_source_count: int
    emitted_count: int
    potential_omission_count: int
    unmatched_emitted_count: int
    coverage_ratio: float
    comparison_mode: str = "source-specific-count"
    source_count_excess: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "confident_source_count": self.confident_source_count,
            "emitted_count": self.emitted_count,
            "potential_omission_count": self.potential_omission_count,
            "unmatched_emitted_count": self.unmatched_emitted_count,
            "coverage_ratio": round(self.coverage_ratio, 6),
            "comparison_mode": self.comparison_mode,
            "source_count_excess": self.source_count_excess,
        }


@dataclass(frozen=True)
class NotationCoverageReport:
    image_path: str
    xml_path: str
    detector_version: str
    candidates: tuple[VisualNotationCandidate, ...]
    kinds: tuple[NotationCoverageKind, ...]
    emitted_unbalanced_slurs: int = 0
    emitted_unbalanced_ties: int = 0
    emitted_unbalanced_wedges: int = 0

    @property
    def potential_omission_count(self) -> int:
        return sum(item.potential_omission_count for item in self.kinds)

    @property
    def severe_structure_issue_count(self) -> int:
        return (
            self.potential_omission_count
            + self.emitted_unbalanced_slurs
            + self.emitted_unbalanced_ties
            + self.emitted_unbalanced_wedges
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": 1,
            "image_path": self.image_path,
            "xml_path": self.xml_path,
            "detector_version": self.detector_version,
            "potential_omission_count": self.potential_omission_count,
            "severe_structure_issue_count": self.severe_structure_issue_count,
            "emitted_unbalanced_slurs": self.emitted_unbalanced_slurs,
            "emitted_unbalanced_ties": self.emitted_unbalanced_ties,
            "emitted_unbalanced_wedges": self.emitted_unbalanced_wedges,
            "kinds": [item.to_dict() for item in self.kinds],
            "candidates": [item.to_dict() for item in self.candidates],
        }


DETECTOR_VERSION = "source-notation-geometry@7"
_CONFIDENT_SOURCE_FLOOR = {
    "curved_connector": 0.78,
    "crescendo": 0.82,
    "diminuendo": 0.82,
}
WEDGE_VERY_HIGH_CONFIDENCE_FLOOR = 0.95
WEDGE_GEOMETRIC_CONFIDENCE_FLOOR = 0.85
WEDGE_INTERSTITIAL_CONFIDENCE_FLOOR = 0.82
WEDGE_MINIMUM_LENGTH_SPACES = 2.5
WEDGE_MINIMUM_OPEN_SEPARATION_SPACES = 0.70
WEDGE_MAXIMUM_APEX_OPEN_RATIO = 0.45
WEDGE_CURVE_CONFLICT_MINIMUM_OVERLAP = 0.55
WEDGE_CURVE_CONFLICT_MINIMUM_CONFIDENCE = 0.84
WEDGE_CURVE_CONFLICT_MAXIMUM_FIT_P90_SPACES = 0.10
WEDGE_MAXIMUM_INK_DENSITY = 0.31
WEDGE_MAXIMUM_SHORT_SCAN_INK_DENSITY = 0.38
WEDGE_MAXIMUM_SHORT_SCAN_LENGTH_SPACES = 3.2


def _read_binary(image_path: Path) -> np.ndarray:
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"无法读取源图像：{image_path}")
    _threshold, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return binary


def _line_removed(binary: np.ndarray, spacing: float) -> np.ndarray:
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(15, int(round(spacing * 5.0))), 1),
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(11, int(round(spacing * 4.0)))),
    )
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    cleaned = cv2.subtract(binary, horizontal)
    cleaned = cv2.subtract(cleaned, vertical)
    return cleaned


def _staff_owner(
    staffs: list[StaffSystem],
    center_y: float,
) -> tuple[StaffSystem, str, float] | None:
    best: tuple[float, StaffSystem, str] | None = None
    for staff in staffs:
        if not staff.line_y:
            continue
        spacing = max(staff.spacing, 1.0)
        top_line = float(staff.line_y[0])
        bottom_line = float(staff.line_y[-1])
        if top_line - spacing * 6.0 <= center_y < top_line - spacing * 0.45:
            distance = (top_line - center_y) / spacing
            candidate = (distance, staff, "above")
        elif bottom_line + spacing * 0.45 < center_y <= bottom_line + spacing * 6.0:
            distance = (center_y - bottom_line) / spacing
            candidate = (distance, staff, "below")
        else:
            continue
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        return None
    return best[1], best[2], best[0]


def _wedge_staff_owner(
    staffs: list[StaffSystem],
    center_y: float,
) -> tuple[StaffSystem, str, float] | None:
    """Assign ordinary wedges plus bounded interstitial auxiliary-staff marks.

    An ossia/cue staff can be about 70 percent of the main staff size and too
    short for the page-wide staff projection.  Its hairpin then sits near the
    middle of the gap between two detected score systems.  Preserve that source
    candidate under the nearest main-staff owner; candidate-conditioned topology
    can subsequently reassign it to the visible hidden part.  The fallback never
    reaches page headings or margins and requires both neighbouring staves.
    """

    direct = _staff_owner(staffs, center_y)
    if direct is not None:
        return direct
    ordered = sorted(
        (staff for staff in staffs if staff.line_y),
        key=lambda staff: (staff.line_y[0], staff.index),
    )
    candidates: list[tuple[float, StaffSystem, str]] = []
    for upper, lower in zip(ordered, ordered[1:], strict=False):
        upper_bottom = float(upper.line_y[-1])
        lower_top = float(lower.line_y[0])
        if not upper_bottom < center_y < lower_top:
            continue
        upper_distance = (
            center_y - upper_bottom
        ) / max(1.0, float(upper.spacing))
        lower_distance = (
            lower_top - center_y
        ) / max(1.0, float(lower.spacing))
        if (
            6.0 < upper_distance <= 12.5
            and 6.0 < lower_distance <= 12.5
        ):
            candidates.extend(
                (
                    (upper_distance, upper, "below"),
                    (lower_distance, lower, "above"),
                )
            )
    if not candidates:
        return None
    distance, staff, placement = min(
        candidates,
        key=lambda item: (item[0], item[1].index),
    )
    return staff, placement, distance


def _segment_y(segment: tuple[float, float, float, float], x: float) -> float:
    x1, y1, x2, y2 = segment
    return y1 + (y2 - y1) * (x - x1) / max(x2 - x1, 1e-6)


def _merge_collinear_wedge_segments(
    segments: list[tuple[float, float, float, float]],
    spacing: float,
) -> list[tuple[float, float, float, float]]:
    """Add long line hypotheses assembled from scan-broken shallow fragments."""

    if len(segments) < 2:
        return segments
    ordered = sorted(segments, key=lambda item: (item[0], item[2], item[1]))
    parents = list(range(len(ordered)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    # Long, shallow hairpins can disappear for several staff spaces under a
    # fold, stem, or low-contrast halftone patch.  The strict slope, extrapolated
    # vertical distance, and final regression-residual gates make this wider
    # horizontal bridge safe while retaining the full physical opening.
    maximum_gap = max(4.0, spacing * 6.5)
    for left_index, left in enumerate(ordered):
        left_slope = (left[3] - left[1]) / max(left[2] - left[0], 1e-6)
        for right_index in range(left_index + 1, len(ordered)):
            right = ordered[right_index]
            if right[0] > left[2] + maximum_gap:
                break
            right_slope = (right[3] - right[1]) / max(
                right[2] - right[0],
                1e-6,
            )
            if (
                left_slope * right_slope <= 0
                or abs(left_slope - right_slope) > 0.018
            ):
                continue
            comparison_x = 0.5 * (
                max(left[0], right[0])
                + min(left[2], right[2])
            )
            if left[2] < right[0]:
                comparison_x = 0.5 * (left[2] + right[0])
            vertical_difference = abs(
                _segment_y(left, comparison_x)
                - _segment_y(right, comparison_x)
            )
            if vertical_difference > spacing * 0.25:
                continue
            union(left_index, right_index)

    groups: dict[int, list[tuple[float, float, float, float]]] = {}
    for index, segment in enumerate(ordered):
        groups.setdefault(find(index), []).append(segment)
    hypotheses = list(ordered)
    for members in groups.values():
        if len(members) < 2:
            continue
        x_values = np.array(
            [value for item in members for value in (item[0], item[2])],
            dtype=np.float64,
        )
        y_values = np.array(
            [value for item in members for value in (item[1], item[3])],
            dtype=np.float64,
        )
        slope, intercept = np.polyfit(x_values, y_values, 1)
        residual = np.abs(y_values - (slope * x_values + intercept))
        if float(np.percentile(residual, 90)) > spacing * 0.18:
            continue
        x1 = min(item[0] for item in members)
        x2 = max(item[2] for item in members)
        longest = max(item[2] - item[0] for item in members)
        if x2 - x1 < longest * 1.35:
            continue
        hypotheses.append(
            (
                float(x1),
                float(slope * x1 + intercept),
                float(x2),
                float(slope * x2 + intercept),
            )
        )
    return hypotheses


def _bbox_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection <= 0:
        return 0.0
    left_area = max(1, (left[2] - left[0]) * (left[3] - left[1]))
    right_area = max(1, (right[2] - right[0]) * (right[3] - right[1]))
    return intersection / max(left_area + right_area - intersection, 1)


def _minimum_area_overlap(
    left: VisualNotationCandidate,
    right: VisualNotationCandidate,
) -> float:
    overlap_width = max(
        0,
        min(left.bbox[2], right.bbox[2])
        - max(left.bbox[0], right.bbox[0]),
    )
    overlap_height = max(
        0,
        min(left.bbox[3], right.bbox[3])
        - max(left.bbox[1], right.bbox[1]),
    )
    left_area = max(
        1,
        (left.bbox[2] - left.bbox[0])
        * (left.bbox[3] - left.bbox[1]),
    )
    right_area = max(
        1,
        (right.bbox[2] - right.bbox[0])
        * (right.bbox[3] - right.bbox[1]),
    )
    return overlap_width * overlap_height / min(left_area, right_area)


def wedge_source_specificity_gate(
    candidate: VisualNotationCandidate,
    detected: tuple[VisualNotationCandidate, ...],
) -> tuple[bool, str]:
    """Whether source geometry specifically supports a hairpin rather than a curve."""

    if candidate.kind not in {"crescendo", "diminuendo"}:
        return False, "candidate is not a wedge"
    geometry = dict(candidate.geometry)
    length = geometry.get("length_spaces")
    open_separation = geometry.get("open_separation_spaces")
    apex_separation = geometry.get("apex_separation_spaces")
    ink_density = geometry.get("ink_density")
    # A release-gated semantic detector may corroborate this independently fitted
    # line pair.  It never supplies the geometry itself, and its class-specific
    # support is deliberately not copied into ``candidate.confidence`` where it
    # could accidentally affect curved-connector logic.
    effective_confidence = max(
        candidate.confidence,
        float(geometry.get("semantic_hairpin_support", 0.0)),
    )
    apex_open_ratio = (
        float(apex_separation) / max(float(open_separation), 1e-6)
        if apex_separation is not None and open_separation is not None
        else float("inf")
    )
    if (
        length is None
        or open_separation is None
        or apex_separation is None
        or float(length) < WEDGE_MINIMUM_LENGTH_SPACES
        or float(open_separation) < WEDGE_MINIMUM_OPEN_SEPARATION_SPACES
        or apex_open_ratio > WEDGE_MAXIMUM_APEX_OPEN_RATIO
        or (
            ink_density is not None
            and float(ink_density) > WEDGE_MAXIMUM_INK_DENSITY
            and not (
                float(length) <= WEDGE_MAXIMUM_SHORT_SCAN_LENGTH_SPACES
                and float(ink_density)
                <= WEDGE_MAXIMUM_SHORT_SCAN_INK_DENSITY
            )
        )
    ):
        return False, "source geometry is not wedge-specific"
    if effective_confidence >= WEDGE_VERY_HIGH_CONFIDENCE_FLOOR:
        return True, "very-high-confidence source wedge"
    confidence_floor = WEDGE_GEOMETRIC_CONFIDENCE_FLOOR
    if (
        float(geometry.get("interstitial_owner", 0.0)) >= 1.0
        and ink_density is not None
        and float(ink_density) <= 0.15
        and float(length) >= 4.0
        and apex_open_ratio <= 0.35
    ):
        confidence_floor = WEDGE_INTERSTITIAL_CONFIDENCE_FLOOR
    if effective_confidence < confidence_floor:
        return False, "source confidence below geometric automatic-write floor"

    for other in detected:
        if (
            other.kind != "curved_connector"
            or other.staff_index != candidate.staff_index
            or other.placement != candidate.placement
            or other.confidence < WEDGE_CURVE_CONFLICT_MINIMUM_CONFIDENCE
        ):
            continue
        fit = dict(other.geometry).get("fit_p90_spaces")
        if (
            fit is not None
            and float(fit) <= WEDGE_CURVE_CONFLICT_MAXIMUM_FIT_P90_SPACES
            and _minimum_area_overlap(candidate, other)
            >= WEDGE_CURVE_CONFLICT_MINIMUM_OVERLAP
        ):
            return False, "candidate overlaps an independently fitted curved connector"
    return True, "curve-disambiguated geometric source wedge"


def _deduplicate(
    candidates: list[VisualNotationCandidate],
    *,
    iou_floor: float,
) -> tuple[VisualNotationCandidate, ...]:
    retained: list[VisualNotationCandidate] = []
    for candidate in sorted(candidates, key=lambda item: (-item.confidence, item.bbox)):
        if any(
            candidate.kind == other.kind
            and candidate.staff_index == other.staff_index
            and _bbox_iou(candidate.bbox, other.bbox) >= iou_floor
            for other in retained
        ):
            continue
        retained.append(candidate)
    return tuple(sorted(retained, key=lambda item: (item.staff_index, item.bbox[0], item.kind)))


def _merge_wedge_fragments(
    candidates: tuple[VisualNotationCandidate, ...],
    staffs: list[StaffSystem],
) -> tuple[VisualNotationCandidate, ...]:
    """Collapse overlapping Hough views of one physical hairpin.

    A long hairpin normally produces several partially overlapping line pairs.
    IoU is a poor duplicate metric for those thin diagonal boxes: the apex fragment
    and open-end fragment can have little area intersection even though their
    horizontal intervals overlap.  This connected-component merge uses only
    same-kind candidates whose horizontal intervals overlap materially and whose
    vertical envelopes touch.  Views of a hairpin centred between two staves may
    initially receive opposite adjacent-staff owners; the merged physical box is
    therefore assigned again after grouping.  Adjacent hairpins do not overlap
    and simultaneous hairpins on separate staves do not touch vertically.
    """

    if len(candidates) < 2:
        return candidates
    spacing_by_staff = {
        staff.index: max(1.0, float(staff.spacing))
        for staff in staffs
    }
    parents = list(range(len(candidates)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, left in enumerate(candidates):
        for right_index in range(left_index + 1, len(candidates)):
            right = candidates[right_index]
            if left.kind != right.kind:
                continue
            same_owner = (
                left.staff_index == right.staff_index
                and left.placement == right.placement
            )
            if (
                not same_owner
                and abs(left.staff_index - right.staff_index) > 1
            ):
                continue
            horizontal_overlap = max(
                0,
                min(left.bbox[2], right.bbox[2]) - max(left.bbox[0], right.bbox[0]),
            )
            minimum_width = max(
                1,
                min(left.bbox[2] - left.bbox[0], right.bbox[2] - right.bbox[0]),
            )
            if horizontal_overlap / minimum_width < 0.22:
                continue
            vertical_gap = max(
                0,
                max(left.bbox[1], right.bbox[1]) - min(left.bbox[3], right.bbox[3]),
            )
            spacing = spacing_by_staff.get(left.staff_index, 12.0)
            if vertical_gap > spacing * 0.45:
                continue
            union(left_index, right_index)

    groups: dict[int, list[VisualNotationCandidate]] = {}
    for index, candidate in enumerate(candidates):
        groups.setdefault(find(index), []).append(candidate)

    merged: list[VisualNotationCandidate] = []
    for members in groups.values():
        if len(members) == 1:
            merged.append(members[0])
            continue
        best = max(members, key=lambda item: (item.confidence, item.bbox))
        x1 = min(item.bbox[0] for item in members)
        y1 = min(item.bbox[1] for item in members)
        x2 = max(item.bbox[2] for item in members)
        y2 = max(item.bbox[3] for item in members)
        spacing = spacing_by_staff.get(best.staff_index, 12.0)
        geometry = dict(best.geometry)
        geometry["length_spaces"] = (x2 - x1) / spacing
        geometry["merged_hough_views"] = float(len(members))
        owner = _staff_owner(staffs, (y1 + y2) * 0.5)
        staff_index = best.staff_index if owner is None else owner[0].index
        placement = best.placement if owner is None else owner[1]
        merged.append(
            VisualNotationCandidate(
                best.kind,
                staff_index,
                placement,
                (x1, y1, x2, y2),
                best.confidence,
                tuple(sorted(geometry.items())),
            )
        )
    return tuple(
        sorted(
            merged,
            key=lambda item: (item.staff_index, item.bbox[0], item.kind),
        )
    )


def _resolve_wedge_kind_conflicts(
    candidates: tuple[VisualNotationCandidate, ...],
    staffs: list[StaffSystem],
) -> tuple[VisualNotationCandidate, ...]:
    """Keep one direction when the same source strokes imply both hairpin kinds.

    A nearby slur or beam edge can pair with one side of a real hairpin and create
    a second, opposite-direction hypothesis over the same horizontal interval.
    Real adjacent ``<>`` hairpins meet at an endpoint but do not substantially
    overlap, so conflict resolution is limited to candidates covering at least
    60 percent of the shorter interval.  The sharper fitted apex is decisive;
    distance from the staff and detector confidence only break near ties.
    """

    if len(candidates) < 2:
        return candidates
    staff_by_index = {staff.index: staff for staff in staffs}
    losers: set[int] = set()

    def rank(candidate: VisualNotationCandidate) -> tuple[float, float, float]:
        geometry = dict(candidate.geometry)
        apex = float(geometry.get("apex_separation_spaces", 99.0))
        opening = max(
            float(geometry.get("open_separation_spaces", 0.0)),
            1e-6,
        )
        apex_quality = 1.0 - min(1.0, apex / opening)
        staff = staff_by_index.get(candidate.staff_index)
        distance = 0.0
        if staff is not None and staff.line_y:
            center_y = (candidate.bbox[1] + candidate.bbox[3]) * 0.5
            if candidate.placement == "above":
                distance = float(staff.line_y[0]) - center_y
            else:
                distance = center_y - float(staff.line_y[-1])
            distance /= max(1.0, float(staff.spacing))
        return (apex_quality, distance, candidate.confidence)

    for left_index, left in enumerate(candidates):
        if left_index in losers:
            continue
        for right_index in range(left_index + 1, len(candidates)):
            if right_index in losers:
                continue
            right = candidates[right_index]
            if (
                left.kind == right.kind
                or left.staff_index != right.staff_index
                or left.placement != right.placement
            ):
                continue
            horizontal_overlap = max(
                0,
                min(left.bbox[2], right.bbox[2])
                - max(left.bbox[0], right.bbox[0]),
            )
            shorter_width = max(
                1,
                min(
                    left.bbox[2] - left.bbox[0],
                    right.bbox[2] - right.bbox[0],
                ),
            )
            if horizontal_overlap / shorter_width < 0.60:
                continue
            left_rank = rank(left)
            right_rank = rank(right)
            # A tiny fit difference is not meaningful at scan resolution.  In
            # that case prefer the source line pair farther from the staff, where
            # hairpins conventionally sit outside slurs and articulation marks.
            if abs(left_rank[0] - right_rank[0]) <= 0.035:
                left_rank = (left_rank[1], left_rank[0], left_rank[2])
                right_rank = (right_rank[1], right_rank[0], right_rank[2])
            if left_rank >= right_rank:
                losers.add(right_index)
            else:
                losers.add(left_index)
                break
    return tuple(
        candidate
        for index, candidate in enumerate(candidates)
        if index not in losers
    )


def detect_wedges(
    binary: np.ndarray,
    layout: PageLayout,
    *,
    grayscale: np.ndarray | None = None,
) -> tuple[VisualNotationCandidate, ...]:
    staffs = layout.systems
    if not staffs:
        return ()
    spacing = float(np.median([staff.spacing for staff in staffs if staff.spacing > 0]))
    # A generic horizontal-line opening erases exactly the long, shallow
    # hairpins we need to retain. Detect sub-pixel source line segments instead.
    # Staff lines have effectively zero slope; later paired-slope, free-space and
    # density gates reject beams, accents and curved connectors.
    source = grayscale if grayscale is not None else cv2.bitwise_not(binary)
    # Printed and scanned shallow diagonals are often broken into a row of
    # halftone dots.  A small blur reconnects those dots for LSD without the
    # long-gap hallucinations produced by a permissive probabilistic Hough pass.
    line_detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    grouped_segments: list[
        tuple[tuple[float, float, float, float], int]
    ] = []
    # The half-resolution pass reconnects longer dotted scan strokes that remain
    # split at native resolution.  Pairing is kept within a scale so two unrelated
    # fragments from different resamplings cannot manufacture a wedge.
    for group, scale in enumerate((1.0, 0.5)):
        if scale == 1.0:
            scaled_source = source
        else:
            scaled_source = cv2.resize(
                source,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
        line_source = cv2.GaussianBlur(scaled_source, (5, 5), 0)
        lines = line_detector.detect(line_source)[0]
        if lines is None:
            continue
        scale_segments: list[tuple[float, float, float, float]] = []
        for raw in lines[:, 0, :]:
            x1, y1, x2, y2 = (
                float(value) / scale
                for value in raw
            )
            if x2 < x1:
                x1, y1, x2, y2 = x2, y2, x1, y1
            dx = x2 - x1
            if dx <= 0:
                continue
            slope = (y2 - y1) / dx
            if not 0.008 <= abs(slope) <= 0.55:
                continue
            if not spacing * 2.0 <= dx <= spacing * 42.0:
                continue
            scale_segments.append((x1, y1, x2, y2))
        grouped_segments.extend(
            (segment, group)
            for segment in _merge_collinear_wedge_segments(
                scale_segments,
                spacing,
            )
        )
    if not grouped_segments:
        return ()

    candidates: list[VisualNotationCandidate] = []
    for left_index, (first, first_group) in enumerate(grouped_segments):
        first_slope = (first[3] - first[1]) / (first[2] - first[0])
        for second, second_group in grouped_segments[left_index + 1:]:
            if first_group != second_group:
                continue
            second_slope = (second[3] - second[1]) / (second[2] - second[0])
            if first_slope * second_slope >= 0:
                continue
            overlap_left = max(first[0], second[0])
            overlap_right = min(first[2], second[2])
            overlap = overlap_right - overlap_left
            if overlap < spacing * 2.2:
                continue
            if overlap / max(min(first[2] - first[0], second[2] - second[0]), 1.0) < 0.50:
                continue
            separation_left = abs(_segment_y(first, overlap_left) - _segment_y(second, overlap_left))
            separation_right = abs(_segment_y(first, overlap_right) - _segment_y(second, overlap_right))
            apex_separation = min(separation_left, separation_right)
            open_separation = max(separation_left, separation_right)
            if apex_separation > spacing * 1.25:
                continue
            if not spacing * 0.55 <= open_separation <= spacing * 6.0:
                continue
            if open_separation - apex_separation < spacing * 0.55:
                continue
            slope_ratio = max(abs(first_slope), abs(second_slope)) / max(
                min(abs(first_slope), abs(second_slope)),
                1e-6,
            )
            if slope_ratio > 3.5:
                continue
            center_y = float(
                np.mean(
                    [
                        _segment_y(first, overlap_left),
                        _segment_y(second, overlap_left),
                        _segment_y(first, overlap_right),
                        _segment_y(second, overlap_right),
                    ]
                )
            )
            owner = _wedge_staff_owner(staffs, center_y)
            if owner is None:
                continue
            staff, placement, distance = owner
            kind = "crescendo" if separation_left < separation_right else "diminuendo"
            ys = [
                _segment_y(first, overlap_left),
                _segment_y(second, overlap_left),
                _segment_y(first, overlap_right),
                _segment_y(second, overlap_right),
            ]
            bbox = (
                int(np.floor(overlap_left)),
                int(np.floor(min(ys) - 2)),
                int(np.ceil(overlap_right)),
                int(np.ceil(max(ys) + 2)),
            )
            # A direction hairpin belongs in notation free-space. Requiring its
            # whole envelope to clear the staff prevents noteheads or a slanted
            # beam from being merged into a nearby real hairpin.
            clearance = spacing * 0.25
            if (
                placement == "below"
                and bbox[1] <= float(staff.line_y[-1]) + clearance
            ) or (
                placement == "above"
                and bbox[3] >= float(staff.line_y[0]) - clearance
            ):
                continue
            crop = binary[
                max(0, bbox[1]):min(binary.shape[0], bbox[3]),
                max(0, bbox[0]):min(binary.shape[1], bbox[2]),
            ]
            ink_density = (
                float(np.count_nonzero(crop)) / float(crop.size)
                if crop.size
                else 1.0
            )
            apex_quality = max(0.0, 1.0 - apex_separation / max(spacing * 1.25, 1.0))
            opening_quality = min(1.0, open_separation / max(spacing * 1.5, 1.0))
            length_quality = min(1.0, overlap / max(spacing * 7.0, 1.0))
            distance_quality = max(0.0, 1.0 - max(0.0, distance - 4.5) / 1.5)
            confidence = (
                0.52
                + apex_quality * 0.16
                + opening_quality * 0.12
                + length_quality * 0.12
                + distance_quality * 0.08
            )
            candidates.append(
                VisualNotationCandidate(
                    kind,
                    staff.index,
                    placement,
                    bbox,
                    min(0.995, confidence),
                    (
                        ("apex_separation_spaces", apex_separation / spacing),
                        ("open_separation_spaces", open_separation / spacing),
                        ("length_spaces", overlap / spacing),
                        ("ink_density", ink_density),
                        ("interstitial_owner", float(distance > 6.0)),
                    ),
                )
            )
    # Merge line-pair views before confidence de-duplication so the retained
    # box spans the physical apex and open end, not only a clean centre fragment.
    merged = _merge_wedge_fragments(tuple(candidates), staffs)
    resolved = _resolve_wedge_kind_conflicts(merged, staffs)
    return _deduplicate(
        list(resolved),
        iou_floor=0.32,
    )


def detect_curved_connectors(
    binary: np.ndarray,
    layout: PageLayout,
) -> tuple[VisualNotationCandidate, ...]:
    staffs = layout.systems
    if not staffs:
        return ()
    spacing = float(np.median([staff.spacing for staff in staffs if staff.spacing > 0]))
    cleaned = _line_removed(binary, spacing)
    # Join one-pixel scan breaks but do not bridge separate note objects.
    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(2, int(round(spacing * 0.28))), 1)),
    )
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (cleaned > 0).astype(np.uint8),
        connectivity=8,
    )
    candidates: list[VisualNotationCandidate] = []
    for label in range(1, component_count):
        x, y, width, height, area = (int(value) for value in stats[label])
        if not spacing * 2.4 <= width <= spacing * 36.0:
            continue
        if not max(3, spacing * 0.35) <= height <= spacing * 8.0:
            continue
        if width / max(height, 1) < 1.75:
            continue
        if not width * 0.18 <= area <= width * 5.0:
            continue
        ys, xs = np.where(labels[y:y + height, x:x + width] == label)
        if len(xs) < max(8, int(width * 0.25)):
            continue
        unique_x = np.unique(xs)
        if len(unique_x) / max(width, 1) < 0.48:
            continue
        median_y = np.asarray([np.median(ys[xs == x_value]) for x_value in unique_x], dtype=float)
        x_values = unique_x.astype(float)
        try:
            coefficients = np.polyfit(x_values, median_y, 2)
        except (ValueError, np.linalg.LinAlgError):
            continue
        predicted = np.polyval(coefficients, x_values)
        residual = np.abs(predicted - median_y)
        if float(np.median(residual)) > spacing * 0.42:
            continue
        if float(np.quantile(residual, 0.90)) > spacing * 1.05:
            continue
        left_x = float(x_values[0])
        right_x = float(x_values[-1])
        middle_x = (left_x + right_x) / 2.0
        left_y = float(np.polyval(coefficients, left_x))
        right_y = float(np.polyval(coefficients, right_x))
        middle_y = float(np.polyval(coefficients, middle_x))
        sagitta = abs(middle_y - (left_y + right_y) / 2.0)
        if not spacing * 0.30 <= sagitta <= spacing * 5.5:
            continue
        derivative_left = 2.0 * coefficients[0] * left_x + coefficients[1]
        derivative_right = 2.0 * coefficients[0] * right_x + coefficients[1]
        if derivative_left * derivative_right >= 0:
            continue
        curvature_denominator = 2.0 * coefficients[0]
        if abs(curvature_denominator) < 1e-12:
            continue
        vertex_x = -coefficients[1] / curvature_denominator
        vertex_ratio = (vertex_x - left_x) / max(right_x - left_x, 1.0)
        if not 0.08 <= vertex_ratio <= 0.92:
            continue
        center_y = y + float(np.median(median_y))
        owner = _staff_owner(staffs, center_y)
        if owner is None:
            # Slurs frequently sit just inside the staff. Assign by nearest staff
            # center, but keep lower confidence than an outside-staff arc.
            staff = min(
                staffs,
                key=lambda item: abs(
                    center_y - float(np.mean(item.line_y or [item.top, item.bottom]))
                ),
            )
            placement = (
                "above"
                if center_y < float(np.mean(staff.line_y or [staff.top, staff.bottom]))
                else "below"
            )
            distance_quality = 0.45
        else:
            staff, placement, distance = owner
            distance_quality = max(0.45, 1.0 - max(0.0, distance - 4.0) / 2.0)
        fit_quality = max(0.0, 1.0 - float(np.quantile(residual, 0.90)) / max(spacing * 1.05, 1.0))
        length_quality = min(1.0, width / max(spacing * 9.0, 1.0))
        curvature_quality = min(1.0, sagitta / max(spacing * 1.2, 1.0))
        confidence = (
            0.46
            + fit_quality * 0.20
            + length_quality * 0.13
            + curvature_quality * 0.13
            + distance_quality * 0.08
        )
        candidates.append(
            VisualNotationCandidate(
                "curved_connector",
                staff.index,
                placement,
                (x, y, x + width, y + height),
                min(0.99, confidence),
                (
                    ("sagitta_spaces", sagitta / spacing),
                    ("length_spaces", width / spacing),
                    ("fit_p90_spaces", float(np.quantile(residual, 0.90)) / spacing),
                ),
            )
        )
    return _deduplicate(candidates, iou_floor=0.38)


def detect_notation_candidates(
    image_path: Path,
    layout: PageLayout,
) -> tuple[VisualNotationCandidate, ...]:
    binary = _read_binary(image_path)
    grayscale = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    candidates = [
        *detect_curved_connectors(binary, layout),
        *detect_wedges(binary, layout, grayscale=grayscale),
    ]
    return tuple(
        sorted(
            candidates,
            key=lambda item: (item.staff_index, item.bbox[0], item.kind),
        )
    )


def _emitted_inventory(
    tree: etree._ElementTree,
) -> tuple[dict[str, int], int, int, int]:
    score = score_from_tree(tree)
    counts = {"curved_connector": 0, "crescendo": 0, "diminuendo": 0}
    slur_starts = slur_stops = 0
    tie_starts = tie_stops = 0
    wedge_starts: dict[tuple[str, int, str], int] = {}
    wedge_stops: dict[tuple[str, int, str], int] = {}
    for part in score.effective_parts:
        for measure in part.measures:
            for note in measure.notes:
                for slur_type, _number in note.slurs:
                    if slur_type == "start":
                        counts["curved_connector"] += 1
                        slur_starts += 1
                    elif slur_type == "stop":
                        slur_stops += 1
                if "start" in note.ties:
                    counts["curved_connector"] += 1
                    tie_starts += 1
                if "stop" in note.ties:
                    tie_stops += 1
            for direction in measure.directions:
                if direction.kind != "wedge":
                    continue
                key = (part.id, direction.staff, direction.end_value or "1")
                if direction.value in {"crescendo", "diminuendo"}:
                    counts[direction.value] += 1
                    wedge_starts[key] = wedge_starts.get(key, 0) + 1
                elif direction.value == "stop":
                    wedge_stops[key] = wedge_stops.get(key, 0) + 1
    unbalanced_slurs = abs(slur_starts - slur_stops)
    unbalanced_ties = abs(tie_starts - tie_stops)
    wedge_keys = set(wedge_starts) | set(wedge_stops)
    unbalanced_wedges = sum(
        abs(wedge_starts.get(key, 0) - wedge_stops.get(key, 0))
        for key in wedge_keys
    )
    return counts, unbalanced_slurs, unbalanced_ties, unbalanced_wedges


def audit_notation_coverage(
    image_path: Path,
    xml_path: Path,
    layout: PageLayout,
    *,
    candidates: tuple[VisualNotationCandidate, ...] | None = None,
) -> NotationCoverageReport:
    candidates = (
        candidates
        if candidates is not None
        else detect_notation_candidates(image_path, layout)
    )
    tree = etree.parse(
        str(xml_path),
        etree.XMLParser(resolve_entities=False, no_network=True),
    )
    emitted, unbalanced_slurs, unbalanced_ties, unbalanced_wedges = _emitted_inventory(tree)
    kinds: list[NotationCoverageKind] = []
    for kind in ("curved_connector", "crescendo", "diminuendo"):
        confident = sum(
            candidate.kind == kind
            and (
                wedge_source_specificity_gate(candidate, tuple(candidates))[0]
                if kind in {"crescendo", "diminuendo"}
                else candidate.confidence >= _CONFIDENT_SOURCE_FLOOR[kind]
            )
            for candidate in candidates
        )
        emitted_count = emitted[kind]
        source_count_excess = max(0, confident - emitted_count)
        # A curve component is not yet an object-level relation observation.  One
        # printed slur can be split into two disconnected components by noteheads,
        # staff-line removal, a scan break, or the curve apex.  Conversely, beams,
        # braces and clef fragments can satisfy the quadratic component detector.
        # Without source-to-note endpoint matching, subtracting the two global
        # counts cannot prove that any particular slur/tie is missing.  Preserve
        # the difference as diagnostic evidence, but do not route it as a user
        # doubt or a release blocker.  Explicit unbalanced start/stop topology
        # below remains actionable, as do source-specific wedge transactions.
        count_only_diagnostic = kind == "curved_connector"
        potential_omissions = 0 if count_only_diagnostic else source_count_excess
        unmatched_emitted = max(0, emitted_count - confident)
        ratio = (
            min(1.0, emitted_count / confident)
            if confident > 0
            else (1.0 if emitted_count == 0 else 0.0)
        )
        kinds.append(
            NotationCoverageKind(
                kind,
                confident,
                emitted_count,
                potential_omissions,
                unmatched_emitted,
                ratio,
                (
                    "diagnostic-count-only"
                    if count_only_diagnostic
                    else "source-specific-count"
                ),
                source_count_excess,
            )
        )
    return NotationCoverageReport(
        str(image_path),
        str(xml_path),
        DETECTOR_VERSION,
        candidates,
        tuple(kinds),
        unbalanced_slurs,
        unbalanced_ties,
        unbalanced_wedges,
    )
