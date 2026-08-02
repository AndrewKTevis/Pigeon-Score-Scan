from __future__ import annotations

"""Staff-aware visual evidence for conservative OMR candidate selection.

This module does not attempt to recognise notes independently.  It extracts low-level,
deterministic measure-region evidence (ink density, symbol proxies, staff-normalised
profiles and coarse event-local grids) and compares it with an already parsed
:class:`MeasureIR`.  A compact bundled CPU model turns that compatibility vector into a
bounded secondary probability.

The visual model is intentionally subordinate to MusicXML validity, rhythm checks and
cross-variant semantic agreement.  Its purpose is to break close ties where candidates
differ in visible event placement or rhythmic complexity, not to invent notation.
"""

import base64
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .layout import PageLayout, StaffSystem
from .model_registry import load_verified_json
from .score_ir import MeasureIR
from .tree_model import (
    VerifiedGradientBoostingModel,
    VerifiedRandomForestModel,
    stable_sigmoid,
)


VISUAL_SCALAR_FEATURE_NAMES = (
    "ink_density",
    "nonstaff_ink_density",
    "component_density",
    "notehead_proxy",
    "open_notehead_proxy",
    "stem_proxy",
    "beam_proxy",
    "onset_proxy",
    "compact_mark_proxy",
    "accidental_proxy",
    "above_ink_density",
    "below_ink_density",
)

X_PROFILE_NAMES = tuple(f"x_ink_profile_{index}" for index in range(8))
STAFF_PROFILE_NAMES = tuple(f"staff_ink_profile_{index}" for index in range(9))
VISUAL_FEATURE_NAMES = VISUAL_SCALAR_FEATURE_NAMES + X_PROFILE_NAMES + STAFF_PROFILE_NAMES

# Coarse enough to tolerate ordinary engraving variation while retaining event order.
EVENT_X_BINS = 16
EVENT_Y_BINS = 17
EVENT_GRID_SIZE = EVENT_X_BINS * EVENT_Y_BINS
RHYTHM_GUARD_WIDTH = 128
RHYTHM_GUARD_HEIGHT = 64
SYMBOL_GUARD_WIDTH = 256
SYMBOL_GUARD_HEIGHT = 96
EVENT_GRID_NAMES = (
    "event_ink_grid",
    "pitched_notehead_grid",
    "pitch_guard_notehead_grid",
    "pitch_guard_strict_notehead_grid",
    "beam_grid",
    "compact_mark_grid",
    "accidental_grid",
    "open_notehead_grid",
)

SEMANTIC_FEATURE_NAMES = (
    "anchor_count",
    "pitched_count",
    "rest_count",
    "chord_extra_count",
    "beam_complexity",
    "direction_count",
    "articulation_count",
    "accidental_count",
    "dot_count",
    "open_notehead_count",
    "grace_count",
)

ONSET_PROFILE_NAMES = tuple(f"onset_profile_{index}" for index in range(8))
PITCH_PROFILE_NAMES = tuple(f"pitch_profile_{index}" for index in range(9))

V3_FEATURE_NAMES = VISUAL_FEATURE_NAMES + SEMANTIC_FEATURE_NAMES + ONSET_PROFILE_NAMES + PITCH_PROFILE_NAMES + (
    "anchor_notehead_gap",
    "anchor_onset_gap",
    "pitched_stem_gap",
    "beam_gap",
    "direction_ink_gap",
    "rest_component_gap",
    "compact_mark_gap",
    "accidental_gap",
    "open_notehead_gap",
    "onset_profile_l1_gap",
    "onset_profile_transport_gap",
    "pitch_profile_l1_gap",
    "pitch_profile_transport_gap",
    "complexity_density_gap",
)

EVENT_LOCAL_FEATURE_NAMES = (
    "event_grid_l1_gap",
    "event_grid_cosine_gap",
    "event_grid_neighbourhood_miss",
    "event_grid_pitch_transport_gap",
    "notehead_grid_l1_gap",
    "notehead_grid_neighbourhood_miss",
    "notehead_grid_pitch_transport_gap",
    "beam_grid_l1_gap",
    "beam_grid_neighbourhood_miss",
    "compact_grid_l1_gap",
    "compact_grid_neighbourhood_miss",
    "accidental_grid_l1_gap",
    "accidental_grid_neighbourhood_miss",
    "open_notehead_grid_l1_gap",
    "open_notehead_grid_neighbourhood_miss",
    "beam_attachment_gap",
    "accidental_attachment_gap",
    "open_notehead_attachment_gap",
)

# The general visual calibrator intentionally remains on the proven v4 feature schema.
# Pitch-only repair has a narrower failure mode: one wrong staff position can be
# diluted by an otherwise correct measure.  These onset-conditioned, bidirectional
# notehead gaps are therefore exposed only to the pitch-patch safety path.  They are
# deterministic and candidate-relative; they cannot infer or write a pitch.
PITCH_LOCAL_FEATURE_NAMES = (
    "notehead_exact_cell_miss",
    "notehead_near_cell_miss",
    "notehead_vertical_chamfer_gap",
    "notehead_severe_vertical_miss",
    "notehead_visual_unmatched_ratio",
    "notehead_column_centroid_gap",
    "notehead_column_order_gap",
)

V4_FEATURE_NAMES = V3_FEATURE_NAMES + EVENT_LOCAL_FEATURE_NAMES
FEATURE_NAMES = V4_FEATURE_NAMES


LEGACY_VISUAL_FEATURE_NAMES = (
    "ink_density",
    "nonstaff_ink_density",
    "component_density",
    "notehead_proxy",
    "stem_proxy",
    "beam_proxy",
    "onset_proxy",
    "above_ink_density",
    "below_ink_density",
)

LEGACY_SEMANTIC_FEATURE_NAMES = (
    "anchor_count",
    "pitched_count",
    "rest_count",
    "chord_extra_count",
    "beam_complexity",
    "direction_count",
    "articulation_count",
    "accidental_count",
    "grace_count",
)

LEGACY_FEATURE_NAMES = LEGACY_VISUAL_FEATURE_NAMES + LEGACY_SEMANTIC_FEATURE_NAMES + (
    "anchor_notehead_gap",
    "anchor_onset_gap",
    "pitched_stem_gap",
    "beam_gap",
    "direction_ink_gap",
    "rest_component_gap",
    "complexity_density_gap",
)


@dataclass(frozen=True)
class VisualMeasureEvidence:
    page_index: int
    system_index: int
    measure_index: int
    bbox: tuple[int, int, int, int]
    spacing: float
    ink_density: float
    nonstaff_ink_density: float
    component_density: float
    notehead_proxy: float
    open_notehead_proxy: float
    stem_proxy: float
    beam_proxy: float
    onset_proxy: float
    compact_mark_proxy: float
    accidental_proxy: float
    above_ink_density: float
    below_ink_density: float
    x_ink_profile: tuple[float, ...]
    staff_ink_profile: tuple[float, ...]
    event_ink_grid: tuple[float, ...] = ()
    pitched_notehead_grid: tuple[float, ...] = ()
    pitch_guard_notehead_grid: tuple[float, ...] = ()
    pitch_guard_strict_notehead_grid: tuple[float, ...] = ()
    beam_grid: tuple[float, ...] = ()
    compact_mark_grid: tuple[float, ...] = ()
    accidental_grid: tuple[float, ...] = ()
    open_notehead_grid: tuple[float, ...] = ()
    rhythm_guard_image: str = ""
    symbol_guard_image: str = ""

    def vector(self) -> tuple[float, ...]:
        scalars = tuple(float(getattr(self, name)) for name in VISUAL_SCALAR_FEATURE_NAMES)
        return scalars + tuple(float(value) for value in self.x_ink_profile) + tuple(
            float(value) for value in self.staff_ink_profile
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "page_index": self.page_index,
            "system_index": self.system_index,
            "measure_index": self.measure_index,
            "bbox": list(self.bbox),
            "spacing": round(self.spacing, 4),
            **{name: round(float(getattr(self, name)), 7) for name in VISUAL_SCALAR_FEATURE_NAMES},
            "x_ink_profile": [round(float(value), 7) for value in self.x_ink_profile],
            "staff_ink_profile": [round(float(value), 7) for value in self.staff_ink_profile],
            **{
                name: [round(float(value), 7) for value in getattr(self, name)]
                for name in EVENT_GRID_NAMES
            },
            "rhythm_guard_image": self.rhythm_guard_image,
            "symbol_guard_image": self.symbol_guard_image,
        }


@dataclass(frozen=True)
class VisualCompatibilityResult:
    probability: float
    weight_factor: float
    model_version: str


def _groups(indices: np.ndarray, max_gap: int = 2) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    groups: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for raw in indices[1:]:
        value = int(raw)
        if value - previous > max_gap:
            groups.append((start, previous))
            start = value
        previous = value
    groups.append((start, previous))
    return groups


def _count_components(mask: np.ndarray, minimum_area: int) -> tuple[int, list[tuple[int, int, int, int, int]]]:
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components: list[tuple[int, int, int, int, int]] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if area >= minimum_area:
            components.append((x, y, width, height, area))
    return len(components), components


def _normalised_bins(values: np.ndarray, bins: int, scale: float = 3.0) -> tuple[float, ...]:
    """Return a fixed-length non-negative distribution robust to crop dimensions."""
    flattened = np.asarray(values, dtype=np.float64).ravel()
    if flattened.size == 0 or bins <= 0:
        return tuple(0.0 for _ in range(max(0, bins)))
    edges = np.linspace(0, flattened.size, bins + 1, dtype=np.int64)
    totals = np.asarray(
        [float(np.sum(flattened[edges[index] : edges[index + 1]])) for index in range(bins)],
        dtype=np.float64,
    )
    denominator = float(np.sum(totals))
    if denominator <= 0.0 or not math.isfinite(denominator):
        return tuple(0.0 for _ in range(bins))
    return tuple(float(max(0.0, min(scale, scale * value / denominator))) for value in totals)


def _normalise_grid(values: np.ndarray, *, scale: float = 3.0) -> tuple[float, ...]:
    grid = np.asarray(values, dtype=np.float64)
    if grid.shape != (EVENT_Y_BINS, EVENT_X_BINS):
        return tuple(0.0 for _ in range(EVENT_GRID_SIZE))
    if np.any(grid > 0.0):
        grid = cv2.GaussianBlur(grid, (3, 3), 0.7, borderType=cv2.BORDER_REPLICATE)
    total = float(np.sum(grid))
    if total <= 0.0 or not math.isfinite(total):
        return tuple(0.0 for _ in range(EVENT_GRID_SIZE))
    normalised = np.clip(scale * grid / total, 0.0, scale)
    return tuple(float(value) for value in normalised.ravel())


def _mask_event_grid(mask: np.ndarray) -> tuple[float, ...]:
    if mask.ndim != 2 or mask.size == 0:
        return tuple(0.0 for _ in range(EVENT_GRID_SIZE))
    height, width = mask.shape
    y_edges = np.linspace(0, height, EVENT_Y_BINS + 1, dtype=np.int64)
    x_edges = np.linspace(0, width, EVENT_X_BINS + 1, dtype=np.int64)
    grid = np.zeros((EVENT_Y_BINS, EVENT_X_BINS), dtype=np.float64)
    for yi in range(EVENT_Y_BINS):
        y1, y2 = int(y_edges[yi]), int(y_edges[yi + 1])
        for xi in range(EVENT_X_BINS):
            x1, x2 = int(x_edges[xi]), int(x_edges[xi + 1])
            if y2 > y1 and x2 > x1:
                grid[yi, xi] = float(np.count_nonzero(mask[y1:y2, x1:x2]))
    return _normalise_grid(grid)


def _point_event_grid(
    points: Iterable[tuple[float, float, float]],
    *,
    width: int,
    profile_top: float,
    profile_bottom: float,
) -> tuple[float, ...]:
    if width <= 0 or profile_bottom <= profile_top:
        return tuple(0.0 for _ in range(EVENT_GRID_SIZE))
    grid = np.zeros((EVENT_Y_BINS, EVENT_X_BINS), dtype=np.float64)
    usable_left = max(0.0, width * 0.025)
    usable_right = max(usable_left + 1.0, width * 0.975)
    for x, y, weight in points:
        if not all(math.isfinite(v) for v in (x, y, weight)) or weight <= 0.0:
            continue
        xr = (x - usable_left) / max(usable_right - usable_left, 1.0)
        yr = (y - profile_top) / max(profile_bottom - profile_top, 1.0)
        if not (-0.10 <= xr <= 1.10 and -0.10 <= yr <= 1.10):
            continue
        xi = min(EVENT_X_BINS - 1, max(0, int(max(0.0, min(0.999999, xr)) * EVENT_X_BINS)))
        yi = min(EVENT_Y_BINS - 1, max(0, int(max(0.0, min(0.999999, yr)) * EVENT_Y_BINS)))
        grid[yi, xi] += float(weight)
    return _normalise_grid(grid)


def extract_crop_features(
    crop: np.ndarray,
    *,
    spacing: float,
    staff_top: float,
    staff_bottom: float,
) -> dict[str, float | tuple[float, ...]]:
    """Extract normalised visual features from a single source measure crop."""
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()
    if gray.size == 0:
        empty: dict[str, float | tuple[float, ...]] = {
            name: 0.0 for name in VISUAL_SCALAR_FEATURE_NAMES
        }
        empty["x_ink_profile"] = tuple(0.0 for _ in range(8))
        empty["staff_ink_profile"] = tuple(0.0 for _ in range(9))
        empty.update({name: tuple(0.0 for _ in range(EVENT_GRID_SIZE)) for name in EVENT_GRID_NAMES})
        return empty

    spacing = max(float(spacing), 3.0)
    height, width = gray.shape
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    staff_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(round(spacing * 5.5))), 1))
    staff_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, staff_kernel)
    raw_nonstaff = cv2.subtract(binary, staff_mask)
    # Remove isolated scan dust without erasing small augmentation dots.
    tiny_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    nonstaff = cv2.morphologyEx(raw_nonstaff, cv2.MORPH_OPEN, tiny_kernel)

    ink_density = float(np.count_nonzero(binary)) / max(binary.size, 1)
    nonstaff_density = float(np.count_nonzero(nonstaff)) / max(nonstaff.size, 1)

    min_area = max(2, int(round(spacing * spacing * 0.025)))
    component_count, components = _count_components(nonstaff, min_area)
    width_units = max(width / spacing, 1.0)
    component_density = min(4.0, component_count / width_units / 1.7)

    noteheads = 0
    notehead_points: list[tuple[float, float, float]] = []
    for _x, _y, comp_w, comp_h, area in components:
        aspect = comp_w / max(comp_h, 1)
        fill = area / max(comp_w * comp_h, 1)
        if (
            0.35 * spacing <= comp_w <= 2.25 * spacing
            and 0.28 * spacing <= comp_h <= 1.65 * spacing
            and 0.42 <= aspect <= 2.4
            and fill >= 0.22
            and 0.10 * spacing * spacing <= area <= 2.2 * spacing * spacing
        ):
            noteheads += 1
    head_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(3, int(round(spacing * 0.46))) | 1, max(3, int(round(spacing * 0.30))) | 1),
    )
    head_mask = cv2.morphologyEx(raw_nonstaff, cv2.MORPH_OPEN, head_kernel)
    _head_count, head_components = _count_components(
        head_mask, max(2, int(round(spacing * spacing * 0.055)))
    )
    head_staff_min = float(staff_top) - spacing * 2.0
    head_staff_max = float(staff_bottom) + spacing * 2.0
    for x, y, comp_w, comp_h, area in head_components:
        centre_y = y + comp_h / 2.0
        aspect = comp_w / max(comp_h, 1)
        if (
            head_staff_min <= centre_y <= head_staff_max
            and 0.30 * spacing <= comp_w <= 2.0 * spacing
            and 0.22 * spacing <= comp_h <= 1.35 * spacing
            and 0.40 <= aspect <= 2.8
            and area <= 2.2 * spacing * spacing
        ):
            notehead_points.append((x + comp_w / 2.0, centre_y, 1.0))

    notehead_proxy = min(3.0, max(noteheads, len(notehead_points)) / 16.0)

    # Pitch repair uses a separate conservative detector rather than changing the
    # general visual-selector evidence.  Filled noteheads joined to stems or damaged by
    # staff-line subtraction still contain a thick ink core, while staff lines, stems
    # and ordinary scan scratches do not.  The extra points are de-duplicated against
    # the established morphology detector and are only consumed by the pitch-patch
    # safety path; visual calibrator v4 continues to receive the original grid.
    pitch_guard_points = list(notehead_points)
    distance = cv2.distanceTransform((binary > 0).astype(np.uint8), cv2.DIST_L2, 5)
    core_mask = (distance >= max(1.0, spacing * 0.23)).astype(np.uint8) * 255
    _core_count, core_components = _count_components(
        core_mask, max(1, int(round(spacing * spacing * 0.006)))
    )
    for x, y, comp_w, comp_h, area in core_components:
        centre_x = x + comp_w / 2.0
        centre_y = y + comp_h / 2.0
        aspect = comp_w / max(comp_h, 1)
        if not (
            head_staff_min <= centre_y <= head_staff_max
            and area <= 1.2 * spacing * spacing
            and comp_w <= 2.2 * spacing
            and comp_h <= 1.4 * spacing
            and 0.25 <= aspect <= 3.5
        ):
            continue
        if any(
            abs(existing_x - centre_x) <= 0.45 * spacing
            and abs(existing_y - centre_y) <= 0.35 * spacing
            for existing_x, existing_y, _weight in pitch_guard_points
        ):
            continue
        pitch_guard_points.append((centre_x, centre_y, 1.0))

    # A second, stricter channel runs the same thick-core test after staff-line
    # subtraction.  It misses some joined heads but rejects staff intersections and
    # coarse print artefacts which can fool the inclusive channel.  Both channels feed
    # one paired visual guard; neither writes or votes for a pitch independently.
    pitch_guard_strict_points = list(notehead_points)
    strict_distance = cv2.distanceTransform(
        (raw_nonstaff > 0).astype(np.uint8), cv2.DIST_L2, 5
    )
    strict_core_mask = (strict_distance >= max(1.0, spacing * 0.23)).astype(np.uint8) * 255
    _strict_core_count, strict_core_components = _count_components(
        strict_core_mask, max(1, int(round(spacing * spacing * 0.006)))
    )
    for x, y, comp_w, comp_h, area in strict_core_components:
        centre_x = x + comp_w / 2.0
        centre_y = y + comp_h / 2.0
        aspect = comp_w / max(comp_h, 1)
        if not (
            head_staff_min <= centre_y <= head_staff_max
            and area <= 1.2 * spacing * spacing
            and comp_w <= 2.2 * spacing
            and comp_h <= 1.4 * spacing
            and 0.25 <= aspect <= 3.5
        ):
            continue
        if any(
            abs(existing_x - centre_x) <= 0.45 * spacing
            and abs(existing_y - centre_y) <= 0.35 * spacing
            for existing_x, existing_y, _weight in pitch_guard_strict_points
        ):
            continue
        pitch_guard_strict_points.append((centre_x, centre_y, 1.0))

    # Hollow noteheads are one of the few direct visual cues separating whole/half
    # notes from filled noteheads.  Count enclosed holes near the staff rather than
    # relying on component fill alone, because half-note stems expand the parent
    # component bounding box substantially.
    open_noteheads = 0
    open_notehead_points: list[tuple[float, float, float]] = []
    contours, hierarchy = cv2.findContours(raw_nonstaff, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        hierarchy_rows = hierarchy[0]
        staff_margin = int(round(spacing * 1.15))
        staff_min_y = max(0, int(round(staff_top)) - staff_margin)
        staff_max_y = min(height, int(round(staff_bottom)) + staff_margin)
        for contour_index, contour in enumerate(contours):
            parent = int(hierarchy_rows[contour_index][3])
            if parent < 0:
                continue
            x, y, comp_w, comp_h = cv2.boundingRect(contour)
            area = abs(float(cv2.contourArea(contour)))
            centre_y = y + comp_h / 2.0
            if not (staff_min_y <= centre_y <= staff_max_y):
                continue
            if (
                0.16 * spacing <= comp_w <= 1.25 * spacing
                and 0.10 * spacing <= comp_h <= 0.95 * spacing
                and 0.012 * spacing * spacing <= area <= 0.65 * spacing * spacing
            ):
                open_noteheads += 1
                open_notehead_points.append((x + comp_w / 2.0, centre_y, 1.0))
    open_notehead_proxy = min(3.0, open_noteheads / 8.0)

    # Use the un-opened mask for tiny musical marks.  The normal dust-removal pass is
    # deliberately allowed to erase isolated specks, but augmentation dots and
    # staccato marks must remain observable to the compatibility model.  Shape and
    # staff-proximity limits prevent page dust from becoming a strong positive cue.
    raw_min_area = max(1, int(round(spacing * spacing * 0.012)))
    _raw_count, raw_components = _count_components(raw_nonstaff, raw_min_area)
    compact_marks = 0
    accidental_shapes = 0
    compact_mark_points: list[tuple[float, float, float]] = []
    accidental_points: list[tuple[float, float, float]] = []
    staff_min_y = max(0.0, float(staff_top) - spacing * 1.8)
    staff_max_y = min(float(height), float(staff_bottom) + spacing * 1.8)
    for x, y, comp_w, comp_h, area in components:
        centre_y = y + comp_h / 2.0
        if not (staff_min_y <= centre_y <= staff_max_y):
            continue
        fill = area / max(comp_w * comp_h, 1)
        if (
            0.10 * spacing <= comp_w <= 0.48 * spacing
            and 0.10 * spacing <= comp_h <= 0.48 * spacing
            and 0.012 * spacing * spacing <= area <= 0.18 * spacing * spacing
            and 0.58 <= comp_w / max(comp_h, 1) <= 1.72
            and fill >= 0.34
        ):
            compact_marks += 1
            compact_mark_points.append((x + comp_w / 2.0, centre_y, 1.0))
    for x, y, comp_w, comp_h, area in raw_components:
        centre_y = y + comp_h / 2.0
        if not (staff_min_y <= centre_y <= staff_max_y):
            continue
        fill = area / max(comp_w * comp_h, 1)
        aspect = comp_w / max(comp_h, 1)
        if (
            0.24 * spacing <= comp_w <= 1.40 * spacing
            and 1.05 * spacing <= comp_h <= 3.25 * spacing
            and 0.12 <= aspect <= 0.88
            and 0.065 * spacing * spacing <= area <= 1.65 * spacing * spacing
            and 0.055 <= fill <= 0.78
        ):
            accidental_shapes += 1
            accidental_points.append((x + comp_w / 2.0, centre_y, 1.0))
    compact_mark_proxy = min(3.0, compact_marks / 8.0)
    accidental_proxy = min(3.0, accidental_shapes / 8.0)

    # Staff-normalised projection evidence gives the visual layer coarse spatial
    # information without attempting symbol-level recognition.  Remove full-height
    # vertical rules first so crop edges and barlines do not dominate the horizontal
    # event profile; stems are shorter than this kernel and remain represented.
    profile_top = max(0, int(round(float(staff_top) - spacing * 2.0)))
    profile_bottom = min(height, int(round(float(staff_bottom) + spacing * 2.0)) + 1)
    profile_mask = raw_nonstaff[profile_top:profile_bottom, :].copy()
    if profile_mask.size:
        bar_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, max(5, int(round(spacing * 3.75)))),
        )
        bar_mask = cv2.morphologyEx(profile_mask, cv2.MORPH_OPEN, bar_kernel)
        profile_mask = cv2.subtract(profile_mask, bar_mask)
        edge = max(1, int(round(width * 0.025)))
        if profile_mask.shape[1] > edge * 2:
            profile_mask[:, :edge] = 0
            profile_mask[:, -edge:] = 0
        x_ink_profile = _normalised_bins(np.count_nonzero(profile_mask, axis=0), 8)
        staff_ink_profile = _normalised_bins(np.count_nonzero(profile_mask, axis=1), 9)
        event_ink_grid = _mask_event_grid(profile_mask)
        rhythm_guard_resized = cv2.resize(
            profile_mask, (RHYTHM_GUARD_WIDTH, RHYTHM_GUARD_HEIGHT), interpolation=cv2.INTER_AREA
        )
        encoded_ok, encoded_png = cv2.imencode(
            ".png", rhythm_guard_resized, [cv2.IMWRITE_PNG_COMPRESSION, 9]
        )
        rhythm_guard_image = (
            base64.b64encode(encoded_png.tobytes()).decode("ascii")
            if encoded_ok
            else ""
        )
        symbol_guard_resized = cv2.resize(
            profile_mask, (SYMBOL_GUARD_WIDTH, SYMBOL_GUARD_HEIGHT), interpolation=cv2.INTER_AREA
        )
        symbol_ok, symbol_png = cv2.imencode(
            ".png", symbol_guard_resized, [cv2.IMWRITE_PNG_COMPRESSION, 9]
        )
        symbol_guard_image = (
            base64.b64encode(symbol_png.tobytes()).decode("ascii")
            if symbol_ok
            else ""
        )
    else:
        x_ink_profile = tuple(0.0 for _ in range(8))
        staff_ink_profile = tuple(0.0 for _ in range(9))
        event_ink_grid = tuple(0.0 for _ in range(EVENT_GRID_SIZE))
        rhythm_guard_image = ""
        symbol_guard_image = ""

    pitched_notehead_grid = _point_event_grid(
        notehead_points, width=width, profile_top=profile_top, profile_bottom=profile_bottom
    )
    pitch_guard_notehead_grid = _point_event_grid(
        pitch_guard_points, width=width, profile_top=profile_top, profile_bottom=profile_bottom
    )
    pitch_guard_strict_notehead_grid = _point_event_grid(
        pitch_guard_strict_points,
        width=width,
        profile_top=profile_top,
        profile_bottom=profile_bottom,
    )
    compact_mark_grid = _point_event_grid(
        compact_mark_points, width=width, profile_top=profile_top, profile_bottom=profile_bottom
    )
    accidental_grid = _point_event_grid(
        accidental_points, width=width, profile_top=profile_top, profile_bottom=profile_bottom
    )
    open_notehead_grid = _point_event_grid(
        open_notehead_points, width=width, profile_top=profile_top, profile_bottom=profile_bottom
    )

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, int(round(spacing * 2.25)))))
    vertical = cv2.morphologyEx(nonstaff, cv2.MORPH_OPEN, vertical_kernel)
    vertical_profile = np.count_nonzero(vertical, axis=0)
    stem_columns = np.where(vertical_profile >= max(4, int(round(spacing * 1.65))))[0]
    stem_groups = _groups(stem_columns, max_gap=max(1, int(round(spacing * 0.18))))
    stem_proxy = min(3.0, len(stem_groups) / 16.0)

    beam_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(5, int(round(spacing * 2.2))), max(1, int(round(spacing * 0.14)))),
    )
    beams = cv2.morphologyEx(nonstaff, cv2.MORPH_OPEN, beam_kernel)
    _beam_count, beam_components = _count_components(
        beams, max(3, int(round(spacing * spacing * 0.08)))
    )
    filtered_beams = [
        (x, y, comp_w, comp_h, area)
        for x, y, comp_w, comp_h, area in beam_components
        if width * 0.025 <= x + comp_w / 2.0 <= width * 0.975
        and staff_min_y <= y + comp_h / 2.0 <= staff_max_y
        and spacing * 0.8 <= comp_w <= width * 0.45
        and comp_h <= spacing * 1.2
    ]
    beam_proxy = min(3.0, len(filtered_beams) / 8.0)
    beam_grid = _point_event_grid(
        [
            (x + comp_w / 2.0, y + comp_h / 2.0, max(1.0, comp_w / max(spacing * 2.0, 1.0)))
            for x, y, comp_w, comp_h, _area in filtered_beams
        ],
        width=width,
        profile_top=profile_top,
        profile_bottom=profile_bottom,
    )

    x_profile = np.count_nonzero(nonstaff, axis=0).astype(np.float32)
    smooth_width = max(3, int(round(spacing * 0.55))) | 1
    x_profile = cv2.GaussianBlur(x_profile.reshape(1, -1), (smooth_width, 1), 0).ravel()
    threshold = max(float(np.percentile(x_profile, 72)), spacing * 0.55)
    onset_columns = np.where(x_profile >= threshold)[0]
    onset_groups = _groups(onset_columns, max_gap=max(1, int(round(spacing * 0.45))))
    onset_proxy = min(3.0, len(onset_groups) / 16.0)

    top = int(round(max(0.0, min(float(height), staff_top))))
    bottom = int(round(max(0.0, min(float(height), staff_bottom))))
    if bottom < top:
        top, bottom = bottom, top
    above = nonstaff[:top, :]
    below = nonstaff[bottom:, :]
    above_density = float(np.count_nonzero(above)) / max(above.size, 1) if above.size else 0.0
    below_density = float(np.count_nonzero(below)) / max(below.size, 1) if below.size else 0.0

    return {
        "ink_density": min(1.0, ink_density),
        "nonstaff_ink_density": min(1.0, nonstaff_density),
        "component_density": component_density,
        "notehead_proxy": notehead_proxy,
        "open_notehead_proxy": open_notehead_proxy,
        "stem_proxy": stem_proxy,
        "beam_proxy": beam_proxy,
        "onset_proxy": onset_proxy,
        "compact_mark_proxy": compact_mark_proxy,
        "accidental_proxy": accidental_proxy,
        "above_ink_density": min(1.0, above_density),
        "below_ink_density": min(1.0, below_density),
        "x_ink_profile": x_ink_profile,
        "staff_ink_profile": staff_ink_profile,
        "event_ink_grid": event_ink_grid,
        "pitched_notehead_grid": pitched_notehead_grid,
        "pitch_guard_notehead_grid": pitch_guard_notehead_grid,
        "pitch_guard_strict_notehead_grid": pitch_guard_strict_notehead_grid,
        "beam_grid": beam_grid,
        "compact_mark_grid": compact_mark_grid,
        "accidental_grid": accidental_grid,
        "open_notehead_grid": open_notehead_grid,
        "rhythm_guard_image": rhythm_guard_image,
        "symbol_guard_image": symbol_guard_image,
    }


def _system_boundaries(system: StaffSystem, count_override: int | None = None) -> list[int]:
    tolerance = max(4, int(round(system.spacing * 0.75)))
    boundaries = [system.left, *system.barlines, system.right]
    boundaries = sorted(max(system.left, min(system.right, int(value))) for value in boundaries)
    merged: list[int] = []
    for value in boundaries:
        if not merged or value - merged[-1] > tolerance:
            merged.append(value)
    if count_override is None and len(merged) >= 2:
        observed_count = len(merged) - 1
        if system.barlines and abs(observed_count - system.measure_count) <= 1:
            return merged
    count = max(1, int(count_override if count_override is not None else system.measure_count))
    return [int(round(system.left + (system.right - system.left) * index / count)) for index in range(count + 1)]


def extract_page_measure_evidence(
    image_path: Path,
    layout: PageLayout | None,
    *,
    page_index: int,
    target_measure_count: int | None = None,
) -> tuple[VisualMeasureEvidence, ...]:
    if layout is None or not layout.systems:
        return ()
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return ()
    height, width = image.shape
    result: list[VisualMeasureEvidence] = []
    global_index = 0
    system_counts: list[int]
    observed_system_counts = [
        max(1, len(_system_boundaries(system)) - 1)
        for system in layout.systems
    ]
    if target_measure_count and target_measure_count >= len(layout.systems):
        if sum(observed_system_counts) == target_measure_count:
            # Preserve source-image barline geometry, including pickups and
            # irregular engraving, when the independent count already agrees.
            system_counts = observed_system_counts
        else:
            # Barline detection is intentionally conservative and can miss internal
            # boundaries on dense scans.  Only after a real count mismatch do we
            # distribute the provisional OMR count by usable system width.
            widths = [max(1.0, float(system.right - system.left)) for system in layout.systems]
            total_width = sum(widths)
            raw = [target_measure_count * width / total_width for width in widths]
            system_counts = [max(1, int(value)) for value in raw]
            remainder = target_measure_count - sum(system_counts)
            fractions = sorted(
                range(len(raw)),
                key=lambda index: (raw[index] - int(raw[index]), widths[index]),
                reverse=remainder > 0,
            )
            cursor = 0
            while remainder != 0 and fractions:
                index = fractions[cursor % len(fractions)]
                if remainder > 0:
                    system_counts[index] += 1
                    remainder -= 1
                elif system_counts[index] > 1:
                    system_counts[index] -= 1
                    remainder += 1
                cursor += 1
                if cursor > len(fractions) * max(target_measure_count, 4):
                    break
    else:
        system_counts = [max(1, system.measure_count) for system in layout.systems]

    for system, system_count, observed_count in zip(
        layout.systems,
        system_counts,
        observed_system_counts,
        strict=True,
    ):
        boundaries = _system_boundaries(
            system,
            None if system_count == observed_count else system_count,
        )
        for local_index, (left, right) in enumerate(zip(boundaries, boundaries[1:], strict=False), start=1):
            if right - left < max(8, system.spacing * 1.5):
                continue
            pad = max(1, int(round(system.spacing * 0.18)))
            x1 = max(0, left - pad)
            x2 = min(width, right + pad)
            y1 = max(0, system.top)
            y2 = min(height, system.bottom + 1)
            crop = image[y1:y2, x1:x2]
            features = extract_crop_features(
                crop,
                spacing=system.spacing,
                staff_top=system.line_y[0] - y1,
                staff_bottom=system.line_y[-1] - y1,
            )
            global_index += 1
            result.append(
                VisualMeasureEvidence(
                    page_index=page_index,
                    system_index=system.index,
                    measure_index=global_index,
                    bbox=(x1, y1, x2, y2),
                    spacing=system.spacing,
                    **features,
                )
            )
    return tuple(result)


def semantic_features(measure: MeasureIR) -> tuple[float, ...]:
    regular = [note for note in measure.notes if not note.grace]
    anchors = [note for note in regular if not note.chord]
    pitched = [note for note in regular if note.pitch is not None and not note.rest]
    rests = [note for note in regular if note.rest]
    chords = [note for note in regular if note.chord]
    grace = [note for note in measure.notes if note.grace]
    beam_units = {
        "eighth": 1.0,
        "16th": 2.0,
        "32nd": 3.0,
        "64th": 4.0,
        "128th": 5.0,
    }
    beam_complexity = sum(beam_units.get(note.note_type, 0.0) for note in regular)
    articulations = sum(len(note.articulations) + len(note.ornaments) for note in measure.notes)
    accidentals = sum(bool(note.accidental) for note in measure.notes)
    dots = sum(max(0, int(note.dots)) for note in measure.notes)
    open_noteheads = sum(
        note.note_type in {"whole", "half"}
        for note in regular
        if not note.rest and note.pitch is not None
    )
    return (
        min(3.0, len(anchors) / 16.0),
        min(3.0, len(pitched) / 16.0),
        min(3.0, len(rests) / 8.0),
        min(3.0, len(chords) / 8.0),
        min(3.0, beam_complexity / 20.0),
        min(3.0, len(measure.directions) / 4.0),
        min(3.0, articulations / 8.0),
        min(3.0, accidentals / 8.0),
        min(3.0, dots / 8.0),
        min(3.0, open_noteheads / 8.0),
        min(3.0, len(grace) / 6.0),
    )


def _profile_from_positions(
    positions: Iterable[tuple[float, float]],
    bins: int,
    *,
    minimum: float,
    maximum: float,
) -> tuple[float, ...]:
    totals = np.zeros(bins, dtype=np.float64)
    span = max(maximum - minimum, 1e-9)
    for position, weight in positions:
        if not math.isfinite(position) or not math.isfinite(weight) or weight <= 0.0:
            continue
        ratio = max(0.0, min(0.999999, (position - minimum) / span))
        totals[min(bins - 1, int(ratio * bins))] += weight
    denominator = float(np.sum(totals))
    if denominator <= 0.0:
        return tuple(0.0 for _ in range(bins))
    return tuple(float(3.0 * value / denominator) for value in totals)


def _diatonic_index(pitch: object) -> int | None:
    if not hasattr(pitch, "step") or not hasattr(pitch, "octave"):
        return None
    order = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}
    step = str(getattr(pitch, "step", "")).upper()
    if step not in order:
        return None
    return int(getattr(pitch, "octave", 4)) * 7 + order[step]


def _staff_position(note: object, clef: tuple[str, int, int] | None) -> float | None:
    pitch = getattr(note, "pitch", None)
    diatonic = _diatonic_index(pitch)
    if diatonic is None:
        return None
    sign, line, octave_change = clef or ("G", 2, 0)
    sign = str(sign).upper()
    line = max(1, min(5, int(line or 2)))
    octave_change = int(octave_change or 0)
    if sign == "F":
        reference = 3 * 7 + 3  # F3
    elif sign == "C":
        reference = 4 * 7 + 0  # C4
    else:
        reference = 4 * 7 + 4  # G4
    reference += octave_change * 7
    # Position zero is the bottom staff line; every line is two diatonic steps.
    return float(2 * (line - 1) + (diatonic - reference))



def staff_position_for_note(
    note: object, clef: tuple[str, int, int] | None
) -> float | None:
    """Return the diatonic half-line position used by visual evidence geometry."""

    return _staff_position(note, clef)

def semantic_projection_features(measure: MeasureIR) -> tuple[tuple[float, ...], tuple[float, ...]]:
    regular = [note for note in measure.notes if not note.grace]
    anchors = [note for note in regular if not note.chord]
    expected = measure.expected_duration
    if expected is None or expected <= 0:
        end_points = [note.onset + max(note.duration, Fraction(0, 1)) for note in anchors]
        expected = max(end_points, default=Fraction(1, 1))
    onset_positions: list[tuple[float, float]] = []
    for note in anchors:
        ratio = float(note.onset / max(expected, Fraction(1, 64)))
        weight = 1.0
        weight += 0.18 * max(0, int(note.dots))
        weight += 0.14 * len(note.articulations)
        weight += 0.12 * bool(note.accidental)
        weight += 0.08 * sum(
            other.chord and other.onset == note.onset
            for other in regular
        )
        onset_positions.append((ratio, weight))
    onset_profile = _profile_from_positions(onset_positions, 8, minimum=0.0, maximum=1.0)

    pitch_positions: list[tuple[float, float]] = []
    for note in regular:
        if note.rest or note.pitch is None:
            continue
        staff_position = _staff_position(note, measure.clef)
        if staff_position is None:
            continue
        weight = 1.0 + 0.12 * bool(note.accidental) + 0.08 * max(0, int(note.dots))
        pitch_positions.append((staff_position, weight))
    # Include four ledger steps below and above the five-line staff.
    pitch_profile_bottom_up = _profile_from_positions(
        pitch_positions,
        9,
        minimum=-4.0,
        maximum=12.0,
    )
    # Visual rows are ordered top-to-bottom; reverse the semantic bottom-up profile.
    pitch_profile = tuple(reversed(pitch_profile_bottom_up))
    return onset_profile, pitch_profile


def _normalised_semantic_grid(
    positions: Iterable[tuple[float, float, float]],
) -> tuple[float, ...]:
    grid = np.zeros((EVENT_Y_BINS, EVENT_X_BINS), dtype=np.float64)
    for x_ratio, y_ratio, weight in positions:
        if not all(math.isfinite(value) for value in (x_ratio, y_ratio, weight)) or weight <= 0.0:
            continue
        xi = min(EVENT_X_BINS - 1, max(0, int(max(0.0, min(0.999999, x_ratio)) * EVENT_X_BINS)))
        yi = min(EVENT_Y_BINS - 1, max(0, int(max(0.0, min(0.999999, y_ratio)) * EVENT_Y_BINS)))
        grid[yi, xi] += float(weight)
    return _normalise_grid(grid)


def _semantic_pitch_grid(
    measure: MeasureIR,
    event_indices: set[int] | None = None,
) -> tuple[float, ...]:
    """Project pitched events to the staff-normalised grid.

    ``event_indices`` refers to ``MeasureIR.notes`` indices.  Restricting the grid to
    the events touched by one pitch transaction prevents unrelated correct notes from
    diluting an isolated visual contradiction.
    """
    anchors = [note for note in measure.notes if not note.grace and not note.chord]
    expected = measure.expected_duration
    if expected is None or expected <= 0:
        expected = max(
            (note.onset + max(note.duration, Fraction(0, 1)) for note in anchors),
            default=Fraction(1, 1),
        )
    expected = max(expected, Fraction(1, 64))
    positions: list[tuple[float, float, float]] = []
    for index, note in enumerate(measure.notes):
        if note.grace or note.rest or note.pitch is None:
            continue
        if event_indices is not None and index not in event_indices:
            continue
        staff_position = _staff_position(note, measure.clef)
        if staff_position is None:
            continue
        onset_ratio = max(0.0, min(1.0, float(note.onset / expected)))
        x_ratio = 0.075 + 0.85 * onset_ratio
        bounded_position = max(-4.0, min(12.0, staff_position))
        y_ratio = (12.0 - bounded_position) / 16.0
        positions.append((x_ratio, y_ratio, 1.0))
    return _normalised_semantic_grid(positions)


def semantic_event_grids(measure: MeasureIR) -> dict[str, tuple[float, ...]]:
    regular = [note for note in measure.notes if not note.grace]
    anchors = [note for note in regular if not note.chord]
    expected = measure.expected_duration
    if expected is None or expected <= 0:
        expected = max(
            (note.onset + max(note.duration, Fraction(0, 1)) for note in anchors),
            default=Fraction(1, 1),
        )
    expected = max(expected, Fraction(1, 64))

    event_positions: list[tuple[float, float, float]] = []
    notehead_positions: list[tuple[float, float, float]] = []
    beam_positions: list[tuple[float, float, float]] = []
    compact_positions: list[tuple[float, float, float]] = []
    accidental_positions: list[tuple[float, float, float]] = []
    open_positions: list[tuple[float, float, float]] = []
    for note in regular:
        onset_ratio = max(0.0, min(1.0, float(note.onset / expected)))
        x_ratio = 0.075 + 0.85 * onset_ratio
        if note.rest or note.pitch is None:
            staff_position = 4.0
        else:
            value = _staff_position(note, measure.clef)
            if value is None:
                continue
            staff_position = max(-4.0, min(12.0, value))
        y_ratio = (12.0 - staff_position) / 16.0
        event_positions.append((x_ratio, y_ratio, 1.0 + 0.18 * bool(note.chord)))
        if not note.rest and note.pitch is not None:
            notehead_positions.append((x_ratio, y_ratio, 1.0))
        beam_weight = {
            "eighth": 1.0,
            "16th": 2.0,
            "32nd": 3.0,
            "64th": 4.0,
            "128th": 5.0,
        }.get(note.note_type, 0.0)
        if beam_weight > 0.0:
            if note.rest or note.pitch is None:
                beam_y = y_ratio
            else:
                beam_y = y_ratio - 0.36 if staff_position <= 4.0 else y_ratio + 0.36
            beam_positions.append((x_ratio, max(0.0, min(1.0, beam_y)), beam_weight))
        for dot_index in range(max(0, int(note.dots))):
            compact_positions.append((x_ratio + 0.026 + dot_index * 0.018, y_ratio, 1.0))
        for mark_index, _mark in enumerate(note.articulations):
            compact_positions.append((x_ratio, max(0.0, y_ratio - 0.11 - 0.035 * mark_index), 0.8))
        if note.accidental:
            accidental_positions.append((x_ratio - 0.035, y_ratio, 1.0))
        if not note.rest and note.pitch is not None and note.note_type in {"whole", "half"}:
            open_positions.append((x_ratio, y_ratio, 1.0))
    return {
        "event_ink_grid": _normalised_semantic_grid(event_positions),
        "pitched_notehead_grid": _semantic_pitch_grid(measure),
        "beam_grid": _normalised_semantic_grid(beam_positions),
        "compact_mark_grid": _normalised_semantic_grid(compact_positions),
        "accidental_grid": _normalised_semantic_grid(accidental_positions),
        "open_notehead_grid": _normalised_semantic_grid(open_positions),
    }


def _grid_array(values: tuple[float, ...]) -> np.ndarray:
    if len(values) != EVENT_GRID_SIZE:
        return np.zeros((EVENT_Y_BINS, EVENT_X_BINS), dtype=np.float64)
    array = np.asarray(values, dtype=np.float64).reshape(EVENT_Y_BINS, EVENT_X_BINS)
    return np.where(np.isfinite(array), np.maximum(array, 0.0), 0.0)


def _grid_pair_metrics(
    source_values: tuple[float, ...], semantic_values: tuple[float, ...]
) -> tuple[float, float, float]:
    source = _grid_array(source_values) / 3.0
    semantic = _grid_array(semantic_values) / 3.0
    l1 = min(1.0, float(np.mean(np.abs(source - semantic))))
    source_flat = source.ravel()
    semantic_flat = semantic.ravel()
    denominator = float(np.linalg.norm(source_flat) * np.linalg.norm(semantic_flat))
    cosine_gap = 1.0 if denominator <= 0.0 and bool(np.any(source_flat) or np.any(semantic_flat)) else 0.0
    if denominator > 0.0:
        cosine_gap = min(1.0, max(0.0, 1.0 - float(np.dot(source_flat, semantic_flat) / denominator)))
    if not np.any(semantic):
        neighbourhood_miss = 0.0 if not np.any(source) else 1.0
    else:
        semantic_neighbourhood = cv2.dilate(
            (semantic > 0.001).astype(np.uint8), np.ones((3, 3), np.uint8)
        )
        mass = float(np.sum(source))
        neighbourhood_miss = 1.0 if mass <= 0.0 else min(
            1.0, float(np.sum(source[semantic_neighbourhood == 0])) / mass
        )
    return l1, cosine_gap, neighbourhood_miss


def _grid_pitch_transport_gap(
    source_values: tuple[float, ...], semantic_values: tuple[float, ...]
) -> float:
    source = _grid_array(source_values)
    semantic = _grid_array(semantic_values)
    total_weight = weighted_error = 0.0
    rows = np.arange(EVENT_Y_BINS, dtype=np.float64)
    for column_index in range(EVENT_X_BINS):
        source_column = source[:, column_index]
        semantic_column = semantic[:, column_index]
        weight = float(np.sum(semantic_column))
        if weight <= 0.0:
            continue
        semantic_y = float(np.sum(rows * semantic_column)) / weight
        source_mass = float(np.sum(source_column))
        error = 1.0 if source_mass <= 0.0 else min(
            1.0,
            abs(float(np.sum(rows * source_column)) / source_mass - semantic_y)
            / max(EVENT_Y_BINS - 1, 1),
        )
        total_weight += weight
        weighted_error += weight * error
    return min(1.0, weighted_error / total_weight) if total_weight > 0.0 else 0.0


def _notehead_local_metrics(
    source_values: tuple[float, ...], semantic_values: tuple[float, ...]
) -> tuple[float, float, float, float, float, float, float]:
    """Compare pitched-notehead grids without averaging away isolated pitch errors.

    Visual detections and semantic anchors are matched only within neighbouring onset
    columns.  Vertical distances are then measured in staff-normalised grid rows.  A
    one-row displacement is approximately one diatonic staff step in the supported
    pitch range, while larger distances capture octave and clef-like failures.

    All returned values are gaps in ``[0, 1]`` where zero is best.  Empty/empty grids
    are compatible; one-sided empty grids are a full mismatch.  The implementation is
    pure and deterministic so training and deployment use the exact same transform.
    """
    source = _grid_array(source_values)
    semantic = _grid_array(semantic_values)
    source_points = [
        (int(y), int(x), float(source[y, x]))
        for y, x in np.argwhere(source >= 0.03)
    ]
    semantic_points = [
        (int(y), int(x), float(semantic[y, x]))
        for y, x in np.argwhere(semantic >= 0.03)
    ]
    if not source_points and not semantic_points:
        return (0.0,) * 7
    if not source_points or not semantic_points:
        return (1.0,) * 7

    semantic_mass = sum(weight for _y, _x, weight in semantic_points)
    source_mass = sum(weight for _y, _x, weight in source_points)

    exact_supported = near_supported = vertical_error = severe_mass = 0.0
    semantic_column_centres: list[tuple[int, float, float]] = []
    for sy, sx, weight in semantic_points:
        local = [(vy, vx, vw) for vy, vx, vw in source_points if abs(vx - sx) <= 1]
        if not local:
            vertical_error += weight
            severe_mass += weight
            continue
        nearest_y = min(abs(vy - sy) for vy, _vx, _vw in local)
        same_cell_mass = sum(vw for vy, vx, vw in local if vy == sy and abs(vx - sx) <= 1)
        near_cell_mass = sum(vw for vy, vx, vw in local if abs(vy - sy) <= 1 and abs(vx - sx) <= 1)
        exact_supported += weight * min(1.0, same_cell_mass / max(weight, 1e-9))
        near_supported += weight * min(1.0, near_cell_mass / max(weight, 1e-9))
        vertical_error += weight * min(1.0, nearest_y / 4.0)
        severe_mass += weight * float(nearest_y > 1)

        column_mass = sum(vw for _vy, _vx, vw in local)
        if column_mass > 0.0:
            visual_y = sum(vy * vw for vy, _vx, vw in local) / column_mass
            semantic_column_centres.append((sx, float(sy), float(visual_y)))

    unmatched_visual = 0.0
    for vy, vx, weight in source_points:
        if not any(abs(sx - vx) <= 1 and abs(sy - vy) <= 1 for sy, sx, _sw in semantic_points):
            unmatched_visual += weight

    exact_miss = 1.0 - exact_supported / max(semantic_mass, 1e-9)
    near_miss = 1.0 - near_supported / max(semantic_mass, 1e-9)
    chamfer = vertical_error / max(semantic_mass, 1e-9)
    severe = severe_mass / max(semantic_mass, 1e-9)
    visual_unmatched = unmatched_visual / max(source_mass, 1e-9)

    if semantic_column_centres:
        centroid_gap = sum(
            min(1.0, abs(visual_y - semantic_y) / 4.0)
            for _x, semantic_y, visual_y in semantic_column_centres
        ) / len(semantic_column_centres)
        ordered = sorted(semantic_column_centres)
        if len(ordered) < 2:
            order_gap = 0.0
        else:
            semantic_deltas = np.diff([item[1] for item in ordered])
            visual_deltas = np.diff([item[2] for item in ordered])
            order_gap = float(np.mean(np.minimum(1.0, np.abs(semantic_deltas - visual_deltas) / 4.0)))
    else:
        centroid_gap = 1.0
        order_gap = 1.0

    return tuple(
        min(1.0, max(0.0, float(value)))
        for value in (
            exact_miss,
            near_miss,
            chamfer,
            severe,
            visual_unmatched,
            centroid_gap,
            order_gap,
        )
    )  # type: ignore[return-value]


def _attachment_profile(
    marker_values: tuple[float, ...], anchor_values: tuple[float, ...]
) -> tuple[float, ...]:
    marker = _grid_array(marker_values)
    anchors = _grid_array(anchor_values)
    anchor_columns = np.flatnonzero(np.sum(anchors, axis=0) >= 0.05)
    profile = np.zeros(EVENT_X_BINS, dtype=np.float64)
    if not len(anchor_columns):
        return tuple(float(value) for value in profile)
    marker_masses = np.sum(marker, axis=0)
    for marker_column in np.flatnonzero(marker_masses >= 0.03):
        nearest = int(np.argmin(np.abs(anchor_columns - marker_column)))
        ordinal_bin = 0 if len(anchor_columns) == 1 else int(
            round(nearest * (EVENT_X_BINS - 1) / (len(anchor_columns) - 1))
        )
        profile[ordinal_bin] += float(marker_masses[marker_column])
    maximum = float(np.max(profile)) if profile.size else 0.0
    if maximum > 0.0:
        profile /= maximum
    return tuple(float(value) for value in profile)


def _attachment_gap(
    source_marker: tuple[float, ...],
    source_anchors: tuple[float, ...],
    semantic_marker: tuple[float, ...],
    semantic_anchors: tuple[float, ...],
) -> float:
    left = np.asarray(_attachment_profile(source_marker, source_anchors), dtype=np.float64)
    right = np.asarray(_attachment_profile(semantic_marker, semantic_anchors), dtype=np.float64)
    if not np.any(left) and not np.any(right):
        return 0.0
    if not np.any(left) or not np.any(right):
        return 1.0
    l1 = float(np.mean(np.abs(left - right)))
    transport = float(np.mean(np.abs(np.cumsum(left) - np.cumsum(right))))
    return min(1.0, 0.65 * l1 + 0.35 * transport)


def _profile_gaps(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, float]:
    if len(left) != len(right) or not left:
        return 1.0, 1.0
    left_array = np.asarray(left, dtype=np.float64) / 3.0
    right_array = np.asarray(right, dtype=np.float64) / 3.0
    l1 = float(np.mean(np.abs(left_array - right_array)))
    transport = float(np.mean(np.abs(np.cumsum(left_array) - np.cumsum(right_array))))
    return min(1.0, l1), min(1.0, transport)


def v3_compatibility_vector(evidence: VisualMeasureEvidence, measure: MeasureIR) -> list[float]:
    visual = list(evidence.vector())
    semantic = list(semantic_features(measure))
    onset_profile, pitch_profile = semantic_projection_features(measure)
    (
        anchor,
        pitched,
        rests,
        _chords,
        beam,
        directions,
        articulations,
        accidentals,
        dots,
        open_noteheads,
        grace,
    ) = semantic
    ink_complexity = min(3.0, evidence.nonstaff_ink_density * 16.0)
    direction_ink = min(3.0, (evidence.above_ink_density + evidence.below_ink_density) * 14.0)
    component_proxy = min(3.0, evidence.component_density)
    compact_semantic = min(3.0, articulations + dots)
    onset_l1, onset_transport = _profile_gaps(evidence.x_ink_profile, onset_profile)
    pitch_l1, pitch_transport = _profile_gaps(evidence.staff_ink_profile, pitch_profile)
    semantic_complexity = min(
        3.0,
        0.42 * anchor
        + 0.23 * beam
        + 0.14 * directions
        + 0.07 * articulations
        + 0.06 * accidentals
        + 0.04 * dots
        + 0.04 * open_noteheads,
    )
    return visual + semantic + list(onset_profile) + list(pitch_profile) + [
        abs(anchor - evidence.notehead_proxy),
        abs(anchor - evidence.onset_proxy),
        abs(pitched - evidence.stem_proxy),
        abs(beam - evidence.beam_proxy),
        abs(directions - direction_ink),
        abs(rests - component_proxy * 0.35),
        abs(compact_semantic - evidence.compact_mark_proxy),
        abs(accidentals - evidence.accidental_proxy),
        abs(open_noteheads - evidence.open_notehead_proxy),
        onset_l1,
        onset_transport,
        pitch_l1,
        pitch_transport,
        abs(semantic_complexity - ink_complexity),
    ]


def compatibility_vector(evidence: VisualMeasureEvidence, measure: MeasureIR) -> list[float]:
    base = v3_compatibility_vector(evidence, measure)
    semantic_grids = semantic_event_grids(measure)
    event_l1, event_cosine, event_miss = _grid_pair_metrics(
        evidence.event_ink_grid, semantic_grids["event_ink_grid"]
    )
    note_l1, _note_cosine, note_miss = _grid_pair_metrics(
        evidence.pitched_notehead_grid, semantic_grids["pitched_notehead_grid"]
    )
    beam_l1, _beam_cosine, beam_miss = _grid_pair_metrics(
        evidence.beam_grid, semantic_grids["beam_grid"]
    )
    compact_l1, _compact_cosine, compact_miss = _grid_pair_metrics(
        evidence.compact_mark_grid, semantic_grids["compact_mark_grid"]
    )
    accidental_l1, _accidental_cosine, accidental_miss = _grid_pair_metrics(
        evidence.accidental_grid, semantic_grids["accidental_grid"]
    )
    open_l1, _open_cosine, open_miss = _grid_pair_metrics(
        evidence.open_notehead_grid, semantic_grids["open_notehead_grid"]
    )
    v4 = base + [
        event_l1,
        event_cosine,
        event_miss,
        _grid_pitch_transport_gap(evidence.event_ink_grid, semantic_grids["event_ink_grid"]),
        note_l1,
        note_miss,
        _grid_pitch_transport_gap(
            evidence.pitched_notehead_grid, semantic_grids["pitched_notehead_grid"]
        ),
        beam_l1,
        beam_miss,
        compact_l1,
        compact_miss,
        accidental_l1,
        accidental_miss,
        open_l1,
        open_miss,
        _attachment_gap(
            evidence.beam_grid,
            evidence.pitched_notehead_grid,
            semantic_grids["beam_grid"],
            semantic_grids["pitched_notehead_grid"],
        ),
        _attachment_gap(
            evidence.accidental_grid,
            evidence.pitched_notehead_grid,
            semantic_grids["accidental_grid"],
            semantic_grids["pitched_notehead_grid"],
        ),
        _attachment_gap(
            evidence.open_notehead_grid,
            evidence.pitched_notehead_grid,
            semantic_grids["open_notehead_grid"],
            semantic_grids["pitched_notehead_grid"],
        ),
    ]
    return v4


def v4_compatibility_vector(evidence: VisualMeasureEvidence, measure: MeasureIR) -> list[float]:
    """Return the exact feature schema used by visual calibrator v4."""
    return compatibility_vector(evidence, measure)[: len(V4_FEATURE_NAMES)]


def _transaction_source_grid(
    source_values: tuple[float, ...],
    semantic_values: tuple[tuple[float, ...], ...],
) -> tuple[float, ...]:
    """Restrict source noteheads to the onset neighbourhood of a pitch transaction."""
    source = _grid_array(source_values)
    columns: set[int] = set()
    for values in semantic_values:
        semantic = _grid_array(values)
        for column in np.flatnonzero(np.sum(semantic, axis=0) >= 0.03):
            for offset in (-2, -1, 0, 1, 2):
                candidate = int(column) + offset
                if 0 <= candidate < EVENT_X_BINS:
                    columns.add(candidate)
    if not columns:
        return tuple(0.0 for _ in range(EVENT_GRID_SIZE))
    mask = np.zeros_like(source)
    ordered = sorted(columns)
    mask[:, ordered] = source[:, ordered]
    total = float(np.sum(mask))
    if total <= 0.0 or not math.isfinite(total):
        return tuple(0.0 for _ in range(EVENT_GRID_SIZE))
    normalised = np.clip(3.0 * mask / total, 0.0, 3.0)
    return tuple(float(value) for value in normalised.ravel())


def pitch_transaction_gap_pair(
    evidence: VisualMeasureEvidence | None,
    template: MeasureIR,
    proposal: MeasureIR,
    changed_event_indices: Sequence[int],
    *,
    strict: bool = False,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return source-crop gaps for only the events changed by one pitch patch.

    Template and proposal share one masked source grid, so the comparison is paired and
    cannot gain support from unrelated notes elsewhere in the measure.  The function is
    deterministic and is used unchanged by CPU training and production inference.
    """
    if evidence is None:
        neutral = (0.5,) * len(PITCH_LOCAL_FEATURE_NAMES)
        return neutral, neutral
    selected = {int(index) for index in changed_event_indices if int(index) >= 0}
    if not selected:
        neutral = (0.5,) * len(PITCH_LOCAL_FEATURE_NAMES)
        return neutral, neutral
    template_grid = _semantic_pitch_grid(template, selected)
    proposal_grid = _semantic_pitch_grid(proposal, selected)
    source_grid = (
        evidence.pitch_guard_strict_notehead_grid
        if strict
        else evidence.pitch_guard_notehead_grid
    )
    if not any(float(value) >= 0.03 for value in source_grid):
        source_grid = evidence.pitched_notehead_grid
    source_grid = _transaction_source_grid(source_grid, (template_grid, proposal_grid))
    return (
        _notehead_local_metrics(source_grid, template_grid),
        _notehead_local_metrics(source_grid, proposal_grid),
    )


def pitch_local_gaps(
    evidence: VisualMeasureEvidence | None,
    measure: MeasureIR,
) -> tuple[float, ...]:
    """Return direct local notehead/pitch gaps for a candidate measure.

    The public helper deliberately exposes only the seven bounded deterministic gaps,
    not the model.  Pitch consensus can therefore compare a proposed repair with its
    template using the same source crop while preserving the rule that learned models
    may veto but never invent a pitch.
    """
    if evidence is None:
        return (0.5,) * len(PITCH_LOCAL_FEATURE_NAMES)
    semantic = semantic_event_grids(measure)
    source_grid = evidence.pitch_guard_notehead_grid
    if not any(float(value) >= 0.03 for value in source_grid):
        source_grid = evidence.pitched_notehead_grid
    return _notehead_local_metrics(
        source_grid,
        semantic["pitched_notehead_grid"],
    )


def legacy_compatibility_vector(evidence: VisualMeasureEvidence, measure: MeasureIR) -> list[float]:
    """Return the exact feature schema used by visual calibrator v2.

    The function exists only for frozen same-test comparison and compatibility audits.
    Production inference uses :func:`compatibility_vector` and the current feature
    schema.  Keeping the legacy transform explicit avoids silently comparing two
    models on different inputs after the visual evidence structure evolves.
    """
    visual = [float(getattr(evidence, name)) for name in LEGACY_VISUAL_FEATURE_NAMES]
    current_semantic = dict(zip(SEMANTIC_FEATURE_NAMES, semantic_features(measure), strict=True))
    semantic = [float(current_semantic[name]) for name in LEGACY_SEMANTIC_FEATURE_NAMES]
    anchor, pitched, rests, _chords, beam, directions, articulations, accidentals, _grace = semantic
    ink_complexity = min(3.0, evidence.nonstaff_ink_density * 16.0)
    direction_ink = min(3.0, (evidence.above_ink_density + evidence.below_ink_density) * 14.0)
    component_proxy = min(3.0, evidence.component_density)
    semantic_complexity = min(
        3.0,
        0.45 * anchor + 0.25 * beam + 0.15 * directions + 0.08 * articulations + 0.07 * accidentals,
    )
    return visual + semantic + [
        abs(anchor - evidence.notehead_proxy),
        abs(anchor - evidence.onset_proxy),
        abs(pitched - evidence.stem_proxy),
        abs(beam - evidence.beam_proxy),
        abs(directions - direction_ink),
        abs(rests - component_proxy * 0.35),
        abs(semantic_complexity - ink_complexity),
    ]


class VisualMeasureCalibrator:
    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).resolve().parent / "resources" / "visual_measure_calibrator.json"
        loaded = load_verified_json(model_path, "visual_measure_compatibility")
        payload = loaded.payload
        self.model_verified = loaded.verified
        self.model_status = loaded.status
        self.model_version = str(payload.get("model_version", "disabled"))
        self.model_type = str(payload.get("model_type", "logistic"))
        self.intercept = float(payload.get("intercept", 0.0))
        self.coefficients = tuple(float(value) for value in payload.get("coefficients", []))
        self.means = tuple(float(value) for value in payload.get("means", []))
        self.scales = tuple(max(float(value), 1e-9) for value in payload.get("scales", []))
        self.boosting = VerifiedGradientBoostingModel.load(
            model_path,
            "visual_measure_compatibility",
            FEATURE_NAMES,
            loaded=loaded,
        )
        self.forest = VerifiedRandomForestModel.load(
            model_path,
            "visual_measure_compatibility",
            FEATURE_NAMES,
            loaded=loaded,
        )
        names_match = tuple(payload.get("feature_names", ())) == FEATURE_NAMES
        logistic_valid = (
            len(self.coefficients) == len(FEATURE_NAMES)
            and len(self.means) == len(FEATURE_NAMES)
            and len(self.scales) == len(FEATURE_NAMES)
        )
        boosting_valid = self.boosting.enabled
        self.enabled = names_match and (
            (self.model_type == "logistic" and logistic_valid)
            or (self.model_type == "gradient_boosting" and boosting_valid)
            or (self.model_type == "random_forest" and self.forest.enabled)
        )

    def predict_probability(self, evidence: VisualMeasureEvidence | None, measure: MeasureIR) -> float:
        if not self.enabled or evidence is None:
            return 0.5
        values = compatibility_vector(evidence, measure)
        if self.model_type == "gradient_boosting":
            return self.boosting.predict(values)
        if self.model_type == "random_forest":
            return self.forest.predict(values)
        standardized = [
            (value - mean) / scale
            for value, mean, scale in zip(values, self.means, self.scales, strict=True)
        ]
        score = self.intercept + sum(
            coefficient * value
            for coefficient, value in zip(self.coefficients, standardized, strict=True)
        )
        return max(0.0, min(1.0, stable_sigmoid(score)))

    def calibrate(self, evidence: VisualMeasureEvidence | None, measure: MeasureIR) -> VisualCompatibilityResult:
        probability = self.predict_probability(evidence, measure)
        # Deliberately narrow influence.  Event-local evidence can reject coarse
        # placement mismatches, but it cannot create semantic support, infer an exact
        # symbol on its own, or overrule an independent-family majority.
        weight = 0.88 + 0.24 * probability
        return VisualCompatibilityResult(round(probability, 6), round(weight, 6), self.model_version)


def map_evidence_to_measure(
    evidence: tuple[VisualMeasureEvidence, ...],
    measure_index: int,
    measure_count: int,
) -> VisualMeasureEvidence | None:
    if not evidence or measure_count <= 0:
        return None
    if measure_count == len(evidence):
        index = measure_index
    elif measure_count <= 1:
        index = 0
    else:
        index = round(measure_index * (len(evidence) - 1) / (measure_count - 1))
    index = max(0, min(len(evidence) - 1, index))
    return evidence[index]


def _compact_evidence_dict(item: VisualMeasureEvidence) -> dict[str, object]:
    payload = item.to_dict()
    encoded: dict[str, str] = {}
    for name in EVENT_GRID_NAMES:
        values = payload.pop(name, ())
        array = np.asarray(values, dtype=np.float64)
        if array.size != EVENT_GRID_SIZE:
            array = np.zeros(EVENT_GRID_SIZE, dtype=np.float64)
        quantised = np.rint(np.clip(array / 3.0, 0.0, 1.0) * 255.0).astype(np.uint8)
        encoded[name] = base64.b64encode(quantised.tobytes()).decode("ascii")
    payload["event_grid_encoding"] = "uint8-base64-v1"
    payload["event_grid_shape"] = [EVENT_Y_BINS, EVENT_X_BINS]
    payload["event_grid_scale"] = 3.0
    payload["event_grids"] = encoded
    return payload


def write_visual_evidence(path: Path, evidence: Iterable[VisualMeasureEvidence]) -> None:
    from .util import atomic_write_json

    items = tuple(evidence)
    atomic_write_json(
        path,
        {
            "format": 5,
            "measure_count": len(items),
            "measures": [_compact_evidence_dict(item) for item in items],
        },
    )
