from __future__ import annotations

"""Veto-only local visual confirmation for simple accent additions.

Independent OMR families and the existing articulation calibrator remain solely
responsible for proposing MusicXML.  This module answers one bounded question: does a
small preserved source crop contain the requested printed accent adjacent to an
otherwise unarticulated pitched event?  It cannot create support, remove a mark, choose
between articulation classes, or edit XML.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .local_symbol_image import decode_symbol_guard_image, event_position
from .model_registry import load_verified_json
from .policy import DEFAULT_POLICY
from .score_ir import MeasureIR
from .tree_model import VerifiedRandomForestModel
from .visual_evidence import VisualMeasureEvidence

PATCH_WIDTH = 40
PATCH_HEIGHT = 32
Y_OFFSET_PIXELS = 17

_SIDE_SCALARS = (
    "row_max_density",
    "row_mean_density",
    "row_max_run",
    "column_max_density",
    "column_mean_density",
    "column_max_run",
    "roi_density",
    "left_diagonal_coverage",
    "right_diagonal_coverage",
    "paired_diagonal_min_coverage",
    "paired_diagonal_mean_coverage",
    "right_wedge_upper_coverage",
    "right_wedge_lower_coverage",
    "right_wedge_paired_coverage",
    "left_wedge_upper_coverage",
    "left_wedge_lower_coverage",
    "left_wedge_paired_coverage",
    "horizontal_centre_coverage",
    "horizontal_centre_run",
    "near_component_count",
    "nearest_component_distance",
    "nearest_component_area",
    "nearest_component_width",
    "nearest_component_height",
    "nearest_component_fill",
    *(f"orientation_bin_{index}" for index in range(8)),
    "diagonal_orientation_balance",
    "diagonal_orientation_mass",
)
_SIDE_FEATURE_NAMES = (
    *_SIDE_SCALARS,
    *(f"row_density_{index}" for index in range(19)),
    *(f"column_density_{index}" for index in range(29)),
)
ACCENT_VISUAL_FEATURE_NAMES = tuple(
    f"{prefix}_{name}"
    for prefix in ("above", "below", "maximum", "minimum")
    for name in _SIDE_FEATURE_NAMES
)


def _maximum_run(values: np.ndarray) -> int:
    best = 0
    current = 0
    for value in values.tolist():
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _crop(image: np.ndarray, centre_x: int, centre_y: int) -> np.ndarray:
    centre_x = max(0, min(image.shape[1] - 1, int(centre_x)))
    centre_y = max(0, min(image.shape[0] - 1, int(centre_y)))
    half_width = PATCH_WIDTH // 2
    half_height = PATCH_HEIGHT // 2
    padded = np.pad(
        image,
        ((half_height, half_height), (half_width, half_width)),
        mode="constant",
        constant_values=0,
    )
    return padded[
        centre_y : centre_y + PATCH_HEIGHT,
        centre_x : centre_x + PATCH_WIDTH,
    ]


def _line_coverage(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> float:
    hits: list[bool] = []
    for raw_x, raw_y in zip(xs, ys, strict=True):
        x = int(round(float(raw_x)))
        y = int(round(float(raw_y)))
        local = mask[
            max(0, y - 1) : min(mask.shape[0], y + 2),
            max(0, x - 1) : min(mask.shape[1], x + 2),
        ]
        hits.append(bool(np.any(local)))
    return float(np.mean(hits)) if hits else 0.0


def _side_descriptor(patch: np.ndarray) -> np.ndarray:
    # Remove staff-length horizontal fragments and long stems.  A normal accent is much
    # shorter than either kernel, so its paired diagonal strokes remain.
    horizontal = cv2.morphologyEx(
        patch,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (29, 1)),
    )
    vertical = cv2.morphologyEx(
        patch,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 23)),
    )
    cleaned = cv2.subtract(patch, cv2.max(horizontal, vertical))
    _threshold, binary = cv2.threshold(cleaned, 48, 255, cv2.THRESH_BINARY)
    mask = binary > 0

    centre_y = PATCH_HEIGHT // 2
    centre_x = PATCH_WIDTH // 2
    roi = mask[7:26, 6:35]
    row_density = np.count_nonzero(roi, axis=1).astype(np.float64) / 29.0
    column_density = np.count_nonzero(roi, axis=0).astype(np.float64) / 19.0

    values: list[float] = [
        float(np.max(row_density, initial=0.0)),
        float(np.mean(row_density)),
        max((_maximum_run(row) for row in roi), default=0) / 29.0,
        float(np.max(column_density, initial=0.0)),
        float(np.mean(column_density)),
        max((_maximum_run(column) for column in roi.T), default=0) / 19.0,
        float(np.mean(roi)),
    ]

    xs_left = np.arange(centre_x - 10, centre_x + 1)
    left_scores: list[float] = []
    right_scores: list[float] = []
    for apex_y in range(centre_y - 2, centre_y + 4):
        left_scores.append(
            _line_coverage(
                mask,
                xs_left,
                np.linspace(apex_y - 3, apex_y, len(xs_left)),
            )
        )
        xs_right = np.arange(centre_x, centre_x + 11)
        right_scores.append(
            _line_coverage(
                mask,
                xs_right,
                np.linspace(apex_y, apex_y - 3, len(xs_right)),
            )
        )
    vertical_scores = [
        max(left_scores, default=0.0),
        max(right_scores, default=0.0),
        max((min(left, right) for left, right in zip(left_scores, right_scores, strict=True)), default=0.0),
        max(((left + right) / 2.0 for left, right in zip(left_scores, right_scores, strict=True)), default=0.0),
    ]
    right_upper: list[float] = []
    right_lower: list[float] = []
    left_upper: list[float] = []
    left_lower: list[float] = []
    for shift in range(-2, 3):
        xs = np.arange(centre_x - 9, centre_x + 7)
        right_upper.append(
            _line_coverage(mask, xs, np.linspace(centre_y - 4 + shift, centre_y + shift, len(xs)))
        )
        right_lower.append(
            _line_coverage(mask, xs, np.linspace(centre_y + 4 + shift, centre_y + shift, len(xs)))
        )
        xs = np.arange(centre_x - 6, centre_x + 10)
        left_upper.append(
            _line_coverage(mask, xs, np.linspace(centre_y + shift, centre_y - 4 + shift, len(xs)))
        )
        left_lower.append(
            _line_coverage(mask, xs, np.linspace(centre_y + shift, centre_y + 4 + shift, len(xs)))
        )
    wedge_scores = [
        max(right_upper, default=0.0),
        max(right_lower, default=0.0),
        max((min(a, b) for a, b in zip(right_upper, right_lower, strict=True)), default=0.0),
        max(left_upper, default=0.0),
        max(left_lower, default=0.0),
        max((min(a, b) for a, b in zip(left_upper, left_lower, strict=True)), default=0.0),
    ]
    values.extend(
        [
            *vertical_scores,
            *wedge_scores,
            max(
                (
                    float(np.mean(mask[row, centre_x - 10 : centre_x + 11]))
                    for row in range(centre_y - 3, centre_y + 4)
                ),
                default=0.0,
            ),
            max(
                (
                    _maximum_run(mask[row, centre_x - 12 : centre_x + 13]) / 25.0
                    for row in range(centre_y - 4, centre_y + 5)
                ),
                default=0.0,
            ),
        ]
    )

    component_count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        8,
    )
    components: list[tuple[float, int, int, int, float]] = []
    for index in range(1, component_count):
        _x, _y, width, height, area = stats[index]
        component_x, component_y = centroids[index]
        distance = float(
            np.hypot(component_x - centre_x, component_y - centre_y)
        )
        if distance < 14.0 and 2 <= area <= 160 and width <= 24 and height <= 16:
            components.append(
                (
                    distance,
                    int(area),
                    int(width),
                    int(height),
                    float(area) / max(int(width) * int(height), 1),
                )
            )
    components.sort(key=lambda item: (item[0], -item[1]))
    nearest = components[0] if components else (99.0, 0, 0, 0, 0.0)
    values.extend(
        [
            min(len(components), 5) / 5.0,
            min(nearest[0], 14.0) / 14.0,
            nearest[1] / 160.0,
            nearest[2] / 24.0,
            nearest[3] / 16.0,
            nearest[4],
        ]
    )
    integer = cleaned.astype(np.int32)
    bordered = np.pad(integer, ((1, 1), (1, 1)), mode="reflect")
    gx = (
        -bordered[:-2, :-2]
        + bordered[:-2, 2:]
        - 2 * bordered[1:-1, :-2]
        + 2 * bordered[1:-1, 2:]
        - bordered[2:, :-2]
        + bordered[2:, 2:]
    )
    gy = (
        -bordered[:-2, :-2]
        - 2 * bordered[:-2, 1:-1]
        - bordered[:-2, 2:]
        + bordered[2:, :-2]
        + 2 * bordered[2:, 1:-1]
        + bordered[2:, 2:]
    )
    magnitude = np.hypot(gx.astype(np.float64), gy.astype(np.float64))[7:26, 6:35]
    orientation = np.mod(np.arctan2(gy, gx), np.pi)[7:26, 6:35]
    orientation_bins = np.minimum(
        (orientation / (np.pi / 8.0)).astype(np.int64), 7
    )
    histogram = np.bincount(
        orientation_bins.ravel(), weights=magnitude.ravel(), minlength=8
    ).astype(np.float64)
    total = float(np.sum(histogram))
    if total > 1e-12:
        histogram /= total
    left_diagonal = float(np.sum(histogram[1:4]))
    right_diagonal = float(np.sum(histogram[5:8]))
    diagonal_maximum = max(left_diagonal, right_diagonal)
    values.extend(float(value) for value in histogram)
    values.extend(
        [
            min(left_diagonal, right_diagonal) / diagonal_maximum
            if diagonal_maximum > 1e-12
            else 0.0,
            left_diagonal + right_diagonal,
        ]
    )
    values.extend(float(value) for value in row_density)
    values.extend(float(value) for value in column_density)
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (len(_SIDE_FEATURE_NAMES),):
        raise AssertionError("accent visual side feature schema mismatch")
    return result


def accent_visual_features(
    evidence: VisualMeasureEvidence | None,
    measure: MeasureIR,
    event_index: int,
) -> list[float] | None:
    if evidence is None or event_index < 0 or event_index >= len(measure.notes):
        return None
    image = decode_symbol_guard_image(evidence.symbol_guard_image)
    if image is None:
        return None
    note = measure.notes[event_index]
    if note.pitch is None or note.rest or note.chord or note.grace:
        return None
    x_ratio, y_ratio = event_position(measure, note)
    centre_x = int(round(x_ratio * (image.shape[1] - 1)))
    centre_y = int(round(y_ratio * (image.shape[0] - 1)))
    above = _side_descriptor(_crop(image, centre_x, centre_y - Y_OFFSET_PIXELS))
    below = _side_descriptor(_crop(image, centre_x, centre_y + Y_OFFSET_PIXELS))
    vector = np.concatenate(
        (above, below, np.maximum(above, below), np.minimum(above, below))
    )
    if vector.shape != (len(ACCENT_VISUAL_FEATURE_NAMES),):
        raise AssertionError("accent visual feature schema mismatch")
    return [round(float(value), 8) for value in vector]


@dataclass(frozen=True)
class AccentVisualCalibration:
    probability: float
    confidence: float
    threshold: float
    accepted: bool
    available: bool
    model_version: str


@dataclass(frozen=True)
class AccentVisualAudit:
    applicable: bool
    changed_accent_count: int
    probability: float
    threshold: float
    accepted: bool
    reason: str
    model_version: str


class AccentVisualGuard:
    """Verified local accent-presence model; removal and substitution are review-only."""

    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).with_name("resources") / "accent_visual_guard.json"
        loaded = load_verified_json(model_path, "accent_visual_guard")
        payload = loaded.payload
        self.model = VerifiedRandomForestModel.load(
            model_path,
            "accent_visual_guard",
            ACCENT_VISUAL_FEATURE_NAMES,
            loaded=loaded,
        )
        try:
            threshold = float(payload.get("present_threshold", 1.0))
        except (TypeError, ValueError, OverflowError):
            threshold = 1.0
        floor = float(DEFAULT_POLICY.accent_visual_guard_probability_floor)
        self.threshold = max(floor, max(0.0, min(1.0, threshold)))
        self.model_verified = self.model.verified and loaded.verified
        self.model_status = self.model.status if self.model.enabled else loaded.status
        self.model_version = self.model.model_version
        self.enabled = self.model.enabled

    def calibrate(
        self,
        evidence: VisualMeasureEvidence | None,
        measure: MeasureIR,
        event_index: int,
    ) -> AccentVisualCalibration:
        if not self.enabled or not self.model_verified:
            probability = 0.5
            available = False
        else:
            features = accent_visual_features(evidence, measure, event_index)
            available = features is not None
            probability = (
                self.model.predict(features, neutral=0.5)
                if features is not None
                else 0.5
            )
        return AccentVisualCalibration(
            probability=round(probability, 6),
            confidence=round(probability, 6),
            threshold=round(self.threshold, 6),
            accepted=bool(available and probability >= self.threshold),
            available=available,
            model_version=self.model_version,
        )

    def audit_transaction(
        self,
        evidence: VisualMeasureEvidence | None,
        base: MeasureIR,
        patched: MeasureIR,
    ) -> AccentVisualAudit:
        if len(base.notes) != len(patched.notes):
            return AccentVisualAudit(
                True, 0, 0.5, self.threshold, False, "event_lattice_changed", self.model_version
            )
        changed = tuple(
            index
            for index, (before, after) in enumerate(zip(base.notes, patched.notes, strict=True))
            if before.articulations != after.articulations
        )
        if not changed:
            return AccentVisualAudit(
                False, 0, 0.5, self.threshold, True, "not_applicable", self.model_version
            )
        accent_additions = tuple(
            index
            for index in changed
            if "accent" not in base.notes[index].articulations
            and "accent" in patched.notes[index].articulations
        )
        if not accent_additions:
            return AccentVisualAudit(
                False, 0, 0.5, self.threshold, True, "not_applicable", self.model_version
            )
        if evidence is None or not str(evidence.symbol_guard_image or "").strip():
            return AccentVisualAudit(
                False,
                len(accent_additions),
                0.5,
                self.threshold,
                True,
                "source_evidence_unavailable",
                self.model_version,
            )
        for index in changed:
            before = tuple(base.notes[index].articulations)
            after = tuple(patched.notes[index].articulations)
            if index not in accent_additions or before or after != ("accent",):
                return AccentVisualAudit(
                    True,
                    len(accent_additions),
                    0.5,
                    self.threshold,
                    False,
                    "mixed_or_nonempty_articulation_transaction_requires_review",
                    self.model_version,
                )
        calibrations = [
            self.calibrate(evidence, patched, index) for index in accent_additions
        ]
        minimum = min((item.confidence for item in calibrations), default=0.5)
        if not all(item.available for item in calibrations):
            reason = "visual_evidence_invalid_or_model_unavailable"
            accepted = False
        elif not all(item.accepted for item in calibrations):
            reason = "visual_accent_conflict"
            accepted = False
        else:
            reason = "visual_accent_confirmed"
            accepted = True
        return AccentVisualAudit(
            True,
            len(accent_additions),
            round(minimum, 6),
            round(self.threshold, 6),
            accepted,
            reason,
            self.model_version,
        )
