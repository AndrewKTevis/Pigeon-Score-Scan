from __future__ import annotations

"""CPU page-orientation classification for printed single-staff scans.

Scanners and PDF extraction tools occasionally emit pages rotated by 90, 180 or 270
degrees.  Skew correction cannot repair those cases, and passing them to OMR typically
produces empty or structurally nonsensical MusicXML.  This module uses a compact,
verified multinomial linear model and deterministic safety margins to rotate only pages
whose orientation is strongly supported.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .model_registry import load_verified_json

ORIENTATIONS = (0, 90, 180, 270)
FEATURE_NAMES = (
    "horizontal_line_density",
    "vertical_line_density",
    "horizontal_vertical_log_ratio",
    "row_projection_variation",
    "column_projection_variation",
    "top_ink",
    "bottom_ink",
    "left_ink",
    "right_ink",
    "top_bottom_asymmetry",
    "left_right_asymmetry",
    "ink_centre_x",
    "ink_centre_y",
    "staff_left_ink",
    "staff_right_ink",
    "staff_left_right_asymmetry",
    "top_text_band_ink",
    "bottom_text_band_ink",
    "wide_component_density",
    "tall_component_density",
)


@dataclass(frozen=True)
class OrientationResult:
    degrees: int
    probability: float
    margin: float
    applied: bool
    model_version: str
    model_status: str
    probabilities: dict[int, float]


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / max(denominator, 1e-9)


def extract_orientation_features(gray: np.ndarray) -> tuple[float, ...]:
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gray.size == 0:
        return (0.0,) * len(FEATURE_NAMES)
    height, width = gray.shape
    scale = min(1.0, 620.0 / max(height, width))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    height, width = gray.shape
    _threshold, binary_raw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Keep the one-pixel staff lines for orientation evidence. A generic 2x2 opening
    # erases them on high-quality scans and turns a strong horizontal/vertical cue
    # into zero. A lightly cleaned copy is used only for ink/component statistics.
    binary = cv2.morphologyEx(binary_raw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    total = max(binary_raw.size, 1)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, width // 16), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, height // 16)))
    horizontal = cv2.morphologyEx(binary_raw, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary_raw, cv2.MORPH_OPEN, vertical_kernel)
    horizontal_density = float(np.count_nonzero(horizontal)) / total
    vertical_density = float(np.count_nonzero(vertical)) / total
    hv_log = math.log((horizontal_density + 1e-5) / (vertical_density + 1e-5))

    row_profile = np.count_nonzero(binary_raw, axis=1).astype(np.float32) / max(width, 1)
    col_profile = np.count_nonzero(binary_raw, axis=0).astype(np.float32) / max(height, 1)
    row_variation = float(row_profile.std())
    column_variation = float(col_profile.std())

    edge_y = max(1, int(round(height * 0.19)))
    edge_x = max(1, int(round(width * 0.19)))
    top_ink = float(np.count_nonzero(binary[:edge_y])) / max(edge_y * width, 1)
    bottom_ink = float(np.count_nonzero(binary[-edge_y:])) / max(edge_y * width, 1)
    left_ink = float(np.count_nonzero(binary[:, :edge_x])) / max(height * edge_x, 1)
    right_ink = float(np.count_nonzero(binary[:, -edge_x:])) / max(height * edge_x, 1)
    top_bottom_asym = (top_ink - bottom_ink) / max(top_ink + bottom_ink, 1e-6)
    left_right_asym = (left_ink - right_ink) / max(left_ink + right_ink, 1e-6)

    ys, xs = np.nonzero(binary)
    if len(xs):
        centre_x = float(xs.mean()) / max(width - 1, 1)
        centre_y = float(ys.mean()) / max(height - 1, 1)
    else:
        centre_x = centre_y = 0.5

    # Staff-row bands: correct pages usually have clef/key/time-signature ink near the
    # left end of each system. A 180-degree page moves that asymmetry to the right.
    # Subtract long horizontal strokes first so staff lines themselves do not dominate
    # the left/right measurement.
    horizontal_profile = np.count_nonzero(horizontal, axis=1)
    row_threshold = max(8, int(width * 0.20))
    staff_rows = np.where(horizontal_profile >= row_threshold)[0]
    bands: list[tuple[int, int]] = []
    if staff_rows.size:
        start = previous = int(staff_rows[0])
        for raw in staff_rows[1:]:
            value = int(raw)
            if value - previous > max(3, height // 180):
                bands.append((start, previous))
                start = value
            previous = value
        bands.append((start, previous))
    nonstaff = cv2.subtract(binary_raw, horizontal)
    # A narrow system-edge band captures clefs/key signatures while avoiding most
    # note traffic and the terminal barline. Twelve percent was stable across the
    # supplied high-resolution and compact scans.
    staff_edge_x = max(1, int(round(width * 0.12)))
    band_left: list[float] = []
    band_right: list[float] = []
    for start, end in bands:
        centre = (start + end) // 2
        margin = max(5, int(height * 0.025))
        y1, y2 = max(0, centre - margin), min(height, centre + margin + 1)
        if y2 <= y1:
            continue
        band = nonstaff[y1:y2]
        band_left.append(float(np.count_nonzero(band[:, :staff_edge_x])) / max((y2 - y1) * staff_edge_x, 1))
        band_right.append(float(np.count_nonzero(band[:, -staff_edge_x:])) / max((y2 - y1) * staff_edge_x, 1))
    staff_left = float(np.mean(band_left)) if band_left else left_ink
    staff_right = float(np.mean(band_right)) if band_right else right_ink
    staff_asym = (staff_left - staff_right) / max(staff_left + staff_right, 1e-6)

    text_band = max(1, int(round(height * 0.13)))
    top_text = float(np.count_nonzero(binary[:text_band])) / max(text_band * width, 1)
    bottom_text = float(np.count_nonzero(binary[-text_band:])) / max(text_band * width, 1)

    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    wide = 0
    tall = 0
    area_floor = max(3, int(total * 0.000003))
    for index in range(1, component_count):
        _x, _y, comp_w, comp_h, area = (int(value) for value in stats[index])
        if area < area_floor:
            continue
        if comp_w >= max(8, comp_h * 3.0):
            wide += 1
        if comp_h >= max(8, comp_w * 3.0):
            tall += 1
    component_scale = max(math.sqrt(total) / 100.0, 1.0)
    return (
        horizontal_density,
        vertical_density,
        hv_log,
        row_variation,
        column_variation,
        top_ink,
        bottom_ink,
        left_ink,
        right_ink,
        top_bottom_asym,
        left_right_asym,
        centre_x,
        centre_y,
        staff_left,
        staff_right,
        staff_asym,
        top_text,
        bottom_text,
        min(5.0, wide / component_scale),
        min(5.0, tall / component_scale),
    )


class PageOrientationClassifier:
    def __init__(self, model_path: Path | None = None) -> None:
        path = model_path or Path(__file__).resolve().parent / "resources" / "page_orientation_classifier.json"
        loaded = load_verified_json(path, "page_orientation_classification")
        payload = loaded.payload
        self.model_version = str(payload.get("model_version", "disabled"))
        self.model_status = loaded.status
        self.feature_names = tuple(str(value) for value in payload.get("feature_names", ()))
        self.means = np.asarray(payload.get("means", ()), dtype=np.float64)
        self.scales = np.maximum(np.asarray(payload.get("scales", ()), dtype=np.float64), 1e-9)
        self.coefficients = np.asarray(payload.get("coefficients", ()), dtype=np.float64)
        self.intercept = float(payload.get("intercept", 0.0))
        self.enabled = (
            self.feature_names == FEATURE_NAMES
            and self.means.shape == (len(FEATURE_NAMES),)
            and self.scales.shape == (len(FEATURE_NAMES),)
            and self.coefficients.shape == (len(FEATURE_NAMES),)
        )

    @staticmethod
    def _sigmoid(score: float) -> float:
        if score >= 0:
            return 1.0 / (1.0 + math.exp(-min(score, 40.0)))
        value = math.exp(max(score, -40.0))
        return value / (1.0 + value)

    def upright_probability(self, gray: np.ndarray) -> float:
        if not self.enabled:
            return 0.5
        values = np.asarray(extract_orientation_features(gray), dtype=np.float64)
        standard = (values - self.means) / self.scales
        score = self.intercept + float(self.coefficients @ standard)
        return max(0.0, min(1.0, self._sigmoid(score)))

    def predict_probabilities(self, gray: np.ndarray) -> dict[int, float]:
        # Each key is a correction rotation applied to the input.  The classifier scores
        # the resulting page's uprightness, avoiding fragile label conventions for
        # clockwise versus counter-clockwise source rotations.
        raw = {degrees: self.upright_probability(rotate_quadrant(gray, degrees)) for degrees in ORIENTATIONS}
        total = sum(raw.values())
        if total <= 1e-9:
            return {value: 0.25 for value in ORIENTATIONS}
        return {key: value / total for key, value in raw.items()}

    def classify(
        self,
        gray: np.ndarray,
        *,
        probability_floor: float = 0.68,
        margin_floor: float = 0.30,
        allow_180: bool = False,
    ) -> OrientationResult:
        probabilities = self.predict_probabilities(gray)
        ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        degrees, probability = ranked[0]
        margin = probability - ranked[1][1]
        # RC policy is intentionally selective. Quarter-turn scans are corrected only
        # with a clear learned margin. A 180-degree correction remains advisory by
        # default because upside-down discrimination is more sensitive to page style.
        rotation_allowed = degrees in {90, 270} or (degrees == 180 and allow_180)
        applied = (
            self.enabled
            and degrees != 0
            and rotation_allowed
            and probability >= probability_floor
            and margin >= margin_floor
        )
        return OrientationResult(
            degrees=degrees,
            probability=round(probability, 6),
            margin=round(margin, 6),
            applied=applied,
            model_version=self.model_version,
            model_status=self.model_status,
            probabilities={key: round(value, 6) for key, value in probabilities.items()},
        )


def rotate_quadrant(image: np.ndarray, degrees: int) -> np.ndarray:
    degrees %= 360
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image
