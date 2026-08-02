from __future__ import annotations

"""Veto-only local visual guard for accidental presence changes.

Independent OMR families and deterministic accidental-state validation remain solely
responsible for proposing notation.  This module answers one bounded question: does a
small preserved source crop contain a printed accidental immediately to the left of the
changed notehead?  The verified CPU model cannot distinguish accidental classes, create
support, or edit MusicXML.  Explicit-symbol class substitutions therefore remain
review-only.
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

PATCH_WIDTH = 32
PATCH_HEIGHT = 64
CELL_SIZE = 8
CELL_COLUMNS = PATCH_WIDTH // CELL_SIZE
CELL_ROWS = PATCH_HEIGHT // CELL_SIZE
ORIENTATION_BINS = 6
X_OFFSET_PIXELS = -9

ACCIDENTAL_PRESENCE_FEATURE_NAMES = tuple(
    f"hog_r{row}_c{column}_b{orientation}"
    for row in range(CELL_ROWS)
    for column in range(CELL_COLUMNS)
    for orientation in range(ORIENTATION_BINS)
) + tuple(
    f"density_r{row}_c{column}"
    for row in range(CELL_ROWS)
    for column in range(CELL_COLUMNS)
)


def _crop(image: np.ndarray, x_ratio: float, y_ratio: float) -> np.ndarray:
    centre_x = int(round(max(0.0, min(1.0, float(x_ratio))) * (image.shape[1] - 1)))
    centre_y = int(round(max(0.0, min(1.0, float(y_ratio))) * (image.shape[0] - 1)))
    centre_x += X_OFFSET_PIXELS
    half_w = PATCH_WIDTH // 2
    half_h = PATCH_HEIGHT // 2
    padded = np.pad(image, ((half_h, half_h), (half_w, half_w)), constant_values=0)
    patch = padded[centre_y : centre_y + PATCH_HEIGHT, centre_x : centre_x + PATCH_WIDTH]
    if patch.shape != (PATCH_HEIGHT, PATCH_WIDTH):
        return np.zeros((PATCH_HEIGHT, PATCH_WIDTH), dtype=np.uint8)
    return patch


def _remove_staff_lines(patch: np.ndarray) -> np.ndarray:
    # Full staff-line fragments span most of this narrow crop.  Accidental crossbars
    # are much shorter, so a 25-pixel opening removes only the former.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horizontal = cv2.morphologyEx(patch, cv2.MORPH_OPEN, kernel)
    return cv2.subtract(patch, horizontal)


def accidental_hog_features(
    evidence: VisualMeasureEvidence | None,
    measure: MeasureIR,
    event_index: int,
) -> list[float] | None:
    """Return one fixed 224-value local descriptor under hard image bounds."""
    if evidence is None or event_index < 0 or event_index >= len(measure.notes):
        return None
    image = decode_symbol_guard_image(evidence.symbol_guard_image)
    if image is None:
        return None
    note = measure.notes[event_index]
    if note.pitch is None or note.rest or note.grace:
        return None
    x_ratio, y_ratio = event_position(measure, note)
    return accidental_hog_features_at_position(image, x_ratio, y_ratio)


def accidental_hog_features_at_position(
    image: np.ndarray,
    x_ratio: float,
    y_ratio: float,
) -> list[float]:
    """Extract the deployed descriptor at an explicit registered-image anchor."""
    if image.ndim != 2 or image.size == 0:
        raise ValueError("accidental presence image must be non-empty grayscale")
    patch = _remove_staff_lines(_crop(image, x_ratio, y_ratio))

    values = patch.astype(np.int32)
    bordered = np.pad(values, ((1, 1), (1, 1)), mode="reflect")
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
    magnitude = np.hypot(gx.astype(np.float64), gy.astype(np.float64))
    orientation = np.mod(np.arctan2(gy, gx), np.pi)
    bin_width = np.pi / ORIENTATION_BINS
    bins = np.minimum((orientation / bin_width).astype(np.int64), ORIENTATION_BINS - 1)

    features: list[float] = []
    density: list[float] = []
    for row in range(CELL_ROWS):
        y0 = row * CELL_SIZE
        y1 = y0 + CELL_SIZE
        for column in range(CELL_COLUMNS):
            x0 = column * CELL_SIZE
            x1 = x0 + CELL_SIZE
            local_magnitude = magnitude[y0:y1, x0:x1]
            local_bins = bins[y0:y1, x0:x1]
            histogram = np.bincount(
                local_bins.ravel(),
                weights=local_magnitude.ravel(),
                minlength=ORIENTATION_BINS,
            ).astype(np.float64)
            norm = float(np.linalg.norm(histogram))
            if norm > 1e-12:
                histogram /= norm
            features.extend(round(float(value), 8) for value in histogram)
            density.append(round(float(np.mean(patch[y0:y1, x0:x1])) / 255.0, 8))
    result = [*features, *density]
    if len(result) != len(ACCIDENTAL_PRESENCE_FEATURE_NAMES):
        raise AssertionError("accidental presence feature schema mismatch")
    return result


@dataclass(frozen=True)
class AccidentalPresenceCalibration:
    probability: float
    expected_present: bool
    confidence: float
    threshold: float
    accepted: bool
    model_version: str


class AccidentalPresenceGuard:
    """Verified binary local-symbol guard with asymmetric safe thresholds."""

    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).with_name("resources") / "accidental_presence_guard.json"
        loaded = load_verified_json(model_path, "accidental_presence_guard")
        payload = loaded.payload
        self.model = VerifiedRandomForestModel.load(
            model_path,
            "accidental_presence_guard",
            ACCIDENTAL_PRESENCE_FEATURE_NAMES,
            loaded=loaded,
        )
        try:
            present = float(payload.get("present_threshold", 1.0))
            absent = float(payload.get("absent_threshold", 1.0))
        except (TypeError, ValueError, OverflowError):
            present = absent = 1.0
        floor = float(DEFAULT_POLICY.accidental_presence_guard_probability_floor)
        self.present_threshold = max(floor, max(0.0, min(1.0, present)))
        self.absent_threshold = max(floor, max(0.0, min(1.0, absent)))
        self.model_verified = self.model.verified and loaded.verified
        self.model_status = self.model.status if self.model.enabled else loaded.status
        self.model_version = self.model.model_version
        self.enabled = self.model.enabled

    def _predict_with_availability(
        self,
        evidence: VisualMeasureEvidence | None,
        measure: MeasureIR,
        event_index: int,
    ) -> tuple[float, bool]:
        if not self.enabled or not self.model_verified:
            return 0.5, False
        features = accidental_hog_features(evidence, measure, event_index)
        if features is None:
            return 0.5, False
        return self.model.predict(features, neutral=0.5), True

    def predict_probability(
        self,
        evidence: VisualMeasureEvidence | None,
        measure: MeasureIR,
        event_index: int,
    ) -> float:
        return self._predict_with_availability(evidence, measure, event_index)[0]

    def calibrate(
        self,
        evidence: VisualMeasureEvidence | None,
        measure: MeasureIR,
        event_index: int,
        *,
        expected_present: bool,
    ) -> AccidentalPresenceCalibration:
        probability, available = self._predict_with_availability(
            evidence, measure, event_index
        )
        confidence = probability if expected_present else 1.0 - probability
        threshold = self.present_threshold if expected_present else self.absent_threshold
        accepted = bool(available and confidence >= threshold)
        return AccidentalPresenceCalibration(
            probability=round(probability, 6),
            expected_present=bool(expected_present),
            confidence=round(confidence, 6),
            threshold=round(threshold, 6),
            accepted=accepted,
            model_version=self.model_version,
        )
