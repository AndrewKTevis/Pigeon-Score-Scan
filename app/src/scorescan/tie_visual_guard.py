from __future__ import annotations

"""Veto-only local visual guard for within-measure tie additions.

Independent candidate families and deterministic tie topology remain solely responsible
for proposing notation.  This module answers one bounded question: does the preserved
source corridor contain a printed tie compatible with an already-proposed adjacent
same-pitch event pair?  It never creates semantic support, classifies a generic curve,
or edits MusicXML.  Tie removal and explicit tie/slur ambiguity remain review-only when
source evidence is available because absence and arc type could not be calibrated to the
same zero-false-accept boundary.
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

PATCH_WIDTH = 96
PATCH_HEIGHT = 48
CELL_SIZE = 8
CELL_COLUMNS = PATCH_WIDTH // CELL_SIZE
CELL_ROWS = PATCH_HEIGHT // CELL_SIZE
ORIENTATION_BINS = 6
VERTICAL_MARGIN_PIXELS = 34
ENDPOINT_INSET_PIXELS = 3
MATCHED_ARCHES = (5.0, 9.0, 13.0, 17.0, 21.0)

TIE_VISUAL_FEATURE_NAMES = tuple(
    f"hog_r{row}_c{column}_b{orientation}"
    for row in range(CELL_ROWS)
    for column in range(CELL_COLUMNS)
    for orientation in range(ORIENTATION_BINS)
) + tuple(
    f"density_r{row}_c{column}"
    for row in range(CELL_ROWS)
    for column in range(CELL_COLUMNS)
) + tuple(
    f"curve_{side}_a{arch_index}_{metric}"
    for side in ("up", "down")
    for arch_index in range(len(MATCHED_ARCHES))
    for metric in ("mean", "contrast", "coverage", "continuity")
) + (
    "baseline_mean",
    "baseline_contrast",
    "baseline_coverage",
    "baseline_continuity",
    "curve_best_contrast",
    "curve_best_coverage",
    "curve_best_up",
    "curve_best_down",
    "curve_contrast_advantage",
    "curve_coverage_advantage",
    "curve_continuity_advantage",
    "curve_midpoint_advantage",
    "horizontal_span_ratio",
    "vertical_gap_ratio",
)


def has_tie_source_evidence(evidence: VisualMeasureEvidence | None) -> bool:
    return bool(evidence is not None and evidence.symbol_guard_image)


def _bounded_crop(
    image: np.ndarray,
    start: tuple[float, float],
    stop: tuple[float, float],
) -> tuple[np.ndarray, float, float] | None:
    height, width = image.shape
    x1 = int(round(max(0.0, min(1.0, start[0])) * (width - 1)))
    y1 = int(round(max(0.0, min(1.0, start[1])) * (height - 1)))
    x2 = int(round(max(0.0, min(1.0, stop[0])) * (width - 1)))
    y2 = int(round(max(0.0, min(1.0, stop[1])) * (height - 1)))
    if x2 <= x1 + ENDPOINT_INSET_PIXELS * 2 + 2:
        return None
    left = x1 + ENDPOINT_INSET_PIXELS
    right = x2 - ENDPOINT_INSET_PIXELS
    top = min(y1, y2) - VERTICAL_MARGIN_PIXELS
    bottom = max(y1, y2) + VERTICAL_MARGIN_PIXELS + 1
    if right <= left or bottom <= top:
        return None

    original_height = max(1, bottom - top)
    start_y = (y1 - top) * (PATCH_HEIGHT - 1) / original_height
    stop_y = (y2 - top) * (PATCH_HEIGHT - 1) / original_height
    pad_left = max(0, -left)
    pad_right = max(0, right - width)
    pad_top = max(0, -top)
    pad_bottom = max(0, bottom - height)
    padded = np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )
    left += pad_left
    right += pad_left
    top += pad_top
    bottom += pad_top
    crop = padded[top:bottom, left:right]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 6:
        return None
    patch = cv2.resize(crop, (PATCH_WIDTH, PATCH_HEIGHT), interpolation=cv2.INTER_AREA)
    return patch, float(start_y), float(stop_y)


def _descriptor(patch: np.ndarray) -> list[float]:
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
    return [*features, *density]


def _longest_run(values: np.ndarray) -> float:
    best = current = 0
    for value in values.tolist():
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best / max(len(values), 1)


def _path_metrics(ink: np.ndarray, rows: np.ndarray) -> tuple[float, float, float, float, float]:
    columns = np.arange(PATCH_WIDTH, dtype=np.int64)
    centre = ink[rows, columns]
    near_a = ink[np.clip(rows - 1, 0, PATCH_HEIGHT - 1), columns]
    near_b = ink[np.clip(rows + 1, 0, PATCH_HEIGHT - 1), columns]
    curve = np.maximum(centre, np.maximum(near_a, near_b))
    flank_a = ink[np.clip(rows - 5, 0, PATCH_HEIGHT - 1), columns]
    flank_b = ink[np.clip(rows + 5, 0, PATCH_HEIGHT - 1), columns]
    flank = 0.5 * (flank_a + flank_b)
    # Ignore endpoint neighbourhoods where noteheads and stems dominate the corridor.
    interior = slice(max(2, PATCH_WIDTH // 10), min(PATCH_WIDTH - 2, PATCH_WIDTH * 9 // 10))
    local = curve[interior]
    local_flank = flank[interior]
    mean = float(np.mean(local))
    contrast = float(np.mean(np.maximum(local - local_flank, 0.0)))
    active = local >= 0.20
    coverage = float(np.mean(active))
    continuity = _longest_run(active)
    middle = slice(PATCH_WIDTH * 3 // 8, PATCH_WIDTH * 5 // 8)
    midpoint = float(np.mean(curve[middle]))
    return mean, contrast, coverage, continuity, midpoint


def _matched_curve_features(
    patch: np.ndarray,
    start_y: float,
    stop_y: float,
) -> list[float]:
    ink = patch.astype(np.float64) / 255.0
    xs = np.arange(PATCH_WIDTH, dtype=np.float64)
    t = xs / max(PATCH_WIDTH - 1, 1)
    baseline = (1.0 - t) * float(start_y) + t * float(stop_y)
    baseline_rows = np.clip(np.rint(baseline).astype(np.int64), 0, PATCH_HEIGHT - 1)
    baseline_metrics = _path_metrics(ink, baseline_rows)
    features: list[float] = []
    contrasts: list[float] = []
    coverages: list[float] = []
    continuities: list[float] = []
    midpoints: list[float] = []
    side_best = {-1: 0.0, 1: 0.0}
    for side in (-1, 1):
        for arch in MATCHED_ARCHES:
            path = baseline + side * (4.0 * arch * t * (1.0 - t))
            rows = np.clip(np.rint(path).astype(np.int64), 0, PATCH_HEIGHT - 1)
            mean, contrast, coverage, continuity, midpoint = _path_metrics(ink, rows)
            features.extend(round(value, 8) for value in (mean, contrast, coverage, continuity))
            contrasts.append(contrast)
            coverages.append(coverage)
            continuities.append(continuity)
            midpoints.append(midpoint)
            side_best[side] = max(side_best[side], contrast)
    best_index = int(np.argmax(np.asarray(contrasts))) if contrasts else 0
    best_contrast = contrasts[best_index] if contrasts else 0.0
    best_coverage = coverages[best_index] if coverages else 0.0
    best_continuity = continuities[best_index] if continuities else 0.0
    best_midpoint = midpoints[best_index] if midpoints else 0.0
    features.extend(round(value, 8) for value in baseline_metrics[:4])
    features.extend(
        round(value, 8)
        for value in (
            best_contrast,
            best_coverage,
            side_best[-1],
            side_best[1],
            best_contrast - baseline_metrics[1],
            best_coverage - baseline_metrics[2],
            best_continuity - baseline_metrics[3],
            best_midpoint - baseline_metrics[4],
        )
    )
    return features


def tie_visual_features(
    evidence: VisualMeasureEvidence | None,
    measure: MeasureIR,
    start_index: int,
    stop_index: int,
) -> list[float] | None:
    """Return one fixed descriptor for an adjacent same-pitch tie corridor."""
    if (
        evidence is None
        or start_index < 0
        or stop_index != start_index + 1
        or stop_index >= len(measure.notes)
    ):
        return None
    image = decode_symbol_guard_image(evidence.symbol_guard_image)
    if image is None:
        return None
    start_note = measure.notes[start_index]
    stop_note = measure.notes[stop_index]
    if (
        start_note.rest
        or stop_note.rest
        or start_note.pitch is None
        or stop_note.pitch is None
        or start_note.grace
        or stop_note.grace
        or start_note.pitch.stable_tuple() != stop_note.pitch.stable_tuple()
    ):
        return None
    start = event_position(measure, start_note)
    stop = event_position(measure, stop_note)
    extracted = _bounded_crop(image, start, stop)
    if extracted is None:
        return None
    patch, start_y, stop_y = extracted
    descriptor = _descriptor(patch)
    descriptor.extend(_matched_curve_features(patch, start_y, stop_y))
    descriptor.extend(
        (
            round(max(0.0, min(1.0, stop[0] - start[0])), 8),
            round(max(0.0, min(1.0, abs(stop[1] - start[1]))), 8),
        )
    )
    if len(descriptor) != len(TIE_VISUAL_FEATURE_NAMES):
        raise AssertionError("tie visual feature schema mismatch")
    return descriptor


def tie_edges(measure: MeasureIR) -> frozenset[tuple[int, int]] | None:
    edges: set[tuple[int, int]] = set()
    for index, note in enumerate(measure.notes):
        state = {str(value).strip().casefold() for value in note.ties if str(value).strip()}
        if state - {"start", "stop"}:
            return None
        if "start" in state:
            if index + 1 >= len(measure.notes):
                return None
            following = {
                str(value).strip().casefold()
                for value in measure.notes[index + 1].ties
                if str(value).strip()
            }
            if "stop" not in following:
                return None
            edges.add((index, index + 1))
        if "stop" in state and (index == 0 or (index - 1, index) not in edges):
            return None
    return frozenset(edges)


def slur_arcs(measure: MeasureIR) -> frozenset[tuple[int, int]] | None:
    endpoints: dict[str, dict[str, list[int]]] = {}
    for index, note in enumerate(measure.notes):
        for kind, number in note.slurs:
            normalized_kind = str(kind).strip().casefold()
            normalized_number = str(number).strip() or "1"
            if normalized_kind not in {"start", "stop"}:
                return None
            endpoints.setdefault(normalized_number, {"start": [], "stop": []})[
                normalized_kind
            ].append(index)
    arcs: set[tuple[int, int]] = set()
    for endpoint in endpoints.values():
        if len(endpoint["start"]) != 1 or len(endpoint["stop"]) != 1:
            return None
        start, stop = endpoint["start"][0], endpoint["stop"][0]
        if start >= stop:
            return None
        arcs.add((start, stop))
    return frozenset(arcs)


@dataclass(frozen=True)
class TieVisualCalibration:
    probability: float
    confidence: float
    threshold: float
    accepted: bool
    available: bool
    model_version: str


@dataclass(frozen=True)
class TieVisualAudit:
    applicable: bool
    changed_tie_count: int
    probability: float
    threshold: float
    accepted: bool
    reason: str
    model_version: str


class TieVisualGuard:
    """Verified local tie-presence model; removal is deliberately review-only."""

    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).with_name("resources") / "tie_visual_guard.json"
        loaded = load_verified_json(model_path, "tie_visual_guard")
        payload = loaded.payload
        self.model = VerifiedRandomForestModel.load(
            model_path,
            "tie_visual_guard",
            TIE_VISUAL_FEATURE_NAMES,
            loaded=loaded,
        )
        try:
            threshold = float(payload.get("present_threshold", 1.0))
        except (TypeError, ValueError, OverflowError):
            threshold = 1.0
        floor = float(DEFAULT_POLICY.tie_visual_guard_probability_floor)
        self.threshold = max(floor, max(0.0, min(1.0, threshold)))
        self.model_verified = self.model.verified and loaded.verified
        self.model_status = self.model.status if self.model.enabled else loaded.status
        self.model_version = self.model.model_version
        self.enabled = self.model.enabled

    def calibrate(
        self,
        evidence: VisualMeasureEvidence | None,
        measure: MeasureIR,
        start_index: int,
        stop_index: int,
    ) -> TieVisualCalibration:
        if not self.enabled or not self.model_verified:
            probability = 0.5
            available = False
        else:
            features = tie_visual_features(evidence, measure, start_index, stop_index)
            available = features is not None
            probability = self.model.predict(features, neutral=0.5) if features is not None else 0.5
        return TieVisualCalibration(
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
    ) -> TieVisualAudit:
        before = tie_edges(base)
        after = tie_edges(patched)
        if before is None or after is None:
            return TieVisualAudit(
                True, 0, 0.5, self.threshold, False, "invalid_tie_topology", self.model_version
            )
        changed = tuple(sorted(before ^ after))
        if not changed:
            return TieVisualAudit(
                False, 0, 0.5, self.threshold, True, "not_applicable", self.model_version
            )
        if not has_tie_source_evidence(evidence):
            return TieVisualAudit(
                False,
                len(changed),
                0.5,
                self.threshold,
                True,
                "source_evidence_unavailable",
                self.model_version,
            )
        removed = before - after
        if removed:
            return TieVisualAudit(
                True,
                len(changed),
                0.5,
                self.threshold,
                False,
                "tie_removal_requires_review",
                self.model_version,
            )
        source_slurs = slur_arcs(base)
        if source_slurs is None:
            return TieVisualAudit(
                True,
                len(changed),
                0.5,
                self.threshold,
                False,
                "invalid_slur_topology",
                self.model_version,
            )
        added = tuple(sorted(after - before))
        if any(edge in source_slurs for edge in added):
            return TieVisualAudit(
                True,
                len(changed),
                0.5,
                self.threshold,
                False,
                "tie_slur_type_ambiguous",
                self.model_version,
            )
        calibrations = [
            self.calibrate(evidence, patched, start, stop) for start, stop in added
        ]
        minimum = min((item.confidence for item in calibrations), default=0.5)
        if not all(item.available for item in calibrations):
            reason = "visual_evidence_invalid_or_model_unavailable"
            accepted = False
        elif not all(item.accepted for item in calibrations):
            reason = "visual_tie_conflict"
            accepted = False
        else:
            reason = "visual_tie_confirmed"
            accepted = True
        return TieVisualAudit(
            True,
            len(changed),
            round(minimum, 6),
            round(self.threshold, 6),
            accepted,
            reason,
            self.model_version,
        )
