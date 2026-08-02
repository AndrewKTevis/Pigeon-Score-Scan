from __future__ import annotations

"""CPU-friendly visual classifier for staff-spanning barline candidates.

The layout detector deliberately generates permissive vertical-stroke proposals.  This
module extracts a compact, scale-normalised feature vector and applies a verified
probability-forest model.  The classifier never creates musical semantics; it only helps
distinguish measure boundaries from note stems, beams, text strokes, and scan noise.
Deterministic geometry remains authoritative when the model is missing or invalid.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .policy import DEFAULT_POLICY
from .tree_model import VerifiedRandomForestModel

FEATURE_NAMES = (
    "band_width_scaled",
    "row_coverage",
    "longest_vertical_run",
    "top_endpoint_ink",
    "bottom_endpoint_ink",
    "staff_line_intersection_ratio",
    "column_peak_ratio",
    "central_density",
    "side_density",
    "side_asymmetry",
    "above_extension",
    "below_extension",
    "mid_horizontal_attachment",
    "local_vertical_dominance",
    "interline_mean_coverage",
    "interline_min_coverage",
)


@dataclass(frozen=True)
class BarlineFeatures:
    band_width_scaled: float
    row_coverage: float
    longest_vertical_run: float
    top_endpoint_ink: float
    bottom_endpoint_ink: float
    staff_line_intersection_ratio: float
    column_peak_ratio: float
    central_density: float
    side_density: float
    side_asymmetry: float
    above_extension: float
    below_extension: float
    mid_horizontal_attachment: float
    local_vertical_dominance: float
    interline_mean_coverage: float
    interline_min_coverage: float

    def vector(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, name)) for name in FEATURE_NAMES)


@dataclass(frozen=True)
class BarlineClassification:
    probability: float
    accepted: bool
    model_version: str
    model_status: str
    features: BarlineFeatures


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _longest_true_run(mask: np.ndarray) -> int:
    values = np.asarray(mask, dtype=np.uint8).ravel()
    if values.size == 0 or not np.any(values):
        return 0
    padded = np.concatenate(([0], values, [0]))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return int(np.max(ends - starts, initial=0))


def _interline_coverages(
    binary: np.ndarray,
    *,
    core_left: int,
    core_right: int,
    line_y: tuple[float, ...] | list[float],
    spacing: float,
) -> tuple[float, float]:
    """Measure continuity between adjacent staff lines, excluding line ink itself."""
    height = binary.shape[0]
    coverages: list[float] = []
    margin = max(1, int(round(spacing * 0.18)))
    for upper, lower in zip(line_y, line_y[1:]):
        y1 = max(0, int(round(upper)) + margin)
        y2 = min(height, int(round(lower)) - margin + 1)
        if y2 <= y1:
            continue
        region = binary[y1:y2, core_left:core_right]
        if region.size:
            coverages.append(float(np.mean(np.any(region > 0, axis=1))))
    if not coverages:
        return 0.0, 0.0
    return float(np.mean(coverages)), float(min(coverages))


def extract_barline_features(
    binary: np.ndarray,
    *,
    x_start: int,
    x_end: int,
    line_y: list[float] | tuple[float, ...],
    spacing: float,
) -> BarlineFeatures:
    """Extract scale-normalised features for one vertical proposal.

    ``binary`` contains foreground ink as non-zero pixels. Sampling windows are clamped,
    so malformed edge proposals cannot crash page analysis.
    """

    height, width = binary.shape[:2]
    spacing = max(float(spacing), 1.0)
    if not line_y:
        line_y = (height * 0.35, height * 0.45, height * 0.55, height * 0.65, height * 0.75)
    line_y = tuple(float(value) for value in line_y)
    first_line = int(round(min(line_y)))
    last_line = int(round(max(line_y)))
    staff_top = max(0, first_line - max(1, int(round(spacing * 0.12))))
    staff_bottom = min(height, last_line + max(2, int(round(spacing * 0.18))) + 1)
    extension = max(2, int(round(spacing * 1.35)))
    sample_top = max(0, staff_top - extension)
    sample_bottom = min(height, staff_bottom + extension)

    x_start = max(0, min(width - 1, int(x_start)))
    x_end = max(x_start, min(width - 1, int(x_end)))
    centre = int(round((x_start + x_end) / 2))
    core_half = max(1, int(round(max(x_end - x_start + 1, spacing * 0.18) / 2)))
    core_left = max(0, centre - core_half)
    core_right = min(width, centre + core_half + 1)
    side_width = max(2, int(round(spacing * 0.85)))
    left_side = binary[sample_top:sample_bottom, max(0, core_left - side_width):core_left]
    right_side = binary[sample_top:sample_bottom, core_right:min(width, core_right + side_width)]
    core = binary[sample_top:sample_bottom, core_left:core_right]
    staff_core = binary[staff_top:staff_bottom, core_left:core_right]

    if core.size == 0 or staff_core.size == 0:
        return BarlineFeatures(*(0.0 for _ in FEATURE_NAMES))

    staff_rows = np.any(staff_core > 0, axis=1)
    row_coverage = float(np.mean(staff_rows))
    longest_vertical_run = _longest_true_run(staff_rows) / max(len(staff_rows), 1)
    endpoint = max(1, int(round(spacing * 0.30)))
    top_endpoint_ink = float(np.mean(staff_rows[:endpoint]))
    bottom_endpoint_ink = float(np.mean(staff_rows[-endpoint:]))

    intersection_hits = 0
    line_window = max(1, int(round(spacing * 0.16)))
    for line in line_y:
        local_y1 = max(0, int(round(line)) - line_window)
        local_y2 = min(height, int(round(line)) + line_window + 1)
        if np.any(binary[local_y1:local_y2, core_left:core_right] > 0):
            intersection_hits += 1
    staff_line_intersection_ratio = intersection_hits / max(len(line_y), 1)

    column_counts = np.count_nonzero(staff_core, axis=0)
    column_peak_ratio = float(column_counts.max(initial=0)) / max(staff_core.shape[0], 1)
    central_density = float(np.count_nonzero(staff_core)) / max(staff_core.size, 1)
    left_density = float(np.count_nonzero(left_side)) / max(left_side.size, 1) if left_side.size else 0.0
    right_density = float(np.count_nonzero(right_side)) / max(right_side.size, 1) if right_side.size else 0.0
    side_density = 0.5 * (left_density + right_density)
    side_asymmetry = abs(left_density - right_density) / max(left_density + right_density, 1e-6)

    above = binary[sample_top:staff_top, core_left:core_right]
    below = binary[staff_bottom:sample_bottom, core_left:core_right]
    above_extension = float(np.count_nonzero(above)) / max(above.size, 1) if above.size else 0.0
    below_extension = float(np.count_nonzero(below)) / max(below.size, 1) if below.size else 0.0

    mid_y = int(round((first_line + last_line) / 2))
    mid_half = max(1, int(round(spacing * 0.22)))
    attachment = binary[
        max(0, mid_y - mid_half):min(height, mid_y + mid_half + 1),
        max(0, core_left - side_width):min(width, core_right + side_width),
    ]
    if attachment.size:
        attachment_core = binary[
            max(0, mid_y - mid_half):min(height, mid_y + mid_half + 1),
            core_left:core_right,
        ]
        attachment_side_ink = max(0, np.count_nonzero(attachment) - np.count_nonzero(attachment_core))
        attachment_side_area = max(1, attachment.size - attachment_core.size)
        mid_horizontal_attachment = attachment_side_ink / attachment_side_area
    else:
        mid_horizontal_attachment = 0.0

    local = binary[sample_top:sample_bottom, max(0, centre - side_width):min(width, centre + side_width + 1)]
    if local.size:
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(round(spacing * 2.4)))))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(round(spacing * 2.4))), 1))
        vertical_ink = np.count_nonzero(cv2.morphologyEx(local, cv2.MORPH_OPEN, vertical_kernel))
        horizontal_ink = np.count_nonzero(cv2.morphologyEx(local, cv2.MORPH_OPEN, horizontal_kernel))
        local_vertical_dominance = vertical_ink / max(vertical_ink + horizontal_ink, 1)
    else:
        local_vertical_dominance = 0.0

    interline_mean_coverage, interline_min_coverage = _interline_coverages(
        binary,
        core_left=core_left,
        core_right=core_right,
        line_y=line_y,
        spacing=spacing,
    )

    return BarlineFeatures(
        band_width_scaled=_clip01((x_end - x_start + 1) / max(spacing * 0.9, 1.0)),
        row_coverage=_clip01(row_coverage),
        longest_vertical_run=_clip01(longest_vertical_run),
        top_endpoint_ink=_clip01(top_endpoint_ink),
        bottom_endpoint_ink=_clip01(bottom_endpoint_ink),
        staff_line_intersection_ratio=_clip01(staff_line_intersection_ratio),
        column_peak_ratio=_clip01(column_peak_ratio),
        central_density=_clip01(central_density * 2.8),
        side_density=_clip01(side_density * 3.2),
        side_asymmetry=_clip01(side_asymmetry),
        above_extension=_clip01(above_extension * 3.0),
        below_extension=_clip01(below_extension * 3.0),
        mid_horizontal_attachment=_clip01(mid_horizontal_attachment * 3.0),
        local_vertical_dominance=_clip01(local_vertical_dominance),
        interline_mean_coverage=_clip01(interline_mean_coverage),
        interline_min_coverage=_clip01(interline_min_coverage),
    )


class BarlineClassifier:
    def __init__(self, model_path: Path | None = None) -> None:
        path = model_path or Path(__file__).resolve().parent / "resources" / "barline_classifier.json"
        self.model = VerifiedRandomForestModel.load(path, "barline_classification", FEATURE_NAMES)

    @property
    def enabled(self) -> bool:
        return self.model.enabled

    @property
    def model_version(self) -> str:
        return self.model.model_version

    @property
    def status(self) -> str:
        return self.model.status

    def classify(self, features: BarlineFeatures, *, threshold: float | None = None) -> BarlineClassification:
        probability = self.model.predict(features.vector(), neutral=0.5)
        floor = DEFAULT_POLICY.barline_probability_floor if threshold is None else float(threshold)
        accepted = probability >= floor
        return BarlineClassification(
            probability=round(probability, 6),
            accepted=accepted,
            model_version=self.model.model_version,
            model_status=self.model.status,
            features=features,
        )
