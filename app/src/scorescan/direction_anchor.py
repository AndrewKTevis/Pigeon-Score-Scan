from __future__ import annotations

"""Conservative staff-direction role and anchoring evidence.

OCR answers *what* text may say.  This module answers the separate question of
whether a detected text box plausibly belongs to the music timeline rather than page
furniture, a title, a composer credit, a page number, or notation fragments.

The bundled CPU models are advisory.  Hard geometry limits remain authoritative, and
unknown text is never discarded: low-confidence items are kept for review rather than
silently injected into MusicXML.  Exact visual/MusicXML measure counts remain fully
deterministic; the embedded forest can only refine a small count mismatch.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .layout import (
    MeasureAnchor,
    PageLayout,
    ScoreSystemLayout,
    StaffSystem,
    anchor_x_to_measure,
    system_measure_bounds,
)
from .linear_model import StandardizedLogisticModel
from .model_registry import load_verified_json
from .policy import DEFAULT_POLICY
from .tree_model import VerifiedRandomForestModel

FEATURE_NAMES = (
    "ocr_score",
    "lexicon_probability",
    "correction_margin",
    "inverse_edit_ratio",
    "backend_agreement",
    "distance_staff_spaces",
    "box_height_staff_spaces",
    "box_width_staff_spaces",
    "relative_page_y",
    "relative_system_x",
    "system_present",
    "placement_above",
    "placement_below",
    "placement_within",
    "kind_dynamic",
    "kind_metronome",
    "kind_direction",
    "kind_text",
    "alpha_ratio",
    "digit_ratio",
    "punctuation_ratio",
    "token_count",
    "title_like",
    "first_system",
)

ANCHOR_FEATURE_NAMES = (
    "baseline_delta_scaled",
    "relative_x",
    "source_index_scaled",
    "source_offset",
    "source_count_scaled",
    "target_count_scaled",
    "count_gap_scaled",
    "candidate_index_scaled",
    "candidate_distance_scaled",
    "candidate_is_baseline",
    "interval_width_ratio",
    "previous_width_ratio",
    "next_width_ratio",
    "left_boundary_distance",
    "right_boundary_distance",
    "nearest_boundary_distance",
    "kind_dynamic",
    "kind_metronome",
    "kind_direction",
    "kind_text",
    "placement_above",
    "placement_below",
)


_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", re.UNICODE)


@dataclass(frozen=True)
class DirectionAnchorFeatures:
    ocr_score: float
    lexicon_probability: float
    correction_margin: float
    inverse_edit_ratio: float
    backend_agreement: float
    distance_staff_spaces: float
    box_height_staff_spaces: float
    box_width_staff_spaces: float
    relative_page_y: float
    relative_system_x: float
    system_present: float
    placement_above: float
    placement_below: float
    placement_within: float
    kind_dynamic: float
    kind_metronome: float
    kind_direction: float
    kind_text: float
    alpha_ratio: float
    digit_ratio: float
    punctuation_ratio: float
    token_count: float
    title_like: float
    first_system: float

    def vector(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, name)) for name in FEATURE_NAMES)


def _bounds(box: Iterable[Iterable[float]]) -> tuple[float, float, float, float]:
    points = np.asarray(list(box), dtype=np.float64)
    if points.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def extract_direction_anchor_features(
    *,
    text: str,
    kind: str,
    score: float,
    box: Iterable[Iterable[float]],
    backend: str | None,
    correction_probability: float,
    correction_margin: float,
    correction_edit_ratio: float,
    system_index: int | None,
    placement: str | None,
    distance_staff_spaces: float,
    layout: PageLayout,
) -> DirectionAnchorFeatures:
    x1, y1, x2, y2 = _bounds(box)
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    page_width = max(float(layout.width), 1.0)
    page_height = max(float(layout.height), 1.0)
    system = None
    if system_index is not None and 0 <= system_index < len(layout.systems):
        system = layout.systems[system_index]
    spacing = max(float(system.spacing if system is not None else 12.0), 1.0)
    relative_system_x = (
        (center_x - system.left) / max(float(system.right - system.left), 1.0)
        if system is not None
        else center_x / page_width
    )

    characters = [char for char in text if not char.isspace()]
    length = max(len(characters), 1)
    alpha = sum(char.isalpha() for char in characters) / length
    digits = sum(char.isdigit() for char in characters) / length
    punctuation = sum((not char.isalnum()) for char in characters) / length
    tokens = len(_TOKEN_RE.findall(text))
    backend_count = len({item for item in (backend or "").split("+") if item})
    box_height_spaces = max(0.0, (y2 - y1) / spacing)
    box_width_spaces = max(0.0, (x2 - x1) / spacing)
    title_like = float(
        system_index is None
        or box_height_spaces >= 3.8
        or (len(layout.systems) > 0 and center_y < layout.systems[0].line_y[0] - spacing * 12.0)
    )
    return DirectionAnchorFeatures(
        ocr_score=_clip(score),
        lexicon_probability=_clip(correction_probability),
        correction_margin=_clip(correction_margin, -1.0, 1.0),
        inverse_edit_ratio=_clip(1.0 - correction_edit_ratio),
        backend_agreement=_clip((backend_count - 1) / 3.0),
        distance_staff_spaces=_clip(distance_staff_spaces / 14.0),
        box_height_staff_spaces=_clip(box_height_spaces / 6.0),
        box_width_staff_spaces=_clip(box_width_spaces / 28.0),
        relative_page_y=_clip(center_y / page_height),
        relative_system_x=_clip(relative_system_x),
        system_present=float(system is not None),
        placement_above=float(placement == "above"),
        placement_below=float(placement == "below"),
        placement_within=float(placement == "within"),
        kind_dynamic=float(kind == "dynamic"),
        kind_metronome=float(kind == "metronome"),
        kind_direction=float(kind == "direction"),
        kind_text=float(kind == "text"),
        alpha_ratio=_clip(alpha),
        digit_ratio=_clip(digits),
        punctuation_ratio=_clip(punctuation),
        token_count=_clip(tokens / 7.0),
        title_like=title_like,
        first_system=float(system_index == 0),
    )


@dataclass(frozen=True)
class _AnchorGeometry:
    baseline: MeasureAnchor
    source_index: int
    source_offset: float
    source_count: int
    target_count: int
    relative_x: float
    interval_width_ratio: float
    previous_width_ratio: float
    next_width_ratio: float
    left_boundary_distance: float
    right_boundary_distance: float


def _anchor_geometry(
    system: StaffSystem | ScoreSystemLayout,
    x: float,
    target_measure_count: int,
) -> _AnchorGeometry:
    target_count = max(1, int(target_measure_count))
    baseline = anchor_x_to_measure(system, x, target_count)
    bounds = system_measure_bounds(system)
    width = max(float(system.right - system.left), 1.0)
    relative_x = max(0.0, min(0.999999, (float(x) - system.left) / width))
    if not bounds:
        source_index = min(target_count - 1, int(relative_x * target_count))
        source_offset = relative_x * target_count - source_index
        return _AnchorGeometry(
            baseline, source_index, source_offset, 0, target_count, relative_x,
            1.0, 1.0, 1.0, source_offset, 1.0 - source_offset,
        )
    clamped_x = max(float(system.left), min(float(system.right) - 1e-6, float(x)))
    source_index = len(bounds) - 1
    source_offset = 0.999999
    for index, (left, right) in enumerate(bounds):
        if clamped_x < right or index == len(bounds) - 1:
            source_index = index
            source_offset = max(0.0, min(0.999999, (clamped_x - left) / max(float(right - left), 1.0)))
            break
    widths = [max(1.0, float(right - left)) for left, right in bounds]
    median_width = float(np.median(widths)) if widths else 1.0
    current = widths[source_index]
    previous = widths[source_index - 1] if source_index > 0 else current
    following = widths[source_index + 1] if source_index + 1 < len(widths) else current
    return _AnchorGeometry(
        baseline=baseline,
        source_index=source_index,
        source_offset=source_offset,
        source_count=len(bounds),
        target_count=target_count,
        relative_x=relative_x,
        interval_width_ratio=max(0.2, min(5.0, current / max(median_width, 1.0))),
        previous_width_ratio=max(0.2, min(5.0, previous / max(median_width, 1.0))),
        next_width_ratio=max(0.2, min(5.0, following / max(median_width, 1.0))),
        left_boundary_distance=source_offset,
        right_boundary_distance=1.0 - source_offset,
    )


def measure_anchor_candidate_indices(
    system: StaffSystem,
    x: float,
    target_measure_count: int,
) -> tuple[int, ...]:
    """Return the bounded target-measure candidates used by training and runtime."""
    geometry = _anchor_geometry(system, x, target_measure_count)
    baseline = geometry.baseline.local_index
    if geometry.target_count <= 1:
        return (0,)
    max_shift = min(2, abs(geometry.source_count - geometry.target_count) + 1)
    return tuple(
        range(
            max(0, baseline - max_shift),
            min(geometry.target_count, baseline + max_shift + 1),
        )
    )


def anchor_candidate_feature_vector(
    system: StaffSystem,
    x: float,
    target_measure_count: int,
    candidate_index: int,
    *,
    kind: str,
    placement: str | None,
) -> tuple[float, ...]:
    geometry = _anchor_geometry(system, x, target_measure_count)
    target_count = geometry.target_count
    candidate = max(0, min(target_count - 1, int(candidate_index)))
    source_scale = max(geometry.source_count - 1, 1)
    target_scale = max(target_count - 1, 1)
    mapped_coordinate = (
        (geometry.source_index + geometry.source_offset)
        / max(float(geometry.source_count), 1.0)
        * target_count
        if geometry.source_count > 0
        else geometry.relative_x * target_count
    )
    candidate_center = candidate + 0.5
    return (
        max(-1.0, min(1.0, (candidate - geometry.baseline.local_index) / 2.0)),
        geometry.relative_x,
        geometry.source_index / source_scale,
        geometry.source_offset,
        min(1.0, geometry.source_count / 12.0),
        min(1.0, target_count / 12.0),
        max(-1.0, min(1.0, (geometry.source_count - target_count) / 4.0)),
        candidate / target_scale,
        min(1.0, abs(candidate_center - mapped_coordinate) / 2.5),
        float(candidate == geometry.baseline.local_index),
        min(1.0, geometry.interval_width_ratio / 3.0),
        min(1.0, geometry.previous_width_ratio / 3.0),
        min(1.0, geometry.next_width_ratio / 3.0),
        geometry.left_boundary_distance,
        geometry.right_boundary_distance,
        min(geometry.left_boundary_distance, geometry.right_boundary_distance),
        float(kind == "dynamic"),
        float(kind == "metronome"),
        float(kind == "direction"),
        float(kind == "text"),
        float(placement == "above"),
        float(placement == "below"),
    )


class DirectionAnchorClassifier:
    def __init__(self, model_path: Path | None = None) -> None:
        path = (
            model_path
            or Path(__file__).resolve().parent
            / "resources"
            / "direction_anchor_classifier.json"
        )
        loaded = load_verified_json(path, "direction_anchor_classification")
        payload = loaded.payload if isinstance(loaded.payload, dict) else {}
        role_payload = payload.get("role_model", payload)
        anchor_payload = payload.get("measure_anchor_model", {})
        self.model = StandardizedLogisticModel.from_payload(
            role_payload,
            FEATURE_NAMES,
            verified=loaded.verified,
            status=loaded.status,
        )
        self.anchor_model = VerifiedRandomForestModel.from_payload(
            anchor_payload,
            ANCHOR_FEATURE_NAMES,
            verified=loaded.verified,
            status=loaded.status,
        )
        self.parent_model_version = str(payload.get("model_version", self.model.model_version))
        try:
            recommended_probability = float(
                anchor_payload.get("selection_probability_threshold", 1.0)
            )
            recommended_margin = float(
                anchor_payload.get("selection_margin_threshold", 1.0)
            )
            if not np.isfinite(recommended_probability) or not np.isfinite(recommended_margin):
                raise ValueError("non-finite measure-anchor gate")
            self.anchor_probability_threshold = max(
                DEFAULT_POLICY.direction_measure_anchor_probability_floor,
                recommended_probability,
            )
            self.anchor_margin_threshold = max(
                DEFAULT_POLICY.direction_measure_anchor_margin_floor,
                recommended_margin,
            )
        except (TypeError, ValueError, OverflowError):
            self.anchor_probability_threshold = 1.0
            self.anchor_margin_threshold = 1.0

    @property
    def enabled(self) -> bool:
        return self.model.enabled

    @property
    def anchor_enabled(self) -> bool:
        return self.anchor_model.enabled

    @property
    def model_version(self) -> str:
        return self.parent_model_version

    @property
    def status(self) -> str:
        return self.model.status

    def predict(self, features: DirectionAnchorFeatures) -> float:
        return self.model.predict(features.vector(), neutral=0.5)

    def refine_measure_anchor(
        self,
        system: StaffSystem | ScoreSystemLayout,
        x: float,
        target_measure_count: int,
        *,
        kind: str,
        placement: str | None,
    ) -> MeasureAnchor:
        baseline = anchor_x_to_measure(system, x, target_measure_count)
        geometry = _anchor_geometry(system, x, target_measure_count)
        if (
            not self.anchor_model.enabled
            or geometry.source_count <= 0
            or geometry.source_count == geometry.target_count
            or geometry.target_count <= 1
        ):
            return baseline
        candidate_indices = measure_anchor_candidate_indices(
            system, x, geometry.target_count
        )
        scored = [
            (
                self.anchor_model.predict(
                    anchor_candidate_feature_vector(
                        system, x, geometry.target_count, candidate_index,
                        kind=kind, placement=placement,
                    )
                ),
                candidate_index,
            )
            for candidate_index in candidate_indices
        ]
        if not scored:
            return baseline
        scored.sort(key=lambda item: (item[0], -abs(item[1] - baseline.local_index), -item[1]), reverse=True)
        best_probability, best_index = scored[0]
        second_probability = scored[1][0] if len(scored) > 1 else 0.0
        margin = best_probability - second_probability
        if (
            best_index == baseline.local_index
            or best_probability < self.anchor_probability_threshold
            or margin < self.anchor_margin_threshold
        ):
            return baseline
        confidence = max(
            baseline.confidence,
            min(0.90, 0.50 + 0.40 * best_probability),
        )
        return MeasureAnchor(
            local_index=int(best_index),
            offset_ratio=float(geometry.source_offset),
            confidence=float(confidence),
            method="barline_model_refined",
        )
