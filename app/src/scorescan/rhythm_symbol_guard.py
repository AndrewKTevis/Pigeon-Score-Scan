from __future__ import annotations

"""Pairwise source-crop guard for rhythm-only patch proposals.

The semantic ensemble remains solely responsible for proposing a duration.  This
module compares the proposed event with the current template against the same small,
staff-normalised evidence image.  A verified CPU model may only veto a proposal, and
acceptance requires mutually consistent forward and reverse comparisons.
"""

import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from .model_registry import load_verified_json
from .policy import DEFAULT_POLICY
from .score_ir import MeasureIR, NoteIR
from .tree_model import VerifiedRandomForestModel
from .local_symbol_image import decode_guard_image, event_position, local_descriptor
from .visual_evidence import VisualMeasureEvidence

PATCH_WIDTH = 24
PATCH_HEIGHT = 48
DESCRIPTOR_WIDTH = 8
DESCRIPTOR_HEIGHT = 16
_PIXEL_COUNT = DESCRIPTOR_WIDTH * DESCRIPTOR_HEIGHT
_MAX_ENCODED_IMAGE_BYTES = 65_536
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

NOTE_TYPE_CLASSES = ("whole", "half", "quarter", "eighth", "16th", "32nd", "other")

_VISUAL_SUMMARY_NAMES = (
    "ink_mean",
    "ink_max",
    "gradient_mean",
    "gradient_max",
    "upper_ink",
    "lower_ink",
    "centre_ink",
    "candidate_stem_side_ink",
    "candidate_flag_zone_ink",
    "dot_zone_ink",
    "head_region_ink",
    "head_core_ink",
    "top_edge_ink",
    "bottom_edge_ink",
    "left_edge_ink",
    "right_edge_ink",
    "upper_horizontal_variation",
    "lower_horizontal_variation",
    "left_vertical_variation",
    "right_vertical_variation",
    "right_dot_peak",
    "upper_peak",
    "lower_peak",
)
_CANDIDATE_SUMMARY_NAMES = (
    tuple(f"candidate_type_{name}" for name in NOTE_TYPE_CLASSES)
    + (
        "candidate_dots_scaled",
        "candidate_is_rest",
        "candidate_beam_level_scaled",
        "candidate_open_notehead",
        "candidate_has_stem",
        "candidate_duration_scaled",
        "candidate_onset_ratio",
        "candidate_staff_y_ratio",
    )
)
_COMPATIBILITY_SUMMARY_NAMES = (
    "expected_dot_ink",
    "expected_no_dot_clearance",
    "expected_open_head_hole",
    "expected_filled_head_ink",
    "expected_stem_ink",
    "expected_stemless_clearance",
    "expected_beam_ink",
    "expected_unbeamed_clearance",
    "expected_rest_centre_ink",
    "expected_note_head_ink",
    "dot_mismatch",
    "open_head_mismatch",
    "beam_mismatch",
    "stem_mismatch",
)
RHYTHM_SYMBOL_OBSERVATION_FEATURE_NAMES = (
    _VISUAL_SUMMARY_NAMES + _CANDIDATE_SUMMARY_NAMES + _COMPATIBILITY_SUMMARY_NAMES
)
_OBSERVATION_FEATURE_COUNT = len(RHYTHM_SYMBOL_OBSERVATION_FEATURE_NAMES)
RHYTHM_SYMBOL_EVENT_FEATURE_NAMES = (
    tuple(f"proposed_{name}" for name in RHYTHM_SYMBOL_OBSERVATION_FEATURE_NAMES)
    + tuple(f"template_{name}" for name in RHYTHM_SYMBOL_OBSERVATION_FEATURE_NAMES)
    + tuple(f"delta_{name}" for name in RHYTHM_SYMBOL_OBSERVATION_FEATURE_NAMES)
)
RHYTHM_SYMBOL_FEATURE_NAMES = (
    tuple(f"mean_{name}" for name in RHYTHM_SYMBOL_EVENT_FEATURE_NAMES)
    + tuple(f"minimum_{name}" for name in RHYTHM_SYMBOL_EVENT_FEATURE_NAMES)
    + tuple(f"maximum_{name}" for name in RHYTHM_SYMBOL_EVENT_FEATURE_NAMES)
)


def _unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _beam_level(note: NoteIR) -> int:
    return {
        "eighth": 1,
        "16th": 2,
        "32nd": 3,
        "64th": 4,
        "128th": 5,
    }.get(note.note_type.strip().casefold(), 0)


def _type_class(note: NoteIR) -> str:
    value = note.note_type.strip().casefold()
    return value if value in NOTE_TYPE_CLASSES[:-1] else "other"


def _local_descriptor(image: np.ndarray, x_ratio: float, y_ratio: float) -> tuple[float, ...]:
    return local_descriptor(
        image,
        x_ratio,
        y_ratio,
        patch_width=PATCH_WIDTH,
        patch_height=PATCH_HEIGHT,
        descriptor_width=DESCRIPTOR_WIDTH,
        descriptor_height=DESCRIPTOR_HEIGHT,
    )


@dataclass(frozen=True)
class RhythmSymbolObservation:
    descriptor: tuple[float, ...]
    note_type: str
    dots: int
    rest: bool
    beam_level: int
    open_notehead: bool
    has_stem: bool
    duration: Fraction
    onset_ratio: float
    staff_y_ratio: float

    def feature_vector(self) -> list[float]:
        if len(self.descriptor) != _PIXEL_COUNT * 2:
            raise ValueError("invalid rhythm symbol descriptor length")
        density = np.asarray(self.descriptor[:_PIXEL_COUNT], dtype=np.float64).reshape(
            DESCRIPTOR_HEIGHT, DESCRIPTOR_WIDTH
        )
        gradient = np.asarray(self.descriptor[_PIXEL_COUNT:], dtype=np.float64).reshape(
            DESCRIPTOR_HEIGHT, DESCRIPTOR_WIDTH
        )
        stem_up = bool(self.staff_y_ratio >= 0.5)
        stem_side = density[:8, 4:7] if stem_up else density[8:, 1:4]
        flag_zone = density[:5, 4:8] if stem_up else density[11:, 0:4]
        horizontal_variation = np.abs(np.diff(density, axis=1)).mean(axis=1)
        vertical_variation = np.abs(np.diff(density, axis=0)).mean(axis=0)
        visual = [
            float(np.mean(density)),
            float(np.max(density, initial=0.0)),
            float(np.mean(gradient)),
            float(np.max(gradient, initial=0.0)),
            float(np.mean(density[:8, :])),
            float(np.mean(density[8:, :])),
            float(np.mean(density[5:11, 2:6])),
            float(np.mean(stem_side)),
            float(np.mean(flag_zone)),
            float(np.mean(density[6:10, 5:8])),
            float(np.mean(density[6:10, 2:6])),
            float(np.mean(density[7:9, 3:5])),
            float(np.mean(density[:4, :])),
            float(np.mean(density[-4:, :])),
            float(np.mean(density[:, :2])),
            float(np.mean(density[:, -2:])),
            float(np.mean(horizontal_variation[:5])),
            float(np.mean(horizontal_variation[-5:])),
            float(np.mean(vertical_variation[:3])),
            float(np.mean(vertical_variation[-3:])),
            float(np.max(density[6:10, 6:8], initial=0.0)),
            float(np.max(density[:5, :], initial=0.0)),
            float(np.max(density[-5:, :], initial=0.0)),
        ]
        type_class = self.note_type if self.note_type in NOTE_TYPE_CLASSES else "other"
        candidate = [float(type_class == value) for value in NOTE_TYPE_CLASSES] + [
            _unit(max(0, int(self.dots)) / 2.0),
            float(bool(self.rest)),
            _unit(max(0, int(self.beam_level)) / 5.0),
            float(bool(self.open_notehead)),
            float(bool(self.has_stem)),
            _unit(float(max(self.duration, Fraction(0, 1))) / 4.0),
            _unit(self.onset_ratio),
            _unit(self.staff_y_ratio),
        ]
        dot_observation = visual[9]
        open_observation = max(0.0, visual[10] - visual[11])
        stem_observation = visual[7]
        beam_observation = visual[8]
        dots = candidate[7]
        rest = candidate[8]
        beam = candidate[9]
        opened = candidate[10]
        stem = candidate[11]
        compatibility = [
            dots * dot_observation,
            (1.0 - dots) * (1.0 - dot_observation),
            opened * open_observation,
            (1.0 - opened) * (1.0 - open_observation),
            stem * stem_observation,
            (1.0 - stem) * (1.0 - stem_observation),
            beam * beam_observation,
            (1.0 - beam) * (1.0 - beam_observation),
            rest * visual[6],
            (1.0 - rest) * visual[10],
            abs(dot_observation - dots),
            abs(open_observation - opened),
            abs(beam_observation - beam),
            abs(stem_observation - stem),
        ]
        result = [*visual, *candidate, *compatibility]
        if len(result) != _OBSERVATION_FEATURE_COUNT:
            raise AssertionError("rhythm observation feature schema mismatch")
        return [_unit(value) for value in result]


@dataclass(frozen=True)
class RhythmSymbolComparisonInput:
    proposed: RhythmSymbolObservation
    template: RhythmSymbolObservation

    def feature_vector(self) -> list[float]:
        proposed = self.proposed.feature_vector()
        template = self.template.feature_vector()
        return [
            *proposed,
            *template,
            *[max(-1.0, min(1.0, left - right)) for left, right in zip(proposed, template, strict=True)],
        ]

    def reversed(self) -> "RhythmSymbolComparisonInput":
        return RhythmSymbolComparisonInput(self.template, self.proposed)


@dataclass(frozen=True)
class RhythmSymbolTransactionInput:
    comparisons: tuple[RhythmSymbolComparisonInput, ...]

    def feature_vector(self) -> list[float]:
        if not self.comparisons:
            raise ValueError("rhythm symbol transaction requires at least one event")
        rows = np.asarray(
            [comparison.feature_vector() for comparison in self.comparisons],
            dtype=np.float64,
        )
        return [
            *[float(value) for value in np.mean(rows, axis=0)],
            *[float(value) for value in np.min(rows, axis=0)],
            *[float(value) for value in np.max(rows, axis=0)],
        ]

    def reversed(self) -> "RhythmSymbolTransactionInput":
        return RhythmSymbolTransactionInput(
            tuple(comparison.reversed() for comparison in self.comparisons)
        )


@dataclass(frozen=True)
class RhythmSymbolGuardCalibration:
    forward_probability: float
    reverse_probability: float
    confidence: float
    threshold: float
    accepted: bool
    model_version: str


def _build_observation(
    evidence: VisualMeasureEvidence | None,
    measure: MeasureIR,
    event_index: int,
) -> RhythmSymbolObservation | None:
    if evidence is None or event_index < 0 or event_index >= len(measure.notes):
        return None
    image = decode_guard_image(evidence.rhythm_guard_image)
    if image is None:
        return None
    note = measure.notes[event_index]
    if note.grace or note.chord or note.duration <= 0:
        return None
    x_ratio, y_ratio = event_position(measure, note)
    note_type = _type_class(note)
    return RhythmSymbolObservation(
        descriptor=_local_descriptor(image, x_ratio, y_ratio),
        note_type=note_type,
        dots=max(0, int(note.dots)),
        rest=bool(note.rest),
        beam_level=_beam_level(note),
        open_notehead=bool(
            not note.rest and note.pitch is not None and note_type in {"whole", "half"}
        ),
        has_stem=bool(not note.rest and note.pitch is not None and note_type != "whole"),
        duration=note.duration,
        onset_ratio=x_ratio,
        staff_y_ratio=y_ratio,
    )


def build_rhythm_symbol_comparison(
    evidence: VisualMeasureEvidence | None,
    proposed_measure: MeasureIR,
    template_measure: MeasureIR,
    event_index: int,
) -> RhythmSymbolComparisonInput | None:
    proposed = _build_observation(evidence, proposed_measure, event_index)
    template = _build_observation(evidence, template_measure, event_index)
    if proposed is None or template is None:
        return None
    return RhythmSymbolComparisonInput(proposed, template)


def build_rhythm_symbol_transaction(
    evidence: VisualMeasureEvidence | None,
    proposed_measure: MeasureIR,
    template_measure: MeasureIR,
    changed_event_indices: tuple[int, ...],
) -> RhythmSymbolTransactionInput | None:
    if not changed_event_indices:
        return None
    comparisons: list[RhythmSymbolComparisonInput] = []
    for event_index in changed_event_indices:
        comparison = build_rhythm_symbol_comparison(
            evidence, proposed_measure, template_measure, event_index
        )
        if comparison is None:
            return None
        comparisons.append(comparison)
    return RhythmSymbolTransactionInput(tuple(comparisons))


class RhythmSymbolGuard:
    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).with_name("resources") / "rhythm_symbol_guard.json"
        loaded = load_verified_json(model_path, "rhythm_symbol_guard")
        payload = loaded.payload
        self.model = VerifiedRandomForestModel.load(
            model_path,
            "rhythm_symbol_guard",
            RHYTHM_SYMBOL_FEATURE_NAMES,
            loaded=loaded,
        )
        try:
            stored_threshold = float(payload.get("auto_patch_threshold", 1.0))
        except (TypeError, ValueError, OverflowError):
            stored_threshold = 1.0
        self.threshold = max(
            float(DEFAULT_POLICY.rhythm_symbol_guard_probability_floor),
            max(0.0, min(1.0, stored_threshold)),
        )
        self.model_verified = self.model.verified and loaded.verified
        self.model_status = self.model.status if self.model.enabled else loaded.status
        self.model_version = self.model.model_version
        self.enabled = self.model.enabled and self.model_verified

    def predict_probability(self, item: RhythmSymbolTransactionInput) -> float:
        if not self.enabled or not self.model_verified:
            return 0.5
        return self.model.predict(item.feature_vector(), neutral=0.5)

    def calibrate(self, item: RhythmSymbolTransactionInput) -> RhythmSymbolGuardCalibration:
        forward = self.predict_probability(item)
        reverse = self.predict_probability(item.reversed())
        confidence = min(forward, 1.0 - reverse)
        accepted = bool(
            self.enabled
            and self.model_verified
            and confidence >= self.threshold
        )
        return RhythmSymbolGuardCalibration(
            forward_probability=round(forward, 6),
            reverse_probability=round(reverse, 6),
            confidence=round(confidence, 6),
            threshold=round(self.threshold, 6),
            accepted=accepted,
            model_version=self.model_version,
        )
