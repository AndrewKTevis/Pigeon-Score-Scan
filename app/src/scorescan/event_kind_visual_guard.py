from __future__ import annotations

"""Veto-only local visual confirmation for one note-versus-rest transaction.

Independent OMR families and :mod:`event_kind_consensus` remain solely responsible for
proposing MusicXML.  This module answers one bounded question: does the preserved local
source evidence support replacing exactly one pitched note with a rest, or one rest with
an already-proposed pitched note?  It cannot create support, choose a pitch or duration,
edit XML, or authorise multiple event changes.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .local_symbol_image import decode_symbol_guard_image, event_position
from .model_registry import load_verified_json
from .policy import DEFAULT_POLICY
from .score_ir import MeasureIR, NoteIR
from .tree_model import VerifiedRandomForestModel
from .visual_evidence import VisualMeasureEvidence

from .local_event_descriptor import (
    FEATURE_NAMES as LOCAL_EVENT_FEATURE_NAMES,
    event_patch_descriptor,
)

PATCH_WIDTH = 48
PATCH_HEIGHT = 48
DESCRIPTOR_WIDTH = 8
DESCRIPTOR_HEIGHT = 8
SUPPORTED_TYPES = ("whole", "half", "quarter", "eighth", "16th")

_CONTEXT_FEATURES = (
    "target_is_pitched_note",
    "template_is_pitched_note",
    *(f"note_type_{value}" for value in SUPPORTED_TYPES),
    "note_type_other",
    "dot_count_scaled",
    "vertical_separation_scaled",
)
_SIDE_NAMES = LOCAL_EVENT_FEATURE_NAMES
EVENT_KIND_VISUAL_FEATURE_NAMES = _CONTEXT_FEATURES + tuple(
    f"{prefix}_{name}"
    for prefix in ("target", "template", "absolute_difference")
    for name in _SIDE_NAMES
)


def _note_type(note: NoteIR) -> str:
    return str(note.note_type or "").strip().casefold()


def _changed_kind_indices(before: MeasureIR, after: MeasureIR) -> tuple[int, ...]:
    if len(before.notes) != len(after.notes):
        return ()
    return tuple(
        index
        for index, (left, right) in enumerate(zip(before.notes, after.notes, strict=True))
        if bool(left.rest) != bool(right.rest)
    )


def event_kind_visual_features(
    evidence: VisualMeasureEvidence,
    before: MeasureIR,
    after: MeasureIR,
    event_index: int,
) -> tuple[float, ...] | None:
    image = decode_symbol_guard_image(evidence.symbol_guard_image)
    if image is None or len(before.notes) != len(after.notes):
        return None
    if not (0 <= event_index < len(before.notes)):
        return None
    template = before.notes[event_index]
    target = after.notes[event_index]
    if bool(template.rest) == bool(target.rest):
        return None
    if template.grace or target.grace or template.chord or target.chord:
        return None
    pitched = target if not target.rest else template
    if pitched.pitch is None:
        return None
    note_type = _note_type(target) or _note_type(template)
    if note_type not in SUPPORTED_TYPES:
        return None
    if template.onset != target.onset or template.duration != target.duration:
        return None

    target_x, target_y = event_position(after, target)
    template_x, template_y = event_position(before, template)
    target_values = event_patch_descriptor(image, target_x, target_y)
    template_values = event_patch_descriptor(image, template_x, template_y)
    difference = np.abs(target_values - template_values)

    context = [
        0.0 if target.rest else 1.0,
        0.0 if template.rest else 1.0,
        *(1.0 if note_type == value else 0.0 for value in SUPPORTED_TYPES),
        1.0 if note_type not in SUPPORTED_TYPES else 0.0,
        min(max(int(target.dots), int(template.dots)), 2) / 2.0,
        min(abs(target_y - template_y) * 2.0, 1.0),
    ]
    vector = np.concatenate(
        (np.asarray(context, dtype=np.float64), target_values, template_values, difference)
    )
    if vector.size != len(EVENT_KIND_VISUAL_FEATURE_NAMES):
        raise AssertionError(
            f"event-kind visual feature mismatch: {vector.size} != {len(EVENT_KIND_VISUAL_FEATURE_NAMES)}"
        )
    return tuple(float(value) for value in vector)


@dataclass(frozen=True)
class EventKindVisualCalibration:
    probability: float
    threshold: float
    accepted: bool
    available: bool
    model_version: str


@dataclass(frozen=True)
class EventKindVisualAudit:
    applicable: bool
    changed_event_count: int
    probability: float
    threshold: float
    accepted: bool
    reason: str
    model_version: str


class EventKindVisualGuard:
    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).with_name("resources") / "event_kind_visual_guard.json"
        loaded = load_verified_json(model_path, "event_kind_visual_guard")
        payload = loaded.payload
        self.model = VerifiedRandomForestModel.load(
            model_path,
            "event_kind_visual_guard",
            EVENT_KIND_VISUAL_FEATURE_NAMES,
            loaded=loaded,
        )
        try:
            threshold = float(payload.get("auto_patch_threshold", 1.0))
        except (TypeError, ValueError, OverflowError):
            threshold = 1.0
        self.threshold = max(
            float(DEFAULT_POLICY.event_kind_visual_guard_probability_floor),
            max(0.0, min(1.0, threshold)),
        )
        self.model_verified = bool(self.model.verified and loaded.verified)
        self.model_version = self.model.model_version
        self.enabled = bool(self.model.enabled)
        self.model_status = self.model.status if self.model.enabled else loaded.status

    def calibrate(
        self,
        evidence: VisualMeasureEvidence,
        before: MeasureIR,
        after: MeasureIR,
        event_index: int,
    ) -> EventKindVisualCalibration:
        values = event_kind_visual_features(evidence, before, after, event_index)
        if values is None or not (self.enabled and self.model_verified):
            return EventKindVisualCalibration(
                probability=0.5,
                threshold=round(self.threshold, 6),
                accepted=False,
                available=False,
                model_version=self.model_version,
            )
        probability = float(self.model.predict(values))
        return EventKindVisualCalibration(
            probability=round(probability, 6),
            threshold=round(self.threshold, 6),
            accepted=probability >= self.threshold,
            available=True,
            model_version=self.model_version,
        )

    def audit_transaction(
        self,
        evidence: VisualMeasureEvidence | None,
        before: MeasureIR,
        after: MeasureIR,
    ) -> EventKindVisualAudit:
        changed = _changed_kind_indices(before, after)
        threshold = round(self.threshold, 6)
        if not changed:
            return EventKindVisualAudit(
                False, 0, 0.5, threshold, True, "not_applicable", self.model_version
            )
        encoded = "" if evidence is None else str(evidence.symbol_guard_image or "")
        if not encoded:
            return EventKindVisualAudit(
                False,
                len(changed),
                0.5,
                threshold,
                True,
                "source_evidence_unavailable",
                self.model_version,
            )
        if len(changed) != 1:
            return EventKindVisualAudit(
                True,
                len(changed),
                0.5,
                threshold,
                False,
                "multiple_event_kind_changes_require_review",
                self.model_version,
            )
        event_index = changed[0]
        template = before.notes[event_index]
        target = after.notes[event_index]
        if template.grace or target.grace or template.chord or target.chord:
            return EventKindVisualAudit(
                True,
                1,
                0.5,
                threshold,
                False,
                "unsupported_event_kind_visual_transaction",
                self.model_version,
            )
        calibration = self.calibrate(evidence, before, after, event_index)  # type: ignore[arg-type]
        if not calibration.available:
            return EventKindVisualAudit(
                True,
                1,
                calibration.probability,
                calibration.threshold,
                False,
                "event_kind_visual_evidence_or_model_unavailable",
                calibration.model_version,
            )
        return EventKindVisualAudit(
            True,
            1,
            calibration.probability,
            calibration.threshold,
            calibration.accepted,
            "visual_event_kind_confirmed" if calibration.accepted else "visual_event_kind_conflict",
            calibration.model_version,
        )
