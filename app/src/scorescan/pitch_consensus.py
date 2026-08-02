from __future__ import annotations

"""Conservative event-level pitch consensus for complementary OMR errors.

Whole-measure voting cannot recover a correct measure when every preprocessing variant
contains a different isolated pitch error.  This module proposes a narrower repair:
only pitch/accidental elements may be copied, and only after every candidate agrees on
the complete non-pitch event skeleton.  Preprocessing siblings are collapsed to one
family vote, conflicting siblings abstain, and a verified CPU model may only veto the
proposal.

The repair never changes rhythm, onset, voice, rests, chords, grace notes, tuplets,
ties, slurs, articulations, ornaments, directions, attributes, or barlines.
"""

import copy
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from lxml import etree

from .accidental_presence_guard import AccidentalPresenceGuard
from .accidental_semantics import (
    accidental_regression,
    chromatic_change_indices,
    normalise_accidental,
)
from .model_registry import load_verified_json
from .policy import DEFAULT_POLICY
from .score_ir import MeasureIR, NoteIR, PitchIR, measure_from_xml
from .tree_model import VerifiedRandomForestModel
from .variant_family import group_complete_families
from .visual_evidence import VisualMeasureEvidence, pitch_transaction_gap_pair

FEATURE_NAMES = (
    "candidate_count_scaled",
    "eligible_family_count_scaled",
    "voting_family_count_scaled",
    "changed_event_count_scaled",
    "changed_event_ratio",
    "minimum_winner_family_support_ratio",
    "mean_winner_family_support_ratio",
    "minimum_winner_margin_ratio",
    "mean_winner_margin_ratio",
    "maximum_template_family_support_ratio",
    "family_abstention_ratio",
    "mean_support_page_probability",
    "mean_support_measure_probability",
    "mean_support_visual_probability",
    "mean_support_event_probability",
    "mean_support_context_probability",
    "mean_support_ensemble_probability",
    "minimum_support_ensemble_probability",
    "mean_support_page_score_margin_scaled",
    "mean_support_vs_template_measure_probability",
    "mean_support_vs_template_visual_probability",
    "mean_support_vs_template_event_probability",
    "mean_support_vs_template_context_probability",
    "mean_support_vs_template_ensemble_probability",
    "visual_evidence_available",
    "changed_staff_position_ratio",
    "maximum_staff_position_delta_scaled",
    "accidental_only_change_ratio",
    "notehead_exact_cell_improvement",
    "notehead_near_cell_improvement",
    "notehead_vertical_chamfer_improvement",
    "notehead_severe_vertical_improvement",
    "notehead_visual_unmatched_improvement",
    "notehead_column_centroid_improvement",
    "notehead_column_order_improvement",
    "template_notehead_exact_cell_gap",
    "template_notehead_near_cell_gap",
    "template_notehead_vertical_chamfer_gap",
    "template_notehead_severe_vertical_gap",
    "template_notehead_visual_unmatched_gap",
    "template_notehead_column_centroid_gap",
    "template_notehead_column_order_gap",
    "proposal_notehead_exact_cell_gap",
    "proposal_notehead_near_cell_gap",
    "proposal_notehead_vertical_chamfer_gap",
    "proposal_notehead_severe_vertical_gap",
    "proposal_notehead_visual_unmatched_gap",
    "proposal_notehead_column_centroid_gap",
    "proposal_notehead_column_order_gap",
    "strict_notehead_exact_cell_improvement",
    "strict_notehead_near_cell_improvement",
    "strict_notehead_vertical_chamfer_improvement",
    "strict_notehead_severe_vertical_improvement",
    "strict_notehead_visual_unmatched_improvement",
    "strict_notehead_column_centroid_improvement",
    "strict_notehead_column_order_improvement",
    "template_strict_notehead_exact_cell_gap",
    "template_strict_notehead_near_cell_gap",
    "template_strict_notehead_vertical_chamfer_gap",
    "template_strict_notehead_severe_vertical_gap",
    "template_strict_notehead_visual_unmatched_gap",
    "template_strict_notehead_column_centroid_gap",
    "template_strict_notehead_column_order_gap",
    "proposal_strict_notehead_exact_cell_gap",
    "proposal_strict_notehead_near_cell_gap",
    "proposal_strict_notehead_vertical_chamfer_gap",
    "proposal_strict_notehead_severe_vertical_gap",
    "proposal_strict_notehead_visual_unmatched_gap",
    "proposal_strict_notehead_column_centroid_gap",
    "proposal_strict_notehead_column_order_gap",
)


PITCH_VISUAL_FEATURE_NAMES = (
    "changed_event_count_scaled",
    "changed_event_ratio",
    "changed_staff_position_ratio",
    "maximum_staff_position_delta_scaled",
    "notehead_exact_cell_improvement",
    "notehead_near_cell_improvement",
    "notehead_vertical_chamfer_improvement",
    "notehead_severe_vertical_improvement",
    "notehead_visual_unmatched_improvement",
    "notehead_column_centroid_improvement",
    "notehead_column_order_improvement",
    "template_notehead_exact_cell_gap",
    "template_notehead_near_cell_gap",
    "template_notehead_vertical_chamfer_gap",
    "template_notehead_severe_vertical_gap",
    "template_notehead_visual_unmatched_gap",
    "template_notehead_column_centroid_gap",
    "template_notehead_column_order_gap",
    "proposal_notehead_exact_cell_gap",
    "proposal_notehead_near_cell_gap",
    "proposal_notehead_vertical_chamfer_gap",
    "proposal_notehead_severe_vertical_gap",
    "proposal_notehead_visual_unmatched_gap",
    "proposal_notehead_column_centroid_gap",
    "proposal_notehead_column_order_gap",
    "strict_notehead_exact_cell_improvement",
    "strict_notehead_near_cell_improvement",
    "strict_notehead_vertical_chamfer_improvement",
    "strict_notehead_severe_vertical_improvement",
    "strict_notehead_visual_unmatched_improvement",
    "strict_notehead_column_centroid_improvement",
    "strict_notehead_column_order_improvement",
    "template_strict_notehead_exact_cell_gap",
    "template_strict_notehead_near_cell_gap",
    "template_strict_notehead_vertical_chamfer_gap",
    "template_strict_notehead_severe_vertical_gap",
    "template_strict_notehead_visual_unmatched_gap",
    "template_strict_notehead_column_centroid_gap",
    "template_strict_notehead_column_order_gap",
    "proposal_strict_notehead_exact_cell_gap",
    "proposal_strict_notehead_near_cell_gap",
    "proposal_strict_notehead_vertical_chamfer_gap",
    "proposal_strict_notehead_severe_vertical_gap",
    "proposal_strict_notehead_visual_unmatched_gap",
    "proposal_strict_notehead_column_centroid_gap",
    "proposal_strict_notehead_column_order_gap",
)
PITCH_VISUAL_FEATURE_INDICES = tuple(FEATURE_NAMES.index(name) for name in PITCH_VISUAL_FEATURE_NAMES)


@dataclass(frozen=True)
class PitchPatchCandidate:
    variant: str
    family: str
    measure: etree._Element
    semantics: MeasureIR
    page_score: float
    page_probability: float
    measure_probability: float
    visual_probability: float
    event_probability: float
    context_probability: float
    ensemble_probability: float
    valid: bool


@dataclass(frozen=True)
class PitchPatchInput:
    candidate_count: int
    eligible_family_count: int
    voting_family_count: int
    changed_event_count: int
    total_event_count: int
    minimum_winner_family_support_ratio: float
    mean_winner_family_support_ratio: float
    minimum_winner_margin_ratio: float
    mean_winner_margin_ratio: float
    maximum_template_family_support_ratio: float
    family_abstention_ratio: float
    mean_support_page_probability: float
    mean_support_measure_probability: float
    mean_support_visual_probability: float
    mean_support_event_probability: float
    mean_support_context_probability: float
    mean_support_ensemble_probability: float
    minimum_support_ensemble_probability: float
    mean_support_page_score_margin: float
    mean_support_vs_template_measure_probability: float
    mean_support_vs_template_visual_probability: float
    mean_support_vs_template_event_probability: float
    mean_support_vs_template_context_probability: float
    mean_support_vs_template_ensemble_probability: float
    visual_evidence_available: bool
    changed_staff_position_ratio: float
    maximum_staff_position_delta: float
    accidental_only_change_ratio: float
    notehead_exact_cell_improvement: float
    notehead_near_cell_improvement: float
    notehead_vertical_chamfer_improvement: float
    notehead_severe_vertical_improvement: float
    notehead_visual_unmatched_improvement: float
    notehead_column_centroid_improvement: float
    notehead_column_order_improvement: float
    template_notehead_exact_cell_gap: float
    template_notehead_near_cell_gap: float
    template_notehead_vertical_chamfer_gap: float
    template_notehead_severe_vertical_gap: float
    template_notehead_visual_unmatched_gap: float
    template_notehead_column_centroid_gap: float
    template_notehead_column_order_gap: float
    proposal_notehead_exact_cell_gap: float
    proposal_notehead_near_cell_gap: float
    proposal_notehead_vertical_chamfer_gap: float
    proposal_notehead_severe_vertical_gap: float
    proposal_notehead_visual_unmatched_gap: float
    proposal_notehead_column_centroid_gap: float
    proposal_notehead_column_order_gap: float
    strict_notehead_exact_cell_improvement: float
    strict_notehead_near_cell_improvement: float
    strict_notehead_vertical_chamfer_improvement: float
    strict_notehead_severe_vertical_improvement: float
    strict_notehead_visual_unmatched_improvement: float
    strict_notehead_column_centroid_improvement: float
    strict_notehead_column_order_improvement: float
    template_strict_notehead_exact_cell_gap: float
    template_strict_notehead_near_cell_gap: float
    template_strict_notehead_vertical_chamfer_gap: float
    template_strict_notehead_severe_vertical_gap: float
    template_strict_notehead_visual_unmatched_gap: float
    template_strict_notehead_column_centroid_gap: float
    template_strict_notehead_column_order_gap: float
    proposal_strict_notehead_exact_cell_gap: float
    proposal_strict_notehead_near_cell_gap: float
    proposal_strict_notehead_vertical_chamfer_gap: float
    proposal_strict_notehead_severe_vertical_gap: float
    proposal_strict_notehead_visual_unmatched_gap: float
    proposal_strict_notehead_column_centroid_gap: float
    proposal_strict_notehead_column_order_gap: float

    def feature_vector(self) -> list[float]:
        def unit(value: float) -> float:
            return max(0.0, min(1.0, float(value)))

        def signed(value: float) -> float:
            return max(-1.0, min(1.0, float(value)))

        return [
            unit(max(0, self.candidate_count - 1) / 7.0),
            unit(self.eligible_family_count / 4.0),
            unit(self.voting_family_count / 4.0),
            unit(self.changed_event_count / 16.0),
            unit(self.changed_event_count / max(1, self.total_event_count)),
            unit(self.minimum_winner_family_support_ratio),
            unit(self.mean_winner_family_support_ratio),
            unit(self.minimum_winner_margin_ratio),
            unit(self.mean_winner_margin_ratio),
            unit(self.maximum_template_family_support_ratio),
            unit(self.family_abstention_ratio),
            unit(self.mean_support_page_probability),
            unit(self.mean_support_measure_probability),
            unit(self.mean_support_visual_probability),
            unit(self.mean_support_event_probability),
            unit(self.mean_support_context_probability),
            unit(self.mean_support_ensemble_probability),
            unit(self.minimum_support_ensemble_probability),
            signed(self.mean_support_page_score_margin / 100.0),
            signed(self.mean_support_vs_template_measure_probability),
            signed(self.mean_support_vs_template_visual_probability),
            signed(self.mean_support_vs_template_event_probability),
            signed(self.mean_support_vs_template_context_probability),
            signed(self.mean_support_vs_template_ensemble_probability),
            float(bool(self.visual_evidence_available)),
            unit(self.changed_staff_position_ratio),
            unit(self.maximum_staff_position_delta / 14.0),
            unit(self.accidental_only_change_ratio),
            signed(self.notehead_exact_cell_improvement),
            signed(self.notehead_near_cell_improvement),
            signed(self.notehead_vertical_chamfer_improvement),
            signed(self.notehead_severe_vertical_improvement),
            signed(self.notehead_visual_unmatched_improvement),
            signed(self.notehead_column_centroid_improvement),
            signed(self.notehead_column_order_improvement),
            unit(self.template_notehead_exact_cell_gap),
            unit(self.template_notehead_near_cell_gap),
            unit(self.template_notehead_vertical_chamfer_gap),
            unit(self.template_notehead_severe_vertical_gap),
            unit(self.template_notehead_visual_unmatched_gap),
            unit(self.template_notehead_column_centroid_gap),
            unit(self.template_notehead_column_order_gap),
            unit(self.proposal_notehead_exact_cell_gap),
            unit(self.proposal_notehead_near_cell_gap),
            unit(self.proposal_notehead_vertical_chamfer_gap),
            unit(self.proposal_notehead_severe_vertical_gap),
            unit(self.proposal_notehead_visual_unmatched_gap),
            unit(self.proposal_notehead_column_centroid_gap),
            unit(self.proposal_notehead_column_order_gap),
            signed(self.strict_notehead_exact_cell_improvement),
            signed(self.strict_notehead_near_cell_improvement),
            signed(self.strict_notehead_vertical_chamfer_improvement),
            signed(self.strict_notehead_severe_vertical_improvement),
            signed(self.strict_notehead_visual_unmatched_improvement),
            signed(self.strict_notehead_column_centroid_improvement),
            signed(self.strict_notehead_column_order_improvement),
            unit(self.template_strict_notehead_exact_cell_gap),
            unit(self.template_strict_notehead_near_cell_gap),
            unit(self.template_strict_notehead_vertical_chamfer_gap),
            unit(self.template_strict_notehead_severe_vertical_gap),
            unit(self.template_strict_notehead_visual_unmatched_gap),
            unit(self.template_strict_notehead_column_centroid_gap),
            unit(self.template_strict_notehead_column_order_gap),
            unit(self.proposal_strict_notehead_exact_cell_gap),
            unit(self.proposal_strict_notehead_near_cell_gap),
            unit(self.proposal_strict_notehead_vertical_chamfer_gap),
            unit(self.proposal_strict_notehead_severe_vertical_gap),
            unit(self.proposal_strict_notehead_visual_unmatched_gap),
            unit(self.proposal_strict_notehead_column_centroid_gap),
            unit(self.proposal_strict_notehead_column_order_gap),
        ]


@dataclass(frozen=True)
class PitchPatchCalibration:
    probability: float
    threshold: float
    accepted: bool
    model_version: str
    target_precision: float


@dataclass(frozen=True)
class PitchPatchResult:
    patched_measure: etree._Element | None
    changed_event_indices: tuple[int, ...]
    probability: float
    threshold: float
    accepted: bool
    reason: str
    input: PitchPatchInput | None = None
    model_version: str = "disabled"
    visual_guard_probability: float = 0.5
    visual_guard_threshold: float = 1.0
    visual_guard_model_version: str = "not_applicable"
    accidental_guard_probability: float = 0.5
    accidental_guard_threshold: float = 1.0
    accidental_guard_model_version: str = "not_applicable"


@dataclass(frozen=True)
class PitchVisualGuardCalibration:
    probability: float
    threshold: float
    accepted: bool
    model_version: str


class PitchVisualGuard:
    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).with_name("resources") / "pitch_visual_guard.json"
        loaded = load_verified_json(model_path, "pitch_visual_guard")
        payload = loaded.payload
        self.model = VerifiedRandomForestModel.load(
            model_path,
            "pitch_visual_guard",
            PITCH_VISUAL_FEATURE_NAMES,
            loaded=loaded,
        )
        try:
            stored_threshold = float(payload.get("auto_patch_threshold", 1.0))
        except (TypeError, ValueError, OverflowError):
            stored_threshold = 1.0
        self.threshold = max(
            float(DEFAULT_POLICY.pitch_visual_guard_probability_floor),
            max(0.0, min(1.0, stored_threshold)),
        )
        self.model_verified = self.model.verified and loaded.verified
        self.model_status = self.model.status if self.model.enabled else loaded.status
        self.model_version = self.model.model_version
        self.enabled = self.model.enabled

    @staticmethod
    def vector(item: PitchPatchInput) -> list[float]:
        values = item.feature_vector()
        return [values[index] for index in PITCH_VISUAL_FEATURE_INDICES]

    def predict_probability(self, item: PitchPatchInput) -> float:
        return self.model.predict(self.vector(item), neutral=0.5)

    def calibrate(self, item: PitchPatchInput) -> PitchVisualGuardCalibration:
        probability = self.predict_probability(item)
        accepted = bool(
            self.enabled
            and self.model_verified
            and probability >= self.threshold
        )
        return PitchVisualGuardCalibration(
            probability=round(probability, 6),
            threshold=round(self.threshold, 6),
            accepted=accepted,
            model_version=self.model_version,
        )


class PitchPatchCalibrator:
    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).with_name("resources") / "pitch_patch_calibrator.json"
        loaded = load_verified_json(model_path, "pitch_patch_calibration")
        payload = loaded.payload
        self.model = VerifiedRandomForestModel.load(
            model_path,
            "pitch_patch_calibration",
            FEATURE_NAMES,
            loaded=loaded,
        )
        try:
            stored_threshold = float(payload.get("auto_patch_threshold", 1.0))
            stored_no_visual_threshold = float(
                payload.get("no_visual_auto_patch_threshold", stored_threshold)
            )
            target_precision = float(payload.get("target_precision", 1.0))
        except (TypeError, ValueError, OverflowError):
            stored_threshold = 1.0
            stored_no_visual_threshold = 1.0
            target_precision = 1.0
        self.threshold = max(
            float(DEFAULT_POLICY.pitch_patch_probability_floor),
            max(0.0, min(1.0, stored_threshold)),
        )
        self.no_visual_threshold = max(
            float(DEFAULT_POLICY.pitch_patch_no_visual_probability_floor),
            max(0.0, min(1.0, stored_no_visual_threshold)),
        )
        self.target_precision = max(0.0, min(1.0, target_precision))
        self.model_verified = self.model.verified and loaded.verified
        self.model_status = self.model.status if self.model.enabled else loaded.status
        self.model_version = self.model.model_version
        self.enabled = self.model.enabled

    def predict_probability(self, item: PitchPatchInput) -> float:
        return self.model.predict(item.feature_vector(), neutral=0.5)

    def calibrate(self, item: PitchPatchInput) -> PitchPatchCalibration:
        probability = self.predict_probability(item)
        threshold = (
            self.threshold
            if item.visual_evidence_available
            else self.no_visual_threshold
        )
        accepted = bool(
            self.enabled
            and self.model_verified
            and probability >= threshold
        )
        return PitchPatchCalibration(
            probability=round(probability, 6),
            threshold=round(threshold, 6),
            accepted=accepted,
            model_version=self.model_version,
            target_precision=round(self.target_precision, 6),
        )


def _pitch_key(note: NoteIR) -> tuple[str, str, int, str] | None:
    if note.rest or note.pitch is None:
        return None
    pitch: PitchIR = note.pitch
    return (
        pitch.step.upper(),
        f"{pitch.alter.numerator}/{pitch.alter.denominator}",
        int(pitch.octave),
        note.accidental.strip().casefold(),
    )


def _diatonic_value(note: NoteIR) -> int | None:
    if note.pitch is None or note.rest:
        return None
    order = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}
    step = note.pitch.step.upper()
    if step not in order:
        return None
    return note.pitch.octave * 7 + order[step]


def _note_skeleton(note: NoteIR) -> tuple[object, ...]:
    return (
        note.onset,
        note.duration,
        note.voice,
        note.rest,
        note.chord,
        note.grace,
        note.note_type,
        note.dots,
        note.ties,
        note.slurs,
        note.articulations,
        note.ornaments,
        note.tuple_ratio,
    )


def _measure_skeleton(measure: MeasureIR) -> tuple[object, ...]:
    return (
        measure.time_signature,
        measure.key_signature,
        measure.clef,
        measure.barlines,
        tuple(_note_skeleton(note) for note in measure.notes),
    )


def _patch_compatible_note_skeleton(note: NoteIR) -> tuple[object, ...]:
    """Structure which remains meaningful after a prior chord-marker repair.

    Chord membership changes both ``note.chord`` and the derived onset.  A verified
    chord patch may therefore supply a new base measure while pitch evidence still
    comes from the original candidates.  All other event properties must remain exact.
    """
    return (
        note.duration,
        note.voice,
        note.rest,
        note.grace,
        note.note_type,
        note.dots,
        note.ties,
        note.slurs,
        note.articulations,
        note.ornaments,
        note.tuple_ratio,
    )


def _patch_compatible_measure_skeleton(measure: MeasureIR) -> tuple[object, ...]:
    return (
        measure.time_signature,
        measure.key_signature,
        measure.clef,
        measure.barlines,
        tuple(_patch_compatible_note_skeleton(note) for note in measure.notes),
    )


def _replace_accidental(target: etree._Element, source: etree._Element) -> None:
    target_accidental = target.find("accidental")
    source_accidental = source.find("accidental")
    if target_accidental is not None:
        target.remove(target_accidental)
    if source_accidental is None:
        return
    replacement = copy.deepcopy(source_accidental)
    children = list(target)
    insertion = len(children)
    for index, child in enumerate(children):
        if child.tag in {
            "time-modification", "stem", "notehead", "notehead-text", "staff",
            "beam", "notations", "lyric", "play", "listen",
        }:
            insertion = index
            break
    target.insert(insertion, replacement)


def _copy_pitch(target: etree._Element, source: etree._Element) -> bool:
    source_pitch = source.find("pitch")
    target_pitch = target.find("pitch")
    if source_pitch is None or target_pitch is None:
        return False
    target.replace(target_pitch, copy.deepcopy(source_pitch))
    _replace_accidental(target, source)
    return True


def _mean(values: Sequence[float], default: float = 0.5) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else default


def propose_pitch_patch(
    candidates: Sequence[PitchPatchCandidate],
    *,
    template_index: int,
    missing_candidate_count: int,
    calibrator: PitchPatchCalibrator | None = None,
    visual_guard: PitchVisualGuard | None = None,
    accidental_guard: AccidentalPresenceGuard | None = None,
    base_measure: etree._Element | None = None,
    visual_evidence: VisualMeasureEvidence | None = None,
) -> PitchPatchResult:
    """Propose and optionally approve one pitch-only measure repair.

    The function is intentionally fail-closed.  Every family gets at most one vote per
    event; disagreement within a family causes abstention.  Every pitch disagreement in
    the measure must have a strict independent-family winner, otherwise no partial patch
    is emitted.
    """
    if not candidates or template_index < 0 or template_index >= len(candidates):
        return PitchPatchResult(None, (), 0.5, 1.0, False, "invalid_input")
    if missing_candidate_count:
        return PitchPatchResult(None, (), 0.5, 1.0, False, "alignment_gap")

    template = candidates[template_index]
    if not template.valid:
        return PitchPatchResult(None, (), 0.5, 1.0, False, "invalid_template")
    family_members, incomplete_families = group_complete_families(
        candidates,
        family_of=lambda item: item.family,
        valid_of=lambda item: item.valid,
    )
    families = sorted(family_members)
    valid = [item for family in families for item in family_members[family]]
    if len(families) < DEFAULT_POLICY.pitch_patch_minimum_families:
        return PitchPatchResult(None, (), 0.5, 1.0, False, "insufficient_families")

    base_semantics = template.semantics
    template_measure = template.measure
    if base_measure is not None:
        inherited: dict[str, object] = {
            "divisions": template.semantics.divisions,
            "time": template.semantics.time_signature,
            "key": template.semantics.key_signature,
            "clef": template.semantics.clef,
        }
        base_semantics, _base_state = measure_from_xml(base_measure, inherited)
        template_measure = base_measure
        compatible = _patch_compatible_measure_skeleton(base_semantics)
        if any(_patch_compatible_measure_skeleton(item.semantics) != compatible for item in valid):
            return PitchPatchResult(None, (), 0.5, 1.0, False, "non_pitch_structure_disagreement")
    else:
        skeleton = _measure_skeleton(base_semantics)
        if any(_measure_skeleton(item.semantics) != skeleton for item in valid):
            return PitchPatchResult(None, (), 0.5, 1.0, False, "non_pitch_structure_disagreement")
    skeleton = _measure_skeleton(base_semantics)
    if not base_semantics.notes:
        return PitchPatchResult(None, (), 0.5, 1.0, False, "non_pitch_structure_disagreement")
    note_elements_by_variant: dict[str, list[etree._Element]] = {
        item.variant: item.measure.findall("note") for item in valid
    }
    note_count = len(base_semantics.notes)
    if any(len(elements) != note_count for elements in note_elements_by_variant.values()):
        return PitchPatchResult(None, (), 0.5, 1.0, False, "xml_event_count_mismatch")
    pitched_event_indices = tuple(
        index
        for index, note in enumerate(base_semantics.notes)
        if _pitch_key(note) is not None
    )
    if not pitched_event_indices:
        return PitchPatchResult(None, (), 0.5, 1.0, False, "no_pitched_events")
    pitched_index_set = set(pitched_event_indices)
    if any(
        _pitch_key(item.semantics.notes[event_index]) is not None
        for item in valid
        for event_index in range(note_count)
        if event_index not in pitched_index_set
    ):
        return PitchPatchResult(
            None,
            (),
            0.5,
            1.0,
            False,
            "non_pitch_structure_disagreement",
        )
    if any(
        _pitch_key(item.semantics.notes[event_index]) is None
        for item in valid
        for event_index in pitched_event_indices
    ):
        return PitchPatchResult(
            None,
            (),
            0.5,
            1.0,
            False,
            "non_pitch_structure_disagreement",
        )

    changed: list[tuple[int, PitchPatchCandidate]] = []
    support_ratios: list[float] = []
    margin_ratios: list[float] = []
    template_support_ratios: list[float] = []
    pitched_event_count = len(pitched_event_indices)
    abstentions = len(incomplete_families) * pitched_event_count
    possible_family_votes = (
        len(families) + len(incomplete_families)
    ) * pitched_event_count
    minimum_voting_families = len(families)
    supporting_rows: list[PitchPatchCandidate] = []

    for event_index in pitched_event_indices:
        family_votes: dict[str, tuple[tuple[str, str, int, str], PitchPatchCandidate]] = {}
        for family, items in family_members.items():
            keys = {_pitch_key(item.semantics.notes[event_index]) for item in items}
            if len(keys) != 1 or None in keys:
                abstentions += 1
                continue
            key = next(iter(keys))
            representative = max(
                items,
                key=lambda item: (
                    item.ensemble_probability,
                    item.event_probability,
                    item.measure_probability,
                    item.visual_probability,
                    item.page_score,
                    item.variant,
                ),
            )
            family_votes[family] = (key, representative)  # type: ignore[arg-type]

        minimum_voting_families = min(minimum_voting_families, len(family_votes))
        if len(family_votes) < DEFAULT_POLICY.pitch_patch_minimum_families:
            return PitchPatchResult(None, (), 0.5, 1.0, False, "insufficient_event_family_votes")
        grouped: dict[tuple[str, str, int, str], list[tuple[str, PitchPatchCandidate]]] = {}
        for family, (key, representative) in family_votes.items():
            grouped.setdefault(key, []).append((family, representative))
        ranked = sorted(
            grouped.items(),
            key=lambda item: (
                len(item[1]),
                _mean([candidate.ensemble_probability for _, candidate in item[1]]),
                item[0],
            ),
            reverse=True,
        )
        winner_key, winner_rows = ranked[0]
        runner_up = len(ranked[1][1]) if len(ranked) > 1 else 0
        winner_count = len(winner_rows)
        voting_count = len(family_votes)
        template_key = _pitch_key(base_semantics.notes[event_index])
        template_count = len(grouped.get(template_key, ())) if template_key is not None else 0

        disagreement = len(grouped) > 1
        if disagreement and not (
            winner_count >= DEFAULT_POLICY.pitch_patch_minimum_supporting_families
            and winner_count > voting_count / 2
            and winner_count - runner_up >= 1
        ):
            return PitchPatchResult(None, (), 0.5, 1.0, False, "no_strict_event_family_majority")
        # The template's own correlation family may abstain when two siblings
        # disagree (for example adaptive=G-sharp and otsu=G-natural).  In that
        # case every remaining independent family can unanimously support a
        # pitch different from the template, leaving ``grouped`` with one key.
        # "No disagreement among voters" is positive consensus, not evidence
        # that the template is already correct.
        if winner_key == template_key:
            continue

        representative = max(
            (candidate for _, candidate in winner_rows),
            key=lambda item: (
                item.ensemble_probability,
                item.event_probability,
                item.measure_probability,
                item.visual_probability,
                item.page_score,
                item.variant,
            ),
        )
        changed.append((event_index, representative))
        supporting_rows.extend(candidate for _, candidate in winner_rows)
        support_ratios.append(winner_count / voting_count)
        margin_ratios.append((winner_count - runner_up) / voting_count)
        template_support_ratios.append(template_count / voting_count)

    if not changed:
        return PitchPatchResult(None, (), 0.5, 1.0, False, "no_pitch_change")

    unique_support: dict[tuple[int, str], PitchPatchCandidate] = {}
    for event_index, representative in changed:
        for family, items in family_members.items():
            matching = [
                item for item in items
                if _pitch_key(item.semantics.notes[event_index])
                == _pitch_key(representative.semantics.notes[event_index])
            ]
            if matching:
                unique_support[(event_index, family)] = max(
                    matching,
                    key=lambda item: (item.ensemble_probability, item.event_probability, item.page_score),
                )
    supporting_rows = list(unique_support.values())
    if not supporting_rows:
        return PitchPatchResult(None, (), 0.5, 1.0, False, "missing_support_quality")

    # Build and validate the exact XML proposal before asking the learned gate.  This
    # makes direct source-crop comparison possible and prevents the model from scoring
    # a proposal which could not actually be represented in MusicXML.
    patched = copy.deepcopy(template_measure)
    patched_notes = patched.findall("note")
    for event_index, representative in changed:
        source_note = note_elements_by_variant[representative.variant][event_index]
        if not _copy_pitch(patched_notes[event_index], source_note):
            return PitchPatchResult(
                None,
                tuple(index for index, _ in changed),
                0.5,
                1.0,
                False,
                "xml_pitch_copy_failed",
            )

    inherited: dict[str, object] = {
        "divisions": base_semantics.divisions,
        "time": base_semantics.time_signature,
        "key": base_semantics.key_signature,
        "clef": base_semantics.clef,
    }
    parsed, _state = measure_from_xml(patched, inherited)
    if _measure_skeleton(parsed) != skeleton:
        return PitchPatchResult(
            None,
            tuple(index for index, _ in changed),
            0.5,
            1.0,
            False,
            "post_patch_structure_changed",
        )
    if any(
        note.pitch is not None and not (1200 <= note.pitch.midi_cents <= 12000)
        for note in parsed.notes
    ):
        return PitchPatchResult(
            None,
            tuple(index for index, _ in changed),
            0.5,
            1.0,
            False,
            "post_patch_pitch_outlier",
        )

    accidental_state = accidental_regression(base_semantics, parsed)
    if not accidental_state.safe:
        return PitchPatchResult(
            None,
            tuple(index for index, _ in changed),
            0.0,
            1.0,
            False,
            "accidental_state_regression",
        )

    chromatic_indices = chromatic_change_indices(base_semantics, parsed)
    presence_change_indices: list[int] = []
    class_change_indices: list[int] = []
    for event_index in chromatic_indices:
        before_symbol = normalise_accidental(base_semantics.notes[event_index].accidental)
        after_symbol = normalise_accidental(parsed.notes[event_index].accidental)
        before_present = bool(before_symbol)
        after_present = bool(after_symbol)
        if before_present != after_present:
            presence_change_indices.append(event_index)
        elif before_present and before_symbol != after_symbol:
            class_change_indices.append(event_index)

    # A presence classifier cannot safely distinguish sharp, flat and natural.  When
    # a persisted source-symbol crop exists, an explicit-class substitution therefore
    # remains review-only instead of borrowing confidence from the narrower binary model.
    # A genuinely absent crop preserves the pre-existing semantic path; a non-empty but
    # malformed crop is still treated as evidence and fails closed in the guard.
    symbol_evidence_declared = bool(
        visual_evidence is not None and visual_evidence.symbol_guard_image
    )
    if symbol_evidence_declared and class_change_indices:
        return PitchPatchResult(
            None,
            tuple(index for index, _ in changed),
            0.0,
            1.0,
            False,
            "accidental_class_change_review",
        )

    accidental_calibrations = []
    if symbol_evidence_declared and presence_change_indices:
        active_accidental_guard = accidental_guard or AccidentalPresenceGuard()
        for event_index in presence_change_indices:
            expected_present = bool(
                normalise_accidental(parsed.notes[event_index].accidental)
            )
            calibration = active_accidental_guard.calibrate(
                visual_evidence,
                parsed,
                event_index,
                expected_present=expected_present,
            )
            accidental_calibrations.append(calibration)
            if not calibration.accepted:
                return PitchPatchResult(
                    None,
                    tuple(index for index, _ in changed),
                    calibration.confidence,
                    calibration.threshold,
                    False,
                    "accidental_presence_guard",
                    model_version=calibration.model_version,
                    accidental_guard_probability=calibration.probability,
                    accidental_guard_threshold=calibration.threshold,
                    accidental_guard_model_version=calibration.model_version,
                )

    staff_position_deltas: list[int] = []
    accidental_only = 0
    for event_index, _representative in changed:
        before_note = base_semantics.notes[event_index]
        after_note = parsed.notes[event_index]
        before_value = _diatonic_value(before_note)
        after_value = _diatonic_value(after_note)
        if before_value is not None and after_value is not None and before_value != after_value:
            staff_position_deltas.append(abs(after_value - before_value))
        elif (
            before_note.pitch is not None
            and after_note.pitch is not None
            and before_note.pitch.step.upper() == after_note.pitch.step.upper()
            and before_note.pitch.octave == after_note.pitch.octave
            and (
                before_note.pitch.alter != after_note.pitch.alter
                or before_note.accidental.strip().casefold()
                != after_note.accidental.strip().casefold()
            )
        ):
            accidental_only += 1

    visual_available = False
    if visual_evidence is not None:
        visual_source_grid = visual_evidence.pitch_guard_notehead_grid
        if not any(float(value) >= 0.03 for value in visual_source_grid):
            visual_source_grid = visual_evidence.pitched_notehead_grid
        visual_available = any(float(value) >= 0.03 for value in visual_source_grid)
    before_gaps, after_gaps = pitch_transaction_gap_pair(
        visual_evidence,
        base_semantics,
        parsed,
        tuple(index for index, _representative in changed),
    )
    visual_improvements = tuple(
        max(-1.0, min(1.0, before - after))
        for before, after in zip(before_gaps, after_gaps, strict=True)
    )
    strict_before_gaps, strict_after_gaps = pitch_transaction_gap_pair(
        visual_evidence,
        base_semantics,
        parsed,
        tuple(index for index, _representative in changed),
        strict=True,
    )
    strict_visual_improvements = tuple(
        max(-1.0, min(1.0, before - after))
        for before, after in zip(strict_before_gaps, strict_after_gaps, strict=True)
    )

    # A strong two-feature contradiction is handled deterministically.  The threshold
    # is deliberately wide: ordinary detector noise or a neutral crop reaches the
    # learned veto, while a proposal which simultaneously worsens near-cell support
    # and severe vertical misses is never auto-applied.
    if (
        visual_available
        and staff_position_deltas
        and (
            (
                visual_improvements[1] <= -0.12
                and visual_improvements[3] <= -0.08
            )
            or (
                strict_visual_improvements[1] <= -0.12
                and strict_visual_improvements[3] <= -0.08
            )
        )
    ):
        return PitchPatchResult(
            None,
            tuple(index for index, _ in changed),
            0.0,
            1.0,
            False,
            "visual_pitch_conflict",
        )

    item = PitchPatchInput(
        candidate_count=len(valid),
        eligible_family_count=len(families),
        voting_family_count=minimum_voting_families,
        changed_event_count=len(changed),
        total_event_count=pitched_event_count,
        minimum_winner_family_support_ratio=min(support_ratios),
        mean_winner_family_support_ratio=_mean(support_ratios, 0.0),
        minimum_winner_margin_ratio=min(margin_ratios),
        mean_winner_margin_ratio=_mean(margin_ratios, 0.0),
        maximum_template_family_support_ratio=max(template_support_ratios),
        family_abstention_ratio=abstentions / max(possible_family_votes, 1),
        mean_support_page_probability=_mean([row.page_probability for row in supporting_rows]),
        mean_support_measure_probability=_mean([row.measure_probability for row in supporting_rows]),
        mean_support_visual_probability=_mean([row.visual_probability for row in supporting_rows]),
        mean_support_event_probability=_mean([row.event_probability for row in supporting_rows]),
        mean_support_context_probability=_mean([row.context_probability for row in supporting_rows]),
        mean_support_ensemble_probability=_mean([row.ensemble_probability for row in supporting_rows]),
        minimum_support_ensemble_probability=min(row.ensemble_probability for row in supporting_rows),
        mean_support_page_score_margin=_mean([row.page_score - template.page_score for row in supporting_rows], 0.0),
        mean_support_vs_template_measure_probability=_mean([
            row.measure_probability - template.measure_probability for row in supporting_rows
        ], 0.0),
        mean_support_vs_template_visual_probability=_mean([
            row.visual_probability - template.visual_probability for row in supporting_rows
        ], 0.0),
        mean_support_vs_template_event_probability=_mean([
            row.event_probability - template.event_probability for row in supporting_rows
        ], 0.0),
        mean_support_vs_template_context_probability=_mean([
            row.context_probability - template.context_probability for row in supporting_rows
        ], 0.0),
        mean_support_vs_template_ensemble_probability=_mean([
            row.ensemble_probability - template.ensemble_probability for row in supporting_rows
        ], 0.0),
        visual_evidence_available=visual_available,
        changed_staff_position_ratio=len(staff_position_deltas) / max(len(changed), 1),
        maximum_staff_position_delta=max(staff_position_deltas, default=0),
        accidental_only_change_ratio=accidental_only / max(len(changed), 1),
        notehead_exact_cell_improvement=visual_improvements[0],
        notehead_near_cell_improvement=visual_improvements[1],
        notehead_vertical_chamfer_improvement=visual_improvements[2],
        notehead_severe_vertical_improvement=visual_improvements[3],
        notehead_visual_unmatched_improvement=visual_improvements[4],
        notehead_column_centroid_improvement=visual_improvements[5],
        notehead_column_order_improvement=visual_improvements[6],
        template_notehead_exact_cell_gap=before_gaps[0],
        template_notehead_near_cell_gap=before_gaps[1],
        template_notehead_vertical_chamfer_gap=before_gaps[2],
        template_notehead_severe_vertical_gap=before_gaps[3],
        template_notehead_visual_unmatched_gap=before_gaps[4],
        template_notehead_column_centroid_gap=before_gaps[5],
        template_notehead_column_order_gap=before_gaps[6],
        proposal_notehead_exact_cell_gap=after_gaps[0],
        proposal_notehead_near_cell_gap=after_gaps[1],
        proposal_notehead_vertical_chamfer_gap=after_gaps[2],
        proposal_notehead_severe_vertical_gap=after_gaps[3],
        proposal_notehead_visual_unmatched_gap=after_gaps[4],
        proposal_notehead_column_centroid_gap=after_gaps[5],
        proposal_notehead_column_order_gap=after_gaps[6],
        strict_notehead_exact_cell_improvement=strict_visual_improvements[0],
        strict_notehead_near_cell_improvement=strict_visual_improvements[1],
        strict_notehead_vertical_chamfer_improvement=strict_visual_improvements[2],
        strict_notehead_severe_vertical_improvement=strict_visual_improvements[3],
        strict_notehead_visual_unmatched_improvement=strict_visual_improvements[4],
        strict_notehead_column_centroid_improvement=strict_visual_improvements[5],
        strict_notehead_column_order_improvement=strict_visual_improvements[6],
        template_strict_notehead_exact_cell_gap=strict_before_gaps[0],
        template_strict_notehead_near_cell_gap=strict_before_gaps[1],
        template_strict_notehead_vertical_chamfer_gap=strict_before_gaps[2],
        template_strict_notehead_severe_vertical_gap=strict_before_gaps[3],
        template_strict_notehead_visual_unmatched_gap=strict_before_gaps[4],
        template_strict_notehead_column_centroid_gap=strict_before_gaps[5],
        template_strict_notehead_column_order_gap=strict_before_gaps[6],
        proposal_strict_notehead_exact_cell_gap=strict_after_gaps[0],
        proposal_strict_notehead_near_cell_gap=strict_after_gaps[1],
        proposal_strict_notehead_vertical_chamfer_gap=strict_after_gaps[2],
        proposal_strict_notehead_severe_vertical_gap=strict_after_gaps[3],
        proposal_strict_notehead_visual_unmatched_gap=strict_after_gaps[4],
        proposal_strict_notehead_column_centroid_gap=strict_after_gaps[5],
        proposal_strict_notehead_column_order_gap=strict_after_gaps[6],
    )
    changed_index_set = {index for index, _representative in changed}
    resolved_implicit_indices = {
        issue.event_index
        for issue in accidental_state.resolved
        if issue.code == "implicit_alter_state_mismatch"
    }
    deterministic_implicit_state_repair = bool(
        changed_index_set
        and changed_index_set == set(chromatic_indices)
        and changed_index_set.issubset(resolved_implicit_indices)
        and accidental_only == len(changed)
        and not presence_change_indices
        and not class_change_indices
        and minimum_voting_families >= 3
        and min(support_ratios) >= 1.0
        and max(template_support_ratios) <= 0.0
    )
    if deterministic_implicit_state_repair:
        return PitchPatchResult(
            patched,
            tuple(index for index, _ in changed),
            1.0,
            1.0,
            True,
            "accepted_implicit_state_repair",
            item,
            model_version="deterministic-implicit-accidental-state@1",
        )

    visual_calibration: PitchVisualGuardCalibration | None = None
    if visual_available and staff_position_deltas:
        active_visual_guard = visual_guard or PitchVisualGuard()
        visual_calibration = active_visual_guard.calibrate(item)
        if not visual_calibration.accepted:
            return PitchPatchResult(
                None,
                tuple(index for index, _ in changed),
                visual_calibration.probability,
                visual_calibration.threshold,
                False,
                "visual_model_guard",
                item,
                model_version=visual_calibration.model_version,
                visual_guard_probability=visual_calibration.probability,
                visual_guard_threshold=visual_calibration.threshold,
                visual_guard_model_version=visual_calibration.model_version,
            )

    active_calibrator = calibrator or PitchPatchCalibrator()
    calibration = active_calibrator.calibrate(item)
    if not calibration.accepted:
        return PitchPatchResult(
            None,
            tuple(index for index, _ in changed),
            calibration.probability,
            calibration.threshold,
            False,
            "model_guard",
            item,
            model_version=calibration.model_version,
            visual_guard_probability=(
                visual_calibration.probability if visual_calibration is not None else 0.5
            ),
            visual_guard_threshold=(
                visual_calibration.threshold if visual_calibration is not None else 1.0
            ),
            visual_guard_model_version=(
                visual_calibration.model_version if visual_calibration is not None else "not_applicable"
            ),
        )

    return PitchPatchResult(
        patched,
        tuple(index for index, _ in changed),
        calibration.probability,
        calibration.threshold,
        True,
        "accepted",
        item,
        model_version=calibration.model_version,
        visual_guard_probability=(
            visual_calibration.probability if visual_calibration is not None else 0.5
        ),
        visual_guard_threshold=(
            visual_calibration.threshold if visual_calibration is not None else 1.0
        ),
        visual_guard_model_version=(
            visual_calibration.model_version if visual_calibration is not None else "not_applicable"
        ),
        accidental_guard_probability=(
            min(item.probability for item in accidental_calibrations)
            if accidental_calibrations else 0.5
        ),
        accidental_guard_threshold=(
            max(item.threshold for item in accidental_calibrations)
            if accidental_calibrations else 1.0
        ),
        accidental_guard_model_version=(
            accidental_calibrations[0].model_version
            if accidental_calibrations else "not_applicable"
        ),
    )
