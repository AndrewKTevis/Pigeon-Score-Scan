from __future__ import annotations

"""Conservative event-level rhythm consensus for complementary OMR errors.

The whole-measure ensemble can fail when every preprocessing family contains a
different isolated duration/type error.  This module repairs only the rhythm-bearing
children of existing notes and only inside a deliberately narrow monophonic subset.
It never inserts, deletes, reorders, or repitches an event.  Correlated preprocessing
siblings are collapsed to one family vote, split siblings abstain, deterministic
MusicXML/metrical checks remain authoritative, and a verified CPU model may only veto
an otherwise valid proposal.
"""

import copy
import math
import statistics
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence

from lxml import etree

from .model_registry import load_verified_json
from .policy import DEFAULT_POLICY
from .rhythm_symbol_guard import (
    RhythmSymbolGuard,
    build_rhythm_symbol_transaction,
)
from .score_ir import MeasureIR, NoteIR, ScoreIR, audit_score, expected_note_duration, measure_from_xml
from .tree_model import VerifiedRandomForestModel
from .visual_evidence import VisualMeasureEvidence
from .variant_family import group_complete_families

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
    "minimum_pitch_coherence_ratio",
    "mean_pitch_coherence_ratio",
    "template_duration_error_scaled",
    "patched_duration_error_scaled",
    "duration_error_improvement_scaled",
    "template_type_mismatch_ratio",
    "patched_type_mismatch_ratio",
    "mean_support_page_probability",
    "mean_support_measure_probability",
    "mean_support_visual_probability",
    "mean_support_event_probability",
    "mean_support_context_probability",
    "mean_support_ensemble_probability",
    "minimum_support_ensemble_probability",
    "mean_support_page_score_margin_scaled",
    "mean_support_vs_template_ensemble_probability",
)


@dataclass(frozen=True)
class RhythmPatchCandidate:
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
class RhythmPatchInput:
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
    minimum_pitch_coherence_ratio: float
    mean_pitch_coherence_ratio: float
    template_duration_error: float
    patched_duration_error: float
    duration_error_improvement: float
    template_type_mismatch_ratio: float
    patched_type_mismatch_ratio: float
    mean_support_page_probability: float
    mean_support_measure_probability: float
    mean_support_visual_probability: float
    mean_support_event_probability: float
    mean_support_context_probability: float
    mean_support_ensemble_probability: float
    minimum_support_ensemble_probability: float
    mean_support_page_score_margin: float
    mean_support_vs_template_ensemble_probability: float

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
            unit(self.minimum_pitch_coherence_ratio),
            unit(self.mean_pitch_coherence_ratio),
            unit(self.template_duration_error / 4.0),
            unit(self.patched_duration_error / 4.0),
            signed(self.duration_error_improvement / 4.0),
            unit(self.template_type_mismatch_ratio),
            unit(self.patched_type_mismatch_ratio),
            unit(self.mean_support_page_probability),
            unit(self.mean_support_measure_probability),
            unit(self.mean_support_visual_probability),
            unit(self.mean_support_event_probability),
            unit(self.mean_support_context_probability),
            unit(self.mean_support_ensemble_probability),
            unit(self.minimum_support_ensemble_probability),
            signed(self.mean_support_page_score_margin / 100.0),
            signed(self.mean_support_vs_template_ensemble_probability),
        ]


@dataclass(frozen=True)
class RhythmPatchCalibration:
    probability: float
    threshold: float
    accepted: bool
    model_version: str
    target_precision: float


@dataclass(frozen=True)
class RhythmPatchResult:
    patched_measure: etree._Element | None
    changed_event_indices: tuple[int, ...]
    probability: float
    threshold: float
    accepted: bool
    reason: str
    input: RhythmPatchInput | None = None
    model_version: str = "disabled"
    symbol_guard_confidence: float = 0.5
    symbol_guard_threshold: float = 1.0
    symbol_guard_model_version: str = "not_applicable"


class RhythmPatchCalibrator:
    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).with_name("resources") / "rhythm_patch_calibrator.json"
        loaded = load_verified_json(model_path, "rhythm_patch_calibration")
        payload = loaded.payload
        self.model = VerifiedRandomForestModel.load(
            model_path,
            "rhythm_patch_calibration",
            FEATURE_NAMES,
            loaded=loaded,
        )
        try:
            stored_threshold = float(payload.get("auto_patch_threshold", 1.0))
            target_precision = float(payload.get("target_precision", 1.0))
        except (TypeError, ValueError, OverflowError):
            stored_threshold = 1.0
            target_precision = 1.0
        self.threshold = max(
            float(DEFAULT_POLICY.rhythm_patch_probability_floor),
            max(0.0, min(1.0, stored_threshold)),
        )
        self.target_precision = max(0.0, min(1.0, target_precision))
        self.model_verified = self.model.verified and loaded.verified
        self.model_status = self.model.status if self.model.enabled else loaded.status
        self.model_version = self.model.model_version
        self.enabled = self.model.enabled

    def predict_probability(self, item: RhythmPatchInput) -> float:
        return self.model.predict(item.feature_vector(), neutral=0.5)

    def calibrate(self, item: RhythmPatchInput) -> RhythmPatchCalibration:
        probability = self.predict_probability(item)
        accepted = bool(self.enabled and self.model_verified and probability >= self.threshold)
        return RhythmPatchCalibration(
            probability=round(probability, 6),
            threshold=round(self.threshold, 6),
            accepted=accepted,
            model_version=self.model_version,
            target_precision=round(self.target_precision, 6),
        )


def _mean(values: Sequence[float], default: float = 0.5) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else default


def _fraction_key(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _pitch_key(note: NoteIR) -> tuple[object, ...]:
    if note.rest:
        return ("rest",)
    if note.pitch is None:
        return ("unpitched",)
    return (
        note.pitch.step.upper(),
        _fraction_key(note.pitch.alter),
        int(note.pitch.octave),
        note.accidental.strip().casefold(),
    )


def _rhythm_key(note: NoteIR) -> tuple[str, str, int]:
    return (_fraction_key(note.duration), note.note_type.strip().casefold(), int(note.dots))


def _note_identity(note: NoteIR) -> tuple[object, ...]:
    return (
        note.voice,
        note.rest,
        note.chord,
        note.grace,
        note.ties,
        note.slurs,
        note.articulations,
        note.ornaments,
        note.tuple_ratio,
    )


def _measure_identity(measure: MeasureIR) -> tuple[object, ...]:
    return (
        measure.time_signature,
        measure.key_signature,
        measure.clef,
        tuple(direction.stable_tuple() for direction in measure.directions),
        measure.barlines,
        tuple(_note_identity(note) for note in measure.notes),
    )


def _duration_total(measure: MeasureIR) -> Fraction:
    return sum(
        (note.duration for note in measure.notes if not note.chord and not note.grace),
        Fraction(0, 1),
    )


def _duration_error(measure: MeasureIR) -> float:
    expected = measure.expected_duration
    if expected is None:
        return 4.0
    return min(4.0, float(abs(_duration_total(measure) - expected)))


def _type_mismatch_ratio(measure: MeasureIR) -> float:
    regular = [note for note in measure.notes if not note.grace]
    if not regular:
        return 1.0
    mismatches = 0
    for note in regular:
        expected = expected_note_duration(note)
        if expected is None or note.duration <= 0 or expected != note.duration:
            mismatches += 1
    return mismatches / len(regular)


_NOTE_CHILD_ORDER = {
    "grace": 0,
    "cue": 1,
    "chord": 2,
    "pitch": 3,
    "unpitched": 3,
    "rest": 3,
    "duration": 4,
    "tie": 5,
    "instrument": 6,
    "footnote": 7,
    "level": 8,
    "voice": 9,
    "type": 10,
    "dot": 11,
    "accidental": 12,
    "time-modification": 13,
    "stem": 14,
    "notehead": 15,
    "notehead-text": 16,
    "staff": 17,
    "beam": 18,
    "notations": 19,
    "lyric": 20,
    "play": 21,
    "listen": 22,
}
_RHYTHM_TAGS = {"duration", "type", "dot"}
_BEAMABLE_TYPES = {"eighth", "16th", "32nd", "64th", "128th", "256th", "512th", "1024th"}


def _copy_rhythm(
    target: etree._Element,
    source: etree._Element,
    *,
    semantic_duration: Fraction,
    target_divisions: int,
) -> bool:
    """Copy voted rhythm without importing the source candidate's time base.

    MusicXML ``duration`` is measured in the surrounding ``divisions`` value, so its
    raw integer is not portable between candidates.  Re-encode the semantic duration
    in the template time base, copy only the voted type/dots, and preserve beaming
    unless the new type cannot legally carry a beam.  Tuplets are rejected before this
    point, so time-modification is never synthesized or copied.
    """
    if target_divisions <= 0 or semantic_duration <= 0:
        return False
    encoded_duration = semantic_duration * target_divisions
    if encoded_duration.denominator != 1 or encoded_duration.numerator <= 0:
        return False
    if source.find("time-modification") is not None or target.find("time-modification") is not None:
        return False
    source_type = source.find("type")
    source_duration = source.find("duration")
    if source_type is None or source_duration is None or not (source_type.text or "").strip():
        return False

    replacements: list[etree._Element] = []
    duration = copy.deepcopy(source_duration)
    duration.text = str(encoded_duration.numerator)
    replacements.append(duration)
    replacements.append(copy.deepcopy(source_type))
    replacements.extend(copy.deepcopy(child) for child in source.findall("dot"))

    for child in list(target):
        if child.tag in _RHYTHM_TAGS:
            target.remove(child)
    for replacement in replacements:
        rank = _NOTE_CHILD_ORDER.get(replacement.tag, 99)
        insertion = len(target)
        for index, child in enumerate(target):
            if _NOTE_CHILD_ORDER.get(child.tag, 99) > rank:
                insertion = index
                break
        target.insert(insertion, replacement)

    if (source_type.text or "").strip().casefold() not in _BEAMABLE_TYPES:
        for beam in list(target.findall("beam")):
            target.remove(beam)
    return True


def _pitch_coherence(
    family_members: dict[str, list[RhythmPatchCandidate]],
    event_index: int,
) -> float:
    votes: list[tuple[object, ...]] = []
    for items in family_members.values():
        keys = {_pitch_key(item.semantics.notes[event_index]) for item in items}
        if len(keys) == 1:
            votes.append(next(iter(keys)))
    if not votes:
        return 0.0
    counts: dict[tuple[object, ...], int] = {}
    for key in votes:
        counts[key] = counts.get(key, 0) + 1
    return max(counts.values()) / len(votes)


def propose_rhythm_patch(
    candidates: Sequence[RhythmPatchCandidate],
    *,
    template_index: int,
    missing_candidate_count: int,
    calibrator: RhythmPatchCalibrator | None = None,
    base_measure: etree._Element | None = None,
    visual_evidence: VisualMeasureEvidence | None = None,
    symbol_guard: RhythmSymbolGuard | None = None,
) -> RhythmPatchResult:
    """Propose and optionally approve one rhythm-only measure repair.

    The supported subset is intentionally narrow: one voice, no chords, no grace notes,
    no tuplets, no backup/forward elements, and a known time signature.  Every disputed
    event must have a strict majority among at least three independent families.  The
    complete winning sequence must exactly fill the measure and have internally
    consistent MusicXML duration/type notation.
    """
    if not candidates or template_index < 0 or template_index >= len(candidates):
        return RhythmPatchResult(None, (), 0.5, 1.0, False, "invalid_input")
    if missing_candidate_count:
        return RhythmPatchResult(None, (), 0.5, 1.0, False, "alignment_gap")

    template = candidates[template_index]
    if not template.valid:
        return RhythmPatchResult(None, (), 0.5, 1.0, False, "invalid_template")
    family_members, incomplete_families = group_complete_families(
        candidates,
        family_of=lambda item: item.family,
        valid_of=lambda item: item.valid,
    )
    families = sorted(family_members)
    valid = [item for family in families for item in family_members[family]]
    if len(families) < DEFAULT_POLICY.rhythm_patch_minimum_families:
        return RhythmPatchResult(None, (), 0.5, 1.0, False, "insufficient_families")
    if template.semantics.expected_duration is None:
        return RhythmPatchResult(None, (), 0.5, 1.0, False, "missing_time_signature")

    identity = _measure_identity(template.semantics)
    if not template.semantics.notes or any(_measure_identity(item.semantics) != identity for item in valid):
        return RhythmPatchResult(None, (), 0.5, 1.0, False, "non_rhythm_structure_disagreement")
    if any(
        item.semantics.voice_count != 1
        or any(note.chord or note.grace or note.tuple_ratio is not None for note in item.semantics.notes)
        or item.measure.find("backup") is not None
        or item.measure.find("forward") is not None
        or item.measure.find(".//time-modification") is not None
        for item in valid
    ):
        return RhythmPatchResult(None, (), 0.5, 1.0, False, "unsupported_rhythm_structure")

    note_elements_by_variant = {item.variant: item.measure.findall("note") for item in valid}
    note_count = len(template.semantics.notes)
    if any(len(elements) != note_count for elements in note_elements_by_variant.values()):
        return RhythmPatchResult(None, (), 0.5, 1.0, False, "xml_event_count_mismatch")

    changed: list[tuple[int, RhythmPatchCandidate, tuple[str, str, int]]] = []
    support_ratios: list[float] = []
    margin_ratios: list[float] = []
    template_support_ratios: list[float] = []
    pitch_coherences: list[float] = []
    abstentions = len(incomplete_families) * note_count
    possible_family_votes = (len(families) + len(incomplete_families)) * note_count
    minimum_voting_families = len(families)
    supporting_rows: dict[tuple[int, str], RhythmPatchCandidate] = {}
    winning_keys: list[tuple[str, str, int]] = []

    for event_index in range(note_count):
        family_votes: dict[str, tuple[tuple[str, str, int], RhythmPatchCandidate]] = {}
        for family, items in family_members.items():
            keys = {_rhythm_key(item.semantics.notes[event_index]) for item in items}
            if len(keys) != 1:
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
            family_votes[family] = (key, representative)
        minimum_voting_families = min(minimum_voting_families, len(family_votes))
        if len(family_votes) < DEFAULT_POLICY.rhythm_patch_minimum_families:
            return RhythmPatchResult(None, (), 0.5, 1.0, False, "insufficient_event_family_votes")

        grouped: dict[tuple[str, str, int], list[tuple[str, RhythmPatchCandidate]]] = {}
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
        template_key = _rhythm_key(template.semantics.notes[event_index])
        template_count = len(grouped.get(template_key, ()))
        disagreement = len(grouped) > 1
        if disagreement and not (
            winner_count >= DEFAULT_POLICY.rhythm_patch_minimum_supporting_families
            and winner_count > voting_count / 2
            and winner_count - runner_up >= 1
        ):
            return RhythmPatchResult(None, (), 0.5, 1.0, False, "no_strict_event_family_majority")

        coherence = _pitch_coherence(family_members, event_index)
        pitch_coherences.append(coherence)
        if coherence < DEFAULT_POLICY.rhythm_patch_pitch_coherence_floor:
            return RhythmPatchResult(None, (), 0.5, 1.0, False, "insufficient_pitch_coherence")

        winning_keys.append(winner_key)
        if not disagreement or winner_key == template_key:
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
        changed.append((event_index, representative, winner_key))
        support_ratios.append(winner_count / voting_count)
        margin_ratios.append((winner_count - runner_up) / voting_count)
        template_support_ratios.append(template_count / voting_count)
        for family, candidate in winner_rows:
            supporting_rows[(event_index, family)] = candidate

    if not changed:
        return RhythmPatchResult(None, (), 0.5, 1.0, False, "no_rhythm_change")

    expected = template.semantics.expected_duration
    assert expected is not None
    winning_total = sum((Fraction(key[0]) for key in winning_keys), Fraction(0, 1))
    if winning_total != expected:
        return RhythmPatchResult(None, tuple(index for index, _, _ in changed), 0.5, 1.0, False, "winning_sequence_not_meter_complete")

    support = list(supporting_rows.values())
    if not support:
        return RhythmPatchResult(None, (), 0.5, 1.0, False, "missing_support_quality")

    template_duration_error = _duration_error(template.semantics)
    provisional = copy.deepcopy(base_measure if base_measure is not None else template.measure)
    provisional_notes = provisional.findall("note")
    for event_index, representative, _winner_key in changed:
        if not _copy_rhythm(
            provisional_notes[event_index],
            note_elements_by_variant[representative.variant][event_index],
            semantic_duration=Fraction(_winner_key[0]),
            target_divisions=template.semantics.divisions,
        ):
            return RhythmPatchResult(None, tuple(index for index, _, _ in changed), 0.5, 1.0, False, "xml_rhythm_copy_failed")

    inherited: dict[str, object] = {
        "divisions": template.semantics.divisions,
        "time": template.semantics.time_signature,
        "key": template.semantics.key_signature,
        "clef": template.semantics.clef,
    }
    parsed, _state = measure_from_xml(provisional, inherited)
    if _measure_identity(parsed) != identity:
        return RhythmPatchResult(None, tuple(index for index, _, _ in changed), 0.5, 1.0, False, "post_patch_structure_changed")
    if tuple(_rhythm_key(note) for note in parsed.notes) != tuple(winning_keys):
        return RhythmPatchResult(None, tuple(index for index, _, _ in changed), 0.5, 1.0, False, "post_patch_rhythm_mismatch")
    patched_duration_error = _duration_error(parsed)
    patched_type_mismatch = _type_mismatch_ratio(parsed)
    forbidden_issues = {
        issue.code for issue in audit_score(ScoreIR((parsed,)))
        if issue.code in {"multiple_voices", "zero_duration", "type_duration_mismatch", "chord_duration_mismatch"}
    }
    if patched_duration_error != 0.0 or patched_type_mismatch != 0.0 or forbidden_issues:
        return RhythmPatchResult(None, tuple(index for index, _, _ in changed), 0.5, 1.0, False, "post_patch_rhythm_validation_failed")

    item = RhythmPatchInput(
        candidate_count=len(valid),
        eligible_family_count=len(families),
        voting_family_count=minimum_voting_families,
        changed_event_count=len(changed),
        total_event_count=note_count,
        minimum_winner_family_support_ratio=min(support_ratios),
        mean_winner_family_support_ratio=_mean(support_ratios, 0.0),
        minimum_winner_margin_ratio=min(margin_ratios),
        mean_winner_margin_ratio=_mean(margin_ratios, 0.0),
        maximum_template_family_support_ratio=max(template_support_ratios),
        family_abstention_ratio=abstentions / max(possible_family_votes, 1),
        minimum_pitch_coherence_ratio=min(pitch_coherences),
        mean_pitch_coherence_ratio=_mean(pitch_coherences, 0.0),
        template_duration_error=template_duration_error,
        patched_duration_error=patched_duration_error,
        duration_error_improvement=template_duration_error - patched_duration_error,
        template_type_mismatch_ratio=_type_mismatch_ratio(template.semantics),
        patched_type_mismatch_ratio=patched_type_mismatch,
        mean_support_page_probability=_mean([row.page_probability for row in support]),
        mean_support_measure_probability=_mean([row.measure_probability for row in support]),
        mean_support_visual_probability=_mean([row.visual_probability for row in support]),
        mean_support_event_probability=_mean([row.event_probability for row in support]),
        mean_support_context_probability=_mean([row.context_probability for row in support]),
        mean_support_ensemble_probability=_mean([row.ensemble_probability for row in support]),
        minimum_support_ensemble_probability=min(row.ensemble_probability for row in support),
        mean_support_page_score_margin=_mean([row.page_score - template.page_score for row in support], 0.0),
        mean_support_vs_template_ensemble_probability=_mean([
            row.ensemble_probability - template.ensemble_probability for row in support
        ], 0.0),
    )
    active_calibrator = calibrator or RhythmPatchCalibrator()
    calibration = active_calibrator.calibrate(item)
    if not calibration.accepted:
        return RhythmPatchResult(
            None,
            tuple(index for index, _, _ in changed),
            calibration.probability,
            calibration.threshold,
            False,
            "model_guard",
            item,
            model_version=calibration.model_version,
        )
    changed_indices = tuple(index for index, _, _ in changed)
    symbol_confidence = 0.5
    symbol_threshold = 1.0
    symbol_model_version = "not_applicable"
    if visual_evidence is not None and visual_evidence.rhythm_guard_image:
        transaction = build_rhythm_symbol_transaction(
            visual_evidence, parsed, template.semantics, changed_indices
        )
        if transaction is None:
            return RhythmPatchResult(
                None,
                changed_indices,
                calibration.probability,
                calibration.threshold,
                False,
                "rhythm_symbol_evidence_invalid",
                item,
                model_version=calibration.model_version,
            )
        active_symbol_guard = symbol_guard or RhythmSymbolGuard()
        symbol_calibration = active_symbol_guard.calibrate(transaction)
        symbol_confidence = symbol_calibration.confidence
        symbol_threshold = symbol_calibration.threshold
        symbol_model_version = symbol_calibration.model_version
        if not symbol_calibration.accepted:
            return RhythmPatchResult(
                None,
                changed_indices,
                calibration.probability,
                calibration.threshold,
                False,
                "rhythm_symbol_guard",
                item,
                model_version=calibration.model_version,
                symbol_guard_confidence=symbol_confidence,
                symbol_guard_threshold=symbol_threshold,
                symbol_guard_model_version=symbol_model_version,
            )
    return RhythmPatchResult(
        provisional,
        changed_indices,
        calibration.probability,
        calibration.threshold,
        True,
        "accepted",
        item,
        model_version=calibration.model_version,
        symbol_guard_confidence=symbol_confidence,
        symbol_guard_threshold=symbol_threshold,
        symbol_guard_model_version=symbol_model_version,
    )
