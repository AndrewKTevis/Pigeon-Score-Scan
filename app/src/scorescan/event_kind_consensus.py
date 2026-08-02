from __future__ import annotations

"""Conservative rest-versus-pitched-note consensus.

OMR candidates occasionally agree on the complete rhythmic event skeleton but disagree
whether one event is a rest or a pitched note.  Whole-measure voting and pitch-only
repair cannot safely recover this case.  This module repairs only the existing event
kind and, for a pitched winner, its already-observed pitch/accidental XML.

The supported subset is intentionally narrow: one voice, fixed event count/order, no
chords, grace notes, tuplets, backup/forward cursor operations, or unpitched notes.  Each
preprocessing family receives one vote; invalid or split siblings abstain.  Every
rest/pitch disagreement must have a strict majority of at least three independent
families.  The CPU model is veto-only and post-patch semantic validation is authoritative.
"""

import copy
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from lxml import etree

from .model_registry import load_verified_json
from .policy import DEFAULT_POLICY
from .score_ir import MeasureIR, NoteIR, PitchIR, measure_from_xml
from .tree_model import VerifiedRandomForestModel
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
    "pitched_winner_ratio",
    "rest_winner_ratio",
    "minimum_pitched_winner_support_ratio",
    "mean_pitched_winner_support_ratio",
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
)


@dataclass(frozen=True)
class EventKindPatchCandidate:
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
class EventKindPatchInput:
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
    pitched_winner_count: int
    rest_winner_count: int
    minimum_pitched_winner_support_ratio: float
    mean_pitched_winner_support_ratio: float
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

    def feature_vector(self) -> list[float]:
        def unit(value: float) -> float:
            return max(0.0, min(1.0, float(value)))

        def signed(value: float) -> float:
            return max(-1.0, min(1.0, float(value)))

        changed = max(1, self.changed_event_count)
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
            unit(self.pitched_winner_count / changed),
            unit(self.rest_winner_count / changed),
            unit(self.minimum_pitched_winner_support_ratio),
            unit(self.mean_pitched_winner_support_ratio),
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
        ]


@dataclass(frozen=True)
class EventKindPatchCalibration:
    probability: float
    threshold: float
    accepted: bool
    model_version: str
    target_precision: float


@dataclass(frozen=True)
class EventKindPatchResult:
    patched_measure: etree._Element | None
    changed_event_indices: tuple[int, ...]
    probability: float
    threshold: float
    accepted: bool
    reason: str
    input: EventKindPatchInput | None = None
    model_version: str = "disabled"


class EventKindPatchCalibrator:
    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).with_name("resources") / "event_kind_patch_calibrator.json"
        loaded = load_verified_json(model_path, "event_kind_patch_calibration")
        payload = loaded.payload
        self.model = VerifiedRandomForestModel.load(
            model_path,
            "event_kind_patch_calibration",
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
            float(DEFAULT_POLICY.event_kind_patch_probability_floor),
            max(0.0, min(1.0, stored_threshold)),
        )
        self.target_precision = max(0.0, min(1.0, target_precision))
        self.model_verified = self.model.verified and loaded.verified
        self.model_status = self.model.status if self.model.enabled else loaded.status
        self.model_version = self.model.model_version
        self.enabled = self.model.enabled

    def predict_probability(self, item: EventKindPatchInput) -> float:
        return self.model.predict(item.feature_vector(), neutral=0.5)

    def calibrate(self, item: EventKindPatchInput) -> EventKindPatchCalibration:
        probability = self.predict_probability(item)
        accepted = bool(self.enabled and self.model_verified and probability >= self.threshold)
        return EventKindPatchCalibration(
            probability=round(probability, 6),
            threshold=round(self.threshold, 6),
            accepted=accepted,
            model_version=self.model_version,
            target_precision=round(self.target_precision, 6),
        )


def _mean(values: Sequence[float], default: float = 0.5) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else default


def _pitch_key(pitch: PitchIR, accidental: str) -> tuple[str, str, int, str]:
    return (
        pitch.step.upper(),
        f"{pitch.alter.numerator}/{pitch.alter.denominator}",
        int(pitch.octave),
        accidental.strip().casefold(),
    )


def _event_key(note: NoteIR) -> tuple[object, ...] | None:
    if note.rest:
        return ("rest",)
    if note.pitch is None:
        return None
    return ("pitch",) + _pitch_key(note.pitch, note.accidental)


def _note_skeleton(note: NoteIR) -> tuple[object, ...]:
    return (
        note.onset,
        note.duration,
        note.voice,
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
        measure.divisions,
        measure.time_signature,
        measure.key_signature,
        measure.clef,
        tuple(direction.stable_tuple() for direction in measure.directions),
        measure.barlines,
        tuple(_note_skeleton(note) for note in measure.notes),
    )


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


def _insert_ordered(note: etree._Element, child: etree._Element) -> None:
    rank = _NOTE_CHILD_ORDER.get(child.tag, 99)
    insertion = len(note)
    for index, existing in enumerate(note):
        if _NOTE_CHILD_ORDER.get(existing.tag, 99) > rank:
            insertion = index
            break
    note.insert(insertion, child)


def _copy_event_kind(target: etree._Element, source: etree._Element) -> bool:
    source_rest = source.find("rest")
    source_pitch = source.find("pitch")
    if (source_rest is None) == (source_pitch is None):
        return False
    for tag in ("pitch", "unpitched", "rest", "accidental"):
        for child in list(target.findall(tag)):
            target.remove(child)
    _insert_ordered(target, copy.deepcopy(source_rest if source_rest is not None else source_pitch))
    if source_pitch is not None:
        accidental = source.find("accidental")
        if accidental is not None:
            _insert_ordered(target, copy.deepcopy(accidental))
    return True


def _best(items: Sequence[EventKindPatchCandidate]) -> EventKindPatchCandidate:
    return max(
        items,
        key=lambda item: (
            item.ensemble_probability,
            item.event_probability,
            item.visual_probability,
            item.context_probability,
            item.measure_probability,
            item.page_score,
            item.variant,
        ),
    )


def propose_event_kind_patch(
    candidates: Sequence[EventKindPatchCandidate],
    *,
    template_index: int,
    missing_candidate_count: int,
    calibrator: EventKindPatchCalibrator | None = None,
    base_measure: etree._Element | None = None,
) -> EventKindPatchResult:
    if not candidates or template_index < 0 or template_index >= len(candidates):
        return EventKindPatchResult(None, (), 0.5, 1.0, False, "invalid_input")
    if missing_candidate_count:
        return EventKindPatchResult(None, (), 0.5, 1.0, False, "alignment_gap")
    template = candidates[template_index]
    if not template.valid:
        return EventKindPatchResult(None, (), 0.5, 1.0, False, "invalid_template")

    family_members, incomplete_families = group_complete_families(
        candidates,
        family_of=lambda item: item.family,
        valid_of=lambda item: item.valid,
    )
    families = sorted(family_members)
    valid = [item for family in families for item in family_members[family]]
    if len(families) < DEFAULT_POLICY.event_kind_patch_minimum_families:
        return EventKindPatchResult(None, (), 0.5, 1.0, False, "insufficient_families")

    skeleton = _measure_skeleton(template.semantics)
    if not template.semantics.notes or any(_measure_skeleton(item.semantics) != skeleton for item in valid):
        return EventKindPatchResult(None, (), 0.5, 1.0, False, "non_kind_structure_disagreement")
    if any(
        item.semantics.voice_count > 1
        or any(note.chord or note.grace or note.tuple_ratio is not None for note in item.semantics.notes)
        or item.measure.find("backup") is not None
        or item.measure.find("forward") is not None
        or item.measure.find(".//time-modification") is not None
        for item in valid
    ):
        return EventKindPatchResult(None, (), 0.5, 1.0, False, "unsupported_event_structure")
    if any(_event_key(note) is None for item in valid for note in item.semantics.notes):
        return EventKindPatchResult(None, (), 0.5, 1.0, False, "unsupported_unpitched_event")

    note_elements = {item.variant: item.measure.findall("note") for item in valid}
    note_count = len(template.semantics.notes)
    if any(len(nodes) != note_count for nodes in note_elements.values()):
        return EventKindPatchResult(None, (), 0.5, 1.0, False, "xml_event_count_mismatch")

    changed: list[tuple[int, EventKindPatchCandidate, tuple[object, ...]]] = []
    support_ratios: list[float] = []
    margin_ratios: list[float] = []
    template_support_ratios: list[float] = []
    pitched_support_ratios: list[float] = []
    support_rows: dict[tuple[int, str], EventKindPatchCandidate] = {}
    abstentions = len(incomplete_families) * note_count
    total_family_count = len(families) + len(incomplete_families)
    possible_family_votes = total_family_count * note_count
    minimum_voting_families = len(families)
    pitched_winners = 0
    rest_winners = 0

    for event_index in range(note_count):
        family_votes: dict[str, tuple[tuple[object, ...], EventKindPatchCandidate]] = {}
        for family, items in family_members.items():
            keys = {_event_key(item.semantics.notes[event_index]) for item in items}
            if len(keys) != 1 or None in keys:
                abstentions += 1
                continue
            family_votes[family] = (next(iter(keys)), _best(items))  # type: ignore[arg-type]
        minimum_voting_families = min(minimum_voting_families, len(family_votes))
        if len(family_votes) < DEFAULT_POLICY.event_kind_patch_minimum_families:
            return EventKindPatchResult(None, (), 0.5, 1.0, False, "insufficient_event_family_votes")

        grouped: dict[tuple[object, ...], list[tuple[str, EventKindPatchCandidate]]] = {}
        for family, (key, representative) in family_votes.items():
            grouped.setdefault(key, []).append((family, representative))
        ranked = sorted(
            grouped.items(),
            key=lambda item: (
                len(item[1]),
                _mean([candidate.ensemble_probability for _, candidate in item[1]]),
                repr(item[0]),
            ),
            reverse=True,
        )
        winner_key, winner_rows = ranked[0]
        runner_up = len(ranked[1][1]) if len(ranked) > 1 else 0
        winner_count = len(winner_rows)
        voting_count = len(family_votes)
        template_key = _event_key(template.semantics.notes[event_index])
        template_count = len(grouped.get(template_key, ())) if template_key is not None else 0

        has_kind_disagreement = any(key[0] != winner_key[0] for key in grouped)
        if not has_kind_disagreement:
            continue
        if not (
            winner_count >= DEFAULT_POLICY.event_kind_patch_minimum_supporting_families
            and winner_count > total_family_count / 2
            and winner_count - runner_up >= 1
        ):
            return EventKindPatchResult(None, (), 0.5, 1.0, False, "no_strict_event_kind_family_majority")
        if winner_key == template_key:
            continue

        representative = _best([candidate for _, candidate in winner_rows])
        changed.append((event_index, representative, winner_key))
        support_ratios.append(winner_count / max(total_family_count, 1))
        margin_ratios.append((winner_count - runner_up) / max(total_family_count, 1))
        template_support_ratios.append(template_count / max(total_family_count, 1))
        if winner_key[0] == "pitch":
            pitched_winners += 1
            pitched_support_ratios.append(winner_count / max(total_family_count, 1))
        else:
            rest_winners += 1
        for family, candidate in winner_rows:
            support_rows[(event_index, family)] = candidate

    if not changed:
        return EventKindPatchResult(None, (), 0.5, 1.0, False, "no_event_kind_change")
    support = list(support_rows.values())
    if not support:
        return EventKindPatchResult(None, (), 0.5, 1.0, False, "missing_support_quality")

    item = EventKindPatchInput(
        candidate_count=len(candidates),
        eligible_family_count=total_family_count,
        voting_family_count=minimum_voting_families,
        changed_event_count=len(changed),
        total_event_count=note_count,
        minimum_winner_family_support_ratio=min(support_ratios),
        mean_winner_family_support_ratio=_mean(support_ratios, 0.0),
        minimum_winner_margin_ratio=min(margin_ratios),
        mean_winner_margin_ratio=_mean(margin_ratios, 0.0),
        maximum_template_family_support_ratio=max(template_support_ratios),
        family_abstention_ratio=abstentions / max(possible_family_votes, 1),
        pitched_winner_count=pitched_winners,
        rest_winner_count=rest_winners,
        minimum_pitched_winner_support_ratio=min(pitched_support_ratios, default=1.0),
        mean_pitched_winner_support_ratio=_mean(pitched_support_ratios, 1.0),
        mean_support_page_probability=_mean([row.page_probability for row in support]),
        mean_support_measure_probability=_mean([row.measure_probability for row in support]),
        mean_support_visual_probability=_mean([row.visual_probability for row in support]),
        mean_support_event_probability=_mean([row.event_probability for row in support]),
        mean_support_context_probability=_mean([row.context_probability for row in support]),
        mean_support_ensemble_probability=_mean([row.ensemble_probability for row in support]),
        minimum_support_ensemble_probability=min(row.ensemble_probability for row in support),
        mean_support_page_score_margin=_mean([row.page_score - template.page_score for row in support], 0.0),
        mean_support_vs_template_measure_probability=_mean([
            row.measure_probability - template.measure_probability for row in support
        ], 0.0),
        mean_support_vs_template_visual_probability=_mean([
            row.visual_probability - template.visual_probability for row in support
        ], 0.0),
        mean_support_vs_template_event_probability=_mean([
            row.event_probability - template.event_probability for row in support
        ], 0.0),
        mean_support_vs_template_context_probability=_mean([
            row.context_probability - template.context_probability for row in support
        ], 0.0),
        mean_support_vs_template_ensemble_probability=_mean([
            row.ensemble_probability - template.ensemble_probability for row in support
        ], 0.0),
    )
    active_calibrator = calibrator or EventKindPatchCalibrator()
    calibration = active_calibrator.calibrate(item)
    if not calibration.accepted:
        return EventKindPatchResult(
            None,
            tuple(index for index, _, _ in changed),
            calibration.probability,
            calibration.threshold,
            False,
            "model_guard",
            item,
            calibration.model_version,
        )

    inherited: dict[str, object] = {
        "divisions": template.semantics.divisions,
        "time": template.semantics.time_signature,
        "key": template.semantics.key_signature,
        "clef": template.semantics.clef,
    }
    source_base = base_measure if base_measure is not None else template.measure
    base_semantics, _ = measure_from_xml(source_base, inherited)
    if len(base_semantics.notes) != note_count:
        return EventKindPatchResult(
            None,
            tuple(index for index, _, _ in changed),
            calibration.probability,
            calibration.threshold,
            False,
            "base_event_count_mismatch",
            item,
            calibration.model_version,
        )
    base_skeleton = _measure_skeleton(base_semantics)
    patched = copy.deepcopy(source_base)
    patched_notes = patched.findall("note")
    for event_index, representative, _winner_key in changed:
        if not _copy_event_kind(patched_notes[event_index], note_elements[representative.variant][event_index]):
            return EventKindPatchResult(
                None,
                tuple(index for index, _, _ in changed),
                calibration.probability,
                calibration.threshold,
                False,
                "xml_event_kind_copy_failed",
                item,
                calibration.model_version,
            )

    parsed, _ = measure_from_xml(patched, inherited)
    if _measure_skeleton(parsed) != base_skeleton:
        return EventKindPatchResult(
            None,
            tuple(index for index, _, _ in changed),
            calibration.probability,
            calibration.threshold,
            False,
            "post_patch_structure_changed",
            item,
            calibration.model_version,
        )
    for event_index, _representative, winner_key in changed:
        if _event_key(parsed.notes[event_index]) != winner_key:
            return EventKindPatchResult(
                None,
                tuple(index for index, _, _ in changed),
                calibration.probability,
                calibration.threshold,
                False,
                "post_patch_event_kind_mismatch",
                item,
                calibration.model_version,
            )
    if any(
        note.pitch is not None and not (1200 <= note.pitch.midi_cents <= 12000)
        for note in parsed.notes
    ):
        return EventKindPatchResult(
            None,
            tuple(index for index, _, _ in changed),
            calibration.probability,
            calibration.threshold,
            False,
            "post_patch_pitch_outlier",
            item,
            calibration.model_version,
        )
    return EventKindPatchResult(
        patched,
        tuple(index for index, _, _ in changed),
        calibration.probability,
        calibration.threshold,
        True,
        "accepted",
        item,
        calibration.model_version,
    )
