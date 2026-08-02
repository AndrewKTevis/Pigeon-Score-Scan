from __future__ import annotations

"""Conservative single-event insertion/deletion consensus.

Whole-measure replacement is intentionally difficult when candidates disagree in pitch
but agree that the template omitted or added one simple event.  This module repairs only
that narrow topology error.  It supports interior, meter-complete, monophonic measures
without chords, grace notes, tuplets, beams, ties, slurs, ornaments, explicit cursor
movement, or unpitched notes.  Each preprocessing family receives one vote; invalid or
split siblings abstain.  The winning event sequence must have a strict majority of at
least three independent families, differ from the template by exactly one uniquely
anchored insertion or deletion, and pass a verified CPU veto model plus post-patch
MusicXML validation.

The module never synthesizes a pitch.  Insertions copy a minimal event already observed
in at least three independent supporting families.  Deletions remove only the uniquely
identified extra template event.  All other XML nodes are preserved byte-for-byte in the
working tree.
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
from .score_ir import MeasureIR, NoteIR, ScoreIR, audit_score, expected_note_duration, measure_from_xml
from .tree_model import VerifiedRandomForestModel
from .variant_family import group_complete_families

FEATURE_NAMES = (
    "candidate_count_scaled",
    "eligible_family_count_scaled",
    "winning_family_support_ratio",
    "winner_margin_ratio",
    "family_abstention_ratio",
    "operation_is_insertion",
    "operation_is_deletion",
    "template_event_count_scaled",
    "winner_event_count_scaled",
    "edit_position_ratio",
    "edit_edge_proximity",
    "anchor_match_ratio",
    "anchor_margin_ratio",
    "inserted_content_support_ratio",
    "inserted_content_margin_ratio",
    "inserted_event_is_rest",
    "inserted_duration_measure_ratio",
    "template_duration_error_scaled",
    "patched_duration_error_scaled",
    "duration_error_improvement_scaled",
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
class EventPresencePatchCandidate:
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
class EventPresencePatchInput:
    candidate_count: int
    eligible_family_count: int
    winning_family_count: int
    runner_up_family_count: int
    incomplete_family_count: int
    operation: str
    template_event_count: int
    winner_event_count: int
    edit_index: int
    anchor_match_ratio: float
    anchor_margin_ratio: float
    inserted_content_family_count: int
    inserted_content_runner_up_count: int
    inserted_event_is_rest: bool
    inserted_event_duration: float
    expected_measure_duration: float
    template_duration_error: float
    patched_duration_error: float
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

        total_families = max(1, self.eligible_family_count + self.incomplete_family_count)
        edit_denominator = max(1, self.winner_event_count - 1)
        position = unit(self.edit_index / edit_denominator)
        edge_proximity = 1.0 - min(1.0, min(position, 1.0 - position) * 2.0)
        expected = max(1e-9, float(self.expected_measure_duration))
        return [
            unit(max(0, self.candidate_count - 1) / 7.0),
            unit(self.eligible_family_count / 4.0),
            unit(self.winning_family_count / total_families),
            unit((self.winning_family_count - self.runner_up_family_count) / total_families),
            unit(self.incomplete_family_count / total_families),
            1.0 if self.operation == "insert" else 0.0,
            1.0 if self.operation == "delete" else 0.0,
            unit(self.template_event_count / 16.0),
            unit(self.winner_event_count / 16.0),
            position,
            edge_proximity,
            unit(self.anchor_match_ratio),
            unit(self.anchor_margin_ratio),
            unit(self.inserted_content_family_count / total_families),
            unit((self.inserted_content_family_count - self.inserted_content_runner_up_count) / total_families),
            1.0 if self.inserted_event_is_rest else 0.0,
            unit(self.inserted_event_duration / expected),
            unit(self.template_duration_error / expected),
            unit(self.patched_duration_error / expected),
            signed((self.template_duration_error - self.patched_duration_error) / expected),
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
class EventPresencePatchCalibration:
    probability: float
    threshold: float
    accepted: bool
    model_version: str
    target_precision: float


@dataclass(frozen=True)
class EventPresencePatchResult:
    patched_measure: etree._Element | None
    operation: str
    changed_event_indices: tuple[int, ...]
    probability: float
    threshold: float
    accepted: bool
    reason: str
    input: EventPresencePatchInput | None = None
    model_version: str = "disabled"


class EventPresencePatchCalibrator:
    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).with_name("resources") / "event_presence_patch_calibrator.json"
        loaded = load_verified_json(model_path, "event_presence_patch_calibration")
        payload = loaded.payload
        self.model = VerifiedRandomForestModel.load(
            model_path,
            "event_presence_patch_calibration",
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
            float(DEFAULT_POLICY.event_presence_patch_probability_floor),
            max(0.0, min(1.0, stored_threshold)),
        )
        self.target_precision = max(0.0, min(1.0, target_precision))
        self.model_verified = self.model.verified and loaded.verified
        self.model_status = self.model.status if self.model.enabled else loaded.status
        self.model_version = self.model.model_version
        self.enabled = self.model.enabled

    def predict_probability(self, item: EventPresencePatchInput) -> float:
        return self.model.predict(item.feature_vector(), neutral=0.5)

    def calibrate(self, item: EventPresencePatchInput) -> EventPresencePatchCalibration:
        probability = self.predict_probability(item)
        accepted = bool(self.enabled and self.model_verified and probability >= self.threshold)
        return EventPresencePatchCalibration(
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


def _event_kind(note: NoteIR) -> str | None:
    if note.rest:
        return "rest"
    return "pitch" if note.pitch is not None else None


def _shape_key(note: NoteIR) -> tuple[object, ...] | None:
    kind = _event_kind(note)
    if kind is None:
        return None
    return (
        _fraction_key(note.duration),
        note.note_type.strip().casefold(),
        int(note.dots),
        note.voice,
        kind,
    )


def _content_key(note: NoteIR) -> tuple[object, ...] | None:
    if note.rest:
        return ("rest",)
    if note.pitch is None:
        return None
    return (
        "pitch",
        note.pitch.step.upper(),
        _fraction_key(note.pitch.alter),
        int(note.pitch.octave),
        note.accidental.strip().casefold(),
    )


def _topology(measure: MeasureIR) -> tuple[tuple[object, ...], ...] | None:
    values = tuple(_shape_key(note) for note in measure.notes)
    if not values or any(value is None for value in values):
        return None
    return tuple(value for value in values if value is not None)


def _duration_total(measure: MeasureIR) -> Fraction:
    return sum((note.duration for note in measure.notes if not note.grace and not note.chord), Fraction(0, 1))


def _duration_error(measure: MeasureIR) -> float:
    expected = measure.expected_duration
    if expected is None:
        return 4.0
    return float(abs(_duration_total(measure) - expected))


def _type_mismatch_ratio(measure: MeasureIR) -> float:
    regular = [note for note in measure.notes if not note.grace]
    if not regular:
        return 1.0
    mismatches = 0
    for note in regular:
        expected = expected_note_duration(note)
        if expected is None or note.duration <= 0:
            mismatches += 1
        elif float(max(note.duration, expected) / min(note.duration, expected)) > 1.08:
            mismatches += 1
    return mismatches / len(regular)


def _supported_measure(candidate: EventPresencePatchCandidate) -> bool:
    semantics = candidate.semantics
    if semantics.time_signature is None or semantics.voice_count != 1 or not semantics.notes:
        return False
    if candidate.measure.find("backup") is not None or candidate.measure.find("forward") is not None:
        return False
    if candidate.measure.find(".//time-modification") is not None or candidate.measure.find(".//beam") is not None:
        return False
    for note, node in zip(semantics.notes, candidate.measure.findall("note"), strict=False):
        if (
            note.chord
            or note.grace
            or note.tuple_ratio is not None
            or note.pitch is None and not note.rest
            or note.ties
            or note.slurs
            or note.articulations
            or note.ornaments
            or node.find("unpitched") is not None
            or node.find(".//notations") is not None
            or (node.find("rest") is not None and node.find("rest").get("measure") == "yes")
        ):
            return False
    return len(candidate.measure.findall("note")) == len(semantics.notes)


def _best(items: Sequence[EventPresencePatchCandidate]) -> EventPresencePatchCandidate:
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


def _anchor_score(left: Sequence[NoteIR], right: Sequence[NoteIR]) -> tuple[int, int]:
    matches = sum(_content_key(a) == _content_key(b) for a, b in zip(left, right, strict=True))
    return matches, len(left)


def _find_single_edit(
    template: Sequence[NoteIR],
    winner: Sequence[NoteIR],
) -> tuple[str, int, float, float] | None:
    """Return a unique single insertion/deletion anchored by unchanged peers.

    Shape equality determines legal edit positions.  Pitch/rest content is used only to
    choose among otherwise identical rhythmic positions, so isolated pitch errors in
    surviving events do not prevent repair.  Ambiguous repeated patterns fail closed.
    """
    if len(winner) == len(template) + 1:
        operation = "insert"
        candidates: list[tuple[int, int, int]] = []
        for index in range(len(winner)):
            reduced = tuple(winner[:index]) + tuple(winner[index + 1 :])
            if tuple(_shape_key(note) for note in reduced) != tuple(_shape_key(note) for note in template):
                continue
            score, count = _anchor_score(template, reduced)
            candidates.append((score, count, index))
    elif len(template) == len(winner) + 1:
        operation = "delete"
        candidates = []
        for index in range(len(template)):
            reduced = tuple(template[:index]) + tuple(template[index + 1 :])
            if tuple(_shape_key(note) for note in reduced) != tuple(_shape_key(note) for note in winner):
                continue
            score, count = _anchor_score(reduced, winner)
            candidates.append((score, count, index))
    else:
        return None
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda item: (item[0], -item[2]), reverse=True)
    best_score, count, index = ranked[0]
    runner_score = ranked[1][0] if len(ranked) > 1 else -1
    if len(ranked) > 1 and best_score <= runner_score:
        return None
    match_ratio = best_score / max(count, 1)
    margin_ratio = (best_score - runner_score) / max(count, 1) if runner_score >= 0 else 1.0
    if match_ratio < DEFAULT_POLICY.event_presence_patch_anchor_match_floor:
        return None
    return operation, index, match_ratio, margin_ratio


def _insert_minimal_event(
    measure: etree._Element,
    *,
    source: etree._Element,
    semantic: NoteIR,
    event_index: int,
    target_divisions: int,
) -> bool:
    if target_divisions <= 0 or semantic.duration <= 0:
        return False
    encoded = semantic.duration * target_divisions
    if encoded.denominator != 1 or encoded.numerator <= 0:
        return False
    source_rest = source.find("rest")
    source_pitch = source.find("pitch")
    if (source_rest is None) == (source_pitch is None):
        return False
    source_type = source.find("type")
    if source_type is None or not (source_type.text or "").strip():
        return False

    note = etree.Element("note")
    note.append(copy.deepcopy(source_rest if source_rest is not None else source_pitch))
    duration = etree.SubElement(note, "duration")
    duration.text = str(encoded.numerator)
    voice = etree.SubElement(note, "voice")
    voice.text = semantic.voice or "1"
    note.append(copy.deepcopy(source_type))
    for dot in source.findall("dot"):
        note.append(copy.deepcopy(dot))
    if source_pitch is not None:
        accidental = source.find("accidental")
        if accidental is not None:
            note.append(copy.deepcopy(accidental))
    staff = source.find("staff")
    if staff is not None:
        note.append(copy.deepcopy(staff))

    notes = measure.findall("note")
    if event_index < len(notes):
        measure.insert(measure.index(notes[event_index]), note)
    elif notes:
        measure.insert(measure.index(notes[-1]) + 1, note)
    else:
        attributes = measure.find("attributes")
        measure.insert(measure.index(attributes) + 1 if attributes is not None else 0, note)
    return True


def _unchanged_note_xml(before: etree._Element, after: etree._Element, operation: str, index: int) -> bool:
    left = [etree.tostring(note) for note in before.findall("note")]
    right = [etree.tostring(note) for note in after.findall("note")]
    if operation == "insert":
        return left == right[:index] + right[index + 1 :]
    return left[:index] + left[index + 1 :] == right


def propose_event_presence_patch(
    candidates: Sequence[EventPresencePatchCandidate],
    *,
    template_index: int,
    missing_candidate_count: int,
    is_first_measure: bool,
    is_last_measure: bool,
    calibrator: EventPresencePatchCalibrator | None = None,
    base_measure: etree._Element | None = None,
) -> EventPresencePatchResult:
    disabled = EventPresencePatchResult(None, "none", (), 0.5, 1.0, False, "invalid_input")
    if not candidates or template_index < 0 or template_index >= len(candidates):
        return disabled
    if missing_candidate_count:
        return EventPresencePatchResult(None, "none", (), 0.5, 1.0, False, "alignment_gap")
    if is_first_measure or is_last_measure:
        return EventPresencePatchResult(None, "none", (), 0.5, 1.0, False, "edge_measure")
    template = candidates[template_index]
    if not template.valid:
        return EventPresencePatchResult(None, "none", (), 0.5, 1.0, False, "invalid_template")

    family_members, incomplete_families = group_complete_families(
        candidates,
        family_of=lambda item: item.family,
        valid_of=lambda item: item.valid,
    )
    if len(family_members) < DEFAULT_POLICY.event_presence_patch_minimum_families:
        return EventPresencePatchResult(None, "none", (), 0.5, 1.0, False, "insufficient_families")
    valid = [item for members in family_members.values() for item in members]
    if any(not _supported_measure(item) for item in valid):
        return EventPresencePatchResult(None, "none", (), 0.5, 1.0, False, "unsupported_event_structure")
    if any(
        item.semantics.time_signature != template.semantics.time_signature
        or item.semantics.key_signature != template.semantics.key_signature
        or item.semantics.clef != template.semantics.clef
        or tuple(direction.stable_tuple() for direction in item.semantics.directions)
        != tuple(direction.stable_tuple() for direction in template.semantics.directions)
        or item.semantics.barlines != template.semantics.barlines
        for item in valid
    ):
        return EventPresencePatchResult(None, "none", (), 0.5, 1.0, False, "non_event_structure_disagreement")

    family_votes: dict[str, tuple[tuple[tuple[object, ...], ...], EventPresencePatchCandidate]] = {}
    for family, items in family_members.items():
        topologies = {_topology(item.semantics) for item in items}
        if len(topologies) != 1 or None in topologies:
            continue
        family_votes[family] = (next(iter(topologies)), _best(items))  # type: ignore[arg-type]
    total_family_count = len(family_members) + len(incomplete_families)
    if len(family_votes) < DEFAULT_POLICY.event_presence_patch_minimum_families:
        return EventPresencePatchResult(None, "none", (), 0.5, 1.0, False, "insufficient_topology_family_votes")

    grouped: dict[tuple[tuple[object, ...], ...], list[tuple[str, EventPresencePatchCandidate]]] = {}
    for family, (topology, representative) in family_votes.items():
        grouped.setdefault(topology, []).append((family, representative))
    ranked = sorted(
        grouped.items(),
        key=lambda item: (
            len(item[1]),
            _mean([candidate.ensemble_probability for _, candidate in item[1]]),
            repr(item[0]),
        ),
        reverse=True,
    )
    winner_topology, winner_rows = ranked[0]
    winner_count = len(winner_rows)
    runner_up_count = len(ranked[1][1]) if len(ranked) > 1 else 0
    if not (
        winner_count >= DEFAULT_POLICY.event_presence_patch_minimum_supporting_families
        and winner_count > total_family_count / 2
        and winner_count - runner_up_count >= 1
    ):
        return EventPresencePatchResult(None, "none", (), 0.5, 1.0, False, "no_strict_topology_family_majority")

    template_topology = _topology(template.semantics)
    if template_topology is None or winner_topology == template_topology:
        return EventPresencePatchResult(None, "none", (), 0.5, 1.0, False, "no_event_presence_change")
    representative = _best([candidate for _, candidate in winner_rows])
    alignment = _find_single_edit(template.semantics.notes, representative.semantics.notes)
    if alignment is None:
        return EventPresencePatchResult(None, "none", (), 0.5, 1.0, False, "not_unique_single_event_edit")
    operation, edit_index, anchor_match_ratio, anchor_margin_ratio = alignment

    expected = template.semantics.expected_duration
    if expected is None or expected <= 0:
        return EventPresencePatchResult(None, operation, (edit_index,), 0.5, 1.0, False, "unknown_meter")
    winner_total = _duration_total(representative.semantics)
    if winner_total != expected or _type_mismatch_ratio(representative.semantics) != 0.0:
        return EventPresencePatchResult(None, operation, (edit_index,), 0.5, 1.0, False, "winning_sequence_not_meter_complete")

    inserted_content_count = 0
    inserted_content_runner_up = 0
    inserted_is_rest = False
    inserted_duration = Fraction(0, 1)
    insertion_source: EventPresencePatchCandidate | None = None
    insertion_semantic: NoteIR | None = None
    if operation == "insert":
        content_groups: dict[tuple[object, ...], list[tuple[str, EventPresencePatchCandidate]]] = {}
        for family, candidate in winner_rows:
            siblings = family_members[family]
            keys = {_content_key(item.semantics.notes[edit_index]) for item in siblings}
            if len(keys) != 1 or None in keys:
                continue
            key = next(iter(keys))
            content_groups.setdefault(key, []).append((family, _best(siblings)))  # type: ignore[arg-type]
        content_ranked = sorted(
            content_groups.items(),
            key=lambda item: (
                len(item[1]),
                _mean([candidate.ensemble_probability for _, candidate in item[1]]),
                repr(item[0]),
            ),
            reverse=True,
        )
        if not content_ranked:
            return EventPresencePatchResult(None, operation, (edit_index,), 0.5, 1.0, False, "missing_inserted_event_content")
        winner_content, content_rows = content_ranked[0]
        inserted_content_count = len(content_rows)
        inserted_content_runner_up = len(content_ranked[1][1]) if len(content_ranked) > 1 else 0
        if not (
            inserted_content_count >= DEFAULT_POLICY.event_presence_patch_minimum_content_families
            and inserted_content_count > total_family_count / 2
            and inserted_content_count - inserted_content_runner_up >= 1
        ):
            return EventPresencePatchResult(None, operation, (edit_index,), 0.5, 1.0, False, "no_strict_inserted_content_majority")
        insertion_source = _best([candidate for _, candidate in content_rows])
        insertion_semantic = insertion_source.semantics.notes[edit_index]
        if _content_key(insertion_semantic) != winner_content:
            return EventPresencePatchResult(None, operation, (edit_index,), 0.5, 1.0, False, "inserted_content_mismatch")
        inserted_is_rest = insertion_semantic.rest
        inserted_duration = insertion_semantic.duration

    source_base = base_measure if base_measure is not None else template.measure
    base_notes = source_base.findall("note")
    if len(base_notes) != len(template.semantics.notes):
        return EventPresencePatchResult(None, operation, (edit_index,), 0.5, 1.0, False, "base_event_count_mismatch")
    patched = copy.deepcopy(source_base)
    if operation == "insert":
        assert insertion_source is not None and insertion_semantic is not None
        source_notes = insertion_source.measure.findall("note")
        if edit_index >= len(source_notes) or not _insert_minimal_event(
            patched,
            source=source_notes[edit_index],
            semantic=insertion_semantic,
            event_index=edit_index,
            target_divisions=template.semantics.divisions,
        ):
            return EventPresencePatchResult(None, operation, (edit_index,), 0.5, 1.0, False, "xml_event_insert_failed")
    else:
        patched_notes = patched.findall("note")
        if edit_index >= len(patched_notes):
            return EventPresencePatchResult(None, operation, (edit_index,), 0.5, 1.0, False, "xml_event_delete_failed")
        patched.remove(patched_notes[edit_index])

    if not _unchanged_note_xml(source_base, patched, operation, edit_index):
        return EventPresencePatchResult(None, operation, (edit_index,), 0.5, 1.0, False, "unrelated_event_xml_changed")

    inherited: dict[str, object] = {
        "divisions": template.semantics.divisions,
        "time": template.semantics.time_signature,
        "key": template.semantics.key_signature,
        "clef": template.semantics.clef,
    }
    parsed, _ = measure_from_xml(patched, inherited)
    patched_error = _duration_error(parsed)
    forbidden = {
        issue.code
        for issue in audit_score(ScoreIR((parsed,)))
        if issue.code in {
            "multiple_voices",
            "zero_duration",
            "type_duration_mismatch",
            "chord_duration_mismatch",
            "empty_measure",
        }
    }
    if (
        len(parsed.notes) != len(representative.semantics.notes)
        or _duration_total(parsed) != expected
        or patched_error != 0.0
        or _type_mismatch_ratio(parsed) != 0.0
        or forbidden
    ):
        return EventPresencePatchResult(None, operation, (edit_index,), 0.5, 1.0, False, "post_patch_event_validation_failed")
    if operation == "insert" and insertion_semantic is not None:
        if _shape_key(parsed.notes[edit_index]) != _shape_key(insertion_semantic):
            return EventPresencePatchResult(None, operation, (edit_index,), 0.5, 1.0, False, "post_patch_inserted_shape_mismatch")
        if _content_key(parsed.notes[edit_index]) != _content_key(insertion_semantic):
            return EventPresencePatchResult(None, operation, (edit_index,), 0.5, 1.0, False, "post_patch_inserted_content_mismatch")

    support = [candidate for _, candidate in winner_rows]
    item = EventPresencePatchInput(
        candidate_count=len(candidates),
        eligible_family_count=len(family_members),
        winning_family_count=winner_count,
        runner_up_family_count=runner_up_count,
        incomplete_family_count=len(incomplete_families),
        operation=operation,
        template_event_count=len(template.semantics.notes),
        winner_event_count=len(representative.semantics.notes),
        edit_index=edit_index,
        anchor_match_ratio=anchor_match_ratio,
        anchor_margin_ratio=anchor_margin_ratio,
        inserted_content_family_count=inserted_content_count,
        inserted_content_runner_up_count=inserted_content_runner_up,
        inserted_event_is_rest=inserted_is_rest,
        inserted_event_duration=float(inserted_duration),
        expected_measure_duration=float(expected),
        template_duration_error=_duration_error(template.semantics),
        patched_duration_error=patched_error,
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
    active_calibrator = calibrator or EventPresencePatchCalibrator()
    calibration = active_calibrator.calibrate(item)
    if not calibration.accepted:
        return EventPresencePatchResult(
            None,
            operation,
            (edit_index,),
            calibration.probability,
            calibration.threshold,
            False,
            "model_guard",
            item,
            calibration.model_version,
        )
    return EventPresencePatchResult(
        patched,
        operation,
        (edit_index,),
        calibration.probability,
        calibration.threshold,
        True,
        "accepted",
        item,
        calibration.model_version,
    )
