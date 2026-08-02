from __future__ import annotations

"""Deterministic semantic evaluation for MusicXML release benchmarks.

The evaluator aligns measures and events globally before scoring.  This prevents one
inserted measure or note from shifting every later comparison, a failure mode of the
legacy index/zip evaluator.  Metrics are emitted together with additive counts so a
frozen multi-score benchmark can be aggregated without averaging incompatible rates.
"""

import hashlib
import tempfile
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence

from lxml import etree

from .accidental_semantics import key_step_alters, normalise_accidental
from .alignment import SequenceAlignment, align_measure_sequences
from .lyric_xml import LyricState, normalized_lyric_topology
from .musicxml_signature import measure_preservation_signatures
from .score_ir import DirectionIR, MeasureIR, NoteIR, note_substitution_cost, score_from_tree
from .util import sha256_file


@dataclass(frozen=True)
class EventAlignmentPair:
    reference_index: int | None
    candidate_index: int | None
    cost: float


@dataclass(frozen=True)
class EventAlignment:
    pairs: tuple[EventAlignmentPair, ...]
    total_cost: float


def _parse(path: Path) -> tuple[etree._ElementTree, tuple[MeasureIR, ...]]:
    parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=False, huge_tree=False)
    tree = etree.parse(str(path), parser)
    return tree, score_from_tree(tree).measures


def align_events(reference: Sequence[NoteIR], candidate: Sequence[NoteIR]) -> EventAlignment:
    """Globally align note/rest sequences using semantic substitution costs."""
    reference = tuple(reference)
    candidate = tuple(candidate)
    m, n = len(reference), len(candidate)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    back: list[list[str | None]] = [[None] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = float(i)
        back[i][0] = "delete"
    for j in range(1, n + 1):
        dp[0][j] = float(j)
        back[0][j] = "insert"
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            substitution = note_substitution_cost(reference[i - 1], candidate[j - 1])
            value, _rank, operation = min(
                (dp[i - 1][j - 1] + substitution, 0, "match"),
                (dp[i - 1][j] + 1.0, 1, "delete"),
                (dp[i][j - 1] + 1.0, 2, "insert"),
            )
            dp[i][j] = value
            back[i][j] = operation

    pairs: list[EventAlignmentPair] = []
    i, j = m, n
    while i > 0 or j > 0:
        operation = back[i][j]
        if operation == "match":
            cost = note_substitution_cost(reference[i - 1], candidate[j - 1])
            pairs.append(EventAlignmentPair(i - 1, j - 1, cost))
            i -= 1
            j -= 1
        elif operation == "delete":
            pairs.append(EventAlignmentPair(i - 1, None, 1.0))
            i -= 1
        elif operation == "insert":
            pairs.append(EventAlignmentPair(None, j - 1, 1.0))
            j -= 1
        elif i > 0:
            pairs.append(EventAlignmentPair(i - 1, None, 1.0))
            i -= 1
        else:
            pairs.append(EventAlignmentPair(None, j - 1, 1.0))
            j -= 1
    return EventAlignment(tuple(reversed(pairs)), round(dp[m][n], 9))


def _pitch_key(note: NoteIR) -> tuple[object, ...] | None:
    if note.rest or note.pitch is None:
        return None
    # ``<accidental>`` is optional in MusicXML when the pitch alteration already
    # determines the displayed sign.  Treating its omission as a wrong pitch
    # falsely penalizes semantically and visually equivalent exports; the
    # authoritative alteration remains part of ``PitchIR.stable_tuple()``.
    return note.pitch.stable_tuple()


def _event_key(note: NoteIR, *, chord_member: bool | None = None) -> tuple[object, ...]:
    # Onset is evaluated separately.  Excluding it here avoids counting one inserted
    # note again as substitutions for every later, otherwise identical event.
    return (
        _pitch_key(note),
        note.duration,
        note.voice,
        note.rest,
        note.chord if chord_member is None else chord_member,
        note.grace,
        note.note_type,
        note.dots,
        note.ties,
        note.slurs,
        note.articulations,
        note.ornaments,
        note.tuple_ratio,
    )


def _rhythm_key(note: NoteIR, *, chord_member: bool | None = None) -> tuple[object, ...]:
    """Return the exact, exporter-independent part of an event's rhythm.

    Voice labels are document-local identifiers.  They are evaluated by the
    dedicated voice metrics and must not turn an otherwise identical sounding
    rhythm into an error merely because one exporter calls a voice ``1`` and
    another calls it ``5``.  Onset and duration need measure-division-aware
    comparison, so they are handled by :func:`_rhythm_equivalent` below.
    """

    return (
        note.note_type,
        note.dots,
        note.tuple_ratio,
        note.chord if chord_member is None else chord_member,
        note.grace,
    )


def _quantized_fraction_equivalent(
    left: Fraction,
    right: Fraction,
    left_divisions: int,
    right_divisions: int,
) -> bool:
    """Compare quarter-note fractions without penalising legal tuplet rounding.

    Some authoritative MusicXML encodes an exact triplet as 85/256, 85/256,
    86/256 while another exporter uses 1/3 for every event.  Those encodings
    engrave and play equivalently.  The tolerance is capped at 1/128 quarter and
    never exceeds one tick of the finer representation, so ordinary duration
    mistakes remain errors.
    """

    if left == right:
        return True
    finest_divisions = max(1, int(left_divisions), int(right_divisions))
    tolerance = min(Fraction(1, 128), Fraction(1, finest_divisions))
    return abs(left - right) <= tolerance


def _rhythm_equivalent(
    left: NoteIR,
    right: NoteIR,
    *,
    left_divisions: int,
    right_divisions: int,
    left_chord_member: bool,
    right_chord_member: bool,
) -> bool:
    return bool(
        _rhythm_key(left, chord_member=left_chord_member)
        == _rhythm_key(right, chord_member=right_chord_member)
        and _quantized_fraction_equivalent(
            left.onset,
            right.onset,
            left_divisions,
            right_divisions,
        )
        and _quantized_fraction_equivalent(
            left.duration,
            right.duration,
            left_divisions,
            right_divisions,
        )
    )


def _key_signature_equivalent(
    left: tuple[int, str] | None,
    right: tuple[int, str] | None,
) -> bool:
    """Compare the printed key signature, not optional analytical mode text."""

    if left is None or right is None:
        return left is right
    return int(left[0]) == int(right[0])


def _canonical_chord_flags(notes: Sequence[NoteIR]) -> tuple[bool, ...]:
    """Return order-independent MusicXML chord-continuation flags.

    MusicXML's first note in a chord lacks ``<chord/>``.  Different exporters may
    choose a different pitch as that anchor while engraving the same chord.  Use the
    lowest-pitch event as a deterministic semantic anchor, then mark every other
    pitch at that staff/voice/onset as a continuation.
    """

    groups: dict[tuple[object, ...], list[int]] = {}
    for index, note in enumerate(notes):
        if note.rest:
            continue
        groups.setdefault(
            (note.onset, max(1, int(note.staff)), str(note.voice or "1")),
            [],
        ).append(index)
    flags = [False] * len(notes)
    for indices in groups.values():
        ordered = sorted(
            indices,
            key=lambda index: (
                notes[index].pitch.midi_cents if notes[index].pitch is not None else Fraction(10_000_000, 1),
                notes[index].duration,
                index,
            ),
        )
        for index in ordered[1:]:
            flags[index] = True
    return tuple(flags)


def _canonical_event_sort_key(note: NoteIR, original_index: int) -> tuple[object, ...]:
    """Order simultaneous polyphonic events independently of XML serialization."""

    midi = note.pitch.midi_cents if note.pitch is not None and not note.rest else Fraction(10_000_000, 1)
    return (
        note.onset,
        bool(note.rest),
        midi,
        note.duration,
        max(1, int(note.staff)),
        str(note.voice or "1"),
        original_index,
    )


def align_measure_events(reference: Sequence[NoteIR], candidate: Sequence[NoteIR]) -> EventAlignment:
    """Align measure events while preserving original indexes for XML annotations."""

    reference_indexed = sorted(
        enumerate(reference),
        key=lambda item: _canonical_event_sort_key(item[1], item[0]),
    )
    candidate_indexed = sorted(
        enumerate(candidate),
        key=lambda item: _canonical_event_sort_key(item[1], item[0]),
    )
    aligned = align_events(
        tuple(note for _index, note in reference_indexed),
        tuple(note for _index, note in candidate_indexed),
    )
    pairs = tuple(
        EventAlignmentPair(
            None if pair.reference_index is None else reference_indexed[pair.reference_index][0],
            None if pair.candidate_index is None else candidate_indexed[pair.candidate_index][0],
            pair.cost,
        )
        for pair in aligned.pairs
    )
    return EventAlignment(pairs, aligned.total_cost)


def _direction_key(item: DirectionIR) -> tuple[object, ...]:
    onset, placement, kind, value, end_value, staff, voice = item.stable_tuple()
    # MusicXML relation numbers are document-local identifiers, not musical
    # content.  A wedge numbered 1 and an otherwise identical wedge numbered 7
    # must compare equal.
    if kind == "wedge":
        end_value = ""
    return onset, placement, kind, value, end_value, staff, voice


def _direction_content_key(item: DirectionIR) -> tuple[object, ...]:
    _onset, _placement, kind, value, end_value, _staff, _voice = _direction_key(item)
    return kind, value, end_value


def _direction_counter(
    measure: MeasureIR,
    *,
    kinds: frozenset[str] | None = None,
    values: frozenset[str] | None = None,
    content_only: bool = False,
) -> Counter[tuple[object, ...]]:
    return Counter(
        _direction_content_key(item) if content_only else _direction_key(item)
        for item in measure.directions
        if (kinds is None or item.kind in kinds)
        and (
            values is None
            or str(item.value).strip().casefold() in values
        )
    )


_DIRECTION_METRIC_FILTERS: dict[
    str, tuple[frozenset[str], frozenset[str] | None]
] = {
    "words_direction": (frozenset({"words"}), None),
    "dynamic_direction": (frozenset({"dynamic"}), None),
    "wedge_direction": (frozenset({"wedge"}), None),
    # Wedge-wide F1 can hide a rare start type behind abundant stop endpoints.
    # Keep start direction and stop placement independently observable so a
    # release cannot pass by recognizing only one hairpin orientation.
    "crescendo_wedge_start": (
        frozenset({"wedge"}),
        frozenset({"crescendo"}),
    ),
    "diminuendo_wedge_start": (
        frozenset({"wedge"}),
        frozenset({"diminuendo"}),
    ),
    "wedge_stop": (frozenset({"wedge"}), frozenset({"stop"})),
}


def _simple_expression_markers(measure: MeasureIR) -> Counter[tuple[object, ...]]:
    return Counter(
        item.stable_tuple()
        for item in measure.directions
        if item.kind in {"dynamic", "metronome"}
    )


def _tie_anchor(note: NoteIR) -> tuple[object, ...] | None:
    pitch = _pitch_key(note)
    if pitch is None:
        return None
    return (
        max(1, int(note.staff)),
        str(note.voice or "1"),
        pitch,
    )


def _cross_tie_boundary_ties(left: MeasureIR, right: MeasureIR) -> Counter[tuple[object, ...]]:
    starts: Counter[tuple[object, ...]] = Counter(
        anchor
        for note in left.notes
        if "start" in set(note.ties)
        for anchor in [_tie_anchor(note)]
        if anchor is not None
    )
    stops: Counter[tuple[object, ...]] = Counter(
        anchor
        for note in right.notes
        if "stop" in set(note.ties)
        for anchor in [_tie_anchor(note)]
        if anchor is not None
    )
    return starts & stops


def _cross_tie_count(measures: Sequence[MeasureIR]) -> int:
    return sum(
        sum(_cross_tie_boundary_ties(measures[index], measures[index + 1]).values())
        for index in range(max(0, len(measures) - 1))
    )


def _slur_endpoint_kinds(note: NoteIR) -> frozenset[str]:
    return frozenset(
        str(kind).strip().casefold()
        for kind, _number in note.slurs
        if str(kind).strip().casefold() in {"start", "stop"}
    )


def _beam_topology(note: etree._Element) -> tuple[tuple[str, str], ...]:
    """Return normalized MusicXML beam levels for one event.

    Beam topology is intentionally evaluated from the source XML rather than Score IR:
    the recognition pipeline currently treats MusicXML as the preservation surface for
    beams, so omitting this metric would allow a stable release to pass while losing
    required engraving semantics.
    """

    values: list[tuple[str, str]] = []
    for beam in note.findall("beam"):
        number = str(beam.get("number") or "1").strip() or "1"
        value = " ".join((beam.text or "").split()).casefold()
        if value:
            values.append((number, value))
    return tuple(sorted(values))


def _measure_beam_topologies(
    tree: etree._ElementTree,
) -> list[tuple[tuple[tuple[str, str], ...], ...]]:
    return [
        tuple(_beam_topology(note) for note in measure.findall("note"))
        for measure in tree.findall("./part/measure")
    ]


def _beam_marker_counter(topology: tuple[tuple[str, str], ...]) -> Counter[tuple[object, ...]]:
    return Counter((number, value) for number, value in topology)


def _articulation_markers(note: NoteIR) -> Counter[tuple[object, ...]]:
    return Counter((str(mark).strip().casefold(),) for mark in note.articulations if str(mark).strip())


def _ornament_markers(note: NoteIR) -> Counter[tuple[object, ...]]:
    return Counter((str(mark).strip().casefold(),) for mark in note.ornaments if str(mark).strip())


def _accidental_semantic_markers(
    measure: MeasureIR,
) -> Counter[tuple[object, ...]]:
    """Return printed/required accidental semantics at their musical anchors.

    MusicXML permits ``<accidental>`` to be omitted when ``<alter>`` already
    causes an engraver to display the required sign.  Comparing the optional XML
    element directly would therefore create false errors.  Instead, reconstruct
    the per-measure accidental state and emit a marker when a sign is explicit,
    when the pitch changes the current state, or when a courtesy sign is
    explicitly retained.  A tie continuation legally carries its alteration
    without reprinting a sign.
    """

    key_state = key_step_alters(measure.key_signature)
    local_state: dict[tuple[str, int], Fraction] = {}
    markers: Counter[tuple[object, ...]] = Counter()
    for note in measure.notes:
        if note.pitch is None or note.rest or note.grace:
            continue
        step = note.pitch.step.upper()
        octave = int(note.pitch.octave)
        position = (step, octave)
        expected = local_state.get(
            position,
            key_state.get(step, Fraction(0, 1)),
        )
        explicit = normalise_accidental(note.accidental)
        tie_kinds = {
            str(value).strip().casefold() for value in note.ties
        }
        # A middle note in a longer tie chain commonly carries both ``stop``
        # and ``start``.  It is still a continuation and does not reprint the
        # accidental unless the XML explicitly requests one.
        tie_continuation = bool(tie_kinds & {"stop", "continue"})
        required = bool(explicit) or (
            not tie_continuation and note.pitch.alter != expected
        )
        if required:
            markers[
                (
                    str(note.onset),
                    max(1, int(note.staff)),
                    str(note.voice or "1"),
                    step,
                    octave,
                    str(note.pitch.alter),
                )
            ] += 1
        if explicit or note.pitch.alter != expected or tie_continuation:
            local_state[position] = note.pitch.alter
    return markers


def _lyric_state_key(state: LyricState | None) -> tuple[str, str, str] | None:
    return state.stable_tuple() if state is not None else None


def _measure_lyric_topologies(tree: etree._ElementTree) -> list[tuple[LyricState | None, ...] | None]:
    return [normalized_lyric_topology(measure.findall("note")) for measure in tree.findall("./part/measure")]


def _slur_event_anchor(note: NoteIR) -> tuple[object, ...]:
    return (
        note.onset,
        max(1, int(note.staff)),
        str(note.voice or "1"),
        _pitch_key(note),
        note.duration,
    )


def _slur_topology(measure: MeasureIR) -> tuple[tuple[tuple[object, ...], tuple[object, ...]], ...] | None:
    endpoints: dict[str, dict[str, list[tuple[object, ...]]]] = {}
    for note in measure.notes:
        for kind, number in note.slurs:
            kind = str(kind).strip().casefold()
            number = str(number).strip() or "1"
            if kind not in {"start", "stop"}:
                return None
            endpoints.setdefault(number, {"start": [], "stop": []})[kind].append(
                _slur_event_anchor(note)
            )
    arcs: list[tuple[tuple[object, ...], tuple[object, ...]]] = []
    for bucket in endpoints.values():
        if len(bucket["start"]) != 1 or len(bucket["stop"]) != 1:
            return None
        start, stop = bucket["start"][0], bucket["stop"][0]
        if start[0] > stop[0]:
            return None
        arcs.append((start, stop))
    return tuple(sorted(arcs))


def _repeat_markers(measure: MeasureIR) -> Counter[tuple[object, ...]]:
    return Counter(
        ((str(location).strip().casefold() or "right"), str(direction).strip().casefold())
        for location, _style, direction, _ending in measure.barlines
        if str(direction).strip()
    )


def _counter_overlap(left: Counter[tuple[object, ...]], right: Counter[tuple[object, ...]]) -> int:
    return sum((left & right).values())


def _safe_rate(numerator: int | float, denominator: int | float, *, empty: float = 1.0) -> float:
    return float(numerator) / float(denominator) if denominator else empty


def _marker_f1(matches: int | float, reference_total: int | float, candidate_total: int | float) -> float:
    """Score exact marker matches without rewarding a complete miss.

    An empty reference and empty candidate is a perfect no-op.  In every other
    case the count form of F1 is unambiguous, including the important case where
    both precision and recall are zero because two non-empty marker sets have no
    overlap.
    """

    denominator = float(reference_total) + float(candidate_total)
    return 1.0 if denominator == 0.0 else (2.0 * float(matches)) / denominator


def _measure_voice_topology(measure: MeasureIR) -> tuple[tuple[int, str, int], ...]:
    counts: Counter[tuple[int, str]] = Counter(
        (max(1, int(note.staff)), str(note.voice or "1"))
        for note in measure.notes
    )
    return tuple(sorted((staff, voice, count) for (staff, voice), count in counts.items()))


def _compare_single_part_musicxml(reference: Path, candidate: Path) -> dict[str, object]:
    """Compare two isolated single-part MusicXML files after global alignment."""
    ref_tree, ref = _parse(reference)
    cand_tree, cand = _parse(candidate)
    ref_lyrics = _measure_lyric_topologies(ref_tree)
    cand_lyrics = _measure_lyric_topologies(cand_tree)
    ref_beams = _measure_beam_topologies(ref_tree)
    cand_beams = _measure_beam_topologies(cand_tree)
    ref_preservation = measure_preservation_signatures(ref_tree.findall("./part/measure"))
    cand_preservation = measure_preservation_signatures(cand_tree.findall("./part/measure"))
    measure_alignment: SequenceAlignment = align_measure_sequences(ref, cand)

    counts: dict[str, int | float] = {
        "reference_measures": len(ref),
        "candidate_measures": len(cand),
        "measure_denominator": max(len(ref), len(cand)),
        "aligned_measures": 0,
        "exact_measures": 0,
        "preservation_exact_measures": 0,
        "deleted_measures": 0,
        "inserted_measures": len(measure_alignment.unmatched_candidate_indices),
        "reference_events": sum(len(measure.notes) for measure in ref),
        "candidate_events": sum(len(measure.notes) for measure in cand),
        "reference_voice_assignments": sum(len(measure.notes) for measure in ref),
        "reference_staff_assignments": sum(len(measure.notes) for measure in ref),
        "voice_assignment_correct": 0,
        "staff_assignment_correct": 0,
        "voice_topology_correct": 0,
        "staff_clef_correct": 0,
        "matched_events": 0,
        "exact_events": 0,
        "substituted_events": 0,
        "deleted_events": 0,
        "inserted_events": 0,
        "weighted_event_cost": 0.0,
        "reference_pitched_events": sum(1 for measure in ref for note in measure.notes if _pitch_key(note) is not None),
        "pitch_correct": 0,
        "reference_rhythm_events": sum(1 for measure in ref for note in measure.notes if not note.grace),
        "duration_correct": 0,
        "rhythm_correct": 0,
        "onset_correct": 0,
        "chord_correct": 0,
        "tuplet_topology_correct": 0,
        "reference_tuplet_event_count": sum(
            1 for measure in ref for note in measure.notes if note.tuple_ratio is not None
        ),
        "candidate_tuplet_event_count": sum(
            1 for measure in cand for note in measure.notes if note.tuple_ratio is not None
        ),
        "tuplet_event_matches": 0,
        "tie_topology_correct": 0,
        "reference_tie_endpoint_count": sum(
            len(set(note.ties)) for measure in ref for note in measure.notes
        ),
        "candidate_tie_endpoint_count": sum(
            len(set(note.ties)) for measure in cand for note in measure.notes
        ),
        "tie_endpoint_matches": 0,
        "slur_topology_correct": 0,
        "reference_slur_endpoint_count": sum(
            len(_slur_endpoint_kinds(note)) for measure in ref for note in measure.notes
        ),
        "candidate_slur_endpoint_count": sum(
            len(_slur_endpoint_kinds(note)) for measure in cand for note in measure.notes
        ),
        "slur_endpoint_matches": 0,
        "reference_beam_marker_count": sum(
            len(markers) for measure in ref_beams for markers in measure
        ),
        "candidate_beam_marker_count": sum(
            len(markers) for measure in cand_beams for markers in measure
        ),
        "beam_marker_matches": 0,
        "beam_topology_correct": 0,
        "articulation_topology_correct": 0,
        "reference_articulation_marker_count": sum(
            len(note.articulations) for measure in ref for note in measure.notes
        ),
        "candidate_articulation_marker_count": sum(
            len(note.articulations) for measure in cand for note in measure.notes
        ),
        "articulation_marker_matches": 0,
        "ornament_topology_correct": 0,
        "reference_ornament_marker_count": sum(
            len(note.ornaments) for measure in ref for note in measure.notes
        ),
        "candidate_ornament_marker_count": sum(
            len(note.ornaments) for measure in cand for note in measure.notes
        ),
        "ornament_marker_matches": 0,
        "reference_accidental_marker_count": sum(
            sum(_accidental_semantic_markers(measure).values())
            for measure in ref
        ),
        "candidate_accidental_marker_count": sum(
            sum(_accidental_semantic_markers(measure).values())
            for measure in cand
        ),
        "accidental_marker_matches": 0,
        "grace_topology_correct": 0,
        "reference_grace_event_count": sum(
            1 for measure in ref for note in measure.notes if note.grace
        ),
        "candidate_grace_event_count": sum(
            1 for measure in cand for note in measure.notes if note.grace
        ),
        "grace_event_matches": 0,
        "lyric_topology_correct": 0,
        "reference_lyric_event_count": sum(
            state is not None
            for topology in ref_lyrics if topology is not None
            for state in topology
        ),
        "candidate_lyric_event_count": sum(
            state is not None
            for topology in cand_lyrics if topology is not None
            for state in topology
        ),
        "lyric_event_matches": 0,
        "reference_cross_tie_boundary_count": max(0, len(ref) - 1),
        "candidate_cross_tie_boundary_count": max(0, len(cand) - 1),
        "reference_cross_tie_count": _cross_tie_count(ref),
        "candidate_cross_tie_count": _cross_tie_count(cand),
        "cross_tie_boundary_correct": 0,
        "cross_tie_matches": 0,
        "rest_correct": 0,
        "reference_direction_count": sum(len(measure.directions) for measure in ref),
        "candidate_direction_count": sum(len(measure.directions) for measure in cand),
        "direction_matches": 0,
        "reference_expression_marker_count": sum(
            sum(_simple_expression_markers(measure).values()) for measure in ref
        ),
        "candidate_expression_marker_count": sum(
            sum(_simple_expression_markers(measure).values()) for measure in cand
        ),
        "expression_marker_matches": 0,
        "time_signature_correct": 0,
        "key_signature_correct": 0,
        "clef_correct": 0,
        "barline_correct": 0,
        "reference_repeat_marker_count": sum(sum(_repeat_markers(measure).values()) for measure in ref),
        "candidate_repeat_marker_count": sum(sum(_repeat_markers(measure).values()) for measure in cand),
        "repeat_marker_matches": 0,
    }
    counts["direction_content_matches"] = 0
    for prefix, (kinds, values) in _DIRECTION_METRIC_FILTERS.items():
        counts[f"reference_{prefix}_count"] = sum(
            sum(
                _direction_counter(
                    measure,
                    kinds=kinds,
                    values=values,
                ).values()
            )
            for measure in ref
        )
        counts[f"candidate_{prefix}_count"] = sum(
            sum(
                _direction_counter(
                    measure,
                    kinds=kinds,
                    values=values,
                ).values()
            )
            for measure in cand
        )
        counts[f"{prefix}_matches"] = 0
        counts[f"{prefix}_content_matches"] = 0
    per_measure: list[dict[str, object]] = []

    for ref_index, ref_measure in enumerate(ref):
        cand_index = measure_alignment.reference_to_candidate[ref_index]
        if cand_index is None:
            counts["deleted_measures"] = int(counts["deleted_measures"]) + 1
            counts["deleted_events"] = int(counts["deleted_events"]) + len(ref_measure.notes)
            counts["weighted_event_cost"] = float(counts["weighted_event_cost"]) + len(ref_measure.notes)
            per_measure.append({
                "reference_index": ref_index + 1,
                "candidate_index": None,
                "exact": False,
                "preservation_exact": False,
                "event_edits": len(ref_measure.notes),
                "weighted_event_cost": float(len(ref_measure.notes)),
            })
            continue

        cand_measure = cand[cand_index]
        counts["aligned_measures"] = int(counts["aligned_measures"]) + 1
        exact = ref_measure.fingerprint == cand_measure.fingerprint
        preservation_exact = (
            ref_index < len(ref_preservation)
            and cand_index < len(cand_preservation)
            and ref_preservation[ref_index] == cand_preservation[cand_index]
        )
        counts["exact_measures"] = int(counts["exact_measures"]) + int(exact)
        counts["preservation_exact_measures"] = int(counts["preservation_exact_measures"]) + int(
            preservation_exact
        )
        counts["time_signature_correct"] = int(counts["time_signature_correct"]) + int(ref_measure.time_signature == cand_measure.time_signature)
        counts["key_signature_correct"] = int(counts["key_signature_correct"]) + int(
            _key_signature_equivalent(
                ref_measure.key_signature,
                cand_measure.key_signature,
            )
        )
        counts["clef_correct"] = int(counts["clef_correct"]) + int(ref_measure.clef == cand_measure.clef)
        counts["staff_clef_correct"] = int(counts["staff_clef_correct"]) + int(
            ref_measure.staff_clefs == cand_measure.staff_clefs
        )
        counts["voice_topology_correct"] = int(counts["voice_topology_correct"]) + int(
            _measure_voice_topology(ref_measure) == _measure_voice_topology(cand_measure)
        )
        counts["barline_correct"] = int(counts["barline_correct"]) + int(ref_measure.barlines == cand_measure.barlines)
        counts["repeat_marker_matches"] = int(counts["repeat_marker_matches"]) + _counter_overlap(
            _repeat_markers(ref_measure), _repeat_markers(cand_measure)
        )
        ref_slurs = _slur_topology(ref_measure)
        cand_slurs = _slur_topology(cand_measure)
        counts["slur_topology_correct"] = int(counts["slur_topology_correct"]) + int(
            ref_slurs is not None and ref_slurs == cand_slurs
        )

        ref_dirs = _direction_counter(ref_measure)
        cand_dirs = _direction_counter(cand_measure)
        counts["direction_matches"] = int(counts["direction_matches"]) + _counter_overlap(ref_dirs, cand_dirs)
        counts["direction_content_matches"] = int(counts["direction_content_matches"]) + _counter_overlap(
            _direction_counter(ref_measure, content_only=True),
            _direction_counter(cand_measure, content_only=True),
        )
        for prefix, (kinds, values) in _DIRECTION_METRIC_FILTERS.items():
            counts[f"{prefix}_matches"] = int(counts[f"{prefix}_matches"]) + _counter_overlap(
                _direction_counter(ref_measure, kinds=kinds, values=values),
                _direction_counter(cand_measure, kinds=kinds, values=values),
            )
            counts[f"{prefix}_content_matches"] = int(
                counts[f"{prefix}_content_matches"]
            ) + _counter_overlap(
                _direction_counter(
                    ref_measure,
                    kinds=kinds,
                    values=values,
                    content_only=True,
                ),
                _direction_counter(
                    cand_measure,
                    kinds=kinds,
                    values=values,
                    content_only=True,
                ),
            )
        counts["expression_marker_matches"] = int(counts["expression_marker_matches"]) + _counter_overlap(
            _simple_expression_markers(ref_measure),
            _simple_expression_markers(cand_measure),
        )
        counts["accidental_marker_matches"] = int(
            counts["accidental_marker_matches"]
        ) + _counter_overlap(
            _accidental_semantic_markers(ref_measure),
            _accidental_semantic_markers(cand_measure),
        )

        ref_chord_flags = _canonical_chord_flags(ref_measure.notes)
        cand_chord_flags = _canonical_chord_flags(cand_measure.notes)
        alignment = align_measure_events(ref_measure.notes, cand_measure.notes)
        local_edits = 0
        for pair in alignment.pairs:
            if pair.reference_index is None:
                counts["inserted_events"] = int(counts["inserted_events"]) + 1
                local_edits += 1
                continue
            if pair.candidate_index is None:
                counts["deleted_events"] = int(counts["deleted_events"]) + 1
                local_edits += 1
                continue
            left = ref_measure.notes[pair.reference_index]
            right = cand_measure.notes[pair.candidate_index]
            counts["matched_events"] = int(counts["matched_events"]) + 1
            counts["voice_assignment_correct"] = int(counts["voice_assignment_correct"]) + int(
                str(left.voice or "1") == str(right.voice or "1")
            )
            counts["staff_assignment_correct"] = int(counts["staff_assignment_correct"]) + int(
                max(1, int(left.staff)) == max(1, int(right.staff))
            )
            left_chord_flag = ref_chord_flags[pair.reference_index]
            right_chord_flag = cand_chord_flags[pair.candidate_index]
            event_exact = _event_key(left, chord_member=left_chord_flag) == _event_key(
                right,
                chord_member=right_chord_flag,
            )
            counts["exact_events"] = int(counts["exact_events"]) + int(event_exact)
            counts["substituted_events"] = int(counts["substituted_events"]) + int(not event_exact)
            local_edits += int(not event_exact)
            left_pitch = _pitch_key(left)
            if left_pitch is not None:
                counts["pitch_correct"] = int(counts["pitch_correct"]) + int(left_pitch == _pitch_key(right))
            if not left.grace:
                duration_equivalent = _quantized_fraction_equivalent(
                    left.duration,
                    right.duration,
                    ref_measure.divisions,
                    cand_measure.divisions,
                )
                onset_equivalent = _quantized_fraction_equivalent(
                    left.onset,
                    right.onset,
                    ref_measure.divisions,
                    cand_measure.divisions,
                )
                counts["duration_correct"] = int(counts["duration_correct"]) + int(
                    duration_equivalent
                )
                counts["rhythm_correct"] = int(counts["rhythm_correct"]) + int(
                    _rhythm_equivalent(
                        left,
                        right,
                        left_divisions=ref_measure.divisions,
                        right_divisions=cand_measure.divisions,
                        left_chord_member=left_chord_flag,
                        right_chord_member=right_chord_flag,
                    )
                )
                counts["onset_correct"] = int(counts["onset_correct"]) + int(
                    onset_equivalent
                )
                counts["chord_correct"] = int(counts["chord_correct"]) + int(
                    left_chord_flag == right_chord_flag
                )
                counts["tuplet_topology_correct"] = int(counts["tuplet_topology_correct"]) + int(
                    left.tuple_ratio == right.tuple_ratio
                )
                counts["tuplet_event_matches"] = int(counts["tuplet_event_matches"]) + int(
                    left.tuple_ratio is not None and left.tuple_ratio == right.tuple_ratio
                )
                counts["tie_topology_correct"] = int(counts["tie_topology_correct"]) + int(
                    tuple(sorted(set(left.ties))) == tuple(sorted(set(right.ties)))
                )
            counts["tie_endpoint_matches"] = int(counts["tie_endpoint_matches"]) + len(
                set(left.ties) & set(right.ties)
            )
            counts["slur_endpoint_matches"] = int(counts["slur_endpoint_matches"]) + len(
                _slur_endpoint_kinds(left) & _slur_endpoint_kinds(right)
            )
            left_beams = (
                ref_beams[ref_index][pair.reference_index]
                if ref_index < len(ref_beams) and pair.reference_index < len(ref_beams[ref_index])
                else ()
            )
            right_beams = (
                cand_beams[cand_index][pair.candidate_index]
                if cand_index < len(cand_beams) and pair.candidate_index < len(cand_beams[cand_index])
                else ()
            )
            counts["beam_topology_correct"] = int(counts["beam_topology_correct"]) + int(
                left_beams == right_beams
            )
            counts["beam_marker_matches"] = int(counts["beam_marker_matches"]) + _counter_overlap(
                _beam_marker_counter(left_beams), _beam_marker_counter(right_beams)
            )
            left_articulations = _articulation_markers(left)
            right_articulations = _articulation_markers(right)
            counts["articulation_topology_correct"] = int(counts["articulation_topology_correct"]) + int(
                left_articulations == right_articulations
            )
            counts["articulation_marker_matches"] = int(counts["articulation_marker_matches"]) + _counter_overlap(
                left_articulations, right_articulations
            )
            left_ornaments = _ornament_markers(left)
            right_ornaments = _ornament_markers(right)
            counts["ornament_topology_correct"] = int(counts["ornament_topology_correct"]) + int(
                left_ornaments == right_ornaments
            )
            counts["ornament_marker_matches"] = int(counts["ornament_marker_matches"]) + _counter_overlap(
                left_ornaments, right_ornaments
            )
            counts["grace_topology_correct"] = int(counts["grace_topology_correct"]) + int(
                left.grace == right.grace
            )
            counts["grace_event_matches"] = int(counts["grace_event_matches"]) + int(
                left.grace and right.grace
            )
            ref_lyric = (
                ref_lyrics[ref_index][pair.reference_index]
                if ref_index < len(ref_lyrics)
                and ref_lyrics[ref_index] is not None
                and pair.reference_index < len(ref_lyrics[ref_index])
                else None
            )
            cand_lyric = (
                cand_lyrics[cand_index][pair.candidate_index]
                if cand_index < len(cand_lyrics)
                and cand_lyrics[cand_index] is not None
                and pair.candidate_index < len(cand_lyrics[cand_index])
                else None
            )
            counts["lyric_topology_correct"] = int(counts["lyric_topology_correct"]) + int(
                _lyric_state_key(ref_lyric) == _lyric_state_key(cand_lyric)
            )
            counts["lyric_event_matches"] = int(counts["lyric_event_matches"]) + int(
                ref_lyric is not None and _lyric_state_key(ref_lyric) == _lyric_state_key(cand_lyric)
            )
            counts["rest_correct"] = int(counts["rest_correct"]) + int(left.rest == right.rest)
        counts["weighted_event_cost"] = float(counts["weighted_event_cost"]) + alignment.total_cost
        per_measure.append({
            "reference_index": ref_index + 1,
            "candidate_index": cand_index + 1,
            "exact": exact,
            "preservation_exact": preservation_exact,
            "event_edits": local_edits,
            "weighted_event_cost": alignment.total_cost,
        })

    for ref_index in range(max(0, len(ref) - 1)):
        left_index = measure_alignment.reference_to_candidate[ref_index]
        right_index = measure_alignment.reference_to_candidate[ref_index + 1]
        if left_index is None or right_index is None or right_index != left_index + 1:
            continue
        reference_ties = _cross_tie_boundary_ties(ref[ref_index], ref[ref_index + 1])
        candidate_ties = _cross_tie_boundary_ties(cand[left_index], cand[right_index])
        counts["cross_tie_boundary_correct"] = int(counts["cross_tie_boundary_correct"]) + int(
            reference_ties == candidate_ties
        )
        counts["cross_tie_matches"] = int(counts["cross_tie_matches"]) + _counter_overlap(
            reference_ties,
            candidate_ties,
        )

    # Candidate-only measures contribute insertion events and direction false positives.
    for cand_index in measure_alignment.unmatched_candidate_indices:
        counts["inserted_events"] = int(counts["inserted_events"]) + len(cand[cand_index].notes)
        counts["weighted_event_cost"] = float(counts["weighted_event_cost"]) + len(cand[cand_index].notes)

    event_edits = int(counts["substituted_events"]) + int(counts["deleted_events"]) + int(counts["inserted_events"])
    reference_events = int(counts["reference_events"])
    candidate_events = int(counts["candidate_events"])
    deleted_events = int(counts["deleted_events"])
    inserted_events = int(counts["inserted_events"])
    event_presence_recall = _safe_rate(reference_events - deleted_events, reference_events)
    event_presence_precision = _safe_rate(candidate_events - inserted_events, candidate_events)
    event_presence_f1 = _safe_rate(
        2.0 * event_presence_precision * event_presence_recall,
        event_presence_precision + event_presence_recall,
    )
    ref_measures = int(counts["reference_measures"])
    direction_matches = int(counts["direction_matches"])
    direction_precision = _safe_rate(direction_matches, int(counts["candidate_direction_count"]), empty=1.0)
    direction_recall = _safe_rate(direction_matches, int(counts["reference_direction_count"]), empty=1.0)
    direction_content_matches = int(counts["direction_content_matches"])
    direction_content_precision = _safe_rate(
        direction_content_matches,
        int(counts["candidate_direction_count"]),
        empty=1.0,
    )
    direction_content_recall = _safe_rate(
        direction_content_matches,
        int(counts["reference_direction_count"]),
        empty=1.0,
    )
    direction_metric_values: dict[str, float] = {}
    for prefix in _DIRECTION_METRIC_FILTERS:
        reference_total = int(counts[f"reference_{prefix}_count"])
        candidate_total = int(counts[f"candidate_{prefix}_count"])
        exact_matches = int(counts[f"{prefix}_matches"])
        content_matches = int(counts[f"{prefix}_content_matches"])
        direction_metric_values[f"{prefix}_precision"] = _safe_rate(
            exact_matches, candidate_total, empty=1.0
        )
        direction_metric_values[f"{prefix}_recall"] = _safe_rate(
            exact_matches, reference_total, empty=1.0
        )
        direction_metric_values[f"{prefix}_f1"] = _marker_f1(
            exact_matches, reference_total, candidate_total
        )
        direction_metric_values[f"{prefix}_content_f1"] = _marker_f1(
            content_matches, reference_total, candidate_total
        )
        direction_metric_values[f"{prefix}_anchor_accuracy"] = (
            1.0
            if reference_total + candidate_total == 0
            else _safe_rate(exact_matches, content_matches, empty=0.0)
        )
    expression_matches = int(counts["expression_marker_matches"])
    expression_precision = _safe_rate(
        expression_matches, int(counts["candidate_expression_marker_count"]), empty=1.0
    )
    expression_recall = _safe_rate(
        expression_matches, int(counts["reference_expression_marker_count"]), empty=1.0
    )
    expression_f1 = _marker_f1(
        expression_matches,
        int(counts["reference_expression_marker_count"]),
        int(counts["candidate_expression_marker_count"]),
    )
    tuplet_matches = int(counts["tuplet_event_matches"])
    tuplet_precision = _safe_rate(
        tuplet_matches, int(counts["candidate_tuplet_event_count"]), empty=1.0
    )
    tuplet_recall = _safe_rate(
        tuplet_matches, int(counts["reference_tuplet_event_count"]), empty=1.0
    )
    tuplet_f1 = _marker_f1(
        tuplet_matches,
        int(counts["reference_tuplet_event_count"]),
        int(counts["candidate_tuplet_event_count"]),
    )
    tie_matches = int(counts["tie_endpoint_matches"])
    tie_precision = _safe_rate(
        tie_matches, int(counts["candidate_tie_endpoint_count"]), empty=1.0
    )
    tie_recall = _safe_rate(
        tie_matches, int(counts["reference_tie_endpoint_count"]), empty=1.0
    )
    tie_f1 = _marker_f1(
        tie_matches,
        int(counts["reference_tie_endpoint_count"]),
        int(counts["candidate_tie_endpoint_count"]),
    )
    slur_matches = int(counts["slur_endpoint_matches"])
    slur_precision = _safe_rate(
        slur_matches, int(counts["candidate_slur_endpoint_count"]), empty=1.0
    )
    slur_recall = _safe_rate(
        slur_matches, int(counts["reference_slur_endpoint_count"]), empty=1.0
    )
    slur_f1 = _marker_f1(
        slur_matches,
        int(counts["reference_slur_endpoint_count"]),
        int(counts["candidate_slur_endpoint_count"]),
    )
    beam_matches = int(counts["beam_marker_matches"])
    beam_precision = _safe_rate(
        beam_matches, int(counts["candidate_beam_marker_count"]), empty=1.0
    )
    beam_recall = _safe_rate(
        beam_matches, int(counts["reference_beam_marker_count"]), empty=1.0
    )
    beam_f1 = _marker_f1(
        beam_matches,
        int(counts["reference_beam_marker_count"]),
        int(counts["candidate_beam_marker_count"]),
    )
    articulation_matches = int(counts["articulation_marker_matches"])
    articulation_precision = _safe_rate(
        articulation_matches, int(counts["candidate_articulation_marker_count"]), empty=1.0
    )
    articulation_recall = _safe_rate(
        articulation_matches, int(counts["reference_articulation_marker_count"]), empty=1.0
    )
    articulation_f1 = _marker_f1(
        articulation_matches,
        int(counts["reference_articulation_marker_count"]),
        int(counts["candidate_articulation_marker_count"]),
    )
    ornament_matches = int(counts["ornament_marker_matches"])
    ornament_precision = _safe_rate(
        ornament_matches, int(counts["candidate_ornament_marker_count"]), empty=1.0
    )
    ornament_recall = _safe_rate(
        ornament_matches, int(counts["reference_ornament_marker_count"]), empty=1.0
    )
    ornament_f1 = _marker_f1(
        ornament_matches,
        int(counts["reference_ornament_marker_count"]),
        int(counts["candidate_ornament_marker_count"]),
    )
    accidental_matches = int(counts["accidental_marker_matches"])
    accidental_precision = _safe_rate(
        accidental_matches,
        int(counts["candidate_accidental_marker_count"]),
        empty=1.0,
    )
    accidental_recall = _safe_rate(
        accidental_matches,
        int(counts["reference_accidental_marker_count"]),
        empty=1.0,
    )
    accidental_f1 = _marker_f1(
        accidental_matches,
        int(counts["reference_accidental_marker_count"]),
        int(counts["candidate_accidental_marker_count"]),
    )
    grace_matches = int(counts["grace_event_matches"])
    grace_precision = _safe_rate(
        grace_matches, int(counts["candidate_grace_event_count"]), empty=1.0
    )
    grace_recall = _safe_rate(
        grace_matches, int(counts["reference_grace_event_count"]), empty=1.0
    )
    grace_f1 = _marker_f1(
        grace_matches,
        int(counts["reference_grace_event_count"]),
        int(counts["candidate_grace_event_count"]),
    )
    lyric_matches = int(counts["lyric_event_matches"])
    lyric_precision = _safe_rate(
        lyric_matches, int(counts["candidate_lyric_event_count"]), empty=1.0
    )
    lyric_recall = _safe_rate(
        lyric_matches, int(counts["reference_lyric_event_count"]), empty=1.0
    )
    lyric_f1 = _marker_f1(
        lyric_matches,
        int(counts["reference_lyric_event_count"]),
        int(counts["candidate_lyric_event_count"]),
    )
    cross_tie_matches = int(counts["cross_tie_matches"])
    cross_tie_precision = _safe_rate(
        cross_tie_matches, int(counts["candidate_cross_tie_count"]), empty=1.0
    )
    cross_tie_recall = _safe_rate(
        cross_tie_matches, int(counts["reference_cross_tie_count"]), empty=1.0
    )
    cross_tie_f1 = _marker_f1(
        cross_tie_matches,
        int(counts["reference_cross_tie_count"]),
        int(counts["candidate_cross_tie_count"]),
    )
    repeat_matches = int(counts["repeat_marker_matches"])
    repeat_precision = _safe_rate(
        repeat_matches, int(counts["candidate_repeat_marker_count"]), empty=1.0
    )
    repeat_recall = _safe_rate(
        repeat_matches, int(counts["reference_repeat_marker_count"]), empty=1.0
    )
    repeat_f1 = _marker_f1(
        repeat_matches,
        int(counts["reference_repeat_marker_count"]),
        int(counts["candidate_repeat_marker_count"]),
    )
    report: dict[str, object] = {
        "reference": str(reference),
        "candidate": str(candidate),
        "reference_sha256": sha256_file(reference),
        "candidate_sha256": sha256_file(candidate),
        "reference_measures": ref_measures,
        "candidate_measures": int(counts["candidate_measures"]),
        "measure_count_exact": len(ref) == len(cand),
        "measure_alignment_similarity": measure_alignment.similarity,
        "exact_measure_rate": _safe_rate(int(counts["exact_measures"]), int(counts["measure_denominator"])),
        "preservation_exact_measure_rate": _safe_rate(
            int(counts["preservation_exact_measures"]), int(counts["measure_denominator"])
        ),
        "event_error_rate": _safe_rate(event_edits, reference_events, empty=0.0),
        "weighted_event_error_rate": _safe_rate(float(counts["weighted_event_cost"]), reference_events, empty=0.0),
        "event_presence_precision": event_presence_precision,
        "event_presence_recall": event_presence_recall,
        "event_presence_f1": event_presence_f1,
        "deleted_event_rate": _safe_rate(deleted_events, reference_events, empty=0.0),
        "inserted_event_rate": _safe_rate(inserted_events, candidate_events, empty=0.0),
        "pitch_accuracy_aligned": _safe_rate(int(counts["pitch_correct"]), int(counts["reference_pitched_events"])),
        "duration_accuracy_aligned": _safe_rate(int(counts["duration_correct"]), int(counts["reference_rhythm_events"])),
        "rhythm_accuracy_aligned": _safe_rate(int(counts["rhythm_correct"]), int(counts["reference_rhythm_events"])),
        "onset_accuracy_aligned": _safe_rate(int(counts["onset_correct"]), int(counts["reference_rhythm_events"])),
        "chord_topology_accuracy_aligned": _safe_rate(int(counts["chord_correct"]), int(counts["reference_rhythm_events"])),
        "tuplet_topology_accuracy_aligned": _safe_rate(
            int(counts["tuplet_topology_correct"]), int(counts["reference_rhythm_events"])
        ),
        "tuplet_event_precision": tuplet_precision,
        "tuplet_event_recall": tuplet_recall,
        "tuplet_event_f1": tuplet_f1,
        "tie_topology_accuracy_aligned": _safe_rate(
            int(counts["tie_topology_correct"]), int(counts["reference_rhythm_events"])
        ),
        "tie_endpoint_precision": tie_precision,
        "tie_endpoint_recall": tie_recall,
        "tie_endpoint_f1": tie_f1,
        "slur_topology_accuracy": _safe_rate(
            int(counts["slur_topology_correct"]), ref_measures
        ),
        "slur_endpoint_precision": slur_precision,
        "slur_endpoint_recall": slur_recall,
        "slur_endpoint_f1": slur_f1,
        "beam_topology_accuracy_aligned": _safe_rate(
            int(counts["beam_topology_correct"]), int(counts["matched_events"])
        ),
        "beam_marker_precision": beam_precision,
        "beam_marker_recall": beam_recall,
        "beam_marker_f1": beam_f1,
        "articulation_topology_accuracy_aligned": _safe_rate(
            int(counts["articulation_topology_correct"]), int(counts["matched_events"])
        ),
        "articulation_marker_precision": articulation_precision,
        "articulation_marker_recall": articulation_recall,
        "articulation_marker_f1": articulation_f1,
        "ornament_topology_accuracy_aligned": _safe_rate(
            int(counts["ornament_topology_correct"]), int(counts["matched_events"])
        ),
        "ornament_marker_precision": ornament_precision,
        "ornament_marker_recall": ornament_recall,
        "ornament_marker_f1": ornament_f1,
        "accidental_marker_precision": accidental_precision,
        "accidental_marker_recall": accidental_recall,
        "accidental_marker_f1": accidental_f1,
        "grace_topology_accuracy_aligned": _safe_rate(
            int(counts["grace_topology_correct"]), int(counts["matched_events"])
        ),
        "grace_event_precision": grace_precision,
        "grace_event_recall": grace_recall,
        "grace_event_f1": grace_f1,
        "lyric_topology_accuracy_aligned": _safe_rate(
            int(counts["lyric_topology_correct"]), int(counts["matched_events"])
        ),
        "lyric_event_precision": lyric_precision,
        "lyric_event_recall": lyric_recall,
        "lyric_event_f1": lyric_f1,
        "cross_tie_boundary_accuracy": _safe_rate(
            int(counts["cross_tie_boundary_correct"]),
            int(counts["reference_cross_tie_boundary_count"]),
        ),
        "cross_tie_precision": cross_tie_precision,
        "cross_tie_recall": cross_tie_recall,
        "cross_tie_f1": cross_tie_f1,
        "event_kind_accuracy_aligned": _safe_rate(int(counts["rest_correct"]), int(counts["matched_events"])),
        "voice_assignment_accuracy_aligned": _safe_rate(
            int(counts["voice_assignment_correct"]),
            int(counts["reference_voice_assignments"]),
        ),
        "staff_assignment_accuracy_aligned": _safe_rate(
            int(counts["staff_assignment_correct"]),
            int(counts["reference_staff_assignments"]),
        ),
        "voice_topology_accuracy": _safe_rate(
            int(counts["voice_topology_correct"]),
            ref_measures,
        ),
        "direction_precision": direction_precision,
        "direction_recall": direction_recall,
        "direction_f1": _marker_f1(
            direction_matches,
            int(counts["reference_direction_count"]),
            int(counts["candidate_direction_count"]),
        ),
        "direction_content_precision": direction_content_precision,
        "direction_content_recall": direction_content_recall,
        "direction_content_f1": _marker_f1(
            direction_content_matches,
            int(counts["reference_direction_count"]),
            int(counts["candidate_direction_count"]),
        ),
        "direction_anchor_accuracy": (
            1.0
            if int(counts["reference_direction_count"]) + int(counts["candidate_direction_count"]) == 0
            else _safe_rate(direction_matches, direction_content_matches, empty=0.0)
        ),
        **direction_metric_values,
        "expression_marker_precision": expression_precision,
        "expression_marker_recall": expression_recall,
        "expression_marker_f1": expression_f1,
        "time_signature_accuracy": _safe_rate(int(counts["time_signature_correct"]), ref_measures),
        "key_signature_accuracy": _safe_rate(int(counts["key_signature_correct"]), ref_measures),
        "clef_accuracy": _safe_rate(int(counts["clef_correct"]), ref_measures),
        "staff_clef_accuracy": _safe_rate(int(counts["staff_clef_correct"]), ref_measures),
        "barline_accuracy": _safe_rate(int(counts["barline_correct"]), ref_measures),
        "repeat_marker_precision": repeat_precision,
        "repeat_marker_recall": repeat_recall,
        "repeat_marker_f1": repeat_f1,
        "reference_event_count": reference_events,
        "candidate_event_count": int(counts["candidate_events"]),
        "counts": counts,
        "measure_alignment": {
            "reference_to_candidate": [None if value is None else value + 1 for value in measure_alignment.reference_to_candidate],
            "unmatched_candidate_indices": [value + 1 for value in measure_alignment.unmatched_candidate_indices],
            "normalized_cost": measure_alignment.normalized_cost,
        },
        "per_measure": per_measure,
    }
    # A bounded utility score is useful for ordering experiments but is not an accuracy
    # probability and must not replace the individual release metrics.
    report["utility_score"] = max(
        0.0,
        min(
            1.0,
            0.28 * float(report["pitch_accuracy_aligned"])
            + 0.24 * float(report["rhythm_accuracy_aligned"])
            + 0.04 * float(report["chord_topology_accuracy_aligned"])
            + 0.03 * float(report["tuplet_event_f1"])
            + 0.03 * float(report["tie_endpoint_f1"])
            + 0.02 * float(report["slur_endpoint_f1"])
            + 0.01 * float(report["beam_marker_f1"])
            + 0.01 * float(report["articulation_marker_f1"])
            + 0.01 * float(report["ornament_marker_f1"])
            + 0.02 * float(report["grace_event_f1"])
            + 0.02 * float(report["lyric_event_f1"])
            + 0.01 * float(report["cross_tie_f1"])
            + 0.05 * float(report["exact_measure_rate"])
            + 0.05 * float(report["preservation_exact_measure_rate"])
            + 0.10 * float(report["time_signature_accuracy"])
            + 0.06 * float(report["direction_f1"])
            + 0.01 * float(report["expression_marker_f1"])
            + 0.01 * float(report["repeat_marker_f1"]),
        ),
    )
    return report


def _normalized_part_name(tree: etree._ElementTree, part_id: str) -> str:
    for item in tree.getroot().xpath("./*[local-name()='part-list']/*[local-name()='score-part']"):
        if str(item.get("id") or "") != part_id:
            continue
        names = item.xpath("./*[local-name()='part-name']/text()")
        return " ".join(str(names[0]).split()).casefold() if names else ""
    return ""


def _part_staff_count(part: etree._Element | None) -> int:
    if part is None:
        return 0
    maximum = 1
    for node in part.xpath(
        "./*[local-name()='measure']/*[local-name()='attributes']/*[local-name()='staves']"
    ):
        try:
            maximum = max(maximum, int(str(node.text or "1")))
        except ValueError:
            continue
    for node in part.xpath(
        "./*[local-name()='measure']/*[local-name()='note']/*[local-name()='staff']"
    ):
        try:
            maximum = max(maximum, int(str(node.text or "1")))
        except ValueError:
            continue
    return maximum


def _write_isolated_part(
    source_tree: etree._ElementTree,
    part: etree._Element | None,
    destination: Path,
    *,
    empty_id: str,
) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_id = str(part.get("id") or empty_id) if part is not None else empty_id
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id=part_id)
    etree.SubElement(score_part, "part-name").text = (
        _normalized_part_name(source_tree, part_id) or "Music"
    )
    if part is None:
        etree.SubElement(root, "part", id=part_id)
    else:
        root.append(deepcopy(part))
    etree.ElementTree(root).write(
        str(destination),
        encoding="utf-8",
        xml_declaration=True,
    )


def _sum_part_counts(part_reports: Sequence[dict[str, object]]) -> dict[str, int | float]:
    combined: dict[str, int | float] = {}
    for report in part_reports:
        for key, value in dict(report["counts"]).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            combined[key] = combined.get(key, 0) + value
    return combined


def _f1(precision: float, recall: float) -> float:
    return _safe_rate(2.0 * precision * recall, precision + recall, empty=1.0)


def _aggregate_full_score_report(
    reference: Path,
    candidate: Path,
    part_reports: Sequence[dict[str, object]],
    part_mappings: Sequence[dict[str, object]],
    *,
    reference_staff_count: int,
    candidate_staff_count: int,
) -> dict[str, object]:
    counts = _sum_part_counts(part_reports)

    def count(name: str) -> int | float:
        return counts.get(name, 0)

    def rate(numerator: str, denominator: str, *, empty: float = 1.0) -> float:
        return _safe_rate(count(numerator), count(denominator), empty=empty)

    event_edits = (
        int(count("substituted_events"))
        + int(count("deleted_events"))
        + int(count("inserted_events"))
    )
    reference_events = int(count("reference_events"))
    candidate_events = int(count("candidate_events"))
    event_presence_recall = _safe_rate(
        reference_events - int(count("deleted_events")),
        reference_events,
    )
    event_presence_precision = _safe_rate(
        candidate_events - int(count("inserted_events")),
        candidate_events,
    )

    marker_specs = {
        "direction": ("direction_matches", "reference_direction_count", "candidate_direction_count"),
        "direction_content": (
            "direction_content_matches",
            "reference_direction_count",
            "candidate_direction_count",
        ),
        "expression_marker": (
            "expression_marker_matches",
            "reference_expression_marker_count",
            "candidate_expression_marker_count",
        ),
        "tuplet_event": (
            "tuplet_event_matches",
            "reference_tuplet_event_count",
            "candidate_tuplet_event_count",
        ),
        "tie_endpoint": (
            "tie_endpoint_matches",
            "reference_tie_endpoint_count",
            "candidate_tie_endpoint_count",
        ),
        "slur_endpoint": (
            "slur_endpoint_matches",
            "reference_slur_endpoint_count",
            "candidate_slur_endpoint_count",
        ),
        "beam_marker": (
            "beam_marker_matches",
            "reference_beam_marker_count",
            "candidate_beam_marker_count",
        ),
        "articulation_marker": (
            "articulation_marker_matches",
            "reference_articulation_marker_count",
            "candidate_articulation_marker_count",
        ),
        "ornament_marker": (
            "ornament_marker_matches",
            "reference_ornament_marker_count",
            "candidate_ornament_marker_count",
        ),
        "accidental_marker": (
            "accidental_marker_matches",
            "reference_accidental_marker_count",
            "candidate_accidental_marker_count",
        ),
        "grace_event": (
            "grace_event_matches",
            "reference_grace_event_count",
            "candidate_grace_event_count",
        ),
        "lyric_event": (
            "lyric_event_matches",
            "reference_lyric_event_count",
            "candidate_lyric_event_count",
        ),
        "cross_tie": (
            "cross_tie_matches",
            "reference_cross_tie_count",
            "candidate_cross_tie_count",
        ),
        "repeat_marker": (
            "repeat_marker_matches",
            "reference_repeat_marker_count",
            "candidate_repeat_marker_count",
        ),
    }
    for prefix in _DIRECTION_METRIC_FILTERS:
        marker_specs[prefix] = (
            f"{prefix}_matches",
            f"reference_{prefix}_count",
            f"candidate_{prefix}_count",
        )
        marker_specs[f"{prefix}_content"] = (
            f"{prefix}_content_matches",
            f"reference_{prefix}_count",
            f"candidate_{prefix}_count",
        )
    marker_metrics: dict[str, float] = {}
    for prefix, (matches, reference_total, candidate_total) in marker_specs.items():
        precision = rate(matches, candidate_total)
        recall = rate(matches, reference_total)
        marker_metrics[f"{prefix}_precision"] = precision
        marker_metrics[f"{prefix}_recall"] = recall
        marker_metrics[f"{prefix}_f1"] = _marker_f1(
            count(matches),
            count(reference_total),
            count(candidate_total),
        )
    marker_metrics["direction_anchor_accuracy"] = (
        1.0
        if count("reference_direction_count") + count("candidate_direction_count") == 0
        else _safe_rate(
            count("direction_matches"),
            count("direction_content_matches"),
            empty=0.0,
        )
    )
    for prefix in _DIRECTION_METRIC_FILTERS:
        marker_metrics[f"{prefix}_anchor_accuracy"] = (
            1.0
            if count(f"reference_{prefix}_count") + count(f"candidate_{prefix}_count") == 0
            else _safe_rate(
                count(f"{prefix}_matches"),
                count(f"{prefix}_content_matches"),
                empty=0.0,
            )
        )

    matched_parts = sum(bool(item["reference_present"] and item["candidate_present"]) for item in part_mappings)
    identity_matches = sum(bool(item["identity_match"]) for item in part_mappings)
    ordered_matches = sum(
        bool(
            item["reference_present"]
            and item["candidate_present"]
            and item["reference_index"] == item["candidate_index"]
        )
        for item in part_mappings
    )
    reference_parts = sum(bool(item["reference_present"]) for item in part_mappings)
    candidate_parts = sum(bool(item["candidate_present"]) for item in part_mappings)
    staff_overlap = sum(
        min(int(item["reference_staff_count"]), int(item["candidate_staff_count"]))
        for item in part_mappings
        if item["reference_present"] and item["candidate_present"]
    )
    staff_precision = _safe_rate(staff_overlap, candidate_staff_count)
    staff_recall = _safe_rate(staff_overlap, reference_staff_count)

    report: dict[str, object] = {
        "schema": "scorescan-full-score-evaluation@1",
        "reference": str(reference),
        "candidate": str(candidate),
        "reference_sha256": sha256_file(reference),
        "candidate_sha256": sha256_file(candidate),
        "reference_parts": reference_parts,
        "candidate_parts": candidate_parts,
        "matched_parts": matched_parts,
        "part_count_exact": reference_parts == candidate_parts,
        "part_mapping_accuracy": _safe_rate(identity_matches, reference_parts),
        "part_order_accuracy": _safe_rate(ordered_matches, reference_parts),
        "reference_staff_count": reference_staff_count,
        "candidate_staff_count": candidate_staff_count,
        "staff_count_exact": reference_staff_count == candidate_staff_count,
        "staff_topology_precision": staff_precision,
        "staff_topology_recall": staff_recall,
        "staff_topology_f1": _f1(staff_precision, staff_recall),
        "reference_measures": int(count("reference_measures")),
        "candidate_measures": int(count("candidate_measures")),
        "measure_count_exact": all(bool(item["measure_count_exact"]) for item in part_reports),
        "measure_alignment_similarity": _safe_rate(
            sum(
                float(item["measure_alignment_similarity"])
                * int(dict(item["counts"])["measure_denominator"])
                for item in part_reports
            ),
            int(count("measure_denominator")),
        ),
        "exact_measure_rate": rate("exact_measures", "measure_denominator"),
        "preservation_exact_measure_rate": rate(
            "preservation_exact_measures", "measure_denominator"
        ),
        "event_error_rate": _safe_rate(event_edits, reference_events, empty=0.0),
        "weighted_event_error_rate": _safe_rate(
            float(count("weighted_event_cost")), reference_events, empty=0.0
        ),
        "event_presence_precision": event_presence_precision,
        "event_presence_recall": event_presence_recall,
        "event_presence_f1": _f1(event_presence_precision, event_presence_recall),
        "deleted_event_rate": _safe_rate(
            int(count("deleted_events")), reference_events, empty=0.0
        ),
        "inserted_event_rate": _safe_rate(
            int(count("inserted_events")), candidate_events, empty=0.0
        ),
        "pitch_accuracy_aligned": rate("pitch_correct", "reference_pitched_events"),
        "duration_accuracy_aligned": rate("duration_correct", "reference_rhythm_events"),
        "rhythm_accuracy_aligned": rate("rhythm_correct", "reference_rhythm_events"),
        "onset_accuracy_aligned": rate("onset_correct", "reference_rhythm_events"),
        "chord_topology_accuracy_aligned": rate("chord_correct", "reference_rhythm_events"),
        "tuplet_topology_accuracy_aligned": rate(
            "tuplet_topology_correct", "reference_rhythm_events"
        ),
        "tie_topology_accuracy_aligned": rate(
            "tie_topology_correct", "reference_rhythm_events"
        ),
        "slur_topology_accuracy": rate("slur_topology_correct", "reference_measures"),
        "beam_topology_accuracy_aligned": rate("beam_topology_correct", "matched_events"),
        "articulation_topology_accuracy_aligned": rate(
            "articulation_topology_correct", "matched_events"
        ),
        "ornament_topology_accuracy_aligned": rate(
            "ornament_topology_correct", "matched_events"
        ),
        "grace_topology_accuracy_aligned": rate("grace_topology_correct", "matched_events"),
        "lyric_topology_accuracy_aligned": rate("lyric_topology_correct", "matched_events"),
        "cross_tie_boundary_accuracy": rate(
            "cross_tie_boundary_correct", "reference_cross_tie_boundary_count"
        ),
        "event_kind_accuracy_aligned": rate("rest_correct", "matched_events"),
        "voice_assignment_accuracy_aligned": rate(
            "voice_assignment_correct", "reference_voice_assignments"
        ),
        "staff_assignment_accuracy_aligned": rate(
            "staff_assignment_correct", "reference_staff_assignments"
        ),
        "voice_topology_accuracy": rate("voice_topology_correct", "reference_measures"),
        "time_signature_accuracy": rate("time_signature_correct", "reference_measures"),
        "key_signature_accuracy": rate("key_signature_correct", "reference_measures"),
        "clef_accuracy": rate("clef_correct", "reference_measures"),
        "staff_clef_accuracy": rate("staff_clef_correct", "reference_measures"),
        "barline_accuracy": rate("barline_correct", "reference_measures"),
        "reference_event_count": reference_events,
        "candidate_event_count": candidate_events,
        "counts": counts,
        "part_mappings": list(part_mappings),
        "parts": list(part_reports),
    }
    report.update(marker_metrics)
    report["utility_score"] = max(
        0.0,
        min(
            1.0,
            0.25 * float(report["pitch_accuracy_aligned"])
            + 0.20 * float(report["rhythm_accuracy_aligned"])
            + 0.05 * float(report["event_presence_f1"])
            + 0.04 * float(report["voice_topology_accuracy"])
            + 0.04 * float(report["staff_topology_f1"])
            + 0.03 * float(report["part_mapping_accuracy"])
            + 0.03 * float(report["chord_topology_accuracy_aligned"])
            + 0.03 * float(report["tuplet_event_f1"])
            + 0.04 * float(report["tie_endpoint_f1"])
            + 0.04 * float(report["slur_endpoint_f1"])
            + 0.02 * float(report["beam_marker_f1"])
            + 0.02 * float(report["articulation_marker_f1"])
            + 0.02 * float(report["ornament_marker_f1"])
            + 0.04 * float(report["exact_measure_rate"])
            + 0.03 * float(report["preservation_exact_measure_rate"])
            + 0.05 * float(report["time_signature_accuracy"])
            + 0.04 * float(report["direction_f1"]),
        ),
    )
    return report


def compare_musicxml(reference: Path, candidate: Path) -> dict[str, object]:
    """Compare complete MusicXML scores without aligning instruments by x position.

    A single-part score uses the established report verbatim.  Full scores are
    evaluated part-by-part in written score order; each part receives an independent
    measure/event alignment and is then micro-aggregated.  Missing parts therefore
    remain omissions rather than shifting every later instrument.
    """

    parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=False, huge_tree=False)
    reference_tree = etree.parse(str(reference), parser)
    candidate_tree = etree.parse(str(candidate), parser)
    reference_parts = list(reference_tree.getroot().xpath("./*[local-name()='part']"))
    candidate_parts = list(candidate_tree.getroot().xpath("./*[local-name()='part']"))
    if len(reference_parts) <= 1 and len(candidate_parts) <= 1:
        return _compare_single_part_musicxml(reference, candidate)

    part_reports: list[dict[str, object]] = []
    mappings: list[dict[str, object]] = []
    reference_descriptors = [
        (
            str(part.get("id") or ""),
            _normalized_part_name(reference_tree, str(part.get("id") or "")),
        )
        for part in reference_parts
    ]
    candidate_descriptors = [
        (
            str(part.get("id") or ""),
            _normalized_part_name(candidate_tree, str(part.get("id") or "")),
        )
        for part in candidate_parts
    ]
    unused_candidates = set(range(len(candidate_parts)))
    pairings: list[tuple[int | None, int | None]] = []
    for reference_index, (reference_id, reference_name) in enumerate(reference_descriptors):
        selected: int | None = None
        if reference_name:
            same_name = [
                index
                for index in sorted(unused_candidates)
                if candidate_descriptors[index][1] == reference_name
            ]
            if len(same_name) == 1:
                selected = same_name[0]
        if selected is None and reference_id:
            same_id = [
                index
                for index in sorted(unused_candidates)
                if candidate_descriptors[index][0] == reference_id
            ]
            if len(same_id) == 1:
                selected = same_id[0]
        if selected is None and reference_index in unused_candidates:
            # Written score order is a valid coarse structural anchor.  It is used
            # only after explicit instrument identity fails, and identity accuracy
            # remains false so this fallback cannot hide a part-mapping error.
            selected = reference_index
        if selected is not None:
            unused_candidates.remove(selected)
        pairings.append((reference_index, selected))
    pairings.extend((None, index) for index in sorted(unused_candidates))

    with tempfile.TemporaryDirectory(prefix="scorescan-full-score-eval-") as temporary:
        temp_root = Path(temporary)
        for index, (reference_index, candidate_index) in enumerate(pairings):
            reference_part = (
                reference_parts[reference_index] if reference_index is not None else None
            )
            candidate_part = (
                candidate_parts[candidate_index] if candidate_index is not None else None
            )
            reference_id = str(reference_part.get("id") or "") if reference_part is not None else ""
            candidate_id = str(candidate_part.get("id") or "") if candidate_part is not None else ""
            reference_name = (
                _normalized_part_name(reference_tree, reference_id) if reference_part is not None else ""
            )
            candidate_name = (
                _normalized_part_name(candidate_tree, candidate_id) if candidate_part is not None else ""
            )
            identity_match = bool(
                reference_part is not None
                and candidate_part is not None
                and (
                    (reference_name and reference_name == candidate_name)
                    or (not reference_name and reference_id and reference_id == candidate_id)
                )
            )
            reference_path = temp_root / f"reference_{index + 1}.musicxml"
            candidate_path = temp_root / f"candidate_{index + 1}.musicxml"
            _write_isolated_part(
                reference_tree,
                reference_part,
                reference_path,
                empty_id=f"REF_EMPTY_{index + 1}",
            )
            _write_isolated_part(
                candidate_tree,
                candidate_part,
                candidate_path,
                empty_id=f"CAND_EMPTY_{index + 1}",
            )
            part_report = _compare_single_part_musicxml(reference_path, candidate_path)
            part_report.update(
                {
                    "part_index": index + 1,
                    "reference_part_index": (
                        reference_index + 1 if reference_index is not None else None
                    ),
                    "candidate_part_index": (
                        candidate_index + 1 if candidate_index is not None else None
                    ),
                    "reference_part_id": reference_id or None,
                    "candidate_part_id": candidate_id or None,
                    "reference_part_name": reference_name or None,
                    "candidate_part_name": candidate_name or None,
                }
            )
            # Temporary paths and hashes describe the isolated evaluation artifact,
            # not a user-visible source.  Keep only semantically useful detail.
            for key in ("reference", "candidate", "reference_sha256", "candidate_sha256"):
                part_report.pop(key, None)
            part_reports.append(part_report)
            mappings.append(
                {
                    "index": index + 1,
                    "reference_index": (
                        reference_index + 1 if reference_index is not None else None
                    ),
                    "candidate_index": (
                        candidate_index + 1 if candidate_index is not None else None
                    ),
                    "reference_present": reference_part is not None,
                    "candidate_present": candidate_part is not None,
                    "reference_part_id": reference_id or None,
                    "candidate_part_id": candidate_id or None,
                    "reference_part_name": reference_name or None,
                    "candidate_part_name": candidate_name or None,
                    "identity_match": identity_match,
                    "reference_staff_count": _part_staff_count(reference_part),
                    "candidate_staff_count": _part_staff_count(candidate_part),
                }
            )

    return _aggregate_full_score_report(
        reference,
        candidate,
        part_reports,
        mappings,
        reference_staff_count=sum(_part_staff_count(part) for part in reference_parts),
        candidate_staff_count=sum(_part_staff_count(part) for part in candidate_parts),
    )


def benchmark_fingerprint(cases: Sequence[tuple[str, Path, Path]]) -> str:
    digest = hashlib.sha256()
    for case_id, reference, candidate in sorted(cases, key=lambda item: item[0]):
        digest.update(case_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(reference).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256_file(candidate).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()
