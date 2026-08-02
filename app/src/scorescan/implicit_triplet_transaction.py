from __future__ import annotations

"""Fail-closed MusicXML transaction for a confirmed continuous triplet grid.

This module does not decide whether a scanned score is in cut time and does not run
automatically.  Its caller must provide independently confirmed meter evidence.  The
transaction then repairs one narrow exporter failure: a continuous stream of twelve
eighth-note triplet slots was serialized as twelve ordinary eighth/quarter events,
causing a real 2/2 measure to be inferred as 3/2.

The global support check is intentionally strict.  A single measure that happens to
contain twelve notes is never enough to authorize a score-wide rewrite.
"""

import math
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

from lxml import etree

from .util import atomic_write_bytes

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


@dataclass(frozen=True)
class TimedNote:
    node: etree._Element
    onset: Fraction
    duration: Fraction
    chord: bool
    grace: bool


@dataclass(frozen=True)
class TimedDirection:
    node: etree._Element
    onset: Fraction


@dataclass(frozen=True)
class MeasureGrid:
    measure: etree._Element
    number: str
    divisions: int
    notes: tuple[TimedNote, ...]
    directions: tuple[TimedDirection, ...]
    slot_onsets: tuple[Fraction, ...]
    trailing_onsets: tuple[Fraction, ...]
    eighth_like_slots: int


@dataclass(frozen=True)
class TripletTransactionReport:
    applied: bool
    reason: str
    confirmed_meter: tuple[int, int]
    expected_slots: int
    parts_seen: int
    regular_measures: int
    supported_measures: int
    support_ratio: float
    measures_repaired: int
    notes_converted: int
    notes_cloned: int
    notes_already_correct: int
    time_signatures_rewritten: int
    repaired_measure_numbers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TripletGridEvidence:
    authorized: bool
    reason: str
    inferred_meter: tuple[int, int] | None
    parts_seen: int
    regular_measures: int
    grid_measures: int
    grid_support_ratio: float
    explicit_triplet_slots: int
    explicit_triplet_measures: int
    fully_explicit_grid_measures: int
    fully_unmarked_grid_measures: int
    plain_four_quarter_coda_measures: int
    source_time_signatures: tuple[tuple[int, int], ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _integer(value: str | None, default: int = 0) -> int:
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def _insert_note_child(note: etree._Element, child: etree._Element) -> None:
    rank = _NOTE_CHILD_ORDER.get(str(child.tag), 99)
    insertion = len(note)
    for index, existing in enumerate(note):
        if _NOTE_CHILD_ORDER.get(str(existing.tag), 99) > rank:
            insertion = index
            break
    note.insert(insertion, child)


def _set_note_child_text(note: etree._Element, tag: str, value: str) -> etree._Element:
    child = note.find(tag)
    if child is None:
        child = etree.Element(tag)
        _insert_note_child(note, child)
    child.text = value
    return child


def _parse_measure_timeline(
    measure: etree._Element,
    inherited_divisions: int,
) -> tuple[int, tuple[TimedNote, ...], tuple[TimedDirection, ...]]:
    divisions = inherited_divisions
    for attributes in measure.findall("attributes"):
        divisions = max(1, _integer(attributes.findtext("divisions"), divisions))

    cursor = Fraction(0, 1)
    last_anchor = Fraction(0, 1)
    notes: list[TimedNote] = []
    directions: list[TimedDirection] = []
    for child in measure:
        if child.tag == "backup":
            cursor = max(
                Fraction(0, 1),
                cursor - Fraction(_integer(child.findtext("duration")), divisions),
            )
            continue
        if child.tag == "forward":
            cursor += Fraction(_integer(child.findtext("duration")), divisions)
            continue
        if child.tag == "direction":
            offset = Fraction(_integer(child.findtext("offset")), divisions)
            directions.append(TimedDirection(child, max(Fraction(0, 1), cursor + offset)))
            continue
        if child.tag != "note":
            continue
        chord = child.find("chord") is not None
        grace = child.find("grace") is not None
        duration = Fraction(_integer(child.findtext("duration")), divisions)
        onset = last_anchor if chord else cursor
        if not chord:
            last_anchor = onset
        notes.append(TimedNote(child, onset, duration, chord, grace))
        if not chord and not grace:
            cursor += duration
    return divisions, tuple(notes), tuple(directions)


def _pitch_value(note: etree._Element) -> int:
    pitch = note.find("pitch")
    if pitch is None:
        return -100_000
    steps = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    step = steps.get((pitch.findtext("step") or "").strip().upper(), 0)
    octave = _integer(pitch.findtext("octave"), 4)
    alter_text = (pitch.findtext("alter") or "0").strip()
    try:
        alter = int(Fraction(alter_text))
    except (ValueError, ZeroDivisionError):
        alter = 0
    return 12 * (octave + 1) + step + alter


def _is_triplet(note: etree._Element) -> bool:
    modification = note.find("time-modification")
    return bool(
        modification is not None
        and _integer(modification.findtext("actual-notes")) == 3
        and _integer(modification.findtext("normal-notes")) == 2
    )


def _is_ordinary_eighth(item: TimedNote) -> bool:
    return bool(
        not item.grace
        and item.duration == Fraction(1, 2)
        and (item.node.findtext("type") or "").strip() == "eighth"
        and not _is_triplet(item.node)
    )


def _candidate_grid(
    measure: etree._Element,
    divisions: int,
    *,
    expected_slots: int,
) -> MeasureGrid | None:
    parsed_divisions, notes, directions = _parse_measure_timeline(measure, divisions)
    groups: dict[Fraction, list[TimedNote]] = defaultdict(list)
    for note in notes:
        if not note.grace and note.duration > 0:
            groups[note.onset].append(note)
    onsets = tuple(sorted(groups))
    if len(onsets) not in {expected_slots, expected_slots + 1} or not onsets or onsets[0] != 0:
        return None

    slot_onsets = onsets[:expected_slots]
    trailing = onsets[expected_slots:]
    if trailing:
        trailing_notes = groups[trailing[0]]
        if not trailing_notes or any(
            (item.node.findtext("type") or "").strip() not in {"16th", "32nd"}
            for item in trailing_notes
        ):
            return None

    eighth_like = 0
    for onset in slot_onsets:
        if any(_is_triplet(item.node) or _is_ordinary_eighth(item) for item in groups[onset]):
            eighth_like += 1
    if eighth_like < max(8, expected_slots - 4):
        return None
    if slot_onsets[-1] <= Fraction(expected_slots - 1, 3) - Fraction(1, 3):
        return None
    return MeasureGrid(
        measure=measure,
        number=(measure.get("number") or "").strip(),
        divisions=parsed_divisions,
        notes=notes,
        directions=directions,
        slot_onsets=slot_onsets,
        trailing_onsets=trailing,
        eighth_like_slots=eighth_like,
    )


def _plain_four_quarter_measure(
    notes: tuple[TimedNote, ...],
) -> bool:
    regular = [item for item in notes if not item.grace and item.duration > 0]
    if not regular or any(_is_triplet(item.node) for item in regular):
        return False
    allowed_types = {"whole", "half", "quarter"}
    if any((item.node.findtext("type") or "").strip() not in allowed_types for item in regular):
        return False
    return max((item.onset + item.duration for item in regular), default=Fraction(0, 1)) == 4


def detect_continuous_triplet_grid_evidence(input_path: Path) -> TripletGridEvidence:
    """Detect one narrow, internally overdetermined 3/2-inference failure.

    This is not a general time-signature classifier.  Authorization requires the
    denominator-2 candidate, repeated marked and unmarked versions of the same
    twelve-slot texture, at least two completely recognised triplet measures that
    span exactly four quarters, and at least two non-grid coda measures that also
    span exactly four quarters.  Missing any one evidence family causes abstention.
    """

    parser = etree.XMLParser(remove_blank_text=True, resolve_entities=False, no_network=True)
    tree = etree.parse(str(input_path), parser)
    root = tree.getroot()
    parts = root.findall("part")
    regular_measures = 0
    grids: list[MeasureGrid] = []
    source_times: list[tuple[int, int]] = []
    nongrid_timelines: list[tuple[TimedNote, ...]] = []

    for part in parts:
        inherited_divisions = 1
        inherited_meter: tuple[int, int] | None = None
        for measure in part.findall("measure"):
            for attributes in measure.findall("attributes"):
                inherited_divisions = max(
                    1,
                    _integer(attributes.findtext("divisions"), inherited_divisions),
                )
                time_node = attributes.find("time")
                if time_node is not None and time_node.find("senza-misura") is None:
                    inherited_meter = (
                        _integer(time_node.findtext("beats"), 4),
                        _integer(time_node.findtext("beat-type"), 4),
                    )
                    source_times.append(inherited_meter)
            parsed_divisions, notes, _directions = _parse_measure_timeline(
                measure,
                inherited_divisions,
            )
            if notes:
                regular_measures += 1
            if inherited_meter != (3, 2):
                if notes:
                    nongrid_timelines.append(notes)
                continue
            grid = _candidate_grid(measure, parsed_divisions, expected_slots=12)
            if grid is None:
                if notes:
                    nongrid_timelines.append(notes)
                continue
            grids.append(grid)

    explicit_slots = 0
    explicit_measures = 0
    fully_explicit = 0
    fully_unmarked = 0
    for grid in grids:
        by_onset: dict[Fraction, list[TimedNote]] = defaultdict(list)
        for note in grid.notes:
            if not note.grace and note.duration > 0:
                by_onset[note.onset].append(note)
        flags = [
            any(_is_triplet(item.node) for item in by_onset[onset])
            for onset in grid.slot_onsets
        ]
        explicit_slots += sum(flags)
        explicit_measures += int(any(flags))
        span = grid.slot_onsets[-1] + (grid.slot_onsets[-1] - grid.slot_onsets[-2])
        fully_explicit += int(all(flags) and span == 4)
        fully_unmarked += int(not any(flags) and span == 6)

    plain_coda = sum(_plain_four_quarter_measure(notes) for notes in nongrid_timelines)
    support_ratio = len(grids) / max(1, regular_measures)
    unique_times = tuple(dict.fromkeys(source_times))
    checks = {
        "one_part": len(parts) == 1,
        "single_denominator_two_meter": unique_times == ((3, 2),),
        "minimum_grid_measures": len(grids) >= 8,
        "grid_support": support_ratio >= 0.75,
        "marked_slots": explicit_slots >= 18,
        "marked_measures": explicit_measures >= 3,
        "fully_marked_four_quarter_measures": fully_explicit >= 2,
        "unmarked_six_quarter_measures": fully_unmarked >= 8,
        "plain_four_quarter_coda": plain_coda >= 2,
    }
    failed = [name for name, passed in checks.items() if not passed]
    authorized = not failed
    return TripletGridEvidence(
        authorized=authorized,
        reason=(
            "independent candidate evidence confirms the continuous-triplet inference failure"
            if authorized
            else "evidence abstained: " + ", ".join(failed)
        ),
        inferred_meter=(2, 2) if authorized else None,
        parts_seen=len(parts),
        regular_measures=regular_measures,
        grid_measures=len(grids),
        grid_support_ratio=round(support_ratio, 6),
        explicit_triplet_slots=explicit_slots,
        explicit_triplet_measures=explicit_measures,
        fully_explicit_grid_measures=fully_explicit,
        fully_unmarked_grid_measures=fully_unmarked,
        plain_four_quarter_coda_measures=plain_coda,
        source_time_signatures=unique_times,
    )


def _choose_fallback(notes: list[TimedNote]) -> TimedNote:
    shortest_duration = min(item.duration for item in notes)
    shortest = [item for item in notes if item.duration == shortest_duration]
    rests = [item for item in shortest if item.node.find("rest") is not None and not item.chord]
    if rests:
        return rests[0]

    # If the shortest event is a sustained chord, the highest chord tone is the
    # most plausible continuation of an ascending accompaniment figure.  This is
    # still only used after whole-score grid evidence has authorized the measure.
    chord_members = [item for item in shortest if item.node.find("pitch") is not None]
    if any(item.chord for item in chord_members):
        return max(chord_members, key=lambda item: (_pitch_value(item.node), not item.chord))
    anchors = [item for item in shortest if not item.chord]
    return (anchors or shortest)[0]


def _texture_participants(grid: MeasureGrid) -> tuple[tuple[TimedNote, ...], ...]:
    by_onset: dict[Fraction, list[TimedNote]] = defaultdict(list)
    for note in grid.notes:
        if not note.grace and note.duration > 0:
            by_onset[note.onset].append(note)

    result: list[tuple[TimedNote, ...]] = []
    for onset in grid.slot_onsets:
        notes = by_onset[onset]
        explicit = tuple(item for item in notes if _is_triplet(item.node))
        if explicit:
            result.append(explicit)
            continue
        eighths = tuple(item for item in notes if _is_ordinary_eighth(item))
        if eighths:
            result.append(eighths)
            continue
        result.append((_choose_fallback(notes),))
    return tuple(result)


def _piecewise_map(value: Fraction, old: tuple[Fraction, ...]) -> Fraction:
    target = tuple(Fraction(index, 3) for index in range(len(old)))
    if value <= old[0]:
        return target[0]
    for index in range(1, len(old)):
        if value <= old[index]:
            left_old, right_old = old[index - 1], old[index]
            if right_old == left_old:
                return target[index]
            ratio = (value - left_old) / (right_old - left_old)
            return target[index - 1] + ratio * (target[index] - target[index - 1])

    final_interval = old[-1] - old[-2] if len(old) >= 2 else Fraction(1, 2)
    final_interval = max(Fraction(1, 12), final_interval)
    old_end = old[-1] + final_interval
    if value <= old_end:
        ratio = (value - old[-1]) / final_interval
        return target[-1] + ratio * (Fraction(4, 1) - target[-1])
    return Fraction(4, 1) + (value - old_end)


def _clone_as_triplet(source: etree._Element) -> etree._Element:
    clone = deepcopy(source)
    for tag in ("chord", "grace", "cue", "tie", "dot", "beam", "notations", "lyric", "play", "listen"):
        for child in list(clone.findall(tag)):
            clone.remove(child)
    clone.tail = None
    return clone


def _set_triplet_note(
    note: etree._Element,
    *,
    voice: str,
    beam_value: str,
    tuplet_value: str | None,
) -> None:
    _set_note_child_text(note, "voice", voice)
    _set_note_child_text(note, "type", "eighth")
    for dot in list(note.findall("dot")):
        note.remove(dot)

    modification = note.find("time-modification")
    if modification is None:
        modification = etree.Element("time-modification")
        _insert_note_child(note, modification)
    modification.clear()
    etree.SubElement(modification, "actual-notes").text = "3"
    etree.SubElement(modification, "normal-notes").text = "2"
    etree.SubElement(modification, "normal-type").text = "eighth"

    for beam in list(note.findall("beam")):
        note.remove(beam)
    if note.find("rest") is None:
        beam = etree.Element("beam", number="1")
        beam.text = beam_value
        _insert_note_child(note, beam)

    notations = note.find("notations")
    if notations is not None:
        for tuplet in list(notations.findall("tuplet")):
            notations.remove(tuplet)
    if tuplet_value:
        if notations is None:
            notations = etree.Element("notations")
            _insert_note_child(note, notations)
        etree.SubElement(
            notations,
            "tuplet",
            type=tuplet_value,
            bracket="no",
            number="1",
        )
    if notations is not None and len(notations) == 0 and not (notations.text or "").strip():
        note.remove(notations)


def _new_texture_voice(grid: MeasureGrid) -> str:
    numeric = [
        _integer(item.node.findtext("voice"), 0)
        for item in grid.notes
        if (item.node.findtext("voice") or "").strip().isdigit()
    ]
    return str(max(numeric, default=0) + 1)


def _lcm_denominators(values: list[Fraction], base: int) -> int:
    result = max(1, base)
    for value in values:
        result = math.lcm(result, value.denominator)
        if result > max(1536, base * 24):
            raise ValueError("triplet repair would require excessive MusicXML divisions")
    return result


def _movement(tag: str, ticks: int) -> etree._Element:
    node = etree.Element(tag)
    etree.SubElement(node, "duration").text = str(ticks)
    return node


def _set_measure_divisions(measure: etree._Element, divisions: int) -> None:
    attributes = measure.find("attributes")
    if attributes is None:
        attributes = etree.Element("attributes")
        measure.insert(0, attributes)
    divisions_node = attributes.find("divisions")
    if divisions_node is None:
        divisions_node = etree.Element("divisions")
        attributes.insert(0, divisions_node)
    divisions_node.text = str(max(1, divisions))


def _rewrite_timeline(
    grid: MeasureGrid,
    note_onsets: dict[etree._Element, Fraction],
    note_durations: dict[etree._Element, Fraction],
    direction_onsets: dict[etree._Element, Fraction],
) -> None:
    measure = grid.measure
    values = list(note_onsets.values()) + list(note_durations.values()) + list(direction_onsets.values())
    divisions = _lcm_denominators(values, grid.divisions)
    # Always make the repaired measure's timing unit explicit.  A preceding
    # repaired measure may need a larger LCM for an unusual local offset; letting
    # that value leak through MusicXML inheritance would reinterpret later ticks
    # (for example 4/36 instead of the intended 4/12 triplet duration).
    _set_measure_divisions(measure, divisions)

    children = [child for child in measure if child.tag not in {"backup", "forward"}]
    for child in list(measure):
        measure.remove(child)

    cursor = Fraction(0, 1)
    last_anchor = Fraction(0, 1)
    for child in children:
        if child.tag == "note":
            onset = note_onsets[child]
            duration = note_durations[child]
            chord = child.find("chord") is not None
            grace = child.find("grace") is not None
            if not chord:
                delta = onset - cursor
                if delta:
                    ticks = abs(delta * divisions)
                    if ticks.denominator != 1:
                        raise ValueError("note onset is not representable after triplet repair")
                    measure.append(_movement("forward" if delta > 0 else "backup", int(ticks)))
                last_anchor = onset
            elif onset != last_anchor:
                raise ValueError("triplet repair produced an orphan chord")
            duration_ticks = duration * divisions
            if duration_ticks.denominator != 1:
                raise ValueError("note duration is not representable after triplet repair")
            duration_node = child.find("duration")
            if duration_node is not None:
                duration_node.text = str(int(duration_ticks))
            measure.append(child)
            if not chord and not grace:
                cursor = onset + duration
            continue

        if child.tag == "direction":
            onset = direction_onsets.get(child, cursor)
            delta = onset - cursor
            if delta:
                ticks = abs(delta * divisions)
                if ticks.denominator != 1:
                    raise ValueError("direction onset is not representable after triplet repair")
                measure.append(_movement("forward" if delta > 0 else "backup", int(ticks)))
                cursor = onset
            offset = child.find("offset")
            if offset is not None:
                offset.text = "0"
        measure.append(child)


def _repair_measure(grid: MeasureGrid) -> tuple[int, int, int]:
    participants = _texture_participants(grid)
    texture_voice = _new_texture_voice(grid)
    note_onsets = {
        item.node: _piecewise_map(item.onset, grid.slot_onsets)
        for item in grid.notes
    }
    note_durations = {item.node: item.duration for item in grid.notes}
    direction_onsets = {
        item.node: _piecewise_map(item.onset, grid.slot_onsets)
        for item in grid.directions
    }

    # A common serialized tail is a sixteenth that follows a simultaneous dotted
    # eighth but was emitted only after the accompaniment stream finished.  Restore
    # that local duration relation instead of pushing it past the barline.
    if grid.trailing_onsets:
        trailing = grid.trailing_onsets[0]
        dotted = [
            item
            for item in grid.notes
            if item.onset < trailing
            and item.duration == Fraction(3, 4)
            and (item.node.findtext("type") or "").strip() == "eighth"
        ]
        trailing_notes = [item for item in grid.notes if item.onset == trailing]
        for tail in trailing_notes:
            same_staff = [
                item for item in dotted
                if (item.node.findtext("staff") or "1") == (tail.node.findtext("staff") or "1")
            ]
            if same_staff:
                source = max(same_staff, key=lambda item: item.onset)
                note_onsets[tail.node] = note_onsets[source.node] + source.duration

    converted = cloned = already = 0
    for slot_index, selected in enumerate(participants):
        target_onset = Fraction(slot_index, 3)
        beam_value = ("begin", "continue", "end")[slot_index % 3]
        tuplet_value = "start" if slot_index % 3 == 0 else "stop" if slot_index % 3 == 2 else None
        for item in selected:
            if _is_triplet(item.node):
                target = item.node
                already += 1
            elif _is_ordinary_eighth(item):
                target = item.node
                converted += 1
            else:
                target = _clone_as_triplet(item.node)
                item.node.addnext(target)
                cloned += 1
            _set_triplet_note(
                target,
                voice=texture_voice,
                beam_value=beam_value,
                tuplet_value=tuplet_value,
            )
            note_onsets[target] = target_onset
            note_durations[target] = Fraction(1, 3)

    _rewrite_timeline(grid, note_onsets, note_durations, direction_onsets)
    return converted, cloned, already


def _rewrite_time_signatures(root: etree._Element, meter: tuple[int, int]) -> int:
    count = 0
    for time_node in root.findall(".//attributes/time"):
        if time_node.find("senza-misura") is not None:
            continue
        beats = time_node.find("beats")
        beat_type = time_node.find("beat-type")
        if beats is None or beat_type is None:
            continue
        beats.text = str(meter[0])
        beat_type.text = str(meter[1])
        if meter == (2, 2):
            time_node.set("symbol", "cut")
        else:
            time_node.attrib.pop("symbol", None)
        count += 1
    return count


def apply_confirmed_continuous_triplet_grid(
    input_path: Path,
    output_path: Path,
    *,
    confirmed_meter: tuple[int, int],
    source_meter: tuple[int, int] = (3, 2),
    minimum_support_ratio: float = 0.75,
    minimum_supported_measures: int = 8,
) -> TripletTransactionReport:
    """Repair a whole-score triplet grid only after explicit meter confirmation.

    ``confirmed_meter`` is deliberately mandatory.  Passing a value is an API-level
    assertion by an upstream visual/metadata gate; this function never infers cut
    time from the candidate it is about to modify.
    """

    if confirmed_meter != (2, 2):
        raise ValueError("continuous triplet transaction currently supports confirmed cut time only")
    if source_meter != (3, 2):
        raise ValueError("continuous triplet transaction expects the known 3/2 inference failure")
    if not (0.75 <= minimum_support_ratio <= 1.0):
        raise ValueError("minimum_support_ratio must remain fail-closed at 0.75 or higher")
    if minimum_supported_measures < 8:
        raise ValueError("at least eight supported measures are required")

    parser = etree.XMLParser(remove_blank_text=True, resolve_entities=False, no_network=True)
    tree = etree.parse(str(input_path), parser)
    root = tree.getroot()
    parts = root.findall("part")
    expected_slots = confirmed_meter[0] * 3 * 4 // confirmed_meter[1]
    grids: list[MeasureGrid] = []
    original_measure_divisions: list[tuple[etree._Element, int]] = []
    regular_measures = 0
    inherited_divisions = 1
    inherited_meter: tuple[int, int] | None = None
    for part in parts:
        inherited_divisions = 1
        inherited_meter = None
        for measure in part.findall("measure"):
            for attributes in measure.findall("attributes"):
                inherited_divisions = max(
                    1,
                    _integer(attributes.findtext("divisions"), inherited_divisions),
                )
                time_node = attributes.find("time")
                if time_node is not None and time_node.find("senza-misura") is None:
                    inherited_meter = (
                        _integer(time_node.findtext("beats"), 4),
                        _integer(time_node.findtext("beat-type"), 4),
                    )
            original_measure_divisions.append((measure, inherited_divisions))
            if measure.findall("note"):
                regular_measures += 1
            if inherited_meter != source_meter:
                continue
            grid = _candidate_grid(
                measure,
                inherited_divisions,
                expected_slots=expected_slots,
            )
            if grid is not None:
                grids.append(grid)

    support_ratio = len(grids) / max(1, regular_measures)
    authorized = bool(
        len(parts) == 1
        and len(grids) >= minimum_supported_measures
        and support_ratio >= minimum_support_ratio
    )
    if not authorized:
        return TripletTransactionReport(
            applied=False,
            reason="whole-score continuous-grid evidence did not meet the fail-closed threshold",
            confirmed_meter=confirmed_meter,
            expected_slots=expected_slots,
            parts_seen=len(parts),
            regular_measures=regular_measures,
            supported_measures=len(grids),
            support_ratio=round(support_ratio, 6),
            measures_repaired=0,
            notes_converted=0,
            notes_cloned=0,
            notes_already_correct=0,
            time_signatures_rewritten=0,
            repaired_measure_numbers=(),
        )

    # Freeze the original inherited unit on every measure before any repaired
    # measure changes it.  This also protects unrepaired codas immediately after
    # a higher-resolution repaired measure.
    for measure, divisions in original_measure_divisions:
        _set_measure_divisions(measure, divisions)

    converted = cloned = already = 0
    for grid in grids:
        measure_converted, measure_cloned, measure_already = _repair_measure(grid)
        converted += measure_converted
        cloned += measure_cloned
        already += measure_already
    rewritten = _rewrite_time_signatures(root, confirmed_meter)
    payload = etree.tostring(
        tree,
        encoding="utf-8",
        xml_declaration=True,
        pretty_print=True,
        doctype=tree.docinfo.doctype or None,
    )
    atomic_write_bytes(output_path, payload)
    return TripletTransactionReport(
        applied=True,
        reason="confirmed cut-time continuous grid repaired",
        confirmed_meter=confirmed_meter,
        expected_slots=expected_slots,
        parts_seen=len(parts),
        regular_measures=regular_measures,
        supported_measures=len(grids),
        support_ratio=round(support_ratio, 6),
        measures_repaired=len(grids),
        notes_converted=converted,
        notes_cloned=cloned,
        notes_already_correct=already,
        time_signatures_rewritten=rewritten,
        repaired_measure_numbers=tuple(grid.number for grid in grids),
    )


def apply_evidence_confirmed_continuous_triplet_grid(
    input_path: Path,
    output_path: Path,
) -> tuple[TripletGridEvidence, TripletTransactionReport]:
    """Apply the transaction only when all independent candidate checks agree."""

    evidence = detect_continuous_triplet_grid_evidence(input_path)
    if not evidence.authorized or evidence.inferred_meter is None:
        report = TripletTransactionReport(
            applied=False,
            reason=evidence.reason,
            confirmed_meter=(2, 2),
            expected_slots=12,
            parts_seen=evidence.parts_seen,
            regular_measures=evidence.regular_measures,
            supported_measures=evidence.grid_measures,
            support_ratio=evidence.grid_support_ratio,
            measures_repaired=0,
            notes_converted=0,
            notes_cloned=0,
            notes_already_correct=0,
            time_signatures_rewritten=0,
            repaired_measure_numbers=(),
        )
        return evidence, report
    report = apply_confirmed_continuous_triplet_grid(
        input_path,
        output_path,
        confirmed_meter=evidence.inferred_meter,
    )
    return evidence, report
