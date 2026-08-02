from __future__ import annotations

"""Canonical helpers for the narrow simple-triplet profile.

ScoreScan never synthesizes arbitrary tuplets.  The automatic repair profile is
limited to contiguous three-note 3:2 groups in one monophonic measure.  MusicXML
stores the sounding ratio in ``<time-modification>`` and the visible bracket/number
in ``<notations><tuplet>``.  These helpers read and write both forms together while
preserving unrelated notation.
"""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from lxml import etree

from .musicxml import MUSICXML_DOCTYPE
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
class SimpleTupletNoteState:
    ratio: tuple[int, int] | None
    start: bool
    stop: bool


def _integer(value: str | None) -> int:
    try:
        return int((value or "").strip())
    except (TypeError, ValueError):
        return 0


def _insert_ordered(note: etree._Element, child: etree._Element) -> None:
    rank = _NOTE_CHILD_ORDER.get(child.tag, 99)
    insertion = len(note)
    for index, existing in enumerate(note):
        if _NOTE_CHILD_ORDER.get(existing.tag, 99) > rank:
            insertion = index
            break
    note.insert(insertion, child)


def read_simple_tuplet_state(note: etree._Element) -> SimpleTupletNoteState | None:
    """Return canonical state, or ``None`` when unsupported/malformed."""

    time_modifications = note.findall("time-modification")
    if len(time_modifications) > 1:
        return None
    ratio: tuple[int, int] | None = None
    if time_modifications:
        node = time_modifications[0]
        actual = _integer(node.findtext("actual-notes"))
        normal = _integer(node.findtext("normal-notes"))
        normal_type = (node.findtext("normal-type") or "").strip().casefold()
        note_type = (note.findtext("type") or "").strip().casefold()
        if (actual, normal) != (3, 2):
            return None
        if normal_type and note_type and normal_type != note_type:
            return None
        ratio = (3, 2)

    tuplets = note.findall("./notations/tuplet")
    start = False
    stop = False
    for marker in tuplets:
        marker_type = (marker.get("type") or "").strip().casefold()
        number = (marker.get("number") or "1").strip()
        if marker_type not in {"start", "stop"} or number not in {"", "1"}:
            return None
        if marker_type == "start":
            if start:
                return None
            start = True
        else:
            if stop:
                return None
            stop = True
    if (start or stop) and ratio is None:
        return None
    return SimpleTupletNoteState(ratio, start, stop)


def set_simple_tuplet_state(
    note: etree._Element,
    *,
    ratio: tuple[int, int] | None,
    start: bool = False,
    stop: bool = False,
) -> None:
    """Write one canonical simple-triplet note state."""

    if ratio not in {None, (3, 2)}:
        raise ValueError("only 3:2 simple tuplets are supported")
    if (start or stop) and ratio is None:
        raise ValueError("tuplet endpoints require a time modification")

    for child in list(note.findall("time-modification")):
        note.remove(child)
    notations = note.find("notations")
    if notations is not None:
        for child in list(notations.findall("tuplet")):
            notations.remove(child)
        if len(notations) == 0 and not notations.attrib and not (notations.text or "").strip():
            note.remove(notations)
            notations = None

    if ratio is not None:
        modification = etree.Element("time-modification")
        etree.SubElement(modification, "actual-notes").text = "3"
        etree.SubElement(modification, "normal-notes").text = "2"
        _insert_ordered(note, modification)
    if start or stop:
        if notations is None:
            notations = etree.Element("notations")
            _insert_ordered(note, notations)
        insertion = 0
        if start:
            notations.insert(insertion, etree.Element("tuplet", type="start", number="1"))
            insertion += 1
        if stop:
            notations.insert(insertion, etree.Element("tuplet", type="stop", number="1"))


def sanitize_incomplete_implicit_triplets(
    source_path: Path,
    destination_path: Path,
) -> dict[str, object]:
    """Remove only incomplete, unmarked 3:2 runs from a raw homr candidate.

    homr's page-level median cleanup can erase sustained triplet texture. Disabling
    it preserves that texture, but can also leave short implicit fragments that do
    not contain a whole group. Such fragments are not defensible tuplet evidence
    and have caused importer failures. Complete implicit groups and every explicitly
    marked group are preserved; only unmarked incomplete runs are restored to their
    normal duration.
    """

    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    tree = etree.parse(str(source_path), parser)
    repaired: list[dict[str, object]] = []
    abstained: list[dict[str, object]] = []
    topology_errors: list[dict[str, object]] = []

    def finish_run(
        measure: etree._Element,
        voice: str,
        staff: str,
        notes: list[etree._Element],
        *,
        explicit: bool,
    ) -> None:
        if not notes:
            return
        if explicit:
            states = [read_simple_tuplet_state(note) for note in notes]
            valid = bool(
                len(states) == 3
                and all(state is not None for state in states)
                and states[0] is not None
                and states[0].start
                and not states[0].stop
                and states[1] is not None
                and not states[1].start
                and not states[1].stop
                and states[2] is not None
                and not states[2].start
                and states[2].stop
            )
            if not valid:
                topology_errors.append(
                    {
                        "measure": measure.get("number"),
                        "voice": voice,
                        "staff": staff,
                        "event_count": len(notes),
                        "reason": "malformed_explicit_simple_triplet",
                    }
                )
            return
        if len(notes) % 3 == 0:
            return
        scaled: list[tuple[etree._Element, str]] = []
        for note in notes:
            duration = note.find("duration")
            try:
                value = int((duration.text if duration is not None else "") or "")
            except (TypeError, ValueError):
                value = 0
            normal_duration = Fraction(value * 3, 2)
            if (
                duration is None
                or value <= 0
                or normal_duration.denominator != 1
            ):
                abstained.append(
                    {
                        "measure": measure.get("number"),
                        "voice": voice,
                        "staff": staff,
                        "event_count": len(notes),
                        "reason": "non_integral_normal_duration",
                    }
                )
                return
            scaled.append((note, str(normal_duration.numerator)))
        for note, normal_duration in scaled:
            duration = note.find("duration")
            assert duration is not None
            duration.text = normal_duration
            set_simple_tuplet_state(note, ratio=None)
        repaired.append(
            {
                "measure": measure.get("number"),
                "voice": voice,
                "staff": staff,
                "event_count": len(notes),
                "ratio": [3, 2],
                "reason": "incomplete_implicit_group",
            }
        )

    for measure in tree.findall(".//measure"):
        streams: dict[tuple[str, str], list[etree._Element]] = {}
        for note in measure.findall("note"):
            if note.find("chord") is not None:
                continue
            streams.setdefault(
                (
                    note.findtext("voice") or "1",
                    note.findtext("staff") or "1",
                ),
                [],
            ).append(note)
        for (voice, staff), notes in streams.items():
            run: list[etree._Element] = []
            explicit = False
            for note in [*notes, None]:
                state = (
                    read_simple_tuplet_state(note)
                    if note is not None
                    else None
                )
                if state is not None and state.ratio == (3, 2):
                    if state.start and run:
                        finish_run(
                            measure,
                            voice,
                            staff,
                            run,
                            explicit=explicit,
                        )
                        run = []
                        explicit = False
                    run.append(note)
                    explicit = explicit or state.start or state.stop
                    if state.stop:
                        finish_run(
                            measure,
                            voice,
                            staff,
                            run,
                            explicit=explicit,
                        )
                        run = []
                        explicit = False
                    continue
                finish_run(
                    measure,
                    voice,
                    staff,
                    run,
                    explicit=explicit,
                )
                run = []
                explicit = False
                if note is not None:
                    modification = note.find("time-modification")
                    try:
                        ratio = (
                            int(modification.findtext("actual-notes") or "0"),
                            int(modification.findtext("normal-notes") or "0"),
                        ) if modification is not None else None
                    except (TypeError, ValueError):
                        ratio = None
                    if ratio == (3, 2) or note.find("./notations/tuplet") is not None:
                        topology_errors.append(
                            {
                                "measure": measure.get("number"),
                                "voice": voice,
                                "staff": staff,
                                "event_count": 1,
                                "reason": "malformed_simple_triplet_state",
                            }
                        )

    atomic_write_bytes(
        destination_path,
        etree.tostring(
            tree,
            encoding="UTF-8",
            xml_declaration=True,
            pretty_print=True,
            doctype=MUSICXML_DOCTYPE,
        ),
    )
    return {
        "format": 1,
        "changed_group_count": len(repaired),
        "changed_event_count": sum(
            int(item["event_count"])
            for item in repaired
        ),
        "repaired": repaired,
        "abstained_group_count": len(abstained),
        "abstained": abstained,
        "topology_valid": not abstained and not topology_errors,
        "topology_error_count": len(topology_errors),
        "topology_errors": topology_errors,
    }
