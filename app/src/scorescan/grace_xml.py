from __future__ import annotations

"""Canonical MusicXML helpers for simple grace-note topology.

The automatic path supports only empty, attribute-free ``<grace/>`` elements on
pitched, non-chord notes.  Grace conversion owns the note's duration/type/dot children
because those fields determine whether the event advances the musical cursor.  All
other note XML is preserved byte-for-byte by the caller's transaction checks.
"""

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from lxml import etree

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


@dataclass(frozen=True, order=True)
class GraceState:
    grace: bool
    duration: Fraction
    note_type: str
    dots: int

    def stable_tuple(self) -> tuple[object, ...]:
        return (
            self.grace,
            f"{self.duration.numerator}/{self.duration.denominator}",
            self.note_type,
            self.dots,
        )


GraceTopology = tuple[GraceState, ...]


def _simple_empty(element: etree._Element) -> bool:
    return (
        not element.attrib
        and len(element) == 0
        and not (element.text or "").strip()
        and not (element.tail or "").strip()
    )


def _insert_note_child(note: etree._Element, child: etree._Element) -> None:
    rank = _NOTE_CHILD_ORDER.get(child.tag, 99)
    insertion = len(note)
    for index, existing in enumerate(note):
        if _NOTE_CHILD_ORDER.get(existing.tag, 99) > rank:
            insertion = index
            break
    note.insert(insertion, child)


def _parse_state(note: etree._Element, divisions: int) -> GraceState | None:
    grace_nodes = note.findall("grace")
    duration_nodes = note.findall("duration")
    type_nodes = note.findall("type")
    dot_nodes = note.findall("dot")
    if len(grace_nodes) > 1 or len(duration_nodes) > 1 or len(type_nodes) != 1:
        return None
    if any(not _simple_empty(node) for node in grace_nodes + dot_nodes):
        return None
    note_type = (type_nodes[0].text or "").strip().casefold()
    if (
        not note_type
        or type_nodes[0].attrib
        or len(type_nodes[0])
        or (type_nodes[0].tail or "").strip()
    ):
        return None
    is_grace = bool(grace_nodes)
    if is_grace:
        if duration_nodes:
            return None
        duration = Fraction(0, 1)
    else:
        if len(duration_nodes) != 1:
            return None
        duration_node = duration_nodes[0]
        if duration_node.attrib or len(duration_node) or (duration_node.tail or "").strip():
            return None
        try:
            raw = int((duration_node.text or "").strip())
        except (TypeError, ValueError, OverflowError):
            return None
        if raw <= 0:
            return None
        duration = Fraction(raw, max(1, int(divisions)))
    return GraceState(is_grace, duration, note_type, len(dot_nodes))


def normalized_grace_topology(
    notes: Sequence[etree._Element],
    divisions: int,
) -> GraceTopology | None:
    rows: list[GraceState] = []
    for note in notes:
        row = _parse_state(note, divisions)
        if row is None:
            return None
        rows.append(row)
    return tuple(rows)


def set_grace_topology(
    notes: Sequence[etree._Element],
    topology: Sequence[GraceState],
    divisions: int,
) -> None:
    if len(notes) != len(topology):
        raise ValueError("grace topology length does not match note sequence")
    divisions = max(1, int(divisions))
    for note, desired in zip(notes, topology, strict=True):
        if desired.dots < 0 or not desired.note_type.strip():
            raise ValueError("invalid grace topology")
        if desired.grace:
            if desired.duration != 0:
                raise ValueError("grace event must have zero duration")
        else:
            raw_duration = desired.duration * divisions
            if desired.duration <= 0 or raw_duration.denominator != 1:
                raise ValueError("regular event duration is not representable in template divisions")

        for tag in ("grace", "duration", "type", "dot"):
            for child in list(note.findall(tag)):
                note.remove(child)
        if desired.grace:
            _insert_note_child(note, etree.Element("grace"))
        else:
            duration = etree.Element("duration")
            duration.text = str(int(desired.duration * divisions))
            _insert_note_child(note, duration)
        note_type = etree.Element("type")
        note_type.text = desired.note_type
        _insert_note_child(note, note_type)
        for _ in range(desired.dots):
            _insert_note_child(note, etree.Element("dot"))


def without_grace_rhythm(note: etree._Element) -> bytes:
    clone = copy.deepcopy(note)
    for tag in ("grace", "duration", "type", "dot"):
        for child in list(clone.findall(tag)):
            clone.remove(child)
    return etree.tostring(clone)
