from __future__ import annotations

"""Canonical MusicXML helpers for simple note ornaments.

The automatic repair path deliberately supports only empty, attribute-free ornament
markers whose semantics are represented completely by the element name.  More complex
MusicXML ornament content remains untouched and makes the local repair fail closed.
"""

import copy
from collections.abc import Sequence

from lxml import etree

ALLOWED_ORNAMENTS = frozenset({
    "trill-mark",
    "turn",
    "inverted-turn",
    "mordent",
    "inverted-mordent",
    "shake",
    "schleifer",
})

_ORNAMENT_ORDER = {
    "trill-mark": 0,
    "turn": 1,
    "inverted-turn": 2,
    "mordent": 3,
    "inverted-mordent": 4,
    "shake": 5,
    "schleifer": 6,
}

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

_NOTATIONS_CHILD_ORDER = {
    "footnote": 0,
    "level": 1,
    "tied": 2,
    "slur": 3,
    "tuplet": 4,
    "glissando": 5,
    "slide": 6,
    "ornaments": 7,
    "technical": 8,
    "articulations": 9,
    "dynamics": 10,
    "fermata": 11,
    "arpeggiate": 12,
    "non-arpeggiate": 13,
    "accidental-mark": 14,
    "other-notation": 15,
}

OrnamentTopology = tuple[tuple[str, ...], ...]


def _insert_note_child(note: etree._Element, child: etree._Element) -> None:
    rank = _NOTE_CHILD_ORDER.get(child.tag, 99)
    insertion = len(note)
    for index, existing in enumerate(note):
        if _NOTE_CHILD_ORDER.get(existing.tag, 99) > rank:
            insertion = index
            break
    note.insert(insertion, child)


def _insert_notations_child(notations: etree._Element, child: etree._Element) -> None:
    rank = _NOTATIONS_CHILD_ORDER.get(child.tag, 99)
    insertion = len(notations)
    for index, existing in enumerate(notations):
        if _NOTATIONS_CHILD_ORDER.get(existing.tag, 99) > rank:
            insertion = index
            break
    notations.insert(insertion, child)


def _simple_container(note: etree._Element) -> tuple[str, ...] | None:
    notations_nodes = note.findall("notations")
    if len(notations_nodes) > 1:
        return None
    if not notations_nodes:
        return ()
    containers = notations_nodes[0].findall("ornaments")
    if len(containers) > 1:
        return None
    if not containers:
        return ()
    result: list[str] = []
    for child in containers[0]:
        tag = str(child.tag)
        if (
            tag not in ALLOWED_ORNAMENTS
            or child.attrib
            or len(child)
            or (child.text or "").strip()
            or (child.tail or "").strip()
            or tag in result
        ):
            return None
        result.append(tag)
    return tuple(sorted(result, key=lambda item: (_ORNAMENT_ORDER.get(item, 99), item)))


def normalized_ornament_topology(notes: Sequence[etree._Element]) -> OrnamentTopology | None:
    rows: list[tuple[str, ...]] = []
    for note in notes:
        row = _simple_container(note)
        if row is None:
            return None
        rows.append(row)
    return tuple(rows)


def set_ornament_topology(notes: Sequence[etree._Element], topology: Sequence[Sequence[str]]) -> None:
    if len(notes) != len(topology):
        raise ValueError("ornament topology length does not match note sequence")
    for note, requested in zip(notes, topology, strict=True):
        desired = tuple(sorted(set(requested), key=lambda item: (_ORNAMENT_ORDER.get(item, 99), item)))
        if any(item not in ALLOWED_ORNAMENTS for item in desired):
            raise ValueError("unsupported ornament topology")
        notations_nodes = note.findall("notations")
        if len(notations_nodes) > 1:
            raise ValueError("multiple notations containers")
        notations = notations_nodes[0] if notations_nodes else None
        if notations is not None:
            for container in list(notations.findall("ornaments")):
                notations.remove(container)
        if desired:
            if notations is None:
                notations = etree.Element("notations")
                _insert_note_child(note, notations)
            container = etree.Element("ornaments")
            for tag in desired:
                container.append(etree.Element(tag))
            _insert_notations_child(notations, container)
        if notations is not None and len(notations) == 0 and not notations.attrib and not (notations.text or "").strip():
            note.remove(notations)


def without_ornaments(note: etree._Element) -> bytes:
    clone = copy.deepcopy(note)
    topology = normalized_ornament_topology([clone])
    if topology is None:
        raise ValueError("unsupported ornament XML")
    set_ornament_topology([clone], [()])
    return etree.tostring(clone)
