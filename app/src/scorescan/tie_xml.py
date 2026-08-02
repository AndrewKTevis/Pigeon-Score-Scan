from __future__ import annotations

"""Canonical MusicXML tie endpoint helpers shared by local and boundary repair.

MusicXML represents playback ties twice: direct ``<tie>`` children and visual
``<notations><tied>`` children.  The recognition pipeline treats their union as one
semantic state and writes both forms together so downstream renderers and players do
not disagree.  Helpers in this module deliberately preserve every unrelated note
child and every endpoint type not selected for mutation.
"""

import copy
from collections.abc import Iterable

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
_VALID_TYPES = ("stop", "start")


def normalized_tie_state(note: etree._Element) -> tuple[str, ...] | None:
    """Return the canonical endpoint union, or ``None`` for unsupported values."""

    values = {
        str(item.get("type", "")).strip().casefold()
        for item in (*note.findall("tie"), *note.findall("./notations/tied"))
        if str(item.get("type", "")).strip()
    }
    if any(value not in _VALID_TYPES for value in values):
        return None
    return tuple(value for value in _VALID_TYPES if value in values)


def _insert_ordered(note: etree._Element, child: etree._Element) -> None:
    rank = _NOTE_CHILD_ORDER.get(child.tag, 99)
    insertion = len(note)
    for index, existing in enumerate(note):
        if _NOTE_CHILD_ORDER.get(existing.tag, 99) > rank:
            insertion = index
            break
    note.insert(insertion, child)


def set_tie_state(note: etree._Element, state: Iterable[str]) -> None:
    """Replace all tie endpoints while preserving unrelated notation."""

    wanted = {str(value).strip().casefold() for value in state if str(value).strip()}
    if any(value not in _VALID_TYPES for value in wanted):
        raise ValueError("unsupported tie endpoint")
    for child in list(note.findall("tie")):
        note.remove(child)
    notations = note.find("notations")
    if notations is not None:
        for child in list(notations.findall("tied")):
            notations.remove(child)
        if len(notations) == 0 and not notations.attrib and not (notations.text or "").strip():
            note.remove(notations)
            notations = None
    ordered = tuple(value for value in _VALID_TYPES if value in wanted)
    for value in ordered:
        _insert_ordered(note, etree.Element("tie", type=value))
    if ordered:
        if notations is None:
            notations = etree.Element("notations")
            _insert_ordered(note, notations)
        for index, value in enumerate(ordered):
            notations.insert(index, etree.Element("tied", type=value))


def set_endpoint(note: etree._Element, endpoint: str, present: bool) -> bool:
    """Toggle one endpoint and preserve the other endpoint, returning success."""

    endpoint = endpoint.strip().casefold()
    if endpoint not in _VALID_TYPES:
        return False
    state = normalized_tie_state(note)
    if state is None:
        return False
    values = set(state)
    if present:
        values.add(endpoint)
    else:
        values.discard(endpoint)
    set_tie_state(note, values)
    return True


def without_ties(note: etree._Element) -> bytes:
    """Canonical byte snapshot used to prove unrelated XML was unchanged."""

    clone = copy.deepcopy(note)
    set_tie_state(clone, ())
    return etree.tostring(clone)
