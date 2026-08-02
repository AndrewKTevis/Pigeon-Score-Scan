from __future__ import annotations

"""Canonical MusicXML slur helpers.

Slurs are visual phrase annotations stored only below ``<notations>``.  Candidate
engines frequently disagree on numbering even when they describe the same arc, so the
consensus layer works with index-pair topology and rewrites deterministic numbers.
Only ``start``/``stop`` endpoints are supported by the automatic repair path; continue,
unknown values and unrelated notation always fail closed.
"""

import copy
from collections.abc import Iterable, Sequence

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


def _insert_ordered(note: etree._Element, child: etree._Element) -> None:
    rank = _NOTE_CHILD_ORDER.get(child.tag, 99)
    insertion = len(note)
    for index, existing in enumerate(note):
        if _NOTE_CHILD_ORDER.get(existing.tag, 99) > rank:
            insertion = index
            break
    note.insert(insertion, child)


def normalized_slur_topology(notes: Sequence[etree._Element]) -> tuple[tuple[int, int], ...] | None:
    """Return number-independent ``(start_index, stop_index)`` arcs.

    Each slur number must have exactly one start followed by exactly one stop.  The
    conservative repair path does not accept ``continue`` endpoints or duplicate
    endpoints because their intended topology is ambiguous.
    """

    endpoints: dict[str, dict[str, list[int]]] = {}
    for index, note in enumerate(notes):
        for item in note.findall("./notations/slur"):
            kind = str(item.get("type", "")).strip().casefold()
            number = str(item.get("number", "1")).strip() or "1"
            if kind not in {"start", "stop"}:
                return None
            bucket = endpoints.setdefault(number, {"start": [], "stop": []})
            bucket[kind].append(index)
    arcs: list[tuple[int, int]] = []
    for bucket in endpoints.values():
        starts = bucket["start"]
        stops = bucket["stop"]
        if len(starts) != 1 or len(stops) != 1 or starts[0] >= stops[0]:
            return None
        arcs.append((starts[0], stops[0]))
    arcs.sort()
    if len(set(arcs)) != len(arcs):
        return None
    return tuple(arcs)


def set_slur_topology(notes: Sequence[etree._Element], arcs: Iterable[tuple[int, int]]) -> None:
    """Replace all slur endpoints while preserving every unrelated notation child."""

    note_list = list(notes)
    normalized = tuple(sorted((int(start), int(stop)) for start, stop in arcs))
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate slur arc")
    for start, stop in normalized:
        if start < 0 or stop >= len(note_list) or start >= stop:
            raise ValueError("invalid slur arc")

    for note in note_list:
        notations = note.find("notations")
        if notations is None:
            continue
        for child in list(notations.findall("slur")):
            notations.remove(child)
        if len(notations) == 0 and not notations.attrib and not (notations.text or "").strip():
            note.remove(notations)

    for number, (start, stop) in enumerate(normalized, start=1):
        for index, kind in ((start, "start"), (stop, "stop")):
            note = note_list[index]
            notations = note.find("notations")
            if notations is None:
                notations = etree.Element("notations")
                _insert_ordered(note, notations)
            notations.append(etree.Element("slur", type=kind, number=str(number)))


def without_slurs(note: etree._Element) -> bytes:
    """Return a snapshot proving unrelated note XML is unchanged."""

    clone = copy.deepcopy(note)
    set_slur_topology([clone], ())
    return etree.tostring(clone)
