from __future__ import annotations

"""Canonical helpers for a deliberately small MusicXML lyric subset.

The automatic path supports at most one verse per note, represented by an
attribute-free ``<lyric>`` containing optional ``<syllabic>``, one ``<text>``, and
optional empty ``<extend>``.  Elisions, multiple verses, pronunciation, humming,
end-line/end-paragraph markers, formatting attributes and mixed content remain outside
this repair path and make it fail closed.
"""

import copy
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

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

_ALLOWED_SYLLABIC = frozenset({"", "single", "begin", "middle", "end"})
_ALLOWED_EXTEND = frozenset({"", "start", "continue", "stop"})
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True, order=True)
class LyricState:
    text: str
    syllabic: str = ""
    extend: str = ""

    def stable_tuple(self) -> tuple[str, str, str]:
        return self.text, self.syllabic, self.extend


LyricTopology = tuple[LyricState | None, ...]


def normalize_lyric_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value).replace("\u00a0", " ")
    return _SPACE.sub(" ", value).strip()


def _simple_text_node(node: etree._Element, *, allow_empty: bool = False) -> str | None:
    if node.attrib or len(node) or (node.tail or "").strip():
        return None
    value = normalize_lyric_text(node.text or "")
    if not value and not allow_empty:
        return None
    if len(value) > 64 or any(ord(char) < 32 and char not in "\t\n\r" for char in value):
        return None
    return value


def _parse_lyric(note: etree._Element) -> LyricState | None | object:
    lyrics = note.findall("lyric")
    if not lyrics:
        return None
    if len(lyrics) != 1:
        return _INVALID
    lyric = lyrics[0]
    if lyric.attrib or (lyric.text or "").strip() or (lyric.tail or "").strip():
        return _INVALID
    allowed = {"syllabic", "text", "extend"}
    if any(child.tag not in allowed for child in lyric):
        return _INVALID
    syllabic_nodes = lyric.findall("syllabic")
    text_nodes = lyric.findall("text")
    extend_nodes = lyric.findall("extend")
    if len(syllabic_nodes) > 1 or len(text_nodes) != 1 or len(extend_nodes) > 1:
        return _INVALID
    syllabic = ""
    if syllabic_nodes:
        parsed = _simple_text_node(syllabic_nodes[0])
        if parsed is None:
            return _INVALID
        syllabic = parsed.casefold()
        if syllabic not in _ALLOWED_SYLLABIC:
            return _INVALID
    text = _simple_text_node(text_nodes[0])
    if text is None:
        return _INVALID
    extend = ""
    if extend_nodes:
        extend_node = extend_nodes[0]
        if len(extend_node) or (extend_node.text or "").strip() or (extend_node.tail or "").strip():
            return _INVALID
        if set(extend_node.attrib) - {"type"}:
            return _INVALID
        extend = (extend_node.get("type") or "start").strip().casefold()
        if extend not in _ALLOWED_EXTEND or not extend:
            return _INVALID
    return LyricState(text=text, syllabic=syllabic, extend=extend)


_INVALID = object()


def normalized_lyric_topology(notes: Sequence[etree._Element]) -> LyricTopology | None:
    rows: list[LyricState | None] = []
    for note in notes:
        row = _parse_lyric(note)
        if row is _INVALID:
            return None
        rows.append(row if isinstance(row, LyricState) else None)
    return tuple(rows)


def _insert_note_child(note: etree._Element, child: etree._Element) -> None:
    rank = _NOTE_CHILD_ORDER.get(child.tag, 99)
    insertion = len(note)
    for index, existing in enumerate(note):
        if _NOTE_CHILD_ORDER.get(existing.tag, 99) > rank:
            insertion = index
            break
    note.insert(insertion, child)


def set_lyric_topology(notes: Sequence[etree._Element], topology: Sequence[LyricState | None]) -> None:
    if len(notes) != len(topology):
        raise ValueError("lyric topology length does not match note sequence")
    for note, requested in zip(notes, topology, strict=True):
        for lyric in list(note.findall("lyric")):
            note.remove(lyric)
        if requested is None:
            continue
        text = normalize_lyric_text(requested.text)
        syllabic = requested.syllabic.strip().casefold()
        extend = requested.extend.strip().casefold()
        if not text or len(text) > 64 or syllabic not in _ALLOWED_SYLLABIC or extend not in _ALLOWED_EXTEND:
            raise ValueError("unsupported lyric topology")
        lyric = etree.Element("lyric")
        if syllabic:
            node = etree.SubElement(lyric, "syllabic")
            node.text = syllabic
        node = etree.SubElement(lyric, "text")
        node.text = text
        if extend:
            etree.SubElement(lyric, "extend", type=extend)
        _insert_note_child(note, lyric)


def without_lyrics(note: etree._Element) -> bytes:
    clone = copy.deepcopy(note)
    for lyric in list(clone.findall("lyric")):
        clone.remove(lyric)
    return etree.tostring(clone)
