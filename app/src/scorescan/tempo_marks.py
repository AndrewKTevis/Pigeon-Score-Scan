from __future__ import annotations

"""Deterministic parsing for printed metronome marks.

OCR engines often return beat units as Unicode music glyphs, ASCII approximations, or
words.  This parser keeps the original text but extracts a conservative MusicXML beat
unit and numeric range when the notation is unambiguous.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MetronomeMark:
    beat_unit: str
    dotted: bool
    per_minute_low: int
    per_minute_high: int | None = None
    approximate: bool = False

    @property
    def per_minute_text(self) -> str:
        if self.per_minute_high is None:
            return str(self.per_minute_low)
        return f"{self.per_minute_low}-{self.per_minute_high}"


_RANGE_RE = re.compile(r"(?<!\d)(\d{2,3})(?:\s*[-–—]\s*(\d{2,3}))?(?!\d)")
_APPROX_RE = re.compile(r"(?:\bca\.?\b|\bcirca\b|≈|~|c\.)", re.IGNORECASE)
_EQUALS_RE = re.compile(r"(?:=|≈|~|\b(?:bpm|m\.?m\.?)\b)", re.IGNORECASE)

_GLYPH_UNITS = {
    "𝅝": ("whole", False),
    "𝅗𝅥": ("half", False),
    "𝅘𝅥": ("quarter", False),
    "♩": ("quarter", False),
    "𝅘𝅥𝅮": ("eighth", False),
    "♪": ("eighth", False),
    "𝅘𝅥𝅯": ("16th", False),
    "𝅘𝅥𝅰": ("32nd", False),
}

_WORD_UNITS = (
    (re.compile(r"\b(?:dotted\s+quarter|quarter\s*(?:note)?\s*[.]|crotchet\s*[.])\b", re.IGNORECASE), ("quarter", True)),
    (re.compile(r"\b(?:dotted\s+eighth|eighth\s*(?:note)?\s*[.]|quaver\s*[.])\b", re.IGNORECASE), ("eighth", True)),
    (re.compile(r"\b(?:dotted\s+half|half\s*(?:note)?\s*[.]|minim\s*[.])\b", re.IGNORECASE), ("half", True)),
    (re.compile(r"\b(?:whole\s*(?:note)?|semibreve)\b", re.IGNORECASE), ("whole", False)),
    (re.compile(r"\b(?:half\s*(?:note)?|minim)\b", re.IGNORECASE), ("half", False)),
    (re.compile(r"\b(?:quarter\s*(?:note)?|crotchet)\b", re.IGNORECASE), ("quarter", False)),
    (re.compile(r"\b(?:eighth\s*(?:note)?|quaver)\b", re.IGNORECASE), ("eighth", False)),
    (re.compile(r"\b(?:sixteenth\s*(?:note)?|semiquaver|16th)\b", re.IGNORECASE), ("16th", False)),
    (re.compile(r"\b(?:thirty[- ]?second\s*(?:note)?|demisemiquaver|32nd)\b", re.IGNORECASE), ("32nd", False)),
)


def _glyph_unit(text: str) -> tuple[str, bool] | None:
    for index, char in enumerate(text):
        if char not in _GLYPH_UNITS:
            continue
        unit, dotted = _GLYPH_UNITS[char]
        tail = text[index + 1:index + 4]
        if any(mark in tail for mark in (".", "·", "•", "‧")):
            dotted = True
        return unit, dotted
    return None


def _ascii_unit(text: str) -> tuple[str, bool] | None:
    # OCR commonly substitutes a notehead/stem with q, J, or an isolated lowercase d.
    # Accept these only when immediately followed by a metronome separator, preventing
    # ordinary prose from being misclassified.
    match = re.search(r"(?:^|\s)([qQjJd])([.]?)\s*(?==|≈|~)", text)
    if not match:
        return None
    return "quarter", bool(match.group(2))


def parse_metronome_mark(text: str) -> MetronomeMark | None:
    value = text.strip().replace("−", "-")
    if not value or not _EQUALS_RE.search(value):
        return None
    number_match = _RANGE_RE.search(value)
    if not number_match:
        return None
    low = int(number_match.group(1))
    high = int(number_match.group(2)) if number_match.group(2) else None
    if not 20 <= low <= 400 or (high is not None and not low <= high <= 400):
        return None

    unit = _glyph_unit(value)
    if unit is None:
        for pattern, candidate in _WORD_UNITS:
            if pattern.search(value):
                unit = candidate
                break
    if unit is None:
        unit = _ascii_unit(value)
    if unit is None:
        # "M.M. = 80" conventionally implies a quarter note but only when M.M. is
        # explicitly present; a bare "= 80" is too ambiguous.
        if re.search(r"\bm\.?m\.?\b", value, re.IGNORECASE):
            unit = ("quarter", False)
        else:
            return None
    return MetronomeMark(
        beat_unit=unit[0],
        dotted=unit[1],
        per_minute_low=low,
        per_minute_high=high,
        approximate=bool(_APPROX_RE.search(value)),
    )
