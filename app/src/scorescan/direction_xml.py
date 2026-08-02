from __future__ import annotations

"""Canonical helpers for simple MusicXML dynamics and metronome directions.

The automatic repair path deliberately supports only compact direction records whose
semantics are completely represented by placement, onset and one value.  Words,
wedges, pedal marks, rehearsal marks and formatted/attributed dynamics remain untouched
and make a local dynamic/metronome direction fail closed when they share the same
``direction`` element.
"""

import copy
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

from lxml import etree

DYNAMIC_TAGS = frozenset(
    {
        "p", "pp", "ppp", "pppp", "ppppp", "pppppp",
        "mp", "mf",
        "f", "ff", "fff", "ffff", "fffff", "ffffff",
        "fp", "pf", "sf", "sfp", "sfpp", "sfz", "sffz",
        "rf", "rfz", "fz",
    }
)
METRONOME_UNITS = frozenset(
    {"maxima", "long", "breve", "whole", "half", "quarter", "eighth", "16th", "32nd", "64th"}
)
_PER_MINUTE_RE = re.compile(r"^[1-9][0-9]{0,2}$")


@dataclass(frozen=True, order=True)
class SimpleDirection:
    onset: Fraction
    kind: str
    placement: str
    value: str
    sound_tempo: bool = False


def _text(value: str | None) -> str:
    return (value or "").strip()


def _simple_direction(
    direction: etree._Element,
    cursor: Fraction,
    divisions: int,
) -> tuple[bool, SimpleDirection | None]:
    """Return ``(mentions_supported_kind, parsed_value)`` for one direction node."""

    dynamics_nodes = direction.findall("./direction-type/dynamics")
    metronome_nodes = direction.findall("./direction-type/metronome")
    mentions = bool(dynamics_nodes or metronome_nodes)
    if not mentions:
        return False, None

    direction_types = direction.findall("direction-type")
    if len(direction_types) != 1:
        return True, None
    direction_type = direction_types[0]
    if len(direction_type) != 1:
        return True, None
    if direction.find("voice") is not None or direction.find("staff") is not None:
        return True, None

    placement = _text(direction.get("placement")).casefold()
    if placement not in {"above", "below"}:
        return True, None
    offset_text = _text(direction.findtext("offset")) or "0"
    try:
        offset_ticks = int(offset_text)
    except (TypeError, ValueError, OverflowError):
        return True, None
    if offset_ticks < 0:
        return True, None
    onset = cursor + Fraction(offset_ticks, max(1, divisions))

    child = direction_type[0]
    if child.tag == "dynamics":
        if len(child) != 1 or child.attrib or _text(child.text):
            return True, None
        marker = child[0]
        if marker.tag not in DYNAMIC_TAGS or marker.attrib or len(marker) or _text(marker.text):
            return True, None
        if direction.find("sound") is not None:
            return True, None
        return True, SimpleDirection(onset, "dynamic", placement, marker.tag, False)

    if child.tag != "metronome":
        return True, None
    if child.attrib and set(child.attrib) != {"parentheses"}:
        return True, None
    if _text(child.get("parentheses")).casefold() not in {"", "no"}:
        return True, None
    allowed_children = {"beat-unit", "beat-unit-dot", "per-minute"}
    if any(grandchild.tag not in allowed_children for grandchild in child):
        return True, None
    beat_nodes = child.findall("beat-unit")
    per_minute_nodes = child.findall("per-minute")
    dot_nodes = child.findall("beat-unit-dot")
    if len(beat_nodes) != 1 or len(per_minute_nodes) != 1 or len(dot_nodes) > 1:
        return True, None
    unit = _text(beat_nodes[0].text)
    per_minute = _text(per_minute_nodes[0].text)
    if unit not in METRONOME_UNITS or not _PER_MINUTE_RE.fullmatch(per_minute):
        return True, None
    bpm = int(per_minute)
    if bpm < 20 or bpm > 300:
        return True, None
    sound = direction.find("sound")
    if sound is not None:
        if set(sound.attrib) != {"tempo"} or _text(sound.get("tempo")) != per_minute or len(sound):
            return True, None
    value = f"{unit}{'.' if dot_nodes else ''}={per_minute}"
    return True, SimpleDirection(onset, "metronome", placement, value, sound is not None)


def normalized_simple_direction_topology(
    measure: etree._Element,
    divisions: int,
) -> tuple[SimpleDirection, ...] | None:
    cursor = Fraction(0, 1)
    result: list[SimpleDirection] = []
    slots: dict[tuple[Fraction, str, str], set[str]] = {}
    for child in measure:
        if child.tag == "direction":
            mentions, parsed = _simple_direction(child, cursor, divisions)
            if mentions and parsed is None:
                return None
            if parsed is not None:
                result.append(parsed)
                slots.setdefault((parsed.onset, parsed.kind, parsed.placement), set()).add(parsed.value)
        elif child.tag == "note":
            if child.find("grace") is not None or child.find("chord") is not None:
                continue
            try:
                duration = int(_text(child.findtext("duration")) or "0")
            except (TypeError, ValueError, OverflowError):
                return None
            if duration < 0:
                return None
            cursor += Fraction(duration, max(1, divisions))
        elif child.tag in {"backup", "forward"}:
            return None
    if any(len(values) > 1 for values in slots.values()):
        return None
    return tuple(sorted(result))


def _remove_supported_directions(measure: etree._Element, divisions: int) -> None:
    cursor = Fraction(0, 1)
    for child in list(measure):
        if child.tag == "direction":
            mentions, parsed = _simple_direction(child, cursor, divisions)
            if mentions and parsed is None:
                raise ValueError("unsupported dynamic/metronome direction")
            if parsed is not None:
                measure.remove(child)
        elif child.tag == "note":
            if child.find("grace") is not None or child.find("chord") is not None:
                continue
            duration = int(_text(child.findtext("duration")) or "0")
            cursor += Fraction(duration, max(1, divisions))
        elif child.tag in {"backup", "forward"}:
            raise ValueError("explicit cursor movement is unsupported")


def _build_direction(item: SimpleDirection, divisions: int) -> etree._Element:
    raw_offset = item.onset * max(1, divisions)
    if raw_offset.denominator != 1 or raw_offset < 0:
        raise ValueError("direction onset is not representable in template divisions")
    direction = etree.Element("direction", placement=item.placement)
    direction_type = etree.SubElement(direction, "direction-type")
    if item.kind == "dynamic":
        if item.value not in DYNAMIC_TAGS:
            raise ValueError("unsupported dynamic")
        dynamics = etree.SubElement(direction_type, "dynamics")
        etree.SubElement(dynamics, item.value)
    elif item.kind == "metronome":
        try:
            unit_with_dot, per_minute = item.value.split("=", 1)
        except ValueError as exc:
            raise ValueError("invalid metronome value") from exc
        dotted = unit_with_dot.endswith(".")
        unit = unit_with_dot[:-1] if dotted else unit_with_dot
        if unit not in METRONOME_UNITS or not _PER_MINUTE_RE.fullmatch(per_minute):
            raise ValueError("invalid metronome value")
        metronome = etree.SubElement(direction_type, "metronome")
        etree.SubElement(metronome, "beat-unit").text = unit
        if dotted:
            etree.SubElement(metronome, "beat-unit-dot")
        etree.SubElement(metronome, "per-minute").text = per_minute
    else:
        raise ValueError("unsupported direction kind")
    offset = int(raw_offset)
    if offset:
        etree.SubElement(direction, "offset").text = str(offset)
    if item.kind == "metronome" and item.sound_tempo:
        etree.SubElement(direction, "sound", tempo=item.value.split("=", 1)[1])
    return direction


def set_simple_direction_topology(
    measure: etree._Element,
    topology: Iterable[SimpleDirection],
    divisions: int,
) -> None:
    items = tuple(sorted(topology))
    _remove_supported_directions(measure, divisions)
    insert_at = 0
    for index, child in enumerate(measure):
        if child.tag in {"attributes", "print"}:
            insert_at = index + 1
        else:
            break
    for item in items:
        measure.insert(insert_at, _build_direction(item, divisions))
        insert_at += 1


def without_simple_directions(measure: etree._Element, divisions: int) -> bytes:
    clone = copy.deepcopy(measure)
    _remove_supported_directions(clone, divisions)
    return etree.tostring(clone, method="c14n", with_comments=False)
