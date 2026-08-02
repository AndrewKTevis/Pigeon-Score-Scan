from __future__ import annotations

"""Canonical MusicXML preservation signatures.

Score IR is intentionally compact and does not model every engraving or playback
object that ScoreScan preserves.  Whole-measure consensus, however, copies the entire
selected MusicXML measure.  Its exact-agreement gate must therefore compare everything
that can be written back, not only the currently modelled note timeline.

The canonicaliser removes layout-only coordinates and representational ``divisions``
choices while preserving musical and printed notation semantics, including unknown
MusicXML children.  Durations and offsets are normalised to exact rational quarter-note
units so equivalent encodings compare equal without rounding.
"""

import copy
import hashlib
from fractions import Fraction
from typing import Iterable, Sequence

from lxml import etree


_LAYOUT_ONLY_ATTRIBUTES = frozenset(
    {
        "default-x",
        "default-y",
        "relative-x",
        "relative-y",
        "font-family",
        "font-size",
        "font-style",
        "font-weight",
        "color",
        "print-spacing",
        "justify",
        "halign",
        "valign",
        "rotation",
        "letter-spacing",
        "line-height",
        "dir",
    }
)

_MEASURE_LAYOUT_ATTRIBUTES = frozenset({"number", "width", "id"})
_COLLAPSE_WHITESPACE_TAGS = frozenset(
    {
        "words",
        "text",
        "other-dynamics",
        "elision",
        "rehearsal",
        "instrument-link",
    }
)
_CASEFOLD_TEXT_TAGS = frozenset({"syllabic"})
_TIMING_TAGS = frozenset({"duration", "offset"})


def _local_name(value: object) -> str:
    return etree.QName(value).localname


def _positive_integer(text: str | None, *, source: str) -> int:
    try:
        value = int((text or "").strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} divisions is not an integer") from exc
    if value <= 0:
        raise ValueError(f"{source} divisions must be positive")
    return value


def _normalised_fraction(text: str | None, *, source: str) -> str:
    try:
        value = Fraction((text or "").strip())
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{source} is not a rational number") from exc
    return f"{value.numerator}/{value.denominator}"


def _normalised_timing(text: str | None, divisions: int | None, *, source: str) -> str:
    try:
        value = int((text or "").strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} is not an integer") from exc
    if divisions is None:
        # Valid MusicXML should establish divisions before performed content.  Keeping
        # a tagged raw value is safer than guessing a unit and still deterministic for
        # malformed candidates that reach diagnostics.
        return f"raw:{value}"
    normalised = Fraction(value, divisions)
    return f"{normalised.numerator}/{normalised.denominator}"


def _canonical_element(node: etree._Element, divisions: int | None) -> etree._Element:
    copied = etree.Element(_local_name(node.tag))
    for key, value in node.attrib.items():
        name = _local_name(key)
        if name in _LAYOUT_ONLY_ATTRIBUTES:
            continue
        copied.set(name, " ".join(value.split()) if name in {"number", "type"} else value.strip())

    tag = _local_name(node.tag)
    if node.text is not None:
        if tag in _TIMING_TAGS:
            copied.text = _normalised_timing(node.text, divisions, source=tag)
        elif tag == "alter":
            copied.text = _normalised_fraction(node.text, source="alter")
        elif tag in _COLLAPSE_WHITESPACE_TAGS:
            copied.text = " ".join(node.text.split())
        elif tag in _CASEFOLD_TEXT_TAGS:
            copied.text = node.text.strip().casefold()
        else:
            copied.text = node.text.strip()

    for child in node:
        if not isinstance(child.tag, str):
            continue
        copied.append(_canonical_element(child, divisions))
    return copied


def _canonical_attributes(
    attributes: etree._Element,
    divisions: int | None,
) -> tuple[etree._Element | None, int | None]:
    canonical = etree.Element("attributes")
    current = divisions
    for child in attributes:
        if not isinstance(child.tag, str):
            continue
        tag = _local_name(child.tag)
        if tag == "divisions":
            current = _positive_integer(child.text, source="MusicXML")
            continue
        canonical.append(_canonical_element(child, current))
    for key, value in attributes.attrib.items():
        name = _local_name(key)
        if name not in _LAYOUT_ONLY_ATTRIBUTES:
            canonical.set(name, value.strip())
    return (canonical if len(canonical) or canonical.attrib else None), current


def canonical_measure_bytes(
    measure: etree._Element,
    *,
    inherited_divisions: int | None = None,
    include_attributes: bool = True,
    require_performed_content: bool = False,
) -> tuple[bytes, int | None]:
    """Return canonical write-back content and the ending divisions state.

    ``print`` elements and layout-only coordinates are removed.  Attribute groups are
    preserved except for ``divisions`` itself, whose effect is used to normalise every
    following duration/offset exactly.  Mid-measure changes are represented in order.
    """

    wrapper = etree.Element("measure-content")
    for key, value in measure.attrib.items():
        name = _local_name(key)
        if name in _MEASURE_LAYOUT_ATTRIBUTES or name in _LAYOUT_ONLY_ATTRIBUTES:
            continue
        wrapper.set(name, value.strip())

    current_divisions = inherited_divisions
    performed_count = 0
    for child in measure:
        if not isinstance(child.tag, str):
            continue
        tag = _local_name(child.tag)
        if tag == "print":
            continue
        if tag == "attributes":
            canonical, current_divisions = _canonical_attributes(child, current_divisions)
            if include_attributes and canonical is not None:
                wrapper.append(canonical)
            continue
        wrapper.append(_canonical_element(child, current_divisions))
        performed_count += 1

    if require_performed_content and performed_count == 0:
        raise ValueError("MusicXML measure contains no canonical performed content")
    return etree.tostring(wrapper, method="c14n", with_comments=False), current_divisions


def measure_preservation_signatures(
    measures: Sequence[etree._Element] | Iterable[etree._Element],
) -> tuple[str, ...]:
    """Hash a measure sequence while carrying inherited ``divisions`` state."""

    signatures: list[str] = []
    divisions: int | None = None
    for measure in measures:
        payload, divisions = canonical_measure_bytes(
            measure,
            inherited_divisions=divisions,
            include_attributes=True,
        )
        signatures.append(hashlib.sha256(payload).hexdigest()[:20])
    return tuple(signatures)


def measure_preservation_signature(measure: etree._Element) -> str:
    """Hash one standalone measure for tests and diagnostics.

    Sequence consumers should use :func:`measure_preservation_signatures` so inherited
    ``divisions`` are interpreted correctly.
    """

    payload, _ = canonical_measure_bytes(measure, include_attributes=True)
    return hashlib.sha256(payload).hexdigest()[:20]


def splice_content_signature(measure: etree._Element, source_divisions: int) -> str:
    """Hash only content that a localised recognition result can splice.

    Crop-local attributes are excluded because the complete-page template remains
    authoritative.  The caller supplies the already validated, stable local divisions
    unit used by every performed child.
    """

    content = etree.Element("measure-content")
    for child in measure:
        if not isinstance(child.tag, str):
            continue
        tag = _local_name(child.tag)
        if tag in {"print", "attributes"}:
            continue
        content.append(_canonical_element(copy.deepcopy(child), source_divisions))
    if len(content) == 0:
        raise ValueError("MusicXML measure contains no spliceable content")
    payload = etree.tostring(content, method="c14n", with_comments=False)
    return hashlib.sha256(payload).hexdigest()[:20]
