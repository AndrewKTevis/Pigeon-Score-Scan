from __future__ import annotations

from lxml import etree

from scorescan.musicxml_signature import (
    canonical_measure_bytes,
    measure_preservation_signature,
    measure_preservation_signatures,
)


def _measure(*, divisions: int = 1, duration: int = 1) -> etree._Element:
    measure = etree.Element("measure", number="1", width="900")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = str(divisions)
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"
    note = etree.SubElement(measure, "note", **{"default-x": "42"})
    pitch = etree.SubElement(note, "pitch")
    etree.SubElement(pitch, "step").text = "C"
    etree.SubElement(pitch, "octave").text = "4"
    etree.SubElement(note, "duration").text = str(duration)
    etree.SubElement(note, "voice").text = "1"
    etree.SubElement(note, "type").text = "quarter"
    return measure


def _note(measure: etree._Element) -> etree._Element:
    note = measure.find("note")
    assert note is not None
    return note


def test_preservation_signature_normalizes_divisions_and_layout() -> None:
    left = _measure(divisions=1, duration=1)
    right = _measure(divisions=4, duration=4)
    right.set("number", "99")
    right.set("width", "300")
    _note(right).set("default-x", "987")

    assert measure_preservation_signature(left) == measure_preservation_signature(right)


def test_preservation_signature_covers_unmodelled_writeback_objects() -> None:
    base = _measure()
    beam_changed = etree.fromstring(etree.tostring(base))
    etree.SubElement(_note(beam_changed), "beam", number="1").text = "begin"

    notehead_changed = etree.fromstring(etree.tostring(base))
    etree.SubElement(_note(notehead_changed), "notehead").text = "diamond"

    fermata_changed = etree.fromstring(etree.tostring(base))
    notations = etree.SubElement(_note(fermata_changed), "notations")
    etree.SubElement(notations, "fermata", type="upright").text = "normal"

    lyric_changed = etree.fromstring(etree.tostring(base))
    lyric = etree.SubElement(_note(lyric_changed), "lyric", number="1")
    etree.SubElement(lyric, "text").text = "La"

    signatures = {
        measure_preservation_signature(item)
        for item in (base, beam_changed, notehead_changed, fermata_changed, lyric_changed)
    }
    assert len(signatures) == 5


def test_preservation_signature_carries_inherited_divisions() -> None:
    first_left = _measure(divisions=2, duration=2)
    second_left = _measure(divisions=2, duration=2)
    second_left.remove(second_left.find("attributes"))

    first_right = _measure(divisions=4, duration=4)
    second_right = _measure(divisions=4, duration=4)
    second_right.remove(second_right.find("attributes"))

    assert measure_preservation_signatures((first_left, second_left)) == measure_preservation_signatures(
        (first_right, second_right)
    )


def test_preservation_signature_keeps_mid_measure_attribute_order() -> None:
    left = _measure()
    right = etree.fromstring(etree.tostring(left))
    mid = etree.Element("attributes")
    clef = etree.SubElement(mid, "clef")
    etree.SubElement(clef, "sign").text = "F"
    etree.SubElement(clef, "line").text = "4"
    right.insert(len(right) - 1, mid)

    assert measure_preservation_signature(left) != measure_preservation_signature(right)
    payload, ending = canonical_measure_bytes(right)
    assert ending == 1
    assert b"<clef>" in payload
