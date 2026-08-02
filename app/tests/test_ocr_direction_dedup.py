from __future__ import annotations

from lxml import etree

from scorescan.text_enrichment import _insert_direction


def _measure() -> etree._Element:
    measure = etree.Element("measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "4"
    note = etree.SubElement(measure, "note")
    etree.SubElement(note, "rest")
    etree.SubElement(note, "duration").text = "16"
    etree.SubElement(note, "voice").text = "1"
    etree.SubElement(note, "type").text = "whole"
    return measure


def test_ocr_does_not_duplicate_existing_metronome_with_different_text_notation() -> None:
    measure = _measure()
    direction = etree.Element("direction", placement="above")
    direction_type = etree.SubElement(direction, "direction-type")
    metronome = etree.SubElement(direction_type, "metronome")
    etree.SubElement(metronome, "beat-unit").text = "quarter"
    etree.SubElement(metronome, "per-minute").text = "120"
    measure.insert(1, direction)

    assert not _insert_direction(measure, "♩ = 120", "metronome", 0.0, "above")
    assert len(measure.findall("direction")) == 1


def test_ocr_dedup_distinguishes_different_metronome_values() -> None:
    measure = _measure()
    assert _insert_direction(measure, "quarter = 120", "metronome", 0.0, "above")
    assert _insert_direction(measure, "quarter = 132", "metronome", 0.0, "above")
    assert len(measure.findall("direction")) == 2
