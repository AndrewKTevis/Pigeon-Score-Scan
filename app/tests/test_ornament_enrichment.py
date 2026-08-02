from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from lxml import etree

from scorescan.layout import PageLayout, StaffSystem
from scorescan.ornament_enrichment import (
    detect_source_mordents,
    detect_source_trills,
    enrich_musicxml_with_source_ornaments,
)


def _layout() -> PageLayout:
    return PageLayout(
        width=400,
        height=180,
        systems=[
            StaffSystem(
                index=1,
                line_y=[100.0, 108.0, 116.0, 124.0, 132.0],
                top=55,
                bottom=170,
                left=10,
                right=390,
                spacing=8.0,
                barlines=[10, 390],
                measure_count=1,
            )
        ],
        confidence=0.99,
    )


def _draw_tr(image: np.ndarray, x: int, y: int) -> None:
    # Connected, staff-normalised two-letter ink group with the same geometry
    # as a compact italic music-font "tr".
    cv2.rectangle(image, (x + 5, y), (x + 7, y + 16), 0, -1)
    cv2.rectangle(image, (x, y + 2), (x + 11, y + 4), 0, -1)
    cv2.rectangle(image, (x + 12, y + 6), (x + 14, y + 16), 0, -1)
    cv2.rectangle(image, (x + 12, y + 6), (x + 18, y + 9), 0, -1)


def _draw_mordent(image: np.ndarray, x: int, y: int) -> None:
    cv2.rectangle(image, (x, y + 2), (x + 15, y + 5), 0, -1)
    cv2.line(image, (x + 8, y), (x + 8, y + 7), 0, 1)


def _score_xml() -> etree._ElementTree:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Instrument"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"
    notes: list[etree._Element] = []
    for step in ("C", "D", "E", "F"):
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "1"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
        notes.append(note)

    # Deliberately swap the two source ornaments.
    for note, kind in ((notes[1], "mordent"), (notes[3], "trill-mark")):
        notations = etree.SubElement(note, "notations")
        ornaments = etree.SubElement(notations, "ornaments")
        etree.SubElement(ornaments, kind)
    return etree.ElementTree(root)


def test_source_trill_geometry_is_distinct_from_mordent(tmp_path: Path) -> None:
    image = np.full((180, 400), 255, np.uint8)
    _draw_tr(image, 90, 60)
    _draw_mordent(image, 291, 63)
    path = tmp_path / "ornaments.png"
    cv2.imwrite(str(path), image)

    trills = detect_source_trills(path, _layout())
    mordents = detect_source_mordents(path, _layout())

    assert len(trills) == 1
    assert trills[0].x == 99.5
    assert len(mordents) == 1
    assert abs(mordents[0].x - 298.5) < 0.1


def test_source_ornaments_commit_as_one_authoritative_transaction(
    tmp_path: Path,
) -> None:
    image = np.full((180, 400), 255, np.uint8)
    _draw_tr(image, 90, 60)
    _draw_mordent(image, 291, 63)
    image_path = tmp_path / "ornaments.png"
    xml_path = tmp_path / "score.musicxml"
    cv2.imwrite(str(image_path), image)
    _score_xml().write(
        str(xml_path),
        encoding="UTF-8",
        xml_declaration=True,
    )

    report = enrich_musicxml_with_source_ornaments(
        image_path,
        xml_path,
        _layout(),
    )

    assert report.authoritative_source_commit is True
    assert report.detected_trill_count == 1
    score = etree.parse(str(xml_path))
    notes = score.getroot().findall("./part/measure/note")
    assert notes[1].find("notations/ornaments/trill-mark") is not None
    assert notes[1].find("notations/ornaments/mordent") is None
    assert notes[3].find("notations/ornaments/mordent") is not None
    assert notes[3].find("notations/ornaments/trill-mark") is None

    second = enrich_musicxml_with_source_ornaments(
        image_path,
        xml_path,
        _layout(),
    )
    assert second.inserted_mordent_count == 0
    assert second.inserted_trill_count == 0
