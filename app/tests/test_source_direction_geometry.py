from __future__ import annotations

import cv2
import numpy as np
import pytest
from lxml import etree

from scorescan.layout import PageLayout, StaffSystem, infer_score_systems
from scorescan.text_enrichment import (
    _insert_direction,
    _measure_duration,
    _snap_direction_to_notehead,
    _source_dynamic_rows,
    _source_staff_target,
    _visual_notehead_columns,
)


def _system() -> StaffSystem:
    return StaffSystem(
        index=0,
        line_y=[100.0, 110.0, 120.0, 130.0, 140.0],
        top=75,
        bottom=195,
        left=10,
        right=390,
        spacing=10.0,
        barlines=[],
        measure_count=1,
    )


def _three_note_measure() -> etree._Element:
    measure = etree.Element("measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    for step in ("C", "D", "E"):
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "1"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
    return measure


def test_source_dynamic_geometry_recovers_sf_and_p() -> None:
    image = np.full((220, 420), 255, np.uint8)

    # An f-like diagonal/hooked component: upper-right and lower-left mass.
    cv2.rectangle(image, (112, 153), (120, 155), 0, -1)
    cv2.line(image, (112, 155), (109, 170), 0, 2)
    cv2.rectangle(image, (102, 169), (108, 175), 0, -1)

    # A separate compact s-like component immediately to the lower-left.
    cv2.rectangle(image, (94, 160), (99, 161), 0, -1)
    cv2.rectangle(image, (94, 160), (95, 164), 0, -1)
    cv2.rectangle(image, (94, 164), (99, 165), 0, -1)
    cv2.rectangle(image, (98, 164), (99, 168), 0, -1)
    cv2.rectangle(image, (94, 168), (99, 169), 0, -1)

    # A p-like component: stem plus bowl strokes.
    cv2.rectangle(image, (200, 155), (202, 172), 0, -1)
    cv2.rectangle(image, (200, 155), (217, 157), 0, -1)
    cv2.rectangle(image, (200, 163), (211, 165), 0, -1)

    layout = PageLayout(
        width=image.shape[1],
        height=image.shape[0],
        systems=[_system()],
        confidence=0.99,
    )
    rows = _source_dynamic_rows(image, layout)

    assert [row[0] for row in rows] == ["sf", "p"]
    assert all(row[3].startswith("source-dynamic-geometry") for row in rows)


def test_direction_snaps_to_matching_visual_notehead_onset() -> None:
    system = _system()
    binary = np.zeros((220, 420), np.uint8)
    for center_x in (100, 200, 300):
        cv2.ellipse(binary, (center_x, 120), (6, 5), 0, 0, 360, 255, -1)

    columns = _visual_notehead_columns(binary, system)

    assert columns == pytest.approx((100.5, 200.5, 300.5), abs=1.0)
    measure = _three_note_measure()
    offset = _snap_direction_to_notehead(
        x=202.0,
        system=system,
        local_index=0,
        group=[measure],
        notehead_columns=columns,
    )
    assert offset == pytest.approx(1.0 / 3.0)


def test_visual_notehead_columns_recovers_hollow_notehead() -> None:
    system = _system()
    binary = np.zeros((220, 420), np.uint8)
    for line_y in system.line_y:
        cv2.line(
            binary,
            (system.left, int(round(line_y))),
            (system.right, int(round(line_y))),
            255,
            1,
        )
    cv2.ellipse(binary, (200, 120), (6, 4), -15, 0, 360, 255, 2)
    cv2.line(binary, (206, 120), (206, 82), 255, 2)

    columns = _visual_notehead_columns(binary, system)

    assert any(abs(column - 200.0) <= 2.0 for column in columns)


def test_direction_notehead_snap_abstains_when_event_counts_disagree() -> None:
    offset = _snap_direction_to_notehead(
        x=200.0,
        system=_system(),
        local_index=0,
        group=[_three_note_measure()],
        notehead_columns=(100.0, 200.0),
    )

    assert offset is None


def test_direction_snaps_to_opening_onset_before_first_filled_notehead() -> None:
    offset = _snap_direction_to_notehead(
        x=102.0,
        system=_system(),
        local_index=0,
        group=[_three_note_measure()],
        # The first (hollow) notehead is deliberately absent.
        notehead_columns=(200.0, 300.0),
    )

    assert offset == 0.0


def test_full_score_source_staff_maps_to_part_local_staff() -> None:
    physical = []
    for index, y0 in enumerate((100, 180, 260, 440, 520, 600), start=1):
        physical.append(
            StaffSystem(
                index=index,
                line_y=[float(y0 + offset) for offset in (0, 10, 20, 30, 40)],
                top=y0 - 20,
                bottom=y0 + 60,
                left=80,
                right=900,
                spacing=10,
            )
        )
    layout = PageLayout(
        1000,
        760,
        physical,
        0.99,
        score_systems=infer_score_systems(physical),
    )

    assert _source_staff_target(layout, 0, [1, 2])[:3] == (0, 0, 1)
    assert _source_staff_target(layout, 1, [1, 2])[:3] == (0, 1, 1)
    assert _source_staff_target(layout, 2, [1, 2])[:3] == (0, 1, 2)
    assert _source_staff_target(layout, 5, [1, 2])[:3] == (1, 1, 2)


def test_direction_uses_maximum_multivoice_timeline_and_target_staff() -> None:
    measure = etree.Element("measure", number="1")
    for voice, duration in (("1", 4), ("2", 4)):
        if voice == "2":
            backup = etree.SubElement(measure, "backup")
            etree.SubElement(backup, "duration").text = "4"
        note = etree.SubElement(measure, "note")
        etree.SubElement(note, "rest")
        etree.SubElement(note, "duration").text = str(duration)
        etree.SubElement(note, "voice").text = voice
        etree.SubElement(note, "staff").text = "2"

    assert _measure_duration(measure) == 4
    assert _insert_direction(
        measure,
        "mf",
        "dynamic",
        0.5,
        "below",
        staff_number=2,
    )
    direction = measure.find("direction")
    assert direction is not None
    assert direction.findtext("offset") == "2"
    assert direction.findtext("staff") == "2"
