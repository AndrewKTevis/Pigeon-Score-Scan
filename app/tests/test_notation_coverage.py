from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from lxml import etree

from scorescan.layout import PageLayout, StaffSystem
from scorescan.models import PageInfo
from scorescan.notation_coverage import (
    VisualNotationCandidate,
    _merge_wedge_fragments,
    _resolve_wedge_kind_conflicts,
    _wedge_staff_owner,
    audit_notation_coverage,
    detect_notation_candidates,
    wedge_source_specificity_gate,
)


def _synthetic_page(path: Path) -> PageLayout:
    image = np.full((260, 800), 255, np.uint8)
    for y in (120, 130, 140, 150, 160):
        cv2.line(image, (40, y), (760, y), 0, 1)
    # A clean slur above the staff.
    cv2.ellipse(image, (510, 100), (70, 25), 0, 180, 360, 0, 2)
    # Crescendo hairpin below the staff (apex on the left).
    cv2.line(image, (180, 190), (310, 176), 0, 2)
    cv2.line(image, (180, 190), (310, 204), 0, 2)
    cv2.imwrite(str(path), image)
    staff = StaffSystem(
        1,
        [120, 130, 140, 150, 160],
        70,
        220,
        40,
        760,
        10,
        [40, 760],
        1,
    )
    return PageLayout(800, 260, [staff], 1.0)


def _write_xml(path: Path, *, include_slur: bool, include_wedge: bool) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    note = etree.SubElement(measure, "note")
    pitch = etree.SubElement(note, "pitch")
    etree.SubElement(pitch, "step").text = "C"
    etree.SubElement(pitch, "octave").text = "4"
    etree.SubElement(note, "duration").text = "4"
    etree.SubElement(note, "voice").text = "1"
    etree.SubElement(note, "type").text = "whole"
    if include_slur:
        notations = etree.SubElement(note, "notations")
        etree.SubElement(notations, "slur", type="start", number="1")
        stop_note = etree.SubElement(measure, "note")
        stop_pitch = etree.SubElement(stop_note, "pitch")
        etree.SubElement(stop_pitch, "step").text = "D"
        etree.SubElement(stop_pitch, "octave").text = "4"
        etree.SubElement(stop_note, "duration").text = "4"
        etree.SubElement(stop_note, "voice").text = "1"
        stop_notations = etree.SubElement(stop_note, "notations")
        etree.SubElement(stop_notations, "slur", type="stop", number="1")
    if include_wedge:
        start = etree.SubElement(measure, "direction", placement="below")
        start_type = etree.SubElement(start, "direction-type")
        etree.SubElement(start_type, "wedge", type="crescendo", number="1")
        stop = etree.SubElement(measure, "direction", placement="below")
        stop_type = etree.SubElement(stop, "direction-type")
        etree.SubElement(stop_type, "wedge", type="stop", number="1")
    path.write_bytes(etree.tostring(root, encoding="UTF-8", xml_declaration=True))


def test_source_inventory_detects_slur_and_hairpin_independently(tmp_path: Path) -> None:
    image_path = tmp_path / "notation.png"
    layout = _synthetic_page(image_path)

    candidates = detect_notation_candidates(image_path, layout)

    assert any(
        candidate.kind == "curved_connector" and candidate.confidence >= 0.78
        for candidate in candidates
    )
    assert any(candidate.kind == "crescendo" and candidate.confidence >= 0.82 for candidate in candidates)


def test_source_inventory_retains_long_shallow_hairpin(tmp_path: Path) -> None:
    image_path = tmp_path / "long-shallow-hairpin.png"
    image = np.full((260, 800), 255, np.uint8)
    for y in (120, 130, 140, 150, 160):
        cv2.line(image, (40, y), (760, y), 0, 1)
    # Both strokes are shallower than the legacy Hough slope floor.
    cv2.line(image, (180, 190), (540, 182), 0, 2)
    cv2.line(image, (180, 190), (540, 198), 0, 2)
    cv2.imwrite(str(image_path), image)
    staff = StaffSystem(
        1,
        [120, 130, 140, 150, 160],
        70,
        220,
        40,
        760,
        10,
        [40, 760],
        1,
    )
    layout = PageLayout(800, 260, [staff], 1.0)

    candidates = detect_notation_candidates(image_path, layout)
    eligible = [
        candidate
        for candidate in candidates
        if candidate.kind == "crescendo"
        and wedge_source_specificity_gate(candidate, candidates)[0]
    ]

    assert len(eligible) == 1
    assert eligible[0].bbox[0] <= 185
    assert eligible[0].bbox[2] >= 535


def test_source_inventory_rejects_filled_slanted_beam_as_hairpin(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "slanted-beam.png"
    image = np.full((260, 800), 255, np.uint8)
    for y in (120, 130, 140, 150, 160):
        cv2.line(image, (40, y), (760, y), 0, 1)
    # Opposite beam-edge slopes can resemble a hairpin to a line-pair detector,
    # but the filled interior is decisive negative source evidence.
    polygon = np.array([(180, 82), (300, 88), (300, 97), (180, 94)], np.int32)
    cv2.fillConvexPoly(image, polygon, 0)
    cv2.line(image, (180, 94), (180, 130), 0, 2)
    cv2.line(image, (300, 97), (300, 130), 0, 2)
    cv2.imwrite(str(image_path), image)
    staff = StaffSystem(
        1,
        [120, 130, 140, 150, 160],
        70,
        220,
        40,
        760,
        10,
        [40, 760],
        1,
    )
    layout = PageLayout(800, 260, [staff], 1.0)

    candidates = detect_notation_candidates(image_path, layout)

    assert not [
        candidate
        for candidate in candidates
        if candidate.kind in {"crescendo", "diminuendo"}
        and wedge_source_specificity_gate(candidate, candidates)[0]
    ]


def test_overlapping_hough_views_form_one_physical_hairpin() -> None:
    staff = StaffSystem(
        1,
        [120, 130, 140, 150, 160],
        70,
        220,
        40,
        760,
        10,
    )
    fragments = (
        VisualNotationCandidate("crescendo", 1, "below", (180, 182, 225, 196), 0.91),
        VisualNotationCandidate("crescendo", 1, "below", (210, 176, 310, 204), 0.97),
        VisualNotationCandidate("crescendo", 1, "below", (285, 175, 340, 205), 0.89),
        VisualNotationCandidate("crescendo", 1, "below", (420, 180, 500, 205), 0.93),
    )

    merged = _merge_wedge_fragments(fragments, [staff])

    assert len(merged) == 2
    assert merged[0].bbox == (180, 175, 340, 205)
    assert merged[0].confidence == 0.97
    assert dict(merged[0].geometry)["merged_hough_views"] == 3.0
    assert merged[1].bbox == (420, 180, 500, 205)


def test_between_staff_hough_views_share_one_recomputed_owner() -> None:
    upper = StaffSystem(
        1,
        [120, 130, 140, 150, 160],
        70,
        220,
        40,
        760,
        10,
    )
    lower = StaffSystem(
        2,
        [280, 290, 300, 310, 320],
        260,
        370,
        40,
        760,
        10,
    )
    fragments = (
        VisualNotationCandidate(
            "diminuendo",
            1,
            "below",
            (180, 198, 420, 224),
            0.96,
        ),
        VisualNotationCandidate(
            "diminuendo",
            2,
            "above",
            (180, 207, 350, 224),
            0.91,
        ),
    )

    merged = _merge_wedge_fragments(fragments, [upper, lower])

    assert len(merged) == 1
    assert merged[0].bbox == (180, 198, 420, 224)
    assert merged[0].staff_index == 1
    assert merged[0].placement == "below"


def test_interstitial_ossia_wedge_keeps_a_bounded_provisional_owner() -> None:
    upper = StaffSystem(
        1,
        [100, 110, 120, 130, 140],
        70,
        170,
        40,
        760,
        10,
    )
    lower = StaffSystem(
        2,
        [360, 370, 380, 390, 400],
        330,
        430,
        40,
        760,
        10,
    )

    owner = _wedge_staff_owner([upper, lower], 250)

    assert owner is not None
    assert owner[0].index == 1
    assert owner[1] == "below"
    assert owner[2] == 11.0
    assert _wedge_staff_owner([upper, lower], 290) is None


def test_opposite_kind_slur_pair_does_not_override_sharper_hairpin() -> None:
    staff = StaffSystem(
        1,
        [120, 130, 140, 150, 160],
        70,
        240,
        40,
        760,
        10,
    )
    candidates = (
        VisualNotationCandidate(
            "crescendo",
            1,
            "below",
            (180, 175, 340, 205),
            0.94,
            (
                ("apex_separation_spaces", 0.45),
                ("open_separation_spaces", 1.50),
            ),
        ),
        VisualNotationCandidate(
            "diminuendo",
            1,
            "below",
            (182, 198, 342, 228),
            0.92,
            (
                ("apex_separation_spaces", 0.18),
                ("open_separation_spaces", 1.45),
            ),
        ),
    )

    resolved = _resolve_wedge_kind_conflicts(candidates, [staff])

    assert len(resolved) == 1
    assert resolved[0].kind == "diminuendo"


def test_short_scan_hairpin_allows_bounded_halftone_density() -> None:
    short = VisualNotationCandidate(
        "diminuendo",
        1,
        "below",
        (180, 180, 210, 200),
        0.90,
        (
            ("apex_separation_spaces", 0.20),
            ("open_separation_spaces", 1.20),
            ("length_spaces", 3.0),
            ("ink_density", 0.36),
        ),
    )
    long_filled = VisualNotationCandidate(
        "diminuendo",
        1,
        "below",
        (180, 180, 230, 200),
        0.90,
        (
            ("apex_separation_spaces", 0.20),
            ("open_separation_spaces", 1.20),
            ("length_spaces", 5.0),
            ("ink_density", 0.36),
        ),
    )

    assert wedge_source_specificity_gate(short, (short,))[0] is True
    assert wedge_source_specificity_gate(long_filled, (long_filled,))[0] is False


def test_clean_interstitial_ossia_hairpin_has_a_bounded_confidence_floor() -> None:
    geometry = (
        ("apex_separation_spaces", 0.25),
        ("open_separation_spaces", 1.20),
        ("length_spaces", 5.8),
        ("ink_density", 0.08),
    )
    ordinary = VisualNotationCandidate(
        "diminuendo",
        1,
        "below",
        (180, 180, 238, 200),
        0.83,
        geometry,
    )
    interstitial = VisualNotationCandidate(
        "diminuendo",
        1,
        "below",
        (180, 180, 238, 200),
        0.83,
        (*geometry, ("interstitial_owner", 1.0)),
    )

    assert wedge_source_specificity_gate(ordinary, (ordinary,))[0] is False
    assert wedge_source_specificity_gate(
        interstitial,
        (interstitial,),
    )[0] is True


def test_coverage_audit_exposes_silent_hairpin_omission(tmp_path: Path) -> None:
    image_path = tmp_path / "notation.png"
    xml_path = tmp_path / "score.musicxml"
    layout = _synthetic_page(image_path)
    _write_xml(xml_path, include_slur=True, include_wedge=False)

    report = audit_notation_coverage(image_path, xml_path, layout)
    kinds = {item.kind: item for item in report.kinds}

    assert kinds["crescendo"].confident_source_count >= 1
    assert kinds["crescendo"].emitted_count == 0
    assert kinds["crescendo"].potential_omission_count >= 1
    assert report.potential_omission_count >= 1


def test_coverage_inventory_excludes_curve_conflicted_wedge_candidate(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "unused.png"
    image_path.write_bytes(b"unused")
    xml_path = tmp_path / "score.musicxml"
    _write_xml(xml_path, include_slur=False, include_wedge=False)
    staff = StaffSystem(
        1,
        [120, 130, 140, 150, 160],
        70,
        220,
        40,
        760,
        10,
        [40, 760],
        1,
    )
    layout = PageLayout(800, 260, [staff], 1.0)
    wedge = VisualNotationCandidate(
        "crescendo",
        1,
        "above",
        (300, 90, 360, 112),
        0.88,
        (
            ("length_spaces", 4.5),
            ("open_separation_spaces", 1.1),
            ("apex_separation_spaces", 0.2),
        ),
    )
    curve = VisualNotationCandidate(
        "curved_connector",
        1,
        "above",
        (295, 92, 370, 110),
        0.90,
        (
            ("fit_p90_spaces", 0.04),
            ("length_spaces", 6.0),
        ),
    )

    report = audit_notation_coverage(
        image_path,
        xml_path,
        layout,
        candidates=(wedge, curve),
    )
    kinds = {item.kind: item for item in report.kinds}

    assert kinds["crescendo"].confident_source_count == 0
    assert kinds["crescendo"].potential_omission_count == 0


def test_coverage_audit_balances_emitted_objects(tmp_path: Path) -> None:
    image_path = tmp_path / "notation.png"
    xml_path = tmp_path / "score.musicxml"
    layout = _synthetic_page(image_path)
    _write_xml(xml_path, include_slur=True, include_wedge=True)

    report = audit_notation_coverage(image_path, xml_path, layout)

    assert report.emitted_unbalanced_slurs == 0
    assert report.emitted_unbalanced_wedges == 0


def test_curve_count_excess_is_diagnostic_without_object_level_matching(
    tmp_path: Path,
) -> None:
    """A split rendering of one physical slur must not create a false doubt."""

    from scorescan.review import build_notation_coverage_review_issues

    image_path = tmp_path / "notation.png"
    xml_path = tmp_path / "score.musicxml"
    layout = _synthetic_page(image_path)
    _write_xml(xml_path, include_slur=True, include_wedge=True)

    report = audit_notation_coverage(image_path, xml_path, layout)
    kinds = {item.kind: item for item in report.kinds}
    curve = kinds["curved_connector"]

    # The independent component detector deliberately sees both disconnected
    # halves of the one printed ellipse in this fixture.  A count subtraction is
    # therefore useful diagnostics but is not evidence of an omitted relation.
    assert curve.confident_source_count > curve.emitted_count
    assert curve.source_count_excess > 0
    assert curve.comparison_mode == "diagnostic-count-only"
    assert curve.potential_omission_count == 0
    assert report.potential_omission_count == 0

    page = PageInfo(1, image_path.name, str(image_path), normalized_path=str(image_path))
    issues = build_notation_coverage_review_issues(
        [page],
        {1: report},
        tmp_path / "result",
    )
    assert not [issue for issue in issues if issue.kind == "curved_connector"]


def test_unbalanced_curve_topology_remains_actionable(tmp_path: Path) -> None:
    image_path = tmp_path / "notation.png"
    xml_path = tmp_path / "score.musicxml"
    layout = _synthetic_page(image_path)
    _write_xml(xml_path, include_slur=True, include_wedge=True)
    tree = etree.parse(str(xml_path))
    stop = tree.find(".//slur[@type='stop']")
    assert stop is not None
    stop.getparent().remove(stop)
    tree.write(str(xml_path), encoding="UTF-8", xml_declaration=True)

    report = audit_notation_coverage(image_path, xml_path, layout)

    assert report.emitted_unbalanced_slurs == 1
    assert report.severe_structure_issue_count == 1


def test_coverage_review_groups_candidates_instead_of_creating_one_doubt_each(
    tmp_path: Path,
) -> None:
    from scorescan.review import build_notation_coverage_review_issues

    image_path = tmp_path / "notation.png"
    xml_path = tmp_path / "score.musicxml"
    layout = _synthetic_page(image_path)
    _write_xml(xml_path, include_slur=True, include_wedge=False)
    report = audit_notation_coverage(image_path, xml_path, layout)
    page = PageInfo(1, image_path.name, str(image_path), normalized_path=str(image_path))

    issues = build_notation_coverage_review_issues(
        [page],
        {1: report},
        tmp_path / "result",
    )

    wedge_issues = [issue for issue in issues if issue.kind == "wedge"]
    assert len(wedge_issues) == 1
    assert wedge_issues[0].writeback_supported is False
    assert "位置未匹配" in wedge_issues[0].message
    assert "高置信" not in wedge_issues[0].message
    assert "程序" not in wedge_issues[0].message
    assert Path(wedge_issues[0].crop_path).is_file()
    assert page.review_issue_count == len(issues)
