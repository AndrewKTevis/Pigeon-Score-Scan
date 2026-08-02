from __future__ import annotations

from pathlib import Path

from lxml import etree

from scorescan.layout import PageLayout, StaffSystem
from scorescan.musicxml import MUSICXML_DOCTYPE, analyze_musicxml
from scorescan.notation_coverage import VisualNotationCandidate
from scorescan.wedge_enrichment import (
    _appearance_location,
    _assign_bbox_to_appearance,
    _build_topology,
    _candidate_for_topology,
    _conditioned_staff_appearances,
    _insert_wedge_direction,
    enrich_musicxml_with_wedges,
)


def _staff(index: int, top_line: int) -> StaffSystem:
    return StaffSystem(
        index,
        [top_line + offset for offset in (0, 10, 20, 30, 40)],
        top_line - 45,
        top_line + 75,
        40,
        760,
        10,
        [240, 440, 640, 760],
        4,
    )


def _write_score(
    path: Path,
    *,
    part_staff_counts: tuple[int, ...] = (1,),
    measures: int = 4,
    existing_wedge: bool = False,
    system_breaks: tuple[int, ...] = (),
) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    for part_index in range(len(part_staff_counts)):
        part_id = f"P{part_index + 1}"
        score_part = etree.SubElement(part_list, "score-part", id=part_id)
        etree.SubElement(score_part, "part-name").text = f"Part {part_index + 1}"
    for part_index, staff_count in enumerate(part_staff_counts):
        part_id = f"P{part_index + 1}"
        part = etree.SubElement(root, "part", id=part_id)
        for measure_index in range(measures):
            measure = etree.SubElement(part, "measure", number=str(measure_index + 1))
            if measure_index in system_breaks:
                etree.SubElement(measure, "print", **{"new-system": "yes"})
            if measure_index == 0:
                attributes = etree.SubElement(measure, "attributes")
                etree.SubElement(attributes, "divisions").text = "1"
                if staff_count > 1:
                    etree.SubElement(attributes, "staves").text = str(staff_count)
                time = etree.SubElement(attributes, "time")
                etree.SubElement(time, "beats").text = "4"
                etree.SubElement(time, "beat-type").text = "4"
            note = etree.SubElement(measure, "note")
            etree.SubElement(note, "rest", measure="yes")
            etree.SubElement(note, "duration").text = "4"
            etree.SubElement(note, "voice").text = "1"
            etree.SubElement(note, "type").text = "whole"
            etree.SubElement(note, "staff").text = "1"
        if existing_wedge and part_index == 0:
            first = part.findall("measure")[0]
            for kind in ("crescendo", "stop"):
                direction = etree.SubElement(first, "direction", placement="below")
                direction_type = etree.SubElement(direction, "direction-type")
                etree.SubElement(direction_type, "wedge", type=kind, number="1")
                etree.SubElement(direction, "staff").text = "1"
    etree.ElementTree(root).write(
        str(path),
        encoding="UTF-8",
        xml_declaration=True,
        doctype=MUSICXML_DOCTYPE,
    )


def _candidate(staff_index: int = 1) -> VisualNotationCandidate:
    return VisualNotationCandidate(
        "crescendo",
        staff_index,
        "below",
        (100, 170, 300, 198),
        0.99,
        (
            ("length_spaces", 20.0),
            ("open_separation_spaces", 1.4),
            ("apex_separation_spaces", 0.1),
            ("ink_density", 0.12),
        ),
    )


def _medium_candidate(
    *,
    apex_separation: float = 0.20,
) -> VisualNotationCandidate:
    return VisualNotationCandidate(
        "diminuendo",
        1,
        "below",
        (300, 170, 360, 190),
        0.88,
        (
            ("length_spaces", 4.5),
            ("open_separation_spaces", 1.10),
            ("apex_separation_spaces", apex_separation),
        ),
    )


def test_high_confidence_hairpin_is_inserted_as_balanced_transaction(tmp_path: Path) -> None:
    xml_path = tmp_path / "score.musicxml"
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"not-read-when-candidates-are-supplied")
    _write_score(xml_path)
    layout = PageLayout(800, 260, [_staff(1, 120)], 1.0)
    before = analyze_musicxml(xml_path)

    report = enrich_musicxml_with_wedges(
        image_path,
        xml_path,
        layout,
        candidates=(_candidate(),),
    )

    assert report.transaction_committed
    assert report.injected_count == 1
    tree = etree.parse(str(xml_path))
    wedges = tree.findall("./part/measure/direction/direction-type/wedge")
    assert [item.get("type") for item in wedges] == ["crescendo", "stop"]
    assert [item.get("number") for item in wedges] == ["1", "1"]
    after = analyze_musicxml(xml_path)
    for key in ("part_count", "measure_count", "note_count", "rest_count", "rhythm_issues"):
        assert after[key] == before[key]


def test_same_measure_hairpin_directions_are_kept_in_time_order() -> None:
    measure = etree.Element("measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "4"
    for _index in range(4):
        note = etree.SubElement(measure, "note")
        etree.SubElement(note, "rest")
        etree.SubElement(note, "duration").text = "4"

    _insert_wedge_direction(
        measure,
        kind="crescendo",
        number=1,
        placement="below",
        staff=1,
        offset_ratio=0.25,
    )
    _insert_wedge_direction(
        measure,
        kind="stop",
        number=1,
        placement="below",
        staff=1,
        offset_ratio=0.75,
    )

    directions = measure.findall("direction")
    assert [
        direction.find("direction-type/wedge").get("type")  # type: ignore[union-attr]
        for direction in directions
    ] == ["crescendo", "stop"]
    assert [direction.findtext("offset") for direction in directions] == ["4", "12"]


def test_clean_medium_confidence_hairpin_is_inserted(tmp_path: Path) -> None:
    xml_path = tmp_path / "score.musicxml"
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"unused")
    _write_score(xml_path)

    report = enrich_musicxml_with_wedges(
        image_path,
        xml_path,
        PageLayout(800, 260, [_staff(1, 120)], 1.0),
        candidates=(_medium_candidate(),),
    )

    assert report.transaction_committed
    assert report.injected_count == 1
    assert report.proposals[0].injected


def test_medium_hairpin_overlapping_clean_curve_is_rejected(tmp_path: Path) -> None:
    xml_path = tmp_path / "score.musicxml"
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"unused")
    _write_score(xml_path)
    candidate = _medium_candidate()
    curve = VisualNotationCandidate(
        "curved_connector",
        1,
        "below",
        (295, 172, 370, 188),
        0.90,
        (
            ("fit_p90_spaces", 0.04),
            ("length_spaces", 6.0),
        ),
    )

    report = enrich_musicxml_with_wedges(
        image_path,
        xml_path,
        PageLayout(800, 260, [_staff(1, 120)], 1.0),
        candidates=(candidate, curve),
    )

    assert not report.transaction_committed
    assert report.injected_count == 0
    assert "curved connector" in report.proposals[0].reason


def test_medium_hairpin_with_blunt_apex_is_rejected(tmp_path: Path) -> None:
    xml_path = tmp_path / "score.musicxml"
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"unused")
    _write_score(xml_path)

    report = enrich_musicxml_with_wedges(
        image_path,
        xml_path,
        PageLayout(800, 260, [_staff(1, 120)], 1.0),
        candidates=(_medium_candidate(apex_separation=0.60),),
    )

    assert not report.transaction_committed
    assert report.injected_count == 0
    assert "wedge-specific" in report.proposals[0].reason


def test_existing_wedges_force_matching_abstention(tmp_path: Path) -> None:
    xml_path = tmp_path / "score.musicxml"
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"unused")
    _write_score(xml_path, existing_wedge=True)
    original = xml_path.read_bytes()

    report = enrich_musicxml_with_wedges(
        image_path,
        xml_path,
        PageLayout(800, 260, [_staff(1, 120)], 1.0),
        candidates=(_candidate(),),
    )

    assert not report.transaction_committed
    assert report.existing_wedge_count == 2
    assert report.injected_count == 0
    assert "matching" in report.proposals[0].reason
    assert xml_path.read_bytes() == original


def test_full_score_staff_slot_maps_to_independent_part_timeline(tmp_path: Path) -> None:
    xml_path = tmp_path / "score.musicxml"
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"unused")
    _write_score(xml_path, part_staff_counts=(2, 1), system_breaks=(2,))
    staffs = [
        _staff(1, 100),
        _staff(2, 190),
        _staff(3, 280),
        _staff(4, 430),
        _staff(5, 520),
        _staff(6, 610),
    ]
    for staff in staffs:
        staff.barlines = [400, 760]
        staff.measure_count = 2
    layout = PageLayout(800, 760, staffs, 1.0)

    report = enrich_musicxml_with_wedges(
        image_path,
        xml_path,
        layout,
        candidates=(_candidate(staff_index=3),),
    )

    assert report.transaction_committed
    proposal = report.proposals[0]
    assert proposal.start is not None and proposal.end is not None
    assert proposal.start.part_id == "P2"
    assert proposal.start.staff == 1
    tree = etree.parse(str(xml_path))
    assert not tree.findall("./part[@id='P1']/measure/direction/direction-type/wedge")
    assert len(tree.findall("./part[@id='P2']/measure/direction/direction-type/wedge")) == 2


def test_incomplete_repeated_staff_topology_fails_closed(tmp_path: Path) -> None:
    xml_path = tmp_path / "score.musicxml"
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"unused")
    _write_score(xml_path, part_staff_counts=(2, 1))
    layout = PageLayout(
        800,
        700,
        [_staff(index, 100 + 100 * index) for index in range(1, 6)],
        1.0,
    )
    original = xml_path.read_bytes()

    report = enrich_musicxml_with_wedges(
        image_path,
        xml_path,
        layout,
        candidates=(_candidate(staff_index=3),),
    )

    assert not report.transaction_committed
    assert report.injected_count == 0
    assert report.error is not None
    assert "incomplete" in report.error
    assert xml_path.read_bytes() == original


def test_recognized_topology_removes_one_isolated_false_staff_comb() -> None:
    ordered = tuple(
        _staff(index, top_line)
        for index, top_line in enumerate(
            (100, 190, 400, 490, 700, 790, 1000, 1090, 1260),
            start=1,
        )
    )
    ordered[-1].barlines = []
    ordered[-1].measure_count = 1

    conditioned = _conditioned_staff_appearances(
        ordered,
        physical_staff_count=2,
        score_system_count=4,
    )

    assert [staff.index for staff in conditioned] == list(range(1, 9))


def test_recognized_topology_keeps_ambiguous_staff_subset_unchanged() -> None:
    ordered = tuple(
        _staff(index, top_line)
        for index, top_line in enumerate((100, 200, 300, 400, 500), start=1)
    )

    conditioned = _conditioned_staff_appearances(
        ordered,
        physical_staff_count=1,
        score_system_count=4,
    )

    assert conditioned == ordered


def test_recognized_repeated_piano_topology_fills_decisive_damaged_staves() -> None:
    ordered = tuple(
        _staff(index, top_line)
        for index, top_line in enumerate(
            (621, 1333, 1596, 2095, 2808),
            start=1,
        )
    )
    for staff in ordered:
        staff.spacing = 25

    conditioned = _conditioned_staff_appearances(
        ordered,
        physical_staff_count=2,
        score_system_count=4,
    )

    assert len(conditioned) == 8
    centres = [round(sum(staff.line_y) / len(staff.line_y)) for staff in conditioned]
    assert centres == [641, 904, 1353, 1616, 2115, 2378, 2828, 3091]


def test_conditioned_topology_reassigns_geometry_to_a_synthetic_lower_staff(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "piano.musicxml"
    _write_score(
        xml_path,
        part_staff_counts=(2,),
        measures=8,
        system_breaks=(2, 4, 6),
    )
    raw = tuple(
        _staff(index, top_line)
        for index, top_line in enumerate(
            (621, 1333, 1596, 2095, 2808),
            start=1,
        )
    )
    for staff in raw:
        staff.spacing = 25
    topology, error = _build_topology(
        etree.parse(str(xml_path)).getroot(),
        PageLayout(1600, 3400, list(raw), 0.98),
    )

    assert error is None
    assert topology is not None
    assigned = _assign_bbox_to_appearance(
        topology,
        (500, 2335, 800, 2370),
    )
    assert assigned is not None
    location = _appearance_location(topology, assigned[0])
    assert location is not None
    assert location[0] == 2
    assert location[2] == 2


def test_conditioned_wedge_uses_synthetic_staff_placement_in_musicxml(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "piano.musicxml"
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"unused")
    _write_score(
        xml_path,
        part_staff_counts=(2,),
        measures=8,
        system_breaks=(2, 4, 6),
    )
    raw = [
        _staff(index, top_line)
        for index, top_line in enumerate(
            (621, 1333, 1596, 2095, 2808),
            start=1,
        )
    ]
    for staff in raw:
        staff.spacing = 25
        staff.barlines = [40, 400, 760]
        staff.measure_count = 2
    layout = PageLayout(1600, 3400, raw, 0.98)
    candidate = VisualNotationCandidate(
        "crescendo",
        4,
        "above",
        (500, 2390, 700, 2420),
        0.99,
        (
            ("length_spaces", 8.0),
            ("open_separation_spaces", 1.4),
            ("apex_separation_spaces", 0.1),
            ("ink_density", 0.12),
        ),
    )

    report = enrich_musicxml_with_wedges(
        image_path,
        xml_path,
        layout,
        candidates=(candidate,),
    )

    assert report.transaction_committed
    assert report.injected_count == 1
    assert report.proposals[0].candidate.staff_index != candidate.staff_index
    assert report.proposals[0].candidate.placement == "below"
    assert report.proposals[0].start is not None
    assert report.proposals[0].start.staff == 2
    directions = etree.parse(str(xml_path)).findall(
        "./part/measure/direction"
    )
    assert [item.get("placement") for item in directions] == [
        "below",
        "below",
    ]
    assert [item.findtext("staff") for item in directions] == ["2", "2"]


def test_hidden_empty_ossia_is_visible_only_in_its_content_system(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "ossia.musicxml"
    _write_score(
        xml_path,
        part_staff_counts=(2, 1),
        measures=8,
        system_breaks=(2, 4, 6),
    )
    tree = etree.parse(str(xml_path))
    tree.find("./part-list/score-part[@id='P2']/part-name").set(  # type: ignore[union-attr]
        "print-object",
        "no",
    )
    visible_note = tree.findall("./part[@id='P2']/measure")[4].find("note")
    visible_note.remove(visible_note.find("rest"))  # type: ignore[arg-type,union-attr]
    pitch = etree.Element("pitch")
    etree.SubElement(pitch, "step").text = "C"
    etree.SubElement(pitch, "octave").text = "4"
    visible_note.insert(0, pitch)  # type: ignore[union-attr]
    tree.write(
        str(xml_path),
        encoding="UTF-8",
        xml_declaration=True,
        doctype=MUSICXML_DOCTYPE,
    )
    layout = PageLayout(
        800,
        1300,
        [
            _staff(index, top_line)
            for index, top_line in enumerate(
                (100, 190, 400, 490, 700, 790, 1000, 1090),
                start=1,
            )
        ],
        1.0,
    )

    topology, error = _build_topology(tree.getroot(), layout)

    assert error is None
    assert topology is not None
    assert [len(group) for group in topology.score_systems] == [2, 2, 3, 2]
    assert any(
        group_index == 2 and part_index == 1 and part_staff == 1
        for _staff_index, group_index, part_index, part_staff in (
            topology.appearance_locations
        )
    )


def test_grand_staff_system_break_continuation_is_one_upper_staff_wedge(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "piano.musicxml"
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"unused")
    _write_score(
        xml_path,
        part_staff_counts=(2,),
        measures=4,
        system_breaks=(2,),
    )
    staffs = [
        _staff(1, 100),
        _staff(2, 190),
        _staff(3, 430),
        _staff(4, 520),
    ]
    for staff in staffs:
        staff.barlines = [40, 400, 760]
        staff.measure_count = 2
    layout = PageLayout(800, 680, staffs, 1.0)
    geometry = (
        ("length_spaces", 8.0),
        ("open_separation_spaces", 1.4),
        ("apex_separation_spaces", 0.1),
        ("ink_density", 0.12),
    )
    system_end = VisualNotationCandidate(
        "crescendo",
        2,
        "above",
        (680, 158, 755, 182),
        0.99,
        geometry,
    )
    next_system = VisualNotationCandidate(
        "crescendo",
        4,
        "above",
        (50, 488, 250, 512),
        0.99,
        geometry,
    )

    report = enrich_musicxml_with_wedges(
        image_path,
        xml_path,
        layout,
        candidates=(system_end, next_system),
    )

    assert report.transaction_committed
    assert report.injected_count == 1
    proposal = report.proposals[0]
    assert proposal.start is not None and proposal.end is not None
    assert proposal.start.staff == 1
    assert proposal.end.staff == 1
    assert proposal.start.measure_index == 1
    assert proposal.end.measure_index == 2
    assert len(
        etree.parse(str(xml_path)).findall(
            "./part/measure/direction/direction-type/wedge"
        )
    ) == 2
