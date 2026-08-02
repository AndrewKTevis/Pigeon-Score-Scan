from __future__ import annotations

from pathlib import Path

from lxml import etree

from scorescan.layout import PageLayout, StaffSystem
from scorescan.musicxml import MUSICXML_DOCTYPE, analyze_musicxml
from scorescan.notation_coverage import VisualNotationCandidate
from scorescan.slur_relation_repair import repair_source_proven_nested_slurs


def _layout() -> PageLayout:
    staff = StaffSystem(
        1,
        [120, 130, 140, 150, 160],
        70,
        220,
        40,
        760,
        10,
        [760],
        1,
    )
    return PageLayout(800, 260, [staff], 1.0)


def _write_orphan_pattern(path: Path) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"
    for index, (step, duration) in enumerate((("C", 1), ("D", 1), ("E", 2))):
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = str(duration)
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter" if duration == 1 else "half"
        etree.SubElement(note, "staff").text = "1"
        notations = etree.SubElement(note, "notations")
        etree.SubElement(
            notations,
            "slur",
            type="start" if index == 0 else "stop",
            number="1",
        )
    etree.ElementTree(root).write(
        str(path),
        encoding="UTF-8",
        xml_declaration=True,
        doctype=MUSICXML_DOCTYPE,
    )


def _nested_source_pair(
    *,
    weaker_confidence: float = 0.91,
) -> tuple[VisualNotationCandidate, ...]:
    return (
        VisualNotationCandidate(
            "curved_connector",
            1,
            "above",
            (180, 85, 420, 115),
            0.98,
        ),
        VisualNotationCandidate(
            "curved_connector",
            1,
            "above",
            (190, 92, 350, 112),
            weaker_confidence,
        ),
    )


def _write_missing_outer_chain_pattern(path: Path) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"
    slurs = (
        (("start", "1"),),
        (("stop", "1"), ("start", "1")),
        (("stop", "1"),),
        (("stop", "1"),),
    )
    for index, step in enumerate(("C", "D", "E", "F")):
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "1"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
        etree.SubElement(note, "staff").text = "1"
        notations = etree.SubElement(note, "notations")
        for slur_type, number in slurs[index]:
            etree.SubElement(
                notations,
                "slur",
                type=slur_type,
                number=number,
            )
    etree.ElementTree(root).write(
        str(path),
        encoding="UTF-8",
        xml_declaration=True,
        doctype=MUSICXML_DOCTYPE,
    )


def _outer_chain_source_pair(
    *,
    long_confidence: float = 0.98,
    short_fit: float = 0.12,
) -> tuple[VisualNotationCandidate, ...]:
    return (
        VisualNotationCandidate(
            "curved_connector",
            1,
            "above",
            (180, 85, 500, 115),
            long_confidence,
            (
                ("fit_p90_spaces", 0.05),
                ("length_spaces", 11.3),
            ),
        ),
        VisualNotationCandidate(
            "curved_connector",
            1,
            "above",
            (280, 92, 420, 112),
            0.80,
            (
                ("fit_p90_spaces", short_fit),
                ("length_spaces", 4.8),
            ),
        ),
    )


def test_source_proven_nested_arc_repairs_only_slur_numbering(tmp_path: Path) -> None:
    xml_path = tmp_path / "score.musicxml"
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"unused")
    _write_orphan_pattern(xml_path)
    before = analyze_musicxml(xml_path)
    assert len(before["slur_issues"]) == 1

    report = repair_source_proven_nested_slurs(
        image_path,
        xml_path,
        _layout(),
        candidates=_nested_source_pair(),
    )

    assert report.transaction_committed
    assert report.repaired_count == 1
    assert report.slur_issue_count_before == 1
    assert report.slur_issue_count_after == 0
    tree = etree.parse(str(xml_path))
    notes = tree.findall("./part/measure/note")
    assert [
        (item.get("type"), item.get("number"))
        for item in notes[0].findall("./notations/slur")
    ] == [("start", "1"), ("start", "2")]
    assert [
        (item.get("type"), item.get("number"))
        for item in notes[1].findall("./notations/slur")
    ] == [("stop", "1")]
    assert [
        (item.get("type"), item.get("number"))
        for item in notes[2].findall("./notations/slur")
    ] == [("stop", "2")]
    after = analyze_musicxml(xml_path)
    for key in ("note_count", "rest_count", "rhythm_issues", "tie_issues"):
        assert after[key] == before[key]


def test_one_source_arc_is_not_enough_to_balance_an_orphan_stop(tmp_path: Path) -> None:
    xml_path = tmp_path / "score.musicxml"
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"unused")
    _write_orphan_pattern(xml_path)
    original = xml_path.read_bytes()

    report = repair_source_proven_nested_slurs(
        image_path,
        xml_path,
        _layout(),
        candidates=(_nested_source_pair()[0],),
    )

    assert not report.transaction_committed
    assert report.repaired_count == 0
    assert report.abstention_count == 1
    assert "nested" in report.proposals[0].reason
    assert xml_path.read_bytes() == original


def test_weak_second_curve_cannot_trigger_automatic_repair(tmp_path: Path) -> None:
    xml_path = tmp_path / "score.musicxml"
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"unused")
    _write_orphan_pattern(xml_path)

    report = repair_source_proven_nested_slurs(
        image_path,
        xml_path,
        _layout(),
        candidates=_nested_source_pair(weaker_confidence=0.87),
    )

    assert not report.transaction_committed
    assert report.repaired_count == 0
    assert report.slur_issue_count_after == 1


def test_geometrically_clean_contained_pair_can_use_narrow_secondary_gate(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "score.musicxml"
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"unused")
    _write_orphan_pattern(xml_path)
    candidates = (
        VisualNotationCandidate(
            "curved_connector",
            1,
            "above",
            (180, 85, 420, 115),
            0.93,
            (("fit_p90_spaces", 0.04),),
        ),
        VisualNotationCandidate(
            "curved_connector",
            1,
            "above",
            (190, 92, 350, 112),
            0.86,
            (("fit_p90_spaces", 0.03),),
        ),
    )

    report = repair_source_proven_nested_slurs(
        image_path,
        xml_path,
        _layout(),
        candidates=candidates,
    )

    assert report.transaction_committed
    assert report.repaired_count == 1
    assert report.slur_issue_count_after == 0


def test_secondary_gate_rejects_a_poor_curve_fit(tmp_path: Path) -> None:
    xml_path = tmp_path / "score.musicxml"
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"unused")
    _write_orphan_pattern(xml_path)
    candidates = (
        VisualNotationCandidate(
            "curved_connector",
            1,
            "above",
            (180, 85, 420, 115),
            0.93,
            (("fit_p90_spaces", 0.04),),
        ),
        VisualNotationCandidate(
            "curved_connector",
            1,
            "above",
            (190, 92, 350, 112),
            0.86,
            (("fit_p90_spaces", 0.20),),
        ),
    )

    report = repair_source_proven_nested_slurs(
        image_path,
        xml_path,
        _layout(),
        candidates=candidates,
    )

    assert not report.transaction_committed
    assert report.repaired_count == 0


def test_one_exact_spanning_arc_removes_only_the_premature_stop(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "score.musicxml"
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"unused")
    _write_orphan_pattern(xml_path)
    spanning = VisualNotationCandidate(
        "curved_connector",
        1,
        "above",
        (60, 85, 340, 115),
        0.95,
        (
            ("fit_p90_spaces", 0.05),
            ("length_spaces", 20.0),
            ("sagitta_spaces", 1.0),
        ),
    )

    report = repair_source_proven_nested_slurs(
        image_path,
        xml_path,
        _layout(),
        candidates=(spanning,),
    )

    assert report.transaction_committed
    assert report.repaired_count == 1
    assert report.proposals[0].operation == "extend_single_arc"
    assert report.proposals[0].assigned_number is None
    notes = etree.parse(str(xml_path)).findall("./part/measure/note")
    assert [
        (node.get("type"), node.get("number"))
        for node in notes[0].findall("./notations/slur")
    ] == [("start", "1")]
    assert notes[1].findall("./notations/slur") == []
    assert [
        (node.get("type"), node.get("number"))
        for node in notes[2].findall("./notations/slur")
    ] == [("stop", "1")]


def test_short_or_off_center_single_arc_cannot_rewire_endpoints(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "score.musicxml"
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"unused")
    _write_orphan_pattern(xml_path)
    original = xml_path.read_bytes()
    short = VisualNotationCandidate(
        "curved_connector",
        1,
        "above",
        (240, 85, 310, 115),
        0.99,
        (
            ("fit_p90_spaces", 0.02),
            ("length_spaces", 6.0),
            ("sagitta_spaces", 1.0),
        ),
    )

    report = repair_source_proven_nested_slurs(
        image_path,
        xml_path,
        _layout(),
        candidates=(short,),
    )

    assert not report.transaction_committed
    assert report.repaired_count == 0
    assert xml_path.read_bytes() == original


def test_clean_outer_arc_repairs_missing_start_across_two_short_arcs(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "score.musicxml"
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"unused")
    _write_missing_outer_chain_pattern(xml_path)
    assert len(analyze_musicxml(xml_path)["slur_issues"]) == 1

    report = repair_source_proven_nested_slurs(
        image_path,
        xml_path,
        _layout(),
        candidates=_outer_chain_source_pair(),
    )

    assert report.transaction_committed
    assert report.repaired_count == 1
    assert report.slur_issue_count_after == 0
    assert report.proposals[0].operation == "add_outer_arc"
    assert report.proposals[0].assigned_number == 2
    notes = etree.parse(str(xml_path)).findall("./part/measure/note")
    assert [
        (node.get("type"), node.get("number"))
        for node in notes[0].findall("./notations/slur")
    ] == [("start", "1"), ("start", "2")]
    assert [
        (node.get("type"), node.get("number"))
        for node in notes[1].findall("./notations/slur")
    ] == [("stop", "1"), ("start", "1")]
    assert [
        (node.get("type"), node.get("number"))
        for node in notes[2].findall("./notations/slur")
    ] == [("stop", "1")]
    assert [
        (node.get("type"), node.get("number"))
        for node in notes[3].findall("./notations/slur")
    ] == [("stop", "2")]


def test_outer_arc_repair_rejects_weak_long_or_poor_short_curve(
    tmp_path: Path,
) -> None:
    for name, candidates in (
        ("weak-long", _outer_chain_source_pair(long_confidence=0.96)),
        ("poor-short", _outer_chain_source_pair(short_fit=0.20)),
    ):
        xml_path = tmp_path / f"{name}.musicxml"
        image_path = tmp_path / f"{name}.png"
        image_path.write_bytes(b"unused")
        _write_missing_outer_chain_pattern(xml_path)
        original = xml_path.read_bytes()

        report = repair_source_proven_nested_slurs(
            image_path,
            xml_path,
            _layout(),
            candidates=candidates,
        )

        assert not report.transaction_committed
        assert report.repaired_count == 0
        assert report.slur_issue_count_after == 1
        assert xml_path.read_bytes() == original
