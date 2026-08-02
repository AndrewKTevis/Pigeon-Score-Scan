from pathlib import Path
import zipfile

from lxml import etree

from scorescan.layout import PageLayout, ScoreSystemLayout, StaffSystem
from scorescan.musicxml import (
    PageDocument,
    analyze_musicxml,
    canonicalize_multivoice_timelines,
    merge_pages,
    package_mxl,
    parse_or_placeholder,
    validate_musicxml,
)


def page_tree(measures: int) -> etree._ElementTree:
    tree = parse_or_placeholder(None, 1, "test")
    part = tree.getroot().find("part")
    template = part.find("measure")
    for i in range(1, measures):
        part.append(etree.fromstring(etree.tostring(template)))
    return tree


def multi_part_page_tree(page: int) -> etree._ElementTree:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    group_start = etree.SubElement(part_list, "part-group", number="1", type="start")
    etree.SubElement(group_start, "group-symbol").text = "bracket"
    for part_id, name, step in (("P1", "Piano", "C"), ("P2", "Violin", "G")):
        score_part = etree.SubElement(part_list, "score-part", id=part_id)
        etree.SubElement(score_part, "part-name").text = name
        part = etree.SubElement(root, "part", id=part_id)
        measure = etree.SubElement(part, "measure", number=str(page))
        attributes = etree.SubElement(measure, "attributes")
        etree.SubElement(attributes, "divisions").text = "1"
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = str(3 + page)
        etree.SubElement(note, "duration").text = "4"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "whole"
    etree.SubElement(part_list, "part-group", number="1", type="stop")
    return etree.ElementTree(root)


def test_placeholder_merge_page_and_system_breaks(tmp_path: Path) -> None:
    layout = PageLayout(1000, 1400, [
        StaffSystem(1, [100,110,120,130,140], 70,170,50,950,10,[],2),
        StaffSystem(2, [300,310,320,330,340], 270,370,50,950,10,[],2),
    ], 1.0)
    output = tmp_path / "out.musicxml"
    summary = merge_pages([
        PageDocument(page_tree(4), 1000, 1400, layout),
        PageDocument(page_tree(4), 1000, 1400, layout),
    ], output)
    assert summary["pages"] == 2
    assert summary["measures"] == 8
    assert validate_musicxml(output) == []
    analysis = analyze_musicxml(output)
    assert analysis["page_count"] == 2
    assert analysis["system_breaks"] == 2
    tree = etree.parse(str(output))
    assert (
        tree.findtext("./part/measure/print/system-layout/system-margins/right-margin")
        == "60"
    )


def test_merge_normalizes_compact_defaults_and_right_page_margin(
    tmp_path: Path,
) -> None:
    tree = page_tree(1)
    root = tree.getroot()
    defaults = etree.Element("defaults")
    page_layout = etree.SubElement(defaults, "page-layout")
    etree.SubElement(page_layout, "page-height").text = "300"
    etree.SubElement(page_layout, "page-width").text = "110"
    scaling = etree.SubElement(defaults, "scaling")
    etree.SubElement(scaling, "millimeters").text = "7.0"
    etree.SubElement(scaling, "tenths").text = "40"
    root.insert(root.index(root.find("part-list")), defaults)
    output = tmp_path / "compact-defaults.musicxml"

    merge_pages([PageDocument(tree, 1000, 1400)], output)

    merged = etree.parse(str(output))
    assert merged.findtext("./defaults/page-layout/page-width") == "1680.0"
    assert merged.findtext("./defaults/page-layout/page-height") == "2352.0"
    assert merged.findtext("./defaults/page-layout/page-margins/right-margin") == "85"
    normalized_defaults = merged.find("./defaults")
    assert [child.tag for child in normalized_defaults][:2] == [
        "scaling",
        "page-layout",
    ]


def test_merge_removes_tempo_word_duplicated_as_work_title(tmp_path: Path) -> None:
    tree = page_tree(1)
    root = tree.getroot()
    title = root.find("./work/work-title")
    assert title is not None
    title.text = "Allegretto"
    measure = root.find("./part/measure")
    assert measure is not None
    direction = etree.Element("direction", placement="above")
    direction_type = etree.SubElement(direction, "direction-type")
    etree.SubElement(direction_type, "words").text = "Allegretto."
    measure.insert(1, direction)
    output = tmp_path / "tempo-not-title.musicxml"

    summary = merge_pages([PageDocument(tree, 1000, 1400)], output)

    merged = etree.parse(str(output))
    assert merged.find("./work/work-title") is None
    assert merged.findtext("./part/measure/direction/direction-type/words") == "Allegretto."
    assert summary["sanitized_duplicate_headers"] == 1


def test_merge_strips_out_of_scope_note_lyrics_from_public_output(
    tmp_path: Path,
) -> None:
    tree = page_tree(1)
    note = tree.find("./part/measure/note")
    assert note is not None
    lyric = etree.SubElement(note, "lyric")
    etree.SubElement(lyric, "syllabic").text = "single"
    etree.SubElement(lyric, "text").text = "la"
    output = tmp_path / "no-semantic-lyrics.musicxml"

    summary = merge_pages([PageDocument(tree, 1000, 1400)], output)

    merged = etree.parse(str(output))
    assert merged.find(".//note/lyric") is None
    assert summary["stripped_out_of_scope_lyrics"] == 1
    assert merged.find(".//note/rest") is not None
    assert merged.findtext(".//note/type") == "whole"


def test_existing_single_staff_breaks_override_false_ensemble_grouping(
    tmp_path: Path,
) -> None:
    tree = page_tree(6)
    measures = tree.getroot().find("part").findall("measure")
    for measure in measures[1:]:
        etree.SubElement(measure, "print", **{"new-system": "yes"})
    physical = [
        StaffSystem(
            index=index + 1,
            line_y=[100 + index * 150 + offset * 10 for offset in range(5)],
            top=70 + index * 150,
            bottom=170 + index * 150,
            left=50,
            right=950,
            spacing=10,
            barlines=[],
            measure_count=1,
        )
        for index in range(6)
    ]
    false_ensemble = ScoreSystemLayout(
        index=1,
        staff_indices=[1, 2, 3, 4, 5, 6],
        top=70,
        bottom=920,
        left=50,
        right=950,
        spacing=10,
        barlines=[],
        measure_count=1,
        grouping_confidence=0.72,
        grouping_method="absolute_vertical_gap",
    )
    layout = PageLayout(
        1000,
        1000,
        physical,
        0.98,
        score_systems=[false_ensemble],
    )
    output = tmp_path / "single-staff-systems.musicxml"

    summary = merge_pages([PageDocument(tree, 1000, 1000, layout)], output)

    merged = etree.parse(str(output))
    right_margins = merged.xpath(
        "./part/measure/print/system-layout/system-margins/right-margin/text()",
    )
    assert summary["page_summaries"][0]["source_systems"] == 1
    assert right_margins == ["60"] * 6
    assert analyze_musicxml(output)["system_breaks"] == 5
    assert sum(float(measure.get("width")) for measure in merged.findall("./part/measure")[:1]) == 1450.0


def test_merge_pages_preserves_all_parts_and_independent_timelines(tmp_path: Path) -> None:
    output = tmp_path / "ensemble.musicxml"

    summary = merge_pages(
        [
            PageDocument(multi_part_page_tree(1), 1000, 1400),
            PageDocument(multi_part_page_tree(2), 1000, 1400),
        ],
        output,
    )

    root = etree.parse(str(output)).getroot()
    parts = root.findall("part")
    assert summary["parts"] == 2
    assert summary["measures"] == 2
    assert [part.get("id") for part in parts] == ["P1", "P2"]
    assert [len(part.findall("measure")) for part in parts] == [2, 2]
    assert [part.findtext("./measure[2]/note/pitch/step") for part in parts] == ["C", "G"]
    assert all(part.find("./measure[2]/print").get("new-page") == "yes" for part in parts)
    assert len(root.findall("./part-list/part-group")) == 2
    assert validate_musicxml(output) == []
    analysis = analyze_musicxml(output)
    assert analysis["part_count"] == 2
    assert analysis["part_measure_counts"] == [2, 2]
    assert analysis["note_count"] == 4
    assert analysis["page_count"] == 2
    assert analysis["rhythm_issues"] == []


def test_merge_pages_refuses_to_guess_when_part_topology_changes(tmp_path: Path) -> None:
    single_part = page_tree(1)

    try:
        merge_pages(
            [
                PageDocument(multi_part_page_tree(1), 1000, 1400),
                PageDocument(single_part, 1000, 1400),
            ],
            tmp_path / "invalid.musicxml",
        )
    except ValueError as exc:
        assert "为避免错配" in str(exc)
    else:
        raise AssertionError("part topology mismatch must not be merged silently")


def test_mxl_package(tmp_path: Path) -> None:
    xml = tmp_path / "score.musicxml"
    merge_pages([PageDocument(page_tree(1), 1000, 1400)], xml)
    mxl = tmp_path / "score.mxl"
    package_mxl(xml, mxl)
    with zipfile.ZipFile(mxl) as archive:
        assert archive.read("mimetype") == b"application/vnd.recordare.musicxml"
        assert "META-INF/container.xml" in archive.namelist()
        assert "score.musicxml" in archive.namelist()


def test_normalize_single_voice_fills_unambiguous_representation(tmp_path: Path) -> None:
    from scorescan.musicxml import MUSICXML_DOCTYPE, normalize_single_voice_musicxml
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "4"
    note = etree.SubElement(measure, "note")
    pitch = etree.SubElement(note, "pitch")
    etree.SubElement(pitch, "step").text = "C"
    etree.SubElement(pitch, "octave").text = "4"
    etree.SubElement(note, "duration").text = "6"
    path = tmp_path / "normalize.musicxml"
    etree.ElementTree(root).write(str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE)
    summary = normalize_single_voice_musicxml(path)
    tree = etree.parse(str(path))
    normalized_note = tree.getroot().find("./part/measure/note")
    assert normalized_note.findtext("voice") == "1"
    assert normalized_note.findtext("type") == "quarter"
    assert len(normalized_note.findall("dot")) == 1
    assert summary["added_voices"] == 1


def test_anacrusis_and_complementary_final_measure_are_not_rhythm_errors(tmp_path: Path) -> None:
    from lxml import etree
    from scorescan.musicxml import analyze_musicxml

    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    durations = (1, 4, 3)
    for index, duration in enumerate(durations, start=1):
        measure = etree.SubElement(part, "measure", number=str(index))
        if index == 1:
            attributes = etree.SubElement(measure, "attributes")
            etree.SubElement(attributes, "divisions").text = "1"
            time = etree.SubElement(attributes, "time")
            etree.SubElement(time, "beats").text = "4"
            etree.SubElement(time, "beat-type").text = "4"
        note = etree.SubElement(measure, "note")
        etree.SubElement(note, "rest")
        etree.SubElement(note, "duration").text = str(duration)
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter" if duration == 1 else "whole"
    path = tmp_path / "pickup.musicxml"
    path.write_bytes(etree.tostring(root, xml_declaration=True, encoding="UTF-8"))
    report = analyze_musicxml(path)
    assert report["rhythm_issues"] == []
    assert report["legal_incomplete_measures"] == [
        {"measure_index": 1, "voice": "1"},
        {"measure_index": 3, "voice": "1"},
    ]


def test_multivoice_timeline_canonicalizer_repairs_interleaved_grand_staff_cursor(
    tmp_path: Path,
) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Piano"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "4"
    etree.SubElement(attributes, "staves").text = "2"
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"

    def note(duration: int, voice: str, staff: str) -> None:
        node = etree.SubElement(measure, "note")
        etree.SubElement(node, "rest")
        etree.SubElement(node, "duration").text = str(duration)
        etree.SubElement(node, "voice").text = voice
        etree.SubElement(node, "staff").text = staff

    def backup(duration: int) -> None:
        node = etree.SubElement(measure, "backup")
        etree.SubElement(node, "duration").text = str(duration)

    note(6, "1", "1")
    backup(6)
    note(4, "5", "2")
    note(4, "5", "2")
    # Missing backup here is the homr grand-staff serialization defect.
    note(2, "1", "1")
    note(4, "1", "1")
    backup(4)
    note(4, "5", "2")
    note(4, "1", "1")
    backup(4)
    note(4, "5", "2")

    path = tmp_path / "interleaved.musicxml"
    path.write_bytes(etree.tostring(root, xml_declaration=True, encoding="UTF-8"))
    assert analyze_musicxml(path)["rhythm_issues"][0]["actual"] == 18

    repair = canonicalize_multivoice_timelines(path)

    assert repair["repaired_count"] == 1
    assert analyze_musicxml(path)["rhythm_issues"] == []
    repaired = etree.parse(str(path)).find("./part/measure")
    assert repaired is not None
    assert [node.findtext("duration") for node in repaired.findall("backup")] == ["16"]


def test_multivoice_timeline_canonicalizer_abstains_with_timed_direction(
    tmp_path: Path,
) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Piano"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    direction = etree.SubElement(measure, "direction")
    etree.SubElement(etree.SubElement(direction, "direction-type"), "words").text = "dolce"
    for voice, duration in (("1", 4), ("2", 5)):
        if voice == "2":
            back = etree.SubElement(measure, "backup")
            etree.SubElement(back, "duration").text = "4"
        note = etree.SubElement(measure, "note")
        etree.SubElement(note, "rest")
        etree.SubElement(note, "duration").text = str(duration)
        etree.SubElement(note, "voice").text = voice
    path = tmp_path / "ambiguous-direction.musicxml"
    path.write_bytes(etree.tostring(root, xml_declaration=True, encoding="UTF-8"))

    report = canonicalize_multivoice_timelines(path)

    assert report["repaired_count"] == 0
    assert report["abstained_count"] == 1


def test_small_opening_pickup_does_not_require_complementary_final_bar(
    tmp_path: Path,
) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    for number, duration in enumerate((3, 9, 9, 9, 9), start=1):
        measure = etree.SubElement(part, "measure", number=str(number))
        if number == 1:
            attributes = etree.SubElement(measure, "attributes")
            etree.SubElement(attributes, "divisions").text = "2"
            time = etree.SubElement(attributes, "time")
            etree.SubElement(time, "beats").text = "9"
            etree.SubElement(time, "beat-type").text = "8"
        note = etree.SubElement(measure, "note")
        etree.SubElement(note, "rest")
        etree.SubElement(note, "duration").text = str(duration)
        etree.SubElement(note, "voice").text = "1"
    path = tmp_path / "pickup-full-ending.musicxml"
    path.write_bytes(etree.tostring(root, xml_declaration=True, encoding="UTF-8"))

    report = analyze_musicxml(path)

    assert report["rhythm_issues"] == []
    assert report["legal_incomplete_measures"] == [
        {"measure_index": 1, "voice": "1"},
    ]


def test_large_or_unstable_opening_shortfall_remains_a_rhythm_issue(
    tmp_path: Path,
) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    for number, duration in enumerate((7, 9, 8, 9, 9), start=1):
        measure = etree.SubElement(part, "measure", number=str(number))
        if number == 1:
            attributes = etree.SubElement(measure, "attributes")
            etree.SubElement(attributes, "divisions").text = "2"
            time = etree.SubElement(attributes, "time")
            etree.SubElement(time, "beats").text = "9"
            etree.SubElement(time, "beat-type").text = "8"
        note = etree.SubElement(measure, "note")
        etree.SubElement(note, "rest")
        etree.SubElement(note, "duration").text = str(duration)
        etree.SubElement(note, "voice").text = "1"
    path = tmp_path / "unstable-shortfall.musicxml"
    path.write_bytes(etree.tostring(root, xml_declaration=True, encoding="UTF-8"))

    report = analyze_musicxml(path)

    assert {item["measure"] for item in report["rhythm_issues"]} == {"1", "3"}
    assert report["legal_incomplete_measures"] == []


def test_analysis_reads_time_signature_from_second_attributes_block(tmp_path: Path) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    divisions_attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(divisions_attributes, "divisions").text = "4"
    musical_attributes = etree.SubElement(measure, "attributes")
    time = etree.SubElement(musical_attributes, "time")
    etree.SubElement(time, "beats").text = "3"
    etree.SubElement(time, "beat-type").text = "4"
    note = etree.SubElement(measure, "note")
    etree.SubElement(note, "rest")
    etree.SubElement(note, "duration").text = "12"
    etree.SubElement(note, "voice").text = "1"
    etree.SubElement(note, "type").text = "half"
    etree.SubElement(note, "dot")
    path = tmp_path / "split-attributes.musicxml"
    path.write_bytes(etree.tostring(root, xml_declaration=True, encoding="UTF-8"))

    report = analyze_musicxml(path)

    assert report["rhythm_issues"] == []
