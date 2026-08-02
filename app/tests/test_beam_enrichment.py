from __future__ import annotations

from pathlib import Path

from lxml import etree

from scorescan.beam_enrichment import (
    _Event,
    _Segment,
    _assign_group,
    _assign_segment,
    _consensus_measure_boundaries,
    _events_for_staff,
    _eligible_runs,
    _merge_repeated_high_level_path_splits,
    _normalize_parallel_beam_staff_assignments,
    _write_segment,
    enrich_musicxml_with_source_beams,
)
from scorescan.layout import PageLayout, StaffSystem
from scorescan.musicxml import MUSICXML_DOCTYPE, analyze_musicxml
from scorescan.semantic_detector import SemanticDetection


def _staff() -> StaffSystem:
    return StaffSystem(
        index=1,
        line_y=[120, 130, 140, 150, 160],
        top=75,
        bottom=195,
        left=40,
        right=760,
        spacing=10,
        barlines=[760],
        measure_count=1,
    )


def _layout(*, confidence: float = 1.0) -> PageLayout:
    return PageLayout(800, 260, [_staff()], confidence)


def _write_score(
    path: Path,
    *,
    note_count: int = 8,
    divisions: int = 2,
    note_type: str = "eighth",
    voices: tuple[int, ...] = (1,),
    existing_beam: bool = False,
    chord_at: int | None = None,
) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Part 1"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = str(divisions)
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"

    for voice_index, voice in enumerate(voices):
        if voice_index:
            backup = etree.SubElement(measure, "backup")
            etree.SubElement(backup, "duration").text = str(note_count)
        for index in range(note_count):
            note = etree.SubElement(measure, "note")
            if chord_at == index:
                etree.SubElement(note, "chord")
            pitch = etree.SubElement(note, "pitch")
            etree.SubElement(pitch, "step").text = "C"
            etree.SubElement(pitch, "octave").text = "4"
            etree.SubElement(note, "duration").text = "1"
            etree.SubElement(note, "voice").text = str(voice)
            etree.SubElement(note, "type").text = note_type
            etree.SubElement(note, "staff").text = "1"
            if existing_beam and voice_index == 0 and index == 0:
                beam = etree.SubElement(note, "beam", number="1")
                beam.text = "begin"

    etree.ElementTree(root).write(
        str(path),
        encoding="UTF-8",
        xml_declaration=True,
        doctype=MUSICXML_DOCTYPE,
    )


def _beam(
    left: int,
    right: int,
    *,
    top: int = 105,
    bottom: int = 111,
) -> SemanticDetection:
    return SemanticDetection(
        class_name="beam",
        label=0,
        bbox=(left, top, right, bottom),
        confidence=0.999,
        staff_index=1,
        placement="within",
    )


def _beam_values(path: Path, number: int) -> list[str]:
    root = etree.parse(str(path)).getroot()
    return [
        str(beam.text)
        for beam in root.findall(f"./part/measure/note/beam[@number='{number}']")
    ]


def test_primary_beam_is_committed_without_changing_note_semantics(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "score.musicxml"
    _write_score(xml_path)
    before = analyze_musicxml(xml_path)

    report = enrich_musicxml_with_source_beams(
        xml_path,
        _layout(),
        (_beam(94, 324),),
    )

    assert report.transaction_committed
    assert report.injected_segment_count == 1
    assert report.injected_marker_count == 4
    assert _beam_values(xml_path, 1) == [
        "begin",
        "continue",
        "continue",
        "end",
    ]
    after = analyze_musicxml(xml_path)
    for key in (
        "part_count",
        "measure_count",
        "note_count",
        "rest_count",
        "rhythm_issues",
        "tie_issues",
        "slur_issues",
    ):
        assert after[key] == before[key]


def test_system_opening_header_translation_uses_unique_beam_span(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "score.musicxml"
    _write_score(xml_path)
    tree = etree.parse(str(xml_path))
    for note in tree.findall("./part/measure/note")[3:]:
        note.find("type").text = "quarter"  # type: ignore[union-attr]
    tree.write(
        str(xml_path),
        encoding="UTF-8",
        xml_declaration=True,
        doctype=MUSICXML_DOCTYPE,
    )

    report = enrich_musicxml_with_source_beams(
        xml_path,
        _layout(),
        # The first note is translated right by a clef/key/time block.  Its
        # three-note within-beam spacing still identifies the only valid group.
        (_beam(220, 373),),
    )

    assert report.transaction_committed
    assert report.injected_segment_count == 1
    assert _beam_values(xml_path, 1) == [
        "begin",
        "continue",
        "end",
    ]
    notes = etree.parse(str(xml_path)).findall("./part/measure/note")
    assert [
        note.findtext("beam")
        for note in notes
    ] == ["begin", "continue", "end", None, None, None, None, None]


def test_shifted_system_opening_suffix_abstains_beyond_calibrated_error() -> None:
    events = [
        _Event(etree.Element("note"), onset, 2, 1, index)
        for index, onset in enumerate((0, 2, 4, 6))
    ]

    segment, error = _assign_segment(
        events,
        start_offset=0.43821510297482835,
        end_offset=0.8409610983981693,
        measure_duration=8,
    )

    assert segment is None
    assert error == "beam endpoints do not align with the event lattice"

    assignments, levels, group_error = _assign_group(
        events,
        [(_beam(406, 582), 0.43821510297482835, 0.8409610983981693)],
        measure_duration=8,
    )
    assert assignments is None
    assert levels is None
    assert "ambiguous across a complete short-note run" in str(group_error)


def test_semantic_barlines_override_noisy_projection_candidates() -> None:
    upper = _staff()
    lower = _staff()
    lower.index = 2
    lower.line_y = [value + 90 for value in lower.line_y]
    upper.barlines = [100, 188, 250, 400, 550, 700]
    lower.barlines = [100, 250, 333, 400, 550, 700]
    semantic = tuple(
        SemanticDetection(
            class_name="genericBarline",
            label=12,
            bbox=(x - 2, 100, x + 2, 260),
            confidence=0.99,
            staff_index=staff_index,
            placement="within",
        )
        for x in (100, 250, 400, 550, 700)
        for staff_index in (1, 2)
    )

    solution, error = _consensus_measure_boundaries(
        (upper, lower),
        target_measure_count=4,
        layout_confidence=0.98,
        page_width=800,
        semantic_barlines=semantic,
    )

    assert error is None
    assert solution is not None
    assert solution.boundaries == (100.0, 250.0, 400.0, 550.0, 700.0)
    assert solution.method == "semantic-barline-recognized-count-exact"
    assert solution.confidence > 0.95


def test_semantic_barlines_from_other_score_systems_are_ignored() -> None:
    semantic = tuple(
        SemanticDetection(
            class_name="genericBarline",
            label=12,
            bbox=(x - 2, 100, x + 2, 160),
            confidence=0.99,
            staff_index=staff_index,
            placement="within",
        )
        for staff_index, positions in (
            (1, (100, 250, 400, 550, 700)),
            (9, (130, 190, 310, 470, 610, 730)),
        )
        for x in positions
    )

    solution, error = _consensus_measure_boundaries(
        (_staff(),),
        target_measure_count=4,
        layout_confidence=0.98,
        page_width=800,
        semantic_barlines=semantic,
    )

    assert error is None
    assert solution is not None
    assert solution.boundaries == (100.0, 250.0, 400.0, 550.0, 700.0)
    assert solution.method == "semantic-barline-recognized-count-exact"


def test_semantic_right_barlines_use_staff_edge_as_ordinary_opening() -> None:
    staff = _staff()
    staff.left = 100
    semantic = tuple(
        SemanticDetection(
            class_name="genericBarline",
            label=12,
            bbox=(x - 2, 100, x + 2, 160),
            confidence=0.99,
            staff_index=1,
            placement="within",
        )
        for x in (250, 400, 550, 700)
    )

    solution, error = _consensus_measure_boundaries(
        (staff,),
        target_measure_count=4,
        layout_confidence=0.98,
        page_width=800,
        semantic_barlines=semantic,
    )

    assert error is None
    assert solution is not None
    assert solution.boundaries == (100.0, 250.0, 400.0, 550.0, 700.0)
    assert solution.method == "semantic-right-barlines-with-staff-opening"
    assert solution.confidence > 0.95


def test_parallel_beam_levels_share_one_clear_staff_owner() -> None:
    from types import SimpleNamespace

    upper = _staff()
    lower = _staff()
    lower.index = 2
    lower.line_y = [260, 270, 280, 290, 300]
    detections = (
        SemanticDetection(
            "beam",
            3,
            (200, 190, 400, 200),
            0.99,
            1,
            "below",
        ),
        SemanticDetection(
            "beam",
            3,
            (200, 210, 400, 220),
            0.99,
            2,
            "above",
        ),
    )

    normalized = _normalize_parallel_beam_staff_assignments(
        SimpleNamespace(ordered_appearances=(upper, lower)),
        detections,
    )

    assert [item.staff_index for item in normalized] == [1, 1]


def test_aligned_piano_beams_with_a_staff_space_gap_keep_separate_owners() -> None:
    from types import SimpleNamespace

    upper = _staff()
    lower = _staff()
    lower.index = 2
    lower.line_y = [240, 250, 260, 270, 280]
    detections = (
        SemanticDetection(
            "beam",
            3,
            (200, 170, 400, 182),
            0.99,
            1,
            "below",
        ),
        SemanticDetection(
            "beam",
            3,
            (200, 220, 400, 252),
            0.99,
            2,
            "above",
        ),
    )

    normalized = _normalize_parallel_beam_staff_assignments(
        SimpleNamespace(ordered_appearances=(upper, lower)),
        detections,
    )

    assert [item.staff_index for item in normalized] == [1, 2]


def test_parallel_beam_normalization_tolerates_unmatched_layout_group() -> None:
    from types import SimpleNamespace

    upper = _staff()
    lower = _staff()
    lower.index = 2
    lower.line_y = [260, 270, 280, 290, 300]
    detections = (
        SemanticDetection(
            "beam",
            3,
            (200, 190, 400, 200),
            0.99,
            1,
            "below",
        ),
        SemanticDetection(
            "beam",
            3,
            (200, 210, 400, 220),
            0.99,
            2,
            "above",
        ),
    )
    part = SimpleNamespace(
        system_measure_counts=(4,),
        system_measure_offsets=(0,),
    )
    topology = SimpleNamespace(
        ordered_appearances=(upper, lower),
        appearance_locations=(
            (1, 1, 0, 1),
            (2, 1, 0, 2),
        ),
        parts=(part,),
    )

    normalized = _normalize_parallel_beam_staff_assignments(
        topology,
        detections,
    )

    assert [item.staff_index for item in normalized] == [1, 1]


def test_repeated_high_level_polygon_splits_merge_inside_primary_span() -> None:
    events = [
        _Event(etree.Element("note"), index, 1, 3, index)
        for index in range(8)
    ]

    def assignment(
        indices: tuple[int, ...],
        top: int,
    ) -> tuple[SemanticDetection, _Segment]:
        return (
            SemanticDetection(
                "beam",
                3,
                (indices[0] * 20, top, (indices[-1] + 1) * 20, top + 5),
                1.0,
                1,
                "above",
            ),
            _Segment(indices, 0.01, 0.02),
        )

    assignments = [
        assignment(tuple(range(8)), 100),
        assignment(tuple(range(4)), 110),
        assignment(tuple(range(4, 8)), 118),
        assignment(tuple(range(4)), 120),
        assignment(tuple(range(4, 8)), 128),
    ]

    merged, levels = _merge_repeated_high_level_path_splits(
        events,
        assignments,
        [1, 2, 2, 3, 3],
        source_staff_spacing=10.0,
    )

    assert levels == [1, 2, 2, 3, 3]
    assert [item[1].event_indices for item in merged] == [
        tuple(range(8)),
        tuple(range(8)),
        tuple(range(8)),
        tuple(range(8)),
        tuple(range(8)),
    ]


def test_one_secondary_level_keeps_intentional_beam_partition() -> None:
    events = [
        _Event(etree.Element("note"), index, 1, 2, index)
        for index in range(8)
    ]
    primary = (
        SemanticDetection("beam", 3, (0, 100, 160, 105), 1.0, 1, "above"),
        _Segment(tuple(range(8)), 0.0, 0.0),
    )
    left = (
        SemanticDetection("beam", 3, (0, 110, 80, 115), 1.0, 1, "above"),
        _Segment(tuple(range(4)), 0.0, 0.0),
    )
    right = (
        SemanticDetection("beam", 3, (80, 110, 160, 115), 1.0, 1, "above"),
        _Segment(tuple(range(4, 8)), 0.0, 0.0),
    )

    merged, _levels = _merge_repeated_high_level_path_splits(
        events,
        [primary, left, right],
        [1, 2, 2],
        source_staff_spacing=10.0,
    )

    assert [item[1].event_indices for item in merged] == [
        tuple(range(8)),
        tuple(range(4)),
        tuple(range(4, 8)),
    ]


def test_explicit_left_barline_prunes_the_preclef_staff_edge() -> None:
    semantic = tuple(
        SemanticDetection(
            class_name="genericBarline",
            label=12,
            bbox=(x - 2, 100, x + 2, 260),
            confidence=0.99,
            staff_index=1,
            placement="within",
        )
        for x in (100, 200, 400, 600)
    )

    solution, error = _consensus_measure_boundaries(
        (_staff(),),
        target_measure_count=2,
        layout_confidence=0.98,
        page_width=800,
        semantic_barlines=semantic,
        system_opening_has_left_barline=True,
    )

    assert error is None
    assert solution is not None
    assert solution.boundaries == (200.0, 400.0, 600.0)
    assert solution.method == "semantic-left-barline-header-pruned"


def test_system_opening_does_not_shift_one_beam_to_a_suffix() -> None:
    events = [
        _Event(etree.Element("note"), onset, 2, 1, index)
        for index, onset in enumerate((0, 2, 4))
    ]

    assignments, levels, error = _assign_group(
        events,
        [(_beam(400, 560), 0.4128, 0.6962)],
        measure_duration=6,
        system_opening=True,
    )

    assert assignments is None
    assert levels is None
    assert "system-opening" in str(error)


def test_irregular_partition_of_complete_16th_run_abstains() -> None:
    events = [
        _Event(etree.Element("note"), index, 1, 2, index)
        for index in range(8)
    ]
    first = (0.08165680473372781, 0.3609467455621302)
    second = (0.4461538461538462, 0.8366863905325443)
    group = [
        (_beam(100, 220, top=100 + level * 8, bottom=104 + level * 8), *span)
        for span in (first, second)
        for level in (1, 2)
    ]

    assignments, levels, error = _assign_group(
        events,
        group,
        measure_duration=8,
    )

    assert assignments is None
    assert levels is None
    assert "regular complete run" in str(error)


def test_source_exact_low_error_four_plus_two_16th_partition_is_kept() -> None:
    events = [
        _Event(etree.Element("note"), index, 1, 2, index)
        for index in range(6)
    ]
    group = [
        (_beam(100, 250, top=100 + level * 8, bottom=104 + level * 8), *span)
        for span in ((0.075, 0.50), (0.6416667, 0.7833333))
        for level in (1, 2)
    ]

    assignments, levels, error = _assign_group(
        events,
        group,
        measure_duration=6,
    )

    assert error is None
    assert assignments is not None
    assert levels == [1, 2, 1, 2]
    assert sorted(
        {
            assignment.event_indices
            for _detection, assignment in assignments
        }
    ) == [(0, 1, 2, 3), (4, 5)]


def test_recognized_default_x_resolves_compressed_system_opening_run() -> None:
    events = [
        _Event(
            etree.Element("note"),
            onset,
            2,
            1,
            index,
            visual_position,
        )
        for index, (onset, visual_position) in enumerate(
            (
                (0, 0.4390),
                (2, 0.5702),
                (4, 0.7014),
                (6, 0.8325),
            )
        )
    ]

    segment, error = _assign_segment(
        events,
        start_offset=0.4382,
        end_offset=0.8410,
        measure_duration=8,
    )

    assert error is None
    assert segment is not None
    assert segment.event_indices == (0, 1, 2, 3)
    assert segment.endpoint_error < 0.01


def test_recognized_default_x_uses_the_up_stem_attachment_edge() -> None:
    measure = etree.Element("measure", width="100")
    for default_x, stem in (("10", "down"), ("50", "up")):
        note = etree.SubElement(measure, "note", **{"default-x": default_x})
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = "C"
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "1"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "16th"
        etree.SubElement(note, "stem").text = stem
        etree.SubElement(note, "staff").text = "1"

    events, error = _events_for_staff(measure, 1)

    assert error is None
    assert [event.visual_position for event in events] == [0.10, 0.62]


def test_grace_beam_run_does_not_interrupt_the_main_voice_run() -> None:
    events = [
        _Event(etree.Element("note"), 0, 1, 2, 0),
        _Event(etree.Element("note"), 1, 1, 2, 1),
        _Event(etree.Element("note"), 2, 0, 2, 2, grace=True),
        _Event(etree.Element("note"), 2, 0, 2, 3, grace=True),
        _Event(etree.Element("note"), 2, 1, 1, 4),
    ]

    assert _eligible_runs(events) == ((0, 1, 4), (2, 3))


def test_single_wide_beam_covers_the_only_complete_short_note_run(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "score.musicxml"
    _write_score(xml_path)

    report = enrich_musicxml_with_source_beams(
        xml_path,
        _layout(),
        # Deliberately compressed/translated relative to the default engraving
        # model, but still more than half of the resolved measure width.
        (_beam(210, 630),),
    )

    assert report.transaction_committed
    notes = etree.parse(str(xml_path)).findall("./part/measure/note")
    assert [note.findtext("beam") for note in notes] == [
        "begin",
        "continue",
        "continue",
        "continue",
        "continue",
        "continue",
        "continue",
        "end",
    ]


def test_parallel_beam_segments_create_two_complete_levels(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "score.musicxml"
    _write_score(
        xml_path,
        note_count=16,
        divisions=4,
        note_type="16th",
    )

    report = enrich_musicxml_with_source_beams(
        xml_path,
        _layout(),
        (
            _beam(94, 209, top=104, bottom=108),
            _beam(94, 209, top=111, bottom=115),
        ),
    )

    assert report.transaction_committed
    assert report.injected_segment_count == 2
    assert _beam_values(xml_path, 1) == [
        "begin",
        "continue",
        "continue",
        "end",
    ]
    assert _beam_values(xml_path, 2) == [
        "begin",
        "continue",
        "continue",
        "end",
    ]


def test_connected_thick_beam_restores_uniform_16th_levels() -> None:
    events = [
        _Event(etree.Element("note"), index, 1, 2, index)
        for index in range(6)
    ]
    thick = _beam(100, 600, top=100, bottom=113)

    assignments, levels, error = _assign_group(
        events,
        [(thick, 0.08, 0.88)],
        measure_duration=6,
        source_staff_spacing=10.0,
    )

    assert error is None
    assert assignments is not None
    assert [segment.event_indices for _item, segment in assignments] == [
        (0, 1, 2, 3, 4, 5),
        (0, 1, 2, 3, 4, 5),
    ]
    assert levels == [1, 2]


def test_split_svg_paths_merge_into_one_nested_semantic_hierarchy() -> None:
    events = [
        _Event(
            etree.Element("note"),
            onset,
            duration,
            level,
            index,
        )
        for index, (onset, duration, level) in enumerate(
            (
                (0, 4, 1),
                (4, 2, 2),
                (6, 1, 3),
                (7, 1, 3),
                (8, 1, 3),
                (9, 1, 3),
                (10, 1, 3),
            )
        )
    ]
    assignments, levels, error = _assign_group(
        events,
        [
            (_beam(100, 800, top=140, bottom=150), 0.10, 0.80),
            (_beam(200, 400, top=120, bottom=130), 0.20, 0.40),
            (_beam(500, 800, top=120, bottom=130), 0.50, 0.80),
            (_beam(300, 400, top=100, bottom=110), 0.30, 0.40),
            (_beam(500, 800, top=100, bottom=110), 0.50, 0.80),
        ],
        measure_duration=11,
        source_staff_spacing=20.0,
    )

    assert error is None
    assert assignments is not None
    assert sorted(set(levels)) == [1, 2, 3]
    assert sorted(levels.count(level) for level in (1, 2, 3)) == [1, 2, 2]
    assert [
        segment.event_indices
        for _detection, segment in assignments
    ] == [
        (0, 1, 2, 3, 4, 5, 6),
        (1, 2, 3, 4, 5, 6),
        (1, 2, 3, 4, 5, 6),
        (2, 3, 4, 5, 6),
        (2, 3, 4, 5, 6),
    ]


def test_three_level_nested_hierarchy_resolves_one_bounded_hook() -> None:
    events = [
        _Event(
            etree.Element("note"),
            onset,
            duration,
            level,
            index,
            visual_position,
        )
        for index, (onset, duration, level, visual_position) in enumerate(
            (
                (0, 12, 1, 0.115),
                (12, 9, 2, 0.352),
                (21, 3, 3, 0.604),
                (24, 12, 1, 0.736),
            )
        )
    ]
    assignments, levels, error = _assign_group(
        events,
        [
            (_beam(100, 700, top=140, bottom=150), 0.24, 0.83),
            (_beam(300, 550, top=120, bottom=130), 0.46, 0.70),
            (_beam(450, 550, top=100, bottom=110), 0.60, 0.70),
        ],
        measure_duration=36,
        source_staff_spacing=20.0,
    )

    assert error is None
    assert assignments is not None
    assert levels == [1, 2, 3]
    assert [
        segment.event_indices
        for _detection, segment in assignments
    ] == [(0, 1, 2, 3), (1, 2), (2,)]


def test_two_event_parallel_levels_survive_a_large_header_translation() -> None:
    events = [
        _Event(
            etree.Element("note"),
            index,
            1,
            2,
            index,
            visual_position,
        )
        for index, visual_position in enumerate((0.62, 0.80))
    ]

    assignments, levels, error = _assign_group(
        events,
        [
            (_beam(200, 600), 0.19, 0.61),
            (_beam(200, 600, top=112, bottom=116), 0.19, 0.61),
        ],
        measure_duration=2,
        system_opening=True,
    )

    assert error is None
    assert assignments is not None
    assert [segment.event_indices for _detection, segment in assignments] == [
        (0, 1),
        (0, 1),
    ]
    assert levels == [1, 2]


def test_parallel_levels_keep_a_direct_subgroup_inside_one_short_note_run() -> None:
    events = [
        _Event(
            etree.Element("note"),
            index,
            1,
            2,
            index,
            visual_position,
        )
        for index, visual_position in enumerate(
            (0.10, 0.25, 0.40, 0.58, 0.76)
        )
    ]

    assignments, levels, error = _assign_group(
        events,
        [
            (_beam(250, 580), 0.25, 0.58),
            (_beam(250, 580, top=112, bottom=116), 0.25, 0.58),
        ],
        measure_duration=5,
    )

    assert error is None
    assert assignments is not None
    assert [
        segment.event_indices
        for _detection, segment in assignments
    ] == [(1, 2, 3), (1, 2, 3)]
    assert levels == [1, 2]


def test_complete_parallel_run_uses_bounded_direct_endpoints() -> None:
    events = [
        _Event(
            etree.Element("note"),
            index,
            1,
            2,
            index,
            visual_position,
        )
        for index, visual_position in enumerate((0.060, 0.258, 0.456, 0.653))
    ]

    assignments, levels, error = _assign_group(
        events,
        [
            (_beam(70, 740), 0.073, 0.741),
            (_beam(70, 740, top=112, bottom=116), 0.073, 0.741),
        ],
        measure_duration=4,
    )

    assert error is None
    assert assignments is not None
    assert levels == [1, 2]
    assert all(
        segment.event_indices == (0, 1, 2, 3)
        for _detection, segment in assignments
    )


def test_parallel_source_levels_select_the_only_scale_bounded_run() -> None:
    events = [
        _Event(etree.Element("note"), 0, 24, 0, 0, 0.13),
        _Event(etree.Element("note"), 24, 3, 0, 1, 0.41),
        _Event(etree.Element("note"), 27, 3, 2, 2, 0.586),
        _Event(etree.Element("note"), 30, 3, 2, 3, 0.690),
        _Event(etree.Element("note"), 33, 3, 2, 4, 0.795),
    ]

    assignments, levels, error = _assign_group(
        events,
        [
            (_beam(650, 890), 0.652, 0.893),
            (_beam(650, 890, top=112, bottom=116), 0.652, 0.893),
        ],
        measure_duration=36,
    )

    assert error is None
    assert assignments is not None
    assert levels == [1, 2]
    assert [
        segment.event_indices
        for _detection, segment in assignments
    ] == [(2, 3, 4), (2, 3, 4)]


def test_parallel_levels_stop_before_a_lower_level_trailing_note() -> None:
    events = [
        _Event(
            etree.Element("note"),
            onset,
            duration,
            level_count,
            index,
            visual_position,
        )
        for index, (onset, duration, level_count, visual_position) in enumerate(
            (
                (0, 1, 2, 0.154),
                (1, 1, 2, 0.293),
                (2, 1, 2, 0.432),
                (3, 1, 2, 0.571),
                (4, 2, 1, 0.710),
            )
        )
    ]

    assignments, levels, error = _assign_group(
        events,
        [
            (_beam(180, 655), 0.180, 0.655),
            (_beam(180, 655, top=112, bottom=116), 0.180, 0.655),
        ],
        measure_duration=6,
    )

    assert error is None
    assert assignments is not None
    assert levels == [1, 2]
    assert [
        segment.event_indices
        for _detection, segment in assignments
    ] == [(0, 1, 2, 3), (0, 1, 2, 3)]


def test_complete_multi_group_partition_resolves_joint_translation() -> None:
    positions = (
        0.137,
        0.207,
        0.276,
        0.346,
        0.416,
        0.486,
        0.556,
        0.626,
        0.696,
        0.766,
        0.836,
        0.906,
    )
    events = [
        _Event(
            etree.Element("note"),
            index * 3,
            3,
            0 if index == 0 else 2,
            index,
            position,
        )
        for index, position in enumerate(positions)
    ]
    group = [
        (_beam(100, 200, top=100 + level * 8, bottom=104 + level * 8), *span)
        for span in (
            (0.150, 0.305),
            (0.375, 0.605),
            (0.676, 0.906),
        )
        for level in (1, 2)
    ]

    assignments, levels, error = _assign_group(
        events,
        group,
        measure_duration=36,
    )

    assert error is None
    assert assignments is not None
    assert levels is not None
    assert sorted(
        {
            segment.event_indices
            for (_detection, segment), level in zip(
                assignments,
                levels,
                strict=True,
            )
            if level == 1
        }
    ) == [(1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)]


def test_single_event_secondary_segment_becomes_forward_hook(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "score.musicxml"
    _write_score(
        xml_path,
        note_count=16,
        divisions=4,
        note_type="16th",
    )

    report = enrich_musicxml_with_source_beams(
        xml_path,
        _layout(),
        (
            _beam(94, 209, top=104, bottom=108),
            _beam(90, 98, top=111, bottom=115),
        ),
    )

    assert report.transaction_committed
    assert _beam_values(xml_path, 1) == [
        "begin",
        "continue",
        "continue",
        "end",
    ]
    assert _beam_values(xml_path, 2) == ["forward hook"]


def test_equidistant_internal_secondary_segment_uses_forward_hook() -> None:
    events = [
        _Event(
            etree.Element("note"),
            index,
            1,
            2,
            index,
            visual_position,
        )
        for index, visual_position in enumerate((0.10, 0.30, 0.50))
    ]
    assignments, levels, error = _assign_group(
        events,
        [
            (_beam(100, 500), 0.10, 0.50),
            (_beam(300, 360, top=112, bottom=116), 0.30, 0.36),
        ],
        measure_duration=3,
    )

    assert error is None
    assert assignments is not None
    assert levels is not None
    for (_detection, segment), level in sorted(
        zip(assignments, levels, strict=True),
        key=lambda item: item[1],
    ):
        _write_segment(events, segment, level)

    assert [
        beam.text
        for event in events
        for beam in event.element.findall("beam[@number='2']")
    ] == ["forward hook"]


def test_hook_direction_uses_its_containing_primary_group() -> None:
    events = [
        _Event(etree.Element("note"), index, 1, 2, index)
        for index in range(4)
    ]
    _write_segment(events, _Segment((0, 1), 0.0, 0.0), 1)
    _write_segment(events, _Segment((2, 3), 0.0, 0.0), 1)
    _write_segment(events, _Segment((1,), 0.0, 0.0), 2)

    assert events[1].element.findtext("beam[@number='2']") == "backward hook"


def test_one_source_strip_restores_a_beam_across_measure_boundaries(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "score.musicxml"
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Part 1"
    part = etree.SubElement(root, "part", id="P1")
    for measure_index in range(2):
        measure = etree.SubElement(
            part,
            "measure",
            number=str(measure_index + 1),
        )
        if measure_index == 0:
            attributes = etree.SubElement(measure, "attributes")
            etree.SubElement(attributes, "divisions").text = "1"
            time = etree.SubElement(attributes, "time")
            etree.SubElement(time, "beats").text = "2"
            etree.SubElement(time, "beat-type").text = "8"
        for _index in range(2):
            note = etree.SubElement(measure, "note")
            pitch = etree.SubElement(note, "pitch")
            etree.SubElement(pitch, "step").text = "C"
            etree.SubElement(pitch, "octave").text = "4"
            etree.SubElement(note, "duration").text = "1"
            etree.SubElement(note, "voice").text = "1"
            etree.SubElement(note, "type").text = "eighth"
            etree.SubElement(note, "staff").text = "1"
    etree.ElementTree(root).write(
        str(xml_path),
        encoding="UTF-8",
        xml_declaration=True,
        doctype=MUSICXML_DOCTYPE,
    )
    staff = _staff()
    staff.barlines = [400]
    staff.measure_count = 2
    layout = PageLayout(800, 260, [staff], 1.0)

    report = enrich_musicxml_with_source_beams(
        xml_path,
        layout,
        (_beam(67, 580),),
    )

    assert report.transaction_committed
    assert _beam_values(xml_path, 1) == [
        "begin",
        "continue",
        "continue",
        "end",
    ]


def test_existing_beam_markup_abstains_and_preserves_original_bytes(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "score.musicxml"
    _write_score(xml_path, existing_beam=True)
    original = xml_path.read_bytes()

    report = enrich_musicxml_with_source_beams(
        xml_path,
        _layout(),
        (_beam(94, 324),),
    )

    assert not report.transaction_committed
    assert report.injected_segment_count == 0
    assert "existing MusicXML beams" in report.proposals[0].reason
    assert xml_path.read_bytes() == original


def test_multiple_voices_abstain_instead_of_guessing_relationships(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "score.musicxml"
    _write_score(xml_path, voices=(1, 2))
    original = xml_path.read_bytes()

    report = enrich_musicxml_with_source_beams(
        xml_path,
        _layout(),
        (_beam(94, 324),),
    )

    assert not report.transaction_committed
    assert "one unique beam assignment" in report.proposals[0].reason
    assert xml_path.read_bytes() == original


def test_multiple_voices_commit_when_only_one_voice_can_form_the_beam(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "score.musicxml"
    _write_score(xml_path, voices=(1, 2))
    tree = etree.parse(str(xml_path))
    for note in tree.findall("./part/measure/note")[8:]:
        note.find("type").text = "quarter"  # type: ignore[union-attr]
    tree.write(
        str(xml_path),
        encoding="UTF-8",
        xml_declaration=True,
        doctype=MUSICXML_DOCTYPE,
    )

    report = enrich_musicxml_with_source_beams(
        xml_path,
        _layout(),
        (_beam(94, 324),),
    )

    assert report.transaction_committed
    notes = etree.parse(str(xml_path)).findall("./part/measure/note")
    assert [note.findtext("beam") for note in notes[:8]] == [
        "begin",
        "continue",
        "continue",
        "end",
        None,
        None,
        None,
        None,
    ]
    assert all(note.find("beam") is None for note in notes[8:])


def test_multiple_voices_use_explicit_stems_and_source_placement(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "score.musicxml"
    _write_score(xml_path, voices=(1, 2))
    tree = etree.parse(str(xml_path))
    notes = tree.findall("./part/measure/note")
    for note in notes[:8]:
        etree.SubElement(note, "stem").text = "up"
    for note in notes[8:]:
        etree.SubElement(note, "stem").text = "down"
    tree.write(
        str(xml_path),
        encoding="UTF-8",
        xml_declaration=True,
        doctype=MUSICXML_DOCTYPE,
    )

    report = enrich_musicxml_with_source_beams(
        xml_path,
        _layout(),
        (
            SemanticDetection(
                "beam",
                0,
                (94, 94, 324, 100),
                0.999,
                1,
                "above",
            ),
        ),
    )

    assert report.transaction_committed
    notes = etree.parse(str(xml_path)).findall("./part/measure/note")
    assert [note.findtext("beam") for note in notes[:4]] == [
        "begin",
        "continue",
        "continue",
        "end",
    ]
    assert all(note.find("beam") is None for note in notes[8:])


def test_one_voice_beam_can_cross_keyboard_staves(tmp_path: Path) -> None:
    xml_path = tmp_path / "score.musicxml"
    _write_score(xml_path)
    tree = etree.parse(str(xml_path))
    attributes = tree.find("./part/measure/attributes")
    assert attributes is not None
    etree.SubElement(attributes, "staves").text = "2"
    notes = tree.findall("./part/measure/note")
    for note in notes[2:]:
        note.find("staff").text = "2"  # type: ignore[union-attr]
    tree.write(
        str(xml_path),
        encoding="UTF-8",
        xml_declaration=True,
        doctype=MUSICXML_DOCTYPE,
    )
    upper = _staff()
    lower = _staff()
    lower.index = 2
    lower.line_y = [220, 230, 240, 250, 260]
    lower.top = 175
    lower.bottom = 295
    layout = PageLayout(800, 340, [upper, lower], 1.0)

    report = enrich_musicxml_with_source_beams(
        xml_path,
        layout,
        (_beam(94, 324),),
    )

    assert report.transaction_committed
    notes = etree.parse(str(xml_path)).findall("./part/measure/note")
    assert [note.findtext("beam") for note in notes[:4]] == [
        "begin",
        "continue",
        "continue",
        "end",
    ]
    assert [note.findtext("staff") for note in notes[:4]] == [
        "1",
        "1",
        "2",
        "2",
    ]


def test_interstaff_beam_can_recover_from_nearest_staff_misassignment(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "score.musicxml"
    _write_score(xml_path, voices=(1, 2))
    tree = etree.parse(str(xml_path))
    attributes = tree.find("./part/measure/attributes")
    assert attributes is not None
    etree.SubElement(attributes, "staves").text = "2"
    notes = tree.findall("./part/measure/note")
    for note in notes[8:]:
        note.find("staff").text = "2"  # type: ignore[union-attr]
        note.find("type").text = "quarter"  # type: ignore[union-attr]
    tree.write(
        str(xml_path),
        encoding="UTF-8",
        xml_declaration=True,
        doctype=MUSICXML_DOCTYPE,
    )
    upper = _staff()
    lower = _staff()
    lower.index = 2
    lower.line_y = [220, 230, 240, 250, 260]
    lower.top = 175
    lower.bottom = 295
    layout = PageLayout(800, 340, [upper, lower], 1.0)
    detection = SemanticDetection(
        "beam",
        0,
        (94, 190, 324, 198),
        0.999,
        2,
        "above",
    )

    report = enrich_musicxml_with_source_beams(
        xml_path,
        layout,
        (detection,),
    )

    assert report.transaction_committed
    notes = etree.parse(str(xml_path)).findall("./part/measure/note")
    assert [note.findtext("beam") for note in notes[:4]] == [
        "begin",
        "continue",
        "continue",
        "end",
    ]
    assert all(note.find("beam") is None for note in notes[8:])


def test_safe_chord_continuation_keeps_beams_on_carrier_notes(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "score.musicxml"
    _write_score(xml_path, chord_at=2)

    report = enrich_musicxml_with_source_beams(
        xml_path,
        _layout(),
        (_beam(94, 356),),
    )

    assert report.transaction_committed
    notes = etree.parse(str(xml_path)).findall("./part/measure/note")
    assert notes[2].find("chord") is not None
    assert notes[2].find("beam") is None
    assert [
        index
        for index, note in enumerate(notes)
        if note.find("beam") is not None
    ] == [0, 1, 3, 4]


def test_orphan_chord_continuation_abstains(tmp_path: Path) -> None:
    xml_path = tmp_path / "score.musicxml"
    _write_score(xml_path, chord_at=0)
    original = xml_path.read_bytes()

    report = enrich_musicxml_with_source_beams(
        xml_path,
        _layout(),
        (_beam(94, 324),),
    )

    assert not report.transaction_committed
    assert "chord continuation" in report.proposals[0].reason
    assert xml_path.read_bytes() == original


def test_weak_source_mapping_abstains(tmp_path: Path) -> None:
    xml_path = tmp_path / "score.musicxml"
    _write_score(xml_path)
    original = xml_path.read_bytes()

    report = enrich_musicxml_with_source_beams(
        xml_path,
        _layout(confidence=0.5),
        (_beam(94, 324),),
    )

    assert not report.transaction_committed
    assert "confidence is too low" in report.proposals[0].reason
    assert xml_path.read_bytes() == original


def test_validation_failure_rolls_back_entire_transaction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    xml_path = tmp_path / "score.musicxml"
    _write_score(xml_path)
    original = xml_path.read_bytes()
    monkeypatch.setattr(
        "scorescan.beam_enrichment.validate_musicxml",
        lambda _path: ["forced validation failure"],
    )

    report = enrich_musicxml_with_source_beams(
        xml_path,
        _layout(),
        (_beam(94, 324),),
    )

    assert not report.transaction_committed
    assert report.injected_segment_count == 0
    assert "forced validation failure" in str(report.error)
    assert "transaction rolled back" in report.proposals[0].reason
    assert xml_path.read_bytes() == original
