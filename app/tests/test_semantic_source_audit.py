from __future__ import annotations

from pathlib import Path

from scorescan.layout import PageLayout, StaffSystem
from scorescan.semantic_detector import SemanticDetection
from scorescan.semantic_source_audit import audit_semantic_source_symbols


def _staff(index: int, y: int) -> StaffSystem:
    return StaffSystem(
        index=index,
        line_y=[y + offset for offset in (0, 10, 20, 30, 40)],
        top=y - 20,
        bottom=y + 60,
        left=0,
        right=200,
        spacing=10.0,
        barlines=[0, 100, 200],
        measure_count=2,
    )


def _xml(measures: list[str], *, staves: int = 1) -> str:
    body = []
    for index, notes in enumerate(measures, start=1):
        attributes = (
            f"<attributes><divisions>1</divisions><staves>{staves}</staves>"
            "<key><fifths>0</fifths></key></attributes>"
            if index == 1
            else ""
        )
        body.append(f'<measure number="{index}">{attributes}{notes}</measure>')
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<score-partwise version="4.0"><part-list><score-part id="P1">'
        "<part-name>Music</part-name></score-part></part-list>"
        f'<part id="P1">{"".join(body)}</part></score-partwise>'
    )


def _note(*, alter: int = 0, staff: int = 1) -> str:
    alter_xml = f"<alter>{alter}</alter>" if alter else ""
    return (
        "<note><pitch><step>G</step>"
        f"{alter_xml}<octave>4</octave></pitch><duration>1</duration>"
        f"<voice>1</voice><type>quarter</type><staff>{staff}</staff></note>"
    )


def _accidental(x: int, staff_index: int) -> SemanticDetection:
    return SemanticDetection(
        "genericAccidental",
        1,
        (x - 3, 90, x + 3, 110),
        0.999,
        staff_index,
        "above",
    )


def _symbol(class_name: str, x: int = 50, staff_index: int = 1) -> SemanticDetection:
    return SemanticDetection(
        class_name,
        1,
        (x - 5, 50, x + 5, 70),
        0.999,
        staff_index,
        "above",
    )


def test_bidirectional_accidental_inventory_finds_shifted_false_sharp(
    tmp_path: Path,
) -> None:
    xml = tmp_path / "score.musicxml"
    xml.write_text(
        _xml([_note(), _note(), _note(alter=1), _note()]),
        encoding="utf-8",
    )
    layout = PageLayout(
        200,
        300,
        [_staff(1, 40), _staff(2, 180)],
        1.0,
    )
    report = audit_semantic_source_symbols(
        xml,
        layout,
        [_accidental(150, 2)],
    )

    assert report.status == "completed"
    assert [item.to_dict() for item in report.mismatches] == [
        {
            "class_name": "genericAccidental",
            "measure_index": 3,
            "staff_slot": 1,
            "source_count": 0,
            "output_count": 1,
            "kind": "source_absent_output_symbol",
        },
        {
            "class_name": "genericAccidental",
            "measure_index": 4,
            "staff_slot": 1,
            "source_count": 1,
            "output_count": 0,
            "kind": "source_symbol_missing",
        },
    ]


def test_staff_topology_maps_second_multistaff_system_to_later_measure(
    tmp_path: Path,
) -> None:
    xml = tmp_path / "score.musicxml"
    xml.write_text(
        _xml(
            [
                _note(staff=1),
                _note(staff=2),
                _note(staff=1),
                _note(alter=1, staff=2),
            ],
            staves=2,
        ),
        encoding="utf-8",
    )
    layout = PageLayout(
        200,
        500,
        [_staff(1, 40), _staff(2, 120), _staff(3, 280), _staff(4, 360)],
        1.0,
    )
    report = audit_semantic_source_symbols(
        xml,
        layout,
        [_accidental(150, 4)],
    )

    assert report.status == "completed"
    assert report.mismatches == ()
    assert report.source_counts["genericAccidental"] == 1
    assert report.output_counts["genericAccidental"] == 1


def test_same_measure_wrong_staff_is_a_positional_mismatch(
    tmp_path: Path,
) -> None:
    xml = tmp_path / "score.musicxml"
    xml.write_text(
        _xml(
            [
                _note(alter=1, staff=1) + _note(staff=2),
                _note(staff=1) + _note(staff=2),
            ],
            staves=2,
        ),
        encoding="utf-8",
    )
    layout = PageLayout(
        200,
        260,
        [_staff(1, 40), _staff(2, 140)],
        1.0,
    )

    report = audit_semantic_source_symbols(
        xml,
        layout,
        [_accidental(50, 2)],
    )

    assert [
        (
            item.measure_index,
            item.staff_slot,
            item.kind,
        )
        for item in report.mismatches
    ] == [
        (1, 1, "source_absent_output_symbol"),
        (1, 2, "source_symbol_missing"),
    ]


def test_instrument_name_prefix_does_not_shift_source_symbol_measure(
    tmp_path: Path,
) -> None:
    xml = tmp_path / "score.musicxml"
    xml.write_text(
        _xml(
            [
                _note(alter=1, staff=1),
                _note(staff=1),
                _note(staff=1),
                _note(staff=1),
                _note(staff=1),
                _note(staff=1),
            ],
            staves=3,
        ),
        encoding="utf-8",
    )

    def physical(index: int, y: int, left: int) -> StaffSystem:
        return StaffSystem(
            index=index,
            line_y=[y + offset for offset in (0, 8, 16, 24, 32)],
            top=y - 34,
            bottom=y + 66,
            left=left,
            right=901,
            spacing=8.0,
            barlines=[158, 368, 472, 577, 664, 794, 900],
            measure_count=6,
        )

    layout = PageLayout(
        928,
        500,
        [
            physical(1, 194, 34),
            physical(2, 279, 145),
            physical(3, 371, 145),
        ],
        1.0,
    )
    report = audit_semantic_source_symbols(
        xml,
        layout,
        [
            SemanticDetection(
                "genericAccidental",
                1,
                (298, 160, 305, 178),
                0.999,
                1,
                "above",
            )
        ],
    )

    assert report.status == "completed"
    assert report.mismatches == ()


def test_multiple_parts_use_distinct_global_staff_slots(
    tmp_path: Path,
) -> None:
    xml = tmp_path / "ensemble.musicxml"
    xml.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<score-partwise version="4.0"><part-list>'
        '<score-part id="P1"><part-name>Trumpet</part-name></score-part>'
        '<score-part id="P2"><part-name>Piano</part-name></score-part>'
        "</part-list>"
        '<part id="P1"><measure number="1"><attributes><divisions>1</divisions>'
        "</attributes>"
        f"{_note(staff=1)}</measure></part>"
        '<part id="P2"><measure number="1"><attributes><divisions>1</divisions>'
        "<staves>2</staves></attributes>"
        f"{_note(alter=1, staff=1)}{_note(staff=2)}</measure></part>"
        "</score-partwise>",
        encoding="utf-8",
    )
    physical_staffs = [_staff(1, 40), _staff(2, 130), _staff(3, 220)]
    for staff in physical_staffs:
        staff.barlines = [0, 200]
        staff.measure_count = 1
    layout = PageLayout(
        200,
        320,
        physical_staffs,
        1.0,
    )

    report = audit_semantic_source_symbols(
        xml,
        layout,
        [_accidental(50, 2)],
    )

    assert report.status == "completed"
    assert report.mismatches == ()
    assert report.source_counts["genericAccidental"] == 1
    assert report.output_counts["genericAccidental"] == 1


def test_rhythm_and_relation_symbols_are_positionally_audited(
    tmp_path: Path,
) -> None:
    xml = tmp_path / "rhythm-and-relations.musicxml"
    marked_note = (
        "<note><pitch><step>G</step><octave>4</octave></pitch>"
        "<duration>1</duration><voice>1</voice><type>eighth</type>"
        "<beam number=\"1\">begin</beam><staff>1</staff><notations>"
        '<slur type="start" number="1"/><tied type="start"/>'
        "</notations></note>"
    )
    flagged_note = (
        "<note><pitch><step>A</step><octave>4</octave></pitch>"
        "<duration>1</duration><voice>1</voice><type>eighth</type>"
        "<staff>1</staff></note>"
    )
    hairpin = (
        "<direction><direction-type><wedge type=\"crescendo\" number=\"1\"/>"
        "</direction-type><staff>1</staff></direction>"
    )
    xml.write_text(
        _xml(
            [
                hairpin + marked_note + flagged_note,
                _note(),
            ]
        ),
        encoding="utf-8",
    )
    layout = PageLayout(200, 160, [_staff(1, 40)], 1.0)

    report = audit_semantic_source_symbols(
        xml,
        layout,
        [
            _symbol("beam"),
            _symbol("flag"),
            _symbol("hairpin"),
            _symbol("slur"),
            _symbol("tie"),
        ],
    )

    assert report.status == "completed"
    assert report.mismatches == ()
    for class_name in ("beam", "flag", "hairpin", "slur", "tie"):
        assert report.source_counts[class_name] == 1
        assert report.output_counts[class_name] == 1
