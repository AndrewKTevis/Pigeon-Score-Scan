from __future__ import annotations

from fractions import Fraction

from lxml import etree

from scorescan.score_ir import (
    MeasureIR,
    NoteIR,
    PartIR,
    PitchIR,
    ScoreIR,
    PRODUCTION_SCAN_POLICY,
    audit_production_score,
    audit_score,
    score_from_tree,
)


def _tree(xml: str) -> etree._ElementTree:
    return etree.ElementTree(etree.fromstring(xml.encode("utf-8")))


def test_score_ir_preserves_parts_staves_voices_and_direction_ownership() -> None:
    tree = _tree(
        """
        <score-partwise version="4.0">
          <part-list>
            <score-part id="P1">
              <part-name>Piano</part-name>
              <part-abbreviation>Pno.</part-abbreviation>
            </score-part>
            <score-part id="P2"><part-name>Violin</part-name></score-part>
          </part-list>
          <part id="P1">
            <measure number="1">
              <attributes>
                <divisions>4</divisions>
                <key><fifths>0</fifths><mode>major</mode></key>
                <time><beats>4</beats><beat-type>4</beat-type></time>
                <staves>2</staves>
                <clef number="1"><sign>G</sign><line>2</line></clef>
                <clef number="2"><sign>F</sign><line>4</line></clef>
              </attributes>
              <note>
                <pitch><step>C</step><octave>5</octave></pitch>
                <duration>4</duration><voice>1</voice><type>quarter</type><staff>1</staff>
              </note>
              <backup><duration>4</duration></backup>
              <note>
                <pitch><step>E</step><octave>4</octave></pitch>
                <duration>4</duration><voice>2</voice><type>quarter</type><staff>1</staff>
              </note>
              <backup><duration>4</duration></backup>
              <note>
                <pitch><step>C</step><octave>3</octave></pitch>
                <duration>4</duration><voice>3</voice><type>quarter</type><staff>2</staff>
              </note>
              <direction placement="below">
                <direction-type><wedge type="crescendo" number="2"/></direction-type>
                <voice>3</voice><staff>2</staff>
              </direction>
            </measure>
          </part>
          <part id="P2">
            <measure number="1">
              <attributes>
                <divisions>4</divisions>
                <key><fifths>0</fifths><mode>major</mode></key>
                <time><beats>4</beats><beat-type>4</beat-type></time>
                <transpose><diatonic>0</diatonic><chromatic>0</chromatic></transpose>
                <clef><sign>G</sign><line>2</line></clef>
              </attributes>
              <note>
                <pitch><step>G</step><octave>4</octave></pitch>
                <duration>4</duration><voice>1</voice><type>quarter</type>
              </note>
            </measure>
          </part>
        </score-partwise>
        """
    )

    score = score_from_tree(tree)

    assert len(score.parts) == 2
    assert score.measures is score.parts[0].measures
    assert score.parts[0].name == "Piano"
    assert score.parts[0].abbreviation == "Pno."
    assert score.parts[0].staff_count == 2
    assert score.parts[1].name == "Violin"
    assert score.parts[1].transposition == (0, 0, 0)

    piano_measure = score.parts[0].measures[0]
    assert piano_measure.number == "1"
    assert piano_measure.staff_clefs == ((1, "G", 2, 0), (2, "F", 4, 0))
    assert {(note.staff, note.voice, note.onset) for note in piano_measure.notes} == {
        (1, "1", Fraction(0)),
        (1, "2", Fraction(0)),
        (2, "3", Fraction(0)),
    }
    assert piano_measure.directions[0].kind == "wedge"
    assert piano_measure.directions[0].staff == 2
    assert piano_measure.directions[0].voice == "3"

    assert "multiple_voices" in {issue.code for issue in audit_score(score)}
    assert "multiple_voices" not in {issue.code for issue in audit_production_score(score)}


def test_production_audit_keeps_single_staff_independent_voices_out_of_scope() -> None:
    notes = (
        NoteIR(
            Fraction(0),
            Fraction(1),
            "1",
            PitchIR("C", Fraction(0), 4),
            False,
            False,
            False,
            "quarter",
            0,
            "",
            (),
            (),
            (),
            (),
            None,
        ),
        NoteIR(
            Fraction(0),
            Fraction(1),
            "2",
            PitchIR("E", Fraction(0), 4),
            False,
            False,
            False,
            "quarter",
            0,
            "",
            (),
            (),
            (),
            (),
            None,
        ),
    )
    measure = MeasureIR(4, (4, 4), (0, "major"), ("G", 2, 0), notes, (), (), number="1")
    part = PartIR("P1", "Violin", "Vln.", (measure,), 1)
    score = ScoreIR((measure,), (part,))

    issues = audit_production_score(score)

    assert [
        issue.code for issue in issues
    ].count("non_keyboard_voice_limit_exceeded") == 1
    assert issues[0].part_id == "P1"


def test_production_audit_matches_four_staff_eight_voice_keyboard_scope() -> None:
    def keyboard_score(voice_count: int, staff_count: int = 4) -> ScoreIR:
        notes = tuple(
            NoteIR(
                Fraction(0),
                Fraction(1),
                str(voice),
                PitchIR("C", Fraction(0), 4),
                False,
                False,
                False,
                "quarter",
                0,
                "",
                (),
                (),
                (),
                (),
                None,
                staff=1,
            )
            for voice in range(1, voice_count + 1)
        )
        measure = MeasureIR(
            4,
            (4, 4),
            (0, "major"),
            ("G", 2, 0),
            notes,
            (),
            (),
            staff_count=staff_count,
        )
        part = PartIR("P1", "Piano", "Pno.", (measure,), staff_count)
        return ScoreIR((measure,), (part,))

    assert PRODUCTION_SCAN_POLICY.max_keyboard_staves == 4
    assert PRODUCTION_SCAN_POLICY.max_keyboard_voices_per_staff == 8
    assert not {
        "keyboard_staff_limit_exceeded",
        "keyboard_voice_limit_exceeded",
    } & {issue.code for issue in audit_production_score(keyboard_score(8))}
    assert "keyboard_voice_limit_exceeded" in {
        issue.code for issue in audit_production_score(keyboard_score(9))
    }
    assert "keyboard_staff_limit_exceeded" in {
        issue.code
        for issue in audit_production_score(
            keyboard_score(1, staff_count=5)
        )
    }


def test_production_audit_allows_independent_ensemble_timelines() -> None:
    note = NoteIR(
        Fraction(0),
        Fraction(1),
        "1",
        PitchIR("C", Fraction(0), 4),
        False,
        False,
        False,
        "quarter",
        0,
        "",
        (),
        (),
        (),
        (),
        None,
    )
    measure = MeasureIR(
        4,
        (4, 4),
        (0, "major"),
        ("G", 2, 0),
        (note,),
        (),
        (),
    )
    score = ScoreIR(
        (measure,),
        (
            PartIR("P1", "Flute", "Fl.", (measure,), 1),
            PartIR("P2", "Violin", "Vln.", (measure, measure), 1),
        ),
    )

    assert "part_measure_count_mismatch" not in {
        issue.code for issue in audit_production_score(score)
    }


def test_production_audit_counts_concurrent_not_reused_voice_labels() -> None:
    notes = tuple(
        NoteIR(
            Fraction(index),
            Fraction(1),
            str(index + 1),
            PitchIR("C", Fraction(0), 4),
            False,
            False,
            False,
            "quarter",
            0,
            "",
            (),
            (),
            (),
            (),
            None,
        )
        for index in range(4)
    )
    measure = MeasureIR(
        4,
        (4, 4),
        (0, "major"),
        ("G", 2, 0),
        notes,
        (),
        (),
    )
    score = ScoreIR(
        (measure,),
        (PartIR("P1", "Flute", "Fl.", (measure,), 1),),
    )

    assert measure.voice_count_for_staff(1) == 4
    assert measure.maximum_simultaneous_voices_for_staff(1) == 1
    assert "non_keyboard_voice_limit_exceeded" not in {
        issue.code for issue in audit_production_score(score)
    }


def test_production_audit_rejects_event_outside_declared_staff_range() -> None:
    note = NoteIR(
        Fraction(0),
        Fraction(1),
        "1",
        PitchIR("C", Fraction(0), 4),
        False,
        False,
        False,
        "quarter",
        0,
        "",
        (),
        (),
        (),
        (),
        None,
        staff=2,
    )
    measure = MeasureIR(4, (4, 4), (0, "major"), ("G", 2, 0), (note,), (), (), staff_count=1)
    part = PartIR("P1", "Flute", "Fl.", (measure,), 1)

    issues = audit_production_score(ScoreIR((measure,), (part,)))

    invalid = [issue for issue in issues if issue.code == "invalid_note_staff"]
    assert len(invalid) == 1
    assert invalid[0].staff == 2
    assert invalid[0].severity == "error"
