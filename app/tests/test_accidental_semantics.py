from dataclasses import replace
from fractions import Fraction

from scorescan.accidental_semantics import (
    accidental_regression,
    accidental_sequence_issues,
    chromatic_change_indices,
    key_step_alters,
)
from scorescan.score_ir import MeasureIR, NoteIR, PitchIR


def _note(step: str, alter: int, octave: int = 4, accidental: str = "", *, onset: int = 0, ties=()):
    return NoteIR(
        Fraction(onset, 1), Fraction(1, 1), "1", PitchIR(step, Fraction(alter, 1), octave),
        False, False, False, "quarter", 0, accidental, tuple(ties), (), (), (), None,
    )


def _measure(*notes: NoteIR, fifths: int = 0) -> MeasureIR:
    return MeasureIR(24, (4, 4), (fifths, "major"), ("G", 2, 0), tuple(notes), (), ())


def test_key_step_alters_cover_sharps_and_flats() -> None:
    assert key_step_alters((3, "major"))["F"] == 1
    assert key_step_alters((3, "major"))["C"] == 1
    assert key_step_alters((3, "major"))["G"] == 1
    assert key_step_alters((-2, "major"))["B"] == -1
    assert key_step_alters((-2, "major"))["E"] == -1


def test_accidental_state_accepts_explicit_and_carried_alteration() -> None:
    measure = _measure(
        _note("F", 1, accidental="sharp", onset=0),
        _note("F", 1, onset=1),
        _note("F", 0, accidental="natural", onset=2),
        _note("F", 0, onset=3),
    )
    assert accidental_sequence_issues(measure) == ()


def test_accidental_state_detects_explicit_and_implicit_mismatch() -> None:
    explicit = _measure(_note("C", -1, accidental="sharp"))
    assert [item.code for item in accidental_sequence_issues(explicit)] == ["explicit_alter_mismatch"]
    implicit = _measure(_note("F", 0), fifths=1)
    assert [item.code for item in accidental_sequence_issues(implicit)] == ["implicit_alter_state_mismatch"]


def test_tie_stop_can_seed_chromatic_state_without_printed_accidental() -> None:
    measure = _measure(
        _note("C", 1, onset=0, ties=("stop",)),
        _note("C", 1, onset=1),
    )
    assert accidental_sequence_issues(measure) == ()


def test_differential_regression_only_rejects_new_issues() -> None:
    before = _measure(_note("D", 1))
    unchanged_bad = _measure(_note("D", 1), _note("E", 0, onset=1))
    regression = accidental_regression(before, unchanged_bad)
    assert regression.safe

    after = _measure(_note("D", 1), _note("E", 1, onset=1))
    regression = accidental_regression(unchanged_bad, after)
    assert not regression.safe
    assert regression.introduced[0].event_index == 1


def test_chromatic_change_indices_ignore_diatonic_motion() -> None:
    before = _measure(_note("C", 0), _note("D", 0, onset=1))
    after = replace(
        before,
        notes=(
            replace(before.notes[0], pitch=PitchIR("C", Fraction(1), 4), accidental="sharp"),
            replace(before.notes[1], pitch=PitchIR("E", Fraction(0), 4)),
        ),
    )
    assert chromatic_change_indices(before, after) == (0,)
