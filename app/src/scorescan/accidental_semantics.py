from __future__ import annotations

"""Deterministic accidental-state validation for narrow pitch transactions.

MusicXML stores both sounding alteration (``<alter>``) and the printed accidental
surface.  A local pitch patch must not introduce an alteration which contradicts the
key signature, an earlier accidental in the same measure, or its own explicit symbol.
The validator is differential: an existing OMR defect does not block an unrelated
repair, but any newly introduced accidental-state defect vetoes the transaction.
"""

from dataclasses import dataclass
from fractions import Fraction

from .score_ir import MeasureIR, NoteIR

_SHARP_ORDER = "FCGDAEB"
_FLAT_ORDER = "BEADGCF"
_ACCIDENTAL_ALTERS: dict[str, Fraction] = {
    "sharp": Fraction(1, 1),
    "flat": Fraction(-1, 1),
    "natural": Fraction(0, 1),
    "double-sharp": Fraction(2, 1),
    "sharp-sharp": Fraction(2, 1),
    "double-flat": Fraction(-2, 1),
    "flat-flat": Fraction(-2, 1),
}
_ACCIDENTAL_ALIASES = {
    "double sharp": "double-sharp",
    "double flat": "double-flat",
    "flat flat": "flat-flat",
    "sharp sharp": "sharp-sharp",
}


@dataclass(frozen=True, order=True)
class AccidentalIssue:
    event_index: int
    code: str
    position: tuple[str, int]


@dataclass(frozen=True)
class AccidentalRegression:
    introduced: tuple[AccidentalIssue, ...]
    resolved: tuple[AccidentalIssue, ...]

    @property
    def safe(self) -> bool:
        return not self.introduced


def normalise_accidental(value: str) -> str:
    token = " ".join((value or "").strip().casefold().replace("_", "-").split())
    return _ACCIDENTAL_ALIASES.get(token, token)


def accidental_alter(value: str) -> Fraction | None:
    token = normalise_accidental(value)
    return _ACCIDENTAL_ALTERS.get(token) if token else None


def key_step_alters(key_signature: tuple[int, str] | None) -> dict[str, Fraction]:
    fifths = int(key_signature[0]) if key_signature else 0
    fifths = max(-7, min(7, fifths))
    result = {step: Fraction(0, 1) for step in "CDEFGAB"}
    if fifths > 0:
        for step in _SHARP_ORDER[:fifths]:
            result[step] = Fraction(1, 1)
    elif fifths < 0:
        for step in _FLAT_ORDER[: -fifths]:
            result[step] = Fraction(-1, 1)
    return result


def _position(note: NoteIR) -> tuple[str, int] | None:
    if note.pitch is None or note.rest:
        return None
    return note.pitch.step.upper(), int(note.pitch.octave)


def accidental_sequence_issues(measure: MeasureIR) -> tuple[AccidentalIssue, ...]:
    """Return state inconsistencies in score order without mutating the measure."""
    key = key_step_alters(measure.key_signature)
    state: dict[tuple[str, int], Fraction] = {}
    issues: list[AccidentalIssue] = []
    for event_index, note in enumerate(measure.notes):
        position = _position(note)
        if position is None or note.grace:
            continue
        assert note.pitch is not None
        explicit = normalise_accidental(note.accidental)
        if explicit:
            explicit_alter = accidental_alter(explicit)
            if explicit_alter is None:
                issues.append(AccidentalIssue(event_index, "unsupported_accidental", position))
                # Unknown printed symbols must remain review-only and must not alter the
                # deterministic state used to assess following notes.
                continue
            if note.pitch.alter != explicit_alter:
                issues.append(AccidentalIssue(event_index, "explicit_alter_mismatch", position))
            state[position] = note.pitch.alter
            continue

        # A tie continuation may legally carry a chromatic pitch over the barline
        # without reprinting the accidental.  It seeds this measure's state so later
        # same-position notes are assessed consistently.
        if any(value.strip().casefold() in {"stop", "continue"} for value in note.ties):
            state[position] = note.pitch.alter
            continue
        expected = state.get(position, key.get(position[0], Fraction(0, 1)))
        if note.pitch.alter != expected:
            issues.append(AccidentalIssue(event_index, "implicit_alter_state_mismatch", position))
    return tuple(issues)


def accidental_regression(before: MeasureIR, after: MeasureIR) -> AccidentalRegression:
    before_issues = set(accidental_sequence_issues(before))
    after_issues = set(accidental_sequence_issues(after))
    return AccidentalRegression(
        introduced=tuple(sorted(after_issues - before_issues)),
        resolved=tuple(sorted(before_issues - after_issues)),
    )


def chromatic_change_indices(before: MeasureIR, after: MeasureIR) -> tuple[int, ...]:
    """Return same-staff-position events whose chromatic/display semantics changed."""
    if len(before.notes) != len(after.notes):
        return ()
    changed: list[int] = []
    for index, (left, right) in enumerate(zip(before.notes, after.notes, strict=True)):
        if left.pitch is None or right.pitch is None or left.rest or right.rest:
            continue
        if (
            left.pitch.step.upper() == right.pitch.step.upper()
            and int(left.pitch.octave) == int(right.pitch.octave)
            and (
                left.pitch.alter != right.pitch.alter
                or normalise_accidental(left.accidental) != normalise_accidental(right.accidental)
            )
        ):
            changed.append(index)
    return tuple(changed)
