from __future__ import annotations

"""Deterministic grouped event-candidate training data.

The generator creates complete seven-variant ensembles with the same correlation
families used in production. Errors may be shared by siblings or by multiple families,
which prevents a model from learning that raw candidate count equals independent
support.
"""

import random
from dataclasses import dataclass, replace
from fractions import Fraction

import numpy as np

from scorescan.event_calibration import agreement_profiles
from scorescan.score_ir import DirectionIR, MeasureIR, NoteIR, PitchIR, measure_distance

VARIANTS = (
    ("primary", "baseline"),
    ("flat", "restoration"),
    ("deblock", "restoration"),
    ("otsu", "binary"),
    ("adaptive", "binary"),
    ("upscale", "scale"),
    ("staffnorm", "scale"),
)
FAMILIES = tuple(family for _variant, family in VARIANTS)
FAMILY_NAMES = ("baseline", "restoration", "binary", "scale")
STEPS = "CDEFGAB"
DURATIONS = (
    Fraction(2, 1),
    Fraction(3, 2),
    Fraction(1, 1),
    Fraction(3, 4),
    Fraction(1, 2),
    Fraction(1, 4),
)
TYPES = {
    Fraction(2, 1): "half",
    Fraction(3, 2): "quarter",
    Fraction(1, 1): "quarter",
    Fraction(3, 4): "eighth",
    Fraction(1, 2): "eighth",
    Fraction(1, 4): "16th",
}
MUTATIONS = (
    "octave",
    "step",
    "alter",
    "duration",
    "onset",
    "delete",
    "extra",
    "rest",
    "accidental",
    "articulation",
    "tie",
    "slur",
    "ornament",
    "voice",
    "chord",
)


@dataclass(frozen=True)
class EventDataset:
    features: np.ndarray
    legacy_features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    decision_groups: tuple[tuple[int, ...], ...]
    scenarios: tuple[str, ...]


def random_measure(rng: random.Random) -> MeasureIR:
    time_signature = rng.choice(((2, 4), (3, 4), (4, 4), (6, 8)))
    expected = Fraction(time_signature[0] * 4, time_signature[1])
    durations: list[Fraction] = []
    remaining = expected
    while remaining > 0:
        duration = rng.choice([value for value in DURATIONS if value <= remaining])
        durations.append(duration)
        remaining -= duration

    notes: list[NoteIR] = []
    onset = Fraction(0, 1)
    octave = rng.choice((3, 4, 4, 5))
    step_index = rng.randrange(7)
    for duration in durations:
        is_rest = rng.random() < 0.13
        if is_rest:
            pitch = None
        else:
            step_index = max(0, min(6, step_index + rng.choice((-2, -1, 0, 0, 1, 2))))
            if rng.random() < 0.08:
                octave = max(2, min(7, octave + rng.choice((-1, 1))))
            pitch = PitchIR(
                STEPS[step_index],
                Fraction(rng.choice((0, 0, 0, 0, -1, 1)), 1),
                octave,
            )
        dots = 1 if duration in {Fraction(3, 2), Fraction(3, 4)} else 0
        accidental = rng.choice(("", "", "", "", "sharp", "flat", "natural")) if pitch else ""
        ties = ("start",) if pitch and rng.random() < 0.04 else ()
        slurs = ("1:start",) if pitch and rng.random() < 0.04 else ()
        articulations = (rng.choice(("staccato", "accent", "tenuto")),) if rng.random() < 0.12 else ()
        ornaments = (rng.choice(("trill-mark", "turn", "mordent")),) if rng.random() < 0.04 else ()
        notes.append(
            NoteIR(
                onset,
                duration,
                "1",
                pitch,
                is_rest,
                False,
                False,
                TYPES[duration],
                dots,
                accidental,
                ties,
                slurs,
                articulations,
                ornaments,
                None,
            )
        )
        if pitch and rng.random() < 0.08:
            chord_pitch = PitchIR(STEPS[(step_index + 2) % 7], pitch.alter, octave)
            notes.append(
                NoteIR(
                    onset,
                    duration,
                    "1",
                    chord_pitch,
                    False,
                    True,
                    False,
                    TYPES[duration],
                    dots,
                    "",
                    (),
                    (),
                    (),
                    (),
                    None,
                )
            )
        onset += duration

    directions = ()
    if rng.random() < 0.15:
        directions = (
            DirectionIR(
                Fraction(0, 1),
                "above",
                "words",
                rng.choice(("Allegro", "dolce", "rit.", "a tempo")),
            ),
        )
    return MeasureIR(
        24,
        time_signature,
        (rng.randint(-4, 4), "major"),
        (rng.choice(("G", "F", "C")), rng.choice((2, 3, 4)), 0),
        tuple(notes),
        directions,
        (),
    )


def _mutate_once(measure: MeasureIR, kind: str, rng: random.Random) -> MeasureIR:
    notes = list(measure.notes)
    indices = [index for index, note in enumerate(notes) if not note.grace]
    if not indices:
        return measure
    index = rng.choice(indices)
    note = notes[index]
    if kind == "octave" and note.pitch:
        notes[index] = replace(
            note,
            pitch=replace(
                note.pitch,
                octave=max(1, min(8, note.pitch.octave + rng.choice((-1, 1)))),
            ),
        )
    elif kind == "step" and note.pitch:
        notes[index] = replace(
            note,
            pitch=replace(note.pitch, step=rng.choice([value for value in STEPS if value != note.pitch.step])),
        )
    elif kind == "alter" and note.pitch:
        values = (Fraction(-2), Fraction(-1), Fraction(0), Fraction(1), Fraction(2))
        notes[index] = replace(
            note,
            pitch=replace(note.pitch, alter=rng.choice([value for value in values if value != note.pitch.alter])),
        )
    elif kind == "duration":
        notes[index] = replace(
            note,
            duration=max(Fraction(1, 16), note.duration * rng.choice((Fraction(1, 2), Fraction(2, 1)))),
        )
    elif kind == "onset":
        notes[index] = replace(
            note,
            onset=max(
                Fraction(0, 1),
                note.onset + rng.choice((Fraction(-1, 4), Fraction(1, 4), Fraction(1, 8))),
            ),
        )
    elif kind == "delete" and len(indices) > 1:
        notes.pop(index)
    elif kind == "extra":
        notes.insert(index, replace(note, onset=note.onset + Fraction(1, 8), chord=False))
    elif kind == "rest":
        notes[index] = replace(
            note,
            rest=not note.rest,
            pitch=None if not note.rest else PitchIR("C", Fraction(0, 1), 4),
        )
    elif kind == "accidental":
        notes[index] = replace(
            note,
            accidental=rng.choice(("sharp", "flat", "natural", "double-sharp", "")),
        )
    elif kind == "articulation":
        notes[index] = replace(note, articulations=(rng.choice(("staccato", "accent", "tenuto")),))
    elif kind == "tie":
        notes[index] = replace(note, ties=() if note.ties else ("start",))
    elif kind == "slur":
        notes[index] = replace(note, slurs=() if note.slurs else ("1:start",))
    elif kind == "ornament":
        notes[index] = replace(note, ornaments=() if note.ornaments else ("trill-mark",))
    elif kind == "voice":
        notes[index] = replace(note, voice="2")
    elif kind == "chord":
        notes[index] = replace(note, chord=not note.chord)
    return replace(measure, notes=tuple(notes))



def mutate_once(measure: MeasureIR, kind: str, rng: random.Random) -> MeasureIR:
    """Public deterministic event mutation used by grouped training tools."""
    return _mutate_once(measure, kind, rng)


def mutate(measure: MeasureIR, rng: random.Random, severity: int) -> MeasureIR:
    result = measure
    for _ in range(severity):
        before = result
        for _attempt in range(8):
            result = _mutate_once(result, rng.choice(MUTATIONS), rng)
            if measure_distance(before, result) > 0.0:
                break
    return result


def _ensemble(rng: random.Random) -> tuple[MeasureIR, list[MeasureIR], str]:
    reference = random_measure(rng)
    correct_families = set(
        rng.sample(FAMILY_NAMES, k=rng.choice((1, 1, 2, 2, 3)))
    )
    if rng.random() < 0.70:
        correct_families.add("baseline")
    wrong_families = [family for family in FAMILY_NAMES if family not in correct_families]
    shared_count = min(len(wrong_families), rng.choice((0, 1, 1, 2)))
    shared_families = set(rng.sample(wrong_families, k=shared_count)) if shared_count else set()
    shared_corruption = mutate(reference, rng, rng.choice((1, 1, 2)))

    family_base: dict[str, MeasureIR] = {}
    for family in FAMILY_NAMES:
        if family in correct_families:
            family_base[family] = reference
        elif family in shared_families:
            family_base[family] = shared_corruption
        else:
            family_base[family] = mutate(reference, rng, rng.choice((1, 1, 2, 2, 3)))

    candidates: list[MeasureIR] = []
    for _variant, family in VARIANTS:
        candidate = family_base[family]
        if candidate is reference:
            candidate = (
                mutate(reference, rng, 1)
                if rng.random() < 0.12
                else replace(reference, divisions=rng.choice((24, 48, 96)))
            )
        elif rng.random() < 0.18:
            candidate = mutate(candidate, rng, 1)
        candidates.append(candidate)
    if not any(measure_distance(reference, candidate) <= 0.002 for candidate in candidates):
        candidates[0] = reference

    if len(shared_families) >= 2:
        scenario = "cross-family-correlated-error"
    elif len(shared_families) == 1:
        scenario = "single-family-correlated-error"
    else:
        scenario = "independent-errors"
    return reference, candidates, scenario


def build_dataset(seed: int, groups: int) -> EventDataset:
    rng = random.Random(seed)
    features: list[list[float]] = []
    legacy_features: list[list[float]] = []
    labels: list[int] = []
    group_ids: list[int] = []
    decisions: list[tuple[int, ...]] = []
    scenarios: list[str] = []
    for group in range(groups):
        reference, candidates, scenario = _ensemble(rng)
        profiles = agreement_profiles(candidates, FAMILIES)
        indices: list[int] = []
        for candidate, profile in zip(candidates, profiles, strict=True):
            indices.append(len(labels))
            features.append(profile.feature_vector())
            legacy_features.append(profile.legacy_feature_vector())
            labels.append(int(measure_distance(reference, candidate) <= 0.002))
            group_ids.append(group)
        decisions.append(tuple(indices))
        scenarios.append(scenario)
    return EventDataset(
        features=np.asarray(features, dtype=np.float64),
        legacy_features=np.asarray(legacy_features, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int32),
        groups=np.asarray(group_ids, dtype=np.int32),
        decision_groups=tuple(decisions),
        scenarios=tuple(scenarios),
    )
