from __future__ import annotations

"""Deterministic grouped data for family-balanced cross-measure context.

Each group is a complete seven-variant candidate ensemble.  Errors may persist across
three adjacent measures and may be shared by sibling preprocessing variants or by
multiple families.  This is the failure mode the production context prior must handle:
internal continuity from one correlated image treatment is not independent musical
evidence.
"""

import random
from dataclasses import dataclass, replace
from fractions import Fraction

import numpy as np

from scorescan.context_calibration import agreement_profiles
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
SCENARIOS = (
    "single-family-correlated-sequence-error",
    "cross-family-correlated-sequence-error",
    "neighbor-corruption",
    "current-boundary-error",
    "legitimate-transition",
    "context-neutral-internal-error",
)


@dataclass(frozen=True)
class ContextDataset:
    features: np.ndarray
    legacy_features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    decision_groups: tuple[tuple[int, ...], ...]
    scenarios: tuple[str, ...]


def _pitch_from_position(position: int, alter: int = 0) -> PitchIR:
    return PitchIR(
        STEPS[position % 7],
        Fraction(alter, 1),
        1 + position // 7,
    )


def _durations_for_measure(rng: random.Random, expected: Fraction) -> tuple[Fraction, ...]:
    remaining = expected
    result: list[Fraction] = []
    while remaining > 0:
        valid = [duration for duration in DURATIONS if duration <= remaining]
        duration = rng.choice(valid)
        result.append(duration)
        remaining -= duration
    return tuple(result)


def _make_measure(
    rng: random.Random,
    *,
    step_position: int,
    time_signature: tuple[int, int],
    key_signature: tuple[int, str],
    clef: tuple[str, int, int],
    pending_tie: PitchIR | None,
    direction: str | None = None,
    density_bias: int = 0,
) -> tuple[MeasureIR, int, PitchIR | None]:
    expected = Fraction(time_signature[0] * 4, time_signature[1])
    durations = list(_durations_for_measure(rng, expected))
    if density_bias > 0:
        for _ in range(density_bias):
            splittable = [index for index, value in enumerate(durations) if value >= Fraction(1, 2)]
            if not splittable:
                break
            index = rng.choice(splittable)
            value = durations[index] / 2
            durations[index:index + 1] = [value, value]
    elif density_bias < 0 and len(durations) >= 2:
        for _ in range(-density_bias):
            if len(durations) < 2:
                break
            index = rng.randrange(len(durations) - 1)
            durations[index:index + 2] = [durations[index] + durations[index + 1]]

    notes: list[NoteIR] = []
    onset = Fraction(0, 1)
    next_tie: PitchIR | None = None
    for event_index, duration in enumerate(durations):
        ties: tuple[str, ...] = ()
        if pending_tie is not None and event_index == 0:
            pitch = pending_tie
            rest = False
            ties = ("stop",)
            pending_tie = None
        else:
            rest = rng.random() < 0.11
            if rest:
                pitch = None
            else:
                step_position = max(
                    18,
                    min(45, step_position + rng.choice((-2, -1, 0, 0, 1, 2))),
                )
                pitch = _pitch_from_position(
                    step_position,
                    rng.choice((0, 0, 0, 0, -1, 1)),
                )
        if (
            pitch is not None
            and event_index == len(durations) - 1
            and rng.random() < 0.13
        ):
            ties = tuple(sorted(set(ties + ("start",))))
            next_tie = pitch
        notes.append(
            NoteIR(
                onset=onset,
                duration=duration,
                voice="1",
                pitch=pitch,
                rest=rest,
                chord=False,
                grace=False,
                note_type=TYPES.get(duration, "quarter"),
                dots=int(duration in {Fraction(3, 2), Fraction(3, 4)}),
                accidental=(
                    ""
                    if pitch is None
                    else rng.choice(("", "", "", "", "natural", "sharp", "flat"))
                ),
                ties=ties,
                slurs=(),
                articulations=(
                    (rng.choice(("staccato", "accent", "tenuto")),)
                    if rng.random() < 0.09
                    else ()
                ),
                ornaments=(),
                tuple_ratio=None,
            )
        )
        onset += duration
    directions = (
        (DirectionIR(Fraction(0, 1), "above", "words", direction),)
        if direction
        else ()
    )
    return (
        MeasureIR(
            divisions=24,
            time_signature=time_signature,
            key_signature=key_signature,
            clef=clef,
            notes=tuple(notes),
            directions=directions,
            barlines=(),
        ),
        step_position,
        next_tie,
    )


def random_segment(
    rng: random.Random,
    *,
    force_transition: bool = False,
) -> tuple[MeasureIR, MeasureIR, MeasureIR]:
    base_time = rng.choice(((2, 4), (3, 4), (4, 4), (6, 8)))
    base_key = (rng.randint(-4, 4), rng.choice(("major", "major", "minor")))
    base_clef = ("G", 2, 0)
    step_position = rng.randint(23, 36)
    pending_tie: PitchIR | None = None
    result: list[MeasureIR] = []
    current_time = base_time
    current_key = base_key
    current_clef = base_clef
    transition_index = 1 if force_transition or rng.random() < 0.24 else -1
    transition_kind = rng.choice(("time", "key", "clef", "leap", "density", "chromatic"))
    for index in range(3):
        direction = None
        density_bias = 0
        if index == transition_index:
            direction = rng.choice(("a tempo", "dolce", "con fuoco", "rit."))
            if transition_kind == "time":
                current_time = rng.choice(
                    [value for value in ((2, 4), (3, 4), (4, 4), (6, 8)) if value != current_time]
                )
            elif transition_kind == "key":
                fifths, mode = current_key
                current_key = (
                    max(-7, min(7, fifths + rng.choice((-4, -3, 3, 4)))),
                    mode,
                )
            elif transition_kind == "clef":
                current_clef = rng.choice((("F", 4, 0), ("C", 3, 0), ("G", 2, 1)))
            elif transition_kind == "leap":
                step_position = max(18, min(45, step_position + rng.choice((-10, -8, 8, 10))))
            elif transition_kind == "density":
                density_bias = rng.choice((-2, 2))
        measure, step_position, pending_tie = _make_measure(
            rng,
            step_position=step_position,
            time_signature=current_time,
            key_signature=current_key,
            clef=current_clef,
            pending_tie=pending_tie,
            direction=direction,
            density_bias=density_bias,
        )
        if index == transition_index and transition_kind == "chromatic":
            chromatic_notes = [
                replace(
                    note,
                    pitch=(
                        replace(note.pitch, alter=Fraction(rng.choice((-1, 1)), 1))
                        if note.pitch is not None
                        else None
                    ),
                    accidental=(rng.choice(("sharp", "flat")) if note.pitch is not None else ""),
                )
                for note in measure.notes
            ]
            measure = replace(measure, notes=tuple(chromatic_notes))
        result.append(measure)
    return result[0], result[1], result[2]


def _shift_octave(measure: MeasureIR, delta: int) -> MeasureIR:
    return replace(
        measure,
        notes=tuple(
            replace(
                note,
                pitch=(
                    replace(
                        note.pitch,
                        octave=max(1, min(8, note.pitch.octave + delta)),
                    )
                    if note.pitch is not None
                    else None
                ),
            )
            for note in measure.notes
        ),
    )


def _mutate_once(measure: MeasureIR, kind: str, rng: random.Random) -> MeasureIR:
    notes = list(measure.notes)
    regular = [index for index, note in enumerate(notes) if not note.grace and not note.chord]
    if kind == "key":
        fifths, mode = measure.key_signature or (0, "major")
        return replace(
            measure,
            key_signature=(max(-7, min(7, fifths + rng.choice((-4, -2, 2, 4)))), mode),
        )
    if kind == "clef":
        return replace(measure, clef=rng.choice((("F", 4, 0), ("C", 3, 0), ("G", 2, 1))))
    if kind == "time":
        return replace(measure, time_signature=rng.choice(((2, 4), (3, 4), (4, 4), (6, 8))))
    if kind == "direction":
        return replace(
            measure,
            directions=measure.directions
            + (DirectionIR(Fraction(0), "above", "words", rng.choice(("Allegro", "rit.", "dolce"))),),
        )
    if not regular:
        return measure
    index = rng.choice(regular)
    note = notes[index]
    if kind == "octave" and note.pitch is not None:
        notes[index] = replace(
            note,
            pitch=replace(
                note.pitch,
                octave=max(1, min(8, note.pitch.octave + rng.choice((-1, 1)))),
            ),
        )
    elif kind == "boundary":
        edge_index = regular[0] if rng.random() < 0.5 else regular[-1]
        edge = notes[edge_index]
        if edge.pitch is not None:
            notes[edge_index] = replace(
                edge,
                pitch=replace(
                    edge.pitch,
                    octave=max(1, min(8, edge.pitch.octave + rng.choice((-2, -1, 1, 2)))),
                ),
            )
    elif kind == "step" and note.pitch is not None:
        notes[index] = replace(
            note,
            pitch=replace(
                note.pitch,
                step=rng.choice([value for value in STEPS if value != note.pitch.step]),
            ),
        )
    elif kind == "duration":
        notes[index] = replace(
            note,
            duration=max(
                Fraction(1, 8),
                note.duration * rng.choice((Fraction(1, 2), Fraction(2, 1))),
            ),
        )
    elif kind == "delete" and len(regular) > 1:
        notes.pop(index)
    elif kind == "extra":
        notes.insert(index, replace(note, onset=note.onset + Fraction(1, 8), ties=()))
    elif kind == "rest":
        notes[index] = replace(
            note,
            rest=not note.rest,
            pitch=None if not note.rest else PitchIR("C", Fraction(0), 4),
            ties=(),
        )
    elif kind == "tie":
        notes[index] = replace(
            note,
            ties=() if note.ties else (rng.choice(("start", "stop")),),
        )
    return replace(measure, notes=tuple(notes))


def mutate(measure: MeasureIR, rng: random.Random, kinds: tuple[str, ...]) -> MeasureIR:
    result = measure
    for kind in kinds:
        result = _mutate_once(result, kind, rng)
    return result


def _corrupt_segment(
    segment: tuple[MeasureIR, MeasureIR, MeasureIR],
    scenario: str,
    rng: random.Random,
) -> tuple[MeasureIR, MeasureIR, MeasureIR]:
    previous, current, following = segment
    if scenario in {
        "single-family-correlated-sequence-error",
        "cross-family-correlated-sequence-error",
    }:
        if rng.random() < 0.62:
            delta = rng.choice((-1, 1))
            return (
                _shift_octave(previous, delta),
                _shift_octave(current, delta),
                _shift_octave(following, delta),
            )
        kind = rng.choice(("key", "clef", "time"))
        return (
            mutate(previous, rng, (kind,)),
            mutate(current, rng, (kind,)),
            mutate(following, rng, (kind,)),
        )
    if scenario == "neighbor-corruption":
        return (
            mutate(previous, rng, (rng.choice(("octave", "boundary", "key", "tie")),)),
            current,
            mutate(following, rng, (rng.choice(("octave", "boundary", "key", "tie")),)),
        )
    if scenario == "current-boundary-error":
        return previous, mutate(current, rng, ("boundary", rng.choice(("tie", "duration")))), following
    if scenario == "legitimate-transition":
        # A false candidate smooths out an intentional transition by copying attributes
        # and boundary pitch from the preceding measure.
        smoothed = replace(
            current,
            time_signature=previous.time_signature,
            key_signature=previous.key_signature,
            clef=previous.clef,
        )
        previous_pitches = [note.pitch for note in previous.notes if note.pitch is not None]
        current_notes = list(smoothed.notes)
        current_pitched = [index for index, note in enumerate(current_notes) if note.pitch is not None]
        if previous_pitches and current_pitched:
            current_notes[current_pitched[0]] = replace(
                current_notes[current_pitched[0]],
                pitch=previous_pitches[-1],
            )
            smoothed = replace(smoothed, notes=tuple(current_notes))
        return previous, smoothed, following
    # Context-neutral internal corruption: do not create an artificial boundary signal.
    return previous, mutate(current, rng, (rng.choice(("step", "rest", "delete", "extra")),)), following



def corrupt_segment(
    reference: tuple[MeasureIR, MeasureIR, MeasureIR],
    scenario: str,
    rng: random.Random,
) -> tuple[MeasureIR, MeasureIR, MeasureIR]:
    """Public deterministic segment corruption used by grouped training tools."""
    return _corrupt_segment(reference, scenario, rng)


def _build_group(
    rng: random.Random,
    scenario: str,
) -> tuple[
    MeasureIR,
    list[MeasureIR],
    list[MeasureIR],
    list[MeasureIR],
]:
    reference_segment = random_segment(
        rng,
        force_transition=scenario == "legitimate-transition",
    )
    if scenario == "cross-family-correlated-sequence-error":
        shared_wrong = set(rng.sample(FAMILY_NAMES, k=2))
    elif scenario == "single-family-correlated-sequence-error":
        shared_wrong = {rng.choice(FAMILY_NAMES)}
    else:
        shared_wrong = set()

    correct_count = rng.choice((1, 2, 2, 3))
    eligible_correct = [family for family in FAMILY_NAMES if family not in shared_wrong]
    correct_families = set(rng.sample(eligible_correct, k=min(correct_count, len(eligible_correct))))
    if "baseline" not in shared_wrong and rng.random() < 0.65:
        correct_families.add("baseline")
    if not correct_families:
        correct_families.add(rng.choice(eligible_correct))

    shared_segment = (
        _corrupt_segment(reference_segment, scenario, rng)
        if shared_wrong
        else None
    )
    family_segments: dict[str, tuple[MeasureIR, MeasureIR, MeasureIR]] = {}
    for family in FAMILY_NAMES:
        if family in correct_families:
            family_segments[family] = reference_segment
        elif family in shared_wrong and shared_segment is not None:
            family_segments[family] = shared_segment
        else:
            family_segments[family] = _corrupt_segment(reference_segment, scenario, rng)

    previous: list[MeasureIR] = []
    current: list[MeasureIR] = []
    following: list[MeasureIR] = []
    for _variant, family in VARIANTS:
        left, center, right = family_segments[family]
        if measure_distance(reference_segment[1], center) <= 0.002:
            center = replace(center, divisions=rng.choice((24, 48, 96)))
            if scenario == "neighbor-corruption" and rng.random() < 0.45:
                left = mutate(left, rng, (rng.choice(("boundary", "key", "tie")),))
        elif rng.random() < 0.12:
            center = mutate(center, rng, (rng.choice(("step", "duration", "tie")),))
        previous.append(left)
        current.append(center)
        following.append(right)

    if not any(
        measure_distance(reference_segment[1], candidate) <= 0.002
        for candidate in current
    ):
        previous[0], current[0], following[0] = reference_segment
    return reference_segment[1], previous, current, following


def build_dataset(seed: int, groups: int) -> ContextDataset:
    rng = random.Random(seed)
    features: list[list[float]] = []
    legacy_features: list[list[float]] = []
    labels: list[int] = []
    group_ids: list[int] = []
    decision_groups: list[tuple[int, ...]] = []
    scenarios: list[str] = []
    scenario_block: list[str] = []
    for group in range(groups):
        scenario = SCENARIOS[group % len(SCENARIOS)]
        # Shuffle scenario order while retaining exact class counts across runs.
        if group % len(SCENARIOS) == 0:
            scenario_block = list(SCENARIOS)
            rng.shuffle(scenario_block)
        scenario = scenario_block[group % len(SCENARIOS)]
        reference, previous, current, following = _build_group(rng, scenario)
        profiles = agreement_profiles(previous, current, following, FAMILIES)
        indices: list[int] = []
        for candidate, profile in zip(current, profiles, strict=True):
            indices.append(len(labels))
            features.append(profile.feature_vector())
            legacy_features.append(profile.legacy_feature_vector())
            labels.append(int(measure_distance(reference, candidate) <= 0.002))
            group_ids.append(group)
        decision_groups.append(tuple(indices))
        scenarios.append(scenario)
    return ContextDataset(
        features=np.asarray(features, dtype=np.float64),
        legacy_features=np.asarray(legacy_features, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int32),
        groups=np.asarray(group_ids, dtype=np.int32),
        decision_groups=tuple(decision_groups),
        scenarios=tuple(scenarios),
    )
