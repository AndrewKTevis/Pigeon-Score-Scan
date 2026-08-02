from __future__ import annotations

"""Deterministic grouped data for the measure-structure calibrator.

The generator works directly on immutable :class:`MeasureIR` objects.  This keeps CPU
training inexpensive while still exercising the exact production feature extractor and
semantic-distance implementation.  Each group represents one source measure and keeps
all correlated candidates in one train/calibration/audit/test partition.
"""

import random
from dataclasses import dataclass, replace
from fractions import Fraction

import numpy as np

from scorescan.measure_calibration import (
    LEGACY_FEATURE_NAMES,
    MeasureCalibrationInput,
    feature_vector,
)
from scorescan.score_ir import DirectionIR, MeasureIR, NoteIR, PitchIR, measure_distance


@dataclass(frozen=True)
class CandidateEvidence:
    score: float
    calibrated_probability: float
    valid: bool


@dataclass(frozen=True)
class MeasureDataset:
    features: np.ndarray
    legacy_features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    decision_groups: tuple[tuple[int, ...], ...]
    scenarios: tuple[str, ...]


_DURATION_NOTATION: dict[Fraction, tuple[str, int, tuple[int, int] | None]] = {
    Fraction(4, 1): ("whole", 0, None),
    Fraction(3, 1): ("half", 1, None),
    Fraction(2, 1): ("half", 0, None),
    Fraction(3, 2): ("quarter", 1, None),
    Fraction(4, 3): ("half", 0, (3, 2)),
    Fraction(1, 1): ("quarter", 0, None),
    Fraction(3, 4): ("eighth", 1, None),
    Fraction(2, 3): ("quarter", 0, (3, 2)),
    Fraction(1, 2): ("eighth", 0, None),
    Fraction(1, 3): ("eighth", 0, (3, 2)),
    Fraction(1, 4): ("16th", 0, None),
}
_ALLOWED_DURATIONS = tuple(sorted(_DURATION_NOTATION, reverse=True))
_TIME_SIGNATURES = ((2, 4), (3, 4), (4, 4), (6, 8), (9, 8))
_STEPS = ("C", "D", "E", "F", "G", "A", "B")
_SCENARIOS = (
    "pickup_boundary",
    "final_boundary",
    "interior_gap",
    "accidental_integrity",
    "chord_integrity",
    "duplicate_integrity",
    "page_score_trap",
    "pitch_trap",
    "mixed_structure",
    "strict_majority",
    "complete_agreement",
)


def _partition(total: Fraction, rng: random.Random) -> tuple[Fraction, ...]:
    units = int(total * 12)
    allowed = tuple(int(value * 12) for value in _ALLOWED_DURATIONS)
    reachable = [False] * (units + 1)
    reachable[0] = True
    for value in range(1, units + 1):
        reachable[value] = any(atom <= value and reachable[value - atom] for atom in allowed)
    if not reachable[units]:
        return (total,)
    result: list[Fraction] = []
    remaining = units
    while remaining:
        choices = [atom for atom in allowed if atom <= remaining and reachable[remaining - atom]]
        weights = [max(1.0, atom / 3.0) for atom in choices]
        selected = rng.choices(choices, weights=weights, k=1)[0]
        result.append(Fraction(selected, 12))
        remaining -= selected
    return tuple(result)


def _pitch(rng: random.Random, *, offset: int = 0) -> PitchIR:
    step_index = max(0, min(6, rng.randint(0, 6) + offset))
    alter = rng.choices((Fraction(0), Fraction(1), Fraction(-1)), (0.82, 0.09, 0.09), k=1)[0]
    return PitchIR(_STEPS[step_index], alter, rng.choice((3, 4, 4, 5, 5, 6)))


def _note(
    rng: random.Random,
    onset: Fraction,
    duration: Fraction,
    *,
    chord: bool = False,
    pitch_offset: int = 0,
) -> NoteIR:
    note_type, dots, tuplet = _DURATION_NOTATION.get(duration, ("quarter", 0, None))
    rest = (not chord) and rng.random() < 0.14
    pitch = None if rest else _pitch(rng, offset=pitch_offset)
    accidental = ""
    if pitch is not None and pitch.alter and rng.random() < 0.78:
        accidental = "sharp" if pitch.alter > 0 else "flat"
    articulations = ()
    if rng.random() < 0.12:
        articulations = (rng.choice(("staccato", "accent", "tenuto")),)
    return NoteIR(
        onset=onset,
        duration=duration,
        voice="1",
        pitch=pitch,
        rest=rest,
        chord=chord,
        grace=False,
        note_type=note_type,
        dots=dots,
        accidental=accidental,
        ties=(),
        slurs=(),
        articulations=articulations,
        ornaments=(),
        tuple_ratio=tuplet,
    )


def _measure_end(measure: MeasureIR) -> Fraction:
    return max(
        (note.onset + note.duration for note in measure.notes if not note.grace),
        default=Fraction(0),
    )


def _build_reference(rng: random.Random, scenario: str) -> tuple[MeasureIR, bool, bool]:
    beats, beat_type = rng.choice(_TIME_SIGNATURES)
    expected = Fraction(beats * 4, beat_type)
    is_first = scenario == "pickup_boundary" or rng.random() < 0.12
    is_last = scenario == "final_boundary" or (not is_first and rng.random() < 0.12)
    actual = expected
    if scenario in {"pickup_boundary", "final_boundary"} or ((is_first or is_last) and rng.random() < 0.55):
        reductions = [value for value in (Fraction(1, 2), Fraction(1), Fraction(3, 2)) if value < expected]
        actual = expected - rng.choice(reductions or [Fraction(1, 2)])
    durations = _partition(actual, rng)
    notes: list[NoteIR] = []
    onset = Fraction(0)
    for duration in durations:
        anchor = _note(rng, onset, duration)
        notes.append(anchor)
        if not anchor.rest and rng.random() < 0.15:
            notes.append(_note(rng, onset, duration, chord=True, pitch_offset=rng.choice((-2, 2))))
        onset += duration
    directions: list[DirectionIR] = []
    if rng.random() < 0.28:
        directions.append(
            DirectionIR(
                onset=Fraction(0),
                placement=rng.choice(("above", "below")),
                kind=rng.choice(("words", "dynamic")),
                value=rng.choice(("Allegro", "dolce", "espressivo", "p", "mf", "f")),
            )
        )
    return (
        MeasureIR(
            divisions=24,
            time_signature=(beats, beat_type),
            key_signature=(rng.randint(-5, 5), "major"),
            clef=(rng.choice(("G", "G", "F")), rng.choice((2, 2, 4)), 0),
            notes=tuple(notes),
            directions=tuple(directions),
            barlines=(("right", "light-heavy", "", ""),) if is_last else (),
        ),
        is_first,
        is_last,
    )


def _replace_note(measure: MeasureIR, index: int, note: NoteIR) -> MeasureIR:
    notes = list(measure.notes)
    notes[index] = note
    return replace(measure, notes=tuple(notes))


def _pitched_indices(measure: MeasureIR) -> list[int]:
    return [index for index, note in enumerate(measure.notes) if note.pitch is not None and not note.rest]


def _anchor_indices(measure: MeasureIR) -> list[int]:
    return [index for index, note in enumerate(measure.notes) if not note.chord and not note.grace]


def mutate(measure: MeasureIR, kind: str, rng: random.Random) -> MeasureIR:
    notes = list(measure.notes)
    anchors = _anchor_indices(measure)
    pitched = _pitched_indices(measure)
    if kind == "pitch" and pitched:
        index = rng.choice(pitched)
        note = notes[index]
        assert note.pitch is not None
        pitch = replace(note.pitch, octave=note.pitch.octave + rng.choice((-1, 1)))
        return _replace_note(measure, index, replace(note, pitch=pitch))
    if kind == "duration" and anchors:
        index = rng.choice(anchors)
        note = notes[index]
        duration = max(Fraction(1, 4), note.duration * rng.choice((Fraction(1, 2), Fraction(2))))
        return _replace_note(measure, index, replace(note, duration=duration))
    if kind == "type_mismatch" and anchors:
        index = rng.choice(anchors)
        note = notes[index]
        other = rng.choice(tuple(value for value in ("whole", "half", "quarter", "eighth", "16th") if value != note.note_type))
        return _replace_note(measure, index, replace(note, note_type=other, dots=0, tuple_ratio=None))
    if kind == "delete_note" and anchors:
        index = rng.choice(anchors)
        onset = notes[index].onset
        notes = [note for note in notes if note.onset != onset]
        return replace(measure, notes=tuple(notes))
    if kind == "onset_shift" and anchors:
        index = rng.choice(anchors)
        onset = notes[index].onset
        shift = rng.choice((Fraction(1, 4), Fraction(-1, 4)))
        notes = [replace(note, onset=max(Fraction(0), note.onset + shift)) if note.onset == onset else note for note in notes]
        return replace(measure, notes=tuple(notes))
    if kind == "accidental_mismatch" and pitched:
        index = rng.choice(pitched)
        note = notes[index]
        assert note.pitch is not None
        accidental = "flat" if note.pitch.alter >= 0 else "sharp"
        return _replace_note(measure, index, replace(note, accidental=accidental))
    if kind == "extreme_alter" and pitched:
        index = rng.choice(pitched)
        note = notes[index]
        assert note.pitch is not None
        return _replace_note(measure, index, replace(note, pitch=replace(note.pitch, alter=Fraction(3)), accidental="double-sharp"))
    if kind == "orphan_chord" and anchors:
        index = anchors[0]
        return _replace_note(measure, index, replace(notes[index], chord=True))
    if kind == "chord_mismatch" and anchors:
        anchor = notes[rng.choice(anchors)]
        chord = _note(rng, anchor.onset, max(Fraction(1, 4), anchor.duration / 2), chord=True, pitch_offset=2)
        notes.append(chord)
        return replace(measure, notes=tuple(notes))
    if kind == "duplicate_event" and notes:
        notes.append(rng.choice(notes))
        return replace(measure, notes=tuple(notes))
    if kind == "rest_pitch" and pitched:
        index = rng.choice(pitched)
        return _replace_note(measure, index, replace(notes[index], rest=True))
    if kind == "voice" and anchors:
        index = rng.choice(anchors)
        return _replace_note(measure, index, replace(notes[index], voice="2"))
    if kind == "unknown_type" and anchors:
        index = rng.choice(anchors)
        return _replace_note(measure, index, replace(notes[index], note_type="unknown"))
    if kind == "duplicate_direction":
        directions = list(measure.directions)
        if not directions:
            directions.append(DirectionIR(Fraction(0), "above", "words", "dolce"))
        directions.append(directions[0])
        return replace(measure, directions=tuple(directions))
    if kind == "attribute_change":
        if rng.random() < 0.5:
            return replace(measure, key_signature=(rng.choice((-6, -3, 3, 6)), "major"))
        return replace(measure, clef=("F" if measure.clef and measure.clef[0] == "G" else "G", 4, 0))
    if kind == "fill_partial" and measure.expected_duration is not None:
        end = _measure_end(measure)
        gap = measure.expected_duration - end
        if gap > 0 and gap in _DURATION_NOTATION:
            note_type, dots, tuplet = _DURATION_NOTATION[gap]
            notes.append(
                NoteIR(end, gap, "1", None, True, False, False, note_type, dots, "", (), (), (), (), tuplet)
            )
            return replace(measure, notes=tuple(notes))
    if kind == "extra_overlap" and anchors:
        anchor = notes[rng.choice(anchors)]
        extra = _note(rng, anchor.onset, anchor.duration, pitch_offset=1)
        notes.append(extra)
        return replace(measure, notes=tuple(notes))
    return mutate(measure, "pitch", rng) if pitched else measure


def _candidate_mutations(scenario: str) -> tuple[str, ...]:
    return {
        "pickup_boundary": ("fill_partial", "pitch", "accidental_mismatch", "duplicate_event"),
        "final_boundary": ("fill_partial", "pitch", "chord_mismatch", "attribute_change"),
        "interior_gap": ("delete_note", "onset_shift", "duration", "type_mismatch"),
        "accidental_integrity": ("accidental_mismatch", "extreme_alter", "pitch", "attribute_change"),
        "chord_integrity": ("orphan_chord", "chord_mismatch", "extra_overlap", "pitch"),
        "duplicate_integrity": ("duplicate_event", "duplicate_direction", "extra_overlap", "voice"),
        "page_score_trap": ("duration", "type_mismatch", "accidental_mismatch", "pitch"),
        "pitch_trap": ("pitch", "pitch", "attribute_change", "accidental_mismatch"),
        "mixed_structure": ("rest_pitch", "unknown_type", "voice", "delete_note", "duration"),
        "strict_majority": ("pitch", "duration", "accidental_mismatch"),
        "complete_agreement": (),
    }[scenario]


def _build_group(seed: int, group_id: int) -> tuple[list[list[float]], list[int], str]:
    rng = random.Random(seed * 1_000_003 + group_id * 97_409)
    scenario = _SCENARIOS[group_id % len(_SCENARIOS)]
    reference, is_first, is_last = _build_reference(rng, scenario)
    hard_scenarios = {
        "pickup_boundary",
        "final_boundary",
        "interior_gap",
        "accidental_integrity",
        "chord_integrity",
        "duplicate_integrity",
        "mixed_structure",
    }
    candidate_count = rng.choice((5, 7)) if scenario in hard_scenarios else rng.randint(3, 7)
    measures: list[MeasureIR] = [reference]
    if scenario == "complete_agreement":
        measures = [reference for _ in range(candidate_count)]
    else:
        positive_copies = 0
        if scenario == "strict_majority":
            positive_copies = candidate_count // 2
        elif rng.random() < 0.20:
            positive_copies = 1
        measures.extend(reference for _ in range(positive_copies))
        mutations = _candidate_mutations(scenario)
        while len(measures) < candidate_count:
            offset = len(measures) - 1
            kind = mutations[offset % len(mutations)] if mutations else "pitch"
            candidate = mutate(reference, kind, rng)
            # Hard groups deliberately contain several different wrong candidates with
            # the same structural failure.  They therefore cannot be solved merely by
            # choosing the semantic medoid or the highest page score.
            if scenario in {"pickup_boundary", "final_boundary"}:
                candidate = mutate(reference, "fill_partial", rng)
                candidate = mutate(
                    candidate,
                    ("pitch", "attribute_change", "accidental_mismatch", "chord_mismatch", "duplicate_direction")[offset % 5],
                    rng,
                )
            elif scenario == "accidental_integrity":
                candidate = mutate(reference, "accidental_mismatch", rng)
                if offset % 2:
                    candidate = mutate(candidate, "pitch", rng)
            elif scenario == "chord_integrity":
                candidate = mutate(reference, ("orphan_chord", "chord_mismatch")[offset % 2], rng)
                if offset >= 2:
                    candidate = mutate(candidate, "pitch", rng)
            elif scenario == "duplicate_integrity":
                candidate = mutate(reference, ("duplicate_event", "extra_overlap", "duplicate_direction")[offset % 3], rng)
                if offset >= 3:
                    candidate = mutate(candidate, "pitch", rng)
            elif scenario == "interior_gap":
                candidate = mutate(reference, ("delete_note", "onset_shift", "duration")[offset % 3], rng)
                if offset >= 3:
                    candidate = mutate(candidate, "pitch", rng)
            if measure_distance(reference, candidate) <= 0.002:
                candidate = mutate(reference, "pitch", rng)
            if measure_distance(reference, candidate) <= 0.002:
                fifths, mode = reference.key_signature or (0, "major")
                replacement = fifths + 1 if fifths < 7 else fifths - 1
                candidate = replace(reference, key_signature=(replacement, mode))
            measures.append(candidate)

    labels = [int(measure_distance(reference, measure) <= 0.002) for measure in measures]
    candidates: list[CandidateEvidence] = []
    for index, label in enumerate(labels):
        score = rng.uniform(910, 1000) if label else rng.uniform(850, 990)
        probability = rng.uniform(0.64, 0.96) if label else rng.uniform(0.50, 0.91)
        valid = True
        if not label and scenario in {"mixed_structure", "interior_gap"} and rng.random() < 0.25:
            valid = False
        candidates.append(CandidateEvidence(score, probability, valid))
    if scenario == "page_score_trap" and len(candidates) > 1:
        trap = next((index for index, label in enumerate(labels) if not label), None)
        if trap is not None:
            candidates[trap] = CandidateEvidence(1008.0, 0.985, True)
    if scenario in {"pickup_boundary", "final_boundary"}:
        # A filled-out but wrong boundary measure can look cleaner to the page model.
        # This is intentionally limited to the boundary scenarios so ordinary high
        # page confidence remains positive evidence rather than becoming inverted.
        trap = next((index for index, label in enumerate(labels) if not label), None)
        if trap is not None:
            candidates[trap] = CandidateEvidence(1003.0, 0.97, True)
    if scenario in {"pickup_boundary", "final_boundary"}:
        for index, measure in enumerate(measures):
            if not labels[index] and _measure_end(measure) == measure.expected_duration:
                candidates[index] = CandidateEvidence(1003.0, 0.97, True)
                break

    distances = np.asarray(
        [[measure_distance(left, right) for right in measures] for left in measures],
        dtype=np.float64,
    )
    mean_distances = distances.mean(axis=1)
    medoid_index = int(np.argmin(mean_distances))
    semantic_cluster = [index for index, value in enumerate(distances[medoid_index]) if value <= 0.075]
    semantic_support = len(semantic_cluster) / len(measures)
    fingerprints: dict[str, int] = {}
    for measure in measures:
        fingerprints[measure.fingerprint] = fingerprints.get(measure.fingerprint, 0) + 1
    exact_support = max(fingerprints.values()) / len(measures)
    template_index = max(range(len(candidates)), key=lambda index: (candidates[index].score, -index))
    missing_ratio = rng.choice((0.0, 0.0, 0.0, 1.0 / (len(measures) + 1)))

    rows: list[list[float]] = []
    for index, measure in enumerate(measures):
        distance = measure_distance(reference, measure)
        alignment = max(0.55, min(1.0, 0.98 - 0.28 * distance + rng.uniform(-0.03, 0.03)))
        if scenario == "page_score_trap" and not labels[index]:
            alignment = max(alignment, 0.97)
        item = MeasureCalibrationInput(
            candidate=candidates[index],
            measure=measure,
            alignment_similarity=alignment,
            exact_support_ratio=exact_support,
            semantic_support_ratio=semantic_support,
            missing_ratio=missing_ratio,
            distance_to_template=float(distances[template_index, index]),
            distance_to_medoid=float(distances[medoid_index, index]),
            mean_peer_distance=float(mean_distances[index]),
            is_first_measure=is_first,
            is_last_measure=is_last,
        )
        rows.append(feature_vector(item))
    return rows, labels, scenario


def build_dataset(seed: int, groups: int) -> MeasureDataset:
    rows: list[list[float]] = []
    labels: list[int] = []
    group_ids: list[int] = []
    decisions: list[tuple[int, ...]] = []
    scenarios: list[str] = []
    for group_id in range(groups):
        group_rows, group_labels, scenario = _build_group(seed, group_id)
        start = len(rows)
        rows.extend(group_rows)
        labels.extend(group_labels)
        group_ids.extend([group_id] * len(group_rows))
        decisions.append(tuple(range(start, len(rows))))
        scenarios.append(scenario)
    features = np.asarray(rows, dtype=np.float64)
    return MeasureDataset(
        features=features,
        legacy_features=features[:, : len(LEGACY_FEATURE_NAMES)],
        labels=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(group_ids, dtype=np.int64),
        decision_groups=tuple(decisions),
        scenarios=tuple(scenarios),
    )
