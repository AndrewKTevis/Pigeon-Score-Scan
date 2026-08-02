from __future__ import annotations

"""Balanced rendered event data for the rhythm symbol compatibility guard.

Each group contains two truths with the same pair of event signatures: a reference
measure and one meter-preserving pairwise corruption.  Both measures are rendered and
both directions are labelled, so every candidate signature occurs equally as a
positive and a negative.  This prevents the CPU model from learning duration priors
instead of source-image compatibility.
"""

import random
from dataclasses import dataclass, replace
from fractions import Fraction

import numpy as np

from event_training_data import DURATIONS, TYPES, random_measure
from scorescan.rhythm_symbol_guard import build_rhythm_symbol_transaction
from scorescan.score_ir import MeasureIR, NoteIR
from visual_training_data import _evidence, render_measure



@dataclass(frozen=True)
class RenderedRhythmSymbolDataset:
    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    scenarios: tuple[str, ...]


def _group_seed(seed: int, group: int) -> int:
    value = (int(seed) + 0x9E3779B97F4A7C15 * (int(group) + 1)) & ((1 << 64) - 1)
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return value ^ (value >> 31)


def _dots(duration: Fraction) -> int:
    return int(duration in {Fraction(3, 2), Fraction(3, 4)})


def _beam_level(note: NoteIR) -> int:
    return {"eighth": 1, "16th": 2, "32nd": 3, "64th": 4, "128th": 5}.get(
        note.note_type.strip().casefold(), 0
    )


def _open(note: NoteIR) -> bool:
    return bool(
        not note.rest
        and note.pitch is not None
        and note.note_type.strip().casefold() in {"whole", "half"}
    )


def _with_durations(measure: MeasureIR, durations: tuple[Fraction, ...]) -> MeasureIR:
    onset = Fraction(0, 1)
    notes: list[NoteIR] = []
    for note, duration in zip(measure.notes, durations, strict=True):
        notes.append(
            replace(
                note,
                onset=onset,
                duration=duration,
                note_type=TYPES[duration],
                dots=_dots(duration),
            )
        )
        onset += duration
    return replace(measure, notes=tuple(notes))


def _scenario(before: MeasureIR, after: MeasureIR, changed: tuple[int, ...]) -> str:
    beam = opened = dots = rest = 0
    for index in changed:
        left = before.notes[index]
        right = after.notes[index]
        beam += int(_beam_level(left) != _beam_level(right))
        opened += int(_open(left) != _open(right))
        dots += int(int(left.dots) != int(right.dots))
        rest += int(left.rest or right.rest)
    parts = []
    if beam:
        parts.append("beam")
    if opened:
        parts.append("open")
    if dots:
        parts.append("dot")
    if rest:
        parts.append("rest")
    return "+".join(parts) or "other"


def introduce_meter_preserving_error(
    measure: MeasureIR,
    rng: random.Random,
) -> tuple[MeasureIR, tuple[int, ...], str] | None:
    if (
        len(measure.notes) < 2
        or len(measure.notes) > 18
        or any(note.chord or note.grace or note.tuple_ratio is not None for note in measure.notes)
        or measure.expected_duration is None
        or sum((note.duration for note in measure.notes), Fraction(0, 1)) != measure.expected_duration
    ):
        return None
    durations = tuple(note.duration for note in measure.notes)
    alternatives_by_total: dict[Fraction, list[tuple[Fraction, Fraction]]] = {}
    for left in DURATIONS:
        for right in DURATIONS:
            alternatives_by_total.setdefault(left + right, []).append((left, right))
    options: list[tuple[int, tuple[Fraction, Fraction]]] = []
    for index in range(len(durations) - 1):
        original = (durations[index], durations[index + 1])
        for alternative in alternatives_by_total.get(original[0] + original[1], ()):
            if alternative == original:
                continue
            proposed = list(durations)
            proposed[index : index + 2] = alternative
            wrong = _with_durations(measure, tuple(proposed))
            changed = (index, index + 1)
            if _scenario(measure, wrong, changed) == "other":
                continue
            options.append((index, alternative))
    if not options:
        return None
    index, alternative = rng.choice(options)
    proposed = list(durations)
    proposed[index : index + 2] = alternative
    wrong = _with_durations(measure, tuple(proposed))
    changed = (index, index + 1)
    return wrong, changed, _scenario(measure, wrong, changed)


def _append_direction(
    rows: list[list[float]],
    labels: list[int],
    groups: list[int],
    scenarios: list[str],
    *,
    evidence,
    truth: MeasureIR,
    alternative: MeasureIR,
    changed: tuple[int, ...],
    group: int,
    scenario: str,
    truth_name: str,
) -> None:
    positive = build_rhythm_symbol_transaction(
        evidence, truth, alternative, changed
    )
    if positive is None:
        raise RuntimeError("rendered rhythm symbol transaction unexpectedly unavailable")
    rows.append(positive.feature_vector())
    labels.append(1)
    groups.append(group)
    scenarios.append(f"{scenario}:{truth_name}:compatible")
    rows.append(positive.reversed().feature_vector())
    labels.append(0)
    groups.append(group)
    scenarios.append(f"{scenario}:{truth_name}:incompatible")


def build_rendered_rhythm_symbol_dataset(
    seed: int,
    groups: int,
    *,
    group_offset: int = 0,
) -> RenderedRhythmSymbolDataset:
    rows: list[list[float]] = []
    labels: list[int] = []
    group_ids: list[int] = []
    scenarios: list[str] = []
    for generated in range(groups):
        rng = random.Random(_group_seed(seed, generated))
        mutation = None
        reference = None
        for _attempt in range(80):
            candidate = random_measure(rng)
            mutation = introduce_meter_preserving_error(candidate, rng)
            if mutation is not None:
                reference = candidate
                break
        if reference is None or mutation is None:
            raise RuntimeError(
                f"could not generate rhythm symbol group {generated}/{groups} within 80 attempts"
            )
        wrong, changed, scenario = mutation
        reference_image, spacing, staff_top, staff_bottom = render_measure(reference, rng)
        wrong_image, wrong_spacing, wrong_top, wrong_bottom = render_measure(wrong, rng)
        reference_evidence = _evidence(
            reference_image, spacing, staff_top, staff_bottom, generated * 2
        )
        wrong_evidence = _evidence(
            wrong_image, wrong_spacing, wrong_top, wrong_bottom, generated * 2 + 1
        )
        group = group_offset + generated
        _append_direction(
            rows,
            labels,
            group_ids,
            scenarios,
            evidence=reference_evidence,
            truth=reference,
            alternative=wrong,
            changed=changed,
            group=group,
            scenario=scenario,
            truth_name="reference",
        )
        _append_direction(
            rows,
            labels,
            group_ids,
            scenarios,
            evidence=wrong_evidence,
            truth=wrong,
            alternative=reference,
            changed=changed,
            group=group,
            scenario=scenario,
            truth_name="corruption",
        )
    return RenderedRhythmSymbolDataset(
        features=np.asarray(rows, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(group_ids, dtype=np.int64),
        scenarios=tuple(scenarios),
    )
