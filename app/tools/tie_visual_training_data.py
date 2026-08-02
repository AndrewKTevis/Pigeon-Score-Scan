from __future__ import annotations

"""Group-isolated rendered transactions for within-measure tie visual safety."""

import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from fractions import Fraction
from multiprocessing import get_context

import cv2
import numpy as np

from scorescan.local_symbol_image import decode_bounded_png
from scorescan.score_ir import MeasureIR, NoteIR, PitchIR
from scorescan.tie_visual_guard import tie_visual_features
from scorescan.visual_evidence import VisualMeasureEvidence, extract_crop_features
from visual_training_data import render_measure

cv2.setNumThreads(1)


@dataclass(frozen=True)
class TieVisualDataset:
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


def _evidence(
    image: np.ndarray,
    spacing: float,
    staff_top: float,
    staff_bottom: float,
    group: int,
) -> VisualMeasureEvidence:
    features = extract_crop_features(
        image,
        spacing=spacing,
        staff_top=staff_top,
        staff_bottom=staff_bottom,
    )
    return VisualMeasureEvidence(
        page_index=0,
        system_index=0,
        measure_index=group,
        bbox=(0, 0, image.shape[1], image.shape[0]),
        spacing=float(spacing),
        **features,
    )


def _note(onset: Fraction, duration: Fraction, rng: random.Random) -> NoteIR:
    pitch = PitchIR(rng.choice("CDEFGAB"), Fraction(0, 1), rng.randint(3, 5))
    return NoteIR(
        onset=onset,
        duration=duration,
        voice="1",
        pitch=pitch,
        rest=False,
        chord=False,
        grace=False,
        note_type=rng.choice(("quarter", "quarter", "eighth", "16th", "half")),
        dots=1 if rng.random() < 0.10 else 0,
        accidental="",
        ties=(),
        slurs=(),
        articulations=(rng.choice(("staccato", "accent", "tenuto")),)
        if rng.random() < 0.14
        else (),
        ornaments=(),
        tuple_ratio=None,
    )


def _base_measure(rng: random.Random) -> tuple[MeasureIR, int, int]:
    count = rng.randint(4, 10)
    duration = Fraction(4, count)
    notes = [_note(Fraction(index * 4, count), duration, rng) for index in range(count)]
    start = rng.randrange(count - 1)
    stop = start + 1
    notes[stop] = replace(notes[stop], pitch=notes[start].pitch)
    return (
        MeasureIR(
            divisions=24,
            time_signature=(4, 4),
            key_signature=(0, "major"),
            clef=rng.choice((("G", 2, 0), ("F", 4, 0))),
            notes=tuple(notes),
            directions=(),
            barlines=(),
        ),
        start,
        stop,
    )


def _set_tie(measure: MeasureIR, start: int, stop: int) -> MeasureIR:
    notes = list(measure.notes)
    notes[stop] = replace(notes[stop], pitch=notes[start].pitch)
    notes[start] = replace(notes[start], ties=("start",), slurs=())
    notes[stop] = replace(notes[stop], ties=("stop",), slurs=())
    return replace(measure, notes=tuple(notes))


def _set_slur(measure: MeasureIR, start: int, stop: int) -> MeasureIR:
    notes = list(measure.notes)
    notes[start] = replace(notes[start], slurs=(("start", "1"),), ties=())
    notes[stop] = replace(notes[stop], slurs=(("stop", "1"),), ties=())
    return replace(measure, notes=tuple(notes))


def _add_distractor(
    measure: MeasureIR,
    target: tuple[int, int],
    rng: random.Random,
) -> MeasureIR:
    if rng.random() >= 0.52 or len(measure.notes) < 4:
        return measure
    candidates = [
        index
        for index in range(len(measure.notes) - 1)
        if index not in target and index + 1 not in target
    ]
    if not candidates:
        return measure
    start = rng.choice(candidates)
    stop = start + 1
    return (
        _set_tie(measure, start, stop)
        if rng.random() < 0.5
        else _set_slur(measure, start, stop)
    )


def _nearby_arc(
    measure: MeasureIR,
    target: tuple[int, int],
    rng: random.Random,
) -> MeasureIR:
    possible = [
        (left, left + 1)
        for left in range(len(measure.notes) - 1)
        if left not in target and left + 1 not in target
    ]
    if not possible:
        return measure
    start, stop = min(
        possible,
        key=lambda pair: abs(pair[0] - target[0]) + abs(pair[1] - target[1]),
    )
    return (
        _set_tie(measure, start, stop)
        if rng.random() < 0.5
        else _set_slur(measure, start, stop)
    )


def _build_group(args: tuple[int, int, int]) -> tuple[list[list[float]], list[int], list[int], list[str]]:
    seed, generated, group_offset = args
    group_seed = _group_seed(seed, generated)
    rng = random.Random(group_seed)
    base, start, stop = _base_measure(rng)
    base = _add_distractor(base, (start, stop), rng)
    scenario = "nearby_arc" if rng.random() < 0.36 else "no_target"
    negative = _nearby_arc(base, (start, stop), rng) if scenario == "nearby_arc" else base
    positive = _set_tie(base, start, stop)

    render_seed = group_seed ^ 0xD1B54A32D192ED03
    rows: list[list[float]] = []
    labels: list[int] = []
    groups: list[int] = []
    scenarios: list[str] = []
    for local_measure, label, local_scenario in (
        (negative, 0, scenario),
        (positive, 1, "target_present"),
    ):
        image, spacing, staff_top, staff_bottom = render_measure(
            local_measure,
            random.Random(render_seed),
            draw_curves=True,
        )
        evidence = _evidence(image, spacing, staff_top, staff_bottom, generated)
        features = tie_visual_features(evidence, local_measure, start, stop)
        if features is None:
            raise RuntimeError("rendered tie visual feature unexpectedly unavailable")
        rows.append(features)
        labels.append(label)
        groups.append(group_offset + generated)
        scenarios.append(local_scenario)
        decode_bounded_png.cache_clear()
    return rows, labels, groups, scenarios


def build_tie_visual_dataset(
    seed: int,
    groups: int,
    *,
    group_offset: int = 0,
    workers: int = 1,
) -> TieVisualDataset:
    arguments = [
        (int(seed), generated, int(group_offset)) for generated in range(int(groups))
    ]
    if int(workers) > 1:
        with ProcessPoolExecutor(
            max_workers=int(workers),
            mp_context=get_context("spawn"),
        ) as executor:
            built = list(executor.map(_build_group, arguments, chunksize=4))
    else:
        built = [_build_group(item) for item in arguments]
    rows: list[list[float]] = []
    labels: list[int] = []
    group_ids: list[int] = []
    scenarios: list[str] = []
    for local_rows, local_labels, local_groups, local_scenarios in built:
        rows.extend(local_rows)
        labels.extend(local_labels)
        group_ids.extend(local_groups)
        scenarios.extend(local_scenarios)
    return TieVisualDataset(
        features=np.asarray(rows, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(group_ids, dtype=np.int64),
        scenarios=tuple(scenarios),
    )


def build_tie_slur_ambiguity_dataset(
    seed: int,
    groups: int,
    *,
    group_offset: int = 0,
    workers: int = 1,
) -> TieVisualDataset:
    """Hard rejected experiment: same endpoints contain a slur instead of a tie."""
    def one(generated: int):
        group_seed = _group_seed(seed, generated)
        rng = random.Random(group_seed)
        base, start, stop = _base_measure(rng)
        slur = _set_slur(base, start, stop)
        tie = _set_tie(base, start, stop)
        render_seed = group_seed ^ 0x94D049BB133111EB
        local = []
        for measure, label, scenario in (
            (slur, 0, "same_endpoint_slur"),
            (tie, 1, "target_tie"),
        ):
            image, spacing, top, bottom = render_measure(
                measure, random.Random(render_seed), draw_curves=True
            )
            evidence = _evidence(image, spacing, top, bottom, generated)
            feature = tie_visual_features(evidence, measure, start, stop)
            if feature is None:
                raise RuntimeError("ambiguity feature unavailable")
            local.append((feature, label, group_offset + generated, scenario))
            decode_bounded_png.cache_clear()
        return local

    # This audit is intentionally small and serial; it is a release-time rejection
    # witness rather than part of the deployed model's training data.
    built = [one(generated) for generated in range(groups)]
    rows, labels, group_ids, scenarios = [], [], [], []
    for local in built:
        for feature, label, group, scenario in local:
            rows.append(feature)
            labels.append(label)
            group_ids.append(group)
            scenarios.append(scenario)
    return TieVisualDataset(
        np.asarray(rows, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(group_ids, dtype=np.int64),
        tuple(scenarios),
    )
