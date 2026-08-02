from __future__ import annotations

"""Grouped programmatic data for the one-event presence visual transaction gate."""

import base64
import random
from dataclasses import dataclass, replace
from fractions import Fraction

import cv2
import numpy as np

from scorescan.event_presence_visual_guard import (
    EVENT_PRESENCE_VISUAL_FEATURE_NAMES,
    event_presence_visual_features,
)
from scorescan.local_symbol_image import event_position
from scorescan.score_ir import MeasureIR, NoteIR, PitchIR
from scorescan.visual_evidence import (
    SYMBOL_GUARD_HEIGHT,
    SYMBOL_GUARD_WIDTH,
    VisualMeasureEvidence,
)
from synthetic_event_render import DURATION_BY_TYPE, PITCHES, draw_note, draw_rest, event_note

_UNIT_BY_TYPE = {name: int(duration * 4) for name, duration in DURATION_BY_TYPE.items()}
_TYPE_BY_UNIT = {value: key for key, value in _UNIT_BY_TYPE.items() if key != "whole"}
_SUPPORTED_UNITS = (1, 2, 4, 8)
_SCENARIOS = (
    "clean",
    "close_neighbours",
    "dense_neighbours",
    "staff_residue",
    "stem_fragment",
    "beam_fragment",
    "compact_noise",
    "notehead_noise",
    "low_contrast",
    "partial_target",
    "absence_residue",
    "overlapping_neighbour",
)


@dataclass(frozen=True)
class EventPresenceVisualDataset:
    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    scenarios: tuple[str, ...]
    operations: tuple[str, ...]
    event_kinds: tuple[str, ...]


def _partition(total: int, rng: random.Random) -> list[int]:
    if total <= 0:
        return []
    result: list[int] = []
    remaining = total
    while remaining > 0:
        choices = [value for value in _SUPPORTED_UNITS if value <= remaining]
        # Prefer moderate event counts without excluding short values.
        weights = [5 if value in {2, 4} else 2 for value in choices]
        value = rng.choices(choices, weights=weights, k=1)[0]
        result.append(value)
        remaining -= value
    return result


def _insertion_measures(rng: random.Random) -> tuple[MeasureIR, MeasureIR, int]:
    for _ in range(100):
        target_units = rng.choice(_SUPPORTED_UNITS)
        remaining = 16 - target_units
        left_total = rng.randint(1, remaining - 1)
        right_total = remaining - left_total
        left = _partition(left_total, rng)
        right = _partition(right_total, rng)
        if left and right:
            units = left + [target_units] + right
            target_index = len(left)
            break
    else:
        raise RuntimeError("failed to build insertion partition")
    full = _measure_from_units(units, rng)
    original_target = full.notes[target_index]
    # The surrounding score keeps its natural 4:1 note/rest distribution, while the
    # event under review is balanced by kind.  This prevents rare rest insertions from
    # being statistically dominated without changing the deployment prior.
    target_kind = rng.choice(("note", "rest"))
    target = event_note(
        target_kind,
        str(original_target.note_type),
        original_target.pitch or rng.choice(PITCHES),
        original_target.onset,
        int(original_target.dots),
    )
    full = _replace_notes(
        full,
        full.notes[:target_index] + (target,) + full.notes[target_index + 1 :],
    )
    shifted_suffix = tuple(
        replace(note, onset=note.onset - target.duration)
        for note in full.notes[target_index + 1 :]
    )
    reduced_notes = full.notes[:target_index] + shifted_suffix
    reduced = _replace_notes(full, reduced_notes)
    return reduced, full, target_index


def _deletion_measures(rng: random.Random) -> tuple[MeasureIR, MeasureIR, int]:
    correct_units = _partition(16, rng)
    while len(correct_units) < 2:
        correct_units = _partition(16, rng)
    correct = _measure_from_units(correct_units, rng)
    target_units = rng.choice(_SUPPORTED_UNITS)
    gap_index = rng.randint(1, len(correct.notes) - 1)
    target_kind = rng.choice(("note", "rest"))
    target = event_note(
        target_kind,
        _TYPE_BY_UNIT[target_units],
        rng.choice(PITCHES),
        correct.notes[gap_index].onset,
        0,
    )
    shifted_suffix = tuple(
        replace(note, onset=note.onset + target.duration)
        for note in correct.notes[gap_index:]
    )
    overfull_notes = correct.notes[:gap_index] + (target,) + shifted_suffix
    overfull = _replace_notes(correct, overfull_notes)
    return overfull, correct, gap_index


def _measure_from_units(units: list[int], rng: random.Random) -> MeasureIR:
    notes: list[NoteIR] = []
    onset_units = 0
    for value in units:
        kind = rng.choices(("note", "rest"), weights=(4, 1), k=1)[0]
        notes.append(
            event_note(
                kind,
                _TYPE_BY_UNIT[value],
                rng.choice(PITCHES),
                Fraction(onset_units, 4),
                0,
            )
        )
        onset_units += value
    return MeasureIR(
        divisions=4,
        time_signature=(4, 4),
        key_signature=(0, "major"),
        clef=("G", 2, 0),
        notes=tuple(notes),
        directions=(),
        barlines=(),
    )


def _replace_notes(measure: MeasureIR, notes: tuple[NoteIR, ...]) -> MeasureIR:
    return MeasureIR(
        divisions=measure.divisions,
        time_signature=measure.time_signature,
        key_signature=measure.key_signature,
        clef=measure.clef,
        notes=notes,
        directions=measure.directions,
        barlines=measure.barlines,
    )


def _draw_event(
    image: np.ndarray,
    measure: MeasureIR,
    note: NoteIR,
    rng: random.Random,
    intensity: int,
    *,
    jitter: bool = True,
) -> tuple[int, int]:
    x_ratio, y_ratio = event_position(measure, note)
    x = int(round(x_ratio * (SYMBOL_GUARD_WIDTH - 1))) + (rng.randint(-2, 2) if jitter else 0)
    y = int(round(y_ratio * (SYMBOL_GUARD_HEIGHT - 1))) + (rng.randint(-2, 2) if jitter else 0)
    if note.rest:
        draw_rest(image, x, y, note.note_type, rng, intensity)
    else:
        draw_note(image, x, y, note.note_type, rng, intensity)
    return x, y


def _base_background(scenario: str, rng: random.Random) -> np.ndarray:
    image = np.zeros((SYMBOL_GUARD_HEIGHT, SYMBOL_GUARD_WIDTH), dtype=np.uint8)
    for _ in range(rng.randint(3, 14)):
        cv2.circle(
            image,
            (rng.randrange(SYMBOL_GUARD_WIDTH), rng.randrange(SYMBOL_GUARD_HEIGHT)),
            rng.choice((1, 1, 1, 2)),
            rng.randint(35, 120),
            -1,
            cv2.LINE_AA,
        )
    if scenario == "staff_residue":
        for y in (31, 39, 47, 55, 63):
            cv2.line(
                image,
                (rng.randint(0, 40), y + rng.randint(-1, 1)),
                (rng.randint(205, 255), y),
                rng.randint(35, 90),
                1,
                cv2.LINE_AA,
            )
    elif scenario == "stem_fragment":
        x = rng.randint(72, 188)
        cv2.line(
            image,
            (x, rng.randint(18, 36)),
            (x, rng.randint(62, 86)),
            rng.randint(85, 180),
            rng.choice((1, 2)),
            cv2.LINE_AA,
        )
    elif scenario == "beam_fragment":
        x = rng.randint(62, 166)
        y = rng.randint(20, 70)
        cv2.line(
            image,
            (x, y),
            (x + rng.randint(24, 62), y + rng.randint(-5, 5)),
            rng.randint(90, 190),
            rng.randint(2, 4),
            cv2.LINE_AA,
        )
    elif scenario == "compact_noise":
        for _ in range(rng.randint(4, 10)):
            cv2.circle(
                image,
                (rng.randint(70, 190), rng.randint(20, 76)),
                rng.randint(1, 3),
                rng.randint(65, 175),
                -1,
                cv2.LINE_AA,
            )
    elif scenario == "notehead_noise":
        for _ in range(rng.randint(1, 4)):
            cv2.ellipse(
                image,
                (rng.randint(70, 190), rng.randint(25, 72)),
                (rng.randint(4, 7), rng.randint(2, 4)),
                rng.randint(-20, 20),
                0,
                360,
                rng.randint(70, 180),
                -1,
                cv2.LINE_AA,
            )
    return image


def _render(
    source: MeasureIR,
    scenario: str,
    rng: random.Random,
    *,
    target_measure: MeasureIR,
    target_event: NoteIR,
    target_present: bool,
) -> np.ndarray:
    image = _base_background(scenario, rng)
    intensity = rng.randint(155, 255)
    for note in source.notes:
        _draw_event(image, source, note, rng, intensity)

    target_x_ratio, target_y_ratio = event_position(target_measure, target_event)
    target_x = int(round(target_x_ratio * (SYMBOL_GUARD_WIDTH - 1)))
    target_y = int(round(target_y_ratio * (SYMBOL_GUARD_HEIGHT - 1)))
    if scenario == "absence_residue" and not target_present:
        for _ in range(rng.randint(1, 3)):
            x1 = target_x + rng.randint(-10, 8)
            y1 = target_y + rng.randint(-14, 14)
            cv2.line(
                image,
                (x1, y1),
                (x1 + rng.randint(-3, 8), y1 + rng.randint(-7, 8)),
                rng.randint(45, 120),
                1,
                cv2.LINE_AA,
            )
    elif scenario == "overlapping_neighbour":
        neighbour_kind = "rest" if target_event.rest is False else "note"
        neighbour = event_note(
            neighbour_kind,
            target_event.note_type,
            target_event.pitch or rng.choice(PITCHES),
            target_event.onset,
            0,
        )
        x = target_x + rng.choice((-1, 1)) * rng.randint(16, 25)
        y = target_y + rng.randint(-5, 5)
        if neighbour.rest:
            draw_rest(image, x, y, neighbour.note_type, rng, max(80, intensity - 35))
        else:
            draw_note(image, x, y, neighbour.note_type, rng, max(80, intensity - 35))
    elif scenario in {"close_neighbours", "dense_neighbours"}:
        count = 1 if scenario == "close_neighbours" else rng.randint(2, 4)
        for offset in range(count):
            x = target_x + rng.choice((-1, 1)) * (18 + offset * 12)
            y = target_y + rng.randint(-8, 8)
            if rng.random() < 0.65:
                draw_note(
                    image,
                    x,
                    y,
                    rng.choice(tuple(_TYPE_BY_UNIT.values())),
                    rng,
                    max(70, intensity - 45),
                )
            else:
                draw_rest(
                    image,
                    x,
                    y,
                    rng.choice(tuple(_TYPE_BY_UNIT.values())),
                    rng,
                    max(70, intensity - 45),
                )

    if scenario == "partial_target" and target_present:
        cv2.rectangle(
            image,
            (target_x + rng.randint(-2, 6), target_y - 25),
            (target_x + 26, target_y + 25),
            0,
            -1,
        )
    sigma = rng.uniform(0.0, 0.9 if scenario != "low_contrast" else 1.45)
    if sigma > 0.08:
        image = cv2.GaussianBlur(image, (0, 0), sigma)
    if scenario == "low_contrast":
        image = np.clip(image.astype(np.float64) * rng.uniform(0.45, 0.72), 0, 255).astype(
            np.uint8
        )
    if rng.random() < 0.65:
        noise = np.random.default_rng(rng.getrandbits(64)).normal(
            0.0, rng.uniform(1.0, 7.5), image.shape
        )
        image = np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return image


def _evidence(image: np.ndarray) -> VisualMeasureEvidence:
    ok, payload = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise RuntimeError("failed to encode event-presence evidence")
    encoded = base64.b64encode(payload.tobytes()).decode("ascii")
    return VisualMeasureEvidence(
        page_index=0,
        system_index=0,
        measure_index=0,
        bbox=(0, 0, SYMBOL_GUARD_WIDTH, SYMBOL_GUARD_HEIGHT),
        spacing=10.0,
        ink_density=float(np.mean(image > 0)),
        nonstaff_ink_density=float(np.mean(image > 0)),
        component_density=0.0,
        notehead_proxy=0.0,
        open_notehead_proxy=0.0,
        stem_proxy=0.0,
        beam_proxy=0.0,
        onset_proxy=0.0,
        compact_mark_proxy=0.0,
        accidental_proxy=0.0,
        above_ink_density=0.0,
        below_ink_density=0.0,
        x_ink_profile=(0.0,) * 8,
        staff_ink_profile=(0.0,) * 9,
        symbol_guard_image=encoded,
    )


def build_event_presence_visual_dataset(
    *, seed: int, groups: int
) -> EventPresenceVisualDataset:
    rows: list[tuple[float, ...]] = []
    labels: list[int] = []
    group_ids: list[int] = []
    scenarios: list[str] = []
    operations: list[str] = []
    event_kinds: list[str] = []
    root = random.Random(seed)
    for group in range(groups):
        rng = random.Random(root.getrandbits(64))
        scenario = rng.choice(_SCENARIOS)
        insert_before, insert_after, insert_index = _insertion_measures(rng)
        delete_before, delete_after, delete_index = _deletion_measures(rng)
        transactions = (
            ("insert", insert_before, insert_after, insert_index),
            ("delete", delete_before, delete_after, delete_index),
        )
        for operation, before, after, event_index in transactions:
            event = after.notes[event_index] if operation == "insert" else before.notes[event_index]
            for source_present in (False, True):
                if operation == "insert":
                    source = insert_after if source_present else insert_before
                else:
                    source = delete_before if source_present else delete_after
                label = 1 if source_present else 0
                evidence = _evidence(
                    _render(
                        source,
                        scenario,
                        rng,
                        target_measure=after if operation == "insert" else before,
                        target_event=event,
                        target_present=source_present,
                    )
                )
                features = event_presence_visual_features(
                    evidence, before, after, operation, event_index
                )
                if features is None:
                    raise RuntimeError("event-presence training feature extraction failed")
                rows.append(features)
                labels.append(label)
                group_ids.append(group)
                scenarios.append(scenario)
                operations.append(operation)
                event_kinds.append("rest" if event.rest else "note")
    matrix = np.asarray(rows, dtype=np.float64)
    expected_shape = (groups * 4, len(EVENT_PRESENCE_VISUAL_FEATURE_NAMES))
    if matrix.shape != expected_shape:
        raise RuntimeError(
            f"unexpected event-presence visual dataset shape: {matrix.shape} != {expected_shape}"
        )
    return EventPresenceVisualDataset(
        features=matrix,
        labels=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(group_ids, dtype=np.int64),
        scenarios=tuple(scenarios),
        operations=tuple(operations),
        event_kinds=tuple(event_kinds),
    )
