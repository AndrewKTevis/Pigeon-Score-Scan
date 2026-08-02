from __future__ import annotations

"""Programmatic grouped data for the note-versus-rest visual transaction gate."""

import base64
import random
from dataclasses import dataclass
from fractions import Fraction

import cv2
import numpy as np

from scorescan.event_kind_visual_guard import (
    EVENT_KIND_VISUAL_FEATURE_NAMES,
    event_kind_visual_features,
)
from scorescan.local_symbol_image import event_position
from scorescan.score_ir import MeasureIR, NoteIR, PitchIR
from scorescan.visual_evidence import (
    SYMBOL_GUARD_HEIGHT,
    SYMBOL_GUARD_WIDTH,
    VisualMeasureEvidence,
)

from synthetic_event_render import (
    DURATION_BY_TYPE as _DURATION_BY_TYPE,
    PITCHES as _PITCHES,
    draw_note as _draw_note,
    draw_rest as _draw_rest,
    event_note as _note,
    single_event_measure as _measure,
)

_SCENARIOS = (
    "clean",
    "nearby_opposite_symbol",
    "stem_fragment",
    "notehead_noise",
    "compact_noise",
    "staff_residue",
    "beam_fragment",
    "low_contrast",
    "partial_symbol",
    "dense_neighbours",
)


@dataclass(frozen=True)
class EventKindVisualDataset:
    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    scenarios: tuple[str, ...]
    target_kinds: tuple[str, ...]


def _draw_symbol(image: np.ndarray, measure: MeasureIR, note: NoteIR, rng: random.Random, intensity: int) -> tuple[int, int]:
    x_ratio, y_ratio = event_position(measure, note)
    x = int(round(x_ratio * (SYMBOL_GUARD_WIDTH - 1))) + rng.randint(-3, 3)
    y = int(round(y_ratio * (SYMBOL_GUARD_HEIGHT - 1))) + rng.randint(-2, 2)
    if note.rest:
        _draw_rest(image, x, y, note.note_type, rng, intensity)
    else:
        _draw_note(image, x, y, note.note_type, rng, intensity)
    if note.dots:
        cv2.circle(image, (x + 11, y), 2, intensity, -1, cv2.LINE_AA)
    return x, y


def _add_background(
    image: np.ndarray,
    scenario: str,
    source_measure: MeasureIR,
    source_note: NoteIR,
    rng: random.Random,
    intensity: int,
) -> None:
    for _ in range(rng.randint(4, 15)):
        x = rng.randrange(SYMBOL_GUARD_WIDTH)
        y = rng.randrange(SYMBOL_GUARD_HEIGHT)
        radius = rng.choice((1, 1, 1, 2))
        cv2.circle(image, (x, y), radius, rng.randint(50, max(51, intensity)), -1, cv2.LINE_AA)
    if scenario == "staff_residue":
        for y in (31, 39, 47, 55, 63):
            start = rng.randint(0, 55)
            cv2.line(image, (start, y + rng.randint(-1, 1)), (rng.randint(180, 255), y), rng.randint(35, 95), 1)
    elif scenario == "stem_fragment":
        x = rng.randint(90, 170)
        cv2.line(image, (x, rng.randint(20, 35)), (x, rng.randint(62, 82)), rng.randint(90, intensity), rng.choice((1, 2)), cv2.LINE_AA)
    elif scenario == "notehead_noise":
        for _ in range(rng.randint(1, 3)):
            cv2.ellipse(image, (rng.randint(80, 180), rng.randint(28, 70)), (rng.randint(4, 7), 3), rng.randint(-20, 20), 0, 360, rng.randint(80, intensity), -1, cv2.LINE_AA)
    elif scenario == "compact_noise":
        for _ in range(rng.randint(3, 7)):
            cv2.circle(image, (rng.randint(85, 175), rng.randint(25, 72)), rng.randint(1, 3), rng.randint(80, intensity), -1, cv2.LINE_AA)
    elif scenario == "beam_fragment":
        x = rng.randint(80, 145)
        y = rng.randint(20, 72)
        cv2.line(image, (x, y), (x + rng.randint(25, 55), y + rng.randint(-4, 4)), rng.randint(100, intensity), rng.randint(2, 4), cv2.LINE_AA)
    elif scenario in {"nearby_opposite_symbol", "dense_neighbours"}:
        opposite = "note" if source_note.rest else "rest"
        count = 1 if scenario == "nearby_opposite_symbol" else rng.randint(2, 3)
        for offset_index in range(count):
            neighbour = _note(
                opposite,
                source_note.note_type,
                source_note.pitch or rng.choice(_PITCHES),
                source_note.onset,
                source_note.dots,
            )
            neighbour_measure = MeasureIR(
                divisions=source_measure.divisions,
                time_signature=source_measure.time_signature,
                key_signature=source_measure.key_signature,
                clef=source_measure.clef,
                notes=(neighbour,),
                directions=(),
                barlines=(),
            )
            x_ratio, y_ratio = event_position(neighbour_measure, neighbour)
            x = int(round(x_ratio * (SYMBOL_GUARD_WIDTH - 1))) + rng.choice((-1, 1)) * (28 + offset_index * 16)
            y = int(round(y_ratio * (SYMBOL_GUARD_HEIGHT - 1))) + rng.randint(-4, 4)
            if neighbour.rest:
                _draw_rest(image, x, y, neighbour.note_type, rng, max(70, intensity - 45))
            else:
                _draw_note(image, x, y, neighbour.note_type, rng, max(70, intensity - 45))


def _render(source_measure: MeasureIR, scenario: str, rng: random.Random) -> np.ndarray:
    image = np.zeros((SYMBOL_GUARD_HEIGHT, SYMBOL_GUARD_WIDTH), dtype=np.uint8)
    source_note = source_measure.notes[0]
    intensity = rng.randint(150, 255)
    _draw_symbol(image, source_measure, source_note, rng, intensity)
    _add_background(image, scenario, source_measure, source_note, rng, intensity)
    if scenario == "partial_symbol":
        x_ratio, y_ratio = event_position(source_measure, source_note)
        x = int(round(x_ratio * (SYMBOL_GUARD_WIDTH - 1)))
        y = int(round(y_ratio * (SYMBOL_GUARD_HEIGHT - 1)))
        cv2.rectangle(image, (x + rng.randint(-2, 5), y - 23), (x + 24, y + 23), 0, -1)
    sigma = rng.uniform(0.0, 0.9 if scenario != "low_contrast" else 1.4)
    if sigma > 0.08:
        image = cv2.GaussianBlur(image, (0, 0), sigma)
    if scenario == "low_contrast":
        image = np.clip(image.astype(np.float64) * rng.uniform(0.48, 0.72), 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        noise = np.random.default_rng(rng.getrandbits(64)).normal(0.0, rng.uniform(1.0, 8.0), image.shape)
        image = np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return image


def _evidence(image: np.ndarray) -> VisualMeasureEvidence:
    ok, payload = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise RuntimeError("failed to encode event-kind evidence")
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


def build_event_kind_visual_dataset(*, seed: int, groups: int) -> EventKindVisualDataset:
    rows: list[tuple[float, ...]] = []
    labels: list[int] = []
    group_ids: list[int] = []
    scenarios: list[str] = []
    target_kinds: list[str] = []
    root = random.Random(seed)
    for group in range(groups):
        rng = random.Random(root.getrandbits(64))
        note_type = rng.choice(tuple(_DURATION_BY_TYPE))
        pitch = rng.choice(_PITCHES)
        onset = rng.choice((Fraction(1, 1), Fraction(3, 2), Fraction(2, 1), Fraction(5, 2), Fraction(3, 1)))
        dots = 1 if note_type in {"half", "quarter", "eighth"} and rng.random() < 0.25 else 0
        source_kind = rng.choice(("note", "rest"))
        opposite_kind = "rest" if source_kind == "note" else "note"
        scenario = rng.choice(_SCENARIOS)
        source = _measure(source_kind, note_type, pitch, onset, dots)
        opposite = _measure(opposite_kind, note_type, pitch, onset, dots)
        evidence = _evidence(_render(source, scenario, rng))
        transactions = (
            (opposite, source, 1),
            (source, opposite, 0),
        )
        for before, after, label in transactions:
            features = event_kind_visual_features(evidence, before, after, 0)
            if features is None:
                raise RuntimeError("event-kind training feature extraction failed")
            rows.append(features)
            labels.append(label)
            group_ids.append(group)
            scenarios.append(scenario)
            target_kinds.append("rest" if after.notes[0].rest else "note")
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape != (groups * 2, len(EVENT_KIND_VISUAL_FEATURE_NAMES)):
        raise RuntimeError(f"unexpected event-kind visual dataset shape: {matrix.shape}")
    return EventKindVisualDataset(
        features=matrix,
        labels=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(group_ids, dtype=np.int64),
        scenarios=tuple(scenarios),
        target_kinds=tuple(target_kinds),
    )
