from __future__ import annotations

"""Group-isolated rendered source crops for the accent-addition visual safety gate."""

import base64
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from fractions import Fraction
from multiprocessing import get_context

import cv2
import numpy as np

from scorescan.accent_visual_guard import accent_visual_features
from scorescan.local_symbol_image import decode_bounded_png, decode_symbol_guard_image, event_position
from scorescan.score_ir import MeasureIR, NoteIR, PitchIR
from scorescan.visual_evidence import VisualMeasureEvidence, extract_crop_features
from visual_training_data import render_measure

cv2.setNumThreads(1)


@dataclass(frozen=True)
class AccentVisualDataset:
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


def _note(onset: Fraction, duration: Fraction, rng: random.Random) -> NoteIR:
    return NoteIR(
        onset=onset,
        duration=duration,
        voice="1",
        pitch=PitchIR(rng.choice("CDEFGAB"), Fraction(0, 1), rng.randint(3, 5)),
        rest=False,
        chord=False,
        grace=False,
        note_type=rng.choice(("quarter", "quarter", "eighth", "16th", "half")),
        dots=1 if rng.random() < 0.20 else 0,
        accidental="",
        ties=(),
        slurs=(),
        articulations=(rng.choice(("staccato", "accent", "tenuto")),)
        if rng.random() < 0.18
        else (),
        ornaments=(),
        tuple_ratio=None,
    )


def _measure(rng: random.Random) -> tuple[MeasureIR, int]:
    count = rng.randint(4, 9)
    duration = Fraction(4, count)
    notes = [
        _note(Fraction(index * 4, count), duration, rng) for index in range(count)
    ]
    target = rng.randrange(count)
    notes[target] = replace(notes[target], articulations=())
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
        target,
    )


def _evidence(
    image: np.ndarray,
    spacing: float,
    staff_top: float,
    staff_bottom: float,
    group: int,
) -> VisualMeasureEvidence:
    return VisualMeasureEvidence(
        page_index=0,
        system_index=0,
        measure_index=group,
        bbox=(0, 0, image.shape[1], image.shape[0]),
        spacing=float(spacing),
        **extract_crop_features(
            image,
            spacing=spacing,
            staff_top=staff_top,
            staff_bottom=staff_bottom,
        ),
    )


def _with_symbol_image(
    evidence: VisualMeasureEvidence,
    image: np.ndarray,
) -> VisualMeasureEvidence:
    ok, payload = cv2.imencode(
        ".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9]
    )
    if not ok:
        raise RuntimeError("failed to encode accent visual evidence")
    return replace(
        evidence,
        symbol_guard_image=base64.b64encode(payload.tobytes()).decode("ascii"),
    )


def _draw_accent(
    image: np.ndarray,
    centre_x: int,
    centre_y: int,
    rng: random.Random,
    value: int,
) -> None:
    variant = rng.choice(("vertical", "vertical", "right", "left"))
    thickness = rng.choice((1, 1, 2))
    if variant == "vertical":
        width = rng.randint(7, 10)
        cv2.line(
            image,
            (centre_x - width, centre_y - 3),
            (centre_x, centre_y + 1),
            value,
            thickness,
            cv2.LINE_AA,
        )
        cv2.line(
            image,
            (centre_x, centre_y + 1),
            (centre_x + width, centre_y - 3),
            value,
            thickness,
            cv2.LINE_AA,
        )
    else:
        direction = 1 if variant == "right" else -1
        apex_x = centre_x + direction * rng.randint(5, 7)
        base_x = centre_x - direction * rng.randint(7, 10)
        cv2.line(
            image,
            (base_x, centre_y - 4),
            (apex_x, centre_y),
            value,
            thickness,
            cv2.LINE_AA,
        )
        cv2.line(
            image,
            (base_x, centre_y + 4),
            (apex_x, centre_y),
            value,
            thickness,
            cv2.LINE_AA,
        )


def _draw_scenario(
    source: np.ndarray,
    measure: MeasureIR,
    event_index: int,
    rng: random.Random,
    scenario: str,
) -> np.ndarray:
    image = source.copy()
    x_ratio, y_ratio = event_position(measure, measure.notes[event_index])
    note_x = int(round(x_ratio * (image.shape[1] - 1)))
    note_y = int(round(y_ratio * (image.shape[0] - 1)))
    centre_x = note_x + rng.randint(-2, 2)
    centre_y = note_y + rng.choice((-1, 1)) * rng.randint(14, 20)
    value = rng.randint(150, 255)

    if scenario == "target_accent":
        _draw_accent(image, centre_x, centre_y, rng, value)
    elif scenario == "tenuto":
        cv2.line(
            image,
            (centre_x - rng.randint(6, 10), centre_y),
            (centre_x + rng.randint(6, 10), centre_y + rng.choice((-1, 0, 0, 1))),
            value,
            rng.choice((1, 1, 2)),
            cv2.LINE_AA,
        )
    elif scenario == "staccato":
        cv2.circle(
            image,
            (centre_x, centre_y),
            rng.choice((1, 2, 2, 3)),
            value,
            -1,
            cv2.LINE_AA,
        )
    elif scenario == "dust":
        for _ in range(rng.randint(1, 5)):
            cv2.circle(
                image,
                (centre_x + rng.randint(-15, 15), centre_y + rng.randint(-10, 10)),
                1,
                value,
                -1,
                cv2.LINE_AA,
            )
    elif scenario == "short_line":
        cv2.line(
            image,
            (centre_x - rng.randint(2, 5), centre_y + rng.randint(-5, 5)),
            (centre_x + rng.randint(2, 5), centre_y + rng.randint(-5, 5)),
            value,
            1,
            cv2.LINE_AA,
        )
    elif scenario == "single_diagonal":
        cv2.line(
            image,
            (centre_x - rng.randint(7, 10), centre_y - 3),
            (centre_x + rng.randint(3, 7), centre_y + 1),
            value,
            rng.choice((1, 2)),
            cv2.LINE_AA,
        )
    elif scenario == "duration_dot":
        cv2.circle(
            image,
            (note_x + rng.randint(9, 15), note_y + rng.randint(-2, 2)),
            rng.choice((1, 2)),
            value,
            -1,
            cv2.LINE_AA,
        )
    elif scenario != "none":
        raise ValueError(f"unknown accent visual scenario: {scenario}")

    if rng.random() < 0.24:
        image = cv2.GaussianBlur(image, (3, 3), rng.uniform(0.2, 0.75))
    if rng.random() < 0.28:
        generator = np.random.default_rng(rng.randrange(2**32))
        noise = generator.normal(0.0, rng.uniform(0.4, 3.0), image.shape)
        image = np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return image


SCENARIOS = (
    "none",
    "target_accent",
    "tenuto",
    "staccato",
    "dust",
    "short_line",
    "single_diagonal",
    "duration_dot",
)


def _build_group(
    args: tuple[int, int, int]
) -> tuple[list[list[float]], list[int], list[int], list[str]]:
    seed, generated, group_offset = args
    group_seed = _group_seed(seed, generated)
    rng = random.Random(group_seed)
    measure, event_index = _measure(rng)
    image, spacing, staff_top, staff_bottom = render_measure(
        measure,
        random.Random(group_seed ^ 0xD1B54A32D192ED03),
    )
    base = _evidence(image, spacing, staff_top, staff_bottom, generated)
    symbol = decode_symbol_guard_image(base.symbol_guard_image)
    if symbol is None:
        raise RuntimeError("rendered accent source evidence unavailable")

    rows: list[list[float]] = []
    labels: list[int] = []
    groups: list[int] = []
    scenarios: list[str] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        local_rng = random.Random(group_seed ^ ((scenario_index + 1) * 0x94D049BB133111EB))
        source = _draw_scenario(symbol, measure, event_index, local_rng, scenario)
        evidence = _with_symbol_image(base, source)
        feature = accent_visual_features(evidence, measure, event_index)
        if feature is None:
            raise RuntimeError("rendered accent feature unexpectedly unavailable")
        rows.append(feature)
        labels.append(int(scenario == "target_accent"))
        groups.append(group_offset + generated)
        scenarios.append(scenario)
        decode_bounded_png.cache_clear()
    return rows, labels, groups, scenarios


def build_accent_visual_dataset(
    seed: int,
    groups: int,
    *,
    group_offset: int = 0,
    workers: int = 1,
) -> AccentVisualDataset:
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
    return AccentVisualDataset(
        features=np.asarray(rows, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(group_ids, dtype=np.int64),
        scenarios=tuple(scenarios),
    )
