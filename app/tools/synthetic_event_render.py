from __future__ import annotations

"""Shared deterministic notation primitives for grouped visual-gate training data.

This is deliberately a training-only renderer.  It creates controlled hard examples for
small veto models and is not used by the ScoreScan recognition runtime.
"""

import random
from fractions import Fraction

import cv2
import numpy as np

from scorescan.score_ir import MeasureIR, NoteIR, PitchIR
from scorescan.visual_evidence import SYMBOL_GUARD_HEIGHT

DURATION_BY_TYPE = {
    "whole": Fraction(4, 1),
    "half": Fraction(2, 1),
    "quarter": Fraction(1, 1),
    "eighth": Fraction(1, 2),
    "16th": Fraction(1, 4),
}
PITCHES = (
    PitchIR("C", Fraction(0, 1), 4),
    PitchIR("D", Fraction(0, 1), 4),
    PitchIR("E", Fraction(0, 1), 4),
    PitchIR("F", Fraction(0, 1), 4),
    PitchIR("G", Fraction(0, 1), 4),
    PitchIR("A", Fraction(0, 1), 4),
    PitchIR("B", Fraction(0, 1), 4),
    PitchIR("C", Fraction(0, 1), 5),
    PitchIR("D", Fraction(0, 1), 5),
    PitchIR("E", Fraction(0, 1), 5),
    PitchIR("F", Fraction(0, 1), 5),
    PitchIR("G", Fraction(0, 1), 5),
)


def event_note(kind: str, note_type: str, pitch: PitchIR, onset: Fraction, dots: int) -> NoteIR:
    return NoteIR(
        onset=onset,
        duration=DURATION_BY_TYPE[note_type],
        voice="1",
        pitch=None if kind == "rest" else pitch,
        rest=kind == "rest",
        chord=False,
        grace=False,
        note_type=note_type,
        dots=dots,
        accidental="",
        ties=(),
        slurs=(),
        articulations=(),
        ornaments=(),
        tuple_ratio=None,
    )


def single_event_measure(
    kind: str, note_type: str, pitch: PitchIR, onset: Fraction, dots: int
) -> MeasureIR:
    return MeasureIR(
        divisions=4,
        time_signature=(4, 4),
        key_signature=(0, "major"),
        clef=("G", 2, 0),
        notes=(event_note(kind, note_type, pitch, onset, dots),),
        directions=(),
        barlines=(),
    )


def draw_note(
    image: np.ndarray,
    x: int,
    y: int,
    note_type: str,
    rng: random.Random,
    intensity: int,
) -> None:
    angle = rng.uniform(-18.0, 18.0)
    axes = (rng.randint(5, 7), rng.randint(3, 4))
    thickness = rng.choice((1, 2))
    if note_type in {"whole", "half"}:
        cv2.ellipse(image, (x, y), axes, angle, 0, 360, intensity, thickness, cv2.LINE_AA)
    else:
        cv2.ellipse(image, (x, y), axes, angle, 0, 360, intensity, -1, cv2.LINE_AA)
    if note_type == "whole":
        return
    stem_up = y >= SYMBOL_GUARD_HEIGHT // 2 or rng.random() < 0.45
    stem_x = x + axes[0] - 1 if stem_up else x - axes[0] + 1
    stem_end = y - rng.randint(18, 25) if stem_up else y + rng.randint(18, 25)
    cv2.line(image, (stem_x, y), (stem_x, stem_end), intensity, rng.choice((1, 2)), cv2.LINE_AA)
    if note_type in {"eighth", "16th"}:
        direction = 1 if stem_up else -1
        for flag_index in range(1 if note_type == "eighth" else 2):
            flag_y = stem_end + direction * flag_index * 6
            points = np.asarray(
                [
                    (stem_x, flag_y),
                    (
                        stem_x + direction * rng.randint(8, 12),
                        flag_y + direction * rng.randint(4, 8),
                    ),
                    (
                        stem_x + direction * rng.randint(4, 7),
                        flag_y + direction * rng.randint(10, 14),
                    ),
                ],
                dtype=np.int32,
            )
            cv2.polylines(image, [points], False, intensity, 2, cv2.LINE_AA)


def draw_rest(
    image: np.ndarray,
    x: int,
    y: int,
    note_type: str,
    rng: random.Random,
    intensity: int,
) -> None:
    if note_type == "whole":
        cv2.rectangle(image, (x - 8, y - 4), (x + 8, y + 1), intensity, -1)
        return
    if note_type == "half":
        cv2.rectangle(image, (x - 8, y - 1), (x + 8, y + 5), intensity, -1)
        return
    if note_type == "quarter":
        points = np.asarray(
            [
                (x - 3, y - 15),
                (x + 4, y - 7),
                (x - 2, y),
                (x + 5, y + 7),
                (x - 4, y + 16),
                (x + 3, y + 13),
            ],
            dtype=np.int32,
        )
        cv2.polylines(image, [points], False, intensity, rng.choice((2, 3)), cv2.LINE_AA)
        return
    cv2.line(image, (x + 3, y - 15), (x - 2, y + 12), intensity, 2, cv2.LINE_AA)
    cv2.ellipse(image, (x + 2, y - 12), (5, 3), -20, 0, 360, intensity, -1, cv2.LINE_AA)
    cv2.ellipse(image, (x - 3, y + 10), (4, 3), -20, 0, 360, intensity, -1, cv2.LINE_AA)
    if note_type == "16th":
        cv2.ellipse(image, (x + 3, y - 4), (4, 3), -20, 0, 360, intensity, -1, cv2.LINE_AA)
