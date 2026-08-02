from __future__ import annotations

"""Group-isolated rendered examples for accidental-presence recognition."""

import base64
import random
from dataclasses import dataclass, replace
from fractions import Fraction

import cv2
import numpy as np

from scorescan.accidental_presence_guard import accidental_hog_features
from scorescan.local_symbol_image import decode_bounded_png
from scorescan.visual_evidence import SYMBOL_GUARD_HEIGHT, SYMBOL_GUARD_WIDTH, VisualMeasureEvidence
from scorescan.score_ir import MeasureIR, NoteIR, PitchIR
from visual_training_data import render_measure

cv2.setNumThreads(1)

SYMBOLS = ("natural", "sharp", "flat", "double-sharp", "double-flat")
ALTERS = {
    "natural": Fraction(0, 1),
    "sharp": Fraction(1, 1),
    "flat": Fraction(-1, 1),
    "double-sharp": Fraction(2, 1),
    "double-flat": Fraction(-2, 1),
}


@dataclass(frozen=True)
class AccidentalPresenceDataset:
    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    symbols: tuple[str, ...]



def _symbol_evidence(
    image: np.ndarray,
    spacing: float,
    staff_top: float,
    staff_bottom: float,
    group: int,
) -> VisualMeasureEvidence:
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(15, int(round(max(spacing, 3.0) * 5.5))), 1)
    )
    staff_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    raw_nonstaff = cv2.subtract(binary, staff_mask)
    top = max(0, int(round(staff_top - spacing * 2.0)))
    bottom = min(image.shape[0], int(round(staff_bottom + spacing * 2.0)) + 1)
    profile = raw_nonstaff[top:bottom, :].copy()
    bar_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(5, int(round(spacing * 3.75))))
    )
    profile = cv2.subtract(profile, cv2.morphologyEx(profile, cv2.MORPH_OPEN, bar_kernel))
    edge = max(1, int(round(image.shape[1] * 0.025)))
    if profile.shape[1] > edge * 2:
        profile[:, :edge] = 0
        profile[:, -edge:] = 0
    resized = cv2.resize(
        profile,
        (SYMBOL_GUARD_WIDTH, SYMBOL_GUARD_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    ok, encoded = cv2.imencode(".png", resized, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    payload = base64.b64encode(encoded.tobytes()).decode("ascii") if ok else ""
    return VisualMeasureEvidence(
        page_index=0,
        system_index=0,
        measure_index=group,
        bbox=(0, 0, image.shape[1], image.shape[0]),
        spacing=float(spacing),
        ink_density=0.0,
        nonstaff_ink_density=0.0,
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
        symbol_guard_image=payload,
    )

def _group_seed(seed: int, group: int) -> int:
    value = (int(seed) + 0x9E3779B97F4A7C15 * (int(group) + 1)) & ((1 << 64) - 1)
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return value ^ (value >> 31)


def _base_measure(rng: random.Random) -> tuple[MeasureIR, int]:
    count = rng.choice((4, 5, 6, 7, 8))
    clef = rng.choice((("G", 2, 0), ("F", 4, 0)))
    notes: list[NoteIR] = []
    target = rng.randrange(count)
    steps = "CDEFGAB"
    for index in range(count):
        rest = index != target and rng.random() < 0.12
        pitch = None if rest else PitchIR(rng.choice(steps), Fraction(0, 1), rng.randint(3, 5))
        symbol = ""
        if index != target and pitch is not None and rng.random() < 0.22:
            symbol = rng.choice(SYMBOLS)
            pitch = replace(pitch, alter=ALTERS[symbol])
        note_type = rng.choice(("quarter", "quarter", "eighth", "16th", "half"))
        notes.append(
            NoteIR(
                onset=Fraction(index * 4, count),
                duration=Fraction(4, count),
                voice="1",
                pitch=pitch,
                rest=rest,
                chord=False,
                grace=False,
                note_type=note_type,
                dots=1 if rng.random() < 0.12 else 0,
                accidental=symbol,
                ties=(),
                slurs=(),
                articulations=("staccato",) if rng.random() < 0.10 else (),
                ornaments=(),
                tuple_ratio=None,
            )
        )
    return (
        MeasureIR(
            divisions=24,
            time_signature=(4, 4),
            key_signature=(0, "major"),
            clef=clef,
            notes=tuple(notes),
            directions=(),
            barlines=(),
        ),
        target,
    )

def _with_symbol(measure: MeasureIR, event_index: int, symbol: str) -> MeasureIR:
    notes = list(measure.notes)
    note = notes[event_index]
    assert note.pitch is not None and not note.rest
    notes[event_index] = replace(
        note,
        pitch=replace(note.pitch, alter=ALTERS[symbol]),
        accidental=symbol,
    )
    return replace(measure, notes=tuple(notes))


def _build_group(args: tuple[int, int, int]) -> tuple[list[list[float]], list[int], list[int], list[str]]:
    seed, generated, group_offset = args
    rng = random.Random(_group_seed(seed, generated))
    base, event_index = _base_measure(rng)
    symbol = SYMBOLS[generated % len(SYMBOLS)]
    present = _with_symbol(base, event_index, symbol)
    rows: list[list[float]] = []
    labels: list[int] = []
    group_ids: list[int] = []
    symbols: list[str] = []
    for local_measure, label, name, image_index in (
        (base, 0, "none", generated * 2),
        (present, 1, symbol, generated * 2 + 1),
    ):
        image, spacing, staff_top, staff_bottom = render_measure(local_measure, rng)
        evidence = _symbol_evidence(image, spacing, staff_top, staff_bottom, image_index)
        features = accidental_hog_features(evidence, local_measure, event_index)
        if features is None:
            raise RuntimeError("rendered accidental-presence feature unexpectedly unavailable")
        rows.append(features)
        labels.append(label)
        group_ids.append(group_offset + generated)
        symbols.append(name)
    decode_bounded_png.cache_clear()
    return rows, labels, group_ids, symbols


def build_accidental_presence_dataset(
    seed: int,
    groups: int,
    *,
    group_offset: int = 0,
) -> AccidentalPresenceDataset:
    built = [
        _build_group((int(seed), generated, int(group_offset)))
        for generated in range(groups)
    ]
    rows: list[list[float]] = []
    labels: list[int] = []
    group_ids: list[int] = []
    symbol_names: list[str] = []
    for local_rows, local_labels, local_groups, local_symbols in built:
        rows.extend(local_rows)
        labels.extend(local_labels)
        group_ids.extend(local_groups)
        symbol_names.extend(local_symbols)
    return AccidentalPresenceDataset(
        features=np.asarray(rows, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(group_ids, dtype=np.int64),
        symbols=tuple(symbol_names),
    )
