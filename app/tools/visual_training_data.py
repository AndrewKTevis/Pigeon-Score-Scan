from __future__ import annotations

"""Deterministic rendered data for visual/semantic measure compatibility.

The visual calibrator is deliberately narrower than the semantic ensemble.  It may
judge whether a source crop is compatible with an already proposed event sequence, but
it never creates semantic support by itself.  Each group contains two visually
compatible candidates plus gross-count traps and harder event-position traps which
preserve global pitch, rhythm or marker histograms while attaching them to the wrong
onsets.  Those paired traps are the failure mode addressed by visual calibrator v4.
"""

import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from fractions import Fraction
from multiprocessing import get_context

import cv2
import numpy as np

from event_training_data import random_measure
from scorescan.score_ir import MeasureIR, NoteIR, PitchIR
from scorescan.visual_evidence import (
    VisualMeasureEvidence,
    compatibility_vector,
    extract_crop_features,
    legacy_compatibility_vector,
    v3_compatibility_vector,
)

KINDS = (
    "exact",
    "invisible-notation",
    "pitch-profile-trap",
    "onset-profile-trap",
    "accidental-trap",
    "compact-mark-trap",
    "open-notehead-trap",
    "density-trap",
    "pitch-order-trap",
    "accidental-position-trap",
    "compact-position-trap",
    "open-notehead-position-trap",
    "event-kind-position-trap",
    "rhythm-position-trap",
)


@dataclass(frozen=True)
class VisualDataset:
    features: np.ndarray
    v3_features: np.ndarray
    legacy_features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    kinds: np.ndarray
    decision_groups: tuple[tuple[int, ...], ...]


def _group_seed(seed: int, group: int) -> int:
    value = (int(seed) + 0x9E3779B97F4A7C15 * (int(group) + 1)) & ((1 << 64) - 1)
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return value ^ (value >> 31)


def _pitch_y(
    pitch: PitchIR,
    staff_top: int,
    spacing: int,
    clef: tuple[str, int, int] | None,
) -> int:
    order = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}
    diatonic = pitch.octave * 7 + order[pitch.step]
    sign, line, octave_change = clef or ("G", 2, 0)
    sign = str(sign).upper()
    line = max(1, min(5, int(line or 2)))
    if sign == "F":
        reference = 3 * 7 + order["F"]
    elif sign == "C":
        reference = 4 * 7 + order["C"]
    else:
        reference = 4 * 7 + order["G"]
    reference += int(octave_change or 0) * 7
    position_from_bottom = 2 * (line - 1) + (diatonic - reference)
    staff_bottom = staff_top + 4 * spacing
    return int(round(staff_bottom - position_from_bottom * spacing / 2.0))


def _draw_accidental(image: np.ndarray, x: int, y: int, spacing: int, value: int, kind: str) -> None:
    if not kind:
        return
    if kind in {"sharp", "double-sharp"}:
        copies = 2 if kind == "double-sharp" else 1
        for copy in range(copies):
            centre = x + copy * max(5, spacing // 2)
            for dx in (-2, 2):
                cv2.line(image, (centre + dx, y - spacing), (centre + dx, y + spacing), value, 1, cv2.LINE_AA)
            for dy in (-3, 3):
                cv2.line(image, (centre - 5, y + dy), (centre + 5, y + dy - 1), value, 1, cv2.LINE_AA)
    elif kind in {"flat", "double-flat"}:
        copies = 2 if kind == "double-flat" else 1
        for copy in range(copies):
            centre = x + copy * max(4, spacing // 2)
            cv2.line(image, (centre, y - spacing), (centre, y + spacing), value, 1, cv2.LINE_AA)
            cv2.ellipse(image, (centre + 3, y + 2), (3, max(4, spacing // 2)), 0, 0, 360, value, 1, cv2.LINE_AA)
    elif kind == "natural":
        cv2.line(image, (x - 2, y - spacing), (x - 2, y + spacing // 2), value, 1, cv2.LINE_AA)
        cv2.line(image, (x + 3, y - spacing // 2), (x + 3, y + spacing), value, 1, cv2.LINE_AA)
        cv2.line(image, (x - 2, y - spacing // 2), (x + 3, y - spacing // 2 - 1), value, 1, cv2.LINE_AA)
        cv2.line(image, (x - 2, y + spacing // 2), (x + 3, y + spacing // 2 - 1), value, 1, cv2.LINE_AA)


def _draw_articulations(image: np.ndarray, x: int, y: int, spacing: int, value: int, note: NoteIR) -> None:
    offset_y = y - int(spacing * 1.45)
    for index, articulation in enumerate(note.articulations):
        mark_y = offset_y - index * max(3, spacing // 2)
        if articulation == "staccato":
            cv2.circle(image, (x, mark_y), max(2, spacing // 7), value, -1, cv2.LINE_AA)
        elif articulation == "tenuto":
            cv2.line(image, (x - spacing // 2, mark_y), (x + spacing // 2, mark_y), value, 1, cv2.LINE_AA)
        else:
            cv2.line(image, (x - spacing // 2, mark_y - 2), (x, mark_y), value, 1, cv2.LINE_AA)
            cv2.line(image, (x, mark_y), (x + spacing // 2, mark_y - 2), value, 1, cv2.LINE_AA)
    for index, _ornament in enumerate(note.ornaments):
        cv2.putText(
            image,
            "tr",
            (x - spacing // 2, offset_y - (index + len(note.articulations)) * max(4, spacing // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.26 + spacing / 90.0,
            value,
            1,
            cv2.LINE_AA,
        )


def _draw_rest(image: np.ndarray, x: int, staff_top: int, spacing: int, value: int, note: NoteIR) -> None:
    centre = staff_top + 2 * spacing
    if note.note_type == "whole":
        cv2.rectangle(image, (x - spacing // 2, centre - 1), (x + spacing // 2, centre + max(2, spacing // 4)), value, -1)
    elif note.note_type == "half":
        cv2.rectangle(image, (x - spacing // 2, centre - max(2, spacing // 4)), (x + spacing // 2, centre + 1), value, -1)
    elif note.note_type == "quarter":
        points = np.asarray(
            [
                (x + 2, centre - spacing),
                (x - 3, centre - spacing // 3),
                (x + 3, centre + spacing // 4),
                (x - 2, centre + spacing),
            ],
            dtype=np.int32,
        )
        cv2.polylines(image, [points], False, value, max(1, spacing // 6), cv2.LINE_AA)
    else:
        cv2.line(image, (x, centre - spacing), (x, centre + spacing), value, 1, cv2.LINE_AA)
        flags = {"eighth": 1, "16th": 2, "32nd": 3, "64th": 4}.get(note.note_type, 1)
        for flag in range(flags):
            fy = centre - spacing + flag * max(3, spacing // 2)
            cv2.circle(image, (x + max(2, spacing // 4), fy), max(2, spacing // 5), value, -1, cv2.LINE_AA)


def _quadratic_curve(
    image: np.ndarray,
    start: tuple[int, int],
    stop: tuple[int, int],
    *,
    arch: float,
    value: int,
    thickness: int,
) -> None:
    x1, y1 = start
    x2, y2 = stop
    if x2 <= x1 + 3:
        return
    midpoint_x = (x1 + x2) / 2.0
    midpoint_y = (y1 + y2) / 2.0 + float(arch)
    points: list[tuple[int, int]] = []
    for index in range(33):
        t = index / 32.0
        one = 1.0 - t
        x = one * one * x1 + 2.0 * one * t * midpoint_x + t * t * x2
        y = one * one * y1 + 2.0 * one * t * midpoint_y + t * t * y2
        points.append((int(round(x)), int(round(y))))
    cv2.polylines(
        image,
        [np.asarray(points, dtype=np.int32)],
        False,
        int(value),
        max(1, int(thickness)),
        cv2.LINE_AA,
    )


def _draw_notation_curves(
    image: np.ndarray,
    measure: MeasureIR,
    event_points: dict[int, tuple[int, int, bool]],
    spacing: int,
    value: int,
) -> None:
    # Ties are only rendered for complete adjacent start/stop pairs.  The training
    # renderer intentionally mirrors the bounded topology accepted by ScoreScan.
    for index, note in enumerate(measure.notes[:-1]):
        following = measure.notes[index + 1]
        if "start" not in note.ties or "stop" not in following.ties:
            continue
        if index not in event_points or index + 1 not in event_points:
            continue
        x1, y1, stem_up = event_points[index]
        x2, y2, _ = event_points[index + 1]
        side = 1.0 if stem_up else -1.0
        inset = max(2, int(round(spacing * 0.45)))
        _quadratic_curve(
            image,
            (x1 + inset, y1 + int(round(side * spacing * 0.42))),
            (x2 - inset, y2 + int(round(side * spacing * 0.42))),
            arch=side * spacing * 0.85,
            value=value,
            thickness=max(2, spacing // 8),
        )

    endpoints: dict[str, dict[str, int]] = {}
    for index, note in enumerate(measure.notes):
        for kind, number in note.slurs:
            normalized = str(kind).strip().casefold()
            if normalized not in {"start", "stop"}:
                continue
            endpoints.setdefault(str(number).strip() or "1", {})[normalized] = index
    for endpoint in endpoints.values():
        start_index = endpoint.get("start")
        stop_index = endpoint.get("stop")
        if (
            start_index is None
            or stop_index is None
            or stop_index <= start_index
            or start_index not in event_points
            or stop_index not in event_points
        ):
            continue
        x1, y1, stem_up = event_points[start_index]
        x2, y2, _ = event_points[stop_index]
        side = 1.0 if stem_up else -1.0
        span = max(1, stop_index - start_index)
        inset = max(2, int(round(spacing * 0.34)))
        _quadratic_curve(
            image,
            (x1 + inset, y1 + int(round(side * spacing * 0.65))),
            (x2 - inset, y2 + int(round(side * spacing * 0.65))),
            arch=side * spacing * min(2.8, 1.15 + 0.16 * span),
            value=value,
            thickness=max(2, spacing // 9),
        )


def render_measure(
    measure: MeasureIR,
    rng: random.Random,
    *,
    width: int = 430,
    height: int = 164,
    draw_curves: bool = False,
) -> tuple[np.ndarray, float, float, float]:
    spacing = rng.choice((9, 10, 11, 12, 13, 14))
    staff_top = rng.randint(54, 68)
    staff_bottom = staff_top + 4 * spacing
    image = np.full((height, width), rng.randint(238, 255), dtype=np.uint8)
    line_value = rng.randint(8, 58)
    for index in range(5):
        y = staff_top + index * spacing
        cv2.line(image, (10, y), (width - 10, y), line_value, rng.choice((1, 1, 1, 2)), cv2.LINE_AA)
    cv2.line(image, (10, staff_top), (10, staff_bottom), line_value, 1, cv2.LINE_AA)
    cv2.line(image, (width - 10, staff_top), (width - 10, staff_bottom), line_value, 1, cv2.LINE_AA)

    anchor_items = [
        (index, note)
        for index, note in enumerate(measure.notes)
        if not note.chord and not note.grace
    ]
    anchors = [note for _index, note in anchor_items]
    expected = measure.expected_duration
    if expected is None or expected <= 0:
        expected = max((note.onset + note.duration for note in anchors), default=Fraction(1, 1))
    x_positions = [
        int(
            round(
                34
                + max(0.0, min(1.0, float(note.onset / max(expected, Fraction(1, 64)))))
                * (width - 68)
            )
        )
        for note in anchors
    ]
    anchor_to_x: dict[Fraction, int] = {}
    event_points: dict[int, tuple[int, int, bool]] = {}
    stem_points: list[tuple[int, int, str, bool]] = []
    for anchor_index, (event_index, note) in enumerate(anchor_items):
        x = x_positions[anchor_index] + rng.randint(-2, 2)
        anchor_to_x[note.onset] = x
        if note.rest:
            _draw_rest(image, x, staff_top, spacing, line_value, note)
            for dot_index in range(max(0, int(note.dots))):
                cv2.circle(
                    image,
                    (x + spacing + dot_index * max(3, spacing // 2), staff_top + 2 * spacing),
                    max(2, spacing // 8),
                    line_value,
                    -1,
                    cv2.LINE_AA,
                )
            continue
        if note.pitch is None:
            continue
        y = _pitch_y(note.pitch, staff_top, spacing, measure.clef)
        open_head = note.note_type in {"whole", "half"}
        cv2.ellipse(
            image,
            (x, y),
            (max(3, spacing // 2), max(2, spacing // 3)),
            -18,
            0,
            360,
            line_value,
            max(1, spacing // 7) if open_head else -1,
            cv2.LINE_AA,
        )
        stem_up = y >= staff_top + 2 * spacing
        event_points[event_index] = (x, y, stem_up)
        stem_x = x + max(3, spacing // 2) if stem_up else x - max(3, spacing // 2)
        stem_end = y - int(spacing * 3.1) if stem_up else y + int(spacing * 3.1)
        if note.note_type != "whole":
            cv2.line(image, (stem_x, y), (stem_x, stem_end), line_value, 1, cv2.LINE_AA)
            stem_points.append((stem_x, stem_end, note.note_type, stem_up))
        flag_count = {"eighth": 1, "16th": 2, "32nd": 3, "64th": 4}.get(note.note_type, 0)
        for flag in range(flag_count):
            direction = 1 if stem_up else -1
            y0 = stem_end + direction * flag * max(2, spacing // 3)
            cv2.line(
                image,
                (stem_x, y0),
                (stem_x + direction * spacing, y0 + direction * spacing // 2),
                line_value,
                2,
                cv2.LINE_AA,
            )
        _draw_accidental(image, x - int(spacing * 1.45), y, spacing, line_value, note.accidental)
        for dot_index in range(max(0, int(note.dots))):
            cv2.circle(
                image,
                (x + spacing + dot_index * max(3, spacing // 2), y),
                max(2, spacing // 8),
                line_value,
                -1,
                cv2.LINE_AA,
            )
        _draw_articulations(image, x, y, spacing, line_value, note)

    beamable = {"eighth", "16th", "32nd", "64th"}
    run: list[tuple[int, int, str, bool]] = []
    for point in stem_points + [(-999, -999, "break", True)]:
        if point[2] in beamable and (not run or point[3] == run[-1][3]):
            run.append(point)
            continue
        if len(run) >= 2:
            levels = min({"eighth": 1, "16th": 2, "32nd": 3, "64th": 4}[item[2]] for item in run)
            direction = 1 if run[0][3] else -1
            for level in range(levels):
                offset = direction * level * max(2, spacing // 3)
                cv2.line(
                    image,
                    (run[0][0], run[0][1] + offset),
                    (run[-1][0], run[-1][1] + offset),
                    line_value,
                    max(2, spacing // 5),
                    cv2.LINE_AA,
                )
        run = [point] if point[2] in beamable else []

    for note in measure.notes:
        if note.chord and note.pitch is not None:
            x = anchor_to_x.get(note.onset, width // 2)
            y = _pitch_y(note.pitch, staff_top, spacing, measure.clef)
            open_head = note.note_type in {"whole", "half"}
            cv2.ellipse(
                image,
                (x, y),
                (max(3, spacing // 2), max(2, spacing // 3)),
                -18,
                0,
                360,
                line_value,
                max(1, spacing // 7) if open_head else -1,
                cv2.LINE_AA,
            )
        elif note.grace and note.pitch is not None:
            x = anchor_to_x.get(note.onset, 32) - spacing
            y = _pitch_y(note.pitch, staff_top, spacing, measure.clef)
            cv2.ellipse(image, (x, y), (max(2, spacing // 3), max(1, spacing // 4)), -18, 0, 360, line_value, -1, cv2.LINE_AA)
            cv2.line(image, (x + 2, y), (x + 2, y - spacing * 2), line_value, 1, cv2.LINE_AA)

    if draw_curves:
        _draw_notation_curves(image, measure, event_points, spacing, line_value)

    for direction_index, direction in enumerate(measure.directions):
        y = (
            staff_top - spacing * 2 - direction_index * max(8, spacing)
            if direction.placement != "below"
            else staff_bottom + spacing * 2 + direction_index * max(8, spacing)
        )
        x = 25 + direction_index * 90
        cv2.putText(
            image,
            direction.value[:12],
            (x, int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35 + spacing / 60.0,
            line_value,
            1,
            cv2.LINE_AA,
        )

    angle = rng.uniform(-0.85, 0.85)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    image = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderValue=255)
    if rng.random() < 0.72:
        sigma = rng.uniform(0.1, 1.15)
        image = cv2.GaussianBlur(image, (3, 3), sigma)
    noise_sigma = abs(rng.normalvariate(0.0, 2.4))
    if noise_sigma > 0.2:
        generator = np.random.default_rng(rng.randrange(2**32))
        image = np.clip(image.astype(np.float32) + generator.normal(0.0, noise_sigma, image.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.48:
        quality = rng.randint(42, 92)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    return image, float(spacing), float(staff_top), float(staff_bottom)


def _evidence(image: np.ndarray, spacing: float, staff_top: float, staff_bottom: float, group: int) -> VisualMeasureEvidence:
    features = extract_crop_features(image, spacing=spacing, staff_top=staff_top, staff_bottom=staff_bottom)
    return VisualMeasureEvidence(
        page_index=0,
        system_index=0,
        measure_index=group,
        bbox=(0, 0, image.shape[1], image.shape[0]),
        spacing=spacing,
        **features,
    )


def _invisible_notation(measure: MeasureIR, rng: random.Random) -> MeasureIR:
    notes = list(measure.notes)
    indices = [index for index, note in enumerate(notes) if note.pitch is not None and not note.rest]
    if not indices:
        return measure
    for index in rng.sample(indices, k=min(len(indices), rng.choice((1, 1, 2)))):
        note = notes[index]
        if rng.random() < 0.5:
            notes[index] = replace(note, ties=() if note.ties else ("start",))
        else:
            notes[index] = replace(note, slurs=() if note.slurs else (("start", "1"),))
    return replace(measure, notes=tuple(notes))


def _pitch_profile_trap(measure: MeasureIR, rng: random.Random) -> MeasureIR:
    notes = list(measure.notes)
    indices = [index for index, note in enumerate(notes) if note.pitch is not None and not note.rest]
    if not indices:
        return measure
    for index in rng.sample(indices, k=min(len(indices), rng.choice((2, 3, 4)))):
        note = notes[index]
        assert note.pitch is not None
        octave_shift = rng.choice((-2, -1, 1, 2))
        steps = [value for value in "CDEFGAB" if value != note.pitch.step]
        notes[index] = replace(
            note,
            pitch=replace(
                note.pitch,
                step=rng.choice(steps),
                octave=max(1, min(8, note.pitch.octave + octave_shift)),
            ),
        )
    return replace(measure, notes=tuple(notes))


def _onset_profile_trap(measure: MeasureIR, rng: random.Random) -> MeasureIR:
    notes = list(measure.notes)
    anchors = [index for index, note in enumerate(notes) if not note.chord and not note.grace]
    if not anchors:
        return measure
    expected = measure.expected_duration or max(
        (notes[index].onset + notes[index].duration for index in anchors),
        default=Fraction(1, 1),
    )
    shifts = (Fraction(-3, 4), Fraction(-1, 2), Fraction(1, 2), Fraction(3, 4))
    for index in rng.sample(anchors, k=min(len(anchors), rng.choice((2, 3, 4)))):
        note = notes[index]
        onset = max(Fraction(0, 1), min(expected, note.onset + rng.choice(shifts)))
        old_onset = note.onset
        notes[index] = replace(note, onset=onset)
        for chord_index, chord_note in enumerate(notes):
            if chord_note.chord and chord_note.onset == old_onset:
                notes[chord_index] = replace(chord_note, onset=onset)
    return replace(measure, notes=tuple(notes))


def _accidental_trap(measure: MeasureIR, rng: random.Random) -> MeasureIR:
    notes = list(measure.notes)
    indices = [index for index, note in enumerate(notes) if note.pitch is not None and not note.rest]
    if not indices:
        return measure
    existing = [index for index in indices if notes[index].accidental]
    if len(existing) >= 2:
        for index in existing:
            notes[index] = replace(notes[index], accidental="")
    else:
        available = [index for index in indices if not notes[index].accidental]
        for index in rng.sample(available, k=min(len(available), rng.choice((2, 3, 4)))):
            notes[index] = replace(notes[index], accidental=rng.choice(("sharp", "flat", "natural")))
    return replace(measure, notes=tuple(notes))


def _compact_mark_trap(measure: MeasureIR, rng: random.Random) -> MeasureIR:
    notes = list(measure.notes)
    indices = [index for index, note in enumerate(notes) if not note.grace]
    if not indices:
        return measure
    compact_count = sum(max(0, int(notes[index].dots)) + len(notes[index].articulations) for index in indices)
    if compact_count >= 3:
        for index in indices:
            notes[index] = replace(notes[index], dots=0, articulations=())
    else:
        available = [
            index
            for index in indices
            if notes[index].dots == 0 and not notes[index].articulations
        ]
        for index in rng.sample(available, k=min(len(available), rng.choice((3, 4, 5)))):
            note = notes[index]
            if rng.random() < 0.55:
                notes[index] = replace(note, dots=rng.choice((1, 1, 2)))
            else:
                notes[index] = replace(note, articulations=(rng.choice(("staccato", "accent", "tenuto")),))
    return replace(measure, notes=tuple(notes))


def _open_notehead_trap(measure: MeasureIR, rng: random.Random) -> MeasureIR:
    notes = list(measure.notes)
    indices = [
        index
        for index, note in enumerate(notes)
        if note.pitch is not None and not note.rest and not note.grace
    ]
    if not indices:
        return measure
    for index in rng.sample(indices, k=min(len(indices), rng.choice((2, 2, 3)))):
        note = notes[index]
        if note.note_type in {"whole", "half"}:
            replacement = rng.choice(("quarter", "eighth"))
        else:
            replacement = rng.choice(("whole", "half"))
        notes[index] = replace(note, note_type=replacement)
    return replace(measure, notes=tuple(notes))


def _density_trap(measure: MeasureIR, rng: random.Random) -> MeasureIR:
    notes = list(measure.notes)
    regular = [index for index, note in enumerate(notes) if not note.grace]
    if len(regular) > 2 and rng.random() < 0.55:
        remove = set(rng.sample(regular, k=max(1, len(regular) // 3)))
        notes = [note for index, note in enumerate(notes) if index not in remove]
    else:
        source = notes[rng.choice(regular)] if regular else None
        if source is not None:
            for _ in range(rng.choice((2, 3, 4))):
                notes.append(replace(source, onset=source.onset + Fraction(rng.choice((1, 2, 3)), 8), chord=False))
    return replace(measure, notes=tuple(notes))


def _pitch_order_trap(measure: MeasureIR, rng: random.Random) -> MeasureIR:
    notes = list(measure.notes)
    indices = [
        index
        for index, note in enumerate(notes)
        if note.pitch is not None and not note.rest and not note.chord and not note.grace
    ]
    if len(indices) < 2 or len({notes[index].pitch for index in indices}) < 2:
        return _pitch_profile_trap(measure, rng)
    selected = rng.sample(indices, k=min(len(indices), rng.choice((2, 3, 4))))
    payload = [notes[index].pitch for index in selected]
    payload = payload[1:] + payload[:1]
    for index, pitch in zip(selected, payload, strict=True):
        notes[index] = replace(notes[index], pitch=pitch)
    return replace(measure, notes=tuple(notes))


def _accidental_position_trap(measure: MeasureIR, rng: random.Random) -> MeasureIR:
    notes = list(measure.notes)
    indices = [index for index, note in enumerate(notes) if note.pitch is not None and not note.rest]
    marked = [index for index in indices if notes[index].accidental]
    unmarked = [index for index in indices if not notes[index].accidental]
    if not marked or not unmarked:
        return _accidental_trap(measure, rng)
    source = rng.choice(marked)
    target = rng.choice(unmarked)
    value = notes[source].accidental
    notes[source] = replace(notes[source], accidental="")
    notes[target] = replace(notes[target], accidental=value)
    return replace(measure, notes=tuple(notes))


def _compact_position_trap(measure: MeasureIR, rng: random.Random) -> MeasureIR:
    notes = list(measure.notes)
    indices = [index for index, note in enumerate(notes) if not note.grace]
    marked = [
        index for index in indices if notes[index].dots > 0 or bool(notes[index].articulations)
    ]
    unmarked = [
        index for index in indices if notes[index].dots == 0 and not notes[index].articulations
    ]
    if not marked or not unmarked:
        return _compact_mark_trap(measure, rng)
    source = rng.choice(marked)
    target = rng.choice(unmarked)
    dots, articulations = notes[source].dots, notes[source].articulations
    notes[source] = replace(notes[source], dots=0, articulations=())
    notes[target] = replace(notes[target], dots=dots, articulations=articulations)
    return replace(measure, notes=tuple(notes))


def _open_notehead_position_trap(measure: MeasureIR, rng: random.Random) -> MeasureIR:
    notes = list(measure.notes)
    pitched = [
        index
        for index, note in enumerate(notes)
        if note.pitch is not None and not note.rest and not note.grace
    ]
    open_indices = [index for index in pitched if notes[index].note_type in {"whole", "half"}]
    filled_indices = [index for index in pitched if notes[index].note_type not in {"whole", "half"}]
    if not open_indices or not filled_indices:
        return _open_notehead_trap(measure, rng)
    left = rng.choice(open_indices)
    right = rng.choice(filled_indices)
    left_type, right_type = notes[left].note_type, notes[right].note_type
    notes[left] = replace(notes[left], note_type=right_type)
    notes[right] = replace(notes[right], note_type=left_type)
    return replace(measure, notes=tuple(notes))


def _event_kind_position_trap(measure: MeasureIR, rng: random.Random) -> MeasureIR:
    notes = list(measure.notes)
    rests = [index for index, note in enumerate(notes) if note.rest and not note.grace and not note.chord]
    pitched = [
        index
        for index, note in enumerate(notes)
        if note.pitch is not None and not note.rest and not note.grace and not note.chord
    ]
    if not rests or not pitched:
        return _density_trap(measure, rng)
    rest_index = rng.choice(rests)
    pitch_index = rng.choice(pitched)
    pitch_payload = (notes[pitch_index].pitch, notes[pitch_index].accidental)
    notes[rest_index] = replace(
        notes[rest_index], rest=False, pitch=pitch_payload[0], accidental=pitch_payload[1]
    )
    notes[pitch_index] = replace(notes[pitch_index], rest=True, pitch=None, accidental="")
    return replace(measure, notes=tuple(notes))


def _rhythm_position_trap(measure: MeasureIR, rng: random.Random) -> MeasureIR:
    notes = list(measure.notes)
    indices = [index for index, note in enumerate(notes) if not note.grace and not note.chord]
    if len(indices) < 2:
        return _open_notehead_trap(measure, rng)
    pairs = [
        (left, right)
        for offset, left in enumerate(indices)
        for right in indices[offset + 1 :]
        if (notes[left].duration, notes[left].note_type, notes[left].dots)
        != (notes[right].duration, notes[right].note_type, notes[right].dots)
    ]
    if not pairs:
        return _compact_mark_trap(measure, rng)
    left, right = rng.choice(pairs)
    left_payload = (notes[left].duration, notes[left].note_type, notes[left].dots)
    right_payload = (notes[right].duration, notes[right].note_type, notes[right].dots)
    notes[left] = replace(
        notes[left], duration=right_payload[0], note_type=right_payload[1], dots=right_payload[2]
    )
    notes[right] = replace(
        notes[right], duration=left_payload[0], note_type=left_payload[1], dots=left_payload[2]
    )
    return replace(measure, notes=tuple(notes))


def _build_group(
    task: tuple[int, int],
) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[int], list[str]]:
    seed, group = task
    cv2.setNumThreads(1)
    rng = random.Random(_group_seed(seed, group))
    measure = random_measure(rng)
    image, spacing, staff_top, staff_bottom = render_measure(measure, rng)
    evidence = _evidence(image, spacing, staff_top, staff_bottom, group)
    items = list(
        zip(
            (
                measure,
                _invisible_notation(measure, rng),
                _pitch_profile_trap(measure, rng),
                _onset_profile_trap(measure, rng),
                _accidental_trap(measure, rng),
                _compact_mark_trap(measure, rng),
                _open_notehead_trap(measure, rng),
                _density_trap(measure, rng),
                _pitch_order_trap(measure, rng),
                _accidental_position_trap(measure, rng),
                _compact_position_trap(measure, rng),
                _open_notehead_position_trap(measure, rng),
                _event_kind_position_trap(measure, rng),
                _rhythm_position_trap(measure, rng),
            ),
            (1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            KINDS,
            strict=True,
        )
    )
    rng.shuffle(items)
    rows = [compatibility_vector(evidence, candidate) for candidate, _label, _kind in items]
    v3_rows = [v3_compatibility_vector(evidence, candidate) for candidate, _label, _kind in items]
    legacy = [legacy_compatibility_vector(evidence, candidate) for candidate, _label, _kind in items]
    labels = [label for _candidate, label, _kind in items]
    kinds = [kind for _candidate, _label, kind in items]
    return rows, v3_rows, legacy, labels, kinds


def build_dataset(seed: int, groups: int, workers: int) -> VisualDataset:
    if groups < 40:
        raise ValueError("visual training requires at least 40 independent groups")
    tasks = [(seed, group) for group in range(groups)]
    if workers <= 1:
        built = [_build_group(task) for task in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
        ) as executor:
            built = list(executor.map(_build_group, tasks, chunksize=8))

    features: list[list[float]] = []
    v3_features: list[list[float]] = []
    legacy_features: list[list[float]] = []
    labels: list[int] = []
    group_ids: list[int] = []
    kinds: list[str] = []
    decisions: list[tuple[int, ...]] = []
    for group, (rows, v3_rows, legacy_rows, local_labels, local_kinds) in enumerate(built):
        start = len(features)
        features.extend(rows)
        v3_features.extend(v3_rows)
        legacy_features.extend(legacy_rows)
        labels.extend(local_labels)
        group_ids.extend([group] * len(rows))
        kinds.extend(local_kinds)
        decisions.append(tuple(range(start, start + len(rows))))
    return VisualDataset(
        features=np.asarray(features, dtype=np.float64),
        v3_features=np.asarray(v3_features, dtype=np.float64),
        legacy_features=np.asarray(legacy_features, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int32),
        groups=np.asarray(group_ids, dtype=np.int32),
        kinds=np.asarray(kinds, dtype="U32"),
        decision_groups=tuple(decisions),
    )
