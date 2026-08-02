from __future__ import annotations

"""Deterministic rendered staff systems for barline training and frozen evaluation.

The generator produces complete single-staff systems rather than isolated crops.  Model
training therefore sees the same proposal distribution as production, including stems,
beams, text, repeat marks, connected notation, and scan degradation.  Related degraded
variants share one group identity and must never cross dataset partitions.
"""

from dataclasses import dataclass
import random
from typing import Iterable

import cv2
import numpy as np

from scorescan.barline_classifier import BarlineFeatures, extract_barline_features
from scorescan.layout import StaffSystem, _vertical_proposals


@dataclass(frozen=True)
class RenderedBarlineSystem:
    group: int
    variant: int
    binary: np.ndarray
    system: StaffSystem
    true_barlines: tuple[int, ...]
    measure_count: int


@dataclass(frozen=True)
class ProposalExample:
    group: int
    variant: int
    x: int
    x_start: int
    x_end: int
    label: int
    features: BarlineFeatures


def _draw_note(
    image: np.ndarray,
    rng: random.Random,
    *,
    x: int,
    y: int,
    spacing: int,
    force_long: bool = False,
) -> None:
    up = rng.random() < 0.58
    radius_x = max(3, int(round(spacing * 0.42)))
    radius_y = max(2, int(round(spacing * 0.29)))
    cv2.ellipse(image, (x, y), (radius_x, radius_y), -18, 0, 360, 255, -1, cv2.LINE_AA)
    stem_x = x + radius_x - 1 if up else x - radius_x + 1
    length_ratio = rng.uniform(2.7, 4.0)
    if force_long or rng.random() < 0.10:
        length_ratio = rng.uniform(4.3, 5.7)
    end_y = int(round(y - spacing * length_ratio if up else y + spacing * length_ratio))
    cv2.line(
        image,
        (stem_x, y),
        (stem_x, max(0, min(image.shape[0] - 1, end_y))),
        255,
        max(1, int(round(spacing * rng.uniform(0.07, 0.15)))),
        cv2.LINE_AA,
    )
    if rng.random() < 0.22:
        beam_dx = int(round(spacing * rng.uniform(1.8, 3.4)))
        beam_dy = rng.choice((-1, 1)) * max(1, int(round(spacing * 0.22)))
        cv2.line(
            image,
            (stem_x, max(0, min(image.shape[0] - 1, end_y))),
            (stem_x + beam_dx, max(0, min(image.shape[0] - 1, end_y + beam_dy))),
            255,
            max(2, int(round(spacing * 0.22))),
            cv2.LINE_AA,
        )


def _draw_barline(
    image: np.ndarray,
    rng: random.Random,
    *,
    x: int,
    top: int,
    bottom: int,
    spacing: int,
    final: bool,
) -> None:
    kind = "final" if final else rng.choices(
        ("single", "single", "double", "repeat", "broken", "thin"),
        weights=(33, 26, 10, 8, 13, 10),
        k=1,
    )[0]
    thickness = max(1, int(round(spacing * rng.uniform(0.07, 0.18))))
    if kind in {"single", "thin", "repeat", "broken"}:
        line_thickness = 1 if kind == "thin" else thickness
        cv2.line(image, (x, top), (x, bottom), 255, line_thickness, cv2.LINE_AA)
        if kind == "broken":
            gap = max(1, int(round(spacing * rng.uniform(0.16, 0.34))))
            gap_y = rng.randint(top + spacing, bottom - spacing)
            cv2.rectangle(image, (x - line_thickness - 1, gap_y - gap), (x + line_thickness + 1, gap_y + gap), 0, -1)
        if kind == "repeat":
            for dot_y in (top + int(round(spacing * 1.5)), top + int(round(spacing * 2.5))):
                cv2.circle(image, (x + spacing, dot_y), max(1, spacing // 5), 255, -1, cv2.LINE_AA)
    else:
        gap = max(2, int(round(spacing * 0.34)))
        cv2.line(image, (x - gap, top), (x - gap, bottom), 255, 1, cv2.LINE_AA)
        thick = max(2, int(round(spacing * (0.22 if final else 0.12))))
        cv2.line(image, (x + gap, top), (x + gap, bottom), 255, thick, cv2.LINE_AA)


def _base_system(group: int, seed: int) -> tuple[np.ndarray, StaffSystem, tuple[int, ...], int]:
    rng = random.Random(seed)
    spacing = rng.randint(9, 20)
    measure_count = rng.randint(3, 9)
    normal = rng.uniform(spacing * 10.5, spacing * 18.0)
    widths = [normal * rng.uniform(0.74, 1.27) for _ in range(measure_count)]
    if rng.random() < 0.32:
        widths[0] *= rng.uniform(0.42, 0.76)  # pickup or compressed opening
    if rng.random() < 0.28:
        widths[-1] *= rng.uniform(0.52, 0.86)
    if measure_count >= 5 and rng.random() < 0.20:
        index = rng.randrange(1, measure_count - 1)
        widths[index] *= rng.choice((rng.uniform(0.58, 0.76), rng.uniform(1.28, 1.48)))

    left = spacing * rng.randint(3, 5)
    first_line = spacing * rng.randint(3, 4)
    lines = [float(first_line + index * spacing) for index in range(5)]
    cursor = float(left)
    boundaries: list[int] = []
    for width in widths:
        cursor += width
        boundaries.append(int(round(cursor)))
    right = boundaries[-1]
    height = first_line + spacing * 8
    width = right + spacing * rng.randint(3, 5)
    image = np.zeros((height, width), dtype=np.uint8)

    staff_thickness = rng.choice((1, 1, 1, 2))
    for y in lines:
        cv2.line(image, (left, int(round(y))), (right, int(round(y))), 255, staff_thickness, cv2.LINE_AA)

    if rng.random() < 0.45:
        cv2.line(image, (left, int(lines[0])), (left, int(lines[-1])), 255, max(1, spacing // 10), cv2.LINE_AA)
    for index, x in enumerate(boundaries):
        _draw_barline(
            image,
            rng,
            x=x,
            top=int(lines[0]),
            bottom=int(lines[-1]),
            spacing=spacing,
            final=index == len(boundaries) - 1,
        )

    edges = [left, *boundaries]
    for measure_index, (a, b) in enumerate(zip(edges, edges[1:])):
        available = max(1, b - a)
        note_count = rng.randint(2, max(3, min(10, int(available / max(spacing * 2.2, 1)))))
        for _ in range(note_count):
            margin = max(spacing, int(available * 0.08))
            if b - a <= margin * 2:
                continue
            x = rng.randint(a + margin, b - margin)
            step = rng.randint(-3, 11)
            y = int(round(lines[0] + step * spacing / 2.0))
            _draw_note(image, rng, x=x, y=y, spacing=spacing)

        # Dense notation immediately beside a true boundary is a real failure mode for
        # fixed side-density rules.  It remains grouped with the same system identity.
        if measure_index < len(boundaries) and rng.random() < 0.28:
            boundary = boundaries[measure_index]
            side = rng.choice((-1, 1))
            x = boundary + side * rng.randint(max(2, spacing // 3), max(3, int(spacing * 0.9)))
            y = int(round(lines[rng.randrange(1, 4)] + rng.choice((-1, 0, 1)) * spacing / 2))
            _draw_note(image, rng, x=x, y=y, spacing=spacing, force_long=rng.random() < 0.25)

    # Hard negative stems and non-musical vertical marks are placed across the complete
    # system so proposal generation sees the runtime distribution, not chosen centres.
    for _ in range(rng.randint(1, max(3, measure_count))):
        x = rng.randint(left + spacing, right - spacing)
        y = int(round(lines[rng.randrange(5)] + rng.randint(-2, 2) * spacing / 2))
        _draw_note(image, rng, x=x, y=y, spacing=spacing, force_long=True)
    if rng.random() < 0.25:
        x = rng.randint(left + spacing, right - spacing)
        cv2.line(
            image,
            (x, max(0, int(lines[0] - spacing * rng.uniform(0.5, 1.8)))),
            (x + rng.choice((-2, -1, 1, 2)), min(height - 1, int(lines[-1] + spacing * rng.uniform(0.4, 1.6)))),
            255,
            1,
            cv2.LINE_AA,
        )
    if rng.random() < 0.30:
        text = rng.choice(("f", "ff", "p", "rit.", "1", "I"))
        x = rng.randint(left + spacing, max(left + spacing + 1, right - spacing * 3))
        y = min(height - 2, int(lines[-1] + spacing * rng.uniform(1.0, 2.2)))
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, spacing / 15.0, 255, max(1, spacing // 10), cv2.LINE_AA)

    system = StaffSystem(
        index=1,
        line_y=lines,
        top=max(0, int(lines[0] - spacing * 2)),
        bottom=min(height - 1, int(lines[-1] + spacing * 2)),
        left=left,
        right=right,
        spacing=float(spacing),
    )
    return image, system, tuple(boundaries), measure_count


def _degrade(base: np.ndarray, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    image = base.copy()
    if rng.random() < 0.70:
        sigma = rng.uniform(0.20, 1.25)
        kernel = max(3, int(round(sigma * 4)) | 1)
        image = cv2.GaussianBlur(image, (kernel, kernel), sigma)
    if rng.random() < 0.30:
        operation = cv2.dilate if rng.random() < 0.62 else cv2.erode
        kernel = np.ones((rng.choice((1, 2)), rng.choice((1, 2))), np.uint8)
        image = operation(image, kernel, iterations=1)
    if rng.random() < 0.24:
        # Sparse white/black defects emulate scan dust without moving geometry.
        count = rng.randint(1, max(2, image.size // 1400))
        for _ in range(count):
            y = rng.randrange(image.shape[0])
            x = rng.randrange(image.shape[1])
            cv2.circle(image, (x, y), rng.choice((0, 1, 1, 2)), rng.choice((0, 255)), -1)
    if rng.random() < 0.42:
        ok, encoded = cv2.imencode(
            ".jpg",
            255 - image,
            [cv2.IMWRITE_JPEG_QUALITY, rng.randint(35, 88)],
        )
        if ok:
            image = 255 - cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    threshold = rng.randint(52, 150)
    _, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return image


def render_group(group: int, seed: int, variants: int = 2) -> tuple[RenderedBarlineSystem, ...]:
    base, system, true_barlines, measure_count = _base_system(group, seed)
    rendered: list[RenderedBarlineSystem] = []
    for variant in range(variants):
        image = _degrade(base, seed + 104729 * (variant + 1))
        rendered.append(
            RenderedBarlineSystem(
                group=group,
                variant=variant,
                binary=image,
                system=system,
                true_barlines=true_barlines,
                measure_count=measure_count,
            )
        )
    return tuple(rendered)


def render_groups(seed: int, groups: int, variants: int = 2) -> list[RenderedBarlineSystem]:
    rng = random.Random(seed)
    result: list[RenderedBarlineSystem] = []
    for group in range(groups):
        group_seed = rng.randrange(1 << 31)
        result.extend(render_group(group, group_seed, variants))
    return result


def proposal_examples(rendered: RenderedBarlineSystem) -> tuple[ProposalExample, ...]:
    system = rendered.system
    y1 = max(0, int(round(system.line_y[0] - system.spacing * 1.35)))
    y2 = min(rendered.binary.shape[0], int(round(system.line_y[-1] + system.spacing * 1.35)))
    crop = rendered.binary[y1:y2, system.left:system.right + 1]
    tolerance = max(2, int(round(system.spacing * 0.52)))
    examples: list[ProposalExample] = []
    for start, end in _vertical_proposals(crop, system.spacing):
        absolute_start = start + system.left
        absolute_end = end + system.left
        x = int(round((absolute_start + absolute_end) / 2))
        label = int(any(abs(x - truth) <= tolerance for truth in rendered.true_barlines))
        examples.append(
            ProposalExample(
                group=rendered.group,
                variant=rendered.variant,
                x=x,
                x_start=absolute_start,
                x_end=absolute_end,
                label=label,
                features=extract_barline_features(
                    rendered.binary,
                    x_start=absolute_start,
                    x_end=absolute_end,
                    line_y=system.line_y,
                    spacing=system.spacing,
                ),
            )
        )
    return tuple(examples)


def flatten_examples(systems: Iterable[RenderedBarlineSystem]) -> list[ProposalExample]:
    result: list[ProposalExample] = []
    for rendered in systems:
        result.extend(proposal_examples(rendered))
    return result
