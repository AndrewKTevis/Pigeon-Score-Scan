from __future__ import annotations

"""Bounded, deterministic local evidence extraction for symbol safety gates.

The page-recognition boundary stores one small staff-normalised PNG per measure.  This
module is the only place that decodes that payload and maps a semantic event into the
fixed image.  Symbol-specific guards receive immutable numeric descriptors and cannot
access the original page or create notation.
"""

import base64
import math
from fractions import Fraction
from functools import lru_cache

import cv2
import numpy as np

from .score_ir import MeasureIR, NoteIR
from .visual_evidence import (
    RHYTHM_GUARD_HEIGHT,
    RHYTHM_GUARD_WIDTH,
    SYMBOL_GUARD_HEIGHT,
    SYMBOL_GUARD_WIDTH,
    staff_position_for_note,
)

MAX_ENCODED_GUARD_BYTES = 65_536
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@lru_cache(maxsize=512)
def decode_bounded_png(encoded: str, width: int, height: int) -> np.ndarray | None:
    """Decode one fixed-size evidence PNG under hard allocation bounds."""
    if not encoded or len(encoded) > MAX_ENCODED_GUARD_BYTES:
        return None
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError):
        return None
    if (
        len(raw) < 24
        or len(raw) > MAX_ENCODED_GUARD_BYTES
        or raw[:8] != PNG_SIGNATURE
        or int.from_bytes(raw[16:20], "big") != int(width)
        or int.from_bytes(raw[20:24], "big") != int(height)
    ):
        return None
    try:
        values = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(values, cv2.IMREAD_GRAYSCALE)
    except cv2.error:
        return None
    if image is None or image.shape != (int(height), int(width)):
        return None
    image.setflags(write=False)
    return image


def decode_guard_image(encoded: str) -> np.ndarray | None:
    return decode_bounded_png(encoded, RHYTHM_GUARD_WIDTH, RHYTHM_GUARD_HEIGHT)


def decode_symbol_guard_image(encoded: str) -> np.ndarray | None:
    return decode_bounded_png(encoded, SYMBOL_GUARD_WIDTH, SYMBOL_GUARD_HEIGHT)


def event_position(measure: MeasureIR, note: NoteIR) -> tuple[float, float]:
    """Map one semantic event into the staff-normalised measure evidence image."""
    anchors = [item for item in measure.notes if not item.grace and not item.chord]
    expected = measure.expected_duration
    if expected is None or expected <= 0:
        expected = max(
            (item.onset + max(item.duration, Fraction(0, 1)) for item in anchors),
            default=Fraction(1, 1),
        )
    expected = max(expected, Fraction(1, 64))
    onset_ratio = max(0.0, min(1.0, float(note.onset / expected)))
    x_ratio = 0.075 + 0.85 * onset_ratio
    if note.rest or note.pitch is None:
        staff_position = 4.0
    else:
        value = staff_position_for_note(note, measure.clef)
        staff_position = 4.0 if value is None else max(-4.0, min(12.0, value))
    y_ratio = (12.0 - staff_position) / 16.0
    return x_ratio, max(0.0, min(1.0, y_ratio))


def _pool_exact(values: np.ndarray, output_height: int, output_width: int) -> np.ndarray:
    if values.ndim != 2 or output_height <= 0 or output_width <= 0:
        raise ValueError("invalid local descriptor shape")
    height, width = values.shape
    if height % output_height or width % output_width:
        raise ValueError("local patch dimensions must divide descriptor dimensions exactly")
    block_h = height // output_height
    block_w = width // output_width
    return values.reshape(output_height, block_h, output_width, block_w).mean(axis=(1, 3))


def local_descriptor(
    image: np.ndarray,
    x_ratio: float,
    y_ratio: float,
    *,
    patch_width: int,
    patch_height: int,
    descriptor_width: int,
    descriptor_height: int,
    x_offset_pixels: int = 0,
    y_offset_pixels: int = 0,
) -> tuple[float, ...]:
    """Return density and deterministic integer-Sobel descriptors for one local patch."""
    if image.ndim != 2 or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("invalid guard image shape")
    if patch_width <= 0 or patch_height <= 0 or patch_width % 2 or patch_height % 2:
        raise ValueError("local patch dimensions must be positive even integers")
    if patch_width % descriptor_width or patch_height % descriptor_height:
        raise ValueError("descriptor dimensions must exactly divide the patch")

    image_height, image_width = image.shape
    centre_x = int(round(_unit(x_ratio) * (image_width - 1))) + int(x_offset_pixels)
    centre_y = int(round(_unit(y_ratio) * (image_height - 1))) + int(y_offset_pixels)
    half_w = patch_width // 2
    half_h = patch_height // 2
    padded = np.pad(
        image,
        ((half_h, half_h), (half_w, half_w)),
        mode="constant",
        constant_values=0,
    )
    start_y = centre_y
    start_x = centre_x
    patch = padded[start_y : start_y + patch_height, start_x : start_x + patch_width]
    if patch.shape != (patch_height, patch_width):
        pixel_count = descriptor_width * descriptor_height
        return tuple(0.0 for _ in range(pixel_count * 2))

    patch64 = patch.astype(np.float64)
    density = _pool_exact(patch64, descriptor_height, descriptor_width) / 255.0
    bordered = np.pad(patch64, ((1, 1), (1, 1)), mode="reflect")
    gx = (
        -bordered[:-2, :-2]
        + bordered[:-2, 2:]
        - 2.0 * bordered[1:-1, :-2]
        + 2.0 * bordered[1:-1, 2:]
        - bordered[2:, :-2]
        + bordered[2:, 2:]
    )
    gy = (
        -bordered[:-2, :-2]
        - 2.0 * bordered[:-2, 1:-1]
        - bordered[:-2, 2:]
        + bordered[2:, :-2]
        + 2.0 * bordered[2:, 1:-1]
        + bordered[2:, 2:]
    )
    magnitude = np.sqrt(gx * gx + gy * gy)
    maximum = float(np.max(magnitude, initial=0.0))
    if maximum > 0.0 and math.isfinite(maximum):
        magnitude /= maximum
    gradient = _pool_exact(magnitude, descriptor_height, descriptor_width)
    return tuple(float(value) for value in density.ravel()) + tuple(
        float(value) for value in gradient.ravel()
    )
