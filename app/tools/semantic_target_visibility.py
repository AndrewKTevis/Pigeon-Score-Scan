from __future__ import annotations

"""Shared visibility rules for complete-page semantic detector targets."""

import math
from typing import Sequence


OVERSIZED_FRAGMENT_VISIBILITY_VERSION = (
    "complete-page-oversized-axis-overlap-fragments@1"
)


def intersection_box(
    box: Sequence[float],
    crop: Sequence[float],
) -> tuple[float, float, float, float]:
    if len(box) != 4 or len(crop) != 4:
        raise ValueError("semantic target box/crop must have four coordinates")
    values = tuple(float(value) for value in (*box, *crop))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("semantic target box/crop must be finite")
    left, top, right, bottom = values[:4]
    crop_left, crop_top, crop_right, crop_bottom = values[4:]
    if right <= left or bottom <= top:
        raise ValueError("semantic target box must have positive area")
    if crop_right <= crop_left or crop_bottom <= crop_top:
        raise ValueError("semantic target crop must have positive area")
    return (
        max(left, crop_left),
        max(top, crop_top),
        min(right, crop_right),
        min(bottom, crop_bottom),
    )


def target_fragment_is_visible(
    box: Sequence[float],
    crop: Sequence[float],
    *,
    minimum_fraction: float,
    long_span_minimum_fraction: float,
    is_long_span: bool,
    tile_overlap: float,
) -> bool:
    """Return whether a crop has an unambiguous trainable target fragment.

    Ordinary objects retain the area-fraction contract. A long object that is
    physically larger than the crop on one or both axes cannot satisfy that
    contract. Such an object is retained only when every oversized axis has at
    least one full overlap-width of ink and every compact axis still satisfies
    the long-span fraction. The overlap-width proof prevents one-pixel edge
    slivers while guaranteeing that omitted endpoint slivers are covered by an
    adjacent overlapping tile.
    """

    if not (
        math.isfinite(minimum_fraction)
        and math.isfinite(long_span_minimum_fraction)
        and 0 < long_span_minimum_fraction <= minimum_fraction <= 1
    ):
        raise ValueError("semantic target visibility fractions are invalid")
    if not math.isfinite(tile_overlap) or tile_overlap <= 0:
        raise ValueError("semantic target tile overlap must be positive")
    intersection = intersection_box(box, crop)
    intersection_width = max(0.0, intersection[2] - intersection[0])
    intersection_height = max(0.0, intersection[3] - intersection[1])
    if intersection_width <= 0 or intersection_height <= 0:
        return False

    object_width = float(box[2]) - float(box[0])
    object_height = float(box[3]) - float(box[1])
    crop_width = float(crop[2]) - float(crop[0])
    crop_height = float(crop[3]) - float(crop[1])
    required_fraction = (
        long_span_minimum_fraction if is_long_span else minimum_fraction
    )
    visible_fraction = (
        intersection_width * intersection_height
        / (object_width * object_height)
    )
    if visible_fraction + 1e-9 >= required_fraction:
        return True
    if not is_long_span:
        return False

    object_spans = (object_width, object_height)
    crop_spans = (crop_width, crop_height)
    intersection_spans = (intersection_width, intersection_height)
    oversized_axes = [
        index
        for index, (object_span, crop_span) in enumerate(
            zip(object_spans, crop_spans, strict=True)
        )
        if object_span > crop_span + 1e-9
    ]
    if not oversized_axes:
        return False
    for index, (object_span, crop_span, visible_span) in enumerate(
        zip(
            object_spans,
            crop_spans,
            intersection_spans,
            strict=True,
        )
    ):
        if index in oversized_axes:
            if visible_span + 1e-9 < min(tile_overlap, crop_span):
                return False
        elif visible_span / object_span + 1e-9 < long_span_minimum_fraction:
            return False
    return True
