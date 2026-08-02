from __future__ import annotations

"""Dependency-free page-tile fragment fusion shared by runtime and release QA."""

from dataclasses import dataclass
import math
import statistics
from typing import Callable

from .semantic_detector_contract import (
    SEMANTIC_DETECTOR_MAXIMUM_SCALE,
    SEMANTIC_DETECTOR_MINIMUM_SCALE,
    SEMANTIC_DETECTOR_TARGET_STAFF_SPACING,
    TILE_FRAGMENT_FUSION_CLASSES,
    TILE_FRAGMENT_STAFF_AGNOSTIC_CLASSES,
)

PAGE_LAYOUT_EVIDENCE_VERSION = "scorescan-semantic-page-layout-evidence@1"
PAGE_LAYOUT_EVIDENCE_BUILDER_VERSION = (
    "hash-bound-product-layout-process-pool@2"
)


def semantic_page_scale(spacings: list[float]) -> float:
    valid = [
        float(value)
        for value in spacings
        if math.isfinite(float(value)) and float(value) > 0
    ]
    if not valid:
        raise ValueError("semantic detector layout has no valid staff spacing")
    scale = (
        SEMANTIC_DETECTOR_TARGET_STAFF_SPACING
        / float(statistics.median(valid))
    )
    bounded = max(
        SEMANTIC_DETECTOR_MINIMUM_SCALE,
        min(SEMANTIC_DETECTOR_MAXIMUM_SCALE, scale),
    )
    return 1.0 if abs(bounded - 1.0) < 0.01 else bounded


def scaled_page_dimension(length: int, scale: float) -> int:
    if length <= 0 or not math.isfinite(scale) or scale <= 0:
        raise ValueError("semantic detector scaled page dimension is invalid")
    return max(1, int(round(length * scale)))


def semantic_tile_origins(
    length: int,
    size: int,
    overlap: int,
) -> list[int]:
    if length <= size:
        return [0]
    step = size - overlap
    values = list(range(0, max(1, length - size + 1), step))
    final = length - size
    if values[-1] != final:
        values.append(final)
    return values


def source_tile_bbox(
    *,
    x: int,
    y: int,
    valid_width: int,
    valid_height: int,
    scale: float,
    source_width: int,
    source_height: int,
) -> tuple[int, int, int, int]:
    return (
        max(0, int(math.floor(x / scale))),
        max(0, int(math.floor(y / scale))),
        min(
            source_width,
            int(math.ceil((x + valid_width) / scale)),
        ),
        min(
            source_height,
            int(math.ceil((y + valid_height) / scale)),
        ),
    )


@dataclass(frozen=True)
class TileFragmentDetection:
    class_name: str
    label: int
    bbox: tuple[int, int, int, int]
    confidence: float
    staff_index: int
    placement: str
    tile_id: int
    tile_bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class FusedFragmentDetection:
    class_name: str
    label: int
    bbox: tuple[int, int, int, int]
    confidence: float
    staff_index: int
    placement: str
    source_tile_count: int


def _interval_overlap_ratio(
    left: tuple[int, int],
    right: tuple[int, int],
) -> float:
    overlap = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    return overlap / max(1, min(left[1] - left[0], right[1] - right[0]))


def _fragment_axis_match_cost(
    left: TileFragmentDetection,
    right: TileFragmentDetection,
    *,
    horizontal: bool,
) -> float | None:
    if horizontal:
        if left.tile_bbox[0] == right.tile_bbox[0]:
            return None
        before, after = (
            (left, right)
            if left.tile_bbox[0] < right.tile_bbox[0]
            else (right, left)
        )
        if (
            after.tile_bbox[0] >= before.tile_bbox[2]
            or _interval_overlap_ratio(
                (before.tile_bbox[1], before.tile_bbox[3]),
                (after.tile_bbox[1], after.tile_bbox[3]),
            )
            <= 0
        ):
            return None
        tile_span = min(
            before.tile_bbox[2] - before.tile_bbox[0],
            after.tile_bbox[2] - after.tile_bbox[0],
        )
        margin = max(4, int(math.ceil(tile_span * 0.04)))
        if (
            before.bbox[2] < before.tile_bbox[2] - margin
            or after.bbox[0] > after.tile_bbox[0] + margin
        ):
            return None
        before_orth = (before.bbox[1], before.bbox[3])
        after_orth = (after.bbox[1], after.bbox[3])
        primary_gap = max(
            0,
            max(before.bbox[0], after.bbox[0])
            - min(before.bbox[2], after.bbox[2]),
        )
    else:
        if left.tile_bbox[1] == right.tile_bbox[1]:
            return None
        before, after = (
            (left, right)
            if left.tile_bbox[1] < right.tile_bbox[1]
            else (right, left)
        )
        if (
            after.tile_bbox[1] >= before.tile_bbox[3]
            or _interval_overlap_ratio(
                (before.tile_bbox[0], before.tile_bbox[2]),
                (after.tile_bbox[0], after.tile_bbox[2]),
            )
            <= 0
        ):
            return None
        tile_span = min(
            before.tile_bbox[3] - before.tile_bbox[1],
            after.tile_bbox[3] - after.tile_bbox[1],
        )
        margin = max(4, int(math.ceil(tile_span * 0.04)))
        if (
            before.bbox[3] < before.tile_bbox[3] - margin
            or after.bbox[1] > after.tile_bbox[1] + margin
        ):
            return None
        before_orth = (before.bbox[0], before.bbox[2])
        after_orth = (after.bbox[0], after.bbox[2])
        primary_gap = max(
            0,
            max(before.bbox[1], after.bbox[1])
            - min(before.bbox[3], after.bbox[3]),
        )
    orth_overlap = _interval_overlap_ratio(before_orth, after_orth)
    if orth_overlap < 0.55 or primary_gap > margin:
        return None
    before_size = max(1, before_orth[1] - before_orth[0])
    after_size = max(1, after_orth[1] - after_orth[0])
    before_centre = 0.5 * (before_orth[0] + before_orth[1])
    after_centre = 0.5 * (after_orth[0] + after_orth[1])
    return (
        abs(before_centre - after_centre) / min(before_size, after_size)
        + abs(math.log(before_size / after_size))
        + primary_gap / margin
    )


def _tile_fragment_match_cost(
    left: TileFragmentDetection,
    right: TileFragmentDetection,
) -> float | None:
    if (
        left.tile_id == right.tile_id
        or left.class_name != right.class_name
        or left.class_name not in TILE_FRAGMENT_FUSION_CLASSES
        or (
            left.staff_index != right.staff_index
            and left.class_name not in TILE_FRAGMENT_STAFF_AGNOSTIC_CLASSES
        )
    ):
        return None
    costs = [
        value
        for value in (
            _fragment_axis_match_cost(left, right, horizontal=True),
            _fragment_axis_match_cost(left, right, horizontal=False),
        )
        if value is not None
    ]
    return min(costs) if costs else None


def fuse_tile_fragments(
    detections: list[TileFragmentDetection],
    *,
    owner_resolver: Callable[
        [tuple[int, int, int, int]],
        tuple[int, str] | None,
    ]
    | None = None,
) -> list[FusedFragmentDetection]:
    """Fuse boundary fragments while preserving independently nested objects."""

    if not detections:
        return []
    parent = list(range(len(detections)))
    component_tiles = [{item.tile_id} for item in detections]

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    edge_groups: dict[
        tuple[int, int, str, int],
        list[tuple[float, int, int]],
    ] = {}
    for left_index, left in enumerate(detections):
        for right_index in range(left_index + 1, len(detections)):
            right = detections[right_index]
            cost = _tile_fragment_match_cost(left, right)
            if cost is None:
                continue
            tile_pair = tuple(sorted((left.tile_id, right.tile_id)))
            key = (
                tile_pair[0],
                tile_pair[1],
                left.class_name,
                (
                    -1
                    if left.class_name in TILE_FRAGMENT_STAFF_AGNOSTIC_CLASSES
                    else left.staff_index
                ),
            )
            edge_groups.setdefault(key, []).append(
                (cost, left_index, right_index)
            )

    for edges in edge_groups.values():
        used_left: set[int] = set()
        used_right: set[int] = set()
        for _cost, left_index, right_index in sorted(edges):
            if left_index in used_left or right_index in used_right:
                continue
            left_root = root(left_index)
            right_root = root(right_index)
            if (
                left_root == right_root
                or component_tiles[left_root] & component_tiles[right_root]
            ):
                continue
            if len(component_tiles[left_root]) < len(component_tiles[right_root]):
                left_root, right_root = right_root, left_root
            parent[right_root] = left_root
            component_tiles[left_root].update(component_tiles[right_root])
            used_left.add(left_index)
            used_right.add(right_index)

    components: dict[int, list[TileFragmentDetection]] = {}
    for index, item in enumerate(detections):
        components.setdefault(root(index), []).append(item)
    fused: list[FusedFragmentDetection] = []
    for component in components.values():
        best = min(
            component,
            key=lambda item: (
                -item.confidence,
                item.bbox,
                item.tile_id,
            ),
        )
        bbox = (
            min(item.bbox[0] for item in component),
            min(item.bbox[1] for item in component),
            max(item.bbox[2] for item in component),
            max(item.bbox[3] for item in component),
        )
        owner = owner_resolver(bbox) if owner_resolver is not None else None
        fused.append(
            FusedFragmentDetection(
                best.class_name,
                best.label,
                bbox,
                min(item.confidence for item in component),
                owner[0] if owner is not None else best.staff_index,
                owner[1] if owner is not None else best.placement,
                len(component),
            )
        )
    return fused
