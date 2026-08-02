#!/usr/bin/env python3
"""Audit the geometric assignment ceiling of ScoreScan detector anchors.

This is a read-only diagnostic.  It evaluates the exact canonical RetinaNet
anchor sizes, ratios, FPN strides, and rounded base-anchor geometry against the
tile-local boxes seen by training/evaluation.  It does not estimate learned
accuracy and cannot authorize a model or a release.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.util import atomic_write_json, sha256_file, utc_now_iso  # noqa: E402


ANCHOR_SIZES = (
    (8.0, 12.0),
    (16.0, 24.0),
    (32.0, 48.0),
    (64.0, 96.0),
    (128.0, 192.0),
)
ANCHOR_ASPECT_RATIOS = (0.1, 0.3, 1.0, 3.0, 10.0)
EXPANDED_ASPECT_RATIOS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
FPN_STRIDES = (8.0, 16.0, 32.0, 64.0, 128.0)
P2_CANDIDATE_ANCHOR_SIZES = ((4.0, 6.0),) + ANCHOR_SIZES
P2_CANDIDATE_FPN_STRIDES = (4.0,) + FPN_STRIDES
IOU_THRESHOLDS = (0.25, 0.35, 0.4, 0.5, 0.7)
CANONICAL_MODEL_CONTRACT = (
    "retinanet-r50-fpnv2-music-anchors-groupnorm-giou-"
    "matcher35-25-nms75@3"
)


def _anchor_dimensions(size: float, aspect_ratio: float) -> tuple[float, float]:
    """Match torchvision AnchorGenerator's rounded base-anchor dimensions."""

    height_ratio = math.sqrt(aspect_ratio)
    width_ratio = 1.0 / height_ratio
    half_width = round((size * width_ratio) / 2.0)
    half_height = round((size * height_ratio) / 2.0)
    return float(2 * half_width), float(2 * half_height)


def anchor_dimensions(
    *,
    sizes_by_level: tuple[tuple[float, ...], ...] = ANCHOR_SIZES,
    strides: tuple[float, ...] = FPN_STRIDES,
    aspect_ratios: tuple[float, ...] = ANCHOR_ASPECT_RATIOS,
) -> tuple[tuple[float, float, float], ...]:
    if len(sizes_by_level) != len(strides):
        raise ValueError("anchor size and stride levels must match")
    anchors: list[tuple[float, float, float]] = []
    for stride, sizes in zip(strides, sizes_by_level, strict=True):
        for size in sizes:
            for ratio in aspect_ratios:
                width, height = _anchor_dimensions(size, ratio)
                if width > 0 and height > 0:
                    anchors.append((stride, width, height))
    return tuple(anchors)


def _iou(
    box: tuple[float, float, float, float],
    anchor: tuple[float, float, float, float],
) -> float:
    intersection_width = max(
        0.0,
        min(box[2], anchor[2]) - max(box[0], anchor[0]),
    )
    intersection_height = max(
        0.0,
        min(box[3], anchor[3]) - max(box[1], anchor[1]),
    )
    intersection = intersection_width * intersection_height
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    anchor_area = (anchor[2] - anchor[0]) * (anchor[3] - anchor[1])
    union = box_area + anchor_area - intersection
    return intersection / union if union > 0 else 0.0


def best_anchor_ious(
    box: Iterable[float],
    *,
    anchors: tuple[tuple[float, float, float], ...] | None = None,
) -> tuple[float, float]:
    """Return shape-only and actual-grid best IoU for one valid target box."""

    values = tuple(float(value) for value in box)
    if (
        len(values) != 4
        or not all(math.isfinite(value) for value in values)
        or values[2] <= values[0]
        or values[3] <= values[1]
    ):
        raise ValueError("target box must contain four finite increasing values")
    target = (values[0], values[1], values[2], values[3])
    center_x = (target[0] + target[2]) / 2.0
    center_y = (target[1] + target[3]) / 2.0
    centered_best = 0.0
    grid_best = 0.0
    for stride, width, height in anchors or anchor_dimensions():
        centered = (
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        )
        centered_best = max(centered_best, _iou(target, centered))
        base_x = math.floor(center_x / stride) * stride
        base_y = math.floor(center_y / stride) * stride
        for grid_y in (base_y, base_y + stride):
            for grid_x in (base_x, base_x + stride):
                anchor = (
                    grid_x - width / 2.0,
                    grid_y - height / 2.0,
                    grid_x + width / 2.0,
                    grid_y + height / 2.0,
                )
                grid_best = max(grid_best, _iou(target, anchor))
    return centered_best, grid_best


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_ious(values: list[float]) -> dict[str, object]:
    if not values:
        return {
            "objects": 0,
            "minimum": 0.0,
            "p10": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "mean": 0.0,
            "coverage": {
                f"iou_at_least_{threshold:.2f}": 0.0
                for threshold in IOU_THRESHOLDS
            },
        }
    return {
        "objects": len(values),
        "minimum": min(values),
        "p10": _quantile(values, 0.10),
        "median": _quantile(values, 0.50),
        "p90": _quantile(values, 0.90),
        "mean": sum(values) / len(values),
        "coverage": {
            f"iou_at_least_{threshold:.2f}": (
                sum(value >= threshold for value in values) / len(values)
            )
            for threshold in IOU_THRESHOLDS
        },
    }


def _category_names(path: Path) -> dict[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("classes") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("categories file has no classes")
    names: dict[int, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("category row is not an object")
        label = int(row.get("label", -1))
        name = str(row.get("name", "")).strip()
        if label <= 0 or not name or label in names:
            raise ValueError("category labels and names must be unique")
        names[label] = name
    return names


def audit_anchor_coverage(
    jsonl_path: Path,
    categories_path: Path,
    *,
    training_source_path: Path,
) -> dict[str, object]:
    category_names = _category_names(categories_path)
    centered_by_class: dict[int, list[float]] = defaultdict(list)
    grid_by_class: dict[int, list[float]] = defaultdict(list)
    p2_grid_by_class: dict[int, list[float]] = defaultdict(list)
    expanded_grid_by_class: dict[int, list[float]] = defaultdict(list)
    centered_all: list[float] = []
    grid_all: list[float] = []
    p2_grid_all: list[float] = []
    expanded_grid_all: list[float] = []
    p2_anchors = anchor_dimensions(
        sizes_by_level=P2_CANDIDATE_ANCHOR_SIZES,
        strides=P2_CANDIDATE_FPN_STRIDES,
    )
    expanded_anchors = anchor_dimensions(
        aspect_ratios=EXPANDED_ASPECT_RATIOS,
    )
    rows = 0
    with jsonl_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            objects = row.get("objects") if isinstance(row, dict) else None
            if not isinstance(objects, list):
                raise ValueError(f"row {line_number} has no object list")
            rows += 1
            for item in objects:
                if not isinstance(item, dict):
                    raise ValueError(f"row {line_number} has invalid object")
                label = int(item.get("label", -1))
                if label not in category_names:
                    raise ValueError(
                        f"row {line_number} uses unknown label {label}"
                    )
                centered_iou, grid_iou = best_anchor_ious(
                    item.get("box_xyxy", ())
                )
                _p2_centered_iou, p2_grid_iou = best_anchor_ious(
                    item.get("box_xyxy", ()),
                    anchors=p2_anchors,
                )
                _expanded_centered_iou, expanded_grid_iou = (
                    best_anchor_ious(
                        item.get("box_xyxy", ()),
                        anchors=expanded_anchors,
                    )
                )
                centered_by_class[label].append(centered_iou)
                grid_by_class[label].append(grid_iou)
                p2_grid_by_class[label].append(p2_grid_iou)
                expanded_grid_by_class[label].append(expanded_grid_iou)
                centered_all.append(centered_iou)
                grid_all.append(grid_iou)
                p2_grid_all.append(p2_grid_iou)
                expanded_grid_all.append(expanded_grid_iou)
    if rows <= 0 or not grid_all:
        raise ValueError("detector JSONL has no target objects")
    by_class = {
        category_names[label]: {
            "label": label,
            "centered_shape_ceiling": summarize_ious(
                centered_by_class[label]
            ),
            "grid_assignment_ceiling": summarize_ious(grid_by_class[label]),
            "hypothetical_p2_grid_assignment_ceiling": summarize_ious(
                p2_grid_by_class[label]
            ),
            "hypothetical_expanded_ratio_grid_assignment_ceiling": (
                summarize_ious(expanded_grid_by_class[label])
            ),
        }
        for label in sorted(grid_by_class)
    }
    return {
        "format": 1,
        "name": "scorescan-detector-anchor-coverage-audit-v1",
        "created_at": utc_now_iso(),
        "diagnostic_only": True,
        "release_authorized": False,
        "model_contract": CANONICAL_MODEL_CONTRACT,
        "anchor_sizes": [list(values) for values in ANCHOR_SIZES],
        "anchor_aspect_ratios": list(ANCHOR_ASPECT_RATIOS),
        "fpn_strides": list(FPN_STRIDES),
        "hypothetical_p2_candidate": {
            "diagnostic_only": True,
            "anchor_sizes": [
                list(values) for values in P2_CANDIDATE_ANCHOR_SIZES
            ],
            "fpn_strides": list(P2_CANDIDATE_FPN_STRIDES),
        },
        "hypothetical_expanded_ratio_candidate": {
            "diagnostic_only": True,
            "anchor_aspect_ratios": list(EXPANDED_ASPECT_RATIOS),
            "head_anchor_count_multiplier": (
                len(EXPANDED_ASPECT_RATIOS)
                / len(ANCHOR_ASPECT_RATIOS)
            ),
        },
        "foreground_iou_threshold": 0.35,
        "background_iou_threshold": 0.25,
        "input": {
            "jsonl": str(jsonl_path.resolve()),
            "jsonl_sha256": sha256_file(jsonl_path),
            "categories": str(categories_path.resolve()),
            "categories_sha256": sha256_file(categories_path),
            "training_source": str(training_source_path.resolve()),
            "training_source_sha256": sha256_file(training_source_path),
            "rows": rows,
            "objects": len(grid_all),
        },
        "overall": {
            "centered_shape_ceiling": summarize_ious(centered_all),
            "grid_assignment_ceiling": summarize_ious(grid_all),
            "hypothetical_p2_grid_assignment_ceiling": summarize_ious(
                p2_grid_all
            ),
            "hypothetical_expanded_ratio_grid_assignment_ceiling": (
                summarize_ious(expanded_grid_all)
            ),
        },
        "classes": by_class,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--categories", type=Path, required=True)
    parser.add_argument(
        "--training-source",
        type=Path,
        default=ROOT / "tools" / "train_deepscores_symbol_detector.py",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_anchor_coverage(
        args.jsonl.resolve(),
        args.categories.resolve(),
        training_source_path=args.training_source.resolve(),
    )
    if args.output is not None:
        atomic_write_json(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
