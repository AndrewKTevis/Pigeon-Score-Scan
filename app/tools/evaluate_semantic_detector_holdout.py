#!/usr/bin/env python3
"""Evaluate a trained semantic detector on a forbidden-to-train scan holdout."""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from app.tools.dense_detection_metrics import (
    COCO_IOU_THRESHOLDS,
    DENSE_MAP_VERSION,
    compute_dense_detection_metrics as _compute_dense_detection_metrics,
)
from app.tools.expand_overlapping_semantic_targets import TRANSFORMATION_VERSION
from app.tools.muse_omr_contract import (
    BENCHMARK_SELECTION_ROLE,
    MAXIMUM_SCAN_PAGE_ASPECT_RATIO,
    SCAN_PAGE_SHAPE_CONTRACT,
    TRAINING_REGION_ROLE,
    scan_page_aspect_ratio,
)
from app.tools.prepare_openscore_svg_regions import (
    COMPLETE_PAGE_TARGET_PROVENANCE,
    LONG_SPAN_SEMANTIC_CATEGORIES,
)
from app.tools.semantic_target_visibility import (
    OVERSIZED_FRAGMENT_VISIBILITY_VERSION,
    target_fragment_is_visible,
)
from app.tools.train_deepscores_symbol_detector import (
    PRIORITY_SELECTION_PROTOCOL,
    assert_compatible_category_manifests,
    build_detector_model,
    category_label_name_map,
    detector_model_contract,
    insufficient_required_class_support,
    is_priority_mark_class,
    load_jsonl,
    load_grayscale_crop,
    normalize_target_boxes,
    parse_required_class_maps,
    priority_selection_score,
    resolve_detector_device,
    sha256_file,
    support_filtered_macro_map,
)
from scorescan.semantic_detector_contract import (
    CALIBRATED_OPERATING_POINT_SELECTION_METHOD,
    FIXED_RARE_CLASS_OPERATING_POINT_THRESHOLD,
    FIXED_RARE_CLASS_SELECTION_METHOD,
    HIGH_RECALL_MARK_CLASSES,
    MINIMUM_HIGH_RECALL_MARK_RECALL,
    MINIMUM_OPERATING_POINT_RECALL,
    SEMANTIC_DETECTOR_INPUT_SIZE,
    SEMANTIC_DETECTOR_MAXIMUM_SCALE,
    SEMANTIC_DETECTOR_MAXIMUM_TILES,
    SEMANTIC_DETECTOR_MINIMUM_SCALE,
    SEMANTIC_DETECTOR_TARGET_STAFF_SPACING,
    SEMANTIC_DETECTOR_TILE_OVERLAP,
    SEMANTIC_PAGE_NMS_IOU,
    SUPPORTED_RUNTIME_CLASSES,
    TILE_FRAGMENT_FUSION_VERSION,
)
from scorescan.semantic_tile_fusion import (
    PAGE_LAYOUT_EVIDENCE_BUILDER_VERSION,
    PAGE_LAYOUT_EVIDENCE_VERSION,
    TileFragmentDetection,
    fuse_tile_fragments,
    scaled_page_dimension,
    semantic_page_scale,
    semantic_tile_origins,
    source_tile_bbox,
)

DEPLOYMENT_OPERATING_POINT_CLASSES = tuple(sorted(SUPPORTED_RUNTIME_CLASSES))
PAGE_STITCHING_VERSION = (
    "runtime-staffnorm-layout-page-nms-complete-page-fragment-fusion@5"
)
RUNTIME_PAGE_TILING_VERSION = (
    "staff-normalized-complete-page-runtime-retiling@2"
)


def minimum_recall_for_class(
    class_name: str,
    *,
    minimum_recall: float,
    minimum_high_recall_mark_recall: float,
) -> float:
    if class_name in HIGH_RECALL_MARK_CLASSES:
        return max(minimum_recall, minimum_high_recall_mark_recall)
    return minimum_recall


def acceptance_failures(
    metrics: dict[str, Any],
    *,
    minimum_map_50: float,
    minimum_map_75: float,
    minimum_priority_map: float,
    required_class_maps: dict[str, float],
    operating_points: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    failures = []
    for name, value, floor in (
        ("map_50", float(metrics.get("map_50", -1.0)), minimum_map_50),
        ("map_75", float(metrics.get("map_75", -1.0)), minimum_map_75),
        (
            "priority_mark_map",
            float(metrics.get("priority_mark_map", -1.0)),
            minimum_priority_map,
        ),
    ):
        if not math.isfinite(value) or value < floor:
            failures.append(f"{name}={value:.6f}<{floor:.6f}")
    named = metrics.get("map_per_class_named")
    if not isinstance(named, dict):
        named = {}
    for class_name, floor in sorted(required_class_maps.items()):
        value = float(named.get(class_name, -1.0))
        if not math.isfinite(value) or value < floor:
            failures.append(
                f"class:{class_name}={value:.6f}<{floor:.6f}"
            )
    if operating_points is not None:
        for class_name, point in sorted(operating_points.items()):
            if not isinstance(point, dict) or point.get("passed") is not True:
                failures.append(f"operating_point:{class_name}")
    return failures


def _box_iou(left: Any, right: Any) -> float:
    import numpy as np

    left = np.asarray(left, dtype=np.float64).reshape(4)
    right = np.asarray(right, dtype=np.float64).reshape(4)
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0.0:
        return 0.0
    left_area = max(
        1e-9,
        (float(left[2]) - float(left[0]))
        * (float(left[3]) - float(left[1])),
    )
    right_area = max(
        1e-9,
        (float(right[2]) - float(right[0]))
        * (float(right[3]) - float(right[1])),
    )
    return intersection / max(left_area + right_area - intersection, 1e-9)


def unique_class_counts(rows: list[dict[str, Any]]) -> dict[int, int]:
    """Count source objects once, independent of overlapping tile instances."""

    labels_by_id: dict[tuple[tuple[str, str, str], str], int] = {}
    for row in rows:
        page_key = _page_key(row)
        if not all(page_key):
            raise ValueError("semantic target has no stable page identity")
        for obj in row.get("objects", []):
            object_id = str(obj.get("source_object_id") or "").strip()
            if not object_id:
                raise ValueError(
                    "overlap-consistent semantic target has no source_object_id"
                )
            if (
                obj.get("target_geometry_provenance")
                != COMPLETE_PAGE_TARGET_PROVENANCE
            ):
                raise ValueError(
                    "complete-page semantic target has invalid geometry provenance"
                )
            label = int(obj["label"])
            previous = labels_by_id.setdefault(
                (page_key, object_id),
                label,
            )
            if previous != label:
                raise ValueError(
                    "semantic source_object_id has inconsistent class labels"
                )
    result: dict[int, int] = {}
    for label in labels_by_id.values():
        result[label] = result.get(label, 0) + 1
    return result


def _page_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("source_key") or ""),
        str(row.get("image") or ""),
        str(row.get("image_id") or ""),
    )


def load_page_layout_evidence(
    rows: list[dict[str, Any]],
    report_path: Path,
    *,
    images_dir: Path,
    prepared_manifest_sha256: str,
    split_jsonl_sha256: dict[str, str],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Load hash-bound layouts produced by the exact Windows product analyzer."""

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    layout_source_path = (
        Path(__file__).parents[1] / "src" / "scorescan" / "layout.py"
    )
    builder_source_path = (
        Path(__file__).parent / "prepare_semantic_page_layout_evidence.py"
    )
    if (
        not isinstance(payload, dict)
        or payload.get("version") != PAGE_LAYOUT_EVIDENCE_VERSION
        or payload.get("passed") is not True
        or payload.get("prepared_manifest_sha256")
        != prepared_manifest_sha256
        or payload.get("split_jsonl_sha256")
        != split_jsonl_sha256
        or payload.get("product_layout_source_sha256")
        != sha256_file(layout_source_path)
        or payload.get("builder_version")
        != PAGE_LAYOUT_EVIDENCE_BUILDER_VERSION
        or payload.get("scan_page_shape_contract")
        != SCAN_PAGE_SHAPE_CONTRACT
        or float(
            payload.get("maximum_scan_page_aspect_ratio", -1)
        )
        != MAXIMUM_SCAN_PAGE_ASPECT_RATIO
        or payload.get("builder_source_sha256")
        != sha256_file(builder_source_path)
        or not isinstance(payload.get("pages"), list)
    ):
        raise ValueError("semantic page-layout evidence is stale or invalid")
    image_root = images_dir.resolve()
    image_hashes: dict[Path, str] = {}
    layouts: dict[tuple[str, str, str], dict[str, Any]] = {}
    for page in payload["pages"]:
        if not isinstance(page, dict) or not isinstance(
            page.get("layout"),
            dict,
        ):
            raise ValueError("semantic page-layout evidence row is invalid")
        key = (
            str(page.get("source_key") or ""),
            str(page.get("image") or ""),
            str(page.get("image_id") or ""),
        )
        systems = page["layout"].get("systems")
        if (
            not all(key)
            or key in layouts
            or not isinstance(systems, list)
            or not systems
            or not isinstance(page.get("image_sha256"), str)
            or len(page["image_sha256"]) != 64
        ):
            raise ValueError("semantic page-layout evidence is incomplete")
        image_path = (image_root / key[1]).resolve()
        try:
            image_path.relative_to(image_root)
        except ValueError as exc:
            raise ValueError(
                "semantic page-layout evidence escapes its image root"
            ) from exc
        if not image_path.is_file():
            raise ValueError(
                "semantic page-layout evidence image is missing"
            )
        image_hash = image_hashes.get(image_path)
        if image_hash is None:
            image_hash = sha256_file(image_path)
            image_hashes[image_path] = image_hash
        if page["image_sha256"] != image_hash:
            raise ValueError(
                "semantic page-layout evidence image hash is stale"
            )
        layouts[key] = page["layout"]
    expected = {_page_key(row) for row in rows}
    if set(layouts) != expected:
        raise ValueError("semantic page-layout evidence does not cover its rows")
    return layouts


def _assign_bbox_to_layout(
    layout: dict[str, Any],
    bbox: tuple[int, int, int, int],
) -> tuple[int, str] | None:
    systems = layout.get("systems")
    if not isinstance(systems, list) or not systems:
        return None
    center_y = 0.5 * (bbox[1] + bbox[3])
    scored: list[tuple[float, int, int, str]] = []
    for sequence_index, staff in enumerate(systems):
        if not isinstance(staff, dict):
            return None
        lines = staff.get("line_y")
        if not isinstance(lines, list) or len(lines) != 5:
            return None
        top_line = float(lines[0])
        bottom_line = float(lines[-1])
        spacing = max(1.0, float(staff.get("spacing", 0.0)))
        if center_y < top_line:
            distance = top_line - center_y
            placement = "above"
        elif center_y > bottom_line:
            distance = center_y - bottom_line
            placement = "below"
        else:
            midpoint = 0.5 * (top_line + bottom_line)
            distance = 0.0
            placement = "above" if center_y <= midpoint else "below"
        scored.append(
            (
                distance / spacing,
                sequence_index,
                int(staff.get("index", 0)),
                placement,
            )
        )
    distance_spaces, _sequence_index, staff_index, placement = min(scored)
    if distance_spaces > 8.0 or staff_index <= 0:
        return None
    return staff_index, placement


def retile_complete_page_rows_for_runtime(
    rows: list[dict[str, Any]],
    layouts_by_page: dict[tuple[str, str, str], dict[str, Any]],
    *,
    minimum_visible_fraction: float,
    long_span_minimum_visible_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rebuild exact runtime tiles from immutable complete-page targets."""

    if not 0 < long_span_minimum_visible_fraction <= minimum_visible_fraction <= 1:
        raise ValueError("semantic runtime retiling visibility floors are invalid")
    pages: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = _page_key(row)
        page = pages.setdefault(
            key,
            {
                "template": row,
                "objects": {},
            },
        )
        for obj in row.get("objects", []):
            object_id = str(obj.get("source_object_id") or "").strip()
            page_box = obj.get("page_box_xyxy")
            if (
                not object_id
                or not isinstance(page_box, list)
                or len(page_box) != 4
                or obj.get("target_geometry_provenance")
                != COMPLETE_PAGE_TARGET_PROVENANCE
            ):
                raise ValueError(
                    "runtime retiling requires complete-page object geometry"
                )
            stable = (
                int(obj["label"]),
                str(obj.get("category_id") or ""),
                tuple(float(value) for value in page_box),
            )
            previous = page["objects"].setdefault(
                object_id,
                (stable, obj),
            )
            if previous[0] != stable:
                raise ValueError(
                    "runtime retiling found unstable page object geometry"
                )

    if set(pages) != set(layouts_by_page):
        raise ValueError("runtime retiling layouts do not cover dataset pages")
    runtime_rows: list[dict[str, Any]] = []
    scale_values: list[float] = []
    page_tile_counts: list[int] = []
    target_instances = 0
    unique_targets = 0
    assigned_source_targets: set[
        tuple[tuple[str, str, str], str]
    ] = set()
    for key in sorted(pages):
        page = pages[key]
        layout = layouts_by_page[key]
        source_width = int(layout.get("width", 0))
        source_height = int(layout.get("height", 0))
        systems = layout.get("systems")
        if (
            source_width <= 0
            or source_height <= 0
            or not isinstance(systems, list)
            or not systems
        ):
            raise ValueError("runtime retiling layout dimensions are invalid")
        page_aspect_ratio = scan_page_aspect_ratio(
            source_width,
            source_height,
        )
        if page_aspect_ratio > MAXIMUM_SCAN_PAGE_ASPECT_RATIO:
            raise ValueError(
                "runtime retiling page violates "
                f"{SCAN_PAGE_SHAPE_CONTRACT}: "
                f"source_key={key[0]!r}, image={key[1]!r}, "
                f"image_id={key[2]!r}, "
                f"dimensions={source_width}x{source_height}, "
                f"aspect_ratio={page_aspect_ratio:.6f}, "
                f"maximum={MAXIMUM_SCAN_PAGE_ASPECT_RATIO:.6f}"
            )
        scale = semantic_page_scale(
            [float(system.get("spacing", 0.0)) for system in systems]
        )
        scaled_width = scaled_page_dimension(source_width, scale)
        scaled_height = scaled_page_dimension(source_height, scale)
        x_origins = semantic_tile_origins(
            scaled_width,
            SEMANTIC_DETECTOR_INPUT_SIZE,
            SEMANTIC_DETECTOR_TILE_OVERLAP,
        )
        y_origins = semantic_tile_origins(
            scaled_height,
            SEMANTIC_DETECTOR_INPUT_SIZE,
            SEMANTIC_DETECTOR_TILE_OVERLAP,
        )
        tile_count = len(x_origins) * len(y_origins)
        if tile_count > SEMANTIC_DETECTOR_MAXIMUM_TILES:
            raise ValueError(
                "runtime retiling page exceeds the deployment tile limit: "
                f"source_key={key[0]!r}, image={key[1]!r}, "
                f"image_id={key[2]!r}, "
                f"source_dimensions={source_width}x{source_height}, "
                f"scaled_dimensions={scaled_width}x{scaled_height}, "
                f"x_tiles={len(x_origins)}, y_tiles={len(y_origins)}, "
                f"tile_count={tile_count}, "
                f"maximum={SEMANTIC_DETECTOR_MAXIMUM_TILES}"
            )
        scale_values.append(scale)
        page_tile_counts.append(tile_count)
        unique_targets += len(page["objects"])
        for y in y_origins:
            for x in x_origins:
                crop = (
                    x,
                    y,
                    min(x + SEMANTIC_DETECTOR_INPUT_SIZE, scaled_width),
                    min(y + SEMANTIC_DETECTOR_INPUT_SIZE, scaled_height),
                )
                tile_objects: list[dict[str, Any]] = []
                for object_id, (_stable, source_obj) in sorted(
                    page["objects"].items()
                ):
                    page_box = tuple(
                        float(value)
                        for value in source_obj["page_box_xyxy"]
                    )
                    scaled_box = (
                        page_box[0] * scale,
                        page_box[1] * scale,
                        page_box[2] * scale,
                        page_box[3] * scale,
                    )
                    intersection = (
                        max(scaled_box[0], crop[0]),
                        max(scaled_box[1], crop[1]),
                        min(scaled_box[2], crop[2]),
                        min(scaled_box[3], crop[3]),
                    )
                    category = str(source_obj.get("category_id") or "")
                    if not target_fragment_is_visible(
                        scaled_box,
                        crop,
                        minimum_fraction=minimum_visible_fraction,
                        long_span_minimum_fraction=(
                            long_span_minimum_visible_fraction
                        ),
                        is_long_span=(
                            category in LONG_SPAN_SEMANTIC_CATEGORIES
                            or category.casefold().endswith("text")
                        ),
                        tile_overlap=SEMANTIC_DETECTOR_TILE_OVERLAP,
                    ):
                        continue
                    copied = dict(source_obj)
                    copied["source_object_id"] = object_id
                    copied["box_xyxy"] = [
                        intersection[0] - crop[0],
                        intersection[1] - crop[1],
                        intersection[2] - crop[0],
                        intersection[3] - crop[1],
                    ]
                    tile_objects.append(copied)
                    assigned_source_targets.add((key, object_id))
                template = page["template"]
                runtime_rows.append(
                    {
                        "split": template.get("split"),
                        "source_key": key[0],
                        "image": key[1],
                        "image_id": key[2],
                        "crop_xyxy": list(crop),
                        "objects": tile_objects,
                        "runtime_page_scale": scale,
                        "runtime_scaled_size": [
                            scaled_width,
                            scaled_height,
                        ],
                        "source_page_size": [
                            source_width,
                            source_height,
                        ],
                    }
                )
                target_instances += len(tile_objects)
    expected_source_targets = {
        (key, object_id)
        for key, page in pages.items()
        for object_id in page["objects"]
    }
    missing_source_targets = sorted(
        expected_source_targets - assigned_source_targets
    )
    if missing_source_targets:
        key, object_id = missing_source_targets[0]
        source_obj = pages[key]["objects"][object_id][1]
        raise ValueError(
            "runtime retiling dropped complete-page target: "
            f"source_key={key[0]!r}, image={key[1]!r}, "
            f"image_id={key[2]!r}, source_object_id={object_id!r}, "
            f"category_id={source_obj.get('category_id')!r}, "
            f"page_box_xyxy={source_obj.get('page_box_xyxy')!r}, "
            f"missing_targets={len(missing_source_targets)}"
        )
    return runtime_rows, {
        "version": RUNTIME_PAGE_TILING_VERSION,
        "input_size": SEMANTIC_DETECTOR_INPUT_SIZE,
        "target_staff_spacing": SEMANTIC_DETECTOR_TARGET_STAFF_SPACING,
        "overlap": SEMANTIC_DETECTOR_TILE_OVERLAP,
        "oversized_fragment_visibility_version": (
            OVERSIZED_FRAGMENT_VISIBILITY_VERSION
        ),
        "minimum_scale": SEMANTIC_DETECTOR_MINIMUM_SCALE,
        "maximum_scale": SEMANTIC_DETECTOR_MAXIMUM_SCALE,
        "maximum_tiles": SEMANTIC_DETECTOR_MAXIMUM_TILES,
        "scan_page_shape_contract": SCAN_PAGE_SHAPE_CONTRACT,
        "maximum_scan_page_aspect_ratio": (
            MAXIMUM_SCAN_PAGE_ASPECT_RATIO
        ),
        "pages": len(pages),
        "tiles": len(runtime_rows),
        "minimum_page_tiles": min(page_tile_counts, default=0),
        "maximum_page_tiles": max(page_tile_counts, default=0),
        "minimum_scale_observed": min(scale_values, default=0.0),
        "maximum_scale_observed": max(scale_values, default=0.0),
        "unique_source_targets": unique_targets,
        "tile_target_instances": target_instances,
    }


def stitch_tiled_detections(
    rows: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    *,
    prediction_nms_iou: float = SEMANTIC_PAGE_NMS_IOU,
    class_name_by_label: dict[int, str] | None = None,
    layouts_by_page: dict[
        tuple[str, str, str],
        dict[str, Any],
    ] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Convert overlapping tile detections to runtime-like page detections."""

    import numpy as np

    if not 0 < prediction_nms_iou <= 1:
        raise ValueError("page prediction NMS IoU must be in (0, 1]")
    if not (len(rows) == len(outputs) == len(targets)):
        raise ValueError("tile rows, outputs and targets have different lengths")
    runtime_fusion = (
        class_name_by_label is not None
        and layouts_by_page is not None
    )
    runtime_retiling = bool(rows) and all(
        "runtime_page_scale" in row for row in rows
    )
    if any("runtime_page_scale" in row for row in rows) != runtime_retiling:
        raise ValueError("semantic page stitching mixed runtime and legacy tiles")
    if (class_name_by_label is None) != (layouts_by_page is None):
        raise ValueError(
            "runtime page stitching requires both class names and layouts"
        )
    pages: dict[tuple[str, str, str], dict[str, Any]] = {}
    tile_predictions = 0
    mapped_tile_predictions = 0
    tile_target_instances = 0
    for tile_index, (row, output, target) in enumerate(
        zip(rows, outputs, targets, strict=True)
    ):
        key = _page_key(row)
        if not all(key):
            raise ValueError("semantic tile has no stable page identity")
        page = pages.setdefault(
            key,
            {
                "targets": {},
                "predictions": [],
                "layout": (
                    layouts_by_page.get(key)
                    if layouts_by_page is not None
                    else None
                ),
            },
        )
        if runtime_fusion and page["layout"] is None:
            raise ValueError("semantic page has no runtime layout")
        crop = np.asarray(row["crop_xyxy"], dtype=np.float64).reshape(4)
        runtime_scale = float(row.get("runtime_page_scale", 1.0))
        if not math.isfinite(runtime_scale) or runtime_scale <= 0:
            raise ValueError("semantic runtime tile scale is invalid")
        source_page_size_raw = row.get("source_page_size")
        if "runtime_page_scale" in row:
            if (
                not isinstance(source_page_size_raw, list)
                or len(source_page_size_raw) != 2
            ):
                raise ValueError("semantic runtime source page size is invalid")
            source_width = int(source_page_size_raw[0])
            source_height = int(source_page_size_raw[1])
        else:
            source_width = int(math.ceil(crop[2]))
            source_height = int(math.ceil(crop[3]))
        target_boxes = np.asarray(
            target.get("boxes", ()),
            dtype=np.float64,
        ).reshape(-1, 4)
        target_labels = np.asarray(
            target.get("labels", ()),
            dtype=np.int64,
        ).reshape(-1)
        objects = list(row.get("objects", []))
        if not (len(target_boxes) == len(target_labels) == len(objects)):
            raise ValueError("semantic tile target metadata is inconsistent")
        tile_target_instances += len(objects)
        for box, label, obj in zip(
            target_boxes,
            target_labels,
            objects,
            strict=True,
        ):
            object_id = str(obj.get("source_object_id") or "").strip()
            if not object_id:
                raise ValueError(
                    "overlap-consistent semantic target has no source_object_id"
                )
            page_box_raw = obj.get("page_box_xyxy")
            if not isinstance(page_box_raw, list) or len(page_box_raw) != 4:
                raise ValueError(
                    "complete-page semantic target has no page-space box"
                )
            page_box = np.asarray(
                page_box_raw,
                dtype=np.float64,
            ).reshape(4)
            scaled_page_box = page_box * runtime_scale
            expected_local = np.asarray(
                [
                    max(float(scaled_page_box[0]), float(crop[0]))
                    - crop[0],
                    max(float(scaled_page_box[1]), float(crop[1]))
                    - crop[1],
                    min(float(scaled_page_box[2]), float(crop[2]))
                    - crop[0],
                    min(float(scaled_page_box[3]), float(crop[3]))
                    - crop[1],
                ],
                dtype=np.float64,
            )
            if (
                not np.all(np.isfinite(page_box))
                or page_box[2] <= page_box[0]
                or page_box[3] <= page_box[1]
                or np.max(np.abs(expected_local - box)) > 0.01
            ):
                raise ValueError(
                    "semantic tile target contradicts its complete-page box"
                )
            existing = page["targets"].get(object_id)
            if existing is None:
                page["targets"][object_id] = {
                    "label": int(label),
                    "box": page_box,
                }
            else:
                if (
                    int(existing["label"]) != int(label)
                    or np.max(
                        np.abs(existing["box"] - page_box)
                    )
                    > 0.01
                ):
                    raise ValueError(
                        "semantic source object changed label/page geometry"
                    )

        boxes = np.asarray(output.get("boxes", ()), dtype=np.float64).reshape(
            -1,
            4,
        )
        labels = np.asarray(
            output.get("labels", ()),
            dtype=np.int64,
        ).reshape(-1)
        scores = np.asarray(
            output.get("scores", ()),
            dtype=np.float64,
        ).reshape(-1)
        if not (len(boxes) == len(labels) == len(scores)):
            raise ValueError("semantic tile detector output is inconsistent")
        tile_predictions += len(boxes)
        tile_id = tile_index
        tile_bbox = source_tile_bbox(
            x=int(round(crop[0])),
            y=int(round(crop[1])),
            valid_width=int(round(crop[2] - crop[0])),
            valid_height=int(round(crop[3] - crop[1])),
            scale=runtime_scale,
            source_width=source_width,
            source_height=source_height,
        )
        valid_width = float(crop[2] - crop[0])
        valid_height = float(crop[3] - crop[1])
        for box, label, score in zip(boxes, labels, scores, strict=True):
            if math.isfinite(float(score)):
                center_x = 0.5 * (float(box[0]) + float(box[2]))
                center_y = 0.5 * (float(box[1]) + float(box[3]))
                if not (
                    0 <= center_x < valid_width
                    and 0 <= center_y < valid_height
                ):
                    continue
                scaled_global_box = np.asarray(
                    [
                        crop[0] + max(0.0, float(box[0])),
                        crop[1] + max(0.0, float(box[1])),
                        crop[0] + min(valid_width, float(box[2])),
                        crop[1] + min(valid_height, float(box[3])),
                    ],
                    dtype=np.float64,
                )
                global_box = np.asarray(
                    [
                        max(
                            0,
                            int(
                                math.floor(
                                    float(scaled_global_box[0])
                                    / runtime_scale
                                )
                            ),
                        ),
                        max(
                            0,
                            int(
                                math.floor(
                                    float(scaled_global_box[1])
                                    / runtime_scale
                                )
                            ),
                        ),
                        min(
                            source_width,
                            int(
                                math.ceil(
                                    float(scaled_global_box[2])
                                    / runtime_scale
                                )
                            ),
                        ),
                        min(
                            source_height,
                            int(
                                math.ceil(
                                    float(scaled_global_box[3])
                                    / runtime_scale
                                )
                            ),
                        ),
                    ],
                    dtype=np.float64,
                )
                if (
                    global_box[2] <= global_box[0]
                    or global_box[3] <= global_box[1]
                ):
                    continue
                if runtime_fusion:
                    label_value = int(label)
                    class_name = class_name_by_label.get(label_value)
                    if class_name not in SUPPORTED_RUNTIME_CLASSES:
                        raise ValueError(
                            "semantic prediction label is not a runtime class"
                        )
                    owner = _assign_bbox_to_layout(
                        page["layout"],
                        tuple(int(round(value)) for value in global_box),
                    )
                    if owner is None:
                        continue
                    page["predictions"].append(
                        TileFragmentDetection(
                            class_name,
                            label_value,
                            tuple(
                                int(round(value))
                                for value in global_box
                            ),
                            float(score),
                            owner[0],
                            owner[1],
                            tile_id,
                            tile_bbox,
                        )
                    )
                    mapped_tile_predictions += 1
                else:
                    page["predictions"].append(
                        {
                            "box": global_box,
                            "label": int(label),
                            "score": float(score),
                        }
                    )
                    mapped_tile_predictions += 1

    stitched_outputs: list[dict[str, Any]] = []
    stitched_targets: list[dict[str, Any]] = []
    page_predictions = 0
    unique_targets = 0
    maximum_page_predictions = 0
    maximum_page_targets = 0
    fused_fragment_predictions = 0
    nms_removed_predictions = 0
    for key, page in pages.items():
        retained: list[dict[str, Any]]
        if runtime_fusion:
            retained_tiled: list[TileFragmentDetection] = []
            for candidate in sorted(
                page["predictions"],
                key=lambda item: (
                    -item.confidence,
                    item.label,
                    item.bbox,
                    item.tile_id,
                ),
            ):
                if any(
                    candidate.label == other.label
                    and candidate.staff_index == other.staff_index
                    and _box_iou(
                        candidate.bbox,
                        other.bbox,
                    )
                    >= prediction_nms_iou
                    for other in retained_tiled
                ):
                    continue
                retained_tiled.append(candidate)
            nms_removed_predictions += (
                len(page["predictions"]) - len(retained_tiled)
            )
            fused = fuse_tile_fragments(
                retained_tiled,
                owner_resolver=lambda bbox, layout=page["layout"]: (
                    _assign_bbox_to_layout(layout, bbox)
                ),
            )
            fused_fragment_predictions += len(retained_tiled) - len(fused)
            retained = [
                {
                    "box": item.bbox,
                    "label": item.label,
                    "score": item.confidence,
                }
                for item in fused
            ]
        else:
            retained = []
            for candidate in sorted(
                page["predictions"],
                key=lambda item: (
                    -float(item["score"]),
                    int(item["label"]),
                    tuple(float(value) for value in item["box"]),
                ),
            ):
                if any(
                    int(candidate["label"]) == int(other["label"])
                    and _box_iou(candidate["box"], other["box"])
                    >= prediction_nms_iou
                    for other in retained
                ):
                    continue
                retained.append(candidate)
            nms_removed_predictions += (
                len(page["predictions"]) - len(retained)
            )
        retained.sort(
            key=lambda item: (
                int(item["label"]),
                tuple(float(value) for value in item["box"]),
                -float(item["score"]),
            )
        )
        target_items = sorted(page["targets"].items())
        stitched_outputs.append(
            {
                "boxes": np.asarray(
                    [item["box"] for item in retained],
                    dtype=np.float32,
                ).reshape(-1, 4),
                "labels": np.asarray(
                    [item["label"] for item in retained],
                    dtype=np.int64,
                ),
                "scores": np.asarray(
                    [item["score"] for item in retained],
                    dtype=np.float32,
                ),
            }
        )
        stitched_targets.append(
            {
                "boxes": np.asarray(
                    [item["box"] for _object_id, item in target_items],
                    dtype=np.float32,
                ).reshape(-1, 4),
                "labels": np.asarray(
                    [item["label"] for _object_id, item in target_items],
                    dtype=np.int64,
                ),
                "source_object_ids": [
                    object_id for object_id, _item in target_items
                ],
                "page_key": key,
            }
        )
        page_predictions += len(retained)
        unique_targets += len(target_items)
        maximum_page_predictions = max(maximum_page_predictions, len(retained))
        maximum_page_targets = max(maximum_page_targets, len(target_items))
    return stitched_outputs, stitched_targets, {
        "version": PAGE_STITCHING_VERSION,
        "tile_fragment_fusion_version": (
            TILE_FRAGMENT_FUSION_VERSION if runtime_fusion else None
        ),
        "runtime_layout_assignment": runtime_fusion,
        "runtime_page_retiling_version": (
            RUNTIME_PAGE_TILING_VERSION if runtime_retiling else None
        ),
        "layout_pages": len(layouts_by_page or {}),
        "prediction_nms_iou": prediction_nms_iou,
        "pages": len(pages),
        "tiles": len(rows),
        "tile_predictions": tile_predictions,
        "mapped_tile_predictions": mapped_tile_predictions,
        "unmapped_tile_predictions": (
            tile_predictions - mapped_tile_predictions
        ),
        "page_predictions": page_predictions,
        "nms_removed_predictions": nms_removed_predictions,
        "fused_fragment_predictions": fused_fragment_predictions,
        "tile_target_instances": tile_target_instances,
        "unique_source_targets": unique_targets,
        "duplicate_target_instances": tile_target_instances - unique_targets,
        "maximum_page_predictions": maximum_page_predictions,
        "maximum_page_targets": maximum_page_targets,
    }


def compute_dense_detection_metrics(
    outputs: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    *,
    iou_thresholds: tuple[float, ...] = COCO_IOU_THRESHOLDS,
) -> dict[str, Any]:
    """Compute COCO-style AP without the legacy 100 detections/page cap.

    TorchMetrics 1.6.2 returns ``map=-1`` and ``map_per_class=-1`` when COCO's
    third maxDets value differs from 100. A full score page routinely contains
    more than 100 semantic marks, so silently restoring that cap would turn
    correct low-scoring page detections into false negatives. This evaluator
    keeps COCO's greedy per-image matching and 101-point interpolation while
    retaining every finite page prediction.
    """

    return _compute_dense_detection_metrics(
        outputs,
        targets,
        iou_thresholds=iou_thresholds,
    )


def select_high_precision_operating_point(
    outputs: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    *,
    label: int,
    class_name: str,
    minimum_precision: float,
    minimum_recall: float,
    minimum_true_positives: int,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Select the lowest safe score threshold on an independent holdout.

    Predictions are globally score ordered, while target assignment remains
    image-local and one-to-one.  Equal-score predictions are committed as one
    group so the persisted threshold exactly reproduces the measured confusion
    counts when runtime inference uses ``score >= threshold``.
    """

    import numpy as np

    if len(outputs) != len(targets):
        raise ValueError("operating-point outputs and targets have different lengths")
    target_boxes_by_image: list[np.ndarray] = []
    total_targets = 0
    predictions: list[tuple[float, int, np.ndarray]] = []
    for image_index, (output, target) in enumerate(zip(outputs, targets, strict=True)):
        target_boxes = np.asarray(target.get("boxes", ()), dtype=np.float64).reshape(-1, 4)
        target_labels = np.asarray(target.get("labels", ()), dtype=np.int64).reshape(-1)
        selected_targets = target_boxes[target_labels == label]
        target_boxes_by_image.append(selected_targets)
        total_targets += len(selected_targets)

        boxes = np.asarray(output.get("boxes", ()), dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(output.get("scores", ()), dtype=np.float64).reshape(-1)
        labels = np.asarray(output.get("labels", ()), dtype=np.int64).reshape(-1)
        if not (len(boxes) == len(scores) == len(labels)):
            raise ValueError("operating-point detector output lengths disagree")
        for box, score in zip(boxes[labels == label], scores[labels == label], strict=True):
            if math.isfinite(float(score)):
                predictions.append((float(score), image_index, box))
    predictions.sort(key=lambda item: (-item[0], item[1], tuple(item[2])))

    used_targets = [set() for _ in targets]
    true_positives = 0
    false_positives = 0
    points: list[dict[str, Any]] = []
    index = 0
    while index < len(predictions):
        threshold = predictions[index][0]
        group_end = index
        while (
            group_end < len(predictions)
            and predictions[group_end][0] == threshold
        ):
            _score, image_index, box = predictions[group_end]
            best_target = None
            best_iou = -1.0
            for target_index, target_box in enumerate(
                target_boxes_by_image[image_index]
            ):
                if target_index in used_targets[image_index]:
                    continue
                iou = _box_iou(box, target_box)
                if iou > best_iou:
                    best_iou = iou
                    best_target = target_index
            if best_target is not None and best_iou >= iou_threshold:
                used_targets[image_index].add(best_target)
                true_positives += 1
            else:
                false_positives += 1
            group_end += 1
        precision = true_positives / max(1, true_positives + false_positives)
        recall = true_positives / max(1, total_targets)
        points.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": max(0, total_targets - true_positives),
            }
        )
        index = group_end

    eligible = [
        point
        for point in points
        if point["precision"] >= minimum_precision
        and point["recall"] >= minimum_recall
        and point["true_positives"] >= minimum_true_positives
    ]
    if eligible:
        selected = max(
            eligible,
            key=lambda point: (
                point["recall"],
                point["true_positives"],
                -point["threshold"],
            ),
        )
    elif points:
        selected = max(
            points,
            key=lambda point: (
                point["precision"],
                point["true_positives"],
                point["recall"],
                point["threshold"],
            ),
        )
    else:
        selected = {
            "threshold": 1.0,
            "precision": 0.0,
            "recall": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": total_targets,
        }
    return {
        "class_name": class_name,
        "label": label,
        "selection_method": CALIBRATED_OPERATING_POINT_SELECTION_METHOD,
        "iou_threshold": iou_threshold,
        "target_objects": total_targets,
        **selected,
        "minimum_precision": minimum_precision,
        "minimum_recall": minimum_recall,
        "minimum_true_positives": minimum_true_positives,
        "passed": selected in eligible,
    }


def fixed_rare_class_operating_point(
    *,
    label: int,
    class_name: str,
    target_objects: int,
    minimum_precision: float,
    minimum_recall: float,
    minimum_true_positives: int,
) -> dict[str, Any]:
    """Return a code-fixed threshold when development support is too sparse.

    No prediction or holdout value influences this threshold. The independent
    holdout still has to pass the ordinary precision, recall and support gates
    at this exact operating point.
    """

    if target_objects < 0:
        raise ValueError("rare-class target count cannot be negative")
    return {
        "class_name": class_name,
        "label": label,
        "selection_method": FIXED_RARE_CLASS_SELECTION_METHOD,
        "iou_threshold": 0.5,
        "target_objects": target_objects,
        "threshold": FIXED_RARE_CLASS_OPERATING_POINT_THRESHOLD,
        "precision": None,
        "recall": None,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": target_objects,
        "minimum_precision": minimum_precision,
        "minimum_recall": minimum_recall,
        "minimum_true_positives": minimum_true_positives,
        # This states that threshold selection is valid and independent, not
        # that the sparse development observations passed a statistical gate.
        "passed": True,
    }


def evaluate_fixed_operating_point(
    outputs: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    *,
    label: int,
    class_name: str,
    threshold: float,
    minimum_precision: float,
    minimum_recall: float,
    minimum_true_positives: int,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate a preselected score threshold without adapting it to this set."""

    import numpy as np

    if len(outputs) != len(targets):
        raise ValueError("operating-point outputs and targets have different lengths")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("fixed operating-point threshold must be in [0, 1]")

    target_boxes_by_image: list[np.ndarray] = []
    total_targets = 0
    predictions: list[tuple[float, int, np.ndarray]] = []
    for image_index, (output, target) in enumerate(zip(outputs, targets, strict=True)):
        target_boxes = np.asarray(
            target.get("boxes", ()), dtype=np.float64
        ).reshape(-1, 4)
        target_labels = np.asarray(
            target.get("labels", ()), dtype=np.int64
        ).reshape(-1)
        selected_targets = target_boxes[target_labels == label]
        target_boxes_by_image.append(selected_targets)
        total_targets += len(selected_targets)

        boxes = np.asarray(output.get("boxes", ()), dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(output.get("scores", ()), dtype=np.float64).reshape(-1)
        labels = np.asarray(output.get("labels", ()), dtype=np.int64).reshape(-1)
        if not (len(boxes) == len(scores) == len(labels)):
            raise ValueError("operating-point detector output lengths disagree")
        for box, score in zip(
            boxes[labels == label],
            scores[labels == label],
            strict=True,
        ):
            numeric_score = float(score)
            if math.isfinite(numeric_score) and numeric_score >= threshold:
                predictions.append((numeric_score, image_index, box))
    predictions.sort(key=lambda item: (-item[0], item[1], tuple(item[2])))

    used_targets = [set() for _ in targets]
    true_positives = 0
    false_positives = 0
    for _score, image_index, box in predictions:
        best_target = None
        best_iou = -1.0
        for target_index, target_box in enumerate(
            target_boxes_by_image[image_index]
        ):
            if target_index in used_targets[image_index]:
                continue
            iou = _box_iou(box, target_box)
            if iou > best_iou:
                best_iou = iou
                best_target = target_index
        if best_target is not None and best_iou >= iou_threshold:
            used_targets[image_index].add(best_target)
            true_positives += 1
        else:
            false_positives += 1

    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, total_targets)
    return {
        "class_name": class_name,
        "label": label,
        "iou_threshold": iou_threshold,
        "target_objects": total_targets,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": max(0, total_targets - true_positives),
        "minimum_precision": minimum_precision,
        "minimum_recall": minimum_recall,
        "minimum_true_positives": minimum_true_positives,
        "passed": (
            precision >= minimum_precision
            and recall >= minimum_recall
            and true_positives >= minimum_true_positives
        ),
    }


def holdout_isolation_failures(
    manifest: dict[str, Any],
    preparation: dict[str, Any],
    *,
    minimum_independent_works: int,
) -> list[str]:
    failures: list[str] = []
    if manifest.get("role") != BENCHMARK_SELECTION_ROLE:
        failures.append("manifest_role")
    if preparation.get("role") != BENCHMARK_SELECTION_ROLE:
        failures.append("preparation_role")
    if manifest.get("target_assignment_version") != TRANSFORMATION_VERSION:
        failures.append("manifest_target_assignment")
    if preparation.get("transformation_version") != TRANSFORMATION_VERSION:
        failures.append("preparation_target_assignment")
    for payload_name, payload in (
        ("manifest", manifest),
        ("preparation", preparation),
    ):
        if (
            payload.get("scan_page_shape_contract")
            != SCAN_PAGE_SHAPE_CONTRACT
            or float(
                payload.get("maximum_scan_page_aspect_ratio", -1)
            )
            != MAXIMUM_SCAN_PAGE_ASPECT_RATIO
        ):
            failures.append(f"{payload_name}_scan_page_shape")
        if (
            payload.get("oversized_fragment_visibility_version")
            != OVERSIZED_FRAGMENT_VISIBILITY_VERSION
        ):
            failures.append(f"{payload_name}_fragment_visibility")
    if int(manifest.get("source_split_overlap", -1)) != 0:
        failures.append("source_split_overlap")
    if manifest.get("forbidden_selection_overlap") != []:
        failures.append("manifest_pair_overlap")
    if manifest.get("forbidden_work_overlap") != []:
        failures.append("manifest_work_overlap")
    if preparation.get("forbidden_selection_overlap") != []:
        failures.append("preparation_pair_overlap")
    if preparation.get("forbidden_work_overlap") != []:
        failures.append("preparation_work_overlap")

    selected_works = int(preparation.get("selected_works", -1))
    accepted_works = int(preparation.get("accepted_works", -1))
    manifest_accepted_works = int(manifest.get("accepted_works", -1))
    test_sources = int(
        preparation.get("source_count_by_split", {}).get("test", -1)
    )
    if selected_works < minimum_independent_works:
        failures.append(
            f"selected_works={selected_works}<{minimum_independent_works}"
        )
    if accepted_works < minimum_independent_works:
        failures.append(
            f"accepted_works={accepted_works}<{minimum_independent_works}"
        )
    if manifest_accepted_works != accepted_works:
        failures.append(
            "manifest_accepted_works="
            f"{manifest_accepted_works}!={accepted_works}"
        )
    if test_sources != accepted_works:
        failures.append(f"test_sources={test_sources}!={accepted_works}")
    return failures


def calibration_isolation_failures(
    manifest: dict[str, Any],
    preparation: dict[str, Any],
) -> list[str]:
    """Verify that threshold selection uses development scans, not the holdout."""

    failures: list[str] = []
    if manifest.get("role") != TRAINING_REGION_ROLE:
        failures.append("manifest_role")
    if preparation.get("role") != TRAINING_REGION_ROLE:
        failures.append("preparation_role")
    if manifest.get("target_assignment_version") != TRANSFORMATION_VERSION:
        failures.append("manifest_target_assignment")
    if preparation.get("transformation_version") != TRANSFORMATION_VERSION:
        failures.append("preparation_target_assignment")
    if int(manifest.get("source_split_overlap", -1)) != 0:
        failures.append("source_split_overlap")
    for payload_name, payload in (
        ("manifest", manifest),
        ("preparation", preparation),
    ):
        if (
            payload.get("scan_page_shape_contract")
            != SCAN_PAGE_SHAPE_CONTRACT
            or float(
                payload.get("maximum_scan_page_aspect_ratio", -1)
            )
            != MAXIMUM_SCAN_PAGE_ASPECT_RATIO
        ):
            failures.append(f"{payload_name}_scan_page_shape")
        if (
            payload.get("oversized_fragment_visibility_version")
            != OVERSIZED_FRAGMENT_VISIBILITY_VERSION
        ):
            failures.append(f"{payload_name}_fragment_visibility")
        if payload.get("forbidden_selection_overlap") != []:
            failures.append(f"{payload_name}_pair_overlap")
        if payload.get("forbidden_work_overlap") != []:
            failures.append(f"{payload_name}_work_overlap")
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument(
        "--operating-point-calibration-prepared-dir",
        type=Path,
        required=True,
        help=(
            "disjoint scan-degraded development regions used only to preselect "
            "deployment score thresholds"
        ),
    )
    parser.add_argument(
        "--operating-point-calibration-images-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--operating-point-calibration-split",
        action="append",
        choices=("train", "calibration", "test"),
        help=(
            "development split used to preselect deployment score thresholds; "
            "repeat to select more than one split (default: all three for "
            "backward compatibility)"
        ),
    )
    parser.add_argument(
        "--page-layout-evidence",
        type=Path,
        required=True,
        help=(
            "hash-bound layouts generated by the Windows product analyzer "
            "for the independent holdout"
        ),
    )
    parser.add_argument(
        "--operating-point-calibration-page-layout-evidence",
        type=Path,
        required=True,
        help=(
            "hash-bound layouts generated by the Windows product analyzer "
            "for the disjoint threshold-calibration data"
        ),
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-categories", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "auto"),
        default="cuda",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="PyTorch intra-op threads in CPU mode; zero keeps the runtime default",
    )
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--detections-per-tile", type=int, default=300)
    parser.add_argument("--minimum-map-50", type=float, default=0.95)
    parser.add_argument("--minimum-map-75", type=float, default=0.90)
    parser.add_argument("--minimum-priority-map", type=float, default=0.85)
    parser.add_argument(
        "--minimum-independent-works",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--minimum-required-class-test-objects",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--required-class-map",
        action="append",
        default=[],
        metavar="NAME=FLOOR",
    )
    parser.add_argument(
        "--operating-point-class",
        action="append",
        default=[],
        metavar="NAME",
        help="class requiring a deployment score threshold (default: all runtime geometry/text classes)",
    )
    parser.add_argument(
        "--minimum-operating-point-precision",
        type=float,
        default=0.995,
    )
    parser.add_argument(
        "--minimum-operating-point-recall",
        type=float,
        default=MINIMUM_OPERATING_POINT_RECALL,
    )
    parser.add_argument(
        "--minimum-high-recall-mark-recall",
        type=float,
        default=MINIMUM_HIGH_RECALL_MARK_RECALL,
        help=(
            "stronger recall floor for accidentals, articulations, dynamics, "
            "ornaments, hairpins, slurs and ties"
        ),
    )
    parser.add_argument(
        "--minimum-operating-point-true-positives",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--minimum-operating-point-calibration-true-positives",
        type=int,
        default=10,
        help=(
            "minimum development positives used to preselect a fixed "
            "threshold; the independent holdout retains its separate floor"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    calibration_splits = tuple(
        dict.fromkeys(
            args.operating_point_calibration_split
            or ("train", "calibration", "test")
        )
    )
    for path in (
        args.prepared_dir,
        args.images_dir,
        args.operating_point_calibration_prepared_dir,
        args.operating_point_calibration_images_dir,
    ):
        if not path.is_dir():
            raise FileNotFoundError(path)
    for path in (
        args.model,
        args.model_categories,
        args.page_layout_evidence,
        args.operating_point_calibration_page_layout_evidence,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_report.exists():
        raise FileExistsError(args.output_report)
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    if args.minimum_required_class_test_objects <= 0:
        raise ValueError("minimum required class support must be positive")
    if args.minimum_independent_works <= 0:
        raise ValueError("minimum independent works must be positive")
    required_class_maps = parse_required_class_maps(
        args.required_class_map
    )
    operating_point_classes = tuple(
        dict.fromkeys(
            str(value).strip()
            for value in (
                args.operating_point_class
                or DEPLOYMENT_OPERATING_POINT_CLASSES
            )
            if str(value).strip()
        )
    )
    if (
        not 0.0 <= args.minimum_operating_point_precision <= 1.0
        or not 0.0 <= args.minimum_operating_point_recall <= 1.0
        or not 0.0 <= args.minimum_high_recall_mark_recall <= 1.0
        or (
            args.minimum_high_recall_mark_recall
            < args.minimum_operating_point_recall
        )
        or args.minimum_operating_point_true_positives <= 0
        or args.minimum_operating_point_calibration_true_positives <= 0
    ):
        raise ValueError("semantic detector operating-point gates are invalid")

    manifest_path = args.prepared_dir / "manifest.json"
    report_path = args.prepared_dir / "prepare-report.json"
    categories_path = args.prepared_dir / "categories.json"
    calibration_manifest_path = (
        args.operating_point_calibration_prepared_dir / "manifest.json"
    )
    calibration_report_path = (
        args.operating_point_calibration_prepared_dir / "prepare-report.json"
    )
    calibration_categories_path = (
        args.operating_point_calibration_prepared_dir / "categories.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preparation = json.loads(report_path.read_text(encoding="utf-8"))
    isolation_failures = holdout_isolation_failures(
        manifest,
        preparation,
        minimum_independent_works=args.minimum_independent_works,
    )
    if isolation_failures:
        raise ValueError(
            "semantic detector holdout isolation contract failed: "
            + "; ".join(isolation_failures)
        )
    calibration_manifest = json.loads(
        calibration_manifest_path.read_text(encoding="utf-8")
    )
    calibration_preparation = json.loads(
        calibration_report_path.read_text(encoding="utf-8")
    )
    calibration_failures = calibration_isolation_failures(
        calibration_manifest,
        calibration_preparation,
    )
    if calibration_failures:
        raise ValueError(
            "semantic detector threshold-calibration isolation contract failed: "
            + "; ".join(calibration_failures)
        )
    prepared_categories = json.loads(
        categories_path.read_text(encoding="utf-8")
    )
    calibration_categories = json.loads(
        calibration_categories_path.read_text(encoding="utf-8")
    )
    model_categories = json.loads(
        args.model_categories.read_text(encoding="utf-8")
    )
    assert_compatible_category_manifests(
        prepared_categories,
        model_categories,
    )
    assert_compatible_category_manifests(
        prepared_categories,
        calibration_categories,
    )
    class_name_by_label = category_label_name_map(prepared_categories)
    label_by_class_name = {
        name: label for label, name in class_name_by_label.items()
    }
    missing_operating_classes = [
        name for name in operating_point_classes if name not in label_by_class_name
    ]
    if missing_operating_classes:
        raise ValueError(
            "semantic detector operating-point classes are missing: "
            + ", ".join(missing_operating_classes)
        )
    number_of_classes = max(class_name_by_label) + 1
    holdout_test_path = args.prepared_dir / "test.jsonl"
    rows = load_jsonl(holdout_test_path)
    rows, target_box_audit = normalize_target_boxes(
        rows,
        minimum_visible_fraction=float(
            preparation.get("minimum_object_fraction", 0.8)
        ),
        long_span_minimum_visible_fraction=float(
            preparation.get("long_span_minimum_object_fraction", 0.25)
        ),
        require_complete_page_geometry=True,
        tile_overlap=float(preparation["overlap"]),
    )
    calibration_rows: list[dict[str, Any]] = []
    calibration_split_hashes: dict[str, str] = {}
    calibration_source_counts: dict[str, int] = {}
    for split in calibration_splits:
        split_path = (
            args.operating_point_calibration_prepared_dir
            / f"{split}.jsonl"
        )
        split_rows = load_jsonl(split_path)
        calibration_split_hashes[split] = sha256_file(split_path)
        calibration_source_counts[split] = len(
            {str(row["source_key"]) for row in split_rows}
        )
        calibration_rows.extend(split_rows)
    calibration_rows, calibration_box_audit = normalize_target_boxes(
        calibration_rows,
        minimum_visible_fraction=float(
            calibration_preparation.get("minimum_object_fraction", 0.8)
        ),
        long_span_minimum_visible_fraction=float(
            calibration_preparation.get(
                "long_span_minimum_object_fraction",
                0.25,
            )
        ),
        require_complete_page_geometry=True,
        tile_overlap=float(calibration_preparation["overlap"]),
    )
    if not calibration_rows:
        raise ValueError("semantic detector operating-point calibration is empty")
    holdout_sources = {str(row["source_key"]) for row in rows}
    calibration_sources = {
        str(row["source_key"]) for row in calibration_rows
    }
    source_overlap = sorted(holdout_sources & calibration_sources)
    if source_overlap:
        raise ValueError(
            "semantic detector threshold calibration overlaps the holdout: "
            + ", ".join(source_overlap[:8])
        )
    holdout_split_hashes = {
        "test": sha256_file(holdout_test_path),
    }
    holdout_layouts = load_page_layout_evidence(
        rows,
        args.page_layout_evidence,
        images_dir=args.images_dir,
        prepared_manifest_sha256=sha256_file(manifest_path),
        split_jsonl_sha256=holdout_split_hashes,
    )
    calibration_layouts = load_page_layout_evidence(
        calibration_rows,
        args.operating_point_calibration_page_layout_evidence,
        images_dir=args.operating_point_calibration_images_dir,
        prepared_manifest_sha256=sha256_file(
            calibration_manifest_path
        ),
        split_jsonl_sha256=calibration_split_hashes,
    )
    rows, holdout_runtime_tiling = retile_complete_page_rows_for_runtime(
        rows,
        holdout_layouts,
        minimum_visible_fraction=float(
            preparation.get("minimum_object_fraction", 0.8)
        ),
        long_span_minimum_visible_fraction=float(
            preparation.get("long_span_minimum_object_fraction", 0.25)
        ),
    )
    (
        calibration_rows,
        calibration_runtime_tiling,
    ) = retile_complete_page_rows_for_runtime(
        calibration_rows,
        calibration_layouts,
        minimum_visible_fraction=float(
            calibration_preparation.get("minimum_object_fraction", 0.8)
        ),
        long_span_minimum_visible_fraction=float(
            calibration_preparation.get(
                "long_span_minimum_object_fraction",
                0.25,
            )
        ),
    )
    test_class_counts = unique_class_counts(rows)
    calibration_class_counts = unique_class_counts(calibration_rows)
    insufficient = insufficient_required_class_support(
        required_class_maps,
        class_name_by_label=class_name_by_label,
        test_class_counts=test_class_counts,
        minimum_objects=args.minimum_required_class_test_objects,
    )
    if insufficient:
        raise ValueError(
            "independent holdout class support is insufficient: "
            + ", ".join(
                f"{name}={count}"
                for name, count in insufficient.items()
            )
        )
    import torch
    import torchvision
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import functional as vision_f

    if args.cpu_threads < 0:
        raise ValueError("cpu-threads cannot be negative")
    device_name = resolve_detector_device(
        args.device,
        cuda_available=torch.cuda.is_available(),
    )
    device = torch.device(device_name)
    use_cuda = device.type == "cuda"
    if not use_cuda and args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(1)

    class DetectionRowsDataset(Dataset):
        def __init__(
            self,
            dataset_rows: list[dict[str, Any]],
            images_dir: Path,
        ) -> None:
            self.rows = dataset_rows
            self.images_dir = images_dir
            self.runtime_page_cache: collections.OrderedDict[
                tuple[str, tuple[int, int]],
                Any,
            ] = collections.OrderedDict()

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
            row = self.rows[index]
            image_path = self.images_dir / row["image"]
            if "runtime_page_scale" in row:
                from PIL import Image, ImageOps

                scaled_size = tuple(
                    int(value) for value in row["runtime_scaled_size"]
                )
                cache_key = (str(image_path), scaled_size)
                scaled_page = self.runtime_page_cache.get(cache_key)
                if scaled_page is None:
                    with Image.open(image_path) as source:
                        grayscale = ImageOps.grayscale(source)
                        expected_source_size = tuple(
                            int(value)
                            for value in row["source_page_size"]
                        )
                        if grayscale.size != expected_source_size:
                            raise ValueError(
                                "semantic runtime page dimensions changed"
                            )
                        scale = float(row["runtime_page_scale"])
                        scaled_page = (
                            grayscale.copy()
                            if scaled_size == grayscale.size
                            else grayscale.resize(
                                scaled_size,
                                (
                                    Image.Resampling.BICUBIC
                                    if scale > 1.0
                                    else Image.Resampling.LANCZOS
                                ),
                            )
                        )
                    self.runtime_page_cache[cache_key] = scaled_page
                    self.runtime_page_cache.move_to_end(cache_key)
                    while len(self.runtime_page_cache) > 2:
                        self.runtime_page_cache.popitem(last=False)
                crop = scaled_page.crop(tuple(row["crop_xyxy"]))
                padded = Image.new(
                    "L",
                    (
                        SEMANTIC_DETECTOR_INPUT_SIZE,
                        SEMANTIC_DETECTOR_INPUT_SIZE,
                    ),
                    255,
                )
                padded.paste(crop, (0, 0))
                image = padded.convert("RGB")
            else:
                image = load_grayscale_crop(
                    image_path,
                    row["crop_xyxy"],
                ).convert("RGB")
            tensor = vision_f.pil_to_tensor(image).float().div_(255.0)
            boxes = torch.tensor(
                [obj["box_xyxy"] for obj in row["objects"]],
                dtype=torch.float32,
            ).reshape(-1, 4)
            labels = torch.tensor(
                [obj["label"] for obj in row["objects"]],
                dtype=torch.int64,
            )
            return tensor, {
                "boxes": boxes,
                "labels": labels,
                "image_id": torch.tensor([index], dtype=torch.int64),
            }

    def collate(
        batch: list[tuple[Any, dict[str, Any]]],
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        images, targets = zip(*batch)
        return list(images), list(targets)

    def make_loader(
        dataset_rows: list[dict[str, Any]],
        images_dir: Path,
    ) -> Any:
        return DataLoader(
            DetectionRowsDataset(dataset_rows, images_dir),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate,
            pin_memory=use_cuda,
            persistent_workers=False,
        )

    loader = make_loader(rows, args.images_dir)
    calibration_loader = make_loader(
        calibration_rows,
        args.operating_point_calibration_images_dir,
    )
    model = build_detector_model(
        number_of_classes=number_of_classes,
        score_threshold=args.score_threshold,
        detections_per_tile=args.detections_per_tile,
        pretrained_backbone=False,
        class_name_by_label=class_name_by_label,
    )
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("semantic detector model is not a state dictionary")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    def infer(
        inference_loader: Any,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        collected_outputs: list[dict[str, Any]] = []
        collected_targets: list[dict[str, Any]] = []
        with torch.inference_mode():
            for images, targets in inference_loader:
                outputs = model(
                    [
                        image.to(device, non_blocking=use_cuda)
                        for image in images
                    ]
                )
                cpu_outputs = [
                    {
                        name: value.detach().cpu()
                        for name, value in output.items()
                    }
                    for output in outputs
                ]
                collected_outputs.extend(cpu_outputs)
                collected_targets.extend(
                    [
                        {
                            name: (
                                value.detach().cpu()
                                if hasattr(value, "detach")
                                else value
                            )
                            for name, value in target.items()
                        }
                        for target in targets
                    ]
                )
        return collected_outputs, collected_targets

    calibration_tile_outputs, calibration_tile_targets = infer(
        calibration_loader
    )
    (
        calibration_outputs,
        calibration_targets,
        calibration_stitching,
    ) = stitch_tiled_detections(
        calibration_rows,
        calibration_tile_outputs,
        calibration_tile_targets,
        class_name_by_label=class_name_by_label,
        layouts_by_page=calibration_layouts,
    )
    selected_calibration_points: dict[str, dict[str, Any]] = {}
    for class_name in operating_point_classes:
        label = label_by_class_name[class_name]
        calibration_objects = int(calibration_class_counts.get(label, 0))
        class_minimum_recall = minimum_recall_for_class(
            class_name,
            minimum_recall=args.minimum_operating_point_recall,
            minimum_high_recall_mark_recall=(
                args.minimum_high_recall_mark_recall
            ),
        )
        if (
            calibration_objects
            < args.minimum_operating_point_calibration_true_positives
        ):
            point = fixed_rare_class_operating_point(
                label=label,
                class_name=class_name,
                target_objects=calibration_objects,
                minimum_precision=args.minimum_operating_point_precision,
                minimum_recall=class_minimum_recall,
                minimum_true_positives=(
                    args.minimum_operating_point_calibration_true_positives
                ),
            )
        else:
            point = select_high_precision_operating_point(
                calibration_outputs,
                calibration_targets,
                label=label,
                class_name=class_name,
                minimum_precision=args.minimum_operating_point_precision,
                minimum_recall=class_minimum_recall,
                minimum_true_positives=(
                    args.minimum_operating_point_calibration_true_positives
                ),
            )
        selected_calibration_points[class_name] = point
    # Threshold selection is complete and immutable before any holdout image is
    # evaluated.  Release the potentially large prediction tensors and worker
    # pool so calibration and final evidence do not coexist in memory.
    del calibration_tile_outputs
    del calibration_tile_targets
    del calibration_outputs
    del calibration_targets
    del calibration_loader
    del calibration_layouts
    holdout_tile_outputs, holdout_tile_targets = infer(loader)
    holdout_outputs, holdout_targets, holdout_stitching = (
        stitch_tiled_detections(
            rows,
            holdout_tile_outputs,
            holdout_tile_targets,
            class_name_by_label=class_name_by_label,
            layouts_by_page=holdout_layouts,
        )
    )
    metrics = compute_dense_detection_metrics(
        holdout_outputs,
        holdout_targets,
    )
    metrics["map_per_class_named"] = {
        class_name_by_label.get(int(label), str(label)): float(value)
        for label, value in zip(
            metrics.get("classes", []),
            metrics.get("map_per_class", []),
        )
    }
    holdout_class_support = {
        class_name_by_label.get(int(label), str(label)): int(count)
        for label, count in collections.Counter(
            int(label)
            for target in holdout_targets
            for label in target["labels"].tolist()
        ).items()
    }
    selection_score, priority_map = priority_selection_score(
        overall_map=float(metrics["map"]),
        per_class_map=metrics["map_per_class_named"],
        class_support=holdout_class_support,
        minimum_support=args.minimum_required_class_test_objects,
    )
    (
        selection_support_filtered_map,
        selection_supported_classes,
    ) = support_filtered_macro_map(
        per_class_map=metrics["map_per_class_named"],
        class_support=holdout_class_support,
        minimum_support=args.minimum_required_class_test_objects,
    )
    metrics["priority_mark_map"] = priority_map
    metrics["priority_mark_minimum_class_support"] = (
        args.minimum_required_class_test_objects
    )
    metrics["priority_mark_supported_classes"] = sorted(
        name
        for name, count in holdout_class_support.items()
        if count >= args.minimum_required_class_test_objects
        and is_priority_mark_class(name)
    )
    metrics["selection_support_filtered_map"] = (
        selection_support_filtered_map
    )
    metrics["selection_minimum_class_support"] = (
        args.minimum_required_class_test_objects
    )
    metrics["selection_supported_classes"] = selection_supported_classes
    metrics["selection_score"] = selection_score
    operating_points: dict[str, dict[str, Any]] = {}
    for class_name in operating_point_classes:
        calibration_point = selected_calibration_points[class_name]
        class_minimum_recall = minimum_recall_for_class(
            class_name,
            minimum_recall=args.minimum_operating_point_recall,
            minimum_high_recall_mark_recall=(
                args.minimum_high_recall_mark_recall
            ),
        )
        point = evaluate_fixed_operating_point(
            holdout_outputs,
            holdout_targets,
            label=label_by_class_name[class_name],
            class_name=class_name,
            threshold=float(calibration_point["threshold"]),
            minimum_precision=args.minimum_operating_point_precision,
            minimum_recall=class_minimum_recall,
            minimum_true_positives=(
                args.minimum_operating_point_true_positives
            ),
        )
        point["calibration_passed"] = calibration_point["passed"]
        point["calibration_selection_method"] = calibration_point[
            "selection_method"
        ]
        point["calibration_precision"] = calibration_point["precision"]
        point["calibration_recall"] = calibration_point["recall"]
        point["calibration_true_positives"] = (
            calibration_point["true_positives"]
        )
        point["calibration_minimum_true_positives"] = (
            args.minimum_operating_point_calibration_true_positives
        )
        point["passed"] = bool(
            calibration_point["passed"] and point["passed"]
        )
        operating_points[class_name] = point
    metrics["operating_points"] = operating_points
    failures = acceptance_failures(
        metrics,
        minimum_map_50=args.minimum_map_50,
        minimum_map_75=args.minimum_map_75,
        minimum_priority_map=args.minimum_priority_map,
        required_class_maps=required_class_maps,
        operating_points=operating_points,
    )
    output = {
        "format": 3,
        "name": "scorescan-semantic-detector-independent-scan-holdout-v3",
        "purpose": (
            "frozen-threshold forbidden-to-train scan-degraded localization "
            "development gate; not physical-scan release evidence"
        ),
        "source_image_origin": "synthetic_scan_degraded_render",
        "production_evidence_eligible": False,
        "model_sha256": sha256_file(args.model),
        "model_categories_sha256": sha256_file(args.model_categories),
        "model_contract": detector_model_contract(),
        "priority_selection_protocol": PRIORITY_SELECTION_PROTOCOL,
        "prepared_manifest_sha256": sha256_file(manifest_path),
        "prepare_report_sha256": sha256_file(report_path),
        "split_jsonl_sha256": holdout_split_hashes,
        "page_layout_evidence": {
            "version": PAGE_LAYOUT_EVIDENCE_VERSION,
            "sha256": sha256_file(args.page_layout_evidence),
            "pages": len(holdout_layouts),
        },
        "runtime_page_tiling": holdout_runtime_tiling,
        "independent_works": int(preparation["accepted_works"]),
        "test_tiles": len(rows),
        "test_class_counts": test_class_counts,
        "page_stitching": holdout_stitching,
        "detection_metric_protocol": DENSE_MAP_VERSION,
        "target_box_normalization": target_box_audit,
        "operating_point_calibration": {
            "selection_dataset_role": calibration_preparation["role"],
            "selected_splits": list(calibration_splits),
            "holdout_reused_for_selection": False,
            "source_overlap_with_holdout": source_overlap,
            "prepared_manifest_sha256": sha256_file(
                calibration_manifest_path
            ),
            "prepare_report_sha256": sha256_file(
                calibration_report_path
            ),
            "categories_sha256": sha256_file(
                calibration_categories_path
            ),
            "split_jsonl_sha256": calibration_split_hashes,
            "page_layout_evidence": {
                "version": PAGE_LAYOUT_EVIDENCE_VERSION,
                "sha256": sha256_file(
                    args.operating_point_calibration_page_layout_evidence
                ),
                "pages": calibration_stitching["layout_pages"],
            },
            "runtime_page_tiling": calibration_runtime_tiling,
            "source_counts_by_split": calibration_source_counts,
            "tiles": len(calibration_rows),
            "class_counts": calibration_class_counts,
            "page_stitching": calibration_stitching,
            "target_box_normalization": calibration_box_audit,
            "selected_points": selected_calibration_points,
            "minimum_true_positives": (
                args.minimum_operating_point_calibration_true_positives
            ),
        },
        "runtime": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda": torch.version.cuda,
            "device": device.type,
            "gpu": torch.cuda.get_device_name(0) if use_cuda else None,
            "cpu_threads": torch.get_num_threads() if not use_cuda else None,
        },
        "metrics": metrics,
        "acceptance": {
            "passed": not failures,
            "minimum_map_50": args.minimum_map_50,
            "minimum_map_75": args.minimum_map_75,
            "minimum_priority_map": args.minimum_priority_map,
            "minimum_independent_works": args.minimum_independent_works,
            "minimum_required_class_test_objects": (
                args.minimum_required_class_test_objects
            ),
            "required_class_maps": required_class_maps,
            "operating_point_classes": list(operating_point_classes),
            "minimum_operating_point_precision": (
                args.minimum_operating_point_precision
            ),
            "minimum_operating_point_recall": (
                args.minimum_operating_point_recall
            ),
            "minimum_high_recall_mark_recall": (
                args.minimum_high_recall_mark_recall
            ),
            "high_recall_mark_classes": sorted(HIGH_RECALL_MARK_CLASSES),
            "minimum_operating_point_true_positives": (
                args.minimum_operating_point_true_positives
            ),
            "minimum_operating_point_calibration_true_positives": (
                args.minimum_operating_point_calibration_true_positives
            ),
            "failures": failures,
        },
        # A PyTorch evaluation pass is not a deployable runtime integration.
        "integration_authorized": False,
        "elapsed_seconds": time.time() - started,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_report.with_suffix(
        args.output_report.suffix + ".tmp"
    )
    temporary.write_text(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output_report)
    print(
        json.dumps(output, sort_keys=True, allow_nan=False),
        flush=True,
    )
    if failures:
        raise RuntimeError(
            "independent semantic detector holdout gate failed: "
            + "; ".join(failures)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
