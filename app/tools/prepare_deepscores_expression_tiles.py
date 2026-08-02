#!/usr/bin/env python3
"""Prepare leakage-safe DeepScoresV2 tiles for notation-mark detection.

The target vocabulary intentionally focuses on the classes that the structural
image-to-sequence model does not localize reliably: accidentals, rests,
articulations, dynamics, ornaments, tuplets, ties, slurs and hairpins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


TARGET_PREFIXES = (
    "accidental",
    "artic",
    "dynamic",
    "fermata",
    "fingering",
    "graceNote",
    "key",
    "ornament",
    "rest",
    "strings",
    "timeSig",
    "tuplet",
)
TARGET_EXACT = {
    "arpeggiato",
    "augmentationDot",
    "caesura",
    "clef15",
    "clef8",
    "clefCAlto",
    "clefCTenor",
    "clefF",
    "clefG",
    "clefUnpitchedPercussion",
    "coda",
    "keyboardPedalPed",
    "keyboardPedalUp",
    "ottavaBracket",
    "repeatDot",
    "segno",
    "slur",
    "tie",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def grid_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("Expected tile_size > overlap >= 0")
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _intersection_fraction(
    box: tuple[float, float, float, float],
    tile: tuple[int, int, int, int],
) -> float:
    left, top, right, bottom = box
    tile_left, tile_top, tile_right, tile_bottom = tile
    intersection_width = max(0.0, min(right, tile_right) - max(left, tile_left))
    intersection_height = max(0.0, min(bottom, tile_bottom) - max(top, tile_top))
    area = max(1.0, (right - left) * (bottom - top))
    return intersection_width * intersection_height / area


def choose_tile(
    box: tuple[float, float, float, float],
    tiles: list[tuple[int, int, int, int]],
    minimum_fraction: float,
) -> int | None:
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    candidates: list[tuple[float, int]] = []
    for index, tile in enumerate(tiles):
        if tile[0] <= center_x <= tile[2] and tile[1] <= center_y <= tile[3]:
            candidates.append((_intersection_fraction(box, tile), index))
    if not candidates:
        return None
    fraction, index = max(candidates, key=lambda item: (item[0], -item[1]))
    return index if fraction >= minimum_fraction else None


def _category_id(annotation: dict[str, Any]) -> str | None:
    for value in annotation.get("cat_id", []):
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= numeric <= 136:
            return str(numeric)
    return None


def parse_deepscores_bbox(
    annotation: dict[str, Any],
) -> tuple[float, float, float, float]:
    """Return the axis-aligned DeepScores box as ``(x1, y1, x2, y2)``.

    DeepScores calls this field ``a_bbox`` but stores it in ordinary Cartesian
    xyxy order.  ``o_bbox`` is used as an independent centre-geometry check
    whenever present; this prevents a silent x/y transposition from producing
    plausible yet completely misplaced training targets.  Its exact extrema
    are deliberately not compared because the oriented box can include rotated
    corners outside the tighter axis-aligned annotation.
    """

    raw = annotation.get("a_bbox")
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError("DeepScores annotation has no valid a_bbox")
    try:
        left, top, right, bottom = (float(value) for value in raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("DeepScores a_bbox contains a non-numeric value") from exc
    if not all(math.isfinite(value) for value in (left, top, right, bottom)):
        raise ValueError("DeepScores a_bbox contains a non-finite value")
    if right < left or bottom < top or (right == left and bottom == top):
        raise ValueError("DeepScores a_bbox is not a usable xyxy box")

    oriented = annotation.get("o_bbox")
    if oriented is not None:
        if not isinstance(oriented, list) or len(oriented) < 8 or len(oriented) % 2:
            raise ValueError("DeepScores annotation has an invalid o_bbox")
        try:
            coordinates = [float(value) for value in oriented]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("DeepScores o_bbox contains a non-numeric value") from exc
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("DeepScores o_bbox contains a non-finite value")
        xs = coordinates[0::2]
        ys = coordinates[1::2]
        oriented_bounds = (min(xs), min(ys), max(xs), max(ys))
        center = ((left + right) / 2.0, (top + bottom) / 2.0)
        oriented_center = (
            (oriented_bounds[0] + oriented_bounds[2]) / 2.0,
            (oriented_bounds[1] + oriented_bounds[3]) / 2.0,
        )
        oriented_diagonal = max(
            1.0,
            math.hypot(
                oriented_bounds[2] - oriented_bounds[0],
                oriented_bounds[3] - oriented_bounds[1],
            ),
        )
        normalized_center_distance = (
            math.hypot(
                center[0] - oriented_center[0],
                center[1] - oriented_center[1],
            )
            / oriented_diagonal
        )
        if normalized_center_distance > 0.20:
            raise ValueError("DeepScores a_bbox and o_bbox disagree")
    return left, top, right, bottom


def _is_target(name: str) -> bool:
    return name in TARGET_EXACT or name.startswith(TARGET_PREFIXES)


def _stable_unit(value: str) -> float:
    integer = int(hashlib.sha256(value.encode()).hexdigest()[:16], 16)
    return integer / float(0xFFFFFFFFFFFFFFFF)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def prepare_split(
    *,
    annotation_path: Path,
    images_dir: Path,
    output_path: Path,
    split: str,
    tile_size: int,
    overlap: int,
    minimum_fraction: float,
    negative_ratio: float,
    selected_categories: dict[str, dict[str, Any]] | None,
    minimum_train_instances: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {
        category_id: details
        for category_id, details in payload["categories"].items()
        if details.get("annotation_set") == "deepscores"
        and _is_target(str(details["name"]))
    }
    raw_counts: Counter[str] = Counter()
    annotations_by_image: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(
        list
    )
    for annotation in payload["annotations"].values():
        category_id = _category_id(annotation)
        if category_id not in categories:
            continue
        raw_counts[category_id] += 1
        annotations_by_image[str(annotation["img_id"])].append(
            (category_id, annotation)
        )

    if selected_categories is None:
        categories = {
            category_id: details
            for category_id, details in categories.items()
            if raw_counts[category_id] >= minimum_train_instances
        }
    else:
        categories = selected_categories
    ordered_ids = sorted(categories, key=lambda value: int(value))
    class_index = {
        category_id: index + 1 for index, category_id in enumerate(ordered_ids)
    }

    rows: list[dict[str, Any]] = []
    tiled_counts: Counter[str] = Counter()
    dropped_counts: Counter[str] = Counter()
    invalid_counts: Counter[str] = Counter()
    expanded_line_counts: Counter[str] = Counter()
    positive_tiles = 0
    negative_tiles = 0
    for image in payload["images"]:
        image_id = str(image["id"])
        image_path = images_dir / image["filename"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        width, height = int(image["width"]), int(image["height"])
        tiles = [
            (left, top, min(left + tile_size, width), min(top + tile_size, height))
            for top in grid_starts(height, tile_size, overlap)
            for left in grid_starts(width, tile_size, overlap)
        ]
        assigned: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for category_id, annotation in annotations_by_image.get(image_id, []):
            if category_id not in categories:
                continue
            raw_box = annotation.get("a_bbox")
            if (
                isinstance(raw_box, list)
                and len(raw_box) == 4
                and raw_box[0] == raw_box[2]
                and raw_box[1] == raw_box[3]
            ):
                # Two source tie records are degenerate points with no recoverable
                # visual extent.  Record and exclude them instead of creating a
                # fabricated target.
                invalid_counts[category_id] += 1
                continue
            left, top, right, bottom = parse_deepscores_bbox(annotation)
            if right == left:
                left = max(0.0, left - 1.0)
                right = min(float(width), right + 1.0)
                expanded_line_counts[category_id] += 1
            if bottom == top:
                top = max(0.0, top - 1.0)
                bottom = min(float(height), bottom + 1.0)
                expanded_line_counts[category_id] += 1
            if left < 0 or top < 0 or right > width or bottom > height:
                raise ValueError(
                    f"DeepScores box outside image {image_id}: "
                    f"{(left, top, right, bottom)} vs {(width, height)}"
                )
            box = (left, top, right, bottom)
            tile_index = choose_tile(box, tiles, minimum_fraction)
            if tile_index is None:
                dropped_counts[category_id] += 1
                continue
            tile = tiles[tile_index]
            clipped = [
                max(0.0, left - tile[0]),
                max(0.0, top - tile[1]),
                min(float(tile[2] - tile[0]), right - tile[0]),
                min(float(tile[3] - tile[1]), bottom - tile[1]),
            ]
            assigned[tile_index].append(
                {
                    "box_xyxy": [round(value, 3) for value in clipped],
                    "category_id": category_id,
                    "label": class_index[category_id],
                }
            )
            tiled_counts[category_id] += 1

        for tile_index, tile in enumerate(tiles):
            objects = assigned.get(tile_index, [])
            if objects:
                positive_tiles += 1
            elif _stable_unit(
                f"{split}\0{image_id}\0{tile_index}"
            ) >= negative_ratio:
                continue
            else:
                negative_tiles += 1
            rows.append(
                {
                    "split": split,
                    "image": image["filename"],
                    "image_id": image_id,
                    "crop_xyxy": list(tile),
                    "objects": objects,
                }
            )
    _write_jsonl(output_path, rows)
    report = {
        "split": split,
        "source_annotation": str(annotation_path.resolve()),
        "source_annotation_sha256": sha256_file(annotation_path),
        "source_images": len(payload["images"]),
        "source_annotations": len(payload["annotations"]),
        "tile_size": tile_size,
        "overlap": overlap,
        "minimum_object_fraction": minimum_fraction,
        "negative_ratio": negative_ratio,
        "source_bbox_format": "xyxy",
        "source_oriented_bbox_verified": True,
        "tiles": len(rows),
        "positive_tiles": positive_tiles,
        "negative_tiles": negative_tiles,
        "target_instances": sum(tiled_counts.values()),
        "dropped_target_instances": sum(dropped_counts.values()),
        "invalid_source_instances": sum(invalid_counts.values()),
        "expanded_zero_thickness_line_instances": sum(expanded_line_counts.values()),
        "class_counts": {
            categories[category_id]["name"]: tiled_counts[category_id]
            for category_id in ordered_ids
        },
        "invalid_source_class_counts": {
            categories[category_id]["name"]: invalid_counts[category_id]
            for category_id in ordered_ids
            if invalid_counts[category_id]
        },
        "expanded_line_class_counts": {
            categories[category_id]["name"]: expanded_line_counts[category_id]
            for category_id in ordered_ids
            if expanded_line_counts[category_id]
        },
    }
    return report, categories


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--minimum-object-fraction", type=float, default=0.8)
    parser.add_argument("--negative-ratio", type=float, default=0.08)
    parser.add_argument("--minimum-train-instances", type=int, default=20)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not 0 <= args.negative_ratio <= 1:
        raise ValueError("--negative-ratio must be in [0, 1]")
    if not 0 < args.minimum_object_fraction <= 1:
        raise ValueError("--minimum-object-fraction must be in (0, 1]")
    args.output_dir.mkdir(parents=True)
    images = args.dataset_root / "images"
    train_report, categories = prepare_split(
        annotation_path=args.dataset_root / "deepscores_train.json",
        images_dir=images,
        output_path=args.output_dir / "train.jsonl",
        split="train",
        tile_size=args.tile_size,
        overlap=args.overlap,
        minimum_fraction=args.minimum_object_fraction,
        negative_ratio=args.negative_ratio,
        selected_categories=None,
        minimum_train_instances=args.minimum_train_instances,
    )
    test_report, _ = prepare_split(
        annotation_path=args.dataset_root / "deepscores_test.json",
        images_dir=images,
        output_path=args.output_dir / "test.jsonl",
        split="test",
        tile_size=args.tile_size,
        overlap=args.overlap,
        minimum_fraction=args.minimum_object_fraction,
        negative_ratio=args.negative_ratio,
        selected_categories=categories,
        minimum_train_instances=args.minimum_train_instances,
    )
    ordered_ids = sorted(categories, key=lambda value: int(value))
    category_manifest = {
        "background_label": 0,
        "classes": [
            {
                "source_category_id": category_id,
                "label": index + 1,
                "name": categories[category_id]["name"],
            }
            for index, category_id in enumerate(ordered_ids)
        ],
    }
    (args.output_dir / "categories.json").write_text(
        json.dumps(category_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "format": 1,
        "name": "scorescan-deepscores-v2-expression-tiles-v3",
        "license": "CC-BY-4.0",
        "license_url": "https://zenodo.org/records/4012193",
        "source_split_overlap": 0,
        "classes": len(ordered_ids),
        "train": train_report,
        "test": test_report,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
