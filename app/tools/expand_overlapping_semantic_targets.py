#!/usr/bin/env python3
"""Remove contradictory labels from retained overlapping detector tiles.

The original semantic tiler assigns every SVG object to exactly one preferred
tile. With overlapping crops, the same ink can remain fully visible in another
retained tile and is then accidentally presented as background. This immutable
derived layer reconstructs page-space objects from their owner tiles and labels
them in every retained crop meeting the same visibility floor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

from app.tools.prepare_openscore_svg_regions import (
    COMPLETE_PAGE_TARGET_ASSIGNMENT_VERSION,
    COMPLETE_PAGE_TARGET_PROVENANCE,
    LONG_SPAN_SEMANTIC_CATEGORIES,
)
from app.tools.semantic_target_visibility import (
    OVERSIZED_FRAGMENT_VISIBILITY_VERSION,
    intersection_box,
    target_fragment_is_visible,
)
from app.tools.train_deepscores_symbol_detector import sha256_file

TRANSFORMATION_VERSION = COMPLETE_PAGE_TARGET_ASSIGNMENT_VERSION


def _page_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("source_key") or ""),
        str(row.get("image") or ""),
        str(row.get("image_id") or ""),
    )


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(
    box: tuple[float, float, float, float],
    crop: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return (
        max(box[0], crop[0]),
        max(box[1], crop[1]),
        min(box[2], crop[2]),
        min(box[3], crop[3]),
    )


def _object_id(
    page_key: tuple[str, str, str],
    category: str,
    box: tuple[float, float, float, float],
) -> str:
    material = "\0".join(
        (
            *page_key,
            category,
            ",".join(f"{value:.4f}" for value in box),
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def expand_page_rows(
    rows: list[dict[str, Any]],
    *,
    minimum_visible_fraction: float,
    long_span_minimum_visible_fraction: float | None = None,
    tile_overlap: float = 256.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        raise ValueError("semantic page group is empty")
    if not 0 < minimum_visible_fraction <= 1:
        raise ValueError("minimum visible fraction must be in (0, 1]")
    if long_span_minimum_visible_fraction is None:
        long_span_minimum_visible_fraction = minimum_visible_fraction
    if not (
        0
        < long_span_minimum_visible_fraction
        <= minimum_visible_fraction
    ):
        raise ValueError(
            "long-span minimum visible fraction must be in "
            "(0, minimum visible fraction]"
        )
    key = _page_key(rows[0])
    if not all(_page_key(row) == key for row in rows):
        raise ValueError("semantic page group contains multiple pages")

    page_objects: list[dict[str, Any]] = []
    objects_by_visual_key: dict[
        tuple[str, tuple[float, float, float, float]],
        dict[str, Any],
    ] = {}
    unique_counts: Counter[str] = Counter()
    duplicate_counts: Counter[str] = Counter()
    input_target_instances = 0
    for row in rows:
        crop = tuple(float(value) for value in row["crop_xyxy"])
        if len(crop) != 4 or crop[2] <= crop[0] or crop[3] <= crop[1]:
            raise ValueError("semantic tile crop is invalid")
        for obj in row.get("objects", []):
            input_target_instances += 1
            local = tuple(float(value) for value in obj["box_xyxy"])
            if len(local) != 4 or local[2] <= local[0] or local[3] <= local[1]:
                raise ValueError("semantic target box is invalid")
            if (
                obj.get("target_geometry_provenance")
                != COMPLETE_PAGE_TARGET_PROVENANCE
            ):
                raise ValueError(
                    "semantic target has no complete-page geometry provenance"
                )
            raw_page_box = obj.get("page_box_xyxy")
            if not isinstance(raw_page_box, list):
                raise ValueError(
                    "semantic target has no complete page-space box"
                )
            global_box = tuple(float(value) for value in raw_page_box)
            if (
                len(global_box) != 4
                or global_box[2] <= global_box[0]
                or global_box[3] <= global_box[1]
            ):
                raise ValueError("semantic complete page-space box is invalid")
            expected_intersection = _intersection(global_box, crop)
            expected_local = (
                expected_intersection[0] - crop[0],
                expected_intersection[1] - crop[1],
                expected_intersection[2] - crop[0],
                expected_intersection[3] - crop[1],
            )
            if (
                _area(expected_intersection) <= 0
                or max(
                    abs(observed - expected)
                    for observed, expected in zip(local, expected_local)
                )
                > 0.01
            ):
                raise ValueError(
                    "tile-local target contradicts its complete page-space box"
                )
            category = str(obj["category_id"])
            rounded_global_box = tuple(
                round(value, 4) for value in global_box
            )
            visual_key = (category, rounded_global_box)
            template = {
                name: value
                for name, value in obj.items()
                if name
                not in {
                    "box_xyxy",
                    "page_box_xyxy",
                    "source_object_id",
                    "target_geometry_provenance",
                }
            }
            previous = objects_by_visual_key.get(visual_key)
            if previous is not None:
                if int(previous["template"]["label"]) != int(template["label"]):
                    raise ValueError(
                        "identical semantic visual target has conflicting labels"
                    )
                duplicate_counts[category] += 1
                continue
            source = {
                "global_box": rounded_global_box,
                "source_object_id": str(
                    obj.get("source_object_id")
                    or _object_id(key, category, rounded_global_box)
                ),
                "template": template,
            }
            objects_by_visual_key[visual_key] = source
            page_objects.append(source)
            unique_counts[category] += 1

    expanded: list[dict[str, Any]] = []
    instance_counts: Counter[str] = Counter()
    rows_with_added_targets = 0
    additional_instances = 0
    object_assignments: Counter[str] = Counter()
    for row in rows:
        crop = tuple(float(value) for value in row["crop_xyxy"])
        objects: list[dict[str, Any]] = []
        original_visual_keys = {
            (
                str(obj["category_id"]),
                tuple(
                    round(float(value) + crop[index % 2], 4)
                    for index, value in enumerate(obj["box_xyxy"])
                ),
            )
            for obj in row.get("objects", [])
        }
        original_count = len(original_visual_keys)
        for source in page_objects:
            box = source["global_box"]
            intersection = intersection_box(box, crop)
            category = str(source["template"]["category_id"])
            is_long_span = (
                category in LONG_SPAN_SEMANTIC_CATEGORIES
                or category.casefold().endswith("text")
            )
            if not target_fragment_is_visible(
                box,
                crop,
                minimum_fraction=minimum_visible_fraction,
                long_span_minimum_fraction=long_span_minimum_visible_fraction,
                is_long_span=is_long_span,
                tile_overlap=tile_overlap,
            ):
                continue
            template = dict(source["template"])
            template["box_xyxy"] = [
                round(intersection[0] - crop[0], 4),
                round(intersection[1] - crop[1], 4),
                round(intersection[2] - crop[0], 4),
                round(intersection[3] - crop[1], 4),
            ]
            template["page_box_xyxy"] = list(box)
            template["source_object_id"] = source["source_object_id"]
            template["target_geometry_provenance"] = (
                COMPLETE_PAGE_TARGET_PROVENANCE
            )
            objects.append(template)
            object_assignments[source["source_object_id"]] += 1
            instance_counts[str(template["category_id"])] += 1
        objects.sort(
            key=lambda obj: (
                int(obj["label"]),
                tuple(float(value) for value in obj["box_xyxy"]),
                str(obj["source_object_id"]),
            )
        )
        if len(objects) < original_count:
            raise RuntimeError(
                "overlap expansion lost a deduplicated owner-tile target"
            )
        if len(objects) > original_count:
            rows_with_added_targets += 1
            additional_instances += len(objects) - original_count
        expanded.append({**row, "objects": objects})
    missing = [
        source["source_object_id"]
        for source in page_objects
        if object_assignments[source["source_object_id"]] <= 0
    ]
    if missing:
        raise RuntimeError("overlap expansion lost source objects")
    return expanded, {
        "pages": 1,
        "rows": len(rows),
        "input_target_instances": input_target_instances,
        "duplicate_source_objects_removed": sum(duplicate_counts.values()),
        "unique_objects": len(page_objects),
        "target_instances": sum(instance_counts.values()),
        "additional_target_instances": additional_instances,
        "rows_with_added_targets": rows_with_added_targets,
        "unique_object_counts": dict(unique_counts),
        "duplicate_source_object_counts": dict(duplicate_counts),
        "target_instance_counts": dict(instance_counts),
        "maximum_assignments_per_object": max(
            object_assignments.values(),
            default=0,
        ),
    }


def _groups(lines: Iterable[str]) -> Iterator[list[dict[str, Any]]]:
    current_key: tuple[str, str, str] | None = None
    group: list[dict[str, Any]] = []
    closed: set[tuple[str, str, str]] = set()
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("semantic JSONL row must be an object")
        key = _page_key(row)
        if not all(key):
            raise ValueError("semantic JSONL row has no stable page identity")
        if current_key is None:
            current_key = key
        if key != current_key:
            if current_key in closed:
                raise ValueError("semantic page rows are not contiguous")
            closed.add(current_key)
            yield group
            group = []
            current_key = key
        group.append(row)
    if group:
        if current_key in closed:
            raise ValueError("semantic page rows are not contiguous")
        yield group


def _merge_counter(
    destination: Counter[str],
    values: dict[str, Any],
) -> None:
    destination.update({str(key): int(value) for key, value in values.items()})


def expand_split(
    source: Path,
    destination: Path,
    *,
    minimum_visible_fraction: float,
    long_span_minimum_visible_fraction: float | None = None,
    tile_overlap: float = 256.0,
) -> dict[str, Any]:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    counters: Counter[str] = Counter()
    unique_counts: Counter[str] = Counter()
    duplicate_counts: Counter[str] = Counter()
    instance_counts: Counter[str] = Counter()
    maximum_assignments = 0
    try:
        with source.open("r", encoding="utf-8") as input_file, temporary.open(
            "x",
            encoding="utf-8",
            newline="\n",
        ) as output_file:
            for group in _groups(input_file):
                expanded, report = expand_page_rows(
                    group,
                    minimum_visible_fraction=minimum_visible_fraction,
                    long_span_minimum_visible_fraction=(
                        long_span_minimum_visible_fraction
                    ),
                    tile_overlap=tile_overlap,
                )
                for row in expanded:
                    output_file.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                for name in (
                    "pages",
                    "rows",
                    "input_target_instances",
                    "duplicate_source_objects_removed",
                    "unique_objects",
                    "target_instances",
                    "additional_target_instances",
                    "rows_with_added_targets",
                ):
                    counters[name] += int(report[name])
                _merge_counter(unique_counts, report["unique_object_counts"])
                _merge_counter(
                    duplicate_counts,
                    report["duplicate_source_object_counts"],
                )
                _merge_counter(instance_counts, report["target_instance_counts"])
                maximum_assignments = max(
                    maximum_assignments,
                    int(report["maximum_assignments_per_object"]),
                )
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        **dict(counters),
        "unique_object_counts": dict(sorted(unique_counts.items())),
        "duplicate_source_object_counts": dict(sorted(duplicate_counts.items())),
        "target_instance_counts": dict(sorted(instance_counts.items())),
        "maximum_assignments_per_object": maximum_assignments,
        "source_jsonl_sha256": sha256_file(source),
        "output_jsonl_sha256": sha256_file(destination),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "calibration", "test"),
    )
    parser.add_argument("--minimum-visible-fraction", type=float)
    parser.add_argument(
        "--long-span-minimum-visible-fraction",
        type=float,
        help=(
            "lower visibility floor for page-spanning marks and text; "
            "must not exceed --minimum-visible-fraction"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.prepared_dir.is_dir():
        raise FileNotFoundError(args.prepared_dir)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest_path = args.prepared_dir / "manifest.json"
    categories_path = args.prepared_dir / "categories.json"
    if not manifest_path.is_file() or not categories_path.is_file():
        raise FileNotFoundError("prepared semantic manifest/categories are missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prepare_path = args.prepared_dir / "prepare-report.json"
    prepare_report = (
        json.loads(prepare_path.read_text(encoding="utf-8"))
        if prepare_path.is_file()
        else {}
    )
    minimum_visible_fraction = (
        float(args.minimum_visible_fraction)
        if args.minimum_visible_fraction is not None
        else float(prepare_report.get("minimum_object_fraction", 0.8))
    )
    if not 0 < minimum_visible_fraction <= 1:
        raise ValueError("minimum visible fraction must be in (0, 1]")
    long_span_minimum_visible_fraction = (
        float(args.long_span_minimum_visible_fraction)
        if args.long_span_minimum_visible_fraction is not None
        else float(
            prepare_report.get(
                "long_span_minimum_object_fraction",
                minimum_visible_fraction,
            )
        )
    )
    tile_overlap = float(prepare_report.get("overlap", 0.0))
    if tile_overlap <= 0:
        raise ValueError("source preparation has no valid tile overlap")
    if (
        prepare_report.get("oversized_fragment_visibility_version")
        != OVERSIZED_FRAGMENT_VISIBILITY_VERSION
    ):
        raise ValueError("source preparation has stale fragment visibility")
    if (
        prepare_report.get("target_assignment_version")
        != COMPLETE_PAGE_TARGET_ASSIGNMENT_VERSION
        or manifest.get("target_assignment_version")
        != COMPLETE_PAGE_TARGET_ASSIGNMENT_VERSION
    ):
        raise ValueError("source preparation has stale target assignment")
    prepare_builder_hash = sha256_file(
        Path(__file__).with_name("prepare_openscore_svg_regions.py")
    )
    if (
        prepare_report.get("builder_source_sha256") != prepare_builder_hash
        or manifest.get("builder_source_sha256") != prepare_builder_hash
    ):
        raise ValueError("source preparation has stale builder source")
    if not (
        0
        < long_span_minimum_visible_fraction
        <= minimum_visible_fraction
    ):
        raise ValueError(
            "long-span minimum visible fraction must be in "
            "(0, minimum visible fraction]"
        )
    splits = tuple(args.split or ("train", "calibration", "test"))
    if len(set(splits)) != len(splits):
        raise ValueError("semantic overlap-expansion splits must be unique")
    missing = [
        split
        for split in splits
        if not (args.prepared_dir / f"{split}.jsonl").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "semantic source splits are missing: " + ", ".join(missing)
        )

    staging = args.output_dir.with_name(
        f"{args.output_dir.name}.building-{os.getpid()}"
    )
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        split_reports = {
            split: expand_split(
                args.prepared_dir / f"{split}.jsonl",
                staging / f"{split}.jsonl",
                minimum_visible_fraction=minimum_visible_fraction,
                long_span_minimum_visible_fraction=(
                    long_span_minimum_visible_fraction
                ),
                tile_overlap=tile_overlap,
            )
            for split in splits
        }
        shutil.copyfile(categories_path, staging / "categories.json")
        output_manifest = {
            **manifest,
            "name": (
                f"{manifest.get('name', 'semantic-regions')}"
                "-overlap-consistent"
            ),
            "target_assignment_version": TRANSFORMATION_VERSION,
            "target_assignment": (
                "complete_page_svg_visual_deduplication_then_every_retained_"
                "tile_with_category_appropriate_minimum_visible_fraction"
            ),
            "target_geometry_provenance": COMPLETE_PAGE_TARGET_PROVENANCE,
            "oversized_fragment_visibility_version": (
                OVERSIZED_FRAGMENT_VISIBILITY_VERSION
            ),
            "tile_size": int(prepare_report["tile_size"]),
            "overlap": tile_overlap,
            "minimum_object_fraction": minimum_visible_fraction,
            "long_span_minimum_object_fraction": (
                long_span_minimum_visible_fraction
            ),
            "source_prepared_manifest_sha256": sha256_file(manifest_path),
            "transformed_splits": list(splits),
        }
        (staging / "manifest.json").write_text(
            json.dumps(
                output_manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        output_prepare_report = {
            **prepare_report,
            "format": max(1, int(prepare_report.get("format", 1))),
            "name": (
                f"{prepare_report.get('name', manifest.get('name', 'semantic-regions'))}"
                "-overlap-consistent"
            ),
            "purpose": (
                str(prepare_report.get("purpose") or "")
                + "; contradiction-free overlapping semantic targets"
                + "; exact duplicate visual targets removed"
            ).strip("; "),
            "transformation_version": TRANSFORMATION_VERSION,
            "target_geometry_provenance": COMPLETE_PAGE_TARGET_PROVENANCE,
            "oversized_fragment_visibility_version": (
                OVERSIZED_FRAGMENT_VISIBILITY_VERSION
            ),
            "tile_size": int(prepare_report["tile_size"]),
            "overlap": tile_overlap,
            "minimum_object_fraction": minimum_visible_fraction,
            "long_span_minimum_object_fraction": (
                long_span_minimum_visible_fraction
            ),
            "source_prepared_dir": str(args.prepared_dir.resolve()),
            "source_prepared_manifest_sha256": sha256_file(manifest_path),
            "source_prepare_report_sha256": (
                sha256_file(prepare_path) if prepare_path.is_file() else None
            ),
            "splits": split_reports,
        }
        (staging / "prepare-report.json").write_text(
            json.dumps(
                output_prepare_report,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(staging, args.output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "state": "overlap_consistent_semantic_targets_completed",
                "output_dir": str(args.output_dir),
                "splits": split_reports,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
