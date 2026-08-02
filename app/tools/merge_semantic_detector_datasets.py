#!/usr/bin/env python3
"""Validate and aggregate semantic detector corpora without data leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from app.tools.merge_ocr_training_labels import (
    EMPTY_INTERSECTIONS,
    SPLITS,
    _path_within,
    sha256_file,
)


SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class InputDataset:
    name: str
    directory: Path


def parse_input(value: str) -> InputDataset:
    name, separator, directory = value.partition("=")
    if not separator or not SAFE_NAME.fullmatch(name) or not directory:
        raise argparse.ArgumentTypeError(
            "input must be a lowercase NAME=PATH pair"
        )
    return InputDataset(name, Path(directory).expanduser())


def _load_input(
    spec: InputDataset,
    *,
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[int, str]]:
    directory = spec.directory.resolve()
    if not directory.is_dir() or not _path_within(directory, project_root):
        raise ValueError(f"{spec.name}: dataset is absent or outside project")
    report_path = directory / "prepare-report.json"
    manifest_path = directory / "manifest.json"
    categories_path = directory / "categories.json"
    for path in (report_path, manifest_path, categories_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    category_manifest = json.loads(
        categories_path.read_text(encoding="utf-8")
    )
    if (
        report.get("purpose")
        != "synthetic semantic geometry; not real-scan validation"
        or report.get("split_intersections") != EMPTY_INTERSECTIONS
        or int(manifest.get("source_split_overlap", -1)) != 0
    ):
        raise ValueError(f"{spec.name}: semantic isolation contract failed")
    classes = category_manifest.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError(f"{spec.name}: categories are empty")
    categories = {
        int(row["label"]): str(row["name"])
        for row in classes
        if isinstance(row, dict)
    }
    if len(categories) != len(classes):
        raise ValueError(f"{spec.name}: malformed or duplicate categories")
    sources = report.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"{spec.name}: source manifest is empty")
    return report, manifest, categories


def _image_dimensions(path: Path) -> tuple[int, int]:
    with Image.open(path) as source:
        source.verify()
    with Image.open(path) as source:
        return int(source.width), int(source.height)


def _normalize_object(
    obj: dict[str, Any],
    *,
    categories: dict[int, str],
    crop_width: int,
    crop_height: int,
    minimum_visible_fraction: float,
    location: str,
) -> tuple[dict[str, Any], bool]:
    label = int(obj.get("label", -1))
    if label not in categories or str(obj.get("category_id", "")) != categories[label]:
        raise ValueError(f"{location}: object category mismatch")
    box = obj.get("box_xyxy")
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f"{location}: malformed object box")
    left, top, right, bottom = (float(value) for value in box)
    if (
        not all(math.isfinite(value) for value in (left, top, right, bottom))
        or right <= left
        or bottom <= top
    ):
        raise ValueError(f"{location}: object box is non-finite or degenerate")
    clipped = (
        max(0.0, left),
        max(0.0, top),
        min(float(crop_width), right),
        min(float(crop_height), bottom),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError(f"{location}: object box has no visible crop intersection")
    area = (right - left) * (bottom - top)
    visible_area = (clipped[2] - clipped[0]) * (clipped[3] - clipped[1])
    visible_fraction = visible_area / area
    if visible_fraction + 1e-6 < minimum_visible_fraction:
        raise ValueError(
            f"{location}: object visible fraction {visible_fraction:.6f} "
            f"is below source floor {minimum_visible_fraction:.6f}"
        )
    was_clipped = any(
        abs(original - normalized) > 1e-3
        for original, normalized in zip(
            (left, top, right, bottom),
            clipped,
            strict=True,
        )
    )
    normalized_object = {
        **obj,
        "box_xyxy": [round(value, 4) for value in clipped],
    }
    return normalized_object, was_clipped


def _namespaced_row(
    row: dict[str, Any],
    *,
    spec: InputDataset,
    split: str,
    project_root: Path,
    categories: dict[int, str],
    source_splits: dict[str, str],
    image_dimensions: dict[Path, tuple[int, int]],
    minimum_visible_fraction: float,
    clipped_object_counts: Counter[str],
) -> dict[str, Any]:
    source_key = str(row.get("source_key", ""))
    if row.get("split") != split or source_splits.get(source_key) != split:
        raise ValueError(f"{spec.name}: row source/split mismatch")
    image_value = str(row.get("image", ""))
    image_path = (spec.directory.resolve() / Path(image_value)).resolve()
    if not _path_within(image_path, spec.directory.resolve()):
        raise ValueError(f"{spec.name}: row image escapes dataset")
    if not image_path.is_file() or image_path.stat().st_size <= 0:
        raise FileNotFoundError(image_path)
    if image_path not in image_dimensions:
        image_dimensions[image_path] = _image_dimensions(image_path)
    image_width, image_height = image_dimensions[image_path]
    crop = row.get("crop_xyxy")
    if not isinstance(crop, list) or len(crop) != 4:
        raise ValueError(f"{spec.name}: malformed crop")
    crop_left, crop_top, crop_right, crop_bottom = (
        int(value) for value in crop
    )
    if (
        crop_left < 0
        or crop_top < 0
        or crop_right <= crop_left
        or crop_bottom <= crop_top
        or crop_right > image_width
        or crop_bottom > image_height
    ):
        raise ValueError(f"{spec.name}: crop escapes page image")
    objects = row.get("objects")
    if not isinstance(objects, list):
        raise ValueError(f"{spec.name}: row objects are malformed")
    location = f"{spec.name}:{source_key}:{row.get('image_id', '')}"
    normalized_objects = []
    for obj in objects:
        if not isinstance(obj, dict):
            raise ValueError(f"{location}: malformed object")
        normalized, was_clipped = _normalize_object(
            obj,
            categories=categories,
            crop_width=crop_right - crop_left,
            crop_height=crop_bottom - crop_top,
            minimum_visible_fraction=minimum_visible_fraction,
            location=location,
        )
        normalized_objects.append(normalized)
        if was_clipped:
            clipped_object_counts[str(normalized["category_id"])] += 1
    namespaced_source = f"{spec.name}:{source_key}"
    original_image_id = str(row.get("image_id", ""))
    if not original_image_id:
        raise ValueError(f"{location}: image id is empty")
    image_id = hashlib.sha256(
        f"{spec.name}\0{original_image_id}".encode()
    ).hexdigest()[:20]
    return {
        **row,
        "source_key": namespaced_source,
        "image": image_path.relative_to(project_root).as_posix(),
        "image_id": image_id,
        "objects": normalized_objects,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--input",
        action="append",
        type=parse_input,
        required=True,
        metavar="NAME=PATH",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    specs: list[InputDataset] = args.input
    if not specs or len({spec.name for spec in specs}) != len(specs):
        raise ValueError("at least one input with a unique name is required")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaded = []
    canonical_categories: dict[int, str] | None = None
    content_sources: dict[str, tuple[str, str, str]] = {}
    global_source_splits: dict[str, str] = {}
    input_reports = []
    for spec in specs:
        report, manifest, categories = _load_input(
            spec,
            project_root=project_root,
        )
        if canonical_categories is None:
            canonical_categories = categories
        elif categories != canonical_categories:
            raise ValueError(f"{spec.name}: incompatible category manifest")
        source_splits: dict[str, str] = {}
        for source in report["sources"]:
            source_key = str(source.get("source_key", ""))
            split = str(source.get("split", ""))
            content_hash = str(source.get("source_sha256", "")).casefold()
            if (
                not source_key
                or split not in SPLITS
                or not re.fullmatch(r"[0-9a-f]{64}", content_hash)
            ):
                raise ValueError(f"{spec.name}: malformed source manifest row")
            if source_key in source_splits:
                raise ValueError(f"{spec.name}: duplicate source key")
            source_splits[source_key] = split
            identity = (spec.name, source_key, split)
            previous = content_sources.setdefault(content_hash, identity)
            if previous != identity:
                raise ValueError(
                    "duplicate source content across semantic datasets: "
                    f"{previous} and {identity}"
                )
            global_source_splits[f"{spec.name}:{source_key}"] = split
        minimum_visible_fraction = float(
            report.get("minimum_object_fraction", 0.0)
        )
        if (
            not math.isfinite(minimum_visible_fraction)
            or not 0 < minimum_visible_fraction <= 1
        ):
            raise ValueError(
                f"{spec.name}: invalid minimum_object_fraction"
            )
        loaded.append((spec, source_splits, minimum_visible_fraction))
        input_reports.append(
            {
                "name": spec.name,
                "directory": str(spec.directory.resolve()),
                "prepare_report_sha256": sha256_file(
                    spec.directory / "prepare-report.json"
                ),
                "manifest_sha256": sha256_file(
                    spec.directory / "manifest.json"
                ),
                "categories_sha256": sha256_file(
                    spec.directory / "categories.json"
                ),
                "source_count": len(source_splits),
                "tiles": {
                    split: int(manifest[split]["tiles"])
                    for split in SPLITS
                },
            }
        )
    assert canonical_categories is not None

    output_counts = {split: 0 for split in SPLITS}
    object_counts: Counter[str] = Counter()
    clipped_object_counts: Counter[str] = Counter()
    image_dimensions: dict[Path, tuple[int, int]] = {}
    artifact_paths = []
    for split in SPLITS:
        output_path = args.output_dir / f"{split}.jsonl"
        temporary = output_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as destination:
            for spec, source_splits, minimum_visible_fraction in loaded:
                input_path = spec.directory / f"{split}.jsonl"
                if not input_path.is_file():
                    raise FileNotFoundError(input_path)
                with input_path.open(encoding="utf-8") as source_stream:
                    for line_number, line in enumerate(source_stream, start=1):
                        if not line.strip():
                            continue
                        raw = json.loads(line)
                        row = _namespaced_row(
                            raw,
                            spec=spec,
                            split=split,
                            project_root=project_root,
                            categories=canonical_categories,
                            source_splits=source_splits,
                            image_dimensions=image_dimensions,
                            minimum_visible_fraction=minimum_visible_fraction,
                            clipped_object_counts=clipped_object_counts,
                        )
                        destination.write(
                            json.dumps(
                                row,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                        )
                        destination.write("\n")
                        output_counts[split] += 1
                        object_counts.update(
                            str(obj["category_id"])
                            for obj in row["objects"]
                        )
        os.replace(temporary, output_path)
        artifact_paths.append(output_path)

    categories_path = args.output_dir / "categories.json"
    categories_path.write_text(
        json.dumps(
            {
                "format": 1,
                "classes": [
                    {
                        "label": label,
                        "name": name,
                        "source": "MuseScore SVG semantic class",
                    }
                    for label, name in sorted(canonical_categories.items())
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    sources_by_split = {
        split: sum(value == split for value in global_source_splits.values())
        for split in SPLITS
    }
    aggregation_kind = "combined" if len(specs) > 1 else "normalized"
    manifest = {
        "format": 1,
        "name": f"scorescan-openscore-{aggregation_kind}-semantic-svg-regions",
        "license": "CC0-1.0",
        "role": "training_only_synthetic_semantic_geometry",
        "classes": len(canonical_categories),
        "source_split_overlap": 0,
        "reserved_holdout_overlap": 0,
        "images_dir": str(project_root),
        **{
            split: {
                "tiles": output_counts[split],
                "sources": sources_by_split[split],
            }
            for split in SPLITS
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "name": f"scorescan-openscore-{aggregation_kind}-semantic-regions-v1",
        "purpose": (
            f"{aggregation_kind} synthetic semantic geometry; "
            "not real-scan validation"
        ),
        "license": "CC0-1.0",
        "project_root": str(project_root),
        "inputs": input_reports,
        "source_content_intersection": [],
        "split_intersections": EMPTY_INTERSECTIONS,
        "sources_by_split": sources_by_split,
        "tiles_by_split": output_counts,
        "object_counts": dict(sorted(object_counts.items())),
        "legacy_tile_box_clips": {
            "total": sum(clipped_object_counts.values()),
            "by_category": dict(sorted(clipped_object_counts.items())),
            "policy": (
                "clip only when the retained area satisfies the input "
                "minimum_object_fraction; otherwise fail closed"
            ),
        },
    }
    report_path = args.output_dir / "merge-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    artifact_paths.extend((categories_path, manifest_path, report_path))
    (args.output_dir / "dataset.sha256").write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}"
            for path in artifact_paths
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "sources_by_split": sources_by_split,
                "tiles_by_split": output_counts,
                "object_classes": len(object_counts),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
