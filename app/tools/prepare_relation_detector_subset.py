#!/usr/bin/env python3
from __future__ import annotations

"""Build a compact, source-isolated detector dataset for curved relations.

The general semantic detector learns dozens of visually unrelated classes.  This
preparation keeps complete-page boxes for slurs, ties and hairpins, while treating
the remaining printed score as hard background.  It never changes train/test source
membership and never promotes the derived data to release evidence.
"""

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from scorescan.util import atomic_write_json, atomic_write_text, sha256_file, utc_now_iso


SUBSET_CONTRACT_VERSION = "relation-detector-class-subset@2"
DEFAULT_CLASSES = ("slur", "tie", "hairpin")
DEFAULT_MINIMUM_TEST_SOURCES_PER_CLASS = 5
DEFAULT_MINIMUM_TEST_UNIQUE_OBJECTS_PER_CLASS = 50
EXPECTED_TARGET_ASSIGNMENT_VERSION = (
    "complete-page-overlap-consistent-deduplicated-semantic-targets@4"
)
EXPECTED_TARGET_GEOMETRY_PROVENANCE = (
    "complete-page-svg-geometry-before-tile-clipping@1"
)


def _stable_negative_rank(row: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "image": row.get("image"),
            "image_id": row.get("image_id"),
            "crop_xyxy": row.get("crop_xyxy"),
            "source_key": row.get("source_key"),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            yield payload


def _filter_split(
    path: Path,
    *,
    labels: dict[str, int],
    limit_negatives: bool,
    negative_ratio: float,
) -> tuple[list[dict[str, Any]], Counter[str], int]:
    positives: list[tuple[int, dict[str, Any]]] = []
    negatives: list[tuple[str, int, dict[str, Any]]] = []
    object_counts: Counter[str] = Counter()
    for index, source in enumerate(_read_jsonl(path)):
        row = copy.deepcopy(source)
        objects: list[dict[str, Any]] = []
        for source_object in source.get("objects", []) or []:
            if not isinstance(source_object, dict):
                raise ValueError(f"{path}: object target must be an object")
            class_name = str(source_object.get("category_id") or "")
            if class_name not in labels:
                continue
            target = copy.deepcopy(source_object)
            target["category_id"] = class_name
            target["label"] = labels[class_name]
            objects.append(target)
            object_counts[class_name] += 1
        row["objects"] = objects
        if objects:
            positives.append((index, row))
        else:
            negatives.append((_stable_negative_rank(row), index, row))

    if limit_negatives:
        maximum = int(round(len(positives) * negative_ratio))
        selected_negatives = sorted(negatives)[:maximum]
    else:
        selected_negatives = negatives
    ordered = sorted(
        [*positives, *((index, row) for _rank, index, row in selected_negatives)],
        key=lambda item: item[0],
    )
    return [row for _index, row in ordered], object_counts, len(selected_negatives)


def _jsonl_payload(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )


def _object_identity(
    row: dict[str, Any],
    target: dict[str, Any],
) -> str:
    source_object_id = str(target.get("source_object_id") or "").strip()
    if source_object_id:
        return source_object_id
    payload = json.dumps(
        {
            "source": row.get("source_key") or row.get("image_id"),
            "category": target.get("category_id"),
            "page_box_xyxy": target.get("page_box_xyxy")
            or target.get("box_xyxy"),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "fallback-" + hashlib.sha256(payload).hexdigest()


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def _split_quality_audit(
    rows: Iterable[dict[str, Any]],
    *,
    classes: tuple[str, ...],
) -> dict[str, Any]:
    instances: Counter[str] = Counter()
    clipped_instances: Counter[str] = Counter()
    sources: dict[str, set[str]] = {name: set() for name in classes}
    unique_objects: dict[str, set[tuple[str, str]]] = {
        name: set() for name in classes
    }
    page_widths: dict[str, list[float]] = {name: [] for name in classes}
    page_aspect_ratios: dict[str, list[float]] = {
        name: [] for name in classes
    }
    unique_geometry: dict[str, dict[tuple[str, str], tuple[float, float]]] = {
        name: {} for name in classes
    }
    for row in rows:
        source = str(row.get("source_key") or row.get("image_id") or "")
        for target in row.get("objects", []) or []:
            category = str(target.get("category_id") or "")
            if category not in sources:
                continue
            instances[category] += 1
            sources[category].add(source)
            identity = (source, _object_identity(row, target))
            unique_objects[category].add(identity)
            box = target.get("box_xyxy")
            page_box = target.get("page_box_xyxy") or box
            if not (
                isinstance(box, list)
                and len(box) == 4
                and isinstance(page_box, list)
                and len(page_box) == 4
            ):
                continue
            coordinates = [*box, *page_box]
            if not all(isinstance(value, (int, float)) for value in coordinates):
                continue
            tile_width = max(0.0, float(box[2]) - float(box[0]))
            tile_height = max(0.0, float(box[3]) - float(box[1]))
            page_width = max(0.0, float(page_box[2]) - float(page_box[0]))
            page_height = max(0.0, float(page_box[3]) - float(page_box[1]))
            # Tile boxes are crop-local while page boxes are page-global, so their
            # absolute coordinates are intentionally different.  Clipping changes
            # dimensions; translation alone must not be counted as clipping.
            if (
                abs(tile_width - page_width) > 1e-4
                or abs(tile_height - page_height) > 1e-4
            ):
                clipped_instances[category] += 1
            if page_width > 0 and page_height > 0:
                unique_geometry[category].setdefault(
                    identity,
                    (page_width, page_width / page_height),
                )

    result: dict[str, Any] = {}
    for category in classes:
        geometry = list(unique_geometry[category].values())
        page_widths[category] = [width for width, _ratio in geometry]
        page_aspect_ratios[category] = [ratio for _width, ratio in geometry]
        instance_count = instances[category]
        result[category] = {
            "instances": instance_count,
            "unique_objects": len(unique_objects[category]),
            "sources": len(sources[category]),
            "duplicate_assignment_factor": (
                instance_count / len(unique_objects[category])
                if unique_objects[category]
                else None
            ),
            "clipped_instance_fraction": (
                clipped_instances[category] / instance_count
                if instance_count
                else None
            ),
            "page_width_px": {
                "median": _percentile(page_widths[category], 0.5),
                "p95": _percentile(page_widths[category], 0.95),
                "maximum": _percentile(page_widths[category], 1.0),
            },
            "page_aspect_ratio": {
                "median": _percentile(page_aspect_ratios[category], 0.5),
                "p95": _percentile(page_aspect_ratios[category], 0.95),
                "maximum": _percentile(page_aspect_ratios[category], 1.0),
                "fraction_above_20": (
                    sum(value > 20 for value in page_aspect_ratios[category])
                    / len(page_aspect_ratios[category])
                    if page_aspect_ratios[category]
                    else None
                ),
            },
        }
    return result


def prepare_relation_detector_subset(
    source_dir: Path,
    output_dir: Path,
    *,
    classes: tuple[str, ...] = DEFAULT_CLASSES,
    train_negative_ratio: float = 1.0,
    minimum_test_sources_per_class: int = (
        DEFAULT_MINIMUM_TEST_SOURCES_PER_CLASS
    ),
    minimum_test_unique_objects_per_class: int = (
        DEFAULT_MINIMUM_TEST_UNIQUE_OBJECTS_PER_CLASS
    ),
) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("classes must be unique and non-empty")
    if train_negative_ratio < 0:
        raise ValueError("train_negative_ratio cannot be negative")
    if minimum_test_sources_per_class <= 0:
        raise ValueError("minimum_test_sources_per_class must be positive")
    if minimum_test_unique_objects_per_class <= 0:
        raise ValueError("minimum_test_unique_objects_per_class must be positive")

    required = (
        source_dir / "manifest.json",
        source_dir / "prepare-report.json",
        source_dir / "categories.json",
        source_dir / "train.jsonl",
        source_dir / "test.jsonl",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    source_manifest = json.loads(required[0].read_text(encoding="utf-8"))
    source_preparation = json.loads(required[1].read_text(encoding="utf-8"))
    source_categories = json.loads(required[2].read_text(encoding="utf-8"))
    if (
        source_manifest.get("target_assignment_version")
        != EXPECTED_TARGET_ASSIGNMENT_VERSION
    ):
        raise ValueError(
            "source dataset does not use complete-page overlap-consistent targets"
        )
    if (
        source_manifest.get("target_geometry_provenance")
        != EXPECTED_TARGET_GEOMETRY_PROVENANCE
    ):
        raise ValueError("source dataset lacks complete-page target provenance")
    available = {
        str(row.get("name") or ""): row
        for row in source_categories.get("classes", [])
        if isinstance(row, dict)
    }
    missing = sorted(set(classes) - set(available))
    if missing:
        raise ValueError("source categories are missing: " + ", ".join(missing))
    labels = {name: index for index, name in enumerate(classes, start=1)}

    split_rows: dict[str, list[dict[str, Any]]] = {}
    split_counts: dict[str, Counter[str]] = {}
    split_negatives: dict[str, int] = {}
    split_quality: dict[str, dict[str, Any]] = {}
    for split in ("train", "test"):
        rows, counts, negatives = _filter_split(
            source_dir / f"{split}.jsonl",
            labels=labels,
            limit_negatives=split == "train",
            negative_ratio=train_negative_ratio,
        )
        if not rows or not counts:
            raise ValueError(f"derived {split} split has no positive relation targets")
        split_rows[split] = rows
        split_counts[split] = counts
        split_negatives[split] = negatives
        split_quality[split] = _split_quality_audit(rows, classes=classes)

    quality_failures: list[str] = []
    for class_name in classes:
        audit = split_quality["test"][class_name]
        if audit["sources"] < minimum_test_sources_per_class:
            quality_failures.append(
                f"{class_name}:sources={audit['sources']}"
                f"<{minimum_test_sources_per_class}"
            )
        if audit["unique_objects"] < minimum_test_unique_objects_per_class:
            quality_failures.append(
                f"{class_name}:unique_objects={audit['unique_objects']}"
                f"<{minimum_test_unique_objects_per_class}"
            )
    if quality_failures:
        raise ValueError(
            "derived test split lacks independent class support: "
            + "; ".join(quality_failures)
        )

    categories = {
        "classes": [
            {
                "label": labels[name],
                "name": name,
                "source": (
                    "complete-page registered geometry; relation-detector subset"
                ),
            }
            for name in classes
        ]
    }
    output_dir.mkdir(parents=True)
    for split, rows in split_rows.items():
        atomic_write_text(output_dir / f"{split}.jsonl", _jsonl_payload(rows))
    atomic_write_json(output_dir / "categories.json", categories)

    manifest = copy.deepcopy(source_manifest)
    manifest.update(
        {
            "name": "scorescan-relation-detector-complete-page-subset-v2",
            "created_at": utc_now_iso(),
            "classes": len(classes),
            "class_subset": list(classes),
            "class_subset_contract": SUBSET_CONTRACT_VERSION,
            "source_prepared_dir": str(source_dir),
            "source_prepared_manifest_sha256": sha256_file(
                source_dir / "manifest.json"
            ),
            "production_evidence_eligible": False,
            "release_evidence_eligible": False,
            "minimum_test_sources_per_class": minimum_test_sources_per_class,
            "minimum_test_unique_objects_per_class": (
                minimum_test_unique_objects_per_class
            ),
        }
    )
    for split in ("train", "test"):
        manifest[split] = {
            "tiles": len(split_rows[split]),
            "negative_tiles": split_negatives[split],
            "sources": len(
                {
                    str(row.get("source_key") or row.get("image_id") or "")
                    for row in split_rows[split]
                }
            ),
            "object_counts": dict(sorted(split_counts[split].items())),
            "class_quality": split_quality[split],
        }
        manifest[f"{split}_jsonl_sha256"] = sha256_file(
            output_dir / f"{split}.jsonl"
        )
    atomic_write_json(output_dir / "manifest.json", manifest)

    preparation = copy.deepcopy(source_preparation)
    preparation.update(
        {
            "name": "scorescan-relation-detector-complete-page-subset-v2",
            "created_at": utc_now_iso(),
            "purpose": "training_only_relation_detector_ablation",
            "class_subset": list(classes),
            "class_subset_contract": SUBSET_CONTRACT_VERSION,
            "source_prepared_dir": str(source_dir),
            "source_prepared_manifest_sha256": sha256_file(
                source_dir / "manifest.json"
            ),
            "production_evidence_eligible": False,
            "object_counts": {
                split: dict(sorted(split_counts[split].items()))
                for split in ("train", "test")
            },
            "negative_tiles_by_split": dict(sorted(split_negatives.items())),
            "class_quality_by_split": split_quality,
            "minimum_test_sources_per_class": minimum_test_sources_per_class,
            "minimum_test_unique_objects_per_class": (
                minimum_test_unique_objects_per_class
            ),
            "tiles_by_split": {
                split: len(split_rows[split]) for split in ("train", "test")
            },
            "train_jsonl_sha256": sha256_file(output_dir / "train.jsonl"),
            "test_jsonl_sha256": sha256_file(output_dir / "test.jsonl"),
        }
    )
    atomic_write_json(output_dir / "prepare-report.json", preparation)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--classes",
        default=",".join(DEFAULT_CLASSES),
        help="comma-separated source category names",
    )
    parser.add_argument("--train-negative-ratio", type=float, default=1.0)
    parser.add_argument(
        "--minimum-test-sources-per-class",
        type=int,
        default=DEFAULT_MINIMUM_TEST_SOURCES_PER_CLASS,
    )
    parser.add_argument(
        "--minimum-test-unique-objects-per-class",
        type=int,
        default=DEFAULT_MINIMUM_TEST_UNIQUE_OBJECTS_PER_CLASS,
    )
    args = parser.parse_args()
    classes = tuple(item.strip() for item in args.classes.split(",") if item.strip())
    report = prepare_relation_detector_subset(
        args.source_dir,
        args.output_dir,
        classes=classes,
        train_negative_ratio=args.train_negative_ratio,
        minimum_test_sources_per_class=args.minimum_test_sources_per_class,
        minimum_test_unique_objects_per_class=(
            args.minimum_test_unique_objects_per_class
        ),
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
