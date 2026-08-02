#!/usr/bin/env python3
"""Audit semantic labels against immutable complete-page SVG geometry."""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
from pathlib import Path
from typing import Any

from app.tools.prepare_openscore_svg_regions import (
    COMPLETE_PAGE_TARGET_PROVENANCE,
    LONG_SPAN_SEMANTIC_CATEGORIES,
)
from app.tools.semantic_target_visibility import (
    OVERSIZED_FRAGMENT_VISIBILITY_VERSION,
    target_fragment_is_visible,
)
from app.tools.train_deepscores_symbol_detector import sha256_file


AUDIT_VERSION = "complete-page-semantic-target-audit@2"


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


def _is_long_span(category: str) -> bool:
    return (
        category in LONG_SPAN_SEMANTIC_CATEGORIES
        or category.casefold().endswith("text")
    )


def audit_dataset(
    prepared_dir: Path,
    *,
    splits: tuple[str, ...],
) -> dict[str, Any]:
    prepared_dir = prepared_dir.resolve(strict=True)
    manifest_path = prepared_dir / "manifest.json"
    preparation_path = prepared_dir / "prepare-report.json"
    if not manifest_path.is_file() or not preparation_path.is_file():
        raise FileNotFoundError("semantic manifest/preparation is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(preparation, dict):
        raise ValueError("semantic manifest/preparation must be JSON objects")
    minimum_fraction = float(
        preparation.get(
            "minimum_object_fraction",
            manifest.get("minimum_object_fraction", 0.8),
        )
    )
    long_span_minimum_fraction = float(
        preparation.get(
            "long_span_minimum_object_fraction",
            manifest.get(
                "long_span_minimum_object_fraction",
                minimum_fraction,
            ),
        )
    )
    if not (
        0
        < long_span_minimum_fraction
        <= minimum_fraction
        <= 1
    ):
        raise ValueError("semantic visibility floors are invalid")
    tile_size = int(preparation.get("tile_size", manifest.get("tile_size", 0)))
    tile_overlap = float(
        preparation.get("overlap", manifest.get("overlap", 0))
    )
    if tile_size <= 0 or tile_overlap <= 0 or tile_overlap >= tile_size:
        raise ValueError("semantic tile geometry is invalid")
    for name, payload in (("manifest", manifest), ("preparation", preparation)):
        if (
            payload.get("oversized_fragment_visibility_version")
            != OVERSIZED_FRAGMENT_VISIBILITY_VERSION
        ):
            raise ValueError(
                f"semantic {name} has stale oversized-fragment visibility"
            )

    failures: list[str] = []
    dropped = preparation.get("dropped_object_counts")
    dropped = dropped if isinstance(dropped, dict) else {}
    dropped_long_span = {
        str(name): int(value)
        for name, value in dropped.items()
        if _is_long_span(str(name)) and int(value) > 0
    }
    if dropped_long_span:
        failures.append(
            "page-spanning source objects were dropped: "
            + ", ".join(
                f"{name}={count}"
                for name, count in sorted(dropped_long_span.items())
            )
        )

    split_reports: dict[str, dict[str, Any]] = {}
    global_sources: dict[
        tuple[str, str, str],
        tuple[str, int, tuple[float, float, float, float]],
    ] = {}
    for split in splits:
        split_path = prepared_dir / f"{split}.jsonl"
        if not split_path.is_file():
            raise FileNotFoundError(split_path)
        rows = 0
        target_instances = 0
        partial_instances = 0
        category_instances: collections.Counter[str] = collections.Counter()
        category_sources: collections.Counter[str] = collections.Counter()
        split_sources: set[tuple[str, str, str]] = set()
        with split_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    failures.append(f"{split}:{line_number}: row is not an object")
                    continue
                rows += 1
                page_key = (
                    str(row.get("source_key") or ""),
                    str(row.get("image") or ""),
                )
                try:
                    crop = tuple(float(value) for value in row["crop_xyxy"])
                except (KeyError, TypeError, ValueError):
                    failures.append(f"{split}:{line_number}: invalid crop")
                    continue
                if (
                    not all(page_key)
                    or len(crop) != 4
                    or crop[2] <= crop[0]
                    or crop[3] <= crop[1]
                    or not all(math.isfinite(value) for value in crop)
                ):
                    failures.append(f"{split}:{line_number}: invalid page/crop")
                    continue
                row_source_ids: set[str] = set()
                for obj in row.get("objects", []):
                    target_instances += 1
                    try:
                        category = str(obj["category_id"])
                        label = int(obj["label"])
                        source_id = str(obj["source_object_id"])
                        local = tuple(float(value) for value in obj["box_xyxy"])
                        page_box = tuple(
                            float(value) for value in obj["page_box_xyxy"]
                        )
                    except (KeyError, TypeError, ValueError):
                        failures.append(
                            f"{split}:{line_number}: malformed target"
                        )
                        continue
                    if (
                        obj.get("target_geometry_provenance")
                        != COMPLETE_PAGE_TARGET_PROVENANCE
                        or not source_id
                        or len(local) != 4
                        or len(page_box) != 4
                        or _area(local) <= 0
                        or _area(page_box) <= 0
                        or not all(
                            math.isfinite(value)
                            for value in (*local, *page_box)
                        )
                    ):
                        failures.append(
                            f"{split}:{line_number}: invalid complete-page target"
                        )
                        continue
                    expected = _intersection(page_box, crop)
                    expected_local = (
                        expected[0] - crop[0],
                        expected[1] - crop[1],
                        expected[2] - crop[0],
                        expected[3] - crop[1],
                    )
                    if (
                        _area(expected) <= 0
                        or max(
                            abs(observed - wanted)
                            for observed, wanted in zip(local, expected_local)
                        )
                        > 0.01
                    ):
                        failures.append(
                            f"{split}:{line_number}: local/page box contradiction"
                        )
                        continue
                    visible_fraction = _area(expected) / _area(page_box)
                    if not target_fragment_is_visible(
                        page_box,
                        crop,
                        minimum_fraction=minimum_fraction,
                        long_span_minimum_fraction=long_span_minimum_fraction,
                        is_long_span=_is_long_span(category),
                        tile_overlap=tile_overlap,
                    ):
                        failures.append(
                            f"{split}:{line_number}: {category} visible fraction "
                            f"{visible_fraction:.6f} has no valid fragment"
                        )
                    source_key = (page_key[0], page_key[1], source_id)
                    source_value = (
                        category,
                        label,
                        tuple(round(value, 4) for value in page_box),
                    )
                    previous = global_sources.get(source_key)
                    if previous is not None and previous != source_value:
                        failures.append(
                            f"{split}:{line_number}: source object changed geometry"
                        )
                    elif previous is None:
                        global_sources[source_key] = source_value
                        split_sources.add(source_key)
                        category_sources[category] += 1
                    if source_id in row_source_ids:
                        failures.append(
                            f"{split}:{line_number}: duplicate source object in tile"
                        )
                    row_source_ids.add(source_id)
                    category_instances[category] += 1
                    if visible_fraction < 1.0 - 1e-9:
                        partial_instances += 1
        split_reports[split] = {
            "rows": rows,
            "target_instances": target_instances,
            "unique_source_objects": len(split_sources),
            "partial_target_instances": partial_instances,
            "target_instances_by_category": dict(
                sorted(category_instances.items())
            ),
            "unique_sources_by_category": dict(
                sorted(category_sources.items())
            ),
            "jsonl_sha256": sha256_file(split_path),
        }
    return {
        "format": 1,
        "audit_version": AUDIT_VERSION,
        "passed": not failures,
        "prepared_dir": str(prepared_dir),
        "manifest_sha256": sha256_file(manifest_path),
        "prepare_report_sha256": sha256_file(preparation_path),
        "target_geometry_provenance": COMPLETE_PAGE_TARGET_PROVENANCE,
        "minimum_object_fraction": minimum_fraction,
        "long_span_minimum_object_fraction": long_span_minimum_fraction,
        "tile_size": tile_size,
        "overlap": tile_overlap,
        "oversized_fragment_visibility_version": (
            OVERSIZED_FRAGMENT_VISIBILITY_VERSION
        ),
        "dropped_long_span_objects": dict(sorted(dropped_long_span.items())),
        "splits": split_reports,
        "failures": failures,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "calibration", "test"),
    )
    args = parser.parse_args(argv)
    splits = tuple(args.split or ("train", "calibration", "test"))
    if len(set(splits)) != len(splits):
        raise ValueError("audit splits must be unique")
    report = audit_dataset(args.prepared_dir, splits=splits)
    _atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
