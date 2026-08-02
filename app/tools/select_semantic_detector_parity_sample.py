#!/usr/bin/env python3
from __future__ import annotations

"""Select a deterministic real-score tile for semantic ONNX parity checks.

This is an implementation-parity sample, not an accuracy holdout.  Accuracy is
gated separately on the work-disjoint Muse OMR holdout.  The selected tile must
come from the prepared scan test split and contain runtime-relevant ground-truth
objects so a blank-output ONNX export cannot pass accidentally.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image

from scorescan.semantic_detector import SUPPORTED_RUNTIME_CLASSES
from scorescan.util import sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser


def _resolve_image(
    raw_path: object,
    *,
    project_root: Path,
    test_jsonl: Path,
) -> Path:
    value = str(raw_path or "").strip()
    if not value:
        raise ValueError("semantic parity row has no image path")
    path = Path(value)
    candidates = (
        path,
        project_root / path,
        test_jsonl.parent / path,
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(value)


def _candidate_key(row: dict[str, Any]) -> tuple[object, ...] | None:
    crop = row.get("crop_xyxy")
    objects = row.get("objects")
    if (
        row.get("split") != "test"
        or not isinstance(crop, list)
        or len(crop) != 4
        or not isinstance(objects, list)
    ):
        return None
    try:
        coordinates = tuple(int(value) for value in crop)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        coordinates[2] - coordinates[0] != 1024
        or coordinates[3] - coordinates[1] != 1024
        or coordinates[0] < 0
        or coordinates[1] < 0
    ):
        return None
    runtime_objects = [
        item
        for item in objects
        if isinstance(item, dict)
        and str(item.get("category_id") or "") in SUPPORTED_RUNTIME_CLASSES
    ]
    if not runtime_objects:
        return None
    class_names = {
        str(item["category_id"])
        for item in runtime_objects
    }
    geometry_count = len(class_names & {"hairpin", "slur", "tie"})
    text_count = len(class_names - {"hairpin", "slur", "tie"})
    stable_identity = json.dumps(
        {
            "image": row.get("image"),
            "crop_xyxy": coordinates,
            "source_key": row.get("source_key"),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    # Descending quality terms are negated because min() selects the winner.
    return (
        -geometry_count,
        -text_count,
        -len(class_names),
        -len(runtime_objects),
        hashlib.sha256(stable_identity.encode("utf-8")).hexdigest(),
    )


def select_parity_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        (key, row)
        for row in rows
        if (key := _candidate_key(row)) is not None
    ]
    if not candidates:
        raise ValueError(
            "scan test split has no 1024x1024 runtime-relevant parity tile"
        )
    return min(candidates, key=lambda item: item[0])[1]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    test_jsonl = args.test_jsonl.resolve()
    project_root = args.project_root.resolve()
    output_report = args.output_report.resolve()
    if not test_jsonl.is_file():
        raise FileNotFoundError(test_jsonl)
    if output_report.exists():
        raise FileExistsError(output_report)
    rows: list[dict[str, Any]] = []
    with test_jsonl.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"test JSONL row {line_number} is not an object")
            rows.append(payload)
    selected = select_parity_row(rows)
    image_path = _resolve_image(
        selected.get("image"),
        project_root=project_root,
        test_jsonl=test_jsonl,
    )
    crop = [int(value) for value in selected["crop_xyxy"]]
    with Image.open(image_path) as image:
        if crop[2] > image.width or crop[3] > image.height:
            raise ValueError("selected parity crop exceeds its source image")
        image_size = [int(image.width), int(image.height)]
    runtime_classes = sorted(
        {
            str(item["category_id"])
            for item in selected["objects"]
            if isinstance(item, dict)
            and str(item.get("category_id") or "") in SUPPORTED_RUNTIME_CLASSES
        }
    )
    report = {
        "format": 1,
        "name": "scorescan-semantic-detector-parity-sample-v1",
        "selection_role": "real_scan_test_split_implementation_parity_only",
        "test_jsonl": str(test_jsonl),
        "test_jsonl_sha256": sha256_file(test_jsonl),
        "image": str(image_path),
        "image_sha256": sha256_file(image_path),
        "image_size": image_size,
        "crop_xyxy": crop,
        "source_key": str(selected.get("source_key") or ""),
        "image_id": str(selected.get("image_id") or ""),
        "runtime_classes": runtime_classes,
        "runtime_object_count": sum(
            1
            for item in selected["objects"]
            if isinstance(item, dict)
            and str(item.get("category_id") or "") in SUPPORTED_RUNTIME_CLASSES
        ),
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_report.with_suffix(output_report.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_report)
    print(json.dumps(report, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
