#!/usr/bin/env python3
"""Generate hash-bound product-layout evidence for semantic detector release QA."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from app.tools.prepare_openscore_svg_regions import (
    COMPLETE_PAGE_TARGET_ASSIGNMENT_VERSION,
)
from scorescan.semantic_detector_contract import (
    SEMANTIC_DETECTOR_MAXIMUM_PAGE_ASPECT_RATIO,
    SEMANTIC_DETECTOR_PAGE_SHAPE_CONTRACT,
    page_aspect_ratio,
)
from scorescan.layout import analyze_layout
from scorescan.semantic_tile_fusion import (
    PAGE_LAYOUT_EVIDENCE_BUILDER_VERSION,
    PAGE_LAYOUT_EVIDENCE_VERSION,
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(payload)
    return rows


def page_key(row: dict[str, Any]) -> tuple[str, str, str]:
    key = (
        str(row.get("source_key") or ""),
        str(row.get("image") or ""),
        str(row.get("image_id") or ""),
    )
    if not all(key):
        raise ValueError("semantic tile has no stable page identity")
    return key


def resolve_image(images_dir: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError("semantic page image path must be relative")
    root = images_dir.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("semantic page image escapes its image root") from exc
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def analyze_page(
    key: tuple[str, str, str],
    image_path: Path,
) -> dict[str, Any]:
    layout = analyze_layout(image_path)
    payload = layout.to_dict()
    systems = payload.get("systems")
    if not isinstance(systems, list) or not systems:
        raise ValueError(f"product layout found no staff: {key!r}")
    if any(
        not isinstance(system, dict)
        or int(system.get("index", 0)) <= 0
        or not isinstance(system.get("line_y"), list)
        or len(system["line_y"]) != 5
        or float(system.get("spacing", 0.0)) <= 0
        for system in systems
    ):
        raise ValueError(f"product layout contains an invalid staff: {key!r}")
    width = int(payload.get("width", 0))
    height = int(payload.get("height", 0))
    aspect_ratio = page_aspect_ratio(width, height)
    if aspect_ratio > SEMANTIC_DETECTOR_MAXIMUM_PAGE_ASPECT_RATIO:
        raise ValueError(
            "product layout page violates "
            f"{SEMANTIC_DETECTOR_PAGE_SHAPE_CONTRACT}: key={key!r}, "
            f"dimensions={width}x{height}, "
            f"aspect_ratio={aspect_ratio:.6f}, "
            "maximum="
            f"{SEMANTIC_DETECTOR_MAXIMUM_PAGE_ASPECT_RATIO:.6f}"
        )
    return {
        "source_key": key[0],
        "image": key[1],
        "image_id": key[2],
        "image_sha256": sha256_file(image_path),
        "layout": payload,
    }


def initialize_layout_worker() -> None:
    # Multiple OpenCV thread pools inside multiple Python workers cause severe
    # oversubscription on full-page scans. One OpenCV thread per process keeps
    # the requested worker count both predictable and memory-bounded.
    import cv2

    cv2.setNumThreads(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "calibration", "test"),
        required=True,
    )
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    if not args.prepared_dir.is_dir():
        raise FileNotFoundError(args.prepared_dir)
    if not args.images_dir.is_dir():
        raise FileNotFoundError(args.images_dir)
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    splits = tuple(dict.fromkeys(args.split))
    manifest_path = args.prepared_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("target_assignment_version")
        != COMPLETE_PAGE_TARGET_ASSIGNMENT_VERSION
    ):
        raise ValueError(
            "semantic layout evidence requires complete-page target assignment"
        )

    split_hashes: dict[str, str] = {}
    pages: dict[tuple[str, str, str], Path] = {}
    tile_counts: dict[str, int] = {}
    for split in splits:
        split_path = args.prepared_dir / f"{split}.jsonl"
        split_rows = load_jsonl(split_path)
        split_hashes[split] = sha256_file(split_path)
        tile_counts[split] = len(split_rows)
        for row in split_rows:
            key = page_key(row)
            image_path = resolve_image(args.images_dir, key[1])
            previous = pages.setdefault(key, image_path)
            if previous != image_path:
                raise ValueError("semantic page identity changed image path")

    failures: list[str] = []
    analyzed: list[dict[str, Any]] = []
    executor_type = (
        concurrent.futures.ThreadPoolExecutor
        if args.workers == 1
        else concurrent.futures.ProcessPoolExecutor
    )
    executor_options: dict[str, Any] = {"max_workers": args.workers}
    if args.workers > 1:
        executor_options["initializer"] = initialize_layout_worker
    with executor_type(**executor_options) as executor:
        future_by_key = {
            executor.submit(analyze_page, key, image_path): key
            for key, image_path in pages.items()
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(future_by_key),
            start=1,
        ):
            key = future_by_key[future]
            try:
                analyzed.append(future.result())
            except Exception as exc:  # release evidence records exact failures
                failures.append(f"{key!r}: {type(exc).__name__}: {exc}")
            if completed % 100 == 0 or completed == len(future_by_key):
                print(
                    json.dumps(
                        {
                            "completed_pages": completed,
                            "total_pages": len(future_by_key),
                            "failures": len(failures),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    analyzed.sort(
        key=lambda item: (
            item["source_key"],
            item["image"],
            item["image_id"],
        )
    )
    output = {
        "format": 1,
        "version": PAGE_LAYOUT_EVIDENCE_VERSION,
        "passed": not failures and len(analyzed) == len(pages),
        "prepared_manifest_sha256": sha256_file(manifest_path),
        "split_jsonl_sha256": split_hashes,
        "target_assignment_version": COMPLETE_PAGE_TARGET_ASSIGNMENT_VERSION,
        "builder_version": PAGE_LAYOUT_EVIDENCE_BUILDER_VERSION,
        "scan_page_shape_contract": (
            SEMANTIC_DETECTOR_PAGE_SHAPE_CONTRACT
        ),
        "maximum_scan_page_aspect_ratio": (
            SEMANTIC_DETECTOR_MAXIMUM_PAGE_ASPECT_RATIO
        ),
        "builder_source_sha256": sha256_file(Path(__file__)),
        "workers": args.workers,
        "executor": "thread" if args.workers == 1 else "process",
        "product_layout_source_sha256": sha256_file(
            Path(__file__).parents[1] / "src" / "scorescan" / "layout.py"
        ),
        "tile_counts": tile_counts,
        "page_count": len(analyzed),
        "staff_count": sum(
            len(item["layout"]["systems"]) for item in analyzed
        ),
        "failures": failures,
        "pages": analyzed,
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps(output | {"pages": []}, sort_keys=True), flush=True)
    if not output["passed"]:
        raise RuntimeError(
            "semantic page-layout evidence failed for "
            f"{len(failures)} of {len(pages)} pages"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
