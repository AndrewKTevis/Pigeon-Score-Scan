#!/usr/bin/env python3
"""Evaluate an exported text detector through ScoreScan's RapidOCR runtime.

This is deliberately separate from Paddle's in-framework metric.  It verifies
the exported ONNX graph, Windows ONNX Runtime preprocessing/postprocessing, and
page-coordinate mapping against the independent clean or registered-scan
Paddle detection labels.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from shapely.geometry import Polygon

from app.tools.merge_ocr_training_labels import sha256_file


RELEASE_MINIMUM_IOU = 0.75


def serializable_runtime_parameters(
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Preserve the audited runtime profile without leaking Enum objects."""
    return {
        name: value.value if isinstance(value, Enum) else value
        for name, value in parameters.items()
    }


def _polygon(points: Iterable[Iterable[float]]) -> Polygon:
    parsed = []
    for point in points:
        values = list(point)
        if len(values) != 2:
            raise ValueError("polygon point must have x and y")
        x, y = (float(value) for value in values)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("polygon coordinates must be finite")
        parsed.append((x, y))
    if len(parsed) < 4:
        raise ValueError("text polygon must have at least four points")
    polygon = Polygon(parsed)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.area <= 0:
        raise ValueError("text polygon must have positive area")
    return polygon


def polygon_iou(left: Polygon, right: Polygon) -> float:
    intersection = left.intersection(right).area
    if intersection <= 0:
        return 0.0
    union = left.area + right.area - intersection
    return float(intersection / union) if union > 0 else 0.0


def polygon_points(polygon: Polygon) -> list[list[float]]:
    """Return compact, deterministic coordinates for error diagnostics."""
    return [
        [round(float(x), 3), round(float(y), 3)]
        for x, y in list(polygon.exterior.coords)[:-1]
    ]


def maximum_matches(
    ground_truth: list[Polygon],
    predicted: list[Polygon],
    *,
    minimum_iou: float = 0.5,
) -> tuple[int, list[tuple[int, int, float]]]:
    if not 0 < minimum_iou <= 1:
        raise ValueError("minimum IoU must be in (0, 1]")
    adjacency: list[list[tuple[int, float]]] = []
    for truth in ground_truth:
        candidates = [
            (prediction_index, polygon_iou(truth, prediction))
            for prediction_index, prediction in enumerate(predicted)
        ]
        adjacency.append(
            sorted(
                (
                    item for item in candidates
                    if item[1] >= minimum_iou
                ),
                key=lambda item: (-item[1], item[0]),
            )
        )

    predicted_to_truth: dict[int, int] = {}

    def augment(truth_index: int, visited: set[int]) -> bool:
        for prediction_index, _iou in adjacency[truth_index]:
            if prediction_index in visited:
                continue
            visited.add(prediction_index)
            previous = predicted_to_truth.get(prediction_index)
            if previous is None or augment(previous, visited):
                predicted_to_truth[prediction_index] = truth_index
                return True
        return False

    for truth_index in range(len(ground_truth)):
        augment(truth_index, set())
    matches = [
        (
            truth_index,
            prediction_index,
            polygon_iou(
                ground_truth[truth_index],
                predicted[prediction_index],
            ),
        )
        for prediction_index, truth_index in predicted_to_truth.items()
    ]
    matches.sort()
    return len(matches), matches


def aggregate_metrics(
    pages: Iterable[tuple[list[Polygon], list[Polygon]]],
    *,
    minimum_iou: float = 0.5,
) -> dict[str, float | int]:
    truth_count = 0
    prediction_count = 0
    true_positive_count = 0
    page_count = 0
    for ground_truth, predicted in pages:
        page_count += 1
        truth_count += len(ground_truth)
        prediction_count += len(predicted)
        matches, _pairs = maximum_matches(
            ground_truth,
            predicted,
            minimum_iou=minimum_iou,
        )
        true_positive_count += matches
    precision = (
        true_positive_count / prediction_count
        if prediction_count
        else 0.0
    )
    recall = (
        true_positive_count / truth_count
        if truth_count
        else 0.0
    )
    hmean = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "pages": page_count,
        "ground_truth_boxes": truth_count,
        "predicted_boxes": prediction_count,
        "true_positive_boxes": true_positive_count,
        "precision": precision,
        "recall": recall,
        "hmean": hmean,
    }


def load_labels(
    path: Path,
    *,
    project_root: Path,
) -> list[tuple[Path, list[Polygon]]]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    project_root = project_root.resolve()
    rows = []
    seen_images: set[Path] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            image_value, separator, annotations_value = line.rstrip(
                "\r\n"
            ).partition("\t")
            if not separator:
                raise ValueError(f"{path}:{line_number}: missing tab separator")
            image_path = (project_root / Path(image_value)).resolve()
            try:
                image_path.relative_to(project_root)
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: image escapes project root"
                ) from error
            if not image_path.is_file() or image_path.stat().st_size <= 0:
                raise FileNotFoundError(image_path)
            if image_path in seen_images:
                raise ValueError(f"duplicate evaluation page: {image_path}")
            seen_images.add(image_path)
            raw_annotations = json.loads(annotations_value)
            if not isinstance(raw_annotations, list) or not raw_annotations:
                raise ValueError(
                    f"{path}:{line_number}: annotations must be nonempty"
                )
            polygons = []
            for annotation in raw_annotations:
                if not isinstance(annotation, dict):
                    raise ValueError(f"{path}:{line_number}: malformed annotation")
                polygons.append(_polygon(annotation.get("points", [])))
            rows.append((image_path, polygons))
    if not rows:
        raise ValueError("detection evaluation label file is empty")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument(
        "--minimum-iou",
        type=float,
        default=RELEASE_MINIMUM_IOU,
    )
    parser.add_argument("--maximum-error-pages", type=int, default=200)
    parser.add_argument("--limit-side-len", type=int, default=736)
    parser.add_argument("--maximum-side-len", type=int, default=2000)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--box-threshold", type=float, default=0.5)
    parser.add_argument("--unclip-ratio", type=float, default=1.6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.model.is_file() or args.model.stat().st_size <= 0:
        raise FileNotFoundError(args.model)
    if args.maximum_error_pages < 0:
        raise ValueError("maximum error pages must be non-negative")
    if args.limit_side_len <= 0 or args.maximum_side_len <= 0:
        raise ValueError("image side limits must be positive")
    if not 0 < args.minimum_iou <= 1:
        raise ValueError("minimum IoU must be in (0, 1]")
    rows = load_labels(
        args.labels,
        project_root=args.project_root,
    )

    from rapidocr import RapidOCR
    from rapidocr.utils.typings import EngineType

    params = {
        "Global.use_det": True,
        "Global.use_cls": False,
        "Global.use_rec": False,
        "Global.max_side_len": args.maximum_side_len,
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Det.model_path": str(args.model.resolve()),
        "Det.limit_side_len": args.limit_side_len,
        "Det.limit_type": "min",
        "Det.mean": [0.485, 0.456, 0.406],
        "Det.std": [0.229, 0.224, 0.225],
        "Det.thresh": args.threshold,
        "Det.box_thresh": args.box_threshold,
        "Det.max_candidates": 1000,
        "Det.unclip_ratio": args.unclip_ratio,
        "Det.use_dilation": True,
        "Det.score_mode": "fast",
    }
    engine = RapidOCR(params=params)
    evaluated_pages = []
    errors = []
    elapsed_seconds = 0.0
    for page_index, (image_path, ground_truth) in enumerate(rows, start=1):
        started = time.perf_counter()
        result = engine(
            image_path,
            use_det=True,
            use_cls=False,
            use_rec=False,
        )
        elapsed = time.perf_counter() - started
        elapsed_seconds += elapsed
        if page_index == 1 or page_index % 5 == 0 or page_index == len(rows):
            print(
                f"evaluated {page_index}/{len(rows)} pages "
                f"({elapsed:.3f}s current, {elapsed_seconds:.3f}s total)",
                flush=True,
            )
        raw_boxes = getattr(result, "boxes", None)
        predicted = [
            _polygon(box)
            for box in ([] if raw_boxes is None else raw_boxes)
        ]
        matched, matches = maximum_matches(
            ground_truth,
            predicted,
            minimum_iou=args.minimum_iou,
        )
        evaluated_pages.append((ground_truth, predicted))
        if matched != len(ground_truth) or matched != len(predicted):
            matched_truth = {pair[0] for pair in matches}
            matched_predictions = {pair[1] for pair in matches}
            unmatched_truth = [
                (index, polygon)
                for index, polygon in enumerate(ground_truth)
                if index not in matched_truth
            ]
            unmatched_predictions = [
                (index, polygon)
                for index, polygon in enumerate(predicted)
                if index not in matched_predictions
            ]
            errors.append(
                {
                    "image": image_path.relative_to(
                        args.project_root.resolve()
                    ).as_posix(),
                    "ground_truth_boxes": len(ground_truth),
                    "predicted_boxes": len(predicted),
                    "matched_boxes": matched,
                    "matched_iou": [
                        round(pair[2], 6) for pair in matches
                    ],
                    "precision": (
                        matched / len(predicted) if predicted else 0.0
                    ),
                    "recall": (
                        matched / len(ground_truth)
                        if ground_truth
                        else 0.0
                    ),
                    "unmatched_ground_truth": [
                        {
                            "index": index,
                            "points": polygon_points(polygon),
                            "best_prediction_iou": round(
                                max(
                                    (
                                        polygon_iou(polygon, candidate)
                                        for candidate in predicted
                                    ),
                                    default=0.0,
                                ),
                                6,
                            ),
                        }
                        for index, polygon in unmatched_truth
                    ],
                    "unmatched_predictions": [
                        {
                            "index": index,
                            "points": polygon_points(polygon),
                            "best_ground_truth_iou": round(
                                max(
                                    (
                                        polygon_iou(candidate, polygon)
                                        for candidate in ground_truth
                                    ),
                                    default=0.0,
                                ),
                                6,
                            ),
                        }
                        for index, polygon in unmatched_predictions
                    ],
                }
            )

    metrics = aggregate_metrics(
        evaluated_pages,
        minimum_iou=args.minimum_iou,
    )
    report = {
        "schema_version": 1,
        "name": "scorescan-rapidocr-onnx-detection-evaluation-v1",
        "model": str(args.model.resolve()),
        "model_sha256": sha256_file(args.model),
        "labels": str(args.labels.resolve()),
        "labels_sha256": sha256_file(args.labels),
        "minimum_iou": args.minimum_iou,
        "runtime_parameters": serializable_runtime_parameters(params),
        "metrics": metrics,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "mean_seconds_per_page": round(
            elapsed_seconds / max(1, len(rows)),
            6,
        ),
        "error_page_count": len(errors),
        "error_pages_truncated": len(errors) > args.maximum_error_pages,
        "error_pages": errors[: args.maximum_error_pages],
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_report.with_suffix(
        args.output_report.suffix + ".tmp"
    )
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output_report)
    print(
        "precision:{precision:.12f} recall:{recall:.12f} "
        "hmean:{hmean:.12f}".format(**metrics)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
