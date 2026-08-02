from __future__ import annotations

"""Calibrate DB detector postprocessing with one inference pass per page."""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Iterable

from app.tools.evaluate_onnx_ocr_detection import (
    RELEASE_MINIMUM_IOU,
    _polygon,
    load_labels,
    maximum_matches,
    serializable_runtime_parameters,
)
from app.tools.merge_ocr_training_labels import sha256_file


DEFAULT_THRESHOLDS = (0.20, 0.25, 0.30, 0.35, 0.40)
DEFAULT_BOX_THRESHOLDS = (
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
)
DEFAULT_UNCLIP_RATIOS = (1.2, 1.4, 1.6, 1.8)
DEFAULT_MINIMUM_IOU = RELEASE_MINIMUM_IOU


def validate_calibration_dataset(
    labels_path: Path,
    dataset_report_path: Path,
    project_root: Path,
    *,
    page_count: int,
) -> dict[str, object]:
    """Fail closed if threshold selection is pointed at a non-calibration split."""
    labels_path = labels_path.resolve()
    dataset_report_path = dataset_report_path.resolve()
    project_root = project_root.resolve()
    if (
        not labels_path.name.startswith("calibration.")
        or "test" in labels_path.name.casefold()
    ):
        raise ValueError(
            "threshold selection requires an explicit calibration.* label file"
        )
    if dataset_report_path.parent != labels_path.parent:
        raise ValueError(
            "dataset report and calibration labels must share a directory"
        )
    if not dataset_report_path.is_file():
        raise FileNotFoundError(dataset_report_path)
    report = json.loads(dataset_report_path.read_text(encoding="utf-8"))
    if (
        report.get("name")
        != "scorescan-ppocrv6-domain-detection-labels-v1"
        or report.get("role")
        != "training_only_disjoint_from_release_benchmarks"
    ):
        raise ValueError("untrusted detection dataset report")
    output_coverage = report.get("output_label_coverage")
    coverage = (
        output_coverage.get(labels_path.name)
        if isinstance(output_coverage, dict)
        else None
    )
    if (
        not isinstance(coverage, dict)
        or coverage.get("precision_evaluation_authorized") is not True
        or coverage.get("hmean_evaluation_authorized") is not True
        or coverage.get("postprocess_threshold_selection_authorized")
        is not True
        or coverage.get("unlabelled_visible_text_may_be_present")
        is not False
    ):
        raise ValueError(
            "threshold selection requires exhaustive visible-text labels"
        )
    if Path(str(report.get("project_root", ""))).resolve() != project_root:
        raise ValueError("dataset report project root does not match")
    output_counts = report.get("output_counts")
    if (
        not isinstance(output_counts, dict)
        or output_counts.get(labels_path.name) != page_count
    ):
        raise ValueError("calibration labels are absent from dataset report")
    intersections = report.get("global_split_intersections")
    if (
        not isinstance(intersections, dict)
        or any(value != [] for value in intersections.values())
        or report.get("physical_image_aliases_across_splits") != []
        or report.get("duplicate_image_content_across_source_assignments")
        != []
    ):
        raise ValueError("detection dataset split isolation did not pass")
    return report


def _grid(
    thresholds: Iterable[float],
    box_thresholds: Iterable[float],
    unclip_ratios: Iterable[float],
) -> list[tuple[float, float, float]]:
    rows = sorted(
        {
            (
                round(float(threshold), 8),
                round(float(box_threshold), 8),
                round(float(unclip_ratio), 8),
            )
            for threshold in thresholds
            for box_threshold in box_thresholds
            for unclip_ratio in unclip_ratios
        }
    )
    if (
        not rows
        or len(rows) > 500
        or any(
            not 0 < threshold < 1
            or not 0 < box_threshold < 1
            or not 0.5 <= unclip_ratio <= 3.0
            for threshold, box_threshold, unclip_ratio in rows
        )
    ):
        raise ValueError("invalid or oversized detection threshold grid")
    return rows


def _metrics(
    truth_count: int,
    prediction_count: int,
    true_positive_count: int,
    page_count: int,
) -> dict[str, float | int]:
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
        "minimum_precision_recall": min(precision, recall),
    }


def select_operating_point(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    if not rows:
        raise ValueError("threshold calibration has no rows")
    # A release needs both precision and recall, so optimize their lower value
    # first. Hmean and precision resolve ties without using any test split.
    return max(
        rows,
        key=lambda row: (
            float(row["metrics"]["minimum_precision_recall"]),  # type: ignore[index]
            float(row["metrics"]["hmean"]),  # type: ignore[index]
            float(row["metrics"]["precision"]),  # type: ignore[index]
            -abs(float(row["threshold"]) - 0.3),
            -abs(float(row["box_threshold"]) - 0.5),
            -abs(float(row["unclip_ratio"]) - 1.6),
        ),
    )


def calibrate(
    model_path: Path,
    labels_path: Path,
    dataset_report_path: Path,
    project_root: Path,
    output_path: Path,
    *,
    thresholds: Iterable[float],
    box_thresholds: Iterable[float],
    unclip_ratios: Iterable[float],
    minimum_iou: float,
    limit_side_len: int,
    maximum_side_len: int,
) -> dict[str, object]:
    if not model_path.is_file() or model_path.stat().st_size <= 0:
        raise FileNotFoundError(model_path)
    if not 0 < minimum_iou <= 1:
        raise ValueError("minimum IoU must be in (0, 1]")
    grid = _grid(thresholds, box_thresholds, unclip_ratios)
    labelled = load_labels(labels_path, project_root=project_root)
    dataset_report = validate_calibration_dataset(
        labels_path,
        dataset_report_path,
        project_root,
        page_count=len(labelled),
    )

    from rapidocr import RapidOCR
    from rapidocr.ch_ppocr_det.utils import DBPostProcess
    from rapidocr.utils.process_img import map_boxes_to_original
    from rapidocr.utils.typings import EngineType

    params = {
        "Global.use_det": True,
        "Global.use_cls": False,
        "Global.use_rec": False,
        "Global.max_side_len": maximum_side_len,
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Det.model_path": str(model_path.resolve()),
        "Det.limit_side_len": limit_side_len,
        "Det.limit_type": "min",
        "Det.mean": [0.485, 0.456, 0.406],
        "Det.std": [0.229, 0.224, 0.225],
        "Det.max_candidates": 1000,
        "Det.use_dilation": True,
        "Det.score_mode": "fast",
    }
    engine = RapidOCR(params=params)
    postprocessors = {
        point: DBPostProcess(
            thresh=point[0],
            box_thresh=point[1],
            max_candidates=1000,
            unclip_ratio=point[2],
            use_dilation=True,
            score_mode="fast",
        )
        for point in grid
    }
    counts = {
        point: {
            "truth": 0,
            "predictions": 0,
            "true_positives": 0,
            "pages": 0,
        }
        for point in grid
    }
    inference_seconds = 0.0
    postprocess_seconds = 0.0
    for position, (image_path, ground_truth) in enumerate(
        labelled,
        start=1,
    ):
        original = engine.load_img(image_path)
        image, operation = engine.preprocess_img(original)
        text_detector = engine.text_det
        detector_preprocess = text_detector.get_preprocess(
            max(image.shape[0], image.shape[1])
        )
        model_input = detector_preprocess(image)
        if model_input is None:
            raise RuntimeError(f"detector preprocessing failed: {image_path}")
        started = time.perf_counter()
        predictions = text_detector.session(model_input)
        inference_seconds += time.perf_counter() - started
        inner_shape = image.shape[0], image.shape[1]
        original_h, original_w = original.shape[:2]
        started = time.perf_counter()
        for point, postprocessor in postprocessors.items():
            boxes, _scores = postprocessor(predictions, inner_shape)
            if len(boxes):
                boxes = text_detector.sorted_boxes(boxes)
                boxes = map_boxes_to_original(
                    boxes,
                    operation,
                    original_h,
                    original_w,
                )
            predicted_polygons = [
                # Keep exactly the release evaluator's polygon validation and
                # repair semantics.
                _polygon(box)
                for box in boxes
            ]
            matches, _pairs = maximum_matches(
                ground_truth,
                predicted_polygons,
                minimum_iou=minimum_iou,
            )
            current = counts[point]
            current["truth"] += len(ground_truth)
            current["predictions"] += len(predicted_polygons)
            current["true_positives"] += matches
            current["pages"] += 1
        postprocess_seconds += time.perf_counter() - started
        if position % 5 == 0 or position == len(labelled):
            print(
                f"[{position}/{len(labelled)}] calibration pages inferred",
                flush=True,
            )
    rows = []
    for point in grid:
        count = counts[point]
        rows.append(
            {
                "threshold": point[0],
                "box_threshold": point[1],
                "unclip_ratio": point[2],
                "metrics": _metrics(
                    count["truth"],
                    count["predictions"],
                    count["true_positives"],
                    count["pages"],
                ),
            }
        )
    selected = select_operating_point(rows)
    report = {
        "schema_version": 1,
        "name": "scorescan-onnx-detection-postprocess-calibration-v1",
        "role": "training_calibration_threshold_selection_only",
        "selection_dataset_role": "calibration_only_not_test",
        "test_split_used_for_selection": False,
        "model": str(model_path.resolve()),
        "model_sha256": sha256_file(model_path),
        "labels": str(labels_path.resolve()),
        "labels_sha256": sha256_file(labels_path),
        "dataset_report": str(dataset_report_path.resolve()),
        "dataset_report_sha256": sha256_file(dataset_report_path),
        "dataset_report_role": dataset_report["role"],
        "minimum_iou": minimum_iou,
        "runtime_parameters": serializable_runtime_parameters(params),
        "grid_size": len(grid),
        "inference_passes_per_page": 1,
        "inference_seconds": round(inference_seconds, 6),
        "postprocess_seconds": round(postprocess_seconds, 6),
        "selected": selected,
        "rows": rows,
        "training_authorized": False,
        "release_authorized": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    return report


def _floats(values: list[str] | None, defaults: tuple[float, ...]):
    return defaults if not values else tuple(float(value) for value in values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--dataset-report", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--threshold", action="append")
    parser.add_argument("--box-threshold", action="append")
    parser.add_argument("--unclip-ratio", action="append")
    parser.add_argument(
        "--minimum-iou",
        type=float,
        default=DEFAULT_MINIMUM_IOU,
    )
    parser.add_argument("--limit-side-len", type=int, default=736)
    parser.add_argument("--maximum-side-len", type=int, default=2000)
    args = parser.parse_args()
    report = calibrate(
        args.model.resolve(),
        args.labels.resolve(),
        args.dataset_report.resolve(),
        args.project_root.resolve(),
        args.output_report.resolve(),
        thresholds=_floats(args.threshold, DEFAULT_THRESHOLDS),
        box_thresholds=_floats(
            args.box_threshold,
            DEFAULT_BOX_THRESHOLDS,
        ),
        unclip_ratios=_floats(
            args.unclip_ratio,
            DEFAULT_UNCLIP_RATIOS,
        ),
        minimum_iou=args.minimum_iou,
        limit_side_len=args.limit_side_len,
        maximum_side_len=args.maximum_side_len,
    )
    print(json.dumps(report["selected"], indent=2))


if __name__ == "__main__":
    main()
