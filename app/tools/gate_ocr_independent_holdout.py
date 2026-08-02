#!/usr/bin/env python3
"""Gate exported OCR models on a disjoint registered-scan holdout."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from app.tools.merge_ocr_training_labels import (
    EMPTY_INTERSECTIONS,
    INDEPENDENT_SCAN_HOLDOUT_ROLE,
    sha256_file,
)


RECOGNITION_REPORT_NAME = (
    "scorescan-domain-ocr-onnx-runtime-evaluation-v1"
)
DETECTION_REPORT_NAME = (
    "scorescan-rapidocr-onnx-detection-evaluation-v1"
)
DETECTION_LABEL_REPORT_NAME = (
    "scorescan-independent-scan-ocr-detection-labels-v1"
)
DETECTION_CALIBRATION_REPORT_NAME = (
    "scorescan-onnx-detection-postprocess-calibration-v1"
)
DETECTION_RUNTIME_FIXED_PARAMETERS = {
    "Global.max_side_len": 2000,
    "Det.engine_type": "onnxruntime",
    "Det.limit_side_len": 736,
    "Det.limit_type": "min",
    "Det.mean": [0.485, 0.456, 0.406],
    "Det.std": [0.229, 0.224, 0.225],
    "Det.max_candidates": 1000,
    "Det.use_dilation": True,
    "Det.score_mode": "fast",
}
DETECTION_DEPLOY_PARAMETER_NAMES = (
    "Global.max_side_len",
    "Det.limit_side_len",
    "Det.limit_type",
    "Det.mean",
    "Det.std",
    "Det.thresh",
    "Det.box_thresh",
    "Det.max_candidates",
    "Det.unclip_ratio",
    "Det.use_dilation",
    "Det.score_mode",
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON report must be an object: {path}")
    return value


def _finite_metric(
    metrics: dict[str, Any],
    name: str,
) -> float:
    try:
        value = float(metrics[name])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"invalid or missing metric: {name}") from error
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"metric {name} is outside [0, 1]")
    return value


def _validate_holdout_dataset(
    report: dict[str, Any],
    *,
    minimum_sources: int,
    minimum_words: int,
) -> tuple[int, int]:
    if (
        report.get("role") != INDEPENDENT_SCAN_HOLDOUT_ROLE
        or report.get("forbidden_selection_overlap") != []
        or report.get("forbidden_work_overlap") != []
        or report.get("split_intersections") != EMPTY_INTERSECTIONS
    ):
        raise ValueError("OCR holdout failed its disjoint benchmark contract")
    sources = report.get("sources")
    words_by_split = report.get("words_by_split")
    sources_by_split = report.get("sources_by_split")
    retained_sources_by_split = report.get(
        "sources_with_retained_words_by_split"
    )
    if (
        not isinstance(sources, list)
        or not isinstance(words_by_split, dict)
        or not isinstance(sources_by_split, dict)
        or not isinstance(retained_sources_by_split, dict)
    ):
        raise ValueError("OCR holdout source/count audit is missing")
    if any(
        str(source.get("split", "")) != "test"
        for source in sources
        if isinstance(source, dict)
    ) or any(not isinstance(source, dict) for source in sources):
        raise ValueError("OCR holdout contains a non-test source")
    work_fingerprints = [
        str(source.get("work_fingerprint", ""))
        for source in sources
    ]
    if (
        any(
            not re.fullmatch(r"[0-9a-f]{64}", fingerprint)
            for fingerprint in work_fingerprints
        )
        or len(set(work_fingerprints)) != len(work_fingerprints)
        or any(
            str(source.get("source_key", ""))
            != f"muse-omr-work/{fingerprint}"
            for source, fingerprint in zip(
                sources,
                work_fingerprints,
                strict=True,
            )
        )
    ):
        raise ValueError("OCR holdout source rows are not unique works")
    try:
        retained_word_counts = [
            int(source.get("retained_words", -1))
            for source in sources
        ]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "OCR holdout retained-word source audit is malformed"
        ) from error
    if any(count < 0 for count in retained_word_counts):
        raise ValueError(
            "OCR holdout retained-word source audit is malformed"
        )
    source_count = sum(count > 0 for count in retained_word_counts)
    word_count = int(words_by_split.get("test", -1))
    if (
        int(words_by_split.get("train", -1)) != 0
        or int(words_by_split.get("calibration", -1)) != 0
        or int(sources_by_split.get("train", -1)) != 0
        or int(sources_by_split.get("calibration", -1)) != 0
        or int(sources_by_split.get("test", -1)) != len(sources)
        or int(retained_sources_by_split.get("train", -1)) != 0
        or int(retained_sources_by_split.get("calibration", -1)) != 0
        or int(retained_sources_by_split.get("test", -1)) != source_count
        or sum(retained_word_counts) != word_count
        or source_count < minimum_sources
        or word_count < minimum_words
    ):
        raise ValueError("OCR holdout coverage is below the release floor")
    return source_count, word_count


def evaluate_gate(
    *,
    recognition_metrics: dict[str, Any],
    detection_metrics: dict[str, Any],
    minimum_accuracy: float,
    minimum_normalized_edit: float,
    minimum_precision: float,
    minimum_recall: float,
    minimum_hmean: float,
) -> tuple[bool, list[dict[str, Any]]]:
    checks = []
    values_and_floors = (
        ("word_accuracy", _finite_metric(recognition_metrics, "acc"), minimum_accuracy),
        (
            "normalized_edit_distance",
            _finite_metric(recognition_metrics, "norm_edit_dis"),
            minimum_normalized_edit,
        ),
        (
            "detection_precision",
            _finite_metric(detection_metrics, "precision"),
            minimum_precision,
        ),
        (
            "detection_recall",
            _finite_metric(detection_metrics, "recall"),
            minimum_recall,
        ),
        (
            "detection_hmean",
            _finite_metric(detection_metrics, "hmean"),
            minimum_hmean,
        ),
    )
    for name, actual, minimum in values_and_floors:
        check = {
            "name": name,
            "actual": actual,
            "minimum": minimum,
            "passed": actual >= minimum,
        }
        checks.append(check)
    return all(bool(check["passed"]) for check in checks), checks


def _validated_detection_runtime_parameters(
    detection: dict[str, Any],
    calibration: dict[str, Any],
    *,
    model_sha256: str,
    minimum_iou: float,
) -> dict[str, Any]:
    if (
        calibration.get("name") != DETECTION_CALIBRATION_REPORT_NAME
        or calibration.get("role")
        != "training_calibration_threshold_selection_only"
        or calibration.get("selection_dataset_role")
        != "calibration_only_not_test"
        or calibration.get("test_split_used_for_selection") is not False
        or calibration.get("training_authorized") is not False
        or calibration.get("release_authorized") is not False
        or calibration.get("dataset_report_role")
        != "training_only_disjoint_from_release_benchmarks"
        or calibration.get("model_sha256") != model_sha256
        or float(calibration.get("minimum_iou", -1)) < minimum_iou
    ):
        raise ValueError("detection calibration provenance is invalid")
    selected = calibration.get("selected")
    calibration_runtime = calibration.get("runtime_parameters")
    detection_runtime = detection.get("runtime_parameters")
    if (
        not isinstance(selected, dict)
        or not isinstance(calibration_runtime, dict)
        or not isinstance(detection_runtime, dict)
        or any(
            calibration_runtime.get(name) != expected
            for name, expected in DETECTION_RUNTIME_FIXED_PARAMETERS.items()
        )
        or any(
            detection_runtime.get(name) != expected
            for name, expected in DETECTION_RUNTIME_FIXED_PARAMETERS.items()
        )
    ):
        raise ValueError("detection runtime profile was changed")
    try:
        threshold = float(selected["threshold"])
        box_threshold = float(selected["box_threshold"])
        unclip_ratio = float(selected["unclip_ratio"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("calibrated detection thresholds are invalid") from error
    if (
        not 0 < threshold < 1
        or not 0 < box_threshold < 1
        or not 0.5 <= unclip_ratio <= 3.0
        or detection_runtime.get("Det.thresh") != threshold
        or detection_runtime.get("Det.box_thresh") != box_threshold
        or detection_runtime.get("Det.unclip_ratio") != unclip_ratio
    ):
        raise ValueError("independent evaluation did not use calibrated thresholds")
    return {
        name: detection_runtime[name]
        for name in DETECTION_DEPLOY_PARAMETER_NAMES
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recognition-report", type=Path, required=True)
    parser.add_argument("--detection-report", type=Path, required=True)
    parser.add_argument(
        "--detection-calibration-report",
        type=Path,
        required=True,
    )
    parser.add_argument("--holdout-dataset-report", type=Path, required=True)
    parser.add_argument(
        "--detection-label-report",
        type=Path,
        required=True,
    )
    parser.add_argument("--recognition-model", type=Path, required=True)
    parser.add_argument("--recognition-keys", type=Path, required=True)
    parser.add_argument("--detection-model", type=Path, required=True)
    parser.add_argument("--recognition-labels", type=Path, required=True)
    parser.add_argument("--detection-labels", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--minimum-sources", type=int, default=200)
    parser.add_argument("--minimum-words", type=int, default=1000)
    parser.add_argument("--minimum-pages", type=int, default=100)
    parser.add_argument("--minimum-iou", type=float, default=0.75)
    parser.add_argument("--minimum-accuracy", type=float, default=0.998)
    parser.add_argument(
        "--minimum-normalized-edit",
        type=float,
        default=0.9995,
    )
    parser.add_argument("--minimum-precision", type=float, default=0.995)
    parser.add_argument("--minimum-recall", type=float, default=0.995)
    parser.add_argument("--minimum-hmean", type=float, default=0.995)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = (
        args.recognition_model,
        args.recognition_keys,
        args.detection_model,
        args.recognition_labels,
        args.detection_labels,
    )
    if any(not path.is_file() or path.stat().st_size <= 0 for path in paths):
        raise FileNotFoundError("a required model or label file is missing")
    floors = (
        args.minimum_iou,
        args.minimum_accuracy,
        args.minimum_normalized_edit,
        args.minimum_precision,
        args.minimum_recall,
        args.minimum_hmean,
    )
    if (
        args.minimum_sources <= 0
        or args.minimum_words <= 0
        or args.minimum_pages <= 0
        or any(not 0 <= value <= 1 for value in floors)
    ):
        raise ValueError("holdout coverage and metric floors are invalid")

    dataset = _load_json(args.holdout_dataset_report)
    source_count, word_count = _validate_holdout_dataset(
        dataset,
        minimum_sources=args.minimum_sources,
        minimum_words=args.minimum_words,
    )
    detection_labels = _load_json(args.detection_label_report)
    if (
        detection_labels.get("name") != DETECTION_LABEL_REPORT_NAME
        or detection_labels.get("purpose")
        != "independent_ocr_detection_runtime_evaluation_only"
        or detection_labels.get("role") != INDEPENDENT_SCAN_HOLDOUT_ROLE
        or detection_labels.get("training_use_authorized") is not False
        or detection_labels.get("forbidden_selection_overlap") != []
        or detection_labels.get("forbidden_work_overlap") != []
        or detection_labels.get("source_report_sha256")
        != sha256_file(args.holdout_dataset_report)
        or detection_labels.get("output_sha256", {}).get("test.paddle.txt")
        != sha256_file(args.recognition_labels)
        or detection_labels.get("output_sha256", {}).get(
            "test.paddle.det.txt"
        )
        != sha256_file(args.detection_labels)
    ):
        raise ValueError("independent detection label provenance is invalid")
    page_count = int(
        detection_labels.get("output_counts", {}).get(
            "test.paddle.det.txt",
            -1,
        )
    )
    label_word_count = int(
        detection_labels.get("output_counts", {}).get(
            "test.paddle.txt",
            -1,
        )
    )
    if label_word_count != word_count:
        raise ValueError("independent recognition label coverage is incomplete")
    if page_count < args.minimum_pages:
        raise ValueError("independent detection page coverage is too small")

    recognition = _load_json(args.recognition_report)
    detection = _load_json(args.detection_report)
    detection_calibration = _load_json(
        args.detection_calibration_report
    )
    if (
        recognition.get("name") != RECOGNITION_REPORT_NAME
        or recognition.get("model_sha256")
        != sha256_file(args.recognition_model)
        or recognition.get("keys_sha256")
        != sha256_file(args.recognition_keys)
        or recognition.get("labels_sha256")
        != sha256_file(args.recognition_labels)
    ):
        raise ValueError("independent recognition evaluation provenance is invalid")
    if (
        detection.get("name") != DETECTION_REPORT_NAME
        or detection.get("model_sha256") != sha256_file(args.detection_model)
        or detection.get("labels_sha256")
        != sha256_file(args.detection_labels)
        or float(detection.get("minimum_iou", -1)) < args.minimum_iou
    ):
        raise ValueError("independent detection evaluation provenance is invalid")
    detection_runtime_parameters = (
        _validated_detection_runtime_parameters(
            detection,
            detection_calibration,
            model_sha256=sha256_file(args.detection_model),
            minimum_iou=args.minimum_iou,
        )
    )

    recognition_metrics = recognition.get("metrics")
    detection_metrics = detection.get("metrics")
    if not isinstance(recognition_metrics, dict) or not isinstance(
        detection_metrics,
        dict,
    ):
        raise ValueError("independent OCR evaluation metrics are missing")
    if int(recognition_metrics.get("samples", -1)) != word_count:
        raise ValueError("recognition evaluation did not cover every holdout word")
    if (
        int(detection_metrics.get("pages", -1)) != page_count
        or int(detection_metrics.get("ground_truth_boxes", -1)) != word_count
    ):
        raise ValueError("detection evaluation did not cover the full holdout")

    passed, checks = evaluate_gate(
        recognition_metrics=recognition_metrics,
        detection_metrics=detection_metrics,
        minimum_accuracy=args.minimum_accuracy,
        minimum_normalized_edit=args.minimum_normalized_edit,
        minimum_precision=args.minimum_precision,
        minimum_recall=args.minimum_recall,
        minimum_hmean=args.minimum_hmean,
    )
    report = {
        "schema_version": 1,
        "name": "scorescan-independent-scan-ocr-release-gate-v1",
        "passed": passed,
        "integration_authorized": passed,
        "role": INDEPENDENT_SCAN_HOLDOUT_ROLE,
        "coverage": {
            "sources": source_count,
            "words": word_count,
            "pages": page_count,
            "minimum_iou": float(detection["minimum_iou"]),
        },
        "recognition_model_sha256": sha256_file(args.recognition_model),
        "recognition_keys_sha256": sha256_file(args.recognition_keys),
        "detection_model_sha256": sha256_file(args.detection_model),
        "detection_runtime_parameters": detection_runtime_parameters,
        "evaluations": {
            "independent_registered_scan_holdout": {
                "recognition": recognition_metrics,
                "detection": detection_metrics,
            }
        },
        "artifacts": {
            "holdout_dataset_report_sha256": sha256_file(
                args.holdout_dataset_report
            ),
            "detection_label_report_sha256": sha256_file(
                args.detection_label_report
            ),
            "recognition_report_sha256": sha256_file(
                args.recognition_report
            ),
            "detection_report_sha256": sha256_file(args.detection_report),
            "detection_calibration_report_sha256": sha256_file(
                args.detection_calibration_report
            ),
            "recognition_labels_sha256": sha256_file(
                args.recognition_labels
            ),
            "detection_labels_sha256": sha256_file(args.detection_labels),
        },
        "checks": checks,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_report.with_suffix(
        args.output_report.suffix + ".tmp"
    )
    temporary.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output_report)
    print(json.dumps({"passed": passed, "checks": checks}, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
