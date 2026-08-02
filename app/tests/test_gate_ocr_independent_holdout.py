from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.tools import gate_ocr_independent_holdout as module
from app.tools.merge_ocr_training_labels import (
    EMPTY_INTERSECTIONS,
    INDEPENDENT_SCAN_HOLDOUT_ROLE,
)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_gate_requires_every_accuracy_and_localization_floor() -> None:
    passed, checks = module.evaluate_gate(
        recognition_metrics={"acc": 0.999, "norm_edit_dis": 0.9998},
        detection_metrics={
            "precision": 0.996,
            "recall": 0.994,
            "hmean": 0.995,
        },
        minimum_accuracy=0.998,
        minimum_normalized_edit=0.9995,
        minimum_precision=0.995,
        minimum_recall=0.995,
        minimum_hmean=0.995,
    )
    assert passed is False
    assert next(
        check for check in checks if check["name"] == "detection_recall"
    )["passed"] is False


def test_runtime_gate_rejects_thresholds_not_selected_on_calibration() -> None:
    model_hash = "a" * 64
    runtime = {
        **module.DETECTION_RUNTIME_FIXED_PARAMETERS,
        "Det.thresh": 0.3,
        "Det.box_thresh": 0.6,
        "Det.unclip_ratio": 1.6,
    }
    calibration = {
        "name": module.DETECTION_CALIBRATION_REPORT_NAME,
        "role": "training_calibration_threshold_selection_only",
        "selection_dataset_role": "calibration_only_not_test",
        "test_split_used_for_selection": False,
        "training_authorized": False,
        "release_authorized": False,
        "dataset_report_role": (
            "training_only_disjoint_from_release_benchmarks"
        ),
        "model_sha256": model_hash,
        "minimum_iou": 0.75,
        "runtime_parameters": {
            **module.DETECTION_RUNTIME_FIXED_PARAMETERS,
        },
        "selected": {
            "threshold": 0.3,
            "box_threshold": 0.7,
            "unclip_ratio": 1.6,
        },
    }
    with pytest.raises(ValueError, match="calibrated thresholds"):
        module._validated_detection_runtime_parameters(
            {"runtime_parameters": runtime},
            calibration,
            model_sha256=model_hash,
            minimum_iou=0.75,
        )


def test_main_gates_full_disjoint_holdout_and_model_hashes(
    tmp_path: Path,
) -> None:
    recognition_model = tmp_path / "rec.onnx"
    detection_model = tmp_path / "det.onnx"
    recognition_labels = tmp_path / "test.paddle.txt"
    recognition_keys = tmp_path / "keys.txt"
    detection_labels = tmp_path / "test.paddle.det.txt"
    recognition_model.write_bytes(b"recognition")
    detection_model.write_bytes(b"detection")
    recognition_labels.write_text("crop.png\tAllegro\n", encoding="utf-8")
    recognition_keys.write_text("A\nl\ne\ng\nr\no\n", encoding="utf-8")
    detection_labels.write_text("page.png\t[]\n", encoding="utf-8")

    dataset_report = tmp_path / "holdout-report.json"
    sources = []
    for index in range(200):
        fingerprint = f"{index:064x}"
        sources.append(
            {
                "source_key": f"muse-omr-work/{fingerprint}",
                "work_fingerprint": fingerprint,
                "split": "test",
                "retained_words": 5,
            }
        )
    _write_json(
        dataset_report,
        {
            "role": INDEPENDENT_SCAN_HOLDOUT_ROLE,
            "forbidden_selection_overlap": [],
            "forbidden_work_overlap": [],
            "split_intersections": EMPTY_INTERSECTIONS,
            "sources": sources,
            "sources_by_split": {
                "train": 0,
                "calibration": 0,
                "test": 200,
            },
            "sources_with_retained_words_by_split": {
                "train": 0,
                "calibration": 0,
                "test": 200,
            },
            "words_by_split": {
                "train": 0,
                "calibration": 0,
                "test": 1000,
            },
        },
    )
    detection_label_report = tmp_path / "detection-label-report.json"
    _write_json(
        detection_label_report,
        {
            "name": module.DETECTION_LABEL_REPORT_NAME,
            "purpose": "independent_ocr_detection_runtime_evaluation_only",
            "role": INDEPENDENT_SCAN_HOLDOUT_ROLE,
            "training_use_authorized": False,
            "forbidden_selection_overlap": [],
            "forbidden_work_overlap": [],
            "source_report_sha256": module.sha256_file(dataset_report),
            "output_counts": {
                "test.paddle.txt": 1000,
                "test.paddle.det.txt": 100,
            },
            "output_sha256": {
                "test.paddle.txt": module.sha256_file(recognition_labels),
                "test.paddle.det.txt": module.sha256_file(detection_labels)
            },
        },
    )
    recognition_report = tmp_path / "recognition-evaluation.json"
    _write_json(
        recognition_report,
        {
            "name": module.RECOGNITION_REPORT_NAME,
            "model_sha256": module.sha256_file(recognition_model),
            "keys_sha256": module.sha256_file(recognition_keys),
            "labels_sha256": module.sha256_file(recognition_labels),
            "metrics": {
                "samples": 1000,
                "correct": 999,
                "acc": 0.999,
                "norm_edit_dis": 0.9998,
            },
        },
    )
    detection_report = tmp_path / "detection-evaluation.json"
    _write_json(
        detection_report,
        {
            "name": module.DETECTION_REPORT_NAME,
            "model_sha256": module.sha256_file(detection_model),
            "labels_sha256": module.sha256_file(detection_labels),
            "minimum_iou": 0.75,
            "runtime_parameters": {
                **module.DETECTION_RUNTIME_FIXED_PARAMETERS,
                "Det.thresh": 0.3,
                "Det.box_thresh": 0.5,
                "Det.unclip_ratio": 1.6,
            },
            "metrics": {
                "pages": 100,
                "ground_truth_boxes": 1000,
                "predicted_boxes": 1000,
                "true_positive_boxes": 997,
                "precision": 0.997,
                "recall": 0.997,
                "hmean": 0.997,
            },
        },
    )
    calibration_report = tmp_path / "detection-calibration.json"
    _write_json(
        calibration_report,
        {
            "name": module.DETECTION_CALIBRATION_REPORT_NAME,
            "role": "training_calibration_threshold_selection_only",
            "selection_dataset_role": "calibration_only_not_test",
            "test_split_used_for_selection": False,
            "training_authorized": False,
            "release_authorized": False,
            "dataset_report_role": (
                "training_only_disjoint_from_release_benchmarks"
            ),
            "model_sha256": module.sha256_file(detection_model),
            "minimum_iou": 0.75,
            "runtime_parameters": {
                **module.DETECTION_RUNTIME_FIXED_PARAMETERS,
            },
            "selected": {
                "threshold": 0.3,
                "box_threshold": 0.5,
                "unclip_ratio": 1.6,
            },
        },
    )
    output = tmp_path / "gate.json"
    assert module.main(
        [
            "--recognition-report",
            str(recognition_report),
            "--detection-report",
            str(detection_report),
            "--detection-calibration-report",
            str(calibration_report),
            "--holdout-dataset-report",
            str(dataset_report),
            "--detection-label-report",
            str(detection_label_report),
            "--recognition-model",
            str(recognition_model),
            "--recognition-keys",
            str(recognition_keys),
            "--detection-model",
            str(detection_model),
            "--recognition-labels",
            str(recognition_labels),
            "--detection-labels",
            str(detection_labels),
            "--output-report",
            str(output),
        ]
    ) == 0
    gate = json.loads(output.read_text(encoding="utf-8"))
    assert gate["passed"] is True
    assert gate["coverage"]["minimum_iou"] == 0.75

    detection_report_value = json.loads(
        detection_report.read_text(encoding="utf-8")
    )
    detection_report_value["metrics"]["ground_truth_boxes"] = 999
    _write_json(detection_report, detection_report_value)
    with pytest.raises(ValueError, match="full holdout"):
        module.main(
            [
                "--recognition-report",
                str(recognition_report),
                "--detection-report",
                str(detection_report),
                "--detection-calibration-report",
                str(calibration_report),
                "--holdout-dataset-report",
                str(dataset_report),
                "--detection-label-report",
                str(detection_label_report),
                "--recognition-model",
                str(recognition_model),
                "--recognition-keys",
                str(recognition_keys),
                "--detection-model",
                str(detection_model),
                "--recognition-labels",
                str(recognition_labels),
                "--detection-labels",
                str(detection_labels),
                "--output-report",
                str(tmp_path / "failed-gate.json"),
            ]
        )


def test_holdout_source_floor_counts_only_sources_with_labels() -> None:
    sources = [
        {
            "source_key": f"muse-omr-work/{index:064x}",
            "work_fingerprint": f"{index:064x}",
            "split": "test",
            "retained_words": 1 if index < 199 else 801,
        }
        for index in range(200)
    ]
    # Make one nominal source contribute no evaluated word.  Counting all
    # registered works would falsely satisfy the 200-source gate.
    sources[198]["retained_words"] = 0
    report = {
        "role": INDEPENDENT_SCAN_HOLDOUT_ROLE,
        "forbidden_selection_overlap": [],
        "forbidden_work_overlap": [],
        "split_intersections": EMPTY_INTERSECTIONS,
        "sources": sources,
        "sources_by_split": {
            "train": 0,
            "calibration": 0,
            "test": 200,
        },
        "sources_with_retained_words_by_split": {
            "train": 0,
            "calibration": 0,
            "test": 199,
        },
        "words_by_split": {
            "train": 0,
            "calibration": 0,
            "test": sum(
                int(source["retained_words"]) for source in sources
            ),
        },
    }
    with pytest.raises(ValueError, match="coverage"):
        module._validate_holdout_dataset(
            report,
            minimum_sources=200,
            minimum_words=1000,
        )
