from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.tools.calibrate_onnx_ocr_detection_thresholds import (
    DEFAULT_MINIMUM_IOU,
    _grid,
    select_operating_point,
    validate_calibration_dataset,
)


def test_calibration_uses_release_localization_iou() -> None:
    assert DEFAULT_MINIMUM_IOU == 0.75


def test_grid_is_bounded_and_deduplicated() -> None:
    assert _grid([0.3, 0.3], [0.5], [1.6]) == [
        (0.3, 0.5, 1.6)
    ]
    with pytest.raises(ValueError, match="invalid or oversized"):
        _grid([0.0], [0.5], [1.6])


def test_selection_optimizes_worse_of_precision_and_recall() -> None:
    rows = [
        {
            "threshold": 0.3,
            "box_threshold": 0.5,
            "unclip_ratio": 1.6,
            "metrics": {
                "minimum_precision_recall": 0.50,
                "hmean": 0.66,
                "precision": 0.50,
            },
        },
        {
            "threshold": 0.3,
            "box_threshold": 0.7,
            "unclip_ratio": 1.6,
            "metrics": {
                "minimum_precision_recall": 0.60,
                "hmean": 0.61,
                "precision": 0.62,
            },
        },
    ]
    assert select_operating_point(rows)["box_threshold"] == 0.7


def _dataset_report(project_root: Path, page_count: int) -> dict[str, object]:
    return {
        "name": "scorescan-ppocrv6-domain-detection-labels-v1",
        "role": "training_only_disjoint_from_release_benchmarks",
        "project_root": str(project_root),
        "label_coverage_contract": {
            "precision_evaluation_authorized": True,
            "hmean_evaluation_authorized": True,
            "postprocess_threshold_selection_authorized": True,
            "unlabelled_visible_text_may_be_present": False,
        },
        "output_counts": {
            "calibration.scan.paddle.det.txt": page_count,
        },
        "output_label_coverage": {
            "calibration.scan.paddle.det.txt": {
                "precision_evaluation_authorized": True,
                "hmean_evaluation_authorized": True,
                "postprocess_threshold_selection_authorized": True,
                "unlabelled_visible_text_may_be_present": False,
            }
        },
        "global_split_intersections": {
            "train_calibration": [],
            "train_test": [],
            "calibration_test": [],
        },
        "physical_image_aliases_across_splits": [],
        "duplicate_image_content_across_source_assignments": [],
    }


def test_calibration_dataset_validation_fails_closed_on_test_split(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "test.scan.paddle.det.txt"
    labels.write_text("placeholder\n", encoding="utf-8")
    report_path = tmp_path / "merge-report.json"
    report_path.write_text(
        json.dumps(_dataset_report(tmp_path, 1)),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="explicit calibration"):
        validate_calibration_dataset(
            labels,
            report_path,
            tmp_path,
            page_count=1,
        )


def test_calibration_dataset_validation_requires_isolation(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "calibration.scan.paddle.det.txt"
    labels.write_text("placeholder\n", encoding="utf-8")
    report = _dataset_report(tmp_path, 1)
    report["global_split_intersections"]["train_calibration"] = ["work"]
    report_path = tmp_path / "merge-report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="split isolation"):
        validate_calibration_dataset(
            labels,
            report_path,
            tmp_path,
            page_count=1,
        )


def test_calibration_rejects_sparse_positive_only_labels(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "calibration.scan.paddle.det.txt"
    labels.write_text("placeholder\n", encoding="utf-8")
    report = _dataset_report(tmp_path, 1)
    report["label_coverage_contract"] = {
        "precision_evaluation_authorized": False,
        "hmean_evaluation_authorized": False,
        "postprocess_threshold_selection_authorized": False,
        "unlabelled_visible_text_may_be_present": True,
    }
    report["output_label_coverage"][
        "calibration.scan.paddle.det.txt"
    ] = {
        "precision_evaluation_authorized": False,
        "hmean_evaluation_authorized": False,
        "postprocess_threshold_selection_authorized": False,
        "unlabelled_visible_text_may_be_present": True,
    }
    report_path = tmp_path / "merge-report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="exhaustive visible-text"):
        validate_calibration_dataset(
            labels,
            report_path,
            tmp_path,
            page_count=1,
        )
