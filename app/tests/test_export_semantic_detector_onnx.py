from __future__ import annotations

from app.tools.export_semantic_detector_onnx import compare_detector_outputs


def test_onnx_parity_comparison_accepts_small_numeric_drift() -> None:
    reference = (
        [[1.0, 2.0, 10.0, 12.0]],
        [0.999],
        [3],
    )
    candidate = (
        [[1.02, 1.98, 10.01, 12.01]],
        [0.99899],
        [3],
    )

    report = compare_detector_outputs(
        reference,
        candidate,
        minimum_detections=1,
        maximum_box_error=0.05,
        maximum_score_error=1e-4,
    )

    assert report["passed"]
    assert not report["failures"]


def test_onnx_parity_comparison_rejects_label_or_count_drift() -> None:
    report = compare_detector_outputs(
        (
            [[1.0, 2.0, 10.0, 12.0], [20.0, 20.0, 30.0, 30.0]],
            [0.999, 0.998],
            [3, 2],
        ),
        (
            [[1.0, 2.0, 10.0, 12.0]],
            [0.999],
            [2],
        ),
        minimum_detections=1,
        maximum_box_error=0.05,
        maximum_score_error=1e-4,
    )

    assert not report["passed"]
    assert "detection_count=1!=2" in report["failures"]
    assert "label_mismatches=1" in report["failures"]
