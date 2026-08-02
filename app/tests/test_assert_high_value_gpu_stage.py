from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.tools.assert_high_value_gpu_stage import verify_stage
from scorescan.util import sha256_file


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_recognition_stage_binds_passed_gate_to_exact_model(
    tmp_path: Path,
) -> None:
    model_dir = (
        tmp_path
        / "training_data/models/"
        "ppocrv6-scorescan-rec-stratified-e18-b8-20260729"
    )
    model_dir.mkdir(parents=True)
    model = model_dir / "scorescan-ppocrv6-rec.onnx"
    model.write_bytes(b"recognition")
    _write_json(
        model_dir / "onnx-release-gate.json",
        {
            "name": "scorescan-ppocrv6-domain-release-gate-v1",
            "passed": True,
            "integration_authorized": True,
            "model_sha256": sha256_file(model),
        },
    )

    report = verify_stage(tmp_path, "ocr_recognition")
    assert report["passed"]
    assert report["artifacts"][1]["sha256"] == sha256_file(model)

    model.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="not accepted"):
        verify_stage(tmp_path, "ocr_recognition")


def test_zeus_stage_requires_both_calibration_and_upstream_acceptance(
    tmp_path: Path,
) -> None:
    completion = (
        tmp_path
        / "training_data/logs/"
        "zeus-family-priority-source-doc-safe-queue.completion.json"
    )
    calibration = (
        tmp_path
        / "training_data/models/"
        "zeus-olimpic-real-family-priority-source-doc-safe-mp-e6-b16-lr5e6-"
        "20260729/family-priority-calibration-gate.json"
    )
    upstream = (
        tmp_path
        / "training_data/models/"
        "zeus-olimpic-real-family-priority-source-doc-safe-upstream-test-"
        "20260729/upstream-candidate-test-gate.json"
    )
    _write_json(
        completion,
        {
            "state": "completed",
            "calibration_passed": True,
            "upstream_candidate_test_opened": True,
            "upstream_candidate_passed": False,
        },
    )
    _write_json(
        calibration,
        {
            "passed": True,
            "candidate_test_evaluation_authorized": True,
        },
    )
    _write_json(
        upstream,
        {"passed": False, "observer_candidate_authorized": False},
    )

    with pytest.raises(ValueError, match="not accepted"):
        verify_stage(tmp_path, "semantic_family_priority")


def test_zeus_stage_binds_each_gate_to_its_exact_input_report(
    tmp_path: Path,
) -> None:
    model_dir = (
        tmp_path
        / "training_data/models/"
        "zeus-olimpic-real-family-priority-source-doc-safe-mp-e6-b16-lr5e6-"
        "20260729"
    )
    evaluation_dir = (
        tmp_path
        / "training_data/models/"
        "zeus-olimpic-real-family-priority-source-doc-safe-upstream-test-"
        "20260729"
    )
    completion = (
        tmp_path
        / "training_data/logs/"
        "zeus-family-priority-source-doc-safe-queue.completion.json"
    )
    training_report = model_dir / "training_report.json"
    evaluation_report = evaluation_dir / "training_report.json"
    calibration = model_dir / "family-priority-calibration-gate.json"
    upstream = evaluation_dir / "upstream-candidate-test-gate.json"
    training_report.parent.mkdir(parents=True)
    evaluation_report.parent.mkdir(parents=True)
    training_report.write_bytes(b"training-report")
    evaluation_report.write_bytes(b"evaluation-report")
    _write_json(
        completion,
        {
            "state": "completed",
            "calibration_passed": True,
            "upstream_candidate_test_opened": True,
            "upstream_candidate_passed": True,
        },
    )
    _write_json(
        calibration,
        {
            "passed": True,
            "candidate_test_evaluation_authorized": True,
            "input": {
                "training_report": {
                    "bytes": training_report.stat().st_size,
                    "sha256": sha256_file(training_report),
                }
            },
        },
    )
    _write_json(
        upstream,
        {
            "passed": True,
            "observer_candidate_authorized": True,
            "input": {
                "evaluation_report": {
                    "bytes": evaluation_report.stat().st_size,
                    "sha256": sha256_file(evaluation_report),
                }
            },
        },
    )

    assert verify_stage(tmp_path, "semantic_family_priority")["passed"]

    evaluation_report.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="not accepted"):
        verify_stage(tmp_path, "semantic_family_priority")
