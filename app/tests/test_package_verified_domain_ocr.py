from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.tools import package_verified_domain_ocr as module


def _gate(path: Path, model: Path, *, detection: bool) -> None:
    metrics = (
        {"precision": 0.997, "recall": 0.996, "hmean": 0.9965}
        if detection
        else {"acc": 0.999, "norm_edit_dis": 0.9998}
    )
    path.write_text(
        json.dumps(
            {
                "passed": True,
                "integration_authorized": True,
                "model_sha256": module.sha256_file(model),
                "evaluations": {
                    "registered_scan_test": metrics,
                    "clean_render_test": metrics,
                },
                "name": "test-gate",
            }
        ),
        encoding="utf-8",
    )


def _holdout_gate(
    path: Path,
    recognition: Path,
    keys: Path,
    detection: Path,
) -> None:
    runtime_parameters = {
        name: module.DETECTION_RUNTIME_FIXED_PARAMETERS[name]
        for name in module.DETECTION_DEPLOY_PARAMETER_NAMES
        if name in module.DETECTION_RUNTIME_FIXED_PARAMETERS
    }
    runtime_parameters.update(
        {
            "Det.thresh": 0.25,
            "Det.box_thresh": 0.7,
            "Det.unclip_ratio": 1.4,
        }
    )
    path.write_text(
        json.dumps(
            {
                "name": "scorescan-independent-scan-ocr-release-gate-v1",
                "passed": True,
                "integration_authorized": True,
                "recognition_model_sha256": module.sha256_file(recognition),
                "recognition_keys_sha256": module.sha256_file(keys),
                "detection_model_sha256": module.sha256_file(detection),
                "detection_runtime_parameters": runtime_parameters,
                "coverage": {
                    "sources": 200,
                    "words": 1000,
                    "pages": 100,
                    "minimum_iou": 0.75,
                },
                "evaluations": {
                    "independent_registered_scan_holdout": {
                        "recognition": {
                            "acc": 0.999,
                            "norm_edit_dis": 0.9998,
                        },
                        "detection": {
                            "precision": 0.997,
                            "recall": 0.996,
                            "hmean": 0.9965,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_packages_only_jointly_gated_models(tmp_path: Path) -> None:
    recognition = tmp_path / "rec.onnx"
    detection = tmp_path / "det.onnx"
    keys = tmp_path / "keys.txt"
    recognition.write_bytes(b"recognition")
    detection.write_bytes(b"detection")
    keys.write_text("a\nb\n", encoding="utf-8")
    rec_gate = tmp_path / "rec-gate.json"
    det_gate = tmp_path / "det-gate.json"
    holdout_gate = tmp_path / "holdout-gate.json"
    _gate(rec_gate, recognition, detection=False)
    _gate(det_gate, detection, detection=True)
    _holdout_gate(holdout_gate, recognition, keys, detection)
    output = tmp_path / "ocr"
    assert module.main(
        [
            "--recognition-model",
            str(recognition),
            "--recognition-keys",
            str(keys),
            "--recognition-gate",
            str(rec_gate),
            "--detection-model",
            str(detection),
            "--detection-gate",
            str(det_gate),
            "--independent-holdout-gate",
            str(holdout_gate),
            "--output-dir",
            str(output),
            "--model-version",
            "scorescan-ocr-test",
        ]
    ) == 0
    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["integration_authorized"] is True
    assert manifest["detection_runtime_profile"] == module.RUNTIME_PROFILE
    assert manifest["evaluations"]["registered_scan_test"] == {
        "acc": 0.999,
        "norm_edit_dis": 0.9998,
        "precision": 0.997,
        "recall": 0.996,
        "hmean": 0.9965,
    }
    assert manifest["evaluations"]["independent_registered_scan_holdout"] == {
        "acc": 0.999,
        "norm_edit_dis": 0.9998,
        "precision": 0.997,
        "recall": 0.996,
        "hmean": 0.9965,
    }
    assert manifest["independent_holdout_coverage"]["minimum_iou"] == 0.75
    assert manifest["detection_runtime_parameters"]["Det.thresh"] == 0.25
    assert (
        manifest["detection_runtime_parameters"]["Det.box_thresh"] == 0.7
    )


def test_rejects_model_whose_hash_does_not_match_gate(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"first")
    gate = tmp_path / "gate.json"
    _gate(gate, model, detection=False)
    model.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="does not authorize"):
        module._load_gate(gate, model=model)


def test_rejects_detection_below_packaging_floor(tmp_path: Path) -> None:
    recognition = {
        "evaluations": {
            domain: {"acc": 0.999, "norm_edit_dis": 0.9998}
            for domain in ("registered_scan_test", "clean_render_test")
        }
    }
    detection = {
        "evaluations": {
            domain: {"precision": 0.994, "recall": 1.0, "hmean": 0.997}
            for domain in ("registered_scan_test", "clean_render_test")
        }
    }
    with pytest.raises(ValueError, match="below packaging floor"):
        module._combined_evaluations(recognition, detection)
    detection["evaluations"]["registered_scan_test"]["precision"] = float(
        "nan"
    )
    with pytest.raises(ValueError, match="below packaging floor"):
        module._combined_evaluations(recognition, detection)


def test_rejects_independent_holdout_with_wrong_model_hash(
    tmp_path: Path,
) -> None:
    recognition = tmp_path / "rec.onnx"
    detection = tmp_path / "det.onnx"
    recognition.write_bytes(b"recognition")
    detection.write_bytes(b"detection")
    gate = tmp_path / "holdout.json"
    keys = tmp_path / "keys.txt"
    keys.write_text("a\n", encoding="utf-8")
    _holdout_gate(gate, recognition, keys, detection)
    detection.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="does not authorize"):
        module._load_independent_holdout_gate(
            gate,
            recognition_model=recognition,
            recognition_keys=keys,
            detection_model=detection,
        )
