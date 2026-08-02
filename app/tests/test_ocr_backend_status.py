from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import cv2
import numpy as np
import pytest

import scorescan.text_enrichment as module


class _Session:
    def __init__(self, providers: list[str]) -> None:
        self._providers = providers

    def get_providers(self) -> list[str]:
        return list(self._providers)


def _rapid_engine(providers: dict[str, list[str]]) -> object:
    def component(name: str) -> object:
        return SimpleNamespace(
            session=SimpleNamespace(session=_Session(providers[name])),
        )

    return SimpleNamespace(
        text_det=component("detection"),
        text_cls=component("classification"),
        text_rec=component("recognition"),
    )


def _reset_engine(monkeypatch) -> None:
    monkeypatch.setattr(module, "_OCR_ENGINE", None)
    monkeypatch.setattr(module, "_OCR_ENGINE_ACCELERATOR", None)
    monkeypatch.setattr(
        module,
        "_OCR_ENGINE_MODEL_VERSION",
        "rapidocr-bundled-default",
    )
    monkeypatch.setattr(
        module,
        "_OCR_ENGINE_MODEL_STATUS",
        "domain_model_absent",
    )


def test_empty_rapidocr_result_does_not_report_backend_unavailable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    cv2.imwrite(str(image_path), np.full((120, 320), 255, np.uint8))
    monkeypatch.setattr(module, "_rapidocr_rows", lambda _path: [])
    monkeypatch.setattr(module.shutil, "which", lambda _name: None)

    rows, backend = module.run_ocr(image_path)

    assert rows == []
    assert backend == "rapidocr"


def test_failed_rapidocr_initialization_reports_unavailable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    cv2.imwrite(str(image_path), np.full((120, 320), 255, np.uint8))

    def fail(_path):
        raise RuntimeError("backend failed")

    monkeypatch.setattr(module, "_rapidocr_rows", fail)
    monkeypatch.setattr(module.shutil, "which", lambda _name: None)

    rows, backend = module.run_ocr(image_path)

    assert rows == []
    assert backend == "unavailable"


def test_release_gated_semantic_region_ocr_maps_contact_sheet_back_to_page(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image = np.full((300, 500), 255, np.uint8)
    cv2.putText(
        image,
        "Allegro",
        (100, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        0,
        2,
    )
    # bbox=(100,100,200,140), default 14px spacing -> 9px source margin.
    source_left, source_top = 91.0, 91.0
    scale = 96.0 / 58.0
    sheet_box = [
        [12.0 + (100.0 - source_left) * scale, 12.0 + (100.0 - source_top) * scale],
        [12.0 + (200.0 - source_left) * scale, 12.0 + (100.0 - source_top) * scale],
        [12.0 + (200.0 - source_left) * scale, 12.0 + (140.0 - source_top) * scale],
        [12.0 + (100.0 - source_left) * scale, 12.0 + (140.0 - source_top) * scale],
    ]
    monkeypatch.setattr(
        module,
        "_rapidocr_rows",
        lambda _path: [("Allegro", 0.99, sheet_box)],
    )

    rows = module._semantic_region_ocr_rows(
        image,
        [
            {
                "class_name": "tempoText",
                "bbox": [100, 100, 200, 140],
                "confidence": 0.999,
            },
            {
                "class_name": "not-a-release-class",
                "bbox": [0, 0, 50, 50],
                "confidence": 1.0,
            },
        ],
        None,
        tmp_path,
    )

    assert len(rows) == 1
    text, score, mapped, backend = rows[0]
    assert text == "Allegro"
    assert score == 0.99
    assert backend == "rapid-semantic-region:tempoText"
    assert np.asarray(mapped) == pytest.approx(
        np.asarray(
            [[100, 100], [200, 100], [200, 140], [100, 140]],
            dtype=float,
        )
    )


def test_semantic_region_ocr_rejects_text_joined_across_contact_crops(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image = np.full((200, 300), 255, np.uint8)
    monkeypatch.setattr(
        module,
        "_rapidocr_rows",
        lambda _path: [
            (
                "Allegro mf",
                0.99,
                [[12, 12], [296, 12], [296, 108], [12, 108]],
            )
        ],
    )

    rows = module._semantic_region_ocr_rows(
        image,
        [
            {
                "class_name": "tempoText",
                "bbox": [10, 50, 60, 80],
                "confidence": 0.999,
            },
            {
                "class_name": "genericDynamic",
                "bbox": [100, 50, 150, 80],
                "confidence": 0.999,
            },
        ],
        None,
        tmp_path,
    )

    assert rows == []


def test_cpu_rapidocr_requires_all_three_native_sessions(monkeypatch) -> None:
    _reset_engine(monkeypatch)
    monkeypatch.setenv(module.OCR_ACCELERATOR_ENVIRONMENT_VARIABLE, "cpu")
    observed: dict[str, object] = {}

    def fake_rapidocr(*, params):
        observed.update(params)
        return _rapid_engine(
            {
                "detection": ["CPUExecutionProvider"],
                "classification": ["CPUExecutionProvider"],
                "recognition": ["OtherExecutionProvider", "CPUExecutionProvider"],
            }
        )

    monkeypatch.setattr(module, "RapidOCR", fake_rapidocr)

    with pytest.raises(RuntimeError, match="CPU 会话未就绪"):
        module._engine()
    assert observed["EngineConfig.onnxruntime.use_cuda"] is False
    assert observed["EngineConfig.onnxruntime.inter_op_num_threads"] == 1


def test_cpu_rapidocr_reports_verified_bound_providers(monkeypatch) -> None:
    _reset_engine(monkeypatch)
    monkeypatch.setenv(module.OCR_ACCELERATOR_ENVIRONMENT_VARIABLE, "cpu")
    providers = {
        name: ["CPUExecutionProvider"]
        for name in ("detection", "classification", "recognition")
    }
    monkeypatch.setattr(
        module,
        "RapidOCR",
        lambda **_kwargs: _rapid_engine(providers),
    )

    runtime = module.ocr_engine_runtime()

    assert runtime["selected"] == "cpu"
    assert runtime["verified"] is True
    assert runtime["component_providers"] == providers


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_domain_ocr_assets_require_hashes_and_release_metrics(
    tmp_path: Path,
) -> None:
    model = b"domain-onnx"
    detection_model = b"domain-det-onnx"
    keys = b"a\nb\nc\n"
    (tmp_path / "recognition.onnx").write_bytes(model)
    (tmp_path / "detection.onnx").write_bytes(detection_model)
    (tmp_path / "keys.txt").write_bytes(keys)
    manifest = {
        "format": 1,
        "model_version": "scorescan-ppocrv6-domain-1",
        "integration_authorized": True,
        "detection_runtime_profile": (
            "ppocrv6-imagenet-db-scorescan-calibrated-v2"
        ),
        "detection_runtime_parameters": {
            **module._DOMAIN_OCR_DETECTION_FIXED_PARAMETERS,
            "Det.thresh": 0.25,
            "Det.box_thresh": 0.7,
            "Det.unclip_ratio": 1.4,
        },
        "evaluations": {
            "registered_scan_test": {
                "acc": 0.999,
                "norm_edit_dis": 0.9998,
                "precision": 0.997,
                "recall": 0.996,
                "hmean": 0.9965,
            },
            "clean_render_test": {
                "acc": 0.9985,
                "norm_edit_dis": 0.9997,
                "precision": 0.998,
                "recall": 0.997,
                "hmean": 0.9975,
            },
            "independent_registered_scan_holdout": {
                "acc": 0.999,
                "norm_edit_dis": 0.9998,
                "precision": 0.997,
                "recall": 0.996,
                "hmean": 0.9965,
            },
        },
        "independent_holdout_coverage": {
            "sources": 200,
            "words": 1000,
            "pages": 100,
            "minimum_iou": 0.75,
        },
        "files": {
            "recognition_model": {
                "file": "recognition.onnx",
                "bytes": len(model),
                "sha256": _sha256(model),
            },
            "recognition_keys": {
                "file": "keys.txt",
                "bytes": len(keys),
                "sha256": _sha256(keys),
            },
            "detection_model": {
                "file": "detection.onnx",
                "bytes": len(detection_model),
                "sha256": _sha256(detection_model),
            },
        },
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    assets, status = module._verified_domain_ocr_assets(tmp_path)

    assert status == "domain_model_verified"
    assert assets is not None
    assert assets.model_path == tmp_path / "recognition.onnx"
    assert assets.detection_model_path == tmp_path / "detection.onnx"

    manifest["evaluations"].pop("independent_registered_scan_holdout")
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    assets, status = module._verified_domain_ocr_assets(tmp_path)
    assert assets is None
    assert status == "domain_evaluation_missing"

    manifest["evaluations"]["independent_registered_scan_holdout"] = {
        "acc": 0.999,
        "norm_edit_dis": 0.9998,
        "precision": 0.997,
        "recall": 0.996,
        "hmean": 0.9965,
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (tmp_path / "recognition.onnx").write_bytes(b"tampered")
    assets, status = module._verified_domain_ocr_assets(tmp_path)
    assert assets is None
    assert status in {"domain_recognition_model_size", "domain_recognition_model_hash"}


def test_engine_binds_only_verified_domain_recognition_assets(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _reset_engine(monkeypatch)
    monkeypatch.setenv(module.OCR_ACCELERATOR_ENVIRONMENT_VARIABLE, "cpu")
    model = tmp_path / "domain.onnx"
    detection_model = tmp_path / "domain-det.onnx"
    keys = tmp_path / "keys.txt"
    model.write_bytes(b"model")
    detection_model.write_bytes(b"detection")
    keys.write_text("a\n", encoding="utf-8")
    assets = module.DomainOcrAssets(
        model,
        keys,
        "scorescan-domain-test",
        tmp_path / "manifest.json",
        detection_model,
        {
            **module._DOMAIN_OCR_DETECTION_FIXED_PARAMETERS,
            "Det.thresh": 0.25,
            "Det.box_thresh": 0.7,
            "Det.unclip_ratio": 1.4,
        },
    )
    monkeypatch.setattr(
        module,
        "_verified_domain_ocr_assets",
        lambda: (assets, "domain_model_verified"),
    )
    observed: dict[str, object] = {}
    providers = {
        name: ["CPUExecutionProvider"]
        for name in ("detection", "classification", "recognition")
    }

    def fake_rapidocr(*, params):
        observed.update(params)
        return _rapid_engine(providers)

    monkeypatch.setattr(module, "RapidOCR", fake_rapidocr)

    runtime = module.ocr_engine_runtime()

    assert observed["Rec.model_path"] == str(model)
    assert observed["Rec.rec_keys_path"] == str(keys)
    assert observed["Det.model_path"] == str(detection_model)
    assert observed["Det.mean"] == [0.485, 0.456, 0.406]
    assert observed["Det.std"] == [0.229, 0.224, 0.225]
    assert observed["Det.limit_side_len"] == 736
    assert observed["Det.thresh"] == 0.25
    assert observed["Det.box_thresh"] == 0.7
    assert observed["Det.unclip_ratio"] == 1.4
    assert runtime["recognition_model_version"] == "scorescan-domain-test"
    assert runtime["recognition_model_status"] == "domain_model_verified"
    assert runtime["detection_model_version"] == "scorescan-domain-test"
    assert runtime["detection_model_status"] == "domain_model_verified"
