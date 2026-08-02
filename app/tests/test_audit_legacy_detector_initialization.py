from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.tools.audit_legacy_detector_initialization import audit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    model = root / "model.best.pt"
    metrics = root / "metrics.partial.json"
    config = root / "run_config.json"
    manifest = root / "manifest.json"
    model.write_bytes(b"model")
    manifest.write_text('{"dataset":"fixed"}', encoding="utf-8")
    metrics.write_text(
        json.dumps(
            {
                "epochs": [
                    {
                        "epoch": epoch,
                        "test": {
                            "map": 0.66,
                            "map_50": 0.76,
                            "map_75": 0.73,
                            "priority_mark_map": 0.58,
                            "selection_score": 0.60,
                        },
                    }
                    for epoch in range(1, 7)
                ]
            }
        ),
        encoding="utf-8",
    )
    config.write_text(
        json.dumps({"prepared_manifest_sha256": _sha256(manifest)}),
        encoding="utf-8",
    )
    return model, metrics, config, manifest


def test_legacy_detector_is_explicitly_initialization_only(tmp_path: Path) -> None:
    model, metrics, config, manifest = _fixture(tmp_path)
    report = audit(
        model_path=model,
        metrics_path=metrics,
        run_config_path=config,
        prepared_manifest_path=manifest,
        minimum_epochs=6,
        minimum_map_50=0.70,
        minimum_map_75=0.65,
        minimum_priority_map=0.50,
    )
    assert report["passed"]
    assert report["deployment_eligible"] is False
    assert report["release_accuracy_evidence"] is False
    assert report["requires_current_contract_retraining_and_evaluation"] is True


def test_current_contract_cannot_be_downgraded_to_legacy_audit(
    tmp_path: Path,
) -> None:
    model, metrics, config, manifest = _fixture(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["model_contract"] = {"version": "current"}
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="normal training report"):
        audit(
            model_path=model,
            metrics_path=metrics,
            run_config_path=config,
            prepared_manifest_path=manifest,
            minimum_epochs=6,
            minimum_map_50=0.70,
            minimum_map_75=0.65,
            minimum_priority_map=0.50,
        )


def test_weak_legacy_detector_is_rejected(tmp_path: Path) -> None:
    model, metrics, config, manifest = _fixture(tmp_path)
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    for record in payload["epochs"]:
        record["test"]["priority_mark_map"] = 0.49
    metrics.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="too weak"):
        audit(
            model_path=model,
            metrics_path=metrics,
            run_config_path=config,
            prepared_manifest_path=manifest,
            minimum_epochs=6,
            minimum_map_50=0.70,
            minimum_map_75=0.65,
            minimum_priority_map=0.50,
        )
