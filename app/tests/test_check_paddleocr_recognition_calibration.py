from __future__ import annotations

import json
import sys
from pathlib import Path

from app.tools.check_paddleocr_recognition_calibration import main


def test_recognition_calibration_pass_never_authorizes_release(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scan = tmp_path / "scan.log"
    clean = tmp_path / "clean.log"
    model = tmp_path / "model.pdparams"
    dataset = tmp_path / "merge-report.json"
    output = tmp_path / "calibration.json"
    scan.write_text(
        "acc: 1.0, norm_edit_dis: 1.0\n",
        encoding="utf-8",
    )
    clean.write_text(
        "acc: 0.9995, norm_edit_dis: 0.9999\n",
        encoding="utf-8",
    )
    model.write_bytes(b"model")
    dataset.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check",
            "--scan-log",
            str(scan),
            "--clean-log",
            str(clean),
            "--output-report",
            str(output),
            "--model",
            str(model),
            "--dataset-report",
            str(dataset),
            "--minimum-accuracy",
            "0.999",
            "--minimum-normalized-edit",
            "0.9997",
            "--completed-epoch",
            "10",
        ],
    )
    assert main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["early_stop_authorized"] is True
    assert report["integration_authorized"] is False
    assert report["release_accuracy_evidence"] is False
    assert report["test_set_used"] is False
    assert report["completed_epoch"] == 10
