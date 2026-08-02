from __future__ import annotations

import json
import sys
from pathlib import Path

from app.tools.check_paddleocr_detection_calibration import main


def test_calibration_pass_never_authorizes_release(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scan = tmp_path / "scan.log"
    clean = tmp_path / "clean.log"
    model = tmp_path / "model.pdparams"
    dataset = tmp_path / "merge-report.json"
    output = tmp_path / "calibration.json"
    scan_labels = tmp_path / "calibration.scan.exhaustive.paddle.det.txt"
    clean_labels = tmp_path / "calibration.clean.exhaustive.paddle.det.txt"
    scan.write_text(
        "precision: 0.999, recall: 0.999, hmean: 0.999\n",
        encoding="utf-8",
    )
    clean.write_text(
        "precision: 0.998, recall: 0.999, hmean: 0.9985\n",
        encoding="utf-8",
    )
    model.write_bytes(b"model")
    scan_labels.write_text("scan\n", encoding="utf-8")
    clean_labels.write_text("clean\n", encoding="utf-8")
    exhaustive = {
        "precision_evaluation_authorized": True,
        "hmean_evaluation_authorized": True,
        "unlabelled_visible_text_may_be_present": False,
    }
    dataset.write_text(
        json.dumps(
            {
                "output_counts": {
                    scan_labels.name: 1,
                    clean_labels.name: 1,
                },
                "output_label_coverage": {
                    scan_labels.name: exhaustive,
                    clean_labels.name: exhaustive,
                },
            }
        ),
        encoding="utf-8",
    )
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
            "--scan-labels",
            str(scan_labels),
            "--clean-labels",
            str(clean_labels),
            "--minimum",
            "0.997",
            "--completed-epoch",
            "8",
        ],
    )
    assert main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["early_stop_authorized"] is True
    assert report["integration_authorized"] is False
    assert report["release_accuracy_evidence"] is False
    assert report["test_set_used"] is False
    assert report["completed_epoch"] == 8
