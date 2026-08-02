from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.tools import gate_paddleocr_evaluation as module


def test_parse_metrics_uses_final_paddle_evaluation_values(
    tmp_path: Path,
) -> None:
    log = tmp_path / "eval.log"
    log.write_text(
        "acc:0.7\nnorm_edit_dis:0.8\n"
        "[2026] metric eval {'acc': 0.9985, 'norm_edit_dis': 0.9997}\n",
        encoding="utf-8",
    )
    assert module.parse_metrics(log) == {
        "acc": 0.9985,
        "norm_edit_dis": 0.9997,
    }


def test_gate_requires_every_scan_and_clean_metric() -> None:
    passed, checks = module.evaluate_gate(
        scan_metrics={"acc": 0.999, "norm_edit_dis": 0.9998},
        clean_metrics={"acc": 0.997, "norm_edit_dis": 1.0},
        minimum_scan_accuracy=0.998,
        minimum_scan_normalized_edit=0.9995,
        minimum_clean_accuracy=0.998,
        minimum_clean_normalized_edit=0.9995,
    )
    assert not passed
    assert [check["name"] for check in checks if not check["passed"]] == [
        "clean_word_accuracy"
    ]


def test_main_writes_failed_report_and_refuses_integration(
    tmp_path: Path,
) -> None:
    scan_log = tmp_path / "scan.log"
    clean_log = tmp_path / "clean.log"
    scan_log.write_text("acc:0.990\nnorm_edit_dis:0.999\n", encoding="utf-8")
    clean_log.write_text("acc:1.0\nnorm_edit_dis:1.0\n", encoding="utf-8")
    model = tmp_path / "best_accuracy.pdparams"
    model.write_bytes(b"model")
    dataset_report = tmp_path / "merge-report.json"
    dataset_report.write_text("{}", encoding="utf-8")
    output = tmp_path / "gate.json"

    assert module.main(
        [
            "--scan-log",
            str(scan_log),
            "--clean-log",
            str(clean_log),
            "--output-report",
            str(output),
            "--model",
            str(model),
            "--dataset-report",
            str(dataset_report),
        ]
    ) == 1

    report = json.loads(output.read_text(encoding="utf-8"))
    assert not report["passed"]
    assert not report["integration_authorized"]


def test_parse_rejects_incomplete_log(tmp_path: Path) -> None:
    log = tmp_path / "bad.log"
    log.write_text("acc: 1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        module.parse_metrics(log)
