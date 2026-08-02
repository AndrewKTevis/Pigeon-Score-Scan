from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.tools import gate_paddleocr_detection as module


def test_parse_metrics_uses_final_evaluation_values(
    tmp_path: Path,
) -> None:
    log = tmp_path / "evaluation.log"
    log.write_text(
        "precision: 0.8, recall: 0.7, hmean: 0.746\n"
        "[eval] {'precision': 0.998, 'recall': 0.997, 'hmean': 0.9975}\n",
        encoding="utf-8",
    )
    assert module.parse_metrics(log) == {
        "precision": 0.998,
        "recall": 0.997,
        "hmean": 0.9975,
    }


def test_gate_requires_precision_recall_and_hmean_in_both_domains() -> None:
    passed, checks = module.evaluate_gate(
        scan_metrics={"precision": 0.996, "recall": 0.994, "hmean": 0.995},
        clean_metrics={"precision": 1.0, "recall": 1.0, "hmean": 1.0},
        minimum_scan_precision=0.995,
        minimum_scan_recall=0.995,
        minimum_scan_hmean=0.995,
        minimum_clean_precision=0.995,
        minimum_clean_recall=0.995,
        minimum_clean_hmean=0.995,
    )
    assert not passed
    assert [check["name"] for check in checks if not check["passed"]] == [
        "registered_scan_recall"
    ]


def test_main_writes_failed_non_authorized_gate(tmp_path: Path) -> None:
    scan_log = tmp_path / "scan.log"
    clean_log = tmp_path / "clean.log"
    scan_log.write_text(
        "precision:0.99 recall:0.99 hmean:0.99\n",
        encoding="utf-8",
    )
    clean_log.write_text(
        "precision:1 recall:1 hmean:1\n",
        encoding="utf-8",
    )
    model = tmp_path / "best_accuracy.pdparams"
    model.write_bytes(b"model")
    dataset = tmp_path / "merge-report.json"
    dataset.write_text("{}", encoding="utf-8")
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
            str(dataset),
        ]
    ) == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert not report["passed"]
    assert not report["integration_authorized"]
    assert report["checks"][0] == {
        "name": "exhaustive_visible_text_label_coverage",
        "actual": False,
        "minimum": True,
        "passed": False,
    }


def test_parse_rejects_incomplete_log(tmp_path: Path) -> None:
    log = tmp_path / "bad.log"
    log.write_text("precision: 1\nrecall: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        module.parse_metrics(log)


def test_selected_output_coverage_is_independent_per_evaluation_file(
    tmp_path: Path,
) -> None:
    scan = tmp_path / "test.scan.exhaustive.paddle.det.txt"
    clean = tmp_path / "test.clean.exhaustive.paddle.det.txt"
    scan.write_text("scan\n", encoding="utf-8")
    clean.write_text("clean\n", encoding="utf-8")
    report_path = tmp_path / "merge-report.json"
    exhaustive = {
        "pages": 1,
        "precision_evaluation_authorized": True,
        "hmean_evaluation_authorized": True,
        "unlabelled_visible_text_may_be_present": False,
    }
    dataset = {
        "label_coverage_contract": {
            "precision_evaluation_authorized": False,
        },
        "output_counts": {
            scan.name: 1,
            clean.name: 1,
        },
        "output_label_coverage": {
            scan.name: exhaustive,
            clean.name: exhaustive,
        },
    }

    passed, selected = module.selected_output_label_coverage(
        dataset,
        report_path,
        scan_labels=scan,
        clean_labels=clean,
    )

    assert passed
    assert selected["registered_scan"]["filename"] == scan.name
    assert selected["clean_render"]["filename"] == clean.name
