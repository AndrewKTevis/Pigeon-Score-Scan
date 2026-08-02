from __future__ import annotations

"""Check training-only OCR detection calibration for safe early stopping."""

import argparse
import json
import os
from pathlib import Path

from app.tools.gate_paddleocr_detection import (
    evaluate_gate,
    parse_metrics,
    selected_output_label_coverage,
)
from app.tools.merge_ocr_training_labels import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-log", type=Path, required=True)
    parser.add_argument("--clean-log", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-report", type=Path, required=True)
    parser.add_argument("--scan-labels", type=Path)
    parser.add_argument("--clean-labels", type=Path)
    parser.add_argument("--minimum", type=float, default=0.997)
    parser.add_argument("--completed-epoch", type=int, required=True)
    args = parser.parse_args()
    if not 0.995 <= args.minimum <= 1.0:
        raise ValueError("calibration early-stop threshold must be in [0.995, 1]")
    if args.completed_epoch < 1:
        raise ValueError("completed epoch must be positive")
    for path in (args.model, args.dataset_report):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    scan_metrics = parse_metrics(args.scan_log)
    clean_metrics = parse_metrics(args.clean_log)
    metrics_passed, checks = evaluate_gate(
        scan_metrics=scan_metrics,
        clean_metrics=clean_metrics,
        minimum_scan_precision=args.minimum,
        minimum_scan_recall=args.minimum,
        minimum_scan_hmean=args.minimum,
        minimum_clean_precision=args.minimum,
        minimum_clean_recall=args.minimum,
        minimum_clean_hmean=args.minimum,
    )
    dataset = json.loads(args.dataset_report.read_text(encoding="utf-8"))
    coverage_passed, selected_coverage = selected_output_label_coverage(
        dataset,
        args.dataset_report,
        scan_labels=args.scan_labels,
        clean_labels=args.clean_labels,
    )
    checks.insert(
        0,
        {
            "name": "exhaustive_visible_text_label_coverage",
            "actual": coverage_passed,
            "minimum": True,
            "passed": coverage_passed,
        },
    )
    passed = metrics_passed and coverage_passed
    report = {
        "schema_version": 1,
        "name": "scorescan-ppocrv6-detection-calibration-early-stop-v1",
        "role": "training_calibration_early_stop_only",
        "passed": passed,
        "early_stop_authorized": passed,
        "integration_authorized": False,
        "release_accuracy_evidence": False,
        "test_set_used": False,
        "minimum": args.minimum,
        "completed_epoch": args.completed_epoch,
        "model_sha256": sha256_file(args.model),
        "dataset_report_sha256": sha256_file(args.dataset_report),
        "selected_label_coverage": selected_coverage,
        "evaluations": {
            "registered_scan_calibration": scan_metrics,
            "clean_render_calibration": clean_metrics,
        },
        "checks": checks,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_report.with_suffix(args.output_report.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output_report)
    print(json.dumps({"passed": passed, "checks": checks}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
