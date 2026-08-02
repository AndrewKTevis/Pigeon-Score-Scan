from __future__ import annotations

"""Check training-only OCR recognition calibration for safe early stopping."""

import argparse
import json
import os
from pathlib import Path

from app.tools.gate_paddleocr_evaluation import (
    evaluate_gate,
    parse_metrics,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-log", type=Path, required=True)
    parser.add_argument("--clean-log", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-report", type=Path, required=True)
    parser.add_argument("--minimum-accuracy", type=float, default=0.999)
    parser.add_argument("--minimum-normalized-edit", type=float, default=0.9997)
    parser.add_argument("--completed-epoch", type=int, required=True)
    args = parser.parse_args()
    if not 0.998 <= args.minimum_accuracy <= 1.0:
        raise ValueError("calibration accuracy threshold must be in [0.998, 1]")
    if not 0.9995 <= args.minimum_normalized_edit <= 1.0:
        raise ValueError(
            "calibration normalized-edit threshold must be in [0.9995, 1]"
        )
    if args.completed_epoch < 1:
        raise ValueError("completed epoch must be positive")
    for path in (args.model, args.dataset_report):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    scan_metrics = parse_metrics(args.scan_log)
    clean_metrics = parse_metrics(args.clean_log)
    passed, checks = evaluate_gate(
        scan_metrics=scan_metrics,
        clean_metrics=clean_metrics,
        minimum_scan_accuracy=args.minimum_accuracy,
        minimum_scan_normalized_edit=args.minimum_normalized_edit,
        minimum_clean_accuracy=args.minimum_accuracy,
        minimum_clean_normalized_edit=args.minimum_normalized_edit,
    )
    report = {
        "schema_version": 1,
        "name": "scorescan-ppocrv6-recognition-calibration-early-stop-v1",
        "role": "training_calibration_early_stop_only",
        "passed": passed,
        "early_stop_authorized": passed,
        "integration_authorized": False,
        "release_accuracy_evidence": False,
        "test_set_used": False,
        "minimum_accuracy": args.minimum_accuracy,
        "minimum_normalized_edit": args.minimum_normalized_edit,
        "completed_epoch": args.completed_epoch,
        "model_sha256": sha256_file(args.model),
        "dataset_report_sha256": sha256_file(args.dataset_report),
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
