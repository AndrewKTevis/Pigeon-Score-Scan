#!/usr/bin/env python3
"""Gate PaddleOCR recognition evaluations before model export or integration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any


METRIC_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?P<name>acc|norm_edit_dis)"
    r"(?![A-Za-z0-9_])\s*['\"]?\s*[:=]\s*"
    r"(?P<value>[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_metrics(log_path: Path) -> dict[str, float]:
    if not log_path.is_file() or log_path.stat().st_size <= 0:
        raise FileNotFoundError(log_path)
    values: dict[str, list[float]] = {
        "acc": [],
        "norm_edit_dis": [],
    }
    for match in METRIC_PATTERN.finditer(
        log_path.read_text(encoding="utf-8", errors="replace")
    ):
        value = float(match.group("value"))
        if math.isfinite(value):
            values[match.group("name")].append(value)
    if any(not metric_values for metric_values in values.values()):
        raise ValueError(f"evaluation metrics missing from {log_path}")
    result = {
        name: metric_values[-1]
        for name, metric_values in values.items()
    }
    if any(not 0 <= value <= 1 for value in result.values()):
        raise ValueError(f"evaluation metrics outside [0, 1]: {result}")
    return result


def evaluate_gate(
    *,
    scan_metrics: dict[str, float],
    clean_metrics: dict[str, float],
    minimum_scan_accuracy: float,
    minimum_scan_normalized_edit: float,
    minimum_clean_accuracy: float,
    minimum_clean_normalized_edit: float,
) -> tuple[bool, list[dict[str, Any]]]:
    checks = [
        {
            "name": "scan_word_accuracy",
            "actual": scan_metrics["acc"],
            "minimum": minimum_scan_accuracy,
        },
        {
            "name": "scan_normalized_edit_distance",
            "actual": scan_metrics["norm_edit_dis"],
            "minimum": minimum_scan_normalized_edit,
        },
        {
            "name": "clean_word_accuracy",
            "actual": clean_metrics["acc"],
            "minimum": minimum_clean_accuracy,
        },
        {
            "name": "clean_normalized_edit_distance",
            "actual": clean_metrics["norm_edit_dis"],
            "minimum": minimum_clean_normalized_edit,
        },
    ]
    for check in checks:
        check["passed"] = check["actual"] >= check["minimum"]
    return all(bool(check["passed"]) for check in checks), checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-log", type=Path, required=True)
    parser.add_argument("--clean-log", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-report", type=Path, required=True)
    parser.add_argument("--minimum-scan-accuracy", type=float, default=0.998)
    parser.add_argument(
        "--minimum-scan-normalized-edit",
        type=float,
        default=0.9995,
    )
    parser.add_argument("--minimum-clean-accuracy", type=float, default=0.998)
    parser.add_argument(
        "--minimum-clean-normalized-edit",
        type=float,
        default=0.9995,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in (args.model, args.dataset_report):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    thresholds = (
        args.minimum_scan_accuracy,
        args.minimum_scan_normalized_edit,
        args.minimum_clean_accuracy,
        args.minimum_clean_normalized_edit,
    )
    if any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("all thresholds must be in [0, 1]")

    scan_metrics = parse_metrics(args.scan_log)
    clean_metrics = parse_metrics(args.clean_log)
    passed, checks = evaluate_gate(
        scan_metrics=scan_metrics,
        clean_metrics=clean_metrics,
        minimum_scan_accuracy=args.minimum_scan_accuracy,
        minimum_scan_normalized_edit=args.minimum_scan_normalized_edit,
        minimum_clean_accuracy=args.minimum_clean_accuracy,
        minimum_clean_normalized_edit=args.minimum_clean_normalized_edit,
    )
    report = {
        "schema_version": 1,
        "name": "scorescan-ppocrv6-domain-release-gate-v1",
        "passed": passed,
        "model": str(args.model.resolve()),
        "model_sha256": sha256_file(args.model),
        "dataset_report": str(args.dataset_report.resolve()),
        "dataset_report_sha256": sha256_file(args.dataset_report),
        "evaluations": {
            "registered_scan_test": {
                **scan_metrics,
                "log": str(args.scan_log.resolve()),
                "log_sha256": sha256_file(args.scan_log),
            },
            "clean_render_test": {
                **clean_metrics,
                "log": str(args.clean_log.resolve()),
                "log_sha256": sha256_file(args.clean_log),
            },
        },
        "checks": checks,
        "integration_authorized": passed,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_report.with_suffix(
        args.output_report.suffix + ".tmp"
    )
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output_report)
    print(json.dumps({"passed": passed, "checks": checks}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
