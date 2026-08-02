#!/usr/bin/env python3
"""Gate PaddleOCR text-detection evaluations before export or integration."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from app.tools.merge_ocr_training_labels import sha256_file


METRIC_NAMES = ("precision", "recall", "hmean")
METRIC_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?P<name>precision|recall|hmean)"
    r"(?![A-Za-z0-9_])\s*['\"]?\s*[:=]\s*"
    r"(?P<value>[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
)


def parse_metrics(log_path: Path) -> dict[str, float]:
    if not log_path.is_file() or log_path.stat().st_size <= 0:
        raise FileNotFoundError(log_path)
    values: dict[str, list[float]] = {
        name: [] for name in METRIC_NAMES
    }
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for match in METRIC_PATTERN.finditer(text):
        value = float(match.group("value"))
        if math.isfinite(value):
            values[match.group("name")].append(value)
    if any(not metric_values for metric_values in values.values()):
        raise ValueError(f"detection metrics missing from {log_path}")
    result = {
        name: metric_values[-1]
        for name, metric_values in values.items()
    }
    if any(not 0 <= value <= 1 for value in result.values()):
        raise ValueError(f"detection metrics outside [0, 1]: {result}")
    return result


def evaluate_gate(
    *,
    scan_metrics: dict[str, float],
    clean_metrics: dict[str, float],
    minimum_scan_precision: float,
    minimum_scan_recall: float,
    minimum_scan_hmean: float,
    minimum_clean_precision: float,
    minimum_clean_recall: float,
    minimum_clean_hmean: float,
) -> tuple[bool, list[dict[str, Any]]]:
    checks = []
    thresholds = {
        "registered_scan": {
            "precision": minimum_scan_precision,
            "recall": minimum_scan_recall,
            "hmean": minimum_scan_hmean,
        },
        "clean_render": {
            "precision": minimum_clean_precision,
            "recall": minimum_clean_recall,
            "hmean": minimum_clean_hmean,
        },
    }
    for domain, metrics in (
        ("registered_scan", scan_metrics),
        ("clean_render", clean_metrics),
    ):
        for metric in METRIC_NAMES:
            check = {
                "name": f"{domain}_{metric}",
                "actual": metrics[metric],
                "minimum": thresholds[domain][metric],
            }
            check["passed"] = check["actual"] >= check["minimum"]
            checks.append(check)
    return all(bool(check["passed"]) for check in checks), checks


def selected_output_label_coverage(
    dataset: dict[str, Any],
    dataset_report_path: Path,
    *,
    scan_labels: Path | None,
    clean_labels: Path | None,
) -> tuple[bool, dict[str, Any]]:
    if scan_labels is None or clean_labels is None:
        return False, {}
    output_counts = dataset.get("output_counts")
    coverages = dataset.get("output_label_coverage")
    if not isinstance(output_counts, dict) or not isinstance(coverages, dict):
        return False, {}
    selected: dict[str, Any] = {}
    for domain, path in (
        ("registered_scan", scan_labels),
        ("clean_render", clean_labels),
    ):
        resolved = path.resolve()
        if (
            resolved.parent != dataset_report_path.resolve().parent
            or not resolved.is_file()
            or output_counts.get(resolved.name) is None
        ):
            return False, {}
        coverage = coverages.get(resolved.name)
        if not isinstance(coverage, dict):
            return False, {}
        selected[domain] = {
            "labels": str(resolved),
            "filename": resolved.name,
            **coverage,
        }
    passed = all(
        coverage.get("precision_evaluation_authorized") is True
        and coverage.get("hmean_evaluation_authorized") is True
        and coverage.get("unlabelled_visible_text_may_be_present") is False
        for coverage in selected.values()
    )
    return passed, selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-log", type=Path, required=True)
    parser.add_argument("--clean-log", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-report", type=Path, required=True)
    parser.add_argument("--scan-labels", type=Path)
    parser.add_argument("--clean-labels", type=Path)
    parser.add_argument("--minimum-scan-precision", type=float, default=0.995)
    parser.add_argument("--minimum-scan-recall", type=float, default=0.995)
    parser.add_argument("--minimum-scan-hmean", type=float, default=0.995)
    parser.add_argument("--minimum-clean-precision", type=float, default=0.995)
    parser.add_argument("--minimum-clean-recall", type=float, default=0.995)
    parser.add_argument("--minimum-clean-hmean", type=float, default=0.995)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in (args.model, args.dataset_report):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    threshold_values = (
        args.minimum_scan_precision,
        args.minimum_scan_recall,
        args.minimum_scan_hmean,
        args.minimum_clean_precision,
        args.minimum_clean_recall,
        args.minimum_clean_hmean,
    )
    if any(not 0 <= value <= 1 for value in threshold_values):
        raise ValueError("all detection thresholds must be in [0, 1]")
    scan_metrics = parse_metrics(args.scan_log)
    clean_metrics = parse_metrics(args.clean_log)
    metrics_passed, checks = evaluate_gate(
        scan_metrics=scan_metrics,
        clean_metrics=clean_metrics,
        minimum_scan_precision=args.minimum_scan_precision,
        minimum_scan_recall=args.minimum_scan_recall,
        minimum_scan_hmean=args.minimum_scan_hmean,
        minimum_clean_precision=args.minimum_clean_precision,
        minimum_clean_recall=args.minimum_clean_recall,
        minimum_clean_hmean=args.minimum_clean_hmean,
    )
    dataset = json.loads(args.dataset_report.read_text(encoding="utf-8"))
    coverage_passed, coverage = selected_output_label_coverage(
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
        "name": "scorescan-ppocrv6-domain-detection-release-gate-v1",
        "passed": passed,
        "model": str(args.model.resolve()),
        "model_sha256": sha256_file(args.model),
        "dataset_report": str(args.dataset_report.resolve()),
        "dataset_report_sha256": sha256_file(args.dataset_report),
        "label_coverage_contract": coverage,
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
