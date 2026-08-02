#!/usr/bin/env python3
"""Gate an OLiMPiC Zeus candidate without treating it as product evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


CORE_FAMILIES = (
    "pitch",
    "rhythm",
    "slur",
    "tie",
    "beam",
    "articulation",
    "accidental",
    "attributes",
)
EXPECTED_MANIFEST = (
    "scorescan-olimpic-real-plus-synthetic-replay-v4-source-document-safe"
)


def _input_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def evaluate_candidate(
    report: dict[str, Any],
    *,
    minimum_ser_improvement: float,
    minimum_tie_improvement: float,
    minimum_slur_improvement: float,
    maximum_other_family_regression: float,
) -> dict[str, Any]:
    runtime = report.get("runtime", {})
    data = report.get("data", {})
    metrics = report.get("metrics_percent", {})
    baseline = metrics.get("calibration_baseline", {})
    best = metrics.get("calibration_best", {})
    baseline_families = baseline.get("families", {})
    best_families = best.get("families", {})

    reasons: list[str] = []
    if runtime.get("keras_policy") != "mixed_float16":
        reasons.append("training did not use the validated mixed_float16 policy")
    if not runtime.get("gpu_devices"):
        reasons.append("training report contains no GPU device")
    if data.get("manifest_name") != EXPECTED_MANIFEST:
        reasons.append("wrong family-priority dataset manifest")
    if data.get("source_document_isolation_verified") is not True:
        reasons.append("physical source-document isolation is not verified")
    source_document_overlap = data.get("source_document_overlap")
    if not isinstance(source_document_overlap, dict) or any(
        int(value) != 0 for value in source_document_overlap.values()
    ):
        reasons.append("physical source-document overlap is non-zero or absent")
    if data.get("candidate_test_opened") is not False:
        reasons.append("candidate-test was opened during model selection")
    if data.get("candidate_test_is_final_product_benchmark") is not False:
        reasons.append("upstream crop test was mislabeled as product evidence")
    if int(metrics.get("best_epoch", 0)) <= 0:
        reasons.append("no trained epoch beat the initialization")

    baseline_ser = float(baseline.get("SER", math.inf))
    best_ser = float(best.get("SER", math.inf))
    ser_improvement = baseline_ser - best_ser
    if not math.isfinite(ser_improvement):
        reasons.append("non-finite calibration SER")
    elif ser_improvement < minimum_ser_improvement:
        reasons.append(
            "calibration SER improvement "
            f"{ser_improvement:.6f} < {minimum_ser_improvement:.6f}"
        )

    family_improvements: dict[str, float] = {}
    for family in CORE_FAMILIES:
        baseline_family = baseline_families.get(family)
        best_family = best_families.get(family)
        if not isinstance(baseline_family, dict) or not isinstance(
            best_family, dict
        ):
            reasons.append(f"missing {family} family metrics")
            continue
        if int(baseline_family.get("reference_tokens", 0)) <= 0:
            continue
        baseline_f1 = float(
            baseline_family.get("positioned_f1_percent", math.nan)
        )
        best_f1 = float(best_family.get("positioned_f1_percent", math.nan))
        improvement = best_f1 - baseline_f1
        family_improvements[family] = improvement
        if not math.isfinite(improvement):
            reasons.append(f"non-finite {family} positioned-F1")
        elif improvement < -maximum_other_family_regression:
            reasons.append(
                f"{family} positioned-F1 regressed by {-improvement:.6f}"
            )

    for family, required in (
        ("tie", minimum_tie_improvement),
        ("slur", minimum_slur_improvement),
    ):
        improvement = family_improvements.get(family, -math.inf)
        if improvement < required:
            reasons.append(
                f"{family} positioned-F1 improvement "
                f"{improvement:.6f} < {required:.6f}"
            )

    return {
        "schema_version": 1,
        "role": "upstream_olimpic_crop_candidate_calibration_gate",
        "passed": not reasons,
        "candidate_test_evaluation_authorized": not reasons,
        "deployment_authorized": False,
        "final_product_release_evidence": False,
        "thresholds": {
            "minimum_ser_improvement": minimum_ser_improvement,
            "minimum_tie_positioned_f1_improvement": minimum_tie_improvement,
            "minimum_slur_positioned_f1_improvement": minimum_slur_improvement,
            "maximum_other_family_positioned_f1_regression": (
                maximum_other_family_regression
            ),
        },
        "observed": {
            "baseline_ser_percent": baseline_ser,
            "best_ser_percent": best_ser,
            "ser_improvement": ser_improvement,
            "family_positioned_f1_improvements": family_improvements,
        },
        "reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--minimum-ser-improvement", type=float, default=0.05)
    parser.add_argument("--minimum-tie-improvement", type=float, default=1.0)
    parser.add_argument("--minimum-slur-improvement", type=float, default=1.0)
    parser.add_argument(
        "--maximum-other-family-regression", type=float, default=0.25
    )
    args = parser.parse_args()
    report = json.loads(args.training_report.read_text(encoding="utf-8"))
    result = evaluate_candidate(
        report,
        minimum_ser_improvement=args.minimum_ser_improvement,
        minimum_tie_improvement=args.minimum_tie_improvement,
        minimum_slur_improvement=args.minimum_slur_improvement,
        maximum_other_family_regression=args.maximum_other_family_regression,
    )
    result["input"] = {
        "training_report": _input_record(args.training_report),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
