#!/usr/bin/env python3
"""Evaluate the one-time OLiMPiC crop test without claiming product quality."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


MINIMUM_POSITIONED_F1 = {
    "pitch": 95.0,
    "rhythm": 95.0,
    "slur": 95.0,
    "tie": 95.0,
    "beam": 95.0,
    "articulation": 95.0,
    "accidental": 95.0,
    "attributes": 95.0,
}
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


def evaluate_upstream_test(
    report: dict[str, Any], *, maximum_ser: float
) -> dict[str, Any]:
    data = report.get("data", {})
    metrics = report.get("metrics_percent", {})
    candidate = metrics.get("candidate_test_best", {})
    families = candidate.get("families", {})
    reasons: list[str] = []

    if data.get("manifest_name") != EXPECTED_MANIFEST:
        reasons.append("wrong source-document-safe candidate-test manifest")
    if data.get("source_document_isolation_verified") is not True:
        reasons.append("physical source-document isolation is not verified")
    source_document_overlap = data.get("source_document_overlap")
    if not isinstance(source_document_overlap, dict) or any(
        int(value) != 0 for value in source_document_overlap.values()
    ):
        reasons.append("physical source-document overlap is non-zero or absent")
    if data.get("candidate_test_opened") is not True:
        reasons.append("upstream candidate-test was not evaluated")
    if data.get("candidate_test_is_final_product_benchmark") is not False:
        reasons.append("crop-level upstream test was mislabeled as product evidence")
    observed_ser = float(candidate.get("SER", math.inf))
    if not math.isfinite(observed_ser) or observed_ser > maximum_ser:
        reasons.append(
            f"candidate-test SER {observed_ser:.6f} > {maximum_ser:.6f}"
        )

    observed_f1: dict[str, float] = {}
    for family, threshold in MINIMUM_POSITIONED_F1.items():
        family_metrics = families.get(family)
        if not isinstance(family_metrics, dict):
            reasons.append(f"missing {family} candidate-test metrics")
            continue
        if int(family_metrics.get("reference_tokens", 0)) <= 0:
            reasons.append(f"candidate-test has no {family} reference tokens")
            continue
        value = float(family_metrics.get("positioned_f1_percent", math.nan))
        observed_f1[family] = value
        if not math.isfinite(value) or value < threshold:
            reasons.append(
                f"{family} positioned-F1 {value:.6f} < {threshold:.6f}"
            )

    return {
        "schema_version": 1,
        "role": "upstream_olimpic_crop_candidate_test_gate",
        "passed": not reasons,
        "observer_candidate_authorized": not reasons,
        "desktop_deployment_authorized": False,
        "final_product_release_evidence": False,
        "limitations": [
            "grandstaff semantic labels rather than complete page semantics",
            "pianoform voice-plus-piano source family only",
            "same OLiMPiC/OpenScore Lieder-derived dataset family despite "
            "source-document-disjoint candidate pages",
        ],
        "thresholds": {
            "maximum_ser_percent": maximum_ser,
            "minimum_positioned_f1_percent": MINIMUM_POSITIONED_F1,
        },
        "observed": {
            "ser_percent": observed_ser,
            "positioned_f1_percent": observed_f1,
        },
        "reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--maximum-ser", type=float, default=5.0)
    args = parser.parse_args()
    report = json.loads(args.evaluation_report.read_text(encoding="utf-8"))
    result = evaluate_upstream_test(report, maximum_ser=args.maximum_ser)
    result["input"] = {
        "evaluation_report": _input_record(args.evaluation_report),
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
