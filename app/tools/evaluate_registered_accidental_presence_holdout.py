from __future__ import annotations

"""Evaluate a frozen accidental-presence model on forbidden-to-train scans."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scorescan.accidental_presence_guard import (  # noqa: E402
    ACCIDENTAL_PRESENCE_FEATURE_NAMES,
)
from scorescan.util import atomic_write_json  # noqa: E402
from train_accidental_presence_guard import (  # noqa: E402
    REGISTERED_MODEL_VERSION,
    _load_dataset,
    _policy,
    _sample,
    _sha256_file,
)
from tree_export import deployed_forest_probabilities  # noqa: E402

HOLDOUT_DATASET_NAME = "scorescan-registered-scan-accidental-presence-holdout-v1"
HOLDOUT_DATASET_ROLE = "independent_holdout_evaluation_only"
HOLDOUT_SOURCE_ROLE = "external_scan_degraded_development_benchmark_not_training"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _work_keys(path: Path) -> set[str]:
    result: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            group_key = str(row.get("group_key", ""))
            marker = "/pair-"
            if marker not in group_key:
                raise ValueError(
                    f"{path}:{line_number}: malformed registered group key"
                )
            result.add(group_key.split(marker, 1)[0])
    if not result:
        raise ValueError(f"registered sample metadata has no works: {path}")
    return result


def evaluate(
    *,
    model_path: Path,
    training_report_path: Path,
    registered_training_dir: Path,
    holdout_dir: Path,
    minimum_works: int,
    minimum_roc_auc: float,
    minimum_class_recall: float,
) -> dict[str, Any]:
    model_path = model_path.resolve(strict=True)
    training_report_path = training_report_path.resolve(strict=True)
    registered_training_dir = registered_training_dir.resolve(strict=True)
    holdout_dir = holdout_dir.resolve(strict=True)
    model = _load_json(model_path)
    training = _load_json(training_report_path)
    holdout_report_path = holdout_dir / "prepare-report.json"
    holdout = _load_json(holdout_report_path)
    if model.get("model_version") != REGISTERED_MODEL_VERSION:
        raise ValueError("holdout evaluation requires the registered-scan model")
    if model.get("feature_names") != list(ACCIDENTAL_PRESENCE_FEATURE_NAMES):
        raise ValueError("accidental model feature contract is stale")
    if training.get("model_version") != REGISTERED_MODEL_VERSION:
        raise ValueError("training report belongs to another accidental model")
    registered_audit = training.get("registered_scan")
    if not isinstance(registered_audit, dict):
        raise ValueError("training report lacks registered-scan provenance")
    training_prepare_report = registered_training_dir / "prepare-report.json"
    if (
        registered_audit.get("prepare_report_sha256")
        != _sha256_file(training_prepare_report)
    ):
        raise ValueError("training report is stale for its registered scan data")
    training_hashes = registered_audit.get("file_sha256")
    if not isinstance(training_hashes, dict):
        raise ValueError("training report lacks registered split hashes")
    for name in ("train.npz", "calibration.npz", "test.npz"):
        if training_hashes.get(name) != _sha256_file(registered_training_dir / name):
            raise ValueError(f"training report is stale for {name}")

    if holdout.get("name") != HOLDOUT_DATASET_NAME:
        raise ValueError("independent accidental holdout name is incompatible")
    if holdout.get("role") != HOLDOUT_DATASET_ROLE:
        raise ValueError("independent accidental holdout role is incompatible")
    if holdout.get("source_role") != HOLDOUT_SOURCE_ROLE:
        raise ValueError("independent accidental holdout source role is incompatible")
    if holdout.get("training_use_authorized") is not False:
        raise ValueError("independent accidental holdout is not forbidden to training")
    if holdout.get("holdout_used_for_training") is not False:
        raise ValueError("independent accidental holdout was marked as training data")
    intersections = holdout.get("work_intersections")
    if not isinstance(intersections, dict) or any(intersections.values()):
        raise ValueError("independent accidental holdout has internal work leakage")
    if any(int(holdout.get("samples_by_split", {}).get(split, -1)) != 0 for split in ("train", "calibration")):
        raise ValueError("independent accidental holdout contains non-test samples")
    works = int(holdout.get("works_by_split", {}).get("test", 0))
    if works < minimum_works:
        raise ValueError("independent accidental holdout has insufficient work coverage")

    training_works = _work_keys(registered_training_dir / "samples.jsonl")
    holdout_works = _work_keys(holdout_dir / "samples.jsonl")
    work_overlap = sorted(training_works & holdout_works)
    if work_overlap:
        raise ValueError("independent accidental holdout overlaps training works")
    if len(holdout_works) != works:
        raise ValueError("independent accidental holdout work count is stale")

    dataset = _load_dataset(holdout_dir / "test.npz")
    if len(dataset.labels) != int(
        holdout.get("samples_by_split", {}).get("test", -1)
    ):
        raise ValueError("independent accidental holdout sample count is stale")
    if int(np.sum(dataset.labels == 1)) != int(
        holdout.get("positive_samples_by_split", {}).get("test", -1)
    ):
        raise ValueError("independent accidental holdout class count is stale")
    probabilities = deployed_forest_probabilities(model, dataset.features)
    indices = np.arange(len(dataset.labels), dtype=np.int64)
    policy = _policy(
        dataset.labels,
        probabilities,
        indices,
        float(model.get("present_threshold", 1.0)),
        float(model.get("absent_threshold", 1.0)),
    )
    sample = _sample(dataset.labels, probabilities)
    passed = bool(
        policy["false_accepts"] == 0
        and min(policy["present_recall"], policy["absent_recall"])
        >= minimum_class_recall
        and sample["roc_auc"] >= minimum_roc_auc
    )
    return {
        "format": 1,
        "name": "scorescan-registered-scan-accidental-presence-holdout-evaluation-v1",
        "role": "independent_holdout_evaluation_only",
        "model_version": REGISTERED_MODEL_VERSION,
        "model_sha256": _sha256_file(model_path),
        "training_report_sha256": _sha256_file(training_report_path),
        "holdout_prepare_report_sha256": _sha256_file(holdout_report_path),
        "training_works": len(training_works),
        "holdout_works": len(holdout_works),
        "work_overlap": work_overlap,
        "samples": len(dataset.labels),
        "positive_samples": int(np.sum(dataset.labels == 1)),
        "negative_samples": int(np.sum(dataset.labels == 0)),
        "sample": sample,
        "policy": policy,
        "thresholds_frozen_before_holdout": True,
        "holdout_used_for_training": False,
        "acceptance": {
            "minimum_works": minimum_works,
            "minimum_roc_auc": minimum_roc_auc,
            "minimum_class_recall": minimum_class_recall,
            "zero_false_accepts_required": True,
            "passed": passed,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--registered-training-dir", type=Path, required=True)
    parser.add_argument("--holdout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-works", type=int, default=200)
    parser.add_argument("--minimum-roc-auc", type=float, default=0.94)
    parser.add_argument("--minimum-class-recall", type=float, default=0.30)
    args = parser.parse_args()
    if args.minimum_works < 200:
        raise ValueError("independent holdout must contain at least 200 works")
    if not 0.5 <= args.minimum_roc_auc <= 1.0:
        raise ValueError("minimum ROC AUC must be in [0.5, 1]")
    if not 0.0 < args.minimum_class_recall <= 1.0:
        raise ValueError("minimum class recall must be in (0, 1]")
    report = evaluate(
        model_path=args.model,
        training_report_path=args.training_report,
        registered_training_dir=args.registered_training_dir,
        holdout_dir=args.holdout_dir,
        minimum_works=args.minimum_works,
        minimum_roc_auc=args.minimum_roc_auc,
        minimum_class_recall=args.minimum_class_recall,
    )
    atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["acceptance"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
