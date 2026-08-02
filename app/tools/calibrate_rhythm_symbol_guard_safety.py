from __future__ import annotations

"""Select a conservative rhythm-symbol threshold on a dedicated safety set."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rhythm_symbol_training_data import build_rendered_rhythm_symbol_dataset  # noqa: E402
from scorescan.rhythm_symbol_guard import RHYTHM_SYMBOL_FEATURE_NAMES  # noqa: E402
from scorescan.util import atomic_write_json  # noqa: E402
from train_rhythm_symbol_guard import _policy, _sample  # noqa: E402
from tree_export import deployed_forest_probabilities  # noqa: E402

EXPECTED_MODEL_VERSION = "scorescan-rhythm-symbol-forest-1"
THRESHOLD_GRID = (0.985, 0.9875, 0.988, 0.9885)
MINIMUM_POSITIVE_RECALL = 0.45


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "src" / "scorescan" / "resources" / "rhythm_symbol_guard.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent / "training" / "rhythm_symbol_guard_safety_calibration_v1.json",
    )
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--groups", type=int, default=1200)
    args = parser.parse_args()

    payload = json.loads(args.model.read_text(encoding="utf-8"))
    if payload.get("model_version") != EXPECTED_MODEL_VERSION:
        raise RuntimeError("unexpected rhythm symbol guard model version")
    if tuple(payload.get("feature_names", ())) != RHYTHM_SYMBOL_FEATURE_NAMES:
        raise RuntimeError("rhythm symbol guard feature schema mismatch")
    initial_threshold = float(payload.get("auto_patch_threshold", 1.0))

    dataset = build_rendered_rhythm_symbol_dataset(args.seed, args.groups)
    probabilities = deployed_forest_probabilities(payload, dataset.features)
    indices = np.arange(len(dataset.labels), dtype=np.int64)
    candidates = sorted(
        set([initial_threshold, *[value for value in THRESHOLD_GRID if value >= initial_threshold]])
    )
    rows = [
        _policy(dataset.labels, probabilities, indices, threshold)
        for threshold in candidates
    ]
    valid = [
        row
        for row in rows
        if int(row["false_accepts"]) == 0
        and float(row["positive_recall"]) >= MINIMUM_POSITIVE_RECALL
    ]
    if not valid:
        raise RuntimeError("no rhythm symbol safety threshold met precision and coverage gates")
    selected = max(
        valid,
        key=lambda row: (
            float(row["positive_recall"]),
            float(row["coverage"]),
            float(row["threshold"]),
        ),
    )
    selected_threshold = float(selected["threshold"])
    payload.update(
        {
            "pre_safety_threshold": initial_threshold,
            "auto_patch_threshold": selected_threshold,
            "reverse_probability_ceiling": 1.0 - selected_threshold,
            "safety_calibration_seed": args.seed,
            "safety_calibration_groups": args.groups,
            "safety_calibration_false_accepts": 0,
            "safety_calibration_positive_recall": float(selected["positive_recall"]),
        }
    )
    atomic_write_json(args.model, payload)
    report = {
        "model_version": EXPECTED_MODEL_VERSION,
        "scope": "dedicated rendered threshold safety calibration; excluded from training and final independent audit",
        "seed": args.seed,
        "groups": args.groups,
        "samples": len(dataset.labels),
        "initial_threshold": initial_threshold,
        "candidate_policies": rows,
        "selected_policy": selected,
        "sample": _sample(dataset.labels, probabilities),
        "publication_gates": {
            "zero_error_accepts": True,
            "minimum_positive_recall": MINIMUM_POSITIVE_RECALL,
        },
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
