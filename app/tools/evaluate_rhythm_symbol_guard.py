from __future__ import annotations

"""Independent rendered audit for the deployed rhythm-symbol transaction guard."""

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
from train_rhythm_symbol_guard import _policy, _sample, _scenario_metrics  # noqa: E402
from tree_export import deployed_forest_probabilities  # noqa: E402

EXPECTED_MODEL_VERSION = "scorescan-rhythm-symbol-forest-1"
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
        default=ROOT.parent / "training" / "rhythm_symbol_guard_independent_audit_v1.json",
    )
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--groups", type=int, default=1200)
    args = parser.parse_args()

    payload = json.loads(args.model.read_text(encoding="utf-8"))
    if payload.get("model_version") != EXPECTED_MODEL_VERSION:
        raise RuntimeError("unexpected rhythm symbol guard model version")
    if tuple(payload.get("feature_names", ())) != RHYTHM_SYMBOL_FEATURE_NAMES:
        raise RuntimeError("rhythm symbol guard feature schema mismatch")
    threshold = float(payload.get("auto_patch_threshold", 1.0))

    dataset = build_rendered_rhythm_symbol_dataset(args.seed, args.groups)
    probabilities = deployed_forest_probabilities(payload, dataset.features)
    indices = np.arange(len(dataset.labels), dtype=np.int64)
    policy = _policy(dataset.labels, probabilities, indices, threshold)
    if int(policy["false_accepts"]):
        raise RuntimeError(
            f"independent rhythm symbol audit observed {policy['false_accepts']} error accepts"
        )
    if float(policy["positive_recall"]) < MINIMUM_POSITIVE_RECALL:
        raise RuntimeError(
            "independent rhythm symbol audit coverage below publication floor: "
            f"{policy['positive_recall']:.4f}"
        )

    report = {
        "model_version": EXPECTED_MODEL_VERSION,
        "scope": "independent rendered rhythm-transaction audit; not real-scan or end-to-end OMR accuracy",
        "seed": args.seed,
        "groups": args.groups,
        "samples": len(dataset.labels),
        "threshold": threshold,
        "sample": _sample(dataset.labels, probabilities),
        "policy": policy,
        "scenarios": _scenario_metrics(
            dataset.scenarios, indices, dataset.labels, probabilities, threshold
        ),
        "publication_gates": {
            "zero_error_accepts": True,
            "minimum_positive_recall": MINIMUM_POSITIVE_RECALL,
        },
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
