from __future__ import annotations

"""Independent rendered audit for pitch-patch visual transaction safety.

Fresh source crops are generated with a seed not used for training.  Every crop yields a
paired correction and inverse regression in one group.  This audit compares the global
pitch calibrator, its prior version, the direct visual guard, and their production AND
gate.  It does not estimate end-to-end OMR accuracy.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pitch_visual_training_data import PITCH_VISUAL_KINDS, build_rendered_pitch_dataset  # noqa: E402
from scorescan.pitch_consensus import (  # noqa: E402
    FEATURE_NAMES,
    PITCH_VISUAL_FEATURE_INDICES,
)
from scorescan.util import atomic_write_json  # noqa: E402
from tree_export import deployed_forest_probabilities  # noqa: E402


def _metrics(labels: np.ndarray, accepted: np.ndarray) -> dict[str, float | int]:
    true_accepts = int(np.sum(accepted & (labels == 1)))
    false_accepts = int(np.sum(accepted & (labels == 0)))
    return {
        "samples": int(len(labels)),
        "accepted": int(np.sum(accepted)),
        "true_accepts": true_accepts,
        "false_accepts": false_accepts,
        "precision": true_accepts / max(true_accepts + false_accepts, 1),
        "positive_recall": true_accepts / max(int(np.sum(labels == 1)), 1),
    }


def _load(path: Path) -> tuple[dict[str, object], float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, float(payload.get("auto_patch_threshold", 1.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--pitch-model",
        type=Path,
        default=ROOT / "src" / "scorescan" / "resources" / "pitch_patch_calibrator.json",
    )
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=ROOT.parent / "training" / "baselines" / "pitch_patch_calibrator_v2.json",
    )
    parser.add_argument(
        "--visual-model",
        type=Path,
        default=ROOT / "src" / "scorescan" / "resources" / "pitch_visual_guard.json",
    )
    args = parser.parse_args()

    dataset = build_rendered_pitch_dataset(args.seed, args.groups)
    labels = dataset.labels
    pitch_payload, pitch_threshold = _load(args.pitch_model)
    baseline_payload, baseline_threshold = _load(args.baseline_model)
    visual_payload, visual_threshold = _load(args.visual_model)

    pitch_probabilities = deployed_forest_probabilities(pitch_payload, dataset.features)
    baseline_count = len(baseline_payload.get("feature_names", ()))
    if baseline_count > len(FEATURE_NAMES):
        raise ValueError("baseline pitch model has an unsupported feature schema")
    baseline_probabilities = deployed_forest_probabilities(
        baseline_payload, dataset.features[:, :baseline_count]
    )
    visual_probabilities = deployed_forest_probabilities(
        visual_payload, dataset.features[:, PITCH_VISUAL_FEATURE_INDICES]
    )

    pitch_accepted = pitch_probabilities >= pitch_threshold
    baseline_accepted = baseline_probabilities >= baseline_threshold
    visual_accepted = visual_probabilities >= visual_threshold
    near_index = FEATURE_NAMES.index("notehead_near_cell_improvement")
    severe_index = FEATURE_NAMES.index("notehead_severe_vertical_improvement")
    hard_conflict = (
        (dataset.features[:, near_index] <= -0.12)
        & (dataset.features[:, severe_index] <= -0.08)
    )
    production_accepted = pitch_accepted & visual_accepted & ~hard_conflict

    scenario_values = np.asarray(dataset.scenarios)
    by_kind: dict[str, object] = {}
    for kind in PITCH_VISUAL_KINDS:
        selected = np.asarray([f"rendered-{kind}-" in value for value in scenario_values])
        by_kind[kind] = {
            "pitch_model": _metrics(labels[selected], pitch_accepted[selected]),
            "v2_same_test": _metrics(labels[selected], baseline_accepted[selected]),
            "visual_guard": _metrics(labels[selected], visual_accepted[selected]),
            "production_and_gate": _metrics(labels[selected], production_accepted[selected]),
            "deterministic_conflicts": int(np.sum(hard_conflict[selected])),
        }

    report = {
        "format": 2,
        "seed": args.seed,
        "rendered_groups": args.groups,
        "samples": len(labels),
        "scope": "independent rendered pitch transaction audit; not end-to-end OMR accuracy",
        "pitch_model_version": str(pitch_payload.get("model_version", "unknown")),
        "pitch_threshold": pitch_threshold,
        "baseline_model_version": str(baseline_payload.get("model_version", "unknown")),
        "baseline_threshold": baseline_threshold,
        "visual_model_version": str(visual_payload.get("model_version", "unknown")),
        "visual_threshold": visual_threshold,
        "pitch_model": _metrics(labels, pitch_accepted),
        "v2_same_test": _metrics(labels, baseline_accepted),
        "visual_guard": _metrics(labels, visual_accepted),
        "production_and_gate": _metrics(labels, production_accepted),
        "by_kind": by_kind,
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
