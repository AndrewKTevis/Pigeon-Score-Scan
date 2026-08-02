from __future__ import annotations

"""Evaluate the exact deployed direction decoder under deterministic OCR noise.

This intentionally reports deployment behaviour separately from classifier training
metrics.  It measures a single synthetic corrupted observation for every maintained
phrase and records the precision/coverage of unattended write-back.
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.direction_model import DirectionCorrector, normalize_direction  # noqa: E402
from train_direction_model import build_corpus, corrupt  # noqa: E402


def evaluate(model_path: Path, seed: int) -> dict[str, object]:
    rng = random.Random(seed)
    corrector = DirectionCorrector(model_path)
    rows: list[tuple[bool, bool, str, str, object]] = []
    for expected, _frequency in build_corpus():
        observed = corrupt(expected, rng, severity=rng.choice([1, 1, 2]))
        suggestion = corrector.suggest(observed)
        correct = normalize_direction(suggestion.text) == normalize_direction(expected)
        rows.append((correct, corrector.should_autocorrect(suggestion), expected, observed, suggestion))

    automatic = [row for row in rows if row[1]]
    errors = [
        {
            "expected": expected,
            "observed": observed,
            "suggested": suggestion.text,
            "probability": suggestion.probability,
            "margin": suggestion.margin,
            "method": suggestion.method,
            "autocorrect_safe": suggestion.autocorrect_safe,
            "edit_ratio": suggestion.edit_ratio,
        }
        for correct, _auto, expected, observed, suggestion in automatic
        if not correct
    ]
    sample_count = len(rows)
    try:
        model_label = str(model_path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        model_label = model_path.name
    return {
        "evaluation": "direction deployment stress test",
        "model_file": model_label,
        "model_version": corrector.model_version,
        "seed": seed,
        "samples": sample_count,
        "corruption": "one deterministic 1-2 operation synthetic OCR corruption per maintained phrase",
        "top1_accuracy": sum(row[0] for row in rows) / max(sample_count, 1),
        "autocorrect_count": len(automatic),
        "autocorrect_coverage": len(automatic) / max(sample_count, 1),
        "autocorrect_precision": sum(row[0] for row in automatic) / max(len(automatic), 1),
        "autocorrect_errors": errors,
        "notes": "This evaluates the deployed hybrid decoder on synthetic corruptions. It is not an end-to-end score OCR metric.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "src/scorescan/resources/direction_model.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent / "training/direction_deployment_eval_v5.json",
    )
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()

    result = evaluate(args.model, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
