from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.layout import analyze_layout  # noqa: E402
from scorescan.util import atomic_write_json, sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--ground-truth-measures", type=int, required=True)
    parser.add_argument("--baseline-estimate", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    layout = analyze_layout(args.image)
    estimate = sum(system.measure_count for system in layout.systems)
    report = {
        "source_file": args.image.name,
        "source_sha256": sha256_file(args.image),
        "ground_truth_measures": args.ground_truth_measures,
        "baseline_estimate": args.baseline_estimate,
        "scorescan_0_15_estimate": estimate,
        "baseline_absolute_error": abs(args.baseline_estimate - args.ground_truth_measures),
        "new_absolute_error": abs(estimate - args.ground_truth_measures),
        "system_count": len(layout.systems),
        "system_measure_counts": [system.measure_count for system in layout.systems],
        "removed_false_split_regression": estimate == args.ground_truth_measures,
        "scope": "one supplied printed scan page; layout boundary regression only, not end-to-end OMR accuracy",
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
