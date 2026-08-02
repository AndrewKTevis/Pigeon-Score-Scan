from __future__ import annotations

"""Generate one deterministic rendered tie-visual split in a fresh process."""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tie_visual_training_data import (  # noqa: E402
    build_tie_slur_ambiguity_dataset,
    build_tie_visual_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--groups", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ambiguity", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    builder = build_tie_slur_ambiguity_dataset if args.ambiguity else build_tie_visual_dataset
    dataset = builder(args.seed, args.groups, workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=dataset.features,
        labels=dataset.labels,
        groups=dataset.groups,
        scenarios=np.asarray(dataset.scenarios, dtype="U32"),
    )


if __name__ == "__main__":
    main()
