from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from event_kind_visual_training_data import build_event_kind_visual_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--groups", type=int, required=True)
    args = parser.parse_args()
    dataset = build_event_kind_visual_dataset(seed=args.seed, groups=args.groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=dataset.features,
        labels=dataset.labels,
        groups=dataset.groups,
        scenarios=np.asarray(dataset.scenarios),
        target_kinds=np.asarray(dataset.target_kinds),
    )


if __name__ == "__main__":
    main()
