from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from accent_visual_training_data import build_accent_visual_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--groups", type=int, required=True)
    parser.add_argument("--group-offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    dataset = build_accent_visual_dataset(
        args.seed,
        args.groups,
        group_offset=args.group_offset,
        workers=args.workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=dataset.features,
        labels=dataset.labels,
        groups=dataset.groups,
        scenarios=np.asarray(dataset.scenarios),
    )
    print(
        f"accent visual dataset: {dataset.features.shape[0]} samples, "
        f"{dataset.features.shape[1]} features, {len(set(dataset.groups.tolist()))} groups"
    )


if __name__ == "__main__":
    main()
