from __future__ import annotations

"""Generate one deterministic rendered accidental-presence split in a fresh process."""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from accidental_presence_training_data import build_accidental_presence_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--groups", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = build_accidental_presence_dataset(args.seed, args.groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=dataset.features,
        labels=dataset.labels,
        groups=dataset.groups,
        symbols=np.asarray(dataset.symbols, dtype="U32"),
    )


if __name__ == "__main__":
    main()
