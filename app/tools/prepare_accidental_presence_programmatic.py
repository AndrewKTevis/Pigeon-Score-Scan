from __future__ import annotations

"""Prepare deterministic rendered accidental-presence regression datasets."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from accidental_presence_training_data import (  # noqa: E402
    build_accidental_presence_dataset,
)
from scorescan.accidental_presence_guard import (  # noqa: E402
    ACCIDENTAL_PRESENCE_FEATURE_NAMES,
)
from scorescan.util import atomic_write_json  # noqa: E402
from train_accidental_presence_guard import _sha256_file  # noqa: E402

SPLIT_CONFIG = {
    "train": {"seed": 20260819, "groups": 800},
    "safety": {"seed": 20260920, "groups": 900},
    "independent": {"seed": 20261030, "groups": 900},
}


def _write_dataset(path: Path, *, seed: int, groups: int) -> dict[str, object]:
    dataset = build_accidental_presence_dataset(seed=seed, groups=groups)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            features=dataset.features,
            labels=dataset.labels,
            groups=dataset.groups,
            symbols=np.asarray(dataset.symbols, dtype="U32"),
        )
    temporary.replace(path)
    return {
        "seed": seed,
        "groups": groups,
        "samples": len(dataset.labels),
        "positive_samples": int(np.sum(dataset.labels == 1)),
        "sha256": _sha256_file(path),
    }


def prepare(output_dir: Path, split_config: dict[str, dict[str, int]]) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        name: _write_dataset(
            output_dir / f"{name}.npz",
            seed=int(config["seed"]),
            groups=int(config["groups"]),
        )
        for name, config in split_config.items()
    }
    report = {
        "format": 1,
        "name": "scorescan-programmatic-accidental-presence-v2",
        "role": "training_safety_and_regression_only",
        "feature_names": list(ACCIDENTAL_PRESENCE_FEATURE_NAMES),
        "splits": splits,
        "registered_scan_evidence": False,
        "independent_real_scan_holdout": False,
        "end_to_end_accuracy_claim": False,
    }
    atomic_write_json(output_dir / "prepare-report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.output_dir, SPLIT_CONFIG)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
