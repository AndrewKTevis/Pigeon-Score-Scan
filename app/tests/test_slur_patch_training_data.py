from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from train_slur_patch_calibrator import (  # noqa: E402
    SAMPLES_PER_GROUP,
    SCENARIO_WEIGHTS,
    _dataset_fingerprint,
    _split,
    build_dataset,
)
from scorescan.slur_consensus import FEATURE_NAMES  # noqa: E402


def test_slur_patch_training_data_is_deterministic_and_grouped() -> None:
    left = build_dataset(seed=20260720, groups=180)
    right = build_dataset(seed=20260720, groups=180)

    assert np.array_equal(left.features, right.features)
    assert np.array_equal(left.labels, right.labels)
    assert np.array_equal(left.groups, right.groups)
    assert left.scenarios == right.scenarios
    assert _dataset_fingerprint(left) == _dataset_fingerprint(right)
    assert left.features.shape == (180 * SAMPLES_PER_GROUP, len(FEATURE_NAMES))
    assert set(int(value) for value in left.groups) == set(range(180))
    assert all(np.sum(left.groups == group) == SAMPLES_PER_GROUP for group in range(180))

    partitions = _split(left.groups, seed=20260720)
    partition_groups = [set(int(value) for value in left.groups[index].tolist()) for index in partitions]
    assert set().union(*partition_groups) == set(range(180))
    for left_index, left_groups in enumerate(partition_groups):
        for right_groups in partition_groups[left_index + 1 :]:
            assert left_groups.isdisjoint(right_groups)


def test_slur_patch_training_data_covers_adversarial_scenarios() -> None:
    dataset = build_dataset(seed=20260720, groups=1600)

    assert set(dataset.scenarios) == set(SCENARIO_WEIGHTS)
    assert set(int(value) for value in dataset.labels) == {0, 1}
    assert np.isfinite(dataset.features).all()
    assert np.all((dataset.features >= -1.0) & (dataset.features <= 1.0))
    confirmation = build_dataset(seed=20260722, groups=1600)
    assert _dataset_fingerprint(dataset) != _dataset_fingerprint(confirmation)
