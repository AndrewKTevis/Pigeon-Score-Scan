from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from train_patch_transaction_calibrator import (  # noqa: E402
    SAMPLES_PER_GROUP,
    SCENARIO_WEIGHTS,
    _split,
    build_dataset,
)
from scorescan.patch_transaction import FEATURE_NAMES  # noqa: E402


def test_patch_transaction_training_data_is_deterministic_and_grouped() -> None:
    left = build_dataset(seed=20260723, groups=128)
    right = build_dataset(seed=20260723, groups=128)

    assert np.array_equal(left.features, right.features)
    assert np.array_equal(left.labels, right.labels)
    assert np.array_equal(left.groups, right.groups)
    assert left.scenarios == right.scenarios
    assert left.features.shape == (128 * SAMPLES_PER_GROUP, len(FEATURE_NAMES))
    assert set(int(value) for value in left.groups) == set(range(128))

    partitions = _split(left.groups, seed=20260723)
    partition_groups = [set(int(value) for value in left.groups[index].tolist()) for index in partitions]
    assert set().union(*partition_groups) == set(range(128))
    for left_index, left_groups in enumerate(partition_groups):
        for right_groups in partition_groups[left_index + 1 :]:
            assert left_groups.isdisjoint(right_groups)


def test_patch_transaction_training_data_covers_compatible_and_adversarial_interactions() -> None:
    dataset = build_dataset(seed=20260723, groups=768)

    assert set(dataset.scenarios) == set(SCENARIO_WEIGHTS)
    assert set(int(value) for value in dataset.labels) == {0, 1}
    assert np.isfinite(dataset.features).all()
    assert np.all((dataset.features >= -1.0) & (dataset.features <= 1.0))
