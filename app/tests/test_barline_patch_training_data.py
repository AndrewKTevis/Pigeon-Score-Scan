from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from train_barline_patch_calibrator import (  # noqa: E402
    SAMPLES_PER_GROUP,
    _group_leakage_audit,
    _split,
    build_dataset,
)


def test_barline_patch_training_groups_are_related_and_split_without_leakage() -> None:
    dataset = build_dataset(seed=17, groups=120)
    assert len(dataset.labels) == 120 * SAMPLES_PER_GROUP
    assert set(np.bincount(dataset.groups)) == {SAMPLES_PER_GROUP}
    partitions = _split(dataset.groups, 17)
    audit = _group_leakage_audit(dataset, partitions)
    assert not audit["leakage_detected"]
    assert all(value == 0 for value in audit["pairwise_group_overlap"].values())
    first_group = dataset.features[dataset.groups == 0]
    assert len({tuple(row) for row in first_group}) == SAMPLES_PER_GROUP


def test_barline_patch_training_data_contains_positive_and_harmful_proposals() -> None:
    dataset = build_dataset(seed=29, groups=240)
    assert 0 < int(dataset.labels.sum()) < len(dataset.labels)
    scenarios = set(dataset.scenarios)
    assert "true-repeat-strong-support" in scenarios
    assert "common-mode-false-repeat" in scenarios
    assert "complex-navigation-conflict" in scenarios
