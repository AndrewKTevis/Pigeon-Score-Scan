from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from train_cross_tie_patch_calibrator import (  # noqa: E402
    FEATURE_NAMES,
    SAMPLES_PER_GROUP,
    SCENARIO_WEIGHTS,
    _dataset_fingerprint,
    _group_leakage_audit,
    _split,
    _zero_false_accept_threshold,
    build_dataset,
)


def test_cross_tie_training_data_is_deterministic_grouped_and_seed_separated() -> None:
    left = build_dataset(seed=20260722, groups=180)
    right = build_dataset(seed=20260722, groups=180)
    assert np.array_equal(left.features, right.features)
    assert np.array_equal(left.labels, right.labels)
    assert np.array_equal(left.groups, right.groups)
    assert left.scenarios == right.scenarios
    assert left.features.shape == (180 * SAMPLES_PER_GROUP, len(FEATURE_NAMES))
    assert len(set(int(value) for value in left.groups)) == 180
    assert all(np.sum(left.groups == group) == SAMPLES_PER_GROUP for group in range(180))
    partitions = _split(left.groups, 20260722)
    audit = _group_leakage_audit(left, partitions)
    assert not audit["leakage_detected"]
    assert not any(audit["pairwise_group_overlap"].values())
    confirmation = build_dataset(seed=20260724, groups=180)
    assert _dataset_fingerprint(left) != _dataset_fingerprint(confirmation)


def test_cross_tie_training_data_covers_adversarial_scenarios() -> None:
    dataset = build_dataset(seed=20260722, groups=1800)
    assert set(dataset.scenarios) == set(SCENARIO_WEIGHTS)
    assert set(int(value) for value in dataset.labels) == {0, 1}
    assert np.isfinite(dataset.features).all()
    assert np.all((dataset.features >= -1.0) & (dataset.features <= 1.0))


def test_cross_tie_safety_threshold_is_strictly_above_all_negative_probabilities() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.20, 0.90, 0.91, 0.99], dtype=np.float64)
    threshold, metrics = _zero_false_accept_threshold(labels, probabilities, 0.80)
    assert threshold > 0.90
    assert metrics["false_accepts"] == 0
    assert metrics["true_accepts"] == 2
    assert metrics["precision"] == 1.0
