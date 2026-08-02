from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from measure_training_data import build_dataset  # noqa: E402
from scorescan.measure_calibration import FEATURE_NAMES, LEGACY_FEATURE_NAMES  # noqa: E402


def test_measure_training_dataset_is_deterministic_and_grouped() -> None:
    first = build_dataset(seed=731, groups=33)
    second = build_dataset(seed=731, groups=33)

    assert np.array_equal(first.features, second.features)
    assert np.array_equal(first.legacy_features, second.legacy_features)
    assert np.array_equal(first.labels, second.labels)
    assert np.array_equal(first.groups, second.groups)
    assert first.decision_groups == second.decision_groups
    assert first.scenarios == second.scenarios
    assert first.features.shape[1] == len(FEATURE_NAMES)
    assert first.legacy_features.shape[1] == len(LEGACY_FEATURE_NAMES)

    for group_id, decision in enumerate(first.decision_groups):
        assert decision
        assert all(first.groups[index] == group_id for index in decision)
        assert any(first.labels[index] == 1 for index in decision)


def test_measure_training_dataset_covers_boundary_and_integrity_scenarios() -> None:
    dataset = build_dataset(seed=991, groups=22)
    assert {
        "pickup_boundary",
        "final_boundary",
        "accidental_integrity",
        "chord_integrity",
        "duplicate_integrity",
        "strict_majority",
        "complete_agreement",
    }.issubset(set(dataset.scenarios))
