from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from train_selection_risk import (  # noqa: E402
    SCENARIO_WEIGHTS,
    _build_dataset,
)
from scorescan.selection_risk import FEATURE_NAMES  # noqa: E402


def test_selection_risk_training_data_is_deterministic_and_grouped() -> None:
    left = _build_dataset(seed=4312, groups=96, decisions_per_group=3)
    right = _build_dataset(seed=4312, groups=96, decisions_per_group=3)
    for first, second in zip(left[:3], right[:3]):
        assert np.array_equal(first, second)
    assert left[3] == right[3]
    features, labels, groups, metadata = left
    assert features.shape == (288, len(FEATURE_NAMES))
    assert labels.shape == groups.shape == (288,)
    assert set(int(value) for value in groups) == set(range(96))
    assert all(int(np.sum(groups == group)) == 3 for group in range(96))
    assert {str(item["scenario"]) for item in metadata} == set(SCENARIO_WEIGHTS)


def test_selection_risk_family_features_expose_duplicate_votes() -> None:
    features, _labels, _groups, metadata = _build_dataset(
        seed=8121,
        groups=192,
        decisions_per_group=3,
    )
    raw_support = FEATURE_NAMES.index("exact_support_ratio")
    family_support = FEATURE_NAMES.index("exact_family_support_ratio")
    redundancy = FEATURE_NAMES.index("candidate_family_redundancy")
    indices = np.asarray(
        [
            index
            for index, item in enumerate(metadata)
            if item["scenario"] == "exact-family-redundant-trap"
        ],
        dtype=np.int64,
    )
    assert indices.size > 0
    assert np.all(features[indices, raw_support] > features[indices, family_support])
    assert np.all(features[indices, redundancy] > 0.0)


def test_selection_risk_localized_scenarios_are_bounded() -> None:
    features, labels, _groups, metadata = _build_dataset(
        seed=9017,
        groups=640,
        decisions_per_group=3,
    )
    family = FEATURE_NAMES.index("eligible_family_count_scaled")
    for scenario in ("localized-clear-gain", "localized-confident-trap", "localized-partial-trap"):
        indices = np.asarray(
            [index for index, item in enumerate(metadata) if item["scenario"] == scenario],
            dtype=np.int64,
        )
        assert indices.size > 0
        assert np.all(features[indices, family] == 1.0)
    traps = np.asarray(
        [
            index
            for index, item in enumerate(metadata)
            if item["scenario"] in {"localized-confident-trap", "localized-partial-trap"}
        ],
        dtype=np.int64,
    )
    assert np.all(labels[traps] == 0)
