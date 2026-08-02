from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from measure_count_training_data import KINDS, build_dataset  # noqa: E402
from scorescan.measure_count_resolver import FEATURE_NAMES, LEGACY_FEATURE_NAMES  # noqa: E402


def test_measure_count_training_data_is_deterministic_across_workers() -> None:
    groups = len(KINDS) * 4
    serial = build_dataset(seed=8107, groups=groups, workers=1)
    parallel = build_dataset(seed=8107, groups=groups, workers=2)

    assert np.array_equal(serial.features, parallel.features)
    assert np.array_equal(serial.legacy_features, parallel.legacy_features)
    assert np.array_equal(serial.labels, parallel.labels)
    assert np.array_equal(serial.groups, parallel.groups)
    assert np.array_equal(serial.option_counts, parallel.option_counts)
    assert np.array_equal(serial.truths, parallel.truths)
    assert np.array_equal(serial.kinds, parallel.kinds)
    assert np.array_equal(serial.deterministic_counts, parallel.deterministic_counts)
    assert serial.decision_groups == parallel.decision_groups
    assert serial.features.shape[1] == len(FEATURE_NAMES)
    assert serial.legacy_features.shape[1] == len(LEGACY_FEATURE_NAMES)


def test_measure_count_training_groups_are_complete_and_cover_scenarios() -> None:
    dataset = build_dataset(seed=9913, groups=len(KINDS) * 4, workers=1)
    observed_kinds: set[str] = set()
    for group, decision in enumerate(dataset.decision_groups):
        assert len(decision) >= 2
        assert all(dataset.groups[index] == group for index in decision)
        assert sum(int(dataset.labels[index]) for index in decision) == 1
        assert int(dataset.truths[decision[0]]) in {
            int(dataset.option_counts[index]) for index in decision
        }
        observed_kinds.add(str(dataset.kinds[decision[0]]))
    assert observed_kinds == set(KINDS)


def test_family_balancing_features_detect_duplicate_support() -> None:
    dataset = build_dataset(seed=1201, groups=len(KINDS) * 4, workers=1)
    duplicate_index = FEATURE_NAMES.index("duplicate_support_ratio")
    family_index = FEATURE_NAMES.index("family_balanced_support_share")
    raw_index = FEATURE_NAMES.index("support_share")
    matching = np.flatnonzero(dataset.kinds == "single-family-duplicate-trap")
    assert matching.size > 0
    assert np.max(dataset.features[matching, duplicate_index]) > 0.0
    # At least one correlated wrong option has more raw support than its family-balanced vote.
    assert np.any(dataset.features[matching, raw_index] > dataset.features[matching, family_index])


def test_incomplete_family_features_cover_invalid_sibling_trap() -> None:
    dataset = build_dataset(seed=4409, groups=len(KINDS) * 4, workers=1)
    incomplete_index = FEATURE_NAMES.index("incomplete_family_share")
    complete_index = FEATURE_NAMES.index("complete_family_share")
    matching = np.flatnonzero(dataset.kinds == "invalid-sibling-family-trap")
    assert matching.size > 0
    assert np.all(dataset.features[matching, incomplete_index] > 0.0)
    assert np.all(dataset.features[matching, complete_index] < 1.0)
