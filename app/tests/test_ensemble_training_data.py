from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from ensemble_training_data import SCENARIOS, build_dataset  # noqa: E402


def test_ensemble_dataset_is_independent_of_worker_count() -> None:
    serial = build_dataset(seed=12345, groups=13, workers=1)
    parallel = build_dataset(seed=12345, groups=13, workers=2)
    assert np.array_equal(serial.features, parallel.features)
    assert np.array_equal(serial.labels, parallel.labels)
    assert np.array_equal(serial.groups, parallel.groups)
    assert serial.decision_groups == parallel.decision_groups
    assert serial.scenarios == parallel.scenarios


def test_ensemble_dataset_covers_localized_family_scenarios() -> None:
    dataset = build_dataset(seed=2030, groups=len(SCENARIOS), workers=1)
    scenario_to_group = {scenario: group for group, scenario in enumerate(dataset.scenarios)}
    for scenario in (
        "localized-rescue",
        "localized-isolation-trap",
        "localized-partial-trap",
    ):
        decision = dataset.decision_groups[scenario_to_group[scenario]]
        assert 5 <= len(decision) <= 8
    rescue = dataset.decision_groups[scenario_to_group["localized-rescue"]]
    assert int(np.sum(dataset.labels[list(rescue)])) == 1
    partial = dataset.decision_groups[scenario_to_group["localized-partial-trap"]]
    page_valid = 2
    assert any(dataset.features[index, page_valid] == 0.0 for index in partial)
