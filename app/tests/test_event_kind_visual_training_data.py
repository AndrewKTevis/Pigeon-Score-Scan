from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from event_kind_visual_training_data import build_event_kind_visual_dataset  # noqa: E402
from scorescan.event_kind_visual_guard import EVENT_KIND_VISUAL_FEATURE_NAMES  # noqa: E402


def test_event_kind_visual_training_data_is_grouped_balanced_and_reproducible() -> None:
    first = build_event_kind_visual_dataset(seed=20261102, groups=12)
    second = build_event_kind_visual_dataset(seed=20261102, groups=12)
    assert first.features.shape == (24, len(EVENT_KIND_VISUAL_FEATURE_NAMES))
    assert np.array_equal(first.labels, second.labels)
    assert np.array_equal(first.groups, second.groups)
    assert np.array_equal(first.features, second.features)
    assert first.scenarios == second.scenarios
    assert first.target_kinds == second.target_kinds
    for group in range(12):
        indices = np.flatnonzero(first.groups == group)
        assert len(indices) == 2
        assert sorted(first.labels[indices].tolist()) == [0, 1]
