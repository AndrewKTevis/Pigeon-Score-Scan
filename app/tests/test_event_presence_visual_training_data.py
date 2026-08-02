from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from event_presence_visual_training_data import (  # noqa: E402
    build_event_presence_visual_dataset,
)
from scorescan.event_presence_visual_guard import (  # noqa: E402
    EVENT_PRESENCE_VISUAL_FEATURE_NAMES,
)


def test_event_presence_visual_training_data_is_grouped_balanced_and_reproducible() -> None:
    first = build_event_presence_visual_dataset(seed=20261132, groups=10)
    second = build_event_presence_visual_dataset(seed=20261132, groups=10)
    assert first.features.shape == (40, len(EVENT_PRESENCE_VISUAL_FEATURE_NAMES))
    assert np.array_equal(first.labels, second.labels)
    assert np.array_equal(first.groups, second.groups)
    assert np.array_equal(first.features, second.features)
    assert first.scenarios == second.scenarios
    assert first.operations == second.operations
    assert first.event_kinds == second.event_kinds
    for group in range(10):
        indices = np.flatnonzero(first.groups == group)
        assert len(indices) == 4
        assert sorted(first.labels[indices].tolist()) == [0, 0, 1, 1]
        assert sorted(np.asarray(first.operations)[indices].tolist()) == [
            "delete",
            "delete",
            "insert",
            "insert",
        ]


def test_event_presence_training_balances_reviewed_note_and_rest_targets() -> None:
    dataset = build_event_presence_visual_dataset(seed=20261242, groups=200)
    operations = np.asarray(dataset.operations)
    kinds = np.asarray(dataset.event_kinds)
    insertion = operations == "insert"
    note_count = int(np.sum(insertion & (kinds == "note")))
    rest_count = int(np.sum(insertion & (kinds == "rest")))
    assert note_count + rest_count == 400
    assert abs(note_count - rest_count) <= 80
