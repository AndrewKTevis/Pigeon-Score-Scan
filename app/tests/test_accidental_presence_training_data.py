from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from accidental_presence_training_data import build_accidental_presence_dataset  # noqa: E402
from scorescan.accidental_presence_guard import ACCIDENTAL_PRESENCE_FEATURE_NAMES  # noqa: E402


def test_accidental_presence_training_groups_are_deterministic_and_isolated() -> None:
    left = build_accidental_presence_dataset(seed=20260819, groups=12)
    right = build_accidental_presence_dataset(seed=20260819, groups=12)
    assert np.array_equal(left.features, right.features)
    assert np.array_equal(left.labels, right.labels)
    assert np.array_equal(left.groups, right.groups)
    assert left.symbols == right.symbols
    assert left.features.shape == (24, len(ACCIDENTAL_PRESENCE_FEATURE_NAMES))
    assert np.isfinite(left.features).all()
    for group in range(12):
        rows = np.flatnonzero(left.groups == group)
        assert len(rows) == 2
        assert set(int(value) for value in left.labels[rows]) == {0, 1}
