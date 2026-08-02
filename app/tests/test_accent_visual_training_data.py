from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from accent_visual_training_data import build_accent_visual_dataset  # noqa: E402
from scorescan.accent_visual_guard import ACCENT_VISUAL_FEATURE_NAMES  # noqa: E402


def test_accent_visual_dataset_is_grouped_balanced_and_reproducible() -> None:
    first = build_accent_visual_dataset(seed=20261411, groups=6, workers=1)
    second = build_accent_visual_dataset(seed=20261411, groups=6, workers=1)
    assert first.features.shape == (48, len(ACCENT_VISUAL_FEATURE_NAMES))
    assert np.array_equal(first.features, second.features)
    assert np.array_equal(first.labels, second.labels)
    assert np.array_equal(first.groups, second.groups)
    assert first.scenarios == second.scenarios
    for group in sorted(set(first.groups.tolist())):
        indices = np.flatnonzero(first.groups == group)
        assert len(indices) == 8
        assert int(np.sum(first.labels[indices])) == 1
        assert set(first.scenarios[index] for index in indices) == {
            "none",
            "target_accent",
            "tenuto",
            "staccato",
            "dust",
            "short_line",
            "single_diagonal",
            "duration_dot",
        }
