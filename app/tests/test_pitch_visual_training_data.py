from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from pitch_visual_training_data import (  # noqa: E402
    PITCH_VISUAL_KINDS,
    build_rendered_pitch_dataset,
)
from scorescan.pitch_consensus import FEATURE_NAMES  # noqa: E402


def test_rendered_pitch_transactions_are_deterministic_and_group_bound() -> None:
    left = build_rendered_pitch_dataset(seed=20260727, groups=24)
    right = build_rendered_pitch_dataset(seed=20260727, groups=24)

    assert np.array_equal(left.features, right.features)
    assert np.array_equal(left.labels, right.labels)
    assert np.array_equal(left.groups, right.groups)
    assert left.scenarios == right.scenarios
    assert left.features.shape == (48, len(FEATURE_NAMES))
    assert set(int(value) for value in left.labels) == {0, 1}

    for group in range(24):
        rows = np.flatnonzero(left.groups == group)
        assert len(rows) == 2
        assert set(int(value) for value in left.labels[rows]) == {0, 1}
        correction = rows[int(np.argmax(left.labels[rows]))]
        regression = rows[int(np.argmin(left.labels[rows]))]
        # All family/candidate evidence is identical inside the pair.  The direct
        # notehead improvement reverses sign, and absolute template/proposal gaps swap.
        assert np.array_equal(left.features[correction, :28], left.features[regression, :28])
        assert np.allclose(left.features[correction, 28:35], -left.features[regression, 28:35])
        assert np.allclose(left.features[correction, 35:42], left.features[regression, 42:49])
        assert np.allclose(left.features[correction, 42:49], left.features[regression, 35:42])


def test_rendered_pitch_transactions_cover_all_error_families() -> None:
    dataset = build_rendered_pitch_dataset(seed=20260727, groups=40)
    observed = {
        kind
        for kind in PITCH_VISUAL_KINDS
        if any(f"rendered-{kind}-" in scenario for scenario in dataset.scenarios)
    }
    assert observed == set(PITCH_VISUAL_KINDS)
    assert np.isfinite(dataset.features).all()
    assert np.all((dataset.features >= -1.0) & (dataset.features <= 1.0))
