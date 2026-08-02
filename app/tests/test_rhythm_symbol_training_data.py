from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from rhythm_symbol_training_data import build_rendered_rhythm_symbol_dataset  # noqa: E402
from scorescan.rhythm_symbol_guard import RHYTHM_SYMBOL_FEATURE_NAMES  # noqa: E402


def test_rendered_rhythm_symbol_transactions_are_deterministic_and_group_bound() -> None:
    left = build_rendered_rhythm_symbol_dataset(seed=20260805, groups=12)
    right = build_rendered_rhythm_symbol_dataset(seed=20260805, groups=12)

    assert np.allclose(left.features, right.features, rtol=0.0, atol=2e-7)
    assert np.array_equal(left.labels, right.labels)
    assert np.array_equal(left.groups, right.groups)
    assert left.scenarios == right.scenarios
    assert left.features.shape == (48, len(RHYTHM_SYMBOL_FEATURE_NAMES))
    assert np.isfinite(left.features).all()
    assert np.all((left.features >= -1.0) & (left.features <= 1.0))

    for group in range(12):
        rows = np.flatnonzero(left.groups == group)
        assert len(rows) == 4
        assert int(np.sum(left.labels[rows])) == 2
        assert set(int(value) for value in left.labels[rows]) == {0, 1}
        # Each rendered truth contributes one compatible transaction and its exact
        # reverse.  Candidate signatures are therefore balanced within the source group.
        scenario_names = [left.scenarios[index] for index in rows]
        assert sum(name.endswith(":compatible") for name in scenario_names) == 2
        assert sum(name.endswith(":incompatible") for name in scenario_names) == 2


def test_rendered_rhythm_symbol_group_offsets_do_not_change_features() -> None:
    left = build_rendered_rhythm_symbol_dataset(seed=20260805, groups=8, group_offset=0)
    right = build_rendered_rhythm_symbol_dataset(seed=20260805, groups=8, group_offset=100)
    assert np.allclose(left.features, right.features, rtol=0.0, atol=2e-7)
    assert np.array_equal(left.labels, right.labels)
    assert np.array_equal(right.groups, left.groups + 100)
