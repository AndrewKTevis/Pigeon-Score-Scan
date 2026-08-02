from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from tie_visual_training_data import (  # noqa: E402
    build_tie_slur_ambiguity_dataset,
    build_tie_visual_dataset,
)
from scorescan.tie_visual_guard import TIE_VISUAL_FEATURE_NAMES  # noqa: E402


def test_tie_visual_dataset_is_group_paired_and_worker_deterministic() -> None:
    serial = build_tie_visual_dataset(seed=20261608, groups=12, workers=1)
    parallel = build_tie_visual_dataset(seed=20261608, groups=12, workers=2)
    assert np.array_equal(serial.features, parallel.features)
    assert np.array_equal(serial.labels, parallel.labels)
    assert np.array_equal(serial.groups, parallel.groups)
    assert serial.scenarios == parallel.scenarios
    assert serial.features.shape == (24, len(TIE_VISUAL_FEATURE_NAMES))
    visually_changed = 0
    for group in range(12):
        indices = np.flatnonzero(serial.groups == group)
        assert indices.tolist() == [group * 2, group * 2 + 1]
        assert serial.labels[indices].tolist() == [0, 1]
        # Candidate geometry is identical inside a pair. Some deliberately clipped or
        # occluded curves are invisible and therefore identical; the safety model must
        # abstain on those instead of learning a semantic prior.
        assert np.array_equal(serial.features[indices[0], -2:], serial.features[indices[1], -2:])
        visually_changed += int(
            not np.array_equal(
                serial.features[indices[0], :-2], serial.features[indices[1], :-2]
            )
        )
    assert visually_changed >= 6


def test_rejected_same_endpoint_slur_dataset_remains_explicitly_separate() -> None:
    dataset = build_tie_slur_ambiguity_dataset(seed=20261609, groups=8)
    assert dataset.features.shape == (16, len(TIE_VISUAL_FEATURE_NAMES))
    assert dataset.labels.tolist() == [0, 1] * 8
    assert set(dataset.scenarios) == {"same_endpoint_slur", "target_tie"}
