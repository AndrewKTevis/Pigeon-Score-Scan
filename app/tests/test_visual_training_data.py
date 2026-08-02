from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from visual_training_data import KINDS, build_dataset  # noqa: E402
from scorescan.visual_evidence import (
    FEATURE_NAMES,
    LEGACY_FEATURE_NAMES,
    V3_FEATURE_NAMES,
    VisualMeasureCalibrator,
)  # noqa: E402


def test_visual_training_dataset_is_deterministic_across_worker_counts() -> None:
    serial = build_dataset(seed=1407, groups=40, workers=1)
    parallel = build_dataset(seed=1407, groups=40, workers=2)

    assert np.array_equal(serial.features, parallel.features)
    assert np.array_equal(serial.v3_features, parallel.v3_features)
    assert np.array_equal(serial.legacy_features, parallel.legacy_features)
    assert np.array_equal(serial.labels, parallel.labels)
    assert np.array_equal(serial.groups, parallel.groups)
    assert np.array_equal(serial.kinds, parallel.kinds)
    assert serial.decision_groups == parallel.decision_groups
    assert serial.features.shape[1] == len(FEATURE_NAMES)
    assert serial.v3_features.shape[1] == len(V3_FEATURE_NAMES)
    assert serial.legacy_features.shape[1] == len(LEGACY_FEATURE_NAMES)


def test_visual_training_groups_are_complete_and_not_order_labelled() -> None:
    dataset = build_dataset(seed=9173, groups=40, workers=1)
    positive_positions: set[tuple[int, ...]] = set()
    expected_kinds = set(KINDS)

    for group_id, decision in enumerate(dataset.decision_groups):
        assert len(decision) == len(KINDS)
        assert all(dataset.groups[index] == group_id for index in decision)
        assert {str(dataset.kinds[index]) for index in decision} == expected_kinds
        positions = tuple(offset for offset, index in enumerate(decision) if dataset.labels[index] == 1)
        assert len(positions) == 2
        positive_positions.add(positions)

    # Candidate order is deterministically shuffled per group, so a model cannot learn
    # the label from a fixed row position when visual evidence is tied.
    assert len(positive_positions) >= 8


def test_bundled_visual_model_retains_signal_on_rendered_groups() -> None:
    dataset = build_dataset(seed=8128, groups=40, workers=1)
    model = VisualMeasureCalibrator()
    assert model.enabled and model.forest.enabled

    correct = 0
    for decision in dataset.decision_groups:
        selected = max(decision, key=lambda index: (model.forest.predict(dataset.features[index]), -index))
        correct += int(dataset.labels[selected] == 1)

    # This is deliberately a broad safety floor rather than a duplicate of the frozen
    # training report.  The visual model is secondary evidence and each group contains
    # six targeted traps, two of which can be visually close to the source crop.
    assert correct >= 8


def test_bundled_visual_model_rejects_event_local_position_traps() -> None:
    dataset = build_dataset(seed=8128, groups=40, workers=1)
    model = VisualMeasureCalibrator()
    assert model.enabled and model.forest.enabled

    floors = {
        "pitch-order-trap": 16,
        "accidental-position-trap": 12,
        "compact-position-trap": 16,
        "open-notehead-position-trap": 24,
        "event-kind-position-trap": 28,
        "rhythm-position-trap": 12,
    }
    wins = {kind: 0 for kind in floors}
    for decision in dataset.decision_groups:
        compatible = max(
            model.forest.predict(dataset.features[index])
            for index in decision
            if dataset.labels[index] == 1
        )
        for index in decision:
            kind = str(dataset.kinds[index])
            if kind in wins:
                wins[kind] += int(compatible > model.forest.predict(dataset.features[index]))

    assert all(wins[kind] >= floor for kind, floor in floors.items()), wins
