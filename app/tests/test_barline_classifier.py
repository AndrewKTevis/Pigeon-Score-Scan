from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from scorescan.barline_classifier import BarlineClassifier, extract_barline_features
from scorescan.layout import PageLayout, StaffSystem


def _staff_scene(*, barline: bool) -> tuple[np.ndarray, int, list[float], float]:
    spacing = 14.0
    image = np.zeros((150, 260), np.uint8)
    lines = [45.0 + spacing * index for index in range(5)]
    for y in lines:
        cv2.line(image, (15, int(y)), (245, int(y)), 255, 1)
    x = 130
    if barline:
        cv2.line(image, (x, int(lines[0])), (x, int(lines[-1])), 255, 2)
    else:
        cv2.ellipse(image, (x - 5, int(lines[3])), (6, 4), -20, 0, 360, 255, -1)
        cv2.line(image, (x, int(lines[3])), (x, int(lines[0] - spacing)), 255, 2)
    return image, x, lines, spacing


def test_trained_barline_model_prefers_staff_spanning_boundary() -> None:
    classifier = BarlineClassifier()
    positive, x, lines, spacing = _staff_scene(barline=True)
    negative, nx, nlines, nspacing = _staff_scene(barline=False)
    positive_result = classifier.classify(
        extract_barline_features(positive, x_start=x - 1, x_end=x + 1, line_y=lines, spacing=spacing)
    )
    negative_result = classifier.classify(
        extract_barline_features(negative, x_start=nx - 1, x_end=nx + 1, line_y=nlines, spacing=nspacing)
    )
    assert classifier.enabled
    assert positive_result.probability > negative_result.probability
    assert positive_result.accepted


def test_layout_round_trip_preserves_barline_confidence() -> None:
    layout = PageLayout(
        1000,
        1400,
        [StaffSystem(1, [100, 112, 124, 136, 148], 60, 190, 30, 970, 12.0, [250, 500], 3, [0.98, 0.87])],
        0.95,
    )
    restored = PageLayout.from_dict(layout.to_dict())
    assert restored.systems[0].barlines == [250, 500]
    assert restored.systems[0].barline_confidences == [0.98, 0.87]


def test_layout_detector_rejects_note_stems_and_keeps_measure_barlines(tmp_path: Path) -> None:
    image_path = tmp_path / "staff.png"
    image = np.full((260, 900), 255, np.uint8)
    lines = [90, 102, 114, 126, 138]
    for y in lines:
        cv2.line(image, (30, y), (870, y), 0, 1)
    for x in (240, 450, 660, 870):
        cv2.line(image, (x, lines[0]), (x, lines[-1]), 0, 2)
    for x, y in ((120, 126), (180, 102), (310, 114), (380, 138), (520, 126), (590, 102), (740, 114), (800, 138)):
        cv2.ellipse(image, (x, y), (6, 4), -20, 0, 360, 0, -1)
        cv2.line(image, (x + 5, y), (x + 5, y - 42), 0, 2)
    cv2.imwrite(str(image_path), image)

    from scorescan.layout import analyze_layout

    layout = analyze_layout(image_path)
    assert len(layout.systems) == 1
    assert layout.systems[0].measure_count == 4
    assert layout.systems[0].barlines == [240, 450, 660, 870]
    assert min(layout.systems[0].barline_confidences[:-1]) > 0.90
    from scorescan.policy import DEFAULT_POLICY

    assert layout.systems[0].barline_confidences[-1] > DEFAULT_POLICY.barline_probability_floor


def _features(**overrides: float):
    from scorescan.barline_classifier import BarlineFeatures, FEATURE_NAMES

    defaults = {name: 0.0 for name in FEATURE_NAMES}
    defaults.update(overrides)
    return BarlineFeatures(**defaults)


def test_dense_connected_boundary_requires_independent_continuity_evidence() -> None:
    from scorescan.layout import BarlineProposalEvidence, _local_barline_accept

    features = _features(
        row_coverage=0.98,
        longest_vertical_run=0.92,
        staff_line_intersection_ratio=1.0,
        side_density=0.88,
        interline_mean_coverage=0.86,
        interline_min_coverage=0.70,
    )
    accepted = BarlineProposalEvidence(200, 199, 201, 0.95, features, True, True, 0.5, True)
    weak = BarlineProposalEvidence(200, 199, 201, 0.90, features, True, True, 0.5, True)
    assert _local_barline_accept(accepted, probability_floor=0.25)
    assert not _local_barline_accept(weak, probability_floor=0.25)


def test_complete_interline_boundary_can_rescue_low_model_probability() -> None:
    from scorescan.layout import BarlineProposalEvidence, _local_barline_accept

    features = _features(
        row_coverage=0.94,
        longest_vertical_run=0.94,
        staff_line_intersection_ratio=1.0,
        side_density=0.28,
        above_extension=0.06,
        below_extension=0.0,
        mid_horizontal_attachment=0.45,
        local_vertical_dominance=0.43,
        interline_mean_coverage=1.0,
        interline_min_coverage=1.0,
    )
    rescued = BarlineProposalEvidence(
        514, 513, 515, 0.08, features, True, True, 0.41, True
    )
    assert _local_barline_accept(rescued, probability_floor=0.25)

    for unsafe_features in (
        _features(**{**features.__dict__, "interline_min_coverage": 0.75}),
        _features(**{**features.__dict__, "above_extension": 0.80}),
        _features(**{**features.__dict__, "side_density": 0.65}),
    ):
        unsafe = BarlineProposalEvidence(
            514, 513, 515, 0.08, unsafe_features, True, True, 0.41, True
        )
        assert not _local_barline_accept(unsafe, probability_floor=0.25)


def test_post_refine_opening_guard_ignores_explicit_left_edge() -> None:
    from scorescan.barline_sequence import RefinedBarline
    from scorescan.layout import _post_refine_barlines

    system = StaffSystem(1, [40, 50, 60, 70, 80], 20, 100, 0, 1000, 10.0)
    sequence = tuple(
        RefinedBarline(x, local, sequence_probability, local, True, False)
        for x, local, sequence_probability in (
            (2, 0.99, 0.99),
            (100, 0.70, 0.20),
            (200, 0.96, 0.95),
            (400, 0.97, 0.96),
            (600, 0.97, 0.96),
            (800, 0.98, 0.97),
        )
    )
    assert [item.x for item in _post_refine_barlines(system, sequence)] == [2, 200, 400, 600, 800]


def test_post_refine_removes_only_close_dominated_duplicate() -> None:
    from scorescan.barline_sequence import RefinedBarline
    from scorescan.layout import _post_refine_barlines

    system = StaffSystem(1, [40, 50, 60, 70, 80], 20, 100, 0, 1000, 10.0)
    sequence = tuple(
        RefinedBarline(x, probability, 0.9, probability, True, False)
        for x, probability in ((200, 0.98), (216, 0.85), (400, 0.96), (650, 0.97))
    )
    assert [item.x for item in _post_refine_barlines(system, sequence)] == [200, 400, 650]
