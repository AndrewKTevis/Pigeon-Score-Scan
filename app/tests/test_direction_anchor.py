from __future__ import annotations

from scorescan.direction_anchor import DirectionAnchorClassifier, extract_direction_anchor_features
from scorescan.layout import PageLayout, StaffSystem, anchor_x_to_measure


def _layout() -> PageLayout:
    spacing = 14.0
    system = StaffSystem(
        index=1,
        line_y=[300 + spacing * i for i in range(5)],
        top=240,
        bottom=420,
        left=100,
        right=1500,
        spacing=spacing,
        barlines=[350, 620, 900, 1200],
        measure_count=5,
    )
    return PageLayout(1600, 1000, [system], 0.98)


def test_direction_anchor_model_prefers_staff_direction() -> None:
    layout = _layout()
    model = DirectionAnchorClassifier()
    positive = extract_direction_anchor_features(
        text="Allegro con brio",
        kind="direction",
        score=0.92,
        box=[[160, 245], [420, 245], [420, 275], [160, 275]],
        backend="rapid+tesseract+contrast",
        correction_probability=0.97,
        correction_margin=0.42,
        correction_edit_ratio=0.05,
        system_index=0,
        placement="above",
        distance_staff_spaces=2.2,
        layout=layout,
    )
    negative = extract_direction_anchor_features(
        text="Johann Sebastian Bach",
        kind="metadata",
        score=0.96,
        box=[[420, 40], [1180, 40], [1180, 115], [420, 115]],
        backend="rapid+tesseract",
        correction_probability=0.12,
        correction_margin=0.01,
        correction_edit_ratio=0.78,
        system_index=None,
        placement="above",
        distance_staff_spaces=20.0,
        layout=layout,
    )
    assert model.enabled
    assert model.predict(positive) > 0.90
    assert model.predict(negative) < 0.10


def test_direction_anchor_model_keeps_dynamic_near_staff() -> None:
    layout = _layout()
    model = DirectionAnchorClassifier()
    features = extract_direction_anchor_features(
        text="mf",
        kind="dynamic",
        score=0.74,
        box=[[480, 380], [520, 380], [520, 405], [480, 405]],
        backend="rapid+tesseract",
        correction_probability=0.99,
        correction_margin=0.50,
        correction_edit_ratio=0.0,
        system_index=0,
        placement="below",
        distance_staff_spaces=1.7,
        layout=layout,
    )
    assert model.predict(features) > 0.80


def test_direction_measure_anchor_model_is_enabled_and_keeps_exact_counts() -> None:
    layout = _layout()
    system = layout.systems[0]
    model = DirectionAnchorClassifier()
    baseline = anchor_x_to_measure(system, 500.0, 5)
    refined = model.refine_measure_anchor(
        system,
        500.0,
        5,
        kind="dynamic",
        placement="below",
    )
    assert model.anchor_enabled
    assert refined == baseline
    assert refined.method == "barline_exact"


def test_direction_measure_anchor_refines_a_missing_boundary_case() -> None:
    system = StaffSystem(
        index=26,
        line_y=[300.0, 312.3535, 324.7070, 337.0605, 349.4140],
        top=250,
        bottom=398,
        left=107,
        right=928,
        spacing=12.3535,
        barlines=[386, 748],
        measure_count=3,
    )
    model = DirectionAnchorClassifier()
    refined = model.refine_measure_anchor(
        system,
        773.3959,
        4,
        kind="dynamic",
        placement="below",
    )
    assert refined.local_index == 3
    assert refined.method == "barline_model_refined"
    assert refined.confidence > 0.75


def test_direction_measure_anchor_feature_vector_is_finite() -> None:
    import math

    from scorescan.direction_anchor import ANCHOR_FEATURE_NAMES, anchor_candidate_feature_vector

    system = _layout().systems[0]
    values = anchor_candidate_feature_vector(
        system,
        720.0,
        6,
        2,
        kind="direction",
        placement="above",
    )
    assert len(values) == len(ANCHOR_FEATURE_NAMES)
    assert all(math.isfinite(value) for value in values)


def test_malformed_embedded_anchor_model_falls_back_without_disabling_role(tmp_path) -> None:
    import json
    from pathlib import Path

    baseline_path = (
        Path(__file__).resolve().parents[2]
        / "training"
        / "baselines"
        / "direction_anchor_classifier_v1.json"
    )
    payload = {
        "format": 2,
        "model_version": "broken-hybrid",
        "role_model": json.loads(baseline_path.read_text(encoding="utf-8")),
        "measure_anchor_model": {
            "model_type": "random_forest",
            "model_version": "broken-anchor",
            "feature_names": ["wrong"],
            "trees": [],
        },
    }
    resource = tmp_path / "direction_anchor_classifier.json"
    resource.write_text(json.dumps(payload), encoding="utf-8")
    model = DirectionAnchorClassifier(resource)
    system = _layout().systems[0]
    baseline = anchor_x_to_measure(system, 720.0, 6)
    refined = model.refine_measure_anchor(
        system,
        720.0,
        6,
        kind="direction",
        placement="above",
    )
    assert model.enabled
    assert not model.anchor_enabled
    assert refined == baseline
