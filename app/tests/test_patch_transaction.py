from __future__ import annotations

from pathlib import Path

from lxml import etree

from scorescan.patch_transaction import (
    FEATURE_NAMES,
    PatchTransactionCalibrator,
    PatchTransactionInput,
    patch_transaction_guard,
    validate_patch_stage,
)


def _input(patch_kinds: tuple[str, ...], **overrides: object) -> PatchTransactionInput:
    values: dict[str, object] = {
        "patch_kinds": patch_kinds,
        "changed_event_count": 3,
        "changed_surface_count": 3,
        "minimum_patch_probability": 1.0,
        "mean_patch_probability": 1.0,
        "minimum_patch_margin": 0.03,
        "mean_patch_margin": 0.065,
        "maximum_patch_threshold": 0.97,
        "eligible_family_count": 4,
        "exact_family_support_ratio": 0.75,
        "semantic_family_support_ratio": 0.75,
        "missing_ratio": 0.0,
        "selected_measure_probability": 0.55,
        "selected_visual_probability": 0.50,
        "selected_event_probability": 0.973,
        "selected_context_probability": 0.277,
        "selected_ensemble_probability": 0.741,
        "semantic_confidence": 0.741,
        "mean_cluster_distance": 0.0,
        "template_distance": 0.129,
    }
    values.update(overrides)
    return PatchTransactionInput(**values)  # type: ignore[arg-type]


def test_patch_transaction_features_are_bounded_and_interaction_aware() -> None:
    item = _input(("chord", "pitch"))
    vector = item.feature_vector()

    assert len(vector) == len(FEATURE_NAMES)
    assert all(-1.0 <= value <= 1.0 for value in vector)
    assert item.patch_count == 2
    assert item.semantic_patch_count == 2
    assert item.requires_model()
    assert vector[FEATURE_NAMES.index("chord_pitch_interaction")] == 1.0
    assert vector[FEATURE_NAMES.index("pitch_rhythm_interaction")] == 0.0


def test_patch_transaction_model_accepts_compatible_bundle_and_vetoes_high_risk_bundle() -> None:
    calibrator = PatchTransactionCalibrator()
    assert calibrator.enabled
    assert calibrator.model_verified

    compatible = calibrator.calibrate(_input(("chord", "pitch")))
    assert compatible.applicable
    assert compatible.accepted
    assert compatible.probability >= compatible.threshold

    risky = calibrator.calibrate(
        _input(
            ("event_presence", "rhythm", "pitch"),
            changed_event_count=8,
            changed_surface_count=8,
            minimum_patch_probability=0.98,
            mean_patch_probability=0.99,
            minimum_patch_margin=0.005,
            mean_patch_margin=0.015,
            maximum_patch_threshold=0.975,
            eligible_family_count=3,
            exact_family_support_ratio=0.80,
            semantic_family_support_ratio=0.85,
            missing_ratio=0.10,
            selected_measure_probability=0.75,
            selected_visual_probability=0.65,
            selected_event_probability=0.55,
            selected_context_probability=0.50,
            selected_ensemble_probability=0.70,
            semantic_confidence=0.70,
            mean_cluster_distance=0.03,
            template_distance=0.06,
        )
    )
    assert risky.applicable
    assert not risky.accepted
    assert risky.reason == "model_veto"


def test_patch_transaction_model_failure_is_conservative_only_for_applicable_bundles(
    tmp_path: Path,
) -> None:
    calibrator = PatchTransactionCalibrator(tmp_path / "missing.json")

    simple = calibrator.calibrate(_input(("pitch",)))
    assert not simple.applicable
    assert simple.accepted
    assert simple.reason == "deterministic_only"

    composed = calibrator.calibrate(_input(("chord", "pitch")))
    assert composed.applicable
    assert not composed.accepted
    assert composed.reason == "model_unavailable"


def test_patch_transaction_guard_rejects_new_semantic_defect() -> None:
    original = etree.fromstring(
        b"""
        <measure number='1'>
          <attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
          <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><type>whole</type></note>
        </measure>
        """
    )
    unsafe = etree.fromstring(etree.tostring(original))
    unsafe.find("note/type").text = "quarter"
    inherited: dict[str, object] = {"divisions": 1, "time": (4, 4), "key": 0, "clef": ("G", 2)}

    accepted, reason = patch_transaction_guard(original, unsafe, inherited)
    assert not accepted
    assert reason == "introduced_type_duration_mismatch"


def test_patch_stage_validation_preserves_previous_safe_repairs() -> None:
    original = etree.fromstring(
        b"""
        <measure number='1'>
          <attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
          <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><type>whole</type></note>
        </measure>
        """
    )
    current = etree.fromstring(etree.tostring(original))
    direction = etree.Element("direction")
    direction_type = etree.SubElement(direction, "direction-type")
    etree.SubElement(direction_type, "words").text = "dolce"
    current.insert(1, direction)
    unsafe = etree.fromstring(etree.tostring(current))
    unsafe.find("note/type").text = "quarter"
    inherited: dict[str, object] = {"divisions": 1, "time": (4, 4), "key": 0, "clef": ("G", 2)}

    result = validate_patch_stage(original, current, unsafe, inherited)

    assert not result.accepted
    assert result.reason == "introduced_type_duration_mismatch"
    assert result.measure is current
    assert result.measure.find("direction/direction-type/words").text == "dolce"
