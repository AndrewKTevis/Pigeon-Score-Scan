from scorescan.policy import DEFAULT_POLICY


def test_policy_is_versioned_and_conservative() -> None:
    assert DEFAULT_POLICY.version.startswith("scorescan-policy-")
    assert DEFAULT_POLICY.semantic_lyric_output_enabled is False
    assert 0 < DEFAULT_POLICY.semantic_distance_max < DEFAULT_POLICY.semantic_cluster_threshold < 1
    assert DEFAULT_POLICY.measure_calibration_weight_floor < 1 < DEFAULT_POLICY.measure_calibration_weight_ceiling
    assert DEFAULT_POLICY.replacement_measure_probability_floor < 0.5
    assert DEFAULT_POLICY.event_calibration_weight_floor < 1 < DEFAULT_POLICY.event_calibration_weight_ceiling
    assert DEFAULT_POLICY.replacement_event_probability_floor < 0.5


def test_policy_bounds_ensemble_meta_calibration() -> None:
    assert DEFAULT_POLICY.ensemble_calibration_weight_floor < 1 < DEFAULT_POLICY.ensemble_calibration_weight_ceiling
    assert DEFAULT_POLICY.replacement_ensemble_probability_floor < 0.5


def test_policy_keeps_event_patchers_fail_closed() -> None:
    assert DEFAULT_POLICY.chord_patch_minimum_families >= 3
    assert DEFAULT_POLICY.chord_patch_minimum_supporting_families >= 3
    assert DEFAULT_POLICY.chord_patch_probability_floor >= 0.95
    assert DEFAULT_POLICY.chord_patch_max_changed_ratio < 0.5
    assert DEFAULT_POLICY.pitch_patch_minimum_families >= 3
    assert DEFAULT_POLICY.rhythm_patch_minimum_families >= 3
    assert DEFAULT_POLICY.rhythm_patch_minimum_supporting_families >= 3
    assert DEFAULT_POLICY.rhythm_patch_pitch_coherence_floor > 0.5
    assert DEFAULT_POLICY.rhythm_patch_probability_floor >= 0.9
    assert DEFAULT_POLICY.grace_patch_minimum_families >= 3
    assert DEFAULT_POLICY.grace_patch_minimum_supporting_families >= 3
    assert DEFAULT_POLICY.grace_patch_probability_floor >= 0.9
    assert DEFAULT_POLICY.grace_patch_max_changed_ratio < 0.5


def test_measure_localized_internal_gate_is_bounded_and_single_family() -> None:
    assert DEFAULT_POLICY.measure_localized_internal_min_valid_variants >= 2
    assert DEFAULT_POLICY.measure_localized_internal_min_exact_support >= 2
    assert DEFAULT_POLICY.measure_localized_internal_min_margin >= 1
    assert (
        DEFAULT_POLICY.measure_localized_max_total_variant_pixels
        <= DEFAULT_POLICY.measure_localized_max_pixels * 3
    )
