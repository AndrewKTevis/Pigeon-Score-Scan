from __future__ import annotations

"""Versioned decision policy for the recognition pipeline.

All thresholds which decide whether ScoreScan runs additional OMR passes, trusts an
ensemble, replaces a measure, or asks for review live here.  Keeping these values in a
single immutable object makes releases auditable and avoids subtle drift between
recognition, consensus, and quality reporting.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RecognitionPolicy:
    version: str = "scorescan-policy-61"

    # Page-level routing.
    risky_page_quality: float = 72.0
    risky_layout_confidence: float = 0.72
    disagreement_agreement_floor: float = 0.82
    disagreement_score_margin: float = 22.0
    strong_candidate_min_score: float = 940.0
    strong_measure_gap_ratio: float = 0.10

    # Fusion of geometric and OMR measure-count evidence.  The CPU model can only
    # choose a count already observed in the layout or candidate set.
    measure_count_probability_floor: float = 0.875
    measure_count_margin_floor: float = 0.50
    measure_count_override_min_families: int = 3
    measure_count_high_confidence_layout_floor: float = 0.90
    measure_count_high_confidence_override_min_families: int = 4
    measure_count_layout_only_confidence_floor: float = 0.97
    consensus_template_count_minimum_families: int = 3

    # Deterministic scan-treatment routing.  Learned probabilities can request extra
    # candidates but never remove the established primary/flat/otsu/adaptive baseline.
    optional_variant_probability: float = 0.10
    staffnorm_variant_probability: float = 0.08
    upscale_target_min_dimension: int = 1900
    # Very short, panoramic staff strips occur in otherwise valid single-staff
    # material.  homr needs both a larger staff scale and quiet context around the
    # staff; ordinary page upscaling alone is insufficient for these inputs.
    lowres_strip_height_ceiling: int = 360
    lowres_strip_aspect_floor: float = 2.5
    lowres_strip_target_height: int = 536
    lowres_strip_low_target_height: int = 430
    lowres_strip_high_target_height: int = 600
    lowres_strip_max_scale: float = 4.5
    lowres_strip_vertical_margin_ratio: float = 0.35
    lowres_strip_horizontal_margin_ratio: float = 0.25
    lowres_strip_internal_min_support: int = 2
    lowres_strip_candidate_bonus: float = 2.0
    lowres_strip_count_override_score_margin: float = 40.0
    max_page_candidates: int = 7
    # Production recognition always starts with evidence from at least three
    # independent preprocessing families.  This costs one additional CPU OMR pass on
    # clean pages but makes strict family-majority repair and selective release
    # meaningful instead of relying on two correlated candidates.
    minimum_initial_candidate_families: int = 3

    # System-localised rescue. This is one additional, segmentation-independent OMR
    # candidate generated only after complete-page candidates remain structurally
    # ambiguous. Weak layouts and partial system results fail closed.
    localized_layout_confidence_floor: float = 0.62
    localized_min_systems: int = 2
    localized_max_systems: int = 12
    localized_vertical_context_ratio: float = 1.8
    localized_horizontal_context_ratio: float = 3.2
    localized_border_ratio: float = 2.2
    localized_min_horizontal_context_pixels: int = 24
    localized_min_border_pixels: int = 24
    localized_max_total_pixels: int = 180_000_000
    localized_measure_gap_ratio: float = 0.34
    localized_measure_gap_absolute: int = 1
    localized_trigger_agreement_floor: float = 0.86
    localized_trigger_score_margin: float = 28.0

    # Measure-localised rescue.  It is an additional OMR observation, not an automatic
    # patch.  Only unresolved measures which already have two independent family votes
    # are eligible; the local result must still create the normal three-family consensus.
    measure_localized_layout_confidence_floor: float = 0.72
    measure_localized_minimum_existing_families: int = 2
    measure_localized_max_measures: int = 3
    measure_localized_horizontal_context_ratio: float = 2.4
    measure_localized_border_ratio: float = 2.2
    measure_localized_min_context_pixels: int = 20
    measure_localized_min_border_pixels: int = 22
    measure_localized_max_pixels: int = 28_000_000
    # Three related crop treatments establish an internal exact majority over the
    # normalised XML content that would actually be spliced (including beams, stems and
    # uncommon notations) before the single measure-localisation family may vote.  They
    # remain one family and cannot manufacture the three-family production permission gate.
    measure_localized_max_total_variant_pixels: int = 84_000_000
    measure_localized_internal_min_valid_variants: int = 2
    measure_localized_internal_min_exact_support: int = 2
    measure_localized_internal_min_margin: int = 1

    # CPU barline classifier. Geometry remains the fallback and hard safety layer.
    barline_probability_floor: float = 0.25
    barline_min_spacing_ratio: float = 0.82
    barline_max_candidates_per_system: int = 24
    # High-side-density candidates are accepted only when independent continuity
    # measurements agree that one stroke crosses the complete staff.
    barline_connected_probability_floor: float = 0.94
    barline_connected_interline_mean_floor: float = 0.30
    barline_connected_run_floor: float = 0.35
    barline_connected_staff_intersection_floor: float = 0.80
    # Final opening-system guard. Both the local and sequence models must be weak.
    barline_opening_post_position_ceiling: float = 0.18
    barline_opening_post_local_ceiling: float = 0.80
    barline_opening_post_sequence_ceiling: float = 0.10
    barline_opening_split_local_ceiling: float = 0.93
    barline_opening_split_sequence_ceiling: float = 0.50
    barline_opening_split_gap_ratio_ceiling: float = 0.74
    barline_opening_split_merged_deviation: float = 0.30
    barline_dominated_pair_spacing_ratio: float = 2.20
    barline_dominated_pair_strong_floor: float = 0.955
    barline_dominated_pair_weak_ceiling: float = 0.94
    barline_dominated_pair_margin_floor: float = 0.025

    # Global barline sequence refinement. The learned model may only remove a
    # candidate when deterministic geometry also shows that it splits one plausible
    # measure into two short intervals.
    barline_sequence_probability_floor: float = 0.10
    barline_sequence_hard_reject_floor: float = 0.02
    barline_sequence_local_override: float = 0.96
    barline_sequence_short_gap_ratio: float = 0.92
    barline_sequence_merged_gap_deviation: float = 0.28
    barline_sequence_edge_distance_floor: float = 0.30
    barline_sequence_candidate_density_floor: float = 1.00
    barline_sequence_probability_margin_ceiling: float = 0.04
    barline_sequence_max_iterations: int = 8
    # A stricter opening-system gate covers high-confidence stems which split the
    # first measure.  It is independent of page identity and still requires the full
    # geometric split signature before the wider probability ceiling is considered.
    barline_sequence_opening_probability_floor: float = 0.15
    barline_sequence_opening_position_ceiling: float = 0.15
    barline_sequence_opening_local_ceiling: float = 0.90
    barline_sequence_opening_gap_ratio_ceiling: float = 0.68
    barline_sequence_opening_margin_ceiling: float = -0.10
    barline_sequence_opening_regularity_gain_floor: float = 0.50
    barline_sequence_opening_merged_gap_deviation: float = 0.24

    # OCR direction-role evidence.  The classifier can suppress unsafe automatic
    # injection, but low-confidence text remains visible for review.
    direction_anchor_dynamic_floor: float = 0.42
    direction_anchor_metronome_floor: float = 0.50
    direction_anchor_direction_floor: float = 0.66
    direction_anchor_text_floor: float = 0.88
    direction_anchor_review_floor: float = 0.48
    # Visual-to-MusicXML measure ownership is refined only for count mismatches and
    # only above both independent probability and winner-margin floors.
    direction_measure_anchor_probability_floor: float = 0.755
    direction_measure_anchor_margin_floor: float = 0.05

    # Candidate-sequence eligibility.
    alignment_min_similarity: float = 0.42
    allowed_measure_gap_ratio: float = 0.18

    # Measure consensus.
    semantic_cluster_threshold: float = 0.075
    semantic_support_min: float = 0.74
    semantic_distance_max: float = 0.045
    semantic_min_candidates: int = 3
    minimum_consensus_families: int = 3
    semantic_family_support_min: float = 0.50
    max_missing_candidate_fraction: float = 0.25
    significant_measure_distance: float = 0.008
    unresolved_measure_distance: float = 0.04
    replacement_page_score_slack: float = 30.0
    replacement_measure_probability_floor: float = 0.43
    replacement_event_probability_floor: float = 0.30
    replacement_context_probability_floor: float = 0.20
    replacement_ensemble_probability_floor: float = 0.34
    ensemble_review_probability_floor: float = 0.30
    # Final unified verifier for exact-majority and semantic-consensus replacements.
    replacement_selection_risk_floor: float = 0.90
    selection_risk_review_probability_floor: float = 0.70

    # Deterministic permission gates around the learned replacement verifier.  These
    # values can only veto a proposed replacement; they cannot create support or select
    # another candidate.  Family counts refer to complete independent preprocessing
    # families after fail-closed sibling validation.
    selection_exact_minimum_families: int = 3
    selection_exact_family_support_min: float = 2.0 / 3.0
    selection_exact_candidate_support_min: float = 2.0 / 3.0
    selection_exact_missing_ratio_max: float = 0.05
    selection_exact_ensemble_delta_floor: float = 0.0
    selection_exact_measure_delta_floor: float = 0.0
    selection_exact_event_delta_floor: float = -0.05
    selection_exact_direct_evidence_delta_floor: float = -0.10
    selection_exact_minimum_nondegrading_direct_layers: int = 2
    selection_semantic_minimum_families: int = 3
    selection_semantic_family_support_min: float = 2.0 / 3.0
    selection_semantic_candidate_support_min: float = 0.84
    # Whole-measure semantic replacement copies preservation surfaces not modelled by
    # Score IR.  At least two complete independent families must agree on the selected
    # canonical full-content signature; otherwise narrow patchers preserve the template.
    selection_semantic_preservation_minimum_families: int = 2
    selection_semantic_missing_ratio_max: float = 0.10
    selection_semantic_ensemble_delta_floor: float = 0.0
    selection_semantic_measure_delta_floor: float = 0.0
    selection_semantic_event_delta_floor: float = -0.05
    selection_semantic_context_delta_floor: float = -0.10
    selection_semantic_medoid_distance_max: float = 1e-9
    selection_semantic_page_score_delta_floor: float = -17.0
    selection_semantic_visual_delta_floor: float = -0.035

    # Chord-topology repair runs before other event-level repairs because one missing
    # ``<chord/>`` marker shifts every later onset.  It may only toggle markers on a
    # fixed simple event sequence, requires a strict majority of independent families,
    # and may not worsen the deterministic meter error.
    chord_patch_minimum_families: int = 3
    chord_patch_minimum_supporting_families: int = 3
    chord_patch_max_changed_markers: int = 3
    chord_patch_max_changed_ratio: float = 0.34
    chord_patch_max_chord_size: int = 6
    chord_patch_probability_floor: float = 0.97

    # Simple tuplet repair is deliberately limited to contiguous 3:2 triplet groups
    # in complete monophonic measures.  It toggles only time-modification and matching
    # visual endpoints; a CPU model may veto but never supplies family support.
    tuplet_patch_minimum_families: int = 3
    tuplet_patch_minimum_supporting_families: int = 3
    tuplet_patch_max_groups: int = 2
    tuplet_patch_max_changed_groups: int = 2
    tuplet_patch_max_changed_events: int = 6
    tuplet_patch_max_changed_ratio: float = 0.60
    tuplet_patch_max_events: int = 24
    tuplet_patch_probability_floor: float = 0.985

    # Simple repeat/barline repair changes only left/right bar styles and forward or
    # backward repeat markers already observed in independent families.  Volta endings,
    # fermatas and other navigation semantics remain review-only.
    barline_patch_minimum_families: int = 3
    barline_patch_minimum_supporting_families: int = 3
    barline_patch_max_changed_locations: int = 2
    barline_patch_max_repeat_changes: int = 2
    barline_patch_probability_floor: float = 0.99

    # Within-measure tie repair is limited to complete ties between consecutive,
    # contiguous pitched notes in one simple monophonic measure.  Cross-measure ties
    # and complex notation remain review-only.  The CPU model may only veto a strict
    # independent-family proposal.
    tie_patch_minimum_families: int = 3
    tie_patch_minimum_supporting_families: int = 3
    tie_patch_max_changed_endpoints: int = 4
    tie_patch_max_changed_pairs: int = 2
    tie_patch_max_changed_ratio: float = 0.75
    tie_patch_probability_floor: float = 0.975

    # The local tie visual guard confirms only additions between adjacent same-pitch
    # events after independent-family semantic consensus. It cannot classify a generic
    # arc, create support or remove a tie. Source-backed removals and tie/slur ambiguity
    # remain review-only; malformed non-empty evidence fails closed.
    tie_visual_guard_probability_floor: float = 0.92

    # The local accent guard confirms only an empty-articulation -> single-accent
    # addition after independent-family semantic consensus. It cannot recognise general
    # articulations, authorise deletion/substitution, or create support.
    accent_visual_guard_probability_floor: float = 0.70

    # Within-measure slur repair is limited to complete, non-overlapping arcs in a
    # simple monophonic measure.  It rewrites only slur endpoints and canonical
    # numbering; a CPU model may veto but never creates family support.
    slur_patch_minimum_families: int = 3
    slur_patch_minimum_supporting_families: int = 3
    slur_patch_max_arcs: int = 2
    slur_patch_max_changed_arcs: int = 2
    slur_patch_max_changed_endpoints: int = 4
    slur_patch_max_changed_ratio: float = 0.50
    slur_patch_max_span_events: int = 12
    slur_patch_max_events: int = 24
    slur_patch_probability_floor: float = 0.99

    # Simple articulation repair is event-local and accepts only empty, attribute-free
    # accent/staccato/tenuto-family markers on a complete monophonic event lattice.
    # It runs after pitch/rhythm/tie/slur repair and may only copy marker sets observed
    # in three independent preprocessing families; the CPU model is veto-only.
    articulation_patch_minimum_families: int = 3
    articulation_patch_minimum_supporting_families: int = 3
    articulation_patch_max_events: int = 24
    articulation_patch_max_changed_events: int = 6
    articulation_patch_max_changed_ratio: float = 0.50
    articulation_patch_max_marks_per_event: int = 2
    articulation_patch_probability_floor: float = 0.99

    # Simple ornament repair is limited to empty, attribute-free trill/turn/mordent
    # markers on a fixed single-voice event lattice. Complex wavy lines, tremolos,
    # accidental marks and ornaments carrying attributes remain review-only.
    ornament_patch_minimum_families: int = 3
    ornament_patch_minimum_supporting_families: int = 3
    ornament_patch_max_events: int = 24
    ornament_patch_max_changed_events: int = 4
    ornament_patch_max_changed_ratio: float = 0.35
    ornament_patch_max_marks_per_event: int = 1
    ornament_patch_probability_floor: float = 0.995

    # Semantic lyric transcription is outside the frozen product boundary.  Keep
    # the legacy proposal parameters only for isolated diagnostic experiments;
    # production consensus must not load or apply the lyric model.
    semantic_lyric_output_enabled: bool = False
    lyric_patch_minimum_families: int = 3
    lyric_patch_minimum_supporting_families: int = 3
    lyric_patch_max_events: int = 32
    lyric_patch_max_changed_events: int = 6
    lyric_patch_max_changed_ratio: float = 0.75
    lyric_patch_max_text_length: int = 64
    lyric_patch_probability_floor: float = 0.995

    # Simple grace-note repair changes whether an existing pitched event advances the
    # musical cursor.  It is considered only on a fixed monophonic sequence, requires
    # exact meter closure after the edit, and supports only empty attribute-free grace
    # elements.  The CPU model is veto-only.
    grace_patch_minimum_families: int = 3
    grace_patch_minimum_supporting_families: int = 3
    grace_patch_max_events: int = 24
    grace_patch_max_changed_events: int = 2
    grace_patch_max_changed_ratio: float = 0.25
    grace_patch_probability_floor: float = 0.90

    # Simple performance-direction repair is restricted to compact dynamics and exact
    # integer metronome marks on a complete monophonic measure.  Words, wedges, pedal
    # marks and formatted or attributed direction content remain review-only.  Family
    # votes determine the proposal and the CPU model can only veto it.
    direction_patch_minimum_families: int = 3
    direction_patch_minimum_supporting_families: int = 3
    direction_patch_max_events: int = 32
    direction_patch_max_directions: int = 4
    direction_patch_max_changed_directions: int = 3
    direction_patch_probability_floor: float = 0.90

    # Cross-measure tie repair runs after all local measure transactions.  It may only
    # toggle the start/stop pair at one aligned adjacent boundary between complete,
    # monophonic measures with matching pitched boundary events.  A model can veto but
    # cannot create independent-family support.
    cross_tie_patch_minimum_families: int = 3
    cross_tie_patch_minimum_supporting_families: int = 3
    cross_tie_patch_probability_floor: float = 0.995

    # Event-level pitch consensus is narrower than whole-measure replacement.  It is
    # considered only when the complete non-pitch event skeleton agrees.  Independent
    # preprocessing families, not raw variant count, determine support.
    pitch_patch_minimum_families: int = 3
    pitch_patch_minimum_supporting_families: int = 3
    pitch_patch_probability_floor: float = 0.90
    pitch_patch_no_visual_probability_floor: float = 0.70
    # Direct rendered source-crop guard for staff-position pitch changes.  The model
    # may only veto a family-majority transaction and fails closed when unavailable.
    pitch_visual_guard_probability_floor: float = 0.90
    # Binary source-crop guard for printed accidental presence changes.  It cannot
    # classify accidental type and therefore only vetoes missing/extra symbol edits.
    accidental_presence_guard_probability_floor: float = 0.70

    # Event-level rhythm consensus is limited to meter-complete monophonic measures
    # without chords, grace notes, tuplets, or explicit cursor movement.  The CPU model
    # remains a veto-only layer after strict independent-family and XML validation.
    rhythm_patch_minimum_families: int = 3
    rhythm_patch_minimum_supporting_families: int = 3
    rhythm_patch_pitch_coherence_floor: float = 0.66
    rhythm_patch_probability_floor: float = 0.92
    # Pairwise source-crop transaction guard.  It compares the complete proposed
    # rhythm edit with the current template after the semantic model has accepted.
    # The guard can only veto and uses forward/reverse consistency.
    rhythm_symbol_guard_probability_floor: float = 0.9875

    # Score attributes have score-wide semantic impact.  Attribute repair therefore
    # requires a strict majority of at least three independent preprocessing families,
    # explicit boundary evidence and a high-precision veto model.
    attribute_patch_minimum_families: int = 3
    attribute_patch_minimum_supporting_families: int = 3
    attribute_patch_probability_floor: float = 0.93

    # Rest-versus-pitched-note repair is limited to a fixed simple event lattice and
    # requires three independent families plus a higher precision floor because changing
    # event kind can remove or introduce sounding material.
    event_kind_patch_minimum_families: int = 3
    event_kind_patch_minimum_supporting_families: int = 3
    event_kind_patch_probability_floor: float = 0.95

    # The local event-kind visual guard confirms exactly one fixed-onset, fixed-duration
    # note-versus-rest replacement after independent-family semantic consensus. It is
    # veto-only; multiple changes, chords, grace notes and unsupported types remain
    # review-only when source evidence is present.
    event_kind_visual_guard_probability_floor: float = 0.93

    # One-event insertion/deletion repair is the highest-risk local edit.  It is only
    # considered for uniquely anchored interior measures and requires three complete
    # independent families, a separately voted inserted event, and a very high-precision
    # CPU veto model.
    event_presence_patch_minimum_families: int = 3
    event_presence_patch_minimum_supporting_families: int = 3
    event_presence_patch_minimum_content_families: int = 3
    event_presence_patch_anchor_match_floor: float = 0.66
    event_presence_patch_probability_floor: float = 0.975
    event_presence_visual_guard_probability_floor: float = 0.90

    # Composed local repairs receive a second, interaction-aware veto only when two or
    # more semantic patch classes (or four total classes) are committed together.  The
    # model cannot approve a failed deterministic transaction, create family support,
    # or alter the patch order.  Missing or corrupt model data fails closed for these
    # high-risk bundles while simple and orthogonal repairs retain deterministic flow.
    patch_transaction_model_minimum_semantic_patches: int = 2
    patch_transaction_model_minimum_total_patches: int = 4
    patch_transaction_probability_floor: float = 0.995

    # Learned priors are intentionally bounded.  Hard validation remains authoritative.
    page_calibration_max_adjustment: float = 32.0
    measure_calibration_weight_floor: float = 0.72
    measure_calibration_weight_ceiling: float = 1.18
    event_calibration_weight_floor: float = 0.88
    event_calibration_weight_ceiling: float = 1.12
    context_calibration_weight_floor: float = 0.90
    context_calibration_weight_ceiling: float = 1.10
    ensemble_calibration_weight_floor: float = 0.86
    ensemble_calibration_weight_ceiling: float = 1.14

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_POLICY = RecognitionPolicy()
