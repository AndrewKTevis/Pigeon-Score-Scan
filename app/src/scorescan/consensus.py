from __future__ import annotations

import copy
import hashlib
from collections import Counter
import math
import statistics
from dataclasses import dataclass, field, replace as dataclass_replace
from pathlib import Path
from typing import Iterable, Protocol

from lxml import etree

from .alignment import SequenceAlignment, align_measure_sequences
from .attribute_consensus import (
    AttributePatchCalibrator,
    AttributePatchCandidate,
    AttributePatchResult,
    propose_attribute_patch,
)
from .articulation_consensus import (
    ArticulationPatchCalibrator,
    ArticulationPatchCandidate,
    ArticulationPatchResult,
    propose_articulation_patch,
)
from .accent_visual_guard import AccentVisualAudit, AccentVisualGuard
from .ornament_consensus import (
    OrnamentPatchCalibrator,
    OrnamentPatchCandidate,
    OrnamentPatchResult,
    propose_ornament_patch,
)
from .grace_consensus import (
    GracePatchCalibrator,
    GracePatchCandidate,
    GracePatchResult,
    propose_grace_patch,
)
from .lyric_consensus import (
    LyricPatchCalibrator,
    LyricPatchCandidate,
    LyricPatchResult,
    propose_lyric_patch,
)
from .direction_consensus import (
    DirectionPatchCalibrator,
    DirectionPatchCandidate,
    DirectionPatchResult,
    propose_direction_patch,
)
from .barline_consensus import (
    BarlinePatchCalibrator,
    BarlinePatchCandidate,
    BarlinePatchResult,
    propose_barline_patch,
)
from .chord_consensus import (
    ChordPatchCalibrator,
    ChordPatchCandidate,
    ChordPatchResult,
    propose_chord_patch,
)
from .context_calibration import ContextCalibrator, agreement_profiles as context_agreement_profiles
from .cross_tie_consensus import (
    CrossTiePatchCalibrator,
    CrossTiePatchCandidate,
    propose_cross_tie_patch,
)
from .event_calibration import EventCalibrator, agreement_profiles
from .ensemble_calibration import EnsembleCalibrationInput, EnsembleCalibrator
from .event_kind_consensus import (
    EventKindPatchCalibrator,
    EventKindPatchCandidate,
    EventKindPatchResult,
    propose_event_kind_patch,
)
from .event_kind_visual_guard import EventKindVisualAudit, EventKindVisualGuard
from .event_presence_consensus import (
    EventPresencePatchCalibrator,
    EventPresencePatchCandidate,
    EventPresencePatchResult,
    propose_event_presence_patch,
)
from .event_presence_visual_guard import (
    EventPresenceVisualAudit,
    EventPresenceVisualGuard,
)
from .measure_calibration import MeasureCalibrationInput, MeasureCalibrator
from .measure_localized import candidate_applies_to_boundary, candidate_applies_to_measure
from .musicxml_signature import (
    canonical_measure_bytes,
    measure_preservation_signature,
    measure_preservation_signatures,
)
from .policy import DEFAULT_POLICY
from .patch_transaction import (
    PatchEvidence,
    PatchTransactionCalibrator,
    PatchTransactionInput,
    patch_transaction_guard as _patch_transaction_guard,
    validate_patch_stage,
)
from .pitch_consensus import (
    PitchPatchCalibrator,
    PitchPatchCandidate,
    PitchPatchResult,
    propose_pitch_patch,
)
from .rhythm_consensus import (
    RhythmPatchCalibrator,
    RhythmPatchCandidate,
    RhythmPatchResult,
    propose_rhythm_patch,
)
from .tie_visual_guard import TieVisualAudit, TieVisualGuard
from .tie_consensus import (
    TiePatchCalibrator,
    TiePatchCandidate,
    TiePatchResult,
    propose_tie_patch,
)
from .slur_consensus import (
    SlurPatchCalibrator,
    SlurPatchCandidate,
    SlurPatchResult,
    propose_slur_patch,
)
from .tuplet_consensus import (
    TupletPatchCalibrator,
    TupletPatchCandidate,
    TupletPatchResult,
    propose_tuplet_patch,
)
from .score_ir import MeasureIR, ScoreIR, audit_score, measure_distance, measure_from_xml, score_from_tree
from .selection_risk import SelectionRiskCalibrator, SelectionRiskInput
from .variant_family import variant_family
from .visual_evidence import (
    VisualMeasureCalibrator,
    VisualMeasureEvidence,
    map_evidence_to_measure,
)
from .util import atomic_write_bytes

MUSICXML_DOCTYPE = (
    '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
    '"http://www.musicxml.org/dtds/partwise.dtd">'
)


class CandidateLike(Protocol):
    variant: str
    xml_path: str | None
    score: float
    valid: bool


@dataclass(frozen=True)
class MeasureVote:
    measure_index: int
    selected_variant: str
    selected_support: int
    eligible_candidates: int
    unanimous: bool
    strict_majority: bool
    replaced_template: bool
    decision: str
    exact_family_support: int = 0
    semantic_family_support: int = 0
    selected_preservation_family_support: int = 0
    preservation_gate_required: bool = False
    preservation_gate_accepted: bool = True
    eligible_family_count: int = 0
    abstaining_family_count: int = 0
    abstaining_families: tuple[str, ...] = ()
    signatures: dict[str, list[str]] = field(default_factory=dict)
    semantic_support_ratio: float = 0.0
    semantic_confidence: float = 0.0
    mean_cluster_distance: float = 0.0
    template_distance: float = 0.0
    cluster_variants: tuple[str, ...] = ()
    aligned_candidates: int = 0
    missing_candidates: int = 0
    selected_measure_probability: float = 0.5
    measure_calibration_model: str = "disabled"
    candidate_measure_probabilities: dict[str, float] = field(default_factory=dict)
    selected_visual_probability: float = 0.5
    visual_calibration_model: str = "disabled"
    candidate_visual_probabilities: dict[str, float] = field(default_factory=dict)
    selected_event_probability: float = 0.5
    event_calibration_model: str = "disabled"
    candidate_event_probabilities: dict[str, float] = field(default_factory=dict)
    selected_context_probability: float = 0.5
    context_calibration_model: str = "disabled"
    candidate_context_probabilities: dict[str, float] = field(default_factory=dict)
    selected_ensemble_probability: float = 0.5
    ensemble_calibration_model: str = "disabled"
    candidate_ensemble_probabilities: dict[str, float] = field(default_factory=dict)
    selection_risk_applicable: bool = False
    selected_selection_risk_probability: float = 0.5
    selection_risk_threshold: float = 1.0
    selection_risk_accepted: bool = False
    selection_risk_model: str = "disabled"
    chord_patch_applicable: bool = False
    chord_patch_event_count: int = 0
    chord_patch_probability: float = 0.5
    chord_patch_threshold: float = 1.0
    chord_patch_accepted: bool = False
    chord_patch_model: str = "disabled"
    chord_patch_reason: str = "not_applicable"
    tuplet_patch_applicable: bool = False
    tuplet_patch_event_count: int = 0
    tuplet_patch_group_count: int = 0
    tuplet_patch_probability: float = 0.5
    tuplet_patch_threshold: float = 1.0
    tuplet_patch_accepted: bool = False
    tuplet_patch_model: str = "disabled"
    tuplet_patch_reason: str = "not_applicable"
    pitch_patch_applicable: bool = False
    pitch_patch_event_count: int = 0
    pitch_patch_probability: float = 0.5
    pitch_patch_threshold: float = 1.0
    pitch_patch_accepted: bool = False
    pitch_patch_model: str = "disabled"
    pitch_patch_reason: str = "not_applicable"
    rhythm_patch_applicable: bool = False
    rhythm_patch_event_count: int = 0
    rhythm_patch_probability: float = 0.5
    rhythm_patch_threshold: float = 1.0
    rhythm_patch_accepted: bool = False
    rhythm_patch_model: str = "disabled"
    rhythm_patch_reason: str = "not_applicable"
    event_kind_patch_applicable: bool = False
    event_kind_patch_event_count: int = 0
    event_kind_patch_probability: float = 0.5
    event_kind_patch_threshold: float = 1.0
    event_kind_patch_accepted: bool = False
    event_kind_patch_model: str = "disabled"
    event_kind_patch_reason: str = "not_applicable"
    event_kind_visual_guard_applicable: bool = False
    event_kind_visual_guard_changed_events: int = 0
    event_kind_visual_guard_probability: float = 0.5
    event_kind_visual_guard_threshold: float = 1.0
    event_kind_visual_guard_accepted: bool = True
    event_kind_visual_guard_model: str = "disabled"
    event_kind_visual_guard_reason: str = "not_applicable"
    attribute_patch_applicable: bool = False
    attribute_patch_attributes: tuple[str, ...] = ()
    attribute_patch_probability: float = 0.5
    attribute_patch_threshold: float = 1.0
    attribute_patch_accepted: bool = False
    attribute_patch_model: str = "disabled"
    attribute_patch_reason: str = "not_applicable"
    barline_patch_applicable: bool = False
    barline_patch_locations: tuple[str, ...] = ()
    barline_patch_repeat_count: int = 0
    barline_patch_probability: float = 0.5
    barline_patch_threshold: float = 1.0
    barline_patch_accepted: bool = False
    barline_patch_model: str = "disabled"
    barline_patch_reason: str = "not_applicable"
    tie_patch_applicable: bool = False
    tie_patch_event_count: int = 0
    tie_patch_probability: float = 0.5
    tie_patch_threshold: float = 1.0
    tie_patch_accepted: bool = False
    tie_patch_model: str = "disabled"
    tie_patch_reason: str = "not_applicable"
    tie_visual_guard_applicable: bool = False
    tie_visual_guard_changed_ties: int = 0
    tie_visual_guard_probability: float = 0.5
    tie_visual_guard_threshold: float = 1.0
    tie_visual_guard_accepted: bool = True
    tie_visual_guard_model: str = "disabled"
    tie_visual_guard_reason: str = "not_applicable"
    slur_patch_applicable: bool = False
    slur_patch_event_count: int = 0
    slur_patch_arc_count: int = 0
    slur_patch_probability: float = 0.5
    slur_patch_threshold: float = 1.0
    slur_patch_accepted: bool = False
    slur_patch_model: str = "disabled"
    slur_patch_reason: str = "not_applicable"
    articulation_patch_applicable: bool = False
    articulation_patch_event_count: int = 0
    articulation_patch_mark_count: int = 0
    articulation_patch_probability: float = 0.5
    articulation_patch_threshold: float = 1.0
    articulation_patch_accepted: bool = False
    articulation_patch_model: str = "disabled"
    articulation_patch_reason: str = "not_applicable"
    accent_visual_guard_applicable: bool = False
    accent_visual_guard_changed_accents: int = 0
    accent_visual_guard_probability: float = 0.5
    accent_visual_guard_threshold: float = 1.0
    accent_visual_guard_accepted: bool = True
    accent_visual_guard_model: str = "disabled"
    accent_visual_guard_reason: str = "not_applicable"
    ornament_patch_applicable: bool = False
    ornament_patch_event_count: int = 0
    ornament_patch_mark_count: int = 0
    ornament_patch_probability: float = 0.5
    ornament_patch_threshold: float = 1.0
    ornament_patch_accepted: bool = False
    ornament_patch_model: str = "disabled"
    ornament_patch_reason: str = "not_applicable"
    grace_patch_applicable: bool = False
    grace_patch_event_count: int = 0
    grace_patch_added_count: int = 0
    grace_patch_removed_count: int = 0
    grace_patch_probability: float = 0.5
    grace_patch_threshold: float = 1.0
    grace_patch_accepted: bool = False
    grace_patch_model: str = "disabled"
    grace_patch_reason: str = "not_applicable"
    lyric_patch_applicable: bool = False
    lyric_patch_event_count: int = 0
    lyric_patch_lyric_count: int = 0
    lyric_patch_probability: float = 0.5
    lyric_patch_threshold: float = 1.0
    lyric_patch_accepted: bool = False
    lyric_patch_model: str = "disabled"
    lyric_patch_reason: str = "not_applicable"
    direction_patch_applicable: bool = False
    direction_patch_direction_count: int = 0
    direction_patch_kinds: tuple[str, ...] = ()
    direction_patch_probability: float = 0.5
    direction_patch_threshold: float = 1.0
    direction_patch_accepted: bool = False
    direction_patch_model: str = "disabled"
    direction_patch_reason: str = "not_applicable"
    patch_transaction_applicable: bool = False
    patch_transaction_patch_count: int = 0
    patch_transaction_semantic_patch_count: int = 0
    patch_transaction_probability: float = 1.0
    patch_transaction_threshold: float = 1.0
    patch_transaction_model: str = "disabled"
    patch_stage_rejections: tuple[str, ...] = ()
    patch_transaction_accepted: bool = True
    patch_transaction_reason: str = "not_applicable"
    event_presence_patch_applicable: bool = False
    event_presence_patch_operation: str = "none"
    event_presence_patch_event_count: int = 0
    event_presence_patch_probability: float = 0.5
    event_presence_patch_threshold: float = 1.0
    event_presence_patch_accepted: bool = False
    event_presence_patch_model: str = "disabled"
    event_presence_patch_reason: str = "not_applicable"
    event_presence_visual_guard_applicable: bool = False
    event_presence_visual_guard_operation: str = "none"
    event_presence_visual_guard_changed_events: int = 0
    event_presence_visual_guard_probability: float = 0.5
    event_presence_visual_guard_threshold: float = 1.0
    event_presence_visual_guard_accepted: bool = True
    event_presence_visual_guard_model: str = "disabled"
    event_presence_visual_guard_reason: str = "not_applicable"

    def to_dict(self) -> dict[str, object]:
        return {
            "measure_index": self.measure_index,
            "selected_variant": self.selected_variant,
            "selected_support": self.selected_support,
            "eligible_candidates": self.eligible_candidates,
            "unanimous": self.unanimous,
            "strict_majority": self.strict_majority,
            "exact_family_support": self.exact_family_support,
            "semantic_family_support": self.semantic_family_support,
            "eligible_family_count": self.eligible_family_count,
            "abstaining_family_count": self.abstaining_family_count,
            "abstaining_families": list(self.abstaining_families),
            "replaced_template": self.replaced_template,
            "decision": self.decision,
            "signatures": self.signatures,
            "semantic_support_ratio": self.semantic_support_ratio,
            "semantic_confidence": self.semantic_confidence,
            "mean_cluster_distance": self.mean_cluster_distance,
            "template_distance": self.template_distance,
            "cluster_variants": list(self.cluster_variants),
            "aligned_candidates": self.aligned_candidates,
            "missing_candidates": self.missing_candidates,
            "selected_measure_probability": self.selected_measure_probability,
            "measure_calibration_model": self.measure_calibration_model,
            "candidate_measure_probabilities": self.candidate_measure_probabilities,
            "selected_visual_probability": self.selected_visual_probability,
            "visual_calibration_model": self.visual_calibration_model,
            "candidate_visual_probabilities": self.candidate_visual_probabilities,
            "selected_event_probability": self.selected_event_probability,
            "event_calibration_model": self.event_calibration_model,
            "candidate_event_probabilities": self.candidate_event_probabilities,
            "selected_context_probability": self.selected_context_probability,
            "context_calibration_model": self.context_calibration_model,
            "candidate_context_probabilities": self.candidate_context_probabilities,
            "selected_ensemble_probability": self.selected_ensemble_probability,
            "ensemble_calibration_model": self.ensemble_calibration_model,
            "candidate_ensemble_probabilities": self.candidate_ensemble_probabilities,
            "selection_risk_applicable": self.selection_risk_applicable,
            "selected_selection_risk_probability": self.selected_selection_risk_probability,
            "selection_risk_threshold": self.selection_risk_threshold,
            "selection_risk_accepted": self.selection_risk_accepted,
            "selection_risk_model": self.selection_risk_model,
            "chord_patch_applicable": self.chord_patch_applicable,
            "chord_patch_event_count": self.chord_patch_event_count,
            "chord_patch_probability": self.chord_patch_probability,
            "chord_patch_threshold": self.chord_patch_threshold,
            "chord_patch_accepted": self.chord_patch_accepted,
            "chord_patch_model": self.chord_patch_model,
            "chord_patch_reason": self.chord_patch_reason,
            "tuplet_patch_applicable": self.tuplet_patch_applicable,
            "tuplet_patch_event_count": self.tuplet_patch_event_count,
            "tuplet_patch_group_count": self.tuplet_patch_group_count,
            "tuplet_patch_probability": self.tuplet_patch_probability,
            "tuplet_patch_threshold": self.tuplet_patch_threshold,
            "tuplet_patch_accepted": self.tuplet_patch_accepted,
            "tuplet_patch_model": self.tuplet_patch_model,
            "tuplet_patch_reason": self.tuplet_patch_reason,
            "pitch_patch_applicable": self.pitch_patch_applicable,
            "pitch_patch_event_count": self.pitch_patch_event_count,
            "pitch_patch_probability": self.pitch_patch_probability,
            "pitch_patch_threshold": self.pitch_patch_threshold,
            "pitch_patch_accepted": self.pitch_patch_accepted,
            "pitch_patch_model": self.pitch_patch_model,
            "pitch_patch_reason": self.pitch_patch_reason,
            "rhythm_patch_applicable": self.rhythm_patch_applicable,
            "rhythm_patch_event_count": self.rhythm_patch_event_count,
            "rhythm_patch_probability": self.rhythm_patch_probability,
            "rhythm_patch_threshold": self.rhythm_patch_threshold,
            "rhythm_patch_accepted": self.rhythm_patch_accepted,
            "rhythm_patch_model": self.rhythm_patch_model,
            "rhythm_patch_reason": self.rhythm_patch_reason,
            "event_kind_patch_applicable": self.event_kind_patch_applicable,
            "event_kind_patch_event_count": self.event_kind_patch_event_count,
            "event_kind_patch_probability": self.event_kind_patch_probability,
            "event_kind_patch_threshold": self.event_kind_patch_threshold,
            "event_kind_patch_accepted": self.event_kind_patch_accepted,
            "event_kind_patch_model": self.event_kind_patch_model,
            "event_kind_patch_reason": self.event_kind_patch_reason,
            "event_kind_visual_guard_applicable": self.event_kind_visual_guard_applicable,
            "event_kind_visual_guard_changed_events": self.event_kind_visual_guard_changed_events,
            "event_kind_visual_guard_probability": self.event_kind_visual_guard_probability,
            "event_kind_visual_guard_threshold": self.event_kind_visual_guard_threshold,
            "event_kind_visual_guard_accepted": self.event_kind_visual_guard_accepted,
            "event_kind_visual_guard_model": self.event_kind_visual_guard_model,
            "event_kind_visual_guard_reason": self.event_kind_visual_guard_reason,
            "attribute_patch_applicable": self.attribute_patch_applicable,
            "attribute_patch_attributes": list(self.attribute_patch_attributes),
            "attribute_patch_probability": self.attribute_patch_probability,
            "attribute_patch_threshold": self.attribute_patch_threshold,
            "attribute_patch_accepted": self.attribute_patch_accepted,
            "attribute_patch_model": self.attribute_patch_model,
            "attribute_patch_reason": self.attribute_patch_reason,
            "barline_patch_applicable": self.barline_patch_applicable,
            "barline_patch_locations": list(self.barline_patch_locations),
            "barline_patch_repeat_count": self.barline_patch_repeat_count,
            "barline_patch_probability": self.barline_patch_probability,
            "barline_patch_threshold": self.barline_patch_threshold,
            "barline_patch_accepted": self.barline_patch_accepted,
            "barline_patch_model": self.barline_patch_model,
            "barline_patch_reason": self.barline_patch_reason,
            "tie_patch_applicable": self.tie_patch_applicable,
            "tie_patch_event_count": self.tie_patch_event_count,
            "tie_patch_probability": self.tie_patch_probability,
            "tie_patch_threshold": self.tie_patch_threshold,
            "tie_patch_accepted": self.tie_patch_accepted,
            "tie_patch_model": self.tie_patch_model,
            "tie_patch_reason": self.tie_patch_reason,
            "tie_visual_guard_applicable": self.tie_visual_guard_applicable,
            "tie_visual_guard_changed_ties": self.tie_visual_guard_changed_ties,
            "tie_visual_guard_probability": self.tie_visual_guard_probability,
            "tie_visual_guard_threshold": self.tie_visual_guard_threshold,
            "tie_visual_guard_accepted": self.tie_visual_guard_accepted,
            "tie_visual_guard_model": self.tie_visual_guard_model,
            "tie_visual_guard_reason": self.tie_visual_guard_reason,
            "slur_patch_applicable": self.slur_patch_applicable,
            "slur_patch_event_count": self.slur_patch_event_count,
            "slur_patch_arc_count": self.slur_patch_arc_count,
            "slur_patch_probability": self.slur_patch_probability,
            "slur_patch_threshold": self.slur_patch_threshold,
            "slur_patch_accepted": self.slur_patch_accepted,
            "slur_patch_model": self.slur_patch_model,
            "slur_patch_reason": self.slur_patch_reason,
            "articulation_patch_applicable": self.articulation_patch_applicable,
            "articulation_patch_event_count": self.articulation_patch_event_count,
            "articulation_patch_mark_count": self.articulation_patch_mark_count,
            "articulation_patch_probability": self.articulation_patch_probability,
            "articulation_patch_threshold": self.articulation_patch_threshold,
            "articulation_patch_accepted": self.articulation_patch_accepted,
            "articulation_patch_model": self.articulation_patch_model,
            "articulation_patch_reason": self.articulation_patch_reason,
            "accent_visual_guard_applicable": self.accent_visual_guard_applicable,
            "accent_visual_guard_changed_accents": self.accent_visual_guard_changed_accents,
            "accent_visual_guard_probability": self.accent_visual_guard_probability,
            "accent_visual_guard_threshold": self.accent_visual_guard_threshold,
            "accent_visual_guard_accepted": self.accent_visual_guard_accepted,
            "accent_visual_guard_model": self.accent_visual_guard_model,
            "accent_visual_guard_reason": self.accent_visual_guard_reason,
            "ornament_patch_applicable": self.ornament_patch_applicable,
            "ornament_patch_event_count": self.ornament_patch_event_count,
            "ornament_patch_mark_count": self.ornament_patch_mark_count,
            "ornament_patch_probability": self.ornament_patch_probability,
            "ornament_patch_threshold": self.ornament_patch_threshold,
            "ornament_patch_accepted": self.ornament_patch_accepted,
            "ornament_patch_model": self.ornament_patch_model,
            "ornament_patch_reason": self.ornament_patch_reason,
            "grace_patch_applicable": self.grace_patch_applicable,
            "grace_patch_event_count": self.grace_patch_event_count,
            "grace_patch_added_count": self.grace_patch_added_count,
            "grace_patch_removed_count": self.grace_patch_removed_count,
            "grace_patch_probability": self.grace_patch_probability,
            "grace_patch_threshold": self.grace_patch_threshold,
            "grace_patch_accepted": self.grace_patch_accepted,
            "grace_patch_model": self.grace_patch_model,
            "grace_patch_reason": self.grace_patch_reason,
            "lyric_patch_applicable": self.lyric_patch_applicable,
            "lyric_patch_event_count": self.lyric_patch_event_count,
            "lyric_patch_lyric_count": self.lyric_patch_lyric_count,
            "lyric_patch_probability": self.lyric_patch_probability,
            "lyric_patch_threshold": self.lyric_patch_threshold,
            "lyric_patch_accepted": self.lyric_patch_accepted,
            "lyric_patch_model": self.lyric_patch_model,
            "lyric_patch_reason": self.lyric_patch_reason,
            "direction_patch_applicable": self.direction_patch_applicable,
            "direction_patch_direction_count": self.direction_patch_direction_count,
            "direction_patch_kinds": list(self.direction_patch_kinds),
            "direction_patch_probability": self.direction_patch_probability,
            "direction_patch_threshold": self.direction_patch_threshold,
            "direction_patch_accepted": self.direction_patch_accepted,
            "direction_patch_model": self.direction_patch_model,
            "direction_patch_reason": self.direction_patch_reason,
            "patch_transaction_applicable": self.patch_transaction_applicable,
            "patch_transaction_patch_count": self.patch_transaction_patch_count,
            "patch_transaction_semantic_patch_count": self.patch_transaction_semantic_patch_count,
            "patch_transaction_probability": self.patch_transaction_probability,
            "patch_transaction_threshold": self.patch_transaction_threshold,
            "patch_transaction_model": self.patch_transaction_model,
            "patch_stage_rejections": list(self.patch_stage_rejections),
            "patch_transaction_accepted": self.patch_transaction_accepted,
            "patch_transaction_reason": self.patch_transaction_reason,
            "event_presence_patch_applicable": self.event_presence_patch_applicable,
            "event_presence_patch_operation": self.event_presence_patch_operation,
            "event_presence_patch_event_count": self.event_presence_patch_event_count,
            "event_presence_patch_probability": self.event_presence_patch_probability,
            "event_presence_patch_threshold": self.event_presence_patch_threshold,
            "event_presence_patch_accepted": self.event_presence_patch_accepted,
            "event_presence_patch_model": self.event_presence_patch_model,
            "event_presence_patch_reason": self.event_presence_patch_reason,
            "event_presence_visual_guard_applicable": self.event_presence_visual_guard_applicable,
            "event_presence_visual_guard_operation": self.event_presence_visual_guard_operation,
            "event_presence_visual_guard_changed_events": self.event_presence_visual_guard_changed_events,
            "event_presence_visual_guard_probability": self.event_presence_visual_guard_probability,
            "event_presence_visual_guard_threshold": self.event_presence_visual_guard_threshold,
            "event_presence_visual_guard_accepted": self.event_presence_visual_guard_accepted,
            "event_presence_visual_guard_model": self.event_presence_visual_guard_model,
            "event_presence_visual_guard_reason": self.event_presence_visual_guard_reason,
        }


@dataclass(frozen=True)
class ConsensusReport:
    output_path: str
    template_variant: str
    candidate_count: int
    eligible_candidate_count: int
    measure_count: int
    agreement_ratio: float
    unanimous_measure_count: int
    majority_measure_count: int
    disagreement_measure_indices: tuple[int, ...]
    unresolved_measure_indices: tuple[int, ...]
    replacements: int
    votes: tuple[MeasureVote, ...]
    exact_agreement_ratio: float = 0.0
    semantic_agreement_ratio: float = 0.0
    mean_measure_confidence: float = 0.0
    preservation_disagreement_measure_indices: tuple[int, ...] = ()
    resolved_disagreement_measure_indices: tuple[int, ...] = ()
    candidate_alignment: dict[str, dict[str, object]] = field(default_factory=dict)
    mean_selected_measure_probability: float = 0.5
    measure_calibration_model: str = "disabled"
    policy_version: str = DEFAULT_POLICY.version
    requested_measure_count: int = 0
    template_measure_count: int = 0
    template_count_family_support: int = 0
    template_count_eligible_family_count: int = 0
    template_count_reselected: bool = False
    mean_visual_probability: float = 0.5
    visual_calibration_model: str = "disabled"
    mean_event_probability: float = 0.5
    event_calibration_model: str = "disabled"
    mean_context_probability: float = 0.5
    context_calibration_model: str = "disabled"
    mean_ensemble_probability: float = 0.5
    ensemble_calibration_model: str = "disabled"
    mean_selection_risk_probability: float = 0.5
    selection_risk_model: str = "disabled"
    selection_risk_threshold: float = 1.0
    chord_patch_measure_count: int = 0
    chord_patch_event_count: int = 0
    mean_chord_patch_probability: float = 0.5
    chord_patch_model: str = "disabled"
    chord_patch_threshold: float = 1.0
    tuplet_patch_measure_count: int = 0
    tuplet_patch_event_count: int = 0
    tuplet_patch_group_count: int = 0
    mean_tuplet_patch_probability: float = 0.5
    tuplet_patch_model: str = "disabled"
    tuplet_patch_threshold: float = 1.0
    pitch_patch_measure_count: int = 0
    pitch_patch_event_count: int = 0
    mean_pitch_patch_probability: float = 0.5
    pitch_patch_model: str = "disabled"
    pitch_patch_threshold: float = 1.0
    rhythm_patch_measure_count: int = 0
    rhythm_patch_event_count: int = 0
    mean_rhythm_patch_probability: float = 0.5
    rhythm_patch_model: str = "disabled"
    rhythm_patch_threshold: float = 1.0
    event_kind_patch_measure_count: int = 0
    event_kind_patch_event_count: int = 0
    mean_event_kind_patch_probability: float = 0.5
    event_kind_patch_model: str = "disabled"
    event_kind_patch_threshold: float = 1.0
    event_kind_visual_guard_transaction_count: int = 0
    event_kind_visual_guard_rejected_count: int = 0
    mean_event_kind_visual_guard_probability: float = 0.5
    event_kind_visual_guard_model: str = "disabled"
    event_kind_visual_guard_threshold: float = 1.0
    attribute_patch_measure_count: int = 0
    attribute_patch_attribute_count: int = 0
    mean_attribute_patch_probability: float = 0.5
    attribute_patch_model: str = "disabled"
    attribute_patch_threshold: float = 1.0
    barline_patch_measure_count: int = 0
    barline_patch_location_count: int = 0
    barline_patch_repeat_count: int = 0
    mean_barline_patch_probability: float = 0.5
    barline_patch_model: str = "disabled"
    barline_patch_threshold: float = 1.0
    tie_patch_measure_count: int = 0
    tie_patch_event_count: int = 0
    mean_tie_patch_probability: float = 0.5
    tie_patch_model: str = "disabled"
    tie_patch_threshold: float = 1.0
    tie_visual_guard_transaction_count: int = 0
    tie_visual_guard_rejected_count: int = 0
    mean_tie_visual_guard_probability: float = 0.5
    tie_visual_guard_model: str = "disabled"
    tie_visual_guard_threshold: float = 1.0
    slur_patch_measure_count: int = 0
    slur_patch_event_count: int = 0
    slur_patch_arc_count: int = 0
    mean_slur_patch_probability: float = 0.5
    slur_patch_model: str = "disabled"
    slur_patch_threshold: float = 1.0
    articulation_patch_measure_count: int = 0
    articulation_patch_event_count: int = 0
    articulation_patch_mark_count: int = 0
    mean_articulation_patch_probability: float = 0.5
    articulation_patch_model: str = "disabled"
    articulation_patch_threshold: float = 1.0
    accent_visual_guard_transaction_count: int = 0
    accent_visual_guard_rejected_count: int = 0
    mean_accent_visual_guard_probability: float = 0.5
    accent_visual_guard_model: str = "disabled"
    accent_visual_guard_threshold: float = 1.0
    ornament_patch_measure_count: int = 0
    ornament_patch_event_count: int = 0
    ornament_patch_mark_count: int = 0
    mean_ornament_patch_probability: float = 0.5
    ornament_patch_model: str = "disabled"
    ornament_patch_threshold: float = 1.0
    grace_patch_measure_count: int = 0
    grace_patch_event_count: int = 0
    grace_patch_added_count: int = 0
    grace_patch_removed_count: int = 0
    mean_grace_patch_probability: float = 0.5
    grace_patch_model: str = "disabled"
    grace_patch_threshold: float = 1.0
    lyric_patch_measure_count: int = 0
    lyric_patch_event_count: int = 0
    lyric_patch_lyric_count: int = 0
    mean_lyric_patch_probability: float = 0.5
    lyric_patch_model: str = "disabled"
    lyric_patch_threshold: float = 1.0
    direction_patch_measure_count: int = 0
    direction_patch_direction_count: int = 0
    mean_direction_patch_probability: float = 0.5
    direction_patch_model: str = "disabled"
    direction_patch_threshold: float = 1.0
    cross_tie_patch_boundary_count: int = 0
    cross_tie_patch_endpoint_count: int = 0
    mean_cross_tie_patch_probability: float = 0.5
    cross_tie_patch_model: str = "disabled"
    cross_tie_patch_threshold: float = 1.0
    cross_tie_patch_transaction_rejected_count: int = 0
    cross_tie_boundaries: tuple[dict[str, object], ...] = ()
    patch_stage_rejected_count: int = 0
    patch_transaction_evaluated_count: int = 0
    patch_transaction_rejected_count: int = 0
    mean_patch_transaction_probability: float = 1.0
    patch_transaction_model: str = "disabled"
    patch_transaction_threshold: float = 1.0
    event_presence_patch_measure_count: int = 0
    event_presence_patch_inserted_event_count: int = 0
    event_presence_patch_deleted_event_count: int = 0
    mean_event_presence_patch_probability: float = 0.5
    event_presence_patch_model: str = "disabled"
    event_presence_patch_threshold: float = 1.0
    event_presence_visual_guard_transaction_count: int = 0
    event_presence_visual_guard_rejected_count: int = 0
    mean_event_presence_visual_guard_probability: float = 0.5
    event_presence_visual_guard_model: str = "disabled"
    event_presence_visual_guard_threshold: float = 1.0
    event_presence_visual_guard_note_threshold: float = 1.0
    event_presence_visual_guard_rest_threshold: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "output_path": self.output_path,
            "template_variant": self.template_variant,
            "candidate_count": self.candidate_count,
            "eligible_candidate_count": self.eligible_candidate_count,
            "measure_count": self.measure_count,
            "agreement_ratio": self.agreement_ratio,
            "exact_agreement_ratio": self.exact_agreement_ratio,
            "semantic_agreement_ratio": self.semantic_agreement_ratio,
            "mean_measure_confidence": self.mean_measure_confidence,
            "unanimous_measure_count": self.unanimous_measure_count,
            "majority_measure_count": self.majority_measure_count,
            "preservation_disagreement_measure_indices": list(
                self.preservation_disagreement_measure_indices
            ),
            "resolved_disagreement_measure_indices": list(
                self.resolved_disagreement_measure_indices
            ),
            "disagreement_measure_indices": list(self.disagreement_measure_indices),
            "unresolved_measure_indices": list(self.unresolved_measure_indices),
            "replacements": self.replacements,
            "votes": [vote.to_dict() for vote in self.votes],
            "candidate_alignment": self.candidate_alignment,
            "mean_selected_measure_probability": self.mean_selected_measure_probability,
            "measure_calibration_model": self.measure_calibration_model,
            "policy_version": self.policy_version,
            "requested_measure_count": self.requested_measure_count,
            "template_measure_count": self.template_measure_count,
            "template_count_family_support": self.template_count_family_support,
            "template_count_eligible_family_count": self.template_count_eligible_family_count,
            "template_count_reselected": self.template_count_reselected,
            "mean_visual_probability": self.mean_visual_probability,
            "visual_calibration_model": self.visual_calibration_model,
            "mean_event_probability": self.mean_event_probability,
            "event_calibration_model": self.event_calibration_model,
            "mean_context_probability": self.mean_context_probability,
            "context_calibration_model": self.context_calibration_model,
            "mean_ensemble_probability": self.mean_ensemble_probability,
            "ensemble_calibration_model": self.ensemble_calibration_model,
            "mean_selection_risk_probability": self.mean_selection_risk_probability,
            "selection_risk_model": self.selection_risk_model,
            "selection_risk_threshold": self.selection_risk_threshold,
            "chord_patch_measure_count": self.chord_patch_measure_count,
            "chord_patch_event_count": self.chord_patch_event_count,
            "mean_chord_patch_probability": self.mean_chord_patch_probability,
            "chord_patch_model": self.chord_patch_model,
            "chord_patch_threshold": self.chord_patch_threshold,
            "tuplet_patch_measure_count": self.tuplet_patch_measure_count,
            "tuplet_patch_event_count": self.tuplet_patch_event_count,
            "tuplet_patch_group_count": self.tuplet_patch_group_count,
            "mean_tuplet_patch_probability": self.mean_tuplet_patch_probability,
            "tuplet_patch_model": self.tuplet_patch_model,
            "tuplet_patch_threshold": self.tuplet_patch_threshold,
            "pitch_patch_measure_count": self.pitch_patch_measure_count,
            "pitch_patch_event_count": self.pitch_patch_event_count,
            "mean_pitch_patch_probability": self.mean_pitch_patch_probability,
            "pitch_patch_model": self.pitch_patch_model,
            "pitch_patch_threshold": self.pitch_patch_threshold,
            "rhythm_patch_measure_count": self.rhythm_patch_measure_count,
            "rhythm_patch_event_count": self.rhythm_patch_event_count,
            "mean_rhythm_patch_probability": self.mean_rhythm_patch_probability,
            "rhythm_patch_model": self.rhythm_patch_model,
            "rhythm_patch_threshold": self.rhythm_patch_threshold,
            "event_kind_patch_measure_count": self.event_kind_patch_measure_count,
            "event_kind_patch_event_count": self.event_kind_patch_event_count,
            "mean_event_kind_patch_probability": self.mean_event_kind_patch_probability,
            "event_kind_patch_model": self.event_kind_patch_model,
            "event_kind_patch_threshold": self.event_kind_patch_threshold,
            "event_kind_visual_guard_transaction_count": self.event_kind_visual_guard_transaction_count,
            "event_kind_visual_guard_rejected_count": self.event_kind_visual_guard_rejected_count,
            "mean_event_kind_visual_guard_probability": self.mean_event_kind_visual_guard_probability,
            "event_kind_visual_guard_model": self.event_kind_visual_guard_model,
            "event_kind_visual_guard_threshold": self.event_kind_visual_guard_threshold,
            "attribute_patch_measure_count": self.attribute_patch_measure_count,
            "attribute_patch_attribute_count": self.attribute_patch_attribute_count,
            "mean_attribute_patch_probability": self.mean_attribute_patch_probability,
            "attribute_patch_model": self.attribute_patch_model,
            "attribute_patch_threshold": self.attribute_patch_threshold,
            "barline_patch_measure_count": self.barline_patch_measure_count,
            "barline_patch_location_count": self.barline_patch_location_count,
            "barline_patch_repeat_count": self.barline_patch_repeat_count,
            "mean_barline_patch_probability": self.mean_barline_patch_probability,
            "barline_patch_model": self.barline_patch_model,
            "barline_patch_threshold": self.barline_patch_threshold,
            "tie_patch_measure_count": self.tie_patch_measure_count,
            "tie_patch_event_count": self.tie_patch_event_count,
            "mean_tie_patch_probability": self.mean_tie_patch_probability,
            "tie_patch_model": self.tie_patch_model,
            "tie_patch_threshold": self.tie_patch_threshold,
            "tie_visual_guard_transaction_count": self.tie_visual_guard_transaction_count,
            "tie_visual_guard_rejected_count": self.tie_visual_guard_rejected_count,
            "mean_tie_visual_guard_probability": self.mean_tie_visual_guard_probability,
            "tie_visual_guard_model": self.tie_visual_guard_model,
            "tie_visual_guard_threshold": self.tie_visual_guard_threshold,
            "slur_patch_measure_count": self.slur_patch_measure_count,
            "slur_patch_event_count": self.slur_patch_event_count,
            "slur_patch_arc_count": self.slur_patch_arc_count,
            "mean_slur_patch_probability": self.mean_slur_patch_probability,
            "slur_patch_model": self.slur_patch_model,
            "slur_patch_threshold": self.slur_patch_threshold,
            "articulation_patch_measure_count": self.articulation_patch_measure_count,
            "articulation_patch_event_count": self.articulation_patch_event_count,
            "articulation_patch_mark_count": self.articulation_patch_mark_count,
            "mean_articulation_patch_probability": self.mean_articulation_patch_probability,
            "articulation_patch_model": self.articulation_patch_model,
            "articulation_patch_threshold": self.articulation_patch_threshold,
            "accent_visual_guard_transaction_count": self.accent_visual_guard_transaction_count,
            "accent_visual_guard_rejected_count": self.accent_visual_guard_rejected_count,
            "mean_accent_visual_guard_probability": self.mean_accent_visual_guard_probability,
            "accent_visual_guard_model": self.accent_visual_guard_model,
            "accent_visual_guard_threshold": self.accent_visual_guard_threshold,
            "ornament_patch_measure_count": self.ornament_patch_measure_count,
            "ornament_patch_event_count": self.ornament_patch_event_count,
            "ornament_patch_mark_count": self.ornament_patch_mark_count,
            "mean_ornament_patch_probability": self.mean_ornament_patch_probability,
            "ornament_patch_model": self.ornament_patch_model,
            "ornament_patch_threshold": self.ornament_patch_threshold,
            "grace_patch_measure_count": self.grace_patch_measure_count,
            "grace_patch_event_count": self.grace_patch_event_count,
            "grace_patch_added_count": self.grace_patch_added_count,
            "grace_patch_removed_count": self.grace_patch_removed_count,
            "mean_grace_patch_probability": self.mean_grace_patch_probability,
            "grace_patch_model": self.grace_patch_model,
            "grace_patch_threshold": self.grace_patch_threshold,
            "lyric_patch_measure_count": self.lyric_patch_measure_count,
            "lyric_patch_event_count": self.lyric_patch_event_count,
            "lyric_patch_lyric_count": self.lyric_patch_lyric_count,
            "mean_lyric_patch_probability": self.mean_lyric_patch_probability,
            "lyric_patch_model": self.lyric_patch_model,
            "lyric_patch_threshold": self.lyric_patch_threshold,
            "direction_patch_measure_count": self.direction_patch_measure_count,
            "direction_patch_direction_count": self.direction_patch_direction_count,
            "mean_direction_patch_probability": self.mean_direction_patch_probability,
            "direction_patch_model": self.direction_patch_model,
            "direction_patch_threshold": self.direction_patch_threshold,
            "cross_tie_patch_boundary_count": self.cross_tie_patch_boundary_count,
            "cross_tie_patch_endpoint_count": self.cross_tie_patch_endpoint_count,
            "mean_cross_tie_patch_probability": self.mean_cross_tie_patch_probability,
            "cross_tie_patch_model": self.cross_tie_patch_model,
            "cross_tie_patch_threshold": self.cross_tie_patch_threshold,
            "cross_tie_patch_transaction_rejected_count": self.cross_tie_patch_transaction_rejected_count,
            "cross_tie_boundaries": list(self.cross_tie_boundaries),
            "patch_stage_rejected_count": self.patch_stage_rejected_count,
            "patch_transaction_evaluated_count": self.patch_transaction_evaluated_count,
            "patch_transaction_rejected_count": self.patch_transaction_rejected_count,
            "mean_patch_transaction_probability": self.mean_patch_transaction_probability,
            "patch_transaction_model": self.patch_transaction_model,
            "patch_transaction_threshold": self.patch_transaction_threshold,
            "event_presence_patch_measure_count": self.event_presence_patch_measure_count,
            "event_presence_patch_inserted_event_count": self.event_presence_patch_inserted_event_count,
            "event_presence_patch_deleted_event_count": self.event_presence_patch_deleted_event_count,
            "mean_event_presence_patch_probability": self.mean_event_presence_patch_probability,
            "event_presence_patch_model": self.event_presence_patch_model,
            "event_presence_patch_threshold": self.event_presence_patch_threshold,
            "event_presence_visual_guard_transaction_count": self.event_presence_visual_guard_transaction_count,
            "event_presence_visual_guard_rejected_count": self.event_presence_visual_guard_rejected_count,
            "mean_event_presence_visual_guard_probability": self.mean_event_presence_visual_guard_probability,
            "event_presence_visual_guard_model": self.event_presence_visual_guard_model,
            "event_presence_visual_guard_threshold": self.event_presence_visual_guard_threshold,
            "event_presence_visual_guard_note_threshold": self.event_presence_visual_guard_note_threshold,
            "event_presence_visual_guard_rest_threshold": self.event_presence_visual_guard_rest_threshold,
        }


@dataclass(frozen=True)
class _LoadedCandidate:
    candidate: CandidateLike
    tree: etree._ElementTree
    measures: tuple[etree._Element, ...]
    semantics: tuple[MeasureIR, ...]
    signatures: tuple[str, ...]


@dataclass(frozen=True)
class _TemplateCountDecision:
    entry: _LoadedCandidate
    requested_count: int
    family_support: int
    eligible_family_count: int
    reselected: bool


def _select_count_consistent_template(
    loaded: list[_LoadedCandidate],
    initial: _LoadedCandidate,
    requested_count: int,
) -> _TemplateCountDecision:
    """Reselect the XML template only with a strict independent-family majority.

    Measure alignment cannot recover a measure which does not exist in the template
    tree.  When the independently resolved page count is supported by a strict
    majority of complete preprocessing families, use the strongest valid candidate
    with that exact count as the structural template.  Correlated siblings count once;
    a family with an invalid sibling or internally conflicting counts abstains.
    """
    requested_count = max(0, int(requested_count))
    if requested_count <= 0 or len(initial.measures) == requested_count:
        return _TemplateCountDecision(initial, requested_count, 0, 0, False)

    grouped: dict[str, list[_LoadedCandidate]] = {}
    for item in loaded:
        grouped.setdefault(variant_family(item.candidate.variant), []).append(item)
    complete: dict[str, list[_LoadedCandidate]] = {
        family: members
        for family, members in grouped.items()
        if members and all(bool(member.candidate.valid) for member in members)
    }
    supporting_families: set[str] = set()
    for family, members in complete.items():
        counts = {len(member.measures) for member in members}
        if counts == {requested_count}:
            supporting_families.add(family)

    family_support = len(supporting_families)
    eligible_family_count = len(complete)
    strict_majority = family_support * 2 > eligible_family_count
    if (
        family_support < DEFAULT_POLICY.consensus_template_count_minimum_families
        or not strict_majority
    ):
        return _TemplateCountDecision(
            initial,
            requested_count,
            family_support,
            eligible_family_count,
            False,
        )

    choices = [
        item
        for item in loaded
        if bool(item.candidate.valid)
        and len(item.measures) == requested_count
        and variant_family(item.candidate.variant) in supporting_families
    ]
    if not choices:
        return _TemplateCountDecision(
            initial,
            requested_count,
            family_support,
            eligible_family_count,
            False,
        )
    selected = max(
        choices,
        key=lambda item: (float(item.candidate.score), item.candidate.variant),
    )
    return _TemplateCountDecision(
        selected,
        requested_count,
        family_support,
        eligible_family_count,
        selected is not initial,
    )


def measure_signature(measure: etree._Element) -> str:
    """Return the canonical full-preservation signature for one measure.

    Whole-measure replacement copies objects beyond Score IR (for example beams,
    noteheads, fermatas and unknown notations), so the exact gate must cover the full
    write-back surface.  Sequence loading uses inherited-divisions-aware signatures.
    """

    return measure_preservation_signature(measure)


def _canonical_attribute_entries(
    measure: etree._Element,
    inherited_divisions: int,
) -> Counter[tuple[str, bytes]]:
    """Return canonical, layout-insensitive attribute children for one measure."""

    entries: Counter[tuple[str, bytes]] = Counter()
    for attributes in measure.findall("attributes"):
        for child in attributes:
            if not isinstance(child.tag, str):
                continue
            tag = etree.QName(child).localname
            if tag == "divisions":
                continue
            wrapper = etree.Element("measure")
            wrapper_attributes = etree.SubElement(wrapper, "attributes")
            wrapper_attributes.append(copy.deepcopy(child))
            payload, _ = canonical_measure_bytes(
                wrapper,
                inherited_divisions=inherited_divisions,
                include_attributes=True,
            )
            entries[(tag, payload)] += 1
    return entries


def _redundant_state_attribute_only_disagreement(
    members: list[tuple["_LoadedCandidate", etree._Element, MeasureIR, str]],
) -> bool:
    """Whether candidates differ only by repeated inherited key/clef declarations.

    System-localised OMR commonly emits the current key and clef again at the first
    measure of every physical system.  Those declarations are semantically redundant,
    but a full-preservation signature must still expose them for audit.  This narrow
    classifier suppresses a user-facing doubt only when:

    * all notation outside ``attributes`` is canonically identical;
    * the complete modelled measure state is identical; and
    * the only differing attribute children are ``key`` or ``clef``.

    Differences in notes, beams, slurs, directions, staff details, transposition,
    time signatures, or unknown MusicXML remain actionable disagreements.
    """

    if len(members) < 2:
        return False
    if len({semantic.fingerprint for _, _, semantic, _ in members}) != 1:
        return False

    content_signatures: set[bytes] = set()
    attribute_entries: list[Counter[tuple[str, bytes]]] = []
    for _, measure, semantic, _ in members:
        payload, _ = canonical_measure_bytes(
            measure,
            inherited_divisions=max(1, semantic.divisions),
            include_attributes=False,
        )
        content_signatures.add(payload)
        attribute_entries.append(
            _canonical_attribute_entries(measure, max(1, semantic.divisions))
        )
    if len(content_signatures) != 1:
        return False

    baseline = attribute_entries[0]
    differing_tags: set[str] = set()
    for entries in attribute_entries[1:]:
        for tag, _ in (baseline - entries):
            differing_tags.add(tag)
        for tag, _ in (entries - baseline):
            differing_tags.add(tag)
    return bool(differing_tags) and differing_tags <= {"key", "clef"}


MAX_CONSENSUS_CANDIDATE_BYTES = 64 * 1024 * 1024
MAX_CONSENSUS_PAGE_MEASURES = 512
MAX_CONSENSUS_PAGE_EVENTS = 20_000


def _load_candidate(candidate: CandidateLike) -> _LoadedCandidate | None:
    if not candidate.xml_path:
        return None
    path = Path(candidate.xml_path)
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size < 100 or size > MAX_CONSENSUS_CANDIDATE_BYTES:
        return None
    try:
        parser = etree.XMLParser(remove_blank_text=True, resolve_entities=False, no_network=True)
        tree = etree.parse(str(path), parser)
        part = tree.getroot().find("part")
        if part is None:
            return None
        measures = tuple(part.findall("measure"))
        if not measures or len(measures) > MAX_CONSENSUS_PAGE_MEASURES:
            return None
        state: dict[str, object] = {}
        semantics: list[MeasureIR] = []
        event_count = 0
        for measure in measures:
            semantic, state = measure_from_xml(measure, state)
            event_count += len(semantic.notes)
            if event_count > MAX_CONSENSUS_PAGE_EVENTS:
                return None
            semantics.append(semantic)
        return _LoadedCandidate(
            candidate=candidate,
            tree=tree,
            measures=measures,
            semantics=tuple(semantics),
            signatures=measure_preservation_signatures(measures),
        )
    except (OSError, etree.XMLSyntaxError, ValueError):
        return None


def _aligned_similarity(left: tuple[MeasureIR, ...], right: tuple[MeasureIR, ...]) -> float:
    alignment = align_measure_sequences(left, right)
    if not left and not right:
        return 1.0
    matched = [pair for pair in alignment.pairs if pair.reference_index is not None and pair.candidate_index is not None]
    if not matched:
        return 0.0
    semantic = sum(1.0 - min(1.0, pair.cost) for pair in matched) / max(len(left), len(right), 1)
    coverage = len(matched) / max(len(left), len(right), 1)
    return max(0.0, min(1.0, 0.72 * semantic + 0.28 * coverage))


def semantic_agreement(candidates: Iterable[CandidateLike]) -> dict[str, float]:
    """Return per-variant semantic agreement, tolerating one-off measure gaps.

    Earlier versions discarded every candidate whose measure count differed from its
    peers.  A single inserted or omitted measure could therefore erase useful evidence
    for the rest of the page.  Global alignment preserves that evidence while still
    penalising gaps.
    """
    loaded = [item for item in (_load_candidate(candidate) for candidate in candidates) if item is not None]
    agreement: dict[str, float] = {item.candidate.variant: 0.0 for item in loaded}
    comparisons: dict[str, int] = {item.candidate.variant: 0 for item in loaded}
    for left_index, left in enumerate(loaded):
        for right in loaded[left_index + 1:]:
            similarity = _aligned_similarity(left.semantics, right.semantics)
            agreement[left.candidate.variant] += similarity
            agreement[right.candidate.variant] += similarity
            comparisons[left.candidate.variant] += 1
            comparisons[right.candidate.variant] += 1
    for variant in agreement:
        if comparisons[variant]:
            agreement[variant] /= comparisons[variant]
    return agreement

def _candidate_weight(candidate: CandidateLike, median_score: float, local_factor: float = 1.0) -> float:
    # Bounded quality prior: ensemble evidence dominates, while structurally invalid
    # candidates can never acquire equal influence merely because they agree.
    centred = max(-5.0, min(5.0, (float(candidate.score) - median_score) / 100.0))
    logistic = 1.0 / (1.0 + math.exp(-centred))
    validity = 1.0 if bool(candidate.valid) else 0.35
    return validity * (0.55 + logistic) * max(0.5, min(1.5, local_factor))


def _committed_family_support(
    member_variants: list[str],
    selected_indices: Iterable[int],
    eligible_family_sizes: dict[str, int],
    healthy_families: set[str] | None = None,
) -> set[str]:
    """Return families whose complete eligible membership supports one decision.

    A family abstains when one sibling is missing at the aligned measure or when any
    aligned sibling falls outside the selected exact signature/semantic cluster.  This
    keeps the deterministic family count consistent with the already family-balanced
    candidate weights and prevents a single agreeable sibling from hiding a conflicting
    sibling in the same preprocessing family.
    """
    selected = set(selected_indices)
    member_indices: dict[str, list[int]] = {}
    for index, variant in enumerate(member_variants):
        member_indices.setdefault(variant_family(variant), []).append(index)
    allowed = healthy_families if healthy_families is not None else set(member_indices)
    return {
        family
        for family, indices in member_indices.items()
        if family in allowed
        and len(indices) == eligible_family_sizes.get(family, len(indices))
        and all(index in selected for index in indices)
    }


def _semantic_cluster(
    members: list[tuple[_LoadedCandidate, etree._Element, MeasureIR, str]],
    median_score: float,
    total_eligible_weight: float | None = None,
    threshold: float = DEFAULT_POLICY.semantic_cluster_threshold,
    local_factors: list[float] | None = None,
    eligible_family_sizes: dict[str, int] | None = None,
) -> tuple[list[int], int, float, float, list[list[float]], list[float]]:
    factors = local_factors or [1.0] * len(members)
    family_sizes = eligible_family_sizes or {
        variant_family(item[0].candidate.variant): sum(
            variant_family(other[0].candidate.variant) == variant_family(item[0].candidate.variant)
            for other in members
        )
        for item in members
    }
    def balanced_weight(index: int, local_factor: float = 1.0) -> float:
        item = members[index]
        family = variant_family(item[0].candidate.variant)
        return _candidate_weight(item[0].candidate, median_score, local_factor) / max(1, family_sizes.get(family, 1))

    if len(members) == 1:
        weights = [balanced_weight(0, factors[0])]
        return [0], 0, 0.0, 1.0, [[0.0]], weights
    base_weights = [balanced_weight(index) for index in range(len(members))]
    weights = [balanced_weight(index, factors[index]) for index in range(len(members))]
    distances = [
        [measure_distance(left[2], right[2]) for right in members]
        for left in members
    ]
    member_weight = sum(base_weights)
    total_weight = max(total_eligible_weight or member_weight, member_weight, 1e-9)
    choices: list[tuple[float, int, float, int, list[int]]] = []
    for centre_index in range(len(members)):
        cluster = [index for index, distance in enumerate(distances[centre_index]) if distance <= threshold]
        cluster_weight = sum(weights[index] for index in cluster)
        mean_distance = (
            sum(weights[index] * distances[centre_index][index] for index in cluster) / max(cluster_weight, 1e-9)
        )
        score = float(members[centre_index][0].candidate.score)
        choices.append((cluster_weight, len(cluster), -mean_distance, int(score), cluster))
    _, _, _, _, best_cluster = max(choices)
    # Select a weighted medoid within the winning cluster rather than its arbitrary centre.
    medoid_index = min(
        best_cluster,
        key=lambda centre: (
            sum(weights[index] * distances[centre][index] for index in best_cluster) / max(sum(weights[index] for index in best_cluster), 1e-9),
            -float(members[centre][0].candidate.score),
            members[centre][0].candidate.variant,
        ),
    )
    selection_cluster_weight = sum(weights[index] for index in best_cluster)
    mean_distance = sum(weights[index] * distances[medoid_index][index] for index in best_cluster) / max(selection_cluster_weight, 1e-9)
    support_cluster_weight = sum(base_weights[index] for index in best_cluster)
    support_ratio = support_cluster_weight / max(total_weight, 1e-9)
    return best_cluster, medoid_index, mean_distance, support_ratio, distances, weights


def build_measure_consensus(
    candidates: Iterable[CandidateLike],
    output_path: Path,
    template_variant: str | None = None,
    visual_evidence: tuple[VisualMeasureEvidence, ...] = (),
    target_measure_count: int = 0,
) -> ConsensusReport | None:
    """Fuse OMR candidates after semantic sequence alignment.

    A page-level template preserves pagination and stable measure numbering.  Every
    other candidate is globally aligned to that template, so one inserted or omitted
    measure no longer shifts all subsequent votes.  Exact majorities remain strongest;
    fuzzy semantic clusters are allowed only with high weighted support and low internal
    distance.  Missing aligned measures count against support rather than disappearing
    from the denominator.
    """
    candidate_list = [candidate for candidate in candidates if candidate.xml_path]
    loaded = [item for item in (_load_candidate(candidate) for candidate in candidate_list) if item is not None]
    if not loaded:
        return None

    template_entry = next((item for item in loaded if item.candidate.variant == template_variant), None)
    if template_entry is None:
        template_entry = max(loaded, key=lambda item: (bool(item.candidate.valid), float(item.candidate.score)))
    template_count_decision = _select_count_consistent_template(
        loaded,
        template_entry,
        target_measure_count,
    )
    template_entry = template_count_decision.entry
    template_candidate = template_entry.candidate

    alignments: dict[str, SequenceAlignment] = {}
    eligible: list[_LoadedCandidate] = []
    alignment_report: dict[str, dict[str, object]] = {}
    for item in loaded:
        if item is template_entry:
            alignment = align_measure_sequences(template_entry.semantics, template_entry.semantics)
        else:
            alignment = align_measure_sequences(template_entry.semantics, item.semantics)
        alignments[item.candidate.variant] = alignment
        missing = sum(value is None for value in alignment.reference_to_candidate)
        extra = len(alignment.unmatched_candidate_indices)
        alignment_report[item.candidate.variant] = {
            "similarity": alignment.similarity,
            "normalized_cost": alignment.normalized_cost,
            "missing_template_measures": missing,
            "extra_candidate_measures": extra,
            "candidate_measure_count": len(item.measures),
        }
        # Very weak alignments are more likely to represent a catastrophic page parse
        # than a useful alternative.  Keep the template unconditionally and otherwise
        # require either reasonable similarity or only a very small count discrepancy.
        count_gap = abs(len(item.measures) - len(template_entry.measures))
        allowed_gap = max(1, round(len(template_entry.measures) * DEFAULT_POLICY.allowed_measure_gap_ratio))
        if item is template_entry or alignment.similarity >= DEFAULT_POLICY.alignment_min_similarity or count_gap <= allowed_gap:
            eligible.append(item)
    if template_entry not in eligible:
        eligible.insert(0, template_entry)

    median_score = statistics.median(float(item.candidate.score) for item in eligible)
    eligible_family_sizes: dict[str, int] = {}
    family_validity: dict[str, list[bool]] = {}
    for item in eligible:
        family = variant_family(item.candidate.variant)
        eligible_family_sizes[family] = eligible_family_sizes.get(family, 0) + 1
        family_validity.setdefault(family, []).append(bool(item.candidate.valid))
    # A correlated preprocessing family is an all-or-nothing source of deterministic
    # support.  Parsable invalid siblings remain low-weight diagnostics, but one invalid
    # member makes the whole family abstain from majority and patch decisions.
    healthy_families = {
        family for family, states in family_validity.items() if states and all(states)
    }
    abstaining_families = tuple(sorted(set(eligible_family_sizes) - healthy_families))
    total_eligible_weight = sum(
        _candidate_weight(item.candidate, median_score)
        / max(1, eligible_family_sizes.get(variant_family(item.candidate.variant), 1))
        for item in eligible
    )

    result_tree = copy.deepcopy(template_entry.tree)
    result_part = result_tree.getroot().find("part")
    if result_part is None:
        return None
    result_measures = result_part.findall("measure")
    votes: list[MeasureVote] = []
    replacements = 0
    unanimous_count = 0
    majority_count = 0
    preservation_disagreement: list[int] = []
    resolved_disagreement: list[int] = []
    disagreement: list[int] = []
    unresolved: list[int] = []
    semantic_similarities: list[float] = []
    confidences: list[float] = []
    selected_measure_probabilities: list[float] = []
    selected_visual_probabilities: list[float] = []
    selected_event_probabilities: list[float] = []
    selected_context_probabilities: list[float] = []
    selected_ensemble_probabilities: list[float] = []
    selected_selection_risk_probabilities: list[float] = []
    chord_patch_probabilities: list[float] = []
    chord_patch_measure_count = 0
    chord_patch_event_count = 0
    tuplet_patch_probabilities: list[float] = []
    tuplet_patch_measure_count = 0
    tuplet_patch_event_count = 0
    tuplet_patch_group_count = 0
    pitch_patch_probabilities: list[float] = []
    pitch_patch_measure_count = 0
    pitch_patch_event_count = 0
    rhythm_patch_probabilities: list[float] = []
    rhythm_patch_measure_count = 0
    rhythm_patch_event_count = 0
    event_kind_patch_probabilities: list[float] = []
    event_kind_patch_measure_count = 0
    event_kind_patch_event_count = 0
    event_kind_visual_guard_probabilities: list[float] = []
    event_kind_visual_guard_transaction_count = 0
    event_kind_visual_guard_rejected_count = 0
    attribute_patch_probabilities: list[float] = []
    attribute_patch_measure_count = 0
    attribute_patch_attribute_count = 0
    barline_patch_probabilities: list[float] = []
    barline_patch_measure_count = 0
    barline_patch_location_count = 0
    barline_patch_repeat_count = 0
    tie_patch_probabilities: list[float] = []
    tie_patch_measure_count = 0
    tie_patch_event_count = 0
    tie_visual_guard_probabilities: list[float] = []
    tie_visual_guard_transaction_count = 0
    tie_visual_guard_rejected_count = 0
    slur_patch_probabilities: list[float] = []
    slur_patch_measure_count = 0
    slur_patch_event_count = 0
    slur_patch_arc_count = 0
    articulation_patch_probabilities: list[float] = []
    articulation_patch_measure_count = 0
    articulation_patch_event_count = 0
    articulation_patch_mark_count = 0
    accent_visual_guard_probabilities: list[float] = []
    accent_visual_guard_transaction_count = 0
    accent_visual_guard_rejected_count = 0
    ornament_patch_probabilities: list[float] = []
    ornament_patch_measure_count = 0
    ornament_patch_event_count = 0
    ornament_patch_mark_count = 0
    grace_patch_probabilities: list[float] = []
    grace_patch_measure_count = 0
    grace_patch_event_count = 0
    grace_patch_added_count = 0
    grace_patch_removed_count = 0
    lyric_patch_probabilities: list[float] = []
    lyric_patch_measure_count = 0
    lyric_patch_event_count = 0
    lyric_patch_lyric_count = 0
    direction_patch_probabilities: list[float] = []
    direction_patch_measure_count = 0
    direction_patch_direction_count = 0
    cross_tie_patch_probabilities: list[float] = []
    cross_tie_patch_boundary_count = 0
    cross_tie_patch_endpoint_count = 0
    cross_tie_patch_transaction_rejected_count = 0
    cross_tie_boundaries: list[dict[str, object]] = []
    measure_quality_by_index: list[dict[str, tuple[float, float, float, float, float]]] = [
        {} for _ in result_measures
    ]
    patch_transaction_probabilities: list[float] = []
    patch_stage_rejected_count = 0
    patch_transaction_evaluated_count = 0
    patch_transaction_rejected_count = 0
    event_presence_patch_probabilities: list[float] = []
    event_presence_patch_measure_count = 0
    event_presence_patch_inserted_event_count = 0
    event_presence_patch_deleted_event_count = 0
    event_presence_visual_guard_probabilities: list[float] = []
    event_presence_visual_guard_transaction_count = 0
    event_presence_visual_guard_rejected_count = 0
    measure_calibrator = MeasureCalibrator()
    visual_calibrator = VisualMeasureCalibrator()
    event_calibrator = EventCalibrator()
    context_calibrator = ContextCalibrator()
    ensemble_calibrator = EnsembleCalibrator()
    selection_risk_calibrator = SelectionRiskCalibrator()
    chord_patch_calibrator = ChordPatchCalibrator()
    tuplet_patch_calibrator = TupletPatchCalibrator()
    pitch_patch_calibrator = PitchPatchCalibrator()
    rhythm_patch_calibrator = RhythmPatchCalibrator()
    event_kind_patch_calibrator = EventKindPatchCalibrator()
    event_kind_visual_guard = EventKindVisualGuard()
    attribute_patch_calibrator = AttributePatchCalibrator()
    barline_patch_calibrator = BarlinePatchCalibrator()
    tie_patch_calibrator = TiePatchCalibrator()
    tie_visual_guard = TieVisualGuard()
    slur_patch_calibrator = SlurPatchCalibrator()
    articulation_patch_calibrator = ArticulationPatchCalibrator()
    accent_visual_guard = AccentVisualGuard()
    ornament_patch_calibrator = OrnamentPatchCalibrator()
    grace_patch_calibrator = GracePatchCalibrator()
    lyric_patch_calibrator = (
        LyricPatchCalibrator()
        if DEFAULT_POLICY.semantic_lyric_output_enabled
        else None
    )
    direction_patch_calibrator = DirectionPatchCalibrator()
    cross_tie_patch_calibrator = CrossTiePatchCalibrator()
    patch_transaction_calibrator = PatchTransactionCalibrator()
    event_presence_patch_calibrator = EventPresencePatchCalibrator()
    event_presence_visual_guard = EventPresenceVisualGuard()

    for measure_index in range(len(result_measures)):
        measure_number = measure_index + 1
        applicable = [
            item for item in eligible
            if candidate_applies_to_measure(item.candidate.variant, measure_number)
        ]
        members: list[tuple[_LoadedCandidate, etree._Element, MeasureIR, str]] = []
        for item in applicable:
            candidate_index = alignments[item.candidate.variant].reference_to_candidate[measure_index]
            if candidate_index is None:
                continue
            members.append(
                (
                    item,
                    item.measures[candidate_index],
                    item.semantics[candidate_index],
                    item.signatures[candidate_index],
                )
            )
        if not members:
            unresolved.append(measure_number)
            continue

        template_index = next(
            index for index, member in enumerate(members)
            if member[0].candidate.variant == template_candidate.variant
        )
        missing_count = len(applicable) - len(members)
        measure_family_sizes: dict[str, int] = {}
        for item in applicable:
            family = variant_family(item.candidate.variant)
            measure_family_sizes[family] = measure_family_sizes.get(family, 0) + 1
        eligible_families = set(measure_family_sizes).intersection(healthy_families)
        measure_abstaining_families = tuple(
            sorted(set(measure_family_sizes) - eligible_families)
        )
        measure_total_eligible_weight = sum(
            _candidate_weight(item.candidate, median_score)
            / max(1, measure_family_sizes.get(variant_family(item.candidate.variant), 1))
            for item in applicable
        )
        groups: dict[str, list[int]] = {}
        for index, (_, _, _, signature) in enumerate(members):
            groups.setdefault(signature, []).append(index)
        member_variants = [member[0].candidate.variant for member in members]
        signature_family_votes = {
            signature: _committed_family_support(
                member_variants,
                indices,
                measure_family_sizes,
                eligible_families,
            )
            for signature, indices in groups.items()
        }
        ranked_groups = sorted(
            groups.items(),
            key=lambda group: (
                len(signature_family_votes.get(group[0], set())),
                len(group[1]),
                max(float(members[index][0].candidate.score) for index in group[1]),
                group[0],
            ),
            reverse=True,
        )
        _winning_signature, exact_indices = ranked_groups[0]
        exact_selected_index = max(exact_indices, key=lambda index: float(members[index][0].candidate.score))
        exact_support = len(exact_indices)
        exact_families = signature_family_votes.get(_winning_signature, set())
        unanimous = (
            exact_support == len(applicable)
            and missing_count == 0
            and not measure_abstaining_families
        )
        required_consensus_families = min(
            DEFAULT_POLICY.minimum_consensus_families,
            len(eligible_families),
        )
        strict_majority = bool(
            eligible_families
            and len(exact_families) > len(eligible_families) / 2
            and len(exact_families) >= required_consensus_families
        )
        if unanimous:
            unanimous_count += 1
        if strict_majority:
            majority_count += 1
        if len(groups) > 1 or missing_count:
            preservation_disagreement.append(measure_index + 1)
        redundant_state_attribute_only = (
            missing_count == 0
            and len(groups) > 1
            and _redundant_state_attribute_only_disagreement(members)
        )

        # First obtain an uncalibrated semantic cluster.  The learned measure model is
        # then evaluated from those ensemble statistics and used only as a bounded local
        # weight before the cluster is recomputed.
        initial_cluster, initial_medoid, initial_mean_distance, initial_support, initial_distances, _ = _semantic_cluster(
            members,
            median_score,
            total_eligible_weight=measure_total_eligible_weight,
        )
        exact_support_ratio = exact_support / max(len(applicable), 1)
        missing_ratio = missing_count / max(len(applicable), 1)
        measure_probabilities: list[float] = []
        visual_probabilities: list[float] = []
        event_probabilities: list[float] = []
        context_probabilities: list[float] = []
        base_local_factors: list[float] = []
        event_profiles = agreement_profiles(
            [member[2] for member in members],
            [variant_family(member[0].candidate.variant) for member in members],
        )
        member_families = [variant_family(member[0].candidate.variant) for member in members]
        member_previous: list[MeasureIR | None] = []
        member_following: list[MeasureIR | None] = []
        for member in members:
            alignment_map = alignments[member[0].candidate.variant].reference_to_candidate
            previous_index = alignment_map[measure_index - 1] if measure_index > 0 else None
            following_index = (
                alignment_map[measure_index + 1]
                if measure_index + 1 < len(alignment_map)
                else None
            )
            member_previous.append(
                member[0].semantics[previous_index] if previous_index is not None else None
            )
            member_following.append(
                member[0].semantics[following_index] if following_index is not None else None
            )
        context_profiles = context_agreement_profiles(
            member_previous,
            [member[2] for member in members],
            member_following,
            member_families,
        )
        template_measure_ir = members[template_index][2]
        source_evidence = map_evidence_to_measure(visual_evidence, measure_index, len(result_measures))
        for member_index, member in enumerate(members):
            peer_distance = sum(initial_distances[member_index]) / max(len(members), 1)
            calibration = measure_calibrator.calibrate(
                MeasureCalibrationInput(
                    candidate=member[0].candidate,
                    measure=member[2],
                    alignment_similarity=alignments[member[0].candidate.variant].similarity,
                    exact_support_ratio=exact_support_ratio,
                    semantic_support_ratio=initial_support,
                    missing_ratio=missing_ratio,
                    distance_to_template=measure_distance(template_measure_ir, member[2]),
                    distance_to_medoid=measure_distance(members[initial_medoid][2], member[2]),
                    mean_peer_distance=peer_distance,
                    is_first_measure=measure_index == 0,
                    is_last_measure=measure_index + 1 == len(result_measures),
                )
            )
            visual = visual_calibrator.calibrate(source_evidence, member[2])
            event = event_calibrator.calibrate(event_profiles[member_index])
            context = context_calibrator.calibrate_profile(context_profiles[member_index])
            measure_probabilities.append(calibration.probability)
            visual_probabilities.append(visual.probability)
            event_probabilities.append(event.probability)
            context_probabilities.append(context.probability)
            # All learned priors are bounded independently. Their product remains
            # secondary to strict semantic majority and hard structural validation.
            base_local_factors.append(
                calibration.weight_factor
                * visual.weight_factor
                * event.weight_factor
                * context.weight_factor
            )
        def probability_margin(values: list[float], index: int) -> float:
            others = [value for other_index, value in enumerate(values) if other_index != index]
            return values[index] - (max(others) if others else 0.5)

        ensemble_probabilities: list[float] = []
        local_factors: list[float] = []
        best_page_score = max(float(member[0].candidate.score) for member in members)
        best_alignment = max(alignments[member[0].candidate.variant].similarity for member in members)
        for member_index, member in enumerate(members):
            signature_support = len(groups[member[3]]) / max(len(applicable), 1)
            ensemble = ensemble_calibrator.calibrate(
                EnsembleCalibrationInput(
                    page_score=float(member[0].candidate.score),
                    page_probability=float(getattr(member[0].candidate, "calibrated_probability", 0.5)),
                    page_valid=bool(member[0].candidate.valid),
                    alignment_similarity=alignments[member[0].candidate.variant].similarity,
                    alignment_margin=alignments[member[0].candidate.variant].similarity - best_alignment,
                    exact_support_ratio=exact_support_ratio,
                    semantic_support_ratio=initial_support,
                    signature_support_ratio=signature_support,
                    missing_ratio=missing_ratio,
                    distance_to_template=measure_distance(template_measure_ir, member[2]),
                    distance_to_medoid=measure_distance(members[initial_medoid][2], member[2]),
                    mean_peer_distance=sum(initial_distances[member_index]) / max(len(members), 1),
                    measure_probability=measure_probabilities[member_index],
                    visual_probability=visual_probabilities[member_index],
                    event_probability=event_probabilities[member_index],
                    context_probability=context_probabilities[member_index],
                    measure_probability_margin=probability_margin(measure_probabilities, member_index),
                    visual_probability_margin=probability_margin(visual_probabilities, member_index),
                    event_probability_margin=probability_margin(event_probabilities, member_index),
                    context_probability_margin=probability_margin(context_probabilities, member_index),
                    page_score_margin=float(member[0].candidate.score) - best_page_score,
                    candidate_count=len(members),
                    initial_cluster_member=member_index in initial_cluster,
                    exact_signature_member=member_index in exact_indices,
                )
            )
            ensemble_probabilities.append(ensemble.probability)
            local_factors.append(base_local_factors[member_index] * ensemble.weight_factor)

        for member_index, member in enumerate(members):
            measure_quality_by_index[measure_index][member[0].candidate.variant] = (
                measure_probabilities[member_index],
                visual_probabilities[member_index],
                event_probabilities[member_index],
                context_probabilities[member_index],
                ensemble_probabilities[member_index],
            )

        cluster, medoid_index, mean_distance, support_ratio, distances, _weights = _semantic_cluster(
            members,
            median_score,
            total_eligible_weight=measure_total_eligible_weight,
            local_factors=local_factors,
            eligible_family_sizes=measure_family_sizes,
        )
        cluster_variants = tuple(sorted(members[index][0].candidate.variant for index in cluster))
        cluster_families = _committed_family_support(
            member_variants,
            cluster,
            measure_family_sizes,
            eligible_families,
        )
        family_support_ratio = len(cluster_families) / max(len(eligible_families), 1)
        fuzzy_consensus = (
            len(cluster) >= min(DEFAULT_POLICY.semantic_min_candidates, len(applicable))
            and support_ratio >= DEFAULT_POLICY.semantic_support_min
            and mean_distance <= DEFAULT_POLICY.semantic_distance_max
            and len(cluster_families) >= min(DEFAULT_POLICY.minimum_consensus_families, len(eligible_families))
            and family_support_ratio >= DEFAULT_POLICY.semantic_family_support_min
            and missing_count <= max(1, round(len(applicable) * DEFAULT_POLICY.max_missing_candidate_fraction))
        )
        if strict_majority:
            selected_index = exact_selected_index
            selection_kind = "exact_majority"
        elif fuzzy_consensus:
            selected_index = medoid_index
            selection_kind = "semantic_consensus"
        else:
            selected_index = template_index
            selection_kind = "template"

        selected_loaded, selected_measure, selected_semantics, selected_signature = members[selected_index]
        selected_measure_probability = measure_probabilities[selected_index]
        selected_visual_probability = visual_probabilities[selected_index]
        selected_event_probability = event_probabilities[selected_index]
        selected_context_probability = context_probabilities[selected_index]
        selected_ensemble_probability = ensemble_probabilities[selected_index]
        selected_signature_support_ratio = len(groups[selected_signature]) / max(len(applicable), 1)
        selected_preservation_families = signature_family_votes.get(selected_signature, set())
        selected_preservation_family_support = len(selected_preservation_families)
        preservation_gate_required = selection_kind == "semantic_consensus"
        preservation_gate_accepted = (
            not preservation_gate_required
            or selected_preservation_family_support
            >= min(
                DEFAULT_POLICY.selection_semantic_preservation_minimum_families,
                len(eligible_families),
            )
        )
        # All automatic measure replacements, including exact majorities, pass the
        # same replacement-benefit verifier.  Exact agreement remains the strongest
        # semantic evidence, but two correlated or independently failing families can
        # still agree on the same wrong measure.  The verifier may only veto a
        # replacement; it cannot create consensus or select another candidate.
        selection_risk_applicable = selection_kind in {"exact_majority", "semantic_consensus"}
        if selection_risk_applicable:
            selection_risk = selection_risk_calibrator.calibrate(
                SelectionRiskInput(
                    selection_kind=selection_kind,
                    selected_page_score=float(selected_loaded.candidate.score),
                    selected_page_probability=float(getattr(selected_loaded.candidate, "calibrated_probability", 0.5)),
                    selected_ensemble_probability=selected_ensemble_probability,
                    ensemble_probability_margin=probability_margin(ensemble_probabilities, selected_index),
                    selected_measure_probability=selected_measure_probability,
                    measure_probability_margin=probability_margin(measure_probabilities, selected_index),
                    selected_visual_probability=selected_visual_probability,
                    visual_probability_margin=probability_margin(visual_probabilities, selected_index),
                    selected_event_probability=selected_event_probability,
                    event_probability_margin=probability_margin(event_probabilities, selected_index),
                    selected_context_probability=selected_context_probability,
                    context_probability_margin=probability_margin(context_probabilities, selected_index),
                    exact_support_ratio=exact_support_ratio,
                    semantic_support_ratio=support_ratio,
                    signature_support_ratio=selected_signature_support_ratio,
                    missing_ratio=missing_ratio,
                    mean_cluster_distance=mean_distance,
                    template_distance=measure_distance(members[template_index][2], selected_semantics),
                    alignment_similarity=alignments[selected_loaded.candidate.variant].similarity,
                    alignment_margin=alignments[selected_loaded.candidate.variant].similarity - best_alignment,
                    selected_distance_to_medoid=measure_distance(members[medoid_index][2], selected_semantics),
                    selected_mean_peer_distance=sum(distances[selected_index]) / max(len(members), 1),
                    page_score_margin=float(selected_loaded.candidate.score) - best_page_score,
                    candidate_count=len(members),
                    exact_support_count=exact_support,
                    distinct_signature_count=len(groups),
                    top_signature_margin=(exact_support - (len(ranked_groups[1][1]) if len(ranked_groups) > 1 else 0)) / max(len(applicable), 1),
                    unanimous=unanimous,
                    strict_majority=strict_majority,
                    selected_is_template=selected_index == template_index,
                    selected_is_exact_signature=selected_index in exact_indices,
                    selected_in_initial_cluster=selected_index in initial_cluster,
                    page_valid=bool(selected_loaded.candidate.valid),
                    selected_vs_template_page_probability=(
                        float(getattr(selected_loaded.candidate, "calibrated_probability", 0.5))
                        - float(getattr(template_candidate, "calibrated_probability", 0.5))
                    ),
                    selected_vs_template_ensemble_probability=(
                        selected_ensemble_probability - ensemble_probabilities[template_index]
                    ),
                    selected_vs_template_measure_probability=(
                        selected_measure_probability - measure_probabilities[template_index]
                    ),
                    selected_vs_template_visual_probability=(
                        selected_visual_probability - visual_probabilities[template_index]
                    ),
                    selected_vs_template_event_probability=(
                        selected_event_probability - event_probabilities[template_index]
                    ),
                    selected_vs_template_context_probability=(
                        selected_context_probability - context_probabilities[template_index]
                    ),
                    selected_vs_template_alignment_similarity=(
                        alignments[selected_loaded.candidate.variant].similarity
                        - alignments[members[template_index][0].candidate.variant].similarity
                    ),
                    template_page_valid=bool(template_candidate.valid),
                    template_in_initial_cluster=template_index in initial_cluster,
                    template_is_exact_signature=template_index in exact_indices,
                    eligible_family_count=len(eligible_families),
                    exact_family_support_count=len(exact_families),
                    semantic_family_support_count=len(cluster_families),
                )
            )
            selected_selection_risk_probabilities.append(selection_risk.probability)
        else:
            selection_risk = None
        selected_measure_probabilities.append(selected_measure_probability)
        selected_visual_probabilities.append(selected_visual_probability)
        selected_event_probabilities.append(selected_event_probability)
        selected_context_probabilities.append(selected_context_probability)
        selected_ensemble_probabilities.append(selected_ensemble_probability)
        template_semantics = members[template_index][2]
        template_signature = members[template_index][3]
        template_distance = measure_distance(template_semantics, selected_semantics)
        max_template_distance = max(measure_distance(template_semantics, member[2]) for member in members)
        cluster_similarity = 1.0 - min(1.0, mean_distance)
        semantic_similarity = max(0.0, min(1.0, support_ratio * cluster_similarity))
        semantic_similarities.append(semantic_similarity)
        confidence = semantic_similarity * (len(members) / max(len(applicable), 1))
        confidences.append(confidence)

        selected_is_safe = (
            bool(selected_loaded.candidate.valid)
            and float(selected_loaded.candidate.score) >= float(template_candidate.score) - DEFAULT_POLICY.replacement_page_score_slack
            and alignments[selected_loaded.candidate.variant].similarity >= DEFAULT_POLICY.alignment_min_similarity
            and measure_probabilities[selected_index] >= DEFAULT_POLICY.replacement_measure_probability_floor
            and event_probabilities[selected_index] >= DEFAULT_POLICY.replacement_event_probability_floor
            and context_probabilities[selected_index] >= DEFAULT_POLICY.replacement_context_probability_floor
            and ensemble_probabilities[selected_index] >= DEFAULT_POLICY.replacement_ensemble_probability_floor
            and preservation_gate_accepted
            and selection_risk is not None
            and selection_risk.accepted
        )
        significant_difference = selected_signature != template_signature and template_distance > DEFAULT_POLICY.significant_measure_distance
        replace = significant_difference and selected_is_safe and selection_kind in {"exact_majority", "semantic_consensus"}
        patch_base: etree._Element | None = None
        patch_stage_rejections: list[str] = []
        patch_stage_inherited = {
            "divisions": template_semantics.divisions,
            "time": template_semantics.time_signature,
            "key": template_semantics.key_signature,
            "clef": template_semantics.clef,
        }

        def commit_patch_stage(
            kind: str,
            proposed: etree._Element | None,
            accepted: bool,
        ) -> bool:
            nonlocal patch_base
            if not accepted or proposed is None:
                return False
            validation = validate_patch_stage(
                result_measures[measure_index],
                patch_base,
                proposed,
                patch_stage_inherited,
            )
            if not validation.accepted:
                patch_stage_rejections.append(f"{kind}:{validation.reason}")
                return False
            patch_base = validation.measure
            return True

        chord_patch = ChordPatchResult(None, (), 0.5, chord_patch_calibrator.threshold, False, "not_applicable")
        chord_patch_applicable = not replace and len(eligible_families) >= DEFAULT_POLICY.chord_patch_minimum_families
        if chord_patch_applicable:
            chord_patch = propose_chord_patch(
                [
                    ChordPatchCandidate(
                        variant=member[0].candidate.variant,
                        family=variant_family(member[0].candidate.variant),
                        measure=member[1],
                        semantics=member[2],
                        page_score=float(member[0].candidate.score),
                        page_probability=float(getattr(member[0].candidate, "calibrated_probability", 0.5)),
                        measure_probability=measure_probabilities[index],
                        visual_probability=visual_probabilities[index],
                        event_probability=event_probabilities[index],
                        context_probability=context_probabilities[index],
                        ensemble_probability=ensemble_probabilities[index],
                        valid=bool(member[0].candidate.valid),
                    )
                    for index, member in enumerate(members)
                ],
                template_index=template_index,
                missing_candidate_count=missing_count,
                calibrator=chord_patch_calibrator,
            )
            if chord_patch.input is not None:
                chord_patch_probabilities.append(chord_patch.probability)

        chord_patch_applied = commit_patch_stage(
            "chord", chord_patch.patched_measure, chord_patch.accepted
        )
        tuplet_patch = TupletPatchResult(
            None, (), 0, 0.5, tuplet_patch_calibrator.threshold, False, "not_applicable"
        )
        tuplet_patch_applicable = (
            not replace
            and len(eligible_families) >= DEFAULT_POLICY.tuplet_patch_minimum_families
        )
        if tuplet_patch_applicable:
            tuplet_patch = propose_tuplet_patch(
                [
                    TupletPatchCandidate(
                        variant=member[0].candidate.variant,
                        family=variant_family(member[0].candidate.variant),
                        measure=member[1],
                        semantics=member[2],
                        page_score=float(member[0].candidate.score),
                        page_probability=float(getattr(member[0].candidate, "calibrated_probability", 0.5)),
                        measure_probability=measure_probabilities[index],
                        visual_probability=visual_probabilities[index],
                        event_probability=event_probabilities[index],
                        context_probability=context_probabilities[index],
                        ensemble_probability=ensemble_probabilities[index],
                        valid=bool(member[0].candidate.valid),
                    )
                    for index, member in enumerate(members)
                ],
                template_index=template_index,
                missing_candidate_count=missing_count,
                calibrator=tuplet_patch_calibrator,
                base_measure=patch_base,
            )
            if tuplet_patch.input is not None:
                tuplet_patch_probabilities.append(tuplet_patch.probability)

        tuplet_patch_applied = commit_patch_stage(
            "tuplet", tuplet_patch.patched_measure, tuplet_patch.accepted
        )
        pitch_patch = PitchPatchResult(None, (), 0.5, pitch_patch_calibrator.threshold, False, "not_applicable")
        pitch_patch_applicable = not replace and len(eligible_families) >= DEFAULT_POLICY.pitch_patch_minimum_families
        if pitch_patch_applicable:
            pitch_patch = propose_pitch_patch(
                [
                    PitchPatchCandidate(
                        variant=member[0].candidate.variant,
                        family=variant_family(member[0].candidate.variant),
                        measure=member[1],
                        semantics=member[2],
                        page_score=float(member[0].candidate.score),
                        page_probability=float(getattr(member[0].candidate, "calibrated_probability", 0.5)),
                        measure_probability=measure_probabilities[index],
                        visual_probability=visual_probabilities[index],
                        event_probability=event_probabilities[index],
                        context_probability=context_probabilities[index],
                        ensemble_probability=ensemble_probabilities[index],
                        valid=bool(member[0].candidate.valid),
                    )
                    for index, member in enumerate(members)
                ],
                template_index=template_index,
                missing_candidate_count=missing_count,
                calibrator=pitch_patch_calibrator,
                base_measure=patch_base,
                visual_evidence=source_evidence,
            )
            if pitch_patch.input is not None:
                pitch_patch_probabilities.append(pitch_patch.probability)

        pitch_patch_applied = commit_patch_stage(
            "pitch", pitch_patch.patched_measure, pitch_patch.accepted
        )
        rhythm_patch = RhythmPatchResult(None, (), 0.5, rhythm_patch_calibrator.threshold, False, "not_applicable")
        rhythm_patch_applicable = not replace and len(eligible_families) >= DEFAULT_POLICY.rhythm_patch_minimum_families
        if rhythm_patch_applicable:
            rhythm_patch = propose_rhythm_patch(
                [
                    RhythmPatchCandidate(
                        variant=member[0].candidate.variant,
                        family=variant_family(member[0].candidate.variant),
                        measure=member[1],
                        semantics=member[2],
                        page_score=float(member[0].candidate.score),
                        page_probability=float(getattr(member[0].candidate, "calibrated_probability", 0.5)),
                        measure_probability=measure_probabilities[index],
                        visual_probability=visual_probabilities[index],
                        event_probability=event_probabilities[index],
                        context_probability=context_probabilities[index],
                        ensemble_probability=ensemble_probabilities[index],
                        valid=bool(member[0].candidate.valid),
                    )
                    for index, member in enumerate(members)
                ],
                template_index=template_index,
                missing_candidate_count=missing_count,
                calibrator=rhythm_patch_calibrator,
                base_measure=patch_base,
                visual_evidence=source_evidence,
            )
            if rhythm_patch.input is not None:
                rhythm_patch_probabilities.append(rhythm_patch.probability)

        rhythm_patch_applied = commit_patch_stage(
            "rhythm", rhythm_patch.patched_measure, rhythm_patch.accepted
        )
        event_kind_patch = EventKindPatchResult(
            None, (), 0.5, event_kind_patch_calibrator.threshold, False, "not_applicable"
        )
        event_kind_patch_applicable = (
            not replace
            and len(eligible_families) >= DEFAULT_POLICY.event_kind_patch_minimum_families
        )
        if event_kind_patch_applicable:
            event_kind_patch = propose_event_kind_patch(
                [
                    EventKindPatchCandidate(
                        variant=member[0].candidate.variant,
                        family=variant_family(member[0].candidate.variant),
                        measure=member[1],
                        semantics=member[2],
                        page_score=float(member[0].candidate.score),
                        page_probability=float(getattr(member[0].candidate, "calibrated_probability", 0.5)),
                        measure_probability=measure_probabilities[index],
                        visual_probability=visual_probabilities[index],
                        event_probability=event_probabilities[index],
                        context_probability=context_probabilities[index],
                        ensemble_probability=ensemble_probabilities[index],
                        valid=bool(member[0].candidate.valid),
                    )
                    for index, member in enumerate(members)
                ],
                template_index=template_index,
                missing_candidate_count=missing_count,
                calibrator=event_kind_patch_calibrator,
                base_measure=patch_base,
            )
            if event_kind_patch.input is not None:
                event_kind_patch_probabilities.append(event_kind_patch.probability)

        event_kind_visual_audit = EventKindVisualAudit(
            applicable=False,
            changed_event_count=0,
            probability=0.5,
            threshold=round(event_kind_visual_guard.threshold, 6),
            accepted=True,
            reason="not_applicable",
            model_version=event_kind_visual_guard.model_version,
        )
        if event_kind_patch.accepted and event_kind_patch.patched_measure is not None:
            try:
                source_measure = patch_base if patch_base is not None else result_measures[measure_index]
                source_semantics, _ = measure_from_xml(
                    source_measure, dict(patch_stage_inherited)
                )
                patched_semantics, _ = measure_from_xml(
                    event_kind_patch.patched_measure, dict(patch_stage_inherited)
                )
                event_kind_visual_audit = event_kind_visual_guard.audit_transaction(
                    source_evidence, source_semantics, patched_semantics
                )
            except (TypeError, ValueError, ArithmeticError, etree.XMLSyntaxError):
                event_kind_visual_audit = EventKindVisualAudit(
                    applicable=True,
                    changed_event_count=len(event_kind_patch.changed_event_indices),
                    probability=0.5,
                    threshold=round(event_kind_visual_guard.threshold, 6),
                    accepted=False,
                    reason="event_kind_visual_audit_failed",
                    model_version=event_kind_visual_guard.model_version,
                )
            if event_kind_visual_audit.applicable:
                event_kind_visual_guard_transaction_count += 1
                event_kind_visual_guard_probabilities.append(event_kind_visual_audit.probability)
                if not event_kind_visual_audit.accepted:
                    event_kind_visual_guard_rejected_count += 1
                    event_kind_patch = dataclass_replace(
                        event_kind_patch,
                        accepted=False,
                        reason=f"{event_kind_patch.reason}:{event_kind_visual_audit.reason}",
                    )

        event_kind_patch_applied = commit_patch_stage(
            "event_kind", event_kind_patch.patched_measure, event_kind_patch.accepted
        )
        attribute_patch = AttributePatchResult(
            None, (), 0.5, attribute_patch_calibrator.threshold, False, "not_applicable"
        )
        attribute_patch_applicable = (
            not replace
            and len(eligible_families) >= DEFAULT_POLICY.attribute_patch_minimum_families
        )
        if attribute_patch_applicable:
            attribute_patch = propose_attribute_patch(
                [
                    AttributePatchCandidate(
                        variant=member[0].candidate.variant,
                        family=variant_family(member[0].candidate.variant),
                        measure=member[1],
                        semantics=member[2],
                        previous_semantics=member_previous[index],
                        following_semantics=member_following[index],
                        page_score=float(member[0].candidate.score),
                        page_probability=float(getattr(member[0].candidate, "calibrated_probability", 0.5)),
                        measure_probability=measure_probabilities[index],
                        visual_probability=visual_probabilities[index],
                        event_probability=event_probabilities[index],
                        context_probability=context_probabilities[index],
                        ensemble_probability=ensemble_probabilities[index],
                        valid=bool(member[0].candidate.valid),
                    )
                    for index, member in enumerate(members)
                ],
                template_index=template_index,
                missing_candidate_count=missing_count,
                is_first_measure=measure_index == 0,
                is_last_measure=measure_index + 1 == len(result_measures),
                calibrator=attribute_patch_calibrator,
                base_measure=patch_base,
            )
            if any(decision.input is not None for decision in attribute_patch.decisions):
                attribute_patch_probabilities.append(attribute_patch.probability)

        attribute_patch_applied = commit_patch_stage(
            "attribute", attribute_patch.patched_measure, attribute_patch.accepted
        )
        tie_patch = TiePatchResult(
            None, (), 0.5, tie_patch_calibrator.threshold, False, "not_applicable"
        )
        tie_patch_applicable = (
            not replace
            and len(eligible_families) >= DEFAULT_POLICY.tie_patch_minimum_families
        )
        if tie_patch_applicable:
            tie_patch = propose_tie_patch(
                [
                    TiePatchCandidate(
                        variant=member[0].candidate.variant,
                        family=variant_family(member[0].candidate.variant),
                        measure=member[1],
                        semantics=member[2],
                        page_score=float(member[0].candidate.score),
                        page_probability=float(getattr(member[0].candidate, "calibrated_probability", 0.5)),
                        measure_probability=measure_probabilities[index],
                        visual_probability=visual_probabilities[index],
                        event_probability=event_probabilities[index],
                        context_probability=context_probabilities[index],
                        ensemble_probability=ensemble_probabilities[index],
                        valid=bool(member[0].candidate.valid),
                    )
                    for index, member in enumerate(members)
                ],
                template_index=template_index,
                missing_candidate_count=missing_count,
                calibrator=tie_patch_calibrator,
                base_measure=patch_base,
            )
            if tie_patch.input is not None:
                tie_patch_probabilities.append(tie_patch.probability)

        tie_visual_audit = TieVisualAudit(
            applicable=False,
            changed_tie_count=0,
            probability=0.5,
            threshold=round(tie_visual_guard.threshold, 6),
            accepted=True,
            reason="not_applicable",
            model_version=tie_visual_guard.model_version,
        )
        if tie_patch.accepted and tie_patch.patched_measure is not None:
            try:
                source_measure = patch_base if patch_base is not None else result_measures[measure_index]
                source_semantics, _ = measure_from_xml(
                    source_measure, dict(patch_stage_inherited)
                )
                patched_semantics, _ = measure_from_xml(
                    tie_patch.patched_measure, dict(patch_stage_inherited)
                )
                tie_visual_audit = tie_visual_guard.audit_transaction(
                    source_evidence, source_semantics, patched_semantics
                )
            except (TypeError, ValueError, ArithmeticError, etree.XMLSyntaxError):
                tie_visual_audit = TieVisualAudit(
                    applicable=True,
                    changed_tie_count=len(tie_patch.changed_event_indices) // 2,
                    probability=0.5,
                    threshold=round(tie_visual_guard.threshold, 6),
                    accepted=False,
                    reason="visual_tie_audit_failed",
                    model_version=tie_visual_guard.model_version,
                )
            if tie_visual_audit.applicable:
                tie_visual_guard_transaction_count += 1
                tie_visual_guard_probabilities.append(tie_visual_audit.probability)
                if not tie_visual_audit.accepted:
                    tie_visual_guard_rejected_count += 1
                    tie_patch = dataclass_replace(
                        tie_patch,
                        accepted=False,
                        reason=f"{tie_patch.reason}:{tie_visual_audit.reason}",
                    )

        tie_patch_applied = commit_patch_stage(
            "tie", tie_patch.patched_measure, tie_patch.accepted
        )
        slur_patch = SlurPatchResult(
            None, (), 0, 0.5, slur_patch_calibrator.threshold, False, "not_applicable"
        )
        slur_patch_applicable = (
            not replace
            and len(eligible_families) >= DEFAULT_POLICY.slur_patch_minimum_families
        )
        if slur_patch_applicable:
            slur_patch = propose_slur_patch(
                [
                    SlurPatchCandidate(
                        variant=member[0].candidate.variant,
                        family=variant_family(member[0].candidate.variant),
                        measure=member[1],
                        semantics=member[2],
                        page_score=float(member[0].candidate.score),
                        page_probability=float(getattr(member[0].candidate, "calibrated_probability", 0.5)),
                        measure_probability=measure_probabilities[index],
                        visual_probability=visual_probabilities[index],
                        event_probability=event_probabilities[index],
                        context_probability=context_probabilities[index],
                        ensemble_probability=ensemble_probabilities[index],
                        valid=bool(member[0].candidate.valid),
                    )
                    for index, member in enumerate(members)
                ],
                template_index=template_index,
                missing_candidate_count=missing_count,
                calibrator=slur_patch_calibrator,
                base_measure=patch_base,
            )
            if slur_patch.input is not None:
                slur_patch_probabilities.append(slur_patch.probability)

        slur_patch_applied = commit_patch_stage(
            "slur", slur_patch.patched_measure, slur_patch.accepted
        )
        articulation_patch = ArticulationPatchResult(
            None, (), 0, 0.5, articulation_patch_calibrator.threshold, False, "not_applicable"
        )
        articulation_patch_applicable = (
            not replace
            and len(eligible_families) >= DEFAULT_POLICY.articulation_patch_minimum_families
        )
        if articulation_patch_applicable:
            articulation_patch = propose_articulation_patch(
                [
                    ArticulationPatchCandidate(
                        variant=member[0].candidate.variant,
                        family=variant_family(member[0].candidate.variant),
                        measure=member[1],
                        semantics=member[2],
                        page_score=float(member[0].candidate.score),
                        page_probability=float(getattr(member[0].candidate, "calibrated_probability", 0.5)),
                        measure_probability=measure_probabilities[index],
                        visual_probability=visual_probabilities[index],
                        event_probability=event_probabilities[index],
                        context_probability=context_probabilities[index],
                        ensemble_probability=ensemble_probabilities[index],
                        valid=bool(member[0].candidate.valid),
                    )
                    for index, member in enumerate(members)
                ],
                template_index=template_index,
                missing_candidate_count=missing_count,
                calibrator=articulation_patch_calibrator,
                base_measure=patch_base,
            )
            if articulation_patch.input is not None:
                articulation_patch_probabilities.append(articulation_patch.probability)

        accent_visual_audit = AccentVisualAudit(
            applicable=False,
            changed_accent_count=0,
            probability=0.5,
            threshold=round(accent_visual_guard.threshold, 6),
            accepted=True,
            reason="not_applicable",
            model_version=accent_visual_guard.model_version,
        )
        if articulation_patch.accepted and articulation_patch.patched_measure is not None:
            try:
                source_measure = patch_base if patch_base is not None else result_measures[measure_index]
                source_semantics, _ = measure_from_xml(
                    source_measure, dict(patch_stage_inherited)
                )
                patched_semantics, _ = measure_from_xml(
                    articulation_patch.patched_measure, dict(patch_stage_inherited)
                )
                accent_visual_audit = accent_visual_guard.audit_transaction(
                    source_evidence, source_semantics, patched_semantics
                )
            except (TypeError, ValueError, ArithmeticError, etree.XMLSyntaxError):
                accent_visual_audit = AccentVisualAudit(
                    applicable=True,
                    changed_accent_count=0,
                    probability=0.5,
                    threshold=round(accent_visual_guard.threshold, 6),
                    accepted=False,
                    reason="visual_accent_audit_failed",
                    model_version=accent_visual_guard.model_version,
                )
            if accent_visual_audit.applicable:
                accent_visual_guard_transaction_count += 1
                accent_visual_guard_probabilities.append(accent_visual_audit.probability)
                if not accent_visual_audit.accepted:
                    accent_visual_guard_rejected_count += 1
                    articulation_patch = dataclass_replace(
                        articulation_patch,
                        accepted=False,
                        reason=f"{articulation_patch.reason}:{accent_visual_audit.reason}",
                    )

        articulation_patch_applied = commit_patch_stage(
            "articulation", articulation_patch.patched_measure, articulation_patch.accepted
        )
        ornament_patch = OrnamentPatchResult(
            None, (), 0, 0.5, ornament_patch_calibrator.threshold, False, "not_applicable"
        )
        ornament_patch_applicable = (
            not replace
            and len(eligible_families) >= DEFAULT_POLICY.ornament_patch_minimum_families
        )
        if ornament_patch_applicable:
            ornament_patch = propose_ornament_patch(
                [
                    OrnamentPatchCandidate(
                        variant=member[0].candidate.variant,
                        family=variant_family(member[0].candidate.variant),
                        measure=member[1],
                        semantics=member[2],
                        page_score=float(member[0].candidate.score),
                        page_probability=float(getattr(member[0].candidate, "calibrated_probability", 0.5)),
                        measure_probability=measure_probabilities[index],
                        visual_probability=visual_probabilities[index],
                        event_probability=event_probabilities[index],
                        context_probability=context_probabilities[index],
                        ensemble_probability=ensemble_probabilities[index],
                        valid=bool(member[0].candidate.valid),
                    )
                    for index, member in enumerate(members)
                ],
                template_index=template_index,
                missing_candidate_count=missing_count,
                calibrator=ornament_patch_calibrator,
                base_measure=patch_base,
            )
            if ornament_patch.input is not None:
                ornament_patch_probabilities.append(ornament_patch.probability)

        ornament_patch_applied = commit_patch_stage(
            "ornament", ornament_patch.patched_measure, ornament_patch.accepted
        )
        grace_patch = GracePatchResult(
            None, (), 0, 0, 0.5, grace_patch_calibrator.threshold, False, "not_applicable"
        )
        grace_patch_applicable = (
            not replace
            and not (
                chord_patch_applied
                or tuplet_patch_applied
                or pitch_patch_applied
                or rhythm_patch_applied
                or event_kind_patch_applied
                or attribute_patch_applied
                or tie_patch_applied
                or slur_patch_applied
                or articulation_patch_applied
                or ornament_patch_applied
            )
            and len(eligible_families) >= DEFAULT_POLICY.grace_patch_minimum_families
        )
        if grace_patch_applicable:
            grace_patch = propose_grace_patch(
                [
                    GracePatchCandidate(
                        variant=member[0].candidate.variant,
                        family=variant_family(member[0].candidate.variant),
                        measure=member[1],
                        semantics=member[2],
                        page_score=float(member[0].candidate.score),
                        page_probability=float(getattr(member[0].candidate, "calibrated_probability", 0.5)),
                        measure_probability=measure_probabilities[index],
                        visual_probability=visual_probabilities[index],
                        event_probability=event_probabilities[index],
                        context_probability=context_probabilities[index],
                        ensemble_probability=ensemble_probabilities[index],
                        valid=bool(member[0].candidate.valid),
                    )
                    for index, member in enumerate(members)
                ],
                template_index=template_index,
                missing_candidate_count=missing_count,
                calibrator=grace_patch_calibrator,
                base_measure=patch_base,
            )
            if grace_patch.input is not None:
                grace_patch_probabilities.append(grace_patch.probability)

        grace_patch_applied = commit_patch_stage(
            "grace", grace_patch.patched_measure, grace_patch.accepted
        )

        lyric_patch = LyricPatchResult(
            None,
            (),
            0,
            0.5,
            (
                lyric_patch_calibrator.threshold
                if lyric_patch_calibrator is not None
                else 1.0
            ),
            False,
            (
                "not_applicable"
                if DEFAULT_POLICY.semantic_lyric_output_enabled
                else "disabled_out_of_scope"
            ),
        )
        lyric_patch_applicable = (
            DEFAULT_POLICY.semantic_lyric_output_enabled
            and not replace
            and len(eligible_families) >= DEFAULT_POLICY.lyric_patch_minimum_families
        )
        if lyric_patch_applicable:
            if lyric_patch_calibrator is None:
                raise RuntimeError("enabled lyric patch has no calibrator")
            lyric_patch = propose_lyric_patch(
                [
                    LyricPatchCandidate(
                        variant=member[0].candidate.variant,
                        family=variant_family(member[0].candidate.variant),
                        measure=member[1],
                        semantics=member[2],
                        page_score=float(member[0].candidate.score),
                        page_probability=float(getattr(member[0].candidate, "calibrated_probability", 0.5)),
                        measure_probability=measure_probabilities[index],
                        visual_probability=visual_probabilities[index],
                        event_probability=event_probabilities[index],
                        context_probability=context_probabilities[index],
                        ensemble_probability=ensemble_probabilities[index],
                        valid=bool(member[0].candidate.valid),
                    )
                    for index, member in enumerate(members)
                ],
                template_index=template_index,
                missing_candidate_count=missing_count,
                calibrator=lyric_patch_calibrator,
                base_measure=patch_base,
            )
            if lyric_patch.input is not None:
                lyric_patch_probabilities.append(lyric_patch.probability)

        lyric_patch_applied = commit_patch_stage(
            "lyric", lyric_patch.patched_measure, lyric_patch.accepted
        )

        direction_patch = DirectionPatchResult(
            None, 0, (), 0.5, direction_patch_calibrator.threshold, False, "not_applicable"
        )
        direction_patch_applicable = (
            not replace
            and len(eligible_families) >= DEFAULT_POLICY.direction_patch_minimum_families
        )
        if direction_patch_applicable:
            direction_patch = propose_direction_patch(
                [
                    DirectionPatchCandidate(
                        variant=member[0].candidate.variant,
                        family=variant_family(member[0].candidate.variant),
                        measure=member[1],
                        semantics=member[2],
                        page_score=float(member[0].candidate.score),
                        page_probability=float(getattr(member[0].candidate, "calibrated_probability", 0.5)),
                        measure_probability=measure_probabilities[index],
                        visual_probability=visual_probabilities[index],
                        event_probability=event_probabilities[index],
                        context_probability=context_probabilities[index],
                        ensemble_probability=ensemble_probabilities[index],
                        valid=bool(member[0].candidate.valid),
                    )
                    for index, member in enumerate(members)
                ],
                template_index=template_index,
                missing_candidate_count=missing_count,
                calibrator=direction_patch_calibrator,
                base_measure=patch_base,
            )
            if direction_patch.input is not None:
                direction_patch_probabilities.append(direction_patch.probability)

        direction_patch_applied = commit_patch_stage(
            "direction", direction_patch.patched_measure, direction_patch.accepted
        )
        barline_patch = BarlinePatchResult(
            None, (), 0, 0.5, barline_patch_calibrator.threshold, False, "not_applicable"
        )
        barline_patch_applicable = (
            not replace
            and len(eligible_families) >= DEFAULT_POLICY.barline_patch_minimum_families
        )
        if barline_patch_applicable:
            barline_patch = propose_barline_patch(
                [
                    BarlinePatchCandidate(
                        variant=member[0].candidate.variant,
                        family=variant_family(member[0].candidate.variant),
                        measure=member[1],
                        semantics=member[2],
                        page_score=float(member[0].candidate.score),
                        page_probability=float(getattr(member[0].candidate, "calibrated_probability", 0.5)),
                        measure_probability=measure_probabilities[index],
                        visual_probability=visual_probabilities[index],
                        event_probability=event_probabilities[index],
                        context_probability=context_probabilities[index],
                        ensemble_probability=ensemble_probabilities[index],
                        valid=bool(member[0].candidate.valid),
                    )
                    for index, member in enumerate(members)
                ],
                template_index=template_index,
                missing_candidate_count=missing_count,
                calibrator=barline_patch_calibrator,
                base_measure=patch_base,
            )
            if barline_patch.input is not None:
                barline_patch_probabilities.append(barline_patch.probability)

        barline_patch_applied = commit_patch_stage(
            "barline", barline_patch.patched_measure, barline_patch.accepted
        )
        event_presence_patch = EventPresencePatchResult(
            None, "none", (), 0.5, event_presence_patch_calibrator.threshold, False, "not_applicable"
        )
        event_presence_patch_applicable = (
            not replace
            and len(eligible_families) >= DEFAULT_POLICY.event_presence_patch_minimum_families
        )
        if event_presence_patch_applicable:
            event_presence_patch = propose_event_presence_patch(
                [
                    EventPresencePatchCandidate(
                        variant=member[0].candidate.variant,
                        family=variant_family(member[0].candidate.variant),
                        measure=member[1],
                        semantics=member[2],
                        page_score=float(member[0].candidate.score),
                        page_probability=float(getattr(member[0].candidate, "calibrated_probability", 0.5)),
                        measure_probability=measure_probabilities[index],
                        visual_probability=visual_probabilities[index],
                        event_probability=event_probabilities[index],
                        context_probability=context_probabilities[index],
                        ensemble_probability=ensemble_probabilities[index],
                        valid=bool(member[0].candidate.valid),
                    )
                    for index, member in enumerate(members)
                ],
                template_index=template_index,
                missing_candidate_count=missing_count,
                is_first_measure=measure_index == 0,
                is_last_measure=measure_index + 1 == len(result_measures),
                calibrator=event_presence_patch_calibrator,
                base_measure=patch_base,
            )
            if event_presence_patch.input is not None:
                event_presence_patch_probabilities.append(event_presence_patch.probability)

        event_presence_visual_audit = EventPresenceVisualAudit(
            applicable=False,
            operation=event_presence_patch.operation,
            changed_event_count=0,
            probability=0.5,
            threshold=round(event_presence_visual_guard.threshold, 6),
            accepted=True,
            reason="not_applicable",
            model_version=event_presence_visual_guard.model_version,
        )
        if event_presence_patch.accepted and event_presence_patch.patched_measure is not None:
            try:
                source_measure = patch_base if patch_base is not None else result_measures[measure_index]
                source_semantics, _ = measure_from_xml(
                    source_measure, dict(patch_stage_inherited)
                )
                patched_semantics, _ = measure_from_xml(
                    event_presence_patch.patched_measure, dict(patch_stage_inherited)
                )
                changed_index = (
                    event_presence_patch.changed_event_indices[0]
                    if len(event_presence_patch.changed_event_indices) == 1
                    else -1
                )
                event_presence_visual_audit = event_presence_visual_guard.audit_transaction(
                    source_evidence,
                    source_semantics,
                    patched_semantics,
                    event_presence_patch.operation,
                    changed_index,
                )
            except (TypeError, ValueError, ArithmeticError, etree.XMLSyntaxError):
                event_presence_visual_audit = EventPresenceVisualAudit(
                    applicable=True,
                    operation=event_presence_patch.operation,
                    changed_event_count=len(event_presence_patch.changed_event_indices),
                    probability=0.5,
                    threshold=round(event_presence_visual_guard.threshold, 6),
                    accepted=False,
                    reason="event_presence_visual_audit_failed",
                    model_version=event_presence_visual_guard.model_version,
                )
            if event_presence_visual_audit.applicable:
                event_presence_visual_guard_transaction_count += 1
                event_presence_visual_guard_probabilities.append(
                    event_presence_visual_audit.probability
                )
                if not event_presence_visual_audit.accepted:
                    event_presence_visual_guard_rejected_count += 1
                    event_presence_patch = dataclass_replace(
                        event_presence_patch,
                        accepted=False,
                        reason=(
                            f"{event_presence_patch.reason}:"
                            f"{event_presence_visual_audit.reason}"
                        ),
                    )

        event_presence_patch_applied = commit_patch_stage(
            "event_presence",
            event_presence_patch.patched_measure,
            event_presence_patch.accepted,
        )
        patch_stage_rejected_count += len(patch_stage_rejections)
        patch_applied = (
            chord_patch_applied
            or tuplet_patch_applied
            or pitch_patch_applied
            or rhythm_patch_applied
            or event_kind_patch_applied
            or attribute_patch_applied
            or tie_patch_applied
            or slur_patch_applied
            or articulation_patch_applied
            or ornament_patch_applied
            or grace_patch_applied
            or lyric_patch_applied
            or direction_patch_applied
            or barline_patch_applied
            or event_presence_patch_applied
        )
        patch_transaction_applicable = False
        patch_transaction_patch_count = 0
        patch_transaction_semantic_patch_count = 0
        patch_transaction_probability = 1.0
        patch_transaction_threshold = patch_transaction_calibrator.threshold
        patch_transaction_model = patch_transaction_calibrator.model_version
        patch_transaction_accepted = True
        patch_transaction_reason = "not_applicable"
        if patch_applied:
            assert patch_base is not None
            patch_evidence = tuple(
                evidence
                for applied, evidence in (
                    (chord_patch_applied, PatchEvidence(
                        "chord", chord_patch.probability, chord_patch.threshold,
                        changed_events=len(chord_patch.changed_event_indices),
                        changed_surfaces=len(chord_patch.changed_event_indices),
                    )),
                    (tuplet_patch_applied, PatchEvidence(
                        "tuplet", tuplet_patch.probability, tuplet_patch.threshold,
                        changed_events=len(tuplet_patch.changed_event_indices),
                        changed_surfaces=tuplet_patch.changed_group_count,
                    )),
                    (pitch_patch_applied, PatchEvidence(
                        "pitch", pitch_patch.probability, pitch_patch.threshold,
                        changed_events=len(pitch_patch.changed_event_indices),
                        changed_surfaces=len(pitch_patch.changed_event_indices),
                    )),
                    (rhythm_patch_applied, PatchEvidence(
                        "rhythm", rhythm_patch.probability, rhythm_patch.threshold,
                        changed_events=len(rhythm_patch.changed_event_indices),
                        changed_surfaces=len(rhythm_patch.changed_event_indices),
                    )),
                    (event_kind_patch_applied, PatchEvidence(
                        "event_kind", event_kind_patch.probability, event_kind_patch.threshold,
                        changed_events=len(event_kind_patch.changed_event_indices),
                        changed_surfaces=len(event_kind_patch.changed_event_indices),
                    )),
                    (attribute_patch_applied, PatchEvidence(
                        "attribute", attribute_patch.probability, attribute_patch.threshold,
                        changed_surfaces=len(attribute_patch.changed_attributes),
                    )),
                    (tie_patch_applied, PatchEvidence(
                        "tie", tie_patch.probability, tie_patch.threshold,
                        changed_events=len(tie_patch.changed_event_indices),
                        changed_surfaces=len(tie_patch.changed_event_indices),
                    )),
                    (slur_patch_applied, PatchEvidence(
                        "slur", slur_patch.probability, slur_patch.threshold,
                        changed_events=len(slur_patch.changed_event_indices),
                        changed_surfaces=slur_patch.changed_arc_count,
                    )),
                    (articulation_patch_applied, PatchEvidence(
                        "articulation", articulation_patch.probability, articulation_patch.threshold,
                        changed_events=len(articulation_patch.changed_event_indices),
                        changed_surfaces=articulation_patch.changed_mark_count,
                    )),
                    (ornament_patch_applied, PatchEvidence(
                        "ornament", ornament_patch.probability, ornament_patch.threshold,
                        changed_events=len(ornament_patch.changed_event_indices),
                        changed_surfaces=ornament_patch.changed_mark_count,
                    )),
                    (grace_patch_applied, PatchEvidence(
                        "grace", grace_patch.probability, grace_patch.threshold,
                        changed_events=len(grace_patch.changed_event_indices),
                        changed_surfaces=grace_patch.added_grace_count + grace_patch.removed_grace_count,
                    )),
                    (lyric_patch_applied, PatchEvidence(
                        "lyric", lyric_patch.probability, lyric_patch.threshold,
                        changed_events=len(lyric_patch.changed_event_indices),
                        changed_surfaces=lyric_patch.changed_lyric_count,
                    )),
                    (direction_patch_applied, PatchEvidence(
                        "direction", direction_patch.probability, direction_patch.threshold,
                        changed_surfaces=direction_patch.changed_direction_count,
                    )),
                    (barline_patch_applied, PatchEvidence(
                        "barline", barline_patch.probability, barline_patch.threshold,
                        changed_surfaces=len(barline_patch.changed_locations) + barline_patch.changed_repeat_count,
                    )),
                    (event_presence_patch_applied, PatchEvidence(
                        "event_presence", event_presence_patch.probability, event_presence_patch.threshold,
                        changed_events=len(event_presence_patch.changed_event_indices),
                        changed_surfaces=len(event_presence_patch.changed_event_indices),
                    )),
                )
                if applied
            )
            transaction_input = PatchTransactionInput.from_evidence(
                patch_evidence,
                eligible_family_count=len(eligible_families),
                exact_family_support_ratio=len(exact_families) / max(len(eligible_families), 1),
                semantic_family_support_ratio=len(cluster_families) / max(len(eligible_families), 1),
                missing_ratio=missing_ratio,
                selected_measure_probability=selected_measure_probability,
                selected_visual_probability=selected_visual_probability,
                selected_event_probability=selected_event_probability,
                selected_context_probability=selected_context_probability,
                selected_ensemble_probability=selected_ensemble_probability,
                semantic_confidence=confidence,
                mean_cluster_distance=mean_distance,
                template_distance=template_distance,
            )
            patch_transaction_patch_count = transaction_input.patch_count
            patch_transaction_semantic_patch_count = transaction_input.semantic_patch_count
            patch_transaction_accepted, patch_transaction_reason = _patch_transaction_guard(
                result_measures[measure_index],
                patch_base,
                patch_stage_inherited,
            )
            if patch_transaction_accepted:
                transaction_calibration = patch_transaction_calibrator.calibrate(transaction_input)
                patch_transaction_applicable = transaction_calibration.applicable
                patch_transaction_probability = transaction_calibration.probability
                patch_transaction_threshold = transaction_calibration.threshold
                patch_transaction_model = transaction_calibration.model_version
                patch_transaction_accepted = transaction_calibration.accepted
                patch_transaction_reason = transaction_calibration.reason
                if transaction_calibration.applicable:
                    patch_transaction_evaluated_count += 1
                    patch_transaction_probabilities.append(transaction_calibration.probability)
            else:
                patch_transaction_probability = 0.0
            if not patch_transaction_accepted:
                patch_transaction_rejected_count += 1
                chord_patch_applied = False
                tuplet_patch_applied = False
                pitch_patch_applied = False
                rhythm_patch_applied = False
                event_kind_patch_applied = False
                attribute_patch_applied = False
                tie_patch_applied = False
                slur_patch_applied = False
                articulation_patch_applied = False
                ornament_patch_applied = False
                grace_patch_applied = False
                lyric_patch_applied = False
                direction_patch_applied = False
                barline_patch_applied = False
                event_presence_patch_applied = False
                patch_applied = False
                patch_base = None
        if replace:
            replacement = copy.deepcopy(selected_measure)
            replacement.set("number", result_measures[measure_index].get("number", str(measure_index + 1)))
            result_part.replace(result_measures[measure_index], replacement)
            result_measures[measure_index] = replacement
            replacements += 1
        elif patch_applied:
            assert patch_base is not None
            replacement = copy.deepcopy(patch_base)
            replacement.set("number", result_measures[measure_index].get("number", str(measure_index + 1)))
            result_part.replace(result_measures[measure_index], replacement)
            result_measures[measure_index] = replacement
            replacements += 1
            if chord_patch_applied:
                chord_patch_measure_count += 1
                chord_patch_event_count += len(chord_patch.changed_event_indices)
            if tuplet_patch_applied:
                tuplet_patch_measure_count += 1
                tuplet_patch_event_count += len(tuplet_patch.changed_event_indices)
                tuplet_patch_group_count += tuplet_patch.changed_group_count
            if pitch_patch_applied:
                pitch_patch_measure_count += 1
                pitch_patch_event_count += len(pitch_patch.changed_event_indices)
            if rhythm_patch_applied:
                rhythm_patch_measure_count += 1
                rhythm_patch_event_count += len(rhythm_patch.changed_event_indices)
            if event_kind_patch_applied:
                event_kind_patch_measure_count += 1
                event_kind_patch_event_count += len(event_kind_patch.changed_event_indices)
            if attribute_patch_applied:
                attribute_patch_measure_count += 1
                attribute_patch_attribute_count += len(attribute_patch.changed_attributes)
            if tie_patch_applied:
                tie_patch_measure_count += 1
                tie_patch_event_count += len(tie_patch.changed_event_indices)
            if slur_patch_applied:
                slur_patch_measure_count += 1
                slur_patch_event_count += len(slur_patch.changed_event_indices)
                slur_patch_arc_count += slur_patch.changed_arc_count
            if articulation_patch_applied:
                articulation_patch_measure_count += 1
                articulation_patch_event_count += len(articulation_patch.changed_event_indices)
                articulation_patch_mark_count += articulation_patch.changed_mark_count
            if ornament_patch_applied:
                ornament_patch_measure_count += 1
                ornament_patch_event_count += len(ornament_patch.changed_event_indices)
                ornament_patch_mark_count += ornament_patch.changed_mark_count
            if grace_patch_applied:
                grace_patch_measure_count += 1
                grace_patch_event_count += len(grace_patch.changed_event_indices)
                grace_patch_added_count += grace_patch.added_grace_count
                grace_patch_removed_count += grace_patch.removed_grace_count
            if lyric_patch_applied:
                lyric_patch_measure_count += 1
                lyric_patch_event_count += len(lyric_patch.changed_event_indices)
                lyric_patch_lyric_count += lyric_patch.changed_lyric_count
            if direction_patch_applied:
                direction_patch_measure_count += 1
                direction_patch_direction_count += direction_patch.changed_direction_count
            if barline_patch_applied:
                barline_patch_measure_count += 1
                barline_patch_location_count += len(barline_patch.changed_locations)
                barline_patch_repeat_count += barline_patch.changed_repeat_count
            if event_presence_patch_applied:
                event_presence_patch_measure_count += 1
                if event_presence_patch.operation == "insert":
                    event_presence_patch_inserted_event_count += len(event_presence_patch.changed_event_indices)
                elif event_presence_patch.operation == "delete":
                    event_presence_patch_deleted_event_count += len(event_presence_patch.changed_event_indices)

        if replace:
            decision = "replace_majority" if selection_kind == "exact_majority" else "replace_semantic_consensus"
        elif not patch_transaction_accepted:
            decision = "retain_template_patch_transaction_guard"
            unresolved.append(measure_index + 1)
        elif patch_applied:
            applied_patch_names = [
                name
                for name, applied in (
                    ("chord", chord_patch_applied),
                    ("tuplet", tuplet_patch_applied),
                    ("pitch", pitch_patch_applied),
                    ("rhythm", rhythm_patch_applied),
                    ("event_kind", event_kind_patch_applied),
                    ("attribute", attribute_patch_applied),
                    ("tie", tie_patch_applied),
                    ("slur", slur_patch_applied),
                    ("articulation", articulation_patch_applied),
                    ("ornament", ornament_patch_applied),
                    ("grace", grace_patch_applied),
                    ("lyric", lyric_patch_applied),
                    ("direction", direction_patch_applied),
                    ("barline", barline_patch_applied),
                    (f"event_presence_{event_presence_patch.operation}", event_presence_patch_applied),
                )
                if applied
            ]
            decision = f"patch_{'_and_'.join(applied_patch_names)}_consensus"
        elif missing_count and selection_kind == "template":
            decision = "retain_template_alignment_gap"
            unresolved.append(measure_index + 1)
        elif selection_kind == "template" and len(groups) > 1 and max_template_distance > DEFAULT_POLICY.unresolved_measure_distance:
            decision = "retain_template_no_majority"
            unresolved.append(measure_index + 1)
        elif significant_difference and preservation_gate_required and not preservation_gate_accepted:
            decision = "retain_template_preservation_support_guard"
            unresolved.append(measure_index + 1)
        elif significant_difference and selection_kind in {"exact_majority", "semantic_consensus"} and (selection_risk is None or not selection_risk.accepted):
            decision = "retain_template_selection_risk_guard"
            unresolved.append(measure_index + 1)
        elif significant_difference and not selected_is_safe:
            decision = "retain_template_quality_guard"
            unresolved.append(measure_index + 1)
        elif len(groups) > 1 and redundant_state_attribute_only:
            decision = "retain_redundant_state_attributes"
        elif (
            len(groups) > 1
            and selection_kind == "exact_majority"
            and strict_majority
            and members[template_index][3] == _winning_signature
        ):
            decision = "retain_exact_majority"
        elif len(groups) > 1:
            decision = "retain_semantic_equivalent"
        else:
            decision = "retain_agreement"

        if len(groups) > 1 or missing_count:
            if decision in {
                "retain_exact_majority",
                "retain_redundant_state_attributes",
            }:
                resolved_disagreement.append(measure_index + 1)
            else:
                disagreement.append(measure_index + 1)

        signatures_for_report = {
            signature: [members[index][0].candidate.variant for index in indices]
            for signature, indices in groups.items()
        }
        votes.append(
            MeasureVote(
                measure_index=measure_index + 1,
                selected_variant=selected_loaded.candidate.variant if selection_kind != "template" else template_candidate.variant,
                selected_support=exact_support if selection_kind == "exact_majority" else len(cluster),
                eligible_candidates=len(applicable),
                unanimous=unanimous,
                strict_majority=strict_majority,
                exact_family_support=len(exact_families),
                semantic_family_support=len(cluster_families),
                selected_preservation_family_support=selected_preservation_family_support,
                preservation_gate_required=preservation_gate_required,
                preservation_gate_accepted=preservation_gate_accepted,
                eligible_family_count=len(eligible_families),
                abstaining_family_count=len(measure_abstaining_families),
                abstaining_families=measure_abstaining_families,
                replaced_template=replace or patch_applied,
                decision=decision,
                signatures=signatures_for_report,
                semantic_support_ratio=round(support_ratio, 6),
                semantic_confidence=round(confidence, 6),
                mean_cluster_distance=round(mean_distance, 6),
                template_distance=round(template_distance, 6),
                cluster_variants=cluster_variants,
                aligned_candidates=len(members),
                missing_candidates=missing_count,
                selected_measure_probability=round(selected_measure_probability, 6),
                measure_calibration_model=measure_calibrator.model_version,
                candidate_measure_probabilities={
                    members[index][0].candidate.variant: round(probability, 6)
                    for index, probability in enumerate(measure_probabilities)
                },
                selected_visual_probability=round(selected_visual_probability, 6),
                visual_calibration_model=visual_calibrator.model_version,
                candidate_visual_probabilities={
                    members[index][0].candidate.variant: round(probability, 6)
                    for index, probability in enumerate(visual_probabilities)
                },
                selected_event_probability=round(selected_event_probability, 6),
                event_calibration_model=event_calibrator.model_version,
                candidate_event_probabilities={
                    members[index][0].candidate.variant: round(probability, 6)
                    for index, probability in enumerate(event_probabilities)
                },
                selected_context_probability=round(selected_context_probability, 6),
                context_calibration_model=context_calibrator.model_version,
                candidate_context_probabilities={
                    members[index][0].candidate.variant: round(probability, 6)
                    for index, probability in enumerate(context_probabilities)
                },
                selected_ensemble_probability=round(selected_ensemble_probability, 6),
                ensemble_calibration_model=ensemble_calibrator.model_version,
                candidate_ensemble_probabilities={
                    members[index][0].candidate.variant: round(probability, 6)
                    for index, probability in enumerate(ensemble_probabilities)
                },
                selection_risk_applicable=selection_risk_applicable,
                selected_selection_risk_probability=selection_risk.probability if selection_risk is not None else 0.5,
                selection_risk_threshold=selection_risk.threshold if selection_risk is not None else selection_risk_calibrator.threshold,
                selection_risk_accepted=selection_risk.accepted if selection_risk is not None else True,
                selection_risk_model=selection_risk_calibrator.model_version,
                chord_patch_applicable=chord_patch_applicable,
                chord_patch_event_count=len(chord_patch.changed_event_indices),
                chord_patch_probability=chord_patch.probability,
                chord_patch_threshold=chord_patch.threshold,
                chord_patch_accepted=chord_patch_applied,
                chord_patch_model=chord_patch_calibrator.model_version,
                chord_patch_reason=chord_patch.reason,
                tuplet_patch_applicable=tuplet_patch_applicable,
                tuplet_patch_event_count=len(tuplet_patch.changed_event_indices),
                tuplet_patch_group_count=tuplet_patch.changed_group_count,
                tuplet_patch_probability=tuplet_patch.probability,
                tuplet_patch_threshold=tuplet_patch.threshold,
                tuplet_patch_accepted=tuplet_patch_applied,
                tuplet_patch_model=tuplet_patch_calibrator.model_version,
                tuplet_patch_reason=tuplet_patch.reason,
                pitch_patch_applicable=pitch_patch_applicable,
                pitch_patch_event_count=len(pitch_patch.changed_event_indices),
                pitch_patch_probability=pitch_patch.probability,
                pitch_patch_threshold=pitch_patch.threshold,
                pitch_patch_accepted=pitch_patch_applied,
                pitch_patch_model=pitch_patch.model_version,
                pitch_patch_reason=pitch_patch.reason,
                rhythm_patch_applicable=rhythm_patch_applicable,
                rhythm_patch_event_count=len(rhythm_patch.changed_event_indices),
                rhythm_patch_probability=rhythm_patch.probability,
                rhythm_patch_threshold=rhythm_patch.threshold,
                rhythm_patch_accepted=rhythm_patch_applied,
                rhythm_patch_model=rhythm_patch_calibrator.model_version,
                rhythm_patch_reason=rhythm_patch.reason,
                event_kind_patch_applicable=event_kind_patch_applicable,
                event_kind_patch_event_count=len(event_kind_patch.changed_event_indices),
                event_kind_patch_probability=event_kind_patch.probability,
                event_kind_patch_threshold=event_kind_patch.threshold,
                event_kind_patch_accepted=event_kind_patch_applied,
                event_kind_patch_model=event_kind_patch_calibrator.model_version,
                event_kind_patch_reason=event_kind_patch.reason,
                event_kind_visual_guard_applicable=event_kind_visual_audit.applicable,
                event_kind_visual_guard_changed_events=event_kind_visual_audit.changed_event_count,
                event_kind_visual_guard_probability=event_kind_visual_audit.probability,
                event_kind_visual_guard_threshold=event_kind_visual_audit.threshold,
                event_kind_visual_guard_accepted=event_kind_visual_audit.accepted,
                event_kind_visual_guard_model=event_kind_visual_audit.model_version,
                event_kind_visual_guard_reason=event_kind_visual_audit.reason,
                attribute_patch_applicable=attribute_patch_applicable,
                attribute_patch_attributes=tuple(attribute_patch.changed_attributes),
                attribute_patch_probability=attribute_patch.probability,
                attribute_patch_threshold=attribute_patch.threshold,
                attribute_patch_accepted=attribute_patch_applied,
                attribute_patch_model=attribute_patch_calibrator.model_version,
                attribute_patch_reason=attribute_patch.reason,
                tie_patch_applicable=tie_patch_applicable,
                tie_patch_event_count=len(tie_patch.changed_event_indices),
                tie_patch_probability=tie_patch.probability,
                tie_patch_threshold=tie_patch.threshold,
                tie_patch_accepted=tie_patch_applied,
                tie_patch_model=tie_patch_calibrator.model_version,
                tie_patch_reason=tie_patch.reason,
                tie_visual_guard_applicable=tie_visual_audit.applicable,
                tie_visual_guard_changed_ties=tie_visual_audit.changed_tie_count,
                tie_visual_guard_probability=tie_visual_audit.probability,
                tie_visual_guard_threshold=tie_visual_audit.threshold,
                tie_visual_guard_accepted=tie_visual_audit.accepted,
                tie_visual_guard_model=tie_visual_audit.model_version,
                tie_visual_guard_reason=tie_visual_audit.reason,
                slur_patch_applicable=slur_patch_applicable,
                slur_patch_event_count=len(slur_patch.changed_event_indices),
                slur_patch_arc_count=slur_patch.changed_arc_count,
                slur_patch_probability=slur_patch.probability,
                slur_patch_threshold=slur_patch.threshold,
                slur_patch_accepted=slur_patch_applied,
                slur_patch_model=slur_patch_calibrator.model_version,
                slur_patch_reason=slur_patch.reason,
                articulation_patch_applicable=articulation_patch_applicable,
                articulation_patch_event_count=len(articulation_patch.changed_event_indices),
                articulation_patch_mark_count=articulation_patch.changed_mark_count,
                articulation_patch_probability=articulation_patch.probability,
                articulation_patch_threshold=articulation_patch.threshold,
                articulation_patch_accepted=articulation_patch_applied,
                articulation_patch_model=articulation_patch_calibrator.model_version,
                articulation_patch_reason=articulation_patch.reason,
                accent_visual_guard_applicable=accent_visual_audit.applicable,
                accent_visual_guard_changed_accents=accent_visual_audit.changed_accent_count,
                accent_visual_guard_probability=accent_visual_audit.probability,
                accent_visual_guard_threshold=accent_visual_audit.threshold,
                accent_visual_guard_accepted=accent_visual_audit.accepted,
                accent_visual_guard_model=accent_visual_audit.model_version,
                accent_visual_guard_reason=accent_visual_audit.reason,
                ornament_patch_applicable=ornament_patch_applicable,
                ornament_patch_event_count=len(ornament_patch.changed_event_indices),
                ornament_patch_mark_count=ornament_patch.changed_mark_count,
                ornament_patch_probability=ornament_patch.probability,
                ornament_patch_threshold=ornament_patch.threshold,
                ornament_patch_accepted=ornament_patch_applied,
                ornament_patch_model=ornament_patch_calibrator.model_version,
                ornament_patch_reason=ornament_patch.reason,
                grace_patch_applicable=grace_patch_applicable,
                grace_patch_event_count=len(grace_patch.changed_event_indices),
                grace_patch_added_count=grace_patch.added_grace_count,
                grace_patch_removed_count=grace_patch.removed_grace_count,
                grace_patch_probability=grace_patch.probability,
                grace_patch_threshold=grace_patch.threshold,
                grace_patch_accepted=grace_patch_applied,
                grace_patch_model=grace_patch_calibrator.model_version,
                grace_patch_reason=grace_patch.reason,
                lyric_patch_applicable=lyric_patch_applicable,
                lyric_patch_event_count=len(lyric_patch.changed_event_indices),
                lyric_patch_lyric_count=lyric_patch.changed_lyric_count,
                lyric_patch_probability=lyric_patch.probability,
                lyric_patch_threshold=lyric_patch.threshold,
                lyric_patch_accepted=lyric_patch_applied,
                lyric_patch_model=(
                    lyric_patch_calibrator.model_version
                    if lyric_patch_calibrator is not None
                    else "disabled_out_of_scope"
                ),
                lyric_patch_reason=lyric_patch.reason,
                direction_patch_applicable=direction_patch_applicable,
                direction_patch_direction_count=direction_patch.changed_direction_count,
                direction_patch_kinds=tuple(direction_patch.changed_kinds),
                direction_patch_probability=direction_patch.probability,
                direction_patch_threshold=direction_patch.threshold,
                direction_patch_accepted=direction_patch_applied,
                direction_patch_model=direction_patch_calibrator.model_version,
                direction_patch_reason=direction_patch.reason,
                patch_transaction_applicable=patch_transaction_applicable,
                patch_transaction_patch_count=patch_transaction_patch_count,
                patch_transaction_semantic_patch_count=patch_transaction_semantic_patch_count,
                patch_transaction_probability=patch_transaction_probability,
                patch_transaction_threshold=patch_transaction_threshold,
                patch_transaction_model=patch_transaction_model,
                patch_stage_rejections=tuple(patch_stage_rejections),
                barline_patch_applicable=barline_patch_applicable,
                barline_patch_locations=tuple(barline_patch.changed_locations),
                barline_patch_repeat_count=barline_patch.changed_repeat_count,
                barline_patch_probability=barline_patch.probability,
                barline_patch_threshold=barline_patch.threshold,
                barline_patch_accepted=barline_patch_applied,
                barline_patch_model=barline_patch_calibrator.model_version,
                barline_patch_reason=barline_patch.reason,
                patch_transaction_accepted=patch_transaction_accepted,
                patch_transaction_reason=patch_transaction_reason,
                event_presence_patch_applicable=event_presence_patch_applicable,
                event_presence_patch_operation=event_presence_patch.operation,
                event_presence_patch_event_count=len(event_presence_patch.changed_event_indices),
                event_presence_patch_probability=event_presence_patch.probability,
                event_presence_patch_threshold=event_presence_patch.threshold,
                event_presence_patch_accepted=event_presence_patch_applied,
                event_presence_patch_model=event_presence_patch_calibrator.model_version,
                event_presence_patch_reason=event_presence_patch.reason,
                event_presence_visual_guard_applicable=event_presence_visual_audit.applicable,
                event_presence_visual_guard_operation=event_presence_visual_audit.operation,
                event_presence_visual_guard_changed_events=event_presence_visual_audit.changed_event_count,
                event_presence_visual_guard_probability=event_presence_visual_audit.probability,
                event_presence_visual_guard_threshold=event_presence_visual_audit.threshold,
                event_presence_visual_guard_accepted=event_presence_visual_audit.accepted,
                event_presence_visual_guard_model=event_presence_visual_audit.model_version,
                event_presence_visual_guard_reason=event_presence_visual_audit.reason,
            )
        )

    # Cross-measure ties are evaluated only after every local measure transaction has
    # settled.  This avoids using stale boundary events and permits a repaired pitch or
    # chord topology to become the conservative base for the boundary decision.
    for boundary_index in range(max(0, len(result_measures) - 1)):
        current_score = score_from_tree(result_tree)
        if boundary_index + 1 >= len(current_score.measures):
            break
        base_left_semantics = current_score.measures[boundary_index]
        base_right_semantics = current_score.measures[boundary_index + 1]
        rows: list[CrossTiePatchCandidate] = []
        scope_abstaining_variants: list[str] = []
        for item in eligible:
            mapping = alignments[item.candidate.variant].reference_to_candidate
            left_index = mapping[boundary_index] if boundary_index < len(mapping) else None
            right_index = mapping[boundary_index + 1] if boundary_index + 1 < len(mapping) else None
            consecutive = (
                left_index is not None
                and right_index is not None
                and right_index == left_index + 1
            )
            left_quality = measure_quality_by_index[boundary_index].get(item.candidate.variant)
            right_quality = measure_quality_by_index[boundary_index + 1].get(item.candidate.variant)
            quality_valid = left_quality is not None and right_quality is not None
            directly_observed = candidate_applies_to_boundary(
                item.candidate.variant,
                boundary_index + 1,
                boundary_index + 2,
            )
            if item.candidate.valid and not directly_observed:
                scope_abstaining_variants.append(item.candidate.variant)
            valid = bool(
                item.candidate.valid
                and directly_observed
                and consecutive
                and quality_valid
            )
            if valid:
                assert left_index is not None and right_index is not None
                left_measure = item.measures[left_index]
                right_measure = item.measures[right_index]
                left_semantics = item.semantics[left_index]
                right_semantics = item.semantics[right_index]
                assert left_quality is not None and right_quality is not None
                quality = tuple((left + right) / 2.0 for left, right in zip(left_quality, right_quality, strict=True))
            else:
                left_measure = result_measures[boundary_index]
                right_measure = result_measures[boundary_index + 1]
                left_semantics = base_left_semantics
                right_semantics = base_right_semantics
                quality = (0.5, 0.5, 0.5, 0.5, 0.5)
            rows.append(
                CrossTiePatchCandidate(
                    variant=item.candidate.variant,
                    family=variant_family(item.candidate.variant),
                    left_measure=left_measure,
                    right_measure=right_measure,
                    left_semantics=left_semantics,
                    right_semantics=right_semantics,
                    page_score=float(item.candidate.score),
                    page_probability=float(getattr(item.candidate, "calibrated_probability", 0.5)),
                    measure_probability=quality[0],
                    visual_probability=quality[1],
                    event_probability=quality[2],
                    context_probability=quality[3],
                    ensemble_probability=quality[4],
                    alignment_similarity=alignments[item.candidate.variant].similarity,
                    valid=valid,
                )
            )
        cross_tie = propose_cross_tie_patch(
            rows,
            template_variant=template_candidate.variant,
            base_left=result_measures[boundary_index],
            base_right=result_measures[boundary_index + 1],
            base_left_semantics=base_left_semantics,
            base_right_semantics=base_right_semantics,
            calibrator=cross_tie_patch_calibrator,
        )
        if cross_tie.input is not None:
            cross_tie_patch_probabilities.append(cross_tie.probability)
        applied = bool(
            cross_tie.accepted
            and cross_tie.left_measure is not None
            and cross_tie.right_measure is not None
        )
        transaction_reason = "not_applicable"
        if applied:
            temporary = copy.deepcopy(result_tree)
            temporary_part = temporary.getroot().find("part")
            assert temporary_part is not None
            temporary_measures = temporary_part.findall("measure")
            left = copy.deepcopy(cross_tie.left_measure)
            right = copy.deepcopy(cross_tie.right_measure)
            left.set("number", temporary_measures[boundary_index].get("number", str(boundary_index + 1)))
            right.set("number", temporary_measures[boundary_index + 1].get("number", str(boundary_index + 2)))
            temporary_part.replace(temporary_measures[boundary_index], left)
            temporary_measures = temporary_part.findall("measure")
            temporary_part.replace(temporary_measures[boundary_index + 1], right)
            before_issues = Counter(
                (issue.code, issue.measure_index, issue.severity)
                for issue in audit_score(score_from_tree(result_tree))
            )
            after_issues = Counter(
                (issue.code, issue.measure_index, issue.severity)
                for issue in audit_score(score_from_tree(temporary))
            )
            introduced = sorted(
                key for key, count in after_issues.items()
                if count > before_issues.get(key, 0)
            )
            if introduced:
                applied = False
                transaction_reason = "score_transaction_guard:" + ",".join(
                    f"{code}@{measure}" for code, measure, _severity in introduced
                )
                cross_tie_patch_transaction_rejected_count += 1
            else:
                result_tree = temporary
                result_part = result_tree.getroot().find("part")
                assert result_part is not None
                result_measures = result_part.findall("measure")
                cross_tie_patch_boundary_count += 1
                cross_tie_patch_endpoint_count += cross_tie.changed_endpoint_count
        cross_tie_boundaries.append({
            "left_measure_index": boundary_index + 1,
            "right_measure_index": boundary_index + 2,
            "applicable": cross_tie.input is not None,
            "probability": cross_tie.probability,
            "threshold": cross_tie.threshold,
            "accepted": applied,
            "changed_endpoint_count": cross_tie.changed_endpoint_count,
            "scope_abstaining_variants": sorted(set(scope_abstaining_variants)),
            "reason": transaction_reason if transaction_reason != "not_applicable" else cross_tie.reason,
            "model": cross_tie_patch_calibrator.model_version,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(
        output_path,
        etree.tostring(
            result_tree,
            encoding="UTF-8",
            xml_declaration=True,
            pretty_print=True,
            doctype=MUSICXML_DOCTYPE,
        ),
    )
    measure_count = len(result_measures)
    exact_ratio = unanimous_count / max(measure_count, 1)
    semantic_ratio = sum(semantic_similarities) / max(len(semantic_similarities), 1)
    mean_confidence = sum(confidences) / max(len(confidences), 1)
    agreement_ratio = 0.45 * exact_ratio + 0.55 * semantic_ratio
    return ConsensusReport(
        output_path=str(output_path),
        template_variant=template_candidate.variant,
        candidate_count=len(candidate_list),
        eligible_candidate_count=len(eligible),
        measure_count=measure_count,
        agreement_ratio=round(agreement_ratio, 6),
        exact_agreement_ratio=round(exact_ratio, 6),
        semantic_agreement_ratio=round(semantic_ratio, 6),
        mean_measure_confidence=round(mean_confidence, 6),
        unanimous_measure_count=unanimous_count,
        majority_measure_count=majority_count,
        preservation_disagreement_measure_indices=tuple(
            sorted(set(preservation_disagreement))
        ),
        resolved_disagreement_measure_indices=tuple(
            sorted(set(resolved_disagreement))
        ),
        disagreement_measure_indices=tuple(sorted(set(disagreement))),
        unresolved_measure_indices=tuple(sorted(set(unresolved))),
        replacements=replacements,
        votes=tuple(votes),
        candidate_alignment=alignment_report,
        mean_selected_measure_probability=round(sum(selected_measure_probabilities) / max(len(selected_measure_probabilities), 1), 6),
        measure_calibration_model=measure_calibrator.model_version,
        policy_version=DEFAULT_POLICY.version,
        requested_measure_count=template_count_decision.requested_count,
        template_measure_count=len(template_entry.measures),
        template_count_family_support=template_count_decision.family_support,
        template_count_eligible_family_count=template_count_decision.eligible_family_count,
        template_count_reselected=template_count_decision.reselected,
        mean_visual_probability=round(sum(selected_visual_probabilities) / max(len(selected_visual_probabilities), 1), 6),
        visual_calibration_model=visual_calibrator.model_version,
        mean_event_probability=round(sum(selected_event_probabilities) / max(len(selected_event_probabilities), 1), 6),
        event_calibration_model=event_calibrator.model_version,
        mean_context_probability=round(sum(selected_context_probabilities) / max(len(selected_context_probabilities), 1), 6),
        context_calibration_model=context_calibrator.model_version,
        mean_ensemble_probability=round(sum(selected_ensemble_probabilities) / max(len(selected_ensemble_probabilities), 1), 6),
        ensemble_calibration_model=ensemble_calibrator.model_version,
        mean_selection_risk_probability=round(sum(selected_selection_risk_probabilities) / len(selected_selection_risk_probabilities), 6) if selected_selection_risk_probabilities else 0.5,
        selection_risk_model=selection_risk_calibrator.model_version,
        selection_risk_threshold=selection_risk_calibrator.threshold,
        chord_patch_measure_count=chord_patch_measure_count,
        chord_patch_event_count=chord_patch_event_count,
        mean_chord_patch_probability=(
            round(sum(chord_patch_probabilities) / len(chord_patch_probabilities), 6)
            if chord_patch_probabilities else 0.5
        ),
        chord_patch_model=chord_patch_calibrator.model_version,
        chord_patch_threshold=chord_patch_calibrator.threshold,
        tuplet_patch_measure_count=tuplet_patch_measure_count,
        tuplet_patch_event_count=tuplet_patch_event_count,
        tuplet_patch_group_count=tuplet_patch_group_count,
        mean_tuplet_patch_probability=(
            round(sum(tuplet_patch_probabilities) / len(tuplet_patch_probabilities), 6)
            if tuplet_patch_probabilities else 0.5
        ),
        tuplet_patch_model=tuplet_patch_calibrator.model_version,
        tuplet_patch_threshold=tuplet_patch_calibrator.threshold,
        pitch_patch_measure_count=pitch_patch_measure_count,
        pitch_patch_event_count=pitch_patch_event_count,
        mean_pitch_patch_probability=(
            round(sum(pitch_patch_probabilities) / len(pitch_patch_probabilities), 6)
            if pitch_patch_probabilities else 0.5
        ),
        pitch_patch_model=pitch_patch.model_version,
        pitch_patch_threshold=pitch_patch_calibrator.threshold,
        rhythm_patch_measure_count=rhythm_patch_measure_count,
        rhythm_patch_event_count=rhythm_patch_event_count,
        mean_rhythm_patch_probability=(
            round(sum(rhythm_patch_probabilities) / len(rhythm_patch_probabilities), 6)
            if rhythm_patch_probabilities else 0.5
        ),
        rhythm_patch_model=rhythm_patch_calibrator.model_version,
        rhythm_patch_threshold=rhythm_patch_calibrator.threshold,
        event_kind_patch_measure_count=event_kind_patch_measure_count,
        event_kind_patch_event_count=event_kind_patch_event_count,
        mean_event_kind_patch_probability=(
            round(sum(event_kind_patch_probabilities) / len(event_kind_patch_probabilities), 6)
            if event_kind_patch_probabilities else 0.5
        ),
        event_kind_patch_model=event_kind_patch_calibrator.model_version,
        event_kind_patch_threshold=event_kind_patch_calibrator.threshold,
        event_kind_visual_guard_transaction_count=event_kind_visual_guard_transaction_count,
        event_kind_visual_guard_rejected_count=event_kind_visual_guard_rejected_count,
        mean_event_kind_visual_guard_probability=(
            round(sum(event_kind_visual_guard_probabilities) / len(event_kind_visual_guard_probabilities), 6)
            if event_kind_visual_guard_probabilities else 0.5
        ),
        event_kind_visual_guard_model=event_kind_visual_guard.model_version,
        event_kind_visual_guard_threshold=event_kind_visual_guard.threshold,
        attribute_patch_measure_count=attribute_patch_measure_count,
        attribute_patch_attribute_count=attribute_patch_attribute_count,
        mean_attribute_patch_probability=(
            round(sum(attribute_patch_probabilities) / len(attribute_patch_probabilities), 6)
            if attribute_patch_probabilities else 0.5
        ),
        attribute_patch_model=attribute_patch_calibrator.model_version,
        attribute_patch_threshold=attribute_patch_calibrator.threshold,
        barline_patch_measure_count=barline_patch_measure_count,
        barline_patch_location_count=barline_patch_location_count,
        barline_patch_repeat_count=barline_patch_repeat_count,
        mean_barline_patch_probability=(
            round(sum(barline_patch_probabilities) / len(barline_patch_probabilities), 6)
            if barline_patch_probabilities else 0.5
        ),
        barline_patch_model=barline_patch_calibrator.model_version,
        barline_patch_threshold=barline_patch_calibrator.threshold,
        tie_patch_measure_count=tie_patch_measure_count,
        tie_patch_event_count=tie_patch_event_count,
        mean_tie_patch_probability=(
            round(sum(tie_patch_probabilities) / len(tie_patch_probabilities), 6)
            if tie_patch_probabilities else 0.5
        ),
        tie_patch_model=tie_patch_calibrator.model_version,
        tie_patch_threshold=tie_patch_calibrator.threshold,
        tie_visual_guard_transaction_count=tie_visual_guard_transaction_count,
        tie_visual_guard_rejected_count=tie_visual_guard_rejected_count,
        mean_tie_visual_guard_probability=(
            round(sum(tie_visual_guard_probabilities) / len(tie_visual_guard_probabilities), 6)
            if tie_visual_guard_probabilities else 0.5
        ),
        tie_visual_guard_model=tie_visual_guard.model_version,
        tie_visual_guard_threshold=tie_visual_guard.threshold,
        slur_patch_measure_count=slur_patch_measure_count,
        slur_patch_event_count=slur_patch_event_count,
        slur_patch_arc_count=slur_patch_arc_count,
        mean_slur_patch_probability=(
            round(sum(slur_patch_probabilities) / len(slur_patch_probabilities), 6)
            if slur_patch_probabilities else 0.5
        ),
        slur_patch_model=slur_patch_calibrator.model_version,
        slur_patch_threshold=slur_patch_calibrator.threshold,
        articulation_patch_measure_count=articulation_patch_measure_count,
        articulation_patch_event_count=articulation_patch_event_count,
        articulation_patch_mark_count=articulation_patch_mark_count,
        mean_articulation_patch_probability=(
            round(sum(articulation_patch_probabilities) / len(articulation_patch_probabilities), 6)
            if articulation_patch_probabilities else 0.5
        ),
        articulation_patch_model=articulation_patch_calibrator.model_version,
        articulation_patch_threshold=articulation_patch_calibrator.threshold,
        accent_visual_guard_transaction_count=accent_visual_guard_transaction_count,
        accent_visual_guard_rejected_count=accent_visual_guard_rejected_count,
        mean_accent_visual_guard_probability=(
            round(sum(accent_visual_guard_probabilities) / len(accent_visual_guard_probabilities), 6)
            if accent_visual_guard_probabilities else 0.5
        ),
        accent_visual_guard_model=accent_visual_guard.model_version,
        accent_visual_guard_threshold=accent_visual_guard.threshold,
        ornament_patch_measure_count=ornament_patch_measure_count,
        ornament_patch_event_count=ornament_patch_event_count,
        ornament_patch_mark_count=ornament_patch_mark_count,
        mean_ornament_patch_probability=(
            round(sum(ornament_patch_probabilities) / len(ornament_patch_probabilities), 6)
            if ornament_patch_probabilities else 0.5
        ),
        ornament_patch_model=ornament_patch_calibrator.model_version,
        ornament_patch_threshold=ornament_patch_calibrator.threshold,
        grace_patch_measure_count=grace_patch_measure_count,
        grace_patch_event_count=grace_patch_event_count,
        grace_patch_added_count=grace_patch_added_count,
        grace_patch_removed_count=grace_patch_removed_count,
        mean_grace_patch_probability=(
            round(sum(grace_patch_probabilities) / len(grace_patch_probabilities), 6)
            if grace_patch_probabilities else 0.5
        ),
        grace_patch_model=grace_patch_calibrator.model_version,
        grace_patch_threshold=grace_patch_calibrator.threshold,
        lyric_patch_measure_count=lyric_patch_measure_count,
        lyric_patch_event_count=lyric_patch_event_count,
        lyric_patch_lyric_count=lyric_patch_lyric_count,
        mean_lyric_patch_probability=(
            round(sum(lyric_patch_probabilities) / len(lyric_patch_probabilities), 6)
            if lyric_patch_probabilities else 0.5
        ),
        lyric_patch_model=(
            lyric_patch_calibrator.model_version
            if lyric_patch_calibrator is not None
            else "disabled_out_of_scope"
        ),
        lyric_patch_threshold=(
            lyric_patch_calibrator.threshold
            if lyric_patch_calibrator is not None
            else 1.0
        ),
        direction_patch_measure_count=direction_patch_measure_count,
        direction_patch_direction_count=direction_patch_direction_count,
        mean_direction_patch_probability=(
            round(sum(direction_patch_probabilities) / len(direction_patch_probabilities), 6)
            if direction_patch_probabilities else 0.5
        ),
        direction_patch_model=direction_patch_calibrator.model_version,
        direction_patch_threshold=direction_patch_calibrator.threshold,
        cross_tie_patch_boundary_count=cross_tie_patch_boundary_count,
        cross_tie_patch_endpoint_count=cross_tie_patch_endpoint_count,
        mean_cross_tie_patch_probability=(
            round(sum(cross_tie_patch_probabilities) / len(cross_tie_patch_probabilities), 6)
            if cross_tie_patch_probabilities else 0.5
        ),
        cross_tie_patch_model=cross_tie_patch_calibrator.model_version,
        cross_tie_patch_threshold=cross_tie_patch_calibrator.threshold,
        cross_tie_patch_transaction_rejected_count=cross_tie_patch_transaction_rejected_count,
        cross_tie_boundaries=tuple(cross_tie_boundaries),
        patch_stage_rejected_count=patch_stage_rejected_count,
        patch_transaction_evaluated_count=patch_transaction_evaluated_count,
        patch_transaction_rejected_count=patch_transaction_rejected_count,
        mean_patch_transaction_probability=(
            round(sum(patch_transaction_probabilities) / len(patch_transaction_probabilities), 6)
            if patch_transaction_probabilities else 1.0
        ),
        patch_transaction_model=patch_transaction_calibrator.model_version,
        patch_transaction_threshold=patch_transaction_calibrator.threshold,
        event_presence_patch_measure_count=event_presence_patch_measure_count,
        event_presence_patch_inserted_event_count=event_presence_patch_inserted_event_count,
        event_presence_patch_deleted_event_count=event_presence_patch_deleted_event_count,
        mean_event_presence_patch_probability=(
            round(sum(event_presence_patch_probabilities) / len(event_presence_patch_probabilities), 6)
            if event_presence_patch_probabilities else 0.5
        ),
        event_presence_patch_model=event_presence_patch_calibrator.model_version,
        event_presence_patch_threshold=event_presence_patch_calibrator.threshold,
        event_presence_visual_guard_transaction_count=event_presence_visual_guard_transaction_count,
        event_presence_visual_guard_rejected_count=event_presence_visual_guard_rejected_count,
        mean_event_presence_visual_guard_probability=(
            round(
                sum(event_presence_visual_guard_probabilities)
                / len(event_presence_visual_guard_probabilities),
                6,
            )
            if event_presence_visual_guard_probabilities
            else 0.5
        ),
        event_presence_visual_guard_model=event_presence_visual_guard.model_version,
        event_presence_visual_guard_threshold=event_presence_visual_guard.threshold,
        event_presence_visual_guard_note_threshold=event_presence_visual_guard.thresholds["note"],
        event_presence_visual_guard_rest_threshold=event_presence_visual_guard.thresholds["rest"],
    )
