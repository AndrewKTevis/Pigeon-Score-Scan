from __future__ import annotations

"""Conservative measure-level consensus for simple dynamics and metronome marks.

The proposal changes only compact dynamic or metronome ``direction`` elements.  It may
use candidates with complementary pitch errors, but every voting sibling must agree on
the complete event lattice and on the full supported direction topology.  Correlated
preprocessing siblings collapse to one family vote and the bundled CPU model is
veto-only.
"""

import copy
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence

from lxml import etree

from .direction_xml import (
    SimpleDirection,
    normalized_simple_direction_topology,
    set_simple_direction_topology,
    without_simple_directions,
)
from .model_registry import load_verified_json
from .policy import DEFAULT_POLICY
from .score_ir import MeasureIR, NoteIR, measure_from_xml
from .tree_model import VerifiedRandomForestModel
from .variant_family import group_complete_families

FEATURE_NAMES = (
    "candidate_count_scaled",
    "eligible_family_count_scaled",
    "voting_family_count_scaled",
    "changed_direction_count_scaled",
    "changed_direction_ratio",
    "added_direction_ratio",
    "removed_direction_ratio",
    "dynamic_change_ratio",
    "metronome_change_ratio",
    "winner_family_support_ratio",
    "winner_margin_ratio",
    "template_family_support_ratio",
    "family_abstention_ratio",
    "winner_direction_count_scaled",
    "maximum_directions_per_onset_scaled",
    "mean_support_page_probability",
    "mean_support_measure_probability",
    "mean_support_visual_probability",
    "mean_support_event_probability",
    "mean_support_context_probability",
    "mean_support_ensemble_probability",
    "minimum_support_ensemble_probability",
    "mean_support_page_score_margin_scaled",
    "mean_support_vs_template_measure_probability",
    "mean_support_vs_template_visual_probability",
    "mean_support_vs_template_event_probability",
    "mean_support_vs_template_context_probability",
    "mean_support_vs_template_ensemble_probability",
)


@dataclass(frozen=True)
class DirectionPatchCandidate:
    variant: str
    family: str
    measure: etree._Element
    semantics: MeasureIR
    page_score: float
    page_probability: float
    measure_probability: float
    visual_probability: float
    event_probability: float
    context_probability: float
    ensemble_probability: float
    valid: bool


@dataclass(frozen=True)
class DirectionPatchInput:
    candidate_count: int
    eligible_family_count: int
    voting_family_count: int
    changed_direction_count: int
    base_direction_count: int
    winner_direction_count: int
    added_direction_count: int
    removed_direction_count: int
    dynamic_change_count: int
    metronome_change_count: int
    winner_family_count: int
    winner_margin_count: int
    template_family_count: int
    incomplete_family_count: int
    maximum_directions_per_onset: int
    mean_support_page_probability: float
    mean_support_measure_probability: float
    mean_support_visual_probability: float
    mean_support_event_probability: float
    mean_support_context_probability: float
    mean_support_ensemble_probability: float
    minimum_support_ensemble_probability: float
    mean_support_page_score_margin: float
    mean_support_vs_template_measure_probability: float
    mean_support_vs_template_visual_probability: float
    mean_support_vs_template_event_probability: float
    mean_support_vs_template_context_probability: float
    mean_support_vs_template_ensemble_probability: float

    def feature_vector(self) -> list[float]:
        def unit(value: float) -> float:
            return max(0.0, min(1.0, float(value)))

        def signed(value: float) -> float:
            return max(-1.0, min(1.0, float(value)))

        families = max(1, self.voting_family_count)
        changed = max(1, self.changed_direction_count)
        total = max(1, max(self.base_direction_count, self.winner_direction_count))
        return [
            unit(max(0, self.candidate_count - 1) / 7.0),
            unit(self.eligible_family_count / 4.0),
            unit(self.voting_family_count / 4.0),
            unit(self.changed_direction_count / max(1, DEFAULT_POLICY.direction_patch_max_changed_directions)),
            unit(self.changed_direction_count / total),
            unit(self.added_direction_count / changed),
            unit(self.removed_direction_count / changed),
            unit(self.dynamic_change_count / changed),
            unit(self.metronome_change_count / changed),
            unit(self.winner_family_count / families),
            unit(self.winner_margin_count / families),
            unit(self.template_family_count / families),
            unit(self.incomplete_family_count / max(1, self.eligible_family_count + self.incomplete_family_count)),
            unit(self.winner_direction_count / max(1, DEFAULT_POLICY.direction_patch_max_directions)),
            unit(self.maximum_directions_per_onset / 2.0),
            unit(self.mean_support_page_probability),
            unit(self.mean_support_measure_probability),
            unit(self.mean_support_visual_probability),
            unit(self.mean_support_event_probability),
            unit(self.mean_support_context_probability),
            unit(self.mean_support_ensemble_probability),
            unit(self.minimum_support_ensemble_probability),
            signed(self.mean_support_page_score_margin / 100.0),
            signed(self.mean_support_vs_template_measure_probability),
            signed(self.mean_support_vs_template_visual_probability),
            signed(self.mean_support_vs_template_event_probability),
            signed(self.mean_support_vs_template_context_probability),
            signed(self.mean_support_vs_template_ensemble_probability),
        ]


@dataclass(frozen=True)
class DirectionPatchCalibration:
    probability: float
    threshold: float
    accepted: bool
    model_version: str
    target_precision: float


@dataclass(frozen=True)
class DirectionPatchResult:
    patched_measure: etree._Element | None
    changed_direction_count: int
    changed_kinds: tuple[str, ...]
    probability: float
    threshold: float
    accepted: bool
    reason: str
    input: DirectionPatchInput | None = None
    model_version: str = "disabled"


class DirectionPatchCalibrator:
    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).with_name("resources") / "direction_patch_calibrator.json"
        loaded = load_verified_json(model_path, "direction_patch_calibration")
        payload = loaded.payload
        self.model = VerifiedRandomForestModel.load(
            model_path,
            "direction_patch_calibration",
            FEATURE_NAMES,
            loaded=loaded,
        )
        try:
            stored_threshold = float(payload.get("auto_patch_threshold", 1.0))
            target_precision = float(payload.get("target_precision", 1.0))
        except (TypeError, ValueError, OverflowError):
            stored_threshold = 1.0
            target_precision = 1.0
        self.threshold = max(
            float(DEFAULT_POLICY.direction_patch_probability_floor),
            max(0.0, min(1.0, stored_threshold)),
        )
        self.target_precision = max(0.0, min(1.0, target_precision))
        self.model_verified = self.model.verified and loaded.verified
        self.model_status = self.model.status if self.model.enabled else loaded.status
        self.model_version = self.model.model_version
        self.enabled = self.model.enabled

    def predict_probability(self, item: DirectionPatchInput) -> float:
        return self.model.predict(item.feature_vector(), neutral=0.5)

    def calibrate(self, item: DirectionPatchInput) -> DirectionPatchCalibration:
        probability = self.predict_probability(item)
        return DirectionPatchCalibration(
            probability=round(probability, 6),
            threshold=round(self.threshold, 6),
            accepted=bool(self.enabled and self.model_verified and probability >= self.threshold),
            model_version=self.model_version,
            target_precision=round(self.target_precision, 6),
        )


def _mean(values: Sequence[float], default: float = 0.5) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else default


def _note_skeleton(note: NoteIR) -> tuple[object, ...]:
    return (
        note.onset,
        note.duration,
        note.voice,
        note.rest,
        note.chord,
        note.grace,
        note.note_type,
        note.dots,
        note.ties,
        note.slurs,
        note.articulations,
        note.ornaments,
        note.tuple_ratio,
    )


def _measure_skeleton(measure: MeasureIR) -> tuple[object, ...]:
    return (
        measure.time_signature,
        measure.key_signature,
        measure.clef,
        tuple(_note_skeleton(note) for note in measure.notes),
    )


def _duration_total(measure: MeasureIR) -> Fraction:
    return sum(
        (note.duration for note in measure.notes if not note.grace and not note.chord),
        Fraction(0, 1),
    )


def _supported_measure_xml(measure: etree._Element, semantics: MeasureIR) -> bool:
    topology = normalized_simple_direction_topology(measure, semantics.divisions)
    return bool(
        semantics.voice_count == 1
        and semantics.expected_duration is not None
        and _duration_total(semantics) == semantics.expected_duration
        and semantics.notes
        and len(semantics.notes) <= DEFAULT_POLICY.direction_patch_max_events
        and measure.find("backup") is None
        and measure.find("forward") is None
        and topology is not None
        and len(topology) <= DEFAULT_POLICY.direction_patch_max_directions
        and all(Fraction(0, 1) <= item.onset < semantics.expected_duration for item in topology)
    )


def _representative(items: Sequence[DirectionPatchCandidate]) -> DirectionPatchCandidate:
    return max(
        items,
        key=lambda item: (
            item.ensemble_probability,
            item.event_probability,
            item.visual_probability,
            item.context_probability,
            item.measure_probability,
            item.page_score,
            item.variant,
        ),
    )


def _topology_counter(topology: Sequence[SimpleDirection]) -> Counter[SimpleDirection]:
    return Counter(topology)


def _changed_counts(
    base: Sequence[SimpleDirection],
    winner: Sequence[SimpleDirection],
) -> tuple[int, int, int, int, tuple[str, ...]]:
    base_counter = _topology_counter(base)
    winner_counter = _topology_counter(winner)
    added_counter = winner_counter - base_counter
    removed_counter = base_counter - winner_counter
    added = sum(added_counter.values())
    removed = sum(removed_counter.values())
    changed_items = list(added_counter.elements()) + list(removed_counter.elements())
    dynamic = sum(item.kind == "dynamic" for item in changed_items)
    metronome = sum(item.kind == "metronome" for item in changed_items)
    kinds = tuple(sorted({item.kind for item in changed_items}))
    return added, removed, dynamic, metronome, kinds


def propose_direction_patch(
    candidates: Sequence[DirectionPatchCandidate],
    *,
    template_index: int,
    missing_candidate_count: int,
    calibrator: DirectionPatchCalibrator | None = None,
    base_measure: etree._Element | None = None,
) -> DirectionPatchResult:
    if not candidates or template_index < 0 or template_index >= len(candidates):
        return DirectionPatchResult(None, 0, (), 0.5, 1.0, False, "invalid_input")
    if missing_candidate_count:
        return DirectionPatchResult(None, 0, (), 0.5, 1.0, False, "alignment_gap")
    template = candidates[template_index]
    if not template.valid:
        return DirectionPatchResult(None, 0, (), 0.5, 1.0, False, "invalid_template")

    family_members, incomplete_families = group_complete_families(
        candidates,
        family_of=lambda item: item.family,
        valid_of=lambda item: item.valid,
    )
    if len(family_members) < DEFAULT_POLICY.direction_patch_minimum_families:
        return DirectionPatchResult(None, 0, (), 0.5, 1.0, False, "insufficient_families")

    source_base = base_measure if base_measure is not None else template.measure
    inherited: dict[str, object] = {
        "divisions": template.semantics.divisions,
        "time": template.semantics.time_signature,
        "key": template.semantics.key_signature,
        "clef": template.semantics.clef,
    }
    base_semantics, _ = measure_from_xml(source_base, inherited)
    if not _supported_measure_xml(source_base, base_semantics):
        return DirectionPatchResult(None, 0, (), 0.5, 1.0, False, "unsupported_base_measure")
    base_topology = normalized_simple_direction_topology(source_base, base_semantics.divisions)
    if base_topology is None:
        return DirectionPatchResult(None, 0, (), 0.5, 1.0, False, "invalid_base_topology")
    base_skeleton = _measure_skeleton(base_semantics)

    votes: dict[str, tuple[SimpleDirection, ...]] = {}
    representatives: dict[str, DirectionPatchCandidate] = {}
    abstentions = len(incomplete_families)
    for family, members in family_members.items():
        sibling_values: set[tuple[SimpleDirection, ...]] = set()
        supported = True
        for member in members:
            if (
                not _supported_measure_xml(member.measure, member.semantics)
                or _measure_skeleton(member.semantics) != base_skeleton
            ):
                supported = False
                break
            topology = normalized_simple_direction_topology(member.measure, member.semantics.divisions)
            if topology is None:
                supported = False
                break
            sibling_values.add(topology)
        if not supported or len(sibling_values) != 1:
            abstentions += 1
            continue
        votes[family] = next(iter(sibling_values))
        representatives[family] = _representative(members)

    if len(votes) < DEFAULT_POLICY.direction_patch_minimum_families:
        return DirectionPatchResult(None, 0, (), 0.5, 1.0, False, "insufficient_voting_families")
    grouped = Counter(votes.values())
    ranked = sorted(grouped.items(), key=lambda item: (item[1], repr(item[0])), reverse=True)
    winner, winner_count = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    if not (
        winner_count >= DEFAULT_POLICY.direction_patch_minimum_supporting_families
        and winner_count > len(votes) / 2
        and winner_count - runner_up >= 1
        and len(winner) <= DEFAULT_POLICY.direction_patch_max_directions
    ):
        return DirectionPatchResult(None, 0, (), 0.5, 1.0, False, "no_strict_majority")
    if winner == base_topology:
        return DirectionPatchResult(None, 0, (), 0.5, 1.0, False, "no_direction_change")

    added, removed, dynamic_changes, metronome_changes, changed_kinds = _changed_counts(base_topology, winner)
    changed_count = added + removed
    if changed_count <= 0:
        return DirectionPatchResult(None, 0, (), 0.5, 1.0, False, "no_direction_change")
    if changed_count > DEFAULT_POLICY.direction_patch_max_changed_directions:
        return DirectionPatchResult(None, changed_count, changed_kinds, 0.5, 1.0, False, "change_scope_too_large")

    patched = copy.deepcopy(source_base)
    try:
        before_without = without_simple_directions(patched, base_semantics.divisions)
        set_simple_direction_topology(patched, winner, base_semantics.divisions)
        after_without = without_simple_directions(patched, base_semantics.divisions)
    except (TypeError, ValueError, OverflowError):
        return DirectionPatchResult(None, changed_count, changed_kinds, 0.5, 1.0, False, "xml_topology_error")
    if before_without != after_without:
        return DirectionPatchResult(None, changed_count, changed_kinds, 0.5, 1.0, False, "unrelated_measure_xml_changed")

    parsed, _ = measure_from_xml(patched, inherited)
    parsed_topology = normalized_simple_direction_topology(patched, parsed.divisions)
    if parsed_topology != winner or _measure_skeleton(parsed) != base_skeleton:
        return DirectionPatchResult(None, changed_count, changed_kinds, 0.5, 1.0, False, "post_patch_validation_failed")

    support_families = {family for family, value in votes.items() if value == winner}
    support = [representatives[family] for family in sorted(support_families)]
    maximum_per_onset = max(
        Counter((item.onset, item.placement) for item in winner).values(),
        default=0,
    )
    item = DirectionPatchInput(
        candidate_count=sum(len(members) for members in family_members.values()),
        eligible_family_count=len(family_members),
        voting_family_count=len(votes),
        changed_direction_count=changed_count,
        base_direction_count=len(base_topology),
        winner_direction_count=len(winner),
        added_direction_count=added,
        removed_direction_count=removed,
        dynamic_change_count=dynamic_changes,
        metronome_change_count=metronome_changes,
        winner_family_count=winner_count,
        winner_margin_count=winner_count - runner_up,
        template_family_count=grouped.get(base_topology, 0),
        incomplete_family_count=abstentions,
        maximum_directions_per_onset=maximum_per_onset,
        mean_support_page_probability=_mean([row.page_probability for row in support]),
        mean_support_measure_probability=_mean([row.measure_probability for row in support]),
        mean_support_visual_probability=_mean([row.visual_probability for row in support]),
        mean_support_event_probability=_mean([row.event_probability for row in support]),
        mean_support_context_probability=_mean([row.context_probability for row in support]),
        mean_support_ensemble_probability=_mean([row.ensemble_probability for row in support]),
        minimum_support_ensemble_probability=min(row.ensemble_probability for row in support),
        mean_support_page_score_margin=_mean([row.page_score - template.page_score for row in support], 0.0),
        mean_support_vs_template_measure_probability=_mean([row.measure_probability - template.measure_probability for row in support], 0.0),
        mean_support_vs_template_visual_probability=_mean([row.visual_probability - template.visual_probability for row in support], 0.0),
        mean_support_vs_template_event_probability=_mean([row.event_probability - template.event_probability for row in support], 0.0),
        mean_support_vs_template_context_probability=_mean([row.context_probability - template.context_probability for row in support], 0.0),
        mean_support_vs_template_ensemble_probability=_mean([row.ensemble_probability - template.ensemble_probability for row in support], 0.0),
    )
    active_calibrator = calibrator or DirectionPatchCalibrator()
    calibration = active_calibrator.calibrate(item)
    if not calibration.accepted:
        return DirectionPatchResult(
            None,
            changed_count,
            changed_kinds,
            calibration.probability,
            calibration.threshold,
            False,
            "model_guard",
            item,
            calibration.model_version,
        )
    return DirectionPatchResult(
        patched,
        changed_count,
        changed_kinds,
        calibration.probability,
        calibration.threshold,
        True,
        "accepted",
        item,
        calibration.model_version,
    )
