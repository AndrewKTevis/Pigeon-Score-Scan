from __future__ import annotations

"""Conservative consensus for simple grace-versus-regular note topology.

A false grace classification changes whether an event advances the MusicXML cursor and
can shift the rhythm of the complete measure.  This repair is intentionally narrow: it
keeps the existing pitched note sequence fixed, votes only on simple attribute-free
``<grace/>`` state plus the owned duration/type/dot fields, requires three independent
preprocessing families, and accepts only a proposal which restores exact meter closure.
The bundled CPU model is veto-only.
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

from .grace_xml import (
    GraceState,
    GraceTopology,
    normalized_grace_topology,
    set_grace_topology,
    without_grace_rhythm,
)
from .model_registry import load_verified_json
from .policy import DEFAULT_POLICY
from .score_ir import MeasureIR, NoteIR, expected_note_duration, measure_from_xml
from .tree_model import VerifiedRandomForestModel
from .variant_family import group_complete_families

FEATURE_NAMES = (
    "candidate_count_scaled",
    "eligible_family_count_scaled",
    "voting_family_count_scaled",
    "changed_event_count_scaled",
    "changed_event_ratio",
    "added_grace_ratio",
    "removed_grace_ratio",
    "winner_family_support_ratio",
    "winner_margin_ratio",
    "template_family_support_ratio",
    "family_abstention_ratio",
    "minimum_content_family_support_ratio",
    "mean_content_family_support_ratio",
    "minimum_content_margin_ratio",
    "base_duration_error_scaled",
    "patched_duration_error_scaled",
    "duration_error_improvement_scaled",
    "mean_support_page_probability",
    "mean_support_measure_probability",
    "mean_support_visual_probability",
    "mean_support_event_probability",
    "mean_support_context_probability",
    "mean_support_ensemble_probability",
    "minimum_support_ensemble_probability",
    "mean_support_page_score_margin_scaled",
    "mean_support_vs_template_ensemble_probability",
)


@dataclass(frozen=True)
class GracePatchCandidate:
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
class GracePatchInput:
    candidate_count: int
    eligible_family_count: int
    voting_family_count: int
    changed_event_count: int
    total_event_count: int
    added_grace_count: int
    removed_grace_count: int
    winner_family_count: int
    winner_margin_count: int
    template_family_count: int
    incomplete_family_count: int
    minimum_content_family_count: int
    mean_content_family_count: float
    minimum_content_margin_count: int
    base_duration_error: float
    patched_duration_error: float
    duration_error_improvement: float
    mean_support_page_probability: float
    mean_support_measure_probability: float
    mean_support_visual_probability: float
    mean_support_event_probability: float
    mean_support_context_probability: float
    mean_support_ensemble_probability: float
    minimum_support_ensemble_probability: float
    mean_support_page_score_margin: float
    mean_support_vs_template_ensemble_probability: float

    def feature_vector(self) -> list[float]:
        def unit(value: float) -> float:
            return max(0.0, min(1.0, float(value)))

        def signed(value: float) -> float:
            return max(-1.0, min(1.0, float(value)))

        families = max(1, self.voting_family_count)
        changed = max(1, self.changed_event_count)
        return [
            unit(max(0, self.candidate_count - 1) / 7.0),
            unit(self.eligible_family_count / 4.0),
            unit(self.voting_family_count / 4.0),
            unit(self.changed_event_count / max(1, DEFAULT_POLICY.grace_patch_max_changed_events)),
            unit(self.changed_event_count / max(1, self.total_event_count)),
            unit(self.added_grace_count / changed),
            unit(self.removed_grace_count / changed),
            unit(self.winner_family_count / families),
            unit(self.winner_margin_count / families),
            unit(self.template_family_count / families),
            unit(self.incomplete_family_count / max(1, self.eligible_family_count + self.incomplete_family_count)),
            unit(self.minimum_content_family_count / families),
            unit(self.mean_content_family_count / families),
            unit(self.minimum_content_margin_count / families),
            unit(self.base_duration_error / 4.0),
            unit(self.patched_duration_error / 4.0),
            signed(self.duration_error_improvement / 4.0),
            unit(self.mean_support_page_probability),
            unit(self.mean_support_measure_probability),
            unit(self.mean_support_visual_probability),
            unit(self.mean_support_event_probability),
            unit(self.mean_support_context_probability),
            unit(self.mean_support_ensemble_probability),
            unit(self.minimum_support_ensemble_probability),
            signed(self.mean_support_page_score_margin / 100.0),
            signed(self.mean_support_vs_template_ensemble_probability),
        ]


@dataclass(frozen=True)
class GracePatchCalibration:
    probability: float
    threshold: float
    accepted: bool
    model_version: str
    target_precision: float


@dataclass(frozen=True)
class GracePatchResult:
    patched_measure: etree._Element | None
    changed_event_indices: tuple[int, ...]
    added_grace_count: int
    removed_grace_count: int
    probability: float
    threshold: float
    accepted: bool
    reason: str
    input: GracePatchInput | None = None
    model_version: str = "disabled"


class GracePatchCalibrator:
    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).with_name("resources") / "grace_patch_calibrator.json"
        loaded = load_verified_json(model_path, "grace_patch_calibration")
        payload = loaded.payload
        self.model = VerifiedRandomForestModel.load(
            model_path,
            "grace_patch_calibration",
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
            float(DEFAULT_POLICY.grace_patch_probability_floor),
            max(0.0, min(1.0, stored_threshold)),
        )
        self.target_precision = max(0.0, min(1.0, target_precision))
        self.model_verified = self.model.verified and loaded.verified
        self.model_status = self.model.status if self.model.enabled else loaded.status
        self.model_version = self.model.model_version
        self.enabled = self.model.enabled

    def predict_probability(self, item: GracePatchInput) -> float:
        return self.model.predict(item.feature_vector(), neutral=0.5)

    def calibrate(self, item: GracePatchInput) -> GracePatchCalibration:
        probability = self.predict_probability(item)
        return GracePatchCalibration(
            probability=round(probability, 6),
            threshold=round(self.threshold, 6),
            accepted=bool(self.enabled and self.model_verified and probability >= self.threshold),
            model_version=self.model_version,
            target_precision=round(self.target_precision, 6),
        )


def _mean(values: Sequence[float], default: float = 0.5) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else default


def _pitch_key(note: NoteIR) -> tuple[object, ...] | None:
    if note.rest or note.pitch is None:
        return None
    return (*note.pitch.stable_tuple(), note.accidental.strip().casefold())


def _event_shape(note: NoteIR) -> tuple[object, ...]:
    return (
        note.voice,
        note.rest,
        note.chord,
        note.ties,
        note.slurs,
        note.articulations,
        note.ornaments,
        note.tuple_ratio,
    )


def _duration_total(measure: MeasureIR) -> Fraction:
    return sum(
        (note.duration for note in measure.notes if not note.grace and not note.chord),
        Fraction(0, 1),
    )


def _duration_error(measure: MeasureIR) -> float:
    expected = measure.expected_duration
    if expected is None:
        return 4.0
    return min(4.0, float(abs(_duration_total(measure) - expected)))


def _supported_measure_xml(measure: etree._Element, semantics: MeasureIR) -> bool:
    notes = measure.findall("note")
    topology = normalized_grace_topology(notes, semantics.divisions)
    if (
        semantics.voice_count != 1
        or semantics.expected_duration is None
        or not semantics.notes
        or len(semantics.notes) > DEFAULT_POLICY.grace_patch_max_events
        or len(notes) != len(semantics.notes)
        or measure.find("backup") is not None
        or measure.find("forward") is not None
        or topology is None
    ):
        return False
    for note, node in zip(semantics.notes, notes, strict=True):
        if (
            note.rest
            or note.pitch is None
            or note.chord
            or note.tuple_ratio is not None
            or node.find("unpitched") is not None
            or node.find("cue") is not None
        ):
            return False
    return True


def _representative(items: Sequence[GracePatchCandidate]) -> GracePatchCandidate:
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


def _flags(topology: GraceTopology) -> tuple[bool, ...]:
    return tuple(state.grace for state in topology)


def propose_grace_patch(
    candidates: Sequence[GracePatchCandidate],
    *,
    template_index: int,
    missing_candidate_count: int,
    calibrator: GracePatchCalibrator | None = None,
    base_measure: etree._Element | None = None,
) -> GracePatchResult:
    empty = GracePatchResult(None, (), 0, 0, 0.5, 1.0, False, "invalid_input")
    if not candidates or template_index < 0 or template_index >= len(candidates):
        return empty
    if missing_candidate_count:
        return GracePatchResult(None, (), 0, 0, 0.5, 1.0, False, "alignment_gap")
    template = candidates[template_index]
    if not template.valid:
        return GracePatchResult(None, (), 0, 0, 0.5, 1.0, False, "invalid_template")

    family_members, incomplete_families = group_complete_families(
        candidates,
        family_of=lambda item: item.family,
        valid_of=lambda item: item.valid,
    )
    if len(family_members) < DEFAULT_POLICY.grace_patch_minimum_families:
        return GracePatchResult(None, (), 0, 0, 0.5, 1.0, False, "insufficient_families")

    source_base = base_measure if base_measure is not None else template.measure
    inherited: dict[str, object] = {
        "divisions": template.semantics.divisions,
        "time": template.semantics.time_signature,
        "key": template.semantics.key_signature,
        "clef": template.semantics.clef,
    }
    base_semantics, _ = measure_from_xml(source_base, inherited)
    if not _supported_measure_xml(source_base, base_semantics):
        return GracePatchResult(None, (), 0, 0, 0.5, 1.0, False, "unsupported_base_measure")
    base_topology = normalized_grace_topology(source_base.findall("note"), base_semantics.divisions)
    if base_topology is None:
        return GracePatchResult(None, (), 0, 0, 0.5, 1.0, False, "invalid_base_topology")

    candidate_positions = {id(item): index for index, item in enumerate(candidates)}
    candidate_topologies: dict[int, GraceTopology] = {}
    for index, item in enumerate(candidates):
        topology = normalized_grace_topology(item.measure.findall("note"), item.semantics.divisions)
        if topology is not None:
            candidate_topologies[index] = topology

    family_flags: dict[str, tuple[bool, ...]] = {}
    family_topologies: dict[str, GraceTopology] = {}
    family_representatives: dict[str, GracePatchCandidate] = {}
    for family, members in family_members.items():
        sibling_flags: set[tuple[bool, ...]] = set()
        sibling_topologies: list[GraceTopology] = []
        supported = True
        for member in members:
            index = candidate_positions.get(id(member))
            topology = candidate_topologies.get(index) if index is not None else None
            if (
                topology is None
                or len(member.semantics.notes) != len(base_semantics.notes)
                or len(topology) != len(base_topology)
                or any(
                    _event_shape(left) != _event_shape(right)
                    for left, right in zip(member.semantics.notes, base_semantics.notes, strict=True)
                )
            ):
                supported = False
                break
            sibling_flags.add(_flags(topology))
            sibling_topologies.append(topology)
        if not supported or len(sibling_flags) != 1:
            continue
        # Siblings may encode a regular duration using different divisions, but their
        # semantic state must still agree before the family obtains a content vote.
        if len(set(sibling_topologies)) != 1:
            continue
        family_flags[family] = next(iter(sibling_flags))
        family_topologies[family] = sibling_topologies[0]
        family_representatives[family] = _representative(members)

    if len(family_flags) < DEFAULT_POLICY.grace_patch_minimum_families:
        return GracePatchResult(None, (), 0, 0, 0.5, 1.0, False, "insufficient_voting_families")
    grouped_flags = Counter(family_flags.values())
    ranked_flags = sorted(grouped_flags.items(), key=lambda item: (item[1], repr(item[0])), reverse=True)
    winner_flags, winner_count = ranked_flags[0]
    runner_up = ranked_flags[1][1] if len(ranked_flags) > 1 else 0
    if not (
        winner_count >= DEFAULT_POLICY.grace_patch_minimum_supporting_families
        and winner_count > len(family_flags) / 2
        and winner_count - runner_up >= 1
    ):
        return GracePatchResult(None, (), 0, 0, 0.5, 1.0, False, "no_grace_majority")

    base_flags = _flags(base_topology)
    changed = tuple(index for index, value in enumerate(winner_flags) if value != base_flags[index])
    if not changed:
        return GracePatchResult(None, (), 0, 0, 0.5, 1.0, False, "no_grace_change")
    if (
        len(changed) > DEFAULT_POLICY.grace_patch_max_changed_events
        or len(changed) / max(1, len(base_topology)) > DEFAULT_POLICY.grace_patch_max_changed_ratio
    ):
        return GracePatchResult(None, changed, 0, 0, 0.5, 1.0, False, "change_scope_too_large")

    winner_families = {family for family, flags in family_flags.items() if flags == winner_flags}
    proposal = list(base_topology)
    content_counts: list[int] = []
    content_margins: list[int] = []
    content_support_sets: list[set[str]] = []
    for event_index in changed:
        values: dict[str, GraceState] = {}
        base_pitch = _pitch_key(base_semantics.notes[event_index])
        for family in winner_families:
            members = family_members[family]
            state = family_topologies[family][event_index]
            if all(
                _pitch_key(member.semantics.notes[event_index]) == base_pitch
                and family_topologies[family][event_index] == state
                for member in members
            ):
                values[family] = state
        grouped = Counter(values.values())
        ranked = sorted(grouped.items(), key=lambda item: (item[1], repr(item[0])), reverse=True)
        if not ranked:
            return GracePatchResult(None, changed, 0, 0, 0.5, 1.0, False, "missing_content_support")
        state, count = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0
        if not (
            count >= DEFAULT_POLICY.grace_patch_minimum_supporting_families
            and count > len(values) / 2
            and count - second >= 1
            and state.grace == winner_flags[event_index]
        ):
            return GracePatchResult(None, changed, 0, 0, 0.5, 1.0, False, "insufficient_content_support")
        proposal[event_index] = state
        content_counts.append(count)
        content_margins.append(count - second)
        content_support_sets.append({family for family, value in values.items() if value == state})

    common_support = set.intersection(*content_support_sets)
    if len(common_support) < DEFAULT_POLICY.grace_patch_minimum_supporting_families:
        return GracePatchResult(None, changed, 0, 0, 0.5, 1.0, False, "inconsistent_cross_event_support")

    added = sum(1 for index in changed if proposal[index].grace and not base_topology[index].grace)
    removed = sum(1 for index in changed if not proposal[index].grace and base_topology[index].grace)
    patched = copy.deepcopy(source_base)
    patched_notes = patched.findall("note")
    try:
        before_unrelated = [without_grace_rhythm(note) for note in patched_notes]
        set_grace_topology(patched_notes, proposal, base_semantics.divisions)
        after_unrelated = [without_grace_rhythm(note) for note in patched.findall("note")]
    except ValueError:
        return GracePatchResult(None, changed, added, removed, 0.5, 1.0, False, "xml_topology_error")
    if before_unrelated != after_unrelated:
        return GracePatchResult(None, changed, added, removed, 0.5, 1.0, False, "unrelated_event_xml_changed")

    parsed, _ = measure_from_xml(patched, inherited)
    parsed_topology = normalized_grace_topology(patched.findall("note"), parsed.divisions)
    if parsed_topology != tuple(proposal) or len(parsed.notes) != len(base_semantics.notes):
        return GracePatchResult(None, changed, added, removed, 0.5, 1.0, False, "post_patch_validation_failed")
    if any(
        _event_shape(left) != _event_shape(right) or _pitch_key(left) != _pitch_key(right)
        for left, right in zip(parsed.notes, base_semantics.notes, strict=True)
    ):
        return GracePatchResult(None, changed, added, removed, 0.5, 1.0, False, "event_identity_changed")
    for note in parsed.notes:
        if note.grace:
            if note.duration != 0:
                return GracePatchResult(None, changed, added, removed, 0.5, 1.0, False, "grace_duration_invalid")
        else:
            expected = expected_note_duration(note)
            if expected is None or expected != note.duration:
                return GracePatchResult(None, changed, added, removed, 0.5, 1.0, False, "type_duration_mismatch")

    base_error = _duration_error(base_semantics)
    patched_error = _duration_error(parsed)
    if patched_error > 1e-9 or patched_error >= base_error:
        return GracePatchResult(None, changed, added, removed, 0.5, 1.0, False, "meter_not_improved")

    support = [family_representatives[family] for family in sorted(common_support)]
    item = GracePatchInput(
        candidate_count=sum(len(members) for members in family_members.values()),
        eligible_family_count=len(family_members),
        voting_family_count=len(family_flags),
        changed_event_count=len(changed),
        total_event_count=len(base_topology),
        added_grace_count=added,
        removed_grace_count=removed,
        winner_family_count=winner_count,
        winner_margin_count=winner_count - runner_up,
        template_family_count=grouped_flags.get(base_flags, 0),
        incomplete_family_count=len(incomplete_families) + max(0, len(family_members) - len(family_flags)),
        minimum_content_family_count=min(content_counts),
        mean_content_family_count=_mean(content_counts, 0.0),
        minimum_content_margin_count=min(content_margins),
        base_duration_error=base_error,
        patched_duration_error=patched_error,
        duration_error_improvement=base_error - patched_error,
        mean_support_page_probability=_mean([row.page_probability for row in support]),
        mean_support_measure_probability=_mean([row.measure_probability for row in support]),
        mean_support_visual_probability=_mean([row.visual_probability for row in support]),
        mean_support_event_probability=_mean([row.event_probability for row in support]),
        mean_support_context_probability=_mean([row.context_probability for row in support]),
        mean_support_ensemble_probability=_mean([row.ensemble_probability for row in support]),
        minimum_support_ensemble_probability=min(row.ensemble_probability for row in support),
        mean_support_page_score_margin=_mean([row.page_score - template.page_score for row in support], 0.0),
        mean_support_vs_template_ensemble_probability=_mean(
            [row.ensemble_probability - template.ensemble_probability for row in support],
            0.0,
        ),
    )
    active_calibrator = calibrator or GracePatchCalibrator()
    calibration = active_calibrator.calibrate(item)
    if not calibration.accepted:
        return GracePatchResult(
            None,
            changed,
            added,
            removed,
            calibration.probability,
            calibration.threshold,
            False,
            "model_guard",
            item,
            calibration.model_version,
        )
    return GracePatchResult(
        patched,
        changed,
        added,
        removed,
        calibration.probability,
        calibration.threshold,
        True,
        "accepted",
        item,
        calibration.model_version,
    )
