from __future__ import annotations

"""Conservative consensus for simple 3:2 tuplet topology.

The repair only toggles ``time-modification`` and matching visual tuplet endpoints on
an existing fixed event sequence.  It is deliberately limited to complete,
monophonic measures and contiguous groups of exactly three equal-duration events.
Independent preprocessing families vote once; split or invalid siblings abstain; a
verified CPU model may only veto an otherwise valid deterministic proposal.
"""

import copy
import math
import statistics
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence

from lxml import etree

from .model_registry import load_verified_json
from .policy import DEFAULT_POLICY
from .score_ir import MeasureIR, NoteIR, measure_from_xml
from .tree_model import VerifiedRandomForestModel
from .tuplet_xml import read_simple_tuplet_state, set_simple_tuplet_state
from .variant_family import group_complete_families

TupletTopology = tuple[tuple[int, int], ...]

FEATURE_NAMES = (
    "candidate_count_scaled",
    "eligible_family_count_scaled",
    "voting_family_count_scaled",
    "changed_event_count_scaled",
    "changed_event_ratio",
    "added_group_ratio",
    "removed_group_ratio",
    "winner_family_support_ratio",
    "winner_margin_ratio",
    "template_family_support_ratio",
    "family_abstention_ratio",
    "winner_group_count_scaled",
    "minimum_group_pitch_span_scaled",
    "maximum_group_pitch_span_scaled",
    "template_duration_error_ratio",
    "patched_duration_error_ratio",
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
class TupletPatchCandidate:
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
class TupletPatchInput:
    candidate_count: int
    eligible_family_count: int
    voting_family_count: int
    changed_event_count: int
    total_event_count: int
    added_group_count: int
    removed_group_count: int
    winner_family_count: int
    runner_up_family_count: int
    template_family_count: int
    incomplete_family_count: int
    winner_group_count: int
    minimum_group_pitch_span: int
    maximum_group_pitch_span: int
    expected_measure_duration: float
    template_duration_error: float
    patched_duration_error: float
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

        events = max(1, self.total_event_count)
        families = max(1, self.voting_family_count)
        groups = max(1, self.winner_group_count + self.removed_group_count)
        expected = max(1e-9, self.expected_measure_duration)
        return [
            unit(max(0, self.candidate_count - 1) / 7.0),
            unit(self.eligible_family_count / 4.0),
            unit(self.voting_family_count / 4.0),
            unit(self.changed_event_count / 12.0),
            unit(self.changed_event_count / events),
            unit(self.added_group_count / groups),
            unit(self.removed_group_count / groups),
            unit(self.winner_family_count / families),
            unit((self.winner_family_count - self.runner_up_family_count) / families),
            unit(self.template_family_count / families),
            unit(self.incomplete_family_count / max(1, self.eligible_family_count + self.incomplete_family_count)),
            unit(self.winner_group_count / max(1, DEFAULT_POLICY.tuplet_patch_max_groups)),
            unit(self.minimum_group_pitch_span / 36.0),
            unit(self.maximum_group_pitch_span / 36.0),
            unit(self.template_duration_error / expected),
            unit(self.patched_duration_error / expected),
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
class TupletPatchCalibration:
    probability: float
    threshold: float
    accepted: bool
    model_version: str
    target_precision: float


@dataclass(frozen=True)
class TupletPatchResult:
    patched_measure: etree._Element | None
    changed_event_indices: tuple[int, ...]
    changed_group_count: int
    probability: float
    threshold: float
    accepted: bool
    reason: str
    input: TupletPatchInput | None = None
    model_version: str = "disabled"


class TupletPatchCalibrator:
    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).with_name("resources") / "tuplet_patch_calibrator.json"
        loaded = load_verified_json(model_path, "tuplet_patch_calibration")
        payload = loaded.payload
        self.model = VerifiedRandomForestModel.load(
            model_path,
            "tuplet_patch_calibration",
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
            float(DEFAULT_POLICY.tuplet_patch_probability_floor),
            max(0.0, min(1.0, stored_threshold)),
        )
        self.target_precision = max(0.0, min(1.0, target_precision))
        self.model_verified = self.model.verified and loaded.verified
        self.model_status = self.model.status if self.model.enabled else loaded.status
        self.model_version = self.model.model_version
        self.enabled = self.model.enabled

    def predict_probability(self, item: TupletPatchInput) -> float:
        return self.model.predict(item.feature_vector(), neutral=0.5)

    def calibrate(self, item: TupletPatchInput) -> TupletPatchCalibration:
        probability = self.predict_probability(item)
        accepted = bool(self.enabled and self.model_verified and probability >= self.threshold)
        return TupletPatchCalibration(
            probability=round(probability, 6),
            threshold=round(self.threshold, 6),
            accepted=accepted,
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
        note.pitch.stable_tuple() if note.pitch else None,
        note.rest,
        note.chord,
        note.grace,
        note.note_type,
        note.dots,
        note.accidental,
        note.ties,
        note.slurs,
        note.articulations,
        note.ornaments,
    )


def _measure_skeleton(measure: MeasureIR) -> tuple[object, ...]:
    return (
        measure.time_signature,
        measure.key_signature,
        measure.clef,
        measure.barlines,
        tuple(direction.stable_tuple() for direction in measure.directions),
        tuple(_note_skeleton(note) for note in measure.notes),
    )


def _duration_total(measure: MeasureIR) -> Fraction:
    return sum((note.duration for note in measure.notes if not note.grace and not note.chord), Fraction(0, 1))


def _duration_error(measure: MeasureIR) -> float:
    expected = measure.expected_duration
    if expected is None:
        return 8.0
    return float(abs(_duration_total(measure) - expected))


def _topology_from_xml(measure: etree._Element, semantics: MeasureIR) -> TupletTopology | None:
    notes = measure.findall("note")
    if len(notes) != len(semantics.notes):
        return None
    states = [read_simple_tuplet_state(note) for note in notes]
    if any(state is None for state in states):
        return None
    topology: list[tuple[int, int]] = []
    index = 0
    while index < len(states):
        state = states[index]
        assert state is not None
        if state.ratio is None:
            if state.start or state.stop:
                return None
            index += 1
            continue
        if not state.start or state.stop or index + 2 >= len(states):
            return None
        group_states = states[index : index + 3]
        group_notes = semantics.notes[index : index + 3]
        if len(group_states) != 3 or len(group_notes) != 3:
            return None
        if any(item is None or item.ratio != (3, 2) for item in group_states):
            return None
        assert all(item is not None for item in group_states)
        if group_states[1].start or group_states[1].stop or group_states[2].start or not group_states[2].stop:
            return None
        first = group_notes[0]
        if any(
            note.voice != first.voice
            or note.duration != first.duration
            or note.note_type != first.note_type
            or note.dots != first.dots
            or note.onset != first.onset + first.duration * offset
            for offset, note in enumerate(group_notes)
        ):
            return None
        topology.append((index, index + 2))
        index += 3
    return tuple(topology)


def _supported_measure(candidate: TupletPatchCandidate) -> bool:
    semantics = candidate.semantics
    notes = semantics.notes
    nodes = candidate.measure.findall("note")
    if (
        semantics.time_signature is None
        or semantics.voice_count != 1
        or len(notes) < 3
        or len(nodes) != len(notes)
        or candidate.measure.find("backup") is not None
        or candidate.measure.find("forward") is not None
        or len(notes) > DEFAULT_POLICY.tuplet_patch_max_events
        or _duration_error(semantics) > 1e-9
    ):
        return False
    for note, node in zip(notes, nodes, strict=True):
        if (
            note.chord
            or note.grace
            or note.duration <= 0
            or note.ties
            or note.slurs
            or node.find("unpitched") is not None
            or (node.find("rest") is not None and node.find("rest").get("measure") == "yes")
            or (note.pitch is None and not note.rest)
        ):
            return False
    topology = _topology_from_xml(candidate.measure, semantics)
    return topology is not None and len(topology) <= DEFAULT_POLICY.tuplet_patch_max_groups


def _representative(items: Sequence[TupletPatchCandidate]) -> TupletPatchCandidate:
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


def _apply_topology(measure: etree._Element, topology: TupletTopology) -> None:
    notes = measure.findall("note")
    by_index: dict[int, tuple[bool, bool]] = {}
    for start, stop in topology:
        for index in range(start, stop + 1):
            by_index[index] = (index == start, index == stop)
    for index, note in enumerate(notes):
        marker = by_index.get(index)
        if marker is None:
            set_simple_tuplet_state(note, ratio=None)
        else:
            set_simple_tuplet_state(note, ratio=(3, 2), start=marker[0], stop=marker[1])


def _pitch_spans(topology: TupletTopology, semantics: MeasureIR) -> tuple[int, int]:
    spans: list[int] = []
    for start, stop in topology:
        cents = [
            note.pitch.midi_cents
            for note in semantics.notes[start : stop + 1]
            if note.pitch is not None and not note.rest
        ]
        spans.append((max(cents) - min(cents)) // 100 if len(cents) >= 2 else 0)
    return (min(spans, default=0), max(spans, default=0))


def propose_tuplet_patch(
    candidates: Sequence[TupletPatchCandidate],
    *,
    template_index: int,
    missing_candidate_count: int,
    calibrator: TupletPatchCalibrator | None = None,
    base_measure: etree._Element | None = None,
) -> TupletPatchResult:
    if not candidates or template_index < 0 or template_index >= len(candidates):
        return TupletPatchResult(None, (), 0, 0.5, 1.0, False, "invalid_input")
    if missing_candidate_count:
        return TupletPatchResult(None, (), 0, 0.5, 1.0, False, "alignment_gap")
    template = candidates[template_index]
    if not template.valid:
        return TupletPatchResult(None, (), 0, 0.5, 1.0, False, "invalid_template")

    family_members, incomplete_families = group_complete_families(
        candidates,
        family_of=lambda item: item.family,
        valid_of=lambda item: item.valid and _supported_measure(item),
    )
    if len(family_members) < DEFAULT_POLICY.tuplet_patch_minimum_families:
        return TupletPatchResult(None, (), 0, 0.5, 1.0, False, "insufficient_families")
    valid = [item for members in family_members.values() for item in members]
    template_skeleton = _measure_skeleton(template.semantics)
    if any(_measure_skeleton(item.semantics) != template_skeleton for item in valid):
        return TupletPatchResult(None, (), 0, 0.5, 1.0, False, "non_tuplet_structure_disagreement")

    family_votes: dict[str, tuple[TupletTopology, TupletPatchCandidate]] = {}
    abstentions = len(incomplete_families)
    for family, members in family_members.items():
        topologies = {_topology_from_xml(item.measure, item.semantics) for item in members}
        if None in topologies or len(topologies) != 1:
            abstentions += 1
            continue
        family_votes[family] = (next(iter(topologies)), _representative(members))  # type: ignore[arg-type]
    if len(family_votes) < DEFAULT_POLICY.tuplet_patch_minimum_families:
        return TupletPatchResult(None, (), 0, 0.5, 1.0, False, "insufficient_topology_family_votes")

    grouped: dict[TupletTopology, list[tuple[str, TupletPatchCandidate]]] = {}
    for family, (topology, representative) in family_votes.items():
        grouped.setdefault(topology, []).append((family, representative))
    ranked = sorted(
        grouped.items(),
        key=lambda item: (len(item[1]), _mean([row.ensemble_probability for _, row in item[1]]), item[0]),
        reverse=True,
    )
    winner, winner_rows = ranked[0]
    runner_up = len(ranked[1][1]) if len(ranked) > 1 else 0
    winner_count = len(winner_rows)
    voting_count = len(family_votes)
    if not (
        winner_count >= DEFAULT_POLICY.tuplet_patch_minimum_supporting_families
        and winner_count > voting_count / 2
        and winner_count - runner_up >= 1
    ):
        return TupletPatchResult(None, (), 0, 0.5, 1.0, False, "no_strict_topology_family_majority")

    source_measure = base_measure if base_measure is not None else template.measure
    inherited = {
        "divisions": template.semantics.divisions,
        "time": template.semantics.time_signature,
        "key": template.semantics.key_signature,
        "clef": template.semantics.clef,
    }
    source_semantics, _ = measure_from_xml(source_measure, inherited)
    source_topology = _topology_from_xml(source_measure, source_semantics)
    if source_topology is None or _measure_skeleton(source_semantics) != template_skeleton:
        return TupletPatchResult(None, (), 0, 0.5, 1.0, False, "base_measure_incompatible")
    if winner == source_topology:
        return TupletPatchResult(None, (), 0, 0.5, 1.0, False, "no_tuplet_change")

    changed = tuple(
        index
        for index in range(len(source_semantics.notes))
        if any(start <= index <= stop for start, stop in source_topology)
        != any(start <= index <= stop for start, stop in winner)
    )
    changed_groups = len(set(source_topology).symmetric_difference(set(winner)))
    if (
        not changed
        or changed_groups > DEFAULT_POLICY.tuplet_patch_max_changed_groups
        or len(changed) > DEFAULT_POLICY.tuplet_patch_max_changed_events
        or len(changed) / max(1, len(source_semantics.notes)) > DEFAULT_POLICY.tuplet_patch_max_changed_ratio
    ):
        return TupletPatchResult(None, changed, changed_groups, 0.5, 1.0, False, "change_scope_too_large")

    patched = copy.deepcopy(source_measure)
    _apply_topology(patched, winner)
    parsed, _ = measure_from_xml(patched, inherited)
    if _measure_skeleton(parsed) != template_skeleton or _topology_from_xml(patched, parsed) != winner:
        return TupletPatchResult(None, changed, changed_groups, 0.5, 1.0, False, "post_patch_structure_changed")
    post = TupletPatchCandidate(
        "patched", "patched", patched, parsed,
        template.page_score, template.page_probability, template.measure_probability,
        template.visual_probability, template.event_probability, template.context_probability,
        template.ensemble_probability, True,
    )
    if not _supported_measure(post):
        return TupletPatchResult(None, changed, changed_groups, 0.5, 1.0, False, "post_patch_invalid_topology")
    template_error = _duration_error(source_semantics)
    patched_error = _duration_error(parsed)
    if patched_error > template_error + 1e-9:
        return TupletPatchResult(None, changed, changed_groups, 0.5, 1.0, False, "meter_error_worsened")

    support = [row for _, row in winner_rows]
    template_support = len(grouped.get(source_topology, ()))
    added = len(set(winner) - set(source_topology))
    removed = len(set(source_topology) - set(winner))
    minimum_span, maximum_span = _pitch_spans(winner, parsed)
    expected = float(parsed.expected_duration or Fraction(4, 1))
    item = TupletPatchInput(
        candidate_count=len(valid),
        eligible_family_count=len(family_members),
        voting_family_count=voting_count,
        changed_event_count=len(changed),
        total_event_count=len(parsed.notes),
        added_group_count=added,
        removed_group_count=removed,
        winner_family_count=winner_count,
        runner_up_family_count=runner_up,
        template_family_count=template_support,
        incomplete_family_count=abstentions,
        winner_group_count=len(winner),
        minimum_group_pitch_span=minimum_span,
        maximum_group_pitch_span=maximum_span,
        expected_measure_duration=expected,
        template_duration_error=template_error,
        patched_duration_error=patched_error,
        mean_support_page_probability=_mean([row.page_probability for row in support]),
        mean_support_measure_probability=_mean([row.measure_probability for row in support]),
        mean_support_visual_probability=_mean([row.visual_probability for row in support]),
        mean_support_event_probability=_mean([row.event_probability for row in support]),
        mean_support_context_probability=_mean([row.context_probability for row in support]),
        mean_support_ensemble_probability=_mean([row.ensemble_probability for row in support]),
        minimum_support_ensemble_probability=min((row.ensemble_probability for row in support), default=0.5),
        mean_support_page_score_margin=_mean([row.page_score - template.page_score for row in support], 0.0),
        mean_support_vs_template_measure_probability=_mean([row.measure_probability - template.measure_probability for row in support], 0.0),
        mean_support_vs_template_visual_probability=_mean([row.visual_probability - template.visual_probability for row in support], 0.0),
        mean_support_vs_template_event_probability=_mean([row.event_probability - template.event_probability for row in support], 0.0),
        mean_support_vs_template_context_probability=_mean([row.context_probability - template.context_probability for row in support], 0.0),
        mean_support_vs_template_ensemble_probability=_mean([row.ensemble_probability - template.ensemble_probability for row in support], 0.0),
    )
    calibrator = calibrator or TupletPatchCalibrator()
    calibration = calibrator.calibrate(item)
    if not calibration.accepted:
        return TupletPatchResult(
            None, changed, changed_groups, calibration.probability, calibration.threshold,
            False, "model_guard", item, calibration.model_version,
        )
    return TupletPatchResult(
        patched, changed, changed_groups, calibration.probability, calibration.threshold,
        True, "accepted", item, calibration.model_version,
    )
