from __future__ import annotations

"""Conservative within-measure slur-topology consensus.

The repair handles only complete, non-overlapping phrase arcs inside one simple
monophonic measure.  It changes ``<notations><slur>`` endpoints on an existing event
sequence and never changes pitch, rhythm, event count, ties, articulations or ornaments.
Preprocessing siblings collapse to one family vote; invalid or split siblings make the
whole family abstain.  The bundled CPU forest is veto-only.
"""

import copy
import math
import statistics
from fractions import Fraction
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from lxml import etree

from .model_registry import load_verified_json
from .policy import DEFAULT_POLICY
from .score_ir import MeasureIR, NoteIR, measure_from_xml
from .slur_xml import set_slur_topology
from .tree_model import VerifiedRandomForestModel
from .variant_family import group_complete_families

FEATURE_NAMES = (
    "candidate_count_scaled",
    "eligible_family_count_scaled",
    "voting_family_count_scaled",
    "changed_endpoint_count_scaled",
    "changed_endpoint_ratio",
    "added_arc_ratio",
    "removed_arc_ratio",
    "winner_family_support_ratio",
    "winner_margin_ratio",
    "template_family_support_ratio",
    "family_abstention_ratio",
    "winner_arc_count_scaled",
    "minimum_arc_span_scaled",
    "maximum_arc_span_scaled",
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
class SlurPatchCandidate:
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
class SlurPatchInput:
    candidate_count: int
    eligible_family_count: int
    voting_family_count: int
    changed_endpoint_count: int
    total_event_count: int
    added_arc_count: int
    removed_arc_count: int
    winner_family_count: int
    runner_up_family_count: int
    template_family_count: int
    incomplete_family_count: int
    winner_arc_count: int
    minimum_arc_span: int
    maximum_arc_span: int
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
        arc_scope = max(1, self.winner_arc_count + self.removed_arc_count)
        return [
            unit(max(0, self.candidate_count - 1) / 7.0),
            unit(self.eligible_family_count / 4.0),
            unit(self.voting_family_count / 4.0),
            unit(self.changed_endpoint_count / 8.0),
            unit(self.changed_endpoint_count / max(1, 2 * events)),
            unit(self.added_arc_count / arc_scope),
            unit(self.removed_arc_count / arc_scope),
            unit(self.winner_family_count / families),
            unit((self.winner_family_count - self.runner_up_family_count) / families),
            unit(self.template_family_count / families),
            unit(self.incomplete_family_count / max(1, self.eligible_family_count + self.incomplete_family_count)),
            unit(self.winner_arc_count / max(1, DEFAULT_POLICY.slur_patch_max_arcs)),
            unit(self.minimum_arc_span / max(1, DEFAULT_POLICY.slur_patch_max_span_events)),
            unit(self.maximum_arc_span / max(1, DEFAULT_POLICY.slur_patch_max_span_events)),
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
class SlurPatchCalibration:
    probability: float
    threshold: float
    accepted: bool
    model_version: str
    target_precision: float


@dataclass(frozen=True)
class SlurPatchResult:
    patched_measure: etree._Element | None
    changed_event_indices: tuple[int, ...]
    changed_arc_count: int
    probability: float
    threshold: float
    accepted: bool
    reason: str
    input: SlurPatchInput | None = None
    model_version: str = "disabled"


class SlurPatchCalibrator:
    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).with_name("resources") / "slur_patch_calibrator.json"
        loaded = load_verified_json(model_path, "slur_patch_calibration")
        payload = loaded.payload
        self.model = VerifiedRandomForestModel.load(
            model_path,
            "slur_patch_calibration",
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
            float(DEFAULT_POLICY.slur_patch_probability_floor),
            max(0.0, min(1.0, stored_threshold)),
        )
        self.target_precision = max(0.0, min(1.0, target_precision))
        self.model_verified = self.model.verified and loaded.verified
        self.model_status = self.model.status if self.model.enabled else loaded.status
        self.model_version = self.model.model_version
        self.enabled = self.model.enabled

    def predict_probability(self, item: SlurPatchInput) -> float:
        return self.model.predict(item.feature_vector(), neutral=0.5)

    def calibrate(self, item: SlurPatchInput) -> SlurPatchCalibration:
        probability = self.predict_probability(item)
        accepted = bool(self.enabled and self.model_verified and probability >= self.threshold)
        return SlurPatchCalibration(
            probability=round(probability, 6),
            threshold=round(self.threshold, 6),
            accepted=accepted,
            model_version=self.model_version,
            target_precision=round(self.target_precision, 6),
        )


def _mean(values: Sequence[float], default: float = 0.5) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else default


def _pitch_key(note: NoteIR) -> tuple[object, ...] | None:
    if note.rest or note.pitch is None:
        return None
    return note.pitch.stable_tuple()


def _event_skeleton(note: NoteIR) -> tuple[object, ...]:
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
        note.articulations,
        note.ornaments,
        note.tuple_ratio,
        _pitch_key(note),
        note.accidental,
    )


def _measure_skeleton(measure: MeasureIR) -> tuple[object, ...]:
    return (
        measure.divisions,
        measure.time_signature,
        measure.key_signature,
        measure.clef,
        tuple(direction.stable_tuple() for direction in measure.directions),
        measure.barlines,
        tuple(_event_skeleton(note) for note in measure.notes),
    )


def _topology(measure: MeasureIR) -> tuple[tuple[int, int], ...] | None:
    endpoints: dict[str, dict[str, list[int]]] = {}
    for index, note in enumerate(measure.notes):
        for kind, number in note.slurs:
            normalized_kind = str(kind).strip().casefold()
            normalized_number = str(number).strip() or "1"
            if normalized_kind not in {"start", "stop"}:
                return None
            bucket = endpoints.setdefault(normalized_number, {"start": [], "stop": []})
            bucket[normalized_kind].append(index)
    arcs: list[tuple[int, int]] = []
    for bucket in endpoints.values():
        if len(bucket["start"]) != 1 or len(bucket["stop"]) != 1:
            return None
        start, stop = bucket["start"][0], bucket["stop"][0]
        if start >= stop:
            return None
        arcs.append((start, stop))
    arcs.sort()
    if len(set(arcs)) != len(arcs):
        return None
    return tuple(arcs)


def _valid_topology(measure: MeasureIR, topology: Sequence[tuple[int, int]]) -> bool:
    if len(topology) > DEFAULT_POLICY.slur_patch_max_arcs:
        return False
    previous_stop = -1
    for start, stop in topology:
        if start < 0 or stop >= len(measure.notes) or start >= stop:
            return False
        span = stop - start
        if span > DEFAULT_POLICY.slur_patch_max_span_events:
            return False
        if stop == start + 1:
            first = measure.notes[start]
            second = measure.notes[stop]
            if (
                _pitch_key(first) is not None
                and _pitch_key(first) == _pitch_key(second)
                and first.voice == second.voice
                and second.onset == first.onset + first.duration
            ):
                # A short same-pitch slur is visually indistinguishable from a tie in
                # the bounded source corridor. Automatic slur repair therefore abstains
                # instead of competing with the dedicated tie transaction.
                return False
        # Reject overlapping and nested arcs.  They are valid MusicXML, but automatic
        # repair cannot safely infer their numbering/engraving from weak OMR evidence.
        if start <= previous_stop:
            return False
        previous_stop = stop
    return True


def _duration_total(measure: MeasureIR) -> Fraction:
    return sum(
        (note.duration for note in measure.notes if not note.grace and not note.chord),
        Fraction(0, 1),
    )


def _supported_measure(candidate: SlurPatchCandidate) -> bool:
    semantics = candidate.semantics
    nodes = candidate.measure.findall("note")
    topology = _topology(semantics)
    if (
        semantics.voice_count != 1
        or semantics.expected_duration is None
        or _duration_total(semantics) != semantics.expected_duration
        or len(semantics.notes) < 2
        or len(semantics.notes) > DEFAULT_POLICY.slur_patch_max_events
        or len(nodes) != len(semantics.notes)
        or candidate.measure.find("backup") is not None
        or candidate.measure.find("forward") is not None
        or topology is None
        or not _valid_topology(semantics, topology)
    ):
        return False
    for note, node in zip(semantics.notes, nodes, strict=True):
        if (
            note.rest
            or note.pitch is None
            or note.chord
            or note.grace
            or note.tuple_ratio is not None
            or note.ties
            or note.duration <= 0
            or node.find("unpitched") is not None
            or (node.find("rest") is not None and node.find("rest").get("measure") == "yes")
        ):
            return False
    return True


def _representative(items: Sequence[SlurPatchCandidate]) -> SlurPatchCandidate:
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


def _endpoint_set(topology: Sequence[tuple[int, int]]) -> frozenset[tuple[int, str]]:
    return frozenset(
        endpoint
        for start, stop in topology
        for endpoint in ((start, "start"), (stop, "stop"))
    )


def propose_slur_patch(
    candidates: Sequence[SlurPatchCandidate],
    *,
    template_index: int,
    missing_candidate_count: int,
    calibrator: SlurPatchCalibrator | None = None,
    base_measure: etree._Element | None = None,
) -> SlurPatchResult:
    """Propose and optionally approve one within-measure slur-only repair."""

    if not candidates or template_index < 0 or template_index >= len(candidates):
        return SlurPatchResult(None, (), 0, 0.5, 1.0, False, "invalid_input")
    if missing_candidate_count:
        return SlurPatchResult(None, (), 0, 0.5, 1.0, False, "alignment_gap")
    template = candidates[template_index]
    if not template.valid:
        return SlurPatchResult(None, (), 0, 0.5, 1.0, False, "invalid_template")

    family_members, incomplete_families = group_complete_families(
        candidates,
        family_of=lambda item: item.family,
        valid_of=lambda item: item.valid and _supported_measure(item),
    )
    if len(family_members) < DEFAULT_POLICY.slur_patch_minimum_families:
        return SlurPatchResult(None, (), 0, 0.5, 1.0, False, "insufficient_families")
    valid = [item for members in family_members.values() for item in members]

    base_semantics = template.semantics
    source_base = template.measure
    if base_measure is not None:
        inherited: dict[str, object] = {
            "divisions": template.semantics.divisions,
            "time": template.semantics.time_signature,
            "key": template.semantics.key_signature,
            "clef": template.semantics.clef,
        }
        base_semantics, _ = measure_from_xml(base_measure, inherited)
        source_base = base_measure
    skeleton = _measure_skeleton(base_semantics)
    if any(_measure_skeleton(item.semantics) != skeleton for item in valid):
        return SlurPatchResult(None, (), 0, 0.5, 1.0, False, "non_slur_structure_disagreement")
    base_topology = _topology(base_semantics)
    if base_topology is None or not _valid_topology(base_semantics, base_topology):
        return SlurPatchResult(None, (), 0, 0.5, 1.0, False, "invalid_base_topology")

    family_votes: dict[str, tuple[tuple[tuple[int, int], ...], SlurPatchCandidate]] = {}
    abstentions = len(incomplete_families)
    for family, members in family_members.items():
        topologies = {_topology(item.semantics) for item in members}
        if len(topologies) != 1 or None in topologies:
            abstentions += 1
            continue
        family_votes[family] = (next(iter(topologies)), _representative(members))  # type: ignore[arg-type]
    if len(family_votes) < DEFAULT_POLICY.slur_patch_minimum_families:
        return SlurPatchResult(None, (), 0, 0.5, 1.0, False, "insufficient_topology_family_votes")

    grouped: dict[tuple[tuple[int, int], ...], list[tuple[str, SlurPatchCandidate]]] = {}
    for family, (topology, representative) in family_votes.items():
        grouped.setdefault(topology, []).append((family, representative))
    ranked = sorted(
        grouped.items(),
        key=lambda item: (
            len(item[1]),
            _mean([row.ensemble_probability for _, row in item[1]]),
            repr(item[0]),
        ),
        reverse=True,
    )
    winner, winner_rows = ranked[0]
    runner_up = len(ranked[1][1]) if len(ranked) > 1 else 0
    winner_count = len(winner_rows)
    voting_count = len(family_votes)
    if not (
        winner_count >= DEFAULT_POLICY.slur_patch_minimum_supporting_families
        and winner_count > voting_count / 2
        and winner_count - runner_up >= 1
    ):
        return SlurPatchResult(None, (), 0, 0.5, 1.0, False, "no_strict_topology_family_majority")
    if not _valid_topology(base_semantics, winner):
        return SlurPatchResult(None, (), 0, 0.5, 1.0, False, "winning_topology_invalid")
    if winner == base_topology:
        return SlurPatchResult(None, (), 0, 0.5, 1.0, False, "no_slur_change")

    base_endpoints = _endpoint_set(base_topology)
    winner_endpoints = _endpoint_set(winner)
    changed_endpoints = base_endpoints ^ winner_endpoints
    changed_indices = tuple(sorted({index for index, _kind in changed_endpoints}))
    changed_arcs = len(set(base_topology) ^ set(winner))
    if (
        not changed_indices
        or len(changed_endpoints) > DEFAULT_POLICY.slur_patch_max_changed_endpoints
        or len(changed_endpoints) / max(1, 2 * len(base_semantics.notes)) > DEFAULT_POLICY.slur_patch_max_changed_ratio
        or changed_arcs > DEFAULT_POLICY.slur_patch_max_changed_arcs
    ):
        return SlurPatchResult(None, changed_indices, changed_arcs, 0.5, 1.0, False, "change_scope_too_large")

    patched = copy.deepcopy(source_base)
    patched_notes = patched.findall("note")
    if len(patched_notes) != len(base_semantics.notes):
        return SlurPatchResult(None, changed_indices, changed_arcs, 0.5, 1.0, False, "xml_event_count_mismatch")
    before_non_slur: list[bytes] = []
    for note in patched_notes:
        clone = copy.deepcopy(note)
        set_slur_topology([clone], ())
        before_non_slur.append(etree.tostring(clone))
    set_slur_topology(patched_notes, winner)
    after_non_slur: list[bytes] = []
    for note in patched.findall("note"):
        clone = copy.deepcopy(note)
        set_slur_topology([clone], ())
        after_non_slur.append(etree.tostring(clone))
    if before_non_slur != after_non_slur:
        return SlurPatchResult(None, changed_indices, changed_arcs, 0.5, 1.0, False, "unrelated_event_xml_changed")

    inherited = {
        "divisions": base_semantics.divisions,
        "time": base_semantics.time_signature,
        "key": base_semantics.key_signature,
        "clef": base_semantics.clef,
    }
    parsed, _ = measure_from_xml(patched, inherited)
    parsed_topology = _topology(parsed)
    if (
        _measure_skeleton(parsed) != skeleton
        or parsed_topology != winner
        or parsed_topology is None
        or not _valid_topology(parsed, parsed_topology)
    ):
        return SlurPatchResult(None, changed_indices, changed_arcs, 0.5, 1.0, False, "post_patch_validation_failed")

    support = [row for _, row in winner_rows]
    added_arcs = len(set(winner) - set(base_topology))
    removed_arcs = len(set(base_topology) - set(winner))
    spans = [stop - start for start, stop in winner]
    template_support = len(grouped.get(base_topology, ()))
    item = SlurPatchInput(
        candidate_count=len(valid),
        eligible_family_count=len(family_members),
        voting_family_count=voting_count,
        changed_endpoint_count=len(changed_endpoints),
        total_event_count=len(base_semantics.notes),
        added_arc_count=added_arcs,
        removed_arc_count=removed_arcs,
        winner_family_count=winner_count,
        runner_up_family_count=runner_up,
        template_family_count=template_support,
        incomplete_family_count=abstentions,
        winner_arc_count=len(winner),
        minimum_arc_span=min(spans, default=0),
        maximum_arc_span=max(spans, default=0),
        mean_support_page_probability=_mean([row.page_probability for row in support]),
        mean_support_measure_probability=_mean([row.measure_probability for row in support]),
        mean_support_visual_probability=_mean([row.visual_probability for row in support]),
        mean_support_event_probability=_mean([row.event_probability for row in support]),
        mean_support_context_probability=_mean([row.context_probability for row in support]),
        mean_support_ensemble_probability=_mean([row.ensemble_probability for row in support]),
        minimum_support_ensemble_probability=min(row.ensemble_probability for row in support),
        mean_support_page_score_margin=_mean([row.page_score - template.page_score for row in support], 0.0),
        mean_support_vs_template_measure_probability=_mean([
            row.measure_probability - template.measure_probability for row in support
        ], 0.0),
        mean_support_vs_template_visual_probability=_mean([
            row.visual_probability - template.visual_probability for row in support
        ], 0.0),
        mean_support_vs_template_event_probability=_mean([
            row.event_probability - template.event_probability for row in support
        ], 0.0),
        mean_support_vs_template_context_probability=_mean([
            row.context_probability - template.context_probability for row in support
        ], 0.0),
        mean_support_vs_template_ensemble_probability=_mean([
            row.ensemble_probability - template.ensemble_probability for row in support
        ], 0.0),
    )
    active_calibrator = calibrator or SlurPatchCalibrator()
    calibration = active_calibrator.calibrate(item)
    if not calibration.accepted:
        return SlurPatchResult(
            None,
            changed_indices,
            changed_arcs,
            calibration.probability,
            calibration.threshold,
            False,
            "model_guard",
            item,
            calibration.model_version,
        )
    return SlurPatchResult(
        patched,
        changed_indices,
        changed_arcs,
        calibration.probability,
        calibration.threshold,
        True,
        "accepted",
        item,
        calibration.model_version,
    )
