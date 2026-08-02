from __future__ import annotations

"""Conservative within-measure tie-topology consensus.

Missing or spurious MusicXML tie endpoints change playback duration even when pitch,
rhythm and note count are otherwise correct.  This module repairs only ``<tie>`` and
``<notations><tied>`` children on an existing fixed event sequence.  It deliberately
supports only complete ties between consecutive, contiguous notes inside one simple
monophonic measure.  Cross-measure ties, chords, grace notes, tuplets, unpitched notes
and explicit cursor movement remain review-only.

Preprocessing siblings receive one correlation-family vote.  Invalid or split siblings
make the whole family abstain.  Deterministic topology checks are authoritative and the
bundled CPU model may only veto a proposal.
"""

import copy
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from lxml import etree

from .model_registry import load_verified_json
from .policy import DEFAULT_POLICY
from .score_ir import MeasureIR, NoteIR, measure_from_xml
from .tree_model import VerifiedRandomForestModel
from .tie_xml import set_tie_state
from .variant_family import group_complete_families

FEATURE_NAMES = (
    "candidate_count_scaled",
    "eligible_family_count_scaled",
    "voting_family_count_scaled",
    "changed_endpoint_count_scaled",
    "changed_endpoint_ratio",
    "added_endpoint_ratio",
    "removed_endpoint_ratio",
    "changed_pair_count_scaled",
    "winner_family_support_ratio",
    "winner_margin_ratio",
    "template_family_support_ratio",
    "family_abstention_ratio",
    "winner_tie_pair_count_scaled",
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
class TiePatchCandidate:
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
class TiePatchInput:
    candidate_count: int
    eligible_family_count: int
    voting_family_count: int
    changed_endpoint_count: int
    total_event_count: int
    added_endpoint_count: int
    removed_endpoint_count: int
    changed_pair_count: int
    winner_family_count: int
    runner_up_family_count: int
    template_family_count: int
    incomplete_family_count: int
    winner_tie_pair_count: int
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

        event_count = max(1, self.total_event_count)
        family_count = max(1, self.voting_family_count)
        return [
            unit(max(0, self.candidate_count - 1) / 7.0),
            unit(self.eligible_family_count / 4.0),
            unit(self.voting_family_count / 4.0),
            unit(self.changed_endpoint_count / 8.0),
            unit(self.changed_endpoint_count / event_count),
            unit(self.added_endpoint_count / max(1, 2 * event_count)),
            unit(self.removed_endpoint_count / max(1, 2 * event_count)),
            unit(self.changed_pair_count / max(1, DEFAULT_POLICY.tie_patch_max_changed_pairs)),
            unit(self.winner_family_count / family_count),
            unit((self.winner_family_count - self.runner_up_family_count) / family_count),
            unit(self.template_family_count / family_count),
            unit(self.incomplete_family_count / max(1, self.eligible_family_count + self.incomplete_family_count)),
            unit(self.winner_tie_pair_count / max(1, event_count - 1)),
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
class TiePatchCalibration:
    probability: float
    threshold: float
    accepted: bool
    model_version: str
    target_precision: float


@dataclass(frozen=True)
class TiePatchResult:
    patched_measure: etree._Element | None
    changed_event_indices: tuple[int, ...]
    probability: float
    threshold: float
    accepted: bool
    reason: str
    input: TiePatchInput | None = None
    model_version: str = "disabled"


class TiePatchCalibrator:
    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).with_name("resources") / "tie_patch_calibrator.json"
        loaded = load_verified_json(model_path, "tie_patch_calibration")
        payload = loaded.payload
        self.model = VerifiedRandomForestModel.load(
            model_path,
            "tie_patch_calibration",
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
            float(DEFAULT_POLICY.tie_patch_probability_floor),
            max(0.0, min(1.0, stored_threshold)),
        )
        self.target_precision = max(0.0, min(1.0, target_precision))
        self.model_verified = self.model.verified and loaded.verified
        self.model_status = self.model.status if self.model.enabled else loaded.status
        self.model_version = self.model.model_version
        self.enabled = self.model.enabled

    def predict_probability(self, item: TiePatchInput) -> float:
        return self.model.predict(item.feature_vector(), neutral=0.5)

    def calibrate(self, item: TiePatchInput) -> TiePatchCalibration:
        probability = self.predict_probability(item)
        accepted = bool(self.enabled and self.model_verified and probability >= self.threshold)
        return TiePatchCalibration(
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
        note.slurs,
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


def _tie_state(note: NoteIR) -> tuple[str, ...] | None:
    values = tuple(sorted(set(str(value).strip().casefold() for value in note.ties if str(value).strip())))
    if any(value not in {"start", "stop"} for value in values):
        return None
    return values


def _topology(measure: MeasureIR) -> tuple[tuple[str, ...], ...] | None:
    states = tuple(_tie_state(note) for note in measure.notes)
    if any(state is None for state in states):
        return None
    return tuple(state for state in states if state is not None)


def _tie_edges(topology: Sequence[tuple[str, ...]]) -> frozenset[tuple[int, int]]:
    return frozenset((index, index + 1) for index, state in enumerate(topology[:-1]) if "start" in state)


def _valid_internal_topology(measure: MeasureIR, topology: Sequence[tuple[str, ...]]) -> bool:
    notes = measure.notes
    if len(notes) != len(topology) or not notes:
        return False
    for index, (note, state) in enumerate(zip(notes, topology, strict=True)):
        if "stop" in state:
            if index == 0 or "start" not in topology[index - 1]:
                return False
            previous = notes[index - 1]
            if (
                _pitch_key(previous) != _pitch_key(note)
                or previous.voice != note.voice
                or previous.onset + previous.duration != note.onset
            ):
                return False
        if "start" in state:
            if index + 1 >= len(notes) or "stop" not in topology[index + 1]:
                return False
            following = notes[index + 1]
            if (
                _pitch_key(following) != _pitch_key(note)
                or following.voice != note.voice
                or note.onset + note.duration != following.onset
            ):
                return False
    return True


def _supported_measure(candidate: TiePatchCandidate) -> bool:
    semantics = candidate.semantics
    nodes = candidate.measure.findall("note")
    topology = _topology(semantics)
    if (
        semantics.voice_count != 1
        or len(semantics.notes) < 2
        or len(nodes) != len(semantics.notes)
        or candidate.measure.find("backup") is not None
        or candidate.measure.find("forward") is not None
        or topology is None
    ):
        return False
    for note, node in zip(semantics.notes, nodes, strict=True):
        if (
            note.rest
            or note.pitch is None
            or note.chord
            or note.grace
            or note.tuple_ratio is not None
            or note.duration <= 0
            or node.find("unpitched") is not None
            or (node.find("rest") is not None and node.find("rest").get("measure") == "yes")
        ):
            return False
    return _valid_internal_topology(semantics, topology)


def _representative(items: Sequence[TiePatchCandidate]) -> TiePatchCandidate:
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


def propose_tie_patch(
    candidates: Sequence[TiePatchCandidate],
    *,
    template_index: int,
    missing_candidate_count: int,
    calibrator: TiePatchCalibrator | None = None,
    base_measure: etree._Element | None = None,
) -> TiePatchResult:
    """Propose and optionally approve one internal tie-endpoint-only repair."""
    if not candidates or template_index < 0 or template_index >= len(candidates):
        return TiePatchResult(None, (), 0.5, 1.0, False, "invalid_input")
    if missing_candidate_count:
        return TiePatchResult(None, (), 0.5, 1.0, False, "alignment_gap")
    template = candidates[template_index]
    if not template.valid:
        return TiePatchResult(None, (), 0.5, 1.0, False, "invalid_template")

    family_members, incomplete_families = group_complete_families(
        candidates,
        family_of=lambda item: item.family,
        valid_of=lambda item: item.valid and _supported_measure(item),
    )
    if len(family_members) < DEFAULT_POLICY.tie_patch_minimum_families:
        return TiePatchResult(None, (), 0.5, 1.0, False, "insufficient_families")
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
        return TiePatchResult(None, (), 0.5, 1.0, False, "non_tie_structure_disagreement")
    base_topology = _topology(base_semantics)
    if base_topology is None or not _valid_internal_topology(base_semantics, base_topology):
        return TiePatchResult(None, (), 0.5, 1.0, False, "invalid_base_topology")

    family_votes: dict[str, tuple[tuple[tuple[str, ...], ...], TiePatchCandidate]] = {}
    abstentions = len(incomplete_families)
    for family, members in family_members.items():
        topologies = {_topology(item.semantics) for item in members}
        if len(topologies) != 1 or None in topologies:
            abstentions += 1
            continue
        family_votes[family] = (next(iter(topologies)), _representative(members))  # type: ignore[arg-type]
    if len(family_votes) < DEFAULT_POLICY.tie_patch_minimum_families:
        return TiePatchResult(None, (), 0.5, 1.0, False, "insufficient_topology_family_votes")

    grouped: dict[tuple[tuple[str, ...], ...], list[tuple[str, TiePatchCandidate]]] = {}
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
        winner_count >= DEFAULT_POLICY.tie_patch_minimum_supporting_families
        and winner_count > voting_count / 2
        and winner_count - runner_up >= 1
    ):
        return TiePatchResult(None, (), 0.5, 1.0, False, "no_strict_topology_family_majority")
    if not _valid_internal_topology(base_semantics, winner):
        return TiePatchResult(None, (), 0.5, 1.0, False, "winning_topology_invalid")
    if winner == base_topology:
        return TiePatchResult(None, (), 0.5, 1.0, False, "no_tie_change")

    changed = tuple(index for index, (left, right) in enumerate(zip(base_topology, winner, strict=True)) if left != right)
    changed_pairs = len(_tie_edges(base_topology) ^ _tie_edges(winner))
    if (
        not changed
        or len(changed) > DEFAULT_POLICY.tie_patch_max_changed_endpoints
        or len(changed) / max(1, len(winner)) > DEFAULT_POLICY.tie_patch_max_changed_ratio
        or changed_pairs > DEFAULT_POLICY.tie_patch_max_changed_pairs
    ):
        return TiePatchResult(None, changed, 0.5, 1.0, False, "change_scope_too_large")

    patched = copy.deepcopy(source_base)
    patched_notes = patched.findall("note")
    if len(patched_notes) != len(winner):
        return TiePatchResult(None, changed, 0.5, 1.0, False, "xml_event_count_mismatch")
    before_non_tie = []
    for note in patched_notes:
        clone = copy.deepcopy(note)
        set_tie_state(clone, ())
        before_non_tie.append(etree.tostring(clone))
    for index, state in enumerate(winner):
        set_tie_state(patched_notes[index], state)
    after_non_tie = []
    for note in patched.findall("note"):
        clone = copy.deepcopy(note)
        set_tie_state(clone, ())
        after_non_tie.append(etree.tostring(clone))
    if before_non_tie != after_non_tie:
        return TiePatchResult(None, changed, 0.5, 1.0, False, "unrelated_event_xml_changed")

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
        or not _valid_internal_topology(parsed, parsed_topology)
    ):
        return TiePatchResult(None, changed, 0.5, 1.0, False, "post_patch_validation_failed")

    support = [row for _, row in winner_rows]
    added = sum(len(set(winner[index]) - set(base_topology[index])) for index in changed)
    removed = sum(len(set(base_topology[index]) - set(winner[index])) for index in changed)
    template_support = len(grouped.get(base_topology, ()))
    item = TiePatchInput(
        candidate_count=len(valid),
        eligible_family_count=len(family_members),
        voting_family_count=voting_count,
        changed_endpoint_count=len(changed),
        total_event_count=len(winner),
        added_endpoint_count=added,
        removed_endpoint_count=removed,
        changed_pair_count=changed_pairs,
        winner_family_count=winner_count,
        runner_up_family_count=runner_up,
        template_family_count=template_support,
        incomplete_family_count=abstentions,
        winner_tie_pair_count=len(_tie_edges(winner)),
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
    active_calibrator = calibrator or TiePatchCalibrator()
    calibration = active_calibrator.calibrate(item)
    if not calibration.accepted:
        return TiePatchResult(
            None,
            changed,
            calibration.probability,
            calibration.threshold,
            False,
            "model_guard",
            item,
            calibration.model_version,
        )
    return TiePatchResult(
        patched,
        changed,
        calibration.probability,
        calibration.threshold,
        True,
        "accepted",
        item,
        calibration.model_version,
    )
