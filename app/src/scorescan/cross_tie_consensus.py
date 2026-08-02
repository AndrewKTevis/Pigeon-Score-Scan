from __future__ import annotations

"""Conservative cross-measure tie consensus for one adjacent measure boundary.

The patch is intentionally narrower than general tie recognition.  It may only toggle
``start`` on the final event of the left measure and ``stop`` on the first event of the
right measure.  Both events must be contiguous, pitched, monophonic and semantically
identical across supporting independent preprocessing families.  Candidate models may
veto a deterministic proposal but never create evidence.
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
from .tie_xml import normalized_tie_state, set_endpoint, without_ties
from .tree_model import VerifiedRandomForestModel
from .variant_family import group_complete_families

FEATURE_NAMES = (
    "candidate_count_scaled",
    "eligible_family_count_scaled",
    "voting_family_count_scaled",
    "winner_family_support_ratio",
    "winner_margin_ratio",
    "template_family_support_ratio",
    "family_abstention_ratio",
    "winner_boundary_present",
    "changed_endpoint_ratio",
    "added_endpoint_ratio",
    "removed_endpoint_ratio",
    "left_boundary_duration_scaled",
    "right_boundary_duration_scaled",
    "mean_support_page_probability",
    "mean_support_measure_probability",
    "mean_support_visual_probability",
    "mean_support_event_probability",
    "mean_support_context_probability",
    "mean_support_ensemble_probability",
    "minimum_support_ensemble_probability",
    "mean_support_alignment_similarity",
    "mean_support_page_score_margin_scaled",
    "mean_support_vs_template_measure_probability",
    "mean_support_vs_template_visual_probability",
    "mean_support_vs_template_event_probability",
    "mean_support_vs_template_context_probability",
    "mean_support_vs_template_ensemble_probability",
)


@dataclass(frozen=True)
class CrossTiePatchCandidate:
    variant: str
    family: str
    left_measure: etree._Element
    right_measure: etree._Element
    left_semantics: MeasureIR
    right_semantics: MeasureIR
    page_score: float
    page_probability: float
    measure_probability: float
    visual_probability: float
    event_probability: float
    context_probability: float
    ensemble_probability: float
    alignment_similarity: float
    valid: bool


@dataclass(frozen=True)
class CrossTiePatchInput:
    candidate_count: int
    eligible_family_count: int
    voting_family_count: int
    winner_family_count: int
    runner_up_family_count: int
    template_family_count: int
    incomplete_family_count: int
    winner_boundary_present: bool
    changed_endpoint_count: int
    added_endpoint_count: int
    removed_endpoint_count: int
    left_boundary_duration: Fraction
    right_boundary_duration: Fraction
    mean_support_page_probability: float
    mean_support_measure_probability: float
    mean_support_visual_probability: float
    mean_support_event_probability: float
    mean_support_context_probability: float
    mean_support_ensemble_probability: float
    minimum_support_ensemble_probability: float
    mean_support_alignment_similarity: float
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

        voting = max(1, self.voting_family_count)
        eligible = max(1, self.eligible_family_count + self.incomplete_family_count)
        return [
            unit(max(0, self.candidate_count - 1) / 7.0),
            unit(self.eligible_family_count / 4.0),
            unit(self.voting_family_count / 4.0),
            unit(self.winner_family_count / voting),
            unit((self.winner_family_count - self.runner_up_family_count) / voting),
            unit(self.template_family_count / voting),
            unit(self.incomplete_family_count / eligible),
            1.0 if self.winner_boundary_present else 0.0,
            unit(self.changed_endpoint_count / 2.0),
            unit(self.added_endpoint_count / 2.0),
            unit(self.removed_endpoint_count / 2.0),
            unit(float(self.left_boundary_duration) / 4.0),
            unit(float(self.right_boundary_duration) / 4.0),
            unit(self.mean_support_page_probability),
            unit(self.mean_support_measure_probability),
            unit(self.mean_support_visual_probability),
            unit(self.mean_support_event_probability),
            unit(self.mean_support_context_probability),
            unit(self.mean_support_ensemble_probability),
            unit(self.minimum_support_ensemble_probability),
            unit(self.mean_support_alignment_similarity),
            signed(self.mean_support_page_score_margin / 100.0),
            signed(self.mean_support_vs_template_measure_probability),
            signed(self.mean_support_vs_template_visual_probability),
            signed(self.mean_support_vs_template_event_probability),
            signed(self.mean_support_vs_template_context_probability),
            signed(self.mean_support_vs_template_ensemble_probability),
        ]


@dataclass(frozen=True)
class CrossTiePatchCalibration:
    probability: float
    threshold: float
    accepted: bool
    model_version: str
    target_precision: float


@dataclass(frozen=True)
class CrossTiePatchResult:
    left_measure: etree._Element | None
    right_measure: etree._Element | None
    changed_endpoint_count: int
    probability: float
    threshold: float
    accepted: bool
    reason: str
    input: CrossTiePatchInput | None = None
    model_version: str = "disabled"


class CrossTiePatchCalibrator:
    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).with_name("resources") / "cross_tie_patch_calibrator.json"
        loaded = load_verified_json(model_path, "cross_tie_patch_calibration")
        payload = loaded.payload
        self.model = VerifiedRandomForestModel.load(
            model_path,
            "cross_tie_patch_calibration",
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
            float(DEFAULT_POLICY.cross_tie_patch_probability_floor),
            max(0.0, min(1.0, stored_threshold)),
        )
        self.target_precision = max(0.0, min(1.0, target_precision))
        self.model_verified = self.model.verified and loaded.verified
        self.model_status = self.model.status if self.model.enabled else loaded.status
        self.model_version = self.model.model_version
        self.enabled = self.model.enabled

    def predict_probability(self, item: CrossTiePatchInput) -> float:
        return self.model.predict(item.feature_vector(), neutral=0.5)

    def calibrate(self, item: CrossTiePatchInput) -> CrossTiePatchCalibration:
        probability = self.predict_probability(item)
        accepted = bool(self.enabled and self.model_verified and probability >= self.threshold)
        return CrossTiePatchCalibration(
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
        _pitch_key(note),
        note.rest,
        note.chord,
        note.grace,
        note.note_type,
        note.dots,
        note.accidental,
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
        tuple(item.stable_tuple() for item in measure.directions),
        measure.barlines,
        tuple(_event_skeleton(note) for note in measure.notes),
    )


def _duration_total(measure: MeasureIR) -> Fraction:
    return sum(
        (note.duration for note in measure.notes if not note.chord and not note.grace),
        Fraction(0, 1),
    )


def _boundary_notes(left: MeasureIR, right: MeasureIR) -> tuple[NoteIR, NoteIR] | None:
    if not left.notes or not right.notes:
        return None
    # Cross-measure repair intentionally excludes notation whose event timing or
    # grouping depends on context beyond a simple monophonic stream.  Checking
    # the complete measures (not only the two boundary notes) prevents a hidden
    # chord/grace/tuplet elsewhere in either measure from making a seemingly
    # local endpoint toggle unsafe.
    if any(
        note.chord or note.grace or note.tuple_ratio is not None
        for note in (*left.notes, *right.notes)
    ):
        return None
    left_note = left.notes[-1]
    right_note = right.notes[0]
    if (
        left.voice_count != 1
        or right.voice_count != 1
        or left.expected_duration is None
        or right.expected_duration is None
        or _duration_total(left) != left.expected_duration
        or left_note.onset + left_note.duration != left.expected_duration
        or right_note.onset != 0
        or left_note.voice != right_note.voice
        or _pitch_key(left_note) is None
        or _pitch_key(left_note) != _pitch_key(right_note)
    ):
        return None
    for note in (left_note, right_note):
        if (
            note.rest
            or note.pitch is None
            or note.chord
            or note.grace
            or note.tuple_ratio is not None
            or note.duration <= 0
        ):
            return None
    return left_note, right_note


def _boundary_state(left: MeasureIR, right: MeasureIR) -> bool | None:
    notes = _boundary_notes(left, right)
    if notes is None:
        return None
    left_state = set(notes[0].ties)
    right_state = set(notes[1].ties)
    if any(value not in {"start", "stop"} for value in left_state | right_state):
        return None
    start = "start" in left_state
    stop = "stop" in right_state
    if start != stop:
        return None
    return start


def _supported_candidate(candidate: CrossTiePatchCandidate) -> bool:
    return bool(
        candidate.valid
        and candidate.left_measure.find("backup") is None
        and candidate.left_measure.find("forward") is None
        and candidate.right_measure.find("backup") is None
        and candidate.right_measure.find("forward") is None
        and len(candidate.left_measure.findall("note")) == len(candidate.left_semantics.notes)
        and len(candidate.right_measure.findall("note")) == len(candidate.right_semantics.notes)
        and _boundary_state(candidate.left_semantics, candidate.right_semantics) is not None
    )


def _representative(items: Sequence[CrossTiePatchCandidate]) -> CrossTiePatchCandidate:
    return max(
        items,
        key=lambda item: (
            item.ensemble_probability,
            item.event_probability,
            item.context_probability,
            item.measure_probability,
            item.visual_probability,
            item.alignment_similarity,
            item.page_score,
            item.variant,
        ),
    )


def _parse_pair(
    left: etree._Element,
    right: etree._Element,
    left_reference: MeasureIR,
) -> tuple[MeasureIR, MeasureIR]:
    inherited: dict[str, object] = {
        "divisions": left_reference.divisions,
        "time": left_reference.time_signature,
        "key": left_reference.key_signature,
        "clef": left_reference.clef,
    }
    parsed_left, state = measure_from_xml(left, inherited)
    parsed_right, _ = measure_from_xml(right, state)
    return parsed_left, parsed_right


def propose_cross_tie_patch(
    candidates: Sequence[CrossTiePatchCandidate],
    *,
    template_variant: str,
    base_left: etree._Element,
    base_right: etree._Element,
    base_left_semantics: MeasureIR,
    base_right_semantics: MeasureIR,
    calibrator: CrossTiePatchCalibrator | None = None,
) -> CrossTiePatchResult:
    if not candidates:
        return CrossTiePatchResult(None, None, 0, 0.5, 1.0, False, "invalid_input")
    template = next((item for item in candidates if item.variant == template_variant), None)
    if template is None or not template.valid:
        return CrossTiePatchResult(None, None, 0, 0.5, 1.0, False, "invalid_template")
    base_state = _boundary_state(base_left_semantics, base_right_semantics)
    base_notes = _boundary_notes(base_left_semantics, base_right_semantics)
    if base_notes is None:
        return CrossTiePatchResult(None, None, 0, 0.5, 1.0, False, "unsupported_base_boundary")

    family_members, incomplete_families = group_complete_families(
        candidates,
        family_of=lambda item: item.family,
        valid_of=lambda item: item.valid,
    )
    if len(family_members) < DEFAULT_POLICY.cross_tie_patch_minimum_families:
        return CrossTiePatchResult(None, None, 0, 0.5, 1.0, False, "insufficient_families")

    base_skeleton = (_measure_skeleton(base_left_semantics), _measure_skeleton(base_right_semantics))
    votes: dict[str, tuple[bool, CrossTiePatchCandidate]] = {}
    abstentions = len(incomplete_families)
    valid_candidate_count = 0
    for family, members in family_members.items():
        if any(
            not _supported_candidate(item)
            or (_measure_skeleton(item.left_semantics), _measure_skeleton(item.right_semantics)) != base_skeleton
            for item in members
        ):
            abstentions += 1
            continue
        states = {_boundary_state(item.left_semantics, item.right_semantics) for item in members}
        if len(states) != 1 or None in states:
            abstentions += 1
            continue
        valid_candidate_count += len(members)
        votes[family] = (bool(next(iter(states))), _representative(members))
    if len(votes) < DEFAULT_POLICY.cross_tie_patch_minimum_families:
        return CrossTiePatchResult(None, None, 0, 0.5, 1.0, False, "insufficient_boundary_family_votes")

    grouped: dict[bool, list[tuple[str, CrossTiePatchCandidate]]] = {}
    for family, (state, representative) in votes.items():
        grouped.setdefault(state, []).append((family, representative))
    ranked = sorted(
        grouped.items(),
        key=lambda item: (
            len(item[1]),
            _mean([row.ensemble_probability for _, row in item[1]]),
            int(item[0]),
        ),
        reverse=True,
    )
    winner, winner_rows = ranked[0]
    runner_up = len(ranked[1][1]) if len(ranked) > 1 else 0
    winner_count = len(winner_rows)
    voting_count = len(votes)
    if not (
        winner_count >= DEFAULT_POLICY.cross_tie_patch_minimum_supporting_families
        and winner_count > voting_count / 2
        and winner_count - runner_up >= 1
    ):
        return CrossTiePatchResult(None, None, 0, 0.5, 1.0, False, "no_strict_boundary_family_majority")
    if base_state is not None and winner == base_state:
        return CrossTiePatchResult(None, None, 0, 0.5, 1.0, False, "no_boundary_tie_change")

    left_xml = copy.deepcopy(base_left)
    right_xml = copy.deepcopy(base_right)
    left_notes = left_xml.findall("note")
    right_notes = right_xml.findall("note")
    if not left_notes or not right_notes:
        return CrossTiePatchResult(None, None, 0, 0.5, 1.0, False, "xml_boundary_event_missing")
    left_boundary = left_notes[-1]
    right_boundary = right_notes[0]
    old_left_state = normalized_tie_state(left_boundary)
    old_right_state = normalized_tie_state(right_boundary)
    if old_left_state is None or old_right_state is None:
        return CrossTiePatchResult(None, None, 0, 0.5, 1.0, False, "invalid_xml_tie_state")
    before_left = without_ties(left_boundary)
    before_right = without_ties(right_boundary)
    if not set_endpoint(left_boundary, "start", winner) or not set_endpoint(right_boundary, "stop", winner):
        return CrossTiePatchResult(None, None, 0, 0.5, 1.0, False, "xml_tie_write_failed")
    if before_left != without_ties(left_boundary) or before_right != without_ties(right_boundary):
        return CrossTiePatchResult(None, None, 0, 0.5, 1.0, False, "unrelated_event_xml_changed")

    parsed_left, parsed_right = _parse_pair(left_xml, right_xml, base_left_semantics)
    if (
        (_measure_skeleton(parsed_left), _measure_skeleton(parsed_right)) != base_skeleton
        or _boundary_state(parsed_left, parsed_right) != winner
    ):
        return CrossTiePatchResult(None, None, 0, 0.5, 1.0, False, "post_patch_validation_failed")

    old_start = "start" in old_left_state
    old_stop = "stop" in old_right_state
    changed = int(old_start != winner) + int(old_stop != winner)
    added = int(winner and not old_start) + int(winner and not old_stop)
    removed = int(not winner and old_start) + int(not winner and old_stop)
    support = [row for _, row in winner_rows]
    template_state = _boundary_state(template.left_semantics, template.right_semantics)
    template_support = len(grouped.get(bool(template_state), ())) if template_state is not None else 0
    item = CrossTiePatchInput(
        candidate_count=valid_candidate_count,
        eligible_family_count=len(family_members),
        voting_family_count=voting_count,
        winner_family_count=winner_count,
        runner_up_family_count=runner_up,
        template_family_count=template_support,
        incomplete_family_count=abstentions,
        winner_boundary_present=winner,
        changed_endpoint_count=changed,
        added_endpoint_count=added,
        removed_endpoint_count=removed,
        left_boundary_duration=base_notes[0].duration,
        right_boundary_duration=base_notes[1].duration,
        mean_support_page_probability=_mean([row.page_probability for row in support]),
        mean_support_measure_probability=_mean([row.measure_probability for row in support]),
        mean_support_visual_probability=_mean([row.visual_probability for row in support]),
        mean_support_event_probability=_mean([row.event_probability for row in support]),
        mean_support_context_probability=_mean([row.context_probability for row in support]),
        mean_support_ensemble_probability=_mean([row.ensemble_probability for row in support]),
        minimum_support_ensemble_probability=min(row.ensemble_probability for row in support),
        mean_support_alignment_similarity=_mean([row.alignment_similarity for row in support]),
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
    active = calibrator or CrossTiePatchCalibrator()
    calibration = active.calibrate(item)
    if not calibration.accepted:
        return CrossTiePatchResult(
            None,
            None,
            changed,
            calibration.probability,
            calibration.threshold,
            False,
            "model_guard",
            item,
            calibration.model_version,
        )
    return CrossTiePatchResult(
        left_xml,
        right_xml,
        changed,
        calibration.probability,
        calibration.threshold,
        True,
        "accepted",
        item,
        calibration.model_version,
    )
