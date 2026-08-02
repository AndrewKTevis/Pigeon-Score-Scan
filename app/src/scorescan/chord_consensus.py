from __future__ import annotations

"""Conservative chord-topology consensus for complementary OMR errors.

A missing or spurious ``<chord/>`` marker changes MusicXML cursor movement and can
shift every later onset in a measure even when all noteheads and durations were
recognized correctly.  This module repairs only chord membership on an existing,
fixed event sequence.  It never inserts, deletes, reorders, repitches, or retimes a
note.  Correlated preprocessing siblings receive one family vote, split or invalid
siblings abstain, deterministic meter/topology checks remain authoritative, and the
bundled CPU model may only veto a proposal.
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
from .variant_family import group_complete_families

FEATURE_NAMES = (
    "candidate_count_scaled",
    "eligible_family_count_scaled",
    "voting_family_count_scaled",
    "changed_marker_count_scaled",
    "changed_marker_ratio",
    "added_marker_ratio",
    "removed_marker_ratio",
    "winner_family_support_ratio",
    "winner_margin_ratio",
    "template_family_support_ratio",
    "family_abstention_ratio",
    "winner_chord_group_count_scaled",
    "winner_max_chord_size_scaled",
    "template_duration_error_ratio",
    "patched_duration_error_ratio",
    "duration_error_improvement_ratio",
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
class ChordPatchCandidate:
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
class ChordPatchInput:
    candidate_count: int
    eligible_family_count: int
    voting_family_count: int
    changed_marker_count: int
    total_event_count: int
    added_marker_count: int
    removed_marker_count: int
    winner_family_count: int
    runner_up_family_count: int
    template_family_count: int
    incomplete_family_count: int
    winner_chord_group_count: int
    winner_max_chord_size: int
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

        event_count = max(1, self.total_event_count)
        family_count = max(1, self.voting_family_count)
        expected = max(1e-9, self.expected_measure_duration)
        return [
            unit(max(0, self.candidate_count - 1) / 7.0),
            unit(self.eligible_family_count / 4.0),
            unit(self.voting_family_count / 4.0),
            unit(self.changed_marker_count / 8.0),
            unit(self.changed_marker_count / event_count),
            unit(self.added_marker_count / event_count),
            unit(self.removed_marker_count / event_count),
            unit(self.winner_family_count / family_count),
            unit((self.winner_family_count - self.runner_up_family_count) / family_count),
            unit(self.template_family_count / family_count),
            unit(self.incomplete_family_count / max(1, self.eligible_family_count + self.incomplete_family_count)),
            unit(self.winner_chord_group_count / 8.0),
            unit(self.winner_max_chord_size / max(1, DEFAULT_POLICY.chord_patch_max_chord_size)),
            unit(self.template_duration_error / expected),
            unit(self.patched_duration_error / expected),
            signed((self.template_duration_error - self.patched_duration_error) / expected),
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
class ChordPatchCalibration:
    probability: float
    threshold: float
    accepted: bool
    model_version: str
    target_precision: float


@dataclass(frozen=True)
class ChordPatchResult:
    patched_measure: etree._Element | None
    changed_event_indices: tuple[int, ...]
    probability: float
    threshold: float
    accepted: bool
    reason: str
    input: ChordPatchInput | None = None
    model_version: str = "disabled"


class ChordPatchCalibrator:
    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).with_name("resources") / "chord_patch_calibrator.json"
        loaded = load_verified_json(model_path, "chord_patch_calibration")
        payload = loaded.payload
        self.model = VerifiedRandomForestModel.load(
            model_path,
            "chord_patch_calibration",
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
            float(DEFAULT_POLICY.chord_patch_probability_floor),
            max(0.0, min(1.0, stored_threshold)),
        )
        self.target_precision = max(0.0, min(1.0, target_precision))
        self.model_verified = self.model.verified and loaded.verified
        self.model_status = self.model.status if self.model.enabled else loaded.status
        self.model_version = self.model.model_version
        self.enabled = self.model.enabled

    def predict_probability(self, item: ChordPatchInput) -> float:
        return self.model.predict(item.feature_vector(), neutral=0.5)

    def calibrate(self, item: ChordPatchInput) -> ChordPatchCalibration:
        probability = self.predict_probability(item)
        accepted = bool(self.enabled and self.model_verified and probability >= self.threshold)
        return ChordPatchCalibration(
            probability=round(probability, 6),
            threshold=round(self.threshold, 6),
            accepted=accepted,
            model_version=self.model_version,
            target_precision=round(self.target_precision, 6),
        )


def _mean(values: Sequence[float], default: float = 0.5) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else default


def _event_skeleton(note: NoteIR) -> tuple[object, ...]:
    return (
        note.duration,
        note.voice,
        note.rest,
        note.grace,
        note.note_type,
        note.dots,
        note.ties,
        note.slurs,
        note.articulations,
        note.ornaments,
        note.tuple_ratio,
        note.pitch is not None,
    )


def _measure_skeleton(measure: MeasureIR) -> tuple[object, ...]:
    return (
        measure.time_signature,
        measure.key_signature,
        measure.clef,
        measure.barlines,
        tuple(_event_skeleton(note) for note in measure.notes),
    )


def _topology(measure: MeasureIR) -> tuple[bool, ...]:
    return tuple(bool(note.chord) for note in measure.notes)


def _duration_total(measure: MeasureIR) -> Fraction:
    return sum(
        (note.duration for note in measure.notes if not note.grace and not note.chord),
        Fraction(0, 1),
    )


def _duration_error(measure: MeasureIR) -> float:
    expected = measure.expected_duration
    if expected is None:
        return 8.0
    return float(abs(_duration_total(measure) - expected))


def _topology_stats(topology: Sequence[bool]) -> tuple[int, int]:
    groups = 0
    maximum = 1 if topology else 0
    current = 0
    for marker in topology:
        if marker:
            current += 1
            maximum = max(maximum, current + 1)
        else:
            if current:
                groups += 1
            current = 0
    if current:
        groups += 1
    return groups, maximum


def _supported_measure(candidate: ChordPatchCandidate) -> bool:
    semantics = candidate.semantics
    notes = semantics.notes
    nodes = candidate.measure.findall("note")
    if (
        semantics.time_signature is None
        or semantics.voice_count != 1
        or len(notes) < 2
        or len(nodes) != len(notes)
        or candidate.measure.find("backup") is not None
        or candidate.measure.find("forward") is not None
    ):
        return False
    anchor: NoteIR | None = None
    chord_size = 0
    for note, node in zip(notes, nodes, strict=True):
        if (
            note.grace
            or note.tuple_ratio is not None
            or node.find("unpitched") is not None
            or (node.find("rest") is not None and node.find("rest").get("measure") == "yes")
            or note.duration <= 0
            or (note.pitch is None and not note.rest)
        ):
            return False
        if note.chord:
            if (
                anchor is None
                or note.rest
                or note.pitch is None
                or anchor.rest
                or anchor.pitch is None
                or note.voice != anchor.voice
                or note.duration != anchor.duration
                or note.note_type != anchor.note_type
                or note.dots != anchor.dots
                or note.onset != anchor.onset
            ):
                return False
            chord_size += 1
            if chord_size + 1 > DEFAULT_POLICY.chord_patch_max_chord_size:
                return False
        else:
            anchor = note
            chord_size = 0
    return not notes[0].chord


def _representative(items: Sequence[ChordPatchCandidate]) -> ChordPatchCandidate:
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


def _set_chord_marker(note: etree._Element, enabled: bool) -> None:
    existing = note.find("chord")
    if existing is not None:
        note.remove(existing)
    if enabled:
        note.insert(0, etree.Element("chord"))


def propose_chord_patch(
    candidates: Sequence[ChordPatchCandidate],
    *,
    template_index: int,
    missing_candidate_count: int,
    calibrator: ChordPatchCalibrator | None = None,
) -> ChordPatchResult:
    """Propose and optionally approve one chord-marker-only repair."""
    if not candidates or template_index < 0 or template_index >= len(candidates):
        return ChordPatchResult(None, (), 0.5, 1.0, False, "invalid_input")
    if missing_candidate_count:
        return ChordPatchResult(None, (), 0.5, 1.0, False, "alignment_gap")
    template = candidates[template_index]
    if not template.valid:
        return ChordPatchResult(None, (), 0.5, 1.0, False, "invalid_template")

    family_members, incomplete_families = group_complete_families(
        candidates,
        family_of=lambda item: item.family,
        valid_of=lambda item: item.valid and _supported_measure(item),
    )
    if len(family_members) < DEFAULT_POLICY.chord_patch_minimum_families:
        return ChordPatchResult(None, (), 0.5, 1.0, False, "insufficient_families")
    valid = [item for members in family_members.values() for item in members]
    template_skeleton = _measure_skeleton(template.semantics)
    if any(_measure_skeleton(item.semantics) != template_skeleton for item in valid):
        return ChordPatchResult(None, (), 0.5, 1.0, False, "non_chord_structure_disagreement")

    family_votes: dict[str, tuple[tuple[bool, ...], ChordPatchCandidate]] = {}
    abstentions = len(incomplete_families)
    for family, members in family_members.items():
        topologies = {_topology(item.semantics) for item in members}
        if len(topologies) != 1:
            abstentions += 1
            continue
        family_votes[family] = (next(iter(topologies)), _representative(members))
    if len(family_votes) < DEFAULT_POLICY.chord_patch_minimum_families:
        return ChordPatchResult(None, (), 0.5, 1.0, False, "insufficient_topology_family_votes")

    grouped: dict[tuple[bool, ...], list[tuple[str, ChordPatchCandidate]]] = {}
    for family, (topology, representative) in family_votes.items():
        grouped.setdefault(topology, []).append((family, representative))
    ranked = sorted(
        grouped.items(),
        key=lambda item: (
            len(item[1]),
            _mean([row.ensemble_probability for _, row in item[1]]),
            item[0],
        ),
        reverse=True,
    )
    winner, winner_rows = ranked[0]
    runner_up = len(ranked[1][1]) if len(ranked) > 1 else 0
    winner_count = len(winner_rows)
    voting_count = len(family_votes)
    if not (
        winner_count >= DEFAULT_POLICY.chord_patch_minimum_supporting_families
        and winner_count > voting_count / 2
        and winner_count - runner_up >= 1
    ):
        return ChordPatchResult(None, (), 0.5, 1.0, False, "no_strict_topology_family_majority")

    template_topology = _topology(template.semantics)
    if winner == template_topology:
        return ChordPatchResult(None, (), 0.5, 1.0, False, "no_chord_change")
    changed = tuple(index for index, (left, right) in enumerate(zip(template_topology, winner, strict=True)) if left != right)
    if (
        not changed
        or len(changed) > DEFAULT_POLICY.chord_patch_max_changed_markers
        or len(changed) / max(1, len(winner)) > DEFAULT_POLICY.chord_patch_max_changed_ratio
    ):
        return ChordPatchResult(None, changed, 0.5, 1.0, False, "change_scope_too_large")

    patched = copy.deepcopy(template.measure)
    patched_notes = patched.findall("note")
    if len(patched_notes) != len(winner):
        return ChordPatchResult(None, changed, 0.5, 1.0, False, "xml_event_count_mismatch")
    for index, marker in enumerate(winner):
        _set_chord_marker(patched_notes[index], marker)

    inherited: dict[str, object] = {
        "divisions": template.semantics.divisions,
        "time": template.semantics.time_signature,
        "key": template.semantics.key_signature,
        "clef": template.semantics.clef,
    }
    parsed, _state = measure_from_xml(patched, inherited)
    if _measure_skeleton(parsed) != template_skeleton or _topology(parsed) != winner:
        return ChordPatchResult(None, changed, 0.5, 1.0, False, "post_patch_structure_changed")
    post_candidate = ChordPatchCandidate(
        variant="patched",
        family="patched",
        measure=patched,
        semantics=parsed,
        page_score=template.page_score,
        page_probability=template.page_probability,
        measure_probability=template.measure_probability,
        visual_probability=template.visual_probability,
        event_probability=template.event_probability,
        context_probability=template.context_probability,
        ensemble_probability=template.ensemble_probability,
        valid=True,
    )
    if not _supported_measure(post_candidate):
        return ChordPatchResult(None, changed, 0.5, 1.0, False, "post_patch_invalid_topology")

    template_error = _duration_error(template.semantics)
    patched_error = _duration_error(parsed)
    if patched_error > template_error + 1e-9:
        return ChordPatchResult(None, changed, 0.5, 1.0, False, "meter_error_worsened")

    support = [row for _, row in winner_rows]
    group_count, max_size = _topology_stats(winner)
    expected = float(template.semantics.expected_duration or Fraction(4, 1))
    added = sum(1 for index in changed if winner[index])
    removed = len(changed) - added
    template_support = len(grouped.get(template_topology, ()))
    item = ChordPatchInput(
        candidate_count=len(valid),
        eligible_family_count=len(family_members),
        voting_family_count=voting_count,
        changed_marker_count=len(changed),
        total_event_count=len(winner),
        added_marker_count=added,
        removed_marker_count=removed,
        winner_family_count=winner_count,
        runner_up_family_count=runner_up,
        template_family_count=template_support,
        incomplete_family_count=abstentions,
        winner_chord_group_count=group_count,
        winner_max_chord_size=max_size,
        expected_measure_duration=expected,
        template_duration_error=template_error,
        patched_duration_error=patched_error,
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
    active_calibrator = calibrator or ChordPatchCalibrator()
    calibration = active_calibrator.calibrate(item)
    if not calibration.accepted:
        return ChordPatchResult(
            None,
            changed,
            calibration.probability,
            calibration.threshold,
            False,
            "model_guard",
            item,
            model_version=calibration.model_version,
        )
    return ChordPatchResult(
        patched,
        changed,
        calibration.probability,
        calibration.threshold,
        True,
        "accepted",
        item,
        model_version=calibration.model_version,
    )
