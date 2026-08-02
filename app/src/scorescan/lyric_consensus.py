from __future__ import annotations

"""Conservative event-level consensus for simple note lyrics.

The repair votes independently per existing note and only changes empty, attribute-free
MusicXML lyric markers.  A family may vote for an event only when every sibling
is valid, has the same event lattice as the current base event and agrees on the marker
set.  This permits complementary errors elsewhere in the measure without treating
correlated preprocessing variants as independent evidence.  The bundled CPU model is
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

from .lyric_xml import (
    LyricState,
    LyricTopology,
    normalized_lyric_topology,
    set_lyric_topology,
    without_lyrics,
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
    "changed_event_count_scaled",
    "changed_event_ratio",
    "added_mark_ratio",
    "removed_mark_ratio",
    "minimum_winner_family_support_ratio",
    "mean_winner_family_support_ratio",
    "minimum_winner_margin_ratio",
    "mean_winner_margin_ratio",
    "maximum_template_family_support_ratio",
    "family_abstention_ratio",
    "winner_lyric_count_scaled",
    "changed_character_count_scaled",
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
class LyricPatchCandidate:
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
class LyricPatchInput:
    candidate_count: int
    eligible_family_count: int
    voting_family_count: int
    changed_event_count: int
    total_event_count: int
    added_mark_count: int
    removed_mark_count: int
    minimum_winner_family_count: int
    mean_winner_family_count: float
    minimum_winner_margin_count: int
    mean_winner_margin_count: float
    maximum_template_family_count: int
    incomplete_family_count: int
    winner_lyric_count: int
    changed_character_count: int
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
        changed = max(1, self.changed_event_count)
        marks = max(1, self.added_mark_count + self.removed_mark_count)
        return [
            unit(max(0, self.candidate_count - 1) / 7.0),
            unit(self.eligible_family_count / 4.0),
            unit(self.voting_family_count / 4.0),
            unit(self.changed_event_count / max(1, DEFAULT_POLICY.lyric_patch_max_changed_events)),
            unit(self.changed_event_count / max(1, self.total_event_count)),
            unit(self.added_mark_count / marks),
            unit(self.removed_mark_count / marks),
            unit(self.minimum_winner_family_count / families),
            unit(self.mean_winner_family_count / families),
            unit(self.minimum_winner_margin_count / families),
            unit(self.mean_winner_margin_count / families),
            unit(self.maximum_template_family_count / families),
            unit(self.incomplete_family_count / max(1, self.eligible_family_count + self.incomplete_family_count)),
            unit(self.winner_lyric_count / max(1, changed)),
            unit(self.changed_character_count / max(1, changed * DEFAULT_POLICY.lyric_patch_max_text_length)),
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
class LyricPatchCalibration:
    probability: float
    threshold: float
    accepted: bool
    model_version: str
    target_precision: float


@dataclass(frozen=True)
class LyricPatchResult:
    patched_measure: etree._Element | None
    changed_event_indices: tuple[int, ...]
    changed_lyric_count: int
    probability: float
    threshold: float
    accepted: bool
    reason: str
    input: LyricPatchInput | None = None
    model_version: str = "disabled"


class LyricPatchCalibrator:
    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).with_name("resources") / "lyric_patch_calibrator.json"
        loaded = load_verified_json(model_path, "lyric_patch_calibration")
        payload = loaded.payload
        self.model = VerifiedRandomForestModel.load(
            model_path,
            "lyric_patch_calibration",
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
            float(DEFAULT_POLICY.lyric_patch_probability_floor),
            max(0.0, min(1.0, stored_threshold)),
        )
        self.target_precision = max(0.0, min(1.0, target_precision))
        self.model_verified = self.model.verified and loaded.verified
        self.model_status = self.model.status if self.model.enabled else loaded.status
        self.model_version = self.model.model_version
        self.enabled = self.model.enabled

    def predict_probability(self, item: LyricPatchInput) -> float:
        return self.model.predict(item.feature_vector(), neutral=0.5)

    def calibrate(self, item: LyricPatchInput) -> LyricPatchCalibration:
        probability = self.predict_probability(item)
        return LyricPatchCalibration(
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


def _supported_measure_xml(measure: etree._Element, semantics: MeasureIR) -> bool:
    notes = measure.findall("note")
    topology = normalized_lyric_topology(notes)
    if (
        semantics.voice_count != 1
        or semantics.expected_duration is None
        or _duration_total(semantics) != semantics.expected_duration
        or not semantics.notes
        or len(semantics.notes) > DEFAULT_POLICY.lyric_patch_max_events
        or len(notes) != len(semantics.notes)
        or measure.find("backup") is not None
        or measure.find("forward") is not None
        or topology is None
    ):
        return False
    for note, node, lyric in zip(semantics.notes, notes, topology, strict=True):
        if (
            note.rest
            or note.pitch is None
            or note.chord
            or note.grace
            or note.tuple_ratio is not None
            or note.duration <= 0
            or node.find("unpitched") is not None
            or (lyric is not None and len(lyric.text) > DEFAULT_POLICY.lyric_patch_max_text_length)
        ):
            return False
    return True


def _representative(items: Sequence[LyricPatchCandidate]) -> LyricPatchCandidate:
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


def propose_lyric_patch(
    candidates: Sequence[LyricPatchCandidate],
    *,
    template_index: int,
    missing_candidate_count: int,
    calibrator: LyricPatchCalibrator | None = None,
    base_measure: etree._Element | None = None,
) -> LyricPatchResult:
    if not candidates or template_index < 0 or template_index >= len(candidates):
        return LyricPatchResult(None, (), 0, 0.5, 1.0, False, "invalid_input")
    if missing_candidate_count:
        return LyricPatchResult(None, (), 0, 0.5, 1.0, False, "alignment_gap")
    template = candidates[template_index]
    if not template.valid:
        return LyricPatchResult(None, (), 0, 0.5, 1.0, False, "invalid_template")

    family_members, incomplete_families = group_complete_families(
        candidates,
        family_of=lambda item: item.family,
        valid_of=lambda item: item.valid,
    )
    if len(family_members) < DEFAULT_POLICY.lyric_patch_minimum_families:
        return LyricPatchResult(None, (), 0, 0.5, 1.0, False, "insufficient_families")

    source_base = base_measure if base_measure is not None else template.measure
    inherited: dict[str, object] = {
        "divisions": template.semantics.divisions,
        "time": template.semantics.time_signature,
        "key": template.semantics.key_signature,
        "clef": template.semantics.clef,
    }
    base_semantics, _ = measure_from_xml(source_base, inherited)
    if not _supported_measure_xml(source_base, base_semantics):
        return LyricPatchResult(None, (), 0, 0.5, 1.0, False, "unsupported_base_measure")
    base_notes = source_base.findall("note")
    base_topology = normalized_lyric_topology(base_notes)
    if base_topology is None:
        return LyricPatchResult(None, (), 0, 0.5, 1.0, False, "invalid_base_topology")

    candidate_topologies: dict[int, LyricTopology] = {}
    candidate_positions = {id(item): index for index, item in enumerate(candidates)}
    for index, item in enumerate(candidates):
        topology = normalized_lyric_topology(item.measure.findall("note"))
        if topology is not None:
            candidate_topologies[index] = topology

    proposal = list(base_topology)
    changed_indices: list[int] = []
    winner_counts: list[int] = []
    margins: list[int] = []
    template_counts: list[int] = []
    support_families_by_event: list[set[str]] = []
    voting_families_by_event: list[set[str]] = []
    family_representatives: dict[str, LyricPatchCandidate] = {}
    abstentions = len(incomplete_families)

    for event_index, base_note in enumerate(base_semantics.notes):
        base_skeleton = _event_skeleton(base_note)
        votes: dict[str, LyricState | None] = {}
        representatives: dict[str, LyricPatchCandidate] = {}
        for family, members in family_members.items():
            sibling_values: set[LyricState | None] = set()
            supported = True
            for member in members:
                candidate_index = candidate_positions.get(id(member))
                if candidate_index is None:
                    supported = False
                    break
                topology = candidate_topologies.get(candidate_index)
                if (
                    topology is None
                    or event_index >= len(member.semantics.notes)
                    or len(member.semantics.notes) != len(base_semantics.notes)
                    or _event_skeleton(member.semantics.notes[event_index]) != base_skeleton
                    or event_index >= len(topology)
                ):
                    supported = False
                    break
                sibling_values.add(topology[event_index])
            if not supported or len(sibling_values) != 1:
                continue
            votes[family] = next(iter(sibling_values))
            representatives[family] = _representative(members)
        if len(votes) < DEFAULT_POLICY.lyric_patch_minimum_families:
            continue
        grouped = Counter(votes.values())
        ranked = sorted(grouped.items(), key=lambda item: (item[1], repr(item[0])), reverse=True)
        winner, winner_count = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0
        if not (
            winner_count >= DEFAULT_POLICY.lyric_patch_minimum_supporting_families
            and winner_count > len(votes) / 2
            and winner_count - runner_up >= 1
        ):
            continue
        if winner == base_topology[event_index]:
            continue
        proposal[event_index] = winner
        changed_indices.append(event_index)
        winner_counts.append(winner_count)
        margins.append(winner_count - runner_up)
        template_counts.append(grouped.get(base_topology[event_index], 0))
        support_families = {family for family, value in votes.items() if value == winner}
        support_families_by_event.append(support_families)
        voting_families_by_event.append(set(votes))
        for family in support_families:
            family_representatives[family] = representatives[family]

    changed = tuple(changed_indices)
    if not changed:
        return LyricPatchResult(None, (), 0, 0.5, 1.0, False, "no_lyric_change")
    if (
        len(changed) > DEFAULT_POLICY.lyric_patch_max_changed_events
        or len(changed) / max(1, len(base_semantics.notes)) > DEFAULT_POLICY.lyric_patch_max_changed_ratio
    ):
        return LyricPatchResult(None, changed, 0, 0.5, 1.0, False, "change_scope_too_large")

    common_support = set.intersection(*support_families_by_event)
    common_voting = set.intersection(*voting_families_by_event)
    if len(common_support) < DEFAULT_POLICY.lyric_patch_minimum_supporting_families:
        return LyricPatchResult(None, changed, 0, 0.5, 1.0, False, "inconsistent_cross_event_support")

    added_marks = sum(proposal[index] is not None and proposal[index] != base_topology[index] for index in changed)
    removed_marks = sum(base_topology[index] is not None and proposal[index] != base_topology[index] for index in changed)
    changed_marks = int(added_marks) + int(removed_marks)
    changed_characters = sum(
        max(
            len(proposal[index].text if proposal[index] is not None else ""),
            len(base_topology[index].text if base_topology[index] is not None else ""),
        )
        for index in changed
    )
    if changed_marks <= 0:
        return LyricPatchResult(None, changed, 0, 0.5, 1.0, False, "no_lyric_change")

    patched = copy.deepcopy(source_base)
    patched_notes = patched.findall("note")
    try:
        before_non_lyric = [without_lyrics(note) for note in patched_notes]
        set_lyric_topology(patched_notes, proposal)
        after_non_lyric = [without_lyrics(note) for note in patched.findall("note")]
    except ValueError:
        return LyricPatchResult(None, changed, changed_marks, 0.5, 1.0, False, "xml_topology_error")
    if before_non_lyric != after_non_lyric:
        return LyricPatchResult(None, changed, changed_marks, 0.5, 1.0, False, "unrelated_event_xml_changed")

    parsed, _ = measure_from_xml(patched, inherited)
    parsed_topology = normalized_lyric_topology(patched.findall("note"))
    if (
        parsed_topology != tuple(proposal)
        or len(parsed.notes) != len(base_semantics.notes)
        or any(_event_skeleton(left) != _event_skeleton(right) for left, right in zip(parsed.notes, base_semantics.notes, strict=True))
    ):
        return LyricPatchResult(None, changed, changed_marks, 0.5, 1.0, False, "post_patch_validation_failed")

    support = [family_representatives[family] for family in sorted(common_support)]
    item = LyricPatchInput(
        candidate_count=sum(len(members) for members in family_members.values()),
        eligible_family_count=len(family_members),
        voting_family_count=len(common_voting),
        changed_event_count=len(changed),
        total_event_count=len(base_semantics.notes),
        added_mark_count=added_marks,
        removed_mark_count=removed_marks,
        minimum_winner_family_count=min(winner_counts),
        mean_winner_family_count=_mean(winner_counts, 0.0),
        minimum_winner_margin_count=min(margins),
        mean_winner_margin_count=_mean(margins, 0.0),
        maximum_template_family_count=max(template_counts, default=0),
        incomplete_family_count=abstentions + max(0, len(family_members) - len(common_voting)),
        winner_lyric_count=sum(proposal[index] is not None for index in changed),
        changed_character_count=changed_characters,
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
    active_calibrator = calibrator or LyricPatchCalibrator()
    calibration = active_calibrator.calibrate(item)
    if not calibration.accepted:
        return LyricPatchResult(
            None,
            changed,
            changed_marks,
            calibration.probability,
            calibration.threshold,
            False,
            "model_guard",
            item,
            calibration.model_version,
        )
    return LyricPatchResult(
        patched,
        changed,
        changed_marks,
        calibration.probability,
        calibration.threshold,
        True,
        "accepted",
        item,
        calibration.model_version,
    )
