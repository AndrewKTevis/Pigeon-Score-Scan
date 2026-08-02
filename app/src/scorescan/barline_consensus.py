from __future__ import annotations

"""Conservative repeat/barline topology consensus.

Whole-measure selection can leave a wrong repeat sign in place when candidates contain
complementary note errors.  This module votes only on simple left/right MusicXML
``<barline>`` elements.  It may copy a bar style and a forward/backward repeat marker
already present in independent preprocessing families, but never creates endings,
fermatas or other navigation semantics.  Invalid or split siblings make the complete
family abstain and the bundled CPU model is veto-only.
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
from .score_ir import MeasureIR, measure_from_xml
from .tree_model import VerifiedRandomForestModel
from .variant_family import group_complete_families

FEATURE_NAMES = (
    "candidate_count_scaled",
    "eligible_family_count_scaled",
    "voting_family_count_scaled",
    "changed_location_count_scaled",
    "changed_location_ratio",
    "added_barline_ratio",
    "removed_barline_ratio",
    "repeat_change_ratio",
    "style_change_ratio",
    "winner_family_support_ratio",
    "winner_margin_ratio",
    "template_family_support_ratio",
    "family_abstention_ratio",
    "winner_barline_count_scaled",
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

_ALLOWED_STYLES = {
    "regular",
    "dotted",
    "dashed",
    "heavy",
    "light-light",
    "light-heavy",
    "heavy-light",
    "heavy-heavy",
    "tick",
    "short",
    "none",
}
_LOCATION_ORDER = {"left": 0, "right": 1}
_MEASURE_ORDER = {
    "print": 0,
    "sound": 1,
    "listening": 2,
    "attributes": 3,
    "direction": 4,
    "harmony": 5,
    "figured-bass": 6,
    "note": 7,
    "backup": 8,
    "forward": 9,
    "barline": 10,
    "grouping": 11,
    "link": 12,
    "bookmark": 13,
}

BarlineTopology = tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class BarlinePatchCandidate:
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
class BarlinePatchInput:
    candidate_count: int
    eligible_family_count: int
    voting_family_count: int
    changed_location_count: int
    added_barline_count: int
    removed_barline_count: int
    repeat_change_count: int
    style_change_count: int
    winner_family_count: int
    runner_up_family_count: int
    template_family_count: int
    incomplete_family_count: int
    winner_barline_count: int
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
        changed = max(1, self.changed_location_count)
        return [
            unit(max(0, self.candidate_count - 1) / 7.0),
            unit(self.eligible_family_count / 4.0),
            unit(self.voting_family_count / 4.0),
            unit(self.changed_location_count / 2.0),
            unit(self.changed_location_count / 2.0),
            unit(self.added_barline_count / changed),
            unit(self.removed_barline_count / changed),
            unit(self.repeat_change_count / changed),
            unit(self.style_change_count / changed),
            unit(self.winner_family_count / families),
            unit((self.winner_family_count - self.runner_up_family_count) / families),
            unit(self.template_family_count / families),
            unit(self.incomplete_family_count / max(1, self.eligible_family_count + self.incomplete_family_count)),
            unit(self.winner_barline_count / 2.0),
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
class BarlinePatchCalibration:
    probability: float
    threshold: float
    accepted: bool
    model_version: str
    target_precision: float


@dataclass(frozen=True)
class BarlinePatchResult:
    patched_measure: etree._Element | None
    changed_locations: tuple[str, ...]
    changed_repeat_count: int
    probability: float
    threshold: float
    accepted: bool
    reason: str
    input: BarlinePatchInput | None = None
    model_version: str = "disabled"


class BarlinePatchCalibrator:
    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).with_name("resources") / "barline_patch_calibrator.json"
        loaded = load_verified_json(model_path, "barline_patch_calibration")
        payload = loaded.payload
        self.model = VerifiedRandomForestModel.load(
            model_path,
            "barline_patch_calibration",
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
            float(DEFAULT_POLICY.barline_patch_probability_floor),
            max(0.0, min(1.0, stored_threshold)),
        )
        self.target_precision = max(0.0, min(1.0, target_precision))
        self.model_verified = self.model.verified and loaded.verified
        self.model_status = self.model.status if self.model.enabled else loaded.status
        self.model_version = self.model.model_version
        self.enabled = self.model.enabled

    def predict_probability(self, item: BarlinePatchInput) -> float:
        return self.model.predict(item.feature_vector(), neutral=0.5)

    def calibrate(self, item: BarlinePatchInput) -> BarlinePatchCalibration:
        probability = self.predict_probability(item)
        return BarlinePatchCalibration(
            probability=round(probability, 6),
            threshold=round(self.threshold, 6),
            accepted=bool(self.enabled and self.model_verified and probability >= self.threshold),
            model_version=self.model_version,
            target_precision=round(self.target_precision, 6),
        )


def _mean(values: Sequence[float], default: float = 0.5) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else default


def _normalize_location(value: str | None) -> str:
    text = str(value or "").strip().casefold()
    return text or "right"


def _topology(measure: etree._Element) -> BarlineTopology | None:
    rows: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for barline in measure.findall("barline"):
        if any(child.tag not in {"bar-style", "repeat"} for child in barline):
            return None
        location = _normalize_location(barline.get("location"))
        if location not in _LOCATION_ORDER or location in seen:
            return None
        seen.add(location)
        style = str(barline.findtext("bar-style") or "").strip().casefold()
        if style and style not in _ALLOWED_STYLES:
            return None
        repeat = barline.find("repeat")
        direction = ""
        if repeat is not None:
            if repeat.get("times") or repeat.get("winged") or len(repeat):
                return None
            direction = str(repeat.get("direction") or "").strip().casefold()
            if direction not in {"forward", "backward"}:
                return None
            if direction == "forward" and location != "left":
                return None
            if direction == "backward" and location != "right":
                return None
        if not style and not direction:
            return None
        rows.append((location, style, direction))
    rows.sort(key=lambda item: _LOCATION_ORDER[item[0]])
    return tuple(rows)


def _location_map(topology: BarlineTopology) -> dict[str, tuple[str, str]]:
    return {location: (style, repeat) for location, style, repeat in topology}


def _non_barline_identity(measure: etree._Element, semantics: MeasureIR) -> tuple[object, ...]:
    # The XML digest catches unsupported note-level details not represented by ScoreIR,
    # while temporarily removing barlines keeps this patch independent from them.
    clone = copy.deepcopy(measure)
    for item in clone.findall("barline"):
        clone.remove(item)
    return (
        semantics.divisions,
        semantics.time_signature,
        semantics.key_signature,
        semantics.clef,
        tuple(note.stable_tuple() for note in semantics.notes),
        tuple(direction.stable_tuple() for direction in semantics.directions),
        etree.tostring(clone, encoding="utf-8"),
    )


def _insert_barline(measure: etree._Element, barline: etree._Element) -> None:
    rank = _MEASURE_ORDER["barline"]
    insertion = len(measure)
    for index, child in enumerate(measure):
        if _MEASURE_ORDER.get(child.tag, 99) > rank:
            insertion = index
            break
    measure.insert(insertion, barline)


def _write_topology(measure: etree._Element, source: etree._Element, topology: BarlineTopology) -> bool:
    source_by_location: dict[str, etree._Element] = {}
    for barline in source.findall("barline"):
        source_by_location[_normalize_location(barline.get("location"))] = barline
    for item in measure.findall("barline"):
        measure.remove(item)
    for location, _style, _repeat in topology:
        source_item = source_by_location.get(location)
        if source_item is None:
            return False
        replacement = copy.deepcopy(source_item)
        replacement.set("location", location)
        _insert_barline(measure, replacement)
    return True


def _best_candidate(items: Sequence[BarlinePatchCandidate]) -> BarlinePatchCandidate:
    return max(
        items,
        key=lambda item: (
            item.ensemble_probability,
            item.context_probability,
            item.visual_probability,
            item.measure_probability,
            item.event_probability,
            item.page_score,
            item.variant,
        ),
    )


def propose_barline_patch(
    candidates: Sequence[BarlinePatchCandidate],
    *,
    template_index: int,
    missing_candidate_count: int,
    calibrator: BarlinePatchCalibrator | None = None,
    base_measure: etree._Element | None = None,
) -> BarlinePatchResult:
    active = calibrator or BarlinePatchCalibrator()
    threshold = float(active.threshold)
    if not candidates or template_index < 0 or template_index >= len(candidates):
        return BarlinePatchResult(None, (), 0, 0.5, threshold, False, "invalid_input", model_version=active.model_version)
    if missing_candidate_count:
        return BarlinePatchResult(None, (), 0, 0.5, threshold, False, "alignment_gap", model_version=active.model_version)
    template = candidates[template_index]
    if not template.valid:
        return BarlinePatchResult(None, (), 0, 0.5, threshold, False, "invalid_template", model_version=active.model_version)

    family_members, incomplete_families = group_complete_families(
        candidates,
        family_of=lambda item: item.family,
        valid_of=lambda item: item.valid,
    )
    if len(family_members) < DEFAULT_POLICY.barline_patch_minimum_families:
        return BarlinePatchResult(None, (), 0, 0.5, threshold, False, "insufficient_families", model_version=active.model_version)

    family_votes: dict[str, tuple[BarlineTopology, BarlinePatchCandidate]] = {}
    topology_invalid_families = 0
    for family, items in family_members.items():
        values = {_topology(item.measure) for item in items}
        if len(values) != 1 or None in values:
            topology_invalid_families += 1
            continue
        family_votes[family] = (next(iter(values)), _best_candidate(items))  # type: ignore[arg-type]
    if len(family_votes) < DEFAULT_POLICY.barline_patch_minimum_families:
        return BarlinePatchResult(None, (), 0, 0.5, threshold, False, "insufficient_barline_family_votes", model_version=active.model_version)

    grouped: dict[BarlineTopology, list[tuple[str, BarlinePatchCandidate]]] = {}
    for family, (value, representative) in family_votes.items():
        grouped.setdefault(value, []).append((family, representative))
    ranked = sorted(
        grouped.items(),
        key=lambda item: (
            len(item[1]),
            _mean([candidate.ensemble_probability for _, candidate in item[1]]),
            repr(item[0]),
        ),
        reverse=True,
    )
    winner, winner_rows = ranked[0]
    runner_up_count = len(ranked[1][1]) if len(ranked) > 1 else 0
    winner_count = len(winner_rows)
    voting_count = len(family_votes)
    if not (
        winner_count >= DEFAULT_POLICY.barline_patch_minimum_supporting_families
        and winner_count > voting_count / 2
        and winner_count - runner_up_count >= 1
    ):
        return BarlinePatchResult(None, (), 0, 0.5, threshold, False, "no_strict_barline_family_majority", model_version=active.model_version)

    template_topology = _topology(template.measure)
    if template_topology is None:
        return BarlinePatchResult(None, (), 0, 0.5, threshold, False, "unsupported_template_barline", model_version=active.model_version)
    if winner == template_topology:
        return BarlinePatchResult(None, (), 0, 0.5, threshold, False, "no_barline_change", model_version=active.model_version)

    template_map = _location_map(template_topology)
    winner_map = _location_map(winner)
    changed_locations = tuple(
        location for location in ("left", "right")
        if template_map.get(location) != winner_map.get(location)
    )
    if not changed_locations or len(changed_locations) > DEFAULT_POLICY.barline_patch_max_changed_locations:
        return BarlinePatchResult(None, changed_locations, 0, 0.5, threshold, False, "barline_change_scope", model_version=active.model_version)

    added = sum(location not in template_map for location in changed_locations)
    removed = sum(location not in winner_map for location in changed_locations)
    repeat_changes = sum(
        template_map.get(location, ("", ""))[1] != winner_map.get(location, ("", ""))[1]
        for location in changed_locations
    )
    style_changes = sum(
        template_map.get(location, ("", ""))[0] != winner_map.get(location, ("", ""))[0]
        for location in changed_locations
    )
    if repeat_changes > DEFAULT_POLICY.barline_patch_max_repeat_changes:
        return BarlinePatchResult(None, changed_locations, repeat_changes, 0.5, threshold, False, "repeat_change_scope", model_version=active.model_version)

    representatives = [candidate for _, candidate in winner_rows]
    representative = _best_candidate(representatives)
    template_family_count = len(grouped.get(template_topology, ()))
    incomplete_count = len(incomplete_families) + topology_invalid_families
    item = BarlinePatchInput(
        candidate_count=sum(len(items) for items in family_members.values()),
        eligible_family_count=len(family_members),
        voting_family_count=voting_count,
        changed_location_count=len(changed_locations),
        added_barline_count=added,
        removed_barline_count=removed,
        repeat_change_count=repeat_changes,
        style_change_count=style_changes,
        winner_family_count=winner_count,
        runner_up_family_count=runner_up_count,
        template_family_count=template_family_count,
        incomplete_family_count=incomplete_count,
        winner_barline_count=len(winner),
        mean_support_page_probability=_mean([row.page_probability for row in representatives]),
        mean_support_measure_probability=_mean([row.measure_probability for row in representatives]),
        mean_support_visual_probability=_mean([row.visual_probability for row in representatives]),
        mean_support_event_probability=_mean([row.event_probability for row in representatives]),
        mean_support_context_probability=_mean([row.context_probability for row in representatives]),
        mean_support_ensemble_probability=_mean([row.ensemble_probability for row in representatives]),
        minimum_support_ensemble_probability=min(row.ensemble_probability for row in representatives),
        mean_support_page_score_margin=_mean([row.page_score - template.page_score for row in representatives], 0.0),
        mean_support_vs_template_measure_probability=_mean([row.measure_probability - template.measure_probability for row in representatives], 0.0),
        mean_support_vs_template_visual_probability=_mean([row.visual_probability - template.visual_probability for row in representatives], 0.0),
        mean_support_vs_template_event_probability=_mean([row.event_probability - template.event_probability for row in representatives], 0.0),
        mean_support_vs_template_context_probability=_mean([row.context_probability - template.context_probability for row in representatives], 0.0),
        mean_support_vs_template_ensemble_probability=_mean([row.ensemble_probability - template.ensemble_probability for row in representatives], 0.0),
    )
    calibration = active.calibrate(item)
    if not calibration.accepted:
        return BarlinePatchResult(
            None, changed_locations, repeat_changes, calibration.probability,
            calibration.threshold, False, "model_guard", item,
            model_version=calibration.model_version,
        )

    working = copy.deepcopy(base_measure if base_measure is not None else template.measure)
    inherited = {
        "divisions": template.semantics.divisions,
        "time": template.semantics.time_signature,
        "key": template.semantics.key_signature,
        "clef": template.semantics.clef,
    }
    base_semantics, _ = measure_from_xml(working, inherited)
    base_identity = _non_barline_identity(working, base_semantics)
    if not _write_topology(working, representative.measure, winner):
        return BarlinePatchResult(None, changed_locations, repeat_changes, calibration.probability, calibration.threshold, False, "xml_barline_copy_failed", item, model_version=calibration.model_version)
    parsed, _ = measure_from_xml(working, inherited)
    if _topology(working) != winner:
        return BarlinePatchResult(None, changed_locations, repeat_changes, calibration.probability, calibration.threshold, False, "post_patch_topology_mismatch", item, model_version=calibration.model_version)
    if _non_barline_identity(working, parsed) != base_identity:
        return BarlinePatchResult(None, changed_locations, repeat_changes, calibration.probability, calibration.threshold, False, "post_patch_non_barline_changed", item, model_version=calibration.model_version)
    if parsed.barlines != tuple((location, style, repeat, ":") for location, style, repeat in winner):
        return BarlinePatchResult(None, changed_locations, repeat_changes, calibration.probability, calibration.threshold, False, "post_patch_semantic_mismatch", item, model_version=calibration.model_version)

    return BarlinePatchResult(
        working,
        changed_locations,
        repeat_changes,
        calibration.probability,
        calibration.threshold,
        True,
        "accepted",
        item,
        model_version=calibration.model_version,
    )
