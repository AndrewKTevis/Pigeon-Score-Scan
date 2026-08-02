from __future__ import annotations

"""Conservative consensus for time, key and clef attributes.

A wrong score attribute has a wider blast radius than an isolated note error: a bad
meter changes rhythm validation, a bad key changes accidental semantics, and a bad clef
changes every written pitch.  Whole-measure replacement can correct these errors, but
only when one complete OMR candidate wins.  This module provides a narrower fallback.

Each attribute is decided independently.  Correlated preprocessing siblings receive at
most one family vote, invalid or split siblings make the family abstain, and at least
three independent families must form a strict majority.  A proposal also needs an
observable attribute boundary and must pass deterministic MusicXML/music guards.  The
bundled CPU model is veto-only: it cannot invent an attribute or choose a value that was
not emitted by a complete candidate family.
"""

import copy
import math
import statistics
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal, Sequence

from lxml import etree

from .model_registry import load_verified_json
from .policy import DEFAULT_POLICY
from .score_ir import MeasureIR, measure_from_xml
from .tree_model import VerifiedRandomForestModel
from .variant_family import group_complete_families

AttributeKind = Literal["time", "key", "clef"]
AttributeValue = tuple[object, ...]
ATTRIBUTE_KINDS: tuple[AttributeKind, ...] = ("time", "key", "clef")

FEATURE_NAMES = (
    "candidate_count_scaled",
    "eligible_family_count_scaled",
    "voting_family_count_scaled",
    "winner_family_support_ratio",
    "winner_family_margin_ratio",
    "template_family_support_ratio",
    "family_abstention_ratio",
    "attribute_is_time",
    "attribute_is_key",
    "attribute_is_clef",
    "is_first_measure",
    "is_last_measure",
    "template_has_explicit_attribute",
    "support_explicit_attribute_ratio",
    "support_previous_continuity_ratio",
    "support_following_continuity_ratio",
    "template_previous_continuity",
    "template_following_continuity",
    "template_meter_error_scaled",
    "winner_meter_error_scaled",
    "meter_error_improvement_scaled",
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
class AttributePatchCandidate:
    variant: str
    family: str
    measure: etree._Element
    semantics: MeasureIR
    previous_semantics: MeasureIR | None
    following_semantics: MeasureIR | None
    page_score: float
    page_probability: float
    measure_probability: float
    visual_probability: float
    event_probability: float
    context_probability: float
    ensemble_probability: float
    valid: bool


@dataclass(frozen=True)
class AttributePatchInput:
    candidate_count: int
    eligible_family_count: int
    voting_family_count: int
    winner_family_support_ratio: float
    winner_family_margin_ratio: float
    template_family_support_ratio: float
    family_abstention_ratio: float
    attribute_kind: AttributeKind
    is_first_measure: bool
    is_last_measure: bool
    template_has_explicit_attribute: bool
    support_explicit_attribute_ratio: float
    support_previous_continuity_ratio: float
    support_following_continuity_ratio: float
    template_previous_continuity: float
    template_following_continuity: float
    template_meter_error: float
    winner_meter_error: float
    meter_error_improvement: float
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

        return [
            unit(max(0, self.candidate_count - 1) / 7.0),
            unit(self.eligible_family_count / 4.0),
            unit(self.voting_family_count / 4.0),
            unit(self.winner_family_support_ratio),
            unit(self.winner_family_margin_ratio),
            unit(self.template_family_support_ratio),
            unit(self.family_abstention_ratio),
            1.0 if self.attribute_kind == "time" else 0.0,
            1.0 if self.attribute_kind == "key" else 0.0,
            1.0 if self.attribute_kind == "clef" else 0.0,
            1.0 if self.is_first_measure else 0.0,
            1.0 if self.is_last_measure else 0.0,
            1.0 if self.template_has_explicit_attribute else 0.0,
            unit(self.support_explicit_attribute_ratio),
            unit(self.support_previous_continuity_ratio),
            unit(self.support_following_continuity_ratio),
            unit(self.template_previous_continuity),
            unit(self.template_following_continuity),
            unit(self.template_meter_error / 4.0),
            unit(self.winner_meter_error / 4.0),
            signed(self.meter_error_improvement / 4.0),
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
class AttributePatchCalibration:
    probability: float
    threshold: float
    accepted: bool
    model_version: str
    target_precision: float


@dataclass(frozen=True)
class AttributePatchDecision:
    kind: AttributeKind
    proposed_value: AttributeValue | None
    probability: float
    threshold: float
    accepted: bool
    reason: str
    input: AttributePatchInput | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "proposed_value": list(self.proposed_value) if self.proposed_value is not None else None,
            "probability": self.probability,
            "threshold": self.threshold,
            "accepted": self.accepted,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AttributePatchResult:
    patched_measure: etree._Element | None
    changed_attributes: tuple[AttributeKind, ...]
    probability: float
    threshold: float
    accepted: bool
    reason: str
    decisions: tuple[AttributePatchDecision, ...] = ()
    model_version: str = "disabled"


class AttributePatchCalibrator:
    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).with_name("resources") / "attribute_patch_calibrator.json"
        loaded = load_verified_json(model_path, "attribute_patch_calibration")
        payload = loaded.payload
        self.model = VerifiedRandomForestModel.load(
            model_path,
            "attribute_patch_calibration",
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
            float(DEFAULT_POLICY.attribute_patch_probability_floor),
            max(0.0, min(1.0, stored_threshold)),
        )
        self.target_precision = max(0.0, min(1.0, target_precision))
        self.model_verified = self.model.verified and loaded.verified
        self.model_status = self.model.status if self.model.enabled else loaded.status
        self.model_version = self.model.model_version
        self.enabled = self.model.enabled

    def predict_probability(self, item: AttributePatchInput) -> float:
        return self.model.predict(item.feature_vector(), neutral=0.5)

    def calibrate(self, item: AttributePatchInput) -> AttributePatchCalibration:
        probability = self.predict_probability(item)
        accepted = bool(
            self.enabled
            and self.model_verified
            and probability >= self.threshold
        )
        return AttributePatchCalibration(
            probability=round(probability, 6),
            threshold=round(self.threshold, 6),
            accepted=accepted,
            model_version=self.model_version,
            target_precision=round(self.target_precision, 6),
        )


def _mean(values: Sequence[float], default: float = 0.5) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else default


def _attribute_value(measure: MeasureIR | None, kind: AttributeKind) -> AttributeValue | None:
    if measure is None:
        return None
    if kind == "time":
        value = measure.time_signature
        return tuple(value) if value is not None else None
    if kind == "key":
        value = measure.key_signature
        if value is None:
            return None
        return (int(value[0]), str(value[1]).strip().casefold())
    value = measure.clef
    if value is None:
        return None
    return (str(value[0]).strip().upper(), int(value[1]), int(value[2]))


def _valid_value(kind: AttributeKind, value: AttributeValue | None) -> bool:
    if value is None:
        return False
    if kind == "time":
        if len(value) != 2:
            return False
        beats, beat_type = int(value[0]), int(value[1])
        return 1 <= beats <= 16 and beat_type in {1, 2, 4, 8, 16, 32}
    if kind == "key":
        if len(value) != 2:
            return False
        fifths, mode = int(value[0]), str(value[1]).strip().casefold()
        return -7 <= fifths <= 7 and mode in {
            "", "major", "minor", "dorian", "phrygian", "lydian", "mixolydian",
            "aeolian", "ionian", "locrian", "none",
        }
    if len(value) != 3:
        return False
    sign, line, octave = str(value[0]).upper(), int(value[1]), int(value[2])
    if octave not in {-2, -1, 0, 1, 2}:
        return False
    return (
        (sign == "G" and line in {1, 2})
        or (sign == "F" and line in {3, 4, 5})
        or (sign == "C" and line in {1, 2, 3, 4, 5})
    )


def _has_explicit_attribute(measure: etree._Element, kind: AttributeKind) -> bool:
    return measure.find(f"./attributes/{kind}") is not None


def _measure_span(measure: MeasureIR) -> Fraction:
    ends = [
        note.onset + note.duration
        for note in measure.notes
        if not note.grace and note.duration > 0
    ]
    return max(ends, default=Fraction(0, 1))


def _meter_error(measure: MeasureIR, value: AttributeValue | None) -> float:
    if value is None or len(value) != 2:
        return 4.0
    beats, beat_type = int(value[0]), int(value[1])
    if beats <= 0 or beat_type <= 0:
        return 4.0
    expected = Fraction(beats * 4, beat_type)
    return min(4.0, float(abs(_measure_span(measure) - expected)))


def _continuity(
    candidates: Sequence[AttributePatchCandidate],
    kind: AttributeKind,
    value: AttributeValue,
    *,
    previous: bool,
    edge_default: float,
) -> float:
    observed: list[float] = []
    for candidate in candidates:
        neighbour = candidate.previous_semantics if previous else candidate.following_semantics
        if neighbour is None:
            continue
        observed.append(1.0 if _attribute_value(neighbour, kind) == value else 0.0)
    return _mean(observed, edge_default)


def _non_attribute_identity(measure: MeasureIR) -> tuple[object, ...]:
    return (
        measure.divisions,
        tuple(note.stable_tuple() for note in measure.notes),
        tuple(direction.stable_tuple() for direction in measure.directions),
        measure.barlines,
    )


_ATTRIBUTE_ORDER = {"divisions": 0, "key": 1, "time": 2, "staves": 3, "part-symbol": 4, "instruments": 5, "clef": 6}
_MEASURE_ORDER = {"print": 0, "sound": 1, "listening": 2, "attributes": 3, "direction": 4, "harmony": 5, "figured-bass": 6, "note": 7, "backup": 8, "forward": 9, "barline": 10, "grouping": 11, "link": 12, "bookmark": 13}


def _ensure_attributes(measure: etree._Element) -> etree._Element:
    attributes = measure.find("attributes")
    if attributes is not None:
        return attributes
    attributes = etree.Element("attributes")
    insertion = len(measure)
    rank = _MEASURE_ORDER["attributes"]
    for index, child in enumerate(measure):
        if _MEASURE_ORDER.get(child.tag, 99) > rank:
            insertion = index
            break
    measure.insert(insertion, attributes)
    return attributes


def _attribute_element(kind: AttributeKind, value: AttributeValue) -> etree._Element:
    element = etree.Element(kind)
    if kind == "time":
        etree.SubElement(element, "beats").text = str(int(value[0]))
        etree.SubElement(element, "beat-type").text = str(int(value[1]))
    elif kind == "key":
        etree.SubElement(element, "fifths").text = str(int(value[0]))
        mode = str(value[1]).strip().casefold()
        if mode:
            etree.SubElement(element, "mode").text = mode
    else:
        etree.SubElement(element, "sign").text = str(value[0]).upper()
        etree.SubElement(element, "line").text = str(int(value[1]))
        octave = int(value[2])
        if octave:
            etree.SubElement(element, "clef-octave-change").text = str(octave)
    return element


def _write_attribute(measure: etree._Element, kind: AttributeKind, value: AttributeValue) -> None:
    attributes = _ensure_attributes(measure)
    existing = attributes.find(kind)
    replacement = _attribute_element(kind, value)
    if existing is not None:
        attributes.replace(existing, replacement)
        return
    rank = _ATTRIBUTE_ORDER[kind]
    insertion = len(attributes)
    for index, child in enumerate(attributes):
        if _ATTRIBUTE_ORDER.get(child.tag, 99) > rank:
            insertion = index
            break
    attributes.insert(insertion, replacement)


def _best_candidate(items: Sequence[AttributePatchCandidate]) -> AttributePatchCandidate:
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


def _decision(
    kind: AttributeKind,
    value: AttributeValue | None,
    threshold: float,
    reason: str,
    *,
    probability: float = 0.5,
    accepted: bool = False,
    item: AttributePatchInput | None = None,
) -> AttributePatchDecision:
    return AttributePatchDecision(kind, value, probability, threshold, accepted, reason, item)


def propose_attribute_patch(
    candidates: Sequence[AttributePatchCandidate],
    *,
    template_index: int,
    missing_candidate_count: int,
    is_first_measure: bool,
    is_last_measure: bool,
    calibrator: AttributePatchCalibrator | None = None,
    base_measure: etree._Element | None = None,
) -> AttributePatchResult:
    """Propose independently gated time/key/clef repairs for one aligned measure."""
    active_calibrator = calibrator or AttributePatchCalibrator()
    threshold = float(active_calibrator.threshold)
    if not candidates or template_index < 0 or template_index >= len(candidates):
        return AttributePatchResult(None, (), 0.5, threshold, False, "invalid_input", model_version=active_calibrator.model_version)
    if missing_candidate_count:
        return AttributePatchResult(None, (), 0.5, threshold, False, "alignment_gap", model_version=active_calibrator.model_version)

    template = candidates[template_index]
    if not template.valid:
        return AttributePatchResult(None, (), 0.5, threshold, False, "invalid_template", model_version=active_calibrator.model_version)
    family_members, incomplete_families = group_complete_families(
        candidates,
        family_of=lambda item: item.family,
        valid_of=lambda item: item.valid,
    )
    families = sorted(family_members)
    if len(families) < DEFAULT_POLICY.attribute_patch_minimum_families:
        return AttributePatchResult(None, (), 0.5, threshold, False, "insufficient_families", model_version=active_calibrator.model_version)

    working = copy.deepcopy(base_measure if base_measure is not None else template.measure)
    base_semantics, _ = measure_from_xml(
        working,
        {
            "divisions": template.semantics.divisions,
            "time": template.semantics.time_signature,
            "key": template.semantics.key_signature,
            "clef": template.semantics.clef,
        },
    )
    base_identity = _non_attribute_identity(base_semantics)
    decisions: list[AttributePatchDecision] = []
    accepted_values: dict[AttributeKind, AttributeValue] = {}

    for kind in ATTRIBUTE_KINDS:
        template_value = _attribute_value(template.semantics, kind)
        family_votes: dict[str, tuple[AttributeValue, AttributePatchCandidate]] = {}
        for family, items in family_members.items():
            values = {_attribute_value(item.semantics, kind) for item in items}
            if len(values) != 1:
                continue
            value = next(iter(values))
            if not _valid_value(kind, value):
                continue
            family_votes[family] = (value, _best_candidate(items))  # type: ignore[arg-type]

        if len(family_votes) < DEFAULT_POLICY.attribute_patch_minimum_families:
            decisions.append(_decision(kind, None, threshold, "insufficient_attribute_family_votes"))
            continue

        grouped: dict[AttributeValue, list[tuple[str, AttributePatchCandidate]]] = {}
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
        winner_value, winner_rows = ranked[0]
        runner_up = len(ranked[1][1]) if len(ranked) > 1 else 0
        winner_count = len(winner_rows)
        template_count = len(grouped.get(template_value, ())) if template_value is not None else 0
        total_family_count = len(families) + len(incomplete_families)
        if winner_value == template_value:
            decisions.append(_decision(kind, winner_value, threshold, "no_change"))
            continue
        if not (
            winner_count >= DEFAULT_POLICY.attribute_patch_minimum_supporting_families
            and winner_count > total_family_count / 2
            and winner_count - runner_up >= 1
        ):
            decisions.append(_decision(kind, winner_value, threshold, "no_strict_attribute_family_majority"))
            continue

        support = [candidate for _, candidate in winner_rows]
        support_explicit = sum(_has_explicit_attribute(candidate.measure, kind) for candidate in support)
        support_boundary = sum(
            candidate.previous_semantics is not None
            and _attribute_value(candidate.previous_semantics, kind) != winner_value
            for candidate in support
        )
        template_explicit = _has_explicit_attribute(template.measure, kind)
        if not (is_first_measure or template_explicit or support_explicit or support_boundary):
            decisions.append(_decision(kind, winner_value, threshold, "missing_attribute_boundary_evidence"))
            continue

        template_meter_error = 0.0
        winner_meter_error = 0.0
        meter_improvement = 0.0
        if kind == "time":
            template_meter_error = _meter_error(base_semantics, template_value)
            winner_meter_error = _meter_error(base_semantics, winner_value)
            meter_improvement = template_meter_error - winner_meter_error
            # Interior measures must exactly fit the proposed meter.  First/last measures
            # may be pickups or shortened endings, but the proposal must still improve the
            # mismatch and may never make it worse.
            if winner_meter_error > template_meter_error + 1e-9:
                decisions.append(_decision(kind, winner_value, threshold, "meter_fit_worsened"))
                continue
            if not (is_first_measure or is_last_measure) and winner_meter_error > 1e-9:
                decisions.append(_decision(kind, winner_value, threshold, "interior_measure_not_meter_complete"))
                continue
            if (is_first_measure or is_last_measure) and winner_meter_error > 1e-9 and meter_improvement <= 1e-9:
                decisions.append(_decision(kind, winner_value, threshold, "edge_meter_not_improved"))
                continue

        input_row = AttributePatchInput(
            candidate_count=len(candidates),
            eligible_family_count=total_family_count,
            voting_family_count=len(family_votes),
            winner_family_support_ratio=winner_count / max(total_family_count, 1),
            winner_family_margin_ratio=(winner_count - runner_up) / max(total_family_count, 1),
            template_family_support_ratio=template_count / max(total_family_count, 1),
            family_abstention_ratio=(total_family_count - len(family_votes)) / max(total_family_count, 1),
            attribute_kind=kind,
            is_first_measure=is_first_measure,
            is_last_measure=is_last_measure,
            template_has_explicit_attribute=template_explicit,
            support_explicit_attribute_ratio=support_explicit / max(len(support), 1),
            support_previous_continuity_ratio=_continuity(
                support,
                kind,
                winner_value,
                previous=True,
                edge_default=1.0 if is_first_measure else 0.5,
            ),
            support_following_continuity_ratio=_continuity(
                support,
                kind,
                winner_value,
                previous=False,
                edge_default=1.0 if is_last_measure else 0.5,
            ),
            template_previous_continuity=(
                1.0
                if template.previous_semantics is None and is_first_measure
                else float(_attribute_value(template.previous_semantics, kind) == template_value)
            ),
            template_following_continuity=(
                1.0
                if template.following_semantics is None and is_last_measure
                else float(_attribute_value(template.following_semantics, kind) == template_value)
            ),
            template_meter_error=template_meter_error,
            winner_meter_error=winner_meter_error,
            meter_error_improvement=meter_improvement,
            mean_support_page_probability=_mean([row.page_probability for row in support]),
            mean_support_measure_probability=_mean([row.measure_probability for row in support]),
            mean_support_visual_probability=_mean([row.visual_probability for row in support]),
            mean_support_event_probability=_mean([row.event_probability for row in support]),
            mean_support_context_probability=_mean([row.context_probability for row in support]),
            mean_support_ensemble_probability=_mean([row.ensemble_probability for row in support]),
            minimum_support_ensemble_probability=min(row.ensemble_probability for row in support),
            mean_support_page_score_margin=_mean([row.page_score - template.page_score for row in support], 0.0),
            mean_support_vs_template_ensemble_probability=_mean([
                row.ensemble_probability - template.ensemble_probability for row in support
            ], 0.0),
        )
        calibration = active_calibrator.calibrate(input_row)
        decisions.append(
            _decision(
                kind,
                winner_value,
                calibration.threshold,
                "accepted" if calibration.accepted else "model_guard",
                probability=calibration.probability,
                accepted=calibration.accepted,
                item=input_row,
            )
        )
        if calibration.accepted:
            accepted_values[kind] = winner_value
            _write_attribute(working, kind, winner_value)

    if not accepted_values:
        probabilities = [decision.probability for decision in decisions if decision.input is not None]
        reason = "model_guard" if any(decision.reason == "model_guard" for decision in decisions) else "no_attribute_change"
        return AttributePatchResult(
            None,
            (),
            round(_mean(probabilities, 0.5), 6),
            threshold,
            False,
            reason,
            tuple(decisions),
            active_calibrator.model_version,
        )

    parsed, _ = measure_from_xml(
        working,
        {
            "divisions": template.semantics.divisions,
            "time": template.semantics.time_signature,
            "key": template.semantics.key_signature,
            "clef": template.semantics.clef,
        },
    )
    if _non_attribute_identity(parsed) != base_identity:
        return AttributePatchResult(
            None,
            (),
            round(_mean([decision.probability for decision in decisions if decision.accepted], 0.5), 6),
            threshold,
            False,
            "post_patch_non_attribute_change",
            tuple(decisions),
            active_calibrator.model_version,
        )
    for kind, value in accepted_values.items():
        if _attribute_value(parsed, kind) != value:
            return AttributePatchResult(
                None,
                (),
                round(_mean([decision.probability for decision in decisions if decision.accepted], 0.5), 6),
                threshold,
                False,
                "post_patch_attribute_mismatch",
                tuple(decisions),
                active_calibrator.model_version,
            )
        if kind == "time" and not (is_first_measure or is_last_measure) and _meter_error(parsed, value) > 1e-9:
            return AttributePatchResult(
                None,
                (),
                round(_mean([decision.probability for decision in decisions if decision.accepted], 0.5), 6),
                threshold,
                False,
                "post_patch_meter_validation_failed",
                tuple(decisions),
                active_calibrator.model_version,
            )

    accepted_decisions = [decision for decision in decisions if decision.accepted]
    return AttributePatchResult(
        working,
        tuple(kind for kind in ATTRIBUTE_KINDS if kind in accepted_values),
        round(_mean([decision.probability for decision in accepted_decisions], 0.5), 6),
        threshold,
        True,
        "accepted",
        tuple(decisions),
        active_calibrator.model_version,
    )
