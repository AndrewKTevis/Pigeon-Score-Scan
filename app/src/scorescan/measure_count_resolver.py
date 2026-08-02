from __future__ import annotations

"""Conservative fusion of layout and OMR measure-count evidence.

The page layout detector and OMR variants fail in different ways. Treating either
count as authoritative can create a cascade: a wrong expected count penalises the
correct candidate, distorts visual measure regions and weakens consensus alignment.

This module scores only counts already observed in the layout or OMR candidates. It
cannot invent measures or edit MusicXML. Correlated preprocessing variants are first
collapsed into equal-weight family evidence so one treatment family cannot manufacture
a majority by containing more variants. A verified CPU model may select among observed
counts when its probability and margin pass versioned safety gates; otherwise a
correlation-aware deterministic mode is used.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Protocol

from .linear_model import StandardizedLogisticModel
from .model_registry import load_verified_json
from .policy import DEFAULT_POLICY
from .tree_model import VerifiedRandomForestModel
from .variant_family import variant_family


LEGACY_FEATURE_NAMES = (
    "count_scaled",
    "layout_indicator",
    "layout_confidence",
    "layout_distance_ratio",
    "support_share",
    "valid_support_share",
    "family_support_share",
    "mean_agreement",
    "mean_page_probability",
    "mean_score_scaled",
    "top_support_margin",
    "count_dispersion",
    "distinct_count_scaled",
    "candidate_total_scaled",
    "median_distance_ratio",
    "mode_indicator",
)

FEATURE_NAMES = LEGACY_FEATURE_NAMES + (
    "family_balanced_support_share",
    "family_balanced_margin",
    "family_mode_indicator",
    "valid_family_support_share",
    "duplicate_support_ratio",
    "quality_std",
    "adjacent_family_support_share",
    "layout_family_support_gap",
    "complete_family_share",
    "incomplete_family_share",
)


class CountCandidate(Protocol):
    variant: str
    measure_count: int
    valid: bool
    agreement_ratio: float
    calibrated_probability: float
    raw_score: float
    measure_gap_penalty: float


@dataclass(frozen=True)
class MeasureCountOption:
    count: int
    probability: float
    support_share: float
    family_balanced_support_share: float
    family_support: int
    candidate_support: int
    deterministic_score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "probability": round(self.probability, 6),
            "support_share": round(self.support_share, 6),
            "family_balanced_support_share": round(self.family_balanced_support_share, 6),
            "family_support": self.family_support,
            "candidate_support": self.candidate_support,
            "deterministic_score": round(self.deterministic_score, 6),
        }


@dataclass(frozen=True)
class MeasureCountResolution:
    selected_count: int
    probability: float
    margin: float
    source: str
    model_version: str
    model_status: str
    layout_count: int
    layout_confidence: float
    deterministic_count: int
    options: tuple[MeasureCountOption, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_count": self.selected_count,
            "probability": round(self.probability, 6),
            "margin": round(self.margin, 6),
            "source": self.source,
            "model_version": self.model_version,
            "model_status": self.model_status,
            "layout_count": self.layout_count,
            "layout_confidence": round(self.layout_confidence, 6),
            "deterministic_count": self.deterministic_count,
            "options": [item.to_dict() for item in self.options],
        }


@dataclass(frozen=True)
class MeasureCountFeatureRow:
    count: int
    features: tuple[float, ...]
    support_share: float
    family_balanced_support_share: float
    family_support: int
    candidate_support: int
    deterministic_score: float


@dataclass(frozen=True)
class MeasureCountFeatureBundle:
    rows: tuple[MeasureCountFeatureRow, ...]
    deterministic_count: int


def measure_count_model_gate(
    *,
    count: int,
    probability: float,
    margin: float,
    family_support: int,
    candidate_support: int,
    deterministic_count: int,
    layout_count: int,
    layout_confidence: float,
    probability_floor: float | None = None,
    margin_floor: float | None = None,
) -> bool:
    """Return whether a learned count decision may affect runtime output.

    The model can rank only observed options.  Changing the deterministic answer is
    more dangerous than confirming it, so an override also needs independent-family
    evidence.  A high-confidence layout which already agrees with the deterministic
    result receives an even stricter four-family guard.  Layout-only rescue remains
    possible, but only for a very confident geometric count.
    """
    probability_floor = (
        DEFAULT_POLICY.measure_count_probability_floor
        if probability_floor is None
        else float(probability_floor)
    )
    margin_floor = (
        DEFAULT_POLICY.measure_count_margin_floor
        if margin_floor is None
        else float(margin_floor)
    )
    if probability < probability_floor or margin < margin_floor:
        return False
    if count == deterministic_count:
        return True
    if candidate_support <= 0:
        return (
            count == layout_count
            and layout_confidence >= DEFAULT_POLICY.measure_count_layout_only_confidence_floor
        )
    minimum_families = DEFAULT_POLICY.measure_count_override_min_families
    if (
        layout_confidence >= DEFAULT_POLICY.measure_count_high_confidence_layout_floor
        and deterministic_count == layout_count
        and count != layout_count
    ):
        minimum_families = DEFAULT_POLICY.measure_count_high_confidence_override_min_families
    return family_support >= minimum_families


@dataclass(frozen=True)
class _Aggregate:
    count: int
    support: float
    family_balanced_support: float
    valid_share: float
    valid_family_share: float
    family_share: float
    mean_agreement: float
    mean_probability: float
    mean_score: float
    quality_std: float
    duplicate_ratio: float
    candidate_support: int
    family_support: int


def _finite(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _candidate_quality(item: CountCandidate) -> float:
    # Remove the circular penalty caused by the provisional layout count. Count
    # resolution must judge OMR evidence before that expectation is reapplied.
    base_score = _finite(item.raw_score) + _finite(getattr(item, "measure_gap_penalty", 0.0))
    score_term = max(0.08, min(1.35, (base_score + 120.0) / 1120.0))
    validity = 1.0 if bool(item.valid) else 0.30
    agreement = 0.45 + max(0.0, min(1.0, _finite(item.agreement_ratio)))
    probability = 0.50 + max(0.0, min(1.0, _finite(item.calibrated_probability)))
    return validity * score_term * agreement * probability


def _aggregate(
    candidates: tuple[CountCandidate, ...],
) -> tuple[dict[int, _Aggregate], float, float, int, int, int]:
    usable = tuple(item for item in candidates if int(item.measure_count) > 0)
    qualities = {id(item): _candidate_quality(item) for item in usable}
    total_support = sum(qualities.values()) or 1.0
    counts = [int(item.measure_count) for item in usable]
    if counts:
        mean_count = sum(counts) / len(counts)
        dispersion = (
            sum((value - mean_count) ** 2 for value in counts) / len(counts)
        ) ** 0.5 / max(mean_count, 1.0)
    else:
        dispersion = 0.0

    family_members: dict[str, tuple[CountCandidate, ...]] = {}
    for family in sorted({variant_family(item.variant) for item in usable}):
        family_members[family] = tuple(
            item for item in usable if variant_family(item.variant) == family
        )
    observed_family_names = tuple(family_members)
    incomplete_family_names = frozenset(
        family
        for family, members in family_members.items()
        if not members or any(not bool(item.valid) for item in members)
    )
    # A preprocessing family is one correlated source.  If one observed sibling is
    # invalid, the complete family abstains from family-balanced count voting.  Keeping
    # the remaining valid sibling would turn a partial failure into false independent
    # evidence.  Invalid candidates still contribute weak raw evidence below, so this
    # is a conservative abstention rather than silent deletion of diagnostics.
    complete_family_names = tuple(
        family for family in observed_family_names if family not in incomplete_family_names
    )
    family_votes: dict[int, float] = {}
    for family in complete_family_names:
        members = family_members[family]
        by_count: dict[int, float] = {}
        # Within one family, repeated variants are correlated. Keep the strongest
        # evidence for each count, then normalise the family to one total vote.
        for item in members:
            count = int(item.measure_count)
            by_count[count] = max(by_count.get(count, 0.0), qualities[id(item)])
        family_total = sum(by_count.values()) or 1.0
        for count, support in by_count.items():
            family_votes[count] = family_votes.get(count, 0.0) + support / family_total
    family_denominator = max(len(complete_family_names), 1)
    family_votes = {
        count: value / family_denominator
        for count, value in family_votes.items()
    }

    result: dict[int, _Aggregate] = {}
    for count in sorted(set(counts)):
        members = tuple(item for item in usable if int(item.measure_count) == count)
        member_qualities = tuple(qualities[id(item)] for item in members)
        member_families = {
            variant_family(item.variant)
            for item in members
            if variant_family(item.variant) in complete_family_names
        }
        valid_families = {
            variant_family(item.variant)
            for item in members
            if bool(item.valid) and variant_family(item.variant) in complete_family_names
        }
        mean_quality = sum(member_qualities) / max(len(member_qualities), 1)
        quality_std = (
            sum((value - mean_quality) ** 2 for value in member_qualities)
            / max(len(member_qualities), 1)
        ) ** 0.5
        result[count] = _Aggregate(
            count=count,
            support=sum(member_qualities),
            family_balanced_support=family_votes.get(count, 0.0),
            valid_share=sum(bool(item.valid) for item in members) / max(len(members), 1),
            valid_family_share=len(valid_families) / family_denominator,
            family_share=len(member_families) / family_denominator,
            mean_agreement=sum(_finite(item.agreement_ratio) for item in members) / max(len(members), 1),
            mean_probability=sum(_finite(item.calibrated_probability) for item in members) / max(len(members), 1),
            mean_score=sum(
                _finite(item.raw_score) + _finite(getattr(item, "measure_gap_penalty", 0.0))
                for item in members
            ) / max(len(members), 1),
            quality_std=quality_std,
            duplicate_ratio=max(0.0, 1.0 - len(member_families) / max(len(members), 1)),
            candidate_support=len(members),
            family_support=len(member_families),
        )
    return (
        result,
        total_support,
        dispersion,
        len(set(counts)),
        len(complete_family_names),
        len(incomplete_family_names),
    )


def build_measure_count_feature_bundle(
    *,
    layout_count: int,
    layout_confidence: float,
    candidates: tuple[CountCandidate, ...],
) -> MeasureCountFeatureBundle:
    """Build immutable model rows shared by runtime and CPU training tools."""
    layout_count = max(0, int(layout_count))
    layout_confidence = max(0.0, min(1.0, float(layout_confidence)))
    (
        aggregates,
        total_support,
        dispersion,
        distinct_count,
        complete_family_total,
        incomplete_family_total,
    ) = _aggregate(candidates)
    option_counts = sorted(set(aggregates) | ({layout_count} if layout_count > 0 else set()))
    if not option_counts:
        return MeasureCountFeatureBundle((), max(layout_count, 1))

    raw_support_values = sorted((item.support for item in aggregates.values()), reverse=True)
    raw_margin = (
        (raw_support_values[0] - raw_support_values[1]) / total_support
        if len(raw_support_values) > 1
        else (raw_support_values[0] / total_support if raw_support_values else 0.0)
    )
    family_support_values = sorted(
        (item.family_balanced_support for item in aggregates.values()), reverse=True
    )
    family_margin = (
        family_support_values[0] - family_support_values[1]
        if len(family_support_values) > 1
        else (family_support_values[0] if family_support_values else 0.0)
    )
    raw_mode_count = max(aggregates, key=lambda count: aggregates[count].support) if aggregates else layout_count
    family_mode_count = (
        max(aggregates, key=lambda count: aggregates[count].family_balanced_support)
        if aggregates
        else layout_count
    )

    deterministic_scores: dict[int, float] = {}
    for count in option_counts:
        item = aggregates.get(count)
        raw_share = item.support / total_support if item else 0.0
        family_share = item.family_balanced_support if item else 0.0
        layout_prior = layout_confidence * (
            1.0
            if count == layout_count
            else max(0.0, 0.35 - abs(count - layout_count) / max(layout_count, 1))
        )
        deterministic_scores[count] = 0.35 * raw_share + 0.65 * family_share + 0.42 * layout_prior
    deterministic_count = max(
        option_counts,
        key=lambda count: (
            deterministic_scores[count],
            -abs(count - layout_count),
            -count,
        ),
    )

    observed_counts = [int(item.measure_count) for item in candidates if int(item.measure_count) > 0]
    observed_median = float(median(observed_counts)) if observed_counts else float(layout_count or deterministic_count)
    total_candidates = max(len(observed_counts), 1)
    layout_family_support = (
        aggregates[layout_count].family_balanced_support
        if layout_count in aggregates
        else 0.0
    )
    rows: list[MeasureCountFeatureRow] = []
    for count in option_counts:
        item = aggregates.get(count)
        raw_share = item.support / total_support if item else 0.0
        family_share = item.family_balanced_support if item else 0.0
        adjacent_family_support = min(
            1.0,
            sum(
                aggregates[nearby].family_balanced_support
                for nearby in (count - 1, count + 1)
                if nearby in aggregates
            ),
        )
        legacy = (
            min(float(count) / 120.0, 2.0),
            float(count == layout_count),
            layout_confidence,
            min(abs(count - layout_count) / max(layout_count, 1), 1.0) if layout_count else 0.0,
            raw_share,
            item.valid_share if item else 0.0,
            item.family_share if item else 0.0,
            max(0.0, min(1.0, item.mean_agreement)) if item else 0.0,
            max(0.0, min(1.0, item.mean_probability)) if item else 0.0,
            max(-1.0, min(1.0, item.mean_score / 1000.0)) if item else 0.0,
            raw_margin if count == raw_mode_count else -raw_margin,
            min(dispersion, 1.0),
            min(distinct_count / 8.0, 1.0),
            min(total_candidates / 8.0, 1.0),
            min(abs(count - observed_median) / max(observed_median, 1.0), 1.0),
            float(count == raw_mode_count),
        )
        extra = (
            family_share,
            family_margin if count == family_mode_count else -family_margin,
            float(count == family_mode_count),
            item.valid_family_share if item else 0.0,
            item.duplicate_ratio if item else 0.0,
            min((item.quality_std if item else 0.0) / 1.5, 1.0),
            adjacent_family_support,
            max(-1.0, min(1.0, family_share - layout_family_support)),
            complete_family_total
            / max(complete_family_total + incomplete_family_total, 1),
            incomplete_family_total
            / max(complete_family_total + incomplete_family_total, 1),
        )
        features = legacy + extra
        features = tuple(value if math.isfinite(value) else 0.0 for value in features)
        if len(features) != len(FEATURE_NAMES):
            raise ValueError("invalid measure-count feature schema")
        rows.append(
            MeasureCountFeatureRow(
                count=count,
                features=features,
                support_share=raw_share,
                family_balanced_support_share=family_share,
                family_support=item.family_support if item else 0,
                candidate_support=item.candidate_support if item else 0,
                deterministic_score=deterministic_scores[count],
            )
        )
    return MeasureCountFeatureBundle(tuple(rows), deterministic_count)


class MeasureCountResolver:
    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).resolve().parent / "resources" / "measure_count_resolver.json"
        loaded = load_verified_json(model_path, "measure_count_resolution")
        payload = loaded.payload
        self.model_type = str(payload.get("model_type", ""))
        self.logistic = StandardizedLogisticModel.from_payload(
            payload,
            FEATURE_NAMES if self.model_type == "standardized_logistic" else LEGACY_FEATURE_NAMES,
            verified=loaded.verified,
            status=loaded.status,
        )
        self.forest = VerifiedRandomForestModel.load(
            model_path,
            "measure_count_resolution",
            FEATURE_NAMES,
            loaded=loaded,
        )
        self._model_version = str(payload.get("model_version", "disabled"))
        self._model_status = loaded.status

    @property
    def enabled(self) -> bool:
        return (
            (self.model_type == "random_forest" and self.forest.enabled)
            or (self.model_type == "standardized_logistic" and self.logistic.enabled)
        )

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def model_status(self) -> str:
        return self._model_status

    def _predict(self, features: tuple[float, ...]) -> float:
        if self.model_type == "random_forest":
            return self.forest.predict(features)
        if self.model_type == "standardized_logistic":
            return self.logistic.predict(features[: len(LEGACY_FEATURE_NAMES)])
        return 0.5

    def resolve(
        self,
        *,
        layout_count: int,
        layout_confidence: float,
        candidates: tuple[CountCandidate, ...],
    ) -> MeasureCountResolution:
        layout_count = max(0, int(layout_count))
        layout_confidence = max(0.0, min(1.0, float(layout_confidence)))
        bundle = build_measure_count_feature_bundle(
            layout_count=layout_count,
            layout_confidence=layout_confidence,
            candidates=candidates,
        )
        if not bundle.rows:
            selected = max(layout_count, 1)
            return MeasureCountResolution(
                selected_count=selected,
                probability=0.5,
                margin=0.0,
                source="layout_only",
                model_version=self.model_version,
                model_status=self.model_status,
                layout_count=layout_count,
                layout_confidence=layout_confidence,
                deterministic_count=selected,
                options=(),
            )

        scored = tuple(
            MeasureCountOption(
                count=row.count,
                probability=self._predict(row.features),
                support_share=row.support_share,
                family_balanced_support_share=row.family_balanced_support_share,
                family_support=row.family_support,
                candidate_support=row.candidate_support,
                deterministic_score=row.deterministic_score,
            )
            for row in bundle.rows
        )
        ranked = sorted(
            scored,
            key=lambda item: (
                item.probability,
                item.deterministic_score,
                -abs(item.count - layout_count),
            ),
            reverse=True,
        )
        best = ranked[0]
        second = ranked[1].probability if len(ranked) > 1 else 0.0
        margin = max(0.0, best.probability - second)
        use_model = self.enabled and measure_count_model_gate(
            count=best.count,
            probability=best.probability,
            margin=margin,
            family_support=best.family_support,
            candidate_support=best.candidate_support,
            deterministic_count=bundle.deterministic_count,
            layout_count=layout_count,
            layout_confidence=layout_confidence,
        )
        selected_count = best.count if use_model else bundle.deterministic_count
        source = "model" if use_model else "deterministic_fallback"
        selected_option = next(item for item in scored if item.count == selected_count)
        return MeasureCountResolution(
            selected_count=selected_count,
            probability=selected_option.probability,
            margin=margin if use_model else 0.0,
            source=source,
            model_version=self.model_version,
            model_status=self.model_status,
            layout_count=layout_count,
            layout_confidence=layout_confidence,
            deterministic_count=bundle.deterministic_count,
            options=tuple(sorted(scored, key=lambda item: item.count)),
        )
