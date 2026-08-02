from __future__ import annotations

"""Fail-closed validation for composed local MusicXML repairs.

Individual repair modules are deliberately narrow.  This module owns the boundary
between those modules and the committed measure: deterministic semantic auditing is
always authoritative, while a compact verified CPU model may only veto combinations
of several interacting repairs which remain structurally plausible.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from lxml import etree

from .model_registry import load_verified_json
from .policy import DEFAULT_POLICY
from .score_ir import ScoreIR, audit_score, measure_from_xml
from .tree_model import VerifiedRandomForestModel

PATCH_KINDS = (
    "chord",
    "tuplet",
    "pitch",
    "rhythm",
    "event_kind",
    "attribute",
    "tie",
    "slur",
    "articulation",
    "ornament",
    "grace",
    "lyric",
    "direction",
    "barline",
    "event_presence",
)

SEMANTIC_PATCH_KINDS = frozenset(
    {
        "chord",
        "tuplet",
        "pitch",
        "rhythm",
        "event_kind",
        "attribute",
        "tie",
        "slur",
        "grace",
        "event_presence",
    }
)
HIGH_RISK_PATCH_KINDS = frozenset(
    {
        "chord",
        "tuplet",
        "pitch",
        "rhythm",
        "event_kind",
        "attribute",
        "grace",
        "event_presence",
    }
)

FEATURE_NAMES = (
    "patch_count_scaled",
    "semantic_patch_count_scaled",
    "high_risk_patch_count_scaled",
    "decorative_patch_count_scaled",
    "changed_event_count_scaled",
    "changed_surface_count_scaled",
    "minimum_patch_probability",
    "mean_patch_probability",
    "minimum_patch_margin",
    "mean_patch_margin",
    "maximum_patch_threshold",
    "eligible_family_count_scaled",
    "exact_family_support_ratio",
    "semantic_family_support_ratio",
    "missing_ratio",
    "selected_measure_probability",
    "selected_visual_probability",
    "selected_event_probability",
    "selected_context_probability",
    "selected_ensemble_probability",
    "semantic_confidence",
    "mean_cluster_distance",
    "template_distance",
    "chord_pitch_interaction",
    "chord_rhythm_interaction",
    "pitch_rhythm_interaction",
    "event_kind_pitch_interaction",
    "event_kind_rhythm_interaction",
    "attribute_rhythm_interaction",
    "tie_slur_interaction",
    "event_presence_with_other",
    "grace_with_other",
    "direction_with_attribute_or_barline",
)

_PATCH_GUARD_ISSUES = frozenset(
    {
        "multiple_voices",
        "empty_measure",
        "zero_duration",
        "type_duration_mismatch",
        "pitch_outlier",
        "chord_duration_mismatch",
        "density_outlier",
        "duplicate_direction",
    }
)


@dataclass(frozen=True)
class PatchEvidence:
    kind: str
    probability: float
    threshold: float
    changed_events: int = 0
    changed_surfaces: int = 0

    def __post_init__(self) -> None:
        if self.kind not in PATCH_KINDS:
            raise ValueError(f"unknown patch kind: {self.kind}")


@dataclass(frozen=True)
class PatchTransactionInput:
    patch_kinds: tuple[str, ...]
    changed_event_count: int
    changed_surface_count: int
    minimum_patch_probability: float
    mean_patch_probability: float
    minimum_patch_margin: float
    mean_patch_margin: float
    maximum_patch_threshold: float
    eligible_family_count: int
    exact_family_support_ratio: float
    semantic_family_support_ratio: float
    missing_ratio: float
    selected_measure_probability: float
    selected_visual_probability: float
    selected_event_probability: float
    selected_context_probability: float
    selected_ensemble_probability: float
    semantic_confidence: float
    mean_cluster_distance: float
    template_distance: float

    @classmethod
    def from_evidence(
        cls,
        evidence: Iterable[PatchEvidence],
        *,
        eligible_family_count: int,
        exact_family_support_ratio: float,
        semantic_family_support_ratio: float,
        missing_ratio: float,
        selected_measure_probability: float,
        selected_visual_probability: float,
        selected_event_probability: float,
        selected_context_probability: float,
        selected_ensemble_probability: float,
        semantic_confidence: float,
        mean_cluster_distance: float,
        template_distance: float,
    ) -> "PatchTransactionInput":
        rows = tuple(evidence)
        probabilities = tuple(max(0.0, min(1.0, float(row.probability))) for row in rows)
        thresholds = tuple(max(0.0, min(1.0, float(row.threshold))) for row in rows)
        margins = tuple(probability - threshold for probability, threshold in zip(probabilities, thresholds, strict=True))
        return cls(
            patch_kinds=tuple(row.kind for row in rows),
            changed_event_count=sum(max(0, int(row.changed_events)) for row in rows),
            changed_surface_count=sum(max(0, int(row.changed_surfaces)) for row in rows),
            minimum_patch_probability=min(probabilities, default=0.5),
            mean_patch_probability=sum(probabilities) / max(len(probabilities), 1),
            minimum_patch_margin=min(margins, default=0.0),
            mean_patch_margin=sum(margins) / max(len(margins), 1),
            maximum_patch_threshold=max(thresholds, default=1.0),
            eligible_family_count=max(0, int(eligible_family_count)),
            exact_family_support_ratio=float(exact_family_support_ratio),
            semantic_family_support_ratio=float(semantic_family_support_ratio),
            missing_ratio=float(missing_ratio),
            selected_measure_probability=float(selected_measure_probability),
            selected_visual_probability=float(selected_visual_probability),
            selected_event_probability=float(selected_event_probability),
            selected_context_probability=float(selected_context_probability),
            selected_ensemble_probability=float(selected_ensemble_probability),
            semantic_confidence=float(semantic_confidence),
            mean_cluster_distance=float(mean_cluster_distance),
            template_distance=float(template_distance),
        )

    @property
    def patch_count(self) -> int:
        return len(self.patch_kinds)

    @property
    def semantic_patch_count(self) -> int:
        return sum(kind in SEMANTIC_PATCH_KINDS for kind in self.patch_kinds)

    @property
    def high_risk_patch_count(self) -> int:
        return sum(kind in HIGH_RISK_PATCH_KINDS for kind in self.patch_kinds)

    @property
    def decorative_patch_count(self) -> int:
        return self.patch_count - self.semantic_patch_count

    def requires_model(self) -> bool:
        return bool(
            self.semantic_patch_count
            >= DEFAULT_POLICY.patch_transaction_model_minimum_semantic_patches
            or self.patch_count
            >= DEFAULT_POLICY.patch_transaction_model_minimum_total_patches
            or (
                "event_presence" in self.patch_kinds
                and self.patch_count > 1
            )
        )

    def feature_vector(self) -> list[float]:
        kinds = set(self.patch_kinds)

        def unit(value: float) -> float:
            return max(0.0, min(1.0, float(value)))

        def signed(value: float) -> float:
            return max(-1.0, min(1.0, float(value)))

        return [
            unit(self.patch_count / 8.0),
            unit(self.semantic_patch_count / 6.0),
            unit(self.high_risk_patch_count / 5.0),
            unit(self.decorative_patch_count / 5.0),
            unit(self.changed_event_count / 24.0),
            unit(self.changed_surface_count / 16.0),
            unit(self.minimum_patch_probability),
            unit(self.mean_patch_probability),
            signed(self.minimum_patch_margin / 0.20),
            signed(self.mean_patch_margin / 0.20),
            unit(self.maximum_patch_threshold),
            unit(self.eligible_family_count / 5.0),
            unit(self.exact_family_support_ratio),
            unit(self.semantic_family_support_ratio),
            unit(self.missing_ratio),
            unit(self.selected_measure_probability),
            unit(self.selected_visual_probability),
            unit(self.selected_event_probability),
            unit(self.selected_context_probability),
            unit(self.selected_ensemble_probability),
            unit(self.semantic_confidence),
            unit(self.mean_cluster_distance / max(DEFAULT_POLICY.semantic_distance_max, 1e-9)),
            unit(self.template_distance / max(DEFAULT_POLICY.unresolved_measure_distance, 1e-9)),
            float({"chord", "pitch"} <= kinds),
            float({"chord", "rhythm"} <= kinds),
            float({"pitch", "rhythm"} <= kinds),
            float({"event_kind", "pitch"} <= kinds),
            float({"event_kind", "rhythm"} <= kinds),
            float({"attribute", "rhythm"} <= kinds),
            float({"tie", "slur"} <= kinds),
            float("event_presence" in kinds and len(kinds) > 1),
            float("grace" in kinds and len(kinds) > 1),
            float("direction" in kinds and bool({"attribute", "barline"} & kinds)),
        ]


@dataclass(frozen=True)
class PatchTransactionCalibration:
    probability: float
    threshold: float
    accepted: bool
    model_version: str
    target_precision: float
    applicable: bool
    reason: str


class PatchTransactionCalibrator:
    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).with_name("resources") / "patch_transaction_calibrator.json"
        loaded = load_verified_json(model_path, "patch_transaction_calibration")
        payload = loaded.payload
        self.model = VerifiedRandomForestModel.load(
            model_path,
            "patch_transaction_calibration",
            FEATURE_NAMES,
            loaded=loaded,
        )
        try:
            stored_threshold = float(payload.get("auto_commit_threshold", 1.0))
            target_precision = float(payload.get("target_precision", 1.0))
        except (TypeError, ValueError, OverflowError):
            stored_threshold = 1.0
            target_precision = 1.0
        self.threshold = max(
            float(DEFAULT_POLICY.patch_transaction_probability_floor),
            max(0.0, min(1.0, stored_threshold)),
        )
        self.target_precision = max(0.0, min(1.0, target_precision))
        self.model_verified = self.model.verified and loaded.verified
        self.model_status = self.model.status if self.model.enabled else loaded.status
        self.model_version = self.model.model_version
        self.enabled = self.model.enabled

    def predict_probability(self, item: PatchTransactionInput) -> float:
        return self.model.predict(item.feature_vector(), neutral=0.5)

    def calibrate(self, item: PatchTransactionInput) -> PatchTransactionCalibration:
        if not item.requires_model():
            return PatchTransactionCalibration(
                probability=1.0,
                threshold=round(self.threshold, 6),
                accepted=True,
                model_version=self.model_version,
                target_precision=round(self.target_precision, 6),
                applicable=False,
                reason="deterministic_only",
            )
        probability = self.predict_probability(item)
        if not self.enabled or not self.model_verified:
            return PatchTransactionCalibration(
                probability=round(probability, 6),
                threshold=round(self.threshold, 6),
                accepted=False,
                model_version=self.model_version,
                target_precision=round(self.target_precision, 6),
                applicable=True,
                reason="model_unavailable",
            )
        accepted = probability >= self.threshold
        return PatchTransactionCalibration(
            probability=round(probability, 6),
            threshold=round(self.threshold, 6),
            accepted=accepted,
            model_version=self.model_version,
            target_precision=round(self.target_precision, 6),
            applicable=True,
            reason="validated_model" if accepted else "model_veto",
        )



@dataclass(frozen=True)
class PatchStageValidation:
    measure: etree._Element | None
    accepted: bool
    reason: str


def validate_patch_stage(
    original: etree._Element,
    current: etree._Element | None,
    candidate: etree._Element | None,
    inherited: dict[str, object],
) -> PatchStageValidation:
    """Keep earlier safe repairs when one later stage introduces a new defect."""

    if candidate is None:
        return PatchStageValidation(current, False, "missing_candidate")
    accepted, reason = patch_transaction_guard(original, candidate, inherited)
    if not accepted:
        return PatchStageValidation(current, False, reason)
    return PatchStageValidation(candidate, True, "validated")

def patch_transaction_guard(
    original: etree._Element,
    patched: etree._Element,
    inherited: dict[str, object],
) -> tuple[bool, str]:
    """Reject a composed repair which introduces a new semantic audit issue."""

    try:
        original_ir, _ = measure_from_xml(original, dict(inherited))
        patched_ir, _ = measure_from_xml(patched, dict(inherited))
    except (TypeError, ValueError, OverflowError):
        return False, "parse_failed"
    before = Counter(
        issue.code
        for issue in audit_score(ScoreIR((original_ir,)))
        if issue.code in _PATCH_GUARD_ISSUES
    )
    after = Counter(
        issue.code
        for issue in audit_score(ScoreIR((patched_ir,)))
        if issue.code in _PATCH_GUARD_ISSUES
    )
    introduced = sorted(code for code in _PATCH_GUARD_ISSUES if after[code] > before[code])
    if introduced:
        return False, "introduced_" + "+".join(introduced)
    return True, "validated"
