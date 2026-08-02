from __future__ import annotations

"""Selective-risk gate for all automatic measure replacement decisions.

Ranking a measure candidate and deciding whether it is safer than the retained page
template are different tasks.  This module verifies both exact-majority and fuzzy
semantic-consensus proposals using evidence already computed by the deterministic
consensus pipeline.  The deployed schema accounts for preprocessing-family diversity,
while deterministic permission gates reject incomplete-family and runtime-invariant
violations before the learned verifier can approve a replacement.

The model cannot select a candidate, create consensus, modify notation, or bypass page,
alignment, visual, event, context, structural, rhythm, or XML guards.  It can only veto
an automatic replacement and route the unresolved measure to review.
"""

from dataclasses import dataclass
from pathlib import Path

from .model_registry import load_verified_json
from .policy import DEFAULT_POLICY
from .tree_model import VerifiedGradientBoostingModel, VerifiedRandomForestModel

FEATURE_NAMES = (
    "selection_exact_majority",
    "selection_semantic_consensus",
    "selection_template",
    "selected_page_score_scaled",
    "selected_page_probability",
    "selected_ensemble_probability",
    "ensemble_probability_margin",
    "selected_measure_probability",
    "measure_probability_margin",
    "selected_visual_probability",
    "visual_probability_margin",
    "selected_event_probability",
    "event_probability_margin",
    "selected_context_probability",
    "context_probability_margin",
    "exact_support_ratio",
    "semantic_support_ratio",
    "signature_support_ratio",
    "missing_ratio",
    "mean_cluster_distance",
    "template_distance",
    "alignment_similarity",
    "alignment_margin",
    "selected_distance_to_medoid",
    "selected_mean_peer_distance",
    "page_score_margin_scaled",
    "candidate_count_scaled",
    "exact_support_count_scaled",
    "distinct_signature_ratio",
    "top_signature_margin",
    "unanimous",
    "strict_majority",
    "selected_is_template",
    "selected_is_exact_signature",
    "selected_in_initial_cluster",
    "page_valid",
    "selected_vs_template_page_probability",
    "selected_vs_template_ensemble_probability",
    "selected_vs_template_measure_probability",
    "selected_vs_template_visual_probability",
    "selected_vs_template_event_probability",
    "selected_vs_template_context_probability",
    "selected_vs_template_alignment_similarity",
    "template_page_valid",
    "template_in_initial_cluster",
    "template_is_exact_signature",
    "eligible_family_count_scaled",
    "exact_family_support_ratio",
    "semantic_family_support_ratio",
    "selected_family_support_ratio",
    "candidate_family_redundancy",
)


@dataclass(frozen=True)
class SelectionRiskInput:
    selection_kind: str
    selected_page_score: float
    selected_page_probability: float
    selected_ensemble_probability: float
    ensemble_probability_margin: float
    selected_measure_probability: float
    measure_probability_margin: float
    selected_visual_probability: float
    visual_probability_margin: float
    selected_event_probability: float
    event_probability_margin: float
    selected_context_probability: float
    context_probability_margin: float
    exact_support_ratio: float
    semantic_support_ratio: float
    signature_support_ratio: float
    missing_ratio: float
    mean_cluster_distance: float
    template_distance: float
    alignment_similarity: float
    alignment_margin: float
    selected_distance_to_medoid: float
    selected_mean_peer_distance: float
    page_score_margin: float
    candidate_count: int
    exact_support_count: int
    distinct_signature_count: int
    top_signature_margin: float
    unanimous: bool
    strict_majority: bool
    selected_is_template: bool
    selected_is_exact_signature: bool
    selected_in_initial_cluster: bool
    page_valid: bool
    selected_vs_template_page_probability: float
    selected_vs_template_ensemble_probability: float
    selected_vs_template_measure_probability: float
    selected_vs_template_visual_probability: float
    selected_vs_template_event_probability: float
    selected_vs_template_context_probability: float
    selected_vs_template_alignment_similarity: float
    template_page_valid: bool
    template_in_initial_cluster: bool
    template_is_exact_signature: bool
    eligible_family_count: int = 1
    exact_family_support_count: int = 1
    semantic_family_support_count: int = 1

    def feature_vector(self) -> list[float]:
        kind = self.selection_kind
        family_count = max(1, int(self.eligible_family_count))
        exact_family_count = max(0, min(family_count, int(self.exact_family_support_count)))
        semantic_family_count = max(0, min(family_count, int(self.semantic_family_support_count)))
        selected_family_count = (
            exact_family_count if kind == "exact_majority" else semantic_family_count
        )
        return [
            float(kind == "exact_majority"),
            float(kind == "semantic_consensus"),
            float(kind == "template"),
            max(-2.0, min(2.0, self.selected_page_score / 1000.0)),
            max(0.0, min(1.0, self.selected_page_probability)),
            max(0.0, min(1.0, self.selected_ensemble_probability)),
            max(-1.0, min(1.0, self.ensemble_probability_margin)),
            max(0.0, min(1.0, self.selected_measure_probability)),
            max(-1.0, min(1.0, self.measure_probability_margin)),
            max(0.0, min(1.0, self.selected_visual_probability)),
            max(-1.0, min(1.0, self.visual_probability_margin)),
            max(0.0, min(1.0, self.selected_event_probability)),
            max(-1.0, min(1.0, self.event_probability_margin)),
            max(0.0, min(1.0, self.selected_context_probability)),
            max(-1.0, min(1.0, self.context_probability_margin)),
            max(0.0, min(1.0, self.exact_support_ratio)),
            max(0.0, min(1.0, self.semantic_support_ratio)),
            max(0.0, min(1.0, self.signature_support_ratio)),
            max(0.0, min(1.0, self.missing_ratio)),
            max(0.0, min(1.0, self.mean_cluster_distance)),
            max(0.0, min(1.0, self.template_distance)),
            max(0.0, min(1.0, self.alignment_similarity)),
            max(-1.0, min(1.0, self.alignment_margin)),
            max(0.0, min(1.0, self.selected_distance_to_medoid)),
            max(0.0, min(1.0, self.selected_mean_peer_distance)),
            max(-2.0, min(2.0, self.page_score_margin / 100.0)),
            min(1.0, max(0, self.candidate_count - 1) / 7.0),
            min(1.0, max(0, self.exact_support_count - 1) / 7.0),
            min(1.0, max(1, self.distinct_signature_count) / max(1, self.candidate_count)),
            max(0.0, min(1.0, self.top_signature_margin)),
            float(self.unanimous),
            float(self.strict_majority),
            float(self.selected_is_template),
            float(self.selected_is_exact_signature),
            float(self.selected_in_initial_cluster),
            float(self.page_valid),
            max(-1.0, min(1.0, self.selected_vs_template_page_probability)),
            max(-1.0, min(1.0, self.selected_vs_template_ensemble_probability)),
            max(-1.0, min(1.0, self.selected_vs_template_measure_probability)),
            max(-1.0, min(1.0, self.selected_vs_template_visual_probability)),
            max(-1.0, min(1.0, self.selected_vs_template_event_probability)),
            max(-1.0, min(1.0, self.selected_vs_template_context_probability)),
            max(-1.0, min(1.0, self.selected_vs_template_alignment_similarity)),
            float(self.template_page_valid),
            float(self.template_in_initial_cluster),
            float(self.template_is_exact_signature),
            min(1.0, family_count / 5.0),
            exact_family_count / family_count,
            semantic_family_count / family_count,
            selected_family_count / family_count,
            max(0.0, min(1.0, 1.0 - family_count / max(1, self.candidate_count))),
        ]


@dataclass(frozen=True)
class SelectionRiskResult:
    probability: float
    threshold: float
    accepted: bool
    model_version: str
    target_precision: float


def corroborated_exact_majority(item: SelectionRiskInput) -> bool:
    """Return whether deterministic evidence permits an exact replacement proposal."""
    family_count = max(1, int(item.eligible_family_count))
    exact_family_count = max(0, min(family_count, int(item.exact_family_support_count)))
    family_ratio = exact_family_count / family_count
    direct_evidence = (
        item.selected_vs_template_visual_probability,
        item.selected_vs_template_event_probability,
        item.selected_vs_template_context_probability,
    )
    nondegrading_direct_layers = sum(
        value >= DEFAULT_POLICY.selection_exact_direct_evidence_delta_floor
        for value in direct_evidence
    )
    return bool(
        item.selection_kind == "exact_majority"
        and item.strict_majority
        and item.page_valid
        and exact_family_count >= DEFAULT_POLICY.selection_exact_minimum_families
        and family_ratio >= DEFAULT_POLICY.selection_exact_family_support_min
        and item.exact_support_ratio >= DEFAULT_POLICY.selection_exact_candidate_support_min
        and item.missing_ratio <= DEFAULT_POLICY.selection_exact_missing_ratio_max
        and item.selected_vs_template_ensemble_probability
        >= DEFAULT_POLICY.selection_exact_ensemble_delta_floor
        and item.selected_vs_template_measure_probability
        >= DEFAULT_POLICY.selection_exact_measure_delta_floor
        and item.selected_vs_template_event_probability
        >= DEFAULT_POLICY.selection_exact_event_delta_floor
        and nondegrading_direct_layers
        >= DEFAULT_POLICY.selection_exact_minimum_nondegrading_direct_layers
    )


def corroborated_semantic_consensus(item: SelectionRiskInput) -> bool:
    """Return whether deterministic evidence permits a semantic replacement proposal.

    At least three complete families must support the semantic cluster.  Ensemble,
    measure and event evidence may not materially regress versus the retained template.
    A proposal is also rejected when both its page score and direct visual evidence are
    materially worse; either signal may remain noisy on its own.
    """
    family_count = max(1, int(item.eligible_family_count))
    semantic_family_count = max(
        0, min(family_count, int(item.semantic_family_support_count))
    )
    family_ratio = semantic_family_count / family_count
    page_or_visual_not_worse = bool(
        item.page_score_margin >= DEFAULT_POLICY.selection_semantic_page_score_delta_floor
        or item.selected_vs_template_visual_probability
        >= DEFAULT_POLICY.selection_semantic_visual_delta_floor
    )
    return bool(
        item.selection_kind == "semantic_consensus"
        and item.page_valid
        and semantic_family_count >= DEFAULT_POLICY.selection_semantic_minimum_families
        and family_ratio >= DEFAULT_POLICY.selection_semantic_family_support_min
        and item.semantic_support_ratio
        >= DEFAULT_POLICY.selection_semantic_candidate_support_min
        and item.missing_ratio <= DEFAULT_POLICY.selection_semantic_missing_ratio_max
        and item.mean_cluster_distance <= DEFAULT_POLICY.semantic_distance_max
        and item.selected_distance_to_medoid
        <= DEFAULT_POLICY.selection_semantic_medoid_distance_max
        and item.selected_vs_template_ensemble_probability
        >= DEFAULT_POLICY.selection_semantic_ensemble_delta_floor
        and item.selected_vs_template_measure_probability
        >= DEFAULT_POLICY.selection_semantic_measure_delta_floor
        and item.selected_vs_template_event_probability
        >= DEFAULT_POLICY.selection_semantic_event_delta_floor
        and item.selected_vs_template_context_probability
        >= DEFAULT_POLICY.selection_semantic_context_delta_floor
        and page_or_visual_not_worse
    )


class SelectionRiskCalibrator:
    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).resolve().parent / "resources" / "selection_risk.json"
        loaded = load_verified_json(model_path, "selection_risk_calibration")
        payload = loaded.payload
        if str(payload.get("model_type", "")) == "random_forest":
            self.model = VerifiedRandomForestModel.load(
                model_path,
                "selection_risk_calibration",
                FEATURE_NAMES,
                loaded=loaded,
            )
        else:
            self.model = VerifiedGradientBoostingModel.load(
                model_path,
                "selection_risk_calibration",
                FEATURE_NAMES,
                loaded=loaded,
            )
        raw_thresholds = payload.get("auto_replace_thresholds", {})
        thresholds = raw_thresholds if isinstance(raw_thresholds, dict) else {}
        try:
            stored_threshold = float(payload.get("auto_replace_threshold", 1.0))
            exact_threshold = float(thresholds.get("exact_majority", stored_threshold))
            semantic_threshold = float(thresholds.get("semantic_consensus", stored_threshold))
            target_precision = float(payload.get("target_precision", 1.0))
        except (TypeError, ValueError, OverflowError):
            stored_threshold = 1.0
            exact_threshold = 1.0
            semantic_threshold = 1.0
            target_precision = 1.0
        self.exact_majority_threshold = max(
            float(DEFAULT_POLICY.replacement_selection_risk_floor),
            max(0.0, min(1.0, exact_threshold)),
        )
        self.semantic_consensus_threshold = max(
            float(DEFAULT_POLICY.replacement_selection_risk_floor),
            max(0.0, min(1.0, semantic_threshold)),
        )
        self.threshold = min(self.exact_majority_threshold, self.semantic_consensus_threshold)
        self.target_precision = max(0.0, min(1.0, target_precision))
        self.model_verified = self.model.verified and loaded.verified
        self.model_status = self.model.status if self.model.enabled else loaded.status
        self.model_version = self.model.model_version
        self.enabled = self.model.enabled

    def predict_probability(self, item: SelectionRiskInput) -> float:
        return self.model.predict(item.feature_vector(), neutral=0.5)

    def calibrate(self, item: SelectionRiskInput) -> SelectionRiskResult:
        probability = self.predict_probability(item)
        threshold = (
            self.exact_majority_threshold
            if item.selection_kind == "exact_majority"
            else self.semantic_consensus_threshold
        )
        # A disabled or unverified gate is deliberately neutral-but-not-accepting.
        # Exact majorities which pass the stricter three-family deterministic gate may
        # use the global verified floor.  This preserves useful exact-majority coverage
        # without allowing the model to bypass family, event, measure, XML or rhythm
        # checks.  Fuzzy semantic consensus always retains its learned high threshold.
        if item.selection_kind == "exact_majority":
            deterministic_guard = corroborated_exact_majority(item)
            effective_threshold = (
                float(DEFAULT_POLICY.replacement_selection_risk_floor)
                if deterministic_guard
                else threshold
            )
        else:
            deterministic_guard = corroborated_semantic_consensus(item)
            effective_threshold = threshold
        accepted = bool(
            self.enabled
            and self.model_verified
            and deterministic_guard
            and probability >= effective_threshold
        )
        return SelectionRiskResult(
            probability=round(probability, 6),
            threshold=round(effective_threshold, 6),
            accepted=accepted,
            model_version=self.model_version,
            target_precision=round(self.target_precision, 6),
        )
