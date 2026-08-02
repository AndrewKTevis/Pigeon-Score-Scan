from __future__ import annotations

"""Bounded meta-calibration for measure ensemble candidates.

ScoreScan keeps page structure, local measure validity, visual compatibility, event
agreement and neighbour context as independent evidence layers.  A plain product of
those weights assumes independence and cannot explicitly represent conflict such as a
high page score paired with weak semantic support.  This module combines the already
computed evidence into one conservative prior.

The calibrator never edits notation, creates a semantic majority, or bypasses strict
MusicXML and rhythm validation.  Its multiplicative influence remains bounded by the
versioned :mod:`scorescan.policy`.
"""

import math
from dataclasses import dataclass
from pathlib import Path

from .linear_model import StandardizedLogisticModel, bounded_weight
from .model_registry import load_verified_json
from .policy import DEFAULT_POLICY
from .tree_model import VerifiedRandomForestModel

FEATURE_NAMES = (
    "page_score_scaled",
    "page_probability",
    "page_valid",
    "alignment_similarity",
    "alignment_margin",
    "exact_support_ratio",
    "semantic_support_ratio",
    "signature_support_ratio",
    "missing_ratio",
    "distance_to_template",
    "distance_to_medoid",
    "mean_peer_distance",
    "measure_probability",
    "visual_probability",
    "event_probability",
    "context_probability",
    "measure_probability_margin",
    "visual_probability_margin",
    "event_probability_margin",
    "context_probability_margin",
    "page_score_margin_scaled",
    "candidate_count_scaled",
    "initial_cluster_member",
    "exact_signature_member",
)


@dataclass(frozen=True)
class EnsembleCalibrationInput:
    page_score: float
    page_probability: float
    page_valid: bool
    alignment_similarity: float
    alignment_margin: float
    exact_support_ratio: float
    semantic_support_ratio: float
    signature_support_ratio: float
    missing_ratio: float
    distance_to_template: float
    distance_to_medoid: float
    mean_peer_distance: float
    measure_probability: float
    visual_probability: float
    event_probability: float
    context_probability: float
    measure_probability_margin: float
    visual_probability_margin: float
    event_probability_margin: float
    context_probability_margin: float
    page_score_margin: float
    candidate_count: int
    initial_cluster_member: bool
    exact_signature_member: bool

    def feature_vector(self) -> list[float]:
        return [
            max(-2.0, min(2.0, self.page_score / 1000.0)),
            max(0.0, min(1.0, self.page_probability)),
            float(self.page_valid),
            max(0.0, min(1.0, self.alignment_similarity)),
            max(-1.0, min(1.0, self.alignment_margin)),
            max(0.0, min(1.0, self.exact_support_ratio)),
            max(0.0, min(1.0, self.semantic_support_ratio)),
            max(0.0, min(1.0, self.signature_support_ratio)),
            max(0.0, min(1.0, self.missing_ratio)),
            max(0.0, min(1.0, self.distance_to_template)),
            max(0.0, min(1.0, self.distance_to_medoid)),
            max(0.0, min(1.0, self.mean_peer_distance)),
            max(0.0, min(1.0, self.measure_probability)),
            max(0.0, min(1.0, self.visual_probability)),
            max(0.0, min(1.0, self.event_probability)),
            max(0.0, min(1.0, self.context_probability)),
            max(-1.0, min(1.0, self.measure_probability_margin)),
            max(-1.0, min(1.0, self.visual_probability_margin)),
            max(-1.0, min(1.0, self.event_probability_margin)),
            max(-1.0, min(1.0, self.context_probability_margin)),
            max(-2.0, min(2.0, self.page_score_margin / 100.0)),
            min(1.0, max(0, self.candidate_count - 1) / 7.0),
            float(self.initial_cluster_member),
            float(self.exact_signature_member),
        ]


@dataclass(frozen=True)
class EnsembleCalibrationResult:
    probability: float
    weight_factor: float
    model_version: str


class EnsembleCalibrator:
    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).resolve().parent / "resources" / "ensemble_calibrator.json"
        loaded = load_verified_json(model_path, "ensemble_candidate_calibration")
        self.forest = VerifiedRandomForestModel.load(
            model_path,
            "ensemble_candidate_calibration",
            FEATURE_NAMES,
            loaded=loaded,
        )
        self.legacy = StandardizedLogisticModel.load(
            model_path,
            "ensemble_candidate_calibration",
            FEATURE_NAMES,
            loaded=loaded,
        )
        self.model_verified = self.forest.verified or self.legacy.verified
        self.model_status = self.forest.status if self.forest.enabled else self.legacy.status
        self.model_version = (
            self.forest.model_version if self.forest.enabled else self.legacy.model_version
        )
        self.enabled = self.forest.enabled or self.legacy.enabled

    def predict_probability(self, item: EnsembleCalibrationInput) -> float:
        if self.forest.enabled:
            return self.forest.predict(item.feature_vector())
        if self.legacy.enabled:
            return self.legacy.predict(item.feature_vector())
        return 0.5

    def calibrate(self, item: EnsembleCalibrationInput) -> EnsembleCalibrationResult:
        probability = self.predict_probability(item)
        if not math.isfinite(probability):
            probability = 0.5
        weight = bounded_weight(
            probability,
            DEFAULT_POLICY.ensemble_calibration_weight_floor,
            DEFAULT_POLICY.ensemble_calibration_weight_ceiling,
        )
        return EnsembleCalibrationResult(
            probability=round(probability, 6),
            weight_factor=round(weight, 6),
            model_version=self.model_version,
        )
