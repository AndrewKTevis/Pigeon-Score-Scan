from __future__ import annotations

"""Small deterministic calibration model for page-level OMR candidates.

The model does not recognise notation.  It estimates whether a candidate is likely to
be the best member of an ensemble from structural audits and cross-variant agreement.
It is deliberately bounded and subordinate to hard XML/music validation.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .model_registry import load_verified_json


FEATURE_NAMES = (
    "raw_score_scaled",
    "valid",
    "agreement_ratio",
    "validation_errors",
    "measure_gap_ratio",
    "notes_per_measure",
    "rhythm_issue_rate",
    "tie_issue_rate",
    "slur_issue_rate",
    "semantic_issue_rate",
    "type_duration_rate",
    "multiple_voice_rate",
    "duplicate_direction_rate",
    "empty_measure_rate",
    "zero_duration_rate",
    "chord_duration_rate",
    "pitch_outlier_rate",
    "density_outlier_rate",
    "engine_error",
)


class CandidateFeatures(Protocol):
    raw_score: float
    valid: bool
    agreement_ratio: float
    validation_errors: tuple[str, ...]
    measure_gap: int | None
    measure_count: int
    note_count: int
    rhythm_issue_count: int
    tie_issue_count: int
    slur_issue_count: int
    semantic_issue_count: int
    type_duration_mismatch_count: int
    multiple_voice_measure_count: int
    duplicate_direction_count: int
    empty_measure_count: int
    zero_duration_count: int
    chord_duration_mismatch_count: int
    pitch_outlier_count: int
    density_outlier_count: int
    error: str | None


def feature_vector(candidate: CandidateFeatures) -> list[float]:
    measures = max(int(candidate.measure_count or 0), 1)
    notes = max(int(candidate.note_count or 0), 0)
    gap = max(int(candidate.measure_gap or 0), 0)
    return [
        max(-2.0, min(2.0, float(candidate.raw_score) / 1000.0)),
        float(bool(candidate.valid)),
        max(0.0, min(1.0, float(candidate.agreement_ratio))),
        min(10.0, float(len(candidate.validation_errors))),
        min(2.0, gap / measures),
        min(40.0, notes / measures),
        min(5.0, candidate.rhythm_issue_count / measures),
        min(5.0, candidate.tie_issue_count / measures),
        min(5.0, candidate.slur_issue_count / measures),
        min(10.0, candidate.semantic_issue_count / measures),
        min(5.0, candidate.type_duration_mismatch_count / measures),
        min(5.0, candidate.multiple_voice_measure_count / measures),
        min(5.0, candidate.duplicate_direction_count / measures),
        min(5.0, candidate.empty_measure_count / measures),
        min(5.0, candidate.zero_duration_count / measures),
        min(5.0, candidate.chord_duration_mismatch_count / measures),
        min(5.0, candidate.pitch_outlier_count / measures),
        min(5.0, candidate.density_outlier_count / measures),
        float(bool(candidate.error)),
    ]


@dataclass(frozen=True)
class CalibrationResult:
    probability: float
    adjustment: float
    model_version: str


class CandidateCalibrator:
    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).resolve().parent / "resources" / "candidate_calibrator.json"
        loaded = load_verified_json(model_path, "page_candidate_calibration")
        payload = loaded.payload
        self.model_verified = loaded.verified
        self.model_status = loaded.status
        self.model_version = str(payload.get("model_version", "disabled"))
        self.intercept = float(payload.get("intercept", 0.0))
        self.coefficients = tuple(float(value) for value in payload.get("coefficients", []))
        self.means = tuple(float(value) for value in payload.get("means", []))
        self.scales = tuple(max(float(value), 1e-9) for value in payload.get("scales", []))
        self.enabled = (
            len(self.coefficients) == len(FEATURE_NAMES)
            and len(self.means) == len(FEATURE_NAMES)
            and len(self.scales) == len(FEATURE_NAMES)
        )

    def predict_probability(self, candidate: CandidateFeatures) -> float:
        if not self.enabled:
            return 0.5
        values = feature_vector(candidate)
        standardized = [(value - mean) / scale for value, mean, scale in zip(values, self.means, self.scales, strict=True)]
        score = self.intercept + sum(coef * value for coef, value in zip(self.coefficients, standardized, strict=True))
        if score >= 0:
            probability = 1.0 / (1.0 + math.exp(-min(score, 40.0)))
        else:
            exp_score = math.exp(max(score, -40.0))
            probability = exp_score / (1.0 + exp_score)
        return max(0.0, min(1.0, probability))

    def calibrate(self, candidate: CandidateFeatures) -> CalibrationResult:
        probability = self.predict_probability(candidate)
        # Calibration is a secondary prior.  It can reorder close candidates but cannot
        # erase hard validation penalties or dominate ensemble evidence.
        adjustment = max(-32.0, min(32.0, (probability - 0.5) * 64.0))
        return CalibrationResult(round(probability, 6), round(adjustment, 3), self.model_version)
