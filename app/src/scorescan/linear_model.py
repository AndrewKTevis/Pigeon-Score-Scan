from __future__ import annotations

"""Small verified linear models shared by ScoreScan's CPU calibrators.

The application bundles several deliberately small logistic models.  They all use the
same JSON representation (feature names, means, scales, coefficients and intercept),
but historically each calibrator duplicated loading, validation and numerically stable
sigmoid code.  This module provides one immutable implementation so new evidence layers
can be added without creating subtly different model semantics.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .model_registry import ModelLoadResult, load_verified_json


@dataclass(frozen=True)
class StandardizedLogisticModel:
    feature_names: tuple[str, ...]
    model_version: str
    intercept: float
    coefficients: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    verified: bool
    status: str
    enabled: bool

    @classmethod
    def load(
        cls,
        path: Path,
        role: str,
        feature_names: tuple[str, ...],
        *,
        loaded: ModelLoadResult | None = None,
    ) -> "StandardizedLogisticModel":
        if loaded is None:
            loaded = load_verified_json(path, role)
        return cls.from_payload(
            loaded.payload,
            feature_names,
            verified=loaded.verified,
            status=loaded.status,
        )

    @classmethod
    def from_payload(
        cls,
        payload: object,
        feature_names: tuple[str, ...],
        *,
        verified: bool = False,
        status: str = "embedded",
    ) -> "StandardizedLogisticModel":
        disabled = cls(
            feature_names=feature_names,
            model_version="disabled",
            intercept=0.0,
            coefficients=(),
            means=(),
            scales=(),
            verified=verified,
            status=status,
            enabled=False,
        )
        if not isinstance(payload, dict):
            return disabled
        try:
            coefficients = tuple(float(value) for value in payload.get("coefficients", []))
            means = tuple(float(value) for value in payload.get("means", []))
            raw_scales = tuple(float(value) for value in payload.get("scales", []))
            intercept = float(payload.get("intercept", 0.0))
            declared = tuple(str(value) for value in payload.get("feature_names", ()))
        except (TypeError, ValueError, OverflowError):
            return disabled
        enabled = (
            declared == feature_names
            and len(coefficients) == len(feature_names)
            and len(means) == len(feature_names)
            and len(raw_scales) == len(feature_names)
            and math.isfinite(intercept)
            and all(math.isfinite(value) for value in coefficients)
            and all(math.isfinite(value) for value in means)
            and all(math.isfinite(value) and value > 0.0 for value in raw_scales)
        )
        if not enabled:
            return cls(
                feature_names=feature_names,
                model_version=str(payload.get("model_version", "disabled")),
                intercept=intercept if math.isfinite(intercept) else 0.0,
                coefficients=coefficients,
                means=means,
                scales=raw_scales,
                verified=verified,
                status=status,
                enabled=False,
            )
        return cls(
            feature_names=feature_names,
            model_version=str(payload.get("model_version", "disabled")),
            intercept=intercept,
            coefficients=coefficients,
            means=means,
            scales=raw_scales,
            verified=verified,
            status=status,
            enabled=True,
        )

    @staticmethod
    def _sigmoid(score: float) -> float:
        if score >= 0:
            return 1.0 / (1.0 + math.exp(-min(score, 40.0)))
        exp_score = math.exp(max(score, -40.0))
        return exp_score / (1.0 + exp_score)

    def predict(self, values: Iterable[float], *, neutral: float = 0.5) -> float:
        if not self.enabled:
            return neutral
        vector = tuple(float(value) for value in values)
        if len(vector) != len(self.feature_names):
            return neutral
        standardized = tuple(
            (value - mean) / scale
            for value, mean, scale in zip(vector, self.means, self.scales, strict=True)
        )
        score = self.intercept + sum(
            coefficient * value
            for coefficient, value in zip(self.coefficients, standardized, strict=True)
        )
        return max(0.0, min(1.0, self._sigmoid(score)))


def bounded_weight(probability: float, floor: float, ceiling: float) -> float:
    """Map a probability to a conservative multiplicative weight.

    ``0.5`` maps to the midpoint.  The mapping is deliberately linear so the maximum
    influence remains obvious in audits and release notes.
    """

    probability = max(0.0, min(1.0, float(probability)))
    floor = float(floor)
    ceiling = float(ceiling)
    if ceiling < floor:
        floor, ceiling = ceiling, floor
    return floor + (ceiling - floor) * probability
