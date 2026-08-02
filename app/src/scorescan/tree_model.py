from __future__ import annotations

"""Dependency-free inference for compact, verified serialized tree ensembles."""

import math
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .model_registry import ModelLoadResult, load_verified_json


def stable_sigmoid(score: float) -> float:
    """Return a numerically stable sigmoid, using neutral evidence for NaN."""
    value = float(score)
    if math.isnan(value):
        return 0.5
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-min(value, 40.0)))
    exp_score = math.exp(max(value, -40.0))
    return exp_score / (1.0 + exp_score)


def _float32(value: float) -> float:
    """Round one value exactly as scikit-learn tree inference does.

    scikit-learn converts input matrices to ``float32`` before traversing trees while
    keeping serialized thresholds as doubles.  Mirroring that conversion prevents rare
    branch changes for values extremely close to a split threshold.
    """
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


def tree_value(tree: object, values: Sequence[float]) -> float:
    """Evaluate one serialized regression tree, returning zero on malformed data."""
    if not isinstance(tree, dict):
        return 0.0
    nodes = tree.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return 0.0
    index = 0
    try:
        for _ in range(len(nodes) + 2):
            node = nodes[index]
            if not isinstance(node, dict):
                return 0.0
            feature = int(node.get("feature", -2))
            if feature == -2:
                value = float(node.get("value", 0.0))
                return value if math.isfinite(value) else 0.0
            if feature < 0 or feature >= len(values):
                return 0.0
            threshold = float(node.get("threshold", 0.0))
            feature_value = _float32(values[feature])
            if not math.isfinite(threshold) or not math.isfinite(feature_value):
                return 0.0
            branch = "left" if feature_value <= threshold else "right"
            index = int(node.get(branch, -1))
            if index < 0 or index >= len(nodes):
                return 0.0
    except (IndexError, TypeError, ValueError, OverflowError):
        return 0.0
    return 0.0


def tree_ensemble_valid(trees: Sequence[object], feature_count: int) -> bool:
    """Validate the structure of every reachable node in a serialized ensemble."""
    if not trees or feature_count <= 0:
        return False
    try:
        for tree in trees:
            if not isinstance(tree, dict):
                return False
            nodes = tree.get("nodes")
            if not isinstance(nodes, list) or not nodes:
                return False
            visiting: set[int] = set()
            visited: set[int] = set()

            def visit(index: int) -> bool:
                if index in visited:
                    return True
                if index in visiting or index < 0 or index >= len(nodes):
                    return False
                node = nodes[index]
                if not isinstance(node, dict):
                    return False
                visiting.add(index)
                feature = int(node.get("feature", -2))
                if feature == -2:
                    value = float(node.get("value", 0.0))
                    valid = math.isfinite(value)
                elif feature < 0:
                    valid = False
                else:
                    threshold = float(node.get("threshold", 0.0))
                    left = int(node.get("left", -1))
                    right = int(node.get("right", -1))
                    valid = (
                        feature < feature_count
                        and math.isfinite(threshold)
                        and visit(left)
                        and visit(right)
                    )
                visiting.remove(index)
                if valid:
                    visited.add(index)
                return valid

            if not visit(0) or len(visited) != len(nodes):
                return False
    except (TypeError, ValueError, OverflowError, RecursionError):
        return False
    return True


def gradient_boosting_probability(
    values: Sequence[float],
    *,
    intercept: float,
    learning_rate: float,
    trees: Iterable[object],
    calibration_intercept: float = 0.0,
    calibration_slope: float = 1.0,
) -> float:
    """Evaluate a serialized binary gradient-boosting ensemble and calibration."""
    try:
        raw_score = float(intercept) + float(learning_rate) * sum(
            tree_value(tree, values) for tree in trees
        )
        calibrated_score = (
            float(calibration_intercept) + float(calibration_slope) * raw_score
        )
    except (TypeError, ValueError, OverflowError):
        return 0.5
    return max(0.0, min(1.0, stable_sigmoid(calibrated_score)))


@dataclass(frozen=True)
class VerifiedGradientBoostingModel:
    """Verified dependency-free binary gradient-boosting model.

    The class centralises JSON validation and calibrated inference for compact models
    which are trained with scikit-learn but executed without a scikit-learn runtime.
    Invalid feature schemas, malformed trees, or non-finite calibration parameters
    disable the model and return neutral evidence instead of partial predictions.
    """

    feature_names: tuple[str, ...]
    model_version: str
    intercept: float
    learning_rate: float
    trees: tuple[object, ...]
    calibration_intercept: float
    calibration_slope: float
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
    ) -> "VerifiedGradientBoostingModel":
        if loaded is None:
            loaded = load_verified_json(path, role)
        payload = loaded.payload
        raw_feature_names = payload.get("feature_names", ())
        declared = (
            tuple(str(value) for value in raw_feature_names)
            if isinstance(raw_feature_names, (list, tuple))
            else ()
        )
        raw_trees = payload.get("trees", ())
        trees = tuple(raw_trees) if isinstance(raw_trees, (list, tuple)) else ()
        try:
            intercept = float(payload.get("intercept", 0.0))
            learning_rate = float(payload.get("learning_rate", 0.0))
            calibration_intercept = float(payload.get("calibration_intercept", 0.0))
            calibration_slope = float(payload.get("calibration_slope", 1.0))
        except (TypeError, ValueError, OverflowError):
            intercept = 0.0
            learning_rate = 0.0
            calibration_intercept = 0.0
            calibration_slope = 1.0
        parameters_valid = (
            math.isfinite(intercept)
            and math.isfinite(learning_rate)
            and learning_rate > 0.0
            and math.isfinite(calibration_intercept)
            and math.isfinite(calibration_slope)
            and calibration_slope > 0.0
        )
        enabled = (
            str(payload.get("model_type", "")) == "gradient_boosting"
            and declared == feature_names
            and parameters_valid
            and tree_ensemble_valid(trees, len(feature_names))
        )
        return cls(
            feature_names=feature_names,
            model_version=str(payload.get("model_version", "disabled")),
            intercept=intercept,
            learning_rate=learning_rate,
            trees=trees,
            calibration_intercept=calibration_intercept,
            calibration_slope=calibration_slope,
            verified=loaded.verified,
            status=loaded.status,
            enabled=enabled,
        )

    def predict(self, values: Sequence[float], *, neutral: float = 0.5) -> float:
        if not self.enabled or len(values) != len(self.feature_names):
            return neutral
        try:
            vector = tuple(float(value) for value in values)
        except (TypeError, ValueError, OverflowError):
            return neutral
        if not all(math.isfinite(value) for value in vector):
            return neutral
        return gradient_boosting_probability(
            vector,
            intercept=self.intercept,
            learning_rate=self.learning_rate,
            trees=self.trees,
            calibration_intercept=self.calibration_intercept,
            calibration_slope=self.calibration_slope,
        )


def random_forest_probability(
    values: Sequence[float],
    *,
    trees: Iterable[object],
    calibration_intercept: float = 0.0,
    calibration_slope: float = 1.0,
) -> float:
    """Evaluate serialized probability trees and an optional logistic calibration."""
    try:
        tree_values = tuple(tree_value(tree, values) for tree in trees)
        if not tree_values:
            return 0.5
        raw_probability = sum(tree_values) / len(tree_values)
        if not math.isfinite(raw_probability):
            return 0.5
        calibrated_score = (
            float(calibration_intercept)
            + float(calibration_slope) * raw_probability
        )
    except (TypeError, ValueError, OverflowError):
        return 0.5
    return max(0.0, min(1.0, stable_sigmoid(calibrated_score)))


@dataclass(frozen=True)
class VerifiedRandomForestModel:
    """Verified dependency-free probability forest used by compact visual models."""

    feature_names: tuple[str, ...]
    model_version: str
    trees: tuple[object, ...]
    calibration_intercept: float
    calibration_slope: float
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
    ) -> "VerifiedRandomForestModel":
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
    ) -> "VerifiedRandomForestModel":
        """Load an embedded forest from an already verified parent resource."""
        if not isinstance(payload, dict):
            payload = {}
        raw_feature_names = payload.get("feature_names", ())
        declared = (
            tuple(str(value) for value in raw_feature_names)
            if isinstance(raw_feature_names, (list, tuple))
            else ()
        )
        raw_trees = payload.get("trees", ())
        trees = tuple(raw_trees) if isinstance(raw_trees, (list, tuple)) else ()
        try:
            calibration_intercept = float(payload.get("calibration_intercept", 0.0))
            calibration_slope = float(payload.get("calibration_slope", 1.0))
        except (TypeError, ValueError, OverflowError):
            calibration_intercept = 0.0
            calibration_slope = 1.0
        parameters_valid = (
            math.isfinite(calibration_intercept)
            and math.isfinite(calibration_slope)
            and calibration_slope > 0.0
        )
        enabled = (
            str(payload.get("model_type", "")) == "random_forest"
            and declared == feature_names
            and parameters_valid
            and tree_ensemble_valid(trees, len(feature_names))
        )
        return cls(
            feature_names=feature_names,
            model_version=str(payload.get("model_version", "disabled")),
            trees=trees,
            calibration_intercept=calibration_intercept,
            calibration_slope=calibration_slope,
            verified=verified,
            status=status,
            enabled=enabled,
        )

    def predict(self, values: Sequence[float], *, neutral: float = 0.5) -> float:
        if not self.enabled or len(values) != len(self.feature_names):
            return neutral
        try:
            vector = tuple(float(value) for value in values)
        except (TypeError, ValueError, OverflowError):
            return neutral
        if not all(math.isfinite(value) for value in vector):
            return neutral
        return random_forest_probability(
            vector,
            trees=self.trees,
            calibration_intercept=self.calibration_intercept,
            calibration_slope=self.calibration_slope,
        )
