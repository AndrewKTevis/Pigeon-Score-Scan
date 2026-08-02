from __future__ import annotations

"""Deterministic exporters for compact scikit-learn tree models.

Training tools share this module so JSON structure and deployment-parity checks cannot
drift between individual CPU models.
"""

from collections.abc import Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from scorescan.tree_model import random_forest_probability


def serialize_probability_forest(model: RandomForestClassifier) -> list[dict[str, object]]:
    trees: list[dict[str, object]] = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        nodes: list[dict[str, object]] = []
        for index in range(tree.node_count):
            counts = np.asarray(tree.value[index]).ravel()
            probability = (
                float(counts[1] / max(float(np.sum(counts)), 1e-12))
                if counts.size >= 2
                else 0.0
            )
            nodes.append(
                {
                    "feature": int(tree.feature[index]),
                    "threshold": float(tree.threshold[index]),
                    "left": int(tree.children_left[index]),
                    "right": int(tree.children_right[index]),
                    "value": probability,
                }
            )
        trees.append({"nodes": nodes})
    return trees


def deployed_forest_probabilities(
    payload: dict[str, object],
    values: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    trees = payload.get("trees", ())
    return np.asarray(
        [
            random_forest_probability(
                row,
                trees=trees if isinstance(trees, (list, tuple)) else (),
                calibration_intercept=float(payload.get("calibration_intercept", 0.0)),
                calibration_slope=float(payload.get("calibration_slope", 1.0)),
            )
            for row in values
        ],
        dtype=np.float64,
    )
