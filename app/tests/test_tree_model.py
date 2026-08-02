from __future__ import annotations

import json
import math

import pytest

from scorescan.util import sha256_file

from scorescan.tree_model import (
    gradient_boosting_probability,
    stable_sigmoid,
    VerifiedGradientBoostingModel,
    tree_ensemble_valid,
    tree_value,
)


def test_tree_runtime_matches_expected_calibrated_probability() -> None:
    tree = {
        "nodes": [
            {"feature": 0, "threshold": 0.25, "left": 1, "right": 2, "value": 0.0},
            {"feature": -2, "value": -0.5},
            {"feature": -2, "value": 0.75},
        ]
    }
    probability = gradient_boosting_probability(
        [0.5],
        intercept=0.1,
        learning_rate=0.4,
        trees=[tree],
        calibration_intercept=-0.2,
        calibration_slope=1.5,
    )
    expected_score = -0.2 + 1.5 * (0.1 + 0.4 * 0.75)
    assert probability == pytest.approx(stable_sigmoid(expected_score))


def test_tree_runtime_neutralises_malformed_or_nan_evidence() -> None:
    malformed = {"nodes": [{"feature": "bad", "value": "bad"}]}
    assert tree_value(malformed, [1.0]) == 0.0
    assert tree_value({"nodes": [{"feature": 4, "threshold": 0.0}]}, [1.0]) == 0.0
    assert gradient_boosting_probability(
        [math.nan],
        intercept=0.0,
        learning_rate=1.0,
        trees=[{"nodes": [{"feature": 0, "threshold": 0.0, "left": 0, "right": 0}]}],
    ) == pytest.approx(0.5)


def test_tree_ensemble_validation_rejects_cycles_and_unreachable_nodes() -> None:
    valid = {
        "nodes": [
            {"feature": 0, "threshold": 0.0, "left": 1, "right": 2},
            {"feature": -2, "value": -0.5},
            {"feature": -2, "value": 0.5},
        ]
    }
    cyclic = {"nodes": [{"feature": 0, "threshold": 0.0, "left": 0, "right": 0}]}
    unreachable = {
        "nodes": [
            {"feature": -2, "value": 0.0},
            {"feature": -2, "value": 1.0},
        ]
    }
    assert tree_ensemble_valid([valid], 1)
    assert not tree_ensemble_valid([cyclic], 1)
    assert not tree_ensemble_valid([unreachable], 1)


def test_verified_gradient_boosting_model_loads_and_rejects_bad_schema(tmp_path) -> None:
    resource = tmp_path / "barline_sequence_classifier.json"
    payload = {
        "model_version": "test-gbdt-1",
        "model_type": "gradient_boosting",
        "feature_names": ["x"],
        "intercept": 0.0,
        "learning_rate": 1.0,
        "calibration_intercept": 0.0,
        "calibration_slope": 1.0,
        "trees": [
            {
                "nodes": [
                    {"feature": 0, "threshold": 0.0, "left": 1, "right": 2},
                    {"feature": -2, "value": -1.0},
                    {"feature": -2, "value": 1.0},
                ]
            }
        ],
    }
    resource.write_text(json.dumps(payload), encoding="utf-8")
    manifest = {
        "format": 1,
        "models": [
            {
                "file": resource.name,
                "role": "barline_sequence_classification",
                "model_version": payload["model_version"],
                "sha256": sha256_file(resource),
                "bytes": resource.stat().st_size,
            }
        ],
    }
    (tmp_path / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    model = VerifiedGradientBoostingModel.load(
        resource,
        "barline_sequence_classification",
        ("x",),
    )
    assert model.enabled
    assert model.predict((1.0,)) > 0.5
    assert model.predict((math.nan,)) == 0.5
    assert model.predict(()) == 0.5

    wrong_schema = VerifiedGradientBoostingModel.load(
        resource,
        "barline_sequence_classification",
        ("different",),
    )
    assert not wrong_schema.enabled
    assert wrong_schema.predict((1.0,)) == 0.5

    payload["feature_names"] = None
    payload["trees"] = None
    resource.write_text(json.dumps(payload), encoding="utf-8")
    manifest["models"][0]["sha256"] = sha256_file(resource)
    manifest["models"][0]["bytes"] = resource.stat().st_size
    (tmp_path / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    malformed_collections = VerifiedGradientBoostingModel.load(
        resource,
        "barline_sequence_classification",
        ("x",),
    )
    assert not malformed_collections.enabled
    assert malformed_collections.predict((1.0,)) == 0.5


def test_tree_runtime_matches_sklearn_float32_branching() -> None:
    # scikit-learn casts input features to float32 before comparing them with the
    # double-precision tree threshold.  The unrounded Python value would go left;
    # its float32 representation correctly goes right.
    tree = {
        "nodes": [
            {"feature": 0, "threshold": 1.00000008, "left": 1, "right": 2},
            {"feature": -2, "value": -1.0},
            {"feature": -2, "value": 1.0},
        ]
    }
    assert tree_value(tree, [1.00000006]) == 1.0


def test_verified_random_forest_model_loads_and_neutralises_invalid_input(tmp_path) -> None:
    from scorescan.tree_model import VerifiedRandomForestModel

    resource = tmp_path / "barline_classifier.json"
    payload = {
        "model_version": "test-forest-1",
        "model_type": "random_forest",
        "feature_names": ["x"],
        "calibration_intercept": -1.0,
        "calibration_slope": 4.0,
        "trees": [
            {
                "nodes": [
                    {"feature": 0, "threshold": 0.0, "left": 1, "right": 2},
                    {"feature": -2, "value": 0.1},
                    {"feature": -2, "value": 0.9},
                ]
            }
        ],
    }
    resource.write_text(json.dumps(payload), encoding="utf-8")
    manifest = {
        "format": 1,
        "models": [
            {
                "file": resource.name,
                "role": "barline_classification",
                "model_version": payload["model_version"],
                "sha256": sha256_file(resource),
                "bytes": resource.stat().st_size,
            }
        ],
    }
    (tmp_path / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    model = VerifiedRandomForestModel.load(resource, "barline_classification", ("x",))
    assert model.enabled
    assert model.predict((1.0,)) > model.predict((-1.0,))
    assert model.predict((math.nan,)) == 0.5
    assert model.predict(()) == 0.5

    payload["trees"][0]["nodes"][0]["left"] = 0
    resource.write_text(json.dumps(payload), encoding="utf-8")
    manifest["models"][0]["sha256"] = sha256_file(resource)
    manifest["models"][0]["bytes"] = resource.stat().st_size
    (tmp_path / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    malformed = VerifiedRandomForestModel.load(resource, "barline_classification", ("x",))
    assert not malformed.enabled
    assert malformed.predict((1.0,)) == 0.5
