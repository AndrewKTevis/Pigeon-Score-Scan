import json
from pathlib import Path

from scorescan.ensemble_calibration import EnsembleCalibrationInput, EnsembleCalibrator


def item(**changes: object) -> EnsembleCalibrationInput:
    values: dict[str, object] = {
        "page_score": 980.0,
        "page_probability": 0.82,
        "page_valid": True,
        "alignment_similarity": 0.94,
        "alignment_margin": 0.0,
        "exact_support_ratio": 0.60,
        "semantic_support_ratio": 0.82,
        "signature_support_ratio": 0.60,
        "missing_ratio": 0.0,
        "distance_to_template": 0.02,
        "distance_to_medoid": 0.01,
        "mean_peer_distance": 0.04,
        "measure_probability": 0.80,
        "visual_probability": 0.64,
        "event_probability": 0.84,
        "context_probability": 0.71,
        "measure_probability_margin": 0.18,
        "visual_probability_margin": 0.08,
        "event_probability_margin": 0.20,
        "context_probability_margin": 0.10,
        "page_score_margin": 0.0,
        "candidate_count": 5,
        "initial_cluster_member": True,
        "exact_signature_member": True,
    }
    values.update(changes)
    return EnsembleCalibrationInput(**values)  # type: ignore[arg-type]


def test_ensemble_calibrator_prefers_coherent_candidate() -> None:
    calibrator = EnsembleCalibrator()
    assert calibrator.enabled
    assert calibrator.model_version == "scorescan-ensemble-forest-3"
    assert calibrator.model_verified
    good = calibrator.calibrate(item())
    poor = calibrator.calibrate(
        item(
            alignment_similarity=0.46,
            alignment_margin=-0.42,
            signature_support_ratio=0.20,
            semantic_support_ratio=0.38,
            distance_to_template=0.65,
            distance_to_medoid=0.54,
            mean_peer_distance=0.61,
            measure_probability=0.24,
            event_probability=0.22,
            context_probability=0.26,
            measure_probability_margin=-0.40,
            event_probability_margin=-0.42,
            initial_cluster_member=False,
            exact_signature_member=False,
        )
    )
    assert good.probability > poor.probability
    assert good.weight_factor > poor.weight_factor
    assert 0.86 <= poor.weight_factor <= 1.14
    assert 0.86 <= good.weight_factor <= 1.14


def test_ensemble_calibrator_keeps_v1_resource_compatibility(tmp_path: Path) -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "training"
        / "baselines"
        / "ensemble_calibrator_v1.json"
    )
    model_path = tmp_path / "ensemble_v1.json"
    model_path.write_bytes(source.read_bytes())
    calibrator = EnsembleCalibrator(model_path)
    assert calibrator.enabled
    assert calibrator.model_version == "scorescan-ensemble-calibrator-1"
    assert 0.0 <= calibrator.predict_probability(item()) <= 1.0


def test_ensemble_calibrator_disables_malformed_forest(tmp_path: Path) -> None:
    payload = {
        "model_version": "broken-forest",
        "model_type": "random_forest",
        "feature_names": [
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
        ],
        "calibration_intercept": 0.0,
        "calibration_slope": 1.0,
        "trees": [
            {
                "nodes": [
                    {"feature": 0, "threshold": 0.5, "left": 0, "right": 0, "value": 0.5}
                ]
            }
        ],
    }
    model_path = tmp_path / "broken.json"
    model_path.write_text(json.dumps(payload), encoding="utf-8")
    calibrator = EnsembleCalibrator(model_path)
    assert not calibrator.enabled
    assert calibrator.predict_probability(item()) == 0.5


def test_current_component_ensemble_compatibility_audit_passes() -> None:
    report_path = (
        Path(__file__).resolve().parents[2]
        / "training"
        / "ensemble_component_compatibility_audit_v3.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["audit_version"] == "scorescan-ensemble-compatibility-audit-3"
    assert report["current_models"]["ensemble"] == "scorescan-ensemble-forest-3"
    assert report["current_models"]["measure"] == "scorescan-measure-forest-3"
    assert report["current_models"]["visual"] == "scorescan-visual-measure-calibrator-4"
    assert report["passed"]
    assert all(report["checks"].values())


def test_candidate_count_feature_distinguishes_seven_from_eight() -> None:
    seven = item(candidate_count=7).feature_vector()
    eight = item(candidate_count=8).feature_vector()
    index = 21
    assert seven[index] == 6.0 / 7.0
    assert eight[index] == 1.0
