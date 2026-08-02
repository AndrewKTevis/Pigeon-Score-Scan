from scorescan.policy import DEFAULT_POLICY
from scorescan.selection_risk import (
    SelectionRiskCalibrator,
    SelectionRiskInput,
    corroborated_exact_majority,
    corroborated_semantic_consensus,
)


def item(**changes: object) -> SelectionRiskInput:
    values: dict[str, object] = {
        "selection_kind": "semantic_consensus",
        "selected_page_score": 980.0,
        "selected_page_probability": 0.90,
        "selected_ensemble_probability": 0.98,
        "ensemble_probability_margin": 0.25,
        "selected_measure_probability": 0.90,
        "measure_probability_margin": 0.25,
        "selected_visual_probability": 0.75,
        "visual_probability_margin": 0.20,
        "selected_event_probability": 0.90,
        "event_probability_margin": 0.25,
        "selected_context_probability": 0.80,
        "context_probability_margin": 0.20,
        "exact_support_ratio": 0.45,
        "semantic_support_ratio": 0.85,
        "signature_support_ratio": 0.45,
        "missing_ratio": 0.0,
        "mean_cluster_distance": 0.01,
        "template_distance": 0.15,
        "alignment_similarity": 0.95,
        "alignment_margin": 0.0,
        "selected_distance_to_medoid": 0.0,
        "selected_mean_peer_distance": 0.03,
        "page_score_margin": -0.05,
        "candidate_count": 5,
        "exact_support_count": 2,
        "distinct_signature_count": 3,
        "top_signature_margin": 0.20,
        "unanimous": False,
        "strict_majority": False,
        "selected_is_template": False,
        "selected_is_exact_signature": True,
        "selected_in_initial_cluster": True,
        "page_valid": True,
        "selected_vs_template_page_probability": 0.20,
        "selected_vs_template_ensemble_probability": 0.30,
        "selected_vs_template_measure_probability": 0.25,
        "selected_vs_template_visual_probability": 0.10,
        "selected_vs_template_event_probability": 0.25,
        "selected_vs_template_context_probability": 0.20,
        "selected_vs_template_alignment_similarity": 0.15,
        "template_page_valid": True,
        "template_in_initial_cluster": False,
        "template_is_exact_signature": False,
        "eligible_family_count": 3,
        "exact_family_support_count": 2,
        "semantic_family_support_count": 3,
    }
    values.update(changes)
    return SelectionRiskInput(**values)  # type: ignore[arg-type]


def test_selection_risk_gate_is_verified_and_conservative() -> None:
    calibrator = SelectionRiskCalibrator()
    assert calibrator.enabled
    assert calibrator.model_verified
    assert calibrator.threshold >= DEFAULT_POLICY.replacement_selection_risk_floor
    assert calibrator.model_version == "scorescan-selection-risk-forest-4"

    strong = calibrator.calibrate(
        item(
            selection_kind="exact_majority",
            strict_majority=True,
            exact_support_ratio=0.75,
            candidate_count=4,
            exact_support_count=3,
            eligible_family_count=4,
            exact_family_support_count=3,
            semantic_family_support_count=3,
        )
    )
    weak = calibrator.calibrate(
        item(
            selected_page_score=700.0,
            selected_page_probability=0.15,
            selected_ensemble_probability=0.20,
            ensemble_probability_margin=-0.40,
            selected_measure_probability=0.20,
            measure_probability_margin=-0.40,
            selected_visual_probability=0.20,
            visual_probability_margin=-0.40,
            selected_event_probability=0.20,
            event_probability_margin=-0.40,
            selected_context_probability=0.20,
            context_probability_margin=-0.40,
            semantic_support_ratio=0.50,
            missing_ratio=0.30,
            mean_cluster_distance=0.12,
            alignment_similarity=0.45,
            alignment_margin=-0.40,
            selected_distance_to_medoid=0.30,
            selected_mean_peer_distance=0.50,
            page_score_margin=-250.0,
            selected_in_initial_cluster=False,
            page_valid=False,
            selected_vs_template_page_probability=-0.60,
            selected_vs_template_ensemble_probability=-0.70,
            selected_vs_template_measure_probability=-0.60,
            selected_vs_template_visual_probability=-0.50,
            selected_vs_template_event_probability=-0.65,
            selected_vs_template_context_probability=-0.55,
            selected_vs_template_alignment_similarity=-0.45,
            template_in_initial_cluster=True,
            template_is_exact_signature=True,
        )
    )
    assert strong.probability > weak.probability
    assert strong.accepted
    assert not weak.accepted


def test_selection_risk_uses_direct_template_comparison() -> None:
    calibrator = SelectionRiskCalibrator()
    favourable = calibrator.predict_probability(item())
    unfavourable = calibrator.predict_probability(
        item(
            selected_vs_template_page_probability=-0.20,
            selected_vs_template_ensemble_probability=-0.30,
            selected_vs_template_measure_probability=-0.25,
            selected_vs_template_visual_probability=-0.10,
            selected_vs_template_event_probability=-0.25,
            selected_vs_template_context_probability=-0.20,
            selected_vs_template_alignment_similarity=-0.15,
            template_in_initial_cluster=True,
            template_is_exact_signature=True,
        )
    )
    assert favourable > unfavourable


def test_selection_risk_penalises_correlated_family_support() -> None:
    calibrator = SelectionRiskCalibrator()
    diverse = calibrator.predict_probability(
        item(
            selection_kind="exact_majority",
            strict_majority=True,
            exact_support_ratio=0.75,
            candidate_count=7,
            exact_support_count=5,
            eligible_family_count=4,
            exact_family_support_count=4,
            semantic_family_support_count=4,
        )
    )
    duplicated = calibrator.predict_probability(
        item(
            selection_kind="exact_majority",
            strict_majority=True,
            exact_support_ratio=0.75,
            candidate_count=7,
            exact_support_count=5,
            eligible_family_count=4,
            exact_family_support_count=2,
            semantic_family_support_count=2,
        )
    )
    assert diverse > duplicated


def test_selection_risk_rejects_malformed_tree_model(tmp_path) -> None:
    import json
    from pathlib import Path

    resource = Path(__file__).resolve().parents[1] / "src" / "scorescan" / "resources" / "selection_risk.json"
    payload = json.loads(resource.read_text(encoding="utf-8"))
    payload["trees"][0]["nodes"][0]["left"] = 0
    malformed = tmp_path / "selection_risk.json"
    malformed.write_text(json.dumps(payload), encoding="utf-8")

    calibrator = SelectionRiskCalibrator(malformed)
    result = calibrator.calibrate(item())
    assert not calibrator.enabled
    assert not result.accepted
    assert result.probability == 0.5


def test_family_count_feature_distinguishes_four_from_five() -> None:
    four = item(eligible_family_count=4).feature_vector()
    five = item(eligible_family_count=5).feature_vector()
    index = 46
    assert four[index] == 0.8
    assert five[index] == 1.0


def test_exact_replacement_permission_requires_three_complete_families() -> None:
    strong = item(
        selection_kind="exact_majority",
        strict_majority=True,
        exact_support_ratio=0.75,
        candidate_count=4,
        exact_support_count=3,
        eligible_family_count=4,
        exact_family_support_count=3,
        semantic_family_support_count=3,
    )
    assert corroborated_exact_majority(strong)
    assert not corroborated_exact_majority(
        item(
            selection_kind="exact_majority",
            strict_majority=True,
            exact_support_ratio=0.75,
            candidate_count=6,
            exact_support_count=5,
            eligible_family_count=4,
            exact_family_support_count=2,
            semantic_family_support_count=2,
        )
    )
    assert not corroborated_exact_majority(
        item(
            selection_kind="exact_majority",
            strict_majority=True,
            exact_support_ratio=0.75,
            candidate_count=4,
            exact_support_count=3,
            eligible_family_count=4,
            exact_family_support_count=3,
            semantic_family_support_count=3,
            selected_vs_template_visual_probability=-0.20,
            selected_vs_template_event_probability=0.01,
            selected_vs_template_context_probability=-0.15,
        )
    )


def test_semantic_replacement_permission_enforces_runtime_invariants() -> None:
    strong = item(
        semantic_support_ratio=0.88,
        eligible_family_count=4,
        semantic_family_support_count=4,
        selected_distance_to_medoid=0.0,
        mean_cluster_distance=0.02,
        page_score_margin=-10.0,
        selected_vs_template_visual_probability=-0.02,
    )
    assert corroborated_semantic_consensus(strong)
    assert not corroborated_semantic_consensus(
        item(semantic_support_ratio=0.83, semantic_family_support_count=3)
    )
    assert not corroborated_semantic_consensus(
        item(selected_distance_to_medoid=0.001, semantic_family_support_count=3)
    )
    assert not corroborated_semantic_consensus(
        item(mean_cluster_distance=0.06, semantic_family_support_count=3)
    )
    assert not corroborated_semantic_consensus(
        item(
            semantic_support_ratio=0.88,
            semantic_family_support_count=3,
            page_score_margin=-20.0,
            selected_vs_template_visual_probability=-0.04,
        )
    )
    assert not corroborated_semantic_consensus(
        item(
            semantic_support_ratio=0.88,
            semantic_family_support_count=3,
            selected_vs_template_context_probability=-0.20,
        )
    )
