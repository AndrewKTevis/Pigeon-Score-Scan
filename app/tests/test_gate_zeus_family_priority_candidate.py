from __future__ import annotations

from copy import deepcopy

from app.tools.gate_zeus_family_priority_candidate import evaluate_candidate


def _report() -> dict:
    family_values = {
        family: {
            "reference_tokens": 10,
            "positioned_f1_percent": 80.0,
        }
        for family in (
            "pitch",
            "rhythm",
            "slur",
            "tie",
            "beam",
            "articulation",
            "accidental",
            "attributes",
        )
    }
    best_values = deepcopy(family_values)
    best_values["tie"]["positioned_f1_percent"] = 82.0
    best_values["slur"]["positioned_f1_percent"] = 82.0
    return {
        "runtime": {
            "keras_policy": "mixed_float16",
            "gpu_devices": ["/physical_device:GPU:0"],
        },
        "data": {
            "manifest_name": (
                "scorescan-olimpic-real-plus-synthetic-replay-v4-source-document-safe"
            ),
            "source_document_isolation_verified": True,
            "source_document_overlap": {
                "train_calibration": 0,
                "train_candidate_test": 0,
                "calibration_candidate_test": 0,
            },
            "candidate_test_opened": False,
            "candidate_test_is_final_product_benchmark": False,
        },
        "metrics_percent": {
            "best_epoch": 2,
            "calibration_baseline": {
                "SER": 15.0,
                "families": family_values,
            },
            "calibration_best": {
                "SER": 14.8,
                "families": best_values,
            },
        },
    }


def _evaluate(report: dict) -> dict:
    return evaluate_candidate(
        report,
        minimum_ser_improvement=0.05,
        minimum_tie_improvement=1.0,
        minimum_slur_improvement=1.0,
        maximum_other_family_regression=0.25,
    )


def test_gate_accepts_material_relation_improvement_without_leakage() -> None:
    result = _evaluate(_report())
    assert result["passed"] is True
    assert result["candidate_test_evaluation_authorized"] is True
    assert result["deployment_authorized"] is False
    assert result["final_product_release_evidence"] is False


def test_gate_rejects_candidate_test_opened_during_selection() -> None:
    report = _report()
    report["data"]["candidate_test_opened"] = True
    result = _evaluate(report)
    assert result["passed"] is False
    assert any("candidate-test" in reason for reason in result["reasons"])


def test_gate_rejects_hidden_family_regression() -> None:
    report = _report()
    report["metrics_percent"]["calibration_best"]["families"]["rhythm"][
        "positioned_f1_percent"
    ] = 79.0
    result = _evaluate(report)
    assert result["passed"] is False
    assert any("rhythm" in reason for reason in result["reasons"])


def test_gate_rejects_missing_physical_source_isolation() -> None:
    report = _report()
    report["data"]["source_document_isolation_verified"] = False
    result = _evaluate(report)
    assert result["passed"] is False
    assert any("source-document" in reason for reason in result["reasons"])
