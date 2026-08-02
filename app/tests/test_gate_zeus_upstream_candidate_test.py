from __future__ import annotations

from app.tools.gate_zeus_upstream_candidate_test import (
    MINIMUM_POSITIONED_F1,
    evaluate_upstream_test,
)


def _report(value: float = 99.0) -> dict:
    return {
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
            "candidate_test_opened": True,
            "candidate_test_is_final_product_benchmark": False,
        },
        "metrics_percent": {
            "candidate_test_best": {
                "SER": 1.0,
                "families": {
                    family: {
                        "reference_tokens": 10,
                        "positioned_f1_percent": value,
                    }
                    for family in MINIMUM_POSITIONED_F1
                },
            }
        },
    }


def test_upstream_gate_never_authorizes_desktop_deployment() -> None:
    result = evaluate_upstream_test(_report(), maximum_ser=5.0)
    assert result["passed"] is True
    assert result["observer_candidate_authorized"] is True
    assert result["desktop_deployment_authorized"] is False
    assert result["final_product_release_evidence"] is False


def test_upstream_gate_rejects_low_positioned_accuracy() -> None:
    report = _report()
    report["metrics_percent"]["candidate_test_best"]["families"]["tie"][
        "positioned_f1_percent"
    ] = 90.0
    result = evaluate_upstream_test(report, maximum_ser=5.0)
    assert result["passed"] is False
    assert any("tie" in reason for reason in result["reasons"])


def test_upstream_gate_rejects_source_document_leakage() -> None:
    report = _report()
    report["data"]["source_document_overlap"]["train_candidate_test"] = 1
    result = evaluate_upstream_test(report, maximum_ser=5.0)
    assert result["passed"] is False
    assert any("source-document" in reason for reason in result["reasons"])
