from scorescan.models import PageInfo, ReviewIssue
from scorescan.quality_certificate import build_quality_certificate


def page(**changes):
    item = PageInfo(
        1,
        "page.png",
        "page.png",
        quality_score=95.0,
        omr_status="completed",
        notation_coverage_status="completed",
        consensus_agreement=1.0,
        consensus_confidence=0.95,
        consensus_ensemble_probability=0.95,
        recognition_candidates=[
            {"variant": "primary", "valid": True},
            {"variant": "flat", "valid": True},
            {"variant": "otsu", "valid": True},
        ],
    )
    for key, value in changes.items():
        setattr(item, key, value)
    return item


def test_clean_result_is_verified() -> None:
    certificate = build_quality_certificate(
        [page(consensus_agreement=1.0)],
        [],
        {"rhythm_issues": [], "tie_issues": [], "slur_issues": []},
        [],
        [],
    )
    assert certificate.state == "verified"
    assert certificate.score >= 92


def test_high_semantic_consensus_can_verify_small_raw_encoding_differences() -> None:
    certificate = build_quality_certificate(
        [
            page(
                consensus_agreement=0.89,
                consensus_semantic_agreement=0.999,
                consensus_confidence=0.99,
                consensus_disagreements=[],
                consensus_unresolved=[],
            )
        ],
        [],
        {"rhythm_issues": [], "tie_issues": [], "slur_issues": []},
        [],
        [],
    )

    assert certificate.state == "verified"
    assert certificate.auto_release_eligible


def test_high_semantic_consensus_can_verify_resolved_audit_disagreements() -> None:
    certificate = build_quality_certificate(
        [
            page(
                consensus_agreement=0.811498,
                consensus_semantic_agreement=0.999,
                consensus_confidence=0.99,
                consensus_disagreements=[4, 19, 24],
                consensus_unresolved=[],
                consensus_measure_probability=0.532261,
                consensus_selection_risk_probability=0.34,
            )
        ],
        [],
        {"rhythm_issues": [], "tie_issues": [], "slur_issues": []},
        [],
        [],
    )

    assert certificate.state == "verified"
    assert certificate.auto_release_eligible
    assert not any("自动替换安全证据偏低" in reason for reason in certificate.reasons)
    assert not any("候选一致率偏低" in reason for reason in certificate.reasons)
    assert not any("小节选择模型置信度偏低" in reason for reason in certificate.reasons)
    assert not any("候选一致率低于" in blocker for blocker in certificate.release_blockers)


def test_low_raw_agreement_without_semantic_evidence_still_requires_review() -> None:
    certificate = build_quality_certificate(
        [page(consensus_agreement=0.89, consensus_semantic_agreement=None)],
        [],
        {"rhythm_issues": [], "tie_issues": [], "slur_issues": []},
        [],
        [],
    )

    assert certificate.state == "review_recommended"
    assert not certificate.auto_release_eligible


def test_pending_consensus_review_prevents_verified() -> None:
    p = page(consensus_agreement=0.75, consensus_unresolved=[2])
    issue = ReviewIssue(
        id="m2", page_index=1, category="measure_consensus", title="检查", message="分歧",
        writeback_supported=False, requires_value=False, risk_preserved=True,
    )
    certificate = build_quality_certificate(
        [p], [], {"rhythm_issues": [], "tie_issues": [], "slur_issues": []}, [issue], []
    )
    assert certificate.state == "review_recommended"
    assert certificate.pending_review_count == 1
    assert certificate.score <= 92


def test_many_unresolved_measures_cannot_receive_cosmetic_perfect_score() -> None:
    certificate = build_quality_certificate(
        [page(consensus_unresolved=list(range(1, 15)))],
        [],
        {"rhythm_issues": [], "tie_issues": [], "slur_issues": []},
        [],
        [],
    )

    assert certificate.state == "best_effort"
    assert certificate.score == 65.0
    assert not certificate.auto_release_eligible


def test_fallback_page_is_best_effort() -> None:
    p = page(omr_status="fallback")
    certificate = build_quality_certificate(
        [p], [], {"rhythm_issues": [], "tie_issues": [], "slur_issues": []}, [], [1]
    )
    assert certificate.state == "best_effort"
    assert certificate.hard_failures


def test_semantic_corruption_prevents_verified_state() -> None:
    certificate = build_quality_certificate(
        [page(consensus_agreement=1.0, consensus_confidence=1.0)],
        [],
        {
            "rhythm_issues": [], "tie_issues": [], "slur_issues": [],
            "semantic_issue_counts": {"empty_measure": 1, "pitch_outlier": 1},
        },
        [],
        [],
    )
    assert certificate.state == "review_recommended"
    assert certificate.score <= 92


def test_production_release_requires_source_notation_coverage_audit() -> None:
    certificate = build_quality_certificate(
        [page(notation_coverage_status="failed")],
        [],
        {"rhythm_issues": [], "tie_issues": [], "slur_issues": []},
        [],
        [],
    )
    assert certificate.state == "review_recommended"
    assert not certificate.auto_release_eligible
    assert any("覆盖审计" in item for item in certificate.release_blockers)


def test_verified_semantic_detector_requires_position_level_source_audit() -> None:
    certificate = build_quality_certificate(
        [
            page(
                semantic_detector_status="verified",
                semantic_detector_enabled=True,
                semantic_detector_accelerator_verified=True,
                semantic_source_audit_status="failed",
            )
        ],
        [],
        {"rhythm_issues": [], "tie_issues": [], "slur_issues": []},
        [],
        [],
    )
    assert not certificate.auto_release_eligible
    assert any("位置级语义符号审计" in item for item in certificate.release_blockers)


def test_position_level_source_symbol_mismatch_blocks_false_confidence() -> None:
    certificate = build_quality_certificate(
        [
            page(
                semantic_detector_status="verified",
                semantic_detector_enabled=True,
                semantic_detector_accelerator_verified=True,
                semantic_source_audit_status="completed",
                semantic_source_audit_extraneous_count=1,
                semantic_source_audit_positional_mismatch_count=1,
            )
        ],
        [],
        {"rhythm_issues": [], "tie_issues": [], "slur_issues": []},
        [],
        [],
    )
    assert certificate.state == "review_recommended"
    assert not certificate.auto_release_eligible
    assert any("位置级语义符号不一致" in item for item in certificate.release_blockers)


def test_production_release_requires_three_independent_families() -> None:
    certificate = build_quality_certificate(
        [page(recognition_candidates=[
            {"variant": "primary", "valid": True},
            {"variant": "flat", "valid": True},
        ])],
        [],
        {"rhythm_issues": [], "tie_issues": [], "slur_issues": []},
        [],
        [],
    )
    assert certificate.state == "review_recommended"
    assert not certificate.auto_release_eligible
    assert any("3 个独立有效候选家族" in item for item in certificate.release_blockers)


def test_production_release_requires_complete_confidence_audit() -> None:
    certificate = build_quality_certificate(
        [page(consensus_confidence=None)],
        [],
        {"rhythm_issues": [], "tie_issues": [], "slur_issues": []},
        [],
        [],
    )
    assert certificate.state == "review_recommended"
    assert any("缺少小节共识置信度" in item for item in certificate.release_blockers)


def test_production_release_blocks_transaction_rejection_and_high_patch_burden() -> None:
    certificate = build_quality_certificate(
        [page(
            resolved_measure_count=10,
            consensus_pitch_patch_measures=3,
            consensus_rhythm_patch_measures=2,
            consensus_patch_transaction_rejections=1,
        )],
        [],
        {"rhythm_issues": [], "tie_issues": [], "slur_issues": []},
        [],
        [],
    )
    assert not certificate.auto_release_eligible
    assert any("事务" in item for item in certificate.release_blockers)
    assert any("修复负担" in item for item in certificate.release_blockers)


def test_production_release_requires_review_after_event_insertion_or_deletion() -> None:
    certificate = build_quality_certificate(
        [page(consensus_event_presence_inserted_events=1)],
        [],
        {"rhythm_issues": [], "tie_issues": [], "slur_issues": []},
        [],
        [],
    )
    assert not certificate.auto_release_eligible
    assert any("事件插入或删除" in item for item in certificate.release_blockers)


def test_production_release_requires_complete_verified_model_resource_set() -> None:
    certificate = build_quality_certificate(
        [page()],
        [],
        {"rhythm_issues": [], "tie_issues": [], "slur_issues": []},
        [],
        [],
        model_resource_audit={
            "verified": False,
            "expected_count": 26,
            "verified_count": 25,
            "errors": ["articulation_patch_calibrator.json:hash_mismatch"],
        },
    )
    assert not certificate.auto_release_eligible
    assert certificate.state == "review_recommended"
    assert any("模型资源" in item for item in certificate.release_blockers)
