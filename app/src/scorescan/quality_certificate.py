from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

from .models import PageInfo, ReviewIssue
from .variant_family import variant_family

DEFAULT_SELECTION_RISK_PAGE_FLOOR = 0.70


@dataclass(frozen=True)
class QualityCertificate:
    score: float
    state: str
    label: str
    reasons: tuple[str, ...]
    hard_failures: tuple[str, ...]
    pending_review_count: int
    consensus_disagreement_count: int
    auto_release_eligible: bool
    release_blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "state": self.state,
            "label": self.label,
            "reasons": list(self.reasons),
            "hard_failures": list(self.hard_failures),
            "pending_review_count": self.pending_review_count,
            "consensus_disagreement_count": self.consensus_disagreement_count,
            "auto_release_eligible": self.auto_release_eligible,
            "release_blockers": list(self.release_blockers),
        }


def _count(items: object) -> int:
    return len(items) if isinstance(items, list) else 0


def _is_semantically_resolved_page(page: PageInfo) -> bool:
    """Distinguish auditable encoding differences from unresolved musical risk."""
    return (
        page.consensus_semantic_agreement is not None
        and page.consensus_semantic_agreement >= 0.995
        and page.consensus_confidence is not None
        and page.consensus_confidence >= 0.98
        and not page.consensus_unresolved
        and page.consensus_patch_transaction_rejections == 0
        and page.consensus_event_presence_inserted_events == 0
        and page.consensus_event_presence_deleted_events == 0
        and page.notation_coverage_status == "completed"
        and page.notation_potential_omission_count == 0
        and page.notation_unbalanced_structure_count == 0
        and (
            page.semantic_detector_status is None
            or (
                page.semantic_source_audit_status == "completed"
                and page.semantic_source_audit_positional_mismatch_count == 0
            )
        )
    )


def build_quality_certificate(
    pages: Iterable[PageInfo],
    validation_errors: list[str],
    semantic_analysis: dict[str, object],
    review_issues: Iterable[ReviewIssue],
    fallback_pages: list[int],
    model_resource_audit: dict[str, object] | None = None,
) -> QualityCertificate:
    """Produce a conservative, deterministic quality score.

    The score is not a probability of correctness. It is an auditable triage value
    that combines scan quality, XML validity, music-structure checks, candidate
    disagreement and unresolved review items. It is deliberately conservative so a
    structurally valid but weakly supported result is not presented as verified.
    """
    page_list = list(pages)
    issue_list = list(review_issues)
    score = 100.0
    reasons: list[str] = []
    hard_failures: list[str] = []

    if validation_errors:
        penalty = min(75.0, 28.0 * len(validation_errors))
        score -= penalty
        hard_failures.append(f"MusicXML 结构错误 {len(validation_errors)} 项")

    if fallback_pages:
        penalty = min(80.0, 38.0 * len(fallback_pages))
        score -= penalty
        hard_failures.append(f"{len(fallback_pages)} 页使用保底输出")

    model_audit_verified = (
        True
        if model_resource_audit is None
        else bool(isinstance(model_resource_audit, dict) and model_resource_audit.get("verified") is True)
    )
    model_audit_errors = (
        list(model_resource_audit.get("errors") or [])
        if isinstance(model_resource_audit, dict)
        else []
    )
    if not model_audit_verified:
        score -= 18.0
        reasons.append("运行时模型资源完整性未通过")

    rhythm_count = _count(semantic_analysis.get("rhythm_issues"))
    tie_count = _count(semantic_analysis.get("tie_issues"))
    slur_count = _count(semantic_analysis.get("slur_issues"))
    if rhythm_count:
        score -= min(34.0, 7.0 * rhythm_count)
        reasons.append(f"{rhythm_count} 个小节时值疑点")
    if tie_count:
        score -= min(18.0, 4.0 * tie_count)
        reasons.append(f"{tie_count} 个延音线疑点")
    if slur_count:
        score -= min(14.0, 3.0 * slur_count)
        reasons.append(f"{slur_count} 个连音线疑点")

    notation_omissions = sum(page.notation_potential_omission_count for page in page_list)
    notation_unbalanced = sum(page.notation_unbalanced_structure_count for page in page_list)
    notation_audit_missing = [
        page.index
        for page in page_list
        if page.omr_status == "completed"
        and page.notation_coverage_status != "completed"
    ]
    if notation_omissions:
        score -= min(30.0, 4.0 * notation_omissions)
        reasons.append(f"源扫描独立审计发现 {notation_omissions} 个潜在静默漏识别")
    if notation_unbalanced:
        score -= min(12.0, 2.0 * notation_unbalanced)
        reasons.append(f"{notation_unbalanced} 个连音线或发夹结构未闭合")
    if notation_audit_missing:
        score -= min(18.0, 9.0 * len(notation_audit_missing))
        reasons.append(f"{len(notation_audit_missing)} 页未完成源扫描记号覆盖审计")
    semantic_detector_failures = [
        page.index
        for page in page_list
        if page.semantic_detector_status is not None
        and (
            not page.semantic_detector_enabled
            or not page.semantic_detector_accelerator_verified
        )
    ]
    if semantic_detector_failures:
        score -= min(18.0, 9.0 * len(semantic_detector_failures))
        reasons.append(
            f"{len(semantic_detector_failures)} 页未完成门禁式语义记号检测"
        )
    semantic_audit_missing = [
        page.index
        for page in page_list
        if page.omr_status == "completed"
        and page.semantic_detector_status is not None
        and page.semantic_source_audit_status != "completed"
    ]
    semantic_source_omissions = sum(
        page.semantic_source_audit_omission_count for page in page_list
    )
    semantic_source_extraneous = sum(
        page.semantic_source_audit_extraneous_count for page in page_list
    )
    semantic_source_mismatches = sum(
        page.semantic_source_audit_positional_mismatch_count for page in page_list
    )
    if semantic_audit_missing:
        score -= min(20.0, 10.0 * len(semantic_audit_missing))
        reasons.append(
            f"{len(semantic_audit_missing)} 页未完成位置级语义符号审计"
        )
    if semantic_source_omissions:
        score -= min(32.0, 6.0 * semantic_source_omissions)
        reasons.append(
            f"源扫描中有 {semantic_source_omissions} 个高置信符号未写入结果"
        )
    if semantic_source_extraneous:
        score -= min(32.0, 8.0 * semantic_source_extraneous)
        reasons.append(
            f"结果中有 {semantic_source_extraneous} 个变音记号缺少源扫描位置证据"
        )

    semantic_counts = semantic_analysis.get("semantic_issue_counts", {})
    if not isinstance(semantic_counts, dict):
        semantic_counts = {}
    type_duration_count = int(semantic_counts.get("type_duration_mismatch", 0) or 0)
    multi_voice_count = int(semantic_counts.get("multiple_voices", 0) or 0)
    chord_duration_count = int(semantic_counts.get("chord_duration_mismatch", 0) or 0)
    zero_duration_count = int(semantic_counts.get("zero_duration", 0) or 0)
    duplicate_direction_count = int(semantic_counts.get("duplicate_direction", 0) or 0)
    density_outlier_count = int(semantic_counts.get("density_outlier", 0) or 0)
    empty_measure_count = int(semantic_counts.get("empty_measure", 0) or 0)
    pitch_outlier_count = int(semantic_counts.get("pitch_outlier", 0) or 0)
    if zero_duration_count:
        score -= min(45.0, 18.0 * zero_duration_count)
        hard_failures.append(f"{zero_duration_count} 个非倚音事件时值为零")
    if type_duration_count:
        score -= min(18.0, 3.0 * type_duration_count)
        reasons.append(f"{type_duration_count} 个音符类型与时值不一致")
    if multi_voice_count:
        score -= min(16.0, 4.0 * multi_voice_count)
        reasons.append(f"{multi_voice_count} 个小节疑似包含多个节奏声部")
    if chord_duration_count:
        score -= min(12.0, 3.0 * chord_duration_count)
        reasons.append(f"{chord_duration_count} 个和弦内部时值不一致")
    if duplicate_direction_count:
        score -= min(6.0, 1.0 * duplicate_direction_count)
        reasons.append(f"{duplicate_direction_count} 处方向标记可能重复")
    if density_outlier_count:
        score -= min(8.0, 2.0 * density_outlier_count)
        reasons.append(f"{density_outlier_count} 个小节事件密度异常")
    if empty_measure_count:
        score -= min(20.0, 5.0 * empty_measure_count)
        reasons.append(f"{empty_measure_count} 个小节没有可播放事件")
    if pitch_outlier_count:
        score -= min(12.0, 3.0 * pitch_outlier_count)
        reasons.append(f"{pitch_outlier_count} 个音高超出常见分谱范围")

    consensus_disagreements = sum(len(page.consensus_disagreements) for page in page_list)
    consensus_unresolved = sum(len(page.consensus_unresolved) for page in page_list)
    # Majority-backed replacements are useful evidence rather than an error. Only
    # no-majority measures should reduce the score, and their pending review status is
    # handled below so a completed human check can clear the uncertainty.
    if consensus_unresolved:
        # Structural validity is not source fidelity. A page with many
        # no-majority measures must not keep a cosmetic 100/100 score merely
        # because the selected MusicXML is internally valid.
        score -= min(35.0, 2.5 * consensus_unresolved)
        reasons.append(f"{consensus_unresolved} 个小节没有形成严格识别多数")

    low_agreement_pages = [
        page for page in page_list
        if page.consensus_agreement is not None and page.consensus_agreement < 0.82
        and not _is_semantically_resolved_page(page)
    ]
    low_confidence_pages = [
        page for page in page_list
        if page.consensus_confidence is not None and page.consensus_confidence < 0.76
    ]
    if low_agreement_pages:
        score -= min(18.0, 6.0 * len(low_agreement_pages))
        reasons.append(f"{len(low_agreement_pages)} 页候选一致率偏低")
    if low_confidence_pages:
        score -= min(12.0, 4.0 * len(low_confidence_pages))
        reasons.append(f"{len(low_confidence_pages)} 页小节共识置信度偏低")
    low_measure_model_pages = [
        page for page in page_list
        if page.consensus_measure_probability is not None and page.consensus_measure_probability < 0.58
        and not _is_semantically_resolved_page(page)
    ]
    if low_measure_model_pages:
        score -= min(10.0, 3.0 * len(low_measure_model_pages))
        reasons.append(f"{len(low_measure_model_pages)} 页的小节选择模型置信度偏低")
    low_visual_pages = [
        page for page in page_list
        if page.consensus_visual_probability is not None and page.consensus_visual_probability < 0.22
    ]
    if low_visual_pages:
        # Visual compatibility is a secondary density check, not an OMR confidence.
        # Keep its quality-certificate influence deliberately small.
        score -= min(6.0, 1.5 * len(low_visual_pages))
        reasons.append(f"{len(low_visual_pages)} 页的谱面密度与候选事件结构匹配度偏低")

    low_event_pages = [
        page for page in page_list
        if page.consensus_event_probability is not None and page.consensus_event_probability < 0.34
    ]
    if low_event_pages:
        # Event-lattice calibration is peer evidence only; its certificate influence is
        # bounded and never substitutes for a review issue or musical validation.
        score -= min(8.0, 2.0 * len(low_event_pages))
        reasons.append(f"{len(low_event_pages)} 页的逐事件候选一致性偏低")

    low_context_pages = [
        page for page in page_list
        if page.consensus_context_probability is not None and page.consensus_context_probability < 0.28
    ]
    if low_context_pages:
        # Cross-measure context is only a weak continuity prior.  Its certificate
        # influence stays smaller than event and structural evidence.
        score -= min(5.0, 1.25 * len(low_context_pages))
        reasons.append(f"{len(low_context_pages)} 页的跨小节上下文一致性偏低")

    low_ensemble_pages = [
        page for page in page_list
        if page.consensus_ensemble_probability is not None and page.consensus_ensemble_probability < 0.36
    ]
    if low_ensemble_pages:
        # The ensemble meta-calibrator combines existing evidence but remains a bounded
        # risk signal.  A low value should increase review visibility, not declare the
        # transcription wrong by itself.
        score -= min(7.0, 1.75 * len(low_ensemble_pages))
        reasons.append(f"{len(low_ensemble_pages)} 页的综合候选可靠度偏低")

    low_selection_risk_pages = [
        page for page in page_list
        if page.consensus_selection_risk_probability is not None
        and page.consensus_selection_risk_probability < DEFAULT_SELECTION_RISK_PAGE_FLOOR
        and not _is_semantically_resolved_page(page)
    ]
    if low_selection_risk_pages:
        # This is a selective acceptance signal. A low value means ScoreScan refused
        # to trust some automatic replacements; it should surface review, not declare
        # the whole page structurally invalid.
        score -= min(8.0, 2.0 * len(low_selection_risk_pages))
        reasons.append(f"{len(low_selection_risk_pages)} 页的自动替换安全证据偏低")

    pending = [issue for issue in issue_list if issue.status != "resolved"]
    preserved_risks = [issue for issue in issue_list if issue.status == "resolved" and issue.risk_preserved]
    text_pending = sum(issue.category == "music_text" for issue in pending)
    measure_pending = sum(issue.category == "measure_consensus" for issue in pending)
    notation_pending = sum(issue.category == "notation_coverage" for issue in pending)
    if text_pending:
        score -= min(14.0, 1.8 * text_pending)
        reasons.append(f"{text_pending} 个文字或力度疑点待确认")
    if measure_pending:
        score -= min(22.0, 4.0 * measure_pending)
        reasons.append(f"{measure_pending} 个结构分歧小节待检查")
    if notation_pending:
        score -= min(24.0, 6.0 * notation_pending)
        reasons.append(f"{notation_pending} 组源扫描记号覆盖风险待检查")
    if preserved_risks:
        score -= min(8.0, 1.5 * len(preserved_risks))
        reasons.append(f"{len(preserved_risks)} 个已检查位置仍保留识别风险")

    quality_values = [float(page.quality_score) for page in page_list if page.quality_score is not None]
    if quality_values:
        mean_quality = sum(quality_values) / len(quality_values)
        minimum_quality = min(quality_values)
        if mean_quality < 72:
            score -= min(16.0, (72.0 - mean_quality) * 0.45)
            reasons.append(f"平均扫描质量 {mean_quality:.0f}/100")
        if minimum_quality < 45:
            score -= min(12.0, (45.0 - minimum_quality) * 0.40)
            reasons.append(f"最低页面质量 {minimum_quality:.0f}/100")

    event_presence_visual_rejections = sum(
        page.consensus_event_presence_visual_guard_rejections for page in page_list
    )
    if event_presence_visual_rejections:
        score -= min(8.0, 2.0 * event_presence_visual_rejections)
        reasons.append(
            f"{event_presence_visual_rejections} 个事件插入或删除提案未通过源扫描视觉复核"
        )

    patch_transaction_rejections = sum(page.consensus_patch_transaction_rejections for page in page_list)
    if patch_transaction_rejections:
        score -= min(12.0, 4.0 * patch_transaction_rejections)
        reasons.append(f"{patch_transaction_rejections} 个局部修复事务因组合风险被回退")

    high_patch_burden_pages: list[tuple[int, int, int]] = []
    for page in page_list:
        local_patch_count = sum((
            page.consensus_chord_patch_measures,
            page.consensus_tuplet_patch_measures,
            page.consensus_pitch_patch_measures,
            page.consensus_rhythm_patch_measures,
            page.consensus_event_kind_patch_measures,
            page.consensus_attribute_patch_measures,
            page.consensus_slur_patch_measures,
            page.consensus_articulation_patch_measures,
            page.consensus_ornament_patch_measures,
            page.consensus_grace_patch_measures,
            page.consensus_lyric_patch_measures,
            page.consensus_direction_patch_measures,
            page.consensus_barline_patch_measures,
            page.consensus_event_presence_patch_measures,
            page.consensus_cross_tie_patch_boundaries,
        ))
        measure_count = max(page.resolved_measure_count, page.estimated_measure_count, 1)
        ceiling = max(4, math.ceil(measure_count * 0.35))
        if local_patch_count > ceiling:
            high_patch_burden_pages.append((page.index, local_patch_count, measure_count))
    if high_patch_burden_pages:
        score -= min(12.0, 3.0 * len(high_patch_burden_pages))
        reasons.append(f"{len(high_patch_burden_pages)} 页自动局部修复负担偏高")

    invalid_candidates = 0
    total_candidates = 0
    for page in page_list:
        for item in page.recognition_candidates:
            total_candidates += 1
            if not bool(item.get("valid", False)):
                invalid_candidates += 1
    if total_candidates and invalid_candidates:
        ratio = invalid_candidates / total_candidates
        score -= min(10.0, ratio * 10.0)
        if ratio >= 0.5:
            reasons.append("多数预处理候选未通过结构验证")

    score = round(max(0.0, min(100.0, score)), 1)

    # Production auto-release is intentionally stricter than the descriptive quality
    # score.  A result can be structurally usable while still lacking enough
    # independent evidence to be presented as automatically verified.
    release_blockers: list[str] = list(hard_failures)
    if not model_audit_verified:
        release_blockers.append(
            "运行时模型资源完整性未通过"
            + (f"（{len(model_audit_errors)} 项）" if model_audit_errors else "")
        )
    if pending:
        release_blockers.append(f"仍有 {len(pending)} 个待处理复核项")
    if preserved_risks:
        release_blockers.append(f"{len(preserved_risks)} 个已复核位置仍保留风险")
    if rhythm_count or tie_count or slur_count:
        release_blockers.append("仍存在节奏、延音或连音线结构疑点")
    if notation_omissions:
        release_blockers.append(f"源扫描独立审计仍有 {notation_omissions} 个潜在静默漏识别")
    if notation_unbalanced:
        release_blockers.append(f"{notation_unbalanced} 个连音线或发夹结构未闭合")
    if notation_audit_missing:
        release_blockers.append(f"{len(notation_audit_missing)} 页未完成源扫描记号覆盖审计")
    if semantic_detector_failures:
        release_blockers.append(
            f"{len(semantic_detector_failures)} 页未完成门禁式语义记号检测"
        )
    if semantic_audit_missing:
        release_blockers.append(
            f"{len(semantic_audit_missing)} 页未完成位置级语义符号审计"
        )
    if semantic_source_mismatches:
        release_blockers.append(
            f"源扫描与结果仍有 {semantic_source_mismatches} 个位置级语义符号不一致"
        )
    if any((type_duration_count, multi_voice_count, chord_duration_count, zero_duration_count, empty_measure_count, pitch_outlier_count)):
        release_blockers.append("仍存在影响演奏语义的结构问题")
    if consensus_unresolved:
        release_blockers.append(f"{consensus_unresolved} 个小节未形成严格候选多数")
    if patch_transaction_rejections:
        release_blockers.append(f"{patch_transaction_rejections} 个局部修复事务曾因组合风险回退")
    if event_presence_visual_rejections:
        release_blockers.append(
            f"{event_presence_visual_rejections} 个事件存在性提案未通过源扫描视觉复核"
        )
    if high_patch_burden_pages:
        release_blockers.append(f"{len(high_patch_burden_pages)} 页自动局部修复负担超过生产放行上限")

    for page in page_list:
        valid_families = {
            variant_family(str(item.get("variant", "")))
            for item in page.recognition_candidates
            if bool(item.get("valid", False)) and str(item.get("variant", "")).strip()
        }
        if len(valid_families) < 3:
            release_blockers.append(f"第 {page.index} 页少于 3 个独立有效候选家族")
        if page.consensus_agreement is None:
            release_blockers.append(f"第 {page.index} 页缺少候选共识审计")
        elif page.consensus_agreement < 0.90 and not _is_semantically_resolved_page(page):
            release_blockers.append(f"第 {page.index} 页候选一致率低于 90%")
        if page.consensus_confidence is None:
            release_blockers.append(f"第 {page.index} 页缺少小节共识置信度")
        elif page.consensus_confidence < 0.82:
            release_blockers.append(f"第 {page.index} 页小节共识置信度低于 82%")
        if page.consensus_ensemble_probability is None:
            release_blockers.append(f"第 {page.index} 页缺少综合候选可靠度")
        elif page.consensus_ensemble_probability < 0.50:
            release_blockers.append(f"第 {page.index} 页综合候选可靠度低于 50%")
        if page.consensus_unresolved:
            release_blockers.append(f"第 {page.index} 页存在未解决的小节分歧")
        if page.quality_score is None or page.quality_score < 60.0:
            release_blockers.append(f"第 {page.index} 页扫描质量不足生产自动放行")
        if page.consensus_event_presence_inserted_events or page.consensus_event_presence_deleted_events:
            release_blockers.append(f"第 {page.index} 页发生事件插入或删除，必须人工复核")

    if score < 95.0:
        release_blockers.append("质量审计分低于生产自动放行阈值 95")
    release_blockers = list(dict.fromkeys(release_blockers))
    auto_release_eligible = not release_blockers

    if hard_failures or score < 70:
        state = "best_effort"
        label = "最佳结果"
    elif not auto_release_eligible:
        state = "review_recommended"
        label = "建议复核"
    else:
        state = "verified"
        label = "自动校验通过"

    return QualityCertificate(
        score=score,
        state=state,
        label=label,
        reasons=tuple(dict.fromkeys(reasons)),
        hard_failures=tuple(dict.fromkeys(hard_failures)),
        pending_review_count=len(pending),
        consensus_disagreement_count=consensus_disagreements,
        auto_release_eligible=auto_release_eligible,
        release_blockers=tuple(release_blockers),
    )
