from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.tools.acquire_nifc_chopin_layout_audit_shortlist import (
    REPORT_ROLE as ACQUISITION_ROLE,
)
from app.tools.audit_active_semantic_dataset_quarantine import (
    _audit_replay_holdout_isolation,
    REPORT_ROLE as QUARANTINE_ROLE,
)
from app.tools.audit_nifc_chopin_subwork_alignment import (
    REPORT_ROLE as ALIGNMENT_ROLE,
)
from app.tools.evaluate_nifc_music_page_classifier import (
    REPORT_ROLE as CLASSIFIER_ROLE,
)
from app.tools.prepare_nifc_chopin_layout_audit_pages import (
    REPORT_ROLE as PREPARATION_ROLE,
)
from app.tools.record_nifc_chopin_primary_visual_review import (
    REPORT_ROLE as PRIMARY_REVIEW_ROLE,
)
from scorescan.util import sha256_file


ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC = ROOT / "training_data" / "external" / "diagnostic"
BENCHMARKS = ROOT / "training_data" / "benchmarks"
pytestmark = pytest.mark.skipif(
    not DIAGNOSTIC.is_dir(),
    reason="optional NIFC diagnostic corpus is not installed",
)


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_never_authorized(report: dict[str, object]) -> None:
    assert report["training_authorized"] is False
    assert report["evaluation_authorized"] is False
    assert report["release_authorized"] is False


def test_acquired_and_prepared_pages_are_identity_bound_and_untransformed() -> None:
    acquisition_path = DIAGNOSTIC / (
        "nifc_chopin_layout_audit_shortlist_v1/candidate_report.json"
    )
    acquisition = _read(acquisition_path)
    assert acquisition["role"] == ACQUISITION_ROLE
    assert acquisition["candidate_count"] == 8
    assert acquisition["unique_reference_count"] == 6
    assert acquisition["raw_child_image_count"] == 56
    assert acquisition["parent_rights_verified_count"] == 8
    assert acquisition["child_membership_verified_count"] == 8
    _assert_never_authorized(acquisition)

    preparation_path = DIAGNOSTIC / (
        "nifc_chopin_layout_audit_pages_v1/page_preparation_report.json"
    )
    preparation = _read(preparation_path)
    assert preparation["role"] == PREPARATION_ROLE
    assert (
        preparation["acquisition_report_sha256"]
        == sha256_file(acquisition_path)
    )
    assert preparation["derived_page_count"] == 102
    for candidate in preparation["candidates"]:
        for page in candidate["pages"]:
            assert page["auto_rotation_applied"] is False
            assert page["auto_deskew_applied"] is False
            assert page["perspective_correction_applied"] is False
            assert page["resampling_applied"] is False
    _assert_never_authorized(preparation)


def test_alignment_failure_and_renderer_warning_are_not_hidden() -> None:
    preparation_path = DIAGNOSTIC / (
        "nifc_chopin_layout_audit_pages_v1/page_preparation_report.json"
    )
    alignment_path = DIAGNOSTIC / (
        "nifc_chopin_subwork_alignment_v1/subwork_alignment_report.json"
    )
    alignment = _read(alignment_path)
    assert alignment["role"] == ALIGNMENT_ROLE
    assert (
        alignment["preparation_report_sha256"]
        == sha256_file(preparation_path)
    )
    assert alignment["automatic_contiguous_alignment_candidate_count"] == 0
    assert alignment["renderer_warning_reference_count"] == 1
    warned = [
        Path(candidate["reference_path"]).name
        for candidate in alignment["candidates"]
        if candidate["renderer_warning_audit"]["warning_count"] > 0
    ]
    assert warned == ["050-1a-Sm-001.krn"]
    _assert_never_authorized(alignment)


def test_primary_review_remains_single_reviewer_and_not_ground_truth() -> None:
    report_path = DIAGNOSTIC / (
        "nifc_chopin_primary_visual_review_v1/"
        "primary_visual_review_report.json"
    )
    report = _read(report_path)
    assert report["role"] == PRIMARY_REVIEW_ROLE
    assert report["mapping_count"] == 8
    assert report["primary_visual_page_role_reviewed_count"] == 102
    assert report["automatic_mapping_agreement_count"] == 0
    assert report["renderer_warning_disqualified_count"] == 1
    assert report["eligible_for_independent_second_range_review_count"] == 7
    assert report["independent_second_review_complete_count"] == 0
    assert report["semantic_symbol_completeness_reviewed_count"] == 0
    _assert_never_authorized(report)


def test_music_page_classifier_uses_leave_one_edition_out_and_stays_diagnostic() -> None:
    report_path = BENCHMARKS / (
        "nifc_music_page_classifier_leave_one_edition_out_v1.json"
    )
    report = _read(report_path)
    assert report["role"] == CLASSIFIER_ROLE
    assert report["split_policy"] == "leave_one_parent_edition_out"
    assert report["page_count"] == 102
    assert report["parent_edition_count"] == 8
    assert report["error_count"] > 0
    assert report["production_threshold_selected"] is False
    _assert_never_authorized(report)


def test_active_semantic_training_quarantines_every_diagnostic_asset() -> None:
    report = _read(
        BENCHMARKS
        / "active_semantic_dataset_diagnostic_quarantine_v1.json"
    )
    assert report["role"] == QUARANTINE_ROLE
    assert report["passed"] is True
    assert report["dataset_count"] == 3
    assert report["diagnostic_report_count"] == 6
    assert report["forbidden_diagnostic_hash_count"] >= 200
    assert any(
        str(item["path"]).endswith(
            "collabscore_pinned_metadata_boundary_audit_v1.json"
        )
        for item in report["diagnostic_reports"]
    )
    assert any(
        str(item["path"]).endswith(
            "beethoven_piano_sonatas_pinned_scan_reference_audit_v1.json"
        )
        for item in report["diagnostic_reports"]
    )
    assert report["training_holdout_source_key_overlap_count"] == 0
    assert report["replay_holdout_source_key_overlap_count"] == 0
    assert report["diagnostic_text_occurrence_count"] == 0
    assert report["diagnostic_hash_occurrence_count"] == 0
    assert report["training_authorized_diagnostic_page_count"] == 0


def test_active_lieder_replay_is_bound_to_work_isolation_evidence() -> None:
    source_list = (
        ROOT
        / "training_data"
        / "prepared"
        / "openscore_lieder_train_1091_sources.txt"
    )
    source_keys = {
        line.strip()
        for line in source_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    evidence = _audit_replay_holdout_isolation(source_keys)

    assert len(source_keys) == 1091
    assert evidence["active_replay_source_key_count"] == 1091
    assert evidence["active_replay_unaudited_source_key_count"] == 0
    assert evidence["holdout_selected_works"] == 395
    assert evidence["replay_works"] == 1474
    assert evidence["work_overlap_count"] == 0
