from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from app.tools.audit_external_real_scan_candidates import audit_catalog
from scorescan.product_scope import PRODUCTION_BOUNDARY_CONTRACT_VERSION


CATALOG = (
    Path(__file__).resolve().parents[2]
    / "training"
    / "external_real_scan_candidates.v1.json"
)
FIXTURE_BENCHMARKS = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "training_metadata"
    / "benchmarks"
)
POLISH_BOUNDARY_CATALOG = (
    FIXTURE_BENCHMARKS
    / "polish_scores_frozen_boundary_scan_candidates_v2.json"
)
COLLABSCORE_AUDIT = (
    FIXTURE_BENCHMARKS
    / "collabscore_pinned_metadata_boundary_audit_v1.json"
)
BEETHOVEN_SONATAS_AUDIT = (
    FIXTURE_BENCHMARKS
    / "beethoven_piano_sonatas_pinned_scan_reference_audit_v1.json"
)
OLIMPIC_FULL_PAGE_AUDIT = (
    FIXTURE_BENCHMARKS
    / "olimpic_full_page_alignment_source_audit_v1.json"
)


def test_checked_in_catalog_has_no_false_release_evidence() -> None:
    report = audit_catalog(json.loads(CATALOG.read_text(encoding="utf-8")))
    assert report["boundary_contract_version"] == (
        PRODUCTION_BOUNDARY_CONTRACT_VERSION
    )
    assert report["candidate_count"] >= 7
    assert report["final_release_evidence_candidate_ids"] == []
    assert report["final_release_evidence_pages"] == 0
    assert report["release_gate_satisfied"] is False
    assert report["research_training_candidate_ids"] == [
        "olimpic-scanned-1.0"
    ]
    assert "polish-scores-boundary-candidates-20260729" not in (
        report["internal_diagnostic_candidate_ids"]
    )


def test_catalog_rejects_noncommercial_product_training_claim() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    changed = deepcopy(catalog)
    smb = next(row for row in changed["candidates"] if row["id"] == "praig-smb-2025")
    smb["distributable_product_training_authorized"] = True
    with pytest.raises(ValueError, match="product training"):
        audit_catalog(changed)


def test_catalog_rejects_incomplete_full_page_release_evidence_claim() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    changed = deepcopy(catalog)
    olimpic = next(
        row for row in changed["candidates"] if row["id"] == "olimpic-scanned-1.0"
    )
    olimpic["final_release_evidence_authorized"] = True
    with pytest.raises(ValueError, match="page_semantic_ground_truth_complete"):
        audit_catalog(changed)


def test_olimpic_candidate_matches_full_page_alignment_audit() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    olimpic = next(
        row
        for row in catalog["candidates"]
        if row["id"] == "olimpic-scanned-1.0"
    )
    audit = json.loads(
        OLIMPIC_FULL_PAGE_AUDIT.read_text(encoding="utf-8")
    )

    assert olimpic["pages_total"] == audit["full_page_png_count"]
    assert olimpic["samples_total"] == audit["published_sample_count"]
    assert olimpic["works_total"] == audit["published_score_group_count"]
    assert audit["mapped_full_page_count"] == 788
    assert audit["mapped_system_count"] == 2961
    assert audit["extra_mapping_sample_count"] == 30
    assert audit["published_dev_test_source_document_overlap_count"] == 0
    assert audit["full_page_semantic_ground_truth_complete"] is False
    assert olimpic["research_training_authorized"] is True
    assert olimpic["distributable_product_training_authorized"] is False
    assert olimpic["final_release_evidence_authorized"] is False


def test_polish_candidate_summary_matches_machine_boundary_audit() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    polish = next(
        row
        for row in catalog["candidates"]
        if row["id"] == "polish-scores-boundary-candidates-20260729"
    )
    boundary = json.loads(
        POLISH_BOUNDARY_CATALOG.read_text(encoding="utf-8")
    )
    assert polish["samples_total"] == boundary["accepted_candidate_count"]
    assert polish["works_total"] == boundary["accepted_work_count"]
    assert boundary["accepted_source_group_count"] == 155
    assert polish["research_training_authorized"] is False
    assert polish["final_release_evidence_authorized"] is False


def test_collabscore_candidate_matches_pinned_pre_download_audit() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    collabscore = next(
        row
        for row in catalog["candidates"]
        if row["id"] == "collabscore-2026"
    )
    audit = json.loads(COLLABSCORE_AUDIT.read_text(encoding="utf-8"))

    assert collabscore["pinned_source_revision"] == audit["source_revision"]
    assert collabscore["pages_total"] == audit["dataset_total_pages"]
    assert collabscore["works_total"] == audit["declared_work_count"]
    assert audit["strict_boundary_accepted_work_count"] == 0
    assert audit["source_images_downloaded"] is False
    assert collabscore["research_training_authorized"] is False
    assert collabscore["internal_diagnostic_evaluation_authorized"] is False
    assert collabscore["final_release_evidence_authorized"] is False


def test_beethoven_candidate_matches_pinned_scan_reference_audit() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    beethoven = next(
        row
        for row in catalog["candidates"]
        if row["id"] == "kernscores-beethoven-piano-sonatas-20260729"
    )
    audit = json.loads(
        BEETHOVEN_SONATAS_AUDIT.read_text(encoding="utf-8")
    )

    assert beethoven["pinned_source_revision"] == audit["source_revision"]
    assert beethoven["pages_total"] == audit[
        "unique_physical_scan_page_count"
    ]
    assert beethoven["works_total"] == audit["sonata_work_count"]
    assert audit["boundary_accepted_humdrum_count"] == audit[
        "humdrum_movement_count"
    ]
    assert audit["source_license_file_present"] is False
    assert beethoven["research_training_authorized"] is False
    assert beethoven["internal_diagnostic_evaluation_authorized"] is False
    assert beethoven["final_release_evidence_authorized"] is False
