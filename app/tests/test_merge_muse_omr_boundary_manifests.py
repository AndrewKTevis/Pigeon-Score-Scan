from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.tools.merge_muse_omr_boundary_manifests import (
    merge_boundary_manifests,
)
from app.tools.muse_omr_contract import SCAN_DEGRADED_IMAGE_ORIGIN
from scorescan.product_scope import PRODUCTION_BOUNDARY_CONTRACT_VERSION


def _manifest(
    root: Path,
    *,
    name: str,
    fingerprint: str,
    shape: str,
    pages: int,
) -> Path:
    directory = root / name
    directory.mkdir()
    reference = directory / "reference.musicxml"
    reference.write_text("<score-partwise/>", encoding="utf-8")
    scan = directory / "scan.pdf"
    scan.write_bytes(b"%PDF-test")
    path = directory / "boundary_manifest.json"
    path.write_text(
        json.dumps(
            {
                "format": 1,
                "boundary_contract_version": (
                    PRODUCTION_BOUNDARY_CONTRACT_VERSION
                ),
                "role": (
                    "external_scan_degraded_development_benchmark_not_training"
                ),
                "source_image_origin": SCAN_DEGRADED_IMAGE_ORIGIN,
                "production_evidence_eligible": False,
                "cases": [
                    {
                        "id": name,
                        "work_fingerprint": fingerprint,
                        "reference": reference.name,
                        "input_pdf": str(scan),
                        "input_pdf_pages": pages,
                        "boundary": {
                            "contract_version": (
                                PRODUCTION_BOUNDARY_CONTRACT_VERSION
                            ),
                            "accepted": True,
                            "score_shape": shape,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_merge_counts_disjoint_documents_and_pages(tmp_path: Path) -> None:
    first = _manifest(
        tmp_path,
        name="first",
        fingerprint="a" * 64,
        shape="single_staff_solo",
        pages=3,
    )
    second = _manifest(
        tmp_path,
        name="second",
        fingerprint="b" * 64,
        shape="keyboard_plus_single_staff_ensemble",
        pages=5,
    )

    report = merge_boundary_manifests(
        [first, second],
        tmp_path / "combined" / "boundary_manifest.json",
    )

    assert report["accepted_submitted_document_count"] == 2
    assert report["accepted_work_count"] == 2
    assert report["accepted_input_page_count"] == 8
    assert report["accepted_input_pages_by_score_configuration"] == {
        "solo_monophonic": 3,
        "piano": 0,
        "monophonic_ensemble": 0,
        "piano_plus_monophonic_ensemble": 5,
    }
    assert report["source_image_origin"] == (
        "synthetic_scan_degraded_render"
    )
    assert report["production_evidence_eligible"] is False
    assert report["production_scope_coverage_complete"] is False
    assert (
        "submitted_scan_page_count"
        in report["development_coverage_against_production_shape_minimum"]
    )


def test_merge_rejects_work_overlap(tmp_path: Path) -> None:
    first = _manifest(
        tmp_path,
        name="first",
        fingerprint="a" * 64,
        shape="single_staff_solo",
        pages=3,
    )
    second = _manifest(
        tmp_path,
        name="second",
        fingerprint="a" * 64,
        shape="keyboard",
        pages=5,
    )

    with pytest.raises(ValueError, match="overlap by independent work"):
        merge_boundary_manifests(
            [first, second],
            tmp_path / "combined" / "boundary_manifest.json",
        )


def test_merge_rejects_stale_boundary_contract(tmp_path: Path) -> None:
    first = _manifest(
        tmp_path,
        name="first",
        fingerprint="a" * 64,
        shape="single_staff_solo",
        pages=3,
    )
    second = _manifest(
        tmp_path,
        name="second",
        fingerprint="b" * 64,
        shape="keyboard",
        pages=5,
    )
    payload = json.loads(first.read_text(encoding="utf-8"))
    payload["boundary_contract_version"] = "obsolete"
    first.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid boundary manifest"):
        merge_boundary_manifests(
            [first, second],
            tmp_path / "combined" / "boundary_manifest.json",
        )
