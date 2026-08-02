from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.tools.project_polish_scores_boundary_rights import (
    CATALOG_ROLE,
    CACHE_ROLE,
    project_rights,
)
from scorescan.util import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _source(url: str, *, verified: bool = False) -> dict[str, object]:
    return {
        "pdf_url": url,
        "metadata_sha256": "a" * 64,
        "status": (
            "verified_cc_by_4"
            if verified
            else "rights_missing_or_not_cc_by_4"
        ),
        "scan_asset_cc_by_4_verified": verified,
    }


def _paths(tmp_path: Path) -> tuple[Path, Path]:
    catalog_path = tmp_path / "catalog.json"
    cache_path = tmp_path / "cache.json"
    catalog_path.write_text("{}", encoding="utf-8")
    cache_path.write_text("{}", encoding="utf-8")
    return catalog_path, cache_path


def test_projects_exact_cached_sources_without_authorizing_use(
    tmp_path: Path,
) -> None:
    catalog_path, cache_path = _paths(tmp_path)
    url = "https://repozytorium.nifc.pl/object/1/PDF"
    catalog = {
        "role": CATALOG_ROLE,
        "training_authorized": False,
        "revision": "fixture",
        "accepted_candidates": [
            {
                "path": "a.krn",
                "work_fingerprint": "b" * 64,
                "source_group_fingerprint": "c" * 64,
                "scan_pdf_url": url,
                "boundary": {"score_shape": "keyboard"},
            }
        ],
    }
    cache = {
        "role": CACHE_ROLE,
        "training_authorized": False,
        "created_at": "today",
        "rights_policy": "explicit only",
        "sources": [_source(url)],
    }
    report = project_rights(
        catalog,
        catalog_path=catalog_path,
        cached_report=cache,
        cached_report_path=cache_path,
    )
    assert report["accepted_candidate_count"] == 1
    assert report["unique_candidate_scan_source_count"] == 1
    assert report["verified_cc_by_4_source_count"] == 0
    assert report["training_authorized"] is False
    assert report["release_authorized"] is False


def test_fails_when_candidate_has_no_cached_source(tmp_path: Path) -> None:
    catalog_path, cache_path = _paths(tmp_path)
    catalog = {
        "role": CATALOG_ROLE,
        "training_authorized": False,
        "accepted_candidates": [
            {
                "scan_pdf_url": "https://example.test/missing",
            }
        ],
    }
    cache = {
        "role": CACHE_ROLE,
        "training_authorized": False,
        "sources": [_source("https://example.test/other")],
    }
    with pytest.raises(ValueError, match="lack cached"):
        project_rights(
            catalog,
            catalog_path=catalog_path,
            cached_report=cache,
            cached_report_path=cache_path,
        )


def test_fails_on_unhashed_cached_metadata(tmp_path: Path) -> None:
    catalog_path, cache_path = _paths(tmp_path)
    url = "https://example.test/source"
    catalog = {
        "role": CATALOG_ROLE,
        "training_authorized": False,
        "accepted_candidates": [{"scan_pdf_url": url}],
    }
    source = _source(url)
    source["metadata_sha256"] = ""
    cache = {
        "role": CACHE_ROLE,
        "training_authorized": False,
        "sources": [source],
    }
    with pytest.raises(ValueError, match="metadata hash"):
        project_rights(
            catalog,
            catalog_path=catalog_path,
            cached_report=cache,
            cached_report_path=cache_path,
        )


def test_checked_in_projection_is_hash_bound_and_still_unauthorized() -> None:
    report_path = (
        PROJECT_ROOT
        / "training_data"
        / "benchmarks"
        / "polish_scores_frozen_boundary_scan_rights_v2.json"
    )
    if not report_path.is_file():
        pytest.skip("optional Polish Scores rights audit is not installed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    catalog_path = Path(report["catalog_path"])
    cache_path = Path(report["cached_rights_report_path"])
    assert report["catalog_sha256"] == sha256_file(catalog_path)
    assert report["cached_rights_report_sha256"] == sha256_file(cache_path)
    assert report["accepted_candidate_count"] == 343
    assert report["unique_candidate_scan_source_count"] == 151
    assert report["candidate_source_group_count"] == 155
    assert report["verified_cc_by_4_source_count"] == 0
    assert report["training_authorized"] is False
    assert report["release_authorized"] is False
