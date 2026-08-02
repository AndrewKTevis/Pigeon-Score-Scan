from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.tools import filter_pdmx_imslp_license_candidates as module
from app.tools.acquire_pdmx_license_table import (
    ROLE as LICENSE_ACQUISITION_ROLE,
)
from app.tools.catalog_pdmx_imslp_provenance_candidates import (
    ROLE as CATALOG_ROLE,
)
from scorescan.util import sha256_file


FIELDS = sorted(module.REQUIRED_COLUMNS)


def test_direct_script_entry_point_loads_project_modules() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "filter_pdmx_imslp_license_candidates.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0
    assert "catalog_path" in completed.stdout


def _write_csv(path: Path, *, conflict: bool = False) -> None:
    row = {name: "" for name in FIELDS}
    row.update(
        {
            "metadata": "./metadata/1/123.json",
            "mxl": "./mxl/1/hash.mxl",
            "pdf": "./pdf/1/hash.pdf",
            "license": "cc-zero",
            "license_conflict": str(conflict),
            "subset:no_license_conflict": str(not conflict),
            "subset:all_valid": "True",
            "has_lyrics": "False",
            "n_lyrics": "0",
            "n_tracks": "1",
            "tracks": "0",
            "complexity": "2",
            "n_notes": "100",
            "n_annotations": "10",
        }
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)


def _fixture_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    conflict: bool = False,
) -> tuple[Path, Path, Path, Path]:
    csv_path = tmp_path / "PDMX.csv"
    _write_csv(csv_path, conflict=conflict)
    monkeypatch.setattr(module, "EXPECTED_BYTES", csv_path.stat().st_size)
    monkeypatch.setattr(module, "EXPECTED_MD5", "fixture-md5")
    acquisition = {
        "role": LICENSE_ACQUISITION_ROLE,
        "record_id": module.RECORD_ID,
        "version": module.VERSION,
        "expected_bytes": csv_path.stat().st_size,
        "expected_md5": "fixture-md5",
        "training_authorized": False,
        "asset": {"sha256": sha256_file(csv_path)},
    }
    acquisition_path = tmp_path / "acquisition.json"
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")
    candidate = {
        "score_id": 123,
        "public_domain_metadata_consistent": True,
        "out_of_boundary_instrumentation_terms": [],
        "boundary_hint": "keyboard_candidate",
        "imslp_reverse_lookup_source_ids": ["04047"],
        "pdmx_license_conflict_verified_false": False,
    }
    catalog = {
        "role": CATALOG_ROLE,
        "record_id": module.RECORD_ID,
        "version": module.VERSION,
        "training_authorized": False,
        "candidates": [candidate],
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    return (
        catalog_path,
        csv_path,
        acquisition_path,
        tmp_path / "filtered.json",
    )


def test_filter_accepts_only_audited_in_boundary_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture_paths(tmp_path, monkeypatch)
    report = module.filter_candidates(*paths)
    assert report["csv_row_count"] == 1
    assert report["accepted_candidate_count"] == 1
    assert report["training_authorized"] is False
    assert report["pdmx_rendered_pdf_is_scan_input"] is False
    candidate = report["candidates"][0]
    assert candidate["pdmx_license_conflict_verified_false"] is True
    assert candidate["pdmx_mxl_archive_member"] == "mxl/1/hash.mxl"


def test_filter_rejects_license_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture_paths(tmp_path, monkeypatch, conflict=True)
    report = module.filter_candidates(*paths)
    assert report["accepted_candidate_count"] == 0
    assert report["rejection_counts"]["license_conflict"] == 1
