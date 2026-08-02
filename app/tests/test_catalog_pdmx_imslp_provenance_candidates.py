from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from app.tools import catalog_pdmx_imslp_provenance_candidates as module
from app.tools.acquire_pdmx_metadata_archive import ROLE as ACQUISITION_ROLE
from scorescan.util import sha256_file


def test_direct_script_entry_point_loads_project_modules() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "catalog_pdmx_imslp_provenance_candidates.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0
    assert "archive_path" in completed.stdout


def _archive(path: Path) -> None:
    candidates = [
        {
            "data": {
                "is_public_domain": True,
                "score": {
                    "id": 123,
                    "title": "Piano Sonata",
                    "file_score_title": "Piano Sonata",
                    "description": (
                        "Transcribed from "
                        "https://imslp.org/wiki/Special:ReverseLookup/04047"
                    ),
                    "license": "cc-zero",
                    "is_public_domain": True,
                    "parts": 1,
                    "parts_names": ["Piano"],
                    "instruments": [{"name": "Piano"}],
                    "pages_count": 8,
                    "measures": 100,
                    "url": "https://musescore.com/user/1/scores/123",
                },
            }
        },
        {
            "data": {
                "is_public_domain": True,
                "score": {
                    "id": 124,
                    "title": "No source",
                    "description": None,
                    "license": "publicdomain",
                    "is_public_domain": True,
                    "parts": 1,
                    "parts_names": ["Piano"],
                },
            }
        },
    ]
    with tarfile.open(path, "w:gz") as archive:
        for payload in candidates:
            score_id = payload["data"]["score"]["id"]
            encoded = json.dumps(payload).encode()
            info = tarfile.TarInfo(f"metadata/1/{score_id}.json")
            info.size = len(encoded)
            archive.addfile(info, io.BytesIO(encoded))


def test_score_row_requires_consistent_public_domain_metadata() -> None:
    payload = {
        "data": {
            "is_public_domain": False,
            "score": {
                "id": 10,
                "description": (
                    "https://imslp.org/wiki/Special:ReverseLookup/01234"
                ),
                "license": "publicdomain",
                "is_public_domain": True,
                "parts": 1,
                "parts_names": ["Piano"],
            },
        }
    }
    row = module._score_row(payload)
    assert row is not None
    assert row["public_domain_metadata_consistent"] is False
    assert row["boundary_hint"] == "keyboard_candidate"


def test_score_row_rejects_non_public_domain_license() -> None:
    payload = {
        "data": {
            "is_public_domain": True,
            "score": {
                "id": 11,
                "description": (
                    "https://imslp.org/wiki/Special:ReverseLookup/01234"
                ),
                "license": "cc-by-nc",
                "is_public_domain": True,
                "parts": 1,
                "parts_names": ["Piano"],
            },
        }
    }
    row = module._score_row(payload)
    assert row is not None
    assert row["public_domain_metadata_consistent"] is False


def test_catalog_is_provenance_only_and_never_authorizes_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "metadata.tar.gz"
    _archive(archive)
    monkeypatch.setattr(module, "EXPECTED_BYTES", archive.stat().st_size)
    monkeypatch.setattr(module, "EXPECTED_MD5", "fixture-md5")
    acquisition = {
        "role": ACQUISITION_ROLE,
        "record_id": module.RECORD_ID,
        "version": module.VERSION,
        "expected_bytes": archive.stat().st_size,
        "expected_md5": "fixture-md5",
        "training_authorized": False,
        "asset": {"sha256": sha256_file(archive)},
    }
    acquisition_path = tmp_path / "acquisition.json"
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")
    output = tmp_path / "catalog.json"
    report = module.catalog(archive, acquisition_path, output)
    assert report["metadata_member_count"] == 2
    assert report["candidate_count"] == 1
    assert report["public_domain_boundary_candidate_count"] == 1
    assert report["unique_imslp_reverse_lookup_source_count"] == 1
    assert report["training_authorized"] is False
    assert report["evaluation_authorized"] is False
    assert report["release_authorized"] is False
    candidate = report["public_domain_boundary_candidates"][0]
    assert candidate["score_id"] == 123
    assert candidate["imslp_reverse_lookup_source_ids"] == ["04047"]
    assert candidate["pdmx_license_conflict_verified_false"] is False
