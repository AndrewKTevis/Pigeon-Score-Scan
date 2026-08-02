from __future__ import annotations

import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from app.tools import prepare_pdmx_imslp_semantic_corpus as module
from app.tools.acquire_pdmx_imslp_scans import ROLE as SCAN_ROLE
from app.tools.acquire_pdmx_mxl_archive import ROLE as MXL_ROLE
from app.tools.filter_pdmx_imslp_license_candidates import ROLE as FILTER_ROLE
from app.tools.probe_pdmx_imslp_scan_sources import ROLE as SOURCE_ROLE
from scorescan.product_scope import PRODUCTION_BOUNDARY_CONTRACT_VERSION
from scorescan.util import sha256_file


MUSICXML = b"""<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Violin</part-name></score-part></part-list>
  <part id="P1"><measure number="1"><attributes><divisions>1</divisions>
  <time><beats>4</beats><beat-type>4</beat-type></time></attributes>
  <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration>
  <voice>1</voice><type>whole</type></note></measure></part>
</score-partwise>"""


def _mxl() -> bytes:
    output = io.BytesIO()
    container = b"""<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
<rootfiles><rootfile full-path="score.musicxml"/></rootfiles></container>"""
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("score.musicxml", MUSICXML)
    return output.getvalue()


def test_extract_musicxml_rejects_unsafe_zip_member() -> None:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("../score.musicxml", MUSICXML)
        archive.writestr("META-INF/container.xml", b"<container/>")
    with pytest.raises(ValueError, match="unsafe MXL ZIP member"):
        module.extract_musicxml(output.getvalue())


def test_prepare_extracts_only_linked_boundary_valid_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    member = "mxl/1/hash.mxl"
    archive_path = tmp_path / "mxl.tar.gz"
    payload = _mxl()
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo(member)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    monkeypatch.setattr(module, "EXPECTED_BYTES", archive_path.stat().st_size)
    monkeypatch.setattr(module, "EXPECTED_MD5", "fixture-md5")
    acquisition = {
        "role": MXL_ROLE,
        "record_id": module.RECORD_ID,
        "version": module.VERSION,
        "expected_bytes": archive_path.stat().st_size,
        "expected_md5": "fixture-md5",
        "training_authorized": False,
        "asset": {"sha256": sha256_file(archive_path)},
    }
    acquisition_path = tmp_path / "mxl-acquisition.json"
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")
    candidate = {
        "score_id": 123,
        "boundary_hint": "single_staff_solo_candidate",
        "pdmx_mxl_archive_member": member,
        "verified_imslp_source_ids": ["103834"],
        "imslp_source_evidence": [
            {
                "direct_pdf_url": (
                    "https://imslp.org/images/e/ed/"
                    "PMLP14377-fixture.pdf"
                )
            }
        ],
    }
    filtered_path = tmp_path / "filtered.json"
    filtered_path.write_text(
        json.dumps(
            {
                "role": FILTER_ROLE,
                "training_authorized": False,
                "candidates": [candidate],
            }
        ),
        encoding="utf-8",
    )
    scan_path = tmp_path / "scan.pdf"
    scan_path.write_bytes(b"%PDF-fixture")
    source_evidence_path = tmp_path / "source-evidence.json"
    source_evidence_path.write_text(
        json.dumps(
            {
                "role": SOURCE_ROLE,
                "training_authorized": False,
                "verified_candidates": [candidate],
            }
        ),
        encoding="utf-8",
    )
    scan_manifest_path = tmp_path / "scan-manifest.json"
    scan_manifest_path.write_text(
        json.dumps(
            {
                "role": SCAN_ROLE,
                "record_id": module.RECORD_ID,
                "version": module.VERSION,
                "training_authorized": False,
                "transport_authenticated": False,
                "evidence_path": str(source_evidence_path),
                "evidence_sha256": sha256_file(source_evidence_path),
                "assets": [
                    {
                        "pdf_path": str(scan_path),
                        "pdf_sha256": sha256_file(scan_path),
                        "pdf_bytes": scan_path.stat().st_size,
                        "actual_page_count": 1,
                        "imslp_source_id": "103834",
                        "pdmx_score_ids": [123],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report = module.prepare(
        archive_path,
        acquisition_path,
        filtered_path,
        scan_manifest_path,
        tmp_path / "output",
        tmp_path / "manifest.json",
    )
    assert report["accepted_case_count"] == 1
    assert report["accepted_page_count"] == 1
    assert report["training_authorized"] is False
    assert (
        report[
            "whole_work_semantic_development_evaluation_authorized"
        ]
        is False
    )
    assert (
        report["boundary_contract_version"]
        == PRODUCTION_BOUNDARY_CONTRACT_VERSION
    )
    assert report["page_training_labels_authorized"] is False
    case = report["cases"][0]
    assert case["boundary_configuration"] == "single_staff_solo"
    assert Path(case["musicxml_path"]).read_bytes() == MUSICXML
    assert len(case["work_fingerprint"]) == 64
    assert (
        case[
            "whole_work_semantic_development_evaluation_authorized"
        ]
        is False
    )
