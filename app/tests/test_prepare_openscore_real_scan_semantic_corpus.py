from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pymupdf as fitz

from app.tools.acquire_openscore_imslp_scans import ROLE as BYTE_MANIFEST_ROLE
from app.tools.catalog_openscore_imslp_scan_candidates import (
    LICENSE_SHA256,
    REVISION,
)
from app.tools.prepare_openscore_real_scan_semantic_corpus import (
    ROLE,
    prepare_corpus,
)
from scorescan.product_scope import PRODUCTION_BOUNDARY_CONTRACT_VERSION


def _write_pdf(path: Path) -> None:
    document = fitz.open()
    document.new_page(width=595, height=842)
    document.save(path)
    document.close()


def _write_ensemble_musicxml(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1"><part-name>Violin I</part-name></score-part>
    <score-part id="P2"><part-name>Violin II</part-name></score-part>
  </part-list>
  <part id="P1"><measure number="1"><attributes><divisions>1</divisions>
    <key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time>
    <clef><sign>G</sign><line>2</line></clef></attributes>
    <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration>
    <voice>1</voice><type>whole</type></note></measure></part>
  <part id="P2"><measure number="1"><attributes><divisions>1</divisions>
    <key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time>
    <clef><sign>G</sign><line>2</line></clef></attributes>
    <note><pitch><step>G</step><octave>4</octave></pitch><duration>4</duration>
    <voice>1</voice><type>whole</type></note></measure></part>
</score-partwise>
""",
        encoding="utf-8",
    )


def test_direct_script_entry_point_loads_project_modules() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "prepare_openscore_real_scan_semantic_corpus.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0
    assert "byte_manifest_path" in completed.stdout


def test_full_scan_queue_preserves_development_only_authorization() -> None:
    queue = (
        Path(__file__).resolve().parents[2]
        / "training"
        / "queue_openscore_imslp_real_scan_preparation.ps1"
    ).read_text(encoding="utf-8")

    assert "[int]$bytes.source_count -ne 6" in queue
    assert "[int]$bytes.page_count -ne 209" in queue
    assert "$bytes.archive_mirror_required -ne $true" in queue
    assert (
        "$semantic.boundary_contract_version -ne $boundaryContract"
        in queue
    )
    assert "$semantic.boundary_audit_complete -ne $true" in queue
    assert "[int]$semantic.source_work_count -ne 6" in queue
    assert "[int]$semantic.source_page_count -ne 209" in queue
    assert "[int]$semantic.case_count -ne 6" not in queue
    assert "[int]$semantic.excluded_source_count -ne 0" not in queue
    assert "$semantic.page_training_labels_authorized -ne $false" in queue
    assert "$semantic.release_evaluation_authorized -ne $false" in queue
    assert "$semantic.independent_holdout -ne $false" in queue


def test_prepare_corpus_preserves_split_and_never_authorizes_labels(
    tmp_path: Path,
) -> None:
    score_root = tmp_path / "corpus"
    score_path = score_root / "scores" / "Composer" / "Work" / "score.mscx"
    score_path.parent.mkdir(parents=True)
    score_path.write_text("<museScore/>", encoding="utf-8")
    pinned_license = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "licenses"
        / "openscore-string-quartets-LICENSE.txt"
    )
    shutil.copyfile(pinned_license, score_root / "LICENSE.txt")
    assert hashlib.sha256(
        (score_root / "LICENSE.txt").read_bytes()
    ).hexdigest() == LICENSE_SHA256

    pdf_path = tmp_path / "scan.pdf"
    _write_pdf(pdf_path)
    pdf_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    score_hash = hashlib.sha256(score_path.read_bytes()).hexdigest()
    work_fingerprint = "a" * 64
    manifest = {
        "role": BYTE_MANIFEST_ROLE,
        "revision": REVISION,
        "license_sha256": LICENSE_SHA256,
        "archive_mirror_required": True,
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_authorized": False,
        "assets": [
            {
                "imslp_source_id": "4047",
                "retrieval_channel": (
                    "internet_archive_exact_imslp_mirror"
                ),
                "pdf_path": str(pdf_path),
                "pdf_sha256": pdf_hash,
                "actual_page_count": 1,
                "candidate_work_fingerprints": [work_fingerprint],
                "source_page_title": "String Quartet (Composer)",
                "archive_identifier": "imslp-example",
            }
        ],
        "candidates": [
            {
                "source_identity_verified": True,
                "imslp_source_id": "4047",
                "semantic_split": "train",
                "work_fingerprint": work_fingerprint,
                "source_group_fingerprint": "b" * 64,
                "path": "scores/Composer/Work/score.mscx",
                "sha256": score_hash,
                "boundary_configuration": "monophonic_ensemble",
            }
        ],
    }
    manifest_path = tmp_path / "bytes.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    musescore = tmp_path / "MuseScore4.exe"
    musescore.write_bytes(b"fixture")

    def exporter(
        _source: Path,
        destination: Path,
        _musescore: Path,
        *,
        timeout_seconds: int,
    ) -> None:
        assert timeout_seconds == 60
        _write_ensemble_musicxml(destination)

    output = tmp_path / "prepared"
    report = prepare_corpus(
        manifest_path,
        score_root,
        output,
        musescore,
        timeout_seconds=60,
        exporter=exporter,
    )
    assert report["role"] == ROLE
    assert (
        report["boundary_contract_version"]
        == PRODUCTION_BOUNDARY_CONTRACT_VERSION
    )
    assert report["boundary_audit_complete"] is True
    assert report["source_count"] == 1
    assert report["source_work_count"] == 1
    assert report["source_page_count"] == 1
    assert report["case_count"] == 1
    assert report["page_count"] == 1
    assert report["excluded_source_count"] == 0
    assert report["excluded_page_count"] == 0
    assert report["case_count_by_semantic_split"] == {"train": 1}
    assert report["whole_work_semantic_development_evaluation_authorized"]
    assert report["training_authorized"] is False
    assert report["page_training_labels_authorized"] is False
    assert report["release_evaluation_authorized"] is False
    assert report["independent_holdout"] is False
    case = report["cases"][0]
    assert case["semantic_split"] == "train"
    assert case["boundary"]["score_shape"] == "single_staff_ensemble"
    assert case["page_training_labels_authorized"] is False
    assert case["alignment_level"].endswith(
        "no_page_or_measure_alignment"
    )
