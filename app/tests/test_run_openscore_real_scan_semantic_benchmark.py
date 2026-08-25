from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz
import pytest

from app.tools.prepare_openscore_real_scan_semantic_corpus import (
    ROLE as SEMANTIC_CORPUS_ROLE,
)
from app.tools.prepare_pdmx_imslp_semantic_corpus import (
    ROLE as PDMX_SEMANTIC_CORPUS_ROLE,
)
from app.tools.run_openscore_real_scan_semantic_benchmark import (
    ROLE,
    _validated_cases,
    run_benchmark,
)
from scorescan.product_scope import PRODUCTION_BOUNDARY_CONTRACT_VERSION
from scorescan.util import sha256_file


MUSICXML = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Violin</part-name></score-part></part-list>
  <part id="P1"><measure number="1"><attributes><divisions>1</divisions>
    <key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time>
    <clef><sign>G</sign><line>2</line></clef></attributes>
    <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration>
    <voice>1</voice><type>whole</type></note></measure></part>
</score-partwise>
"""

NON_KEYBOARD_MULTIVOICE_MUSICXML = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Violin</part-name></score-part></part-list>
  <part id="P1"><measure number="1"><attributes><divisions>1</divisions>
    <key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time>
    <clef><sign>G</sign><line>2</line></clef></attributes>
    <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration>
      <voice>1</voice><type>whole</type></note>
    <backup><duration>4</duration></backup>
    <note><pitch><step>E</step><octave>4</octave></pitch><duration>4</duration>
      <voice>2</voice><type>whole</type></note>
  </measure></part>
</score-partwise>
"""


@dataclass
class FakeJob:
    id: str
    result_musicxml: str
    status: str = "completed"
    error: str = ""
    quality_state: str = "pass"
    quality_score: float = 1.0
    warnings: list[str] = field(default_factory=list)
    review_issues: list[str] = field(default_factory=list)
    report_path: str | None = None


class FakeManager:
    result: Path

    def __init__(self, _settings: object) -> None:
        self.job = FakeJob("fixture-job", str(self.result))

    def create_job(
        self,
        _paths: list[Path],
        _names: list[str],
    ) -> FakeJob:
        return self.job

    def get(self, _job_id: str) -> FakeJob:
        return self.job

    def remove(self, _job_id: str) -> None:
        return None


def _write_pdf(path: Path) -> None:
    document = fitz.open()
    document.new_page(width=595, height=842)
    document.save(path)
    document.close()


def test_direct_script_entry_point_loads_project_modules() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "run_openscore_real_scan_semantic_benchmark.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0
    assert "semantic_manifest" in completed.stdout


def test_runner_preserves_split_and_cannot_emit_release_gate(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "scan.pdf"
    _write_pdf(pdf)
    reference = tmp_path / "reference.musicxml"
    reference.write_text(MUSICXML, encoding="utf-8")
    candidate = tmp_path / "candidate.musicxml"
    candidate.write_text(MUSICXML, encoding="utf-8")
    FakeManager.result = candidate
    manifest = {
        "role": SEMANTIC_CORPUS_ROLE,
        "boundary_contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
        "whole_work_semantic_development_evaluation_authorized": True,
        "training_authorized": False,
        "page_training_labels_authorized": False,
        "release_evaluation_authorized": False,
        "release_authorized": False,
        "independent_holdout": False,
        "cases": [
            {
                "id": "fixture",
                "semantic_split": "calibration",
                "split_role": (
                    "calibration_split_semantic_development_only"
                ),
                "work_fingerprint": "a" * 64,
                "input_pdf": str(pdf),
                "input_pdf_sha256": sha256_file(pdf),
                "input_pdf_pages": 1,
                "reference": reference.name,
                "reference_sha256": sha256_file(reference),
                "whole_work_semantic_development_evaluation_authorized": (
                    True
                ),
                "page_training_labels_authorized": False,
                "boundary_identity_consistent": True,
            }
        ],
    }
    manifest_path = tmp_path / "semantic_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "output"
    report = run_benchmark(
        manifest_path,
        tmp_path,
        output,
        manager_factory=FakeManager,
        timeout_seconds=5,
    )
    assert report["role"] == ROLE
    assert (
        report["boundary_contract_version"]
        == PRODUCTION_BOUNDARY_CONTRACT_VERSION
    )
    assert report["completed_case_count"] == 1
    assert report["aggregate"]["utility_score"] == 1.0
    assert set(report["aggregate_by_semantic_split"]) == {"calibration"}
    assert report["cases"][0]["semantic_split"] == "calibration"
    assert report["training_authorized"] is False
    assert report["release_evaluation_authorized"] is False
    assert report["release_gate_evaluated"] is False
    assert report["independent_holdout"] is False


def test_runner_accepts_multiple_cases_from_one_pdmx_work_group(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "scan.pdf"
    _write_pdf(pdf)
    reference = tmp_path / "reference.musicxml"
    reference.write_text(MUSICXML, encoding="utf-8")
    candidate = tmp_path / "candidate.musicxml"
    candidate.write_text(MUSICXML, encoding="utf-8")
    FakeManager.result = candidate
    base_case = {
        "semantic_split": "train",
        "split_role": "training_split_semantic_development_only",
        "work_fingerprint": "b" * 64,
        "input_pdf": str(pdf),
        "input_pdf_sha256": sha256_file(pdf),
        "input_pdf_pages": 1,
        "reference": reference.name,
        "reference_sha256": sha256_file(reference),
        "whole_work_semantic_development_evaluation_authorized": True,
        "page_training_labels_authorized": False,
        "page_level_training_authorized": False,
        "page_level_release_evaluation_authorized": False,
        "independent_release_evaluation_authorized": False,
        "boundary_identity_consistent": True,
        "exact_scan_to_semantic_alignment_verified": True,
    }
    manifest = {
        "role": PDMX_SEMANTIC_CORPUS_ROLE,
        "boundary_contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
        "whole_work_semantic_development_evaluation_authorized": True,
        "training_authorized": False,
        "page_training_labels_authorized": False,
        "page_level_training_authorized": False,
        "page_level_release_evaluation_authorized": False,
        "release_evaluation_authorized": False,
        "release_authorized": False,
        "independent_holdout": False,
        "independent_release_evaluation_authorized": False,
        "cases": [
            {**base_case, "id": "movement-1"},
            {**base_case, "id": "movement-2"},
        ],
    }
    manifest_path = tmp_path / "semantic_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = run_benchmark(
        manifest_path,
        tmp_path,
        tmp_path / "output",
        manager_factory=FakeManager,
        timeout_seconds=5,
    )
    assert report["completed_case_count"] == 2
    assert report["requested_work_count"] == 1


def test_runner_rejects_stale_boundary_contract(tmp_path: Path) -> None:
    manifest_path = tmp_path / "semantic_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "role": SEMANTIC_CORPUS_ROLE,
                "boundary_contract_version": "obsolete-boundary",
                "whole_work_semantic_development_evaluation_authorized": True,
                "training_authorized": False,
                "page_training_labels_authorized": False,
                "release_evaluation_authorized": False,
                "release_authorized": False,
                "independent_holdout": False,
                "cases": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="unexpected semantic development manifest contract",
    ):
        _validated_cases(
            manifest_path,
            splits=None,
            case_ids=None,
            limit=None,
        )


def test_runner_rechecks_reference_boundary_before_starting_pipeline(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "scan.pdf"
    _write_pdf(pdf)
    reference = tmp_path / "stale-accepted-reference.musicxml"
    reference.write_text(
        NON_KEYBOARD_MULTIVOICE_MUSICXML,
        encoding="utf-8",
    )
    manifest = {
        "role": SEMANTIC_CORPUS_ROLE,
        "boundary_contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
        "whole_work_semantic_development_evaluation_authorized": True,
        "training_authorized": False,
        "page_training_labels_authorized": False,
        "release_evaluation_authorized": False,
        "release_authorized": False,
        "independent_holdout": False,
        "cases": [
            {
                "id": "stale-accepted",
                "semantic_split": "test",
                "split_role": "test_split_semantic_development_only",
                "work_fingerprint": "d" * 64,
                "input_pdf": str(pdf),
                "input_pdf_sha256": sha256_file(pdf),
                "input_pdf_pages": 1,
                "reference": reference.name,
                "reference_sha256": sha256_file(reference),
                "whole_work_semantic_development_evaluation_authorized": True,
                "page_training_labels_authorized": False,
                "boundary_identity_consistent": True,
            }
        ],
    }
    manifest_path = tmp_path / "semantic_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=(
            "failed current product-boundary preflight: "
            "more_than_one_independent_voice_per_non_keyboard_staff"
        ),
    ):
        _validated_cases(
            manifest_path,
            splits=None,
            case_ids=None,
            limit=None,
        )


def test_runner_rejects_unaligned_pdmx_whole_score_reference(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "complete-work.pdf"
    _write_pdf(pdf)
    reference = tmp_path / "single-movement.musicxml"
    reference.write_text(MUSICXML, encoding="utf-8")
    manifest = {
        "role": PDMX_SEMANTIC_CORPUS_ROLE,
        "boundary_contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
        "whole_work_semantic_development_evaluation_authorized": True,
        "training_authorized": False,
        "page_training_labels_authorized": False,
        "page_level_training_authorized": False,
        "page_level_release_evaluation_authorized": False,
        "release_evaluation_authorized": False,
        "release_authorized": False,
        "independent_holdout": False,
        "independent_release_evaluation_authorized": False,
        "cases": [
            {
                "id": "unaligned",
                "semantic_split": "test",
                "split_role": "test_split_semantic_development_only",
                "work_fingerprint": "c" * 64,
                "input_pdf": str(pdf),
                "input_pdf_sha256": sha256_file(pdf),
                "input_pdf_pages": 1,
                "reference": reference.name,
                "reference_sha256": sha256_file(reference),
                "whole_work_semantic_development_evaluation_authorized": True,
                "page_training_labels_authorized": False,
                "page_level_training_authorized": False,
                "page_level_release_evaluation_authorized": False,
                "independent_release_evaluation_authorized": False,
                "boundary_identity_consistent": True,
                "exact_scan_to_semantic_alignment_verified": False,
            }
        ],
    }
    manifest_path = tmp_path / "semantic_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="exact scan-to-reference alignment",
    ):
        run_benchmark(
            manifest_path,
            tmp_path,
            tmp_path / "output",
            manager_factory=FakeManager,
            timeout_seconds=5,
        )
