from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.tools import acquire_bpsd_piano_scan_corpus as acquisition
from app.tools import prepare_bpsd_boundary_benchmark as module
from scorescan.product_scope import PRODUCTION_BOUNDARY_CONTRACT_VERSION
from scorescan.util import sha256_file


def _case(
    root: Path,
    index: int,
    *,
    eligible: bool,
) -> dict[str, object]:
    work = f"Op{index:03d}"
    work_dir = root / work
    work_dir.mkdir(parents=True)
    scan = work_dir / "scan.pdf"
    reference = work_dir / "reference.musicxml"
    scan.write_bytes(b"%PDF-fixture")
    reference.write_text(
        "<score-partwise/>",
        encoding="utf-8",
    )
    reasons = [] if eligible else ["lyrics"]
    return {
        "work": work,
        "scan_path": str(scan),
        "scan_sha256": sha256_file(scan),
        "pages": 2,
        "reference_musicxml_path": str(reference),
        "reference_musicxml_sha256": sha256_file(reference),
        "boundary": {
            "contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
            "accepted": eligible,
            "score_shape": "keyboard",
            "reasons": reasons,
        },
        "boundary_eligible_alignment_candidate": eligible,
        "reference_quarantined": not eligible,
        "reference_quarantine_reasons": reasons,
        "same_named_work_pair": True,
        "exact_page_measure_alignment_verified": False,
    }


def _manifest(tmp_path: Path) -> Path:
    cases = [
        _case(tmp_path / "corpus", index, eligible=index == 0)
        for index in range(acquisition.EXPECTED_WORKS)
    ]
    path = tmp_path / "acquisition.json"
    path.write_text(
        json.dumps(
            {
                "role": acquisition.ROLE,
                "record_id": acquisition.RECORD_ID,
                "record_license": acquisition.LICENSE_ID,
                "work_count": acquisition.EXPECTED_WORKS,
                "training_authorized": False,
                "release_authorized": False,
                "boundary_eligible_alignment_candidate_count": 1,
                "quarantined_reference_count": (
                    acquisition.EXPECTED_WORKS - 1
                ),
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_direct_script_entry_point_loads_project_modules() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "prepare_bpsd_boundary_benchmark.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0
    assert "acquisition_manifest" in completed.stdout


def test_prepare_rehashes_and_quarantines_boundary_failures(
    tmp_path: Path,
) -> None:
    source = _manifest(tmp_path)
    output = tmp_path / "boundary.json"
    report = module.prepare(source, output)
    assert report["selected_work_count"] == 1
    assert report["selected_physical_scan_page_count"] == 2
    assert report["quarantined_work_count"] == (
        acquisition.EXPECTED_WORKS - 1
    )
    assert report["cases"][0]["id"] == "bpsd-op000"
    assert report["source_image_origin"] == "physical_scan"
    assert report["production_evidence_eligible"] is False
    assert report["selection_used_model_outputs"] is False
    assert report["release_authorized"] is False


def test_prepare_rejects_changed_acquired_bytes(tmp_path: Path) -> None:
    source = _manifest(tmp_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    Path(payload["cases"][0]["scan_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash or schema"):
        module.prepare(source, tmp_path / "boundary.json")
