from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from app.tools import probe_archive_pdmx_imslp_mirrors as module
from app.tools.probe_pdmx_imslp_scan_sources import ROLE as EVIDENCE_ROLE


def test_direct_script_entry_point_loads_project_modules() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "probe_archive_pdmx_imslp_mirrors.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0
    assert "evidence_path" in completed.stdout


def test_report_preserves_no_authorization_without_exact_mirror(
    tmp_path: Path,
    monkeypatch,
) -> None:
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text("{}", encoding="utf-8")
    evidence = {
        "role": EVIDENCE_ROLE,
        "version": "v9",
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_authorized": False,
        "verified_candidates": [
            {
                "score_id": 123,
                "artist_name": "Ludwig van Beethoven",
                "title": "Für Elise, WoO 59",
                "boundary_hint": "keyboard_candidate",
                "pdmx_mxl_archive_member": "mxl/1/hash.mxl",
                "imslp_source_evidence": [
                    {
                        "imslp_source_id": "103834",
                        "direct_pdf_url": (
                            "https://imslp.org/images/a/a0/file.pdf"
                        ),
                        "page_count": 6,
                    }
                ],
            }
        ],
    }

    def fake_probe(candidate, *, searcher, metadata_fetcher):
        return {
            **candidate,
            "status": "no_unique_exact_archive_mirror",
        }

    monkeypatch.setattr(module, "probe_candidate", fake_probe)
    report = module.build_report(
        evidence,
        evidence_path=evidence_path,
        timeout_seconds=10,
        minimum_interval_seconds=0,
        searcher=lambda _query: [],
        metadata_fetcher=lambda _identifier: {},
    )
    assert report["candidate_count"] == 1
    assert report["exact_mirror_candidate_count"] == 0
    assert report["training_authorized"] is False
