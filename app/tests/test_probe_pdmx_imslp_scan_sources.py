from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from app.tools import probe_pdmx_imslp_scan_sources as module
from app.tools.filter_pdmx_imslp_license_candidates import ROLE as FILTER_ROLE


def test_direct_script_entry_point_loads_project_modules() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "probe_pdmx_imslp_scan_sources.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0
    assert "filtered_path" in completed.stdout


def test_probe_requires_scan_evidence_and_work_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    filtered_path = tmp_path / "filtered.json"
    filtered_path.write_text("{}", encoding="utf-8")
    candidates = [
        {
            "score_id": 1,
            "composer_name": "Ludwig van Beethoven",
            "title": "Bagatelle in A minor WoO 59",
            "imslp_reverse_lookup_source_ids": ["103834"],
        },
        {
            "score_id": 2,
            "composer_name": "Joseph Haydn",
            "title": "Piano Sonata",
            "imslp_reverse_lookup_source_ids": ["999999"],
        },
    ]
    filtered = {
        "role": FILTER_ROLE,
        "training_authorized": False,
        "pdmx_rendered_pdf_is_scan_input": False,
        "record_id": "fixture",
        "version": "fixture",
        "candidates": candidates,
    }

    def fake_probe(source_id: str, *, fetcher):
        if source_id == "103834":
            return {
                "imslp_source_id": source_id,
                "verified_public_domain_printed_scan": True,
                "status": "verified_public_domain_printed_scan",
                "page_title": (
                    "Bagatelle in A minor, WoO 59 "
                    "(Beethoven, Ludwig van)"
                ),
                "page_count": 4,
                "direct_pdf_url": "https://imslp.org/images/a/a0/file.pdf",
            }
        return {
            "imslp_source_id": source_id,
            "verified_public_domain_printed_scan": False,
            "status": "rejected_source_evidence",
            "page_title": "Other work (Haydn, Joseph)",
            "page_count": 0,
        }

    monkeypatch.setattr(module, "probe_source", fake_probe)
    report = module.build_report(
        filtered,
        filtered_path=filtered_path,
        workers=2,
        timeout_seconds=10,
        minimum_interval_seconds=0,
        fetcher_factory=lambda _timeout: lambda _url: (b"", "", {}),
    )
    assert report["source_count"] == 2
    assert report["verified_source_count"] == 1
    assert report["verified_candidate_count"] == 1
    assert report["training_authorized"] is False
    assert report["verified_candidates"][0][
        "scan_source_identity_verified"
    ] is True


def test_identity_handles_birth_dates_transliteration_and_translated_opus() -> None:
    source = {
        "page_title": (
            "Veränderungen über einen Walzer, Op.120 "
            "(Beethoven, Ludwig van)"
        )
    }
    candidate = {
        "composer_name": "Ludwig van Beethoven (1770-1827)",
        "artist_name": "Ludwig van Beethoven",
        "title": "33 Variations on a Waltz, Op. 120",
        "file_score_title": "Diabelli Variations",
    }
    assert any(
        module.candidate_identity_matches(identity, source)
        for identity in module._identity_candidates(candidate)
    )

    transliteration_source = {
        "page_title": "Pictures at an Exhibition (Mussorgsky, Modest)"
    }
    transliteration_candidate = {
        "composer_name": "Modeste Moussorgsky",
        "artist_name": "Modest Mussorgsky",
        "title": "Pictures at an Exhibition",
    }
    assert any(
        module.candidate_identity_matches(
            identity,
            transliteration_source,
        )
        for identity in module._identity_candidates(
            transliteration_candidate,
        )
    )
