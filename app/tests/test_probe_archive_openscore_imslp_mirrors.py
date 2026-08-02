from __future__ import annotations

from pathlib import Path

from app.tools.probe_archive_openscore_imslp_mirrors import (
    archive_query,
    build_report,
    exact_mirror_file,
)
from app.tools.probe_openscore_imslp_scan_sources import (
    ROLE as IMSLP_EVIDENCE_ROLE,
)


def _candidate() -> dict[str, object]:
    return {
        "imslp_source_id": "04047",
        "source_identity_verified": True,
        "composer": "Franz Schubert",
        "work_title": "String Quartet in D minor, D.810, No.14",
        "work_fingerprint": "a" * 64,
        "direct_pdf_url": (
            "https://imslp.org/images/6/67/"
            "SchubertStringQuartetNo14.pdf"
        ),
    }


def _item() -> dict[str, object]:
    return {
        "metadata": {
            "identifier": "imslp-quartet-no14-d810-schubert-franz",
            "title": "String Quartet No.14, D.810",
            "creator": "Schubert, Franz",
            "date": "1826",
            "collection": ["imslp", "patron-library-collection"],
        },
        "files": [
            {
                "name": "SchubertStringQuartetNo14.pdf",
                "source": "original",
                "format": "Image Container PDF",
                "size": "7186294",
                "md5": "edb1b64a615550bbd3ac8a6fd18cb446",
                "sha1": "6757e40bcd3340594927b6a68812f9f33d1fb80b",
            }
        ],
    }


def test_archive_query_uses_composer_and_work_identity() -> None:
    query = archive_query(_candidate())
    assert "collection:imslp" in query
    assert 'creator:"schubert"' in query
    assert 'title:"810"' in query


def test_exact_mirror_requires_original_filename_and_imslp_collection() -> None:
    match = exact_mirror_file(_candidate(), _item())
    assert match is not None
    assert match["original_bytes"] == 7_186_294
    assert match["download_url"].endswith(
        "/SchubertStringQuartetNo14.pdf"
    )
    wrong = _item()
    wrong["files"][0]["name"] = "different.pdf"
    assert exact_mirror_file(_candidate(), wrong) is None


def test_report_never_authorizes_undownloaded_mirror(
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text("{}", encoding="utf-8")
    evidence = {
        "role": IMSLP_EVIDENCE_ROLE,
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_authorized": False,
        "verified_candidates": [_candidate()],
    }

    report = build_report(
        evidence,
        evidence_path=evidence_path,
        timeout_seconds=5,
        minimum_interval_seconds=0,
        searcher=lambda _query: [
            {
                "identifier": (
                    "imslp-quartet-no14-d810-schubert-franz"
                )
            }
        ],
        metadata_fetcher=lambda _identifier: _item(),
    )
    assert report["exact_mirror_candidate_count"] == 1
    assert report["training_authorized"] is False
    assert report["evaluation_authorized"] is False
    assert report["release_authorized"] is False
