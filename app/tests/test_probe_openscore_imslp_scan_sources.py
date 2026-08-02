from __future__ import annotations

from pathlib import Path

from app.tools.probe_openscore_imslp_scan_sources import (
    build_report,
    candidate_identity_matches,
    parse_source_page,
)


VERIFIED_HTML = b"""\
<html><head><title>String Quartet No.1, Op.35 (Indy, Vincent d') - IMSLP</title></head><body>
<div class="we">
  <div id="IMSLP17471" class="we_file_first we_fileblock_1">
    <div class="we_file_download">
      <b><a><span>Complete Score</span></a></b>
      <a href="/images/5/59/source.pdf">*</a>
      <a>#17471</a> - 8.03MB, 61 pp.
    </div>
    <div class="we_file_info">PDF scanned by Example Library</div>
  </div>
  <table><tr><th>Copyright</th><td>
    <a href="/wiki/IMSLP:Public_Domain">Public Domain</a>
  </td></tr></table>
</div>
</body></html>
"""


def test_parse_source_page_requires_printed_scan_and_public_domain() -> None:
    source = parse_source_page(
        VERIFIED_HTML,
        imslp_source_id="17471",
        final_url="https://imslp.org/wiki/Work#IMSLP17471",
    )
    assert source["status"] == "verified_public_domain_printed_scan"
    assert source["page_count"] == 61
    assert source["page_title"].startswith("String Quartet No.1")
    assert source["direct_pdf_url"] == (
        "https://imslp.org/images/5/59/source.pdf"
    )


def test_parse_source_page_rejects_typeset_file() -> None:
    source = parse_source_page(
        VERIFIED_HTML.replace(
            b"PDF scanned by Example Library",
            b"PDF typeset by Example Editor",
        ),
        imslp_source_id="17471",
        final_url="https://imslp.org/wiki/Work#IMSLP17471",
    )
    assert source["verified_public_domain_printed_scan"] is False
    assert "file_not_explicitly_printed_pdf_scan" in source["reasons"]


def test_candidate_identity_rejects_wrong_work_even_for_valid_scan() -> None:
    candidate = {
        "composer": "Franz Schubert",
        "work_title": "String Quartet in C minor, D.703, No.12",
    }
    wrong = {
        "page_title": "Piano Sonata No.24, Op.78 (Beethoven, Ludwig van)"
    }
    right = {
        "page_title": "String Quartet in C minor, D.703 (Schubert, Franz)"
    }
    assert candidate_identity_matches(candidate, wrong) is False
    assert candidate_identity_matches(candidate, right) is True


def test_candidate_identity_accepts_named_four_part_work_without_quartet_word() -> None:
    candidate = {
        "composer": "Ludwig van Beethoven",
        "work_title": "Grosse Fuge in B-flat major, Op.133",
    }
    source = {
        "page_title": "Große Fuge, Op.133 (Beethoven, Ludwig van)"
    }
    assert candidate_identity_matches(candidate, source) is True


def test_candidate_identity_rejects_same_first_name_wrong_composer() -> None:
    candidate = {
        "composer": "Franz Schubert",
        "work_title": "String Quartet No.12, D.703",
    }
    wrong_composer = {
        "page_title": "String Quartet No.12, Op.96 (Krommer, Franz)"
    }
    assert candidate_identity_matches(candidate, wrong_composer) is False


def test_report_does_not_authorize_unaligned_sources(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text("{}", encoding="utf-8")
    candidate = {
        "imslp_source_id": "17471",
        "work_fingerprint": "a" * 64,
        "composer": "Vincent d'Indy",
        "work_title": "String Quartet No.1, Op.35",
    }
    catalog = {
        "role": "imslp_scan_candidate_catalog_not_training_or_evaluation",
        "revision": "fixture",
        "training_authorized": False,
        "accepted_candidates": [candidate],
    }

    def factory(_timeout: float):
        return lambda _url: (
            VERIFIED_HTML,
            "https://imslp.org/wiki/Work#IMSLP17471",
            {"content_type": "text/html"},
        )

    report = build_report(
        catalog,
        catalog_path=catalog_path,
        workers=1,
        timeout_seconds=5,
        fetcher_factory=factory,
    )
    assert report["verified_source_count"] == 1
    assert report["verified_candidate_count"] == 1
    assert report["identity_mismatch_candidate_count"] == 0
    assert report["training_authorized"] is False
    assert report["evaluation_authorized"] is False
    assert report["release_authorized"] is False


def test_report_reuses_legacy_evidence_title_from_final_url(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text("{}", encoding="utf-8")
    candidate = {
        "imslp_source_id": "17471",
        "work_fingerprint": "a" * 64,
        "composer": "Vincent d'Indy",
        "work_title": "String Quartet No.1, Op.35",
    }
    catalog = {
        "role": "imslp_scan_candidate_catalog_not_training_or_evaluation",
        "revision": "fixture",
        "training_authorized": False,
        "accepted_candidates": [candidate],
    }
    legacy_source = {
        "imslp_source_id": "17471",
        "status": "verified_public_domain_printed_scan",
        "verified_public_domain_printed_scan": True,
        "final_page_url": (
            "https://imslp.org/wiki/"
            "String_Quartet_No.1,_Op.35_(Indy,_Vincent_d')#IMSLP17471"
        ),
        "page_count": 61,
        "direct_pdf_url": "https://imslp.org/images/5/59/source.pdf",
    }

    def unexpected_factory(_timeout: float):
        raise AssertionError("valid legacy evidence must not be fetched again")

    report = build_report(
        catalog,
        catalog_path=catalog_path,
        workers=1,
        timeout_seconds=5,
        prior_sources=[legacy_source],
        fetcher_factory=unexpected_factory,
    )
    assert report["verified_candidate_count"] == 1
    assert report["sources"][0]["page_title_source"] == "final_redirect_url"


def test_report_offline_reclassification_reuses_failed_sources(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text("{}", encoding="utf-8")
    catalog = {
        "role": "imslp_scan_candidate_catalog_not_training_or_evaluation",
        "revision": "fixture",
        "training_authorized": False,
        "accepted_candidates": [
            {
                "imslp_source_id": "1",
                "work_fingerprint": "a" * 64,
                "composer": "Franz Schubert",
                "work_title": "String Quartet No.1",
            }
        ],
    }
    failed_source = {
        "imslp_source_id": "1",
        "status": "fetch_or_parse_failed",
        "verified_public_domain_printed_scan": False,
        "reasons": ["fetch_or_parse_failed"],
    }

    def unexpected_factory(_timeout: float):
        raise AssertionError("offline reclassification must not fetch")

    report = build_report(
        catalog,
        catalog_path=catalog_path,
        workers=1,
        timeout_seconds=5,
        prior_sources=[failed_source],
        retry_failed_sources=False,
        fetcher_factory=unexpected_factory,
    )
    assert report["source_count"] == 1
    assert report["failed_source_count"] == 1
