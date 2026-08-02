from __future__ import annotations

from pathlib import Path

import pytest

from app.tools.probe_polish_scores_scan_rights import (
    build_report,
    metadata_url,
    parse_rights,
    probe_source,
)


CC_BY_MODS = b"""\
<mods xmlns="http://www.loc.gov/mods/v3">
  <accessCondition type="restriction on access">
    CC BY 4.0 (Creative Commons Attribution 4.0 International)
  </accessCondition>
</mods>
"""
AMBIGUOUS_MODS = b"""\
<mods xmlns="http://www.loc.gov/mods/v3">
  <accessCondition>Treasury of the Library 123</accessCondition>
</mods>
"""


def test_metadata_url_accepts_only_expected_nifc_pdf_urls() -> None:
    assert metadata_url(
        "https://repozytorium.nifc.pl/islandora/object/nifc%3A212915/datastream/PDF/view"
    ).endswith("nifc%3A212915/datastream/MODS/view")
    with pytest.raises(ValueError, match="unsupported"):
        metadata_url("https://example.test/scan.pdf")


def test_parse_rights_requires_explicit_cc_by_4() -> None:
    assert parse_rights(CC_BY_MODS) == (
        ["CC BY 4.0 (Creative Commons Attribution 4.0 International)"],
        True,
    )
    assert parse_rights(AMBIGUOUS_MODS) == (
        ["Treasury of the Library 123"],
        False,
    )


def test_probe_source_records_hash_and_rejects_ambiguous_rights() -> None:
    source = probe_source(
        "https://repozytorium.nifc.pl/islandora/object/nifc%3A1/datastream/PDF/view",
        timeout_seconds=5,
        fetcher=lambda _url: (
            AMBIGUOUS_MODS,
            {
                "content_type": "application/xml",
                "etag": "fixture",
                "last_modified": "today",
            },
        ),
    )
    assert source["status"] == "rights_missing_or_not_cc_by_4"
    assert source["scan_asset_cc_by_4_verified"] is False
    assert len(source["metadata_sha256"]) == 64


def test_build_report_keeps_rights_and_benchmark_authority_separate(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text("{}", encoding="utf-8")
    pdf_url = (
        "https://repozytorium.nifc.pl/islandora/object/"
        "nifc%3A1/datastream/PDF/view"
    )
    catalog = {
        "role": "candidate_catalog_not_training_or_evaluation",
        "training_authorized": False,
        "revision": "fixture",
        "cases": [
            {
                "accepted": True,
                "scan_url": (
                    "https://polish.musicsources.pl/pl/lokalizacje/"
                    "galeria/druki-muzyczne/1/2"
                ),
                "scan_pdf_url": pdf_url,
            }
        ],
    }

    def factory(_timeout: float):
        return lambda _url: (CC_BY_MODS, {"content_type": "application/xml"})

    report = build_report(
        catalog,
        catalog_path=catalog_path,
        workers=1,
        timeout_seconds=5,
        fetcher_factory=factory,
    )
    assert report["verified_cc_by_4_source_count"] == 1
    assert report["strict_catalog_candidates_with_verified_scan_rights"] == 1
    assert report["training_authorized"] is False
    assert report["evaluation_authorized"] is False
    assert report["release_authorized"] is False
