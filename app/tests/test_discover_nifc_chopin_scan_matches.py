from __future__ import annotations

from app.tools.discover_nifc_chopin_scan_matches import (
    described_collection_item_count,
    described_page_extent,
    edition_number_tokens,
    metadata_match_evidence,
    publication_year_candidates,
    search_result_pids,
)


def test_search_results_are_deduplicated_in_document_order() -> None:
    assert search_result_pids(
        b"""<html><body>
        <a href="/islandora/object/nifc%3A99">breadcrumb ignored</a>
        <div class="islandora-solr-search-result odd">
          <a href="/islandora/object/nifc%3A12">one</a>
          <a href="/islandora/object/nifc:12">duplicate</a>
        </div>
        <div class="islandora-solr-search-result even">
          <a href="/islandora/object/nifc%3A13">two</a>
        </div>
        </body></html>"""
    ) == ["nifc:12", "nifc:13"]


def test_strict_match_requires_publisher_opus_genre_and_cc_by() -> None:
    case = {
        "records": {
            "PPR": "Breitkopf & Härtel",
            "OPS": "Op. 28",
            "rism-genre": "Prelude",
        }
    }
    metadata = {
        "normalized_document_text": (
            "vingt quatre preludes op 28 breitkopf härtel"
        ),
        "titles": ["24 Preludes 6088"],
        "notes": [
            "Printed music, Breitkopf & Hartel, op. 28, p. 1-40"
        ],
        "cc_by_4_explicit": True,
    }
    assert metadata_match_evidence(
        case,
        metadata,
        catalog_source_metadata={
            "titles": ["Vingt-quatre preludes 6088"],
            "notes": ["Plate 6088, p. 1-40"],
        },
    )["layout_audit_candidate"] is True
    metadata["cc_by_4_explicit"] = False
    assert metadata_match_evidence(
        case,
        metadata,
        catalog_source_metadata={
            "titles": ["Vingt-quatre preludes 6088"],
            "notes": ["Plate 6088, p. 1-40"],
        },
    )["strict_metadata_candidate"] is False


def test_edition_tokens_exclude_years_and_keep_plate_numbers() -> None:
    assert edition_number_tokens(
        {
            "titles": ["Chopin 6088"],
            "notes": ["Leipzig [1839], plate 6088, id 1001013103"],
        }
    ) == {"6088"}


def test_contributor_name_cannot_impersonate_edition_publisher() -> None:
    evidence = metadata_match_evidence(
        {
            "records": {
                "PPR": "Kistner",
                "OPS": "Op. 7",
                "rism-genre": "Mazurka",
            }
        },
        {
            "normalized_document_text": (
                "kistner holding relation breitkopf edition"
            ),
            "titles": ["Mazurek a-moll op. 7 nr 2"],
            "notes": [
                "reprodukcja druku muzycznego, "
                "Breitkopf & Hartel, 997"
            ],
            "cc_by_4_explicit": True,
        },
        catalog_source_metadata={
            "titles": ["Cinq Mazurkas 997"],
            "notes": [],
        },
    )
    assert evidence["publisher_matches"] is False
    assert evidence["strict_metadata_candidate"] is False


def test_fragment_title_page_and_editorial_copy_are_not_layout_candidates() -> None:
    base = {
        "titles": ["Quatre Mazurkas op. 33"],
        "notes": [
            "reprodukcja druku muzycznego, "
            "Breitkopf & Hartel, 5985, s. 1-17"
        ],
        "cc_by_4_explicit": True,
    }
    case = {
        "records": {
            "PPR": "Breitkopf & Hartel",
            "OPS": "Op. 33",
            "rism-genre": "Mazurka",
        }
    }
    source = {
        "titles": ["Quatre Mazurkas 5985"],
        "notes": ["p. 1-18"],
    }
    assert metadata_match_evidence(
        case,
        base,
        catalog_source_metadata=source,
    )["layout_audit_candidate"] is True
    for phrase in (
        "reprodukcja fragmentu druku muzycznego",
        "reprodukcja strony tytułowej druku muzycznego",
        "Egzemplarz z adnotacjami redaktora",
    ):
        metadata = dict(base)
        metadata["notes"] = [*base["notes"], phrase]
        assert metadata_match_evidence(
            case,
            metadata,
            catalog_source_metadata=source,
        )["layout_audit_candidate"] is False


def test_described_extent_and_publication_year_ignore_scan_year() -> None:
    metadata = {
        "titles": ["Mazurkas"],
        "notes": [
            "reprodukcja (1971), Leipzig [po 1841], s. 1-10"
        ],
    }
    assert described_page_extent(metadata) == 10
    assert publication_year_candidates(metadata) == [1841]


def test_collection_count_disambiguates_same_plate_reprint() -> None:
    assert described_collection_item_count(
        {
            "titles": ["CINQ MAZURKAS"],
            "notes": [],
        }
    ) == 5
    assert described_collection_item_count(
        {
            "titles": ["4 Mazurki op. 7"],
            "notes": ["QUATRE MAZURKAS"],
        }
    ) == 4


def test_extent_does_not_treat_plate_number_as_page_number() -> None:
    assert described_page_extent(
        {
            "titles": [],
            "notes": [
                "M. S. 3692, p. [16] blank, p. 2-15 engraved"
            ],
        }
    ) == 16
