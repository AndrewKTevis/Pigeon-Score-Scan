from __future__ import annotations

import io
import json
from pathlib import Path

from PIL import Image
import pytest

from app.tools.acquire_nifc_chopin_matched_scans import (
    MANIFEST_ROLE,
    _split_spread_pages,
    inspect_image,
    parse_child_membership,
    parse_mods,
    parse_parent_children,
    reference_page_profile,
    reference_problem_profile,
    validate_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_parses_explicit_rights_and_parent_child_order() -> None:
    mods = b"""\
<mods xmlns="http://www.loc.gov/mods/v3">
  <titleInfo><title>Work A</title></titleInfo>
  <accessCondition>CC BY 4.0 (Attribution 4.0)</accessCondition>
  <note>Kistner 1033.</note>
</mods>"""
    parsed = parse_mods(mods)
    assert parsed["titles"] == ["Work A"]
    assert parsed["cc_by_4_explicit"] is True
    page = b"""\
<html><body>
<dl class="islandora-object nifc-2">
  <a href="/islandora/object/nifc%3A2" title="scan--001">first</a>
  <a href="/islandora/object/nifc%3A2" title="scan--001">duplicate</a>
</dl>
<dl class="islandora-object nifc-3">
  <a href="/islandora/object/nifc:3" title="scan--002">second</a>
</dl>
</body></html>"""
    assert parse_parent_children(page) == [
        {"pid": "nifc:2", "title": "scan--001"},
        {"pid": "nifc:3", "title": "scan--002"},
    ]


def test_child_membership_must_be_exact() -> None:
    rels = b"""\
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
 xmlns:fedora="info:fedora/fedora-system:def/relations-external#">
 <rdf:Description rdf:about="info:fedora/nifc:2">
  <fedora:isMemberOfCollection rdf:resource="info:fedora/nifc:1"/>
 </rdf:Description>
</rdf:RDF>"""
    assert parse_child_membership(rels) == ("nifc:2", "nifc:1")


def test_terminal_original_page_break_does_not_create_empty_page() -> None:
    assert reference_page_profile(
        [
            "**kern",
            "4c",
            "!!LO:PB:g=original",
            "4d",
            "!!LO:PB:g=original",
            "==",
            "*-",
        ]
    ) == {
        "original_page_break_count": 2,
        "terminal_trailing_page_break": True,
        "encoded_music_page_count": 2,
    }
    assert reference_page_profile(
        [
            "**kern",
            "4c",
            "!!LO:PB:g=original",
            "4d",
            "*-",
        ]
    )["encoded_music_page_count"] == 2


def test_problem_annotations_are_counted_per_reference_page() -> None:
    assert reference_problem_profile(
        [
            "**kern",
            "!LO:TX:t=P:mixed beam:problem",
            "!!LO:PB:g=original",
            "!LO:S:s=1:problem:note-level slur:problem",
            "!!LO:PB:g=original",
            "==",
            "*-",
        ]
    ) == {
        "problem_record_count": 2,
        "problem_marker_count": 3,
        "affected_page_count": 2,
        "problem_records_per_page": {"1": 1, "2": 1},
        "pages_without_problem_records": [],
    }


def test_spread_split_is_fixed_midpoint_crop_without_rotation() -> None:
    source = Image.new("L", (1201, 800), 255)
    buffer = io.BytesIO()
    source.save(buffer, format="PNG")
    pages = _split_spread_pages(buffer.getvalue())
    assert [crop for _, crop in pages] == [
        (0, 0, 600, 800),
        (600, 0, 1201, 800),
    ]
    assert [inspect_image(payload)["width"] for payload, _ in pages] == [
        600,
        601,
    ]
    assert all(
        inspect_image(payload)["exif_orientation"] == 1
        for payload, _ in pages
    )


def test_checked_in_manifest_is_pinned_boundary_only_and_unauthorized() -> None:
    manifest_path = (
        PROJECT_ROOT
        / "training"
        / "nifc_chopin_matched_scan_candidates.v1.json"
    )
    repository = (
        PROJECT_ROOT
        / "training_data"
        / "external"
        / "catalogs"
        / "humdrum-chopin-first-editions-ccby4"
    )
    if not manifest_path.is_file() or not repository.is_dir():
        pytest.skip("optional NIFC reference repository is not installed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    works = validate_manifest(
        manifest,
        manifest_path=manifest_path,
        repository=repository,
    )
    assert manifest["role"] == MANIFEST_ROLE
    assert manifest["training_authorized"] is False
    assert manifest["evaluation_authorized"] is False
    assert manifest["release_authorized"] is False
    assert [work["_reference_profile"]["encoded_music_page_count"] for work in works] == [
        18,
        8,
    ]
    assert all(work["_boundary"]["score_shape"] == "keyboard" for work in works)
