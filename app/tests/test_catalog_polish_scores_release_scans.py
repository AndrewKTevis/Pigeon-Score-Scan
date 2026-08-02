from __future__ import annotations

from pathlib import Path

from app.tools.catalog_polish_scores_release_scans import (
    candidate_row,
    parse_kern,
    rejection_reasons,
)


HEADER = """\
!!!!SEGMENT: fixture.krn
!!!COM: Example, Ada
!!!OTL: Etude
!!!AGN: Etude
!!!YEC: Copyright 2024 Narodowy Instytut Fryderyka Chopina (https://polishscores.org)
!!!YEM: License CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0)
!!!SMS-siglum: PL-X
!!!SMS-shelfmark: 42
!!!SMS-shelfwork: 001
!!!NIFC-rismSourceID: 1001
!!!URL-pdf-islandora: https://repozytorium.nifc.pl/islandora/object/nifc%3A99/datastream/PDF/view
!!!URL-scan: https://polish.musicsources.pl/pl/lokalizacje/galeria/druki-muzyczne/12/3 @EN{Scan}
!!!ONB-nifc: pages 2-3
!!!AIN: 1 violin
"""


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "pl-x" / "kern" / "fixture.krn"
    path.parent.mkdir(parents=True)
    path.write_text(HEADER + body, encoding="utf-8")
    return path


def test_accepts_one_printed_monophonic_instrument(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """\
**kern\t**dynam
*staff1\t*
*Iviolin\t*
=1\t=1
4c\t.
4d\t.
*-\t*-
""",
    )
    parsed = parse_kern(path)
    assert parsed.maximum_kern_spines == 1
    assert rejection_reasons(parsed) == ()
    row = candidate_row(parsed, tmp_path)
    assert row["accepted"] is True
    assert row["scan_pdf_url"].endswith("/nifc%3A99/datastream/PDF/view")
    assert row["scan_page_note"] == "pages 2-3"
    assert row["scan_asset_rights_status"] == "unverified"
    assert row["scan_asset_rights_evidence"] == ""
    assert len(row["source_group_fingerprint"]) == 64
    assert len(row["work_fingerprint"]) == 64


def test_rejects_a_kern_voice_split_even_when_it_merges(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """\
**kern
*staff1
*^
4c\t4e
*v\t*v
4d
*-
""",
    )
    parsed = parse_kern(path)
    assert parsed.maximum_kern_spines == 2
    assert "independent_voice_or_staff_split" in rejection_reasons(parsed)


def test_rejects_lyrics_and_keyboard(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """\
**kern\t**text
*staff1\t*
4c\tla
*-\t*-
""",
    )
    text = path.read_text(encoding="utf-8").replace(
        "!!!AIN: 1 violin",
        "!!!AIN: 1 piano",
    )
    path.write_text(text, encoding="utf-8")
    reasons = rejection_reasons(parse_kern(path))
    assert "lyrics_harmony_or_auxiliary_pitch_encoding" in reasons
    assert "not_one_supported_non_keyboard_instrument" in reasons


def test_rejects_manuscript_and_non_exact_license(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """\
**kern
*staff1
4c
*-
""",
    )
    text = (
        path.read_text(encoding="utf-8")
        .replace("/galeria/druki-muzyczne/", "/galeria/rekopisy/")
        .replace("License CC-BY-4.0", "License unknown")
    )
    path.write_text(text, encoding="utf-8")
    reasons = rejection_reasons(parse_kern(path))
    assert "not_explicitly_printed_scan" in reasons
    assert "transcription_license_not_exact_cc_by_4" in reasons


def test_invalid_spine_exchange_is_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """\
**kern\t**dynam
*x\t*
4c\t.
*-\t*-
""",
    )
    reasons = rejection_reasons(parse_kern(path))
    assert "invalid_or_unsupported_spine_structure" in reasons
