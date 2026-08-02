from __future__ import annotations

from pathlib import Path

from app.tools.catalog_polish_scores_boundary_scans import candidate_row


HEADER = """\
!!!COM: Example, Ada
!!!OTL: Concert Piece
!!!AGN: Chamber music
!!!YEC: Copyright 2024 Narodowy Instytut Fryderyka Chopina (https://polishscores.org)
!!!YEM: License CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0)
!!!SMS-siglum: PL-X
!!!SMS-shelfmark: 42
!!!SMS-shelfwork: 001
!!!NIFC-rismSourceID: 1001
!!!URL-scan: https://polish.musicsources.pl/pl/lokalizacje/galeria/druki-muzyczne/12/3 @EN{Scan}
!!!AIN: 1 piano 1 violin
"""


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "pl-x" / "kern" / "fixture.krn"
    path.parent.mkdir(parents=True)
    path.write_text(HEADER + body, encoding="utf-8")
    return path


def test_expanded_catalog_accepts_keyboard_plus_ensemble_candidate(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        """\
**kern\t**kern\t**dynam\t**kern
*part2\t*part2\t*part2\t*part1
*staff3\t*staff2\t*\t*staff1
*Ipiano\t*Ipiano\t*\t*Iviolin
*M4/4\t*M4/4\t*\t*M4/4
=1\t=1\t=1\t=1
4C\t4c\tf\t2g
4D\t4d\t.\t.
*-\t*-\t*-\t*-
""",
    )
    row = candidate_row(path, tmp_path)
    assert row["accepted"] is True
    assert row["boundary"]["score_shape"] == (
        "keyboard_plus_single_staff_ensemble"
    )
    assert row["scan_medium_status"] == "explicit_printed"
    assert row["full_score_without_part_file_merge_status"] == "unverified"


def test_expanded_catalog_keeps_unknown_scan_medium_out(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        """\
**kern
*part1
*staff1
*Iviolin
4c
*-
""",
    )
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "https://polish.musicsources.pl/pl/lokalizacje/galeria/druki-muzyczne/12/3",
            "https://example.test/unknown-source/12",
        ),
        encoding="utf-8",
    )
    row = candidate_row(path, tmp_path)
    assert row["accepted"] is False
    assert "scan_not_proved_printed" in row["reasons"]
    assert row["boundary"]["accepted"] is True


def test_expanded_catalog_rejects_second_non_keyboard_voice(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        """\
**kern
*part1
*staff1
*Iviolin
*^
4c\t4e
*v\t*v
*-
""",
    )
    text = path.read_text(encoding="utf-8").replace(
        "!!!AIN: 1 piano 1 violin",
        "!!!AIN: 1 violin",
    )
    path.write_text(text, encoding="utf-8")
    row = candidate_row(path, tmp_path)
    assert row["accepted"] is False
    assert (
        "more_than_one_independent_voice_per_non_keyboard_staff"
        in row["reasons"]
    )
