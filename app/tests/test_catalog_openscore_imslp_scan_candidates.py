from __future__ import annotations

from pathlib import Path

from app.tools.catalog_openscore_imslp_scan_candidates import catalog_row


def _write(tmp_path: Path, *, arranger: str, parts: int = 4) -> Path:
    path = tmp_path / "scores" / "Composer" / "Work" / "score.mscx"
    path.parent.mkdir(parents=True)
    part_xml = "\n".join(
        f"<Part><Staff id=\"{index}\"/><trackName>Part {index}</trackName></Part>"
        for index in range(1, parts + 1)
    )
    path.write_text(
        f"""\
<museScore><Score>
<metaTag name="arranger">{arranger}</metaTag>
<metaTag name="composer">Example Composer</metaTag>
<metaTag name="copyright">OpenScore (CC0)</metaTag>
<metaTag name="movementTitle"></metaTag>
<metaTag name="workTitle">Example Quartet</metaTag>
{part_xml}
</Score></museScore>
""",
        encoding="utf-8",
    )
    return path


def test_accepts_four_part_cc0_score_with_exact_imslp_file(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        arranger=(
            "Transcribed from https://imslp.org/wiki/"
            "Special:ReverseLookup/17471"
        ),
    )
    row = catalog_row(path, tmp_path)
    assert row["accepted"] is True
    assert row["imslp_source_id"] == "17471"
    assert row["part_count"] == 4
    assert row["boundary_configuration"] == "monophonic_ensemble"


def test_rejects_manuscript_and_non_four_part_score(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        arranger=(
            "Transcribed from original manuscript "
            "https://imslp.org/wiki/Special:ReverseLookup/1"
        ),
        parts=3,
    )
    row = catalog_row(path, tmp_path)
    assert row["accepted"] is False
    assert "source_described_as_manuscript" in row["reasons"]
    assert "not_exactly_four_score_parts" in row["reasons"]


def test_rejects_imslp_page_ranges(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        arranger=(
            "Transcribed from https://imslp.org/wiki/"
            "Special:ReverseLookup/88976-9"
        ),
    )
    row = catalog_row(path, tmp_path)
    assert row["accepted"] is False
    assert "non_numeric_imslp_source_id" in row["reasons"]
