from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.tools.audit_collabscore_diagnostic_source import audit


SCORE = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Flute</part-name></score-part></part-list>
  <part id="P1"><measure number="1"><attributes><divisions>1</divisions>
    <time><beats>4</beats><beat-type>4</beat-type></time></attributes>
    <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration>
      <voice>1</voice><type>whole</type>{lyric}</note>
  </measure></part>
</score-partwise>
"""


def _source(root: Path) -> Path:
    source = root / "collabscore"
    (source / "ground_truth").mkdir(parents=True)
    (source / "iiif").mkdir()
    (source / "LICENSE.md").write_text(
        "CC BY-NC-SA 4.0",
        encoding="utf-8",
    )
    rows = []
    for reference_id, lyric in (
        ("C001_0", ""),
        (
            "C002_0",
            "<lyric><syllabic>single</syllabic><text>la</text></lyric>",
        ),
    ):
        rows.append(
            {
                "ref": reference_id,
                "title": reference_id,
                "genre": "fixture",
                "nb_parts": 1,
                "nb_music_pages": 1,
                "nb_systems": 1,
                "nb_measures": 1,
                "iiif_link": f"https://example.test/{reference_id}",
            }
        )
        (source / "ground_truth" / f"{reference_id}.musicxml").write_text(
            SCORE.format(lyric=lyric),
            encoding="utf-8",
        )
        (source / "ground_truth" / f"{reference_id}.mei").write_text(
            "<mei/>",
            encoding="utf-8",
        )
        for suffix in ("mnf", "annot"):
            (
                source / "iiif" / f"{reference_id}_{suffix}.json"
            ).write_text("{}", encoding="utf-8")
    (source / "dataset.json").write_text(
        json.dumps({"total_pages": 2, "list_opus": rows}),
        encoding="utf-8",
    )
    return source


def test_audit_stops_before_image_download_and_separates_lyric_exclusion(
    tmp_path: Path,
) -> None:
    report = audit(_source(tmp_path), revision="a" * 40)

    assert report["declared_work_count"] == 2
    assert report["dataset_total_pages"] == 2
    assert report["page_count_consistent"] is True
    assert report["musicxml_reference_count"] == 2
    assert report["strict_boundary_accepted_work_count"] == 1
    assert report["lyrics_only_excluded_work_count"] == 1
    assert report["boundary_reason_counts"] == {"lyrics": 1}
    assert report["source_images_downloaded"] is False
    assert report["image_download_authorized"] is False
    assert report["training_authorized"] is False
    assert report["release_evaluation_authorized"] is False
    assert report["release_authorized"] is False
    assert all(
        row["image_download_authorized"] is False
        for row in report["works"]
    )


def test_audit_requires_full_pinned_revision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="full lowercase Git commit"):
        audit(_source(tmp_path), revision="main")
