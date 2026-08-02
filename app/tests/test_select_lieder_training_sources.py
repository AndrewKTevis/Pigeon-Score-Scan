from __future__ import annotations

from pathlib import Path

from app.tools.select_lieder_training_sources import (
    diverse_round_robin,
    mark_richness,
    score_ids_from_olimpic_yaml,
)


def test_reads_only_top_level_numeric_score_ids(tmp_path: Path) -> None:
    path = tmp_path / "scores.yaml"
    path.write_text(
        "123:\n  set_id: 9\n456:\n  name: Test\n",
        encoding="utf-8",
    )
    assert score_ids_from_olimpic_yaml(path) == {123, 456}


def test_mark_richness_prioritizes_critical_marks() -> None:
    score, counts = mark_richness(
        "<Slur/><Slur/><HairPin><Dynamic><Tempo><Lyrics>"
    )
    assert counts["Slur"] == 2
    assert counts["HairPin"] == 1
    assert score == 2 * 2 + 7 + 5 + 5 + 1


def test_diverse_round_robin_covers_composers_before_second_score() -> None:
    rows = [
        {
            "composer": "A",
            "mark_richness": value,
            "relative_path": f"A/{value}.mscx",
        }
        for value in (10, 5, 1)
    ] + [
        {
            "composer": "B",
            "mark_richness": 3,
            "relative_path": "B/3.mscx",
        }
    ]
    selected = diverse_round_robin(rows, 3)
    assert {row["composer"] for row in selected[:2]} == {"A", "B"}
    assert len(selected) == 3
