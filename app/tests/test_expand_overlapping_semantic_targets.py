from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.tools.expand_overlapping_semantic_targets import (
    expand_page_rows,
    expand_split,
)
from app.tools.prepare_openscore_svg_regions import (
    COMPLETE_PAGE_TARGET_PROVENANCE,
)


def _row(
    *,
    crop: list[int],
    objects: list[dict[str, object]],
    image_id: str = "page-1",
) -> dict[str, object]:
    return {
        "split": "train",
        "source_key": "work-1",
        "image": "page.png",
        "image_id": image_id,
        "crop_xyxy": crop,
        "objects": objects,
    }


def test_overlap_expansion_labels_visible_ink_in_every_retained_tile() -> None:
    rows = [
        _row(
            crop=[0, 0, 10, 10],
            objects=[
                {
                    "box_xyxy": [8, 2, 10, 4],
                    "page_box_xyxy": [8, 2, 10, 4],
                    "category_id": "tie",
                    "label": 37,
                    "svg_class": "TieSegment",
                    "source_object_id": "t" * 24,
                    "target_geometry_provenance": (
                        COMPLETE_PAGE_TARGET_PROVENANCE
                    ),
                }
            ],
        ),
        _row(
            crop=[5, 0, 15, 10],
            objects=[
                {
                    "box_xyxy": [7, 5, 9, 7],
                    "page_box_xyxy": [12, 5, 14, 7],
                    "category_id": "slur",
                    "label": 31,
                    "svg_class": "SlurSegment",
                    "source_object_id": "s" * 24,
                    "target_geometry_provenance": (
                        COMPLETE_PAGE_TARGET_PROVENANCE
                    ),
                }
            ],
        ),
    ]

    expanded, report = expand_page_rows(
        rows,
        minimum_visible_fraction=0.8,
    )

    assert [len(row["objects"]) for row in expanded] == [1, 2]
    added = next(
        obj for obj in expanded[1]["objects"] if obj["category_id"] == "tie"
    )
    assert added["box_xyxy"] == [3.0, 2.0, 5.0, 4.0]
    assert len(added["source_object_id"]) == 24
    assert (
        expanded[0]["objects"][0]["source_object_id"]
        == added["source_object_id"]
    )
    assert report["unique_objects"] == 2
    assert report["target_instances"] == 3
    assert report["input_target_instances"] == 2
    assert report["duplicate_source_objects_removed"] == 0
    assert report["additional_target_instances"] == 1
    assert report["rows_with_added_targets"] == 1
    assert report["maximum_assignments_per_object"] == 2


def test_overlap_expansion_deduplicates_exact_visual_targets() -> None:
    duplicate = {
        "box_xyxy": [2, 2, 8, 4],
        "page_box_xyxy": [2, 2, 8, 4],
        "category_id": "slur",
        "label": 31,
        "svg_class": "SlurSegment",
        "source_object_id": "slur-source",
        "target_geometry_provenance": COMPLETE_PAGE_TARGET_PROVENANCE,
    }
    rows = [
        _row(
            crop=[0, 0, 10, 10],
            objects=[duplicate, dict(duplicate)],
        ),
        _row(crop=[5, 0, 15, 10], objects=[]),
    ]

    expanded, report = expand_page_rows(
        rows,
        minimum_visible_fraction=0.5,
    )

    assert [len(row["objects"]) for row in expanded] == [1, 1]
    assert (
        expanded[0]["objects"][0]["source_object_id"]
        == expanded[1]["objects"][0]["source_object_id"]
    )
    assert report["input_target_instances"] == 2
    assert report["duplicate_source_objects_removed"] == 1
    assert report["duplicate_source_object_counts"] == {"slur": 1}
    assert report["unique_objects"] == 1
    assert report["target_instances"] == 2
    assert report["additional_target_instances"] == 1


def test_overlap_expansion_recovers_complete_page_box_after_owner_clip() -> None:
    rows = [
        _row(
            crop=[0, 0, 10, 10],
            objects=[
                {
                    "box_xyxy": [8, 2, 10, 4],
                    "page_box_xyxy": [8, 2, 12, 4],
                    "category_id": "tie",
                    "label": 37,
                    "svg_class": "TieSegment",
                    "source_object_id": "tie-source",
                    "target_geometry_provenance": (
                        COMPLETE_PAGE_TARGET_PROVENANCE
                    ),
                }
            ],
        ),
        _row(crop=[5, 0, 15, 10], objects=[]),
    ]

    expanded, report = expand_page_rows(
        rows,
        minimum_visible_fraction=0.8,
        long_span_minimum_visible_fraction=0.5,
    )

    assert expanded[0]["objects"][0]["box_xyxy"] == [8.0, 2.0, 10.0, 4.0]
    assert expanded[1]["objects"][0]["box_xyxy"] == [3.0, 2.0, 7.0, 4.0]
    assert all(
        row["objects"][0]["page_box_xyxy"] == [8.0, 2.0, 12.0, 4.0]
        for row in expanded
    )
    assert report["unique_objects"] == 1
    assert report["target_instances"] == 2


def test_overlap_expansion_rejects_legacy_clipped_box_without_page_geometry() -> None:
    rows = [
        _row(
            crop=[0, 0, 10, 10],
            objects=[
                {
                    "box_xyxy": [8, 2, 10, 4],
                    "category_id": "tie",
                    "label": 37,
                    "svg_class": "TieSegment",
                }
            ],
        )
    ]

    with pytest.raises(ValueError, match="complete-page geometry provenance"):
        expand_page_rows(rows, minimum_visible_fraction=0.8)


def test_overlap_expansion_keeps_contiguous_oversized_bracket_fragments() -> None:
    source_id = "page-bracket"
    rows = [
        _row(
            crop=[0, 0, 10, 10],
            objects=[
                {
                    "box_xyxy": [2, 2, 3, 10],
                    "page_box_xyxy": [2, 2, 3, 28],
                    "category_id": "bracket",
                    "label": 4,
                    "svg_class": "Bracket",
                    "source_object_id": source_id,
                    "target_geometry_provenance": (
                        COMPLETE_PAGE_TARGET_PROVENANCE
                    ),
                }
            ],
        ),
        _row(crop=[0, 5, 10, 15], objects=[]),
        _row(crop=[0, 10, 10, 20], objects=[]),
        _row(crop=[0, 15, 10, 25], objects=[]),
        _row(crop=[0, 20, 10, 30], objects=[]),
    ]

    expanded, report = expand_page_rows(
        rows,
        minimum_visible_fraction=0.8,
        long_span_minimum_visible_fraction=0.25,
        tile_overlap=5,
    )

    assert [len(row["objects"]) for row in expanded] == [1, 1, 1, 1, 1]
    assert {
        row["objects"][0]["source_object_id"] for row in expanded
    } == {source_id}
    assert report["maximum_assignments_per_object"] == 5


def test_overlap_expansion_rejects_noncontiguous_page_rows(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "destination.jsonl"
    rows = [
        _row(crop=[0, 0, 10, 10], objects=[], image_id="page-a"),
        _row(crop=[0, 0, 10, 10], objects=[], image_id="page-b"),
        _row(crop=[5, 0, 15, 10], objects=[], image_id="page-a"),
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not contiguous"):
        expand_split(
            source,
            destination,
            minimum_visible_fraction=0.8,
        )
    assert not destination.exists()
    assert not destination.with_suffix(".jsonl.tmp").exists()
