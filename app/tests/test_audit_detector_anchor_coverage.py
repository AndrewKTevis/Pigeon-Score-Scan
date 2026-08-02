from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.tools.audit_detector_anchor_coverage import (
    _anchor_dimensions,
    anchor_dimensions,
    audit_anchor_coverage,
    best_anchor_ious,
    summarize_ious,
)


def test_anchor_dimensions_match_rounded_torchvision_geometry() -> None:
    assert _anchor_dimensions(8.0, 1.0) == (8.0, 8.0)
    assert _anchor_dimensions(8.0, 0.1) == (26.0, 2.0)
    assert _anchor_dimensions(8.0, 10.0) == (2.0, 26.0)


def test_best_anchor_iou_separates_shape_and_grid_ceiling() -> None:
    centered, grid = best_anchor_ious((4.0, 4.0, 12.0, 12.0))
    assert centered == pytest.approx(1.0)
    assert grid == pytest.approx(1.0)

    centered, grid = best_anchor_ious((5.0, 5.0, 13.0, 13.0))
    assert centered == pytest.approx(1.0)
    assert 0.0 < grid < centered

    p2 = anchor_dimensions(
        sizes_by_level=((4.0, 6.0), (8.0, 12.0)),
        strides=(4.0, 8.0),
    )
    _centered, canonical_small_grid = best_anchor_ious(
        (2.0, 2.0, 6.0, 6.0),
    )
    _centered, p2_grid = best_anchor_ious(
        (2.0, 2.0, 6.0, 6.0),
        anchors=p2,
    )
    assert p2_grid > canonical_small_grid


def test_summary_is_deterministic_and_closed_on_empty() -> None:
    empty = summarize_ious([])
    assert empty["objects"] == 0
    assert empty["coverage"]["iou_at_least_0.35"] == 0.0

    summary = summarize_ious([0.1, 0.4, 0.8])
    assert summary["objects"] == 3
    assert summary["median"] == pytest.approx(0.4)
    assert summary["coverage"]["iou_at_least_0.35"] == pytest.approx(2 / 3)


def test_audit_binds_inputs_and_reports_per_class(tmp_path: Path) -> None:
    categories = tmp_path / "categories.json"
    categories.write_text(
        json.dumps(
            {
                "classes": [
                    {"label": 1, "name": "square"},
                    {"label": 2, "name": "long"},
                ]
            }
        ),
        encoding="utf-8",
    )
    rows = tmp_path / "test.jsonl"
    rows.write_text(
        json.dumps(
            {
                "objects": [
                    {"label": 1, "box_xyxy": [4, 4, 12, 12]},
                    {"label": 2, "box_xyxy": [0, 0, 200, 5]},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    training_source = tmp_path / "trainer.py"
    training_source.write_text("# bound source\n", encoding="utf-8")

    report = audit_anchor_coverage(
        rows,
        categories,
        training_source_path=training_source,
    )

    assert report["release_authorized"] is False
    assert report["input"]["rows"] == 1
    assert report["input"]["objects"] == 2
    assert set(report["classes"]) == {"square", "long"}
    assert (
        report["overall"]["hypothetical_p2_grid_assignment_ceiling"][
            "mean"
        ]
        >= report["overall"]["grid_assignment_ceiling"]["mean"]
    )
    assert (
        report["overall"][
            "hypothetical_expanded_ratio_grid_assignment_ceiling"
        ]["mean"]
        >= report["overall"]["grid_assignment_ceiling"]["mean"]
    )
    assert (
        report["classes"]["square"]["grid_assignment_ceiling"]["mean"]
        > report["classes"]["long"]["grid_assignment_ceiling"]["mean"]
    )
