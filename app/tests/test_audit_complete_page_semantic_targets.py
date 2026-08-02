from __future__ import annotations

import json
from pathlib import Path

from app.tools.audit_complete_page_semantic_targets import audit_dataset
from app.tools.prepare_openscore_svg_regions import (
    COMPLETE_PAGE_TARGET_PROVENANCE,
)
from app.tools.semantic_target_visibility import (
    OVERSIZED_FRAGMENT_VISIBILITY_VERSION,
)


def _dataset(tmp_path: Path) -> Path:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "format": 1,
                "tile_size": 10,
                "overlap": 5,
                "oversized_fragment_visibility_version": (
                    OVERSIZED_FRAGMENT_VISIBILITY_VERSION
                ),
            }
        ),
        encoding="utf-8",
    )
    (prepared / "prepare-report.json").write_text(
        json.dumps(
            {
                "minimum_object_fraction": 0.8,
                "long_span_minimum_object_fraction": 0.25,
                "dropped_object_counts": {},
                "tile_size": 10,
                "overlap": 5,
                "oversized_fragment_visibility_version": (
                    OVERSIZED_FRAGMENT_VISIBILITY_VERSION
                ),
            }
        ),
        encoding="utf-8",
    )
    row = {
        "source_key": "work",
        "image": "page.png",
        "crop_xyxy": [0, 0, 10, 10],
        "objects": [
            {
                "box_xyxy": [8, 2, 10, 4],
                "page_box_xyxy": [8, 2, 12, 4],
                "category_id": "slur",
                "label": 31,
                "source_object_id": "s" * 24,
                "target_geometry_provenance": COMPLETE_PAGE_TARGET_PROVENANCE,
            }
        ],
    }
    (prepared / "test.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )
    return prepared


def test_complete_page_audit_accepts_consistent_partial_long_mark(
    tmp_path: Path,
) -> None:
    report = audit_dataset(_dataset(tmp_path), splits=("test",))

    assert report["passed"] is True
    assert report["failures"] == []
    assert report["splits"]["test"]["target_instances"] == 1
    assert report["splits"]["test"]["partial_target_instances"] == 1
    assert report["splits"]["test"]["unique_source_objects"] == 1


def test_complete_page_audit_rejects_geometry_contradiction(
    tmp_path: Path,
) -> None:
    prepared = _dataset(tmp_path)
    row = json.loads((prepared / "test.jsonl").read_text(encoding="utf-8"))
    row["objects"][0]["box_xyxy"] = [7, 2, 10, 4]
    (prepared / "test.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )

    report = audit_dataset(prepared, splits=("test",))

    assert report["passed"] is False
    assert any("local/page box contradiction" in row for row in report["failures"])


def test_complete_page_audit_rejects_dropped_long_span_object(
    tmp_path: Path,
) -> None:
    prepared = _dataset(tmp_path)
    preparation = json.loads(
        (prepared / "prepare-report.json").read_text(encoding="utf-8")
    )
    preparation["dropped_object_counts"] = {"hairpin": 1}
    (prepared / "prepare-report.json").write_text(
        json.dumps(preparation),
        encoding="utf-8",
    )

    report = audit_dataset(prepared, splits=("test",))

    assert report["passed"] is False
    assert report["dropped_long_span_objects"] == {"hairpin": 1}
