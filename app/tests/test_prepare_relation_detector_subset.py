from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.tools.prepare_relation_detector_subset import (
    SUBSET_CONTRACT_VERSION,
    prepare_relation_detector_subset,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    common = {
        "target_assignment_version": (
            "complete-page-overlap-consistent-deduplicated-semantic-targets@4"
        ),
        "target_geometry_provenance": (
            "complete-page-svg-geometry-before-tile-clipping@1"
        ),
        "oversized_fragment_visibility_version": (
            "complete-page-oversized-axis-overlap-fragments@1"
        ),
        "tile_size": 1024,
        "overlap": 256,
        "minimum_object_fraction": 0.8,
        "long_span_minimum_object_fraction": 0.25,
        "role": "training_only_disjoint_from_external_release_holdout",
        "source_split_overlap": 0,
        "reserved_holdout_overlap": 0,
        "train": {"tiles": 3},
        "test": {"tiles": 2},
    }
    (source / "manifest.json").write_text(json.dumps(common), encoding="utf-8")
    preparation = dict(common)
    preparation["transformation_version"] = common["target_assignment_version"]
    preparation["dropped_object_counts"] = {}
    (source / "prepare-report.json").write_text(
        json.dumps(preparation), encoding="utf-8"
    )
    (source / "categories.json").write_text(
        json.dumps(
            {
                "classes": [
                    {"label": 1, "name": "beam"},
                    {"label": 2, "name": "slur"},
                    {"label": 3, "name": "tie"},
                    {"label": 4, "name": "hairpin"},
                ]
            }
        ),
        encoding="utf-8",
    )
    positive = {
        "image": "page.jpg",
        "image_id": "positive",
        "source_key": "work-a",
        "crop_xyxy": [0, 0, 1024, 1024],
        "objects": [
            {"category_id": "beam", "label": 1, "box_xyxy": [1, 1, 9, 9]},
            {
                "category_id": "slur",
                "label": 2,
                "box_xyxy": [2, 2, 20, 8],
                "page_box_xyxy": [102, 202, 120, 208],
            },
            {
                "category_id": "tie",
                "label": 3,
                "box_xyxy": [3, 3, 12, 7],
                "page_box_xyxy": [103, 203, 112, 207],
            },
            {
                "category_id": "hairpin",
                "label": 4,
                "box_xyxy": [4, 4, 22, 9],
                "page_box_xyxy": [104, 204, 122, 209],
            },
        ],
    }
    negatives = [
        {
            "image": "page.jpg",
            "image_id": f"negative-{index}",
            "source_key": "work-a",
            "crop_xyxy": [index, 0, index + 10, 10],
            "objects": [
                {"category_id": "beam", "label": 1, "box_xyxy": [1, 1, 9, 9]}
            ],
        }
        for index in range(2)
    ]
    _write_jsonl(source / "train.jsonl", [positive, *negatives])
    _write_jsonl(source / "test.jsonl", [positive, negatives[0]])
    return source


def test_relation_subset_remaps_targets_and_bounds_training_negatives(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    output = tmp_path / "subset"

    manifest = prepare_relation_detector_subset(
        source,
        output,
        train_negative_ratio=1.0,
        minimum_test_sources_per_class=1,
        minimum_test_unique_objects_per_class=1,
    )

    train = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    test = [json.loads(line) for line in (output / "test.jsonl").read_text().splitlines()]
    assert len(train) == 2
    assert len(test) == 2
    assert manifest["train"]["negative_tiles"] == 1
    assert manifest["test"]["negative_tiles"] == 1
    objects = next(row["objects"] for row in train if row["objects"])
    assert [(item["category_id"], item["label"]) for item in objects] == [
        ("slur", 1),
        ("tie", 2),
        ("hairpin", 3),
    ]
    assert manifest["class_subset_contract"] == SUBSET_CONTRACT_VERSION
    assert manifest["test"]["class_quality"]["hairpin"]["sources"] == 1
    assert manifest["test"]["class_quality"]["hairpin"]["unique_objects"] == 1
    assert (
        manifest["test"]["class_quality"]["hairpin"][
            "duplicate_assignment_factor"
        ]
        == 1.0
    )
    assert (
        manifest["test"]["class_quality"]["hairpin"][
            "clipped_instance_fraction"
        ]
        == 0.0
    )
    preparation = json.loads((output / "prepare-report.json").read_text())
    assert preparation["transformation_version"].endswith("@4")
    assert preparation["dropped_object_counts"] == {}


def test_relation_subset_refuses_missing_classes_and_existing_output(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    with pytest.raises(ValueError, match="missing"):
        prepare_relation_detector_subset(
            source,
            tmp_path / "missing",
            classes=("slur", "unknown"),
        )
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        prepare_relation_detector_subset(source, output)


def test_relation_subset_rejects_instance_counts_from_too_few_sources(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)

    with pytest.raises(
        ValueError,
        match=r"hairpin:sources=1<2.*hairpin:unique_objects=1<2",
    ):
        prepare_relation_detector_subset(
            source,
            tmp_path / "unsupported-test-split",
            minimum_test_sources_per_class=2,
            minimum_test_unique_objects_per_class=2,
        )


def test_relation_subset_rejects_pre_overlap_consistent_source(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("target_assignment_version")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="overlap-consistent"):
        prepare_relation_detector_subset(
            source,
            tmp_path / "legacy-clipped-source",
            minimum_test_sources_per_class=1,
            minimum_test_unique_objects_per_class=1,
        )
