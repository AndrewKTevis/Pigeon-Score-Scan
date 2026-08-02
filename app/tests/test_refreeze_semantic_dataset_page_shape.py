from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from app.tools.muse_omr_contract import (
    SCAN_DEGRADED_IMAGE_ORIGIN,
    TRAINING_REGION_ROLE,
)
from app.tools.prepare_openscore_svg_regions import (
    COMPLETE_PAGE_TARGET_ASSIGNMENT_VERSION,
    COMPLETE_PAGE_TARGET_PROVENANCE,
)
from app.tools.refreeze_semantic_dataset_page_shape import (
    PAGE_SHAPE_REFREEZE_CONTRACT,
    refreeze_dataset,
    sha256_file,
)
from app.tools.semantic_target_visibility import (
    OVERSIZED_FRAGMENT_VISIBILITY_VERSION,
)
from scorescan.semantic_detector_contract import (
    SEMANTIC_DETECTOR_MAXIMUM_PAGE_ASPECT_RATIO,
    page_aspect_ratio,
    page_shape_is_supported,
)


def _write_source(root: Path, images: Path) -> Path:
    root.mkdir()
    images.mkdir()
    categories = root / "categories.json"
    categories.write_text(
        '{"classes":[{"label":1,"name":"tie"}]}',
        encoding="utf-8",
    )
    common = {
        "role": TRAINING_REGION_ROLE,
        "source_split_overlap": 0,
        "target_assignment_version": (
            COMPLETE_PAGE_TARGET_ASSIGNMENT_VERSION
        ),
        "target_geometry_provenance": COMPLETE_PAGE_TARGET_PROVENANCE,
        "oversized_fragment_visibility_version": (
            OVERSIZED_FRAGMENT_VISIBILITY_VERSION
        ),
        "forbidden_selection_overlap": [],
        "forbidden_work_overlap": [],
        "tile_size": 1024,
        "overlap": 256,
        "minimum_object_fraction": 0.8,
        "long_span_minimum_object_fraction": 0.25,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                **common,
                "format": 1,
                "name": "source",
                "accepted_pairs": 2,
                "accepted_works": 2,
                "rejected_pairs": 0,
                "selected_pairs": 2,
                "selected_works": 2,
                "train": {
                    "tiles": 2,
                    "sources": 2,
                    "negative_tiles": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "prepare-report.json").write_text(
        json.dumps(
            {
                **common,
                "transformation_version": (
                    COMPLETE_PAGE_TARGET_ASSIGNMENT_VERSION
                ),
                "license": "CC0-1.0",
                "selected_pairs": 2,
                "selected_works": 2,
                "accepted_pairs": 2,
                "accepted_works": 2,
                "rejected_pairs": 0,
                "accepted": [
                    {
                        "pair_id": 1,
                        "source_key": "work-good",
                        "variant_key": "pair-1",
                    },
                    {
                        "pair_id": 2,
                        "source_key": "work-bad",
                        "variant_key": "pair-2",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "split": "train",
            "source_key": source,
            "image": image,
            "image_id": source,
            "crop_xyxy": [0, 0, 100, 100],
            "objects": [
                {
                    "category_id": "tie",
                    "label": 1,
                    "box_xyxy": [1, 1, 5, 5],
                }
            ],
        }
        for source, image in (
            ("work-good", "good.png"),
            ("work-bad", "bad.png"),
        )
    ]
    split_path = root / "train.jsonl"
    split_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    Image.new("L", (300, 100), 255).save(images / "good.png")
    Image.new("L", (301, 100), 255).save(images / "bad.png")
    evidence_path = root / "layout-evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "passed": True,
                "prepared_manifest_sha256": sha256_file(manifest_path),
                "split_jsonl_sha256": {
                    "train": sha256_file(split_path)
                },
                "target_assignment_version": (
                    COMPLETE_PAGE_TARGET_ASSIGNMENT_VERSION
                ),
                "pages": [
                    {
                        "source_key": source,
                        "image": image,
                        "image_id": source,
                        "layout": {
                            "width": width,
                            "height": 100,
                            "systems": [{"spacing": 10}],
                        },
                    }
                    for source, image, width in (
                        ("work-good", "good.png", 300),
                        ("work-bad", "bad.png", 301),
                    )
                ],
            }
        ),
        encoding="utf-8",
    )
    return evidence_path


def test_page_shape_contract_is_orientation_independent_and_inclusive() -> None:
    assert SEMANTIC_DETECTOR_MAXIMUM_PAGE_ASPECT_RATIO == 3.0
    assert page_aspect_ratio(300, 100) == 3.0
    assert page_aspect_ratio(100, 300) == 3.0
    assert page_shape_is_supported(300, 100)
    assert not page_shape_is_supported(301, 100)
    with pytest.raises(ValueError, match="positive"):
        page_aspect_ratio(0, 100)


def test_refreeze_rejects_entire_source_without_model_outputs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    images = tmp_path / "images"
    output = tmp_path / "output"
    evidence = _write_source(source, images)

    report = refreeze_dataset(
        source,
        images,
        evidence,
        output,
        ("train",),
    )
    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    rows = [
        json.loads(line)
        for line in (output / "train.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert report["page_shape_refreeze_contract"] == (
        PAGE_SHAPE_REFREEZE_CONTRACT
    )
    assert report["model_predictions_observed_for_refreeze"] is False
    assert report["rejected_sources"] == ["work-bad"]
    assert report["accepted_works"] == 1
    assert report["accepted_pairs"] == 1
    assert report["rejected_pairs"] == 1
    assert rows[0]["source_key"] == "work-good"
    assert manifest["train"] == {
        "tiles": 1,
        "sources": 1,
        "negative_tiles": 0,
    }
    assert manifest["source_image_origin"] == SCAN_DEGRADED_IMAGE_ORIGIN
    assert manifest["production_evidence_eligible"] is False


def test_refreeze_rejects_stale_layout_dimensions(tmp_path: Path) -> None:
    source = tmp_path / "source"
    images = tmp_path / "images"
    evidence_path = _write_source(source, images)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["pages"][0]["layout"]["width"] = 299
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(ValueError, match="dimensions changed"):
        refreeze_dataset(
            source,
            images,
            evidence_path,
            tmp_path / "output",
            ("train",),
        )
