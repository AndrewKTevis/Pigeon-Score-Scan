from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.tools.muse_omr_contract import TRAINING_REGION_ROLE
from app.tools.prepare_detector_final_refit_partition import (
    FINAL_REFIT_PARTITION_CONTRACT,
    prepare_final_refit_partition,
)
from app.tools.prepare_openscore_svg_regions import (
    COMPLETE_PAGE_TARGET_PROVENANCE,
)
from app.tools.semantic_target_visibility import (
    OVERSIZED_FRAGMENT_VISIBILITY_VERSION,
)


def _write_source(root: Path) -> None:
    root.mkdir()
    base = {
        "role": TRAINING_REGION_ROLE,
        "source_split_overlap": 0,
        "target_geometry_provenance": COMPLETE_PAGE_TARGET_PROVENANCE,
        "oversized_fragment_visibility_version": (
            OVERSIZED_FRAGMENT_VISIBILITY_VERSION
        ),
        "minimum_object_fraction": 0.8,
        "long_span_minimum_object_fraction": 0.25,
        "overlap": 256,
    }
    (root / "manifest.json").write_text(
        json.dumps(
            {
                **base,
                "format": 1,
                "train": {"tiles": 1},
                "calibration": {"tiles": 1},
                "test": {"tiles": 1},
            }
        ),
        encoding="utf-8",
    )
    (root / "prepare-report.json").write_text(
        json.dumps(
            {
                **base,
                "split_intersections": {
                    "train_calibration": [],
                    "train_test": [],
                    "calibration_test": [],
                },
                "dropped_object_counts": {},
            }
        ),
        encoding="utf-8",
    )
    (root / "categories.json").write_text(
        '{"classes":[{"label":1,"name":"tie"}]}',
        encoding="utf-8",
    )
    for split, source in (
        ("train", "work-a"),
        ("calibration", "work-b"),
        ("test", "work-c"),
    ):
        (root / f"{split}.jsonl").write_text(
            json.dumps(
                {
                    "split": split,
                    "source_key": source,
                    "image": f"{source}.png",
                    "crop_xyxy": [0, 0, 32, 32],
                    "objects": [
                        {"label": 1, "box_xyxy": [1, 1, 3, 3]}
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )


def test_final_refit_moves_used_test_to_train_and_keeps_calibration_disjoint(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source(source)

    report = prepare_final_refit_partition(source, output)
    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    train = [
        json.loads(line)
        for line in (output / "train.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    test = [
        json.loads(line)
        for line in (output / "test.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert report["final_refit_partition_contract"] == (
        FINAL_REFIT_PARTITION_CONTRACT
    )
    assert report["production_evidence_eligible"] is False
    assert manifest["model_outputs_used_for_partition"] is False
    assert manifest["train"]["sources"] == 2
    assert manifest["test"]["sources"] == 1
    assert {row["source_key"] for row in train} == {"work-a", "work-c"}
    assert {row["final_refit_origin_split"] for row in train} == {
        "train",
        "test",
    }
    assert test[0]["source_key"] == "work-b"
    assert test[0]["split"] == "test"


def test_final_refit_refuses_nontraining_or_overlapping_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_source(source)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["role"] = "external_test_only"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="not final-refit eligible"):
        prepare_final_refit_partition(source, tmp_path / "bad-role")

    manifest["role"] = TRAINING_REGION_ROLE
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    test_path = source / "test.jsonl"
    test = json.loads(test_path.read_text(encoding="utf-8"))
    test["source_key"] = "work-a"
    test_path.write_text(json.dumps(test) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps by source group"):
        prepare_final_refit_partition(source, tmp_path / "overlap")
