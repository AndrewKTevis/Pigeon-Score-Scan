from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from app.tools import merge_semantic_detector_datasets as module


def _dataset(root: Path, name: str) -> Path:
    directory = root / name
    directory.mkdir()
    categories = {
        "format": 1,
        "classes": [
            {"label": 1, "name": "slur", "source": "test"},
            {"label": 2, "name": "tie", "source": "test"},
        ],
    }
    (directory / "categories.json").write_text(
        json.dumps(categories),
        encoding="utf-8",
    )
    sources = []
    for split_index, split in enumerate(module.SPLITS):
        page = directory / "pages" / split / "page.png"
        page.parent.mkdir(parents=True, exist_ok=True)
        Image.new(
            "RGB",
            (64, 64),
            (250 - split_index, 240 if name == "right" else 250, 250),
        ).save(page)
        source_key = f"{split}/source"
        sources.append(
            {
                "source_key": source_key,
                "source_sha256": hashlib.sha256(
                    f"{name}/{source_key}".encode()
                ).hexdigest(),
                "split": split,
            }
        )
        row = {
            "split": split,
            "source_key": source_key,
            "image": page.relative_to(directory).as_posix(),
            "image_id": f"{split}-tile",
            "crop_xyxy": [0, 0, 64, 64],
            "objects": [
                {
                    "box_xyxy": [2, 3, 30, 10],
                    "category_id": "slur",
                    "label": 1,
                }
            ],
        }
        (directory / f"{split}.jsonl").write_text(
            json.dumps(row) + "\n",
            encoding="utf-8",
        )
    report = {
        "purpose": "synthetic semantic geometry; not real-scan validation",
        "split_intersections": module.EMPTY_INTERSECTIONS,
        "minimum_object_fraction": 0.8,
        "sources": sources,
    }
    (directory / "prepare-report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    manifest = {
        "source_split_overlap": 0,
        **{
            split: {"tiles": 1, "sources": 1}
            for split in module.SPLITS
        },
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return directory


def test_merges_compatible_sources_and_rewrites_project_image_paths(
    tmp_path: Path,
) -> None:
    left = _dataset(tmp_path, "left")
    right = _dataset(tmp_path, "right")
    output = tmp_path / "merged"
    assert module.main(
        [
            "--project-root",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--input",
            f"quartet={left}",
            "--input",
            f"lieder={right}",
        ]
    ) == 0
    rows = [
        json.loads(line)
        for line in (output / "train.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    assert rows[0]["source_key"].startswith("quartet:")
    assert rows[0]["image"].startswith("left/pages/")
    report = json.loads(
        (output / "merge-report.json").read_text(encoding="utf-8")
    )
    assert report["sources_by_split"] == {
        "train": 2,
        "calibration": 2,
        "test": 2,
    }
    assert report["object_counts"] == {"slur": 6}
    assert report["legacy_tile_box_clips"]["total"] == 0


def test_clips_legacy_tile_box_only_when_visible_fraction_is_safe(
    tmp_path: Path,
) -> None:
    left = _dataset(tmp_path, "left")
    right = _dataset(tmp_path, "right")
    train_path = left / "train.jsonl"
    row = json.loads(train_path.read_text(encoding="utf-8"))
    row["objects"][0]["box_xyxy"] = [-2, 3, 30, 10]
    train_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    output = tmp_path / "merged"

    assert module.main(
        [
            "--project-root",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--input",
            f"quartet={left}",
            "--input",
            f"lieder={right}",
        ]
    ) == 0
    normalized = json.loads(
        (output / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert normalized["objects"][0]["box_xyxy"] == [0.0, 3.0, 30.0, 10.0]
    report = json.loads(
        (output / "merge-report.json").read_text(encoding="utf-8")
    )
    assert report["legacy_tile_box_clips"] == {
        "total": 1,
        "by_category": {"slur": 1},
        "policy": (
            "clip only when the retained area satisfies the input "
            "minimum_object_fraction; otherwise fail closed"
        ),
    }

    unsafe = _dataset(tmp_path, "unsafe")
    unsafe_train_path = unsafe / "train.jsonl"
    unsafe_row = json.loads(unsafe_train_path.read_text(encoding="utf-8"))
    unsafe_row["objects"][0]["box_xyxy"] = [-20, 3, 30, 10]
    unsafe_train_path.write_text(
        json.dumps(unsafe_row) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="visible fraction"):
        module.main(
            [
                "--project-root",
                str(tmp_path),
                "--output-dir",
                str(tmp_path / "unsafe-merged"),
                "--input",
                f"unsafe={unsafe}",
                "--input",
                f"lieder={right}",
            ]
        )


def test_rejects_duplicate_source_content_between_inputs(
    tmp_path: Path,
) -> None:
    left = _dataset(tmp_path, "left")
    right = _dataset(tmp_path, "right")
    left_report = json.loads(
        (left / "prepare-report.json").read_text(encoding="utf-8")
    )
    right_report_path = right / "prepare-report.json"
    right_report = json.loads(right_report_path.read_text(encoding="utf-8"))
    right_report["sources"][0]["source_sha256"] = left_report["sources"][0][
        "source_sha256"
    ]
    right_report_path.write_text(json.dumps(right_report), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate source content"):
        module.main(
            [
                "--project-root",
                str(tmp_path),
                "--output-dir",
                str(tmp_path / "merged"),
                "--input",
                f"quartet={left}",
                "--input",
                f"lieder={right}",
            ]
        )


def test_single_input_normalizes_legacy_dataset(tmp_path: Path) -> None:
    source = _dataset(tmp_path, "source")
    output = tmp_path / "normalized"

    assert module.main(
        [
            "--project-root",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--input",
            f"quartet={source}",
        ]
    ) == 0
    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    report = json.loads(
        (output / "merge-report.json").read_text(encoding="utf-8")
    )
    assert manifest["name"] == (
        "scorescan-openscore-normalized-semantic-svg-regions"
    )
    assert report["purpose"] == (
        "normalized synthetic semantic geometry; not real-scan validation"
    )
