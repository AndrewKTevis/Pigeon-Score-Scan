from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from app.tools import merge_ocr_detection_labels
from app.tools import prepare_ocr_holdout_detection_labels as module
from app.tools.merge_ocr_training_labels import (
    EMPTY_INTERSECTIONS,
    INDEPENDENT_SCAN_HOLDOUT_ROLE,
    MINIMUM_SAFE_VISUAL_PRESENCE_NCC,
    SCAN_TEXT_PAGE_LABEL_COMPLETENESS_VERSION,
    SCAN_TEXT_REFERENCE_SOURCE_VERSION,
    SCAN_TEXT_VISUAL_PRESENCE_VERSION,
    SOURCE_TEXT_SELECTION_VERSION,
)


def _holdout_dataset(project_root: Path) -> Path:
    directory = project_root / "holdout-text"
    image_root = project_root / "holdout-pages"
    directory.mkdir()
    image_root.mkdir()
    for split in ("train", "calibration"):
        (directory / f"{split}.jsonl").write_text("", encoding="utf-8")
    rows = []
    sources = []
    for index in range(2):
        work_fingerprint = hashlib.sha256(f"work/{index}".encode()).hexdigest()
        source_key = f"muse-omr-work/{work_fingerprint}"
        image = image_root / f"{index}.png"
        Image.new("RGB", (128, 64), "white").save(image)
        crop = directory / "crops" / "test" / f"{index}.png"
        crop.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (60, 20), "white").save(crop)
        rows.append(
            {
                "split": "test",
                "source_key": source_key,
                "page": 1,
                "word_index": 0,
                "image": image.name,
                "crop_image": crop.relative_to(directory).as_posix(),
                "box_xyxy": [10, 10, 70, 30],
                "text": "Allegro",
                "text_role": "supported",
                "visual_presence_ncc": 0.9,
            }
        )
        sources.append(
            {
                "source_key": source_key,
                "split": "test",
                "work_fingerprint": work_fingerprint,
                "retained_words": 1,
            }
        )
    (directory / "test.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "source_text_selection_version": SOURCE_TEXT_SELECTION_VERSION,
        "lyrics_included": False,
        "role": INDEPENDENT_SCAN_HOLDOUT_ROLE,
        "scan_text_visual_presence_version": (
            SCAN_TEXT_VISUAL_PRESENCE_VERSION
        ),
        "scan_text_page_label_completeness_version": (
            SCAN_TEXT_PAGE_LABEL_COMPLETENESS_VERSION
        ),
        "scan_text_reference_source_version": (
            SCAN_TEXT_REFERENCE_SOURCE_VERSION
        ),
        "reference_page_source_counts": {
            "registered_reference_cache": 2,
            "source_mscz_pdf_rerender": 0,
        },
        "reference_page_count": 2,
        "minimum_visual_presence_ncc": (
            MINIMUM_SAFE_VISUAL_PRESENCE_NCC
        ),
        "forbidden_selection_overlap": [],
        "forbidden_work_overlap": [],
        "split_intersections": EMPTY_INTERSECTIONS,
        "region_dir": str(image_root.resolve()),
        "sources": sources,
        "sources_by_split": {"train": 0, "calibration": 0, "test": 2},
        "sources_with_retained_words_by_split": {
            "train": 0,
            "calibration": 0,
            "test": 2,
        },
        "words_by_split": {"train": 0, "calibration": 0, "test": 2},
    }
    (directory / "prepare-report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    return directory


def test_builds_test_only_detection_labels(tmp_path: Path) -> None:
    dataset = _holdout_dataset(tmp_path)
    output = tmp_path / "labels"
    assert module.main(
        [
            "--project-root",
            str(tmp_path),
            "--dataset-dir",
            str(dataset),
            "--output-dir",
            str(output),
        ]
    ) == 0
    assert (
        output / "test.paddle.det.txt"
    ).read_text(encoding="utf-8").count("\n") == 2
    recognition_lines = (
        output / "test.paddle.txt"
    ).read_text(encoding="utf-8").splitlines()
    assert len(recognition_lines) == 2
    assert recognition_lines[0].startswith(
        "holdout-text/crops/test/"
    )
    report = json.loads(
        (output / "prepare-report.json").read_text(encoding="utf-8")
    )
    assert report["training_use_authorized"] is False
    assert report["forbidden_selection_overlap"] == []
    assert report["forbidden_work_overlap"] == []


def test_training_loader_rejects_independent_holdout(tmp_path: Path) -> None:
    dataset = _holdout_dataset(tmp_path)
    with pytest.raises(ValueError, match="approved training"):
        merge_ocr_detection_labels.load_dataset(
            merge_ocr_detection_labels.DatasetSpec(
                "holdout",
                "scan",
                dataset,
            ),
            project_root=tmp_path,
            required_nonempty_splits=("test",),
        )


def test_converter_rejects_holdout_with_training_overlap(
    tmp_path: Path,
) -> None:
    dataset = _holdout_dataset(tmp_path)
    report_path = dataset / "prepare-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["forbidden_selection_overlap"] = [7]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps training"):
        module.main(
            [
                "--project-root",
                str(tmp_path),
                "--dataset-dir",
                str(dataset),
                "--output-dir",
                str(tmp_path / "labels"),
            ]
        )


def test_converter_rejects_stale_text_selection_contract(
    tmp_path: Path,
) -> None:
    dataset = _holdout_dataset(tmp_path)
    report_path = dataset / "prepare-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["source_text_selection_version"] = "contaminated-legacy"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="selection contract"):
        module.main(
            [
                "--project-root",
                str(tmp_path),
                "--dataset-dir",
                str(dataset),
                "--output-dir",
                str(tmp_path / "labels"),
            ]
        )
