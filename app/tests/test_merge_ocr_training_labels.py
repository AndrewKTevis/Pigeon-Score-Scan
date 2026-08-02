from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
from PIL import Image

from app.tools import merge_ocr_training_labels as module


def _make_dataset(
    project_root: Path,
    *,
    name: str,
    kind: str,
    rows_per_split: int = 2,
) -> Path:
    directory = project_root / name
    image_root = project_root / f"{name}-pages"
    directory.mkdir()
    image_root.mkdir()
    sources = []
    for split in module.SPLITS:
        rows = []
        for index in range(rows_per_split):
            source_key = f"{split}/source-{index}"
            page = image_root / split / f"{index}.png"
            page.parent.mkdir(parents=True, exist_ok=True)
            mode = "RGB" if kind == "clean" else "L"
            color = (
                (220 - index, 230, 240)
                if mode == "RGB"
                else 220 - index
            )
            Image.new(mode, (48, 32), color).save(page)
            crop = directory / "crops" / split / f"{index}.png"
            crop.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(page) as source_image:
                source_image.convert(mode).crop((4, 5, 28, 21)).save(crop)
            rows.append(
                {
                    "split": split,
                    "source_key": source_key,
                    "page": 1,
                    "word_index": index,
                    "image": page.relative_to(image_root).as_posix(),
                    "box_xyxy": [4, 5, 28, 21],
                    "crop_image": crop.relative_to(directory).as_posix(),
                    "text": f"Allegro {index}",
                    "text_role": "supported",
                    "visual_presence_ncc": 0.9,
                }
            )
            hash_key = (
                "source_sha256"
                if kind == "clean"
                else "work_fingerprint"
            )
            sources.append(
                {
                    "source_key": source_key,
                    "split": split,
                    hash_key: hashlib.sha256(
                        f"{name}/{source_key}".encode()
                    ).hexdigest(),
                }
            )
        (directory / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
    report = {
        "source_text_selection_version": module.SOURCE_TEXT_SELECTION_VERSION,
        "lyrics_included": False,
        "split_intersections": module.EMPTY_INTERSECTIONS,
        "sources": sources,
        "words_by_split": {
            split: rows_per_split
            for split in module.SPLITS
        },
    }
    if kind == "clean":
        report.update(
            {
                "purpose": "rendered exact-text OCR training/calibration",
                "self_contained_crops": True,
                "region_dataset_dir": str(image_root.resolve()),
                "excluded_pdf_geometry_words_by_split": {
                    split: {}
                    for split in module.SPLITS
                },
            }
        )
    else:
        report["role"] = (
            "training_only_disjoint_from_external_release_holdout"
        )
        report["region_dir"] = str(image_root.resolve())
        report["scan_text_visual_presence_version"] = (
            module.SCAN_TEXT_VISUAL_PRESENCE_VERSION
        )
        report["scan_text_page_label_completeness_version"] = (
            module.SCAN_TEXT_PAGE_LABEL_COMPLETENESS_VERSION
        )
        report["scan_text_reference_source_version"] = (
            module.SCAN_TEXT_REFERENCE_SOURCE_VERSION
        )
        report["reference_page_source_counts"] = {
            "registered_reference_cache": rows_per_split * len(module.SPLITS),
            "source_mscz_pdf_rerender": 0,
        }
        report["reference_page_count"] = (
            rows_per_split * len(module.SPLITS)
        )
        report["minimum_visual_presence_ncc"] = (
            module.MINIMUM_SAFE_VISUAL_PRESENCE_NCC
        )
    (directory / "prepare-report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    return directory


def test_load_dataset_validates_crops_and_project_relative_paths(
    tmp_path: Path,
) -> None:
    clean = _make_dataset(tmp_path, name="clean", kind="clean")
    rows, report = module.load_dataset(
        module.DatasetSpec("clean", "clean", clean),
        project_root=tmp_path,
    )
    assert report["rows_by_split"] == {
        "train": 2,
        "calibration": 2,
        "test": 2,
    }
    assert report["crop_source_pixel_bindings_verified"] == 6
    assert rows["train"][0].project_relative_crop.startswith(
        "clean/crops/train/"
    )


def test_balancing_repeats_scan_rows_deterministically() -> None:
    make_row = lambda kind, index: module.LabelRow(  # noqa: E731
        dataset=kind,
        kind=kind,
        split="train",
        source_key=f"{kind}-{index}",
        project_relative_crop=f"{kind}/{index}.png",
        text=str(index),
    )
    clean = [make_row("clean", index) for index in range(10)]
    scan = [make_row("scan", index) for index in range(2)]
    first, report = module.build_balanced_training_rows(
        clean_rows=clean,
        scan_rows=scan,
        scan_target_fraction=0.5,
        seed=7,
    )
    second, _ = module.build_balanced_training_rows(
        clean_rows=clean,
        scan_rows=scan,
        scan_target_fraction=0.5,
        seed=7,
    )
    assert first == second
    assert report["effective_scan_rows"] == 10
    assert report["achieved_scan_fraction"] == 0.5


def test_balancing_caps_repeated_labels_before_domain_balance() -> None:
    def make(kind: str, index: int, text: str) -> module.LabelRow:
        return module.LabelRow(
            dataset=kind,
            kind=kind,
            split="train",
            source_key=f"{kind}-{index}",
            project_relative_crop=f"{kind}/{index}.png",
            text=text,
        )

    clean = [make("clean", index, "Tempo") for index in range(20)]
    clean.extend(make("clean", 100 + index, f"rare-{index}") for index in range(4))
    scan = [make("scan", index, "Tempo") for index in range(10)]
    first, report = module.build_balanced_training_rows(
        clean_rows=clean,
        scan_rows=scan,
        scan_target_fraction=0.5,
        seed=17,
        maximum_rows_per_normalized_text=3,
    )
    second, _ = module.build_balanced_training_rows(
        clean_rows=clean,
        scan_rows=scan,
        scan_target_fraction=0.5,
        seed=17,
        maximum_rows_per_normalized_text=3,
    )
    assert first == second
    assert report["unique_clean_rows"] == 7
    assert report["unique_scan_rows"] == 3
    assert report["frequency_cap_dropped_clean_rows"] == 17
    assert report["frequency_cap_dropped_scan_rows"] == 7


def test_main_writes_isolated_scan_validation_and_auditable_hashes(
    tmp_path: Path,
) -> None:
    clean = _make_dataset(tmp_path, name="clean", kind="clean")
    scan = _make_dataset(tmp_path, name="scan", kind="scan")
    output = tmp_path / "merged"

    assert module.main(
        [
            "--project-root",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--clean-dataset",
            f"lieder={clean}",
            "--scan-dataset",
            f"muse_scan={scan}",
            "--scan-target-fraction",
            "0.5",
        ]
    ) == 0

    report = json.loads(
        (output / "merge-report.json").read_text(encoding="utf-8")
    )
    assert report["balance"]["achieved_scan_fraction"] == 0.5
    assert (output / "calibration.scan.paddle.txt").read_text(
        encoding="utf-8"
    ).count("\n") == 2
    assert (output / "test.scan.paddle.txt").read_text(
        encoding="utf-8"
    ).count("\n") == 2
    assert "train.balanced.paddle.txt" in (
        output / "dataset.sha256"
    ).read_text(encoding="utf-8")


def test_rejects_holdout_role_masquerading_as_scan_training(
    tmp_path: Path,
) -> None:
    scan = _make_dataset(tmp_path, name="scan", kind="scan")
    report_path = scan / "prepare-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["role"] = "release_holdout"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="approved training"):
        module.load_dataset(
            module.DatasetSpec("scan", "scan", scan),
            project_root=tmp_path,
        )


def test_rejects_scan_dataset_without_visual_presence_contract(
    tmp_path: Path,
) -> None:
    scan = _make_dataset(tmp_path, name="scan", kind="scan")
    report_path = scan / "prepare-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.pop("scan_text_visual_presence_version")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="visual-presence contract"):
        module.load_dataset(
            module.DatasetSpec("scan", "scan", scan),
            project_root=tmp_path,
        )


def test_rejects_scan_row_below_visual_presence_floor(
    tmp_path: Path,
) -> None:
    scan = _make_dataset(tmp_path, name="scan", kind="scan")
    path = scan / "train.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["visual_presence_ncc"] = 0.01
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not visually present"):
        module.load_dataset(
            module.DatasetSpec("scan", "scan", scan),
            project_root=tmp_path,
        )


def test_rejects_scan_dataset_with_partial_page_contract(
    tmp_path: Path,
) -> None:
    scan = _make_dataset(tmp_path, name="scan", kind="scan")
    report_path = scan / "prepare-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.pop("scan_text_page_label_completeness_version")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="partial pages"):
        module.load_dataset(
            module.DatasetSpec("scan", "scan", scan),
            project_root=tmp_path,
        )


def test_rejects_duplicate_source_content_across_splits(
    tmp_path: Path,
) -> None:
    clean = _make_dataset(tmp_path, name="clean", kind="clean")
    report_path = clean / "prepare-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["sources"][2]["source_sha256"] = report["sources"][0][
        "source_sha256"
    ]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate source content"):
        module.load_dataset(
            module.DatasetSpec("clean", "clean", clean),
            project_root=tmp_path,
        )


def test_rejects_excessive_clean_geometry_exclusion(
    tmp_path: Path,
) -> None:
    clean = _make_dataset(tmp_path, name="clean", kind="clean")
    report_path = clean / "prepare-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["excluded_pdf_geometry_words_by_split"]["test"] = {
        "outside_page": 1
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="geometry exclusion fraction"):
        module.load_dataset(
            module.DatasetSpec("clean", "clean", clean),
            project_root=tmp_path,
        )


def test_rejects_row_not_proven_as_supported_score_text(
    tmp_path: Path,
) -> None:
    clean = _make_dataset(tmp_path, name="clean", kind="clean")
    path = clean / "train.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["text_role"] = "unproven"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source-proven"):
        module.load_dataset(
            module.DatasetSpec("clean", "clean", clean),
            project_root=tmp_path,
        )


def test_rejects_crop_pixels_not_bound_to_source_page(
    tmp_path: Path,
) -> None:
    clean = _make_dataset(tmp_path, name="clean", kind="clean")
    crop = clean / "crops" / "train" / "0.png"
    Image.new("RGB", (24, 16), "black").save(crop)
    with pytest.raises(ValueError, match="crop pixels"):
        module.load_dataset(
            module.DatasetSpec("clean", "clean", clean),
            project_root=tmp_path,
        )
