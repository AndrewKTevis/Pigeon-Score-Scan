from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
from PIL import Image

from app.tools import merge_ocr_detection_labels as module


def _make_dataset(
    project_root: Path,
    *,
    name: str,
    kind: str,
    sources_per_split: int = 2,
    words_per_page: int = 2,
    exhaustive_detection: bool = False,
) -> Path:
    if kind not in {"clean", "scan"}:
        raise ValueError("unsupported fixture kind")
    directory = project_root / name
    image_root = project_root / f"{name}-pages"
    directory.mkdir()
    image_root.mkdir()
    sources = []
    words_by_split = {}
    for split_index, split in enumerate(module.SPLITS):
        rows = []
        for source_index in range(sources_per_split):
            source_key = f"{split}/source-{source_index}"
            page = image_root / split / f"{source_index}.png"
            page.parent.mkdir(parents=True, exist_ok=True)
            shade = 250 - split_index * 10 - source_index
            domain_shade = 240 if kind == "scan" else 250
            Image.new(
                "RGB",
                (128, 64),
                (shade, domain_shade, 250),
            ).save(page)
            for word_index in range(words_per_page):
                left = 5 + 40 * word_index
                row = {
                        "split": split,
                        "source_key": source_key,
                        "page": 1,
                        "word_index": word_index,
                        "image": page.relative_to(image_root).as_posix(),
                        "box_xyxy": [left, 10, left + 30, 30],
                        "text": f"Allegro {word_index}",
                        "text_role": (
                            "visible_text"
                            if exhaustive_detection
                            else "supported"
                        ),
                        "visual_presence_ncc": 0.9,
                    }
                if exhaustive_detection:
                    row["hard_negative_sampling_authorized"] = True
                    row["page_geometry_exclusion_count"] = 0
                rows.append(row)
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
        words_by_split[split] = len(rows)
        (directory / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
    report = {
        "source_text_selection_version": module.SOURCE_TEXT_SELECTION_VERSION,
        "lyrics_included": exhaustive_detection,
        "split_intersections": module.EMPTY_INTERSECTIONS,
        "sources": sources,
        "words_by_split": words_by_split,
    }
    if kind == "clean":
        report.update(
            {
                "purpose": (
                    "rendered exhaustive visible-text detection "
                    "training/calibration"
                    if exhaustive_detection
                    else "rendered exact-text OCR training/calibration"
                ),
                "self_contained_crops": not exhaustive_detection,
                "region_dataset_dir": str(image_root.resolve()),
                "excluded_pdf_geometry_words_by_split": {
                    split: {}
                    for split in module.SPLITS
                },
            }
        )
    else:
        report.update(
            {
                "role": (
                    "training_only_disjoint_from_external_release_holdout"
                ),
                "forbidden_selection_overlap": [],
                "forbidden_work_overlap": [],
                "region_dir": str(image_root.resolve()),
                "scan_text_visual_presence_version": (
                    module.SCAN_TEXT_VISUAL_PRESENCE_VERSION
                ),
                "scan_text_page_label_completeness_version": (
                    module.SCAN_TEXT_PAGE_LABEL_COMPLETENESS_VERSION
                ),
                "scan_text_reference_source_version": (
                    module.SCAN_TEXT_REFERENCE_SOURCE_VERSION
                ),
                "reference_page_source_counts": {
                    "registered_reference_cache": (
                        sources_per_split * len(module.SPLITS)
                    ),
                    "source_mscz_pdf_rerender": 0,
                },
                "reference_page_count": (
                    sources_per_split * len(module.SPLITS)
                ),
                "minimum_visual_presence_ncc": (
                    module.MINIMUM_SAFE_VISUAL_PRESENCE_NCC
                ),
                "excluded_pdf_geometry_words_by_split": {
                    split: {}
                    for split in module.SPLITS
                },
            }
        )
    if exhaustive_detection:
        report.update(
            {
                "purpose": (
                    "rendered exhaustive visible-text detection "
                    "training/calibration"
                    if kind == "clean"
                    else "registered exhaustive visible-text detection "
                    "training/calibration"
                ),
                "detection_label_contract": (
                    module.EXHAUSTIVE_DETECTION_LABEL_CONTRACT
                ),
                "all_usable_pdf_text_included": True,
                "positive_region_coverage": (
                    "all_nonmusic_font_visible_pdf_text"
                    if kind == "clean"
                    else "all_registered_nonmusic_font_visible_pdf_text"
                ),
                "recall_evaluation_authorized": True,
                "precision_evaluation_authorized": True,
                "hmean_evaluation_authorized": True,
                "hard_negative_sampling_authorized": True,
                "hard_negative_authorization_scope": (
                    "page_without_pdf_geometry_exclusions"
                    if kind == "clean"
                    else (
                        "retained_registered_page_all_visible_text_verified"
                    )
                ),
                "unlabelled_visible_text_may_be_present": False,
                "text_pages_by_split": {
                    split: sources_per_split
                    for split in module.SPLITS
                },
                "hard_negative_authorized_pages_by_split": {
                    split: sources_per_split
                    for split in module.SPLITS
                },
                "geometry_excluded_text_pages_by_split": {
                    split: 0
                    for split in module.SPLITS
                },
            }
        )
        if kind == "scan":
            report["detection_page_label_completeness_version"] = (
                module.EXHAUSTIVE_REGISTERED_DETECTION_PAGE_CONTRACT
            )
            report["detection_page_selection_policy"] = (
                module.EXHAUSTIVE_REGISTERED_DETECTION_SELECTION_POLICY
            )
    (directory / "prepare-report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    return directory


def test_load_groups_words_by_page_and_validates_geometry(
    tmp_path: Path,
) -> None:
    clean = _make_dataset(tmp_path, name="clean", kind="clean")
    pages, report = module.load_dataset(
        module.DatasetSpec("clean", "clean", clean),
        project_root=tmp_path,
    )
    assert report["words_by_split"]["train"] == 4
    assert report["pages_by_split"]["train"] == 2
    assert len(pages["train"][0].annotations) == 2
    assert pages["train"][0].width == 128
    assert pages["train"][0].height == 64


def test_load_rejects_out_of_bounds_box(tmp_path: Path) -> None:
    clean = _make_dataset(
        tmp_path,
        name="clean",
        kind="clean",
        sources_per_split=1,
        words_per_page=1,
    )
    path = clean / "test.jsonl"
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    row["box_xyxy"] = [5, 10, 200, 30]
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="out-of-bounds"):
        module.load_dataset(
            module.DatasetSpec("clean", "clean", clean),
            project_root=tmp_path,
        )


def test_load_rejects_unproven_text_box(tmp_path: Path) -> None:
    clean = _make_dataset(
        tmp_path,
        name="clean",
        kind="clean",
        sources_per_split=1,
        words_per_page=1,
    )
    path = clean / "test.jsonl"
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    row["text_role"] = "unproven"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="coverage contract"):
        module.load_dataset(
            module.DatasetSpec("clean", "clean", clean),
            project_root=tmp_path,
        )


def test_balancing_repeats_scan_pages_deterministically() -> None:
    annotation = module.DetectionAnnotation(
        transcription="p",
        points=((1, 1), (3, 1), (3, 3), (1, 3)),
    )

    def make(kind: str, index: int) -> module.PageLabel:
        return module.PageLabel(
            dataset=kind,
            kind=kind,
            split="train",
            source_key=f"{kind}/{index}",
            project_relative_image=f"{kind}/{index}.png",
            width=20,
            height=20,
            image_sha256=f"{index:064x}",
            annotations=(annotation,),
        )

    clean = [make("clean", index) for index in range(10)]
    scan = [make("scan", index) for index in range(2)]
    first, report = module.build_balanced_training_pages(
        clean_pages=clean,
        scan_pages=scan,
        scan_target_fraction=0.5,
        seed=9,
    )
    second, _ = module.build_balanced_training_pages(
        clean_pages=clean,
        scan_pages=scan,
        scan_target_fraction=0.5,
        seed=9,
    )
    assert first == second
    assert report["effective_scan_pages"] == 10
    assert report["achieved_scan_fraction"] == 0.5


def test_exact_same_split_page_alias_is_deduplicated_but_conflict_fails() -> None:
    annotation = module.DetectionAnnotation(
        transcription="Allegro",
        points=((1, 1), (9, 1), (9, 4), (1, 4)),
    )

    def page(
        dataset: str,
        source: str,
        split: str = "train",
        annotations: tuple[module.DetectionAnnotation, ...] = (
            annotation,
        ),
    ) -> module.PageLabel:
        return module.PageLabel(
            dataset=dataset,
            kind="clean",
            split=split,
            source_key=source,
            project_relative_image=f"{dataset}/{source}.png",
            width=20,
            height=10,
            image_sha256="a" * 64,
            annotations=annotations,
            hard_negative_sampling_authorized=True,
        )

    specs = [
        module.DatasetSpec("left", "clean", Path("left")),
        module.DatasetSpec("right", "clean", Path("right")),
    ]
    loaded = {
        "left": {
            "train": [page("left", "one")],
            "calibration": [],
            "test": [],
        },
        "right": {
            "train": [page("right", "two")],
            "calibration": [],
            "test": [],
        },
    }
    aliases = module.deduplicate_same_split_exact_pages(loaded, specs)

    assert len(aliases) == 1
    assert aliases[0]["resolution"] == (
        "same_split_exact_annotation_deduplicated"
    )
    assert loaded["left"]["train"]
    assert loaded["right"]["train"] == []

    conflicting = {
        "left": {
            "train": [page("left", "one")],
            "calibration": [],
            "test": [],
        },
        "right": {
            "train": [],
            "calibration": [page("right", "two", "calibration")],
            "test": [],
        },
    }
    with pytest.raises(RuntimeError, match="crosses a split/domain"):
        module.deduplicate_same_split_exact_pages(conflicting, specs)


def test_main_writes_paddle_polygons_and_independent_scan_test(
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
    test_lines = (
        output / "test.scan.paddle.det.txt"
    ).read_text(encoding="utf-8").splitlines()
    assert len(test_lines) == 2
    _image, annotations = test_lines[0].split("\t", 1)
    decoded = json.loads(annotations)
    assert len(decoded) == 2
    assert decoded[0]["points"] == [
        [5.0, 10.0],
        [35.0, 10.0],
        [35.0, 30.0],
        [5.0, 30.0],
    ]
    report = json.loads(
        (output / "merge-report.json").read_text(encoding="utf-8")
    )
    assert report["balance"]["achieved_scan_fraction"] == 0.5
    assert report["label_coverage_contract"] == {
        "positive_region_coverage": "mixed_page_scoped_contracts",
        "recall_evaluation_authorized": True,
        "precision_evaluation_authorized": False,
        "hmean_evaluation_authorized": False,
        "postprocess_threshold_selection_authorized": False,
        "hard_negative_sampling_authorized": False,
        "hard_negative_authorization_scope": "per_page_annotation",
        "hard_negative_authorized_unique_train_pages": 0,
        "unlabelled_visible_text_may_be_present": True,
    }
    assert all(
        dataset["hard_negative_sampling_authorized"] is False
        for dataset in report["datasets"]
    )
    assert "train.balanced.paddle.det.txt" in (
        output / "dataset.sha256"
    ).read_text(encoding="utf-8")


def test_exhaustive_clean_pages_carry_page_scoped_negative_authorization(
    tmp_path: Path,
) -> None:
    clean = _make_dataset(
        tmp_path,
        name="clean",
        kind="clean",
        exhaustive_detection=True,
    )
    scan = _make_dataset(tmp_path, name="scan", kind="scan")
    output = tmp_path / "merged"

    assert module.main(
        [
            "--project-root",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--clean-dataset",
            f"clean={clean}",
            "--scan-dataset",
            f"scan={scan}",
            "--scan-target-fraction",
            "0.5",
        ]
    ) == 0

    clean_line = (
        output / "train.clean.paddle.det.txt"
    ).read_text(encoding="utf-8").splitlines()[0]
    scan_line = (
        output / "train.scan.paddle.det.txt"
    ).read_text(encoding="utf-8").splitlines()[0]
    clean_annotations = json.loads(clean_line.split("\t", 1)[1])
    scan_annotations = json.loads(scan_line.split("\t", 1)[1])
    assert all(
        row["hard_negative_sampling_authorized"] is True
        for row in clean_annotations
    )
    assert all(
        "hard_negative_sampling_authorized" not in row
        for row in scan_annotations
    )
    report = json.loads(
        (output / "merge-report.json").read_text(encoding="utf-8")
    )
    assert (
        report["label_coverage_contract"][
            "hard_negative_authorized_unique_train_pages"
        ]
        == 2
    )
    assert (
        report["label_coverage_contract"][
            "hard_negative_sampling_authorized"
        ]
        is True
    )
    assert (
        report["label_coverage_contract"][
            "precision_evaluation_authorized"
        ]
        is False
    )


def test_geometry_exclusion_disables_only_the_affected_clean_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clean = _make_dataset(
        tmp_path,
        name="clean",
        kind="clean",
        exhaustive_detection=True,
    )
    train_path = clean / "train.jsonl"
    rows = [
        json.loads(line)
        for line in train_path.read_text(encoding="utf-8").splitlines()
    ]
    affected_source = rows[0]["source_key"]
    for row in rows:
        if row["source_key"] == affected_source:
            row["hard_negative_sampling_authorized"] = False
            row["page_geometry_exclusion_count"] = 1
    train_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    report_path = clean / "prepare-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["excluded_pdf_geometry_words_by_split"]["train"] = {
        "materially_clipped": 1
    }
    report["hard_negative_authorized_pages_by_split"]["train"] = 1
    report["geometry_excluded_text_pages_by_split"]["train"] = 1
    report["precision_evaluation_authorized"] = False
    report["hmean_evaluation_authorized"] = False
    report["unlabelled_visible_text_may_be_present"] = True
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        module,
        "MAXIMUM_CLEAN_GEOMETRY_EXCLUSION_FRACTION",
        1.0,
    )

    pages, dataset_report = module.load_dataset(
        module.DatasetSpec("clean", "clean", clean),
        project_root=tmp_path,
    )

    assert sorted(
        page.hard_negative_sampling_authorized
        for page in pages["train"]
    ) == [False, True]
    assert (
        dataset_report["hard_negative_authorized_pages_by_split"]["train"]
        == 1
    )
    assert dataset_report["precision_evaluation_authorized"] is False
    assert dataset_report["unlabelled_visible_text_may_be_present"] is True


def test_exhaustive_registered_scan_authorizes_precision_and_negatives(
    tmp_path: Path,
) -> None:
    clean = _make_dataset(
        tmp_path,
        name="clean",
        kind="clean",
        exhaustive_detection=True,
    )
    scan = _make_dataset(
        tmp_path,
        name="scan",
        kind="scan",
        exhaustive_detection=True,
    )
    output = tmp_path / "merged"

    assert module.main(
        [
            "--project-root",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--clean-dataset",
            f"clean={clean}",
            "--scan-dataset",
            f"scan={scan}",
            "--scan-target-fraction",
            "0.5",
        ]
    ) == 0

    scan_annotations = json.loads(
        (output / "calibration.scan.paddle.det.txt")
        .read_text(encoding="utf-8")
        .splitlines()[0]
        .split("\t", 1)[1]
    )
    assert all(
        annotation["hard_negative_sampling_authorized"] is True
        for annotation in scan_annotations
    )
    report = json.loads(
        (output / "merge-report.json").read_text(encoding="utf-8")
    )
    coverage = report["label_coverage_contract"]
    assert coverage["precision_evaluation_authorized"] is True
    assert coverage["hmean_evaluation_authorized"] is True
    assert coverage["postprocess_threshold_selection_authorized"] is True
    assert coverage["unlabelled_visible_text_may_be_present"] is False
