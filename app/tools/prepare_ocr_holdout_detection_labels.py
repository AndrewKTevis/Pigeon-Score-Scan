#!/usr/bin/env python3
"""Build text-detection labels for the independent registered-scan holdout.

This deliberately accepts only the non-training benchmark role.  It cannot be
used by the training merger, and it writes no training or calibration labels.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from app.tools.merge_ocr_detection_labels import (
    DatasetSpec,
    _write_labels as _write_detection_labels,
    load_dataset as load_detection_dataset,
)
from app.tools.merge_ocr_training_labels import (
    EMPTY_INTERSECTIONS,
    INDEPENDENT_SCAN_HOLDOUT_ROLE,
    _write_labels as _write_recognition_labels,
    load_dataset as load_recognition_dataset,
    sha256_file,
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.project_root = args.project_root.resolve()
    args.dataset_dir = args.dataset_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    spec = DatasetSpec(
        name="muse_scan_holdout",
        kind="scan",
        directory=args.dataset_dir,
    )
    pages_by_split, dataset_report = load_detection_dataset(
        spec,
        project_root=args.project_root,
        allowed_scan_roles=(INDEPENDENT_SCAN_HOLDOUT_ROLE,),
        required_nonempty_splits=("test",),
    )
    if pages_by_split["train"] or pages_by_split["calibration"]:
        raise ValueError(
            "independent OCR holdout must not contain training/calibration pages"
        )
    if (
        dataset_report["forbidden_selection_overlap"] != []
        or dataset_report.get("forbidden_work_overlap") != []
    ):
        raise ValueError("independent OCR holdout overlaps training selection")

    rows_by_split, recognition_dataset_report = load_recognition_dataset(
        spec,
        project_root=args.project_root,
        allowed_scan_roles=(INDEPENDENT_SCAN_HOLDOUT_ROLE,),
    )
    if rows_by_split["train"] or rows_by_split["calibration"]:
        raise ValueError(
            "independent OCR holdout must not contain training/calibration rows"
        )
    word_count = len(rows_by_split["test"])
    detection_word_count = sum(
        len(page.annotations) for page in pages_by_split["test"]
    )
    if (
        word_count <= 0
        or word_count != detection_word_count
        or word_count
        != int(dataset_report["words_by_split"].get("test", -1))
        or word_count
        != int(recognition_dataset_report["rows_by_split"].get("test", -1))
    ):
        raise ValueError(
            "independent OCR recognition/detection word coverage differs"
        )

    recognition_labels_path = args.output_dir / "test.paddle.txt"
    recognition_count = _write_recognition_labels(
        recognition_labels_path,
        rows_by_split["test"],
    )
    detection_labels_path = args.output_dir / "test.paddle.det.txt"
    page_count = _write_detection_labels(
        detection_labels_path,
        pages_by_split["test"],
    )
    if page_count <= 0:
        raise ValueError("independent OCR holdout has no detection pages")
    source_report_path = args.dataset_dir / "prepare-report.json"
    report = {
        "schema_version": 1,
        "name": "scorescan-independent-scan-ocr-detection-labels-v1",
        "purpose": "independent_ocr_detection_runtime_evaluation_only",
        "role": INDEPENDENT_SCAN_HOLDOUT_ROLE,
        "training_use_authorized": False,
        "integration_authorized": False,
        "project_root": str(args.project_root),
        "source_dataset": dataset_report,
        "recognition_source_dataset": recognition_dataset_report,
        "source_report": str(source_report_path),
        "source_report_sha256": sha256_file(source_report_path),
        "source_split_intersections": EMPTY_INTERSECTIONS,
        "forbidden_selection_overlap": [],
        "forbidden_work_overlap": [],
        "output_counts": {
            "test.paddle.txt": recognition_count,
            "test.paddle.det.txt": page_count,
        },
        "output_sha256": {
            "test.paddle.txt": sha256_file(recognition_labels_path),
            "test.paddle.det.txt": sha256_file(detection_labels_path),
        },
    }
    report_path = args.output_dir / "prepare-report.json"
    _atomic_json(report_path, report)
    (args.output_dir / "dataset.sha256").write_text(
        "\n".join(
            (
                f"{sha256_file(recognition_labels_path)}  "
                f"{recognition_labels_path.name}",
                f"{sha256_file(detection_labels_path)}  "
                f"{detection_labels_path.name}",
                f"{sha256_file(report_path)}  {report_path.name}",
                "",
            )
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "pages": page_count,
                "words": word_count,
                "role": INDEPENDENT_SCAN_HOLDOUT_ROLE,
                "training_use_authorized": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
