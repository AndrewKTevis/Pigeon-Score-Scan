from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.tools.validate_ocr_artifact_freshness import (
    MINIMUM_SAFE_VISUAL_PRESENCE_NCC,
    SCAN_TEXT_PAGE_LABEL_COMPLETENESS_VERSION,
    SCAN_TEXT_REFERENCE_SOURCE_VERSION,
    SCAN_TEXT_VISUAL_PRESENCE_VERSION,
    SOURCE_TEXT_SELECTION_VERSION,
    validate_holdout_labels_artifact,
    validate_merged_artifact,
    validate_text_artifact,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_manifest(directory: Path, paths: list[Path]) -> None:
    (directory / "dataset.sha256").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in paths),
        encoding="utf-8",
    )


def test_text_artifact_is_bound_to_region_report(tmp_path: Path) -> None:
    region_report = tmp_path / "regions" / "prepare-report.json"
    _write_json(region_report, {"version": 1})
    artifact_report = tmp_path / "text" / "prepare-report.json"
    _write_json(
        artifact_report,
        {
            "source_text_selection_version": SOURCE_TEXT_SELECTION_VERSION,
            "lyrics_included": False,
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
                "registered_reference_cache": 1,
                "source_mscz_pdf_rerender": 0,
            },
            "reference_page_count": 1,
            "minimum_visual_presence_ncc": (
                MINIMUM_SAFE_VISUAL_PRESENCE_NCC
            ),
            "region_dir": str(region_report.parent),
            "region_report_sha256": _sha256(region_report),
        },
    )
    _write_manifest(artifact_report.parent, [artifact_report])
    validate_text_artifact(
        artifact_report=artifact_report,
        source_report=region_report,
    )

    _write_json(region_report, {"version": 2})
    with pytest.raises(ValueError, match="stale"):
        validate_text_artifact(
            artifact_report=artifact_report,
            source_report=region_report,
        )


def test_text_artifact_rejects_stale_visual_presence_contract(
    tmp_path: Path,
) -> None:
    region_report = tmp_path / "regions" / "prepare-report.json"
    _write_json(region_report, {"version": 1})
    artifact_report = tmp_path / "text" / "prepare-report.json"
    _write_json(
        artifact_report,
        {
            "source_text_selection_version": (
                SOURCE_TEXT_SELECTION_VERSION
            ),
            "lyrics_included": False,
            "region_dir": str(region_report.parent),
            "region_report_sha256": _sha256(region_report),
        },
    )
    _write_manifest(artifact_report.parent, [artifact_report])
    with pytest.raises(ValueError, match="visual-presence"):
        validate_text_artifact(
            artifact_report=artifact_report,
            source_report=region_report,
        )


def test_holdout_labels_reject_wrong_source_and_tampering(
    tmp_path: Path,
) -> None:
    source_report = tmp_path / "text" / "prepare-report.json"
    _write_json(source_report, {"version": 2})
    artifact_dir = tmp_path / "labels"
    recognition = artifact_dir / "test.paddle.txt"
    detection = artifact_dir / "test.paddle.det.txt"
    recognition.parent.mkdir(parents=True)
    recognition.write_text("crop.png\tAllegro\n", encoding="utf-8")
    detection.write_text("page.png\t[]\n", encoding="utf-8")
    artifact_report = artifact_dir / "prepare-report.json"
    _write_json(
        artifact_report,
        {
            "source_report": str(source_report),
            "source_report_sha256": _sha256(source_report),
            "output_counts": {
                recognition.name: 1,
                detection.name: 1,
            },
            "output_sha256": {
                recognition.name: _sha256(recognition),
                detection.name: _sha256(detection),
            },
        },
    )
    _write_manifest(
        artifact_dir,
        [recognition, detection, artifact_report],
    )
    validate_holdout_labels_artifact(
        artifact_report=artifact_report,
        source_report=source_report,
    )

    recognition.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_holdout_labels_artifact(
            artifact_report=artifact_report,
            source_report=source_report,
        )


def test_merged_labels_require_exact_current_source_set(tmp_path: Path) -> None:
    first = tmp_path / "clean" / "prepare-report.json"
    second = tmp_path / "scan" / "prepare-report.json"
    _write_json(first, {"source": "clean"})
    _write_json(second, {"source": "scan"})
    artifact_report = tmp_path / "merged" / "merge-report.json"
    _write_json(
        artifact_report,
        {
            "datasets": [
                {
                    "directory": str(first.parent),
                    "prepare_report_sha256": _sha256(first),
                },
                {
                    "directory": str(second.parent),
                    "prepare_report_sha256": _sha256(second),
                },
            ]
        },
    )
    _write_manifest(artifact_report.parent, [artifact_report])
    validate_merged_artifact(
        artifact_report=artifact_report,
        source_reports=[first, second],
    )

    _write_json(second, {"source": "new-scan"})
    with pytest.raises(ValueError, match="stale"):
        validate_merged_artifact(
            artifact_report=artifact_report,
            source_reports=[first, second],
        )
