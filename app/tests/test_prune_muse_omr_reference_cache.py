from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.tools import prune_muse_omr_reference_cache as module


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _dataset(tmp_path: Path) -> tuple[Path, Path]:
    region_dir = tmp_path / "regions"
    text_dir = tmp_path / "text"
    reference_dir = region_dir / "reference_pages" / "pair-0001"
    reference_dir.mkdir(parents=True)
    text_dir.mkdir()
    (reference_dir / "page-1.svg").write_text("<svg/>", encoding="utf-8")
    (reference_dir / "page-1.png").write_bytes(b"png")
    region_report = region_dir / "prepare-report.json"
    _write_json(
        region_report,
        {
            "name": "scorescan-muse-omr-registered-scan-regions-v1",
            "role": "external_scan_degraded_training_only",
            "accepted_pairs": 1,
        },
    )
    _write_json(
        text_dir / "prepare-report.json",
        {
            "name": "scorescan-muse-omr-registered-scan-text-v1",
            "role": "external_scan_degraded_training_only",
            "region_dir": str(region_dir.resolve()),
            "region_report_sha256": module.sha256_file(region_report),
            "selected_pairs": 1,
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
                "registered_reference_cache": 1,
                "source_mscz_pdf_rerender": 0,
            },
            "reference_page_count": 1,
            "minimum_visual_presence_ncc": (
                module.MINIMUM_SAFE_VISUAL_PRESENCE_NCC
            ),
        },
    )
    return region_dir, text_dir


def test_dry_run_preserves_cache_and_execute_removes_only_reference_pages(
    tmp_path: Path,
) -> None:
    region_dir, text_dir = _dataset(tmp_path)
    assert module.main(
        ["--region-dir", str(region_dir), "--text-dir", str(text_dir)]
    ) == 0
    assert (region_dir / "reference_pages").is_dir()
    assert module.main(
        [
            "--region-dir",
            str(region_dir),
            "--text-dir",
            str(text_dir),
            "--execute",
        ]
    ) == 0
    assert not (region_dir / "reference_pages").exists()
    assert (region_dir / "prepare-report.json").is_file()
    report = json.loads(
        (region_dir / "reference-cache-prune-report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["executed"] is True
    assert report["removed_files"] == 2
    assert module.main(
        [
            "--region-dir",
            str(region_dir),
            "--text-dir",
            str(text_dir),
            "--execute",
        ]
    ) == 0


def test_refuses_text_report_bound_to_another_region_report(
    tmp_path: Path,
) -> None:
    region_dir, text_dir = _dataset(tmp_path)
    report_path = text_dir / "prepare-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["region_report_sha256"] = "0" * 64
    _write_json(report_path, report)
    with pytest.raises(ValueError, match="not bound"):
        module.validate_completed_consumers(
            region_dir=region_dir,
            text_dir=text_dir,
        )


def test_missing_cache_accepts_only_proven_all_rerender_consumer(
    tmp_path: Path,
) -> None:
    region_dir, text_dir = _dataset(tmp_path)
    report_path = text_dir / "prepare-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["reference_page_source_counts"] = {
        "registered_reference_cache": 0,
        "source_mscz_pdf_rerender": 1,
    }
    _write_json(report_path, report)
    shutil_target = region_dir / "reference_pages"
    import shutil

    shutil.rmtree(shutil_target)

    reference_dir, prune_report = module.validate_completed_consumers(
        region_dir=region_dir,
        text_dir=text_dir,
    )

    assert not reference_dir.exists()
    assert prune_report["cache_absent_consumer_safe"] is True
    assert prune_report["removed_files"] == 0
