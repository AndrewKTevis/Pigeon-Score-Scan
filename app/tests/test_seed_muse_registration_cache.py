from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.tools.prepare_muse_omr_scan_regions import (
    REGISTRATION_QUALITY_POLICY_VERSION,
    REGISTRATION_VERSION,
)
from app.tools.seed_muse_registration_cache import seed_registration_cache


def _source_cache(path: Path) -> Path:
    path.mkdir()
    (path / "prepare-report.json").write_text(
        json.dumps(
            {
                "registration_version": REGISTRATION_VERSION,
                "registration_quality_policy_version": (
                    REGISTRATION_QUALITY_POLICY_VERSION
                ),
            }
        ),
        encoding="utf-8",
    )
    for index, directory in enumerate(
        ("acceptances", "rejections", "pages", "reference_pages")
    ):
        target = path / directory / f"artifact-{index}.bin"
        target.parent.mkdir()
        target.write_bytes(f"cache-{index}".encode())
    return path


def test_seed_hardlinks_only_cache_artifacts_and_not_old_evidence(
    tmp_path: Path,
) -> None:
    source = _source_cache(tmp_path / "source")
    destination = tmp_path / "destination"

    report = seed_registration_cache(source, destination)

    assert report["partition_level_evidence_reused"] is False
    assert report["file_count"] == 4
    assert report["hardlinked"] == 4
    assert not (destination / "prepare-report.json").exists()
    for directory in (
        "acceptances",
        "rejections",
        "pages",
        "reference_pages",
    ):
        source_file = next((source / directory).iterdir())
        destination_file = destination / directory / source_file.name
        assert os.path.samefile(source_file, destination_file)


def test_seed_rejects_completed_or_stale_destination(tmp_path: Path) -> None:
    source = _source_cache(tmp_path / "source")
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "prepare-report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="completed"):
        seed_registration_cache(source, destination)


def test_seed_accepts_intentionally_pruned_reference_render_cache(
    tmp_path: Path,
) -> None:
    source = _source_cache(tmp_path / "source")
    reference_pages = source / "reference_pages"
    for path in reference_pages.iterdir():
        path.unlink()
    reference_pages.rmdir()

    report = seed_registration_cache(source, tmp_path / "destination")

    assert report["file_count"] == 3
    assert report["missing_optional_directories"] == ["reference_pages"]
