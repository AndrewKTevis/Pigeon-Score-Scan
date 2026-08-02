from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.tools.seed_openscore_render_cache import seed_render_cache


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    page = source / "pages" / "piece"
    page.mkdir(parents=True)
    (source / "prepare-report.json").write_text("{}", encoding="utf-8")
    (page / "page-1.svg").write_text("<svg/>", encoding="utf-8")
    (page / "page-1.png").write_bytes(b"png")
    return source


def test_seed_openscore_render_cache_hardlinks_only_renders(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "destination"

    report = seed_render_cache(source, destination)

    assert report["render_files"] == 2
    assert report["hardlinked"] == 2
    assert report["partition_level_evidence_reused"] is False
    assert not (destination / "prepare-report.json").exists()
    assert os.path.samefile(
        source / "pages" / "piece" / "page-1.svg",
        destination / "pages" / "piece" / "page-1.svg",
    )


def test_seed_openscore_render_cache_rejects_completed_destination(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "prepare-report.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="completed"):
        seed_render_cache(source, destination)
