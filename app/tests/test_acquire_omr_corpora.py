from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest


TOOL = Path(__file__).resolve().parents[1] / "tools" / "acquire_omr_corpora.py"
SPEC = importlib.util.spec_from_file_location("acquire_omr_corpora", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_catalog_uses_https_and_unique_targets() -> None:
    assets = list(MODULE.CATALOG.values())
    assert len(assets) >= 6
    assert all(asset.url.startswith("https://") for asset in assets)
    assert all(asset.expected_bytes > 0 for asset in assets)
    assert len({asset.filename for asset in assets}) == len(assets)
    assert len({asset.extracted_directory for asset in assets}) == len(assets)
    assert MODULE.CATALOG["doremi_v1"].license_review_required is True
    assert MODULE.CATALOG["deepscores_v2_dense"].license == "CC-BY-4.0"
    quartets = MODULE.CATALOG["openscore_string_quartets"]
    assert quartets.license == "CC0-1.0"
    assert "d13289cd70797da94646e5cf64f7296a4c4fee40" in quartets.url
    lieder = MODULE.CATALOG["openscore_lieder"]
    assert lieder.license == "CC0-1.0"
    assert "6b2dc542ce2e8aa4b78c8ee62103b210efc07015" in lieder.url
    assert "candidate_score_ids_excluded" in lieder.role


def test_safe_member_path_rejects_traversal(tmp_path: Path) -> None:
    for value in ("../escape", "/absolute", "C:/absolute", "safe/../../escape"):
        with pytest.raises(ValueError):
            MODULE._safe_member_path(tmp_path, value)
    assert MODULE._safe_member_path(tmp_path, "safe/file.txt") == tmp_path / "safe" / "file.txt"


def test_extract_zip_enforces_path_safety(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError):
        MODULE.extract_archive(archive, tmp_path / "target", 1024)
    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path / ".target.extracting").exists()


def test_extract_tar_enforces_uncompressed_limit(tmp_path: Path) -> None:
    archive = tmp_path / "bounded.tar.gz"
    payload = b"x" * 2048
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo("data/payload.bin")
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError, match="extraction byte limit"):
        MODULE.extract_archive(archive, tmp_path / "target", 1024)
    assert not (tmp_path / ".target.extracting").exists()


def test_extract_zip_records_exact_statistics(tmp_path: Path) -> None:
    archive = tmp_path / "valid.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("a.txt", "abc")
        bundle.writestr("nested/b.bin", b"12345")
    result = MODULE.extract_archive(archive, tmp_path / "target", 1024)
    assert result == {"files": 2, "bytes": 8}
    assert (tmp_path / "target" / "a.txt").read_text(encoding="utf-8") == "abc"
