from __future__ import annotations

from collections import namedtuple
from dataclasses import replace
from pathlib import Path

import pytest

from scorescan.config import Settings
from scorescan.storage import (
    StorageCapacityError,
    directory_size_bounded,
    require_free_space,
    require_workspace_capacity,
)


def test_directory_size_is_bounded_and_ignores_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "a.bin").write_bytes(b"a" * 20)
    sub = root / "sub"
    sub.mkdir()
    (sub / "b.bin").write_bytes(b"b" * 30)
    try:
        (root / "loop").symlink_to(root, target_is_directory=True)
    except OSError:
        pass
    assert directory_size_bounded(root, 100) == 50
    assert directory_size_bounded(root, 10) >= 20


def test_free_space_guard_preserves_reserve(tmp_path: Path, monkeypatch) -> None:
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr("scorescan.storage.shutil.disk_usage", lambda _path: Usage(1000, 850, 150))
    with pytest.raises(StorageCapacityError, match="当前仅有"):
        require_free_space(tmp_path, required_bytes=60, reserve_bytes=100, context="测试写入")
    require_free_space(tmp_path, required_bytes=40, reserve_bytes=100, context="测试写入")


def test_workspace_quota_rejects_growth_before_write(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "portable"
    settings = replace(
        Settings.from_root(root),
        minimum_free_space_bytes=0,
        max_workspace_bytes=100,
    )
    settings.workspace.mkdir(parents=True)
    (settings.workspace / "existing.bin").write_bytes(b"x" * 80)
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr("scorescan.storage.shutil.disk_usage", lambda _path: Usage(10_000, 100, 9_900))
    with pytest.raises(StorageCapacityError, match="工作区超过"):
        require_workspace_capacity(settings, additional_bytes=30, context="测试任务")


def test_directory_size_fails_closed_on_unreadable_entry(tmp_path, monkeypatch):
    import scorescan.storage as storage

    target = tmp_path / "workspace"
    target.mkdir()
    real_scandir = storage.os.scandir

    def guarded(path):
        if Path(path) == target:
            raise PermissionError("denied")
        return real_scandir(path)

    monkeypatch.setattr(storage.os, "scandir", guarded)
    with pytest.raises(storage.StorageCapacityError, match="无法审计工作区目录"):
        storage.directory_size_bounded(target, 100)
