from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path

import pytest

from scorescan.instance_lock import WorkspaceInstanceError, WorkspaceInstanceLock


def _hold_lock(path: str, ready, release) -> None:
    with WorkspaceInstanceLock(Path(path)):
        ready.set()
        release.wait(10)


def test_workspace_instance_lock_excludes_other_process_and_recovers(tmp_path: Path) -> None:
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    release = ctx.Event()
    lock_path = tmp_path / "runtime" / "server.lock"
    process = ctx.Process(target=_hold_lock, args=(str(lock_path), ready, release))
    process.start()
    try:
        assert ready.wait(10)
        with pytest.raises(WorkspaceInstanceError):
            WorkspaceInstanceLock(lock_path).acquire()
        metadata = WorkspaceInstanceLock.read_metadata(lock_path)
        assert metadata["pid"] == process.pid
        assert metadata["hostname"]
    finally:
        release.set()
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(5)
    assert process.exitcode == 0
    with WorkspaceInstanceLock(lock_path) as acquired:
        assert acquired.metadata.pid == os.getpid()


def test_workspace_instance_lock_release_is_idempotent(tmp_path: Path) -> None:
    lock = WorkspaceInstanceLock(tmp_path / "server.lock").acquire()
    lock.release()
    lock.release()
    WorkspaceInstanceLock(tmp_path / "server.lock").acquire().release()


def test_workspace_instance_lock_reads_legacy_metadata(tmp_path: Path) -> None:
    lock_path = tmp_path / "server.lock"
    lock_path.write_text('{"hostname":"legacy-host","pid":42,"started_unix":1.0}\n', encoding="utf-8")

    assert WorkspaceInstanceLock.read_metadata(lock_path)["pid"] == 42


def test_workspace_instance_lock_releases_when_metadata_sync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / "server.lock"

    def fail_sync(_fd: int) -> None:
        raise OSError("simulated metadata sync failure")

    monkeypatch.setattr(os, "fsync", fail_sync)
    with pytest.raises(OSError, match="simulated metadata sync failure"):
        WorkspaceInstanceLock(lock_path).acquire()

    monkeypatch.undo()
    WorkspaceInstanceLock(lock_path).acquire().release()
