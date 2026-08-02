from __future__ import annotations

"""Cross-platform process lock for one mutable ScoreScan workspace.

A persistent lock file is intentionally retained after shutdown.  The operating-system
lock, not file existence, is the source of truth, so a crash releases ownership without
requiring stale-file deletion.  Metadata is rewritten only after the exclusive lock has
been acquired and is diagnostic rather than authoritative.
"""

import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

_LOCK_BYTE_OFFSET = 0
_METADATA_OFFSET = 1
_MAX_METADATA_BYTES = 4096


class WorkspaceInstanceError(RuntimeError):
    """Raised when another process already owns the workspace runtime lock."""


@dataclass(frozen=True)
class WorkspaceLockMetadata:
    pid: int
    started_unix: float
    hostname: str

    def to_dict(self) -> dict[str, object]:
        return {
            "pid": self.pid,
            "started_unix": self.started_unix,
            "hostname": self.hostname,
        }


class WorkspaceInstanceLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: IO[bytes] | None = None
        self.metadata = WorkspaceLockMetadata(os.getpid(), time.time(), socket.gethostname())

    @staticmethod
    def _try_lock(handle: IO[bytes]) -> bool:
        if os.name == "nt":
            import msvcrt

            try:
                handle.seek(_LOCK_BYTE_OFFSET)
                if handle.read(1) == b"":
                    handle.seek(_LOCK_BYTE_OFFSET)
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(_LOCK_BYTE_OFFSET)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                return True
            except OSError:
                return False
        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    @staticmethod
    def _unlock(handle: IO[bytes]) -> None:
        if os.name == "nt":
            import msvcrt

            handle.seek(_LOCK_BYTE_OFFSET)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            return
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def read_metadata(path: Path) -> dict[str, object]:
        def read_at(offset: int) -> dict[str, object]:
            try:
                with path.open("rb") as handle:
                    handle.seek(offset)
                    raw = handle.read(_MAX_METADATA_BYTES + 1)
                if len(raw) > _MAX_METADATA_BYTES:
                    return {}
                text = raw.lstrip(b"\0").decode("utf-8", errors="strict").strip()
                payload = json.loads(text) if text else {}
            except (OSError, json.JSONDecodeError, UnicodeError):
                return {}
            return payload if isinstance(payload, dict) else {}

        # Byte zero belongs only to the operating-system lock. This keeps the
        # diagnostics readable while a Windows process owns the lock region.
        payload = read_at(_METADATA_OFFSET)
        if payload:
            return payload
        # Accept the legacy layout after a clean shutdown.
        return read_at(0)

    def acquire(self) -> "WorkspaceInstanceLock":
        if self._handle is not None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b", buffering=0)
        if not self._try_lock(handle):
            handle.close()
            existing = self.read_metadata(self.path)
            pid = existing.get("pid")
            detail = f"（PID {pid}）" if isinstance(pid, int) and pid > 0 else ""
            raise WorkspaceInstanceError(f"Another Pigeon Score Scan instance is using this workspace{detail}")
        payload = (json.dumps(self.metadata.to_dict(), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        try:
            handle.seek(0)
            handle.truncate(0)
            handle.write(b"\0" + payload)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            # Never leave a process-owned lock behind when diagnostic metadata cannot
            # be committed.  The caller may retry or fail startup, and another process
            # must be able to acquire the workspace immediately.
            try:
                self._unlock(handle)
            finally:
                handle.close()
            raise
        self._handle = handle
        return self

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            self._unlock(handle)
        finally:
            handle.close()

    def __enter__(self) -> "WorkspaceInstanceLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()
