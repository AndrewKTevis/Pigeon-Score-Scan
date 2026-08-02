from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_NATURAL_RE = re.compile(r"(\d+)")
_INVALID_WINDOWS_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_REPLACE_RETRY_DELAYS_SECONDS = (0.025, 0.05, 0.1, 0.2, 0.4, 0.4, 0.4, 0.4)
_TRANSIENT_REPLACE_ERRNOS = {errno.EACCES, errno.EBUSY}
_TRANSIENT_REPLACE_WINERRORS = {5, 32, 33}


def natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.casefold() for part in _NATURAL_RE.split(value)]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _is_transient_replace_error(exc: OSError) -> bool:
    return (
        isinstance(exc, PermissionError)
        or exc.errno in _TRANSIENT_REPLACE_ERRNOS
        or getattr(exc, "winerror", None) in _TRANSIENT_REPLACE_WINERRORS
    )


def replace_file_with_retry(source: str | Path, destination: str | Path) -> None:
    for delay in _REPLACE_RETRY_DELAYS_SECONDS:
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            if not _is_transient_replace_error(exc):
                raise
        time.sleep(delay)
    os.replace(source, destination)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        replace_file_with_retry(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def atomic_write_text(path: Path, payload: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, payload.encode(encoding))


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_filename(value: str, fallback: str = "converted_score") -> str:
    cleaned = _INVALID_WINDOWS_FILENAME.sub("_", value).strip(" .")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:160] or fallback


def path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False
