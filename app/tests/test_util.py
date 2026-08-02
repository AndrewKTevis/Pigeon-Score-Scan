import errno
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import scorescan.util as util
from scorescan.util import atomic_write_bytes, natural_key, safe_filename


def test_natural_sort() -> None:
    values = ["10.png", "2.png", "1.png"]
    assert sorted(values, key=natural_key) == ["1.png", "2.png", "10.png"]


def test_safe_filename() -> None:
    assert safe_filename('A: B / C?') == "A_ B _ C_"


def test_atomic_write_retries_transient_replace_denials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "job.json"
    target.write_bytes(b"old")
    real_replace = util.os.replace
    attempts = 0
    delays: list[float] = []

    def flaky_replace(source: str, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts <= 3:
            raise PermissionError(errno.EACCES, "sharing violation")
        real_replace(source, destination)

    monkeypatch.setattr(util.os, "replace", flaky_replace)
    monkeypatch.setattr(util.time, "sleep", delays.append)

    atomic_write_bytes(target, b"new")

    assert target.read_bytes() == b"new"
    assert attempts == 4
    assert delays == list(util._REPLACE_RETRY_DELAYS_SECONDS[:3])
    assert list(tmp_path.glob("job.json.*.tmp")) == []


def test_atomic_write_preserves_target_and_cleans_temp_after_permanent_denial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "job.json"
    target.write_bytes(b"old")
    attempts = 0

    def denied_replace(_source: str, _destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        raise PermissionError(errno.EACCES, "sharing violation")

    monkeypatch.setattr(util.os, "replace", denied_replace)
    monkeypatch.setattr(util.time, "sleep", lambda _delay: None)

    with pytest.raises(PermissionError):
        atomic_write_bytes(target, b"new")

    assert attempts == len(util._REPLACE_RETRY_DELAYS_SECONDS) + 1
    assert target.read_bytes() == b"old"
    assert list(tmp_path.glob("job.json.*.tmp")) == []


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing semantics")
def test_atomic_write_survives_a_short_lived_windows_reader(tmp_path: Path) -> None:
    target = tmp_path / "job.json"
    target.write_bytes(b"old")

    with ThreadPoolExecutor(max_workers=1) as pool:
        locked_reader = target.open("rb")
        try:
            future = pool.submit(atomic_write_bytes, target, b"new")
            time.sleep(0.1)
        finally:
            locked_reader.close()
        future.result(timeout=3)

    assert target.read_bytes() == b"new"
    assert list(tmp_path.glob("job.json.*.tmp")) == []
