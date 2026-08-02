from __future__ import annotations

import hashlib
import io
import subprocess
import sys
from pathlib import Path

import pytest

from app.tools.acquire_pdmx_metadata_archive import (
    URL,
    download_archive,
)


class Response(io.BytesIO):
    def __init__(self, payload: bytes, *, status: int) -> None:
        super().__init__(payload)
        self.status = status

    def geturl(self) -> str:
        return URL

    def __enter__(self) -> Response:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def test_direct_script_entry_point_loads_project_modules() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "acquire_pdmx_metadata_archive.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0
    assert "output_dir" in completed.stdout


def test_download_archive_resumes_and_verifies_identity(
    tmp_path: Path,
) -> None:
    payload = b"pinned-pdmx-metadata-fixture"
    destination = tmp_path / "metadata.tar.gz"
    partial = destination.with_suffix(".gz.part")
    partial.write_bytes(payload[:8])
    calls: list[int] = []

    def opener(
        _url: str,
        *,
        timeout_seconds: float,
        offset: int,
    ) -> Response:
        assert timeout_seconds == 30
        calls.append(offset)
        return Response(payload[offset:], status=206)

    result = download_archive(
        URL,
        destination,
        expected_bytes=len(payload),
        expected_md5=hashlib.md5(
            payload,
            usedforsecurity=False,
        ).hexdigest(),
        timeout_seconds=30,
        opener=opener,
    )
    assert calls == [8]
    assert destination.read_bytes() == payload
    assert result["resumed_from_bytes"] == 8
    assert result["downloaded_bytes"] == len(payload) - 8


def test_download_archive_rejects_resume_response_without_range(
    tmp_path: Path,
) -> None:
    payload = b"fixture"
    destination = tmp_path / "metadata.tar.gz"
    destination.with_suffix(".gz.part").write_bytes(payload[:2])

    def opener(
        _url: str,
        *,
        timeout_seconds: float,
        offset: int,
    ) -> Response:
        return Response(payload, status=200)

    with pytest.raises(ValueError, match="resume range"):
        download_archive(
            URL,
            destination,
            expected_bytes=len(payload),
            expected_md5=hashlib.md5(
                payload,
                usedforsecurity=False,
            ).hexdigest(),
            timeout_seconds=30,
            opener=opener,
        )
