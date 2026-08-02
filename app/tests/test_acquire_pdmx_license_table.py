from __future__ import annotations

import hashlib
import io
import subprocess
import sys
from pathlib import Path

import pytest

from app.tools.acquire_pdmx_license_table import MAXIMUM_BYTES, URL
from app.tools.acquire_pdmx_metadata_archive import download_archive


class Response(io.BytesIO):
    def __init__(
        self,
        payload: bytes,
        *,
        status: int,
        response_url: str = URL,
    ) -> None:
        super().__init__(payload)
        self.status = status
        self.response_url = response_url

    def geturl(self) -> str:
        return self.response_url

    def __enter__(self) -> Response:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def test_direct_script_entry_point_loads_project_modules() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "acquire_pdmx_license_table.py"
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


def test_license_table_download_uses_its_exact_pinned_url(
    tmp_path: Path,
) -> None:
    payload = b"pinned-pdmx-license-table"
    destination = tmp_path / "PDMX.csv"

    def opener(
        _url: str,
        *,
        timeout_seconds: float,
        offset: int,
    ) -> Response:
        assert timeout_seconds == 30
        assert offset == 0
        return Response(payload, status=200)

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
        approved_url=URL,
        maximum_bytes=MAXIMUM_BYTES,
        asset_name="PDMX license table",
    )
    assert destination.read_bytes() == payload
    assert result["downloaded_bytes"] == len(payload)


def test_license_table_download_rejects_redirect_to_other_asset(
    tmp_path: Path,
) -> None:
    payload = b"fixture"

    def opener(
        _url: str,
        *,
        timeout_seconds: float,
        offset: int,
    ) -> Response:
        return Response(
            payload,
            status=200,
            response_url=(
                "https://zenodo.org/api/records/15571083/files/"
                "metadata.tar.gz/content"
            ),
        )

    with pytest.raises(ValueError, match="redirect left pinned Zenodo"):
        download_archive(
            URL,
            tmp_path / "PDMX.csv",
            expected_bytes=len(payload),
            expected_md5=hashlib.md5(
                payload,
                usedforsecurity=False,
            ).hexdigest(),
            timeout_seconds=30,
            opener=opener,
            approved_url=URL,
            maximum_bytes=MAXIMUM_BYTES,
            asset_name="PDMX license table",
        )
