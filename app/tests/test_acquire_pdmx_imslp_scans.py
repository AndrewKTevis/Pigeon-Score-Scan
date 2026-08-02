from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import pytest

from app.tools import acquire_pdmx_imslp_scans as module


class Headers(dict[str, str]):
    pass


class Response(io.BytesIO):
    def __init__(self, payload: bytes, *, url: str) -> None:
        super().__init__(payload)
        self.url = url
        self.headers = Headers(
            {
                "Content-Type": "application/pdf",
                "Content-Length": str(len(payload)),
            }
        )

    def geturl(self) -> str:
        return self.url

    def __enter__(self) -> Response:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def test_direct_script_entry_point_loads_project_modules() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "acquire_pdmx_imslp_scans.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0
    assert "evidence_path" in completed.stdout


def test_mirror_url_is_exact_and_size_label_is_rounded() -> None:
    source = {
        "imslp_source_id": "103834",
        "direct_pdf_url": (
            "https://imslp.org/images/e/ed/"
            "PMLP14377-Beethoven-WoO.059nohl1867.pdf"
        ),
    }
    assert module.mirror_url(source) == (
        "http://conquest.imslp.info/files/imglnks/usimg/e/ed/"
        "IMSLP103834-PMLP14377-Beethoven-WoO.059nohl1867.pdf"
    )
    assert module.size_matches_label(309_603, "302KB")
    assert module.size_matches_label(5_376_416, "5.13MB")
    assert not module.size_matches_label(3_000_000, "5.13MB")


def test_download_rejects_redirect_and_non_pdf(tmp_path: Path) -> None:
    url = (
        "http://conquest.imslp.info/files/imglnks/usimg/e/ed/"
        "IMSLP103834-PMLP14377-Beethoven-WoO.059nohl1867.pdf"
    )

    def redirected(
        _url: str,
        *,
        timeout_seconds: float,
    ) -> Response:
        return Response(b"%PDF-fixture", url=url + "?other=1")

    with pytest.raises(ValueError, match="redirected unexpectedly"):
        module.download_pdf(
            url,
            tmp_path / "scan.pdf",
            timeout_seconds=10,
            opener=redirected,
        )

    def html(_url: str, *, timeout_seconds: float) -> Response:
        response = Response(b"<html>not pdf</html>", url=url)
        response.headers["Content-Type"] = "text/html"
        return response

    with pytest.raises(ValueError, match="did not return a PDF"):
        module.download_pdf(
            url,
            tmp_path / "scan.pdf",
            timeout_seconds=10,
            opener=html,
        )
