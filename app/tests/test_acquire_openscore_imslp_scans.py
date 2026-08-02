from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import fitz
import pytest

from app.tools.acquire_openscore_imslp_scans import (
    acquire,
    download_pdf,
)
from app.tools.catalog_openscore_imslp_scan_candidates import LICENSE_SHA256
from app.tools.probe_openscore_imslp_scan_sources import ROLE as EVIDENCE_ROLE


class Response(io.BytesIO):
    def __init__(
        self,
        payload: bytes,
        *,
        url: str = "https://imslp.org/images/a/ab/source.pdf",
    ) -> None:
        super().__init__(payload)
        self._url = url
        self.headers = {
            "Content-Type": "application/pdf",
            "Content-Length": str(len(payload)),
        }

    def geturl(self) -> str:
        return self._url

    def __enter__(self) -> Response:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _pdf_bytes(page_count: int = 2) -> bytes:
    document = fitz.open()
    for _ in range(page_count):
        document.new_page(width=595, height=842)
    payload = document.tobytes()
    document.close()
    return payload


def test_direct_script_entry_point_loads_project_modules() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "acquire_openscore_imslp_scans.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0
    assert "semantic_split_report" in completed.stdout


def test_download_pdf_rejects_redirect_to_unapproved_host(
    tmp_path: Path,
) -> None:
    payload = _pdf_bytes()

    def opener(_url: str, **_kwargs: object) -> Response:
        return Response(payload, url="https://example.com/source.pdf")

    with pytest.raises(ValueError, match="redirect"):
        download_pdf(
            "https://imslp.org/images/a/ab/source.pdf",
            tmp_path / "source.pdf",
            timeout_seconds=5,
            opener=opener,
        )
    assert not (tmp_path / "source.pdf").exists()
    assert not (tmp_path / "source.pdf.part").exists()


def test_download_pdf_accepts_archive_mirror_and_records_all_hashes(
    tmp_path: Path,
) -> None:
    payload = _pdf_bytes()

    def opener(_url: str, **_kwargs: object) -> Response:
        return Response(
            payload,
            url=(
                "https://ia902800.us.archive.org/1/items/example/"
                "source.pdf"
            ),
        )

    result = download_pdf(
        "https://archive.org/download/example/source.pdf",
        tmp_path / "source.pdf",
        timeout_seconds=5,
        opener=opener,
    )
    assert result["pdf_md5"] == hashlib.md5(
        payload,
        usedforsecurity=False,
    ).hexdigest()
    assert result["pdf_sha1"] == hashlib.sha1(
        payload,
        usedforsecurity=False,
    ).hexdigest()


def test_acquire_keeps_bytes_unauthorized_until_alignment(
    tmp_path: Path,
) -> None:
    score_root = tmp_path / "scores"
    score_path = score_root / "scores" / "Composer" / "Work" / "score.mscx"
    score_path.parent.mkdir(parents=True)
    score_path.write_bytes(b"<museScore/>")
    license_path = score_root / "LICENSE.txt"
    license_path.write_bytes(b"")
    assert hashlib.sha256(b"").hexdigest() != LICENSE_SHA256
    # The production constant is immutable; construct a matching fixture by
    # monkeypatching the file hash through the real pinned license bytes.
    pinned_license = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "licenses"
        / "openscore-string-quartets-LICENSE.txt"
    )
    license_path.write_bytes(pinned_license.read_bytes())
    score_hash = hashlib.sha256(score_path.read_bytes()).hexdigest()
    candidate = {
        "source_identity_verified": True,
        "imslp_source_id": "17471",
        "imslp_page_title": "String Quartet No.1 (Composer, Example)",
        "direct_pdf_url": "https://imslp.org/images/a/ab/source.pdf",
        "path": "scores/Composer/Work/score.mscx",
        "sha256": score_hash,
        "work_fingerprint": "a" * 64,
    }
    source = {
        "imslp_source_id": "17471",
        "verified_public_domain_printed_scan": True,
        "direct_pdf_url": candidate["direct_pdf_url"],
        "page_title": candidate["imslp_page_title"],
        "page_count": 2,
        "page_sha256": "b" * 64,
        "public_domain_evidence": "Copyright Public Domain",
        "scan_attribution_text": "PDF scanned by Example",
    }
    evidence = {
        "role": EVIDENCE_ROLE,
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_authorized": False,
        "verified_candidates": [candidate],
        "sources": [source],
    }
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    semantic_split_report = tmp_path / "semantic-splits.json"
    semantic_split_report.write_text(
        json.dumps(
            {
                "purpose": (
                    "synthetic semantic geometry; not real-scan validation"
                ),
                "sources": [
                    {
                        "source_key": candidate["path"],
                        "source_sha256": score_hash,
                        "split": "train",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    payload = _pdf_bytes()

    def opener(_url: str, **_kwargs: object) -> Response:
        return Response(payload)

    report = acquire(
        evidence_path,
        score_root,
        semantic_split_report,
        tmp_path / "pdf",
        tmp_path / "manifest.json",
        timeout_seconds=5,
        minimum_interval_seconds=0,
        limit=None,
        opener=opener,
    )
    assert report["source_count"] == 1
    assert report["page_count"] == 2
    assert report["training_authorized"] is False
    assert report["evaluation_authorized"] is False
    assert report["release_authorized"] is False
    assert report["candidate_count_by_semantic_split"] == {"train": 1}
    assert report["assets"][0]["pdf_sha256"] == hashlib.sha256(payload).hexdigest()
    assert report["archive_mirror_required"] is False
    assert report["archive_mirror_filter_omitted_source_count"] == 0

    with pytest.raises(
        ValueError,
        match="needs archive mirror evidence",
    ):
        acquire(
            evidence_path,
            score_root,
            semantic_split_report,
            tmp_path / "pdf-required",
            tmp_path / "manifest-required.json",
            timeout_seconds=5,
            minimum_interval_seconds=0,
            limit=None,
            require_archive_mirror=True,
            opener=opener,
        )
