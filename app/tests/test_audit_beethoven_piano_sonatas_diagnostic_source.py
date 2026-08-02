from __future__ import annotations

import base64
from pathlib import Path

import fitz
import pytest

from app.tools.audit_beethoven_piano_sonatas_diagnostic_source import (
    audit,
)


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4"
    "nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)
KRN = """!!!!SEGMENT: {name}.krn
!!!OTL: Fixture sonata
!!!OMV: {segment}
!!!OMD: Allegro
**kern\t**kern\t**dynam
*staff2\t*staff1\t*staff1/2
*Ipiano\t*Ipiano\t*Ipiano
*M4/4\t*M4/4\t*
=1\t=1\t=1
4C\t4c\tp
*-\t*-\t*-
"""


def _source(root: Path) -> Path:
    source = root / "source"
    kern = source / "kern"
    scans = source / "reference-edition"
    kern.mkdir(parents=True)
    scans.mkdir()
    (source / "README.md").write_text(
        "Scans of the source edition can be downloaded with make reference.",
        encoding="utf-8",
    )
    (source / "Makefile").write_text(
        "reference:\n\t@echo reference\n",
        encoding="utf-8",
    )
    for segment in (1, 2):
        name = f"sonata01-{segment}"
        (kern / f"{name}.krn").write_text(
            KRN.format(name=name, segment=segment),
            encoding="utf-8",
        )
    first = scans / "sonata01-1.pdf"
    with fitz.open() as document:
        page = document.new_page(width=100, height=100)
        page.insert_image(page.rect, stream=PNG)
        document.save(first)
    (scans / "sonata01-2.pdf").write_bytes(first.read_bytes())
    return source


def test_audit_counts_unique_scan_pages_and_never_authorizes(
    tmp_path: Path,
) -> None:
    report = audit(_source(tmp_path), revision="b" * 40)

    assert report["sonata_work_count"] == 1
    assert report["humdrum_movement_count"] == 2
    assert report["boundary_accepted_humdrum_count"] == 2
    assert report["pdf_filename_count"] == 2
    assert report["unique_pdf_count"] == 1
    assert report["same_named_pair_count"] == 2
    assert report["unique_same_named_pair_count"] == 0
    assert report["duplicate_pdf_group_count"] == 1
    assert report["unique_physical_scan_page_count"] == 1
    assert report["all_unique_pdfs_image_only"] is True
    assert report["source_license_file_present"] is False
    assert report["movement_page_ranges_independently_verified"] is False
    assert report["training_authorized"] is False
    assert report["evaluation_authorized"] is False
    assert report["release_authorized"] is False


def test_audit_requires_pinned_revision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="full lowercase Git commit"):
        audit(_source(tmp_path), revision="main")
