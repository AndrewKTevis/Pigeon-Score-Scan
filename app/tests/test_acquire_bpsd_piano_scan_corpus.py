from __future__ import annotations

import io
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from app.tools import acquire_bpsd_piano_scan_corpus as module


def _archive() -> bytes:
    destination = io.BytesIO()
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr("safe/example.xml", b"<score-partwise/>")
        archive.writestr("safe/example.pdf", b"%PDF-fixture")
    return destination.getvalue()


def _catalog(payload: bytes) -> list[module.ZipEntry]:
    tail_start = max(0, len(payload) - module.EOCD_SEARCH_BYTES)
    offset, size, count = module.parse_eocd(
        payload[tail_start:],
        tail_start=tail_start,
        archive_size=len(payload),
    )
    return module.parse_central_directory(
        payload[offset : offset + size],
        expected_entries=count,
    )


def test_direct_script_entry_point_loads_project_modules() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "acquire_bpsd_piano_scan_corpus.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0
    assert "--manifest" in completed.stdout


def test_catalog_and_member_crc_round_trip() -> None:
    payload = _archive()
    entries = _catalog(payload)
    assert [entry.name for entry in entries] == [
        "safe/example.xml",
        "safe/example.pdf",
    ]
    for entry in entries:
        decoded = module.decode_entry_from_span(
            payload,
            span_start=0,
            entry=entry,
        )
        assert decoded in {
            b"<score-partwise/>",
            b"%PDF-fixture",
        }


def test_member_crc_and_local_header_mismatch_fail_closed() -> None:
    payload = _archive()
    entry = _catalog(payload)[0]
    damaged_payload = bytearray(payload)
    name_size, extra_size = struct.unpack_from(
        "<2H",
        damaged_payload,
        entry.local_header_offset + 26,
    )
    data_start = entry.local_header_offset + 30 + name_size + extra_size
    damaged_payload[data_start + entry.compressed_size // 2] ^= 1
    with pytest.raises(ValueError, match="deflate|CRC or size"):
        module.decode_entry_from_span(
            bytes(damaged_payload),
            span_start=0,
            entry=entry,
        )
    damaged = bytearray(payload)
    struct.pack_into(
        "<L",
        damaged,
        entry.local_header_offset + 14,
        entry.crc32 ^ 1,
    )
    with pytest.raises(ValueError, match="local member header"):
        module.decode_entry_from_span(
            bytes(damaged),
            span_start=0,
            entry=entry,
        )


def test_eocd_rejects_trailing_or_multidisk_archives() -> None:
    payload = _archive()
    with pytest.raises(ValueError, match="central-directory contract"):
        module.parse_eocd(
            payload + b"x",
            tail_start=0,
            archive_size=len(payload) + 1,
        )
    position = payload.rfind(b"PK\x05\x06")
    damaged = bytearray(payload)
    struct.pack_into("<H", damaged, position + 4, 1)
    with pytest.raises(ValueError, match="central-directory contract"):
        module.parse_eocd(
            bytes(damaged),
            tail_start=0,
            archive_size=len(damaged),
        )


def test_central_directory_rejects_unsafe_member_name() -> None:
    destination = io.BytesIO()
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr("../escape.xml", b"<score-partwise/>")
    payload = destination.getvalue()
    position = payload.rfind(b"PK\x05\x06")
    (
        _signature,
        _disk,
        _central_disk,
        _disk_entries,
        entries,
        size,
        offset,
        _comment,
    ) = struct.unpack_from("<4s4H2LH", payload, position)
    with pytest.raises(ValueError, match="unsafe or duplicated"):
        module.parse_central_directory(
            payload[offset : offset + size],
            expected_entries=entries,
        )


def test_corpus_selection_requires_exact_work_pairing() -> None:
    def entry(name: str) -> module.ZipEntry:
        return module.ZipEntry(name, 0, 8, 1, 1, 1, 1)

    scans = [
        entry(
            module.SCAN_PREFIX
            + f"Beethoven_Op{index:03d}-01.pdf"
        )
        for index in range(module.EXPECTED_WORKS)
    ]
    references = [
        entry(
            module.XML_PREFIX
            + f"Beethoven_Op{index:03d}-01.xml"
        )
        for index in range(module.EXPECTED_WORKS)
    ]
    selected_scans, selected_references = (
        module.select_corpus_entries(scans + references)
    )
    assert len(selected_scans) == module.EXPECTED_WORKS
    assert len(selected_references) == module.EXPECTED_WORKS
    with pytest.raises(ValueError, match="pairing is incomplete"):
        module.select_corpus_entries(scans + references[:-1])
