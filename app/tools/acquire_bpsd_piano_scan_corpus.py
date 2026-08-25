from __future__ import annotations

"""Acquire the bounded BPSD physical-scan/MusicXML piano corpus.

The published archive is mostly audio.  This tool pins the official Zenodo
record and uses verified HTTPS byte ranges to extract only the 32 first-
movement scan PDFs and their 32 same-named MusicXML references.  Acquisition
does not imply page alignment or training/release authorization.
"""

import argparse
import binascii
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import pymupdf as fitz
from lxml import etree


APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_ROOT / "src"))

from app.tools.prepare_muse_omr_benchmark import (  # noqa: E402
    analyze_reference_boundary,
)
from scorescan.product_scope import (  # noqa: E402
    PRODUCTION_BOUNDARY_CONTRACT_VERSION,
)
from scorescan.util import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_json,
    sha256_file,
    utc_now_iso,
)


RECORD_ID = 18_662_551
RECORD_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
DOI = "10.5281/zenodo.18662551"
LICENSE_ID = "cc-by-3.0"
ARCHIVE_NAME = "Beethoven_Piano_Sonata_Dataset_v2.zip"
ARCHIVE_BYTES = 2_121_553_242
ARCHIVE_MD5 = "17f73a2d608fd6e6da65d3bd9dda1923"
ARCHIVE_URL = (
    f"{RECORD_URL}/files/{ARCHIVE_NAME}/content"
)
ARCHIVE_ROOT = "Beethoven_Piano_Sonata_Dataset_v2/0_RawData"
SCAN_PREFIX = f"{ARCHIVE_ROOT}/score_pdf_scan/"
XML_PREFIX = f"{ARCHIVE_ROOT}/score_xml_repetitions/"
ROLE = (
    "bpsd_ccby3_physical_scan_musicxml_work_pairs_not_page_aligned"
)
EOCD_SEARCH_BYTES = 128 * 1024
MAXIMUM_CENTRAL_DIRECTORY_BYTES = 8 * 1024 * 1024
MAXIMUM_RANGE_BYTES = 128 * 1024 * 1024
MAXIMUM_MEMBER_BYTES = 16 * 1024 * 1024
EXPECTED_WORKS = 32
SAFE_MEMBER_NAME = re.compile(
    r"Beethoven_(?P<work>Op[0-9A-Za-z]+)-01\.(?P<suffix>pdf|xml)"
)


@dataclass(frozen=True)
class ZipEntry:
    name: str
    flags: int
    method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_member_name(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(
        not path.is_absolute()
        and path.parts
        and all(part not in {"", ".", ".."} for part in path.parts)
        and ":" not in path.parts[0]
        and "\\" not in value
    )


def parse_eocd(
    tail: bytes,
    *,
    tail_start: int,
    archive_size: int,
) -> tuple[int, int, int]:
    """Return central-directory offset, byte count and member count."""

    position = tail.rfind(b"PK\x05\x06")
    if position < 0 or len(tail) - position < 22:
        raise ValueError("ZIP end-of-central-directory record is missing")
    (
        signature,
        disk,
        central_disk,
        disk_entries,
        total_entries,
        central_size,
        central_offset,
        comment_size,
    ) = struct.unpack_from("<4s4H2LH", tail, position)
    if (
        signature != b"PK\x05\x06"
        or disk != 0
        or central_disk != 0
        or disk_entries != total_entries
        or total_entries <= 0
        or total_entries == 0xFFFF
        or central_size <= 0
        or central_size == 0xFFFFFFFF
        or central_offset == 0xFFFFFFFF
        or comment_size != len(tail) - position - 22
        or tail_start + position + 22 + comment_size != archive_size
        or central_size > MAXIMUM_CENTRAL_DIRECTORY_BYTES
        or central_offset + central_size != tail_start + position
    ):
        raise ValueError("ZIP central-directory contract is unsupported")
    return central_offset, central_size, total_entries


def parse_central_directory(
    payload: bytes,
    *,
    expected_entries: int,
) -> list[ZipEntry]:
    entries: list[ZipEntry] = []
    position = 0
    seen: set[str] = set()
    while position < len(payload):
        if len(payload) - position < 46:
            raise ValueError("ZIP central-directory entry is truncated")
        (
            signature,
            _made_by,
            _required,
            flags,
            method,
            _modified_time,
            _modified_date,
            crc32,
            compressed_size,
            uncompressed_size,
            name_size,
            extra_size,
            comment_size,
            disk,
            _internal_attributes,
            _external_attributes,
            local_header_offset,
        ) = struct.unpack_from("<4s6H3L5H2L", payload, position)
        end = (
            position
            + 46
            + name_size
            + extra_size
            + comment_size
        )
        if (
            signature != b"PK\x01\x02"
            or disk != 0
            or name_size <= 0
            or end > len(payload)
            or compressed_size == 0xFFFFFFFF
            or uncompressed_size == 0xFFFFFFFF
            or local_header_offset == 0xFFFFFFFF
        ):
            raise ValueError("ZIP central-directory entry is unsupported")
        name_raw = payload[position + 46 : position + 46 + name_size]
        encoding = "utf-8" if flags & 0x800 else "cp437"
        try:
            name = name_raw.decode(encoding)
        except UnicodeDecodeError as exc:
            raise ValueError("ZIP member name cannot be decoded") from exc
        if not _safe_member_name(name) or name in seen:
            raise ValueError("ZIP member name is unsafe or duplicated")
        seen.add(name)
        entries.append(
            ZipEntry(
                name=name,
                flags=flags,
                method=method,
                crc32=crc32,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                local_header_offset=local_header_offset,
            )
        )
        position = end
    if position != len(payload) or len(entries) != expected_entries:
        raise ValueError("ZIP central-directory member count is inconsistent")
    return entries


def select_corpus_entries(
    entries: list[ZipEntry],
) -> tuple[list[ZipEntry], list[ZipEntry]]:
    scans: dict[str, ZipEntry] = {}
    references: dict[str, ZipEntry] = {}
    for entry in entries:
        if entry.name.startswith(SCAN_PREFIX):
            prefix = SCAN_PREFIX
            target = scans
            expected_suffix = "pdf"
        elif entry.name.startswith(XML_PREFIX):
            prefix = XML_PREFIX
            target = references
            expected_suffix = "xml"
        else:
            continue
        relative = entry.name[len(prefix) :]
        if not relative:
            continue
        match = SAFE_MEMBER_NAME.fullmatch(relative)
        if (
            match is None
            or match.group("suffix") != expected_suffix
            or "/" in relative
            or entry.flags & 0x1
            or entry.flags & 0x8
            or entry.flags & ~0x800
            or entry.method != 8
            or entry.compressed_size <= 0
            or entry.uncompressed_size <= 0
            or entry.uncompressed_size > MAXIMUM_MEMBER_BYTES
        ):
            raise ValueError("BPSD corpus member violates the pinned contract")
        work = match.group("work")
        if work in target:
            raise ValueError("BPSD corpus work is duplicated")
        target[work] = entry
    if (
        len(scans) != EXPECTED_WORKS
        or len(references) != EXPECTED_WORKS
        or set(scans) != set(references)
    ):
        raise ValueError("BPSD scan/MusicXML work pairing is incomplete")
    works = sorted(scans)
    return [scans[work] for work in works], [
        references[work] for work in works
    ]


def decode_entry_from_span(
    span: bytes,
    *,
    span_start: int,
    entry: ZipEntry,
) -> bytes:
    position = entry.local_header_offset - span_start
    if position < 0 or len(span) - position < 30:
        raise ValueError("ZIP member is outside its fetched byte span")
    (
        signature,
        _required,
        flags,
        method,
        _modified_time,
        _modified_date,
        local_crc32,
        local_compressed_size,
        local_uncompressed_size,
        name_size,
        extra_size,
    ) = struct.unpack_from("<4s5H3L2H", span, position)
    data_start = position + 30 + name_size + extra_size
    data_end = data_start + entry.compressed_size
    if (
        signature != b"PK\x03\x04"
        or flags != entry.flags
        or method != entry.method
        or local_crc32 != entry.crc32
        or local_compressed_size != entry.compressed_size
        or local_uncompressed_size != entry.uncompressed_size
        or data_end > len(span)
    ):
        raise ValueError("ZIP local member header contradicts its catalog")
    name_raw = span[position + 30 : position + 30 + name_size]
    encoding = "utf-8" if flags & 0x800 else "cp437"
    if name_raw.decode(encoding) != entry.name:
        raise ValueError("ZIP local member name contradicts its catalog")
    try:
        result = zlib.decompress(span[data_start:data_end], -15)
    except zlib.error as exc:
        raise ValueError("ZIP deflate stream is invalid") from exc
    if (
        len(result) != entry.uncompressed_size
        or binascii.crc32(result) & 0xFFFFFFFF != entry.crc32
    ):
        raise ValueError("ZIP member CRC or size verification failed")
    return result


def _header_value(headers: Any, name: str) -> str:
    value = headers.get(name)
    return str(value or "").strip()


def fetch_range(
    url: str,
    start: int,
    end: int,
    *,
    timeout_seconds: float,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[bytes, dict[str, str]]:
    if (
        not url.startswith("https://")
        or start < 0
        or end < start
        or end - start + 1 > MAXIMUM_RANGE_BYTES
    ):
        raise ValueError("HTTPS byte range is invalid or oversized")
    request = urllib.request.Request(
        url,
        headers={
            "Accept-Encoding": "identity",
            "Range": f"bytes={start}-{end}",
            "User-Agent": "ScoreScan-BPSD-acquisition/1",
        },
    )
    with opener(request, timeout=timeout_seconds) as response:
        status = int(
            getattr(response, "status", 0)
            or getattr(response, "getcode", lambda: 0)()
        )
        final_url = str(
            getattr(response, "geturl", lambda: url)()
        )
        expected_length = end - start + 1
        headers = {
            "content_range": _header_value(
                response.headers,
                "Content-Range",
            ),
            "content_length": _header_value(
                response.headers,
                "Content-Length",
            ),
            "content_type": _header_value(
                response.headers,
                "Content-Type",
            ),
            "last_modified": _header_value(
                response.headers,
                "Last-Modified",
            ),
        }
        payload = response.read(expected_length + 1)
    if (
        status != 206
        or final_url != url
        or headers["content_range"]
        != f"bytes {start}-{end}/{ARCHIVE_BYTES}"
        or headers["content_length"] != str(expected_length)
        or len(payload) != expected_length
    ):
        raise ValueError("server did not honor the exact pinned byte range")
    return payload, headers


def fetch_record(
    *,
    timeout_seconds: float,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[dict[str, Any], bytes]:
    request = urllib.request.Request(
        RECORD_URL,
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "User-Agent": "ScoreScan-BPSD-acquisition/1",
        },
    )
    with opener(request, timeout=timeout_seconds) as response:
        status = int(
            getattr(response, "status", 0)
            or getattr(response, "getcode", lambda: 0)()
        )
        final_url = str(
            getattr(response, "geturl", lambda: RECORD_URL)()
        )
        payload = response.read(4 * 1024 * 1024 + 1)
    if (
        status != 200
        or final_url != RECORD_URL
        or len(payload) > 4 * 1024 * 1024
    ):
        raise ValueError("pinned Zenodo record response is invalid")
    try:
        record = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("pinned Zenodo record JSON is invalid") from exc
    files = record.get("files")
    archive = next(
        (
            item
            for item in files
            if isinstance(item, dict)
            and item.get("key") == ARCHIVE_NAME
        ),
        None,
    ) if isinstance(files, list) else None
    metadata = record.get("metadata")
    license_row = (
        metadata.get("license")
        if isinstance(metadata, dict)
        else None
    )
    archive_link = (
        archive.get("links", {}).get("self")
        if isinstance(archive, dict)
        and isinstance(archive.get("links"), dict)
        else None
    )
    if (
        record.get("id") != RECORD_ID
        or not isinstance(metadata, dict)
        or metadata.get("access_right") != "open"
        or not isinstance(license_row, dict)
        or license_row.get("id") != LICENSE_ID
        or not isinstance(archive, dict)
        or archive.get("size") != ARCHIVE_BYTES
        or archive.get("checksum") != f"md5:{ARCHIVE_MD5}"
        or archive_link != ARCHIVE_URL
    ):
        raise ValueError("pinned Zenodo BPSD metadata contract changed")
    return record, payload


def _entry_fetch_span(
    entries: list[ZipEntry],
    catalog: list[ZipEntry],
) -> tuple[int, int]:
    positions = {
        entry.local_header_offset: index
        for index, entry in enumerate(catalog)
    }
    first = min(entries, key=lambda item: item.local_header_offset)
    last = max(entries, key=lambda item: item.local_header_offset)
    last_index = positions[last.local_header_offset]
    if last_index + 1 >= len(catalog):
        raise ValueError("selected ZIP member has no bounded successor")
    start = first.local_header_offset
    end = catalog[last_index + 1].local_header_offset - 1
    if end < start or end - start + 1 > MAXIMUM_RANGE_BYTES:
        raise ValueError("selected ZIP member span is invalid or oversized")
    return start, end


def _inspect_pdf(payload: bytes) -> dict[str, Any]:
    if not payload.startswith(b"%PDF-"):
        raise ValueError("BPSD scan member is not a PDF")
    try:
        with fitz.open(stream=payload, filetype="pdf") as document:
            page_count = len(document)
            text_characters = sum(
                len(page.get_text().strip()) for page in document
            )
            pages_with_images = sum(
                bool(page.get_images(full=True)) for page in document
            )
            rotations = sorted(
                {int(page.rotation) for page in document}
            )
            page_sizes = sorted(
                {
                    (
                        round(float(page.rect.width), 3),
                        round(float(page.rect.height), 3),
                    )
                    for page in document
                }
            )
    except (RuntimeError, ValueError) as exc:
        raise ValueError("BPSD scan PDF parser rejected member") from exc
    if page_count <= 0:
        raise ValueError("BPSD scan PDF contains no pages")
    return {
        "pages": page_count,
        "text_characters": text_characters,
        "pages_with_images": pages_with_images,
        "page_rotations": rotations,
        "page_sizes_points": [list(value) for value in page_sizes],
        "image_only_scan_candidate": (
            text_characters == 0 and pages_with_images == page_count
        ),
    }


def _inspect_musicxml(payload: bytes) -> None:
    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        recover=False,
        huge_tree=False,
    )
    try:
        root = etree.fromstring(payload, parser)
    except etree.XMLSyntaxError as exc:
        raise ValueError("BPSD MusicXML parser rejected member") from exc
    if etree.QName(root).localname != "score-partwise":
        raise ValueError("BPSD reference is not score-partwise MusicXML")


def acquire(
    output_dir: Path,
    manifest_path: Path,
    *,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    manifest_path = manifest_path.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"BPSD output directory already exists: {output_dir}"
        )
    if timeout_seconds <= 0:
        raise ValueError("network timeout must be positive")

    record, record_bytes = fetch_record(
        timeout_seconds=timeout_seconds,
    )
    tail_start = ARCHIVE_BYTES - EOCD_SEARCH_BYTES
    tail, tail_headers = fetch_range(
        ARCHIVE_URL,
        tail_start,
        ARCHIVE_BYTES - 1,
        timeout_seconds=timeout_seconds,
    )
    central_offset, central_size, expected_entries = parse_eocd(
        tail,
        tail_start=tail_start,
        archive_size=ARCHIVE_BYTES,
    )
    central, central_headers = fetch_range(
        ARCHIVE_URL,
        central_offset,
        central_offset + central_size - 1,
        timeout_seconds=timeout_seconds,
    )
    catalog = parse_central_directory(
        central,
        expected_entries=expected_entries,
    )
    scan_entries, xml_entries = select_corpus_entries(catalog)
    scan_start, scan_end = _entry_fetch_span(scan_entries, catalog)
    xml_start, xml_end = _entry_fetch_span(xml_entries, catalog)
    scan_span, scan_headers = fetch_range(
        ARCHIVE_URL,
        scan_start,
        scan_end,
        timeout_seconds=timeout_seconds,
    )
    xml_span, xml_headers = fetch_range(
        ARCHIVE_URL,
        xml_start,
        xml_end,
        timeout_seconds=timeout_seconds,
    )

    staging_parent = output_dir.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=output_dir.name + ".staging-",
            dir=staging_parent,
        )
    )
    cases: list[dict[str, Any]] = []
    try:
        record_path = staging / "zenodo-record.json"
        central_path = staging / "zip-central-directory.bin"
        atomic_write_bytes(record_path, record_bytes)
        atomic_write_bytes(central_path, central)
        scan_by_work = {
            SAFE_MEMBER_NAME.fullmatch(Path(entry.name).name).group("work"): entry
            for entry in scan_entries
        }
        xml_by_work = {
            SAFE_MEMBER_NAME.fullmatch(Path(entry.name).name).group("work"): entry
            for entry in xml_entries
        }
        for work in sorted(scan_by_work):
            scan_entry = scan_by_work[work]
            xml_entry = xml_by_work[work]
            scan_payload = decode_entry_from_span(
                scan_span,
                span_start=scan_start,
                entry=scan_entry,
            )
            xml_payload = decode_entry_from_span(
                xml_span,
                span_start=xml_start,
                entry=xml_entry,
            )
            pdf_info = _inspect_pdf(scan_payload)
            _inspect_musicxml(xml_payload)
            work_dir = staging / "works" / work
            scan_path = work_dir / "scan.pdf"
            reference_path = work_dir / "reference.musicxml"
            atomic_write_bytes(scan_path, scan_payload)
            atomic_write_bytes(reference_path, xml_payload)
            boundary = analyze_reference_boundary(reference_path)
            boundary_eligible = bool(
                boundary.get("accepted") is True
                and boundary.get("score_shape") == "keyboard"
            )
            final_work_dir = output_dir / "works" / work
            cases.append(
                {
                    "case_id": f"bpsd-{work.casefold()}",
                    "work": work,
                    "scan_path": str(final_work_dir / "scan.pdf"),
                    "scan_sha256": sha256_bytes(scan_payload),
                    "scan_bytes": len(scan_payload),
                    **pdf_info,
                    "reference_musicxml_path": str(
                        final_work_dir / "reference.musicxml"
                    ),
                    "reference_musicxml_sha256": sha256_bytes(
                        xml_payload
                    ),
                    "reference_musicxml_bytes": len(xml_payload),
                    "boundary": boundary,
                    "boundary_eligible_alignment_candidate": (
                        boundary_eligible
                    ),
                    "reference_quarantined": not boundary_eligible,
                    "reference_quarantine_reasons": (
                        []
                        if boundary_eligible
                        else list(boundary.get("reasons") or ())
                    ),
                    "source_scan_member": scan_entry.name,
                    "source_musicxml_member": xml_entry.name,
                    "same_named_work_pair": True,
                    "exact_page_measure_alignment_verified": False,
                    "independent_double_annotation_complete": False,
                    "training_authorized": False,
                    "evaluation_authorized": False,
                    "release_authorized": False,
                }
            )
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    report = {
        "format": 1,
        "created_at": utc_now_iso(),
        "name": "BPSD selectively acquired piano scan corpus",
        "role": ROLE,
        "record_id": RECORD_ID,
        "record_url": RECORD_URL,
        "record_doi": DOI,
        "record_license": LICENSE_ID,
        "record_metadata_path": str(output_dir / "zenodo-record.json"),
        "record_metadata_sha256": sha256_file(
            output_dir / "zenodo-record.json"
        ),
        "archive_name": ARCHIVE_NAME,
        "archive_url": ARCHIVE_URL,
        "archive_bytes": ARCHIVE_BYTES,
        "archive_md5": ARCHIVE_MD5,
        "full_archive_downloaded": False,
        "selective_https_range_extraction": True,
        "central_directory_path": str(
            output_dir / "zip-central-directory.bin"
        ),
        "central_directory_sha256": sha256_file(
            output_dir / "zip-central-directory.bin"
        ),
        "central_directory_entries": expected_entries,
        "source_range_evidence": {
            "tail": {
                "start": tail_start,
                "end": ARCHIVE_BYTES - 1,
                "sha256": sha256_bytes(tail),
                "headers": tail_headers,
            },
            "central_directory": {
                "start": central_offset,
                "end": central_offset + central_size - 1,
                "sha256": sha256_bytes(central),
                "headers": central_headers,
            },
            "scan_members": {
                "start": scan_start,
                "end": scan_end,
                "sha256": sha256_bytes(scan_span),
                "headers": scan_headers,
            },
            "musicxml_members": {
                "start": xml_start,
                "end": xml_end,
                "sha256": sha256_bytes(xml_span),
                "headers": xml_headers,
            },
        },
        "boundary_contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
        "work_count": len(cases),
        "physical_scan_page_count": sum(
            int(row["pages"]) for row in cases
        ),
        "scan_bytes": sum(int(row["scan_bytes"]) for row in cases),
        "reference_musicxml_bytes": sum(
            int(row["reference_musicxml_bytes"]) for row in cases
        ),
        "all_references_inside_keyboard_boundary": all(
            row["boundary"]["accepted"] is True
            and row["boundary"]["score_shape"] == "keyboard"
            for row in cases
        ),
        "boundary_eligible_alignment_candidate_count": sum(
            row["boundary_eligible_alignment_candidate"] is True
            for row in cases
        ),
        "quarantined_reference_count": sum(
            row["reference_quarantined"] is True for row in cases
        ),
        "quarantined_references": [
            {
                "work": row["work"],
                "reasons": row["reference_quarantine_reasons"],
            }
            for row in cases
            if row["reference_quarantined"] is True
        ],
        "all_scan_pages_unrotated": all(
            row["page_rotations"] == [0] for row in cases
        ),
        "all_scans_image_only_candidates": all(
            row["image_only_scan_candidate"] is True
            for row in cases
        ),
        "same_named_work_pairing_complete": True,
        "exact_page_measure_alignment_verified": False,
        "independent_double_annotation_complete": False,
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_evaluation_authorized": False,
        "release_authorized": False,
        "authorization_reason": (
            "official CC BY 3.0 physical scans and same-named MusicXML "
            "references are hash/CRC/parser verified; references outside "
            "the product boundary are quarantined, and exact page-to-measure "
            "alignment plus independent double annotation remain incomplete"
        ),
        "cases": cases,
    }
    atomic_write_json(manifest_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    report = acquire(
        args.output_dir,
        args.manifest,
        timeout_seconds=args.timeout_seconds,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "work_count",
                    "physical_scan_page_count",
                    "scan_bytes",
                    "reference_musicxml_bytes",
                    "all_references_inside_keyboard_boundary",
                    "boundary_eligible_alignment_candidate_count",
                    "quarantined_reference_count",
                    "all_scan_pages_unrotated",
                    "all_scans_image_only_candidates",
                    "training_authorized",
                    "release_authorized",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
