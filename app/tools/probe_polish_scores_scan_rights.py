from __future__ import annotations

"""Verify object-level rights for Polish Scores scan assets.

The Humdrum repository licenses its transcriptions under CC BY 4.0.  That
license is not evidence that a linked scan has the same rights.  This tool
therefore queries the authoritative NIFC object metadata and keeps the two
rights decisions separate.  It never downloads scan PDFs and never authorizes
training or evaluation by itself.
"""

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.util import atomic_write_json, sha256_file, utc_now_iso


USER_AGENT = "ScoreScan-OMR-rights-probe/1"
ROLE = "scan_rights_evidence_only_not_training_or_evaluation"
MAX_METADATA_BYTES = 4 * 1024 * 1024
PDF_URL_PATTERN = re.compile(
    r"^https://repozytorium\.nifc\.pl/islandora/object/"
    r"(?P<namespace>[a-z0-9_-]+)(?:%3A|:)(?P<object_id>[0-9]+)"
    r"/datastream/PDF/view$",
    re.IGNORECASE,
)
CC_BY_4_PATTERN = re.compile(
    r"(?:^|\b)CC[\s-]*BY[\s-]*4\.0(?:\b|\s|\()",
    re.IGNORECASE,
)


def metadata_url(pdf_url: str) -> str:
    match = PDF_URL_PATTERN.fullmatch(pdf_url.strip())
    if match is None:
        raise ValueError(f"unsupported NIFC PDF URL: {pdf_url}")
    namespace = match.group("namespace")
    object_id = match.group("object_id")
    return (
        "https://repozytorium.nifc.pl/islandora/object/"
        f"{namespace}%3A{object_id}/datastream/MODS/view"
    )


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_rights(metadata: bytes) -> tuple[list[str], bool]:
    if len(metadata) > MAX_METADATA_BYTES:
        raise ValueError("NIFC metadata exceeds the safety limit")
    root = ET.fromstring(metadata)
    values = sorted(
        {
            " ".join((element.text or "").split())
            for element in root.iter()
            if _local_name(element.tag) == "accessCondition"
            and (element.text or "").strip()
        }
    )
    verified = any(CC_BY_4_PATTERN.search(value) is not None for value in values)
    return values, verified


def fetch_metadata(
    url: str,
    *,
    timeout_seconds: float,
    attempts: int = 3,
) -> tuple[bytes, Mapping[str, str]]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/xml,text/xml;q=0.9",
                "User-Agent": USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout_seconds,
            ) as response:
                length = response.headers.get("Content-Length")
                if length and int(length) > MAX_METADATA_BYTES:
                    raise ValueError("NIFC metadata exceeds the safety limit")
                payload = response.read(MAX_METADATA_BYTES + 1)
                if len(payload) > MAX_METADATA_BYTES:
                    raise ValueError("NIFC metadata exceeds the safety limit")
                return payload, {
                    "content_type": response.headers.get("Content-Type", ""),
                    "etag": response.headers.get("ETag", ""),
                    "last_modified": response.headers.get("Last-Modified", ""),
                }
        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            ValueError,
        ) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(0.5 * (2**attempt))
    assert last_error is not None
    raise last_error


def probe_source(
    pdf_url: str,
    *,
    timeout_seconds: float,
    fetcher: Callable[
        [str], tuple[bytes, Mapping[str, str]]
    ],
) -> dict[str, object]:
    try:
        mods_url = metadata_url(pdf_url)
    except ValueError as error:
        return {
            "pdf_url": pdf_url,
            "metadata_url": "",
            "status": "unsupported_url",
            "rights_values": [],
            "scan_asset_cc_by_4_verified": False,
            "error": str(error),
        }
    try:
        payload, headers = fetcher(mods_url)
        rights, verified = parse_rights(payload)
    except (OSError, TimeoutError, urllib.error.URLError, ValueError, ET.ParseError) as error:
        return {
            "pdf_url": pdf_url,
            "metadata_url": mods_url,
            "status": "fetch_or_parse_failed",
            "rights_values": [],
            "scan_asset_cc_by_4_verified": False,
            "error": f"{type(error).__name__}: {error}",
        }
    return {
        "pdf_url": pdf_url,
        "metadata_url": mods_url,
        "metadata_sha256": hashlib.sha256(payload).hexdigest(),
        "metadata_content_type": headers.get("content_type", ""),
        "metadata_etag": headers.get("etag", ""),
        "metadata_last_modified": headers.get("last_modified", ""),
        "status": (
            "verified_cc_by_4"
            if verified
            else "rights_missing_or_not_cc_by_4"
        ),
        "rights_values": rights,
        "scan_asset_cc_by_4_verified": verified,
        "error": "",
    }


def build_report(
    catalog: dict[str, object],
    *,
    catalog_path: Path,
    workers: int,
    timeout_seconds: float,
    fetcher_factory: Callable[
        [float], Callable[[str], tuple[bytes, Mapping[str, str]]]
    ]
    | None = None,
) -> dict[str, object]:
    if catalog.get("role") != "candidate_catalog_not_training_or_evaluation":
        raise ValueError("unexpected Polish Scores catalog role")
    if catalog.get("training_authorized") is not False:
        raise ValueError("source catalog unexpectedly authorizes training")
    cases = catalog.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Polish Scores catalog cases are missing")
    printed_rows = [
        row
        for row in cases
        if isinstance(row, dict)
        and "/galeria/druki-muzyczne/" in str(row.get("scan_url", ""))
    ]
    pdf_urls = sorted(
        {
            str(row.get("scan_pdf_url", "")).strip()
            for row in printed_rows
            if str(row.get("scan_pdf_url", "")).strip()
        }
    )
    if not pdf_urls:
        raise ValueError("Polish Scores catalog has no printed scan PDF URLs")

    if fetcher_factory is None:
        fetcher_factory = lambda timeout: lambda url: fetch_metadata(
            url,
            timeout_seconds=timeout,
        )
    fetcher = fetcher_factory(timeout_seconds)
    sources: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                probe_source,
                pdf_url,
                timeout_seconds=timeout_seconds,
                fetcher=fetcher,
            ): pdf_url
            for pdf_url in pdf_urls
        }
        for position, future in enumerate(as_completed(futures), start=1):
            sources.append(future.result())
            if position % 25 == 0 or position == len(futures):
                print(f"[{position}/{len(futures)}] scan rights probed", flush=True)
    sources.sort(key=lambda source: str(source["pdf_url"]))
    source_by_url = {
        str(source["pdf_url"]): source
        for source in sources
    }
    verified_rows = [
        row
        for row in printed_rows
        if source_by_url.get(
            str(row.get("scan_pdf_url", "")).strip(),
            {},
        ).get("scan_asset_cc_by_4_verified")
        is True
    ]
    eligible_catalog_candidates = [
        row for row in verified_rows if row.get("accepted") is True
    ]
    verified_sources = [
        source
        for source in sources
        if source["scan_asset_cc_by_4_verified"] is True
    ]
    failed_sources = [
        source
        for source in sources
        if source["status"] in {"unsupported_url", "fetch_or_parse_failed"}
    ]
    return {
        "format": 1,
        "created_at": utc_now_iso(),
        "name": "Polish Scores authoritative scan-rights evidence",
        "role": ROLE,
        "catalog_path": str(catalog_path),
        "catalog_sha256": sha256_file(catalog_path),
        "catalog_revision": catalog.get("revision", ""),
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_authorized": False,
        "authorization_reason": (
            "object-level rights evidence alone does not prove download integrity, "
            "page alignment, reference conversion, boundary validity, or benchmark "
            "isolation"
        ),
        "rights_policy": (
            "only an authoritative MODS accessCondition explicitly naming CC BY "
            "4.0 is accepted; missing or ambiguous rights are rejected"
        ),
        "printed_case_count": len(printed_rows),
        "unique_scan_source_count": len(sources),
        "verified_cc_by_4_source_count": len(verified_sources),
        "failed_source_count": len(failed_sources),
        "printed_rows_with_verified_scan_rights": len(verified_rows),
        "strict_catalog_candidates_with_verified_scan_rights": len(
            eligible_catalog_candidates
        ),
        "verified_source_urls": [
            str(source["pdf_url"]) for source in verified_sources
        ],
        "strict_catalog_candidates": eligible_catalog_candidates,
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    if not 2.0 <= args.timeout_seconds <= 60.0:
        raise ValueError("timeout-seconds must be between 2 and 60")
    catalog_path = args.catalog_path.resolve()
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    report = build_report(
        catalog,
        catalog_path=catalog_path,
        workers=args.workers,
        timeout_seconds=args.timeout_seconds,
    )
    atomic_write_json(args.output_path.resolve(), report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "printed_case_count",
                    "unique_scan_source_count",
                    "verified_cc_by_4_source_count",
                    "failed_source_count",
                    "printed_rows_with_verified_scan_rights",
                    "strict_catalog_candidates_with_verified_scan_rights",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
