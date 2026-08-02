from __future__ import annotations

"""Find exact Internet Archive mirrors of identity-verified IMSLP scans."""

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Callable, Mapping

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_ROOT / "src"))

from scorescan.util import atomic_write_json, sha256_file, utc_now_iso  # noqa: E402

from app.tools.probe_openscore_imslp_scan_sources import (  # noqa: E402
    ROLE as IMSLP_EVIDENCE_ROLE,
    candidate_identity_matches,
)


ROLE = "archive_imslp_exact_filename_mirror_candidates_not_downloaded"
USER_AGENT = "ScoreScan-OMR-Archive-mirror-probe/1"
ARCHIVE_METADATA_ROOT = "https://archive.org/metadata/"
ARCHIVE_DOWNLOAD_ROOT = "https://archive.org/download/"
ARCHIVE_SEARCH_URL = "https://archive.org/advancedsearch.php"
MAXIMUM_JSON_BYTES = 32 * 1024 * 1024
GENERIC_QUERY_TOKENS = {
    "flat",
    "major",
    "minor",
    "number",
    "opus",
    "quartet",
    "string",
    "the",
}


def _tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", value).casefold()
    return re.findall(r"[a-z]+|[0-9]+", normalized)


def _lucene_term(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def archive_query(candidate: Mapping[str, object]) -> str:
    composer_tokens = [
        token
        for token in _tokens(str(candidate.get("composer", "")))
        if len(token) >= 4
    ]
    title_tokens = [
        token
        for token in _tokens(str(candidate.get("work_title", "")))
        if (
            (len(token) >= 4 or token.isdigit())
            and token not in GENERIC_QUERY_TOKENS
        )
    ]
    if not composer_tokens or not title_tokens:
        raise ValueError("candidate has insufficient archive search identity")
    composer_terms = sorted(
        composer_tokens,
        key=lambda token: (-len(token), token),
    )[:1]
    number_terms = sorted(
        {token for token in title_tokens if token.isdigit()},
        key=lambda token: (-len(token), token),
    )
    distinctive_terms = sorted(
        {token for token in title_tokens if not token.isdigit()},
        key=lambda token: (-len(token), token),
    )
    title_terms = (
        number_terms[:2]
        if number_terms
        else distinctive_terms[:2]
    )
    clauses = [
        "collection:imslp",
        "mediatype:texts",
        *[f"creator:{_lucene_term(token)}" for token in composer_terms],
        *[f"title:{_lucene_term(token)}" for token in title_terms],
    ]
    return " AND ".join(clauses)


def _read_json_url(url: str, *, timeout_seconds: float) -> dict[str, object]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "archive.org":
        raise ValueError("archive request left the approved host")
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        length = response.headers.get("Content-Length")
        if length and int(length) > MAXIMUM_JSON_BYTES:
            raise ValueError("archive metadata exceeds the safety limit")
        payload = response.read(MAXIMUM_JSON_BYTES + 1)
    if len(payload) > MAXIMUM_JSON_BYTES:
        raise ValueError("archive metadata exceeds the safety limit")
    result = json.loads(payload)
    if not isinstance(result, dict):
        raise ValueError("archive metadata is not an object")
    return result


def search_archive(
    query: str,
    *,
    timeout_seconds: float,
    reader: Callable[[str], dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    parameters = urllib.parse.urlencode(
        {
            "q": query,
            "fl[]": ["identifier", "title", "creator"],
            "rows": 10,
            "page": 1,
            "output": "json",
        },
        doseq=True,
    )
    url = f"{ARCHIVE_SEARCH_URL}?{parameters}"
    payload = (
        reader(url)
        if reader is not None
        else _read_json_url(url, timeout_seconds=timeout_seconds)
    )
    response = payload.get("response")
    documents = response.get("docs") if isinstance(response, dict) else None
    if not isinstance(documents, list):
        raise ValueError("archive search response has no documents")
    return [row for row in documents if isinstance(row, dict)]


def fetch_archive_metadata(
    identifier: str,
    *,
    timeout_seconds: float,
    reader: Callable[[str], dict[str, object]] | None = None,
) -> dict[str, object]:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", identifier):
        raise ValueError("unsafe archive identifier")
    url = ARCHIVE_METADATA_ROOT + urllib.parse.quote(identifier)
    return (
        reader(url)
        if reader is not None
        else _read_json_url(url, timeout_seconds=timeout_seconds)
    )


def _expected_filename(candidate: Mapping[str, object]) -> str:
    parsed = urllib.parse.urlparse(str(candidate.get("direct_pdf_url", "")))
    filename = urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1])
    if (
        parsed.scheme != "https"
        or parsed.hostname != "imslp.org"
        or not filename.casefold().endswith(".pdf")
        or "/" in filename
        or "\\" in filename
    ):
        raise ValueError("candidate has no safe IMSLP PDF filename")
    return filename


def exact_mirror_file(
    candidate: Mapping[str, object],
    item: Mapping[str, object],
) -> dict[str, object] | None:
    metadata = item.get("metadata")
    files = item.get("files")
    if not isinstance(metadata, dict) or not isinstance(files, list):
        return None
    collections = metadata.get("collection")
    if isinstance(collections, str):
        collection_values = {collections}
    elif isinstance(collections, list):
        collection_values = {str(value) for value in collections}
    else:
        return None
    if "imslp" not in collection_values:
        return None
    page_title = (
        f"{metadata.get('title', '')} ({metadata.get('creator', '')})"
    )
    if not candidate_identity_matches(
        candidate,
        {"page_title": page_title},
    ):
        return None
    expected = _expected_filename(candidate)
    matches = [
        row
        for row in files
        if isinstance(row, dict)
        and row.get("name") == expected
        and row.get("source") == "original"
        and row.get("format") == "Image Container PDF"
    ]
    if len(matches) != 1:
        return None
    row = matches[0]
    md5 = str(row.get("md5", "")).casefold()
    sha1 = str(row.get("sha1", "")).casefold()
    try:
        byte_count = int(str(row.get("size", "")))
    except ValueError:
        return None
    if (
        re.fullmatch(r"[0-9a-f]{32}", md5) is None
        or re.fullmatch(r"[0-9a-f]{40}", sha1) is None
        or byte_count <= 0
    ):
        return None
    identifier = str(metadata.get("identifier", ""))
    if re.fullmatch(r"[A-Za-z0-9._-]+", identifier) is None:
        return None
    quoted_name = urllib.parse.quote(expected)
    return {
        "archive_identifier": identifier,
        "archive_item_url": f"https://archive.org/details/{identifier}",
        "archive_metadata_url": f"{ARCHIVE_METADATA_ROOT}{identifier}",
        "archive_collection": sorted(collection_values),
        "archive_title": metadata.get("title", ""),
        "archive_creator": metadata.get("creator", ""),
        "archive_date": metadata.get("date", ""),
        "original_filename": expected,
        "original_bytes": byte_count,
        "original_md5": md5,
        "original_sha1": sha1,
        "download_url": (
            f"{ARCHIVE_DOWNLOAD_ROOT}{identifier}/{quoted_name}"
        ),
    }


def probe_candidate(
    candidate: Mapping[str, object],
    *,
    searcher: Callable[[str], list[dict[str, object]]],
    metadata_fetcher: Callable[[str], dict[str, object]],
) -> dict[str, object]:
    query = archive_query(candidate)
    try:
        documents = searcher(query)
        matches: list[dict[str, object]] = []
        inspected_identifiers: list[str] = []
        for document in documents:
            identifier = str(document.get("identifier", ""))
            if not identifier or identifier in inspected_identifiers:
                continue
            inspected_identifiers.append(identifier)
            item = metadata_fetcher(identifier)
            match = exact_mirror_file(candidate, item)
            if match is not None:
                matches.append(match)
        if len(matches) != 1:
            return {
                "imslp_source_id": candidate.get("imslp_source_id", ""),
                "status": "no_unique_exact_archive_mirror",
                "archive_query": query,
                "search_result_count": len(documents),
                "inspected_archive_identifiers": inspected_identifiers,
                "exact_match_count": len(matches),
                "error": "",
            }
        return {
            **dict(candidate),
            **matches[0],
            "status": "exact_archive_mirror_candidate",
            "archive_query": query,
            "search_result_count": len(documents),
            "inspected_archive_identifiers": inspected_identifiers,
            "exact_match_count": 1,
            "error": "",
        }
    except (OSError, TimeoutError, ValueError) as error:
        return {
            "imslp_source_id": candidate.get("imslp_source_id", ""),
            "status": "archive_fetch_or_parse_failed",
            "archive_query": query,
            "search_result_count": 0,
            "inspected_archive_identifiers": [],
            "exact_match_count": 0,
            "error": f"{type(error).__name__}: {error}",
        }


def build_report(
    evidence: Mapping[str, object],
    *,
    evidence_path: Path,
    timeout_seconds: float,
    minimum_interval_seconds: float,
    limit: int | None = None,
    searcher: Callable[[str], list[dict[str, object]]] | None = None,
    metadata_fetcher: Callable[[str], dict[str, object]] | None = None,
) -> dict[str, object]:
    if (
        evidence.get("role") != IMSLP_EVIDENCE_ROLE
        or evidence.get("training_authorized") is not False
        or evidence.get("evaluation_authorized") is not False
        or evidence.get("release_authorized") is not False
    ):
        raise ValueError("unexpected IMSLP evidence authorization contract")
    candidates = evidence.get("verified_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("identity-verified IMSLP candidates are missing")
    selected = sorted(
        (row for row in candidates if isinstance(row, dict)),
        key=lambda row: int(str(row["imslp_source_id"])),
    )
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        selected = selected[:limit]
    polite_reader: Callable[[str], dict[str, object]] | None = None
    if searcher is None or metadata_fetcher is None:
        last_request = 0.0

        def polite_reader(url: str) -> dict[str, object]:
            nonlocal last_request
            remaining = minimum_interval_seconds - (
                time.monotonic() - last_request
            )
            if remaining > 0:
                time.sleep(remaining)
            last_request = time.monotonic()
            return _read_json_url(url, timeout_seconds=timeout_seconds)

    if searcher is None:
        assert polite_reader is not None
        searcher = lambda query: search_archive(
            query,
            timeout_seconds=timeout_seconds,
            reader=polite_reader,
        )
    if metadata_fetcher is None:
        assert polite_reader is not None
        metadata_fetcher = lambda identifier: fetch_archive_metadata(
            identifier,
            timeout_seconds=timeout_seconds,
            reader=polite_reader,
        )
    rows: list[dict[str, object]] = []
    for position, candidate in enumerate(selected, start=1):
        rows.append(
            probe_candidate(
                candidate,
                searcher=searcher,
                metadata_fetcher=metadata_fetcher,
            )
        )
        print(
            f"[{position}/{len(selected)}] archive mirror candidate probed",
            flush=True,
        )
        if position < len(selected) and minimum_interval_seconds > 0:
            time.sleep(minimum_interval_seconds)
    verified = [
        row
        for row in rows
        if row.get("status") == "exact_archive_mirror_candidate"
    ]
    statuses = Counter(str(row["status"]) for row in rows)
    return {
        "format": 1,
        "created_at": utc_now_iso(),
        "name": "OpenScore IMSLP Internet Archive exact mirror evidence",
        "role": ROLE,
        "evidence_path": str(evidence_path),
        "evidence_sha256": sha256_file(evidence_path),
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_authorized": False,
        "authorization_reason": (
            "item identity and exact original filename are cross-verified, "
            "but the mirrored PDF bytes, page count, semantic alignment, and "
            "split isolation have not all been verified"
        ),
        "candidate_count": len(rows),
        "exact_mirror_candidate_count": len(verified),
        "exact_mirror_work_count": len(
            {str(row["work_fingerprint"]) for row in verified}
        ),
        "status_counts": dict(sorted(statuses.items())),
        "exact_mirror_candidates": verified,
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--minimum-request-interval-seconds",
        type=float,
        default=1.0,
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not 3 <= args.timeout_seconds <= 180:
        raise ValueError("timeout-seconds must be between 3 and 180")
    if not 0 <= args.minimum_request_interval_seconds <= 30:
        raise ValueError("minimum request interval must be between 0 and 30")
    evidence_path = args.evidence_path.resolve()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    report = build_report(
        evidence,
        evidence_path=evidence_path,
        timeout_seconds=args.timeout_seconds,
        minimum_interval_seconds=args.minimum_request_interval_seconds,
        limit=args.limit,
    )
    atomic_write_json(args.output_path.resolve(), report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "candidate_count",
                    "exact_mirror_candidate_count",
                    "exact_mirror_work_count",
                    "status_counts",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
