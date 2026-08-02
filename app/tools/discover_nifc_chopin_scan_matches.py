from __future__ import annotations

"""Discover authoritative NIFC scan objects for clean Chopin references.

Search results are deliberately only candidates.  A publisher/opus metadata
match does not prove edition, page order or exact correspondence.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from lxml import html

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT / "src"))

from app.tools.acquire_nifc_chopin_matched_scans import (  # noqa: E402
    MAXIMUM_METADATA_BYTES,
    USER_AGENT,
    parse_mods,
    parse_child_membership,
)
from app.tools.catalog_nifc_chopin_reference_quality import (  # noqa: E402
    ROLE as QUALITY_CATALOG_ROLE,
)
from scorescan.util import atomic_write_bytes, atomic_write_json, sha256_file, utc_now_iso  # noqa: E402


ROLE = "nifc_chopin_scan_match_discovery_not_training_or_evaluation"
SEARCH_BASE = "https://repozytorium.nifc.pl/islandora/search"
NIFC_OBJECT_PATTERN = re.compile(
    r"/islandora/object/nifc(?:%3A|:)(\d+)",
    re.IGNORECASE,
)
OPUS_PATTERN = re.compile(r"\bop(?:us)?[.\s:]*(\d+)\b", re.IGNORECASE)
PRINT_REPRODUCTION_PATTERN = re.compile(
    r"\b(?:druku\s+muzycznego|pierwodruku|printed\s+music|music\s+print)\b",
    re.IGNORECASE,
)
INCOMPLETE_REPRODUCTION_PATTERN = re.compile(
    r"\breprodukcj[ae]\b.{0,80}\b(?:fragmentu|strony\s+tytułowej)\b",
    re.IGNORECASE,
)
EDITORIAL_ANNOTATION_PATTERN = re.compile(
    r"\badnotacjami\s+redaktora\b",
    re.IGNORECASE,
)


def search_result_pids(payload: bytes) -> list[str]:
    document = html.fromstring(payload)
    values: list[str] = []
    seen: set[str] = set()
    for href in document.xpath(
        "//div[contains(concat(' ', normalize-space(@class), ' '), "
        "' islandora-solr-search-result ')]//a[@href]/@href"
    ):
        match = NIFC_OBJECT_PATTERN.search(str(href))
        if match is None:
            continue
        value = f"nifc:{match.group(1)}"
        if value not in seen:
            seen.add(value)
            values.append(value)
    if len(values) > 100:
        raise ValueError("NIFC search result hard limit exceeded")
    return values


def _request(
    url: str,
    *,
    timeout: float,
    attempts: int = 4,
) -> bytes:
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Connection": "close",
            },
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout,
            ) as response:
                payload = response.read(MAXIMUM_METADATA_BYTES + 1)
            if len(payload) > MAXIMUM_METADATA_BYTES:
                raise ValueError(
                    f"NIFC response exceeds byte limit: {url}"
                )
            return payload
        except urllib.error.HTTPError as error:
            last_error = error
            if (
                error.code not in {408, 425, 429}
                and error.code < 500
            ):
                raise
            if attempt + 1 < attempts:
                time.sleep(0.5 * (2**attempt))
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


def _search_url(query: str) -> str:
    return (
        f"{SEARCH_BASE}/{urllib.parse.quote(query, safe='')}"
        "?type=lucene"
    )


def _publisher_key(value: str) -> str:
    folded = value.casefold()
    for marker in (
        "breitkopf",
        "kistner",
        "troupenas",
        "schlesinger",
        "wessel",
        "pleyel",
        "hofmeister",
        "haslinger",
    ):
        if marker in folded:
            return marker
    words = re.findall(r"[a-z]{4,}", folded)
    return words[0] if words else ""


def edition_document_text(metadata: dict[str, object]) -> str:
    """Return edition-imprint text without contributor/holding names.

    MODS ``namePart`` includes holding institutions and related publishers.
    Those names are useful for discovery but cannot prove the publisher of the
    photographed edition. Titles and descriptive notes contain the actual
    imprint evidence used below.
    """

    values: list[str] = []
    for field in ("titles", "notes"):
        raw = metadata.get(field, [])
        if isinstance(raw, list):
            values.extend(str(value) for value in raw)
    return re.sub(r"\s+", " ", " ".join(values)).strip().casefold()


def described_page_extent(metadata: dict[str, object]) -> int:
    """Extract only explicitly page-prefixed page numbers from MODS notes."""

    values: list[int] = []
    for start, end in re.findall(
        r"\b(?:p|s|k)\.?\s*\[?(\d{1,3})\]?"
        r"(?:\s*[-–—]\s*\[?(\d{1,3})\]?)?",
        edition_document_text(metadata),
    ):
        values.extend(
            value
            for value in (int(start), int(end) if end else 0)
            if 0 < value <= 200
        )
    for first, second in re.findall(
        r"\b(?:p|s|k)\.?\s*\[\s*(\d{1,3})\s*,\s*(\d{1,3})",
        edition_document_text(metadata),
    ):
        values.extend(
            value
            for value in (int(first), int(second))
            if 0 < value <= 200
        )
    return max(values, default=0)


def publication_year_candidates(
    metadata: dict[str, object],
) -> list[int]:
    """Return plausible historical publication years, not scan dates."""

    return sorted(
        {
            int(value)
            for value in re.findall(
                r"(?<!\d)(1[89]\d{2})(?!\d)",
                edition_document_text(metadata),
            )
            if int(value) <= 1950
        }
    )


def title_similarity(
    source_metadata: dict[str, object],
    candidate_metadata: dict[str, object],
) -> float:
    def tokens(metadata: dict[str, object]) -> set[str]:
        raw = metadata.get("titles", [])
        if not isinstance(raw, list):
            return set()
        return {
            value
            for value in re.findall(
                r"[a-zà-öø-ÿ]{3,}",
                " ".join(str(item) for item in raw).casefold(),
            )
            if value
            not in {
                "the",
                "pour",
                "par",
                "chez",
                "des",
                "les",
                "une",
            }
        }

    source = tokens(source_metadata)
    candidate = tokens(candidate_metadata)
    if not source or not candidate:
        return 0.0
    return len(source & candidate) / len(source | candidate)


def described_collection_item_count(
    metadata: dict[str, object],
) -> int:
    text = edition_document_text(metadata)
    word_values = {
        "trois": 3,
        "three": 3,
        "quatre": 4,
        "four": 4,
        "cinq": 5,
        "five": 5,
        "vingt quatre": 24,
        "twenty four": 24,
    }
    folded = re.sub(r"[-–—]+", " ", text)
    for marker, value in sorted(
        word_values.items(),
        key=lambda item: -len(item[0]),
    ):
        if re.search(rf"\b{re.escape(marker)}\b", folded):
            return value
    match = re.search(
        r"\b(\d{1,2})\s+"
        r"(?:mazurk|mazourk|prelud|étud|etud)",
        folded,
    )
    return int(match.group(1)) if match else 0


def metadata_match_evidence(
    case: dict[str, object],
    metadata: dict[str, object],
    *,
    catalog_source_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    records = case.get("records")
    if not isinstance(records, dict):
        raise ValueError("quality case records are missing")
    publisher = str(records.get("PPR", ""))
    publisher_key = _publisher_key(publisher)
    opus_value = str(records.get("OPS", ""))
    opus_match = OPUS_PATTERN.search(opus_value)
    opus = opus_match.group(1) if opus_match else ""
    document = edition_document_text(metadata)
    normalized_tokens = re.sub(r"[^a-z0-9]+", " ", document)
    publisher_matches = bool(
        publisher_key and publisher_key in normalized_tokens
    )
    opus_matches = bool(
        opus
        and re.search(
            rf"\bop(?:us)?\s*{re.escape(opus)}\b",
            normalized_tokens,
        )
    )
    title_markers = [
        marker
        for marker in re.findall(
            r"[a-z]{5,}",
            str(records.get("rism-genre", "")).casefold(),
        )
        if marker not in {"major", "minor"}
    ]
    genre_matches = (
        not title_markers
        or any(marker in normalized_tokens for marker in title_markers)
    )
    source_edition_tokens = edition_number_tokens(
        catalog_source_metadata or {},
    )
    candidate_edition_tokens = edition_number_tokens(metadata)
    shared_edition_tokens = sorted(
        source_edition_tokens & candidate_edition_tokens
    )
    edition_token_matches = bool(shared_edition_tokens)
    print_reproduction = (
        PRINT_REPRODUCTION_PATTERN.search(document) is not None
    )
    incomplete_reproduction = (
        INCOMPLETE_REPRODUCTION_PATTERN.search(document) is not None
    )
    editorial_annotations = (
        EDITORIAL_ANNOTATION_PATTERN.search(document) is not None
    )
    strict_metadata_candidate = (
        publisher_matches
        and opus_matches
        and genre_matches
        and edition_token_matches
        and metadata.get("cc_by_4_explicit") is True
    )
    layout_audit_candidate = (
        strict_metadata_candidate
        and print_reproduction
        and not incomplete_reproduction
        and not editorial_annotations
    )
    source_page_extent = described_page_extent(
        catalog_source_metadata or {}
    )
    candidate_page_extent = described_page_extent(metadata)
    source_years = publication_year_candidates(
        catalog_source_metadata or {}
    )
    candidate_years = publication_year_candidates(metadata)
    source_year = min(source_years, default=0)
    candidate_year = min(candidate_years, default=0)
    year_distance = (
        abs(source_year - candidate_year)
        if source_year and candidate_year
        else 999
    )
    title_score = title_similarity(
        catalog_source_metadata or {},
        metadata,
    )
    source_item_count = described_collection_item_count(
        catalog_source_metadata or {}
    )
    candidate_item_count = described_collection_item_count(metadata)
    item_count_matches = bool(
        source_item_count
        and candidate_item_count
        and source_item_count == candidate_item_count
    )
    selection_score = (
        (200 if "pierwodruku" in document else 0)
        + (150 if item_count_matches else 0)
        - (
            300
            if source_item_count
            and candidate_item_count
            and not item_count_matches
            else 0
        )
        + (120 if source_page_extent == candidate_page_extent else 0)
        - (
            10 * abs(source_page_extent - candidate_page_extent)
            if source_page_extent and candidate_page_extent
            else 100
        )
        - min(year_distance, 100)
        + round(100 * title_score)
    )
    return {
        "publisher_key": publisher_key,
        "publisher_matches": publisher_matches,
        "opus": opus,
        "opus_matches": opus_matches,
        "genre_markers": title_markers,
        "genre_matches": genre_matches,
        "catalog_source_edition_tokens": sorted(
            source_edition_tokens
        ),
        "candidate_edition_tokens": sorted(
            candidate_edition_tokens
        ),
        "shared_edition_tokens": shared_edition_tokens,
        "edition_token_matches": edition_token_matches,
        "print_reproduction": print_reproduction,
        "incomplete_reproduction": incomplete_reproduction,
        "editorial_annotations": editorial_annotations,
        "source_page_extent": source_page_extent,
        "candidate_page_extent": candidate_page_extent,
        "source_publication_year_candidates": source_years,
        "candidate_publication_year_candidates": candidate_years,
        "publication_year_distance": year_distance,
        "title_similarity": title_score,
        "source_collection_item_count": source_item_count,
        "candidate_collection_item_count": candidate_item_count,
        "collection_item_count_matches": item_count_matches,
        "selection_score": selection_score,
        "strict_metadata_candidate": strict_metadata_candidate,
        "layout_audit_candidate": layout_audit_candidate,
    }


def edition_number_tokens(
    metadata: dict[str, object],
) -> set[str]:
    values: list[str] = []
    for field in ("titles", "notes"):
        raw = metadata.get(field, [])
        if isinstance(raw, list):
            values.extend(str(value) for value in raw)
    tokens = {
        match.group(0)
        for match in re.finditer(r"(?<!\d)\d{3,6}(?!\d)", " ".join(values))
    }
    return {
        token
        for token in tokens
        if not 1700 <= int(token) <= 2099
    }


def discover(
    quality_catalog: dict[str, object],
    *,
    quality_catalog_path: Path,
    timeout: float = 30.0,
    workers: int = 8,
    cache_dir: Path | None = None,
) -> dict[str, object]:
    if quality_catalog.get("role") != QUALITY_CATALOG_ROLE:
        raise ValueError("unexpected Chopin reference quality catalog role")
    for field in (
        "training_authorized",
        "evaluation_authorized",
        "release_authorized",
    ):
        if quality_catalog.get(field) is not False:
            raise ValueError(f"quality catalog unexpectedly sets {field}")
    cases = quality_catalog.get("high_priority_candidates")
    if not isinstance(cases, list) or not cases:
        raise ValueError("quality catalog has no clean match candidates")
    if workers <= 0 or timeout <= 0:
        raise ValueError("workers and timeout must be positive")
    if cache_dir is None:
        cache_dir = quality_catalog_path.parent / (
            "nifc_chopin_discovery_cache_v1"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)

    def cached_request(url: str, path: Path) -> bytes:
        if path.is_file():
            payload = path.read_bytes()
            if payload and len(payload) <= MAXIMUM_METADATA_BYTES:
                return payload
            raise ValueError(f"invalid NIFC discovery cache: {path}")
        payload = _request(url, timeout=timeout)
        atomic_write_bytes(path, payload)
        return payload

    searches: list[dict[str, object]] = []
    source_searches: dict[str, dict[str, object]] = {}
    all_pids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("invalid clean reference candidate")
        records = case.get("records")
        if not isinstance(records, dict):
            raise ValueError("clean reference candidate has no records")
        publisher = str(records.get("PPR", "")).strip()
        opus = str(records.get("OPS", "")).strip()
        if not publisher or not opus:
            raise ValueError("clean reference candidate lacks search identity")
        publisher_key = _publisher_key(publisher)
        opus_match = OPUS_PATTERN.search(opus)
        if not publisher_key or opus_match is None:
            raise ValueError("clean reference has no safe field query")
        query = (
            "mods_name_corporate_namePart_mt:"
            f"{publisher_key} AND mods_titleInfo_title_mt:"
            f"{opus_match.group(1)}"
        )
        url = _search_url(query)
        search_key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        pids = search_result_pids(
            cached_request(
                url,
                cache_dir / f"search-{search_key}.html",
            )
        )
        searches.append(
            {
                "reference_path": case.get("path", ""),
                "query": query,
                "url": url,
                "result_pids": pids,
            }
        )
        all_pids.update(pids)
        source_id = str(records.get("rism-773a-OPR", "")).strip()
        if source_id:
            source_url = (
                f"{SEARCH_BASE}/{urllib.parse.quote(source_id, safe='')}"
                "?type=dismax"
            )
            source_key = hashlib.sha256(
                source_url.encode("utf-8")
            ).hexdigest()
            source_pids = search_result_pids(
                cached_request(
                    source_url,
                    cache_dir / f"search-{source_key}.html",
                )
            )
            source_searches[str(case["path"])] = {
                "rism_parent_id": source_id,
                "url": source_url,
                "result_pids": source_pids,
            }
            all_pids.update(source_pids)

    metadata_by_pid: dict[str, dict[str, object]] = {}

    def fetch_pid(
        pid: str,
    ) -> tuple[str, dict[str, object], str, str]:
        url = (
            "https://repozytorium.nifc.pl/islandora/object/"
            f"{pid}/datastream/MODS/view"
        )
        stem = pid.replace(":", "-")
        payload = cached_request(
            url,
            cache_dir / f"{stem}.mods.xml",
        )
        parsed = parse_mods(payload)
        parsed["sha256"] = hashlib.sha256(payload).hexdigest()
        rels_url = (
            "https://repozytorium.nifc.pl/islandora/object/"
            f"{pid}/datastream/RELS-EXT/view"
        )
        parent_pid = ""
        try:
            child_pid, parent_pid = parse_child_membership(
                cached_request(
                    rels_url,
                    cache_dir / f"{stem}.rels-ext.xml",
                )
            )
            if child_pid != pid:
                raise ValueError("NIFC RELS-EXT child PID drifted")
        except ValueError:
            parent_pid = ""
        return pid, parsed, url, parent_pid

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_pid, pid): pid for pid in sorted(all_pids)
        }
        for future in as_completed(futures):
            pid, metadata, url, parent_pid = future.result()
            metadata["url"] = url
            metadata["parent_pid"] = parent_pid
            metadata_by_pid[pid] = metadata

    parent_pids = sorted(
        {
            str(metadata["parent_pid"])
            for metadata in metadata_by_pid.values()
            if metadata.get("parent_pid")
            and metadata.get("parent_pid") not in metadata_by_pid
        }
    )
    parent_metadata_by_pid: dict[str, dict[str, object]] = {}

    def fetch_parent(pid: str) -> tuple[str, dict[str, object], str]:
        url = (
            "https://repozytorium.nifc.pl/islandora/object/"
            f"{pid}/datastream/MODS/view"
        )
        try:
            payload = cached_request(
                url,
                cache_dir / f"{pid.replace(':', '-')}.mods.xml",
            )
            parsed = parse_mods(payload)
            parsed["sha256"] = hashlib.sha256(payload).hexdigest()
            parsed["url"] = url
            parsed["fetch_error"] = ""
            return pid, parsed, url
        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            ValueError,
        ) as error:
            return (
                pid,
                {
                    "url": url,
                    "sha256": "",
                    "titles": [],
                    "names": [],
                    "notes": [],
                    "access_conditions": [],
                    "normalized_document_text": "",
                    "cc_by_4_explicit": False,
                    "fetch_error": f"{type(error).__name__}: {error}",
                },
                url,
            )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_parent, pid): pid for pid in parent_pids
        }
        for future in as_completed(futures):
            pid, metadata, _url = future.result()
            parent_metadata_by_pid[pid] = metadata

    search_by_path = {
        str(search["reference_path"]): search for search in searches
    }
    case_reports: list[dict[str, object]] = []
    for case in cases:
        path = str(case["path"])
        search = search_by_path[path]
        source_search = source_searches.get(path, {})
        source_pids = source_search.get("result_pids", [])
        source_pid = (
            str(source_pids[0]) if len(source_pids) == 1 else ""
        )
        catalog_source_metadata = metadata_by_pid.get(source_pid, {})
        candidates: list[dict[str, object]] = []
        for pid in search["result_pids"]:
            metadata = metadata_by_pid[str(pid)]
            parent_pid = str(metadata.get("parent_pid", ""))
            parent_metadata = (
                metadata_by_pid.get(parent_pid)
                or parent_metadata_by_pid.get(parent_pid)
                or {}
            )
            effective_metadata = dict(metadata)
            effective_metadata["normalized_document_text"] = " ".join(
                (
                    str(metadata.get("normalized_document_text", "")),
                    str(
                        parent_metadata.get(
                            "normalized_document_text",
                            "",
                        )
                    ),
                )
            )
            effective_metadata["titles"] = [
                *list(metadata.get("titles", [])),
                *list(parent_metadata.get("titles", [])),
            ]
            effective_metadata["notes"] = [
                *list(metadata.get("notes", [])),
                *list(parent_metadata.get("notes", [])),
            ]
            effective_metadata["cc_by_4_explicit"] = (
                metadata.get("cc_by_4_explicit") is True
                or parent_metadata.get("cc_by_4_explicit") is True
            )
            evidence = metadata_match_evidence(
                case,
                effective_metadata,
                catalog_source_metadata=catalog_source_metadata,
            )
            candidates.append(
                {
                    "pid": pid,
                    "page_url": (
                        "https://repozytorium.nifc.pl/islandora/object/"
                        + str(pid).replace(":", "%3A")
                    ),
                    "mods_url": metadata["url"],
                    "mods_sha256": metadata["sha256"],
                    "titles": metadata["titles"],
                    "names": metadata["names"],
                    "notes": metadata["notes"],
                    "access_conditions": metadata["access_conditions"],
                    "cc_by_4_explicit": metadata["cc_by_4_explicit"],
                    "parent_pid": parent_pid,
                    "parent_page_url": (
                        "https://repozytorium.nifc.pl/islandora/object/"
                        + parent_pid.replace(":", "%3A")
                        if parent_pid
                        else ""
                    ),
                    "parent_mods_url": parent_metadata.get("url", ""),
                    "parent_mods_sha256": parent_metadata.get(
                        "sha256",
                        "",
                    ),
                    "parent_titles": parent_metadata.get("titles", []),
                    "parent_notes": parent_metadata.get("notes", []),
                    "parent_access_conditions": parent_metadata.get(
                        "access_conditions",
                        [],
                    ),
                    "parent_cc_by_4_explicit": parent_metadata.get(
                        "cc_by_4_explicit",
                        False,
                    ),
                    "parent_fetch_error": parent_metadata.get(
                        "fetch_error",
                        "",
                    ),
                    "effective_cc_by_4_explicit": effective_metadata[
                        "cc_by_4_explicit"
                    ],
                    "match_evidence": evidence,
                }
            )
        strict = [
            candidate
            for candidate in candidates
            if candidate["match_evidence"]["strict_metadata_candidate"]
            is True
        ]
        layout_audit = [
            candidate
            for candidate in strict
            if candidate["match_evidence"]["layout_audit_candidate"]
            is True
        ]
        ranked_layout_audit = sorted(
            layout_audit,
            key=lambda candidate: (
                -int(candidate["match_evidence"]["selection_score"]),
                int(str(candidate["pid"]).split(":")[-1]),
            ),
        )
        preferred = (
            ranked_layout_audit[0] if ranked_layout_audit else None
        )
        case_reports.append(
            {
                "reference_path": path,
                "reference_sha256": case["sha256"],
                "reference_pages": case["reference_page_profile"][
                    "encoded_music_page_count"
                ],
                "reference_problem_record_count": case[
                    "reference_problem_profile"
                ]["problem_record_count"],
                "records": case["records"],
                "search": search,
                "catalog_source_search": source_search,
                "catalog_source_pid": source_pid,
                "catalog_source_mods_url": catalog_source_metadata.get(
                    "url",
                    "",
                ),
                "catalog_source_mods_sha256": catalog_source_metadata.get(
                    "sha256",
                    "",
                ),
                "catalog_source_titles": catalog_source_metadata.get(
                    "titles",
                    [],
                ),
                "catalog_source_notes": catalog_source_metadata.get(
                    "notes",
                    [],
                ),
                "candidate_count": len(candidates),
                "strict_metadata_candidate_count": len(strict),
                "strict_metadata_candidates": strict,
                "layout_audit_candidate_count": len(layout_audit),
                "layout_audit_candidates_ranked": ranked_layout_audit,
                "preferred_layout_audit_pid": (
                    preferred["pid"] if preferred is not None else ""
                ),
                "preferred_layout_audit_selection_score": (
                    preferred["match_evidence"]["selection_score"]
                    if preferred is not None
                    else None
                ),
                "candidates": candidates,
                "training_authorized": False,
                "evaluation_authorized": False,
                "release_authorized": False,
            }
        )
    strict_pairs = [
        (case, candidate)
        for case in case_reports
        for candidate in case["strict_metadata_candidates"]
    ]
    layout_audit_pairs = [
        (case, candidate)
        for case in case_reports
        for candidate in case["layout_audit_candidates_ranked"]
    ]
    return {
        "format": 1,
        "created_at": utc_now_iso(),
        "role": ROLE,
        "quality_catalog_path": str(quality_catalog_path),
        "quality_catalog_sha256": sha256_file(quality_catalog_path),
        "cache_dir": str(cache_dir.resolve()),
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_authorized": False,
        "authorization_blockers": [
            "search_metadata_match_does_not_prove_exact_edition",
            "child_image_sequence_not_acquired",
            "all_page_alignment_not_verified",
            "absence_of_problem_comments_does_not_prove_completeness",
            "independent_double_annotation_not_started",
        ],
        "reference_candidate_count": len(case_reports),
        "searched_unique_object_count": len(metadata_by_pid),
        "searched_unique_parent_object_count": len(
            parent_metadata_by_pid
        ),
        "parent_metadata_fetch_failure_count": sum(
            bool(metadata.get("fetch_error"))
            for metadata in parent_metadata_by_pid.values()
        ),
        "strict_metadata_match_count": len(strict_pairs),
        "strict_metadata_matched_reference_count": len(
            {
                str(case["reference_path"])
                for case, _candidate in strict_pairs
            }
        ),
        "layout_audit_candidate_count": len(layout_audit_pairs),
        "layout_audit_matched_reference_count": len(
            {
                str(case["reference_path"])
                for case, _candidate in layout_audit_pairs
            }
        ),
        "preferred_layout_audit_candidate_count": sum(
            bool(case["preferred_layout_audit_pid"])
            for case in case_reports
        ),
        "cases": case_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("quality_catalog_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    catalog_path = args.quality_catalog_path.resolve()
    report = discover(
        json.loads(catalog_path.read_text(encoding="utf-8")),
        quality_catalog_path=catalog_path,
        timeout=args.timeout_seconds,
        workers=args.workers,
        cache_dir=(
            args.cache_dir.resolve()
            if args.cache_dir is not None
            else None
        ),
    )
    atomic_write_json(args.output_path.resolve(), report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "reference_candidate_count",
                    "searched_unique_object_count",
                    "strict_metadata_match_count",
                    "strict_metadata_matched_reference_count",
                    "layout_audit_candidate_count",
                    "layout_audit_matched_reference_count",
                    "preferred_layout_audit_candidate_count",
                    "training_authorized",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
