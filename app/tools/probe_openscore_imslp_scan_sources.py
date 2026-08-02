from __future__ import annotations

"""Probe exact IMSLP files linked by fixed-revision OpenScore quartets."""

import argparse
import hashlib
import json
import re
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Mapping

from lxml import html as lxml_html

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.util import atomic_write_json, sha256_file, utc_now_iso  # noqa: E402


ROLE = "imslp_file_evidence_not_training_or_evaluation"
USER_AGENT = "ScoreScan-OMR-IMSLP-source-probe/1"
MAX_PAGE_BYTES = 8 * 1024 * 1024
PAGE_COUNT_PATTERN = re.compile(r"\b([0-9]+)\s+pp\.", re.IGNORECASE)
SIZE_PATTERN = re.compile(r"#0*[0-9]+\s*-\s*([^,]+),", re.IGNORECASE)


def _classes(element) -> set[str]:
    return set((element.get("class") or "").split())


def _identity_tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", value).casefold()
    return set(re.findall(r"[a-z]+|[0-9]+", normalized))


def _opus_numbers(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", value).casefold()
    return set(
        re.findall(
            r"\bop(?:us)?\.?\s*([0-9]+)\b",
            normalized,
        )
    )


def _page_title_from_final_url(final_url: str) -> str:
    parsed = urllib.parse.urlparse(final_url)
    if parsed.scheme != "https" or parsed.hostname != "imslp.org":
        return ""
    marker = "/wiki/"
    if marker not in parsed.path:
        return ""
    encoded_title = parsed.path.split(marker, 1)[1]
    return urllib.parse.unquote(encoded_title).replace("_", " ").strip()


def candidate_identity_matches(
    candidate: Mapping[str, object],
    source: Mapping[str, object],
) -> bool:
    page_title = str(source.get("page_title", ""))
    page_tokens = _identity_tokens(page_title)
    composer_tokens = {
        token
        for token in _identity_tokens(str(candidate.get("composer", "")))
        if len(token) >= 4
        and not token.isdigit()
    }
    title_tokens = _identity_tokens(str(candidate.get("work_title", "")))
    matching_composer_tokens = composer_tokens & page_tokens
    minimum_composer_matches = (len(composer_tokens) // 2) + 1
    if (
        not composer_tokens
        or len(matching_composer_tokens) < minimum_composer_matches
    ):
        return False
    # Translated work titles can share no lexical tokens (for example,
    # "Variations on a Waltz" versus "Veränderungen über einen Walzer").
    # A matching explicit opus number plus a strict composer majority is an
    # auditable work-level identity signal; arbitrary numbers are not.
    candidate_opus_numbers = _opus_numbers(
        str(candidate.get("work_title", ""))
    )
    page_opus_numbers = _opus_numbers(page_title)
    if candidate_opus_numbers & page_opus_numbers:
        return True
    if "quartet" in title_tokens:
        if not ({"quartet", "quartett"} & page_tokens):
            return False
    else:
        generic_title_tokens = {
            "flat",
            "major",
            "minor",
            "number",
            "opus",
            "the",
        }
        distinctive_title_tokens = {
            token
            for token in title_tokens
            if len(token) >= 3
            and not token.isdigit()
            and token not in generic_title_tokens
        }
        if len(distinctive_title_tokens & page_tokens) < min(
            2,
            len(distinctive_title_tokens),
        ):
            return False
    candidate_numbers = {token for token in title_tokens if token.isdigit()}
    page_numbers = {token for token in page_tokens if token.isdigit()}
    return not candidate_numbers or bool(candidate_numbers & page_numbers)


def parse_source_page(
    payload: bytes,
    *,
    imslp_source_id: str,
    final_url: str,
) -> dict[str, object]:
    if len(payload) > MAX_PAGE_BYTES:
        raise ValueError("IMSLP source page exceeds the safety limit")
    document = lxml_html.fromstring(payload, base_url=final_url)
    page_titles = document.xpath("//title[1]/text()")
    page_title = " ".join(page_titles[0].split()) if page_titles else ""
    if page_title.endswith(" - IMSLP"):
        page_title = page_title[: -len(" - IMSLP")].strip()
    normalized_id = str(int(imslp_source_id))
    element = None
    for candidate_id in (
        f"IMSLP{imslp_source_id}",
        f"IMSLP{normalized_id}",
    ):
        try:
            element = document.get_element_by_id(candidate_id)
            break
        except KeyError:
            continue
    if element is None:
        raise ValueError("exact IMSLP file block is missing")
    group = next(
        (
            ancestor
            for ancestor in element.iterancestors()
            if "we" in _classes(ancestor)
        ),
        None,
    )
    if group is None:
        raise ValueError("IMSLP edition group is missing")
    file_text = " ".join(element.text_content().split())
    group_text = " ".join(group.text_content().split())
    hrefs = [str(value) for value in element.xpath(".//a/@href")]
    pdf_paths = sorted(
        {
            href
            for href in hrefs
            if href.startswith("/images/")
            and ".pdf" in href.casefold()
        }
    )
    rights_links = [
        str(value)
        for value in group.xpath(".//a/@href")
        if "Public_Domain" in str(value)
    ]
    public_domain = bool(rights_links) and "Copyright Public Domain" in group_text
    printed_scan = (
        "PDF scanned by" in file_text
        and "manuscript" not in file_text.casefold()
        and "manuscript" not in group_text.casefold()
    )
    direct_pdf = pdf_paths[0] if len(pdf_paths) == 1 else ""
    page_match = PAGE_COUNT_PATTERN.search(file_text)
    size_match = SIZE_PATTERN.search(file_text)
    title_nodes = element.xpath(".//div[contains(@class, 'we_file_download')]//b[1]")
    title = (
        " ".join(title_nodes[0].text_content().split())
        if title_nodes
        else ""
    )
    verified = public_domain and printed_scan and bool(direct_pdf) and page_match is not None
    reasons: list[str] = []
    if not public_domain:
        reasons.append("file_group_not_explicit_public_domain")
    if not printed_scan:
        reasons.append("file_not_explicitly_printed_pdf_scan")
    if len(pdf_paths) != 1:
        reasons.append("exact_direct_pdf_path_missing_or_ambiguous")
    if page_match is None:
        reasons.append("page_count_missing")
    return {
        "imslp_source_id": imslp_source_id,
        "reverse_lookup_url": (
            f"https://imslp.org/wiki/Special:ReverseLookup/{imslp_source_id}"
        ),
        "final_page_url": final_url,
        "page_title": page_title,
        "status": (
            "verified_public_domain_printed_scan"
            if verified
            else "rejected_source_evidence"
        ),
        "verified_public_domain_printed_scan": verified,
        "reasons": reasons,
        "file_title": title,
        "file_size_label": size_match.group(1).strip() if size_match else "",
        "page_count": int(page_match.group(1)) if page_match else 0,
        "scan_attribution_text": (
            file_text[file_text.index("PDF scanned by") :]
            if "PDF scanned by" in file_text
            else ""
        ),
        "public_domain_evidence": (
            "Copyright Public Domain" if public_domain else ""
        ),
        "direct_pdf_path": direct_pdf,
        "direct_pdf_url": (
            urllib.parse.urljoin("https://imslp.org", direct_pdf)
            if direct_pdf
            else ""
        ),
    }


def fetch_page(
    url: str,
    *,
    timeout_seconds: float,
    attempts: int = 3,
) -> tuple[bytes, str, Mapping[str, str]]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "text/html,application/xhtml+xml;q=0.9",
                "User-Agent": USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                final_url = response.geturl()
                parsed = urllib.parse.urlparse(final_url)
                if parsed.scheme != "https" or parsed.hostname != "imslp.org":
                    raise ValueError("IMSLP redirect left the approved host")
                length = response.headers.get("Content-Length")
                if length and int(length) > MAX_PAGE_BYTES:
                    raise ValueError("IMSLP source page exceeds the safety limit")
                payload = response.read(MAX_PAGE_BYTES + 1)
                if len(payload) > MAX_PAGE_BYTES:
                    raise ValueError("IMSLP source page exceeds the safety limit")
                return payload, final_url, {
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


class PoliteFetcher:
    def __init__(
        self,
        *,
        timeout_seconds: float,
        minimum_interval_seconds: float,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.minimum_interval_seconds = minimum_interval_seconds
        self._lock = threading.Lock()
        self._last_request = 0.0

    def __call__(
        self,
        url: str,
    ) -> tuple[bytes, str, Mapping[str, str]]:
        with self._lock:
            remaining = (
                self.minimum_interval_seconds
                - (time.monotonic() - self._last_request)
            )
            if remaining > 0:
                time.sleep(remaining)
            self._last_request = time.monotonic()
        return fetch_page(url, timeout_seconds=self.timeout_seconds)


def probe_source(
    imslp_source_id: str,
    *,
    fetcher: Callable[
        [str], tuple[bytes, str, Mapping[str, str]]
    ],
) -> dict[str, object]:
    reverse_url = (
        f"https://imslp.org/wiki/Special:ReverseLookup/{imslp_source_id}"
    )
    try:
        payload, final_url, headers = fetcher(reverse_url)
        source = parse_source_page(
            payload,
            imslp_source_id=imslp_source_id,
            final_url=final_url,
        )
    except (
        OSError,
        TimeoutError,
        urllib.error.URLError,
        ValueError,
    ) as error:
        return {
            "imslp_source_id": imslp_source_id,
            "reverse_lookup_url": reverse_url,
            "status": "fetch_or_parse_failed",
            "verified_public_domain_printed_scan": False,
            "reasons": ["fetch_or_parse_failed"],
            "error": f"{type(error).__name__}: {error}",
        }
    source.update(
        {
            "page_sha256": hashlib.sha256(payload).hexdigest(),
            "page_content_type": headers.get("content_type", ""),
            "page_etag": headers.get("etag", ""),
            "page_last_modified": headers.get("last_modified", ""),
            "error": "",
        }
    )
    return source


def build_report(
    catalog: dict[str, object],
    *,
    catalog_path: Path,
    workers: int,
    timeout_seconds: float,
    minimum_interval_seconds: float = 0.0,
    prior_sources: list[dict[str, object]] | None = None,
    retry_failed_sources: bool = True,
    fetcher_factory: Callable[
        [float],
        Callable[[str], tuple[bytes, str, Mapping[str, str]]],
    ]
    | None = None,
) -> dict[str, object]:
    if catalog.get("role") != (
        "imslp_scan_candidate_catalog_not_training_or_evaluation"
    ):
        raise ValueError("unexpected OpenScore IMSLP catalog role")
    if catalog.get("training_authorized") is not False:
        raise ValueError("source catalog unexpectedly authorizes training")
    candidates = catalog.get("accepted_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("OpenScore IMSLP candidates are missing")
    source_ids = sorted(
        {
            str(row.get("imslp_source_id", ""))
            for row in candidates
            if isinstance(row, dict)
            and str(row.get("imslp_source_id", "")).isdigit()
        },
        key=int,
    )
    reusable: dict[str, dict[str, object]] = {}
    for source in prior_sources or []:
        if not isinstance(source, dict):
            continue
        if (
            retry_failed_sources
            and source.get("status") == "fetch_or_parse_failed"
        ):
            continue
        reusable_source = dict(source)
        if (
            reusable_source.get("status") != "fetch_or_parse_failed"
            and not str(reusable_source.get("page_title", "")).strip()
        ):
            derived_title = _page_title_from_final_url(
                str(reusable_source.get("final_page_url", ""))
            )
            if not derived_title:
                continue
            reusable_source["page_title"] = derived_title
            reusable_source["page_title_source"] = "final_redirect_url"
        reusable[str(reusable_source.get("imslp_source_id", ""))] = (
            reusable_source
        )
    sources: list[dict[str, object]] = [
        reusable[source_id]
        for source_id in source_ids
        if source_id in reusable
    ]
    pending_ids = [
        source_id for source_id in source_ids if source_id not in reusable
    ]
    if pending_ids:
        if fetcher_factory is None:
            fetcher_factory = lambda timeout: PoliteFetcher(
                timeout_seconds=timeout,
                minimum_interval_seconds=minimum_interval_seconds,
            )
        fetcher = fetcher_factory(timeout_seconds)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    probe_source,
                    source_id,
                    fetcher=fetcher,
                ): source_id
                for source_id in pending_ids
            }
            for position, future in enumerate(as_completed(futures), start=1):
                sources.append(future.result())
                if position % 20 == 0 or position == len(futures):
                    print(
                        f"[{position}/{len(futures)}] pending IMSLP sources "
                        f"probed ({len(reusable)} reused)",
                        flush=True,
                    )
    sources.sort(key=lambda source: int(str(source["imslp_source_id"])))
    by_id = {str(source["imslp_source_id"]): source for source in sources}
    verified_candidates: list[dict[str, object]] = []
    identity_mismatch_count = 0
    for row in candidates:
        source = by_id.get(str(row.get("imslp_source_id", "")), {})
        if source.get("verified_public_domain_printed_scan") is not True:
            continue
        identity_matches = candidate_identity_matches(row, source)
        if not identity_matches:
            identity_mismatch_count += 1
            continue
        enriched = dict(row)
        enriched["imslp_page_title"] = source.get("page_title", "")
        enriched["source_identity_verified"] = True
        enriched["direct_pdf_url"] = source.get("direct_pdf_url", "")
        enriched["reported_source_page_count"] = source.get("page_count", 0)
        verified_candidates.append(enriched)
    verified_sources = [
        source
        for source in sources
        if source.get("verified_public_domain_printed_scan") is True
    ]
    failed = [
        source
        for source in sources
        if source.get("status") == "fetch_or_parse_failed"
    ]
    return {
        "format": 1,
        "created_at": utc_now_iso(),
        "name": "OpenScore quartet authoritative IMSLP scan evidence",
        "role": ROLE,
        "catalog_path": str(catalog_path),
        "catalog_sha256": sha256_file(catalog_path),
        "catalog_revision": catalog.get("revision", ""),
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_authorized": False,
        "authorization_reason": (
            "file metadata evidence does not prove PDF byte integrity, exact "
            "page-to-score alignment, production-boundary validity, or split "
            "isolation"
        ),
        "source_count": len(sources),
        "verified_source_count": len(verified_sources),
        "failed_source_count": len(failed),
        "verified_candidate_count": len(verified_candidates),
        "identity_mismatch_candidate_count": identity_mismatch_count,
        "verified_work_count": len(
            {str(row["work_fingerprint"]) for row in verified_candidates}
        ),
        "reported_verified_source_pages": sum(
            int(source["page_count"]) for source in verified_sources
        ),
        "verified_candidates": verified_candidates,
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--minimum-request-interval-seconds", type=float, default=1.0)
    parser.add_argument("--resume-evidence", type=Path)
    parser.add_argument(
        "--offline-reclassify",
        action="store_true",
        help="reuse every prior source row without issuing network requests",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        raise ValueError("workers must be between 1 and 12")
    if not 3.0 <= args.timeout_seconds <= 60.0:
        raise ValueError("timeout-seconds must be between 3 and 60")
    if not 0.0 <= args.minimum_request_interval_seconds <= 10.0:
        raise ValueError("minimum request interval must be between 0 and 10 seconds")
    catalog_path = args.catalog_path.resolve()
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    prior_sources = None
    if args.resume_evidence is not None:
        prior = json.loads(
            args.resume_evidence.resolve().read_text(encoding="utf-8")
        )
        if prior.get("role") != ROLE:
            raise ValueError("resume evidence has an unexpected role")
        if prior.get("catalog_sha256") != sha256_file(catalog_path):
            raise ValueError("resume evidence belongs to a different catalog")
        prior_sources = prior.get("sources")
        if not isinstance(prior_sources, list):
            raise ValueError("resume evidence sources are missing")
    if args.offline_reclassify:
        if prior_sources is None:
            raise ValueError("offline reclassification requires resume evidence")
        candidate_ids = {
            str(row.get("imslp_source_id", ""))
            for row in catalog.get("accepted_candidates", [])
            if isinstance(row, dict)
            and str(row.get("imslp_source_id", "")).isdigit()
        }
        prior_ids = {
            str(row.get("imslp_source_id", ""))
            for row in prior_sources
            if isinstance(row, dict)
        }
        if candidate_ids != prior_ids:
            raise ValueError(
                "offline reclassification evidence is incomplete"
            )
    report = build_report(
        catalog,
        catalog_path=catalog_path,
        workers=args.workers,
        timeout_seconds=args.timeout_seconds,
        minimum_interval_seconds=args.minimum_request_interval_seconds,
        prior_sources=prior_sources,
        retry_failed_sources=not args.offline_reclassify,
    )
    atomic_write_json(args.output_path.resolve(), report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "source_count",
                    "verified_source_count",
                    "failed_source_count",
                    "verified_candidate_count",
                    "identity_mismatch_candidate_count",
                    "verified_work_count",
                    "reported_verified_source_pages",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
