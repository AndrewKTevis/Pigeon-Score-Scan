from __future__ import annotations

"""Probe IMSLP file evidence for license-filtered PDMX candidates."""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Mapping

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_ROOT / "src"))

from app.tools.filter_pdmx_imslp_license_candidates import (  # noqa: E402
    ROLE as FILTER_ROLE,
)
from app.tools.probe_openscore_imslp_scan_sources import (  # noqa: E402
    PoliteFetcher,
    candidate_identity_matches,
    probe_source,
)
from scorescan.util import atomic_write_json, sha256_file, utc_now_iso  # noqa: E402


ROLE = "pdmx_imslp_file_evidence_not_training_or_evaluation"


def _identity_candidate(candidate: Mapping[str, object]) -> dict[str, object]:
    composer = str(candidate.get("composer_name", "")).strip()
    if not composer:
        composer = str(candidate.get("artist_name", "")).strip()
    title = str(candidate.get("title", "")).strip()
    if not title:
        title = str(candidate.get("file_score_title", "")).strip()
    return {
        "composer": composer,
        "work_title": title,
    }


def _identity_candidates(
    candidate: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    composers = []
    for key in ("composer_name", "artist_name"):
        value = str(candidate.get(key, "")).strip()
        if value and value not in composers:
            composers.append(value)
    titles = []
    for key in ("title", "file_score_title"):
        value = str(candidate.get(key, "")).strip()
        if value and value not in titles:
            titles.append(value)
    return tuple(
        {"composer": composer, "work_title": title}
        for composer in composers
        for title in titles
    )


def build_report(
    filtered: dict[str, object],
    *,
    filtered_path: Path,
    workers: int,
    timeout_seconds: float,
    minimum_interval_seconds: float,
    fetcher_factory: Callable[
        [float],
        Callable[[str], tuple[bytes, str, Mapping[str, str]]],
    ]
    | None = None,
    prior_sources: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    candidates = filtered.get("candidates")
    if (
        filtered.get("role") != FILTER_ROLE
        or filtered.get("training_authorized") is not False
        or filtered.get("pdmx_rendered_pdf_is_scan_input") is not False
        or not isinstance(candidates, list)
        or not candidates
    ):
        raise ValueError("unexpected PDMX license-filtered candidate contract")
    source_ids = sorted(
        {
            str(source_id)
            for candidate in candidates
            if isinstance(candidate, dict)
            for source_id in candidate.get(
                "imslp_reverse_lookup_source_ids",
                [],
            )
            if str(source_id).isdigit()
        },
        key=int,
    )
    if not source_ids:
        raise ValueError("PDMX candidates have no IMSLP source ids")
    sources: list[dict[str, object]]
    if prior_sources is not None:
        if any(not isinstance(source, dict) for source in prior_sources):
            raise ValueError("offline IMSLP source evidence is invalid")
        sources = [dict(source) for source in prior_sources]
        prior_ids = [str(source.get("imslp_source_id", "")) for source in sources]
        if (
            len(prior_ids) != len(set(prior_ids))
            or set(prior_ids) != set(source_ids)
        ):
            raise ValueError(
                "offline IMSLP evidence does not exactly cover candidates"
            )
    else:
        if fetcher_factory is None:
            fetcher_factory = lambda timeout: PoliteFetcher(
                timeout_seconds=timeout,
                minimum_interval_seconds=minimum_interval_seconds,
            )
        fetcher = fetcher_factory(timeout_seconds)
        sources = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    probe_source,
                    source_id,
                    fetcher=fetcher,
                ): source_id
                for source_id in source_ids
            }
            for position, future in enumerate(
                as_completed(futures),
                start=1,
            ):
                sources.append(future.result())
                if position % 5 == 0 or position == len(futures):
                    print(
                        f"[{position}/{len(futures)}] IMSLP sources probed",
                        flush=True,
                    )
    sources.sort(key=lambda source: int(str(source["imslp_source_id"])))
    by_id = {
        str(source["imslp_source_id"]): source for source in sources
    }
    verified_candidates: list[dict[str, object]] = []
    identity_mismatch_candidate_count = 0
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("invalid PDMX filtered candidate")
        matching_sources: list[dict[str, object]] = []
        for source_id in candidate.get(
            "imslp_reverse_lookup_source_ids",
            [],
        ):
            source = by_id.get(str(source_id))
            if (
                source is None
                or source.get("verified_public_domain_printed_scan")
                is not True
            ):
                continue
            if any(
                candidate_identity_matches(identity, source)
                for identity in _identity_candidates(candidate)
            ):
                matching_sources.append(source)
        if not matching_sources:
            if any(
                by_id.get(str(source_id), {}).get(
                    "verified_public_domain_printed_scan",
                )
                is True
                for source_id in candidate.get(
                    "imslp_reverse_lookup_source_ids",
                    [],
                )
            ):
                identity_mismatch_candidate_count += 1
            continue
        enriched = dict(candidate)
        enriched.update(
            {
                "scan_source_identity_verified": True,
                "verified_imslp_source_ids": [
                    str(source["imslp_source_id"])
                    for source in matching_sources
                ],
                "imslp_source_evidence": [
                    {
                        key: source.get(key)
                        for key in (
                            "imslp_source_id",
                            "final_page_url",
                            "page_title",
                            "file_title",
                            "page_count",
                            "direct_pdf_url",
                            "page_sha256",
                            "public_domain_evidence",
                            "scan_attribution_text",
                        )
                    }
                    for source in matching_sources
                ],
            }
        )
        verified_candidates.append(enriched)
    verified_sources = [
        source
        for source in sources
        if source.get("verified_public_domain_printed_scan") is True
    ]
    return {
        "format": 1,
        "created_at": utc_now_iso(),
        "name": "PDMX IMSLP authoritative scan-source evidence",
        "role": ROLE,
        "filtered_path": str(filtered_path),
        "filtered_sha256": sha256_file(filtered_path),
        "record_id": filtered.get("record_id"),
        "version": filtered.get("version"),
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_authorized": False,
        "authorization_reason": (
            "IMSLP page evidence still does not prove acquired PDF bytes, "
            "pinned PDMX MXL bytes, exact scan-to-score alignment, declared "
            "boundary after parsing, or immutable work-level splits"
        ),
        "source_count": len(sources),
        "verified_source_count": len(verified_sources),
        "failed_source_count": sum(
            source.get("status") == "fetch_or_parse_failed"
            for source in sources
        ),
        "verified_candidate_count": len(verified_candidates),
        "identity_mismatch_candidate_count": (
            identity_mismatch_candidate_count
        ),
        "reported_verified_source_pages": sum(
            int(source.get("page_count", 0) or 0)
            for source in verified_sources
        ),
        "verified_candidates": verified_candidates,
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("filtered_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--minimum-request-interval-seconds",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--offline-source-evidence",
        type=Path,
        help="reuse a prior report's exact source rows without network access",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    if not 3.0 <= args.timeout_seconds <= 60.0:
        raise ValueError("timeout-seconds must be between 3 and 60")
    if not 0.0 <= args.minimum_request_interval_seconds <= 10.0:
        raise ValueError("minimum request interval must be between 0 and 10")
    filtered_path = args.filtered_path.resolve()
    filtered = json.loads(filtered_path.read_text(encoding="utf-8"))
    prior_sources = None
    if args.offline_source_evidence is not None:
        prior = json.loads(
            args.offline_source_evidence.resolve().read_text(
                encoding="utf-8",
            )
        )
        if (
            prior.get("role") != ROLE
            or prior.get("filtered_sha256") != sha256_file(filtered_path)
            or not isinstance(prior.get("sources"), list)
        ):
            raise ValueError("offline source evidence contract mismatch")
        prior_sources = prior["sources"]
    report = build_report(
        filtered,
        filtered_path=filtered_path,
        workers=args.workers,
        timeout_seconds=args.timeout_seconds,
        minimum_interval_seconds=args.minimum_request_interval_seconds,
        prior_sources=prior_sources,
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
                    "reported_verified_source_pages",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
