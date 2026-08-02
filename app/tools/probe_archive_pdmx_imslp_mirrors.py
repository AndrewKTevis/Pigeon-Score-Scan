from __future__ import annotations

"""Find exact Internet Archive mirrors for verified PDMX/IMSLP pairs."""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Mapping

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_ROOT / "src"))

from app.tools.probe_archive_openscore_imslp_mirrors import (  # noqa: E402
    fetch_archive_metadata,
    probe_candidate,
    search_archive,
)
from app.tools.probe_pdmx_imslp_scan_sources import (  # noqa: E402
    ROLE as IMSLP_EVIDENCE_ROLE,
)
from scorescan.util import atomic_write_json, sha256_file, utc_now_iso  # noqa: E402


ROLE = "archive_pdmx_imslp_exact_filename_mirror_candidates_not_downloaded"


def _archive_candidates(
    evidence: Mapping[str, object],
) -> list[dict[str, object]]:
    verified = evidence.get("verified_candidates")
    if not isinstance(verified, list):
        raise ValueError("verified PDMX IMSLP candidates are missing")
    rows: list[dict[str, object]] = []
    seen_pairs: set[tuple[int, str]] = set()
    for candidate in verified:
        if not isinstance(candidate, dict):
            raise ValueError("invalid verified PDMX IMSLP candidate")
        score_id = candidate.get("score_id")
        if isinstance(score_id, bool) or not isinstance(score_id, int):
            raise ValueError("invalid PDMX score id")
        sources = candidate.get("imslp_source_evidence")
        if not isinstance(sources, list) or not sources:
            raise ValueError("verified candidate has no IMSLP source evidence")
        composer = str(candidate.get("artist_name", "")).strip()
        if not composer:
            composer = str(candidate.get("composer_name", "")).strip()
        title = str(candidate.get("title", "")).strip()
        if not title:
            title = str(candidate.get("file_score_title", "")).strip()
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError("invalid IMSLP source evidence")
            source_id = str(source.get("imslp_source_id", ""))
            pair = (score_id, source_id)
            if not source_id.isdigit() or pair in seen_pairs:
                raise ValueError("invalid or duplicate PDMX/IMSLP pair")
            seen_pairs.add(pair)
            rows.append(
                {
                    "score_id": score_id,
                    "work_fingerprint": (
                        f"pdmx-{evidence.get('version', '')}-{score_id}"
                    ),
                    "composer": composer,
                    "work_title": title,
                    "boundary_hint": candidate.get("boundary_hint", ""),
                    "pdmx_mxl_archive_member": candidate.get(
                        "pdmx_mxl_archive_member",
                        "",
                    ),
                    "imslp_source_id": source_id,
                    "direct_pdf_url": source.get("direct_pdf_url", ""),
                    "reported_source_page_count": source.get(
                        "page_count",
                        0,
                    ),
                    "source_identity_verified": True,
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            int(str(row["imslp_source_id"])),
            int(row["score_id"]),
        ),
    )


def build_report(
    evidence: Mapping[str, object],
    *,
    evidence_path: Path,
    timeout_seconds: float,
    minimum_interval_seconds: float,
    searcher: Callable[[str], list[dict[str, object]]] | None = None,
    metadata_fetcher: Callable[[str], dict[str, object]] | None = None,
) -> dict[str, object]:
    if (
        evidence.get("role") != IMSLP_EVIDENCE_ROLE
        or evidence.get("training_authorized") is not False
        or evidence.get("evaluation_authorized") is not False
        or evidence.get("release_authorized") is not False
    ):
        raise ValueError("unexpected PDMX IMSLP evidence contract")
    candidates = _archive_candidates(evidence)
    if not candidates:
        raise ValueError("no PDMX IMSLP mirror candidates")
    last_request = 0.0

    def wait_for_request() -> None:
        nonlocal last_request
        remaining = minimum_interval_seconds - (
            time.monotonic() - last_request
        )
        if remaining > 0:
            time.sleep(remaining)
        last_request = time.monotonic()

    if searcher is None:
        def searcher(query: str) -> list[dict[str, object]]:
            wait_for_request()
            return search_archive(
                query,
                timeout_seconds=timeout_seconds,
            )
    if metadata_fetcher is None:
        def metadata_fetcher(identifier: str) -> dict[str, object]:
            wait_for_request()
            return fetch_archive_metadata(
                identifier,
                timeout_seconds=timeout_seconds,
            )

    rows: list[dict[str, object]] = []
    for position, candidate in enumerate(candidates, start=1):
        rows.append(
            probe_candidate(
                candidate,
                searcher=searcher,
                metadata_fetcher=metadata_fetcher,
            )
        )
        print(
            f"[{position}/{len(candidates)}] PDMX archive mirror probed",
            flush=True,
        )
    exact = [
        row
        for row in rows
        if row.get("status") == "exact_archive_mirror_candidate"
    ]
    status_counts = Counter(str(row["status"]) for row in rows)
    return {
        "format": 1,
        "created_at": utc_now_iso(),
        "name": "PDMX IMSLP Internet Archive exact mirror evidence",
        "role": ROLE,
        "evidence_path": str(evidence_path),
        "evidence_sha256": sha256_file(evidence_path),
        "record_id": evidence.get("record_id"),
        "version": evidence.get("version"),
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_authorized": False,
        "authorization_reason": (
            "exact mirror candidates still require downloaded byte hashes, "
            "PDF page verification, pinned MXL bytes, parsed boundary checks, "
            "scan-to-score alignment, and immutable work-level splits"
        ),
        "candidate_count": len(rows),
        "exact_mirror_candidate_count": len(exact),
        "exact_mirror_work_count": len(
            {str(row["work_fingerprint"]) for row in exact}
        ),
        "reported_exact_mirror_pages": sum(
            int(row.get("reported_source_page_count", 0) or 0)
            for row in exact
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "exact_mirror_candidates": exact,
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
                    "reported_exact_mirror_pages",
                    "status_counts",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
