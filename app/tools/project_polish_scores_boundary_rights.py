from __future__ import annotations

"""Project cached NIFC rights evidence onto the expanded boundary catalog.

The authoritative MODS endpoints were already probed for the narrow catalog.
This projection avoids repeating network requests while binding every expanded
candidate to the exact cached metadata result.  It never upgrades authorization
and fails if any accepted candidate lacks cached source evidence.
"""

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.util import atomic_write_json, sha256_file, utc_now_iso  # noqa: E402


CATALOG_ROLE = "expanded_boundary_candidate_catalog_not_training_or_evaluation"
CACHE_ROLE = "scan_rights_evidence_only_not_training_or_evaluation"
ROLE = "expanded_boundary_scan_rights_projection_not_training_or_evaluation"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def project_rights(
    catalog: dict[str, object],
    *,
    catalog_path: Path,
    cached_report: dict[str, object],
    cached_report_path: Path,
) -> dict[str, object]:
    if catalog.get("role") != CATALOG_ROLE:
        raise ValueError("unexpected expanded Polish Scores catalog role")
    if catalog.get("training_authorized") is not False:
        raise ValueError("expanded catalog unexpectedly authorizes training")
    if cached_report.get("role") != CACHE_ROLE:
        raise ValueError("unexpected cached scan-rights report role")
    if cached_report.get("training_authorized") is not False:
        raise ValueError("cached rights report unexpectedly authorizes training")

    accepted = catalog.get("accepted_candidates")
    sources = cached_report.get("sources")
    if not isinstance(accepted, list) or not accepted:
        raise ValueError("expanded catalog has no accepted candidates")
    if not isinstance(sources, list) or not sources:
        raise ValueError("cached rights report has no sources")
    source_by_url: dict[str, dict[str, object]] = {}
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("cached rights source is invalid")
        url = str(source.get("pdf_url", "")).strip()
        if not url or url in source_by_url:
            raise ValueError("cached rights source URLs are missing or duplicated")
        metadata_hash = str(source.get("metadata_sha256", ""))
        if SHA256_PATTERN.fullmatch(metadata_hash) is None:
            raise ValueError(f"cached source has no metadata hash: {url}")
        source_by_url[url] = source

    candidate_rights: list[dict[str, object]] = []
    selected_urls: set[str] = set()
    missing_urls: list[str] = []
    for row in accepted:
        if not isinstance(row, dict):
            raise ValueError("expanded catalog candidate is invalid")
        url = str(row.get("scan_pdf_url", "")).strip()
        if not url or url not in source_by_url:
            missing_urls.append(url or "<missing>")
            continue
        selected_urls.add(url)
        source = source_by_url[url]
        candidate_rights.append(
            {
                "path": row.get("path", ""),
                "work_fingerprint": row.get("work_fingerprint", ""),
                "source_group_fingerprint": row.get(
                    "source_group_fingerprint",
                    "",
                ),
                "score_shape": (
                    row.get("boundary", {}).get("score_shape", "")
                    if isinstance(row.get("boundary"), dict)
                    else ""
                ),
                "scan_pdf_url": url,
                "rights_status": source.get("status", ""),
                "scan_asset_cc_by_4_verified": source.get(
                    "scan_asset_cc_by_4_verified",
                    False,
                ),
                "metadata_sha256": source.get("metadata_sha256", ""),
            }
        )
    if missing_urls:
        raise ValueError(
            "accepted candidates lack cached scan-rights evidence: "
            + ", ".join(sorted(set(missing_urls))[:5])
        )

    selected_sources = [
        source_by_url[url] for url in sorted(selected_urls)
    ]
    verified_sources = [
        source
        for source in selected_sources
        if source.get("scan_asset_cc_by_4_verified") is True
    ]
    verified_candidates = [
        row
        for row in candidate_rights
        if row["scan_asset_cc_by_4_verified"] is True
    ]
    return {
        "format": 2,
        "created_at": utc_now_iso(),
        "name": "Polish Scores expanded-boundary scan-rights projection",
        "role": ROLE,
        "catalog_path": str(catalog_path),
        "catalog_sha256": sha256_file(catalog_path),
        "catalog_revision": catalog.get("revision", ""),
        "cached_rights_report_path": str(cached_report_path),
        "cached_rights_report_sha256": sha256_file(cached_report_path),
        "cached_rights_created_at": cached_report.get("created_at", ""),
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_authorized": False,
        "authorization_blockers": [
            "no_candidate_has_explicit_scan_asset_cc_by_4",
            "physical_pages_not_acquired_or_aligned",
            "full_score_without_part_file_merge_unverified",
            "independent_double_annotation_not_started",
            "production_holdout_split_not_assigned",
        ],
        "accepted_candidate_count": len(accepted),
        "unique_candidate_scan_source_count": len(selected_sources),
        "verified_cc_by_4_source_count": len(verified_sources),
        "candidates_with_verified_scan_rights": len(verified_candidates),
        "candidate_source_group_count": len(
            {
                str(row["source_group_fingerprint"])
                for row in candidate_rights
            }
        ),
        "rights_policy": cached_report.get("rights_policy", ""),
        "candidate_rights": candidate_rights,
        "sources": selected_sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog_path", type=Path)
    parser.add_argument("cached_report_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    catalog_path = args.catalog_path.resolve()
    cached_report_path = args.cached_report_path.resolve()
    report = project_rights(
        json.loads(catalog_path.read_text(encoding="utf-8")),
        catalog_path=catalog_path,
        cached_report=json.loads(
            cached_report_path.read_text(encoding="utf-8")
        ),
        cached_report_path=cached_report_path,
    )
    atomic_write_json(args.output_path.resolve(), report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "accepted_candidate_count",
                    "unique_candidate_scan_source_count",
                    "candidate_source_group_count",
                    "verified_cc_by_4_source_count",
                    "candidates_with_verified_scan_rights",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
