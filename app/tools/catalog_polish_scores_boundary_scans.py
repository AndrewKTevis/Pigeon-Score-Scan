from __future__ import annotations

"""Catalog Polish Scores scans that fit the complete frozen score boundary.

Rows remain discovery candidates.  No row becomes training or release evidence
until exact scan rights, physical page identity, full-score/part-file status,
page alignment, and independent annotation have been verified.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT / "src"))

from scorescan.util import atomic_write_json, sha256_file, utc_now_iso  # noqa: E402
from app.tools.catalog_polish_scores_release_scans import (  # noqa: E402
    COPYRIGHT_MARKER,
    LICENSE,
    LICENSE_MARKER,
    REPOSITORY,
    validate_corpus,
)
from app.tools.humdrum_boundary import (  # noqa: E402
    analyze_humdrum_boundary,
    reference_records,
)


ROLE = "expanded_boundary_candidate_catalog_not_training_or_evaluation"
CATALOG_FORMAT = 2
PRINTED_SCAN_MARKERS = (
    "/galeria/druki-muzyczne/",
    "chopinonline.ac.uk/cfeo/",
)
MANUSCRIPT_SCAN_MARKERS = (
    "/galeria/rekopisy/",
    "/manuscript",
    "/manuscripts",
)
FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{64}")


def _scan_url(value: str) -> str:
    return value.split(" @", 1)[0].strip()


def _scan_medium(scan_url: str) -> str:
    folded = scan_url.casefold()
    if any(marker in folded for marker in PRINTED_SCAN_MARKERS):
        return "explicit_printed"
    if any(marker in folded for marker in MANUSCRIPT_SCAN_MARKERS):
        return "explicit_manuscript"
    return "unverified"


def _fingerprint(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def _provenance_fingerprints(
    records: dict[str, str],
) -> tuple[str, str]:
    source_identity = (
        records.get("NIFC-rismSourceID", "").strip()
        or "|".join(
            (
                records.get("SMS-siglum", "").strip(),
                records.get("SMS-shelfmark", "").strip(),
            )
        )
    )
    source_group = _fingerprint(REPOSITORY, "physical-source", source_identity)
    work_identity = "|".join(
        (
            source_identity,
            records.get("SMS-shelfwork", "").strip(),
            records.get("COM", "").strip().casefold(),
            records.get("OTL", "").strip().casefold(),
        )
    )
    work = _fingerprint(REPOSITORY, "work", work_identity)
    if (
        FINGERPRINT_PATTERN.fullmatch(source_group) is None
        or FINGERPRINT_PATTERN.fullmatch(work) is None
    ):
        raise AssertionError("invalid Polish Scores provenance fingerprint")
    return source_group, work


def candidate_row(path: Path, corpus_root: Path) -> dict[str, object]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    records = reference_records(lines)
    instrumentation = records.get("AIN", "")
    boundary = analyze_humdrum_boundary(
        path,
        instrumentation=instrumentation,
        source_lines=lines,
    )
    scan_url = _scan_url(records.get("URL-scan", ""))
    scan_medium = _scan_medium(scan_url)
    reasons = list(boundary.reasons)
    if not scan_url:
        reasons.append("scan_url_missing")
    if scan_medium != "explicit_printed":
        reasons.append("scan_not_proved_printed")
    if records.get("YEM") != LICENSE_MARKER:
        reasons.append("transcription_license_not_exact_cc_by_4")
    if COPYRIGHT_MARKER not in records.get("YEC", ""):
        reasons.append("transcription_copyright_provenance_missing")
    if "-pc" in path.stem.casefold():
        reasons.append("post_correction_variant")
    required = ("COM", "OTL", "SMS-siglum", "SMS-shelfmark")
    if any(not records.get(key, "").strip() for key in required):
        reasons.append("insufficient_work_provenance")
    reasons = sorted(set(reasons))
    source_group, work = _provenance_fingerprints(records)
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "path": str(path.relative_to(corpus_root)).replace("\\", "/"),
        "sha256": sha256_file(path),
        "composer": records.get("COM", ""),
        "title": records.get("OTL", ""),
        "instrumentation": instrumentation,
        "genre": records.get("AGN", ""),
        "scan_url": scan_url,
        "scan_pdf_url": _scan_url(records.get("URL-pdf-islandora", "")),
        "scan_page_note": records.get("ONB-nifc", ""),
        "iiif_url": records.get("IIIF", ""),
        "scan_medium_status": scan_medium,
        "scan_asset_rights_status": "unverified",
        "scan_asset_rights_evidence": "",
        "full_score_without_part_file_merge_status": "unverified",
        "physical_page_alignment_status": "unverified",
        "independent_annotation_status": "not_started",
        "source_siglum": records.get("SMS-siglum", ""),
        "source_shelfmark": records.get("SMS-shelfmark", ""),
        "source_work": records.get("SMS-shelfwork", ""),
        "source_group_fingerprint": source_group,
        "work_fingerprint": work,
        "boundary": boundary.to_dict(),
        "transcription_license": records.get("YEM", ""),
        "transcription_copyright": records.get("YEC", ""),
    }


def catalog(corpus_root: Path, output_path: Path) -> dict[str, object]:
    corpus = validate_corpus(corpus_root)
    paths = sorted(corpus_root.glob("*/kern/*.krn"))
    if len(paths) < 8_000:
        raise ValueError("Polish Scores corpus checkout is incomplete")
    rows: list[dict[str, object]] = []
    rejection_counts: Counter[str] = Counter()
    shape_counts: Counter[str] = Counter()
    boundary_accepted_count = 0
    for position, path in enumerate(paths, start=1):
        row = candidate_row(path, corpus_root)
        rows.append(row)
        rejection_counts.update(str(reason) for reason in row["reasons"])
        boundary = row["boundary"]
        if not isinstance(boundary, dict):
            raise AssertionError("candidate boundary record is missing")
        shape_counts[str(boundary["score_shape"])] += 1
        boundary_accepted_count += int(bool(boundary["accepted"]))
        if position % 500 == 0 or position == len(paths):
            print(f"[{position}/{len(paths)}] boundary-cataloged", flush=True)

    accepted = [row for row in rows if row["accepted"]]
    report = {
        "format": CATALOG_FORMAT,
        "created_at": utc_now_iso(),
        "name": "Polish Scores frozen-boundary printed-scan candidates",
        "role": ROLE,
        "repository": REPOSITORY,
        "revision": corpus["revision"],
        "repository_license": LICENSE,
        "repository_license_sha256": corpus["license_sha256"],
        "release_authorized": False,
        "training_authorized": False,
        "evaluation_authorized": False,
        "authorization_blockers": [
            "object_level_scan_rights_unverified",
            "physical_page_identity_and_alignment_unverified",
            "full_score_without_part_file_merge_unverified",
            "independent_double_annotation_not_started",
            "production_holdout_split_not_assigned",
        ],
        "case_count": len(rows),
        "boundary_accepted_case_count": boundary_accepted_count,
        "accepted_candidate_count": len(accepted),
        "accepted_work_count": len(
            {str(row["work_fingerprint"]) for row in accepted}
        ),
        "accepted_source_group_count": len(
            {str(row["source_group_fingerprint"]) for row in accepted}
        ),
        "score_shape_counts": dict(sorted(shape_counts.items())),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "accepted_candidates": accepted,
        "cases": rows,
    }
    atomic_write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_root", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    report = catalog(args.corpus_root.resolve(), args.output_path.resolve())
    print(
        json.dumps(
            {
                "case_count": report["case_count"],
                "boundary_accepted_case_count": report[
                    "boundary_accepted_case_count"
                ],
                "accepted_candidate_count": report[
                    "accepted_candidate_count"
                ],
                "accepted_work_count": report["accepted_work_count"],
                "accepted_source_group_count": report[
                    "accepted_source_group_count"
                ],
                "score_shape_counts": report["score_shape_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
