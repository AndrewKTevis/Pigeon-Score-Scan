from __future__ import annotations

"""Catalog fixed-revision OpenScore quartets with exact IMSLP scan IDs.

This stage validates the CC0 transcription source and structural fit only.
The linked IMSLP asset remains a candidate until its exact file block proves
that it is a scanned, public-domain full score.
"""

import argparse
import hashlib
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.util import atomic_write_json, sha256_file, utc_now_iso  # noqa: E402


REVISION = "d13289cd70797da94646e5cf64f7296a4c4fee40"
ARCHIVE_SHA256 = "f331e31d1a6700c6bed4f7703f4732fd27c288d2f03d187afb1ed58ad06cedbe"
LICENSE_SHA256 = "a2010f343487d3f7618affe54f789f5487602331c0a8d03f49e9a7c547cf0499"
ROLE = "imslp_scan_candidate_catalog_not_training_or_evaluation"
HEAD_BYTES = 768 * 1024
REVERSE_LOOKUP_PATTERN = re.compile(
    r"Special:ReverseLookup/([0-9]+(?:-[0-9]+)?)",
    re.IGNORECASE,
)
META_PATTERN = re.compile(
    r'<metaTag\s+name="([^"]+)">(.*?)</metaTag>',
    re.IGNORECASE | re.DOTALL,
)


def _metadata(head: str) -> dict[str, str]:
    return {
        key: " ".join(html.unescape(re.sub(r"<[^>]+>", "", value)).split())
        for key, value in META_PATTERN.findall(head)
    }


def _fingerprint(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def catalog_row(path: Path, score_root: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        head = stream.read(HEAD_BYTES).decode("utf-8-sig", errors="replace")
    metadata = _metadata(head)
    raw_ids = sorted(set(REVERSE_LOOKUP_PATTERN.findall(head)))
    part_count = len(re.findall(r"<Part(?:\s[^>]*)?>", head))
    reasons: list[str] = []
    if len(raw_ids) != 1:
        reasons.append("not_exactly_one_imslp_source_id")
    elif not raw_ids[0].isdigit():
        reasons.append("non_numeric_imslp_source_id")
    if part_count != 4:
        reasons.append("not_exactly_four_score_parts")
    if metadata.get("copyright") != "OpenScore (CC0)":
        reasons.append("score_metadata_not_exact_openscore_cc0")
    arranger = metadata.get("arranger", "")
    if "manuscript" in arranger.casefold():
        reasons.append("source_described_as_manuscript")
    if not metadata.get("composer") or not metadata.get("workTitle"):
        reasons.append("work_metadata_missing")
    imslp_id = raw_ids[0] if len(raw_ids) == 1 else ""
    relative_path = str(path.relative_to(score_root)).replace("\\", "/")
    work_fingerprint = _fingerprint(
        "OpenScore/StringQuartets",
        REVISION,
        relative_path.rsplit("/", 1)[0],
        imslp_id,
    )
    source_fingerprint = _fingerprint("IMSLP", "file", imslp_id)
    return {
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "path": relative_path,
        "sha256": sha256_file(path),
        "composer": metadata.get("composer", ""),
        "work_title": metadata.get("workTitle", ""),
        "movement_title": metadata.get("movementTitle", ""),
        "arranger_provenance": arranger,
        "score_license_marker": metadata.get("copyright", ""),
        "part_count": part_count,
        "boundary_configuration": "monophonic_ensemble",
        "imslp_source_id": imslp_id,
        "imslp_reverse_lookup_url": (
            f"https://imslp.org/wiki/Special:ReverseLookup/{imslp_id}"
            if imslp_id
            else ""
        ),
        "scan_asset_status": "unverified",
        "source_group_fingerprint": source_fingerprint,
        "work_fingerprint": work_fingerprint,
    }


def validate_source(
    corpus_root: Path,
    provenance_path: Path,
) -> tuple[Path, Path]:
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assets = provenance.get("assets")
    if not isinstance(assets, list):
        raise ValueError("external provenance assets are missing")
    matches = [
        asset
        for asset in assets
        if isinstance(asset, dict)
        and asset.get("key") == "openscore_string_quartets"
    ]
    if len(matches) != 1:
        raise ValueError("OpenScore quartet provenance is ambiguous")
    asset = matches[0]
    expected = {
        "archive_sha256": ARCHIVE_SHA256,
        "license": "CC0-1.0",
        "license_review_required": False,
        "extracted_directory": "openscore_string_quartets_d13289cd",
    }
    for key, value in expected.items():
        if asset.get(key) != value:
            raise ValueError(f"OpenScore provenance contract changed: {key}")
    roots = list(corpus_root.glob(f"StringQuartets-{REVISION}"))
    if len(roots) != 1:
        raise ValueError("fixed OpenScore quartet revision is missing")
    score_root = roots[0]
    license_path = score_root / "LICENSE.txt"
    if sha256_file(license_path) != LICENSE_SHA256:
        raise ValueError("OpenScore quartet CC0 license changed")
    return score_root, license_path


def catalog(
    corpus_root: Path,
    provenance_path: Path,
    output_path: Path,
) -> dict[str, object]:
    score_root, license_path = validate_source(corpus_root, provenance_path)
    score_paths = sorted((score_root / "scores").rglob("*.mscx"))
    if len(score_paths) < 100:
        raise ValueError("OpenScore quartet checkout is incomplete")
    rows: list[dict[str, object]] = []
    rejection_counts: Counter[str] = Counter()
    for position, path in enumerate(score_paths, start=1):
        row = catalog_row(path, score_root)
        rows.append(row)
        rejection_counts.update(str(reason) for reason in row["reasons"])
        if position % 25 == 0 or position == len(score_paths):
            print(f"[{position}/{len(score_paths)}] quartet scores cataloged", flush=True)
    accepted = [row for row in rows if row["accepted"]]
    report = {
        "format": 1,
        "created_at": utc_now_iso(),
        "name": "OpenScore quartet IMSLP scan candidate catalog",
        "role": ROLE,
        "revision": REVISION,
        "archive_sha256": ARCHIVE_SHA256,
        "score_license": "CC0-1.0",
        "license_path": str(license_path),
        "license_sha256": LICENSE_SHA256,
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_authorized": False,
        "authorization_reason": (
            "IMSLP file-level public-domain and printed-scan status, exact PDF "
            "bytes, page alignment, boundary validation, and split isolation "
            "remain unverified"
        ),
        "case_count": len(rows),
        "accepted_candidate_count": len(accepted),
        "accepted_work_count": len(
            {str(row["work_fingerprint"]) for row in accepted}
        ),
        "accepted_source_count": len(
            {str(row["imslp_source_id"]) for row in accepted}
        ),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "accepted_candidates": accepted,
        "cases": rows,
    }
    atomic_write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_root", type=Path)
    parser.add_argument("provenance_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    report = catalog(
        args.corpus_root.resolve(),
        args.provenance_path.resolve(),
        args.output_path.resolve(),
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "case_count",
                    "accepted_candidate_count",
                    "accepted_work_count",
                    "accepted_source_count",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
