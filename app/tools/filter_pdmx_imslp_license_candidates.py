from __future__ import annotations

"""Join PDMX IMSLP-provenance candidates to the pinned license table."""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_ROOT / "src"))

from app.tools.acquire_pdmx_license_table import (  # noqa: E402
    EXPECTED_BYTES,
    EXPECTED_MD5,
    RECORD_ID,
    ROLE as LICENSE_ACQUISITION_ROLE,
    VERSION,
)
from app.tools.catalog_pdmx_imslp_provenance_candidates import (  # noqa: E402
    PUBLIC_DOMAIN_LICENSES,
    ROLE as CATALOG_ROLE,
)
from scorescan.util import atomic_write_json, sha256_file, utc_now_iso  # noqa: E402


ROLE = "pdmx_imslp_license_filtered_candidates_not_training"
METADATA_PATH = re.compile(r"\./metadata/[0-9]+/([0-9]+)\.json")
ARCHIVE_PATH = re.compile(
    r"\./(mxl|pdf)/[A-Za-z0-9_./=-]+\.(mxl|pdf)",
)
REQUIRED_COLUMNS = {
    "metadata",
    "mxl",
    "pdf",
    "license",
    "license_conflict",
    "subset:no_license_conflict",
    "subset:all_valid",
    "has_lyrics",
    "n_lyrics",
    "n_tracks",
    "tracks",
    "complexity",
    "n_notes",
    "n_annotations",
}
TRUE_VALUES = {"true", "1"}
FALSE_VALUES = {"false", "0"}
MISSING_VALUES = {"", "na", "n/a", "none", "nan"}


def _boolean(value: str, *, column: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"invalid boolean in PDMX column {column}: {value!r}")


def _archive_member(value: str, *, kind: str) -> str | None:
    normalized = value.strip()
    if normalized.casefold() in MISSING_VALUES:
        return None
    match = ARCHIVE_PATH.fullmatch(normalized)
    if match is None or match.group(1) != kind or match.group(2) != kind:
        raise ValueError(f"invalid PDMX {kind} archive path: {value!r}")
    return normalized.removeprefix("./")


def _validate_acquisition(
    csv_path: Path,
    acquisition_path: Path,
) -> dict[str, object]:
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    asset = acquisition.get("asset")
    if (
        acquisition.get("role") != LICENSE_ACQUISITION_ROLE
        or acquisition.get("record_id") != RECORD_ID
        or acquisition.get("version") != VERSION
        or acquisition.get("expected_bytes") != EXPECTED_BYTES
        or acquisition.get("expected_md5") != EXPECTED_MD5
        or acquisition.get("training_authorized") is not False
        or not isinstance(asset, dict)
        or asset.get("sha256") != sha256_file(csv_path)
        or csv_path.stat().st_size != EXPECTED_BYTES
    ):
        raise ValueError("unexpected PDMX license-table acquisition contract")
    return acquisition


def filter_candidates(
    catalog_path: Path,
    csv_path: Path,
    acquisition_path: Path,
    output_path: Path,
) -> dict[str, object]:
    _validate_acquisition(csv_path, acquisition_path)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    candidates = catalog.get("candidates")
    if (
        catalog.get("role") != CATALOG_ROLE
        or catalog.get("record_id") != RECORD_ID
        or catalog.get("version") != VERSION
        or catalog.get("training_authorized") is not False
        or not isinstance(candidates, list)
    ):
        raise ValueError("unexpected PDMX provenance catalog contract")
    by_id: dict[int, dict[str, object]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("invalid PDMX provenance candidate")
        score_id = candidate.get("score_id")
        if (
            isinstance(score_id, bool)
            or not isinstance(score_id, int)
            or score_id in by_id
        ):
            raise ValueError("duplicate or invalid PDMX provenance score id")
        by_id[score_id] = candidate

    matched: dict[int, dict[str, str]] = {}
    csv_row_count = 0
    duplicate_candidate_rows: set[int] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(
            reader.fieldnames,
        ):
            raise ValueError("PDMX.csv is missing required columns")
        for row in reader:
            csv_row_count += 1
            metadata_path = row.get("metadata", "")
            match = METADATA_PATH.fullmatch(metadata_path)
            if match is None:
                raise ValueError(
                    f"invalid PDMX metadata path at CSV row {csv_row_count + 1}"
                )
            score_id = int(match.group(1))
            if score_id not in by_id:
                continue
            if score_id in matched:
                duplicate_candidate_rows.add(score_id)
                continue
            matched[score_id] = row
    if duplicate_candidate_rows:
        raise ValueError("candidate score id occurs more than once in PDMX.csv")

    accepted: list[dict[str, object]] = []
    rejection_counts: Counter[str] = Counter()
    missing_csv_score_ids = sorted(set(by_id) - set(matched))
    for score_id, candidate in sorted(by_id.items()):
        row = matched.get(score_id)
        if row is None:
            rejection_counts["missing_csv_row"] += 1
            continue
        license_conflict = _boolean(
            row["license_conflict"],
            column="license_conflict",
        )
        no_conflict_subset = _boolean(
            row["subset:no_license_conflict"],
            column="subset:no_license_conflict",
        )
        all_valid = _boolean(
            row["subset:all_valid"],
            column="subset:all_valid",
        )
        has_lyrics = _boolean(row["has_lyrics"], column="has_lyrics")
        try:
            lyric_count = int(row["n_lyrics"])
        except ValueError as exc:
            raise ValueError("invalid n_lyrics in PDMX.csv") from exc
        mxl_member = _archive_member(row["mxl"], kind="mxl")
        pdf_member = _archive_member(row["pdf"], kind="pdf")
        reasons: list[str] = []
        if license_conflict or not no_conflict_subset:
            reasons.append("license_conflict")
        if (
            candidate.get("public_domain_metadata_consistent") is not True
            or row["license"].strip().casefold()
            not in PUBLIC_DOMAIN_LICENSES
        ):
            reasons.append("public_domain_metadata_inconsistent")
        if has_lyrics or lyric_count != 0:
            reasons.append("lyrics_present")
        if candidate.get("out_of_boundary_instrumentation_terms"):
            reasons.append("out_of_boundary_instrumentation")
        if not str(candidate.get("boundary_hint", "")).endswith("candidate"):
            reasons.append("out_of_boundary_shape")
        if mxl_member is None:
            reasons.append("mxl_missing")
        if not all_valid:
            reasons.append("not_all_valid")
        if reasons:
            rejection_counts.update(set(reasons))
            continue
        accepted_row = dict(candidate)
        accepted_row.update(
            {
                "pdmx_license_conflict_verified_false": True,
                "pdmx_no_license_conflict_subset_verified_true": True,
                "pdmx_all_valid_subset_verified_true": True,
                "pdmx_csv_license": row["license"],
                "pdmx_mxl_archive_member": mxl_member,
                # The PDMX PDF is a MuseScore render. It is retained only as
                # provenance metadata and is never a scan benchmark input.
                "pdmx_rendered_pdf_archive_member": pdf_member,
                "pdmx_has_lyrics": False,
                "pdmx_n_tracks": row["n_tracks"],
                "pdmx_tracks": row["tracks"],
                "pdmx_complexity": row["complexity"],
                "pdmx_n_notes": row["n_notes"],
                "pdmx_n_annotations": row["n_annotations"],
            }
        )
        accepted.append(accepted_row)

    boundary_counts = Counter(
        str(candidate["boundary_hint"]) for candidate in accepted
    )
    report = {
        "format": 1,
        "created_at": utc_now_iso(),
        "name": "PDMX IMSLP candidates after license and boundary filtering",
        "role": ROLE,
        "record_id": RECORD_ID,
        "version": VERSION,
        "catalog_path": str(catalog_path),
        "catalog_sha256": sha256_file(catalog_path),
        "license_table_path": str(csv_path),
        "license_table_sha256": sha256_file(csv_path),
        "license_acquisition_path": str(acquisition_path),
        "license_acquisition_sha256": sha256_file(acquisition_path),
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_authorized": False,
        "authorization_reason": (
            "license-filtered candidates still require pinned symbolic bytes, "
            "exact IMSLP scan bytes, exact-work identity, declared-boundary "
            "parsing, scan-to-symbolic alignment, and immutable work splits"
        ),
        "pdmx_rendered_pdf_is_scan_input": False,
        "csv_row_count": csv_row_count,
        "catalog_candidate_count": len(by_id),
        "matched_candidate_count": len(matched),
        "missing_csv_score_ids": missing_csv_score_ids,
        "accepted_candidate_count": len(accepted),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "boundary_hint_counts": dict(sorted(boundary_counts.items())),
        "unique_imslp_reverse_lookup_source_count": len(
            {
                source_id
                for candidate in accepted
                for source_id in candidate[
                    "imslp_reverse_lookup_source_ids"
                ]
            }
        ),
        "candidates": accepted,
    }
    atomic_write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog_path", type=Path)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("license_acquisition_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    report = filter_candidates(
        args.catalog_path.resolve(),
        args.csv_path.resolve(),
        args.license_acquisition_path.resolve(),
        args.output_path.resolve(),
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "csv_row_count",
                    "catalog_candidate_count",
                    "accepted_candidate_count",
                    "rejection_counts",
                    "boundary_hint_counts",
                    "unique_imslp_reverse_lookup_source_count",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
