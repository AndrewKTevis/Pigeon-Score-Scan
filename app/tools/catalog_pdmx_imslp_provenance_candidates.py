from __future__ import annotations

"""Catalog PDMX public-domain scores that explicitly cite an IMSLP source."""

import argparse
import json
import re
import sys
import tarfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_ROOT / "src"))

from scorescan.util import atomic_write_json, sha256_file, utc_now_iso  # noqa: E402

from app.tools.acquire_pdmx_metadata_archive import (  # noqa: E402
    EXPECTED_BYTES,
    EXPECTED_MD5,
    RECORD_ID,
    ROLE as ACQUISITION_ROLE,
    VERSION,
)


ROLE = "pdmx_imslp_provenance_candidates_not_identity_or_license_verified"
MEMBER_NAME = re.compile(r"metadata/[0-9]+/([0-9]+)\.json")
REVERSE_LOOKUP = re.compile(
    r"imslp\.org/wiki/Special:ReverseLookup/([0-9]{4,7})",
    re.IGNORECASE,
)
DIRECT_PDF = re.compile(
    r"https://(?:www\.)?imslp\.org/images/[^\s\"'<>]+?\.pdf",
    re.IGNORECASE,
)
MAXIMUM_MEMBER_BYTES = 2 * 1024 * 1024
MAXIMUM_PROVENANCE_STRINGS = 20
OUT_OF_BOUNDARY_TERMS = {
    "accordion",
    "choir",
    "choral",
    "drum",
    "guitar",
    "lyrics",
    "percussion",
    "tablature",
    "vocal",
    "voice",
}
KEYBOARD_TERMS = {"piano", "pianoforte", "keyboard", "organ"}
PUBLIC_DOMAIN_LICENSES = {"cc-zero", "publicdomain"}


def _strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def _score_row(payload: Mapping[str, object]) -> dict[str, object] | None:
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    score = data.get("score")
    if not isinstance(score, dict):
        return None
    score_id = score.get("id")
    if isinstance(score_id, bool) or not isinstance(score_id, int):
        return None
    provenance = []
    for value in _strings(payload):
        if "imslp" in value.casefold() and value not in provenance:
            provenance.append(value)
            if len(provenance) >= MAXIMUM_PROVENANCE_STRINGS:
                break
    if not provenance:
        return None
    combined = "\n".join(provenance)
    source_ids = sorted(set(REVERSE_LOOKUP.findall(combined)), key=int)
    direct_urls = sorted(set(DIRECT_PDF.findall(combined)))
    if not source_ids and not direct_urls:
        return None
    public_domain = bool(
        str(score.get("license", "")).casefold()
        in PUBLIC_DOMAIN_LICENSES
        and score.get("is_public_domain") is True
        and data.get("is_public_domain") is True
    )
    parts = score.get("parts")
    part_count = (
        int(parts)
        if isinstance(parts, int) and not isinstance(parts, bool)
        else 0
    )
    instrumentation_values = [
        score.get("parts_names"),
        score.get("instruments"),
        score.get("instrumentations"),
    ]
    instrumentation_text = " ".join(
        value
        for item in instrumentation_values
        for value in _strings(item)
    )
    tokens = {
        token
        for token in re.findall(
            r"[a-z]+",
            instrumentation_text.casefold(),
        )
    }
    out_terms = sorted(tokens & OUT_OF_BOUNDARY_TERMS)
    keyboard = bool(tokens & KEYBOARD_TERMS)
    if out_terms:
        boundary_hint = "out_of_boundary_instrumentation_term"
    elif not 1 <= part_count <= 16:
        boundary_hint = "part_count_outside_product_limit"
    elif keyboard and part_count > 1:
        boundary_hint = "keyboard_plus_single_staff_ensemble_candidate"
    elif keyboard:
        boundary_hint = "keyboard_candidate"
    elif part_count > 1:
        boundary_hint = "single_staff_ensemble_candidate"
    else:
        boundary_hint = "single_staff_solo_candidate"
    return {
        "score_id": score_id,
        "musescore_url": score.get("url", ""),
        "title": score.get("title", ""),
        "file_score_title": score.get("file_score_title", ""),
        "subtitle": score.get("subtitle", ""),
        "composer_name": score.get("composer_name", ""),
        "artist_name": score.get("artist_name", ""),
        "license": score.get("license", ""),
        "score_is_public_domain": score.get("is_public_domain"),
        "data_is_public_domain": data.get("is_public_domain"),
        "public_domain_metadata_consistent": public_domain,
        "part_count": part_count,
        "parts_names": score.get("parts_names"),
        "instruments": score.get("instruments"),
        "pages_count": score.get("pages_count"),
        "measures": score.get("measures"),
        "imslp_reverse_lookup_source_ids": source_ids,
        "imslp_direct_pdf_urls": direct_urls,
        "provenance_strings": provenance,
        "boundary_hint": boundary_hint,
        "out_of_boundary_instrumentation_terms": out_terms,
        "pdmx_license_conflict_verified_false": False,
        "scan_source_identity_verified": False,
        "symbolic_score_bytes_acquired": False,
    }


def catalog(
    archive_path: Path,
    acquisition_report_path: Path,
    output_path: Path,
) -> dict[str, object]:
    acquisition = json.loads(
        acquisition_report_path.read_text(encoding="utf-8")
    )
    asset = acquisition.get("asset")
    if (
        acquisition.get("role") != ACQUISITION_ROLE
        or acquisition.get("record_id") != RECORD_ID
        or acquisition.get("version") != VERSION
        or acquisition.get("expected_bytes") != EXPECTED_BYTES
        or acquisition.get("expected_md5") != EXPECTED_MD5
        or acquisition.get("training_authorized") is not False
        or not isinstance(asset, dict)
        or asset.get("sha256") != sha256_file(archive_path)
        or archive_path.stat().st_size != EXPECTED_BYTES
    ):
        raise ValueError("unexpected PDMX metadata acquisition contract")

    candidates: list[dict[str, object]] = []
    member_count = 0
    invalid_member_count = 0
    duplicate_score_ids: set[int] = set()
    seen_score_ids: set[int] = set()
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            match = MEMBER_NAME.fullmatch(member.name)
            if not member.isfile() or match is None:
                continue
            member_count += 1
            if member_count % 25_000 == 0:
                print(
                    f"{member_count} PDMX metadata rows scanned",
                    flush=True,
                )
            if (
                member.size <= 0
                or member.size > MAXIMUM_MEMBER_BYTES
                or int(match.group(1)) in seen_score_ids
            ):
                invalid_member_count += 1
                if int(match.group(1)) in seen_score_ids:
                    duplicate_score_ids.add(int(match.group(1)))
                continue
            stream = archive.extractfile(member)
            if stream is None:
                invalid_member_count += 1
                continue
            try:
                payload = json.loads(stream.read(member.size + 1))
            except (json.JSONDecodeError, UnicodeDecodeError):
                invalid_member_count += 1
                continue
            if not isinstance(payload, dict):
                invalid_member_count += 1
                continue
            row = _score_row(payload)
            score_id = int(match.group(1))
            seen_score_ids.add(score_id)
            if row is None:
                continue
            if int(row["score_id"]) != score_id:
                invalid_member_count += 1
                continue
            row["metadata_member"] = member.name
            candidates.append(row)

    candidates.sort(key=lambda row: int(row["score_id"]))
    status_counts = Counter(
        str(row["boundary_hint"]) for row in candidates
    )
    eligible = [
        row
        for row in candidates
        if row["public_domain_metadata_consistent"] is True
        and not row["out_of_boundary_instrumentation_terms"]
        and str(row["boundary_hint"]).endswith("candidate")
    ]
    report = {
        "format": 1,
        "created_at": utc_now_iso(),
        "name": "PDMX scores with explicit IMSLP provenance",
        "role": ROLE,
        "record_id": RECORD_ID,
        "version": VERSION,
        "archive_path": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "acquisition_report_path": str(acquisition_report_path),
        "acquisition_report_sha256": sha256_file(
            acquisition_report_path
        ),
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_authorized": False,
        "authorization_reason": (
            "metadata candidates still require PDMX license-conflict "
            "verification, symbolic-byte acquisition, IMSLP scan identity, "
            "declared-boundary parsing, exact-work linkage, and immutable "
            "work-level split assignment"
        ),
        "metadata_member_count": member_count,
        "invalid_member_count": invalid_member_count,
        "duplicate_score_ids": sorted(duplicate_score_ids),
        "candidate_count": len(candidates),
        "public_domain_boundary_candidate_count": len(eligible),
        "unique_imslp_reverse_lookup_source_count": len(
            {
                source_id
                for row in candidates
                for source_id in row["imslp_reverse_lookup_source_ids"]
            }
        ),
        "boundary_hint_counts": dict(sorted(status_counts.items())),
        "public_domain_boundary_candidates": eligible,
        "candidates": candidates,
    }
    atomic_write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_path", type=Path)
    parser.add_argument("acquisition_report_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    report = catalog(
        args.archive_path.resolve(),
        args.acquisition_report_path.resolve(),
        args.output_path.resolve(),
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "metadata_member_count",
                    "invalid_member_count",
                    "candidate_count",
                    "public_domain_boundary_candidate_count",
                    "unique_imslp_reverse_lookup_source_count",
                    "boundary_hint_counts",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
