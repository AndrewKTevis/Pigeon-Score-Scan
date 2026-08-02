from __future__ import annotations

"""Catalog release-authorized printed scans from the Polish Scores corpus.

This is deliberately a discovery stage, not an evaluation dataset builder.
Rows are candidates until the exact scan page has been acquired, the Humdrum
reference has been converted and boundary-validated, and page-to-reference
alignment has been proved.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.util import atomic_write_json, sha256_file, utc_now_iso  # noqa: E402


REPOSITORY = "https://github.com/pl-wnifc/humdrum-polish-scores"
REVISION = "13ac964e0dd8bcd5fffd837169cbf653242c12e8"
LICENSE = "CC-BY-4.0"
ROLE = "candidate_catalog_not_training_or_evaluation"
LICENSE_MARKER = "License CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0)"
COPYRIGHT_MARKER = "Narodowy Instytut Fryderyka Chopina"
REPOSITORY_COPYRIGHT_MARKER = "The Fryderyk Chopin Institute"
PRINT_SCAN_MARKER = "/galeria/druki-muzyczne/"
FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{64}")
INTEGER_PATTERN = re.compile(r"(?<!\w)\d+(?!\w)")
EXCLUDED_INSTRUMENT_PATTERN = re.compile(
    r"\b("
    r"piano|fortepian|keyboard|organ|organo|harmonium|"
    r"harpsichord|cembalo|clavicembalo|clavichord|"
    r"voice|vocal|vox|canto|choir|chorus|ch[oó]r|"
    r"orchestra|orchestr|ensemble|consort|continuo|"
    r"percussion|drum|timpani"
    r")\b",
    re.IGNORECASE,
)
DISALLOWED_EXCLUSIVE_TYPES = {
    "**text",
    "**silbe",
    "**harm",
    "**fb",
    "**fba",
    "**recip",
}


@dataclass(frozen=True)
class ParsedKern:
    path: Path
    records: dict[str, str]
    initial_types: tuple[str, ...]
    maximum_kern_spines: int
    exclusive_types: frozenset[str]
    structurally_valid: bool


def _records(lines: Iterable[str]) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in lines:
        if not line.startswith("!!!") or ":" not in line:
            continue
        key, value = line[3:].split(":", 1)
        records.setdefault(key.strip(), value.strip())
    return records


def _apply_spine_manipulators(
    active: list[str],
    fields: list[str],
) -> tuple[list[str], bool]:
    if len(active) != len(fields):
        return active, False
    exchanged = [index for index, token in enumerate(fields) if token == "*x"]
    if exchanged:
        if len(exchanged) != 2:
            return active, False
        left, right = exchanged
        active[left], active[right] = active[right], active[left]

    result: list[str] = []
    index = 0
    while index < len(active):
        token = fields[index]
        spine_type = active[index]
        if token == "*^":
            result.extend((spine_type, spine_type))
        elif token == "*+":
            result.extend((spine_type, ""))
        elif token == "*-":
            pass
        elif token == "*v":
            end = index + 1
            while end < len(active) and fields[end] == "*v":
                end += 1
            if end - index < 2 or len(set(active[index:end])) != 1:
                return active, False
            result.append(spine_type)
            index = end - 1
        else:
            if token.startswith("**"):
                spine_type = token
            result.append(spine_type)
        index += 1
    return result, True


def parse_kern(path: Path) -> ParsedKern:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    initial_line = next((line for line in lines if line.startswith("**")), "")
    initial_types = tuple(initial_line.split("\t")) if initial_line else ()
    active = list(initial_types)
    maximum_kern = sum(value == "**kern" for value in active)
    exclusive_types = {value for value in active if value.startswith("**")}
    valid = bool(active)
    initial_seen = False
    for line in lines:
        if not initial_seen:
            if line == initial_line:
                initial_seen = True
            continue
        if not line.startswith("*") or line.startswith("!!"):
            continue
        fields = line.split("\t")
        active, row_valid = _apply_spine_manipulators(active, fields)
        valid = valid and row_valid
        exclusive_types.update(
            value for value in active if value.startswith("**")
        )
        maximum_kern = max(
            maximum_kern,
            sum(value == "**kern" for value in active),
        )
    return ParsedKern(
        path=path,
        records=_records(lines),
        initial_types=initial_types,
        maximum_kern_spines=maximum_kern,
        exclusive_types=frozenset(exclusive_types),
        structurally_valid=valid,
    )


def _scan_url(value: str) -> str:
    return value.split(" @", 1)[0].strip()


def _single_instrument(value: str) -> bool:
    normalized = " ".join(value.split())
    numbers = INTEGER_PATTERN.findall(normalized)
    return (
        bool(normalized)
        and numbers == ["1"]
        and EXCLUDED_INSTRUMENT_PATTERN.search(normalized) is None
    )


def rejection_reasons(parsed: ParsedKern) -> tuple[str, ...]:
    records = parsed.records
    reasons: list[str] = []
    scan_url = _scan_url(records.get("URL-scan", ""))
    if PRINT_SCAN_MARKER not in scan_url:
        reasons.append("not_explicitly_printed_scan")
    if records.get("YEM") != LICENSE_MARKER:
        reasons.append("transcription_license_not_exact_cc_by_4")
    if COPYRIGHT_MARKER not in records.get("YEC", ""):
        reasons.append("transcription_copyright_provenance_missing")
    if not parsed.structurally_valid:
        reasons.append("invalid_or_unsupported_spine_structure")
    if parsed.initial_types.count("**kern") != 1:
        reasons.append("not_single_initial_kern_spine")
    if parsed.maximum_kern_spines != 1:
        reasons.append("independent_voice_or_staff_split")
    if parsed.exclusive_types & DISALLOWED_EXCLUSIVE_TYPES:
        reasons.append("lyrics_harmony_or_auxiliary_pitch_encoding")
    if not _single_instrument(records.get("AIN", "")):
        reasons.append("not_one_supported_non_keyboard_instrument")
    if "-pc" in parsed.path.stem.casefold():
        reasons.append("post_correction_variant")
    required = ("COM", "OTL", "SMS-siglum", "SMS-shelfmark")
    if any(not records.get(key, "").strip() for key in required):
        reasons.append("insufficient_work_provenance")
    return tuple(sorted(set(reasons)))


def _fingerprint(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def candidate_row(parsed: ParsedKern, corpus_root: Path) -> dict[str, object]:
    reasons = rejection_reasons(parsed)
    records = parsed.records
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
    work_fingerprint = _fingerprint(REPOSITORY, "work", work_identity)
    if (
        FINGERPRINT_PATTERN.fullmatch(source_group) is None
        or FINGERPRINT_PATTERN.fullmatch(work_fingerprint) is None
    ):
        raise AssertionError("invalid Polish Scores fingerprint")
    return {
        "accepted": not reasons,
        "reasons": list(reasons),
        "path": str(parsed.path.relative_to(corpus_root)).replace("\\", "/"),
        "sha256": sha256_file(parsed.path),
        "composer": records.get("COM", ""),
        "title": records.get("OTL", ""),
        "instrumentation": records.get("AIN", ""),
        "genre": records.get("AGN", ""),
        "scan_url": _scan_url(records.get("URL-scan", "")),
        "scan_pdf_url": _scan_url(records.get("URL-pdf-islandora", "")),
        "scan_page_note": records.get("ONB-nifc", ""),
        "iiif_url": records.get("IIIF", ""),
        "scan_asset_rights_status": "unverified",
        "scan_asset_rights_evidence": "",
        "source_siglum": records.get("SMS-siglum", ""),
        "source_shelfmark": records.get("SMS-shelfmark", ""),
        "source_work": records.get("SMS-shelfwork", ""),
        "source_group_fingerprint": source_group,
        "work_fingerprint": work_fingerprint,
        "initial_kern_spines": parsed.initial_types.count("**kern"),
        "maximum_kern_spines": parsed.maximum_kern_spines,
        "exclusive_types": sorted(parsed.exclusive_types),
        "transcription_license": records.get("YEM", ""),
        "transcription_copyright": records.get("YEC", ""),
    }


def _git_output(corpus_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(corpus_root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    if completed.returncode != 0:
        raise ValueError(completed.stderr.strip() or "git validation failed")
    return completed.stdout.strip()


def validate_corpus(corpus_root: Path) -> dict[str, str]:
    revision = _git_output(corpus_root, "rev-parse", "HEAD")
    if revision != REVISION:
        raise ValueError(f"Polish Scores revision is not pinned: {revision}")
    if _git_output(corpus_root, "status", "--porcelain"):
        raise ValueError("Polish Scores corpus worktree is dirty")
    license_path = corpus_root / "LICENSE.txt"
    license_text = license_path.read_text(encoding="utf-8")
    if (
        "License: https://creativecommons.org/licenses/by/4.0" not in license_text
        or REPOSITORY_COPYRIGHT_MARKER not in license_text
    ):
        raise ValueError("Polish Scores repository license contract changed")
    return {
        "revision": revision,
        "license_sha256": sha256_file(license_path),
    }


def catalog(corpus_root: Path, output_path: Path) -> dict[str, object]:
    corpus = validate_corpus(corpus_root)
    paths = sorted(corpus_root.glob("*/kern/*.krn"))
    if len(paths) < 8_000:
        raise ValueError("Polish Scores corpus checkout is incomplete")
    rows: list[dict[str, object]] = []
    rejection_counts: Counter[str] = Counter()
    for position, path in enumerate(paths, start=1):
        row = candidate_row(parse_kern(path), corpus_root)
        rows.append(row)
        rejection_counts.update(str(reason) for reason in row["reasons"])
        if position % 500 == 0 or position == len(paths):
            print(f"[{position}/{len(paths)}] cataloged", flush=True)
    accepted = [row for row in rows if row["accepted"]]
    report = {
        "format": 1,
        "created_at": utc_now_iso(),
        "name": "Polish Scores printed single-staff candidate catalog",
        "role": ROLE,
        "repository": REPOSITORY,
        "revision": corpus["revision"],
        "repository_license": LICENSE,
        "repository_license_sha256": corpus["license_sha256"],
        "release_authorized": False,
        "release_authorization_reason": (
            "candidate discovery only; object-level scan rights, scan acquisition, "
            "exact page alignment, reference conversion, and production-boundary "
            "validation are pending"
        ),
        "training_authorized": False,
        "evaluation_authorized": False,
        "case_count": len(rows),
        "accepted_candidate_count": len(accepted),
        "accepted_work_count": len(
            {str(row["work_fingerprint"]) for row in accepted}
        ),
        "accepted_source_group_count": len(
            {str(row["source_group_fingerprint"]) for row in accepted}
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
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    report = catalog(args.corpus_root.resolve(), args.output_path.resolve())
    print(
        json.dumps(
            {
                "case_count": report["case_count"],
                "accepted_candidate_count": report[
                    "accepted_candidate_count"
                ],
                "accepted_work_count": report["accepted_work_count"],
                "accepted_source_group_count": report[
                    "accepted_source_group_count"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
