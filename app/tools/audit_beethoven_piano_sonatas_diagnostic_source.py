from __future__ import annotations

"""Audit the pinned Beethoven sonata scans and Humdrum references.

The upstream repository documents the scans as the reference edition for its
Humdrum encodings, but it provides no license file and some logical movements
share or lack a same-named PDF.  This tool records those facts without granting
training, evaluation, or release authority.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pymupdf as fitz


APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
sys.path.insert(0, str(APP_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from app.tools.humdrum_boundary import (  # noqa: E402
    analyze_humdrum_boundary,
)
from scorescan.product_scope import (  # noqa: E402
    PRODUCTION_BOUNDARY_CONTRACT_VERSION,
)
from scorescan.util import atomic_write_json, sha256_file, utc_now_iso  # noqa: E402


ROLE = "beethoven_piano_sonatas_pinned_scan_reference_diagnostic"
REVISION = re.compile(r"[0-9a-f]{40}")
MOVEMENT = re.compile(r"sonata(?P<sonata>\d{2})-(?P<segment>\d+)")
LICENSE_NAMES = (
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "COPYING",
    "COPYING.txt",
)


def _metadata(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(
        encoding="utf-8-sig",
        errors="strict",
    ).splitlines():
        if not line.startswith("!!!") or ":" not in line:
            continue
        key, value = line[3:].split(":", 1)
        if key in {"OTL", "OTP", "OPS", "OMV", "OMD"}:
            result[key] = value.strip()
    return result


def _pdf_record(path: Path) -> dict[str, object]:
    with fitz.open(path) as document:
        pages = len(document)
        text_characters = sum(
            len(page.get_text().strip()) for page in document
        )
        pages_with_images = sum(
            bool(page.get_images(full=True)) for page in document
        )
        rotations = sorted({int(page.rotation) for page in document})
        producer = str(document.metadata.get("producer") or "")
        creator = str(document.metadata.get("creator") or "")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "pages": pages,
        "text_characters": text_characters,
        "pages_with_images": pages_with_images,
        "image_only_scan_candidate": (
            pages > 0
            and text_characters == 0
            and pages_with_images == pages
        ),
        "page_rotations": rotations,
        "producer": producer,
        "creator": creator,
    }


def audit(source_root: Path, *, revision: str) -> dict[str, object]:
    source_root = source_root.resolve()
    if not REVISION.fullmatch(revision):
        raise ValueError("revision must be a full lowercase Git commit")
    readme = source_root / "README.md"
    makefile = source_root / "Makefile"
    kern_root = source_root / "kern"
    pdf_root = source_root / "reference-edition"
    for required in (readme, makefile, kern_root, pdf_root):
        if not required.exists():
            raise FileNotFoundError(required)

    kern_paths = sorted(kern_root.glob("*.krn"))
    pdf_paths = sorted(pdf_root.glob("*.pdf"))
    if not kern_paths or not pdf_paths:
        raise ValueError("pinned sonata source is empty")
    invalid_names = [
        path.name
        for path in (*kern_paths, *pdf_paths)
        if MOVEMENT.fullmatch(path.stem) is None
    ]
    if invalid_names:
        raise ValueError(f"invalid sonata movement names: {invalid_names}")

    pdf_records = {
        path.stem: _pdf_record(path) for path in pdf_paths
    }
    hashes_to_names: dict[str, list[str]] = defaultdict(list)
    for name, record in pdf_records.items():
        hashes_to_names[str(record["sha256"])].append(name)
    duplicate_groups = [
        {
            "sha256": digest,
            "movement_names": sorted(names),
            "pages": int(pdf_records[names[0]]["pages"]),
        }
        for digest, names in sorted(hashes_to_names.items())
        if len(names) > 1
    ]

    movements: list[dict[str, object]] = []
    reason_counts: Counter[str] = Counter()
    sonata_numbers: set[int] = set()
    for path in kern_paths:
        match = MOVEMENT.fullmatch(path.stem)
        assert match is not None
        sonata_number = int(match.group("sonata"))
        sonata_numbers.add(sonata_number)
        boundary = analyze_humdrum_boundary(
            path,
            instrumentation="1 piano",
        ).to_dict()
        reason_counts.update(str(item) for item in boundary["reasons"])
        pdf = pdf_records.get(path.stem)
        duplicate_pdf = bool(
            pdf
            and len(hashes_to_names[str(pdf["sha256"])]) > 1
        )
        movements.append(
            {
                "movement_name": path.stem,
                "sonata_number": sonata_number,
                "segment_number": int(match.group("segment")),
                "humdrum_path": str(path),
                "humdrum_sha256": sha256_file(path),
                "metadata": _metadata(path),
                "boundary": boundary,
                "same_named_pdf_present": pdf is not None,
                "same_named_pdf": pdf,
                "same_named_pdf_is_duplicate": duplicate_pdf,
                "exact_movement_page_range_independently_verified": False,
                "training_authorized": False,
                "evaluation_authorized": False,
                "release_authorized": False,
            }
        )

    unique_pdf_records = {
        digest: pdf_records[names[0]]
        for digest, names in hashes_to_names.items()
    }
    unique_page_count = sum(
        int(record["pages"]) for record in unique_pdf_records.values()
    )
    license_files = [
        source_root / name
        for name in LICENSE_NAMES
        if (source_root / name).is_file()
    ]
    readme_text = readme.read_text(encoding="utf-8", errors="strict")
    documented_reference_relation = (
        "Scans of the source edition" in readme_text
        and "reference" in readme_text
    )
    paired_names = set(pdf_records) & {path.stem for path in kern_paths}
    missing_pdf_names = sorted(
        {path.stem for path in kern_paths} - set(pdf_records)
    )
    duplicate_movement_names = {
        name
        for group in duplicate_groups
        for name in group["movement_names"]
    }
    unique_filename_pair_count = len(
        paired_names - duplicate_movement_names
    )
    return {
        "format": 1,
        "created_at": utc_now_iso(),
        "name": "Beethoven piano sonatas pinned scan/reference audit",
        "role": ROLE,
        "source_revision": revision,
        "source_readme_sha256": sha256_file(readme),
        "source_makefile_sha256": sha256_file(makefile),
        "boundary_contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
        "declared_instrumentation": "1 piano",
        "sonata_work_count": len(sonata_numbers),
        "sonata_numbers": sorted(sonata_numbers),
        "humdrum_movement_count": len(kern_paths),
        "boundary_accepted_humdrum_count": sum(
            row["boundary"]["accepted"] is True for row in movements
        ),
        "boundary_reason_counts": dict(sorted(reason_counts.items())),
        "pdf_filename_count": len(pdf_paths),
        "unique_pdf_count": len(unique_pdf_records),
        "same_named_pair_count": len(paired_names),
        "unique_same_named_pair_count": unique_filename_pair_count,
        "missing_same_named_pdf_count": len(missing_pdf_names),
        "missing_same_named_pdf_names": missing_pdf_names,
        "duplicate_pdf_group_count": len(duplicate_groups),
        "duplicate_pdf_groups": duplicate_groups,
        "unique_physical_scan_page_count": unique_page_count,
        "all_unique_pdfs_image_only": all(
            bool(record["image_only_scan_candidate"])
            for record in unique_pdf_records.values()
        ),
        "all_unique_pdf_pages_unrotated": all(
            record["page_rotations"] == [0]
            for record in unique_pdf_records.values()
        ),
        "upstream_documents_reference_edition_relation": (
            documented_reference_relation
        ),
        "source_license_file_present": bool(license_files),
        "source_license_files": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for path in license_files
        ],
        "source_images_downloaded": True,
        "source_images_hashed": True,
        "work_identity_candidate": True,
        "movement_page_ranges_independently_verified": False,
        "semantic_completeness_independently_verified": False,
        "independent_human_annotation": False,
        "metadata_boundary_audit_complete": True,
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_evaluation_authorized": False,
        "release_authorized": False,
        "stop_reasons": [
            "The upstream repository contains no explicit license file.",
            "Two logical movements share one byte-identical multi-movement PDF.",
            "Two Humdrum segments have no same-named PDF and may be sections of another scan.",
            "Movement page ranges and complete target semantics have no independent second review.",
        ],
        "movements": movements,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.source_root, revision=args.revision)
    atomic_write_json(args.output, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "sonata_work_count",
                    "humdrum_movement_count",
                    "boundary_accepted_humdrum_count",
                    "pdf_filename_count",
                    "unique_pdf_count",
                    "unique_physical_scan_page_count",
                    "source_license_file_present",
                    "training_authorized",
                    "release_authorized",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
