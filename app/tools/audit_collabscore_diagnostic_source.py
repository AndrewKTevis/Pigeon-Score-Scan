from __future__ import annotations

"""Audit a pinned CollabScore source tree before downloading any IIIF images.

CollabScore is useful diagnostic metadata, but its non-commercial share-alike
license and its vocal/condensed content do not authorize product training or
release evidence.  This audit intentionally stops at local reference metadata
and makes every downstream authorization false.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
sys.path.insert(0, str(APP_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from app.tools.prepare_muse_omr_benchmark import (  # noqa: E402
    analyze_reference_boundary,
)
from scorescan.product_scope import (  # noqa: E402
    PRODUCTION_BOUNDARY_CONTRACT_VERSION,
)
from scorescan.util import atomic_write_json, sha256_file, utc_now_iso  # noqa: E402


ROLE = "collabscore_pinned_metadata_boundary_diagnostic_no_image_download"
LICENSE = "CC-BY-NC-SA-4.0"
REFERENCE_ID = re.compile(r"[A-Z]\d+_\d+")
REVISION = re.compile(r"[0-9a-f]{40}")


def _read_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def audit(source_root: Path, *, revision: str) -> dict[str, object]:
    source_root = source_root.resolve()
    if not REVISION.fullmatch(revision):
        raise ValueError("revision must be a full lowercase Git commit")
    dataset_path = source_root / "dataset.json"
    license_path = source_root / "LICENSE.md"
    ground_truth = source_root / "ground_truth"
    iiif = source_root / "iiif"
    for required in (dataset_path, license_path, ground_truth, iiif):
        if not required.exists():
            raise FileNotFoundError(required)

    dataset = _read_object(dataset_path)
    raw_works = dataset.get("list_opus")
    if not isinstance(raw_works, list) or not raw_works:
        raise ValueError("CollabScore dataset has no list_opus entries")

    works: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in raw_works:
        if not isinstance(raw, dict):
            raise ValueError("invalid CollabScore work row")
        reference_id = str(raw.get("ref", ""))
        if (
            not REFERENCE_ID.fullmatch(reference_id)
            or reference_id in seen
        ):
            raise ValueError("invalid or duplicate CollabScore reference id")
        seen.add(reference_id)
        musicxml = ground_truth / f"{reference_id}.musicxml"
        mei = ground_truth / f"{reference_id}.mei"
        manifest = iiif / f"{reference_id}_mnf.json"
        annotations = iiif / f"{reference_id}_annot.json"
        boundary = (
            analyze_reference_boundary(musicxml)
            if musicxml.is_file()
            else None
        )
        reasons = (
            [str(value) for value in boundary.get("reasons", [])]
            if isinstance(boundary, dict)
            else ["musicxml_reference_missing"]
        )
        works.append(
            {
                "reference_id": reference_id,
                "title": str(raw.get("title", "")).strip(),
                "genre": str(raw.get("genre", "")).strip(),
                "declared_parts": int(raw.get("nb_parts", 0) or 0),
                "declared_music_pages": int(
                    raw.get("nb_music_pages", 0) or 0
                ),
                "declared_systems": int(
                    raw.get("nb_systems", 0) or 0
                ),
                "declared_measures": int(
                    raw.get("nb_measures", 0) or 0
                ),
                "iiif_source": str(raw.get("iiif_link", "")).strip(),
                "musicxml_present": musicxml.is_file(),
                "musicxml_sha256": (
                    sha256_file(musicxml) if musicxml.is_file() else None
                ),
                "mei_present": mei.is_file(),
                "mei_sha256": sha256_file(mei) if mei.is_file() else None,
                "iiif_manifest_present": manifest.is_file(),
                "iiif_manifest_sha256": (
                    sha256_file(manifest)
                    if manifest.is_file()
                    else None
                ),
                "iiif_annotations_present": annotations.is_file(),
                "iiif_annotations_sha256": (
                    sha256_file(annotations)
                    if annotations.is_file()
                    else None
                ),
                "boundary": boundary,
                "strict_boundary_accepted": bool(
                    isinstance(boundary, dict)
                    and boundary.get("accepted") is True
                ),
                "lyrics_only_exclusion": reasons == ["lyrics"],
                "image_download_authorized": False,
                "training_authorized": False,
                "release_evaluation_authorized": False,
                "release_authorized": False,
            }
        )

    reason_counts = Counter(
        reason
        for row in works
        for reason in (
            row["boundary"].get("reasons", [])
            if isinstance(row.get("boundary"), dict)
            else ["musicxml_reference_missing"]
        )
    )
    declared_page_sum = sum(
        int(row["declared_music_pages"]) for row in works
    )
    strict_count = sum(bool(row["strict_boundary_accepted"]) for row in works)
    lyrics_only_count = sum(bool(row["lyrics_only_exclusion"]) for row in works)
    dataset_total_pages = int(dataset.get("total_pages", 0) or 0)
    return {
        "format": 1,
        "created_at": utc_now_iso(),
        "name": "CollabScore pinned source pre-download boundary audit",
        "role": ROLE,
        "source_revision": revision,
        "source_dataset_json": str(dataset_path),
        "source_dataset_sha256": sha256_file(dataset_path),
        "source_license": LICENSE,
        "source_license_sha256": sha256_file(license_path),
        "boundary_contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
        "declared_work_count": len(works),
        "dataset_total_pages": dataset_total_pages,
        "summed_declared_music_pages": declared_page_sum,
        "page_count_consistent": dataset_total_pages == declared_page_sum,
        "musicxml_reference_count": sum(
            bool(row["musicxml_present"]) for row in works
        ),
        "strict_boundary_accepted_work_count": strict_count,
        "lyrics_only_excluded_work_count": lyrics_only_count,
        "other_or_missing_excluded_work_count": (
            len(works) - strict_count - lyrics_only_count
        ),
        "boundary_reason_counts": dict(sorted(reason_counts.items())),
        "metadata_boundary_audit_complete": True,
        "source_images_downloaded": False,
        "source_images_hashed": False,
        "page_alignment_revalidated": False,
        "independent_human_annotation": False,
        "noncommercial_use_clearance_recorded": False,
        "image_download_authorized": False,
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_evaluation_authorized": False,
        "release_authorized": False,
        "stop_reason": (
            "No reference work passes the frozen instrumental boundary; "
            "the pinned source is CC-BY-NC-SA and no separate use clearance "
            "is recorded, so downloading IIIF images would not advance the "
            "production evidence gate."
        ),
        "works": works,
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
                "declared_work_count": report["declared_work_count"],
                "dataset_total_pages": report["dataset_total_pages"],
                "musicxml_reference_count": report[
                    "musicxml_reference_count"
                ],
                "strict_boundary_accepted_work_count": report[
                    "strict_boundary_accepted_work_count"
                ],
                "lyrics_only_excluded_work_count": report[
                    "lyrics_only_excluded_work_count"
                ],
                "image_download_authorized": report[
                    "image_download_authorized"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
