from __future__ import annotations

"""Audit OLiMPiC's full-page IMSLP sources without overstating their labels.

The public ``sources-for-scanned`` archive contains complete page images and
system coordinates in addition to the better-known grand-staff crops.  The
published MusicXML, however, labels only those piano grand-staff crops.  This
audit proves the crop-to-page mapping, measures source-document leakage across
published splits, and keeps incomplete full-page semantics out of release
evidence.
"""

import argparse
import hashlib
import json
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

import yaml


APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
sys.path.insert(0, str(APP_ROOT / "src"))

from scorescan.product_scope import (  # noqa: E402
    PRODUCTION_BOUNDARY_CONTRACT_VERSION,
)
from scorescan.util import atomic_write_json, sha256_file, utc_now_iso  # noqa: E402


ROLE = "olimpic_full_page_physical_scan_alignment_audit"
SOURCE_ASSET_KEY = "olimpic_scanned_sources"
SOURCE_ARCHIVE_SHA256 = (
    "8b77529d06cbf3d0f392af7ea5457906a510cf6ca7dad8eb751f6839bfde39f8"
)
DOCUMENT_NAME = re.compile(r"^IMSLP(?P<document>\d+)(?:-|\.pdf$)")


def _contained_file(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes OLiMPiC source root: {relative}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if (
        len(header) != 24
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
    ):
        raise ValueError(f"invalid PNG header: {path}")
    return struct.unpack(">II", header[16:24])


def _split_samples(path: Path) -> set[str]:
    samples: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip().replace("\\", "/")
        if not value:
            continue
        prefix = "samples/"
        if not value.startswith(prefix):
            raise ValueError(f"unexpected OLiMPiC split entry: {value}")
        sample = value[len(prefix) :]
        if sample in samples:
            raise ValueError(f"duplicate OLiMPiC split entry: {value}")
        samples.add(sample)
    return samples


def _document_ids(paths: list[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        match = DOCUMENT_NAME.match(path.name)
        if match is None:
            raise ValueError(f"unexpected IMSLP source filename: {path.name}")
        document = match.group("document")
        if document in result:
            raise ValueError(f"duplicate IMSLP source document: {document}")
        result.add(document)
    return result


def _mapping_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _provenance_asset(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assets = payload.get("assets")
    if not isinstance(assets, list):
        raise ValueError("external provenance has no asset list")
    matches = [
        row
        for row in assets
        if isinstance(row, dict) and row.get("key") == SOURCE_ASSET_KEY
    ]
    if len(matches) != 1:
        raise ValueError("external provenance must bind one OLiMPiC source asset")
    asset = matches[0]
    if asset.get("archive_sha256") != SOURCE_ARCHIVE_SHA256:
        raise ValueError("OLiMPiC source archive hash is stale or unexpected")
    if asset.get("license") != "CC-BY-SA":
        raise ValueError("OLiMPiC source archive license is stale or unexpected")
    return asset


def audit(
    *,
    scanned_root: Path,
    source_root: Path,
    provenance_path: Path,
) -> dict[str, object]:
    scanned_root = scanned_root.resolve()
    source_root = source_root.resolve()
    provenance_path = provenance_path.resolve()
    samples_root = scanned_root / "samples"
    mapping_root = source_root / "corpus_to_imslp"
    pdf_root = source_root / "imslp_pdfs"
    page_root = source_root / "imslp_pngs"
    system_root = source_root / "imslp_systems"
    license_path = scanned_root / "LICENSE"
    readme_path = scanned_root / "README.md"
    for required in (
        samples_root,
        mapping_root,
        pdf_root,
        page_root,
        system_root,
        license_path,
        readme_path,
        provenance_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    asset = _provenance_asset(provenance_path)
    license_text = license_path.read_text(encoding="utf-8")
    readme_text = readme_path.read_text(encoding="utf-8")
    if "Attribution-ShareAlike 4.0 International" not in license_text:
        raise ValueError("OLiMPiC scanned license is not CC BY-SA 4.0")
    if "CC BY-SA" not in readme_text:
        raise ValueError("OLiMPiC scanned README does not bind its license")

    splits = {
        "dev": _split_samples(scanned_root / "samples.dev.txt"),
        "test": _split_samples(scanned_root / "samples.test.txt"),
    }
    split_overlap = splits["dev"] & splits["test"]
    if split_overlap:
        raise ValueError("published OLiMPiC dev/test samples overlap")
    published_samples = splits["dev"] | splits["test"]

    mapping_paths = sorted(mapping_root.glob("*.yaml"))
    system_paths = sorted(system_root.glob("IMSLP*.yaml"))
    pdf_paths = sorted(pdf_root.glob("*.pdf"))
    page_paths = sorted(page_root.glob("IMSLP*/*.png"))
    if not all((mapping_paths, system_paths, pdf_paths, page_paths)):
        raise ValueError("OLiMPiC full-page source tree is incomplete")

    pdf_documents = _document_ids(pdf_paths)
    page_documents = {
        path.parent.name.removeprefix("IMSLP") for path in page_paths
    }
    if any(not value.isdigit() for value in page_documents):
        raise ValueError("unexpected IMSLP page directory")
    system_documents = {
        path.stem.removeprefix("IMSLP") for path in system_paths
    }
    if any(not value.isdigit() for value in system_documents):
        raise ValueError("unexpected IMSLP system-coordinate filename")

    invalid_png_dimensions: list[str] = []
    for path in page_paths:
        width, height = _png_dimensions(path)
        if width < 1 or height < 1:
            invalid_png_dimensions.append(str(path))
    if invalid_png_dimensions:
        raise ValueError(
            f"invalid OLiMPiC page dimensions: {invalid_png_dimensions[:3]}"
        )

    systems_by_document: dict[str, dict[object, object]] = {}
    for path in system_paths:
        document = path.stem.removeprefix("IMSLP")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(
            payload.get("pages"), dict
        ):
            raise ValueError(f"invalid OLiMPiC system coordinates: {path}")
        systems_by_document[document] = payload["pages"]

    mapped_samples: set[str] = set()
    mapped_documents: set[str] = set()
    mapped_pages: set[tuple[str, int]] = set()
    mapped_systems: set[tuple[str, int, int]] = set()
    duplicate_source_systems: list[tuple[str, int, int]] = []
    sample_to_document: dict[str, str] = {}
    sample_to_page: dict[str, tuple[str, int]] = {}
    sample_to_system: dict[str, tuple[str, int, int]] = {}

    for path in mapping_paths:
        score_id = path.stem
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not payload:
            raise ValueError(f"invalid OLiMPiC mapping: {path}")
        for raw_sample, raw_mapping in payload.items():
            sample = str(raw_sample).replace("\\", "/")
            if not sample.startswith(f"{score_id}/"):
                raise ValueError(f"mapping score mismatch: {sample}")
            if sample in mapped_samples:
                raise ValueError(f"duplicate mapped OLiMPiC sample: {sample}")
            if not isinstance(raw_mapping, dict):
                raise ValueError(f"invalid mapping row: {sample}")
            document = str(raw_mapping.get("imslpDocument", "")).lstrip("#")
            page_number = int(raw_mapping.get("imslpPage", 0))
            system_number = int(raw_mapping.get("imslpSystem", 0))
            if (
                not document.isdigit()
                or page_number < 1
                or system_number < 1
            ):
                raise ValueError(f"invalid IMSLP mapping row: {sample}")

            if sample in published_samples:
                for suffix in (".png", ".musicxml", ".lmx"):
                    _contained_file(samples_root, sample + suffix)
            pages = systems_by_document.get(document)
            if pages is None:
                raise ValueError(f"missing system map for IMSLP{document}")
            page = pages.get(page_number)
            if page is None:
                page = pages.get(str(page_number))
            if not isinstance(page, dict):
                raise ValueError(
                    f"missing page {page_number} for IMSLP{document}"
                )
            image_relative = str(page.get("image", ""))
            image = _contained_file(page_root, image_relative)
            width, height = _png_dimensions(image)
            if (
                width != int(page.get("width", 0))
                or height != int(page.get("height", 0))
            ):
                raise ValueError(f"page dimensions disagree for {image}")
            systems = page.get("systems")
            if (
                not isinstance(systems, list)
                or system_number > len(systems)
                or not isinstance(systems[system_number - 1], dict)
            ):
                raise ValueError(f"missing source system for {sample}")

            source_system = (document, page_number, system_number)
            if source_system in mapped_systems:
                duplicate_source_systems.append(source_system)
            mapped_samples.add(sample)
            mapped_documents.add(document)
            mapped_pages.add((document, page_number))
            mapped_systems.add(source_system)
            sample_to_document[sample] = document
            sample_to_page[sample] = (document, page_number)
            sample_to_system[sample] = source_system

    missing_mappings = sorted(published_samples - mapped_samples)
    extra_mappings = sorted(mapped_samples - published_samples)
    if missing_mappings:
        raise ValueError(
            "published OLiMPiC samples are missing full-page mappings: "
            f"{missing_mappings[:3]}"
        )

    split_documents = {
        split: {sample_to_document[sample] for sample in samples}
        for split, samples in splits.items()
    }
    split_pages = {
        split: {sample_to_page[sample] for sample in samples}
        for split, samples in splits.items()
    }
    split_systems = {
        split: {sample_to_system[sample] for sample in samples}
        for split, samples in splits.items()
    }
    split_scores = {
        split: {sample.split("/", 1)[0] for sample in samples}
        for split, samples in splits.items()
    }
    document_overlap = split_documents["dev"] & split_documents["test"]
    page_overlap = split_pages["dev"] & split_pages["test"]
    system_overlap = split_systems["dev"] & split_systems["test"]
    score_overlap = split_scores["dev"] & split_scores["test"]

    stop_reasons = [
        "Published MusicXML labels piano grand-staff crops, not every visible "
        "voice, lyric, title, direction, and page-level relation.",
        "The full pages are voice-and-piano Lieder with semantic lyrics, while "
        "production-v2 excludes lyric transcription and cannot certify such "
        "pages as complete in-boundary output.",
        "Only one piano-plus-vocal configuration is represented; solo "
        "instrument, instrumental ensemble, and piano-plus-instrument coverage "
        "remain absent.",
        "No independent human full-page completeness adjudication is recorded.",
        "Redistributable derived model packaging under CC BY-SA remains a "
        "separate release decision.",
    ]
    if document_overlap:
        stop_reasons.append(
            "Published dev/test score IDs are disjoint but reuse IMSLP source "
            "documents; source-document regrouping is required before an "
            "unseen-edition claim."
        )

    return {
        "format": 1,
        "created_at": utc_now_iso(),
        "name": "OLiMPiC full-page physical scan alignment audit",
        "role": ROLE,
        "boundary_contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
        "source_archive_asset_key": SOURCE_ASSET_KEY,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_archive_provenance_url": asset.get("provenance_url"),
        "source_archive_license": asset.get("license"),
        "source_archive_bytes": asset.get("downloaded_bytes"),
        "scanned_license_sha256": sha256_file(license_path),
        "scanned_readme_sha256": sha256_file(readme_path),
        "mapping_manifest_sha256": _mapping_digest(mapping_paths),
        "system_manifest_sha256": _mapping_digest(system_paths),
        "mapping_score_file_count": len(mapping_paths),
        "published_score_group_count": len(
            split_scores["dev"] | split_scores["test"]
        ),
        "published_sample_count": len(published_samples),
        "published_dev_sample_count": len(splits["dev"]),
        "published_test_sample_count": len(splits["test"]),
        "published_dev_score_group_count": len(split_scores["dev"]),
        "published_test_score_group_count": len(split_scores["test"]),
        "published_score_group_overlap_count": len(score_overlap),
        "extra_mapping_sample_count": len(extra_mappings),
        "extra_mapping_score_group_count": len(
            {sample.split("/", 1)[0] for sample in extra_mappings}
        ),
        "extra_mapping_samples": extra_mappings,
        "source_pdf_count": len(pdf_paths),
        "source_pdf_document_count": len(pdf_documents),
        "full_page_png_count": len(page_paths),
        "full_page_png_document_count": len(page_documents),
        "system_coordinate_document_count": len(system_documents),
        "mapped_source_document_count": len(mapped_documents),
        "mapped_full_page_count": len(mapped_pages),
        "mapped_system_count": len(mapped_systems),
        "duplicate_source_system_count": len(duplicate_source_systems),
        "published_dev_test_source_document_overlap_count": len(
            document_overlap
        ),
        "published_dev_test_source_document_overlap": sorted(document_overlap),
        "published_dev_test_source_page_overlap_count": len(page_overlap),
        "published_dev_test_source_system_overlap_count": len(system_overlap),
        "all_published_samples_mapped": True,
        "all_mapped_sample_triplets_present": True,
        "all_mapped_pages_present": True,
        "all_mapped_page_dimensions_match": True,
        "physical_full_page_assets_present": True,
        "page_system_geometry_present": True,
        "grandstaff_semantic_ground_truth_present": True,
        "full_page_semantic_ground_truth_complete": False,
        "semantic_lyrics_in_product_boundary": False,
        "work_disjoint_split_verified": len(score_overlap) == 0,
        "source_document_disjoint_split_verified": len(document_overlap) == 0,
        "independent_full_page_adjudication": False,
        "research_training_authorized": True,
        "existing_published_split_unseen_source_evaluation_authorized": (
            len(document_overlap) == 0
        ),
        "distributable_product_training_authorized": False,
        "internal_diagnostic_evaluation_authorized": True,
        "final_release_evidence_authorized": False,
        "release_authorized": False,
        "stop_reasons": stop_reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scanned-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        scanned_root=args.scanned_root,
        source_root=args.source_root,
        provenance_path=args.provenance,
    )
    atomic_write_json(args.output, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "published_sample_count",
                    "full_page_png_count",
                    "mapped_full_page_count",
                    "mapped_system_count",
                    "published_dev_test_source_document_overlap_count",
                    "final_release_evidence_authorized",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
