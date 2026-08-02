#!/usr/bin/env python3
"""Remove regenerable MuseScore reference renders after scan-text preparation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from app.tools.prepare_openscore_pdf_text import sha256_file
from app.tools.prepare_muse_omr_scan_text import (
    MINIMUM_SAFE_VISUAL_PRESENCE_NCC,
    SCAN_TEXT_PAGE_LABEL_COMPLETENESS_VERSION,
    SCAN_TEXT_REFERENCE_SOURCE_VERSION,
    SCAN_TEXT_VISUAL_PRESENCE_VERSION,
    validate_reference_page_source_evidence,
)


ALLOWED_REGION_NAMES = {
    "scorescan-muse-omr-registered-scan-regions-v1",
    "scorescan-muse-omr-registered-scan-holdout-v1",
}
ALLOWED_TEXT_NAMES = {
    "scorescan-muse-omr-registered-scan-text-v1",
    "scorescan-muse-omr-registered-scan-text-holdout-v1",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _cache_inventory(path: Path) -> tuple[int, int]:
    files = 0
    total_bytes = 0
    for item in path.rglob("*"):
        if item.is_symlink():
            raise ValueError(f"reference cache contains a symbolic link: {item}")
        if item.is_file():
            files += 1
            total_bytes += item.stat().st_size
    return files, total_bytes


def validate_completed_consumers(
    *,
    region_dir: Path,
    text_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    region_dir = region_dir.resolve(strict=True)
    text_dir = text_dir.resolve(strict=True)
    region_report_path = region_dir / "prepare-report.json"
    text_report_path = text_dir / "prepare-report.json"
    region_report = _load_json(region_report_path)
    text_report = _load_json(text_report_path)
    if region_report.get("name") not in ALLOWED_REGION_NAMES:
        raise ValueError("unrecognized Muse OMR region report")
    if text_report.get("name") not in ALLOWED_TEXT_NAMES:
        raise ValueError("unrecognized Muse OMR scan-text report")
    if text_report.get("role") != region_report.get("role"):
        raise ValueError("region/text roles do not match")
    if (
        text_report.get("scan_text_visual_presence_version")
        != SCAN_TEXT_VISUAL_PRESENCE_VERSION
        or float(text_report.get("minimum_visual_presence_ncc", -1.0))
        < MINIMUM_SAFE_VISUAL_PRESENCE_NCC
    ):
        raise ValueError(
            "scan-text consumer lacks current visual-presence evidence"
        )
    if (
        text_report.get("scan_text_page_label_completeness_version")
        != SCAN_TEXT_PAGE_LABEL_COMPLETENESS_VERSION
    ):
        raise ValueError("scan-text consumer contains partial page labels")
    reference_source_counts = validate_reference_page_source_evidence(
        text_report
    )
    if Path(str(text_report.get("region_dir", ""))).resolve() != region_dir:
        raise ValueError("scan-text report belongs to a different region dataset")
    if text_report.get("region_report_sha256") != sha256_file(
        region_report_path
    ):
        raise ValueError("scan-text report is not bound to this region report")
    if int(text_report.get("selected_pairs", -1)) != int(
        region_report.get("accepted_pairs", -2)
    ):
        raise ValueError("scan-text preparation did not cover every accepted pair")
    region_hash = sha256_file(region_report_path)
    text_hash = sha256_file(text_report_path)
    reference_dir = (region_dir / "reference_pages").resolve()
    if reference_dir.parent != region_dir:
        raise ValueError("unsafe reference-cache path")
    if not reference_dir.is_dir():
        prior_path = region_dir / "reference-cache-prune-report.json"
        if prior_path.is_file():
            prior = _load_json(prior_path)
            if (
                prior.get("name")
                == "scorescan-muse-omr-reference-cache-prune-v1"
                and prior.get("executed") is True
                and prior.get("region_report_sha256") == region_hash
                and prior.get("text_report_sha256") == text_hash
            ):
                return reference_dir, {**prior, "already_pruned": True}
        if (
            reference_source_counts["registered_reference_cache"] != 0
            or reference_source_counts["source_mscz_pdf_rerender"] <= 0
        ):
            raise ValueError(
                "missing reference cache has no valid prune report or "
                "cache-independent consumer evidence"
            )
        return reference_dir, {
            "format": 1,
            "name": "scorescan-muse-omr-reference-cache-prune-v1",
            "region_report_sha256": region_hash,
            "text_report_sha256": text_hash,
            "scan_text_reference_source_version": (
                SCAN_TEXT_REFERENCE_SOURCE_VERSION
            ),
            "removed_files": 0,
            "removed_bytes": 0,
            "already_pruned": True,
            "cache_absent_consumer_safe": True,
        }
    files, total_bytes = _cache_inventory(reference_dir)
    if files <= 0 or total_bytes <= 0:
        raise ValueError("reference cache is empty")
    return reference_dir, {
        "format": 1,
        "name": "scorescan-muse-omr-reference-cache-prune-v1",
        "region_report_sha256": region_hash,
        "text_report_sha256": text_hash,
        "removed_files": files,
        "removed_bytes": total_bytes,
        "already_pruned": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-dir", type=Path, required=True)
    parser.add_argument("--text-dir", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Delete the validated reference_pages directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reference_dir, report = validate_completed_consumers(
        region_dir=args.region_dir,
        text_dir=args.text_dir,
    )
    report["executed"] = bool(args.execute)
    if args.execute and reference_dir.is_dir():
        shutil.rmtree(reference_dir)
    if args.execute:
        report_path = args.region_dir.resolve() / "reference-cache-prune-report.json"
        temporary = report_path.with_suffix(report_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, report_path)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
