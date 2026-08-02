#!/usr/bin/env python3
"""Hardlink immutable OpenScore SVG/PNG renders into a fresh label build."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from app.tools.train_deepscores_symbol_detector import sha256_file


CACHE_SEED_VERSION = "openscore-render-cache-seed@1"


def seed_render_cache(source: Path, destination: Path) -> dict[str, Any]:
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if source == destination:
        raise ValueError("render-cache source and destination must differ")
    source_report = source / "prepare-report.json"
    source_pages = source / "pages"
    if not source_report.is_file() or not source_pages.is_dir():
        raise FileNotFoundError("source OpenScore render cache is incomplete")
    if (destination / "prepare-report.json").exists():
        raise FileExistsError("destination already has completed evidence")
    destination.mkdir(parents=True, exist_ok=True)

    linked = 0
    copied = 0
    existing = 0
    total_bytes = 0
    files = sorted(path for path in source_pages.rglob("*") if path.is_file())
    if not files:
        raise ValueError("source OpenScore render cache is empty")
    for source_file in files:
        relative = source_file.relative_to(source)
        output_file = destination / relative
        output_file.parent.mkdir(parents=True, exist_ok=True)
        size = source_file.stat().st_size
        total_bytes += size
        if output_file.exists():
            if (
                output_file.stat().st_size != size
                or (
                    not os.path.samefile(source_file, output_file)
                    and sha256_file(source_file) != sha256_file(output_file)
                )
            ):
                raise FileExistsError(
                    f"existing render-cache artifact differs: {output_file}"
                )
            existing += 1
            continue
        try:
            os.link(source_file, output_file)
            if not os.path.samefile(source_file, output_file):
                raise RuntimeError(f"hardlink verification failed: {output_file}")
            linked += 1
        except OSError:
            shutil.copy2(source_file, output_file)
            if (
                output_file.stat().st_size != size
                or sha256_file(source_file) != sha256_file(output_file)
            ):
                raise RuntimeError(f"copy verification failed: {output_file}")
            copied += 1
    report = {
        "format": 1,
        "cache_seed_version": CACHE_SEED_VERSION,
        "source": str(source),
        "destination": str(destination),
        "source_prepare_report_sha256": sha256_file(source_report),
        "partition_level_evidence_reused": False,
        "render_files": len(files),
        "total_bytes": total_bytes,
        "hardlinked": linked,
        "copied": copied,
        "already_present": existing,
    }
    report_path = destination / "render-cache-seed-report.json"
    temporary = report_path.with_name(
        f".{report_path.name}.{os.getpid()}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, report_path)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            seed_render_cache(args.source, args.destination),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
