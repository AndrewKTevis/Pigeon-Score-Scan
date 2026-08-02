from __future__ import annotations

"""Seed immutable Muse registration caches into a fresh partition output.

The destination receives only per-pair cache artifacts and page images.  It
never receives manifests or acceptance reports, so the normal preparation tool
must still validate every cache signature and regenerate all partition-level
evidence for the new selection.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.tools.prepare_muse_omr_scan_regions import (  # noqa: E402
    REGISTRATION_QUALITY_POLICY_VERSION,
    REGISTRATION_VERSION,
)
from scorescan.util import atomic_write_json, sha256_file, utc_now_iso  # noqa: E402


CACHE_SEED_VERSION = "muse-registration-cache-seed@1"
CACHE_DIRECTORIES = (
    "acceptances",
    "rejections",
    "pages",
)
OPTIONAL_CACHE_DIRECTORIES = ("reference_pages",)


def seed_registration_cache(
    source: Path,
    destination: Path,
) -> dict[str, object]:
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if source == destination:
        raise ValueError("cache source and destination must differ")
    source_report_path = source / "prepare-report.json"
    source_report = json.loads(
        source_report_path.read_text(encoding="utf-8")
    )
    if (
        not isinstance(source_report, dict)
        or source_report.get("registration_version") != REGISTRATION_VERSION
        or source_report.get("registration_quality_policy_version")
        != REGISTRATION_QUALITY_POLICY_VERSION
    ):
        raise ValueError("source registration cache uses a stale policy")
    if (destination / "prepare-report.json").exists():
        raise FileExistsError(
            "destination already has completed partition evidence"
        )

    destination.mkdir(parents=True, exist_ok=True)
    linked = 0
    copied = 0
    existing = 0
    file_count = 0
    total_bytes = 0
    missing_optional_directories: list[str] = []
    for directory_name in (*CACHE_DIRECTORIES, *OPTIONAL_CACHE_DIRECTORIES):
        source_directory = source / directory_name
        if not source_directory.is_dir():
            if directory_name in OPTIONAL_CACHE_DIRECTORIES:
                missing_optional_directories.append(directory_name)
                continue
            raise FileNotFoundError(
                f"source cache directory is missing: {source_directory}"
            )
        for source_file in sorted(
            path for path in source_directory.rglob("*") if path.is_file()
        ):
            relative = source_file.relative_to(source)
            output_file = destination / relative
            output_file.parent.mkdir(parents=True, exist_ok=True)
            size = source_file.stat().st_size
            file_count += 1
            total_bytes += size
            if output_file.exists():
                if output_file.stat().st_size != size:
                    raise FileExistsError(
                        f"existing cache artifact differs: {output_file}"
                    )
                if not os.path.samefile(source_file, output_file):
                    if sha256_file(source_file) != sha256_file(output_file):
                        raise FileExistsError(
                            f"existing cache artifact differs: {output_file}"
                        )
                existing += 1
                continue
            try:
                os.link(source_file, output_file)
                if not os.path.samefile(source_file, output_file):
                    raise RuntimeError(
                        f"hardlink identity check failed: {output_file}"
                    )
                linked += 1
            except OSError:
                shutil.copy2(source_file, output_file)
                if (
                    output_file.stat().st_size != size
                    or sha256_file(source_file) != sha256_file(output_file)
                ):
                    raise RuntimeError(
                        f"copied cache verification failed: {output_file}"
                    )
                copied += 1

    report = {
        "format": 1,
        "cache_seed_version": CACHE_SEED_VERSION,
        "created_at": utc_now_iso(),
        "source": str(source),
        "destination": str(destination),
        "source_prepare_report_sha256": sha256_file(source_report_path),
        "registration_version": REGISTRATION_VERSION,
        "registration_quality_policy_version": (
            REGISTRATION_QUALITY_POLICY_VERSION
        ),
        "partition_level_evidence_reused": False,
        "cache_signatures_must_be_revalidated": True,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "hardlinked": linked,
        "copied": copied,
        "already_present": existing,
        "missing_optional_directories": missing_optional_directories,
    }
    atomic_write_json(destination / "cache-seed-report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    report = seed_registration_cache(args.source, args.destination)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
