from __future__ import annotations

"""Acquire the pinned PDMX v9 CSV used to reject license conflicts."""

import argparse
import json
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_ROOT / "src"))

from app.tools.acquire_pdmx_metadata_archive import download_archive  # noqa: E402
from scorescan.util import atomic_write_json, utc_now_iso  # noqa: E402


RECORD_ID = "15571083"
VERSION = "v9"
URL = "https://zenodo.org/api/records/15571083/files/PDMX.csv/content"
EXPECTED_BYTES = 225_399_738
EXPECTED_MD5 = "30392ccf38bb63ce70e7afae70f9c88c"
MAXIMUM_BYTES = 300 * 1024 * 1024
ROLE = "pdmx_license_conflict_filter_only_not_training"


def acquire(
    output_dir: Path,
    report_path: Path,
    *,
    timeout_seconds: float,
) -> dict[str, object]:
    asset = download_archive(
        URL,
        output_dir / "PDMX.csv",
        expected_bytes=EXPECTED_BYTES,
        expected_md5=EXPECTED_MD5,
        timeout_seconds=timeout_seconds,
        approved_url=URL,
        maximum_bytes=MAXIMUM_BYTES,
        asset_name="PDMX license table",
    )
    report = {
        "format": 1,
        "created_at": utc_now_iso(),
        "name": "PDMX v9 pinned license-conflict table",
        "role": ROLE,
        "record_id": RECORD_ID,
        "version": VERSION,
        "record_url": f"https://zenodo.org/records/{RECORD_ID}",
        "download_url": URL,
        "expected_bytes": EXPECTED_BYTES,
        "expected_md5": EXPECTED_MD5,
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_authorized": False,
        "authorization_reason": (
            "the CSV is only a provenance and license-conflict filter; "
            "candidate score bytes, source scans, musical boundary, exact "
            "work identity, alignment, and immutable splits remain unverified"
        ),
        "asset": asset,
    }
    atomic_write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("report_path", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()
    if not 10 <= args.timeout_seconds <= 1800:
        raise ValueError("timeout-seconds must be between 10 and 1800")
    report = acquire(
        args.output_dir.resolve(),
        args.report_path.resolve(),
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(report["asset"], indent=2))


if __name__ == "__main__":
    main()
