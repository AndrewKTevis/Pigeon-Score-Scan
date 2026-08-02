from __future__ import annotations

"""Acquire the pinned PDMX v9 metadata archive for provenance discovery only."""

import argparse
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import BinaryIO, Callable

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "src"))

from scorescan.util import atomic_write_json, sha256_file, utc_now_iso  # noqa: E402


RECORD_ID = "15571083"
VERSION = "v9"
URL = (
    "https://zenodo.org/api/records/15571083/files/"
    "metadata.tar.gz/content"
)
EXPECTED_BYTES = 159_444_765
EXPECTED_MD5 = "5bc79445090dd2fe5e96cffa77a3461c"
MAXIMUM_BYTES = 200 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024
ROLE = "pdmx_metadata_provenance_discovery_only_not_training"
USER_AGENT = "ScoreScan-PDMX-metadata-acquirer/1"


def _approved_url(url: str, *, approved_url: str = URL) -> bool:
    parsed = urllib.parse.urlparse(url)
    approved = urllib.parse.urlparse(approved_url)
    return (
        parsed.scheme == "https"
        and parsed.hostname == "zenodo.org"
        and parsed.scheme == approved.scheme
        and parsed.hostname == approved.hostname
        and parsed.path == approved.path
        and not parsed.query
        and not parsed.fragment
        and not approved.query
        and not approved.fragment
    )


def _open_url(
    url: str,
    *,
    timeout_seconds: float,
    offset: int,
) -> BinaryIO:
    headers = {
        # Zenodo's file-content endpoint returns 406 for a narrow media-range
        # even though the successful response is application/octet-stream.
        "Accept": "*/*",
        "User-Agent": USER_AGENT,
    }
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(request, timeout=timeout_seconds)


def _status(response: BinaryIO) -> int:
    value = getattr(response, "status", None)
    if isinstance(value, int):
        return value
    getter = getattr(response, "getcode", None)
    return int(getter()) if callable(getter) else 200


def _response_url(response: BinaryIO, fallback: str) -> str:
    getter = getattr(response, "geturl", None)
    return str(getter()) if callable(getter) else fallback


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(
    url: str,
    destination: Path,
    *,
    expected_bytes: int,
    expected_md5: str,
    timeout_seconds: float,
    opener: Callable[..., BinaryIO] = _open_url,
    approved_url: str = URL,
    maximum_bytes: int = MAXIMUM_BYTES,
    asset_name: str = "PDMX metadata",
) -> dict[str, object]:
    if not _approved_url(url, approved_url=approved_url):
        raise ValueError(f"{asset_name} URL is not the pinned Zenodo asset")
    if (
        expected_bytes <= 0
        or expected_bytes > maximum_bytes
        or len(expected_md5) != 32
    ):
        raise ValueError(f"invalid pinned {asset_name} identity")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        destination.is_file()
        and destination.stat().st_size == expected_bytes
    ):
        md5 = _md5_file(destination)
        if md5 == expected_md5:
            return {
                "path": str(destination),
                "bytes": expected_bytes,
                "md5": md5,
                "sha256": sha256_file(destination),
                "resumed_from_bytes": expected_bytes,
                "downloaded_bytes": 0,
            }

    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.is_file() and partial.stat().st_size > expected_bytes:
        partial.unlink()
    offset = partial.stat().st_size if partial.is_file() else 0
    md5_digest = hashlib.md5(usedforsecurity=False)
    sha256_digest = hashlib.sha256()
    if offset:
        with partial.open("rb") as existing:
            for chunk in iter(lambda: existing.read(CHUNK_BYTES), b""):
                md5_digest.update(chunk)
                sha256_digest.update(chunk)
    downloaded = 0
    try:
        with opener(
            url,
            timeout_seconds=timeout_seconds,
            offset=offset,
        ) as response:
            final_url = _response_url(response, url)
            if not _approved_url(final_url, approved_url=approved_url):
                raise ValueError(f"{asset_name} redirect left pinned Zenodo")
            status = _status(response)
            if offset and status != 206:
                raise ValueError("Zenodo did not honor the safe resume range")
            if not offset and status not in {200, 206}:
                raise ValueError(f"unexpected Zenodo response status {status}")
            mode = "ab" if offset else "wb"
            with partial.open(mode) as output:
                while chunk := response.read(CHUNK_BYTES):
                    downloaded += len(chunk)
                    if offset + downloaded > expected_bytes:
                        raise ValueError(
                            f"{asset_name} exceeded pinned byte count"
                        )
                    md5_digest.update(chunk)
                    sha256_digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        total = offset + downloaded
        md5 = md5_digest.hexdigest()
        if total != expected_bytes or md5 != expected_md5:
            partial.unlink(missing_ok=True)
            raise ValueError(f"{asset_name} size or MD5 mismatch")
        partial.replace(destination)
        return {
            "path": str(destination),
            "bytes": total,
            "md5": md5,
            "sha256": sha256_digest.hexdigest(),
            "resumed_from_bytes": offset,
            "downloaded_bytes": downloaded,
        }
    except BaseException:
        # Keep an in-bound partial file for an authenticated range retry.
        if partial.is_file() and partial.stat().st_size > expected_bytes:
            partial.unlink(missing_ok=True)
        raise


def acquire(
    output_dir: Path,
    report_path: Path,
    *,
    timeout_seconds: float,
) -> dict[str, object]:
    asset = download_archive(
        URL,
        output_dir / "metadata.tar.gz",
        expected_bytes=EXPECTED_BYTES,
        expected_md5=EXPECTED_MD5,
        timeout_seconds=timeout_seconds,
    )
    report = {
        "format": 1,
        "created_at": utc_now_iso(),
        "name": "PDMX v9 pinned metadata archive",
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
            "metadata is acquired only to discover public-domain symbolic "
            "scores with explicit scan-source provenance; no score or scan "
            "pair has yet passed identity, boundary, alignment, or split gates"
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
