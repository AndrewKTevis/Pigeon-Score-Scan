from __future__ import annotations

"""Acquire exact PDMX-linked IMSLP scans from the IMSLP mirror network."""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import BinaryIO, Callable, Mapping

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_ROOT / "src"))

from app.tools.acquire_openscore_imslp_scans import inspect_pdf  # noqa: E402
from app.tools.probe_pdmx_imslp_scan_sources import (  # noqa: E402
    ROLE as EVIDENCE_ROLE,
)
from scorescan.util import atomic_write_json, sha256_file, utc_now_iso  # noqa: E402


ROLE = "pdmx_linked_imslp_scan_bytes_not_aligned_or_training_authorized"
USER_AGENT = "ScoreScan-OMR-PDMX-IMSLP-scan-acquirer/1"
MIRROR_HOST = "conquest.imslp.info"
MAXIMUM_PDF_BYTES = 128 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024
SIZE_LABEL = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(KB|MB)", re.IGNORECASE)


def mirror_url(source: Mapping[str, object]) -> str:
    source_id = str(source.get("imslp_source_id", ""))
    direct = urllib.parse.urlparse(str(source.get("direct_pdf_url", "")))
    parts = direct.path.split("/")
    if (
        not source_id.isdigit()
        or direct.scheme != "https"
        or direct.hostname != "imslp.org"
        or len(parts) != 5
        or parts[1] != "images"
        or re.fullmatch(r"[0-9a-f]", parts[2], re.IGNORECASE) is None
        or re.fullmatch(r"[0-9a-f]{2}", parts[3], re.IGNORECASE) is None
    ):
        raise ValueError("invalid verified IMSLP direct PDF identity")
    filename = urllib.parse.unquote(parts[4])
    if (
        not filename.casefold().endswith(".pdf")
        or "/" in filename
        or "\\" in filename
    ):
        raise ValueError("unsafe IMSLP PDF filename")
    mirror_filename = f"IMSLP{source_id}-{filename}"
    return (
        f"http://{MIRROR_HOST}/files/imglnks/usimg/"
        f"{parts[2]}/{parts[3]}/"
        f"{urllib.parse.quote(mirror_filename)}"
    )


def _approved_mirror_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return bool(
        parsed.scheme == "http"
        and parsed.hostname == MIRROR_HOST
        and not parsed.query
        and not parsed.fragment
        and re.fullmatch(
            r"/files/imglnks/usimg/[0-9a-f]/[0-9a-f]{2}/"
            r"IMSLP[0-9]+-[^/\\]+\.pdf",
            urllib.parse.unquote(parsed.path),
            re.IGNORECASE,
        )
    )


def _open_url(url: str, *, timeout_seconds: float) -> BinaryIO:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/pdf",
            "User-Agent": USER_AGENT,
        },
    )
    return urllib.request.urlopen(request, timeout=timeout_seconds)


def _header(response: BinaryIO, name: str) -> str:
    getter = getattr(getattr(response, "headers", {}), "get", None)
    return str(getter(name, "")) if callable(getter) else ""


def _response_url(response: BinaryIO, fallback: str) -> str:
    getter = getattr(response, "geturl", None)
    return str(getter()) if callable(getter) else fallback


def size_matches_label(byte_count: int, label: str) -> bool:
    match = SIZE_LABEL.fullmatch(label.strip())
    if match is None or byte_count <= 0:
        return False
    numeric = match.group(1)
    value = float(numeric)
    unit = 1024 if match.group(2).casefold() == "kb" else 1024 * 1024
    decimals = len(numeric.partition(".")[2])
    half_step = 0.5 * (10 ** (-decimals))
    return (
        max(0.0, value - half_step) * unit
        <= byte_count
        < (value + half_step) * unit
    )


def download_pdf(
    url: str,
    destination: Path,
    *,
    timeout_seconds: float,
    opener: Callable[..., BinaryIO] = _open_url,
) -> dict[str, object]:
    if not _approved_mirror_url(url):
        raise ValueError("URL is not an exact approved IMSLP mirror path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pdf.part")
    temporary.unlink(missing_ok=True)
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    sha1 = hashlib.sha1(usedforsecurity=False)
    byte_count = 0
    content_type = ""
    final_url = url
    try:
        with opener(url, timeout_seconds=timeout_seconds) as response:
            final_url = _response_url(response, url)
            if final_url != url or not _approved_mirror_url(final_url):
                raise ValueError("IMSLP mirror redirected unexpectedly")
            content_type = _header(response, "Content-Type").split(";", 1)[0]
            if content_type.casefold() != "application/pdf":
                raise ValueError("IMSLP mirror did not return a PDF")
            content_length = _header(response, "Content-Length")
            if content_length:
                expected_length = int(content_length)
                if not 8 <= expected_length <= MAXIMUM_PDF_BYTES:
                    raise ValueError("IMSLP PDF content length is unsafe")
            else:
                expected_length = 0
            with temporary.open("wb") as output:
                while chunk := response.read(CHUNK_BYTES):
                    byte_count += len(chunk)
                    if byte_count > MAXIMUM_PDF_BYTES:
                        raise ValueError("IMSLP PDF exceeds byte limit")
                    sha256.update(chunk)
                    md5.update(chunk)
                    sha1.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        if expected_length and byte_count != expected_length:
            raise ValueError("IMSLP PDF content length mismatch")
        with temporary.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise ValueError("downloaded IMSLP asset is not a PDF")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "retrieval_url": url,
        "final_url": final_url,
        "pdf_path": str(destination),
        "pdf_bytes": byte_count,
        "pdf_sha256": sha256.hexdigest(),
        "pdf_md5": md5.hexdigest(),
        "pdf_sha1": sha1.hexdigest(),
        "content_type": content_type,
    }


def acquire(
    evidence_path: Path,
    output_dir: Path,
    manifest_path: Path,
    *,
    timeout_seconds: float,
    minimum_interval_seconds: float,
) -> dict[str, object]:
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    candidates = evidence.get("verified_candidates")
    sources_value = evidence.get("sources")
    if (
        evidence.get("role") != EVIDENCE_ROLE
        or evidence.get("training_authorized") is not False
        or evidence.get("evaluation_authorized") is not False
        or evidence.get("release_authorized") is not False
        or not isinstance(candidates, list)
        or not isinstance(sources_value, list)
    ):
        raise ValueError("unexpected PDMX IMSLP source evidence contract")
    sources = {
        str(source.get("imslp_source_id", "")): source
        for source in sources_value
        if isinstance(source, dict)
        and source.get("verified_public_domain_printed_scan") is True
    }
    selected_ids: set[str] = set()
    candidate_by_source: dict[str, list[int]] = {}
    for candidate in candidates:
        if (
            not isinstance(candidate, dict)
            or candidate.get("scan_source_identity_verified") is not True
        ):
            raise ValueError("candidate bypassed IMSLP identity verification")
        score_id = candidate.get("score_id")
        if isinstance(score_id, bool) or not isinstance(score_id, int):
            raise ValueError("invalid PDMX score id")
        source_ids = candidate.get("verified_imslp_source_ids")
        if not isinstance(source_ids, list) or not source_ids:
            raise ValueError("verified IMSLP source ids are missing")
        for source_id in source_ids:
            source_key = str(source_id)
            if source_key not in sources:
                raise ValueError("candidate/source evidence linkage mismatch")
            selected_ids.add(source_key)
            candidate_by_source.setdefault(source_key, []).append(score_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets: list[dict[str, object]] = []
    rejected_sources: list[dict[str, object]] = []
    ordered_ids = sorted(selected_ids, key=int)
    for position, source_id in enumerate(ordered_ids, start=1):
        source = sources[source_id]
        url = mirror_url(source)
        destination = output_dir / f"IMSLP{int(source_id):06d}.pdf"
        try:
            download = download_pdf(
                url,
                destination,
                timeout_seconds=timeout_seconds,
            )
            inspection = inspect_pdf(destination)
            reported_pages = int(source.get("page_count", 0) or 0)
            size_label = str(source.get("file_size_label", ""))
            if inspection["actual_page_count"] != reported_pages:
                raise ValueError(
                    "page count differs from source evidence"
                )
            if not size_matches_label(
                int(download["pdf_bytes"]),
                size_label,
            ):
                raise ValueError(
                    "byte count differs from source size label"
                )
        except (OSError, TimeoutError, ValueError) as error:
            destination.unlink(missing_ok=True)
            rejected_sources.append(
                {
                    "imslp_source_id": source_id,
                    "retrieval_url": url,
                    "reported_page_count": source.get("page_count", 0),
                    "reported_file_size_label": source.get(
                        "file_size_label",
                        "",
                    ),
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            print(
                f"[{position}/{len(ordered_ids)}] IMSLP{source_id} "
                "quarantined",
                flush=True,
            )
            if (
                position < len(ordered_ids)
                and minimum_interval_seconds > 0
            ):
                time.sleep(minimum_interval_seconds)
            continue
        assets.append(
            {
                "imslp_source_id": source_id,
                **download,
                **inspection,
                "reported_page_count": reported_pages,
                "reported_file_size_label": size_label,
                "source_page_sha256": source.get("page_sha256", ""),
                "source_page_title": source.get("page_title", ""),
                "public_domain_evidence": source.get(
                    "public_domain_evidence",
                    "",
                ),
                "scan_attribution_text": source.get(
                    "scan_attribution_text",
                    "",
                ),
                "pdmx_score_ids": sorted(
                    set(candidate_by_source[source_id])
                ),
                "transport_authenticated": False,
                "cross_mirror_hash_verified": False,
            }
        )
        print(
            f"[{position}/{len(ordered_ids)}] PDMX-linked IMSLP scan acquired",
            flush=True,
        )
        if position < len(ordered_ids) and minimum_interval_seconds > 0:
            time.sleep(minimum_interval_seconds)
    if not assets:
        raise ValueError("no PDMX-linked IMSLP scan passed byte gates")
    report = {
        "format": 1,
        "created_at": utc_now_iso(),
        "name": "PDMX-linked IMSLP scan byte manifest",
        "role": ROLE,
        "record_id": evidence.get("record_id"),
        "version": evidence.get("version"),
        "evidence_path": str(evidence_path),
        "evidence_sha256": sha256_file(evidence_path),
        "retrieval_channel": "official_imslp_mirror_exact_filename",
        "transport_authenticated": False,
        "cross_mirror_hash_verified": False,
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_authorized": False,
        "authorization_reason": (
            "PDF magic, parser, exact filename, source-page identity, displayed "
            "size, and page count are verified, but HTTP transport has no TLS "
            "or independent mirror hash; pinned MXL bytes, exact alignment, "
            "parsed boundary, and immutable work splits remain incomplete"
        ),
        "source_count": len(assets),
        "source_candidate_count": len(ordered_ids),
        "rejected_source_count": len(rejected_sources),
        "work_count": len(
            {
                score_id
                for asset in assets
                for score_id in asset["pdmx_score_ids"]
            }
        ),
        "page_count": sum(
            int(asset["actual_page_count"]) for asset in assets
        ),
        "assets": assets,
        "rejected_sources": rejected_sources,
    }
    atomic_write_json(manifest_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("manifest_path", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--minimum-request-interval-seconds",
        type=float,
        default=1.0,
    )
    args = parser.parse_args()
    if not 3 <= args.timeout_seconds <= 180:
        raise ValueError("timeout-seconds must be between 3 and 180")
    if not 0 <= args.minimum_request_interval_seconds <= 30:
        raise ValueError("minimum request interval must be between 0 and 30")
    report = acquire(
        args.evidence_path.resolve(),
        args.output_dir.resolve(),
        args.manifest_path.resolve(),
        timeout_seconds=args.timeout_seconds,
        minimum_interval_seconds=args.minimum_request_interval_seconds,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "source_count",
                    "source_candidate_count",
                    "rejected_source_count",
                    "work_count",
                    "page_count",
                    "transport_authenticated",
                    "training_authorized",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
