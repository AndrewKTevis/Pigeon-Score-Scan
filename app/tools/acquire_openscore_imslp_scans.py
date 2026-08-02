from __future__ import annotations

"""Acquire identity-verified IMSLP quartet scans without authorizing training."""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Callable, Mapping

import fitz

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_ROOT / "src"))

from scorescan.util import atomic_write_json, sha256_file, utc_now_iso  # noqa: E402

from app.tools.catalog_openscore_imslp_scan_candidates import (  # noqa: E402
    LICENSE_SHA256,
    REVISION,
)
from app.tools.probe_openscore_imslp_scan_sources import (  # noqa: E402
    ROLE as EVIDENCE_ROLE,
)
from app.tools.probe_archive_openscore_imslp_mirrors import (  # noqa: E402
    ROLE as ARCHIVE_MIRROR_ROLE,
)


ROLE = "downloaded_imslp_scan_bytes_not_aligned_or_training_authorized"
USER_AGENT = "ScoreScan-OMR-IMSLP-scan-acquirer/1"
MAXIMUM_PDF_BYTES = 256 * 1024 * 1024
MAXIMUM_PAGE_COUNT = 1000
CHUNK_BYTES = 1024 * 1024


def _safe_score_path(score_root: Path, relative_value: str) -> Path:
    relative = PurePosixPath(relative_value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or ":" in relative.parts[0]
    ):
        raise ValueError("unsafe OpenScore candidate path")
    candidate = score_root.joinpath(*relative.parts).resolve()
    root = score_root.resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError("OpenScore candidate path escapes the fixed corpus")
    return candidate


def _approved_scan_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    hostname = (parsed.hostname or "").casefold()
    imslp_host = hostname == "imslp.org" or hostname.endswith(".imslp.org")
    archive_host = (
        hostname == "archive.org"
        or hostname.endswith(".archive.org")
    )
    return (
        parsed.scheme == "https"
        and (imslp_host or archive_host)
        and parsed.path.casefold().endswith(".pdf")
    )


def _open_url(
    url: str,
    *,
    timeout_seconds: float,
) -> BinaryIO:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/pdf,application/octet-stream;q=0.9",
            "User-Agent": USER_AGENT,
        },
    )
    return urllib.request.urlopen(request, timeout=timeout_seconds)


def _response_url(response: BinaryIO, fallback: str) -> str:
    getter = getattr(response, "geturl", None)
    return str(getter()) if callable(getter) else fallback


def _response_header(response: BinaryIO, name: str) -> str:
    headers = getattr(response, "headers", {})
    getter = getattr(headers, "get", None)
    return str(getter(name, "")) if callable(getter) else ""


def download_pdf(
    url: str,
    destination: Path,
    *,
    timeout_seconds: float,
    opener: Callable[..., BinaryIO] = _open_url,
    maximum_bytes: int = MAXIMUM_PDF_BYTES,
) -> dict[str, object]:
    if not _approved_scan_url(url):
        raise ValueError("PDF URL is outside the approved scan archive hosts")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    md5_digest = hashlib.md5(usedforsecurity=False)
    sha1_digest = hashlib.sha1(usedforsecurity=False)
    byte_count = 0
    final_url = url
    content_type = ""
    try:
        with opener(url, timeout_seconds=timeout_seconds) as response:
            final_url = _response_url(response, url)
            if not _approved_scan_url(final_url):
                raise ValueError(
                    "PDF redirect left the approved scan archive hosts"
                )
            content_length = _response_header(response, "Content-Length")
            if content_length and int(content_length) > maximum_bytes:
                raise ValueError("PDF exceeds the configured byte limit")
            content_type = _response_header(response, "Content-Type")
            with temporary.open("wb") as output:
                while chunk := response.read(CHUNK_BYTES):
                    byte_count += len(chunk)
                    if byte_count > maximum_bytes:
                        raise ValueError("PDF exceeds the configured byte limit")
                    digest.update(chunk)
                    md5_digest.update(chunk)
                    sha1_digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        if byte_count < 8:
            raise ValueError("downloaded PDF is empty or truncated")
        with temporary.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise ValueError("downloaded asset is not a PDF")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "direct_pdf_url": url,
        "final_pdf_url": final_url,
        "pdf_path": str(destination),
        "pdf_sha256": digest.hexdigest(),
        "pdf_md5": md5_digest.hexdigest(),
        "pdf_sha1": sha1_digest.hexdigest(),
        "pdf_bytes": byte_count,
        "content_type": content_type,
    }


def inspect_pdf(path: Path) -> dict[str, object]:
    try:
        document = fitz.open(path)
    except Exception as error:
        raise ValueError(f"PDF parser rejected the asset: {error}") from error
    try:
        if not document.is_pdf:
            raise ValueError("downloaded asset is not a PDF document")
        if document.needs_pass:
            raise ValueError("encrypted PDF is not accepted")
        if not 1 <= document.page_count <= MAXIMUM_PAGE_COUNT:
            raise ValueError("PDF page count is outside the safety boundary")
        rotations: dict[str, int] = defaultdict(int)
        page_sizes: set[tuple[float, float]] = set()
        for page in document:
            rotations[str(int(page.rotation))] += 1
            page_sizes.add(
                (
                    round(float(page.rect.width), 3),
                    round(float(page.rect.height), 3),
                )
            )
        return {
            "actual_page_count": document.page_count,
            "intrinsic_rotation_counts": dict(sorted(rotations.items())),
            "page_size_count": len(page_sizes),
            "page_sizes_points": [
                {"width": width, "height": height}
                for width, height in sorted(page_sizes)
            ],
        }
    finally:
        document.close()


def _validated_candidates(
    evidence: Mapping[str, object],
    score_root: Path,
    semantic_split_report: Mapping[str, object],
) -> list[dict[str, object]]:
    candidates = evidence.get("verified_candidates")
    if not isinstance(candidates, list):
        raise ValueError("verified IMSLP candidates are missing")
    split_rows = semantic_split_report.get("sources")
    if (
        semantic_split_report.get("purpose")
        != "synthetic semantic geometry; not real-scan validation"
        or not isinstance(split_rows, list)
    ):
        raise ValueError("unexpected OpenScore semantic split contract")
    split_by_path: dict[str, dict[str, object]] = {}
    for split_row in split_rows:
        if not isinstance(split_row, dict):
            raise ValueError("invalid OpenScore semantic split row")
        source_key = str(split_row.get("source_key", ""))
        split = str(split_row.get("split", ""))
        if source_key in split_by_path or split not in {
            "train",
            "calibration",
            "test",
        }:
            raise ValueError("invalid OpenScore semantic source split")
        split_by_path[source_key] = split_row
    validated: list[dict[str, object]] = []
    for row in candidates:
        if (
            not isinstance(row, dict)
            or row.get("source_identity_verified") is not True
        ):
            raise ValueError("candidate bypassed the source identity gate")
        path = _safe_score_path(score_root, str(row.get("path", "")))
        expected_hash = str(row.get("sha256", ""))
        if (
            len(expected_hash) != 64
            or not path.is_file()
            or sha256_file(path) != expected_hash
        ):
            raise ValueError("fixed OpenScore transcription hash mismatch")
        relative_path = str(row.get("path", ""))
        split_row = split_by_path.get(relative_path)
        if (
            split_row is None
            or split_row.get("source_sha256") != expected_hash
        ):
            raise ValueError(
                "candidate is absent from the immutable semantic split"
            )
        validated_row = dict(row)
        validated_row["semantic_split"] = split_row["split"]
        validated.append(validated_row)
    return validated


def acquire(
    evidence_path: Path,
    score_root: Path,
    semantic_split_report_path: Path,
    output_dir: Path,
    manifest_path: Path,
    *,
    archive_mirror_evidence_path: Path | None = None,
    timeout_seconds: float,
    minimum_interval_seconds: float,
    limit: int | None,
    require_archive_mirror: bool = False,
    opener: Callable[..., BinaryIO] = _open_url,
) -> dict[str, object]:
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        evidence.get("role") != EVIDENCE_ROLE
        or evidence.get("training_authorized") is not False
        or evidence.get("evaluation_authorized") is not False
        or evidence.get("release_authorized") is not False
    ):
        raise ValueError("unexpected IMSLP evidence authorization contract")
    if not (score_root / "LICENSE.txt").is_file():
        raise ValueError("fixed OpenScore license is missing")
    if sha256_file(score_root / "LICENSE.txt") != LICENSE_SHA256:
        raise ValueError("fixed OpenScore license hash mismatch")
    semantic_split_report = json.loads(
        semantic_split_report_path.read_text(encoding="utf-8")
    )
    candidates = _validated_candidates(
        evidence,
        score_root,
        semantic_split_report,
    )
    sources_value = evidence.get("sources")
    if not isinstance(sources_value, list):
        raise ValueError("IMSLP source evidence is missing")
    sources = {
        str(row.get("imslp_source_id", "")): row
        for row in sources_value
        if isinstance(row, dict)
        and row.get("verified_public_domain_printed_scan") is True
    }
    archive_mirrors: dict[str, dict[str, object]] = {}
    archive_mirror_evidence_sha256 = ""
    if require_archive_mirror and archive_mirror_evidence_path is None:
        raise ValueError(
            "require_archive_mirror needs archive mirror evidence"
        )
    if archive_mirror_evidence_path is not None:
        mirror_evidence = json.loads(
            archive_mirror_evidence_path.read_text(encoding="utf-8")
        )
        if (
            mirror_evidence.get("role") != ARCHIVE_MIRROR_ROLE
            or mirror_evidence.get("training_authorized") is not False
            or mirror_evidence.get("evaluation_authorized") is not False
            or mirror_evidence.get("release_authorized") is not False
            or mirror_evidence.get("evidence_sha256")
            != sha256_file(evidence_path)
        ):
            raise ValueError("unexpected archive mirror evidence contract")
        mirror_rows = mirror_evidence.get("exact_mirror_candidates")
        if not isinstance(mirror_rows, list):
            raise ValueError("archive mirror candidates are missing")
        for mirror_row in mirror_rows:
            if not isinstance(mirror_row, dict):
                raise ValueError("invalid archive mirror candidate")
            source_id = str(mirror_row.get("imslp_source_id", ""))
            if source_id in archive_mirrors:
                raise ValueError("ambiguous archive mirror source")
            archive_mirrors[source_id] = mirror_row
        archive_mirror_evidence_sha256 = sha256_file(
            archive_mirror_evidence_path
        )
    candidates_by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in candidates:
        source_id = str(row.get("imslp_source_id", ""))
        source = sources.get(source_id)
        if (
            source is None
            or row.get("direct_pdf_url") != source.get("direct_pdf_url")
            or row.get("imslp_page_title") != source.get("page_title")
        ):
            raise ValueError("candidate/source evidence linkage mismatch")
        candidates_by_source[source_id].append(row)
    for source_id, source_candidates in candidates_by_source.items():
        source_splits = {
            str(row["semantic_split"]) for row in source_candidates
        }
        if len(source_splits) != 1:
            raise ValueError(
                f"IMSLP{source_id} crosses immutable semantic splits"
            )
    all_source_ids = sorted(candidates_by_source, key=int)
    source_ids = all_source_ids
    if require_archive_mirror:
        source_ids = [
            source_id
            for source_id in source_ids
            if source_id in archive_mirrors
        ]
        if not source_ids:
            raise ValueError(
                "archive mirror evidence contains no selected source"
            )
    mirror_filter_omitted_source_count = len(all_source_ids) - len(source_ids)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        source_ids = source_ids[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    assets: list[dict[str, object]] = []
    for position, source_id in enumerate(source_ids, start=1):
        source = sources[source_id]
        destination = output_dir / f"IMSLP{int(source_id):06d}.pdf"
        mirror = archive_mirrors.get(source_id)
        retrieval_url = (
            str(mirror["download_url"])
            if mirror is not None
            else str(source["direct_pdf_url"])
        )
        download = download_pdf(
            retrieval_url,
            destination,
            timeout_seconds=timeout_seconds,
            opener=opener,
        )
        if mirror is not None:
            expected_mirror = {
                "pdf_bytes": int(mirror["original_bytes"]),
                "pdf_md5": str(mirror["original_md5"]),
                "pdf_sha1": str(mirror["original_sha1"]),
            }
            mismatches = {
                key: {
                    "expected": expected,
                    "actual": download.get(key),
                }
                for key, expected in expected_mirror.items()
                if download.get(key) != expected
            }
            if mismatches:
                destination.unlink(missing_ok=True)
                raise ValueError(
                    f"IMSLP{source_id} archive mirror hash mismatch: "
                    f"{mismatches}"
                )
        inspection = inspect_pdf(destination)
        reported_count = int(source.get("page_count", 0))
        if inspection["actual_page_count"] != reported_count:
            destination.unlink(missing_ok=True)
            raise ValueError(
                f"IMSLP{source_id} page count differs from file evidence"
            )
        assets.append(
            {
                "imslp_source_id": source_id,
                **download,
                **inspection,
                "retrieval_channel": (
                    "internet_archive_exact_imslp_mirror"
                    if mirror is not None
                    else "imslp_direct"
                ),
                "archive_identifier": (
                    mirror.get("archive_identifier", "")
                    if mirror is not None
                    else ""
                ),
                "reported_page_count": reported_count,
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
                "candidate_work_fingerprints": sorted(
                    {
                        str(row["work_fingerprint"])
                        for row in candidates_by_source[source_id]
                    }
                ),
            }
        )
        print(f"[{position}/{len(source_ids)}] IMSLP scan acquired", flush=True)
        if position < len(source_ids) and minimum_interval_seconds > 0:
            time.sleep(minimum_interval_seconds)
    selected_candidates = [
        row
        for source_id in source_ids
        for row in candidates_by_source[source_id]
    ]
    report = {
        "format": 1,
        "created_at": utc_now_iso(),
        "name": "OpenScore quartet IMSLP scan byte manifest",
        "role": ROLE,
        "revision": REVISION,
        "evidence_path": str(evidence_path),
        "evidence_sha256": sha256_file(evidence_path),
        "score_root": str(score_root),
        "license_sha256": LICENSE_SHA256,
        "semantic_split_report_path": str(semantic_split_report_path),
        "semantic_split_report_sha256": sha256_file(
            semantic_split_report_path
        ),
        "archive_mirror_evidence_path": (
            str(archive_mirror_evidence_path)
            if archive_mirror_evidence_path is not None
            else ""
        ),
        "archive_mirror_evidence_sha256": (
            archive_mirror_evidence_sha256
        ),
        "archive_mirror_required": require_archive_mirror,
        "archive_mirror_filter_omitted_source_count": (
            mirror_filter_omitted_source_count
        ),
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_authorized": False,
        "authorization_reason": (
            "PDF bytes and source identity are verified, but exact page/measure "
            "alignment, production-boundary validation, and work-level split "
            "isolation remain incomplete"
        ),
        "source_count": len(assets),
        "candidate_count": len(selected_candidates),
        "work_count": len(
            {str(row["work_fingerprint"]) for row in selected_candidates}
        ),
        "page_count": sum(int(row["actual_page_count"]) for row in assets),
        "candidate_count_by_semantic_split": dict(
            sorted(
                Counter(
                    str(row["semantic_split"])
                    for row in selected_candidates
                ).items()
            )
        ),
        "candidates": selected_candidates,
        "assets": assets,
    }
    atomic_write_json(manifest_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_path", type=Path)
    parser.add_argument("score_root", type=Path)
    parser.add_argument("semantic_split_report", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("manifest_path", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--minimum-request-interval-seconds",
        type=float,
        default=2.5,
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--archive-mirror-evidence", type=Path)
    parser.add_argument(
        "--require-archive-mirror",
        action="store_true",
        help=(
            "acquire only sources with an exact mirror-evidence row; "
            "never fall back to IMSLP direct download"
        ),
    )
    args = parser.parse_args()
    if not 3.0 <= args.timeout_seconds <= 180.0:
        raise ValueError("timeout-seconds must be between 3 and 180")
    if not 0.0 <= args.minimum_request_interval_seconds <= 30.0:
        raise ValueError("minimum request interval must be between 0 and 30")
    report = acquire(
        args.evidence_path.resolve(),
        args.score_root.resolve(),
        args.semantic_split_report.resolve(),
        args.output_dir.resolve(),
        args.manifest_path.resolve(),
        archive_mirror_evidence_path=(
            args.archive_mirror_evidence.resolve()
            if args.archive_mirror_evidence is not None
            else None
        ),
        timeout_seconds=args.timeout_seconds,
        minimum_interval_seconds=args.minimum_request_interval_seconds,
        limit=args.limit,
        require_archive_mirror=args.require_archive_mirror,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "source_count",
                    "candidate_count",
                    "work_count",
                    "page_count",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
