from __future__ import annotations

"""Acquire externally published OMR corpora with provenance and safety checks.

The downloader intentionally separates immutable archives from extracted data and
records the exact bytes used by a training run.  It is not imported by the desktop
runtime and has no third-party dependency.
"""

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable


CATALOG_FORMAT = 1
PROVENANCE_FORMAT = 1
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "training_data" / "external"
USER_AGENT = "ScoreScan-OMR-corpus-acquirer/1"


@dataclass(frozen=True)
class CorpusAsset:
    key: str
    filename: str
    url: str
    expected_bytes: int
    license: str
    license_url: str
    provenance_url: str
    role: str
    extracted_directory: str
    license_review_required: bool = False


CATALOG: dict[str, CorpusAsset] = {
    "olimpic_scanned": CorpusAsset(
        key="olimpic_scanned",
        filename="olimpic-1.0-scanned.2024-02-12.tar.gz",
        url=(
            "https://github.com/ufal/olimpic-icdar24/releases/download/datasets/"
            "olimpic-1.0-scanned.2024-02-12.tar.gz"
        ),
        expected_bytes=225_607_163,
        license="CC-BY-SA",
        license_url="https://github.com/ufal/olimpic-icdar24#licenses",
        provenance_url="https://github.com/ufal/olimpic-icdar24/releases/tag/datasets",
        role="real_scan_development_and_frozen_candidate_benchmark",
        extracted_directory="olimpic_scanned",
    ),
    "olimpic_synthetic": CorpusAsset(
        key="olimpic_synthetic",
        filename="olimpic-1.0-synthetic.2024-02-12.tar.gz",
        url=(
            "https://github.com/ufal/olimpic-icdar24/releases/download/datasets/"
            "olimpic-1.0-synthetic.2024-02-12.tar.gz"
        ),
        expected_bytes=1_156_534_823,
        license="CC-BY-SA",
        license_url="https://github.com/ufal/olimpic-icdar24#licenses",
        provenance_url="https://github.com/ufal/olimpic-icdar24/releases/tag/datasets",
        role="pianoform_system_pretraining",
        extracted_directory="olimpic_synthetic",
    ),
    "olimpic_scanned_sources": CorpusAsset(
        key="olimpic_scanned_sources",
        filename="olimpic-1.0-sources-for-scanned.2024-02-12.tar.gz.tgz",
        url=(
            "https://github.com/ufal/olimpic-icdar24/releases/download/datasets/"
            "olimpic-1.0-sources-for-scanned.2024-02-12.tar.gz.tgz"
        ),
        expected_bytes=1_589_585_159,
        license="CC-BY-SA",
        license_url="https://github.com/ufal/olimpic-icdar24#licenses",
        provenance_url="https://github.com/ufal/olimpic-icdar24/releases/tag/datasets",
        role="real_scan_page_and_alignment_source",
        extracted_directory="olimpic_scanned_sources",
    ),
    "olimpic_zeus_model": CorpusAsset(
        key="olimpic_zeus_model",
        filename="zeus-olimpic-1.0-2024-02-12.model.tar.gz",
        url=(
            "https://github.com/ufal/olimpic-icdar24/releases/download/zeus-release/"
            "zeus-olimpic-1.0-2024-02-12.model.tar.gz"
        ),
        expected_bytes=27_051_365,
        license="CC-BY-SA",
        license_url="https://github.com/ufal/olimpic-icdar24#licenses",
        provenance_url="https://github.com/ufal/olimpic-icdar24/releases/tag/zeus-release",
        role="external_pianoform_baseline_only",
        extracted_directory="olimpic_zeus_model",
    ),
    "grandstaff_lmx": CorpusAsset(
        key="grandstaff_lmx",
        filename="grandstaff-lmx.2024-02-12.tar.gz",
        url=(
            "https://github.com/ufal/olimpic-icdar24/releases/download/datasets/"
            "grandstaff-lmx.2024-02-12.tar.gz"
        ),
        expected_bytes=68_620_885,
        license="CC-BY-SA",
        license_url="https://github.com/ufal/olimpic-icdar24#licenses",
        provenance_url="https://github.com/ufal/olimpic-icdar24/releases/tag/datasets",
        role="pianoform_musicxml_and_lmx_labels",
        extracted_directory="grandstaff_lmx",
    ),
    "doremi_v1": CorpusAsset(
        key="doremi_v1",
        filename="DoReMi_v1.zip",
        url="https://github.com/steinbergmedia/DoReMi/releases/download/v1.0/DoReMi_v1.zip",
        expected_bytes=109_775_551,
        license="REVIEW_REQUIRED",
        license_url="https://github.com/steinbergmedia/DoReMi",
        provenance_url="https://github.com/steinbergmedia/DoReMi/releases/tag/v1.0",
        role="symbol_detection_and_relation_pretraining",
        extracted_directory="doremi_v1",
        license_review_required=True,
    ),
    "deepscores_v2_dense": CorpusAsset(
        key="deepscores_v2_dense",
        filename="ds2_dense.tar.gz",
        url=(
            "https://zenodo.org/api/records/4012193/files/"
            "ds2_dense.tar.gz/content"
        ),
        expected_bytes=741_814_529,
        license="CC-BY-4.0",
        license_url="https://zenodo.org/records/4012193",
        provenance_url="https://doi.org/10.5281/zenodo.4012193",
        role="dense_music_symbol_detection_and_segmentation",
        extracted_directory="deepscores_v2_dense",
    ),
    "openscore_string_quartets": CorpusAsset(
        key="openscore_string_quartets",
        filename="openscore-string-quartets-d13289cd.zip",
        url=(
            "https://codeload.github.com/OpenScore/StringQuartets/zip/"
            "d13289cd70797da94646e5cf64f7296a4c4fee40"
        ),
        expected_bytes=31_800_158,
        license="CC0-1.0",
        license_url=(
            "https://github.com/OpenScore/StringQuartets/blob/"
            "d13289cd70797da94646e5cf64f7296a4c4fee40/LICENSE.txt"
        ),
        provenance_url=(
            "https://github.com/OpenScore/StringQuartets/tree/"
            "d13289cd70797da94646e5cf64f7296a4c4fee40"
        ),
        role="independent_multi_part_ensemble_rendering_and_semantic_training",
        extracted_directory="openscore_string_quartets_d13289cd",
    ),
    "openscore_lieder": CorpusAsset(
        key="openscore_lieder",
        filename="openscore-lieder-6b2dc542.zip",
        url=(
            "https://codeload.github.com/OpenScore/Lieder/zip/"
            "6b2dc542ce2e8aa4b78c8ee62103b210efc07015"
        ),
        expected_bytes=192_441_518,
        license="CC0-1.0",
        license_url=(
            "https://github.com/OpenScore/Lieder/blob/"
            "6b2dc542ce2e8aa4b78c8ee62103b210efc07015/LICENSE.txt"
        ),
        provenance_url=(
            "https://github.com/OpenScore/Lieder/tree/"
            "6b2dc542ce2e8aa4b78c8ee62103b210efc07015"
        ),
        role=(
            "voice_piano_text_and_semantic_training_with_olimpic_"
            "development_and_candidate_score_ids_excluded"
        ),
        extracted_directory="openscore_lieder_6b2dc542",
    ),
}


def sha256_file(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member_path(root: Path, archive_name: str) -> Path:
    normalized = archive_name.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if relative.is_absolute() or not relative.parts:
        raise ValueError(f"unsafe archive path: {archive_name!r}")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"unsafe archive path: {archive_name!r}")
    if ":" in relative.parts[0]:
        raise ValueError(f"unsafe archive path: {archive_name!r}")
    candidate = root.joinpath(*relative.parts)
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve(strict=False)
    if resolved_root != resolved_candidate and resolved_root not in resolved_candidate.parents:
        raise ValueError(f"archive path escapes extraction root: {archive_name!r}")
    return candidate


def _copy_bounded(source: BinaryIO, target: BinaryIO, remaining: list[int]) -> int:
    written = 0
    while chunk := source.read(1024 * 1024):
        remaining[0] -= len(chunk)
        if remaining[0] < 0:
            raise ValueError("archive exceeds configured extraction byte limit")
        target.write(chunk)
        written += len(chunk)
    return written


def _extract_zip(archive: Path, target: Path, max_bytes: int) -> tuple[int, int]:
    file_count = 0
    written = 0
    remaining = [max_bytes]
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            destination = _safe_member_path(target, info.filename)
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            # Unix symlinks stored in zip files must not be followed.
            mode = (info.external_attr >> 16) & 0xF000
            if mode == 0xA000:
                raise ValueError(f"archive symlink is not allowed: {info.filename!r}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, destination.open("wb") as output:
                written += _copy_bounded(source, output, remaining)
            file_count += 1
    return file_count, written


def _extract_tar(archive: Path, target: Path, max_bytes: int) -> tuple[int, int]:
    file_count = 0
    written = 0
    remaining = [max_bytes]
    with tarfile.open(archive, "r:*") as bundle:
        for info in bundle:
            destination = _safe_member_path(target, info.name)
            if info.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not info.isfile():
                raise ValueError(f"archive link or special file is not allowed: {info.name!r}")
            source = bundle.extractfile(info)
            if source is None:
                raise ValueError(f"archive member cannot be read: {info.name!r}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("wb") as output:
                written += _copy_bounded(source, output, remaining)
            file_count += 1
    return file_count, written


def extract_archive(archive: Path, target: Path, max_bytes: int) -> dict[str, int]:
    """Safely extract one trusted-but-validated external archive."""

    if target.exists():
        raise FileExistsError(f"refusing to overwrite extracted corpus: {target}")
    staging = target.with_name(f".{target.name}.extracting")
    if staging.exists():
        resolved = staging.resolve()
        expected_parent = target.parent.resolve()
        if resolved.parent != expected_parent or not resolved.name.endswith(".extracting"):
            raise ValueError(f"unsafe extraction staging path: {staging}")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        if zipfile.is_zipfile(archive):
            file_count, written = _extract_zip(archive, staging, max_bytes)
        elif tarfile.is_tarfile(archive):
            file_count, written = _extract_tar(archive, staging, max_bytes)
        else:
            raise ValueError(f"unsupported or corrupt archive: {archive}")
        if file_count <= 0:
            raise ValueError(f"archive contains no regular files: {archive}")
        staging.replace(target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {"files": file_count, "bytes": written}


def _download_once(asset: CorpusAsset, archive_path: Path, timeout: float) -> None:
    part_path = archive_path.with_suffix(archive_path.suffix + ".part")
    existing = part_path.stat().st_size if part_path.exists() else 0
    if existing > asset.expected_bytes:
        raise ValueError(f"partial archive is larger than expected: {part_path}")
    headers = {"User-Agent": USER_AGENT}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = urllib.request.Request(asset.url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = int(getattr(response, "status", response.getcode()) or 0)
        append = existing > 0 and status == 206
        mode = "ab" if append else "wb"
        with part_path.open(mode) as output:
            while chunk := response.read(4 * 1024 * 1024):
                output.write(chunk)
    actual_size = part_path.stat().st_size
    if actual_size != asset.expected_bytes:
        raise ValueError(
            f"download size mismatch for {asset.key}: "
            f"expected {asset.expected_bytes}, got {actual_size}"
        )
    os.replace(part_path, archive_path)


def download_asset(
    asset: CorpusAsset,
    archive_path: Path,
    *,
    retries: int = 4,
    timeout: float = 60.0,
) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        if archive_path.stat().st_size == asset.expected_bytes:
            return
        raise ValueError(f"existing archive has unexpected size: {archive_path}")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            _download_once(asset, archive_path, timeout)
            return
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {asset.key}") from last_error


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def acquire(
    keys: Iterable[str],
    root: Path,
    *,
    extract: bool,
    max_extract_bytes: int,
) -> dict[str, object]:
    archives = root / "archives"
    extracted = root / "corpora"
    provenance_path = root / "provenance.json"
    rows_by_key: dict[str, dict[str, object]] = {}
    if provenance_path.is_file():
        existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        if existing.get("format") != PROVENANCE_FORMAT:
            raise ValueError(f"unsupported provenance format: {provenance_path}")
        for row in existing.get("assets", []):
            if isinstance(row, dict) and isinstance(row.get("key"), str):
                rows_by_key[str(row["key"])] = row
    for key in keys:
        asset = CATALOG[key]
        archive_path = archives / asset.filename
        print(f"[download] {key}: {asset.url}", flush=True)
        download_asset(asset, archive_path)
        row: dict[str, object] = {
            **asdict(asset),
            "archive_path": str(archive_path.resolve()),
            "archive_sha256": sha256_file(archive_path),
            "downloaded_bytes": archive_path.stat().st_size,
        }
        if extract:
            target = extracted / asset.extracted_directory
            if target.exists():
                row["extraction"] = {
                    "status": "already_present",
                    "path": str(target.resolve()),
                }
            else:
                print(f"[extract] {key}: {target}", flush=True)
                statistics = extract_archive(archive_path, target, max_extract_bytes)
                row["extraction"] = {
                    "status": "extracted",
                    "path": str(target.resolve()),
                    **statistics,
                }
        rows_by_key[key] = row
        rows = [rows_by_key[name] for name in sorted(rows_by_key)]
        _atomic_json(
            provenance_path,
            {
                "format": PROVENANCE_FORMAT,
                "catalog_format": CATALOG_FORMAT,
                "assets": rows,
            },
        )
    rows = [rows_by_key[name] for name in sorted(rows_by_key)]
    return {
        "format": PROVENANCE_FORMAT,
        "catalog_format": CATALOG_FORMAT,
        "assets": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=sorted(CATALOG),
        default=list(CATALOG),
        help="corpus assets to acquire (default: all catalogued assets)",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument(
        "--max-extract-gb",
        type=float,
        default=40.0,
        help="hard uncompressed-byte limit per archive",
    )
    parser.add_argument("--list", action="store_true", help="print catalog and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        print(json.dumps({key: asdict(value) for key, value in CATALOG.items()}, indent=2))
        return 0
    if args.max_extract_gb <= 0:
        raise SystemExit("--max-extract-gb must be positive")
    payload = acquire(
        args.datasets or list(CATALOG),
        args.root.resolve(),
        extract=not args.no_extract,
        max_extract_bytes=int(args.max_extract_gb * 1024**3),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
