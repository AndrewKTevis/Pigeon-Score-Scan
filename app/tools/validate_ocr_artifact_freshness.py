#!/usr/bin/env python3
"""Fail closed when a reusable OCR artifact no longer matches its inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from app.tools.ocr_text_contract import SOURCE_TEXT_SELECTION_VERSION
from app.tools.prepare_muse_omr_scan_text import (
    MINIMUM_SAFE_VISUAL_PRESENCE_NCC,
    SCAN_TEXT_PAGE_LABEL_COMPLETENESS_VERSION,
    SCAN_TEXT_REFERENCE_SOURCE_VERSION,
    SCAN_TEXT_VISUAL_PRESENCE_VERSION,
    validate_reference_page_source_evidence,
)


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _resolve_report_directory(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("artifact report has an invalid source directory")
    return Path(value).resolve()


def _validate_hash_manifest(artifact_report: Path) -> None:
    artifact_dir = artifact_report.parent.resolve()
    manifest = artifact_dir / "dataset.sha256"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    seen: set[Path] = set()
    for line_number, raw_line in enumerate(
        manifest.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            expected, relative_value = line.split(maxsplit=1)
        except ValueError as error:
            raise ValueError(
                f"{manifest}:{line_number}: malformed hash row"
            ) from error
        expected = expected.casefold()
        if not SHA256_PATTERN.fullmatch(expected):
            raise ValueError(
                f"{manifest}:{line_number}: invalid SHA-256 digest"
            )
        candidate = (artifact_dir / relative_value.strip()).resolve()
        try:
            candidate.relative_to(artifact_dir)
        except ValueError as error:
            raise ValueError(
                f"{manifest}:{line_number}: artifact escapes its directory"
            ) from error
        if candidate in seen:
            raise ValueError(
                f"{manifest}:{line_number}: duplicate artifact path"
            )
        seen.add(candidate)
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        if sha256_file(candidate) != expected:
            raise ValueError(f"artifact hash mismatch: {candidate}")
    if artifact_report.resolve() not in seen:
        raise ValueError("hash manifest does not cover the artifact report")


def validate_text_artifact(
    *,
    artifact_report: Path,
    source_report: Path,
) -> None:
    artifact = _load_json(artifact_report)
    if artifact.get("source_text_selection_version") != SOURCE_TEXT_SELECTION_VERSION:
        raise ValueError("scan-text artifact uses a stale text selection contract")
    if artifact.get("lyrics_included") is not False:
        raise ValueError("scan-text artifact includes out-of-boundary lyrics")
    if (
        artifact.get("scan_text_visual_presence_version")
        != SCAN_TEXT_VISUAL_PRESENCE_VERSION
        or float(artifact.get("minimum_visual_presence_ncc", -1.0))
        < MINIMUM_SAFE_VISUAL_PRESENCE_NCC
    ):
        raise ValueError(
            "scan-text artifact lacks current visual-presence evidence"
        )
    if (
        artifact.get("scan_text_page_label_completeness_version")
        != SCAN_TEXT_PAGE_LABEL_COMPLETENESS_VERSION
    ):
        raise ValueError("scan-text artifact contains partial page labels")
    validate_reference_page_source_evidence(artifact)
    source_report = source_report.resolve(strict=True)
    source_dir = source_report.parent
    if _resolve_report_directory(artifact.get("region_dir")) != source_dir:
        raise ValueError("scan-text artifact belongs to another region dataset")
    if artifact.get("region_report_sha256") != sha256_file(source_report):
        raise ValueError("scan-text artifact is stale for its region report")
    _validate_hash_manifest(artifact_report)


def validate_holdout_labels_artifact(
    *,
    artifact_report: Path,
    source_report: Path,
) -> None:
    artifact = _load_json(artifact_report)
    source_report = source_report.resolve(strict=True)
    if Path(str(artifact.get("source_report", ""))).resolve() != source_report:
        raise ValueError("holdout labels belong to another scan-text dataset")
    if artifact.get("source_report_sha256") != sha256_file(source_report):
        raise ValueError("holdout labels are stale for their scan-text report")
    output_hashes = artifact.get("output_sha256")
    output_counts = artifact.get("output_counts")
    if not isinstance(output_hashes, dict) or not isinstance(output_counts, dict):
        raise ValueError("holdout label integrity audit is missing")
    for name in ("test.paddle.txt", "test.paddle.det.txt"):
        expected = str(output_hashes.get(name, "")).casefold()
        path = artifact_report.parent / name
        if not SHA256_PATTERN.fullmatch(expected) or not path.is_file():
            raise ValueError(f"holdout label artifact is missing: {name}")
        if sha256_file(path) != expected:
            raise ValueError(f"holdout label hash mismatch: {name}")
        expected_count = int(output_counts.get(name, -1))
        with path.open(encoding="utf-8") as stream:
            actual_count = sum(1 for line in stream if line.strip())
        if expected_count < 1 or actual_count != expected_count:
            raise ValueError(f"holdout label row count mismatch: {name}")
    _validate_hash_manifest(artifact_report)


def validate_merged_artifact(
    *,
    artifact_report: Path,
    source_reports: list[Path],
) -> None:
    artifact = _load_json(artifact_report)
    datasets = artifact.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("merged OCR artifact has no source audit")
    expected = {
        report.resolve(strict=True).parent: sha256_file(report.resolve())
        for report in source_reports
    }
    if len(expected) != len(source_reports):
        raise ValueError("duplicate source reports were supplied")
    actual: dict[Path, str] = {}
    for row in datasets:
        if not isinstance(row, dict):
            raise ValueError("merged OCR source audit is malformed")
        directory = _resolve_report_directory(row.get("directory"))
        digest = str(row.get("prepare_report_sha256", "")).casefold()
        if directory in actual:
            raise ValueError("merged OCR source directory occurs more than once")
        actual[directory] = digest
    if set(actual) != set(expected):
        raise ValueError("merged OCR artifact uses a different source set")
    for directory, digest in expected.items():
        if actual[directory] != digest:
            raise ValueError(f"merged OCR artifact is stale for {directory}")
    _validate_hash_manifest(artifact_report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        choices=("text", "holdout-labels", "merged-labels"),
        required=True,
    )
    parser.add_argument("--artifact-report", type=Path, required=True)
    parser.add_argument(
        "--source-report",
        type=Path,
        action="append",
        required=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact_report = args.artifact_report.resolve(strict=True)
    source_reports = [
        source_report.resolve(strict=True)
        for source_report in args.source_report
    ]
    if args.kind == "text":
        if len(source_reports) != 1:
            raise ValueError("text validation requires exactly one source report")
        validate_text_artifact(
            artifact_report=artifact_report,
            source_report=source_reports[0],
        )
    elif args.kind == "holdout-labels":
        if len(source_reports) != 1:
            raise ValueError(
                "holdout-label validation requires exactly one source report"
            )
        validate_holdout_labels_artifact(
            artifact_report=artifact_report,
            source_report=source_reports[0],
        )
    else:
        validate_merged_artifact(
            artifact_report=artifact_report,
            source_reports=source_reports,
        )
    print(
        json.dumps(
            {
                "fresh": True,
                "kind": args.kind,
                "artifact_report": str(artifact_report),
                "source_reports": [str(path) for path in source_reports],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
