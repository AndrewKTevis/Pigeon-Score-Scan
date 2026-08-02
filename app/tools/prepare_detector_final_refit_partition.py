#!/usr/bin/env python3
"""Prepare a leakage-safe final-refit partition for the semantic detector.

The previously used internal test fold may join training only after architecture,
matcher, epoch-window, and sampling decisions are frozen.  The untouched
calibration fold becomes a diagnostic test fold.  This is standard final refit,
not release evidence: the forbidden external holdout still controls candidate
acceptance and production-v2 still requires physical scans.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.tools.muse_omr_contract import (  # noqa: E402
    SCAN_DEGRADED_IMAGE_ORIGIN,
    TRAINING_REGION_ROLE,
)
from app.tools.prepare_openscore_svg_regions import (  # noqa: E402
    COMPLETE_PAGE_TARGET_PROVENANCE,
)
from app.tools.semantic_target_visibility import (  # noqa: E402
    OVERSIZED_FRAGMENT_VISIBILITY_VERSION,
)
from scorescan.util import atomic_write_json, sha256_file, utc_now_iso  # noqa: E402


FINAL_REFIT_PARTITION_CONTRACT = (
    "semantic-detector-final-refit-train-plus-used-test-calibration-diagnostic@1"
)
REQUIRED_SPLITS = ("train", "calibration", "test")
EMPTY_INTERSECTIONS = {
    "train_calibration": [],
    "train_test": [],
    "calibration_test": [],
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _load_rows(path: Path, expected_split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                not isinstance(row, dict)
                or row.get("split") != expected_split
                or not str(row.get("source_key", "")).strip()
                or not isinstance(row.get("objects"), list)
            ):
                raise ValueError(
                    f"{path.name}:{line_number} violates its split contract"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"source split is empty: {path}")
    return rows


def _source_keys(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row["source_key"]) for row in rows}


def _split_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "tiles": len(rows),
        "sources": len(_source_keys(rows)),
        "negative_tiles": sum(not row["objects"] for row in rows),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    with path.open("xb") as stream:
        for row in rows:
            encoded = (
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            stream.write(encoded)
            digest.update(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return digest.hexdigest()


def prepare_final_refit_partition(
    source_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    manifest_path = source_dir / "manifest.json"
    report_path = source_dir / "prepare-report.json"
    categories_path = source_dir / "categories.json"
    manifest = _load_json(manifest_path)
    preparation = _load_json(report_path)
    if (
        manifest.get("role") != TRAINING_REGION_ROLE
        or manifest.get("source_split_overlap") != 0
        or preparation.get("split_intersections") != EMPTY_INTERSECTIONS
        or manifest.get("target_geometry_provenance")
        != COMPLETE_PAGE_TARGET_PROVENANCE
        or preparation.get("target_geometry_provenance")
        != COMPLETE_PAGE_TARGET_PROVENANCE
        or manifest.get("oversized_fragment_visibility_version")
        != OVERSIZED_FRAGMENT_VISIBILITY_VERSION
        or preparation.get("oversized_fragment_visibility_version")
        != OVERSIZED_FRAGMENT_VISIBILITY_VERSION
    ):
        raise ValueError("source semantic partition is not final-refit eligible")
    if not categories_path.is_file():
        raise FileNotFoundError(categories_path)

    source_rows = {
        split: _load_rows(source_dir / f"{split}.jsonl", split)
        for split in REQUIRED_SPLITS
    }
    source_keys = {
        split: _source_keys(rows)
        for split, rows in source_rows.items()
    }
    if (
        source_keys["train"] & source_keys["calibration"]
        or source_keys["train"] & source_keys["test"]
        or source_keys["calibration"] & source_keys["test"]
    ):
        raise ValueError("source semantic partition overlaps by source group")

    refit_train = [
        {
            **row,
            "split": "train",
            "final_refit_origin_split": origin_split,
        }
        for origin_split in ("train", "test")
        for row in source_rows[origin_split]
    ]
    diagnostic_test = [
        {
            **row,
            "split": "test",
            "final_refit_origin_split": "calibration",
        }
        for row in source_rows["calibration"]
    ]
    if _source_keys(refit_train) & _source_keys(diagnostic_test):
        raise ValueError("final-refit train/diagnostic split overlap")

    staging = output_dir.with_name(
        f"{output_dir.name}.building-{os.getpid()}-{uuid.uuid4().hex}"
    )
    staging.mkdir(parents=True)
    try:
        train_hash = _write_jsonl(staging / "train.jsonl", refit_train)
        test_hash = _write_jsonl(staging / "test.jsonl", diagnostic_test)
        shutil.copyfile(categories_path, staging / "categories.json")
        source_hashes = {
            split: sha256_file(source_dir / f"{split}.jsonl")
            for split in REQUIRED_SPLITS
        }
        output_manifest = {
            **manifest,
            "name": f"{manifest.get('name', 'semantic-regions')}-final-refit",
            "created_at": utc_now_iso(),
            "final_refit_partition_contract": FINAL_REFIT_PARTITION_CONTRACT,
            "model_outputs_used_for_partition": False,
            "external_holdout_reused_for_partition": False,
            "release_evidence_eligible": False,
            "source_image_origin": SCAN_DEGRADED_IMAGE_ORIGIN,
            "production_evidence_eligible": False,
            "source_manifest_sha256": sha256_file(manifest_path),
            "source_prepare_report_sha256": sha256_file(report_path),
            "source_split_jsonl_sha256": source_hashes,
            "train": _split_summary(refit_train),
            "test": _split_summary(diagnostic_test),
            "source_split_overlap": 0,
            "transformed_splits": ["train", "test"],
            "train_jsonl_sha256": train_hash,
            "test_jsonl_sha256": test_hash,
        }
        output_manifest.pop("calibration", None)
        output_report = {
            **preparation,
            "name": (
                f"{preparation.get('name', 'semantic-regions')}-final-refit"
            ),
            "created_at": utc_now_iso(),
            "purpose": (
                "final refit after model decisions were frozen; prior internal "
                "test joins training; prior calibration is diagnostic only; "
                "not external holdout or physical-scan release evidence"
            ),
            "final_refit_partition_contract": FINAL_REFIT_PARTITION_CONTRACT,
            "model_outputs_used_for_partition": False,
            "external_holdout_reused_for_partition": False,
            "source_image_origin": SCAN_DEGRADED_IMAGE_ORIGIN,
            "production_evidence_eligible": False,
            "source_manifest_sha256": sha256_file(manifest_path),
            "source_prepare_report_sha256": sha256_file(report_path),
            "source_split_jsonl_sha256": source_hashes,
            "tiles_by_split": {
                "train": len(refit_train),
                "test": len(diagnostic_test),
            },
            "source_count_by_split": {
                "train": len(_source_keys(refit_train)),
                "test": len(_source_keys(diagnostic_test)),
            },
            "split_intersections": {
                "train_test": [],
            },
            "train_jsonl_sha256": train_hash,
            "test_jsonl_sha256": test_hash,
        }
        atomic_write_json(staging / "manifest.json", output_manifest)
        atomic_write_json(staging / "prepare-report.json", output_report)
        os.replace(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return _load_json(output_dir / "prepare-report.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = prepare_final_refit_partition(
        args.source_dir,
        args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
