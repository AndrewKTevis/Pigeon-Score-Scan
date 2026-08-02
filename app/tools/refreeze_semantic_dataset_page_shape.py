#!/usr/bin/env python3
"""Refreeze a semantic dataset after a model-independent page-shape preflight.

The input layout evidence is hash-bound to the source prepared dataset.  Page
dimensions are independently re-read from the current images, and an entire
source work is excluded when any of its pages is a stitched scroll/panorama.
No model, prediction, label frequency, or evaluation metric participates in
the selection rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from app.tools.muse_omr_contract import (
    BENCHMARK_SELECTION_ROLE,
    MAXIMUM_SCAN_PAGE_ASPECT_RATIO,
    SCAN_DEGRADED_IMAGE_ORIGIN,
    SCAN_PAGE_SHAPE_CONTRACT,
    TRAINING_REGION_ROLE,
    scan_page_aspect_ratio,
)
from app.tools.prepare_openscore_svg_regions import (
    COMPLETE_PAGE_TARGET_ASSIGNMENT_VERSION,
    COMPLETE_PAGE_TARGET_PROVENANCE,
)
from app.tools.semantic_target_visibility import (
    OVERSIZED_FRAGMENT_VISIBILITY_VERSION,
)


PAGE_SHAPE_REFREEZE_CONTRACT = (
    "pre-inference-model-independent-source-page-shape-refreeze@1"
)
ALLOWED_ROLES = frozenset(
    {TRAINING_REGION_ROLE, BENCHMARK_SELECTION_ROLE}
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
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
                or not all(
                    str(row.get(field) or "").strip()
                    for field in ("source_key", "image", "image_id")
                )
                or not isinstance(row.get("objects"), list)
            ):
                raise ValueError(
                    f"{path.name}:{line_number} violates its split contract"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"source split is empty: {path}")
    return rows


def _page_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["source_key"]),
        str(row["image"]),
        str(row["image_id"]),
    )


def _resolve_image(images_dir: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise ValueError("semantic page image path must be relative")
    root = images_dir.resolve()
    image_path = (root / relative).resolve()
    try:
        image_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("semantic page image escapes its image root") from exc
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    return image_path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


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
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            stream.write(encoded)
            digest.update(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return digest.hexdigest()


def _split_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "tiles": len(rows),
        "sources": len({str(row["source_key"]) for row in rows}),
        "negative_tiles": sum(not row["objects"] for row in rows),
    }


def refreeze_dataset(
    source_dir: Path,
    images_dir: Path,
    layout_evidence_path: Path,
    output_dir: Path,
    splits: tuple[str, ...],
) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    images_dir = images_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(images_dir)
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("one or more unique dataset splits are required")
    if any(split not in {"train", "calibration", "test"} for split in splits):
        raise ValueError("unsupported semantic dataset split")

    manifest_path = source_dir / "manifest.json"
    report_path = source_dir / "prepare-report.json"
    categories_path = source_dir / "categories.json"
    manifest = _load_object(manifest_path)
    source_report = _load_object(report_path)
    evidence = _load_object(layout_evidence_path)
    role = str(manifest.get("role") or "")
    if (
        role not in ALLOWED_ROLES
        or source_report.get("role") != role
        or int(manifest.get("source_split_overlap", -1)) != 0
        or manifest.get("target_assignment_version")
        != COMPLETE_PAGE_TARGET_ASSIGNMENT_VERSION
        or manifest.get("target_geometry_provenance")
        != COMPLETE_PAGE_TARGET_PROVENANCE
        or manifest.get("oversized_fragment_visibility_version")
        != OVERSIZED_FRAGMENT_VISIBILITY_VERSION
        or source_report.get("transformation_version")
        != COMPLETE_PAGE_TARGET_ASSIGNMENT_VERSION
        or source_report.get("target_geometry_provenance")
        != COMPLETE_PAGE_TARGET_PROVENANCE
        or source_report.get("oversized_fragment_visibility_version")
        != OVERSIZED_FRAGMENT_VISIBILITY_VERSION
        or manifest.get("forbidden_selection_overlap") != []
        or manifest.get("forbidden_work_overlap") != []
        or source_report.get("forbidden_selection_overlap") != []
        or source_report.get("forbidden_work_overlap") != []
    ):
        raise ValueError("source semantic dataset is not refreeze-eligible")
    if not categories_path.is_file():
        raise FileNotFoundError(categories_path)

    rows_by_split = {
        split: _load_rows(source_dir / f"{split}.jsonl", split)
        for split in splits
    }
    split_hashes = {
        split: sha256_file(source_dir / f"{split}.jsonl")
        for split in splits
    }
    sources_by_split = {
        split: {str(row["source_key"]) for row in rows}
        for split, rows in rows_by_split.items()
    }
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            if sources_by_split[left] & sources_by_split[right]:
                raise ValueError(
                    f"source split leakage detected: {left}/{right}"
                )

    if (
        evidence.get("passed") is not True
        or evidence.get("prepared_manifest_sha256")
        != sha256_file(manifest_path)
        or evidence.get("split_jsonl_sha256") != split_hashes
        or evidence.get("target_assignment_version")
        != COMPLETE_PAGE_TARGET_ASSIGNMENT_VERSION
        or not isinstance(evidence.get("pages"), list)
    ):
        raise ValueError(
            "input page-layout evidence is stale or not hash-bound "
            "to the selected source splits"
        )

    expected_pages = {
        _page_key(row)
        for rows in rows_by_split.values()
        for row in rows
    }
    evidence_pages: dict[
        tuple[str, str, str],
        dict[str, Any],
    ] = {}
    for page in evidence["pages"]:
        if not isinstance(page, dict) or not isinstance(
            page.get("layout"),
            dict,
        ):
            raise ValueError("page-layout evidence contains an invalid row")
        key = (
            str(page.get("source_key") or ""),
            str(page.get("image") or ""),
            str(page.get("image_id") or ""),
        )
        if not all(key) or key in evidence_pages:
            raise ValueError(
                "page-layout evidence contains an unstable page identity"
            )
        evidence_pages[key] = page
    if set(evidence_pages) != expected_pages:
        raise ValueError(
            "page-layout evidence does not exactly cover dataset pages"
        )

    rejected_pages: list[dict[str, Any]] = []
    audited_pages: list[dict[str, Any]] = []
    for key in sorted(expected_pages):
        page = evidence_pages[key]
        layout = page["layout"]
        evidence_width = int(layout.get("width", 0))
        evidence_height = int(layout.get("height", 0))
        image_path = _resolve_image(images_dir, key[1])
        with Image.open(image_path) as image:
            width, height = image.size
        if (width, height) != (evidence_width, evidence_height):
            raise ValueError(
                "page dimensions changed since layout evidence: "
                f"{key!r}, evidence={evidence_width}x{evidence_height}, "
                f"current={width}x{height}"
            )
        ratio = scan_page_aspect_ratio(width, height)
        audit = {
            "source_key": key[0],
            "image": key[1],
            "image_id": key[2],
            "width": width,
            "height": height,
            "aspect_ratio": ratio,
        }
        audited_pages.append(audit)
        if ratio > MAXIMUM_SCAN_PAGE_ASPECT_RATIO:
            rejected_pages.append(audit)

    rejected_sources = {
        str(page["source_key"]) for page in rejected_pages
    }
    kept_rows = {
        split: [
            row
            for row in rows
            if str(row["source_key"]) not in rejected_sources
        ]
        for split, rows in rows_by_split.items()
    }
    if any(not rows for rows in kept_rows.values()):
        raise ValueError("page-shape refreeze emptied a requested split")
    kept_sources = {
        str(row["source_key"])
        for rows in kept_rows.values()
        for row in rows
    }
    source_sources = set().union(*sources_by_split.values())
    if kept_sources | rejected_sources != source_sources:
        raise RuntimeError("page-shape refreeze source accounting failed")
    if kept_sources & rejected_sources:
        raise RuntimeError("page-shape refreeze kept a rejected source")

    source_accepted = source_report.get("accepted")
    moved_pairs: list[dict[str, Any]] = []
    if isinstance(source_accepted, list):
        moved_pairs = [
            row
            for row in source_accepted
            if isinstance(row, dict)
            and str(row.get("source_key") or "") in rejected_sources
        ]
        accepted_pairs = int(
            source_report.get("accepted_pairs", len(source_accepted))
        ) - len(moved_pairs)
    else:
        accepted_pairs = len(kept_sources)
    rejected_pairs = int(source_report.get("rejected_pairs", 0)) + len(
        moved_pairs
    )

    staging = output_dir.with_name(
        f"{output_dir.name}.building-{os.getpid()}-{uuid.uuid4().hex}"
    )
    staging.mkdir(parents=True)
    try:
        output_split_hashes: dict[str, str] = {}
        for split, rows in kept_rows.items():
            output_split_hashes[split] = _write_jsonl(
                staging / f"{split}.jsonl",
                rows,
            )
        shutil.copyfile(categories_path, staging / "categories.json")
        summaries = {
            split: _split_summary(rows)
            for split, rows in kept_rows.items()
        }
        input_hashes = {
            "manifest": sha256_file(manifest_path),
            "prepare_report": sha256_file(report_path),
            "categories": sha256_file(categories_path),
            "layout_evidence": sha256_file(layout_evidence_path),
            **{
                f"{split}_jsonl": split_hashes[split]
                for split in splits
            },
        }
        target_instance_counts = Counter(
            str(obj.get("category_id") or "")
            for rows in kept_rows.values()
            for row in rows
            for obj in row["objects"]
        )
        output_manifest = {
            **manifest,
            "name": (
                f"{manifest.get('name', 'semantic-regions')}"
                "-ordinary-page-shape-refreeze"
            ),
            "source_image_origin": SCAN_DEGRADED_IMAGE_ORIGIN,
            "production_evidence_eligible": False,
            "page_shape_refreeze_contract": PAGE_SHAPE_REFREEZE_CONTRACT,
            "scan_page_shape_contract": SCAN_PAGE_SHAPE_CONTRACT,
            "maximum_scan_page_aspect_ratio": (
                MAXIMUM_SCAN_PAGE_ASPECT_RATIO
            ),
            "selection_rule_model_independent": True,
            "model_predictions_observed_for_refreeze": False,
            "source_artifact_sha256": input_hashes,
            "accepted_pairs": accepted_pairs,
            "accepted_works": len(kept_sources),
            "rejected_pairs": rejected_pairs,
            "source_split_overlap": 0,
            "transformed_splits": list(splits),
            "output_split_jsonl_sha256": output_split_hashes,
        }
        for split in ("train", "calibration", "test"):
            output_manifest[split] = summaries.get(
                split,
                {"tiles": 0, "sources": 0, "negative_tiles": 0},
            )

        intersections = {
            f"{left}_{right}": []
            for index, left in enumerate(("train", "calibration", "test"))
            for right in ("train", "calibration", "test")[index + 1 :]
        }
        output_report = {
            "schema_version": 1,
            "name": output_manifest["name"],
            "purpose": (
                "model-independent, pre-inference refreeze to exclude "
                "stitched whole-work scrolls and panoramas outside the "
                "ordinary scan-page product boundary"
            ),
            "role": role,
            "license": source_report.get("license"),
            "repository": source_report.get("repository"),
            "revision": source_report.get("revision"),
            "source_image_origin": SCAN_DEGRADED_IMAGE_ORIGIN,
            "production_evidence_eligible": False,
            "page_shape_refreeze_contract": PAGE_SHAPE_REFREEZE_CONTRACT,
            "scan_page_shape_contract": SCAN_PAGE_SHAPE_CONTRACT,
            "maximum_scan_page_aspect_ratio": (
                MAXIMUM_SCAN_PAGE_ASPECT_RATIO
            ),
            "selection_rule_model_independent": True,
            "model_predictions_observed_for_refreeze": False,
            "source_artifact_sha256": input_hashes,
            "target_assignment_version": (
                COMPLETE_PAGE_TARGET_ASSIGNMENT_VERSION
            ),
            "transformation_version": (
                COMPLETE_PAGE_TARGET_ASSIGNMENT_VERSION
            ),
            "target_geometry_provenance": (
                COMPLETE_PAGE_TARGET_PROVENANCE
            ),
            "oversized_fragment_visibility_version": (
                OVERSIZED_FRAGMENT_VISIBILITY_VERSION
            ),
            "tile_size": manifest.get("tile_size"),
            "overlap": source_report.get("overlap"),
            "minimum_object_fraction": source_report.get(
                "minimum_object_fraction"
            ),
            "long_span_minimum_object_fraction": source_report.get(
                "long_span_minimum_object_fraction"
            ),
            "selected_pairs": source_report.get("selected_pairs"),
            "selected_works": source_report.get("selected_works"),
            "accepted_pairs": accepted_pairs,
            "accepted_works": len(kept_sources),
            "rejected_pairs": rejected_pairs,
            "forbidden_selection_overlap": [],
            "forbidden_work_overlap": [],
            "tiles_by_split": {
                split: summaries.get(split, {}).get("tiles", 0)
                for split in ("train", "calibration", "test")
            },
            "negative_tiles_by_split": {
                split: summaries.get(split, {}).get("negative_tiles", 0)
                for split in ("train", "calibration", "test")
            },
            "source_count_by_split": {
                split: summaries.get(split, {}).get("sources", 0)
                for split in ("train", "calibration", "test")
            },
            "split_intersections": intersections,
            "source_pages": len(audited_pages),
            "accepted_pages": len(audited_pages) - len(rejected_pages),
            "rejected_pages": rejected_pages,
            "rejected_sources": sorted(rejected_sources),
            "rejected_source_count": len(rejected_sources),
            "removed_pairs": [
                {
                    "pair_id": row.get("pair_id"),
                    "source_key": row.get("source_key"),
                    "variant_key": row.get("variant_key"),
                    "reason": "source_contains_out_of_boundary_page_shape",
                }
                for row in moved_pairs
            ],
            "kept_tile_target_instance_counts": dict(
                sorted(target_instance_counts.items())
            ),
            "output_split_jsonl_sha256": output_split_hashes,
        }
        _write_json(staging / "manifest.json", output_manifest)
        _write_json(staging / "prepare-report.json", output_report)
        output_files = (
            ["categories.json", "manifest.json", "prepare-report.json"]
            + [f"{split}.jsonl" for split in splits]
        )
        (staging / "dataset.sha256").write_text(
            "\n".join(
                f"{sha256_file(staging / name)}  {name}"
                for name in output_files
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return _load_object(output_dir / "prepare-report.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--layout-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "calibration", "test"),
        required=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = refreeze_dataset(
        args.source_dir,
        args.images_dir,
        args.layout_evidence,
        args.output_dir,
        tuple(dict.fromkeys(args.split)),
    )
    print(
        json.dumps(
            {
                "accepted_works": report["accepted_works"],
                "rejected_source_count": report[
                    "rejected_source_count"
                ],
                "tiles_by_split": report["tiles_by_split"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
