#!/usr/bin/env python3
"""Merge isolated exact-text datasets into auditable PaddleOCR label files.

The input datasets remain immutable.  This tool validates their source-level
splits and self-contained crops, then writes paths relative to the project
root so the same labels work in the WSL training environment.  Real-scan
training rows can be deterministically repeated to prevent a much larger clean
rendered corpus from overwhelming the target-domain examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from app.tools.ocr_text_contract import SOURCE_TEXT_SELECTION_VERSION
from app.tools.prepare_muse_omr_scan_text import (
    MINIMUM_SAFE_VISUAL_PRESENCE_NCC,
    SCAN_TEXT_PAGE_LABEL_COMPLETENESS_VERSION,
    SCAN_TEXT_REFERENCE_SOURCE_VERSION,
    SCAN_TEXT_VISUAL_PRESENCE_VERSION,
    validate_reference_page_source_evidence,
)


SPLITS = ("train", "calibration", "test")
EMPTY_INTERSECTIONS = {
    "train_calibration": [],
    "train_test": [],
    "calibration_test": [],
}
MAXIMUM_CLEAN_GEOMETRY_EXCLUSION_FRACTION = 0.005
SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
TRAINING_SCAN_ROLE = "training_only_disjoint_from_external_release_holdout"
INDEPENDENT_SCAN_HOLDOUT_ROLE = (
    "external_scan_degraded_development_benchmark_not_training"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    kind: str
    directory: Path


@dataclass(frozen=True)
class LabelRow:
    dataset: str
    kind: str
    split: str
    source_key: str
    project_relative_crop: str
    text: str


def parse_dataset_spec(value: str, *, kind: str) -> DatasetSpec:
    name, separator, directory = value.partition("=")
    if not separator or not SAFE_NAME.fullmatch(name):
        raise argparse.ArgumentTypeError(
            "dataset must be NAME=PATH with a lowercase safe NAME"
        )
    path = Path(directory).expanduser()
    if not directory:
        raise argparse.ArgumentTypeError("dataset path is empty")
    return DatasetSpec(name=name, kind=kind, directory=path)


def _validate_report(
    report: dict[str, Any],
    *,
    spec: DatasetSpec,
    allowed_scan_roles: tuple[str, ...] = (TRAINING_SCAN_ROLE,),
) -> dict[str, str]:
    if report.get("source_text_selection_version") != SOURCE_TEXT_SELECTION_VERSION:
        raise ValueError(
            f"{spec.name}: source text selection contract is stale or missing"
        )
    if report.get("lyrics_included") is not False:
        raise ValueError(f"{spec.name}: lyrics are outside the OCR product boundary")
    if report.get("split_intersections") != EMPTY_INTERSECTIONS:
        raise ValueError(f"{spec.name}: source split isolation failed")
    if spec.kind == "scan":
        role = report.get("role")
        if role not in allowed_scan_roles:
            raise ValueError(f"{spec.name}: scan data is not approved training data")
        if (
            role == INDEPENDENT_SCAN_HOLDOUT_ROLE
            and (
                report.get("forbidden_selection_overlap") != []
                or report.get("forbidden_work_overlap") != []
            )
        ):
            raise ValueError(
                f"{spec.name}: independent scan holdout overlaps training data"
            )
        if (
            report.get("scan_text_visual_presence_version")
            != SCAN_TEXT_VISUAL_PRESENCE_VERSION
            or float(report.get("minimum_visual_presence_ncc", -1.0))
            < MINIMUM_SAFE_VISUAL_PRESENCE_NCC
        ):
            raise ValueError(
                f"{spec.name}: scan text lacks the visual-presence contract"
            )
        if (
            report.get("scan_text_page_label_completeness_version")
            != SCAN_TEXT_PAGE_LABEL_COMPLETENESS_VERSION
        ):
            raise ValueError(
                f"{spec.name}: scan text contains potentially partial pages"
            )
        try:
            validate_reference_page_source_evidence(report)
        except ValueError as error:
            raise ValueError(
                f"{spec.name}: scan text lacks reference-source evidence"
            ) from error
    elif spec.kind == "clean":
        if (
            report.get("purpose")
            != "rendered exact-text OCR training/calibration"
            or report.get("self_contained_crops") is not True
        ):
            raise ValueError(
                f"{spec.name}: clean data lacks its self-contained training contract"
            )
        words_by_split = report.get("words_by_split")
        exclusions_by_split = report.get(
            "excluded_pdf_geometry_words_by_split"
        )
        if not isinstance(words_by_split, dict) or not isinstance(
            exclusions_by_split,
            dict,
        ):
            raise ValueError(
                f"{spec.name}: clean geometry exclusion audit is missing"
            )
        for split in SPLITS:
            retained = int(words_by_split.get(split, -1))
            reasons = exclusions_by_split.get(split)
            if retained < 0 or not isinstance(reasons, dict):
                raise ValueError(
                    f"{spec.name}: malformed clean geometry exclusion audit"
                )
            try:
                excluded = sum(int(value) for value in reasons.values())
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{spec.name}: malformed clean geometry exclusion count"
                ) from exc
            if excluded < 0 or any(int(value) < 0 for value in reasons.values()):
                raise ValueError(
                    f"{spec.name}: negative clean geometry exclusion count"
                )
            exclusion_fraction = excluded / max(1, retained + excluded)
            if (
                exclusion_fraction
                > MAXIMUM_CLEAN_GEOMETRY_EXCLUSION_FRACTION
            ):
                raise ValueError(
                    f"{spec.name}: {split} clean geometry exclusion fraction "
                    f"{exclusion_fraction:.6f} exceeds "
                    f"{MAXIMUM_CLEAN_GEOMETRY_EXCLUSION_FRACTION:.6f}"
                )
    else:
        raise ValueError(f"unsupported dataset kind: {spec.kind}")

    source_splits: dict[str, str] = {}
    content_identities: dict[str, tuple[str, str]] = {}
    sources = report.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"{spec.name}: source manifest is empty")
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError(f"{spec.name}: malformed source manifest row")
        source_key = str(source.get("source_key", ""))
        split = str(source.get("split", ""))
        if not source_key or split not in SPLITS:
            raise ValueError(f"{spec.name}: invalid source manifest row")
        previous = source_splits.setdefault(source_key, split)
        if previous != split:
            raise ValueError(
                f"{spec.name}: source {source_key!r} occurs in multiple splits"
            )
        hash_key = (
            "source_sha256"
            if spec.kind == "clean"
            else "work_fingerprint"
        )
        content_hash = str(source.get(hash_key, "")).casefold()
        if not re.fullmatch(r"[0-9a-f]{64}", content_hash):
            raise ValueError(
                f"{spec.name}: source {source_key!r} lacks a valid {hash_key}"
            )
        previous_identity = content_identities.setdefault(
            content_hash,
            (source_key, split),
        )
        if previous_identity != (source_key, split):
            raise ValueError(
                f"{spec.name}: duplicate source content {content_hash} occurs "
                f"as {previous_identity[0]!r}/{previous_identity[1]} and "
                f"{source_key!r}/{split}"
            )
    return source_splits


def _path_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _source_image_root(
    report: dict[str, Any],
    *,
    spec: DatasetSpec,
    project_root: Path,
) -> Path:
    key = "region_dataset_dir" if spec.kind == "clean" else "region_dir"
    value = report.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{spec.name}: preparation report lacks {key}")
    root = Path(value).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not _path_within(root, project_root):
        raise ValueError(f"{spec.name}: source image root escapes project")
    return root


def _integer_crop_box(
    row: dict[str, Any],
    *,
    width: int,
    height: int,
    location: str,
) -> tuple[int, int, int, int]:
    values = row.get("box_xyxy")
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError(f"{location}: source crop box is missing")
    try:
        left, top, right, bottom = (float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{location}: source crop box is invalid") from error
    if not all(math.isfinite(value) for value in (left, top, right, bottom)):
        raise ValueError(f"{location}: source crop box is non-finite")
    box = (
        max(0, math.floor(left)),
        max(0, math.floor(top)),
        min(width, math.ceil(right)),
        min(height, math.ceil(bottom)),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"{location}: source crop box is empty")
    return box


def load_dataset(
    spec: DatasetSpec,
    *,
    project_root: Path,
    allowed_scan_roles: tuple[str, ...] = (TRAINING_SCAN_ROLE,),
) -> tuple[dict[str, list[LabelRow]], dict[str, Any]]:
    dataset_dir = spec.directory.resolve()
    project_root = project_root.resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(dataset_dir)
    if not _path_within(dataset_dir, project_root):
        raise ValueError(
            f"{spec.name}: dataset must be contained by project root {project_root}"
        )
    report_path = dataset_dir / "prepare-report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    source_splits = _validate_report(
        report,
        spec=spec,
        allowed_scan_roles=allowed_scan_roles,
    )
    image_root = _source_image_root(
        report,
        spec=spec,
        project_root=project_root,
    )

    rows_by_split: dict[str, list[LabelRow]] = {split: [] for split in SPLITS}
    seen_crops: set[Path] = set()
    from PIL import Image, ImageChops

    cached_source_path: Path | None = None
    cached_source_image: Image.Image | None = None
    for split in SPLITS:
        jsonl_path = dataset_dir / f"{split}.jsonl"
        if not jsonl_path.is_file():
            raise FileNotFoundError(jsonl_path)
        with jsonl_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get("split", "")) != split:
                    raise ValueError(
                        f"{spec.name}:{jsonl_path.name}:{line_number}: split mismatch"
                    )
                source_key = str(row.get("source_key", ""))
                if source_splits.get(source_key) != split:
                    raise ValueError(
                        f"{spec.name}:{jsonl_path.name}:{line_number}: "
                        "source is missing or assigned to another split"
                    )
                text = str(row.get("text", ""))
                if (
                    not text.strip()
                    or any(character in text for character in "\t\r\n")
                ):
                    raise ValueError(
                        f"{spec.name}:{jsonl_path.name}:{line_number}: unsafe label"
                    )
                if spec.kind == "scan":
                    try:
                        visual_presence_ncc = float(
                            row["visual_presence_ncc"]
                        )
                        minimum_presence = float(
                            report["minimum_visual_presence_ncc"]
                        )
                    except (KeyError, TypeError, ValueError) as error:
                        raise ValueError(
                            f"{spec.name}:{jsonl_path.name}:{line_number}: "
                            "visual-presence evidence is missing"
                        ) from error
                    if (
                        not math.isfinite(visual_presence_ncc)
                        or visual_presence_ncc < minimum_presence
                    ):
                        raise ValueError(
                            f"{spec.name}:{jsonl_path.name}:{line_number}: "
                            "scan text is not visually present"
                        )
                if row.get("text_role") != "supported":
                    raise ValueError(
                        f"{spec.name}:{jsonl_path.name}:{line_number}: "
                        "label is not source-proven supported score text"
                    )
                crop_value = str(row.get("crop_image", ""))
                if not crop_value:
                    raise ValueError(
                        f"{spec.name}:{jsonl_path.name}:{line_number}: crop missing"
                    )
                crop_path = (dataset_dir / Path(crop_value)).resolve()
                if not _path_within(crop_path, dataset_dir):
                    raise ValueError(
                        f"{spec.name}:{jsonl_path.name}:{line_number}: "
                        "crop escapes dataset"
                    )
                if not crop_path.is_file() or crop_path.stat().st_size <= 0:
                    raise FileNotFoundError(crop_path)
                if crop_path in seen_crops:
                    raise ValueError(f"{spec.name}: duplicate crop {crop_path}")
                seen_crops.add(crop_path)
                image_value = str(row.get("image", ""))
                if not image_value:
                    raise ValueError(
                        f"{spec.name}:{jsonl_path.name}:{line_number}: "
                        "source page image is missing"
                    )
                source_path = (image_root / Path(image_value)).resolve()
                if not _path_within(source_path, image_root):
                    raise ValueError(
                        f"{spec.name}:{jsonl_path.name}:{line_number}: "
                        "source page escapes its dataset"
                    )
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                if source_path != cached_source_path:
                    if cached_source_image is not None:
                        cached_source_image.close()
                    with Image.open(source_path) as source:
                        cached_source_image = source.convert(
                            "RGB" if spec.kind == "clean" else "L"
                        )
                    cached_source_path = source_path
                assert cached_source_image is not None
                location = f"{spec.name}:{jsonl_path.name}:{line_number}"
                box = _integer_crop_box(
                    row,
                    width=cached_source_image.width,
                    height=cached_source_image.height,
                    location=location,
                )
                expected_crop = cached_source_image.crop(box)
                with Image.open(crop_path) as crop_source:
                    actual_crop = crop_source.convert(expected_crop.mode)
                if (
                    actual_crop.size != expected_crop.size
                    or ImageChops.difference(
                        actual_crop,
                        expected_crop,
                    ).getbbox()
                    is not None
                ):
                    raise ValueError(
                        f"{location}: crop pixels do not match the signed source page"
                    )
                rows_by_split[split].append(
                    LabelRow(
                        dataset=spec.name,
                        kind=spec.kind,
                        split=split,
                        source_key=source_key,
                        project_relative_crop=crop_path.relative_to(
                            project_root
                        ).as_posix(),
                        text=text,
                    )
                )
    if cached_source_image is not None:
        cached_source_image.close()

    expected_counts = report.get("words_by_split")
    actual_counts = {
        split: len(rows_by_split[split])
        for split in SPLITS
    }
    if not isinstance(expected_counts, dict) or any(
        int(expected_counts.get(split, -1)) != actual_counts[split]
        for split in SPLITS
    ):
        raise ValueError(
            f"{spec.name}: JSONL counts do not match the signed preparation report"
        )
    return rows_by_split, {
        "name": spec.name,
        "kind": spec.kind,
        "directory": str(dataset_dir),
        "prepare_report_sha256": sha256_file(report_path),
        "sources_by_split": {
            split: sum(value == split for value in source_splits.values())
            for split in SPLITS
        },
        "rows_by_split": actual_counts,
        "crop_source_pixel_bindings_verified": sum(actual_counts.values()),
    }


def _label_line(row: LabelRow) -> str:
    return f"{row.project_relative_crop}\t{row.text}"


def _write_labels(path: Path, rows: Iterable[LabelRow]) -> int:
    count = 0
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(_label_line(row))
            stream.write("\n")
            count += 1
    os.replace(temporary, path)
    return count


def build_balanced_training_rows(
    *,
    clean_rows: list[LabelRow],
    scan_rows: list[LabelRow],
    scan_target_fraction: float,
    seed: int,
    maximum_rows_per_normalized_text: int = 256,
) -> tuple[list[LabelRow], dict[str, Any]]:
    if not clean_rows or not scan_rows:
        raise ValueError("balanced training requires nonempty clean and scan rows")
    if not 0 < scan_target_fraction < 1:
        raise ValueError("scan target fraction must be between zero and one")
    if maximum_rows_per_normalized_text <= 0:
        raise ValueError("maximum rows per normalized text must be positive")

    def capped(rows: list[LabelRow], namespace: str) -> list[LabelRow]:
        groups: dict[str, list[LabelRow]] = {}
        for row in rows:
            key = " ".join(
                unicodedata.normalize("NFKC", row.text).casefold().split()
            )
            groups.setdefault(key, []).append(row)
        selected: list[LabelRow] = []
        for key, group in sorted(groups.items()):
            ranked = sorted(
                group,
                key=lambda row: hashlib.sha256(
                    (
                        f"{seed}\0{namespace}\0{key}\0"
                        f"{row.project_relative_crop}"
                    ).encode("utf-8")
                ).hexdigest(),
            )
            selected.extend(ranked[:maximum_rows_per_normalized_text])
        return selected

    capped_clean = capped(clean_rows, "clean")
    capped_scan = capped(scan_rows, "scan")

    required_scan = max(
        len(capped_scan),
        math.ceil(
            len(capped_clean)
            * scan_target_fraction
            / (1.0 - scan_target_fraction)
        ),
    )
    repeated_scan = [
        capped_scan[index % len(capped_scan)]
        for index in range(required_scan)
    ]
    combined = [*capped_clean, *repeated_scan]
    random.Random(seed).shuffle(combined)
    achieved = len(repeated_scan) / len(combined)
    return combined, {
        "available_clean_rows": len(clean_rows),
        "available_scan_rows": len(scan_rows),
        "unique_clean_rows": len(capped_clean),
        "unique_scan_rows": len(capped_scan),
        "frequency_cap_per_normalized_text": maximum_rows_per_normalized_text,
        "frequency_cap_dropped_clean_rows": len(clean_rows) - len(capped_clean),
        "frequency_cap_dropped_scan_rows": len(scan_rows) - len(capped_scan),
        "effective_scan_rows": len(repeated_scan),
        "scan_repeat_factor": round(len(repeated_scan) / len(capped_scan), 6),
        "target_scan_fraction": scan_target_fraction,
        "achieved_scan_fraction": round(achieved, 8),
        "seed": seed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--clean-dataset",
        action="append",
        default=[],
        type=lambda value: parse_dataset_spec(value, kind="clean"),
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--scan-dataset",
        action="append",
        default=[],
        type=lambda value: parse_dataset_spec(value, kind="scan"),
        metavar="NAME=PATH",
    )
    parser.add_argument("--scan-target-fraction", type=float, default=0.35)
    parser.add_argument(
        "--maximum-rows-per-normalized-text",
        type=int,
        default=256,
    )
    parser.add_argument("--seed", type=int, default=20260728)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    specs: list[DatasetSpec] = [*args.clean_dataset, *args.scan_dataset]
    if not args.clean_dataset or not args.scan_dataset:
        raise ValueError("at least one clean and one scan dataset are required")
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("dataset names must be unique")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaded: dict[str, dict[str, list[LabelRow]]] = {}
    dataset_reports = []
    global_source_splits: dict[tuple[str, str], str] = {}
    for spec in specs:
        rows_by_split, dataset_report = load_dataset(
            spec,
            project_root=args.project_root,
        )
        loaded[spec.name] = rows_by_split
        dataset_reports.append(dataset_report)
        for split in SPLITS:
            for row in rows_by_split[split]:
                key = (row.dataset, row.source_key)
                previous = global_source_splits.setdefault(key, split)
                if previous != split:
                    raise RuntimeError(
                        f"global source leakage: {row.dataset}/{row.source_key}"
                    )

    output_counts: dict[str, int] = {}
    clean_train: list[LabelRow] = []
    scan_train: list[LabelRow] = []
    for spec in specs:
        for split in SPLITS:
            rows = loaded[spec.name][split]
            filename = f"{split}.{spec.name}.paddle.txt"
            output_counts[filename] = _write_labels(
                args.output_dir / filename,
                rows,
            )
        destination = clean_train if spec.kind == "clean" else scan_train
        destination.extend(loaded[spec.name]["train"])

    balanced, balance_report = build_balanced_training_rows(
        clean_rows=clean_train,
        scan_rows=scan_train,
        scan_target_fraction=args.scan_target_fraction,
        seed=args.seed,
        maximum_rows_per_normalized_text=args.maximum_rows_per_normalized_text,
    )
    balanced_path = args.output_dir / "train.balanced.paddle.txt"
    output_counts[balanced_path.name] = _write_labels(balanced_path, balanced)

    for split in ("calibration", "test"):
        for kind in ("clean", "scan"):
            rows = [
                row
                for spec in specs
                if spec.kind == kind
                for row in loaded[spec.name][split]
            ]
            if not rows:
                raise ValueError(f"{kind} {split} split is empty")
            filename = f"{split}.{kind}.paddle.txt"
            output_counts[filename] = _write_labels(
                args.output_dir / filename,
                rows,
            )

    report = {
        "schema_version": 1,
        "name": "scorescan-ppocrv6-domain-training-labels-v1",
        "role": "training_only_disjoint_from_release_benchmarks",
        "project_root": str(args.project_root.resolve()),
        "datasets": dataset_reports,
        "global_split_intersections": EMPTY_INTERSECTIONS,
        "balance": balance_report,
        "output_counts": output_counts,
    }
    report_path = args.output_dir / "merge-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    hashed_paths = [
        *sorted(args.output_dir.glob("*.paddle.txt")),
        report_path,
    ]
    (args.output_dir / "dataset.sha256").write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}"
            for path in hashed_paths
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["output_counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
