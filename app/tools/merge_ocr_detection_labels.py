#!/usr/bin/env python3
"""Build leakage-isolated PaddleOCR text-detection page labels.

Exact word boxes prepared from clean MuseScore renders and strictly registered
historical scans are grouped by page.  Every source assignment, image path,
image dimension, polygon, and transcription is validated before a label file
is written.  Registered-scan pages can be deterministically repeated so the
larger clean-rendered corpus does not overwhelm the target scan domain.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from app.tools.merge_ocr_training_labels import (
    EMPTY_INTERSECTIONS,
    MAXIMUM_CLEAN_GEOMETRY_EXCLUSION_FRACTION,
    MINIMUM_SAFE_VISUAL_PRESENCE_NCC,
    SCAN_TEXT_PAGE_LABEL_COMPLETENESS_VERSION,
    SCAN_TEXT_REFERENCE_SOURCE_VERSION,
    SCAN_TEXT_VISUAL_PRESENCE_VERSION,
    SOURCE_TEXT_SELECTION_VERSION,
    SPLITS,
    DatasetSpec,
    TRAINING_SCAN_ROLE,
    _path_within,
    _validate_report,
    parse_dataset_spec,
    sha256_file,
)
from app.tools.prepare_openscore_pdf_text import (
    EXHAUSTIVE_DETECTION_LABEL_CONTRACT,
)
from app.tools.prepare_muse_omr_scan_text import (
    EXHAUSTIVE_REGISTERED_DETECTION_PAGE_CONTRACT,
    EXHAUSTIVE_REGISTERED_DETECTION_SELECTION_POLICY,
    validate_reference_page_source_evidence,
)


@dataclass(frozen=True)
class DetectionAnnotation:
    transcription: str
    points: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]


@dataclass(frozen=True)
class PageLabel:
    dataset: str
    kind: str
    split: str
    source_key: str
    project_relative_image: str
    width: int
    height: int
    image_sha256: str
    annotations: tuple[DetectionAnnotation, ...]
    hard_negative_sampling_authorized: bool = False


def _validate_exhaustive_detection_report(
    report: dict[str, Any],
    *,
    spec: DatasetSpec,
) -> dict[str, str]:
    if spec.kind == "clean":
        expected_purpose = (
            "rendered exhaustive visible-text detection training/calibration"
        )
        expected_scope = "page_without_pdf_geometry_exclusions"
    elif spec.kind == "scan":
        expected_purpose = (
            "registered exhaustive visible-text detection "
            "training/calibration"
        )
        expected_scope = (
            "retained_registered_page_all_visible_text_verified"
        )
        if (
            report.get("role") != TRAINING_SCAN_ROLE
            or report.get("forbidden_selection_overlap") != []
            or report.get("forbidden_work_overlap") != []
            or report.get("scan_text_visual_presence_version")
            != SCAN_TEXT_VISUAL_PRESENCE_VERSION
            or float(report.get("minimum_visual_presence_ncc", -1.0))
            < MINIMUM_SAFE_VISUAL_PRESENCE_NCC
            or report.get("scan_text_reference_source_version")
            != SCAN_TEXT_REFERENCE_SOURCE_VERSION
            or report.get("detection_page_label_completeness_version")
            != EXHAUSTIVE_REGISTERED_DETECTION_PAGE_CONTRACT
            or report.get("detection_page_selection_policy")
            != EXHAUSTIVE_REGISTERED_DETECTION_SELECTION_POLICY
        ):
            raise ValueError(
                f"{spec.name}: registered exhaustive detection evidence "
                "is invalid"
            )
        try:
            validate_reference_page_source_evidence(report)
        except ValueError as error:
            raise ValueError(
                f"{spec.name}: registered reference-page evidence is invalid"
            ) from error
    else:
        raise ValueError(f"{spec.name}: unsupported dataset kind")
    if (
        report.get("detection_label_contract")
        != EXHAUSTIVE_DETECTION_LABEL_CONTRACT
        or report.get("purpose")
        != expected_purpose
        or report.get("all_usable_pdf_text_included") is not True
        or report.get("hard_negative_sampling_authorized") is not True
        or report.get("hard_negative_authorization_scope")
        != expected_scope
        or report.get("split_intersections") != EMPTY_INTERSECTIONS
    ):
        raise ValueError(
            f"{spec.name}: exhaustive detection label contract is invalid"
        )
    words_by_split = report.get("words_by_split")
    exclusions_by_split = report.get(
        "excluded_pdf_geometry_words_by_split"
    )
    text_pages_by_split = report.get("text_pages_by_split")
    authorized_pages_by_split = report.get(
        "hard_negative_authorized_pages_by_split"
    )
    excluded_pages_by_split = report.get(
        "geometry_excluded_text_pages_by_split"
    )
    if not isinstance(words_by_split, dict) or not isinstance(
        exclusions_by_split,
        dict,
    ) or not isinstance(text_pages_by_split, dict) or not isinstance(
        authorized_pages_by_split,
        dict,
    ) or not isinstance(
        excluded_pages_by_split,
        dict,
    ):
        raise ValueError(
            f"{spec.name}: exhaustive detection geometry audit is missing"
        )
    total_excluded = 0
    total_authorized_pages = 0
    for split in SPLITS:
        retained = int(words_by_split.get(split, -1))
        reasons = exclusions_by_split.get(split)
        if retained <= 0 or not isinstance(reasons, dict):
            raise ValueError(
                f"{spec.name}: exhaustive detection split is empty or malformed"
            )
        excluded = sum(int(value) for value in reasons.values())
        total_excluded += excluded
        if excluded < 0 or any(int(value) < 0 for value in reasons.values()):
            raise ValueError(
                f"{spec.name}: exhaustive detection exclusion count is invalid"
            )
        if (
            spec.kind == "clean"
            and
            excluded / max(1, retained + excluded)
            > MAXIMUM_CLEAN_GEOMETRY_EXCLUSION_FRACTION
        ):
            raise ValueError(
                f"{spec.name}: exhaustive detection geometry exclusions "
                "exceed the audited limit"
            )
        text_pages = int(text_pages_by_split.get(split, -1))
        authorized_pages = int(authorized_pages_by_split.get(split, -1))
        excluded_pages = int(excluded_pages_by_split.get(split, -1))
        if (
            text_pages <= 0
            or authorized_pages < 0
            or excluded_pages < 0
            or (
                spec.kind == "clean"
                and (
                    authorized_pages + excluded_pages != text_pages
                    or (excluded_pages == 0) != (excluded == 0)
                )
            )
            or (
                spec.kind == "scan"
                and authorized_pages != text_pages
            )
        ):
            raise ValueError(
                f"{spec.name}: exhaustive page authorization audit is invalid"
            )
        total_authorized_pages += authorized_pages
    globally_exhaustive = bool(
        spec.kind == "scan" or total_excluded == 0
    )
    if (
        total_authorized_pages <= 0
        or (report.get("precision_evaluation_authorized") is True)
        != globally_exhaustive
        or (report.get("hmean_evaluation_authorized") is True)
        != globally_exhaustive
        or (report.get("unlabelled_visible_text_may_be_present") is True)
        != (not globally_exhaustive)
    ):
        raise ValueError(
            f"{spec.name}: exhaustive global coverage flags are inconsistent"
        )

    source_splits: dict[str, str] = {}
    content_identities: dict[str, tuple[str, str]] = {}
    sources = report.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(
            f"{spec.name}: exhaustive detection source manifest is empty"
        )
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError(
                f"{spec.name}: exhaustive detection source row is malformed"
            )
        source_key = str(source.get("source_key", ""))
        split = str(source.get("split", ""))
        hash_key = (
            "source_sha256"
            if spec.kind == "clean"
            else "work_fingerprint"
        )
        source_hash = str(source.get(hash_key, "")).casefold()
        if (
            not source_key
            or split not in SPLITS
            or not re.fullmatch(r"[0-9a-f]{64}", source_hash)
        ):
            raise ValueError(
                f"{spec.name}: exhaustive detection source identity is invalid"
            )
        previous_split = source_splits.setdefault(source_key, split)
        if previous_split != split:
            raise ValueError(
                f"{spec.name}: exhaustive detection source crosses splits"
            )
        previous_identity = content_identities.setdefault(
            source_hash,
            (source_key, split),
        )
        if previous_identity != (source_key, split):
            raise ValueError(
                f"{spec.name}: exhaustive detection source content is duplicated"
            )
    return source_splits


def _image_root(
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
        raise ValueError(
            f"{spec.name}: image root must be contained by project root"
        )
    return root


def _safe_text(
    row: dict[str, Any],
    *,
    location: str,
) -> str:
    value = str(row.get("text", ""))
    if not value.strip() or any(character in value for character in "\t\r\n"):
        raise ValueError(f"{location}: unsafe transcription")
    return value


def _box_to_annotation(
    row: dict[str, Any],
    *,
    width: int,
    height: int,
    location: str,
) -> DetectionAnnotation:
    values = row.get("box_xyxy")
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError(f"{location}: box_xyxy must have four coordinates")
    try:
        left, top, right, bottom = (float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{location}: non-numeric text box") from error
    if not all(math.isfinite(value) for value in (left, top, right, bottom)):
        raise ValueError(f"{location}: non-finite text box")
    epsilon = 1e-3
    if (
        left < -epsilon
        or top < -epsilon
        or right > width + epsilon
        or bottom > height + epsilon
        or right - left < 2.0
        or bottom - top < 2.0
    ):
        raise ValueError(
            f"{location}: invalid/out-of-bounds text box "
            f"{(left, top, right, bottom)} for {width}x{height}"
        )
    left = min(float(width), max(0.0, left))
    top = min(float(height), max(0.0, top))
    right = min(float(width), max(0.0, right))
    bottom = min(float(height), max(0.0, bottom))
    return DetectionAnnotation(
        transcription=_safe_text(row, location=location),
        points=(
            (left, top),
            (right, top),
            (right, bottom),
            (left, bottom),
        ),
    )


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as source:
        source.verify()
    with Image.open(path) as source:
        width, height = source.size
    if width < 16 or height < 16:
        raise ValueError(f"text-detection page is too small: {path}")
    return int(width), int(height)


def load_dataset(
    spec: DatasetSpec,
    *,
    project_root: Path,
    allowed_scan_roles: tuple[str, ...] = (TRAINING_SCAN_ROLE,),
    required_nonempty_splits: tuple[str, ...] = SPLITS,
) -> tuple[dict[str, list[PageLabel]], dict[str, Any]]:
    project_root = project_root.resolve()
    dataset_dir = spec.directory.resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(dataset_dir)
    if not _path_within(dataset_dir, project_root):
        raise ValueError(f"{spec.name}: dataset escapes project root")
    report_path = dataset_dir / "prepare-report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    exhaustive_detection = (
        report.get("detection_label_contract")
        == EXHAUSTIVE_DETECTION_LABEL_CONTRACT
    )
    if exhaustive_detection:
        source_splits = _validate_exhaustive_detection_report(
            report,
            spec=spec,
        )
    else:
        source_splits = _validate_report(
            report,
            spec=spec,
            allowed_scan_roles=allowed_scan_roles,
        )
    image_root = _image_root(
        report,
        spec=spec,
        project_root=project_root,
    )

    annotations_by_page: dict[
        tuple[str, str, Path], list[DetectionAnnotation]
    ] = defaultdict(list)
    hard_negative_authorization_by_page: dict[
        tuple[str, str, Path], bool
    ] = {}
    dimensions: dict[Path, tuple[int, int]] = {}
    seen_words: set[tuple[str, str, int, int]] = set()
    actual_word_counts: dict[str, int] = {split: 0 for split in SPLITS}
    for split in SPLITS:
        jsonl_path = dataset_dir / f"{split}.jsonl"
        if not jsonl_path.is_file():
            raise FileNotFoundError(jsonl_path)
        with jsonl_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                location = f"{spec.name}:{jsonl_path.name}:{line_number}"
                row = json.loads(line)
                if str(row.get("split", "")) != split:
                    raise ValueError(f"{location}: split mismatch")
                source_key = str(row.get("source_key", ""))
                if source_splits.get(source_key) != split:
                    raise ValueError(
                        f"{location}: source is absent or assigned elsewhere"
                    )
                expected_role = (
                    "visible_text"
                    if exhaustive_detection
                    else "supported"
                )
                if row.get("text_role") != expected_role:
                    raise ValueError(
                        f"{location}: label does not satisfy its detection "
                        "coverage contract"
                    )
                image_value = str(row.get("image", ""))
                if not image_value:
                    raise ValueError(f"{location}: page image is missing")
                image_path = (image_root / Path(image_value)).resolve()
                if not _path_within(image_path, image_root):
                    raise ValueError(f"{location}: page image escapes root")
                if not image_path.is_file() or image_path.stat().st_size <= 0:
                    raise FileNotFoundError(image_path)
                if image_path not in dimensions:
                    dimensions[image_path] = _image_size(image_path)
                page_number = int(row.get("page", 0))
                word_index = int(row.get("word_index", -1))
                if page_number <= 0 or word_index < 0:
                    raise ValueError(f"{location}: invalid page/word index")
                word_key = (
                    source_key,
                    image_path.as_posix(),
                    page_number,
                    word_index,
                )
                if word_key in seen_words:
                    raise ValueError(f"{location}: duplicate word identity")
                seen_words.add(word_key)
                width, height = dimensions[image_path]
                annotation = _box_to_annotation(
                    row,
                    width=width,
                    height=height,
                    location=location,
                )
                annotations_by_page[(split, source_key, image_path)].append(
                    annotation
                )
                if exhaustive_detection:
                    authorization = row.get(
                        "hard_negative_sampling_authorized"
                    )
                    exclusion_count = row.get(
                        "page_geometry_exclusion_count"
                    )
                    if (
                        not isinstance(authorization, bool)
                        or not isinstance(exclusion_count, int)
                        or isinstance(exclusion_count, bool)
                        or exclusion_count < 0
                        or authorization != (exclusion_count == 0)
                    ):
                        raise ValueError(
                            f"{location}: invalid page-scoped hard-negative "
                            "authorization"
                        )
                    page_key = (split, source_key, image_path)
                    previous_authorization = (
                        hard_negative_authorization_by_page.setdefault(
                            page_key,
                            authorization,
                        )
                    )
                    if previous_authorization != authorization:
                        raise ValueError(
                            f"{location}: inconsistent page-scoped "
                            "hard-negative authorization"
                        )
                actual_word_counts[split] += 1

    expected_counts = report.get("words_by_split")
    if not isinstance(expected_counts, dict) or any(
        int(expected_counts.get(split, -1)) != actual_word_counts[split]
        for split in SPLITS
    ):
        raise ValueError(
            f"{spec.name}: JSONL word counts do not match preparation report"
        )
    if exhaustive_detection:
        expected_authorized_pages = report.get(
            "hard_negative_authorized_pages_by_split"
        )
        actual_authorized_pages = {
            split: sum(
                authorization
                for (page_split, _source, _image), authorization
                in hard_negative_authorization_by_page.items()
                if page_split == split
            )
            for split in SPLITS
        }
        if (
            not isinstance(expected_authorized_pages, dict)
            or any(
                int(expected_authorized_pages.get(split, -1))
                != actual_authorized_pages[split]
                for split in SPLITS
            )
        ):
            raise ValueError(
                f"{spec.name}: page authorization counts do not match "
                "preparation report"
            )

    pages_by_split: dict[str, list[PageLabel]] = {
        split: [] for split in SPLITS
    }
    for (split, source_key, image_path), annotations in sorted(
        annotations_by_page.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            item[0][2].as_posix(),
        ),
    ):
        width, height = dimensions[image_path]
        pages_by_split[split].append(
            PageLabel(
                dataset=spec.name,
                kind=spec.kind,
                split=split,
                source_key=source_key,
                project_relative_image=image_path.relative_to(
                    project_root
                ).as_posix(),
                width=width,
                height=height,
                image_sha256=sha256_file(image_path),
                annotations=tuple(annotations),
                hard_negative_sampling_authorized=bool(
                    exhaustive_detection
                    and hard_negative_authorization_by_page.get(
                        (split, source_key, image_path),
                        False,
                    )
                ),
            )
        )
    if (
        any(split not in SPLITS for split in required_nonempty_splits)
        or len(set(required_nonempty_splits)) != len(required_nonempty_splits)
    ):
        raise ValueError("required nonempty splits are invalid")
    if any(not pages_by_split[split] for split in required_nonempty_splits):
        raise ValueError(
            f"{spec.name}: required splits must contain text pages: "
            f"{required_nonempty_splits}"
        )
    return pages_by_split, {
        "name": spec.name,
        "kind": spec.kind,
        "role": report.get("role"),
        "forbidden_selection_overlap": report.get(
            "forbidden_selection_overlap"
        ),
        "forbidden_work_overlap": report.get(
            "forbidden_work_overlap"
        ),
        "directory": str(dataset_dir),
        "image_root": str(image_root),
        "prepare_report_sha256": sha256_file(report_path),
        "sources_by_split": {
            split: sum(value == split for value in source_splits.values())
            for split in SPLITS
        },
        "words_by_split": actual_word_counts,
        "positive_region_coverage": report.get(
            "positive_region_coverage",
            "source_proven_supported_score_text_only",
        ),
        "recall_evaluation_authorized": (
            report.get("recall_evaluation_authorized") is True
            if exhaustive_detection
            else True
        ),
        "precision_evaluation_authorized": bool(
            exhaustive_detection
            and report.get("precision_evaluation_authorized") is True
        ),
        "hmean_evaluation_authorized": bool(
            exhaustive_detection
            and report.get("hmean_evaluation_authorized") is True
        ),
        "hard_negative_sampling_authorized": bool(
            exhaustive_detection
            and report.get("hard_negative_sampling_authorized") is True
        ),
        "hard_negative_authorized_pages_by_split": (
            report.get("hard_negative_authorized_pages_by_split")
            if exhaustive_detection
            else {split: 0 for split in SPLITS}
        ),
        "unlabelled_visible_text_may_be_present": (
            report.get("unlabelled_visible_text_may_be_present") is not False
            if exhaustive_detection
            else True
        ),
        "pages_by_split": {
            split: len(pages_by_split[split])
            for split in SPLITS
        },
    }


def _annotation_json(annotation: DetectionAnnotation) -> dict[str, Any]:
    return {
        "transcription": annotation.transcription,
        "points": [
            [round(x, 3), round(y, 3)]
            for x, y in annotation.points
        ],
    }


def _label_line(row: PageLabel) -> str:
    annotations = []
    for item in row.annotations:
        annotation = _annotation_json(item)
        if row.hard_negative_sampling_authorized:
            annotation["hard_negative_sampling_authorized"] = True
        annotations.append(annotation)
    return (
        f"{row.project_relative_image}\t"
        f"{json.dumps(annotations, ensure_ascii=False, separators=(',', ':'))}"
    )


def _write_labels(path: Path, rows: Iterable[PageLabel]) -> int:
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(_label_line(row))
            stream.write("\n")
            count += 1
    os.replace(temporary, path)
    return count


def _output_coverage(
    rows: list[PageLabel],
    *,
    threshold_selection_split: bool = False,
) -> dict[str, Any]:
    authorized_pages = sum(
        page.hard_negative_sampling_authorized
        for page in rows
    )
    exhaustive = bool(rows) and authorized_pages == len(rows)
    return {
        "pages": len(rows),
        "recall_evaluation_authorized": bool(rows),
        "precision_evaluation_authorized": exhaustive,
        "hmean_evaluation_authorized": exhaustive,
        "postprocess_threshold_selection_authorized": bool(
            exhaustive and threshold_selection_split
        ),
        "hard_negative_authorized_pages": authorized_pages,
        "unlabelled_visible_text_may_be_present": not exhaustive,
    }


def build_balanced_training_pages(
    *,
    clean_pages: list[PageLabel],
    scan_pages: list[PageLabel],
    scan_target_fraction: float,
    seed: int,
) -> tuple[list[PageLabel], dict[str, Any]]:
    if not clean_pages or not scan_pages:
        raise ValueError("balanced detection training needs both domains")
    if not 0 < scan_target_fraction < 1:
        raise ValueError("scan target fraction must be between zero and one")
    required_scan = max(
        len(scan_pages),
        math.ceil(
            len(clean_pages)
            * scan_target_fraction
            / (1.0 - scan_target_fraction)
        ),
    )
    repeated_scan = [
        scan_pages[index % len(scan_pages)]
        for index in range(required_scan)
    ]
    combined = [*clean_pages, *repeated_scan]
    random.Random(seed).shuffle(combined)
    return combined, {
        "unique_clean_pages": len(clean_pages),
        "unique_scan_pages": len(scan_pages),
        "effective_scan_pages": len(repeated_scan),
        "scan_repeat_factor": round(
            len(repeated_scan) / len(scan_pages),
            6,
        ),
        "target_scan_fraction": scan_target_fraction,
        "achieved_scan_fraction": round(
            len(repeated_scan) / len(combined),
            8,
        ),
        "seed": seed,
    }


def deduplicate_same_split_exact_pages(
    loaded: dict[str, dict[str, list[PageLabel]]],
    specs: list[DatasetSpec],
) -> list[dict[str, object]]:
    """Remove only proven-safe aliases before writing labels.

    OpenScore occasionally contains different source records whose later pages
    are byte-identical (for example, two files embedding the same song from a
    collection).  Keeping both would overweight that page.  Moving either copy
    to a different split would be leakage.  We therefore retain the first page
    only when domain, split, pixels, dimensions, exhaustive-negative authority,
    and every annotation agree exactly.  Any cross-split or conflicting alias
    still fails closed.
    """

    retained_by_content: dict[str, PageLabel] = {}
    aliases: list[dict[str, object]] = []
    for spec in specs:
        pages_by_split = loaded[spec.name]
        for split in SPLITS:
            unique_pages: list[PageLabel] = []
            for page in pages_by_split[split]:
                previous = retained_by_content.get(page.image_sha256)
                if previous is None:
                    retained_by_content[page.image_sha256] = page
                    unique_pages.append(page)
                    continue
                safe_alias = (
                    previous.kind == page.kind
                    and previous.split == page.split
                    and previous.width == page.width
                    and previous.height == page.height
                    and previous.annotations == page.annotations
                    and previous.hard_negative_sampling_authorized
                    == page.hard_negative_sampling_authorized
                )
                if not safe_alias:
                    raise RuntimeError(
                        "duplicate page content crosses a split/domain or "
                        "has conflicting labels: "
                        f"{previous.project_relative_image} and "
                        f"{page.project_relative_image}"
                    )
                aliases.append(
                    {
                        "image_sha256": page.image_sha256,
                        "kind": page.kind,
                        "split": page.split,
                        "retained_dataset": previous.dataset,
                        "retained_source_key": previous.source_key,
                        "retained_image": (
                            previous.project_relative_image
                        ),
                        "removed_dataset": page.dataset,
                        "removed_source_key": page.source_key,
                        "removed_image": page.project_relative_image,
                        "annotation_count": len(page.annotations),
                        "resolution": (
                            "same_split_exact_annotation_deduplicated"
                        ),
                    }
                )
            pages_by_split[split] = unique_pages
    return aliases


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
    parser.add_argument("--seed", type=int, default=20260728)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.project_root = args.project_root.resolve()
    specs: list[DatasetSpec] = [*args.clean_dataset, *args.scan_dataset]
    if not args.clean_dataset or not args.scan_dataset:
        raise ValueError("at least one clean and one scan dataset are required")
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("dataset names must be unique")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaded: dict[str, dict[str, list[PageLabel]]] = {}
    dataset_reports: list[dict[str, Any]] = []
    image_assignments: dict[str, tuple[str, str, str]] = {}
    for spec in specs:
        pages_by_split, dataset_report = load_dataset(
            spec,
            project_root=args.project_root,
        )
        loaded[spec.name] = pages_by_split
        dataset_reports.append(dataset_report)
        for split in SPLITS:
            for page in pages_by_split[split]:
                assignment = (page.dataset, page.source_key, split)
                previous = image_assignments.setdefault(
                    page.project_relative_image,
                    assignment,
                )
                if previous != assignment:
                    raise RuntimeError(
                        "physical page leakage/alias across source splits: "
                        f"{page.project_relative_image}"
                    )
    same_split_aliases = deduplicate_same_split_exact_pages(loaded, specs)

    output_counts: dict[str, int] = {}
    output_label_coverage: dict[str, dict[str, Any]] = {}
    clean_train: list[PageLabel] = []
    scan_train: list[PageLabel] = []
    for spec in specs:
        for split in SPLITS:
            filename = f"{split}.{spec.name}.paddle.det.txt"
            pages = loaded[spec.name][split]
            output_counts[filename] = _write_labels(
                args.output_dir / filename,
                pages,
            )
            output_label_coverage[filename] = _output_coverage(
                pages,
                threshold_selection_split=(split == "calibration"),
            )
        destination = clean_train if spec.kind == "clean" else scan_train
        destination.extend(loaded[spec.name]["train"])

    balanced, balance_report = build_balanced_training_pages(
        clean_pages=clean_train,
        scan_pages=scan_train,
        scan_target_fraction=args.scan_target_fraction,
        seed=args.seed,
    )
    balanced_path = args.output_dir / "train.balanced.paddle.det.txt"
    output_counts[balanced_path.name] = _write_labels(
        balanced_path,
        balanced,
    )
    output_label_coverage[balanced_path.name] = _output_coverage(
        balanced,
    )

    for split in ("calibration", "test"):
        for kind in ("clean", "scan"):
            pages = [
                page
                for spec in specs
                if spec.kind == kind
                for page in loaded[spec.name][split]
            ]
            if not pages:
                raise ValueError(f"{kind} {split} split is empty")
            filename = f"{split}.{kind}.paddle.det.txt"
            output_counts[filename] = _write_labels(
                args.output_dir / filename,
                pages,
            )
            output_label_coverage[filename] = _output_coverage(
                pages,
                threshold_selection_split=(split == "calibration"),
            )
            exhaustive_pages = [
                page
                for page in pages
                if page.hard_negative_sampling_authorized
            ]
            if exhaustive_pages:
                exhaustive_filename = (
                    f"{split}.{kind}.exhaustive.paddle.det.txt"
                )
                output_counts[exhaustive_filename] = _write_labels(
                    args.output_dir / exhaustive_filename,
                    exhaustive_pages,
                )
                output_label_coverage[exhaustive_filename] = (
                    _output_coverage(
                        exhaustive_pages,
                        threshold_selection_split=(
                            split == "calibration"
                        ),
                    )
                )

    all_precision_authorized = all(
        row.get("precision_evaluation_authorized") is True
        for row in dataset_reports
    )
    all_hmean_authorized = all(
        row.get("hmean_evaluation_authorized") is True
        for row in dataset_reports
    )
    unlabelled_text_may_be_present = any(
        row.get("unlabelled_visible_text_may_be_present") is not False
        for row in dataset_reports
    )
    hard_negative_page_count = sum(
        1
        for spec in specs
        for page in loaded[spec.name]["train"]
        if page.hard_negative_sampling_authorized
    )
    report = {
        "schema_version": 1,
        "name": "scorescan-ppocrv6-domain-detection-labels-v1",
        "role": "training_only_disjoint_from_release_benchmarks",
        "project_root": str(args.project_root),
        "datasets": dataset_reports,
        "global_split_intersections": EMPTY_INTERSECTIONS,
        "physical_image_aliases_across_splits": [],
        "duplicate_image_content_across_source_assignments": [],
        "same_split_exact_content_aliases_deduplicated": (
            same_split_aliases
        ),
        "same_split_exact_content_alias_count": len(same_split_aliases),
        "label_coverage_contract": {
            "positive_region_coverage": "mixed_page_scoped_contracts",
            "recall_evaluation_authorized": True,
            "precision_evaluation_authorized": (
                all_precision_authorized
                and not unlabelled_text_may_be_present
            ),
            "hmean_evaluation_authorized": (
                all_hmean_authorized
                and not unlabelled_text_may_be_present
            ),
            "postprocess_threshold_selection_authorized": (
                all_precision_authorized
                and all_hmean_authorized
                and not unlabelled_text_may_be_present
            ),
            "hard_negative_sampling_authorized": (
                hard_negative_page_count > 0
            ),
            "hard_negative_authorization_scope": "per_page_annotation",
            "hard_negative_authorized_unique_train_pages": (
                hard_negative_page_count
            ),
            "unlabelled_visible_text_may_be_present": (
                unlabelled_text_may_be_present
            ),
        },
        "balance": balance_report,
        "output_counts": output_counts,
        "output_label_coverage": output_label_coverage,
    }
    report_path = args.output_dir / "merge-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    hashed_paths = [
        *sorted(args.output_dir.glob("*.paddle.det.txt")),
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
    print(json.dumps(output_counts, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
