#!/usr/bin/env python3
"""Prepare exact word crops from leakage-isolated registered Muse OMR scans."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageStat

from app.tools.prepare_muse_omr_scan_regions import (
    BENCHMARK_SELECTION_ROLE,
    DEFAULT_DATASET_DIR,
    DEFAULT_OUTPUT_DIR as DEFAULT_REGION_DIR,
    TRAINING_REGION_ROLE,
    _atomic_json,
    _registered_page_path,
    _stable_subset,
)
from app.tools.muse_omr_contract import SCAN_DEGRADED_IMAGE_ORIGIN
from app.tools.prepare_openscore_pdf_text import (
    EXHAUSTIVE_DETECTION_LABEL_CONTRACT,
    _render_pdf,
    consume_source_text_role,
    extract_page_words,
    reuse_rendered_pdf,
    sha256_file,
    source_text_token_counts,
)
from app.tools.ocr_text_contract import SOURCE_TEXT_SELECTION_VERSION


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "training_data"
    / "prepared"
    / "muse_omr_scan_text_v1"
)
SCAN_TEXT_VISUAL_PRESENCE_VERSION = (
    "registered-word-template-highpass-ncc@1"
)
SCAN_TEXT_PAGE_LABEL_COMPLETENESS_VERSION = (
    "supported-text-page-atomic@1"
)
SCAN_TEXT_REFERENCE_SOURCE_VERSION = (
    "registered-cache-or-source-mscz-pdf-rerender@1"
)
EXHAUSTIVE_REGISTERED_DETECTION_PAGE_CONTRACT = (
    "registered-all-visible-text-page-atomic@1"
)
EXHAUSTIVE_REGISTERED_DETECTION_SELECTION_POLICY = (
    "discard-page-on-any-geometry-visual-or-quality-rejection@1"
)
MINIMUM_SAFE_VISUAL_PRESENCE_NCC = 0.15
VISUAL_PRESENCE_HIGHPASS_SIGMA = 5.0
REFERENCE_PAGE_SOURCE_KEYS = (
    "registered_reference_cache",
    "source_mscz_pdf_rerender",
)


def validate_reference_page_source_evidence(
    report: dict[str, Any],
) -> dict[str, int]:
    if (
        report.get("scan_text_reference_source_version")
        != SCAN_TEXT_REFERENCE_SOURCE_VERSION
    ):
        raise ValueError("scan-text reference-source contract is stale")
    raw_counts = report.get("reference_page_source_counts")
    raw_total = report.get("reference_page_count")
    if not isinstance(raw_counts, dict):
        raise ValueError("scan-text reference-source counts are missing")
    counts: dict[str, int] = {}
    for key in REFERENCE_PAGE_SOURCE_KEYS:
        value = raw_counts.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("scan-text reference-source counts are invalid")
        counts[key] = value
    if (
        set(raw_counts) != set(REFERENCE_PAGE_SOURCE_KEYS)
        or isinstance(raw_total, bool)
        or not isinstance(raw_total, int)
        or raw_total <= 0
        or sum(counts.values()) != raw_total
    ):
        raise ValueError("scan-text reference-source total is inconsistent")
    return counts


def _load_registered_report(
    region_dir: Path,
    *,
    expected_role: str = TRAINING_REGION_ROLE,
) -> dict[str, Any]:
    report_path = region_dir / "prepare-report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("role") != expected_role
        or report.get("source_image_origin")
        != SCAN_DEGRADED_IMAGE_ORIGIN
        or report.get("production_evidence_eligible") is not False
        or report.get("split_intersections")
        != {
            "train_calibration": [],
            "train_test": [],
            "calibration_test": [],
        }
        or not isinstance(report.get("accepted"), list)
        or not report["accepted"]
    ):
        raise ValueError("registered scan report failed its isolation contract")
    if (
        expected_role == BENCHMARK_SELECTION_ROLE
        and (
            report.get("forbidden_selection_overlap") != []
            or report.get("forbidden_work_overlap") != []
        )
    ):
        raise ValueError(
            "registered scan holdout overlaps its forbidden selection"
        )
    return report


def _crop_quality(
    crop: Image.Image,
    *,
    minimum_stddev: float,
    minimum_dark_fraction: float,
) -> tuple[bool, dict[str, float]]:
    gray = crop.convert("L")
    if gray.width < 4 or gray.height < 4:
        return False, {"stddev": 0.0, "dark_fraction": 0.0}
    stat = ImageStat.Stat(gray)
    histogram = gray.histogram()
    dark_pixels = sum(histogram[:220])
    dark_fraction = dark_pixels / max(1, gray.width * gray.height)
    stddev = float(stat.stddev[0])
    return (
        stddev >= minimum_stddev and dark_fraction >= minimum_dark_fraction,
        {
            "stddev": round(stddev, 6),
            "dark_fraction": round(dark_fraction, 8),
        },
    )


def _dark_highpass(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        VISUAL_PRESENCE_HIGHPASS_SIGMA,
    )
    return np.maximum(
        background.astype(np.float32) - gray.astype(np.float32),
        0.0,
    )


def _visual_presence_ncc(
    reference_crop: Image.Image,
    scan_crop: Image.Image,
) -> float:
    """Find the clean reference glyph in the registered scan crop.

    Smooth paper texture, stains and exposure gradients are removed before
    matching.  The tight reference-ink template can move inside the padded
    scan crop, which tolerates small residual registration errors without
    accepting a merely high-contrast but text-free patch.
    """

    if reference_crop.size != scan_crop.size:
        raise ValueError("reference/scan text crops must have equal size")
    reference = _dark_highpass(reference_crop)
    scan = _dark_highpass(scan_crop)
    if min(reference.shape) < 2 or min(scan.shape) < 2:
        return 0.0
    threshold = max(4.0, float(np.percentile(reference, 80)))
    ys, xs = np.where(reference > threshold)
    if len(xs) < 4:
        return 0.0
    left = max(0, int(xs.min()) - 2)
    right = min(reference.shape[1], int(xs.max()) + 3)
    top = max(0, int(ys.min()) - 2)
    bottom = min(reference.shape[0], int(ys.max()) + 3)
    template = reference[top:bottom, left:right]
    if (
        min(template.shape) < 2
        or template.shape[0] > scan.shape[0]
        or template.shape[1] > scan.shape[1]
        or float(template.std()) <= 1e-5
        or float(scan.std()) <= 1e-5
    ):
        return 0.0
    matched = cv2.matchTemplate(
        scan,
        template,
        cv2.TM_CCOEFF_NORMED,
    )
    finite = matched[np.isfinite(matched)]
    if finite.size == 0:
        return 0.0
    matched = np.where(np.isfinite(matched), matched, -1.0).astype(
        np.float32,
        copy=False,
    )
    _minimum, correlation, _minimum_location, location = cv2.minMaxLoc(
        matched
    )
    matched_patch = scan[
        location[1] : location[1] + template.shape[0],
        location[0] : location[0] + template.shape[1],
    ]
    reference_ink = max(1, int(np.count_nonzero(template > 4.0)))
    scan_ink = int(np.count_nonzero(matched_patch > 4.0))
    # A smooth stain edge can produce a modest mean-centred correlation with
    # a tiny glyph despite containing too little local stroke energy.  Require
    # at least half of the reference ink occupancy for full credit.
    occupancy_factor = min(1.0, (scan_ink / reference_ink) / 0.5)
    score = float(correlation) * occupancy_factor
    return max(0.0, min(1.0, score))


def _render_pdf_reference_page(
    page: Any,
    *,
    width: int,
    height: int,
) -> Image.Image:
    import pymupdf as fitz

    if width <= 0 or height <= 0:
        raise ValueError("reference render dimensions must be positive")
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(
            width / max(float(page.rect.width), 1.0),
            height / max(float(page.rect.height), 1.0),
        ),
        colorspace=fitz.csGRAY,
        alpha=False,
    )
    array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height,
        pixmap.width,
    )
    if array.shape != (height, width):
        array = cv2.resize(
            array,
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
    return Image.fromarray(array, mode="L")


def _score_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "minimum": round(float(array.min()), 6),
        "p10": round(float(np.quantile(array, 0.10)), 6),
        "median": round(float(np.median(array)), 6),
        "p90": round(float(np.quantile(array, 0.90)), 6),
        "maximum": round(float(array.max()), 6),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            )
            stream.write("\n")


def _write_paddle_labels(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            text = str(row["text"])
            if "\t" in text or "\n" in text or "\r" in text:
                raise ValueError(f"unsafe OCR label: {text!r}")
            stream.write(f"{row['crop_image']}\t{text}\n")


def _verified_reusable_pdf_hashes(report_path: Path) -> dict[int, str]:
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_reference_page_source_evidence(report)
    result: dict[int, str] = {}
    sources = report.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("reusable scan-text report has no sources")
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("reusable scan-text source row is malformed")
        variants = source.get("variants")
        if not isinstance(variants, list):
            raise ValueError("reusable scan-text source variants are missing")
        for variant in variants:
            if not isinstance(variant, dict):
                raise ValueError("reusable scan-text variant is malformed")
            pair_id = variant.get("pair_id")
            pdf_hash = str(variant.get("pdf_sha256", "")).casefold()
            if (
                isinstance(pair_id, bool)
                or not isinstance(pair_id, int)
                or pair_id <= 0
                or len(pdf_hash) != 64
                or any(character not in "0123456789abcdef" for character in pdf_hash)
                or pair_id in result
            ):
                raise ValueError("reusable scan-text PDF identity is invalid")
            result[pair_id] = pdf_hash
    if not result:
        raise ValueError("reusable scan-text report contains no PDFs")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--region-dir", type=Path, default=DEFAULT_REGION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--musescore-exe", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--padding-pixels", type=float, default=10.0)
    parser.add_argument("--minimum-crop-stddev", type=float, default=4.0)
    parser.add_argument("--minimum-dark-fraction", type=float, default=0.003)
    parser.add_argument(
        "--minimum-visual-presence-ncc",
        type=float,
        default=MINIMUM_SAFE_VISUAL_PRESENCE_NCC,
    )
    parser.add_argument("--render-timeout-seconds", type=int, default=900)
    parser.add_argument("--include-lyrics", action="store_true")
    parser.add_argument(
        "--detection-all-visible-text",
        action="store_true",
        help=(
            "retain complete registered pages with every visually verified "
            "non-music-font PDF word for text detection"
        ),
    )
    parser.add_argument(
        "--reuse-pdf-dir",
        type=Path,
        help="reuse hash-verified pair PDFs from a completed scan-text dataset",
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--expected-region-role",
        choices=(TRAINING_REGION_ROLE, BENCHMARK_SELECTION_ROLE),
        default=TRAINING_REGION_ROLE,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for directory in (args.dataset_dir, args.region_dir):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    if not args.musescore_exe.is_file():
        raise FileNotFoundError(args.musescore_exe)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(args.output_dir)
    if args.resume and (args.output_dir / "prepare-report.json").is_file():
        raise FileExistsError("refusing to resume a completed scan-text dataset")
    if args.padding_pixels < 0:
        raise ValueError("padding-pixels must be non-negative")
    if args.minimum_crop_stddev < 0:
        raise ValueError("minimum-crop-stddev must be non-negative")
    if not 0 <= args.minimum_dark_fraction <= 1:
        raise ValueError("minimum-dark-fraction must be in [0, 1]")
    if not MINIMUM_SAFE_VISUAL_PRESENCE_NCC <= (
        args.minimum_visual_presence_ncc
    ) <= 1:
        raise ValueError(
            "minimum-visual-presence-ncc is below the safe floor or above 1"
        )
    if args.shard_count <= 0:
        raise ValueError("shard-count must be positive")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    reusable_pdf_hashes: dict[int, str] = {}
    reusable_pdf_report_path: Path | None = None
    if args.reuse_pdf_dir is not None:
        if not args.reuse_pdf_dir.is_dir():
            raise FileNotFoundError(args.reuse_pdf_dir)
        reusable_pdf_report_path = (
            args.reuse_pdf_dir.parent / "prepare-report.json"
        )
        reusable_pdf_hashes = _verified_reusable_pdf_hashes(
            reusable_pdf_report_path
        )

    import pdfplumber

    region_report = _load_registered_report(
        args.region_dir,
        expected_role=args.expected_region_role,
    )
    accepted_by_id = {
        int(row["pair_id"]): row
        for row in region_report["accepted"]
    }
    selected_ids = _stable_subset(
        accepted_by_id,
        args.max_pairs,
        args.seed,
    )
    selected_ids = [
        pair_id
        for index, pair_id in enumerate(selected_ids)
        if index % args.shard_count == args.shard_index
    ]
    if not selected_ids:
        raise ValueError("registered scan-text source shard is empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    split_sources: dict[str, set[str]] = defaultdict(set)
    excluded_lyrics: Counter[str] = Counter()
    excluded_ambiguous: Counter[str] = Counter()
    excluded_unproven: Counter[str] = Counter()
    rejected_crops: Counter[str] = Counter()
    rejected_visual_presence: Counter[str] = Counter()
    retained_visual_presence_scores: dict[str, list[float]] = defaultdict(
        list
    )
    rejected_visual_presence_scores: dict[str, list[float]] = defaultdict(
        list
    )
    discarded_incomplete_pages: Counter[str] = Counter()
    discarded_incomplete_page_rows: Counter[str] = Counter()
    retained_complete_detection_pages: Counter[str] = Counter()
    geometry_excluded_detection_pages: Counter[str] = Counter()
    retained_low_quality_crops: Counter[str] = Counter()
    reference_page_sources: Counter[str] = Counter()
    excluded_geometry: dict[str, Counter[str]] = defaultdict(Counter)
    source_manifest: dict[str, dict[str, Any]] = {}
    included_source_contexts: Counter[str] = Counter()
    excluded_source_contexts: Counter[str] = Counter()

    for completed, pair_id in enumerate(selected_ids, start=1):
        accepted = accepted_by_id[pair_id]
        split = str(accepted["split"])
        if split not in {"train", "calibration", "test"}:
            raise ValueError(f"invalid registered split: {split}")
        source_key = str(accepted["source_key"])
        variant_key = str(accepted["variant_key"])
        work_fingerprint = str(accepted["work_fingerprint"])
        mscz = args.dataset_dir / "mscz" / f"score_file_{pair_id}.mscz"
        registration_path = (
            args.region_dir
            / "pages"
            / f"pair-{pair_id:04d}"
            / "registration.json"
        )
        if not registration_path.is_file():
            raise FileNotFoundError(registration_path)
        registration = json.loads(
            registration_path.read_text(encoding="utf-8")
        )
        accepted_page_rows = registration.get("pages", [])
        accepted_page_numbers = [
            int(page["page"]) for page in accepted_page_rows
        ]
        reference_pages = [
            args.region_dir
            / "reference_pages"
            / f"pair-{pair_id:04d}"
            / f"page-{page_number}.png"
            for page_number in accepted_page_numbers
        ]
        scan_pages = [
            _registered_page_path(
                args.region_dir / "pages" / f"pair-{pair_id:04d}",
                page,
            )
            for page in accepted_page_rows
        ]
        if (
            not mscz.is_file()
            or not reference_pages
            or not all(path.is_file() for path in scan_pages)
        ):
            raise FileNotFoundError(f"incomplete registered text pair {pair_id}")
        pdf_path = args.output_dir / "pdf" / f"pair-{pair_id:04d}.pdf"
        if args.reuse_pdf_dir is not None:
            expected_pdf_hash = reusable_pdf_hashes.get(pair_id)
            if expected_pdf_hash is None:
                raise ValueError(
                    f"reusable scan-text report has no pair {pair_id}"
                )
            reuse_rendered_pdf(
                args.reuse_pdf_dir / pdf_path.name,
                pdf_path,
                expected_sha256=expected_pdf_hash,
            )
        _render_pdf(
            mscz,
            musescore_exe=args.musescore_exe,
            output_path=pdf_path,
            timeout_seconds=args.render_timeout_seconds,
            reuse_existing=args.resume,
        )
        if args.detection_all_visible_text:
            lyric_tokens: Counter[str] = Counter()
            non_lyric_tokens: Counter[str] = Counter()
        else:
            lyric_tokens, non_lyric_tokens = source_text_token_counts(
                mscz,
                included_contexts=included_source_contexts,
                excluded_contexts=excluded_source_contexts,
            )
        ambiguous_keys = set(lyric_tokens) & set(non_lyric_tokens)
        remaining_lyrics = lyric_tokens.copy()
        remaining_non_lyrics = non_lyric_tokens.copy()
        source_rows = 0
        source_rejected = 0
        import pymupdf as fitz

        with pdfplumber.open(pdf_path) as pdf, fitz.open(
            str(pdf_path)
        ) as pixel_pdf:
            if len(pdf.pages) < max(accepted_page_numbers):
                raise ValueError(
                    f"text PDF/registered page mismatch for pair {pair_id}: "
                    f"{len(pdf.pages)} < {max(accepted_page_numbers)}"
                )
            for page_number, reference_path, scan_path in zip(
                accepted_page_numbers,
                reference_pages,
                scan_pages,
                strict=True,
            ):
                page = pdf.pages[page_number - 1]
                with Image.open(scan_path) as scan_source:
                    scan = scan_source.convert("L")
                if reference_path.is_file():
                    with Image.open(reference_path) as reference_source:
                        reference = reference_source.convert("L")
                    reference_page_sources[
                        "registered_reference_cache"
                    ] += 1
                else:
                    reference = _render_pdf_reference_page(
                        pixel_pdf[page_number - 1],
                        width=scan.width,
                        height=scan.height,
                    )
                    reference_page_sources[
                        "source_mscz_pdf_rerender"
                    ] += 1
                if reference.size != scan.size:
                    raise ValueError(
                        f"registered/reference size mismatch: {scan.size} != "
                        f"{reference.size}"
                    )
                geometry_exclusions_before = sum(
                    excluded_geometry[split].values()
                )
                words = extract_page_words(
                    page,
                    image_width=reference.width,
                    image_height=reference.height,
                    padding_pixels=args.padding_pixels,
                    exclusion_counts=excluded_geometry[split],
                    include_nonmusic_punctuation=(
                        args.detection_all_visible_text
                    ),
                )
                page_geometry_exclusions = (
                    sum(excluded_geometry[split].values())
                    - geometry_exclusions_before
                )
                page_candidates: list[
                    tuple[dict[str, Any], Image.Image | None]
                ] = []
                page_has_rejected_supported_word = bool(
                    args.detection_all_visible_text
                    and page_geometry_exclusions > 0
                )
                if page_has_rejected_supported_word:
                    geometry_excluded_detection_pages[split] += 1
                for word_index, word in enumerate(words):
                    if args.detection_all_visible_text:
                        role = "visible_text"
                    else:
                        role = consume_source_text_role(
                            str(word["text"]),
                            remaining_lyrics=remaining_lyrics,
                            remaining_non_lyrics=remaining_non_lyrics,
                            ambiguous_keys=ambiguous_keys,
                        )
                        if not args.include_lyrics and role == "lyric":
                            excluded_lyrics[split] += 1
                            continue
                        if role == "ambiguous":
                            excluded_ambiguous[split] += 1
                            continue
                        if role == "unproven":
                            excluded_unproven[split] += 1
                            continue
                    left, top, right, bottom = (
                        float(value) for value in word["box_xyxy"]
                    )
                    box = (
                        max(0, math.floor(left)),
                        max(0, math.floor(top)),
                        min(scan.width, math.ceil(right)),
                        min(scan.height, math.ceil(bottom)),
                    )
                    reference_crop = reference.crop(box)
                    scan_crop = scan.crop(box)
                    reference_ok, _reference_quality = _crop_quality(
                        reference_crop,
                        minimum_stddev=args.minimum_crop_stddev,
                        minimum_dark_fraction=args.minimum_dark_fraction,
                    )
                    scan_ok, scan_quality = _crop_quality(
                        scan_crop,
                        minimum_stddev=args.minimum_crop_stddev,
                        minimum_dark_fraction=args.minimum_dark_fraction,
                    )
                    visual_presence_ncc = _visual_presence_ncc(
                        reference_crop,
                        scan_crop,
                    )
                    if (
                        visual_presence_ncc
                        < args.minimum_visual_presence_ncc
                    ):
                        rejected_crops["scan_text_not_visually_present"] += 1
                        rejected_visual_presence[split] += 1
                        rejected_visual_presence_scores[split].append(
                            visual_presence_ncc
                        )
                        page_has_rejected_supported_word = True
                        source_rejected += 1
                        reference_crop.close()
                        scan_crop.close()
                        continue
                    crop_quality_accepted = reference_ok and scan_ok
                    if not crop_quality_accepted:
                        reason = (
                            "reference_low_contrast"
                            if not reference_ok
                            else "scan_low_contrast"
                        )
                        if args.expected_region_role == TRAINING_REGION_ROLE:
                            rejected_crops[reason] += 1
                            page_has_rejected_supported_word = True
                            source_rejected += 1
                            reference_crop.close()
                            scan_crop.close()
                            continue
                        retained_low_quality_crops[reason] += 1
                    relative_crop = (
                        Path("crops")
                        / split
                        / f"pair-{pair_id:04d}-page-{page_number:03d}"
                        f"-word-{word_index:04d}.png"
                    )
                    row = {
                                "split": split,
                                "source_key": source_key,
                                "variant_key": variant_key,
                                "work_fingerprint": work_fingerprint,
                                "pair_id": pair_id,
                                "page": page_number,
                                "word_index": word_index,
                                "image": scan_path.relative_to(
                                    args.region_dir
                                ).as_posix(),
                                "crop_image": relative_crop.as_posix(),
                                "text": str(word["text"]),
                                "text_role": role,
                                "font_name": str(word["font_name"]),
                                "font_size_pt": word["font_size_pt"],
                                "box_xyxy": list(box),
                                "scan_crop_quality": scan_quality,
                                "visual_presence_ncc": round(
                                    visual_presence_ncc,
                                    6,
                                ),
                                "crop_quality_accepted": (
                                    crop_quality_accepted
                                ),
                            }
                    if args.detection_all_visible_text:
                        row["hard_negative_sampling_authorized"] = True
                        row["page_geometry_exclusion_count"] = 0
                    page_candidates.append(
                        (
                            row,
                            (
                                None
                                if args.detection_all_visible_text
                                else scan_crop.copy()
                            ),
                        )
                    )
                    reference_crop.close()
                    scan_crop.close()
                if page_has_rejected_supported_word:
                    discarded_incomplete_pages[split] += 1
                    discarded_incomplete_page_rows[split] += len(
                        page_candidates
                    )
                    rejected_crops[
                        "page_rows_discarded_after_supported_word_rejection"
                    ] += len(page_candidates)
                    source_rejected += len(page_candidates)
                    for _row, candidate_crop in page_candidates:
                        if candidate_crop is not None:
                            candidate_crop.close()
                    continue
                if args.detection_all_visible_text and page_candidates:
                    retained_complete_detection_pages[split] += 1
                for row, candidate_crop in page_candidates:
                    if candidate_crop is not None:
                        crop_path = args.output_dir / row["crop_image"]
                        crop_path.parent.mkdir(parents=True, exist_ok=True)
                        temporary = crop_path.with_name(
                            f"{crop_path.stem}.tmp{crop_path.suffix}"
                        )
                        candidate_crop.save(
                            temporary,
                            format="PNG",
                            optimize=True,
                        )
                        candidate_crop.close()
                        os.replace(temporary, crop_path)
                    rows_by_split[split].append(row)
                    retained_visual_presence_scores[split].append(
                        float(row["visual_presence_ncc"])
                    )
                    source_rows += 1
        split_sources[split].add(source_key)
        source = source_manifest.setdefault(
            source_key,
            {
                "source_key": source_key,
                "work_fingerprint": work_fingerprint,
                "split": split,
                "pair_ids": [],
                "pages": 0,
                "retained_words": 0,
                "rejected_words": 0,
                "variants": [],
            },
        )
        if (
            source["split"] != split
            or source["work_fingerprint"] != work_fingerprint
        ):
            raise RuntimeError("work variants crossed scan-text splits")
        source["pair_ids"].append(pair_id)
        source["pages"] += len(scan_pages)
        source["retained_words"] += source_rows
        source["rejected_words"] += source_rejected
        source["variants"].append(
            {
                "pair_id": pair_id,
                "variant_key": variant_key,
                "mscz_sha256": sha256_file(mscz),
                "pdf_sha256": sha256_file(pdf_path),
            }
        )
        print(
            f"[{completed}/{len(selected_ids)}] pair {pair_id}: "
            f"{source_rows} words",
            flush=True,
        )

    intersections = {
        f"{left}_{right}": sorted(split_sources[left] & split_sources[right])
        for index, left in enumerate(("train", "calibration", "test"))
        for right in ("train", "calibration", "test")[index + 1 :]
    }
    if any(intersections.values()):
        raise RuntimeError(f"scan text source leakage detected: {intersections}")

    artifact_paths: list[Path] = []
    for split in ("train", "calibration", "test"):
        jsonl_path = args.output_dir / f"{split}.jsonl"
        _write_jsonl(jsonl_path, rows_by_split[split])
        artifact_paths.append(jsonl_path)
        if not args.detection_all_visible_text:
            labels_path = args.output_dir / f"{split}.paddle.txt"
            _write_paddle_labels(labels_path, rows_by_split[split])
            artifact_paths.append(labels_path)
    report = {
        "schema_version": 1,
        "name": (
            "scorescan-muse-omr-registered-scan-visible-text-detection-v1"
            if args.detection_all_visible_text
            else (
                "scorescan-muse-omr-registered-scan-text-holdout-v1"
                if args.expected_region_role == BENCHMARK_SELECTION_ROLE
                else "scorescan-muse-omr-registered-scan-text-v1"
            )
        ),
        "license": region_report["license"],
        "source_image_origin": SCAN_DEGRADED_IMAGE_ORIGIN,
        "production_evidence_eligible": False,
        "purpose": (
            "registered exhaustive visible-text detection training/calibration"
            if args.detection_all_visible_text
            else "registered exact-text OCR training/calibration"
        ),
        "source_text_selection_version": SOURCE_TEXT_SELECTION_VERSION,
        "scan_text_visual_presence_version": (
            SCAN_TEXT_VISUAL_PRESENCE_VERSION
        ),
        "scan_text_page_label_completeness_version": (
            SCAN_TEXT_PAGE_LABEL_COMPLETENESS_VERSION
        ),
        "scan_text_reference_source_version": (
            SCAN_TEXT_REFERENCE_SOURCE_VERSION
        ),
        "reference_page_source_counts": {
            key: int(reference_page_sources[key])
            for key in REFERENCE_PAGE_SOURCE_KEYS
        },
        "reference_page_count": int(sum(reference_page_sources.values())),
        "minimum_visual_presence_ncc": (
            args.minimum_visual_presence_ncc
        ),
        "role": args.expected_region_role,
        "forbidden_selection_overlap": region_report.get(
            "forbidden_selection_overlap"
        ),
        "forbidden_work_overlap": region_report.get(
            "forbidden_work_overlap"
        ),
        "region_dir": str(args.region_dir.resolve()),
        "region_report_sha256": sha256_file(
            args.region_dir / "prepare-report.json"
        ),
        "reused_pdf_dir": (
            str(args.reuse_pdf_dir.resolve())
            if args.reuse_pdf_dir is not None
            else None
        ),
        "reused_pdf_report_sha256": (
            sha256_file(reusable_pdf_report_path)
            if reusable_pdf_report_path is not None
            else None
        ),
        "source_shard": {
            "count": args.shard_count,
            "index": args.shard_index,
        },
        "selected_pairs": len(selected_ids),
        "selected_works": len(source_manifest),
        "sources_by_split": {
            split: len(split_sources[split])
            for split in ("train", "calibration", "test")
        },
        "sources_with_retained_words_by_split": {
            split: sum(
                source["split"] == split
                and int(source["retained_words"]) > 0
                for source in source_manifest.values()
            )
            for split in ("train", "calibration", "test")
        },
        "words_by_split": {
            split: len(rows_by_split[split])
            for split in ("train", "calibration", "test")
        },
        "excluded_lyrics_by_split": {
            split: excluded_lyrics[split]
            for split in ("train", "calibration", "test")
        },
        "excluded_ambiguous_source_words_by_split": {
            split: excluded_ambiguous[split]
            for split in ("train", "calibration", "test")
        },
        "excluded_unproven_pdf_words_by_split": {
            split: excluded_unproven[split]
            for split in ("train", "calibration", "test")
        },
        "included_source_token_contexts": dict(
            sorted(included_source_contexts.items())
        ),
        "excluded_source_token_contexts": dict(
            sorted(excluded_source_contexts.items())
        ),
        "rejected_crop_counts": dict(sorted(rejected_crops.items())),
        "rejected_visual_presence_by_split": {
            split: rejected_visual_presence[split]
            for split in ("train", "calibration", "test")
        },
        "discarded_incomplete_text_pages_by_split": {
            split: discarded_incomplete_pages[split]
            for split in ("train", "calibration", "test")
        },
        "discarded_rows_on_incomplete_text_pages_by_split": {
            split: discarded_incomplete_page_rows[split]
            for split in ("train", "calibration", "test")
        },
        "text_pages_by_split": {
            split: retained_complete_detection_pages[split]
            for split in ("train", "calibration", "test")
        },
        "hard_negative_authorized_pages_by_split": {
            split: retained_complete_detection_pages[split]
            for split in ("train", "calibration", "test")
        },
        "geometry_excluded_text_pages_by_split": {
            split: geometry_excluded_detection_pages[split]
            for split in ("train", "calibration", "test")
        },
        "visual_presence_ncc_summary_by_split": {
            split: {
                "retained": _score_summary(
                    retained_visual_presence_scores[split]
                ),
                "rejected": _score_summary(
                    rejected_visual_presence_scores[split]
                ),
            }
            for split in ("train", "calibration", "test")
        },
        "retained_low_quality_crop_counts": dict(
            sorted(retained_low_quality_crops.items())
        ),
        "excluded_pdf_geometry_words_by_split": {
            split: dict(sorted(excluded_geometry[split].items()))
            for split in ("train", "calibration", "test")
        },
        "split_intersections": intersections,
        "lyrics_included": bool(
            args.include_lyrics or args.detection_all_visible_text
        ),
        "detection_label_contract": (
            EXHAUSTIVE_DETECTION_LABEL_CONTRACT
            if args.detection_all_visible_text
            else None
        ),
        "detection_page_label_completeness_version": (
            EXHAUSTIVE_REGISTERED_DETECTION_PAGE_CONTRACT
            if args.detection_all_visible_text
            else None
        ),
        "detection_page_selection_policy": (
            EXHAUSTIVE_REGISTERED_DETECTION_SELECTION_POLICY
            if args.detection_all_visible_text
            else None
        ),
        "all_usable_pdf_text_included": bool(
            args.detection_all_visible_text
        ),
        "positive_region_coverage": (
            "all_registered_nonmusic_font_visible_pdf_text"
            if args.detection_all_visible_text
            else "source_proven_supported_score_text_only"
        ),
        "recall_evaluation_authorized": True,
        "precision_evaluation_authorized": bool(
            args.detection_all_visible_text
        ),
        "hmean_evaluation_authorized": bool(
            args.detection_all_visible_text
        ),
        "hard_negative_sampling_authorized": bool(
            args.detection_all_visible_text
            and sum(retained_complete_detection_pages.values()) > 0
        ),
        "hard_negative_authorization_scope": (
            "retained_registered_page_all_visible_text_verified"
            if args.detection_all_visible_text
            else None
        ),
        "unlabelled_visible_text_may_be_present": not bool(
            args.detection_all_visible_text
        ),
        "self_contained_crops": not bool(
            args.detection_all_visible_text
        ),
        "sources": [
            source_manifest[key] for key in sorted(source_manifest)
        ],
    }
    report_path = args.output_dir / "prepare-report.json"
    _atomic_json(report_path, report)
    artifact_paths.append(report_path)
    (args.output_dir / "dataset.sha256").write_text(
        "\n".join(
            [
                *[
                    f"{sha256_file(path)}  {path.relative_to(args.output_dir).as_posix()}"
                    for path in artifact_paths
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(report["words_by_split"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
