#!/usr/bin/env python3
"""Prepare registered scan-degraded regions from disjoint Muse OMR pairs.

Ground-truth SVG geometry is exported from each CC0 MuseScore source.  Its paired
scan-degraded PDF pages are then registered into the exact SVG page coordinate
system.  Unmatched pages and pages that fail affine or image-similarity gates are
excluded rather than silently emitting noisy bounding boxes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import cv2
import pymupdf as fitz
import numpy as np
from PIL import Image

from app.tools.acquire_muse_omr_benchmark import LICENSE, REPOSITORY, REVISION
from app.tools.muse_omr_contract import (
    BENCHMARK_SELECTION_ROLE,
    MAXIMUM_SCAN_PAGE_ASPECT_RATIO,
    SCAN_DEGRADED_IMAGE_ORIGIN,
    SCAN_PAGE_SHAPE_CONTRACT,
    TRAINING_REGION_ROLE,
    TRAINING_SELECTION_ROLE,
    scan_page_aspect_ratio,
)
from app.tools.pair_cache_pruning import (
    prune_unselected_pair_caches as _prune_unselected_pair_caches,
)
from app.tools.prepare_openscore_svg_regions import (
    COMPLETE_PAGE_TARGET_PROVENANCE,
    SVG_CLASS_TO_CATEGORY,
    _render_score,
    _tile_page,
    _write_jsonl,
    sha256_file,
    split_for_source,
)
from app.tools.semantic_target_visibility import (
    OVERSIZED_FRAGMENT_VISIBILITY_VERSION,
)


REGISTRATION_VERSION = "muse-omr-bounded-elastic-page-filter-jpeg95@8"
REGISTRATION_QUALITY_POLICY_VERSION = (
    "strict-bounded-local-correlation@1"
)
REGISTERED_IMAGE_EXTENSION = ".jpg"
REGISTERED_JPEG_QUALITY = 95
MAXIMUM_REGISTERED_STORAGE_MAE = 0.75
MAXIMUM_REGISTERED_STORAGE_P999 = 8.0
LOCAL_ALIGNMENT_MAXIMUM_DIMENSION = 1600
MINIMUM_DOWNSAMPLED_LOCAL_CORRELATION_10P = 0.85
MINIMUM_DOWNSAMPLED_MEDIAN_LOCAL_CORRELATION = 0.92
MASKED_ECC_MAXIMUM_DIMENSION = 800.0
UNMASKED_ECC_MAXIMUM_DIMENSION = 1000.0
MASKED_ECC_MAXIMUM_ITERATIONS = 160
UNMASKED_ECC_MAXIMUM_ITERATIONS = 120
ELASTIC_MINIMUM_COARSE_ECC = 0.70
ELASTIC_MAXIMUM_DIMENSION = 800.0
ELASTIC_INK_GAIN = 2.0
ELASTIC_PATCH_SIZE = 16
ELASTIC_PATCH_STRIDE = 8
ELASTIC_GRADIENT_DESCENT_ITERATIONS = 25
ELASTIC_VARIATIONAL_REFINEMENT_ITERATIONS = 5
ELASTIC_FLOW_MAXIMUM_FRACTION = 0.02
ELASTIC_SMOOTHING_SIGMA_FRACTION = 0.015
ELASTIC_MINIMUM_ABSOLUTE_JACOBIAN = 0.35
ELASTIC_MINIMUM_JACOBIAN_001 = 0.50
ELASTIC_MAXIMUM_JACOBIAN_999 = 2.0
ELASTIC_MAXIMUM_ABSOLUTE_JACOBIAN = 2.5
DEFAULT_DATASET_DIR = (
    Path(__file__).resolve().parents[2]
    / "training_data"
    / "external"
    / "training"
    / f"muse_omr_scan_train_{REVISION[:10]}"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "training_data"
    / "prepared"
    / "muse_omr_scan_regions_v1"
)


def _validate_registered_page_shapes(
    pair_id: int,
    registered_pages: list[tuple[Path, Path]],
) -> list[dict[str, Any]]:
    """Reject stitched whole-work images before tiling or cache acceptance."""

    audited: list[dict[str, Any]] = []
    for page_number, (_svg_path, scan_path) in enumerate(
        registered_pages,
        start=1,
    ):
        with Image.open(scan_path) as image:
            width, height = image.size
        aspect_ratio = scan_page_aspect_ratio(width, height)
        page = {
            "page": page_number,
            "image": scan_path.name,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
        }
        audited.append(page)
        if aspect_ratio > MAXIMUM_SCAN_PAGE_ASPECT_RATIO:
            raise ValueError(
                "registered scan page violates "
                f"{SCAN_PAGE_SHAPE_CONTRACT}: pair={pair_id}, "
                f"page={page_number}, dimensions={width}x{height}, "
                f"aspect_ratio={aspect_ratio:.6f}, "
                f"maximum={MAXIMUM_SCAN_PAGE_ASPECT_RATIO:.6f}"
            )
    if not audited:
        raise ValueError("registered score contains no accepted scan pages")
    return audited


def _stable_subset(pair_ids: Iterable[int], limit: int | None, seed: int) -> list[int]:
    values = sorted(set(int(value) for value in pair_ids))
    if limit is None or limit >= len(values):
        return values
    if limit <= 0:
        raise ValueError("pair limit must be positive")
    ranked = sorted(
        values,
        key=lambda pair_id: hashlib.sha256(
            f"{seed}\0{pair_id}".encode("ascii")
        ).hexdigest(),
    )
    return sorted(ranked[:limit])


def _load_selection(
    dataset_dir: Path,
    *,
    expected_role: str,
) -> dict[str, Any]:
    path = dataset_dir / "selection.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("repository") != REPOSITORY
        or payload.get("revision") != REVISION
        or payload.get("license") != LICENSE
        or payload.get("role") != expected_role
        or payload.get("source_image_origin")
        != SCAN_DEGRADED_IMAGE_ORIGIN
        or payload.get("production_evidence_eligible") is not False
    ):
        raise ValueError(
            "dataset selection does not match the pinned Muse OMR role"
        )
    selected = payload.get("selected_pair_ids")
    if not isinstance(selected, list) or not selected:
        raise ValueError("Muse OMR selection has no pair ids")
    if len(set(int(value) for value in selected)) != len(selected):
        raise ValueError("Muse OMR selection contains duplicate pair ids")
    work_by_pair = _selection_work_map(payload)
    if set(work_by_pair) != {int(value) for value in selected}:
        raise ValueError("Muse OMR selection work mapping is incomplete")
    if expected_role == TRAINING_SELECTION_ROLE:
        reserved = payload.get("reserved_holdout_pair_ids")
        reserved_works = payload.get("reserved_holdout_work_fingerprints")
        if (
            payload.get("training_holdout_overlap") != []
            or payload.get("training_holdout_work_overlap") != []
            or not isinstance(reserved, list)
            or not reserved
            or not isinstance(reserved_works, list)
            or not reserved_works
        ):
            raise ValueError(
                "training selection has no disjoint reserved holdout"
            )
        if set(int(value) for value in selected) & set(
            int(value) for value in reserved
        ):
            raise ValueError("training selection overlaps the reserved holdout")
        if set(work_by_pair.values()) & {
            str(value) for value in reserved_works
        }:
            raise ValueError(
                "training selection overlaps the reserved holdout works"
            )
    return payload


def _selection_work_map(payload: dict[str, Any]) -> dict[int, str]:
    rows = payload.get("pair_work_fingerprints")
    works = payload.get("selected_work_fingerprints")
    catalog_hash = str(payload.get("work_catalog_sha256", ""))
    if (
        not isinstance(rows, list)
        or not isinstance(works, list)
        or not works
        or len(catalog_hash) != 64
    ):
        raise ValueError("Muse OMR selection has no work-level isolation data")
    mapping: dict[int, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid Muse OMR pair/work row")
        pair_id = int(row.get("pair_id", -1))
        fingerprint = str(row.get("work_fingerprint", ""))
        if (
            pair_id in mapping
            or len(fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in fingerprint
            )
        ):
            raise ValueError("invalid Muse OMR pair/work mapping")
        mapping[pair_id] = fingerprint
    selected_works = {str(value) for value in works}
    if (
        len(selected_works) != len(works)
        or set(mapping.values()) != selected_works
        or int(payload.get("selected_work_count", -1))
        != len(selected_works)
    ):
        raise ValueError("Muse OMR selected work inventory is inconsistent")
    return mapping


def _load_training_selection(dataset_dir: Path) -> dict[str, Any]:
    return _load_selection(
        dataset_dir,
        expected_role=TRAINING_SELECTION_ROLE,
    )


def _selection_ids(path: Path) -> tuple[set[int], set[str], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("repository") != REPOSITORY
        or payload.get("revision") != REVISION
        or payload.get("license") != LICENSE
    ):
        raise ValueError("forbidden selection does not match pinned Muse OMR")
    selected = payload.get("selected_pair_ids")
    if not isinstance(selected, list) or not selected:
        raise ValueError("forbidden selection has no pair ids")
    values = {int(value) for value in selected}
    if len(values) != len(selected):
        raise ValueError("forbidden selection contains duplicate pair ids")
    work_by_pair = _selection_work_map(payload)
    if set(work_by_pair) != values:
        raise ValueError("forbidden selection work mapping is incomplete")
    return values, set(work_by_pair.values()), sha256_file(path)


def _ink_map(gray: np.ndarray) -> np.ndarray:
    normalized = gray.astype(np.float32) / 255.0
    # A local black-hat representation retains thin staff, glyph and text
    # strokes while suppressing paper colour, folds, shadows and even broad
    # ink stains.  A simple darkness threshold lets large damaged regions
    # dominate ECC and incorrectly rejects otherwise exact paired pages.
    kernel_size = max(7, round(max(gray.shape) * 15 / 1400))
    if kernel_size % 2 == 0:
        kernel_size += 1
    background = cv2.morphologyEx(
        normalized,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        ),
    )
    ink = np.clip(background - normalized, 0.0, 1.0)
    return cv2.GaussianBlur(ink, (5, 5), 0)


def _local_alignment_quality(
    reference_gray: np.ndarray,
    registered_gray: np.ndarray,
) -> dict[str, float | int]:
    source_height, source_width = reference_gray.shape
    reduction = min(
        1.0,
        LOCAL_ALIGNMENT_MAXIMUM_DIMENSION
        / max(source_width, source_height),
    )
    if reduction < 1.0:
        evaluation_size = (
            max(128, int(round(source_width * reduction))),
            max(128, int(round(source_height * reduction))),
        )
        reference_gray = cv2.resize(
            reference_gray,
            evaluation_size,
            interpolation=cv2.INTER_AREA,
        )
        registered_gray = cv2.resize(
            registered_gray,
            evaluation_size,
            interpolation=cv2.INTER_AREA,
        )
    reference_ink = _ink_map(reference_gray)
    registered_ink = _ink_map(registered_gray)
    correlations: list[float] = []
    height, width = reference_gray.shape
    for row in range(8):
        y0 = round(row * height / 8)
        y1 = round((row + 1) * height / 8)
        for column in range(6):
            x0 = round(column * width / 6)
            x1 = round((column + 1) * width / 6)
            reference_cell = reference_ink[y0:y1, x0:x1].reshape(-1)
            if float(np.mean(reference_cell > 0.12)) < 0.003:
                continue
            registered_cell = registered_ink[y0:y1, x0:x1].reshape(-1)
            reference_std = float(np.std(reference_cell))
            registered_std = float(np.std(registered_cell))
            if reference_std <= 1e-6 or registered_std <= 1e-6:
                correlations.append(0.0)
                continue
            correlation = float(
                np.corrcoef(reference_cell, registered_cell)[0, 1]
            )
            correlations.append(correlation if math.isfinite(correlation) else 0.0)
    if len(correlations) < 3:
        raise ValueError("registration has too few notation-bearing local cells")
    return {
        "local_quality_evaluation_width": int(reference_gray.shape[1]),
        "local_quality_evaluation_height": int(reference_gray.shape[0]),
        "local_quality_downsampled": reduction < 1.0,
        "local_cells": len(correlations),
        "local_correlation_10p": round(
            float(np.quantile(correlations, 0.1)),
            8,
        ),
        "local_correlation_median": round(float(np.median(correlations)), 8),
        "global_correlation": round(
            float(
                np.corrcoef(
                    reference_ink.reshape(-1),
                    registered_ink.reshape(-1),
                )[0, 1]
            ),
            8,
        ),
    }


def _reference_alignment_mask(reference_ink: np.ndarray) -> np.ndarray:
    """Return a generous notation-centred mask for robust ECC alignment.

    Real Muse OMR degradations can cover most of a page with paper texture,
    stains, and folds.  Those marks are not present in the reference and must
    not dominate the global objective.  Dilating the reference strokes leaves
    enough capture range for moderate displacement while keeping the
    optimisation centred on actual score content.
    """

    core = (reference_ink > 0.02).astype(np.uint8) * 255
    kernel_size = max(31, round(max(reference_ink.shape) * 0.075))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.dilate(
        core,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        ),
    )


def _ecc_affine_candidate(
    resized_scan: np.ndarray,
    reference_gray: np.ndarray,
    *,
    masked: bool,
) -> tuple[float, np.ndarray, tuple[int, int]]:
    reference_height, reference_width = reference_gray.shape
    maximum_dimension = (
        MASKED_ECC_MAXIMUM_DIMENSION
        if masked
        else UNMASKED_ECC_MAXIMUM_DIMENSION
    )
    reduction = min(
        1.0,
        maximum_dimension / max(reference_width, reference_height),
    )
    small_size = (
        max(128, int(round(reference_width * reduction))),
        max(128, int(round(reference_height * reduction))),
    )
    reference_small = cv2.resize(
        reference_gray,
        small_size,
        interpolation=cv2.INTER_AREA,
    )
    scan_small = cv2.resize(
        resized_scan,
        small_size,
        interpolation=cv2.INTER_AREA,
    )
    reference_ink = _ink_map(reference_small)
    scan_ink = _ink_map(scan_small)
    mask = _reference_alignment_mask(reference_ink) if masked else None
    warp_small = np.eye(2, 3, dtype=np.float32)
    ecc, warp_small = cv2.findTransformECC(
        reference_ink,
        scan_ink,
        warp_small,
        cv2.MOTION_AFFINE,
        (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            (
                MASKED_ECC_MAXIMUM_ITERATIONS
                if masked
                else UNMASKED_ECC_MAXIMUM_ITERATIONS
            ),
            1e-6,
        ),
        mask,
        5,
    )
    return float(ecc), warp_small, small_size


def _bounded_elastic_registration(
    registered_affine: np.ndarray,
    reference_gray: np.ndarray,
    *,
    minimum_local_correlation_10p: float,
    minimum_median_local_correlation: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove only smooth, bounded paper deformation after affine alignment.

    Muse OMR's scan degradation includes locally crumpled paper.  A whole-page
    affine transform cannot align those pages closely enough for safe object
    boxes.  DIS flow is therefore estimated on notation ink at a bounded
    resolution, heavily low-pass filtered, displacement-clipped, and rejected
    when its Jacobian can fold or strongly rescale a local region.  The
    resulting page must still pass the same strict local-correlation gate as an
    affine-only page.

    The patch and smoothing scales are deliberately larger than individual
    glyph strokes.  This lets the transform follow paper shape without giving
    it enough spatial freedom to turn one notation symbol into another.
    """

    reference_height, reference_width = reference_gray.shape
    reduction = min(
        1.0,
        ELASTIC_MAXIMUM_DIMENSION
        / max(reference_width, reference_height),
    )
    evaluation_size = (
        max(128, int(round(reference_width * reduction))),
        max(128, int(round(reference_height * reduction))),
    )
    reference_small = cv2.resize(
        reference_gray,
        evaluation_size,
        interpolation=cv2.INTER_AREA,
    )
    registered_small = cv2.resize(
        registered_affine,
        evaluation_size,
        interpolation=cv2.INTER_AREA,
    )
    reference_ink = np.clip(
        _ink_map(reference_small) * (255.0 * ELASTIC_INK_GAIN),
        0.0,
        255.0,
    ).astype(np.uint8)
    registered_ink = np.clip(
        _ink_map(registered_small) * (255.0 * ELASTIC_INK_GAIN),
        0.0,
        255.0,
    ).astype(np.uint8)

    optical_flow = cv2.DISOpticalFlow_create(
        cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    )
    optical_flow.setPatchSize(ELASTIC_PATCH_SIZE)
    optical_flow.setPatchStride(ELASTIC_PATCH_STRIDE)
    optical_flow.setFinestScale(0)
    optical_flow.setGradientDescentIterations(
        ELASTIC_GRADIENT_DESCENT_ITERATIONS
    )
    optical_flow.setVariationalRefinementIterations(
        ELASTIC_VARIATIONAL_REFINEMENT_ITERATIONS
    )
    optical_flow.setUseMeanNormalization(True)
    optical_flow.setUseSpatialPropagation(True)
    flow = optical_flow.calc(reference_ink, registered_ink, None)
    if (
        flow is None
        or flow.shape != (*reference_ink.shape, 2)
        or not np.all(np.isfinite(flow))
    ):
        raise ValueError("elastic registration produced invalid flow")

    sigma = max(
        4.0,
        max(evaluation_size) * ELASTIC_SMOOTHING_SIGMA_FRACTION,
    )
    flow = cv2.GaussianBlur(flow, (0, 0), sigma)
    magnitude = np.linalg.norm(flow, axis=2)
    maximum_magnitude = (
        max(evaluation_size) * ELASTIC_FLOW_MAXIMUM_FRACTION
    )
    flow *= np.minimum(
        1.0,
        maximum_magnitude / np.maximum(magnitude, 1e-6),
    )[..., None]
    bounded_magnitude = np.linalg.norm(flow, axis=2)

    flow_x_dx = np.gradient(flow[:, :, 0], axis=1)
    flow_x_dy = np.gradient(flow[:, :, 0], axis=0)
    flow_y_dx = np.gradient(flow[:, :, 1], axis=1)
    flow_y_dy = np.gradient(flow[:, :, 1], axis=0)
    jacobian = (
        (1.0 + flow_x_dx) * (1.0 + flow_y_dy)
        - flow_x_dy * flow_y_dx
    )
    jacobian_minimum = float(np.min(jacobian))
    jacobian_001 = float(np.quantile(jacobian, 0.001))
    jacobian_999 = float(np.quantile(jacobian, 0.999))
    jacobian_maximum = float(np.max(jacobian))
    if (
        jacobian_minimum < ELASTIC_MINIMUM_ABSOLUTE_JACOBIAN
        or jacobian_001 < ELASTIC_MINIMUM_JACOBIAN_001
        or jacobian_999 > ELASTIC_MAXIMUM_JACOBIAN_999
        or jacobian_maximum > ELASTIC_MAXIMUM_ABSOLUTE_JACOBIAN
    ):
        raise ValueError(
            "elastic deformation Jacobian outside bounds: "
            f"min={jacobian_minimum:.6f}, "
            f"q001={jacobian_001:.6f}, "
            f"q999={jacobian_999:.6f}, "
            f"max={jacobian_maximum:.6f}"
        )

    flow_x = cv2.resize(
        flow[:, :, 0],
        (reference_width, reference_height),
        interpolation=cv2.INTER_CUBIC,
    )
    flow_y = cv2.resize(
        flow[:, :, 1],
        (reference_width, reference_height),
        interpolation=cv2.INTER_CUBIC,
    )
    flow_x *= reference_width / float(evaluation_size[0])
    flow_y *= reference_height / float(evaluation_size[1])
    grid_y, grid_x = np.mgrid[
        :reference_height,
        :reference_width,
    ].astype(np.float32)
    registered = cv2.remap(
        registered_affine,
        grid_x + flow_x,
        grid_y + flow_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    local_quality = _local_alignment_quality(
        reference_gray,
        registered,
    )
    effective_minimum_local_correlation_10p = max(
        minimum_local_correlation_10p,
        MINIMUM_DOWNSAMPLED_LOCAL_CORRELATION_10P,
    )
    effective_minimum_median_local_correlation = max(
        minimum_median_local_correlation,
        MINIMUM_DOWNSAMPLED_MEDIAN_LOCAL_CORRELATION,
    )
    if (
        float(local_quality["local_correlation_10p"])
        < effective_minimum_local_correlation_10p
    ):
        raise ValueError(
            "elastic local 10th-percentile correlation "
            f"{float(local_quality['local_correlation_10p']):.6f} "
            f"< {effective_minimum_local_correlation_10p:.6f}"
        )
    if (
        float(local_quality["local_correlation_median"])
        < effective_minimum_median_local_correlation
    ):
        raise ValueError(
            "elastic median local correlation "
            f"{float(local_quality['local_correlation_median']):.6f} "
            f"< {effective_minimum_median_local_correlation:.6f}"
        )

    magnitude_quantiles = np.quantile(
        bounded_magnitude,
        (0.5, 0.95, 0.999),
    )
    return registered, {
        "elastic_flow_algorithm": "DIS-medium-lowpass-bounded@1",
        "elastic_evaluation_size": [
            int(evaluation_size[0]),
            int(evaluation_size[1]),
        ],
        "elastic_patch_size": ELASTIC_PATCH_SIZE,
        "elastic_smoothing_sigma": round(float(sigma), 8),
        "elastic_flow_maximum_fraction": ELASTIC_FLOW_MAXIMUM_FRACTION,
        "elastic_flow_magnitude_fraction_50p": round(
            float(magnitude_quantiles[0]) / max(evaluation_size),
            8,
        ),
        "elastic_flow_magnitude_fraction_95p": round(
            float(magnitude_quantiles[1]) / max(evaluation_size),
            8,
        ),
        "elastic_flow_magnitude_fraction_999": round(
            float(magnitude_quantiles[2]) / max(evaluation_size),
            8,
        ),
        "elastic_jacobian_minimum": round(jacobian_minimum, 8),
        "elastic_jacobian_001": round(jacobian_001, 8),
        "elastic_jacobian_999": round(jacobian_999, 8),
        "elastic_jacobian_maximum": round(jacobian_maximum, 8),
        "effective_minimum_local_correlation_10p": (
            effective_minimum_local_correlation_10p
        ),
        "effective_minimum_median_local_correlation": (
            effective_minimum_median_local_correlation
        ),
        **local_quality,
    }


def register_scan_page(
    scan_gray: np.ndarray,
    reference_gray: np.ndarray,
    *,
    minimum_ecc: float,
    maximum_linear_deviation: float,
    maximum_translation_fraction: float,
    minimum_local_correlation_10p: float = 0.62,
    minimum_median_local_correlation: float = 0.72,
) -> tuple[np.ndarray, dict[str, Any]]:
    if scan_gray.ndim != 2 or reference_gray.ndim != 2:
        raise ValueError("registration expects grayscale images")
    if min(scan_gray.shape) < 128 or min(reference_gray.shape) < 128:
        raise ValueError("registration images are too small")
    reference_height, reference_width = reference_gray.shape
    resized_scan = cv2.resize(
        scan_gray,
        (reference_width, reference_height),
        interpolation=cv2.INTER_AREA
        if scan_gray.shape[0] >= reference_height
        else cv2.INTER_CUBIC,
    )
    failures: list[str] = []
    for method, masked in (
        ("reference_masked_affine", True),
        ("unmasked_affine_fallback", False),
    ):
        try:
            ecc, warp_small, small_size = _ecc_affine_candidate(
                resized_scan,
                reference_gray,
                masked=masked,
            )
            if not math.isfinite(ecc):
                raise ValueError(
                    f"ECC is not finite: {ecc!r}"
                )
            linear = warp_small[:, :2]
            identity = np.eye(2, dtype=np.float32)
            linear_deviation = float(np.max(np.abs(linear - identity)))
            if linear_deviation > maximum_linear_deviation:
                raise ValueError(
                    "affine deformation "
                    f"{linear_deviation:.6f} > {maximum_linear_deviation:.6f}"
                )
            translation_fraction = max(
                abs(float(warp_small[0, 2]))
                / max(1.0, float(small_size[0])),
                abs(float(warp_small[1, 2]))
                / max(1.0, float(small_size[1])),
            )
            if translation_fraction > maximum_translation_fraction:
                raise ValueError(
                    "translation "
                    f"{translation_fraction:.6f} "
                    f"> {maximum_translation_fraction:.6f}"
                )

            warp_full = warp_small.copy()
            warp_full[0, 2] *= reference_width / float(small_size[0])
            warp_full[1, 2] *= reference_height / float(small_size[1])
            registered_affine = cv2.warpAffine(
                resized_scan,
                warp_full,
                (reference_width, reference_height),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )
            effective_minimum_local_correlation_10p = max(
                minimum_local_correlation_10p,
                MINIMUM_DOWNSAMPLED_LOCAL_CORRELATION_10P,
            )
            effective_minimum_median_local_correlation = max(
                minimum_median_local_correlation,
                MINIMUM_DOWNSAMPLED_MEDIAN_LOCAL_CORRELATION,
            )
            affine_failure: str | None = None
            if ecc >= minimum_ecc:
                local_quality = _local_alignment_quality(
                    reference_gray,
                    registered_affine,
                )
                if (
                    float(local_quality["local_correlation_10p"])
                    < effective_minimum_local_correlation_10p
                ):
                    affine_failure = (
                        "local 10th-percentile correlation "
                        f"{float(local_quality['local_correlation_10p']):.6f} "
                        f"< {effective_minimum_local_correlation_10p:.6f}"
                    )
                elif (
                    float(local_quality["local_correlation_median"])
                    < effective_minimum_median_local_correlation
                ):
                    affine_failure = (
                        "median local correlation "
                        f"{float(local_quality['local_correlation_median']):.6f} "
                        f"< {effective_minimum_median_local_correlation:.6f}"
                    )
                else:
                    report = {
                        "version": REGISTRATION_VERSION,
                        "quality_policy_version": (
                            REGISTRATION_QUALITY_POLICY_VERSION
                        ),
                        "method": method,
                        "ecc": round(ecc, 8),
                        "linear_deviation": round(linear_deviation, 8),
                        "translation_fraction": round(
                            translation_fraction,
                            8,
                        ),
                        "effective_minimum_local_correlation_10p": (
                            effective_minimum_local_correlation_10p
                        ),
                        "effective_minimum_median_local_correlation": (
                            effective_minimum_median_local_correlation
                        ),
                        "warp_scan_to_reference": [
                            [round(float(value), 8) for value in row]
                            for row in warp_full
                        ],
                        "source_size": [
                            int(scan_gray.shape[1]),
                            int(scan_gray.shape[0]),
                        ],
                        "reference_size": [
                            reference_width,
                            reference_height,
                        ],
                        **local_quality,
                    }
                    return registered_affine, report
            else:
                affine_failure = (
                    f"ECC {ecc:.6f} < {minimum_ecc:.6f}"
                )

            if affine_failure is not None:
                failures.append(f"{method}: {affine_failure}")
            if ecc < ELASTIC_MINIMUM_COARSE_ECC:
                continue
            try:
                registered, elastic_report = (
                    _bounded_elastic_registration(
                        registered_affine,
                        reference_gray,
                        minimum_local_correlation_10p=(
                            minimum_local_correlation_10p
                        ),
                        minimum_median_local_correlation=(
                            minimum_median_local_correlation
                        ),
                    )
                )
                report = {
                    "version": REGISTRATION_VERSION,
                    "quality_policy_version": (
                        REGISTRATION_QUALITY_POLICY_VERSION
                    ),
                    "method": f"{method}_bounded_elastic",
                    "ecc": round(ecc, 8),
                    "elastic_minimum_coarse_ecc": (
                        ELASTIC_MINIMUM_COARSE_ECC
                    ),
                    "linear_deviation": round(linear_deviation, 8),
                    "translation_fraction": round(
                        translation_fraction,
                        8,
                    ),
                    "warp_scan_to_reference": [
                        [round(float(value), 8) for value in row]
                        for row in warp_full
                    ],
                    "source_size": [
                        int(scan_gray.shape[1]),
                        int(scan_gray.shape[0]),
                    ],
                    "reference_size": [
                        reference_width,
                        reference_height,
                    ],
                    **elastic_report,
                }
                return registered, report
            except (cv2.error, ValueError) as exc:
                failures.append(
                    f"{method}_bounded_elastic: "
                    f"{str(exc).splitlines()[0]}"
                )
        except (cv2.error, ValueError) as exc:
            failures.append(f"{method}: {str(exc).splitlines()[0]}")
    raise ValueError(
        "registration failed all affine candidates; " + "; ".join(failures)
    )


def _render_pdf_pages(
    pdf_path: Path,
    reference_pngs: list[Path],
) -> tuple[list[np.ndarray], int]:
    document = fitz.open(str(pdf_path))
    try:
        pdf_page_count = len(document)
        pages: list[np.ndarray] = []
        # A different MuseScore version can reflow only the tail of a work.
        # Render the monotonic page-index prefix and let the strict registration
        # gates reject pages after the reflow point instead of discarding every
        # correctly matching earlier page.
        for page, reference_path in zip(document, reference_pngs):
            with Image.open(reference_path) as reference:
                reference_width, reference_height = reference.size
            scale_x = reference_width / max(float(page.rect.width), 1.0)
            scale_y = reference_height / max(float(page.rect.height), 1.0)
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(scale_x, scale_y),
                colorspace=fitz.csGRAY,
                alpha=False,
            )
            gray = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height,
                pixmap.width,
            )
            pages.append(gray.copy())
        return pages, pdf_page_count
    finally:
        document.close()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _discard_rejected_pair_cache(output_dir: Path, pair_id: int) -> None:
    if pair_id < 0:
        raise ValueError("pair id must be non-negative")
    root = output_dir.resolve()
    for relative in (
        Path("reference_pages") / f"pair-{pair_id:04d}",
        Path("pages") / f"pair-{pair_id:04d}",
    ):
        target = (output_dir / relative).resolve()
        if not target.is_relative_to(root):
            raise ValueError("pair cache path escaped output directory")
        shutil.rmtree(target, ignore_errors=True)


def _coverage_failures(
    *,
    selected_pairs: int,
    accepted_pairs: int,
    accepted_works: int,
    minimum_accepted_fraction: float,
    minimum_accepted_works: int,
) -> list[str]:
    accepted_fraction = accepted_pairs / max(1, selected_pairs)
    failures: list[str] = []
    if accepted_fraction < minimum_accepted_fraction:
        failures.append(
            "accepted pair fraction "
            f"{accepted_fraction:.6f} < {minimum_accepted_fraction:.6f}"
        )
    if accepted_works < minimum_accepted_works:
        failures.append(
            "accepted independent works "
            f"{accepted_works} < {minimum_accepted_works}"
        )
    return failures


def _load_cached_rejection(
    path: Path,
    signature: dict[str, Any],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        path.unlink(missing_ok=True)
        return None
    rejection = payload.get("rejection")
    if (
        payload.get("format") != 1
        or payload.get("signature") != signature
        or not isinstance(rejection, dict)
    ):
        path.unlink(missing_ok=True)
        return None
    return rejection


def _load_cached_acceptance(
    path: Path,
    signature: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any] | None:
    """Load a complete per-pair tile replay cache under strict path bounds."""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        path.unlink(missing_ok=True)
        return None
    accepted = payload.get("accepted")
    rows = payload.get("rows")
    counters = payload.get("counters")
    if (
        payload.get("format") != 1
        or payload.get("signature") != signature
        or not isinstance(accepted, dict)
        or not isinstance(rows, list)
        or not isinstance(counters, dict)
    ):
        path.unlink(missing_ok=True)
        return None
    dropped = counters.get("dropped_counts")
    excluded = counters.get("excluded_page_counts")
    try:
        negative_tiles = int(counters.get("negative_tiles", -1))
        valid_counters = (
            negative_tiles >= 0
            and isinstance(dropped, dict)
            and isinstance(excluded, dict)
            and all(int(value) >= 0 for value in dropped.values())
            and all(int(value) >= 0 for value in excluded.values())
        )
    except (TypeError, ValueError, OverflowError):
        valid_counters = False
    if not valid_counters:
        path.unlink(missing_ok=True)
        return None
    if not _cached_acceptance_registration_is_valid(
        accepted=accepted,
        signature=signature,
        output_dir=output_dir,
    ):
        path.unlink(missing_ok=True)
        return None
    root = output_dir.resolve()
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("objects"), list)
        ):
            path.unlink(missing_ok=True)
            return None
        relative = Path(str(row.get("image") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            path.unlink(missing_ok=True)
            return None
        image = (root / relative).resolve()
        if not image.is_relative_to(root) or not image.is_file():
            path.unlink(missing_ok=True)
            return None
    return payload


def _write_registration_progress(
    output_dir: Path,
    *,
    completed: int,
    selected_pairs: int,
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    pair_id: int,
) -> bool:
    accepted_current = bool(
        accepted and accepted[-1]["pair_id"] == pair_id
    )
    _atomic_json(
        output_dir / "registration-progress.json",
        {
            "schema_version": 1,
            "registration_version": REGISTRATION_VERSION,
            "completed_pairs": completed,
            "selected_pairs": selected_pairs,
            "accepted_pairs": len(accepted),
            "accepted_works": len(
                {
                    str(row["work_fingerprint"])
                    for row in accepted
                }
            ),
            "rejected_pairs": len(rejected),
            "accepted_fraction_so_far": (
                len(accepted) / max(1, completed)
            ),
            "last_pair_id": pair_id,
            "last_pair_status": (
                "accepted" if accepted_current else "rejected"
            ),
            "rejected": rejected,
        },
    )
    return accepted_current


def _write_registered_page(
    path: Path,
    registered: np.ndarray,
) -> dict[str, Any]:
    if registered.ndim != 2 or registered.dtype != np.uint8:
        raise ValueError("registered page must be an 8-bit grayscale image")
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    Image.fromarray(registered, mode="L").save(
        temporary,
        format="JPEG",
        quality=REGISTERED_JPEG_QUALITY,
        optimize=True,
        subsampling=0,
    )
    try:
        with Image.open(temporary) as encoded:
            encoded.load()
            decoded = np.asarray(encoded.convert("L"), dtype=np.uint8)
        if decoded.shape != registered.shape:
            raise ValueError(
                "registered JPEG dimensions changed during encoding"
            )
        difference = np.abs(
            decoded.astype(np.int16) - registered.astype(np.int16)
        ).astype(np.float32)
        mae = float(np.mean(difference))
        p999 = float(np.quantile(difference, 0.999))
        if (
            not math.isfinite(mae)
            or not math.isfinite(p999)
            or mae > MAXIMUM_REGISTERED_STORAGE_MAE
            or p999 > MAXIMUM_REGISTERED_STORAGE_P999
        ):
            raise ValueError(
                "registered JPEG storage drift exceeds audit floor: "
                f"mae={mae:.6f}, p999={p999:.6f}"
            )
        os.replace(temporary, path)
        return {
            "codec": "jpeg",
            "quality": REGISTERED_JPEG_QUALITY,
            "optimize": True,
            "subsampling": 0,
            "mean_absolute_pixel_error": round(mae, 8),
            "pixel_error_99_9p": round(p999, 8),
            "maximum_pixel_error": int(np.max(difference)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _registered_page_path(
    destination: Path,
    page: dict[str, Any],
) -> Path:
    page_number = int(page.get("page", 0))
    value = str(page.get("registered_image", ""))
    expected = f"page-{page_number}{REGISTERED_IMAGE_EXTENSION}"
    if page_number <= 0 or value != expected or Path(value).name != value:
        raise ValueError("registered page report contains an unsafe image path")
    return destination / value


def _cached_acceptance_registration_is_valid(
    *,
    accepted: dict[str, Any],
    signature: dict[str, Any],
    output_dir: Path,
) -> bool:
    """Verify the registered source behind a tile replay cache."""
    try:
        pair_id = int(accepted["pair_id"])
        if pair_id < 0:
            return False
        destination = output_dir / "pages" / f"pair-{pair_id:04d}"
        registration_path = destination / "registration.json"
        report = json.loads(registration_path.read_text(encoding="utf-8"))
        if (
            report.get("version") != REGISTRATION_VERSION
            or report.get("quality_policy_version")
            != REGISTRATION_QUALITY_POLICY_VERSION
        ):
            return False
        pages = report.get("pages")
        if not isinstance(pages, list) or not pages:
            return False
        effective_10p = max(
            float(signature["minimum_local_correlation_10p"]),
            MINIMUM_DOWNSAMPLED_LOCAL_CORRELATION_10P,
        )
        effective_median = max(
            float(signature["minimum_median_local_correlation"]),
            MINIMUM_DOWNSAMPLED_MEDIAN_LOCAL_CORRELATION,
        )
        for page in pages:
            if (
                not isinstance(page, dict)
                or page.get("quality_policy_version")
                != REGISTRATION_QUALITY_POLICY_VERSION
                or float(page["local_correlation_10p"]) < effective_10p
                or float(page["local_correlation_median"])
                < effective_median
            ):
                return False
            registered_path = _registered_page_path(destination, page)
            storage = page.get("storage")
            if (
                not registered_path.is_file()
                or not isinstance(storage, dict)
                or not isinstance(storage.get("sha256"), str)
                or sha256_file(registered_path) != storage["sha256"]
            ):
                return False
        accepted_fraction = len(pages) / max(
            1,
            int(report.get("page_denominator", len(pages))),
        )
        return accepted_fraction >= float(
            signature["minimum_accepted_page_fraction"]
        )
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        return False


def _cached_registration_under_current_quality_policy(
    *,
    report: dict[str, Any],
    destination: Path,
    reference_pages: list[tuple[Path, Path]],
    minimum_local_correlation_10p: float,
    minimum_median_local_correlation: float,
) -> dict[str, Any] | None:
    """Upgrade and validate cached registration evidence without weakening gates."""
    raw_pages = report.get("pages")
    if not isinstance(raw_pages, list) or not raw_pages:
        return None
    effective_10p = max(
        minimum_local_correlation_10p,
        MINIMUM_DOWNSAMPLED_LOCAL_CORRELATION_10P,
    )
    effective_median = max(
        minimum_median_local_correlation,
        MINIMUM_DOWNSAMPLED_MEDIAN_LOCAL_CORRELATION,
    )
    upgraded_pages: list[dict[str, Any]] = []
    changed = (
        report.get("quality_policy_version")
        != REGISTRATION_QUALITY_POLICY_VERSION
    )
    for raw_page in raw_pages:
        if not isinstance(raw_page, dict):
            return None
        page = dict(raw_page)
        try:
            page_number = int(page["page"])
            if page_number <= 0 or page_number > len(reference_pages):
                return None
            registered_path = _registered_page_path(destination, page)
            if not registered_path.is_file():
                return None
            storage = page.get("storage")
            if not isinstance(storage, dict):
                return None
            registered_sha256 = sha256_file(registered_path)
            stored_sha256 = storage.get("sha256")
            if (
                stored_sha256 is not None
                and stored_sha256 != registered_sha256
            ):
                return None
            has_current_evidence = (
                isinstance(page.get("local_quality_downsampled"), bool)
                and 0 < int(page["local_quality_evaluation_width"])
                <= LOCAL_ALIGNMENT_MAXIMUM_DIMENSION
                and 0 < int(page["local_quality_evaluation_height"])
                <= LOCAL_ALIGNMENT_MAXIMUM_DIMENSION
                and float(
                    page["effective_minimum_local_correlation_10p"]
                )
                >= effective_10p
                and float(
                    page["effective_minimum_median_local_correlation"]
                )
                >= effective_median
            )
        except (KeyError, OSError, TypeError, ValueError, OverflowError):
            return None
        if not has_current_evidence:
            reference = cv2.imread(
                str(reference_pages[page_number - 1][1]),
                cv2.IMREAD_GRAYSCALE,
            )
            registered = cv2.imread(
                str(registered_path),
                cv2.IMREAD_GRAYSCALE,
            )
            if (
                reference is None
                or registered is None
                or reference.shape != registered.shape
            ):
                return None
            page.update(_local_alignment_quality(reference, registered))
            page["effective_minimum_local_correlation_10p"] = effective_10p
            page["effective_minimum_median_local_correlation"] = (
                effective_median
            )
            changed = True
        if stored_sha256 is None:
            page["storage"] = {
                **storage,
                "sha256": registered_sha256,
            }
            changed = True
        try:
            local_10p = float(page["local_correlation_10p"])
            local_median = float(page["local_correlation_median"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if (
            not math.isfinite(local_10p)
            or not math.isfinite(local_median)
            or local_10p < effective_10p
            or local_median < effective_median
        ):
            return None
        if (
            page.get("quality_policy_version")
            != REGISTRATION_QUALITY_POLICY_VERSION
        ):
            page["quality_policy_version"] = (
                REGISTRATION_QUALITY_POLICY_VERSION
            )
            changed = True
        upgraded_pages.append(page)
    upgraded = dict(report)
    upgraded["pages"] = upgraded_pages
    upgraded["quality_policy_version"] = (
        REGISTRATION_QUALITY_POLICY_VERSION
    )
    if changed:
        _atomic_json(destination / "registration.json", upgraded)
    return upgraded


def _registered_pages(
    *,
    pair_id: int,
    pdf_path: Path,
    reference_pages: list[tuple[Path, Path]],
    output_dir: Path,
    minimum_ecc: float,
    maximum_linear_deviation: float,
    maximum_translation_fraction: float,
    minimum_local_correlation_10p: float,
    minimum_median_local_correlation: float,
    minimum_accepted_page_fraction: float,
    registration_workers: int = 1,
) -> tuple[list[tuple[Path, Path]], dict[str, Any]]:
    if registration_workers <= 0:
        raise ValueError("registration workers must be positive")
    destination = output_dir / "pages" / f"pair-{pair_id:04d}"
    registration_path = destination / "registration.json"
    registration_inputs = {
        "pdf_sha256": sha256_file(pdf_path),
        "reference_pages": [
            {
                "svg_sha256": sha256_file(svg_path),
                "png_sha256": sha256_file(png_path),
            }
            for svg_path, png_path in reference_pages
        ],
        "configuration": {
            "minimum_ecc": minimum_ecc,
            "maximum_linear_deviation": maximum_linear_deviation,
            "maximum_translation_fraction": maximum_translation_fraction,
            "minimum_local_correlation_10p": minimum_local_correlation_10p,
            "minimum_median_local_correlation": minimum_median_local_correlation,
            "minimum_accepted_page_fraction": minimum_accepted_page_fraction,
        },
    }
    if registration_path.is_file():
        try:
            report = json.loads(
                registration_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            report = None
        if (
            isinstance(report, dict)
            and report.get("version") == REGISTRATION_VERSION
            and report.get("inputs") == registration_inputs
        ):
            report = _cached_registration_under_current_quality_policy(
                report=report,
                destination=destination,
                reference_pages=reference_pages,
                minimum_local_correlation_10p=(
                    minimum_local_correlation_10p
                ),
                minimum_median_local_correlation=(
                    minimum_median_local_correlation
                ),
            )
        else:
            report = None
        if report is not None:
            accepted_page_numbers = [
                int(page["page"])
                for page in report.get("pages", [])
            ]
            registered_paths = [
                _registered_page_path(destination, page)
                for page in report.get("pages", [])
            ]
            accepted_fraction = len(registered_paths) / max(
                1,
                int(report.get("page_denominator", len(reference_pages))),
            )
            if accepted_fraction < minimum_accepted_page_fraction:
                raise ValueError(
                    "registered page acceptance below floor: "
                    f"{accepted_fraction:.6f} "
                    f"< {minimum_accepted_page_fraction:.6f}"
                )
            if not all(path.is_file() for path in registered_paths):
                raise ValueError("cached registered pages are incomplete")
            return [
                (
                    reference_pages[page_number - 1][0],
                    registered_path,
                )
                for page_number, registered_path in zip(
                    accepted_page_numbers,
                    registered_paths,
                    strict=True,
                )
            ], report

    destination.mkdir(parents=True, exist_ok=True)
    rendered_scans, pdf_page_count = _render_pdf_pages(
        pdf_path,
        [png_path for _svg_path, png_path in reference_pages],
    )
    page_reports: list[dict[str, Any]] = []
    rejected_pages: list[dict[str, Any]] = [
        {
            "page": page_number,
            "error": (
                "unpaired reference page after PDF/MuseScore page-count "
                f"mismatch: {pdf_page_count} != {len(reference_pages)}"
            ),
        }
        for page_number in range(
            len(rendered_scans) + 1,
            len(reference_pages) + 1,
        )
    ]
    def register_one(
        task: tuple[int, np.ndarray, Path, Path],
    ) -> tuple[
        int,
        Path,
        Path | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
    ]:
        page_number, scan, svg_path, reference_png = task
        registered_path = (
            destination
            / f"page-{page_number}{REGISTERED_IMAGE_EXTENSION}"
        )
        try:
            reference_gray = cv2.imread(
                str(reference_png),
                cv2.IMREAD_GRAYSCALE,
            )
            if reference_gray is None:
                raise ValueError(
                    f"cannot read reference render: {reference_png}"
                )
            registered, page_report = register_scan_page(
                scan,
                reference_gray,
                minimum_ecc=minimum_ecc,
                maximum_linear_deviation=maximum_linear_deviation,
                maximum_translation_fraction=maximum_translation_fraction,
                minimum_local_correlation_10p=minimum_local_correlation_10p,
                minimum_median_local_correlation=(
                    minimum_median_local_correlation
                ),
            )
            storage_report = _write_registered_page(
                registered_path,
                registered,
            )
            return (
                page_number,
                svg_path,
                registered_path,
                {
                    "page": page_number,
                    "registered_image": registered_path.name,
                    "storage": storage_report,
                    **page_report,
                },
                None,
            )
        except (OSError, ValueError) as exc:
            registered_path.unlink(missing_ok=True)
            return (
                page_number,
                svg_path,
                None,
                None,
                {
                    "page": page_number,
                    "error": str(exc),
                },
            )

    tasks = [
        (page_number, scan, svg_path, reference_png)
        for page_number, (scan, (svg_path, reference_png)) in enumerate(
            zip(rendered_scans, reference_pages),
            start=1,
        )
    ]
    with ThreadPoolExecutor(max_workers=registration_workers) as executor:
        results = list(executor.map(register_one, tasks))
    registered_pages: list[tuple[Path, Path]] = []
    for _page_number, svg_path, registered_path, page_report, rejection in results:
        if page_report is not None and registered_path is not None:
            page_reports.append(page_report)
            registered_pages.append((svg_path, registered_path))
        elif rejection is not None:
            rejected_pages.append(rejection)
    report = {
        "version": REGISTRATION_VERSION,
        "quality_policy_version": REGISTRATION_QUALITY_POLICY_VERSION,
        "pair_id": pair_id,
        "inputs": registration_inputs,
        "pdf_pages": pdf_page_count,
        "reference_pages": len(reference_pages),
        "page_denominator": max(pdf_page_count, len(reference_pages)),
        "unmatched_scan_pages": list(
            range(len(reference_pages) + 1, pdf_page_count + 1)
        ),
        "pages": page_reports,
        "rejected_pages": rejected_pages,
        "accepted_page_fraction": (
            len(page_reports)
            / max(1, pdf_page_count, len(reference_pages))
        ),
    }
    _atomic_json(registration_path, report)
    if report["accepted_page_fraction"] < minimum_accepted_page_fraction:
        raise ValueError(
            "registered page acceptance below floor: "
            f"{report['accepted_page_fraction']:.6f} "
            f"< {minimum_accepted_page_fraction:.6f}"
        )
    return registered_pages, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--musescore-exe", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--minimum-object-fraction", type=float, default=0.8)
    parser.add_argument(
        "--long-span-minimum-object-fraction",
        type=float,
        default=0.25,
        help=(
            "minimum visible fraction for page-spanning marks and text; "
            "complete SVG page boxes remain authoritative"
        ),
    )
    parser.add_argument("--negative-ratio", type=float, default=0.08)
    parser.add_argument("--calibration-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--minimum-ecc", type=float, default=0.86)
    parser.add_argument("--maximum-linear-deviation", type=float, default=0.1)
    parser.add_argument("--maximum-translation-fraction", type=float, default=0.1)
    parser.add_argument("--minimum-local-correlation-10p", type=float, default=0.62)
    parser.add_argument(
        "--minimum-median-local-correlation",
        type=float,
        default=0.72,
    )
    parser.add_argument(
        "--minimum-accepted-page-fraction",
        type=float,
        default=0.75,
    )
    parser.add_argument("--minimum-accepted-fraction", type=float, default=0.85)
    parser.add_argument("--minimum-accepted-works", type=int, default=1)
    parser.add_argument("--render-timeout-seconds", type=int, default=900)
    parser.add_argument("--registration-workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--expected-selection-role",
        choices=(TRAINING_SELECTION_ROLE, BENCHMARK_SELECTION_ROLE),
        default=TRAINING_SELECTION_ROLE,
    )
    parser.add_argument(
        "--forbidden-selection",
        type=Path,
        help="selection.json whose selected pair ids must not occur here",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dataset_dir.is_dir():
        raise FileNotFoundError(args.dataset_dir)
    if not args.musescore_exe.is_file():
        raise FileNotFoundError(args.musescore_exe)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(args.output_dir)
    if args.resume and (args.output_dir / "prepare-report.json").is_file():
        raise FileExistsError("refusing to resume a completed scan dataset")
    for name, value in (
        ("minimum-object-fraction", args.minimum_object_fraction),
        (
            "long-span-minimum-object-fraction",
            args.long_span_minimum_object_fraction,
        ),
        ("minimum-accepted-fraction", args.minimum_accepted_fraction),
        (
            "minimum-accepted-page-fraction",
            args.minimum_accepted_page_fraction,
        ),
    ):
        if not 0 < value <= 1:
            raise ValueError(f"{name} must be in (0, 1]")
    if (
        args.long_span_minimum_object_fraction
        > args.minimum_object_fraction
    ):
        raise ValueError(
            "long-span-minimum-object-fraction must not exceed "
            "minimum-object-fraction"
        )
    if not 0 <= args.negative_ratio <= 1:
        raise ValueError("negative-ratio must be in [0, 1]")
    if args.minimum_accepted_works <= 0:
        raise ValueError("minimum-accepted-works must be positive")
    if not 0 < args.minimum_ecc <= 1:
        raise ValueError("minimum-ecc must be in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "registration-failure-report.json").unlink(
        missing_ok=True
    )

    selection = _load_selection(
        args.dataset_dir,
        expected_role=args.expected_selection_role,
    )
    selected_pair_ids = _stable_subset(
        (int(value) for value in selection["selected_pair_ids"]),
        args.max_pairs,
        args.seed,
    )
    work_by_pair = _selection_work_map(selection)
    selected_work_fingerprints = {
        work_by_pair[pair_id] for pair_id in selected_pair_ids
    }
    forbidden_selection_sha256: str | None = None
    forbidden_overlap: list[int] = []
    forbidden_work_overlap: list[str] = []
    if args.forbidden_selection is not None:
        (
            forbidden_ids,
            forbidden_works,
            forbidden_selection_sha256,
        ) = _selection_ids(args.forbidden_selection)
        forbidden_overlap = sorted(set(selected_pair_ids) & forbidden_ids)
        forbidden_work_overlap = sorted(
            selected_work_fingerprints & forbidden_works
        )
        if forbidden_overlap or forbidden_work_overlap:
            raise ValueError(
                "Muse OMR evaluation/training selection overlap: "
                f"pairs={forbidden_overlap}, works={forbidden_work_overlap}"
            )
    categories = {
        name: index + 1
        for index, name in enumerate(sorted(set(SVG_CLASS_TO_CATEGORY.values())))
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    split_sources: dict[str, set[str]] = defaultdict(set)
    negative_tiles_by_split: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    dropped_counts: Counter[str] = Counter()
    excluded_page_counts: Counter[str] = Counter()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for completed, pair_id in enumerate(selected_pair_ids, start=1):
        mscz = args.dataset_dir / "mscz" / f"score_file_{pair_id}.mscz"
        pdf = args.dataset_dir / "pdf" / f"score_file_{pair_id}.pdf"
        if not mscz.is_file() or not pdf.is_file():
            raise FileNotFoundError(f"incomplete Muse OMR pair {pair_id}")
        work_fingerprint = work_by_pair[pair_id]
        source_key = f"muse-omr-work/{work_fingerprint}"
        variant_key = f"muse-omr/{pair_id}"
        split = (
            "test"
            if args.expected_selection_role == BENCHMARK_SELECTION_ROLE
            else split_for_source(
                source_key,
                calibration_fraction=args.calibration_fraction,
                test_fraction=args.test_fraction,
            )
        )
        mscz_sha256 = sha256_file(mscz)
        pdf_sha256 = sha256_file(pdf)
        rejection_signature = {
            "registration_version": REGISTRATION_VERSION,
            "registration_quality_policy_version": (
                REGISTRATION_QUALITY_POLICY_VERSION
            ),
            "pair_id": pair_id,
            "work_fingerprint": work_fingerprint,
            "mscz_sha256": mscz_sha256,
            "pdf_sha256": pdf_sha256,
            "minimum_ecc": args.minimum_ecc,
            "maximum_linear_deviation": args.maximum_linear_deviation,
            "maximum_translation_fraction": (
                args.maximum_translation_fraction
            ),
            "minimum_local_correlation_10p": (
                args.minimum_local_correlation_10p
            ),
            "minimum_median_local_correlation": (
                args.minimum_median_local_correlation
            ),
            "minimum_accepted_page_fraction": (
                args.minimum_accepted_page_fraction
            ),
            "scan_page_shape_contract": SCAN_PAGE_SHAPE_CONTRACT,
            "maximum_scan_page_aspect_ratio": (
                MAXIMUM_SCAN_PAGE_ASPECT_RATIO
            ),
        }
        rejection_cache = (
            args.output_dir
            / "rejections"
            / f"pair-{pair_id:04d}.json"
        )
        acceptance_signature = {
            **rejection_signature,
            "split": split,
            "source_key": source_key,
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "minimum_object_fraction": args.minimum_object_fraction,
            "long_span_minimum_object_fraction": (
                args.long_span_minimum_object_fraction
            ),
            "target_geometry_provenance": COMPLETE_PAGE_TARGET_PROVENANCE,
            "oversized_fragment_visibility_version": (
                OVERSIZED_FRAGMENT_VISIBILITY_VERSION
            ),
            "negative_ratio": args.negative_ratio,
            "categories": categories,
        }
        acceptance_cache = (
            args.output_dir
            / "acceptances"
            / f"pair-{pair_id:04d}.json"
        )
        cached_acceptance = (
            _load_cached_acceptance(
                acceptance_cache,
                acceptance_signature,
                args.output_dir,
            )
            if args.resume
            else None
        )
        if cached_acceptance is not None:
            cached_rows = list(cached_acceptance["rows"])
            cached_counters = dict(cached_acceptance["counters"])
            accepted.append(dict(cached_acceptance["accepted"]))
            rows_by_split[split].extend(cached_rows)
            split_sources[split].add(source_key)
            negative_tiles_by_split[split] += int(
                cached_counters.get("negative_tiles", 0)
            )
            dropped_counts.update(
                dict(cached_counters.get("dropped_counts") or {})
            )
            excluded_page_counts.update(
                dict(cached_counters.get("excluded_page_counts") or {})
            )
            for row in cached_rows:
                object_counts.update(
                    str(obj["category_id"])
                    for obj in row.get("objects", [])
                )
            rejection_cache.unlink(missing_ok=True)
            _write_registration_progress(
                args.output_dir,
                completed=completed,
                selected_pairs=len(selected_pair_ids),
                accepted=accepted,
                rejected=rejected,
                pair_id=pair_id,
            )
            print(
                f"[{completed}/{len(selected_pair_ids)}] pair {pair_id}: "
                "cached acceptance",
                flush=True,
            )
            continue
        cached_rejection = (
            _load_cached_rejection(
                rejection_cache,
                rejection_signature,
            )
            if args.resume
            else None
        )
        if cached_rejection is not None:
            rejected.append(cached_rejection)
            _write_registration_progress(
                args.output_dir,
                completed=completed,
                selected_pairs=len(selected_pair_ids),
                accepted=accepted,
                rejected=rejected,
                pair_id=pair_id,
            )
            print(
                f"[{completed}/{len(selected_pair_ids)}] pair {pair_id}: "
                f"cached rejection: "
                f"{str(cached_rejection['error']).splitlines()[0]}",
                flush=True,
            )
            continue
        try:
            reference_pages = _render_score(
                mscz,
                musescore_exe=args.musescore_exe,
                piece_dir=args.output_dir / "reference_pages" / f"pair-{pair_id:04d}",
                timeout_seconds=args.render_timeout_seconds,
            )
            registered_pages, registration = _registered_pages(
                pair_id=pair_id,
                pdf_path=pdf,
                reference_pages=reference_pages,
                output_dir=args.output_dir,
                minimum_ecc=args.minimum_ecc,
                maximum_linear_deviation=args.maximum_linear_deviation,
                maximum_translation_fraction=args.maximum_translation_fraction,
                minimum_local_correlation_10p=args.minimum_local_correlation_10p,
                minimum_median_local_correlation=(
                    args.minimum_median_local_correlation
                ),
                minimum_accepted_page_fraction=(
                    args.minimum_accepted_page_fraction
                ),
                registration_workers=args.registration_workers,
            )
            page_shape_audit = _validate_registered_page_shapes(
                pair_id,
                registered_pages,
            )
            positive_tiles = 0
            negative_tiles = 0
            pair_rows: list[dict[str, Any]] = []
            pair_dropped: Counter[str] = Counter()
            pair_excluded: Counter[str] = Counter()
            for svg_path, scan_path in registered_pages:
                rows, dropped, excluded, page_negatives = _tile_page(
                    split=split,
                    source_key=source_key,
                    svg_path=svg_path,
                    png_path=scan_path,
                    output_dir=args.output_dir,
                    categories=categories,
                    tile_size=args.tile_size,
                    overlap=args.overlap,
                    minimum_fraction=args.minimum_object_fraction,
                    negative_ratio=args.negative_ratio,
                    long_span_minimum_fraction=(
                        args.long_span_minimum_object_fraction
                    ),
                )
                pair_rows.extend(rows)
                positive_tiles += sum(bool(row["objects"]) for row in rows)
                negative_tiles += page_negatives
                pair_dropped.update(dropped)
                pair_excluded.update(excluded)
            page_ecc = [
                float(page["ecc"])
                for page in registration.get("pages", [])
            ]
            accepted_row = {
                "pair_id": pair_id,
                "source_key": source_key,
                "variant_key": variant_key,
                "work_fingerprint": work_fingerprint,
                "split": split,
                "pages": len(registered_pages),
                "rejected_pages": len(
                    registration.get("rejected_pages", [])
                ),
                "positive_tiles": positive_tiles,
                "negative_tiles": negative_tiles,
                "minimum_ecc": min(page_ecc),
                "median_ecc": median(page_ecc),
                "maximum_page_aspect_ratio": max(
                    float(page["aspect_ratio"])
                    for page in page_shape_audit
                ),
                "mscz_sha256": mscz_sha256,
                "pdf_sha256": pdf_sha256,
            }
            accepted.append(accepted_row)
            rows_by_split[split].extend(pair_rows)
            split_sources[split].add(source_key)
            negative_tiles_by_split[split] += negative_tiles
            dropped_counts.update(pair_dropped)
            excluded_page_counts.update(pair_excluded)
            for row in pair_rows:
                object_counts.update(
                    str(obj["category_id"]) for obj in row["objects"]
                )
            _atomic_json(
                acceptance_cache,
                {
                    "format": 1,
                    "signature": acceptance_signature,
                    "accepted": accepted_row,
                    "rows": pair_rows,
                    "counters": {
                        "negative_tiles": negative_tiles,
                        "dropped_counts": dict(pair_dropped),
                        "excluded_page_counts": dict(pair_excluded),
                    },
                },
            )
            rejection_cache.unlink(missing_ok=True)
        except (OSError, RuntimeError, ValueError) as exc:
            rejection = {
                "pair_id": pair_id,
                "source_key": source_key,
                "variant_key": variant_key,
                "work_fingerprint": work_fingerprint,
                "error": str(exc),
            }
            _atomic_json(
                rejection_cache,
                {
                    "format": 1,
                    "signature": rejection_signature,
                    "rejection": rejection,
                },
            )
            acceptance_cache.unlink(missing_ok=True)
            _discard_rejected_pair_cache(args.output_dir, pair_id)
            rejected.append(rejection)
        accepted_current = _write_registration_progress(
            args.output_dir,
            completed=completed,
            selected_pairs=len(selected_pair_ids),
            accepted=accepted,
            rejected=rejected,
            pair_id=pair_id,
        )
        rejection_suffix = (
            ""
            if accepted_current
            else f": {rejected[-1]['error'].splitlines()[0]}"
        )
        print(
            f"[{completed}/{len(selected_pair_ids)}] pair {pair_id}: "
            f"{'accepted' if accepted_current else 'rejected'}"
            f"{rejection_suffix}",
            flush=True,
        )

    accepted_fraction = len(accepted) / max(1, len(selected_pair_ids))
    accepted_work_count = len(
        {str(row["work_fingerprint"]) for row in accepted}
    )
    coverage_failures = _coverage_failures(
        selected_pairs=len(selected_pair_ids),
        accepted_pairs=len(accepted),
        accepted_works=accepted_work_count,
        minimum_accepted_fraction=args.minimum_accepted_fraction,
        minimum_accepted_works=args.minimum_accepted_works,
    )
    if coverage_failures:
        _atomic_json(
            args.output_dir / "registration-failure-report.json",
            {
                "schema_version": 1,
                "registration_version": REGISTRATION_VERSION,
                "selected_pairs": len(selected_pair_ids),
                "accepted_pairs": len(accepted),
                "accepted_works": accepted_work_count,
                "rejected_pairs": len(rejected),
                "accepted_fraction": accepted_fraction,
                "required_accepted_fraction": args.minimum_accepted_fraction,
                "required_accepted_works": args.minimum_accepted_works,
                "failures": coverage_failures,
                "accepted": accepted,
                "rejected": rejected,
            },
        )
        raise RuntimeError(
            "registered scan coverage gate failed: "
            + "; ".join(coverage_failures)
        )
    stale_cache_cleanup = _prune_unselected_pair_caches(
        args.output_dir,
        selected_pair_ids,
    )
    jsonl_paths: dict[str, Path] = {}
    for split in ("train", "calibration", "test"):
        jsonl_path = args.output_dir / f"{split}.jsonl"
        _write_jsonl(jsonl_path, rows_by_split[split])
        jsonl_paths[split] = jsonl_path
    intersections = {
        f"{left}_{right}": sorted(split_sources[left] & split_sources[right])
        for index, left in enumerate(("train", "calibration", "test"))
        for right in ("train", "calibration", "test")[index + 1 :]
    }
    if any(intersections.values()):
        raise RuntimeError(f"source split leakage detected: {intersections}")

    categories_path = args.output_dir / "categories.json"
    _atomic_json(
        categories_path,
        {
            "format": 1,
            "classes": [
                {
                    "label": label,
                    "name": name,
                    "source": (
                        "registered MuseScore SVG geometry on paired "
                        "synthetic scan-degraded render"
                    ),
                }
                for name, label in sorted(categories.items(), key=lambda item: item[1])
            ],
        },
    )
    report = {
        "schema_version": 1,
        "name": (
            "scorescan-muse-omr-registered-scan-holdout-v1"
            if args.expected_selection_role == BENCHMARK_SELECTION_ROLE
            else "scorescan-muse-omr-registered-scan-regions-v1"
        ),
        "license": LICENSE,
        "source_image_origin": SCAN_DEGRADED_IMAGE_ORIGIN,
        "production_evidence_eligible": False,
        "role": (
            TRAINING_REGION_ROLE
            if args.expected_selection_role == TRAINING_SELECTION_ROLE
            else BENCHMARK_SELECTION_ROLE
        ),
        "repository": REPOSITORY,
        "revision": REVISION,
        "registration_version": REGISTRATION_VERSION,
        "registration_quality_policy_version": (
            REGISTRATION_QUALITY_POLICY_VERSION
        ),
        "scan_page_shape_contract": SCAN_PAGE_SHAPE_CONTRACT,
        "maximum_scan_page_aspect_ratio": (
            MAXIMUM_SCAN_PAGE_ASPECT_RATIO
        ),
        "registered_image_storage": {
            "codec": "jpeg",
            "quality": REGISTERED_JPEG_QUALITY,
            "optimize": True,
            "subsampling": 0,
            "maximum_mean_absolute_pixel_error": (
                MAXIMUM_REGISTERED_STORAGE_MAE
            ),
            "maximum_pixel_error_99_9p": (
                MAXIMUM_REGISTERED_STORAGE_P999
            ),
        },
        "selected_pairs": len(selected_pair_ids),
        "selected_works": len(selected_work_fingerprints),
        "accepted_pairs": len(accepted),
        "accepted_works": accepted_work_count,
        "rejected_pairs": len(rejected),
        "accepted_fraction": accepted_fraction,
        "minimum_ecc": args.minimum_ecc,
        "maximum_linear_deviation": args.maximum_linear_deviation,
        "maximum_translation_fraction": args.maximum_translation_fraction,
        "minimum_local_correlation_10p": args.minimum_local_correlation_10p,
        "effective_minimum_local_correlation_10p": max(
            args.minimum_local_correlation_10p,
            MINIMUM_DOWNSAMPLED_LOCAL_CORRELATION_10P,
        ),
        "minimum_median_local_correlation": (
            args.minimum_median_local_correlation
        ),
        "effective_minimum_median_local_correlation": max(
            args.minimum_median_local_correlation,
            MINIMUM_DOWNSAMPLED_MEDIAN_LOCAL_CORRELATION,
        ),
        "minimum_accepted_page_fraction": (
            args.minimum_accepted_page_fraction
        ),
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "minimum_object_fraction": args.minimum_object_fraction,
        "long_span_minimum_object_fraction": (
            args.long_span_minimum_object_fraction
        ),
        "target_geometry_provenance": COMPLETE_PAGE_TARGET_PROVENANCE,
        "oversized_fragment_visibility_version": (
            OVERSIZED_FRAGMENT_VISIBILITY_VERSION
        ),
        "minimum_accepted_works": args.minimum_accepted_works,
        "selection_sha256": sha256_file(
            args.dataset_dir / "selection.json"
        ),
        "forbidden_selection_sha256": forbidden_selection_sha256,
        "forbidden_selection_overlap": forbidden_overlap,
        "forbidden_work_overlap": forbidden_work_overlap,
        "negative_ratio": args.negative_ratio,
        "tiles_by_split": {
            split: len(rows_by_split[split])
            for split in ("train", "calibration", "test")
        },
        "negative_tiles_by_split": {
            split: negative_tiles_by_split[split]
            for split in ("train", "calibration", "test")
        },
        "source_count_by_split": {
            split: len(split_sources[split])
            for split in ("train", "calibration", "test")
        },
        "split_intersections": intersections,
        "object_counts": dict(sorted(object_counts.items())),
        "dropped_object_counts": dict(sorted(dropped_counts.items())),
        "excluded_page_object_counts": dict(sorted(excluded_page_counts.items())),
        "stale_pair_cache_cleanup": stale_cache_cleanup,
        "accepted": accepted,
        "rejected": rejected,
    }
    report_path = args.output_dir / "prepare-report.json"
    _atomic_json(report_path, report)
    manifest_path = args.output_dir / "manifest.json"
    _atomic_json(
        manifest_path,
        {
            "format": 1,
            "name": report["name"],
            "license": LICENSE,
            "role": report["role"],
            "classes": len(categories),
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "oversized_fragment_visibility_version": (
                OVERSIZED_FRAGMENT_VISIBILITY_VERSION
            ),
            "scan_page_shape_contract": SCAN_PAGE_SHAPE_CONTRACT,
            "maximum_scan_page_aspect_ratio": (
                MAXIMUM_SCAN_PAGE_ASPECT_RATIO
            ),
            "source_split_overlap": 0,
            "reserved_holdout_overlap": 0,
            "forbidden_selection_overlap": forbidden_overlap,
            "forbidden_work_overlap": forbidden_work_overlap,
            "selected_pairs": len(selected_pair_ids),
            "selected_works": len(selected_work_fingerprints),
            "accepted_pairs": len(accepted),
            "accepted_works": report["accepted_works"],
            "rejected_pairs": len(rejected),
            **{
                split: {
                    "tiles": len(rows_by_split[split]),
                    "sources": len(split_sources[split]),
                    "negative_tiles": negative_tiles_by_split[split],
                }
                for split in ("train", "calibration", "test")
            },
        },
    )
    (args.output_dir / "dataset.sha256").write_text(
        "\n".join(
            [
                f"{sha256_file(categories_path)}  categories.json",
                f"{sha256_file(manifest_path)}  manifest.json",
                f"{sha256_file(report_path)}  prepare-report.json",
                *[
                    f"{sha256_file(jsonl_paths[split])}  {split}.jsonl"
                    for split in ("train", "calibration", "test")
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "accepted_pairs": len(accepted),
                "rejected_pairs": len(rejected),
                "tiles_by_split": report["tiles_by_split"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
