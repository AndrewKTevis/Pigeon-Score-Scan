from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .models import PageInfo
from .layout import PageLayout
from .util import atomic_write_json, sha256_file
from .scan_routing import ScanVariantRouter
from .policy import DEFAULT_POLICY


def _estimate_skew(gray: np.ndarray) -> float:
    height, width = gray.shape
    scale = min(1.0, 1800.0 / max(width, height))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(small, 60, 180, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 1800,
        threshold=max(80, small.shape[1] // 5),
        minLineLength=max(120, small.shape[1] // 3),
        maxLineGap=max(10, small.shape[1] // 50),
    )
    if lines is None:
        return 0.0
    angles: list[float] = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = map(float, line)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if -8 <= angle <= 8:
            angles.append(angle)
    return float(np.median(angles)) if angles else 0.0


def preprocess_page(page: PageInfo, output_dir: Path) -> PageInfo:
    source = Path(page.image_path)
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        page.quality_notes.append("页面无法由图像处理模块读取")
        page.quality_score = 0.0
        page.normalized_path = page.image_path
        return page

    # Product policy: never rotate a submitted page automatically.  Earlier
    # releases let a single-staff orientation classifier apply quarter turns to
    # full-score pages.  A confidently wrong prediction could therefore destroy
    # all horizontal staff evidence before OMR.  Orientation must now be corrected
    # by the user at the source; preprocessing preserves the submitted dimensions
    # and direction unconditionally.
    corrected = image
    page.orientation_degrees = 0
    page.orientation_probability = None
    page.orientation_margin = None
    page.orientation_applied = False
    page.orientation_model_version = None
    page.orientation_model_status = "automatic-rotation-disabled"
    page.orientation_probabilities = {}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(np.percentile(gray, 95) - np.percentile(gray, 5))
    skew = _estimate_skew(gray)

    if abs(skew) >= 0.5:
        page.quality_notes.append(
            f"检测到页面倾斜约 {skew:.1f}°；程序不会自动旋转，请在源文件中确认方向"
        )

    corrected_gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(16, 16))
    enhanced = clahe.apply(corrected_gray)
    # Mild denoise and sharpening preserve thin staff lines better than aggressive binarization.
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=3, templateWindowSize=7, searchWindowSize=21)
    sharpened = cv2.addWeighted(enhanced, 1.30, denoised, -0.30, 0)

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"page_{page.index:04d}.png"
    cv2.imwrite(str(output), sharpened, [cv2.IMWRITE_PNG_COMPRESSION, 2])

    current_height, current_width = corrected.shape[:2]
    min_dimension = min(current_width, current_height)
    resolution_score = min(100.0, max(0.0, min_dimension / 18.0))
    blur_score = min(100.0, max(0.0, blur / 4.0))
    contrast_score = min(100.0, max(0.0, contrast / 1.5))
    skew_penalty = min(30.0, abs(skew) * 5.0)
    quality = max(0.0, min(100.0, 0.35 * resolution_score + 0.35 * blur_score + 0.30 * contrast_score - skew_penalty))

    page.height, page.width = current_height, current_width
    page.blur_score = round(blur, 2)
    page.contrast_score = round(contrast, 2)
    page.skew_degrees = round(skew, 3)
    page.quality_score = round(quality, 1)
    page.normalized_path = str(output)
    page.sha256 = sha256_file(output)

    if min_dimension < 1200:
        page.quality_notes.append("页面像素尺寸偏低，细小附点和奏法记号可能受影响")
    if blur < 80:
        page.quality_notes.append("页面较模糊，将启用增强处理")
    if contrast < 90:
        page.quality_notes.append("页面对比较低，将启用局部对比度增强")
    if quality < 45:
        page.quality_notes.append("扫描质量较低，程序仍会继续转换并加强复核")
    return page


def generate_omr_variants(
    page: PageInfo,
    output_dir: Path,
    *,
    layout: PageLayout | None = None,
) -> list[tuple[str, Path]]:
    """Create deterministic, staff-preserving candidates ordered by a bounded router.

    The bundled CPU model only controls ordering and whether expensive restoration
    candidates are generated.  Long-standing baseline treatments remain available,
    and later MusicXML validation/consensus remains authoritative.
    """
    source = Path(page.normalized_path or page.image_path)
    gray = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return [("primary", source)]

    output_dir.mkdir(parents=True, exist_ok=True)
    router = ScanVariantRouter()
    plan = router.plan(gray, page=page, layout=layout)
    plan_path = output_dir / "variant_plan.json"
    atomic_write_json(plan_path, plan.to_dict())
    page.variant_plan_path = str(plan_path)
    page.variant_router_model = plan.model_version
    page.variant_order = list(plan.ordered_variants)

    generated: dict[str, Path] = {"primary": source}

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_path = output_dir / f"page_{page.index:04d}_otsu.png"
    cv2.imwrite(str(otsu_path), otsu, [cv2.IMWRITE_PNG_COMPRESSION, 2])
    generated["otsu"] = otsu_path

    block_size = max(31, (min(gray.shape) // 60) | 1)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 13
    )
    adaptive_path = output_dir / f"page_{page.index:04d}_adaptive.png"
    cv2.imwrite(str(adaptive_path), adaptive, [cv2.IMWRITE_PNG_COMPRESSION, 2])
    generated["adaptive"] = adaptive_path

    kernel_size = max(31, (min(gray.shape) // 35) | 1)
    background = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    flattened = cv2.divide(gray, background, scale=245)
    flattened = cv2.normalize(flattened, None, 0, 255, cv2.NORM_MINMAX)
    flat_path = output_dir / f"page_{page.index:04d}_flat.png"
    cv2.imwrite(str(flat_path), flattened, [cv2.IMWRITE_PNG_COMPRESSION, 2])
    generated["flat"] = flat_path

    if plan.should_generate("deblock", DEFAULT_POLICY.optional_variant_probability):
        # Edge-preserving deblocking helps heavily compressed scans without erasing
        # augmentation dots as a median filter often does.
        deblocked = cv2.bilateralFilter(gray, 5, 28, 28)
        deblocked = cv2.addWeighted(gray, 0.45, deblocked, 0.55, 0)
        deblock_path = output_dir / f"page_{page.index:04d}_deblock.png"
        cv2.imwrite(str(deblock_path), deblocked, [cv2.IMWRITE_PNG_COMPRESSION, 2])
        generated["deblock"] = deblock_path

    height, width = gray.shape
    lowres_strip = (
        height <= DEFAULT_POLICY.lowres_strip_height_ceiling
        and width / max(height, 1) >= DEFAULT_POLICY.lowres_strip_aspect_floor
    )
    if lowres_strip or plan.should_generate("upscale", DEFAULT_POLICY.optional_variant_probability):
        if lowres_strip:
            targets = (
                ("low", DEFAULT_POLICY.lowres_strip_low_target_height),
                ("", DEFAULT_POLICY.lowres_strip_target_height),
                ("high", DEFAULT_POLICY.lowres_strip_high_target_height),
            )
            used_scales: set[float] = set()
            upscaled = None
            for label, target_height in targets:
                scale = max(
                    1.0,
                    min(
                        DEFAULT_POLICY.lowres_strip_max_scale,
                        float(target_height) / max(height, 1),
                    ),
                )
                scale_key = round(scale, 6)
                if scale_key in used_scales:
                    continue
                used_scales.add(scale_key)
                treatment = cv2.resize(
                    gray,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )
                vertical_margin = max(
                    24,
                    int(
                        round(
                            treatment.shape[0]
                            * DEFAULT_POLICY.lowres_strip_vertical_margin_ratio
                        )
                    ),
                )
                horizontal_margin = max(
                    24,
                    int(
                        round(
                            treatment.shape[0]
                            * DEFAULT_POLICY.lowres_strip_horizontal_margin_ratio
                        )
                    ),
                )
                treatment = cv2.copyMakeBorder(
                    treatment,
                    vertical_margin,
                    vertical_margin,
                    horizontal_margin,
                    horizontal_margin,
                    cv2.BORDER_CONSTANT,
                    value=255,
                )
                suffix = f"_{label}" if label else ""
                target = output_dir / f"page_{page.index:04d}_upscale{suffix}.png"
                cv2.imwrite(
                    str(target),
                    treatment,
                    [cv2.IMWRITE_PNG_COMPRESSION, 2],
                )
                if not label:
                    upscaled = treatment
                    upscale_path = target
            if upscaled is None:
                # The centre target can coincide with the capped low target only on
                # extremely tiny strips.  Promote the first distinct treatment to the
                # centre filename without leaving a duplicate internal vote.
                first_path = output_dir / f"page_{page.index:04d}_upscale_low.png"
                upscale_path = output_dir / f"page_{page.index:04d}_upscale.png"
                if not first_path.is_file():
                    raise RuntimeError("failed to generate low-resolution upscale")
                first_path.replace(upscale_path)
        else:
            minimum = min(gray.shape)
            scale = max(
                1.25,
                min(
                    1.85,
                    float(DEFAULT_POLICY.upscale_target_min_dimension)
                    / max(minimum, 1),
                ),
            )
            upscaled = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
            upscaled = cv2.GaussianBlur(upscaled, (0, 0), 0.35)
            upscale_path = output_dir / f"page_{page.index:04d}_upscale.png"
            cv2.imwrite(str(upscale_path), upscaled, [cv2.IMWRITE_PNG_COMPRESSION, 2])
        generated["upscale"] = upscale_path

    if layout is not None and layout.systems:
        spacings = [system.spacing for system in layout.systems if system.spacing > 0]
        if spacings:
            median_spacing = float(np.median(spacings))
            target_spacing = 14.0
            scale = max(0.68, min(1.55, target_spacing / median_spacing))
            if abs(scale - 1.0) >= 0.10 or plan.should_generate("staffnorm", threshold=DEFAULT_POLICY.staffnorm_variant_probability):
                interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
                normalised = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interpolation)
                norm_path = output_dir / f"page_{page.index:04d}_staffnorm.png"
                cv2.imwrite(str(norm_path), normalised, [cv2.IMWRITE_PNG_COMPRESSION, 2])
                generated["staffnorm"] = norm_path

    ordered = ["primary"]
    ordered.extend(name for name in plan.ordered_variants if name != "primary" and name in generated)
    # Guarantee the established primary/flat pair is evaluated before an early stop,
    # while preserving learned ordering for all remaining candidates.
    if "flat" in generated and "flat" in ordered:
        ordered.remove("flat")
        ordered.insert(1, "flat")
    return [(name, generated[name]) for name in ordered[: DEFAULT_POLICY.max_page_candidates]]
