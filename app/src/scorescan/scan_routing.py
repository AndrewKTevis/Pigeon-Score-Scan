from __future__ import annotations

"""CPU-friendly routing of deterministic scan preprocessing candidates.

The router does not recognise music.  It estimates which image treatments are worth
running for a page from low-level scan statistics.  Runtime influence is deliberately
limited to candidate order and optional generation; MusicXML validity, musical audits,
and cross-candidate consensus remain authoritative.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .layout import PageLayout
from .model_registry import load_verified_json
from .models import PageInfo

VARIANT_NAMES = ("primary", "flat", "otsu", "adaptive", "deblock", "upscale", "staffnorm")
FEATURE_NAMES = (
    "min_dimension_scaled",
    "blur_log_scaled",
    "contrast_scaled",
    "skew_abs_scaled",
    "background_variation",
    "illumination_gradient",
    "noise_ratio",
    "ink_density",
    "edge_density",
    "blockiness",
    "staff_spacing_scaled",
    "layout_confidence",
)


@dataclass(frozen=True)
class ScanRoutingFeatures:
    min_dimension_scaled: float
    blur_log_scaled: float
    contrast_scaled: float
    skew_abs_scaled: float
    background_variation: float
    illumination_gradient: float
    noise_ratio: float
    ink_density: float
    edge_density: float
    blockiness: float
    staff_spacing_scaled: float
    layout_confidence: float

    def vector(self) -> list[float]:
        return [float(getattr(self, name)) for name in FEATURE_NAMES]

    def to_dict(self) -> dict[str, float]:
        return {name: round(float(getattr(self, name)), 7) for name in FEATURE_NAMES}


@dataclass(frozen=True)
class VariantPlan:
    ordered_variants: tuple[str, ...]
    probabilities: dict[str, float]
    model_version: str
    model_verified: bool
    features: ScanRoutingFeatures

    def should_generate(self, name: str, threshold: float = 0.10) -> bool:
        if name in {"primary", "flat", "otsu", "adaptive"}:
            return True
        return float(self.probabilities.get(name, 0.0)) >= threshold

    def to_dict(self) -> dict[str, object]:
        return {
            "ordered_variants": list(self.ordered_variants),
            "probabilities": {key: round(value, 7) for key, value in self.probabilities.items()},
            "model_version": self.model_version,
            "model_verified": self.model_verified,
            "features": self.features.to_dict(),
        }


def _blockiness(gray: np.ndarray) -> float:
    """Return a bounded JPEG-grid discontinuity proxy."""
    if min(gray.shape) < 24:
        return 0.0
    work = gray.astype(np.float32)
    vertical = np.abs(np.diff(work, axis=1))
    horizontal = np.abs(np.diff(work, axis=0))
    boundary_v = vertical[:, 7::8].mean() if vertical.shape[1] > 8 else 0.0
    boundary_h = horizontal[7::8, :].mean() if horizontal.shape[0] > 8 else 0.0
    all_v = vertical.mean() + 1e-6
    all_h = horizontal.mean() + 1e-6
    ratio = 0.5 * (boundary_v / all_v + boundary_h / all_h)
    return float(max(0.0, min(3.0, ratio)) / 3.0)


def _estimate_staff_spacing(gray: np.ndarray) -> float:
    """Estimate median staff-line spacing from horizontal dark-pixel projection."""
    work = gray
    scale = min(1.0, 1200.0 / max(gray.shape))
    if scale < 1.0:
        work = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    darkness = (255.0 - work.astype(np.float32)).mean(axis=1)
    darkness = cv2.GaussianBlur(darkness.reshape(-1, 1), (1, 5), 0).ravel()
    threshold = float(np.percentile(darkness, 88))
    peaks = np.where(darkness >= threshold)[0]
    if peaks.size < 5:
        return 14.0
    groups: list[int] = []
    start = previous = int(peaks[0])
    for raw in peaks[1:]:
        value = int(raw)
        if value - previous > 2:
            groups.append((start + previous) // 2)
            start = value
        previous = value
    groups.append((start + previous) // 2)
    diffs = [right - left for left, right in zip(groups, groups[1:]) if 3 <= right - left <= 30]
    if not diffs:
        return 14.0
    # Repeated staff intervals dominate; inter-system gaps are excluded by the cap.
    spacing = float(np.median(diffs)) / max(scale, 1e-9)
    return max(5.0, min(30.0, spacing))


def extract_scan_routing_features(
    gray: np.ndarray,
    *,
    page: PageInfo | None = None,
    layout: PageLayout | None = None,
) -> ScanRoutingFeatures:
    if gray.ndim != 2:
        raise ValueError("scan routing expects a grayscale image")
    height, width = gray.shape
    min_dimension = min(height, width)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(np.percentile(gray, 95) - np.percentile(gray, 5))

    # Low-frequency background and illumination statistics.
    reduced = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    background_variation = float(np.std(cv2.GaussianBlur(reduced, (0, 0), 5.0)))
    row_gradient = abs(float(reduced[:8].mean() - reduced[-8:].mean()))
    col_gradient = abs(float(reduced[:, :8].mean() - reduced[:, -8:].mean()))
    illumination_gradient = min(1.0, row_gradient + col_gradient)

    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    noise_ratio = float(np.mean(np.abs(gray.astype(np.float32) - denoised.astype(np.float32))) / 24.0)
    noise_ratio = max(0.0, min(1.0, noise_ratio))
    ink_density = float(np.mean(gray < 205))
    edges = cv2.Canny(gray, 50, 160)
    edge_density = float(np.count_nonzero(edges)) / max(edges.size, 1)

    spacings = [system.spacing for system in layout.systems if system.spacing > 0] if layout else []
    staff_spacing = float(np.median(spacings)) if spacings else _estimate_staff_spacing(gray)
    layout_confidence = float(layout.confidence) if layout else 0.5
    skew = abs(float(page.skew_degrees or 0.0)) if page else 0.0

    return ScanRoutingFeatures(
        min_dimension_scaled=max(0.0, min(2.5, min_dimension / 1800.0)),
        blur_log_scaled=max(0.0, min(2.5, math.log1p(max(0.0, blur)) / 6.0)),
        contrast_scaled=max(0.0, min(2.0, contrast / 120.0)),
        skew_abs_scaled=max(0.0, min(2.0, skew / 3.0)),
        background_variation=max(0.0, min(1.0, background_variation * 5.0)),
        illumination_gradient=illumination_gradient,
        noise_ratio=noise_ratio,
        ink_density=max(0.0, min(1.0, ink_density * 3.0)),
        edge_density=max(0.0, min(1.0, edge_density * 5.0)),
        blockiness=_blockiness(gray),
        staff_spacing_scaled=max(0.0, min(2.5, staff_spacing / 14.0)),
        layout_confidence=max(0.0, min(1.0, layout_confidence)),
    )


class ScanVariantRouter:
    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).resolve().parent / "resources" / "scan_variant_router.json"
        loaded = load_verified_json(model_path, "scan_variant_routing")
        payload = loaded.payload
        self.model_verified = loaded.verified
        self.model_status = loaded.status
        self.model_version = str(payload.get("model_version", "disabled"))
        self.classes = tuple(str(value) for value in payload.get("classes", ()))
        self.means = tuple(float(value) for value in payload.get("means", ()))
        self.scales = tuple(max(float(value), 1e-9) for value in payload.get("scales", ()))
        self.intercepts = tuple(float(value) for value in payload.get("intercepts", ()))
        self.coefficients = tuple(
            tuple(float(value) for value in row)
            for row in payload.get("coefficients", ())
            if isinstance(row, list)
        )
        self.enabled = (
            tuple(payload.get("feature_names", ())) == FEATURE_NAMES
            and self.classes == VARIANT_NAMES
            and len(self.means) == len(FEATURE_NAMES)
            and len(self.scales) == len(FEATURE_NAMES)
            and len(self.intercepts) == len(VARIANT_NAMES)
            and len(self.coefficients) == len(VARIANT_NAMES)
            and all(len(row) == len(FEATURE_NAMES) for row in self.coefficients)
        )

    def probabilities(self, features: ScanRoutingFeatures) -> dict[str, float]:
        if not self.enabled:
            # Conservative deterministic fallback: keep the long-standing candidates
            # dominant and only request expensive variants for obvious scan conditions.
            values = features
            raw = {
                "primary": 0.25,
                "flat": 0.18 + 0.35 * values.background_variation + 0.20 * values.illumination_gradient,
                "otsu": 0.16 + 0.25 * max(0.0, 1.0 - values.contrast_scaled),
                "adaptive": 0.14 + 0.35 * values.illumination_gradient,
                "deblock": 0.08 + 0.45 * values.blockiness + 0.20 * values.noise_ratio,
                "upscale": 0.06 + 0.45 * max(0.0, 0.85 - values.min_dimension_scaled),
                "staffnorm": 0.08 + 0.35 * min(1.0, abs(values.staff_spacing_scaled - 1.0)),
            }
        else:
            standard = [
                (value - mean) / scale
                for value, mean, scale in zip(features.vector(), self.means, self.scales, strict=True)
            ]
            logits = [
                intercept + sum(weight * value for weight, value in zip(row, standard, strict=True))
                for intercept, row in zip(self.intercepts, self.coefficients, strict=True)
            ]
            peak = max(logits)
            exps = [math.exp(max(-40.0, min(40.0, value - peak))) for value in logits]
            total = sum(exps) or 1.0
            raw = {name: value / total for name, value in zip(self.classes, exps, strict=True)}
        total = sum(max(0.0, value) for value in raw.values()) or 1.0
        return {name: max(0.0, value) / total for name, value in raw.items()}

    def plan(self, gray: np.ndarray, *, page: PageInfo | None = None, layout: PageLayout | None = None) -> VariantPlan:
        features = extract_scan_routing_features(gray, page=page, layout=layout)
        probabilities = self.probabilities(features)
        ordered = tuple(
            sorted(
                VARIANT_NAMES,
                key=lambda name: (-probabilities.get(name, 0.0), VARIANT_NAMES.index(name)),
            )
        )
        return VariantPlan(ordered, probabilities, self.model_version, self.model_verified, features)
