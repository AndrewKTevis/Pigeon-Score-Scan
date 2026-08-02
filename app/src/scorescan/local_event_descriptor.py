from __future__ import annotations

"""Deterministic local symbol descriptors shared by bounded visual safety gates.

The descriptor intentionally performs no recognition.  It converts one fixed-size local
crop into density, integer-Sobel and compact geometry features so independent transaction
guards can compare source evidence without duplicating OpenCV-sensitive code.
"""

import cv2
import numpy as np

PATCH_WIDTH = 48
PATCH_HEIGHT = 48
DESCRIPTOR_WIDTH = 8
DESCRIPTOR_HEIGHT = 8

DESCRIPTOR_NAMES = (
    *(f"density_{index}" for index in range(DESCRIPTOR_WIDTH * DESCRIPTOR_HEIGHT)),
    *(f"gradient_{index}" for index in range(DESCRIPTOR_WIDTH * DESCRIPTOR_HEIGHT)),
)
SCALAR_NAMES = (
    "ink_density",
    "centre_density",
    "upper_density",
    "lower_density",
    "left_density",
    "right_density",
    "maximum_row_density",
    "maximum_column_density",
    "maximum_horizontal_run",
    "maximum_vertical_run",
    "component_count_scaled",
    "nearest_component_area",
    "nearest_component_width",
    "nearest_component_height",
    "nearest_component_fill",
    "open_centre_proxy",
    "stem_proxy",
    "compact_symbol_proxy",
)
FEATURE_NAMES = DESCRIPTOR_NAMES + SCALAR_NAMES


def _maximum_run(values: np.ndarray) -> int:
    best = 0
    current = 0
    for value in values.tolist():
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _pool_exact(values: np.ndarray, output_height: int, output_width: int) -> np.ndarray:
    height, width = values.shape
    block_height = height // output_height
    block_width = width // output_width
    return values.reshape(
        output_height, block_height, output_width, block_width
    ).mean(axis=(1, 3))


def crop_event_patch(
    image: np.ndarray,
    x_ratio: float,
    y_ratio: float,
    *,
    x_offset_pixels: int = 0,
    y_offset_pixels: int = 0,
) -> np.ndarray:
    centre_x = int(round(max(0.0, min(1.0, x_ratio)) * (image.shape[1] - 1))) + int(
        x_offset_pixels
    )
    centre_y = int(round(max(0.0, min(1.0, y_ratio)) * (image.shape[0] - 1))) + int(
        y_offset_pixels
    )
    half_width = PATCH_WIDTH // 2
    half_height = PATCH_HEIGHT // 2
    padded = np.pad(
        image,
        ((half_height, half_height), (half_width, half_width)),
        mode="constant",
        constant_values=0,
    )
    return padded[
        centre_y : centre_y + PATCH_HEIGHT,
        centre_x : centre_x + PATCH_WIDTH,
    ]


def clean_event_patch(patch: np.ndarray) -> np.ndarray:
    """Remove only staff-length fragments while preserving local notation strokes."""
    if patch.shape != (PATCH_HEIGHT, PATCH_WIDTH):
        return np.zeros((PATCH_HEIGHT, PATCH_WIDTH), dtype=np.uint8)
    horizontal = cv2.morphologyEx(
        patch,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (37, 1)),
    )
    vertical = cv2.morphologyEx(
        patch,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 43)),
    )
    return cv2.subtract(patch, cv2.max(horizontal, vertical))


def describe_event_patch(patch: np.ndarray) -> np.ndarray:
    if patch.shape != (PATCH_HEIGHT, PATCH_WIDTH):
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)

    cleaned = clean_event_patch(patch)
    binary = cleaned >= 40
    density = _pool_exact(
        binary.astype(np.float64), DESCRIPTOR_HEIGHT, DESCRIPTOR_WIDTH
    )

    # Integer Sobel keeps descriptor bytes deterministic across SIMD/OpenCV builds.
    integer = cleaned.astype(np.int32)
    bordered = np.pad(integer, ((1, 1), (1, 1)), mode="reflect")
    gx = (
        -bordered[:-2, :-2]
        + bordered[:-2, 2:]
        - 2 * bordered[1:-1, :-2]
        + 2 * bordered[1:-1, 2:]
        - bordered[2:, :-2]
        + bordered[2:, 2:]
    )
    gy = (
        -bordered[:-2, :-2]
        - 2 * bordered[:-2, 1:-1]
        - bordered[:-2, 2:]
        + bordered[2:, :-2]
        + 2 * bordered[2:, 1:-1]
        + bordered[2:, 2:]
    )
    magnitude = np.minimum(np.abs(gx) + np.abs(gy), 2040).astype(np.float64) / 2040.0
    gradient = _pool_exact(magnitude, DESCRIPTOR_HEIGHT, DESCRIPTOR_WIDTH)

    centre = binary[15:33, 13:35]
    upper = binary[4:24, 8:40]
    lower = binary[24:44, 8:40]
    left = binary[8:40, 4:24]
    right = binary[8:40, 24:44]
    row_density = np.mean(binary, axis=1)
    column_density = np.mean(binary, axis=0)

    component_count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8) * 255, 8
    )
    components: list[tuple[float, int, int, int, float]] = []
    centre_x = PATCH_WIDTH / 2.0
    centre_y = PATCH_HEIGHT / 2.0
    for index in range(1, component_count):
        _x, _y, width, height, area = stats[index]
        cx, cy = centroids[index]
        distance = float(np.hypot(cx - centre_x, cy - centre_y))
        if distance <= 22.0 and 2 <= area <= 500:
            components.append(
                (
                    distance,
                    int(area),
                    int(width),
                    int(height),
                    float(area) / max(int(width) * int(height), 1),
                )
            )
    components.sort(key=lambda item: (item[0], -item[1]))
    nearest = components[0] if components else (99.0, 0, 0, 0, 0.0)

    centre_box = binary[20:28, 18:30]
    annulus = binary[16:32, 14:34].copy()
    annulus[4:12, 4:16] = False
    open_centre_proxy = max(0.0, float(np.mean(annulus)) - float(np.mean(centre_box)))
    stem_proxy = max((_maximum_run(column) for column in binary.T), default=0) / PATCH_HEIGHT
    compact_symbol_proxy = max(
        (
            float(np.mean(binary[y : y + 14, x : x + 14]))
            for y in range(8, 27, 3)
            for x in range(8, 27, 3)
        ),
        default=0.0,
    )

    scalars = np.asarray(
        [
            float(np.mean(binary)),
            float(np.mean(centre)),
            float(np.mean(upper)),
            float(np.mean(lower)),
            float(np.mean(left)),
            float(np.mean(right)),
            float(np.max(row_density, initial=0.0)),
            float(np.max(column_density, initial=0.0)),
            max((_maximum_run(row) for row in binary), default=0) / PATCH_WIDTH,
            max((_maximum_run(column) for column in binary.T), default=0) / PATCH_HEIGHT,
            min(len(components), 8) / 8.0,
            nearest[1] / 500.0,
            nearest[2] / PATCH_WIDTH,
            nearest[3] / PATCH_HEIGHT,
            nearest[4],
            open_centre_proxy,
            stem_proxy,
            compact_symbol_proxy,
        ],
        dtype=np.float64,
    )
    return np.concatenate((density.ravel(), gradient.ravel(), scalars))


def event_patch_descriptor(
    image: np.ndarray,
    x_ratio: float,
    y_ratio: float,
    *,
    x_offset_pixels: int = 0,
    y_offset_pixels: int = 0,
) -> np.ndarray:
    return describe_event_patch(
        crop_event_patch(
            image,
            x_ratio,
            y_ratio,
            x_offset_pixels=x_offset_pixels,
            y_offset_pixels=y_offset_pixels,
        )
    )
