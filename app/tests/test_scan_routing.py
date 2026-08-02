from pathlib import Path

import cv2
import numpy as np

from scorescan.imaging import generate_omr_variants
from scorescan.models import PageInfo
from scorescan.scan_routing import ScanVariantRouter, VARIANT_NAMES, extract_scan_routing_features


def staff_page(width: int = 900, height: int = 1200) -> np.ndarray:
    image = np.full((height, width), 250, dtype=np.uint8)
    for base in (180, 430, 680, 930):
        for line in range(5):
            cv2.line(image, (60, base + line * 12), (width - 60, base + line * 12), 25, 1)
        for x in range(110, width - 100, 65):
            cv2.ellipse(image, (x, base + 24), (6, 4), -10, 0, 360, 20, -1)
            cv2.line(image, (x + 6, base + 24), (x + 6, base - 16), 20, 1)
    return image


def test_scan_router_packaged_model_and_probabilities() -> None:
    image = staff_page()
    features = extract_scan_routing_features(image)
    assert len(features.vector()) == 12
    router = ScanVariantRouter()
    assert router.enabled
    assert router.model_version == "scorescan-scan-router-1"
    plan = router.plan(image)
    assert set(plan.probabilities) == set(VARIANT_NAMES)
    assert abs(sum(plan.probabilities.values()) - 1.0) < 1e-6
    assert plan.ordered_variants[0] in VARIANT_NAMES


def test_variant_generation_is_deterministic_and_writes_plan(tmp_path: Path) -> None:
    source = tmp_path / "scan.png"
    image = staff_page(700, 900)
    cv2.imwrite(str(source), image)
    page = PageInfo(1, source.name, str(source), width=700, height=900, normalized_path=str(source))
    first = generate_omr_variants(page, tmp_path / "variants")
    second = generate_omr_variants(page, tmp_path / "variants")
    assert [name for name, _ in first] == [name for name, _ in second]
    assert first[0][0] == "primary"
    assert first[1][0] == "flat"
    assert (tmp_path / "variants" / "variant_plan.json").exists()
    assert all(path.exists() for _name, path in first)


def test_low_resolution_staff_strip_always_gets_scaled_white_context(tmp_path: Path) -> None:
    source = tmp_path / "strip.png"
    image = np.full((134, 631), 255, dtype=np.uint8)
    for line in range(5):
        cv2.line(image, (0, 32 + line * 18), (630, 32 + line * 18), 20, 2)
    cv2.imwrite(str(source), image)
    page = PageInfo(
        1,
        source.name,
        str(source),
        width=631,
        height=134,
        normalized_path=str(source),
    )

    variants = dict(generate_omr_variants(page, tmp_path / "variants"))
    upscaled = cv2.imread(str(variants["upscale"]), cv2.IMREAD_GRAYSCALE)

    assert upscaled.shape[0] >= 850
    assert upscaled.shape[1] >= 2700
    assert np.all(upscaled[:100, :] == 255)
    assert np.all(upscaled[-100:, :] == 255)
    assert (tmp_path / "variants" / "page_0001_upscale_low.png").is_file()
    assert (tmp_path / "variants" / "page_0001_upscale_high.png").is_file()
