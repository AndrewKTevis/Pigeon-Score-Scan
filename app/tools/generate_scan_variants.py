from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.preview import _configure_toolkit


def _svg_to_png(svg: str, destination: Path, *, target_width: int = 2480) -> None:
    edge = shutil.which("msedge") or shutil.which("microsoft-edge")
    if edge is None:
        for candidate in (
            Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
            Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
        ):
            if candidate.is_file():
                edge = str(candidate)
                break
    if edge is None:
        raise RuntimeError("Microsoft Edge is required to rasterize Verovio SVG")
    scale = max(1.0, target_width / 706.0)
    with tempfile.TemporaryDirectory(prefix="scorescan-svg-render-") as temporary:
        root = Path(temporary)
        source = root / "page.svg"
        source.write_text(svg, encoding="utf-8")
        completed = subprocess.run(
            [
                edge,
                "--headless",
                "--disable-gpu",
                "--hide-scrollbars",
                "--run-all-compositor-stages-before-draw",
                f"--force-device-scale-factor={scale:.6f}",
                "--window-size=706,998",
                f"--user-data-dir={root / 'profile'}",
                f"--screenshot={destination.resolve()}",
                source.as_uri(),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=45,
            check=False,
        )
        deadline = time.monotonic() + 8.0
        while not destination.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        if completed.returncode != 0 or not destination.is_file():
            detail = (completed.stderr or completed.stdout).strip()[:500]
            raise RuntimeError(f"Edge SVG rasterization failed ({completed.returncode}): {detail}")
    image = cv2.imread(str(destination), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0 or int(image.min()) > 245:
        destination.unlink(missing_ok=True)
        raise RuntimeError("SVG rasterization produced an empty page")


def render_musicxml(path: Path, output: Path) -> list[Path]:
    import verovio
    toolkit = _configure_toolkit(verovio)
    toolkit.setOptions({"inputFrom": "musicxml", "breaks": "encoded", "pageWidth": 1680, "pageHeight": 2376, "scale": 45})
    if not toolkit.loadFile(str(path)):
        raise RuntimeError(f"无法渲染 {path}")
    results: list[Path] = []
    for page in range(1, int(toolkit.getPageCount()) + 1):
        svg = toolkit.renderToSVG(page)
        png = output / f"clean_page_{page:04d}.png"
        _svg_to_png(svg, png)
        results.append(png)
    return results


def degrade(source: Path, destination: Path, rng: random.Random) -> dict[str, float | int]:
    image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(source)
    angle = rng.uniform(-1.5, 1.5)
    blur_sigma = rng.uniform(0.0, 1.4)
    contrast = rng.uniform(0.72, 1.18)
    brightness = rng.uniform(-18, 14)
    noise_std = rng.uniform(0.0, 8.0)
    jpeg_quality = rng.randint(45, 96)
    height, width = image.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    result = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderValue=245)
    if blur_sigma > 0.15:
        k = max(3, int(round(blur_sigma * 4)) * 2 + 1)
        result = cv2.GaussianBlur(result, (k, k), blur_sigma)
    result = np.clip(result.astype(np.float32) * contrast + brightness, 0, 255)
    if noise_std:
        result += np.random.default_rng(rng.randrange(2**32)).normal(0, noise_std, result.shape)
    # Mild paper texture and uneven illumination.
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = rng.uniform(0, width), rng.uniform(0, height)
    radius = max(width, height) * rng.uniform(0.8, 1.8)
    shade = ((xx - cx) ** 2 + (yy - cy) ** 2) / (radius ** 2) * rng.uniform(0, 16)
    result = np.clip(result - shade, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(str(destination), decoded, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return {
        "angle": angle,
        "blur_sigma": blur_sigma,
        "contrast": contrast,
        "brightness": brightness,
        "noise_std": noise_std,
        "jpeg_quality": jpeg_quality,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variants", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    manifest = {"seed": args.seed, "items": []}
    for source_index, musicxml in enumerate(args.inputs, start=1):
        item_dir = args.output / f"score_{source_index:04d}"
        item_dir.mkdir(parents=True, exist_ok=True)
        clean_pages = render_musicxml(musicxml, item_dir)
        for page_index, clean in enumerate(clean_pages, start=1):
            for variant in range(args.variants):
                output = item_dir / f"page_{page_index:04d}_variant_{variant:03d}.png"
                params = degrade(clean, output, rng)
                manifest["items"].append({
                    "musicxml": str(musicxml), "page": page_index, "image": str(output.relative_to(args.output)), "params": params,
                })
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated {len(manifest['items'])} images")


if __name__ == "__main__":
    main()
