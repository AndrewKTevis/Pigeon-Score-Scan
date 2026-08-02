from __future__ import annotations

import argparse
import io
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from train_direction_model import COMPOSITIONAL_TEST, COMPOSITIONAL_THRESHOLD, build_corpus  # noqa: E402

DEFAULT_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerifItalic.ttf",
    "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Italic.otf",
    "/usr/share/fonts/truetype/lato/Lato-Italic.ttf",
]


def tesseract_languages() -> str:
    executable = shutil.which("tesseract")
    if not executable:
        raise RuntimeError("tesseract is not installed")
    completed = subprocess.run([executable, "--list-langs"], capture_output=True, text=True, timeout=20)
    available = {line.strip() for line in completed.stdout.splitlines()[1:] if line.strip()}
    selected = [item for item in ("eng", "ita", "deu", "fra", "spa") if item in available]
    return "+".join(selected) if selected else "eng"


def render_phrase(text: str, font_path: str, rng: random.Random) -> Image.Image:
    size = rng.randint(25, 48)
    font = ImageFont.truetype(font_path, size)
    probe = Image.new("L", (20, 20), 255)
    draw = ImageDraw.Draw(probe)
    bbox = draw.textbbox((0, 0), text, font=font)
    width = max(180, bbox[2] - bbox[0] + rng.randint(35, 80))
    height = max(62, bbox[3] - bbox[1] + rng.randint(25, 50))
    background = rng.randint(230, 255)
    image = Image.new("L", (width, height), background)
    draw = ImageDraw.Draw(image)
    x = rng.randint(12, 28)
    y = max(2, (height - (bbox[3] - bbox[1])) // 2 - bbox[1] + rng.randint(-4, 4))
    ink = rng.randint(0, 48)
    draw.text((x, y), text, font=font, fill=ink)

    # Occasionally place one or two thin staff-like lines near the text baseline.
    if rng.random() < 0.45:
        line_y = rng.randint(max(1, height // 2), height - 5)
        draw.line((0, line_y, width, line_y), fill=rng.randint(80, 170), width=1)
        if rng.random() < 0.35 and line_y + 5 < height:
            draw.line((0, line_y + 5, width, line_y + 5), fill=rng.randint(110, 185), width=1)

    if rng.random() < 0.75:
        image = image.rotate(rng.uniform(-1.2, 1.2), resample=Image.Resampling.BICUBIC, fillcolor=background)
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.65, 1.35))
    if rng.random() < 0.65:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.15, 1.15)))
    array = np.asarray(image, dtype=np.float32)
    if rng.random() < 0.80:
        noise = rng.uniform(1.0, 8.0)
        np_rng = np.random.default_rng(rng.randrange(2**32))
        array += np_rng.normal(0, noise, array.shape)
    array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array, mode="L")
    if rng.random() < 0.45:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=rng.randint(35, 85))
        image = Image.open(io.BytesIO(buffer.getvalue())).convert("L")
    return image


def ocr_image(image: Image.Image, languages: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        path = Path(handle.name)
    try:
        image.save(path)
        completed = subprocess.run(
            ["tesseract", str(path), "stdout", "-l", languages, "--psm", "7"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if completed.returncode != 0:
            return ""
        return " ".join(completed.stdout.strip().split())
    finally:
        path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=320)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--languages", default=None)
    parser.add_argument(
        "--dataset",
        choices=("lexicon", "compositional-threshold", "compositional-test", "compositional-all"),
        default="lexicon",
    )
    args = parser.parse_args()

    fonts = [item for item in DEFAULT_FONTS if Path(item).exists()]
    if not fonts:
        raise RuntimeError("No suitable fonts found")
    languages = args.languages or tesseract_languages()
    rng = random.Random(args.seed)
    if args.dataset == "compositional-threshold":
        corpus = list(COMPOSITIONAL_THRESHOLD)
    elif args.dataset == "compositional-test":
        corpus = list(COMPOSITIONAL_TEST)
    elif args.dataset == "compositional-all":
        corpus = list(COMPOSITIONAL_THRESHOLD + COMPOSITIONAL_TEST)
    else:
        corpus = [text for text, _ in build_corpus()]
    # Stratified deterministic selection: short dynamics, medium directions and long phrases.
    buckets = [
        [item for item in corpus if len(item) <= 5],
        [item for item in corpus if 6 <= len(item) <= 16],
        [item for item in corpus if len(item) > 16],
    ]
    buckets = [bucket for bucket in buckets if bucket]
    records: list[dict[str, object]] = []
    for sample_index in range(args.samples):
        bucket = buckets[sample_index % len(buckets)]
        expected = rng.choice(bucket)
        font_path = rng.choice(fonts)
        image = render_phrase(expected, font_path, rng)
        observed = ocr_image(image, languages)
        records.append(
            {
                "expected": expected,
                "observed": observed,
                "font": Path(font_path).name,
                "sample": sample_index,
                "dataset": args.dataset,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n", encoding="utf-8")
    exact = sum(str(item["expected"]).casefold() == str(item["observed"]).casefold() for item in records)
    nonempty = sum(bool(item["observed"]) for item in records)
    print(json.dumps({"samples": len(records), "nonempty": nonempty, "raw_exact": exact / max(len(records), 1), "languages": languages, "dataset": args.dataset}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
