#!/usr/bin/env python3
"""Prepare exact text recognition crops from OpenScore MuseScore/PDF pages.

MuseScore SVG output preserves text *regions* but converts glyphs to paths and
therefore loses the underlying string.  Its PDF output retains embedded text.
This preparer filters out the SMuFL music font, extracts real words with PDF
coordinates, and maps them to the corresponding high-resolution PNG pages.

The resulting labels are suitable for OCR calibration or fine-tuning.  They
must not be used as the sole validation set for scanned input: clean rendered
text and historical scans have materially different noise distributions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from app.tools.prepare_openscore_svg_regions import (
    sha256_file,
    split_for_source,
)
from app.tools.ocr_text_contract import (
    SOURCE_TEXT_SELECTION_VERSION,
    SUPPORTED_DIRECT_TEXT_TAGS,
    SUPPORTED_TEXT_OBJECTS,
)


MUSIC_FONT_MARKERS = (
    "leland",
    "bravura",
    "emmentaler",
    "gonville",
    "musescore",
    "smufl",
)

# MuseScore may round PDF media-box dimensions and raster export dimensions
# independently.  A small anisotropy is therefore expected even when the two
# pages are the same export.  Keep the limit deliberately tight: independent
# axis mapping below corrects sub-percent rounding without accepting a cropped
# or otherwise unrelated page.
MAX_PDF_IMAGE_SCALE_ANISOTROPY = 0.005
EXHAUSTIVE_DETECTION_LABEL_CONTRACT = (
    "pdf-nonmusic-visible-text-exhaustive@2"
)


class ExcludedTextBoxError(ValueError):
    """A valid PDF fragment that is not sufficiently visible on the PNG page."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def is_usable_text(
    text: str,
    font_name: str,
    *,
    include_punctuation: bool = False,
) -> bool:
    normalized = " ".join(text.split()).strip()
    if not normalized or any(marker in font_name.casefold() for marker in MUSIC_FONT_MARKERS):
        return False
    if any(unicodedata.category(character) == "Co" for character in normalized):
        return False
    # Punctuation-only fragments are not stable recognition targets for the
    # semantic OCR corpus. They are nevertheless visible text and must be
    # labelled in the exhaustive detector corpus; otherwise an isolated "="
    # in a tempo mark could be turned into negative supervision.
    return include_punctuation or any(
        character.isalnum() for character in normalized
    )


def text_token_keys(value: str) -> tuple[str, ...]:
    result = []
    for token in value.split():
        normalized = unicodedata.normalize("NFKC", token).casefold()
        key = "".join(character for character in normalized if character.isalnum())
        if key:
            result.append(key)
    return tuple(result)


def _musescore_xml_root(source: Path) -> ET.Element:
    if source.suffix.casefold() == ".mscx":
        return ET.parse(source).getroot()
    if source.suffix.casefold() != ".mscz":
        raise ValueError(f"unsupported MuseScore source: {source}")
    with zipfile.ZipFile(source) as archive:
        candidates = sorted(
            name
            for name in archive.namelist()
            if name.casefold().endswith(".mscx")
            and not name.casefold().startswith("meta-inf/")
            and "excerpts" not in {
                component.casefold()
                for component in name.replace("\\", "/").split("/")[:-1]
            }
        )
        if len(candidates) != 1:
            raise ValueError(
                "MuseScore archive must contain exactly one non-excerpt score "
                f"XML: {source}"
            )
        return ET.fromstring(archive.read(candidates[0]))


def _local_tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _supported_source_token_keys(value: str, *, context: str) -> tuple[str, ...]:
    normalized = " ".join(value.split()).strip()
    if not normalized:
        return ()
    # Paragraph-sized SystemText is commonly pasted lyrics or editorial prose.
    # It is outside the supported score-text boundary.
    maximum_characters = 128 if context == "SystemText" else 256
    maximum_tokens = 16 if context == "SystemText" else 32
    keys = text_token_keys(normalized)
    if len(normalized) > maximum_characters or len(keys) > maximum_tokens:
        return ()
    # Numeric PDF fragments cannot be distinguished from tuplet/fingering
    # glyphs using source XML alone. The base OCR model already covers digits.
    return tuple(key for key in keys if not key.isdecimal())


def source_text_token_counts(
    source: Path,
    *,
    included_contexts: Counter[str] | None = None,
    excluded_contexts: Counter[str] | None = None,
) -> tuple[Counter[str], Counter[str]]:
    """Return lyric and source-proven supported score-text token counts."""

    root = _musescore_xml_root(source)
    parent_by_id = {
        id(child): parent
        for parent in root.iter()
        for child in parent
    }
    lyric_counts: Counter[str] = Counter()
    supported_counts: Counter[str] = Counter()
    for element in root.iter():
        tag = _local_tag(element)
        if tag in SUPPORTED_DIRECT_TEXT_TAGS:
            keys = _supported_source_token_keys(
                element.text or "",
                context=tag,
            )
            supported_counts.update(keys)
            if included_contexts is not None:
                included_contexts[tag] += len(keys)
            continue
        if tag != "text":
            continue
        parent = parent_by_id.get(id(element))
        context = _local_tag(parent) if parent is not None else ""
        raw_keys = text_token_keys(element.text or "")
        if context == "Lyrics":
            lyric_counts.update(raw_keys)
            if excluded_contexts is not None:
                excluded_contexts["Lyrics"] += len(raw_keys)
            continue
        if context not in SUPPORTED_TEXT_OBJECTS:
            if excluded_contexts is not None:
                excluded_contexts[context or "<root>"] += len(raw_keys)
            continue
        keys = _supported_source_token_keys(
            element.text or "",
            context=context,
        )
        supported_counts.update(keys)
        if included_contexts is not None:
            included_contexts[context] += len(keys)
        if excluded_contexts is not None:
            excluded_contexts[f"{context}:policy"] += len(raw_keys) - len(keys)
    return lyric_counts, supported_counts


def consume_source_text_role(
    value: str,
    *,
    remaining_lyrics: Counter[str],
    remaining_non_lyrics: Counter[str],
    ambiguous_keys: set[str] | frozenset[str] = frozenset(),
) -> str:
    """Match one PDF word to source-proven text without guessing symbols."""

    keys = text_token_keys(value)
    if len(keys) != 1:
        return "unproven"
    key = keys[0]
    if key in ambiguous_keys:
        return "ambiguous"
    if remaining_non_lyrics[key] > 0:
        remaining_non_lyrics[key] -= 1
        return "supported"
    if remaining_lyrics[key] > 0:
        remaining_lyrics[key] -= 1
        return "lyric"
    return "unproven"


def reuse_rendered_pdf(
    source_path: Path,
    output_path: Path,
    *,
    expected_sha256: str,
) -> bool:
    """Reuse a hash-verified PDF via hard link, with a copy fallback."""

    if output_path.is_file() and output_path.stat().st_size > 0:
        if sha256_file(output_path) != expected_sha256:
            raise ValueError(f"existing reused PDF hash mismatch: {output_path}")
        return True
    if not source_path.is_file() or source_path.stat().st_size <= 0:
        return False
    if sha256_file(source_path) != expected_sha256:
        raise ValueError(f"reusable PDF hash mismatch: {source_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.stem}.reuse.tmp.pdf")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source_path, temporary)
    except OSError:
        shutil.copy2(source_path, temporary)
    os.replace(temporary, output_path)
    return True


def map_pdf_box_to_image(
    box: tuple[float, float, float, float],
    *,
    pdf_width: float,
    pdf_height: float,
    image_width: int,
    image_height: int,
    padding_pixels: float = 2.0,
) -> tuple[float, float, float, float]:
    if pdf_width <= 0 or pdf_height <= 0 or image_width <= 0 or image_height <= 0:
        raise ValueError("page dimensions must be positive")
    scale_x = image_width / pdf_width
    scale_y = image_height / pdf_height
    if abs(scale_x / scale_y - 1.0) > MAX_PDF_IMAGE_SCALE_ANISOTROPY:
        raise ValueError(
            f"PDF/PNG page aspect mismatch: scales {scale_x:.6f}, {scale_y:.6f}"
        )
    left, top, right, bottom = box
    scaled = (
        left * scale_x,
        top * scale_y,
        right * scale_x,
        bottom * scale_y,
    )
    if (
        not all(math.isfinite(value) for value in scaled)
        or scaled[2] <= scaled[0]
        or scaled[3] <= scaled[1]
    ):
        raise ExcludedTextBoxError(
            "invalid_geometry",
            f"PDF text fragment has invalid mapped geometry: {scaled}",
        )
    visible = (
        max(0.0, scaled[0]),
        max(0.0, scaled[1]),
        min(float(image_width), scaled[2]),
        min(float(image_height), scaled[3]),
    )
    if visible[2] <= visible[0] or visible[3] <= visible[1]:
        raise ExcludedTextBoxError(
            "outside_page",
            f"PDF text box is outside rendered page: {scaled}",
        )
    area = (scaled[2] - scaled[0]) * (scaled[3] - scaled[1])
    visible_area = (visible[2] - visible[0]) * (visible[3] - visible[1])
    visible_fraction = visible_area / area
    if visible_fraction < 0.8:
        raise ExcludedTextBoxError(
            "insufficient_visible_fraction",
            "PDF text box is materially clipped by rendered page: "
            f"fraction={visible_fraction:.6f}, box={scaled}",
        )
    mapped = (
        max(0.0, visible[0] - padding_pixels),
        max(0.0, visible[1] - padding_pixels),
        min(float(image_width), visible[2] + padding_pixels),
        min(float(image_height), visible[3] + padding_pixels),
    )
    return mapped


def extract_page_words(
    page: Any,
    *,
    image_width: int,
    image_height: int,
    padding_pixels: float = 2.0,
    exclusion_counts: Counter[str] | None = None,
    include_nonmusic_punctuation: bool = False,
) -> list[dict[str, Any]]:
    words = page.extract_words(
        x_tolerance=1,
        y_tolerance=1,
        keep_blank_chars=False,
        use_text_flow=False,
        extra_attrs=["fontname", "size"],
    )
    result = []
    for word in words:
        text = " ".join(str(word["text"]).split()).strip()
        font_name = str(word.get("fontname", ""))
        if not is_usable_text(
            text,
            font_name,
            include_punctuation=include_nonmusic_punctuation,
        ):
            continue
        try:
            box = map_pdf_box_to_image(
                (
                    float(word["x0"]),
                    float(word["top"]),
                    float(word["x1"]),
                    float(word["bottom"]),
                ),
                pdf_width=float(page.width),
                pdf_height=float(page.height),
                image_width=image_width,
                image_height=image_height,
                padding_pixels=padding_pixels,
            )
        except ExcludedTextBoxError as exc:
            if exclusion_counts is not None:
                exclusion_counts[exc.reason] += 1
            continue
        result.append(
            {
                "text": text,
                "box_xyxy": [round(value, 3) for value in box],
                "font_name": font_name,
                "font_size_pt": round(float(word.get("size", 0)), 3),
            }
        )
    return result


def _page_number(path: Path) -> int:
    match = re.search(r"-(\d+)$", path.stem)
    if not match:
        raise ValueError(f"page image has no numeric suffix: {path}")
    return int(match.group(1))


def select_source_shard(
    sources: list[Path],
    *,
    shard_count: int,
    shard_index: int,
) -> list[Path]:
    if shard_count <= 0:
        raise ValueError("shard-count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    selected = [
        source
        for index, source in enumerate(sources)
        if index % shard_count == shard_index
    ]
    if not selected:
        raise ValueError("source shard is empty")
    return selected


def _render_pdf(
    source: Path,
    *,
    musescore_exe: Path,
    output_path: Path,
    timeout_seconds: int,
    reuse_existing: bool = False,
) -> None:
    if reuse_existing and output_path.is_file() and output_path.stat().st_size > 0:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(musescore_exe), "-o", str(output_path), str(source)],
        check=True,
        timeout=timeout_seconds,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"MuseScore produced no PDF for {source}")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--region-dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--musescore-exe", type=Path, required=True)
    parser.add_argument(
        "--source-list",
        type=Path,
        help="UTF-8 file of corpus-relative .mscx paths rendered in the region dataset",
    )
    parser.add_argument("--max-scores", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--calibration-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--render-timeout-seconds", type=int, default=900)
    parser.add_argument(
        "--include-lyrics",
        action="store_true",
        help="include lyric words (disabled by default because lyrics are out of scope)",
    )
    parser.add_argument(
        "--detection-all-visible-text",
        action="store_true",
        help=(
            "retain every non-music-font PDF word for exhaustive text "
            "detection labels; these rows are not recognition/semantic labels"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse rendered PDFs in an incomplete output directory",
    )
    parser.add_argument(
        "--reuse-pdf-dir",
        type=Path,
        help="reuse same-named PDFs from another preparation directory",
    )
    parser.add_argument(
        "--write-crops",
        action="store_true",
        help="write self-contained word crops and PaddleOCR label files",
    )
    parser.add_argument("--crop-padding-pixels", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.corpus_dir.is_dir():
        raise FileNotFoundError(args.corpus_dir)
    if not args.region_dataset_dir.is_dir():
        raise FileNotFoundError(args.region_dataset_dir)
    if not args.musescore_exe.is_file():
        raise FileNotFoundError(args.musescore_exe)
    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
        and not args.resume
    ):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    if args.resume and (args.output_dir / "prepare-report.json").is_file():
        raise FileExistsError("refusing to resume a completed text dataset")
    if args.crop_padding_pixels < 0:
        raise ValueError("crop-padding-pixels must be non-negative")
    if args.reuse_pdf_dir is not None and not args.reuse_pdf_dir.is_dir():
        raise FileNotFoundError(args.reuse_pdf_dir)
    reusable_sources: dict[str, dict[str, Any]] = {}
    reuse_report_path: Path | None = None
    if args.reuse_pdf_dir is not None:
        reuse_report_path = args.reuse_pdf_dir.parent / "prepare-report.json"
        if not reuse_report_path.is_file():
            raise FileNotFoundError(reuse_report_path)
        reuse_report = json.loads(reuse_report_path.read_text(encoding="utf-8"))
        reuse_rows = reuse_report.get("sources")
        if not isinstance(reuse_rows, list) or not reuse_rows:
            raise ValueError("reusable PDF report has no source manifest")
        for row in reuse_rows:
            if not isinstance(row, dict):
                raise ValueError("reusable PDF source manifest is malformed")
            key = str(row.get("source_key", ""))
            if not key or key in reusable_sources:
                raise ValueError("reusable PDF source identity is invalid")
            reusable_sources[key] = row

    corpus_root = args.corpus_dir.resolve()
    sources = sorted(args.corpus_dir.rglob("*.mscx"))
    if args.source_list is not None:
        if not args.source_list.is_file():
            raise FileNotFoundError(args.source_list)
        relative_sources = [
            Path(line.strip())
            for line in args.source_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(set(relative_sources)) != len(relative_sources):
            raise ValueError("source list contains duplicate paths")
        sources = []
        for relative in relative_sources:
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe source-list path: {relative}")
            source = (corpus_root / relative).resolve()
            if corpus_root not in source.parents:
                raise ValueError(f"source-list path escapes corpus: {relative}")
            if source.suffix.casefold() != ".mscx" or not source.is_file():
                raise FileNotFoundError(source)
            sources.append(source)
        sources.sort()
    if args.max_scores is not None:
        if args.max_scores <= 0:
            raise ValueError("max-scores must be positive")
        sources = sorted(
            sources,
            key=lambda path: hashlib.sha256(
                path.resolve().relative_to(corpus_root).as_posix().encode("utf-8")
            ).hexdigest(),
        )[: args.max_scores]
        sources.sort()
    sources = select_source_shard(
        sources,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    if not sources:
        raise ValueError("no MuseScore sources found")

    import pdfplumber
    from PIL import Image

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_counts: Counter[str] = Counter()
    word_counts: Counter[str] = Counter()
    font_counts: Counter[str] = Counter()
    excluded_lyric_counts: Counter[str] = Counter()
    excluded_ambiguous_counts: Counter[str] = Counter()
    excluded_unproven_counts: Counter[str] = Counter()
    excluded_geometry_counts: dict[str, Counter[str]] = defaultdict(Counter)
    text_page_counts: Counter[str] = Counter()
    hard_negative_page_counts: Counter[str] = Counter()
    geometry_excluded_page_counts: Counter[str] = Counter()
    included_source_contexts: Counter[str] = Counter()
    excluded_source_contexts: Counter[str] = Counter()
    source_manifest = []
    split_sources: dict[str, set[str]] = defaultdict(set)

    for source_position, source in enumerate(sources, start=1):
        source_key = source.resolve().relative_to(corpus_root).as_posix()
        split = split_for_source(
            source_key,
            calibration_fraction=args.calibration_fraction,
            test_fraction=args.test_fraction,
        )
        split_sources[split].add(source_key)
        piece_id = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:20]
        source_sha256 = sha256_file(source)
        page_dir = args.region_dataset_dir / "pages" / piece_id
        png_pages = sorted(page_dir.glob("page-*.png"), key=_page_number)
        if not png_pages:
            raise FileNotFoundError(f"rendered PNG pages missing for {source_key}")
        pdf_path = args.output_dir / "pdf" / f"{piece_id}.pdf"
        if args.detection_all_visible_text:
            # Exhaustive detection is defined directly by the rendered PDF.
            # Parsing source XML here does not affect a single label and adds
            # substantial overhead on large corpora.
            lyric_tokens: Counter[str] = Counter()
            non_lyric_tokens: Counter[str] = Counter()
        else:
            lyric_tokens, non_lyric_tokens = source_text_token_counts(
                source,
                included_contexts=included_source_contexts,
                excluded_contexts=excluded_source_contexts,
            )
        ambiguous_keys = set(lyric_tokens) & set(non_lyric_tokens)
        remaining_lyric_tokens = lyric_tokens.copy()
        remaining_non_lyric_tokens = non_lyric_tokens.copy()
        source_excluded_lyrics = 0
        if args.reuse_pdf_dir is not None:
            reuse_row = reusable_sources.get(source_key)
            if (
                reuse_row is None
                or reuse_row.get("source_sha256") != source_sha256
                or not isinstance(reuse_row.get("pdf_sha256"), str)
            ):
                raise ValueError(
                    f"reusable PDF provenance mismatch for {source_key}"
                )
            reuse_rendered_pdf(
                args.reuse_pdf_dir / pdf_path.name,
                pdf_path,
                expected_sha256=str(reuse_row["pdf_sha256"]),
            )
        _render_pdf(
            source,
            musescore_exe=args.musescore_exe,
            output_path=pdf_path,
            timeout_seconds=args.render_timeout_seconds,
            reuse_existing=args.resume,
        )
        source_words = 0
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) != len(png_pages):
                raise ValueError(
                    f"PDF/PNG page count mismatch for {source_key}: "
                    f"{len(pdf.pages)} vs {len(png_pages)}"
                )
            for page_index, (page, png_path) in enumerate(
                zip(pdf.pages, png_pages), start=1
            ):
                with Image.open(png_path) as image_source:
                    image = image_source.convert("RGB")
                    image_width, image_height = image.size
                    geometry_exclusions_before = sum(
                        excluded_geometry_counts[split].values()
                    )
                    words = extract_page_words(
                        page,
                        image_width=image_width,
                        image_height=image_height,
                        padding_pixels=args.crop_padding_pixels,
                        exclusion_counts=excluded_geometry_counts[split],
                        include_nonmusic_punctuation=(
                            args.detection_all_visible_text
                        ),
                    )
                    page_geometry_exclusions = (
                        sum(excluded_geometry_counts[split].values())
                        - geometry_exclusions_before
                    )
                    page_negative_authorized = bool(
                        args.detection_all_visible_text
                        and words
                        and page_geometry_exclusions == 0
                    )
                    if args.detection_all_visible_text and words:
                        text_page_counts[split] += 1
                        if page_negative_authorized:
                            hard_negative_page_counts[split] += 1
                        else:
                            geometry_excluded_page_counts[split] += 1
                    relative_image = png_path.relative_to(
                        args.region_dataset_dir
                    ).as_posix()
                    retained_word_index = 0
                    for word in words:
                        if args.detection_all_visible_text:
                            # extract_page_words has already rejected music
                            # fonts, private-use glyphs and punctuation-only
                            # fragments.  A clean MuseScore PDF/PNG pair gives
                            # exact coordinates for every remaining word,
                            # including lyrics and numbers.  Semantics are
                            # intentionally deferred until after OCR.
                            role = "visible_text"
                        else:
                            role = consume_source_text_role(
                                word["text"],
                                remaining_lyrics=remaining_lyric_tokens,
                                remaining_non_lyrics=remaining_non_lyric_tokens,
                                ambiguous_keys=ambiguous_keys,
                            )
                            if role == "lyric" and not args.include_lyrics:
                                source_excluded_lyrics += 1
                                excluded_lyric_counts[split] += 1
                                continue
                            if role == "ambiguous":
                                excluded_ambiguous_counts[split] += 1
                                continue
                            if role == "unproven":
                                excluded_unproven_counts[split] += 1
                                continue
                        row = {
                            "split": split,
                            "source_key": source_key,
                            "page": page_index,
                            "word_index": retained_word_index,
                            "image": relative_image,
                            "text_role": role,
                            **word,
                        }
                        if args.detection_all_visible_text:
                            row["hard_negative_sampling_authorized"] = (
                                page_negative_authorized
                            )
                            row["page_geometry_exclusion_count"] = (
                                page_geometry_exclusions
                            )
                        if args.write_crops:
                            left, top, right, bottom = (
                                float(value) for value in word["box_xyxy"]
                            )
                            box = (
                                max(0, math.floor(left)),
                                max(0, math.floor(top)),
                                min(image.width, math.ceil(right)),
                                min(image.height, math.ceil(bottom)),
                            )
                            relative_crop = (
                                Path("crops")
                                / split
                                / f"{piece_id}-page-{page_index:03d}"
                                f"-word-{retained_word_index:04d}.png"
                            )
                            crop_path = args.output_dir / relative_crop
                            crop_path.parent.mkdir(parents=True, exist_ok=True)
                            temporary = crop_path.with_name(
                                f"{crop_path.stem}.tmp{crop_path.suffix}"
                            )
                            image.crop(box).save(
                                temporary,
                                format="PNG",
                                optimize=True,
                            )
                            os.replace(temporary, crop_path)
                            row["crop_image"] = relative_crop.as_posix()
                        rows_by_split[split].append(row)
                        retained_word_index += 1
                        source_words += 1
                        word_counts[word["text"].casefold()] += 1
                        font_counts[word["font_name"]] += 1
        source_counts[split] += 1
        source_manifest.append(
            {
                "source_key": source_key,
                "source_sha256": source_sha256,
                "split": split,
                "pages": len(png_pages),
                "words": source_words,
                "excluded_unambiguous_lyric_words": source_excluded_lyrics,
                "pdf_sha256": sha256_file(pdf_path),
            }
        )
        print(
            f"[{source_position}/{len(sources)}] "
            f"{piece_id}: {len(png_pages)} pages, {source_words} words",
            flush=True,
        )

    all_splits = ("train", "calibration", "test")
    artifact_paths: list[Path] = []
    for split in all_splits:
        jsonl_path = args.output_dir / f"{split}.jsonl"
        _write_jsonl(jsonl_path, rows_by_split[split])
        artifact_paths.append(jsonl_path)
        if args.write_crops:
            labels_path = args.output_dir / f"{split}.paddle.txt"
            with labels_path.open("w", encoding="utf-8", newline="\n") as stream:
                for row in rows_by_split[split]:
                    text = str(row["text"])
                    if any(character in text for character in "\t\r\n"):
                        raise ValueError(f"unsafe OCR label: {text!r}")
                    stream.write(f"{row['crop_image']}\t{text}\n")
            artifact_paths.append(labels_path)
    intersections = {
        f"{left}_{right}": sorted(split_sources[left] & split_sources[right])
        for index, left in enumerate(all_splits)
        for right in all_splits[index + 1 :]
    }
    if any(intersections.values()):
        raise RuntimeError(f"source leakage detected: {intersections}")
    total_geometry_exclusions = sum(
        sum(excluded_geometry_counts[split].values())
        for split in all_splits
    )
    globally_exhaustive_detection = bool(
        args.detection_all_visible_text
        and total_geometry_exclusions == 0
    )
    report = {
        "schema_version": 1,
        "license": "CC0-1.0",
        "purpose": (
            "rendered exhaustive visible-text detection training/calibration"
            if args.detection_all_visible_text
            else "rendered exact-text OCR training/calibration"
        ),
        "source_text_selection_version": SOURCE_TEXT_SELECTION_VERSION,
        "warning": "clean rendered text is not a substitute for real-scan validation",
        "corpus_dir": str(args.corpus_dir.resolve()),
        "region_dataset_dir": str(args.region_dataset_dir.resolve()),
        "musescore_exe": str(args.musescore_exe.resolve()),
        "source_shard": {
            "count": args.shard_count,
            "index": args.shard_index,
        },
        "reused_pdf_dir": (
            str(args.reuse_pdf_dir.resolve())
            if args.reuse_pdf_dir is not None
            else None
        ),
        "reused_pdf_report_sha256": (
            sha256_file(reuse_report_path)
            if reuse_report_path is not None
            else None
        ),
        "sources": source_manifest,
        "sources_by_split": dict(source_counts),
        "words_by_split": {
            split: len(rows_by_split[split]) for split in all_splits
        },
        "excluded_unambiguous_lyric_words_by_split": {
            split: excluded_lyric_counts[split] for split in all_splits
        },
        "excluded_ambiguous_source_words_by_split": {
            split: excluded_ambiguous_counts[split] for split in all_splits
        },
        "excluded_unproven_pdf_words_by_split": {
            split: excluded_unproven_counts[split] for split in all_splits
        },
        "included_source_token_contexts": dict(
            sorted(included_source_contexts.items())
        ),
        "excluded_source_token_contexts": dict(
            sorted(excluded_source_contexts.items())
        ),
        "excluded_pdf_geometry_words_by_split": {
            split: dict(sorted(excluded_geometry_counts[split].items()))
            for split in all_splits
        },
        "text_pages_by_split": {
            split: text_page_counts[split] for split in all_splits
        },
        "hard_negative_authorized_pages_by_split": {
            split: hard_negative_page_counts[split] for split in all_splits
        },
        "geometry_excluded_text_pages_by_split": {
            split: geometry_excluded_page_counts[split]
            for split in all_splits
        },
        "lyrics_included": bool(
            args.include_lyrics or args.detection_all_visible_text
        ),
        "detection_label_contract": (
            EXHAUSTIVE_DETECTION_LABEL_CONTRACT
            if args.detection_all_visible_text
            else None
        ),
        "all_usable_pdf_text_included": bool(
            args.detection_all_visible_text
        ),
        "positive_region_coverage": (
            "all_nonmusic_font_visible_pdf_text"
            if args.detection_all_visible_text
            else "source_proven_supported_score_text_only"
        ),
        "recall_evaluation_authorized": True,
        "precision_evaluation_authorized": (
            globally_exhaustive_detection
        ),
        "hmean_evaluation_authorized": (
            globally_exhaustive_detection
        ),
        "hard_negative_sampling_authorized": bool(
            args.detection_all_visible_text
            and sum(hard_negative_page_counts.values()) > 0
        ),
        "hard_negative_authorization_scope": (
            "page_without_pdf_geometry_exclusions"
            if args.detection_all_visible_text
            else None
        ),
        "unlabelled_visible_text_may_be_present": (
            not args.detection_all_visible_text
            or total_geometry_exclusions > 0
        ),
        "self_contained_crops": bool(args.write_crops),
        "crop_padding_pixels": args.crop_padding_pixels,
        "unique_casefolded_words": len(word_counts),
        "most_common_words": word_counts.most_common(100),
        "font_counts": dict(font_counts),
        "split_intersections": intersections,
    }
    report_path = args.output_dir / "prepare-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "dataset.sha256").write_text(
        "\n".join(
            [
                f"{sha256_file(report_path)}  prepare-report.json",
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
