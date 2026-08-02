#!/usr/bin/env python3
"""Prepare leakage-safe semantic-region tiles from MuseScore SVG exports.

OpenScore String Quartets is CC0 and contains editable MuseScore sources.
MuseScore's SVG renderer preserves useful semantic class names on paths
(``SlurSegment``, ``TieSegment``, ``Dynamic``, ``Text`` and others), which
provides exact rendered geometry for training a positioned notation verifier.

This preparer deliberately keeps this synthetic/rendered test set separate
from the real-scan OLiMPiC calibration and candidate sets.  A source score is
assigned wholly to one split, so pages and tiles from one work cannot leak.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.tools.prepare_deepscores_expression_tiles import (
    choose_tile,
    grid_starts,
)
from app.tools.semantic_target_visibility import (
    OVERSIZED_FRAGMENT_VISIBILITY_VERSION,
    intersection_box,
    target_fragment_is_visible,
)


# Broad semantic geometry complements DeepScores' fine-grained symbol names.
# It is not valid to relabel a generic rendered Dynamic as a particular p/f
# class without reading and aligning the score semantics.
SVG_CLASS_TO_CATEGORY = {
    "Accidental": "genericAccidental",
    "Arpeggio": "arpeggio",
    "Articulation": "genericArticulation",
    "BarLine": "genericBarline",
    "Beam": "beam",
    "Breath": "breathMark",
    "Bracket": "bracket",
    "Clef": "genericClef",
    "Dynamic": "genericDynamic",
    "Expression": "expressionText",
    "Fermata": "fermata",
    "Fingering": "fingeringText",
    "GlissandoSegment": "glissando",
    "HairpinSegment": "hairpin",
    "Hook": "flag",
    "InstrumentName": "instrumentNameText",
    "Jump": "jumpText",
    "KeySig": "genericKeySignature",
    "Marker": "markerText",
    "MeasureNumber": "measureNumberText",
    "NoteDot": "augmentationDot",
    "Ornament": "genericOrnament",
    "OttavaSegment": "ottava",
    "Parenthesis": "parenthesis",
    "PedalSegment": "pedal",
    "PlayTechAnnotation": "techniqueText",
    "RehearsalMark": "rehearsalMarkText",
    "Rest": "genericRest",
    "SlurSegment": "slur",
    "StaffText": "staffText",
    "StemSlash": "graceSlash",
    "SystemText": "systemText",
    "Tempo": "tempoText",
    "Text": "scoreText",
    "TextLineSegment": "textLine",
    "TieSegment": "tie",
    "TimeSig": "genericTimeSignature",
    "TremoloSingleChord": "tremoloSingle",
    "TremoloTwoChord": "tremoloBetweenNotes",
    "TrillSegment": "trillExtension",
    "Tuplet": "tuplet",
    "VoltaSegment": "volta",
}

# These classes commonly span a large fraction of a system or page.  Treating
# every target as a compact glyph silently drops or truncates the exact classes
# that need the most positional supervision (slurs, hairpins, pedal/ottava
# lines, long text, and multi-staff structure).  The lower floor is used only
# to retain a sufficiently large visible fragment; the complete page-space box
# remains attached to every owner target for contradiction-free expansion.
LONG_SPAN_SEMANTIC_CATEGORIES = frozenset(
    {
        "beam",
        "bracket",
        "genericBarline",
        "glissando",
        "hairpin",
        "ottava",
        "pedal",
        "slur",
        "textLine",
        "tie",
        "trillExtension",
        "tuplet",
        "volta",
    }
)
COMPLETE_PAGE_TARGET_PROVENANCE = "complete-page-svg-geometry-before-tile-clipping@1"
COMPLETE_PAGE_TARGET_ASSIGNMENT_VERSION = (
    "complete-page-overlap-consistent-deduplicated-semantic-targets@4"
)

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_PATH_TOKEN_RE = re.compile(rf"[A-Za-z]|{_NUMBER}")
_MATRIX_RE = re.compile(
    rf"^\s*matrix\(\s*({_NUMBER})[\s,]+({_NUMBER})[\s,]+"
    rf"({_NUMBER})[\s,]+({_NUMBER})[\s,]+({_NUMBER})[\s,]+"
    rf"({_NUMBER})\s*\)\s*$"
)

Point = tuple[float, float]
Box = tuple[float, float, float, float]
Matrix = tuple[float, float, float, float, float, float]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_unit(value: str) -> float:
    integer = int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)
    return integer / float(0xFFFFFFFFFFFFFFFF)


def split_for_source(
    source_key: str,
    *,
    calibration_fraction: float,
    test_fraction: float,
) -> str:
    if calibration_fraction < 0 or test_fraction < 0:
        raise ValueError("split fractions must be non-negative")
    if calibration_fraction + test_fraction >= 1:
        raise ValueError("calibration_fraction + test_fraction must be < 1")
    value = _stable_unit(source_key.replace("\\", "/").casefold())
    if value < test_fraction:
        return "test"
    if value < test_fraction + calibration_fraction:
        return "calibration"
    return "train"


def parse_view_box(value: str) -> Box:
    values = [float(item) for item in re.findall(_NUMBER, value)]
    if len(values) != 4 or not all(math.isfinite(item) for item in values):
        raise ValueError(f"invalid SVG viewBox: {value!r}")
    left, top, width, height = values
    if width <= 0 or height <= 0:
        raise ValueError(f"non-positive SVG viewBox: {value!r}")
    return left, top, left + width, top + height


def parse_matrix(value: str | None) -> Matrix:
    if not value:
        return 1.0, 0.0, 0.0, 1.0, 0.0, 0.0
    match = _MATRIX_RE.fullmatch(value)
    if not match:
        raise ValueError(f"unsupported SVG transform: {value!r}")
    matrix = tuple(float(item) for item in match.groups())
    if not all(math.isfinite(item) for item in matrix):
        raise ValueError(f"non-finite SVG transform: {value!r}")
    return matrix  # type: ignore[return-value]


def transform_point(point: Point, matrix: Matrix) -> Point:
    x, y = point
    a, b, c, d, e, f = matrix
    return a * x + c * y + e, b * x + d * y + f


def _cubic_value(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    one_minus_t = 1.0 - t
    return (
        one_minus_t**3 * p0
        + 3 * one_minus_t**2 * t * p1
        + 3 * one_minus_t * t**2 * p2
        + t**3 * p3
    )


def _cubic_extrema_parameters(
    p0: float, p1: float, p2: float, p3: float
) -> list[float]:
    # derivative / 3 = a*t^2 + b*t + c
    a = -p0 + 3 * p1 - 3 * p2 + p3
    b = 2 * (p0 - 2 * p1 + p2)
    c = p1 - p0
    epsilon = 1e-12
    if abs(a) < epsilon:
        if abs(b) < epsilon:
            return []
        root = -c / b
        return [root] if 0 < root < 1 else []
    discriminant = b * b - 4 * a * c
    if discriminant < -epsilon:
        return []
    discriminant = max(0.0, discriminant)
    square_root = math.sqrt(discriminant)
    roots = [(-b - square_root) / (2 * a), (-b + square_root) / (2 * a)]
    return sorted({root for root in roots if 0 < root < 1})


def _cubic_points(
    start: Point, control1: Point, control2: Point, end: Point
) -> list[Point]:
    parameters = {0.0, 1.0}
    parameters.update(
        _cubic_extrema_parameters(
            start[0], control1[0], control2[0], end[0]
        )
    )
    parameters.update(
        _cubic_extrema_parameters(
            start[1], control1[1], control2[1], end[1]
        )
    )
    return [
        (
            _cubic_value(start[0], control1[0], control2[0], end[0], value),
            _cubic_value(start[1], control1[1], control2[1], end[1], value),
        )
        for value in sorted(parameters)
    ]


def _box_for_points(points: Sequence[Point]) -> Box:
    if not points:
        raise ValueError("cannot bound an empty point set")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    if not all(math.isfinite(value) for value in (*xs, *ys)):
        raise ValueError("non-finite SVG geometry")
    return min(xs), min(ys), max(xs), max(ys)


def _expand_for_stroke(box: Box, stroke_width: float, matrix: Matrix) -> Box:
    if stroke_width <= 0:
        return box
    a, b, c, d, _e, _f = matrix
    maximum_scale = max(math.hypot(a, b), math.hypot(c, d))
    padding = stroke_width * maximum_scale / 2.0
    return (
        box[0] - padding,
        box[1] - padding,
        box[2] + padding,
        box[3] + padding,
    )


def clip_small_page_overflow(box: Box, view_box: Box) -> Box:
    """Clip renderer edge bleed, but reject materially misplaced geometry."""

    intersection = (
        max(box[0], view_box[0]),
        max(box[1], view_box[1]),
        min(box[2], view_box[2]),
        min(box[3], view_box[3]),
    )
    intersection_width = max(0.0, intersection[2] - intersection[0])
    intersection_height = max(0.0, intersection[3] - intersection[1])
    object_area = max(1e-9, (box[2] - box[0]) * (box[3] - box[1]))
    intersection_fraction = intersection_width * intersection_height / object_area
    page_width = view_box[2] - view_box[0]
    page_height = view_box[3] - view_box[1]
    maximum_x_overflow = max(16.0, page_width * 0.02)
    maximum_y_overflow = max(16.0, page_height * 0.02)
    overflows = (
        max(0.0, view_box[0] - box[0]),
        max(0.0, view_box[1] - box[1]),
        max(0.0, box[2] - view_box[2]),
        max(0.0, box[3] - view_box[3]),
    )
    if (
        intersection_fraction < 0.95
        or overflows[0] > maximum_x_overflow
        or overflows[2] > maximum_x_overflow
        or overflows[1] > maximum_y_overflow
        or overflows[3] > maximum_y_overflow
    ):
        raise ValueError(
            "semantic geometry materially outside page viewBox: "
            f"box={box}, viewBox={view_box}, "
            f"intersection_fraction={intersection_fraction:.6f}"
        )
    return intersection


def path_bbox(
    path_data: str,
    *,
    matrix: Matrix | None = None,
    stroke_width: float = 0.0,
) -> Box:
    """Return the exact axis-aligned bound for MuseScore M/L/C SVG paths.

    The parser fails closed on unsupported or relative commands.  This avoids
    silently manufacturing misplaced annotations if a future MuseScore
    renderer changes its SVG dialect.
    """

    matrix = matrix or parse_matrix(None)
    tokens = _PATH_TOKEN_RE.findall(path_data)
    if not tokens:
        raise ValueError("empty SVG path")
    index = 0
    command: str | None = None
    current: Point | None = None
    subpath_start: Point | None = None
    bounded_points: list[Point] = []

    def read_number() -> float:
        nonlocal index
        if index >= len(tokens) or tokens[index].isalpha():
            raise ValueError("malformed SVG path")
        value = float(tokens[index])
        index += 1
        if not math.isfinite(value):
            raise ValueError("non-finite SVG path coordinate")
        return value

    while index < len(tokens):
        if tokens[index].isalpha():
            command = tokens[index]
            index += 1
            if command not in {"M", "L", "C", "Z"}:
                raise ValueError(f"unsupported SVG path command: {command}")
            if command == "Z":
                if current is None or subpath_start is None:
                    raise ValueError("malformed SVG close command")
                bounded_points.extend(
                    [transform_point(current, matrix), transform_point(subpath_start, matrix)]
                )
                current = subpath_start
                command = None
                continue
        if command is None:
            raise ValueError("SVG path coordinates have no command")
        if command == "M":
            point = (read_number(), read_number())
            current = point
            subpath_start = point
            bounded_points.append(transform_point(point, matrix))
            # Additional coordinate pairs after M are implicit L commands.
            command = "L"
        elif command == "L":
            if current is None:
                raise ValueError("SVG line has no starting point")
            point = (read_number(), read_number())
            bounded_points.extend(
                [transform_point(current, matrix), transform_point(point, matrix)]
            )
            current = point
        elif command == "C":
            if current is None:
                raise ValueError("SVG cubic has no starting point")
            control1 = (read_number(), read_number())
            control2 = (read_number(), read_number())
            end = (read_number(), read_number())
            transformed = [
                transform_point(point, matrix)
                for point in (current, control1, control2, end)
            ]
            bounded_points.extend(_cubic_points(*transformed))
            current = end
    return _expand_for_stroke(_box_for_points(bounded_points), stroke_width, matrix)


def polyline_bbox(
    points_value: str,
    *,
    matrix: Matrix | None = None,
    stroke_width: float = 0.0,
) -> Box:
    matrix = matrix or parse_matrix(None)
    values = [float(item) for item in re.findall(_NUMBER, points_value)]
    if not values or len(values) % 2:
        raise ValueError("invalid SVG polyline points")
    points = [
        transform_point((values[index], values[index + 1]), matrix)
        for index in range(0, len(values), 2)
    ]
    return _expand_for_stroke(_box_for_points(points), stroke_width, matrix)


def _mapped_svg_objects(
    svg_path: Path,
    class_mapping: dict[str, str],
) -> tuple[Box, list[dict[str, Any]], str, Counter[str]]:
    root = ET.parse(svg_path).getroot()
    view_box = parse_view_box(root.attrib.get("viewBox", ""))
    description = ""
    objects: list[dict[str, Any]] = []
    excluded_page_objects: Counter[str] = Counter()
    for element in root.iter():
        local_name = element.tag.rsplit("}", 1)[-1]
        if local_name == "desc":
            description = (element.text or "").strip()
            continue
        svg_class = element.attrib.get("class", "")
        category = class_mapping.get(svg_class)
        if category is None:
            continue
        matrix = parse_matrix(element.attrib.get("transform"))
        stroke_width = float(element.attrib.get("stroke-width", "0"))
        if local_name == "path":
            box = path_bbox(
                element.attrib.get("d", ""),
                matrix=matrix,
                stroke_width=stroke_width,
            )
        elif local_name == "polyline":
            box = polyline_bbox(
                element.attrib.get("points", ""),
                matrix=matrix,
                stroke_width=stroke_width,
            )
        else:
            raise ValueError(
                f"unsupported semantic SVG element {local_name!r} in {svg_path}"
            )
        if (
            box[0] < view_box[0]
            or box[1] < view_box[1]
            or box[2] > view_box[2]
            or box[3] > view_box[3]
        ):
            try:
                box = clip_small_page_overflow(box, view_box)
            except ValueError:
                # MuseScore can emit page numbers, measure numbers, or text
                # paths mostly outside the printable viewBox.  Those paths
                # have no complete rendered target and must not become
                # truncated detector labels.  Count and exclude only this
                # known page-boundary condition; unsupported SVG geometry
                # and parsing errors still fail closed above.
                excluded_page_objects[category] += 1
                continue
        if box[2] <= box[0] or box[3] <= box[1]:
            # Rendered hairlines are made usable by their stroke width; a
            # remaining zero-area object cannot train an object detector.
            continue
        objects.append(
            {
                "svg_class": svg_class,
                "category": category,
                "box_xyxy": [round(value, 4) for value in box],
            }
        )
    return view_box, objects, description, excluded_page_objects


def svg_class_objects(
    svg_path: Path,
    class_names: Iterable[str],
) -> tuple[Box, list[dict[str, Any]], str, Counter[str]]:
    """Read exact geometry for an explicit, non-semantic SVG class allowlist.

    The production semantic detector intentionally excludes noteheads and staff
    geometry.  Scan-backed local safety models still need those exact rendered
    anchors, so this separate entry point exposes only classes named by the
    caller and never changes the detector category contract.
    """
    normalized = sorted(
        {
            str(name).strip()
            for name in class_names
            if str(name).strip()
        }
    )
    if not normalized:
        raise ValueError("at least one SVG class name is required")
    return _mapped_svg_objects(
        svg_path,
        {name: name for name in normalized},
    )


def semantic_svg_objects(
    svg_path: Path,
) -> tuple[Box, list[dict[str, Any]], str, Counter[str]]:
    return _mapped_svg_objects(svg_path, SVG_CLASS_TO_CATEGORY)


def _page_number(path: Path) -> int:
    match = re.search(r"-(\d+)$", path.stem)
    if not match:
        raise ValueError(f"rendered page has no numeric suffix: {path}")
    return int(match.group(1))


def _render_score(
    source: Path,
    *,
    musescore_exe: Path,
    piece_dir: Path,
    timeout_seconds: int,
) -> list[tuple[Path, Path]]:
    piece_dir.mkdir(parents=True, exist_ok=True)
    existing_svg_pages = sorted(piece_dir.glob("page-*.svg"), key=_page_number)
    existing_png_pages = sorted(piece_dir.glob("page-*.png"), key=_page_number)
    if existing_svg_pages or existing_png_pages:
        if existing_svg_pages and [
            _page_number(path) for path in existing_svg_pages
        ] == [_page_number(path) for path in existing_png_pages]:
            return list(zip(existing_svg_pages, existing_png_pages))
        # MuseScore writes every SVG page before it starts the PNG export. An
        # interrupted render can therefore leave a complete-looking SVG prefix
        # and a shorter PNG prefix. Only discard the bounded page artifacts
        # owned by this exact piece before rendering the piece again.
        for partial_page in (*existing_svg_pages, *existing_png_pages):
            partial_page.unlink()
    svg_base = piece_dir / "page.svg"
    png_base = piece_dir / "page.png"
    for output in (svg_base, png_base):
        subprocess.run(
            [str(musescore_exe), "-o", str(output), str(source)],
            check=True,
            timeout=timeout_seconds,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    svg_pages = sorted(piece_dir.glob("page-*.svg"), key=_page_number)
    png_pages = sorted(piece_dir.glob("page-*.png"), key=_page_number)
    if not svg_pages or [_page_number(path) for path in svg_pages] != [
        _page_number(path) for path in png_pages
    ]:
        raise RuntimeError(f"incomplete SVG/PNG page render for {source}")
    return list(zip(svg_pages, png_pages))


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def _tile_page(
    *,
    split: str,
    source_key: str,
    svg_path: Path,
    png_path: Path,
    output_dir: Path,
    categories: dict[str, int],
    tile_size: int,
    overlap: int,
    minimum_fraction: float,
    negative_ratio: float,
    long_span_minimum_fraction: float | None = None,
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str], int]:
    from PIL import Image

    view_box, objects, renderer, excluded_page_objects = semantic_svg_objects(svg_path)
    view_width = view_box[2] - view_box[0]
    view_height = view_box[3] - view_box[1]
    with Image.open(png_path) as image:
        width, height = image.size
    if abs(width - view_width) > 1.0 or abs(height - view_height) > 1.0:
        raise ValueError(
            f"PNG/SVG geometry mismatch for {svg_path}: "
            f"PNG {width}x{height}, SVG {view_width}x{view_height}"
        )
    tiles = [
        (left, top, min(left + tile_size, width), min(top + tile_size, height))
        for top in grid_starts(height, tile_size, overlap)
        for left in grid_starts(width, tile_size, overlap)
    ]
    if long_span_minimum_fraction is None:
        long_span_minimum_fraction = minimum_fraction
    if not 0 < long_span_minimum_fraction <= minimum_fraction <= 1:
        raise ValueError(
            "long-span minimum fraction must be in "
            "(0, minimum object fraction]"
        )
    page_id = hashlib.sha256(
        f"{source_key}\0{svg_path.name}".encode("utf-8")
    ).hexdigest()[:20]
    assigned: dict[int, list[dict[str, Any]]] = defaultdict(list)
    dropped: Counter[str] = Counter()
    for obj in objects:
        box = tuple(float(value) for value in obj["box_xyxy"])
        category = str(obj["category"])
        is_long_span = (
            category in LONG_SPAN_SEMANTIC_CATEGORIES
            or category.casefold().endswith("text")
        )
        object_minimum_fraction = (
            long_span_minimum_fraction
            if is_long_span
            else minimum_fraction
        )
        is_oversized = is_long_span and any(
            object_span > tile_span + 1e-9
            for object_span, tile_span in (
                (box[2] - box[0], float(tile_size)),
                (box[3] - box[1], float(tile_size)),
            )
        )
        if is_oversized:
            tile_indices = [
                index
                for index, tile in enumerate(tiles)
                if target_fragment_is_visible(
                    box,
                    tile,
                    minimum_fraction=minimum_fraction,
                    long_span_minimum_fraction=long_span_minimum_fraction,
                    is_long_span=True,
                    tile_overlap=overlap,
                )
            ]
        else:
            tile_index = choose_tile(box, tiles, object_minimum_fraction)
            tile_indices = [] if tile_index is None else [tile_index]
        if not tile_indices:
            dropped[category] += 1
            continue
        source_object_id = hashlib.sha256(
            (
                f"{page_id}\0{category}\0"
                + ",".join(f"{value:.4f}" for value in box)
            ).encode("utf-8")
        ).hexdigest()[:24]
        for tile_index in tile_indices:
            tile = tiles[tile_index]
            intersection = intersection_box(box, tile)
            clipped = [
                intersection[0] - tile[0],
                intersection[1] - tile[1],
                intersection[2] - tile[0],
                intersection[3] - tile[1],
            ]
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                raise ValueError(
                    "selected semantic object has no visible tile intersection: "
                    f"box={box}, tile={tile}"
                )
            assigned[tile_index].append(
                {
                    "box_xyxy": [round(value, 4) for value in clipped],
                    "page_box_xyxy": [round(value, 4) for value in box],
                    "category_id": category,
                    "label": categories[category],
                    "svg_class": obj["svg_class"],
                    "source_object_id": source_object_id,
                    "target_geometry_provenance": COMPLETE_PAGE_TARGET_PROVENANCE,
                }
            )
    rows = []
    relative_image = png_path.relative_to(output_dir).as_posix()
    negative_tiles = 0
    for tile_index, tile in enumerate(tiles):
        tile_objects = assigned.get(tile_index, [])
        if not tile_objects:
            if _stable_unit(
                f"negative\0{split}\0{page_id}\0{tile_index}"
            ) >= negative_ratio:
                continue
            negative_tiles += 1
        rows.append(
            {
                "split": split,
                "source_key": source_key,
                "image": relative_image,
                "image_id": page_id,
                "crop_xyxy": list(tile),
                "renderer": renderer,
                "objects": tile_objects,
            }
        )
    return rows, dropped, excluded_page_objects, negative_tiles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--musescore-exe", type=Path, required=True)
    parser.add_argument(
        "--source-list",
        type=Path,
        help="UTF-8 file of corpus-relative .mscx paths to render",
    )
    parser.add_argument("--max-scores", type=int)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--minimum-object-fraction", type=float, default=0.8)
    parser.add_argument(
        "--long-span-minimum-object-fraction",
        type=float,
        default=0.25,
        help=(
            "minimum visible fraction for page-spanning marks and text; "
            "complete page geometry is retained before tile clipping"
        ),
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=0.08,
        help="deterministic fraction of empty tiles retained as false-positive controls",
    )
    parser.add_argument("--calibration-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--render-timeout-seconds", type=int, default=900)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse complete per-score SVG/PNG pairs in an unfinished output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.corpus_dir.is_dir():
        raise FileNotFoundError(args.corpus_dir)
    if not args.musescore_exe.is_file():
        raise FileNotFoundError(args.musescore_exe)
    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
        and not args.resume
    ):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    if args.resume and (args.output_dir / "prepare-report.json").exists():
        raise FileExistsError(
            f"refusing to resume an already completed dataset: {args.output_dir}"
        )
    if not 0 < args.minimum_object_fraction <= 1:
        raise ValueError("minimum object fraction must be in (0, 1]")
    if not (
        0
        < args.long_span_minimum_object_fraction
        <= args.minimum_object_fraction
    ):
        raise ValueError(
            "long-span minimum object fraction must be in "
            "(0, minimum object fraction]"
        )
    if not 0 <= args.negative_ratio <= 1:
        raise ValueError("negative ratio must be in [0, 1]")
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
        corpus_root = args.corpus_dir.resolve()
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
        # A stable hash-ranked subset prevents path-order bias toward one
        # composer or collection directory.
        sources = sorted(
            sources,
            key=lambda path: hashlib.sha256(
                path.relative_to(args.corpus_dir).as_posix().encode("utf-8")
            ).hexdigest(),
        )[: args.max_scores]
        sources.sort()
    if not sources:
        raise ValueError("no MuseScore sources found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    category_names = sorted(set(SVG_CLASS_TO_CATEGORY.values()))
    categories = {name: index + 1 for index, name in enumerate(category_names)}
    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    dropped_counts: Counter[str] = Counter()
    excluded_page_counts: Counter[str] = Counter()
    negative_tiles_by_split: Counter[str] = Counter()
    source_manifest: list[dict[str, Any]] = []
    split_sources: dict[str, set[str]] = defaultdict(set)

    for source in sources:
        source_key = source.relative_to(args.corpus_dir).as_posix()
        split = split_for_source(
            source_key,
            calibration_fraction=args.calibration_fraction,
            test_fraction=args.test_fraction,
        )
        split_sources[split].add(source_key)
        piece_id = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:20]
        pages = _render_score(
            source,
            musescore_exe=args.musescore_exe,
            piece_dir=args.output_dir / "pages" / piece_id,
            timeout_seconds=args.render_timeout_seconds,
        )
        page_positive_rows = 0
        page_negative_rows = 0
        for svg_path, png_path in pages:
            rows, dropped, excluded_page_objects, negative_tiles = _tile_page(
                split=split,
                source_key=source_key,
                svg_path=svg_path,
                png_path=png_path,
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
            rows_by_split[split].extend(rows)
            page_positive_rows += sum(bool(row["objects"]) for row in rows)
            page_negative_rows += negative_tiles
            negative_tiles_by_split[split] += negative_tiles
            dropped_counts.update(dropped)
            excluded_page_counts.update(excluded_page_objects)
            for row in rows:
                counts.update(obj["category_id"] for obj in row["objects"])
        source_manifest.append(
            {
                "source_key": source_key,
                "source_sha256": sha256_file(source),
                "split": split,
                "pages": len(pages),
                "positive_tiles": page_positive_rows,
                "negative_tiles": page_negative_rows,
            }
        )

    all_splits = ("train", "calibration", "test")
    for split in all_splits:
        _write_jsonl(args.output_dir / f"{split}.jsonl", rows_by_split[split])
    intersections = {
        f"{left}_{right}": sorted(split_sources[left] & split_sources[right])
        for index, left in enumerate(all_splits)
        for right in all_splits[index + 1 :]
    }
    if any(intersections.values()):
        raise RuntimeError(f"source leakage detected: {intersections}")
    report = {
        "schema_version": 1,
        "license": "CC0-1.0",
        "role": "training_only_synthetic_semantic_geometry",
        "purpose": "synthetic semantic geometry; not real-scan validation",
        "corpus_dir": str(args.corpus_dir.resolve()),
        "musescore_exe": str(args.musescore_exe.resolve()),
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "minimum_object_fraction": args.minimum_object_fraction,
        "long_span_minimum_object_fraction": (
            args.long_span_minimum_object_fraction
        ),
        "target_geometry_provenance": COMPLETE_PAGE_TARGET_PROVENANCE,
        "target_assignment_version": (
            COMPLETE_PAGE_TARGET_ASSIGNMENT_VERSION
        ),
        "builder_source_sha256": sha256_file(Path(__file__)),
        "oversized_fragment_visibility_version": (
            OVERSIZED_FRAGMENT_VISIBILITY_VERSION
        ),
        "negative_ratio": args.negative_ratio,
        "categories": categories,
        "sources": source_manifest,
        "source_count_by_split": {
            split: len(split_sources[split]) for split in all_splits
        },
        "tiles_by_split": {
            split: len(rows_by_split[split]) for split in all_splits
        },
        "negative_tiles_by_split": {
            split: negative_tiles_by_split[split] for split in all_splits
        },
        "object_counts": dict(sorted(counts.items())),
        "dropped_object_counts": dict(sorted(dropped_counts.items())),
        "excluded_page_object_counts": dict(sorted(excluded_page_counts.items())),
        "split_intersections": intersections,
    }
    categories_path = args.output_dir / "categories.json"
    categories_path.write_text(
        json.dumps(
            {
                "format": 1,
                "classes": [
                    {
                        "label": label,
                        "name": name,
                        "source": "MuseScore SVG semantic class",
                    }
                    for name, label in sorted(
                        categories.items(), key=lambda item: item[1]
                    )
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": 1,
                "name": "scorescan-openscore-semantic-svg-regions",
                "license": "CC0-1.0",
                "role": "training_only_synthetic_semantic_geometry",
                "classes": len(categories),
                "tile_size": args.tile_size,
                "overlap": args.overlap,
                "target_assignment_version": (
                    COMPLETE_PAGE_TARGET_ASSIGNMENT_VERSION
                ),
                "builder_source_sha256": sha256_file(Path(__file__)),
                "oversized_fragment_visibility_version": (
                    OVERSIZED_FRAGMENT_VISIBILITY_VERSION
                ),
                "source_split_overlap": 0,
                "train": {
                    "tiles": len(rows_by_split["train"]),
                    "sources": len(split_sources["train"]),
                    "negative_tiles": negative_tiles_by_split["train"],
                },
                "calibration": {
                    "tiles": len(rows_by_split["calibration"]),
                    "sources": len(split_sources["calibration"]),
                    "negative_tiles": negative_tiles_by_split["calibration"],
                },
                "test": {
                    "tiles": len(rows_by_split["test"]),
                    "sources": len(split_sources["test"]),
                    "negative_tiles": negative_tiles_by_split["test"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = args.output_dir / "prepare-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "dataset.sha256").write_text(
        "\n".join(
            [
                f"{sha256_file(categories_path)}  categories.json",
                f"{sha256_file(manifest_path)}  manifest.json",
                f"{sha256_file(report_path)}  prepare-report.json",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(report["tiles_by_split"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
