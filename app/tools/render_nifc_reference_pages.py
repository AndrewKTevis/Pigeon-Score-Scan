from __future__ import annotations

"""Isolated Verovio/Edge renderer for NIFC alignment diagnostics.

Verovio is native code and malformed or unsupported page content can terminate
the interpreter.  The alignment auditor invokes this module in a subprocess so
that one crashing page is recorded instead of aborting the whole audit.
"""

import argparse
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from lxml import etree
import verovio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.util import atomic_write_bytes


EDGE_CANDIDATES = (
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
)
PIXEL_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)px$")


def svg_pixel_size(root: etree._Element) -> tuple[int, int]:
    values: list[int] = []
    for name in ("width", "height"):
        match = PIXEL_PATTERN.fullmatch(str(root.get(name, "")))
        if match is None:
            raise ValueError(f"rendered SVG has no pixel {name}")
        values.append(max(1, int(round(float(match.group(1))))))
    return values[0], values[1]


def svg_measure_count(root: etree._Element) -> int:
    return len(
        root.xpath(
            ".//*[local-name()='g' and "
            "contains(concat(' ', normalize-space(@class), ' '), "
            "' measure ')]"
        )
    )


def _edge_path() -> Path:
    for path in EDGE_CANDIDATES:
        if path.is_file():
            return path
    raise FileNotFoundError("Microsoft Edge is required for SVG diagnostics")


def _rasterize_svg(svg_path: Path, png_path: Path) -> None:
    root = etree.fromstring(svg_path.read_bytes())
    width, height = svg_pixel_size(root)
    with tempfile.TemporaryDirectory(prefix="scorescan-edge-svg-") as profile:
        completed = subprocess.run(
            [
                str(_edge_path()),
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--force-device-scale-factor=1",
                "--run-all-compositor-stages-before-draw",
                "--virtual-time-budget=1000",
                f"--user-data-dir={profile}",
                f"--window-size={width},{height}",
                f"--screenshot={png_path}",
                svg_path.resolve().as_uri(),
            ],
            check=False,
            capture_output=True,
            timeout=45,
        )
    if completed.returncode != 0 or not png_path.is_file():
        raise RuntimeError(
            "Edge SVG rasterization failed: "
            + completed.stderr.decode("utf-8", errors="replace")[-1000:]
        )


def render(
    reference_path: Path,
    output_dir: Path,
    *,
    expected_pages: int,
    page: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    toolkit = verovio.toolkit()
    toolkit.setOptions(
        {
            "breaks": "encoded",
            "footer": "none",
            "header": "none",
            "scale": 50,
        }
    )
    if not toolkit.loadData(
        reference_path.read_text(encoding="utf-8-sig")
    ):
        raise ValueError(f"Verovio could not load {reference_path}")
    rendered_count = int(toolkit.getPageCount())
    if rendered_count not in {expected_pages, expected_pages + 1}:
        raise ValueError(
            f"unexpected rendered page count {rendered_count}: "
            f"{reference_path}"
        )
    selected = (
        [page]
        if page is not None
        else list(range(1, expected_pages + 1))
    )
    if any(
        value is None or value < 1 or value > expected_pages
        for value in selected
    ):
        raise ValueError("page is outside expected reference range")
    for page_index in selected:
        assert page_index is not None
        svg = toolkit.renderToSVG(page_index)
        root = etree.fromstring(svg.encode("utf-8"))
        if svg_measure_count(root) == 0:
            raise ValueError(
                f"rendered reference page {page_index} has no measures"
            )
        svg_path = output_dir / f"page-{page_index:03d}.svg"
        png_path = output_dir / f"page-{page_index:03d}.png"
        atomic_write_bytes(svg_path, svg.encode("utf-8"))
        _rasterize_svg(svg_path, png_path)
    if page is None and rendered_count == expected_pages + 1:
        trailing = toolkit.renderToSVG(rendered_count)
        trailing_root = etree.fromstring(trailing.encode("utf-8"))
        if svg_measure_count(trailing_root) != 0:
            raise ValueError("extra rendered page is not empty")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-pages", type=int, required=True)
    parser.add_argument("--page", type=int)
    args = parser.parse_args()
    render(
        args.reference_path.resolve(),
        args.output_dir.resolve(),
        expected_pages=args.expected_pages,
        page=args.page,
    )


if __name__ == "__main__":
    main()
