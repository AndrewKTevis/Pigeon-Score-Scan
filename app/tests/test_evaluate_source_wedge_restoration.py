from __future__ import annotations

from pathlib import Path

from app.tools.evaluate_source_wedge_restoration import oracle_svg_wedges


def test_oracle_ignores_text_hairpin_and_groups_legacy_two_line_wedge(
    tmp_path: Path,
) -> None:
    svg = tmp_path / "page.svg"
    svg.write_text(
        """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 300">
  <path class="HairpinSegment" d="M10,20 L80,20" />
  <polyline class="HairpinSegment" fill="none" stroke="#000"
    stroke-dasharray="10,12" stroke-width="2"
    points="90,40 220,40" />
  <polyline class="HairpinSegment" fill="none" stroke="#000"
    stroke-width="2" points="180,90 100,100 180,110" />
  <polyline class="HairpinSegment" fill="none" stroke="#000"
    stroke-width="2" points="300,160 500,150" />
  <polyline class="HairpinSegment" fill="none" stroke="#000"
    stroke-width="2" points="300,170 500,180" />
</svg>
""",
        encoding="utf-8",
    )

    wedges = oracle_svg_wedges(svg)

    assert [(item.kind, item.bbox) for item in wedges] == [
        ("crescendo", (98, 88, 182, 112)),
        ("crescendo", (298, 148, 502, 182)),
    ]
