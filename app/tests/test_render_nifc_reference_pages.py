from __future__ import annotations

from lxml import etree
import pytest

from app.tools.render_nifc_reference_pages import (
    svg_measure_count,
    svg_pixel_size,
)


def test_svg_contract_extracts_size_and_measure_groups() -> None:
    root = etree.fromstring(
        b"""<svg xmlns="http://www.w3.org/2000/svg"
        width="1050px" height="1485px">
        <g class="page-margin"><g class="measure" id="m1"/></g>
        </svg>"""
    )
    assert svg_pixel_size(root) == (1050, 1485)
    assert svg_measure_count(root) == 1


def test_svg_contract_rejects_non_pixel_dimensions() -> None:
    root = etree.fromstring(
        b'<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="50"/>'
    )
    with pytest.raises(ValueError, match="pixel width"):
        svg_pixel_size(root)
