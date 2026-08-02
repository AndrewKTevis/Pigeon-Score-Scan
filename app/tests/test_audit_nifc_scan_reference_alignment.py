from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw

from app.tools.audit_nifc_scan_reference_alignment import (
    _maximum_shifted_correlation,
    estimated_staff_count,
    reference_page_measure_ranges,
)


def test_reference_measure_ranges_honor_original_page_breaks() -> None:
    assert reference_page_measure_ranges(
        [
            "**kern",
            "=1",
            "4c",
            "=2",
            "4d",
            "!!LO:PB:g=original",
            "=3",
            "4e",
            "!!LO:PB:g=original",
            "==",
            "*-",
        ]
    ) == [
        {
            "page_index": 1,
            "first_measure": 1,
            "last_measure": 2,
            "measure_count": 2,
        },
        {
            "page_index": 2,
            "first_measure": 3,
            "last_measure": 3,
            "measure_count": 1,
        },
    ]


def test_shifted_correlation_recovers_small_page_offset() -> None:
    first = np.zeros(200, dtype=np.float32)
    first[30:40] = 1
    first[100:115] = 1
    second = np.zeros(200, dtype=np.float32)
    second[37:47] = 1
    second[107:122] = 1
    assert _maximum_shifted_correlation(
        first,
        second,
        maximum_shift=10,
    ) > 0.95


def test_staff_estimator_counts_two_five_line_staves() -> None:
    image = Image.new("L", (900, 1200), 255)
    draw = ImageDraw.Draw(image)
    for base in (360, 470):
        for offset in range(0, 50, 10):
            draw.line((100, base + offset, 800, base + offset), fill=0, width=2)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    array = np.asarray(Image.open(io.BytesIO(buffer.getvalue())))
    assert estimated_staff_count(array) == 2
