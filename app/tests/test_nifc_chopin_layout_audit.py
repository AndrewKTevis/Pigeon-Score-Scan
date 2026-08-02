from __future__ import annotations

import numpy as np

from app.tools.audit_nifc_chopin_subwork_alignment import (
    _best_contiguous_window,
)
from app.tools.prepare_nifc_chopin_layout_audit_pages import (
    classify_scan_layout,
)


def test_landscape_spread_and_portrait_page_classification() -> None:
    assert classify_scan_layout(2120, 1473).startswith("left_to_right")
    assert classify_scan_layout(1419, 1734).startswith("single_page")


def test_best_window_requires_ordered_contiguous_reference_pages() -> None:
    matrix = np.asarray(
        [
            [0.01, 0.02],
            [0.80, 0.05],
            [0.04, 0.90],
            [0.70, 0.04],
        ],
        dtype=np.float64,
    )
    result = _best_contiguous_window(matrix)
    assert result["scan_page_indices"] == [2, 3]
    assert result["selected_mapping_is_bidirectional_best"] is True
    assert result["best_window_margin"] > 0


def test_best_window_does_not_claim_bidirectional_mapping_on_duplicate() -> None:
    matrix = np.asarray(
        [
            [0.90, 0.01],
            [0.80, 0.85],
            [0.01, 0.10],
        ],
        dtype=np.float64,
    )
    result = _best_contiguous_window(matrix)
    assert result["scan_page_indices"] == [1, 2]
    assert result["selected_mapping_is_bidirectional_best"] is True
    matrix[2, 1] = 0.95
    result = _best_contiguous_window(matrix)
    assert result["selected_mapping_is_bidirectional_best"] is False
