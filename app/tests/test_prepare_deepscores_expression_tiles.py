from __future__ import annotations

from app.tools.prepare_deepscores_expression_tiles import (
    choose_tile,
    grid_starts,
    parse_deepscores_bbox,
)
import pytest


def test_grid_covers_last_pixels_without_duplicate_last_tile() -> None:
    assert grid_starts(800, 1024, 256) == [0]
    assert grid_starts(2048, 1024, 256) == [0, 768, 1024]


def test_choose_tile_prefers_fullest_center_containing_tile() -> None:
    tiles = [(0, 0, 100, 100), (50, 0, 150, 100)]
    assert choose_tile((60, 10, 90, 40), tiles, 0.8) == 0
    assert choose_tile((90, 10, 140, 40), tiles, 0.8) == 1


def test_choose_tile_rejects_heavily_clipped_object() -> None:
    assert choose_tile((80, 10, 140, 40), [(0, 0, 100, 100)], 0.8) is None


def test_deepscores_bbox_is_xyxy_and_verified_against_oriented_box() -> None:
    # Real DeepScoresV2 train annotation 1113315, a crescendo hairpin.  Treating
    # a_bbox as yxyx moves this horizontal mark into an empty vertical strip.
    annotation = {
        "a_bbox": [596.0, 1565.0, 758.0, 1588.0],
        "o_bbox": [
            758.0,
            1565.0,
            596.0,
            1565.0,
            596.0,
            1588.0,
            758.0,
            1588.0,
        ],
    }
    assert parse_deepscores_bbox(annotation) == (596.0, 1565.0, 758.0, 1588.0)


def test_deepscores_bbox_rejects_axis_swapped_oriented_geometry() -> None:
    with pytest.raises(ValueError, match="disagree"):
        parse_deepscores_bbox(
            {
                "a_bbox": [1565.0, 596.0, 1588.0, 758.0],
                "o_bbox": [
                    758.0,
                    1565.0,
                    596.0,
                    1565.0,
                    596.0,
                    1588.0,
                    758.0,
                    1588.0,
                ],
            }
        )


def test_deepscores_bbox_allows_annotated_line_but_rejects_point() -> None:
    assert parse_deepscores_bbox(
        {
            "a_bbox": [373.0, 1746.0, 1867.0, 1746.0],
            "o_bbox": [
                373.0,
                1746.0,
                373.0,
                1746.0,
                1867.0,
                1746.0,
                1867.0,
                1746.0,
            ],
        }
    ) == (373.0, 1746.0, 1867.0, 1746.0)
    with pytest.raises(ValueError, match="usable"):
        parse_deepscores_bbox(
            {
                "a_bbox": [1836.0, 198.0, 1836.0, 198.0],
                "o_bbox": [1836.0, 198.0] * 4,
            }
        )
