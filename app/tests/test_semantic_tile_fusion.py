from __future__ import annotations

from scorescan.semantic_tile_fusion import (
    TileFragmentDetection,
    fuse_tile_fragments,
)


def _fragment(
    *,
    bbox: tuple[int, int, int, int],
    tile_id: int,
    tile_bbox: tuple[int, int, int, int],
    staff_index: int,
) -> TileFragmentDetection:
    return TileFragmentDetection(
        class_name="bracket",
        label=4,
        bbox=bbox,
        confidence=0.999,
        staff_index=staff_index,
        placement="above",
        tile_id=tile_id,
        tile_bbox=tile_bbox,
    )


def test_page_spanning_bracket_fragments_fuse_across_staff_owners() -> None:
    fused = fuse_tile_fragments(
        [
            _fragment(
                bbox=(100, 100, 112, 1024),
                tile_id=0,
                tile_bbox=(0, 0, 1024, 1024),
                staff_index=1,
            ),
            _fragment(
                bbox=(100, 768, 112, 1792),
                tile_id=1,
                tile_bbox=(0, 768, 1024, 1792),
                staff_index=5,
            ),
        ]
    )

    assert len(fused) == 1
    assert fused[0].bbox == (100, 100, 112, 1792)
    assert fused[0].source_tile_count == 2
