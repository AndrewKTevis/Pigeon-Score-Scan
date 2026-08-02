from __future__ import annotations

from app.tools.semantic_target_visibility import target_fragment_is_visible


def test_oversized_long_target_requires_overlap_width_not_area_floor() -> None:
    box = (10.0, 100.0, 20.0, 5100.0)

    assert target_fragment_is_visible(
        box,
        (0.0, 0.0, 1024.0, 1024.0),
        minimum_fraction=0.8,
        long_span_minimum_fraction=0.25,
        is_long_span=True,
        tile_overlap=256,
    )
    assert not target_fragment_is_visible(
        box,
        (0.0, 0.0, 1024.0, 200.0),
        minimum_fraction=0.8,
        long_span_minimum_fraction=0.25,
        is_long_span=True,
        tile_overlap=256,
    )


def test_oversized_fragment_exception_never_applies_to_compact_object() -> None:
    assert not target_fragment_is_visible(
        (900.0, 100.0, 1100.0, 200.0),
        (0.0, 0.0, 1024.0, 1024.0),
        minimum_fraction=0.8,
        long_span_minimum_fraction=0.25,
        is_long_span=False,
        tile_overlap=256,
    )
