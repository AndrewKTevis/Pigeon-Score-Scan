from __future__ import annotations

import pytest

from app.tools.catalog_muse_omr_boundary_supplement import (
    unused_work_representatives,
)


def test_unused_work_representatives_are_disjoint_and_deterministic() -> None:
    mapping = {
        9: "c" * 64,
        2: "a" * 64,
        7: "b" * 64,
        3: "a" * 64,
        8: "c" * 64,
    }

    result = unused_work_representatives(mapping, {"b" * 64})

    assert result == [(2, "a" * 64), (8, "c" * 64)]


def test_unused_work_representatives_reject_unknown_reserved_work() -> None:
    with pytest.raises(ValueError, match="outside the catalog"):
        unused_work_representatives({1: "a" * 64}, {"b" * 64})
