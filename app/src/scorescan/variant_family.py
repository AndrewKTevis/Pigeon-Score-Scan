from __future__ import annotations

"""Correlation families for deterministic scan preprocessing variants.

A family represents variants that share enough image operations that their errors
must not be counted as independent evidence.  Keeping this mapping in one module
prevents consensus, count resolution and diagnostics from drifting apart.
"""

from collections.abc import Callable, Iterable
from typing import TypeVar


T = TypeVar("T")


def group_complete_families(
    items: Iterable[T],
    *,
    family_of: Callable[[T], str],
    valid_of: Callable[[T], bool],
) -> tuple[dict[str, list[T]], frozenset[str]]:
    """Group evidence by correlation family and fail closed on invalid siblings.

    A family is usable only when every observed sibling is valid.  Silently dropping an
    invalid sibling would make the remaining correlated candidate look like independent
    evidence.  Returning incomplete family names separately lets callers account for the
    abstention in diagnostics without exposing those candidates to voting or quality
    aggregation.
    """
    grouped: dict[str, list[T]] = {}
    for item in items:
        grouped.setdefault(family_of(item), []).append(item)
    incomplete = frozenset(
        family for family, members in grouped.items()
        if not members or any(not valid_of(member) for member in members)
    )
    complete = {
        family: members
        for family, members in grouped.items()
        if family not in incomplete
    }
    return complete, incomplete


def variant_family(variant: str) -> str:
    raw = str(variant or "").lower().strip()
    name = raw.split(":", 1)[0]
    if name == "primary":
        return "baseline"
    if name in {"flat", "deblock"}:
        return "restoration"
    if name in {"otsu", "adaptive"}:
        return "binary"
    if name in {"staffnorm", "upscale"}:
        return "scale"
    if name in {"system_localized", "localized"}:
        return "localization"
    if name == "measure_localized":
        # Sparse local candidates are scoped to exactly one measure.  Keep the target
        # in the family identity so unrelated rescue crops never become siblings or
        # increase one another's majority denominator.
        suffix = raw.split(":", 1)[1] if ":" in raw else "unknown"
        return f"measure_localization:{suffix}"
    return f"other:{name or 'unknown'}"
