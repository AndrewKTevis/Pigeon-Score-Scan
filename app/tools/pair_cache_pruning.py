"""Safely prune reproducible pair caches outside a pinned selection."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Iterable


def prune_unselected_pair_caches(
    output_dir: Path,
    selected_pair_ids: Iterable[int],
) -> dict[str, Any]:
    """Remove exact ``pair-N`` caches left by an older pinned selection.

    Unrelated entries remain untouched. Linked cache areas, linked candidates,
    linked descendants and paths escaping ``output_dir`` fail closed.
    """

    selected = {int(pair_id) for pair_id in selected_pair_ids}
    if any(pair_id < 0 for pair_id in selected):
        raise ValueError("selected pair ids must be non-negative")
    root = output_dir.resolve()
    areas = {
        "acceptances": ".json",
        "rejections": ".json",
        "pages": "",
        "reference_pages": "",
    }
    removed_files = 0
    removed_directories = 0
    removed_bytes = 0
    removed_by_area: dict[str, dict[str, int]] = {}

    def is_link_or_junction(path: Path) -> bool:
        is_junction = getattr(path, "is_junction", None)
        return path.is_symlink() or bool(
            callable(is_junction) and is_junction()
        )

    def pair_id_for_name(name: str, suffix: str) -> int | None:
        if suffix:
            if not name.endswith(suffix):
                return None
            name = name[: -len(suffix)]
        if not name.startswith("pair-"):
            return None
        raw_id = name[len("pair-") :]
        return int(raw_id) if raw_id.isdigit() else None

    def cache_bytes(path: Path) -> int:
        if path.is_file():
            return int(path.stat().st_size)
        total = 0
        for child in path.rglob("*"):
            if is_link_or_junction(child):
                raise RuntimeError(
                    f"refusing linked stale pair-cache entry: {child}"
                )
            if child.is_file():
                total += int(child.stat().st_size)
        return total

    for area_name, suffix in areas.items():
        area = output_dir / area_name
        if not area.exists():
            continue
        if is_link_or_junction(area) or not area.is_dir():
            raise RuntimeError(f"invalid pair-cache area: {area}")
        area_removed_files = 0
        area_removed_directories = 0
        area_removed_bytes = 0
        for candidate in sorted(area.iterdir(), key=lambda path: path.name):
            pair_id = pair_id_for_name(candidate.name, suffix)
            if pair_id is None or pair_id in selected:
                continue
            if is_link_or_junction(candidate):
                raise RuntimeError(
                    f"refusing linked stale pair cache: {candidate}"
                )
            resolved = candidate.resolve()
            if not resolved.is_relative_to(root):
                raise RuntimeError("stale pair cache escaped output directory")
            candidate_bytes = cache_bytes(candidate)
            if candidate.is_dir():
                shutil.rmtree(candidate)
                removed_directories += 1
                area_removed_directories += 1
            elif candidate.is_file():
                candidate.unlink()
                removed_files += 1
                area_removed_files += 1
            else:
                raise RuntimeError(
                    f"unsupported stale pair-cache entry: {candidate}"
                )
            removed_bytes += candidate_bytes
            area_removed_bytes += candidate_bytes
        removed_by_area[area_name] = {
            "files": area_removed_files,
            "directories": area_removed_directories,
            "bytes": area_removed_bytes,
        }
    return {
        "contract": "scorescan-selected-pair-cache-prune@1",
        "selected_pairs": len(selected),
        "removed_files": removed_files,
        "removed_directories": removed_directories,
        "removed_bytes": removed_bytes,
        "by_area": removed_by_area,
    }
