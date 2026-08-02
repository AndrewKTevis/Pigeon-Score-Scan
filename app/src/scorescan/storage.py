from __future__ import annotations

"""Bounded workspace and free-space checks for local production operation."""

import os
import shutil
from pathlib import Path

from .config import Settings


class StorageCapacityError(ValueError):
    """Raised before writing when workspace or disk capacity is insufficient."""


def directory_size_bounded(root: Path, stop_after: int) -> int:
    """Return directory bytes, stopping as soon as ``stop_after`` is exceeded."""

    total = 0
    if not root.exists():
        return 0
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += int(entry.stat(follow_symlinks=False).st_size)
                            if total > stop_after:
                                return total
                    except FileNotFoundError:
                        # Concurrent retention cleanup may remove a completed entry.
                        continue
                    except OSError as exc:
                        raise StorageCapacityError(
                            f"无法审计工作区条目 {entry.path}：{exc}"
                        ) from exc
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise StorageCapacityError(
                f"无法审计工作区目录 {directory}：{exc}"
            ) from exc
    return total


def require_free_space(
    path: Path,
    *,
    required_bytes: int,
    reserve_bytes: int,
    context: str,
) -> None:
    """Require temporary headroom plus a non-consumable safety reserve."""

    required = max(0, int(required_bytes))
    reserve = max(0, int(reserve_bytes))
    path.mkdir(parents=True, exist_ok=True)
    free = int(shutil.disk_usage(path).free)
    if free < required + reserve:
        raise StorageCapacityError(
            f"{context} 需要至少 {(required + reserve) / (1024 ** 3):.2f} GiB 可用空间，"
            f"当前仅有 {free / (1024 ** 3):.2f} GiB。"
        )


def require_workspace_capacity(settings: Settings, *, additional_bytes: int, context: str) -> None:
    """Enforce both disk reserve and the configured persistent workspace quota."""

    additional = max(0, int(additional_bytes))
    require_free_space(
        settings.workspace,
        required_bytes=additional,
        reserve_bytes=settings.minimum_free_space_bytes,
        context=context,
    )
    current = directory_size_bounded(settings.workspace, settings.max_workspace_bytes + 1)
    if current + additional > settings.max_workspace_bytes:
        raise StorageCapacityError(
            f"{context} 会使工作区超过 {settings.max_workspace_bytes / (1024 ** 3):.1f} GiB 上限；"
            "请删除旧任务或调低保留天数。"
        )
