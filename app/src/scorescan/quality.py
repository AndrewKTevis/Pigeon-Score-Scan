from __future__ import annotations

from pathlib import Path

from .imaging import preprocess_page
from .models import PageInfo


def inspect_page(page: PageInfo, output_dir: Path | None = None) -> PageInfo:
    target = output_dir or (Path(page.image_path).parent.parent / "normalized")
    return preprocess_page(page, target)
