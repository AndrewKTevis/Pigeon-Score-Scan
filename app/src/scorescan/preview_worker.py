from __future__ import annotations

import argparse
from pathlib import Path

from .preview import _render_preview
from .util import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("musicxml_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("result_path", type=Path)
    args = parser.parse_args()

    preview_path, warnings = _render_preview(
        args.musicxml_path,
        args.output_dir,
    )
    atomic_write_json(
        args.result_path,
        {
            "format": 1,
            "preview_path": (
                str(preview_path.resolve()) if preview_path is not None else None
            ),
            "warnings": warnings,
        },
    )


if __name__ == "__main__":
    main()
