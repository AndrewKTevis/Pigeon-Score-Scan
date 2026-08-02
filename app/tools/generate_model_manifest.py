from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.model_registry import build_manifest  # noqa: E402
from scorescan.util import atomic_write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resources",
        type=Path,
        default=ROOT / "src" / "scorescan" / "resources",
    )
    args = parser.parse_args()
    manifest = build_manifest(args.resources)
    atomic_write_json(args.resources / "model_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
