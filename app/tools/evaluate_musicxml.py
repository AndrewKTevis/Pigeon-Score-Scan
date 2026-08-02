from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.evaluation import compare_musicxml  # noqa: E402


def compare(reference: Path, candidate: Path) -> dict[str, object]:
    return compare_musicxml(reference, candidate)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Globally align and compare complete single-part or full-score MusicXML files."
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path, help="Also write the report to this JSON file")
    args = parser.parse_args()
    report = compare(args.reference, args.candidate)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
