#!/usr/bin/env python3
"""Prune local Muse OMR pair files not named by a verified selection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ALLOWED_ROLES = {
    "external_scan_degraded_training_only",
    "external_scan_degraded_development_benchmark_not_training",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def unselected_files(dataset_dir: Path) -> list[Path]:
    dataset_dir = dataset_dir.resolve(strict=True)
    selection = _load_json(dataset_dir / "selection.json")
    provenance = _load_json(dataset_dir / "provenance.json")
    selected = selection.get("selected_pair_ids")
    if (
        selection.get("role") not in ALLOWED_ROLES
        or provenance.get("role") != selection.get("role")
        or not isinstance(selected, list)
        or not selected
        or selected != provenance.get("selected_pair_ids")
        or int(selection.get("selected_pair_count", -1)) != len(selected)
        or int(provenance.get("selected_pair_count", -1)) != len(selected)
    ):
        raise ValueError("Muse OMR selection/provenance contract failed")
    selected_ids = {int(value) for value in selected}
    if len(selected_ids) != len(selected) or min(selected_ids) < 0:
        raise ValueError("Muse OMR selection contains invalid pair ids")

    expected = {
        dataset_dir / "mscz" / f"score_file_{pair_id}.mscz"
        for pair_id in selected_ids
    } | {
        dataset_dir / "pdf" / f"score_file_{pair_id}.pdf"
        for pair_id in selected_ids
    }
    if not all(path.is_file() and not path.is_symlink() for path in expected):
        raise ValueError("selected Muse OMR pair files are incomplete")

    candidates: set[Path] = set()
    for directory_name, suffix in (("mscz", ".mscz"), ("pdf", ".pdf")):
        directory = (dataset_dir / directory_name).resolve(strict=True)
        if directory.parent != dataset_dir or directory.is_symlink():
            raise ValueError("unsafe Muse OMR pair directory")
        for path in directory.iterdir():
            if path.is_symlink():
                raise ValueError(f"Muse OMR pair directory contains a symlink: {path}")
            if path.is_file() and path.suffix.lower() == suffix:
                candidates.add(path.resolve(strict=True))
    return sorted(candidates - expected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = unselected_files(args.dataset_dir)
    total_bytes = sum(path.stat().st_size for path in paths)
    report = {
        "format": 1,
        "name": "scorescan-muse-omr-unselected-file-prune-v1",
        "executed": bool(args.execute),
        "removed_files": len(paths),
        "removed_bytes": total_bytes,
    }
    if args.execute:
        for path in paths:
            path.unlink()
        report_path = args.dataset_dir.resolve() / "prune-unselected-report.json"
        temporary = report_path.with_suffix(report_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, report_path)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
