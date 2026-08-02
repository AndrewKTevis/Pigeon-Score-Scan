#!/usr/bin/env python3
"""Prove synthetic detector replay is work-disjoint from the real holdout."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from typing import Any

from app.tools.build_muse_omr_work_catalog import (
    mscx_payload_fingerprint,
    work_fingerprint,
)
from app.tools.prepare_openscore_svg_regions import sha256_file


def _fingerprint_source(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.casefold()
    if suffix == ".mscz":
        fingerprint = work_fingerprint(path)
    elif suffix == ".mscx":
        fingerprint = mscx_payload_fingerprint(path.read_bytes())
    else:
        raise ValueError(f"unsupported replay score: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "work_fingerprint": fingerprint,
    }


def _score_files(root: Path) -> list[Path]:
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(root)
    # Region preparation consumes editable MSCX whenever both representations
    # are present. Isolation must fingerprint that exact representation rather
    # than silently preferring a potentially stale sibling MSCZ archive.
    mscx = sorted(path for path in root.rglob("*.mscx") if path.is_file())
    if mscx:
        return mscx
    return sorted(path for path in root.rglob("*.mscz") if path.is_file())


def audit(
    *,
    project_root: Path,
    holdout_selection: Path,
    replay_prepare_report: Path,
    replay_roots: list[Path],
    workers: int,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    project_root = project_root.resolve()
    selection = json.loads(holdout_selection.read_text(encoding="utf-8"))
    selected_works = selection.get("selected_work_fingerprints")
    if not isinstance(selected_works, list) or not selected_works:
        raise ValueError("holdout selection has no work fingerprints")
    holdout_works = {str(value) for value in selected_works}
    if (
        len(holdout_works) != len(selected_works)
        or any(len(value) != 64 for value in holdout_works)
    ):
        raise ValueError("holdout work fingerprints are malformed")
    replay_report = json.loads(
        replay_prepare_report.read_text(encoding="utf-8")
    )
    if (
        replay_report.get("purpose")
        != "combined synthetic semantic geometry; not real-scan validation"
        or replay_report.get("split_intersections")
        != {"calibration_test": [], "train_calibration": [], "train_test": []}
    ):
        raise ValueError("replay preparation isolation contract failed")

    root_reports = []
    replay_works: set[str] = set()
    for raw_root in replay_roots:
        root = raw_root.resolve()
        try:
            root.relative_to(project_root)
        except ValueError as exc:
            raise ValueError(f"replay root is outside project: {root}") from exc
        files = _score_files(root)
        if not files:
            raise ValueError(f"replay root contains no scores: {root}")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_fingerprint_source, files))
        root_works = {str(row["work_fingerprint"]) for row in rows}
        replay_works.update(root_works)
        root_reports.append(
            {
                "root": str(root),
                "score_files": len(rows),
                "works": len(root_works),
                "files": rows,
            }
        )
    overlap = sorted(replay_works & holdout_works)
    if overlap:
        raise ValueError(
            "synthetic replay overlaps independent holdout works: "
            + ", ".join(overlap[:10])
        )
    return {
        "schema_version": 1,
        "role": "training_holdout_work_isolation_evidence",
        "fingerprint_version": "mscx-c14n-without-eid-v1",
        "score_source_priority": "mscx-before-mscz",
        "tool_source_sha256": sha256_file(Path(__file__)),
        "holdout_selection": str(holdout_selection.resolve()),
        "holdout_selection_sha256": sha256_file(holdout_selection),
        "holdout_selected_works": len(holdout_works),
        "replay_prepare_report": str(replay_prepare_report.resolve()),
        "replay_prepare_report_sha256": sha256_file(replay_prepare_report),
        "replay_roots": root_reports,
        "replay_works": len(replay_works),
        "work_overlap": overlap,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--holdout-selection", type=Path, required=True)
    parser.add_argument("--replay-prepare-report", type=Path, required=True)
    parser.add_argument(
        "--replay-root",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit(
        project_root=args.project_root,
        holdout_selection=args.holdout_selection,
        replay_prepare_report=args.replay_prepare_report,
        replay_roots=args.replay_root,
        workers=args.workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    # Windows training queues redirect stdout and may otherwise inherit a GBK
    # console encoding.  The report file retains Unicode; stdout is an
    # ASCII-safe progress record and must never abort a completed audit.
    print(json.dumps(report, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
