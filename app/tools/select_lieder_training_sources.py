#!/usr/bin/env python3
"""Select diverse, mark-rich OpenScore Lieder sources without benchmark leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCORE_ID_RE = re.compile(r"^(\d+):\s*$", re.MULTILINE)
FILENAME_ID_RE = re.compile(r"^lc(\d+)\.mscx$", re.IGNORECASE)
TAG_WEIGHTS = {
    "Articulation": 3,
    "Dynamic": 5,
    "Fermata": 5,
    "HairPin": 7,
    "Lyrics": 1,
    "Ottava": 6,
    "Pedal": 6,
    "RehearsalMark": 6,
    "Slur": 2,
    "StaffText": 5,
    "SystemText": 5,
    "Tempo": 5,
    "Tie": 2,
    "Trill": 6,
}
TAG_RE = re.compile(
    r"<(" + "|".join(re.escape(name) for name in TAG_WEIGHTS) + r")\b"
)


def score_ids_from_olimpic_yaml(path: Path) -> set[int]:
    ids = {int(value) for value in SCORE_ID_RE.findall(path.read_text(encoding="utf-8"))}
    if not ids:
        raise ValueError(f"no score ids found in {path}")
    return ids


def mark_richness(source_text: str) -> tuple[int, Counter[str]]:
    counts: Counter[str] = Counter(TAG_RE.findall(source_text))
    weighted = sum(TAG_WEIGHTS[name] * count for name, count in counts.items())
    return weighted, counts


def diverse_round_robin(
    rows: list[dict[str, Any]], maximum_scores: int
) -> list[dict[str, Any]]:
    if maximum_scores <= 0:
        raise ValueError("maximum_scores must be positive")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["composer"])].append(row)
    for composer_rows in grouped.values():
        composer_rows.sort(
            key=lambda row: (
                -int(row["mark_richness"]),
                hashlib.sha256(str(row["relative_path"]).encode()).hexdigest(),
            )
        )
    composer_order = sorted(
        grouped,
        key=lambda composer: hashlib.sha256(composer.encode()).hexdigest(),
    )
    selected = []
    depth = 0
    while len(selected) < min(maximum_scores, len(rows)):
        added = False
        for composer in composer_order:
            if depth < len(grouped[composer]):
                selected.append(grouped[composer][depth])
                added = True
                if len(selected) >= maximum_scores:
                    break
        if not added:
            break
        depth += 1
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--olimpic-splits-dir", type=Path, required=True)
    parser.add_argument("--output-list", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--maximum-scores", type=int, default=320)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.corpus_dir.is_dir():
        raise FileNotFoundError(args.corpus_dir)
    if not args.olimpic_splits_dir.is_dir():
        raise FileNotFoundError(args.olimpic_splits_dir)
    if args.output_list.exists() or args.output_report.exists():
        raise FileExistsError("output list/report already exists")
    split_ids = {
        split: score_ids_from_olimpic_yaml(
            args.olimpic_splits_dir / f"{split}_scores.yaml"
        )
        for split in ("train", "dev", "test")
    }
    if (
        split_ids["train"] & split_ids["dev"]
        or split_ids["train"] & split_ids["test"]
        or split_ids["dev"] & split_ids["test"]
    ):
        raise ValueError("OLiMPiC score split ids overlap")

    rows = []
    missing_train_ids = set(split_ids["train"])
    for source in sorted(args.corpus_dir.rglob("*.mscx")):
        match = FILENAME_ID_RE.fullmatch(source.name)
        if not match:
            continue
        score_id = int(match.group(1))
        if score_id not in split_ids["train"]:
            continue
        missing_train_ids.discard(score_id)
        relative = source.relative_to(args.corpus_dir)
        text = source.read_text(encoding="utf-8", errors="replace")
        richness, counts = mark_richness(text)
        path_parts = relative.parts
        composer = path_parts[1] if len(path_parts) > 1 and path_parts[0] == "scores" else path_parts[0]
        rows.append(
            {
                "score_id": score_id,
                "relative_path": relative.as_posix(),
                "composer": composer,
                "mark_richness": richness,
                "tag_counts": dict(counts),
                "bytes": source.stat().st_size,
            }
        )
    if not rows:
        raise ValueError("no OLiMPiC-train Lieder sources found")
    selected = diverse_round_robin(rows, args.maximum_scores)
    selected_ids = {int(row["score_id"]) for row in selected}
    leakage = {
        "dev": sorted(selected_ids & split_ids["dev"]),
        "test": sorted(selected_ids & split_ids["test"]),
    }
    if any(leakage.values()):
        raise RuntimeError(f"benchmark score leakage: {leakage}")

    args.output_list.parent.mkdir(parents=True, exist_ok=True)
    args.output_list.write_text(
        "".join(f"{row['relative_path']}\n" for row in selected),
        encoding="utf-8",
    )
    aggregate_tags: Counter[str] = Counter()
    for row in selected:
        aggregate_tags.update(row["tag_counts"])
    report = {
        "schema_version": 1,
        "selection": "composer-diverse round-robin, mark-rich within composer",
        "available_olimpic_train_sources": len(rows),
        "selected_sources": len(selected),
        "selected_composers": len({row["composer"] for row in selected}),
        "selected_bytes": sum(int(row["bytes"]) for row in selected),
        "aggregate_tag_counts": dict(sorted(aggregate_tags.items())),
        "olimpic_split_sizes": {
            split: len(values) for split, values in split_ids.items()
        },
        "missing_olimpic_train_source_ids": sorted(missing_train_ids),
        "benchmark_leakage": leakage,
        "sources": selected,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_sources": report["selected_sources"],
                "selected_composers": report["selected_composers"],
                "benchmark_leakage": leakage,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
