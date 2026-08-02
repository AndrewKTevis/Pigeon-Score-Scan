#!/usr/bin/env python3
"""Audit rare MuseScore tag evidence before expensive scan registration."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from app.tools.acquire_muse_omr_benchmark import sha256_file

AUDIT_VERSION = "muse-semantic-tag-evidence@2"


def parse_required_tags(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        name, separator, raw_floor = str(value).partition("=")
        name = name.strip()
        if (
            not separator
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", name)
        ):
            raise ValueError(f"invalid required MuseScore tag: {value!r}")
        try:
            floor = int(raw_floor)
        except ValueError as exc:
            raise ValueError(
                f"invalid required MuseScore tag floor: {value!r}"
            ) from exc
        if floor <= 0 or name in result:
            raise ValueError(f"invalid required MuseScore tag: {value!r}")
        result[name] = floor
    if not result:
        raise ValueError("at least one required MuseScore tag is needed")
    return dict(sorted(result.items()))


def _selection(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pair_ids = payload.get("selected_pair_ids")
    works = payload.get("selected_work_fingerprints")
    rows = payload.get("pair_work_fingerprints")
    if (
        not isinstance(pair_ids, list)
        or not isinstance(works, list)
        or not isinstance(rows, list)
        or len(pair_ids) != len(rows)
        or int(payload.get("selected_pair_count", -1)) != len(pair_ids)
        or int(payload.get("selected_work_count", -1)) != len(works)
    ):
        raise ValueError(f"invalid Muse OMR selection: {path}")
    mapping = {
        int(row["pair_id"]): str(row["work_fingerprint"])
        for row in rows
        if isinstance(row, dict)
    }
    if (
        len(mapping) != len(pair_ids)
        or set(mapping) != {int(value) for value in pair_ids}
        or set(mapping.values()) != {str(value) for value in works}
    ):
        raise ValueError(f"inconsistent Muse OMR work mapping: {path}")
    return payload


def _mscx_payload(path: Path) -> str:
    with ZipFile(path) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".mscx")
            and not name.startswith("META-INF/")
        ]
        if not names:
            raise ValueError(f"MuseScore archive has no score payload: {path}")
        name = max(names, key=lambda item: archive.getinfo(item).file_size)
        return archive.read(name).decode("utf-8", errors="strict")


def count_opening_tags(payload: str, tag_names: tuple[str, ...]) -> dict[str, int]:
    return {
        name: len(
            re.findall(
                rf"<{re.escape(name)}(?:\s[^<>]*?)?\s*/?>",
                payload,
            )
        )
        for name in tag_names
    }


def audit(
    *,
    dataset_dir: Path,
    required_tags: dict[str, int],
    forbidden_selection: Path | None = None,
) -> dict[str, Any]:
    selection_path = dataset_dir / "selection.json"
    selection = _selection(selection_path)
    pair_rows = {
        int(row["pair_id"]): str(row["work_fingerprint"])
        for row in selection["pair_work_fingerprints"]
    }
    pair_ids = sorted(pair_rows)

    pair_overlap: list[int] = []
    work_overlap: list[str] = []
    forbidden_selection_sha256: str | None = None
    if forbidden_selection is not None:
        forbidden = _selection(forbidden_selection)
        pair_overlap = sorted(
            set(pair_ids)
            & {int(value) for value in forbidden["selected_pair_ids"]}
        )
        work_overlap = sorted(
            set(pair_rows.values())
            & {
                str(value)
                for value in forbidden["selected_work_fingerprints"]
            }
        )
        forbidden_selection_sha256 = sha256_file(forbidden_selection)

    tag_names = tuple(sorted(required_tags))
    totals: Counter[str] = Counter()
    works_by_tag: dict[str, set[str]] = defaultdict(set)
    pairs_by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair_id in pair_ids:
        archive = dataset_dir / "mscz" / f"score_file_{pair_id}.mscz"
        if not archive.is_file():
            raise FileNotFoundError(archive)
        counts = count_opening_tags(_mscx_payload(archive), tag_names)
        for name, count in counts.items():
            if count <= 0:
                continue
            totals[name] += count
            works_by_tag[name].add(pair_rows[pair_id])
            pairs_by_tag[name].append(
                {
                    "pair_id": pair_id,
                    "work_fingerprint": pair_rows[pair_id],
                    "objects": count,
                }
            )

    failures = []
    if pair_overlap:
        failures.append(f"pair_overlap={pair_overlap}")
    if work_overlap:
        failures.append(f"work_overlap={work_overlap}")
    for name, floor in required_tags.items():
        if totals[name] < floor:
            failures.append(f"{name}={totals[name]}<{floor}")
    return {
        "schema_version": 1,
        "audit_version": AUDIT_VERSION,
        "name": "scorescan-muse-semantic-tag-evidence-audit-v1",
        "purpose": "pre-registration rare semantic source-evidence gate",
        "selection_sha256": sha256_file(selection_path),
        "forbidden_selection_sha256": forbidden_selection_sha256,
        "selected_pairs": len(pair_ids),
        "selected_works": len(set(pair_rows.values())),
        "pair_overlap": pair_overlap,
        "work_overlap": work_overlap,
        "required_tag_floors": required_tags,
        "tag_counts": {
            name: int(totals[name]) for name in tag_names
        },
        "independent_works_by_tag": {
            name: len(works_by_tag[name]) for name in tag_names
        },
        "pairs_by_tag": {
            name: pairs_by_tag[name] for name in tag_names
        },
        "passed": not failures,
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--required-tag",
        action="append",
        default=[],
        metavar="TAG=MINIMUM",
    )
    parser.add_argument("--forbidden-selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dataset_dir.is_dir():
        raise FileNotFoundError(args.dataset_dir)
    if args.output.exists():
        raise FileExistsError(args.output)
    required_tags = parse_required_tags(args.required_tag)
    result = audit(
        dataset_dir=args.dataset_dir,
        required_tags=required_tags,
        forbidden_selection=args.forbidden_selection,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps(result, sort_keys=True), flush=True)
    if not result["passed"]:
        raise RuntimeError(
            "Muse semantic tag evidence failed: "
            + "; ".join(result["failures"])
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
