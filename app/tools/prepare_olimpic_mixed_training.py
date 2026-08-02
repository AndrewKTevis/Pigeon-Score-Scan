#!/usr/bin/env python3
"""Add deterministic synthetic replay to real-scan OLiMPiC fine-tuning data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

DEFAULT_SEED = "scorescan-olimpic-real-plus-replay-v2"
DEFAULT_FAMILY_PRIORITY_SEED = "scorescan-olimpic-real-plus-replay-v3"
SOURCE_DOCUMENT_SAFE_BASE = "scorescan-olimpic-real-v2-source-document-safe"
SOURCE_DOCUMENT_SAFE_FAMILY_PRIORITY_NAME = (
    "scorescan-olimpic-real-plus-synthetic-replay-v4-source-document-safe"
)
FAMILY_PRIORITY_WEIGHTS = {
    "tie": 8.0,
    "slur": 6.0,
    "ornament": 6.0,
    "articulation": 4.0,
    "beam": 1.0,
}
_ARTICULATION_TOKENS = frozenset(
    {
        "accent",
        "breath-mark",
        "caesura",
        "detached-legato",
        "doit",
        "falloff",
        "marcato",
        "plop",
        "scoop",
        "spiccato",
        "staccatissimo",
        "staccato",
        "stress",
        "strong-accent",
        "tenuto",
        "unstress",
    }
)
_ORNAMENT_TOKENS = frozenset(
    {
        "delayed-inverted-turn",
        "delayed-turn",
        "inverted-mordent",
        "inverted-turn",
        "mordent",
        "schleifer",
        "shake",
        "tremolo",
        "trill-mark",
        "turn",
    }
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _rank(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def select_group_balanced_paths(
    paths: list[str], count: int, seed: str = DEFAULT_SEED
) -> list[str]:
    if count < 0 or count > len(paths):
        raise ValueError(f"Cannot select {count} from {len(paths)} paths")
    grouped: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        parts = Path(path).parts
        if len(parts) < 3 or parts[0] != "samples":
            raise ValueError(f"Unexpected OLiMPiC sample path: {path!r}")
        grouped[parts[1]].append(path)
    for group, members in grouped.items():
        members.sort(key=lambda value: _rank(f"{seed}:{group}", value))
    group_order = sorted(grouped, key=lambda value: _rank(seed, value))

    selected: list[str] = []
    round_index = 0
    while len(selected) < count:
        added = False
        for group in group_order:
            members = grouped[group]
            if round_index < len(members):
                selected.append(members[round_index])
                added = True
                if len(selected) == count:
                    return selected
        if not added:
            break
        round_index += 1
    if len(selected) != count:
        raise RuntimeError(f"Selected {len(selected)} paths, expected {count}")
    return selected


def lmx_family_counts(lmx: str) -> dict[str, int]:
    counts = {family: 0 for family in FAMILY_PRIORITY_WEIGHTS}
    for token in lmx.split():
        if token.startswith(("tie:", "tied:")):
            counts["tie"] += 1
        if token.startswith("slur:"):
            counts["slur"] += 1
        if token.startswith("beam:"):
            counts["beam"] += 1
        if token in _ARTICULATION_TOKENS:
            counts["articulation"] += 1
        if token in _ORNAMENT_TOKENS or token.startswith("wavy-line:"):
            counts["ornament"] += 1
    return counts


def select_family_priority_paths(
    paths: list[str],
    count: int,
    lmx_loader: Callable[[str], str],
    seed: str = DEFAULT_FAMILY_PRIORITY_SEED,
) -> list[str]:
    """Keep work balance while preferring underperforming notation families.

    This is deliberately not a global top-k. A global ranking would collapse
    source-work diversity and make calibration gains less likely to generalize.
    Each work therefore contributes in deterministic round-robin order, while
    family-rich samples are preferred *within* that work.
    """

    if count < 0 or count > len(paths):
        raise ValueError(f"Cannot select {count} from {len(paths)} paths")
    grouped: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for path in paths:
        parts = Path(path).parts
        if len(parts) < 3 or parts[0] != "samples":
            raise ValueError(f"Unexpected OLiMPiC sample path: {path!r}")
        family_counts = lmx_family_counts(lmx_loader(path))
        priority = sum(
            FAMILY_PRIORITY_WEIGHTS[family] * min(tokens, 8)
            for family, tokens in family_counts.items()
        )
        # Prefer samples that cover several families when weighted scores tie.
        priority += 0.25 * sum(tokens > 0 for tokens in family_counts.values())
        grouped[parts[1]].append((priority, path))
    for group, members in grouped.items():
        members.sort(
            key=lambda item: (
                -item[0],
                _rank(f"{seed}:{group}", item[1]),
            )
        )
    group_order = sorted(grouped, key=lambda value: _rank(seed, value))

    selected: list[str] = []
    round_index = 0
    while len(selected) < count:
        added = False
        for group in group_order:
            members = grouped[group]
            if round_index < len(members):
                selected.append(members[round_index][1])
                added = True
                if len(selected) == count:
                    return selected
        if not added:
            break
        round_index += 1
    if len(selected) != count:
        raise RuntimeError(f"Selected {len(selected)} paths, expected {count}")
    return selected


def summarize_lmx_families(
    paths: list[str], lmx_loader: Callable[[str], str]
) -> dict[str, dict[str, int]]:
    tokens = {family: 0 for family in FAMILY_PRIORITY_WEIGHTS}
    samples = {family: 0 for family in FAMILY_PRIORITY_WEIGHTS}
    for path in paths:
        counts = lmx_family_counts(lmx_loader(path))
        for family, count in counts.items():
            tokens[family] += count
            samples[family] += int(count > 0)
    return {"tokens": tokens, "samples_with_family": samples}


def _load_synthetic_entry(root: Path, relative: str) -> dict[str, Any]:
    prefix = root.joinpath(*Path(relative).parts)
    image = prefix.with_suffix(".png")
    lmx = prefix.with_suffix(".lmx")
    musicxml = prefix.with_suffix(".musicxml")
    for path in (image, lmx, musicxml):
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "path": f"synthetic-replay/{relative}",
        "image": image.read_bytes(),
        "lmx": lmx.read_text(encoding="utf-8").strip(),
        "musicxml": musicxml.read_text(encoding="utf-8"),
    }


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def build_mixed_dataset(
    *,
    real_prepared: Path,
    synthetic_root: Path,
    output_dir: Path,
    replay_samples: int,
    seed: str,
    selection_profile: str = "group-balanced-v1",
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    base_manifest = json.loads(
        (real_prepared / "manifest.json").read_text(encoding="utf-8")
    )
    if any(
        int(value) != 0
        for value in base_manifest["source_group_overlap"].values()
    ):
        raise ValueError("The real-scan prepared dataset already contains leakage")
    if base_manifest.get("name") == SOURCE_DOCUMENT_SAFE_BASE:
        source_document_overlap = base_manifest.get(
            "source_document_overlap"
        )
        if not isinstance(source_document_overlap, dict) or any(
            int(value) != 0 for value in source_document_overlap.values()
        ):
            raise ValueError(
                "The real-scan prepared dataset contains physical "
                "source-document leakage"
            )

    train_list = synthetic_root / "samples.train.txt"
    all_paths = [line.strip() for line in train_list.read_text().splitlines() if line]

    def load_lmx(relative: str) -> str:
        return (
            (synthetic_root / Path(relative))
            .with_suffix(".lmx")
            .read_text(encoding="utf-8")
        )

    uniform_selected = select_group_balanced_paths(
        all_paths, replay_samples, seed
    )
    if selection_profile == "group-balanced-v1":
        selected = uniform_selected
    elif selection_profile == "family-priority-work-balanced-v1":
        selected = select_family_priority_paths(
            all_paths, replay_samples, load_lmx, seed
        )
    else:
        raise ValueError(f"Unknown replay selection profile: {selection_profile}")
    replay_group_ids = sorted({Path(path).parts[1] for path in selected})
    protected_groups = set()
    for split in ("calibration", "candidate_test"):
        protected_groups.update(base_manifest["splits"][split]["group_ids"])
    shared = protected_groups & set(replay_group_ids)
    if shared:
        raise ValueError(f"Synthetic replay leaks protected source works: {sorted(shared)}")

    with (real_prepared / "train.pickle").open("rb") as stream:
        real_rows = pickle.load(stream)
    replay_rows = [_load_synthetic_entry(synthetic_root, path) for path in selected]
    mixed_rows = real_rows + replay_rows
    mixed_rows.sort(key=lambda row: _rank(seed, str(row["path"])))

    output_dir.mkdir(parents=True)
    with (output_dir / "train.pickle").open("wb") as stream:
        pickle.dump(mixed_rows, stream, protocol=pickle.HIGHEST_PROTOCOL)
    for split in ("calibration", "candidate_test"):
        _link_or_copy(
            real_prepared / f"{split}.pickle",
            output_dir / f"{split}.pickle",
        )

    manifest = json.loads(json.dumps(base_manifest))
    if (
        base_manifest.get("name") == SOURCE_DOCUMENT_SAFE_BASE
        and selection_profile == "family-priority-work-balanced-v1"
    ):
        manifest["name"] = SOURCE_DOCUMENT_SAFE_FAMILY_PRIORITY_NAME
    else:
        manifest["name"] = (
            "scorescan-olimpic-real-plus-synthetic-replay-v3"
            if selection_profile == "family-priority-work-balanced-v1"
            else "scorescan-olimpic-real-plus-synthetic-replay-v2"
        )
    manifest["seed"] = seed
    manifest["derived_from"] = {
        "manifest": str(real_prepared / "manifest.json"),
        "manifest_sha256": sha256_file(real_prepared / "manifest.json"),
    }
    manifest["policy"]["synthetic_replay_source"] = (
        "published_synthetic_train_only"
    )
    manifest["policy"]["training_source"] = (
        "published_dev_real_scan_plus_published_synthetic_train_replay"
    )
    manifest["sources"] = [
        manifest.pop("source"),
        {
            "license": "CC-BY-SA-4.0",
            "name": "OLiMPiC 1.0 Synthetic",
            "path": str(synthetic_root),
            "role": "catastrophic-forgetting replay only",
            "upstream_repository": "https://github.com/ufal/olimpic-icdar24",
        },
    ]
    train_details = manifest["splits"]["train"]
    real_group_ids = list(train_details["group_ids"])
    selection_sha256 = hashlib.sha256("\n".join(selected).encode()).hexdigest()
    train_details.update(
        {
            "fingerprint": hashlib.sha256(
                (
                    train_details["fingerprint"]
                    + "\0"
                    + selection_sha256
                ).encode()
            ).hexdigest(),
            "group_ids": sorted(set(real_group_ids) | set(replay_group_ids)),
            "image_bytes": sum(len(row["image"]) for row in mixed_rows),
            "image_extensions": {".png": len(mixed_rows)},
            "lmx_tokens": sum(len(row["lmx"].split()) for row in mixed_rows),
            "pickle": str(output_dir / "train.pickle"),
            "pickle_bytes": (output_dir / "train.pickle").stat().st_size,
            "pickle_sha256": sha256_file(output_dir / "train.pickle"),
            "samples": len(mixed_rows),
            "source_groups": len(set(real_group_ids) | set(replay_group_ids)),
            "real_scan_samples": len(real_rows),
            "synthetic_replay_samples": len(replay_rows),
            "synthetic_replay_groups": len(replay_group_ids),
            "synthetic_selection_sha256": selection_sha256,
            "synthetic_selection_profile": selection_profile,
            "synthetic_family_priority_weights": (
                FAMILY_PRIORITY_WEIGHTS
                if selection_profile == "family-priority-work-balanced-v1"
                else {}
            ),
            "synthetic_selected_family_summary": summarize_lmx_families(
                selected, load_lmx
            ),
            "synthetic_uniform_family_summary": summarize_lmx_families(
                uniform_selected, load_lmx
            ),
        }
    )
    for split in ("calibration", "candidate_test"):
        details = manifest["splits"][split]
        details["pickle"] = str(output_dir / f"{split}.pickle")
        details["pickle_bytes"] = (output_dir / f"{split}.pickle").stat().st_size
        details["pickle_sha256"] = sha256_file(output_dir / f"{split}.pickle")

    train_groups = set(train_details["group_ids"])
    calibration_groups = set(manifest["splits"]["calibration"]["group_ids"])
    candidate_groups = set(manifest["splits"]["candidate_test"]["group_ids"])
    manifest["source_group_overlap"] = {
        "train_calibration": len(train_groups & calibration_groups),
        "train_candidate_test": len(train_groups & candidate_groups),
        "calibration_candidate_test": len(calibration_groups & candidate_groups),
    }
    if any(manifest["source_group_overlap"].values()):
        raise ValueError(f"Unexpected leakage: {manifest['source_group_overlap']}")
    if (
        manifest.get("name") == SOURCE_DOCUMENT_SAFE_FAMILY_PRIORITY_NAME
        and any(manifest["source_document_overlap"].values())
    ):
        raise ValueError(
            "Unexpected physical source-document leakage: "
            f"{manifest['source_document_overlap']}"
        )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-prepared", type=Path, required=True)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-samples", type=int, default=1179)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument(
        "--selection-profile",
        choices=(
            "group-balanced-v1",
            "family-priority-work-balanced-v1",
        ),
        default="group-balanced-v1",
    )
    args = parser.parse_args()
    manifest = build_mixed_dataset(
        real_prepared=args.real_prepared,
        synthetic_root=args.synthetic_root,
        output_dir=args.output_dir,
        replay_samples=args.replay_samples,
        seed=args.seed,
        selection_profile=args.selection_profile,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
