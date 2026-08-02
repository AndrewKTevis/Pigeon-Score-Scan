from __future__ import annotations

"""Prepare source-group-isolated OLiMPiC real-scan pickles for Zeus training.

Only the published OLiMPiC ``dev`` scores are split into fine-tuning and
calibration groups.  The published ``test`` scores remain a separate candidate
benchmark.  The output format is compatible with the upstream Zeus loader.
"""

import argparse
import hashlib
import json
import os
import pickle
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Iterable

import yaml


FORMAT = 1
DEFAULT_SEED = "scorescan-olimpic-real-v1"
SOURCE_DOCUMENT_SAFE_NAME = "scorescan-olimpic-real-v2-source-document-safe"


def read_samples(path: Path) -> list[str]:
    samples = [
        line.strip().replace("\\", "/")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not samples:
        raise ValueError(f"split is empty: {path}")
    if len(samples) != len(set(samples)):
        raise ValueError(f"split contains duplicate samples: {path}")
    return samples


def source_group(sample: str) -> str:
    parts = PurePosixPath(sample).parts
    if len(parts) < 3 or parts[0] != "samples":
        raise ValueError(f"unexpected OLiMPiC sample path: {sample!r}")
    return parts[1]


def group_samples(samples: Iterable[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for sample in samples:
        result.setdefault(source_group(sample), []).append(sample)
    for values in result.values():
        values.sort()
    return result


def read_source_documents(mapping_root: Path) -> dict[str, str]:
    """Map each OpenScore work id to exactly one physical IMSLP document."""

    result: dict[str, str] = {}
    for path in sorted(mapping_root.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not payload:
            raise ValueError(f"invalid OLiMPiC page mapping: {path}")
        documents = {
            str(row.get("imslpDocument", "")).lstrip("#")
            for row in payload.values()
            if isinstance(row, dict)
        }
        if len(documents) != 1 or not next(iter(documents), "").isdigit():
            raise ValueError(
                f"OLiMPiC score must map to one IMSLP document: {path}"
            )
        result[path.stem] = next(iter(documents))
    if not result:
        raise ValueError(f"no OLiMPiC page mappings found: {mapping_root}")
    return result


def split_development_groups(
    development_samples: list[str],
    published_test_samples: list[str],
    *,
    training_group_count: int,
    seed: str,
    source_documents: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    development = group_samples(development_samples)
    published_test = group_samples(published_test_samples)
    overlap = set(development) & set(published_test)
    if overlap:
        raise ValueError(f"published OLiMPiC dev/test source groups overlap: {sorted(overlap)}")
    if training_group_count <= 0 or training_group_count >= len(development):
        raise ValueError(
            "training_group_count must leave at least one calibration group "
            f"(development groups: {len(development)})"
        )
    ranked_groups = sorted(
        development,
        key=lambda group: hashlib.sha256(f"{seed}\0{group}".encode()).hexdigest(),
    )
    training_groups = set(ranked_groups[:training_group_count])
    calibration_groups = set(ranked_groups[training_group_count:])
    if source_documents is not None:
        required_groups = set(development) | set(published_test)
        missing_documents = sorted(required_groups - set(source_documents))
        if missing_documents:
            raise ValueError(
                "OLiMPiC source-document map is incomplete: "
                f"{missing_documents[:3]}"
            )
        published_document_overlap = (
            {source_documents[group] for group in development}
            & {source_documents[group] for group in published_test}
        )
        if published_document_overlap:
            raise ValueError(
                "published OLiMPiC dev/test IMSLP documents overlap: "
                f"{sorted(published_document_overlap)}"
            )

        # Preserve every sample the existing base model has already seen as a
        # training sample. When an initially held-out score shares its physical
        # edition with training, move that held-out score into training instead
        # of moving a previously trained score into calibration.
        training_documents = {
            source_documents[group] for group in training_groups
        }
        contaminated_calibration = {
            group
            for group in calibration_groups
            if source_documents[group] in training_documents
        }
        training_groups.update(contaminated_calibration)
        calibration_groups.difference_update(contaminated_calibration)
        if not calibration_groups:
            raise ValueError(
                "source-document isolation leaves no calibration groups"
            )
        remaining_overlap = (
            {source_documents[group] for group in training_groups}
            & {source_documents[group] for group in calibration_groups}
        )
        if remaining_overlap:
            raise AssertionError(
                "source-document leakage remains after regrouping"
            )
    return {
        "train": sorted(
            sample
            for group in training_groups
            for sample in development[group]
        ),
        "calibration": sorted(
            sample
            for group in calibration_groups
            for sample in development[group]
        ),
        "candidate_test": sorted(published_test_samples),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_sample_files(corpus: Path, sample: str) -> tuple[Path, Path, Path]:
    base = corpus.joinpath(*PurePosixPath(sample).parts)
    images = [base.with_suffix(".png"), base.with_suffix(".jpg")]
    image = next((path for path in images if path.is_file()), None)
    lmx = base.with_suffix(".lmx")
    musicxml = base.with_suffix(".musicxml")
    if image is None or not lmx.is_file() or not musicxml.is_file():
        raise FileNotFoundError(f"incomplete OLiMPiC sample: {sample}")
    return image, lmx, musicxml


def pack_split(
    corpus: Path,
    samples: list[str],
    output: Path,
    *,
    source_documents: dict[str, str] | None = None,
) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    digest = hashlib.sha256()
    image_bytes = 0
    lmx_tokens = 0
    extensions: Counter[str] = Counter()
    for sample in samples:
        image, lmx, musicxml = _resolve_sample_files(corpus, sample)
        image_payload = image.read_bytes()
        lmx_payload = lmx.read_text(encoding="utf-8").rstrip("\r\n")
        musicxml_payload = musicxml.read_text(encoding="utf-8")
        if "\n" in lmx_payload or "\r" in lmx_payload:
            raise ValueError(f"LMX is not one line: {lmx}")
        entries.append(
            {
                "path": sample,
                "image": image_payload,
                "lmx": lmx_payload,
                "musicxml": musicxml_payload,
            }
        )
        digest.update(sample.encode("utf-8"))
        for path in (image, lmx, musicxml):
            digest.update(bytes.fromhex(_sha256(path)))
        image_bytes += len(image_payload)
        lmx_tokens += len(lmx_payload.split())
        extensions[image.suffix.lower()] += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(entries, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    groups = sorted({source_group(sample) for sample in samples})
    report: dict[str, object] = {
        "samples": len(samples),
        "source_groups": len(groups),
        "group_ids": groups,
        "image_bytes": image_bytes,
        "lmx_tokens": lmx_tokens,
        "image_extensions": dict(sorted(extensions.items())),
        "fingerprint": digest.hexdigest(),
        "pickle": str(output.resolve()),
        "pickle_bytes": output.stat().st_size,
        "pickle_sha256": _sha256(output),
    }
    if source_documents is not None:
        documents = sorted({source_documents[group] for group in groups})
        report["source_documents"] = len(documents)
        report["source_document_ids"] = documents
    return report


def prepare(
    corpus: Path,
    output: Path,
    *,
    training_group_count: int,
    seed: str,
    source_mapping_root: Path | None = None,
) -> dict[str, object]:
    development_samples = read_samples(corpus / "samples.dev.txt")
    published_test_samples = read_samples(corpus / "samples.test.txt")
    source_documents = (
        read_source_documents(source_mapping_root)
        if source_mapping_root is not None
        else None
    )
    score_only_splits = split_development_groups(
        development_samples,
        published_test_samples,
        training_group_count=training_group_count,
        seed=seed,
    )
    splits = split_development_groups(
        development_samples,
        published_test_samples,
        training_group_count=training_group_count,
        seed=seed,
        source_documents=source_documents,
    )
    output.mkdir(parents=True, exist_ok=True)
    split_reports: dict[str, dict[str, object]] = {}
    for split_name, samples in splits.items():
        print(f"[pack] {split_name}: {len(samples)} systems", flush=True)
        split_reports[split_name] = pack_split(
            corpus,
            samples,
            output / f"{split_name}.pickle",
            source_documents=source_documents,
        )
    group_sets = {
        name: set(report["group_ids"])
        for name, report in split_reports.items()
    }
    if (
        group_sets["train"] & group_sets["calibration"]
        or group_sets["train"] & group_sets["candidate_test"]
        or group_sets["calibration"] & group_sets["candidate_test"]
    ):
        raise AssertionError("source-group leakage detected after packing")
    source_document_overlap = {
        "train_calibration": 0,
        "train_candidate_test": 0,
        "calibration_candidate_test": 0,
    }
    moved_to_training: list[str] = []
    if source_documents is not None:
        document_sets = {
            name: set(report["source_document_ids"])
            for name, report in split_reports.items()
        }
        source_document_overlap = {
            "train_calibration": len(
                document_sets["train"] & document_sets["calibration"]
            ),
            "train_candidate_test": len(
                document_sets["train"] & document_sets["candidate_test"]
            ),
            "calibration_candidate_test": len(
                document_sets["calibration"]
                & document_sets["candidate_test"]
            ),
        }
        if any(source_document_overlap.values()):
            raise AssertionError(
                "physical source-document leakage detected after packing"
            )
        moved_to_training = sorted(
            {
                source_group(sample) for sample in splits["train"]
            }
            - {
                source_group(sample)
                for sample in score_only_splits["train"]
            }
        )

    manifest: dict[str, object] = {
        "format": 2 if source_documents is not None else FORMAT,
        "name": (
            SOURCE_DOCUMENT_SAFE_NAME
            if source_documents is not None
            else "scorescan-olimpic-real-v1"
        ),
        "seed": seed,
        "source": {
            "name": "OLiMPiC 1.0 Scanned",
            "path": str(corpus.resolve()),
            "license": "CC-BY-SA-4.0",
            "official_dataset": "https://hdl.handle.net/11234/1-5419",
            "upstream_repository": "https://github.com/ufal/olimpic-icdar24",
        },
        "policy": {
            "training_source": "published_dev_only",
            "calibration_source": "published_dev_only",
            "candidate_test_source": "published_test_only",
            "group_key": (
                "OpenScore Lieder score id plus physical IMSLP document"
                if source_documents is not None
                else "OpenScore Lieder score id"
            ),
            "target_training_score_groups": training_group_count,
            "calibration_scores_moved_to_training_for_document_isolation": (
                moved_to_training
            ),
            "final_product_frozen_benchmark": False,
        },
        "splits": split_reports,
        "source_group_overlap": {
            "train_calibration": 0,
            "train_candidate_test": 0,
            "calibration_candidate_test": 0,
        },
        "source_document_overlap": source_document_overlap,
    }
    temporary = output / "manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output / "manifest.json")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--training-groups", type=int, default=80)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument(
        "--source-mappings",
        type=Path,
        help=(
            "corpus_to_imslp directory; when provided, calibration is also "
            "isolated by physical IMSLP source document"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = prepare(
        args.corpus.resolve(),
        args.output.resolve(),
        training_group_count=args.training_groups,
        seed=args.seed,
        source_mapping_root=(
            args.source_mappings.resolve()
            if args.source_mappings is not None
            else None
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
