#!/usr/bin/env python3
"""Train a RetinaNet verifier for positioned music marks and symbols."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import copy
from contextlib import contextmanager
from functools import partial
import hashlib
import json
import math
import os
import random
import time
import types
import uuid
from pathlib import Path
from typing import Any

from app.tools.dense_detection_metrics import compute_dense_detection_metrics
from app.tools.prepare_openscore_svg_regions import (
    COMPLETE_PAGE_TARGET_ASSIGNMENT_VERSION,
    COMPLETE_PAGE_TARGET_PROVENANCE,
    LONG_SPAN_SEMANTIC_CATEGORIES,
)
from app.tools.semantic_target_visibility import (
    OVERSIZED_FRAGMENT_VISIBILITY_VERSION,
    target_fragment_is_visible,
)

DETECTOR_NMS_IOU = 0.75
DETECTOR_FOREGROUND_IOU_THRESHOLD = 0.35
DETECTOR_BACKGROUND_IOU_THRESHOLD = 0.25
DETECTOR_STRICT_FOREGROUND_IOU_THRESHOLD = 0.50
DETECTOR_STRICT_BACKGROUND_IOU_THRESHOLD = 0.40
DETECTOR_STRICT_MATCH_CLASSES = frozenset(
    {
        "arpeggio",
        "augmentationDot",
        "beam",
        "expressionText",
        "genericBarline",
        "genericTimeSignature",
        "graceSlash",
        "jumpText",
        "markerText",
        "scoreText",
        "staffText",
    }
)
DETECTOR_MODEL_CONTRACT_VERSION = (
    "retinanet-r50-fpnv2-music-anchors-groupnorm-giou-"
    "matcher35-25-nms75@3"
)
DETECTOR_CLASS_AWARE_MODEL_CONTRACT_VERSION = (
    "retinanet-r50-fpnv2-music-anchors-groupnorm-giou-"
    "classaware-matcher35-25-strict50-40-nms75@4"
)
PRIORITY_SELECTION_PROTOCOL = (
    "support-filtered-overall-and-priority-macro-map@2"
)
DETECTOR_TRAINING_DATASET_ROLES = frozenset(
    {
        "training_only",
        "training_only_synthetic_semantic_geometry",
        "training_only_disjoint_from_external_release_holdout",
    }
)
LEGACY_ROLELESS_DEEPSCORES_NAME_PREFIX = "scorescan-deepscores-v2-"
RELATION_SUBSET_CONTRACT_VERSION = "relation-detector-class-subset@2"


def detector_model_contract(
    *,
    class_aware_matcher: bool = False,
) -> dict[str, Any]:
    """Return the canonical, release-audited detector architecture contract."""

    contract: dict[str, Any] = {
        "version": DETECTOR_MODEL_CONTRACT_VERSION,
        "nms_iou": DETECTOR_NMS_IOU,
        "foreground_iou_threshold": DETECTOR_FOREGROUND_IOU_THRESHOLD,
        "background_iou_threshold": DETECTOR_BACKGROUND_IOU_THRESHOLD,
    }
    if not class_aware_matcher:
        return contract
    contract.update(
        {
            "version": DETECTOR_CLASS_AWARE_MODEL_CONTRACT_VERSION,
            "strict_foreground_iou_threshold": (
            DETECTOR_STRICT_FOREGROUND_IOU_THRESHOLD
            ),
            "strict_background_iou_threshold": (
                DETECTOR_STRICT_BACKGROUND_IOU_THRESHOLD
            ),
            "strict_match_classes": sorted(DETECTOR_STRICT_MATCH_CLASSES),
        }
    )
    return contract


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def remove_empty_detector_resume_directory(
    output_dir: Path,
    *,
    resume: bool,
) -> bool:
    """Remove only a legacy, completely empty pre-checkpoint output directory.

    Older entry points created the model directory before completing dataset
    and class-support validation.  A validation refusal could therefore leave
    an empty directory that looked like a corrupt resume on the next attempt.
    No non-empty or linked path is ever repaired implicitly.
    """

    if not resume or not output_dir.exists():
        return False
    if output_dir.is_symlink():
        raise RuntimeError(
            f"refusing linked detector resume output: {output_dir}"
        )
    if not output_dir.is_dir() or next(output_dir.iterdir(), None) is not None:
        return False
    output_dir.rmdir()
    return True


def reopen_runtime_truncated_detector_run(
    output_dir: Path,
    *,
    resume: bool,
    planned_epochs: int,
    runtime_stop_after_epoch: int | None,
) -> dict[str, Any] | None:
    """Reopen a deliberately truncated, checkpointed run without losing evidence.

    A bounded ablation writes the normal completed report and metrics so its
    decision window is independently inspectable.  If that window justifies a
    later stage, ``--resume`` may continue the exact optimizer/RNG state.  The
    prior report is first copied to a content-identical epoch snapshot, while
    the finalized metrics are atomically moved back to the partial name that
    the normal recovery path requires.
    """

    completed_report = output_dir / "training_report.json"
    if not completed_report.is_file():
        return None
    if not resume:
        return None
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise RuntimeError(
            f"refusing linked or invalid detector output: {output_dir}"
        )
    report = json.loads(completed_report.read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or report.get("format") != 1
        or report.get("runtime_truncated") is not True
    ):
        return None
    completed_epochs = int(report.get("completed_epochs", -1))
    report_stop_epoch = int(report.get("runtime_stop_after_epoch", -1))
    report_planned_epochs = int(report.get("planned_epochs", -1))
    requested_final_epoch = min(
        planned_epochs,
        (
            runtime_stop_after_epoch
            if runtime_stop_after_epoch is not None
            else planned_epochs
        ),
    )
    if (
        planned_epochs <= 0
        or report_planned_epochs != planned_epochs
        or completed_epochs <= 0
        or report_stop_epoch != completed_epochs
        or requested_final_epoch <= completed_epochs
    ):
        raise RuntimeError(
            "completed detector ablation cannot be continued to the requested "
            "runtime epoch"
        )

    root = output_dir.resolve()
    checkpoint = output_dir / "checkpoint.last.pt"
    run_config = output_dir / "run_config.json"
    metrics = output_dir / "metrics.json"
    partial_metrics = output_dir / "metrics.partial.json"
    snapshot = (
        output_dir
        / f"training_report.runtime-stop-{completed_epochs}.json"
    )
    for required in (completed_report, checkpoint, run_config):
        if (
            required.is_symlink()
            or not required.is_file()
            or not required.resolve().is_relative_to(root)
        ):
            raise RuntimeError(
                f"runtime-truncated detector artifact is invalid: {required}"
            )
    if metrics.exists() == partial_metrics.exists():
        raise RuntimeError(
            "runtime-truncated detector must contain exactly one metrics state"
        )

    report_sha256 = sha256_file(completed_report)
    if snapshot.exists():
        if (
            snapshot.is_symlink()
            or not snapshot.is_file()
            or not snapshot.resolve().is_relative_to(root)
            or sha256_file(snapshot) != report_sha256
        ):
            raise RuntimeError(
                "runtime-stop report snapshot conflicts with completed report"
            )
    else:
        copied_sha256, copied_bytes = _copy_file_with_sha256(
            completed_report,
            snapshot,
        )
        if (
            copied_sha256 != report_sha256
            or copied_bytes != completed_report.stat().st_size
            or sha256_file(snapshot) != report_sha256
        ):
            raise RuntimeError("runtime-stop report snapshot copy mismatch")
    if metrics.is_file():
        os.replace(metrics, partial_metrics)
    completed_report.unlink()
    return {
        "state": "reopened_runtime_truncated_detector_run",
        "completed_epochs": completed_epochs,
        "requested_final_epoch": requested_final_epoch,
        "report_snapshot": str(snapshot),
        "report_snapshot_sha256": report_sha256,
    }


def finalize_detector_recovery_checkpoint(
    checkpoint_path: Path,
    *,
    acceptance_failures: list[str],
    external_acceptance_pending: bool = False,
) -> bool:
    """Retain resumable state while any configured accuracy gate is unresolved."""

    if acceptance_failures or external_acceptance_pending:
        return checkpoint_path.is_file()
    checkpoint_path.unlink(missing_ok=True)
    return checkpoint_path.is_file()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_grayscale_crop(
    image_path: Path,
    crop_xyxy: list[int] | tuple[int, int, int, int],
) -> Any:
    """Decode a page crop before color conversion.

    Prepared detector pages are much larger than one 1024px training tile.
    Converting the complete page to RGB before cropping performs avoidable
    work and multiplies transient memory.  Cropping first is pixel-equivalent
    for the supported printed-score image modes and leaves augmentation and
    model tensors unchanged.
    """

    from PIL import Image, ImageOps

    with Image.open(image_path) as source:
        return ImageOps.grayscale(source.crop(tuple(crop_xyxy)))


def _copy_file_with_sha256(source: Path, destination: Path) -> tuple[str, int]:
    """Atomically copy one cache object while hashing the exact bytes read."""

    digest = hashlib.sha256()
    size = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
            while chunk := input_stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
                output_stream.write(chunk)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return digest.hexdigest(), size


def prepare_verified_detector_image_cache(
    items: list[tuple[dict[str, Any], Path]],
    cache_dir: Path,
    *,
    populate: bool,
) -> tuple[dict[tuple[str, str], Path], dict[str, Any]]:
    """Map detector images to a content-addressed, byte-verified local cache.

    The cache is a runtime I/O optimization only. Every current source image is
    hashed, every cache blob is independently hashed, and any mismatch fails
    closed. Thus changing the cache location cannot silently change training
    pixels or weaken the immutable run configuration.
    """

    unique_pairs: dict[tuple[str, str], Path] = {}
    for row, raw_root in items:
        relative_text = str(row.get("image", ""))
        relative = Path(relative_text)
        if (
            not relative_text
            or relative.is_absolute()
            or ".." in relative.parts
        ):
            raise ValueError(
                f"detector image path is not a safe relative path: {relative_text!r}"
            )
        root_text = str(raw_root)
        unique_pairs[(root_text, relative_text)] = raw_root

    unique_sources: dict[tuple[str, str], Path] = {}
    resolved_roots: dict[str, Path] = {}
    for (root_text, relative_text), raw_root in sorted(unique_pairs.items()):
        relative = Path(relative_text)
        resolved_root = resolved_roots.get(root_text)
        if resolved_root is None:
            resolved_root = raw_root.resolve()
            resolved_roots[root_text] = resolved_root
        source = (resolved_root / relative).resolve()
        try:
            source.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(
                f"detector image escapes its image root: {relative_text!r}"
            ) from exc
        unique_sources[(root_text, relative_text)] = source

    if populate:
        cache_dir.mkdir(parents=True, exist_ok=True)
    elif not cache_dir.is_dir():
        raise FileNotFoundError(
            f"verified detector image cache is absent: {cache_dir}"
        )

    def audit_source(
        item: tuple[tuple[str, str], Path],
    ) -> tuple[tuple[str, str], Path, str, int, int]:
        key, source = item
        if not source.is_file():
            raise FileNotFoundError(f"detector source image is absent: {source}")
        before = source.stat()
        source_sha256 = sha256_file(source)
        after = source.stat()
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise RuntimeError(
                f"detector source image changed while caching: {source}"
            )
        return (
            key,
            source,
            source_sha256,
            before.st_size,
            before.st_mtime_ns,
        )

    # The cache contains thousands of independent small page files. Four
    # bounded I/O workers hide 9p/open latency while retaining full-byte hashes
    # and deterministic result order.
    audit_workers = min(4, max(1, len(unique_sources)))
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=audit_workers
    ) as executor:
        source_audits = list(
            executor.map(
                audit_source,
                [
                    (key, unique_sources[key])
                    for key in sorted(unique_sources)
                ],
            )
        )

    blob_specs: dict[str, tuple[Path, int, Path, int]] = {}
    for _, source, source_sha256, source_size, source_mtime_ns in source_audits:
        blob = (
            cache_dir
            / "objects"
            / source_sha256[:2]
            / f"{source_sha256}.img"
        )
        previous = blob_specs.setdefault(
            source_sha256,
            (blob, source_size, source, source_mtime_ns),
        )
        if previous[1] != source_size:
            raise RuntimeError(
                "identical detector source hashes have different sizes"
            )

    def populate_blob(
        item: tuple[str, tuple[Path, int, Path, int]],
    ) -> None:
        source_sha256, (blob, source_size, source, source_mtime_ns) = item
        blob.parent.mkdir(parents=True, exist_ok=True)
        if blob.exists():
            return
        temporary_target = (
            cache_dir / "incoming" / f"{uuid.uuid4().hex}.img"
        )
        copied_sha256, copied_size = _copy_file_with_sha256(
            source,
            temporary_target,
        )
        after_copy = source.stat()
        if (
            copied_sha256 != source_sha256
            or copied_size != source_size
            or after_copy.st_size != source_size
            or after_copy.st_mtime_ns != source_mtime_ns
        ):
            temporary_target.unlink(missing_ok=True)
            raise RuntimeError(
                f"detector source image changed while caching: {source}"
            )
        os.replace(temporary_target, blob)

    if populate:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(4, max(1, len(blob_specs)))
        ) as executor:
            list(executor.map(populate_blob, sorted(blob_specs.items())))

    def verify_blob(
        item: tuple[str, tuple[Path, int, Path, int]],
    ) -> tuple[str, Path, int]:
        source_sha256, (blob, source_size, _, _) = item
        if not blob.is_file():
            raise FileNotFoundError(
                f"verified detector cache blob is absent: {blob}"
            )
        if blob.stat().st_size != source_size:
            raise RuntimeError(
                f"detector cache blob size mismatch: {blob}"
            )
        if sha256_file(blob) != source_sha256:
            raise RuntimeError(
                f"detector cache blob hash mismatch: {blob}"
            )
        return source_sha256, blob, source_size

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(4, max(1, len(blob_specs)))
    ) as executor:
        verified_results = list(
            executor.map(verify_blob, sorted(blob_specs.items()))
        )
    verified_blobs = {
        source_sha256: (blob, source_size)
        for source_sha256, blob, source_size in verified_results
    }

    mappings: dict[tuple[str, str], Path] = {}
    records: list[dict[str, Any]] = []
    for key, _, source_sha256, source_size, _ in source_audits:
        blob = verified_blobs[source_sha256][0]
        mappings[key] = blob
        records.append(
            {
                "image_root": key[0],
                "relative_path": key[1],
                "bytes": source_size,
                "sha256": source_sha256,
            }
        )

    manifest = {
        "format": 1,
        "contract": "scorescan-byte-verified-detector-image-cache@1",
        "source_files": len(records),
        "unique_blobs": len(verified_blobs),
        "bytes": sum(size for _, size in verified_blobs.values()),
        "records": records,
    }
    manifest_path = cache_dir / "scorescan-image-cache-v1.json"
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    summary = {
        key: value for key, value in manifest.items() if key != "records"
    }
    summary["manifest"] = str(manifest_path)
    summary["manifest_sha256"] = sha256_file(manifest_path)
    summary["populated"] = bool(populate)
    return mappings, summary


def detector_checkpoint_position(
    checkpoint: dict[str, Any],
    *,
    total_epochs: int,
    accumulate: int,
) -> tuple[int, int]:
    """Validate and return completed epochs plus an optional current-epoch step."""

    completed_epochs = int(checkpoint.get("epoch", 0))
    in_progress_epoch = int(checkpoint.get("in_progress_epoch", 0))
    in_progress_step = int(checkpoint.get("in_progress_step", 0))
    if completed_epochs < 0 or completed_epochs > total_epochs:
        raise ValueError(
            f"invalid detector checkpoint epoch: {completed_epochs}"
        )
    if in_progress_step < 0:
        raise ValueError(
            f"invalid detector checkpoint step: {in_progress_step}"
        )
    if in_progress_step:
        if (
            completed_epochs >= total_epochs
            or in_progress_epoch != completed_epochs + 1
        ):
            raise ValueError(
                "detector checkpoint current epoch does not follow its "
                "completed epoch count"
            )
        if in_progress_step % accumulate:
            raise ValueError(
                "detector checkpoint step is not an optimizer boundary"
            )
    elif in_progress_epoch not in (0, completed_epochs + 1):
        raise ValueError("detector checkpoint has a step-less current epoch")
    return completed_epochs, in_progress_step


def resolve_detector_device(requested: str, *, cuda_available: bool) -> str:
    """Resolve an explicit training device without silently occupying the GPU."""

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not cuda_available:
            raise RuntimeError("CUDA was requested but is not available")
        return "cuda"
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    raise ValueError(f"unsupported detector device: {requested!r}")


def stable_subset(
    rows: list[dict[str, Any]], limit: int | None, seed: int
) -> list[dict[str, Any]]:
    if limit is None or limit >= len(rows):
        return rows
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}\0{row['split']}\0{row['image_id']}\0{row['crop_xyxy']}".encode()
        ).hexdigest(),
    )
    return ranked[:limit]


def detector_occurrence_seed(
    *,
    base_seed: int,
    epoch_number: int,
    sample_position: int,
    item_index: int,
) -> int:
    """Derive one stable local augmentation seed for a sampled occurrence."""

    if epoch_number <= 0 or sample_position < 0 or item_index < 0:
        raise ValueError("invalid detector occurrence seed coordinates")
    digest = hashlib.sha256(
        (
            f"scorescan-detector-augmentation-v3\0{base_seed}\0"
            f"{epoch_number}\0{sample_position}\0{item_index}"
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def remaining_detector_occurrences(
    sampled_indices: list[int],
    *,
    base_seed: int,
    epoch_number: int,
    completed_steps: int,
    batch_size: int,
) -> list[tuple[int, int]]:
    """Return the exact unprocessed occurrence suffix for a resumed epoch."""

    if completed_steps < 0 or batch_size <= 0:
        raise ValueError("invalid detector resume coordinates")
    maximum_steps = math.ceil(len(sampled_indices) / batch_size)
    if completed_steps > maximum_steps:
        raise ValueError("completed detector steps exceed the sampled epoch")
    start_sample = min(
        completed_steps * batch_size,
        len(sampled_indices),
    )
    return [
        (
            int(item_index),
            detector_occurrence_seed(
                base_seed=base_seed,
                epoch_number=epoch_number,
                sample_position=sample_position,
                item_index=int(item_index),
            ),
        )
        for sample_position, item_index in enumerate(sampled_indices)
        if sample_position >= start_sample
    ]


def detector_microbatch_plan(
    batch_count: int,
    microbatch_size: int,
) -> list[tuple[int, int, float]]:
    """Partition one logical batch while preserving its mean-loss gradient."""

    if batch_count <= 0 or microbatch_size <= 0:
        raise ValueError("detector batch and microbatch sizes must be positive")
    if microbatch_size > batch_count:
        raise ValueError("detector microbatch cannot exceed the logical batch")
    return [
        (
            start,
            min(batch_count, start + microbatch_size),
            (min(batch_count, start + microbatch_size) - start) / batch_count,
        )
        for start in range(0, batch_count, microbatch_size)
    ]


def detector_runtime_microbatch_size(
    targets: list[dict[str, Any]],
    *,
    configured_microbatch_size: int,
    adaptive_full_batch_object_limit: int | None,
    adaptive_full_batch_min_free_mib: int | None = None,
    effective_free_mib: int | None = None,
) -> tuple[int, int]:
    """Select a safe fast path from CPU target density before CUDA transfer.

    ``configured_microbatch_size`` remains the dense-page fallback.  When an
    explicit object limit is configured, the complete logical batch may use one
    forward pass only if its combined ground-truth object count is within that
    audited limit.  This keeps sparse score regions fast without exposing dense
    pages to the memory behaviour of the full logical batch.
    """

    batch_count = len(targets)
    if batch_count <= 0 or configured_microbatch_size <= 0:
        raise ValueError("detector batch and microbatch sizes must be positive")
    if configured_microbatch_size > batch_count:
        raise ValueError("detector microbatch cannot exceed the logical batch")
    if (
        adaptive_full_batch_object_limit is not None
        and adaptive_full_batch_object_limit <= 0
    ):
        raise ValueError("adaptive full-batch object limit must be positive")
    if (
        adaptive_full_batch_min_free_mib is not None
        and adaptive_full_batch_min_free_mib <= 0
    ):
        raise ValueError("adaptive full-batch free-memory floor must be positive")
    if (
        adaptive_full_batch_min_free_mib is not None
        and adaptive_full_batch_object_limit is None
    ):
        raise ValueError(
            "adaptive free-memory floor requires an object limit"
        )
    if (
        adaptive_full_batch_min_free_mib is not None
        and effective_free_mib is None
    ):
        raise ValueError(
            "adaptive free-memory floor requires current memory evidence"
        )

    object_count = 0
    for target in targets:
        boxes = target.get("boxes")
        if boxes is None:
            raise ValueError("detector target is missing boxes")
        object_count += len(boxes)

    if (
        adaptive_full_batch_object_limit is not None
        and object_count <= adaptive_full_batch_object_limit
        and (
            adaptive_full_batch_min_free_mib is None
            or effective_free_mib >= adaptive_full_batch_min_free_mib
        )
    ):
        return batch_count, object_count
    return min(configured_microbatch_size, batch_count), object_count


def legacy_detector_sampled_indices(
    sample_weights: list[float],
    *,
    num_samples: int,
    epoch_loader_generator_state: Any,
) -> list[int]:
    """Recreate legacy WeightedRandomSampler indices without decoding images.

    PyTorch constructs the lazy sampler iterator, consumes one int64 from the
    shared generator for DataLoader worker base seeds, and only then advances
    WeightedRandomSampler. Mirroring that order preserves the sampled item
    sequence at an explicitly recorded worker-count transition.
    """

    import torch

    generator = torch.Generator()
    generator.set_state(epoch_loader_generator_state)
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    return torch.multinomial(
        torch.as_tensor(sample_weights, dtype=torch.double),
        int(num_samples),
        replacement=True,
        generator=generator,
    ).tolist()


def scalarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in metrics.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "numel") and value.numel() == 1:
            result[name] = float(value.item())
        elif hasattr(value, "tolist"):
            result[name] = value.tolist()
        else:
            result[name] = value
    return result


def class_aware_sample_weights(
    rows: list[dict[str, Any]],
    *,
    power: float,
    maximum_repeat: float,
) -> tuple[list[float], dict[int, int]]:
    """Build bounded class-aware weights without discarding background tiles."""

    if not 0 <= power <= 1:
        raise ValueError("class sampling power must be in [0, 1]")
    if maximum_repeat < 1:
        raise ValueError("maximum class repeat must be at least 1")
    counts: collections.Counter[int] = collections.Counter()
    for row in rows:
        counts.update({int(obj["label"]) for obj in row["objects"]})
    if not counts:
        return [1.0] * len(rows), {}
    maximum_count = max(counts.values())
    weights: list[float] = []
    for row in rows:
        labels = {int(obj["label"]) for obj in row["objects"]}
        if not labels:
            weights.append(1.0)
            continue
        rarest_repeat = max(
            (maximum_count / max(1, counts[label])) ** power for label in labels
        )
        # Half uniform and half class-aware sampling keeps broad page/style
        # coverage while still exposing rare musical marks more often.
        weights.append(
            min(maximum_repeat, 0.5 + 0.5 * max(1.0, rarest_repeat))
        )
    mean_weight = sum(weights) / max(1, len(weights))
    normalized = [weight / mean_weight for weight in weights]
    if not all(math.isfinite(weight) and weight > 0 for weight in normalized):
        raise ValueError("invalid class-aware sample weight")
    return normalized, dict(sorted(counts.items()))


def replay_mixture_sample_weights(
    primary_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    *,
    replay_fraction: float,
    power: float,
    maximum_repeat: float,
    replay_maximum_repeat: float | None = None,
) -> tuple[list[float], dict[int, int], dict[int, int]]:
    """Return class-aware weights with an exact bounded replay probability.

    Real-scan fine-tuning otherwise presents every absent synthetic class only
    as a negative and quickly erases rare notation knowledge.  Weighting each
    source independently and then assigning exact probability mass prevents a
    large replay corpus from swamping the real scans while retaining bounded
    rare-class sampling inside both sources.
    """

    if not primary_rows:
        raise ValueError("primary detector training rows are empty")
    if not 0 <= replay_fraction < 1:
        raise ValueError("replay fraction must be in [0, 1)")
    if bool(replay_rows) != (replay_fraction > 0):
        raise ValueError(
            "replay rows and a positive replay fraction must be supplied together"
        )
    primary_weights, primary_counts = class_aware_sample_weights(
        primary_rows,
        power=power,
        maximum_repeat=maximum_repeat,
    )
    if not replay_rows:
        return primary_weights, primary_counts, {}
    replay_weights, replay_counts = class_aware_sample_weights(
        replay_rows,
        power=power,
        maximum_repeat=(
            replay_maximum_repeat
            if replay_maximum_repeat is not None
            else maximum_repeat
        ),
    )
    primary_mass = 1.0 - replay_fraction
    primary_sum = sum(primary_weights)
    replay_sum = sum(replay_weights)
    combined = [
        weight * primary_mass / primary_sum
        for weight in primary_weights
    ] + [
        weight * replay_fraction / replay_sum
        for weight in replay_weights
    ]
    # WeightedRandomSampler only uses ratios.  Keeping the combined mean at one
    # makes diagnostics comparable with non-replay runs.
    combined = [weight * len(combined) for weight in combined]
    if not all(math.isfinite(weight) and weight > 0 for weight in combined):
        raise ValueError("invalid replay mixture sample weight")
    return combined, primary_counts, replay_counts


def detector_class_counts(
    rows: list[dict[str, Any]],
) -> dict[int, int]:
    counts: collections.Counter[int] = collections.Counter()
    for row in rows:
        counts.update(int(obj["label"]) for obj in row["objects"])
    return dict(sorted(counts.items()))


def insufficient_required_class_support(
    required_classes: dict[str, float],
    *,
    class_name_by_label: dict[int, str],
    test_class_counts: dict[int, int],
    minimum_objects: int,
) -> dict[str, int]:
    if minimum_objects <= 0:
        raise ValueError("minimum required class support must be positive")
    label_by_name = {
        name: label for label, name in class_name_by_label.items()
    }
    return {
        name: test_class_counts.get(label_by_name.get(name, -1), 0)
        for name in sorted(required_classes)
        if test_class_counts.get(label_by_name.get(name, -1), 0)
        < minimum_objects
    }


def is_priority_mark_class(name: str) -> bool:
    normalized = name.casefold()
    return normalized.startswith(
        (
            "artic",
            "dynamic",
            "fermata",
            "fingering",
            "glissando",
            "grace",
            "ornament",
            "ottava",
            "pedal",
            "rehearsal",
            "strings",
            "textline",
            "tremolo",
            "trill",
            "tuplet",
            "volta",
            "genericarticulation",
            "genericdynamic",
        )
    ) or normalized.endswith("text") or normalized in {
        "arpeggiato",
        "arpeggio",
        "breathmark",
        "caesura",
        "coda",
        "hairpin",
        "keyboardpedalped",
        "keyboardpedalup",
        "ottavabracket",
        "parenthesis",
        "pedal",
        "segno",
        "slur",
        "tie",
    }


def priority_selection_score(
    *,
    overall_map: float,
    per_class_map: dict[str, float],
    class_support: dict[str, int] | None = None,
    minimum_support: int = 1,
) -> tuple[float, float]:
    if minimum_support <= 0:
        raise ValueError("priority selection minimum support must be positive")

    def supported_values(*, priority_only: bool) -> list[float]:
        return [
            float(value)
            for name, value in per_class_map.items()
            if math.isfinite(float(value))
            and float(value) >= 0
            and (not priority_only or is_priority_mark_class(name))
            and (
                class_support is None
                or int(class_support.get(name, 0)) >= minimum_support
            )
        ]

    priority_values = supported_values(priority_only=True)
    priority_map = (
        sum(priority_values) / len(priority_values)
        if priority_values
        else 0.0
    )
    if class_support is None:
        # Compatibility for callers that do not own support evidence. Release
        # training and holdout evaluation always provide the support mapping.
        bounded_overall_map = (
            max(0.0, float(overall_map))
            if math.isfinite(float(overall_map))
            else 0.0
        )
    else:
        overall_values = supported_values(priority_only=False)
        bounded_overall_map = (
            sum(overall_values) / len(overall_values)
            if overall_values
            else 0.0
        )
    return 0.25 * bounded_overall_map + 0.75 * priority_map, priority_map


def support_filtered_macro_map(
    *,
    per_class_map: dict[str, float],
    class_support: dict[str, int],
    minimum_support: int,
) -> tuple[float, list[str]]:
    """Return macro mAP over only statistically supported evaluated classes."""

    if minimum_support <= 0:
        raise ValueError("selection minimum support must be positive")
    supported_classes = sorted(
        name
        for name, value in per_class_map.items()
        if math.isfinite(float(value))
        and float(value) >= 0
        and int(class_support.get(name, 0)) >= minimum_support
    )
    values = [float(per_class_map[name]) for name in supported_classes]
    return (
        sum(values) / len(values) if values else 0.0,
        supported_classes,
    )


def detector_selection_evidence_failures(
    metrics: dict[str, Any],
    *,
    class_support: dict[str, int],
    minimum_support: int,
) -> list[str]:
    """Validate that serialized checkpoint-selection evidence is reproducible."""

    named = metrics.get("map_per_class_named")
    if not isinstance(named, dict) or not named:
        return ["map_per_class_named"]
    try:
        score, priority_map = priority_selection_score(
            overall_map=float(metrics.get("map", -1.0)),
            per_class_map=named,
            class_support=class_support,
            minimum_support=minimum_support,
        )
        filtered_map, supported_classes = support_filtered_macro_map(
            per_class_map=named,
            class_support=class_support,
            minimum_support=minimum_support,
        )
        expected_priority_classes = sorted(
            name
            for name in supported_classes
            if is_priority_mark_class(name)
        )
        numeric_expectations = {
            "selection_score": score,
            "selection_support_filtered_map": filtered_map,
            "priority_mark_map": priority_map,
        }
        failures = [
            name
            for name, expected in numeric_expectations.items()
            if not math.isfinite(float(metrics.get(name, math.nan)))
            or not math.isclose(
                float(metrics[name]),
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ]
        for name in (
            "selection_minimum_class_support",
            "priority_mark_minimum_class_support",
        ):
            if int(metrics.get(name, -1)) != minimum_support:
                failures.append(name)
        if metrics.get("selection_supported_classes") != supported_classes:
            failures.append("selection_supported_classes")
        if (
            metrics.get("priority_mark_supported_classes")
            != expected_priority_classes
        ):
            failures.append("priority_mark_supported_classes")
        return failures
    except (TypeError, ValueError, OverflowError):
        return ["invalid"]


def detector_acceptance_failures(
    metrics: dict[str, Any],
    *,
    minimum_map_50: float | None,
    minimum_map_75: float | None,
    minimum_priority_map: float | None,
    required_class_maps: dict[str, float],
) -> list[str]:
    map_50 = float(metrics.get("map_50", -1.0))
    map_75 = float(metrics.get("map_75", -1.0))
    priority_map = float(metrics.get("priority_mark_map", -1.0))
    named_maps = metrics.get("map_per_class_named")
    named_maps = named_maps if isinstance(named_maps, dict) else {}
    failures: list[str] = []
    for name, value, floor in (
        ("best_map_50", map_50, minimum_map_50),
        ("best_map_75", map_75, minimum_map_75),
        ("best_priority_mark_map", priority_map, minimum_priority_map),
    ):
        if floor is not None and value < floor:
            failures.append(f"{name}={value:.6f}<{floor:.6f}")
    for class_name, floor in sorted(required_class_maps.items()):
        value = float(named_maps.get(class_name, -1.0))
        if value < floor:
            failures.append(
                f"class:{class_name}={value:.6f}<{floor:.6f}"
            )
    return failures


def detector_acceptance_gates_configured(
    *,
    minimum_map_50: float | None,
    minimum_map_75: float | None,
    minimum_priority_map: float | None,
    required_class_maps: dict[str, float],
) -> bool:
    """Return whether an early-stop decision has at least one real accuracy gate."""

    return (
        minimum_map_50 is not None
        or minimum_map_75 is not None
        or minimum_priority_map is not None
        or bool(required_class_maps)
    )


def detector_runtime_final_epoch(
    *,
    planned_epochs: int,
    eval_every: int,
    stop_after_epoch: int | None,
    stop_reason: str | None,
) -> int:
    """Validate an auditable runtime cap and return the final epoch to execute."""

    if planned_epochs <= 0:
        raise ValueError("planned epochs must be positive")
    if eval_every <= 0:
        raise ValueError("eval-every must be positive")
    if stop_after_epoch is None:
        if stop_reason is not None:
            raise ValueError(
                "runtime-stop-reason requires runtime-stop-after-epoch"
            )
        return planned_epochs
    if not 1 <= stop_after_epoch <= planned_epochs:
        raise ValueError(
            "runtime-stop-after-epoch must be within the planned epoch range"
        )
    if stop_after_epoch < planned_epochs and stop_after_epoch % eval_every != 0:
        raise ValueError(
            "runtime-stop-after-epoch must end on an evaluated epoch"
        )
    if (
        stop_after_epoch < planned_epochs
        and not (stop_reason or "").strip()
    ):
        raise ValueError(
            "runtime-stop-reason is required for an early runtime cap"
        )
    return stop_after_epoch


def should_replace_detector_best(
    *,
    current_gate_passed: bool,
    current_selection_score: float,
    best_gate_passed: bool,
    best_selection_score: float,
) -> bool:
    """A gate-passing epoch always outranks an aggregate-only epoch."""

    if current_gate_passed != best_gate_passed:
        return current_gate_passed
    return current_selection_score > best_selection_score


def assert_matching_run_config(
    expected: dict[str, Any], actual: dict[str, Any]
) -> None:
    if expected == actual:
        return
    changed = sorted(
        key
        for key in set(expected) | set(actual)
        if expected.get(key) != actual.get(key)
    )
    raise ValueError(
        "refusing detector resume with changed data or configuration: "
        + ", ".join(changed)
    )


def reconcile_resume_run_config(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    allow_worker_change: bool,
    checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Permit only an explicit, provenance-recorded DataLoader worker change.

    Dataset identity, sampling weights, optimizer hyperparameters and every
    model contract remain immutable. Legacy augmentation uses worker-local RNG,
    so a worker-count transition is recorded as a changed augmentation stream
    rather than misrepresented as a bit-identical resume.
    """

    previous = copy.deepcopy(expected)
    requested = copy.deepcopy(actual)
    transitions = previous.pop("runtime_worker_transitions", [])
    if not isinstance(transitions, list):
        raise ValueError("detector runtime worker transition history is invalid")
    requested.pop("runtime_worker_transitions", None)
    execution_records = previous.pop("runtime_execution_records", [])
    if not isinstance(execution_records, list):
        raise ValueError("detector runtime execution history is invalid")
    requested.pop("runtime_execution_records", None)
    if previous == requested:
        if transitions:
            requested["runtime_worker_transitions"] = transitions
        if execution_records:
            requested["runtime_execution_records"] = execution_records
        return requested

    previous_arguments = previous.get("arguments")
    requested_arguments = requested.get("arguments")
    if (
        not allow_worker_change
        or not isinstance(previous_arguments, dict)
        or not isinstance(requested_arguments, dict)
    ):
        assert_matching_run_config(expected, actual)
    previous_workers = previous_arguments.get("workers")
    requested_workers = requested_arguments.get("workers")
    comparison = copy.deepcopy(previous)
    comparison["arguments"]["workers"] = requested_workers
    if (
        comparison != requested
        or not isinstance(previous_workers, int)
        or not isinstance(requested_workers, int)
        or previous_workers < 0
        or requested_workers < 0
        or previous_workers == requested_workers
        or checkpoint_sha256 is None
        or len(checkpoint_sha256) != 64
        or any(char not in "0123456789abcdef" for char in checkpoint_sha256)
    ):
        assert_matching_run_config(expected, actual)
    transitions.append(
        {
            "field": "workers",
            "from": previous_workers,
            "to": requested_workers,
            "resume_checkpoint_sha256": checkpoint_sha256,
            "augmentation_continuity": "legacy_worker_rng_stream_changed",
        }
    )
    requested["runtime_worker_transitions"] = transitions
    if execution_records:
        requested["runtime_execution_records"] = execution_records
    return requested


def json_ready(value: Any) -> Any:
    """Recursively normalize report values without weakening their content."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): json_ready(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def category_label_name_map(manifest: dict[str, Any]) -> dict[int, str]:
    classes = manifest.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("category manifest has no classes")
    result: dict[int, str] = {}
    for row in classes:
        if not isinstance(row, dict) or "label" not in row or "name" not in row:
            raise ValueError("category manifest has an invalid class row")
        label = int(row["label"])
        name = str(row["name"])
        if label <= 0 or not name:
            raise ValueError("category labels and names must be non-empty")
        if label in result:
            raise ValueError(f"duplicate category label: {label}")
        result[label] = name
    return result


def assert_compatible_category_manifests(
    target: dict[str, Any],
    initialization: dict[str, Any],
) -> None:
    target_map = category_label_name_map(target)
    initialization_map = category_label_name_map(initialization)
    if target_map == initialization_map:
        return
    changed = sorted(
        {
            *(f"label {label}" for label in target_map.keys() - initialization_map),
            *(
                f"label {label}"
                for label in initialization_map.keys() - target_map
            ),
            *(
                f"label {label}"
                for label in target_map.keys() & initialization_map
                if target_map[label] != initialization_map[label]
            ),
        }
    )
    raise ValueError(
        "initial detector categories do not match target categories: "
        + ", ".join(changed)
    )


def normalize_target_boxes(
    rows: list[dict[str, Any]],
    *,
    minimum_visible_fraction: float,
    require_complete_page_geometry: bool = False,
    long_span_minimum_visible_fraction: float | None = None,
    tile_overlap: float = 256.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Clip legacy tile-edge boxes, but fail if too little object is visible."""

    if (
        not math.isfinite(minimum_visible_fraction)
        or not 0 < minimum_visible_fraction <= 1
    ):
        raise ValueError("minimum visible target fraction must be in (0, 1]")
    if long_span_minimum_visible_fraction is None:
        long_span_minimum_visible_fraction = minimum_visible_fraction
    if (
        not math.isfinite(long_span_minimum_visible_fraction)
        or not 0
        < long_span_minimum_visible_fraction
        <= minimum_visible_fraction
    ):
        raise ValueError(
            "long-span minimum visible target fraction must be in "
            "(0, minimum visible fraction]"
        )
    normalized_rows: list[dict[str, Any]] = []
    clipped_by_category: collections.Counter[str] = collections.Counter()
    object_count = 0
    page_objects: dict[
        tuple[str, str, str],
        tuple[str, int, tuple[float, float, float, float]],
    ] = {}
    for row in rows:
        crop = row.get("crop_xyxy")
        if not isinstance(crop, list) or len(crop) != 4:
            raise ValueError("detector row has a malformed crop")
        crop_width = float(crop[2]) - float(crop[0])
        crop_height = float(crop[3]) - float(crop[1])
        if (
            not math.isfinite(crop_width)
            or not math.isfinite(crop_height)
            or crop_width <= 0
            or crop_height <= 0
        ):
            raise ValueError("detector row has a non-positive crop")
        objects = row.get("objects")
        if not isinstance(objects, list):
            raise ValueError("detector row has malformed objects")
        normalized_objects: list[dict[str, Any]] | None = None
        row_source_ids: set[str] = set()
        for object_index, obj in enumerate(objects):
            object_count += 1
            if not isinstance(obj, dict):
                raise ValueError("detector row has a malformed object")
            box = obj.get("box_xyxy")
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError("detector object has a malformed box")
            left, top, right, bottom = (float(value) for value in box)
            if (
                not all(
                    math.isfinite(value)
                    for value in (left, top, right, bottom)
                )
                or right <= left
                or bottom <= top
            ):
                raise ValueError("detector object box is non-finite or degenerate")
            if require_complete_page_geometry:
                source_key = str(row.get("source_key") or "")
                image = str(row.get("image") or "")
                source_id = str(obj.get("source_object_id") or "")
                page_box_raw = obj.get("page_box_xyxy")
                category = str(obj.get("category_id") or "")
                if (
                    not source_key
                    or not image
                    or not source_id
                    or source_id in row_source_ids
                    or obj.get("target_geometry_provenance")
                    != COMPLETE_PAGE_TARGET_PROVENANCE
                    or not isinstance(page_box_raw, list)
                    or len(page_box_raw) != 4
                    or not category
                ):
                    raise ValueError(
                        "detector object lacks unique complete-page geometry"
                    )
                row_source_ids.add(source_id)
                page_box = tuple(float(value) for value in page_box_raw)
                if (
                    not all(math.isfinite(value) for value in page_box)
                    or page_box[2] <= page_box[0]
                    or page_box[3] <= page_box[1]
                ):
                    raise ValueError(
                        "detector object has invalid complete-page box"
                    )
                expected_page = (
                    max(page_box[0], float(crop[0])),
                    max(page_box[1], float(crop[1])),
                    min(page_box[2], float(crop[2])),
                    min(page_box[3], float(crop[3])),
                )
                if (
                    expected_page[2] <= expected_page[0]
                    or expected_page[3] <= expected_page[1]
                ):
                    raise ValueError(
                        "detector complete-page object misses its crop"
                    )
                expected_local = (
                    expected_page[0] - float(crop[0]),
                    expected_page[1] - float(crop[1]),
                    expected_page[2] - float(crop[0]),
                    expected_page[3] - float(crop[1]),
                )
                if max(
                    abs(observed - expected)
                    for observed, expected in zip(
                        (left, top, right, bottom),
                        expected_local,
                        strict=True,
                    )
                ) > 0.01:
                    raise ValueError(
                        "detector local box contradicts complete-page geometry"
                    )
                page_area = (
                    (page_box[2] - page_box[0])
                    * (page_box[3] - page_box[1])
                )
                visible_fraction = (
                    (expected_page[2] - expected_page[0])
                    * (expected_page[3] - expected_page[1])
                    / page_area
                )
                if not target_fragment_is_visible(
                    page_box,
                    tuple(float(value) for value in crop),
                    minimum_fraction=minimum_visible_fraction,
                    long_span_minimum_fraction=(
                        long_span_minimum_visible_fraction
                    ),
                    is_long_span=(
                        category in LONG_SPAN_SEMANTIC_CATEGORIES
                        or category.casefold().endswith("text")
                    ),
                    tile_overlap=tile_overlap,
                ):
                    raise ValueError(
                        "detector complete-page object visible fraction "
                        f"{visible_fraction:.6f} has no valid fragment"
                    )
                page_key = (source_key, image, source_id)
                page_value = (
                    category,
                    int(obj.get("label")),
                    tuple(round(value, 4) for value in page_box),
                )
                previous_page_value = page_objects.get(page_key)
                if (
                    previous_page_value is not None
                    and previous_page_value != page_value
                ):
                    raise ValueError(
                        "detector source object changed complete-page geometry"
                    )
                page_objects[page_key] = page_value
            clipped = (
                max(0.0, left),
                max(0.0, top),
                min(crop_width, right),
                min(crop_height, bottom),
            )
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                raise ValueError("detector object has no visible crop intersection")
            area = (right - left) * (bottom - top)
            visible_area = (
                (clipped[2] - clipped[0]) * (clipped[3] - clipped[1])
            )
            visible_fraction = visible_area / area
            if visible_fraction + 1e-6 < minimum_visible_fraction:
                raise ValueError(
                    "detector object visible fraction "
                    f"{visible_fraction:.6f} is below "
                    f"{minimum_visible_fraction:.6f}"
                )
            changed = any(
                abs(original - bounded) > 1e-3
                for original, bounded in zip(
                    (left, top, right, bottom),
                    clipped,
                    strict=True,
                )
            )
            if changed:
                if normalized_objects is None:
                    # Preserve already-audited objects by reference and
                    # allocate replacements only from the first actual clip.
                    # Current overlap-consistent datasets normally need no
                    # clips, avoiding millions of throwaway dictionaries on
                    # every exact checkpoint resume.
                    normalized_objects = list(objects[:object_index])
                clipped_by_category[
                    str(obj.get("category_id", obj.get("label", "unknown")))
                ] += 1
                normalized_objects.append(
                    {
                        **obj,
                        "box_xyxy": [round(value, 4) for value in clipped],
                    }
                )
            elif normalized_objects is not None:
                normalized_objects.append(obj)
        normalized_rows.append(
            {**row, "objects": normalized_objects}
            if normalized_objects is not None
            else row
        )
    audit = {
        "objects": object_count,
        "clipped_objects": sum(clipped_by_category.values()),
        "clipped_by_category": dict(sorted(clipped_by_category.items())),
        "minimum_visible_fraction": minimum_visible_fraction,
    }
    if require_complete_page_geometry:
        audit.update(
            {
                "complete_page_geometry_required": True,
                "target_geometry_provenance": (
                    COMPLETE_PAGE_TARGET_PROVENANCE
                ),
                "long_span_minimum_visible_fraction": (
                    long_span_minimum_visible_fraction
                ),
                "unique_source_objects": len(page_objects),
                "tile_overlap": tile_overlap,
                "oversized_fragment_visibility_version": (
                    OVERSIZED_FRAGMENT_VISIBILITY_VERSION
                ),
            }
        )
    return normalized_rows, audit


def reusable_zero_clip_target_box_audit(
    binding: Any,
    *,
    jsonl_sha256: str,
    row_count: int,
    minimum_visible_fraction: float,
    require_complete_page_geometry: bool = False,
    long_span_minimum_visible_fraction: float | None = None,
    tile_overlap: float = 256.0,
) -> dict[str, Any] | None:
    """Return a hash-bound prior audit only when no row mutation was needed."""

    if not isinstance(binding, dict):
        return None
    audit = binding.get("audit")
    expected_contract = (
        "scorescan-zero-clip-complete-page-target-box-audit@3"
        if require_complete_page_geometry
        else "scorescan-zero-clip-target-box-audit@1"
    )
    if long_span_minimum_visible_fraction is None:
        long_span_minimum_visible_fraction = minimum_visible_fraction
    if (
        binding.get("contract") != expected_contract
        or binding.get("jsonl_sha256") != jsonl_sha256
        or int(binding.get("rows", -1)) != row_count
        or not isinstance(audit, dict)
        or int(audit.get("objects", -1)) < 0
        or int(audit.get("clipped_objects", -1)) != 0
        or audit.get("clipped_by_category") != {}
        or not math.isclose(
            float(audit.get("minimum_visible_fraction", -1)),
            minimum_visible_fraction,
            rel_tol=0,
            abs_tol=1e-12,
        )
        or (
            require_complete_page_geometry
            and (
                audit.get("complete_page_geometry_required") is not True
                or audit.get("target_geometry_provenance")
                != COMPLETE_PAGE_TARGET_PROVENANCE
                or int(audit.get("unique_source_objects", 0)) <= 0
                or audit.get("oversized_fragment_visibility_version")
                != OVERSIZED_FRAGMENT_VISIBILITY_VERSION
                or not math.isclose(
                    float(audit.get("tile_overlap", -1)),
                    tile_overlap,
                    rel_tol=0,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    float(
                        audit.get(
                            "long_span_minimum_visible_fraction",
                            -1,
                        )
                    ),
                    long_span_minimum_visible_fraction,
                    rel_tol=0,
                    abs_tol=1e-12,
                )
            )
        )
    ):
        return None
    return copy.deepcopy(audit)


def bind_zero_clip_target_box_audit(
    audit: dict[str, Any],
    *,
    jsonl_sha256: str,
    row_count: int,
) -> dict[str, Any] | None:
    """Bind a reusable audit to immutable JSONL bytes, or decline mutations."""

    if int(audit.get("clipped_objects", -1)) != 0:
        return None
    return {
        "contract": (
            "scorescan-zero-clip-complete-page-target-box-audit@3"
            if audit.get("complete_page_geometry_required") is True
            else "scorescan-zero-clip-target-box-audit@1"
        ),
        "jsonl_sha256": jsonl_sha256,
        "rows": row_count,
        "audit": copy.deepcopy(audit),
    }


def assert_training_dataset_manifest(manifest: dict[str, Any]) -> None:
    role = str(manifest.get("role") or "").casefold()
    name = str(manifest.get("name") or "").casefold()
    legacy_roleless_deepscores = (
        not role
        and name.startswith(LEGACY_ROLELESS_DEEPSCORES_NAME_PREFIX)
    )
    if (
        role not in DETECTOR_TRAINING_DATASET_ROLES
        and not legacy_roleless_deepscores
    ):
        raise ValueError(f"refusing non-training detector dataset role: {role}")
    for field in ("source_split_overlap", "reserved_holdout_overlap"):
        if field in manifest and int(manifest[field]) != 0:
            raise ValueError(f"detector dataset has nonzero {field}")
    for split in ("train", "test"):
        split_report = manifest.get(split)
        if not isinstance(split_report, dict):
            raise ValueError(f"detector dataset has no {split} split")
        if int(split_report.get("tiles", 0)) <= 0:
            raise ValueError(f"detector dataset {split} split is empty")
    if manifest.get("class_subset_contract") != RELATION_SUBSET_CONTRACT_VERSION:
        return
    class_subset = manifest.get("class_subset")
    if not isinstance(class_subset, list) or not class_subset:
        raise ValueError("relation detector dataset has no class subset")
    minimum_sources = int(manifest.get("minimum_test_sources_per_class", 0))
    minimum_unique_objects = int(
        manifest.get("minimum_test_unique_objects_per_class", 0)
    )
    if minimum_sources <= 0 or minimum_unique_objects <= 0:
        raise ValueError("relation detector independent support gates are invalid")
    test_quality = manifest["test"].get("class_quality")
    if not isinstance(test_quality, dict):
        raise ValueError("relation detector test split has no class-quality audit")
    failures: list[str] = []
    for category in class_subset:
        audit = test_quality.get(category)
        if not isinstance(audit, dict):
            failures.append(f"{category}:missing_audit")
            continue
        sources = int(audit.get("sources", 0))
        unique_objects = int(audit.get("unique_objects", 0))
        if sources < minimum_sources:
            failures.append(f"{category}:sources={sources}<{minimum_sources}")
        if unique_objects < minimum_unique_objects:
            failures.append(
                f"{category}:unique_objects={unique_objects}"
                f"<{minimum_unique_objects}"
            )
    if failures:
        raise ValueError(
            "relation detector test support is not independent: "
            + "; ".join(failures)
        )


def assert_complete_page_target_dataset(
    prepared_dir: Path,
    manifest: dict[str, Any],
) -> None:
    """Refuse semantic supervision derived from clipped owner-tile boxes.

    Page-spanning symbols were historically clipped to one owner tile before
    overlap expansion.  That destroys the original SVG geometry and can train
    plausible-looking but systematically shortened or displaced slurs,
    hairpins, beams, text, and other marks.  Production semantic runs opt into
    this assertion so every primary, evaluation, and replay dataset must carry
    the immutable complete-page geometry contract.
    """

    preparation_path = prepared_dir / "prepare-report.json"
    if not preparation_path.is_file():
        raise FileNotFoundError(
            f"complete-page semantic preparation is missing: {preparation_path}"
        )
    preparation = json.loads(
        preparation_path.read_text(encoding="utf-8")
    )
    if not isinstance(preparation, dict):
        raise ValueError("complete-page semantic preparation must be an object")
    if (
        manifest.get("target_assignment_version")
        != COMPLETE_PAGE_TARGET_ASSIGNMENT_VERSION
    ):
        raise ValueError(
            "semantic manifest does not use complete-page target assignment"
        )
    if (
        preparation.get("transformation_version")
        != COMPLETE_PAGE_TARGET_ASSIGNMENT_VERSION
    ):
        raise ValueError(
            "semantic preparation does not use complete-page target assignment"
        )
    for name, payload in (
        ("manifest", manifest),
        ("preparation", preparation),
    ):
        if (
            payload.get("target_geometry_provenance")
            != COMPLETE_PAGE_TARGET_PROVENANCE
        ):
            raise ValueError(
                f"semantic {name} does not preserve complete-page geometry"
            )
        if (
            payload.get("oversized_fragment_visibility_version")
            != OVERSIZED_FRAGMENT_VISIBILITY_VERSION
        ):
            raise ValueError(
                f"semantic {name} has stale oversized-fragment visibility"
            )
    tile_size = int(
        preparation.get("tile_size", manifest.get("tile_size", 0))
    )
    tile_overlap = float(
        preparation.get("overlap", manifest.get("overlap", 0))
    )
    if tile_size <= 0 or tile_overlap <= 0 or tile_overlap >= tile_size:
        raise ValueError("complete-page semantic tile geometry is invalid")
    minimum_fraction = float(
        preparation.get(
            "minimum_object_fraction",
            manifest.get("minimum_object_fraction", -1),
        )
    )
    long_span_minimum_fraction = float(
        preparation.get(
            "long_span_minimum_object_fraction",
            manifest.get("long_span_minimum_object_fraction", -1),
        )
    )
    if not (
        math.isfinite(minimum_fraction)
        and math.isfinite(long_span_minimum_fraction)
        and 0
        < long_span_minimum_fraction
        <= minimum_fraction
        <= 1
    ):
        raise ValueError("complete-page semantic visibility floors are invalid")
    dropped = preparation.get("dropped_object_counts", {})
    if not isinstance(dropped, dict):
        raise ValueError("semantic dropped-object counts are invalid")
    dropped_spanning = {
        str(category): int(count)
        for category, count in dropped.items()
        if int(count) > 0
        and (
            str(category) in LONG_SPAN_SEMANTIC_CATEGORIES
            or str(category).casefold().endswith("text")
        )
    }
    if dropped_spanning:
        details = ", ".join(
            f"{category}={count}"
            for category, count in sorted(dropped_spanning.items())
        )
        raise ValueError(
            f"complete-page semantic source dropped spanning targets: {details}"
        )


def build_detector_model(
    *,
    number_of_classes: int,
    score_threshold: float,
    detections_per_tile: int,
    pretrained_backbone: bool,
    class_name_by_label: dict[int, str] | None = None,
    class_aware_matcher: bool = False,
) -> Any:
    """Build the one canonical RetinaNet architecture used by train and eval."""

    import torch
    from torch import nn
    from torchvision.models import ResNet50_Weights
    from torchvision.models.detection import retinanet_resnet50_fpn_v2
    from torchvision.models.detection._utils import Matcher
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.retinanet import RetinaNetHead
    from torchvision.ops import boxes as box_ops

    class CategoryAwareMatcher(Matcher):
        """Use stricter assignment only for classes harmed by dense positives."""

        def __init__(self, strict_labels: frozenset[int]) -> None:
            super().__init__(
                DETECTOR_FOREGROUND_IOU_THRESHOLD,
                DETECTOR_BACKGROUND_IOU_THRESHOLD,
                allow_low_quality_matches=True,
            )
            self.strict_labels = strict_labels

        def __call__(
            self,
            match_quality_matrix: Any,
            ground_truth_labels: Any | None = None,
        ) -> Any:
            if ground_truth_labels is None or not self.strict_labels:
                return super().__call__(match_quality_matrix)
            if match_quality_matrix.numel() == 0:
                return super().__call__(match_quality_matrix)
            if int(ground_truth_labels.numel()) != int(
                match_quality_matrix.shape[0]
            ):
                raise ValueError(
                    "ground-truth labels do not match quality-matrix rows"
                )
            matched_values, matches = match_quality_matrix.max(dim=0)
            all_matches = matches.clone()
            matched_labels = ground_truth_labels[matches]
            strict = matched_labels == -1
            for label in self.strict_labels:
                strict |= matched_labels == label
            foreground_thresholds = matched_values.new_full(
                matched_values.shape,
                DETECTOR_FOREGROUND_IOU_THRESHOLD,
            )
            background_thresholds = matched_values.new_full(
                matched_values.shape,
                DETECTOR_BACKGROUND_IOU_THRESHOLD,
            )
            foreground_thresholds[strict] = (
                DETECTOR_STRICT_FOREGROUND_IOU_THRESHOLD
            )
            background_thresholds[strict] = (
                DETECTOR_STRICT_BACKGROUND_IOU_THRESHOLD
            )
            below = matched_values < background_thresholds
            between = (
                (matched_values >= background_thresholds)
                & (matched_values < foreground_thresholds)
            )
            matches[below] = self.BELOW_LOW_THRESHOLD
            matches[between] = self.BETWEEN_THRESHOLDS
            self.set_low_quality_matches_(
                matches,
                all_matches,
                match_quality_matrix,
            )
            return matches

    anchor_generator = AnchorGenerator(
        sizes=((8, 12), (16, 24), (32, 48), (64, 96), (128, 192)),
        aspect_ratios=((0.1, 0.3, 1.0, 3.0, 10.0),) * 5,
    )
    model = retinanet_resnet50_fpn_v2(
        weights=None,
        weights_backbone=(
            ResNet50_Weights.DEFAULT if pretrained_backbone else None
        ),
        num_classes=number_of_classes,
        trainable_backbone_layers=3,
        score_thresh=score_threshold,
        nms_thresh=DETECTOR_NMS_IOU,
        detections_per_img=detections_per_tile,
        min_size=1024,
        max_size=1024,
    )
    model.anchor_generator = anchor_generator
    model.head = RetinaNetHead(
        model.backbone.out_channels,
        anchor_generator.num_anchors_per_location()[0],
        number_of_classes,
        norm_layer=partial(nn.GroupNorm, 32),
    )
    def category_aware_compute_loss(
        detector: Any,
        targets: list[dict[str, Any]],
        head_outputs: dict[str, Any],
        anchors: list[Any],
    ) -> dict[str, Any]:
        matched_indices = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image["boxes"].numel() == 0:
                matched_indices.append(
                    torch.full(
                        (anchors_per_image.size(0),),
                        -1,
                        dtype=torch.int64,
                        device=anchors_per_image.device,
                    )
                )
                continue
            match_quality_matrix = box_ops.box_iou(
                targets_per_image["boxes"],
                anchors_per_image,
            )
            matched_indices.append(
                detector.proposal_matcher(
                    match_quality_matrix,
                    targets_per_image["labels"],
                )
            )
        return detector.head.compute_loss(
            targets,
            head_outputs,
            anchors,
            matched_indices,
        )

    if class_aware_matcher:
        strict_labels = frozenset(
            int(label)
            for label, name in (class_name_by_label or {}).items()
            if name in DETECTOR_STRICT_MATCH_CLASSES
        )
        unknown_strict_classes = DETECTOR_STRICT_MATCH_CLASSES - set(
            (class_name_by_label or {}).values()
        )
        if class_name_by_label is not None and unknown_strict_classes:
            raise ValueError(
                "strict matcher classes are absent from category manifest: "
                + ", ".join(sorted(unknown_strict_classes))
            )
        model.proposal_matcher = CategoryAwareMatcher(strict_labels)
        model.compute_loss = types.MethodType(
            category_aware_compute_loss,
            model,
        )
    else:
        model.proposal_matcher = Matcher(
            DETECTOR_FOREGROUND_IOU_THRESHOLD,
            DETECTOR_BACKGROUND_IOU_THRESHOLD,
            allow_low_quality_matches=True,
        )
    model.head.regression_head._loss_type = "giou"
    return model


@contextmanager
def evaluation_detection_limit(model: Any, maximum: int = 100):
    """Temporarily cap post-NMS output to the exact COCO evaluation limit.

    Deployment keeps a wider proposal cap for dense pages.  COCO mAP consumes
    only the first 100 score-sorted detections, so producing and transferring
    another 200 boxes per tile during a 10k+ tile validation pass is pure work.
    The cap is restored even when evaluation raises.
    """

    if maximum <= 0:
        raise ValueError("evaluation detection limit must be positive")
    original = int(model.detections_per_img)
    model.detections_per_img = min(original, maximum)
    try:
        yield model.detections_per_img
    finally:
        model.detections_per_img = original


def parse_required_class_maps(values: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for value in values:
        name, separator, raw_floor = value.partition("=")
        name = name.strip()
        if not separator or not name or name in result:
            raise ValueError(f"invalid or duplicate required class mAP: {value!r}")
        try:
            floor = float(raw_floor)
        except ValueError as exc:
            raise ValueError(f"invalid required class mAP: {value!r}") from exc
        if not 0 <= floor <= 1 or not math.isfinite(floor):
            raise ValueError(f"required class mAP must be in [0, 1]: {value!r}")
        result[name] = floor
    return result


def synthetic_training_evidence(
    manifest: dict[str, Any],
    preparation: dict[str, Any] | None,
) -> bool:
    values = [
        manifest.get("role"),
        manifest.get("name"),
        (preparation or {}).get("purpose"),
    ]
    text = " ".join(str(value or "") for value in values).casefold()
    return "synthetic" in text and "semantic" in text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument(
        "--image-cache-dir",
        type=Path,
        help=(
            "runtime-only content-addressed image cache, preferably on the "
            "native training filesystem; every source and cache blob is hashed"
        ),
    )
    parser.add_argument(
        "--populate-image-cache",
        action="store_true",
        help=(
            "atomically populate missing cache blobs from the immutable image "
            "roots before training; requires --image-cache-dir"
        ),
    )
    parser.add_argument(
        "--evaluation-prepared-dir",
        type=Path,
        help=(
            "compatible disjoint test manifest used for model selection while "
            "the primary prepared directory remains the training source"
        ),
    )
    parser.add_argument(
        "--evaluation-images-dir",
        type=Path,
        help="image root paired with --evaluation-prepared-dir",
    )
    parser.add_argument(
        "--require-complete-page-targets",
        action="store_true",
        help=(
            "fail closed unless primary, evaluation, and replay semantic "
            "datasets preserve immutable complete-page object geometry"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--microbatch-size",
        type=int,
        help=(
            "runtime-only maximum images per model forward pass; smaller "
            "microbatches retain the logical batch mean and optimizer schedule"
        ),
    )
    parser.add_argument(
        "--adaptive-full-batch-object-limit",
        type=int,
        help=(
            "runtime-only sparse-batch fast path: use the complete logical "
            "batch in one forward pass only when its combined ground-truth "
            "object count does not exceed this positive limit; otherwise use "
            "--microbatch-size"
        ),
    )
    parser.add_argument(
        "--adaptive-full-batch-min-free-mib",
        type=int,
        help=(
            "runtime-only second gate for the sparse full-batch fast path; "
            "requires this many MiB of CUDA free plus PyTorch reclaimable "
            "reserved memory and requires --adaptive-full-batch-object-limit"
        ),
    )
    parser.add_argument(
        "--evaluation-batch-size",
        type=int,
        default=8,
        help=(
            "larger inference-only batch; independent from the memory-bound "
            "training batch"
        ),
    )
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "auto"),
        default="cuda",
        help=(
            "training device; the default remains explicit CUDA so an unavailable "
            "GPU cannot silently turn a production run into a long CPU job"
        ),
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help=(
            "PyTorch intra-op threads in CPU mode; zero keeps the runtime default. "
            "This is a runtime resource limit and does not alter resume compatibility."
        ),
    )
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=1000,
        help=(
            "write an optimizer-boundary recovery checkpoint during long epochs; "
            "zero disables in-epoch checkpoints"
        ),
    )
    parser.add_argument(
        "--resumable-augmentation-v3",
        action="store_true",
        help=(
            "derive sampling and augmentation randomness from epoch/occurrence "
            "coordinates so an in-epoch resume can start at the next batch "
            "without decoding every completed sample"
        ),
    )
    parser.add_argument(
        "--class-aware-matcher-ablation",
        action="store_true",
        help=(
            "development-only matcher experiment; not the release default"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--max-train-tiles", type=int)
    parser.add_argument("--max-test-tiles", type=int)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--detections-per-tile", type=int, default=300)
    parser.add_argument("--rare-class-sampling-power", type=float, default=0.35)
    parser.add_argument("--rare-class-max-repeat", type=float, default=4.0)
    parser.add_argument(
        "--replay-prepared-dir",
        type=Path,
        help=(
            "compatible synthetic training dataset used only for bounded "
            "anti-forgetting replay"
        ),
    )
    parser.add_argument(
        "--replay-images-dir",
        type=Path,
        help="image root paired with --replay-prepared-dir",
    )
    parser.add_argument(
        "--replay-fraction",
        type=float,
        default=0.0,
        help="probability mass reserved for replay samples in each epoch",
    )
    parser.add_argument("--replay-max-train-tiles", type=int)
    parser.add_argument(
        "--replay-rare-class-max-repeat",
        type=float,
        default=20.0,
        help=(
            "within-replay rare-class repeat cap; total replay probability "
            "remains bounded by --replay-fraction"
        ),
    )
    parser.add_argument(
        "--initial-model",
        type=Path,
        help="strictly compatible model.best.pt used only for a new run",
    )
    parser.add_argument(
        "--initial-categories",
        type=Path,
        help="categories.json paired with --initial-model",
    )
    parser.add_argument(
        "--initial-backbone-model",
        type=Path,
        help=(
            "detector state with different categories; loads only shape-compatible "
            "feature/regression weights for a new run"
        ),
    )
    parser.add_argument("--minimum-best-map-50", type=float)
    parser.add_argument("--minimum-best-map-75", type=float)
    parser.add_argument("--minimum-best-priority-map", type=float)
    parser.add_argument(
        "--minimum-required-class-test-objects",
        type=int,
        default=1,
        help=(
            "minimum independent test objects required for every "
            "--required-class-map"
        ),
    )
    parser.add_argument(
        "--required-class-map",
        action="append",
        default=[],
        metavar="NAME=FLOOR",
        help="repeatable final-run acceptance floor for a named class COCO mAP",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an incomplete, exactly configuration-matched output directory",
    )
    parser.add_argument(
        "--allow-resume-worker-change",
        action="store_true",
        help=(
            "allow only the DataLoader worker count to change during resume; "
            "records the checkpoint hash and legacy augmentation-stream change"
        ),
    )
    parser.add_argument(
        "--stop-when-accepted",
        action="store_true",
        help=(
            "stop at the first evaluated epoch that passes every configured "
            "accuracy gate"
        ),
    )
    parser.add_argument(
        "--external-acceptance-pending",
        action="store_true",
        help=(
            "retain the exact recovery checkpoint after the in-domain gate "
            "because an independent external holdout still controls final "
            "acceptance"
        ),
    )
    parser.add_argument(
        "--runtime-stop-after-epoch",
        type=int,
        help=(
            "runtime-only efficiency cap: finish and evaluate this epoch, then "
            "finalize the best checkpoint without changing the immutable planned "
            "training configuration"
        ),
    )
    parser.add_argument(
        "--runtime-stop-reason",
        help=(
            "auditable reason for --runtime-stop-after-epoch; required when the "
            "runtime cap is earlier than the planned epoch count"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    removed_empty_resume_directory = remove_empty_detector_resume_directory(
        args.output_dir,
        resume=bool(args.resume),
    )
    if removed_empty_resume_directory:
        print(
            json.dumps(
                {
                    "state": "removed_empty_legacy_detector_resume_directory",
                    "output_dir": str(args.output_dir),
                },
                sort_keys=True,
            )
        )
    checkpoint_path = args.output_dir / "checkpoint.last.pt"
    partial_metrics_path = args.output_dir / "metrics.partial.json"
    run_config_path = args.output_dir / "run_config.json"
    completed_report_path = args.output_dir / "training_report.json"
    reopened_runtime_run = reopen_runtime_truncated_detector_run(
        args.output_dir,
        resume=bool(args.resume),
        planned_epochs=args.epochs,
        runtime_stop_after_epoch=args.runtime_stop_after_epoch,
    )
    if reopened_runtime_run is not None:
        print(json.dumps(reopened_runtime_run, sort_keys=True), flush=True)
    resuming = False
    restarting_uncheckpointed = False
    stored_run_config_for_resume: dict[str, Any] | None = None
    if args.output_dir.exists():
        if completed_report_path.is_file():
            raise FileExistsError(
                f"refusing to resume completed detector run: {args.output_dir}"
            )
        if not args.resume:
            raise FileExistsError(args.output_dir)
        checkpoint_exists = checkpoint_path.is_file()
        partial_metrics_exists = partial_metrics_path.is_file()
        if not run_config_path.is_file() or checkpoint_exists != partial_metrics_exists:
            raise RuntimeError(
                "incomplete detector output has inconsistent recovery files: "
                f"{args.output_dir}"
            )
        if checkpoint_exists:
            resuming = True
        else:
            # A failure between writing the immutable run configuration and the
            # first checkpoint is safe to reconstruct because all seeds and
            # initialization inputs are part of that configuration.
            restarting_uncheckpointed = True
        stored_run_config_for_resume = json.loads(
            run_config_path.read_text(encoding="utf-8")
        )
    if (
        args.accumulate <= 0
        or args.epochs <= 0
        or args.batch_size <= 0
        or args.evaluation_batch_size <= 0
    ):
        raise ValueError(
            "epochs, batch sizes, and accumulate must be positive"
        )
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    if (
        args.microbatch_size is not None
        and (
            args.microbatch_size <= 0
            or args.microbatch_size > args.batch_size
        )
    ):
        raise ValueError(
            "microbatch-size must be positive and cannot exceed batch-size"
        )
    if (
        args.adaptive_full_batch_object_limit is not None
        and args.adaptive_full_batch_object_limit <= 0
    ):
        raise ValueError(
            "adaptive-full-batch-object-limit must be positive"
        )
    if (
        args.adaptive_full_batch_min_free_mib is not None
        and args.adaptive_full_batch_min_free_mib <= 0
    ):
        raise ValueError(
            "adaptive-full-batch-min-free-mib must be positive"
        )
    if (
        args.adaptive_full_batch_min_free_mib is not None
        and args.adaptive_full_batch_object_limit is None
    ):
        raise ValueError(
            "adaptive-full-batch-min-free-mib requires "
            "adaptive-full-batch-object-limit"
        )
    if args.checkpoint_every_steps < 0:
        raise ValueError("checkpoint-every-steps cannot be negative")
    runtime_final_epoch = detector_runtime_final_epoch(
        planned_epochs=args.epochs,
        eval_every=args.eval_every,
        stop_after_epoch=args.runtime_stop_after_epoch,
        stop_reason=args.runtime_stop_reason,
    )
    if args.cpu_threads < 0:
        raise ValueError("cpu-threads cannot be negative")
    if args.populate_image_cache and args.image_cache_dir is None:
        raise ValueError(
            "populate-image-cache requires image-cache-dir"
        )
    if args.minimum_required_class_test_objects <= 0:
        raise ValueError(
            "minimum-required-class-test-objects must be positive"
        )
    if not 0 <= args.replay_fraction < 1:
        raise ValueError("replay-fraction must be in [0, 1)")
    replay_paths_supplied = (
        args.replay_prepared_dir is not None
        or args.replay_images_dir is not None
    )
    if replay_paths_supplied != (args.replay_fraction > 0):
        raise ValueError(
            "replay directories and a positive replay-fraction are required together"
        )
    evaluation_paths_supplied = (
        args.evaluation_prepared_dir is not None
        or args.evaluation_images_dir is not None
    )
    if evaluation_paths_supplied and (
        args.evaluation_prepared_dir is None
        or args.evaluation_images_dir is None
    ):
        raise ValueError(
            "evaluation-prepared-dir and evaluation-images-dir "
            "must be supplied together"
        )
    if (
        args.replay_max_train_tiles is not None
        and args.replay_max_train_tiles <= 0
    ):
        raise ValueError("replay-max-train-tiles must be positive")
    if args.replay_rare_class_max_repeat < 1:
        raise ValueError("replay-rare-class-max-repeat must be at least 1")
    for name, value in (
        ("minimum-best-map-50", args.minimum_best_map_50),
        ("minimum-best-map-75", args.minimum_best_map_75),
        ("minimum-best-priority-map", args.minimum_best_priority_map),
    ):
        if value is not None and (
            not math.isfinite(value) or not 0 <= value <= 1
        ):
            raise ValueError(f"{name} must be in [0, 1]")
    required_class_maps = parse_required_class_maps(args.required_class_map)
    if args.stop_when_accepted and not detector_acceptance_gates_configured(
        minimum_map_50=args.minimum_best_map_50,
        minimum_map_75=args.minimum_best_map_75,
        minimum_priority_map=args.minimum_best_priority_map,
        required_class_maps=required_class_maps,
    ):
        raise ValueError(
            "stop-when-accepted requires at least one explicit accuracy gate"
        )
    if (args.initial_model is None) != (args.initial_categories is None):
        raise ValueError(
            "initial-model and initial-categories must be supplied together"
        )
    if args.initial_model is not None and args.initial_backbone_model is not None:
        raise ValueError(
            "strict initial-model and initial-backbone-model are mutually exclusive"
        )
    for directory in (
        args.prepared_dir,
        args.images_dir,
        args.evaluation_prepared_dir,
        args.evaluation_images_dir,
        args.replay_prepared_dir,
        args.replay_images_dir,
    ):
        if directory is None:
            continue
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    for initial_path in (
        args.initial_model,
        args.initial_categories,
        args.initial_backbone_model,
    ):
        if initial_path is not None and not initial_path.is_file():
            raise FileNotFoundError(initial_path)
    prepared_manifest = json.loads(
        (args.prepared_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert_training_dataset_manifest(prepared_manifest)
    if args.require_complete_page_targets:
        assert_complete_page_target_dataset(
            args.prepared_dir,
            prepared_manifest,
        )

    import torch
    import torchvision
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from torchvision.transforms import functional as vision_f

    device_name = resolve_detector_device(
        args.device,
        cuda_available=torch.cuda.is_available(),
    )
    device = torch.device(device_name)
    use_cuda = device.type == "cuda"
    if not use_cuda and args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(1)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
        # Every detector tile is exactly 1024x1024.  cuDNN's one-time algorithm
        # selection is therefore reusable across the entire multi-epoch run.
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    category_manifest = json.loads(
        (args.prepared_dir / "categories.json").read_text(encoding="utf-8")
    )
    evaluation_manifest: dict[str, Any] | None = None
    evaluation_category_manifest: dict[str, Any] | None = None
    if args.evaluation_prepared_dir is not None:
        evaluation_manifest = json.loads(
            (args.evaluation_prepared_dir / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert_training_dataset_manifest(evaluation_manifest)
        if args.require_complete_page_targets:
            assert_complete_page_target_dataset(
                args.evaluation_prepared_dir,
                evaluation_manifest,
            )
        evaluation_category_manifest = json.loads(
            (args.evaluation_prepared_dir / "categories.json").read_text(
                encoding="utf-8"
            )
        )
        assert_compatible_category_manifests(
            category_manifest,
            evaluation_category_manifest,
        )
    replay_manifest: dict[str, Any] | None = None
    replay_category_manifest: dict[str, Any] | None = None
    if args.replay_prepared_dir is not None:
        replay_manifest = json.loads(
            (args.replay_prepared_dir / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert_training_dataset_manifest(replay_manifest)
        if args.require_complete_page_targets:
            assert_complete_page_target_dataset(
                args.replay_prepared_dir,
                replay_manifest,
            )
        replay_preparation_path = (
            args.replay_prepared_dir / "prepare-report.json"
        )
        replay_preparation = (
            json.loads(replay_preparation_path.read_text(encoding="utf-8"))
            if replay_preparation_path.is_file()
            else None
        )
        if not synthetic_training_evidence(
            replay_manifest,
            replay_preparation,
        ):
            raise ValueError(
                "anti-forgetting replay must use a synthetic training role"
            )
        replay_category_manifest = json.loads(
            (args.replay_prepared_dir / "categories.json").read_text(
                encoding="utf-8"
            )
        )
        assert_compatible_category_manifests(
            category_manifest,
            replay_category_manifest,
        )
    initialization: dict[str, str] | None = None
    if args.initial_model is not None and args.initial_categories is not None:
        initial_category_manifest = json.loads(
            args.initial_categories.read_text(encoding="utf-8")
        )
        assert_compatible_category_manifests(
            category_manifest,
            initial_category_manifest,
        )
        initialization = {
            "mode": "strict_full_detector",
            "model_sha256": sha256_file(args.initial_model),
            "categories_sha256": sha256_file(args.initial_categories),
        }
    elif args.initial_backbone_model is not None:
        initialization = {
            "mode": "shape_compatible_except_class_logits",
            "model_sha256": sha256_file(args.initial_backbone_model),
        }
    classes = category_manifest["classes"]
    number_of_classes = max(int(row["label"]) for row in classes) + 1
    class_name_by_label = {int(row["label"]): row["name"] for row in classes}
    train_jsonl_path = args.prepared_dir / "train.jsonl"
    train_rows = stable_subset(
        load_jsonl(train_jsonl_path),
        args.max_train_tiles,
        args.seed,
    )
    evaluation_prepared_dir = (
        args.evaluation_prepared_dir or args.prepared_dir
    )
    evaluation_images_dir = (
        args.evaluation_images_dir or args.images_dir
    )
    test_jsonl_path = evaluation_prepared_dir / "test.jsonl"
    test_rows = stable_subset(
        load_jsonl(test_jsonl_path),
        args.max_test_tiles,
        args.seed + 1,
    )
    replay_rows: list[dict[str, Any]] = []
    replay_jsonl_path: Path | None = None
    if args.replay_prepared_dir is not None:
        replay_jsonl_path = args.replay_prepared_dir / "train.jsonl"
        replay_rows = stable_subset(
            load_jsonl(replay_jsonl_path),
            args.replay_max_train_tiles,
            args.seed + 2,
        )
    train_jsonl_sha256 = sha256_file(train_jsonl_path)
    test_jsonl_sha256 = sha256_file(test_jsonl_path)
    replay_jsonl_sha256 = (
        sha256_file(replay_jsonl_path)
        if replay_jsonl_path is not None
        else None
    )
    minimum_visible_fraction = 1.0
    long_span_minimum_visible_fraction = 1.0
    target_tile_overlap = 256.0
    prepare_report_path = args.prepared_dir / "prepare-report.json"
    if prepare_report_path.is_file():
        prepare_report = json.loads(
            prepare_report_path.read_text(encoding="utf-8")
        )
        minimum_visible_fraction = float(
            prepare_report.get("minimum_object_fraction", 0.8)
        )
        long_span_minimum_visible_fraction = float(
            prepare_report.get(
                "long_span_minimum_object_fraction",
                minimum_visible_fraction,
            )
        )
        target_tile_overlap = float(prepare_report.get("overlap", 256.0))
    test_minimum_visible_fraction = 1.0
    test_long_span_minimum_visible_fraction = 1.0
    test_target_tile_overlap = 256.0
    evaluation_prepare_report_path = (
        evaluation_prepared_dir / "prepare-report.json"
    )
    if evaluation_prepare_report_path.is_file():
        evaluation_prepare_report = json.loads(
            evaluation_prepare_report_path.read_text(encoding="utf-8")
        )
        test_minimum_visible_fraction = float(
            evaluation_prepare_report.get("minimum_object_fraction", 0.8)
        )
        test_long_span_minimum_visible_fraction = float(
            evaluation_prepare_report.get(
                "long_span_minimum_object_fraction",
                test_minimum_visible_fraction,
            )
        )
        test_target_tile_overlap = float(
            evaluation_prepare_report.get("overlap", 256.0)
        )
    stored_target_box_audits = (
        stored_run_config_for_resume.get("target_box_audits", {})
        if stored_run_config_for_resume is not None
        else {}
    )
    if not isinstance(stored_target_box_audits, dict):
        stored_target_box_audits = {}
    train_box_audit = reusable_zero_clip_target_box_audit(
        stored_target_box_audits.get("train"),
        jsonl_sha256=train_jsonl_sha256,
        row_count=len(train_rows),
        minimum_visible_fraction=minimum_visible_fraction,
        require_complete_page_geometry=args.require_complete_page_targets,
        long_span_minimum_visible_fraction=(
            long_span_minimum_visible_fraction
        ),
        tile_overlap=target_tile_overlap,
    )
    if train_box_audit is None:
        train_rows, train_box_audit = normalize_target_boxes(
            train_rows,
            minimum_visible_fraction=minimum_visible_fraction,
            require_complete_page_geometry=args.require_complete_page_targets,
            long_span_minimum_visible_fraction=(
                long_span_minimum_visible_fraction
            ),
            tile_overlap=target_tile_overlap,
        )
    test_box_audit = reusable_zero_clip_target_box_audit(
        stored_target_box_audits.get("test"),
        jsonl_sha256=test_jsonl_sha256,
        row_count=len(test_rows),
        minimum_visible_fraction=test_minimum_visible_fraction,
        require_complete_page_geometry=args.require_complete_page_targets,
        long_span_minimum_visible_fraction=(
            test_long_span_minimum_visible_fraction
        ),
        tile_overlap=test_target_tile_overlap,
    )
    if test_box_audit is None:
        test_rows, test_box_audit = normalize_target_boxes(
            test_rows,
            minimum_visible_fraction=test_minimum_visible_fraction,
            require_complete_page_geometry=args.require_complete_page_targets,
            long_span_minimum_visible_fraction=(
                test_long_span_minimum_visible_fraction
            ),
            tile_overlap=test_target_tile_overlap,
        )
    replay_box_audit: dict[str, Any] | None = None
    if args.replay_prepared_dir is not None:
        replay_minimum_visible_fraction = 1.0
        replay_long_span_minimum_visible_fraction = 1.0
        replay_target_tile_overlap = 256.0
        replay_prepare_report_path = (
            args.replay_prepared_dir / "prepare-report.json"
        )
        if replay_prepare_report_path.is_file():
            replay_prepare_report = json.loads(
                replay_prepare_report_path.read_text(encoding="utf-8")
            )
            replay_minimum_visible_fraction = float(
                replay_prepare_report.get("minimum_object_fraction", 0.8)
            )
            replay_long_span_minimum_visible_fraction = float(
                replay_prepare_report.get(
                    "long_span_minimum_object_fraction",
                    replay_minimum_visible_fraction,
                )
            )
            replay_target_tile_overlap = float(
                replay_prepare_report.get("overlap", 256.0)
            )
        if replay_jsonl_sha256 is None:
            raise RuntimeError("replay JSONL hash is unavailable")
        replay_box_audit = reusable_zero_clip_target_box_audit(
            stored_target_box_audits.get("replay"),
            jsonl_sha256=replay_jsonl_sha256,
            row_count=len(replay_rows),
            minimum_visible_fraction=replay_minimum_visible_fraction,
            require_complete_page_geometry=args.require_complete_page_targets,
            long_span_minimum_visible_fraction=(
                replay_long_span_minimum_visible_fraction
            ),
            tile_overlap=replay_target_tile_overlap,
        )
        if replay_box_audit is None:
            replay_rows, replay_box_audit = normalize_target_boxes(
                replay_rows,
                minimum_visible_fraction=replay_minimum_visible_fraction,
                require_complete_page_geometry=(
                    args.require_complete_page_targets
                ),
                long_span_minimum_visible_fraction=(
                    replay_long_span_minimum_visible_fraction
                ),
                tile_overlap=replay_target_tile_overlap,
            )
    (
        sample_weights,
        sampled_class_counts,
        replay_sampled_class_counts,
    ) = replay_mixture_sample_weights(
        train_rows,
        replay_rows,
        replay_fraction=args.replay_fraction,
        power=args.rare_class_sampling_power,
        maximum_repeat=args.rare_class_max_repeat,
        replay_maximum_repeat=args.replay_rare_class_max_repeat,
    )
    test_class_counts = detector_class_counts(test_rows)
    unsupported_required_classes = insufficient_required_class_support(
        required_class_maps,
        class_name_by_label=class_name_by_label,
        test_class_counts=test_class_counts,
        minimum_objects=args.minimum_required_class_test_objects,
    )
    if unsupported_required_classes:
        details = ", ".join(
            f"{name}={count}"
            for name, count in unsupported_required_classes.items()
        )
        raise ValueError(
            "independent detector test support is below "
            f"{args.minimum_required_class_test_objects}: {details}"
        )
    serializable_args = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
        if name != "resume"
        and name != "allow_resume_worker_change"
        and name
        not in {
            "checkpoint_every_steps",
            "device",
            "cpu_threads",
            "image_cache_dir",
            "populate_image_cache",
            "microbatch_size",
            "adaptive_full_batch_object_limit",
            "adaptive_full_batch_min_free_mib",
            "runtime_stop_after_epoch",
            "runtime_stop_reason",
        }
        and (
            name not in {
                "stop_when_accepted",
                "resumable_augmentation_v3",
            }
            or bool(value)
        )
    }
    run_config = {
        "format": 1,
        "model_contract": detector_model_contract(
            class_aware_matcher=args.class_aware_matcher_ablation,
        ),
        "priority_selection_protocol": PRIORITY_SELECTION_PROTOCOL,
        "arguments": serializable_args,
        "prepared_manifest_sha256": sha256_file(
            args.prepared_dir / "manifest.json"
        ),
        "prepared_prepare_report_sha256": (
            sha256_file(prepare_report_path)
            if prepare_report_path.is_file()
            else None
        ),
        "categories_sha256": sha256_file(args.prepared_dir / "categories.json"),
        "train_jsonl_sha256": train_jsonl_sha256,
        "test_jsonl_sha256": test_jsonl_sha256,
        "train_tiles": len(train_rows),
        "test_tiles": len(test_rows),
        "initialization": initialization,
    }
    if args.evaluation_prepared_dir is not None:
        run_config["evaluation"] = {
            "prepared_manifest_sha256": sha256_file(
                evaluation_prepared_dir / "manifest.json"
            ),
            "prepared_prepare_report_sha256": (
                sha256_file(evaluation_prepare_report_path)
                if evaluation_prepare_report_path.is_file()
                else None
            ),
            "categories_sha256": sha256_file(
                evaluation_prepared_dir / "categories.json"
            ),
            "test_jsonl_sha256": test_jsonl_sha256,
            "test_tiles": len(test_rows),
        }
    if args.replay_prepared_dir is not None:
        run_config["replay"] = {
            "prepared_manifest_sha256": sha256_file(
                args.replay_prepared_dir / "manifest.json"
            ),
            "prepared_prepare_report_sha256": (
                sha256_file(replay_prepare_report_path)
                if replay_prepare_report_path.is_file()
                else None
            ),
            "categories_sha256": sha256_file(
                args.replay_prepared_dir / "categories.json"
            ),
            "train_jsonl_sha256": replay_jsonl_sha256,
            "train_tiles": len(replay_rows),
            "fraction": args.replay_fraction,
        }
    run_config["target_box_audits"] = {
        "train": bind_zero_clip_target_box_audit(
            train_box_audit,
            jsonl_sha256=train_jsonl_sha256,
            row_count=len(train_rows),
        ),
        "test": bind_zero_clip_target_box_audit(
            test_box_audit,
            jsonl_sha256=test_jsonl_sha256,
            row_count=len(test_rows),
        ),
        "replay": (
            bind_zero_clip_target_box_audit(
                replay_box_audit,
                jsonl_sha256=replay_jsonl_sha256,
                row_count=len(replay_rows),
            )
            if replay_box_audit is not None
            and replay_jsonl_sha256 is not None
            else None
        ),
    }
    if (
        int(train_box_audit["clipped_objects"]) > 0
        or int(test_box_audit["clipped_objects"]) > 0
        or (
            replay_box_audit is not None
            and int(replay_box_audit["clipped_objects"]) > 0
        )
    ):
        run_config["legacy_target_box_normalization"] = {
            "train": train_box_audit,
            "test": test_box_audit,
            "replay": replay_box_audit,
        }
    resume_checkpoint_sha256 = (
        sha256_file(checkpoint_path) if resuming else None
    )
    # Do not leave an empty model directory when any dataset, class-support,
    # initialization or immutable-run validation above refuses the run.
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    if resuming or restarting_uncheckpointed:
        if stored_run_config_for_resume is None:
            raise RuntimeError("stored detector run configuration is unavailable")
        stored_run_config = stored_run_config_for_resume
        run_config = reconcile_resume_run_config(
            stored_run_config,
            run_config,
            allow_worker_change=bool(
                args.allow_resume_worker_change and resuming
            ),
            checkpoint_sha256=resume_checkpoint_sha256,
        )
        if run_config != stored_run_config:
            run_config_path.write_text(
                json.dumps(run_config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    else:
        run_config_path.write_text(
            json.dumps(run_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    verified_image_cache: dict[tuple[str, str], Path] = {}
    verified_image_cache_report: dict[str, Any] | None = None

    class SymbolTileDataset(Dataset):
        def __init__(
            self,
            items: list[tuple[dict[str, Any], Path]],
            training: bool,
        ) -> None:
            self.items = items
            self.training = training

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(
            self,
            index: int | tuple[int, int],
        ) -> tuple[Any, dict[str, Any]]:
            occurrence_seed: int | None = None
            if isinstance(index, tuple):
                item_index, occurrence_seed = index
            else:
                item_index = index
            row, images_dir = self.items[item_index]
            augmentation_rng: Any = (
                random.Random(occurrence_seed)
                if occurrence_seed is not None
                else random
            )
            image_key = (str(images_dir), str(row["image"]))
            image_path = verified_image_cache.get(
                image_key,
                images_dir / row["image"],
            )
            image = load_grayscale_crop(
                image_path,
                row["crop_xyxy"],
            ).convert("RGB")
            if self.training:
                image = ImageEnhance.Contrast(image).enhance(
                    augmentation_rng.uniform(0.65, 1.35)
                )
                image = ImageEnhance.Brightness(image).enhance(
                    augmentation_rng.uniform(0.88, 1.08)
                )
                image = ImageEnhance.Sharpness(image).enhance(
                    augmentation_rng.uniform(0.65, 1.45)
                )
                if augmentation_rng.random() < 0.35:
                    image = image.filter(
                        ImageFilter.GaussianBlur(
                            radius=augmentation_rng.uniform(0.15, 0.7)
                        )
                    )
                if augmentation_rng.random() < 0.18:
                    # Printed scans vary in ink spread and thresholding.  MinFilter
                    # thickens dark strokes; MaxFilter weakens them.
                    image = image.filter(
                        ImageFilter.MinFilter(3)
                        if augmentation_rng.random() < 0.5
                        else ImageFilter.MaxFilter(3)
                    )
                if augmentation_rng.random() < 0.30:
                    scale = augmentation_rng.uniform(0.62, 0.92)
                    reduced = image.resize(
                        (
                            max(32, round(image.width * scale)),
                            max(32, round(image.height * scale)),
                        ),
                        Image.Resampling.BILINEAR,
                    )
                    image = reduced.resize(
                        image.size,
                        Image.Resampling.BILINEAR,
                    )
            tensor = vision_f.pil_to_tensor(image).float().div_(255.0)
            if self.training and augmentation_rng.random() < 0.45:
                sigma = augmentation_rng.uniform(0.002, 0.018)
                if occurrence_seed is None:
                    noise = torch.randn_like(tensor)
                else:
                    noise_generator = torch.Generator().manual_seed(
                        occurrence_seed ^ 0x5A17C9E3
                    )
                    noise = torch.randn(
                        tensor.shape,
                        dtype=tensor.dtype,
                        generator=noise_generator,
                    )
                tensor = (tensor + noise * sigma).clamp_(0, 1)
            if self.training and augmentation_rng.random() < 0.35:
                # A smooth exposure ramp approximates uneven book/scanner lighting
                # without changing symbol geometry or target boxes.
                horizontal = torch.linspace(
                    augmentation_rng.uniform(0.82, 1.02),
                    augmentation_rng.uniform(0.82, 1.02),
                    tensor.shape[2],
                ).view(1, 1, -1)
                vertical = torch.linspace(
                    augmentation_rng.uniform(0.90, 1.04),
                    augmentation_rng.uniform(0.90, 1.04),
                    tensor.shape[1],
                ).view(1, -1, 1)
                tensor = (tensor * horizontal * vertical).clamp_(0, 1)
            boxes = torch.tensor(
                [obj["box_xyxy"] for obj in row["objects"]], dtype=torch.float32
            ).reshape(-1, 4)
            labels = torch.tensor(
                [obj["label"] for obj in row["objects"]], dtype=torch.int64
            )
            target = {
                "boxes": boxes,
                "labels": labels,
                "image_id": torch.tensor([item_index], dtype=torch.int64),
            }
            return tensor, target

    def collate(
        batch: list[tuple[Any, dict[str, Any]]],
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        images, targets = zip(*batch)
        return list(images), list(targets)

    loader_generator = torch.Generator().manual_seed(args.seed)
    train_items = [
        (row, args.images_dir)
        for row in train_rows
    ]
    if args.replay_images_dir is not None:
        train_items.extend(
            (row, args.replay_images_dir)
            for row in replay_rows
        )
    test_items = [
        (row, evaluation_images_dir)
        for row in test_rows
    ]
    if args.image_cache_dir is not None:
        (
            verified_image_cache,
            verified_image_cache_report,
        ) = prepare_verified_detector_image_cache(
            train_items + test_items,
            args.image_cache_dir,
            populate=bool(args.populate_image_cache),
        )
        print(
            json.dumps(
                {
                    "state": "verified_detector_image_cache_ready",
                    **verified_image_cache_report,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if resuming:
        execution_record = {
            "resume_checkpoint_sha256": resume_checkpoint_sha256,
            "logical_batch_size": args.batch_size,
            "microbatch_size": args.microbatch_size or args.batch_size,
            "adaptive_full_batch_object_limit": (
                args.adaptive_full_batch_object_limit
            ),
            "adaptive_full_batch_min_free_mib": (
                args.adaptive_full_batch_min_free_mib
            ),
            "cuda_allocator_config": os.environ.get(
                "PYTORCH_CUDA_ALLOC_CONF"
            ),
            "numerical_continuity": (
                "logical_batch_execution"
                if (
                    (args.microbatch_size or args.batch_size)
                    == args.batch_size
                    and args.adaptive_full_batch_object_limit is None
                )
                else (
                    "mean_loss_gradient_equivalent_in_real_arithmetic;"
                    "cuda_fp16_not_bit_identical"
                )
            ),
            "verified_image_cache": (
                {
                    key: verified_image_cache_report[key]
                    for key in (
                        "contract",
                        "manifest_sha256",
                        "source_files",
                        "unique_blobs",
                        "bytes",
                    )
                }
                if verified_image_cache_report is not None
                else None
            ),
            "runtime_stop_after_epoch": args.runtime_stop_after_epoch,
            "runtime_stop_reason": args.runtime_stop_reason,
        }
        execution_records = run_config.setdefault(
            "runtime_execution_records",
            [],
        )
        if execution_record not in execution_records:
            execution_records.append(execution_record)
            temporary_run_config = run_config_path.with_name(
                f".{run_config_path.name}.{os.getpid()}.tmp"
            )
            temporary_run_config.write_text(
                json.dumps(run_config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_run_config, run_config_path)
    train_dataset = SymbolTileDataset(train_items, training=True)
    training_steps_per_epoch = math.ceil(
        len(train_rows) / args.batch_size
    )
    legacy_train_loader: Any | None = None
    if not args.resumable_augmentation_v3:
        train_sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(train_rows),
            replacement=True,
            generator=loader_generator,
        )
        legacy_train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=args.workers,
            collate_fn=collate,
            pin_memory=use_cuda,
            # Training and evaluation loaders otherwise keep two independent
            # worker pools alive together. Full 1024px score pages plus Python
            # target metadata make that waste several GB under WSL. Epoch startup
            # is negligible relative to an epoch, so release workers at iterator
            # exhaustion and give the next phase the memory immediately.
            persistent_workers=False,
            generator=loader_generator,
        )

    def resumable_train_loader(
        *,
        epoch_number: int,
        completed_steps: int,
    ) -> Any:
        if not args.resumable_augmentation_v3:
            raise RuntimeError("resumable augmentation v3 is disabled")
        if not 0 <= completed_steps <= training_steps_per_epoch:
            raise ValueError("invalid completed detector steps for fast resume")
        sample_generator = torch.Generator().manual_seed(
            detector_occurrence_seed(
                base_seed=args.seed,
                epoch_number=epoch_number,
                sample_position=0,
                item_index=0,
            )
        )
        sampled_indices = torch.multinomial(
            torch.as_tensor(sample_weights, dtype=torch.double),
            len(train_rows),
            replacement=True,
            generator=sample_generator,
        ).tolist()
        start_sample = completed_steps * args.batch_size
        occurrences = remaining_detector_occurrences(
            sampled_indices,
            base_seed=args.seed,
            epoch_number=epoch_number,
            completed_steps=completed_steps,
            batch_size=args.batch_size,
        )
        worker_generator = torch.Generator().manual_seed(
            detector_occurrence_seed(
                base_seed=args.seed ^ 0x6D2B79F5,
                epoch_number=epoch_number,
                sample_position=start_sample,
                item_index=0,
            )
        )
        return DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=occurrences,
            num_workers=args.workers,
            collate_fn=collate,
            pin_memory=use_cuda,
            persistent_workers=False,
            generator=worker_generator,
        )

    def migrated_legacy_suffix_loader(
        *,
        epoch_number: int,
        completed_steps: int,
        epoch_loader_generator_state: Any,
    ) -> Any:
        sampled_indices = legacy_detector_sampled_indices(
            sample_weights,
            num_samples=len(train_rows),
            epoch_loader_generator_state=epoch_loader_generator_state,
        )
        occurrences = remaining_detector_occurrences(
            sampled_indices,
            base_seed=args.seed,
            epoch_number=epoch_number,
            completed_steps=completed_steps,
            batch_size=args.batch_size,
        )
        worker_generator = torch.Generator().manual_seed(
            detector_occurrence_seed(
                base_seed=args.seed ^ 0x4C454741,
                epoch_number=epoch_number,
                sample_position=completed_steps * args.batch_size,
                item_index=0,
            )
        )
        return DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=occurrences,
            num_workers=args.workers,
            collate_fn=collate,
            pin_memory=use_cuda,
            persistent_workers=False,
            generator=worker_generator,
        )

    test_loader = DataLoader(
        SymbolTileDataset(test_items, training=False),
        batch_size=args.evaluation_batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate,
        pin_memory=use_cuda,
        persistent_workers=False,
    )

    # torchvision's v2 builder creates and passes its own anchor generator and
    # head internally.  Supplying ``anchor_generator=`` through kwargs therefore
    # raises a duplicate-key TypeError in torchvision 0.21.  Build the official
    # v2 backbone first, then replace the two coupled modules together.  The
    # wider ratios are essential for long ties, slurs and hairpins.
    model = build_detector_model(
        number_of_classes=number_of_classes,
        score_threshold=args.score_threshold,
        detections_per_tile=args.detections_per_tile,
        pretrained_backbone=True,
        class_name_by_label=class_name_by_label,
        class_aware_matcher=args.class_aware_matcher_ablation,
    )
    initialization_stats: dict[str, Any] | None = None
    if not resuming and args.initial_model is not None:
        initial_state = torch.load(
            args.initial_model,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(initial_state, dict):
            raise ValueError("initial detector model is not a state dictionary")
        model.load_state_dict(initial_state, strict=True)
        initialization_stats = {
            "loaded_tensors": len(initial_state),
            "skipped_tensors": 0,
        }
        print(
            json.dumps(
                {
                    "initialized_from": str(args.initial_model),
                    "model_sha256": initialization["model_sha256"],
                }
            ),
            flush=True,
        )
    elif not resuming and args.initial_backbone_model is not None:
        initial_state = torch.load(
            args.initial_backbone_model,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(initial_state, dict):
            raise ValueError(
                "initial backbone detector model is not a state dictionary"
            )
        target_state = model.state_dict()
        compatible = {
            name: value
            for name, value in initial_state.items()
            if name in target_state
            and tuple(value.shape) == tuple(target_state[name].shape)
            and not name.startswith(
                "head.classification_head.cls_logits."
            )
        }
        backbone_tensors = sum(
            name.startswith("backbone.") for name in compatible
        )
        regression_tensors = sum(
            name.startswith("head.regression_head.") for name in compatible
        )
        if backbone_tensors < 50 or regression_tensors < 5:
            raise ValueError(
                "initial detector has insufficient compatible feature weights: "
                f"backbone={backbone_tensors}, regression={regression_tensors}"
            )
        target_state.update(compatible)
        model.load_state_dict(target_state, strict=True)
        initialization_stats = {
            "loaded_tensors": len(compatible),
            "skipped_tensors": len(initial_state) - len(compatible),
            "backbone_tensors": backbone_tensors,
            "regression_tensors": regression_tensors,
        }
        print(
            json.dumps(
                {
                    "initialized_compatible_weights_from": str(
                        args.initial_backbone_model
                    ),
                    **initialization_stats,
                    "model_sha256": initialization["model_sha256"],
                }
            ),
            flush=True,
        )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    run_log: dict[str, Any] = {"epochs": []}
    best_map50 = -1.0
    best_map75 = -1.0
    best_priority_map = -1.0
    best_selection_score = -1.0
    best_gate_passed = False
    best_epoch = 0
    best_path = args.output_dir / "model.best.pt"
    start_epoch = 0
    resume_in_progress_step = 0
    resume_epoch_loader_generator_state: Any | None = None
    resume_epoch_python_random_state: Any | None = None
    resume_epoch_torch_rng_state: Any | None = None

    def atomic_torch_save(payload: Any, path: Path) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def atomic_json_write(payload: Any, path: Path) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(
                json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def save_recovery_checkpoint(
        *,
        completed_epochs: int,
        in_progress_epoch: int = 0,
        in_progress_step: int = 0,
        running_loss: float = 0.0,
        loss_components: dict[str, float] | None = None,
        epoch_loader_generator_state: Any | None = None,
        epoch_python_random_state: Any | None = None,
        epoch_torch_rng_state: Any | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "format": 2,
            "epoch": completed_epochs,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "loader_generator": loader_generator.get_state(),
            "python_random_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            # Keep the human-readable partial report and the binary recovery
            # state transactionally reconcilable after a power loss.
            "run_log": run_log,
        }
        if use_cuda:
            payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        if in_progress_step:
            payload.update(
                {
                    "in_progress_epoch": in_progress_epoch,
                    "in_progress_step": in_progress_step,
                    "running_loss": running_loss,
                    "loss_components": loss_components or {},
                    "epoch_loader_generator": epoch_loader_generator_state,
                    "epoch_python_random_state": epoch_python_random_state,
                    "epoch_torch_rng_state": epoch_torch_rng_state,
                }
            )
        atomic_torch_save(payload, checkpoint_path)
        atomic_json_write(run_log, partial_metrics_path)

    if resuming:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_epoch, resume_in_progress_step = detector_checkpoint_position(
            checkpoint,
            total_epochs=args.epochs,
            accumulate=args.accumulate,
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        if "loader_generator" in checkpoint:
            loader_generator.set_state(checkpoint["loader_generator"])
        if "python_random_state" in checkpoint:
            random.setstate(checkpoint["python_random_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if use_cuda and "cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        checkpoint_run_log = checkpoint.get("run_log")
        if isinstance(checkpoint_run_log, dict):
            run_log = checkpoint_run_log
            atomic_json_write(run_log, partial_metrics_path)
        else:
            run_log = json.loads(partial_metrics_path.read_text(encoding="utf-8"))
        epoch_records = run_log.get("epochs", [])
        if (
            not isinstance(epoch_records, list)
            or len(epoch_records) != checkpoint_epoch
            or (
                checkpoint_epoch > 0
                and int(epoch_records[-1].get("epoch", 0)) != checkpoint_epoch
            )
        ):
            raise ValueError("detector checkpoint and partial metrics disagree")
        for record in epoch_records:
            test_metrics = record.get("test")
            if not isinstance(test_metrics, dict):
                continue
            selection_score = float(test_metrics.get("selection_score", -1.0))
            acceptance_failures = detector_acceptance_failures(
                test_metrics,
                minimum_map_50=args.minimum_best_map_50,
                minimum_map_75=args.minimum_best_map_75,
                minimum_priority_map=args.minimum_best_priority_map,
                required_class_maps=required_class_maps,
            )
            gate_passed = not acceptance_failures
            if should_replace_detector_best(
                current_gate_passed=gate_passed,
                current_selection_score=selection_score,
                best_gate_passed=best_gate_passed,
                best_selection_score=best_selection_score,
            ):
                best_selection_score = selection_score
                best_gate_passed = gate_passed
                best_priority_map = float(
                    test_metrics.get("priority_mark_map", -1.0)
                )
                best_map50 = float(test_metrics.get("map_50", -1.0))
                best_map75 = float(test_metrics.get("map_75", -1.0))
                best_epoch = int(record["epoch"])
        if best_epoch > 0 and not best_path.is_file():
            raise FileNotFoundError(
                f"best detector model is missing during resume: {best_path}"
            )
        start_epoch = checkpoint_epoch
        if resume_in_progress_step:
            resume_epoch_loader_generator_state = checkpoint.get(
                "epoch_loader_generator"
            )
            resume_epoch_python_random_state = checkpoint.get(
                "epoch_python_random_state"
            )
            resume_epoch_torch_rng_state = checkpoint.get("epoch_torch_rng_state")
            if any(
                state is None
                for state in (
                    resume_epoch_loader_generator_state,
                    resume_epoch_python_random_state,
                    resume_epoch_torch_rng_state,
                )
            ):
                raise ValueError(
                    "in-progress detector checkpoint is missing epoch RNG state"
                )
        print(
            json.dumps(
                {
                    "resume": True,
                    "completed_epochs": checkpoint_epoch,
                    "remaining_epochs": args.epochs - checkpoint_epoch,
                    "in_progress_step": resume_in_progress_step,
                }
            ),
            flush=True,
        )
    else:
        # Establish a valid recovery point before the first expensive batch.
        save_recovery_checkpoint(completed_epochs=0)

    def evaluate() -> dict[str, Any]:
        model.eval()
        collected_outputs: list[dict[str, Any]] = []
        collected_targets: list[dict[str, Any]] = []
        with evaluation_detection_limit(model, args.detections_per_tile):
            with torch.inference_mode():
                for images, targets in test_loader:
                    images = [
                        image.to(device, non_blocking=use_cuda) for image in images
                    ]
                    outputs = model(images)
                    cpu_outputs = [
                        {
                            name: value.detach().cpu()
                            for name, value in output.items()
                        }
                        for output in outputs
                    ]
                    collected_outputs.extend(cpu_outputs)
                    collected_targets.extend(targets)
        raw = compute_dense_detection_metrics(
            collected_outputs,
            collected_targets,
        )
        class_ids = raw.get("classes", [])
        per_class_map = raw.get("map_per_class", [])
        raw["map_per_class_named"] = {
            class_name_by_label.get(int(label), str(label)): float(value)
            for label, value in zip(class_ids, per_class_map)
        }
        return raw

    if (
        args.runtime_stop_after_epoch is not None
        and start_epoch >= args.runtime_stop_after_epoch
    ):
        run_log["runtime_stopped_after_epoch"] = (
            args.runtime_stop_after_epoch
        )
        run_log["runtime_stop_reason"] = args.runtime_stop_reason
        atomic_json_write(run_log, partial_metrics_path)
    for epoch_index in range(start_epoch, runtime_final_epoch):
        epoch_number = epoch_index + 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if resume_in_progress_step and epoch_number == start_epoch + 1:
            epoch_loader_generator_state = resume_epoch_loader_generator_state
            epoch_python_random_state = resume_epoch_python_random_state
            epoch_torch_rng_state = resume_epoch_torch_rng_state
            if not args.resumable_augmentation_v3:
                loader_generator.set_state(epoch_loader_generator_state)
                random.setstate(epoch_python_random_state)
                torch.set_rng_state(epoch_torch_rng_state)
            running_loss = float(checkpoint.get("running_loss", 0.0))
            loss_components = {
                str(name): float(value)
                for name, value in checkpoint.get("loss_components", {}).items()
            }
            first_training_step = resume_in_progress_step + 1
        else:
            epoch_loader_generator_state = loader_generator.get_state()
            epoch_python_random_state = random.getstate()
            epoch_torch_rng_state = torch.get_rng_state()
            running_loss = 0.0
            loss_components = {}
            first_training_step = 1
        migrated_worker_suffix = bool(
            resume_in_progress_step
            and epoch_number == start_epoch + 1
            and run_config.get("runtime_worker_transitions")
            and not args.resumable_augmentation_v3
        )
        if args.resumable_augmentation_v3:
            train_loader = resumable_train_loader(
                epoch_number=epoch_number,
                completed_steps=first_training_step - 1,
            )
            training_iterator = enumerate(
                train_loader,
                start=first_training_step,
            )
        elif migrated_worker_suffix:
            train_loader = migrated_legacy_suffix_loader(
                epoch_number=epoch_number,
                completed_steps=first_training_step - 1,
                epoch_loader_generator_state=epoch_loader_generator_state,
            )
            training_iterator = enumerate(
                train_loader,
                start=first_training_step,
            )
        else:
            if legacy_train_loader is None:
                raise RuntimeError("legacy detector loader is unavailable")
            train_loader = legacy_train_loader
            training_iterator = enumerate(train_loader, start=1)
        runtime_batching_window = {
            "full_logical_batch_steps": 0,
            "fallback_microbatch_steps": 0,
            "target_objects": 0,
            "minimum_effective_free_mib": None,
        }
        for step, (images, targets) in training_iterator:
            if (
                not args.resumable_augmentation_v3
                and not migrated_worker_suffix
                and step < first_training_step
            ):
                # Recreate the sampler position and worker augmentation stream
                # without repeating any optimizer or GPU work.
                continue
            batch_count = len(images)
            effective_free_mib: int | None = None
            if (
                use_cuda
                and args.adaptive_full_batch_min_free_mib is not None
            ):
                raw_free_bytes, _ = torch.cuda.mem_get_info()
                reclaimable_bytes = max(
                    0,
                    torch.cuda.memory_reserved()
                    - torch.cuda.memory_allocated(),
                )
                effective_free_mib = int(
                    (raw_free_bytes + reclaimable_bytes) // (1024 * 1024)
                )
            microbatch_size, target_object_count = (
                detector_runtime_microbatch_size(
                    targets,
                    configured_microbatch_size=min(
                        args.microbatch_size or batch_count,
                        batch_count,
                    ),
                    adaptive_full_batch_object_limit=(
                        args.adaptive_full_batch_object_limit
                    ),
                    adaptive_full_batch_min_free_mib=(
                        args.adaptive_full_batch_min_free_mib
                    ),
                    effective_free_mib=effective_free_mib,
                )
            )
            runtime_batching_window["target_objects"] += target_object_count
            if effective_free_mib is not None:
                previous_minimum = runtime_batching_window[
                    "minimum_effective_free_mib"
                ]
                runtime_batching_window["minimum_effective_free_mib"] = (
                    effective_free_mib
                    if previous_minimum is None
                    else min(previous_minimum, effective_free_mib)
                )
            if microbatch_size == batch_count:
                runtime_batching_window["full_logical_batch_steps"] += 1
            else:
                runtime_batching_window["fallback_microbatch_steps"] += 1
            batch_loss_value = 0.0
            batch_component_values: dict[str, float] = {}
            for micro_start, micro_end, micro_fraction in detector_microbatch_plan(
                batch_count,
                microbatch_size,
            ):
                device_images = [
                    image.to(device, non_blocking=use_cuda)
                    for image in images[micro_start:micro_end]
                ]
                device_targets = [
                    {
                        name: value.to(device, non_blocking=use_cuda)
                        if hasattr(value, "to")
                        else value
                        for name, value in target.items()
                    }
                    for target in targets[micro_start:micro_end]
                ]
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_cuda,
                ):
                    micro_losses = model(device_images, device_targets)
                    micro_loss = (
                        sum(micro_losses.values())
                        * micro_fraction
                        / args.accumulate
                    )
                scaler.scale(micro_loss).backward()
                batch_loss_value += (
                    float(sum(micro_losses.values()).detach().cpu())
                    * micro_fraction
                )
                for name, value in micro_losses.items():
                    batch_component_values[name] = (
                        batch_component_values.get(name, 0.0)
                        + float(value.detach().cpu()) * micro_fraction
                    )
                del device_images, device_targets, micro_losses, micro_loss
            if (
                step % args.accumulate == 0
                or step == training_steps_per_epoch
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += batch_loss_value
            for name, value in batch_component_values.items():
                loss_components[name] = loss_components.get(name, 0.0) + value
            if step % 100 == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch_number,
                            "step": step,
                            # Keep the denominator absolute after an exact
                            # suffix resume; ``len(train_loader)`` is only the
                            # number of remaining batches in v3.
                            "steps": training_steps_per_epoch,
                            "loss": running_loss / step,
                            "runtime_batching_last_100": {
                                **runtime_batching_window,
                                "adaptive_full_batch_object_limit": (
                                    args.adaptive_full_batch_object_limit
                                ),
                                "adaptive_full_batch_min_free_mib": (
                                    args.adaptive_full_batch_min_free_mib
                                ),
                            },
                        }
                    ),
                    flush=True,
                )
                runtime_batching_window = {
                    "full_logical_batch_steps": 0,
                    "fallback_microbatch_steps": 0,
                    "target_objects": 0,
                    "minimum_effective_free_mib": None,
                }
            if (
                args.checkpoint_every_steps
                and step % args.checkpoint_every_steps == 0
                and step % args.accumulate == 0
                and step < training_steps_per_epoch
            ):
                save_recovery_checkpoint(
                    completed_epochs=epoch_index,
                    in_progress_epoch=epoch_number,
                    in_progress_step=step,
                    running_loss=running_loss,
                    loss_components=loss_components,
                    epoch_loader_generator_state=epoch_loader_generator_state,
                    epoch_python_random_state=epoch_python_random_state,
                    epoch_torch_rng_state=epoch_torch_rng_state,
                )
                print(
                    json.dumps(
                        {
                            "checkpoint": "in_epoch",
                            "epoch": epoch_number,
                            "step": step,
                        }
                    ),
                    flush=True,
                )
        resume_in_progress_step = 0
        scheduler.step()
        record: dict[str, Any] = {
            "epoch": epoch_number,
            "loss": running_loss / max(1, training_steps_per_epoch),
            "loss_components": {
                name: total / max(1, training_steps_per_epoch)
                for name, total in loss_components.items()
            },
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if epoch_number % args.eval_every == 0 or epoch_number == args.epochs:
            record["test"] = evaluate()
            current_map50 = float(record["test"]["map_50"])
            current_class_support = {
                class_name_by_label.get(int(label), str(label)): int(count)
                for label, count in test_class_counts.items()
            }
            current_selection_score, current_priority_map = priority_selection_score(
                overall_map=float(record["test"]["map"]),
                per_class_map=record["test"]["map_per_class_named"],
                class_support=current_class_support,
                minimum_support=args.minimum_required_class_test_objects,
            )
            (
                current_support_filtered_map,
                current_supported_classes,
            ) = support_filtered_macro_map(
                per_class_map=record["test"]["map_per_class_named"],
                class_support=current_class_support,
                minimum_support=args.minimum_required_class_test_objects,
            )
            record["test"]["priority_mark_map"] = current_priority_map
            record["test"]["priority_mark_minimum_class_support"] = (
                args.minimum_required_class_test_objects
            )
            record["test"]["priority_mark_supported_classes"] = sorted(
                name
                for label, count in test_class_counts.items()
                if count >= args.minimum_required_class_test_objects
                for name in [class_name_by_label.get(int(label), str(label))]
                if is_priority_mark_class(name)
            )
            record["test"]["selection_support_filtered_map"] = (
                current_support_filtered_map
            )
            record["test"]["selection_minimum_class_support"] = (
                args.minimum_required_class_test_objects
            )
            record["test"]["selection_supported_classes"] = (
                current_supported_classes
            )
            record["test"]["selection_score"] = current_selection_score
            current_acceptance_failures = detector_acceptance_failures(
                record["test"],
                minimum_map_50=args.minimum_best_map_50,
                minimum_map_75=args.minimum_best_map_75,
                minimum_priority_map=args.minimum_best_priority_map,
                required_class_maps=required_class_maps,
            )
            current_gate_passed = not current_acceptance_failures
            record["test"]["acceptance_probe"] = {
                "passed": current_gate_passed,
                "failures": current_acceptance_failures,
            }
            if should_replace_detector_best(
                current_gate_passed=current_gate_passed,
                current_selection_score=current_selection_score,
                best_gate_passed=best_gate_passed,
                best_selection_score=best_selection_score,
            ):
                best_map50 = current_map50
                best_map75 = float(record["test"]["map_75"])
                best_priority_map = current_priority_map
                best_selection_score = current_selection_score
                best_epoch = epoch_number
                best_gate_passed = current_gate_passed
                atomic_torch_save(model.state_dict(), best_path)
            if args.stop_when_accepted and current_gate_passed:
                run_log["early_stopped_after_epoch"] = epoch_number
        run_log["epochs"].append(record)
        save_recovery_checkpoint(completed_epochs=epoch_number)
        print(json.dumps(record, sort_keys=True), flush=True)
        if (
            args.stop_when_accepted
            and run_log.get("early_stopped_after_epoch") == epoch_number
        ):
            break
        if (
            args.runtime_stop_after_epoch is not None
            and epoch_number >= args.runtime_stop_after_epoch
        ):
            run_log["runtime_stopped_after_epoch"] = epoch_number
            run_log["runtime_stop_reason"] = args.runtime_stop_reason
            atomic_json_write(run_log, partial_metrics_path)
            break

    if not best_path.is_file():
        raise RuntimeError("No evaluated detector checkpoint was produced")
    best_record = next(
        (
            record
            for record in run_log["epochs"]
            if int(record["epoch"]) == best_epoch
        ),
        None,
    )
    if best_record is None or not isinstance(best_record.get("test"), dict):
        raise RuntimeError("best detector epoch has no evaluation record")
    acceptance_failures = detector_acceptance_failures(
        best_record["test"],
        minimum_map_50=args.minimum_best_map_50,
        minimum_map_75=args.minimum_best_map_75,
        minimum_priority_map=args.minimum_best_priority_map,
        required_class_maps=required_class_maps,
    )
    report = json_ready({
        "format": 1,
        "purpose": "positioned notation-mark verifier; not sole semantic OMR output",
        "runtime": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda": torch.version.cuda,
            "device": device.type,
            "gpu": torch.cuda.get_device_name(0) if use_cuda else None,
            "cpu_threads": torch.get_num_threads() if not use_cuda else None,
            "mixed_precision": (
                "float16 autocast with GradScaler" if use_cuda else "disabled"
            ),
            "cudnn_benchmark_fixed_1024_tiles": use_cuda,
            "cuda_allocator_config": os.environ.get(
                "PYTORCH_CUDA_ALLOC_CONF"
            ),
            "augmentation_profile": (
                "scan-photometric-resumable-occurrence-v3"
                if args.resumable_augmentation_v3
                else (
                    "scan-photometric-v2-with-coordinate-seeded-worker-transition-suffix"
                    if run_config.get("runtime_worker_transitions")
                    else "scan-photometric-v2"
                )
            ),
            "loader_workers": args.workers,
            "worker_transitions": run_config.get(
                "runtime_worker_transitions",
                [],
            ),
            "recovery_checkpoint_policy": (
                "retain_until_external_acceptance"
                if args.external_acceptance_pending
                else "retain_only_when_acceptance_fails"
            ),
            "recovery_checkpoint_retained": bool(acceptance_failures)
            or args.external_acceptance_pending,
        },
        "model_contract": run_config["model_contract"],
        "priority_selection_protocol": PRIORITY_SELECTION_PROTOCOL,
        "data": {
            "prepared_manifest_sha256": sha256_file(
                args.prepared_dir / "manifest.json"
            ),
            "evaluation_prepared_manifest_sha256": sha256_file(
                evaluation_prepared_dir / "manifest.json"
            ),
            "train_tiles": len(train_rows),
            "epoch_training_samples": len(train_rows),
            "replay_tiles": len(replay_rows),
            "replay_fraction": args.replay_fraction,
            "test_tiles": len(test_rows),
            "classes": len(classes),
            "source_split_overlap": 0,
            # Evaluation deliberately uses the same dense-page proposal cap as
            # deployment.  Do not report COCO's legacy 100-detection cap here:
            # ``evaluate()`` passes this exact argument through
            # ``evaluation_detection_limit`` and the dense metric keeps all
            # proposals produced under that cap.
            "evaluation_max_detections_per_tile": args.detections_per_tile,
            "sampled_class_counts": sampled_class_counts,
            "replay_sampled_class_counts": replay_sampled_class_counts,
            "test_class_counts": test_class_counts,
            "sample_weight_min": min(sample_weights),
            "sample_weight_max": max(sample_weights),
            "sample_weight_mean": sum(sample_weights) / max(1, len(sample_weights)),
            "target_box_normalization": {
                "train": train_box_audit,
                "test": test_box_audit,
                "replay": replay_box_audit,
            },
        },
        "initialization": initialization,
        "initialization_stats": initialization_stats,
        "configuration": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        } | {
            "prepared_dir": str(args.prepared_dir),
            "images_dir": str(args.images_dir),
            "output_dir": str(args.output_dir),
        },
        "verified_image_cache": verified_image_cache_report,
        "best_epoch": best_epoch,
        "best_map_50": best_map50,
        "best_map_75": best_map75,
        "best_priority_mark_map": best_priority_map,
        "best_selection_score": best_selection_score,
        "best_model_sha256": sha256_file(best_path),
        "completed_epochs": len(run_log["epochs"]),
        "planned_epochs": args.epochs,
        "early_stopped": "early_stopped_after_epoch" in run_log,
        "runtime_truncated": (
            len(run_log["epochs"]) < args.epochs
            and "runtime_stopped_after_epoch" in run_log
        ),
        "runtime_stop_after_epoch": run_log.get(
            "runtime_stopped_after_epoch"
        ),
        "runtime_stop_reason": run_log.get("runtime_stop_reason"),
        "acceptance": {
            "passed": not acceptance_failures,
            "minimum_best_map_50": args.minimum_best_map_50,
            "minimum_best_map_75": args.minimum_best_map_75,
            "minimum_best_priority_map": args.minimum_best_priority_map,
            "required_class_maps": required_class_maps,
            "failures": acceptance_failures,
        },
        "metrics": run_log,
        "elapsed_seconds": time.time() - started,
    })
    completed_report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(
        partial_metrics_path,
        args.output_dir / "metrics.json",
    )
    recovery_checkpoint_retained = finalize_detector_recovery_checkpoint(
        checkpoint_path,
        acceptance_failures=acceptance_failures,
        external_acceptance_pending=args.external_acceptance_pending,
    )
    if recovery_checkpoint_retained != (
        bool(acceptance_failures) or args.external_acceptance_pending
    ):
        raise RuntimeError(
            "detector recovery-checkpoint retention policy was not satisfied"
        )
    print(json.dumps(report, sort_keys=True), flush=True)
    if acceptance_failures:
        raise RuntimeError(
            "detector acceptance gates failed: "
            + "; ".join(acceptance_failures)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
