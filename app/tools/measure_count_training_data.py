from __future__ import annotations

"""Deterministic grouped training data for measure-count evidence fusion."""

import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np

from scorescan.measure_count_resolver import (
    FEATURE_NAMES,
    LEGACY_FEATURE_NAMES,
    build_measure_count_feature_bundle,
)


VARIANTS = (
    "primary",
    "flat",
    "deblock",
    "otsu",
    "adaptive",
    "upscale",
    "staffnorm",
    "system_localized",
)
KINDS = (
    "agreement",
    "high-confidence-layout-wrong",
    "single-family-duplicate-trap",
    "two-family-shared-error",
    "adjacent-count-split",
    "layout-only-rescue",
    "high-score-single-family-trap",
    "invalid-true-candidates",
    "candidate-majority-trap",
    "independent-family-consensus",
    "large-layout-error",
    "low-confidence-layout",
    "localized-correct-rescue",
    "localized-wrong-isolation",
    "localized-invalid-trap",
    "template-count-outlier",
    "invalid-sibling-family-trap",
)


@dataclass(frozen=True)
class CountTrainingCandidate:
    variant: str
    measure_count: int
    valid: bool
    agreement_ratio: float
    calibrated_probability: float
    raw_score: float
    measure_gap_penalty: float = 0.0


@dataclass(frozen=True)
class MeasureCountDataset:
    features: np.ndarray
    legacy_features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    option_counts: np.ndarray
    truths: np.ndarray
    kinds: np.ndarray
    decision_groups: tuple[tuple[int, ...], ...]
    deterministic_counts: np.ndarray
    layout_counts: np.ndarray
    layout_confidences: np.ndarray
    family_supports: np.ndarray
    candidate_supports: np.ndarray


def _seed(seed: int, group: int) -> int:
    return (seed * 1_000_003 + group * 97_409 + 0xC0A17) & 0xFFFFFFFF


def _jitter(rng: random.Random, value: float, spread: float, low: float, high: float) -> float:
    return max(low, min(high, rng.gauss(value, spread)))


def _candidate(
    rng: random.Random,
    variant: str,
    count: int,
    quality: str,
) -> CountTrainingCandidate:
    presets = {
        "true": (0.93, 0.84, 0.80, 980.0),
        "true-soft": (0.80, 0.72, 0.68, 930.0),
        "true-invalid": (0.20, 0.76, 0.70, 920.0),
        "wrong": (0.62, 0.52, 0.44, 850.0),
        "wrong-strong": (0.96, 0.88, 0.86, 1005.0),
        "wrong-medium": (0.86, 0.70, 0.66, 940.0),
        "wrong-invalid": (0.0, 0.90, 0.88, 1020.0),
    }
    valid_probability, agreement, probability, score = presets[quality]
    return CountTrainingCandidate(
        variant=variant,
        measure_count=max(1, int(count)),
        valid=rng.random() < valid_probability,
        agreement_ratio=_jitter(rng, agreement, 0.07, 0.0, 1.0),
        calibrated_probability=_jitter(rng, probability, 0.09, 0.0, 1.0),
        raw_score=_jitter(rng, score, 42.0, 500.0, 1100.0),
    )


def _build_group(
    task: tuple[int, int],
) -> tuple[
    list[list[float]],
    list[list[float]],
    list[int],
    list[int],
    list[int],
    list[int],
    int,
    str,
    int,
    int,
    float,
]:
    seed, group = task
    rng = random.Random(_seed(seed, group))
    kind = KINDS[group % len(KINDS)]
    truth = rng.randint(5, 112)
    direction = rng.choice((-1, 1))
    near = max(1, truth + direction)
    far = max(1, truth + direction * rng.randint(2, 7))
    layout_count = truth
    layout_confidence = _jitter(rng, 0.88, 0.08, 0.15, 0.995)
    specs: list[tuple[str, int, str]]

    if kind == "agreement":
        specs = [
            ("primary", truth, "true"),
            ("flat", truth, "true"),
            ("otsu", truth, "true-soft"),
            ("upscale", near, "wrong"),
        ]
    elif kind == "high-confidence-layout-wrong":
        layout_count = far
        layout_confidence = _jitter(rng, 0.965, 0.018, 0.90, 0.995)
        specs = [
            ("primary", truth, "true"),
            ("flat", truth, "true-soft"),
            ("otsu", truth, "true-soft"),
            ("staffnorm", near, "wrong-medium"),
        ]
    elif kind == "single-family-duplicate-trap":
        specs = [
            ("flat", near, "wrong-strong"),
            ("deblock", near, "wrong-strong"),
            ("primary", truth, "true-soft"),
            ("otsu", truth, "true-soft"),
            ("upscale", truth, "true-soft"),
        ]
    elif kind == "two-family-shared-error":
        layout_count = near if rng.random() < 0.6 else truth
        specs = [
            ("flat", near, "wrong-medium"),
            ("deblock", near, "wrong-medium"),
            ("otsu", near, "wrong-medium"),
            ("adaptive", near, "wrong-medium"),
            ("primary", truth, "true"),
            ("staffnorm", truth, "true-soft"),
        ]
    elif kind == "adjacent-count-split":
        layout_count = truth
        specs = [
            ("primary", truth, "true-soft"),
            ("flat", max(1, truth - 1), "wrong-medium"),
            ("otsu", truth + 1, "wrong-medium"),
            ("upscale", truth, "true-soft"),
            ("staffnorm", truth + 1, "wrong"),
        ]
    elif kind == "layout-only-rescue":
        layout_count = truth
        layout_confidence = _jitter(rng, 0.97, 0.015, 0.92, 0.995)
        specs = [
            ("primary", near, "wrong-medium"),
            ("flat", far, "wrong"),
            ("otsu", max(1, truth - direction), "wrong"),
        ]
    elif kind == "high-score-single-family-trap":
        specs = [
            ("primary", near, "wrong-strong"),
            ("flat", truth, "true-soft"),
            ("otsu", truth, "true-soft"),
            ("upscale", truth, "true-soft"),
        ]
    elif kind == "invalid-true-candidates":
        specs = [
            ("primary", truth, "true-invalid"),
            ("flat", truth, "true-invalid"),
            ("otsu", near, "wrong-strong"),
            ("upscale", truth, "true-soft"),
        ]
    elif kind == "candidate-majority-trap":
        specs = [
            ("flat", near, "wrong-strong"),
            ("deblock", near, "wrong-strong"),
            ("otsu", near, "wrong-medium"),
            ("adaptive", near, "wrong-medium"),
            ("primary", truth, "true"),
            ("staffnorm", truth, "true-soft"),
        ]
    elif kind == "independent-family-consensus":
        layout_count = near if rng.random() < 0.5 else truth
        specs = [
            ("primary", truth, "true-soft"),
            ("flat", truth, "true-soft"),
            ("otsu", truth, "true-soft"),
            ("upscale", truth, "true-soft"),
            ("adaptive", near, "wrong-strong"),
        ]
    elif kind == "large-layout-error":
        layout_count = far
        layout_confidence = _jitter(rng, 0.72, 0.12, 0.35, 0.95)
        specs = [
            ("primary", truth, "true"),
            ("flat", truth, "true-soft"),
            ("otsu", truth, "true-soft"),
            ("upscale", near, "wrong"),
        ]
    elif kind == "low-confidence-layout":
        layout_count = far
        layout_confidence = _jitter(rng, 0.35, 0.10, 0.08, 0.58)
        specs = [
            ("primary", truth, "true-soft"),
            ("flat", near, "wrong-medium"),
            ("otsu", truth, "true-soft"),
            ("adaptive", max(1, truth - direction), "wrong"),
            ("staffnorm", truth, "true-soft"),
        ]
    elif kind == "localized-correct-rescue":
        layout_count = near if rng.random() < 0.45 else truth
        layout_confidence = _jitter(rng, 0.72, 0.12, 0.35, 0.94)
        specs = [
            ("primary", truth, "true-soft"),
            ("flat", near, "wrong-medium"),
            ("otsu", near, "wrong-medium"),
            ("upscale", truth, "true-soft"),
            ("system_localized", truth, "true"),
        ]
    elif kind == "localized-wrong-isolation":
        specs = [
            ("primary", truth, "true"),
            ("flat", truth, "true-soft"),
            ("otsu", truth, "true-soft"),
            ("upscale", truth, "true-soft"),
            ("system_localized", near, "wrong-strong"),
        ]
    elif kind == "localized-invalid-trap":
        specs = [
            ("primary", truth, "true"),
            ("flat", truth, "true-soft"),
            ("otsu", truth, "true-soft"),
            ("upscale", truth, "true-soft"),
            ("system_localized", near, "wrong-invalid"),
        ]
    elif kind == "template-count-outlier":
        layout_count = truth
        specs = [
            ("primary", near, "wrong-strong"),
            ("flat", truth, "true-soft"),
            ("otsu", truth, "true-soft"),
            ("upscale", truth, "true-soft"),
            ("system_localized", truth, "true-soft"),
        ]
    else:  # invalid-sibling-family-trap
        layout_count = near if rng.random() < 0.45 else truth
        specs = [
            ("flat", near, "wrong-strong"),
            ("deblock", near, "wrong-invalid"),
            ("primary", truth, "true-soft"),
            ("otsu", truth, "true-soft"),
            ("upscale", truth, "true-soft"),
        ]

    protected_kinds = {
        "localized-correct-rescue",
        "localized-wrong-isolation",
        "localized-invalid-trap",
        "template-count-outlier",
        "invalid-sibling-family-trap",
    }
    # Deterministically vary legacy group size without deleting the defining evidence
    # of the new localisation and incomplete-family scenarios.
    if kind not in protected_kinds and len(specs) > 3 and rng.random() < 0.35:
        specs.pop(rng.randrange(len(specs)))
    if kind not in protected_kinds and len(specs) < 8 and rng.random() < 0.35:
        existing_variants = {item[0] for item in specs}
        choices = [variant for variant in VARIANTS if variant not in existing_variants]
        if choices:
            specs.append((rng.choice(choices), truth if rng.random() < 0.65 else near, "true-soft" if rng.random() < 0.65 else "wrong"))

    # Every decision group must contain at least one wrong option.  This invariant is
    # required by grouped ranking metrics and prevents random size variation from
    # turning a hard case into a trivial one-option sample.
    if len({count for _variant, count, _quality in specs}) < 2:
        existing_variants = {item[0] for item in specs}
        fallback_variant = next(
            (variant for variant in VARIANTS if variant not in existing_variants),
            "adaptive",
        )
        specs.append((fallback_variant, near, "wrong"))

    candidates = tuple(_candidate(rng, *spec) for spec in specs)
    bundle = build_measure_count_feature_bundle(
        layout_count=layout_count,
        layout_confidence=layout_confidence,
        candidates=candidates,
    )
    rows = [list(row.features) for row in bundle.rows]
    legacy = [list(row.features[: len(LEGACY_FEATURE_NAMES)]) for row in bundle.rows]
    labels = [int(row.count == truth) for row in bundle.rows]
    counts = [row.count for row in bundle.rows]
    family_supports = [row.family_support for row in bundle.rows]
    candidate_supports = [row.candidate_support for row in bundle.rows]
    return (
        rows,
        legacy,
        labels,
        counts,
        family_supports,
        candidate_supports,
        truth,
        kind,
        bundle.deterministic_count,
        layout_count,
        layout_confidence,
    )


def build_dataset(seed: int, groups: int, workers: int) -> MeasureCountDataset:
    if groups < len(KINDS) * 4:
        raise ValueError("measure-count training requires at least four groups per scenario kind")
    tasks = [(seed, group) for group in range(groups)]
    if workers <= 1:
        built = [_build_group(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
            built = list(executor.map(_build_group, tasks, chunksize=32))

    features: list[list[float]] = []
    legacy_features: list[list[float]] = []
    labels: list[int] = []
    group_ids: list[int] = []
    option_counts: list[int] = []
    truths: list[int] = []
    kinds: list[str] = []
    decisions: list[tuple[int, ...]] = []
    deterministic_counts: list[int] = []
    layout_counts: list[int] = []
    layout_confidences: list[float] = []
    family_supports: list[int] = []
    candidate_supports: list[int] = []
    for group, (
        rows,
        legacy,
        local_labels,
        counts,
        local_family_supports,
        local_candidate_supports,
        truth,
        kind,
        deterministic,
        layout,
        layout_confidence,
    ) in enumerate(built):
        start = len(features)
        features.extend(rows)
        legacy_features.extend(legacy)
        labels.extend(local_labels)
        group_ids.extend([group] * len(rows))
        option_counts.extend(counts)
        family_supports.extend(local_family_supports)
        candidate_supports.extend(local_candidate_supports)
        truths.extend([truth] * len(rows))
        kinds.extend([kind] * len(rows))
        decisions.append(tuple(range(start, start + len(rows))))
        deterministic_counts.append(deterministic)
        layout_counts.append(layout)
        layout_confidences.append(layout_confidence)

    return MeasureCountDataset(
        features=np.asarray(features, dtype=np.float64),
        legacy_features=np.asarray(legacy_features, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int32),
        groups=np.asarray(group_ids, dtype=np.int32),
        option_counts=np.asarray(option_counts, dtype=np.int32),
        truths=np.asarray(truths, dtype=np.int32),
        kinds=np.asarray(kinds, dtype="U40"),
        decision_groups=tuple(decisions),
        deterministic_counts=np.asarray(deterministic_counts, dtype=np.int32),
        layout_counts=np.asarray(layout_counts, dtype=np.int32),
        layout_confidences=np.asarray(layout_confidences, dtype=np.float64),
        family_supports=np.asarray(family_supports, dtype=np.int32),
        candidate_supports=np.asarray(candidate_supports, dtype=np.int32),
    )
