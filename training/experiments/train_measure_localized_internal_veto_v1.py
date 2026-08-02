from __future__ import annotations

"""Train and audit a CPU veto for measure-localised internal agreement.

This experiment is intentionally separate from the runtime model registry.  The accepted
0.25 design uses a deterministic exact majority across three related crop treatments and
then the existing page-level selection-risk verifier.  This script asks whether another
small forest adds independent selective safety.  It may only veto a local-family proposal;
it cannot create agreement, select a different XML, or count subvariants as independent
families.

The data are programmatic grouped perturbations, not real scan labels.  A model is exported
only for reproducibility and remains rejected unless a later real frozen benchmark proves
independent value beyond the established verifier.
"""

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

SEED = 20260721
MODEL_VERSION = "scorescan-measure-localized-internal-veto-experiment-1"
FEATURE_NAMES = (
    "valid_variant_count",
    "winning_exact_support",
    "runner_up_support",
    "winning_margin",
    "primary_in_winner",
    "all_variants_valid",
    "note_count",
    "note_count_spread",
    "local_rhythm_issue_max",
    "page_score_delta",
    "measure_probability_delta",
    "event_probability_delta",
    "visual_probability_delta",
    "context_probability_delta",
)
KINDS = (
    "all-correct-agree",
    "one-wrong-two-correct",
    "one-invalid-two-correct",
    "primary-wrong-two-correct",
    "two-wrong-one-correct",
    "all-wrong-agree",
    "partial-common-error",
    "three-way-split",
)
WEIGHTS = (0.22, 0.18, 0.12, 0.10, 0.12, 0.09, 0.09, 0.08)
CONFIGS = (
    {"trees": 32, "max_depth": 7, "min_samples_leaf": 8},
    {"trees": 48, "max_depth": 8, "min_samples_leaf": 7},
    {"trees": 64, "max_depth": 9, "min_samples_leaf": 6},
)


@dataclass(frozen=True)
class Dataset:
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    kinds: np.ndarray
    old_accept: np.ndarray
    majority_accept: np.ndarray


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _scenario(kind: str, rng: random.Random) -> tuple[list[float], int, bool, bool]:
    if kind == "all-correct-agree":
        valid, win, runner, primary, label = 3, 3, 0, 1, 1
        evidence = rng.uniform(0.02, 0.20)
    elif kind == "one-wrong-two-correct":
        valid, win, runner, primary, label = 3, 2, 1, int(rng.random() < 0.67), 1
        evidence = rng.uniform(0.00, 0.16)
    elif kind == "one-invalid-two-correct":
        valid, win, runner, primary, label = 2, 2, 0, int(rng.random() < 0.72), 1
        evidence = rng.uniform(-0.02, 0.14)
    elif kind == "primary-wrong-two-correct":
        valid, win, runner, primary, label = 3, 2, 1, 0, 1
        evidence = rng.uniform(0.01, 0.18)
    elif kind == "two-wrong-one-correct":
        valid, win, runner, primary, label = 3, 2, 1, int(rng.random() < 0.66), 0
        evidence = rng.uniform(-0.24, 0.05)
    elif kind == "all-wrong-agree":
        valid, win, runner, primary, label = 3, 3, 0, 1, 0
        evidence = rng.uniform(-0.28, 0.02)
    elif kind == "partial-common-error":
        valid, win, runner, primary, label = rng.choice((2, 3)), 2, 0, 1, 0
        evidence = rng.uniform(-0.32, -0.02)
    else:
        valid, win, runner, primary, label = 3, 1, 1, 1, 0
        evidence = rng.uniform(-0.12, 0.10)

    note_count = rng.randint(1, 18)
    note_spread = 0 if win == valid else rng.randint(1, max(1, min(5, note_count)))
    rhythm_max = 0 if label else int(rng.random() < 0.32)
    style = rng.gauss(0.0, 0.035)
    page_delta = evidence * 80.0 + rng.gauss(0.0, 5.0)
    measure_delta = _clip(0.52 + evidence * 1.5 + style, 0.0, 1.0) - 0.5
    event_delta = _clip(0.50 + evidence * 1.3 + rng.gauss(0.0, 0.05), 0.0, 1.0) - 0.5
    visual_delta = _clip(0.50 + evidence * 1.1 + rng.gauss(0.0, 0.06), 0.0, 1.0) - 0.5
    context_delta = _clip(0.50 + evidence * 1.0 + rng.gauss(0.0, 0.06), 0.0, 1.0) - 0.5
    margin = win - runner
    features = [
        float(valid),
        float(win),
        float(runner),
        float(margin),
        float(primary),
        float(valid == 3),
        float(note_count),
        float(note_spread),
        float(rhythm_max),
        float(page_delta),
        float(measure_delta),
        float(event_delta),
        float(visual_delta),
        float(context_delta),
    ]
    old_accept = bool(primary and valid >= 1)
    majority_accept = bool(valid >= 2 and win >= 2 and margin >= 1)
    return features, label, old_accept, majority_accept


def build_dataset(seed: int, groups: int) -> Dataset:
    rows: list[list[float]] = []
    labels: list[int] = []
    ids: list[int] = []
    kinds: list[str] = []
    old: list[bool] = []
    majority: list[bool] = []
    for group in range(groups):
        rng = random.Random(seed * 1_000_003 + group * 97_409)
        kind = rng.choices(KINDS, weights=WEIGHTS, k=1)[0]
        features, label, old_accept, majority_accept = _scenario(kind, rng)
        rows.append(features)
        labels.append(label)
        ids.append(group)
        kinds.append(kind)
        old.append(old_accept)
        majority.append(majority_accept)
    return Dataset(
        x=np.asarray(rows, dtype=np.float64),
        y=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(ids, dtype=np.int64),
        kinds=np.asarray(kinds, dtype=object),
        old_accept=np.asarray(old, dtype=bool),
        majority_accept=np.asarray(majority, dtype=bool),
    )


def split_indices(dataset: Dataset, seed: int) -> dict[str, np.ndarray]:
    ids = np.unique(dataset.groups)
    rng = np.random.default_rng(seed)
    ids = ids.copy()
    rng.shuffle(ids)
    n = len(ids)
    a, b, c = int(n * 0.70), int(n * 0.80), int(n * 0.90)
    parts = {"train": ids[:a], "calibration": ids[a:b], "audit": ids[b:c], "frozen": ids[c:]}
    return {name: np.flatnonzero(np.isin(dataset.groups, values)) for name, values in parts.items()}


def fit_model(dataset: Dataset, indices: np.ndarray, config: dict[str, int]) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=config["trees"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=1,
    )
    model.fit(dataset.x[indices], dataset.y[indices])
    return model


def calibrate(model: RandomForestClassifier, dataset: Dataset, indices: np.ndarray) -> LogisticRegression:
    raw = model.predict_proba(dataset.x[indices])[:, 1].reshape(-1, 1)
    result = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    result.fit(raw, dataset.y[indices])
    return result


def probabilities(model: RandomForestClassifier, cal: LogisticRegression, x: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(x)[:, 1].reshape(-1, 1)
    return cal.predict_proba(raw)[:, 1]


def selective_metrics(dataset: Dataset, indices: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, object]:
    labels = dataset.y[indices]
    majority = dataset.majority_accept[indices]
    accepted = majority & (probs >= threshold)
    correct = int(np.sum(accepted & (labels == 1)))
    errors = int(np.sum(accepted & (labels == 0)))
    by_kind: dict[str, dict[str, int]] = {}
    for kind in KINDS:
        mask = dataset.kinds[indices] == kind
        by_kind[kind] = {
            "groups": int(np.sum(mask)),
            "accepted": int(np.sum(accepted & mask)),
            "correct": int(np.sum(accepted & mask & (labels == 1))),
            "errors": int(np.sum(accepted & mask & (labels == 0))),
        }
    return {
        "groups": int(len(indices)),
        "accepted": int(np.sum(accepted)),
        "correct_accepts": correct,
        "error_accepts": errors,
        "selective_precision": correct / max(correct + errors, 1),
        "coverage": float(np.mean(accepted)),
        "by_kind": by_kind,
    }


def baseline_metrics(dataset: Dataset, indices: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    labels = dataset.y[indices]
    accepted = mask[indices]
    correct = int(np.sum(accepted & (labels == 1)))
    errors = int(np.sum(accepted & (labels == 0)))
    return {
        "groups": int(len(indices)),
        "accepted": int(np.sum(accepted)),
        "correct_accepts": correct,
        "error_accepts": errors,
        "selective_precision": correct / max(correct + errors, 1),
        "coverage": float(np.mean(accepted)),
    }


def choose_threshold(dataset: Dataset, indices: np.ndarray, probs: np.ndarray) -> float:
    best: tuple[int, float] | None = None
    for value in sorted(set(float(v) for v in probs), reverse=True):
        metrics = selective_metrics(dataset, indices, probs, value)
        if int(metrics["error_accepts"]) != 0:
            continue
        key = (int(metrics["correct_accepts"]), -value)
        if best is None or key > (best[0], -best[1]):
            best = (key[0], value)
    return best[1] if best is not None else 1.0


def sample_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float | int]:
    return {
        "samples": int(len(labels)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "log_loss": float(log_loss(labels, probs)),
        "brier_score": float(brier_score_loss(labels, probs)),
    }


def serialize(model: RandomForestClassifier, cal: LogisticRegression, config: dict[str, int], groups: int) -> dict[str, object]:
    trees: list[dict[str, object]] = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        trees.append({
            "children_left": [int(v) for v in tree.children_left],
            "children_right": [int(v) for v in tree.children_right],
            "feature": [int(v) for v in tree.feature],
            "threshold": [float(v) for v in tree.threshold],
            "value": [[float(v) for v in row[0]] for row in tree.value],
        })
    return {
        "model_version": MODEL_VERSION,
        "model_type": "random_forest_experiment",
        "feature_names": list(FEATURE_NAMES),
        "configuration": dict(config),
        "training_seed": SEED,
        "training_groups": groups,
        "trees": trees,
        "calibration_intercept": float(cal.intercept_[0]),
        "calibration_slope": float(cal.coef_[0, 0]),
        "runtime_deployed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=int, default=6000)
    parser.add_argument("--confirmation-groups", type=int, default=3000)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()

    dataset = build_dataset(SEED, args.groups)
    split = split_indices(dataset, SEED)
    records: list[tuple[dict[str, int], RandomForestClassifier, LogisticRegression, float, dict[str, object]]] = []
    for config in CONFIGS:
        model = fit_model(dataset, split["train"], config)
        cal = calibrate(model, dataset, split["calibration"])
        audit_probs = probabilities(model, cal, dataset.x[split["audit"]])
        threshold = choose_threshold(dataset, split["audit"], audit_probs)
        frozen_probs = probabilities(model, cal, dataset.x[split["frozen"]])
        frozen = selective_metrics(dataset, split["frozen"], frozen_probs, threshold)
        records.append((config, model, cal, threshold, frozen))

    eligible = [item for item in records if int(item[4]["error_accepts"]) == 0]
    selected = max(eligible or records, key=lambda item: (int(item[4]["correct_accepts"]), -item[0]["trees"]))
    config, model, cal, threshold, frozen = selected
    frozen_probs = probabilities(model, cal, dataset.x[split["frozen"]])

    confirmation = build_dataset(SEED + 91_117, args.confirmation_groups)
    confirmation_indices = np.arange(args.confirmation_groups, dtype=np.int64)
    confirmation_probs = probabilities(model, cal, confirmation.x)
    confirmation_metrics = selective_metrics(confirmation, confirmation_indices, confirmation_probs, threshold)

    model_payload = serialize(model, cal, config, args.groups)
    model_payload["probability_floor"] = threshold
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.write_text(json.dumps(model_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")

    report = {
        "format": 1,
        "model_version": MODEL_VERSION,
        "runtime_deployed": False,
        "rejection_reason": (
            "programmatic-only labels and responsibility overlap with the deployed page-level selection-risk verifier; "
            "the deterministic three-subvariant exact-majority gate is deployed instead"
        ),
        "dataset": {
            "groups": args.groups,
            "confirmation_groups": args.confirmation_groups,
            "seed": SEED,
            "feature_names": list(FEATURE_NAMES),
            "kinds": list(KINDS),
            "programmatic": True,
        },
        "selected_configuration": config,
        "probability_floor": threshold,
        "frozen_sample": sample_metrics(dataset.y[split["frozen"]], frozen_probs),
        "frozen": frozen,
        "confirmation": confirmation_metrics,
        "old_single_pass_baseline": {
            "frozen": baseline_metrics(dataset, split["frozen"], dataset.old_accept),
            "confirmation": baseline_metrics(confirmation, confirmation_indices, confirmation.old_accept),
        },
        "deterministic_internal_majority": {
            "frozen": baseline_metrics(dataset, split["frozen"], dataset.majority_accept),
            "confirmation": baseline_metrics(confirmation, confirmation_indices, confirmation.majority_accept),
        },
        "configuration_ablation": [
            {
                "configuration": item[0],
                "probability_floor": item[3],
                "frozen": item[4],
            }
            for item in records
        ],
        "limitations": [
            "All labels are programmatic grouped perturbations, not real scanned pages.",
            "Subvariants remain correlated and never count as independent candidate families.",
            "The experiment is veto-only and is intentionally absent from model_manifest.json.",
        ],
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
