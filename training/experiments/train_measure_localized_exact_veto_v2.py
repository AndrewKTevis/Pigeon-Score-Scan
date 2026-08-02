from __future__ import annotations

"""Train a CPU veto experiment for the strict local splice-content gate.

The accepted runtime design is deterministic: two of three related crop treatments must
produce exactly the same normalized spliceable MusicXML, then the resulting one-family
candidate still passes the existing page-level selection-risk verifier.  This experiment
asks whether another forest can safely veto common-mode errors using only internal-local
features.  It cannot create agreement or choose alternate XML and is not deployed unless
it demonstrates independent value beyond the existing verifier on real frozen scans.
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

SEED = 20260721
MODEL_VERSION = "scorescan-measure-localized-exact-veto-experiment-2"
FEATURE_NAMES = (
    "valid_variant_count",
    "strict_winner_support",
    "strict_runner_up_support",
    "strict_margin",
    "semantic_winner_support",
    "semantic_runner_up_support",
    "semantic_minus_strict_support",
    "primary_in_strict_winner",
    "all_variants_valid",
    "note_count_scaled",
    "note_count_spread_scaled",
    "notation_disagreement_ratio",
    "page_score_delta_scaled",
    "measure_probability_delta",
    "event_probability_delta",
    "visual_probability_delta",
    "context_probability_delta",
)
KINDS = (
    "all-exact-correct",
    "one-substantive-split-two-correct",
    "one-invalid-two-correct",
    "two-correlated-wrong-one-correct",
    "all-correlated-wrong",
    "semantic-only-three-way-split",
    "mid-attribute-invalid",
    "only-one-valid",
)
WEIGHTS = (0.24, 0.20, 0.13, 0.13, 0.09, 0.09, 0.06, 0.06)
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
    deterministic_accept: np.ndarray


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _scenario(kind: str, rng: random.Random) -> tuple[list[float], int, bool]:
    if kind == "all-exact-correct":
        valid, strict, runner, semantic, semantic_runner, label = 3, 3, 0, 3, 0, 1
        evidence = rng.uniform(0.03, 0.20)
        primary = 1
    elif kind == "one-substantive-split-two-correct":
        valid, strict, runner, semantic, semantic_runner, label = 3, 2, 1, 3, 0, 1
        evidence = rng.uniform(0.00, 0.17)
        primary = int(rng.random() < 0.68)
    elif kind == "one-invalid-two-correct":
        valid, strict, runner, semantic, semantic_runner, label = 2, 2, 0, 2, 0, 1
        evidence = rng.uniform(-0.02, 0.14)
        primary = int(rng.random() < 0.72)
    elif kind == "two-correlated-wrong-one-correct":
        valid, strict, runner, semantic, semantic_runner, label = 3, 2, 1, 3, 0, 0
        evidence = rng.uniform(-0.18, 0.09)
        primary = int(rng.random() < 0.67)
    elif kind == "all-correlated-wrong":
        valid, strict, runner, semantic, semantic_runner, label = 3, 3, 0, 3, 0, 0
        evidence = rng.uniform(-0.24, 0.05)
        primary = 1
    elif kind == "semantic-only-three-way-split":
        valid, strict, runner, semantic, semantic_runner, label = 3, 1, 1, 3, 0, 0
        evidence = rng.uniform(-0.10, 0.12)
        primary = 1
    elif kind == "mid-attribute-invalid":
        valid, strict, runner, semantic, semantic_runner, label = rng.choice((0, 1)), 0, 0, 0, 0, 0
        evidence = rng.uniform(-0.28, -0.03)
        primary = 0
    else:
        valid, strict, runner, semantic, semantic_runner, label = 1, 1, 0, 1, 0, 0
        evidence = rng.uniform(-0.18, 0.08)
        primary = 1

    note_count = rng.randint(1, 20)
    note_spread = 0 if strict == valid else rng.randint(0, min(5, note_count))
    notation_disagreement = max(0, semantic - strict) / max(semantic, 1)
    page_delta = _clip(evidence * 1.8 + rng.gauss(0.0, 0.12))
    measure_delta = _clip(evidence * 1.5 + rng.gauss(0.0, 0.09))
    event_delta = _clip(evidence * 1.3 + rng.gauss(0.0, 0.10))
    visual_delta = _clip(evidence * 1.15 + rng.gauss(0.0, 0.11))
    context_delta = _clip(evidence * 1.0 + rng.gauss(0.0, 0.11))
    features = [
        valid / 3.0,
        strict / 3.0,
        runner / 3.0,
        (strict - runner) / 3.0,
        semantic / 3.0,
        semantic_runner / 3.0,
        max(0, semantic - strict) / 3.0,
        float(primary),
        float(valid == 3),
        min(note_count, 20) / 20.0,
        min(note_spread, 5) / 5.0,
        notation_disagreement,
        page_delta,
        measure_delta,
        event_delta,
        visual_delta,
        context_delta,
    ]
    deterministic = bool(valid >= 2 and strict >= 2 and strict - runner >= 1)
    return features, label, deterministic


def build_dataset(seed: int, groups: int) -> Dataset:
    rows: list[list[float]] = []
    labels: list[int] = []
    kinds: list[str] = []
    deterministic: list[bool] = []
    for group in range(groups):
        rng = random.Random(seed * 1_000_003 + group * 97_409)
        kind = rng.choices(KINDS, weights=WEIGHTS, k=1)[0]
        features, label, accepted = _scenario(kind, rng)
        rows.append(features)
        labels.append(label)
        kinds.append(kind)
        deterministic.append(accepted)
    return Dataset(
        x=np.asarray(rows, dtype=np.float64),
        y=np.asarray(labels, dtype=np.int64),
        groups=np.arange(groups, dtype=np.int64),
        kinds=np.asarray(kinds, dtype=object),
        deterministic_accept=np.asarray(deterministic, dtype=bool),
    )


def split_indices(dataset: Dataset, seed: int) -> dict[str, np.ndarray]:
    ids = dataset.groups.copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    a, b, c = int(n * 0.70), int(n * 0.80), int(n * 0.90)
    return {
        "train": ids[:a],
        "calibration": ids[a:b],
        "audit": ids[b:c],
        "frozen": ids[c:],
    }


def fit(dataset: Dataset, indices: np.ndarray, config: dict[str, int]) -> RandomForestClassifier:
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


def calibrator(model: RandomForestClassifier, dataset: Dataset, indices: np.ndarray) -> LogisticRegression:
    raw = model.predict_proba(dataset.x[indices])[:, 1].reshape(-1, 1)
    result = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    result.fit(raw, dataset.y[indices])
    return result


def probabilities(model: RandomForestClassifier, cal: LogisticRegression, x: np.ndarray) -> np.ndarray:
    return cal.predict_proba(model.predict_proba(x)[:, 1].reshape(-1, 1))[:, 1]


def selective(dataset: Dataset, indices: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, object]:
    labels = dataset.y[indices]
    accepted = dataset.deterministic_accept[indices] & (probs >= threshold)
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


def baseline(dataset: Dataset, indices: np.ndarray) -> dict[str, object]:
    labels = dataset.y[indices]
    accepted = dataset.deterministic_accept[indices]
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


def threshold_for_zero_errors(dataset: Dataset, indices: np.ndarray, probs: np.ndarray) -> float:
    best: tuple[int, float] | None = None
    for value in sorted(set(float(item) for item in probs), reverse=True):
        metrics = selective(dataset, indices, probs, value)
        if int(metrics["error_accepts"]) != 0:
            continue
        key = (int(metrics["correct_accepts"]), -value)
        if best is None or key > (best[0], -best[1]):
            best = (key[0], value)
    return best[1] if best is not None else 1.0


def serialize(model: RandomForestClassifier, cal: LogisticRegression, config: dict[str, int], groups: int) -> dict[str, object]:
    trees: list[dict[str, object]] = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        trees.append(
            {
                "children_left": [int(v) for v in tree.children_left],
                "children_right": [int(v) for v in tree.children_right],
                "feature": [int(v) for v in tree.feature],
                "threshold": [float(v) for v in tree.threshold],
                "value": [[float(v) for v in row[0]] for row in tree.value],
            }
        )
    return {
        "model_version": MODEL_VERSION,
        "model_type": "random_forest_experiment",
        "feature_names": list(FEATURE_NAMES),
        "configuration": config,
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
    records = []
    for config in CONFIGS:
        model = fit(dataset, split["train"], config)
        cal = calibrator(model, dataset, split["calibration"])
        audit_probs = probabilities(model, cal, dataset.x[split["audit"]])
        threshold = threshold_for_zero_errors(dataset, split["audit"], audit_probs)
        frozen_probs = probabilities(model, cal, dataset.x[split["frozen"]])
        records.append((config, model, cal, threshold, selective(dataset, split["frozen"], frozen_probs, threshold)))
    eligible = [record for record in records if int(record[4]["error_accepts"]) == 0]
    selected = max(eligible or records, key=lambda record: (int(record[4]["correct_accepts"]), -record[0]["trees"]))
    config, model, cal, threshold, frozen = selected

    confirmation = build_dataset(SEED + 191_117, args.confirmation_groups)
    confirmation_indices = np.arange(args.confirmation_groups, dtype=np.int64)
    confirmation_probs = probabilities(model, cal, confirmation.x)
    confirmation_metrics = selective(confirmation, confirmation_indices, confirmation_probs, threshold)

    frozen_probs = probabilities(model, cal, dataset.x[split["frozen"]])
    report = {
        "format": 1,
        "model_version": MODEL_VERSION,
        "runtime_deployed": False,
        "programmatic_training": True,
        "end_to_end_accuracy_claim": False,
        "training_groups": args.groups,
        "confirmation_groups": args.confirmation_groups,
        "feature_names": list(FEATURE_NAMES),
        "selected_configuration": config,
        "probability_floor": threshold,
        "frozen_sample_metrics": {
            "roc_auc": float(roc_auc_score(dataset.y[split["frozen"]], frozen_probs)),
            "log_loss": float(log_loss(dataset.y[split["frozen"]], frozen_probs)),
            "brier_score": float(brier_score_loss(dataset.y[split["frozen"]], frozen_probs)),
        },
        "frozen": frozen,
        "frozen_deterministic_baseline": baseline(dataset, split["frozen"]),
        "confirmation": confirmation_metrics,
        "confirmation_deterministic_baseline": baseline(confirmation, confirmation_indices),
        "decision": "reject; deterministic exact-content gate plus production page verifier remains simpler and authoritative",
        "rejection_reasons": [
            "The training data are programmatic rather than independent real scan labels.",
            "The model can only remove deterministic coverage and overlaps the existing page-level selection-risk verifier.",
            "Strict normalized XML equality already detects the newly identified beam/stem/unmodelled-notation disagreement without learned authority.",
        ],
    }
    payload = serialize(model, cal, config, args.groups)
    payload["probability_floor"] = threshold
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
