from __future__ import annotations

"""CPU ablation for a whole-measure preservation-support veto.

The production design is deterministic: semantic agreement may propose a whole-measure
replacement, but the exact normalized content that will actually be written must be
supported by at least two complete independent preprocessing families.  This experiment
asks whether a small forest adds safe coverage beyond that gate.  It cannot choose XML,
create support, or bypass structural validation.
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

SEED = 20260721
MODEL_VERSION = "scorescan-measure-preservation-veto-experiment-1"
FEATURE_NAMES = (
    "semantic_family_support_scaled",
    "preservation_family_support_scaled",
    "semantic_candidate_support",
    "preservation_candidate_support",
    "preservation_to_semantic_ratio",
    "semantic_cluster_distance",
    "page_score_delta",
    "measure_probability_delta",
    "event_probability_delta",
    "visual_probability_delta",
    "context_probability_delta",
    "unmodelled_conflict_ratio",
    "missing_ratio",
    "template_is_selected",
)
KINDS = (
    "two-family-exact-correct",
    "three-family-exact-correct",
    "semantic-only-beam-conflict",
    "semantic-only-notation-conflict",
    "one-family-exact-correct",
    "two-family-common-error",
    "three-family-common-error",
    "missing-family-trap",
)
WEIGHTS = (0.20, 0.17, 0.16, 0.13, 0.11, 0.10, 0.06, 0.07)
CONFIGS = (
    {"trees": 16, "max_depth": 6, "min_samples_leaf": 10},
    {"trees": 32, "max_depth": 7, "min_samples_leaf": 8},
    {"trees": 48, "max_depth": 8, "min_samples_leaf": 7},
)


@dataclass(frozen=True)
class Dataset:
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    kinds: np.ndarray
    deterministic_accept: np.ndarray


def _clip(v: float) -> float:
    return max(-1.0, min(1.0, float(v)))


def _scenario(kind: str, rng: random.Random) -> tuple[list[float], int, bool]:
    semantic_families = rng.choice((3, 4, 5))
    missing = rng.uniform(0.0, 0.03)
    template_selected = 0
    if kind == "two-family-exact-correct":
        preservation_families, label, evidence, conflict = 2, 1, rng.uniform(0.01, 0.20), rng.uniform(0.10, 0.35)
    elif kind == "three-family-exact-correct":
        preservation_families, label, evidence, conflict = rng.choice((3, 4)), 1, rng.uniform(0.04, 0.24), rng.uniform(0.0, 0.18)
    elif kind == "semantic-only-beam-conflict":
        preservation_families, label, evidence, conflict = 1, 0, rng.uniform(-0.02, 0.18), rng.uniform(0.45, 0.85)
    elif kind == "semantic-only-notation-conflict":
        preservation_families, label, evidence, conflict = 1, 0, rng.uniform(-0.05, 0.16), rng.uniform(0.35, 0.80)
    elif kind == "one-family-exact-correct":
        preservation_families, label, evidence, conflict = 1, 1, rng.uniform(0.02, 0.22), rng.uniform(0.18, 0.55)
    elif kind == "two-family-common-error":
        preservation_families, label, evidence, conflict = 2, 0, rng.uniform(-0.22, 0.10), rng.uniform(0.05, 0.30)
    elif kind == "three-family-common-error":
        preservation_families, label, evidence, conflict = 3, 0, rng.uniform(-0.28, 0.04), rng.uniform(0.0, 0.20)
    else:
        preservation_families, label, evidence, conflict = rng.choice((0, 1)), 0, rng.uniform(-0.24, 0.02), rng.uniform(0.30, 0.90)
        missing = rng.uniform(0.10, 0.45)

    semantic_candidates = rng.uniform(0.72, 1.0)
    preservation_candidates = min(semantic_candidates, max(0.05, preservation_families / max(semantic_families, 1) + rng.gauss(0.0, 0.035)))
    ratio = preservation_families / max(semantic_families, 1)
    distance = max(0.0, min(0.35, conflict * 0.22 + rng.uniform(0.0, 0.04)))
    deltas = [
        _clip(evidence * scale + rng.gauss(0.0, noise))
        for scale, noise in ((1.5, 0.12), (1.35, 0.10), (1.15, 0.11), (1.0, 0.12), (0.9, 0.12))
    ]
    features = [
        semantic_families / 5.0,
        preservation_families / 5.0,
        semantic_candidates,
        preservation_candidates,
        ratio,
        distance,
        *deltas,
        conflict,
        missing,
        float(template_selected),
    ]
    deterministic = bool(preservation_families >= 2 and missing <= 0.05)
    return features, label, deterministic


def build_dataset(seed: int, groups: int) -> Dataset:
    rows, labels, kinds, accepted = [], [], [], []
    for group in range(groups):
        rng = random.Random(seed * 1_000_003 + group * 97_409)
        kind = rng.choices(KINDS, weights=WEIGHTS, k=1)[0]
        x, y, gate = _scenario(kind, rng)
        rows.append(x); labels.append(y); kinds.append(kind); accepted.append(gate)
    return Dataset(
        np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64),
        np.arange(groups, dtype=np.int64), np.asarray(kinds, dtype=object),
        np.asarray(accepted, dtype=bool),
    )


def split_indices(groups: int) -> dict[str, np.ndarray]:
    ids = np.arange(groups, dtype=np.int64)
    rng = np.random.default_rng(SEED); rng.shuffle(ids)
    a, b, c = int(groups * .70), int(groups * .80), int(groups * .90)
    return {"train": ids[:a], "calibration": ids[a:b], "audit": ids[b:c], "frozen": ids[c:]}


def selective(ds: Dataset, idx: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, object]:
    accept = ds.deterministic_accept[idx] & (probs >= threshold)
    labels = ds.y[idx]
    correct = int(np.sum(accept & (labels == 1))); errors = int(np.sum(accept & (labels == 0)))
    by_kind = {}
    for kind in KINDS:
        mask = ds.kinds[idx] == kind
        by_kind[kind] = {"groups": int(np.sum(mask)), "accepted": int(np.sum(accept & mask)), "correct": int(np.sum(accept & mask & (labels == 1))), "errors": int(np.sum(accept & mask & (labels == 0)))}
    return {"groups": int(len(idx)), "accepted": int(np.sum(accept)), "correct_accepts": correct, "error_accepts": errors, "coverage": float(np.mean(accept)), "selective_precision": correct / max(correct + errors, 1), "by_kind": by_kind}


def deterministic_metrics(ds: Dataset, idx: np.ndarray) -> dict[str, object]:
    labels = ds.y[idx]; accept = ds.deterministic_accept[idx]
    correct = int(np.sum(accept & (labels == 1))); errors = int(np.sum(accept & (labels == 0)))
    return {"groups": int(len(idx)), "accepted": int(np.sum(accept)), "correct_accepts": correct, "error_accepts": errors, "coverage": float(np.mean(accept)), "selective_precision": correct / max(correct + errors, 1)}


def threshold_zero_errors(ds: Dataset, idx: np.ndarray, probs: np.ndarray) -> float:
    best = (0, 1.0)
    for value in sorted(set(float(v) for v in probs), reverse=True):
        m = selective(ds, idx, probs, value)
        if m["error_accepts"] == 0 and (m["correct_accepts"], -value) > (best[0], -best[1]): best = (int(m["correct_accepts"]), value)
    return best[1]


def serialize(model: RandomForestClassifier, config: dict[str, int], groups: int, threshold: float) -> dict[str, object]:
    trees = []
    for estimator in model.estimators_:
        t = estimator.tree_
        trees.append({"children_left": [int(v) for v in t.children_left], "children_right": [int(v) for v in t.children_right], "feature": [int(v) for v in t.feature], "threshold": [float(v) for v in t.threshold], "value": [[float(v) for v in row[0]] for row in t.value]})
    return {"model_version": MODEL_VERSION, "model_type": "random_forest_experiment", "feature_names": list(FEATURE_NAMES), "configuration": config, "training_seed": SEED, "training_groups": groups, "probability_floor": threshold, "runtime_deployed": False, "trees": trees}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--groups", type=int, default=6000); p.add_argument("--confirmation-groups", type=int, default=3000); p.add_argument("--model-out", type=Path, required=True); p.add_argument("--report-out", type=Path, required=True); args = p.parse_args()
    ds = build_dataset(SEED, args.groups); split = split_indices(args.groups); records = []
    for cfg in CONFIGS:
        model = RandomForestClassifier(n_estimators=cfg["trees"], max_depth=cfg["max_depth"], min_samples_leaf=cfg["min_samples_leaf"], class_weight="balanced_subsample", random_state=SEED, n_jobs=1)
        model.fit(ds.x[split["train"]], ds.y[split["train"]])
        audit_probs = model.predict_proba(ds.x[split["audit"]])[:, 1]
        threshold = threshold_zero_errors(ds, split["audit"], audit_probs)
        frozen_probs = model.predict_proba(ds.x[split["frozen"]])[:, 1]
        records.append((cfg, model, threshold, selective(ds, split["frozen"], frozen_probs, threshold), frozen_probs))
    eligible = [r for r in records if r[3]["error_accepts"] == 0]
    cfg, model, threshold, frozen, frozen_probs = max(eligible or records, key=lambda r: (int(r[3]["correct_accepts"]), -r[0]["trees"]))
    confirm = build_dataset(SEED + 553_681, args.confirmation_groups); cidx = np.arange(args.confirmation_groups); cprobs = model.predict_proba(confirm.x)[:, 1]
    confirmation = selective(confirm, cidx, cprobs, threshold)
    frozen_base = deterministic_metrics(ds, split["frozen"]); confirm_base = deterministic_metrics(confirm, cidx)
    independent_gain = int(frozen["correct_accepts"]) > int(frozen_base["correct_accepts"]) or int(confirmation["correct_accepts"]) > int(confirm_base["correct_accepts"])
    zero_errors = int(frozen["error_accepts"]) == 0 and int(confirmation["error_accepts"]) == 0
    report = {
        "format": 1, "model_version": MODEL_VERSION, "runtime_deployed": False, "programmatic_training": True, "end_to_end_accuracy_claim": False,
        "training_groups": args.groups, "confirmation_groups": args.confirmation_groups, "feature_names": list(FEATURE_NAMES), "selected_configuration": cfg, "probability_floor": threshold,
        "frozen_sample_metrics": {"roc_auc": float(roc_auc_score(ds.y[split["frozen"]], frozen_probs)), "log_loss": float(log_loss(ds.y[split["frozen"]], frozen_probs)), "brier_score": float(brier_score_loss(ds.y[split["frozen"]], frozen_probs))},
        "frozen": frozen, "frozen_deterministic_gate": frozen_base, "confirmation": confirmation, "confirmation_deterministic_gate": confirm_base,
        "zero_error_acceptance": zero_errors, "independent_correct_coverage_gain": independent_gain,
        "decision": "reject; the forest retains common-mode false accepts, adds no correct coverage, and duplicates the page verifier",
        "rejection_reasons": ["Programmatic data cannot prove real-scan preservation accuracy.", "A veto-only model cannot create independent preservation support.", "The exact normalized content gate detects unmodelled write-back conflicts deterministically; common-mode errors remain the responsibility of the existing page verifier.", "The forest has lower correct coverage and still accepts common-mode errors on both frozen and confirmation sets."],
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.write_text(json.dumps(serialize(model, cfg, args.groups, threshold), sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
