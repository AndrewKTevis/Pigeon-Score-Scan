from __future__ import annotations

"""Train the CPU visual/semantic measure compatibility model.

The model receives one immutable source-crop evidence vector and one MusicXML measure
candidate.  It may judge visible notation compatibility (density, hollow noteheads,
accidentals, compact marks and staff-normalised event profiles).  Only notation
changes that are not rendered by this evidence layer remain labelled compatible.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tree_export import deployed_forest_probabilities, serialize_probability_forest  # noqa: E402
from visual_training_data import KINDS, VisualDataset, build_dataset  # noqa: E402
from scorescan.tree_model import gradient_boosting_probability  # noqa: E402
from scorescan.visual_evidence import FEATURE_NAMES, LEGACY_FEATURE_NAMES, V3_FEATURE_NAMES  # noqa: E402

MODEL_VERSION = "scorescan-visual-measure-calibrator-4"
SEED = 20260720
CONFIGS = (
    {"estimator": "random_forest", "trees": 32, "max_depth": 8, "min_samples_leaf": 6, "max_features": "sqrt"},
    {"estimator": "extra_trees", "trees": 48, "max_depth": 8, "min_samples_leaf": 6, "max_features": "sqrt"},
    {"estimator": "extra_trees", "trees": 64, "max_depth": 10, "min_samples_leaf": 4, "max_features": 0.7},
    {"estimator": "extra_trees", "trees": 96, "max_depth": 10, "min_samples_leaf": 4, "max_features": 0.7},
)


def _split_groups(group_ids: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    groups = sorted(set(int(value) for value in group_ids.tolist()))
    random.Random(seed).shuffle(groups)
    total = len(groups)
    train_end = int(round(total * 0.70))
    calibration_end = int(round(total * 0.80))
    audit_end = int(round(total * 0.90))
    partitions = {
        "train": set(groups[:train_end]),
        "calibration": set(groups[train_end:calibration_end]),
        "audit": set(groups[calibration_end:audit_end]),
        "frozen": set(groups[audit_end:]),
    }
    return {
        name: np.flatnonzero(np.isin(group_ids, np.asarray(sorted(values), dtype=group_ids.dtype)))
        for name, values in partitions.items()
    }


def _fit_forest(
    values: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    config: dict[str, object],
    seed: int,
) -> RandomForestClassifier | ExtraTreesClassifier:
    common = dict(
        n_estimators=int(config["trees"]),
        max_depth=int(config["max_depth"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        max_features=config["max_features"],
        n_jobs=1,
        random_state=seed,
    )
    if config["estimator"] == "extra_trees":
        model = ExtraTreesClassifier(
            **common,
            class_weight="balanced",
            bootstrap=False,
        )
    else:
        model = RandomForestClassifier(
            **common,
            class_weight="balanced_subsample",
            bootstrap=True,
        )
    model.fit(values[indices], labels[indices])
    return model


def _calibration(model: object, values: np.ndarray, labels: np.ndarray, indices: np.ndarray, seed: int) -> LogisticRegression:
    raw = model.predict_proba(values[indices])[:, 1].reshape(-1, 1)
    calibrator = LogisticRegression(
        C=1000.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=seed,
    )
    calibrator.fit(raw, labels[indices])
    return calibrator


def _payload(
    model: object,
    calibrator: LogisticRegression,
    *,
    feature_names: tuple[str, ...],
    config: dict[str, object],
    groups: int,
    seed: int,
    model_version: str = MODEL_VERSION,
) -> dict[str, object]:
    return {
        "model_version": model_version,
        "model_type": "random_forest",
        "feature_names": list(feature_names),
        "trees": serialize_probability_forest(model),
        "calibration_intercept": float(calibrator.intercept_[0]),
        "calibration_slope": float(calibrator.coef_[0, 0]),
        "training_seed": seed,
        "training_groups": groups,
        "configuration": dict(config),
        "target_definition": (
            "exact and visually unrendered notation changes are compatible; pitch-profile, onset-profile, "
            "accidental, compact-mark, open-notehead, event-order, local attachment and gross-density "
            "mismatches are incompatible"
        ),
    }


def _probabilities(payload: dict[str, object], values: np.ndarray) -> np.ndarray:
    return deployed_forest_probabilities(payload, values)


def _legacy_probabilities(payload: dict[str, object], values: np.ndarray) -> np.ndarray:
    model_type = str(payload.get("model_type", ""))
    if model_type == "random_forest":
        return deployed_forest_probabilities(payload, values)
    if model_type != "gradient_boosting":
        raise ValueError(f"unsupported visual baseline model type: {model_type}")
    return np.asarray(
        [
            gradient_boosting_probability(
                row,
                intercept=float(payload.get("intercept", 0.0)),
                learning_rate=float(payload.get("learning_rate", 0.0)),
                trees=payload.get("trees", ()),
                calibration_intercept=float(payload.get("calibration_intercept", 0.0)),
                calibration_slope=float(payload.get("calibration_slope", 1.0)),
            )
            for row in values
        ],
        dtype=np.float64,
    )


def _decision_metrics(dataset: VisualDataset, indices: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    position = {int(index): offset for offset, index in enumerate(indices)}
    correct = total = 0
    trap_wins: dict[str, int] = {
        kind: 0 for kind in KINDS if kind not in {"exact", "invisible-notation"}
    }
    pairwise_correct = {kind: 0 for kind in trap_wins}
    pairwise_total = {kind: 0 for kind in trap_wins}
    for decision in dataset.decision_groups:
        local = [index for index in decision if index in position]
        if len(local) != len(decision):
            continue
        selected = max(local, key=lambda index: (probabilities[position[index]], -index))
        selected_kind = str(dataset.kinds[selected])
        correct += int(dataset.labels[selected] == 1)
        total += 1
        if selected_kind in trap_wins:
            trap_wins[selected_kind] += 1
        compatible = [
            probabilities[position[index]]
            for index in local
            if dataset.labels[index] == 1
        ]
        best_compatible = max(compatible, default=0.0)
        for index in local:
            kind = str(dataset.kinds[index])
            if kind not in pairwise_total:
                continue
            pairwise_total[kind] += 1
            pairwise_correct[kind] += int(best_compatible > probabilities[position[index]])
    return {
        "groups": total,
        "correct": correct,
        "compatible_top1": correct / max(total, 1),
        "trap_wins": trap_wins,
        "compatible_over_trap": {
            kind: pairwise_correct[kind] / max(pairwise_total[kind], 1)
            for kind in pairwise_total
        },
    }


def _sample_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    predictions = probabilities >= 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_accepts": int(np.sum(predictions & (labels == 0))),
        "false_rejects": int(np.sum((~predictions) & (labels == 1))),
    }


def _kind_metrics(kinds: np.ndarray, probabilities: np.ndarray) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for kind in KINDS:
        selected = probabilities[kinds == kind]
        if selected.size:
            result[kind] = {
                "samples": int(selected.size),
                "mean_probability": float(np.mean(selected)),
                "accepted_fraction": float(np.mean(selected >= 0.5)),
            }
    return result


def _evaluate(dataset: VisualDataset, indices: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    return {
        "sample": _sample_metrics(dataset.labels[indices], probabilities),
        "decision": _decision_metrics(dataset, indices, probabilities),
        "by_kind": _kind_metrics(dataset.kinds[indices], probabilities),
    }


def _model_size(payload: dict[str, object]) -> int:
    return len((json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))


def _select_config(results: list[dict[str, object]]) -> dict[str, object]:
    best_top1 = max(float(item["evaluation"]["decision"]["compatible_top1"]) for item in results)  # type: ignore[index]
    top1_candidates = [
        item
        for item in results
        if float(item["evaluation"]["decision"]["compatible_top1"]) >= best_top1 - 0.002  # type: ignore[index]
    ]
    best_loss = min(float(item["evaluation"]["sample"]["log_loss"]) for item in top1_candidates)  # type: ignore[index]
    eligible = [
        item
        for item in top1_candidates
        if float(item["evaluation"]["sample"]["log_loss"]) <= best_loss + 0.012  # type: ignore[index]
    ]
    return min(eligible, key=lambda item: (int(item["model_bytes"]), int(item["config"]["trees"])))  # type: ignore[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=ROOT.parent / "training" / "baselines" / "visual_measure_calibrator_v3.json",
    )
    args = parser.parse_args()

    dataset = build_dataset(args.seed, args.groups, max(1, args.workers))
    split = _split_groups(dataset.groups, args.seed)

    selection_results: list[dict[str, object]] = []
    trained: dict[tuple[str, int, int, int, str], tuple[object, LogisticRegression, dict[str, object]]] = {}
    for config in CONFIGS:
        key = (
            str(config["estimator"]),
            int(config["trees"]),
            int(config["max_depth"]),
            int(config["min_samples_leaf"]),
            str(config["max_features"]),
        )
        model = _fit_forest(dataset.features, dataset.labels, split["train"], config, args.seed)
        calibrator = _calibration(model, dataset.features, dataset.labels, split["calibration"], args.seed)
        payload = _payload(
            model,
            calibrator,
            feature_names=FEATURE_NAMES,
            config=config,
            groups=args.groups,
            seed=args.seed,
        )
        probabilities = _probabilities(payload, dataset.features[split["audit"]])
        item = {
            "config": dict(config),
            "model_bytes": _model_size(payload),
            "evaluation": _evaluate(dataset, split["audit"], probabilities),
        }
        selection_results.append(item)
        trained[key] = (model, calibrator, payload)

    selected = _select_config(selection_results)
    selected_config = selected["config"]
    selected_key = (
        str(selected_config["estimator"]),
        int(selected_config["trees"]),
        int(selected_config["max_depth"]),
        int(selected_config["min_samples_leaf"]),
        str(selected_config["max_features"]),
    )
    model, calibrator, payload = trained[selected_key]

    sklearn_raw = model.predict_proba(dataset.features[split["frozen"]])[:, 1].reshape(-1, 1)
    sklearn_probabilities = calibrator.predict_proba(sklearn_raw)[:, 1]
    deployed_probabilities = _probabilities(payload, dataset.features[split["frozen"]])
    deployment_delta = float(np.max(np.abs(sklearn_probabilities - deployed_probabilities)))
    if deployment_delta > 1e-10:
        raise RuntimeError(f"deployment inference mismatch: {deployment_delta}")

    event_grid_ablation_model = _fit_forest(
        dataset.v3_features,
        dataset.labels,
        split["train"],
        selected_config,
        args.seed,
    )
    event_grid_ablation_calibrator = _calibration(
        event_grid_ablation_model,
        dataset.v3_features,
        dataset.labels,
        split["calibration"],
        args.seed,
    )
    event_grid_ablation_payload = _payload(
        event_grid_ablation_model,
        event_grid_ablation_calibrator,
        feature_names=V3_FEATURE_NAMES,
        config=selected_config,
        groups=args.groups,
        seed=args.seed,
        model_version="scorescan-visual-measure-calibrator-4-event-grid-ablation",
    )
    event_grid_ablation_probabilities = _probabilities(
        event_grid_ablation_payload, dataset.v3_features[split["frozen"]]
    )

    legacy_ablation_model = _fit_forest(
        dataset.legacy_features,
        dataset.labels,
        split["train"],
        selected_config,
        args.seed,
    )
    legacy_ablation_calibrator = _calibration(
        legacy_ablation_model,
        dataset.legacy_features,
        dataset.labels,
        split["calibration"],
        args.seed,
    )
    legacy_ablation_payload = _payload(
        legacy_ablation_model,
        legacy_ablation_calibrator,
        feature_names=LEGACY_FEATURE_NAMES,
        config=selected_config,
        groups=args.groups,
        seed=args.seed,
        model_version="scorescan-visual-measure-calibrator-4-legacy-feature-ablation",
    )
    legacy_ablation_probabilities = _probabilities(
        legacy_ablation_payload, dataset.legacy_features[split["frozen"]]
    )

    baseline_payload = json.loads(args.baseline_model.read_text(encoding="utf-8"))
    baseline_probabilities = _legacy_probabilities(
        baseline_payload,
        dataset.v3_features[split["frozen"]],
    )

    report: dict[str, object] = {
        "model_version": MODEL_VERSION,
        "seed": args.seed,
        "measure_groups": args.groups,
        "samples": int(len(dataset.labels)),
        "split": {
            name: {
                "samples": int(len(indices)),
                "groups": int(len(set(int(value) for value in dataset.groups[indices].tolist()))),
            }
            for name, indices in split.items()
        },
        "positive_rate": float(np.mean(dataset.labels)),
        "feature_names": list(FEATURE_NAMES),
        "v3_feature_names": list(V3_FEATURE_NAMES),
        "legacy_feature_names": list(LEGACY_FEATURE_NAMES),
        "target_scope": (
            "local source-crop and immutable MusicXML candidate compatibility, including event order, "
            "symbol-to-event attachment, hollow noteheads, accidentals and compact marks; exact symbol "
            "recognition, candidate generation and end-to-end OMR remain outside scope"
        ),
        "model_selection": {
            "candidates": selection_results,
            "selected": selected,
            "rule": "maximise audit compatible Top-1; within 0.2 percentage points and 0.012 Log Loss choose smallest model",
        },
        "frozen_test": {
            "v4": _evaluate(dataset, split["frozen"], deployed_probabilities),
            "v3_same_test": _evaluate(dataset, split["frozen"], baseline_probabilities),
            "event_grid_ablation": _evaluate(dataset, split["frozen"], event_grid_ablation_probabilities),
            "legacy_feature_ablation": _evaluate(dataset, split["frozen"], legacy_ablation_probabilities),
        },
        "deployment_parity": {
            "max_absolute_probability_delta": deployment_delta,
            "samples": int(len(split["frozen"])),
        },
        "model_bytes": _model_size(payload),
        "baseline_model_version": str(baseline_payload.get("model_version", "unknown")),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
