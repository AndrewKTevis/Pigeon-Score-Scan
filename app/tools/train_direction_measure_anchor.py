from __future__ import annotations

"""Train the conservative CPU measure-index refiner for OCR music directions."""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from direction_measure_anchor_training_data import (  # noqa: E402
    DirectionMeasureAnchorDataset,
    build_dataset,
)
from scorescan.direction_anchor import ANCHOR_FEATURE_NAMES  # noqa: E402
from scorescan.model_registry import build_manifest  # noqa: E402
from scorescan.tree_model import VerifiedRandomForestModel  # noqa: E402
from scorescan.util import atomic_write_json  # noqa: E402
from tree_export import deployed_forest_probabilities, serialize_probability_forest  # noqa: E402

MODEL_VERSION = "scorescan-direction-anchor-hybrid-2"
ANCHOR_MODEL_VERSION = "scorescan-direction-measure-anchor-forest-1"
MODEL_CONFIGS = (
    {"n_estimators": 24, "max_depth": 7, "min_samples_leaf": 6},
    {"n_estimators": 32, "max_depth": 8, "min_samples_leaf": 6},
    {"n_estimators": 48, "max_depth": 9, "min_samples_leaf": 5},
)


def _split_groups(group_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique = sorted(set(int(value) for value in group_ids.tolist()))
    random.Random(seed).shuffle(unique)
    count = len(unique)
    cuts = (int(count * 0.70), int(count * 0.80), int(count * 0.90))
    partitions = (
        set(unique[: cuts[0]]),
        set(unique[cuts[0] : cuts[1]]),
        set(unique[cuts[1] : cuts[2]]),
        set(unique[cuts[2] :]),
    )
    return tuple(
        np.flatnonzero(np.isin(group_ids, np.asarray(sorted(partition), dtype=group_ids.dtype)))
        for partition in partitions
    )  # type: ignore[return-value]


def _fit(
    dataset: DirectionMeasureAnchorDataset,
    train: np.ndarray,
    calibration: np.ndarray,
    seed: int,
    config: dict[str, int],
) -> tuple[RandomForestClassifier, LogisticRegression, dict[str, object]]:
    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=1,
        random_state=seed,
    )
    model.fit(dataset.features[train], dataset.labels[train])
    calibration_raw = model.predict_proba(dataset.features[calibration])[:, 1]
    calibrator = LogisticRegression(C=1000.0, max_iter=2000, random_state=seed)
    calibrator.fit(calibration_raw.reshape(-1, 1), dataset.labels[calibration])
    payload: dict[str, object] = {
        "model_type": "random_forest",
        "model_version": ANCHOR_MODEL_VERSION,
        "feature_names": list(ANCHOR_FEATURE_NAMES),
        "trees": serialize_probability_forest(model),
        "calibration_intercept": float(calibrator.intercept_[0]),
        "calibration_slope": float(calibrator.coef_[0, 0]),
        "training_seed": seed,
        "model_config": dict(config),
    }
    return model, calibrator, payload


def _sklearn_probabilities(
    model: RandomForestClassifier,
    calibrator: LogisticRegression,
    values: np.ndarray,
) -> np.ndarray:
    raw = model.predict_proba(values)[:, 1]
    return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


def _decision_subset(dataset: DirectionMeasureAnchorDataset, row_indices: np.ndarray) -> list[int]:
    available = set(int(value) for value in row_indices.tolist())
    return [
        index
        for index, decision in enumerate(dataset.decisions)
        if decision.candidate_rows and all(row in available for row in decision.candidate_rows)
    ]


def _rank_top1_accuracy(
    dataset: DirectionMeasureAnchorDataset,
    probabilities: np.ndarray,
    decision_indices: list[int],
) -> float:
    correct = 0
    for decision_index in decision_indices:
        decision = dataset.decisions[decision_index]
        best_index = max(
            zip(decision.candidate_rows, decision.candidate_indices, strict=True),
            key=lambda item: (
                float(probabilities[item[0]]),
                -abs(item[1] - decision.baseline_index),
                -item[1],
            ),
        )[1]
        correct += int(best_index == decision.true_index)
    return correct / max(len(decision_indices), 1)


def _decision_arrays(
    dataset: DirectionMeasureAnchorDataset,
    probabilities: np.ndarray,
    decision_indices: list[int],
) -> dict[str, object]:
    baseline: list[int] = []
    truth: list[int] = []
    best: list[int] = []
    best_probability: list[float] = []
    margin: list[float] = []
    scenarios: list[str] = []
    kinds: list[str] = []
    for decision_index in decision_indices:
        decision = dataset.decisions[decision_index]
        ranked = sorted(
            (
                (float(probabilities[row]), candidate_index)
                for row, candidate_index in zip(
                    decision.candidate_rows, decision.candidate_indices, strict=True
                )
            ),
            key=lambda item: (
                item[0],
                -abs(item[1] - decision.baseline_index),
                -item[1],
            ),
            reverse=True,
        )
        baseline.append(decision.baseline_index)
        truth.append(decision.true_index)
        best.append(ranked[0][1])
        best_probability.append(ranked[0][0])
        margin.append(ranked[0][0] - (ranked[1][0] if len(ranked) > 1 else 0.0))
        scenarios.append(decision.scenario)
        kinds.append(decision.kind)
    return {
        "baseline": np.asarray(baseline, dtype=np.int64),
        "truth": np.asarray(truth, dtype=np.int64),
        "best": np.asarray(best, dtype=np.int64),
        "probability": np.asarray(best_probability, dtype=np.float64),
        "margin": np.asarray(margin, dtype=np.float64),
        "scenarios": np.asarray(scenarios, dtype=object),
        "kinds": np.asarray(kinds, dtype=object),
    }


def _array_metrics(
    arrays: dict[str, object],
    probability_threshold: float,
    margin_threshold: float,
) -> dict[str, object]:
    baseline = arrays["baseline"]
    truth = arrays["truth"]
    best = arrays["best"]
    probability = arrays["probability"]
    margins = arrays["margin"]
    scenarios = arrays["scenarios"]
    kinds = arrays["kinds"]
    assert isinstance(baseline, np.ndarray) and isinstance(truth, np.ndarray)
    assert isinstance(best, np.ndarray) and isinstance(probability, np.ndarray)
    assert isinstance(margins, np.ndarray) and isinstance(scenarios, np.ndarray)
    assert isinstance(kinds, np.ndarray)
    changed_mask = (best != baseline) & (probability >= probability_threshold) & (margins >= margin_threshold)
    selected = np.where(changed_mask, best, baseline)
    baseline_hits = baseline == truth
    refined_hits = selected == truth
    changed = int(np.sum(changed_mask))
    changed_correct = int(np.sum(changed_mask & refined_hits))
    total = max(len(baseline), 1)

    def grouped(values: np.ndarray) -> dict[str, dict[str, float | int]]:
        result: dict[str, dict[str, float | int]] = {}
        for raw_name in sorted(set(str(item) for item in values.tolist())):
            mask = values == raw_name
            count = int(np.sum(mask))
            result[raw_name] = {
                "baseline_accuracy": float(np.mean(baseline_hits[mask])) if count else 0.0,
                "refined_accuracy": float(np.mean(refined_hits[mask])) if count else 0.0,
                "decisions": count,
            }
        return result

    return {
        "decisions": len(baseline),
        "baseline_accuracy": float(np.mean(baseline_hits)) if len(baseline) else 0.0,
        "refined_accuracy": float(np.mean(refined_hits)) if len(baseline) else 0.0,
        "absolute_accuracy_gain": (
            float(np.mean(refined_hits) - np.mean(baseline_hits))
            if len(baseline)
            else 0.0
        ),
        "changed_decisions": changed,
        "changed_coverage": changed / total,
        "changed_precision": changed_correct / max(changed, 1),
        "changed_errors": changed - changed_correct,
        "by_scenario": grouped(scenarios),
        "by_kind": grouped(kinds),
    }


def _decision_metrics(
    dataset: DirectionMeasureAnchorDataset,
    probabilities: np.ndarray,
    decision_indices: list[int],
    probability_threshold: float,
    margin_threshold: float,
) -> dict[str, object]:
    return _array_metrics(
        _decision_arrays(dataset, probabilities, decision_indices),
        probability_threshold,
        margin_threshold,
    )


def _choose_gate(
    dataset: DirectionMeasureAnchorDataset,
    probabilities: np.ndarray,
    decisions: list[int],
) -> tuple[float, float, dict[str, object]]:
    arrays = _decision_arrays(dataset, probabilities, decisions)
    best: tuple[tuple[float, float, float, float], float, float, dict[str, object]] | None = None
    for probability_threshold in np.arange(0.65, 0.901, 0.005):
        for margin_threshold in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
            metrics = _array_metrics(
                arrays, float(probability_threshold), float(margin_threshold)
            )
            if metrics["changed_decisions"] < 6:
                continue
            if metrics["changed_precision"] < 0.985:
                continue
            if metrics["refined_accuracy"] < metrics["baseline_accuracy"]:
                continue
            scenario_safe = all(
                item["refined_accuracy"] + 0.002 >= item["baseline_accuracy"]
                for item in metrics["by_scenario"].values()
            )
            if not scenario_safe:
                continue
            rank = (
                float(metrics["refined_accuracy"]),
                float(metrics["changed_precision"]),
                -float(metrics["changed_errors"]),
                -probability_threshold - margin_threshold * 0.1,
            )
            if best is None or rank > best[0]:
                best = (rank, float(probability_threshold), float(margin_threshold), metrics)
    if best is None:
        fallback = _array_metrics(arrays, 1.0, 1.0)
        return 1.0, 1.0, fallback
    return best[1], best[2], best[3]


def _candidate_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "positive_samples": int(np.sum(labels == 1)),
        "negative_samples": int(np.sum(labels == 0)),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20270112)
    parser.add_argument("--groups", type=int, default=3200)
    parser.add_argument(
        "--role-baseline",
        type=Path,
        default=REPOSITORY / "training" / "baselines" / "direction_anchor_classifier_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "src" / "scorescan" / "resources" / "direction_anchor_classifier.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPOSITORY / "training" / "direction_measure_anchor_report_v1.json",
    )
    args = parser.parse_args()

    dataset = build_dataset(args.seed, args.groups)
    train, calibration, audit, test = _split_groups(dataset.group_ids, args.seed)
    audit_decisions = _decision_subset(dataset, audit)
    test_decisions = _decision_subset(dataset, test)

    candidates: list[dict[str, object]] = []
    fitted: list[tuple[RandomForestClassifier, LogisticRegression, dict[str, object]]] = []
    for config in MODEL_CONFIGS:
        model, calibrator, payload = _fit(dataset, train, calibration, args.seed, config)
        audit_probabilities = _sklearn_probabilities(model, calibrator, dataset.features)
        probability_threshold, margin_threshold, gate_metrics = _choose_gate(
            dataset, audit_probabilities, audit_decisions
        )
        payload["selection_probability_threshold"] = probability_threshold
        payload["selection_margin_threshold"] = margin_threshold
        candidates.append(
            {
                "config": dict(config),
                "audit_top1": _rank_top1_accuracy(dataset, audit_probabilities, audit_decisions),
                "audit_candidates": _candidate_metrics(
                    dataset.labels[audit], audit_probabilities[audit]
                ),
                "probability_threshold": probability_threshold,
                "margin_threshold": margin_threshold,
                "gate": gate_metrics,
                "serialized_bytes": len(json.dumps(payload, sort_keys=True, separators=(",", ":"))),
            }
        )
        fitted.append((model, calibrator, payload))

    viable = [
        (index, item)
        for index, item in enumerate(candidates)
        if float(item["probability_threshold"]) < 1.0
    ]
    if not viable:
        raise RuntimeError("no direction measure anchor gate met the conservative audit criteria")
    best_accuracy = max(float(item["gate"]["refined_accuracy"]) for _index, item in viable)
    accuracy_tolerance = 0.0030
    competitive = [
        (index, item)
        for index, item in viable
        if float(item["gate"]["refined_accuracy"]) + accuracy_tolerance >= best_accuracy
    ]
    selected_index, selected_audit = min(
        competitive, key=lambda item: int(item[1]["serialized_bytes"])
    )
    model, calibrator, anchor_payload = fitted[selected_index]
    test_probabilities = _sklearn_probabilities(model, calibrator, dataset.features)
    parity_rows = test[: min(1024, len(test))]
    deployed = deployed_forest_probabilities(anchor_payload, dataset.features[parity_rows])
    deployment_difference = float(
        np.max(np.abs(test_probabilities[parity_rows] - deployed))
    )
    test_metrics = _decision_metrics(
        dataset,
        test_probabilities,
        test_decisions,
        float(anchor_payload["selection_probability_threshold"]),
        float(anchor_payload["selection_margin_threshold"]),
    )
    if test_metrics["refined_accuracy"] <= test_metrics["baseline_accuracy"]:
        raise RuntimeError("frozen direction anchor accuracy did not improve")
    if test_metrics["changed_precision"] < 0.98 or test_metrics["changed_errors"] > 2:
        raise RuntimeError("frozen direction anchor gate is not conservative enough")
    if deployment_difference > 1e-10:
        raise RuntimeError("serialized direction anchor inference does not match training runtime")

    role_payload = json.loads(args.role_baseline.read_text(encoding="utf-8"))
    resource = {
        "format": 2,
        "model_version": MODEL_VERSION,
        "role_model": role_payload,
        "measure_anchor_model": anchor_payload,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, resource)
    canonical_resources = ROOT / "src" / "scorescan" / "resources"
    if args.output.parent.resolve() == canonical_resources.resolve():
        atomic_write_json(args.output.parent / "model_manifest.json", build_manifest(args.output.parent))
    runtime = VerifiedRandomForestModel.from_payload(
        anchor_payload,
        ANCHOR_FEATURE_NAMES,
        verified=True,
        status="trained",
    )
    if not runtime.enabled:
        raise RuntimeError("serialized direction measure anchor model failed runtime validation")

    report = {
        "model_version": MODEL_VERSION,
        "anchor_model_version": ANCHOR_MODEL_VERSION,
        "seed": args.seed,
        "system_groups": args.groups,
        "candidate_samples": int(len(dataset.labels)),
        "decision_samples": int(len(dataset.decisions)),
        "features": list(ANCHOR_FEATURE_NAMES),
        "splits": {
            "train_candidates": int(len(train)),
            "calibration_candidates": int(len(calibration)),
            "audit_candidates": int(len(audit)),
            "test_candidates": int(len(test)),
            "audit_decisions": len(audit_decisions),
            "test_decisions": len(test_decisions),
        },
        "model_candidates": candidates,
        "selected_config": selected_audit["config"],
        "selected_probability_threshold": anchor_payload["selection_probability_threshold"],
        "selected_margin_threshold": anchor_payload["selection_margin_threshold"],
        "audit": selected_audit["gate"],
        "test_candidates": _candidate_metrics(dataset.labels[test], test_probabilities[test]),
        "test_decisions": test_metrics,
        "deployment_max_abs_probability_difference": deployment_difference,
        "scope": (
            "CPU candidate ranking for OCR direction ownership when visual and MusicXML "
            "measure counts differ; not OCR text accuracy or end-to-end OMR accuracy"
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"resource_sha256={_sha256(args.output)}")
    print(f"report_sha256={_sha256(args.report)}")


if __name__ == "__main__":
    main()
