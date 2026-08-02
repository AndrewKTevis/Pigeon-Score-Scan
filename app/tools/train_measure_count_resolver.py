from __future__ import annotations

"""Train the correlation-aware CPU measure-count resolver."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from measure_count_training_data import KINDS, MeasureCountDataset, build_dataset  # noqa: E402
from scorescan.measure_count_resolver import (  # noqa: E402
    FEATURE_NAMES,
    LEGACY_FEATURE_NAMES,
    measure_count_model_gate,
)
from scorescan.tree_model import stable_sigmoid  # noqa: E402
from tree_export import deployed_forest_probabilities, serialize_probability_forest  # noqa: E402


SEED = 20260721
MODEL_VERSION = "scorescan-measure-count-resolver-4"
CONFIGS = (
    {"trees": 48, "max_depth": 8, "min_samples_leaf": 5, "max_features": "sqrt"},
    {"trees": 96, "max_depth": 10, "min_samples_leaf": 3, "max_features": "sqrt"},
)


def _split_groups(groups: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * 0.70)
    calibration_end = int(total * 0.80)
    audit_end = int(total * 0.90)
    parts = {
        "train": shuffled[:train_end],
        "calibration": shuffled[train_end:calibration_end],
        "audit": shuffled[calibration_end:audit_end],
        "frozen": shuffled[audit_end:],
    }
    return {
        name: np.flatnonzero(np.isin(groups, values))
        for name, values in parts.items()
    }


def _fit_forest(
    values: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    config: dict[str, object],
    seed: int,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=int(config["trees"]),
        max_depth=int(config["max_depth"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        max_features=str(config["max_features"]),
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=1,
    )
    model.fit(values[indices], labels[indices])
    return model


def _fit_calibrator(
    model: RandomForestClassifier,
    values: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    seed: int,
) -> LogisticRegression:
    raw = model.predict_proba(values[indices])[:, 1].reshape(-1, 1)
    calibrator = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    calibrator.fit(raw, labels[indices])
    return calibrator


def _payload(
    model: RandomForestClassifier,
    calibrator: LogisticRegression,
    *,
    config: dict[str, object],
    groups: int,
    seed: int,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
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
            "select the true count only from layout/OMR-observed options while equalising correlated preprocessing-family support"
        ),
    }


def _model_size(payload: dict[str, object]) -> int:
    return len((json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))


def _logistic_baseline_probabilities(payload: dict[str, object], values: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(payload.get("coefficients", ()), dtype=np.float64)
    means = np.asarray(payload.get("means", ()), dtype=np.float64)
    scales = np.asarray(payload.get("scales", ()), dtype=np.float64)
    intercept = float(payload.get("intercept", 0.0))
    if not (
        coefficients.shape == (len(LEGACY_FEATURE_NAMES),)
        and means.shape == coefficients.shape
        and scales.shape == coefficients.shape
        and np.all(np.isfinite(coefficients))
        and np.all(np.isfinite(means))
        and np.all(np.isfinite(scales))
        and np.all(scales > 0.0)
    ):
        raise ValueError("invalid measure-count v1 baseline")
    scores = intercept + ((values - means) / scales) @ coefficients
    return np.asarray([stable_sigmoid(value) for value in scores], dtype=np.float64)


def _baseline_probabilities(
    payload: dict[str, object],
    features: np.ndarray,
    legacy_features: np.ndarray,
) -> np.ndarray:
    model_type = str(payload.get("model_type", ""))
    if model_type == "random_forest":
        feature_names = tuple(str(item) for item in payload.get("feature_names", ()))
        if not feature_names or len(feature_names) > features.shape[1]:
            raise ValueError("invalid random-forest measure-count baseline schema")
        return deployed_forest_probabilities(payload, features[:, : len(feature_names)])
    return _logistic_baseline_probabilities(payload, legacy_features)


def _sample_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    return {
        "samples": int(len(labels)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "false_accepts_at_0_5": int(np.sum((probabilities >= 0.5) & (labels == 0))),
        "false_rejects_at_0_5": int(np.sum((probabilities < 0.5) & (labels == 1))),
    }


def _decision_records(
    dataset: MeasureCountDataset,
    indices: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, object]]:
    position = {int(index): offset for offset, index in enumerate(indices)}
    records: list[dict[str, object]] = []
    for group, decision in enumerate(dataset.decision_groups):
        local = [index for index in decision if index in position]
        if len(local) != len(decision):
            continue
        ranked = sorted(
            local,
            key=lambda index: (
                probabilities[position[index]],
                -abs(int(dataset.option_counts[index]) - int(dataset.layout_counts[group])),
                -int(dataset.option_counts[index]),
            ),
            reverse=True,
        )
        best = ranked[0]
        second_probability = probabilities[position[ranked[1]]] if len(ranked) > 1 else 0.0
        records.append(
            {
                "group": group,
                "kind": str(dataset.kinds[best]),
                "truth": int(dataset.truths[best]),
                "selected": int(dataset.option_counts[best]),
                "probability": float(probabilities[position[best]]),
                "margin": float(max(0.0, probabilities[position[best]] - second_probability)),
                "deterministic": int(dataset.deterministic_counts[group]),
                "layout": int(dataset.layout_counts[group]),
                "layout_confidence": float(dataset.layout_confidences[group]),
                "family_support": int(dataset.family_supports[best]),
                "candidate_support": int(dataset.candidate_supports[best]),
            }
        )
    return records


def _decision_metrics(records: list[dict[str, object]]) -> dict[str, object]:
    correct = sum(int(item["selected"] == item["truth"]) for item in records)
    deterministic = sum(int(item["deterministic"] == item["truth"]) for item in records)
    by_kind: dict[str, dict[str, float | int]] = {}
    for kind in KINDS:
        selected = [item for item in records if item["kind"] == kind]
        if selected:
            by_kind[kind] = {
                "groups": len(selected),
                "top1": sum(int(item["selected"] == item["truth"]) for item in selected) / len(selected),
                "deterministic_top1": sum(int(item["deterministic"] == item["truth"]) for item in selected) / len(selected),
            }
    return {
        "groups": len(records),
        "top1": correct / max(len(records), 1),
        "deterministic_top1": deterministic / max(len(records), 1),
        "by_kind": by_kind,
    }


def _gated_metrics(
    records: list[dict[str, object]],
    probability_floor: float,
    margin_floor: float,
) -> dict[str, float | int]:
    correct = overrides = improvements = harms = 0
    for item in records:
        use_model = measure_count_model_gate(
            count=int(item["selected"]),
            probability=float(item["probability"]),
            margin=float(item["margin"]),
            family_support=int(item["family_support"]),
            candidate_support=int(item["candidate_support"]),
            deterministic_count=int(item["deterministic"]),
            layout_count=int(item["layout"]),
            layout_confidence=float(item["layout_confidence"]),
            probability_floor=probability_floor,
            margin_floor=margin_floor,
        )
        selected = item["selected"] if use_model else item["deterministic"]
        correct += int(selected == item["truth"])
        if use_model and item["selected"] != item["deterministic"]:
            overrides += 1
            improvements += int(item["selected"] == item["truth"] and item["deterministic"] != item["truth"])
            harms += int(item["selected"] != item["truth"] and item["deterministic"] == item["truth"])
    return {
        "accuracy": correct / max(len(records), 1),
        "overrides": overrides,
        "improvements": improvements,
        "harms": harms,
        "override_precision": improvements / max(overrides, 1),
    }


def _select_threshold(records: list[dict[str, object]]) -> dict[str, float | int]:
    candidates: list[dict[str, float | int]] = []
    for probability_floor in np.arange(0.80, 0.976, 0.025):
        for margin_floor in np.arange(0.20, 0.501, 0.02):
            metrics = _gated_metrics(records, float(probability_floor), float(margin_floor))
            candidates.append(
                {
                    "probability_floor": round(float(probability_floor), 3),
                    "margin_floor": round(float(margin_floor), 3),
                    **metrics,
                }
            )
    safe = [item for item in candidates if int(item["harms"]) == 0]
    pool = safe or candidates
    return max(
        pool,
        key=lambda item: (
            float(item["accuracy"]),
            int(item["improvements"]),
            float(item["override_precision"]),
            -int(item["overrides"]),
            float(item["probability_floor"]),
            float(item["margin_floor"]),
        ),
    )


def _select_config(results: list[dict[str, object]]) -> dict[str, object]:
    best_top1 = max(float(item["decision"]["top1"]) for item in results)  # type: ignore[index]
    top = [item for item in results if float(item["decision"]["top1"]) >= best_top1 - 0.005]  # type: ignore[index]
    best_loss = min(float(item["sample"]["log_loss"]) for item in top)  # type: ignore[index]
    eligible = [item for item in top if float(item["sample"]["log_loss"]) <= best_loss + 0.01]  # type: ignore[index]
    return min(eligible, key=lambda item: (int(item["model_bytes"]), int(item["config"]["trees"])))  # type: ignore[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=6000)
    parser.add_argument("--confirmation-groups", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=ROOT.parent / "training" / "baselines" / "measure_count_resolver_v2.json",
    )
    args = parser.parse_args()

    dataset = build_dataset(args.seed, args.groups, max(1, args.workers))
    split = _split_groups(dataset.groups, args.seed)

    selection_results: list[dict[str, object]] = []
    trained: dict[tuple[int, int, int, str], tuple[RandomForestClassifier, LogisticRegression, dict[str, object]]] = {}
    for config in CONFIGS:
        key = (
            int(config["trees"]),
            int(config["max_depth"]),
            int(config["min_samples_leaf"]),
            str(config["max_features"]),
        )
        model = _fit_forest(dataset.features, dataset.labels, split["train"], config, args.seed)
        calibrator = _fit_calibrator(model, dataset.features, dataset.labels, split["calibration"], args.seed)
        payload = _payload(model, calibrator, config=config, groups=args.groups, seed=args.seed)
        probabilities = deployed_forest_probabilities(payload, dataset.features[split["audit"]])
        records = _decision_records(dataset, split["audit"], probabilities)
        selection_results.append(
            {
                "config": dict(config),
                "model_bytes": _model_size(payload),
                "sample": _sample_metrics(dataset.labels[split["audit"]], probabilities),
                "decision": _decision_metrics(records),
            }
        )
        trained[key] = (model, calibrator, payload)

    selected_result = _select_config(selection_results)
    selected_config = selected_result["config"]
    key = (
        int(selected_config["trees"]),
        int(selected_config["max_depth"]),
        int(selected_config["min_samples_leaf"]),
        str(selected_config["max_features"]),
    )
    model, calibrator, payload = trained[key]

    sklearn_raw = model.predict_proba(dataset.features[split["frozen"]])[:, 1].reshape(-1, 1)
    sklearn_probabilities = calibrator.predict_proba(sklearn_raw)[:, 1]
    deployed_probabilities = deployed_forest_probabilities(payload, dataset.features[split["frozen"]])
    deployment_delta = float(np.max(np.abs(sklearn_probabilities - deployed_probabilities)))
    if deployment_delta > 1e-10:
        raise RuntimeError(f"deployment inference mismatch: {deployment_delta}")

    audit_probabilities = deployed_forest_probabilities(payload, dataset.features[split["audit"]])
    audit_records = _decision_records(dataset, split["audit"], audit_probabilities)
    threshold = _select_threshold(audit_records)
    frozen_records = _decision_records(dataset, split["frozen"], deployed_probabilities)

    ablation_model = _fit_forest(dataset.legacy_features, dataset.labels, split["train"], selected_config, args.seed)
    ablation_calibrator = _fit_calibrator(
        ablation_model,
        dataset.legacy_features,
        dataset.labels,
        split["calibration"],
        args.seed,
    )
    ablation_payload = _payload(
        ablation_model,
        ablation_calibrator,
        config=selected_config,
        groups=args.groups,
        seed=args.seed,
        feature_names=LEGACY_FEATURE_NAMES,
        model_version=f"{MODEL_VERSION}-legacy-feature-ablation",
    )
    ablation_probabilities = deployed_forest_probabilities(
        ablation_payload,
        dataset.legacy_features[split["frozen"]],
    )

    baseline_payload = json.loads(args.baseline_model.read_text(encoding="utf-8"))
    baseline_probabilities = _baseline_probabilities(
        baseline_payload,
        dataset.features[split["frozen"]],
        dataset.legacy_features[split["frozen"]],
    )

    confirmation = build_dataset(
        args.seed + 104_729,
        args.confirmation_groups,
        max(1, args.workers),
    )
    confirmation_probabilities = deployed_forest_probabilities(payload, confirmation.features)
    confirmation_indices = np.arange(len(confirmation.features), dtype=np.int64)
    confirmation_records = _decision_records(
        confirmation,
        confirmation_indices,
        confirmation_probabilities,
    )
    confirmation_gated = _gated_metrics(
        confirmation_records,
        float(threshold["probability_floor"]),
        float(threshold["margin_floor"]),
    )
    if int(confirmation_gated["harms"]) != 0:
        raise RuntimeError(
            "independent confirmation found harmful measure-count overrides: "
            f"{confirmation_gated['harms']}"
        )

    report: dict[str, object] = {
        "model_version": MODEL_VERSION,
        "training_seed": args.seed,
        "independent_groups": args.groups,
        "option_samples": int(len(dataset.features)),
        "feature_count": len(FEATURE_NAMES),
        "split_option_samples": {name: int(len(indices)) for name, indices in split.items()},
        "selected_configuration": dict(selected_config),
        "model_bytes": _model_size(payload),
        "model_selection": selection_results,
        "deployment_max_probability_delta": deployment_delta,
        "selected_policy_thresholds": threshold,
        "frozen": {
            "sample": _sample_metrics(dataset.labels[split["frozen"]], deployed_probabilities),
            "decision": _decision_metrics(frozen_records),
            "gated": _gated_metrics(
                frozen_records,
                float(threshold["probability_floor"]),
                float(threshold["margin_floor"]),
            ),
        },
        "baseline_v2_same_test": {
            "sample": _sample_metrics(dataset.labels[split["frozen"]], baseline_probabilities),
            "decision": _decision_metrics(_decision_records(dataset, split["frozen"], baseline_probabilities)),
        },
        "legacy_feature_forest_ablation": {
            "sample": _sample_metrics(dataset.labels[split["frozen"]], ablation_probabilities),
            "decision": _decision_metrics(_decision_records(dataset, split["frozen"], ablation_probabilities)),
        },
        "independent_confirmation": {
            "training_seed": args.seed + 104_729,
            "groups": args.confirmation_groups,
            "sample": _sample_metrics(confirmation.labels, confirmation_probabilities),
            "decision": _decision_metrics(confirmation_records),
            "gated": confirmation_gated,
        },
        "scope": "programmatic grouped layout/OMR measure-count evidence fusion; not end-to-end OMR accuracy",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
