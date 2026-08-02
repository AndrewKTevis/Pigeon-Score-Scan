from __future__ import annotations

"""Train the verified CPU visual veto for one already-proposed event insertion."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from event_presence_visual_training_data import EventPresenceVisualDataset  # noqa: E402
from scorescan.event_presence_visual_guard import (  # noqa: E402
    EVENT_PRESENCE_VISUAL_CONTEXT_FEATURE_COUNT,
    EVENT_PRESENCE_VISUAL_FEATURE_NAMES,
)
from scorescan.util import atomic_write_json  # noqa: E402
from tree_export import deployed_forest_probabilities, serialize_probability_forest  # noqa: E402

MODEL_VERSION = "scorescan-event-presence-visual-forest-1"
MODEL_CONFIGS = (
    {"n_estimators": 96, "max_depth": 9, "min_samples_leaf": 3},
    {"n_estimators": 128, "max_depth": 10, "min_samples_leaf": 3},
    {"n_estimators": 160, "max_depth": 11, "min_samples_leaf": 2},
)
MINIMUM_THRESHOLD = 0.82
MINIMUM_INSERT_RECALL = 0.38
MINIMUM_KIND_RECALL = 0.35
THRESHOLD_MARGIN = 0.001
CONTEXT_FEATURE_COUNT = EVENT_PRESENCE_VISUAL_CONTEXT_FEATURE_COUNT


def _load(path: Path) -> EventPresenceVisualDataset:
    with np.load(path, allow_pickle=False) as payload:
        return EventPresenceVisualDataset(
            features=np.asarray(payload["features"], dtype=np.float64),
            labels=np.asarray(payload["labels"], dtype=np.int64),
            groups=np.asarray(payload["groups"], dtype=np.int64),
            scenarios=tuple(str(value) for value in payload["scenarios"].tolist()),
            operations=tuple(str(value) for value in payload["operations"].tolist()),
            event_kinds=tuple(str(value) for value in payload["event_kinds"].tolist()),
        )


def _split(groups: np.ndarray, seed: int) -> tuple[np.ndarray, ...]:
    unique = sorted(set(int(value) for value in groups.tolist()))
    random.Random(seed).shuffle(unique)
    count = len(unique)
    cuts = (int(count * 0.60), int(count * 0.72), int(count * 0.82), int(count * 0.90))
    partitions = (
        set(unique[: cuts[0]]),
        set(unique[cuts[0] : cuts[1]]),
        set(unique[cuts[1] : cuts[2]]),
        set(unique[cuts[2] : cuts[3]]),
        set(unique[cuts[3] :]),
    )
    return tuple(
        np.flatnonzero(np.isin(groups, np.asarray(sorted(partition), dtype=groups.dtype)))
        for partition in partitions
    )


def _fit(features, labels, train, calibration, seed, config):
    model = RandomForestClassifier(
        random_state=seed,
        n_jobs=1,
        class_weight="balanced_subsample",
        max_features="sqrt",
        **config,
    )
    model.fit(features[train], labels[train])
    raw = model.predict_proba(features[calibration])[:, 1]
    calibrator = LogisticRegression(C=1.0, random_state=seed, solver="lbfgs")
    calibrator.fit(raw.reshape(-1, 1), labels[calibration])
    return model, calibrator


def _probabilities(model, calibrator, features):
    raw = model.predict_proba(features)[:, 1]
    return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


def _sample(labels, probabilities):
    return {
        "accuracy": float(accuracy_score(labels, probabilities >= 0.5)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }


def _threshold(dataset, probabilities, indices, floor, event_kind):
    local_labels = dataset.labels[indices]
    local_probabilities = probabilities[indices]
    local_kinds = np.asarray(dataset.event_kinds)[indices]
    negatives = local_probabilities[(local_labels == 0) & (local_kinds == event_kind)]
    observed = float(np.nextafter(np.max(negatives, initial=0.0), 1.0))
    return min(1.0, max(float(floor), observed + THRESHOLD_MARGIN))


def _policy(dataset, probabilities, indices, thresholds):
    labels = dataset.labels[indices]
    local_probabilities = probabilities[indices]
    operations = np.asarray(dataset.operations)[indices]
    event_kinds = np.asarray(dataset.event_kinds)[indices]
    # Deployment is deliberately insertion-only.  Deletions with source evidence are
    # review-only because partially visible source events produced unsafe absence tails.
    applicable = operations == "insert"
    local_thresholds = np.asarray(
        [float(thresholds[str(kind)]) for kind in event_kinds], dtype=np.float64
    )
    accepted = applicable & (local_probabilities >= local_thresholds)
    result = {
        "auto_patch_thresholds": {
            kind: float(thresholds[kind]) for kind in ("note", "rest")
        },
        "correct_accepts": int(np.sum((labels == 1) & accepted)),
        "false_accepts": int(np.sum((labels == 0) & accepted)),
        "selective_precision": float(np.sum((labels == 1) & accepted))
        / max(int(np.sum(accepted)), 1),
        "coverage": float(np.mean(accepted)),
        "delete_transactions_review_only": int(np.sum(operations == "delete")),
    }
    insert_positive = (labels == 1) & applicable
    result["insert_positive_total"] = int(np.sum(insert_positive))
    result["insert_correct_accepts"] = int(np.sum(insert_positive & accepted))
    result["insert_recall"] = int(np.sum(insert_positive & accepted)) / max(
        int(np.sum(insert_positive)), 1
    )
    for kind in ("note", "rest"):
        mask = insert_positive & (event_kinds == kind)
        result[f"{kind}_insert_positive_total"] = int(np.sum(mask))
        result[f"{kind}_insert_correct_accepts"] = int(np.sum(mask & accepted))
        result[f"{kind}_insert_recall"] = int(np.sum(mask & accepted)) / max(
            int(np.sum(mask)), 1
        )
    return result


def _scenario_false_accepts(dataset, probabilities, thresholds):
    labels = dataset.labels
    scenarios = np.asarray(dataset.scenarios)
    operations = np.asarray(dataset.operations)
    kinds = np.asarray(dataset.event_kinds)
    local_thresholds = np.asarray(
        [float(thresholds[str(kind)]) for kind in kinds], dtype=np.float64
    )
    return {
        scenario: {
            "negative_samples": int(
                np.sum((labels == 0) & (operations == "insert") & (scenarios == scenario))
            ),
            "false_accepts": int(
                np.sum(
                    (labels == 0)
                    & (operations == "insert")
                    & (scenarios == scenario)
                    & (probabilities >= local_thresholds)
                )
            ),
            "maximum_negative_probability": float(
                np.max(
                    probabilities[
                        (labels == 0)
                        & (operations == "insert")
                        & (scenarios == scenario)
                    ],
                    initial=0.0,
                )
            ),
        }
        for scenario in sorted(set(dataset.scenarios))
    }


def _canonical(value):
    if isinstance(value, float):
        return round(value, 12) if np.isfinite(value) else value
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "src" / "scorescan" / "resources" / "event_presence_visual_guard.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT.parent / "training" / "event_presence_visual_guard_report_v1.json",
    )
    parser.add_argument(
        "--external-report",
        type=Path,
        default=ROOT.parent / "training" / "event_presence_visual_guard_external_tail_v1.json",
    )
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--safety-data", type=Path, required=True)
    parser.add_argument("--independent-data", type=Path, required=True)
    parser.add_argument("--external-data", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    dataset = _load(args.training_data)
    train, calibration, selection_indices, threshold_indices, test = _split(
        dataset.groups, args.seed
    )
    trained = []
    selection = []
    for config in MODEL_CONFIGS:
        model, calibrator = _fit(
            dataset.features, dataset.labels, train, calibration, args.seed, config
        )
        values = _probabilities(model, calibrator, dataset.features[selection_indices])
        selection.append(
            {"config": dict(config), "sample": _sample(dataset.labels[selection_indices], values)}
        )
        trained.append((model, calibrator))
    selected_index = min(
        range(len(selection)),
        key=lambda index: (
            -selection[index]["sample"]["roc_auc"],
            selection[index]["sample"]["log_loss"],
            selection[index]["config"]["n_estimators"],
        ),
    )
    model, calibrator = trained[selected_index]
    config = dict(selection[selected_index]["config"])
    safety = _load(args.safety_data)
    independent = _load(args.independent_data)
    external = _load(args.external_data)

    raw_probabilities = _probabilities(model, calibrator, dataset.features)
    raw_safety_probabilities = _probabilities(model, calibrator, safety.features)
    raw_independent_probabilities = _probabilities(model, calibrator, independent.features)
    raw_external_probabilities = _probabilities(model, calibrator, external.features)

    payload = {
        "model_version": MODEL_VERSION,
        "model_type": "random_forest",
        "feature_names": list(EVENT_PRESENCE_VISUAL_FEATURE_NAMES),
        "trees": serialize_probability_forest(model),
        "calibration_intercept": float(calibrator.intercept_[0]),
        "calibration_slope": float(calibrator.coef_[0, 0]),
        "auto_patch_threshold": 1.0,
        "auto_patch_thresholds": {"note": 1.0, "rest": 1.0},
        "target_precision": 1.0,
        "training_seed": int(args.seed),
        "training_groups": len(set(int(value) for value in dataset.groups.tolist())),
        "model_config": config,
        "target": "the preserved local source image is more compatible with the complete after-insertion event sequence than the coherent before-insertion sequence",
        "scope": "veto-only confirmation for one fixed-content insertion using proposed-versus-displaced event templates and complete before/after sequence evidence; deletions with source evidence, pitch, duration, event kind, multiple events, chords and grace notes are review-only",
    }
    probabilities = deployed_forest_probabilities(payload, dataset.features)
    safety_probabilities = deployed_forest_probabilities(payload, safety.features)
    independent_probabilities = deployed_forest_probabilities(payload, independent.features)
    external_probabilities = deployed_forest_probabilities(payload, external.features)
    safety_indices = np.arange(len(safety.labels), dtype=np.int64)
    independent_indices = np.arange(len(independent.labels), dtype=np.int64)
    external_indices = np.arange(len(external.labels), dtype=np.int64)
    thresholds = {
        kind: max(
            _threshold(dataset, probabilities, threshold_indices, MINIMUM_THRESHOLD, kind),
            _threshold(safety, safety_probabilities, safety_indices, MINIMUM_THRESHOLD, kind),
        )
        for kind in ("note", "rest")
    }
    payload["auto_patch_threshold"] = float(max(thresholds.values()))
    payload["auto_patch_thresholds"] = {
        kind: float(thresholds[kind]) for kind in ("note", "rest")
    }

    policies = {
        "frozen_test": _policy(dataset, probabilities, test, thresholds),
        "safety_calibration": _policy(safety, safety_probabilities, safety_indices, thresholds),
        "independent_test": _policy(
            independent, independent_probabilities, independent_indices, thresholds
        ),
        "external_tail": _policy(external, external_probabilities, external_indices, thresholds),
    }
    if any(int(value["false_accepts"]) for value in policies.values()):
        raise RuntimeError(f"event-presence visual guard false accepts: {policies}")
    for name, value in policies.items():
        if float(value["insert_recall"]) < MINIMUM_INSERT_RECALL:
            raise RuntimeError(
                f"event-presence visual guard insertion recall too low in {name}: {value}"
            )
        if (
            float(value["note_insert_recall"]) < MINIMUM_KIND_RECALL
            or float(value["rest_insert_recall"]) < MINIMUM_KIND_RECALL
        ):
            raise RuntimeError(
                f"event-presence visual guard kind recall too low in {name}: {value}"
            )

    deployment_deltas = {
        "training_all": float(np.max(np.abs(probabilities - raw_probabilities), initial=0.0)),
        "safety_calibration": float(
            np.max(np.abs(safety_probabilities - raw_safety_probabilities), initial=0.0)
        ),
        "independent_test": float(
            np.max(np.abs(independent_probabilities - raw_independent_probabilities), initial=0.0)
        ),
        "external_tail": float(
            np.max(np.abs(external_probabilities - raw_external_probabilities), initial=0.0)
        ),
    }
    deployment_delta = max(deployment_deltas.values(), default=0.0)
    if deployment_delta > 1e-9:
        raise RuntimeError(f"deployment prediction mismatch: {deployment_deltas}")

    ablation = RandomForestClassifier(
        random_state=args.seed,
        n_jobs=1,
        class_weight="balanced_subsample",
        max_features="sqrt",
        **config,
    )
    ablation.fit(dataset.features[train, :CONTEXT_FEATURE_COUNT], dataset.labels[train])
    ablation_probabilities = ablation.predict_proba(
        independent.features[:, :CONTEXT_FEATURE_COUNT]
    )[:, 1]
    ablation_metrics = _sample(independent.labels, ablation_probabilities)

    rejected_thresholds = {
        f"{candidate:.2f}": _policy(
            external,
            external_probabilities,
            external_indices,
            {"note": candidate, "rest": candidate},
        )
        for candidate in (0.90, 0.88, 0.85)
        if candidate < max(thresholds.values())
    }

    report = {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "training_seed": int(args.seed),
        "feature_count": len(EVENT_PRESENCE_VISUAL_FEATURE_NAMES),
        "dataset": {
            "training_groups": len(set(int(value) for value in dataset.groups.tolist())),
            "training_samples": int(len(dataset.labels)),
            "safety_groups": len(set(int(value) for value in safety.groups.tolist())),
            "safety_samples": int(len(safety.labels)),
            "independent_groups": len(set(int(value) for value in independent.groups.tolist())),
            "independent_samples": int(len(independent.labels)),
            "external_groups": len(set(int(value) for value in external.groups.tolist())),
            "external_samples": int(len(external.labels)),
        },
        "selection": selection,
        "selected_config": config,
        "threshold_margin": THRESHOLD_MARGIN,
        "transaction_contract": {
            "operation": "single_event_insertion_only",
            "accepted_suffix_forms": ["unchanged_explicit_gap", "coherent_shift_by_inserted_duration"],
            "target_event_kind_sampling": "balanced_note_rest",
            "surrounding_score_sampling": "natural_4_to_1_note_rest",
            "source_comparison": "proposed_event_vs_displaced_event_plus_complete_before_after_sequence",
        },
        "auto_patch_thresholds": {
            kind: float(thresholds[kind]) for kind in ("note", "rest")
        },
        "sample_metrics": {
            "frozen_test": _sample(dataset.labels[test], probabilities[test]),
            "safety_calibration": _sample(safety.labels, safety_probabilities),
            "independent_test": _sample(independent.labels, independent_probabilities),
            "external_tail": _sample(external.labels, external_probabilities),
        },
        "policies": policies,
        "scenario_false_accepts": {
            "safety_calibration": _scenario_false_accepts(
                safety, safety_probabilities, thresholds
            ),
            "independent_test": _scenario_false_accepts(
                independent, independent_probabilities, thresholds
            ),
            "external_tail": _scenario_false_accepts(external, external_probabilities, thresholds),
        },
        "context_only_ablation": ablation_metrics,
        "deployment_max_probability_delta": deployment_delta,
        "deployment_probability_deltas": deployment_deltas,
        "rejected_lower_thresholds": rejected_thresholds,
        "limitations": [
            "Programmatic rendered event-presence transactions are not a substitute for an independently reviewed real-scan benchmark.",
            "The runtime gate confirms only one already-proposed event insertion and cannot choose pitch, duration or event kind.",
            "The source-image comparison models both a coherent suffix shift by the inserted duration and an existing explicit gap; partially shifted suffixes fail closed.",
            "Event deletion was rejected for automatic deployment because partially visible source events produced unsafe absence tails.",
            "Multiple edits, chords, grace notes, unpitched events and unsupported note types remain review-only when source evidence is present.",
        ],
    }
    external_report = {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "groups": len(set(int(value) for value in external.groups.tolist())),
        "samples": int(len(external.labels)),
        "policy": policies["external_tail"],
        "sample_metrics": report["sample_metrics"]["external_tail"],
        "scenario_false_accepts": report["scenario_false_accepts"]["external_tail"],
        "rejected_lower_thresholds": rejected_thresholds,
        "used_for_training_or_threshold_selection": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.external_report.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, payload)
    atomic_write_json(args.report, _canonical(report))
    atomic_write_json(args.external_report, _canonical(external_report))
    print(json.dumps(_canonical({"model": payload, "report": report}), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
