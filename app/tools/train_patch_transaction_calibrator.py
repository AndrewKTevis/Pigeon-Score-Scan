from __future__ import annotations

"""Train the CPU veto for interacting local MusicXML repair transactions."""

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.patch_transaction import FEATURE_NAMES, PatchTransactionInput  # noqa: E402
from scorescan.policy import DEFAULT_POLICY  # noqa: E402
from scorescan.util import atomic_write_json, sha256_file  # noqa: E402
from tree_export import deployed_forest_probabilities, serialize_probability_forest  # noqa: E402

MODEL_VERSION = "scorescan-patch-transaction-forest-1"
TARGET_PRECISION = 0.9995
SAMPLES_PER_GROUP = 3
MODEL_CONFIGS = (
    {"n_estimators": 64, "max_depth": 8, "min_samples_leaf": 8},
    {"n_estimators": 80, "max_depth": 9, "min_samples_leaf": 7},
    {"n_estimators": 96, "max_depth": 10, "min_samples_leaf": 6},
)
SCENARIO_WEIGHTS = {
    "chord-pitch-compatible": 0.10,
    "tie-slur-compatible": 0.08,
    "attribute-rhythm-compatible": 0.07,
    "pitch-decoration-compatible": 0.08,
    "strong-mixed-compatible": 0.07,
    "boundary-decoration-compatible": 0.06,
    "chord-rhythm-conflict": 0.08,
    "pitch-rhythm-conflict": 0.08,
    "event-kind-pitch-conflict": 0.07,
    "event-kind-rhythm-conflict": 0.06,
    "attribute-rhythm-conflict": 0.06,
    "event-presence-bundle": 0.08,
    "grace-bundle": 0.05,
    "low-margin-pileup": 0.07,
    "common-mode-bundle": 0.05,
    "missing-family-bundle": 0.04,
}
POSITIVE_SCENARIOS = {
    "chord-pitch-compatible",
    "tie-slur-compatible",
    "attribute-rhythm-compatible",
    "pitch-decoration-compatible",
    "strong-mixed-compatible",
    "boundary-decoration-compatible",
}
SCENARIO_KINDS = {
    "chord-pitch-compatible": ("chord", "pitch"),
    "tie-slur-compatible": ("tie", "slur"),
    "attribute-rhythm-compatible": ("attribute", "rhythm"),
    "pitch-decoration-compatible": ("pitch", "articulation", "lyric", "direction"),
    "strong-mixed-compatible": ("pitch", "tie", "articulation", "direction"),
    "boundary-decoration-compatible": ("attribute", "barline", "direction", "lyric"),
    "chord-rhythm-conflict": ("chord", "rhythm"),
    "pitch-rhythm-conflict": ("pitch", "rhythm"),
    "event-kind-pitch-conflict": ("event_kind", "pitch"),
    "event-kind-rhythm-conflict": ("event_kind", "rhythm"),
    "attribute-rhythm-conflict": ("attribute", "rhythm"),
    "event-presence-bundle": ("event_presence", "rhythm", "pitch"),
    "grace-bundle": ("grace", "rhythm"),
    "low-margin-pileup": ("pitch", "tie", "articulation", "direction"),
    "common-mode-bundle": ("chord", "pitch", "tie"),
    "missing-family-bundle": ("pitch", "rhythm", "direction"),
}


@dataclass(frozen=True)
class Dataset:
    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    scenarios: tuple[str, ...]


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _choose_scenario(rng: random.Random) -> str:
    point = rng.random() * sum(SCENARIO_WEIGHTS.values())
    total = 0.0
    for name, weight in SCENARIO_WEIGHTS.items():
        total += weight
        if point <= total:
            return name
    return next(reversed(SCENARIO_WEIGHTS))


def _quality(rng: random.Random, centre: float, spread: float = 0.055) -> float:
    return _clip(rng.gauss(centre, spread), 0.01, 0.9999)


def _row(seed: int, group_id: int, sample_id: int) -> tuple[PatchTransactionInput, int, str]:
    group_seed = (seed << 19) ^ (group_id * 0x9E3779B1)
    scenario = _choose_scenario(random.Random(group_seed))
    rng = random.Random(group_seed ^ ((sample_id + 1) * 0x85EBCA77))
    positive = scenario in POSITIVE_SCENARIOS
    kinds = SCENARIO_KINDS[scenario]

    eligible_families = rng.choice((3, 3, 4, 4, 5))
    exact_family_support = rng.uniform(0.72, 1.0) if positive else rng.uniform(0.54, 1.0)
    semantic_family_support = rng.uniform(0.80, 1.0) if positive else rng.uniform(0.58, 1.0)
    missing_ratio = rng.uniform(0.0, 0.035) if positive else rng.uniform(0.0, 0.16)

    threshold = rng.uniform(0.91, 0.997)
    min_margin = rng.uniform(0.008, 0.055) if positive else rng.uniform(-0.035, 0.035)
    mean_margin = min_margin + rng.uniform(0.002, 0.045)
    min_probability = _clip(threshold + min_margin)
    mean_probability = _clip(threshold + mean_margin)

    quality_centre = 0.90 if positive else 0.82
    measure = _quality(rng, quality_centre)
    visual = _quality(rng, quality_centre - (0.01 if positive else 0.08), 0.075)
    event = _quality(rng, quality_centre + (0.015 if positive else -0.02), 0.065)
    context = _quality(rng, quality_centre - (0.005 if positive else 0.07), 0.075)
    ensemble = _quality(rng, quality_centre + (0.02 if positive else -0.01), 0.055)
    semantic_confidence = _quality(rng, 0.91 if positive else 0.70, 0.075)
    mean_distance = _clip(rng.uniform(0.0, 0.018) if positive else rng.uniform(0.004, 0.052), 0.0, 0.08)
    template_distance = _clip(rng.uniform(0.004, 0.040) if positive else rng.uniform(0.002, 0.075), 0.0, 0.10)
    changed_events = rng.randint(1, 5) + max(0, len(kinds) - 2)
    changed_surfaces = rng.randint(len(kinds), len(kinds) * 3)

    if scenario == "chord-pitch-compatible":
        # Real consensus can be strongly event-corroborated while page/context
        # calibrators remain deliberately conservative.  This compatible pair must
        # not be vetoed merely because generic contextual evidence is modest.
        exact_family_support = rng.uniform(0.72, 1.0)
        semantic_family_support = rng.uniform(0.72, 1.0)
        measure = _quality(rng, 0.60, 0.11)
        visual = _quality(rng, 0.52, 0.08)
        event = _quality(rng, 0.95, 0.035)
        context = _quality(rng, 0.38, 0.13)
        ensemble = _quality(rng, 0.77, 0.08)
        semantic_confidence = _quality(rng, 0.79, 0.08)
        template_distance = rng.uniform(0.045, 0.10)
        min_margin = rng.uniform(0.018, 0.055)
    elif scenario == "tie-slur-compatible":
        min_margin += 0.01
        visual = _quality(rng, 0.82, 0.08)
        event = _quality(rng, 0.94, 0.04)
    elif scenario == "attribute-rhythm-compatible":
        context = _quality(rng, 0.94, 0.04)
        event = _quality(rng, 0.92, 0.04)
        template_distance = rng.uniform(0.018, 0.055)
    elif scenario == "boundary-decoration-compatible":
        measure = _quality(rng, 0.86, 0.06)
        context = _quality(rng, 0.94, 0.04)
        changed_events = rng.randint(0, 2)
    elif scenario == "chord-rhythm-conflict":
        # Strong-looking but incompatible onset/duration interaction.  Continuous
        # evidence intentionally overlaps compatible bundles so the interaction
        # feature must contribute independent value.
        event = _quality(rng, 0.86, 0.055)
        context = _quality(rng, 0.85, 0.060)
        visual = _quality(rng, 0.86, 0.060)
        changed_events += rng.randint(2, 5)
        mean_distance = rng.uniform(0.006, 0.030)
        min_margin = rng.uniform(0.004, 0.032)
    elif scenario == "pitch-rhythm-conflict":
        visual = _quality(rng, 0.86, 0.060)
        event = _quality(rng, 0.86, 0.060)
        context = _quality(rng, 0.85, 0.065)
        min_margin = rng.uniform(0.003, 0.030)
        mean_distance = rng.uniform(0.004, 0.028)
    elif scenario == "event-kind-pitch-conflict":
        visual = _quality(rng, 0.84, 0.065)
        event = _quality(rng, 0.85, 0.060)
        context = _quality(rng, 0.84, 0.065)
        template_distance = rng.uniform(0.010, 0.042)
        min_margin = rng.uniform(0.002, 0.028)
    elif scenario == "event-kind-rhythm-conflict":
        event = _quality(rng, 0.84, 0.065)
        context = _quality(rng, 0.83, 0.070)
        visual = _quality(rng, 0.84, 0.070)
        changed_events += rng.randint(2, 6)
        min_margin = rng.uniform(0.002, 0.026)
    elif scenario == "attribute-rhythm-conflict":
        context = _quality(rng, 0.38, 0.11)
        measure = _quality(rng, 0.62, 0.09)
        mean_distance = rng.uniform(0.020, 0.058)
    elif scenario == "event-presence-bundle":
        event = _quality(rng, 0.42, 0.12)
        context = _quality(rng, 0.47, 0.12)
        changed_events += rng.randint(3, 7)
        missing_ratio = rng.uniform(0.03, 0.18)
    elif scenario == "grace-bundle":
        event = _quality(rng, 0.56, 0.10)
        context = _quality(rng, 0.48, 0.12)
        min_margin = rng.uniform(-0.025, 0.015)
    elif scenario == "low-margin-pileup":
        quality_centre = 0.80
        measure = _quality(rng, quality_centre, 0.07)
        visual = _quality(rng, quality_centre, 0.08)
        event = _quality(rng, quality_centre, 0.07)
        context = _quality(rng, quality_centre, 0.08)
        ensemble = _quality(rng, quality_centre, 0.06)
        min_margin = rng.uniform(-0.035, 0.004)
        mean_margin = rng.uniform(-0.005, 0.015)
    elif scenario == "common-mode-bundle":
        # Deliberately strong-looking evidence: the interaction flags and subtle
        # cross-layer disagreement must carry the veto, not one low score.
        exact_family_support = rng.uniform(0.82, 1.0)
        semantic_family_support = rng.uniform(0.86, 1.0)
        missing_ratio = rng.uniform(0.0, 0.04)
        measure = _quality(rng, 0.88, 0.04)
        visual = _quality(rng, 0.61, 0.08)
        event = _quality(rng, 0.73, 0.07)
        context = _quality(rng, 0.67, 0.08)
        ensemble = _quality(rng, 0.81, 0.05)
        semantic_confidence = _quality(rng, 0.88, 0.04)
        min_margin = rng.uniform(0.002, 0.030)
        mean_margin = min_margin + rng.uniform(0.006, 0.030)
    elif scenario == "missing-family-bundle":
        missing_ratio = rng.uniform(0.10, 0.28)
        exact_family_support = rng.uniform(0.50, 0.76)
        semantic_family_support = rng.uniform(0.55, 0.80)
        ensemble = _quality(rng, 0.64, 0.09)
        semantic_confidence = _quality(rng, 0.60, 0.10)

    # Recompute probabilities after scenario-specific margin changes.
    min_probability = _clip(threshold + min_margin)
    mean_probability = _clip(threshold + mean_margin)

    # Controlled overlap prevents trivial separation and exercises conservative
    # threshold selection.  Labels remain scenario-defined, not feature-defined.
    if rng.random() < 0.20:
        visual = _clip(visual + rng.gauss(0.0, 0.09))
        event = _clip(event + rng.gauss(0.0, 0.08))
        context = _clip(context + rng.gauss(0.0, 0.09))
        ensemble = _clip(ensemble + rng.gauss(0.0, 0.06))
        semantic_confidence = _clip(semantic_confidence + rng.gauss(0.0, 0.07))
        min_margin = _clip(min_margin + rng.gauss(0.0, 0.012), -0.08, 0.12)
        mean_margin = _clip(mean_margin + rng.gauss(0.0, 0.014), -0.08, 0.16)
        min_probability = _clip(threshold + min_margin)
        mean_probability = _clip(threshold + mean_margin)

    item = PatchTransactionInput(
        patch_kinds=kinds,
        changed_event_count=changed_events,
        changed_surface_count=changed_surfaces,
        minimum_patch_probability=min_probability,
        mean_patch_probability=mean_probability,
        minimum_patch_margin=min_margin,
        mean_patch_margin=mean_margin,
        maximum_patch_threshold=threshold,
        eligible_family_count=eligible_families,
        exact_family_support_ratio=exact_family_support,
        semantic_family_support_ratio=semantic_family_support,
        missing_ratio=missing_ratio,
        selected_measure_probability=measure,
        selected_visual_probability=visual,
        selected_event_probability=event,
        selected_context_probability=context,
        selected_ensemble_probability=ensemble,
        semantic_confidence=semantic_confidence,
        mean_cluster_distance=mean_distance,
        template_distance=template_distance,
    )
    if not item.requires_model():
        raise RuntimeError(f"training scenario is not model-applicable: {scenario}")
    return item, int(positive), scenario


def build_dataset(seed: int, groups: int) -> Dataset:
    rows: list[list[float]] = []
    labels: list[int] = []
    group_ids: list[int] = []
    scenarios: list[str] = []
    for group_id in range(groups):
        for sample_id in range(SAMPLES_PER_GROUP):
            item, label, scenario = _row(seed, group_id, sample_id)
            rows.append(item.feature_vector())
            labels.append(label)
            group_ids.append(group_id)
            scenarios.append(scenario)
    return Dataset(
        np.asarray(rows, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(group_ids, dtype=np.int64),
        tuple(scenarios),
    )


def _dataset_fingerprint(dataset: Dataset) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(dataset.features, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(dataset.labels, dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(dataset.groups, dtype="<i8").tobytes(order="C"))
    for scenario in dataset.scenarios:
        encoded = scenario.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _split(groups: np.ndarray, seed: int) -> tuple[np.ndarray, ...]:
    unique = sorted(set(int(value) for value in groups.tolist()))
    random.Random(seed).shuffle(unique)
    count = len(unique)
    cuts = (int(count * 0.65), int(count * 0.75), int(count * 0.85), int(count * 0.90))
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


def _fit(
    dataset: Dataset,
    train: np.ndarray,
    calibration: np.ndarray,
    seed: int,
    config: dict[str, int],
    feature_indices: np.ndarray | None = None,
):
    indices = feature_indices if feature_indices is not None else np.arange(dataset.features.shape[1])
    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=1,
        random_state=seed,
    )
    model.fit(dataset.features[train][:, indices], dataset.labels[train])
    raw = model.predict_proba(dataset.features[calibration][:, indices])[:, 1]
    calibrator = LogisticRegression(C=1000.0, max_iter=3000, random_state=seed)
    calibrator.fit(raw.reshape(-1, 1), dataset.labels[calibration])
    return model, calibrator, indices


def _payload(model, calibrator, config: dict[str, int]) -> dict[str, object]:
    return {
        "model_version": MODEL_VERSION,
        "model_type": "random_forest",
        "feature_names": list(FEATURE_NAMES),
        "trees": serialize_probability_forest(model),
        "calibration_intercept": float(calibrator.intercept_[0]),
        "calibration_slope": float(calibrator.coef_[0, 0]),
        "model_config": dict(config),
        "target": "composed local repair bundle is safe to commit after deterministic audit",
        "scope": "veto-only interaction calibration for model-applicable multi-patch transactions",
    }


def _probabilities(model, calibrator, values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(values[:, indices])[:, 1]
    return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    accepted = probabilities >= threshold
    true_accepts = int(np.sum(accepted & (labels == 1)))
    false_accepts = int(np.sum(accepted & (labels == 0)))
    accepts = true_accepts + false_accepts
    return {
        "threshold": float(threshold),
        "accepted": accepts,
        "true_accepts": true_accepts,
        "false_accepts": false_accepts,
        "precision": true_accepts / max(accepts, 1),
        "coverage": accepts / max(len(labels), 1),
        "positive_recall": true_accepts / max(int(np.sum(labels == 1)), 1),
    }


def _sample_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities >= 0.5
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }


def _select_threshold(labels: np.ndarray, probabilities: np.ndarray):
    candidates = sorted(set(float(value) for value in probabilities))
    candidates.extend([0.975, 0.985, 0.99, 0.995, 0.9975, 0.999, 0.9995])
    valid = []
    for threshold in sorted(set(candidates)):
        metrics = _metrics(labels, probabilities, threshold)
        if metrics["accepted"] and float(metrics["precision"]) >= TARGET_PRECISION:
            valid.append((threshold, metrics))
    if not valid:
        return 1.0, _metrics(labels, probabilities, 1.0)
    return max(
        valid,
        key=lambda item: (
            float(item[1]["coverage"]),
            float(item[1]["positive_recall"]),
            item[0],
        ),
    )


def _scenario_metrics(dataset: Dataset, indices: np.ndarray, probabilities: np.ndarray, threshold: float):
    result = {}
    for scenario in sorted(set(dataset.scenarios[index] for index in indices)):
        local = np.asarray(
            [offset for offset, index in enumerate(indices) if dataset.scenarios[index] == scenario]
        )
        result[scenario] = _metrics(
            dataset.labels[indices][local], probabilities[local], threshold
        )
    return result


def _group_leakage_audit(dataset: Dataset, partitions: tuple[np.ndarray, ...]) -> dict[str, object]:
    names = ("train", "calibration", "model_selection_audit", "threshold_selection", "frozen_test")
    sets = {
        name: set(int(value) for value in dataset.groups[index].tolist())
        for name, index in zip(names, partitions, strict=True)
    }
    overlaps: dict[str, int] = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlaps[f"{left}__{right}"] = len(sets[left] & sets[right])
    return {
        "groups_per_partition": {name: len(sets[name]) for name in names},
        "pairwise_group_overlap": overlaps,
        "leakage_detected": any(overlaps.values()),
    }


def _margin_rule(dataset: Dataset, indices: np.ndarray) -> np.ndarray:
    feature = {name: offset for offset, name in enumerate(FEATURE_NAMES)}
    values = dataset.features[indices]
    accepted = (
        (values[:, feature["minimum_patch_margin"]] >= 0.0)
        & (values[:, feature["selected_ensemble_probability"]] >= 0.78)
        & (values[:, feature["missing_ratio"]] <= 0.08)
        & (values[:, feature["semantic_confidence"]] >= 0.74)
    )
    return accepted.astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "src/scorescan/resources/patch_transaction_calibrator.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT.parent / "training/patch_transaction_calibrator_report_v1.json",
    )
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--groups", type=int, default=9000)
    parser.add_argument("--confirmation-groups", type=int, default=3000)
    args = parser.parse_args()

    dataset = build_dataset(args.seed, args.groups)
    partitions = _split(dataset.groups, args.seed)
    train, calibration, audit, threshold_indices, test = partitions
    leakage_audit = _group_leakage_audit(dataset, partitions)
    if leakage_audit["leakage_detected"]:
        raise RuntimeError("group leakage detected between training partitions")

    trained = []
    audit_rows = []
    for config in MODEL_CONFIGS:
        model, calibrator, indices = _fit(dataset, train, calibration, args.seed, config)
        probabilities = _probabilities(model, calibrator, dataset.features[audit], indices)
        audit_rows.append(
            {"config": dict(config), "sample": _sample_metrics(dataset.labels[audit], probabilities)}
        )
        trained.append((model, calibrator, indices))
    best_index = min(
        range(len(audit_rows)),
        key=lambda index: (
            -float(audit_rows[index]["sample"]["roc_auc"]),
            float(audit_rows[index]["sample"]["log_loss"]),
            int(audit_rows[index]["config"]["n_estimators"]),
        ),
    )
    model, calibrator, indices = trained[best_index]
    selected_config = dict(audit_rows[best_index]["config"])
    payload = _payload(model, calibrator, selected_config)

    threshold_probabilities = _probabilities(
        model, calibrator, dataset.features[threshold_indices], indices
    )
    raw_threshold, raw_threshold_metrics = _select_threshold(
        dataset.labels[threshold_indices], threshold_probabilities
    )
    threshold = max(float(DEFAULT_POLICY.patch_transaction_probability_floor), raw_threshold)
    threshold_metrics = _metrics(
        dataset.labels[threshold_indices], threshold_probabilities, threshold
    )
    payload.update(
        {
            "training_seed": args.seed,
            "training_groups": args.groups,
            "selected_on": "independent grouped model-selection audit",
            "auto_commit_threshold": threshold,
            "target_precision": TARGET_PRECISION,
        }
    )

    test_probabilities = _probabilities(model, calibrator, dataset.features[test], indices)
    deployed = deployed_forest_probabilities(payload, dataset.features[test])
    deployment_delta = float(np.max(np.abs(test_probabilities - deployed), initial=0.0))
    if deployment_delta > 1e-10:
        raise RuntimeError(f"deployment prediction mismatch: {deployment_delta}")

    confirmation = build_dataset(args.seed + 17, args.confirmation_groups)
    confirmation_probabilities = _probabilities(
        model, calibrator, confirmation.features, indices
    )

    # Interaction-feature ablation is trained independently and evaluated with its own
    # threshold-selection partition.  It is report-only and is never deployed.
    interaction_start = FEATURE_NAMES.index("chord_pitch_interaction")
    ablation_indices = np.arange(interaction_start)
    ablation_model, ablation_calibrator, ablation_indices = _fit(
        dataset,
        train,
        calibration,
        args.seed,
        selected_config,
        ablation_indices,
    )
    ablation_threshold_probabilities = _probabilities(
        ablation_model,
        ablation_calibrator,
        dataset.features[threshold_indices],
        ablation_indices,
    )
    ablation_raw_threshold, _ = _select_threshold(
        dataset.labels[threshold_indices], ablation_threshold_probabilities
    )
    ablation_threshold = max(
        float(DEFAULT_POLICY.patch_transaction_probability_floor), ablation_raw_threshold
    )
    ablation_test_probabilities = _probabilities(
        ablation_model, ablation_calibrator, dataset.features[test], ablation_indices
    )

    accept_all = np.ones(len(test), dtype=np.float64)
    margin_rule = _margin_rule(dataset, test)
    atomic_write_json(args.output, payload)
    report = {
        "model_version": MODEL_VERSION,
        "seed": args.seed,
        "groups": args.groups,
        "samples_per_group": SAMPLES_PER_GROUP,
        "samples": len(dataset.labels),
        "dataset_kind": "synthetic interaction scenarios derived from ScoreScan patch contracts",
        "dataset_disclosure": (
            "This training corpus is synthetic and does not establish real-scan end-to-end accuracy. "
            "The model is a veto-only safety layer and must be validated again on frozen reviewed scans."
        ),
        "dataset_fingerprint": _dataset_fingerprint(dataset),
        "group_leakage_audit": leakage_audit,
        "partitions": {
            "train": len(train),
            "calibration": len(calibration),
            "model_selection_audit": len(audit),
            "threshold_selection": len(threshold_indices),
            "frozen_test": len(test),
        },
        "model_selection": audit_rows,
        "selected_config": selected_config,
        "raw_precision_threshold": raw_threshold_metrics,
        "policy_probability_floor": DEFAULT_POLICY.patch_transaction_probability_floor,
        "selected_threshold": threshold_metrics,
        "frozen_test": {
            "sample": _sample_metrics(dataset.labels[test], test_probabilities),
            "policy": _metrics(dataset.labels[test], test_probabilities, threshold),
            "accept_all_model_applicable_transactions": _metrics(
                dataset.labels[test], accept_all, 0.5
            ),
            "deterministic_margin_rule": _metrics(
                dataset.labels[test], margin_rule, 0.5
            ),
            "scenarios": _scenario_metrics(dataset, test, test_probabilities, threshold),
        },
        "independent_confirmation": {
            "seed": args.seed + 17,
            "groups": args.confirmation_groups,
            "samples": len(confirmation.labels),
            "dataset_fingerprint": _dataset_fingerprint(confirmation),
            "policy": _metrics(
                confirmation.labels, confirmation_probabilities, threshold
            ),
        },
        "interaction_feature_ablation": {
            "removed_features": list(FEATURE_NAMES[interaction_start:]),
            "threshold": ablation_threshold,
            "frozen_test": {
                "sample": _sample_metrics(
                    dataset.labels[test], ablation_test_probabilities
                ),
                "policy": _metrics(
                    dataset.labels[test],
                    ablation_test_probabilities,
                    ablation_threshold,
                ),
            },
        },
        "deployment_parity": {"max_absolute_probability_delta": deployment_delta},
        "feature_names": list(FEATURE_NAMES),
        "model_bytes": args.output.stat().st_size,
        "model_sha256": sha256_file(args.output),
    }
    atomic_write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
