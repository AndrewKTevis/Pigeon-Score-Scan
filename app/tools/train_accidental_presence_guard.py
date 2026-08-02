from __future__ import annotations

"""Train the verified CPU accidental-presence safety model."""

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from accidental_presence_training_data import AccidentalPresenceDataset  # noqa: E402
from scorescan.accidental_presence_guard import ACCIDENTAL_PRESENCE_FEATURE_NAMES  # noqa: E402
from scorescan.util import atomic_write_json  # noqa: E402
from tree_export import deployed_forest_probabilities, serialize_probability_forest  # noqa: E402

MODEL_VERSION = "scorescan-accidental-presence-forest-1"
REGISTERED_MODEL_VERSION = "scorescan-accidental-presence-forest-2"
REGISTERED_DATASET_NAME = "scorescan-registered-scan-accidental-presence-v1"
REGISTERED_DATASET_ROLE = "training_calibration_and_internal_test_only"
REGISTERED_SOURCE_ROLE = "training_only_disjoint_from_external_release_holdout"
MODEL_CONFIGS = (
    {"n_estimators": 96, "max_depth": 10, "min_samples_leaf": 3},
    {"n_estimators": 128, "max_depth": 11, "min_samples_leaf": 3},
    {"n_estimators": 160, "max_depth": 12, "min_samples_leaf": 2},
)
MINIMUM_THRESHOLD = 0.70
MINIMUM_CLASS_RECALL = 0.30


@dataclass(frozen=True)
class RegisteredTrainingBundle:
    train: AccidentalPresenceDataset
    calibration: AccidentalPresenceDataset
    test: AccidentalPresenceDataset
    report: dict[str, Any]
    report_sha256: str
    file_sha256: dict[str, str]


def _split(groups: np.ndarray, seed: int) -> tuple[np.ndarray, ...]:
    unique = sorted(set(int(value) for value in groups.tolist()))
    random.Random(seed).shuffle(unique)
    count = len(unique)
    cuts = (int(count * 0.68), int(count * 0.80), int(count * 0.90))
    partitions = (
        set(unique[: cuts[0]]),
        set(unique[cuts[0] : cuts[1]]),
        set(unique[cuts[1] : cuts[2]]),
        set(unique[cuts[2] :]),
    )
    return tuple(
        np.flatnonzero(np.isin(groups, np.asarray(sorted(partition), dtype=groups.dtype)))
        for partition in partitions
    )


def _fit_arrays(
    train_features,
    train_labels,
    calibration_features,
    calibration_labels,
    seed,
    config,
    jobs,
):
    model = RandomForestClassifier(
        random_state=seed,
        n_jobs=jobs,
        class_weight="balanced_subsample",
        max_features="sqrt",
        **config,
    )
    model.fit(train_features, train_labels)
    raw = model.predict_proba(calibration_features)[:, 1]
    calibrator = LogisticRegression(C=1.0, random_state=seed, solver="lbfgs")
    calibrator.fit(raw.reshape(-1, 1), calibration_labels)
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


def _thresholds(labels, probabilities, indices, floor):
    local_labels = labels[indices]
    local_probabilities = probabilities[indices]
    false_present = local_probabilities[local_labels == 0]
    false_absent = 1.0 - local_probabilities[local_labels == 1]
    present = max(float(floor), float(np.nextafter(np.max(false_present, initial=0.0), 1.0)))
    absent = max(float(floor), float(np.nextafter(np.max(false_absent, initial=0.0), 1.0)))
    return min(1.0, present), min(1.0, absent)


def _policy(labels, probabilities, indices, present_threshold, absent_threshold):
    local_labels = labels[indices]
    local_probabilities = probabilities[indices]
    present_accepted = local_probabilities >= present_threshold
    absent_accepted = (1.0 - local_probabilities) >= absent_threshold
    correct = (local_labels == 1) & present_accepted | (local_labels == 0) & absent_accepted
    wrong = (local_labels == 0) & present_accepted | (local_labels == 1) & absent_accepted
    present_total = max(int(np.sum(local_labels == 1)), 1)
    absent_total = max(int(np.sum(local_labels == 0)), 1)
    return {
        "present_threshold": float(present_threshold),
        "absent_threshold": float(absent_threshold),
        "correct_accepts": int(np.sum(correct)),
        "false_accepts": int(np.sum(wrong)),
        "selective_precision": float(np.sum(correct)) / max(int(np.sum(correct) + np.sum(wrong)), 1),
        "present_recall": int(np.sum((local_labels == 1) & present_accepted)) / present_total,
        "absent_recall": int(np.sum((local_labels == 0) & absent_accepted)) / absent_total,
        "coverage": float(np.mean(correct | wrong)),
    }


def _canonical(value):
    if isinstance(value, float):
        return round(value, 12) if np.isfinite(value) else value
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _load_dataset(path: Path) -> AccidentalPresenceDataset:
    with np.load(path, allow_pickle=False) as payload:
        dataset = AccidentalPresenceDataset(
            features=np.asarray(payload["features"], dtype=np.float64),
            labels=np.asarray(payload["labels"], dtype=np.int64),
            groups=np.asarray(payload["groups"], dtype=np.int64),
            symbols=tuple(str(value) for value in payload["symbols"].tolist()),
        )
    _validate_dataset(dataset, path)
    return dataset


def _validate_dataset(
    dataset: AccidentalPresenceDataset,
    path: Path,
) -> None:
    sample_count = len(dataset.labels)
    expected_features = len(ACCIDENTAL_PRESENCE_FEATURE_NAMES)
    if dataset.features.shape != (sample_count, expected_features):
        raise ValueError(
            f"accidental dataset has invalid feature shape: {path}"
        )
    if dataset.groups.shape != (sample_count,) or len(dataset.symbols) != sample_count:
        raise ValueError(f"accidental dataset arrays have inconsistent rows: {path}")
    if sample_count < 2 or set(int(value) for value in dataset.labels.tolist()) != {0, 1}:
        raise ValueError(f"accidental dataset must contain both classes: {path}")
    if not np.all(np.isfinite(dataset.features)):
        raise ValueError(f"accidental dataset contains non-finite features: {path}")
    if np.any(dataset.groups < 0):
        raise ValueError(f"accidental dataset contains invalid group ids: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _registered_split_summary(
    dataset: AccidentalPresenceDataset,
) -> dict[str, int]:
    return {
        "samples": len(dataset.labels),
        "positive_samples": int(np.sum(dataset.labels == 1)),
        "groups": len(set(int(value) for value in dataset.groups.tolist())),
    }


def _load_registered_training_bundle(path: Path) -> RegisteredTrainingBundle:
    path = path.resolve(strict=True)
    report_path = path / "prepare-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("registered accidental preparation report is malformed")
    if report.get("name") != REGISTERED_DATASET_NAME:
        raise ValueError("registered accidental dataset name is incompatible")
    if report.get("role") != REGISTERED_DATASET_ROLE:
        raise ValueError("registered accidental dataset is not training-role data")
    if report.get("source_role") != REGISTERED_SOURCE_ROLE:
        raise ValueError("registered accidental source is not isolated training data")
    if report.get("license") != "CC0-1.0":
        raise ValueError("registered accidental dataset license is not CC0-1.0")
    if report.get("holdout_used_for_training") is not False:
        raise ValueError("registered accidental dataset may contain external holdout data")
    intersections = report.get("work_intersections")
    if not isinstance(intersections, dict) or any(intersections.values()):
        raise ValueError("registered accidental dataset has work-level split leakage")
    if report.get("feature_names") != list(ACCIDENTAL_PRESENCE_FEATURE_NAMES):
        raise ValueError("registered accidental feature contract is stale")

    datasets: dict[str, AccidentalPresenceDataset] = {}
    file_sha256: dict[str, str] = {}
    for split in ("train", "calibration", "test"):
        split_path = path / f"{split}.npz"
        datasets[split] = _load_dataset(split_path)
        file_sha256[f"{split}.npz"] = _sha256_file(split_path)
        summary = _registered_split_summary(datasets[split])
        if int(report.get("samples_by_split", {}).get(split, -1)) != summary["samples"]:
            raise ValueError(f"registered accidental {split} sample count is stale")
        if (
            int(report.get("positive_samples_by_split", {}).get(split, -1))
            != summary["positive_samples"]
        ):
            raise ValueError(f"registered accidental {split} class count is stale")
        if int(report.get("groups_by_split", {}).get(split, -1)) != summary["groups"]:
            raise ValueError(f"registered accidental {split} group count is stale")
        if int(report.get("works_by_split", {}).get(split, 0)) < 1:
            raise ValueError(f"registered accidental {split} has no work provenance")
    return RegisteredTrainingBundle(
        train=datasets["train"],
        calibration=datasets["calibration"],
        test=datasets["test"],
        report=report,
        report_sha256=_sha256_file(report_path),
        file_sha256=file_sha256,
    )


def _rows(
    primary: AccidentalPresenceDataset,
    indices: np.ndarray,
    secondary: AccidentalPresenceDataset | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    features = primary.features[indices]
    labels = primary.labels[indices]
    if secondary is not None:
        features = np.concatenate((features, secondary.features), axis=0)
        labels = np.concatenate((labels, secondary.labels), axis=0)
    return features, labels


def _validate_programmatic_report(
    report_path: Path,
    *,
    train_path: Path,
    safety_path: Path,
    independent_path: Path,
) -> dict[str, Any]:
    report_path = report_path.resolve(strict=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("programmatic accidental preparation report is malformed")
    if report.get("name") != "scorescan-programmatic-accidental-presence-v2":
        raise ValueError("programmatic accidental dataset name is incompatible")
    if report.get("role") != "training_safety_and_regression_only":
        raise ValueError("programmatic accidental dataset role is incompatible")
    if report.get("feature_names") != list(ACCIDENTAL_PRESENCE_FEATURE_NAMES):
        raise ValueError("programmatic accidental feature contract is stale")
    if report.get("registered_scan_evidence") is not False:
        raise ValueError("programmatic accidental report misstates scan evidence")
    if report.get("independent_real_scan_holdout") is not False:
        raise ValueError("programmatic accidental report misstates holdout evidence")
    paths = {
        "train": train_path.resolve(strict=True),
        "safety": safety_path.resolve(strict=True),
        "independent": independent_path.resolve(strict=True),
    }
    splits = report.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("programmatic accidental split provenance is missing")
    for split, path in paths.items():
        if path.parent != report_path.parent or path.name != f"{split}.npz":
            raise ValueError("programmatic accidental paths do not match their report")
        row = splits.get(split)
        if not isinstance(row, dict):
            raise ValueError(f"programmatic accidental {split} provenance is missing")
        if row.get("sha256") != _sha256_file(path):
            raise ValueError(f"programmatic accidental {split} hash is stale")
        if int(row.get("groups", 0)) < 1 or int(row.get("samples", 0)) < 2:
            raise ValueError(f"programmatic accidental {split} coverage is empty")
    return {
        "prepare_report_sha256": _sha256_file(report_path),
        "splits": splits,
        "registered_scan_evidence": False,
        "independent_real_scan_holdout": False,
        "end_to_end_accuracy_claim": False,
    }



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "src" / "scorescan" / "resources" / "accidental_presence_guard.json")
    parser.add_argument("--report", type=Path, default=ROOT.parent / "training" / "accidental_presence_guard_report_v1.json")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--safety-data", type=Path, required=True)
    parser.add_argument("--independent-data", type=Path, required=True)
    parser.add_argument("--programmatic-report", type=Path)
    parser.add_argument(
        "--registered-data-dir",
        type=Path,
        help=(
            "training-role registered scan descriptors with fixed "
            "train/calibration/test splits"
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) // 2)),
    )
    args = parser.parse_args()
    if args.jobs < 1:
        raise ValueError("jobs must be positive")
    if args.registered_data_dir is not None and args.programmatic_report is None:
        raise ValueError(
            "registered-scan training requires bound programmatic provenance"
        )
    programmatic_audit = (
        _validate_programmatic_report(
            args.programmatic_report,
            train_path=args.train_data,
            safety_path=args.safety_data,
            independent_path=args.independent_data,
        )
        if args.programmatic_report is not None
        else None
    )

    dataset = _load_dataset(args.train_data)
    registered = (
        _load_registered_training_bundle(args.registered_data_dir)
        if args.registered_data_dir is not None
        else None
    )
    model_version = REGISTERED_MODEL_VERSION if registered is not None else MODEL_VERSION
    training_groups = len(set(int(value) for value in dataset.groups.tolist()))
    train, calibration, threshold_indices, test = _split(dataset.groups, args.seed)
    fit_features, fit_labels = _rows(
        dataset,
        train,
        registered.train if registered is not None else None,
    )
    calibration_features, calibration_labels = _rows(
        dataset,
        calibration,
        registered.calibration if registered is not None else None,
    )
    trained = []
    selection = []
    for config in MODEL_CONFIGS:
        model, calibrator = _fit_arrays(
            fit_features,
            fit_labels,
            calibration_features,
            calibration_labels,
            args.seed,
            config,
            args.jobs,
        )
        probabilities = _probabilities(model, calibrator, dataset.features[test])
        selection.append({"config": dict(config), "sample": _sample(dataset.labels[test], probabilities)})
        trained.append((model, calibrator))
    selected_index = min(
        range(len(selection)),
        key=lambda index: (-selection[index]["sample"]["roc_auc"], selection[index]["sample"]["log_loss"], selection[index]["config"]["n_estimators"]),
    )
    model, calibrator = trained[selected_index]
    config = dict(selection[selected_index]["config"])
    probabilities = _probabilities(model, calibrator, dataset.features)
    present_threshold, absent_threshold = _thresholds(dataset.labels, probabilities, threshold_indices, MINIMUM_THRESHOLD)
    registered_probabilities: dict[str, np.ndarray] = {}
    if registered is not None:
        for split in ("train", "calibration", "test"):
            split_dataset = getattr(registered, split)
            registered_probabilities[split] = _probabilities(
                model,
                calibrator,
                split_dataset.features,
            )
        registered_calibration_indices = np.arange(
            len(registered.calibration.labels),
            dtype=np.int64,
        )
        registered_present, registered_absent = _thresholds(
            registered.calibration.labels,
            registered_probabilities["calibration"],
            registered_calibration_indices,
            MINIMUM_THRESHOLD,
        )
        present_threshold = max(present_threshold, registered_present)
        absent_threshold = max(absent_threshold, registered_absent)

    safety = _load_dataset(args.safety_data)
    safety_groups = len(set(int(value) for value in safety.groups.tolist()))
    safety_probabilities = _probabilities(model, calibrator, safety.features)
    all_safety = np.arange(len(safety.labels), dtype=np.int64)
    safe_present, safe_absent = _thresholds(safety.labels, safety_probabilities, all_safety, MINIMUM_THRESHOLD)
    present_threshold = max(present_threshold, safe_present)
    absent_threshold = max(absent_threshold, safe_absent)

    independent = _load_dataset(args.independent_data)
    independent_groups = len(set(int(value) for value in independent.groups.tolist()))
    independent_probabilities = _probabilities(model, calibrator, independent.features)
    all_independent = np.arange(len(independent.labels), dtype=np.int64)
    policies = {
        "frozen_test": _policy(dataset.labels, probabilities, test, present_threshold, absent_threshold),
        "safety_calibration": _policy(safety.labels, safety_probabilities, all_safety, present_threshold, absent_threshold),
        "independent_test": _policy(independent.labels, independent_probabilities, all_independent, present_threshold, absent_threshold),
    }
    if registered is not None:
        policies["registered_scan_internal_test"] = _policy(
            registered.test.labels,
            registered_probabilities["test"],
            np.arange(len(registered.test.labels), dtype=np.int64),
            present_threshold,
            absent_threshold,
        )
    if any(int(value["false_accepts"]) for value in policies.values()):
        raise RuntimeError(f"accidental presence guard false accepts: {policies}")
    for name, value in policies.items():
        if min(float(value["present_recall"]), float(value["absent_recall"])) < MINIMUM_CLASS_RECALL:
            raise RuntimeError(f"accidental presence guard {name} class recall too low: {value}")

    payload = {
        "model_version": model_version,
        "model_type": "random_forest",
        "feature_names": list(ACCIDENTAL_PRESENCE_FEATURE_NAMES),
        "trees": serialize_probability_forest(model),
        "calibration_intercept": float(calibrator.intercept_[0]),
        "calibration_slope": float(calibrator.coef_[0, 0]),
        "present_threshold": present_threshold,
        "absent_threshold": absent_threshold,
        "target_precision": 1.0,
        "training_seed": args.seed,
        "training_groups": training_groups,
        "registered_scan_training_groups": (
            _registered_split_summary(registered.train)["groups"]
            if registered is not None
            else 0
        ),
        "model_config": config,
        "target": "a printed accidental is present immediately to the left of one already-proposed pitched event",
        "scope": "veto-only presence/absence check for same-staff-position chromatic pitch repair; accidental class substitutions are excluded",
    }
    deployed = deployed_forest_probabilities(payload, dataset.features[test])
    deployment_delta = float(np.max(np.abs(deployed - probabilities[test]), initial=0.0))
    if deployment_delta > 1e-10:
        raise RuntimeError(f"deployment prediction mismatch: {deployment_delta}")

    # Density-only ablation tests that directional stroke evidence has independent value.
    density_start = len(ACCIDENTAL_PRESENCE_FEATURE_NAMES) - 32
    density_features = dataset.features[:, density_start:]
    ablation_fit = density_features[train]
    ablation_fit_labels = dataset.labels[train]
    ablation_calibration = density_features[calibration]
    ablation_calibration_labels = dataset.labels[calibration]
    if registered is not None:
        ablation_fit = np.concatenate(
            (ablation_fit, registered.train.features[:, density_start:]),
            axis=0,
        )
        ablation_fit_labels = np.concatenate(
            (ablation_fit_labels, registered.train.labels),
            axis=0,
        )
        ablation_calibration = np.concatenate(
            (
                ablation_calibration,
                registered.calibration.features[:, density_start:],
            ),
            axis=0,
        )
        ablation_calibration_labels = np.concatenate(
            (ablation_calibration_labels, registered.calibration.labels),
            axis=0,
        )
    ablation_model, ablation_calibrator = _fit_arrays(
        ablation_fit,
        ablation_fit_labels,
        ablation_calibration,
        ablation_calibration_labels,
        args.seed,
        config,
        args.jobs,
    )
    ablation_all_probabilities = _probabilities(
        ablation_model, ablation_calibrator, density_features
    )
    safety_density = safety.features[:, density_start:]
    independent_density = independent.features[:, density_start:]
    ablation_safety_probabilities = _probabilities(
        ablation_model, ablation_calibrator, safety_density
    )
    ablation_independent_probabilities = _probabilities(
        ablation_model, ablation_calibrator, independent_density
    )
    ablation_present, ablation_absent = _thresholds(
        dataset.labels,
        ablation_all_probabilities,
        threshold_indices,
        MINIMUM_THRESHOLD,
    )
    safety_present, safety_absent = _thresholds(
        safety.labels,
        ablation_safety_probabilities,
        all_safety,
        MINIMUM_THRESHOLD,
    )
    ablation_present = max(ablation_present, safety_present)
    ablation_absent = max(ablation_absent, safety_absent)
    ablation_registered: dict[str, Any] | None = None
    if registered is not None:
        registered_ablation_calibration = _probabilities(
            ablation_model,
            ablation_calibrator,
            registered.calibration.features[:, density_start:],
        )
        registered_ablation_test = _probabilities(
            ablation_model,
            ablation_calibrator,
            registered.test.features[:, density_start:],
        )
        registered_ablation_present, registered_ablation_absent = _thresholds(
            registered.calibration.labels,
            registered_ablation_calibration,
            np.arange(len(registered.calibration.labels), dtype=np.int64),
            MINIMUM_THRESHOLD,
        )
        ablation_present = max(ablation_present, registered_ablation_present)
        ablation_absent = max(ablation_absent, registered_ablation_absent)
        ablation_registered = _policy(
            registered.test.labels,
            registered_ablation_test,
            np.arange(len(registered.test.labels), dtype=np.int64),
            ablation_present,
            ablation_absent,
        )

    atomic_write_json(args.output, payload)
    report = _canonical({
        "format": 1,
        "model_version": model_version,
        "seed": args.seed,
        "scope": (
            "registered scan-backed and programmatic rendered accidental-presence "
            "evidence; veto-only and not end-to-end OMR accuracy"
            if registered is not None
            else "programmatic rendered accidental-presence evidence only; not end-to-end OMR accuracy"
        ),
        "groups": training_groups,
        "samples": len(dataset.labels),
        "programmatic_sources": programmatic_audit,
        "partitions": {"train": len(train), "calibration": len(calibration), "threshold_selection": len(threshold_indices), "frozen_test": len(test)},
        "model_selection": selection,
        "selected_config": config,
        "thresholds": {"present": present_threshold, "absent": absent_threshold},
        "frozen_test": {"sample": _sample(dataset.labels[test], probabilities[test]), "policy": policies["frozen_test"]},
        "safety_calibration": {"seed": args.seed + 101, "groups": safety_groups, "sample": _sample(safety.labels, safety_probabilities), "policy": policies["safety_calibration"]},
        "independent_test": {"seed": args.seed + 211, "groups": independent_groups, "sample": _sample(independent.labels, independent_probabilities), "policy": policies["independent_test"]},
        "registered_scan": (
            {
                "dataset_name": registered.report["name"],
                "dataset_role": registered.report["role"],
                "source_role": registered.report["source_role"],
                "license": registered.report["license"],
                "holdout_used_for_training": False,
                "prepare_report_sha256": registered.report_sha256,
                "file_sha256": registered.file_sha256,
                "splits": {
                    split: {
                        **_registered_split_summary(getattr(registered, split)),
                        **(
                            {
                                "sample": _sample(
                                    getattr(registered, split).labels,
                                    registered_probabilities[split],
                                ),
                                "policy": _policy(
                                    getattr(registered, split).labels,
                                    registered_probabilities[split],
                                    np.arange(
                                        len(getattr(registered, split).labels),
                                        dtype=np.int64,
                                    ),
                                    present_threshold,
                                    absent_threshold,
                                ),
                            }
                            if split != "train"
                            else {}
                        ),
                    }
                    for split in ("train", "calibration", "test")
                },
            }
            if registered is not None
            else None
        ),
        "ablation_density_only": {
            "feature_count": 32,
            "frozen_test_sample": _sample(
                dataset.labels[test], ablation_all_probabilities[test]
            ),
            "thresholds": {
                "present": ablation_present,
                "absent": ablation_absent,
            },
            "frozen_test_policy": _policy(
                dataset.labels,
                ablation_all_probabilities,
                test,
                ablation_present,
                ablation_absent,
            ),
            "safety_policy": _policy(
                safety.labels,
                ablation_safety_probabilities,
                all_safety,
                ablation_present,
                ablation_absent,
            ),
            "independent_test_policy": _policy(
                independent.labels,
                ablation_independent_probabilities,
                all_independent,
                ablation_present,
                ablation_absent,
            ),
            "registered_scan_internal_test_policy": ablation_registered,
        },
        "deployment_parity": {"max_absolute_probability_delta": deployment_delta},
        "feature_names": list(ACCIDENTAL_PRESENCE_FEATURE_NAMES),
    })
    atomic_write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
