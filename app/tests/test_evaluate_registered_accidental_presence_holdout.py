from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.tools.evaluate_registered_accidental_presence_holdout import evaluate
from app.tools.train_accidental_presence_guard import (
    REGISTERED_MODEL_VERSION,
    _sha256_file,
)
from scorescan.accidental_presence_guard import ACCIDENTAL_PRESENCE_FEATURE_NAMES


def _write_npz(path: Path, labels: np.ndarray) -> None:
    features = np.zeros(
        (len(labels), len(ACCIDENTAL_PRESENCE_FEATURE_NAMES)),
        dtype=np.float64,
    )
    features[:, 0] = labels
    np.savez_compressed(
        path,
        features=features,
        labels=labels,
        groups=np.arange(1, len(labels) + 1, dtype=np.int64),
        symbols=np.asarray(
            ["present" if value else "none" for value in labels],
            dtype="<U7",
        ),
    )


def _write_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    training_dir = root / "training"
    holdout_dir = root / "holdout"
    training_dir.mkdir()
    holdout_dir.mkdir()
    for split in ("train", "calibration", "test"):
        _write_npz(
            training_dir / f"{split}.npz",
            np.asarray([0, 1, 0, 1], dtype=np.int64),
        )
    training_prepare = training_dir / "prepare-report.json"
    training_prepare.write_text(
        json.dumps({"name": "training-preparation"}),
        encoding="utf-8",
    )
    with (training_dir / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for index in range(3):
            stream.write(
                json.dumps(
                    {
                        "group_key": (
                            f"muse-omr-work/training-{index}/pair-{index:04d}/"
                            "page-1/staff-0/measure-0"
                        )
                    }
                )
                + "\n"
            )

    holdout_labels = np.asarray(
        [value for _index in range(200) for value in (0, 1)],
        dtype=np.int64,
    )
    _write_npz(holdout_dir / "test.npz", holdout_labels)
    with (holdout_dir / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for index in range(200):
            for label in (0, 1):
                stream.write(
                    json.dumps(
                        {
                            "group_key": (
                                f"muse-omr-work/holdout-{index}/pair-{index:04d}/"
                                f"page-1/staff-0/measure-{label}"
                            )
                        }
                    )
                    + "\n"
                )
    holdout_prepare = {
        "name": "scorescan-registered-scan-accidental-presence-holdout-v1",
        "role": "independent_holdout_evaluation_only",
        "source_role": (
            "external_scan_degraded_development_benchmark_not_training"
        ),
        "training_use_authorized": False,
        "holdout_used_for_training": False,
        "samples_by_split": {"train": 0, "calibration": 0, "test": 400},
        "positive_samples_by_split": {
            "train": 0,
            "calibration": 0,
            "test": 200,
        },
        "works_by_split": {"train": 0, "calibration": 0, "test": 200},
        "work_intersections": {
            "train_calibration": [],
            "train_test": [],
            "calibration_test": [],
        },
    }
    (holdout_dir / "prepare-report.json").write_text(
        json.dumps(holdout_prepare),
        encoding="utf-8",
    )

    model_path = root / "model.json"
    model_path.write_text(
        json.dumps(
            {
                "model_version": REGISTERED_MODEL_VERSION,
                "feature_names": list(ACCIDENTAL_PRESENCE_FEATURE_NAMES),
                "trees": [
                    {
                        "nodes": [
                            {
                                "feature": 0,
                                "threshold": 0.5,
                                "left": 1,
                                "right": 2,
                                "value": 0.5,
                            },
                            {
                                "feature": -2,
                                "threshold": -2.0,
                                "left": -1,
                                "right": -1,
                                "value": 0.0,
                            },
                            {
                                "feature": -2,
                                "threshold": -2.0,
                                "left": -1,
                                "right": -1,
                                "value": 1.0,
                            },
                        ]
                    }
                ],
                "calibration_intercept": -10.0,
                "calibration_slope": 20.0,
                "present_threshold": 0.99,
                "absent_threshold": 0.99,
            }
        ),
        encoding="utf-8",
    )
    training_report_path = root / "training-report.json"
    training_report_path.write_text(
        json.dumps(
            {
                "model_version": REGISTERED_MODEL_VERSION,
                "registered_scan": {
                    "prepare_report_sha256": _sha256_file(training_prepare),
                    "file_sha256": {
                        f"{split}.npz": _sha256_file(
                            training_dir / f"{split}.npz"
                        )
                        for split in ("train", "calibration", "test")
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return model_path, training_report_path, training_dir, holdout_dir


def test_frozen_holdout_requires_zero_false_accepts_and_200_works(
    tmp_path: Path,
) -> None:
    model, training_report, training_dir, holdout_dir = _write_fixture(tmp_path)
    report = evaluate(
        model_path=model,
        training_report_path=training_report,
        registered_training_dir=training_dir,
        holdout_dir=holdout_dir,
        minimum_works=200,
        minimum_roc_auc=0.94,
        minimum_class_recall=0.30,
    )
    assert report["acceptance"]["passed"]
    assert report["policy"]["false_accepts"] == 0
    assert report["holdout_works"] == 200
    assert report["work_overlap"] == []


def test_frozen_holdout_rejects_work_overlap(tmp_path: Path) -> None:
    model, training_report, training_dir, holdout_dir = _write_fixture(tmp_path)
    metadata = holdout_dir / "samples.jsonl"
    rows = metadata.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["group_key"] = (
        "muse-omr-work/training-0/pair-9999/page-1/staff-0/measure-0"
    )
    rows[0] = json.dumps(first)
    metadata.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps training works"):
        evaluate(
            model_path=model,
            training_report_path=training_report,
            registered_training_dir=training_dir,
            holdout_dir=holdout_dir,
            minimum_works=200,
            minimum_roc_auc=0.94,
            minimum_class_recall=0.30,
        )
