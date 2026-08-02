from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from scorescan.accidental_presence_guard import (  # noqa: E402
    ACCIDENTAL_PRESENCE_FEATURE_NAMES,
)
from prepare_accidental_presence_programmatic import prepare  # noqa: E402
from train_accidental_presence_guard import (  # noqa: E402
    REGISTERED_DATASET_NAME,
    REGISTERED_DATASET_ROLE,
    REGISTERED_SOURCE_ROLE,
    _load_registered_training_bundle,
    _validate_programmatic_report,
)


def _write_registered_dataset(root: Path) -> None:
    feature_count = len(ACCIDENTAL_PRESENCE_FEATURE_NAMES)
    split_counts = {"train": 8, "calibration": 6, "test": 4}
    positives: dict[str, int] = {}
    groups: dict[str, int] = {}
    for split, count in split_counts.items():
        labels = np.asarray([index % 2 for index in range(count)], dtype=np.int64)
        group_ids = np.asarray(
            [index // 2 + 1 for index in range(count)],
            dtype=np.int64,
        )
        np.savez_compressed(
            root / f"{split}.npz",
            features=np.zeros((count, feature_count), dtype=np.float64),
            labels=labels,
            groups=group_ids,
            symbols=np.asarray(
                ["present" if value else "none" for value in labels],
                dtype="<U7",
            ),
        )
        positives[split] = int(np.sum(labels))
        groups[split] = len(set(group_ids.tolist()))
    report = {
        "format": 1,
        "name": REGISTERED_DATASET_NAME,
        "role": REGISTERED_DATASET_ROLE,
        "source_role": REGISTERED_SOURCE_ROLE,
        "license": "CC0-1.0",
        "feature_names": list(ACCIDENTAL_PRESENCE_FEATURE_NAMES),
        "samples_by_split": split_counts,
        "positive_samples_by_split": positives,
        "groups_by_split": groups,
        "works_by_split": {"train": 3, "calibration": 2, "test": 2},
        "work_intersections": {
            "train_calibration": [],
            "train_test": [],
            "calibration_test": [],
        },
        "holdout_used_for_training": False,
    }
    (root / "prepare-report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )


def test_registered_training_bundle_binds_all_fixed_splits(tmp_path: Path) -> None:
    _write_registered_dataset(tmp_path)
    loaded = _load_registered_training_bundle(tmp_path)
    assert len(loaded.train.labels) == 8
    assert len(loaded.calibration.labels) == 6
    assert len(loaded.test.labels) == 4
    assert set(loaded.file_sha256) == {
        "train.npz",
        "calibration.npz",
        "test.npz",
    }
    assert len(loaded.report_sha256) == 64


def test_registered_training_bundle_rejects_holdout_role(tmp_path: Path) -> None:
    _write_registered_dataset(tmp_path)
    report_path = tmp_path / "prepare-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["role"] = "independent_holdout_evaluation_only"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="not training-role"):
        _load_registered_training_bundle(tmp_path)


def test_registered_training_bundle_rejects_stale_counts(tmp_path: Path) -> None:
    _write_registered_dataset(tmp_path)
    report_path = tmp_path / "prepare-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["samples_by_split"]["test"] += 1
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="sample count is stale"):
        _load_registered_training_bundle(tmp_path)


def test_registered_training_bundle_rejects_work_leakage(tmp_path: Path) -> None:
    _write_registered_dataset(tmp_path)
    report_path = tmp_path / "prepare-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["work_intersections"]["train_test"] = ["forbidden-work"]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="work-level split leakage"):
        _load_registered_training_bundle(tmp_path)


def test_programmatic_report_binds_all_three_regression_splits(
    tmp_path: Path,
) -> None:
    config = {
        "train": {"seed": 101, "groups": 2},
        "safety": {"seed": 102, "groups": 2},
        "independent": {"seed": 103, "groups": 2},
    }
    report = prepare(tmp_path, config)
    audit = _validate_programmatic_report(
        tmp_path / "prepare-report.json",
        train_path=tmp_path / "train.npz",
        safety_path=tmp_path / "safety.npz",
        independent_path=tmp_path / "independent.npz",
    )
    assert audit["prepare_report_sha256"]
    assert report["splits"]["train"]["groups"] == 2
    assert audit["independent_real_scan_holdout"] is False

    with (tmp_path / "train.npz").open("ab") as stream:
        stream.write(b"stale")
    with pytest.raises(ValueError, match="train hash is stale"):
        _validate_programmatic_report(
            tmp_path / "prepare-report.json",
            train_path=tmp_path / "train.npz",
            safety_path=tmp_path / "safety.npz",
            independent_path=tmp_path / "independent.npz",
        )
