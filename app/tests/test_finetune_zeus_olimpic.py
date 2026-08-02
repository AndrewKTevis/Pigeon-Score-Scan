from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.tools.finetune_zeus_olimpic import (
    build_parser,
    family_regression_gate,
    load_and_validate_manifest,
    qualifies_for_selection,
    sha256_file,
    token_family_metrics,
)


def _make_prepared(
    tmp_path: Path,
    *,
    overlap: bool = False,
    document_overlap: bool = False,
) -> Path:
    splits = {}
    groups = {
        "train": ["a"],
        "calibration": ["a" if overlap else "b"],
        "candidate_test": ["c"],
    }
    for split, group_ids in groups.items():
        payload = f"{split}-pickle".encode()
        (tmp_path / f"{split}.pickle").write_bytes(payload)
        splits[split] = {
            "group_ids": group_ids,
            "pickle_sha256": hashlib.sha256(payload).hexdigest(),
            "source_document_ids": [
                "doc-a"
                if split == "train"
                or (split == "calibration" and document_overlap)
                else f"doc-{split}"
            ],
        }
    manifest = {
        "name": "test",
        "source_group_overlap": {
            "train_calibration": int(overlap),
            "train_candidate_test": 0,
            "calibration_candidate_test": 0,
        },
        "source_document_overlap": {
            "train_calibration": int(document_overlap),
            "train_candidate_test": 0,
            "calibration_candidate_test": 0,
        },
        "splits": splits,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "data"
    path.write_bytes(b"score-scan")
    assert sha256_file(path) == hashlib.sha256(b"score-scan").hexdigest()


def test_manifest_accepts_isolated_verified_splits(tmp_path: Path) -> None:
    manifest = load_and_validate_manifest(_make_prepared(tmp_path))
    assert manifest["name"] == "test"


def test_manifest_rejects_group_leakage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="leakage|share source groups"):
        load_and_validate_manifest(_make_prepared(tmp_path, overlap=True))


def test_manifest_rejects_physical_source_document_leakage(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="source-document leakage"):
        load_and_validate_manifest(
            _make_prepared(tmp_path, document_overlap=True)
        )


def test_manifest_rejects_changed_pickle(tmp_path: Path) -> None:
    prepared = _make_prepared(tmp_path)
    (prepared / "train.pickle").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_and_validate_manifest(prepared)


def test_model_selection_requires_a_material_calibration_gain() -> None:
    assert not qualifies_for_selection(
        current_ser=15.44,
        best_ser=15.45,
        baseline_ser=15.45,
        minimum_improvement=0.10,
    )
    assert qualifies_for_selection(
        current_ser=15.20,
        best_ser=15.45,
        baseline_ser=15.45,
        minimum_improvement=0.10,
    )


def test_mixed_float16_is_an_explicit_gpu_training_option() -> None:
    parser = build_parser()
    arguments = parser.parse_args(
        [
            "--upstream-dir",
            "upstream",
            "--base-model",
            "base",
            "--prepared-dir",
            "prepared",
            "--output-dir",
            "output",
            "--precision",
            "mixed_float16",
        ]
    )
    assert arguments.precision == "mixed_float16"


def test_token_family_metrics_penalize_wrong_symbol_attachment() -> None:
    metrics = token_family_metrics(
        ["measure C4 quarter slur:start D4 quarter slur:stop tied:start"],
        ["measure C4 quarter D4 quarter slur:start slur:stop tied:start"],
    )
    assert metrics["slur"]["f1_percent"] == pytest.approx(100.0)
    assert metrics["slur"]["positioned_f1_percent"] == pytest.approx(50.0)
    assert metrics["tie"]["positioned_f1_percent"] == pytest.approx(100.0)


def test_token_family_metrics_reports_rhythm_and_pitch_errors() -> None:
    metrics = token_family_metrics(
        ["measure C4 quarter D4 eighth"],
        ["measure C4 quarter E4 quarter"],
    )
    assert metrics["pitch"]["f1_percent"] == pytest.approx(50.0)
    assert metrics["rhythm"]["recall_percent"] == pytest.approx(50.0)


def test_family_regression_gate_rejects_hidden_local_degradation() -> None:
    baseline = token_family_metrics(
        ["C4 quarter slur:start D4 quarter slur:stop"],
        ["C4 quarter slur:start D4 quarter slur:stop"],
    )
    candidate = token_family_metrics(
        ["C4 quarter slur:start D4 quarter slur:stop"],
        ["C4 quarter D4 quarter slur:start slur:stop"],
    )
    accepted, regressions = family_regression_gate(
        baseline_metrics={"families": baseline},
        candidate_metrics={"families": candidate},
        maximum_regression=0.5,
    )
    assert not accepted
    assert regressions["slur"] == pytest.approx(50.0)
