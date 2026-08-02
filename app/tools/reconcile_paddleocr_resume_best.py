#!/usr/bin/env python3
"""Reconcile a PaddleOCR best model overwritten by the legacy resume reset."""

from __future__ import annotations

import argparse
import math
import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_SRC = PROJECT_ROOT / "app" / "src"
if str(APP_SRC) not in sys.path:
    sys.path.insert(0, str(APP_SRC))

from app.tools.gate_paddleocr_evaluation import parse_metrics
from scorescan.util import (
    atomic_write_bytes,
    atomic_write_json,
    replace_file_with_retry,
    sha256_file,
    utc_now_iso,
)


def _read_state(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    state = pickle.loads(path.read_bytes())
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint state is not a dictionary: {path}")
    return state


def _metric(mapping: dict[str, Any], name: str) -> float:
    value = float(mapping[name])
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"invalid {name}: {value}")
    return value


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(file_descriptor, "wb") as target, source.open("rb") as stream:
            shutil.copyfileobj(stream, target, length=4 * 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        replace_file_with_retry(temporary_name, destination)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def reconcile(
    *,
    latest_prefix: Path,
    best_prefix: Path,
    evaluation_log: Path,
    expected_latest_sha256: str,
    expected_epoch: int,
    minimum_accuracy: float,
    minimum_normalized_edit: float,
    execute: bool,
) -> dict[str, Any]:
    latest_params = latest_prefix.with_suffix(".pdparams")
    latest_states = latest_prefix.with_suffix(".states")
    best_params = best_prefix.with_suffix(".pdparams")
    best_states = best_prefix.with_suffix(".states")
    for path in (latest_params, best_params):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)

    latest_hash = sha256_file(latest_params)
    if latest_hash != expected_latest_sha256.lower():
        raise ValueError("latest PaddleOCR parameter hash changed")
    latest_state = _read_state(latest_states)
    overwritten_best_state = _read_state(best_states)
    if int(latest_state.get("epoch", -1)) != expected_epoch:
        raise ValueError("latest PaddleOCR epoch changed")

    stored_best = latest_state.get("best_model_dict")
    overwritten_best = overwritten_best_state.get("best_model_dict")
    if not isinstance(stored_best, dict) or not isinstance(overwritten_best, dict):
        raise ValueError("PaddleOCR best-model metadata is missing")
    stored_accuracy = _metric(stored_best, "acc")
    overwritten_accuracy = _metric(overwritten_best, "acc")
    metrics = parse_metrics(evaluation_log)
    if (
        metrics["acc"] < minimum_accuracy
        or metrics["norm_edit_dis"] < minimum_normalized_edit
    ):
        raise ValueError("latest checkpoint failed the reconciliation floor")
    if overwritten_accuracy >= metrics["acc"]:
        raise ValueError("selected best artifact is not demonstrably degraded")
    if stored_accuracy <= metrics["acc"]:
        raise ValueError("resume metadata does not contain a lost prior best")

    updated_best = dict(stored_best)
    updated_best.update(metrics)
    updated_best["best_epoch"] = expected_epoch
    reconciled_latest_state = dict(latest_state)
    reconciled_latest_state["best_model_dict"] = updated_best
    reconciled_best_state = {
        "best_model_dict": updated_best,
        "epoch": expected_epoch,
        "global_step": int(latest_state.get("global_step", 0)),
    }
    report = {
        "format": 1,
        "name": "scorescan-paddleocr-resume-best-reconciliation-v1",
        "generated_at_utc": utc_now_iso(),
        "executed": execute,
        "reason": "legacy_resume_reset_overwrote_best_with_lower_accuracy",
        "latest": {
            "parameters": str(latest_params.resolve()),
            "sha256": latest_hash,
            "epoch": expected_epoch,
            "global_step": int(latest_state.get("global_step", 0)),
            "stored_best_metrics": {
                "acc": stored_accuracy,
                "norm_edit_dis": _metric(stored_best, "norm_edit_dis"),
            },
            "independent_current_weight_metrics": metrics,
        },
        "overwritten_best": {
            "parameters": str(best_params.resolve()),
            "sha256": sha256_file(best_params),
            "metrics": {
                "acc": overwritten_accuracy,
                "norm_edit_dis": _metric(overwritten_best, "norm_edit_dis"),
            },
        },
        "reconciled_best_metrics": metrics,
    }
    if execute:
        _atomic_copy(latest_params, best_params)
        atomic_write_bytes(
            latest_states,
            pickle.dumps(reconciled_latest_state, protocol=2),
        )
        atomic_write_bytes(
            best_states,
            pickle.dumps(reconciled_best_state, protocol=2),
        )
        report["reconciled_best_sha256"] = sha256_file(best_params)
        if report["reconciled_best_sha256"] != latest_hash:
            raise RuntimeError("reconciled best parameter copy is inconsistent")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latest-prefix", type=Path, required=True)
    parser.add_argument("--best-prefix", type=Path, required=True)
    parser.add_argument("--evaluation-log", type=Path, required=True)
    parser.add_argument("--expected-latest-sha256", required=True)
    parser.add_argument("--expected-epoch", type=int, required=True)
    parser.add_argument("--minimum-accuracy", type=float, required=True)
    parser.add_argument("--minimum-normalized-edit", type=float, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = reconcile(
        latest_prefix=args.latest_prefix,
        best_prefix=args.best_prefix,
        evaluation_log=args.evaluation_log,
        expected_latest_sha256=args.expected_latest_sha256,
        expected_epoch=args.expected_epoch,
        minimum_accuracy=args.minimum_accuracy,
        minimum_normalized_edit=args.minimum_normalized_edit,
        execute=args.execute,
    )
    atomic_write_json(args.output_report, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
