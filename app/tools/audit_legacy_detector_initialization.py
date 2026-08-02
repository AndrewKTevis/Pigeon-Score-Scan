from __future__ import annotations

"""Audit a completed legacy detector strictly as initialization weights."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.util import atomic_write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def audit(
    *,
    model_path: Path,
    metrics_path: Path,
    run_config_path: Path,
    prepared_manifest_path: Path,
    minimum_epochs: int,
    minimum_map_50: float,
    minimum_map_75: float,
    minimum_priority_map: float,
) -> dict[str, Any]:
    model_path = model_path.resolve(strict=True)
    metrics_path = metrics_path.resolve(strict=True)
    run_config_path = run_config_path.resolve(strict=True)
    prepared_manifest_path = prepared_manifest_path.resolve(strict=True)
    metrics = _load_json(metrics_path)
    run_config = _load_json(run_config_path)
    epochs = metrics.get("epochs")
    if not isinstance(epochs, list) or len(epochs) < minimum_epochs:
        raise ValueError("legacy detector has insufficient completed epochs")
    if any(
        not isinstance(record, dict)
        or int(record.get("epoch", -1)) != index
        for index, record in enumerate(epochs, start=1)
    ):
        raise ValueError("legacy detector epoch sequence is incomplete")
    evaluated = [
        record
        for record in epochs
        if isinstance(record.get("test"), dict)
    ]
    if not evaluated:
        raise ValueError("legacy detector has no completed evaluation")
    best = max(
        evaluated,
        key=lambda record: float(record["test"].get("selection_score", -1.0)),
    )
    test = best["test"]
    floors = {
        "map_50": minimum_map_50,
        "map_75": minimum_map_75,
        "priority_mark_map": minimum_priority_map,
    }
    failures = [
        f"{name}={float(test.get(name, -1.0)):.6f} < {floor:.6f}"
        for name, floor in floors.items()
        if float(test.get(name, -1.0)) < floor
    ]
    if failures:
        raise ValueError(
            "legacy detector is too weak even for initialization: "
            + "; ".join(failures)
        )
    expected_manifest_hash = str(
        run_config.get("prepared_manifest_sha256", "")
    )
    actual_manifest_hash = _sha256(prepared_manifest_path)
    if expected_manifest_hash != actual_manifest_hash:
        raise ValueError("legacy detector belongs to another prepared dataset")
    legacy_contract_missing = not isinstance(
        run_config.get("model_contract"),
        dict,
    )
    if not legacy_contract_missing:
        raise ValueError(
            "current-contract detector should produce a normal training report"
        )
    return {
        "format": 1,
        "name": "scorescan-legacy-detector-initialization-audit-v1",
        "purpose": "initialization_weights_only",
        "passed": True,
        "deployment_eligible": False,
        "release_accuracy_evidence": False,
        "legacy_model_contract_missing": True,
        "requires_current_contract_retraining_and_evaluation": True,
        "model_sha256": _sha256(model_path),
        "metrics_sha256": _sha256(metrics_path),
        "run_config_sha256": _sha256(run_config_path),
        "prepared_manifest_sha256": actual_manifest_hash,
        "completed_epochs": len(epochs),
        "best_epoch": int(best["epoch"]),
        "best_metrics": {
            name: float(test[name])
            for name in (
                "map",
                "map_50",
                "map_75",
                "priority_mark_map",
                "selection_score",
            )
        },
        "initialization_floors": floors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-epochs", type=int, default=6)
    parser.add_argument("--minimum-map-50", type=float, default=0.70)
    parser.add_argument("--minimum-map-75", type=float, default=0.65)
    parser.add_argument("--minimum-priority-map", type=float, default=0.50)
    args = parser.parse_args()
    report = audit(
        model_path=args.model,
        metrics_path=args.metrics,
        run_config_path=args.run_config,
        prepared_manifest_path=args.prepared_manifest,
        minimum_epochs=args.minimum_epochs,
        minimum_map_50=args.minimum_map_50,
        minimum_map_75=args.minimum_map_75,
        minimum_priority_map=args.minimum_priority_map,
    )
    atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
