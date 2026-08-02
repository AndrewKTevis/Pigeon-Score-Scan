#!/usr/bin/env python3
"""Re-score completed detector ablations under the current selection policy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import app.tools.train_deepscores_symbol_detector as selection_implementation
from app.tools.train_deepscores_symbol_detector import (
    PRIORITY_SELECTION_PROTOCOL,
    is_priority_mark_class,
    priority_selection_score,
    sha256_file,
    should_replace_detector_best,
    support_filtered_macro_map,
)


def _read_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or int(payload.get("format", 0)) != 1:
        raise ValueError(f"invalid detector training report: {path}")
    return payload


def audit_training_report(path: Path) -> dict[str, Any]:
    report = _read_report(path)
    minimum_support = int(
        report.get("configuration", {}).get(
            "minimum_required_class_test_objects",
            0,
        )
    )
    if minimum_support <= 0:
        raise ValueError(f"missing positive selection support floor: {path}")
    raw_counts = report.get("data", {}).get("test_class_counts")
    if not isinstance(raw_counts, dict):
        raise ValueError(f"missing test class support: {path}")
    support_by_label = {
        int(label): int(count) for label, count in raw_counts.items()
    }

    audited_epochs: list[dict[str, Any]] = []
    selected_epoch = -1
    selected_score = -1.0
    selected_gate_passed = False
    for record in report.get("metrics", {}).get("epochs", []):
        test = record.get("test")
        if not isinstance(test, dict):
            continue
        labels = [int(label) for label in test.get("classes", [])]
        named = test.get("map_per_class_named")
        if not isinstance(named, dict) or len(labels) != len(named):
            raise ValueError(
                f"unaligned per-class metric names in epoch {record.get('epoch')}"
            )
        class_support = {
            name: support_by_label.get(label, 0)
            for label, name in zip(labels, named)
        }
        score, priority_map = priority_selection_score(
            overall_map=float(test.get("map", -1.0)),
            per_class_map=named,
            class_support=class_support,
            minimum_support=minimum_support,
        )
        filtered_map, supported_classes = support_filtered_macro_map(
            per_class_map=named,
            class_support=class_support,
            minimum_support=minimum_support,
        )
        excluded_classes = [
            {
                "name": name,
                "support": class_support.get(name, 0),
                "map": float(value),
            }
            for name, value in named.items()
            if math.isfinite(float(value))
            and float(value) >= 0
            and class_support.get(name, 0) < minimum_support
        ]
        gate_passed = (
            test.get("acceptance_probe", {}).get("passed") is True
        )
        epoch = int(record["epoch"])
        audited_epochs.append(
            {
                "epoch": epoch,
                "gate_passed": gate_passed,
                "reported_selection_score": float(
                    test.get("selection_score", -1.0)
                ),
                "support_filtered_map": filtered_map,
                "priority_mark_map": priority_map,
                "selection_score": score,
                "supported_classes": supported_classes,
                "excluded_low_support_classes": excluded_classes,
            }
        )
        if should_replace_detector_best(
            current_gate_passed=gate_passed,
            current_selection_score=score,
            best_gate_passed=selected_gate_passed,
            best_selection_score=selected_score,
        ):
            selected_epoch = epoch
            selected_score = score
            selected_gate_passed = gate_passed

    if selected_epoch < 0:
        raise ValueError(f"training report has no evaluated epoch: {path}")
    selected = next(
        item for item in audited_epochs if item["epoch"] == selected_epoch
    )
    return {
        "training_report": str(path.resolve()),
        "training_report_sha256": sha256_file(path),
        "model_contract": report.get("model_contract"),
        "reported_selection_protocol": report.get(
            "priority_selection_protocol"
        ),
        "minimum_class_support": minimum_support,
        "reported_best_epoch": int(report.get("best_epoch", -1)),
        "reported_best_selection_score": float(
            report.get("best_selection_score", -1.0)
        ),
        "rescored_best_epoch": selected_epoch,
        "rescored_best_selection_score": selected_score,
        "rescored_best_gate_passed": selected_gate_passed,
        "rescored_best_support_filtered_map": selected[
            "support_filtered_map"
        ],
        "rescored_best_priority_mark_map": selected["priority_mark_map"],
        "best_model_sha256": report.get("best_model_sha256"),
        "training_acceptance_passed": (
            report.get("acceptance", {}).get("passed") is True
        ),
        "epochs": audited_epochs,
    }


def build_comparison(paths: list[Path]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("at least two reports are required for comparison")
    candidates = [audit_training_report(path) for path in paths]
    old_winner = max(
        candidates,
        key=lambda item: item["reported_best_selection_score"],
    )
    new_winner = max(
        candidates,
        key=lambda item: (
            item["rescored_best_gate_passed"],
            item["rescored_best_selection_score"],
        ),
    )
    return {
        "format": 1,
        "name": "scorescan-detector-selection-policy-audit-v1",
        "purpose": (
            "prevent statistically tiny classes from selecting a detector "
            "checkpoint or ablation"
        ),
        "auditor_source_sha256": sha256_file(Path(__file__)),
        "selection_implementation_source_sha256": sha256_file(
            Path(selection_implementation.__file__)
        ),
        "selection_protocol": PRIORITY_SELECTION_PROTOCOL,
        "release_authorized": False,
        "old_policy_winner_report_sha256": old_winner[
            "training_report_sha256"
        ],
        "new_policy_winner_report_sha256": new_winner[
            "training_report_sha256"
        ],
        "decision_changed": (
            old_winner["training_report_sha256"]
            != new_winner["training_report_sha256"]
        ),
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-report",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_comparison(args.training_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
