from __future__ import annotations

import json
from pathlib import Path

from app.tools.audit_detector_selection_policy import build_comparison
from app.tools.train_deepscores_symbol_detector import (
    PRIORITY_SELECTION_PROTOCOL,
)


def _report(
    path: Path,
    *,
    generic_ornament_map: float,
    slur_map: float,
) -> None:
    named = {
        "genericRest": 0.8,
        "genericOrnament": generic_ornament_map,
        "slur": slur_map,
    }
    old_priority = (generic_ornament_map + slur_map) / 2
    old_map = sum(named.values()) / len(named)
    old_score = 0.25 * old_map + 0.75 * old_priority
    path.write_text(
        json.dumps(
            {
                "format": 1,
                "configuration": {
                    "minimum_required_class_test_objects": 25
                },
                "data": {"test_class_counts": {"1": 100, "2": 6, "3": 80}},
                "model_contract": {"version": path.stem},
                "priority_selection_protocol": (
                    "support-filtered-priority-macro-map@1"
                ),
                "best_epoch": 2,
                "best_selection_score": old_score,
                "best_model_sha256": path.stem,
                "acceptance": {"passed": False},
                "metrics": {
                    "epochs": [
                        {
                            "epoch": 2,
                            "test": {
                                "classes": [1, 2, 3],
                                "map": old_map,
                                "map_per_class_named": named,
                                "selection_score": old_score,
                                "acceptance_probe": {"passed": False},
                            },
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )


def test_low_support_spike_cannot_choose_an_ablation(tmp_path: Path) -> None:
    stable = tmp_path / "stable.json"
    rare_spike = tmp_path / "rare-spike.json"
    _report(stable, generic_ornament_map=0.1, slur_map=0.7)
    _report(rare_spike, generic_ornament_map=1.0, slur_map=0.6)

    comparison = build_comparison([stable, rare_spike])

    assert comparison["selection_protocol"] == PRIORITY_SELECTION_PROTOCOL
    assert comparison["decision_changed"]
    old_winner = next(
        candidate
        for candidate in comparison["candidates"]
        if candidate["training_report_sha256"]
        == comparison["old_policy_winner_report_sha256"]
    )
    new_winner = next(
        candidate
        for candidate in comparison["candidates"]
        if candidate["training_report_sha256"]
        == comparison["new_policy_winner_report_sha256"]
    )
    assert Path(old_winner["training_report"]).name == "rare-spike.json"
    assert Path(new_winner["training_report"]).name == "stable.json"
    assert (
        new_winner["epochs"][0]["excluded_low_support_classes"][0]["name"]
        == "genericOrnament"
    )
