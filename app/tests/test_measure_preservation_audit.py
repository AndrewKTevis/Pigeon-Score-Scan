from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "evaluate_measure_preservation_consensus.py"
spec = importlib.util.spec_from_file_location("measure_preservation_audit", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def test_measure_preservation_audit_catches_legacy_false_majorities() -> None:
    report = module.run_audit()
    assert report["passed"] is True
    assert report["legacy_false_collapse_count"] >= 5
    assert report["three_way_conflict"]["old_exact_majority_would_form"] is True
    assert report["three_way_conflict"]["new_exact_majority_would_form"] is False
    assert report["two_family_preservation_support"]["accepted"] is True
