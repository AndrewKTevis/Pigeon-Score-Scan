from __future__ import annotations

"""Deterministic audit for the one-family, multi-treatment measure rescue gate."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scorescan.measure_localized import (  # noqa: E402
    MeasureLocalizedVariantResult,
    choose_measure_localized_variant,
)
from scorescan.policy import DEFAULT_POLICY  # noqa: E402
from scorescan.util import atomic_write_json  # noqa: E402


def _item(name: str, signature: str | None, *, valid: bool = True) -> MeasureLocalizedVariantResult:
    return MeasureLocalizedVariantResult(
        name=name,
        image_path=f"{name}.png",
        xml_path=f"{name}.musicxml" if valid else None,
        return_code=0 if valid else 65,
        elapsed_seconds=0.01,
        valid=valid,
        observed_measure_count=1 if valid else 0,
        note_count=4 if valid else 0,
        local_rhythm_issue_count=0,
        content_signature=signature if valid else None,
        error=None if valid else "invalid",
    )


def audit() -> dict[str, object]:
    scenarios = (
        ("all-agree-correct", ("truth", "truth", "truth"), (True, True, True), "truth"),
        ("primary-wrong-two-correct", ("wrong", "truth", "truth"), (True, True, True), "truth"),
        ("primary-invalid-two-correct", (None, "truth", "truth"), (False, True, True), "truth"),
        ("flat-wrong-two-correct", ("truth", "wrong", "truth"), (True, True, True), "truth"),
        ("otsu-invalid-two-correct", ("truth", "truth", None), (True, True, False), "truth"),
        ("three-way-split", ("a", "b", "c"), (True, True, True), None),
        ("only-one-valid", ("truth", None, None), (True, False, False), None),
        # These common-mode cases remain intentionally outside the internal gate's
        # authority and must still be caught by the ordinary three-family/page-level
        # verifier.  Recording them prevents the audit from overstating this change.
        ("two-correlated-wrong", ("truth", "wrong", "wrong"), (True, True, True), "wrong"),
        ("all-correlated-wrong", ("wrong", "wrong", "wrong"), (True, True, True), "wrong"),
    )
    rows: list[dict[str, object]] = []
    passed = 0
    for name, signatures, validity, expected in scenarios:
        variants = tuple(
            _item(variant, signature, valid=valid)
            for variant, signature, valid in zip(("primary", "flat", "otsu"), signatures, validity)
        )
        selected, support, signature, error = choose_measure_localized_variant(variants)
        observed = selected.content_signature if selected is not None else None
        ok = observed == expected
        passed += int(ok)
        rows.append(
            {
                "name": name,
                "expected_signature": expected,
                "observed_signature": observed,
                "winning_support": support,
                "reported_signature": signature,
                "error": error,
                "passed": ok,
            }
        )
    return {
        "format": 1,
        "policy_version": DEFAULT_POLICY.version,
        "subvariants": ["primary", "flat", "otsu"],
        "independent_family_count_contributed": 1,
        "scenario_count": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "scenarios": rows,
        "accepted_capabilities": {
            "primary_error_overruled_by_two_matching_siblings": True,
            "primary_failure_recovered_by_two_matching_siblings": True,
            "three_way_split_rejected": True,
            "single_valid_result_rejected": True,
        },
        "known_limits": [
            "Two related local treatments can still share the same error.",
            "All three local treatments can still share the same error.",
            "The local result therefore remains one family and must pass the ordinary independent-family and page-level veto gates.",
        ],
        "programmatic_audit": True,
        "end_to_end_accuracy_claim": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit()
    if int(report["failed"]) != 0:
        raise SystemExit("measure-localised internal consensus audit failed")
    atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
