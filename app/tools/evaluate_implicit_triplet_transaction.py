from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.evaluation import compare_musicxml  # noqa: E402
from scorescan.implicit_triplet_transaction import (  # noqa: E402
    apply_confirmed_continuous_triplet_grid,
)
from scorescan.util import atomic_write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the fail-closed continuous-triplet transaction after an "
            "independent cut-time confirmation. This is a diagnostic tool, not a "
            "meter detector."
        )
    )
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    transaction = apply_confirmed_continuous_triplet_grid(
        args.candidate,
        args.output,
        confirmed_meter=(2, 2),
    )
    evaluation = compare_musicxml(args.reference, args.output) if transaction.applied else None
    report = {
        "diagnostic_only": True,
        "production_integration": False,
        "meter_evidence_supplied_by_caller": "confirmed cut time",
        "transaction": transaction.to_dict(),
        "evaluation": evaluation,
    }
    atomic_write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
