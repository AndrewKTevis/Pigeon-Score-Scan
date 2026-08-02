from __future__ import annotations

"""Audit the narrow implicit-triplet transaction on positive and negative works."""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.evaluation import compare_musicxml  # noqa: E402
from scorescan.implicit_triplet_transaction import (  # noqa: E402
    apply_evidence_confirmed_continuous_triplet_grid,
    detect_continuous_triplet_grid_evidence,
)
from scorescan.util import atomic_write_json, sha256_file  # noqa: E402


def run(args: argparse.Namespace) -> dict[str, object]:
    positive = args.positive.resolve()
    reference = args.reference.resolve()
    output = args.output.resolve()
    evidence, transaction = apply_evidence_confirmed_continuous_triplet_grid(
        positive,
        output,
    )
    evaluation = compare_musicxml(reference, output) if transaction.applied else None

    negative_rows: list[dict[str, object]] = []
    for path in sorted(args.negative_root.resolve().glob(args.negative_glob)):
        item = detect_continuous_triplet_grid_evidence(path)
        negative_rows.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "authorized": item.authorized,
                "evidence": item.to_dict(),
            }
        )

    false_positive_count = sum(bool(row["authorized"]) for row in negative_rows)
    return {
        "schema": "scorescan-implicit-triplet-transaction-audit@1",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "diagnostic_only": True,
        "production_evidence_eligible": False,
        "positive": {
            "path": str(positive),
            "sha256": sha256_file(positive),
            "reference": str(reference),
            "reference_sha256": sha256_file(reference),
            "output": str(output),
            "output_sha256": sha256_file(output) if output.is_file() else None,
            "evidence": evidence.to_dict(),
            "transaction": transaction.to_dict(),
            "evaluation": evaluation,
        },
        "negative_set": {
            "description": "independent Muse-OMR works whose first declared meter is true 3/2",
            "count": len(negative_rows),
            "false_positive_count": false_positive_count,
            "true_negative_count": len(negative_rows) - false_positive_count,
            "rows": negative_rows,
        },
        "audit_passed": bool(
            transaction.applied
            and evaluation is not None
            and false_positive_count == 0
            and len(negative_rows) >= 6
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-root", type=Path, required=True)
    parser.add_argument("--negative-glob", default="score_file_*.musicxml")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    atomic_write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
