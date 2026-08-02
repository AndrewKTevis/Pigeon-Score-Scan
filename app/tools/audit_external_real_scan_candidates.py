#!/usr/bin/env python3
"""Validate external real-scan roles without silently expanding product scope."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "src"))

from scorescan.product_scope import (  # noqa: E402
    PRODUCTION_BOUNDARY_CONTRACT_VERSION,
)


NON_COMMERCIAL_LICENSE_MARKERS = ("-NC-", "NOT_DECLARED", "REVIEW_REQUIRED")


def audit_catalog(catalog: dict[str, Any]) -> dict[str, Any]:
    if catalog.get("schema_version") != 1:
        raise ValueError("unsupported external scan catalog schema")
    if (
        catalog.get("boundary_contract_version")
        != PRODUCTION_BOUNDARY_CONTRACT_VERSION
    ):
        raise ValueError("external scan catalog uses a stale boundary contract")
    candidates = catalog.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("external scan catalog is empty")
    ids = [row.get("id") for row in candidates]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("external scan catalog ids must be non-empty and unique")

    for row in candidates:
        license_name = str(row.get("license", ""))
        if row.get("distributable_product_training_authorized") is True and any(
            marker in license_name
            for marker in NON_COMMERCIAL_LICENSE_MARKERS
        ):
            raise ValueError(
                f"{row['id']} authorizes product training under {license_name}"
            )
        if row.get("final_release_evidence_authorized") is True:
            required = {
                "image_origin": "physical_scan",
                "granularity": "full_page",
                "page_semantic_ground_truth_complete": True,
                "boundary_audit_status": "accepted",
                "work_disjoint_split_verified": True,
                "distributable_product_training_authorized": False,
            }
            for field, expected in required.items():
                if row.get(field) != expected:
                    raise ValueError(
                        f"{row['id']} release evidence violates {field}"
                    )
        for flag in (
            "research_training_authorized",
            "distributable_product_training_authorized",
            "internal_diagnostic_evaluation_authorized",
            "final_release_evidence_authorized",
        ):
            if not isinstance(row.get(flag), bool):
                raise ValueError(f"{row['id']} has non-boolean {flag}")
        if not isinstance(row.get("reasons"), list) or not row["reasons"]:
            raise ValueError(f"{row['id']} is missing exclusion reasons")

    def total_pages(rows: list[dict[str, Any]]) -> int:
        return sum(
            int(row["pages_total"])
            for row in rows
            if row.get("pages_total") is not None
        )

    physical = [
        row
        for row in candidates
        if str(row.get("image_origin", "")).startswith("physical_scan")
    ]
    research_training = [
        row for row in candidates if row["research_training_authorized"]
    ]
    diagnostic = [
        row
        for row in candidates
        if row["internal_diagnostic_evaluation_authorized"]
    ]
    release = [
        row for row in candidates if row["final_release_evidence_authorized"]
    ]
    return {
        "schema_version": 1,
        "boundary_contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
        "catalog_audit_date": catalog.get("audit_date"),
        "candidate_count": len(candidates),
        "physical_full_page_candidate_pages_known": total_pages(
            [
                row
                for row in physical
                if row.get("granularity") == "full_page"
            ]
        ),
        "research_training_candidate_ids": [
            row["id"] for row in research_training
        ],
        "internal_diagnostic_candidate_ids": [
            row["id"] for row in diagnostic
        ],
        "final_release_evidence_candidate_ids": [
            row["id"] for row in release
        ],
        "final_release_evidence_pages": total_pages(release),
        "release_gate_satisfied": False,
        "release_gate_reason": (
            "No audited external source is currently authorized and complete "
            "for the frozen full-page production release benchmark."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    report = audit_catalog(catalog)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
