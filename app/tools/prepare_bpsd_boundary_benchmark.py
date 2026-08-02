from __future__ import annotations

"""Refreeze acquired BPSD pairs as a physical-scan diagnostic benchmark."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_ROOT / "src"))

from app.tools.acquire_bpsd_piano_scan_corpus import (  # noqa: E402
    EXPECTED_WORKS,
    LICENSE_ID,
    RECORD_ID,
    ROLE as ACQUISITION_ROLE,
)
from scorescan.product_scope import (  # noqa: E402
    PRODUCTION_BOUNDARY_CONTRACT_VERSION,
)
from scorescan.util import atomic_write_json, sha256_file, utc_now_iso  # noqa: E402


ROLE = "external_real_scan_diagnostic_only_bpsd_ccby3_keyboard_v1"


def _stable_fingerprint(work: str) -> str:
    return hashlib.sha256(
        f"zenodo:{RECORD_ID}:bpsd:{work}".encode()
    ).hexdigest()


def prepare(
    acquisition_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    acquisition_path = acquisition_path.resolve()
    output_path = output_path.resolve()
    payload = json.loads(
        acquisition_path.read_text(encoding="utf-8")
    )
    cases = payload.get("cases")
    if (
        not isinstance(payload, dict)
        or payload.get("role") != ACQUISITION_ROLE
        or payload.get("record_id") != RECORD_ID
        or payload.get("record_license") != LICENSE_ID
        or payload.get("work_count") != EXPECTED_WORKS
        or payload.get("source_image_origin") not in {None, "physical_scan"}
        or payload.get("training_authorized") is not False
        or payload.get("release_authorized") is not False
        or not isinstance(cases, list)
        or len(cases) != EXPECTED_WORKS
    ):
        raise ValueError("BPSD acquisition manifest contract is invalid")

    selected: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    seen_works: set[str] = set()
    seen_fingerprints: set[str] = set()
    for row in cases:
        if not isinstance(row, dict):
            raise ValueError("BPSD acquisition case is invalid")
        work = str(row.get("work") or "")
        scan_path = Path(str(row.get("scan_path") or "")).resolve()
        reference_path = Path(
            str(row.get("reference_musicxml_path") or "")
        ).resolve()
        boundary = row.get("boundary")
        fingerprint = _stable_fingerprint(work)
        if (
            not work
            or work in seen_works
            or fingerprint in seen_fingerprints
            or not scan_path.is_file()
            or not reference_path.is_file()
            or row.get("scan_sha256") != sha256_file(scan_path)
            or row.get("reference_musicxml_sha256")
            != sha256_file(reference_path)
            or isinstance(row.get("pages"), bool)
            or not isinstance(row.get("pages"), int)
            or int(row["pages"]) <= 0
            or not isinstance(boundary, dict)
            or boundary.get("contract_version")
            != PRODUCTION_BOUNDARY_CONTRACT_VERSION
        ):
            raise ValueError("BPSD acquisition case hash or schema is invalid")
        seen_works.add(work)
        seen_fingerprints.add(fingerprint)
        eligible = bool(
            row.get("boundary_eligible_alignment_candidate") is True
            and row.get("reference_quarantined") is False
            and boundary.get("accepted") is True
            and boundary.get("score_shape") == "keyboard"
            and row.get("same_named_work_pair") is True
            and row.get("exact_page_measure_alignment_verified") is False
        )
        if not eligible:
            quarantined.append(
                {
                    "work": work,
                    "reasons": list(
                        row.get("reference_quarantine_reasons") or ()
                    ),
                }
            )
            continue
        selected.append(
            {
                "id": f"bpsd-{work.casefold()}",
                "work": work,
                "work_fingerprint": fingerprint,
                "input_pdf": str(scan_path),
                "input_pdf_sha256": row["scan_sha256"],
                "input_pdf_pages": int(row["pages"]),
                "reference": str(reference_path),
                "reference_sha256": row[
                    "reference_musicxml_sha256"
                ],
                "boundary": boundary,
                "same_named_work_pair": True,
                "exact_page_measure_alignment_verified": False,
                "independent_double_annotation_complete": False,
                "training_authorized": False,
                "evaluation_authorized": False,
                "release_authorized": False,
            }
        )
    if (
        not selected
        or len(selected) + len(quarantined) != EXPECTED_WORKS
        or len(selected)
        != int(
            payload.get(
                "boundary_eligible_alignment_candidate_count",
                -1,
            )
        )
        or len(quarantined)
        != int(payload.get("quarantined_reference_count", -1))
    ):
        raise ValueError("BPSD boundary selection counts are inconsistent")
    selected.sort(key=lambda row: (row["work"], row["id"]))
    quarantined.sort(key=lambda row: row["work"])
    result = {
        "format": 1,
        "created_at": utc_now_iso(),
        "name": "BPSD physical-scan keyboard diagnostic benchmark",
        "role": ROLE,
        "source_image_origin": "physical_scan",
        "production_evidence_eligible": False,
        "boundary_contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
        "source_acquisition_manifest": str(acquisition_path),
        "source_acquisition_manifest_sha256": sha256_file(
            acquisition_path
        ),
        "source_record_id": RECORD_ID,
        "source_license": LICENSE_ID,
        "selection_used_model_outputs": False,
        "same_named_work_pairing_required": True,
        "exact_page_measure_alignment_verified": False,
        "independent_double_annotation_complete": False,
        "selected_work_count": len(selected),
        "selected_physical_scan_page_count": sum(
            int(row["input_pdf_pages"]) for row in selected
        ),
        "quarantined_work_count": len(quarantined),
        "quarantined_references": quarantined,
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_evaluation_authorized": False,
        "release_authorized": False,
        "authorization_reason": (
            "development diagnostic only: official physical scans and "
            "same-named MusicXML are hash verified and inside the keyboard "
            "boundary, but page/measure alignment and independent double "
            "annotation are incomplete"
        ),
        "cases": selected,
    }
    atomic_write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("acquisition_manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.acquisition_manifest, args.output)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "selected_work_count",
                    "selected_physical_scan_page_count",
                    "quarantined_work_count",
                    "production_evidence_eligible",
                    "release_authorized",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
