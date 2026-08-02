from __future__ import annotations

"""Catalog pinned Chopin Humdrum references by boundary and issue quality.

The catalog is a source-discovery aid only.  It does not prove that a matching
physical scan exists, that scan rights are suitable, or that an absence of
explicit ``:problem`` comments means the transcription is complete.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.tools.acquire_nifc_chopin_matched_scans import (  # noqa: E402
    reference_page_profile,
    reference_problem_profile,
)
from app.tools.humdrum_boundary import (  # noqa: E402
    analyze_humdrum_boundary,
    reference_records,
)
from scorescan.product_scope import PRODUCTION_BOUNDARY_CONTRACT_VERSION  # noqa: E402
from scorescan.util import atomic_write_json, sha256_file, utc_now_iso  # noqa: E402


ROLE = "nifc_chopin_reference_quality_catalog_not_training_or_evaluation"
EXPECTED_REVISION = "95dfb105c1669c72d10b04088566154f12d3dc1c"
EXPECTED_LICENSE_SHA256 = (
    "49a77b80d9c010c02c5f49f7dc7a63855128b1b591414b803ca3b507d3d466da"
)
KEYBOARD_TITLE_PATTERN = re.compile(
    r"\b(?:piano(?:fort[ée])?|pianoforte|fortepian|klavier|clavier)\b",
    re.IGNORECASE,
)
NON_KEYBOARD_OR_MULTI_PLAYER_PATTERN = re.compile(
    r"\b(?:"
    r"viol(?:in|on|oncelle|oncello)?|cello|fl[uû]te|oboe|hautbois|"
    r"clarinet|bassoon|fagott|cor|horn|trumpet|trombone|orchestre|"
    r"orchestra|voix|voice|chant|soprano|tenor|baryton|baritone|"
    r"quatre\s+mains|four\s+hands|deux\s+pianos|two\s+pianos"
    r")\b",
    re.IGNORECASE,
)


def conservative_instrumentation_hint(
    records: dict[str, str],
) -> tuple[str, str]:
    title = " ".join(
        records.get(key, "") for key in ("OTL", "PTL", "SCT")
    )
    if (
        KEYBOARD_TITLE_PATTERN.search(title) is not None
        and NON_KEYBOARD_OR_MULTI_PLAYER_PATTERN.search(title) is None
    ):
        return "1 piano", "title_explicit_single_keyboard_only"
    return "", "no_safe_instrumentation_inference"


def _revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def catalog(repository: Path) -> dict[str, object]:
    if _revision(repository) != EXPECTED_REVISION:
        raise ValueError("Chopin Humdrum repository revision drifted")
    license_path = repository / "LICENSE.txt"
    if (
        not license_path.is_file()
        or sha256_file(license_path) != EXPECTED_LICENSE_SHA256
    ):
        raise ValueError("Chopin Humdrum repository license drifted")
    kern_dir = repository / "kern"
    files = sorted(kern_dir.glob("*.krn"))
    if not files:
        raise ValueError("Chopin Humdrum repository has no kern files")
    cases: list[dict[str, object]] = []
    for path in files:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        records = reference_records(lines)
        instrumentation, inference = conservative_instrumentation_hint(
            records
        )
        boundary = analyze_humdrum_boundary(
            path,
            instrumentation=instrumentation,
            source_lines=lines,
        )
        problem = reference_problem_profile(lines)
        try:
            pages = reference_page_profile(lines)
        except ValueError:
            pages = {
                "original_page_break_count": 0,
                "terminal_trailing_page_break": False,
                "encoded_music_page_count": 0,
            }
        problem["pages_without_problem_records"] = [
            page_index
            for page_index in range(
                1,
                int(pages["encoded_music_page_count"]) + 1,
            )
            if str(page_index)
            not in problem["problem_records_per_page"]
        ]
        high_priority = (
            boundary.accepted
            and boundary.score_shape == "keyboard"
            and boundary.keyboard_part_count == 1
            and int(pages["encoded_music_page_count"]) > 0
            and int(problem["problem_record_count"]) == 0
            and bool(records.get("NIFC-shelfmark", "").strip())
        )
        cases.append(
            {
                "path": str(path.relative_to(repository)),
                "sha256": sha256_file(path),
                "records": {
                    key: records.get(key, "")
                    for key in (
                        "COM",
                        "OTL",
                        "PTL",
                        "OPS",
                        "PPR",
                        "PPP",
                        "NIFC-rismID",
                        "NIFC-shelfmark",
                        "rism-genre",
                        "rism-key",
                        "rism-opus",
                        "rism-title",
                        "rism-773a-OPR",
                        "rism-parent-permalink",
                    )
                },
                "instrumentation_hint": instrumentation,
                "instrumentation_inference": inference,
                "boundary": boundary.to_dict(),
                "reference_page_profile": pages,
                "reference_problem_profile": problem,
                "high_priority_clean_scan_match_candidate": high_priority,
                "training_authorized": False,
                "evaluation_authorized": False,
                "release_authorized": False,
            }
        )
    high_priority_cases = [
        case
        for case in cases
        if case["high_priority_clean_scan_match_candidate"] is True
    ]
    return {
        "format": 1,
        "created_at": utc_now_iso(),
        "role": ROLE,
        "boundary_contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
        "repository_path": str(repository.resolve()),
        "repository_revision": EXPECTED_REVISION,
        "license": "CC-BY-4.0",
        "license_sha256": EXPECTED_LICENSE_SHA256,
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_authorized": False,
        "authorization_blockers": [
            "absence_of_problem_comments_does_not_prove_semantic_completeness",
            "matching_physical_scan_not_discovered_or_aligned",
            "scan_asset_rights_not_verified",
            "independent_double_annotation_not_started",
            "production_holdout_split_not_assigned",
        ],
        "reference_count": len(cases),
        "boundary_accepted_count": sum(
            case["boundary"]["accepted"] is True for case in cases
        ),
        "zero_problem_reference_count": sum(
            int(case["reference_problem_profile"]["problem_record_count"])
            == 0
            for case in cases
        ),
        "high_priority_clean_scan_match_candidate_count": len(
            high_priority_cases
        ),
        "high_priority_candidate_encoded_page_count": sum(
            int(
                case["reference_page_profile"][
                    "encoded_music_page_count"
                ]
            )
            for case in high_priority_cases
        ),
        "high_priority_candidates": high_priority_cases,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    report = catalog(args.repository.resolve())
    atomic_write_json(args.output_path.resolve(), report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "reference_count",
                    "boundary_accepted_count",
                    "zero_problem_reference_count",
                    "high_priority_clean_scan_match_candidate_count",
                    "high_priority_candidate_encoded_page_count",
                    "training_authorized",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
