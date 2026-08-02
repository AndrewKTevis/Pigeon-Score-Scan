from __future__ import annotations

"""Prepare identity-linked OpenScore/IMSLP scans for split-safe development.

The historical scan and the OpenScore transcription describe the same work, but
their page and measure layouts are not assumed to match.  Consequently this tool
authorizes only whole-work semantic development evaluation.  It deliberately
does not create page labels, detector labels, release holdouts, or training
authorization.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Mapping

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_ROOT / "src"))

from scorescan.product_scope import (  # noqa: E402
    PRODUCTION_BOUNDARY_CONTRACT_VERSION,
)
from scorescan.util import atomic_write_json, sha256_file, utc_now_iso  # noqa: E402

from app.tools.acquire_openscore_imslp_scans import (  # noqa: E402
    ROLE as BYTE_MANIFEST_ROLE,
    _safe_score_path,
    inspect_pdf,
)
from app.tools.catalog_openscore_imslp_scan_candidates import (  # noqa: E402
    LICENSE_SHA256,
    REVISION,
)
from app.tools.prepare_muse_omr_benchmark import (  # noqa: E402
    _export_musicxml,
    analyze_reference_boundary,
)


ROLE = (
    "external_real_scan_split_inherited_semantic_development_"
    "not_release_holdout"
)
WORK_FINGERPRINT = re.compile(r"[0-9a-f]{64}")
SPLIT_ROLES = {
    "train": "train_split_error_mining_only_not_training_labels",
    "calibration": "calibration_split_semantic_development_only",
    "test": "test_split_semantic_development_not_independent_holdout",
}
BOUNDARY_SHAPES = {
    "monophonic_ensemble": "single_staff_ensemble",
}


def _asset_by_source(
    manifest: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    rows = manifest.get("assets")
    if not isinstance(rows, list) or not rows:
        raise ValueError("scan byte manifest has no assets")
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid scan asset row")
        source_id = str(row.get("imslp_source_id", ""))
        if not source_id.isdigit() or source_id in result:
            raise ValueError("invalid or duplicate IMSLP scan asset")
        if row.get("retrieval_channel") != (
            "internet_archive_exact_imslp_mirror"
        ):
            raise ValueError("scan asset did not use the exact archive mirror")
        result[source_id] = row
    return result


def _candidate_by_source(
    manifest: Mapping[str, object],
) -> dict[str, list[dict[str, object]]]:
    rows = manifest.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError("scan byte manifest has no candidates")
    result: dict[str, list[dict[str, object]]] = {}
    work_splits: dict[str, str] = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("source_identity_verified") is not True
        ):
            raise ValueError("candidate bypassed source identity verification")
        source_id = str(row.get("imslp_source_id", ""))
        split = str(row.get("semantic_split", ""))
        fingerprint = str(row.get("work_fingerprint", ""))
        if (
            not source_id.isdigit()
            or split not in SPLIT_ROLES
            or WORK_FINGERPRINT.fullmatch(fingerprint) is None
        ):
            raise ValueError("candidate has invalid split provenance")
        previous_split = work_splits.setdefault(fingerprint, split)
        if previous_split != split:
            raise ValueError("one work crosses inherited semantic splits")
        result.setdefault(source_id, []).append(row)
    return result


def _verified_pdf(
    asset: Mapping[str, object],
) -> tuple[Path, dict[str, object]]:
    path = Path(str(asset.get("pdf_path", ""))).resolve()
    expected_hash = str(asset.get("pdf_sha256", "")).casefold()
    if (
        WORK_FINGERPRINT.fullmatch(expected_hash) is None
        or not path.is_file()
        or sha256_file(path) != expected_hash
    ):
        raise ValueError("downloaded scan PDF hash mismatch")
    inspection = inspect_pdf(path)
    if inspection["actual_page_count"] != int(
        asset.get("actual_page_count", 0)
    ):
        raise ValueError("downloaded scan PDF page count changed")
    return path, inspection


def prepare_corpus(
    byte_manifest_path: Path,
    score_root: Path,
    output_dir: Path,
    musescore: Path,
    *,
    timeout_seconds: int = 180,
    force: bool = False,
    exporter: Callable[..., None] = _export_musicxml,
) -> dict[str, object]:
    manifest = json.loads(byte_manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("role") != BYTE_MANIFEST_ROLE
        or manifest.get("revision") != REVISION
        or manifest.get("license_sha256") != LICENSE_SHA256
        or manifest.get("archive_mirror_required") is not True
        or manifest.get("training_authorized") is not False
        or manifest.get("evaluation_authorized") is not False
        or manifest.get("release_authorized") is not False
    ):
        raise ValueError("unexpected scan byte manifest contract")
    license_path = score_root / "LICENSE.txt"
    if (
        not license_path.is_file()
        or sha256_file(license_path) != LICENSE_SHA256
    ):
        raise ValueError("fixed OpenScore license hash mismatch")
    if not musescore.is_file():
        raise FileNotFoundError("MuseScore executable is missing")

    assets = _asset_by_source(manifest)
    candidates = _candidate_by_source(manifest)
    if set(assets) != set(candidates):
        raise ValueError("scan assets and candidates have different sources")

    output_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = output_dir / "references"
    cases: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    pdf_splits: dict[str, str] = {}
    for position, source_id in enumerate(sorted(assets, key=int), start=1):
        source_candidates = candidates[source_id]
        asset = assets[source_id]
        if len(source_candidates) != 1:
            excluded.append(
                {
                    "imslp_source_id": source_id,
                    "reason": (
                        "one scan maps to multiple transcription candidates; "
                        "page/measure ranges are not proven"
                    ),
                    "candidate_count": len(source_candidates),
                    "input_pdf_pages": int(asset["actual_page_count"]),
                }
            )
            continue
        candidate = source_candidates[0]
        candidate_fingerprint = str(candidate["work_fingerprint"])
        asset_fingerprints = asset.get("candidate_work_fingerprints")
        if asset_fingerprints != [candidate_fingerprint]:
            raise ValueError("scan/candidate work fingerprints do not match")
        split = str(candidate["semantic_split"])
        pdf_hash = str(asset["pdf_sha256"])
        previous_pdf_split = pdf_splits.setdefault(pdf_hash, split)
        if previous_pdf_split != split:
            raise ValueError("one scan PDF crosses inherited semantic splits")

        pdf_path, inspection = _verified_pdf(asset)
        score_path = _safe_score_path(
            score_root,
            str(candidate.get("path", "")),
        )
        expected_score_hash = str(candidate.get("sha256", ""))
        if (
            not score_path.is_file()
            or sha256_file(score_path) != expected_score_hash
        ):
            raise ValueError("fixed OpenScore transcription hash mismatch")
        reference_path = (
            reference_dir / f"work_{candidate_fingerprint}.musicxml"
        )
        if force or not reference_path.is_file():
            exporter(
                score_path,
                reference_path,
                musescore,
                timeout_seconds=timeout_seconds,
            )
        boundary = analyze_reference_boundary(reference_path)
        expected_shape = BOUNDARY_SHAPES.get(
            str(candidate.get("boundary_configuration", ""))
        )
        boundary_identity_consistent = (
            expected_shape is not None
            and boundary.get("score_shape") == expected_shape
            and boundary.get("accepted") is True
        )
        print(
            f"[{position}/{len(assets)}] IMSLP{source_id} semantic reference",
            flush=True,
        )
        case = {
            "id": (
                f"openscore-imslp-{int(source_id):06d}-"
                f"{candidate_fingerprint[:12]}"
            ),
            "imslp_source_id": source_id,
            "work_fingerprint": candidate_fingerprint,
            "source_group_fingerprint": candidate.get(
                "source_group_fingerprint",
                "",
            ),
            "semantic_split": split,
            "split_role": SPLIT_ROLES[split],
            "role": "whole_work_semantic_development_only",
            "input_pdf": str(pdf_path),
            "input_pdf_sha256": pdf_hash,
            "input_pdf_pages": int(inspection["actual_page_count"]),
            "reference": str(reference_path.relative_to(output_dir)),
            "reference_sha256": sha256_file(reference_path),
            "source_mscx": str(candidate["path"]),
            "source_mscx_sha256": expected_score_hash,
            "source_page_title": asset.get("source_page_title", ""),
            "archive_identifier": asset.get("archive_identifier", ""),
            "retrieval_channel": asset.get("retrieval_channel", ""),
            "alignment_level": (
                "same_work_identity_only_no_page_or_measure_alignment"
            ),
            "page_training_labels_authorized": False,
            "whole_work_semantic_development_evaluation_authorized": (
                boundary_identity_consistent
            ),
            "boundary_identity_consistent": boundary_identity_consistent,
            "boundary": boundary,
        }
        if not boundary_identity_consistent:
            excluded.append(
                {
                    "imslp_source_id": source_id,
                    "reason": (
                        "reference is outside the declared product boundary "
                        "or disagrees with the source catalog shape"
                    ),
                    "candidate_count": 1,
                    "semantic_split": split,
                    "input_pdf_pages": int(inspection["actual_page_count"]),
                    "boundary": boundary,
                }
            )
            continue
        cases.append(case)

    split_counts = Counter(str(case["semantic_split"]) for case in cases)
    split_pages = Counter()
    for case in cases:
        split_pages[str(case["semantic_split"])] += int(
            case["input_pdf_pages"]
        )
    source_page_count = sum(
        int(asset["actual_page_count"]) for asset in assets.values()
    )
    excluded_page_count = sum(
        int(source["input_pdf_pages"]) for source in excluded
    )
    report = {
        "format": 1,
        "created_at": utc_now_iso(),
        "name": (
            "OpenScore/IMSLP exact-mirror real-scan semantic "
            "development corpus"
        ),
        "role": ROLE,
        "boundary_contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
        "byte_manifest_path": str(byte_manifest_path),
        "byte_manifest_sha256": sha256_file(byte_manifest_path),
        "score_root": str(score_root),
        "revision": REVISION,
        "license_sha256": LICENSE_SHA256,
        "training_authorized": False,
        "page_training_labels_authorized": False,
        "whole_work_semantic_development_evaluation_authorized": all(
            bool(case["whole_work_semantic_development_evaluation_authorized"])
            for case in cases
        )
        and bool(cases),
        "release_evaluation_authorized": False,
        "release_authorized": False,
        "independent_holdout": False,
        "authorization_reason": (
            "exact scan bytes and same-work transcriptions are verified and "
            "the immutable work-level split is inherited; historical scan "
            "pages/measures are not geometrically aligned, and these works "
            "already occur in the synthetic semantic development corpus"
        ),
        "alignment_level": (
            "same_work_identity_only_no_page_or_measure_alignment"
        ),
        # Keep acquisition coverage separate from accepted in-boundary cases.
        # A complete audit with zero accepted cases is still useful evidence:
        # it proves that the source was rejected instead of silently entering
        # training or being misreported as an empty acquisition.
        "boundary_audit_complete": len(cases) + len(excluded) == len(assets),
        "source_count": len(assets),
        "source_work_count": len(
            {
                str(candidate["work_fingerprint"])
                for source_candidates in candidates.values()
                for candidate in source_candidates
            }
        ),
        "source_page_count": source_page_count,
        "case_count": len(cases),
        "work_count": len(
            {str(case["work_fingerprint"]) for case in cases}
        ),
        "page_count": sum(int(case["input_pdf_pages"]) for case in cases),
        "case_count_by_semantic_split": dict(sorted(split_counts.items())),
        "page_count_by_semantic_split": dict(sorted(split_pages.items())),
        "excluded_source_count": len(excluded),
        "excluded_page_count": excluded_page_count,
        "excluded_sources": excluded,
        "cases": cases,
    }
    atomic_write_json(output_dir / "semantic_manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("byte_manifest_path", type=Path)
    parser.add_argument("score_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--musescore", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 30 <= args.timeout_seconds <= 1800:
        raise ValueError("timeout-seconds must be between 30 and 1800")
    report = prepare_corpus(
        args.byte_manifest_path.resolve(),
        args.score_root.resolve(),
        args.output_dir.resolve(),
        args.musescore.resolve(),
        timeout_seconds=args.timeout_seconds,
        force=args.force,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "case_count",
                    "work_count",
                    "page_count",
                    "case_count_by_semantic_split",
                    "excluded_source_count",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
