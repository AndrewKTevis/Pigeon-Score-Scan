from __future__ import annotations

"""Validate and record the primary visual page-range review.

This is intentionally not independent double annotation and never authorizes
the selected pages for training, evaluation or release evidence.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_ROOT / "src"))

from app.tools.audit_nifc_scan_reference_alignment import (  # noqa: E402
    _make_contact_sheet,
    reference_page_measure_ranges,
)
from app.tools.audit_nifc_chopin_subwork_alignment import (  # noqa: E402
    REPORT_ROLE as AUTOMATIC_ALIGNMENT_ROLE,
)
from app.tools.prepare_nifc_chopin_layout_audit_pages import (  # noqa: E402
    REPORT_ROLE as PREPARATION_ROLE,
)
from scorescan.util import (  # noqa: E402
    atomic_write_json,
    sha256_file,
    utc_now_iso,
)


MANIFEST_ROLE = (
    "nifc_chopin_primary_visual_page_range_review_not_training_or_evaluation"
)
REPORT_ROLE = (
    "recorded_nifc_chopin_primary_visual_page_range_review_not_training_or_evaluation"
)


def _load_pinned_report(
    manifest: dict[str, object],
    *,
    path_field: str,
    hash_field: str,
    expected_role: str,
) -> tuple[Path, dict[str, object]]:
    raw_path = manifest.get(path_field)
    expected_hash = manifest.get(hash_field)
    if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
        raise ValueError(f"primary review lacks {path_field}")
    path = (PROJECT_ROOT / raw_path).resolve()
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise ValueError(f"primary review pinned report drifted: {path_field}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("role") != expected_role:
        raise ValueError(f"unexpected report role: {path_field}")
    for field in (
        "training_authorized",
        "evaluation_authorized",
        "release_authorized",
    ):
        if report.get(field) is not False:
            raise ValueError(f"pinned report unexpectedly sets {field}")
    return path, report


def record_primary_review(
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unexpected primary visual review schema")
    if manifest.get("role") != MANIFEST_ROLE:
        raise ValueError("unexpected primary visual review role")
    for field in (
        "training_authorized",
        "evaluation_authorized",
        "release_authorized",
    ):
        if manifest.get(field) is not False:
            raise ValueError(f"primary review unexpectedly sets {field}")
    if manifest.get("independent_second_review_complete") is not False:
        raise ValueError("primary review cannot claim independent review")
    if manifest.get("semantic_symbol_completeness_reviewed") is not False:
        raise ValueError("primary review cannot claim semantic completeness")
    reviewer = manifest.get("reviewer")
    if (
        not isinstance(reviewer, dict)
        or reviewer.get("kind")
        != "primary_agent_visual_inspection_not_independent_human_annotation"
        or reviewer.get("independent_of_candidate_discovery") is not False
    ):
        raise ValueError("primary visual reviewer identity is invalid")

    preparation_path, preparation = _load_pinned_report(
        manifest,
        path_field="preparation_report_path",
        hash_field="preparation_report_sha256",
        expected_role=PREPARATION_ROLE,
    )
    automatic_path, automatic = _load_pinned_report(
        manifest,
        path_field="automatic_alignment_report_path",
        hash_field="automatic_alignment_report_sha256",
        expected_role=AUTOMATIC_ALIGNMENT_ROLE,
    )
    if (
        automatic.get("preparation_report_sha256")
        != sha256_file(preparation_path)
    ):
        raise ValueError("automatic alignment is not bound to preparation")
    raw_prepared = preparation.get("candidates")
    raw_automatic = automatic.get("candidates")
    raw_mappings = manifest.get("mappings")
    if not all(
        isinstance(value, list)
        for value in (raw_prepared, raw_automatic, raw_mappings)
    ):
        raise ValueError("primary review inputs are incomplete")
    prepared_by_pid = {
        str(item["parent_pid"]): item
        for item in raw_prepared
        if isinstance(item, dict)
    }
    automatic_by_pid = {
        str(item["parent_pid"]): item
        for item in raw_automatic
        if isinstance(item, dict)
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    mappings: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw_mapping in raw_mappings:
        if not isinstance(raw_mapping, dict):
            raise ValueError("invalid primary review mapping")
        mapping: dict[str, Any] = raw_mapping
        pid = str(mapping.get("parent_pid", ""))
        if (
            not pid
            or pid in seen
            or pid not in prepared_by_pid
            or pid not in automatic_by_pid
        ):
            raise ValueError("primary review mapping PID is invalid")
        prepared = prepared_by_pid[pid]
        automatic_case = automatic_by_pid[pid]
        reference_path = Path(
            str(prepared["reference_path"])
        ).resolve()
        if reference_path.name != mapping.get("reference_path"):
            raise ValueError("primary review reference identity drifted")
        lines = reference_path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        reference_pages = reference_page_measure_ranges(lines)
        indices = mapping.get("selected_scan_page_indices")
        if (
            not isinstance(indices, list)
            or not indices
            or any(not isinstance(value, int) for value in indices)
            or indices != list(range(indices[0], indices[0] + len(indices)))
            or len(indices) != len(reference_pages)
        ):
            raise ValueError("primary review page range is not contiguous")
        raw_pages = prepared.get("pages")
        if (
            not isinstance(raw_pages, list)
            or indices[0] < 1
            or indices[-1] > len(raw_pages)
        ):
            raise ValueError("primary review page range is outside scan")
        reviewed_music = mapping.get("reviewed_music_page_indices")
        reviewed_non_music = mapping.get(
            "reviewed_non_music_page_indices"
        )
        if (
            not isinstance(reviewed_music, list)
            or not isinstance(reviewed_non_music, list)
            or any(
                not isinstance(value, int)
                for value in [*reviewed_music, *reviewed_non_music]
            )
            or set(reviewed_music) & set(reviewed_non_music)
            or sorted([*reviewed_music, *reviewed_non_music])
            != list(range(1, len(raw_pages) + 1))
            or not set(indices).issubset(set(reviewed_music))
        ):
            raise ValueError("primary review page-role partition is invalid")
        selected = [raw_pages[index - 1] for index in indices]
        if any(
            not Path(str(page["path"])).is_file()
            or sha256_file(Path(str(page["path"]))) != page["sha256"]
            for page in selected
        ):
            raise ValueError("primary review selected page hash drifted")
        warning_count = automatic_case[
            "renderer_warning_audit"
        ]["warning_count"]
        if warning_count != mapping.get("renderer_warning_count"):
            raise ValueError("primary review renderer warning count drifted")
        evidence = mapping.get("visual_evidence")
        if (
            not isinstance(evidence, list)
            or len(evidence) < 4
            or "selected_page_count_equals_encoded_reference_page_count"
            not in evidence
        ):
            raise ValueError("primary review boundary evidence is incomplete")
        reference_render_dir = (
            automatic_path.parent / "rendered" / reference_path.stem
        )
        reference_render_paths = [
            reference_render_dir / f"page-{index:03d}.png"
            for index in range(1, len(indices) + 1)
        ]
        if any(not path.is_file() for path in reference_render_paths):
            raise ValueError("primary review reference rendering is missing")
        contact_sheet = output_dir / (
            f"{pid.replace(':', '-')}-{reference_path.stem}"
            "-primary-review.png"
        )
        _make_contact_sheet(
            pid,
            [Path(str(page["path"])) for page in selected],
            reference_render_paths,
            contact_sheet,
        )
        automatic_indices = automatic_case.get("scan_page_indices", [])
        seen.add(pid)
        mappings.append(
            {
                "parent_pid": pid,
                "reference_path": str(reference_path),
                "reference_sha256": sha256_file(reference_path),
                "selected_scan_page_indices": indices,
                "reviewed_music_page_indices": reviewed_music,
                "reviewed_non_music_page_indices": reviewed_non_music,
                "selected_scan_page_sha256": [
                    page["sha256"] for page in selected
                ],
                "visual_evidence": evidence,
                "renderer_warning_count": warning_count,
                "renderer_warning_disqualified": warning_count > 0,
                "automatic_selected_scan_page_indices": automatic_indices,
                "automatic_mapping_agreed": automatic_indices == indices,
                "contact_sheet_path": str(contact_sheet.resolve()),
                "contact_sheet_sha256": sha256_file(contact_sheet),
                "primary_visual_range_reviewed": True,
                "independent_second_review_complete": False,
                "semantic_symbol_completeness_reviewed": False,
                "training_authorized": False,
                "evaluation_authorized": False,
                "release_authorized": False,
            }
        )
    if len(seen) != len(prepared_by_pid):
        raise ValueError("primary review does not cover every prepared candidate")
    report = {
        "format": 1,
        "created_at": utc_now_iso(),
        "role": REPORT_ROLE,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "preparation_report_path": str(preparation_path),
        "preparation_report_sha256": sha256_file(preparation_path),
        "automatic_alignment_report_path": str(automatic_path),
        "automatic_alignment_report_sha256": sha256_file(automatic_path),
        "reviewer": reviewer,
        "mapping_count": len(mappings),
        "primary_visual_range_reviewed_count": len(mappings),
        "primary_visual_page_role_reviewed_count": sum(
            len(mapping["reviewed_music_page_indices"])
            + len(mapping["reviewed_non_music_page_indices"])
            for mapping in mappings
        ),
        "automatic_mapping_agreement_count": sum(
            mapping["automatic_mapping_agreed"] is True
            for mapping in mappings
        ),
        "renderer_warning_disqualified_count": sum(
            mapping["renderer_warning_disqualified"] is True
            for mapping in mappings
        ),
        "eligible_for_independent_second_range_review_count": sum(
            mapping["renderer_warning_disqualified"] is False
            for mapping in mappings
        ),
        "independent_second_review_complete_count": 0,
        "semantic_symbol_completeness_reviewed_count": 0,
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_authorized": False,
        "authorization_blockers": [
            "primary_visual_review_is_not_independent_double_annotation",
            "one_reference_has_native_renderer_semantic_warnings",
            "symbol_by_symbol_semantic_completeness_not_reviewed",
            "production_holdout_split_not_assigned",
        ],
        "mappings": mappings,
    }
    atomic_write_json(output_dir / "primary_visual_review_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    report = record_primary_review(
        args.manifest_path.resolve(),
        args.output_dir.resolve(),
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "mapping_count",
                    "primary_visual_range_reviewed_count",
                    "automatic_mapping_agreement_count",
                    "renderer_warning_disqualified_count",
                    "eligible_for_independent_second_range_review_count",
                    "independent_second_review_complete_count",
                    "semantic_symbol_completeness_reviewed_count",
                    "training_authorized",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
