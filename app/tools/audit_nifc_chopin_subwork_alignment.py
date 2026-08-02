from __future__ import annotations

"""Locate encoded Chopin subworks inside full-edition NIFC scans.

The best contiguous page window is diagnostic only. Native renderer warnings,
page-similarity scores and zero explicit ``:problem:`` comments are all
insufficient to establish semantic ground truth.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2
import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_ROOT / "src"))

from app.tools.acquire_nifc_chopin_matched_scans import (  # noqa: E402
    reference_problem_profile,
)
from app.tools.audit_nifc_scan_reference_alignment import (  # noqa: E402
    _load_gray,
    _make_contact_sheet,
    _render_reference_pages,
    _sift_descriptors,
    page_similarity,
    reference_page_measure_ranges,
)
from app.tools.prepare_nifc_chopin_layout_audit_pages import (  # noqa: E402
    REPORT_ROLE as PREPARATION_REPORT_ROLE,
)
from scorescan.util import (  # noqa: E402
    atomic_write_json,
    sha256_file,
    utc_now_iso,
)


REPORT_ROLE = (
    "nifc_chopin_subwork_alignment_diagnostic_not_training_or_evaluation"
)


def renderer_warning_audit(
    reference_path: Path,
    output_dir: Path,
    *,
    expected_pages: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.tools.render_nifc_reference_pages",
            str(reference_path),
            str(output_dir),
            "--expected-pages",
            str(expected_pages),
            "--page",
            "1",
        ],
        check=False,
        capture_output=True,
        timeout=180,
        cwd=PROJECT_ROOT,
    )
    stderr = completed.stderr.decode("utf-8", errors="replace")
    warning_lines = [
        line.strip()
        for line in stderr.splitlines()
        if "[Warning]" in line
    ]
    return {
        "return_code": completed.returncode,
        "warning_count": len(warning_lines),
        "warning_lines": warning_lines,
        "stderr_tail": stderr[-4000:],
    }


def _best_contiguous_window(
    matrix: np.ndarray,
) -> dict[str, object]:
    scan_count, reference_count = matrix.shape
    if scan_count < reference_count or reference_count <= 0:
        raise ValueError("similarity matrix cannot contain a reference window")
    window_scores = [
        float(
            np.mean(
                [
                    matrix[start + offset, offset]
                    for offset in range(reference_count)
                ]
            )
        )
        for start in range(scan_count - reference_count + 1)
    ]
    ranked = sorted(
        range(len(window_scores)),
        key=lambda index: (-window_scores[index], index),
    )
    best_start = ranked[0]
    next_score = (
        window_scores[ranked[1]]
        if len(ranked) > 1
        else window_scores[best_start]
    )
    selected = matrix[
        best_start : best_start + reference_count,
        :,
    ]
    expected = list(range(reference_count))
    scan_best = np.argmax(selected, axis=1).tolist()
    reference_best_global = np.argmax(matrix, axis=0).tolist()
    expected_scan = [
        best_start + offset for offset in range(reference_count)
    ]
    diagonal = np.asarray(
        [
            matrix[best_start + offset, offset]
            for offset in range(reference_count)
        ],
        dtype=np.float64,
    )
    return {
        "best_start_zero_based": best_start,
        "scan_page_indices": [
            best_start + offset + 1
            for offset in range(reference_count)
        ],
        "window_scores": window_scores,
        "best_window_score": window_scores[best_start],
        "next_best_window_score": next_score,
        "best_window_margin": window_scores[best_start] - next_score,
        "selected_scan_best_reference_zero_based": scan_best,
        "reference_best_scan_global_zero_based": reference_best_global,
        "selected_mapping_is_bidirectional_best": (
            scan_best == expected
            and reference_best_global == expected_scan
        ),
        "selected_diagonal_similarity": diagonal.tolist(),
        "selected_diagonal_similarity_minimum": float(diagonal.min()),
        "selected_diagonal_similarity_mean": float(diagonal.mean()),
    }


def audit_subwork_alignment(
    preparation_report_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    preparation = json.loads(
        preparation_report_path.read_text(encoding="utf-8")
    )
    if preparation.get("role") != PREPARATION_REPORT_ROLE:
        raise ValueError("unexpected NIFC page-preparation report role")
    for field in (
        "training_authorized",
        "evaluation_authorized",
        "release_authorized",
    ):
        if preparation.get(field) is not False:
            raise ValueError(f"preparation report unexpectedly sets {field}")
    raw_candidates = preparation.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("preparation report has no candidates")
    output_dir.mkdir(parents=True, exist_ok=True)

    render_cache: dict[
        str,
        tuple[
            list[Path],
            list[int],
            list[dict[str, object]],
            list[dict[str, object]],
            dict[str, object],
        ],
    ] = {}
    reports: list[dict[str, object]] = []
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, dict):
            raise ValueError("invalid prepared candidate")
        candidate: dict[str, Any] = raw_candidate
        parent_pid = str(candidate["parent_pid"])
        reference_path = Path(str(candidate["reference_path"])).resolve()
        if (
            not reference_path.is_file()
            or sha256_file(reference_path)
            != candidate["reference_sha256"]
        ):
            raise ValueError(f"reference hash drifted: {parent_pid}")
        reference_key = str(reference_path)
        lines = reference_path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        measures = reference_page_measure_ranges(lines)
        expected_pages = len(measures)
        if expected_pages <= 0:
            raise ValueError(f"reference has no encoded pages: {parent_pid}")
        if reference_key not in render_cache:
            render_dir = (
                output_dir / "rendered" / reference_path.stem
            )
            paths, failures, attempts = _render_reference_pages(
                reference_path,
                render_dir,
                expected_pages=expected_pages,
            )
            warning_audit = renderer_warning_audit(
                reference_path,
                output_dir / "warning-audit" / reference_path.stem,
                expected_pages=expected_pages,
            )
            problem_profile = reference_problem_profile(lines)
            render_cache[reference_key] = (
                paths,
                failures,
                attempts,
                measures,
                {
                    "renderer_warning_audit": warning_audit,
                    "reference_problem_profile": problem_profile,
                },
            )
        (
            reference_paths,
            render_failures,
            render_attempts,
            measures,
            quality,
        ) = render_cache[reference_key]
        if render_failures:
            reports.append(
                {
                    "parent_pid": parent_pid,
                    "reference_path": str(reference_path),
                    "reference_page_count": expected_pages,
                    "render_failure_pages": render_failures,
                    "render_attempts": render_attempts,
                    **quality,
                    "automatic_contiguous_alignment_candidate": False,
                    "all_page_alignment_verified": False,
                    "semantic_completeness_verified": False,
                    "training_authorized": False,
                    "evaluation_authorized": False,
                    "release_authorized": False,
                }
            )
            continue
        raw_pages = candidate.get("pages")
        if not isinstance(raw_pages, list):
            raise ValueError("prepared candidate pages are missing")
        scan_paths = [Path(str(page["path"])).resolve() for page in raw_pages]
        if any(
            not path.is_file()
            or sha256_file(path) != page["sha256"]
            or page["auto_rotation_applied"] is not False
            or page["auto_deskew_applied"] is not False
            or page["resampling_applied"] is not False
            for path, page in zip(scan_paths, raw_pages, strict=True)
        ):
            raise ValueError(f"prepared scan integrity failed: {parent_pid}")
        scan_images = [_load_gray(path) for path in scan_paths]
        reference_images = [
            _load_gray(path) for path in reference_paths
        ]
        scan_descriptors = [
            _sift_descriptors(image) for image in scan_images
        ]
        reference_descriptors = [
            _sift_descriptors(image) for image in reference_images
        ]
        details: list[list[dict[str, float]]] = []
        for scan_index, scan in enumerate(scan_images):
            detail_row: list[dict[str, float]] = []
            for reference_index, reference in enumerate(reference_images):
                detail_row.append(
                    page_similarity(
                        scan,
                        reference,
                        scan_descriptors=scan_descriptors[scan_index],
                        reference_descriptors=reference_descriptors[
                            reference_index
                        ],
                    )
                )
            details.append(detail_row)
        matrix = np.asarray(
            [
                [detail["combined"] for detail in row]
                for row in details
            ],
            dtype=np.float64,
        )
        window = _best_contiguous_window(matrix)
        selected_indices = [
            int(value) for value in window["scan_page_indices"]
        ]
        contact_sheet = output_dir / (
            f"{parent_pid.replace(':', '-')}-{reference_path.stem}"
            "-best-window.png"
        )
        _make_contact_sheet(
            parent_pid,
            [scan_paths[index - 1] for index in selected_indices],
            reference_paths,
            contact_sheet,
        )
        warning_audit = quality["renderer_warning_audit"]
        automatic_candidate = (
            window["selected_mapping_is_bidirectional_best"] is True
            and float(window["best_window_margin"]) > 0
            and float(window["selected_diagonal_similarity_minimum"]) > 0
            and warning_audit["return_code"] == 0
            and warning_audit["warning_count"] == 0
            and quality["reference_problem_profile"][
                "problem_record_count"
            ]
            == 0
        )
        reports.append(
            {
                "parent_pid": parent_pid,
                "reference_path": str(reference_path),
                "reference_sha256": sha256_file(reference_path),
                "reference_page_count": expected_pages,
                "scan_page_count": len(scan_paths),
                "reference_page_measure_ranges": measures,
                "render_failure_pages": [],
                "render_attempts": render_attempts,
                **quality,
                **window,
                "similarity_matrix": matrix.tolist(),
                "similarity_details": details,
                "contact_sheet_path": str(contact_sheet.resolve()),
                "contact_sheet_sha256": sha256_file(contact_sheet),
                "automatic_contiguous_alignment_candidate": automatic_candidate,
                "all_page_alignment_verified": False,
                "semantic_completeness_verified": False,
                "training_authorized": False,
                "evaluation_authorized": False,
                "release_authorized": False,
            }
        )
    report = {
        "format": 1,
        "created_at": utc_now_iso(),
        "role": REPORT_ROLE,
        "preparation_report_path": str(preparation_report_path.resolve()),
        "preparation_report_sha256": sha256_file(preparation_report_path),
        "candidate_count": len(reports),
        "unique_reference_count": len(render_cache),
        "automatic_contiguous_alignment_candidate_count": sum(
            item["automatic_contiguous_alignment_candidate"] is True
            for item in reports
        ),
        "renderer_warning_reference_count": sum(
            cached[4]["renderer_warning_audit"]["warning_count"] > 0
            for cached in render_cache.values()
        ),
        "page_alignment_manually_verified_count": 0,
        "semantic_completeness_verified_count": 0,
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_authorized": False,
        "authorization_blockers": [
            "automatic_similarity_does_not_prove_page_identity",
            "renderer_warnings_disqualify_affected_references",
            "zero_problem_comments_do_not_prove_semantic_completeness",
            "manual_all_page_review_not_recorded",
            "independent_double_annotation_not_started",
        ],
        "candidates": reports,
    }
    atomic_write_json(output_dir / "subwork_alignment_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("preparation_report_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    report = audit_subwork_alignment(
        args.preparation_report_path.resolve(),
        args.output_dir.resolve(),
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "candidate_count",
                    "unique_reference_count",
                    "automatic_contiguous_alignment_candidate_count",
                    "renderer_warning_reference_count",
                    "page_alignment_manually_verified_count",
                    "semantic_completeness_verified_count",
                    "training_authorized",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
