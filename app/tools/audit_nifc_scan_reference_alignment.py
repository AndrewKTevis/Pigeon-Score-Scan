from __future__ import annotations

"""Render and compare pinned NIFC scan/reference candidate pages.

The resulting similarity evidence is diagnostic.  Even a perfect page-order
match does not prove that every supported symbol was encoded, so this tool
never authorizes training, evaluation or release use.
"""

import argparse
import io
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.tools.acquire_nifc_chopin_matched_scans import (  # noqa: E402
    REPORT_ROLE as ACQUISITION_ROLE,
)
from scorescan.util import atomic_write_bytes, atomic_write_json, sha256_file, utc_now_iso  # noqa: E402


REPORT_ROLE = "nifc_scan_reference_page_alignment_diagnostic_not_authorized"
MEASURE_PATTERN = re.compile(r"^=+(\d+)")


def reference_page_measure_ranges(
    lines: list[str],
) -> list[dict[str, object]]:
    pages: list[list[int]] = [[]]
    for line in lines:
        if line == "!!LO:PB:g=original":
            pages.append([])
            continue
        if not line.startswith("="):
            continue
        first = line.split("\t", 1)[0]
        match = MEASURE_PATTERN.match(first)
        if match is None:
            continue
        measure = int(match.group(1))
        if measure not in pages[-1]:
            pages[-1].append(measure)
    if pages and not pages[-1]:
        pages.pop()
    return [
        {
            "page_index": index,
            "first_measure": min(measures),
            "last_measure": max(measures),
            "measure_count": len(measures),
        }
        for index, measures in enumerate(pages, start=1)
        if measures
    ]


def _render_reference_pages(
    reference_path: Path,
    output_dir: Path,
    *,
    expected_pages: int,
) -> tuple[list[Path], list[int], list[dict[str, object]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_paths = [
        output_dir / f"page-{page_index:03d}.png"
        for page_index in range(1, expected_pages + 1)
    ]
    expected_svg_paths = [
        output_dir / f"page-{page_index:03d}.svg"
        for page_index in range(1, expected_pages + 1)
    ]
    attempts: list[dict[str, object]] = []

    def run_renderer(page: int | None) -> int:
        command = [
            sys.executable,
            "-m",
            "app.tools.render_nifc_reference_pages",
            str(reference_path),
            str(output_dir),
            "--expected-pages",
            str(expected_pages),
        ]
        if page is not None:
            command.extend(("--page", str(page)))
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            timeout=180,
        )
        attempts.append(
            {
                "page": page if page is not None else "all",
                "return_code": completed.returncode,
                "stdout_tail": completed.stdout.decode(
                    "utf-8", errors="replace"
                )[-2000:],
                "stderr_tail": completed.stderr.decode(
                    "utf-8", errors="replace"
                )[-4000:],
            }
        )
        return completed.returncode

    missing = [
        index
        for index, (path, svg_path) in enumerate(
            zip(expected_paths, expected_svg_paths, strict=True),
            start=1,
        )
        if not path.is_file() or not svg_path.is_file()
    ]
    if missing:
        run_renderer(None)
    missing = [
        index
        for index, (path, svg_path) in enumerate(
            zip(expected_paths, expected_svg_paths, strict=True),
            start=1,
        )
        if not path.is_file() or not svg_path.is_file()
    ]
    for page_index in missing:
        run_renderer(page_index)
    failed = [
        index
        for index, (path, svg_path) in enumerate(
            zip(expected_paths, expected_svg_paths, strict=True),
            start=1,
        )
        if not path.is_file() or not svg_path.is_file()
    ]
    available = [
        path
        for path, svg_path in zip(
            expected_paths,
            expected_svg_paths,
            strict=True,
        )
        if path.is_file() and svg_path.is_file()
    ]
    return available, failed, attempts


def _load_gray(path: Path) -> np.ndarray:
    image = cv2.imdecode(
        np.frombuffer(path.read_bytes(), dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )
    if image is None or min(image.shape[:2]) < 500:
        raise ValueError(f"invalid alignment image: {path}")
    return image


def _paper_crop(gray: np.ndarray) -> np.ndarray:
    height, width = gray.shape
    light = gray >= max(110, int(np.percentile(gray, 35)))
    row_coverage = light.mean(axis=1)
    column_coverage = light.mean(axis=0)
    rows = np.flatnonzero(row_coverage >= 0.55)
    columns = np.flatnonzero(column_coverage >= 0.55)
    if rows.size and columns.size:
        top, bottom = int(rows[0]), int(rows[-1]) + 1
        left, right = int(columns[0]), int(columns[-1]) + 1
        if (
            bottom - top >= height * 0.65
            and right - left >= width * 0.65
        ):
            return gray[top:bottom, left:right]
    return gray


def _normalized_page(gray: np.ndarray) -> np.ndarray:
    cropped = _paper_crop(gray)
    resized = cv2.resize(
        cropped,
        (900, 1200),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
        resized
    )


def _cluster_rows(rows: np.ndarray) -> list[int]:
    if not rows.size:
        return []
    clusters: list[list[int]] = [[int(rows[0])]]
    for raw in rows[1:]:
        value = int(raw)
        if value <= clusters[-1][-1] + 2:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [int(round(sum(cluster) / len(cluster))) for cluster in clusters]


def staff_line_centers(gray: np.ndarray) -> list[int]:
    normalized = _normalized_page(gray)
    dark = cv2.threshold(
        normalized,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )[1]
    horizontal = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (90, 1)),
    )
    strengths = np.count_nonzero(horizontal, axis=1)
    rows = np.flatnonzero(strengths >= 90)
    return _cluster_rows(rows)


def estimated_staff_count(gray: np.ndarray) -> int:
    centers = staff_line_centers(gray)
    if len(centers) < 5:
        return 0
    staves = 0
    index = 0
    while index + 4 < len(centers):
        window = centers[index : index + 5]
        gaps = np.diff(window)
        if (
            gaps.min() >= 2
            and gaps.max() <= 20
            and float(gaps.max() / max(1, gaps.min())) <= 2.25
        ):
            staves += 1
            index += 5
        else:
            index += 1
    return staves


def _profile(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized = _normalized_page(gray)
    ink = (255.0 - normalized.astype(np.float32)) / 255.0
    return ink.mean(axis=1), ink.mean(axis=0)


def _maximum_shifted_correlation(
    first: np.ndarray,
    second: np.ndarray,
    *,
    maximum_shift: int,
) -> float:
    first = (first - first.mean()) / max(float(first.std()), 1e-6)
    second = (second - second.mean()) / max(float(second.std()), 1e-6)
    scores: list[float] = []
    for shift in range(-maximum_shift, maximum_shift + 1):
        if shift < 0:
            left, right = first[-shift:], second[:shift]
        elif shift > 0:
            left, right = first[:-shift], second[shift:]
        else:
            left, right = first, second
        if left.size >= first.size * 0.8:
            scores.append(float(np.mean(left * right)))
    return max(scores, default=-1.0)


def _sift_descriptors(gray: np.ndarray) -> tuple[list[Any], np.ndarray | None]:
    normalized = _normalized_page(gray)
    detector = cv2.SIFT_create(nfeatures=1500, contrastThreshold=0.02)
    return detector.detectAndCompute(normalized, None)


def page_similarity(
    scan: np.ndarray,
    reference: np.ndarray,
    *,
    scan_descriptors: tuple[list[Any], np.ndarray | None] | None = None,
    reference_descriptors: tuple[list[Any], np.ndarray | None] | None = None,
) -> dict[str, float]:
    scan_rows, scan_columns = _profile(scan)
    reference_rows, reference_columns = _profile(reference)
    row_correlation = _maximum_shifted_correlation(
        scan_rows,
        reference_rows,
        maximum_shift=80,
    )
    column_correlation = _maximum_shifted_correlation(
        scan_columns,
        reference_columns,
        maximum_shift=60,
    )
    scan_keypoints, scan_values = (
        scan_descriptors or _sift_descriptors(scan)
    )
    reference_keypoints, reference_values = (
        reference_descriptors or _sift_descriptors(reference)
    )
    good = 0
    if (
        scan_values is not None
        and reference_values is not None
        and len(scan_values) >= 2
        and len(reference_values) >= 2
    ):
        matcher = cv2.FlannBasedMatcher(
            {"algorithm": 1, "trees": 5},
            {"checks": 64},
        )
        pairs = matcher.knnMatch(scan_values, reference_values, k=2)
        good = sum(
            len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance
            for pair in pairs
        )
    sift_score = good / max(
        1.0,
        float(np.sqrt(len(scan_keypoints) * len(reference_keypoints))),
    )
    combined = (
        max(-1.0, row_correlation) * 0.30
        + max(-1.0, column_correlation) * 0.15
        + sift_score * 0.55
    )
    return {
        "combined": combined,
        "row_correlation": row_correlation,
        "column_correlation": column_correlation,
        "sift_score": sift_score,
        "sift_good_matches": float(good),
    }


def _make_contact_sheet(
    work_id: str,
    scan_paths: list[Path],
    reference_paths: list[Path],
    output_path: Path,
) -> None:
    width = 1200
    pair_height = 450
    sheet = Image.new(
        "RGB",
        (width, pair_height * len(scan_paths)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, (scan_path, reference_path) in enumerate(
        zip(scan_paths, reference_paths, strict=True),
        start=1,
    ):
        top = (index - 1) * pair_height
        for column, path in enumerate((scan_path, reference_path)):
            with Image.open(path) as source:
                page = source.convert("RGB")
                page.thumbnail((540, pair_height - 45))
                left = column * 600 + (600 - page.width) // 2
                sheet.paste(page, (left, top + 35))
        draw.text(
            (10, top + 8),
            f"{work_id} page {index:02d}: scan | reference",
            fill="black",
        )
    destination = io.BytesIO()
    sheet.save(destination, format="PNG")
    atomic_write_bytes(output_path, destination.getvalue())


def audit(
    acquisition_report_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    acquisition = json.loads(
        acquisition_report_path.read_text(encoding="utf-8")
    )
    if acquisition.get("role") != ACQUISITION_ROLE:
        raise ValueError("unexpected NIFC acquisition report role")
    for field in (
        "training_authorized",
        "evaluation_authorized",
        "release_authorized",
    ):
        if acquisition.get(field) is not False:
            raise ValueError(f"acquisition report unexpectedly sets {field}")
    works = acquisition.get("works")
    if not isinstance(works, list) or not works:
        raise ValueError("acquisition report has no works")
    output_dir.mkdir(parents=True, exist_ok=True)
    work_reports: list[dict[str, object]] = []
    for raw_work in works:
        if not isinstance(raw_work, dict):
            raise ValueError("invalid acquisition work")
        work: dict[str, Any] = raw_work
        work_id = str(work["id"])
        scan_paths = [Path(page["path"]) for page in work["pages"]]
        if any(
            not path.is_file()
            or sha256_file(path) != page["sha256"]
            or page["auto_rotation_applied"] is not False
            or page["auto_deskew_applied"] is not False
            or page["resampling_applied"] is not False
            for path, page in zip(scan_paths, work["pages"], strict=True)
        ):
            raise ValueError(f"scan page integrity failed: {work_id}")
        expected_pages = int(work["music_page_count"])
        reference_path = Path(work["reference_path"])
        if (
            not reference_path.is_file()
            or sha256_file(reference_path) != work["reference_sha256"]
        ):
            raise ValueError(f"reference integrity failed: {work_id}")
        render_dir = output_dir / "rendered" / work_id
        reference_paths, render_failures, render_attempts = (
            _render_reference_pages(
            reference_path,
            render_dir,
            expected_pages=expected_pages,
            )
        )
        measures = reference_page_measure_ranges(
            reference_path.read_text(
                encoding="utf-8-sig"
            ).splitlines()
        )
        problem_profile = work.get("reference_problem_profile", {})
        if render_failures:
            available_indices = [
                int(path.stem.rsplit("-", 1)[-1])
                for path in reference_paths
            ]
            contact_sheet = (
                output_dir / f"{work_id}-partial-contact-sheet.png"
            )
            if reference_paths:
                _make_contact_sheet(
                    work_id,
                    [
                        scan_paths[index - 1]
                        for index in available_indices
                    ],
                    reference_paths,
                    contact_sheet,
                )
            work_reports.append(
                {
                    "id": work_id,
                    "page_count": expected_pages,
                    "rendered_page_indices": available_indices,
                    "render_failure_pages": render_failures,
                    "render_attempts": render_attempts,
                    "reference_page_measure_ranges": measures,
                    "reference_problem_profile": problem_profile,
                    "contact_sheet_path": (
                        str(contact_sheet.resolve())
                        if contact_sheet.is_file()
                        else ""
                    ),
                    "contact_sheet_sha256": (
                        sha256_file(contact_sheet)
                        if contact_sheet.is_file()
                        else ""
                    ),
                    "automatic_alignment_candidate": False,
                    "page_alignment_verified": False,
                    "semantic_completeness_verified": False,
                    "training_authorized": False,
                    "evaluation_authorized": False,
                    "release_authorized": False,
                }
            )
            continue
        scan_images = [_load_gray(path) for path in scan_paths]
        reference_images = [_load_gray(path) for path in reference_paths]
        scan_descriptors = [
            _sift_descriptors(image) for image in scan_images
        ]
        reference_descriptors = [
            _sift_descriptors(image) for image in reference_images
        ]
        matrix: list[list[float]] = []
        details: list[list[dict[str, float]]] = []
        for scan_index, scan in enumerate(scan_images):
            row: list[float] = []
            detail_row: list[dict[str, float]] = []
            for reference_index, reference in enumerate(reference_images):
                result = page_similarity(
                    scan,
                    reference,
                    scan_descriptors=scan_descriptors[scan_index],
                    reference_descriptors=reference_descriptors[
                        reference_index
                    ],
                )
                row.append(result["combined"])
                detail_row.append(result)
            matrix.append(row)
            details.append(detail_row)
        values = np.asarray(matrix, dtype=np.float64)
        scan_best = (np.argmax(values, axis=1) + 1).tolist()
        reference_best = (np.argmax(values, axis=0) + 1).tolist()
        expected = list(range(1, expected_pages + 1))
        diagonal = np.diag(values)
        margins: list[float] = []
        for index in range(expected_pages):
            alternatives = np.delete(values[index], index)
            margins.append(
                float(diagonal[index] - np.max(alternatives))
                if alternatives.size
                else float(diagonal[index])
            )
        scan_staff_counts = [
            estimated_staff_count(image) for image in scan_images
        ]
        reference_staff_counts = [
            estimated_staff_count(image) for image in reference_images
        ]
        contact_sheet = output_dir / f"{work_id}-contact-sheet.png"
        _make_contact_sheet(
            work_id,
            scan_paths,
            reference_paths,
            contact_sheet,
        )
        automatic_candidate = (
            scan_best == expected
            and reference_best == expected
            and len(measures) == expected_pages
            and all(value > 0 for value in margins)
        )
        work_reports.append(
            {
                "id": work_id,
                "page_count": expected_pages,
                "scan_best_reference_pages": scan_best,
                "reference_best_scan_pages": reference_best,
                "identity_mapping_is_bidirectional_best": (
                    scan_best == expected and reference_best == expected
                ),
                "diagonal_combined_similarity": diagonal.tolist(),
                "diagonal_similarity_minimum": float(diagonal.min()),
                "diagonal_similarity_mean": float(diagonal.mean()),
                "diagonal_margin_over_next_best": margins,
                "minimum_diagonal_margin": min(margins),
                "scan_estimated_staff_counts": scan_staff_counts,
                "reference_estimated_staff_counts": reference_staff_counts,
                "staff_count_sequences_match": (
                    scan_staff_counts == reference_staff_counts
                ),
                "reference_page_measure_ranges": measures,
                "reference_problem_profile": problem_profile,
                "render_failure_pages": [],
                "render_attempts": render_attempts,
                "similarity_matrix": matrix,
                "similarity_details": details,
                "contact_sheet_path": str(contact_sheet.resolve()),
                "contact_sheet_sha256": sha256_file(contact_sheet),
                "automatic_alignment_candidate": automatic_candidate,
                "page_alignment_verified": False,
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
        "acquisition_report_path": str(acquisition_report_path.resolve()),
        "acquisition_report_sha256": sha256_file(acquisition_report_path),
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_authorized": False,
        "authorization_blockers": [
            "automatic_page_similarity_is_not_semantic_ground_truth",
            "references_contain_explicit_problem_annotations",
            "one_or_more_reference_pages_may_fail_native_rendering",
            "manual_all_page_alignment_review_not_recorded",
            "supported_symbol_family_completeness_not_audited",
            "independent_double_annotation_not_started",
            "production_holdout_split_not_assigned",
        ],
        "work_count": len(work_reports),
        "page_count": sum(int(work["page_count"]) for work in work_reports),
        "identity_mapping_candidate_work_count": sum(
            work["automatic_alignment_candidate"] is True
            for work in work_reports
        ),
        "works": work_reports,
    }
    atomic_write_json(output_dir / "alignment_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("acquisition_report_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    report = audit(
        args.acquisition_report_path.resolve(),
        args.output_dir.resolve(),
    )
    print(
        json.dumps(
            {
                "work_count": report["work_count"],
                "page_count": report["page_count"],
                "identity_mapping_candidate_work_count": report[
                    "identity_mapping_candidate_work_count"
                ],
                "training_authorized": report["training_authorized"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
