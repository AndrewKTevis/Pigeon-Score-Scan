from __future__ import annotations

"""Prepare uncropped or fixed-midpoint NIFC pages for layout auditing.

The only permitted geometric operation is an exact midpoint crop of an
obvious landscape book spread. No rotation, deskew, perspective correction,
resampling or contrast processing is applied to the persisted page images.
"""

import argparse
import io
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_ROOT / "src"))

from app.tools.acquire_nifc_chopin_layout_audit_shortlist import (  # noqa: E402
    REPORT_ROLE as ACQUISITION_REPORT_ROLE,
)
from app.tools.acquire_nifc_chopin_matched_scans import (  # noqa: E402
    _split_spread_pages,
    inspect_image,
)
from app.tools.audit_nifc_scan_reference_alignment import (  # noqa: E402
    estimated_staff_count,
)
from scorescan.util import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_json,
    sha256_file,
    utc_now_iso,
)


REPORT_ROLE = (
    "prepared_nifc_chopin_layout_audit_pages_not_training_or_evaluation"
)
LANDSCAPE_SPREAD_ASPECT_FLOOR = 1.20


def classify_scan_layout(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        raise ValueError("invalid source image dimensions")
    if width / height >= LANDSCAPE_SPREAD_ASPECT_FLOOR:
        return "left_to_right_two_page_spread_fixed_midpoint"
    return "single_page_no_geometric_transform"


def _staff_count(payload: bytes) -> int:
    image = cv2.imdecode(
        np.frombuffer(payload, dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )
    if image is None:
        raise ValueError("prepared page cannot be decoded")
    return estimated_staff_count(image)


def _make_contact_sheet(
    parent_pid: str,
    pages: list[dict[str, object]],
    output_path: Path,
) -> None:
    columns = 4
    cell_width = 300
    cell_height = 430
    rows = (len(pages) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * cell_width, rows * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for offset, page_record in enumerate(pages):
        row, column = divmod(offset, columns)
        left = column * cell_width
        top = row * cell_height
        with Image.open(Path(str(page_record["path"]))) as source:
            page = source.convert("RGB")
            page.thumbnail((cell_width - 20, cell_height - 45))
            sheet.paste(
                page,
                (
                    left + (cell_width - page.width) // 2,
                    top + 35,
                ),
            )
        draw.text(
            (left + 8, top + 8),
            (
                f"{parent_pid} page {offset + 1:02d} "
                f"staff={page_record['estimated_five_line_staff_count']}"
            ),
            fill="black",
        )
    destination = io.BytesIO()
    sheet.save(destination, format="PNG")
    atomic_write_bytes(output_path, destination.getvalue())


def prepare_pages(
    acquisition_report_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    acquisition = json.loads(
        acquisition_report_path.read_text(encoding="utf-8")
    )
    if acquisition.get("role") != ACQUISITION_REPORT_ROLE:
        raise ValueError("unexpected NIFC acquisition report role")
    for field in (
        "training_authorized",
        "evaluation_authorized",
        "release_authorized",
    ):
        if acquisition.get(field) is not False:
            raise ValueError(f"acquisition report unexpectedly sets {field}")
    raw_candidates = acquisition.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("acquisition report has no candidates")
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[dict[str, object]] = []
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            raise ValueError("invalid acquired candidate")
        parent_pid = str(candidate["parent_pid"])
        page_dir = output_dir / "pages" / parent_pid.replace(":", "-")
        page_dir.mkdir(parents=True, exist_ok=True)
        pages: list[dict[str, object]] = []
        raw_children = candidate.get("children")
        if not isinstance(raw_children, list) or not raw_children:
            raise ValueError(f"acquired candidate has no children: {parent_pid}")
        layouts: set[str] = set()
        for child in raw_children:
            if not isinstance(child, dict):
                raise ValueError("invalid acquired child")
            source_path = Path(str(child["image_path"])).resolve()
            if (
                not source_path.is_file()
                or sha256_file(source_path) != child.get("image_sha256")
            ):
                raise ValueError("source child image hash drifted")
            payload = source_path.read_bytes()
            image = child.get("image")
            if not isinstance(image, dict):
                raise ValueError("source child image profile is missing")
            layout = classify_scan_layout(
                int(image["width"]),
                int(image["height"]),
            )
            layouts.add(layout)
            if layout.startswith("left_to_right"):
                derived = _split_spread_pages(payload)
                suffix = ".png"
            else:
                width = int(image["width"])
                height = int(image["height"])
                derived = [(payload, (0, 0, width, height))]
                suffix = ".jpg"
            for page_payload, crop in derived:
                page_index = len(pages) + 1
                page_path = page_dir / f"page-{page_index:03d}{suffix}"
                atomic_write_bytes(page_path, page_payload)
                staff_count = _staff_count(page_payload)
                pages.append(
                    {
                        "sequence_index": page_index,
                        "source_child_sequence_index": child[
                            "sequence_index"
                        ],
                        "source_child_pid": child["pid"],
                        "source_image_sha256": child["image_sha256"],
                        "source_crop_xyxy": list(crop),
                        "geometric_transform": (
                            "fixed_midpoint_crop_only"
                            if layout.startswith("left_to_right")
                            else "none"
                        ),
                        "auto_rotation_applied": False,
                        "auto_deskew_applied": False,
                        "perspective_correction_applied": False,
                        "resampling_applied": False,
                        "path": str(page_path.resolve()),
                        "sha256": sha256_file(page_path),
                        "image": inspect_image(page_payload),
                        "estimated_five_line_staff_count": staff_count,
                        "automated_page_role": (
                            "music_candidate"
                            if staff_count > 0
                            else "non_music_candidate"
                        ),
                        "page_role_manually_verified": False,
                        "reference_page_alignment_status": "not_verified",
                    }
                )
        contact_sheet = output_dir / (
            f"{parent_pid.replace(':', '-')}-all-pages-contact-sheet.png"
        )
        _make_contact_sheet(parent_pid, pages, contact_sheet)
        candidates.append(
            {
                "parent_pid": parent_pid,
                "reference_path": candidate["reference_path"],
                "reference_sha256": candidate["reference_sha256"],
                "source_layouts": sorted(layouts),
                "source_child_count": len(raw_children),
                "derived_page_count": len(pages),
                "automated_music_candidate_count": sum(
                    page["automated_page_role"] == "music_candidate"
                    for page in pages
                ),
                "automated_non_music_candidate_count": sum(
                    page["automated_page_role"] == "non_music_candidate"
                    for page in pages
                ),
                "contact_sheet_role": (
                    "resampled_audit_visualization_never_training_input"
                ),
                "contact_sheet_path": str(contact_sheet.resolve()),
                "contact_sheet_sha256": sha256_file(contact_sheet),
                "pages": pages,
                "page_roles_manually_verified": False,
                "all_reference_pages_aligned": False,
                "training_authorized": False,
                "evaluation_authorized": False,
                "release_authorized": False,
            }
        )
    report = {
        "format": 1,
        "created_at": utc_now_iso(),
        "role": REPORT_ROLE,
        "acquisition_report_path": str(
            acquisition_report_path.resolve()
        ),
        "acquisition_report_sha256": sha256_file(
            acquisition_report_path
        ),
        "output_dir": str(output_dir.resolve()),
        "candidate_count": len(candidates),
        "derived_page_count": sum(
            int(candidate["derived_page_count"]) for candidate in candidates
        ),
        "automated_music_candidate_count": sum(
            int(candidate["automated_music_candidate_count"])
            for candidate in candidates
        ),
        "automated_non_music_candidate_count": sum(
            int(candidate["automated_non_music_candidate_count"])
            for candidate in candidates
        ),
        "allowed_geometric_operations": [
            "none",
            "fixed_midpoint_crop_only",
        ],
        "page_roles_manually_verified": False,
        "all_page_alignment_verified": False,
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_authorized": False,
        "authorization_blockers": [
            "automated_staff_presence_is_not_manual_page_role_verification",
            "reference_subwork_page_range_not_identified",
            "all_page_alignment_not_verified",
            "independent_double_annotation_not_started",
        ],
        "candidates": candidates,
    }
    atomic_write_json(output_dir / "page_preparation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("acquisition_report_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    report = prepare_pages(
        args.acquisition_report_path.resolve(),
        args.output_dir.resolve(),
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "candidate_count",
                    "derived_page_count",
                    "automated_music_candidate_count",
                    "automated_non_music_candidate_count",
                    "all_page_alignment_verified",
                    "training_authorized",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
