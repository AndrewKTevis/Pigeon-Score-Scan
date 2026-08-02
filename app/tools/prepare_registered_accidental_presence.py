from __future__ import annotations

"""Prepare scan-backed accidental-presence descriptors from registered pages.

This tool is deliberately narrower than general OMR training.  Exact MuseScore
SVG notehead/accidental anchors are projected onto already registered scan
pages, then passed through the same 256x96 measure evidence transform and HOG
descriptor deployed by ``AccidentalPresenceGuard``.

Training and independent-holdout preparation use explicit, mutually exclusive
contracts.  Holdout output is evaluation-only and the training loader rejects
it even if its directory is supplied accidentally.
"""

import argparse
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.tools.prepare_openscore_svg_regions import (  # noqa: E402
    Box,
    sha256_file,
    svg_class_objects,
)
from scorescan.accidental_presence_guard import (  # noqa: E402
    ACCIDENTAL_PRESENCE_FEATURE_NAMES,
    accidental_hog_features_at_position,
)
from scorescan.util import atomic_write_json  # noqa: E402
from scorescan.visual_evidence import (  # noqa: E402
    SYMBOL_GUARD_HEIGHT,
    SYMBOL_GUARD_WIDTH,
)

EXPECTED_DATASET_NAME = "scorescan-muse-omr-registered-scan-regions-v1"
EXPECTED_ROLE = "training_only_disjoint_from_external_release_holdout"
EXPECTED_HOLDOUT_DATASET_NAME = "scorescan-muse-omr-registered-scan-holdout-v1"
EXPECTED_HOLDOUT_ROLE = "external_scan_degraded_development_benchmark_not_training"
EXPECTED_LICENSE = "CC0-1.0"
GEOMETRY_CLASSES = {"Accidental", "BarLine", "Note", "StaffLines"}
PAGE_RE = re.compile(r"page-(\d+)\.svg$", re.IGNORECASE)
JITTER_PIXELS = (-3, 0, 3)


@dataclass(frozen=True)
class StaffGeometry:
    box: Box
    spacing: float


@dataclass(frozen=True)
class RegisteredSample:
    features: tuple[float, ...]
    label: int
    group_key: str
    pair_id: int
    page_number: int
    staff_index: int
    measure_index: int
    note_box: Box
    accidental_box: Box | None
    jitter_x: int


def _centre(box: Box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _scale_box(box: Sequence[float], view_box: Box, image: np.ndarray) -> Box:
    vx0, vy0, vx1, vy1 = view_box
    if vx1 <= vx0 or vy1 <= vy0:
        raise ValueError("invalid SVG viewBox")
    sx = image.shape[1] / (vx1 - vx0)
    sy = image.shape[0] / (vy1 - vy0)
    return (
        (float(box[0]) - vx0) * sx,
        (float(box[1]) - vy0) * sy,
        (float(box[2]) - vx0) * sx,
        (float(box[3]) - vy0) * sy,
    )


def group_staff_lines(line_boxes: Iterable[Box]) -> list[StaffGeometry]:
    """Group five rendered staff-line paths without assuming a fixed page scale."""
    lines = sorted(
        (
            tuple(float(value) for value in box)
            for box in line_boxes
            if box[2] > box[0] and box[3] > box[1]
        ),
        key=lambda box: (_centre(box)[1], box[0], box[2]),
    )
    staffs: list[StaffGeometry] = []
    index = 0
    while index + 4 < len(lines):
        window = lines[index : index + 5]
        centres = [_centre(box)[1] for box in window]
        gaps = np.diff(np.asarray(centres, dtype=np.float64))
        mean_gap = float(np.mean(gaps))
        shared_left = max(box[0] for box in window)
        shared_right = min(box[2] for box in window)
        line_height = max(box[3] - box[1] for box in window)
        consistent = (
            mean_gap > max(1.0, line_height * 2.0)
            and float(np.max(np.abs(gaps - mean_gap), initial=0.0))
            <= max(1.0, mean_gap * 0.08)
            and shared_right - shared_left >= mean_gap * 8.0
        )
        if not consistent:
            index += 1
            continue
        staffs.append(
            StaffGeometry(
                box=(
                    min(box[0] for box in window),
                    centres[0],
                    max(box[2] for box in window),
                    centres[-1],
                ),
                spacing=mean_gap,
            )
        )
        index += 5
    return staffs


def _assign_to_staff(
    boxes: Sequence[Box],
    staffs: Sequence[StaffGeometry],
) -> dict[int, list[Box]]:
    assigned: dict[int, list[Box]] = defaultdict(list)
    for box in boxes:
        x, y = _centre(box)
        candidates: list[tuple[float, int]] = []
        for index, staff in enumerate(staffs):
            sx0, sy0, sx1, sy1 = staff.box
            if not (
                sx0 - staff.spacing <= x <= sx1 + staff.spacing
                and sy0 - 3.0 * staff.spacing
                <= y
                <= sy1 + 3.0 * staff.spacing
            ):
                continue
            distance = abs(y - (sy0 + sy1) / 2.0) / staff.spacing
            candidates.append((distance, index))
        if candidates:
            assigned[min(candidates)[1]].append(box)
    return assigned


def _measure_edges(
    staff: StaffGeometry,
    barlines: Sequence[Box],
) -> list[float]:
    sx0, sy0, sx1, sy1 = staff.box
    values = [sx0, sx1]
    for box in barlines:
        x, _y = _centre(box)
        vertical_overlap = min(sy1, box[3]) - max(sy0, box[1])
        if sx0 <= x <= sx1 and vertical_overlap >= staff.spacing * 1.5:
            values.append(x)
    values.sort()
    merged: list[float] = []
    for value in values:
        if merged and value - merged[-1] <= staff.spacing * 0.35:
            merged[-1] = (merged[-1] + value) / 2.0
        else:
            merged.append(value)
    return merged


def _paired_accidentals(
    notes: Sequence[Box],
    accidentals: Sequence[Box],
    spacing: float,
) -> dict[int, Box]:
    """Associate each printed accidental with at most one nearby notehead."""
    candidates: list[tuple[float, int, int]] = []
    for accidental_index, accidental in enumerate(accidentals):
        ax, ay = _centre(accidental)
        for note_index, note in enumerate(notes):
            nx, ny = _centre(note)
            horizontal_gap = note[0] - accidental[2]
            if not (-0.25 * spacing <= horizontal_gap <= 1.65 * spacing):
                continue
            vertical_gap = abs(ny - ay)
            if vertical_gap > 0.95 * spacing:
                continue
            score = (
                max(0.0, horizontal_gap) / spacing
                + vertical_gap / spacing
                + max(0.0, ax - nx) / spacing * 4.0
            )
            candidates.append((score, accidental_index, note_index))
    matches: dict[int, Box] = {}
    used_accidentals: set[int] = set()
    used_notes: set[int] = set()
    for _score, accidental_index, note_index in sorted(candidates):
        if accidental_index in used_accidentals or note_index in used_notes:
            continue
        used_accidentals.add(accidental_index)
        used_notes.add(note_index)
        matches[note_index] = accidentals[accidental_index]
    return matches


def _guard_image(
    page: np.ndarray,
    *,
    left: float,
    right: float,
    staff: StaffGeometry,
) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
    sx0, sy0, sx1, sy1 = staff.box
    spacing = staff.spacing
    x0 = max(0, int(math.floor(left)))
    x1 = min(page.shape[1], int(math.ceil(right)))
    y0 = max(0, int(math.floor(sy0 - 2.0 * spacing)))
    y1 = min(page.shape[0], int(math.ceil(sy1 + 2.0 * spacing)))
    if x1 - x0 < spacing * 4.0 or y1 - y0 < spacing * 5.5:
        return None
    crop = page[y0:y1, x0:x1]
    _threshold, binary = cv2.threshold(
        crop,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(15, int(round(max(spacing, 3.0) * 5.5))), 1),
    )
    staff_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    profile = cv2.subtract(binary, staff_mask)
    bar_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(5, int(round(spacing * 3.75)))),
    )
    profile = cv2.subtract(
        profile,
        cv2.morphologyEx(profile, cv2.MORPH_OPEN, bar_kernel),
    )
    edge = max(1, int(round(profile.shape[1] * 0.025)))
    if profile.shape[1] > edge * 2:
        profile[:, :edge] = 0
        profile[:, -edge:] = 0
    resized = cv2.resize(
        profile,
        (SYMBOL_GUARD_WIDTH, SYMBOL_GUARD_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    return resized, (float(x0), float(y0), float(x1), float(y1))


def page_samples(
    *,
    svg_path: Path,
    scan_path: Path,
    pair_id: int,
    page_number: int,
    source_key: str,
    negative_ratio: float = 2.0,
) -> tuple[list[RegisteredSample], Counter[str]]:
    page = cv2.imread(str(scan_path), cv2.IMREAD_GRAYSCALE)
    if page is None:
        raise ValueError(f"registered scan page is unreadable: {scan_path}")
    view_box, objects, _renderer, excluded = svg_class_objects(
        svg_path,
        GEOMETRY_CLASSES,
    )
    scaled: dict[str, list[Box]] = defaultdict(list)
    for obj in objects:
        scaled[str(obj["svg_class"])].append(
            _scale_box(obj["box_xyxy"], view_box, page)
        )
    staffs = group_staff_lines(scaled["StaffLines"])
    counters: Counter[str] = Counter(excluded)
    if not staffs:
        counters["pages_without_five_line_staff"] += 1
        return [], counters
    notes_by_staff = _assign_to_staff(scaled["Note"], staffs)
    accidentals_by_staff = _assign_to_staff(scaled["Accidental"], staffs)
    result: list[RegisteredSample] = []
    for staff_index, staff in enumerate(staffs):
        edges = _measure_edges(staff, scaled["BarLine"])
        staff_notes = notes_by_staff.get(staff_index, [])
        staff_accidentals = accidentals_by_staff.get(staff_index, [])
        for measure_index, (left, right) in enumerate(zip(edges, edges[1:])):
            notes = sorted(
                [
                    box
                    for box in staff_notes
                    if left < _centre(box)[0] < right
                ],
                key=lambda box: (_centre(box)[0], _centre(box)[1]),
            )
            if not notes:
                continue
            accidentals = [
                box
                for box in staff_accidentals
                if left < _centre(box)[0] < right
            ]
            matches = _paired_accidentals(notes, accidentals, staff.spacing)
            positives = sorted(matches)
            negatives = [index for index in range(len(notes)) if index not in matches]
            maximum_negatives = max(1, int(math.ceil(len(positives) * negative_ratio)))
            if positives:
                # Deterministic coverage across the complete measure instead of
                # selecting only its earliest easy negatives.
                stride = max(1, len(negatives) // max(maximum_negatives, 1))
                negatives = negatives[::stride][:maximum_negatives]
            else:
                negatives = negatives[:1]
            selected = [(index, 1) for index in positives] + [
                (index, 0) for index in negatives
            ]
            evidence = _guard_image(
                page,
                left=left,
                right=right,
                staff=staff,
            )
            if evidence is None:
                counters["undersized_measure_crop"] += len(selected)
                continue
            guard, crop_box = evidence
            cx0, cy0, cx1, cy1 = crop_box
            group_key = (
                f"{source_key}/pair-{pair_id:04d}/page-{page_number}/"
                f"staff-{staff_index}/measure-{measure_index}"
            )
            for note_index, label in selected:
                note = notes[note_index]
                nx, ny = _centre(note)
                base_x = (nx - cx0) / max(cx1 - cx0, 1.0)
                y_ratio = (ny - cy0) / max(cy1 - cy0, 1.0)
                for jitter_x in JITTER_PIXELS:
                    x_ratio = base_x + jitter_x / (SYMBOL_GUARD_WIDTH - 1)
                    features = accidental_hog_features_at_position(
                        guard,
                        x_ratio,
                        y_ratio,
                    )
                    result.append(
                        RegisteredSample(
                            features=tuple(features),
                            label=label,
                            group_key=group_key,
                            pair_id=pair_id,
                            page_number=page_number,
                            staff_index=staff_index,
                            measure_index=measure_index,
                            note_box=note,
                            accidental_box=matches.get(note_index),
                            jitter_x=jitter_x,
                        )
                    )
    counters["staffs"] += len(staffs)
    counters["notes"] += len(scaled["Note"])
    counters["accidentals"] += len(scaled["Accidental"])
    return result, counters


def _validate_region_report(region_dir: Path) -> dict[str, Any]:
    path = region_dir / "prepare-report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("name") != EXPECTED_DATASET_NAME:
        raise ValueError("registered region dataset name is not training-compatible")
    if report.get("role") != EXPECTED_ROLE:
        raise ValueError("registered region dataset is not training-role isolated")
    if report.get("license") != EXPECTED_LICENSE:
        raise ValueError("registered region dataset license is not CC0-1.0")
    if report.get("forbidden_work_overlap"):
        raise ValueError("registered region dataset overlaps its forbidden holdout")
    intersections = report.get("split_intersections", {})
    if any(intersections.get(name) for name in intersections):
        raise ValueError("registered region dataset has split leakage")
    return report


def _validate_holdout_region_report(region_dir: Path) -> dict[str, Any]:
    path = region_dir / "prepare-report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("name") != EXPECTED_HOLDOUT_DATASET_NAME:
        raise ValueError("registered holdout region dataset name is incompatible")
    if report.get("role") != EXPECTED_HOLDOUT_ROLE:
        raise ValueError("registered holdout is not evaluation-only")
    if report.get("license") != EXPECTED_LICENSE:
        raise ValueError("registered holdout license is not CC0-1.0")
    if report.get("forbidden_selection_overlap"):
        raise ValueError("registered holdout overlaps the forbidden training selection")
    if report.get("forbidden_work_overlap"):
        raise ValueError("registered holdout overlaps a training work")
    intersections = report.get("split_intersections", {})
    if any(intersections.get(name) for name in intersections):
        raise ValueError("registered holdout has split leakage")
    accepted = report.get("accepted", [])
    if not isinstance(accepted, list) or len(accepted) < 200:
        raise ValueError("registered holdout has insufficient accepted coverage")
    if any(str(row.get("split")) != "test" for row in accepted):
        raise ValueError("registered holdout contains a non-test source")
    return report


def _page_number(path: Path) -> int:
    match = PAGE_RE.search(path.name)
    if not match:
        raise ValueError(f"reference SVG has no page number: {path}")
    return int(match.group(1))


def _render_svg_pages(
    *,
    source: Path,
    output_dir: Path,
    musescore_exe: Path,
    timeout_seconds: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("page-*.svg"), key=_page_number)
    if existing:
        return existing
    output = output_dir / "page.svg"
    subprocess.run(
        [str(musescore_exe), "-o", str(output), str(source)],
        check=True,
        timeout=timeout_seconds,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    rendered = sorted(output_dir.glob("page-*.svg"), key=_page_number)
    if not rendered and output.is_file():
        single = output_dir / "page-1.svg"
        output.replace(single)
        rendered = [single]
    if not rendered:
        raise RuntimeError(f"MuseScore did not render SVG pages for {source}")
    return rendered


def _write_npz(path: Path, samples: Sequence[RegisteredSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    groups_by_key = {
        key: index
        for index, key in enumerate(
            sorted({sample.group_key for sample in samples}),
            start=1,
        )
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            features=np.asarray(
                [sample.features for sample in samples],
                dtype=np.float64,
            ).reshape((-1, len(ACCIDENTAL_PRESENCE_FEATURE_NAMES))),
            labels=np.asarray([sample.label for sample in samples], dtype=np.int64),
            groups=np.asarray(
                [groups_by_key[sample.group_key] for sample in samples],
                dtype=np.int64,
            ),
            symbols=np.asarray(
                ["present" if sample.label else "none" for sample in samples],
                dtype="<U7",
            ),
        )
    temporary.replace(path)


def _cap_pair_samples(
    samples: Sequence[RegisteredSample],
    maximum_samples: int,
) -> list[RegisteredSample]:
    """Keep complete jitter/measure groups with deterministic page-wide coverage."""
    if maximum_samples <= 0 or len(samples) <= maximum_samples:
        return list(samples)
    grouped: dict[str, list[RegisteredSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.group_key].append(sample)
    positive_groups = sorted(
        (
            rows
            for rows in grouped.values()
            if any(sample.label for sample in rows)
        ),
        key=lambda rows: rows[0].group_key,
    )
    negative_groups = sorted(
        (
            rows
            for rows in grouped.values()
            if not any(sample.label for sample in rows)
        ),
        key=lambda rows: rows[0].group_key,
    )
    ordered: list[list[RegisteredSample]] = []
    while positive_groups or negative_groups:
        if positive_groups:
            ordered.append(positive_groups.pop(0))
        if negative_groups:
            ordered.append(negative_groups.pop(0))
    selected: list[RegisteredSample] = []
    for rows in ordered:
        if selected and len(selected) + len(rows) > maximum_samples:
            continue
        selected.extend(rows)
        if len(selected) >= maximum_samples:
            break
    if not selected:
        # One extremely dense measure can exceed the cap; group isolation is
        # more important than a hard byte ceiling.
        selected.extend(ordered[0])
    return selected


def prepare_dataset(args: argparse.Namespace) -> dict[str, Any]:
    independent_holdout = bool(getattr(args, "independent_holdout", False))
    report = (
        _validate_holdout_region_report(args.region_dir)
        if independent_holdout
        else _validate_region_report(args.region_dir)
    )
    accepted = report.get("accepted", [])
    if not isinstance(accepted, list) or not accepted:
        raise ValueError("registered region report has no accepted pairs")
    all_samples: dict[str, list[RegisteredSample]] = defaultdict(list)
    counters: Counter[str] = Counter()
    work_sets: dict[str, set[str]] = defaultdict(set)
    metadata: list[dict[str, Any]] = []
    for pair_index, row in enumerate(accepted, start=1):
        pair_id = int(row["pair_id"])
        split = str(row["split"])
        source_key = str(row["source_key"])
        work_sets[split].add(source_key)
        reference_dir = args.region_dir / "reference_pages" / f"pair-{pair_id:04d}"
        reference_pages = sorted(reference_dir.glob("page-*.svg"), key=_page_number)
        if not reference_pages:
            if args.musescore_exe is None:
                raise FileNotFoundError(
                    f"reference SVG pages are missing for pair {pair_id}"
                )
            reference_pages = _render_svg_pages(
                source=args.dataset_dir / "mscz" / f"score_file_{pair_id}.mscz",
                output_dir=reference_dir,
                musescore_exe=args.musescore_exe,
                timeout_seconds=args.render_timeout_seconds,
            )
        pair_samples: list[RegisteredSample] = []
        for svg_path in reference_pages:
            page_number = _page_number(svg_path)
            scan_path = (
                args.region_dir
                / "pages"
                / f"pair-{pair_id:04d}"
                / f"page-{page_number}.jpg"
            )
            if not scan_path.is_file():
                counters["registered_page_missing_or_rejected"] += 1
                continue
            samples, page_counters = page_samples(
                svg_path=svg_path,
                scan_path=scan_path,
                pair_id=pair_id,
                page_number=page_number,
                source_key=source_key,
                negative_ratio=args.negative_ratio,
            )
            pair_samples.extend(samples)
            counters.update(page_counters)
            counters["pages"] += 1
        uncapped_samples = len(pair_samples)
        pair_samples = _cap_pair_samples(
            pair_samples,
            args.maximum_samples_per_pair,
        )
        all_samples[split].extend(pair_samples)
        counters[f"{split}_pairs"] += 1
        counters[f"{split}_samples"] += len(pair_samples)
        counters["samples_removed_by_per_pair_cap"] += (
            uncapped_samples - len(pair_samples)
        )
        metadata.extend(
            {
                "split": split,
                "group_key": sample.group_key,
                "pair_id": sample.pair_id,
                "page_number": sample.page_number,
                "staff_index": sample.staff_index,
                "measure_index": sample.measure_index,
                "label": sample.label,
                "note_box": [round(value, 4) for value in sample.note_box],
                "accidental_box": (
                    [round(value, 4) for value in sample.accidental_box]
                    if sample.accidental_box is not None
                    else None
                ),
                "jitter_x": sample.jitter_x,
            }
            for sample in pair_samples
        )
        print(
            f"[{pair_index}/{len(accepted)}] pair {pair_id}: "
            f"{len(pair_samples)} samples",
            flush=True,
        )
    split_names = ("train", "calibration", "test")
    for split in split_names:
        _write_npz(args.output_dir / f"{split}.npz", all_samples[split])
    metadata_path = args.output_dir / "samples.jsonl"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = metadata_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in metadata:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(metadata_path)
    intersections = {
        "train_calibration": sorted(work_sets["train"] & work_sets["calibration"]),
        "train_test": sorted(work_sets["train"] & work_sets["test"]),
        "calibration_test": sorted(work_sets["calibration"] & work_sets["test"]),
    }
    if any(intersections.values()):
        raise RuntimeError("prepared accidental dataset has work-level leakage")
    output_report = {
        "format": 1,
        "name": (
            "scorescan-registered-scan-accidental-presence-holdout-v1"
            if independent_holdout
            else "scorescan-registered-scan-accidental-presence-v1"
        ),
        "role": (
            "independent_holdout_evaluation_only"
            if independent_holdout
            else "training_calibration_and_internal_test_only"
        ),
        "source_region_report": str(args.region_dir / "prepare-report.json"),
        "source_region_report_sha256": sha256_file(
            args.region_dir / "prepare-report.json"
        ),
        "source_role": report["role"],
        "license": report["license"],
        "feature_names": list(ACCIDENTAL_PRESENCE_FEATURE_NAMES),
        "jitter_pixels": list(JITTER_PIXELS),
        "negative_ratio": args.negative_ratio,
        "maximum_samples_per_pair": args.maximum_samples_per_pair,
        "samples_by_split": {
            split: len(all_samples[split])
            for split in split_names
        },
        "positive_samples_by_split": {
            split: sum(sample.label for sample in all_samples[split])
            for split in split_names
        },
        "groups_by_split": {
            split: len({sample.group_key for sample in all_samples[split]})
            for split in split_names
        },
        "works_by_split": {
            split: len(work_sets[split])
            for split in split_names
        },
        "work_intersections": intersections,
        "counters": dict(sorted(counters.items())),
        "holdout_used_for_training": False,
        "training_use_authorized": not independent_holdout,
    }
    atomic_write_json(args.output_dir / "prepare-report.json", output_report)
    return output_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--region-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--musescore-exe", type=Path)
    parser.add_argument("--render-timeout-seconds", type=int, default=180)
    parser.add_argument("--negative-ratio", type=float, default=2.0)
    parser.add_argument("--maximum-samples-per-pair", type=int, default=240)
    parser.add_argument(
        "--independent-holdout",
        action="store_true",
        help=(
            "accept only the external evaluation role and emit an "
            "evaluation-only artifact"
        ),
    )
    args = parser.parse_args()
    if args.negative_ratio < 1.0 or args.negative_ratio > 8.0:
        raise ValueError("negative ratio must be in [1, 8]")
    if args.maximum_samples_per_pair < 60:
        raise ValueError("maximum samples per pair must be at least 60")
    report = prepare_dataset(args)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
