from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.tools.prepare_registered_accidental_presence import (
    EXPECTED_DATASET_NAME,
    EXPECTED_HOLDOUT_DATASET_NAME,
    EXPECTED_HOLDOUT_ROLE,
    EXPECTED_LICENSE,
    EXPECTED_ROLE,
    RegisteredSample,
    _cap_pair_samples,
    _paired_accidentals,
    _validate_holdout_region_report,
    _validate_region_report,
    group_staff_lines,
    page_samples,
)
from scorescan.accidental_presence_guard import ACCIDENTAL_PRESENCE_FEATURE_NAMES


def _line(y: float) -> tuple[float, float, float, float]:
    return (10.0, y, 190.0, y + 1.0)


def test_groups_exact_five_line_staffs_and_ignores_noise() -> None:
    boxes = [_line(y) for y in (20, 30, 40, 50, 60, 90, 100, 110, 120, 130)]
    boxes.append((5.0, 8.0, 7.0, 9.0))
    staffs = group_staff_lines(boxes)
    assert len(staffs) == 2
    assert staffs[0].spacing == pytest.approx(10.0)
    assert staffs[0].box == (10.0, 20.5, 190.0, 60.5)


def test_accidental_matching_is_one_to_one_and_position_bounded() -> None:
    notes = [
        (50.0, 35.0, 62.0, 43.0),
        (50.0, 55.0, 62.0, 63.0),
        (120.0, 35.0, 132.0, 43.0),
    ]
    accidentals = [
        (36.0, 29.0, 46.0, 49.0),
        (10.0, 29.0, 20.0, 49.0),
    ]
    matches = _paired_accidentals(notes, accidentals, spacing=10.0)
    assert matches == {0: accidentals[0]}


def _write_page(svg: Path, scan: Path) -> None:
    lines = "\n".join(
        f'<path class="StaffLines" d="M10,{y} L190,{y} L190,{y + 1} L10,{y + 1} Z"/>'
        for y in (50, 60, 70, 80, 90)
    )
    svg.write_text(
        f"""\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 140">
  {lines}
  <path class="BarLine" d="M10,48 L12,48 L12,92 L10,92 Z"/>
  <path class="BarLine" d="M100,48 L102,48 L102,92 L100,92 Z"/>
  <path class="BarLine" d="M188,48 L190,48 L190,92 L188,92 Z"/>
  <path class="Note" d="M65,65 L77,65 L77,73 L65,73 Z"/>
  <path class="Accidental" d="M50,57 L60,57 L60,77 L50,77 Z"/>
  <path class="Note" d="M145,75 L157,75 L157,83 L145,83 Z"/>
</svg>
""",
        encoding="utf-8",
    )
    image = np.full((140, 200), 255, dtype=np.uint8)
    for y in (50, 60, 70, 80, 90):
        cv2.line(image, (10, y), (190, y), 0, 1)
    cv2.rectangle(image, (65, 65), (77, 73), 0, -1)
    cv2.rectangle(image, (50, 57), (60, 77), 0, 2)
    cv2.rectangle(image, (145, 75), (157, 83), 0, -1)
    assert cv2.imwrite(str(scan), image)


def test_registered_page_produces_balanced_deployed_descriptors(
    tmp_path: Path,
) -> None:
    svg = tmp_path / "page-1.svg"
    scan = tmp_path / "page-1.jpg"
    _write_page(svg, scan)
    samples, counters = page_samples(
        svg_path=svg,
        scan_path=scan,
        pair_id=1,
        page_number=1,
        source_key="work/one",
    )
    assert counters["staffs"] == 1
    assert {sample.label for sample in samples} == {0, 1}
    assert len(samples) == 6
    assert all(
        len(sample.features) == len(ACCIDENTAL_PRESENCE_FEATURE_NAMES)
        for sample in samples
    )
    assert len({sample.group_key for sample in samples}) == 2


def test_region_contract_rejects_independent_holdout(tmp_path: Path) -> None:
    report = {
        "name": EXPECTED_DATASET_NAME,
        "role": EXPECTED_ROLE,
        "license": EXPECTED_LICENSE,
        "forbidden_work_overlap": [],
        "split_intersections": {
            "train_calibration": [],
            "train_test": [],
            "calibration_test": [],
        },
    }
    (tmp_path / "prepare-report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    assert _validate_region_report(tmp_path)["role"] == EXPECTED_ROLE

    report["role"] = "external_scan_degraded_development_benchmark_not_training"
    (tmp_path / "prepare-report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="training-role"):
        _validate_region_report(tmp_path)


def test_holdout_contract_requires_200_test_works_and_no_overlap(
    tmp_path: Path,
) -> None:
    report = {
        "name": EXPECTED_HOLDOUT_DATASET_NAME,
        "role": EXPECTED_HOLDOUT_ROLE,
        "license": EXPECTED_LICENSE,
        "forbidden_selection_overlap": [],
        "forbidden_work_overlap": [],
        "split_intersections": {
            "train_calibration": [],
            "train_test": [],
            "calibration_test": [],
        },
        "accepted": [
            {"pair_id": index, "split": "test"}
            for index in range(200)
        ],
    }
    path = tmp_path / "prepare-report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert _validate_holdout_region_report(tmp_path)["role"] == EXPECTED_HOLDOUT_ROLE

    report["accepted"][0]["split"] = "train"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="non-test"):
        _validate_holdout_region_report(tmp_path)


def test_pair_cap_preserves_complete_measure_groups() -> None:
    feature = (0.0,) * len(ACCIDENTAL_PRESENCE_FEATURE_NAMES)
    samples = [
        RegisteredSample(
            features=feature,
            label=1 if group % 2 == 0 else 0,
            group_key=f"group-{group}",
            pair_id=1,
            page_number=1,
            staff_index=0,
            measure_index=group,
            note_box=(0, 0, 1, 1),
            accidental_box=(0, 0, 1, 1) if group % 2 == 0 else None,
            jitter_x=jitter,
        )
        for group in range(10)
        for jitter in (-3, 0, 3)
    ]
    capped = _cap_pair_samples(samples, 12)
    assert len(capped) == 12
    counts: dict[str, int] = {}
    for sample in capped:
        counts[sample.group_key] = counts.get(sample.group_key, 0) + 1
    assert set(counts.values()) == {3}
    assert {sample.label for sample in capped} == {0, 1}
