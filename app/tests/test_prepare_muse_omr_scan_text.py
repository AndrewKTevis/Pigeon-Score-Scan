from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.tools import prepare_muse_omr_scan_text as module


def test_crop_quality_rejects_blank_and_accepts_text_like_crop() -> None:
    blank = Image.new("L", (120, 30), 255)
    accepted, _quality = module._crop_quality(
        blank,
        minimum_stddev=4.0,
        minimum_dark_fraction=0.003,
    )
    assert not accepted

    text_like = blank.copy()
    ImageDraw.Draw(text_like).rectangle((20, 8, 95, 20), fill=30)
    accepted, quality = module._crop_quality(
        text_like,
        minimum_stddev=4.0,
        minimum_dark_fraction=0.003,
    )
    assert accepted
    assert quality["dark_fraction"] > 0.1


def test_visual_presence_distinguishes_registered_text_from_stain() -> None:
    reference = Image.new("L", (180, 50), 255)
    draw = ImageDraw.Draw(reference)
    draw.rectangle((28, 12, 38, 39), fill=20)
    draw.rectangle((42, 12, 52, 39), fill=20)
    draw.rectangle((56, 12, 90, 20), fill=20)
    scan = Image.new("L", reference.size, 225)
    scan.paste(reference.crop((20, 5, 100, 45)), (23, 7))
    # A smooth exposure gradient is deliberately unlike glyph strokes.
    stain = Image.new("L", reference.size, 235)
    stain_draw = ImageDraw.Draw(stain)
    stain_draw.ellipse((15, -30, 165, 80), fill=190)

    present = module._visual_presence_ncc(reference, scan)
    absent = module._visual_presence_ncc(reference, stain)

    assert present >= module.MINIMUM_SAFE_VISUAL_PRESENCE_NCC
    assert absent < module.MINIMUM_SAFE_VISUAL_PRESENCE_NCC


def test_visual_presence_rejects_blank_crop() -> None:
    blank = Image.new("L", (100, 30), 255)
    assert module._visual_presence_ncc(blank, blank) == 0.0


def test_registered_report_requires_disjoint_training_role(
    tmp_path: Path,
) -> None:
    report = {
        "role": "training_only_disjoint_from_external_release_holdout",
        "source_image_origin": module.SCAN_DEGRADED_IMAGE_ORIGIN,
        "production_evidence_eligible": False,
        "split_intersections": {
            "train_calibration": [],
            "train_test": [],
            "calibration_test": [],
        },
        "accepted": [{"pair_id": 1}],
    }
    (tmp_path / "prepare-report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    assert module._load_registered_report(tmp_path) == report

    missing_origin = dict(report)
    missing_origin.pop("source_image_origin")
    (tmp_path / "prepare-report.json").write_text(
        json.dumps(missing_origin),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="isolation"):
        module._load_registered_report(tmp_path)

    report["role"] = "development_benchmark"
    (tmp_path / "prepare-report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="isolation"):
        module._load_registered_report(tmp_path)

    report.update(
        {
            "role": module.BENCHMARK_SELECTION_ROLE,
            "forbidden_selection_overlap": [],
            "forbidden_work_overlap": [],
        }
    )
    (tmp_path / "prepare-report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    assert module._load_registered_report(
        tmp_path,
        expected_role=module.BENCHMARK_SELECTION_ROLE,
    ) == report
    report["forbidden_selection_overlap"] = [1]
    (tmp_path / "prepare-report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overlaps"):
        module._load_registered_report(
            tmp_path,
            expected_role=module.BENCHMARK_SELECTION_ROLE,
        )


def test_reusable_scan_pdf_manifest_requires_unique_verified_pairs(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "prepare-report.json"
    report = {
        "scan_text_reference_source_version": (
            module.SCAN_TEXT_REFERENCE_SOURCE_VERSION
        ),
        "reference_page_source_counts": {
            "registered_reference_cache": 0,
            "source_mscz_pdf_rerender": 2,
        },
        "reference_page_count": 2,
        "sources": [
            {
                "variants": [
                    {"pair_id": 7, "pdf_sha256": "a" * 64},
                    {"pair_id": 11, "pdf_sha256": "b" * 64},
                ]
            }
        ],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert module._verified_reusable_pdf_hashes(report_path) == {
        7: "a" * 64,
        11: "b" * 64,
    }

    report["sources"][0]["variants"].append(
        {"pair_id": 7, "pdf_sha256": "c" * 64}
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        module._verified_reusable_pdf_hashes(report_path)
