from __future__ import annotations

import base64
import json
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pytest

from scorescan.imaging import generate_omr_variants
from scorescan.layout import PageLayout, StaffSystem
from scorescan.models import PageInfo
from scorescan.score_ir import MeasureIR, NoteIR, PitchIR
from scorescan.visual_evidence import (
    EVENT_GRID_NAMES,
    EVENT_GRID_SIZE,
    FEATURE_NAMES,
    VisualMeasureCalibrator,
    VisualMeasureEvidence,
    extract_crop_features,
    pitch_local_gaps,
    pitch_transaction_gap_pair,
    semantic_event_grids,
    extract_page_measure_evidence,
    map_evidence_to_measure,
    write_visual_evidence,
)
from scorescan.util import sha256_file


def make_note(index: int) -> NoteIR:
    return NoteIR(
        onset=Fraction(index, 2),
        duration=Fraction(1, 2),
        voice="1",
        pitch=PitchIR("C", Fraction(0, 1), 4),
        rest=False,
        chord=False,
        grace=False,
        note_type="eighth",
        dots=0,
        accidental="",
        ties=(),
        slurs=(),
        articulations=(),
        ornaments=(),
        tuple_ratio=None,
    )


def test_visual_measure_model_prefers_matching_density() -> None:
    image = np.full((140, 420), 255, np.uint8)
    spacing = 12
    staff_top = 55
    for line in range(5):
        cv2.line(image, (10, staff_top + line * spacing), (410, staff_top + line * spacing), 0, 1)
    notes = tuple(make_note(index) for index in range(10))
    for index in range(10):
        x = 35 + index * 35
        y = staff_top + 2 * spacing + ((index % 5) - 2) * spacing // 2
        cv2.ellipse(image, (x, y), (6, 4), -15, 0, 360, 0, -1)
        cv2.line(image, (x + 5, y), (x + 5, y - 30), 0, 1)
    features = extract_crop_features(
        image,
        spacing=spacing,
        staff_top=staff_top,
        staff_bottom=staff_top + 4 * spacing,
    )
    evidence = VisualMeasureEvidence(1, 1, 1, (0, 0, 420, 140), spacing, **features)
    dense = MeasureIR(8, (5, 4), (0, "major"), ("G", 2, 0), notes, (), ())
    sparse = MeasureIR(8, (5, 4), (0, "major"), ("G", 2, 0), notes[:1], (), ())
    model = VisualMeasureCalibrator()
    assert model.enabled
    assert model.predict_probability(evidence, dense) > model.predict_probability(evidence, sparse)


def test_rhythm_guard_evidence_is_fixed_size_bounded_png() -> None:
    image = np.full((180, 480), 255, np.uint8)
    spacing = 14
    staff_top = 60
    for line in range(5):
        cv2.line(image, (8, staff_top + line * spacing), (472, staff_top + line * spacing), 0, 1)
    cv2.ellipse(image, (180, staff_top + 2 * spacing), (7, 5), -15, 0, 360, 0, -1)
    cv2.line(image, (186, staff_top + 2 * spacing), (186, staff_top - spacing), 0, 2)
    features = extract_crop_features(
        image, spacing=spacing, staff_top=staff_top, staff_bottom=staff_top + 4 * spacing
    )
    encoded = features["rhythm_guard_image"]
    assert isinstance(encoded, str)
    assert 0 < len(encoded) < 65_536
    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(encoded, validate=True), dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )
    assert decoded is not None
    assert decoded.shape == (64, 128)

    symbol_encoded = features["symbol_guard_image"]
    assert isinstance(symbol_encoded, str)
    assert 0 < len(symbol_encoded) < 65_536
    symbol_decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(symbol_encoded, validate=True), dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )
    assert symbol_decoded is not None
    assert symbol_decoded.shape == (96, 256)


def test_page_evidence_and_staff_normalised_variant(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image = np.full((300, 800), 255, np.uint8)
    lines = [100, 108, 116, 124, 132]
    for y in lines:
        cv2.line(image, (40, y), (760, y), 0, 1)
    for x in (40, 280, 520, 760):
        cv2.line(image, (x, lines[0]), (x, lines[-1]), 0, 1)
    cv2.imwrite(str(image_path), image)
    system = StaffSystem(1, [float(y) for y in lines], 70, 165, 40, 760, 8.0, [280, 520], 3)
    layout = PageLayout(800, 300, [system], 1.0)
    evidence = extract_page_measure_evidence(image_path, layout, page_index=1)
    assert len(evidence) == 3
    assert map_evidence_to_measure(evidence, 2, 3) == evidence[2]

    page = PageInfo(1, "page.png", str(image_path), normalized_path=str(image_path))
    variants = dict(generate_omr_variants(page, tmp_path / "variants", layout=layout))
    assert "staffnorm" in variants
    scaled = cv2.imread(str(variants["staffnorm"]), cv2.IMREAD_GRAYSCALE)
    assert scaled is not None
    assert scaled.shape[1] > image.shape[1]


def test_page_evidence_uses_omr_measure_count_when_barlines_are_missing(tmp_path: Path) -> None:
    image_path = tmp_path / "page_no_barlines.png"
    image = np.full((520, 900), 255, np.uint8)
    systems: list[StaffSystem] = []
    for system_index, top in enumerate((90, 290), start=1):
        lines = [top + offset * 10 for offset in range(5)]
        for y in lines:
            cv2.line(image, (50, y), (850, y), 0, 1)
        systems.append(
            StaffSystem(
                system_index,
                [float(y) for y in lines],
                top - 45,
                top + 85,
                50,
                850,
                10.0,
                [],
                1,
            )
        )
    cv2.imwrite(str(image_path), image)
    layout = PageLayout(900, 520, systems, 0.92)
    evidence = extract_page_measure_evidence(
        image_path,
        layout,
        page_index=1,
        target_measure_count=11,
    )
    assert len(evidence) == 11
    assert [item.system_index for item in evidence].count(1) in {5, 6}
    assert [item.system_index for item in evidence].count(2) in {5, 6}
    assert all(item.bbox[2] > item.bbox[0] for item in evidence)


def test_matching_target_count_preserves_irregular_source_barlines(tmp_path: Path) -> None:
    image_path = tmp_path / "irregular.png"
    image = np.full((260, 800), 255, np.uint8)
    lines = [90, 100, 110, 120, 130]
    for y in lines:
        cv2.line(image, (40, y), (760, y), 0, 1)
    for x in (40, 180, 520, 760):
        cv2.line(image, (x, lines[0]), (x, lines[-1]), 0, 1)
    cv2.imwrite(str(image_path), image)
    system = StaffSystem(
        1,
        [float(y) for y in lines],
        55,
        165,
        40,
        760,
        10.0,
        [180, 520],
        3,
    )

    evidence = extract_page_measure_evidence(
        image_path,
        PageLayout(800, 260, [system], 1.0),
        page_index=1,
        target_measure_count=3,
    )

    assert len(evidence) == 3
    # One-pixel feature padding is expected; equal-width rebucketing would put
    # these boundaries near x=280 and x=520 instead of x=180 and x=520.
    assert evidence[0].bbox[2] == 182
    assert evidence[1].bbox[0] == 178
    assert evidence[1].bbox[2] == 522


def test_gradient_boosting_probability_calibration_is_applied(tmp_path: Path) -> None:
    model_path = tmp_path / "visual_measure_calibrator.json"
    model_path.write_text(
        json.dumps(
            {
                "model_version": "test-visual-calibration-1",
                "model_type": "gradient_boosting",
                    "feature_names": list(FEATURE_NAMES),
                "intercept": 0.25,
                "learning_rate": 0.5,
                "trees": [
                    {
                        "nodes": [
                            {
                                "feature": -2,
                                "threshold": -2.0,
                                "left": -1,
                                "right": -1,
                                "value": 0.5,
                            }
                        ]
                    }
                ],
                "calibration_intercept": -1.0,
                "calibration_slope": 2.0,
            }
        ),
        encoding="utf-8",
    )
    evidence = VisualMeasureEvidence(
        page_index=1,
        system_index=1,
        measure_index=1,
        bbox=(0, 0, 100, 100),
        spacing=10.0,
        ink_density=0.0,
        nonstaff_ink_density=0.0,
        component_density=0.0,
        notehead_proxy=0.0,
        open_notehead_proxy=0.0,
        stem_proxy=0.0,
        beam_proxy=0.0,
        onset_proxy=0.0,
        compact_mark_proxy=0.0,
        accidental_proxy=0.0,
        above_ink_density=0.0,
        below_ink_density=0.0,
        x_ink_profile=(0.0,) * 8,
        staff_ink_profile=(0.0,) * 9,
    )
    measure = MeasureIR(8, (4, 4), (0, "major"), ("G", 2, 0), (), (), ())
    model = VisualMeasureCalibrator(model_path)
    assert model.enabled
    # Raw score = 0.25 + 0.5 * 0.5 = 0.5.  The deployed calibration maps it
    # to -1 + 2 * 0.5 = 0, therefore the final probability must be exactly 0.5.
    assert model.predict_probability(evidence, measure) == pytest.approx(0.5)


def test_bundled_visual_model_is_verified_v4() -> None:
    model = VisualMeasureCalibrator()
    assert model.enabled
    assert model.model_verified
    assert model.model_version == "scorescan-visual-measure-calibrator-4"



def test_pitch_local_gaps_prefer_matching_semantic_grid() -> None:
    notes = (
        NoteIR(
            onset=Fraction(0, 1),
            duration=Fraction(1, 1),
            voice="1",
            pitch=PitchIR("C", Fraction(0, 1), 4),
            rest=False,
            chord=False,
            grace=False,
            note_type="quarter",
            dots=0,
            accidental="",
            ties=(),
            slurs=(),
            articulations=(),
            ornaments=(),
            tuple_ratio=None,
        ),
        NoteIR(
            onset=Fraction(1, 1),
            duration=Fraction(1, 1),
            voice="1",
            pitch=PitchIR("E", Fraction(0, 1), 4),
            rest=False,
            chord=False,
            grace=False,
            note_type="quarter",
            dots=0,
            accidental="",
            ties=(),
            slurs=(),
            articulations=(),
            ornaments=(),
            tuple_ratio=None,
        ),
    )
    matching = MeasureIR(1, (4, 4), (0, "major"), ("G", 2, 0), notes, (), ())
    shifted_notes = tuple(
        NoteIR(
            onset=note.onset,
            duration=note.duration,
            voice=note.voice,
            pitch=PitchIR("D" if index == 0 else "F", Fraction(0, 1), 4),
            rest=note.rest,
            chord=note.chord,
            grace=note.grace,
            note_type=note.note_type,
            dots=note.dots,
            accidental=note.accidental,
            ties=note.ties,
            slurs=note.slurs,
            articulations=note.articulations,
            ornaments=note.ornaments,
            tuple_ratio=note.tuple_ratio,
        )
        for index, note in enumerate(notes)
    )
    shifted = MeasureIR(1, (4, 4), (0, "major"), ("G", 2, 0), shifted_notes, (), ())
    grid = semantic_event_grids(matching)["pitched_notehead_grid"]
    evidence = VisualMeasureEvidence(
        1, 1, 1, (0, 0, 200, 100), 10.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        (0.0,) * 8, (0.0,) * 9, pitched_notehead_grid=grid,
    )
    matching_gaps = pitch_local_gaps(evidence, matching)
    shifted_gaps = pitch_local_gaps(evidence, shifted)
    assert matching_gaps[:5] == pytest.approx((0.0,) * 5)
    assert sum(shifted_gaps) > sum(matching_gaps)
    assert shifted_gaps[0] > matching_gaps[0]
    assert shifted_gaps[2] > matching_gaps[2]



def test_pitch_transaction_gaps_ignore_unmodified_events() -> None:
    notes = (
        NoteIR(Fraction(0), Fraction(1), "1", PitchIR("C", Fraction(0), 4), False, False, False, "quarter", 0, "", (), (), (), (), None),
        NoteIR(Fraction(1), Fraction(1), "1", PitchIR("E", Fraction(0), 4), False, False, False, "quarter", 0, "", (), (), (), (), None),
        NoteIR(Fraction(2), Fraction(1), "1", PitchIR("G", Fraction(0), 4), False, False, False, "quarter", 0, "", (), (), (), (), None),
    )
    truth = MeasureIR(1, (4, 4), (0, "major"), ("G", 2, 0), notes, (), ())
    wrong_notes = list(notes)
    wrong_notes[1] = NoteIR(
        Fraction(1), Fraction(1), "1", PitchIR("F", Fraction(0), 4), False, False, False,
        "quarter", 0, "", (), (), (), (), None,
    )
    wrong = MeasureIR(1, (4, 4), (0, "major"), ("G", 2, 0), tuple(wrong_notes), (), ())
    grid = semantic_event_grids(truth)["pitched_notehead_grid"]
    evidence = VisualMeasureEvidence(
        1, 1, 1, (0, 0, 200, 100), 10.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        (0.0,) * 8, (0.0,) * 9,
        pitched_notehead_grid=grid,
        pitch_guard_notehead_grid=grid,
        pitch_guard_strict_notehead_grid=grid,
    )
    before, after = pitch_transaction_gap_pair(evidence, wrong, truth, (1,))
    strict_before, strict_after = pitch_transaction_gap_pair(
        evidence, wrong, truth, (1,), strict=True
    )
    assert sum(after) < sum(before)
    assert sum(strict_after) < sum(strict_before)


def test_extracted_visual_profiles_are_bounded_and_staff_normalised() -> None:
    image = np.full((150, 360), 255, np.uint8)
    spacing = 12
    staff_top = 50
    for line in range(5):
        cv2.line(image, (8, staff_top + line * spacing), (352, staff_top + line * spacing), 0, 1)
    for x, line in ((42, 0), (108, 2), (204, 4), (304, 1)):
        y = staff_top + line * spacing
        cv2.ellipse(image, (x, y), (6, 4), -15, 0, 360, 0, -1)
        cv2.line(image, (x + 5, y), (x + 5, y - 28), 0, 1)
    features = extract_crop_features(
        image,
        spacing=spacing,
        staff_top=staff_top,
        staff_bottom=staff_top + 4 * spacing,
    )
    assert len(features["x_ink_profile"]) == 8
    assert len(features["staff_ink_profile"]) == 9
    assert all(0.0 <= value <= 3.0 for value in features["x_ink_profile"])
    assert all(0.0 <= value <= 3.0 for value in features["staff_ink_profile"])
    assert sum(features["x_ink_profile"]) > 0.0
    assert sum(features["staff_ink_profile"]) > 0.0
    for name in EVENT_GRID_NAMES:
        assert len(features[name]) == EVENT_GRID_SIZE
        assert all(0.0 <= value <= 3.0 for value in features[name])


def test_visual_evidence_checkpoint_uses_compact_event_grid_encoding(tmp_path: Path) -> None:
    grid = tuple((index % 17) / 16.0 * 3.0 for index in range(EVENT_GRID_SIZE))
    evidence = VisualMeasureEvidence(
        page_index=1,
        system_index=1,
        measure_index=1,
        bbox=(0, 0, 200, 100),
        spacing=10.0,
        ink_density=0.1,
        nonstaff_ink_density=0.08,
        component_density=0.2,
        notehead_proxy=0.3,
        open_notehead_proxy=0.1,
        stem_proxy=0.2,
        beam_proxy=0.1,
        onset_proxy=0.3,
        compact_mark_proxy=0.05,
        accidental_proxy=0.05,
        above_ink_density=0.01,
        below_ink_density=0.01,
        x_ink_profile=(0.1,) * 8,
        staff_ink_profile=(0.1,) * 9,
        event_ink_grid=grid,
        pitched_notehead_grid=grid,
        beam_grid=grid,
        compact_mark_grid=grid,
        accidental_grid=grid,
        open_notehead_grid=grid,
    )
    path = tmp_path / "visual_evidence.json"
    write_visual_evidence(path, (evidence,))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == 5
    measure = payload["measures"][0]
    assert measure["event_grid_encoding"] == "uint8-base64-v1"
    assert measure["event_grid_shape"] == [17, 16]
    assert measure["event_grid_scale"] == 3.0
    assert set(measure["event_grids"]) == set(EVENT_GRID_NAMES)
    assert all(
        len(base64.b64decode(measure["event_grids"][name])) == EVENT_GRID_SIZE
        for name in EVENT_GRID_NAMES
    )
    dense_bytes = len(json.dumps(evidence.to_dict(), ensure_ascii=False).encode("utf-8"))
    assert path.stat().st_size < dense_bytes


def test_visual_calibrator_disables_malformed_forest(tmp_path: Path) -> None:
    model_path = tmp_path / "visual_measure_calibrator.json"
    payload = {
        "model_version": "broken-visual-forest",
        "model_type": "random_forest",
        "feature_names": list(FEATURE_NAMES),
        "calibration_intercept": 0.0,
        "calibration_slope": 1.0,
        "trees": [
            {
                "nodes": [
                    {"feature": 0, "threshold": 0.0, "left": 0, "right": 0, "value": 0.0}
                ]
            }
        ],
    }
    model_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = {
        "format": 1,
        "models": [
            {
                "file": model_path.name,
                "role": "visual_measure_compatibility",
                "model_version": payload["model_version"],
                "sha256": sha256_file(model_path),
                "bytes": model_path.stat().st_size,
            }
        ],
    }
    (tmp_path / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    evidence = VisualMeasureEvidence(
        page_index=1,
        system_index=1,
        measure_index=1,
        bbox=(0, 0, 100, 100),
        spacing=10.0,
        ink_density=0.0,
        nonstaff_ink_density=0.0,
        component_density=0.0,
        notehead_proxy=0.0,
        open_notehead_proxy=0.0,
        stem_proxy=0.0,
        beam_proxy=0.0,
        onset_proxy=0.0,
        compact_mark_proxy=0.0,
        accidental_proxy=0.0,
        above_ink_density=0.0,
        below_ink_density=0.0,
        x_ink_profile=(0.0,) * 8,
        staff_ink_profile=(0.0,) * 9,
    )
    measure = MeasureIR(8, (4, 4), (0, "major"), ("G", 2, 0), (), (), ())
    model = VisualMeasureCalibrator(model_path)
    assert not model.enabled
    assert model.predict_probability(evidence, measure) == pytest.approx(0.5)
