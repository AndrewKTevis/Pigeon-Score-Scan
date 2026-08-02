import json
from dataclasses import dataclass, replace
from pathlib import Path

from lxml import etree

from scorescan.measure_calibration import (
    FEATURE_NAMES,
    MeasureCalibrationInput,
    MeasureCalibrator,
    feature_vector,
)
from scorescan.score_ir import measure_from_xml


@dataclass(frozen=True)
class Candidate:
    score: float = 1000.0
    calibrated_probability: float = 0.8
    valid: bool = True


def make_measure(duration: int = 4, note_type: str = "whole"):
    measure = etree.Element("measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"
    clef = etree.SubElement(attributes, "clef")
    etree.SubElement(clef, "sign").text = "G"
    etree.SubElement(clef, "line").text = "2"
    key = etree.SubElement(attributes, "key")
    etree.SubElement(key, "fifths").text = "0"
    note = etree.SubElement(measure, "note")
    pitch = etree.SubElement(note, "pitch")
    etree.SubElement(pitch, "step").text = "C"
    etree.SubElement(pitch, "octave").text = "4"
    etree.SubElement(note, "duration").text = str(duration)
    etree.SubElement(note, "voice").text = "1"
    etree.SubElement(note, "type").text = note_type
    parsed, _ = measure_from_xml(measure)
    return parsed


def item(measure):
    return MeasureCalibrationInput(
        candidate=Candidate(),
        measure=measure,
        alignment_similarity=0.95,
        exact_support_ratio=0.75,
        semantic_support_ratio=0.90,
        missing_ratio=0.0,
        distance_to_template=0.02,
        distance_to_medoid=0.01,
        mean_peer_distance=0.03,
    )


def test_measure_feature_vector_and_packaged_model() -> None:
    good = item(make_measure())
    bad = item(make_measure(duration=1, note_type="whole"))
    assert len(feature_vector(good)) == len(FEATURE_NAMES)
    calibrator = MeasureCalibrator()
    assert calibrator.enabled
    assert calibrator.hybrid_enabled
    assert calibrator.model_version == "scorescan-measure-forest-3"
    assert calibrator.predict_probability(good) > calibrator.predict_probability(bad)
    assert calibrator.predict_probability(good) >= 0.43
    assert 0.72 <= calibrator.calibrate(good).weight_factor <= 1.18


def test_measure_model_penalises_explicit_accidental_pitch_conflict() -> None:
    clean = make_measure()
    measure = etree.Element("measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"
    key = etree.SubElement(attributes, "key")
    etree.SubElement(key, "fifths").text = "0"
    clef = etree.SubElement(attributes, "clef")
    etree.SubElement(clef, "sign").text = "G"
    etree.SubElement(clef, "line").text = "2"
    note = etree.SubElement(measure, "note")
    pitch = etree.SubElement(note, "pitch")
    etree.SubElement(pitch, "step").text = "C"
    etree.SubElement(pitch, "alter").text = "0"
    etree.SubElement(pitch, "octave").text = "4"
    etree.SubElement(note, "accidental").text = "sharp"
    etree.SubElement(note, "duration").text = "4"
    etree.SubElement(note, "voice").text = "1"
    etree.SubElement(note, "type").text = "whole"
    conflict, _ = measure_from_xml(measure)
    calibrator = MeasureCalibrator()
    assert calibrator.predict_probability(item(clean)) > calibrator.predict_probability(item(conflict))


def test_boundary_context_can_rescue_a_partial_measure() -> None:
    partial = item(make_measure(duration=1, note_type="quarter"))
    calibrator = MeasureCalibrator()
    interior = calibrator.predict_probability(partial)
    boundary = max(
        calibrator.predict_probability(replace(partial, is_first_measure=True)),
        calibrator.predict_probability(replace(partial, is_last_measure=True)),
    )
    assert boundary > interior


def test_corrupt_measure_forest_uses_neutral_fallback(tmp_path: Path) -> None:
    path = tmp_path / "measure_calibrator.json"
    path.write_text(
        json.dumps(
            {
                "model_version": "broken",
                "model_type": "random_forest",
                "feature_names": list(FEATURE_NAMES),
                "trees": [{"nodes": [{"feature": 0, "threshold": 0.5, "left": 0, "right": 0, "value": 0.0}]}],
                "calibration_intercept": 0.0,
                "calibration_slope": 1.0,
            }
        ),
        encoding="utf-8",
    )
    calibrator = MeasureCalibrator(path)
    assert not calibrator.enabled
    assert calibrator.predict_probability(item(make_measure())) == 0.5


def test_corrupt_measure_forest_falls_back_to_embedded_v2(tmp_path: Path) -> None:
    baseline_path = (
        Path(__file__).resolve().parents[2]
        / "training"
        / "baselines"
        / "measure_calibrator_v2.json"
    )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    payload = {
        "model_version": "broken-forest-with-v2",
        "model_type": "random_forest",
        "feature_names": list(FEATURE_NAMES),
        "trees": [
            {
                "nodes": [
                    {"feature": 0, "threshold": 0.5, "left": 0, "right": 0, "value": 0.5}
                ]
            }
        ],
        "calibration_intercept": 0.0,
        "calibration_slope": 1.0,
        "legacy_model": baseline,
        "legacy_preservation_floor": 0.55,
    }
    path = tmp_path / "measure_hybrid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    calibrator = MeasureCalibrator(path)
    assert calibrator.enabled
    assert not calibrator.forest.enabled
    assert calibrator.legacy.enabled
    assert not calibrator.hybrid_enabled
    assert calibrator.model_version == "scorescan-measure-calibrator-2"
    assert 0.0 <= calibrator.predict_probability(item(make_measure())) <= 1.0
