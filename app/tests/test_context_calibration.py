from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from scorescan.context_calibration import (
    ContextCalibrator,
    agreement_profiles,
    context_profile,
)
from scorescan.score_ir import MeasureIR, NoteIR, PitchIR


def measure(step: str, octave: int = 4, ties: tuple[str, ...] = ()) -> MeasureIR:
    note = NoteIR(
        onset=Fraction(0),
        duration=Fraction(4),
        voice="1",
        pitch=PitchIR(step, Fraction(0), octave),
        rest=False,
        chord=False,
        grace=False,
        note_type="whole",
        dots=0,
        accidental="",
        ties=ties,
        slurs=(),
        articulations=(),
        ornaments=(),
        tuple_ratio=None,
    )
    return MeasureIR(1, (4, 4), (0, "major"), ("G", 2, 0), (note,), (), ())


def test_context_profile_detects_large_boundary_jump() -> None:
    previous = measure("C", 4)
    coherent = measure("D", 4)
    distant = measure("D", 7)
    following = measure("E", 4)
    good = context_profile(previous, coherent, following)
    bad = context_profile(previous, distant, following)
    assert good.previous_boundary_interval < bad.previous_boundary_interval
    assert good.next_boundary_interval < bad.next_boundary_interval


def test_context_profile_tracks_tie_continuity() -> None:
    previous = measure("C", 4, ("start",))
    current = measure("C", 4, ("stop", "start"))
    following = measure("C", 4, ("stop",))
    profile = context_profile(previous, current, following)
    assert profile.matched_tie_stop_ratio == 1.0
    assert profile.matched_tie_start_ratio == 1.0
    assert profile.orphan_tie_stop_ratio == 0.0
    assert profile.orphan_tie_start_ratio == 0.0


def test_family_profiles_do_not_double_count_sibling_variants() -> None:
    reference = (measure("C"), measure("D"), measure("E"))
    shifted = (measure("C", 5), measure("D", 5), measure("E", 5))
    compact = agreement_profiles(
        (reference[0], shifted[0], reference[0], reference[0]),
        (reference[1], shifted[1], reference[1], reference[1]),
        (reference[2], shifted[2], reference[2], reference[2]),
        ("baseline", "restoration", "binary", "scale"),
    )[0]
    duplicated = agreement_profiles(
        (reference[0], shifted[0], shifted[0], reference[0], reference[0]),
        (reference[1], shifted[1], shifted[1], reference[1], reference[1]),
        (reference[2], shifted[2], shifted[2], reference[2], reference[2]),
        ("baseline", "restoration", "restoration", "binary", "scale"),
    )[0]
    assert duplicated.independent_median == compact.independent_median
    assert duplicated.independent_family_count_scaled == compact.independent_family_count_scaled


def test_family_profiles_require_aligned_inputs() -> None:
    with pytest.raises(ValueError):
        agreement_profiles(
            (measure("C"),),
            (measure("D"), measure("D")),
            (measure("E"),),
            ("baseline",),
        )


def test_context_calibrator_loads_verified_resource() -> None:
    calibrator = ContextCalibrator()
    assert calibrator.enabled
    assert calibrator.model_verified
    assert calibrator.model_version == "scorescan-context-forest-2"
    result = calibrator.calibrate(measure("C"), measure("D"), measure("E"))
    assert 0.0 <= result.probability <= 1.0
    assert 0.90 <= result.weight_factor <= 1.10


def test_family_context_penalises_one_correlated_octave_sequence() -> None:
    reference = (measure("C"), measure("D"), measure("E"))
    shifted = (measure("C", 5), measure("D", 5), measure("E", 5))
    families = (
        "baseline",
        "restoration",
        "restoration",
        "binary",
        "binary",
        "scale",
        "scale",
    )
    profiles = agreement_profiles(
        (
            reference[0],
            shifted[0],
            shifted[0],
            reference[0],
            reference[0],
            reference[0],
            reference[0],
        ),
        (
            reference[1],
            shifted[1],
            shifted[1],
            reference[1],
            reference[1],
            reference[1],
            reference[1],
        ),
        (
            reference[2],
            shifted[2],
            shifted[2],
            reference[2],
            reference[2],
            reference[2],
            reference[2],
        ),
        families,
    )
    calibrator = ContextCalibrator()
    probabilities = [
        calibrator.calibrate_profile(profile).probability
        for profile in profiles
    ]
    assert min(probabilities[index] for index in (0, 3, 4, 5, 6)) > max(
        probabilities[index] for index in (1, 2)
    )


def test_context_model_penalises_orphan_tie_and_octave_jump() -> None:
    calibrator = ContextCalibrator()
    previous = measure("C", 4)
    following = measure("E", 4)
    good = calibrator.calibrate(previous, measure("D", 4), following)
    bad_measure = replace(
        measure("D", 7),
        notes=(replace(measure("D", 7).notes[0], ties=("stop", "start")),),
    )
    bad = calibrator.calibrate(previous, bad_measure, following)
    assert good.probability > bad.probability


def test_context_model_disables_malformed_forest(tmp_path) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "scorescan"
        / "resources"
        / "context_calibrator.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["trees"][0]["nodes"][0]["left"] = 0
    damaged = tmp_path / "context_calibrator.json"
    damaged.write_text(json.dumps(payload), encoding="utf-8")
    calibrator = ContextCalibrator(damaged)
    assert not calibrator.enabled
    result = calibrator.calibrate(measure("C"), measure("D"), measure("E"))
    assert result.probability == 0.5
    assert result.weight_factor == 1.0
