from dataclasses import replace
from fractions import Fraction

import scorescan.event_calibration as event_calibration
from scorescan.event_calibration import EventCalibrator, agreement_profiles, align_note_sequences
from scorescan.score_ir import MeasureIR, NoteIR, PitchIR


def note(step: str, onset: Fraction, duration: Fraction) -> NoteIR:
    return NoteIR(
        onset=onset,
        duration=duration,
        voice="1",
        pitch=PitchIR(step, Fraction(0), 4),
        rest=False,
        chord=False,
        grace=False,
        note_type="quarter" if duration == 1 else "eighth",
        dots=0,
        accidental="",
        ties=(),
        slurs=(),
        articulations=(),
        ornaments=(),
        tuple_ratio=None,
    )


def measure(notes: tuple[NoteIR, ...]) -> MeasureIR:
    return MeasureIR(24, (4, 4), (0, "major"), ("G", 2, 0), notes, (), ())


def test_note_alignment_recovers_middle_insertion() -> None:
    left = (note("C", Fraction(0), Fraction(1)), note("D", Fraction(1), Fraction(1)), note("E", Fraction(2), Fraction(1)))
    right = (left[0], replace(left[1], pitch=PitchIR("C", Fraction(1), 4), onset=Fraction(1, 2), duration=Fraction(1, 2)), left[1], left[2])
    alignment = align_note_sequences(left, right)
    assert alignment.unmatched_right == 1
    assert alignment.unmatched_left == 0
    assert alignment.similarity > 0.65


def test_event_profiles_reward_supported_clean_candidate() -> None:
    clean = measure((note("C", Fraction(0), Fraction(1)), note("D", Fraction(1), Fraction(1)), note("E", Fraction(2), Fraction(1)), note("F", Fraction(3), Fraction(1))))
    clean_equivalent = replace(clean, divisions=48)
    wrong_duration = replace(clean, notes=(replace(clean.notes[0], duration=Fraction(2)), *clean.notes[1:]))
    wrong_voice = replace(clean, notes=(replace(clean.notes[0], voice="2"), *clean.notes[1:]))
    missing = replace(clean, notes=clean.notes[:-1])
    profiles = agreement_profiles([clean, clean_equivalent, wrong_duration, wrong_voice, missing])
    calibrator = EventCalibrator()
    assert calibrator.enabled
    clean_probability = calibrator.predict_probability(profiles[0])
    corrupted_probability = calibrator.predict_probability(profiles[2])
    assert clean_probability > corrupted_probability
    assert 0.88 <= calibrator.calibrate(profiles[0]).weight_factor <= 1.12


def test_family_balancing_prevents_related_variants_from_manufacturing_support() -> None:
    clean = measure(
        (
            note("C", Fraction(0), Fraction(1)),
            note("D", Fraction(1), Fraction(1)),
            note("E", Fraction(2), Fraction(1)),
            note("F", Fraction(3), Fraction(1)),
        )
    )
    wrong = replace(clean, notes=(replace(clean.notes[1], pitch=PitchIR("A", Fraction(0), 5)), *clean.notes[2:]))
    candidates = [
        clean,
        wrong,
        wrong,
        replace(clean, divisions=48),
        replace(clean, divisions=96),
        replace(clean, divisions=72),
        replace(clean, divisions=120),
    ]
    families = ["baseline", "restoration", "restoration", "binary", "binary", "scale", "scale"]
    profiles = agreement_profiles(candidates, families)
    calibrator = EventCalibrator()
    assert calibrator.enabled
    assert profiles[0].independent_pitch_support > profiles[1].independent_pitch_support
    assert profiles[1].same_family_alignment_similarity > profiles[0].same_family_alignment_similarity
    assert calibrator.predict_probability(profiles[0]) > calibrator.predict_probability(profiles[1])


def test_family_profiles_are_permutation_stable() -> None:
    clean = measure((note("C", Fraction(0), Fraction(1)), note("D", Fraction(1), Fraction(1))))
    wrong = replace(clean, notes=(replace(clean.notes[0], duration=Fraction(2)), clean.notes[1]))
    measures = [clean, wrong, replace(clean, divisions=48), wrong]
    families = ["baseline", "restoration", "binary", "restoration"]
    original = agreement_profiles(measures, families)
    order = [2, 0, 3, 1]
    permuted = agreement_profiles([measures[index] for index in order], [families[index] for index in order])
    reverse = {old_index: new_index for new_index, old_index in enumerate(order)}
    for old_index, profile in enumerate(original):
        assert profile.feature_vector() == permuted[reverse[old_index]].feature_vector()


def test_invalid_event_forest_uses_neutral_probability(tmp_path) -> None:
    model_path = tmp_path / "event_calibrator.json"
    model_path.write_text(
        '{"model_version":"broken","model_type":"random_forest",'
        '"feature_names":[],"trees":[{"nodes":[{"feature":999,"left":0,"right":0,"value":0}]}],'
        '"calibration_intercept":0,"calibration_slope":1}',
        encoding="utf-8",
    )
    calibrator = EventCalibrator(model_path)
    clean = measure((note("C", Fraction(0), Fraction(1)),))
    profile = agreement_profiles([clean, replace(clean, divisions=48)])[0]
    assert not calibrator.enabled
    assert calibrator.predict_probability(profile) == 0.5


def test_pairwise_alignment_is_cached_once_per_unordered_pair(monkeypatch) -> None:
    clean = measure((note("C", Fraction(0), Fraction(1)), note("D", Fraction(1), Fraction(1))))
    candidates = [clean, replace(clean, divisions=48), replace(clean, divisions=72), replace(clean, divisions=96)]
    calls = 0
    original = event_calibration.align_note_sequences

    def counted(left, right):
        nonlocal calls
        calls += 1
        return original(left, right)

    monkeypatch.setattr(event_calibration, "align_note_sequences", counted)
    profiles = agreement_profiles(candidates, ["baseline", "restoration", "binary", "scale"])
    assert len(profiles) == 4
    assert calls == 6


def test_family_count_mismatch_is_rejected() -> None:
    clean = measure((note("C", Fraction(0), Fraction(1)),))
    try:
        agreement_profiles([clean, clean], ["baseline"])
    except ValueError as error:
        assert "family count" in str(error)
    else:
        raise AssertionError("expected family length validation")
