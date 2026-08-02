from dataclasses import dataclass

from scorescan.candidate_calibration import CandidateCalibrator, FEATURE_NAMES, feature_vector


@dataclass(frozen=True)
class Candidate:
    raw_score: float = 1000.0
    valid: bool = True
    agreement_ratio: float = 0.95
    validation_errors: tuple[str, ...] = ()
    measure_gap: int | None = 0
    measure_count: int = 10
    note_count: int = 80
    rhythm_issue_count: int = 0
    tie_issue_count: int = 0
    slur_issue_count: int = 0
    semantic_issue_count: int = 0
    type_duration_mismatch_count: int = 0
    multiple_voice_measure_count: int = 0
    duplicate_direction_count: int = 0
    empty_measure_count: int = 0
    zero_duration_count: int = 0
    chord_duration_mismatch_count: int = 0
    pitch_outlier_count: int = 0
    density_outlier_count: int = 0
    error: str | None = None


def test_candidate_calibrator_resource_is_well_formed() -> None:
    calibrator = CandidateCalibrator()
    assert calibrator.enabled
    assert len(feature_vector(Candidate())) == len(FEATURE_NAMES)


def test_candidate_calibrator_penalizes_structural_failure() -> None:
    calibrator = CandidateCalibrator()
    good = calibrator.predict_probability(Candidate())
    bad = calibrator.predict_probability(
        Candidate(
            raw_score=300,
            valid=False,
            agreement_ratio=0.10,
            validation_errors=("bad",),
            rhythm_issue_count=4,
            semantic_issue_count=8,
            zero_duration_count=2,
            error="engine",
        )
    )
    assert good > bad


def test_embedded_standardized_logistic_rejects_non_finite_parameters() -> None:
    from scorescan.linear_model import StandardizedLogisticModel

    valid = {
        "model_version": "embedded-test",
        "feature_names": ["x"],
        "intercept": 0.0,
        "coefficients": [1.0],
        "means": [0.0],
        "scales": [1.0],
    }
    model = StandardizedLogisticModel.from_payload(valid, ("x",))
    assert model.enabled
    assert model.predict((1.0,)) > 0.5

    invalid = dict(valid)
    invalid["scales"] = [0.0]
    disabled = StandardizedLogisticModel.from_payload(invalid, ("x",))
    assert not disabled.enabled
    assert disabled.predict((1.0,)) == 0.5
