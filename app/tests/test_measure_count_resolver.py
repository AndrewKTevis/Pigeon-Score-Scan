from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from scorescan.measure_count_resolver import MeasureCountResolver, measure_count_model_gate
from scorescan.recognition import RecognitionCandidate, apply_expected_measure_count


@dataclass(frozen=True)
class Candidate:
    variant: str
    measure_count: int
    valid: bool = True
    agreement_ratio: float = 0.90
    calibrated_probability: float = 0.80
    raw_score: float = 980.0
    measure_gap_penalty: float = 0.0


def test_measure_count_resolver_preserves_strong_layout_and_omr_agreement() -> None:
    resolver = MeasureCountResolver()
    result = resolver.resolve(
        layout_count=46,
        layout_confidence=0.98,
        candidates=(
            Candidate("primary", 46),
            Candidate("flat", 46),
            Candidate("otsu", 45, agreement_ratio=0.45, calibrated_probability=0.35, raw_score=840.0),
        ),
    )
    assert result.selected_count == 46
    assert result.probability > 0.5
    assert result.model_version == "scorescan-measure-count-resolver-4"


def test_measure_count_resolver_can_correct_wrong_layout_count() -> None:
    resolver = MeasureCountResolver()
    result = resolver.resolve(
        layout_count=47,
        layout_confidence=0.98,
        candidates=(
            Candidate("primary", 44),
            Candidate("flat", 44),
            Candidate("otsu", 44),
            Candidate("adaptive", 45, agreement_ratio=0.40, calibrated_probability=0.30, raw_score=820.0),
        ),
    )
    assert result.selected_count == 44
    assert result.deterministic_count == 44
    assert result.source in {"model", "deterministic_fallback"}


def test_measure_count_resolver_damaged_model_uses_deterministic_fallback(tmp_path: Path) -> None:
    path = tmp_path / "measure_count_resolver.json"
    path.write_text("{}", encoding="utf-8")
    resolver = MeasureCountResolver(path)
    result = resolver.resolve(
        layout_count=47,
        layout_confidence=0.8,
        candidates=(Candidate("primary", 44), Candidate("flat", 44), Candidate("otsu", 45)),
    )
    assert not resolver.enabled
    assert result.selected_count == 44
    assert result.source == "deterministic_fallback"


def test_rebase_expected_measure_count_only_changes_gap_penalty() -> None:
    candidate = RecognitionCandidate(
        variant="primary",
        image_path="page.png",
        xml_path="page.musicxml",
        score=874.0,
        raw_score=874.0,
        valid=True,
        elapsed_seconds=1.0,
        measure_count=44,
        measure_gap=3,
        expected_measure_count=47,
        measure_gap_penalty=126.0,
    )
    rebased = apply_expected_measure_count(candidate, 44)
    assert rebased.measure_gap == 0
    assert rebased.measure_gap_penalty == 0
    assert rebased.raw_score == 1000.0
    assert rebased.score == 1000.0


def test_measure_count_resolver_equalises_correlated_family_duplicates() -> None:
    resolver = MeasureCountResolver()
    result = resolver.resolve(
        layout_count=45,
        layout_confidence=0.70,
        candidates=(
            Candidate("flat", 45, agreement_ratio=0.95, calibrated_probability=0.92, raw_score=1040.0),
            Candidate("deblock", 45, agreement_ratio=0.93, calibrated_probability=0.90, raw_score=1020.0),
            Candidate("primary", 44, agreement_ratio=0.72, calibrated_probability=0.65, raw_score=920.0),
            Candidate("upscale", 44, agreement_ratio=0.70, calibrated_probability=0.62, raw_score=910.0),
        ),
    )
    # Two independent families are informative but are no longer sufficient to
    # override a deterministic count.  The family-balanced option remains visible in
    # diagnostics and a third family is required for an automatic change.
    assert result.selected_count == 45
    assert result.source == "deterministic_fallback"
    options = {item.count: item for item in result.options}
    assert options[45].candidate_support == 2
    assert options[45].family_support == 1
    assert options[44].family_support == 2
    assert options[44].family_balanced_support_share > options[45].family_balanced_support_share


def test_deterministic_fallback_uses_family_balanced_support(tmp_path: Path) -> None:
    path = tmp_path / "measure_count_resolver.json"
    path.write_text("{}", encoding="utf-8")
    resolver = MeasureCountResolver(path)
    result = resolver.resolve(
        layout_count=45,
        layout_confidence=0.45,
        candidates=(
            Candidate("flat", 45, agreement_ratio=0.92, calibrated_probability=0.90, raw_score=1020.0),
            Candidate("deblock", 45, agreement_ratio=0.90, calibrated_probability=0.88, raw_score=1010.0),
            Candidate("primary", 44, agreement_ratio=0.78, calibrated_probability=0.72, raw_score=950.0),
            Candidate("otsu", 44, agreement_ratio=0.76, calibrated_probability=0.70, raw_score=940.0),
            Candidate("upscale", 44, agreement_ratio=0.74, calibrated_probability=0.68, raw_score=930.0),
        ),
    )
    assert result.source == "deterministic_fallback"
    assert result.selected_count == 44


def test_bundled_measure_count_model_is_verified_v4() -> None:
    resolver = MeasureCountResolver()
    assert resolver.enabled
    assert resolver.model_version == "scorescan-measure-count-resolver-4"
    assert resolver.model_status == "verified"


def test_measure_count_resolver_sanitizes_non_finite_evidence() -> None:
    resolver = MeasureCountResolver()
    result = resolver.resolve(
        layout_count=44,
        layout_confidence=float("nan"),
        candidates=(
            Candidate(
                "primary",
                44,
                agreement_ratio=float("nan"),
                calibrated_probability=float("inf"),
                raw_score=float("-inf"),
                measure_gap_penalty=float("nan"),
            ),
            Candidate("flat", 45),
        ),
    )
    assert result.selected_count in {44, 45}
    assert math.isfinite(result.probability)
    assert math.isfinite(result.margin)
    assert all(math.isfinite(option.probability) for option in result.options)
    assert all(math.isfinite(option.deterministic_score) for option in result.options)


def test_measure_count_resolver_invalid_sibling_makes_family_abstain(tmp_path: Path) -> None:
    path = tmp_path / "measure_count_resolver.json"
    path.write_text("{}", encoding="utf-8")
    result = MeasureCountResolver(path).resolve(
        layout_count=44,
        layout_confidence=0.65,
        candidates=(
            Candidate("flat", 45, valid=True, raw_score=1040.0),
            Candidate("deblock", 45, valid=False, raw_score=1030.0),
            Candidate("primary", 44, raw_score=950.0),
            Candidate("otsu", 44, raw_score=940.0),
            Candidate("upscale", 44, raw_score=930.0),
        ),
    )
    options = {item.count: item for item in result.options}
    assert result.selected_count == 44
    assert options[45].candidate_support == 2
    assert options[45].family_support == 0
    assert options[44].family_support == 3


def test_high_confidence_layout_override_requires_four_families() -> None:
    common = dict(
        count=45,
        probability=0.99,
        margin=0.80,
        candidate_support=3,
        deterministic_count=44,
        layout_count=44,
        layout_confidence=0.98,
    )
    assert not measure_count_model_gate(family_support=3, **common)
    assert measure_count_model_gate(family_support=4, **common)
