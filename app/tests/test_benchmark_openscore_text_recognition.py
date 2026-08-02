from __future__ import annotations

import pytest

from app.tools.benchmark_openscore_text_recognition import (
    edit_distance,
    normalized_text,
    select_recognizer,
    stable_subset,
    summarize_predictions,
    text_family,
)


def test_text_normalization_and_edit_distance() -> None:
    assert normalized_text(" Allegretto  molto ") == "allegretto molto"
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3


def test_text_family_separates_measure_numbers_and_words() -> None:
    assert text_family("116") == "numeric"
    assert text_family("cresc.") == "music_word"
    assert text_family("K.428/421b") == "metadata_mixed"


def test_prediction_summary_uses_micro_character_error_rate() -> None:
    summary = summarize_predictions(
        [
            {"truth": "abc", "prediction": "axc", "confidence": 0.8},
            {"truth": "d", "prediction": "d", "confidence": 1.0},
        ]
    )
    assert summary["samples"] == 2
    assert summary["exact_match"] == 0.5
    assert summary["character_error_rate"] == 0.25
    assert summary["mean_confidence"] == 0.9


def test_stable_subset_is_input_order_independent() -> None:
    rows = [
        {
            "source_key": "piece",
            "page": 1,
            "word_index": index,
            "text": str(index),
        }
        for index in range(20)
    ]
    assert stable_subset(rows, 5, 7) == stable_subset(list(reversed(rows)), 5, 7)


def test_selection_accepts_real_hard_scan_gain_but_requires_speed_audit() -> None:
    def stats(exact: float, cer: float) -> dict[str, dict[str, float]]:
        row = {"exact_match": exact, "character_error_rate": cer}
        return {
            "overall": row,
            "numeric": row,
            "music_word": row,
            "metadata_mixed": row,
        }

    result = select_recognizer(
        small={
            "clean": stats(1.0, 0.0),
            "scan_hard": stats(0.993, 0.002),
        },
        medium={
            "clean": stats(1.0, 0.0),
            "scan_hard": stats(1.0, 0.0),
        },
        profiles=["clean", "scan_hard"],
        small_samples_per_second=90,
        medium_samples_per_second=4,
    )
    assert result["accuracy_selected"] == "ppocrv6_medium"
    assert result["hard_exact_match_gain"] == pytest.approx(0.007)
    assert result["requires_accelerated_runtime_benchmark"] is True
