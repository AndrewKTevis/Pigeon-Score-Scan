from __future__ import annotations

import base64
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from scorescan.rhythm_symbol_guard import (
    RHYTHM_SYMBOL_FEATURE_NAMES,
    RhythmSymbolComparisonInput,
    RhythmSymbolGuard,
    RhythmSymbolObservation,
    RhythmSymbolTransactionInput,
    build_rhythm_symbol_transaction,
)
from scorescan.score_ir import MeasureIR, NoteIR, PitchIR
from scorescan.visual_evidence import VisualMeasureEvidence


def _observation(note_type: str, duration: Fraction, *, ink: float) -> RhythmSymbolObservation:
    descriptor = tuple([ink] * (8 * 16) + [1.0 - ink] * (8 * 16))
    return RhythmSymbolObservation(
        descriptor=descriptor,
        note_type=note_type,
        dots=0,
        rest=False,
        beam_level=1 if note_type == "eighth" else 0,
        open_notehead=note_type in {"whole", "half"},
        has_stem=note_type != "whole",
        duration=duration,
        onset_ratio=0.25,
        staff_y_ratio=0.5,
    )


def _measure() -> MeasureIR:
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
            pitch=PitchIR("D", Fraction(0, 1), 4),
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
    return MeasureIR(1, (2, 4), (0, "major"), ("G", 2, 0), notes, (), ())


def _evidence(encoded: str) -> VisualMeasureEvidence:
    return VisualMeasureEvidence(
        page_index=1,
        system_index=1,
        measure_index=1,
        bbox=(0, 0, 128, 64),
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
        rhythm_guard_image=encoded,
    )


def test_transaction_features_are_bounded_and_reverse_symmetric() -> None:
    proposed = _observation("eighth", Fraction(1, 2), ink=0.75)
    template = _observation("quarter", Fraction(1, 1), ink=0.25)
    comparison = RhythmSymbolComparisonInput(proposed, template)
    transaction = RhythmSymbolTransactionInput((comparison, comparison))

    features = transaction.feature_vector()
    reversed_features = transaction.reversed().feature_vector()
    assert len(features) == len(RHYTHM_SYMBOL_FEATURE_NAMES) == 468
    assert all(-1.0 <= value <= 1.0 for value in features)
    assert features != reversed_features
    assert transaction.reversed().reversed() == transaction


def test_guard_rejects_invalid_or_oversized_persisted_image() -> None:
    measure = _measure()
    malformed = _evidence(base64.b64encode(b"not a png").decode("ascii"))
    oversized = _evidence("A" * 65_537)
    assert build_rhythm_symbol_transaction(malformed, measure, measure, (0,)) is None
    assert build_rhythm_symbol_transaction(oversized, measure, measure, (0,)) is None


def test_bundled_rhythm_symbol_guard_is_verified_and_fail_closed(tmp_path: Path) -> None:
    guard = RhythmSymbolGuard()
    assert guard.enabled
    assert guard.model_verified
    assert guard.model_version == "scorescan-rhythm-symbol-forest-1"
    assert guard.threshold == 0.9875

    comparison = RhythmSymbolComparisonInput(
        _observation("eighth", Fraction(1, 2), ink=0.75),
        _observation("quarter", Fraction(1, 1), ink=0.25),
    )
    transaction = RhythmSymbolTransactionInput((comparison,))

    # A copied model without its signed manifest is deliberately unusable.  Loading
    # failure must remain neutral and cannot authorize a MusicXML modification.
    model_path = tmp_path / "rhythm_symbol_guard.json"
    source = Path(__file__).resolve().parents[1] / "src" / "scorescan" / "resources" / model_path.name
    model_path.write_bytes(source.read_bytes())
    unverified = RhythmSymbolGuard(model_path)
    decision = unverified.calibrate(transaction)
    assert not unverified.model_verified
    assert not decision.accepted
    assert decision.confidence <= 0.5
