from __future__ import annotations

import base64
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import MethodType

import cv2
import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from tie_visual_training_data import build_tie_visual_dataset  # noqa: E402
from scorescan.score_ir import MeasureIR, NoteIR, PitchIR  # noqa: E402
from scorescan.tie_visual_guard import (  # noqa: E402
    TIE_VISUAL_FEATURE_NAMES,
    TieVisualCalibration,
    TieVisualGuard,
    tie_visual_features,
)
from scorescan.visual_evidence import (  # noqa: E402
    SYMBOL_GUARD_HEIGHT,
    SYMBOL_GUARD_WIDTH,
    VisualMeasureEvidence,
)


def _note(onset: int, ties: tuple[str, ...] = (), slurs=()) -> NoteIR:
    return NoteIR(
        onset=Fraction(onset, 1),
        duration=Fraction(1, 1),
        voice="1",
        pitch=PitchIR("C", Fraction(0, 1), 4),
        rest=False,
        chord=False,
        grace=False,
        note_type="quarter",
        dots=0,
        accidental="",
        ties=ties,
        slurs=tuple(slurs),
        articulations=(),
        ornaments=(),
        tuple_ratio=None,
    )


def _measure(*, tied: bool = False, slur: bool = False) -> MeasureIR:
    first = _note(
        0,
        ("start",) if tied else (),
        (("start", "1"),) if slur else (),
    )
    second = _note(
        1,
        ("stop",) if tied else (),
        (("stop", "1"),) if slur else (),
    )
    return MeasureIR(
        divisions=1,
        time_signature=(2, 4),
        key_signature=(0, "major"),
        clef=("G", 2, 0),
        notes=(first, second),
        directions=(),
        barlines=(),
    )


def _encoded_image() -> str:
    image = np.zeros((SYMBOL_GUARD_HEIGHT, SYMBOL_GUARD_WIDTH), dtype=np.uint8)
    cv2.ellipse(image, (128, 58), (40, 12), 0, 200, 340, 255, 2, cv2.LINE_AA)
    ok, payload = cv2.imencode(".png", image)
    assert ok
    return base64.b64encode(payload.tobytes()).decode("ascii")


def _evidence(encoded: str) -> VisualMeasureEvidence:
    return VisualMeasureEvidence(
        page_index=0,
        system_index=0,
        measure_index=0,
        bbox=(0, 0, SYMBOL_GUARD_WIDTH, SYMBOL_GUARD_HEIGHT),
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
        symbol_guard_image=encoded,
    )


def _force_calibration(guard: TieVisualGuard, accepted: bool, probability: float) -> None:
    def calibrate(self, _evidence, _measure, _start, _stop):
        return TieVisualCalibration(
            probability=probability,
            confidence=probability,
            threshold=self.threshold,
            accepted=accepted,
            available=True,
            model_version=self.model_version,
        )

    guard.calibrate = MethodType(calibrate, guard)  # type: ignore[method-assign]


def test_bundled_tie_visual_guard_is_verified_and_selective() -> None:
    guard = TieVisualGuard()
    assert guard.enabled
    assert guard.model_verified
    assert guard.model_version == "scorescan-tie-visual-forest-1"
    assert guard.threshold == 0.92

    dataset = build_tie_visual_dataset(seed=20261607, groups=30, workers=1)
    probabilities = np.asarray(
        [guard.model.predict(row) for row in dataset.features], dtype=np.float64
    )
    assert dataset.features.shape == (60, len(TIE_VISUAL_FEATURE_NAMES))
    assert float(np.mean(probabilities[dataset.labels == 1])) > float(
        np.mean(probabilities[dataset.labels == 0])
    )


def test_tie_visual_features_reject_malformed_and_oversized_evidence() -> None:
    measure = _measure()
    malformed = _evidence(base64.b64encode(b"not a png").decode("ascii"))
    oversized = _evidence("A" * 65_537)
    assert tie_visual_features(malformed, measure, 0, 1) is None
    assert tie_visual_features(oversized, measure, 0, 1) is None


def test_missing_manifest_is_neutral_and_cannot_authorize_tie(tmp_path: Path) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "scorescan"
        / "resources"
        / "tie_visual_guard.json"
    )
    copied = tmp_path / source.name
    copied.write_bytes(source.read_bytes())
    guard = TieVisualGuard(copied)
    decision = guard.calibrate(_evidence(_encoded_image()), _measure(tied=True), 0, 1)
    assert not guard.model_verified
    assert decision.probability == 0.5
    assert not decision.accepted


def test_no_source_evidence_preserves_existing_high_level_semantic_path() -> None:
    guard = TieVisualGuard()
    audit = guard.audit_transaction(None, _measure(), _measure(tied=True))
    assert not audit.applicable
    assert audit.accepted
    assert audit.reason == "source_evidence_unavailable"


def test_source_backed_tie_addition_is_veto_only() -> None:
    evidence = _evidence(_encoded_image())
    guard = TieVisualGuard()
    _force_calibration(guard, True, 0.99)
    accepted = guard.audit_transaction(evidence, _measure(), _measure(tied=True))
    assert accepted.applicable
    assert accepted.accepted
    assert accepted.reason == "visual_tie_confirmed"

    _force_calibration(guard, False, 0.10)
    rejected = guard.audit_transaction(evidence, _measure(), _measure(tied=True))
    assert rejected.applicable
    assert not rejected.accepted
    assert rejected.reason == "visual_tie_conflict"


def test_source_backed_removal_and_tie_slur_type_ambiguity_are_review_only() -> None:
    evidence = _evidence(_encoded_image())
    guard = TieVisualGuard()
    removal = guard.audit_transaction(evidence, _measure(tied=True), _measure())
    assert removal.applicable
    assert not removal.accepted
    assert removal.reason == "tie_removal_requires_review"

    ambiguous = guard.audit_transaction(evidence, _measure(slur=True), _measure(tied=True, slur=True))
    assert ambiguous.applicable
    assert not ambiguous.accepted
    assert ambiguous.reason == "tie_slur_type_ambiguous"
