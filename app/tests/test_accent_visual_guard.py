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

from accent_visual_training_data import build_accent_visual_dataset  # noqa: E402
from scorescan.accent_visual_guard import (  # noqa: E402
    ACCENT_VISUAL_FEATURE_NAMES,
    AccentVisualCalibration,
    AccentVisualGuard,
    accent_visual_features,
)
from scorescan.score_ir import MeasureIR, NoteIR, PitchIR  # noqa: E402
from scorescan.visual_evidence import (  # noqa: E402
    SYMBOL_GUARD_HEIGHT,
    SYMBOL_GUARD_WIDTH,
    VisualMeasureEvidence,
)


def _note(articulations: tuple[str, ...] = ()) -> NoteIR:
    return NoteIR(
        onset=Fraction(0, 1),
        duration=Fraction(4, 1),
        voice="1",
        pitch=PitchIR("C", Fraction(0, 1), 4),
        rest=False,
        chord=False,
        grace=False,
        note_type="whole",
        dots=0,
        accidental="",
        ties=(),
        slurs=(),
        articulations=articulations,
        ornaments=(),
        tuple_ratio=None,
    )


def _measure(articulations: tuple[str, ...] = ()) -> MeasureIR:
    return MeasureIR(
        divisions=1,
        time_signature=(4, 4),
        key_signature=(0, "major"),
        clef=("G", 2, 0),
        notes=(_note(articulations),),
        directions=(),
        barlines=(),
    )


def _encoded_image() -> str:
    image = np.zeros((SYMBOL_GUARD_HEIGHT, SYMBOL_GUARD_WIDTH), dtype=np.uint8)
    cv2.line(image, (118, 25), (128, 29), 255, 2, cv2.LINE_AA)
    cv2.line(image, (128, 29), (138, 25), 255, 2, cv2.LINE_AA)
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


def _force_calibration(guard: AccentVisualGuard, accepted: bool, probability: float) -> None:
    def calibrate(self, _evidence, _measure, _event_index):
        return AccentVisualCalibration(
            probability=probability,
            confidence=probability,
            threshold=self.threshold,
            accepted=accepted,
            available=True,
            model_version=self.model_version,
        )

    guard.calibrate = MethodType(calibrate, guard)  # type: ignore[method-assign]


def test_bundled_accent_visual_guard_is_verified_and_selective() -> None:
    guard = AccentVisualGuard()
    assert guard.enabled
    assert guard.model_verified
    assert guard.model_version == "scorescan-accent-visual-forest-1"
    assert 0.70 <= guard.threshold < 0.80

    dataset = build_accent_visual_dataset(seed=20261409, groups=24, workers=1)
    probabilities = np.asarray(
        [guard.model.predict(row) for row in dataset.features], dtype=np.float64
    )
    assert dataset.features.shape == (24 * 8, len(ACCENT_VISUAL_FEATURE_NAMES))
    assert float(np.mean(probabilities[dataset.labels == 1])) > float(
        np.mean(probabilities[dataset.labels == 0])
    )


def test_accent_features_reject_malformed_and_oversized_evidence() -> None:
    malformed = _evidence(base64.b64encode(b"not a png").decode("ascii"))
    oversized = _evidence("A" * 65_537)
    assert accent_visual_features(malformed, _measure(), 0) is None
    assert accent_visual_features(oversized, _measure(), 0) is None


def test_missing_manifest_is_neutral_and_cannot_authorize_accent(tmp_path: Path) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "scorescan"
        / "resources"
        / "accent_visual_guard.json"
    )
    copied = tmp_path / source.name
    copied.write_bytes(source.read_bytes())
    guard = AccentVisualGuard(copied)
    decision = guard.calibrate(_evidence(_encoded_image()), _measure(("accent",)), 0)
    assert not guard.model_verified
    assert decision.probability == 0.5
    assert not decision.accepted


def test_no_source_evidence_preserves_existing_high_level_path() -> None:
    guard = AccentVisualGuard()
    audit = guard.audit_transaction(None, _measure(), _measure(("accent",)))
    assert not audit.applicable
    assert audit.accepted
    assert audit.reason == "source_evidence_unavailable"


def test_source_backed_simple_accent_addition_is_veto_only() -> None:
    evidence = _evidence(_encoded_image())
    guard = AccentVisualGuard()
    _force_calibration(guard, True, 0.99)
    accepted = guard.audit_transaction(evidence, _measure(), _measure(("accent",)))
    assert accepted.applicable
    assert accepted.accepted
    assert accepted.reason == "visual_accent_confirmed"

    _force_calibration(guard, False, 0.10)
    rejected = guard.audit_transaction(evidence, _measure(), _measure(("accent",)))
    assert rejected.applicable
    assert not rejected.accepted
    assert rejected.reason == "visual_accent_conflict"


def test_removal_substitution_and_stacked_articulation_are_review_only() -> None:
    evidence = _evidence(_encoded_image())
    guard = AccentVisualGuard()
    removal = guard.audit_transaction(evidence, _measure(("accent",)), _measure())
    assert not removal.applicable
    assert removal.accepted

    substitution = guard.audit_transaction(
        evidence, _measure(("tenuto",)), _measure(("accent",))
    )
    assert substitution.applicable
    assert not substitution.accepted
    assert substitution.reason == "mixed_or_nonempty_articulation_transaction_requires_review"

    stacked = guard.audit_transaction(
        evidence, _measure(), _measure(("accent", "staccato"))
    )
    assert stacked.applicable
    assert not stacked.accepted
    assert stacked.reason == "mixed_or_nonempty_articulation_transaction_requires_review"
