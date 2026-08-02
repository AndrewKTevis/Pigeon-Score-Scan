from __future__ import annotations

import base64
import sys
from fractions import Fraction
from pathlib import Path
from types import MethodType

import cv2
import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from event_kind_visual_training_data import build_event_kind_visual_dataset  # noqa: E402
from scorescan.event_kind_visual_guard import (  # noqa: E402
    EVENT_KIND_VISUAL_FEATURE_NAMES,
    EventKindVisualCalibration,
    EventKindVisualGuard,
    event_kind_visual_features,
)
from scorescan.score_ir import MeasureIR, NoteIR, PitchIR  # noqa: E402
from scorescan.visual_evidence import (  # noqa: E402
    SYMBOL_GUARD_HEIGHT,
    SYMBOL_GUARD_WIDTH,
    VisualMeasureEvidence,
)


def _note(rest: bool, *, onset: Fraction = Fraction(2, 1), note_type: str = "quarter") -> NoteIR:
    return NoteIR(
        onset=onset,
        duration=Fraction(1, 1),
        voice="1",
        pitch=None if rest else PitchIR("C", Fraction(0, 1), 4),
        rest=rest,
        chord=False,
        grace=False,
        note_type=note_type,
        dots=0,
        accidental="",
        ties=(),
        slurs=(),
        articulations=(),
        ornaments=(),
        tuple_ratio=None,
    )


def _measure(rest: bool, *, count: int = 1) -> MeasureIR:
    notes = tuple(_note(rest, onset=Fraction(index + 1, 1)) for index in range(count))
    return MeasureIR(
        divisions=4,
        time_signature=(4, 4),
        key_signature=(0, "major"),
        clef=("G", 2, 0),
        notes=notes,
        directions=(),
        barlines=(),
    )


def _encoded_image() -> str:
    image = np.zeros((SYMBOL_GUARD_HEIGHT, SYMBOL_GUARD_WIDTH), dtype=np.uint8)
    cv2.ellipse(image, (128, 58), (6, 4), -12, 0, 360, 255, -1, cv2.LINE_AA)
    cv2.line(image, (133, 58), (133, 34), 255, 2, cv2.LINE_AA)
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


def _force_calibration(guard: EventKindVisualGuard, accepted: bool, probability: float) -> None:
    def calibrate(self, _evidence, _before, _after, _event_index):
        return EventKindVisualCalibration(
            probability=probability,
            threshold=self.threshold,
            accepted=accepted,
            available=True,
            model_version=self.model_version,
        )

    guard.calibrate = MethodType(calibrate, guard)  # type: ignore[method-assign]


def test_bundled_event_kind_visual_guard_is_verified_and_selective() -> None:
    guard = EventKindVisualGuard()
    assert guard.enabled
    assert guard.model_verified
    assert guard.model_version == "scorescan-event-kind-visual-forest-1"
    assert 0.90 < guard.threshold < 0.98

    dataset = build_event_kind_visual_dataset(seed=20261101, groups=24)
    probabilities = np.asarray(
        [guard.model.predict(row) for row in dataset.features], dtype=np.float64
    )
    assert dataset.features.shape == (48, len(EVENT_KIND_VISUAL_FEATURE_NAMES))
    assert float(np.mean(probabilities[dataset.labels == 1])) > float(
        np.mean(probabilities[dataset.labels == 0])
    )


def test_event_kind_features_reject_malformed_and_oversized_evidence() -> None:
    malformed = _evidence(base64.b64encode(b"not a png").decode("ascii"))
    oversized = _evidence("A" * 65_537)
    assert event_kind_visual_features(malformed, _measure(True), _measure(False), 0) is None
    assert event_kind_visual_features(oversized, _measure(True), _measure(False), 0) is None


def test_missing_manifest_is_neutral_and_cannot_authorize_event_kind(tmp_path: Path) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "scorescan"
        / "resources"
        / "event_kind_visual_guard.json"
    )
    copied = tmp_path / source.name
    copied.write_bytes(source.read_bytes())
    guard = EventKindVisualGuard(copied)
    decision = guard.calibrate(_evidence(_encoded_image()), _measure(True), _measure(False), 0)
    assert not guard.model_verified
    assert decision.probability == 0.5
    assert not decision.accepted
    assert not decision.available


def test_no_source_evidence_preserves_existing_high_level_path() -> None:
    guard = EventKindVisualGuard()
    audit = guard.audit_transaction(None, _measure(True), _measure(False))
    assert not audit.applicable
    assert audit.accepted
    assert audit.reason == "source_evidence_unavailable"


def test_single_event_kind_transaction_is_veto_only() -> None:
    evidence = _evidence(_encoded_image())
    guard = EventKindVisualGuard()
    _force_calibration(guard, True, 0.99)
    accepted = guard.audit_transaction(evidence, _measure(True), _measure(False))
    assert accepted.applicable
    assert accepted.accepted
    assert accepted.reason == "visual_event_kind_confirmed"

    _force_calibration(guard, False, 0.10)
    rejected = guard.audit_transaction(evidence, _measure(True), _measure(False))
    assert rejected.applicable
    assert not rejected.accepted
    assert rejected.reason == "visual_event_kind_conflict"


def test_multiple_event_kind_changes_require_review_when_source_exists() -> None:
    guard = EventKindVisualGuard()
    audit = guard.audit_transaction(
        _evidence(_encoded_image()), _measure(True, count=2), _measure(False, count=2)
    )
    assert audit.applicable
    assert not audit.accepted
    assert audit.changed_event_count == 2
    assert audit.reason == "multiple_event_kind_changes_require_review"
