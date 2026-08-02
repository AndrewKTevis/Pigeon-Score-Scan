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

from event_presence_visual_training_data import (  # noqa: E402
    build_event_presence_visual_dataset,
)
from scorescan.event_presence_visual_guard import (  # noqa: E402
    EVENT_PRESENCE_VISUAL_FEATURE_NAMES,
    EventPresenceVisualCalibration,
    EventPresenceVisualGuard,
    _single_edit_event,
    event_presence_visual_features,
)
from scorescan.local_symbol_image import event_position  # noqa: E402
from scorescan.score_ir import MeasureIR, NoteIR, PitchIR  # noqa: E402
from scorescan.visual_evidence import (  # noqa: E402
    SYMBOL_GUARD_HEIGHT,
    SYMBOL_GUARD_WIDTH,
    VisualMeasureEvidence,
)


def _note(onset: int, *, rest: bool = False) -> NoteIR:
    return NoteIR(
        onset=Fraction(onset, 1),
        duration=Fraction(1, 1),
        voice="1",
        pitch=None if rest else PitchIR("C", Fraction(0, 1), 4 + onset % 2),
        rest=rest,
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
    )


def _measure(include_target: bool) -> MeasureIR:
    notes = (_note(0), _note(1, rest=True), _note(2), _note(3))
    if not include_target:
        notes = notes[:2] + notes[3:]
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
    measure = _measure(True)
    target = measure.notes[2]
    x_ratio, y_ratio = event_position(measure, target)
    x = int(round(x_ratio * (SYMBOL_GUARD_WIDTH - 1)))
    y = int(round(y_ratio * (SYMBOL_GUARD_HEIGHT - 1)))
    image = np.zeros((SYMBOL_GUARD_HEIGHT, SYMBOL_GUARD_WIDTH), dtype=np.uint8)
    cv2.ellipse(image, (x, y), (6, 4), -12, 0, 360, 255, -1, cv2.LINE_AA)
    cv2.line(image, (x + 5, y), (x + 5, y - 22), 255, 2, cv2.LINE_AA)
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


def _force_calibration(
    guard: EventPresenceVisualGuard, accepted: bool, probability: float
) -> None:
    def calibrate(self, _evidence, _before, _after, _operation, _event_index):
        return EventPresenceVisualCalibration(
            probability=probability,
            threshold=self.threshold,
            accepted=accepted,
            available=True,
            model_version=self.model_version,
        )

    guard.calibrate = MethodType(calibrate, guard)  # type: ignore[method-assign]


def test_event_presence_visual_features_are_bounded_and_candidate_relative() -> None:
    features = event_presence_visual_features(
        _evidence(_encoded_image()), _measure(False), _measure(True), "insert", 2
    )
    assert features is not None
    assert len(features) == len(EVENT_PRESENCE_VISUAL_FEATURE_NAMES)
    assert all(np.isfinite(value) for value in features)


def test_event_presence_features_reject_malformed_and_unsupported_transactions() -> None:
    malformed = _evidence(base64.b64encode(b"not a png").decode("ascii"))
    assert (
        event_presence_visual_features(
            malformed, _measure(False), _measure(True), "insert", 2
        )
        is None
    )
    assert (
        event_presence_visual_features(
            _evidence(_encoded_image()), _measure(False), _measure(True), "insert", 99
        )
        is None
    )


def test_no_source_evidence_preserves_existing_high_level_path() -> None:
    guard = EventPresenceVisualGuard()
    audit = guard.audit_transaction(None, _measure(False), _measure(True), "insert", 2)
    assert not audit.applicable
    assert audit.accepted
    assert audit.reason == "source_evidence_unavailable"


def test_single_insertion_visual_gate_is_veto_only() -> None:
    guard = EventPresenceVisualGuard()
    evidence = _evidence(_encoded_image())
    _force_calibration(guard, True, 0.99)
    accepted = guard.audit_transaction(
        evidence, _measure(False), _measure(True), "insert", 2
    )
    assert accepted.applicable
    assert accepted.accepted
    assert accepted.reason == "visual_event_presence_confirmed"

    _force_calibration(guard, False, 0.1)
    rejected = guard.audit_transaction(
        evidence, _measure(False), _measure(True), "insert", 2
    )
    assert rejected.applicable
    assert not rejected.accepted
    assert rejected.reason == "visual_event_presence_conflict"


def test_event_deletion_with_source_evidence_is_review_only() -> None:
    guard = EventPresenceVisualGuard()
    audit = guard.audit_transaction(
        _evidence(_encoded_image()), _measure(True), _measure(False), "delete", 2
    )
    assert audit.applicable
    assert not audit.accepted
    assert audit.reason == "event_deletion_requires_review"


def test_missing_manifest_is_neutral_and_cannot_authorize_insertion(tmp_path: Path) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "scorescan"
        / "resources"
        / "event_presence_visual_guard.json"
    )
    if not source.exists():
        return
    copied = tmp_path / source.name
    copied.write_bytes(source.read_bytes())
    guard = EventPresenceVisualGuard(copied)
    decision = guard.calibrate(
        _evidence(_encoded_image()), _measure(False), _measure(True), "insert", 2
    )
    assert not guard.model_verified
    assert decision.probability == 0.5
    assert not decision.accepted
    assert not decision.available


def test_bundled_event_presence_visual_guard_is_verified_and_selective() -> None:
    resource = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "scorescan"
        / "resources"
        / "event_presence_visual_guard.json"
    )
    if not resource.exists():
        return
    guard = EventPresenceVisualGuard()
    assert guard.enabled
    assert guard.model_verified
    assert guard.model_version == "scorescan-event-presence-visual-forest-1"
    dataset = build_event_presence_visual_dataset(seed=20261131, groups=20)
    probabilities = np.asarray(
        [guard.model.predict(row) for row in dataset.features], dtype=np.float64
    )
    assert dataset.features.shape == (80, len(EVENT_PRESENCE_VISUAL_FEATURE_NAMES))
    assert float(np.mean(probabilities[dataset.labels == 1])) > float(
        np.mean(probabilities[dataset.labels == 0])
    )


def test_bundled_event_presence_visual_guard_uses_kind_specific_thresholds() -> None:
    resource = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "scorescan"
        / "resources"
        / "event_presence_visual_guard.json"
    )
    if not resource.exists():
        return
    guard = EventPresenceVisualGuard()
    assert guard.model_verified
    assert set(guard.thresholds) == {"note", "rest"}
    assert guard.thresholds["note"] >= 0.90
    assert guard.thresholds["rest"] >= 0.90
    assert guard.thresholds["note"] != guard.thresholds["rest"]
    assert guard.threshold == max(guard.thresholds.values())


def test_single_edit_validation_accepts_coherent_shifted_suffix() -> None:
    after = _measure(True)
    inserted = after.notes[2]
    before = MeasureIR(
        divisions=after.divisions,
        time_signature=after.time_signature,
        key_signature=after.key_signature,
        clef=after.clef,
        notes=after.notes[:2]
        + tuple(
            replace(note, onset=note.onset - inserted.duration)
            for note in after.notes[3:]
        ),
        directions=after.directions,
        barlines=after.barlines,
    )
    assert _single_edit_event(before, after, "insert", 2) == inserted


def test_single_edit_validation_rejects_partially_shifted_suffix() -> None:
    notes = (_note(0), _note(1), _note(2), _note(3), _note(4))
    after = MeasureIR(
        divisions=4,
        time_signature=(5, 4),
        key_signature=(0, "major"),
        clef=("G", 2, 0),
        notes=notes,
        directions=(),
        barlines=(),
    )
    before = MeasureIR(
        divisions=4,
        time_signature=(5, 4),
        key_signature=(0, "major"),
        clef=("G", 2, 0),
        notes=(notes[0], notes[1], replace(notes[3], onset=Fraction(2, 1)), notes[4]),
        directions=(),
        barlines=(),
    )
    assert _single_edit_event(before, after, "insert", 2) is None


def test_competing_event_visual_features_separate_present_and_absent_sources() -> None:
    dataset = build_event_presence_visual_dataset(seed=20261241, groups=40)
    names = list(EVENT_PRESENCE_VISUAL_FEATURE_NAMES)
    margin_index = names.index("proposed_minus_displaced_f1")
    operations = np.asarray(dataset.operations)
    insertion = operations == "insert"
    positives = dataset.features[insertion & (dataset.labels == 1), margin_index]
    negatives = dataset.features[insertion & (dataset.labels == 0), margin_index]
    assert len(positives) == len(negatives) == 40
    assert float(np.mean(positives)) > float(np.mean(negatives)) + 0.05


def test_bundled_event_presence_model_declares_complete_transaction_scope() -> None:
    resource = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "scorescan"
        / "resources"
        / "event_presence_visual_guard.json"
    )
    assert resource.is_file()
    import json

    payload = json.loads(resource.read_text(encoding="utf-8"))
    assert len(payload["feature_names"]) == len(EVENT_PRESENCE_VISUAL_FEATURE_NAMES) == 640
    assert payload["target"].startswith(
        "the preserved local source image is more compatible with the complete after-insertion"
    )
    assert "proposed-versus-displaced event templates" in payload["scope"]
