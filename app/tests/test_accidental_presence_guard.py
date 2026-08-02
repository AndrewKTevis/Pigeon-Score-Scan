from __future__ import annotations

import base64
import sys
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from accidental_presence_training_data import build_accidental_presence_dataset  # noqa: E402
from scorescan.accidental_presence_guard import (  # noqa: E402
    ACCIDENTAL_PRESENCE_FEATURE_NAMES,
    AccidentalPresenceGuard,
    accidental_hog_features,
    accidental_hog_features_at_position,
)
from scorescan.local_symbol_image import decode_symbol_guard_image, event_position  # noqa: E402
from scorescan.score_ir import MeasureIR, NoteIR, PitchIR  # noqa: E402
from scorescan.visual_evidence import VisualMeasureEvidence  # noqa: E402


def _measure() -> MeasureIR:
    note = NoteIR(
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
    )
    return MeasureIR(1, (4, 4), (0, "major"), ("G", 2, 0), (note,), (), ())


def _evidence(encoded: str) -> VisualMeasureEvidence:
    return VisualMeasureEvidence(
        page_index=1,
        system_index=1,
        measure_index=1,
        bbox=(0, 0, 256, 96),
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


def test_bundled_accidental_presence_guard_is_verified_and_selective() -> None:
    guard = AccidentalPresenceGuard()
    assert guard.enabled
    assert guard.model_verified
    assert guard.model_version == "scorescan-accidental-presence-forest-1"
    assert guard.present_threshold > 0.8
    assert guard.absent_threshold > 0.8

    dataset = build_accidental_presence_dataset(seed=20260831, groups=30)
    probabilities = np.asarray(
        [guard.model.predict(row) for row in dataset.features], dtype=np.float64
    )
    assert dataset.features.shape == (60, len(ACCIDENTAL_PRESENCE_FEATURE_NAMES))
    assert float(np.mean(probabilities[dataset.labels == 1])) > float(
        np.mean(probabilities[dataset.labels == 0])
    )


def test_accidental_presence_features_reject_malformed_or_oversized_png() -> None:
    measure = _measure()
    malformed = _evidence(base64.b64encode(b"not a png").decode("ascii"))
    oversized = _evidence("A" * 65_537)
    assert accidental_hog_features(malformed, measure, 0) is None
    assert accidental_hog_features(oversized, measure, 0) is None


def test_explicit_registered_anchor_uses_deployed_feature_extractor() -> None:
    # Reconstruct deterministic evidence to verify the explicit-anchor entry
    # point through the same public feature schema.
    measure = _measure()
    image = np.zeros((96, 256), dtype=np.uint8)
    image[30:66, 80:84] = 255
    encoded_ok, encoded = cv2.imencode(".png", image)
    assert encoded_ok
    evidence = _evidence(base64.b64encode(encoded.tobytes()).decode("ascii"))
    decoded = decode_symbol_guard_image(evidence.symbol_guard_image)
    assert decoded is not None
    x_ratio, y_ratio = event_position(measure, measure.notes[0])
    explicit = accidental_hog_features_at_position(decoded, x_ratio, y_ratio)
    deployed = accidental_hog_features(evidence, measure, 0)
    assert deployed == explicit
    assert len(explicit) == len(ACCIDENTAL_PRESENCE_FEATURE_NAMES)


def test_explicit_registered_anchor_rejects_non_grayscale_image() -> None:
    with pytest.raises(ValueError, match="grayscale"):
        accidental_hog_features_at_position(
            np.zeros((96, 256, 3), dtype=np.uint8),
            0.5,
            0.5,
        )


def test_accidental_presence_guard_missing_manifest_fails_closed(tmp_path: Path) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "scorescan"
        / "resources"
        / "accidental_presence_guard.json"
    )
    model_path = tmp_path / source.name
    model_path.write_bytes(source.read_bytes())
    guard = AccidentalPresenceGuard(model_path)
    calibration = guard.calibrate(None, _measure(), 0, expected_present=True)
    assert not guard.model_verified
    assert calibration.probability == 0.5
    assert not calibration.accepted
