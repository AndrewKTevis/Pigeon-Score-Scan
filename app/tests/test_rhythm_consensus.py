from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from lxml import etree

from scorescan.rhythm_consensus import (
    RhythmPatchCalibration,
    RhythmPatchCalibrator,
    RhythmPatchCandidate,
    RhythmPatchInput,
    propose_rhythm_patch,
)
from scorescan.rhythm_symbol_guard import RhythmSymbolGuardCalibration
from scorescan.score_ir import measure_from_xml
from scorescan.visual_evidence import VisualMeasureEvidence, extract_crop_features


@dataclass
class AlwaysAccept:
    threshold: float = 0.92
    model_version: str = "test-rhythm-patch"

    def calibrate(self, item: RhythmPatchInput) -> RhythmPatchCalibration:
        return RhythmPatchCalibration(1.0, self.threshold, True, self.model_version, 1.0)


@dataclass
class SymbolDecision:
    accepted: bool
    threshold: float = 0.9875
    model_version: str = "test-rhythm-symbol"

    def calibrate(self, _item) -> RhythmSymbolGuardCalibration:
        confidence = 0.995 if self.accepted else 0.5
        return RhythmSymbolGuardCalibration(
            forward_probability=confidence,
            reverse_probability=1.0 - confidence,
            confidence=confidence,
            threshold=self.threshold,
            accepted=self.accepted,
            model_version=self.model_version,
        )


def _visual_evidence() -> VisualMeasureEvidence:
    image = np.full((120, 360), 255, np.uint8)
    spacing = 12
    staff_top = 36
    for line in range(5):
        cv2.line(image, (4, staff_top + line * spacing), (356, staff_top + line * spacing), 0, 1)
    for index, x in enumerate((35, 115, 195, 275)):
        y = staff_top + (index % 4) * spacing
        cv2.ellipse(image, (x, y), (6, 4), -18, 0, 360, 0, -1)
        cv2.line(image, (x + 5, y), (x + 5, y - 30), 0, 1)
    features = extract_crop_features(
        image, spacing=spacing, staff_top=staff_top, staff_bottom=staff_top + 4 * spacing
    )
    return VisualMeasureEvidence(1, 1, 1, (0, 0, 360, 120), spacing, **features)


def _measure(
    durations: list[int],
    *,
    types: list[str] | None = None,
    pitches: list[str] | None = None,
    chord_at: int | None = None,
    divisions: int = 2,
) -> etree._Element:
    measure = etree.Element("measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = str(divisions)
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"
    types = types or [
        "quarter" if value == divisions else "eighth"
        for value in durations
    ]
    pitches = pitches or ["C", "D", "E", "F"][: len(durations)]
    for index, (duration, note_type, step) in enumerate(zip(durations, types, pitches, strict=True)):
        note = etree.SubElement(measure, "note")
        if chord_at == index:
            etree.SubElement(note, "chord")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = str(duration)
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = note_type
    return measure


def _candidate(
    variant: str,
    family: str,
    durations: list[int],
    *,
    types: list[str] | None = None,
    pitches: list[str] | None = None,
    chord_at: int | None = None,
    divisions: int = 2,
    valid: bool = True,
) -> RhythmPatchCandidate:
    measure = _measure(
        durations,
        types=types,
        pitches=pitches,
        chord_at=chord_at,
        divisions=divisions,
    )
    semantics, _ = measure_from_xml(measure)
    return RhythmPatchCandidate(
        variant=variant,
        family=family,
        measure=measure,
        semantics=semantics,
        page_score=1000.0,
        page_probability=0.91,
        measure_probability=0.92,
        visual_probability=0.84,
        event_probability=0.94,
        context_probability=0.85,
        ensemble_probability=0.95,
        valid=valid,
    )


def test_rhythm_consensus_recovers_complementary_family_errors() -> None:
    candidates = [
        _candidate("primary", "baseline", [2, 2, 2, 1]),
        _candidate("flat", "restoration", [2, 2, 1, 2]),
        _candidate("otsu", "binary", [2, 1, 2, 2]),
        _candidate("upscale", "scale", [2, 2, 2, 2]),
    ]
    result = propose_rhythm_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert result.accepted
    assert result.changed_event_indices == (3,)
    assert result.patched_measure is not None
    parsed, _ = measure_from_xml(result.patched_measure)
    assert [str(note.duration) for note in parsed.notes] == ["1", "1", "1", "1"]
    assert [note.note_type for note in parsed.notes] == ["quarter"] * 4


def test_rhythm_consensus_collapses_correlated_sibling_votes() -> None:
    candidates = [
        _candidate("primary", "baseline", [2, 2, 2, 1]),
        _candidate("flat", "restoration", [2, 2, 2, 2]),
        _candidate("deblock", "restoration", [2, 2, 2, 2]),
        _candidate("otsu", "binary", [2, 2, 2, 4], types=["quarter", "quarter", "quarter", "half"]),
        _candidate("upscale", "scale", [2, 2, 2, 1]),
    ]
    result = propose_rhythm_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert not result.accepted
    assert result.reason == "no_strict_event_family_majority"



def test_rhythm_consensus_invalid_sibling_makes_family_abstain() -> None:
    candidates = [
        _candidate("primary", "baseline", [2, 2, 2, 1]),
        _candidate("flat", "restoration", [2, 2, 2, 2]),
        _candidate("deblock", "restoration", [2, 2, 2, 2], valid=False),
        _candidate("otsu", "binary", [2, 2, 2, 2]),
        _candidate("upscale", "scale", [2, 2, 2, 2]),
    ]
    result = propose_rhythm_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert not result.accepted
    assert result.reason == "no_strict_event_family_majority"


def test_rhythm_consensus_reencodes_duration_in_template_divisions() -> None:
    candidates = [
        _candidate("primary", "baseline", [2, 2, 2, 1], divisions=2),
        _candidate("flat", "restoration", [4, 4, 4, 4], divisions=4),
        _candidate("otsu", "binary", [4, 4, 4, 4], divisions=4),
        _candidate("upscale", "scale", [4, 4, 4, 4], divisions=4),
    ]
    result = propose_rhythm_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert result.accepted
    assert result.patched_measure is not None
    assert [node.text for node in result.patched_measure.findall("./note/duration")] == ["2"] * 4
    parsed, _ = measure_from_xml(result.patched_measure)
    assert [str(note.duration) for note in parsed.notes] == ["1"] * 4

def test_rhythm_consensus_rejects_complex_structure() -> None:
    candidates = [
        _candidate("primary", "baseline", [2, 2, 2, 1], chord_at=1),
        _candidate("flat", "restoration", [2, 2, 2, 2], chord_at=1),
        _candidate("otsu", "binary", [2, 2, 2, 2], chord_at=1),
        _candidate("upscale", "scale", [2, 2, 2, 2], chord_at=1),
    ]
    result = propose_rhythm_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert not result.accepted
    assert result.reason == "unsupported_rhythm_structure"


def test_rhythm_patch_composes_onto_pitch_corrected_base() -> None:
    candidates = [
        _candidate("primary", "baseline", [2, 2, 2, 1], pitches=["C", "D", "F", "F"]),
        _candidate("flat", "restoration", [2, 2, 1, 2], pitches=["C", "D", "E", "F"]),
        _candidate("otsu", "binary", [2, 1, 2, 2], pitches=["C", "D", "E", "F"]),
        _candidate("upscale", "scale", [2, 2, 2, 2], pitches=["C", "D", "E", "F"]),
    ]
    base = _measure([2, 2, 2, 1], pitches=["C", "D", "E", "F"])
    result = propose_rhythm_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
        base_measure=base,
    )
    assert result.accepted
    assert result.patched_measure is not None
    assert [node.text for node in result.patched_measure.findall("./note/pitch/step")] == ["C", "D", "E", "F"]


def test_rhythm_patch_model_is_verified_and_rejects_conflict() -> None:
    calibrator = RhythmPatchCalibrator()
    assert calibrator.enabled
    assert calibrator.model_verified
    assert calibrator.model_version == "scorescan-rhythm-patch-forest-1"

    base = dict(
        candidate_count=6,
        eligible_family_count=4,
        voting_family_count=4,
        changed_event_count=1,
        total_event_count=8,
        minimum_winner_family_support_ratio=0.75,
        mean_winner_family_support_ratio=0.78,
        minimum_winner_margin_ratio=0.50,
        mean_winner_margin_ratio=0.52,
        maximum_template_family_support_ratio=0.20,
        family_abstention_ratio=0.02,
        minimum_pitch_coherence_ratio=0.75,
        mean_pitch_coherence_ratio=0.90,
        template_duration_error=0.5,
        patched_duration_error=0.0,
        duration_error_improvement=0.5,
        template_type_mismatch_ratio=0.125,
        patched_type_mismatch_ratio=0.0,
        mean_support_page_probability=0.91,
        mean_support_measure_probability=0.93,
        mean_support_visual_probability=0.85,
        mean_support_event_probability=0.94,
        mean_support_context_probability=0.85,
        mean_support_ensemble_probability=0.96,
        minimum_support_ensemble_probability=0.88,
        mean_support_page_score_margin=18.0,
        mean_support_vs_template_ensemble_probability=0.20,
    )
    favourable = calibrator.predict_probability(RhythmPatchInput(**base))
    conflict = calibrator.predict_probability(RhythmPatchInput(**{
        **base,
        "minimum_pitch_coherence_ratio": 0.50,
        "mean_support_visual_probability": 0.18,
        "mean_support_context_probability": 0.25,
        "mean_support_vs_template_ensemble_probability": -0.08,
        "duration_error_improvement": 0.0,
    }))
    assert favourable > conflict


def test_rhythm_symbol_guard_can_only_veto_semantically_accepted_patch() -> None:
    candidates = [
        _candidate("primary", "baseline", [2, 2, 2, 1]),
        _candidate("flat", "restoration", [2, 2, 1, 2]),
        _candidate("otsu", "binary", [2, 1, 2, 2]),
        _candidate("upscale", "scale", [2, 2, 2, 2]),
    ]
    evidence = _visual_evidence()
    rejected = propose_rhythm_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
        visual_evidence=evidence,
        symbol_guard=SymbolDecision(False),  # type: ignore[arg-type]
    )
    assert not rejected.accepted
    assert rejected.reason == "rhythm_symbol_guard"
    assert rejected.symbol_guard_model_version == "test-rhythm-symbol"

    accepted = propose_rhythm_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
        visual_evidence=evidence,
        symbol_guard=SymbolDecision(True),  # type: ignore[arg-type]
    )
    assert accepted.accepted
    assert accepted.patched_measure is not None
    assert accepted.symbol_guard_confidence == 0.995


def test_rhythm_patch_preserves_existing_behavior_without_visual_evidence() -> None:
    candidates = [
        _candidate("primary", "baseline", [2, 2, 2, 1]),
        _candidate("flat", "restoration", [2, 2, 1, 2]),
        _candidate("otsu", "binary", [2, 1, 2, 2]),
        _candidate("upscale", "scale", [2, 2, 2, 2]),
    ]
    result = propose_rhythm_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
        visual_evidence=None,
        symbol_guard=SymbolDecision(False),  # type: ignore[arg-type]
    )
    assert result.accepted
    assert result.symbol_guard_model_version == "not_applicable"
