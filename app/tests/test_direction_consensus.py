from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from lxml import etree

from scorescan.direction_consensus import (
    DirectionPatchCalibrator,
    DirectionPatchCandidate,
    DirectionPatchInput,
    propose_direction_patch,
)
from scorescan.direction_xml import (
    SimpleDirection,
    normalized_simple_direction_topology,
    set_simple_direction_topology,
)
from scorescan.score_ir import measure_from_xml


@dataclass
class AlwaysAccept:
    threshold: float = 0.0
    model_version: str = "test"

    def calibrate(self, item: DirectionPatchInput):
        return type(
            "Decision",
            (),
            {"probability": 1.0, "threshold": 0.0, "accepted": True, "model_version": "test"},
        )()


def _measure(
    topology: tuple[SimpleDirection, ...],
    *,
    divisions: int = 4,
    pitch_error_at: int | None = None,
    malformed: bool = False,
) -> etree._Element:
    measure = etree.Element("measure", number="1")
    attrs = etree.SubElement(measure, "attributes")
    etree.SubElement(attrs, "divisions").text = str(divisions)
    time = etree.SubElement(attrs, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"
    for index in range(4):
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = "G" if pitch_error_at == index else "CDEFGAB"[index]
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = str(divisions)
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
    set_simple_direction_topology(measure, topology, divisions)
    if malformed:
        direction = measure.find("direction")
        assert direction is not None
        direction_type = direction.find("direction-type")
        assert direction_type is not None
        etree.SubElement(direction_type, "words").text = "Allegro"
    return measure


def _candidate(
    variant: str,
    family: str,
    topology: tuple[SimpleDirection, ...],
    *,
    valid: bool = True,
    divisions: int = 4,
    pitch_error_at: int | None = None,
    malformed: bool = False,
) -> DirectionPatchCandidate:
    measure = _measure(
        topology,
        divisions=divisions,
        pitch_error_at=pitch_error_at,
        malformed=malformed,
    )
    semantics, _ = measure_from_xml(measure)
    return DirectionPatchCandidate(
        variant=variant,
        family=family,
        measure=measure,
        semantics=semantics,
        page_score=1000.0,
        page_probability=0.96,
        measure_probability=0.95,
        visual_probability=0.91,
        event_probability=0.96,
        context_probability=0.93,
        ensemble_probability=0.97,
        valid=valid,
    )


def _propose(candidates: list[DirectionPatchCandidate], *, base_measure: etree._Element | None = None):
    return propose_direction_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
        base_measure=base_measure,
    )


def test_direction_patch_adds_and_removes_dynamic_and_metronome() -> None:
    empty: tuple[SimpleDirection, ...] = ()
    correct = (
        SimpleDirection(Fraction(0), "dynamic", "below", "mf"),
        SimpleDirection(Fraction(2), "metronome", "above", "quarter=120"),
    )
    candidates = [
        _candidate("primary", "baseline", empty),
        _candidate("flat", "restoration", correct),
        _candidate("otsu", "binary", correct),
        _candidate("upscale", "scale", correct),
    ]
    result = _propose(candidates)
    assert result.accepted
    assert result.changed_direction_count == 2
    assert result.changed_kinds == ("dynamic", "metronome")
    assert result.patched_measure is not None
    assert normalized_simple_direction_topology(result.patched_measure, 4) == correct

    remove = [
        _candidate("primary", "baseline", correct),
        _candidate("flat", "restoration", empty),
        _candidate("otsu", "binary", empty),
        _candidate("upscale", "scale", empty),
    ]
    removed = _propose(remove)
    assert removed.accepted
    assert removed.patched_measure is not None
    assert normalized_simple_direction_topology(removed.patched_measure, 4) == empty


def test_direction_patch_allows_complementary_pitch_errors() -> None:
    empty: tuple[SimpleDirection, ...] = ()
    correct = (SimpleDirection(Fraction(1), "dynamic", "below", "p"),)
    candidates = [
        _candidate("primary", "baseline", empty),
        _candidate("flat", "restoration", correct, pitch_error_at=3),
        _candidate("otsu", "binary", correct, pitch_error_at=2),
        _candidate("upscale", "scale", correct, pitch_error_at=0),
    ]
    assert _propose(candidates).accepted


def test_direction_patch_split_or_invalid_sibling_abstains() -> None:
    empty: tuple[SimpleDirection, ...] = ()
    correct = (SimpleDirection(Fraction(0), "dynamic", "below", "ff"),)
    split = [
        _candidate("primary", "baseline", empty),
        _candidate("flat", "restoration", correct),
        _candidate("deblock", "restoration", empty),
        _candidate("otsu", "binary", correct),
        _candidate("upscale", "scale", correct),
    ]
    assert not _propose(split).accepted

    invalid = [
        _candidate("primary", "baseline", empty),
        _candidate("flat", "restoration", correct),
        _candidate("deblock", "restoration", correct, valid=False),
        _candidate("otsu", "binary", correct),
        _candidate("upscale", "scale", correct),
    ]
    assert not _propose(invalid).accepted


def test_direction_patch_rejects_complex_direction_content() -> None:
    empty: tuple[SimpleDirection, ...] = ()
    correct = (SimpleDirection(Fraction(0), "dynamic", "below", "sfz"),)
    candidates = [
        _candidate("primary", "baseline", empty),
        _candidate("flat", "restoration", correct, malformed=True),
        _candidate("otsu", "binary", correct, malformed=True),
        _candidate("upscale", "scale", correct, malformed=True),
    ]
    assert not _propose(candidates).accepted


def test_direction_patch_reencodes_onset_in_template_divisions() -> None:
    empty: tuple[SimpleDirection, ...] = ()
    correct = (SimpleDirection(Fraction(3, 2), "metronome", "above", "eighth.=96"),)
    base = _measure(empty, divisions=4)
    candidates = [
        _candidate("primary", "baseline", empty, divisions=4),
        _candidate("flat", "restoration", correct, divisions=8),
        _candidate("otsu", "binary", correct, divisions=12),
        _candidate("upscale", "scale", correct, divisions=16),
    ]
    result = _propose(candidates, base_measure=base)
    assert result.accepted
    assert result.patched_measure is not None
    assert result.patched_measure.findtext("./direction/offset") == "6"
    assert normalized_simple_direction_topology(result.patched_measure, 4) == correct


def test_direction_patch_missing_model_and_alignment_gap_fail_closed(tmp_path: Path) -> None:
    empty: tuple[SimpleDirection, ...] = ()
    correct = (SimpleDirection(Fraction(0), "dynamic", "below", "mp"),)
    candidates = [
        _candidate("primary", "baseline", empty),
        _candidate("flat", "restoration", correct),
        _candidate("otsu", "binary", correct),
        _candidate("upscale", "scale", correct),
    ]
    calibrator = DirectionPatchCalibrator(tmp_path / "missing.json")
    assert not calibrator.enabled
    result = propose_direction_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=calibrator,
    )
    assert result.reason == "model_guard"
    gap = propose_direction_patch(
        candidates,
        template_index=0,
        missing_candidate_count=1,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert gap.reason == "alignment_gap"


def test_direction_patch_preserves_metronome_sound_semantics_and_splits_on_disagreement() -> None:
    empty: tuple[SimpleDirection, ...] = ()
    with_sound = (
        SimpleDirection(Fraction(0), "metronome", "above", "quarter=108", True),
    )
    without_sound = (
        SimpleDirection(Fraction(0), "metronome", "above", "quarter=108", False),
    )
    candidates = [
        _candidate("primary", "baseline", empty),
        _candidate("flat", "restoration", with_sound),
        _candidate("otsu", "binary", with_sound),
        _candidate("upscale", "scale", with_sound),
    ]
    result = _propose(candidates)
    assert result.accepted
    assert result.patched_measure is not None
    assert result.patched_measure.find("./direction/sound[@tempo='108']") is not None
    assert normalized_simple_direction_topology(result.patched_measure, 4) == with_sound

    split = [
        _candidate("primary", "baseline", empty),
        _candidate("flat", "restoration", with_sound),
        _candidate("deblock", "restoration", without_sound),
        _candidate("otsu", "binary", with_sound),
        _candidate("upscale", "scale", with_sound),
    ]
    assert not _propose(split).accepted


def test_direction_patch_rejects_mark_outside_measure_duration() -> None:
    empty: tuple[SimpleDirection, ...] = ()
    outside = (SimpleDirection(Fraction(4), "dynamic", "below", "f"),)
    candidates = [
        _candidate("primary", "baseline", empty),
        _candidate("flat", "restoration", outside),
        _candidate("otsu", "binary", outside),
        _candidate("upscale", "scale", outside),
    ]
    assert not _propose(candidates).accepted


def test_bundled_direction_model_accepts_strong_independent_majority() -> None:
    empty: tuple[SimpleDirection, ...] = ()
    correct = (SimpleDirection(Fraction(0), "dynamic", "below", "mf"),)
    candidates = [
        _candidate("primary", "baseline", empty),
        _candidate("flat", "restoration", correct),
        _candidate("otsu", "binary", correct),
        _candidate("upscale", "scale", correct),
    ]
    calibrator = DirectionPatchCalibrator()
    assert calibrator.enabled
    result = propose_direction_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=calibrator,
    )
    assert result.accepted
    assert result.probability >= calibrator.threshold
    assert result.reason == "accepted"
