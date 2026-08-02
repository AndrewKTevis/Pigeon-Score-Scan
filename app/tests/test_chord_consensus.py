from __future__ import annotations

from pathlib import Path

from lxml import etree

from scorescan.chord_consensus import (
    ChordPatchCalibration,
    ChordPatchCalibrator,
    ChordPatchCandidate,
    ChordPatchInput,
    propose_chord_patch,
)
from scorescan.score_ir import measure_from_xml


class AlwaysAccept:
    threshold = 0.0
    model_version = "test"

    def calibrate(self, _item: ChordPatchInput) -> ChordPatchCalibration:
        return ChordPatchCalibration(1.0, 0.0, True, self.model_version, 1.0)


def _measure(
    topology: list[bool],
    *,
    steps: list[str] | None = None,
    divisions: int = 1,
    rest_at: int | None = None,
    beam_at: int | None = None,
) -> etree._Element:
    steps = steps or ["C", "E", "G", "D", "F"][: len(topology)]
    measure = etree.Element("measure", number="2")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = str(divisions)
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"
    for index, marker in enumerate(topology):
        note = etree.SubElement(measure, "note")
        if marker:
            etree.SubElement(note, "chord")
        if rest_at == index:
            etree.SubElement(note, "rest")
        else:
            pitch = etree.SubElement(note, "pitch")
            etree.SubElement(pitch, "step").text = steps[index]
            etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = str(divisions)
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
        if beam_at == index:
            etree.SubElement(note, "beam", number="1").text = "begin"
    return measure


def _candidate(
    variant: str,
    family: str,
    topology: list[bool],
    *,
    steps: list[str] | None = None,
    valid: bool = True,
    rest_at: int | None = None,
    beam_at: int | None = None,
) -> ChordPatchCandidate:
    measure = _measure(topology, steps=steps, rest_at=rest_at, beam_at=beam_at)
    semantics, _ = measure_from_xml(measure)
    return ChordPatchCandidate(
        variant=variant,
        family=family,
        measure=measure,
        semantics=semantics,
        page_score=1000.0,
        page_probability=0.94,
        measure_probability=0.95,
        visual_probability=0.91,
        event_probability=0.96,
        context_probability=0.93,
        ensemble_probability=0.97,
        valid=valid,
    )


def _propose(candidates: list[ChordPatchCandidate]):
    return propose_chord_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )


def test_chord_patch_adds_missing_marker_and_restores_meter() -> None:
    wrong = [False, False, False, False, False]
    correct = [False, True, False, False, False]
    candidates = [
        _candidate("primary", "baseline", wrong),
        _candidate("flat", "restoration", correct),
        _candidate("otsu", "binary", correct),
        _candidate("upscale", "scale", correct),
    ]
    result = _propose(candidates)
    assert result.accepted
    assert result.changed_event_indices == (1,)
    assert result.patched_measure is not None
    assert result.patched_measure.findall("note")[1].find("chord") is not None
    parsed, _ = measure_from_xml(result.patched_measure)
    assert [note.chord for note in parsed.notes] == correct
    assert sum(note.duration for note in parsed.notes if not note.chord) == 4


def test_chord_patch_removes_spurious_marker_and_restores_meter() -> None:
    wrong = [False, True, False, False]
    correct = [False, False, False, False]
    candidates = [
        _candidate("primary", "baseline", wrong),
        _candidate("flat", "restoration", correct),
        _candidate("otsu", "binary", correct),
        _candidate("upscale", "scale", correct),
    ]
    result = _propose(candidates)
    assert result.accepted
    assert result.patched_measure is not None
    assert result.patched_measure.findall("note")[1].find("chord") is None


def test_chord_patch_allows_isolated_pitch_disagreement() -> None:
    wrong = [False, False, False, False, False]
    correct = [False, True, False, False, False]
    candidates = [
        _candidate("primary", "baseline", wrong, steps=["C", "D", "E", "F", "G"]),
        _candidate("flat", "restoration", correct, steps=["C", "E", "E", "F", "G"]),
        _candidate("otsu", "binary", correct, steps=["C", "E", "D", "F", "G"]),
        _candidate("upscale", "scale", correct, steps=["C", "E", "E", "F", "A"]),
    ]
    result = _propose(candidates)
    assert result.accepted
    assert result.patched_measure is not None
    # Chord repair never copies a pitch from another candidate.
    assert [node.text for node in result.patched_measure.findall("note/pitch/step")] == ["C", "D", "E", "F", "G"]


def test_chord_patch_split_siblings_make_family_abstain() -> None:
    wrong = [False, False, False, False, False]
    correct = [False, True, False, False, False]
    candidates = [
        _candidate("primary", "baseline", wrong),
        _candidate("flat", "restoration", correct),
        _candidate("deblock", "restoration", wrong),
        _candidate("otsu", "binary", correct),
        _candidate("upscale", "scale", correct),
    ]
    result = _propose(candidates)
    assert not result.accepted
    assert result.reason == "no_strict_topology_family_majority"


def test_chord_patch_invalid_sibling_makes_family_abstain() -> None:
    wrong = [False, False, False, False, False]
    correct = [False, True, False, False, False]
    candidates = [
        _candidate("primary", "baseline", wrong),
        _candidate("flat", "restoration", correct),
        _candidate("deblock", "restoration", correct, valid=False),
        _candidate("otsu", "binary", correct),
        _candidate("upscale", "scale", correct),
    ]
    result = _propose(candidates)
    assert not result.accepted
    assert result.reason == "no_strict_topology_family_majority"


def test_chord_patch_rejects_marker_on_rest() -> None:
    wrong = [False, False, False, False, False]
    unsafe = [False, True, False, False, False]
    candidates = [
        _candidate("primary", "baseline", wrong, rest_at=1),
        _candidate("flat", "restoration", unsafe, rest_at=1),
        _candidate("otsu", "binary", unsafe, rest_at=1),
        _candidate("upscale", "scale", unsafe, rest_at=1),
    ]
    result = _propose(candidates)
    assert not result.accepted
    assert result.reason == "insufficient_families"


def test_chord_patch_rejects_meter_worsening() -> None:
    correct_meter = [False, False, False, False]
    underfull = [False, True, False, False]
    candidates = [
        _candidate("primary", "baseline", correct_meter),
        _candidate("flat", "restoration", underfull),
        _candidate("otsu", "binary", underfull),
        _candidate("upscale", "scale", underfull),
    ]
    result = _propose(candidates)
    assert not result.accepted
    assert result.reason == "meter_error_worsened"


def test_chord_patch_alignment_gap_and_scope_fail_closed() -> None:
    wrong = [False] * 10
    many = [False, True, True, False, True, True, False, True, True, False]
    candidates = [
        _candidate("primary", "baseline", wrong, steps=list("CDEFGABCDE")),
        _candidate("flat", "restoration", many, steps=list("CDEFGABCDE")),
        _candidate("otsu", "binary", many, steps=list("CDEFGABCDE")),
        _candidate("upscale", "scale", many, steps=list("CDEFGABCDE")),
    ]
    gap = propose_chord_patch(
        candidates,
        template_index=0,
        missing_candidate_count=1,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert not gap.accepted and gap.reason == "alignment_gap"
    result = _propose(candidates)
    assert not result.accepted and result.reason == "change_scope_too_large"


def test_chord_patch_missing_model_fails_closed(tmp_path: Path) -> None:
    calibrator = ChordPatchCalibrator(tmp_path / "missing.json")
    assert not calibrator.enabled
    wrong = [False, False, False, False, False]
    correct = [False, True, False, False, False]
    candidates = [
        _candidate("primary", "baseline", wrong),
        _candidate("flat", "restoration", correct),
        _candidate("otsu", "binary", correct),
        _candidate("upscale", "scale", correct),
    ]
    result = propose_chord_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=calibrator,
    )
    assert not result.accepted
    assert result.reason == "model_guard"


def test_chord_patch_model_is_verified_and_rejects_conflict() -> None:
    calibrator = ChordPatchCalibrator()
    assert calibrator.enabled
    assert calibrator.model_verified
    assert calibrator.model_version == "scorescan-chord-patch-forest-1"

    base = dict(
        candidate_count=6,
        eligible_family_count=4,
        voting_family_count=4,
        changed_marker_count=1,
        total_event_count=7,
        added_marker_count=1,
        removed_marker_count=0,
        winner_family_count=3,
        runner_up_family_count=1,
        template_family_count=1,
        incomplete_family_count=0,
        winner_chord_group_count=1,
        winner_max_chord_size=2,
        expected_measure_duration=4.0,
        template_duration_error=1.0,
        patched_duration_error=0.0,
        mean_support_page_probability=0.92,
        mean_support_measure_probability=0.94,
        mean_support_visual_probability=0.91,
        mean_support_event_probability=0.95,
        mean_support_context_probability=0.93,
        mean_support_ensemble_probability=0.96,
        minimum_support_ensemble_probability=0.89,
        mean_support_page_score_margin=18.0,
        mean_support_vs_template_measure_probability=0.14,
        mean_support_vs_template_visual_probability=0.16,
        mean_support_vs_template_event_probability=0.15,
        mean_support_vs_template_context_probability=0.13,
        mean_support_vs_template_ensemble_probability=0.17,
    )
    favourable = calibrator.predict_probability(ChordPatchInput(**base))
    conflict = calibrator.predict_probability(ChordPatchInput(**{
        **base,
        "changed_marker_count": 3,
        "added_marker_count": 2,
        "removed_marker_count": 1,
        "template_duration_error": 0.25,
        "patched_duration_error": 0.25,
        "mean_support_visual_probability": 0.18,
        "mean_support_context_probability": 0.22,
        "mean_support_ensemble_probability": 0.51,
        "minimum_support_ensemble_probability": 0.24,
        "mean_support_page_score_margin": -26.0,
        "mean_support_vs_template_ensemble_probability": -0.17,
    }))
    assert favourable > conflict
