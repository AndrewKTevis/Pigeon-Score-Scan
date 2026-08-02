from __future__ import annotations

from pathlib import Path

from lxml import etree

from scorescan.score_ir import measure_from_xml
from scorescan.tuplet_consensus import (
    TupletPatchCalibration,
    TupletPatchCalibrator,
    TupletPatchCandidate,
    TupletPatchInput,
    propose_tuplet_patch,
)
from scorescan.tuplet_xml import read_simple_tuplet_state, set_simple_tuplet_state


class AlwaysAccept:
    threshold = 0.0
    model_version = "test"

    def calibrate(self, _item: TupletPatchInput) -> TupletPatchCalibration:
        return TupletPatchCalibration(1.0, 0.0, True, self.model_version, 1.0)


def _measure(
    groups: tuple[tuple[int, int], ...],
    *,
    malformed: bool = False,
    beam_at: int | None = None,
) -> etree._Element:
    measure = etree.Element("measure", number="2")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "6"
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"
    durations = [2, 2, 2, 6, 6, 6]
    types = ["eighth", "eighth", "eighth", "quarter", "quarter", "quarter"]
    steps = ["C", "D", "E", "F", "G", "A"]
    grouped = {index: (index == start, index == stop) for start, stop in groups for index in range(start, stop + 1)}
    for index, (duration, note_type, step) in enumerate(zip(durations, types, steps, strict=True)):
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = str(duration)
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = note_type
        if index in grouped:
            start, stop = grouped[index]
            set_simple_tuplet_state(note, ratio=(3, 2), start=start, stop=stop)
        if malformed and index == 0:
            marker = note.find("./notations/tuplet")
            assert marker is not None
            marker.set("number", "2")
        if beam_at == index:
            etree.SubElement(note, "beam", number="1").text = "begin"
    return measure


def _candidate(
    variant: str,
    family: str,
    groups: tuple[tuple[int, int], ...],
    *,
    valid: bool = True,
    malformed: bool = False,
) -> TupletPatchCandidate:
    measure = _measure(groups, malformed=malformed)
    semantics, _ = measure_from_xml(measure)
    return TupletPatchCandidate(
        variant=variant,
        family=family,
        measure=measure,
        semantics=semantics,
        page_score=1000.0,
        page_probability=0.94,
        measure_probability=0.95,
        visual_probability=0.92,
        event_probability=0.96,
        context_probability=0.93,
        ensemble_probability=0.97,
        valid=valid,
    )


def _propose(candidates: list[TupletPatchCandidate]):
    return propose_tuplet_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )


def test_tuplet_patch_adds_missing_simple_triplet() -> None:
    candidates = [
        _candidate("primary", "baseline", ()),
        _candidate("flat", "restoration", ((0, 2),)),
        _candidate("otsu", "binary", ((0, 2),)),
        _candidate("upscale", "scale", ((0, 2),)),
    ]
    result = _propose(candidates)
    assert result.accepted
    assert result.changed_event_indices == (0, 1, 2)
    assert result.changed_group_count == 1
    assert result.patched_measure is not None
    states = [read_simple_tuplet_state(note) for note in result.patched_measure.findall("note")]
    assert states[0] is not None and states[0].start and states[0].ratio == (3, 2)
    assert states[1] is not None and states[1].ratio == (3, 2) and not states[1].start and not states[1].stop
    assert states[2] is not None and states[2].stop and states[2].ratio == (3, 2)
    parsed, _ = measure_from_xml(result.patched_measure)
    assert [note.tuple_ratio for note in parsed.notes[:3]] == [(3, 2)] * 3
    assert sum(note.duration for note in parsed.notes) == 4


def test_tuplet_patch_removes_spurious_triplet() -> None:
    candidates = [
        _candidate("primary", "baseline", ((0, 2),)),
        _candidate("flat", "restoration", ()),
        _candidate("otsu", "binary", ()),
        _candidate("upscale", "scale", ()),
    ]
    result = _propose(candidates)
    assert result.accepted
    assert result.patched_measure is not None
    assert all(note.find("time-modification") is None for note in result.patched_measure.findall("note"))
    assert not result.patched_measure.findall("./note/notations/tuplet")


def test_tuplet_patch_split_siblings_abstain() -> None:
    candidates = [
        _candidate("primary", "baseline", ()),
        _candidate("flat", "restoration", ((0, 2),)),
        _candidate("deblock", "restoration", ()),
        _candidate("otsu", "binary", ((0, 2),)),
        _candidate("upscale", "scale", ((0, 2),)),
    ]
    result = _propose(candidates)
    assert not result.accepted
    assert result.reason == "no_strict_topology_family_majority"


def test_tuplet_patch_invalid_sibling_abstains_family() -> None:
    candidates = [
        _candidate("primary", "baseline", ()),
        _candidate("flat", "restoration", ((0, 2),)),
        _candidate("deblock", "restoration", ((0, 2),), valid=False),
        _candidate("otsu", "binary", ((0, 2),)),
        _candidate("upscale", "scale", ((0, 2),)),
    ]
    result = _propose(candidates)
    assert not result.accepted
    assert result.reason == "no_strict_topology_family_majority"


def test_tuplet_patch_rejects_malformed_candidate_family() -> None:
    candidates = [
        _candidate("primary", "baseline", ()),
        _candidate("flat", "restoration", ((0, 2),), malformed=True),
        _candidate("otsu", "binary", ((0, 2),)),
        _candidate("upscale", "scale", ((0, 2),)),
    ]
    result = _propose(candidates)
    assert not result.accepted
    assert result.reason == "no_strict_topology_family_majority"


def test_tuplet_patch_alignment_gap_fails_closed() -> None:
    candidates = [
        _candidate("primary", "baseline", ()),
        _candidate("flat", "restoration", ((0, 2),)),
        _candidate("otsu", "binary", ((0, 2),)),
        _candidate("upscale", "scale", ((0, 2),)),
    ]
    result = propose_tuplet_patch(
        candidates,
        template_index=0,
        missing_candidate_count=1,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert not result.accepted and result.reason == "alignment_gap"


def test_tuplet_patch_missing_model_fails_closed(tmp_path: Path) -> None:
    calibrator = TupletPatchCalibrator(tmp_path / "missing.json")
    assert not calibrator.enabled
    candidates = [
        _candidate("primary", "baseline", ()),
        _candidate("flat", "restoration", ((0, 2),)),
        _candidate("otsu", "binary", ((0, 2),)),
        _candidate("upscale", "scale", ((0, 2),)),
    ]
    result = propose_tuplet_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=calibrator,
    )
    assert not result.accepted
    assert result.reason == "model_guard"


def _calibration_input(*, favorable: bool) -> TupletPatchInput:
    if favorable:
        return TupletPatchInput(
            candidate_count=5,
            eligible_family_count=4,
            voting_family_count=4,
            changed_event_count=3,
            total_event_count=9,
            added_group_count=1,
            removed_group_count=0,
            winner_family_count=3,
            runner_up_family_count=1,
            template_family_count=1,
            incomplete_family_count=0,
            winner_group_count=1,
            minimum_group_pitch_span=2,
            maximum_group_pitch_span=7,
            expected_measure_duration=4.0,
            template_duration_error=0.0,
            patched_duration_error=0.0,
            mean_support_page_probability=0.96,
            mean_support_measure_probability=0.97,
            mean_support_visual_probability=0.95,
            mean_support_event_probability=0.98,
            mean_support_context_probability=0.96,
            mean_support_ensemble_probability=0.98,
            minimum_support_ensemble_probability=0.96,
            mean_support_page_score_margin=14.0,
            mean_support_vs_template_measure_probability=0.14,
            mean_support_vs_template_visual_probability=0.12,
            mean_support_vs_template_event_probability=0.16,
            mean_support_vs_template_context_probability=0.13,
            mean_support_vs_template_ensemble_probability=0.15,
        )
    return TupletPatchInput(
        candidate_count=7,
        eligible_family_count=3,
        voting_family_count=3,
        changed_event_count=6,
        total_event_count=10,
        added_group_count=2,
        removed_group_count=0,
        winner_family_count=2,
        runner_up_family_count=1,
        template_family_count=1,
        incomplete_family_count=2,
        winner_group_count=2,
        minimum_group_pitch_span=18,
        maximum_group_pitch_span=31,
        expected_measure_duration=4.0,
        template_duration_error=0.0,
        patched_duration_error=0.0,
        mean_support_page_probability=0.58,
        mean_support_measure_probability=0.57,
        mean_support_visual_probability=0.42,
        mean_support_event_probability=0.55,
        mean_support_context_probability=0.46,
        mean_support_ensemble_probability=0.53,
        minimum_support_ensemble_probability=0.41,
        mean_support_page_score_margin=-28.0,
        mean_support_vs_template_measure_probability=-0.16,
        mean_support_vs_template_visual_probability=-0.24,
        mean_support_vs_template_event_probability=-0.18,
        mean_support_vs_template_context_probability=-0.20,
        mean_support_vs_template_ensemble_probability=-0.21,
    )


def test_bundled_tuplet_calibrator_is_verified_and_ranks_clean_evidence_higher() -> None:
    calibrator = TupletPatchCalibrator()
    assert calibrator.enabled
    assert calibrator.model_verified
    assert calibrator.model_version == "scorescan-tuplet-patch-forest-1"
    favorable = calibrator.predict_probability(_calibration_input(favorable=True))
    conflict = calibrator.predict_probability(_calibration_input(favorable=False))
    assert favorable > conflict
    assert favorable >= calibrator.threshold
    assert conflict < calibrator.threshold
