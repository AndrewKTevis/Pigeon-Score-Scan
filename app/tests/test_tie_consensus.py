from __future__ import annotations

from pathlib import Path

from lxml import etree

from scorescan.score_ir import measure_from_xml
from scorescan.tie_consensus import (
    TiePatchCalibration,
    TiePatchCalibrator,
    TiePatchCandidate,
    TiePatchInput,
    propose_tie_patch,
)


class AlwaysAccept:
    threshold = 0.0
    model_version = "test"

    def calibrate(self, _item: TiePatchInput) -> TiePatchCalibration:
        return TiePatchCalibration(1.0, 0.0, True, self.model_version, 1.0)


def _measure(
    states: list[tuple[str, ...]],
    *,
    steps: list[str] | None = None,
    direct_only: bool = False,
    notation_only: bool = False,
    rest_at: int | None = None,
    chord_at: int | None = None,
) -> etree._Element:
    steps = steps or ["C", "C", "D", "E", "F", "G"][: len(states)]
    measure = etree.Element("measure", number="2")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = str(len(states))
    etree.SubElement(time, "beat-type").text = "4"
    for index, state in enumerate(states):
        note = etree.SubElement(measure, "note")
        if chord_at == index:
            etree.SubElement(note, "chord")
        if rest_at == index:
            etree.SubElement(note, "rest")
        else:
            pitch = etree.SubElement(note, "pitch")
            etree.SubElement(pitch, "step").text = steps[index]
            etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "1"
        for value in state:
            if not notation_only:
                etree.SubElement(note, "tie", type=value)
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
        if state and not direct_only:
            notations = etree.SubElement(note, "notations")
            for value in state:
                etree.SubElement(notations, "tied", type=value)
    return measure


def _candidate(
    variant: str,
    family: str,
    states: list[tuple[str, ...]],
    *,
    steps: list[str] | None = None,
    valid: bool = True,
    direct_only: bool = False,
    notation_only: bool = False,
    rest_at: int | None = None,
    chord_at: int | None = None,
) -> TiePatchCandidate:
    measure = _measure(
        states,
        steps=steps,
        direct_only=direct_only,
        notation_only=notation_only,
        rest_at=rest_at,
        chord_at=chord_at,
    )
    semantics, _ = measure_from_xml(measure)
    return TiePatchCandidate(
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


def _propose(candidates: list[TiePatchCandidate], *, base_measure: etree._Element | None = None):
    return propose_tie_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
        base_measure=base_measure,
    )


def test_tie_patch_adds_missing_internal_pair_and_canonicalizes_xml() -> None:
    wrong = [(), (), (), ()]
    correct = [("start",), ("stop",), (), ()]
    candidates = [
        _candidate("primary", "baseline", wrong),
        _candidate("flat", "restoration", correct, direct_only=True),
        _candidate("otsu", "binary", correct, notation_only=True),
        _candidate("upscale", "scale", correct),
    ]
    result = _propose(candidates)
    assert result.accepted
    assert result.changed_event_indices == (0, 1)
    assert result.patched_measure is not None
    notes = result.patched_measure.findall("note")
    assert [node.get("type") for node in notes[0].findall("tie")] == ["start"]
    assert [node.get("type") for node in notes[0].findall("./notations/tied")] == ["start"]
    assert [node.get("type") for node in notes[1].findall("tie")] == ["stop"]
    assert [node.get("type") for node in notes[1].findall("./notations/tied")] == ["stop"]
    parsed, _ = measure_from_xml(result.patched_measure)
    assert parsed.notes[0].ties == ("start",)
    assert parsed.notes[1].ties == ("stop",)


def test_tie_patch_removes_spurious_pair_without_changing_other_notation() -> None:
    wrong = [("start",), ("stop",), (), ()]
    correct = [(), (), (), ()]
    candidates = [
        _candidate("primary", "baseline", wrong),
        _candidate("flat", "restoration", correct),
        _candidate("otsu", "binary", correct),
        _candidate("upscale", "scale", correct),
    ]
    base = _measure(wrong)
    articulation = etree.SubElement(base.findall("note")[2], "notations")
    articulations = etree.SubElement(articulation, "articulations")
    etree.SubElement(articulations, "staccato")
    # The base contains unrelated notation not present in candidates, so fail closed.
    rejected = _propose(candidates, base_measure=base)
    assert not rejected.accepted
    assert rejected.reason == "non_tie_structure_disagreement"

    result = _propose(candidates)
    assert result.accepted
    assert result.patched_measure is not None
    assert not result.patched_measure.findall(".//tie")
    assert not result.patched_measure.findall(".//tied")


def test_tie_patch_supports_a_valid_three_note_chain() -> None:
    wrong = [(), (), (), ()]
    correct = [("start",), ("stop", "start"), ("stop",), ()]
    steps = ["C", "C", "C", "D"]
    candidates = [
        _candidate("primary", "baseline", wrong, steps=steps),
        _candidate("flat", "restoration", correct, steps=steps),
        _candidate("otsu", "binary", correct, steps=steps),
        _candidate("upscale", "scale", correct, steps=steps),
    ]
    result = _propose(candidates)
    assert result.accepted
    assert result.changed_event_indices == (0, 1, 2)


def test_tie_patch_split_or_invalid_sibling_makes_family_abstain() -> None:
    wrong = [(), (), (), ()]
    correct = [("start",), ("stop",), (), ()]
    split = [
        _candidate("primary", "baseline", wrong),
        _candidate("flat", "restoration", correct),
        _candidate("deblock", "restoration", wrong),
        _candidate("otsu", "binary", correct),
        _candidate("upscale", "scale", correct),
    ]
    result = _propose(split)
    assert not result.accepted
    assert result.reason == "no_strict_topology_family_majority"

    invalid = [
        _candidate("primary", "baseline", wrong),
        _candidate("flat", "restoration", correct),
        _candidate("deblock", "restoration", correct, valid=False),
        _candidate("otsu", "binary", correct),
        _candidate("upscale", "scale", correct),
    ]
    result = _propose(invalid)
    assert not result.accepted
    assert result.reason == "no_strict_topology_family_majority"


def test_tie_patch_rejects_cross_measure_pitch_mismatch_and_complex_events() -> None:
    no_tie = [(), (), (), ()]
    dangling = [(), (), (), ("start",)]
    candidates = [
        _candidate("primary", "baseline", no_tie),
        _candidate("flat", "restoration", dangling),
        _candidate("otsu", "binary", dangling),
        _candidate("upscale", "scale", dangling),
    ]
    result = _propose(candidates)
    assert not result.accepted
    assert result.reason == "insufficient_families"

    mismatch = [("start",), ("stop",), (), ()]
    candidates = [
        _candidate("primary", "baseline", no_tie, steps=["C", "D", "E", "F"]),
        _candidate("flat", "restoration", mismatch, steps=["C", "D", "E", "F"]),
        _candidate("otsu", "binary", mismatch, steps=["C", "D", "E", "F"]),
        _candidate("upscale", "scale", mismatch, steps=["C", "D", "E", "F"]),
    ]
    result = _propose(candidates)
    assert not result.accepted
    assert result.reason == "insufficient_families"

    rests = [
        _candidate("primary", "baseline", no_tie, rest_at=1),
        _candidate("flat", "restoration", mismatch, rest_at=1),
        _candidate("otsu", "binary", mismatch, rest_at=1),
        _candidate("upscale", "scale", mismatch, rest_at=1),
    ]
    assert _propose(rests).reason == "insufficient_families"

    chords = [
        _candidate("primary", "baseline", no_tie, chord_at=1),
        _candidate("flat", "restoration", mismatch, chord_at=1),
        _candidate("otsu", "binary", mismatch, chord_at=1),
        _candidate("upscale", "scale", mismatch, chord_at=1),
    ]
    assert _propose(chords).reason == "insufficient_families"


def test_tie_patch_alignment_gap_scope_and_missing_model_fail_closed(tmp_path: Path) -> None:
    wrong = [(), (), (), (), (), ()]
    correct = [("start",), ("stop",), ("start",), ("stop",), ("start",), ("stop",)]
    candidates = [
        _candidate("primary", "baseline", wrong, steps=["C", "C", "D", "D", "E", "E"]),
        _candidate("flat", "restoration", correct, steps=["C", "C", "D", "D", "E", "E"]),
        _candidate("otsu", "binary", correct, steps=["C", "C", "D", "D", "E", "E"]),
        _candidate("upscale", "scale", correct, steps=["C", "C", "D", "D", "E", "E"]),
    ]
    gap = propose_tie_patch(
        candidates,
        template_index=0,
        missing_candidate_count=1,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert not gap.accepted and gap.reason == "alignment_gap"
    result = _propose(candidates)
    assert not result.accepted and result.reason == "change_scope_too_large"

    calibrator = TiePatchCalibrator(tmp_path / "missing.json")
    assert not calibrator.enabled
    simple = [
        _candidate("primary", "baseline", [(), (), (), ()]),
        _candidate("flat", "restoration", [("start",), ("stop",), (), ()]),
        _candidate("otsu", "binary", [("start",), ("stop",), (), ()]),
        _candidate("upscale", "scale", [("start",), ("stop",), (), ()]),
    ]
    result = propose_tie_patch(
        simple,
        template_index=0,
        missing_candidate_count=0,
        calibrator=calibrator,
    )
    assert not result.accepted
    assert result.reason == "model_guard"


def test_bundled_tie_model_is_verified_and_separates_safe_from_conflicting() -> None:
    from dataclasses import replace

    from scorescan.tie_consensus import TiePatchInput

    calibrator = TiePatchCalibrator()
    assert calibrator.enabled
    assert calibrator.model_verified
    assert calibrator.model_status == "verified"
    assert calibrator.model_version == "scorescan-tie-patch-forest-1"

    safe = TiePatchInput(
        candidate_count=6,
        eligible_family_count=4,
        voting_family_count=4,
        changed_endpoint_count=2,
        total_event_count=8,
        added_endpoint_count=2,
        removed_endpoint_count=0,
        changed_pair_count=1,
        winner_family_count=3,
        runner_up_family_count=1,
        template_family_count=1,
        incomplete_family_count=0,
        winner_tie_pair_count=1,
        mean_support_page_probability=0.95,
        mean_support_measure_probability=0.94,
        mean_support_visual_probability=0.92,
        mean_support_event_probability=0.94,
        mean_support_context_probability=0.93,
        mean_support_ensemble_probability=0.95,
        minimum_support_ensemble_probability=0.91,
        mean_support_page_score_margin=30.0,
        mean_support_vs_template_measure_probability=0.20,
        mean_support_vs_template_visual_probability=0.18,
        mean_support_vs_template_event_probability=0.21,
        mean_support_vs_template_context_probability=0.17,
        mean_support_vs_template_ensemble_probability=0.22,
    )
    conflicting = replace(
        safe,
        winner_family_count=2,
        runner_up_family_count=2,
        incomplete_family_count=2,
        mean_support_visual_probability=0.30,
        mean_support_context_probability=0.30,
        mean_support_ensemble_probability=0.40,
        minimum_support_ensemble_probability=0.20,
        mean_support_page_score_margin=-40.0,
        mean_support_vs_template_measure_probability=-0.30,
        mean_support_vs_template_visual_probability=-0.40,
        mean_support_vs_template_event_probability=-0.30,
        mean_support_vs_template_context_probability=-0.40,
        mean_support_vs_template_ensemble_probability=-0.50,
    )
    safe_result = calibrator.calibrate(safe)
    conflict_result = calibrator.calibrate(conflicting)
    assert safe_result.accepted
    assert not conflict_result.accepted
    assert safe_result.probability > conflict_result.probability + 0.90
