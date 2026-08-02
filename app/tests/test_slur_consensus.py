from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from lxml import etree

from scorescan.score_ir import measure_from_xml
from scorescan.slur_consensus import (
    SlurPatchCalibrator,
    SlurPatchCandidate,
    SlurPatchInput,
    propose_slur_patch,
)


@dataclass
class AlwaysAccept:
    threshold: float = 0.0
    model_version: str = "test"

    def calibrate(self, item: SlurPatchInput):
        return type("Decision", (), {
            "probability": 1.0,
            "threshold": 0.0,
            "accepted": True,
            "model_version": "test",
        })()


def _measure(arcs: tuple[tuple[int, int], ...], *, event_count: int = 6, rest_at: int | None = None,
             chord_at: int | None = None, tie_at: int | None = None,
             steps: str | None = None) -> etree._Element:
    measure = etree.Element("measure", number="1")
    attrs = etree.SubElement(measure, "attributes")
    etree.SubElement(attrs, "divisions").text = "4"
    time = etree.SubElement(attrs, "time")
    etree.SubElement(time, "beats").text = "6"
    etree.SubElement(time, "beat-type").text = "4"
    for index in range(event_count):
        note = etree.SubElement(measure, "note")
        if chord_at == index:
            etree.SubElement(note, "chord")
        if rest_at == index:
            etree.SubElement(note, "rest")
        else:
            pitch = etree.SubElement(note, "pitch")
            pitch_steps = steps or "CDEFGAB"
            etree.SubElement(pitch, "step").text = pitch_steps[index % len(pitch_steps)]
            etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "4"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
        if tie_at == index:
            etree.SubElement(note, "tie", type="start")
            notations = etree.SubElement(note, "notations")
            etree.SubElement(notations, "tied", type="start")
    for number, (start, stop) in enumerate(arcs, start=1):
        for index, kind in ((start, "start"), (stop, "stop")):
            note = measure.findall("note")[index]
            notations = note.find("notations")
            if notations is None:
                notations = etree.SubElement(note, "notations")
            etree.SubElement(notations, "slur", type=kind, number=str(number))
    return measure


def _candidate(variant: str, family: str, arcs: tuple[tuple[int, int], ...], *, valid: bool = True,
               event_count: int = 6, rest_at: int | None = None, chord_at: int | None = None,
               tie_at: int | None = None, steps: str | None = None) -> SlurPatchCandidate:
    measure = _measure(
        arcs, event_count=event_count, rest_at=rest_at, chord_at=chord_at,
        tie_at=tie_at, steps=steps
    )
    semantics, _ = measure_from_xml(measure)
    return SlurPatchCandidate(
        variant=variant,
        family=family,
        measure=measure,
        semantics=semantics,
        page_score=1000.0,
        page_probability=0.95,
        measure_probability=0.95,
        visual_probability=0.92,
        event_probability=0.95,
        context_probability=0.94,
        ensemble_probability=0.97,
        valid=valid,
    )


def _propose(candidates: list[SlurPatchCandidate], *, base_measure: etree._Element | None = None):
    return propose_slur_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
        base_measure=base_measure,
    )


def test_slur_patch_adds_missing_arc_and_preserves_unrelated_notation() -> None:
    correct = ((0, 4),)
    candidates = [
        _candidate("primary", "baseline", ()),
        _candidate("flat", "restoration", correct),
        _candidate("otsu", "binary", correct),
        _candidate("upscale", "scale", correct),
    ]
    base = _measure(())
    notations = etree.SubElement(base.findall("note")[2], "notations")
    articulations = etree.SubElement(notations, "articulations")
    etree.SubElement(articulations, "staccato")
    # Candidate structure does not contain this articulation, so fail closed.
    rejected = _propose(candidates, base_measure=base)
    assert not rejected.accepted
    assert rejected.reason == "non_slur_structure_disagreement"

    result = _propose(candidates)
    assert result.accepted
    assert result.changed_event_indices == (0, 4)
    assert result.changed_arc_count == 1
    assert result.patched_measure is not None
    slurs = [
        (index, item.get("type"), item.get("number"))
        for index, note in enumerate(result.patched_measure.findall("note"))
        for item in note.findall("./notations/slur")
    ]
    assert slurs == [(0, "start", "1"), (4, "stop", "1")]


def test_slur_patch_removes_spurious_arc_and_normalizes_candidate_numbering() -> None:
    wrong = ((0, 4),)
    correct: tuple[tuple[int, int], ...] = ()
    candidates = [
        _candidate("primary", "baseline", wrong),
        _candidate("flat", "restoration", correct),
        _candidate("otsu", "binary", correct),
        _candidate("upscale", "scale", correct),
    ]
    result = _propose(candidates)
    assert result.accepted
    assert result.patched_measure is not None
    assert not result.patched_measure.findall(".//slur")

    # Different MusicXML slur numbers still describe the same topology.
    candidate = _candidate("flat", "restoration", ((1, 5),))
    for item in candidate.measure.findall(".//slur"):
        item.set("number", "6")
    semantics, _ = measure_from_xml(candidate.measure)
    candidate = replace(candidate, semantics=semantics)
    same = [
        _candidate("primary", "baseline", ()),
        candidate,
        _candidate("otsu", "binary", ((1, 5),)),
        _candidate("upscale", "scale", ((1, 5),)),
    ]
    result = _propose(same)
    assert result.accepted


def test_slur_patch_split_or_invalid_sibling_makes_family_abstain() -> None:
    correct = ((0, 4),)
    split = [
        _candidate("primary", "baseline", ()),
        _candidate("flat", "restoration", correct),
        _candidate("deblock", "restoration", ()),
        _candidate("otsu", "binary", correct),
        _candidate("upscale", "scale", correct),
    ]
    result = _propose(split)
    assert not result.accepted
    assert result.reason == "no_strict_topology_family_majority"

    invalid = [
        _candidate("primary", "baseline", ()),
        _candidate("flat", "restoration", correct),
        _candidate("deblock", "restoration", correct, valid=False),
        _candidate("otsu", "binary", correct),
        _candidate("upscale", "scale", correct),
    ]
    assert _propose(invalid).reason == "no_strict_topology_family_majority"


def test_slur_patch_rejects_overlap_complex_events_and_alignment_gap() -> None:
    overlapping = ((0, 4), (2, 5))
    candidates = [
        _candidate("primary", "baseline", ()),
        _candidate("flat", "restoration", overlapping),
        _candidate("otsu", "binary", overlapping),
        _candidate("upscale", "scale", overlapping),
    ]
    assert _propose(candidates).reason == "insufficient_families"

    for kwargs in ({"rest_at": 2}, {"chord_at": 2}, {"tie_at": 2}):
        items = [
            _candidate("primary", "baseline", (), **kwargs),
            _candidate("flat", "restoration", ((0, 4),), **kwargs),
            _candidate("otsu", "binary", ((0, 4),), **kwargs),
            _candidate("upscale", "scale", ((0, 4),), **kwargs),
        ]
        assert _propose(items).reason == "insufficient_families"

    gap = propose_slur_patch(
        [
            _candidate("primary", "baseline", ()),
            _candidate("flat", "restoration", ((0, 4),)),
            _candidate("otsu", "binary", ((0, 4),)),
            _candidate("upscale", "scale", ((0, 4),)),
        ],
        template_index=0,
        missing_candidate_count=1,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert gap.reason == "alignment_gap"


def test_slur_patch_change_scope_and_missing_model_fail_closed(tmp_path: Path) -> None:
    too_many = ((0, 1), (2, 3), (4, 5))
    candidates = [
        _candidate("primary", "baseline", (), event_count=8),
        _candidate("flat", "restoration", too_many, event_count=8),
        _candidate("otsu", "binary", too_many, event_count=8),
        _candidate("upscale", "scale", too_many, event_count=8),
    ]
    assert _propose(candidates).reason == "insufficient_families"

    calibrator = SlurPatchCalibrator(tmp_path / "missing.json")
    assert not calibrator.enabled
    simple = [
        _candidate("primary", "baseline", ()),
        _candidate("flat", "restoration", ((0, 4),)),
        _candidate("otsu", "binary", ((0, 4),)),
        _candidate("upscale", "scale", ((0, 4),)),
    ]
    result = propose_slur_patch(
        simple,
        template_index=0,
        missing_candidate_count=0,
        calibrator=calibrator,
    )
    assert not result.accepted
    assert result.reason == "model_guard"


def test_slur_patch_abstains_on_adjacent_same_pitch_tie_ambiguity() -> None:
    ambiguous = ((0, 1),)
    candidates = [
        _candidate("primary", "baseline", (), steps="CCDEFG"),
        _candidate("flat", "restoration", ambiguous, steps="CCDEFG"),
        _candidate("otsu", "binary", ambiguous, steps="CCDEFG"),
        _candidate("upscale", "scale", ambiguous, steps="CCDEFG"),
    ]
    result = _propose(candidates)
    assert not result.accepted
    assert result.reason == "insufficient_families"


def test_slur_writer_preserves_musicxml_note_child_order() -> None:
    from scorescan.slur_xml import set_slur_topology

    measure = _measure(())
    notes = measure.findall("note")
    lyric = etree.SubElement(notes[0], "lyric")
    etree.SubElement(lyric, "text").text = "la"
    set_slur_topology(notes, ((0, 4),))

    tags = [child.tag for child in notes[0]]
    assert tags.index("notations") < tags.index("lyric")
    assert notes[0].findtext("./lyric/text") == "la"


def test_slur_patch_rejects_incomplete_measure() -> None:
    correct = ((0, 3),)
    candidates = [
        _candidate("primary", "baseline", (), event_count=4),
        _candidate("flat", "restoration", correct, event_count=4),
        _candidate("otsu", "binary", correct, event_count=4),
        _candidate("upscale", "scale", correct, event_count=4),
    ]
    result = _propose(candidates)
    assert not result.accepted
    assert result.reason == "insufficient_families"
