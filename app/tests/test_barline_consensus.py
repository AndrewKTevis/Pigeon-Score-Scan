from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lxml import etree

from scorescan.barline_consensus import (
    BarlinePatchCalibrator,
    BarlinePatchCandidate,
    BarlinePatchInput,
    propose_barline_patch,
)
from scorescan.score_ir import measure_from_xml


@dataclass
class AlwaysAccept:
    threshold: float = 0.0
    model_version: str = "test"

    def calibrate(self, item: BarlinePatchInput):
        return type("Decision", (), {
            "probability": 1.0,
            "threshold": 0.0,
            "accepted": True,
            "model_version": "test",
        })()


def _measure(
    topology: tuple[tuple[str, str, str], ...],
    *,
    ending: bool = False,
    fermata: bool = False,
) -> etree._Element:
    measure = etree.Element("measure", number="1")
    attrs = etree.SubElement(measure, "attributes")
    etree.SubElement(attrs, "divisions").text = "4"
    time = etree.SubElement(attrs, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"
    for step in "CDEF":
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "4"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
    for location, style, repeat in topology:
        barline = etree.SubElement(measure, "barline", location=location)
        if style:
            etree.SubElement(barline, "bar-style").text = style
        if ending:
            etree.SubElement(barline, "ending", number="1", type="start")
        if fermata:
            etree.SubElement(barline, "fermata").text = "normal"
        if repeat:
            etree.SubElement(barline, "repeat", direction=repeat)
    return measure


def _candidate(
    variant: str,
    family: str,
    topology: tuple[tuple[str, str, str], ...],
    *,
    valid: bool = True,
    ending: bool = False,
    fermata: bool = False,
) -> BarlinePatchCandidate:
    measure = _measure(topology, ending=ending, fermata=fermata)
    semantics, _ = measure_from_xml(measure)
    return BarlinePatchCandidate(
        variant=variant,
        family=family,
        measure=measure,
        semantics=semantics,
        page_score=1000.0,
        page_probability=0.96,
        measure_probability=0.95,
        visual_probability=0.90,
        event_probability=0.94,
        context_probability=0.95,
        ensemble_probability=0.97,
        valid=valid,
    )


def _propose(candidates: list[BarlinePatchCandidate], *, base_measure: etree._Element | None = None):
    return propose_barline_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
        base_measure=base_measure,
    )


def test_barline_patch_adds_forward_repeat_and_preserves_notes() -> None:
    none: tuple[tuple[str, str, str], ...] = ()
    repeat = (("left", "heavy-light", "forward"),)
    candidates = [
        _candidate("primary", "baseline", none),
        _candidate("flat", "restoration", repeat),
        _candidate("otsu", "binary", repeat),
        _candidate("upscale", "scale", repeat),
    ]
    base = _measure(none)
    accidental = etree.Element("accidental")
    accidental.text = "sharp"
    base.findall("note")[1].insert(1, accidental)
    result = _propose(candidates, base_measure=base)
    assert result.accepted
    assert result.changed_locations == ("left",)
    assert result.changed_repeat_count == 1
    assert result.patched_measure is not None
    repeat_node = result.patched_measure.find("./barline[@location='left']/repeat")
    assert repeat_node is not None and repeat_node.get("direction") == "forward"
    assert result.patched_measure.findtext("./note[2]/accidental") == "sharp"


def test_barline_patch_removes_spurious_backward_repeat() -> None:
    wrong = (("right", "light-heavy", "backward"),)
    correct = (("right", "light-heavy", ""),)
    candidates = [
        _candidate("primary", "baseline", wrong),
        _candidate("flat", "restoration", correct),
        _candidate("otsu", "binary", correct),
        _candidate("upscale", "scale", correct),
    ]
    result = _propose(candidates)
    assert result.accepted
    assert result.patched_measure is not None
    assert result.patched_measure.find("./barline/repeat") is None
    assert result.patched_measure.findtext("./barline/bar-style") == "light-heavy"


def test_barline_patch_family_split_and_invalid_sibling_abstain() -> None:
    repeat = (("right", "light-heavy", "backward"),)
    none: tuple[tuple[str, str, str], ...] = ()
    split = [
        _candidate("primary", "baseline", none),
        _candidate("flat", "restoration", repeat),
        _candidate("deblock", "restoration", none),
        _candidate("otsu", "binary", repeat),
        _candidate("upscale", "scale", repeat),
    ]
    assert _propose(split).reason == "no_strict_barline_family_majority"

    invalid = [
        _candidate("primary", "baseline", none),
        _candidate("flat", "restoration", repeat),
        _candidate("deblock", "restoration", repeat, valid=False),
        _candidate("otsu", "binary", repeat),
        _candidate("upscale", "scale", repeat),
    ]
    assert _propose(invalid).reason == "no_strict_barline_family_majority"


def test_barline_patch_rejects_complex_navigation_and_invalid_direction() -> None:
    repeat = (("right", "light-heavy", "backward"),)
    candidates = [
        _candidate("primary", "baseline", ()),
        _candidate("flat", "restoration", repeat, ending=True),
        _candidate("otsu", "binary", repeat, ending=True),
        _candidate("upscale", "scale", repeat, ending=True),
    ]
    assert _propose(candidates).reason == "insufficient_barline_family_votes"

    invalid_direction = (("left", "heavy-light", "backward"),)
    candidates = [
        _candidate("primary", "baseline", ()),
        _candidate("flat", "restoration", invalid_direction),
        _candidate("otsu", "binary", invalid_direction),
        _candidate("upscale", "scale", invalid_direction),
    ]
    assert _propose(candidates).reason == "insufficient_barline_family_votes"


def test_barline_patch_missing_model_and_alignment_gap_fail_closed(tmp_path: Path) -> None:
    repeat = (("right", "light-heavy", "backward"),)
    candidates = [
        _candidate("primary", "baseline", ()),
        _candidate("flat", "restoration", repeat),
        _candidate("otsu", "binary", repeat),
        _candidate("upscale", "scale", repeat),
    ]
    calibrator = BarlinePatchCalibrator(tmp_path / "missing.json")
    assert not calibrator.enabled
    result = propose_barline_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=calibrator,
    )
    assert not result.accepted and result.reason == "model_guard"

    gap = propose_barline_patch(
        candidates,
        template_index=0,
        missing_candidate_count=1,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert gap.reason == "alignment_gap"


def test_barline_patch_writer_keeps_measure_child_order() -> None:
    repeat = (("right", "light-heavy", "backward"),)
    candidates = [
        _candidate("primary", "baseline", ()),
        _candidate("flat", "restoration", repeat),
        _candidate("otsu", "binary", repeat),
        _candidate("upscale", "scale", repeat),
    ]
    base = _measure(())
    grouping = etree.SubElement(base, "grouping", type="start")
    result = _propose(candidates, base_measure=base)
    assert result.accepted and result.patched_measure is not None
    tags = [child.tag for child in result.patched_measure]
    assert tags.index("barline") < tags.index("grouping")
    assert grouping.get("type") == "start"
