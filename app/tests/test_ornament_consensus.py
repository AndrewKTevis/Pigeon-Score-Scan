from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from lxml import etree

from scorescan.ornament_consensus import (
    OrnamentPatchCalibrator,
    OrnamentPatchCandidate,
    OrnamentPatchInput,
    propose_ornament_patch,
)
from scorescan.ornament_xml import normalized_ornament_topology, set_ornament_topology
from scorescan.score_ir import measure_from_xml


@dataclass
class AlwaysAccept:
    threshold: float = 0.0
    model_version: str = "test"

    def calibrate(self, item: OrnamentPatchInput):
        return type("Decision", (), {
            "probability": 1.0,
            "threshold": 0.0,
            "accepted": True,
            "model_version": "test",
        })()


def _measure(
    topology: tuple[tuple[str, ...], ...],
    *,
    pitch_error_at: int | None = None,
    unsupported_at: int | None = None,
    rest_at: int | None = None,
) -> etree._Element:
    measure = etree.Element("measure", number="1")
    attrs = etree.SubElement(measure, "attributes")
    etree.SubElement(attrs, "divisions").text = "4"
    time = etree.SubElement(attrs, "time")
    etree.SubElement(time, "beats").text = str(len(topology))
    etree.SubElement(time, "beat-type").text = "4"
    for index, marks in enumerate(topology):
        note = etree.SubElement(measure, "note")
        if rest_at == index:
            etree.SubElement(note, "rest")
        else:
            pitch = etree.SubElement(note, "pitch")
            etree.SubElement(pitch, "step").text = "G" if pitch_error_at == index else "CDEFGAB"[index % 7]
            etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "4"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
        if marks or unsupported_at == index:
            notations = etree.SubElement(note, "notations")
            ornaments = etree.SubElement(notations, "ornaments")
            for mark in marks:
                etree.SubElement(ornaments, mark)
            if unsupported_at == index:
                etree.SubElement(ornaments, "wavy-line", type="start", number="1")
    return measure


def _candidate(
    variant: str,
    family: str,
    topology: tuple[tuple[str, ...], ...],
    *,
    valid: bool = True,
    pitch_error_at: int | None = None,
    unsupported_at: int | None = None,
    rest_at: int | None = None,
) -> OrnamentPatchCandidate:
    measure = _measure(
        topology,
        pitch_error_at=pitch_error_at,
        unsupported_at=unsupported_at,
        rest_at=rest_at,
    )
    semantics, _ = measure_from_xml(measure)
    return OrnamentPatchCandidate(
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


def _propose(candidates: list[OrnamentPatchCandidate], *, base_measure: etree._Element | None = None):
    return propose_ornament_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
        base_measure=base_measure,
    )


def test_ornament_patch_adds_and_removes_simple_markers() -> None:
    empty = ((), (), (), ())
    correct = (("trill-mark",), (), (), ())
    candidates = [
        _candidate("primary", "baseline", empty),
        _candidate("flat", "restoration", correct),
        _candidate("otsu", "binary", correct),
        _candidate("upscale", "scale", correct),
    ]
    result = _propose(candidates)
    assert result.accepted
    assert result.changed_event_indices == (0,)
    assert result.changed_mark_count == 1
    assert result.patched_measure is not None
    assert normalized_ornament_topology(result.patched_measure.findall("note")) == correct

    remove = [
        _candidate("primary", "baseline", correct),
        _candidate("flat", "restoration", empty),
        _candidate("otsu", "binary", empty),
        _candidate("upscale", "scale", empty),
    ]
    removed = _propose(remove)
    assert removed.accepted
    assert removed.patched_measure is not None
    assert normalized_ornament_topology(removed.patched_measure.findall("note")) == empty


def test_ornament_patch_uses_event_local_support_despite_other_pitch_error() -> None:
    empty = ((), (), (), ())
    correct = ((), ("inverted-mordent",), (), ())
    candidates = [
        _candidate("primary", "baseline", empty),
        _candidate("flat", "restoration", correct, pitch_error_at=3),
        _candidate("otsu", "binary", correct, pitch_error_at=2),
        _candidate("upscale", "scale", correct, pitch_error_at=0),
    ]
    result = _propose(candidates)
    assert result.accepted
    assert result.changed_event_indices == (1,)


def test_ornament_patch_split_or_invalid_sibling_abstains() -> None:
    empty = ((), (), (), ())
    correct = (("mordent",), (), (), ())
    split = [
        _candidate("primary", "baseline", empty),
        _candidate("flat", "restoration", correct),
        _candidate("deblock", "restoration", empty),
        _candidate("otsu", "binary", correct),
        _candidate("upscale", "scale", correct),
    ]
    assert _propose(split).reason == "no_ornament_change"

    invalid = [
        _candidate("primary", "baseline", empty),
        _candidate("flat", "restoration", correct),
        _candidate("deblock", "restoration", correct, valid=False),
        _candidate("otsu", "binary", correct),
        _candidate("upscale", "scale", correct),
    ]
    assert _propose(invalid).reason == "no_ornament_change"


def test_ornament_patch_rejects_unsupported_or_complex_measure() -> None:
    empty = ((), (), (), ())
    correct = (("mordent",), (), (), ())
    unsupported = [
        _candidate("primary", "baseline", empty),
        _candidate("flat", "restoration", correct, unsupported_at=2),
        _candidate("otsu", "binary", correct, unsupported_at=2),
        _candidate("upscale", "scale", correct, unsupported_at=2),
    ]
    assert not _propose(unsupported).accepted

    rests = [
        _candidate("primary", "baseline", empty, rest_at=3),
        _candidate("flat", "restoration", correct, rest_at=3),
        _candidate("otsu", "binary", correct, rest_at=3),
        _candidate("upscale", "scale", correct, rest_at=3),
    ]
    assert _propose(rests).reason == "unsupported_base_measure"


def test_ornament_patch_preserves_other_notation_and_child_order() -> None:
    empty = ((), (), (), ())
    correct = (("turn",), (), (), ())
    base = _measure(empty)
    lyric = etree.SubElement(base.findall("note")[0], "lyric")
    etree.SubElement(lyric, "text").text = "la"
    candidates = [
        _candidate("primary", "baseline", empty),
        _candidate("flat", "restoration", correct),
        _candidate("otsu", "binary", correct),
        _candidate("upscale", "scale", correct),
    ]
    result = _propose(candidates, base_measure=base)
    assert result.accepted
    assert result.patched_measure is not None
    note = result.patched_measure.findall("note")[0]
    tags = [child.tag for child in note]
    assert tags.index("notations") < tags.index("lyric")
    assert note.findtext("./lyric/text") == "la"

    notes = _measure(empty).findall("note")
    set_ornament_topology(notes, correct)
    assert normalized_ornament_topology(notes) == correct


def test_ornament_patch_missing_model_and_alignment_gap_fail_closed(tmp_path: Path) -> None:
    empty = ((), (), (), ())
    correct = (("mordent",), (), (), ())
    candidates = [
        _candidate("primary", "baseline", empty),
        _candidate("flat", "restoration", correct),
        _candidate("otsu", "binary", correct),
        _candidate("upscale", "scale", correct),
    ]
    calibrator = OrnamentPatchCalibrator(tmp_path / "missing.json")
    assert not calibrator.enabled
    result = propose_ornament_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=calibrator,
    )
    assert result.reason == "model_guard"
    gap = propose_ornament_patch(
        candidates,
        template_index=0,
        missing_candidate_count=1,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert gap.reason == "alignment_gap"


def test_ornament_patch_detects_candidate_semantics_change() -> None:
    empty = ((), (), (), ())
    correct = (("mordent",), (), (), ())
    item = _candidate("flat", "restoration", correct)
    changed = replace(item, semantics=measure_from_xml(_measure(correct, pitch_error_at=0))[0])
    candidates = [
        _candidate("primary", "baseline", empty),
        changed,
        _candidate("otsu", "binary", correct),
        _candidate("upscale", "scale", correct),
    ]
    assert not _propose(candidates).accepted
