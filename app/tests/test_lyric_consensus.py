from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from lxml import etree

from scorescan.lyric_consensus import (
    LyricPatchCalibrator,
    LyricPatchCandidate,
    LyricPatchInput,
    propose_lyric_patch,
)
from scorescan.lyric_xml import LyricState, normalized_lyric_topology, set_lyric_topology
from scorescan.score_ir import measure_from_xml


@dataclass
class AlwaysAccept:
    threshold: float = 0.0
    model_version: str = "test"

    def calibrate(self, item: LyricPatchInput):
        return type("Decision", (), {
            "probability": 1.0,
            "threshold": 0.0,
            "accepted": True,
            "model_version": "test",
        })()


def _measure(
    topology: tuple[LyricState | None, ...],
    *,
    pitch_error_at: int | None = None,
    invalid_at: int | None = None,
    rest_at: int | None = None,
) -> etree._Element:
    measure = etree.Element("measure", number="1")
    attrs = etree.SubElement(measure, "attributes")
    etree.SubElement(attrs, "divisions").text = "4"
    time = etree.SubElement(attrs, "time")
    etree.SubElement(time, "beats").text = str(len(topology))
    etree.SubElement(time, "beat-type").text = "4"
    for index, lyric_state in enumerate(topology):
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
        if lyric_state is not None:
            set_lyric_topology([note], [lyric_state])
        if invalid_at == index:
            lyric = note.find("lyric")
            if lyric is None:
                lyric = etree.SubElement(note, "lyric")
                etree.SubElement(lyric, "text").text = "la"
            etree.SubElement(lyric, "elision").text = "~"
    return measure


def _candidate(
    variant: str,
    family: str,
    topology: tuple[LyricState | None, ...],
    *,
    valid: bool = True,
    pitch_error_at: int | None = None,
    invalid_at: int | None = None,
    rest_at: int | None = None,
) -> LyricPatchCandidate:
    measure = _measure(topology, pitch_error_at=pitch_error_at, invalid_at=invalid_at, rest_at=rest_at)
    semantics, _ = measure_from_xml(measure)
    return LyricPatchCandidate(
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


def _propose(candidates: list[LyricPatchCandidate], *, base_measure: etree._Element | None = None):
    return propose_lyric_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
        base_measure=base_measure,
    )


def test_lyric_patch_adds_replaces_and_removes_plain_lyrics() -> None:
    empty = (None, None, None, None)
    correct = (
        LyricState("Hel", "begin"),
        LyricState("lo", "end"),
        LyricState("world", "single", "start"),
        None,
    )
    candidates = [
        _candidate("primary", "baseline", empty),
        _candidate("flat", "restoration", correct),
        _candidate("otsu", "binary", correct),
        _candidate("upscale", "scale", correct),
    ]
    result = _propose(candidates)
    assert result.accepted
    assert result.changed_event_indices == (0, 1, 2)
    assert result.patched_measure is not None
    assert normalized_lyric_topology(result.patched_measure.findall("note")) == correct

    replacement = (LyricState("Hi", "single"), None, None, None)
    replace_candidates = [
        _candidate("primary", "baseline", correct),
        _candidate("flat", "restoration", replacement),
        _candidate("otsu", "binary", replacement),
        _candidate("upscale", "scale", replacement),
    ]
    replaced = _propose(replace_candidates)
    assert replaced.accepted
    assert replaced.patched_measure is not None
    assert normalized_lyric_topology(replaced.patched_measure.findall("note")) == replacement


def test_lyric_patch_uses_event_local_support_despite_other_pitch_error() -> None:
    empty = (None, None, None, None)
    correct = (None, LyricState("la", "single"), None, None)
    candidates = [
        _candidate("primary", "baseline", empty),
        _candidate("flat", "restoration", correct, pitch_error_at=3),
        _candidate("otsu", "binary", correct, pitch_error_at=2),
        _candidate("upscale", "scale", correct, pitch_error_at=0),
    ]
    result = _propose(candidates)
    assert result.accepted
    assert result.changed_event_indices == (1,)


def test_lyric_patch_split_invalid_or_complex_family_abstains() -> None:
    empty = (None, None, None, None)
    correct = (LyricState("la", "single"), None, None, None)
    split = [
        _candidate("primary", "baseline", empty),
        _candidate("flat", "restoration", correct),
        _candidate("deblock", "restoration", empty),
        _candidate("otsu", "binary", correct),
        _candidate("upscale", "scale", correct),
    ]
    assert _propose(split).reason == "no_lyric_change"

    invalid = [
        _candidate("primary", "baseline", empty),
        _candidate("flat", "restoration", correct),
        _candidate("deblock", "restoration", correct, valid=False),
        _candidate("otsu", "binary", correct),
        _candidate("upscale", "scale", correct),
    ]
    assert _propose(invalid).reason == "no_lyric_change"

    complex_rows = [
        _candidate("primary", "baseline", empty),
        _candidate("flat", "restoration", correct, invalid_at=0),
        _candidate("otsu", "binary", correct, invalid_at=0),
        _candidate("upscale", "scale", correct, invalid_at=0),
    ]
    assert not _propose(complex_rows).accepted


def test_lyric_patch_rejects_rest_and_preserves_child_order() -> None:
    empty = (None, None, None, None)
    correct = (LyricState("la", "single"), None, None, None)
    rests = [
        _candidate("primary", "baseline", empty, rest_at=3),
        _candidate("flat", "restoration", correct, rest_at=3),
        _candidate("otsu", "binary", correct, rest_at=3),
        _candidate("upscale", "scale", correct, rest_at=3),
    ]
    assert _propose(rests).reason == "unsupported_base_measure"

    base = _measure(empty)
    note = base.findall("note")[0]
    etree.SubElement(note, "play")
    candidates = [
        _candidate("primary", "baseline", empty),
        _candidate("flat", "restoration", correct),
        _candidate("otsu", "binary", correct),
        _candidate("upscale", "scale", correct),
    ]
    result = _propose(candidates, base_measure=base)
    assert result.accepted and result.patched_measure is not None
    tags = [child.tag for child in result.patched_measure.findall("note")[0]]
    assert tags.index("lyric") < tags.index("play")


def test_lyric_patch_missing_model_alignment_gap_and_semantics_change_fail_closed(tmp_path: Path) -> None:
    empty = (None, None, None, None)
    correct = (LyricState("la", "single"), None, None, None)
    candidates = [
        _candidate("primary", "baseline", empty),
        _candidate("flat", "restoration", correct),
        _candidate("otsu", "binary", correct),
        _candidate("upscale", "scale", correct),
    ]
    calibrator = LyricPatchCalibrator(tmp_path / "missing.json")
    assert not calibrator.enabled
    assert propose_lyric_patch(candidates, template_index=0, missing_candidate_count=0, calibrator=calibrator).reason == "model_guard"
    assert propose_lyric_patch(candidates, template_index=0, missing_candidate_count=1, calibrator=AlwaysAccept()).reason == "alignment_gap"  # type: ignore[arg-type]

    item = _candidate("flat", "restoration", correct)
    changed = replace(item, semantics=measure_from_xml(_measure(correct, pitch_error_at=0))[0])
    rows = [candidates[0], changed, candidates[2], candidates[3]]
    assert not _propose(rows).accepted
