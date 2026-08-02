from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lxml import etree

from scorescan.grace_consensus import (
    GracePatchCalibrator,
    GracePatchCandidate,
    GracePatchInput,
    propose_grace_patch,
)
from scorescan.grace_xml import normalized_grace_topology
from scorescan.score_ir import measure_from_xml


@dataclass
class AlwaysAccept:
    threshold: float = 0.0
    model_version: str = "test"

    def calibrate(self, item: GracePatchInput):
        return type("Decision", (), {
            "probability": 1.0,
            "threshold": 0.0,
            "accepted": True,
            "model_version": "test",
        })()


def _measure(
    count: int,
    grace_indices: set[int],
    *,
    divisions: int = 4,
    invalid_grace_at: int | None = None,
    pitch_error_at: int | None = None,
    chord_at: int | None = None,
) -> etree._Element:
    measure = etree.Element("measure", number="1")
    attrs = etree.SubElement(measure, "attributes")
    etree.SubElement(attrs, "divisions").text = str(divisions)
    time = etree.SubElement(attrs, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"
    for index in range(count):
        note = etree.SubElement(measure, "note")
        if index in grace_indices:
            grace = etree.SubElement(note, "grace")
            if invalid_grace_at == index:
                grace.set("slash", "yes")
        if chord_at == index:
            etree.SubElement(note, "chord")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = "G" if pitch_error_at == index else "CDEFGAB"[index % 7]
        etree.SubElement(pitch, "octave").text = "4"
        if index not in grace_indices:
            etree.SubElement(note, "duration").text = str(divisions)
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
    return measure


def _candidate(
    variant: str,
    family: str,
    count: int,
    grace_indices: set[int],
    *,
    divisions: int = 4,
    valid: bool = True,
    invalid_grace_at: int | None = None,
    pitch_error_at: int | None = None,
    chord_at: int | None = None,
) -> GracePatchCandidate:
    measure = _measure(
        count,
        grace_indices,
        divisions=divisions,
        invalid_grace_at=invalid_grace_at,
        pitch_error_at=pitch_error_at,
        chord_at=chord_at,
    )
    semantics, _ = measure_from_xml(measure)
    return GracePatchCandidate(
        variant=variant,
        family=family,
        measure=measure,
        semantics=semantics,
        page_score=1000.0,
        page_probability=0.97,
        measure_probability=0.96,
        visual_probability=0.93,
        event_probability=0.97,
        context_probability=0.94,
        ensemble_probability=0.98,
        valid=valid,
    )


def _propose(candidates: list[GracePatchCandidate], base_measure: etree._Element | None = None):
    return propose_grace_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
        base_measure=base_measure,
    )


def test_grace_patch_converts_overfull_regular_note_to_grace() -> None:
    candidates = [
        _candidate("primary", "baseline", 5, set()),
        _candidate("flat", "restoration", 5, {0}),
        _candidate("otsu", "binary", 5, {0}),
        _candidate("upscale", "scale", 5, {0}),
    ]
    result = _propose(candidates)
    assert result.accepted
    assert result.changed_event_indices == (0,)
    assert result.added_grace_count == 1
    assert result.removed_grace_count == 0
    assert result.patched_measure is not None
    parsed, _ = measure_from_xml(result.patched_measure)
    assert parsed.notes[0].grace
    assert parsed.notes[0].duration == 0
    assert sum(note.duration for note in parsed.notes if not note.grace) == 4


def test_grace_patch_converts_underfull_grace_to_regular_across_divisions() -> None:
    candidates = [
        _candidate("primary", "baseline", 4, {0}, divisions=4),
        _candidate("flat", "restoration", 4, set(), divisions=8),
        _candidate("otsu", "binary", 4, set(), divisions=4),
        _candidate("upscale", "scale", 4, set(), divisions=8),
    ]
    result = _propose(candidates)
    assert result.accepted
    assert result.removed_grace_count == 1
    assert result.patched_measure is not None
    first = result.patched_measure.findall("note")[0]
    assert first.find("grace") is None
    assert first.findtext("duration") == "4"
    parsed, _ = measure_from_xml(result.patched_measure)
    assert sum(note.duration for note in parsed.notes if not note.grace) == 4


def test_grace_patch_allows_complementary_pitch_errors_away_from_target() -> None:
    candidates = [
        _candidate("primary", "baseline", 5, set()),
        _candidate("flat", "restoration", 5, {0}, pitch_error_at=4),
        _candidate("otsu", "binary", 5, {0}, pitch_error_at=3),
        _candidate("upscale", "scale", 5, {0}, pitch_error_at=2),
    ]
    result = _propose(candidates)
    assert result.accepted


def test_grace_patch_rejects_target_pitch_error_and_split_family() -> None:
    target_error = [
        _candidate("primary", "baseline", 5, set()),
        _candidate("flat", "restoration", 5, {0}, pitch_error_at=0),
        _candidate("otsu", "binary", 5, {0}, pitch_error_at=0),
        _candidate("upscale", "scale", 5, {0}, pitch_error_at=0),
    ]
    assert not _propose(target_error).accepted

    split = [
        _candidate("primary", "baseline", 5, set()),
        _candidate("flat", "restoration", 5, {0}),
        _candidate("deblock", "restoration", 5, set()),
        _candidate("otsu", "binary", 5, {0}),
        _candidate("upscale", "scale", 5, {0}),
    ]
    assert not _propose(split).accepted


def test_grace_patch_invalid_sibling_and_complex_grace_fail_closed() -> None:
    invalid = [
        _candidate("primary", "baseline", 5, set()),
        _candidate("flat", "restoration", 5, {0}),
        _candidate("deblock", "restoration", 5, {0}, valid=False),
        _candidate("otsu", "binary", 5, {0}),
        _candidate("upscale", "scale", 5, {0}),
    ]
    assert not _propose(invalid).accepted

    complex_rows = [
        _candidate("primary", "baseline", 5, set()),
        _candidate("flat", "restoration", 5, {0}, invalid_grace_at=0),
        _candidate("otsu", "binary", 5, {0}, invalid_grace_at=0),
        _candidate("upscale", "scale", 5, {0}, invalid_grace_at=0),
    ]
    assert not _propose(complex_rows).accepted

    chord = [
        _candidate("primary", "baseline", 5, set(), chord_at=1),
        _candidate("flat", "restoration", 5, {0}, chord_at=1),
        _candidate("otsu", "binary", 5, {0}, chord_at=1),
        _candidate("upscale", "scale", 5, {0}, chord_at=1),
    ]
    assert _propose(chord).reason == "unsupported_base_measure"


def test_grace_patch_preserves_unrelated_xml_and_child_order() -> None:
    base = _measure(5, set())
    lyric = etree.SubElement(base.findall("note")[0], "lyric")
    etree.SubElement(lyric, "text").text = "la"
    candidates = [
        _candidate("primary", "baseline", 5, set()),
        _candidate("flat", "restoration", 5, {0}),
        _candidate("otsu", "binary", 5, {0}),
        _candidate("upscale", "scale", 5, {0}),
    ]
    result = _propose(candidates, base)
    assert result.accepted
    assert result.patched_measure is not None
    note = result.patched_measure.findall("note")[0]
    tags = [child.tag for child in note]
    assert tags[0] == "grace"
    assert tags.index("type") < tags.index("lyric")
    assert note.findtext("./lyric/text") == "la"
    topology = normalized_grace_topology(result.patched_measure.findall("note"), 4)
    assert topology is not None and topology[0].grace


def test_grace_patch_missing_model_and_alignment_gap_fail_closed(tmp_path: Path) -> None:
    candidates = [
        _candidate("primary", "baseline", 5, set()),
        _candidate("flat", "restoration", 5, {0}),
        _candidate("otsu", "binary", 5, {0}),
        _candidate("upscale", "scale", 5, {0}),
    ]
    calibrator = GracePatchCalibrator(tmp_path / "missing.json")
    assert not calibrator.enabled
    result = propose_grace_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=calibrator,
    )
    assert result.reason == "model_guard"
    gap = propose_grace_patch(
        candidates,
        template_index=0,
        missing_candidate_count=1,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert gap.reason == "alignment_gap"
