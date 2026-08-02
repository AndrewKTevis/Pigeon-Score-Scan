from __future__ import annotations

from pathlib import Path

from lxml import etree

from scorescan.cross_tie_consensus import (
    CrossTiePatchCalibration,
    CrossTiePatchCalibrator,
    CrossTiePatchCandidate,
    CrossTiePatchInput,
    propose_cross_tie_patch,
)
from scorescan.score_ir import measure_from_xml


class AlwaysAccept:
    threshold = 0.0
    model_version = "test"

    def calibrate(self, _item: CrossTiePatchInput) -> CrossTiePatchCalibration:
        return CrossTiePatchCalibration(1.0, 0.0, True, self.model_version, 1.0)


def _measure(
    number: int,
    steps: list[str],
    *,
    boundary: str,
    tied: bool,
    direct_only: bool = False,
    notation_only: bool = False,
    incomplete: bool = False,
    chord_boundary: bool = False,
    chord_interior: bool = False,
) -> etree._Element:
    measure = etree.Element("measure", number=str(number))
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"
    count = 3 if incomplete else 4
    for index, step in enumerate(steps[:count]):
        note = etree.SubElement(measure, "note")
        is_boundary = (boundary == "left" and index + 1 == count) or (boundary == "right" and index == 0)
        if (chord_boundary and is_boundary) or (chord_interior and index == 1):
            etree.SubElement(note, "chord")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "1"
        endpoint = "start" if boundary == "left" else "stop"
        if tied and is_boundary and not notation_only:
            etree.SubElement(note, "tie", type=endpoint)
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
        if tied and is_boundary and not direct_only:
            notations = etree.SubElement(note, "notations")
            etree.SubElement(notations, "tied", type=endpoint)
    return measure


def _candidate(
    variant: str,
    family: str,
    tied: bool,
    *,
    valid: bool = True,
    right_first: str = "C",
    direct_only: bool = False,
    notation_only: bool = False,
    incomplete_left: bool = False,
    chord_boundary: bool = False,
    chord_interior: bool = False,
) -> CrossTiePatchCandidate:
    left = _measure(
        2,
        ["G", "A", "B", "C"],
        boundary="left",
        tied=tied,
        direct_only=direct_only,
        notation_only=notation_only,
        incomplete=incomplete_left,
        chord_boundary=chord_boundary,
        chord_interior=chord_interior,
    )
    right = _measure(
        3,
        [right_first, "D", "E", "F"],
        boundary="right",
        tied=tied,
        direct_only=direct_only,
        notation_only=notation_only,
        chord_boundary=chord_boundary,
        chord_interior=chord_interior,
    )
    left_semantics, state = measure_from_xml(left)
    right_semantics, _ = measure_from_xml(right, state)
    return CrossTiePatchCandidate(
        variant=variant,
        family=family,
        left_measure=left,
        right_measure=right,
        left_semantics=left_semantics,
        right_semantics=right_semantics,
        page_score=1000.0,
        page_probability=0.95,
        measure_probability=0.96,
        visual_probability=0.91,
        event_probability=0.96,
        context_probability=0.95,
        ensemble_probability=0.98,
        alignment_similarity=0.99,
        valid=valid,
    )


def _propose(candidates: list[CrossTiePatchCandidate], *, base_tied: bool):
    base = _candidate("base", "base", base_tied)
    return propose_cross_tie_patch(
        candidates,
        template_variant="primary",
        base_left=base.left_measure,
        base_right=base.right_measure,
        base_left_semantics=base.left_semantics,
        base_right_semantics=base.right_semantics,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )


def test_cross_tie_adds_missing_boundary_pair_and_canonicalizes_xml() -> None:
    candidates = [
        _candidate("primary", "baseline", False),
        _candidate("flat", "restoration", True, direct_only=True),
        _candidate("otsu", "binary", True, notation_only=True),
        _candidate("upscale", "scale", True),
    ]
    result = _propose(candidates, base_tied=False)
    assert result.accepted
    assert result.changed_endpoint_count == 2
    assert result.left_measure is not None and result.right_measure is not None
    left = result.left_measure.findall("note")[-1]
    right = result.right_measure.findall("note")[0]
    assert [item.get("type") for item in left.findall("tie")] == ["start"]
    assert [item.get("type") for item in left.findall("./notations/tied")] == ["start"]
    assert [item.get("type") for item in right.findall("tie")] == ["stop"]
    assert [item.get("type") for item in right.findall("./notations/tied")] == ["stop"]


def test_cross_tie_removes_spurious_boundary_pair() -> None:
    candidates = [
        _candidate("primary", "baseline", True),
        _candidate("flat", "restoration", False),
        _candidate("otsu", "binary", False),
        _candidate("upscale", "scale", False),
    ]
    result = _propose(candidates, base_tied=True)
    assert result.accepted
    assert result.left_measure is not None and result.right_measure is not None
    assert not result.left_measure.findall("note")[-1].findall("tie")
    assert not result.right_measure.findall("note")[0].findall("tie")


def test_cross_tie_invalid_or_split_sibling_makes_family_abstain() -> None:
    split = [
        _candidate("primary", "baseline", False),
        _candidate("flat", "restoration", True),
        _candidate("deblock", "restoration", False),
        _candidate("otsu", "binary", True),
        _candidate("upscale", "scale", True),
    ]
    result = _propose(split, base_tied=False)
    assert not result.accepted
    assert result.reason == "no_strict_boundary_family_majority"

    invalid = [
        _candidate("primary", "baseline", False),
        _candidate("flat", "restoration", True),
        _candidate("deblock", "restoration", True, valid=False),
        _candidate("otsu", "binary", True),
        _candidate("upscale", "scale", True),
    ]
    result = _propose(invalid, base_tied=False)
    assert not result.accepted
    assert result.reason == "no_strict_boundary_family_majority"


def test_cross_tie_rejects_pitch_meter_and_complex_boundary_disagreement() -> None:
    mismatch = [
        _candidate("primary", "baseline", False),
        _candidate("flat", "restoration", True, right_first="D"),
        _candidate("otsu", "binary", True, right_first="D"),
        _candidate("upscale", "scale", True, right_first="D"),
    ]
    assert _propose(mismatch, base_tied=False).reason == "insufficient_boundary_family_votes"

    incomplete = [
        _candidate("primary", "baseline", False),
        _candidate("flat", "restoration", True, incomplete_left=True),
        _candidate("otsu", "binary", True, incomplete_left=True),
        _candidate("upscale", "scale", True, incomplete_left=True),
    ]
    assert _propose(incomplete, base_tied=False).reason == "insufficient_boundary_family_votes"

    chords = [
        _candidate("primary", "baseline", False),
        _candidate("flat", "restoration", True, chord_boundary=True),
        _candidate("otsu", "binary", True, chord_boundary=True),
        _candidate("upscale", "scale", True, chord_boundary=True),
    ]
    assert _propose(chords, base_tied=False).reason == "insufficient_boundary_family_votes"


def test_cross_tie_missing_model_fails_closed(tmp_path: Path) -> None:
    candidates = [
        _candidate("primary", "baseline", False),
        _candidate("flat", "restoration", True),
        _candidate("otsu", "binary", True),
        _candidate("upscale", "scale", True),
    ]
    base = _candidate("base", "base", False)
    calibrator = CrossTiePatchCalibrator(tmp_path / "missing.json")
    result = propose_cross_tie_patch(
        candidates,
        template_variant="primary",
        base_left=base.left_measure,
        base_right=base.right_measure,
        base_left_semantics=base.left_semantics,
        base_right_semantics=base.right_semantics,
        calibrator=calibrator,
    )
    assert not result.accepted
    assert result.reason == "model_guard"
    assert not calibrator.enabled


def test_cross_tie_rejects_complex_structure_away_from_boundary() -> None:
    candidates = [
        _candidate("primary", "baseline", False, chord_interior=True),
        _candidate("flat", "restoration", True, chord_interior=True),
        _candidate("otsu", "binary", True, chord_interior=True),
        _candidate("upscale", "scale", True, chord_interior=True),
    ]
    result = _propose(candidates, base_tied=False)
    assert not result.accepted
    assert result.reason == "insufficient_boundary_family_votes"
