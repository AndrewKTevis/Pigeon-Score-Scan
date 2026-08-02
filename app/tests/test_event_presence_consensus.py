from __future__ import annotations

from dataclasses import dataclass

from lxml import etree

from scorescan.event_presence_consensus import (
    EventPresencePatchCalibration,
    EventPresencePatchCalibrator,
    EventPresencePatchCandidate,
    EventPresencePatchInput,
    propose_event_presence_patch,
)
from scorescan.score_ir import measure_from_xml


@dataclass
class AlwaysAccept:
    threshold: float = 0.97
    model_version: str = "test-event-presence"

    def calibrate(self, item: EventPresencePatchInput) -> EventPresencePatchCalibration:
        return EventPresencePatchCalibration(1.0, self.threshold, True, self.model_version, 1.0)


def _measure(
    steps: list[str],
    *,
    divisions: int = 1,
    rest_at: int | None = None,
    beam_at: int | None = None,
) -> etree._Element:
    measure = etree.Element("measure", number="2")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = str(divisions)
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"
    for index, step in enumerate(steps):
        note = etree.SubElement(measure, "note")
        if rest_at == index:
            etree.SubElement(note, "rest")
        else:
            pitch = etree.SubElement(note, "pitch")
            etree.SubElement(pitch, "step").text = step
            etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = str(divisions)
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
        if beam_at == index:
            beam = etree.SubElement(note, "beam", number="1")
            beam.text = "begin"
    return measure


def _candidate(
    variant: str,
    family: str,
    steps: list[str],
    *,
    divisions: int = 1,
    rest_at: int | None = None,
    beam_at: int | None = None,
    valid: bool = True,
) -> EventPresencePatchCandidate:
    measure = _measure(steps, divisions=divisions, rest_at=rest_at, beam_at=beam_at)
    semantics, _ = measure_from_xml(measure)
    return EventPresencePatchCandidate(
        variant=variant,
        family=family,
        measure=measure,
        semantics=semantics,
        page_score=1000.0,
        page_probability=0.93,
        measure_probability=0.94,
        visual_probability=0.90,
        event_probability=0.95,
        context_probability=0.92,
        ensemble_probability=0.96,
        valid=valid,
    )


def _propose(candidates: list[EventPresencePatchCandidate], **kwargs: object):
    return propose_event_presence_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        is_first_measure=False,
        is_last_measure=False,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
        **kwargs,
    )


def test_event_presence_inserts_uniquely_anchored_missing_note() -> None:
    candidates = [
        _candidate("primary", "baseline", ["C", "D", "F"]),
        _candidate("flat", "restoration", ["C", "D", "E", "F"]),
        _candidate("otsu", "binary", ["C", "D", "E", "F"]),
        _candidate("upscale", "scale", ["C", "D", "E", "F"]),
    ]
    result = _propose(candidates)
    assert result.accepted
    assert result.operation == "insert"
    assert result.changed_event_indices == (2,)
    assert result.patched_measure is not None
    assert result.patched_measure.findall("note")[2].findtext("pitch/step") == "E"
    parsed, _ = measure_from_xml(result.patched_measure)
    assert [note.pitch.step if note.pitch else "rest" for note in parsed.notes] == ["C", "D", "E", "F"]
    assert sum(note.duration for note in parsed.notes) == 4


def test_event_presence_deletes_uniquely_anchored_extra_note() -> None:
    candidates = [
        _candidate("primary", "baseline", ["C", "D", "E", "F", "G"]),
        _candidate("flat", "restoration", ["C", "D", "E", "G"]),
        _candidate("otsu", "binary", ["C", "D", "E", "G"]),
        _candidate("upscale", "scale", ["C", "D", "E", "G"]),
    ]
    result = _propose(candidates)
    assert result.accepted
    assert result.operation == "delete"
    assert result.changed_event_indices == (3,)
    assert result.patched_measure is not None
    assert [node.text for node in result.patched_measure.findall("note/pitch/step")] == ["C", "D", "E", "G"]


def test_event_presence_reencodes_inserted_duration_in_template_divisions() -> None:
    candidates = [
        _candidate("primary", "baseline", ["C", "D", "F"], divisions=2),
        _candidate("flat", "restoration", ["C", "D", "E", "F"], divisions=4),
        _candidate("otsu", "binary", ["C", "D", "E", "F"], divisions=4),
        _candidate("upscale", "scale", ["C", "D", "E", "F"], divisions=4),
    ]
    result = _propose(candidates)
    assert result.accepted
    assert result.patched_measure is not None
    assert result.patched_measure.findall("note")[2].findtext("duration") == "2"


def test_event_presence_preserves_existing_base_edits() -> None:
    candidates = [
        _candidate("primary", "baseline", ["C", "D", "F"]),
        _candidate("flat", "restoration", ["C", "D", "E", "F"]),
        _candidate("otsu", "binary", ["C", "D", "E", "F"]),
        _candidate("upscale", "scale", ["C", "D", "E", "F"]),
    ]
    base = _measure(["C", "B", "F"])
    key = etree.SubElement(base.find("attributes"), "key")
    etree.SubElement(key, "fifths").text = "2"
    result = _propose(candidates, base_measure=base)
    assert result.accepted
    assert result.patched_measure is not None
    assert [node.text for node in result.patched_measure.findall("note/pitch/step")] == ["C", "B", "E", "F"]
    assert result.patched_measure.findtext("attributes/key/fifths") == "2"


def test_event_presence_invalid_sibling_makes_family_abstain() -> None:
    candidates = [
        _candidate("primary", "baseline", ["C", "D", "F"]),
        _candidate("flat", "restoration", ["C", "D", "E", "F"]),
        _candidate("deblock", "restoration", ["C", "D", "E", "F"], valid=False),
        _candidate("otsu", "binary", ["C", "D", "E", "F"]),
        _candidate("upscale", "scale", ["C", "D", "E", "F"]),
    ]
    result = _propose(candidates)
    assert not result.accepted
    assert result.reason == "no_strict_topology_family_majority"


def test_event_presence_rejects_ambiguous_repeated_pattern() -> None:
    candidates = [
        _candidate("primary", "baseline", ["C", "C", "C"]),
        _candidate("flat", "restoration", ["C", "C", "C", "C"]),
        _candidate("otsu", "binary", ["C", "C", "C", "C"]),
        _candidate("upscale", "scale", ["C", "C", "C", "C"]),
    ]
    result = _propose(candidates)
    assert not result.accepted
    assert result.reason == "not_unique_single_event_edit"


def test_event_presence_rejects_edge_measure_and_complex_beam() -> None:
    simple = [
        _candidate("primary", "baseline", ["C", "D", "F"]),
        _candidate("flat", "restoration", ["C", "D", "E", "F"]),
        _candidate("otsu", "binary", ["C", "D", "E", "F"]),
        _candidate("upscale", "scale", ["C", "D", "E", "F"]),
    ]
    edge = propose_event_presence_patch(
        simple,
        template_index=0,
        missing_candidate_count=0,
        is_first_measure=True,
        is_last_measure=False,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert not edge.accepted and edge.reason == "edge_measure"

    complex_candidates = [
        _candidate("primary", "baseline", ["C", "D", "F"], beam_at=0),
        _candidate("flat", "restoration", ["C", "D", "E", "F"], beam_at=0),
        _candidate("otsu", "binary", ["C", "D", "E", "F"], beam_at=0),
        _candidate("upscale", "scale", ["C", "D", "E", "F"], beam_at=0),
    ]
    complex_result = _propose(complex_candidates)
    assert not complex_result.accepted and complex_result.reason == "unsupported_event_structure"


def test_event_presence_missing_model_fails_closed(tmp_path) -> None:
    calibrator = EventPresencePatchCalibrator(tmp_path / "missing.json")
    assert not calibrator.enabled
    candidates = [
        _candidate("primary", "baseline", ["C", "D", "F"]),
        _candidate("flat", "restoration", ["C", "D", "E", "F"]),
        _candidate("otsu", "binary", ["C", "D", "E", "F"]),
        _candidate("upscale", "scale", ["C", "D", "E", "F"]),
    ]
    result = propose_event_presence_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        is_first_measure=False,
        is_last_measure=False,
        calibrator=calibrator,
    )
    assert not result.accepted
    assert result.reason == "model_guard"


def test_event_presence_model_is_verified_and_rejects_conflict() -> None:
    calibrator = EventPresencePatchCalibrator()
    assert calibrator.enabled
    assert calibrator.model_verified
    assert calibrator.model_version == "scorescan-event-presence-patch-forest-1"

    base = dict(
        candidate_count=6,
        eligible_family_count=4,
        winning_family_count=3,
        runner_up_family_count=1,
        incomplete_family_count=0,
        operation="insert",
        template_event_count=7,
        winner_event_count=8,
        edit_index=3,
        anchor_match_ratio=0.95,
        anchor_margin_ratio=0.50,
        inserted_content_family_count=3,
        inserted_content_runner_up_count=0,
        inserted_event_is_rest=False,
        inserted_event_duration=0.5,
        expected_measure_duration=4.0,
        template_duration_error=0.5,
        patched_duration_error=0.0,
        mean_support_page_probability=0.91,
        mean_support_measure_probability=0.93,
        mean_support_visual_probability=0.90,
        mean_support_event_probability=0.94,
        mean_support_context_probability=0.91,
        mean_support_ensemble_probability=0.95,
        minimum_support_ensemble_probability=0.87,
        mean_support_page_score_margin=18.0,
        mean_support_vs_template_ensemble_probability=0.18,
    )
    favourable = calibrator.predict_probability(EventPresencePatchInput(**base))
    conflict = calibrator.predict_probability(EventPresencePatchInput(**{
        **base,
        "anchor_match_ratio": 0.68,
        "anchor_margin_ratio": 0.05,
        "mean_support_visual_probability": 0.18,
        "mean_support_context_probability": 0.22,
        "mean_support_ensemble_probability": 0.52,
        "minimum_support_ensemble_probability": 0.24,
        "mean_support_page_score_margin": -24.0,
        "mean_support_vs_template_ensemble_probability": -0.16,
    }))
    assert favourable > conflict
