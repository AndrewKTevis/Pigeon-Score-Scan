from __future__ import annotations

from dataclasses import dataclass

from lxml import etree

from scorescan.event_kind_consensus import (
    EventKindPatchCalibration,
    EventKindPatchCalibrator,
    EventKindPatchCandidate,
    EventKindPatchInput,
    propose_event_kind_patch,
)
from scorescan.score_ir import measure_from_xml


@dataclass
class AlwaysAccept:
    threshold: float = 0.95
    model_version: str = "test-event-kind-patch"

    def calibrate(self, item: EventKindPatchInput) -> EventKindPatchCalibration:
        return EventKindPatchCalibration(1.0, self.threshold, True, self.model_version, 1.0)


def _measure(kinds: list[str], *, steps: list[str] | None = None, chord_at: int | None = None) -> etree._Element:
    measure = etree.Element("measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"
    steps = steps or ["C", "D", "E", "F"][: len(kinds)]
    for index, kind in enumerate(kinds):
        note = etree.SubElement(measure, "note")
        if chord_at == index:
            etree.SubElement(note, "chord")
        if kind == "rest":
            etree.SubElement(note, "rest")
        else:
            pitch = etree.SubElement(note, "pitch")
            etree.SubElement(pitch, "step").text = steps[index]
            etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "1"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
    return measure


def _candidate(
    variant: str,
    family: str,
    kinds: list[str],
    *,
    steps: list[str] | None = None,
    chord_at: int | None = None,
    valid: bool = True,
) -> EventKindPatchCandidate:
    measure = _measure(kinds, steps=steps, chord_at=chord_at)
    semantics, _ = measure_from_xml(measure)
    return EventKindPatchCandidate(
        variant=variant,
        family=family,
        measure=measure,
        semantics=semantics,
        page_score=1000.0,
        page_probability=0.92,
        measure_probability=0.93,
        visual_probability=0.88,
        event_probability=0.95,
        context_probability=0.90,
        ensemble_probability=0.96,
        valid=valid,
    )


def test_event_kind_consensus_repairs_false_rest() -> None:
    candidates = [
        _candidate("primary", "baseline", ["rest", "pitch", "pitch", "pitch"]),
        _candidate("flat", "restoration", ["pitch", "pitch", "pitch", "pitch"]),
        _candidate("otsu", "binary", ["pitch", "pitch", "pitch", "pitch"]),
        _candidate("upscale", "scale", ["pitch", "pitch", "pitch", "pitch"]),
    ]
    result = propose_event_kind_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert result.accepted
    assert result.changed_event_indices == (0,)
    assert result.patched_measure is not None
    assert result.patched_measure.find("./note/pitch") is not None
    parsed, _ = measure_from_xml(result.patched_measure)
    assert not parsed.notes[0].rest
    assert parsed.notes[0].pitch is not None
    assert parsed.notes[0].pitch.step == "C"


def test_event_kind_consensus_repairs_false_note_to_rest() -> None:
    candidates = [
        _candidate("primary", "baseline", ["pitch", "pitch", "pitch", "pitch"]),
        _candidate("flat", "restoration", ["rest", "pitch", "pitch", "pitch"]),
        _candidate("otsu", "binary", ["rest", "pitch", "pitch", "pitch"]),
        _candidate("upscale", "scale", ["rest", "pitch", "pitch", "pitch"]),
    ]
    result = propose_event_kind_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert result.accepted
    assert result.patched_measure is not None
    first = result.patched_measure.find("./note")
    assert first is not None and first.find("rest") is not None and first.find("pitch") is None


def test_event_kind_consensus_collapses_correlated_sibling_votes() -> None:
    candidates = [
        _candidate("primary", "baseline", ["rest", "pitch", "pitch", "pitch"]),
        _candidate("flat", "restoration", ["pitch", "pitch", "pitch", "pitch"]),
        _candidate("deblock", "restoration", ["pitch", "pitch", "pitch", "pitch"]),
        _candidate("otsu", "binary", ["rest", "pitch", "pitch", "pitch"]),
        _candidate("upscale", "scale", ["rest", "pitch", "pitch", "pitch"]),
    ]
    result = propose_event_kind_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert not result.accepted
    assert result.reason == "no_event_kind_change"


def test_event_kind_consensus_invalid_sibling_makes_family_abstain() -> None:
    candidates = [
        _candidate("primary", "baseline", ["rest", "pitch", "pitch", "pitch"]),
        _candidate("flat", "restoration", ["pitch", "pitch", "pitch", "pitch"]),
        _candidate("deblock", "restoration", ["pitch", "pitch", "pitch", "pitch"], valid=False),
        _candidate("otsu", "binary", ["pitch", "pitch", "pitch", "pitch"]),
        _candidate("upscale", "scale", ["pitch", "pitch", "pitch", "pitch"]),
    ]
    result = propose_event_kind_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert not result.accepted
    assert result.reason == "no_strict_event_kind_family_majority"


def test_event_kind_consensus_leaves_pitch_only_disagreement_to_pitch_consensus() -> None:
    candidates = [
        _candidate("primary", "baseline", ["pitch"] * 4, steps=["C", "D", "F", "F"]),
        _candidate("flat", "restoration", ["pitch"] * 4, steps=["C", "D", "E", "F"]),
        _candidate("otsu", "binary", ["pitch"] * 4, steps=["C", "D", "E", "F"]),
        _candidate("upscale", "scale", ["pitch"] * 4, steps=["C", "D", "E", "F"]),
    ]
    result = propose_event_kind_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert not result.accepted
    assert result.reason == "no_event_kind_change"


def test_event_kind_consensus_rejects_complex_event_structure() -> None:
    candidates = [
        _candidate("primary", "baseline", ["rest", "pitch", "pitch", "pitch"], chord_at=1),
        _candidate("flat", "restoration", ["pitch", "pitch", "pitch", "pitch"], chord_at=1),
        _candidate("otsu", "binary", ["pitch", "pitch", "pitch", "pitch"], chord_at=1),
        _candidate("upscale", "scale", ["pitch", "pitch", "pitch", "pitch"], chord_at=1),
    ]
    result = propose_event_kind_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert not result.accepted
    assert result.reason == "unsupported_event_structure"


def test_event_kind_patch_composes_onto_existing_attribute_base() -> None:
    candidates = [
        _candidate("primary", "baseline", ["rest", "pitch", "pitch", "pitch"]),
        _candidate("flat", "restoration", ["pitch", "pitch", "pitch", "pitch"]),
        _candidate("otsu", "binary", ["pitch", "pitch", "pitch", "pitch"]),
        _candidate("upscale", "scale", ["pitch", "pitch", "pitch", "pitch"]),
    ]
    base = _measure(["rest", "pitch", "pitch", "pitch"])
    base.findtext("./attributes/time/beats")
    beats = base.find("./attributes/time/beats")
    assert beats is not None
    beats.text = "5"
    result = propose_event_kind_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
        base_measure=base,
    )
    assert result.accepted
    assert result.patched_measure is not None
    assert result.patched_measure.findtext("./attributes/time/beats") == "5"


def test_event_kind_patch_model_is_verified_and_rejects_conflict() -> None:
    calibrator = EventKindPatchCalibrator()
    assert calibrator.enabled
    assert calibrator.model_verified
    assert calibrator.model_version == "scorescan-event-kind-patch-forest-1"

    base = dict(
        candidate_count=6,
        eligible_family_count=4,
        voting_family_count=4,
        changed_event_count=1,
        total_event_count=8,
        minimum_winner_family_support_ratio=0.75,
        mean_winner_family_support_ratio=0.78,
        minimum_winner_margin_ratio=0.50,
        mean_winner_margin_ratio=0.52,
        maximum_template_family_support_ratio=0.20,
        family_abstention_ratio=0.02,
        pitched_winner_count=1,
        rest_winner_count=0,
        minimum_pitched_winner_support_ratio=0.75,
        mean_pitched_winner_support_ratio=0.78,
        mean_support_page_probability=0.90,
        mean_support_measure_probability=0.92,
        mean_support_visual_probability=0.87,
        mean_support_event_probability=0.94,
        mean_support_context_probability=0.88,
        mean_support_ensemble_probability=0.95,
        minimum_support_ensemble_probability=0.86,
        mean_support_page_score_margin=18.0,
        mean_support_vs_template_measure_probability=0.16,
        mean_support_vs_template_visual_probability=0.18,
        mean_support_vs_template_event_probability=0.20,
        mean_support_vs_template_context_probability=0.12,
        mean_support_vs_template_ensemble_probability=0.22,
    )
    favourable = calibrator.predict_probability(EventKindPatchInput(**base))
    conflict = calibrator.predict_probability(EventKindPatchInput(**{
        **base,
        "mean_support_visual_probability": 0.16,
        "mean_support_context_probability": 0.20,
        "mean_support_vs_template_visual_probability": -0.32,
        "mean_support_vs_template_context_probability": -0.24,
        "mean_support_vs_template_ensemble_probability": -0.05,
    }))
    assert favourable > conflict


def test_event_kind_patch_missing_model_fails_closed(tmp_path) -> None:
    calibrator = EventKindPatchCalibrator(tmp_path / "missing.json")
    assert not calibrator.enabled
    candidates = [
        _candidate("primary", "baseline", ["rest", "pitch", "pitch", "pitch"]),
        _candidate("flat", "restoration", ["pitch", "pitch", "pitch", "pitch"]),
        _candidate("otsu", "binary", ["pitch", "pitch", "pitch", "pitch"]),
        _candidate("upscale", "scale", ["pitch", "pitch", "pitch", "pitch"]),
    ]
    result = propose_event_kind_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=calibrator,
    )
    assert not result.accepted
    assert result.reason == "model_guard"
