from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lxml import etree

from scorescan.pitch_consensus import (
    PitchPatchCalibration,
    PitchPatchCalibrator,
    PitchPatchCandidate,
    PitchPatchInput,
    PitchVisualGuard,
    propose_pitch_patch,
)
from scorescan.score_ir import measure_from_xml
from scorescan.visual_evidence import VisualMeasureEvidence, semantic_event_grids


@dataclass
class AlwaysAccept:
    threshold: float = 0.9
    model_version: str = "test-pitch-patch"

    def calibrate(self, item: PitchPatchInput) -> PitchPatchCalibration:
        return PitchPatchCalibration(1.0, self.threshold, True, self.model_version, 1.0)


@dataclass
class AlwaysReject:
    threshold: float = 1.0
    model_version: str = "test-reject-pitch-patch"

    def calibrate(self, item: PitchPatchInput) -> PitchPatchCalibration:
        return PitchPatchCalibration(0.0, self.threshold, False, self.model_version, 1.0)


def _measure(pitches: list[str], durations: list[int] | None = None) -> etree._Element:
    measure = etree.Element("measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"
    durations = durations or [1] * len(pitches)
    for step, duration in zip(pitches, durations, strict=True):
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = str(duration)
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
    return measure


def _candidate(
    variant: str,
    family: str,
    pitches: list[str],
    *,
    durations: list[int] | None = None,
    valid: bool = True,
) -> PitchPatchCandidate:
    measure = _measure(pitches, durations)
    semantics, _ = measure_from_xml(measure)
    return PitchPatchCandidate(
        variant=variant,
        family=family,
        measure=measure,
        semantics=semantics,
        page_score=1000.0,
        page_probability=0.90,
        measure_probability=0.90,
        visual_probability=0.82,
        event_probability=0.92,
        context_probability=0.82,
        ensemble_probability=0.94,
        valid=valid,
    )


def test_pitch_consensus_recovers_complementary_family_errors() -> None:
    candidates = [
        _candidate("primary", "baseline", ["C", "D", "F"]),
        _candidate("flat", "restoration", ["C", "E", "E"]),
        _candidate("otsu", "binary", ["B", "D", "E"]),
        _candidate("upscale", "scale", ["C", "D", "E"]),
    ]
    result = propose_pitch_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert result.accepted
    assert result.changed_event_indices == (2,)
    assert result.reason == "accepted"
    assert result.patched_measure is not None
    assert [item.text for item in result.patched_measure.findall("./note/pitch/step")] == ["C", "D", "E"]
    parsed, _ = measure_from_xml(result.patched_measure)
    assert [note.duration for note in parsed.notes] == [note.duration for note in candidates[0].semantics.notes]


def test_pitch_consensus_collapses_correlated_sibling_votes() -> None:
    candidates = [
        _candidate("primary", "baseline", ["C"]),
        _candidate("flat", "restoration", ["E"]),
        _candidate("deblock", "restoration", ["E"]),
        _candidate("otsu", "binary", ["F"]),
        _candidate("upscale", "scale", ["G"]),
    ]
    result = propose_pitch_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert not result.accepted
    assert result.reason == "no_strict_event_family_majority"



def test_pitch_consensus_invalid_sibling_makes_family_abstain() -> None:
    candidates = [
        _candidate("primary", "baseline", ["F"]),
        _candidate("flat", "restoration", ["E"]),
        _candidate("deblock", "restoration", ["E"], valid=False),
        _candidate("otsu", "binary", ["E"]),
        _candidate("upscale", "scale", ["E"]),
    ]
    result = propose_pitch_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert not result.accepted
    assert result.reason == "no_strict_event_family_majority"

def test_pitch_consensus_rejects_non_pitch_structure_disagreement() -> None:
    candidates = [
        _candidate("primary", "baseline", ["C", "F"], durations=[1, 1]),
        _candidate("flat", "restoration", ["C", "E"], durations=[1, 2]),
        _candidate("otsu", "binary", ["C", "E"], durations=[1, 1]),
        _candidate("upscale", "scale", ["C", "E"], durations=[1, 1]),
    ]
    result = propose_pitch_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert not result.accepted
    assert result.reason == "non_pitch_structure_disagreement"


def _pitch_input_values() -> dict[str, object]:
    return dict(
        candidate_count=6,
        eligible_family_count=4,
        voting_family_count=4,
        changed_event_count=2,
        total_event_count=8,
        minimum_winner_family_support_ratio=0.75,
        mean_winner_family_support_ratio=0.78,
        minimum_winner_margin_ratio=0.50,
        mean_winner_margin_ratio=0.52,
        maximum_template_family_support_ratio=0.20,
        family_abstention_ratio=0.02,
        mean_support_page_probability=0.90,
        mean_support_measure_probability=0.92,
        mean_support_visual_probability=0.82,
        mean_support_event_probability=0.93,
        mean_support_context_probability=0.82,
        mean_support_ensemble_probability=0.95,
        minimum_support_ensemble_probability=0.86,
        mean_support_page_score_margin=18.0,
        mean_support_vs_template_measure_probability=0.18,
        mean_support_vs_template_visual_probability=0.16,
        mean_support_vs_template_event_probability=0.20,
        mean_support_vs_template_context_probability=0.12,
        mean_support_vs_template_ensemble_probability=0.22,
        visual_evidence_available=True,
        changed_staff_position_ratio=1.0,
        maximum_staff_position_delta=2.0,
        accidental_only_change_ratio=0.0,
        notehead_exact_cell_improvement=0.22,
        notehead_near_cell_improvement=0.25,
        notehead_vertical_chamfer_improvement=0.20,
        notehead_severe_vertical_improvement=0.24,
        notehead_visual_unmatched_improvement=0.12,
        notehead_column_centroid_improvement=0.18,
        notehead_column_order_improvement=0.10,
        template_notehead_exact_cell_gap=0.42,
        template_notehead_near_cell_gap=0.38,
        template_notehead_vertical_chamfer_gap=0.34,
        template_notehead_severe_vertical_gap=0.30,
        template_notehead_visual_unmatched_gap=0.28,
        template_notehead_column_centroid_gap=0.32,
        template_notehead_column_order_gap=0.24,
        proposal_notehead_exact_cell_gap=0.20,
        proposal_notehead_near_cell_gap=0.13,
        proposal_notehead_vertical_chamfer_gap=0.14,
        proposal_notehead_severe_vertical_gap=0.06,
        proposal_notehead_visual_unmatched_gap=0.16,
        proposal_notehead_column_centroid_gap=0.14,
        proposal_notehead_column_order_gap=0.14,
        strict_notehead_exact_cell_improvement=0.20,
        strict_notehead_near_cell_improvement=0.23,
        strict_notehead_vertical_chamfer_improvement=0.18,
        strict_notehead_severe_vertical_improvement=0.22,
        strict_notehead_visual_unmatched_improvement=0.10,
        strict_notehead_column_centroid_improvement=0.16,
        strict_notehead_column_order_improvement=0.08,
        template_strict_notehead_exact_cell_gap=0.40,
        template_strict_notehead_near_cell_gap=0.36,
        template_strict_notehead_vertical_chamfer_gap=0.32,
        template_strict_notehead_severe_vertical_gap=0.28,
        template_strict_notehead_visual_unmatched_gap=0.26,
        template_strict_notehead_column_centroid_gap=0.30,
        template_strict_notehead_column_order_gap=0.22,
        proposal_strict_notehead_exact_cell_gap=0.20,
        proposal_strict_notehead_near_cell_gap=0.13,
        proposal_strict_notehead_vertical_chamfer_gap=0.14,
        proposal_strict_notehead_severe_vertical_gap=0.06,
        proposal_strict_notehead_visual_unmatched_gap=0.16,
        proposal_strict_notehead_column_centroid_gap=0.14,
        proposal_strict_notehead_column_order_gap=0.14,
    )


def test_pitch_patch_model_is_verified_and_separates_visual_conflict() -> None:
    calibrator = PitchPatchCalibrator()
    assert calibrator.enabled
    assert calibrator.model_verified
    assert calibrator.model_version == "scorescan-pitch-patch-forest-4"

    base = _pitch_input_values()
    favourable = calibrator.predict_probability(PitchPatchInput(**base))
    conflict = calibrator.predict_probability(PitchPatchInput(**{
        **base,
        "mean_support_visual_probability": 0.20,
        "mean_support_vs_template_visual_probability": -0.30,
        "mean_support_vs_template_context_probability": -0.10,
        "mean_support_vs_template_ensemble_probability": 0.01,
        "notehead_exact_cell_improvement": -0.24,
        "notehead_near_cell_improvement": -0.28,
        "notehead_vertical_chamfer_improvement": -0.22,
        "notehead_severe_vertical_improvement": -0.30,
        "template_notehead_exact_cell_gap": 0.18,
        "template_notehead_near_cell_gap": 0.14,
        "template_notehead_vertical_chamfer_gap": 0.12,
        "template_notehead_severe_vertical_gap": 0.08,
        "proposal_notehead_exact_cell_gap": 0.52,
        "proposal_notehead_near_cell_gap": 0.46,
        "proposal_notehead_vertical_chamfer_gap": 0.42,
        "proposal_notehead_severe_vertical_gap": 0.38,
    }))
    assert favourable > conflict


def test_pitch_patch_calibrator_uses_separate_no_visual_threshold() -> None:
    calibrator = PitchPatchCalibrator()
    visual_item = PitchPatchInput(**_pitch_input_values())
    no_visual_item = PitchPatchInput(**{
        **_pitch_input_values(),
        "visual_evidence_available": False,
    })

    visual = calibrator.calibrate(visual_item)
    no_visual = calibrator.calibrate(no_visual_item)

    assert visual.threshold == round(calibrator.threshold, 6)
    assert no_visual.threshold == round(calibrator.no_visual_threshold, 6)
    assert no_visual.threshold == 0.7
    assert visual.threshold > no_visual.threshold


def test_pitch_visual_guard_is_verified_and_selective() -> None:
    guard = PitchVisualGuard()
    assert guard.enabled
    assert guard.model_verified
    assert guard.model_version == "scorescan-pitch-visual-guard-2"

    favourable = PitchPatchInput(**{
        **_pitch_input_values(),
        "notehead_exact_cell_improvement": 0.38,
        "notehead_near_cell_improvement": 0.42,
        "notehead_vertical_chamfer_improvement": 0.34,
        "notehead_severe_vertical_improvement": 0.46,
        "template_notehead_near_cell_gap": 0.58,
        "template_notehead_severe_vertical_gap": 0.52,
        "proposal_notehead_near_cell_gap": 0.16,
        "proposal_notehead_severe_vertical_gap": 0.06,
        "strict_notehead_exact_cell_improvement": 0.35,
        "strict_notehead_near_cell_improvement": 0.39,
        "strict_notehead_vertical_chamfer_improvement": 0.31,
        "strict_notehead_severe_vertical_improvement": 0.43,
        "template_strict_notehead_near_cell_gap": 0.55,
        "template_strict_notehead_severe_vertical_gap": 0.49,
        "proposal_strict_notehead_near_cell_gap": 0.16,
        "proposal_strict_notehead_severe_vertical_gap": 0.06,
    })
    conflicting = PitchPatchInput(**{
        **_pitch_input_values(),
        "notehead_exact_cell_improvement": -0.32,
        "notehead_near_cell_improvement": -0.38,
        "notehead_vertical_chamfer_improvement": -0.30,
        "notehead_severe_vertical_improvement": -0.42,
        "template_notehead_near_cell_gap": 0.12,
        "template_notehead_severe_vertical_gap": 0.04,
        "proposal_notehead_near_cell_gap": 0.50,
        "proposal_notehead_severe_vertical_gap": 0.46,
        "strict_notehead_exact_cell_improvement": -0.29,
        "strict_notehead_near_cell_improvement": -0.35,
        "strict_notehead_vertical_chamfer_improvement": -0.27,
        "strict_notehead_severe_vertical_improvement": -0.39,
        "template_strict_notehead_near_cell_gap": 0.12,
        "template_strict_notehead_severe_vertical_gap": 0.04,
        "proposal_strict_notehead_near_cell_gap": 0.47,
        "proposal_strict_notehead_severe_vertical_gap": 0.43,
    })
    assert guard.predict_probability(favourable) > guard.predict_probability(conflicting)


def test_pitch_visual_guard_missing_model_fails_closed(tmp_path: Path) -> None:
    candidates = [
        _candidate("primary", "baseline", ["C"]),
        _candidate("flat", "restoration", ["G"]),
        _candidate("otsu", "binary", ["G"]),
        _candidate("upscale", "scale", ["G"]),
    ]
    proposal_grid = semantic_event_grids(candidates[1].semantics)["pitched_notehead_grid"]
    evidence = VisualMeasureEvidence(
        1, 1, 1, (0, 0, 200, 100), 10.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        (0.0,) * 8, (0.0,) * 9, pitched_notehead_grid=proposal_grid,
    )
    result = propose_pitch_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
        visual_guard=PitchVisualGuard(tmp_path / "missing.json"),
        visual_evidence=evidence,
    )
    assert not result.accepted
    assert result.reason == "visual_model_guard"


def test_pitch_consensus_rejects_strong_source_crop_conflict() -> None:
    candidates = [
        _candidate("primary", "baseline", ["C"]),
        _candidate("flat", "restoration", ["G"]),
        _candidate("otsu", "binary", ["G"]),
        _candidate("upscale", "scale", ["G"]),
    ]
    source_grid = semantic_event_grids(candidates[0].semantics)["pitched_notehead_grid"]
    evidence = VisualMeasureEvidence(
        1, 1, 1, (0, 0, 200, 100), 10.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        (0.0,) * 8, (0.0,) * 9, pitched_notehead_grid=source_grid,
    )
    result = propose_pitch_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
        visual_evidence=evidence,
    )
    assert not result.accepted
    assert result.reason == "visual_pitch_conflict"
    assert result.changed_event_indices == (0,)


def test_pitch_consensus_preserves_prior_chord_topology_patch() -> None:
    candidates = [
        _candidate("primary", "baseline", ["C", "D", "F", "F", "G"]),
        _candidate("flat", "restoration", ["C", "E", "E", "F", "G"]),
        _candidate("otsu", "binary", ["C", "E", "E", "F", "G"]),
        _candidate("upscale", "scale", ["C", "E", "E", "F", "G"]),
    ]
    # A preceding chord-topology repair has established that event 2 belongs to the
    # first onset.  Pitch voting must use that base without restoring the old marker.
    base = _measure(["C", "D", "F", "F", "G"])
    base.findall("note")[1].insert(0, etree.Element("chord"))
    result = propose_pitch_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
        base_measure=base,
    )
    assert result.accepted
    assert result.patched_measure is not None
    notes = result.patched_measure.findall("note")
    assert notes[1].find("chord") is not None
    assert [node.text for node in result.patched_measure.findall("note/pitch/step")] == ["C", "E", "E", "F", "G"]


def _accidental_candidate(
    variant: str,
    family: str,
    *,
    accidental: str,
    alter: int,
    include_rest: bool = False,
) -> PitchPatchCandidate:
    measure = _measure(["C"])
    pitch = measure.find("./note/pitch")
    assert pitch is not None
    if alter:
        etree.SubElement(pitch, "alter").text = str(alter)
    if accidental:
        note = measure.find("note")
        assert note is not None
        etree.SubElement(note, "accidental").text = accidental
    if include_rest:
        rest = etree.SubElement(measure, "note")
        etree.SubElement(rest, "rest")
        etree.SubElement(rest, "duration").text = "1"
        etree.SubElement(rest, "voice").text = "1"
        etree.SubElement(rest, "type").text = "quarter"
    semantics, _ = measure_from_xml(measure)
    return PitchPatchCandidate(
        variant=variant,
        family=family,
        measure=measure,
        semantics=semantics,
        page_score=1000.0,
        page_probability=0.90,
        measure_probability=0.90,
        visual_probability=0.82,
        event_probability=0.92,
        context_probability=0.82,
        ensemble_probability=0.94,
        valid=True,
    )


@dataclass
class AlwaysAcceptAccidentalPresence:
    accepted: bool = True

    def calibrate(self, evidence, measure, event_index, *, expected_present):
        from scorescan.accidental_presence_guard import AccidentalPresenceCalibration

        return AccidentalPresenceCalibration(
            probability=0.99 if expected_present else 0.01,
            expected_present=expected_present,
            confidence=0.99,
            threshold=0.90,
            accepted=self.accepted,
            model_version="test-accidental-presence",
        )


def _blank_visual_evidence() -> VisualMeasureEvidence:
    return VisualMeasureEvidence(
        1, 1, 1, (0, 0, 256, 96), 10.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        (0.0,) * 8, (0.0,) * 9,
        symbol_guard_image="declared-source-crop",
    )


def test_pitch_consensus_rejects_new_accidental_state_inconsistency() -> None:
    candidates = [
        _accidental_candidate("primary", "baseline", accidental="", alter=0),
        _accidental_candidate("flat", "restoration", accidental="", alter=1),
        _accidental_candidate("otsu", "binary", accidental="", alter=1),
        _accidental_candidate("upscale", "scale", accidental="", alter=1),
    ]
    result = propose_pitch_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert not result.accepted
    assert result.reason == "accidental_state_regression"


def test_pitch_consensus_repairs_implicit_sharp_with_invariant_rest() -> None:
    """A rest must not disable a strict accidental majority on pitched events."""

    candidates = [
        _accidental_candidate(
            "primary",
            "baseline",
            accidental="",
            alter=1,
            include_rest=True,
        ),
        _accidental_candidate(
            "flat",
            "restoration",
            accidental="",
            alter=0,
            include_rest=True,
        ),
        _accidental_candidate(
            "otsu",
            "binary",
            accidental="",
            alter=0,
            include_rest=True,
        ),
        _accidental_candidate(
            "upscale",
            "scale",
            accidental="",
            alter=0,
            include_rest=True,
        ),
    ]

    result = propose_pitch_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )

    assert result.accepted
    assert result.changed_event_indices == (0,)
    assert result.patched_measure is not None
    assert result.patched_measure.find("./note[1]/pitch/alter") is None
    assert result.patched_measure.find("./note[2]/rest") is not None


def test_pitch_consensus_repairs_when_template_family_abstains() -> None:
    """Unanimous independent voters may correct an abstaining template family."""

    candidates = [
        _accidental_candidate(
            "adaptive",
            "binary",
            accidental="",
            alter=1,
            include_rest=True,
        ),
        _accidental_candidate(
            "otsu",
            "binary",
            accidental="",
            alter=0,
            include_rest=True,
        ),
        _accidental_candidate(
            "flat",
            "restoration",
            accidental="",
            alter=0,
            include_rest=True,
        ),
        _accidental_candidate(
            "deblock",
            "restoration",
            accidental="",
            alter=0,
            include_rest=True,
        ),
        _accidental_candidate(
            "upscale",
            "scale",
            accidental="",
            alter=0,
            include_rest=True,
        ),
        _accidental_candidate(
            "system_localized",
            "localization",
            accidental="",
            alter=0,
            include_rest=True,
        ),
    ]

    result = propose_pitch_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysReject(),  # type: ignore[arg-type]
    )

    assert result.accepted
    assert result.reason == "accepted_implicit_state_repair"
    assert result.model_version == "deterministic-implicit-accidental-state@1"
    assert result.changed_event_indices == (0,)
    assert result.patched_measure is not None
    assert result.patched_measure.find("./note[1]/pitch/alter") is None
    assert result.patched_measure.find("./note[2]/rest") is not None
    assert result.input is not None
    assert result.input.accidental_only_change_ratio == 1.0


def test_pitch_consensus_accidental_presence_guard_can_accept_safe_proposal() -> None:
    candidates = [
        _accidental_candidate("primary", "baseline", accidental="", alter=0),
        _accidental_candidate("flat", "restoration", accidental="sharp", alter=1),
        _accidental_candidate("otsu", "binary", accidental="sharp", alter=1),
        _accidental_candidate("upscale", "scale", accidental="sharp", alter=1),
    ]
    result = propose_pitch_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
        accidental_guard=AlwaysAcceptAccidentalPresence(),  # type: ignore[arg-type]
        visual_evidence=_blank_visual_evidence(),
    )
    assert result.accepted
    assert result.patched_measure is not None
    assert result.patched_measure.findtext("./note/accidental") == "sharp"
    assert result.accidental_guard_model_version == "test-accidental-presence"


def test_pitch_consensus_accidental_presence_guard_is_veto_only() -> None:
    candidates = [
        _accidental_candidate("primary", "baseline", accidental="", alter=0),
        _accidental_candidate("flat", "restoration", accidental="sharp", alter=1),
        _accidental_candidate("otsu", "binary", accidental="sharp", alter=1),
        _accidental_candidate("upscale", "scale", accidental="sharp", alter=1),
    ]
    result = propose_pitch_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
        accidental_guard=AlwaysAcceptAccidentalPresence(False),  # type: ignore[arg-type]
        visual_evidence=_blank_visual_evidence(),
    )
    assert not result.accepted
    assert result.reason == "accidental_presence_guard"


def test_pitch_consensus_explicit_accidental_class_change_remains_review_only() -> None:
    candidates = [
        _accidental_candidate("primary", "baseline", accidental="sharp", alter=1),
        _accidental_candidate("flat", "restoration", accidental="flat", alter=-1),
        _accidental_candidate("otsu", "binary", accidental="flat", alter=-1),
        _accidental_candidate("upscale", "scale", accidental="flat", alter=-1),
    ]
    result = propose_pitch_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
        visual_evidence=_blank_visual_evidence(),
    )
    assert not result.accepted
    assert result.reason == "accidental_class_change_review"


def test_pitch_consensus_without_source_crop_preserves_strong_accidental_majority() -> None:
    candidates = [
        _accidental_candidate("primary", "baseline", accidental="", alter=0),
        _accidental_candidate("flat", "restoration", accidental="sharp", alter=1),
        _accidental_candidate("otsu", "binary", accidental="sharp", alter=1),
        _accidental_candidate("upscale", "scale", accidental="sharp", alter=1),
    ]
    result = propose_pitch_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert result.accepted


def test_pitch_consensus_empty_symbol_crop_preserves_existing_semantic_path() -> None:
    candidates = [
        _accidental_candidate("primary", "baseline", accidental="", alter=0),
        _accidental_candidate("flat", "restoration", accidental="sharp", alter=1),
        _accidental_candidate("otsu", "binary", accidental="sharp", alter=1),
        _accidental_candidate("upscale", "scale", accidental="sharp", alter=1),
    ]
    evidence = VisualMeasureEvidence(
        1, 1, 1, (0, 0, 256, 96), 10.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        (0.0,) * 8, (0.0,) * 9,
    )
    result = propose_pitch_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
        accidental_guard=AlwaysAcceptAccidentalPresence(False),  # type: ignore[arg-type]
        visual_evidence=evidence,
    )
    assert result.accepted
    assert result.accidental_guard_model_version == "not_applicable"
