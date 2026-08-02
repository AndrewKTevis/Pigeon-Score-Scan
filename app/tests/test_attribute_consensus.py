from __future__ import annotations

from dataclasses import dataclass

from lxml import etree

from scorescan.attribute_consensus import (
    AttributePatchCalibration,
    AttributePatchCalibrator,
    AttributePatchCandidate,
    AttributePatchInput,
    propose_attribute_patch,
)
from scorescan.score_ir import MeasureIR, measure_from_xml


@dataclass
class AlwaysAccept:
    threshold: float = 0.93
    model_version: str = "test-attribute-patch"

    def calibrate(self, item: AttributePatchInput) -> AttributePatchCalibration:
        return AttributePatchCalibration(1.0, self.threshold, True, self.model_version, 1.0)


def _measure(
    *,
    time: tuple[int, int] | None = (4, 4),
    key: tuple[int, str] | None = (0, "major"),
    clef: tuple[str, int, int] | None = ("G", 2, 0),
    explicit_time: bool = True,
    explicit_key: bool = True,
    explicit_clef: bool = True,
    steps: tuple[str, ...] = ("C", "D", "E", "F"),
) -> etree._Element:
    measure = etree.Element("measure", number="1")
    if explicit_time or explicit_key or explicit_clef:
        attributes = etree.SubElement(measure, "attributes")
        etree.SubElement(attributes, "divisions").text = "1"
        if explicit_key and key is not None:
            key_node = etree.SubElement(attributes, "key")
            etree.SubElement(key_node, "fifths").text = str(key[0])
            if key[1]:
                etree.SubElement(key_node, "mode").text = key[1]
        if explicit_time and time is not None:
            time_node = etree.SubElement(attributes, "time")
            etree.SubElement(time_node, "beats").text = str(time[0])
            etree.SubElement(time_node, "beat-type").text = str(time[1])
        if explicit_clef and clef is not None:
            clef_node = etree.SubElement(attributes, "clef")
            etree.SubElement(clef_node, "sign").text = clef[0]
            etree.SubElement(clef_node, "line").text = str(clef[1])
            if clef[2]:
                etree.SubElement(clef_node, "clef-octave-change").text = str(clef[2])
    for step in steps:
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "1"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
    return measure


def _semantics(
    measure: etree._Element,
    *,
    inherited_time: tuple[int, int] = (4, 4),
    inherited_key: tuple[int, str] = (0, "major"),
    inherited_clef: tuple[str, int, int] = ("G", 2, 0),
) -> MeasureIR:
    parsed, _ = measure_from_xml(
        measure,
        {
            "divisions": 1,
            "time": inherited_time,
            "key": inherited_key,
            "clef": inherited_clef,
        },
    )
    return parsed


def _candidate(
    variant: str,
    family: str,
    *,
    time: tuple[int, int] = (4, 4),
    key: tuple[int, str] = (0, "major"),
    clef: tuple[str, int, int] = ("G", 2, 0),
    explicit_time: bool = True,
    explicit_key: bool = True,
    explicit_clef: bool = True,
    previous_time: tuple[int, int] | None = None,
    previous_key: tuple[int, str] | None = None,
    previous_clef: tuple[str, int, int] | None = None,
    following_time: tuple[int, int] | None = None,
    following_key: tuple[int, str] | None = None,
    following_clef: tuple[str, int, int] | None = None,
    valid: bool = True,
) -> AttributePatchCandidate:
    measure = _measure(
        time=time,
        key=key,
        clef=clef,
        explicit_time=explicit_time,
        explicit_key=explicit_key,
        explicit_clef=explicit_clef,
    )
    semantics = _semantics(measure, inherited_time=time, inherited_key=key, inherited_clef=clef)

    def neighbour(
        t: tuple[int, int] | None,
        k: tuple[int, str] | None,
        c: tuple[str, int, int] | None,
    ) -> MeasureIR | None:
        if t is None and k is None and c is None:
            return None
        node = _measure(
            time=t or time,
            key=k or key,
            clef=c or clef,
            explicit_time=True,
            explicit_key=True,
            explicit_clef=True,
        )
        return _semantics(node, inherited_time=t or time, inherited_key=k or key, inherited_clef=c or clef)

    return AttributePatchCandidate(
        variant=variant,
        family=family,
        measure=measure,
        semantics=semantics,
        previous_semantics=neighbour(previous_time, previous_key, previous_clef),
        following_semantics=neighbour(following_time, following_key, following_clef),
        page_score=1000.0,
        page_probability=0.92,
        measure_probability=0.93,
        visual_probability=0.86,
        event_probability=0.94,
        context_probability=0.90,
        ensemble_probability=0.96,
        valid=valid,
    )


def test_attribute_consensus_repairs_first_measure_time_signature() -> None:
    candidates = [
        _candidate("primary", "baseline", time=(3, 4)),
        _candidate("flat", "restoration", time=(4, 4)),
        _candidate("otsu", "binary", time=(4, 4)),
        _candidate("upscale", "scale", time=(4, 4)),
    ]
    result = propose_attribute_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        is_first_measure=True,
        is_last_measure=False,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert result.accepted
    assert result.changed_attributes == ("time",)
    assert result.patched_measure is not None
    parsed = _semantics(result.patched_measure, inherited_time=(3, 4))
    assert parsed.time_signature == (4, 4)
    assert [note.pitch.step for note in parsed.notes if note.pitch] == ["C", "D", "E", "F"]


def test_attribute_consensus_removes_spurious_key_change_using_inherited_majority() -> None:
    candidates = [
        _candidate(
            "primary", "baseline", key=(2, "major"), explicit_key=True,
            previous_key=(0, "major"), following_key=(0, "major"),
        ),
        _candidate(
            "flat", "restoration", key=(0, "major"), explicit_key=False,
            previous_key=(0, "major"), following_key=(0, "major"),
        ),
        _candidate(
            "otsu", "binary", key=(0, "major"), explicit_key=False,
            previous_key=(0, "major"), following_key=(0, "major"),
        ),
        _candidate(
            "upscale", "scale", key=(0, "major"), explicit_key=False,
            previous_key=(0, "major"), following_key=(0, "major"),
        ),
    ]
    result = propose_attribute_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        is_first_measure=False,
        is_last_measure=False,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert result.accepted
    assert result.changed_attributes == ("key",)
    assert result.patched_measure is not None
    assert result.patched_measure.findtext("./attributes/key/fifths") == "0"


def test_attribute_consensus_requires_boundary_evidence() -> None:
    candidates = [
        _candidate(
            "primary", "baseline", key=(2, "major"), explicit_key=False,
            previous_key=(2, "major"), following_key=(2, "major"),
        ),
        _candidate(
            "flat", "restoration", key=(0, "major"), explicit_key=False,
            previous_key=(0, "major"), following_key=(0, "major"),
        ),
        _candidate(
            "otsu", "binary", key=(0, "major"), explicit_key=False,
            previous_key=(0, "major"), following_key=(0, "major"),
        ),
        _candidate(
            "upscale", "scale", key=(0, "major"), explicit_key=False,
            previous_key=(0, "major"), following_key=(0, "major"),
        ),
    ]
    result = propose_attribute_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        is_first_measure=False,
        is_last_measure=False,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert not result.accepted
    key_decision = next(item for item in result.decisions if item.kind == "key")
    assert key_decision.reason == "missing_attribute_boundary_evidence"


def test_attribute_consensus_invalid_sibling_makes_family_abstain() -> None:
    candidates = [
        _candidate("primary", "baseline", key=(2, "major")),
        _candidate("flat", "restoration", key=(0, "major")),
        _candidate("deblock", "restoration", key=(0, "major"), valid=False),
        _candidate("otsu", "binary", key=(0, "major")),
        _candidate("upscale", "scale", key=(0, "major")),
    ]
    result = propose_attribute_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        is_first_measure=True,
        is_last_measure=False,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert not result.accepted
    key_decision = next(item for item in result.decisions if item.kind == "key")
    assert key_decision.reason == "no_strict_attribute_family_majority"


def test_attribute_consensus_rejects_time_signature_that_worsens_meter_fit() -> None:
    candidates = [
        _candidate("primary", "baseline", time=(4, 4)),
        _candidate("flat", "restoration", time=(3, 4)),
        _candidate("otsu", "binary", time=(3, 4)),
        _candidate("upscale", "scale", time=(3, 4)),
    ]
    result = propose_attribute_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        is_first_measure=False,
        is_last_measure=False,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
    )
    assert not result.accepted
    time_decision = next(item for item in result.decisions if item.kind == "time")
    assert time_decision.reason == "meter_fit_worsened"


def test_attribute_patch_composes_onto_existing_event_patch() -> None:
    candidates = [
        _candidate("primary", "baseline", clef=("F", 4, 0)),
        _candidate("flat", "restoration", clef=("G", 2, 0)),
        _candidate("otsu", "binary", clef=("G", 2, 0)),
        _candidate("upscale", "scale", clef=("G", 2, 0)),
    ]
    base = _measure(clef=("F", 4, 0))
    base.find("./note/pitch/step").text = "G"  # simulate an accepted pitch patch
    result = propose_attribute_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        is_first_measure=True,
        is_last_measure=False,
        calibrator=AlwaysAccept(),  # type: ignore[arg-type]
        base_measure=base,
    )
    assert result.accepted
    assert result.patched_measure is not None
    assert result.patched_measure.findtext("./attributes/clef/sign") == "G"
    assert result.patched_measure.findtext("./note/pitch/step") == "G"


def test_attribute_patch_model_is_verified_and_rejects_conflict() -> None:
    calibrator = AttributePatchCalibrator()
    assert calibrator.enabled
    assert calibrator.model_verified
    assert calibrator.model_version == "scorescan-attribute-patch-forest-1"

    base = dict(
        candidate_count=6,
        eligible_family_count=4,
        voting_family_count=4,
        winner_family_support_ratio=0.75,
        winner_family_margin_ratio=0.50,
        template_family_support_ratio=0.25,
        family_abstention_ratio=0.0,
        attribute_kind="key",
        is_first_measure=False,
        is_last_measure=False,
        template_has_explicit_attribute=True,
        support_explicit_attribute_ratio=0.75,
        support_previous_continuity_ratio=0.92,
        support_following_continuity_ratio=0.94,
        template_previous_continuity=0.10,
        template_following_continuity=0.08,
        template_meter_error=0.0,
        winner_meter_error=0.0,
        meter_error_improvement=0.0,
        mean_support_page_probability=0.91,
        mean_support_measure_probability=0.93,
        mean_support_visual_probability=0.85,
        mean_support_event_probability=0.92,
        mean_support_context_probability=0.94,
        mean_support_ensemble_probability=0.96,
        minimum_support_ensemble_probability=0.88,
        mean_support_page_score_margin=18.0,
        mean_support_vs_template_ensemble_probability=0.20,
    )
    favourable = calibrator.predict_probability(AttributePatchInput(**base))
    conflict = calibrator.predict_probability(AttributePatchInput(**{
        **base,
        "support_previous_continuity_ratio": 0.20,
        "support_following_continuity_ratio": 0.18,
        "template_previous_continuity": 0.95,
        "template_following_continuity": 0.94,
        "mean_support_visual_probability": 0.20,
        "mean_support_context_probability": 0.24,
        "mean_support_vs_template_ensemble_probability": -0.08,
    }))
    assert favourable > conflict


def test_attribute_patch_missing_model_fails_closed(tmp_path) -> None:
    calibrator = AttributePatchCalibrator(tmp_path / "missing.json")
    assert not calibrator.enabled
    candidates = [
        _candidate("primary", "baseline", time=(3, 4)),
        _candidate("flat", "restoration", time=(4, 4)),
        _candidate("otsu", "binary", time=(4, 4)),
        _candidate("upscale", "scale", time=(4, 4)),
    ]
    result = propose_attribute_patch(
        candidates,
        template_index=0,
        missing_candidate_count=0,
        is_first_measure=True,
        is_last_measure=False,
        calibrator=calibrator,
    )
    assert not result.accepted
    assert result.reason == "model_guard"
