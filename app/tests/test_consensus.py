from dataclasses import dataclass
from pathlib import Path

from lxml import etree

from scorescan.consensus import (
    _committed_family_support,
    _patch_transaction_guard,
    build_measure_consensus,
    measure_signature,
    semantic_agreement,
)
from scorescan.musicxml import MUSICXML_DOCTYPE


@dataclass(frozen=True)
class Candidate:
    variant: str
    xml_path: str
    score: float
    valid: bool = True


def write_score(path: Path, pitches: list[str]) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    for number, step in enumerate(pitches, start=1):
        measure = etree.SubElement(part, "measure", number=str(number))
        if number == 1:
            attributes = etree.SubElement(measure, "attributes")
            etree.SubElement(attributes, "divisions").text = "1"
            time = etree.SubElement(attributes, "time")
            etree.SubElement(time, "beats").text = "4"
            etree.SubElement(time, "beat-type").text = "4"
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "4"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "whole"
    etree.ElementTree(root).write(str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE)


def add_repeated_key_and_clef(
    path: Path,
    measure_number: int | None = None,
) -> None:
    tree = etree.parse(str(path))
    measures = tree.getroot().findall("./part/measure")
    first_attributes = measures[0].find("attributes")
    key = etree.SubElement(first_attributes, "key")
    etree.SubElement(key, "fifths").text = "0"
    clef = etree.SubElement(first_attributes, "clef")
    etree.SubElement(clef, "sign").text = "G"
    etree.SubElement(clef, "line").text = "2"

    if measure_number is not None:
        repeated = etree.Element("attributes")
        repeated.append(etree.fromstring(etree.tostring(key)))
        repeated.append(etree.fromstring(etree.tostring(clef)))
        measures[measure_number - 1].insert(0, repeated)
    tree.write(
        str(path),
        encoding="UTF-8",
        xml_declaration=True,
        doctype=MUSICXML_DOCTYPE,
    )


def add_slur_stop(path: Path, measure_number: int) -> None:
    tree = etree.parse(str(path))
    note = tree.getroot().find(f"./part/measure[{measure_number}]/note")
    notations = etree.SubElement(note, "notations")
    etree.SubElement(notations, "slur", type="stop", number="1")
    tree.write(
        str(path),
        encoding="UTF-8",
        xml_declaration=True,
        doctype=MUSICXML_DOCTYPE,
    )


def test_measure_signature_ignores_measure_number(tmp_path: Path) -> None:
    path = tmp_path / "a.musicxml"
    write_score(path, ["C"])
    measure = etree.parse(str(path)).getroot().find("./part/measure")
    before = measure_signature(measure)
    measure.set("number", "999")
    assert measure_signature(measure) == before


def test_consensus_replaces_isolated_template_error(tmp_path: Path) -> None:
    primary = tmp_path / "primary.musicxml"
    flat = tmp_path / "flat.musicxml"
    otsu = tmp_path / "otsu.musicxml"
    upscale = tmp_path / "upscale.musicxml"
    output = tmp_path / "consensus.musicxml"
    write_score(primary, ["C", "D", "F", "G"])
    write_score(flat, ["C", "D", "E", "G"])
    write_score(otsu, ["C", "D", "E", "G"])
    write_score(upscale, ["C", "D", "E", "G"])
    candidates = [
        Candidate("primary", str(primary), 1050),
        Candidate("flat", str(flat), 1030),
        Candidate("otsu", str(otsu), 1010),
        Candidate("upscale", str(upscale), 1000),
    ]
    report = build_measure_consensus(candidates, output, "primary")
    assert report is not None
    assert report.replacements == 1
    assert report.disagreement_measure_indices == (3,)
    assert etree.parse(str(output)).getroot().findtext("./part/measure[3]/note/pitch/step") == "E"
    agreement = semantic_agreement(candidates)
    assert agreement["flat"] > agreement["primary"]


def test_consensus_audits_but_does_not_surface_redundant_system_state(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary.musicxml"
    system_localized = tmp_path / "system_localized.musicxml"
    output = tmp_path / "consensus.musicxml"
    write_score(primary, ["C", "D"])
    write_score(system_localized, ["C", "D"])
    add_repeated_key_and_clef(primary)
    add_repeated_key_and_clef(system_localized, 2)

    report = build_measure_consensus(
        [
            Candidate("primary", str(primary), 1050),
            Candidate("system_localized", str(system_localized), 1030),
        ],
        output,
        "primary",
    )

    assert report is not None
    assert report.preservation_disagreement_measure_indices == (2,)
    assert report.disagreement_measure_indices == ()
    assert report.votes[1].decision == "retain_redundant_state_attributes"
    assert report.to_dict()["preservation_disagreement_measure_indices"] == [2]


def test_consensus_keeps_slur_difference_as_actionable_disagreement(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary.musicxml"
    adaptive = tmp_path / "adaptive.musicxml"
    output = tmp_path / "consensus.musicxml"
    write_score(primary, ["C", "D"])
    write_score(adaptive, ["C", "D"])
    add_slur_stop(adaptive, 2)

    report = build_measure_consensus(
        [
            Candidate("primary", str(primary), 1050),
            Candidate("adaptive", str(adaptive), 1030),
        ],
        output,
        "primary",
    )

    assert report is not None
    assert report.preservation_disagreement_measure_indices == (2,)
    assert report.disagreement_measure_indices == (2,)
    assert report.votes[1].decision != "retain_redundant_state_attributes"


def test_consensus_resolves_retained_strict_exact_majority(
    tmp_path: Path,
) -> None:
    variants = (
        "primary",
        "flat",
        "staffnorm",
        "deblock",
        "adaptive",
        "otsu",
        "system_localized",
    )
    candidates: list[Candidate] = []
    for index, variant in enumerate(variants):
        path = tmp_path / f"{variant}.musicxml"
        write_score(path, ["C", "D"])
        if variant == "adaptive":
            add_slur_stop(path, 2)
        candidates.append(Candidate(variant, str(path), 1050 - index * 5))

    report = build_measure_consensus(
        candidates,
        tmp_path / "consensus.musicxml",
        "primary",
    )

    assert report is not None
    assert report.preservation_disagreement_measure_indices == (2,)
    assert report.resolved_disagreement_measure_indices == (2,)
    assert report.disagreement_measure_indices == ()
    assert report.votes[1].decision == "retain_exact_majority"


def test_consensus_retains_template_for_two_family_exact_vote_without_visual_corroboration(tmp_path: Path) -> None:
    primary = tmp_path / "primary.musicxml"
    flat = tmp_path / "flat.musicxml"
    otsu = tmp_path / "otsu.musicxml"
    output = tmp_path / "consensus.musicxml"
    write_score(primary, ["C", "D", "F", "G"])
    write_score(flat, ["C", "D", "E", "G"])
    write_score(otsu, ["C", "D", "E", "G"])
    report = build_measure_consensus(
        [
            Candidate("primary", str(primary), 1050),
            Candidate("flat", str(flat), 1030),
            Candidate("otsu", str(otsu), 1010),
        ],
        output,
        "primary",
    )
    assert report is not None
    assert report.replacements == 0
    assert report.unresolved_measure_indices == (3,)
    assert report.votes[2].decision == "retain_template_no_majority"
    assert etree.parse(str(output)).getroot().findtext("./part/measure[3]/note/pitch/step") == "F"



def test_measure_localized_candidate_is_sparse_and_forms_third_family(tmp_path: Path) -> None:
    primary = tmp_path / "primary.musicxml"
    flat = tmp_path / "flat.musicxml"
    otsu = tmp_path / "otsu.musicxml"
    local = tmp_path / "local.musicxml"
    output = tmp_path / "consensus.musicxml"
    write_score(primary, ["C", "F"])
    write_score(flat, ["C", "E"])
    write_score(otsu, ["C", "E"])
    # The first measure is deliberately wrong.  Sparse scope must prevent it from
    # affecting the first measure while it supplies the third family for measure two.
    write_score(local, ["G", "E"])
    report = build_measure_consensus(
        [
            Candidate("primary", str(primary), 1050),
            Candidate("flat", str(flat), 1030),
            Candidate("otsu", str(otsu), 1020),
            Candidate("measure_localized:2", str(local), 1015),
        ],
        output,
        "primary",
    )
    assert report is not None
    assert report.votes[0].eligible_candidates == 3
    assert report.votes[1].eligible_candidates == 4
    assert etree.parse(str(output)).getroot().findtext("./part/measure[1]/note/pitch/step") == "C"
    assert etree.parse(str(output)).getroot().findtext("./part/measure[2]/note/pitch/step") == "E"
    assert report.replacements == 1

def test_consensus_keeps_template_without_strict_majority(tmp_path: Path) -> None:
    primary = tmp_path / "primary.musicxml"
    flat = tmp_path / "flat.musicxml"
    otsu = tmp_path / "otsu.musicxml"
    adaptive = tmp_path / "adaptive.musicxml"
    output = tmp_path / "consensus.musicxml"
    write_score(primary, ["C", "F"])
    write_score(flat, ["C", "E"])
    write_score(otsu, ["C", "E"])
    write_score(adaptive, ["C", "G"])
    candidates = [
        Candidate("primary", str(primary), 1100),
        Candidate("flat", str(flat), 1060),
        Candidate("otsu", str(otsu), 1040),
        Candidate("adaptive", str(adaptive), 1020),
    ]
    report = build_measure_consensus(candidates, output, "primary")
    assert report is not None
    assert report.replacements == 0
    assert report.unresolved_measure_indices == (2,)
    assert report.votes[1].decision == "retain_template_no_majority"
    assert etree.parse(str(output)).getroot().findtext("./part/measure[2]/note/pitch/step") == "F"


def write_score_with_divisions(path: Path, step: str, divisions: int) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = str(divisions)
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"
    note = etree.SubElement(measure, "note")
    pitch = etree.SubElement(note, "pitch")
    etree.SubElement(pitch, "step").text = step
    etree.SubElement(pitch, "octave").text = "4"
    etree.SubElement(note, "duration").text = str(4 * divisions)
    etree.SubElement(note, "voice").text = "1"
    etree.SubElement(note, "type").text = "whole"
    etree.ElementTree(root).write(str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE)


def test_semantic_consensus_normalizes_musicxml_representation(tmp_path: Path) -> None:
    candidates = []
    for variant, divisions, score in (("primary", 1, 1040), ("flat", 2, 1030), ("otsu", 4, 1020)):
        path = tmp_path / f"{variant}.musicxml"
        write_score_with_divisions(path, "C", divisions)
        candidates.append(Candidate(variant, str(path), score))
    output = tmp_path / "consensus.musicxml"
    report = build_measure_consensus(candidates, output, "primary")
    assert report is not None
    assert report.exact_agreement_ratio == 1.0
    assert report.semantic_agreement_ratio > 0.95
    assert report.mean_measure_confidence > 0.95
    assert report.unresolved_measure_indices == ()
    assert report.votes[0].decision == "retain_agreement"


def test_semantic_agreement_does_not_reward_singleton_clusters(tmp_path: Path) -> None:
    candidates = []
    for variant, step in zip(("primary", "flat", "otsu", "adaptive"), ("C", "D", "E", "F"), strict=True):
        path = tmp_path / f"{variant}.musicxml"
        write_score_with_divisions(path, step, 1)
        candidates.append(Candidate(variant, str(path), 1000))
    output = tmp_path / "consensus.musicxml"
    report = build_measure_consensus(candidates, output, "primary")
    assert report is not None
    assert report.exact_agreement_ratio == 0.0
    assert report.semantic_agreement_ratio < 0.40
    assert report.mean_measure_confidence < 0.40
    assert report.unresolved_measure_indices == (1,)


def test_consensus_reports_event_lattice_calibration(tmp_path: Path) -> None:
    candidates = []
    for variant, step, score in (("primary", "C", 1040), ("flat", "C", 1030), ("otsu", "D", 1020)):
        path = tmp_path / f"{variant}.musicxml"
        write_score_with_divisions(path, step, 1)
        candidates.append(Candidate(variant, str(path), score))
    output = tmp_path / "consensus.musicxml"
    report = build_measure_consensus(candidates, output, "primary")
    assert report is not None
    assert report.event_calibration_model == "scorescan-event-forest-2"
    assert 0.0 <= report.mean_event_probability <= 1.0
    vote = report.votes[0]
    assert vote.event_calibration_model == "scorescan-event-forest-2"
    assert set(vote.candidate_event_probabilities) == {"primary", "flat", "otsu"}


def test_consensus_reports_context_calibration(tmp_path: Path) -> None:
    candidates = []
    for variant, pitches, score in (("primary", ["C", "D", "E"], 1040), ("flat", ["C", "D", "E"], 1030), ("otsu", ["C", "G", "E"], 1020)):
        path = tmp_path / f"{variant}.musicxml"
        write_score(path, pitches)
        candidates.append(Candidate(variant, str(path), score))
    output = tmp_path / "consensus.musicxml"
    report = build_measure_consensus(candidates, output, "primary")
    assert report is not None
    assert report.context_calibration_model == "scorescan-context-forest-2"
    assert 0.0 <= report.mean_context_probability <= 1.0
    vote = report.votes[1]
    assert vote.context_calibration_model == "scorescan-context-forest-2"
    assert set(vote.candidate_context_probabilities) == {"primary", "flat", "otsu"}


def test_consensus_reports_ensemble_meta_calibration(tmp_path: Path) -> None:
    candidates = []
    for variant, step, score in (("primary", "C", 1040), ("flat", "C", 1030), ("otsu", "D", 1020)):
        path = tmp_path / f"{variant}.musicxml"
        write_score_with_divisions(path, step, 1)
        candidates.append(Candidate(variant, str(path), score))
    output = tmp_path / "consensus.musicxml"
    report = build_measure_consensus(candidates, output, "primary")
    assert report is not None
    assert report.ensemble_calibration_model == "scorescan-ensemble-forest-3"
    assert 0.0 <= report.mean_ensemble_probability <= 1.0
    vote = report.votes[0]
    assert vote.ensemble_calibration_model == "scorescan-ensemble-forest-3"
    assert set(vote.candidate_ensemble_probabilities) == {"primary", "flat", "otsu"}
    assert report.selection_risk_model == "scorescan-selection-risk-forest-4"
    assert 0.0 <= report.mean_selection_risk_probability <= 1.0
    assert vote.selection_risk_model == "scorescan-selection-risk-forest-4"
    assert 0.0 <= vote.selected_selection_risk_probability <= 1.0


def test_correlated_variant_family_cannot_create_false_majority(tmp_path: Path) -> None:
    primary = tmp_path / "primary.musicxml"
    flat = tmp_path / "flat.musicxml"
    deblock = tmp_path / "deblock.musicxml"
    output = tmp_path / "consensus.musicxml"
    write_score(primary, ["C", "F"])
    write_score(flat, ["C", "E"])
    write_score(deblock, ["C", "E"])
    candidates = [
        Candidate("primary", str(primary), 1080),
        Candidate("flat", str(flat), 1040),
        Candidate("deblock", str(deblock), 1030),
    ]
    report = build_measure_consensus(candidates, output, "primary")
    assert report is not None
    assert report.replacements == 0
    assert report.votes[1].strict_majority is False
    assert report.votes[1].exact_family_support == 1
    assert report.votes[1].eligible_family_count == 2
    assert etree.parse(str(output)).getroot().findtext("./part/measure[2]/note/pitch/step") == "F"


def test_semantic_family_support_abstains_for_split_or_missing_siblings() -> None:
    variants = ["primary", "flat", "deblock", "otsu", "upscale"]
    family_sizes = {"baseline": 1, "restoration": 2, "binary": 1, "scale": 1}

    # One restoration sibling supports the cluster while the other does not: that
    # correlated family must abstain instead of being counted as independent support.
    assert _committed_family_support(variants, {1, 3, 4}, family_sizes) == {"binary", "scale"}
    assert _committed_family_support(variants, {1, 2, 3, 4}, family_sizes) == {
        "restoration",
        "binary",
        "scale",
    }

    # A missing aligned sibling is also an abstention, even if the visible member is
    # inside the cluster.
    assert _committed_family_support(
        ["primary", "flat", "otsu", "upscale"],
        {1, 2, 3},
        family_sizes,
    ) == {"binary", "scale"}


def write_single_measure_notes(path: Path, pitches: list[str]) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"
    for step in pitches:
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "1"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
    etree.ElementTree(root).write(str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE)


def test_consensus_applies_pitch_only_patch_when_whole_measure_gate_rejects(tmp_path: Path, monkeypatch) -> None:
    from scorescan.pitch_consensus import PitchPatchCalibration, PitchPatchCalibrator
    from scorescan.selection_risk import SelectionRiskCalibrator, SelectionRiskResult

    monkeypatch.setattr(
        SelectionRiskCalibrator,
        "calibrate",
        lambda self, item: SelectionRiskResult(0.1, 0.99, False, self.model_version, 0.995),
    )
    monkeypatch.setattr(
        PitchPatchCalibrator,
        "calibrate",
        lambda self, item: PitchPatchCalibration(1.0, 0.9, True, self.model_version, 0.995),
    )

    variants = {
        "primary": ["C", "D", "F"],
        "flat": ["C", "E", "E"],
        "otsu": ["B", "D", "E"],
        "upscale": ["C", "D", "E"],
    }
    candidates = []
    for index, (variant, pitches) in enumerate(variants.items()):
        path = tmp_path / f"{variant}.musicxml"
        write_single_measure_notes(path, pitches)
        candidates.append(Candidate(variant, str(path), 1050 - index * 10))

    output = tmp_path / "consensus.musicxml"
    report = build_measure_consensus(candidates, output, "primary")
    assert report is not None
    assert report.replacements == 1
    assert report.pitch_patch_measure_count == 1
    assert report.pitch_patch_event_count == 1
    assert report.unresolved_measure_indices == ()
    assert report.votes[0].decision == "patch_pitch_consensus"
    assert report.votes[0].pitch_patch_accepted
    assert [node.text for node in etree.parse(str(output)).getroot().findall("./part/measure/note/pitch/step")] == ["C", "D", "E"]


def write_single_measure_rhythms(path: Path, durations: list[int]) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "2"
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"
    for step, duration in zip(("C", "D", "E", "F"), durations, strict=True):
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = str(duration)
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter" if duration == 2 else "eighth"
    etree.ElementTree(root).write(str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE)


def test_consensus_applies_rhythm_only_patch_when_whole_measure_gate_rejects(tmp_path: Path, monkeypatch) -> None:
    from scorescan.rhythm_consensus import RhythmPatchCalibration, RhythmPatchCalibrator
    from scorescan.selection_risk import SelectionRiskCalibrator, SelectionRiskResult

    monkeypatch.setattr(
        SelectionRiskCalibrator,
        "calibrate",
        lambda self, item: SelectionRiskResult(0.1, 0.99, False, self.model_version, 0.995),
    )
    monkeypatch.setattr(
        RhythmPatchCalibrator,
        "calibrate",
        lambda self, item: RhythmPatchCalibration(1.0, 0.92, True, self.model_version, 0.9975),
    )

    variants = {
        "primary": [2, 2, 2, 1],
        "flat": [2, 2, 1, 2],
        "otsu": [2, 1, 2, 2],
        "upscale": [2, 2, 2, 2],
    }
    candidates = []
    for index, (variant, durations) in enumerate(variants.items()):
        path = tmp_path / f"{variant}.musicxml"
        write_single_measure_rhythms(path, durations)
        candidates.append(Candidate(variant, str(path), 1050 - index * 10))

    output = tmp_path / "consensus.musicxml"
    report = build_measure_consensus(candidates, output, "primary")
    assert report is not None
    assert report.replacements == 1
    assert report.rhythm_patch_measure_count == 1
    assert report.rhythm_patch_event_count == 1
    assert report.unresolved_measure_indices == ()
    assert report.votes[0].decision == "patch_rhythm_consensus"
    assert report.votes[0].rhythm_patch_accepted
    assert [node.text for node in etree.parse(str(output)).getroot().findall("./part/measure/note/duration")] == ["2"] * 4


def write_score_with_time(path: Path, beats: int) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    key = etree.SubElement(attributes, "key")
    etree.SubElement(key, "fifths").text = "0"
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = str(beats)
    etree.SubElement(time, "beat-type").text = "4"
    clef = etree.SubElement(attributes, "clef")
    etree.SubElement(clef, "sign").text = "G"
    etree.SubElement(clef, "line").text = "2"
    for step in ("C", "D", "E", "F"):
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "1"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
    etree.ElementTree(root).write(str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE)


def test_consensus_integrates_attribute_patch_when_whole_measure_replacement_is_vetoed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scorescan.attribute_consensus import AttributePatchCalibration
    from scorescan.selection_risk import SelectionRiskResult
    import scorescan.consensus as consensus_module

    class AcceptAttribute:
        threshold = 0.93
        model_version = "test-attribute"

        def calibrate(self, item):
            return AttributePatchCalibration(1.0, self.threshold, True, self.model_version, 1.0)

    class RejectReplacement:
        threshold = 0.99
        model_version = "test-selection-risk"

        def calibrate(self, item):
            return SelectionRiskResult(0.0, self.threshold, False, self.model_version, 1.0)

    monkeypatch.setattr(consensus_module, "AttributePatchCalibrator", AcceptAttribute)
    monkeypatch.setattr(consensus_module, "SelectionRiskCalibrator", RejectReplacement)

    paths = {}
    for variant, beats in (("primary", 3), ("flat", 4), ("otsu", 4), ("upscale", 4)):
        path = tmp_path / f"{variant}.musicxml"
        write_score_with_time(path, beats)
        paths[variant] = path
    output = tmp_path / "consensus.musicxml"
    report = build_measure_consensus(
        [
            Candidate("primary", str(paths["primary"]), 1050),
            Candidate("flat", str(paths["flat"]), 1030),
            Candidate("otsu", str(paths["otsu"]), 1020),
            Candidate("upscale", str(paths["upscale"]), 1010),
        ],
        output,
        "primary",
    )
    assert report is not None
    assert report.attribute_patch_measure_count == 1
    assert report.attribute_patch_attribute_count == 1
    assert report.votes[0].decision == "patch_attribute_consensus"
    assert report.votes[0].attribute_patch_attributes == ("time",)
    assert etree.parse(str(output)).findtext("./part/measure/attributes/time/beats") == "4"


def write_single_measure_event_kinds(path: Path, kinds: list[str]) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"
    for step, kind in zip(("C", "D", "E", "F"), kinds, strict=True):
        note = etree.SubElement(measure, "note")
        if kind == "rest":
            etree.SubElement(note, "rest")
        else:
            pitch = etree.SubElement(note, "pitch")
            etree.SubElement(pitch, "step").text = step
            etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "1"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
    etree.ElementTree(root).write(str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE)


def test_consensus_applies_event_kind_patch_when_whole_measure_gate_rejects(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scorescan.event_kind_consensus import EventKindPatchCalibration
    from scorescan.selection_risk import SelectionRiskResult
    import scorescan.consensus as consensus_module

    class AcceptEventKind:
        threshold = 0.95
        model_version = "test-event-kind"

        def calibrate(self, item):
            return EventKindPatchCalibration(1.0, self.threshold, True, self.model_version, 1.0)

    class RejectReplacement:
        threshold = 0.99
        model_version = "test-selection-risk"

        def calibrate(self, item):
            return SelectionRiskResult(0.0, self.threshold, False, self.model_version, 1.0)

    monkeypatch.setattr(consensus_module, "EventKindPatchCalibrator", AcceptEventKind)
    monkeypatch.setattr(consensus_module, "SelectionRiskCalibrator", RejectReplacement)

    variants = {
        "primary": ["rest", "pitch", "pitch", "pitch"],
        "flat": ["pitch", "pitch", "pitch", "pitch"],
        "otsu": ["pitch", "pitch", "pitch", "pitch"],
        "upscale": ["pitch", "pitch", "pitch", "pitch"],
    }
    candidates = []
    for index, (variant, kinds) in enumerate(variants.items()):
        path = tmp_path / f"{variant}.musicxml"
        write_single_measure_event_kinds(path, kinds)
        candidates.append(Candidate(variant, str(path), 1050 - index * 10))

    output = tmp_path / "consensus.musicxml"
    report = build_measure_consensus(candidates, output, "primary")
    assert report is not None
    assert report.replacements == 1
    assert report.event_kind_patch_measure_count == 1
    assert report.event_kind_patch_event_count == 1
    assert report.votes[0].decision == "patch_event_kind_consensus"
    assert report.votes[0].event_kind_patch_accepted
    assert not report.votes[0].event_kind_visual_guard_applicable
    assert report.votes[0].event_kind_visual_guard_reason == "source_evidence_unavailable"
    assert report.event_kind_visual_guard_transaction_count == 0
    first = etree.parse(str(output)).find("./part/measure/note")
    assert first is not None and first.find("pitch") is not None and first.find("rest") is None



def test_consensus_event_kind_visual_guard_can_only_veto_source_backed_patch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scorescan.event_kind_consensus import EventKindPatchCalibration
    from scorescan.event_kind_visual_guard import EventKindVisualAudit
    from scorescan.selection_risk import SelectionRiskResult
    from scorescan.visual_evidence import VisualMeasureEvidence
    import scorescan.consensus as consensus_module

    class AcceptEventKind:
        threshold = 0.95
        model_version = "test-event-kind"

        def calibrate(self, item):
            return EventKindPatchCalibration(1.0, self.threshold, True, self.model_version, 1.0)

    class RejectReplacement:
        threshold = 0.99
        model_version = "test-selection-risk"

        def calibrate(self, item):
            return SelectionRiskResult(0.0, self.threshold, False, self.model_version, 1.0)

    class RejectEventKindVisual:
        threshold = 0.93
        model_version = "test-event-kind-visual"

        def audit_transaction(self, evidence, before, after):
            return EventKindVisualAudit(
                applicable=True,
                changed_event_count=1,
                probability=0.1,
                threshold=self.threshold,
                accepted=False,
                reason="visual_event_kind_conflict",
                model_version=self.model_version,
            )

    monkeypatch.setattr(consensus_module, "EventKindPatchCalibrator", AcceptEventKind)
    monkeypatch.setattr(consensus_module, "SelectionRiskCalibrator", RejectReplacement)
    monkeypatch.setattr(consensus_module, "EventKindVisualGuard", RejectEventKindVisual)

    variants = {
        "primary": ["rest", "pitch", "pitch", "pitch"],
        "flat": ["pitch", "pitch", "pitch", "pitch"],
        "otsu": ["pitch", "pitch", "pitch", "pitch"],
        "upscale": ["pitch", "pitch", "pitch", "pitch"],
    }
    candidates = []
    for index, (variant, kinds) in enumerate(variants.items()):
        path = tmp_path / f"visual-kind-{variant}.musicxml"
        write_single_measure_event_kinds(path, kinds)
        candidates.append(Candidate(variant, str(path), 1050 - index * 10))

    evidence = VisualMeasureEvidence(
        page_index=0, system_index=0, measure_index=0, bbox=(0, 0, 10, 10),
        spacing=10.0, ink_density=0.0, nonstaff_ink_density=0.0,
        component_density=0.0, notehead_proxy=0.0, open_notehead_proxy=0.0,
        stem_proxy=0.0, beam_proxy=0.0, onset_proxy=0.0, compact_mark_proxy=0.0,
        accidental_proxy=0.0, above_ink_density=0.0, below_ink_density=0.0,
        x_ink_profile=(0.0,) * 8, staff_ink_profile=(0.0,) * 9,
        symbol_guard_image="non-empty-test-evidence",
    )
    output = tmp_path / "visual-kind-veto.musicxml"
    report = build_measure_consensus(
        candidates, output, "primary", visual_evidence=(evidence,)
    )
    assert report is not None
    assert report.event_kind_patch_measure_count == 0
    assert report.event_kind_visual_guard_transaction_count == 1
    assert report.event_kind_visual_guard_rejected_count == 1
    assert report.votes[0].event_kind_visual_guard_applicable
    assert not report.votes[0].event_kind_visual_guard_accepted
    assert report.votes[0].event_kind_visual_guard_reason == "visual_event_kind_conflict"
    assert not report.votes[0].event_kind_patch_accepted
    first = etree.parse(str(output)).find("./part/measure/note")
    assert first is not None and first.find("rest") is not None and first.find("pitch") is None

def write_three_measure_event_presence(path: Path, middle_pitches: list[str]) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    for measure_index, pitches in enumerate((["C", "D", "E", "F"], middle_pitches, ["G", "A", "B", "C"]), start=1):
        measure = etree.SubElement(part, "measure", number=str(measure_index))
        if measure_index == 1:
            attributes = etree.SubElement(measure, "attributes")
            etree.SubElement(attributes, "divisions").text = "1"
            time = etree.SubElement(attributes, "time")
            etree.SubElement(time, "beats").text = "4"
            etree.SubElement(time, "beat-type").text = "4"
        for step in pitches:
            note = etree.SubElement(measure, "note")
            pitch = etree.SubElement(note, "pitch")
            etree.SubElement(pitch, "step").text = step
            etree.SubElement(pitch, "octave").text = "4"
            etree.SubElement(note, "duration").text = "1"
            etree.SubElement(note, "voice").text = "1"
            etree.SubElement(note, "type").text = "quarter"
    etree.ElementTree(root).write(str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE)


def test_consensus_applies_event_presence_patch_to_internal_measure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scorescan.event_presence_consensus import EventPresencePatchCalibration
    from scorescan.selection_risk import SelectionRiskResult
    import scorescan.consensus as consensus_module

    class AcceptPresence:
        threshold = 0.975
        model_version = "test-event-presence"

        def calibrate(self, item):
            return EventPresencePatchCalibration(1.0, self.threshold, True, self.model_version, 1.0)

    class RejectReplacement:
        threshold = 0.99
        model_version = "test-selection-risk"

        def calibrate(self, item):
            return SelectionRiskResult(0.0, self.threshold, False, self.model_version, 1.0)

    monkeypatch.setattr(consensus_module, "EventPresencePatchCalibrator", AcceptPresence)
    monkeypatch.setattr(consensus_module, "SelectionRiskCalibrator", RejectReplacement)

    variants = {
        "primary": ["C", "D", "F"],
        "flat": ["C", "D", "E", "F"],
        "otsu": ["C", "D", "E", "F"],
        "upscale": ["C", "D", "E", "F"],
    }
    candidates = []
    for index, (variant, pitches) in enumerate(variants.items()):
        path = tmp_path / f"{variant}.musicxml"
        write_three_measure_event_presence(path, pitches)
        candidates.append(Candidate(variant, str(path), 1050 - index * 10))

    output = tmp_path / "consensus.musicxml"
    report = build_measure_consensus(candidates, output, "primary")
    assert report is not None
    assert report.event_presence_patch_measure_count == 1
    assert report.event_presence_patch_inserted_event_count == 1
    assert report.event_presence_patch_deleted_event_count == 0
    assert report.votes[1].decision == "patch_event_presence_insert_consensus"
    assert report.votes[1].event_presence_patch_accepted
    assert report.votes[1].event_presence_patch_operation == "insert"
    middle = etree.parse(str(output)).findall("./part/measure")[1]
    assert [node.text for node in middle.findall("note/pitch/step")] == ["C", "D", "E", "F"]


def test_event_presence_visual_guard_vetoes_supported_insertion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scorescan.event_presence_consensus import EventPresencePatchCalibration
    from scorescan.event_presence_visual_guard import EventPresenceVisualAudit
    from scorescan.selection_risk import SelectionRiskResult
    from scorescan.visual_evidence import VisualMeasureEvidence
    import scorescan.consensus as consensus_module

    class AcceptPresence:
        threshold = 0.975
        model_version = "test-event-presence"

        def calibrate(self, item):
            return EventPresencePatchCalibration(1.0, self.threshold, True, self.model_version, 1.0)

    class RejectReplacement:
        threshold = 0.99
        model_version = "test-selection-risk"

        def calibrate(self, item):
            return SelectionRiskResult(0.0, self.threshold, False, self.model_version, 1.0)

    class RejectPresenceVisual:
        threshold = 0.96
        thresholds = {"note": 0.96, "rest": 0.95}
        model_version = "test-event-presence-visual"

        def audit_transaction(self, evidence, before, after, operation, event_index):
            assert evidence is not None
            assert operation == "insert"
            assert event_index == 2
            return EventPresenceVisualAudit(
                applicable=True,
                operation=operation,
                changed_event_count=1,
                probability=0.1,
                threshold=self.thresholds["note"],
                accepted=False,
                reason="visual_event_presence_conflict",
                model_version=self.model_version,
            )

    monkeypatch.setattr(consensus_module, "EventPresencePatchCalibrator", AcceptPresence)
    monkeypatch.setattr(consensus_module, "SelectionRiskCalibrator", RejectReplacement)
    monkeypatch.setattr(consensus_module, "EventPresenceVisualGuard", RejectPresenceVisual)

    variants = {
        "primary": ["C", "D", "F"],
        "flat": ["C", "D", "E", "F"],
        "otsu": ["C", "D", "E", "F"],
        "upscale": ["C", "D", "E", "F"],
    }
    candidates = []
    for index, (variant, pitches) in enumerate(variants.items()):
        path = tmp_path / f"visual-presence-{variant}.musicxml"
        write_three_measure_event_presence(path, pitches)
        candidates.append(Candidate(variant, str(path), 1050 - index * 10))

    evidence = VisualMeasureEvidence(
        page_index=0, system_index=0, measure_index=1, bbox=(0, 0, 10, 10),
        spacing=10.0, ink_density=0.0, nonstaff_ink_density=0.0,
        component_density=0.0, notehead_proxy=0.0, open_notehead_proxy=0.0,
        stem_proxy=0.0, beam_proxy=0.0, onset_proxy=0.0, compact_mark_proxy=0.0,
        accidental_proxy=0.0, above_ink_density=0.0, below_ink_density=0.0,
        x_ink_profile=(0.0,) * 8, staff_ink_profile=(0.0,) * 9,
        symbol_guard_image="non-empty-test-evidence",
    )
    output = tmp_path / "visual-presence-veto.musicxml"
    report = build_measure_consensus(
        candidates, output, "primary", visual_evidence=(evidence,)
    )
    assert report is not None
    assert report.event_presence_patch_measure_count == 0
    assert report.event_presence_visual_guard_transaction_count == 1
    assert report.event_presence_visual_guard_rejected_count == 1
    assert report.event_presence_visual_guard_note_threshold == 0.96
    assert report.event_presence_visual_guard_rest_threshold == 0.95
    assert report.votes[1].event_presence_visual_guard_applicable
    assert not report.votes[1].event_presence_visual_guard_accepted
    assert report.votes[1].event_presence_visual_guard_reason == "visual_event_presence_conflict"
    middle = etree.parse(str(output)).findall("./part/measure")[1]
    assert [node.text for node in middle.findall("note/pitch/step")] == ["C", "D", "F"]


def write_single_measure_chord_topology(path: Path, topology: list[bool], pitches: list[str] | None = None) -> None:
    pitches = pitches or ["C", "E", "G", "D", "F"]
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"
    for marker, step in zip(topology, pitches, strict=True):
        note = etree.SubElement(measure, "note")
        if marker:
            etree.SubElement(note, "chord")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "1"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
    etree.ElementTree(root).write(str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE)


def test_consensus_applies_chord_then_pitch_patch_in_one_transaction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scorescan.chord_consensus import ChordPatchCalibration
    from scorescan.pitch_consensus import PitchPatchCalibration
    from scorescan.selection_risk import SelectionRiskResult
    import scorescan.consensus as consensus_module

    class AcceptChord:
        threshold = 0.97
        model_version = "test-chord"

        def calibrate(self, item):
            return ChordPatchCalibration(1.0, self.threshold, True, self.model_version, 1.0)

    class AcceptPitch:
        threshold = 0.90
        model_version = "test-pitch"

        def calibrate(self, item):
            return PitchPatchCalibration(1.0, self.threshold, True, self.model_version, 1.0)

    class RejectReplacement:
        threshold = 0.99
        model_version = "test-selection-risk"

        def calibrate(self, item):
            return SelectionRiskResult(0.0, self.threshold, False, self.model_version, 1.0)

    monkeypatch.setattr(consensus_module, "ChordPatchCalibrator", AcceptChord)
    monkeypatch.setattr(consensus_module, "PitchPatchCalibrator", AcceptPitch)
    monkeypatch.setattr(consensus_module, "SelectionRiskCalibrator", RejectReplacement)

    wrong_topology = [False, False, False, False, False]
    correct_topology = [False, True, False, False, False]
    rows = {
        "primary": (wrong_topology, ["C", "D", "F", "D", "F"]),
        "flat": (correct_topology, ["C", "E", "E", "D", "F"]),
        "otsu": (correct_topology, ["C", "E", "E", "D", "F"]),
        "upscale": (correct_topology, ["C", "E", "E", "D", "F"]),
    }
    candidates = []
    for index, (variant, (topology, pitches)) in enumerate(rows.items()):
        path = tmp_path / f"{variant}.musicxml"
        write_single_measure_chord_topology(path, topology, pitches)
        candidates.append(Candidate(variant, str(path), 1050 - index * 10))

    output = tmp_path / "consensus.musicxml"
    report = build_measure_consensus(candidates, output, "primary")
    assert report is not None
    assert report.chord_patch_measure_count == 1
    assert report.chord_patch_event_count == 1
    assert report.pitch_patch_measure_count == 1
    assert report.votes[0].decision == "patch_chord_and_pitch_consensus"
    assert report.votes[0].chord_patch_accepted
    assert report.votes[0].pitch_patch_accepted
    notes = etree.parse(str(output)).findall("./part/measure/note")
    assert notes[1].find("chord") is not None
    assert [note.findtext("pitch/step") for note in notes] == ["C", "E", "E", "D", "F"]


def write_single_measure_ties(path: Path, states: list[tuple[str, ...]]) -> None:
    pitches = ["C", "C", "D", "E"]
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"
    for state, step in zip(states, pitches, strict=True):
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "1"
        for value in state:
            etree.SubElement(note, "tie", type=value)
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
        if state:
            notations = etree.SubElement(note, "notations")
            for value in state:
                etree.SubElement(notations, "tied", type=value)
    etree.ElementTree(root).write(
        str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE
    )


def test_consensus_applies_tie_patch_when_measure_replacement_is_vetoed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scorescan.selection_risk import SelectionRiskResult
    from scorescan.tie_consensus import TiePatchCalibration
    import scorescan.consensus as consensus_module

    class AcceptTie:
        threshold = 0.975
        model_version = "test-tie"

        def calibrate(self, item):
            return TiePatchCalibration(1.0, self.threshold, True, self.model_version, 1.0)

    class RejectReplacement:
        threshold = 0.99
        model_version = "test-selection-risk"

        def calibrate(self, item):
            return SelectionRiskResult(0.0, self.threshold, False, self.model_version, 1.0)

    monkeypatch.setattr(consensus_module, "TiePatchCalibrator", AcceptTie)
    monkeypatch.setattr(consensus_module, "SelectionRiskCalibrator", RejectReplacement)

    wrong = [(), (), (), ()]
    correct = [("start",), ("stop",), (), ()]
    candidates = []
    for index, (variant, states) in enumerate(
        (("primary", wrong), ("flat", correct), ("otsu", correct), ("upscale", correct))
    ):
        path = tmp_path / f"{variant}.musicxml"
        write_single_measure_ties(path, states)
        candidates.append(Candidate(variant, str(path), 1050 - index * 10))

    output = tmp_path / "consensus.musicxml"
    report = build_measure_consensus(candidates, output, "primary")
    assert report is not None
    assert report.tie_patch_measure_count == 1
    assert report.tie_patch_event_count == 2
    assert report.votes[0].decision == "patch_tie_consensus"
    assert report.votes[0].tie_patch_accepted
    assert not report.votes[0].tie_visual_guard_applicable
    assert report.votes[0].tie_visual_guard_reason == "source_evidence_unavailable"
    assert report.tie_visual_guard_transaction_count == 0
    notes = etree.parse(str(output)).findall("./part/measure/note")
    assert notes[0].find("tie").get("type") == "start"
    assert notes[1].find("tie").get("type") == "stop"


def test_consensus_tie_visual_guard_can_only_veto_source_backed_patch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scorescan.selection_risk import SelectionRiskResult
    from scorescan.tie_consensus import TiePatchCalibration
    from scorescan.tie_visual_guard import TieVisualAudit
    from scorescan.visual_evidence import VisualMeasureEvidence
    import scorescan.consensus as consensus_module

    class AcceptTie:
        threshold = 0.975
        model_version = "test-tie"

        def calibrate(self, item):
            return TiePatchCalibration(1.0, self.threshold, True, self.model_version, 1.0)

    class RejectReplacement:
        threshold = 0.99
        model_version = "test-selection-risk"

        def calibrate(self, item):
            return SelectionRiskResult(0.0, self.threshold, False, self.model_version, 1.0)

    class RejectTieVisual:
        threshold = 0.92
        model_version = "test-tie-visual"

        def audit_transaction(self, evidence, base, patched):
            return TieVisualAudit(
                applicable=True,
                changed_tie_count=1,
                probability=0.1,
                threshold=self.threshold,
                accepted=False,
                reason="visual_tie_conflict",
                model_version=self.model_version,
            )

    monkeypatch.setattr(consensus_module, "TiePatchCalibrator", AcceptTie)
    monkeypatch.setattr(consensus_module, "SelectionRiskCalibrator", RejectReplacement)
    monkeypatch.setattr(consensus_module, "TieVisualGuard", RejectTieVisual)

    wrong = [(), (), (), ()]
    correct = [("start",), ("stop",), (), ()]
    candidates = []
    for index, (variant, states) in enumerate(
        (("primary", wrong), ("flat", correct), ("otsu", correct), ("upscale", correct))
    ):
        path = tmp_path / f"visual-{variant}.musicxml"
        write_single_measure_ties(path, states)
        candidates.append(Candidate(variant, str(path), 1050 - index * 10))

    evidence = VisualMeasureEvidence(
        page_index=0, system_index=0, measure_index=0, bbox=(0, 0, 10, 10),
        spacing=10.0, ink_density=0.0, nonstaff_ink_density=0.0,
        component_density=0.0, notehead_proxy=0.0, open_notehead_proxy=0.0,
        stem_proxy=0.0, beam_proxy=0.0, onset_proxy=0.0, compact_mark_proxy=0.0,
        accidental_proxy=0.0, above_ink_density=0.0, below_ink_density=0.0,
        x_ink_profile=(0.0,) * 8, staff_ink_profile=(0.0,) * 9,
        symbol_guard_image="non-empty-test-evidence",
    )
    output = tmp_path / "visual-veto.musicxml"
    report = build_measure_consensus(
        candidates, output, "primary", visual_evidence=(evidence,)
    )
    assert report is not None
    assert report.tie_patch_measure_count == 0
    assert report.tie_visual_guard_transaction_count == 1
    assert report.tie_visual_guard_rejected_count == 1
    assert report.votes[0].tie_visual_guard_applicable
    assert not report.votes[0].tie_visual_guard_accepted
    assert report.votes[0].tie_visual_guard_reason == "visual_tie_conflict"
    assert not report.votes[0].tie_patch_accepted
    assert not etree.parse(str(output)).findall(".//tie")


def test_patch_transaction_guard_rejects_composed_semantic_regression() -> None:
    original = etree.fromstring(
        b"<measure number='1'><attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>"
        b"<note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration><voice>1</voice><type>quarter</type></note></measure>"
    )
    safe = etree.fromstring(etree.tostring(original))
    note = safe.find("note")
    etree.SubElement(note, "tie", type="start")
    notations = etree.SubElement(note, "notations")
    etree.SubElement(notations, "tied", type="start")
    inherited = {"divisions": 1, "time": (4, 4), "key": None, "clef": None}
    accepted, reason = _patch_transaction_guard(original, safe, inherited)
    assert accepted
    assert reason == "validated"

    unsafe = etree.fromstring(etree.tostring(original))
    unsafe.find("note/type").text = "whole"
    accepted, reason = _patch_transaction_guard(original, unsafe, inherited)
    assert not accepted
    assert reason == "introduced_type_duration_mismatch"


def write_two_measure_cross_tie(path: Path, tied: bool) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    for number, pitches in enumerate((("G", "A", "B", "C"), ("C", "D", "E", "F")), start=1):
        measure = etree.SubElement(part, "measure", number=str(number))
        if number == 1:
            attributes = etree.SubElement(measure, "attributes")
            etree.SubElement(attributes, "divisions").text = "1"
            time = etree.SubElement(attributes, "time")
            etree.SubElement(time, "beats").text = "4"
            etree.SubElement(time, "beat-type").text = "4"
        for index, step in enumerate(pitches):
            note = etree.SubElement(measure, "note")
            pitch = etree.SubElement(note, "pitch")
            etree.SubElement(pitch, "step").text = step
            etree.SubElement(pitch, "octave").text = "4"
            etree.SubElement(note, "duration").text = "1"
            endpoint = None
            if tied and number == 1 and index == 3:
                endpoint = "start"
            elif tied and number == 2 and index == 0:
                endpoint = "stop"
            if endpoint:
                etree.SubElement(note, "tie", type=endpoint)
            etree.SubElement(note, "voice").text = "1"
            etree.SubElement(note, "type").text = "quarter"
            if endpoint:
                notations = etree.SubElement(note, "notations")
                etree.SubElement(notations, "tied", type=endpoint)
    etree.ElementTree(root).write(
        str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE
    )


def test_consensus_applies_cross_measure_tie_after_local_transactions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scorescan.cross_tie_consensus import CrossTiePatchCalibration
    from scorescan.selection_risk import SelectionRiskResult
    import scorescan.consensus as consensus_module

    class AcceptCrossTie:
        threshold = 0.98
        model_version = "test-cross-tie"

        def calibrate(self, item):
            return CrossTiePatchCalibration(1.0, self.threshold, True, self.model_version, 1.0)

    class RejectReplacement:
        threshold = 0.99
        model_version = "test-selection-risk"

        def calibrate(self, item):
            return SelectionRiskResult(0.0, self.threshold, False, self.model_version, 1.0)

    monkeypatch.setattr(consensus_module, "CrossTiePatchCalibrator", AcceptCrossTie)
    monkeypatch.setattr(consensus_module, "SelectionRiskCalibrator", RejectReplacement)

    candidates = []
    for index, (variant, tied) in enumerate(
        (("primary", False), ("flat", True), ("otsu", True), ("upscale", True))
    ):
        path = tmp_path / f"{variant}.musicxml"
        write_two_measure_cross_tie(path, tied)
        candidates.append(Candidate(variant, str(path), 1050 - index * 10))

    output = tmp_path / "consensus.musicxml"
    report = build_measure_consensus(candidates, output, "primary")
    assert report is not None
    assert report.replacements == 0
    assert report.cross_tie_patch_boundary_count == 1
    assert report.cross_tie_patch_endpoint_count == 2
    assert report.cross_tie_patch_model == "test-cross-tie"
    assert report.cross_tie_boundaries[0]["accepted"] is True
    measures = etree.parse(str(output)).findall("./part/measure")
    assert measures[0].findall("note")[-1].find("tie").get("type") == "start"
    assert measures[1].findall("note")[0].find("tie").get("type") == "stop"


def test_sparse_measure_candidate_cannot_manufacture_cross_tie_family_vote(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scorescan.cross_tie_consensus import CrossTiePatchCalibration
    from scorescan.selection_risk import SelectionRiskResult
    import scorescan.consensus as consensus_module

    class AcceptCrossTie:
        threshold = 0.0
        model_version = "test-cross-tie"

        def calibrate(self, item):
            return CrossTiePatchCalibration(1.0, self.threshold, True, self.model_version, 1.0)

    class RejectReplacement:
        threshold = 0.99
        model_version = "test-selection-risk"

        def calibrate(self, item):
            return SelectionRiskResult(0.0, self.threshold, False, self.model_version, 1.0)

    monkeypatch.setattr(consensus_module, "CrossTiePatchCalibrator", AcceptCrossTie)
    monkeypatch.setattr(consensus_module, "SelectionRiskCalibrator", RejectReplacement)

    candidates = []
    for index, (variant, tied) in enumerate(
        (
            ("primary", False),
            ("flat", True),
            ("otsu", True),
            # This file is a complete-page splice, but the family only observed measure 1.
            # Its measure 2 and the boundary are copied from the template and cannot vote.
            ("measure_localized:1", True),
        )
    ):
        path = tmp_path / f"candidate_{index}.musicxml"
        write_two_measure_cross_tie(path, tied)
        candidates.append(Candidate(variant, str(path), 1050 - index * 10))

    output = tmp_path / "consensus.musicxml"
    report = build_measure_consensus(candidates, output, "primary")
    assert report is not None
    assert report.cross_tie_patch_boundary_count == 0
    assert report.cross_tie_boundaries[0]["accepted"] is False
    assert report.cross_tie_boundaries[0]["reason"] == "no_strict_boundary_family_majority"
    assert report.cross_tie_boundaries[0]["scope_abstaining_variants"] == ["measure_localized:1"]
    measures = etree.parse(str(output)).findall("./part/measure")
    assert measures[0].findall("note")[-1].find("tie") is None
    assert measures[1].findall("note")[0].find("tie") is None


def test_consensus_ignores_oversized_candidate_before_xml_parse(tmp_path: Path) -> None:
    primary = tmp_path / "primary.musicxml"
    oversized = tmp_path / "oversized.musicxml"
    output = tmp_path / "consensus.musicxml"
    write_score(primary, ["C", "D"])
    with oversized.open("wb") as handle:
        handle.truncate(64 * 1024 * 1024 + 1)
    report = build_measure_consensus(
        [
            Candidate("primary", str(primary), 1050),
            Candidate("flat", str(oversized), 1040),
        ],
        output,
        "primary",
    )
    assert report is not None
    assert report.candidate_count == 2
    assert report.eligible_candidate_count == 1
    assert output.exists()


def test_consensus_ignores_candidate_with_excessive_measure_count(tmp_path: Path) -> None:
    primary = tmp_path / "primary.musicxml"
    pathological = tmp_path / "pathological.musicxml"
    output = tmp_path / "consensus.musicxml"
    write_score(primary, ["C", "D"])
    write_score(pathological, ["C"] * 513)
    report = build_measure_consensus(
        [
            Candidate("primary", str(primary), 1050),
            Candidate("flat", str(pathological), 1040),
        ],
        output,
        "primary",
    )
    assert report is not None
    assert report.eligible_candidate_count == 1
    assert len(etree.parse(str(output)).findall("./part/measure")) == 2


def write_single_measure_tuplet(path: Path, enabled: bool) -> None:
    from scorescan.tuplet_xml import set_simple_tuplet_state

    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "6"
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"
    for index, (step, duration, note_type) in enumerate(
        zip(
            ["C", "D", "E", "F", "G", "A"],
            [2, 2, 2, 6, 6, 6],
            ["eighth", "eighth", "eighth", "quarter", "quarter", "quarter"],
            strict=True,
        )
    ):
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = str(duration)
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = note_type
        if enabled and index < 3:
            set_simple_tuplet_state(
                note,
                ratio=(3, 2),
                start=index == 0,
                stop=index == 2,
            )
    etree.ElementTree(root).write(
        str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE
    )


def test_consensus_applies_tuplet_patch_when_measure_replacement_is_vetoed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scorescan.selection_risk import SelectionRiskResult
    from scorescan.tuplet_consensus import TupletPatchCalibration
    import scorescan.consensus as consensus_module

    class AcceptTuplet:
        threshold = 0.985
        model_version = "test-tuplet"

        def calibrate(self, item):
            return TupletPatchCalibration(1.0, self.threshold, True, self.model_version, 1.0)

    class RejectReplacement:
        threshold = 0.99
        model_version = "test-selection-risk"

        def calibrate(self, item):
            return SelectionRiskResult(0.0, self.threshold, False, self.model_version, 1.0)

    monkeypatch.setattr(consensus_module, "TupletPatchCalibrator", AcceptTuplet)
    monkeypatch.setattr(consensus_module, "SelectionRiskCalibrator", RejectReplacement)

    candidates = []
    for index, (variant, enabled) in enumerate(
        (("primary", False), ("flat", True), ("otsu", True), ("upscale", True))
    ):
        path = tmp_path / f"{variant}.musicxml"
        write_single_measure_tuplet(path, enabled)
        candidates.append(Candidate(variant, str(path), 1050 - index * 10))

    output = tmp_path / "consensus.musicxml"
    report = build_measure_consensus(candidates, output, "primary")
    assert report is not None
    assert report.tuplet_patch_measure_count == 1
    assert report.tuplet_patch_event_count == 3
    assert report.tuplet_patch_group_count == 1
    assert report.votes[0].decision == "patch_tuplet_consensus"
    assert report.votes[0].tuplet_patch_accepted
    notes = etree.parse(str(output)).findall("./part/measure/note")
    assert [note.find("time-modification") is not None for note in notes] == [True, True, True, False, False, False]
    assert notes[0].find("./notations/tuplet").get("type") == "start"
    assert notes[2].find("./notations/tuplet").get("type") == "stop"


def test_consensus_applies_barline_patch_after_other_measure_disagreement(
    tmp_path: Path, monkeypatch
) -> None:
    from scorescan.barline_consensus import BarlinePatchInput

    class AcceptBarline:
        threshold = 0.0
        model_version = "test-barline"

        def calibrate(self, item: BarlinePatchInput):
            return type("Decision", (), {
                "probability": 1.0,
                "threshold": 0.0,
                "accepted": True,
                "model_version": "test-barline",
            })()

    monkeypatch.setattr("scorescan.consensus.BarlinePatchCalibrator", AcceptBarline)
    candidates = []
    for variant, step, repeat, score in (
        ("primary", "C", False, 1050),
        ("flat", "D", True, 1030),
        ("otsu", "E", True, 1020),
        ("upscale", "F", True, 1010),
    ):
        path = tmp_path / f"{variant}.musicxml"
        write_score(path, [step])
        if repeat:
            tree = etree.parse(str(path))
            measure = tree.getroot().find("./part/measure")
            assert measure is not None
            barline = etree.SubElement(measure, "barline", location="right")
            etree.SubElement(barline, "bar-style").text = "light-heavy"
            etree.SubElement(barline, "repeat", direction="backward")
            tree.write(str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE)
        candidates.append(Candidate(variant, str(path), score))

    output = tmp_path / "consensus.musicxml"
    report = build_measure_consensus(candidates, output, "primary")
    assert report is not None
    assert report.barline_patch_measure_count == 1
    assert report.barline_patch_location_count == 1
    assert report.barline_patch_repeat_count == 1
    vote = report.votes[0]
    assert vote.barline_patch_accepted
    assert vote.barline_patch_locations == ("right",)
    assert "barline" in vote.decision
    repeat = etree.parse(str(output)).getroot().find("./part/measure/barline/repeat")
    assert repeat is not None and repeat.get("direction") == "backward"


def write_single_measure_articulations(
    path: Path,
    pitches: list[str],
    articulation_events: dict[int, tuple[str, ...]],
) -> None:
    from scorescan.articulation_xml import set_articulation_topology

    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"
    notes: list[etree._Element] = []
    for step in pitches:
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "1"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
        notes.append(note)
    set_articulation_topology(
        notes,
        tuple(tuple(articulation_events.get(index, ())) for index in range(len(notes))),
    )
    etree.ElementTree(root).write(
        str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE
    )


def test_consensus_applies_event_local_articulation_patch_with_complementary_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scorescan.articulation_consensus import ArticulationPatchCalibration
    from scorescan.selection_risk import SelectionRiskResult
    import scorescan.consensus as consensus_module

    class AcceptArticulation:
        threshold = 0.99
        model_version = "test-articulation"

        def calibrate(self, item):
            return ArticulationPatchCalibration(1.0, self.threshold, True, self.model_version, 1.0)

    class RejectReplacement:
        threshold = 0.99
        model_version = "test-selection-risk"

        def calibrate(self, item):
            return SelectionRiskResult(0.0, self.threshold, False, self.model_version, 1.0)

    monkeypatch.setattr(consensus_module, "ArticulationPatchCalibrator", AcceptArticulation)
    monkeypatch.setattr(consensus_module, "SelectionRiskCalibrator", RejectReplacement)

    variants = {
        "primary": (["C", "D", "E", "F"], {}),
        "flat": (["C", "E", "E", "F"], {0: ("staccato",)}),
        "otsu": (["C", "D", "G", "F"], {0: ("staccato",)}),
        "upscale": (["C", "D", "E", "A"], {0: ("staccato",)}),
    }
    candidates = []
    for index, (variant, (pitches, markers)) in enumerate(variants.items()):
        path = tmp_path / f"{variant}.musicxml"
        write_single_measure_articulations(path, pitches, markers)
        candidates.append(Candidate(variant, str(path), 1050 - index * 10))

    output = tmp_path / "consensus.musicxml"
    report = build_measure_consensus(candidates, output, "primary")
    assert report is not None
    assert report.articulation_patch_measure_count == 1
    assert report.articulation_patch_event_count == 1
    assert report.articulation_patch_mark_count == 1
    assert report.votes[0].articulation_patch_accepted
    assert "articulation" in report.votes[0].decision
    root = etree.parse(str(output)).getroot()
    assert root.find("./part/measure/note/notations/articulations/staccato") is not None
    assert [node.text for node in root.findall("./part/measure/note/pitch/step")] == ["C", "D", "E", "F"]


def write_single_measure_ornaments(
    path: Path,
    pitches: list[str],
    ornament_events: dict[int, tuple[str, ...]],
) -> None:
    from scorescan.ornament_xml import set_ornament_topology

    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"
    notes: list[etree._Element] = []
    for step in pitches:
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "1"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
        notes.append(note)
    set_ornament_topology(
        notes,
        tuple(tuple(ornament_events.get(index, ())) for index in range(len(notes))),
    )
    etree.ElementTree(root).write(
        str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE
    )


def test_consensus_applies_event_local_ornament_patch_with_complementary_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scorescan.ornament_consensus import OrnamentPatchCalibration
    from scorescan.selection_risk import SelectionRiskResult
    import scorescan.consensus as consensus_module

    class AcceptOrnament:
        threshold = 0.995
        model_version = "test-ornament"

        def calibrate(self, item):
            return OrnamentPatchCalibration(1.0, self.threshold, True, self.model_version, 1.0)

    class RejectReplacement:
        threshold = 0.99
        model_version = "test-selection-risk"

        def calibrate(self, item):
            return SelectionRiskResult(0.0, self.threshold, False, self.model_version, 1.0)

    monkeypatch.setattr(consensus_module, "OrnamentPatchCalibrator", AcceptOrnament)
    monkeypatch.setattr(consensus_module, "SelectionRiskCalibrator", RejectReplacement)

    variants = {
        "primary": (["C", "D", "E", "F"], {}),
        "flat": (["C", "E", "E", "F"], {0: ("trill-mark",)}),
        "otsu": (["C", "D", "G", "F"], {0: ("trill-mark",)}),
        "upscale": (["C", "D", "E", "A"], {0: ("trill-mark",)}),
    }
    candidates = []
    for index, (variant, (pitches, markers)) in enumerate(variants.items()):
        path = tmp_path / f"ornament-{variant}.musicxml"
        write_single_measure_ornaments(path, pitches, markers)
        candidates.append(Candidate(variant, str(path), 1050 - index * 10))

    output = tmp_path / "ornament-consensus.musicxml"
    report = build_measure_consensus(candidates, output, "primary")
    assert report is not None
    assert report.ornament_patch_measure_count == 1
    assert report.ornament_patch_event_count == 1
    assert report.ornament_patch_mark_count == 1
    assert report.votes[0].ornament_patch_accepted
    assert "ornament" in report.votes[0].decision
    root = etree.parse(str(output)).getroot()
    assert root.find("./part/measure/note/notations/ornaments/trill-mark") is not None
    assert [node.text for node in root.findall("./part/measure/note/pitch/step")] == ["C", "D", "E", "F"]


def write_single_measure_grace(
    path: Path,
    pitches: list[str],
    grace_events: set[int],
) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"
    for index, step in enumerate(pitches):
        note = etree.SubElement(measure, "note")
        if index in grace_events:
            etree.SubElement(note, "grace")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        if index not in grace_events:
            etree.SubElement(note, "duration").text = "1"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
    etree.ElementTree(root).write(
        str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE
    )


def test_consensus_applies_simple_grace_patch_to_restore_meter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scorescan.grace_consensus import GracePatchCalibration
    from scorescan.selection_risk import SelectionRiskResult
    import scorescan.consensus as consensus_module

    class AcceptGrace:
        threshold = 0.90
        model_version = "test-grace"

        def calibrate(self, item):
            return GracePatchCalibration(1.0, self.threshold, True, self.model_version, 1.0)

    class RejectReplacement:
        threshold = 0.99
        model_version = "test-selection-risk"

        def calibrate(self, item):
            return SelectionRiskResult(0.0, self.threshold, False, self.model_version, 1.0)

    monkeypatch.setattr(consensus_module, "GracePatchCalibrator", AcceptGrace)
    monkeypatch.setattr(consensus_module, "SelectionRiskCalibrator", RejectReplacement)

    variants = {
        "primary": (["C", "D", "E", "F", "G"], set()),
        "flat": (["C", "E", "E", "F", "G"], {0}),
        "otsu": (["C", "D", "G", "F", "G"], {0}),
        "upscale": (["C", "D", "E", "A", "G"], {0}),
    }
    candidates = []
    for index, (variant, (pitches, grace_events)) in enumerate(variants.items()):
        path = tmp_path / f"grace-{variant}.musicxml"
        write_single_measure_grace(path, pitches, grace_events)
        candidates.append(Candidate(variant, str(path), 1050 - index * 10))

    output = tmp_path / "grace-consensus.musicxml"
    report = build_measure_consensus(candidates, output, "primary")
    assert report is not None
    assert report.grace_patch_measure_count == 1
    assert report.grace_patch_event_count == 1
    assert report.grace_patch_added_count == 1
    assert report.votes[0].grace_patch_accepted
    assert "grace" in report.votes[0].decision
    root = etree.parse(str(output)).getroot()
    notes = root.findall("./part/measure/note")
    assert notes[0].find("grace") is not None
    assert notes[0].find("duration") is None
    assert [node.text for node in root.findall("./part/measure/note/pitch/step")] == ["C", "D", "E", "F", "G"]


def write_single_measure_lyrics(
    path: Path,
    pitches: list[str],
    lyric_events: dict[int, tuple[str, str, str]],
) -> None:
    from scorescan.lyric_xml import LyricState, set_lyric_topology

    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"
    notes: list[etree._Element] = []
    for step in pitches:
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "1"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
        notes.append(note)
    topology = []
    for index in range(len(notes)):
        value = lyric_events.get(index)
        topology.append(LyricState(*value) if value else None)
    set_lyric_topology(notes, tuple(topology))
    etree.ElementTree(root).write(
        str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE
    )


def test_product_consensus_never_applies_out_of_scope_lyric_semantics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scorescan.selection_risk import SelectionRiskResult
    import scorescan.consensus as consensus_module

    class RejectReplacement:
        threshold = 0.99
        model_version = "test-selection-risk"

        def calibrate(self, item):
            return SelectionRiskResult(0.0, self.threshold, False, self.model_version, 1.0)

    monkeypatch.setattr(consensus_module, "SelectionRiskCalibrator", RejectReplacement)

    variants = {
        "primary": (["C", "D", "E", "F"], {}),
        "flat": (["C", "E", "E", "F"], {0: ("Hel", "begin", "")}),
        "otsu": (["C", "D", "G", "F"], {0: ("Hel", "begin", "")}),
        "upscale": (["C", "D", "E", "A"], {0: ("Hel", "begin", "")}),
    }
    candidates = []
    for index, (variant, (pitches, lyrics)) in enumerate(variants.items()):
        path = tmp_path / f"lyric-{variant}.musicxml"
        write_single_measure_lyrics(path, pitches, lyrics)
        candidates.append(Candidate(variant, str(path), 1050 - index * 10))

    output = tmp_path / "lyric-consensus.musicxml"
    report = build_measure_consensus(candidates, output, "primary")
    assert report is not None
    assert report.lyric_patch_measure_count == 0
    assert report.lyric_patch_event_count == 0
    assert report.lyric_patch_lyric_count == 0
    assert report.lyric_patch_model == "disabled_out_of_scope"
    assert not report.votes[0].lyric_patch_applicable
    assert not report.votes[0].lyric_patch_accepted
    assert report.votes[0].lyric_patch_reason == "disabled_out_of_scope"
    root = etree.parse(str(output)).getroot()
    assert root.find("./part/measure/note/lyric") is None
    assert [node.text for node in root.findall("./part/measure/note/pitch/step")] == ["C", "D", "E", "F"]


def test_consensus_reselects_complete_template_with_strict_family_majority(tmp_path: Path) -> None:
    primary = tmp_path / "primary.musicxml"
    flat = tmp_path / "flat.musicxml"
    otsu = tmp_path / "otsu.musicxml"
    upscale = tmp_path / "upscale.musicxml"
    localized = tmp_path / "localized.musicxml"
    output = tmp_path / "consensus.musicxml"
    write_score(primary, ["C", "D", "E"])
    for path in (flat, otsu, upscale, localized):
        write_score(path, ["C", "D", "E", "F"])
    report = build_measure_consensus(
        [
            Candidate("primary", str(primary), 1120),
            Candidate("flat", str(flat), 1080),
            Candidate("otsu", str(otsu), 1060),
            Candidate("upscale", str(upscale), 1040),
            Candidate("system_localized", str(localized), 1020),
        ],
        output,
        "primary",
        target_measure_count=4,
    )
    assert report is not None
    assert report.template_count_reselected is True
    assert report.template_variant == "flat"
    assert report.requested_measure_count == 4
    assert report.template_measure_count == 4
    assert report.template_count_family_support == 4
    assert report.template_count_eligible_family_count == 5
    assert len(etree.parse(str(output)).getroot().findall("./part/measure")) == 4


def test_consensus_correlated_duplicates_cannot_reselect_template_count(tmp_path: Path) -> None:
    primary = tmp_path / "primary.musicxml"
    flat = tmp_path / "flat.musicxml"
    deblock = tmp_path / "deblock.musicxml"
    otsu = tmp_path / "otsu.musicxml"
    output = tmp_path / "consensus.musicxml"
    write_score(primary, ["C", "D", "E"])
    write_score(flat, ["C", "D", "E", "F"])
    write_score(deblock, ["C", "D", "E", "F"])
    write_score(otsu, ["C", "D", "E"])
    report = build_measure_consensus(
        [
            Candidate("primary", str(primary), 1120),
            Candidate("flat", str(flat), 1080),
            Candidate("deblock", str(deblock), 1070),
            Candidate("otsu", str(otsu), 1060),
        ],
        output,
        "primary",
        target_measure_count=4,
    )
    assert report is not None
    assert report.template_count_reselected is False
    assert report.template_variant == "primary"
    assert report.template_count_family_support == 1
    assert len(etree.parse(str(output)).getroot().findall("./part/measure")) == 3


def test_consensus_invalid_sibling_makes_template_family_abstain(tmp_path: Path) -> None:
    primary = tmp_path / "primary.musicxml"
    flat = tmp_path / "flat.musicxml"
    deblock = tmp_path / "deblock.musicxml"
    otsu = tmp_path / "otsu.musicxml"
    upscale = tmp_path / "upscale.musicxml"
    localized = tmp_path / "localized.musicxml"
    output = tmp_path / "consensus.musicxml"
    write_score(primary, ["C", "D", "E"])
    for path in (flat, deblock, otsu, upscale, localized):
        write_score(path, ["C", "D", "E", "F"])
    report = build_measure_consensus(
        [
            Candidate("primary", str(primary), 1120),
            Candidate("flat", str(flat), 1110, True),
            Candidate("deblock", str(deblock), 1100, False),
            Candidate("otsu", str(otsu), 1080),
            Candidate("upscale", str(upscale), 1060),
            Candidate("system_localized", str(localized), 1040),
        ],
        output,
        "primary",
        target_measure_count=4,
    )
    assert report is not None
    assert report.template_count_reselected is True
    # Restoration abstains because one sibling is invalid; binary, scale and
    # localisation still form a strict 3/4 majority over complete families.
    assert report.template_count_family_support == 3
    assert report.template_count_eligible_family_count == 4
    assert report.template_variant == "otsu"


def test_measure_family_support_excludes_structurally_invalid_family() -> None:
    variants = ["primary", "flat", "deblock", "otsu", "upscale"]
    family_sizes = {"baseline": 1, "restoration": 2, "binary": 1, "scale": 1}
    healthy = {"baseline", "binary", "scale"}
    assert _committed_family_support(
        variants,
        {1, 2, 3, 4},
        family_sizes,
        healthy,
    ) == {"binary", "scale"}


def test_invalid_sibling_cannot_create_measure_majority(tmp_path: Path, monkeypatch) -> None:
    from scorescan.selection_risk import SelectionRiskCalibrator, SelectionRiskResult

    monkeypatch.setattr(
        SelectionRiskCalibrator,
        "calibrate",
        lambda self, item: SelectionRiskResult(1.0, 0.0, True, self.model_version, 1.0),
    )
    primary = tmp_path / "primary.musicxml"
    flat = tmp_path / "flat.musicxml"
    deblock = tmp_path / "deblock.musicxml"
    otsu = tmp_path / "otsu.musicxml"
    upscale = tmp_path / "upscale.musicxml"
    localized = tmp_path / "localized.musicxml"
    output = tmp_path / "consensus.musicxml"
    write_score(primary, ["C", "F"])
    for path in (flat, deblock, otsu, upscale):
        write_score(path, ["C", "E"])
    write_score(localized, ["C", "G"])
    report = build_measure_consensus(
        [
            Candidate("primary", str(primary), 1100),
            Candidate("flat", str(flat), 1090, True),
            Candidate("deblock", str(deblock), 1080, False),
            Candidate("otsu", str(otsu), 1070),
            Candidate("upscale", str(upscale), 1060),
            Candidate("system_localized", str(localized), 1050),
        ],
        output,
        "primary",
    )
    assert report is not None
    vote = report.votes[1]
    assert vote.exact_family_support == 2
    assert vote.eligible_family_count == 4
    assert vote.abstaining_families == ("restoration",)
    assert vote.strict_majority is False
    assert vote.decision == "retain_template_no_majority"
    assert report.replacements == 0
    assert etree.parse(str(output)).getroot().findtext("./part/measure[2]/note/pitch/step") == "F"



def write_score_with_preserved_decoration(
    path: Path,
    step: str,
    *,
    stem: str | None = None,
    beam: str | None = None,
    fermata: bool = False,
) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "1"
    etree.SubElement(time, "beat-type").text = "4"
    note = etree.SubElement(measure, "note")
    pitch = etree.SubElement(note, "pitch")
    etree.SubElement(pitch, "step").text = step
    etree.SubElement(pitch, "octave").text = "4"
    etree.SubElement(note, "duration").text = "1"
    etree.SubElement(note, "voice").text = "1"
    etree.SubElement(note, "type").text = "quarter"
    if stem is not None:
        etree.SubElement(note, "stem").text = stem
    if beam is not None:
        etree.SubElement(note, "beam", number="1").text = beam
    if fermata:
        notations = etree.SubElement(note, "notations")
        etree.SubElement(notations, "fermata", type="upright").text = "normal"
    etree.ElementTree(root).write(
        str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE
    )


def test_semantic_whole_measure_replacement_falls_back_to_narrow_patch_without_preservation_support(tmp_path: Path) -> None:
    primary = tmp_path / "primary.musicxml"
    flat = tmp_path / "flat.musicxml"
    otsu = tmp_path / "otsu.musicxml"
    upscale = tmp_path / "upscale.musicxml"
    output = tmp_path / "consensus.musicxml"
    write_score_with_preserved_decoration(primary, "F", fermata=True)
    write_score_with_preserved_decoration(flat, "E", stem="up")
    write_score_with_preserved_decoration(otsu, "E", stem="down")
    write_score_with_preserved_decoration(upscale, "E", fermata=True)

    report = build_measure_consensus(
        [
            Candidate("primary", str(primary), 1050),
            Candidate("flat", str(flat), 1040),
            Candidate("otsu", str(otsu), 1030),
            Candidate("upscale", str(upscale), 1020),
        ],
        output,
        "primary",
    )

    assert report is not None
    vote = report.votes[0]
    assert vote.preservation_gate_required is True
    assert vote.preservation_gate_accepted is False
    assert vote.selected_preservation_family_support == 1
    # Whole-measure copying is vetoed, but the narrow pitch patch may still update the
    # modelled pitch while preserving the template's unmodelled fermata.
    assert vote.decision == "patch_pitch_consensus"
    tree = etree.parse(str(output))
    assert tree.findtext("./part/measure/note/pitch/step") == "E"
    assert tree.find("./part/measure/note/notations/fermata") is not None


def test_semantic_preservation_gate_accepts_two_family_writeback_support(tmp_path: Path) -> None:
    primary = tmp_path / "primary.musicxml"
    flat = tmp_path / "flat.musicxml"
    otsu = tmp_path / "otsu.musicxml"
    upscale = tmp_path / "upscale.musicxml"
    output = tmp_path / "consensus.musicxml"
    write_score_with_preserved_decoration(primary, "F")
    write_score_with_preserved_decoration(flat, "E", stem="up")
    write_score_with_preserved_decoration(otsu, "E", stem="up")
    write_score_with_preserved_decoration(upscale, "E", fermata=True)

    report = build_measure_consensus(
        [
            Candidate("primary", str(primary), 1050),
            Candidate("flat", str(flat), 1040),
            Candidate("otsu", str(otsu), 1030),
            Candidate("upscale", str(upscale), 1020),
        ],
        output,
        "primary",
    )

    assert report is not None
    vote = report.votes[0]
    assert vote.preservation_gate_required is True
    assert vote.preservation_gate_accepted is True
    assert vote.selected_preservation_family_support == 2
