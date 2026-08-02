from dataclasses import replace
from pathlib import Path

from lxml import etree

from scorescan.musicxml import MUSICXML_DOCTYPE
from scorescan.omr import EngineResult
from scorescan.measure_count_resolver import MeasureCountOption, MeasureCountResolution
from scorescan.recognition import (
    RecognitionCandidate,
    assess_candidate,
    choose_candidate,
    choose_lowres_upscale_internal_candidate,
    choose_lowres_upscale_center_fallback,
    choose_tuplet_cleanup_internal_candidate,
    inherit_sparse_page_evidence,
    promote_lowres_consensus_measure_count,
    raw_tuplet_candidate_warranted,
    reconcile_measure_count_resolution,
)


def write_score(path: Path, measures: int, bad_rhythm: bool = False) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    for number in range(1, measures + 1):
        measure = etree.SubElement(part, "measure", number=str(number))
        if number == 1:
            attributes = etree.SubElement(measure, "attributes")
            etree.SubElement(attributes, "divisions").text = "2"
            time = etree.SubElement(attributes, "time")
            etree.SubElement(time, "beats").text = "4"
            etree.SubElement(time, "beat-type").text = "4"
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = "C"
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "6" if bad_rhythm else "8"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "whole"
    etree.ElementTree(root).write(str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE)


def write_piano_ensemble_score(path: Path) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    for part_id, name in (("P1", "Piano"), ("P2", "Violin")):
        score_part = etree.SubElement(part_list, "score-part", id=part_id)
        etree.SubElement(score_part, "part-name").text = name
    piano = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(piano, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    etree.SubElement(attributes, "staves").text = "2"
    for step, octave, duration, voice, staff, backup in (
        ("C", "5", "4", "1", "1", "4"),
        ("E", "4", "2", "2", "1", "2"),
        ("C", "3", "4", "3", "2", ""),
    ):
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = octave
        etree.SubElement(note, "duration").text = duration
        etree.SubElement(note, "voice").text = voice
        etree.SubElement(note, "type").text = "whole" if duration == "4" else "half"
        etree.SubElement(note, "staff").text = staff
        if backup:
            backup_node = etree.SubElement(measure, "backup")
            etree.SubElement(backup_node, "duration").text = backup
    violin = etree.SubElement(root, "part", id="P2")
    measure = etree.SubElement(violin, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    note = etree.SubElement(measure, "note")
    pitch = etree.SubElement(note, "pitch")
    etree.SubElement(pitch, "step").text = "G"
    etree.SubElement(pitch, "octave").text = "4"
    etree.SubElement(note, "duration").text = "4"
    etree.SubElement(note, "voice").text = "1"
    etree.SubElement(note, "type").text = "whole"
    etree.ElementTree(root).write(
        str(path),
        encoding="UTF-8",
        xml_declaration=True,
        doctype=MUSICXML_DOCTYPE,
    )


def test_candidate_ranker_prefers_structural_match(tmp_path: Path) -> None:
    good = tmp_path / "good.musicxml"
    bad = tmp_path / "bad.musicxml"
    write_score(good, 4)
    write_score(bad, 2, bad_rhythm=True)
    good_candidate = assess_candidate("flat", tmp_path / "flat.png", EngineResult(0, good, 1.0), 4)
    bad_candidate = assess_candidate("otsu", tmp_path / "otsu.png", EngineResult(0, bad, 1.0), 4)
    assert good_candidate.score > bad_candidate.score
    assert choose_candidate([bad_candidate, good_candidate]) == good_candidate


def test_generalized_candidate_accepts_piano_voices_and_tracks_all_staves(tmp_path: Path) -> None:
    from scorescan.recognition import _page_preservation_signature

    score_path = tmp_path / "piano-ensemble.musicxml"
    changed_path = tmp_path / "piano-ensemble-changed.musicxml"
    write_piano_ensemble_score(score_path)
    write_piano_ensemble_score(changed_path)
    changed = etree.parse(str(changed_path))
    changed.find("./part[@id='P2']/measure/note/pitch/step").text = "A"
    changed.write(str(changed_path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE)

    candidate = assess_candidate(
        "primary",
        tmp_path / "page.png",
        EngineResult(0, score_path, 1.0),
        expected_measures=1,
        expected_physical_staves=3,
    )

    assert candidate.valid
    assert candidate.generalized_score
    assert candidate.part_count == 2
    assert candidate.physical_staff_count == 3
    assert candidate.staff_count_gap == 0
    assert candidate.multiple_voice_measure_count == 0
    assert candidate.rhythm_issue_count == 0
    assert _page_preservation_signature(score_path) != _page_preservation_signature(changed_path)


def test_generalized_candidate_penalizes_missing_physical_staff(tmp_path: Path) -> None:
    score_path = tmp_path / "piano-ensemble.musicxml"
    write_piano_ensemble_score(score_path)

    candidate = assess_candidate(
        "primary",
        tmp_path / "page.png",
        EngineResult(0, score_path, 1.0),
        expected_measures=1,
        expected_physical_staves=4,
    )

    assert candidate.staff_count_gap == 1
    assert candidate.staff_count_gap_penalty == 180.0


def test_candidate_layout_counts_repeated_staff_appearances_as_score_systems(
    tmp_path: Path,
) -> None:
    from scorescan.layout import PageLayout, StaffSystem

    score_path = tmp_path / "solo.musicxml"
    write_score(score_path, 30)
    physical = [
        StaffSystem(
            index=index,
            line_y=[y0 + offset for offset in (0, 10, 20, 30, 40)],
            top=y0 - 20,
            bottom=y0 + 60,
            left=80,
            right=1120,
            spacing=10,
            barlines=[80, 340, 600, 860, 1120],
            measure_count=5,
        )
        for index, y0 in enumerate((100, 190, 310, 400, 520, 610), start=1)
    ]
    layout = PageLayout(1200, 800, physical, 0.95)

    candidate = assess_candidate(
        "primary",
        tmp_path / "page.png",
        EngineResult(0, score_path, 1.0),
        expected_measures=5,
        expected_physical_staves=6,
        layout=layout,
    )

    assert candidate.expected_measure_count == 30
    assert candidate.measure_gap == 0
    assert candidate.expected_physical_staff_count == 1
    assert candidate.staff_count_gap == 0
    assert candidate.layout_score_system_count == 6
    assert candidate.layout_physical_staff_appearances == 6


def test_candidate_ranker_penalizes_semantic_corruption(tmp_path: Path) -> None:
    good = tmp_path / "good_semantic.musicxml"
    bad = tmp_path / "bad_semantic.musicxml"
    write_score(good, 1)
    write_score(bad, 1)
    tree = etree.parse(str(bad))
    note = tree.getroot().find("./part/measure/note")
    assert note is not None
    note.find("duration").text = "0"
    note.find("pitch/octave").text = "20"
    tree.write(str(bad), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE)
    good_candidate = assess_candidate("good", tmp_path / "good.png", EngineResult(0, good, 1.0), 1)
    bad_candidate = assess_candidate("bad", tmp_path / "bad.png", EngineResult(0, bad, 1.0), 1)
    assert bad_candidate.zero_duration_count == 1
    assert bad_candidate.pitch_outlier_count == 1
    assert good_candidate.score > bad_candidate.score


def test_initial_recognition_plan_covers_three_independent_families(tmp_path: Path) -> None:
    from scorescan.recognition import _initial_variant_names

    variants = [
        ("primary", tmp_path / "primary.png"),
        ("flat", tmp_path / "flat.png"),
        ("deblock", tmp_path / "deblock.png"),
        ("otsu", tmp_path / "otsu.png"),
        ("adaptive", tmp_path / "adaptive.png"),
        ("upscale", tmp_path / "upscale.png"),
    ]
    assert _initial_variant_names(variants) == ["primary", "flat", "otsu"]


def test_output_count_reconciliation_never_persists_unattained_target() -> None:
    resolution = MeasureCountResolution(
        selected_count=46,
        probability=0.97,
        margin=0.70,
        source="model",
        model_version="test",
        model_status="verified",
        layout_count=46,
        layout_confidence=0.98,
        deterministic_count=46,
        options=(
            MeasureCountOption(45, 0.22, 0.2, 0.2, 1, 1, 0.2),
            MeasureCountOption(46, 0.97, 0.8, 0.8, 4, 4, 0.9),
        ),
    )
    reconciled = reconcile_measure_count_resolution(
        resolution,
        45,
        source="output_candidate_fallback",
    )
    assert reconciled.selected_count == 45
    assert reconciled.probability == 0.22
    assert reconciled.margin == 0.0
    assert reconciled.source == "output_candidate_fallback"


def test_sparse_candidate_inherits_page_prior_without_getting_a_bonus() -> None:
    template = RecognitionCandidate(
        variant="primary",
        image_path="page.png",
        xml_path="primary.musicxml",
        score=1055.0,
        raw_score=1000.0,
        valid=True,
        elapsed_seconds=1.0,
        agreement_ratio=0.92,
        calibrated_probability=0.88,
        calibration_model="page-model",
        visual_compatibility=0.73,
        visual_calibration_model="visual-model",
    )
    sparse = RecognitionCandidate(
        variant="measure_localized:2",
        image_path="crop.png",
        xml_path="candidate.musicxml",
        score=1000.0,
        raw_score=990.0,
        valid=True,
        elapsed_seconds=1.0,
    )
    inherited = inherit_sparse_page_evidence(sparse, template)
    assert inherited.score == 1045.0
    assert inherited.score <= template.score
    assert inherited.agreement_ratio == template.agreement_ratio
    assert inherited.calibrated_probability == template.calibrated_probability
    assert inherited.visual_compatibility == template.visual_compatibility
    assert inherited.raw_score == sparse.raw_score


def test_invalid_sparse_candidate_never_inherits_template_confidence() -> None:
    template = RecognitionCandidate(
        variant="primary", image_path="p", xml_path="p.xml", score=1000,
        raw_score=1000, valid=True, elapsed_seconds=1, calibrated_probability=0.95,
    )
    invalid = RecognitionCandidate(
        variant="measure_localized:1", image_path="c", xml_path=None, score=-10000,
        raw_score=-10000, valid=False, elapsed_seconds=1, calibrated_probability=0.5,
    )
    assert inherit_sparse_page_evidence(invalid, template) is invalid


def test_lowres_upscale_internal_consensus_selects_strict_exact_majority(
    tmp_path: Path,
) -> None:
    correct_a = tmp_path / "correct-a.musicxml"
    correct_b = tmp_path / "correct-b.musicxml"
    wrong = tmp_path / "wrong.musicxml"
    write_score(correct_a, 5)
    write_score(correct_b, 5)
    write_score(wrong, 3)
    internal = [
        assess_candidate(
            "upscale:low",
            tmp_path / "low.png",
            EngineResult(0, correct_a, 1.0),
            3,
        ),
        assess_candidate(
            "upscale",
            tmp_path / "mid.png",
            EngineResult(0, wrong, 1.0),
            3,
        ),
        assess_candidate(
            "upscale:high",
            tmp_path / "high.png",
            EngineResult(0, correct_b, 1.0),
            3,
        ),
    ]

    selected = choose_lowres_upscale_internal_candidate(internal)

    assert selected is not None
    assert selected.variant == "upscale"
    assert selected.measure_count == 5
    assert selected.internal_consensus_support == 2
    assert selected.internal_consensus_total == 3
    assert selected.elapsed_seconds == 3.0


def test_lowres_upscale_internal_consensus_abstains_on_three_way_split(
    tmp_path: Path,
) -> None:
    internal = []
    for index, measures in enumerate((2, 3, 4), start=1):
        path = tmp_path / f"split-{index}.musicxml"
        write_score(path, measures)
        internal.append(
            assess_candidate(
                f"upscale:{index}",
                tmp_path / f"{index}.png",
                EngineResult(0, path, 1.0),
                3,
            )
        )

    assert choose_lowres_upscale_internal_candidate(internal) is None


def test_lowres_upscale_split_keeps_unboosted_center_candidate(
    tmp_path: Path,
) -> None:
    internal: list[RecognitionCandidate] = []
    for variant, measures, raw_score in (
        ("upscale:low", 2, 1100.0),
        ("upscale", 3, 1000.0),
        ("upscale:high", 4, 1200.0),
    ):
        xml_path = tmp_path / f"{variant.replace(':', '_')}.musicxml"
        write_score(xml_path, measures)
        internal.append(
            RecognitionCandidate(
                variant=variant,
                image_path=f"{variant}.png",
                xml_path=str(xml_path),
                score=raw_score,
                raw_score=raw_score,
                valid=True,
                elapsed_seconds=1.25,
            )
        )

    selected = choose_lowres_upscale_center_fallback(internal)

    assert selected is not None
    assert selected.variant == "upscale"
    assert selected.raw_score == 1000.0
    assert selected.score == 1000.0
    assert selected.internal_consensus_support == 1
    assert selected.internal_consensus_total == 3
    assert selected.elapsed_seconds == 3.75


def test_raw_tuplet_candidate_runs_only_for_dense_high_rhythm_risk_page() -> None:
    risky = RecognitionCandidate(
        variant="staffnorm",
        image_path="page.png",
        xml_path="page.musicxml",
        score=100.0,
        raw_score=100.0,
        valid=True,
        elapsed_seconds=1.0,
        measure_count=20,
        note_count=200,
        rhythm_issue_count=4,
        generalized_score=True,
    )
    safe = replace(risky, rhythm_issue_count=3)

    assert raw_tuplet_candidate_warranted(risky)
    assert not raw_tuplet_candidate_warranted(safe)


def test_tuplet_cleanup_internal_selection_keeps_only_stronger_sibling() -> None:
    cleaned = RecognitionCandidate(
        variant="staffnorm",
        image_path="cleaned.png",
        xml_path="cleaned.musicxml",
        score=100.0,
        raw_score=100.0,
        valid=True,
        elapsed_seconds=2.0,
    )
    raw = replace(
        cleaned,
        variant="staffnorm:raw_tuplets",
        image_path="raw.png",
        xml_path="raw.musicxml",
        score=125.0,
        raw_score=125.0,
        elapsed_seconds=3.0,
    )

    selected = choose_tuplet_cleanup_internal_candidate([cleaned, raw])

    assert selected is not None
    assert selected.variant == "staffnorm:raw_tuplets"
    assert selected.elapsed_seconds == 5.0
    assert selected.internal_consensus_support == 1
    assert selected.internal_consensus_total == 2


def test_clean_lowres_internal_consensus_can_escape_weak_count_prior() -> None:
    resolution = MeasureCountResolution(
        selected_count=3,
        probability=0.95,
        margin=0.7,
        source="model",
        model_version="test",
        model_status="verified",
        layout_count=3,
        layout_confidence=0.3,
        deterministic_count=3,
        options=(
            MeasureCountOption(3, 0.95, 0.8, 0.8, 3, 4, 1.0),
            MeasureCountOption(5, 0.2, 0.2, 0.2, 1, 1, 0.4),
        ),
    )
    correlated = RecognitionCandidate(
        variant="primary",
        image_path="primary.png",
        xml_path="primary.musicxml",
        score=941.5,
        raw_score=941.5,
        valid=True,
        elapsed_seconds=1.0,
        measure_count=3,
        note_count=19,
        rhythm_issue_count=2,
    )
    scale_consensus = RecognitionCandidate(
        variant="upscale",
        image_path="upscale.png",
        xml_path="upscale.musicxml",
        score=923.7,
        raw_score=923.7,
        valid=True,
        elapsed_seconds=3.0,
        measure_count=5,
        note_count=19,
        measure_gap_penalty=84.0,
        internal_consensus_support=2,
        internal_consensus_total=2,
    )

    promoted = promote_lowres_consensus_measure_count(
        resolution,
        [correlated, scale_consensus],
    )

    assert promoted.selected_count == 5
    assert promoted.source == "lowres_internal_consensus"
    assert promoted.margin == 0.0
