from __future__ import annotations

import importlib.util
import json

import pytest
from pathlib import Path

from lxml import etree

from scorescan.musicxml import MUSICXML_DOCTYPE

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "evaluate_release_dataset.py"
spec = importlib.util.spec_from_file_location("scorescan_release_dataset_evaluator", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def _write_score(path: Path, steps: list[str]) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    for number, step in enumerate(steps, start=1):
        measure = etree.SubElement(part, "measure", number=str(number))
        if number == 1:
            attributes = etree.SubElement(measure, "attributes")
            etree.SubElement(attributes, "divisions").text = "1"
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "1"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
    etree.ElementTree(root).write(str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE)


def _write_single_measure(path: Path, steps: list[str]) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    for step in steps:
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "1"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
    etree.ElementTree(root).write(str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE)


def test_release_dataset_evaluator_aggregates_counts_and_gates(tmp_path: Path) -> None:
    ref_a = tmp_path / "ref-a.musicxml"
    cand_a = tmp_path / "cand-a.musicxml"
    ref_b = tmp_path / "ref-b.musicxml"
    cand_b = tmp_path / "cand-b.musicxml"
    _write_score(ref_a, ["C", "D"])
    _write_score(cand_a, ["C", "D"])
    _write_score(ref_b, ["E", "F"])
    _write_score(cand_b, ["E", "G"])
    manifest = tmp_path / "benchmark.json"
    manifest.write_text(json.dumps({
        "format": 1,
        "name": "fixture",
        "bootstrap_samples": 32,
        "cases": [
            {
                "id": "a",
                "reference": ref_a.name,
                "candidate": cand_a.name,
                "split": "test",
                "source_group": "score-a",
                "strata": {"scan_quality": "clean", "density": "low"},
            },
            {
                "id": "b",
                "reference": ref_b.name,
                "candidate": cand_b.name,
                "split": "test",
                "source_group": "score-b",
                "strata": {"scan_quality": "degraded", "density": "low"},
            },
        ],
        "gates": {
            "minimum": {"pitch_accuracy_aligned": 0.70},
            "maximum": {"event_error_rate": 0.30},
        },
    }), encoding="utf-8")

    report = module.evaluate_manifest(manifest, split="test")

    assert report["case_count"] == 2
    assert report["production_scope_coverage"] == {
        "case_unit": "one_submitted_document",
        "document_count": 2,
        "page_count": 2,
        "verified_unique_scan_page_count": 0,
        "duplicate_scan_page_count": 0,
        "unverified_scan_page_identity_count": 2,
        "source_group_count": 2,
        "score_configuration_stratum": "score_configuration",
        "pages_by_score_configuration": {
            "solo_monophonic": 0,
            "piano": 0,
            "monophonic_ensemble": 0,
            "piano_plus_monophonic_ensemble": 0,
        },
        "scope_classified_page_count": 0,
        "out_of_contract_page_count": 2,
    }
    assert report["aggregate"]["pitch_accuracy_aligned"] == 0.75
    assert report["aggregate"]["event_error_rate"] == 0.25
    assert report["aggregate"]["event_presence_precision"] == 1.0
    assert report["aggregate"]["event_presence_recall"] == 1.0
    assert report["aggregate"]["event_presence_f1"] == 1.0
    assert report["aggregate"]["deleted_event_rate"] == 0.0
    assert report["aggregate"]["inserted_event_rate"] == 0.0
    assert report["release_gate"]["passed"] is True
    assert len(report["benchmark_fingerprint"]) == 64
    assert len(report["manifest_sha256"]) == 64
    assert report["cases"][0]["source_group"] == "score-a"
    assert report["cases"][0]["strata"] == {"density": "low", "scan_quality": "clean"}
    assert report["stratified"]["density"]["low"]["case_count"] == 2
    assert report["stratified"]["scan_quality"]["clean"]["pitch_accuracy_aligned"] == 1.0
    assert report["stratified"]["scan_quality"]["degraded"]["pitch_accuracy_aligned"] == 0.5
    assert "pitch_accuracy_aligned" in report["bootstrap_95"]
    assert "event_presence_recall" in report["bootstrap_95"]
    assert "event_presence_precision" in report["bootstrap_95"]

    strict = module.evaluate_manifest(
        manifest,
        split="test",
        bootstrap_seed_override=17,
        enforce_stable_gate=True,
    )
    assert strict["bootstrap_seed"] == 17
    assert strict["release_gate"]["profile"] == "stable-v1"
    assert strict["release_gate"]["passed"] is False
    assert any(item["metric"] == "case_count" for item in strict["release_gate"]["checks"])

    production = module.evaluate_manifest(
        manifest,
        split="test",
        bootstrap_samples_override=4,
        gate_profile="production-v2",
    )
    assert production["release_gate"]["profile"] == "production-v2"
    assert production["release_gate"]["passed"] is False
    with pytest.raises(ValueError, match="choose either"):
        module.evaluate_manifest(
            manifest,
            enforce_stable_gate=True,
            gate_profile="production-v2",
        )

    leaking_manifest = tmp_path / "leaking.json"
    leaking_manifest.write_text(json.dumps({
        "format": 1,
        "cases": [
            {"id": "train", "reference": ref_a.name, "candidate": cand_a.name, "split": "train", "source_group": "same-score"},
            {"id": "test", "reference": ref_b.name, "candidate": cand_b.name, "split": "test", "source_group": "same-score"},
        ],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="source groups cross benchmark splits"):
        module.evaluate_manifest(leaking_manifest, split="test")

    invalid_strata_manifest = tmp_path / "invalid-strata.json"
    invalid_strata_manifest.write_text(json.dumps({
        "format": 1,
        "cases": [{
            "id": "invalid",
            "reference": ref_a.name,
            "candidate": cand_a.name,
            "strata": {"quality": ["not", "scalar"]},
        }],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a scalar"):
        module.evaluate_manifest(invalid_strata_manifest)


def test_release_dataset_evaluator_separates_insertions_and_deletions(tmp_path: Path) -> None:
    reference = tmp_path / "reference.musicxml"
    deleted = tmp_path / "deleted.musicxml"
    inserted = tmp_path / "inserted.musicxml"
    _write_single_measure(reference, ["C", "D", "E", "F"])
    _write_single_measure(deleted, ["C", "D", "F"])
    _write_single_measure(inserted, ["C", "D", "E", "F", "G"])

    deleted_report = module.aggregate_reports([module.compare_musicxml(reference, deleted)])
    assert deleted_report["deleted_events"] == 1
    assert deleted_report["inserted_events"] == 0
    assert deleted_report["event_presence_precision"] == 1.0
    assert deleted_report["event_presence_recall"] == 0.75
    assert deleted_report["deleted_event_rate"] == 0.25
    assert deleted_report["inserted_event_rate"] == 0.0

    inserted_report = module.aggregate_reports([module.compare_musicxml(reference, inserted)])
    assert inserted_report["deleted_events"] == 0
    assert inserted_report["inserted_events"] == 1
    assert inserted_report["event_presence_precision"] == 0.8
    assert inserted_report["event_presence_recall"] == 1.0
    assert inserted_report["deleted_event_rate"] == 0.0
    assert inserted_report["inserted_event_rate"] == 0.2


def _write_chord_measure(path: Path, topology: list[bool]) -> None:
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
    for marker, step in zip(topology, ["C", "E", "G", "D", "F"], strict=True):
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


def test_release_dataset_evaluator_reports_chord_topology_accuracy(tmp_path: Path) -> None:
    reference = tmp_path / "reference-chord.musicxml"
    candidate = tmp_path / "candidate-chord.musicxml"
    _write_chord_measure(reference, [False, True, False, False, False])
    _write_chord_measure(candidate, [False, False, False, False, False])

    report = module.compare_musicxml(reference, candidate)
    assert report["chord_topology_accuracy_aligned"] == 0.8
    aggregate = module.aggregate_reports([report])
    assert aggregate["chord_topology_accuracy_aligned"] == 0.8
    intervals = module.bootstrap_intervals([report], samples=8, seed=3)
    assert intervals["chord_topology_accuracy_aligned"]["low_95"] == 0.8
    assert module.STABLE_RELEASE_GATES["minimum"]["chord_topology_accuracy_aligned"] == 0.985


def _write_reordered_two_staff_chords(
    path: Path,
    *,
    staff_order: tuple[int, int],
    reverse_pitch_order: bool,
) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Piano"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    etree.SubElement(attributes, "staves").text = "2"
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "1"
    etree.SubElement(time, "beat-type").text = "4"
    for clef_number, sign, line in ((1, "G", "2"), (2, "F", "4")):
        clef = etree.SubElement(attributes, "clef", number=str(clef_number))
        etree.SubElement(clef, "sign").text = sign
        etree.SubElement(clef, "line").text = line

    pitches = {1: (("C", "4"), ("E", "4")), 2: (("G", "2"), ("B", "2"))}
    for group_index, staff in enumerate(staff_order):
        if group_index:
            backup = etree.SubElement(measure, "backup")
            etree.SubElement(backup, "duration").text = "1"
        ordered = list(pitches[staff])
        if reverse_pitch_order:
            ordered.reverse()
        for pitch_index, (step, octave) in enumerate(ordered):
            note = etree.SubElement(measure, "note")
            if pitch_index:
                etree.SubElement(note, "chord")
            pitch = etree.SubElement(note, "pitch")
            etree.SubElement(pitch, "step").text = step
            etree.SubElement(pitch, "octave").text = octave
            etree.SubElement(note, "duration").text = "1"
            etree.SubElement(note, "voice").text = "1" if staff == 1 else "5"
            etree.SubElement(note, "type").text = "quarter"
            etree.SubElement(note, "staff").text = str(staff)
    etree.ElementTree(root).write(
        str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE
    )


def test_polyphonic_event_metrics_ignore_equivalent_xml_serialization_order(tmp_path: Path) -> None:
    reference = tmp_path / "reference-two-staff.musicxml"
    candidate = tmp_path / "candidate-two-staff.musicxml"
    _write_reordered_two_staff_chords(
        reference,
        staff_order=(1, 2),
        reverse_pitch_order=False,
    )
    _write_reordered_two_staff_chords(
        candidate,
        staff_order=(2, 1),
        reverse_pitch_order=True,
    )

    report = module.compare_musicxml(reference, candidate)
    assert report["pitch_accuracy_aligned"] == 1.0
    assert report["rhythm_accuracy_aligned"] == 1.0
    assert report["chord_topology_accuracy_aligned"] == 1.0
    assert report["staff_assignment_accuracy_aligned"] == 1.0
    assert report["event_error_rate"] == 0.0


def _write_tie_measure(path: Path, states: list[tuple[str, ...]]) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = str(len(states))
    etree.SubElement(time, "beat-type").text = "4"
    for state, step in zip(states, ["C", "C", "D", "E"], strict=True):
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


def test_release_dataset_evaluator_reports_tie_endpoint_metrics(tmp_path: Path) -> None:
    reference = tmp_path / "reference-tie.musicxml"
    missing = tmp_path / "missing-tie.musicxml"
    spurious = tmp_path / "spurious-tie.musicxml"
    correct = [("start",), ("stop",), (), ()]
    none = [(), (), (), ()]
    _write_tie_measure(reference, correct)
    _write_tie_measure(missing, none)
    _write_tie_measure(spurious, [("start",), ("stop",), ("start",), ("stop",)])

    missing_report = module.compare_musicxml(reference, missing)
    assert missing_report["tie_topology_accuracy_aligned"] == 0.5
    assert missing_report["tie_endpoint_precision"] == 1.0
    assert missing_report["tie_endpoint_recall"] == 0.0
    assert missing_report["tie_endpoint_f1"] == 0.0

    spurious_report = module.compare_musicxml(reference, spurious)
    assert spurious_report["tie_endpoint_precision"] == 0.5
    assert spurious_report["tie_endpoint_recall"] == 1.0
    assert spurious_report["tie_endpoint_f1"] == pytest.approx(2 / 3)

    aggregate = module.aggregate_reports([spurious_report])
    assert aggregate["tie_endpoint_precision"] == 0.5
    intervals = module.bootstrap_intervals([spurious_report], samples=8, seed=5)
    assert intervals["tie_endpoint_f1"]["low_95"] == pytest.approx(2 / 3)
    assert module.STABLE_RELEASE_GATES["minimum"]["tie_endpoint_f1"] == 0.97


def _write_cross_tie_score(path: Path, tied: bool) -> None:
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


def test_release_dataset_evaluator_reports_cross_measure_tie_metrics(tmp_path: Path) -> None:
    reference = tmp_path / "reference-cross-tie.musicxml"
    exact = tmp_path / "exact-cross-tie.musicxml"
    missing = tmp_path / "missing-cross-tie.musicxml"
    spurious_reference = tmp_path / "reference-no-cross-tie.musicxml"
    spurious = tmp_path / "spurious-cross-tie.musicxml"
    _write_cross_tie_score(reference, True)
    _write_cross_tie_score(exact, True)
    _write_cross_tie_score(missing, False)
    _write_cross_tie_score(spurious_reference, False)
    _write_cross_tie_score(spurious, True)

    exact_report = module.compare_musicxml(reference, exact)
    assert exact_report["cross_tie_boundary_accuracy"] == 1.0
    assert exact_report["cross_tie_precision"] == 1.0
    assert exact_report["cross_tie_recall"] == 1.0
    assert exact_report["cross_tie_f1"] == 1.0

    missing_report = module.compare_musicxml(reference, missing)
    assert missing_report["cross_tie_boundary_accuracy"] == 0.0
    assert missing_report["cross_tie_precision"] == 1.0
    assert missing_report["cross_tie_recall"] == 0.0
    assert missing_report["cross_tie_f1"] == 0.0

    spurious_report = module.compare_musicxml(spurious_reference, spurious)
    assert spurious_report["cross_tie_boundary_accuracy"] == 0.0
    assert spurious_report["cross_tie_precision"] == 0.0
    assert spurious_report["cross_tie_recall"] == 1.0
    assert spurious_report["cross_tie_f1"] == 0.0

    aggregate = module.aggregate_reports([exact_report])
    assert aggregate["cross_tie_f1"] == 1.0
    intervals = module.bootstrap_intervals([exact_report], samples=8, seed=7)
    assert intervals["cross_tie_f1"]["low_95"] == 1.0
    assert module.STABLE_RELEASE_GATES["minimum"]["cross_tie_f1"] == 0.97


def _write_tuplet_measure(path: Path, enabled: bool) -> None:
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
            set_simple_tuplet_state(note, ratio=(3, 2), start=index == 0, stop=index == 2)
    etree.ElementTree(root).write(
        str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE
    )


def test_release_dataset_evaluator_reports_tuplet_metrics(tmp_path: Path) -> None:
    reference = tmp_path / "reference-tuplet.musicxml"
    candidate = tmp_path / "candidate-tuplet.musicxml"
    _write_tuplet_measure(reference, True)
    _write_tuplet_measure(candidate, False)

    report = module.compare_musicxml(reference, candidate)
    assert report["tuplet_topology_accuracy_aligned"] == 0.5
    assert report["tuplet_event_precision"] == 1.0
    assert report["tuplet_event_recall"] == 0.0
    assert report["tuplet_event_f1"] == 0.0
    aggregate = module.aggregate_reports([report])
    assert aggregate["tuplet_topology_accuracy_aligned"] == 0.5
    assert aggregate["tuplet_event_recall"] == 0.0
    intervals = module.bootstrap_intervals([report], samples=8, seed=7)
    assert intervals["tuplet_event_f1"]["low_95"] == 0.0
    assert module.STABLE_RELEASE_GATES["minimum"]["tuplet_event_f1"] == 0.97


def _write_slur_measure(path: Path, arcs: list[tuple[int, int, int]]) -> None:
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
    for step in ["C", "D", "E", "F"]:
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "1"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
        notes.append(note)
    for start, stop, number in arcs:
        start_notations = notes[start].find("notations")
        if start_notations is None:
            start_notations = etree.SubElement(notes[start], "notations")
        etree.SubElement(start_notations, "slur", type="start", number=str(number))
        stop_notations = notes[stop].find("notations")
        if stop_notations is None:
            stop_notations = etree.SubElement(notes[stop], "notations")
        etree.SubElement(stop_notations, "slur", type="stop", number=str(number))
    etree.ElementTree(root).write(
        str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE
    )


def test_release_dataset_evaluator_reports_slur_metrics(tmp_path: Path) -> None:
    reference = tmp_path / "reference-slur.musicxml"
    exact = tmp_path / "exact-slur.musicxml"
    missing = tmp_path / "missing-slur.musicxml"
    empty_reference = tmp_path / "empty-reference-slur.musicxml"
    spurious = tmp_path / "spurious-slur.musicxml"
    _write_slur_measure(reference, [(0, 2, 1)])
    _write_slur_measure(exact, [(0, 2, 8)])
    _write_slur_measure(missing, [])
    _write_slur_measure(empty_reference, [])
    _write_slur_measure(spurious, [(0, 2, 3)])

    exact_report = module.compare_musicxml(reference, exact)
    assert exact_report["slur_topology_accuracy"] == 1.0
    assert exact_report["slur_endpoint_f1"] == 1.0

    missing_report = module.compare_musicxml(reference, missing)
    assert missing_report["slur_topology_accuracy"] == 0.0
    assert missing_report["slur_endpoint_precision"] == 1.0
    assert missing_report["slur_endpoint_recall"] == 0.0
    assert missing_report["slur_endpoint_f1"] == 0.0

    spurious_report = module.compare_musicxml(empty_reference, spurious)
    assert spurious_report["slur_endpoint_precision"] == 0.0
    assert spurious_report["slur_endpoint_recall"] == 1.0
    assert spurious_report["slur_endpoint_f1"] == 0.0

    aggregate = module.aggregate_reports([exact_report])
    assert aggregate["slur_endpoint_f1"] == 1.0
    intervals = module.bootstrap_intervals([exact_report], samples=8, seed=11)
    assert intervals["slur_endpoint_f1"]["low_95"] == 1.0
    assert module.STABLE_RELEASE_GATES["minimum"]["slur_endpoint_f1"] == 0.97


def _write_repeat_measure(path: Path, markers: list[tuple[str, str, str]]) -> None:
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
    for step in "CDEF":
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "1"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
    for location, style, direction in markers:
        barline = etree.SubElement(measure, "barline", location=location)
        etree.SubElement(barline, "bar-style").text = style
        if direction:
            etree.SubElement(barline, "repeat", direction=direction)
    etree.ElementTree(root).write(
        str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE
    )


def test_release_dataset_evaluator_reports_repeat_marker_metrics(tmp_path: Path) -> None:
    reference = tmp_path / "reference-repeat.musicxml"
    missing = tmp_path / "missing-repeat.musicxml"
    spurious_reference = tmp_path / "reference-no-repeat.musicxml"
    spurious = tmp_path / "spurious-repeat.musicxml"
    markers = [("left", "heavy-light", "forward"), ("right", "light-heavy", "backward")]
    _write_repeat_measure(reference, markers)
    _write_repeat_measure(missing, [("left", "heavy-light", "forward")])
    _write_repeat_measure(spurious_reference, [])
    _write_repeat_measure(spurious, [("right", "light-heavy", "backward")])

    missing_report = module.compare_musicxml(reference, missing)
    assert missing_report["barline_accuracy"] == 0.0
    assert missing_report["repeat_marker_precision"] == 1.0
    assert missing_report["repeat_marker_recall"] == 0.5
    assert missing_report["repeat_marker_f1"] == pytest.approx(2 / 3)

    spurious_report = module.compare_musicxml(spurious_reference, spurious)
    assert spurious_report["repeat_marker_precision"] == 0.0
    assert spurious_report["repeat_marker_recall"] == 1.0
    assert spurious_report["repeat_marker_f1"] == 0.0

    aggregate = module.aggregate_reports([missing_report])
    assert aggregate["repeat_marker_f1"] == pytest.approx(2 / 3)
    intervals = module.bootstrap_intervals([missing_report], samples=8, seed=7)
    assert intervals["repeat_marker_f1"]["low_95"] == pytest.approx(2 / 3)
    gates = module.STABLE_RELEASE_GATES["minimum"]
    assert gates["barline_accuracy"] == 0.99
    assert gates["repeat_marker_f1"] == 0.97


def _write_articulation_measure(path: Path, markers: dict[int, tuple[str, ...]]) -> None:
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
    for step in "CDEF":
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "1"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
        notes.append(note)
    set_articulation_topology(notes, tuple(tuple(markers.get(index, ())) for index in range(len(notes))))
    etree.ElementTree(root).write(
        str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE
    )


def test_release_dataset_evaluator_reports_articulation_metrics(tmp_path: Path) -> None:
    reference = tmp_path / "reference-articulation.musicxml"
    exact = tmp_path / "exact-articulation.musicxml"
    missing = tmp_path / "missing-articulation.musicxml"
    empty_reference = tmp_path / "empty-reference-articulation.musicxml"
    spurious = tmp_path / "spurious-articulation.musicxml"
    _write_articulation_measure(reference, {0: ("staccato",), 2: ("accent", "tenuto")})
    _write_articulation_measure(exact, {0: ("staccato",), 2: ("accent", "tenuto")})
    _write_articulation_measure(missing, {0: ("staccato",)})
    _write_articulation_measure(empty_reference, {})
    _write_articulation_measure(spurious, {1: ("accent",)})

    exact_report = module.compare_musicxml(reference, exact)
    assert exact_report["articulation_topology_accuracy_aligned"] == 1.0
    assert exact_report["articulation_marker_f1"] == 1.0

    missing_report = module.compare_musicxml(reference, missing)
    assert missing_report["articulation_topology_accuracy_aligned"] == 0.75
    assert missing_report["articulation_marker_precision"] == 1.0
    assert missing_report["articulation_marker_recall"] == pytest.approx(1 / 3)

    spurious_report = module.compare_musicxml(empty_reference, spurious)
    assert spurious_report["articulation_marker_precision"] == 0.0
    assert spurious_report["articulation_marker_recall"] == 1.0
    assert spurious_report["articulation_marker_f1"] == 0.0

    aggregate = module.aggregate_reports([exact_report])
    assert aggregate["articulation_marker_f1"] == 1.0
    intervals = module.bootstrap_intervals([exact_report], samples=8, seed=13)
    assert intervals["articulation_marker_f1"]["low_95"] == 1.0
    gates = module.STABLE_RELEASE_GATES["minimum"]
    assert gates["articulation_topology_accuracy_aligned"] == 0.985
    assert gates["articulation_marker_f1"] == 0.97


def _write_ornament_measure(path: Path, markers: dict[int, tuple[str, ...]]) -> None:
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
    for step in "CDEF":
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "1"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
        notes.append(note)
    set_ornament_topology(notes, tuple(tuple(markers.get(index, ())) for index in range(len(notes))))
    etree.ElementTree(root).write(
        str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE
    )


def test_release_dataset_evaluator_reports_ornament_metrics(tmp_path: Path) -> None:
    reference = tmp_path / "reference-ornament.musicxml"
    exact = tmp_path / "exact-ornament.musicxml"
    missing = tmp_path / "missing-ornament.musicxml"
    empty_reference = tmp_path / "empty-reference-ornament.musicxml"
    spurious = tmp_path / "spurious-ornament.musicxml"
    _write_ornament_measure(reference, {0: ("trill-mark",), 2: ("mordent",)})
    _write_ornament_measure(exact, {0: ("trill-mark",), 2: ("mordent",)})
    _write_ornament_measure(missing, {0: ("trill-mark",)})
    _write_ornament_measure(empty_reference, {})
    _write_ornament_measure(spurious, {1: ("turn",)})

    exact_report = module.compare_musicxml(reference, exact)
    assert exact_report["ornament_topology_accuracy_aligned"] == 1.0
    assert exact_report["ornament_marker_f1"] == 1.0

    missing_report = module.compare_musicxml(reference, missing)
    assert missing_report["ornament_topology_accuracy_aligned"] == 0.75
    assert missing_report["ornament_marker_precision"] == 1.0
    assert missing_report["ornament_marker_recall"] == 0.5

    spurious_report = module.compare_musicxml(empty_reference, spurious)
    assert spurious_report["ornament_marker_precision"] == 0.0
    assert spurious_report["ornament_marker_recall"] == 1.0
    assert spurious_report["ornament_marker_f1"] == 0.0

    aggregate = module.aggregate_reports([exact_report])
    assert aggregate["ornament_marker_f1"] == 1.0
    intervals = module.bootstrap_intervals([exact_report], samples=8, seed=17)
    assert intervals["ornament_marker_f1"]["low_95"] == 1.0
    gates = module.STABLE_RELEASE_GATES["minimum"]
    assert gates["ornament_topology_accuracy_aligned"] == 0.985
    assert gates["ornament_marker_f1"] == 0.97


def _write_expression_measure(path: Path, *, dynamic: str | None, bpm: int | None) -> None:
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
    if dynamic:
        direction = etree.SubElement(measure, "direction", placement="below")
        direction_type = etree.SubElement(direction, "direction-type")
        dynamics = etree.SubElement(direction_type, "dynamics")
        etree.SubElement(dynamics, dynamic)
    if bpm is not None:
        direction = etree.SubElement(measure, "direction", placement="above")
        direction_type = etree.SubElement(direction, "direction-type")
        metronome = etree.SubElement(direction_type, "metronome")
        etree.SubElement(metronome, "beat-unit").text = "quarter"
        etree.SubElement(metronome, "per-minute").text = str(bpm)
    for step in ("C", "D", "E", "F"):
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "1"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
    etree.ElementTree(root).write(
        str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE
    )


def test_release_dataset_evaluator_reports_simple_expression_marker_metrics(tmp_path: Path) -> None:
    reference = tmp_path / "reference-expression.musicxml"
    candidate = tmp_path / "candidate-expression.musicxml"
    _write_expression_measure(reference, dynamic="mf", bpm=120)
    _write_expression_measure(candidate, dynamic="mf", bpm=None)

    report = module.compare_musicxml(reference, candidate)
    assert report["expression_marker_precision"] == 1.0
    assert report["expression_marker_recall"] == 0.5
    assert report["expression_marker_f1"] == pytest.approx(2 / 3)

    aggregate = module.aggregate_reports([report])
    assert aggregate["expression_marker_f1"] == pytest.approx(2 / 3)
    intervals = module.bootstrap_intervals([report], samples=8, seed=29)
    assert intervals["expression_marker_recall"]["low_95"] == 0.5
    assert module.STABLE_RELEASE_GATES["minimum"]["expression_marker_f1"] == 0.97
    assert module.STABLE_RELEASE_GATES["minimum"]["expression_marker_f1_low_95"] == 0.95


def test_marker_f1_is_zero_when_nonempty_sets_have_no_matches(tmp_path: Path) -> None:
    reference = tmp_path / "reference-expression-miss.musicxml"
    candidate = tmp_path / "candidate-expression-miss.musicxml"
    _write_expression_measure(reference, dynamic="mf", bpm=None)
    _write_expression_measure(candidate, dynamic="pp", bpm=None)

    report = module.compare_musicxml(reference, candidate)
    assert report["direction_precision"] == 0.0
    assert report["direction_recall"] == 0.0
    assert report["direction_f1"] == 0.0
    assert report["expression_marker_precision"] == 0.0
    assert report["expression_marker_recall"] == 0.0
    assert report["expression_marker_f1"] == 0.0

    aggregate = module.aggregate_reports([report])
    assert aggregate["direction_f1"] == 0.0
    assert aggregate["expression_marker_f1"] == 0.0


def test_direction_metrics_separate_content_from_anchor_errors(tmp_path: Path) -> None:
    reference = tmp_path / "reference-direction-anchor.musicxml"
    misplaced = tmp_path / "misplaced-direction-anchor.musicxml"
    _write_expression_measure(reference, dynamic="mf", bpm=None)
    _write_expression_measure(misplaced, dynamic="mf", bpm=None)
    tree = etree.parse(str(misplaced))
    direction = tree.find("./part/measure/direction")
    assert direction is not None
    direction.set("placement", "above")
    tree.write(
        str(misplaced),
        encoding="UTF-8",
        xml_declaration=True,
        doctype=MUSICXML_DOCTYPE,
    )

    report = module.compare_musicxml(reference, misplaced)
    assert report["direction_content_f1"] == 1.0
    assert report["direction_f1"] == 0.0
    assert report["direction_anchor_accuracy"] == 0.0
    assert report["dynamic_direction_content_f1"] == 1.0
    assert report["dynamic_direction_f1"] == 0.0
    assert report["dynamic_direction_anchor_accuracy"] == 0.0

    aggregate = module.aggregate_reports([report])
    assert aggregate["direction_content_f1"] == 1.0
    assert aggregate["direction_f1"] == 0.0
    assert aggregate["direction_anchor_accuracy"] == 0.0
    assert module.STABLE_RELEASE_GATES["minimum"]["direction_f1"] == 0.97
    assert module.STABLE_RELEASE_GATES["minimum"]["direction_anchor_accuracy"] == 0.985


def _write_wedge_measure(path: Path, *, start_type: str) -> None:
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
    for wedge_type, offset in ((start_type, "0"), ("stop", "4")):
        direction = etree.SubElement(measure, "direction", placement="below")
        direction_type = etree.SubElement(direction, "direction-type")
        etree.SubElement(
            direction_type,
            "wedge",
            type=wedge_type,
            number="1",
        )
        etree.SubElement(direction, "offset").text = offset
        etree.SubElement(direction, "staff").text = "1"
    for step in ("C", "D", "E", "F"):
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "1"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
    etree.ElementTree(root).write(
        str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE
    )


def test_wedge_subtypes_cannot_hide_behind_aggregate_stop_accuracy(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference-crescendo.musicxml"
    candidate = tmp_path / "candidate-diminuendo.musicxml"
    _write_wedge_measure(reference, start_type="crescendo")
    _write_wedge_measure(candidate, start_type="diminuendo")

    report = module.compare_musicxml(reference, candidate)
    assert report["wedge_direction_f1"] == 0.5
    assert report["crescendo_wedge_start_f1"] == 0.0
    assert report["diminuendo_wedge_start_f1"] == 0.0
    assert report["wedge_stop_f1"] == 1.0
    assert report["counts"]["reference_crescendo_wedge_start_count"] == 1
    assert report["counts"]["reference_diminuendo_wedge_start_count"] == 0

    aggregate = module.aggregate_reports([report])
    assert aggregate["crescendo_wedge_start_f1"] == 0.0
    assert aggregate["diminuendo_wedge_start_f1"] == 0.0
    assert aggregate["wedge_stop_f1"] == 1.0
    intervals = module.bootstrap_intervals([report], samples=8, seed=31)
    assert intervals["crescendo_wedge_start_f1"]["low_95"] == 0.0
    assert intervals["diminuendo_wedge_start_f1"]["low_95"] == 0.0
    assert intervals["wedge_stop_f1"]["low_95"] == 1.0


def _write_grace_measure(path: Path, topology: list[bool]) -> None:
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
    for is_grace, step in zip(topology, ["C", "D", "E", "F"], strict=True):
        note = etree.SubElement(measure, "note")
        if is_grace:
            etree.SubElement(note, "grace")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        if not is_grace:
            etree.SubElement(note, "duration").text = "1"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "eighth" if is_grace else "quarter"
    etree.ElementTree(root).write(
        str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE
    )


def test_release_dataset_evaluator_reports_grace_topology_metrics(tmp_path: Path) -> None:
    reference = tmp_path / "reference-grace.musicxml"
    missing = tmp_path / "missing-grace.musicxml"
    spurious = tmp_path / "spurious-grace.musicxml"
    _write_grace_measure(reference, [True, False, False, False])
    _write_grace_measure(missing, [False, False, False, False])
    _write_grace_measure(spurious, [True, True, False, False])

    missing_report = module.compare_musicxml(reference, missing)
    assert missing_report["grace_topology_accuracy_aligned"] == 0.75
    assert missing_report["grace_event_precision"] == 1.0
    assert missing_report["grace_event_recall"] == 0.0
    assert missing_report["grace_event_f1"] == 0.0

    spurious_report = module.compare_musicxml(reference, spurious)
    assert spurious_report["grace_topology_accuracy_aligned"] == 0.75
    assert spurious_report["grace_event_precision"] == 0.5
    assert spurious_report["grace_event_recall"] == 1.0
    assert spurious_report["grace_event_f1"] == pytest.approx(2 / 3)

    aggregate = module.aggregate_reports([spurious_report])
    assert aggregate["grace_event_precision"] == 0.5
    intervals = module.bootstrap_intervals([spurious_report], samples=8, seed=11)
    assert intervals["grace_event_f1"]["low_95"] == pytest.approx(2 / 3)
    assert module.STABLE_RELEASE_GATES["minimum"]["grace_topology_accuracy_aligned"] == 0.99


def _write_lyric_score(path: Path, lyric: str | None) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    note = etree.SubElement(measure, "note")
    pitch = etree.SubElement(note, "pitch")
    etree.SubElement(pitch, "step").text = "C"
    etree.SubElement(pitch, "octave").text = "4"
    etree.SubElement(note, "duration").text = "1"
    etree.SubElement(note, "voice").text = "1"
    etree.SubElement(note, "type").text = "quarter"
    if lyric is not None:
        node = etree.SubElement(note, "lyric")
        etree.SubElement(node, "syllabic").text = "single"
        etree.SubElement(node, "text").text = lyric
    etree.ElementTree(root).write(str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE)


def test_release_dataset_evaluator_aggregates_lyric_metrics(tmp_path: Path) -> None:
    reference = tmp_path / "reference-lyric.musicxml"
    exact = tmp_path / "exact-lyric.musicxml"
    missing = tmp_path / "missing-lyric.musicxml"
    spurious_reference = tmp_path / "no-reference-lyric.musicxml"
    spurious = tmp_path / "spurious-lyric.musicxml"
    _write_lyric_score(reference, "la")
    _write_lyric_score(exact, "la")
    _write_lyric_score(missing, None)
    _write_lyric_score(spurious_reference, None)
    _write_lyric_score(spurious, "la")

    aggregate = module.aggregate_reports([
        module.compare_musicxml(reference, exact),
        module.compare_musicxml(reference, missing),
        module.compare_musicxml(spurious_reference, spurious),
    ])

    assert aggregate["lyric_topology_accuracy_aligned"] == pytest.approx(1 / 3)
    assert aggregate["lyric_event_precision"] == pytest.approx(1 / 2)
    assert aggregate["lyric_event_recall"] == pytest.approx(1 / 2)
    assert aggregate["lyric_event_f1"] == pytest.approx(1 / 2)


def _write_beam_score(path: Path, states: list[tuple[tuple[str, str], ...]]) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "2"
    for index, beams in enumerate(states):
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = ("C", "D", "E", "F")[index]
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "1"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "eighth"
        for number, value in beams:
            etree.SubElement(note, "beam", number=number).text = value
    etree.ElementTree(root).write(
        str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE
    )


def test_release_dataset_evaluator_aggregates_beam_metrics(tmp_path: Path) -> None:
    reference = tmp_path / "reference-beam.musicxml"
    candidate = tmp_path / "candidate-beam.musicxml"
    _write_beam_score(
        reference,
        [(('1', 'begin'),), (('1', 'continue'),), (('1', 'end'),), ()],
    )
    _write_beam_score(
        candidate,
        [(('1', 'begin'),), (), (('1', 'end'),), ()],
    )
    report = module.compare_musicxml(reference, candidate)
    aggregate = module.aggregate_reports([report])
    assert aggregate["beam_topology_accuracy_aligned"] == 0.75
    assert aggregate["beam_marker_precision"] == 1.0
    assert aggregate["beam_marker_recall"] == 2 / 3
    assert aggregate["beam_marker_f1"] == 0.8
    intervals = module.bootstrap_intervals([report], samples=8, seed=11)
    assert intervals["beam_topology_accuracy_aligned"]["low_95"] == pytest.approx(0.75)
    assert intervals["beam_marker_f1"]["low_95"] == pytest.approx(0.8)
    assert module.STABLE_RELEASE_GATES["minimum"]["beam_topology_accuracy_aligned"] == 0.985
    assert module.STABLE_RELEASE_GATES["minimum"]["beam_marker_f1_low_95"] == 0.95


def test_stable_gate_populates_every_bootstrap_lower_bound() -> None:
    minimum = module.STABLE_RELEASE_GATES["minimum"]
    interval_gate_keys = sorted(key for key in minimum if key.endswith("_low_95"))
    bootstrap: dict[str, dict[str, float]] = {}
    for gate_key in interval_gate_keys:
        source_key = module._BOOTSTRAP_GATE_ALIASES.get(
            gate_key, gate_key[: -len("_low_95")]
        )
        bootstrap[source_key] = {"low_95": 0.987, "median": 0.99, "high_95": 1.0}
    gate_metrics: dict[str, object] = {}
    module._add_bootstrap_gate_metrics(gate_metrics, bootstrap)
    assert sorted(gate_metrics) == interval_gate_keys
    assert all(value == pytest.approx(0.987) for value in gate_metrics.values())


def test_production_gate_matches_near_correction_free_product_target() -> None:
    minimum = module.PRODUCTION_RELEASE_GATES_V2["minimum"]
    maximum = module.PRODUCTION_RELEASE_GATES_V2["maximum"]

    assert minimum["case_count"] == 200
    assert minimum["source_group_count"] == 200
    assert minimum["submitted_scan_page_count"] == 2_000
    assert minimum["verified_unique_scan_page_count"] == 2_000
    assert minimum["scope_classified_page_count"] == 2_000
    assert minimum["solo_monophonic_page_count"] == 400
    assert minimum["piano_page_count"] == 400
    assert minimum["monophonic_ensemble_page_count"] == 400
    assert minimum["piano_plus_monophonic_ensemble_page_count"] == 400
    assert minimum["production_boundary_contract_evidence"] == 1
    assert minimum["physical_scan_origin_evidence"] == 1
    assert minimum["scan_page_shape_contract_evidence"] == 1
    assert minimum["scan_page_aspect_limit_evidence"] == 1
    assert minimum["ordinary_scan_page_shape_audit_evidence"] == 1
    assert minimum["complete_page_semantics_evidence"] == 1
    assert minimum["work_isolation_evidence"] == 1
    assert minimum["required_evidence_file_role_count"] == len(
        module.PRODUCTION_EVIDENCE_FILE_ROLES
    )
    assert minimum["reference_event_count"] == 50_000
    assert minimum["pitch_accuracy_aligned"] == 0.999
    assert minimum["rhythm_accuracy_aligned"] == 0.999
    assert minimum["tie_endpoint_f1"] == 0.997
    assert minimum["slur_endpoint_f1"] == 0.997
    assert minimum["accidental_marker_f1"] == 0.997
    assert minimum["direction_content_f1"] == 0.997
    assert minimum["direction_anchor_accuracy"] == 0.998
    assert minimum["crescendo_wedge_start_f1"] == 0.997
    assert minimum["diminuendo_wedge_start_f1"] == 0.997
    assert minimum["wedge_stop_f1"] == 0.997
    assert minimum["reference_crescendo_wedge_start_count"] == 200
    assert minimum["reference_diminuendo_wedge_start_count"] == 200
    assert minimum["piano__reference_crescendo_wedge_start_count"] == 50
    assert minimum["piano__reference_diminuendo_wedge_start_count"] == 50
    assert not any(key.startswith("lyric_") for key in minimum)
    assert (
        module.STABLE_RELEASE_GATES["minimum"][
            "lyric_topology_accuracy_aligned"
        ]
        == 0.985
    )
    assert minimum["exact_measure_rate"] == 0.995
    assert minimum["exact_measure_rate_low_95"] == 0.99
    assert minimum["piano__rhythm_accuracy_aligned"] == 0.997
    assert minimum["piano__rhythm_accuracy_aligned_low_95"] == 0.99
    assert (
        minimum[
            "piano_plus_monophonic_ensemble__direction_anchor_accuracy"
        ]
        == 0.997
    )
    assert minimum["solo_monophonic__reference_event_count"] == 5_000
    assert minimum["reference_accidental_marker_count"] == 1_000
    assert (
        minimum["piano__reference_accidental_marker_count"]
        == 100
    )
    assert maximum["event_error_rate"] == 0.005
    assert maximum["piano__event_error_rate"] == 0.01
    assert maximum["out_of_contract_page_count"] == 0
    assert maximum["duplicate_scan_page_count"] == 0
    assert maximum["unverified_scan_page_identity_count"] == 0


def test_production_evidence_requires_hashed_audits_and_physical_pages(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "benchmark.json"
    manifest.write_text("{}\n", encoding="utf-8")
    evidence_files = []
    for role in module.PRODUCTION_EVIDENCE_FILE_ROLES:
        path = tmp_path / f"{role}.json"
        path.write_text(
            json.dumps({"format": 1, "role": role}) + "\n",
            encoding="utf-8",
        )
        evidence_files.append(
            {
                "role": role,
                "path": path.name,
                "sha256": module.sha256_file(path),
            }
        )
    payload = {
        "production_evidence": {
            "boundary_contract_version": (
                module.PRODUCTION_BOUNDARY_CONTRACT_VERSION
            ),
            "source_image_origin": "physical_scan",
            "page_identity_audited": True,
            "evaluation_use_authorized": True,
            "complete_page_level_semantics": True,
            "instrumental_lyrics_excluded_or_isolated": True,
            "independent_double_annotation_adjudicated": True,
            "work_disjoint_from_training_and_tuning": True,
            "frozen_before_candidate_evaluation": True,
            "submitted_orientation_preserved": True,
            "scan_page_shape_contract": (
                module.PRODUCTION_SCAN_PAGE_SHAPE_CONTRACT
            ),
            "maximum_scan_page_aspect_ratio": (
                module.PRODUCTION_MAXIMUM_SCAN_PAGE_ASPECT_RATIO
            ),
            "ordinary_scan_page_shape_audited": True,
            "evidence_files": evidence_files,
        }
    }

    report = module.validate_production_evidence(payload, manifest)

    assert report["passed"] is True
    assert report["errors"] == []
    assert len(report["verified_files"]) == len(
        module.PRODUCTION_EVIDENCE_FILE_ROLES
    )
    assert all(value >= 1 for value in report["metrics"].values())

    payload["production_evidence"].pop(
        "ordinary_scan_page_shape_audited"
    )
    missing_shape_audit = module.validate_production_evidence(
        payload,
        manifest,
    )
    assert missing_shape_audit["passed"] is False
    assert (
        missing_shape_audit["metrics"][
            "ordinary_scan_page_shape_audit_evidence"
        ]
        == 0
    )
    payload["production_evidence"][
        "ordinary_scan_page_shape_audited"
    ] = True

    (tmp_path / f"{module.PRODUCTION_EVIDENCE_FILE_ROLES[0]}.json").write_text(
        '{"tampered": true}\n',
        encoding="utf-8",
    )
    tampered = module.validate_production_evidence(payload, manifest)
    assert tampered["passed"] is False
    assert any("hash mismatch" in error for error in tampered["errors"])


def test_production_evidence_fails_closed_when_missing(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "benchmark.json"
    manifest.write_text("{}\n", encoding="utf-8")

    report = module.validate_production_evidence({}, manifest)

    assert report["passed"] is False
    assert report["metrics"]["physical_scan_origin_evidence"] == 0
    assert report["metrics"]["required_evidence_file_role_count"] == 0


def test_production_scope_coverage_is_closed_and_work_grouped() -> None:
    reports = [
        {
            "source_group": "work-a",
            "submitted_scan_page_count": 4,
            "submitted_scan_page_ids": [
                "scan-a/page-1",
                "scan-a/page-2",
                "scan-a/page-3",
                "scan-a/page-4",
            ],
            "strata": {"score_configuration": "piano"},
        },
        {
            "source_group": "work-a",
            "submitted_scan_page_ids": ["scan-b/page-1"],
            "strata": {"score_configuration": "piano"},
        },
        {
            "source_group": "work-b",
            "submitted_scan_page_ids": ["scan-c/page-1"],
            "strata": {"score_configuration": "monophonic_ensemble"},
        },
        {
            "source_group": "work-c",
            "submitted_scan_page_ids": ["scan-c/page-1"],
            "strata": {"score_configuration": "unknown-layout"},
        },
        {"source_group": "work-d", "strata": {}},
    ]

    coverage = module.production_scope_coverage(reports)
    assert coverage["document_count"] == 5
    assert coverage["page_count"] == 8
    assert coverage["source_group_count"] == 4
    assert coverage["verified_unique_scan_page_count"] == 6
    assert coverage["duplicate_scan_page_count"] == 1
    assert coverage["unverified_scan_page_identity_count"] == 1
    assert coverage["scope_classified_page_count"] == 6
    assert coverage["out_of_contract_page_count"] == 2
    assert coverage["pages_by_score_configuration"] == {
        "solo_monophonic": 0,
        "piano": 5,
        "monophonic_ensemble": 1,
        "piano_plus_monophonic_ensemble": 0,
    }
    metrics = {"case_count": 5}
    module._add_production_scope_gate_metrics(metrics, coverage)
    assert metrics["submitted_scan_page_count"] == 8
    assert metrics["verified_unique_scan_page_count"] == 6
    assert metrics["duplicate_scan_page_count"] == 1
    assert metrics["unverified_scan_page_identity_count"] == 1
    assert metrics["piano_page_count"] == 5
    assert metrics["monophonic_ensemble_page_count"] == 1
    assert metrics["out_of_contract_page_count"] == 2


def test_production_scope_coverage_rejects_invalid_page_count() -> None:
    with pytest.raises(ValueError, match="positive integers"):
        module.production_scope_coverage(
            [
                {
                    "source_group": "work-a",
                    "submitted_scan_page_count": 0,
                    "strata": {"score_configuration": "piano"},
                }
            ]
        )


def test_reference_feature_coverage_is_flattened_and_missing_fails_closed() -> None:
    gate_metrics: dict[str, object] = {}
    aggregate = {
        "counts": {
            "reference_accidental_marker_count": 1234,
            "reference_tie_endpoint_count": 2345,
        }
    }
    coverage = module._add_reference_feature_gate_metrics(
        gate_metrics,
        aggregate,
    )

    assert coverage["reference_accidental_marker_count"] == 1234
    assert gate_metrics["reference_tie_endpoint_count"] == 2345
    assert coverage["reference_ornament_marker_count"] == 0
    assert gate_metrics["reference_ornament_marker_count"] == 0


def test_production_configuration_quality_metrics_are_independent(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.musicxml"
    candidate = tmp_path / "candidate.musicxml"
    _write_score(reference, ["C", "D"])
    _write_score(candidate, ["C", "D"])
    reports = []
    for configuration in module.PRODUCTION_SCORE_CONFIGURATIONS:
        report = module.compare_musicxml(reference, candidate)
        report["strata"] = {"score_configuration": configuration}
        reports.append(report)

    gate_metrics: dict[str, object] = {}
    intervals = module._add_production_configuration_quality_gate_metrics(
        gate_metrics,
        reports,
        samples=8,
        seed=31,
    )

    for configuration in module.PRODUCTION_SCORE_CONFIGURATIONS:
        assert gate_metrics[
            f"{configuration}__pitch_accuracy_aligned"
        ] == 1.0
        assert gate_metrics[
            f"{configuration}__pitch_accuracy_aligned_low_95"
        ] == 1.0
        assert gate_metrics[f"{configuration}__event_error_rate"] == 0.0
        assert intervals[configuration]["pitch_accuracy_aligned"][
            "low_95"
        ] == 1.0


def test_production_configuration_quality_metrics_fail_closed_when_missing() -> None:
    gate_metrics: dict[str, object] = {}
    intervals = module._add_production_configuration_quality_gate_metrics(
        gate_metrics,
        [],
        samples=8,
        seed=31,
    )

    for configuration in module.PRODUCTION_SCORE_CONFIGURATIONS:
        assert gate_metrics[
            f"{configuration}__pitch_accuracy_aligned"
        ] == 0.0
        assert gate_metrics[
            f"{configuration}__pitch_accuracy_aligned_low_95"
        ] == 0.0
        assert gate_metrics[f"{configuration}__event_error_rate"] == 1.0
        assert intervals[configuration] == {}
    evaluated = module._evaluate_gates(
        gate_metrics,
        module.PRODUCTION_RELEASE_GATES_V2,
    )
    assert evaluated["passed"] is False


def test_release_manifest_rejects_non_integer_scan_page_count(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.musicxml"
    candidate = tmp_path / "candidate.musicxml"
    _write_score(reference, ["C"])
    _write_score(candidate, ["C"])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": 1,
                "cases": [
                    {
                        "id": "invalid-pages",
                        "reference": reference.name,
                        "candidate": candidate.name,
                        "submitted_scan_page_count": 1.5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="submitted_scan_page_count must be"
    ):
        module.evaluate_manifest(manifest)


def test_release_manifest_tracks_duplicate_and_unverified_scan_pages(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.musicxml"
    candidate = tmp_path / "candidate.musicxml"
    _write_score(reference, ["C"])
    _write_score(candidate, ["C"])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": 1,
                "cases": [
                    {
                        "id": "verified-a",
                        "reference": reference.name,
                        "candidate": candidate.name,
                        "submitted_scan_page_count": 1,
                        "submitted_scan_page_ids": ["source/page-1"],
                    },
                    {
                        "id": "verified-duplicate",
                        "reference": reference.name,
                        "candidate": candidate.name,
                        "submitted_scan_page_count": 1,
                        "submitted_scan_page_ids": ["source/page-1"],
                    },
                    {
                        "id": "legacy-unverified",
                        "reference": reference.name,
                        "candidate": candidate.name,
                        "submitted_scan_page_count": 2,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = module.evaluate_manifest(manifest)
    coverage = report["production_scope_coverage"]
    assert coverage["page_count"] == 4
    assert coverage["verified_unique_scan_page_count"] == 1
    assert coverage["duplicate_scan_page_count"] == 1
    assert coverage["unverified_scan_page_identity_count"] == 2


@pytest.mark.parametrize(
    "page_ids, message",
    [
        ("not-a-list", "must be a list"),
        (["only-one"], "length must equal"),
        (["valid", ""], "non-empty strings"),
    ],
)
def test_release_manifest_rejects_invalid_scan_page_ids(
    tmp_path: Path,
    page_ids: object,
    message: str,
) -> None:
    reference = tmp_path / "reference.musicxml"
    candidate = tmp_path / "candidate.musicxml"
    _write_score(reference, ["C"])
    _write_score(candidate, ["C"])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": 1,
                "cases": [
                    {
                        "id": "invalid-page-ids",
                        "reference": reference.name,
                        "candidate": candidate.name,
                        "submitted_scan_page_count": 2,
                        "submitted_scan_page_ids": page_ids,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        module.evaluate_manifest(manifest)
