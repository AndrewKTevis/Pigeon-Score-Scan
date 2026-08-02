from pathlib import Path

from lxml import etree

from scorescan.musicxml import MUSICXML_DOCTYPE

# The evaluator is a release/development tool rather than runtime package code.
import importlib.util

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "evaluate_musicxml.py"
spec = importlib.util.spec_from_file_location("scorescan_evaluator", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def write_score(path: Path, steps: list[str]) -> None:
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


def test_pitch_metric_uses_alter_not_optional_accidental_element(tmp_path: Path) -> None:
    reference = tmp_path / "reference-accidental.musicxml"
    equivalent = tmp_path / "equivalent-accidental.musicxml"
    wrong = tmp_path / "wrong-accidental.musicxml"
    for path in (reference, equivalent, wrong):
        write_score(path, ["D"])

    for path in (reference, equivalent):
        tree = etree.parse(str(path))
        pitch = tree.find("./part/measure/note/pitch")
        assert pitch is not None
        alter = etree.Element("alter")
        alter.text = "1"
        pitch.insert(1, alter)
        if path == reference:
            note = tree.find("./part/measure/note")
            assert note is not None
            etree.SubElement(note, "accidental").text = "sharp"
        tree.write(
            str(path),
            encoding="UTF-8",
            xml_declaration=True,
            doctype=MUSICXML_DOCTYPE,
        )

    equivalent_report = module.compare(reference, equivalent)
    missing_report = module.compare(reference, wrong)
    false_sharp_report = module.compare(wrong, reference)
    assert equivalent_report["pitch_accuracy_aligned"] == 1.0
    # Omitting the optional XML surface does not lose an accidental when the
    # altered pitch still requires the engraver to display it.
    assert equivalent_report["accidental_marker_f1"] == 1.0
    assert missing_report["pitch_accuracy_aligned"] == 0.0
    assert missing_report["accidental_marker_f1"] == 0.0
    # This is the reported Sol -> sharp-Sol class of defect: an added sign is a
    # false positive even when the reference contains no accidental markers.
    assert false_sharp_report["accidental_marker_precision"] == 0.0
    assert false_sharp_report["accidental_marker_f1"] == 0.0


def test_accidental_metric_does_not_reprint_middle_of_tie_chain(
    tmp_path: Path,
) -> None:
    score = tmp_path / "tie-chain.musicxml"
    write_score(score, ["D", "D", "D"])
    tree = etree.parse(str(score))
    notes = tree.findall("./part/measure/note")
    for index, note in enumerate(notes):
        pitch = note.find("pitch")
        assert pitch is not None
        alter = etree.Element("alter")
        alter.text = "1"
        pitch.insert(1, alter)
        notations = etree.SubElement(note, "notations")
        if index == 0:
            etree.SubElement(note, "accidental").text = "sharp"
            etree.SubElement(note, "tie", type="start")
            etree.SubElement(notations, "tied", type="start")
        elif index == 1:
            etree.SubElement(note, "tie", type="stop")
            etree.SubElement(note, "tie", type="start")
            etree.SubElement(notations, "tied", type="stop")
            etree.SubElement(notations, "tied", type="start")
        else:
            etree.SubElement(note, "tie", type="stop")
            etree.SubElement(notations, "tied", type="stop")
    tree.write(
        str(score),
        encoding="UTF-8",
        xml_declaration=True,
        doctype=MUSICXML_DOCTYPE,
    )

    report = module.compare(score, score)
    assert report["counts"]["reference_accidental_marker_count"] == 1
    assert report["accidental_marker_f1"] == 1.0


def _write_quantized_triplet(
    path: Path,
    *,
    divisions: int,
    durations: tuple[int, int, int],
    voice: str,
) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = str(divisions)
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "2"
    etree.SubElement(time, "beat-type").text = "2"
    for step, duration in zip(("C", "D", "E"), durations, strict=True):
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = str(duration)
        etree.SubElement(note, "voice").text = voice
        etree.SubElement(note, "type").text = "eighth"
        time_modification = etree.SubElement(note, "time-modification")
        etree.SubElement(time_modification, "actual-notes").text = "3"
        etree.SubElement(time_modification, "normal-notes").text = "2"
    etree.ElementTree(root).write(
        str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE
    )


def test_rhythm_metric_accepts_legal_triplet_rounding_and_renumbered_voice(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference-triplet.musicxml"
    candidate = tmp_path / "candidate-triplet.musicxml"
    _write_quantized_triplet(
        reference,
        divisions=256,
        durations=(85, 85, 86),
        voice="3",
    )
    _write_quantized_triplet(
        candidate,
        divisions=12,
        durations=(4, 4, 4),
        voice="5",
    )

    report = module.compare(reference, candidate)

    assert report["duration_accuracy_aligned"] == 1.0
    assert report["onset_accuracy_aligned"] == 1.0
    assert report["rhythm_accuracy_aligned"] == 1.0
    # The dedicated voice metric remains strict and exposes the exporter-local
    # renumbering instead of double-counting it as a rhythm defect.
    assert report["voice_assignment_accuracy_aligned"] == 0.0


def _write_key_mode_score(path: Path, mode: str | None) -> None:
    write_score(path, ["E"])
    tree = etree.parse(str(path))
    attributes = tree.find("./part/measure/attributes")
    assert attributes is not None
    key = etree.SubElement(attributes, "key")
    etree.SubElement(key, "fifths").text = "4"
    if mode is not None:
        etree.SubElement(key, "mode").text = mode
    tree.write(
        str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE
    )


def test_key_signature_metric_ignores_optional_mode_text(tmp_path: Path) -> None:
    reference = tmp_path / "reference-key.musicxml"
    candidate = tmp_path / "candidate-key.musicxml"
    wrong = tmp_path / "wrong-key.musicxml"
    _write_key_mode_score(reference, "major")
    _write_key_mode_score(candidate, None)
    _write_key_mode_score(wrong, None)
    wrong_tree = etree.parse(str(wrong))
    fifths = wrong_tree.find("./part/measure/attributes/key/fifths")
    assert fifths is not None
    fifths.text = "3"
    wrong_tree.write(
        str(wrong), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE
    )

    assert module.compare(reference, candidate)["key_signature_accuracy"] == 1.0
    assert module.compare(reference, wrong)["key_signature_accuracy"] == 0.0


def test_evaluator_exact_and_changed_event(tmp_path: Path) -> None:
    reference = tmp_path / "reference.musicxml"
    exact = tmp_path / "exact.musicxml"
    changed = tmp_path / "changed.musicxml"
    write_score(reference, ["C", "D"])
    write_score(exact, ["C", "D"])
    write_score(changed, ["C", "E"])
    exact_report = module.compare(reference, exact)
    changed_report = module.compare(reference, changed)
    assert exact_report["exact_measure_rate"] == 1.0
    assert exact_report["event_error_rate"] == 0.0
    assert changed_report["exact_measure_rate"] == 0.5
    assert changed_report["event_error_rate"] == 0.5


def write_single_measure(path: Path, steps: list[str]) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = str(len(steps))
    etree.SubElement(time, "beat-type").text = "4"
    for step in steps:
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "1"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
    etree.ElementTree(root).write(str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE)


def test_evaluator_global_measure_alignment_contains_one_insertion(tmp_path: Path) -> None:
    reference = tmp_path / "reference.musicxml"
    candidate = tmp_path / "candidate.musicxml"
    write_score(reference, ["C", "D", "E"])
    write_score(candidate, ["C", "G", "D", "E"])

    report = module.compare(reference, candidate)

    assert report["measure_alignment"]["reference_to_candidate"] == [1, 3, 4]
    assert report["measure_alignment"]["unmatched_candidate_indices"] == [2]
    assert report["counts"]["inserted_measures"] == 1
    assert report["pitch_accuracy_aligned"] == 1.0
    assert report["event_error_rate"] == 1 / 3


def test_evaluator_global_event_alignment_contains_one_insertion(tmp_path: Path) -> None:
    reference = tmp_path / "reference.musicxml"
    candidate = tmp_path / "candidate.musicxml"
    write_single_measure(reference, ["C", "D", "E"])
    write_single_measure(candidate, ["C", "G", "D", "E"])

    report = module.compare(reference, candidate)

    assert report["counts"]["inserted_events"] == 1
    assert report["counts"]["substituted_events"] == 0
    assert report["pitch_accuracy_aligned"] == 1.0
    assert report["event_error_rate"] == 1 / 3


def write_simple_triplet_score(path: Path, enabled: bool) -> None:
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


def test_evaluator_reports_tuplet_topology_metrics(tmp_path: Path) -> None:
    reference = tmp_path / "reference.musicxml"
    exact = tmp_path / "exact.musicxml"
    missing = tmp_path / "missing.musicxml"
    spurious_reference = tmp_path / "spurious_reference.musicxml"
    spurious = tmp_path / "spurious.musicxml"
    write_simple_triplet_score(reference, True)
    write_simple_triplet_score(exact, True)
    write_simple_triplet_score(missing, False)
    write_simple_triplet_score(spurious_reference, False)
    write_simple_triplet_score(spurious, True)

    exact_report = module.compare(reference, exact)
    missing_report = module.compare(reference, missing)
    spurious_report = module.compare(spurious_reference, spurious)

    assert exact_report["tuplet_topology_accuracy_aligned"] == 1.0
    assert exact_report["tuplet_event_precision"] == 1.0
    assert exact_report["tuplet_event_recall"] == 1.0
    assert exact_report["tuplet_event_f1"] == 1.0
    assert missing_report["tuplet_topology_accuracy_aligned"] == 0.5
    assert missing_report["tuplet_event_precision"] == 1.0
    assert missing_report["tuplet_event_recall"] == 0.0
    assert missing_report["tuplet_event_f1"] == 0.0
    assert spurious_report["tuplet_event_precision"] == 0.0
    assert spurious_report["tuplet_event_recall"] == 1.0
    assert spurious_report["tuplet_event_f1"] == 0.0


def write_slur_measure(path: Path, arcs: list[tuple[int, int, int]]) -> None:
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


def test_evaluator_reports_number_independent_slur_metrics(tmp_path: Path) -> None:
    reference = tmp_path / "reference-slur.musicxml"
    exact_different_number = tmp_path / "exact-slur.musicxml"
    missing = tmp_path / "missing-slur.musicxml"
    empty_reference = tmp_path / "empty-reference-slur.musicxml"
    spurious = tmp_path / "spurious-slur.musicxml"
    write_slur_measure(reference, [(0, 2, 1)])
    write_slur_measure(exact_different_number, [(0, 2, 7)])
    write_slur_measure(missing, [])
    write_slur_measure(empty_reference, [])
    write_slur_measure(spurious, [(0, 2, 2)])

    exact_report = module.compare(reference, exact_different_number)
    missing_report = module.compare(reference, missing)
    spurious_report = module.compare(empty_reference, spurious)

    assert exact_report["slur_topology_accuracy"] == 1.0
    assert exact_report["slur_endpoint_precision"] == 1.0
    assert exact_report["slur_endpoint_recall"] == 1.0
    assert exact_report["slur_endpoint_f1"] == 1.0
    assert missing_report["slur_topology_accuracy"] == 0.0
    assert missing_report["slur_endpoint_precision"] == 1.0
    assert missing_report["slur_endpoint_recall"] == 0.0
    assert missing_report["slur_endpoint_f1"] == 0.0
    assert spurious_report["slur_endpoint_precision"] == 0.0
    assert spurious_report["slur_endpoint_recall"] == 1.0
    assert spurious_report["slur_endpoint_f1"] == 0.0


def write_articulation_measure(path: Path, marks: list[list[str]]) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = str(len(marks))
    etree.SubElement(time, "beat-type").text = "4"
    for index, event_marks in enumerate(marks):
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = "CDEFGAB"[index % 7]
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "1"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
        if event_marks:
            notations = etree.SubElement(note, "notations")
            articulations = etree.SubElement(notations, "articulations")
            for mark in event_marks:
                etree.SubElement(articulations, mark)
    etree.ElementTree(root).write(
        str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE
    )


def test_evaluator_reports_articulation_topology_and_marker_metrics(tmp_path: Path) -> None:
    reference = tmp_path / "reference-articulation.musicxml"
    exact = tmp_path / "exact-articulation.musicxml"
    missing = tmp_path / "missing-articulation.musicxml"
    empty_reference = tmp_path / "empty-reference-articulation.musicxml"
    spurious = tmp_path / "spurious-articulation.musicxml"
    write_articulation_measure(reference, [["staccato"], [], ["accent", "tenuto"], []])
    write_articulation_measure(exact, [["staccato"], [], ["accent", "tenuto"], []])
    write_articulation_measure(missing, [[], [], ["accent"], []])
    write_articulation_measure(empty_reference, [[], [], [], []])
    write_articulation_measure(spurious, [["staccato"], [], [], []])

    exact_report = module.compare(reference, exact)
    missing_report = module.compare(reference, missing)
    spurious_report = module.compare(empty_reference, spurious)

    assert exact_report["articulation_topology_accuracy_aligned"] == 1.0
    assert exact_report["articulation_marker_precision"] == 1.0
    assert exact_report["articulation_marker_recall"] == 1.0
    assert exact_report["articulation_marker_f1"] == 1.0
    assert missing_report["articulation_topology_accuracy_aligned"] == 0.5
    assert missing_report["articulation_marker_precision"] == 1.0
    assert missing_report["articulation_marker_recall"] == 1 / 3
    assert missing_report["articulation_marker_f1"] == 0.5
    assert spurious_report["articulation_marker_precision"] == 0.0
    assert spurious_report["articulation_marker_recall"] == 1.0
    assert spurious_report["articulation_marker_f1"] == 0.0


def write_ornament_measure(path: Path, marks: list[list[str]]) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = str(len(marks))
    etree.SubElement(time, "beat-type").text = "4"
    for index, event_marks in enumerate(marks):
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = "CDEFGAB"[index % 7]
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "1"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
        if event_marks:
            notations = etree.SubElement(note, "notations")
            ornaments = etree.SubElement(notations, "ornaments")
            for mark in event_marks:
                etree.SubElement(ornaments, mark)
    etree.ElementTree(root).write(
        str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE
    )


def test_evaluator_reports_ornament_topology_and_marker_metrics(tmp_path: Path) -> None:
    reference = tmp_path / "reference-ornament.musicxml"
    exact = tmp_path / "exact-ornament.musicxml"
    missing = tmp_path / "missing-ornament.musicxml"
    empty_reference = tmp_path / "empty-reference-ornament.musicxml"
    spurious = tmp_path / "spurious-ornament.musicxml"
    write_ornament_measure(reference, [["trill-mark"], [], ["turn"], []])
    write_ornament_measure(exact, [["trill-mark"], [], ["turn"], []])
    write_ornament_measure(missing, [[], [], ["turn"], []])
    write_ornament_measure(empty_reference, [[], [], [], []])
    write_ornament_measure(spurious, [["mordent"], [], [], []])

    exact_report = module.compare(reference, exact)
    missing_report = module.compare(reference, missing)
    spurious_report = module.compare(empty_reference, spurious)

    assert exact_report["ornament_topology_accuracy_aligned"] == 1.0
    assert exact_report["ornament_marker_precision"] == 1.0
    assert exact_report["ornament_marker_recall"] == 1.0
    assert exact_report["ornament_marker_f1"] == 1.0
    assert missing_report["ornament_topology_accuracy_aligned"] == 0.75
    assert missing_report["ornament_marker_precision"] == 1.0
    assert missing_report["ornament_marker_recall"] == 0.5
    assert missing_report["ornament_marker_f1"] == 2 / 3
    assert spurious_report["ornament_marker_precision"] == 0.0
    assert spurious_report["ornament_marker_recall"] == 1.0
    assert spurious_report["ornament_marker_f1"] == 0.0


def write_lyric_measure(path: Path, lyrics: list[tuple[str, str, str] | None]) -> None:
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
    etree.SubElement(time, "beats").text = str(len(lyrics))
    etree.SubElement(time, "beat-type").text = "4"
    notes: list[etree._Element] = []
    for index in range(len(lyrics)):
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = "CDEFGAB"[index % 7]
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "1"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "quarter"
        notes.append(note)
    set_lyric_topology(
        notes,
        [LyricState(*row) if row is not None else None for row in lyrics],
    )
    etree.ElementTree(root).write(
        str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE
    )


def test_evaluator_reports_simple_lyric_topology_and_event_metrics(tmp_path: Path) -> None:
    reference = tmp_path / "reference-lyric.musicxml"
    exact = tmp_path / "exact-lyric.musicxml"
    missing = tmp_path / "missing-lyric.musicxml"
    empty_reference = tmp_path / "empty-reference-lyric.musicxml"
    spurious = tmp_path / "spurious-lyric.musicxml"
    rows = [("Hel", "begin", ""), ("lo", "end", ""), ("world", "single", "start"), None]
    write_lyric_measure(reference, rows)
    write_lyric_measure(exact, rows)
    write_lyric_measure(missing, [("Hel", "begin", ""), None, ("world", "single", "start"), None])
    write_lyric_measure(empty_reference, [None, None, None, None])
    write_lyric_measure(spurious, [("la", "single", ""), None, None, None])

    exact_report = module.compare(reference, exact)
    missing_report = module.compare(reference, missing)
    spurious_report = module.compare(empty_reference, spurious)

    assert exact_report["lyric_topology_accuracy_aligned"] == 1.0
    assert exact_report["lyric_event_precision"] == 1.0
    assert exact_report["lyric_event_recall"] == 1.0
    assert exact_report["lyric_event_f1"] == 1.0
    assert missing_report["lyric_topology_accuracy_aligned"] == 0.75
    assert missing_report["lyric_event_precision"] == 1.0
    assert missing_report["lyric_event_recall"] == 2 / 3
    assert missing_report["lyric_event_f1"] == 0.8
    assert spurious_report["lyric_event_precision"] == 0.0
    assert spurious_report["lyric_event_recall"] == 1.0
    assert spurious_report["lyric_event_f1"] == 0.0


def _write_beam_measure(path: Path, states: list[tuple[tuple[str, str], ...]]) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "2"
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "2"
    etree.SubElement(time, "beat-type").text = "4"
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


def test_evaluator_reports_beam_topology_and_marker_metrics(tmp_path: Path) -> None:
    reference = tmp_path / "reference-beam.musicxml"
    candidate = tmp_path / "candidate-beam.musicxml"
    _write_beam_measure(
        reference,
        [(('1', 'begin'),), (('1', 'continue'),), (('1', 'end'),), ()],
    )
    _write_beam_measure(
        candidate,
        [(('1', 'begin'),), (), (('1', 'end'),), ()],
    )

    report = module.compare(reference, candidate)
    assert report["beam_topology_accuracy_aligned"] == 0.75
    assert report["beam_marker_precision"] == 1.0
    assert report["beam_marker_recall"] == 2 / 3
    assert report["beam_marker_f1"] == 0.8
    assert report["counts"]["reference_beam_marker_count"] == 3
    assert report["counts"]["candidate_beam_marker_count"] == 2
    assert report["counts"]["beam_marker_matches"] == 2


def test_evaluator_preservation_exact_measure_detects_beam_only_difference(tmp_path: Path) -> None:
    reference = tmp_path / "reference-beam-preservation.musicxml"
    candidate = tmp_path / "candidate-beam-preservation.musicxml"
    write_score(reference, ["C"])
    write_score(candidate, ["C"])
    candidate_tree = etree.parse(str(candidate))
    note = candidate_tree.find("./part/measure/note")
    assert note is not None
    etree.SubElement(note, "beam", number="1").text = "begin"
    candidate_tree.write(
        str(candidate), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE
    )

    report = module.compare(reference, candidate)

    # Score IR deliberately does not model beam topology in the measure fingerprint.
    assert report["exact_measure_rate"] == 1.0
    assert report["preservation_exact_measure_rate"] == 0.0
    assert report["counts"]["preservation_exact_measures"] == 0
    assert report["per_measure"][0]["preservation_exact"] is False


def _write_full_score(
    path: Path,
    *,
    include_violin: bool = True,
    piano_second_voice: str = "2",
    piano_upper_staff: str = "1",
) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    definitions = [("P1", "Piano", 2)]
    if include_violin:
        definitions.append(("P2", "Violin", 1))
    for part_id, name, _staves in definitions:
        score_part = etree.SubElement(part_list, "score-part", id=part_id)
        etree.SubElement(score_part, "part-name").text = name
    for part_id, _name, staves in definitions:
        part = etree.SubElement(root, "part", id=part_id)
        measure = etree.SubElement(part, "measure", number="1")
        attributes = etree.SubElement(measure, "attributes")
        etree.SubElement(attributes, "divisions").text = "1"
        etree.SubElement(attributes, "staves").text = str(staves)
        if part_id == "P1":
            notes = [
                ("C", "1", piano_upper_staff),
                ("E", piano_second_voice, "1"),
                ("G", "1", "2"),
            ]
        else:
            notes = [("D", "1", "1")]
        for step, voice, staff in notes:
            note = etree.SubElement(measure, "note")
            pitch = etree.SubElement(note, "pitch")
            etree.SubElement(pitch, "step").text = step
            etree.SubElement(pitch, "octave").text = "4"
            etree.SubElement(note, "duration").text = "1"
            etree.SubElement(note, "voice").text = voice
            etree.SubElement(note, "staff").text = staff
            etree.SubElement(note, "type").text = "quarter"
    etree.ElementTree(root).write(
        str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE
    )


def test_evaluator_micro_aggregates_complete_piano_and_ensemble_score(tmp_path: Path) -> None:
    reference = tmp_path / "reference-full.musicxml"
    candidate = tmp_path / "candidate-full.musicxml"
    _write_full_score(reference)
    _write_full_score(candidate)

    report = module.compare(reference, candidate)

    assert report["schema"] == "scorescan-full-score-evaluation@1"
    assert report["part_count_exact"] is True
    assert report["part_mapping_accuracy"] == 1.0
    assert report["staff_count_exact"] is True
    assert report["staff_topology_f1"] == 1.0
    assert report["voice_topology_accuracy"] == 1.0
    assert report["staff_assignment_accuracy_aligned"] == 1.0
    assert report["pitch_accuracy_aligned"] == 1.0
    assert report["reference_event_count"] == 4
    assert len(report["parts"]) == 2


def test_evaluator_missing_part_is_an_omission_without_shifting_piano(tmp_path: Path) -> None:
    reference = tmp_path / "reference-full.musicxml"
    candidate = tmp_path / "candidate-piano-only.musicxml"
    _write_full_score(reference)
    _write_full_score(candidate, include_violin=False)

    report = module.compare(reference, candidate)

    assert report["part_count_exact"] is False
    assert report["part_mapping_accuracy"] == 0.5
    assert report["staff_topology_recall"] == 2 / 3
    assert report["event_presence_recall"] == 0.75
    assert report["pitch_accuracy_aligned"] == 0.75
    assert report["parts"][0]["pitch_accuracy_aligned"] == 1.0
    assert report["parts"][1]["counts"]["deleted_events"] == 1


def test_evaluator_reports_staff_voice_assignment_and_topology_errors(tmp_path: Path) -> None:
    reference = tmp_path / "reference-full.musicxml"
    candidate = tmp_path / "candidate-topology.musicxml"
    _write_full_score(reference)
    _write_full_score(candidate, piano_second_voice="3", piano_upper_staff="2")

    report = module.compare(reference, candidate)

    assert report["voice_assignment_accuracy_aligned"] == 0.75
    assert report["staff_assignment_accuracy_aligned"] == 0.75
    assert report["voice_topology_accuracy"] == 0.5
