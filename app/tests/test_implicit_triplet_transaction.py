from __future__ import annotations

from pathlib import Path

import pytest
from lxml import etree

from scorescan.implicit_triplet_transaction import (
    apply_confirmed_continuous_triplet_grid,
    apply_evidence_confirmed_continuous_triplet_grid,
    detect_continuous_triplet_grid_evidence,
)
from scorescan.score_ir import score_from_tree


def _note(measure: etree._Element, duration: int, note_type: str, step: str = "C") -> etree._Element:
    note = etree.SubElement(measure, "note")
    pitch = etree.SubElement(note, "pitch")
    etree.SubElement(pitch, "step").text = step
    etree.SubElement(pitch, "octave").text = "4"
    etree.SubElement(note, "duration").text = str(duration)
    etree.SubElement(note, "voice").text = "1"
    etree.SubElement(note, "type").text = note_type
    etree.SubElement(note, "staff").text = "1"
    return note


def _score(
    path: Path,
    *,
    measures: int = 8,
    divisions: int = 4,
    existing_triplet_prefix: bool = False,
    clone_first_note: bool = False,
    coda: bool = False,
    parts: int = 1,
) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    for part_index in range(parts):
        part_id = f"P{part_index + 1}"
        score_part = etree.SubElement(part_list, "score-part", id=part_id)
        etree.SubElement(score_part, "part-name").text = "Piano"
        part = etree.SubElement(root, "part", id=part_id)
        for measure_index in range(measures):
            measure = etree.SubElement(part, "measure", number=str(measure_index + 1))
            if measure_index == 0:
                attributes = etree.SubElement(measure, "attributes")
                etree.SubElement(attributes, "divisions").text = str(divisions)
                time = etree.SubElement(attributes, "time")
                etree.SubElement(time, "beats").text = "3"
                etree.SubElement(time, "beat-type").text = "2"

            if existing_triplet_prefix and measure_index == 0:
                assert divisions % 3 == 0
                for step in ("C", "D", "E"):
                    note = _note(measure, divisions // 3, "eighth", step)
                    modification = etree.SubElement(note, "time-modification")
                    etree.SubElement(modification, "actual-notes").text = "3"
                    etree.SubElement(modification, "normal-notes").text = "2"
                ordinary_count = 9
            else:
                ordinary_count = 12

            for note_index in range(ordinary_count):
                if clone_first_note and measure_index == 0 and note_index == 0:
                    _note(measure, divisions, "quarter", "G")
                    backup = etree.SubElement(measure, "backup")
                    etree.SubElement(backup, "duration").text = str(divisions // 2)
                else:
                    _note(measure, divisions // 2, "eighth", chr(ord("C") + note_index % 5))

        if coda:
            measure = etree.SubElement(part, "measure", number=str(measures + 1))
            _note(measure, divisions * 2, "half", "C")
            _note(measure, divisions * 2, "half", "G")

    etree.ElementTree(root).write(str(path), encoding="utf-8", xml_declaration=True)


def _evidence_score(path: Path, *, include_coda: bool = True) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Piano"
    part = etree.SubElement(root, "part", id="P1")
    for measure_index in range(11):
        measure = etree.SubElement(part, "measure", number=str(measure_index + 1))
        if measure_index == 0:
            attributes = etree.SubElement(measure, "attributes")
            etree.SubElement(attributes, "divisions").text = "12"
            time = etree.SubElement(attributes, "time")
            etree.SubElement(time, "beats").text = "3"
            etree.SubElement(time, "beat-type").text = "2"
        explicit = measure_index >= 8
        for note_index in range(12):
            note = _note(
                measure,
                4 if explicit else 6,
                "eighth",
                chr(ord("C") + note_index % 5),
            )
            if explicit:
                modification = etree.SubElement(note, "time-modification")
                etree.SubElement(modification, "actual-notes").text = "3"
                etree.SubElement(modification, "normal-notes").text = "2"
    if include_coda:
        half_coda = etree.SubElement(part, "measure", number="12")
        _note(half_coda, 24, "half", "C")
        _note(half_coda, 24, "half", "G")
        whole_coda = etree.SubElement(part, "measure", number="13")
        _note(whole_coda, 48, "whole", "C")
    etree.ElementTree(root).write(str(path), encoding="utf-8", xml_declaration=True)


def test_transaction_repairs_supported_grid_and_freezes_coda_divisions(tmp_path: Path) -> None:
    source = tmp_path / "source.musicxml"
    output = tmp_path / "output.musicxml"
    _score(source, coda=True)

    report = apply_confirmed_continuous_triplet_grid(
        source,
        output,
        confirmed_meter=(2, 2),
    )

    assert report.applied is True
    assert report.supported_measures == 8
    assert report.notes_converted == 96
    parsed = score_from_tree(etree.parse(str(output)))
    assert parsed.measures[0].time_signature == (2, 2)
    assert all(
        note.duration.denominator == 3 and note.tuple_ratio == (3, 2)
        for measure in parsed.measures[:8]
        for note in measure.notes
    )
    # The unrepaired coda inherited divisions=4 in the source.  It must not
    # inherit the repaired grid's divisions=12 and shrink to two thirds.
    assert [note.duration for note in parsed.measures[8].notes] == [2, 2]
    assert parsed.measures[8].divisions == 4


def test_transaction_preserves_existing_triplets_and_clones_conflated_long_note(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing.musicxml"
    existing_output = tmp_path / "existing-output.musicxml"
    _score(existing, divisions=12, existing_triplet_prefix=True)
    existing_report = apply_confirmed_continuous_triplet_grid(
        existing,
        existing_output,
        confirmed_meter=(2, 2),
    )
    assert existing_report.notes_already_correct == 3
    assert existing_report.notes_converted == 93
    assert existing_report.notes_cloned == 0
    assert len(score_from_tree(etree.parse(str(existing_output))).measures[0].notes) == 12

    conflated = tmp_path / "conflated.musicxml"
    conflated_output = tmp_path / "conflated-output.musicxml"
    _score(conflated, clone_first_note=True)
    conflated_report = apply_confirmed_continuous_triplet_grid(
        conflated,
        conflated_output,
        confirmed_meter=(2, 2),
    )
    assert conflated_report.notes_cloned == 1
    first = score_from_tree(etree.parse(str(conflated_output))).measures[0]
    assert len(first.notes) == 13
    assert any(note.note_type == "quarter" and note.duration == 1 for note in first.notes)
    assert any(
        note.note_type == "eighth" and note.duration == pytest.approx(1 / 3)
        for note in first.notes
    )


def test_transaction_abstains_without_whole_score_support(tmp_path: Path) -> None:
    source = tmp_path / "short.musicxml"
    output = tmp_path / "should-not-exist.musicxml"
    _score(source, measures=7)

    report = apply_confirmed_continuous_triplet_grid(
        source,
        output,
        confirmed_meter=(2, 2),
    )

    assert report.applied is False
    assert report.supported_measures == 7
    assert not output.exists()


def test_transaction_rejects_wrong_meter_and_multi_part_scores(tmp_path: Path) -> None:
    source = tmp_path / "source.musicxml"
    output = tmp_path / "output.musicxml"
    _score(source)
    with pytest.raises(ValueError, match="confirmed cut time"):
        apply_confirmed_continuous_triplet_grid(
            source,
            output,
            confirmed_meter=(3, 2),
        )

    multi = tmp_path / "multi.musicxml"
    _score(multi, parts=2)
    report = apply_confirmed_continuous_triplet_grid(
        multi,
        output,
        confirmed_meter=(2, 2),
    )
    assert report.applied is False
    assert report.parts_seen == 2


def test_evidence_detector_requires_all_four_independent_families(tmp_path: Path) -> None:
    supported = tmp_path / "supported.musicxml"
    output = tmp_path / "supported-output.musicxml"
    _evidence_score(supported)

    evidence = detect_continuous_triplet_grid_evidence(supported)
    assert evidence.authorized is True
    assert evidence.inferred_meter == (2, 2)
    assert evidence.fully_explicit_grid_measures == 3
    assert evidence.fully_unmarked_grid_measures == 8
    assert evidence.plain_four_quarter_coda_measures == 2
    wrapper_evidence, report = apply_evidence_confirmed_continuous_triplet_grid(
        supported,
        output,
    )
    assert wrapper_evidence == evidence
    assert report.applied is True
    assert output.is_file()

    no_coda = tmp_path / "no-coda.musicxml"
    no_coda_output = tmp_path / "no-coda-output.musicxml"
    _evidence_score(no_coda, include_coda=False)
    no_coda_evidence, no_coda_report = apply_evidence_confirmed_continuous_triplet_grid(
        no_coda,
        no_coda_output,
    )
    assert no_coda_evidence.authorized is False
    assert "plain_four_quarter_coda" in no_coda_evidence.reason
    assert no_coda_report.applied is False
    assert not no_coda_output.exists()

    unmarked = tmp_path / "real-three-two-like.musicxml"
    _score(unmarked)
    unmarked_evidence = detect_continuous_triplet_grid_evidence(unmarked)
    assert unmarked_evidence.authorized is False
    assert unmarked_evidence.explicit_triplet_slots == 0
