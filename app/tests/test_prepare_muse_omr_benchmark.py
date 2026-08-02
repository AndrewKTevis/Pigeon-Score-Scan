from __future__ import annotations

from pathlib import Path

from lxml import etree

import pytest

from app.tools.prepare_muse_omr_benchmark import (
    TRAINING_BOUNDARY_CLASSIFICATION_ROLE,
    _boundary_output_role,
    _selection_work_map,
    analyze_reference_boundary,
    production_page_coverage,
    unique_work_cases,
)
from app.tools.muse_omr_contract import (
    BENCHMARK_SELECTION_ROLE,
    TRAINING_SELECTION_ROLE,
)
from scorescan.product_scope import PRODUCTION_BOUNDARY_CONTRACT_VERSION


def _score(
    path: Path,
    *,
    staff_counts: tuple[int, ...],
    voices: tuple[tuple[int, ...], ...],
    cross_staff_beam: bool = False,
    lyrics: bool = False,
) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    for part_index, staff_count in enumerate(staff_counts, start=1):
        part_id = f"P{part_index}"
        score_part = etree.SubElement(part_list, "score-part", id=part_id)
        etree.SubElement(score_part, "part-name").text = part_id
        part = etree.SubElement(root, "part", id=part_id)
        measure = etree.SubElement(part, "measure", number="1")
        attributes = etree.SubElement(measure, "attributes")
        etree.SubElement(attributes, "divisions").text = "1"
        if staff_count > 1:
            etree.SubElement(attributes, "staves").text = str(staff_count)
        for staff_number, staff_voices in enumerate(voices[part_index - 1], start=1):
            for voice_number in range(1, staff_voices + 1):
                if staff_number > 1 or voice_number > 1:
                    backup = etree.SubElement(measure, "backup")
                    etree.SubElement(backup, "duration").text = "1"
                note = etree.SubElement(measure, "note")
                pitch = etree.SubElement(note, "pitch")
                etree.SubElement(pitch, "step").text = "C"
                etree.SubElement(pitch, "octave").text = "4"
                etree.SubElement(note, "duration").text = "1"
                etree.SubElement(note, "voice").text = str(voice_number)
                etree.SubElement(note, "staff").text = str(staff_number)
                if lyrics:
                    lyric = etree.SubElement(note, "lyric")
                    etree.SubElement(lyric, "text").text = "la"
        if cross_staff_beam and staff_count == 2:
            begin, end = measure.findall("note")[:2]
            begin.find("staff").text = "1"
            end.find("staff").text = "2"
            etree.SubElement(begin, "beam", number="1").text = "begin"
            etree.SubElement(end, "beam", number="1").text = "end"
    etree.ElementTree(root).write(str(path), encoding="UTF-8", xml_declaration=True)


def test_boundary_accepts_common_two_voice_piano(tmp_path: Path) -> None:
    path = tmp_path / "piano.musicxml"
    _score(path, staff_counts=(2,), voices=((2, 2),))

    result = analyze_reference_boundary(path)

    assert result["accepted"] is True
    assert result["contract_version"] == PRODUCTION_BOUNDARY_CONTRACT_VERSION
    assert result["score_shape"] == "keyboard"
    assert result["counts"]["maximum_voices_per_staff"] == 2


def test_boundary_accepts_keyboard_plus_single_staff_ensemble(tmp_path: Path) -> None:
    path = tmp_path / "mixed.musicxml"
    _score(path, staff_counts=(1, 2, 1), voices=((1,), (2, 1), (1,)))

    result = analyze_reference_boundary(path)

    assert result["accepted"] is True
    assert result["score_shape"] == "keyboard_plus_single_staff_ensemble"


def test_boundary_keeps_cross_staff_piano_but_rejects_lyrics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "outside.musicxml"
    _score(
        path,
        staff_counts=(2,),
        voices=((1, 1),),
        cross_staff_beam=True,
        lyrics=True,
    )

    result = analyze_reference_boundary(path)

    assert result["accepted"] is False
    assert "cross_staff_beaming" not in result["reasons"]
    assert "lyrics" in result["reasons"]
    assert result["counts"]["cross_staff_beam_groups"] == 1

    piano_only = tmp_path / "cross-staff-piano.musicxml"
    _score(
        piano_only,
        staff_counts=(2,),
        voices=((1, 1),),
        cross_staff_beam=True,
    )
    piano_result = analyze_reference_boundary(piano_only)
    assert piano_result["accepted"] is True
    assert piano_result["counts"]["cross_staff_beam_groups"] == 1


def test_boundary_rejects_two_keyboard_parts_or_three_voices(tmp_path: Path) -> None:
    path = tmp_path / "outside.musicxml"
    _score(path, staff_counts=(2, 2), voices=((3, 1), (1, 1)))

    result = analyze_reference_boundary(path)

    assert result["accepted"] is False
    assert "more_than_one_keyboard_part" in result["reasons"]
    assert (
        "more_than_one_independent_voice_per_non_keyboard_staff"
        not in result["reasons"]
    )
    assert result["counts"]["maximum_voices_per_keyboard_staff"] == 3


def test_boundary_accepts_eight_keyboard_voices_but_not_nine(
    tmp_path: Path,
) -> None:
    accepted_path = tmp_path / "eight-voice-piano.musicxml"
    _score(accepted_path, staff_counts=(2,), voices=((8, 2),))
    accepted = analyze_reference_boundary(accepted_path)
    assert accepted["accepted"] is True
    assert accepted["counts"]["maximum_voices_per_keyboard_staff"] == 8

    rejected_path = tmp_path / "nine-voice-piano.musicxml"
    _score(rejected_path, staff_counts=(2,), voices=((9, 2),))
    rejected = analyze_reference_boundary(rejected_path)
    assert rejected["accepted"] is False
    assert (
        "more_than_eight_independent_voices_per_keyboard_staff"
        in rejected["reasons"]
    )


def test_boundary_counts_simultaneous_voices_per_measure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "voice-numbers-reused.musicxml"
    _score(path, staff_counts=(2,), voices=((1, 1),))
    tree = etree.parse(str(path))
    part = tree.getroot().find("part")
    assert part is not None
    original = part.find("measure")
    assert original is not None
    for index, voice_number in enumerate((2, 3, 4, 5), start=2):
        measure = etree.fromstring(etree.tostring(original))
        measure.set("number", str(index))
        for voice in measure.findall("./note/voice"):
            voice.text = str(voice_number)
        part.append(measure)
    tree.write(str(path), encoding="UTF-8", xml_declaration=True)

    result = analyze_reference_boundary(path)

    assert result["accepted"] is True
    assert result["counts"]["maximum_voices_per_keyboard_staff"] == 1


def test_non_keyboard_sequential_voice_labels_are_not_divisi(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sequential-voice-labels.musicxml"
    _score(path, staff_counts=(1,), voices=((1,),))
    tree = etree.parse(str(path))
    measure = tree.getroot().find("./part/measure")
    assert measure is not None
    second = etree.fromstring(etree.tostring(measure.find("note")))
    second.find("voice").text = "2"
    measure.append(second)
    tree.write(str(path), encoding="UTF-8", xml_declaration=True)

    result = analyze_reference_boundary(path)

    assert result["accepted"] is True
    assert result["counts"]["maximum_voices_per_non_keyboard_staff"] == 1


def test_boundary_accepts_temporary_third_keyboard_staff(
    tmp_path: Path,
) -> None:
    path = tmp_path / "three-staff-piano.musicxml"
    _score(path, staff_counts=(3,), voices=((2, 1, 1),))
    result = analyze_reference_boundary(path)
    assert result["accepted"] is True
    assert result["score_shape"] == "keyboard"
    assert result["part_staff_counts"] == [3]


def test_boundary_rejects_more_than_one_non_keyboard_voice(
    tmp_path: Path,
) -> None:
    path = tmp_path / "divisi-solo.musicxml"
    _score(path, staff_counts=(1,), voices=((2,),))
    result = analyze_reference_boundary(path)
    assert result["accepted"] is False
    assert (
        "more_than_one_independent_voice_per_non_keyboard_staff"
        in result["reasons"]
    )


def test_boundary_rejects_more_than_sixteen_physical_staves(
    tmp_path: Path,
) -> None:
    path = tmp_path / "oversized-ensemble.musicxml"
    _score(
        path,
        staff_counts=(1,) * 17,
        voices=((1,),) * 17,
    )

    result = analyze_reference_boundary(path)

    assert result["accepted"] is False
    assert "more_than_16_physical_staves" in result["reasons"]


def test_selection_work_map_rejects_duplicate_variant_rows() -> None:
    selection = {
        "pair_work_fingerprints": [
            {"pair_id": 1, "work_fingerprint": "a" * 64},
            {"pair_id": 1, "work_fingerprint": "a" * 64},
        ],
        "selected_work_fingerprints": ["a" * 64],
        "selected_work_count": 1,
    }
    with pytest.raises(ValueError, match="invalid pair/work provenance"):
        _selection_work_map(selection, [1])


def test_unique_work_coverage_counts_one_submitted_document() -> None:
    cases = [
        {
            "id": "muse-1",
            "work_fingerprint": "a" * 64,
            "input_pdf_pages": 5,
            "boundary": {"score_shape": "keyboard"},
        },
        {
            "id": "muse-2",
            "work_fingerprint": "a" * 64,
            "input_pdf_pages": 7,
            "boundary": {"score_shape": "keyboard"},
        },
        {
            "id": "muse-3",
            "work_fingerprint": "b" * 64,
            "input_pdf_pages": 3,
            "boundary": {"score_shape": "single_staff_solo"},
        },
    ]

    unique = unique_work_cases(cases)
    total, by_configuration = production_page_coverage(unique)

    assert [case["id"] for case in unique] == ["muse-1", "muse-3"]
    assert total == 8
    assert by_configuration == {
        "solo_monophonic": 3,
        "piano": 5,
        "monophonic_ensemble": 0,
        "piano_plus_monophonic_ensemble": 0,
    }


def test_training_selection_cannot_be_mislabeled_as_release_benchmark() -> None:
    assert (
        _boundary_output_role(
            BENCHMARK_SELECTION_ROLE,
            allow_training_classification=False,
        )
        == BENCHMARK_SELECTION_ROLE
    )
    with pytest.raises(ValueError, match="not authorized"):
        _boundary_output_role(
            TRAINING_SELECTION_ROLE,
            allow_training_classification=False,
        )
    assert (
        _boundary_output_role(
            TRAINING_SELECTION_ROLE,
            allow_training_classification=True,
        )
        == TRAINING_BOUNDARY_CLASSIFICATION_ROLE
    )
