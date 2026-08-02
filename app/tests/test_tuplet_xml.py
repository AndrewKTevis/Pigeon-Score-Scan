from __future__ import annotations

from pathlib import Path

from lxml import etree

from scorescan.tuplet_xml import (
    read_simple_tuplet_state,
    sanitize_incomplete_implicit_triplets,
    set_simple_tuplet_state,
)


def _score(path: Path, *, implicit_count: int, explicit_count: int = 0) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Piano"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "12"
    for index in range(implicit_count + explicit_count):
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = "C"
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "4"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "eighth"
        etree.SubElement(note, "staff").text = "1"
        explicit_index = index - implicit_count
        set_simple_tuplet_state(
            note,
            ratio=(3, 2),
            start=explicit_index == 0 and explicit_count > 0,
            stop=explicit_index == explicit_count - 1 and explicit_count > 0,
        )
    path.write_bytes(
        etree.tostring(
            root,
            encoding="UTF-8",
            xml_declaration=True,
            pretty_print=True,
        )
    )


def test_sanitizer_restores_only_incomplete_implicit_run(tmp_path: Path) -> None:
    source = tmp_path / "raw.musicxml"
    output = tmp_path / "sanitized.musicxml"
    _score(source, implicit_count=2, explicit_count=3)

    report = sanitize_incomplete_implicit_triplets(source, output)
    notes = etree.parse(str(output)).findall(".//note")
    states = [read_simple_tuplet_state(note) for note in notes]

    assert report["changed_group_count"] == 1
    assert report["changed_event_count"] == 2
    assert report["topology_valid"] is True
    assert [note.findtext("duration") for note in notes[:2]] == ["6", "6"]
    assert all(state is not None and state.ratio is None for state in states[:2])
    assert states[2] is not None and states[2].start
    assert states[4] is not None and states[4].stop


def test_sanitizer_preserves_complete_implicit_run(tmp_path: Path) -> None:
    source = tmp_path / "raw.musicxml"
    output = tmp_path / "sanitized.musicxml"
    _score(source, implicit_count=6)

    report = sanitize_incomplete_implicit_triplets(source, output)
    notes = etree.parse(str(output)).findall(".//note")

    assert report["changed_group_count"] == 0
    assert report["topology_valid"] is True
    assert [note.findtext("duration") for note in notes] == ["4"] * 6
    assert all(
        read_simple_tuplet_state(note).ratio == (3, 2)  # type: ignore[union-attr]
        for note in notes
    )


def test_sanitizer_rejects_unclosed_explicit_simple_triplet(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw.musicxml"
    output = tmp_path / "sanitized.musicxml"
    _score(source, implicit_count=0, explicit_count=3)
    tree = etree.parse(str(source))
    stop = tree.find(".//note[3]/notations/tuplet")
    assert stop is not None
    stop.getparent().remove(stop)
    tree.write(str(source), encoding="UTF-8", xml_declaration=True)

    report = sanitize_incomplete_implicit_triplets(source, output)

    assert report["topology_valid"] is False
    assert report["topology_error_count"] == 1
    assert report["topology_errors"][0]["reason"] == (
        "malformed_explicit_simple_triplet"
    )
