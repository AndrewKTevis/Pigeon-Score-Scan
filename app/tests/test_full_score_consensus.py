from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lxml import etree

from scorescan.full_score_consensus import build_full_score_consensus


@dataclass(frozen=True)
class Candidate:
    variant: str
    xml_path: str | None
    score: float = 100.0
    valid: bool = True


def _write_score(
    path: Path,
    *,
    second_steps: tuple[str, str],
    part_count: int = 2,
) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    for part_index in range(part_count):
        part_id = f"P{part_index + 1}"
        score_part = etree.SubElement(part_list, "score-part", id=part_id)
        etree.SubElement(score_part, "part-name").text = part_id
        part = etree.SubElement(root, "part", id=part_id)
        for measure_index, step in enumerate(
            ("C", second_steps[part_index]),
            start=1,
        ):
            measure = etree.SubElement(part, "measure", number=str(measure_index))
            if measure_index == 2:
                print_element = etree.SubElement(measure, "print")
                print_element.set("new-system", "yes")
            if measure_index == 1:
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
    etree.ElementTree(root).write(
        str(path),
        encoding="UTF-8",
        xml_declaration=True,
    )


def test_full_score_consensus_replaces_all_parts_as_one_measure_transaction(
    tmp_path: Path,
) -> None:
    template = tmp_path / "primary.musicxml"
    correct_paths = {
        variant: tmp_path / f"{variant}.musicxml"
        for variant in ("flat", "deblock", "upscale", "staffnorm", "adaptive")
    }
    _write_score(template, second_steps=("C", "E"))
    for path in correct_paths.values():
        _write_score(path, second_steps=("D", "F"))
    candidates = [
        Candidate("primary", str(template), 120.0),
        *(Candidate(variant, str(path)) for variant, path in correct_paths.items()),
    ]
    output = tmp_path / "selected.musicxml"

    report = build_full_score_consensus(
        candidates,
        output,
        "primary",
        target_measure_count=2,
    )

    assert report is not None
    assert report.replacements == 1
    assert report.unresolved_measure_indices == ()
    tree = etree.parse(str(output))
    parts = tree.getroot().findall("part")
    assert [
        part.findall("measure")[1].findtext("note/pitch/step")
        for part in parts
    ] == ["D", "F"]
    assert parts[0].find("measure[@number='2']/print").get("new-system") == "yes"


def test_full_score_consensus_abstains_when_one_family_is_incomplete(
    tmp_path: Path,
) -> None:
    template = tmp_path / "primary.musicxml"
    _write_score(template, second_steps=("C", "E"))
    candidates = [Candidate("primary", str(template), 120.0)]
    for variant in ("flat", "deblock", "upscale", "staffnorm", "adaptive"):
        path = tmp_path / f"{variant}.musicxml"
        _write_score(path, second_steps=("D", "F"))
        candidates.append(Candidate(variant, str(path)))
    # The invalid sibling makes the whole correlated binary family abstain.
    candidates.append(Candidate("otsu", None, valid=False))
    output = tmp_path / "selected.musicxml"

    report = build_full_score_consensus(candidates, output, "primary")

    assert report is not None
    assert report.replacements == 0
    assert 1 in report.unresolved_measure_indices
    tree = etree.parse(str(output))
    assert tree.findtext("./part/measure[@number='2']/note/pitch/step") == "C"
