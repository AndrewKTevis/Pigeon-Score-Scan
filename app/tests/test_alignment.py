from pathlib import Path

from lxml import etree

from scorescan.alignment import align_measure_sequences
from scorescan.consensus import build_measure_consensus
from scorescan.musicxml import MUSICXML_DOCTYPE
from scorescan.score_ir import score_from_tree
from scorescan.selection_risk import SelectionRiskResult


def write_score(path: Path, pitches: list[str]) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    for number, step in enumerate(pitches, start=1):
        measure = etree.SubElement(part, "measure", number=str(number))
        if number == 1:
            attrs = etree.SubElement(measure, "attributes")
            etree.SubElement(attrs, "divisions").text = "1"
            time = etree.SubElement(attrs, "time")
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


class Candidate:
    def __init__(self, variant: str, path: Path, score: float = 1000.0) -> None:
        self.variant = variant
        self.xml_path = str(path)
        self.score = score
        self.valid = True


def test_alignment_recovers_after_inserted_measure(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.musicxml"
    candidate_path = tmp_path / "candidate.musicxml"
    write_score(reference_path, ["C", "D", "E", "F"])
    write_score(candidate_path, ["C", "G", "D", "E", "F"])
    reference = score_from_tree(etree.parse(str(reference_path)))
    candidate = score_from_tree(etree.parse(str(candidate_path)))
    alignment = align_measure_sequences(reference.measures, candidate.measures)
    assert alignment.reference_to_candidate == (0, 2, 3, 4)
    assert alignment.unmatched_candidate_indices == (1,)
    assert alignment.similarity > 0.75


def test_consensus_uses_aligned_candidate_after_gap(tmp_path: Path, monkeypatch) -> None:
    # This test isolates global measure alignment. Replacement-risk behavior is
    # covered separately, so accept an otherwise eligible vote after alignment.
    monkeypatch.setattr(
        "scorescan.consensus.SelectionRiskCalibrator.calibrate",
        lambda self, item: SelectionRiskResult(
            probability=1.0,
            threshold=0.9,
            accepted=True,
            model_version="alignment-test",
            target_precision=1.0,
        ),
    )
    primary = tmp_path / "primary.musicxml"
    inserted = tmp_path / "inserted.musicxml"
    supporting = tmp_path / "supporting.musicxml"
    restoration = tmp_path / "restoration.musicxml"
    output = tmp_path / "output.musicxml"
    write_score(primary, ["C", "D", "F", "G"])
    write_score(inserted, ["C", "A", "D", "E", "G"])
    write_score(supporting, ["C", "D", "E", "G"])
    write_score(restoration, ["C", "D", "E", "G"])
    report = build_measure_consensus(
        [
            Candidate("primary", primary, 1040),
            Candidate("inserted", inserted, 1030),
            Candidate("supporting", supporting, 1020),
            Candidate("flat", restoration, 1010),
        ],
        output,
        "primary",
    )
    assert report is not None
    assert report.candidate_alignment["inserted"]["extra_candidate_measures"] == 1
    assert etree.parse(str(output)).getroot().findtext("./part/measure[3]/note/pitch/step") == "E"
