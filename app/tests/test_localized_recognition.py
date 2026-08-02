from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from lxml import etree

from scorescan.layout import PageLayout, StaffSystem
from scorescan.localized_recognition import (
    create_system_crops,
    localized_recognition_eligible,
    merge_localized_system_musicxml,
    validate_localized_system_xml,
)
from scorescan.models import PageInfo
from scorescan.musicxml import MUSICXML_DOCTYPE, validate_musicxml
from scorescan.omr import EngineResult
from scorescan.recognition import (
    RecognitionCandidate,
    RecognitionEnsemble,
    _analysis_issues_regressed,
    candidate_set_is_ambiguous,
)
from scorescan.variant_family import variant_family


def _system(index: int, top: int, bottom: int, measures: int = 1) -> StaffSystem:
    return StaffSystem(
        index=index,
        line_y=[top + 20 + offset * 12 for offset in range(5)],
        top=top,
        bottom=bottom,
        left=80,
        right=920,
        spacing=12.0,
        barlines=[],
        measure_count=measures,
    )


def _write_score(path: Path, measures: int, step: str = "C") -> None:
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
            clef = etree.SubElement(attributes, "clef")
            etree.SubElement(clef, "sign").text = "G"
            etree.SubElement(clef, "line").text = "2"
            time = etree.SubElement(attributes, "time")
            etree.SubElement(time, "beats").text = "4"
            etree.SubElement(time, "beat-type").text = "4"
        note = etree.SubElement(measure, "note")
        pitch = etree.SubElement(note, "pitch")
        etree.SubElement(pitch, "step").text = step
        etree.SubElement(pitch, "octave").text = "4"
        etree.SubElement(note, "duration").text = "8"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "whole"
    etree.ElementTree(root).write(
        str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE
    )


def test_system_crops_are_bounded_and_provenanced(tmp_path: Path) -> None:
    image = np.full((600, 1000), 255, np.uint8)
    for y in (120, 132, 144, 156, 168, 360, 372, 384, 396, 408):
        cv2.line(image, (80, y), (920, y), 0, 1)
    page = tmp_path / "page.png"
    cv2.imwrite(str(page), image)
    layout = PageLayout(
        width=1000,
        height=600,
        systems=[_system(1, 80, 210, 3), _system(2, 320, 450, 4)],
        confidence=0.95,
    )

    crops = create_system_crops(page, layout, tmp_path / "localized")

    assert len(crops) == 2
    assert crops[0].expected_measure_count == 3
    assert crops[1].expected_measure_count == 4
    assert crops[0].source_bbox[3] <= crops[1].source_bbox[1]
    assert all(Path(crop.image_path).is_file() for crop in crops)
    assert all(len(crop.sha256) == 64 for crop in crops)
    assert (tmp_path / "localized" / "system_crops.json").is_file()


def test_localized_recognition_fails_closed_on_weak_layout() -> None:
    eligible, reason = localized_recognition_eligible(
        PageLayout(1000, 600, [_system(1, 80, 210), _system(2, 320, 450)], 0.2)
    )
    assert not eligible
    assert reason == "layout_confidence_low"


def test_merge_localized_system_musicxml_is_complete_and_renumbered(tmp_path: Path) -> None:
    first = tmp_path / "first.musicxml"
    second = tmp_path / "second.musicxml"
    merged = tmp_path / "merged.musicxml"
    _write_score(first, 2, "C")
    _write_score(second, 1, "D")

    count = merge_localized_system_musicxml((first, second), merged)

    assert count == 3
    assert validate_musicxml(merged) == []
    tree = etree.parse(str(merged))
    measures = tree.getroot().findall("./part/measure")
    assert [measure.get("number") for measure in measures] == ["1", "2", "3"]
    assert measures[2].findtext("note/pitch/step") == "D"


def test_localized_system_measure_gap_is_rejected(tmp_path: Path) -> None:
    score = tmp_path / "system.musicxml"
    _write_score(score, 5)
    valid, observed, error = validate_localized_system_xml(score, 1)
    assert not valid
    assert observed == 5
    assert "gap" in str(error)


def test_localized_variant_is_an_independent_family() -> None:
    assert variant_family("system_localized") == "localization"


def test_ambiguity_uses_actual_top_two_candidates_not_input_order() -> None:
    def candidate(name: str, score: float, measures: int) -> RecognitionCandidate:
        return RecognitionCandidate(
            variant=name,
            image_path=f"{name}.png",
            xml_path=f"{name}.musicxml",
            score=score,
            valid=True,
            elapsed_seconds=1.0,
            measure_count=measures,
            note_count=20,
            agreement_ratio=0.95,
        )

    low_unrelated = candidate("low", 100.0, 99)
    best = candidate("best", 1000.0, 8)
    runner_up = candidate("runner", 900.0, 8)
    assert not candidate_set_is_ambiguous([low_unrelated, best, runner_up])


def test_consensus_issue_gate_rejects_only_new_or_additional_issues() -> None:
    pickup = {"measure": "1", "actual": 3, "expected": 9}
    new_issue = {"measure": "8", "actual": 8, "expected": 9}

    assert not _analysis_issues_regressed(
        {"rhythm_issues": [pickup]},
        {"rhythm_issues": [pickup]},
        "rhythm_issues",
    )
    assert not _analysis_issues_regressed(
        {"rhythm_issues": [pickup, new_issue]},
        {"rhythm_issues": [pickup]},
        "rhythm_issues",
    )
    assert _analysis_issues_regressed(
        {"rhythm_issues": [pickup]},
        {"rhythm_issues": [pickup, new_issue]},
        "rhythm_issues",
    )
    assert _analysis_issues_regressed(
        {"rhythm_issues": [pickup]},
        {"rhythm_issues": [pickup, pickup]},
        "rhythm_issues",
    )


class _FakeRunner:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def run_page(self, image_path: Path, _cancel_event: object = None) -> EngineResult:
        self.calls.append(image_path.name)
        xml = image_path.with_suffix(".musicxml")
        _write_score(xml, 1 if image_path.name.startswith("system_") else 2)
        return EngineResult(0, xml, 0.01)


def test_risky_page_adds_complete_system_localized_candidate(tmp_path: Path, monkeypatch) -> None:
    image = np.full((600, 1000), 255, np.uint8)
    page_path = tmp_path / "page.png"
    cv2.imwrite(str(page_path), image)
    variants = []
    for name in ("primary", "flat", "otsu"):
        path = tmp_path / f"{name}.png"
        cv2.imwrite(str(path), image)
        variants.append((name, path))
    layout = PageLayout(
        1000,
        600,
        [_system(1, 80, 210, 1), _system(2, 320, 450, 1)],
        0.95,
    )
    page = PageInfo(index=1, source_name="page.png", image_path=str(page_path))
    page.normalized_path = str(page_path)
    page.estimated_measure_count = 2
    page.quality_score = 50.0

    import scorescan.recognition as recognition

    monkeypatch.setattr(recognition, "generate_omr_variants", lambda *_args, **_kwargs: variants)
    monkeypatch.setattr(recognition, "extract_page_measure_evidence", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(recognition, "build_measure_consensus", lambda *_args, **_kwargs: None)
    runner = _FakeRunner()
    ensemble = RecognitionEnsemble(runner, lambda _message: None, tmp_path / "work")
    progress: list[tuple[float, str]] = []

    result = ensemble.run_page(
        page,
        layout,
        progress_callback=lambda fraction, stage: progress.append((fraction, stage)),
    )

    assert result.selected is not None
    assert any(item.variant == "system_localized" and item.valid for item in result.candidates)
    assert sum(name.startswith("system_") for name in runner.calls) == 2
    assert (tmp_path / "work" / "page_0001" / "localized" / "localized_recognition.json").is_file()
    assert progress
    assert progress[-1][0] == 1.0
    assert [fraction for fraction, _stage in progress] == sorted(
        fraction for fraction, _stage in progress
    )
    assert any(0.08 < fraction < 0.68 for fraction, _stage in progress)
    assert any("system" in stage.casefold() for _fraction, stage in progress)


def test_progress_callback_failure_does_not_fail_recognition(tmp_path: Path, monkeypatch) -> None:
    image = np.full((600, 1000), 255, np.uint8)
    page_path = tmp_path / "page.png"
    cv2.imwrite(str(page_path), image)
    variant_path = tmp_path / "primary.png"
    cv2.imwrite(str(variant_path), image)
    page = PageInfo(index=1, source_name="page.png", image_path=str(page_path))
    page.normalized_path = str(page_path)
    page.estimated_measure_count = 2
    page.quality_score = 95.0

    import scorescan.recognition as recognition

    monkeypatch.setattr(
        recognition,
        "generate_omr_variants",
        lambda *_args, **_kwargs: [("primary", variant_path)],
    )
    monkeypatch.setattr(recognition, "extract_page_measure_evidence", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(recognition, "build_measure_consensus", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        recognition,
        "should_run_localized_recognition",
        lambda *_args, **_kwargs: (False, "not_needed"),
    )
    messages: list[str] = []
    callback_calls = 0

    def broken_callback(_fraction: float, _stage: str) -> None:
        nonlocal callback_calls
        callback_calls += 1
        raise RuntimeError("synthetic persistence failure")

    result = RecognitionEnsemble(
        _FakeRunner(), messages.append, tmp_path / "work"
    ).run_page(page, None, progress_callback=broken_callback)

    assert result.selected is not None
    assert callback_calls == 1
    assert any("进度更新失败" in message for message in messages)


def test_localized_recognition_rejects_overlapping_system_geometry() -> None:
    eligible, reason = localized_recognition_eligible(
        PageLayout(
            1000,
            600,
            [_system(1, 80, 240), _system(2, 210, 370)],
            0.95,
        )
    )
    assert not eligible
    assert reason == "systems_overlap"
