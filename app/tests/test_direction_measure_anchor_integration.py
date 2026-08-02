from __future__ import annotations

import cv2
import numpy as np
from lxml import etree

from scorescan.layout import PageLayout, StaffSystem
from scorescan.text_enrichment import enrich_musicxml_with_ocr


def _musicxml(measures: int) -> bytes:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    for index in range(measures):
        measure = etree.SubElement(part, "measure", number=str(index + 1))
        if index == 0:
            attributes = etree.SubElement(measure, "attributes")
            etree.SubElement(attributes, "divisions").text = "1"
        note = etree.SubElement(measure, "note")
        etree.SubElement(note, "rest")
        etree.SubElement(note, "duration").text = "4"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "whole"
    return etree.tostring(root, encoding="UTF-8", xml_declaration=True)


def test_ocr_direction_uses_refined_measure_ownership(monkeypatch, tmp_path) -> None:
    import scorescan.text_enrichment as module

    system = StaffSystem(
        index=1,
        line_y=[300.0, 312.3535, 324.7070, 337.0605, 349.4140],
        top=250,
        bottom=398,
        left=107,
        right=928,
        spacing=12.3535,
        barlines=[386, 748],
        measure_count=3,
    )
    layout = PageLayout(width=1000, height=500, systems=[system], confidence=0.95)
    x = 773.3959
    box = [[x - 16, 365], [x + 16, 365], [x + 16, 384], [x - 16, 384]]
    monkeypatch.setattr(
        module,
        "run_ocr",
        lambda image_path, page_layout: (
            [("mf", 0.98, box, "rapid+tesseract")],
            "rapid+tesseract",
        ),
    )
    image_path = tmp_path / "page.png"
    cv2.imwrite(str(image_path), np.full((500, 1000), 255, np.uint8))
    xml_path = tmp_path / "score.musicxml"
    xml_path.write_bytes(_musicxml(4))

    marks, warnings = enrich_musicxml_with_ocr(image_path, xml_path, layout)

    assert not warnings
    assert len(marks) == 1
    assert marks[0].measure_index == 3
    assert marks[0].measure_anchor_method == "barline_model_refined"
    assert marks[0].injected
    tree = etree.parse(str(xml_path))
    measures = tree.getroot().find("part").findall("measure")
    assert measures[2].find("direction") is None
    assert measures[3].find("./direction/direction-type/dynamics/mf") is not None


def test_initial_scanned_metronome_ocr_substitution_is_injected(
    monkeypatch,
    tmp_path,
) -> None:
    """Regression for the user's ``d=62`` quarter-note OCR result."""

    import scorescan.text_enrichment as module

    system = StaffSystem(
        index=1,
        line_y=[194.0, 202.0, 210.0, 218.0, 226.0],
        top=150,
        bottom=250,
        left=158,
        right=368,
        spacing=8.0,
        barlines=[158, 368],
        measure_count=1,
    )
    layout = PageLayout(928, 500, [system], 0.98)
    monkeypatch.setattr(
        module,
        "run_ocr",
        lambda image_path, page_layout: (
            [
                (
                    "d=62",
                    0.995,
                    [[171, 151], [232, 151], [232, 184], [171, 184]],
                    "rapid-no-lines+rapid-original",
                )
            ],
            "rapid-no-lines+rapid-original",
        ),
    )
    image_path = tmp_path / "page.png"
    cv2.imwrite(str(image_path), np.full((500, 928), 255, np.uint8))
    xml_path = tmp_path / "score.musicxml"
    xml_path.write_bytes(_musicxml(1))

    marks, warnings = enrich_musicxml_with_ocr(
        image_path,
        xml_path,
        layout,
    )

    assert not warnings
    assert len(marks) == 1
    assert marks[0].kind == "metronome"
    assert marks[0].measure_anchor_method == "initial_metronome_start"
    assert marks[0].injected
    metronome = etree.parse(str(xml_path)).find(
        "./part/measure/direction/direction-type/metronome"
    )
    assert metronome is not None
    assert metronome.findtext("beat-unit") == "quarter"
    assert metronome.findtext("per-minute") == "62"


def test_tempo_term_is_not_promoted_to_work_title(
    monkeypatch,
    tmp_path,
) -> None:
    import scorescan.text_enrichment as module

    system = StaffSystem(
        index=1,
        line_y=[194.0, 202.0, 210.0, 218.0, 226.0],
        top=150,
        bottom=250,
        left=158,
        right=768,
        spacing=8.0,
        barlines=[158, 768],
        measure_count=1,
    )
    layout = PageLayout(928, 500, [system], 0.98)
    monkeypatch.setattr(
        module,
        "run_ocr",
        lambda image_path, page_layout: (
            [
                (
                    "Allegretto",
                    0.995,
                    [[330, 55], [500, 55], [500, 82], [330, 82]],
                    "rapid-no-lines+rapid-semantic-region:tempoText",
                )
            ],
            "rapid-no-lines+rapid-semantic-region:tempoText",
        ),
    )
    image_path = tmp_path / "page.png"
    cv2.imwrite(str(image_path), np.full((500, 928), 255, np.uint8))
    xml_path = tmp_path / "score.musicxml"
    xml_path.write_bytes(_musicxml(1))

    marks, warnings = enrich_musicxml_with_ocr(
        image_path,
        xml_path,
        layout,
    )

    assert not warnings
    assert marks[0].kind == "direction"
    assert marks[0].injected
    tree = etree.parse(str(xml_path))
    assert tree.findtext("./work/work-title") is None
    assert tree.findtext(
        "./part/measure/direction/direction-type/words"
    ) == "Allegretto"


def test_instrument_name_semantic_role_cannot_become_timeline_direction(
    monkeypatch,
    tmp_path,
) -> None:
    import scorescan.text_enrichment as module

    system = StaffSystem(
        index=1,
        line_y=[194.0, 202.0, 210.0, 218.0, 226.0],
        top=150,
        bottom=250,
        left=158,
        right=768,
        spacing=8.0,
        barlines=[158, 460, 768],
        measure_count=2,
    )
    layout = PageLayout(928, 500, [system], 0.98)
    monkeypatch.setattr(
        module,
        "run_ocr",
        lambda image_path, page_layout: (
            [
                (
                    "Piano",
                    0.995,
                    [[180, 151], [280, 151], [280, 178], [180, 178]],
                    "phrase:rapid-semantic-region:instrumentNameText+rapid-no-lines",
                )
            ],
            "phrase:rapid-semantic-region:instrumentNameText+rapid-no-lines",
        ),
    )
    image_path = tmp_path / "page.png"
    cv2.imwrite(str(image_path), np.full((500, 928), 255, np.uint8))
    xml_path = tmp_path / "score.musicxml"
    xml_path.write_bytes(_musicxml(2))

    marks, warnings = enrich_musicxml_with_ocr(
        image_path,
        xml_path,
        layout,
    )

    assert not warnings
    assert len(marks) == 1
    assert marks[0].kind in {"direction", "text"}
    assert not marks[0].injected
    assert etree.parse(str(xml_path)).find(
        "./part/measure/direction"
    ) is None


def test_text_left_of_system_cannot_be_clamped_into_first_measure(
    monkeypatch,
    tmp_path,
) -> None:
    import scorescan.text_enrichment as module

    system = StaffSystem(
        index=1,
        line_y=[194.0, 202.0, 210.0, 218.0, 226.0],
        top=150,
        bottom=250,
        left=200,
        right=768,
        spacing=8.0,
        barlines=[200, 460, 768],
        measure_count=2,
    )
    layout = PageLayout(928, 500, [system], 0.98)
    monkeypatch.setattr(
        module,
        "run_ocr",
        lambda image_path, page_layout: (
            [
                (
                    "Allegro",
                    0.995,
                    [[70, 174], [170, 174], [170, 198], [70, 198]],
                    "rapid-no-lines",
                )
            ],
            "rapid-no-lines",
        ),
    )
    image_path = tmp_path / "page.png"
    cv2.imwrite(str(image_path), np.full((500, 928), 255, np.uint8))
    xml_path = tmp_path / "score.musicxml"
    xml_path.write_bytes(_musicxml(2))

    marks, warnings = enrich_musicxml_with_ocr(
        image_path,
        xml_path,
        layout,
    )

    assert not warnings
    assert len(marks) == 1
    assert marks[0].kind == "direction"
    assert not marks[0].injected
    assert marks[0].measure_index is None
    assert etree.parse(str(xml_path)).find(
        "./part/measure/direction"
    ) is None


def test_rehearsal_semantic_role_cannot_reanchor_words_direction(
    monkeypatch,
    tmp_path,
) -> None:
    import scorescan.text_enrichment as module

    system = StaffSystem(
        index=1,
        line_y=[194.0, 202.0, 210.0, 218.0, 226.0],
        top=150,
        bottom=250,
        left=158,
        right=768,
        spacing=8.0,
        barlines=[158, 460, 768],
        measure_count=2,
    )
    layout = PageLayout(928, 500, [system], 0.98)
    monkeypatch.setattr(
        module,
        "run_ocr",
        lambda image_path, page_layout: (
            [
                (
                    "Fine",
                    0.995,
                    [[180, 151], [280, 151], [280, 178], [180, 178]],
                    "rapid-no-lines+rapid-semantic-region:rehearsalMarkText",
                )
            ],
            "rapid-no-lines+rapid-semantic-region:rehearsalMarkText",
        ),
    )
    image_path = tmp_path / "page.png"
    cv2.imwrite(str(image_path), np.full((500, 928), 255, np.uint8))
    xml_path = tmp_path / "score.musicxml"
    root = etree.fromstring(_musicxml(2))
    wrong_measure = root.find("part").findall("measure")[1]
    direction = etree.SubElement(wrong_measure, "direction", placement="above")
    direction_type = etree.SubElement(direction, "direction-type")
    etree.SubElement(direction_type, "words").text = "Fine"
    xml_path.write_bytes(
        etree.tostring(root, encoding="UTF-8", xml_declaration=True)
    )

    marks, warnings = enrich_musicxml_with_ocr(
        image_path,
        xml_path,
        layout,
    )

    assert not warnings
    assert len(marks) == 1
    assert not marks[0].reanchored_existing
    measures = etree.parse(str(xml_path)).find("part").findall("measure")
    assert measures[0].find("direction") is None
    assert measures[1].findtext(
        "./direction/direction-type/words"
    ) == "Fine"


def test_source_proven_exact_text_reanchors_existing_wrong_measure(
    monkeypatch,
    tmp_path,
) -> None:
    import scorescan.text_enrichment as module

    system = StaffSystem(
        index=1,
        line_y=[194.0, 202.0, 210.0, 218.0, 226.0],
        top=150,
        bottom=250,
        left=158,
        right=768,
        spacing=8.0,
        barlines=[158, 460, 768],
        measure_count=2,
    )
    layout = PageLayout(928, 500, [system], 0.98)
    monkeypatch.setattr(
        module,
        "run_ocr",
        lambda image_path, page_layout: (
            [
                (
                    "Allegretto",
                    0.995,
                    [[180, 151], [300, 151], [300, 178], [180, 178]],
                    "rapid-no-lines+rapid-semantic-region:tempoText",
                )
            ],
            "rapid-no-lines+rapid-semantic-region:tempoText",
        ),
    )
    image_path = tmp_path / "page.png"
    cv2.imwrite(str(image_path), np.full((500, 928), 255, np.uint8))
    xml_path = tmp_path / "score.musicxml"
    root = etree.fromstring(_musicxml(2))
    wrong_measure = root.find("part").findall("measure")[1]
    direction = etree.SubElement(
        wrong_measure,
        "direction",
        placement="above",
    )
    direction_type = etree.SubElement(direction, "direction-type")
    etree.SubElement(
        direction_type,
        "words",
        **{"font-style": "italic"},
    ).text = "Allegretto"
    xml_path.write_bytes(
        etree.tostring(root, encoding="UTF-8", xml_declaration=True)
    )

    marks, warnings = enrich_musicxml_with_ocr(
        image_path,
        xml_path,
        layout,
    )

    assert not warnings
    assert marks[0].reanchored_existing
    tree = etree.parse(str(xml_path))
    measures = tree.find("part").findall("measure")
    assert measures[0].findtext(
        "./direction/direction-type/words"
    ) == "Allegretto"
    assert measures[1].find("direction") is None
