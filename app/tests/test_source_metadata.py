from lxml import etree

from scorescan.layout import PageLayout, ScoreSystemLayout, StaffSystem
from scorescan.text_enrichment import OcrMark, _write_source_metadata


def _mark(text: str, box: list[list[float]], backend: str = "rapid+tesseract") -> OcrMark:
    return OcrMark(
        raw_text=text,
        text=text,
        score=0.995,
        box=box,
        kind="text",
        backend=backend,
    )


def _score_root() -> etree._Element:
    root = etree.Element("score-partwise", version="4.0")
    defaults = etree.SubElement(root, "defaults")
    page_layout = etree.SubElement(defaults, "page-layout")
    etree.SubElement(page_layout, "page-height").text = "300"
    etree.SubElement(page_layout, "page-width").text = "220"
    part_list = etree.SubElement(root, "part-list")
    for part_id, name in (("P1", "Voice"), ("P2", "Piano")):
        score_part = etree.SubElement(part_list, "score-part", id=part_id)
        etree.SubElement(score_part, "part-name").text = name
    etree.SubElement(root, "part", id="P1")
    etree.SubElement(root, "part", id="P2")
    return root


def _layout() -> PageLayout:
    staves = [
        StaffSystem(
            index=1,
            line_y=[300, 310, 320, 330, 340],
            top=270,
            bottom=365,
            left=200,
            right=950,
            spacing=10,
            barlines=[200, 500, 950],
            measure_count=2,
        ),
        StaffSystem(
            index=2,
            line_y=[400, 410, 420, 430, 440],
            top=375,
            bottom=455,
            left=200,
            right=950,
            spacing=10,
            barlines=[200, 500, 950],
            measure_count=2,
        ),
        StaffSystem(
            index=3,
            line_y=[480, 490, 500, 510, 520],
            top=460,
            bottom=540,
            left=200,
            right=950,
            spacing=10,
            barlines=[200, 500, 950],
            measure_count=2,
        ),
    ]
    score_system = ScoreSystemLayout(
        index=1,
        staff_indices=[1, 2, 3],
        top=270,
        bottom=540,
        left=200,
        right=950,
        spacing=10,
        barlines=[200, 500, 950],
        measure_count=2,
        grouping_confidence=0.99,
        grouping_method="test",
    )
    return PageLayout(
        width=1000,
        height=1400,
        systems=staves,
        confidence=0.99,
        score_systems=[score_system],
    )


def test_high_confidence_page_furniture_populates_score_metadata() -> None:
    root = _score_root()
    marks = [
        _mark("The Prince of Denmark's March", [[250, 20], [750, 20], [750, 60], [250, 60]]),
        _mark("Trumpet in C and Organ", [[350, 85], [650, 85], [650, 105], [350, 105]]),
        _mark("J. Clarke - S. Depolo - M. Rondeau - L. Grosso", [[470, 210], [900, 210], [900, 230], [470, 230]]),
        _mark("Trumpet in C", [[55, 305], [180, 305], [180, 335], [55, 335]]),
        _mark("Organo", [[80, 435], [175, 435], [175, 470], [80, 470]]),
    ]

    _write_source_metadata(root, marks, _layout(), [1, 2])

    assert root.findtext("./work/work-title") == "The Prince of Denmark's March"
    assert root.findtext("./movement-title") == "The Prince of Denmark's March"
    assert root.findtext("./identification/creator[@type='composer']") == (
        "J. Clarke - S. Depolo - M. Rondeau - L. Grosso"
    )
    credits = {
        credit.findtext("credit-type"): credit.findtext("credit-words")
        for credit in root.findall("credit")
    }
    assert credits == {
        "title": "The Prince of Denmark's March",
        "subtitle": "Trumpet in C and Organ",
        "composer": "J. Clarke - S. Depolo - M. Rondeau - L. Grosso",
    }
    assert root.xpath("./part-list/score-part/part-name/text()") == ["Trumpet in C", "Organo"]
    assert all(mark.injected and mark.kind == "metadata" for mark in marks)
    assert [child.tag for child in root].index("identification") < [child.tag for child in root].index("defaults")
    assert [child.tag for child in root].index("credit") < [child.tag for child in root].index("part-list")


def test_single_backend_does_not_overwrite_public_metadata() -> None:
    root = _score_root()
    mark = _mark(
        "Possible OCR Header",
        [[250, 20], [750, 20], [750, 60], [250, 60]],
        backend="rapid",
    )

    _write_source_metadata(root, [mark], _layout(), [1, 2])

    assert root.find("./work") is None
    assert root.find("./movement-title") is None
    assert root.xpath("./part-list/score-part/part-name/text()") == ["Voice", "Piano"]
    assert not mark.injected
