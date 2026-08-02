from __future__ import annotations

from lxml import etree

from app.tools import evaluate_source_beam_restoration as module


def test_oracle_boxes_reassemble_tiles_without_merging_beam_levels() -> None:
    acceptance = {
        "rows": [
            {
                "crop_xyxy": [0, 0, 1024, 1024],
                "objects": [
                    {
                        "category_id": "beam",
                        "box_xyxy": [900.2, 100.2, 980.2, 110.2],
                    },
                    {
                        "category_id": "genericBarline",
                        "box_xyxy": [500.0, 90.0, 503.0, 180.0],
                    },
                ],
            },
            {
                "crop_xyxy": [768, 0, 1792, 1024],
                "objects": [
                    {
                        "category_id": "beam",
                        "box_xyxy": [132.2, 100.2, 212.2, 110.2],
                    },
                    {
                        "category_id": "beam",
                        "box_xyxy": [132.2, 114.2, 212.2, 124.2],
                    },
                ],
            },
        ]
    }

    boxes = module.oracle_beam_boxes(acceptance)

    assert boxes == (
        (900, 100, 980, 110),
        (900, 114, 980, 124),
    )


def test_pair_parser_is_ordered_and_deduplicated() -> None:
    assert module._parse_pair_ids(("34, 51", "34", "84")) == (34, 51, 84)
    assert module._parse_page_cases(("180:2, 180:17", "180:2")) == (
        (180, 2),
        (180, 17),
    )


def test_oracle_boxes_can_select_one_registered_page() -> None:
    acceptance = {
        "rows": [
            {
                "image": "pages/pair-0180/page-2.jpg",
                "crop_xyxy": [0, 0, 1024, 1024],
                "objects": [
                    {"category_id": "beam", "box_xyxy": [10, 20, 30, 40]},
                ],
            },
            {
                "image": "pages/pair-0180/page-17.jpg",
                "crop_xyxy": [0, 0, 1024, 1024],
                "objects": [
                    {"category_id": "beam", "box_xyxy": [50, 60, 70, 80]},
                ],
            },
        ],
    }
    assert module.oracle_beam_boxes(acceptance, page_number=17) == (
        (50, 60, 70, 80),
    )


def test_musicxml_page_slice_keeps_aligned_parts_and_page_measures(
    tmp_path,
) -> None:
    source = tmp_path / "full.musicxml"
    destination = tmp_path / "page-2.musicxml"
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    for part_id in ("P1", "P2"):
        score_part = etree.SubElement(part_list, "score-part", id=part_id)
        etree.SubElement(score_part, "part-name").text = part_id
        part = etree.SubElement(root, "part", id=part_id)
        for number in range(1, 5):
            measure = etree.SubElement(part, "measure", number=str(number))
            if number == 3:
                etree.SubElement(measure, "print", attrib={"new-page": "yes"})
            note = etree.SubElement(measure, "note")
            etree.SubElement(note, "rest")
            etree.SubElement(note, "duration").text = "4"
            etree.SubElement(note, "type").text = "whole"
    source.write_bytes(
        etree.tostring(
            root,
            encoding="UTF-8",
            xml_declaration=True,
            doctype=module.MUSICXML_DOCTYPE,
        )
    )

    assert module._extract_musicxml_page(source, destination, 2) == 2
    sliced = etree.parse(str(destination)).getroot()
    assert [
        [measure.get("number") for measure in part.findall("measure")]
        for part in sliced.findall("part")
    ] == [["3", "4"], ["3", "4"]]


def test_page_svg_oracle_keeps_objects_dropped_at_tile_edges(tmp_path) -> None:
    svg = tmp_path / "page-1.svg"
    svg.write_text(
        """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2000 1000">
          <path class="Beam" d="M900,100 L1100,100 L1100,110 L900,110 Z"/>
          <path class="Beam" d="M1200,120 L1300,120 L1300,130 L1200,130 Z"/>
        </svg>
        """,
        encoding="utf-8",
    )

    assert module.oracle_svg_category_boxes(svg, "beam") == (
        (900, 100, 1100, 110),
        (1200, 120, 1300, 130),
    )
