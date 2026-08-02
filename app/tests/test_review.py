from pathlib import Path

from PIL import Image, ImageDraw

from scorescan.models import PageInfo
from scorescan.review import build_text_review_issues
from scorescan.text_enrichment import OcrMark


def test_build_text_review_issue_and_crop(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image = Image.new("RGB", (1200, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.text((300, 200), "Allegro", fill="black")
    image.save(image_path)
    page = PageInfo(1, "page.png", str(image_path), normalized_path=str(image_path))
    mark = OcrMark(
        raw_text="Allegro con brlo",
        text="Allegro con brlo",
        score=0.62,
        box=[[290, 185], [520, 185], [520, 235], [290, 235]],
        kind="direction",
        system_index=0,
        measure_index=0,
        placement="above",
        injected=True,
        correction_probability=0.95,
        correction_margin=0.15,
        distance_staff_spaces=4.0,
    )
    issues = build_text_review_issues([page], {1: [mark]}, {1: 0}, tmp_path / "result")
    assert len(issues) == 1
    assert issues[0].suggested_value == "Allegro con brio"
    assert Path(issues[0].crop_path).exists()
    assert issues[0].global_measure_number == 1


def test_unwritten_single_glyph_dynamic_is_diagnostic_not_review_work(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (1200, 800), "white").save(image_path)
    page = PageInfo(1, "page.png", str(image_path), normalized_path=str(image_path))
    mark = OcrMark(
        raw_text="p",
        text="p",
        score=0.74,
        box=[[360, 620], [382, 620], [382, 648], [360, 648]],
        kind="dynamic",
        system_index=2,
        measure_index=7,
        placement="below",
        injected=False,
        measure_anchor_confidence=0.70,
        musical_direction_probability=0.99,
        distance_staff_spaces=0.1,
        backend="rapid-no-lines+rapid-staff",
    )

    issues = build_text_review_issues([page], {1: [mark]}, {1: 0}, tmp_path / "result")

    assert issues == []


def test_source_geometry_verified_dynamic_does_not_require_manual_review(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (1200, 800), "white").save(image_path)
    page = PageInfo(1, "page.png", str(image_path), normalized_path=str(image_path))
    mark = OcrMark(
        raw_text="f",
        text="f",
        score=0.985,
        box=[[370, 670], [386, 670], [386, 690], [370, 690]],
        kind="dynamic",
        system_index=3,
        measure_index=8,
        placement="above",
        injected=True,
        measure_anchor_confidence=0.40,
        musical_direction_probability=0.999,
        distance_staff_spaces=3.75,
        backend="source-dynamic-geometry-v2",
    )

    issues = build_text_review_issues([page], {1: [mark]}, {1: 0}, tmp_path / "result")

    assert issues == []


def test_build_measure_consensus_review_crop(tmp_path: Path) -> None:
    import json
    from scorescan.review import build_consensus_review_issues

    image_path = tmp_path / "page.png"
    image = Image.new("RGB", (1200, 700), "white")
    draw = ImageDraw.Draw(image)
    for line in range(5):
        draw.line((80, 250 + line * 14, 1120, 250 + line * 14), fill="black", width=2)
    for x in (80, 340, 600, 860, 1120):
        draw.line((x, 250, x, 306), fill="black", width=3)
    image.save(image_path)
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(json.dumps({
        "width": 1200,
        "height": 700,
        "confidence": 0.95,
        "systems": [{
            "index": 1,
            "line_y": [250, 264, 278, 292, 306],
            "top": 190,
            "bottom": 370,
            "left": 80,
            "right": 1120,
            "spacing": 14,
            "barlines": [80, 340, 600, 860, 1120],
            "measure_count": 4,
        }],
    }), encoding="utf-8")
    page = PageInfo(
        1, "page.png", str(image_path), normalized_path=str(image_path),
        layout_path=str(layout_path), consensus_unresolved=[2],
    )
    issues = build_consensus_review_issues([page], {1: 0}, tmp_path / "result")
    # Candidate disagreement retained the conservative template and therefore did
    # not alter the score. It belongs in diagnostics, not the user's review queue.
    assert issues == []


def test_build_measure_review_for_low_ensemble_probability(tmp_path: Path) -> None:
    import json
    from scorescan.review import build_consensus_review_issues

    image_path = tmp_path / "page.png"
    image = Image.new("RGB", (1200, 700), "white")
    draw = ImageDraw.Draw(image)
    for line in range(5):
        draw.line((80, 250 + line * 14, 1120, 250 + line * 14), fill="black", width=2)
    for x in (80, 340, 600, 860, 1120):
        draw.line((x, 250, x, 306), fill="black", width=3)
    image.save(image_path)
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(json.dumps({
        "width": 1200,
        "height": 700,
        "confidence": 0.95,
        "systems": [{
            "index": 1,
            "line_y": [250, 264, 278, 292, 306],
            "top": 190,
            "bottom": 370,
            "left": 80,
            "right": 1120,
            "spacing": 14,
            "barlines": [80, 340, 600, 860, 1120],
            "measure_count": 4,
        }],
    }), encoding="utf-8")
    consensus_path = tmp_path / "consensus.json"
    consensus_path.write_text(json.dumps({
        "votes": [{
            "measure_index": 3,
            "decision": "replace_semantic_consensus",
            "semantic_confidence": 0.61,
            "selected_ensemble_probability": 0.18,
        }]
    }), encoding="utf-8")
    page = PageInfo(
        1, "page.png", str(image_path), normalized_path=str(image_path),
        layout_path=str(layout_path), consensus_report_path=str(consensus_path),
    )
    issues = build_consensus_review_issues([page], {1: 0}, tmp_path / "result")
    assert len(issues) == 1
    assert issues[0].global_measure_number == 3
    assert "符号替换条件未满足" in issues[0].message
    assert "程序" not in issues[0].message


def test_semantically_equivalent_strict_consensus_does_not_create_false_review(tmp_path: Path) -> None:
    import json
    from scorescan.review import build_consensus_review_issues

    image_path = tmp_path / "page.png"
    Image.new("RGB", (1200, 700), "white").save(image_path)
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(json.dumps({
        "width": 1200,
        "height": 700,
        "confidence": 0.95,
        "systems": [{
            "index": 1,
            "line_y": [250, 264, 278, 292, 306],
            "top": 190,
            "bottom": 370,
            "left": 80,
            "right": 1120,
            "spacing": 14,
            "barlines": [80, 340, 600, 860, 1120],
            "measure_count": 4,
        }],
    }), encoding="utf-8")
    consensus_path = tmp_path / "consensus.json"
    consensus_path.write_text(json.dumps({
        "votes": [{
            "measure_index": 2,
            "decision": "retain_semantic_equivalent",
            "strict_majority": True,
            "semantic_confidence": 0.9966,
            "semantic_support_ratio": 1.0,
            "missing_candidates": 0,
            "selected_ensemble_probability": 0.91,
            "selection_risk_applicable": True,
            "selection_risk_accepted": False,
            "selected_selection_risk_probability": 0.626,
        }]
    }), encoding="utf-8")
    page = PageInfo(
        1, "page.png", str(image_path), normalized_path=str(image_path),
        layout_path=str(layout_path), consensus_report_path=str(consensus_path),
    )

    issues = build_consensus_review_issues([page], {1: 0}, tmp_path / "result")

    assert issues == []
