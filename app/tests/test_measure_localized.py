from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from lxml import etree

from scorescan.measure_localized import (
    MeasureLocalizedVariantResult,
    candidate_applies_to_boundary,
    candidate_applies_to_measure,
    choose_measure_localized_variant,
    create_measure_crop,
    create_measure_crop_variants,
    eligible_measure_indices,
    measure_localized_content_signature,
    measure_localized_semantic_signature,
    measure_localized_target,
    measure_localized_variant,
    splice_measure_candidate,
    validate_measure_localized_context,
    validate_measure_localized_xml,
)
from scorescan.musicxml import MUSICXML_DOCTYPE
from scorescan.variant_family import variant_family
from scorescan.visual_evidence import VisualMeasureEvidence


def _write_score(path: Path, pitches: list[str], *, local_attributes: bool = False) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    for number, step in enumerate(pitches, start=1):
        measure = etree.SubElement(part, "measure", number=str(number))
        if number == 1 or local_attributes:
            attributes = etree.SubElement(measure, "attributes")
            etree.SubElement(attributes, "divisions").text = "1"
            time = etree.SubElement(attributes, "time")
            etree.SubElement(time, "beats").text = "4"
            etree.SubElement(time, "beat-type").text = "4"
            if local_attributes:
                key = etree.SubElement(attributes, "key")
                etree.SubElement(key, "fifths").text = "7"
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
        doctype=MUSICXML_DOCTYPE,
    )


def _set_leading_notation_context(
    path: Path,
    *,
    clef: tuple[str, int] | None = None,
    fifths: int | None = None,
    time: tuple[int, int] | None = None,
    transpose: tuple[int, int, int, bool] | None = None,
    after_note: bool = False,
) -> None:
    tree = etree.parse(str(path))
    measure = tree.find("./part/measure")
    assert measure is not None
    attributes = etree.Element("attributes")
    if clef is not None:
        clef_node = etree.SubElement(attributes, "clef")
        etree.SubElement(clef_node, "sign").text = clef[0]
        etree.SubElement(clef_node, "line").text = str(clef[1])
    if fifths is not None:
        key = etree.SubElement(attributes, "key")
        etree.SubElement(key, "fifths").text = str(fifths)
    if time is not None:
        time_node = etree.SubElement(attributes, "time")
        etree.SubElement(time_node, "beats").text = str(time[0])
        etree.SubElement(time_node, "beat-type").text = str(time[1])
    if transpose is not None:
        diatonic, chromatic, octave_change, doubled = transpose
        node = etree.SubElement(attributes, "transpose")
        etree.SubElement(node, "diatonic").text = str(diatonic)
        etree.SubElement(node, "chromatic").text = str(chromatic)
        etree.SubElement(node, "octave-change").text = str(octave_change)
        if doubled:
            etree.SubElement(node, "double")
    if after_note:
        note = measure.find("note")
        assert note is not None
        measure.insert(measure.index(note) + 1, attributes)
    else:
        existing = measure.find("attributes")
        if existing is None:
            measure.insert(0, attributes)
        else:
            for child in list(attributes):
                existing.append(child)
    tree.write(str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE)


def _write_single_measure_with_divisions(
    path: Path,
    *,
    divisions: int,
    duration: int,
    note_type: str = "quarter",
) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = str(divisions)
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"
    note = etree.SubElement(measure, "note")
    pitch = etree.SubElement(note, "pitch")
    etree.SubElement(pitch, "step").text = "E"
    etree.SubElement(pitch, "octave").text = "4"
    etree.SubElement(note, "duration").text = str(duration)
    etree.SubElement(note, "voice").text = "1"
    etree.SubElement(note, "type").text = note_type
    direction = etree.SubElement(measure, "direction")
    etree.SubElement(direction, "offset").text = str(duration)
    direction_type = etree.SubElement(direction, "direction-type")
    etree.SubElement(direction_type, "words").text = "dolce"
    etree.ElementTree(root).write(
        str(path),
        encoding="UTF-8",
        xml_declaration=True,
        doctype=MUSICXML_DOCTYPE,
    )


def _evidence() -> VisualMeasureEvidence:
    return VisualMeasureEvidence(
        page_index=1,
        system_index=1,
        measure_index=2,
        bbox=(40, 30, 150, 90),
        spacing=10.0,
        ink_density=0.1,
        nonstaff_ink_density=0.05,
        component_density=0.1,
        notehead_proxy=1.0,
        open_notehead_proxy=1.0,
        stem_proxy=1.0,
        beam_proxy=0.0,
        onset_proxy=1.0,
        compact_mark_proxy=0.0,
        accidental_proxy=0.0,
        above_ink_density=0.0,
        below_ink_density=0.0,
        x_ink_profile=(0.0,) * 8,
        staff_ink_profile=(0.0,) * 9,
    )




def _variant_result(
    name: str,
    signature: str | None,
    *,
    valid: bool = True,
) -> MeasureLocalizedVariantResult:
    return MeasureLocalizedVariantResult(
        name=name,
        image_path=f"{name}.png",
        xml_path=f"{name}.musicxml" if valid else None,
        return_code=0 if valid else 65,
        elapsed_seconds=0.01,
        valid=valid,
        observed_measure_count=1 if valid else 0,
        note_count=1 if valid else 0,
        local_rhythm_issue_count=0,
        content_signature=signature if valid else None,
        error=None if valid else "failed",
    )


def test_measure_crop_variants_are_bounded_and_related(tmp_path: Path) -> None:
    image = np.full((120, 200), 255, dtype=np.uint8)
    cv2.ellipse(image, (95, 60), (10, 7), 0, 0, 360, 0, -1)
    source = tmp_path / "page.png"
    assert cv2.imwrite(str(source), image)
    crop = create_measure_crop(source, _evidence(), tmp_path / "crops")
    variants = create_measure_crop_variants(crop, tmp_path / "crops")
    assert tuple(item.name for item in variants) == ("primary", "flat", "otsu")
    assert len({item.sha256 for item in variants}) >= 2
    assert all(Path(item.image_path).exists() for item in variants)
    assert sum(item.pixel_count for item in variants) == 3 * variants[0].pixel_count
    assert (tmp_path / "crops" / "measure_0002_variants.json").exists()


def test_sparse_measure_candidate_never_observes_cross_measure_boundary() -> None:
    assert candidate_applies_to_measure("measure_localized:2", 2)
    assert not candidate_applies_to_measure("measure_localized:2", 1)
    assert candidate_applies_to_boundary("primary", 1, 2)
    assert candidate_applies_to_boundary("system_localized", 4, 5)
    assert not candidate_applies_to_boundary("measure_localized:1", 1, 2)
    assert not candidate_applies_to_boundary("measure_localized:2", 1, 2)
    assert not candidate_applies_to_boundary("measure_localized:2", 2, 3)


def test_measure_localized_content_signature_ignores_local_attributes(tmp_path: Path) -> None:
    plain = tmp_path / "plain.musicxml"
    attributed = tmp_path / "attributed.musicxml"
    div1 = tmp_path / "div1.musicxml"
    div4 = tmp_path / "div4.musicxml"
    _write_score(plain, ["E"], local_attributes=False)
    _write_score(attributed, ["E"], local_attributes=True)
    _write_single_measure_with_divisions(div1, divisions=1, duration=1)
    _write_single_measure_with_divisions(div4, divisions=4, duration=4)
    assert measure_localized_content_signature(plain) == measure_localized_content_signature(attributed)
    assert measure_localized_content_signature(div1) == measure_localized_content_signature(div4)


def _set_note_detail(
    path: Path,
    tag: str,
    text: str,
    *,
    attributes: dict[str, str] | None = None,
    under_notations: bool = False,
) -> None:
    tree = etree.parse(str(path))
    note = tree.find("./part/measure/note")
    assert note is not None
    parent = note
    if under_notations:
        parent = note.find("notations")
        if parent is None:
            parent = etree.SubElement(note, "notations")
    node = etree.SubElement(parent, tag, **(attributes or {}))
    node.text = text
    tree.write(str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE)


def test_measure_localized_exact_signature_covers_spliced_notation(tmp_path: Path) -> None:
    begin = tmp_path / "beam_begin.musicxml"
    end = tmp_path / "beam_end.musicxml"
    _write_single_measure_with_divisions(begin, divisions=2, duration=1, note_type="eighth")
    _write_single_measure_with_divisions(end, divisions=2, duration=1, note_type="eighth")
    _set_note_detail(begin, "beam", "begin", attributes={"number": "1"})
    _set_note_detail(end, "beam", "end", attributes={"number": "1"})

    # The legacy Score IR cannot see beam topology, so it remains diagnostic only.
    assert measure_localized_semantic_signature(begin) == measure_localized_semantic_signature(end)
    # The permission signature hashes exactly what would be spliced and must disagree.
    assert measure_localized_content_signature(begin) != measure_localized_content_signature(end)


def test_measure_localized_exact_signature_covers_unmodelled_notations(tmp_path: Path) -> None:
    fermata = tmp_path / "fermata.musicxml"
    technical = tmp_path / "technical.musicxml"
    _write_single_measure_with_divisions(fermata, divisions=1, duration=1)
    _write_single_measure_with_divisions(technical, divisions=1, duration=1)
    _set_note_detail(fermata, "fermata", "normal", under_notations=True)
    tree = etree.parse(str(technical))
    note = tree.find("./part/measure/note")
    assert note is not None
    notations = etree.SubElement(note, "notations")
    tech = etree.SubElement(notations, "technical")
    etree.SubElement(tech, "fingering").text = "2"
    tree.write(str(technical), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE)

    assert measure_localized_semantic_signature(fermata) == measure_localized_semantic_signature(technical)
    assert measure_localized_content_signature(fermata) != measure_localized_content_signature(technical)


def test_measure_localized_exact_signature_ignores_layout_coordinates(tmp_path: Path) -> None:
    left = tmp_path / "left.musicxml"
    right = tmp_path / "right.musicxml"
    _write_single_measure_with_divisions(left, divisions=1, duration=1)
    _write_single_measure_with_divisions(right, divisions=1, duration=1)
    for path, x, color in ((left, "17", "#000000"), (right, "93", "#123456")):
        tree = etree.parse(str(path))
        note = tree.find("./part/measure/note")
        assert note is not None
        note.set("default-x", x)
        note.set("relative-y", x)
        note.set("color", color)
        tree.write(str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE)

    assert measure_localized_content_signature(left) == measure_localized_content_signature(right)


def test_choose_measure_localized_variant_requires_strict_internal_majority() -> None:
    agreed = (
        _variant_result("primary", "same"),
        _variant_result("flat", "same"),
        _variant_result("otsu", "other"),
    )
    selected, support, signature, error = choose_measure_localized_variant(agreed)
    assert selected is agreed[0]
    assert (support, signature, error) == (2, "same", None)

    split = (
        _variant_result("primary", "a"),
        _variant_result("flat", "b"),
        _variant_result("otsu", "c"),
    )
    selected, support, _signature, error = choose_measure_localized_variant(split)
    assert selected is None
    assert support == 1
    assert error == "internal_exact_support_low"

    insufficient = (
        _variant_result("primary", "a"),
        _variant_result("flat", None, valid=False),
        _variant_result("otsu", None, valid=False),
    )
    selected, support, signature, error = choose_measure_localized_variant(insufficient)
    assert selected is None
    assert (support, signature, error) == (0, None, "insufficient_valid_internal_variants")


def test_measure_localized_variant_scope_and_family() -> None:
    variant = measure_localized_variant(3)
    assert variant == "measure_localized:3"
    assert measure_localized_target(variant) == 3
    assert candidate_applies_to_measure(variant, 3)
    assert not candidate_applies_to_measure(variant, 2)
    assert candidate_applies_to_measure("primary", 99)
    assert variant_family(variant) == "measure_localization:3"
    assert variant_family(measure_localized_variant(4)) != variant_family(variant)


def test_eligible_measure_indices_requires_two_existing_families_and_is_bounded(monkeypatch) -> None:
    votes = (
        SimpleNamespace(measure_index=1, exact_family_support=1, semantic_family_support=1),
        SimpleNamespace(measure_index=2, exact_family_support=2, semantic_family_support=1),
        SimpleNamespace(measure_index=3, exact_family_support=1, semantic_family_support=3),
        SimpleNamespace(measure_index=4, exact_family_support=2, semantic_family_support=2),
        SimpleNamespace(measure_index=5, exact_family_support=4, semantic_family_support=4),
    )
    consensus = SimpleNamespace(unresolved_measure_indices=(1, 2, 3, 4), votes=votes)
    # The already-resolved fifth measure must never be selected even with higher support.
    assert eligible_measure_indices(consensus) == (2, 3, 4)


def test_create_measure_crop_is_bounded_and_atomic(tmp_path: Path) -> None:
    image = np.full((120, 200), 255, dtype=np.uint8)
    cv2.rectangle(image, (55, 45), (135, 75), 0, 2)
    source = tmp_path / "page.png"
    assert cv2.imwrite(str(source), image)
    crop = create_measure_crop(source, _evidence(), tmp_path / "crops")
    crop_image = cv2.imread(crop.image_path, cv2.IMREAD_GRAYSCALE)
    assert crop_image is not None
    assert crop.measure_index == 2
    assert crop.source_bbox[0] < 40
    assert crop.source_bbox[2] > 150
    assert crop.padded_shape == (crop_image.shape[1], crop_image.shape[0])
    assert (tmp_path / "crops" / "measure_0002_crop.json").exists()
    assert not list((tmp_path / "crops").glob("*.tmp"))


def test_validate_measure_localized_xml_requires_one_nonempty_measure(tmp_path: Path) -> None:
    one = tmp_path / "one.musicxml"
    two = tmp_path / "two.musicxml"
    _write_score(one, ["C"])
    _write_score(two, ["C", "D"])
    valid, measure_count, note_count, rhythm_issues, error = validate_measure_localized_xml(one)
    assert valid
    assert (measure_count, note_count, rhythm_issues, error) == (1, 1, 0, None)
    valid, measure_count, note_count, rhythm_issues, error = validate_measure_localized_xml(two)
    assert not valid
    assert measure_count == 2
    assert note_count == 2
    assert "exactly one measure" in str(error)


def test_splice_replaces_only_target_content_and_preserves_template_attributes(tmp_path: Path) -> None:
    template = tmp_path / "template.musicxml"
    local = tmp_path / "local.musicxml"
    output = tmp_path / "candidate.musicxml"
    _write_score(template, ["C", "F"])
    _write_score(local, ["E"], local_attributes=False)
    splice_measure_candidate(template, local, 2, output)
    tree = etree.parse(str(output))
    assert tree.findtext("./part/measure[1]/note/pitch/step") == "C"
    assert tree.findtext("./part/measure[2]/note/pitch/step") == "E"
    # The crop contributes performed content only; page attributes remain authoritative.
    assert tree.find("./part/measure[2]/attributes/key") is None
    assert tree.findtext("./part/measure[1]/attributes/time/beats") == "4"



def test_measure_localized_context_accepts_matching_nondefault_context(tmp_path: Path) -> None:
    template = tmp_path / "template.musicxml"
    local = tmp_path / "local.musicxml"
    _write_score(template, ["C"])
    _write_score(local, ["E"])
    context = {
        "clef": ("F", 4),
        "fifths": -2,
        "time": (3, 4),
        "transpose": (-1, -2, 0, False),
    }
    _set_leading_notation_context(template, **context)
    _set_leading_notation_context(local, **context)
    valid, error = validate_measure_localized_context(local, template, 1)
    assert valid
    assert error is None


def test_splice_rejects_conflicting_local_notation_context(tmp_path: Path) -> None:
    template = tmp_path / "template.musicxml"
    local = tmp_path / "local.musicxml"
    output = tmp_path / "candidate.musicxml"
    _write_score(template, ["C"])
    _write_score(local, ["E"], local_attributes=True)
    valid, error = validate_measure_localized_context(local, template, 1)
    assert not valid
    assert "key context" in str(error)
    try:
        splice_measure_candidate(template, local, 1, output)
    except ValueError as exc:
        assert "key context" in str(exc)
    else:
        raise AssertionError("conflicting crop notation context must fail closed")
    assert not output.exists()


def test_measure_localized_context_requires_explicit_nondefault_template_context(
    tmp_path: Path,
) -> None:
    template = tmp_path / "template.musicxml"
    local = tmp_path / "local.musicxml"
    _write_score(template, ["C"])
    _write_score(local, ["E"])
    _set_leading_notation_context(template, clef=("F", 4), time=(3, 4))
    _set_leading_notation_context(local, time=(3, 4))
    valid, error = validate_measure_localized_context(local, template, 1)
    assert not valid
    assert "clef context is missing" in str(error)


def test_measure_localized_context_rejects_each_conflicting_context_kind(
    tmp_path: Path,
) -> None:
    conflicts = (
        ("clef", {"clef": ("F", 4)}),
        ("key", {"fifths": 3}),
        ("time", {"time": (6, 8)}),
        ("transpose", {"transpose": (1, 2, 0, False)}),
    )
    for name, values in conflicts:
        template = tmp_path / f"template_{name}.musicxml"
        local = tmp_path / f"local_{name}.musicxml"
        _write_score(template, ["C"])
        _write_score(local, ["E"])
        _set_leading_notation_context(local, **values)
        valid, error = validate_measure_localized_context(local, template, 1)
        assert not valid
        assert f"local {name} context" in str(error)


def test_measure_localized_context_allows_explicit_default_against_implicit_default(
    tmp_path: Path,
) -> None:
    template = tmp_path / "template.musicxml"
    local = tmp_path / "local.musicxml"
    _write_score(template, ["C"])
    _write_score(local, ["E"])
    _set_leading_notation_context(
        local,
        clef=("G", 2),
        fifths=0,
        transpose=(0, 0, 0, False),
    )
    valid, error = validate_measure_localized_context(local, template, 1)
    assert valid
    assert error is None


def test_measure_localized_context_rejects_change_after_performed_content(
    tmp_path: Path,
) -> None:
    template = tmp_path / "template.musicxml"
    local = tmp_path / "local.musicxml"
    _write_score(template, ["C"])
    _write_score(local, ["E"])
    _set_leading_notation_context(local, clef=("G", 2), after_note=True)
    valid, error = validate_measure_localized_context(local, template, 1)
    assert not valid
    assert "after performed content" in str(error)


class _MeasureRescueRunner:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def run_page(self, image_path: Path, _cancel_event: object = None):
        from scorescan.omr import EngineResult

        self.calls.append(image_path.name)
        xml = image_path.with_suffix(".musicxml")
        if image_path.name.startswith("measure_0002"):
            _write_score(xml, ["E"])
        elif image_path.stem == "primary":
            _write_score(xml, ["C", "F"])
        else:
            _write_score(xml, ["C", "E"])
        return EngineResult(0, xml, 0.01)


def test_recognition_measure_rescue_requires_and_supplies_third_family(
    tmp_path: Path, monkeypatch
) -> None:
    from scorescan.layout import PageLayout, StaffSystem
    from scorescan.models import PageInfo
    from scorescan.recognition import RecognitionEnsemble
    from scorescan.util import read_json
    import scorescan.recognition as recognition

    image = np.full((140, 240), 255, dtype=np.uint8)
    page_path = tmp_path / "page.png"
    assert cv2.imwrite(str(page_path), image)
    variants = []
    for name in ("primary", "flat", "otsu"):
        path = tmp_path / f"{name}.png"
        assert cv2.imwrite(str(path), image)
        variants.append((name, path))
    evidence = (
        VisualMeasureEvidence(**{**_evidence().__dict__, "measure_index": 1, "bbox": (10, 25, 105, 100)}),
        VisualMeasureEvidence(**{**_evidence().__dict__, "measure_index": 2, "bbox": (115, 25, 225, 100)}),
    )
    system = StaffSystem(
        index=1,
        line_y=[45, 55, 65, 75, 85],
        top=20,
        bottom=105,
        left=5,
        right=235,
        spacing=10.0,
        barlines=[],
        measure_count=2,
    )
    layout = PageLayout(width=240, height=140, systems=[system], confidence=0.95)
    page = PageInfo(index=1, source_name="page.png", image_path=str(page_path))
    page.normalized_path = str(page_path)
    page.estimated_measure_count = 2
    page.quality_score = 95.0

    monkeypatch.setattr(recognition, "generate_omr_variants", lambda *_a, **_k: variants)
    monkeypatch.setattr(recognition, "extract_page_measure_evidence", lambda *_a, **_k: evidence)
    monkeypatch.setattr(
        recognition,
        "should_run_localized_recognition",
        lambda *_a, **_k: (False, "not_needed"),
    )
    monkeypatch.setattr(
        recognition,
        "choose_candidate",
        lambda items: next((item for item in items if item.variant == "primary"), None),
    )
    runner = _MeasureRescueRunner()
    work = tmp_path / "work"
    result = RecognitionEnsemble(runner, lambda _message: None, work).run_page(page, layout)

    assert result.selected is not None
    assert result.consensus is not None
    assert result.consensus.unresolved_measure_indices == ()
    final = etree.parse(str(result.selected.xml_path))
    assert final.findtext("./part/measure[1]/note/pitch/step") == "C"
    assert final.findtext("./part/measure[2]/note/pitch/step") == "E"
    assert any(name.startswith("measure_0002") for name in runner.calls)
    assert runner.calls.count("primary.png") == 1
    assert runner.calls.count("flat.png") == 1
    assert runner.calls.count("otsu.png") == 1
    summary = read_json(work / "page_0001" / "measure_localized" / "measure_rescue_summary.json")
    assert summary["accepted"] is True
    assert summary["before_unresolved"] == [2]
    assert summary["after_unresolved"] == []
    detail = read_json(work / "page_0001" / "measure_localized" / "measure_0002_result.json")
    assert detail["internal_variant_count"] == 3
    assert detail["internal_valid_count"] == 3
    assert detail["winning_exact_support"] == 3

class _ContextMismatchMeasureRescueRunner(_MeasureRescueRunner):
    def run_page(self, image_path: Path, _cancel_event: object = None):
        from scorescan.omr import EngineResult

        self.calls.append(image_path.name)
        xml = image_path.with_suffix(".musicxml")
        if image_path.name.startswith("measure_0002"):
            _write_score(xml, ["E"], local_attributes=True)
        elif image_path.stem == "primary":
            _write_score(xml, ["C", "F"])
        else:
            _write_score(xml, ["C", "E"])
        return EngineResult(0, xml, 0.01)


def test_context_mismatch_invalidates_local_variants_before_internal_majority(
    tmp_path: Path, monkeypatch
) -> None:
    from scorescan.layout import PageLayout, StaffSystem
    from scorescan.models import PageInfo
    from scorescan.recognition import RecognitionEnsemble
    from scorescan.util import read_json
    import scorescan.recognition as recognition

    image = np.full((140, 240), 255, dtype=np.uint8)
    page_path = tmp_path / "page.png"
    assert cv2.imwrite(str(page_path), image)
    variants = []
    for name in ("primary", "flat", "otsu"):
        path = tmp_path / f"{name}.png"
        assert cv2.imwrite(str(path), image)
        variants.append((name, path))
    evidence = (
        VisualMeasureEvidence(**{**_evidence().__dict__, "measure_index": 1, "bbox": (10, 25, 105, 100)}),
        VisualMeasureEvidence(**{**_evidence().__dict__, "measure_index": 2, "bbox": (115, 25, 225, 100)}),
    )
    layout = PageLayout(
        width=240,
        height=140,
        systems=[StaffSystem(
            index=1, line_y=[45, 55, 65, 75, 85], top=20, bottom=105,
            left=5, right=235, spacing=10.0, barlines=[], measure_count=2,
        )],
        confidence=0.95,
    )
    page = PageInfo(index=1, source_name="page.png", image_path=str(page_path))
    page.normalized_path = str(page_path)
    page.estimated_measure_count = 2
    page.quality_score = 95.0

    monkeypatch.setattr(recognition, "generate_omr_variants", lambda *_a, **_k: variants)
    monkeypatch.setattr(recognition, "extract_page_measure_evidence", lambda *_a, **_k: evidence)
    monkeypatch.setattr(recognition, "should_run_localized_recognition", lambda *_a, **_k: (False, "not_needed"))
    monkeypatch.setattr(recognition, "choose_candidate", lambda items: next((item for item in items if item.variant == "primary"), None))
    work = tmp_path / "work"
    result = RecognitionEnsemble(
        _ContextMismatchMeasureRescueRunner(), lambda _message: None, work
    ).run_page(page, layout)

    assert result.selected is not None
    assert result.consensus is not None
    assert result.consensus.unresolved_measure_indices == (2,)
    detail = read_json(work / "page_0001" / "measure_localized" / "measure_0002_result.json")
    assert detail["valid"] is False
    assert detail["internal_valid_count"] == 0
    assert detail["winning_exact_support"] == 0
    assert all("key context" in str(item["error"]) for item in detail["internal_variants"])


class _NonImprovingMeasureRescueRunner(_MeasureRescueRunner):
    def run_page(self, image_path: Path, _cancel_event: object = None):
        from scorescan.omr import EngineResult

        self.calls.append(image_path.name)
        xml = image_path.with_suffix(".musicxml")
        if image_path.name.startswith("measure_0002"):
            _write_score(xml, ["D"])
        elif image_path.stem == "primary":
            _write_score(xml, ["C", "F"])
        else:
            _write_score(xml, ["C", "E"])
        return EngineResult(0, xml, 0.01)


def test_non_improving_measure_rescue_preserves_initial_consensus(
    tmp_path: Path, monkeypatch
) -> None:
    from scorescan.layout import PageLayout, StaffSystem
    from scorescan.models import PageInfo
    from scorescan.recognition import RecognitionEnsemble
    from scorescan.util import read_json
    import scorescan.recognition as recognition

    image = np.full((140, 240), 255, dtype=np.uint8)
    page_path = tmp_path / "page.png"
    assert cv2.imwrite(str(page_path), image)
    variants = []
    for name in ("primary", "flat", "otsu"):
        path = tmp_path / f"{name}.png"
        assert cv2.imwrite(str(path), image)
        variants.append((name, path))
    evidence = (
        VisualMeasureEvidence(**{**_evidence().__dict__, "measure_index": 1, "bbox": (10, 25, 105, 100)}),
        VisualMeasureEvidence(**{**_evidence().__dict__, "measure_index": 2, "bbox": (115, 25, 225, 100)}),
    )
    layout = PageLayout(
        width=240,
        height=140,
        systems=[StaffSystem(
            index=1, line_y=[45, 55, 65, 75, 85], top=20, bottom=105,
            left=5, right=235, spacing=10.0, barlines=[], measure_count=2,
        )],
        confidence=0.95,
    )
    page = PageInfo(index=1, source_name="page.png", image_path=str(page_path))
    page.normalized_path = str(page_path)
    page.estimated_measure_count = 2
    page.quality_score = 95.0

    monkeypatch.setattr(recognition, "generate_omr_variants", lambda *_a, **_k: variants)
    monkeypatch.setattr(recognition, "extract_page_measure_evidence", lambda *_a, **_k: evidence)
    monkeypatch.setattr(recognition, "should_run_localized_recognition", lambda *_a, **_k: (False, "not_needed"))
    monkeypatch.setattr(recognition, "choose_candidate", lambda items: next((item for item in items if item.variant == "primary"), None))
    work = tmp_path / "work"
    result = RecognitionEnsemble(
        _NonImprovingMeasureRescueRunner(), lambda _message: None, work
    ).run_page(page, layout)

    assert result.selected is not None
    assert result.consensus is not None
    assert result.consensus.unresolved_measure_indices == (2,)
    final = etree.parse(str(result.selected.xml_path))
    assert final.findtext("./part/measure[2]/note/pitch/step") == "F"
    summary = read_json(work / "page_0001" / "measure_localized" / "measure_rescue_summary.json")
    assert summary["accepted"] is False
    assert summary["before_unresolved"] == [2]
    assert summary["after_unresolved"] == [2]


class _AmbiguousMeasureRescueRunner(_MeasureRescueRunner):
    def run_page(self, image_path: Path, _cancel_event: object = None):
        from scorescan.omr import EngineResult

        self.calls.append(image_path.name)
        xml = image_path.with_suffix(".musicxml")
        if image_path.name.startswith("measure_0002_primary"):
            _write_score(xml, ["E"])
        elif image_path.name.startswith("measure_0002_flat"):
            _write_score(xml, ["D"])
        elif image_path.name.startswith("measure_0002_otsu"):
            _write_score(xml, ["G"])
        elif image_path.stem == "primary":
            _write_score(xml, ["C", "F"])
        else:
            _write_score(xml, ["C", "E"])
        return EngineResult(0, xml, 0.01)


def test_ambiguous_internal_measure_rescue_fails_closed(tmp_path: Path, monkeypatch) -> None:
    from scorescan.layout import PageLayout, StaffSystem
    from scorescan.models import PageInfo
    from scorescan.recognition import RecognitionEnsemble
    from scorescan.util import read_json
    import scorescan.recognition as recognition

    image = np.full((140, 240), 255, dtype=np.uint8)
    page_path = tmp_path / "page.png"
    assert cv2.imwrite(str(page_path), image)
    variants = []
    for name in ("primary", "flat", "otsu"):
        path = tmp_path / f"{name}.png"
        assert cv2.imwrite(str(path), image)
        variants.append((name, path))
    evidence = (
        VisualMeasureEvidence(**{**_evidence().__dict__, "measure_index": 1, "bbox": (10, 25, 105, 100)}),
        VisualMeasureEvidence(**{**_evidence().__dict__, "measure_index": 2, "bbox": (115, 25, 225, 100)}),
    )
    layout = PageLayout(
        width=240,
        height=140,
        systems=[StaffSystem(
            index=1, line_y=[45, 55, 65, 75, 85], top=20, bottom=105,
            left=5, right=235, spacing=10.0, barlines=[], measure_count=2,
        )],
        confidence=0.95,
    )
    page = PageInfo(index=1, source_name="page.png", image_path=str(page_path))
    page.normalized_path = str(page_path)
    page.estimated_measure_count = 2
    page.quality_score = 95.0

    monkeypatch.setattr(recognition, "generate_omr_variants", lambda *_a, **_k: variants)
    monkeypatch.setattr(recognition, "extract_page_measure_evidence", lambda *_a, **_k: evidence)
    monkeypatch.setattr(recognition, "should_run_localized_recognition", lambda *_a, **_k: (False, "not_needed"))
    monkeypatch.setattr(recognition, "choose_candidate", lambda items: next((item for item in items if item.variant == "primary"), None))
    work = tmp_path / "work"
    result = RecognitionEnsemble(
        _AmbiguousMeasureRescueRunner(), lambda _message: None, work
    ).run_page(page, layout)

    assert result.selected is not None
    assert result.consensus is not None
    assert result.consensus.unresolved_measure_indices == (2,)
    final = etree.parse(str(result.selected.xml_path))
    assert final.findtext("./part/measure[2]/note/pitch/step") == "F"
    detail = read_json(work / "page_0001" / "measure_localized" / "measure_0002_result.json")
    assert detail["valid"] is False
    assert detail["internal_valid_count"] == 3
    assert detail["winning_exact_support"] == 1
    assert detail["error"] == "internal_exact_support_low"


def test_splice_rescales_local_divisions_exactly(tmp_path: Path) -> None:
    template = tmp_path / "template_div4.musicxml"
    local = tmp_path / "local_div1.musicxml"
    output = tmp_path / "scaled.musicxml"
    _write_single_measure_with_divisions(template, divisions=4, duration=4)
    _write_single_measure_with_divisions(local, divisions=1, duration=1)
    local_tree = etree.parse(str(local))
    local_measure = local_tree.find("./part/measure")
    assert local_measure is not None
    backup = etree.SubElement(local_measure, "backup")
    etree.SubElement(backup, "duration").text = "1"
    forward = etree.SubElement(local_measure, "forward")
    etree.SubElement(forward, "duration").text = "1"
    local_tree.write(str(local), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE)
    splice_measure_candidate(template, local, 1, output)
    tree = etree.parse(str(output))
    assert tree.findtext("./part/measure/note/duration") == "4"
    assert tree.findtext("./part/measure/direction/offset") == "4"
    assert tree.findtext("./part/measure/backup/duration") == "4"
    assert tree.findtext("./part/measure/forward/duration") == "4"
    assert tree.findtext("./part/measure/attributes/divisions") == "4"


def test_splice_rejects_nonintegral_division_conversion(tmp_path: Path) -> None:
    template = tmp_path / "template_div3.musicxml"
    local = tmp_path / "local_div2.musicxml"
    output = tmp_path / "nonintegral.musicxml"
    _write_single_measure_with_divisions(template, divisions=3, duration=3)
    _write_single_measure_with_divisions(local, divisions=2, duration=1)
    try:
        splice_measure_candidate(template, local, 1, output)
    except ValueError as exc:
        assert "cannot be represented exactly" in str(exc)
    else:
        raise AssertionError("non-integral duration conversion should fail closed")
    assert not output.exists()


class _OneVariantCrashesRunner(_MeasureRescueRunner):
    def run_page(self, image_path: Path, _cancel_event: object = None):
        if image_path.name.startswith("measure_0002_flat"):
            raise RuntimeError("simulated local treatment failure")
        return super().run_page(image_path, _cancel_event)


def test_one_internal_variant_failure_is_isolated(tmp_path: Path, monkeypatch) -> None:
    from scorescan.layout import PageLayout, StaffSystem
    from scorescan.models import PageInfo
    from scorescan.recognition import RecognitionEnsemble
    from scorescan.util import read_json
    import scorescan.recognition as recognition

    image = np.full((140, 240), 255, dtype=np.uint8)
    page_path = tmp_path / "page.png"
    assert cv2.imwrite(str(page_path), image)
    variants = []
    for name in ("primary", "flat", "otsu"):
        path = tmp_path / f"{name}.png"
        assert cv2.imwrite(str(path), image)
        variants.append((name, path))
    evidence = (
        VisualMeasureEvidence(**{**_evidence().__dict__, "measure_index": 1, "bbox": (10, 25, 105, 100)}),
        VisualMeasureEvidence(**{**_evidence().__dict__, "measure_index": 2, "bbox": (115, 25, 225, 100)}),
    )
    layout = PageLayout(
        width=240,
        height=140,
        systems=[StaffSystem(
            index=1, line_y=[45, 55, 65, 75, 85], top=20, bottom=105,
            left=5, right=235, spacing=10.0, barlines=[], measure_count=2,
        )],
        confidence=0.95,
    )
    page = PageInfo(index=1, source_name="page.png", image_path=str(page_path))
    page.normalized_path = str(page_path)
    page.estimated_measure_count = 2
    page.quality_score = 95.0

    monkeypatch.setattr(recognition, "generate_omr_variants", lambda *_a, **_k: variants)
    monkeypatch.setattr(recognition, "extract_page_measure_evidence", lambda *_a, **_k: evidence)
    monkeypatch.setattr(recognition, "should_run_localized_recognition", lambda *_a, **_k: (False, "not_needed"))
    monkeypatch.setattr(recognition, "choose_candidate", lambda items: next((item for item in items if item.variant == "primary"), None))
    work = tmp_path / "work"
    result = RecognitionEnsemble(
        _OneVariantCrashesRunner(), lambda _message: None, work
    ).run_page(page, layout)

    assert result.selected is not None
    assert result.consensus is not None
    assert result.consensus.unresolved_measure_indices == ()
    detail = read_json(work / "page_0001" / "measure_localized" / "measure_0002_result.json")
    assert detail["valid"] is True
    assert detail["internal_valid_count"] == 2
    assert detail["winning_exact_support"] == 2
    flat = next(item for item in detail["internal_variants"] if item["name"] == "flat")
    assert flat["valid"] is False
    assert "simulated local treatment failure" in flat["error"]


def test_splice_requires_explicit_local_divisions(tmp_path: Path) -> None:
    template = tmp_path / "template.musicxml"
    local = tmp_path / "local_without_divisions.musicxml"
    output = tmp_path / "output.musicxml"
    _write_single_measure_with_divisions(template, divisions=4, duration=4)
    _write_single_measure_with_divisions(local, divisions=1, duration=1)
    tree = etree.parse(str(local))
    divisions = tree.find("./part/measure/attributes/divisions")
    assert divisions is not None
    divisions.getparent().remove(divisions)
    tree.write(str(local), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE)
    try:
        splice_measure_candidate(template, local, 1, output)
    except ValueError as exc:
        assert "explicit divisions" in str(exc)
    else:
        raise AssertionError("local splice without explicit divisions should fail closed")
    assert not output.exists()


def test_exact_signature_rejects_mid_measure_divisions_change(tmp_path: Path) -> None:
    local = tmp_path / "local_divisions_change.musicxml"
    _write_single_measure_with_divisions(local, divisions=2, duration=2)
    tree = etree.parse(str(local))
    measure = tree.find("./part/measure")
    note = tree.find("./part/measure/note")
    assert measure is not None and note is not None
    later_attributes = etree.Element("attributes")
    etree.SubElement(later_attributes, "divisions").text = "4"
    measure.insert(measure.index(note) + 1, later_attributes)
    tree.write(str(local), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE)

    try:
        measure_localized_content_signature(local)
    except ValueError as exc:
        assert "changes divisions after performed content" in str(exc)
    else:
        raise AssertionError("mid-measure divisions changes must not form local exact support")


def test_splice_rejects_source_divisions_change_after_content(tmp_path: Path) -> None:
    template = tmp_path / "template.musicxml"
    local = tmp_path / "local.musicxml"
    output = tmp_path / "output.musicxml"
    _write_single_measure_with_divisions(template, divisions=4, duration=4)
    _write_single_measure_with_divisions(local, divisions=2, duration=2)
    tree = etree.parse(str(local))
    measure = tree.find("./part/measure")
    note = tree.find("./part/measure/note")
    assert measure is not None and note is not None
    later_attributes = etree.Element("attributes")
    etree.SubElement(later_attributes, "divisions").text = "4"
    measure.insert(measure.index(note) + 1, later_attributes)
    tree.write(str(local), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE)

    try:
        splice_measure_candidate(template, local, 1, output)
    except ValueError as exc:
        assert "changes divisions after performed content" in str(exc)
    else:
        raise AssertionError("source divisions changes must fail local splice")
    assert not output.exists()


def test_splice_rejects_template_mid_measure_attributes(tmp_path: Path) -> None:
    template = tmp_path / "template_mid_attribute.musicxml"
    local = tmp_path / "local.musicxml"
    output = tmp_path / "output.musicxml"
    _write_single_measure_with_divisions(template, divisions=4, duration=4)
    _write_single_measure_with_divisions(local, divisions=1, duration=1)
    tree = etree.parse(str(template))
    measure = tree.find("./part/measure")
    note = tree.find("./part/measure/note")
    assert measure is not None and note is not None
    later_attributes = etree.Element("attributes")
    clef = etree.SubElement(later_attributes, "clef")
    etree.SubElement(clef, "sign").text = "F"
    etree.SubElement(clef, "line").text = "4"
    measure.insert(measure.index(note) + 1, later_attributes)
    tree.write(str(template), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE)

    try:
        splice_measure_candidate(template, local, 1, output)
    except ValueError as exc:
        assert "mid-measure attributes" in str(exc)
    else:
        raise AssertionError("local rescue must not reorder a template attribute change")
    assert not output.exists()


class _NotationAmbiguousMeasureRescueRunner(_MeasureRescueRunner):
    def run_page(self, image_path: Path, _cancel_event: object = None):
        from scorescan.omr import EngineResult

        self.calls.append(image_path.name)
        xml = image_path.with_suffix(".musicxml")
        if image_path.name.startswith("measure_0002"):
            _write_score(xml, ["E"])
            if image_path.name.startswith("measure_0002_primary"):
                _set_note_detail(xml, "beam", "begin", attributes={"number": "1"})
            elif image_path.name.startswith("measure_0002_flat"):
                _set_note_detail(xml, "beam", "end", attributes={"number": "1"})
            else:
                _set_note_detail(xml, "stem", "up")
        elif image_path.stem == "primary":
            _write_score(xml, ["C", "F"])
        else:
            _write_score(xml, ["C", "E"])
        return EngineResult(0, xml, 0.01)


def test_semantically_equal_but_notationally_split_local_rescue_fails_closed(
    tmp_path: Path, monkeypatch
) -> None:
    from scorescan.layout import PageLayout, StaffSystem
    from scorescan.models import PageInfo
    from scorescan.recognition import RecognitionEnsemble
    from scorescan.util import read_json
    import scorescan.recognition as recognition

    image = np.full((140, 240), 255, dtype=np.uint8)
    page_path = tmp_path / "page.png"
    assert cv2.imwrite(str(page_path), image)
    variants = []
    for name in ("primary", "flat", "otsu"):
        path = tmp_path / f"{name}.png"
        assert cv2.imwrite(str(path), image)
        variants.append((name, path))
    evidence = (
        VisualMeasureEvidence(**{**_evidence().__dict__, "measure_index": 1, "bbox": (10, 25, 105, 100)}),
        VisualMeasureEvidence(**{**_evidence().__dict__, "measure_index": 2, "bbox": (115, 25, 225, 100)}),
    )
    layout = PageLayout(
        width=240,
        height=140,
        systems=[StaffSystem(
            index=1, line_y=[45, 55, 65, 75, 85], top=20, bottom=105,
            left=5, right=235, spacing=10.0, barlines=[], measure_count=2,
        )],
        confidence=0.95,
    )
    page = PageInfo(index=1, source_name="page.png", image_path=str(page_path))
    page.normalized_path = str(page_path)
    page.estimated_measure_count = 2
    page.quality_score = 95.0

    monkeypatch.setattr(recognition, "generate_omr_variants", lambda *_a, **_k: variants)
    monkeypatch.setattr(recognition, "extract_page_measure_evidence", lambda *_a, **_k: evidence)
    monkeypatch.setattr(recognition, "should_run_localized_recognition", lambda *_a, **_k: (False, "not_needed"))
    monkeypatch.setattr(recognition, "choose_candidate", lambda items: next((item for item in items if item.variant == "primary"), None))
    work = tmp_path / "work"
    result = RecognitionEnsemble(
        _NotationAmbiguousMeasureRescueRunner(), lambda _message: None, work
    ).run_page(page, layout)

    assert result.selected is not None
    assert result.consensus is not None
    assert result.consensus.unresolved_measure_indices == (2,)
    detail = read_json(work / "page_0001" / "measure_localized" / "measure_0002_result.json")
    assert detail["valid"] is False
    assert detail["winning_exact_support"] == 1
    strict = {item["content_signature"] for item in detail["internal_variants"]}
    semantic = {item["semantic_signature"] for item in detail["internal_variants"]}
    assert len(strict) == 3
    assert len(semantic) == 1
