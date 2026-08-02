from __future__ import annotations

import copy
import io
import math
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

from lxml import etree

from .config import APP_NAME, APP_VERSION
from .layout import PageLayout
from .score_ir import audit_production_score, audit_score, score_from_tree
from .util import atomic_write_bytes

MUSICXML_DOCTYPE = (
    '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
    '"http://www.musicxml.org/dtds/partwise.dtd">'
)


@dataclass(frozen=True)
class PageDocument:
    tree: etree._ElementTree
    width: int
    height: int
    layout: PageLayout | None = None


def _ensure_print(measure: etree._Element) -> etree._Element:
    node = measure.find("print")
    if node is None:
        node = etree.Element("print")
        insert_at = 1 if len(measure) and measure[0].tag == "attributes" else 0
        measure.insert(insert_at, node)
    return node


def _set_text(parent: etree._Element, tag: str, value: str) -> etree._Element:
    node = parent.find(tag)
    if node is None:
        node = etree.SubElement(parent, tag)
    node.text = value
    return node


def _page_dimensions(width_px: int, height_px: int) -> tuple[float, float]:
    page_width = 1680.0
    ratio = height_px / max(width_px, 1)
    page_height = max(1800.0, min(3400.0, page_width * ratio))
    return page_width, page_height


def _set_page_layout(print_node: etree._Element, width_px: int, height_px: int) -> None:
    # MuseScore interprets these values in tenths. Preserve source aspect ratio while
    # retaining a stable readable staff size.
    page_width, page_height = _page_dimensions(width_px, height_px)
    layout = print_node.find("page-layout")
    if layout is None:
        layout = etree.SubElement(print_node, "page-layout")
    _set_text(layout, "page-height", f"{page_height:.1f}")
    _set_text(layout, "page-width", f"{page_width:.1f}")
    margins = layout.find("page-margins")
    if margins is None:
        margins = etree.SubElement(layout, "page-margins", type="both")
    for tag, value in (
        ("left-margin", "85"), ("right-margin", "85"),
        ("top-margin", "90"), ("bottom-margin", "90"),
    ):
        _set_text(margins, tag, value)


def _normalize_default_page_layout(
    root: etree._Element,
    width_px: int,
    height_px: int,
) -> None:
    defaults = root.find("defaults")
    if defaults is None:
        return
    layout = defaults.find("page-layout")
    if layout is None:
        layout = etree.SubElement(defaults, "page-layout")
    # MusicXML requires scaling/concert-score before page-layout. homr can put
    # page-layout first, so normalize child order before importers read it.
    defaults.remove(layout)
    insert_at = 0
    for child in defaults:
        if child.tag in {"scaling", "concert-score"}:
            insert_at = defaults.index(child) + 1
    defaults.insert(insert_at, layout)

    try:
        old_width = float(layout.findtext("page-width") or 0)
        old_height = float(layout.findtext("page-height") or 0)
    except ValueError:
        old_width = 0.0
        old_height = 0.0
    page_width, page_height = _page_dimensions(width_px, height_px)
    _set_text(layout, "page-height", f"{page_height:.1f}")
    _set_text(layout, "page-width", f"{page_width:.1f}")
    margins = layout.find("page-margins")
    if margins is None:
        margins = etree.SubElement(layout, "page-margins", type="both")
    for tag, value in (
        ("left-margin", "85"),
        ("right-margin", "85"),
        ("top-margin", "90"),
        ("bottom-margin", "90"),
    ):
        _set_text(margins, tag, value)

    # OCR metadata credits were positioned in the recognizer's compact default
    # page coordinates (commonly 110 x 300).  Once the canonical page geometry is
    # expanded, preserve their normalized position instead of leaving the title and
    # composer clustered in the upper-left corner.
    if old_width > 0 and old_height > 0:
        x_scale = page_width / old_width
        y_scale = page_height / old_height
        for words in root.findall("./credit/credit-words"):
            for attribute, scale in (
                ("default-x", x_scale),
                ("default-y", y_scale),
            ):
                raw = words.get(attribute)
                if raw is None:
                    continue
                try:
                    words.set(attribute, f"{float(raw) * scale:.3f}")
                except ValueError:
                    continue


def _set_system_layout(print_node: etree._Element, system_index: int, system_count: int) -> None:
    layout = print_node.find("system-layout")
    if layout is None:
        layout = etree.SubElement(print_node, "system-layout")
    margins = layout.find("system-margins")
    if margins is None:
        margins = etree.SubElement(layout, "system-margins")
    _set_text(margins, "left-margin", "0")
    # Keep a visible safety margin inside the page's printable area. A zero
    # system margin makes imported scores appear to collide with MuseScore's
    # right page boundary and leaves no room for end-of-system symbols.
    _set_text(margins, "right-margin", "60")
    if system_index > 0:
        # A conservative default; source-specific vertical placement is not reliably
        # portable across notation engines, but the system boundaries remain stable.
        _set_text(layout, "system-distance", "145")
    elif system_count > 1:
        _set_text(layout, "top-system-distance", "95")


def _empty_page_document(page_number: int, message: str) -> etree._ElementTree:
    root = etree.Element("score-partwise", version="4.0")
    work = etree.SubElement(root, "work")
    etree.SubElement(work, "work-title").text = "Pigeon Score Scan conversion"
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "4"
    etree.SubElement(time, "beat-type").text = "4"
    clef = etree.SubElement(attributes, "clef")
    etree.SubElement(clef, "sign").text = "G"
    etree.SubElement(clef, "line").text = "2"
    direction = etree.SubElement(measure, "direction", placement="above")
    direction_type = etree.SubElement(direction, "direction-type")
    etree.SubElement(direction_type, "words").text = f"第 {page_number} 页：{message}"
    note = etree.SubElement(measure, "note")
    etree.SubElement(note, "rest", measure="yes")
    etree.SubElement(note, "duration").text = "4"
    etree.SubElement(note, "voice").text = "1"
    etree.SubElement(note, "type").text = "whole"
    return etree.ElementTree(root)


def parse_or_placeholder(path: Path | None, page_number: int, error: str | None) -> etree._ElementTree:
    if path is None or not path.exists():
        return _empty_page_document(page_number, error or "本页未生成识别结果")
    parser = etree.XMLParser(remove_blank_text=True, recover=False, resolve_entities=False, no_network=True)
    return etree.parse(str(path), parser)


def _distribute_counts(estimates: list[int], total: int) -> list[int]:
    if not estimates:
        return [total]
    estimates = [max(1, int(value)) for value in estimates]
    if sum(estimates) == total:
        return estimates
    weights = [value / sum(estimates) for value in estimates]
    raw = [weight * total for weight in weights]
    counts = [max(1, int(math.floor(value))) for value in raw]
    while sum(counts) > total:
        candidates = [i for i, value in enumerate(counts) if value > 1]
        if not candidates:
            break
        index = min(candidates, key=lambda i: raw[i] - counts[i])
        counts[index] -= 1
    while sum(counts) < total:
        index = max(range(len(counts)), key=lambda i: raw[i] - counts[i])
        counts[index] += 1
    return counts


def _apply_system_measure_widths(
    measures: list[etree._Element],
    start: int,
    count: int,
    system: object | None,
    target_width: float = 1450.0,
) -> None:
    # The page's printable width is 1510 tenths (1680 minus 85 on each side).
    # Reserve the same 60-tenth safety area encoded in system-margins instead
    # of forcing explicit measure widths to consume the entire printable area.
    if count <= 0 or start >= len(measures):
        return
    raw_widths: list[float] = []
    if system is not None:
        left = float(getattr(system, "left", 0))
        right = float(getattr(system, "right", left + count))
        barlines = [float(value) for value in getattr(system, "barlines", [])]
        boundaries = [left] + [value for value in barlines if left + 2 < value < right - 2] + [right]
        boundaries = sorted(set(boundaries))
        segments = [max(1.0, boundaries[i + 1] - boundaries[i]) for i in range(len(boundaries) - 1)]
        if len(segments) == count:
            raw_widths = segments
    if not raw_widths:
        raw_widths = [1.0] * count
    total = max(sum(raw_widths), 1.0)
    widths = [max(70.0, target_width * value / total) for value in raw_widths]
    # Re-normalize after minimum-width clamping.
    scale = target_width / max(sum(widths), 1.0)
    widths = [max(55.0, value * scale) for value in widths]
    for local_index, width in enumerate(widths):
        measure_index = start + local_index
        if measure_index < len(measures):
            measures[measure_index].set("width", f"{width:.1f}")


def apply_source_system_breaks(tree: etree._ElementTree, layout: PageLayout | None) -> dict[str, int | bool]:
    root = tree.getroot()
    parts = root.findall("part")
    if not parts:
        return {"systems": 0, "breaks_applied": False}
    reference_measures = parts[0].findall("measure")
    if not reference_measures:
        return {"systems": 0, "breaks_applied": False}
    existing = [
        index for index, measure in enumerate(reference_measures)
        if index > 0 and measure.find("print") is not None and measure.find("print").get("new-system") == "yes"
    ]
    existing_system_count = len(existing) + 1
    score_systems = list(getattr(layout, "score_systems", ()) or ()) if layout else []
    physical_systems = list(layout.systems) if layout else []
    # Existing OMR breaks are strong evidence for sequential systems. A layout
    # grouper can occasionally mistake evenly spaced single-staff systems for one
    # vertically stacked ensemble system. When the physical staff sequence exactly
    # agrees with the MusicXML break count, prefer it for engraving and widths.
    if score_systems and len(score_systems) == existing_system_count:
        source_systems = score_systems
    elif physical_systems and len(physical_systems) == existing_system_count:
        source_systems = physical_systems
    else:
        source_systems = score_systems or physical_systems
    target_systems = len(source_systems) if source_systems else len(existing) + 1
    if target_systems <= 1:
        return {"systems": 1, "breaks_applied": False}
    if len(existing) == target_systems - 1:
        starts = [0] + existing
        ends = existing + [len(reference_measures)]
        for part in parts:
            measures = part.findall("measure")
            if len(measures) != len(reference_measures):
                continue
            for system_index, (measure_index, end_index) in enumerate(zip(starts, ends, strict=False)):
                _set_system_layout(_ensure_print(measures[measure_index]), system_index, target_systems)
                source_system = source_systems[system_index] if system_index < len(source_systems) else None
                _apply_system_measure_widths(measures, measure_index, end_index - measure_index, source_system)
        return {"systems": target_systems, "breaks_applied": False, "widths_applied": True}

    estimates = [int(getattr(system, "measure_count", 1)) for system in source_systems] or [1] * target_systems
    reference_counts = _distribute_counts(estimates, len(reference_measures))
    for part in parts:
        measures = part.findall("measure")
        counts = (
            reference_counts
            if len(measures) == len(reference_measures)
            else _distribute_counts(estimates, len(measures))
        )
        for measure in measures:
            print_node = measure.find("print")
            if print_node is not None:
                print_node.attrib.pop("new-system", None)
        cursor = 0
        for system_index, count in enumerate(counts):
            if system_index > 0 and cursor < len(measures):
                _ensure_print(measures[cursor]).set("new-system", "yes")
            if cursor < len(measures):
                _set_system_layout(_ensure_print(measures[cursor]), system_index, len(counts))
            source_system = source_systems[system_index] if system_index < len(source_systems) else None
            _apply_system_measure_widths(measures, cursor, count, source_system)
            cursor += count
    return {"systems": len(reference_counts), "breaks_applied": True, "widths_applied": True}


def _first_part(root: etree._Element) -> etree._Element | None:
    return root.find("part")


def _ensure_identification(result_root: etree._Element) -> None:
    identification = result_root.find("identification")
    if identification is None:
        identification = etree.Element("identification")
        part_list = result_root.find("part-list")
        insert_pos = result_root.index(part_list) if part_list is not None else len(result_root)
        result_root.insert(insert_pos, identification)
    encoding = identification.find("encoding")
    if encoding is None:
        encoding = etree.SubElement(identification, "encoding")
    # Avoid duplicating software entries when a result is reprocessed.
    if not any((node.text or "").startswith(APP_NAME) for node in encoding.findall("software")):
        etree.SubElement(encoding, "software").text = f"{APP_NAME} {APP_VERSION}"
    _set_text(encoding, "encoding-date", date.today().isoformat())
    for element, attribute in (("print", "new-page"), ("print", "new-system")):
        supports = etree.SubElement(encoding, "supports")
        supports.set("element", element)
        supports.set("type", "yes")
        supports.set("attribute", attribute)
        supports.set("value", "yes")


def _ensure_defaults(result_root: etree._Element) -> None:
    defaults = result_root.find("defaults")
    if defaults is None:
        defaults = etree.Element("defaults")
        identification = result_root.find("identification")
        insert_at = result_root.index(identification) + 1 if identification is not None else 0
        result_root.insert(insert_at, defaults)
    scaling = defaults.find("scaling")
    if scaling is None:
        scaling = etree.SubElement(defaults, "scaling")
    _set_text(scaling, "millimeters", "7.0")
    _set_text(scaling, "tenths", "40")
    music_font = defaults.find("music-font")
    if music_font is None:
        etree.SubElement(defaults, "music-font", **{"font-family": "Leland"})
    word_font = defaults.find("word-font")
    if word_font is None:
        etree.SubElement(defaults, "word-font", **{"font-family": "Edwin", "font-size": "10"})


def _sanitize_dynamics(root: etree._Element) -> int:
    valid = {
        "p", "pp", "ppp", "pppp", "ppppp", "pppppp", "mp", "mf", "f", "ff", "fff",
        "ffff", "fffff", "ffffff", "fp", "pf", "sf", "sfp", "sfpp", "sfz", "sffz",
        "rf", "rfz", "fz", "n", "other-dynamics",
    }
    changed = 0
    for dynamics in root.findall(".//dynamics"):
        for child in list(dynamics):
            if child.tag not in valid:
                index = dynamics.index(child)
                text = child.text or child.tag
                dynamics.remove(child)
                replacement = etree.Element("other-dynamics")
                replacement.text = text
                dynamics.insert(index, replacement)
                changed += 1
    return changed


def _metadata_key(value: str | None) -> str:
    return "".join(
        character
        for character in (value or "").casefold()
        if character.isalnum()
    )


def _sanitize_duplicate_header_metadata(root: etree._Element) -> int:
    """Remove a header that merely duplicates the first musical direction.

    Some OMR outputs promote a first-system tempo word to ``work-title`` while
    also emitting the correct MusicXML direction. Importers then engrave it both
    as a large centred title and above the staff. The direction is the
    semantically correct representation, so retain it and discard only exact,
    punctuation-insensitive metadata duplicates.
    """

    first_measure = root.find("./part/measure")
    if first_measure is None:
        return 0
    direction_keys = {
        key
        for node in first_measure.findall("./direction/direction-type/words")
        if (key := _metadata_key(node.text))
    }
    if not direction_keys:
        return 0

    removed = 0
    for xpath in ("./work/work-title", "./movement-title"):
        for node in list(root.findall(xpath)):
            if _metadata_key(node.text) not in direction_keys:
                continue
            parent = node.getparent()
            if parent is None:
                continue
            parent.remove(node)
            removed += 1
            if parent.tag == "work" and len(parent) == 0:
                root.remove(parent)
    for credit in list(root.findall("credit")):
        words = credit.findall("credit-words")
        credit_keys = [
            key
            for node in words
            if (key := _metadata_key(node.text))
        ]
        if not credit_keys or not all(key in direction_keys for key in credit_keys):
            continue
        root.remove(credit)
        removed += 1
    return removed


def _strip_out_of_scope_lyric_semantics(root: etree._Element) -> int:
    """Remove note-attached lyric semantics from the public product output."""

    removed = 0
    for lyric in list(root.findall(".//note/lyric")):
        parent = lyric.getparent()
        if parent is None:
            continue
        parent.remove(lyric)
        removed += 1
    return removed


def merge_pages(page_documents: Iterable[PageDocument], output_path: Path) -> dict[str, object]:
    documents = list(page_documents)
    if not documents:
        raise ValueError("没有可合并的页面")
    first_root = documents[0].tree.getroot()
    canonical_parts = first_root.findall("part")
    if not canonical_parts:
        raise ValueError("第一页没有可合并的乐器声部")
    canonical_ids = [part.get("id") or f"P{index}" for index, part in enumerate(canonical_parts, start=1)]
    if len(canonical_ids) != len(set(canonical_ids)):
        raise ValueError("第一页包含重复的乐器声部 ID")
    result_root = etree.Element("score-partwise", version="4.0")

    for tag in ("work", "movement-number", "movement-title", "identification", "defaults", "credit"):
        for node in first_root.findall(tag):
            result_root.append(copy.deepcopy(node))

    part_list = first_root.find("part-list")
    if part_list is None:
        part_list = etree.Element("part-list")
        for part_id in canonical_ids:
            score_part = etree.SubElement(part_list, "score-part", id=part_id)
            etree.SubElement(score_part, "part-name").text = "Music"
    else:
        part_list = copy.deepcopy(part_list)
        listed_ids = {node.get("id") for node in part_list.findall("score-part")}
        for part_id in canonical_ids:
            if part_id in listed_ids:
                continue
            score_part = etree.SubElement(part_list, "score-part", id=part_id)
            etree.SubElement(score_part, "part-name").text = "Music"
    result_root.append(part_list)
    _ensure_identification(result_root)
    _ensure_defaults(result_root)
    _normalize_default_page_layout(
        result_root,
        documents[0].width,
        documents[0].height,
    )

    result_parts = {
        part_id: etree.SubElement(result_root, "part", id=part_id)
        for part_id in canonical_ids
    }
    measure_number = 1
    page_summaries: list[dict[str, object]] = []

    for page_index, document in enumerate(documents, start=1):
        break_summary = apply_source_system_breaks(document.tree, document.layout)
        source_parts = document.tree.getroot().findall("part")
        if len(source_parts) != len(canonical_ids):
            raise ValueError(
                f"第 {page_index} 页识别到 {len(source_parts)} 个乐器声部，"
                f"但第一页为 {len(canonical_ids)} 个；为避免错配，已停止合并"
            )
        measures_by_part = [source_part.findall("measure") for source_part in source_parts]
        measure_counts = {len(measures) for measures in measures_by_part}
        if len(measure_counts) != 1 or not measure_counts or next(iter(measure_counts)) <= 0:
            raise ValueError(
                f"第 {page_index} 页各乐器声部的小节数不一致；"
                "程序不会用横坐标或猜测强行对齐"
            )
        page_measure_count = next(iter(measure_counts))
        part_mapping: list[dict[str, str]] = []
        for canonical_id, source_part, measures in zip(
            canonical_ids,
            source_parts,
            measures_by_part,
            strict=True,
        ):
            part_mapping.append(
                {
                    "source_id": source_part.get("id") or "",
                    "result_id": canonical_id,
                }
            )
            result_part = result_parts[canonical_id]
            for local_index, source_measure in enumerate(measures):
                measure = copy.deepcopy(source_measure)
                measure.set("number", str(measure_number + local_index))
                if local_index == 0:
                    print_node = _ensure_print(measure)
                    if page_index > 1:
                        print_node.set("new-page", "yes")
                        print_node.attrib.pop("new-system", None)
                    _set_page_layout(print_node, document.width, document.height)
                result_part.append(measure)
        page_summaries.append(
            {
                "page": page_index,
                "measures": page_measure_count,
                "parts": len(source_parts),
                "part_mapping": part_mapping,
                "source_systems": (
                    len(getattr(document.layout, "score_systems", ()) or document.layout.systems)
                    if document.layout
                    else None
                ),
                **break_summary,
            }
        )
        measure_number += page_measure_count

    sanitized_dynamics = _sanitize_dynamics(result_root)
    sanitized_duplicate_headers = _sanitize_duplicate_header_metadata(result_root)
    stripped_out_of_scope_lyrics = _strip_out_of_scope_lyric_semantics(
        result_root
    )
    tree = etree.ElementTree(result_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = etree.tostring(
        tree, encoding="UTF-8", xml_declaration=True,
        pretty_print=True, doctype=MUSICXML_DOCTYPE,
    )
    atomic_write_bytes(output_path, payload)
    return {
        "pages": len(documents),
        "parts": len(canonical_ids),
        "measures": measure_number - 1,
        "page_summaries": page_summaries,
        "sanitized_dynamics": sanitized_dynamics,
        "sanitized_duplicate_headers": sanitized_duplicate_headers,
        "stripped_out_of_scope_lyrics": stripped_out_of_scope_lyrics,
    }


def package_mxl(musicxml_path: Path, mxl_path: Path) -> None:
    container_xml = b'''<?xml version="1.0" encoding="UTF-8"?>
<container>
  <rootfiles>
    <rootfile full-path="score.musicxml" media-type="application/vnd.recordare.musicxml+xml"/>
  </rootfiles>
</container>
'''
    mxl_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = mxl_path.with_name(mxl_path.name + ".tmp")
    temp_path.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(temp_path, "w") as archive:
            archive.writestr("mimetype", "application/vnd.recordare.musicxml", compress_type=zipfile.ZIP_STORED)
            archive.writestr("META-INF/container.xml", container_xml, compress_type=zipfile.ZIP_DEFLATED)
            archive.write(musicxml_path, "score.musicxml", compress_type=zipfile.ZIP_DEFLATED)
        temp_path.replace(mxl_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _measure_expected_duration(divisions: int, beats: int, beat_type: int) -> int:
    return int(round(divisions * beats * 4 / beat_type))


def _measure_voice_durations(measure: etree._Element) -> dict[str, int]:
    voices: dict[str, int] = {}
    active_voice = "1"
    cursor = 0
    last_anchor_onset = 0
    for child in measure:
        if child.tag == "note":
            active_voice = child.findtext("voice") or active_voice
            try:
                duration = int(child.findtext("duration") or 0)
            except ValueError:
                duration = 0
            chord = child.find("chord") is not None
            grace = child.find("grace") is not None
            onset = last_anchor_onset if chord else cursor
            if not chord:
                last_anchor_onset = onset
            voices[active_voice] = max(voices.get(active_voice, 0), onset + duration)
            if not chord and not grace:
                cursor += duration
        elif child.tag == "backup":
            try:
                cursor -= int(child.findtext("duration") or 0)
                cursor = max(0, cursor)
            except ValueError:
                pass
        elif child.tag == "forward":
            try:
                duration = int(child.findtext("duration") or 0)
            except ValueError:
                duration = 0
            cursor += duration
            voices[active_voice] = max(voices.get(active_voice, 0), cursor)
    return voices


def _canonical_voice_streams(
    measure: etree._Element,
) -> tuple[list[tuple[str, int, list[etree._Element], int]], str | None]:
    """Extract ordered voice streams and their first encoded onset.

    homr occasionally interleaves grand-staff voices without the backup required
    when returning to an earlier voice.  Event order *within* each voice remains
    correct.  This helper deliberately refuses directions, forwards and unusual
    mid-measure structures so the repair is limited to that exact failure mode.
    """

    allowed_static = {"print", "attributes", "barline"}
    unsupported = [
        child.tag
        for child in measure
        if child.tag not in allowed_static | {"note", "backup"}
    ]
    if unsupported:
        return [], f"unsupported children: {sorted(set(unsupported))}"

    cursor = 0
    last_anchor_onset = 0
    order: list[str] = []
    first_onsets: dict[str, int] = {}
    streams: dict[str, list[etree._Element]] = {}
    totals: dict[str, int] = {}
    previous_voice: str | None = None
    previous_was_anchor = False
    for child in measure:
        if child.tag == "backup":
            try:
                cursor = max(0, cursor - max(0, int(child.findtext("duration") or 0)))
            except ValueError:
                return [], "invalid backup duration"
            previous_was_anchor = False
            continue
        if child.tag != "note":
            continue
        voice = (child.findtext("voice") or "").strip()
        if not voice:
            return [], "missing voice"
        try:
            duration = max(0, int(child.findtext("duration") or 0))
        except ValueError:
            return [], "invalid note duration"
        chord = child.find("chord") is not None
        grace = child.find("grace") is not None
        if not grace and duration <= 0:
            return [], "non-grace note has no duration"
        onset = last_anchor_onset if chord else cursor
        if chord and (not previous_was_anchor or previous_voice != voice):
            return [], "orphan or cross-voice chord"
        if not chord:
            last_anchor_onset = onset
        if voice not in streams:
            order.append(voice)
            streams[voice] = []
            totals[voice] = 0
            first_onsets[voice] = onset
        streams[voice].append(child)
        if not chord and not grace:
            totals[voice] += duration
            cursor += duration
        previous_voice = voice
        previous_was_anchor = True

    return [
        (voice, first_onsets[voice], streams[voice], totals[voice])
        for voice in order
    ], None


def canonicalize_multivoice_timelines(path: Path) -> dict[str, object]:
    """Repair a narrow, deterministic grand-staff cursor-serialization defect.

    Pitches, durations, voices, staves and notation nodes are never changed.  A
    measure is rewritten only when its current cursor overflows the meter but the
    per-voice streams, using their original first onsets, fit the meter exactly.
    Ambiguous gaps, directions and forward elements cause abstention.
    """

    parser = etree.XMLParser(remove_blank_text=True, resolve_entities=False, no_network=True)
    tree = etree.parse(str(path), parser)
    repaired: list[dict[str, object]] = []
    abstained: list[dict[str, object]] = []

    for part in tree.getroot().findall("part"):
        divisions, beats, beat_type = 1, 4, 4
        for index, measure in enumerate(part.findall("measure"), start=1):
            for attributes in measure.findall("attributes"):
                try:
                    divisions = max(1, int(attributes.findtext("divisions") or divisions))
                except ValueError:
                    pass
                time_node = attributes.find("time")
                if time_node is not None and time_node.find("senza-misura") is None:
                    try:
                        beats = max(1, int(time_node.findtext("beats") or beats))
                        beat_type = max(1, int(time_node.findtext("beat-type") or beat_type))
                    except ValueError:
                        pass
            if measure.find("backup") is None:
                continue
            expected = _measure_expected_duration(divisions, beats, beat_type)
            before = max(_measure_voice_durations(measure).values(), default=0)
            if before <= expected:
                continue
            streams, error = _canonical_voice_streams(measure)
            label = measure.get("number", str(index))
            if error or len(streams) < 2:
                abstained.append(
                    {
                        "part_id": part.get("id") or "",
                        "measure": label,
                        "reason": error or "fewer than two voices",
                    }
                )
                continue
            canonical_ends = [first + total for _voice, first, _notes, total in streams]
            if (
                any(first < 0 or total <= 0 or end > expected for (_v, first, _n, total), end in zip(streams, canonical_ends, strict=True))
                or max(canonical_ends, default=0) != expected
                or before - expected > max(2, expected // 2)
            ):
                abstained.append(
                    {
                        "part_id": part.get("id") or "",
                        "measure": label,
                        "reason": "voice streams do not form an exact bounded meter",
                    }
                )
                continue

            prefix: list[etree._Element] = []
            suffix: list[etree._Element] = []
            seen_timed = False
            structurally_safe = True
            for child in measure:
                if child.tag in {"note", "backup"}:
                    seen_timed = True
                elif child.tag == "barline":
                    suffix.append(child)
                elif child.tag in {"print", "attributes"} and not seen_timed:
                    prefix.append(child)
                else:
                    structurally_safe = False
                    break
            if not structurally_safe:
                abstained.append(
                    {
                        "part_id": part.get("id") or "",
                        "measure": label,
                        "reason": "non-prefix static element",
                    }
                )
                continue

            for child in list(measure):
                measure.remove(child)
            for child in prefix:
                measure.append(child)
            previous_end = 0
            for stream_index, (_voice, first_onset, notes, total) in enumerate(streams):
                if stream_index:
                    backup = etree.SubElement(measure, "backup")
                    etree.SubElement(backup, "duration").text = str(previous_end)
                if first_onset:
                    forward = etree.SubElement(measure, "forward")
                    etree.SubElement(forward, "duration").text = str(first_onset)
                for note in notes:
                    measure.append(note)
                previous_end = first_onset + total
            for child in suffix:
                measure.append(child)

            after = max(_measure_voice_durations(measure).values(), default=0)
            if after != expected:
                raise RuntimeError(
                    f"internal multivoice canonicalization invariant failed for "
                    f"{part.get('id') or ''} measure {label}: {after} != {expected}"
                )
            repaired.append(
                {
                    "part_id": part.get("id") or "",
                    "measure": label,
                    "before": before,
                    "after": after,
                    "expected": expected,
                    "voices": [voice for voice, _first, _notes, _total in streams],
                }
            )

    if repaired:
        atomic_write_bytes(
            path,
            etree.tostring(
                tree,
                encoding="UTF-8",
                xml_declaration=True,
                pretty_print=True,
                doctype=MUSICXML_DOCTYPE,
            ),
        )
    return {
        "repaired_count": len(repaired),
        "repaired": repaired,
        "abstained_count": len(abstained),
        "abstained": abstained,
    }


def validate_musicxml(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        tree = etree.parse(str(path), etree.XMLParser(resolve_entities=False, no_network=True))
    except Exception as exc:
        return [f"XML 无法解析：{exc}"]
    root = tree.getroot()
    if root.tag != "score-partwise":
        errors.append("根元素不是 score-partwise")
    part_list = root.find("part-list")
    parts = root.findall("part")
    if part_list is None:
        errors.append("MusicXML 中没有 part-list")
    if not parts:
        errors.append("MusicXML 中没有 part")
        return errors
    listed_ids = {node.get("id") for node in root.findall("./part-list/score-part")}
    for part in parts:
        if part.get("id") not in listed_ids:
            errors.append(f"part {part.get('id')} 未在 part-list 中声明")
        if not part.findall("measure"):
            errors.append(f"part {part.get('id')} 没有 measure")
    return errors


def _analyze_part(part: etree._Element) -> dict[str, object]:
    measures = part.findall("measure")
    divisions, beats, beat_type = 1, 4, 4
    page_breaks = 0
    system_breaks = 0
    rhythm_issues: list[dict[str, object]] = []
    duration_observations: list[dict[str, object]] = []
    note_count = rest_count = direction_count = 0
    tie_issues: list[dict[str, object]] = []
    slur_issues: list[dict[str, object]] = []
    open_ties: dict[tuple[str, str, str, str, str], int] = {}
    open_slurs: dict[str, int] = {}

    for index, measure in enumerate(measures):
        print_node = measure.find("print")
        if print_node is not None:
            page_breaks += print_node.get("new-page") == "yes"
            system_breaks += print_node.get("new-system") == "yes"
        # MusicXML permits multiple attribute blocks and changes between measures.
        for attributes in measure.findall("attributes"):
            try:
                divisions = max(1, int(attributes.findtext("divisions") or divisions))
            except ValueError:
                pass
            time_node = attributes.find("time")
            if time_node is not None and time_node.find("senza-misura") is None:
                try:
                    beats = int(time_node.findtext("beats") or beats)
                    beat_type = int(time_node.findtext("beat-type") or beat_type)
                except ValueError:
                    pass
        note_count += len(measure.findall("note"))
        rest_count += len(measure.findall("note/rest"))
        direction_count += len(measure.findall("direction"))
        for note in measure.findall("note"):
            pitch_node = note.find("pitch")
            if pitch_node is not None:
                pitch = (
                    pitch_node.findtext("step") or "",
                    pitch_node.findtext("alter") or "0",
                    pitch_node.findtext("octave") or "",
                )
                voice = note.findtext("voice") or "1"
                staff = note.findtext("staff") or "1"
                tie_key = (voice, staff, *pitch)
                tie_types = {node.get("type") for node in note.findall("tie")}
                tie_types |= {node.get("type") for node in note.findall("./notations/tied")}
                if "stop" in tie_types:
                    if open_ties.get(tie_key, 0) <= 0:
                        tie_issues.append(
                            {
                                "measure": measure.get("number"),
                                "pitch": pitch,
                                "voice": voice,
                                "staff": staff,
                                "issue": "stop_without_start",
                            }
                        )
                    else:
                        open_ties[tie_key] -= 1
                if "start" in tie_types:
                    open_ties[tie_key] = open_ties.get(tie_key, 0) + 1
            for slur in note.findall("./notations/slur"):
                # Slurs may cross staves and voices. MusicXML's number identifies
                # the open arc within a part, so staff/x-position must not be used
                # as an alignment requirement.
                number = slur.get("number", "1")
                if slur.get("type") == "start":
                    open_slurs[number] = open_slurs.get(number, 0) + 1
                elif slur.get("type") == "stop":
                    if open_slurs.get(number, 0) <= 0:
                        slur_issues.append(
                            {
                                "measure": measure.get("number"),
                                "number": number,
                                "issue": "stop_without_start",
                            }
                        )
                    else:
                        open_slurs[number] -= 1

        durations = _measure_voice_durations(measure)
        expected = _measure_expected_duration(divisions, beats, beat_type)
        # A secondary piano voice need not itself occupy the full measure.  The
        # semantic completeness invariant is the furthest reached time in the
        # part, not equality of every independent voice duration.
        actual = max(durations.values(), default=0)
        voice_label = next(iter(durations)) if len(durations) == 1 else "*"
        duration_observations.append(
            {
                "index": index,
                "measure": measure.get("number", str(index + 1)),
                "voice": voice_label,
                "actual": actual,
                "expected": expected,
                "implicit": measure.get("implicit") == "yes",
            }
        )

    # A conventional anacrusis is frequently exported without implicit="yes".
    legal_incomplete: set[tuple[int, str]] = set()
    if len(duration_observations) >= 2:
        first = duration_observations[0]
        last = duration_observations[-1]
        first_actual, first_expected = int(first["actual"]), int(first["expected"])
        last_actual, last_expected = int(last["actual"]), int(last["expected"])
        if (
            first_expected == last_expected
            and 0 < first_actual < first_expected
            and 0 < last_actual < last_expected
            and first_actual + last_actual == first_expected
        ):
            legal_incomplete.add((int(first["index"]), str(first["voice"])))
            legal_incomplete.add((int(last["index"]), str(last["voice"])))
    # A complementary shortened final bar is conventional but not mandatory: editions
    # may end a movement with a complete bar, a repeat, or a separately balanced
    # section.  Infer a small opening pickup only with stable local meter evidence.
    # Requiring three following complete bars and limiting the opening to at most half
    # a bar keeps interior omissions and large first-bar recognition failures visible.
    if len(duration_observations) >= 4:
        first = duration_observations[0]
        first_actual = int(first["actual"])
        first_expected = int(first["expected"])
        stable_following = duration_observations[1:4]
        if (
            first_expected > 0
            and 0 < first_actual * 2 <= first_expected
            and all(
                int(item["expected"]) == first_expected
                and int(item["actual"]) == first_expected
                for item in stable_following
            )
        ):
            legal_incomplete.add((int(first["index"]), str(first["voice"])))
    for item in duration_observations:
        actual, expected = int(item["actual"]), int(item["expected"])
        key = (int(item["index"]), str(item["voice"]))
        if bool(item["implicit"]) or key in legal_incomplete or actual in {0, expected}:
            continue
        rhythm_issues.append(
            {
                "measure": item["measure"],
                "voice": item["voice"],
                "actual": actual,
                "expected": expected,
            }
        )

    for tie_key, count in open_ties.items():
        if count > 0:
            voice, staff, step, alter, octave = tie_key
            tie_issues.append(
                {
                    "pitch": (step, alter, octave),
                    "voice": voice,
                    "staff": staff,
                    "issue": "start_without_stop",
                    "count": count,
                }
            )
    for number, count in open_slurs.items():
        if count > 0:
            slur_issues.append({"number": number, "issue": "start_without_stop", "count": count})

    return {
        "part_id": part.get("id") or "",
        "measure_count": len(measures),
        "page_breaks": page_breaks,
        "system_breaks": system_breaks,
        "note_count": note_count,
        "rest_count": rest_count,
        "direction_count": direction_count,
        "rhythm_issues": rhythm_issues,
        "legal_incomplete_measures": [
            {"measure_index": index + 1, "voice": voice}
            for index, voice in sorted(legal_incomplete)
        ],
        "tie_issues": tie_issues,
        "slur_issues": slur_issues,
    }


def analyze_musicxml(path: Path) -> dict[str, object]:
    tree = etree.parse(str(path), etree.XMLParser(resolve_entities=False, no_network=True))
    root = tree.getroot()
    xml_parts = root.findall("part")
    if not xml_parts:
        return {
            "part_count": 0,
            "measure_count": 0,
            "page_count": 0,
            "system_breaks": 0,
            "rhythm_issues": ["没有 part"],
        }

    part_summaries = [_analyze_part(part) for part in xml_parts]
    multi_part = len(part_summaries) > 1

    def with_part_id(items: object, part_id: str) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for raw in items if isinstance(items, list) else []:
            item = dict(raw)
            if multi_part:
                item["part_id"] = part_id
            result.append(item)
        return result

    rhythm_issues: list[dict[str, object]] = []
    legal_incomplete: list[dict[str, object]] = []
    tie_issues: list[dict[str, object]] = []
    slur_issues: list[dict[str, object]] = []
    for summary in part_summaries:
        part_id = str(summary["part_id"])
        rhythm_issues.extend(with_part_id(summary["rhythm_issues"], part_id))
        legal_incomplete.extend(with_part_id(summary["legal_incomplete_measures"], part_id))
        tie_issues.extend(with_part_id(summary["tie_issues"], part_id))
        slur_issues.extend(with_part_id(summary["slur_issues"], part_id))

    score = score_from_tree(tree)
    generalized = len(score.parts) > 1 or any(part.staff_count > 1 for part in score.effective_parts)
    audit = audit_production_score(score) if generalized else audit_score(score)
    semantic_issues = [issue.to_dict() for issue in audit]
    semantic_issue_counts: dict[str, int] = {}
    for issue in semantic_issues:
        code = str(issue.get("code", "unknown"))
        semantic_issue_counts[code] = semantic_issue_counts.get(code, 0) + 1

    reference = part_summaries[0]
    measure_counts = [int(summary["measure_count"]) for summary in part_summaries]
    return {
        "part_count": len(part_summaries),
        "part_measure_counts": measure_counts,
        "measure_count_consistent": len(set(measure_counts)) == 1,
        "measure_count": int(reference["measure_count"]),
        # Print layout is duplicated in each MusicXML part. Count it once from
        # the reference part rather than multiplying pages by instrument count.
        "page_count": int(reference["page_breaks"]) + (1 if int(reference["measure_count"]) else 0),
        "page_breaks": int(reference["page_breaks"]),
        "system_breaks": int(reference["system_breaks"]),
        "note_count": sum(int(summary["note_count"]) for summary in part_summaries),
        "rest_count": sum(int(summary["rest_count"]) for summary in part_summaries),
        "direction_count": sum(int(summary["direction_count"]) for summary in part_summaries),
        "rhythm_issues": rhythm_issues,
        "legal_incomplete_measures": legal_incomplete,
        "tie_issues": tie_issues,
        "slur_issues": slur_issues,
        "semantic_issues": semantic_issues,
        "semantic_issue_counts": semantic_issue_counts,
        "parts": part_summaries,
    }


def extract_title(path: Path) -> str | None:
    try:
        root = etree.parse(str(path), etree.XMLParser(resolve_entities=False, no_network=True)).getroot()
    except Exception:
        return None
    for xpath in ("./work/work-title", "./movement-title", "./credit/credit-words"):
        value = root.findtext(xpath)
        if value and value.strip():
            return value.strip()
    return None


def normalize_single_voice_musicxml(path: Path) -> dict[str, int]:
    """Conservatively normalise a page-level MusicXML document for MuseScore.

    This function does not alter pitches, onsets, durations or measure boundaries. It
    only fills representation details that are unambiguous in a one-voice document:
    missing voice numbers, missing note types derived exactly from duration/divisions,
    and illegal dynamic child names.
    """
    from fractions import Fraction

    parser = etree.XMLParser(remove_blank_text=True, resolve_entities=False, no_network=True)
    tree = etree.parse(str(path), parser)
    root = tree.getroot()
    added_voices = 0
    inferred_types = 0
    inferred_dots = 0
    divisions = 1
    ratio_map: dict[Fraction, tuple[str, int]] = {
        Fraction(4, 1): ("whole", 0),
        Fraction(6, 1): ("whole", 1),
        Fraction(7, 1): ("whole", 2),
        Fraction(2, 1): ("half", 0),
        Fraction(3, 1): ("half", 1),
        Fraction(7, 2): ("half", 2),
        Fraction(1, 1): ("quarter", 0),
        Fraction(3, 2): ("quarter", 1),
        Fraction(7, 4): ("quarter", 2),
        Fraction(1, 2): ("eighth", 0),
        Fraction(3, 4): ("eighth", 1),
        Fraction(7, 8): ("eighth", 2),
        Fraction(1, 4): ("16th", 0),
        Fraction(3, 8): ("16th", 1),
        Fraction(7, 16): ("16th", 2),
        Fraction(1, 8): ("32nd", 0),
        Fraction(3, 16): ("32nd", 1),
        Fraction(1, 16): ("64th", 0),
    }

    for part in root.findall("part"):
        for measure in part.findall("measure"):
            attributes = measure.find("attributes")
            if attributes is not None:
                try:
                    divisions = max(1, int(attributes.findtext("divisions") or divisions))
                except ValueError:
                    divisions = max(1, divisions)
            notes = measure.findall("note")
            explicit_voices = {note.findtext("voice") for note in notes if note.findtext("voice")}
            single_timeline = measure.find("backup") is None and len(explicit_voices) <= 1
            for note in notes:
                if single_timeline and note.find("voice") is None:
                    duration_node = note.find("duration")
                    insert_at = note.index(duration_node) + 1 if duration_node is not None else len(note)
                    voice = etree.Element("voice")
                    voice.text = "1"
                    note.insert(insert_at, voice)
                    added_voices += 1
                if note.find("type") is not None or note.find("grace") is not None or note.find("time-modification") is not None:
                    continue
                try:
                    duration = int(note.findtext("duration") or 0)
                except ValueError:
                    duration = 0
                if duration <= 0:
                    continue
                inferred = ratio_map.get(Fraction(duration, divisions))
                if inferred is None:
                    continue
                note_type, dots = inferred
                type_node = etree.Element("type")
                type_node.text = note_type
                # MusicXML order places type after voice and before dot/accidental.
                insert_at = len(note)
                for index, child in enumerate(note):
                    if child.tag in {"dot", "accidental", "time-modification", "stem", "notehead", "beam", "notations", "lyric"}:
                        insert_at = index
                        break
                note.insert(insert_at, type_node)
                inferred_types += 1
                for _ in range(dots):
                    type_index = note.index(type_node)
                    note.insert(type_index + 1, etree.Element("dot"))
                    inferred_dots += 1

    sanitized_dynamics = _sanitize_dynamics(root)
    changed = added_voices + inferred_types + inferred_dots + sanitized_dynamics
    if changed:
        atomic_write_bytes(
            path,
            etree.tostring(
                tree,
                encoding="UTF-8",
                xml_declaration=True,
                pretty_print=True,
                doctype=MUSICXML_DOCTYPE,
            ),
        )
    return {
        "added_voices": added_voices,
        "inferred_types": inferred_types,
        "inferred_dots": inferred_dots,
        "sanitized_dynamics": sanitized_dynamics,
    }
