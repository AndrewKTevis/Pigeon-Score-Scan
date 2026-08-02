from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from lxml import etree

from .layout import PageLayout, anchor_x_to_measure
from .musicxml import MUSICXML_DOCTYPE
from .text_enrichment import (
    _measure_duration,
    _part_staff_count,
    _source_staff_target,
    _xml_system_measure_groups,
)
from .util import atomic_write_bytes


@dataclass(frozen=True)
class SourceMordentCandidate:
    x: float
    y: float
    width: int
    height: int
    physical_staff_position: int
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SourceTrillCandidate:
    x: float
    y: float
    width: int
    height: int
    physical_staff_position: int
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OrnamentEnrichmentReport:
    detected_count: int
    reclassified_trill_count: int
    inserted_mordent_count: int
    abstention_count: int
    candidates: tuple[SourceMordentCandidate, ...]
    detected_trill_count: int = 0
    inserted_trill_count: int = 0
    removed_spurious_ornament_count: int = 0
    authoritative_source_commit: bool = False
    trill_candidates: tuple[SourceTrillCandidate, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["candidates"] = [item.to_dict() for item in self.candidates]
        payload["trill_candidates"] = [item.to_dict() for item in self.trill_candidates]
        return payload


def detect_source_mordents(
    image_path: Path,
    layout: PageLayout,
) -> tuple[SourceMordentCandidate, ...]:
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return ()
    binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8),
        8,
    )
    candidates: list[SourceMordentCandidate] = []
    for physical_position, staff in enumerate(layout.systems):
        spacing = max(1.0, float(staff.spacing))
        top = float(staff.line_y[0])
        for label_id in range(1, count):
            x, y, width, height, area = (int(value) for value in stats[label_id])
            center_x, center_y = (float(value) for value in centroids[label_id])
            if not (
                float(staff.left) + 1.5 * spacing
                <= center_x
                <= float(staff.right) + spacing
                and top - 4.5 * spacing
                <= center_y
                <= top - 0.55 * spacing
                and 1.55 * spacing <= width <= 2.55 * spacing
                and 0.65 * spacing <= height <= 1.35 * spacing
                and 0.68 * spacing * spacing
                <= area
                <= 1.35 * spacing * spacing
                and 0.34 <= area / max(float(width * height), 1.0) <= 0.68
                and 1.45 <= width / max(float(height), 1.0) <= 3.1
            ):
                continue
            candidates.append(
                SourceMordentCandidate(
                    x=center_x,
                    y=center_y,
                    width=width,
                    height=height,
                    physical_staff_position=physical_position,
                    confidence=0.985,
                )
            )
    deduplicated: list[SourceMordentCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.y,
            item.x,
            item.physical_staff_position,
        ),
    ):
        if any(
            abs(candidate.x - existing.x) <= 2.0
            and abs(candidate.y - existing.y) <= 2.0
            for existing in deduplicated
        ):
            continue
        deduplicated.append(candidate)
    return tuple(deduplicated)


def detect_source_trills(
    image_path: Path,
    layout: PageLayout,
) -> tuple[SourceTrillCandidate, ...]:
    """Detect the compact italic ``tr`` glyph from source geometry.

    General-purpose OCR is unusually unstable on music-font ``tr`` glyphs:
    the same printed glyph is commonly decoded as ``dr``, ``t`` or unrelated
    Cyrillic characters.  This detector therefore uses the stable,
    staff-normalised geometry of the two-letter ink group.  The fill-ratio
    floor deliberately excludes italic ``f`` dynamics, while the height floor
    excludes mordent/wave symbols.
    """

    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return ()
    binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]
    candidates: list[SourceTrillCandidate] = []
    for physical_position, staff in enumerate(layout.systems):
        spacing = max(1.0, float(staff.spacing))
        top_line = float(staff.line_y[0])
        band_top = max(0, int(round(top_line - 7.0 * spacing)))
        band_bottom = min(binary.shape[0], int(round(top_line - 1.2 * spacing)))
        if band_bottom <= band_top:
            continue
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            (binary[band_top:band_bottom] > 0).astype(np.uint8),
            8,
        )
        components: list[tuple[int, int, int, int, int]] = []
        for label_id in range(1, count):
            x, local_y, width, height, area = (
                int(value) for value in stats[label_id]
            )
            y = local_y + band_top
            if not (
                float(staff.left) + 1.5 * spacing <= x + width / 2.0
                <= float(staff.right) + spacing
                and 0.45 * spacing <= width <= 2.75 * spacing
                and 0.70 * spacing <= height <= 2.75 * spacing
                and area >= 0.12 * spacing * spacing
            ):
                continue
            components.append((x, y, width, height, area))

        hypotheses: list[tuple[int, int, int, int, int]] = list(components)
        ordered = sorted(components, key=lambda item: (item[0], item[1]))
        for left, right in zip(ordered, ordered[1:]):
            lx, ly, lw, lh, la = left
            rx, ry, rw, rh, ra = right
            gap = rx - (lx + lw)
            vertical_overlap = min(ly + lh, ry + rh) - max(ly, ry)
            if gap > 0.40 * spacing or vertical_overlap < 0.35 * min(lh, rh):
                continue
            x = min(lx, rx)
            y = min(ly, ry)
            far_right = max(lx + lw, rx + rw)
            bottom = max(ly + lh, ry + rh)
            hypotheses.append((x, y, far_right - x, bottom - y, la + ra))

        for x, y, width, height, area in hypotheses:
            center_x = x + width / 2.0
            center_y = y + height / 2.0
            fill_ratio = area / max(float(width * height), 1.0)
            if not (
                1.65 * spacing <= width <= 2.70 * spacing
                and 1.85 * spacing <= height <= 2.65 * spacing
                and 1.25 * spacing * spacing
                <= area
                <= 2.25 * spacing * spacing
                and 0.295 <= fill_ratio <= 0.46
                and 0.72 <= width / max(float(height), 1.0) <= 1.38
                and 2.0 * spacing
                <= top_line - center_y
                <= 6.1 * spacing
            ):
                continue
            candidates.append(
                SourceTrillCandidate(
                    x=center_x,
                    y=center_y,
                    width=width,
                    height=height,
                    physical_staff_position=physical_position,
                    confidence=0.985,
                )
            )

    deduplicated: list[SourceTrillCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.y,
            item.x,
            -item.width,
        ),
    ):
        if any(
            candidate.physical_staff_position == existing.physical_staff_position
            and abs(candidate.x - existing.x) <= 0.8 * max(candidate.width, existing.width)
            and abs(candidate.y - existing.y) <= 0.8 * max(candidate.height, existing.height)
            for existing in deduplicated
        ):
            continue
        deduplicated.append(candidate)
    return tuple(deduplicated)


def _pitched_notes_at_onsets(
    measure: etree._Element,
    staff_number: int,
) -> list[tuple[int, etree._Element]]:
    cursor = 0
    last_anchor = 0
    result: list[tuple[int, etree._Element]] = []
    for child in measure:
        if child.tag == "note":
            chord = child.find("chord") is not None
            grace = child.find("grace") is not None
            onset = last_anchor if chord else cursor
            if not chord:
                last_anchor = onset
            try:
                duration = max(0, int(child.findtext("duration") or 0))
            except ValueError:
                duration = 0
            try:
                note_staff = max(1, int(child.findtext("staff") or 1))
            except ValueError:
                note_staff = 1
            if child.find("pitch") is not None and note_staff == staff_number:
                result.append((onset, child))
            if not chord and not grace:
                cursor += duration
        elif child.tag == "backup":
            try:
                cursor = max(0, cursor - max(0, int(child.findtext("duration") or 0)))
            except ValueError:
                pass
        elif child.tag == "forward":
            try:
                cursor += max(0, int(child.findtext("duration") or 0))
            except ValueError:
                pass
    return result


def _set_mordent(note: etree._Element) -> tuple[bool, bool]:
    notations = note.find("notations")
    if notations is None:
        notations = etree.SubElement(note, "notations")
    ornaments = notations.find("ornaments")
    if ornaments is None:
        ornaments = etree.SubElement(notations, "ornaments")
    if ornaments.find("mordent") is not None:
        return False, False
    trill = ornaments.find("trill-mark")
    reclassified = trill is not None
    if trill is not None:
        ornaments.remove(trill)
    etree.SubElement(ornaments, "mordent")
    return True, reclassified


def _set_trill(note: etree._Element) -> tuple[bool, bool]:
    notations = note.find("notations")
    if notations is None:
        notations = etree.SubElement(note, "notations")
    ornaments = notations.find("ornaments")
    if ornaments is None:
        ornaments = etree.SubElement(notations, "ornaments")
    if ornaments.find("trill-mark") is not None:
        return False, False
    mordent = ornaments.find("mordent")
    reclassified = mordent is not None
    if mordent is not None:
        ornaments.remove(mordent)
    etree.SubElement(ornaments, "trill-mark")
    return True, reclassified


def _remove_source_ornament_tags(root: etree._Element) -> int:
    removed = 0
    for ornaments in root.findall("./part/measure/note/notations/ornaments"):
        for child in list(ornaments):
            if child.tag in {"trill-mark", "mordent"}:
                ornaments.remove(child)
                removed += 1
    return removed


def enrich_musicxml_with_source_ornaments(
    image_path: Path,
    xml_path: Path,
    layout: PageLayout,
) -> OrnamentEnrichmentReport:
    candidates = detect_source_mordents(image_path, layout)
    trill_candidates = detect_source_trills(image_path, layout)
    source_count = len(candidates) + len(trill_candidates)
    if not source_count or not xml_path.is_file():
        return OrnamentEnrichmentReport(
            len(candidates),
            0,
            0,
            source_count,
            candidates,
            detected_trill_count=len(trill_candidates),
            trill_candidates=trill_candidates,
        )
    parser = etree.XMLParser(remove_blank_text=True, resolve_entities=False, no_network=True)
    tree = etree.parse(str(xml_path), parser)
    root = tree.getroot()
    parts = root.findall("part")
    part_staff_counts = [_part_staff_count(part) for part in parts]
    part_systems = [_xml_system_measure_groups(part) for part in parts]
    assignments: list[
        tuple[str, SourceMordentCandidate | SourceTrillCandidate, etree._Element]
    ] = []
    abstained = 0
    touched_notes: set[etree._Element] = set()

    source_candidates: list[
        tuple[str, SourceMordentCandidate | SourceTrillCandidate]
    ] = [
        *(("mordent", candidate) for candidate in candidates),
        *(("trill", candidate) for candidate in trill_candidates),
    ]
    for kind, candidate in source_candidates:
        mapping = _source_staff_target(
            layout,
            candidate.physical_staff_position,
            part_staff_counts,
        )
        if mapping is None:
            abstained += 1
            continue
        score_system_index, part_index, staff_number, _source_staff = mapping
        systems = part_systems[part_index]
        if not 0 <= score_system_index < len(systems):
            abstained += 1
            continue
        group = systems[score_system_index]
        source_measure_system = layout.effective_score_systems[score_system_index]
        anchor = anchor_x_to_measure(source_measure_system, candidate.x, len(group))
        if not 0 <= anchor.local_index < len(group):
            abstained += 1
            continue
        measure = group[anchor.local_index]
        notes = _pitched_notes_at_onsets(measure, staff_number)
        if not notes:
            abstained += 1
            continue
        target_time = anchor.offset_ratio * _measure_duration(measure)
        onset, note = min(notes, key=lambda item: abs(float(item[0]) - target_time))
        if note in touched_notes or (
            len(notes) > 1
            and abs(float(onset) - target_time)
            > max(1.0, _measure_duration(measure) * 0.30)
        ):
            abstained += 1
            continue
        touched_notes.add(note)
        assignments.append((kind, candidate, note))

    original_count = len(
        root.findall("./part/measure/note/notations/ornaments/trill-mark")
    ) + len(root.findall("./part/measure/note/notations/ornaments/mordent"))
    # Commit an authoritative source transaction only when every high-confidence
    # source glyph has a unique, rhythmically plausible note target and the
    # source inventory is at least as complete as the OMR inventory.
    authoritative = (
        source_count >= original_count
        and len(assignments) == source_count
        and abstained == 0
        and source_count >= 2
    )
    already_exact = authoritative and original_count == source_count and all(
        note.find(
            "notations/ornaments/"
            + ("mordent" if kind == "mordent" else "trill-mark")
        )
        is not None
        for kind, _candidate, note in assignments
    )
    if already_exact:
        return OrnamentEnrichmentReport(
            len(candidates),
            0,
            0,
            abstained,
            candidates,
            detected_trill_count=len(trill_candidates),
            inserted_trill_count=0,
            authoritative_source_commit=True,
            trill_candidates=trill_candidates,
        )
    removed = _remove_source_ornament_tags(root) if authoritative else 0
    inserted_mordents = 0
    inserted_trills = 0
    reclassified = 0
    for kind, _candidate, note in assignments:
        if kind == "mordent":
            changed, was_other = _set_mordent(note)
            inserted_mordents += int(changed)
        else:
            changed, was_other = _set_trill(note)
            inserted_trills += int(changed)
        reclassified += int(was_other)

    if inserted_mordents or inserted_trills or removed:
        atomic_write_bytes(
            xml_path,
            etree.tostring(
                tree,
                encoding="UTF-8",
                xml_declaration=True,
                pretty_print=True,
                doctype=MUSICXML_DOCTYPE,
            ),
        )
    return OrnamentEnrichmentReport(
        len(candidates),
        reclassified,
        inserted_mordents,
        abstained,
        candidates,
        detected_trill_count=len(trill_candidates),
        inserted_trill_count=inserted_trills,
        removed_spurious_ornament_count=max(0, removed - source_count),
        authoritative_source_commit=authoritative,
        trill_candidates=trill_candidates,
    )
