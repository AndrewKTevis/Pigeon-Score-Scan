from __future__ import annotations

"""System-localised OMR rescue for structurally ambiguous pages.

The ordinary ensemble recognises complete pages under several deterministic image
preprocessings.  Those candidates remain correlated through the same page-level staff
segmentation.  This module supplies one genuinely different failure mode: each detected
single-staff system is cropped with bounded context, recognised in isolation, and then
transactionally concatenated into one page candidate.

Localised recognition never runs on weak layouts, never accepts partial pages, and never
modifies notation itself.  Its MusicXML enters the same validation, family consensus and
veto gates as every other candidate.
"""

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
from lxml import etree

from .layout import PageLayout, StaffSystem
from .musicxml import MUSICXML_DOCTYPE, validate_musicxml
from .policy import DEFAULT_POLICY
from .util import atomic_write_bytes, atomic_write_json, sha256_file


@dataclass(frozen=True)
class SystemCrop:
    system_index: int
    image_path: str
    source_bbox: tuple[int, int, int, int]
    padded_shape: tuple[int, int]
    expected_measure_count: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["source_bbox"] = list(self.source_bbox)
        payload["padded_shape"] = list(self.padded_shape)
        return payload


@dataclass(frozen=True)
class LocalizedSystemResult:
    system_index: int
    image_path: str
    xml_path: str | None
    return_code: int
    elapsed_seconds: float
    valid: bool
    expected_measure_count: int
    observed_measure_count: int
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def localized_recognition_eligible(layout: PageLayout | None) -> tuple[bool, str]:
    if layout is None or not layout.systems:
        return False, "layout_missing"
    if layout.confidence < DEFAULT_POLICY.localized_layout_confidence_floor:
        return False, "layout_confidence_low"
    if len(layout.systems) < DEFAULT_POLICY.localized_min_systems:
        return False, "too_few_systems"
    if len(layout.systems) > DEFAULT_POLICY.localized_max_systems:
        return False, "too_many_systems"
    seen_indices: set[int] = set()
    previous: StaffSystem | None = None
    for system in layout.systems:
        if system.spacing < 3.0 or system.right <= system.left or system.bottom <= system.top:
            return False, "invalid_system_geometry"
        if system.measure_count <= 0:
            return False, "invalid_system_measure_count"
        if system.index in seen_indices:
            return False, "duplicate_system_index"
        seen_indices.add(system.index)
        if previous is not None:
            if system.top <= previous.top or system.bottom <= previous.bottom:
                return False, "systems_not_monotonic"
            overlap = max(0.0, float(previous.bottom - system.top))
            if overlap > min(previous.spacing, system.spacing) * 1.5:
                return False, "systems_overlap"
        previous = system
    return True, "eligible"


def _vertical_crop_bounds(layout: PageLayout, index: int, system: StaffSystem) -> tuple[int, int]:
    spacing = max(3.0, float(system.spacing))
    top = max(0, int(round(system.top - spacing * DEFAULT_POLICY.localized_vertical_context_ratio)))
    bottom = min(
        layout.height,
        int(round(system.bottom + spacing * DEFAULT_POLICY.localized_vertical_context_ratio)) + 1,
    )
    if index > 0:
        previous = layout.systems[index - 1]
        midpoint = int(round((previous.bottom + system.top) / 2.0))
        top = max(top, midpoint)
    if index + 1 < len(layout.systems):
        following = layout.systems[index + 1]
        midpoint = int(round((system.bottom + following.top) / 2.0))
        bottom = min(bottom, midpoint)
    if bottom <= top:
        top = max(0, int(system.top))
        bottom = min(layout.height, int(system.bottom) + 1)
    return top, bottom


def create_system_crops(
    image_path: Path,
    layout: PageLayout,
    output_dir: Path,
) -> tuple[SystemCrop, ...]:
    eligible, reason = localized_recognition_eligible(layout)
    if not eligible:
        raise ValueError(f"system-localised recognition unavailable: {reason}")
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("system-localised recognition cannot read page image")
    height, width = image.shape
    if height != layout.height or width != layout.width:
        # Layout/image drift would make every crop provenance claim unreliable.
        raise ValueError("system-localised recognition layout dimensions do not match page image")

    output_dir.mkdir(parents=True, exist_ok=True)
    crops: list[SystemCrop] = []
    total_pixels = 0
    for offset, system in enumerate(layout.systems):
        spacing = max(3.0, float(system.spacing))
        horizontal_context = max(
            DEFAULT_POLICY.localized_min_horizontal_context_pixels,
            int(round(spacing * DEFAULT_POLICY.localized_horizontal_context_ratio)),
        )
        left = max(0, int(system.left) - horizontal_context)
        right = min(width, int(system.right) + horizontal_context + 1)
        top, bottom = _vertical_crop_bounds(layout, offset, system)
        source = image[top:bottom, left:right]
        if source.size == 0:
            raise ValueError(f"system {system.index} produced an empty localisation crop")

        border_x = max(
            DEFAULT_POLICY.localized_min_border_pixels,
            int(round(spacing * DEFAULT_POLICY.localized_border_ratio)),
        )
        border_y = max(
            DEFAULT_POLICY.localized_min_border_pixels,
            int(round(spacing * DEFAULT_POLICY.localized_border_ratio)),
        )
        padded = cv2.copyMakeBorder(
            source,
            border_y,
            border_y,
            border_x,
            border_x,
            cv2.BORDER_CONSTANT,
            value=255,
        )
        total_pixels += int(padded.size)
        if total_pixels > DEFAULT_POLICY.localized_max_total_pixels:
            raise ValueError("system-localised recognition crop budget exceeded")
        encoded, payload = cv2.imencode(".png", padded, [cv2.IMWRITE_PNG_COMPRESSION, 2])
        if not encoded:
            raise OSError(f"failed to encode localised system {system.index}")
        path = output_dir / f"system_{system.index:03d}_primary.png"
        atomic_write_bytes(path, payload.tobytes())
        crops.append(
            SystemCrop(
                system_index=int(system.index),
                image_path=str(path),
                source_bbox=(left, top, right, bottom),
                padded_shape=(int(padded.shape[1]), int(padded.shape[0])),
                expected_measure_count=max(1, int(system.measure_count)),
                sha256=sha256_file(path),
            )
        )
    atomic_write_json(
        output_dir / "system_crops.json",
        {
            "format": 1,
            "page_image": str(image_path),
            "layout_confidence": round(float(layout.confidence), 6),
            "system_count": len(crops),
            "total_pixels": total_pixels,
            "crops": [crop.to_dict() for crop in crops],
        },
    )
    return tuple(crops)


def _first_part(root: etree._Element) -> etree._Element | None:
    return root.find("part")


def _measure_count(path: Path) -> int:
    tree = etree.parse(str(path))
    root = tree.getroot()
    part = _first_part(root)
    return len(part.findall("measure")) if part is not None else 0


def validate_localized_system_xml(path: Path, expected_measure_count: int) -> tuple[bool, int, str | None]:
    errors = validate_musicxml(path)
    if errors:
        return False, 0, "; ".join(errors[:3])
    try:
        observed = _measure_count(path)
    except Exception as exc:
        return False, 0, f"MusicXML parse failed: {exc}"
    if observed <= 0:
        return False, observed, "system MusicXML contains no measures"
    gap = abs(observed - max(1, int(expected_measure_count)))
    allowed = max(
        DEFAULT_POLICY.localized_measure_gap_absolute,
        int(round(max(1, expected_measure_count) * DEFAULT_POLICY.localized_measure_gap_ratio)),
    )
    if gap > allowed:
        return False, observed, f"system measure count gap {gap} exceeds {allowed}"
    return True, observed, None


def merge_localized_system_musicxml(
    xml_paths: Iterable[Path],
    output_path: Path,
) -> int:
    paths = tuple(xml_paths)
    if not paths:
        raise ValueError("no localised system MusicXML documents")
    trees = [etree.parse(str(path)) for path in paths]
    first_root = trees[0].getroot()
    if etree.QName(first_root).localname != "score-partwise":
        raise ValueError("localised system output is not score-partwise MusicXML")

    result_root = etree.Element("score-partwise", version=first_root.get("version", "4.0"))
    for tag in ("work", "movement-number", "movement-title", "identification", "defaults", "credit"):
        for node in first_root.findall(tag):
            result_root.append(copy.deepcopy(node))
    part_list = first_root.find("part-list")
    if part_list is None:
        part_list = etree.Element("part-list")
        score_part = etree.SubElement(part_list, "score-part", id="P1")
        etree.SubElement(score_part, "part-name").text = "Music"
    else:
        part_list = copy.deepcopy(part_list)
        for extra in part_list.findall("score-part")[1:]:
            part_list.remove(extra)
    result_root.append(part_list)
    first_score_part = part_list.find("score-part")
    part_id = first_score_part.get("id", "P1") if first_score_part is not None else "P1"
    result_part = etree.SubElement(result_root, "part", id=part_id)

    measure_number = 1
    for tree in trees:
        root = tree.getroot()
        if etree.QName(root).localname != "score-partwise":
            raise ValueError("localised system output is not score-partwise MusicXML")
        source_part = _first_part(root)
        if source_part is None:
            raise ValueError("localised system output has no part")
        measures = source_part.findall("measure")
        if not measures:
            raise ValueError("localised system output has no measures")
        for source_measure in measures:
            measure = copy.deepcopy(source_measure)
            measure.set("number", str(measure_number))
            for print_node in measure.findall("print"):
                print_node.attrib.pop("new-page", None)
                print_node.attrib.pop("new-system", None)
                if not print_node.attrib and len(print_node) == 0 and not (print_node.text or "").strip():
                    measure.remove(print_node)
            result_part.append(measure)
            measure_number += 1

    payload = etree.tostring(
        etree.ElementTree(result_root),
        encoding="UTF-8",
        xml_declaration=True,
        pretty_print=True,
        doctype=MUSICXML_DOCTYPE,
    )
    atomic_write_bytes(output_path, payload)
    errors = validate_musicxml(output_path)
    if errors:
        output_path.unlink(missing_ok=True)
        raise ValueError("merged localised MusicXML failed validation: " + "; ".join(errors[:3]))
    return measure_number - 1
