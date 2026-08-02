from __future__ import annotations

"""Position-aware source-symbol audit backed by the release-gated detector.

The audit never invents MusicXML.  It compares high-precision source detections with
the already selected score by simultaneous staff slot and measure.  Most classes are
omission-only evidence because a false negative cannot prove that an emitted symbol is
wrong. ``genericAccidental`` is bidirectional only because its release manifest has a
separate 99% recall floor in addition to the 99.5% precision floor.
"""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

from lxml import etree

from .accidental_semantics import normalise_accidental
from .layout import PageLayout, StaffSystem, anchor_x_to_measure
from .score_ir import ScoreIR, score_from_tree
from .semantic_detector import (
    GEOMETRY_CORROBORATION_CLASSES,
    POSITIONAL_INVENTORY_CLASSES,
    SYMBOL_AUDIT_CLASSES,
    SemanticDetection,
)


_FIFTHS_ORDER = ("F", "C", "G", "D", "A", "E", "B")
_COMPARABLE_CLASSES = frozenset(
    {
        "arpeggio",
        "augmentationDot",
        "beam",
        "breathMark",
        "fermata",
        "flag",
        "genericAccidental",
        "genericArticulation",
        "genericDynamic",
        "genericOrnament",
        "genericRest",
        "glissando",
        "graceSlash",
        "hairpin",
        "ottava",
        "pedal",
        "slur",
        "tie",
        "tremoloBetweenNotes",
        "tremoloSingle",
        "trillExtension",
        "tuplet",
        "volta",
    }
)
assert _COMPARABLE_CLASSES <= (
    SYMBOL_AUDIT_CLASSES
    | GEOMETRY_CORROBORATION_CLASSES
    | frozenset({"genericDynamic"})
)


@dataclass(frozen=True)
class SemanticSourceMismatch:
    class_name: str
    measure_index: int
    staff_slot: int
    source_count: int
    output_count: int
    kind: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticSourceAuditReport:
    status: str
    simultaneous_staff_count: int
    output_measure_count: int
    layout_measure_count: int
    mapped_detection_count: int
    unmapped_detection_count: int
    source_counts: dict[str, int]
    output_counts: dict[str, int]
    mismatches: tuple[SemanticSourceMismatch, ...]

    @property
    def omission_count(self) -> int:
        return sum(item.kind == "source_symbol_missing" for item in self.mismatches)

    @property
    def extraneous_count(self) -> int:
        return sum(item.kind == "source_absent_output_symbol" for item in self.mismatches)

    @property
    def positional_mismatch_count(self) -> int:
        return len(self.mismatches)

    def to_dict(self) -> dict[str, object]:
        return {
            "format": 2,
            "status": self.status,
            "simultaneous_staff_count": self.simultaneous_staff_count,
            "output_measure_count": self.output_measure_count,
            "layout_measure_count": self.layout_measure_count,
            "mapped_detection_count": self.mapped_detection_count,
            "unmapped_detection_count": self.unmapped_detection_count,
            "source_counts": dict(sorted(self.source_counts.items())),
            "output_counts": dict(sorted(self.output_counts.items())),
            "omission_count": self.omission_count,
            "extraneous_count": self.extraneous_count,
            "positional_mismatch_count": self.positional_mismatch_count,
            "mismatches": [item.to_dict() for item in self.mismatches],
        }


def _ordered_staffs(layout: PageLayout) -> list[StaffSystem]:
    return sorted(
        layout.systems,
        key=lambda item: (
            item.line_y[0] if item.line_y else item.top,
            item.index,
        ),
    )


def _group_measure_count(group: list[StaffSystem]) -> int:
    counts = sorted(max(1, int(item.measure_count)) for item in group)
    return counts[len(counts) // 2]


def _measure_for_detection(
    detection: SemanticDetection,
    layout: PageLayout,
    staffs: list[StaffSystem],
    simultaneous_staff_count: int,
) -> tuple[int, int] | None:
    position_by_index = {staff.index: index for index, staff in enumerate(staffs)}
    position = position_by_index.get(detection.staff_index)
    if position is None:
        return None
    simultaneous = max(1, int(simultaneous_staff_count))
    group_index = position // simultaneous
    group_start = group_index * simultaneous
    group = staffs[group_start : group_start + simultaneous]
    if len(group) != simultaneous:
        return None
    offset = 0
    for start in range(0, group_start, simultaneous):
        previous = staffs[start : start + simultaneous]
        if len(previous) != simultaneous:
            return None
        offset += _group_measure_count(previous)

    score_systems = layout.effective_score_systems
    if not 0 <= group_index < len(score_systems):
        return None
    score_system = score_systems[group_index]
    if score_system.staff_indices != [item.index for item in group]:
        # A symbol must never be assigned through a geometry group that does not
        # describe the exact candidate-conditioned simultaneous staff set.
        return None
    centre_x = 0.5 * (detection.bbox[0] + detection.bbox[2])
    group_count = _group_measure_count(group)
    anchor = anchor_x_to_measure(score_system, centre_x, group_count)
    if not 0 <= anchor.local_index < group_count:
        return None
    staff_slot = position % simultaneous + 1
    return offset + anchor.local_index + 1, staff_slot


def _key_alter(fifths: int, step: str) -> Fraction:
    token = str(step).upper()
    if fifths > 0 and token in _FIFTHS_ORDER[: min(7, fifths)]:
        return Fraction(1, 1)
    flat_order = tuple(reversed(_FIFTHS_ORDER))
    if fifths < 0 and token in flat_order[: min(7, -fifths)]:
        return Fraction(-1, 1)
    return Fraction(0, 1)


def _accidental_inventory(score: ScoreIR) -> Counter[tuple[int, int]]:
    result: Counter[tuple[int, int]] = Counter()
    staff_base = 0
    for part in score.effective_parts:
        for measure_index, measure in enumerate(part.measures, start=1):
            fifths = int(measure.key_signature[0]) if measure.key_signature else 0
            state: dict[tuple[int, str, int], Fraction] = {}
            for note in measure.notes:
                if note.pitch is None or note.rest:
                    continue
                position = (
                    max(1, int(note.staff)),
                    note.pitch.step.upper(),
                    int(note.pitch.octave),
                )
                explicit = normalise_accidental(note.accidental)
                if explicit:
                    result[(measure_index, staff_base + position[0])] += 1
                    state[position] = note.pitch.alter
                    continue
                current = state.get(
                    position,
                    _key_alter(fifths, note.pitch.step),
                )
                if "stop" in note.ties:
                    # A tie continuation carries the sounding alteration over the
                    # barline without printing it, then seeds this measure's state.
                    state[position] = note.pitch.alter
                elif note.pitch.alter != current:
                    result[(measure_index, staff_base + position[0])] += 1
                    state[position] = note.pitch.alter
        staff_base += max(1, int(part.staff_count))
    return result


def _part_staff_count(part: etree._Element) -> int:
    result = 1
    for value in part.xpath("./measure/attributes/staves/text()"):
        try:
            result = max(result, int(value))
        except (TypeError, ValueError):
            pass
    for value in part.xpath("./measure/note/staff/text()"):
        try:
            result = max(result, int(value))
        except (TypeError, ValueError):
            pass
    return result


def _node_staff(node: etree._Element) -> int:
    try:
        return max(1, int(node.findtext("staff") or 1))
    except (TypeError, ValueError):
        return 1


def _raw_xml_inventory(
    tree: etree._ElementTree,
) -> dict[str, Counter[tuple[int, int]]]:
    inventories = {
        class_name: Counter()
        for class_name in _COMPARABLE_CLASSES
        if class_name != "genericAccidental"
    }
    staff_base = 0
    for part in tree.getroot().findall("part"):
        for measure_index, measure in enumerate(part.findall("measure"), start=1):
            for note in measure.findall("note"):
                key = (measure_index, staff_base + _node_staff(note))
                beam_values = [
                    str(node.text or "").strip().casefold()
                    for node in note.findall("beam")
                ]
                note_type = str(note.findtext("type") or "").strip().casefold()
                counts = {
                    "arpeggio": len(note.findall("notations/arpeggiate")),
                    "augmentationDot": len(note.findall("dot")),
                    "beam": sum(
                        value in {"begin", "forward hook", "backward hook"}
                        for value in beam_values
                    ),
                    "breathMark": len(
                        note.findall("notations/articulations/breath-mark")
                    ),
                    "fermata": len(note.findall("notations/fermata")),
                    "flag": int(
                        note.find("rest") is None
                        and note.find("chord") is None
                        and note_type
                        in {
                            "eighth",
                            "16th",
                            "32nd",
                            "64th",
                            "128th",
                            "256th",
                            "512th",
                            "1024th",
                        }
                        and not beam_values
                    ),
                    "genericArticulation": len(
                        note.findall("notations/articulations/*")
                    ),
                    "genericOrnament": len(
                        [
                            node
                            for node in note.findall("notations/ornaments/*")
                            if node.tag not in {"tremolo", "wavy-line"}
                        ]
                    ),
                    "genericRest": int(note.find("rest") is not None),
                    "glissando": len(note.findall("notations/glissando")),
                    "graceSlash": len(
                        [
                            node
                            for node in note.findall("grace")
                            if str(node.get("slash") or "").casefold()
                            in {"yes", "true", "1"}
                        ]
                    ),
                    "slur": len(
                        [
                            node
                            for node in note.findall("notations/slur")
                            if str(node.get("type") or "").casefold() == "start"
                        ]
                    ),
                    "tie": len(
                        [
                            node
                            for node in note.findall("notations/tied")
                            if str(node.get("type") or "").casefold() == "start"
                        ]
                    ),
                    "tremoloBetweenNotes": len(
                        [
                            node
                            for node in note.findall(
                                "notations/ornaments/tremolo"
                            )
                            if str(node.get("type") or "").casefold() == "start"
                        ]
                    ),
                    "tremoloSingle": len(
                        [
                            node
                            for node in note.findall(
                                "notations/ornaments/tremolo"
                            )
                            if str(node.get("type") or "").casefold()
                            in {"", "single"}
                        ]
                    ),
                    "trillExtension": len(
                        [
                            node
                            for node in note.findall(
                                "notations/ornaments/wavy-line"
                            )
                            if str(node.get("type") or "").casefold() == "start"
                        ]
                    ),
                    "tuplet": len(
                        [
                            node
                            for node in note.findall("notations/tuplet")
                            if str(node.get("type") or "").casefold() == "start"
                        ]
                    ),
                }
                for class_name, count in counts.items():
                    if count:
                        inventories[class_name][key] += count

            for direction in measure.findall("direction"):
                key = (measure_index, staff_base + _node_staff(direction))
                counts = {
                    "genericDynamic": len(
                        direction.findall("direction-type/dynamics/*")
                    ),
                    "hairpin": len(
                        [
                            node
                            for node in direction.findall("direction-type/wedge")
                            if str(node.get("type") or "").casefold()
                            in {"crescendo", "diminuendo"}
                        ]
                    ),
                    "ottava": len(
                        [
                            node
                            for node in direction.findall(
                                "direction-type/octave-shift"
                            )
                            if str(node.get("type") or "").casefold()
                            not in {"stop"}
                        ]
                    ),
                    "pedal": len(
                        [
                            node
                            for node in direction.findall("direction-type/pedal")
                            if str(node.get("type") or "").casefold()
                            not in {"stop"}
                        ]
                    ),
                }
                for class_name, count in counts.items():
                    if count:
                        inventories[class_name][key] += count

            volta_count = len(
                [
                    node
                    for node in measure.findall("./barline/ending")
                    if str(node.get("type") or "").casefold() == "start"
                ]
            )
            if volta_count:
                inventories["volta"][(measure_index, staff_base + 1)] += volta_count
        staff_base += _part_staff_count(part)
    return inventories


def _totals(
    inventories: dict[str, Counter[tuple[int, int]]],
) -> dict[str, int]:
    return {
        class_name: sum(counter.values())
        for class_name, counter in inventories.items()
        if sum(counter.values())
    }


def audit_semantic_source_symbols(
    xml_path: Path,
    layout: PageLayout,
    detections: Iterable[SemanticDetection],
) -> SemanticSourceAuditReport:
    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        recover=False,
        huge_tree=False,
    )
    tree = etree.parse(str(xml_path), parser)
    score = score_from_tree(tree)
    simultaneous = sum(
        max(1, int(part.staff_count))
        for part in score.effective_parts
    )
    output_measure_count = max(
        (len(part.measures) for part in score.effective_parts),
        default=0,
    )
    staffs = _ordered_staffs(layout)
    expectation = layout.expectation_for_staff_topology(simultaneous)
    layout_measure_count = int(expectation.measure_count)
    if (
        output_measure_count <= 0
        or not staffs
        or len(staffs) % max(1, simultaneous) != 0
        or layout_measure_count != output_measure_count
    ):
        return SemanticSourceAuditReport(
            "layout_unresolved",
            simultaneous,
            output_measure_count,
            layout_measure_count,
            0,
            0,
            {},
            {},
            (),
        )

    source: dict[str, Counter[tuple[int, int]]] = defaultdict(Counter)
    mapped = 0
    unmapped = 0
    for detection in detections:
        if detection.class_name not in _COMPARABLE_CLASSES:
            continue
        position = _measure_for_detection(
            detection,
            layout,
            staffs,
            simultaneous,
        )
        if position is None or position[0] > output_measure_count:
            unmapped += 1
            continue
        source[detection.class_name][position] += 1
        mapped += 1

    output = _raw_xml_inventory(tree)
    output["genericAccidental"] = _accidental_inventory(score)
    mismatches: list[SemanticSourceMismatch] = []
    for class_name in sorted(_COMPARABLE_CLASSES):
        source_counter = source.get(class_name, Counter())
        output_counter = output.get(class_name, Counter())
        for measure_index, staff_slot in sorted(
            set(source_counter) | set(output_counter)
        ):
            key = (measure_index, staff_slot)
            source_count = int(source_counter[key])
            output_count = int(output_counter[key])
            if source_count > output_count:
                mismatches.append(
                    SemanticSourceMismatch(
                        class_name,
                        measure_index,
                        staff_slot,
                        source_count,
                        output_count,
                        "source_symbol_missing",
                    )
                )
            elif (
                class_name in POSITIONAL_INVENTORY_CLASSES
                and output_count > source_count
            ):
                mismatches.append(
                    SemanticSourceMismatch(
                        class_name,
                        measure_index,
                        staff_slot,
                        source_count,
                        output_count,
                        "source_absent_output_symbol",
                    )
                )
    return SemanticSourceAuditReport(
        "completed",
        simultaneous,
        output_measure_count,
        layout_measure_count,
        mapped,
        unmapped,
        _totals(dict(source)),
        _totals(output),
        tuple(mismatches),
    )
