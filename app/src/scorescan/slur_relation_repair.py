from __future__ import annotations

"""Repair source-proven slur relations without guessing missing arcs.

Some recognizers emit one slur start followed by two same-number stops when two
physical nested arcs share the same note.  This module repairs only that exact local
pattern and only when the source detector independently sees two contained,
same-placement curved ink components in the same mapped measure.  It also handles one
strict chain pattern where two short arcs are already balanced in MusicXML and a clean,
contained long outer source arc proves the missing outer start.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lxml import etree

from .layout import PageLayout, system_measure_bounds
from .musicxml import MUSICXML_DOCTYPE, analyze_musicxml, validate_musicxml
from .notation_coverage import VisualNotationCandidate, detect_notation_candidates
from .util import atomic_write_bytes
from .wedge_enrichment import (
    _build_topology,
    _map_anchor,
)

SLUR_RELATION_REPAIR_VERSION = "source-slur-relation-transaction@4"
MINIMUM_PAIR_CONFIDENCE = 0.88
MINIMUM_STRONG_MEMBER_CONFIDENCE = 0.95
MINIMUM_HORIZONTAL_CONTAINMENT = 0.78
CLEAN_PAIR_MINIMUM_CONFIDENCE = 0.85
CLEAN_PAIR_STRONG_MEMBER_CONFIDENCE = 0.92
CLEAN_PAIR_HORIZONTAL_CONTAINMENT = 0.95
CLEAN_PAIR_MAXIMUM_FIT_P90_SPACES = 0.08
SINGLE_ARC_MINIMUM_CONFIDENCE = 0.92
SINGLE_ARC_MINIMUM_LENGTH_SPACES = 7.0
SINGLE_ARC_MAXIMUM_FIT_P90_SPACES = 0.30
SINGLE_ARC_MAXIMUM_START_RATIO = 0.12
SINGLE_ARC_MINIMUM_END_RATIO = 0.30
SINGLE_ARC_MAXIMUM_END_RATIO = 0.55
OUTER_CHAIN_LONG_MINIMUM_CONFIDENCE = 0.97
OUTER_CHAIN_SHORT_MINIMUM_CONFIDENCE = 0.78
OUTER_CHAIN_MINIMUM_CONTAINMENT = 0.98
OUTER_CHAIN_LONG_MAXIMUM_FIT_P90_SPACES = 0.08
OUTER_CHAIN_SHORT_MAXIMUM_FIT_P90_SPACES = 0.15
OUTER_CHAIN_LONG_MINIMUM_LENGTH_SPACES = 10.0
OUTER_CHAIN_SHORT_MINIMUM_LENGTH_SPACES = 4.0
OUTER_CHAIN_MINIMUM_WIDTH_RATIO = 1.80


def _slur_confidence(candidate: VisualNotationCandidate) -> float:
    """Combine geometry only with release-gated, class-specific slur support."""

    return max(
        candidate.confidence,
        float(dict(candidate.geometry).get("semantic_slur_support", 0.0)),
    )


@dataclass(frozen=True)
class SlurRelationProposal:
    part_id: str
    part_index: int
    staff: int
    voice: str
    measure_index: int
    measure_number: str
    start_note_index: int
    first_stop_note_index: int
    orphan_stop_note_index: int
    source_candidates: tuple[VisualNotationCandidate, ...]
    eligible: bool
    repaired: bool
    reason: str
    assigned_number: int | None = None
    operation: str = "none"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_candidates"] = [
            item.to_dict()
            for item in self.source_candidates
        ]
        return payload


@dataclass(frozen=True)
class SlurRelationRepairReport:
    image_path: str
    xml_path: str
    version: str
    transaction_committed: bool
    slur_issue_count_before: int
    slur_issue_count_after: int
    proposals: tuple[SlurRelationProposal, ...]
    error: str | None = None

    @property
    def repaired_count(self) -> int:
        return sum(item.repaired for item in self.proposals)

    @property
    def abstention_count(self) -> int:
        return sum(not item.repaired for item in self.proposals)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": 1,
            "image_path": self.image_path,
            "xml_path": self.xml_path,
            "version": self.version,
            "transaction_committed": self.transaction_committed,
            "slur_issue_count_before": self.slur_issue_count_before,
            "slur_issue_count_after": self.slur_issue_count_after,
            "repaired_count": self.repaired_count,
            "abstention_count": self.abstention_count,
            "proposals": [item.to_dict() for item in self.proposals],
            "error": self.error,
        }


@dataclass
class _Segment:
    part_id: str
    part_index: int
    staff: int
    voice: str
    measure_index: int
    measure_number: str
    number: str
    start_note_index: int
    start_node: etree._Element
    first_stop_note_index: int | None = None
    first_stop_node: etree._Element | None = None
    orphan_stop_note_index: int | None = None
    orphan_stop_node: etree._Element | None = None


def _note_staff(note: etree._Element) -> int:
    try:
        return max(1, int(note.findtext("staff") or "1"))
    except (TypeError, ValueError, OverflowError):
        return 1


def _slur_segments_with_one_orphan(
    topology,
) -> list[_Segment]:
    """Return same-measure start/stop/extra-stop segments.

    Stops are processed before starts on one note, matching MusicXML's temporal
    semantics for an arc that ends and another that begins on the same event.
    """

    active: set[tuple[int, str, str]] = set()
    result: list[_Segment] = []
    for part in topology.parts:
        active.clear()
        for measure_index, measure in enumerate(part.measures):
            current: dict[tuple[int, str, str], _Segment] = {}
            notes = measure.findall("note")
            for note_index, note in enumerate(notes):
                staff = _note_staff(note)
                voice = str(note.findtext("voice") or "1")
                slurs = note.findall("./notations/slur")
                for slur in (item for item in slurs if item.get("type") == "stop"):
                    number = str(slur.get("number") or "1")
                    key = (staff, voice, number)
                    segment = current.get(key)
                    if key in active:
                        active.remove(key)
                        if segment is not None and segment.first_stop_note_index is None:
                            segment.first_stop_note_index = note_index
                            segment.first_stop_node = slur
                    elif (
                        segment is not None
                        and segment.first_stop_note_index is not None
                        and segment.orphan_stop_note_index is None
                    ):
                        segment.orphan_stop_note_index = note_index
                        segment.orphan_stop_node = slur
                for slur in (item for item in slurs if item.get("type") == "start"):
                    number = str(slur.get("number") or "1")
                    key = (staff, voice, number)
                    previous = current.get(key)
                    if (
                        previous is not None
                        and previous.first_stop_note_index is not None
                        and previous.orphan_stop_note_index is not None
                        and previous.orphan_stop_node is not None
                    ):
                        result.append(previous)
                    active.add(key)
                    current[key] = _Segment(
                        part.id,
                        part.index,
                        staff,
                        voice,
                        measure_index,
                        str(measure.get("number") or measure_index + 1),
                        number,
                        note_index,
                        slur,
                    )
            for segment in current.values():
                if (
                    segment.first_stop_note_index is not None
                    and segment.orphan_stop_note_index is not None
                    and segment.orphan_stop_node is not None
                ):
                    result.append(segment)
    return result


def _horizontal_containment(
    left: VisualNotationCandidate,
    right: VisualNotationCandidate,
) -> float:
    overlap = max(
        0,
        min(left.bbox[2], right.bbox[2]) - max(left.bbox[0], right.bbox[0]),
    )
    minimum_width = max(
        1,
        min(left.bbox[2] - left.bbox[0], right.bbox[2] - right.bbox[0]),
    )
    return overlap / minimum_width


def _best_nested_pair(
    candidates: list[VisualNotationCandidate],
) -> tuple[VisualNotationCandidate, VisualNotationCandidate] | None:
    eligible: list[
        tuple[float, float, VisualNotationCandidate, VisualNotationCandidate]
    ] = []
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1:]:
            if left.placement != right.placement:
                continue
            containment = _horizontal_containment(left, right)
            minimum_confidence = min(_slur_confidence(left), _slur_confidence(right))
            maximum_confidence = max(_slur_confidence(left), _slur_confidence(right))
            strict_pair = (
                containment >= MINIMUM_HORIZONTAL_CONTAINMENT
                and minimum_confidence >= MINIMUM_PAIR_CONFIDENCE
                and maximum_confidence >= MINIMUM_STRONG_MEMBER_CONFIDENCE
            )
            left_geometry = dict(left.geometry)
            right_geometry = dict(right.geometry)
            fit_values = (
                left_geometry.get("fit_p90_spaces"),
                right_geometry.get("fit_p90_spaces"),
            )
            clean_pair = (
                all(value is not None for value in fit_values)
                and containment >= CLEAN_PAIR_HORIZONTAL_CONTAINMENT
                and minimum_confidence >= CLEAN_PAIR_MINIMUM_CONFIDENCE
                and maximum_confidence >= CLEAN_PAIR_STRONG_MEMBER_CONFIDENCE
                and max(float(value) for value in fit_values if value is not None)
                <= CLEAN_PAIR_MAXIMUM_FIT_P90_SPACES
            )
            if not (strict_pair or clean_pair):
                continue
            eligible.append(
                (
                    containment,
                    minimum_confidence,
                    left,
                    right,
                )
            )
    if not eligible:
        return None
    _containment, _confidence, left, right = max(
        eligible,
        key=lambda item: (
            item[0],
            item[1],
            max(_slur_confidence(item[2]), _slur_confidence(item[3])),
            item[2].bbox,
            item[3].bbox,
        ),
    )
    return left, right


def _best_outer_chain_pair(
    candidates: list[VisualNotationCandidate],
) -> tuple[VisualNotationCandidate, VisualNotationCandidate] | None:
    """Find one clean long outer arc containing one corroborating short arc."""

    eligible: list[
        tuple[float, float, VisualNotationCandidate, VisualNotationCandidate]
    ] = []
    for long_arc in candidates:
        long_geometry = dict(long_arc.geometry)
        long_fit = long_geometry.get("fit_p90_spaces")
        long_length = long_geometry.get("length_spaces")
        if (
            long_fit is None
            or long_length is None
            or _slur_confidence(long_arc) < OUTER_CHAIN_LONG_MINIMUM_CONFIDENCE
            or float(long_fit) > OUTER_CHAIN_LONG_MAXIMUM_FIT_P90_SPACES
            or float(long_length) < OUTER_CHAIN_LONG_MINIMUM_LENGTH_SPACES
        ):
            continue
        long_width = max(1, long_arc.bbox[2] - long_arc.bbox[0])
        for short_arc in candidates:
            if short_arc is long_arc or short_arc.placement != long_arc.placement:
                continue
            short_geometry = dict(short_arc.geometry)
            short_fit = short_geometry.get("fit_p90_spaces")
            short_length = short_geometry.get("length_spaces")
            short_width = max(1, short_arc.bbox[2] - short_arc.bbox[0])
            containment = _horizontal_containment(long_arc, short_arc)
            if (
                short_fit is None
                or short_length is None
                or _slur_confidence(short_arc) < OUTER_CHAIN_SHORT_MINIMUM_CONFIDENCE
                or float(short_fit) > OUTER_CHAIN_SHORT_MAXIMUM_FIT_P90_SPACES
                or float(short_length) < OUTER_CHAIN_SHORT_MINIMUM_LENGTH_SPACES
                or containment < OUTER_CHAIN_MINIMUM_CONTAINMENT
                or long_width / short_width < OUTER_CHAIN_MINIMUM_WIDTH_RATIO
            ):
                continue
            eligible.append(
                (
                    containment,
                    min(long_arc.confidence, short_arc.confidence),
                    long_arc,
                    short_arc,
                )
            )
    if len(eligible) != 1:
        return None
    _, _, long_arc, short_arc = eligible[0]
    return long_arc, short_arc


def _is_missing_outer_chain_pattern(
    measure: etree._Element,
    segment: _Segment,
) -> bool:
    """Match two balanced adjacent arcs followed by one orphan outer stop."""

    if (
        segment.start_note_index < 1
        or segment.first_stop_note_index != segment.start_note_index + 1
        or segment.orphan_stop_note_index != segment.start_note_index + 2
    ):
        return False
    notes = measure.findall("note")
    prior_index = segment.start_note_index - 1
    if segment.orphan_stop_note_index >= len(notes):
        return False
    number = segment.number

    def slur_types(note_index: int) -> list[tuple[str, str]]:
        return [
            (str(node.get("type") or ""), str(node.get("number") or "1"))
            for node in notes[note_index].findall("./notations/slur")
        ]

    return (
        slur_types(prior_index) == [("start", number)]
        and slur_types(segment.start_note_index)
        == [("stop", number), ("start", number)]
        and slur_types(int(segment.first_stop_note_index)) == [("stop", number)]
        and slur_types(int(segment.orphan_stop_note_index)) == [("stop", number)]
    )


def _mapped_curves_by_measure(
    candidates: tuple[VisualNotationCandidate, ...],
    topology,
    layout: PageLayout,
) -> dict[tuple[int, int, int], list[VisualNotationCandidate]]:
    grouped: dict[tuple[int, int, int], list[VisualNotationCandidate]] = {}
    for candidate in candidates:
        if candidate.kind != "curved_connector":
            continue
        center_x = 0.5 * (candidate.bbox[0] + candidate.bbox[2])
        anchor, error = _map_anchor(topology, layout, candidate, center_x)
        if anchor is None or error is not None:
            continue
        grouped.setdefault(
            (anchor.part_index, anchor.staff, anchor.measure_index),
            [],
        ).append(candidate)
    return grouped


def _exact_source_measure_geometry(
    topology,
    segment: _Segment,
) -> tuple[int, int, float] | None:
    """Return source bounds only when visual and MusicXML system counts agree."""

    part = topology.parts[segment.part_index]
    group_index = next(
        (
            index
            for index, (offset, count) in enumerate(
                zip(part.system_measure_offsets, part.system_measure_counts, strict=True)
            )
            if offset <= segment.measure_index < offset + count
        ),
        None,
    )
    if group_index is None or group_index >= len(topology.score_systems):
        return None
    target_count = part.system_measure_counts[group_index]
    if topology.group_measure_counts[group_index] != target_count:
        return None
    source_group = topology.score_systems[group_index]
    source_index = next(
        (
            staff_index
            for staff_index, mapped_group, part_index, part_staff in (
                topology.appearance_locations
            )
            if (
                mapped_group == group_index
                and part_index == segment.part_index
                and part_staff == segment.staff
            )
        ),
        None,
    )
    if source_index is None:
        return None
    source_staff = next(
        (
            staff
            for staff in source_group
            if staff.index == source_index
        ),
        None,
    )
    if source_staff is None:
        return None
    bounds = system_measure_bounds(source_staff)
    if len(bounds) != target_count:
        return None
    local_index = segment.measure_index - part.system_measure_offsets[group_index]
    if not 0 <= local_index < len(bounds):
        return None
    left, right = bounds[local_index]
    return int(left), int(right), float(source_staff.spacing)


def _best_single_spanning_curve(
    candidates: list[VisualNotationCandidate],
    topology,
    segment: _Segment,
) -> VisualNotationCandidate | None:
    """Find one source arc proving that the earlier duplicate stop is premature."""

    if (
        segment.start_note_index != 0
        or segment.first_stop_note_index != 1
        or segment.orphan_stop_note_index != 2
        or segment.first_stop_node is None
    ):
        return None
    geometry = _exact_source_measure_geometry(topology, segment)
    if geometry is None:
        return None
    left, right, spacing = geometry
    width = max(1.0, float(right - left))
    eligible: list[VisualNotationCandidate] = []
    for candidate in candidates:
        values = dict(candidate.geometry)
        fit = values.get("fit_p90_spaces")
        length = values.get("length_spaces")
        if fit is None or length is None:
            continue
        start_ratio = (candidate.bbox[0] - left) / width
        end_ratio = (candidate.bbox[2] - left) / width
        bbox_length_spaces = (candidate.bbox[2] - candidate.bbox[0]) / max(spacing, 1.0)
        if (
            _slur_confidence(candidate) >= SINGLE_ARC_MINIMUM_CONFIDENCE
            and float(fit) <= SINGLE_ARC_MAXIMUM_FIT_P90_SPACES
            and min(float(length), bbox_length_spaces)
            >= SINGLE_ARC_MINIMUM_LENGTH_SPACES
            and -0.03 <= start_ratio <= SINGLE_ARC_MAXIMUM_START_RATIO
            and SINGLE_ARC_MINIMUM_END_RATIO
            <= end_ratio
            <= SINGLE_ARC_MAXIMUM_END_RATIO
        ):
            eligible.append(candidate)
    return eligible[0] if len(eligible) == 1 else None


def _unused_slur_number(
    measure: etree._Element,
    reserved: set[int] | None = None,
) -> int | None:
    used = {
        int(number)
        for node in measure.findall("./note/notations/slur")
        if (number := node.get("number")) is not None and number.isdigit()
    }
    used.update(reserved or ())
    return next((number for number in range(1, 7) if number not in used), None)


def _guard_snapshot(path: Path) -> dict[str, Any]:
    analysis = analyze_musicxml(path)
    return {
        "part_count": analysis.get("part_count"),
        "measure_count": analysis.get("measure_count"),
        "part_measure_counts": analysis.get("part_measure_counts"),
        "note_count": analysis.get("note_count"),
        "rest_count": analysis.get("rest_count"),
        "rhythm_issues": analysis.get("rhythm_issues"),
        "tie_issues": analysis.get("tie_issues"),
        "semantic_issues": analysis.get("semantic_issues"),
    }


def repair_source_proven_nested_slurs(
    image_path: Path,
    xml_path: Path,
    layout: PageLayout,
    *,
    candidates: tuple[VisualNotationCandidate, ...] | None = None,
) -> SlurRelationRepairReport:
    image_path = image_path.resolve()
    xml_path = xml_path.resolve()
    detected = candidates if candidates is not None else detect_notation_candidates(image_path, layout)
    tree = etree.parse(
        str(xml_path),
        etree.XMLParser(resolve_entities=False, no_network=True),
    )
    root = tree.getroot()
    before_analysis = analyze_musicxml(xml_path)
    before_issues = list(before_analysis.get("slur_issues", []) or [])
    topology, topology_error = _build_topology(root, layout)
    if topology is None:
        return SlurRelationRepairReport(
            str(image_path),
            str(xml_path),
            SLUR_RELATION_REPAIR_VERSION,
            False,
            len(before_issues),
            len(before_issues),
            (),
            topology_error,
        )
    curves = _mapped_curves_by_measure(detected, topology, layout)
    available_curves = {
        key: list(items)
        for key, items in curves.items()
    }
    proposals: list[SlurRelationProposal] = []
    repair_nodes: list[
        tuple[_Segment, tuple[VisualNotationCandidate, ...], int | None, str]
    ] = []
    reserved_numbers: dict[tuple[int, int], set[int]] = {}
    for segment in _slur_segments_with_one_orphan(topology):
        source_candidates = available_curves.get(
            (segment.part_index, segment.staff, segment.measure_index),
            [],
        )
        pair = _best_nested_pair(source_candidates)
        measure = topology.parts[segment.part_index].measures[segment.measure_index]
        outer_chain_pair = (
            None
            if pair is not None
            or not _is_missing_outer_chain_pattern(measure, segment)
            else _best_outer_chain_pair(source_candidates)
        )
        single_curve = (
            None
            if pair is not None or outer_chain_pair is not None
            else _best_single_spanning_curve(source_candidates, topology, segment)
        )
        reservation_key = (segment.part_index, segment.measure_index)
        reserved = reserved_numbers.setdefault(reservation_key, set())
        number = _unused_slur_number(measure, reserved)
        reason = None
        if (pair is not None or outer_chain_pair is not None) and number is None:
            reason = "no free MusicXML slur number is available"
        elif pair is None and outer_chain_pair is None and single_curve is None:
            reason = (
                "source image proves neither a nested pair, a missing outer arc, "
                "nor one exact spanning arc"
            )
        if reason is not None:
            proposals.append(
                SlurRelationProposal(
                    segment.part_id,
                    segment.part_index,
                    segment.staff,
                    segment.voice,
                    segment.measure_index,
                    segment.measure_number,
                    segment.start_note_index,
                    int(segment.first_stop_note_index),
                    int(segment.orphan_stop_note_index),
                    tuple(pair or outer_chain_pair or ()),
                    False,
                    False,
                    reason,
                )
            )
            continue
        operation = (
            "renumber_nested"
            if pair is not None
            else (
                "add_outer_arc"
                if outer_chain_pair is not None
                else "extend_single_arc"
            )
        )
        selected_source = tuple(pair or outer_chain_pair or (single_curve,))
        assigned_number = (
            number
            if pair is not None or outer_chain_pair is not None
            else None
        )
        if pair is not None or outer_chain_pair is not None:
            assert number is not None
            reserved.add(number)
        for member in selected_source:
            assert member is not None
            source_candidates.remove(member)
        repair_nodes.append((segment, selected_source, assigned_number, operation))
        proposals.append(
            SlurRelationProposal(
                segment.part_id,
                segment.part_index,
                segment.staff,
                segment.voice,
                segment.measure_index,
                segment.measure_number,
                segment.start_note_index,
                int(segment.first_stop_note_index),
                int(segment.orphan_stop_note_index),
                selected_source,
                True,
                False,
                "eligible",
                assigned_number,
                operation,
            )
        )

    if not repair_nodes:
        return SlurRelationRepairReport(
            str(image_path),
            str(xml_path),
            SLUR_RELATION_REPAIR_VERSION,
            False,
            len(before_issues),
            len(before_issues),
            tuple(proposals),
        )

    for segment, _source, number, operation in repair_nodes:
        if operation == "renumber_nested":
            parent = segment.start_node.getparent()
            if parent is None or number is None:
                raise ValueError("nested slur start has no valid parent or number")
            new_start = etree.Element(
                "slur",
                type="start",
                number=str(number),
            )
            parent.insert(parent.index(segment.start_node) + 1, new_start)
            assert segment.orphan_stop_node is not None
            segment.orphan_stop_node.set("number", str(number))
        elif operation == "add_outer_arc":
            if number is None or segment.start_note_index < 1:
                raise ValueError("missing outer slur has no valid start or number")
            measure = topology.parts[segment.part_index].measures[
                segment.measure_index
            ]
            prior_note = measure.findall("note")[segment.start_note_index - 1]
            notations = prior_note.find("notations")
            if notations is None:
                raise ValueError("missing outer slur start note has no notations")
            existing = prior_note.findall("./notations/slur")
            insert_at = (
                notations.index(existing[-1]) + 1
                if existing
                else len(notations)
            )
            notations.insert(
                insert_at,
                etree.Element("slur", type="start", number=str(number)),
            )
            assert segment.orphan_stop_node is not None
            segment.orphan_stop_node.set("number", str(number))
        elif operation == "extend_single_arc":
            assert segment.first_stop_node is not None
            parent = segment.first_stop_node.getparent()
            if parent is None:
                raise ValueError("premature slur stop has no notations parent")
            parent.remove(segment.first_stop_node)
        else:
            raise ValueError(f"unknown slur relation operation: {operation}")

    temporary = xml_path.with_name(xml_path.name + ".slur-transaction.tmp")
    try:
        temporary.write_bytes(
            etree.tostring(
                tree,
                encoding="UTF-8",
                xml_declaration=True,
                doctype=tree.docinfo.doctype or MUSICXML_DOCTYPE,
                pretty_print=True,
            )
        )
        errors = validate_musicxml(temporary)
        after_analysis = analyze_musicxml(temporary) if not errors else {}
        after_issues = list(after_analysis.get("slur_issues", []) or [])
        if errors:
            raise ValueError("; ".join(errors))
        if _guard_snapshot(xml_path) != _guard_snapshot(temporary):
            raise ValueError("non-slur MusicXML semantics changed during repair")
        expected_after = len(before_issues) - len(repair_nodes)
        if len(after_issues) != expected_after:
            raise ValueError(
                f"slur issues changed from {len(before_issues)} to {len(after_issues)}, "
                f"expected {expected_after}"
            )
        atomic_write_bytes(xml_path, temporary.read_bytes())
    except Exception as exc:
        return SlurRelationRepairReport(
            str(image_path),
            str(xml_path),
            SLUR_RELATION_REPAIR_VERSION,
            False,
            len(before_issues),
            len(before_issues),
            tuple(
                SlurRelationProposal(
                    item.part_id,
                    item.part_index,
                    item.staff,
                    item.voice,
                    item.measure_index,
                    item.measure_number,
                    item.start_note_index,
                    item.first_stop_note_index,
                    item.orphan_stop_note_index,
                    item.source_candidates,
                    item.eligible,
                    False,
                    (
                        f"transaction rolled back: {type(exc).__name__}: {exc}"
                        if item.eligible
                        else item.reason
                    ),
                    item.assigned_number,
                    item.operation,
                )
                for item in proposals
            ),
            f"{type(exc).__name__}: {exc}",
        )
    finally:
        temporary.unlink(missing_ok=True)

    return SlurRelationRepairReport(
        str(image_path),
        str(xml_path),
        SLUR_RELATION_REPAIR_VERSION,
        True,
        len(before_issues),
        len(after_issues),
        tuple(
            SlurRelationProposal(
                item.part_id,
                item.part_index,
                item.staff,
                item.voice,
                item.measure_index,
                item.measure_number,
                item.start_note_index,
                item.first_stop_note_index,
                item.orphan_stop_note_index,
                item.source_candidates,
                item.eligible,
                item.eligible,
                (
                    "source-proven nested slur numbering committed"
                    if item.operation == "renumber_nested"
                    else (
                        "source-proven missing outer slur start committed"
                        if item.operation == "add_outer_arc"
                        else (
                            "source-proven single arc endpoint committed"
                            if item.operation == "extend_single_arc"
                            else item.reason
                        )
                    )
                ),
                item.assigned_number,
                item.operation,
            )
            for item in proposals
        ),
    )
