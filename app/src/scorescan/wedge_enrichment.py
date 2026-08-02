from __future__ import annotations

"""Source-backed, transaction-safe MusicXML hairpin enrichment.

The notation coverage detector answers whether a physical hairpin is visible.  This
module separately proves where that object belongs in a recognized score.  It writes
very-high-confidence candidates, plus geometrically clean medium-confidence candidates
that do not overlap independently detected curves, when the complete repeated-system
topology can be mapped to MusicXML parts/staves.  It then verifies that all non-wedge
semantics are unchanged.  Unsupported or ambiguous cases are explicit abstentions.
"""

from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from lxml import etree

from .layout import (
    PageLayout,
    ScoreSystemLayout,
    StaffSystem,
    anchor_x_to_measure,
    system_measure_bounds,
)
from .musicxml import MUSICXML_DOCTYPE, analyze_musicxml, validate_musicxml
from .notation_coverage import (
    DETECTOR_VERSION,
    VisualNotationCandidate,
    detect_notation_candidates,
    wedge_source_specificity_gate,
)
from .util import atomic_write_bytes

WEDGE_ENRICHMENT_VERSION = "source-wedge-transaction@5"
MINIMUM_MAPPING_CONFIDENCE = 0.76
MAXIMUM_PHYSICAL_STAVES = 16


@dataclass(frozen=True)
class WedgeAnchor:
    part_id: str
    part_index: int
    staff: int
    measure_index: int
    offset_ratio: float
    confidence: float
    method: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WedgeProposal:
    candidate: VisualNotationCandidate
    start: WedgeAnchor | None
    end: WedgeAnchor | None
    eligible: bool
    injected: bool
    reason: str
    number: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "start": self.start.to_dict() if self.start is not None else None,
            "end": self.end.to_dict() if self.end is not None else None,
            "eligible": self.eligible,
            "injected": self.injected,
            "reason": self.reason,
            "number": self.number,
        }


@dataclass(frozen=True)
class WedgeEnrichmentReport:
    image_path: str
    xml_path: str
    detector_version: str
    enrichment_version: str
    transaction_committed: bool
    existing_wedge_count: int
    physical_staff_count: int
    physical_staff_appearance_count: int
    score_system_count: int
    proposals: tuple[WedgeProposal, ...]
    error: str | None = None

    @property
    def injected_count(self) -> int:
        return sum(item.injected for item in self.proposals)

    @property
    def abstention_count(self) -> int:
        return sum(not item.injected for item in self.proposals)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": 1,
            "image_path": self.image_path,
            "xml_path": self.xml_path,
            "detector_version": self.detector_version,
            "enrichment_version": self.enrichment_version,
            "transaction_committed": self.transaction_committed,
            "existing_wedge_count": self.existing_wedge_count,
            "physical_staff_count": self.physical_staff_count,
            "physical_staff_appearance_count": self.physical_staff_appearance_count,
            "score_system_count": self.score_system_count,
            "injected_count": self.injected_count,
            "abstention_count": self.abstention_count,
            "proposals": [item.to_dict() for item in self.proposals],
            "error": self.error,
        }


@dataclass(frozen=True)
class _PartTopology:
    index: int
    element: etree._Element
    id: str
    staff_count: int
    measures: tuple[etree._Element, ...]
    system_measure_counts: tuple[int, ...]
    system_measure_offsets: tuple[int, ...]


@dataclass(frozen=True)
class _Topology:
    parts: tuple[_PartTopology, ...]
    physical_staff_count: int
    ordered_appearances: tuple[StaffSystem, ...]
    score_systems: tuple[tuple[StaffSystem, ...], ...]
    group_measure_counts: tuple[int, ...]
    appearance_locations: tuple[tuple[int, int, int, int], ...]


def _integer(value: str | None, default: int = 0) -> int:
    try:
        return int((value or "").strip())
    except (TypeError, ValueError, OverflowError):
        return default


def _part_staff_count(part: etree._Element) -> int:
    count = 1
    for node in part.findall("./measure/attributes/staves"):
        count = max(count, _integer(node.text, 1))
    for node in part.findall("./measure/note/staff"):
        count = max(count, _integer(node.text, 1))
    for node in part.findall("./measure/direction/staff"):
        count = max(count, _integer(node.text, 1))
    return count


def _part_system_measure_groups(
    part: etree._Element,
) -> tuple[tuple[etree._Element, ...], ...]:
    groups: list[list[etree._Element]] = []
    for measure in part.findall("measure"):
        print_node = measure.find("print")
        if (
            groups
            and print_node is not None
            and print_node.get("new-system") == "yes"
        ):
            groups.append([])
        if not groups:
            groups.append([])
        groups[-1].append(measure)
    return tuple(tuple(group) for group in groups if group)


def _conditioned_staff_appearances(
    ordered: tuple[StaffSystem, ...],
    *,
    physical_staff_count: int,
    score_system_count: int,
) -> tuple[StaffSystem, ...]:
    """Remove at most two isolated false staff combs using recognized topology.

    The layout detector remains recognition-independent.  Once MusicXML has
    established the simultaneous staff count and system count, however, an extra
    page-fold/text comb must not shift every later source annotation.  Candidate
    subsets are accepted only when their repeated within-system vertical geometry
    is decisively more self-consistent than every alternative.
    """

    expected = physical_staff_count * score_system_count
    extra = len(ordered) - expected
    if expected <= 0:
        return ordered
    if extra < 0:
        completed = _complete_repeated_two_staff_appearances(
            ordered,
            score_system_count=score_system_count,
        )
        return completed
    if extra == 0 or extra > 2:
        return ordered
    ranked: list[tuple[float, tuple[StaffSystem, ...]]] = []
    for indices in combinations(range(len(ordered)), expected):
        selected = tuple(ordered[index] for index in indices)
        groups = [
            selected[start:start + physical_staff_count]
            for start in range(0, expected, physical_staff_count)
        ]
        spacing = float(
            np.median(
                [
                    max(1.0, float(staff.spacing))
                    for staff in selected
                ]
            )
        )
        centres = [
            [
                float(np.mean(staff.line_y))
                for staff in group
            ]
            for group in groups
        ]
        loss = float(
            np.std([staff.spacing for staff in selected])
            / max(spacing, 1.0)
        )
        within_patterns: list[list[float]] = []
        if physical_staff_count > 1:
            for values in centres:
                gaps = [
                    (right - left) / spacing
                    for left, right in zip(values, values[1:], strict=False)
                ]
                if any(gap < 3.0 or gap > 30.0 for gap in gaps):
                    loss += 10.0
                within_patterns.append(gaps)
            for slot in range(physical_staff_count - 1):
                values = [
                    pattern[slot]
                    for pattern in within_patterns
                ]
                loss += float(
                    np.std(values) / max(float(np.mean(values)), 1.0)
                )
        between_gaps: list[float] = []
        for left_group, right_group in zip(
            groups,
            groups[1:],
            strict=False,
        ):
            between_gaps.append(
                (
                    float(np.mean(right_group[0].line_y))
                    - float(np.mean(left_group[-1].line_y))
                )
                / spacing
            )
        if between_gaps and within_patterns:
            typical_within = float(np.median(within_patterns))
            loss += sum(
                max(0.0, typical_within * 1.05 - gap)
                for gap in between_gaps
            )
        removed = [
            ordered[index]
            for index in range(len(ordered))
            if index not in indices
        ]
        # Removing a staff with multiple coherent barlines is expensive; isolated
        # page-fold combs normally have none or one.
        loss += sum(
            0.08 * min(4, len(staff.barlines))
            + 0.03 * min(4, max(1, int(staff.measure_count)))
            for staff in removed
        )
        ranked.append((loss, selected))
    ranked.sort(
        key=lambda item: (
            item[0],
            tuple(staff.index for staff in item[1]),
        )
    )
    if not ranked:
        return ordered
    if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < 0.12:
        return ordered
    return ranked[0][1]


def _complete_repeated_two_staff_appearances(
    ordered: tuple[StaffSystem, ...],
    *,
    score_system_count: int,
) -> tuple[StaffSystem, ...]:
    """Fill decisively located damaged staves in a repeated two-staff score."""

    expected = 2 * score_system_count
    missing = expected - len(ordered)
    if (
        not ordered
        or score_system_count < 3
        or missing <= 0
        or missing > score_system_count
    ):
        return ordered
    centres = [
        float(np.mean(staff.line_y))
        for staff in ordered
    ]
    spacing = float(np.median([max(1.0, staff.spacing) for staff in ordered]))
    candidate_gaps = sorted(
        {
            round(right - left, 3)
            for left, right in zip(centres, centres[1:], strict=False)
            if 6.0 * spacing <= right - left <= 16.0 * spacing
        }
    )
    solutions: list[
        tuple[
            float,
            float,
            list[tuple[tuple[StaffSystem, ...], tuple[int, ...]]],
        ]
    ] = []
    for within_gap in candidate_gaps:
        tolerance = max(3.0 * spacing, 0.18 * within_gap)
        raw_groups: list[tuple[StaffSystem, ...]] = []
        index = 0
        while index < len(ordered):
            if (
                index + 1 < len(ordered)
                and abs(
                    float(np.mean(ordered[index + 1].line_y))
                    - float(np.mean(ordered[index].line_y))
                    - within_gap
                )
                <= tolerance
            ):
                raw_groups.append((ordered[index], ordered[index + 1]))
                index += 2
            else:
                raw_groups.append((ordered[index],))
                index += 1
        if len(raw_groups) != score_system_count:
            continue
        singleton_indices = [
            index
            for index, group in enumerate(raw_groups)
            if len(group) == 1
        ]
        if len(singleton_indices) != missing:
            continue
        for singleton_slots in product((0, 1), repeat=len(singleton_indices)):
            slot_by_group = dict(zip(singleton_indices, singleton_slots, strict=True))
            mapped_groups: list[
                tuple[tuple[StaffSystem, ...], tuple[int, ...]]
            ] = []
            bases: list[float] = []
            complete_gaps: list[float] = []
            valid = True
            for group_index, group in enumerate(raw_groups):
                if len(group) == 2:
                    slots = (0, 1)
                    first = float(np.mean(group[0].line_y))
                    second = float(np.mean(group[1].line_y))
                    complete_gaps.append(second - first)
                    base = first
                else:
                    slot = slot_by_group[group_index]
                    slots = (slot,)
                    center = float(np.mean(group[0].line_y))
                    base = center - slot * within_gap
                if bases and base - (bases[-1] + within_gap) < 3.0 * spacing:
                    valid = False
                    break
                bases.append(base)
                mapped_groups.append((group, slots))
            if not valid:
                continue
            base_gaps = np.diff(bases)
            if len(base_gaps) < 2 or min(base_gaps) <= 0:
                continue
            loss = float(
                np.std(base_gaps) / max(float(np.mean(base_gaps)), 1.0)
            )
            if complete_gaps:
                loss += float(
                    np.mean(
                        [
                            abs(value - within_gap) / within_gap
                            for value in complete_gaps
                        ]
                    )
                )
            solutions.append((loss, within_gap, mapped_groups))
    solutions.sort(
        key=lambda item: (
            item[0],
            item[1],
            tuple(
                slots
                for _group, slots in item[2]
            ),
        )
    )
    if not solutions:
        return ordered
    if len(solutions) > 1 and solutions[1][0] - solutions[0][0] < 0.08:
        return ordered

    _loss, within_gap, mapped_groups = solutions[0]
    next_index = max((staff.index for staff in ordered), default=0) + 1
    result: list[StaffSystem] = []
    for group, slots in mapped_groups:
        result.extend(group)
        if len(group) == 2:
            continue
        observed = group[0]
        missing_slot = 1 - slots[0]
        observed_center = float(np.mean(observed.line_y))
        missing_center = observed_center + (
            within_gap if missing_slot == 1 else -within_gap
        )
        line_y = [
            missing_center + (offset - 2) * spacing
            for offset in range(5)
        ]
        result.append(
            StaffSystem(
                index=next_index,
                line_y=line_y,
                top=max(0, int(round(line_y[0] - spacing * 4.2))),
                bottom=int(round(line_y[-1] + spacing * 4.2)),
                left=observed.left,
                right=observed.right,
                spacing=spacing,
                barlines=list(observed.barlines),
                measure_count=observed.measure_count,
                barline_confidences=list(observed.barline_confidences),
                barline_sequence_confidences=list(
                    observed.barline_sequence_confidences
                ),
            )
        )
        next_index += 1
    return tuple(
        sorted(
            result,
            key=lambda staff: (
                float(np.mean(staff.line_y)),
                staff.index,
            ),
        )
    )


def _hidden_empty_part_visibility(
    root: etree._Element,
    parts: tuple[_PartTopology, ...],
) -> dict[int, tuple[bool, ...]]:
    score_parts = {
        str(item.get("id") or ""): item
        for item in root.findall("./part-list/score-part")
    }
    result: dict[int, tuple[bool, ...]] = {}
    for part in parts:
        score_part = score_parts.get(part.id)
        part_name = (
            score_part.find("part-name")
            if score_part is not None
            else None
        )
        if (
            part.staff_count != 1
            or part_name is None
            or part_name.get("print-object") != "no"
        ):
            continue
        groups = _part_system_measure_groups(part.element)
        visible = tuple(
            any(
                note.find("rest") is None
                and (
                    note.find("pitch") is not None
                    or note.find("unpitched") is not None
                )
                for measure in group
                for note in measure.findall("note")
            )
            for group in groups
        )
        if any(visible) and not all(visible):
            result[part.index] = visible
    return result


def _synthetic_trailing_staff(
    source_group: tuple[StaffSystem, ...],
    *,
    center_gap: float,
    index: int,
) -> StaffSystem:
    observed = source_group[-1]
    spacing = float(
        np.median([max(1.0, staff.spacing) for staff in source_group])
    )
    center = float(np.mean(observed.line_y)) + center_gap
    line_y = [center + (offset - 2) * spacing for offset in range(5)]
    return StaffSystem(
        index=index,
        line_y=line_y,
        top=max(0, int(round(line_y[0] - spacing * 4.2))),
        bottom=int(round(line_y[-1] + spacing * 4.2)),
        left=observed.left,
        right=observed.right,
        spacing=spacing,
        barlines=list(observed.barlines),
        measure_count=observed.measure_count,
        barline_confidences=list(observed.barline_confidences),
        barline_sequence_confidences=list(
            observed.barline_sequence_confidences
        ),
    )


def _variable_visible_topology(
    root: etree._Element,
    layout: PageLayout,
    parts: tuple[_PartTopology, ...],
    *,
    hidden_visibility: dict[int, tuple[bool, ...]],
) -> tuple[
    tuple[StaffSystem, ...],
    tuple[tuple[StaffSystem, ...], ...],
    tuple[tuple[int, int, int, int], ...],
] | None:
    system_counts = {
        len(part.system_measure_counts)
        for part in parts
    }
    if len(system_counts) != 1:
        return None
    score_system_count = next(iter(system_counts))
    source_score_systems = layout.effective_score_systems
    if len(source_score_systems) != score_system_count:
        return None
    staff_by_index = {
        staff.index: staff
        for staff in layout.systems
    }
    source_groups = [
        tuple(
            staff_by_index[index]
            for index in score_system.staff_indices
            if index in staff_by_index
        )
        for score_system in source_score_systems
    ]
    adjacent_gaps = [
        float(np.mean(right.line_y)) - float(np.mean(left.line_y))
        for group in source_groups
        for left, right in zip(group, group[1:], strict=False)
        if float(np.mean(right.line_y)) > float(np.mean(left.line_y))
    ]
    if not adjacent_gaps:
        return None
    typical_gap = float(np.median(adjacent_gaps))
    next_index = max(staff_by_index, default=0) + 1
    completed_groups: list[tuple[StaffSystem, ...]] = []
    locations: list[tuple[int, int, int, int]] = []
    for group_index, source_group in enumerate(source_groups):
        visible_slots = [
            (part.index, staff)
            for part in parts
            for staff in range(1, part.staff_count + 1)
            if (
                part.index not in hidden_visibility
                or hidden_visibility[part.index][group_index]
            )
        ]
        completed = source_group
        if len(completed) + 1 == len(visible_slots):
            missing_part_index, missing_staff = visible_slots[-1]
            if (
                missing_part_index not in hidden_visibility
                or missing_staff != 1
                or not completed
            ):
                return None
            completed = (
                *completed,
                _synthetic_trailing_staff(
                    completed,
                    center_gap=typical_gap,
                    index=next_index,
                ),
            )
            next_index += 1
        if len(completed) != len(visible_slots):
            return None
        completed_groups.append(tuple(completed))
        locations.extend(
            (
                staff.index,
                group_index,
                part_index,
                part_staff,
            )
            for staff, (part_index, part_staff) in zip(
                completed,
                visible_slots,
                strict=True,
            )
        )
    ordered = tuple(
        staff
        for group in completed_groups
        for staff in group
    )
    return ordered, tuple(completed_groups), tuple(locations)


def _build_topology(root: etree._Element, layout: PageLayout) -> tuple[_Topology | None, str | None]:
    parts_list: list[_PartTopology] = []
    for index, part in enumerate(root.findall("part")):
        measures = tuple(part.findall("measure"))
        groups = _part_system_measure_groups(part)
        counts = tuple(len(group) for group in groups)
        offsets: list[int] = []
        running = 0
        for count in counts:
            offsets.append(running)
            running += count
        parts_list.append(
            _PartTopology(
                index=index,
                element=part,
                id=str(part.get("id") or f"P{index + 1}"),
                staff_count=_part_staff_count(part),
                measures=measures,
                system_measure_counts=counts,
                system_measure_offsets=tuple(offsets),
            )
        )
    parts = tuple(parts_list)
    if not parts or any(not part.measures for part in parts):
        return None, "MusicXML part/measure topology is incomplete"
    physical_count = sum(part.staff_count for part in parts)
    if not 1 <= physical_count <= MAXIMUM_PHYSICAL_STAVES:
        return None, "simultaneous physical staff count is outside the supported limit"
    hidden_visibility = _hidden_empty_part_visibility(root, parts)
    if hidden_visibility:
        variable = _variable_visible_topology(
            root,
            layout,
            parts,
            hidden_visibility=hidden_visibility,
        )
        if variable is None:
            return None, "variable visible staff topology is unresolved"
        ordered, groups, appearance_locations = variable
        group_counts = tuple(
            max(
                1,
                int(
                    round(
                        float(
                            np.median(
                                [
                                    max(1, len(system_measure_bounds(staff)))
                                    for staff in group
                                ]
                            )
                        )
                    )
                ),
            )
            for group in groups
        )
        return (
            _Topology(
                parts,
                physical_count,
                ordered,
                groups,
                group_counts,
                appearance_locations,
            ),
            None,
        )
    ordered = tuple(
        sorted(
            layout.systems,
            key=lambda item: (
                item.line_y[0] if item.line_y else item.top,
                item.index,
            ),
        )
    )
    system_counts = {
        len(part.system_measure_counts)
        for part in parts
    }
    if len(system_counts) == 1:
        ordered = _conditioned_staff_appearances(
            ordered,
            physical_staff_count=physical_count,
            score_system_count=next(iter(system_counts)),
        )
    if not ordered:
        return None, "source layout has no physical staff appearances"
    if len(ordered) % physical_count:
        return None, "source layout is incomplete for the recognized staff topology"
    groups = tuple(
        tuple(ordered[start:start + physical_count])
        for start in range(0, len(ordered), physical_count)
    )
    group_counts = tuple(
        max(
            1,
            int(
                round(
                    float(
                        np.median(
                            [
                                max(1, len(system_measure_bounds(staff)))
                                for staff in group
                            ]
                        )
                    )
                )
            ),
        )
        for group in groups
    )
    appearance_locations = tuple(
        (
            staff.index,
            group_index,
            part.index,
            part_staff,
        )
        for group_index, group in enumerate(groups)
        for staff, (part, part_staff) in zip(
            group,
            (
                mapped
                for slot in range(physical_count)
                for mapped in [_slot_to_part_from_parts(parts, slot)]
                if mapped is not None
            ),
            strict=True,
        )
    )
    return (
        _Topology(
            parts,
            physical_count,
            ordered,
            groups,
            group_counts,
            appearance_locations,
        ),
        None,
    )


def _slot_to_part_from_parts(
    parts: tuple[_PartTopology, ...],
    slot: int,
) -> tuple[_PartTopology, int] | None:
    cursor = 0
    for part in parts:
        if cursor <= slot < cursor + part.staff_count:
            return part, slot - cursor + 1
        cursor += part.staff_count
    return None


def _slot_to_part(topology: _Topology, slot: int) -> tuple[_PartTopology, int] | None:
    return _slot_to_part_from_parts(topology.parts, slot)


def _appearance_location(
    topology: _Topology,
    staff_index: int,
) -> tuple[int, _PartTopology, int] | None:
    for source_index, group_index, part_index, part_staff in (
        topology.appearance_locations
    ):
        if source_index == staff_index:
            return group_index, topology.parts[part_index], part_staff
    return None


def _assign_bbox_to_appearance(
    topology: _Topology,
    bbox: tuple[int, int, int, int],
) -> tuple[int, str] | None:
    """Assign geometry against topology-conditioned, including synthetic, staves."""

    if not topology.ordered_appearances:
        return None
    center_y = 0.5 * (float(bbox[1]) + float(bbox[3]))
    scored: list[tuple[float, StaffSystem, str]] = []
    for staff in topology.ordered_appearances:
        top_line = float(staff.line_y[0])
        bottom_line = float(staff.line_y[-1])
        if center_y < top_line:
            distance = top_line - center_y
            placement = "above"
        elif center_y > bottom_line:
            distance = center_y - bottom_line
            placement = "below"
        else:
            midpoint = 0.5 * (top_line + bottom_line)
            distance = 0.0
            placement = "above" if center_y <= midpoint else "below"
        scored.append(
            (
                distance / max(float(staff.spacing), 1.0),
                staff,
                placement,
            )
        )
    distance_spaces, staff, placement = min(
        scored,
        key=lambda item: (item[0], item[1].index),
    )
    if distance_spaces > 8.0:
        return None
    return int(staff.index), placement


def _topology_uses_conditioned_layout(
    topology: _Topology,
    layout: PageLayout,
) -> bool:
    raw = tuple(
        staff.index
        for staff in sorted(
            layout.systems,
            key=lambda item: (
                item.line_y[0] if item.line_y else item.top,
                item.index,
            ),
        )
    )
    conditioned = tuple(
        staff.index
        for staff in topology.ordered_appearances
    )
    return conditioned != raw


def _grand_staff_interstitial_owner(
    topology: _Topology,
    bbox: tuple[int, int, int, int],
    provisional_staff_index: int,
) -> tuple[int, str] | None:
    """Attach a between-staff hairpin to the upper staff of one grand staff."""

    center_y = 0.5 * (float(bbox[1]) + float(bbox[3]))
    location_by_index = {
        source_index: (group_index, part_index, part_staff)
        for source_index, group_index, part_index, part_staff in (
            topology.appearance_locations
        )
    }
    for group in topology.score_systems:
        for upper, lower in zip(group, group[1:], strict=False):
            if not (
                provisional_staff_index in {upper.index, lower.index}
                and
                float(upper.line_y[-1])
                < center_y
                < float(lower.line_y[0])
            ):
                continue
            upper_location = location_by_index.get(upper.index)
            lower_location = location_by_index.get(lower.index)
            if (
                upper_location is None
                or lower_location is None
                or upper_location[0] != lower_location[0]
                or upper_location[1] != lower_location[1]
                or lower_location[2] != upper_location[2] + 1
            ):
                continue
            return upper.index, "below"
    return None


def _candidate_for_topology(
    topology: _Topology,
    layout: PageLayout,
    candidate: VisualNotationCandidate,
) -> VisualNotationCandidate:
    """Keep staff ownership and placement consistent with conditioned geometry."""

    assigned = (
        _assign_bbox_to_appearance(topology, candidate.bbox)
        if _topology_uses_conditioned_layout(topology, layout)
        else None
    )
    grand_staff_owner = _grand_staff_interstitial_owner(
        topology,
        candidate.bbox,
        assigned[0] if assigned is not None else candidate.staff_index,
    )
    if grand_staff_owner is not None:
        assigned = grand_staff_owner
    if assigned is None:
        return candidate
    staff_index, placement = assigned
    if (
        staff_index == candidate.staff_index
        and placement == candidate.placement
    ):
        return candidate
    return VisualNotationCandidate(
        candidate.kind,
        staff_index,
        placement,
        candidate.bbox,
        candidate.confidence,
        candidate.geometry,
    )


def _source_wedge_spans(
    topology: _Topology,
    wedges: tuple[VisualNotationCandidate, ...],
) -> tuple[
    tuple[VisualNotationCandidate, VisualNotationCandidate],
    ...,
]:
    """Join a system-ending hairpin with its next-system continuation."""

    records: list[
        tuple[
            VisualNotationCandidate,
            int,
            _PartTopology,
            int,
            StaffSystem,
        ]
    ] = []
    for candidate in wedges:
        location = _appearance_location(topology, candidate.staff_index)
        if location is None:
            continue
        group_index, part, part_staff = location
        source_staff = next(
            (
                staff
                for staff in topology.score_systems[group_index]
                if staff.index == candidate.staff_index
            ),
            None,
        )
        if source_staff is not None:
            records.append(
                (candidate, group_index, part, part_staff, source_staff)
            )
    used: set[int] = set()
    spans: list[
        tuple[VisualNotationCandidate, VisualNotationCandidate]
    ] = []
    for index, (
        candidate,
        group_index,
        part,
        part_staff,
        source_staff,
    ) in enumerate(records):
        if index in used:
            continue
        continuation_index: int | None = None
        if (
            candidate.bbox[2]
            >= source_staff.right - 3.0 * source_staff.spacing
        ):
            for other_index, (
                other,
                other_group,
                other_part,
                other_part_staff,
                other_staff,
            ) in enumerate(records):
                if (
                    other_index in used
                    or other_index == index
                    or other.kind != candidate.kind
                    or other_group != group_index + 1
                    or other_part.index != part.index
                    or other_part_staff != part_staff
                    or other.bbox[0]
                    > other_staff.left + 7.0 * other_staff.spacing
                ):
                    continue
                continuation_index = other_index
                break
        if continuation_index is None:
            spans.append((candidate, candidate))
            used.add(index)
            continue
        spans.append((candidate, records[continuation_index][0]))
        used.update((index, continuation_index))
    # Preserve candidates whose provisional owner was absent from the topology;
    # the ordinary proposal path will expose a precise mapping abstention.
    recorded_ids = {id(item[0]) for item in records}
    spans.extend(
        (candidate, candidate)
        for candidate in wedges
        if id(candidate) not in recorded_ids
    )
    return tuple(
        sorted(
            spans,
            key=lambda item: (
                item[0].staff_index,
                item[0].bbox[1],
                item[0].bbox[0],
                item[0].kind,
            ),
        )
    )


def _source_coordinate(staff: StaffSystem, x: float) -> tuple[float, int, float]:
    bounds = system_measure_bounds(staff)
    if not bounds:
        width = max(1.0, float(staff.right - staff.left))
        offset = max(0.0, min(0.999999, (float(x) - staff.left) / width))
        return offset, 1, 0.45
    clamped = max(float(staff.left), min(float(staff.right) - 1e-6, float(x)))
    for index, (left, right) in enumerate(bounds):
        if clamped < right or index == len(bounds) - 1:
            offset = max(0.0, min(0.999999, (clamped - left) / max(float(right - left), 1.0)))
            return index + offset, len(bounds), 0.93
    return float(len(bounds)) - 1e-6, len(bounds), 0.70


def _target_conditioned_score_system(
    layout: PageLayout,
    group_index: int,
    target_measure_count: int,
) -> tuple[ScoreSystemLayout | None, float]:
    """Select the recognized number of barlines using cross-staff evidence.

    Individual staff barline lists are intentionally recall-oriented and can
    contain stems, cue-staff edges, or both strokes of a damaged double bar.  The
    MusicXML candidate already establishes the measure count for this system.
    Rank clustered source barlines by staff support plus local/sequence confidence,
    retain exactly the required interior boundaries when possible, and keep an
    explicit confidence penalty for a close selection decision.
    """

    score_systems = layout.effective_score_systems
    if not 0 <= group_index < len(score_systems):
        return None, 0.0
    source = score_systems[group_index]
    target_count = max(1, int(target_measure_count))
    needed_interior = max(0, target_count - 1)
    staff_by_index = {staff.index: staff for staff in layout.systems}
    staffs = [
        staff_by_index[index]
        for index in source.staff_indices
        if index in staff_by_index
    ]
    if not staffs:
        return source, 0.70
    tolerance = max(5.0, float(source.spacing) * 0.9)
    rows: list[tuple[float, int, float, float]] = []
    for staff in staffs:
        for bar_index, raw_x in enumerate(staff.barlines):
            x = float(raw_x)
            local = (
                float(staff.barline_confidences[bar_index])
                if bar_index < len(staff.barline_confidences)
                else 0.5
            )
            sequence = (
                float(staff.barline_sequence_confidences[bar_index])
                if bar_index < len(staff.barline_sequence_confidences)
                else 0.5
            )
            rows.append((x, staff.index, local, sequence))
    clusters: list[list[tuple[float, int, float, float]]] = []
    for row in sorted(rows, key=lambda item: item[0]):
        if (
            not clusters
            or row[0]
            - float(np.median([item[0] for item in clusters[-1]]))
            > tolerance
        ):
            clusters.append([row])
        else:
            clusters[-1].append(row)
    ranked: list[tuple[float, int]] = []
    for cluster in clusters:
        support = len({item[1] for item in cluster}) / len(staffs)
        sequences = [item[3] for item in cluster]
        locals_ = [item[2] for item in cluster]
        score = (
            2.0 * support
            + max(sequences)
            + 0.5 * float(np.mean(sequences))
            + 0.8 * max(locals_)
        )
        ranked.append(
            (
                float(score),
                int(round(float(np.median([item[0] for item in cluster])))),
            )
        )
    outer_window = max(tolerance, float(source.spacing) * 10.0)
    possible_left = [
        item
        for item in ranked
        if 0.0 <= item[1] - int(source.left) <= outer_window
    ]
    possible_right = [
        item
        for item in ranked
        if 0.0 <= int(source.right) - item[1] <= outer_window
    ]
    left_boundary = (
        min(possible_left, key=lambda item: item[1])[1]
        if possible_left
        else int(source.left)
    )
    right_boundary = (
        max(possible_right, key=lambda item: item[1])[1]
        if possible_right
        else int(source.right)
    )
    if right_boundary - left_boundary <= tolerance:
        return source, 0.70
    interior = [
        item
        for item in ranked
        if left_boundary + tolerance < item[1] < right_boundary - tolerance
    ]
    if len(interior) < needed_interior:
        return (
            source,
            0.86
            if len(system_measure_bounds(source)) == target_count
            else 0.78,
        )
    ordered_by_evidence = sorted(
        interior,
        key=lambda item: (-item[0], item[1]),
    )
    selected = ordered_by_evidence[:needed_interior]
    unselected = ordered_by_evidence[needed_interior:]
    if unselected and selected:
        margin = selected[-1][0] - unselected[0][0]
        selection_confidence = min(0.96, 0.86 + max(0.0, margin) * 0.10)
    else:
        selection_confidence = 0.92
    conditioned = ScoreSystemLayout(
        index=source.index,
        staff_indices=list(source.staff_indices),
        top=source.top,
        bottom=source.bottom,
        left=left_boundary,
        right=right_boundary,
        spacing=source.spacing,
        barlines=sorted(item[1] for item in selected),
        measure_count=target_count,
        grouping_confidence=source.grouping_confidence,
        grouping_method=f"{source.grouping_method}+target_barline_selection",
    )
    return conditioned, float(selection_confidence)


def _map_anchor(
    topology: _Topology,
    layout: PageLayout,
    candidate: VisualNotationCandidate,
    x: float,
) -> tuple[WedgeAnchor | None, str | None]:
    assigned = (
        _assign_bbox_to_appearance(topology, candidate.bbox)
        if _topology_uses_conditioned_layout(topology, layout)
        else None
    )
    source_staff_index = (
        assigned[0]
        if assigned is not None
        else candidate.staff_index
    )
    location = _appearance_location(topology, source_staff_index)
    if location is None:
        return None, "candidate staff is absent from the ordered source topology"
    group_index, part, part_staff = location
    source_staff = next(
        staff
        for staff in topology.score_systems[group_index]
        if staff.index == source_staff_index
    )
    target_group_count = part.system_measure_counts[group_index]
    target_group_offset = part.system_measure_offsets[group_index]
    conditioned_system, selection_confidence = (
        _target_conditioned_score_system(
            layout,
            group_index,
            target_group_count,
        )
    )
    if conditioned_system is not None:
        local_anchor = anchor_x_to_measure(
            conditioned_system,
            x,
            target_group_count,
        )
        measure_index = target_group_offset + local_anchor.local_index
        mapping_confidence = (
            max(0.0, min(1.0, float(layout.confidence)))
            * local_anchor.confidence
            * selection_confidence
        )
        return (
            WedgeAnchor(
                part.id,
                part.index,
                part_staff,
                measure_index,
                float(local_anchor.offset_ratio),
                float(mapping_confidence),
                f"target_conditioned_{local_anchor.method}",
            ),
            None,
        )
    local_coordinate, local_count, local_confidence = _source_coordinate(source_staff, x)
    group_count = topology.group_measure_counts[group_index]
    group_coordinate = local_coordinate / max(float(local_count), 1.0) * group_count
    if len(part.system_measure_counts) != len(topology.score_systems):
        return None, "MusicXML system breaks do not match the source layout"
    mapped_local = group_coordinate / max(float(group_count), 1.0) * target_group_count
    mapped_local = max(0.0, min(float(target_group_count) - 1e-6, mapped_local))
    measure_index = target_group_offset + min(target_group_count - 1, int(mapped_local))
    offset = mapped_local - int(mapped_local)
    count_mismatch = abs(group_count - target_group_count) / max(
        group_count,
        target_group_count,
        1,
    )
    mapping_confidence = (
        max(0.0, min(1.0, float(layout.confidence)))
        * local_confidence
        * max(0.55, 1.0 - count_mismatch * 2.0)
    )
    method = (
        "barline_exact_system"
        if group_count == target_group_count and local_count == group_count
        else "barline_rescaled_system"
    )
    return (
        WedgeAnchor(
            part.id,
            part.index,
            part_staff,
            measure_index,
            float(offset),
            float(mapping_confidence),
            method,
        ),
        None,
    )


def _measure_duration(measure: etree._Element) -> int:
    cursor = 0
    maximum = 0
    for child in measure:
        if child.tag == "note":
            if child.find("chord") is not None or child.find("grace") is not None:
                continue
            cursor += max(0, _integer(child.findtext("duration"), 0))
            maximum = max(maximum, cursor)
        elif child.tag == "forward":
            cursor += max(0, _integer(child.findtext("duration"), 0))
            maximum = max(maximum, cursor)
        elif child.tag == "backup":
            cursor = max(0, cursor - max(0, _integer(child.findtext("duration"), 0)))
    return max(1, maximum)


def _insert_wedge_direction(
    measure: etree._Element,
    *,
    kind: str,
    number: int,
    placement: str,
    staff: int,
    offset_ratio: float,
) -> None:
    direction = etree.Element(
        "direction",
        placement="below" if placement == "below" else "above",
    )
    direction_type = etree.SubElement(direction, "direction-type")
    etree.SubElement(
        direction_type,
        "wedge",
        type=kind,
        number=str(number),
    )
    offset = int(round(max(0.0, min(0.999999, offset_ratio)) * _measure_duration(measure)))
    if offset:
        etree.SubElement(direction, "offset").text = str(offset)
    etree.SubElement(direction, "staff").text = str(staff)
    insert_at = 0
    for index, child in enumerate(measure):
        if child.tag in {"attributes", "print"}:
            insert_at = index + 1
        else:
            break
    # Source-backed directions are represented at the measure head with offsets.
    # Keep those leading directions in chronological order. Inserting every new
    # element at the same index reverses start/stop pairs in one measure, which
    # some notation engines interpret as a zero-length or backward hairpin.
    for index in range(insert_at, len(measure)):
        child = measure[index]
        if child.tag != "direction":
            break
        child_offset = _integer(child.findtext("offset"), 0)
        if child_offset <= offset:
            insert_at = index + 1
        else:
            break
    measure.insert(insert_at, direction)


def _semantic_guard_snapshot(path: Path) -> dict[str, Any]:
    analysis = analyze_musicxml(path)
    return {
        "part_count": analysis.get("part_count"),
        "measure_count": analysis.get("measure_count"),
        "part_measure_counts": analysis.get("part_measure_counts"),
        "note_count": analysis.get("note_count"),
        "rest_count": analysis.get("rest_count"),
        "rhythm_issues": analysis.get("rhythm_issues"),
        "tie_issues": analysis.get("tie_issues"),
        "slur_issues": analysis.get("slur_issues"),
        "semantic_issues": analysis.get("semantic_issues"),
    }


def _wedge_balance(root: etree._Element) -> tuple[int, int]:
    starts = 0
    stops = 0
    for wedge in root.findall("./part/measure/direction/direction-type/wedge"):
        if wedge.get("type") in {"crescendo", "diminuendo"}:
            starts += 1
        elif wedge.get("type") == "stop":
            stops += 1
    return starts, stops


def _proposal_sort_key(item: WedgeProposal) -> tuple[int, int, float, int, float]:
    assert item.start is not None and item.end is not None
    return (
        item.start.part_index,
        item.start.staff,
        item.start.measure_index + item.start.offset_ratio,
        item.end.measure_index,
        item.end.offset_ratio,
    )


def _allocate_numbers(
    proposals: Iterable[WedgeProposal],
) -> dict[tuple[int, int, int, int], int]:
    allocated: dict[tuple[int, int, int, int], int] = {}
    active_until: dict[tuple[int, int, int], float] = {}
    for index, proposal in enumerate(sorted(proposals, key=_proposal_sort_key)):
        assert proposal.start is not None and proposal.end is not None
        start_position = proposal.start.measure_index + proposal.start.offset_ratio
        end_position = proposal.end.measure_index + proposal.end.offset_ratio
        for number in range(1, 7):
            key = (proposal.start.part_index, proposal.start.staff, number)
            if active_until.get(key, -1.0) <= start_position:
                active_until[key] = end_position
                allocated[
                    (
                        proposal.start.part_index,
                        proposal.start.staff,
                        proposal.start.measure_index,
                        index,
                    )
                ] = number
                break
    return allocated


def enrich_musicxml_with_wedges(
    image_path: Path,
    xml_path: Path,
    layout: PageLayout,
    *,
    candidates: tuple[VisualNotationCandidate, ...] | None = None,
) -> WedgeEnrichmentReport:
    image_path = image_path.resolve()
    xml_path = xml_path.resolve()
    detected = candidates if candidates is not None else detect_notation_candidates(image_path, layout)
    wedges = tuple(
        item
        for item in detected
        if item.kind in {"crescendo", "diminuendo"}
    )
    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    tree = etree.parse(str(xml_path), parser)
    root = tree.getroot()
    existing = root.findall("./part/measure/direction/direction-type/wedge")
    topology, topology_error = _build_topology(root, layout)
    physical_count = topology.physical_staff_count if topology is not None else 0
    appearances = len(topology.ordered_appearances) if topology is not None else len(layout.systems)
    system_count = len(topology.score_systems) if topology is not None else 0

    proposals: list[WedgeProposal] = []
    if existing:
        proposals = [
            WedgeProposal(item, None, None, False, False, "existing MusicXML wedges require matching before enrichment")
            for item in wedges
        ]
        return WedgeEnrichmentReport(
            str(image_path),
            str(xml_path),
            DETECTOR_VERSION,
            WEDGE_ENRICHMENT_VERSION,
            False,
            len(existing),
            physical_count,
            appearances,
            system_count,
            tuple(proposals),
        )
    if topology is None:
        proposals = [
            WedgeProposal(item, None, None, False, False, topology_error or "unsupported topology")
            for item in wedges
        ]
        return WedgeEnrichmentReport(
            str(image_path),
            str(xml_path),
            DETECTOR_VERSION,
            WEDGE_ENRICHMENT_VERSION,
            False,
            0,
            0,
            appearances,
            0,
            tuple(proposals),
            topology_error,
        )

    detected = tuple(
        _candidate_for_topology(topology, layout, item)
        for item in detected
    )
    wedges = tuple(
        item
        for item in detected
        if item.kind in {"crescendo", "diminuendo"}
    )
    for candidate, end_candidate in _source_wedge_spans(
        topology,
        wedges,
    ):
        source_gate, source_reason = wedge_source_specificity_gate(
            candidate,
            tuple(detected),
        )
        if source_gate and end_candidate is not candidate:
            end_gate, end_reason = wedge_source_specificity_gate(
                end_candidate,
                tuple(detected),
            )
            if not end_gate:
                source_gate = False
                source_reason = (
                    f"system continuation rejected: {end_reason}"
                )
        if not source_gate:
            proposals.append(
                WedgeProposal(
                    candidate,
                    None,
                    None,
                    False,
                    False,
                    source_reason,
                )
            )
            continue
        start, start_error = _map_anchor(topology, layout, candidate, candidate.bbox[0])
        end, end_error = _map_anchor(
            topology,
            layout,
            end_candidate,
            end_candidate.bbox[2],
        )
        reason = start_error or end_error
        if start is None or end is None:
            proposals.append(
                WedgeProposal(candidate, start, end, False, False, reason or "anchor mapping failed")
            )
            continue
        if start.part_index != end.part_index or start.staff != end.staff:
            proposals.append(
                WedgeProposal(candidate, start, end, False, False, "wedge endpoints map to different staves")
            )
            continue
        if (end.measure_index, end.offset_ratio) <= (start.measure_index, start.offset_ratio):
            proposals.append(
                WedgeProposal(candidate, start, end, False, False, "wedge end is not after its start")
            )
            continue
        if min(start.confidence, end.confidence) < MINIMUM_MAPPING_CONFIDENCE:
            proposals.append(
                WedgeProposal(candidate, start, end, False, False, "source-to-measure mapping confidence is too low")
            )
            continue
        proposals.append(WedgeProposal(candidate, start, end, True, False, "eligible"))

    eligible = [item for item in proposals if item.eligible]
    if not eligible:
        return WedgeEnrichmentReport(
            str(image_path),
            str(xml_path),
            DETECTOR_VERSION,
            WEDGE_ENRICHMENT_VERSION,
            False,
            0,
            physical_count,
            appearances,
            system_count,
            tuple(proposals),
        )

    before = _semantic_guard_snapshot(xml_path)
    allocated = _allocate_numbers(eligible)
    committed: list[WedgeProposal] = []
    eligible_order = {id(item): index for index, item in enumerate(sorted(eligible, key=_proposal_sort_key))}
    for proposal in proposals:
        if not proposal.eligible:
            committed.append(proposal)
            continue
        assert proposal.start is not None and proposal.end is not None
        sorted_index = eligible_order[id(proposal)]
        number = allocated.get(
            (
                proposal.start.part_index,
                proposal.start.staff,
                proposal.start.measure_index,
                sorted_index,
            )
        )
        if number is None:
            committed.append(
                WedgeProposal(
                    proposal.candidate,
                    proposal.start,
                    proposal.end,
                    False,
                    False,
                    "more than six overlapping wedges on one staff",
                )
            )
            continue
        part = topology.parts[proposal.start.part_index]
        start_measure = part.measures[proposal.start.measure_index]
        end_measure = part.measures[proposal.end.measure_index]
        _insert_wedge_direction(
            start_measure,
            kind=proposal.candidate.kind,
            number=number,
            placement=proposal.candidate.placement,
            staff=proposal.start.staff,
            offset_ratio=proposal.start.offset_ratio,
        )
        _insert_wedge_direction(
            end_measure,
            kind="stop",
            number=number,
            placement=proposal.candidate.placement,
            staff=proposal.end.staff,
            offset_ratio=proposal.end.offset_ratio,
        )
        committed.append(
            WedgeProposal(
                proposal.candidate,
                proposal.start,
                proposal.end,
                True,
                True,
                "source-backed transaction committed",
                number,
            )
        )

    if not any(item.injected for item in committed):
        return WedgeEnrichmentReport(
            str(image_path),
            str(xml_path),
            DETECTOR_VERSION,
            WEDGE_ENRICHMENT_VERSION,
            False,
            0,
            physical_count,
            appearances,
            system_count,
            tuple(committed),
        )

    temporary = xml_path.with_name(xml_path.name + ".wedge-transaction.tmp")
    try:
        doctype = tree.docinfo.doctype or MUSICXML_DOCTYPE
        temporary.write_bytes(
            etree.tostring(
                tree,
                encoding="UTF-8",
                xml_declaration=True,
                doctype=doctype,
                pretty_print=True,
            )
        )
        errors = validate_musicxml(temporary)
        after = _semantic_guard_snapshot(temporary) if not errors else {}
        starts, stops = _wedge_balance(root)
        if errors:
            raise ValueError("; ".join(errors))
        if before != after:
            raise ValueError("non-wedge MusicXML semantics changed during enrichment")
        if starts != stops:
            raise ValueError(f"wedge transaction is unbalanced ({starts} starts, {stops} stops)")
        atomic_write_bytes(xml_path, temporary.read_bytes())
    except Exception as exc:
        return WedgeEnrichmentReport(
            str(image_path),
            str(xml_path),
            DETECTOR_VERSION,
            WEDGE_ENRICHMENT_VERSION,
            False,
            0,
            physical_count,
            appearances,
            system_count,
            tuple(
                WedgeProposal(
                    item.candidate,
                    item.start,
                    item.end,
                    item.eligible,
                    False,
                    f"transaction rolled back: {type(exc).__name__}: {exc}",
                    item.number,
                )
                if item.injected
                else item
                for item in committed
            ),
            f"{type(exc).__name__}: {exc}",
        )
    finally:
        temporary.unlink(missing_ok=True)

    return WedgeEnrichmentReport(
        str(image_path),
        str(xml_path),
        DETECTOR_VERSION,
        WEDGE_ENRICHMENT_VERSION,
        True,
        0,
        physical_count,
        appearances,
        system_count,
        tuple(committed),
    )
