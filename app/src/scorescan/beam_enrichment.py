from __future__ import annotations

"""Transaction-safe restoration of source-proven MusicXML beam segments.

The release-gated semantic detector supplies one positioned ``beam`` box for each
rendered beam segment.  This module maps those boxes to a fixed, already-recognised
event lattice.  It never changes notes, rests, pitches, durations, voices, staves or
measure structure, and it abstains on chords, multiple voices, existing beam markup,
weak layout mappings and ambiguous endpoint assignments.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from itertools import product
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from lxml import etree

from .layout import PageLayout
from .musicxml import MUSICXML_DOCTYPE, validate_musicxml
from .semantic_detector import SemanticDetection
from .util import atomic_write_bytes
from .wedge_enrichment import (
    MINIMUM_MAPPING_CONFIDENCE,
    WedgeAnchor,
    _assign_bbox_to_appearance,
    _appearance_location,
    _build_topology,
    _measure_duration,
    _topology_uses_conditioned_layout,
)


BEAM_ENRICHMENT_VERSION = "source-semantic-beam-transaction@3"
MAXIMUM_EVENTS_PER_STAFF_MEASURE = 32
# Exact registered-scan calibration produced a clean separation: all 185 correct
# source assignments were <= 0.085, while the first observed false assignment was
# 0.128.  Preserve margin instead of accepting a visually shifted suffix group.
MAXIMUM_ENDPOINT_ERROR = 0.09
MAXIMUM_TOTAL_ENDPOINT_ERROR = 0.26
MINIMUM_ASSIGNMENT_MARGIN = 0.012
# MusicXML ``default-x`` locates the notehead, while an up-stem beam attaches
# at the notehead's right edge.  In standard tenths this offset is one regular
# notehead width.  Leaving it out creates a false alternate solution whenever
# one measure contains both down-stem and up-stem beam groups.
UP_STEM_ATTACHMENT_OFFSET_TENTHS = 12.0
_TYPE_LEVELS = {
    "eighth": 1,
    "16th": 2,
    "32nd": 3,
    "64th": 4,
    "128th": 5,
    "256th": 6,
    "512th": 7,
    "1024th": 8,
}


@dataclass(frozen=True)
class BeamProposal:
    detection: SemanticDetection
    part_index: int | None
    staff: int | None
    measure_index: int | None
    event_indices: tuple[int, ...]
    beam_level: int | None
    endpoint_error: float | None
    eligible: bool
    injected: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["detection"] = self.detection.to_dict()
        payload["event_indices"] = list(self.event_indices)
        return payload


@dataclass(frozen=True)
class BeamEnrichmentReport:
    xml_path: str
    enrichment_version: str
    transaction_committed: bool
    detected_count: int
    proposals: tuple[BeamProposal, ...]
    error: str | None = None

    @property
    def injected_segment_count(self) -> int:
        return len(
            {
                (
                    item.part_index,
                    item.staff,
                    item.measure_index,
                    item.beam_level,
                    item.event_indices,
                )
                for item in self.proposals
                if item.injected
            }
        )

    @property
    def injected_marker_count(self) -> int:
        return len(
            {
                (
                    item.part_index,
                    item.staff,
                    item.measure_index,
                    item.beam_level,
                    event_index,
                )
                for item in self.proposals
                if item.injected
                for event_index in item.event_indices
            }
        )

    @property
    def assigned_source_count(self) -> int:
        return len(
            {
                item.detection
                for item in self.proposals
                if item.injected
            }
        )

    @property
    def abstention_count(self) -> int:
        return sum(not item.injected for item in self.proposals)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": 1,
            "xml_path": self.xml_path,
            "enrichment_version": self.enrichment_version,
            "transaction_committed": self.transaction_committed,
            "detected_count": self.detected_count,
            "injected_segment_count": self.injected_segment_count,
            "injected_marker_count": self.injected_marker_count,
            "abstention_count": self.abstention_count,
            "error": self.error,
            "proposals": [item.to_dict() for item in self.proposals],
        }


@dataclass(frozen=True)
class _Event:
    element: etree._Element
    onset: int
    duration: int
    level_count: int
    sequence_index: int
    visual_position: float | None = None
    grace: bool = False


@dataclass(frozen=True)
class _Segment:
    event_indices: tuple[int, ...]
    endpoint_error: float
    total_error: float


@dataclass(frozen=True)
class _ConsensusBounds:
    boundaries: tuple[float, ...]
    confidence: float
    method: str


def _integer(value: str | None, default: int = 0) -> int:
    try:
        return int((value or "").strip())
    except (TypeError, ValueError, OverflowError):
        return default


def _beam_free_digest(root: etree._Element) -> bytes:
    copy_root = deepcopy(root)
    for beam in copy_root.findall("./part/measure/note/beam"):
        parent = beam.getparent()
        if parent is not None:
            parent.remove(beam)
    return etree.tostring(copy_root, method="c14n", with_comments=False)


def _interval_dispersion(boundaries: list[float]) -> float:
    widths = [
        right - left
        for left, right in zip(boundaries, boundaries[1:], strict=False)
    ]
    if not widths or min(widths) <= 0:
        return math.inf
    mean = sum(widths) / len(widths)
    return math.sqrt(
        sum((width / mean - 1.0) ** 2 for width in widths)
        / len(widths)
    )


def _consensus_measure_boundaries(
    staffs: tuple[Any, ...],
    *,
    target_measure_count: int,
    layout_confidence: float,
    page_width: int,
    semantic_barlines: tuple[SemanticDetection, ...] = (),
    system_opening_has_left_barline: bool = False,
) -> tuple[_ConsensusBounds | None, str | None]:
    if not staffs or target_measure_count <= 0:
        return None, "source score-system boundary evidence is incomplete"
    spacing = max(
        1.0,
        float(median(float(staff.spacing) for staff in staffs)),
    )
    tolerance = 1.6 * spacing
    page_edge_margin = max(4.0, 0.004 * max(1, int(page_width)))
    target_staff_indices = {
        int(staff.index)
        for staff in staffs
    }
    semantic_values = sorted(
        (
            0.5 * (float(item.bbox[0]) + float(item.bbox[2])),
            int(item.staff_index),
            max(0.0, min(1.0, float(item.confidence))),
        )
        for item in semantic_barlines
        if (
            item.class_name == "genericBarline"
            and int(item.staff_index) in target_staff_indices
            and 0.5 * (float(item.bbox[0]) + float(item.bbox[2]))
            > page_edge_margin
            and 0.5 * (float(item.bbox[0]) + float(item.bbox[2]))
            < float(page_width) - page_edge_margin
        )
    )
    semantic_clusters: list[list[tuple[float, int, float]]] = []
    for value in semantic_values:
        # Repeat/final barlines can contain several semantic BarLine paths.  Use
        # connected single-link clustering so the complete printed compound
        # barline remains one measure boundary.
        if not semantic_clusters or value[0] - semantic_clusters[-1][-1][0] > tolerance:
            semantic_clusters.append([value])
        else:
            semantic_clusters[-1].append(value)
    semantic_candidates = [
        (
            float(median(item[0] for item in cluster)),
            len({item[1] for item in cluster}),
            max(item[2] for item in cluster),
        )
        for cluster in semantic_clusters
    ]
    required = target_measure_count + 1
    semantic_header_pruned = False
    semantic_staff_opening_added = False
    if (
        not system_opening_has_left_barline
        and len(semantic_candidates) == target_measure_count
    ):
        opening = float(
            median(float(staff.left) for staff in staffs)
        )
        if (
            opening > page_edge_margin
            and semantic_candidates
            and semantic_candidates[0][0] - opening >= 2.0 * spacing
        ):
            # MuseScore's semantic BarLine paths normally represent the right
            # boundary of every measure; the ordinary opening is a staff edge,
            # not a BarLine object.  Reconstruct only that one edge from the
            # staff geometry when the semantic count exactly proves the pattern.
            semantic_candidates = [
                (
                    opening,
                    len(staffs),
                    max(0.0, min(1.0, float(layout_confidence))),
                ),
                *semantic_candidates,
            ]
            semantic_staff_opening_added = True
    if (
        system_opening_has_left_barline
        and len(semantic_candidates) == required + 1
    ):
        # At a new system, staff lines begin before the clef/key block.  When
        # MusicXML explicitly starts with a left repeat/barline, semantic SVG
        # paths therefore contain both the staff edge and the later musical
        # boundary.  The latter starts the first measure.
        semantic_candidates = semantic_candidates[1:]
        semantic_header_pruned = True
    if len(semantic_candidates) == required:
        boundaries = [item[0] for item in semantic_candidates]
        if any(
            right - left < 2.0 * spacing
            for left, right in zip(boundaries, boundaries[1:], strict=False)
        ):
            return None, "semantic source boundaries contain an implausible interval"
        support_ratio = sum(
            min(len(staffs), item[1]) / len(staffs)
            for item in semantic_candidates
        ) / len(semantic_candidates)
        semantic_confidence = sum(
            item[2] for item in semantic_candidates
        ) / len(semantic_candidates)
        confidence = (
            max(0.0, min(1.0, float(layout_confidence)))
            * semantic_confidence
            * (0.94 + 0.06 * support_ratio)
            * (0.98 if semantic_header_pruned else 1.0)
        )
        return (
            _ConsensusBounds(
                tuple(boundaries),
                float(confidence),
                (
                    "semantic-left-barline-header-pruned"
                    if semantic_header_pruned
                    else (
                        "semantic-right-barlines-with-staff-opening"
                        if semantic_staff_opening_added
                        else "semantic-barline-recognized-count-exact"
                    )
                ),
            ),
            None,
        )

    values = sorted(
        (float(x), int(staff.index))
        for staff in staffs
        for x in staff.barlines
        if (
            float(x) > page_edge_margin
            and float(x) < float(page_width) - page_edge_margin
        )
    )
    values.extend(
        (value[0], value[1])
        for value in semantic_values
    )
    values.sort()
    clusters: list[list[tuple[float, int]]] = []
    for value in values:
        if (
            not clusters
            or value[0] - float(median(item[0] for item in clusters[-1]))
            > tolerance
        ):
            clusters.append([value])
        else:
            clusters[-1].append(value)

    # A physical staff edge is useful only when no detected barline already
    # explains that edge.  This keeps tests and partially faded scans usable
    # without turning a brace/instrument-name extent into an extra measure.
    for edge_name in ("left", "right"):
        edge = float(median(float(getattr(staff, edge_name)) for staff in staffs))
        if not (
            edge > page_edge_margin
            and edge < float(page_width) - page_edge_margin
        ):
            continue
        if not any(
            abs(edge - float(median(item[0] for item in cluster)))
            <= 2.0 * spacing
            for cluster in clusters
        ):
            clusters.append(
                [(edge, int(staff.index)) for staff in staffs]
            )
    clusters.sort(key=lambda cluster: float(median(item[0] for item in cluster)))

    candidates = [
        (
            float(median(item[0] for item in cluster)),
            len({item[1] for item in cluster}),
        )
        for cluster in clusters
    ]
    if len(candidates) < required:
        return None, (
            "source system has fewer consensus boundaries than recognized measures"
        )
    extra = len(candidates) - required
    if extra > 2:
        return None, "source system has too many unresolved boundary candidates"

    removed_full_support = False
    weakest_removals = 0
    while len(candidates) > required:
        weakest_support = min(item[1] for item in candidates)
        removable = [
            index
            for index, item in enumerate(candidates)
            if item[1] == weakest_support
        ]
        ranked: list[tuple[float, int]] = []
        for index in removable:
            remaining = [
                item[0]
                for candidate_index, item in enumerate(candidates)
                if candidate_index != index
            ]
            ranked.append((_interval_dispersion(remaining), index))
        ranked.sort()
        if (
            len(ranked) > 1
            and ranked[1][0] - ranked[0][0] < 0.015
        ):
            return None, "source boundary pruning is geometrically ambiguous"
        selected_index = ranked[0][1]
        removed_full_support = (
            removed_full_support
            or candidates[selected_index][1] >= len(staffs)
        )
        weakest_removals += int(
            candidates[selected_index][1] < len(staffs)
        )
        del candidates[selected_index]

    boundaries = [item[0] for item in candidates]
    if (
        len(boundaries) != required
        or any(
            right - left < 2.0 * spacing
            for left, right in zip(
                boundaries,
                boundaries[1:],
                strict=False,
            )
        )
    ):
        return None, "resolved source boundaries contain an implausible interval"
    support_ratio = sum(
        min(len(staffs), item[1]) / len(staffs)
        for item in candidates
    ) / len(candidates)
    confidence = (
        max(0.0, min(1.0, float(layout_confidence)))
        * 0.96
        * support_ratio
        * (0.92 if removed_full_support else 1.0)
        * (0.98**weakest_removals)
    )
    return (
        _ConsensusBounds(
            tuple(boundaries),
            float(confidence),
            (
                "recognized-count-constrained-consensus"
                if extra
                else "recognized-count-exact-consensus"
            ),
        ),
        None,
    )


def _map_beam_anchor(
    topology: Any,
    layout: PageLayout,
    detection: SemanticDetection,
    x: float,
    *,
    semantic_barlines: tuple[SemanticDetection, ...] = (),
) -> tuple[WedgeAnchor | None, str | None]:
    location = _appearance_location(topology, detection.staff_index)
    if location is None:
        return None, "candidate staff is absent from the ordered source topology"
    group_index, part, part_staff = location
    if (
        len(part.system_measure_counts) != len(topology.score_systems)
        or not 0 <= group_index < len(topology.score_systems)
    ):
        return None, "MusicXML system breaks do not match the source layout"
    target_count = part.system_measure_counts[group_index]
    solution, error = _consensus_measure_boundaries(
        topology.score_systems[group_index],
        target_measure_count=target_count,
        layout_confidence=layout.confidence,
        page_width=layout.width,
        semantic_barlines=semantic_barlines,
        system_opening_has_left_barline=any(
            str(barline.get("location") or "right").strip().casefold()
            == "left"
            for barline in part.measures[
                part.system_measure_offsets[group_index]
            ].findall("barline")
        ),
    )
    if solution is None:
        return None, error or "source boundary consensus is unresolved"
    boundaries = solution.boundaries
    if x < boundaries[0] or x > boundaries[-1]:
        return None, "beam endpoint falls outside the resolved score system"
    local_index = target_count - 1
    offset = 0.999999
    for index, (left, right) in enumerate(
        zip(boundaries, boundaries[1:], strict=False)
    ):
        if x < right or index == target_count - 1:
            local_index = index
            offset = max(
                0.0,
                min(0.999999, (float(x) - left) / max(1.0, right - left)),
            )
            break
    measure_index = part.system_measure_offsets[group_index] + local_index
    return (
        WedgeAnchor(
            part.id,
            part.index,
            part_staff,
            measure_index,
            float(offset),
            solution.confidence,
            solution.method,
        ),
        None,
    )


def _normalize_parallel_beam_staff_assignments(
    topology: Any,
    detections: tuple[SemanticDetection, ...],
) -> tuple[SemanticDetection, ...]:
    """Keep one connected beam hierarchy on one physical staff appearance.

    Long stems can place adjacent beam levels near the midpoint between two
    systems.  Assigning every thin strip independently can therefore send the
    primary level upward and the secondary level downward.  Connected,
    horizontally overlapping levels are one engraving object; choose their staff
    jointly, and change ownership only when the aggregate geometry has a clear
    margin.
    """

    beam_indices = [
        index
        for index, item in enumerate(detections)
        if item.class_name == "beam"
    ]
    staffs = tuple(topology.ordered_appearances)
    if len(beam_indices) < 2 or not staffs:
        return detections
    spacing = max(
        1.0,
        float(median(float(staff.spacing) for staff in staffs)),
    )

    def connected(left: SemanticDetection, right: SemanticDetection) -> bool:
        left_width = max(1.0, float(left.bbox[2] - left.bbox[0]))
        right_width = max(1.0, float(right.bbox[2] - right.bbox[0]))
        overlap = max(
            0.0,
            min(float(left.bbox[2]), float(right.bbox[2]))
            - max(float(left.bbox[0]), float(right.bbox[0])),
        )
        left_center = 0.5 * (float(left.bbox[1]) + float(left.bbox[3]))
        right_center = 0.5 * (float(right.bbox[1]) + float(right.bbox[3]))
        vertical_gap = max(
            0.0,
            max(float(left.bbox[1]), float(right.bbox[1]))
            - min(float(left.bbox[3]), float(right.bbox[3])),
        )
        return bool(
            overlap / min(left_width, right_width) >= 0.72
            and abs(left_center - right_center) <= 3.0 * spacing
            and vertical_gap <= 1.25 * spacing
        )

    remaining = set(beam_indices)
    components: list[list[int]] = []
    while remaining:
        seed = remaining.pop()
        component = [seed]
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            neighbours = [
                candidate
                for candidate in tuple(remaining)
                if connected(detections[current], detections[candidate])
            ]
            for candidate in neighbours:
                remaining.remove(candidate)
                component.append(candidate)
                frontier.append(candidate)
        components.append(component)

    normalized = list(detections)
    for component in components:
        cluster_x = float(
            median(
                0.5
                * (
                    float(detections[index].bbox[0])
                    + float(detections[index].bbox[2])
                )
                for index in component
            )
        )
        cluster_y = float(
            median(
                0.5
                * (
                    float(detections[index].bbox[1])
                    + float(detections[index].bbox[3])
                )
                for index in component
            )
        )
        ranked: list[tuple[float, int, Any]] = []
        for staff in staffs:
            top_line = float(staff.line_y[0])
            bottom_line = float(staff.line_y[-1])
            total = 0.0
            for index in component:
                bbox = detections[index].bbox
                center_y = 0.5 * (float(bbox[1]) + float(bbox[3]))
                if center_y < top_line:
                    distance = top_line - center_y
                elif center_y > bottom_line:
                    distance = center_y - bottom_line
                else:
                    distance = 0.0
                total += distance / max(1.0, float(staff.spacing))
            location = (
                _appearance_location(topology, int(staff.index))
                if hasattr(topology, "appearance_locations")
                else None
            )
            if location is not None:
                group_index, part, part_staff = location
                if (
                    not 0 <= group_index < len(part.system_measure_counts)
                    or group_index >= len(part.system_measure_offsets)
                ):
                    location = None
            if location is not None:
                group_index, part, part_staff = location
                target_count = part.system_measure_counts[group_index]
                fraction = max(
                    0.0,
                    min(
                        0.999999,
                        (cluster_x - float(staff.left))
                        / max(1.0, float(staff.right - staff.left)),
                    ),
                )
                measure_index = (
                    part.system_measure_offsets[group_index]
                    + min(target_count - 1, int(fraction * target_count))
                )
                measure = part.measures[measure_index]
                stems = [
                    str(note.findtext("stem") or "").strip().casefold()
                    for note in measure.findall("note")
                    if (
                        max(1, _integer(note.findtext("staff"), 1))
                        == part_staff
                        and note.find("rest") is None
                        and str(note.findtext("type") or "")
                        .strip()
                        .casefold()
                        in _TYPE_LEVELS
                    )
                ]
                dominant = (
                    "up"
                    if stems.count("up") >= max(2, math.ceil(0.75 * len(stems)))
                    else (
                        "down"
                        if stems.count("down")
                        >= max(2, math.ceil(0.75 * len(stems)))
                        else None
                    )
                )
                placement = (
                    "above"
                    if cluster_y < top_line
                    else "below"
                    if cluster_y > bottom_line
                    else None
                )
                expected = (
                    "above"
                    if dominant == "up"
                    else "below"
                    if dominant == "down"
                    else None
                )
                if placement is not None and expected is not None:
                    if placement == expected:
                        total -= 0.35 * len(component)
                    else:
                        total += 1.25 * len(component)
            ranked.append((total, int(staff.index), staff))
        ranked.sort(key=lambda item: (item[0], item[1]))
        if (
            len(ranked) > 1
            and ranked[1][0] - ranked[0][0] < 0.20
        ):
            continue
        owner = ranked[0][2]
        midpoint = 0.5 * (
            float(owner.line_y[0]) + float(owner.line_y[-1])
        )
        for index in component:
            item = detections[index]
            center_y = 0.5 * (float(item.bbox[1]) + float(item.bbox[3]))
            placement = "above" if center_y <= midpoint else "below"
            if (
                item.staff_index == int(owner.index)
                and item.placement == placement
            ):
                continue
            normalized[index] = SemanticDetection(
                item.class_name,
                item.label,
                item.bbox,
                item.confidence,
                int(owner.index),
                placement,
            )
    return tuple(normalized)


def _events_for_staff(
    measure: etree._Element,
    staff: int,
    *,
    target_voice: int | None = None,
    voice_staves: tuple[int, ...] | None = None,
) -> tuple[list[_Event], str | None]:
    cursor = 0
    last_onset: dict[tuple[int, int], int] = {}
    events: list[_Event] = []
    voices: set[int] = set()
    chord_error: str | None = None
    try:
        measure_width = float(measure.get("width") or "")
    except (TypeError, ValueError, OverflowError):
        measure_width = 0.0
    for child in measure:
        if child.tag == "backup":
            cursor = max(0, cursor - max(0, _integer(child.findtext("duration"))))
            continue
        if child.tag == "forward":
            cursor += max(0, _integer(child.findtext("duration")))
            continue
        if child.tag != "note":
            continue
        note_staff = max(1, _integer(child.findtext("staff"), 1))
        voice = max(1, _integer(child.findtext("voice"), 1))
        duration = max(0, _integer(child.findtext("duration")))
        chord = child.find("chord") is not None
        grace = child.find("grace") is not None
        key = (note_staff, voice)
        onset = last_onset.get(key, cursor) if chord else cursor
        if not chord and not grace:
            last_onset[key] = onset
            cursor += duration
        staff_is_selected = (
            note_staff == staff
            or (
                target_voice is not None
                and voice == target_voice
                and voice_staves is not None
                and note_staff in voice_staves
            )
        )
        if not staff_is_selected:
            continue
        voices.add(voice)
        if target_voice is not None and voice != target_voice:
            continue
        note_type = str(child.findtext("type") or "").strip().casefold()
        level_count = _TYPE_LEVELS.get(note_type, 0)
        if (
            child.find("rest") is not None
            or chord
            or (duration <= 0 and not grace)
        ):
            level_count = 0
        if chord:
            primary = events[-1] if events else None
            if (
                primary is None
                or primary.onset != onset
                or primary.duration != duration
                or str(
                    primary.element.findtext("type") or ""
                ).strip().casefold()
                != note_type
                or primary.element.find("rest") is not None
            ):
                chord_error = (
                    "chord continuation does not match one beam-carrier event"
                )
            # MusicXML beam elements belong to the first note of a chord.  Chord
            # continuations share its onset and never become separate beam events.
            continue
        visual_position: float | None = None
        if measure_width > 0.0 and child.get("default-x") is not None:
            try:
                visual_position = (
                    (
                        float(child.get("default-x"))
                        + (
                            UP_STEM_ATTACHMENT_OFFSET_TENTHS
                            if str(child.findtext("stem") or "")
                            .strip()
                            .casefold()
                            == "up"
                            else 0.0
                        )
                    )
                    / measure_width
                )
            except (TypeError, ValueError, OverflowError):
                visual_position = None
        events.append(
            _Event(
                child,
                onset,
                duration,
                level_count,
                len(events),
                visual_position,
                grace,
            )
        )
    if chord_error is not None:
        return [], chord_error
    if target_voice is None and len(voices) != 1:
        return [], "staff measure does not contain exactly one voice"
    if target_voice is not None and target_voice not in voices:
        return [], "requested staff voice is absent"
    if not 1 <= len(events) <= MAXIMUM_EVENTS_PER_STAFF_MEASURE:
        return [], "staff measure event count is outside the bounded range"
    if any(
        note.find("beam") is not None
        for note in measure.findall("note")
        if (
            (
                max(1, _integer(note.findtext("staff"), 1)) == staff
                or (
                    target_voice is not None
                    and voice_staves is not None
                    and max(1, _integer(note.findtext("staff"), 1))
                    in voice_staves
                )
            )
            and (
                target_voice is None
                or max(1, _integer(note.findtext("voice"), 1))
                == target_voice
            )
        )
    ):
        return [], "existing MusicXML beams require relation matching"
    visual_positions = [event.visual_position for event in events]
    visual_valid = bool(
        visual_positions
        and all(
            value is not None
            and math.isfinite(value)
            and -0.05 <= value <= 1.05
            for value in visual_positions
        )
        and all(
            float(right) > float(left)
            for left, right in zip(
                visual_positions,
                visual_positions[1:],
                strict=False,
            )
        )
    )
    if not visual_valid:
        events = [
            _Event(
                event.element,
                event.onset,
                event.duration,
                event.level_count,
                event.sequence_index,
                None,
                event.grace,
            )
            for event in events
        ]
    return events, None


def _staff_voices(
    measure: etree._Element,
    staff: int,
) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                max(1, _integer(note.findtext("voice"), 1))
                for note in measure.findall("note")
                if max(1, _integer(note.findtext("staff"), 1)) == staff
            }
        )
    )


def _eligible_runs(events: list[_Event]) -> tuple[tuple[int, ...], ...]:
    runs: list[list[int]] = []
    for grace_lane in (False, True):
        active: list[int] = []
        for index, event in enumerate(events):
            if event.grace != grace_lane:
                continue
            if event.level_count <= 0:
                if active:
                    runs.append(active)
                    active = []
                continue
            if active:
                previous = events[active[-1]]
                if event.onset != previous.onset + previous.duration:
                    runs.append(active)
                    active = []
            active.append(index)
        if active:
            runs.append(active)
    runs.sort(key=lambda run: (run[0], run[-1], tuple(run)))
    return tuple(tuple(run) for run in runs)


def _event_lattice_position(
    event: _Event,
    *,
    measure_duration: int,
) -> float:
    if event.visual_position is not None:
        return float(event.visual_position)
    return 0.075 + 0.85 * event.onset / max(1, measure_duration)


def _segments(
    events: list[_Event],
    *,
    start_offset: float,
    end_offset: float,
    measure_duration: int,
) -> list[_Segment]:
    result: list[_Segment] = []
    duration = max(1, measure_duration)
    for run in _eligible_runs(events):
        for start in range(len(run)):
            for stop in range(start, len(run)):
                indices = run[start : stop + 1]
                first = events[indices[0]]
                last = events[indices[-1]]
                predicted_start = _event_lattice_position(
                    first,
                    measure_duration=duration,
                )
                predicted_end = _event_lattice_position(
                    last,
                    measure_duration=duration,
                )
                if len(indices) == 1:
                    event_index = indices[0]
                    run_position = run.index(event_index)
                    neighbour_gaps = [
                        abs(
                            _event_lattice_position(
                                events[run[neighbour_position]],
                                measure_duration=duration,
                            )
                            - predicted_start
                        )
                        for neighbour_position in (
                            run_position - 1,
                            run_position + 1,
                        )
                        if 0 <= neighbour_position < len(run)
                    ]
                    if (
                        neighbour_gaps
                        and end_offset - start_offset
                        >= 0.80 * min(neighbour_gaps)
                    ):
                        continue
                    detected_centre = 0.5 * (start_offset + end_offset)
                    centre_error = abs(predicted_start - detected_centre)
                    span_penalty = min(0.08, abs(end_offset - start_offset) * 0.25)
                    endpoint_error = centre_error + span_penalty
                    total_error = endpoint_error
                else:
                    left_error = abs(predicted_start - start_offset)
                    right_error = abs(predicted_end - end_offset)
                    endpoint_error = max(left_error, right_error)
                    total_error = left_error + right_error
                result.append(
                    _Segment(
                        tuple(indices),
                        float(endpoint_error),
                        float(total_error),
                    )
                )
    return sorted(
        result,
        key=lambda item: (
            item.endpoint_error,
            item.total_error,
            -len(item.event_indices),
            item.event_indices,
        ),
    )


def _assign_segment(
    events: list[_Event],
    *,
    start_offset: float,
    end_offset: float,
    measure_duration: int,
) -> tuple[_Segment | None, str | None]:
    options = _segments(
        events,
        start_offset=start_offset,
        end_offset=end_offset,
        measure_duration=measure_duration,
    )
    if not options:
        return None, "no contiguous short-note segment is available"
    best = options[0]
    if (
        best.endpoint_error > MAXIMUM_ENDPOINT_ERROR
        or best.total_error > MAXIMUM_TOTAL_ENDPOINT_ERROR
    ):
        return None, "beam endpoints do not align with the event lattice"
    if len(options) > 1:
        runner_up = options[1]
        same_assignment = runner_up.event_indices == best.event_indices
        if (
            not same_assignment
            and runner_up.total_error - best.total_error
            < MINIMUM_ASSIGNMENT_MARGIN
        ):
            return None, "beam endpoint assignment is ambiguous"
    return best, None


def _segments_for_transform(
    events: list[_Event],
    *,
    start_offset: float,
    end_offset: float,
    measure_duration: int,
    intercept: float,
    scale: float,
) -> list[_Segment]:
    result: list[_Segment] = []
    duration = max(1, measure_duration)
    for run in _eligible_runs(events):
        for start in range(len(run)):
            for stop in range(start, len(run)):
                indices = run[start : stop + 1]
                first = events[indices[0]]
                last = events[indices[-1]]
                predicted_start = intercept + scale * _event_lattice_position(
                    first,
                    measure_duration=duration,
                )
                predicted_end = intercept + scale * _event_lattice_position(
                    last,
                    measure_duration=duration,
                )
                if len(indices) == 1:
                    event_index = indices[0]
                    run_position = run.index(event_index)
                    neighbour_gaps = [
                        abs(
                            intercept
                            + scale
                            * _event_lattice_position(
                                events[run[neighbour_position]],
                                measure_duration=duration,
                            )
                            - predicted_start
                        )
                        for neighbour_position in (
                            run_position - 1,
                            run_position + 1,
                        )
                        if 0 <= neighbour_position < len(run)
                    ]
                    if (
                        neighbour_gaps
                        and end_offset - start_offset
                        >= 0.80 * min(neighbour_gaps)
                    ):
                        continue
                    detected_centre = 0.5 * (start_offset + end_offset)
                    centre_error = abs(predicted_start - detected_centre)
                    span_penalty = min(
                        0.08,
                        abs(end_offset - start_offset) * 0.25,
                    )
                    endpoint_error = centre_error + span_penalty
                    total_error = endpoint_error
                else:
                    # Engravers compress or expand the white gap between two
                    # consecutive beam groups independently of the within-group
                    # note spacing.  The beam's own span is therefore the stronger
                    # event-count signal; its centre remains a bounded, lower-weight
                    # ordering signal.
                    detected_span = end_offset - start_offset
                    predicted_span = predicted_end - predicted_start
                    span_error = abs(predicted_span - detected_span)
                    centre_error = abs(
                        0.5 * (predicted_start + predicted_end)
                        - 0.5 * (start_offset + end_offset)
                    )
                    endpoint_error = max(
                        span_error,
                        0.35 * centre_error,
                    )
                    total_error = span_error + 0.35 * centre_error
                result.append(
                    _Segment(
                        tuple(indices),
                        float(endpoint_error),
                        float(total_error),
                    )
                )
    return sorted(
        result,
        key=lambda item: (
            item.total_error,
            item.endpoint_error,
            -len(item.event_indices),
            item.event_indices,
        ),
    )


def _assign_group(
    events: list[_Event],
    group: list[tuple[SemanticDetection, float, float]],
    *,
    measure_duration: int,
    system_opening: bool = False,
    source_staff_spacing: float | None = None,
) -> tuple[
    list[tuple[SemanticDetection, _Segment]] | None,
    list[int] | None,
    str | None,
]:
    ordered = sorted(
        group,
        key=lambda item: (item[1], item[2], item[0].bbox),
    )
    eligible_runs = _eligible_runs(events)
    parallel_source_spans = bool(
        len(ordered) > 1
        and max(item[1] for item in ordered)
        - min(item[1] for item in ordered)
        <= 0.02
        and max(item[2] for item in ordered)
        - min(item[2] for item in ordered)
        <= 0.02
    )
    if parallel_source_spans:
        direct_segments: list[_Segment] = []
        for _detection, start, end in ordered:
            segment, _error = _assign_segment(
                events,
                start_offset=start,
                end_offset=end,
                measure_duration=measure_duration,
            )
            if segment is None:
                direct_segments = []
                break
            direct_segments.append(segment)
        direct_levels = (
            _assign_levels(events, direct_segments)
            if direct_segments
            and len(
                {
                    segment.event_indices
                    for segment in direct_segments
                }
            )
            == 1
            else None
        )
        if (
            direct_levels is not None
            and max(
                segment.endpoint_error
                for segment in direct_segments
            )
            <= 0.025
        ):
            return (
                [
                    (detection, segment)
                    for (detection, _start, _end), segment in zip(
                        ordered,
                        direct_segments,
                        strict=True,
                    )
                ],
                direct_levels,
                None,
            )
    if parallel_source_spans:
        required_levels = len(ordered)
        high_level_runs: list[tuple[int, ...]] = []
        active_high_level: list[int] = []
        for index, event in enumerate(events):
            if event.level_count >= required_levels:
                active_high_level.append(index)
            elif active_high_level:
                high_level_runs.append(tuple(active_high_level))
                active_high_level = []
        if active_high_level:
            high_level_runs.append(tuple(active_high_level))
        if len(high_level_runs) == 1 and len(high_level_runs[0]) >= 2:
            run = high_level_runs[0]
            event_span = (
                _event_lattice_position(
                    events[run[-1]],
                    measure_duration=measure_duration,
                )
                - _event_lattice_position(
                    events[run[0]],
                    measure_duration=measure_duration,
                )
            )
            detected_span = float(
                median(item[2] - item[1] for item in ordered)
            )
            if (
                event_span > 0
                and 0.30 <= detected_span / event_span <= 1.20
            ):
                complete_segments = [
                    _Segment(run, 0.0, 0.0)
                    for _item in ordered
                ]
                complete_levels = _assign_levels(events, complete_segments)
                if complete_levels == list(
                    range(1, required_levels + 1)
                ):
                    return (
                        [
                            (detection, segment)
                            for (detection, _start, _end), segment in zip(
                                ordered,
                                complete_segments,
                                strict=True,
                            )
                        ],
                        complete_levels,
                        None,
                    )
    if (
        parallel_source_spans
        and len(events) == 2
        and eligible_runs == ((0, 1),)
    ):
        # With exactly two adjacent short-note events there is only one legal
        # primary segment.  Coincident source strips are its parallel levels,
        # even when a large clef/repeat header translates or rescales the pair.
        complete_segments = [
            _Segment((0, 1), 0.0, 0.0)
            for _item in ordered
        ]
        complete_levels = _assign_levels(events, complete_segments)
        if complete_levels is not None:
            return (
                [
                    (detection, segment)
                    for (detection, _start, _end), segment in zip(
                        ordered,
                        complete_segments,
                        strict=True,
                    )
                ],
                complete_levels,
                None,
            )
    if parallel_source_spans and len(eligible_runs) == 1:
        run = eligible_runs[0]
        detected_span = float(
            median(item[2] - item[1] for item in ordered)
        )
        uniquely_scaled: list[tuple[int, ...]] = []
        for start_index in range(len(run)):
            for stop_index in range(start_index + 1, len(run)):
                indices = tuple(run[start_index : stop_index + 1])
                event_span = (
                    _event_lattice_position(
                        events[indices[-1]],
                        measure_duration=measure_duration,
                    )
                    - _event_lattice_position(
                        events[indices[0]],
                        measure_duration=measure_duration,
                    )
                )
                if (
                    event_span > 0
                    and 0.30 <= detected_span / event_span <= 1.20
                ):
                    uniquely_scaled.append(indices)
        if uniquely_scaled == [run]:
            complete_segments = [
                _Segment(run, 0.0, 0.0)
                for _item in ordered
            ]
            complete_levels = _assign_levels(events, complete_segments)
            if complete_levels is not None:
                return (
                    [
                        (detection, segment)
                        for (detection, _start, _end), segment in zip(
                            ordered,
                            complete_segments,
                            strict=True,
                        )
                    ],
                    complete_levels,
                    None,
                    )
    if (
        len(ordered) == 1
        and source_staff_spacing is not None
        and source_staff_spacing > 0.0
        and eligible_runs == (tuple(range(len(events))),)
        and len(events) >= 2
        and min(event.level_count for event in events) >= 2
        and (
            ordered[0][0].bbox[3] - ordered[0][0].bbox[1]
        )
        / source_staff_spacing
        >= 0.88
    ):
        # Some engravers emit a connected polygon for both levels of a
        # uniformly beamed 16th-note run.  Its semantic box is visibly thicker
        # than one beam but supplies only one object.  On a complete uniform run,
        # the two coincident hierarchy levels are the only legal interpretation.
        complete = _Segment(
            tuple(range(len(events))),
            0.0,
            0.0,
        )
        return (
            [
                (ordered[0][0], complete),
                (ordered[0][0], complete),
            ],
            [1, 2],
            None,
        )
    nested_hierarchy = _nested_level_band_assignment(
        events,
        ordered,
        measure_duration=measure_duration,
        source_staff_spacing=source_staff_spacing,
    )
    if nested_hierarchy is not None:
        return nested_hierarchy[0], nested_hierarchy[1], None
    if len(ordered) == 1:
        detection, start, end = ordered[0]
        detected_span = end - start
        if (
            len(eligible_runs) == 1
            and len(eligible_runs[0]) >= 2
            and len(eligible_runs[0]) == len(events)
            and detected_span >= 0.52
        ):
            complete = _Segment(
                eligible_runs[0],
                0.0,
                0.0,
            )
            complete_levels = _assign_levels(events, [complete])
            if complete_levels is not None:
                return (
                    [(detection, complete)],
                    complete_levels,
                    None,
                )
        segment, error = _assign_segment(
            events,
            start_offset=start,
            end_offset=end,
            measure_duration=measure_duration,
        )
        levels = (
            _assign_levels(events, [segment])
            if segment is not None
            else None
        )
        if segment is not None and levels is not None:
            if (
                system_opening
                and len(eligible_runs) == 1
                and segment.event_indices[0] > eligible_runs[0][0]
            ):
                return (
                    None,
                    None,
                    "single system-opening beam span has an unresolved header translation",
                )
            return [(detection, segment)], levels, None

        if (
            len(eligible_runs) == 1
            and len(eligible_runs[0]) == len(events)
            and len(events) >= 4
        ):
            # With no rest/non-beamable event to delimit groups, a translated
            # system-opening span can equally describe a proper subgroup or a
            # compressed whole-run beam.  A single box cannot distinguish them.
            return (
                None,
                None,
                "single translated beam span is ambiguous across a complete short-note run",
            )

        # A system-opening clef/key/time block can translate every note in the
        # first measure without changing the beam's within-group span.  If the
        # ordinary absolute-position assignment chose an impossible primary
        # hook, use the span only when it identifies one unique complete group.
        duration = max(1, measure_duration)
        detected_centre = 0.5 * (start + end)
        candidates: list[tuple[float, _Segment, list[int]]] = []
        for run in _eligible_runs(events):
            for run_start in range(len(run)):
                for run_stop in range(run_start + 1, len(run)):
                    indices = tuple(run[run_start : run_stop + 1])
                    first = events[indices[0]]
                    last = events[indices[-1]]
                    first_position = _event_lattice_position(
                        first,
                        measure_duration=duration,
                    )
                    last_position = _event_lattice_position(
                        last,
                        measure_duration=duration,
                    )
                    predicted_span = last_position - first_position
                    predicted_centre = 0.5 * (
                        first_position + last_position
                    )
                    span_error = abs(predicted_span - detected_span)
                    score = span_error + 0.25 * abs(
                        predicted_centre - detected_centre
                    )
                    candidate = _Segment(
                        indices,
                        float(max(span_error, score - span_error)),
                        float(score),
                    )
                    candidate_levels = _assign_levels(events, [candidate])
                    if candidate_levels is not None and score <= 0.10:
                        candidates.append(
                            (score, candidate, candidate_levels)
                        )
        candidates.sort(key=lambda item: (item[0], item[1].event_indices))
        if not candidates:
            return (
                None,
                None,
                error
                or "detected beam segment cannot form a complete level hierarchy",
            )
        if (
            len(candidates) > 1
            and candidates[1][0] - candidates[0][0] < 0.012
        ):
            return None, None, "single beam span assignment is ambiguous"
        return (
            [(detection, candidates[0][1])],
            candidates[0][2],
            None,
        )

    fixed_assignments: list[tuple[SemanticDetection, _Segment]] = []
    for detection, start, end in ordered:
        segment, _error = _assign_segment(
            events,
            start_offset=start,
            end_offset=end,
            measure_duration=measure_duration,
        )
        if segment is None:
            fixed_assignments = []
            break
        fixed_assignments.append((detection, segment))
    fixed_levels = (
        _assign_levels(
            events,
            [item[1] for item in fixed_assignments],
        )
        if len(fixed_assignments) == len(ordered)
        else None
    )
    fixed_is_strict = (
        fixed_levels is not None
        and sum(
            item[1].total_error
            for item in fixed_assignments
        ) / len(ordered)
        <= 0.04
    )
    fixed_is_complete_partition = (
        fixed_levels is not None
        and _primary_partition_is_complete(
            events,
            fixed_assignments,
            fixed_levels,
        )
    )

    duration = max(1, measure_duration)
    all_segments = [
        _Segment(indices, 0.0, 0.0)
        for indices in (
            run[start : stop + 1]
            for run in _eligible_runs(events)
            for start in range(len(run))
            for stop in range(start, len(run))
        )
    ]
    visual_lattice = all(
        event.visual_position is not None
        for event in events
    )
    hypotheses: set[tuple[float, float]] = {
        (0.0, 1.0) if visual_lattice else (0.075, 0.85)
    }
    for _detection, detected_start, detected_end in ordered:
        for segment in all_segments:
            if len(segment.event_indices) < 2:
                continue
            first = events[segment.event_indices[0]]
            last = events[segment.event_indices[-1]]
            first_position = _event_lattice_position(
                first,
                measure_duration=duration,
            )
            last_position = _event_lattice_position(
                last,
                measure_duration=duration,
            )
            event_span = last_position - first_position
            if event_span <= 0:
                continue
            scale = (detected_end - detected_start) / event_span
            intercept = detected_start - scale * first_position
            if (
                0.30 <= scale <= 1.20
                and -0.04 <= intercept <= 0.45
                and intercept + scale <= 1.08
            ):
                hypotheses.add((round(intercept, 8), round(scale, 8)))

    solutions: dict[
        tuple[tuple[tuple[int, ...], int], ...],
        tuple[
            float,
            list[tuple[SemanticDetection, _Segment]],
            list[int],
        ],
    ] = {}
    for intercept, scale in hypotheses:
        option_sets: list[list[_Segment]] = []
        valid = True
        for _detection, start, end in ordered:
            options = [
                item
                for item in _segments_for_transform(
                    events,
                    start_offset=start,
                    end_offset=end,
                    measure_duration=duration,
                    intercept=intercept,
                    scale=scale,
                )
                if (
                    item.endpoint_error <= 0.12
                    and item.total_error <= 0.20
                )
            ]
            if not options:
                valid = False
                break
            best_error = options[0].total_error
            option_sets.append(
                [
                    item
                    for item in options[:3]
                    if item.total_error <= best_error + 0.07
                ]
            )
        if not valid:
            continue
        combination_count = math.prod(len(items) for items in option_sets)
        combinations = (
            product(*option_sets)
            if combination_count <= 4096
            else [tuple(items[0] for items in option_sets)]
        )
        for combination in combinations:
            segments = list(combination)
            levels = _assign_levels(events, segments)
            if levels is None:
                continue
            score = (
                sum(item.total_error for item in segments)
                + 0.004 * abs(scale - (1.0 if visual_lattice else 0.85))
                + 0.001 * abs(intercept - (0.0 if visual_lattice else 0.075))
            )
            key = tuple(
                (segment.event_indices, level)
                for segment, level in zip(segments, levels, strict=True)
            )
            assignments = [
                (detection, segment)
                for (detection, _start, _end), segment in zip(
                    ordered,
                    segments,
                    strict=True,
                )
            ]
            current = solutions.get(key)
            if current is None or score < current[0]:
                solutions[key] = (score, assignments, levels)
    ranked = sorted(solutions.values(), key=lambda item: item[0])
    if not ranked:
        return (
            None,
            None,
            "detected beam segments cannot form a complete level hierarchy",
        )
    best = ranked[0]
    if best[0] / len(ordered) > 0.065:
        if (
            fixed_is_complete_partition
            and not _irregular_complete_high_level_partition(
                events,
                fixed_assignments,
                fixed_levels,
            )
        ):
            return fixed_assignments, fixed_levels, None
        return None, None, "joint beam alignment exceeds its error bound"
    if (
        len(ranked) > 1
        and (ranked[1][0] - best[0]) / len(ordered) < 0.012
    ):
        best_is_complete = _primary_partition_is_complete(
            events,
            best[1],
            best[2],
        )
        runner_up_is_complete = _primary_partition_is_complete(
            events,
            ranked[1][1],
            ranked[1][2],
        )
        if (
            best_is_complete
            and not runner_up_is_complete
            and not _irregular_complete_high_level_partition(
                events,
                best[1],
                best[2],
            )
        ):
            return best[1], best[2], None
        if fixed_is_strict or fixed_is_complete_partition:
            if _irregular_complete_high_level_partition(
                events,
                fixed_assignments,
                fixed_levels,
            ):
                return (
                    None,
                    None,
                    "high-level beam partition conflicts with a regular complete run",
                )
            return fixed_assignments, fixed_levels, None
        return None, None, "joint beam assignment is ambiguous"
    if _irregular_complete_high_level_partition(events, best[1], best[2]):
        return (
            None,
            None,
            "high-level beam partition conflicts with a regular complete run",
        )
    return best[1], best[2], None


def _connected_beam_groups(
    group: list[tuple[SemanticDetection, float, float]],
    *,
    source_staff_spacing: float,
) -> list[list[tuple[SemanticDetection, float, float]]]:
    """Split one staff-measure's detections into engraved beam objects."""

    spacing = max(1.0, float(source_staff_spacing))

    def connected(
        left: tuple[SemanticDetection, float, float],
        right: tuple[SemanticDetection, float, float],
    ) -> bool:
        left_box = left[0].bbox
        right_box = right[0].bbox
        left_width = max(1.0, float(left_box[2] - left_box[0]))
        right_width = max(1.0, float(right_box[2] - right_box[0]))
        overlap = max(
            0.0,
            min(float(left_box[2]), float(right_box[2]))
            - max(float(left_box[0]), float(right_box[0])),
        )
        left_center = 0.5 * (float(left_box[1]) + float(left_box[3]))
        right_center = 0.5 * (float(right_box[1]) + float(right_box[3]))
        vertical_gap = max(
            0.0,
            max(float(left_box[1]), float(right_box[1]))
            - min(float(left_box[3]), float(right_box[3])),
        )
        return bool(
            overlap / min(left_width, right_width) >= 0.72
            and abs(left_center - right_center) <= 3.0 * spacing
            and vertical_gap <= 1.25 * spacing
        )

    remaining = set(range(len(group)))
    components: list[list[tuple[SemanticDetection, float, float]]] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        indices = [seed]
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            neighbours = [
                candidate
                for candidate in sorted(remaining)
                if connected(group[current], group[candidate])
            ]
            for candidate in neighbours:
                remaining.remove(candidate)
                indices.append(candidate)
                frontier.append(candidate)
        components.append([group[index] for index in sorted(indices)])
    return sorted(
        components,
        key=lambda items: (
            min(item[0].bbox[0] for item in items),
            min(item[0].bbox[1] for item in items),
        ),
    )


def _beam_assignment_stem_placement_score(
    events: list[_Event],
    assignments: list[tuple[SemanticDetection, _Segment]],
) -> float | None:
    """Score explicit source-side placement against recognized stem direction."""

    evidence: list[float] = []
    for detection, segment in assignments:
        placement = str(detection.placement or "").strip().casefold()
        if placement not in {"above", "below"}:
            continue
        stems = [
            str(events[index].element.findtext("stem") or "")
            .strip()
            .casefold()
            for index in segment.event_indices
            if 0 <= index < len(events)
        ]
        stems = [item for item in stems if item in {"up", "down"}]
        if not stems:
            continue
        up_count = stems.count("up")
        down_count = stems.count("down")
        dominant_count = max(up_count, down_count)
        if dominant_count / len(stems) < 0.75:
            continue
        expected = "above" if up_count > down_count else "below"
        evidence.append(1.0 if placement == expected else 0.0)
    if not evidence:
        return None
    return float(sum(evidence) / len(evidence))


def _beam_assignment_staff_membership_score(
    events: list[_Event],
    assignments: list[tuple[SemanticDetection, _Segment]],
    *,
    source_staff: int,
) -> float:
    selected = sorted(
        {
            index
            for _detection, segment in assignments
            for index in segment.event_indices
            if 0 <= index < len(events)
        }
    )
    if not selected:
        return 0.0
    matches = sum(
        max(
            1,
            _integer(events[index].element.findtext("staff"), 1),
        )
        == source_staff
        for index in selected
    )
    return float(matches / len(selected))


def _select_unique_voice_by_geometry(
    evidence: list[tuple[int, float, float | None]],
) -> int | None:
    """Select a voice only when staff ownership or stem placement separates it."""

    if len(evidence) == 1:
        return evidence[0][0]
    if len(evidence) < 2:
        return None
    membership = sorted(
        ((score, voice) for voice, score, _stem in evidence),
        reverse=True,
    )
    if (
        membership[0][0] >= 0.75
        and membership[0][0] - membership[1][0] >= 0.50
    ):
        return membership[0][1]
    stems = sorted(
        (
            (score, voice)
            for voice, _membership, score in evidence
            if score is not None
        ),
        reverse=True,
    )
    if (
        len(stems) == len(evidence)
        and stems[0][0] >= 0.75
        and stems[0][0] - stems[1][0] >= 0.50
    ):
        return stems[0][1]
    return None


def _merge_repeated_high_level_path_splits(
    events: list[_Event],
    assignments: list[tuple[SemanticDetection, _Segment]],
    levels: list[int],
    *,
    source_staff_spacing: float,
) -> tuple[
    list[tuple[SemanticDetection, _Segment]],
    list[int],
]:
    """Merge repeated SVG polygon splits without merging musical beam groups.

    Some engravers split every high-level beam polygon at the same stem while
    retaining one semantic beam.  The evidence is accepted only when at least
    two higher levels repeat the exact partition inside one primary span and
    every event in that span requires every affected level.  A lone secondary
    level, the common case where 16th-note groups are intentionally separated,
    is therefore left unchanged.
    """

    if len(assignments) != len(levels):
        return assignments, levels
    updated = list(assignments)
    primary_segments = {
        segment.event_indices
        for (_detection, segment), level in zip(
            assignments,
            levels,
            strict=True,
        )
        if level == 1 and len(segment.event_indices) >= 4
    }
    # Bounding boxes of one sloped band move vertically from polygon to
    # polygon; allow that slope while the repeated two-level partition remains
    # the decisive evidence.
    band_tolerance = max(2.0, 1.25 * max(1.0, source_staff_spacing))
    for primary in sorted(primary_segments):
        primary_set = set(primary)
        partitions: dict[
            tuple[tuple[int, ...], ...],
            list[tuple[int, list[int]]],
        ] = {}
        maximum_level = min(events[index].level_count for index in primary)
        for level in range(2, maximum_level + 1):
            member_indices = [
                index
                for index, ((_detection, segment), assigned_level)
                in enumerate(zip(assignments, levels, strict=True))
                if (
                    assigned_level == level
                    and set(segment.event_indices) <= primary_set
                )
            ]
            if len(member_indices) < 2:
                continue
            members = sorted(
                (
                    assignments[index][1].event_indices
                    for index in member_indices
                ),
                key=lambda item: (item[0], item[-1]),
            )
            flattened = [
                event_index
                for member in members
                for event_index in member
            ]
            if (
                flattened != list(primary)
                or len(set(flattened)) != len(flattened)
            ):
                continue
            centres = [
                0.5
                * (
                    assignments[index][0].bbox[1]
                    + assignments[index][0].bbox[3]
                )
                for index in member_indices
            ]
            if max(centres) - min(centres) > band_tolerance:
                continue
            partitions.setdefault(tuple(members), []).append(
                (level, member_indices)
            )

        for repeated in partitions.values():
            if len(repeated) < 2:
                continue
            for _level, member_indices in repeated:
                endpoint_error = max(
                    assignments[index][1].endpoint_error
                    for index in member_indices
                )
                total_error = sum(
                    assignments[index][1].total_error
                    for index in member_indices
                )
                merged = _Segment(
                    primary,
                    float(endpoint_error),
                    float(total_error),
                )
                for index in member_indices:
                    updated[index] = (assignments[index][0], merged)
    return updated, levels


def _nested_level_band_assignment(
    events: list[_Event],
    ordered: list[tuple[SemanticDetection, float, float]],
    *,
    measure_duration: int,
    source_staff_spacing: float | None,
) -> tuple[
    list[tuple[SemanticDetection, _Segment]],
    list[int],
] | None:
    """Resolve a complete nested hierarchy whose SVG paths split at stems.

    MuseScore can draw one semantic beam level as multiple adjacent polygons
    when its slope changes at a stem.  Treating those polygons as independent
    MusicXML segments creates false ``end``/``begin`` pairs.  This resolver is
    intentionally narrow: one event run, one vertically distinct band per
    semantic level, strictly nested horizontal unions, and strictly shrinking
    event spans.  Those constraints distinguish the case from two real beam
    groups at the same level.
    """

    if (
        len(ordered) < 2
        or source_staff_spacing is None
        or source_staff_spacing <= 0.0
    ):
        return None
    runs = _eligible_runs(events)
    if len(runs) != 1:
        return None
    run = runs[0]
    maximum_level = max((events[index].level_count for index in run), default=0)
    if maximum_level < 2:
        return None

    expected: dict[int, tuple[int, ...]] = {}
    expected_spans: list[tuple[float, int]] = []
    for level in range(1, maximum_level + 1):
        level_groups: list[list[int]] = []
        active: list[int] = []
        for index in run:
            if events[index].level_count >= level:
                active.append(index)
            elif active:
                level_groups.append(active)
                active = []
        if active:
            level_groups.append(active)
        if len(level_groups) != 1:
            return None
        indices = tuple(level_groups[0])
        if level == 1 and len(indices) < 2:
            return None
        expected[level] = indices
        expected_spans.append(
            (
                _event_lattice_position(
                    events[indices[-1]],
                    measure_duration=measure_duration,
                )
                - _event_lattice_position(
                    events[indices[0]],
                    measure_duration=measure_duration,
                ),
                level,
            )
        )
    ranked_expected = sorted(
        expected_spans,
        key=lambda item: (-item[0], item[1]),
    )

    heights = [
        float(item[0].bbox[3] - item[0].bbox[1])
        for item in ordered
    ]
    band_tolerance = max(
        2.0,
        min(
            0.42 * source_staff_spacing,
            0.75 * float(median(heights)),
        ),
    )
    vertical = sorted(
        ordered,
        key=lambda item: (
            0.5 * (float(item[0].bbox[1]) + float(item[0].bbox[3])),
            item[1],
            item[2],
            item[0].bbox,
        ),
    )
    bands: list[list[tuple[SemanticDetection, float, float]]] = []
    band_centres: list[float] = []
    for item in vertical:
        centre = 0.5 * (float(item[0].bbox[1]) + float(item[0].bbox[3]))
        if bands and abs(centre - band_centres[-1]) <= band_tolerance:
            bands[-1].append(item)
            band_centres[-1] = float(
                median(
                    0.5
                    * (
                        float(member[0].bbox[1])
                        + float(member[0].bbox[3])
                    )
                    for member in bands[-1]
                )
            )
        else:
            bands.append([item])
            band_centres.append(centre)
    if len(bands) != maximum_level:
        return None
    # For two-level hierarchies, one polygon per band is common and ordinary
    # endpoint assignment retains more information.  A complete three-or-more
    # level nested hierarchy is itself sufficiently discriminative (including
    # a bounded top-level hook), while a same-level path split is required for
    # the more common two-level case.
    if (
        not any(len(band) > 1 for band in bands)
        and maximum_level < 3
    ):
        return None

    band_extents = [
        (
            min(item[1] for item in band),
            max(item[2] for item in band),
            band,
        )
        for band in bands
    ]
    if (
        [level for _span, level in ranked_expected]
        != list(range(1, maximum_level + 1))
        or any(
            left[0] - right[0] <= 0.01
            for left, right in zip(
                ranked_expected,
                ranked_expected[1:],
                strict=False,
            )
        )
    ):
        return None
    ranked_bands = sorted(
        band_extents,
        key=lambda item: (-(item[1] - item[0]), item[0], item[1]),
    )
    if any(
        (left[1] - left[0]) - (right[1] - right[0])
        <= max(0.01, 0.06 * (left[1] - left[0]))
        for left, right in zip(
            ranked_bands,
            ranked_bands[1:],
            strict=False,
        )
    ):
        return None
    for outer, inner in zip(
        ranked_bands,
        ranked_bands[1:],
        strict=False,
    ):
        if inner[0] < outer[0] - 0.035 or inner[1] > outer[1] + 0.035:
            return None

    assignments: list[tuple[SemanticDetection, _Segment]] = []
    levels: list[int] = []
    for level, (_left, _right, band) in enumerate(ranked_bands, start=1):
        segment = _Segment(expected[level], 0.0, 0.0)
        for detection, _start, _end in band:
            assignments.append((detection, segment))
            levels.append(level)
    return assignments, levels


def _primary_partition_is_complete(
    events: list[_Event],
    assignments: list[tuple[SemanticDetection, _Segment]],
    levels: list[int],
) -> bool:
    covered = sorted(
        index
        for (_detection, segment), level in zip(
            assignments,
            levels,
            strict=True,
        )
        if level == 1
        for index in segment.event_indices
    )
    expected = sorted(
        index
        for run in _eligible_runs(events)
        for index in run
    )
    return bool(expected and covered == expected)


def _irregular_complete_high_level_partition(
    events: list[_Event],
    assignments: list[tuple[SemanticDetection, _Segment]],
    levels: list[int],
) -> bool:
    primary = [
        segment
        for (_detection, segment), level in zip(
            assignments,
            levels,
            strict=True,
        )
        if level == 1
    ]
    if len(primary) < 2:
        return False
    runs = _eligible_runs(events)
    if len(runs) != 1:
        return False
    run = runs[0]
    covered = sorted(
        index
        for segment in primary
        for index in segment.event_indices
    )
    if (
        covered != list(run)
        or len(run) % len(primary)
        or any(events[index].level_count < 2 for index in run)
        or len({events[index].duration for index in run}) != 1
        or max(segment.endpoint_error for segment in primary) <= 0.04
    ):
        return False
    expected = len(run) // len(primary)
    return any(len(segment.event_indices) != expected for segment in primary)


def _assign_levels(
    events: list[_Event],
    segments: list[_Segment],
) -> list[int] | None:
    order = sorted(
        range(len(segments)),
        key=lambda index: (
            -len(segments[index].event_indices),
            segments[index].event_indices[0],
            segments[index].event_indices[-1],
            index,
        ),
    )
    assigned: list[int | None] = [None] * len(segments)
    occupied: dict[int, set[int]] = {}
    for segment_index in order:
        indices = segments[segment_index].event_indices
        maximum_level = min(events[index].level_count for index in indices)
        selected_level = None
        for level in range(1, maximum_level + 1):
            if occupied.get(level, set()).intersection(indices):
                continue
            if level > 1 and not set(indices) <= occupied.get(level - 1, set()):
                continue
            if level == 1 and len(indices) < 2:
                continue
            selected_level = level
            break
        if selected_level is None:
            return None
        assigned[segment_index] = selected_level
        occupied.setdefault(selected_level, set()).update(indices)
    return [int(value) for value in assigned if value is not None]


def _insert_beam(
    note: etree._Element,
    *,
    level: int,
    value: str,
) -> None:
    beam = etree.Element("beam", number=str(level))
    beam.text = value
    insert_at = len(note)
    for index, child in enumerate(note):
        if child.tag in {"notations", "lyric", "play", "listen"}:
            insert_at = index
            break
    note.insert(insert_at, beam)


def _write_segment(
    events: list[_Event],
    segment: _Segment,
    level: int,
) -> None:
    indices = segment.event_indices
    if len(indices) == 1:
        index = indices[0]
        lower_groups: list[list[int]] = []
        active: list[int] = []
        for other_index, event in enumerate(events):
            lower = event.element.find(f"beam[@number='{level - 1}']")
            value = (
                " ".join((lower.text or "").split()).casefold()
                if lower is not None
                else ""
            )
            if value == "begin":
                active = [other_index]
            elif value == "continue" and active:
                active.append(other_index)
            elif value == "end" and active:
                active.append(other_index)
                lower_groups.append(active)
                active = []
        containing_lower = next(
            (
                group
                for group in lower_groups
                if index in group
            ),
            [],
        )
        if containing_lower and index == min(containing_lower):
            value = "forward hook"
        elif containing_lower and index == max(containing_lower):
            value = "backward hook"
        else:
            previous_distance = min(
                (index - item for item in containing_lower if item < index),
                default=10**6,
            )
            next_distance = min(
                (item - index for item in containing_lower if item > index),
                default=10**6,
            )
            value = (
                "backward hook"
                if previous_distance < next_distance
                else "forward hook"
            )
        _insert_beam(events[index].element, level=level, value=value)
        return
    for position, index in enumerate(indices):
        value = (
            "begin"
            if position == 0
            else "end"
            if position == len(indices) - 1
            else "continue"
        )
        _insert_beam(events[index].element, level=level, value=value)


def _measure_system_index(part: Any, measure_index: int) -> int | None:
    return next(
        (
            group_index
            for group_index, (offset, count) in enumerate(
                zip(
                    part.system_measure_offsets,
                    part.system_measure_counts,
                    strict=True,
                )
            )
            if offset <= measure_index < offset + count
        ),
        None,
    )


def _nearest_run_event(
    events: list[_Event],
    run: tuple[int, ...],
    *,
    offset: float,
    measure_duration: int,
) -> tuple[int | None, float | None, str | None]:
    ranked = sorted(
        (
            abs(
                _event_lattice_position(
                    events[index],
                    measure_duration=measure_duration,
                )
                - offset
            ),
            index,
        )
        for index in run
    )
    if not ranked or ranked[0][0] > MAXIMUM_ENDPOINT_ERROR:
        return None, None, "cross-measure beam endpoint misses the event lattice"
    if (
        len(ranked) > 1
        and ranked[1][0] - ranked[0][0] < MINIMUM_ASSIGNMENT_MARGIN
    ):
        return None, None, "cross-measure beam endpoint is ambiguous"
    return ranked[0][1], float(ranked[0][0]), None


def enrich_musicxml_with_source_beams(
    xml_path: Path,
    layout: PageLayout,
    detections: Iterable[SemanticDetection],
) -> BeamEnrichmentReport:
    xml_path = xml_path.resolve()
    all_detections = tuple(detections)
    beam_detections = tuple(
        item for item in all_detections if item.class_name == "beam"
    )
    if not beam_detections:
        return BeamEnrichmentReport(
            str(xml_path),
            BEAM_ENRICHMENT_VERSION,
            False,
            0,
            (),
        )
    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        recover=False,
        huge_tree=False,
    )
    tree = etree.parse(str(xml_path), parser)
    root = tree.getroot()
    topology, topology_error = _build_topology(root, layout)
    if topology is None:
        proposals = tuple(
            BeamProposal(
                item,
                None,
                None,
                None,
                (),
                None,
                None,
                False,
                False,
                topology_error or "source topology is unresolved",
            )
            for item in beam_detections
        )
        return BeamEnrichmentReport(
            str(xml_path),
            BEAM_ENRICHMENT_VERSION,
            False,
            len(beam_detections),
            proposals,
            topology_error,
        )

    if _topology_uses_conditioned_layout(topology, layout):
        normalized_detections: list[SemanticDetection] = []
        for item in all_detections:
            assigned = _assign_bbox_to_appearance(topology, item.bbox)
            if assigned is None or assigned == (item.staff_index, item.placement):
                normalized_detections.append(item)
                continue
            normalized_detections.append(
                SemanticDetection(
                    item.class_name,
                    item.label,
                    item.bbox,
                    item.confidence,
                    assigned[0],
                    assigned[1],
                )
            )
        all_detections = tuple(normalized_detections)
        beam_detections = tuple(
            item for item in all_detections if item.class_name == "beam"
        )

    all_detections = _normalize_parallel_beam_staff_assignments(
        topology,
        all_detections,
    )
    beam_detections = tuple(
        item for item in all_detections if item.class_name == "beam"
    )

    group_by_staff_index = {
        staff_index: group_index
        for staff_index, group_index, _part_index, _part_staff in (
            topology.appearance_locations
        )
    }
    semantic_barlines_by_group: dict[int, list[SemanticDetection]] = {}
    for item in all_detections:
        if item.class_name != "genericBarline":
            continue
        group_index = group_by_staff_index.get(item.staff_index)
        if group_index is None:
            continue
        semantic_barlines_by_group.setdefault(group_index, []).append(item)

    mapped: list[
        tuple[
            SemanticDetection,
            int,
            int,
            int,
            float,
            float,
        ]
    ] = []
    cross_mapped: list[
        tuple[
            SemanticDetection,
            int,
            int,
            int,
            int,
            float,
            float,
        ]
    ] = []
    initial: list[BeamProposal] = []
    for detection in beam_detections:
        detection_group = group_by_staff_index.get(detection.staff_index)
        group_barlines = (
            tuple(
                semantic_barlines_by_group.get(
                    detection_group,
                    (),
                )
            )
            if detection_group is not None
            else ()
        )
        start, start_error = _map_beam_anchor(
            topology,
            layout,
            detection,
            detection.bbox[0],
            semantic_barlines=group_barlines,
        )
        end, end_error = _map_beam_anchor(
            topology,
            layout,
            detection,
            detection.bbox[2],
            semantic_barlines=group_barlines,
        )
        reason = start_error or end_error
        if (
            start is None
            or end is None
            or start.part_index != end.part_index
            or start.staff != end.staff
        ):
            initial.append(
                BeamProposal(
                    detection,
                    start.part_index if start is not None else None,
                    start.staff if start is not None else None,
                    start.measure_index if start is not None else None,
                    (),
                    None,
                    None,
                    False,
                    False,
                    reason or "beam endpoints do not map to one staff measure",
                )
            )
            continue
        if min(start.confidence, end.confidence) < MINIMUM_MAPPING_CONFIDENCE:
            initial.append(
                BeamProposal(
                    detection,
                    start.part_index,
                    start.staff,
                    start.measure_index,
                    (),
                    None,
                    None,
                    False,
                    False,
                    "source-to-measure mapping confidence is too low",
                )
            )
            continue
        if start.measure_index != end.measure_index:
            if start.measure_index < end.measure_index:
                cross_mapped.append(
                    (
                        detection,
                        start.part_index,
                        start.staff,
                        start.measure_index,
                        end.measure_index,
                        start.offset_ratio,
                        end.offset_ratio,
                    )
                )
            else:
                initial.append(
                    BeamProposal(
                        detection,
                        start.part_index,
                        start.staff,
                        start.measure_index,
                        (),
                        None,
                        None,
                        False,
                        False,
                        "cross-measure beam endpoints are reversed",
                    )
                )
            continue
        mapped.append(
            (
                detection,
                start.part_index,
                start.staff,
                start.measure_index,
                start.offset_ratio,
                end.offset_ratio,
            )
        )

    grouped: dict[
        tuple[int, int, int],
        list[tuple[SemanticDetection, float, float]],
    ] = {}
    for detection, part_index, staff, measure_index, start, end in mapped:
        grouped.setdefault((part_index, staff, measure_index), []).append(
            (detection, start, end)
        )

    committed: list[
        tuple[
            SemanticDetection,
            int,
            int,
            int,
            list[_Event],
            _Segment,
            int,
        ]
    ] = []
    proposals = list(initial)
    cross_grouped: dict[
        tuple[int, int, int, int],
        list[tuple[SemanticDetection, float, float]],
    ] = {}
    written_targets: set[
        tuple[int, int, int, int, tuple[int, ...]]
    ] = set()
    for (
        detection,
        part_index,
        staff,
        start_measure,
        end_measure,
        start_offset,
        end_offset,
    ) in cross_mapped:
        cross_grouped.setdefault(
            (part_index, staff, start_measure, end_measure),
            [],
        ).append((detection, start_offset, end_offset))

    for key, group in sorted(cross_grouped.items()):
        part_index, staff, start_measure, end_measure = key
        part = topology.parts[part_index]
        assignment_error: str | None = None
        if (
            end_measure - start_measure > 3
            or _measure_system_index(part, start_measure)
            != _measure_system_index(part, end_measure)
        ):
            assignment_error = (
                "cross-measure beam is outside one bounded score system"
            )
        elif (
            max(item[1] for item in group)
            - min(item[1] for item in group)
            > 0.03
            or max(item[2] for item in group)
            - min(item[2] for item in group)
            > 0.03
        ):
            assignment_error = "cross-measure beam levels do not share endpoints"

        measure_events: list[tuple[list[_Event], tuple[int, ...], int]] = []
        if assignment_error is None:
            for measure_index in range(start_measure, end_measure + 1):
                measure = part.measures[measure_index]
                events, event_error = _events_for_staff(measure, staff)
                runs = _eligible_runs(events) if event_error is None else ()
                if event_error is not None:
                    assignment_error = event_error
                    break
                if len(runs) != 1:
                    assignment_error = (
                        "cross-measure beam does not have one contiguous run per measure"
                    )
                    break
                run = runs[0]
                if (
                    measure_index == start_measure
                    and run[-1] != len(events) - 1
                ) or (
                    measure_index == end_measure
                    and run[0] != 0
                ) or (
                    start_measure < measure_index < end_measure
                    and run != tuple(range(len(events)))
                ):
                    assignment_error = (
                        "cross-measure beam is interrupted inside the recognized voice"
                    )
                    break
                measure_events.append(
                    (events, run, _measure_duration(measure))
                )

        start_index: int | None = None
        end_index: int | None = None
        start_error_value: float | None = None
        end_error_value: float | None = None
        if assignment_error is None:
            start_index, start_error_value, assignment_error = _nearest_run_event(
                measure_events[0][0],
                measure_events[0][1],
                offset=float(median(item[1] for item in group)),
                measure_duration=measure_events[0][2],
            )
        if assignment_error is None:
            end_index, end_error_value, assignment_error = _nearest_run_event(
                measure_events[-1][0],
                measure_events[-1][1],
                offset=float(median(item[2] for item in group)),
                measure_duration=measure_events[-1][2],
            )

        flattened: list[_Event] = []
        if (
            assignment_error is None
            and start_index is not None
            and end_index is not None
        ):
            for measure_position, (events, run, _duration) in enumerate(
                measure_events
            ):
                selected = run
                if measure_position == 0:
                    selected = selected[selected.index(start_index) :]
                if measure_position == len(measure_events) - 1:
                    selected = selected[: selected.index(end_index) + 1]
                for source_index in selected:
                    source = events[source_index]
                    flattened.append(
                        _Event(
                            source.element,
                            len(flattened),
                            1,
                            source.level_count,
                            len(flattened),
                            None,
                        )
                    )
            if len(flattened) < 2:
                assignment_error = (
                    "cross-measure beam contains fewer than two events"
                )

        segments = [
            _Segment(tuple(range(len(flattened))), 0.0, 0.0)
            for _item in group
        ]
        levels = (
            _assign_levels(flattened, segments)
            if assignment_error is None
            else None
        )
        if levels is None and assignment_error is None:
            assignment_error = (
                "cross-measure beam cannot form a complete level hierarchy"
            )
        if assignment_error is not None or levels is None:
            proposals.extend(
                BeamProposal(
                    detection,
                    part_index,
                    staff,
                    start_measure,
                    (),
                    None,
                    None,
                    False,
                    False,
                    assignment_error or "cross-measure beam assignment failed",
                )
                for detection, _start, _end in group
            )
            continue
        endpoint_error = max(
            start_error_value or 0.0,
            end_error_value or 0.0,
        )
        for (
            detection,
            _start,
            _end,
        ), segment, level in zip(
            group,
            segments,
            levels,
            strict=True,
        ):
            committed.append(
                (
                    detection,
                    part_index,
                    staff,
                    start_measure,
                    flattened,
                    _Segment(
                        segment.event_indices,
                        endpoint_error,
                        (start_error_value or 0.0)
                        + (end_error_value or 0.0),
                    ),
                    level,
                )
            )

    for key, group in sorted(grouped.items()):
        part_index, staff, measure_index = key
        measure = topology.parts[part_index].measures[measure_index]
        duration = _measure_duration(measure)
        system_opening = (
            measure_index
            in topology.parts[part_index].system_measure_offsets
        )
        voices = _staff_voices(measure, staff)
        part = topology.parts[part_index]
        source_staff_spacing = float(
            median(
                appearance.spacing
                for appearance in topology.ordered_appearances
            )
        )
        voice_staves: dict[int, set[int]] = {}
        for note in measure.findall("note"):
            note_voice = max(1, _integer(note.findtext("voice"), 1))
            note_staff = max(1, _integer(note.findtext("staff"), 1))
            if 1 <= note_staff <= part.staff_count:
                voice_staves.setdefault(note_voice, set()).add(note_staff)

        def attempt_group(
            target_voice: int | None,
            target_group: list[
                tuple[SemanticDetection, float, float]
            ],
        ) -> tuple[
            list[_Event] | None,
            list[tuple[SemanticDetection, _Segment]] | None,
            list[int] | None,
            str | None,
        ]:
            selected_staves = (
                tuple(sorted(voice_staves.get(target_voice, ())))
                if (
                    target_voice is not None
                    and (
                        len(voice_staves.get(target_voice, ())) > 1
                        or staff
                        not in voice_staves.get(target_voice, ())
                    )
                )
                else None
            )
            events, event_error = _events_for_staff(
                measure,
                staff,
                target_voice=target_voice,
                voice_staves=selected_staves,
            )
            if event_error is not None:
                return None, None, None, event_error
            assignments, levels, assignment_error = _assign_group(
                events,
                target_group,
                measure_duration=duration,
                system_opening=system_opening,
                source_staff_spacing=source_staff_spacing,
            )
            if (
                assignment_error is None
                and assignments is not None
                and levels is not None
            ):
                assignments, levels = (
                    _merge_repeated_high_level_path_splits(
                        events,
                        assignments,
                        levels,
                        source_staff_spacing=source_staff_spacing,
                    )
                )
                return events, assignments, levels, None
            return (
                None,
                None,
                None,
                assignment_error or "joint beam assignment failed",
            )

        accepted_units: list[
            tuple[
                list[_Event],
                list[tuple[SemanticDetection, _Segment]],
                list[int],
            ]
        ] = []
        rejected_units: list[
            tuple[
                list[tuple[SemanticDetection, float, float]],
                str,
            ]
        ] = []
        all_part_voices = tuple(sorted(voice_staves))
        component_voices: tuple[int, ...] = ()
        if len(voices) <= 1:
            target_voice: int | None = None
            if (
                len(voices) == 1
                and len(voice_staves.get(voices[0], ())) > 1
            ):
                target_voice = voices[0]
            events, assignments, levels, assignment_error = attempt_group(
                target_voice,
                group,
            )
            if (
                assignment_error is None
                and events is not None
                and assignments is not None
                and levels is not None
            ):
                accepted_units.append((events, assignments, levels))
            elif (
                part.staff_count > 1
                and len(all_part_voices) > 1
            ):
                # A long stem or cross-staff beam can lie closer to the adjacent
                # keyboard staff.  Only after the owner-staff interpretation
                # fails, retry connected beam objects against every voice in
                # this keyboard part; unique event-lattice fit remains required.
                component_voices = all_part_voices
            else:
                rejected_units.append(
                    (
                        group,
                        assignment_error or "joint beam assignment failed",
                    )
                )
        else:
            component_voices = tuple(voices)

        if component_voices:
            full_group_candidates: list[
                tuple[
                    int,
                    list[_Event],
                    list[tuple[SemanticDetection, _Segment]],
                    list[int],
                    float,
                    float | None,
                ]
            ] = []
            for target_voice in component_voices:
                (
                    events,
                    assignments,
                    levels,
                    assignment_error,
                ) = attempt_group(target_voice, group)
                if (
                    assignment_error is None
                    and events is not None
                    and assignments is not None
                    and levels is not None
                ):
                    full_group_candidates.append(
                        (
                            target_voice,
                            events,
                            assignments,
                            levels,
                            _beam_assignment_staff_membership_score(
                                events,
                                assignments,
                                source_staff=staff,
                            ),
                            _beam_assignment_stem_placement_score(
                                events,
                                assignments,
                            )
                            if staff
                            in voice_staves.get(target_voice, ())
                            else None,
                        )
                    )
            selected_full: tuple[
                int,
                list[_Event],
                list[tuple[SemanticDetection, _Segment]],
                list[int],
                float,
                float | None,
            ] | None = None
            selected_voice = _select_unique_voice_by_geometry(
                [
                    (candidate[0], candidate[4], candidate[5])
                    for candidate in full_group_candidates
                ]
            )
            if selected_voice is not None:
                selected_full = next(
                    candidate
                    for candidate in full_group_candidates
                    if candidate[0] == selected_voice
                )
            if selected_full is not None:
                accepted_units.append(
                    (
                        selected_full[1],
                        selected_full[2],
                        selected_full[3],
                    )
                )
                component_voices = ()

        if component_voices:
            components = _connected_beam_groups(
                group,
                source_staff_spacing=source_staff_spacing,
            )
            assigned_by_voice: dict[
                int,
                list[tuple[SemanticDetection, float, float]],
            ] = {}
            for component in components:
                candidates: list[
                    tuple[
                        int,
                        list[_Event],
                        list[tuple[SemanticDetection, _Segment]],
                        list[int],
                        float,
                        float | None,
                    ]
                ] = []
                candidate_errors: list[str] = []
                for target_voice in component_voices:
                    (
                        events,
                        assignments,
                        levels,
                        assignment_error,
                    ) = attempt_group(target_voice, component)
                    if (
                        assignment_error is None
                        and events is not None
                        and assignments is not None
                        and levels is not None
                    ):
                        candidates.append(
                            (
                                target_voice,
                                events,
                                assignments,
                                levels,
                                _beam_assignment_staff_membership_score(
                                    events,
                                    assignments,
                                    source_staff=staff,
                                ),
                                _beam_assignment_stem_placement_score(
                                    events,
                                    assignments,
                                )
                                if staff
                                in voice_staves.get(target_voice, ())
                                else None,
                            )
                        )
                    elif assignment_error is not None:
                        candidate_errors.append(assignment_error)

                selected_voice = _select_unique_voice_by_geometry(
                    [
                        (
                            voice,
                            membership_score,
                            stem_score,
                        )
                        for (
                            voice,
                            _events,
                            _assignments,
                            _levels,
                            membership_score,
                            stem_score,
                        ) in candidates
                    ]
                )
                if selected_voice is None:
                    rejected_units.append(
                        (
                            component,
                            (
                                "multiple voices do not yield one unique "
                                "beam assignment"
                                if candidates
                                else (
                                    candidate_errors[0]
                                    if candidate_errors
                                    else "joint beam assignment failed"
                                )
                            ),
                        )
                    )
                    continue
                assigned_by_voice.setdefault(selected_voice, []).extend(
                    component
                )

            for target_voice, target_group in sorted(
                assigned_by_voice.items()
            ):
                (
                    events,
                    assignments,
                    levels,
                    assignment_error,
                ) = attempt_group(target_voice, target_group)
                if (
                    assignment_error is None
                    and events is not None
                    and assignments is not None
                    and levels is not None
                ):
                    accepted_units.append((events, assignments, levels))
                else:
                    rejected_units.append(
                        (
                            target_group,
                            assignment_error
                            or "joint beam assignment failed",
                        )
                    )

        for rejected_group, assignment_error in rejected_units:
            proposals.extend(
                BeamProposal(
                    detection,
                    part_index,
                    staff,
                    measure_index,
                    (),
                    None,
                    None,
                    False,
                    False,
                    assignment_error,
                )
                for detection, _start, _end in rejected_group
            )
        for events, assignments, levels in accepted_units:
            for (detection, segment), level in zip(
                assignments,
                levels,
                strict=True,
            ):
                committed.append(
                    (
                        detection,
                        part_index,
                        staff,
                        measure_index,
                        events,
                        segment,
                        level,
                    )
                )

    if not committed:
        return BeamEnrichmentReport(
            str(xml_path),
            BEAM_ENRICHMENT_VERSION,
            False,
            len(beam_detections),
            tuple(proposals),
        )

    before = _beam_free_digest(root)
    for (
        detection,
        part_index,
        staff,
        measure_index,
        events,
        segment,
        level,
    ) in sorted(
        committed,
        key=lambda item: (
            item[1],
            item[3],
            item[2],
            item[6],
            item[5].event_indices,
            item[0].bbox,
        ),
    ):
        target = (
            part_index,
            staff,
            measure_index,
            level,
            tuple(id(events[index].element) for index in segment.event_indices),
        )
        if target not in written_targets:
            _write_segment(events, segment, level)
            written_targets.add(target)
        proposals.append(
            BeamProposal(
                detection,
                part_index,
                staff,
                measure_index,
                segment.event_indices,
                level,
                segment.endpoint_error,
                True,
                True,
                "source-backed beam segment committed",
            )
        )

    temporary = xml_path.with_name(xml_path.name + ".beam-transaction.tmp")
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
        if errors:
            raise ValueError("; ".join(errors))
        if _beam_free_digest(root) != before:
            raise ValueError(
                "non-beam MusicXML semantics changed during enrichment"
            )
        atomic_write_bytes(xml_path, temporary.read_bytes())
    except Exception as exc:
        rolled_back = tuple(
            BeamProposal(
                item.detection,
                item.part_index,
                item.staff,
                item.measure_index,
                item.event_indices,
                item.beam_level,
                item.endpoint_error,
                item.eligible,
                False,
                (
                    f"transaction rolled back: {type(exc).__name__}: {exc}"
                    if item.injected
                    else item.reason
                ),
            )
            for item in proposals
        )
        return BeamEnrichmentReport(
            str(xml_path),
            BEAM_ENRICHMENT_VERSION,
            False,
            len(beam_detections),
            rolled_back,
            f"{type(exc).__name__}: {exc}",
        )
    finally:
        temporary.unlink(missing_ok=True)

    return BeamEnrichmentReport(
        str(xml_path),
        BEAM_ENRICHMENT_VERSION,
        True,
        len(beam_detections),
        tuple(proposals),
    )
