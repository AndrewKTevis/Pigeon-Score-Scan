from __future__ import annotations

"""Veto-only source-image confirmation for one proposed event insertion.

The independent candidate families and :mod:`event_presence_consensus` remain solely
responsible for proposing MusicXML.  This module compares the preserved local source
image with the complete before/after event sequence for exactly one already-specified
insertion.  It cannot choose pitch, duration or event kind, create candidate support,
edit XML, authorise multiple-event changes, or approve deletion.
"""

from dataclasses import dataclass
from functools import lru_cache
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from .local_event_descriptor import FEATURE_NAMES as LOCAL_EVENT_FEATURE_NAMES
from .local_event_descriptor import (
    clean_event_patch,
    crop_event_patch,
    event_patch_descriptor,
)
from .local_symbol_image import decode_symbol_guard_image, event_position
from .model_registry import load_verified_json
from .policy import DEFAULT_POLICY
from .score_ir import MeasureIR, NoteIR
from .tree_model import VerifiedRandomForestModel
from .visual_evidence import SYMBOL_GUARD_HEIGHT, SYMBOL_GUARD_WIDTH, VisualMeasureEvidence

SUPPORTED_TYPES = ("half", "quarter", "eighth", "16th")
SIDE_OFFSET_PIXELS = 24

_CONTEXT_FEATURES = (
    "operation_is_insert",
    "operation_is_delete",
    "event_is_pitched_note",
    "event_is_rest",
    *(f"note_type_{value}" for value in SUPPORTED_TYPES),
    "note_type_other",
    "dot_count_scaled",
    "onset_ratio",
    "duration_measure_ratio",
    "edge_proximity",
    "left_neighbour_gap",
    "right_neighbour_gap",
    "neighbour_count_scaled",
    "event_vertical_position",
)
_TEMPLATE_FEATURES = (
    "template_best_f1",
    "template_best_precision",
    "template_best_recall",
    "template_best_iou",
    "template_best_chamfer_support",
    "template_variant_margin",
    "template_central_ink_coverage",
    "template_outside_ink_ratio",
    "template_notehead_support",
    "template_stem_or_rest_support",
)
_COMPETING_EVENT_FEATURES = (
    "displaced_event_available",
    "displaced_event_same_kind",
    "displaced_event_same_type",
    "displaced_event_same_pitch",
    "displaced_template_f1",
    "displaced_template_precision",
    "displaced_template_recall",
    "displaced_template_iou",
    "displaced_template_chamfer",
    "proposed_minus_displaced_f1",
    "proposed_minus_displaced_chamfer",
)
_TRANSACTION_TEMPLATE_FEATURES = (
    "before_sequence_precision",
    "before_sequence_recall",
    "before_sequence_f1",
    "before_sequence_iou",
    "before_sequence_chamfer",
    "after_sequence_precision",
    "after_sequence_recall",
    "after_sequence_f1",
    "after_sequence_iou",
    "after_sequence_chamfer",
    "after_minus_before_f1",
    "after_minus_before_chamfer",
    "after_unique_observed_coverage",
    "before_unique_observed_coverage",
    "after_unique_observed_share",
    "before_unique_observed_share",
    "unique_coverage_margin",
    "unique_share_margin",
)
_DESCRIPTOR_FEATURES = tuple(
    f"{prefix}_{name}"
    for prefix in ("centre", "left_context", "right_context", "centre_vs_context")
    for name in LOCAL_EVENT_FEATURE_NAMES
)
EVENT_PRESENCE_VISUAL_CONTEXT_FEATURE_COUNT = len(_CONTEXT_FEATURES)
EVENT_PRESENCE_VISUAL_FEATURE_NAMES = (
    _CONTEXT_FEATURES
    + _TEMPLATE_FEATURES
    + _COMPETING_EVENT_FEATURES
    + _TRANSACTION_TEMPLATE_FEATURES
    + _DESCRIPTOR_FEATURES
)


def _note_type(note: NoteIR) -> str:
    return str(note.note_type or "").strip().casefold()


def _draw_rest_template(canvas: np.ndarray, note_type: str) -> None:
    x = y = 24
    if note_type == "half":
        cv2.rectangle(canvas, (x - 8, y - 1), (x + 8, y + 5), 255, -1)
    elif note_type == "quarter":
        points = np.asarray(
            [(x - 3, y - 15), (x + 4, y - 7), (x - 2, y),
             (x + 5, y + 7), (x - 4, y + 16), (x + 3, y + 13)],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [points], False, 255, 3, cv2.LINE_AA)
    else:
        cv2.line(canvas, (x + 3, y - 15), (x - 2, y + 12), 255, 2, cv2.LINE_AA)
        cv2.ellipse(canvas, (x + 2, y - 12), (5, 3), -20, 0, 360, 255, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (x - 3, y + 10), (4, 3), -20, 0, 360, 255, -1, cv2.LINE_AA)
        if note_type == "16th":
            cv2.ellipse(canvas, (x + 3, y - 4), (4, 3), -20, 0, 360, 255, -1, cv2.LINE_AA)


def _draw_note_template(canvas: np.ndarray, note_type: str, stem_up: bool) -> None:
    x = y = 24
    if note_type == "half":
        cv2.ellipse(canvas, (x, y), (6, 4), -12, 0, 360, 255, 2, cv2.LINE_AA)
    else:
        cv2.ellipse(canvas, (x, y), (6, 4), -12, 0, 360, 255, -1, cv2.LINE_AA)
    stem_x = x + 5 if stem_up else x - 5
    stem_end = y - 22 if stem_up else y + 22
    cv2.line(canvas, (stem_x, y), (stem_x, stem_end), 255, 2, cv2.LINE_AA)
    if note_type in {"eighth", "16th"}:
        direction = 1 if stem_up else -1
        count = 1 if note_type == "eighth" else 2
        for index in range(count):
            flag_y = stem_end + direction * index * 6
            points = np.asarray(
                [(stem_x, flag_y), (stem_x + direction * 10, flag_y + direction * 6),
                 (stem_x + direction * 5, flag_y + direction * 12)],
                dtype=np.int32,
            )
            cv2.polylines(canvas, [points], False, 255, 2, cv2.LINE_AA)


@lru_cache(maxsize=16)
def _event_templates(rest: bool, note_type: str) -> tuple[np.ndarray, ...]:
    bases: list[np.ndarray] = []
    if rest:
        canvas = np.zeros((48, 48), dtype=np.uint8)
        _draw_rest_template(canvas, note_type)
        bases.append(canvas)
    else:
        for stem_up in (False, True):
            canvas = np.zeros((48, 48), dtype=np.uint8)
            _draw_note_template(canvas, note_type, stem_up)
            bases.append(canvas)
    variants: list[np.ndarray] = []
    for base in bases:
        for dx, dy in ((0, 0), (-2, 0), (2, 0), (0, -2), (0, 2)):
            matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(base, matrix, (48, 48), flags=cv2.INTER_NEAREST)
            shifted.setflags(write=False)
            variants.append(shifted)
    return tuple(variants)


def _template_features(image: np.ndarray, x_ratio: float, y_ratio: float, event: NoteIR) -> np.ndarray:
    patch = crop_event_patch(image, x_ratio, y_ratio)
    observed = clean_event_patch(patch) >= 40
    observed_centre = observed[3:45, 3:45]
    observed_count = max(int(np.sum(observed_centre)), 1)
    distance = cv2.distanceTransform((~observed).astype(np.uint8), cv2.DIST_L2, 3)
    rows: list[tuple[float, ...]] = []
    for raw in _event_templates(bool(event.rest), _note_type(event)):
        template = raw >= 40
        expanded = cv2.dilate(template.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        overlap = int(np.sum(observed & expanded))
        template_count = max(int(np.sum(template)), 1)
        precision = overlap / observed_count
        recall = overlap / template_count
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
        union = max(int(np.sum(observed | expanded)), 1)
        iou = int(np.sum(observed & expanded)) / union
        points = distance[template]
        chamfer = 1.0 - min(1.0, float(np.mean(points.astype(np.float64))) / 8.0) if points.size else 0.0
        rows.append((f1, precision, recall, iou, chamfer))
    ranked = sorted(rows, key=lambda row: (row[0], row[4]), reverse=True)
    best = ranked[0] if ranked else (0.0,) * 5
    margin = best[0] - (ranked[1][0] if len(ranked) > 1 else 0.0)
    central = observed[14:34, 12:36]
    outside = observed.copy()
    outside[10:38, 8:40] = False
    notehead = float(np.mean(observed[19:30, 16:32]))
    if event.rest:
        structure = float(np.mean(observed[9:40, 14:34]))
    else:
        up = float(np.mean(observed[2:25, 26:32]))
        down = float(np.mean(observed[23:46, 16:22]))
        structure = max(up, down)
    return np.asarray(
        [*best, margin, float(np.mean(central)), float(np.mean(outside)), notehead, structure],
        dtype=np.float64,
    )


def _best_template_score(
    image: np.ndarray, x_ratio: float, y_ratio: float, event: NoteIR
) -> tuple[float, float, float, float, float]:
    """Return the best local symbol-template score without context-only extras."""
    values = _template_features(image, x_ratio, y_ratio, event)
    return tuple(float(value) for value in values[:5])  # type: ignore[return-value]


def _competing_event_features(
    image: np.ndarray,
    before: MeasureIR,
    after: MeasureIR,
    operation: str,
    event_index: int,
    event: NoteIR,
    proposed_template_values: np.ndarray,
) -> np.ndarray:
    """Compare the proposed insertion with the event displaced from that position.

    Under a coherent insertion, the source-before sequence has its next event at the
    proposed onset, while the source-after sequence has the inserted event there and
    shifts the suffix.  Comparing both symbol hypotheses at the same transaction
    position is substantially more informative than asking whether any ink exists.
    """
    displaced: NoteIR | None = None
    displaced_measure = before
    if operation == "insert" and 0 <= event_index < len(before.notes):
        displaced = before.notes[event_index]
    elif operation == "delete" and 0 <= event_index < len(after.notes):
        displaced = after.notes[event_index]
        displaced_measure = after
    if (
        displaced is None
        or displaced.grace
        or displaced.chord
        or (displaced.pitch is None and not displaced.rest)
        or _note_type(displaced) not in SUPPORTED_TYPES
    ):
        return np.zeros(len(_COMPETING_EVENT_FEATURES), dtype=np.float64)

    displaced_x, displaced_y = event_position(displaced_measure, displaced)
    displaced_score = _best_template_score(
        image, displaced_x, displaced_y, displaced
    )
    proposed_score = tuple(float(value) for value in proposed_template_values[:5])
    same_pitch = (
        not event.rest
        and not displaced.rest
        and event.pitch is not None
        and event.pitch == displaced.pitch
    )
    return np.asarray(
        [
            1.0,
            1.0 if bool(event.rest) == bool(displaced.rest) else 0.0,
            1.0 if _note_type(event) == _note_type(displaced) else 0.0,
            1.0 if same_pitch else 0.0,
            *displaced_score,
            proposed_score[0] - displaced_score[0],
            proposed_score[4] - displaced_score[4],
        ],
        dtype=np.float64,
    )


def _paste_mask(canvas: np.ndarray, mask: np.ndarray, centre_x: int, centre_y: int) -> None:
    half_h = mask.shape[0] // 2
    half_w = mask.shape[1] // 2
    x0 = centre_x - half_w
    y0 = centre_y - half_h
    x1 = x0 + mask.shape[1]
    y1 = y0 + mask.shape[0]
    target_x0 = max(0, x0)
    target_y0 = max(0, y0)
    target_x1 = min(canvas.shape[1], x1)
    target_y1 = min(canvas.shape[0], y1)
    if target_x0 >= target_x1 or target_y0 >= target_y1:
        return
    source_x0 = target_x0 - x0
    source_y0 = target_y0 - y0
    source_x1 = source_x0 + (target_x1 - target_x0)
    source_y1 = source_y0 + (target_y1 - target_y0)
    canvas[target_y0:target_y1, target_x0:target_x1] |= mask[
        source_y0:source_y1, source_x0:source_x1
    ]


def _semantic_template_mask(measure: MeasureIR) -> np.ndarray:
    canvas = np.zeros((SYMBOL_GUARD_HEIGHT, SYMBOL_GUARD_WIDTH), dtype=bool)
    for event in _anchors(measure):
        note_type = _note_type(event)
        if event.grace or event.chord or note_type not in SUPPORTED_TYPES:
            continue
        x_ratio, y_ratio = event_position(measure, event)
        x = int(round(x_ratio * (SYMBOL_GUARD_WIDTH - 1)))
        y = int(round(y_ratio * (SYMBOL_GUARD_HEIGHT - 1)))
        variants = _event_templates(bool(event.rest), note_type)
        union = np.zeros((48, 48), dtype=bool)
        for variant in variants:
            union |= variant >= 40
        _paste_mask(canvas, union, x, y)
        for dot_index in range(min(max(int(event.dots), 0), 2)):
            dot_x = min(SYMBOL_GUARD_WIDTH - 1, x + 11 + dot_index * 6)
            cv2.circle(canvas.view(np.uint8), (dot_x, y), 2, 1, -1, cv2.LINE_8)
    return canvas


def _sequence_template_score(observed: np.ndarray, template: np.ndarray) -> tuple[float, ...]:
    expanded = cv2.dilate(template.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    observed_count = max(int(np.sum(observed)), 1)
    template_count = max(int(np.sum(template)), 1)
    overlap = int(np.sum(observed & expanded))
    precision = overlap / observed_count
    recall = overlap / template_count
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    union = max(int(np.sum(observed | expanded)), 1)
    iou = int(np.sum(observed & expanded)) / union
    distance = cv2.distanceTransform((~observed).astype(np.uint8), cv2.DIST_L2, 3)
    points = distance[template]
    chamfer = (
        1.0 - min(1.0, float(np.mean(points.astype(np.float64))) / 8.0)
        if points.size
        else 0.0
    )
    return precision, recall, f1, iou, chamfer


def _transaction_template_features(
    image: np.ndarray, before: MeasureIR, after: MeasureIR, event: NoteIR
) -> np.ndarray:
    x_ratio, _ = event_position(after, event)
    centre_x = int(round(x_ratio * (SYMBOL_GUARD_WIDTH - 1)))
    x0 = max(0, centre_x - 72)
    x1 = min(SYMBOL_GUARD_WIDTH, centre_x + 73)
    observed_full = image >= 40
    before_full = _semantic_template_mask(before)
    after_full = _semantic_template_mask(after)
    observed = observed_full[:, x0:x1]
    before_mask = before_full[:, x0:x1]
    after_mask = after_full[:, x0:x1]
    before_score = _sequence_template_score(observed, before_mask)
    after_score = _sequence_template_score(observed, after_mask)
    before_expanded = cv2.dilate(before_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    after_expanded = cv2.dilate(after_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    after_unique = after_expanded & ~before_expanded
    before_unique = before_expanded & ~after_expanded
    observed_count = max(int(np.sum(observed)), 1)

    def coverage(mask: np.ndarray) -> float:
        return float(np.sum(observed & mask)) / max(int(np.sum(mask)), 1)

    def share(mask: np.ndarray) -> float:
        return float(np.sum(observed & mask)) / observed_count

    after_coverage = coverage(after_unique)
    before_coverage = coverage(before_unique)
    after_share = share(after_unique)
    before_share = share(before_unique)
    return np.asarray(
        [
            *before_score,
            *after_score,
            after_score[2] - before_score[2],
            after_score[4] - before_score[4],
            after_coverage,
            before_coverage,
            after_share,
            before_share,
            after_coverage - before_coverage,
            after_share - before_share,
        ],
        dtype=np.float64,
    )


def _anchors(measure: MeasureIR) -> tuple[NoteIR, ...]:
    return tuple(note for note in measure.notes if not note.grace and not note.chord)


def _same_event_content(left: NoteIR, right: NoteIR) -> bool:
    """Compare one event while deliberately excluding its timeline onset."""
    return left.stable_tuple()[1:] == right.stable_tuple()[1:]


def _consistent_suffix_shift(
    before: tuple[NoteIR, ...],
    after: tuple[NoteIR, ...],
    allowed_shifts: tuple[Fraction, ...],
) -> bool:
    if len(before) != len(after):
        return False
    if not before:
        return True
    shifts: list[Fraction] = []
    for old, new in zip(before, after, strict=True):
        if not _same_event_content(old, new):
            return False
        shifts.append(new.onset - old.onset)
    return len(set(shifts)) == 1 and shifts[0] in allowed_shifts


def _single_edit_event(
    before: MeasureIR,
    after: MeasureIR,
    operation: str,
    event_index: int,
) -> NoteIR | None:
    """Return the sole changed event under a coherent insertion/deletion timeline.

    MusicXML event insertion normally shifts every following event by the inserted
    duration.  A pre-existing explicit gap may instead leave following onsets
    unchanged.  Both representations are accepted, but a partially shifted suffix is
    rejected because it no longer describes one atomic event transaction.
    """
    if operation == "insert":
        if len(after.notes) != len(before.notes) + 1 or not (0 <= event_index < len(after.notes)):
            return None
        event = after.notes[event_index]
        if tuple(note.stable_tuple() for note in before.notes[:event_index]) != tuple(
            note.stable_tuple() for note in after.notes[:event_index]
        ):
            return None
        if not _consistent_suffix_shift(
            before.notes[event_index:],
            after.notes[event_index + 1 :],
            (Fraction(0, 1), event.duration),
        ):
            return None
        return event
    if operation == "delete":
        if len(before.notes) != len(after.notes) + 1 or not (0 <= event_index < len(before.notes)):
            return None
        event = before.notes[event_index]
        if tuple(note.stable_tuple() for note in before.notes[:event_index]) != tuple(
            note.stable_tuple() for note in after.notes[:event_index]
        ):
            return None
        if not _consistent_suffix_shift(
            before.notes[event_index + 1 :],
            after.notes[event_index:],
            (Fraction(0, 1), -event.duration),
        ):
            return None
        return event
    return None


def _expected_duration(measure: MeasureIR) -> Fraction:
    expected = measure.expected_duration
    if expected is None or expected <= 0:
        expected = max(
            (note.onset + max(note.duration, Fraction(0, 1)) for note in _anchors(measure)),
            default=Fraction(1, 1),
        )
    return max(expected, Fraction(1, 64))


def _neighbour_gaps(measure: MeasureIR, event: NoteIR) -> tuple[float, float, int]:
    expected = _expected_duration(measure)
    other_onsets = sorted(
        note.onset
        for note in _anchors(measure)
        if note is not event and note.onset != event.onset
    )
    left = max((onset for onset in other_onsets if onset < event.onset), default=None)
    right = min((onset for onset in other_onsets if onset > event.onset), default=None)
    left_gap = 1.0 if left is None else min(1.0, float((event.onset - left) / expected) * 4.0)
    right_gap = 1.0 if right is None else min(1.0, float((right - event.onset) / expected) * 4.0)
    near_count = sum(abs(float((onset - event.onset) / expected)) <= 0.22 for onset in other_onsets)
    return left_gap, right_gap, near_count


def event_presence_visual_features(
    evidence: VisualMeasureEvidence,
    before: MeasureIR,
    after: MeasureIR,
    operation: str,
    event_index: int,
) -> tuple[float, ...] | None:
    image = decode_symbol_guard_image(evidence.symbol_guard_image)
    if image is None:
        return None
    event = _single_edit_event(before, after, operation, event_index)
    if event is None or event.grace or event.chord or (event.pitch is None and not event.rest):
        return None
    note_type = _note_type(event)
    if note_type not in SUPPORTED_TYPES:
        return None
    measure_with_event = after if operation == "insert" else before
    expected = _expected_duration(measure_with_event)
    x_ratio, y_ratio = event_position(measure_with_event, event)
    left_gap, right_gap, near_count = _neighbour_gaps(measure_with_event, event)

    template_values = _template_features(image, x_ratio, y_ratio, event)
    competing_event_values = _competing_event_features(
        image, before, after, operation, event_index, event, template_values
    )
    transaction_template_values = _transaction_template_features(
        image, before, after, event
    )
    centre = event_patch_descriptor(image, x_ratio, y_ratio)
    left = event_patch_descriptor(
        image, x_ratio, y_ratio, x_offset_pixels=-SIDE_OFFSET_PIXELS
    )
    right = event_patch_descriptor(
        image, x_ratio, y_ratio, x_offset_pixels=SIDE_OFFSET_PIXELS
    )
    context_mean = (left + right) / 2.0
    contrast = np.abs(centre - context_mean)

    onset_ratio = max(0.0, min(1.0, float(event.onset / expected)))
    edge_proximity = 1.0 - min(1.0, min(onset_ratio, 1.0 - onset_ratio) * 2.0)
    context = np.asarray(
        [
            1.0 if operation == "insert" else 0.0,
            1.0 if operation == "delete" else 0.0,
            0.0 if event.rest else 1.0,
            1.0 if event.rest else 0.0,
            *(1.0 if note_type == value else 0.0 for value in SUPPORTED_TYPES),
            1.0 if note_type not in SUPPORTED_TYPES else 0.0,
            min(max(int(event.dots), 0), 2) / 2.0,
            onset_ratio,
            max(0.0, min(1.0, float(event.duration / expected))),
            edge_proximity,
            left_gap,
            right_gap,
            min(near_count, 4) / 4.0,
            max(0.0, min(1.0, y_ratio)),
        ],
        dtype=np.float64,
    )
    vector = np.concatenate(
        (
            context,
            template_values,
            competing_event_values,
            transaction_template_values,
            centre,
            left,
            right,
            contrast,
        )
    )
    if vector.size != len(EVENT_PRESENCE_VISUAL_FEATURE_NAMES):
        raise AssertionError(
            "event-presence visual feature mismatch: "
            f"{vector.size} != {len(EVENT_PRESENCE_VISUAL_FEATURE_NAMES)}"
        )
    return tuple(float(value) for value in vector)


@dataclass(frozen=True)
class EventPresenceVisualCalibration:
    probability: float
    threshold: float
    accepted: bool
    available: bool
    model_version: str


@dataclass(frozen=True)
class EventPresenceVisualAudit:
    applicable: bool
    operation: str
    changed_event_count: int
    probability: float
    threshold: float
    accepted: bool
    reason: str
    model_version: str


class EventPresenceVisualGuard:
    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).with_name("resources") / "event_presence_visual_guard.json"
        loaded = load_verified_json(model_path, "event_presence_visual_guard")
        payload = loaded.payload
        self.model = VerifiedRandomForestModel.load(
            model_path,
            "event_presence_visual_guard",
            EVENT_PRESENCE_VISUAL_FEATURE_NAMES,
            loaded=loaded,
        )
        raw_thresholds = payload.get("auto_patch_thresholds", {})
        if not isinstance(raw_thresholds, dict):
            raw_thresholds = {}
        try:
            fallback_threshold = float(payload.get("auto_patch_threshold", 1.0))
        except (TypeError, ValueError, OverflowError):
            fallback_threshold = 1.0
        floor = float(DEFAULT_POLICY.event_presence_visual_guard_probability_floor)
        thresholds: dict[str, float] = {}
        for kind in ("note", "rest"):
            try:
                value = float(raw_thresholds.get(kind, fallback_threshold))
            except (TypeError, ValueError, OverflowError):
                value = 1.0
            thresholds[kind] = max(floor, max(0.0, min(1.0, value)))
        self.thresholds = thresholds
        self.threshold = max(thresholds.values())
        self.model_verified = bool(self.model.verified and loaded.verified)
        self.model_version = self.model.model_version
        self.enabled = bool(self.model.enabled)
        self.model_status = self.model.status if self.model.enabled else loaded.status

    def calibrate(
        self,
        evidence: VisualMeasureEvidence,
        before: MeasureIR,
        after: MeasureIR,
        operation: str,
        event_index: int,
    ) -> EventPresenceVisualCalibration:
        event = _single_edit_event(before, after, operation, event_index)
        threshold = self.thresholds["rest" if event is not None and event.rest else "note"]
        values = event_presence_visual_features(
            evidence, before, after, operation, event_index
        )
        if values is None or not (self.enabled and self.model_verified):
            return EventPresenceVisualCalibration(
                probability=0.5,
                threshold=round(threshold, 6),
                accepted=False,
                available=False,
                model_version=self.model_version,
            )
        probability = float(self.model.predict(values))
        return EventPresenceVisualCalibration(
            probability=round(probability, 6),
            threshold=round(threshold, 6),
            accepted=probability >= threshold,
            available=True,
            model_version=self.model_version,
        )

    def audit_transaction(
        self,
        evidence: VisualMeasureEvidence | None,
        before: MeasureIR,
        after: MeasureIR,
        operation: str,
        event_index: int,
    ) -> EventPresenceVisualAudit:
        encoded = "" if evidence is None else str(evidence.symbol_guard_image or "")
        if not encoded:
            return EventPresenceVisualAudit(
                False,
                operation,
                1,
                0.5,
                round(self.threshold, 6),
                True,
                "source_evidence_unavailable",
                self.model_version,
            )
        event = _single_edit_event(before, after, operation, event_index)
        threshold = round(
            self.thresholds["rest" if event is not None and event.rest else "note"], 6
        )
        if event is None:
            return EventPresenceVisualAudit(
                True,
                operation,
                1,
                0.5,
                threshold,
                False,
                "invalid_event_presence_visual_transaction",
                self.model_version,
            )
        if event.grace or event.chord or (event.pitch is None and not event.rest):
            return EventPresenceVisualAudit(
                True,
                operation,
                1,
                0.5,
                threshold,
                False,
                "unsupported_event_presence_visual_transaction",
                self.model_version,
            )
        if operation == "delete":
            return EventPresenceVisualAudit(
                True,
                operation,
                1,
                0.5,
                threshold,
                False,
                "event_deletion_requires_review",
                self.model_version,
            )
        calibration = self.calibrate(
            evidence, before, after, operation, event_index  # type: ignore[arg-type]
        )
        if not calibration.available:
            return EventPresenceVisualAudit(
                True,
                operation,
                1,
                calibration.probability,
                calibration.threshold,
                False,
                "event_presence_visual_evidence_or_model_unavailable",
                calibration.model_version,
            )
        return EventPresenceVisualAudit(
            True,
            operation,
            1,
            calibration.probability,
            calibration.threshold,
            calibration.accepted,
            (
                "visual_event_presence_confirmed"
                if calibration.accepted
                else "visual_event_presence_conflict"
            ),
            calibration.model_version,
        )
