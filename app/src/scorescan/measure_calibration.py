from __future__ import annotations

"""Learned, bounded calibration for measure-level ensemble decisions.

The calibrator never recognises notation and never bypasses XML or musical validation.
It estimates which already-parsed candidate measure is most likely to be the clean
member of an ensemble, using local semantic audits and cross-candidate agreement.  The
result only changes a candidate's weight within conservative bounds.
"""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Protocol

from .linear_model import StandardizedLogisticModel, bounded_weight
from .model_registry import load_verified_json
from .policy import DEFAULT_POLICY
from .score_ir import MeasureIR, expected_note_duration
from .tree_model import VerifiedRandomForestModel


LEGACY_FEATURE_NAMES = (
    "page_score_scaled",
    "page_probability",
    "page_valid",
    "alignment_similarity",
    "exact_support_ratio",
    "semantic_support_ratio",
    "missing_ratio",
    "distance_to_template",
    "distance_to_medoid",
    "mean_peer_distance",
    "note_count_scaled",
    "direction_count_scaled",
    "voice_count_scaled",
    "duration_error_ratio",
    "zero_duration_count",
    "type_duration_mismatch_count",
    "chord_duration_mismatch_count",
    "duplicate_direction_count",
    "pitch_outlier_count",
    "attribute_completeness",
    "internal_gap_ratio",
    "overlap_ratio",
    "terminal_gap_ratio",
    "rest_ratio",
    "chord_ratio",
    "duration_vocabulary_ratio",
    "onset_collision_count",
    "notation_density_scaled",
)

FEATURE_NAMES = LEGACY_FEATURE_NAMES + (
    "is_first_measure",
    "is_last_measure",
    "expected_duration_known",
    "accidental_alter_mismatch_count",
    "extreme_alter_count",
    "orphan_chord_count",
    "duplicate_event_count",
    "rest_pitch_conflict_count",
    "unknown_note_type_ratio",
)

_ACCIDENTAL_ALTERS = {
    "flat-flat": Fraction(-2, 1),
    "double-flat": Fraction(-2, 1),
    "flat": Fraction(-1, 1),
    "quarter-flat": Fraction(-1, 2),
    "natural": Fraction(0, 1),
    "quarter-sharp": Fraction(1, 2),
    "sharp": Fraction(1, 1),
    "double-sharp": Fraction(2, 1),
    "sharp-sharp": Fraction(2, 1),
}


class PageCandidate(Protocol):
    score: float
    calibrated_probability: float
    valid: bool


@dataclass(frozen=True)
class MeasureCalibrationInput:
    candidate: PageCandidate
    measure: MeasureIR
    alignment_similarity: float
    exact_support_ratio: float
    semantic_support_ratio: float
    missing_ratio: float
    distance_to_template: float
    distance_to_medoid: float
    mean_peer_distance: float
    is_first_measure: bool = False
    is_last_measure: bool = False


@dataclass(frozen=True)
class MeasureCalibrationResult:
    probability: float
    weight_factor: float
    model_version: str


def _measure_end(measure: MeasureIR) -> Fraction:
    regular = [note for note in measure.notes if not note.grace]
    if not regular:
        return Fraction(0, 1)
    return max((note.onset + max(note.duration, Fraction(0, 1)) for note in regular), default=Fraction(0, 1))


def _measure_statistics(measure: MeasureIR) -> dict[str, float]:
    regular = [note for note in measure.notes if not note.grace]
    expected = measure.expected_duration
    actual = _measure_end(measure)
    if expected and expected > 0:
        duration_error = min(4.0, float(abs(actual - expected) / expected))
    else:
        duration_error = 0.0

    type_mismatches = 0
    zero_durations = 0
    pitch_outliers = 0
    chord_mismatches = 0
    accidental_alter_mismatches = 0
    extreme_alters = 0
    orphan_chords = 0
    rest_pitch_conflicts = 0
    unknown_note_types = 0
    anchors: dict[Fraction, Fraction] = {}
    event_keys: list[tuple[object, ...]] = []
    for note in regular:
        if note.duration <= 0:
            zero_durations += 1
        expected_note = expected_note_duration(note)
        if expected_note is not None and note.duration > 0:
            ratio = float(max(note.duration, expected_note) / min(note.duration, expected_note))
            if ratio > 1.08:
                type_mismatches += 1
        if note.pitch and not (1200 <= note.pitch.midi_cents <= 12000):
            pitch_outliers += 1
        if note.pitch and abs(note.pitch.alter) > 2:
            extreme_alters += 1
        accidental = note.accidental.strip().casefold().replace("_", "-")
        expected_alter = _ACCIDENTAL_ALTERS.get(accidental)
        if expected_alter is not None and note.pitch is not None and note.pitch.alter != expected_alter:
            accidental_alter_mismatches += 1
        if note.rest and note.pitch is not None:
            rest_pitch_conflicts += 1
        if note.note_type and expected_note is None:
            unknown_note_types += 1
        if not note.chord:
            anchors[note.onset] = note.duration
        elif note.onset not in anchors:
            orphan_chords += 1
        elif anchors[note.onset] != note.duration:
            chord_mismatches += 1
        event_keys.append(note.stable_tuple())

    direction_keys = [direction.stable_tuple() for direction in measure.directions]
    duplicate_directions = len(direction_keys) - len(set(direction_keys))
    attributes = sum(value is not None for value in (measure.time_signature, measure.key_signature, measure.clef)) / 3.0
    anchors_sorted = sorted((note for note in regular if not note.chord), key=lambda note: (note.onset, note.duration))
    expected_denominator = expected if expected and expected > 0 else max(actual, Fraction(1, 1))
    internal_gap = Fraction(0, 1)
    overlap = Fraction(0, 1)
    collisions = 0
    previous_end: Fraction | None = None
    previous_onset: Fraction | None = None
    for note in anchors_sorted:
        if previous_onset is not None and note.onset == previous_onset:
            collisions += 1
        if previous_end is not None:
            if note.onset > previous_end:
                internal_gap += note.onset - previous_end
            elif note.onset < previous_end:
                overlap += previous_end - note.onset
        previous_onset = note.onset
        previous_end = max(previous_end or Fraction(0, 1), note.onset + max(note.duration, Fraction(0, 1)))
    terminal_gap = max(Fraction(0, 1), expected_denominator - actual)
    rests = sum(note.rest for note in regular)
    chords = sum(note.chord for note in regular)
    conventional = sum(expected_note_duration(note) is not None for note in regular)
    notation_count = sum(
        len(note.ties) + len(note.slurs) + len(note.articulations) + len(note.ornaments)
        + int(bool(note.accidental)) + int(note.dots > 0)
        for note in regular
    )
    duplicate_events = len(event_keys) - len(set(event_keys))
    return {
        "note_count": float(len(regular)),
        "direction_count": float(len(measure.directions)),
        "voice_count": float(measure.voice_count),
        "duration_error": duration_error,
        "zero_durations": float(zero_durations),
        "type_mismatches": float(type_mismatches),
        "chord_mismatches": float(chord_mismatches),
        "duplicate_directions": float(max(0, duplicate_directions)),
        "pitch_outliers": float(pitch_outliers),
        "attribute_completeness": attributes,
        "internal_gap_ratio": min(4.0, float(internal_gap / expected_denominator)),
        "overlap_ratio": min(4.0, float(overlap / expected_denominator)),
        "terminal_gap_ratio": min(4.0, float(terminal_gap / expected_denominator)),
        "rest_ratio": rests / max(len(regular), 1),
        "chord_ratio": chords / max(len(regular), 1),
        "duration_vocabulary_ratio": conventional / max(len(regular), 1),
        "onset_collision_count": float(collisions),
        "notation_density": notation_count / max(len(regular), 1),
        "accidental_alter_mismatches": float(accidental_alter_mismatches),
        "extreme_alters": float(extreme_alters),
        "orphan_chords": float(orphan_chords),
        "duplicate_events": float(max(0, duplicate_events)),
        "rest_pitch_conflicts": float(rest_pitch_conflicts),
        "unknown_note_type_ratio": unknown_note_types / max(len(regular), 1),
    }


def feature_vector(item: MeasureCalibrationInput) -> list[float]:
    stats = _measure_statistics(item.measure)
    return [
        max(-2.0, min(2.0, float(item.candidate.score) / 1000.0)),
        max(0.0, min(1.0, float(getattr(item.candidate, "calibrated_probability", 0.5)))),
        float(bool(item.candidate.valid)),
        max(0.0, min(1.0, float(item.alignment_similarity))),
        max(0.0, min(1.0, float(item.exact_support_ratio))),
        max(0.0, min(1.0, float(item.semantic_support_ratio))),
        max(0.0, min(1.0, float(item.missing_ratio))),
        max(0.0, min(1.0, float(item.distance_to_template))),
        max(0.0, min(1.0, float(item.distance_to_medoid))),
        max(0.0, min(1.0, float(item.mean_peer_distance))),
        min(4.0, stats["note_count"] / 8.0),
        min(4.0, stats["direction_count"] / 4.0),
        min(4.0, stats["voice_count"]),
        stats["duration_error"],
        min(4.0, stats["zero_durations"]),
        min(6.0, stats["type_mismatches"]),
        min(4.0, stats["chord_mismatches"]),
        min(4.0, stats["duplicate_directions"]),
        min(4.0, stats["pitch_outliers"]),
        stats["attribute_completeness"],
        stats["internal_gap_ratio"],
        stats["overlap_ratio"],
        stats["terminal_gap_ratio"],
        stats["rest_ratio"],
        stats["chord_ratio"],
        stats["duration_vocabulary_ratio"],
        min(4.0, stats["onset_collision_count"]),
        min(4.0, stats["notation_density"] / 2.0),
        float(item.is_first_measure),
        float(item.is_last_measure),
        float(item.measure.expected_duration is not None),
        min(4.0, stats["accidental_alter_mismatches"]),
        min(4.0, stats["extreme_alters"]),
        min(4.0, stats["orphan_chords"]),
        min(6.0, stats["duplicate_events"]),
        min(4.0, stats["rest_pitch_conflicts"]),
        max(0.0, min(1.0, stats["unknown_note_type_ratio"])),
    ]


class MeasureCalibrator:
    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).resolve().parent / "resources" / "measure_calibrator.json"
        loaded = load_verified_json(model_path, "measure_candidate_calibration")
        payload = loaded.payload
        self.forest = VerifiedRandomForestModel.from_payload(
            payload,
            FEATURE_NAMES,
            verified=loaded.verified,
            status=loaded.status,
        )
        legacy_payload = payload.get("legacy_model", payload)
        self.legacy = StandardizedLogisticModel.from_payload(
            legacy_payload,
            LEGACY_FEATURE_NAMES,
            verified=loaded.verified,
            status=loaded.status,
        )
        try:
            self.legacy_preservation_floor = float(
                payload.get("legacy_preservation_floor", 0.55)
            )
        except (TypeError, ValueError, OverflowError):
            self.legacy_preservation_floor = 0.55
        self.hybrid_enabled = (
            self.forest.enabled
            and self.legacy.enabled
            and 0.0 <= self.legacy_preservation_floor <= 1.0
        )
        self.model_verified = self.forest.verified or self.legacy.verified
        self.model_status = self.forest.status if self.forest.enabled else self.legacy.status
        self.model_version = self.forest.model_version if self.forest.enabled else self.legacy.model_version
        self.enabled = self.forest.enabled or self.legacy.enabled

    def predict_probability(self, item: MeasureCalibrationInput) -> float:
        values = feature_vector(item)
        if self.forest.enabled:
            forest_probability = self.forest.predict(values)
            if self.hybrid_enabled:
                legacy_probability = self.legacy.predict(values[: len(LEGACY_FEATURE_NAMES)])
                preserved_legacy = legacy_probability * (
                    self.legacy_preservation_floor
                    + (1.0 - self.legacy_preservation_floor) * forest_probability
                )
                return max(forest_probability, preserved_legacy)
            return forest_probability
        if self.legacy.enabled:
            return self.legacy.predict(values[: len(LEGACY_FEATURE_NAMES)])
        return 0.5

    def calibrate(self, item: MeasureCalibrationInput) -> MeasureCalibrationResult:
        probability = self.predict_probability(item)
        # Linear, bounded influence around 0.5.  The learned prior can break close ties,
        # but cannot overpower a strict majority or a hard structural validation failure.
        weight = bounded_weight(
            probability,
            DEFAULT_POLICY.measure_calibration_weight_floor,
            DEFAULT_POLICY.measure_calibration_weight_ceiling,
        )
        return MeasureCalibrationResult(round(probability, 6), round(weight, 6), self.model_version)
