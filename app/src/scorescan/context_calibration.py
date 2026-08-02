from __future__ import annotations

"""Bounded cross-measure context calibration for ensemble selection.

The model in this module does not recognise notation and must never override hard
MusicXML, rhythm, or voice validation.  It estimates whether one already-parsed
candidate measure fits its immediate neighbours in a single-staff, single-voice score.
Only conservative properties are used: boundary pitch continuity, rhythmic texture,
attribute continuity, key-fit, and tie continuity.  The resulting weight is deliberately
narrow so unusual but valid music is not normalised away.
"""

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from .policy import DEFAULT_POLICY
from .score_ir import MeasureIR, NoteIR
from .tree_model import VerifiedRandomForestModel

LEGACY_FEATURE_NAMES = (
    "previous_present",
    "next_present",
    "previous_boundary_interval",
    "next_boundary_interval",
    "previous_density_change",
    "next_density_change",
    "previous_duration_texture",
    "next_duration_texture",
    "previous_pitch_center_change",
    "next_pitch_center_change",
    "pitch_span_scaled",
    "key_diatonic_ratio",
    "previous_time_continuity",
    "next_time_continuity",
    "previous_key_continuity",
    "next_key_continuity",
    "previous_clef_continuity",
    "next_clef_continuity",
    "orphan_tie_stop_ratio",
    "orphan_tie_start_ratio",
    "matched_tie_stop_ratio",
    "matched_tie_start_ratio",
    "rest_ratio",
    "direction_change_scaled",
)


# Current-only properties are already represented by the candidate's local profile.
# Independent-family evidence is reserved for features which genuinely compare a
# measure with its neighbours, avoiding duplicate columns with identical values.
_INDEPENDENT_SOURCE_NAMES = tuple(
    name
    for name in LEGACY_FEATURE_NAMES
    if name not in {"pitch_span_scaled", "key_diatonic_ratio", "rest_ratio"}
)
INDEPENDENT_FEATURE_NAMES = tuple(
    f"independent_median_{name}" for name in _INDEPENDENT_SOURCE_NAMES
)
FEATURE_NAMES = LEGACY_FEATURE_NAMES + INDEPENDENT_FEATURE_NAMES + (
    "independent_family_count_scaled",
)
_INDEPENDENT_SOURCE_INDICES = tuple(
    LEGACY_FEATURE_NAMES.index(name) for name in _INDEPENDENT_SOURCE_NAMES
)


@dataclass(frozen=True)
class ContextProfile:
    previous_present: float
    next_present: float
    previous_boundary_interval: float
    next_boundary_interval: float
    previous_density_change: float
    next_density_change: float
    previous_duration_texture: float
    next_duration_texture: float
    previous_pitch_center_change: float
    next_pitch_center_change: float
    pitch_span_scaled: float
    key_diatonic_ratio: float
    previous_time_continuity: float
    next_time_continuity: float
    previous_key_continuity: float
    next_key_continuity: float
    previous_clef_continuity: float
    next_clef_continuity: float
    orphan_tie_stop_ratio: float
    orphan_tie_start_ratio: float
    matched_tie_stop_ratio: float
    matched_tie_start_ratio: float
    rest_ratio: float
    direction_change_scaled: float

    def feature_vector(self) -> list[float]:
        return [float(getattr(self, name)) for name in LEGACY_FEATURE_NAMES]


@dataclass(frozen=True)
class ContextCalibrationResult:
    probability: float
    weight_factor: float
    model_version: str


def _regular(measure: MeasureIR | None) -> tuple[NoteIR, ...]:
    if measure is None:
        return ()
    return tuple(note for note in measure.notes if not note.grace and not note.chord)


def _pitched(measure: MeasureIR | None) -> tuple[NoteIR, ...]:
    return tuple(
        note
        for note in _regular(measure)
        if not note.rest and note.pitch is not None
    )


def _pitch_values(measure: MeasureIR | None) -> list[int]:
    return [note.pitch.midi_cents for note in _pitched(measure) if note.pitch is not None]


def _first_pitch(measure: MeasureIR | None) -> int | None:
    notes = sorted(_pitched(measure), key=lambda note: (note.onset, note.duration))
    return notes[0].pitch.midi_cents if notes and notes[0].pitch else None


def _last_pitch(measure: MeasureIR | None) -> int | None:
    notes = sorted(_pitched(measure), key=lambda note: (note.onset + note.duration, note.onset))
    return notes[-1].pitch.midi_cents if notes and notes[-1].pitch else None


def _interval(left: int | None, right: int | None) -> float:
    if left is None or right is None:
        return 0.5
    # 24 semitones already represents a large but plausible single-line leap.
    return min(1.0, abs(right - left) / 2400.0)


def _density_change(left: MeasureIR | None, right: MeasureIR | None) -> float:
    if left is None or right is None:
        return 0.0
    left_count = len(_regular(left))
    right_count = len(_regular(right))
    return min(
        1.0,
        abs(math.log((right_count + 1.0) / (left_count + 1.0)))
        / math.log(8.0),
    )


def _duration_histogram(measure: MeasureIR | None) -> dict[Fraction, float]:
    notes = _regular(measure)
    if not notes:
        return {}
    counts: dict[Fraction, float] = {}
    for note in notes:
        counts[note.duration] = counts.get(note.duration, 0.0) + 1.0
    total = sum(counts.values())
    return {key: value / total for key, value in counts.items()}


def _duration_texture(left: MeasureIR | None, right: MeasureIR | None) -> float:
    if left is None or right is None:
        return 0.5
    a = _duration_histogram(left)
    b = _duration_histogram(right)
    if not a and not b:
        return 1.0
    keys = set(a) | set(b)
    distance = 0.5 * sum(abs(a.get(key, 0.0) - b.get(key, 0.0)) for key in keys)
    return max(0.0, 1.0 - min(1.0, distance))


def _pitch_center_change(left: MeasureIR | None, right: MeasureIR | None) -> float:
    left_values = _pitch_values(left)
    right_values = _pitch_values(right)
    if not left_values or not right_values:
        return 0.5
    difference = abs(statistics.median(right_values) - statistics.median(left_values))
    return min(1.0, difference / 2400.0)


def _pitch_span(measure: MeasureIR) -> float:
    values = _pitch_values(measure)
    if len(values) < 2:
        return 0.0
    return min(1.0, (max(values) - min(values)) / 3600.0)


def _tonic_pitch_class(fifths: int, mode: str) -> int:
    # Circle-of-fifths tonic mapping.  Relative minor is three semitones below major.
    major = (7 * fifths) % 12
    return (major - 3) % 12 if mode.casefold().startswith("minor") else major


def _key_diatonic_ratio(measure: MeasureIR) -> float:
    values = _pitch_values(measure)
    if not values:
        return 1.0
    fifths, mode = measure.key_signature or (0, "major")
    tonic = _tonic_pitch_class(int(fifths), str(mode))
    intervals = (
        {0, 2, 4, 5, 7, 9, 11}
        if not str(mode).casefold().startswith("minor")
        else {0, 2, 3, 5, 7, 8, 10}
    )
    scale = {(tonic + interval) % 12 for interval in intervals}
    matches = sum(((value // 100) % 12) in scale for value in values)
    return matches / max(len(values), 1)


def _tie_pitch_set(measure: MeasureIR | None, tie_type: str) -> set[int]:
    result: set[int] = set()
    if measure is None:
        return result
    for note in measure.notes:
        if note.pitch is not None and tie_type in note.ties:
            result.add(note.pitch.midi_cents)
    return result


def _ratio(numerator: int, denominator: int, empty_value: float) -> float:
    return numerator / denominator if denominator else empty_value


def context_profile(
    previous: MeasureIR | None,
    current: MeasureIR,
    following: MeasureIR | None,
) -> ContextProfile:
    previous_starts = _tie_pitch_set(previous, "start")
    current_stops = _tie_pitch_set(current, "stop")
    current_starts = _tie_pitch_set(current, "start")
    following_stops = _tie_pitch_set(following, "stop")

    matched_stops = len(current_stops & previous_starts)
    matched_starts = len(current_starts & following_stops)
    orphan_stops = len(current_stops - previous_starts)
    orphan_starts = len(current_starts - following_stops)
    regular = _regular(current)
    rest_ratio = sum(note.rest for note in regular) / max(len(regular), 1)
    neighbor_directions = (
        (len(previous.directions) if previous else 0)
        + (len(following.directions) if following else 0)
    )
    direction_change = min(1.0, abs(2 * len(current.directions) - neighbor_directions) / 6.0)

    return ContextProfile(
        previous_present=float(previous is not None),
        next_present=float(following is not None),
        previous_boundary_interval=_interval(_last_pitch(previous), _first_pitch(current)),
        next_boundary_interval=_interval(_last_pitch(current), _first_pitch(following)),
        previous_density_change=_density_change(previous, current),
        next_density_change=_density_change(current, following),
        previous_duration_texture=_duration_texture(previous, current),
        next_duration_texture=_duration_texture(current, following),
        previous_pitch_center_change=_pitch_center_change(previous, current),
        next_pitch_center_change=_pitch_center_change(current, following),
        pitch_span_scaled=_pitch_span(current),
        key_diatonic_ratio=_key_diatonic_ratio(current),
        previous_time_continuity=float(previous is None or previous.time_signature == current.time_signature),
        next_time_continuity=float(following is None or following.time_signature == current.time_signature),
        previous_key_continuity=float(previous is None or previous.key_signature == current.key_signature),
        next_key_continuity=float(following is None or following.key_signature == current.key_signature),
        previous_clef_continuity=float(previous is None or previous.clef == current.clef),
        next_clef_continuity=float(following is None or following.clef == current.clef),
        orphan_tie_stop_ratio=_ratio(orphan_stops, len(current_stops), 0.0),
        orphan_tie_start_ratio=_ratio(orphan_starts, len(current_starts), 0.0),
        matched_tie_stop_ratio=_ratio(matched_stops, len(current_stops), 1.0),
        matched_tie_start_ratio=_ratio(matched_starts, len(current_starts), 1.0),
        rest_ratio=rest_ratio,
        direction_change_scaled=direction_change,
    )



@dataclass(frozen=True)
class ContextAgreementProfile:
    """Local candidate context plus family-balanced independent neighbours.

    Related preprocessing variants are averaged within one family before a median is
    taken across independent families.  Consequently ``flat`` and ``deblock`` cannot
    contribute two votes for the same restoration artefact.  When no independent
    family is available the local values are repeated but the explicit family-count
    feature remains zero, allowing the model to fall back conservatively.
    """

    local: ContextProfile
    independent_median: tuple[float, ...]
    independent_family_count_scaled: float

    def feature_vector(self) -> list[float]:
        return (
            self.local.feature_vector()
            + [float(value) for value in self.independent_median]
            + [float(self.independent_family_count_scaled)]
        )

    def legacy_feature_vector(self) -> list[float]:
        return self.local.feature_vector()


def _mean_vectors(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not vectors:
        return ()
    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        raise ValueError("context vectors must have a consistent width")
    return tuple(
        sum(float(vector[index]) for vector in vectors) / len(vectors)
        for index in range(width)
    )


def agreement_profiles(
    previous: Sequence[MeasureIR | None],
    current: Sequence[MeasureIR],
    following: Sequence[MeasureIR | None],
    families: Sequence[str],
) -> tuple[ContextAgreementProfile, ...]:
    """Build family-balanced cross-measure profiles for one aligned candidate set.

    The current measure is held fixed while its boundary compatibility is evaluated
    against the previous and following measures supplied by each *other* preprocessing
    family.  This exposes a family-wide coherent error (for example, the same octave
    shift across three binary candidates) without allowing sibling variants to
    manufacture independent support.
    """

    size = len(current)
    if len(previous) != size or len(following) != size or len(families) != size:
        raise ValueError("context candidates, neighbours, and families must align")
    if not size:
        return ()

    normalized_families = tuple(str(value or "unknown") for value in families)
    distinct_families = tuple(sorted(set(normalized_families)))
    results: list[ContextAgreementProfile] = []
    for candidate_index, candidate in enumerate(current):
        local = context_profile(
            previous[candidate_index],
            candidate,
            following[candidate_index],
        )
        local_vector = local.feature_vector()
        independent_family_vectors: list[tuple[float, ...]] = []
        candidate_family = normalized_families[candidate_index]
        for family in distinct_families:
            if family == candidate_family:
                continue
            family_profiles = [
                context_profile(
                    previous[index],
                    candidate,
                    following[index],
                ).feature_vector()
                for index, other_family in enumerate(normalized_families)
                if other_family == family
            ]
            if family_profiles:
                independent_family_vectors.append(_mean_vectors(family_profiles))

        if independent_family_vectors:
            independent_median = tuple(
                float(statistics.median(vector[index] for vector in independent_family_vectors))
                for index in _INDEPENDENT_SOURCE_INDICES
            )
        else:
            independent_median = tuple(
                float(local_vector[index])
                for index in _INDEPENDENT_SOURCE_INDICES
            )
        results.append(
            ContextAgreementProfile(
                local=local,
                independent_median=independent_median,
                independent_family_count_scaled=min(
                    1.0,
                    len(independent_family_vectors) / 3.0,
                ),
            )
        )
    return tuple(results)


class ContextCalibrator:
    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = (
                Path(__file__).resolve().parent
                / "resources"
                / "context_calibrator.json"
            )
        self.model = VerifiedRandomForestModel.load(
            model_path,
            "context_candidate_calibration",
            FEATURE_NAMES,
        )
        self.model_verified = self.model.verified
        self.model_status = self.model.status
        self.model_version = self.model.model_version
        self.enabled = self.model.enabled

    def predict_probability(self, profile: ContextAgreementProfile) -> float:
        return self.model.predict(profile.feature_vector(), neutral=0.5)

    def calibrate_profile(
        self,
        profile: ContextAgreementProfile,
    ) -> ContextCalibrationResult:
        probability = self.predict_probability(profile)
        floor = DEFAULT_POLICY.context_calibration_weight_floor
        ceiling = DEFAULT_POLICY.context_calibration_weight_ceiling
        weight = floor + (ceiling - floor) * probability
        return ContextCalibrationResult(
            round(probability, 6),
            round(weight, 6),
            self.model_version,
        )

    def calibrate(
        self,
        previous: MeasureIR | None,
        current: MeasureIR,
        following: MeasureIR | None,
    ) -> ContextCalibrationResult:
        """Backward-compatible single-candidate calibration.

        Production consensus uses :func:`agreement_profiles` with all aligned
        preprocessing families.  The single-candidate form remains useful for focused
        diagnostics and degrades conservatively through a zero family-count feature.
        """

        profile = agreement_profiles(
            (previous,),
            (current,),
            (following,),
            ("baseline",),
        )[0]
        return self.calibrate_profile(profile)
