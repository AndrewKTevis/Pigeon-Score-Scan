from __future__ import annotations

"""Bounded, family-aware event evidence for measure-level candidate selection.

Page and measure calibrators operate on aggregate structure. This module compares the
single-voice event lattice of each candidate with its peers. Related preprocessing
variants are balanced by family so two near-identical binary or restoration passes
cannot manufacture independent support. The model never edits a note, creates a
majority, or bypasses deterministic MusicXML and rhythm validation.
"""

import statistics
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from collections.abc import Sequence

from .policy import DEFAULT_POLICY
from .score_ir import MeasureIR, NoteIR, note_substitution_cost
from .tree_model import VerifiedRandomForestModel

LEGACY_FEATURE_NAMES = (
    "mean_alignment_similarity",
    "median_alignment_similarity",
    "pitch_support",
    "duration_support",
    "onset_support",
    "rest_support",
    "notation_support",
    "unmatched_ratio",
    "note_count_consistency",
    "anchor_count_consistency",
    "duration_fill_error",
    "voice_count_scaled",
    "peer_count_scaled",
)

FAMILY_FEATURE_NAMES = (
    "independent_mean_alignment_similarity",
    "independent_median_alignment_similarity",
    "independent_pitch_support",
    "independent_duration_support",
    "independent_onset_support",
    "independent_rest_support",
    "independent_notation_support",
    "independent_unmatched_ratio",
    "independent_strong_family_fraction",
    "independent_exact_family_fraction",
    "same_family_alignment_similarity",
    "independent_family_count_scaled",
)

FEATURE_NAMES = LEGACY_FEATURE_NAMES + FAMILY_FEATURE_NAMES


@dataclass(frozen=True)
class NoteAlignment:
    pairs: tuple[tuple[int | None, int | None, float], ...]
    similarity: float
    unmatched_left: int
    unmatched_right: int


@dataclass(frozen=True)
class _PairEvidence:
    similarity: float
    pitch_votes: float
    duration_votes: float
    onset_votes: float
    rest_votes: float
    notation_votes: float
    comparable_weight: float
    unmatched: int
    total_slots: int


@dataclass(frozen=True)
class _EvidenceSummary:
    mean_similarity: float
    median_similarity: float
    pitch_support: float
    duration_support: float
    onset_support: float
    rest_support: float
    notation_support: float
    unmatched_ratio: float


@dataclass(frozen=True)
class EventAgreementProfile:
    mean_alignment_similarity: float
    median_alignment_similarity: float
    pitch_support: float
    duration_support: float
    onset_support: float
    rest_support: float
    notation_support: float
    unmatched_ratio: float
    note_count_consistency: float
    anchor_count_consistency: float
    duration_fill_error: float
    voice_count_scaled: float
    peer_count_scaled: float
    independent_mean_alignment_similarity: float
    independent_median_alignment_similarity: float
    independent_pitch_support: float
    independent_duration_support: float
    independent_onset_support: float
    independent_rest_support: float
    independent_notation_support: float
    independent_unmatched_ratio: float
    independent_strong_family_fraction: float
    independent_exact_family_fraction: float
    same_family_alignment_similarity: float
    independent_family_count_scaled: float

    def feature_vector(self) -> list[float]:
        return [float(getattr(self, name)) for name in FEATURE_NAMES]

    def legacy_feature_vector(self) -> list[float]:
        return [float(getattr(self, name)) for name in LEGACY_FEATURE_NAMES]


@dataclass(frozen=True)
class EventCalibrationResult:
    probability: float
    weight_factor: float
    model_version: str


def _regular_notes(measure: MeasureIR) -> tuple[NoteIR, ...]:
    return tuple(note for note in measure.notes if not note.grace)


def align_note_sequences(left: tuple[NoteIR, ...], right: tuple[NoteIR, ...]) -> NoteAlignment:
    """Globally align note/rest events with a bounded semantic substitution cost."""
    rows, cols = len(left) + 1, len(right) + 1
    gap = 0.92
    costs = [[0.0] * cols for _ in range(rows)]
    trace = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        costs[i][0] = i * gap
        trace[i][0] = "up"
    for j in range(1, cols):
        costs[0][j] = j * gap
        trace[0][j] = "left"
    for i in range(1, rows):
        for j in range(1, cols):
            substitution = costs[i - 1][j - 1] + note_substitution_cost(left[i - 1], right[j - 1])
            deletion = costs[i - 1][j] + gap
            insertion = costs[i][j - 1] + gap
            best = min(
                (substitution, "diag"),
                (deletion, "up"),
                (insertion, "left"),
                key=lambda item: (item[0], item[1]),
            )
            costs[i][j], trace[i][j] = best

    i, j = len(left), len(right)
    pairs: list[tuple[int | None, int | None, float]] = []
    unmatched_left = unmatched_right = 0
    while i or j:
        direction = trace[i][j]
        if i and j and direction == "diag":
            pairs.append((i - 1, j - 1, note_substitution_cost(left[i - 1], right[j - 1])))
            i -= 1
            j -= 1
        elif i and (not j or direction == "up"):
            pairs.append((i - 1, None, 1.0))
            unmatched_left += 1
            i -= 1
        else:
            pairs.append((None, j - 1, 1.0))
            unmatched_right += 1
            j -= 1
    pairs.reverse()
    normalizer = max(len(left), len(right), 1)
    similarity = max(0.0, min(1.0, 1.0 - costs[-1][-1] / normalizer))
    return NoteAlignment(tuple(pairs), similarity, unmatched_left, unmatched_right)


def _same_pitch(left: NoteIR, right: NoteIR) -> bool:
    if left.rest or right.rest:
        return left.rest == right.rest
    if left.pitch is None or right.pitch is None:
        return left.pitch is right.pitch
    return left.pitch.midi_cents == right.pitch.midi_cents


def _same_notation(left: NoteIR, right: NoteIR) -> bool:
    return (
        left.accidental.casefold() == right.accidental.casefold()
        and left.ties == right.ties
        and left.slurs == right.slurs
        and left.articulations == right.articulations
        and left.ornaments == right.ornaments
        and left.tuple_ratio == right.tuple_ratio
        and left.grace == right.grace
        and left.chord == right.chord
    )


def _pair_evidence(left_measure: MeasureIR, right_measure: MeasureIR) -> _PairEvidence:
    left = _regular_notes(left_measure)
    right = _regular_notes(right_measure)
    alignment = align_note_sequences(left, right)
    pitch_votes = duration_votes = onset_votes = rest_votes = notation_votes = 0.0
    comparable = 0.0
    for left_index, right_index, cost in alignment.pairs:
        if left_index is None or right_index is None or cost >= 0.90:
            continue
        left_note, right_note = left[left_index], right[right_index]
        weight = max(0.20, 1.0 - cost)
        comparable += weight
        pitch_votes += weight * float(_same_pitch(left_note, right_note))
        duration_votes += weight * float(left_note.duration == right_note.duration)
        onset_votes += weight * float(left_note.onset == right_note.onset)
        rest_votes += weight * float(left_note.rest == right_note.rest)
        notation_votes += weight * float(_same_notation(left_note, right_note))
    return _PairEvidence(
        similarity=alignment.similarity,
        pitch_votes=pitch_votes,
        duration_votes=duration_votes,
        onset_votes=onset_votes,
        rest_votes=rest_votes,
        notation_votes=notation_votes,
        comparable_weight=comparable,
        unmatched=alignment.unmatched_left + alignment.unmatched_right,
        total_slots=max(len(left), len(right), 1),
    )


def _summarize(evidence: Sequence[_PairEvidence]) -> _EvidenceSummary:
    if not evidence:
        return _EvidenceSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    similarities = [item.similarity for item in evidence]
    comparable = sum(item.comparable_weight for item in evidence)
    denominator = max(comparable, 1e-9)
    return _EvidenceSummary(
        mean_similarity=sum(similarities) / len(similarities),
        median_similarity=statistics.median(similarities),
        pitch_support=sum(item.pitch_votes for item in evidence) / denominator,
        duration_support=sum(item.duration_votes for item in evidence) / denominator,
        onset_support=sum(item.onset_votes for item in evidence) / denominator,
        rest_support=sum(item.rest_votes for item in evidence) / denominator,
        notation_support=sum(item.notation_votes for item in evidence) / denominator,
        unmatched_ratio=min(
            1.0,
            sum(item.unmatched for item in evidence)
            / max(sum(item.total_slots for item in evidence), 1),
        ),
    )


def _mean_family_summary(summaries: Sequence[_EvidenceSummary]) -> _EvidenceSummary:
    if not summaries:
        return _EvidenceSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    return _EvidenceSummary(
        mean_similarity=sum(item.mean_similarity for item in summaries) / len(summaries),
        median_similarity=statistics.median(item.mean_similarity for item in summaries),
        pitch_support=sum(item.pitch_support for item in summaries) / len(summaries),
        duration_support=sum(item.duration_support for item in summaries) / len(summaries),
        onset_support=sum(item.onset_support for item in summaries) / len(summaries),
        rest_support=sum(item.rest_support for item in summaries) / len(summaries),
        notation_support=sum(item.notation_support for item in summaries) / len(summaries),
        unmatched_ratio=sum(item.unmatched_ratio for item in summaries) / len(summaries),
    )


def _measure_end(measure: MeasureIR) -> Fraction:
    notes = _regular_notes(measure)
    if not notes:
        return Fraction(0, 1)
    return max(
        (note.onset + max(note.duration, Fraction(0, 1)) for note in notes),
        default=Fraction(0, 1),
    )


def agreement_profiles(
    measures: list[MeasureIR],
    families: Sequence[str] | None = None,
) -> list[EventAgreementProfile]:
    """Return one cached, family-balanced peer-agreement profile per candidate.

    When family labels are unavailable each candidate is treated as independent. The
    production consensus path always supplies preprocessing-family labels.
    """
    if not measures:
        return []
    if families is None:
        normalized_families = tuple(f"candidate:{index}" for index in range(len(measures)))
    else:
        if len(families) != len(measures):
            raise ValueError("family count must match measure count")
        normalized_families = tuple(str(value or "unknown") for value in families)

    pair_cache: dict[tuple[int, int], _PairEvidence] = {}
    for left_index in range(len(measures)):
        for right_index in range(left_index + 1, len(measures)):
            pair_cache[(left_index, right_index)] = _pair_evidence(
                measures[left_index], measures[right_index]
            )

    note_counts = [len(_regular_notes(measure)) for measure in measures]
    anchor_counts = [sum(not note.chord for note in _regular_notes(measure)) for measure in measures]
    median_notes = statistics.median(note_counts)
    median_anchors = statistics.median(anchor_counts)
    profiles: list[EventAgreementProfile] = []

    for index, measure in enumerate(measures):
        all_evidence: list[_PairEvidence] = []
        same_family: list[_PairEvidence] = []
        independent_by_family: dict[str, list[_PairEvidence]] = {}
        for peer_index in range(len(measures)):
            if peer_index == index:
                continue
            key = (min(index, peer_index), max(index, peer_index))
            evidence = pair_cache[key]
            all_evidence.append(evidence)
            peer_family = normalized_families[peer_index]
            if peer_family == normalized_families[index]:
                same_family.append(evidence)
            else:
                independent_by_family.setdefault(peer_family, []).append(evidence)

        aggregate = _summarize(all_evidence)
        family_summaries = [
            _summarize(independent_by_family[name])
            for name in sorted(independent_by_family)
        ]
        independent = _mean_family_summary(family_summaries)
        family_similarities = [item.mean_similarity for item in family_summaries]
        same_family_similarity = (
            sum(item.similarity for item in same_family) / len(same_family)
            if same_family
            else 0.5
        )

        notes = _regular_notes(measure)
        expected = measure.expected_duration
        if expected and expected > 0:
            duration_fill_error = min(
                4.0,
                float(abs(_measure_end(measure) - expected) / expected),
            )
        else:
            duration_fill_error = 0.0
        note_consistency = 1.0 - min(
            1.0,
            abs(len(notes) - median_notes) / max(median_notes, 1.0),
        )
        anchors = sum(not note.chord for note in notes)
        anchor_consistency = 1.0 - min(
            1.0,
            abs(anchors - median_anchors) / max(median_anchors, 1.0),
        )
        independent_count = len(family_summaries)
        profiles.append(
            EventAgreementProfile(
                mean_alignment_similarity=aggregate.mean_similarity,
                median_alignment_similarity=aggregate.median_similarity,
                pitch_support=aggregate.pitch_support,
                duration_support=aggregate.duration_support,
                onset_support=aggregate.onset_support,
                rest_support=aggregate.rest_support,
                notation_support=aggregate.notation_support,
                unmatched_ratio=aggregate.unmatched_ratio,
                note_count_consistency=max(0.0, note_consistency),
                anchor_count_consistency=max(0.0, anchor_consistency),
                duration_fill_error=duration_fill_error,
                voice_count_scaled=min(4.0, float(measure.voice_count)),
                peer_count_scaled=min(1.0, len(all_evidence) / 5.0),
                independent_mean_alignment_similarity=independent.mean_similarity,
                independent_median_alignment_similarity=independent.median_similarity,
                independent_pitch_support=independent.pitch_support,
                independent_duration_support=independent.duration_support,
                independent_onset_support=independent.onset_support,
                independent_rest_support=independent.rest_support,
                independent_notation_support=independent.notation_support,
                independent_unmatched_ratio=independent.unmatched_ratio,
                independent_strong_family_fraction=(
                    sum(value >= 0.90 for value in family_similarities)
                    / max(independent_count, 1)
                ),
                independent_exact_family_fraction=(
                    sum(value >= 0.98 for value in family_similarities)
                    / max(independent_count, 1)
                ),
                same_family_alignment_similarity=max(0.0, min(1.0, same_family_similarity)),
                independent_family_count_scaled=min(1.0, independent_count / 3.0),
            )
        )
    return profiles


class EventCalibrator:
    def __init__(self, model_path: Path | None = None) -> None:
        path = model_path or Path(__file__).resolve().parent / "resources" / "event_calibrator.json"
        self.model = VerifiedRandomForestModel.load(
            path,
            "event_candidate_calibration",
            FEATURE_NAMES,
        )

    @property
    def enabled(self) -> bool:
        return self.model.enabled

    @property
    def model_verified(self) -> bool:
        return self.model.verified

    @property
    def model_status(self) -> str:
        return self.model.status

    @property
    def model_version(self) -> str:
        return self.model.model_version

    def predict_probability(self, profile: EventAgreementProfile) -> float:
        return self.model.predict(profile.feature_vector(), neutral=0.5)

    def calibrate(self, profile: EventAgreementProfile) -> EventCalibrationResult:
        probability = self.predict_probability(profile)
        floor = DEFAULT_POLICY.event_calibration_weight_floor
        ceiling = DEFAULT_POLICY.event_calibration_weight_ceiling
        weight = floor + (ceiling - floor) * probability
        return EventCalibrationResult(
            round(probability, 6),
            round(weight, 6),
            self.model_version,
        )
