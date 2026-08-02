from __future__ import annotations

"""Deterministic semantic alignment for independently recognised measure sequences.

OMR variants frequently insert or omit one measure.  Index-wise comparison then shifts
all later measures and destroys useful ensemble evidence.  This module performs a
bounded global alignment over :class:`MeasureIR` sequences.  It is intentionally small,
pure and dependency-free so it can be unit-tested independently from recognition and
XML export.
"""

from dataclasses import dataclass
from typing import Sequence

from .score_ir import MeasureIR, measure_distance


@dataclass(frozen=True)
class AlignmentPair:
    reference_index: int | None
    candidate_index: int | None
    cost: float


@dataclass(frozen=True)
class SequenceAlignment:
    pairs: tuple[AlignmentPair, ...]
    reference_to_candidate: tuple[int | None, ...]
    unmatched_candidate_indices: tuple[int, ...]
    total_cost: float
    normalized_cost: float
    similarity: float


def _substitution_cost(
    reference: MeasureIR,
    candidate: MeasureIR,
    reference_index: int,
    candidate_index: int,
    reference_count: int,
    candidate_count: int,
) -> float:
    semantic = measure_distance(reference, candidate)
    # A weak monotonic position prior disambiguates repeated empty or nearly identical
    # measures without overriding actual semantic evidence.
    ref_position = (reference_index + 0.5) / max(reference_count, 1)
    cand_position = (candidate_index + 0.5) / max(candidate_count, 1)
    position_penalty = min(0.08, abs(ref_position - cand_position) * 0.12)
    return min(1.0, semantic + position_penalty)


def align_measure_sequences(
    reference: Sequence[MeasureIR],
    candidate: Sequence[MeasureIR],
    *,
    gap_penalty: float = 0.62,
) -> SequenceAlignment:
    """Globally align two measure sequences using deterministic dynamic programming.

    The returned mapping always has one entry per reference measure.  ``None`` denotes
    a missing candidate measure.  Candidate-only insertions are reported separately.
    The similarity is normalised to ``0..1`` and includes insertion/deletion penalties.
    """

    reference = tuple(reference)
    candidate = tuple(candidate)
    m, n = len(reference), len(candidate)
    if m == 0 and n == 0:
        return SequenceAlignment((), (), (), 0.0, 0.0, 1.0)

    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    back: list[list[str | None]] = [[None] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i * gap_penalty
        back[i][0] = "delete"
    for j in range(1, n + 1):
        dp[0][j] = j * gap_penalty
        back[0][j] = "insert"

    # Stable tie order is important for byte-for-byte repeatability: match, delete,
    # insert.  The tuple includes an operation rank for deterministic ``min``.
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            substitution = _substitution_cost(reference[i - 1], candidate[j - 1], i - 1, j - 1, m, n)
            choices = (
                (dp[i - 1][j - 1] + substitution, 0, "match"),
                (dp[i - 1][j] + gap_penalty, 1, "delete"),
                (dp[i][j - 1] + gap_penalty, 2, "insert"),
            )
            value, _rank, operation = min(choices)
            dp[i][j] = value
            back[i][j] = operation

    pairs_reversed: list[AlignmentPair] = []
    mapping: list[int | None] = [None] * m
    unmatched: list[int] = []
    i, j = m, n
    while i > 0 or j > 0:
        operation = back[i][j]
        if operation == "match":
            cost = _substitution_cost(reference[i - 1], candidate[j - 1], i - 1, j - 1, m, n)
            mapping[i - 1] = j - 1
            pairs_reversed.append(AlignmentPair(i - 1, j - 1, cost))
            i -= 1
            j -= 1
        elif operation == "delete":
            pairs_reversed.append(AlignmentPair(i - 1, None, gap_penalty))
            i -= 1
        elif operation == "insert":
            unmatched.append(j - 1)
            pairs_reversed.append(AlignmentPair(None, j - 1, gap_penalty))
            j -= 1
        else:  # defensive fallback for the origin or malformed state
            if i > 0:
                pairs_reversed.append(AlignmentPair(i - 1, None, gap_penalty))
                i -= 1
            elif j > 0:
                unmatched.append(j - 1)
                pairs_reversed.append(AlignmentPair(None, j - 1, gap_penalty))
                j -= 1

    pairs = tuple(reversed(pairs_reversed))
    unmatched_indices = tuple(sorted(unmatched))
    scale = max(max(m, n), 1) * max(gap_penalty, 1e-9)
    normalized = min(1.0, dp[m][n] / scale)
    return SequenceAlignment(
        pairs=pairs,
        reference_to_candidate=tuple(mapping),
        unmatched_candidate_indices=unmatched_indices,
        total_cost=round(dp[m][n], 9),
        normalized_cost=round(normalized, 9),
        similarity=round(max(0.0, 1.0 - normalized), 9),
    )
