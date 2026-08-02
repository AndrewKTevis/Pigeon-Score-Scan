from __future__ import annotations

"""Rendered, group-isolated hard examples for pitch-patch calibration.

The rows in this module are not an end-to-end OMR benchmark.  Each source image is
rendered from a known :class:`MeasureIR`, then one controlled pitch error is introduced.
The correction and the inverse regression share one group so no source crop can cross a
training/evaluation boundary.  High-level candidate evidence is intentionally plausible
for both rows; only the direct source-crop pitch evidence distinguishes them.
"""

import random
from dataclasses import dataclass, replace

import numpy as np

from event_training_data import random_measure
from scorescan.pitch_consensus import PitchPatchInput
from scorescan.score_ir import MeasureIR, PitchIR
from scorescan.visual_evidence import pitch_transaction_gap_pair
from visual_training_data import _evidence, render_measure

PITCH_VISUAL_KINDS = ("single-step", "octave", "adjacent-swap", "chord-tone")


@dataclass(frozen=True)
class RenderedPitchDataset:
    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    scenarios: tuple[str, ...]


def diatonic_value(pitch: PitchIR) -> int:
    return int(pitch.octave) * 7 + "CDEFGAB".index(pitch.step.upper())


def shift_diatonic(pitch: PitchIR, steps: int) -> PitchIR:
    order = "CDEFGAB"
    absolute = diatonic_value(pitch) + int(steps)
    octave, index = divmod(absolute, 7)
    return replace(pitch, step=order[index], octave=max(1, min(8, octave)))


def introduce_pitch_error(
    measure: MeasureIR,
    kind: str,
    rng: random.Random,
) -> tuple[MeasureIR, int, tuple[int, ...]] | None:
    """Return ``(wrong_measure, max_staff_delta, changed_event_indices)``."""
    notes = list(measure.notes)
    pitched = [
        index
        for index, note in enumerate(notes)
        if note.pitch is not None and not note.rest and not note.grace
    ]
    if not pitched:
        return None
    if kind == "single-step":
        index = rng.choice(pitched)
        note = notes[index]
        assert note.pitch is not None
        delta = rng.choice((-2, -1, 1, 2))
        notes[index] = replace(note, pitch=shift_diatonic(note.pitch, delta))
        return replace(measure, notes=tuple(notes)), abs(delta), (index,)
    if kind == "octave":
        index = rng.choice(pitched)
        note = notes[index]
        assert note.pitch is not None
        octave = note.pitch.octave + rng.choice((-1, 1))
        if not 1 <= octave <= 8:
            octave = note.pitch.octave - 1 if octave > 8 else note.pitch.octave + 1
        notes[index] = replace(note, pitch=replace(note.pitch, octave=octave))
        return replace(measure, notes=tuple(notes)), 7, (index,)
    if kind == "adjacent-swap":
        anchors = [index for index in pitched if not notes[index].chord]
        pairs = [
            (left, right)
            for left, right in zip(anchors, anchors[1:])
            if notes[left].pitch != notes[right].pitch
        ]
        if not pairs:
            return None
        left, right = rng.choice(pairs)
        left_pitch, right_pitch = notes[left].pitch, notes[right].pitch
        assert left_pitch is not None and right_pitch is not None
        delta = max(1, abs(diatonic_value(left_pitch) - diatonic_value(right_pitch)))
        notes[left] = replace(notes[left], pitch=right_pitch)
        notes[right] = replace(notes[right], pitch=left_pitch)
        return replace(measure, notes=tuple(notes)), min(14, delta), (left, right)
    if kind != "chord-tone":
        raise ValueError(f"unsupported pitch error kind: {kind}")
    chord = [index for index in pitched if notes[index].chord]
    if not chord:
        return None
    index = rng.choice(chord)
    note = notes[index]
    assert note.pitch is not None
    delta = rng.choice((-2, -1, 1, 2))
    notes[index] = replace(note, pitch=shift_diatonic(note.pitch, delta))
    return replace(measure, notes=tuple(notes)), abs(delta), (index,)


def _bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _input(
    *,
    improvements: tuple[float, ...],
    template_gaps: tuple[float, ...],
    proposal_gaps: tuple[float, ...],
    strict_improvements: tuple[float, ...],
    strict_template_gaps: tuple[float, ...],
    strict_proposal_gaps: tuple[float, ...],
    maximum_delta: int,
    changed_events: int,
    total_events: int,
    rng: random.Random,
) -> PitchPatchInput:
    family_support = rng.choice((0.75, 0.75, 0.75, 1.0))
    margin = 0.50 if family_support == 0.75 else rng.uniform(0.72, 1.0)
    ensemble = rng.uniform(0.86, 0.98)
    return PitchPatchInput(
        candidate_count=rng.randint(5, 7),
        eligible_family_count=4,
        voting_family_count=rng.choice((3, 4, 4)),
        changed_event_count=changed_events,
        total_event_count=max(total_events, changed_events),
        minimum_winner_family_support_ratio=_bounded(family_support - rng.uniform(0.0, 0.025)),
        mean_winner_family_support_ratio=_bounded(family_support + rng.uniform(-0.01, 0.02)),
        minimum_winner_margin_ratio=_bounded(margin - rng.uniform(0.0, 0.04)),
        mean_winner_margin_ratio=_bounded(margin + rng.uniform(-0.02, 0.03)),
        maximum_template_family_support_ratio=rng.uniform(0.0, 0.22),
        family_abstention_ratio=rng.uniform(0.0, 0.08),
        mean_support_page_probability=rng.uniform(0.80, 0.97),
        mean_support_measure_probability=rng.uniform(0.82, 0.98),
        mean_support_visual_probability=rng.uniform(0.48, 0.90),
        mean_support_event_probability=rng.uniform(0.82, 0.98),
        mean_support_context_probability=rng.uniform(0.62, 0.92),
        mean_support_ensemble_probability=ensemble,
        minimum_support_ensemble_probability=_bounded(ensemble - rng.uniform(0.02, 0.12)),
        mean_support_page_score_margin=rng.uniform(8.0, 28.0),
        mean_support_vs_template_measure_probability=rng.uniform(0.05, 0.24),
        # Deliberately weak-to-positive high-level visual evidence simulates a family
        # majority which would be tempting without direct notehead localisation.
        mean_support_vs_template_visual_probability=rng.uniform(-0.03, 0.12),
        mean_support_vs_template_event_probability=rng.uniform(0.06, 0.25),
        mean_support_vs_template_context_probability=rng.uniform(-0.02, 0.18),
        mean_support_vs_template_ensemble_probability=rng.uniform(0.06, 0.25),
        visual_evidence_available=True,
        changed_staff_position_ratio=1.0,
        maximum_staff_position_delta=float(maximum_delta),
        accidental_only_change_ratio=0.0,
        notehead_exact_cell_improvement=improvements[0],
        notehead_near_cell_improvement=improvements[1],
        notehead_vertical_chamfer_improvement=improvements[2],
        notehead_severe_vertical_improvement=improvements[3],
        notehead_visual_unmatched_improvement=improvements[4],
        notehead_column_centroid_improvement=improvements[5],
        notehead_column_order_improvement=improvements[6],
        template_notehead_exact_cell_gap=template_gaps[0],
        template_notehead_near_cell_gap=template_gaps[1],
        template_notehead_vertical_chamfer_gap=template_gaps[2],
        template_notehead_severe_vertical_gap=template_gaps[3],
        template_notehead_visual_unmatched_gap=template_gaps[4],
        template_notehead_column_centroid_gap=template_gaps[5],
        template_notehead_column_order_gap=template_gaps[6],
        proposal_notehead_exact_cell_gap=proposal_gaps[0],
        proposal_notehead_near_cell_gap=proposal_gaps[1],
        proposal_notehead_vertical_chamfer_gap=proposal_gaps[2],
        proposal_notehead_severe_vertical_gap=proposal_gaps[3],
        proposal_notehead_visual_unmatched_gap=proposal_gaps[4],
        proposal_notehead_column_centroid_gap=proposal_gaps[5],
        proposal_notehead_column_order_gap=proposal_gaps[6],
        strict_notehead_exact_cell_improvement=strict_improvements[0],
        strict_notehead_near_cell_improvement=strict_improvements[1],
        strict_notehead_vertical_chamfer_improvement=strict_improvements[2],
        strict_notehead_severe_vertical_improvement=strict_improvements[3],
        strict_notehead_visual_unmatched_improvement=strict_improvements[4],
        strict_notehead_column_centroid_improvement=strict_improvements[5],
        strict_notehead_column_order_improvement=strict_improvements[6],
        template_strict_notehead_exact_cell_gap=strict_template_gaps[0],
        template_strict_notehead_near_cell_gap=strict_template_gaps[1],
        template_strict_notehead_vertical_chamfer_gap=strict_template_gaps[2],
        template_strict_notehead_severe_vertical_gap=strict_template_gaps[3],
        template_strict_notehead_visual_unmatched_gap=strict_template_gaps[4],
        template_strict_notehead_column_centroid_gap=strict_template_gaps[5],
        template_strict_notehead_column_order_gap=strict_template_gaps[6],
        proposal_strict_notehead_exact_cell_gap=strict_proposal_gaps[0],
        proposal_strict_notehead_near_cell_gap=strict_proposal_gaps[1],
        proposal_strict_notehead_vertical_chamfer_gap=strict_proposal_gaps[2],
        proposal_strict_notehead_severe_vertical_gap=strict_proposal_gaps[3],
        proposal_strict_notehead_visual_unmatched_gap=strict_proposal_gaps[4],
        proposal_strict_notehead_column_centroid_gap=strict_proposal_gaps[5],
        proposal_strict_notehead_column_order_gap=strict_proposal_gaps[6],
    )


def build_rendered_pitch_dataset(
    seed: int,
    groups: int,
    *,
    group_offset: int = 0,
) -> RenderedPitchDataset:
    rows: list[list[float]] = []
    labels: list[int] = []
    group_ids: list[int] = []
    scenarios: list[str] = []
    rng = random.Random(seed)
    generated = 0
    attempts = 0
    while generated < groups and attempts < max(100, groups * 24):
        attempts += 1
        kind = PITCH_VISUAL_KINDS[generated % len(PITCH_VISUAL_KINDS)]
        truth = random_measure(rng)
        trapped = introduce_pitch_error(truth, kind, rng)
        if trapped is None:
            continue
        wrong, maximum_delta, changed_indices = trapped
        changed_events = len(changed_indices)
        image, spacing, staff_top, staff_bottom = render_measure(truth, rng)
        evidence = _evidence(image, spacing, staff_top, staff_bottom, generated)
        wrong_gaps, truth_gaps = pitch_transaction_gap_pair(
            evidence, wrong, truth, changed_indices
        )
        strict_wrong_gaps, strict_truth_gaps = pitch_transaction_gap_pair(
            evidence, wrong, truth, changed_indices, strict=True
        )
        correction = tuple(
            max(-1.0, min(1.0, wrong - correct))
            for wrong, correct in zip(wrong_gaps, truth_gaps, strict=True)
        )
        regression = tuple(-value for value in correction)
        strict_correction = tuple(
            max(-1.0, min(1.0, wrong_gap - correct_gap))
            for wrong_gap, correct_gap in zip(
                strict_wrong_gaps, strict_truth_gaps, strict=True
            )
        )
        strict_regression = tuple(-value for value in strict_correction)
        total_events = sum(1 for note in truth.notes if not note.grace)
        # Pair-local RNG state is copied so correction/regression receive identical
        # high-level evidence.  Only the source-crop delta is allowed to distinguish
        # the labels.
        state = rng.getstate()
        correction_input = _input(
            improvements=correction,
            template_gaps=wrong_gaps,
            proposal_gaps=truth_gaps,
            strict_improvements=strict_correction,
            strict_template_gaps=strict_wrong_gaps,
            strict_proposal_gaps=strict_truth_gaps,
            maximum_delta=maximum_delta,
            changed_events=changed_events,
            total_events=total_events,
            rng=rng,
        )
        rng.setstate(state)
        regression_input = _input(
            improvements=regression,
            template_gaps=truth_gaps,
            proposal_gaps=wrong_gaps,
            strict_improvements=strict_regression,
            strict_template_gaps=strict_truth_gaps,
            strict_proposal_gaps=strict_wrong_gaps,
            maximum_delta=maximum_delta,
            changed_events=changed_events,
            total_events=total_events,
            rng=rng,
        )
        group_id = group_offset + generated
        for item, label, direction in (
            (correction_input, 1, "correction"),
            (regression_input, 0, "regression"),
        ):
            rows.append(item.feature_vector())
            labels.append(label)
            group_ids.append(group_id)
            scenarios.append(f"rendered-{kind}-{direction}")
        generated += 1
    if generated != groups:
        raise RuntimeError(f"could only generate {generated}/{groups} rendered pitch groups")
    return RenderedPitchDataset(
        features=np.asarray(rows, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(group_ids, dtype=np.int64),
        scenarios=tuple(scenarios),
    )
