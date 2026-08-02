from __future__ import annotations

"""Deterministic grouped data for the ensemble meta-calibrator.

Every group is a complete three-to-eight-variant decision built from the same preprocessing
families used in production, including the independent system-localized rescue family.  The current released measure, event and context models
are executed to produce their real probability distributions; page and source-visual
signals are generated independently with explicit traps.  Each group contains one
semantically exact candidate, while related candidates may repeat or nearly repeat a
shared error.  This prevents exact-signature size or raw variant count from becoming a
label shortcut.
"""

import math
import random
from dataclasses import dataclass, replace
import numpy as np

from context_training_data import corrupt_segment, mutate as context_mutate, random_segment
from event_training_data import (
    FAMILIES as BASE_FAMILIES,
    FAMILY_NAMES as BASE_FAMILY_NAMES,
    VARIANTS as BASE_VARIANTS,
    mutate_once as event_mutate_once,
)
from scorescan.context_calibration import ContextCalibrator
from scorescan.context_calibration import agreement_profiles as context_agreement_profiles
from scorescan.ensemble_calibration import EnsembleCalibrationInput
from scorescan.event_calibration import EventCalibrator, agreement_profiles
from scorescan.measure_calibration import MeasureCalibrationInput, MeasureCalibrator
from scorescan.score_ir import MeasureIR, measure_distance
from scorescan.visual_evidence import (
    VisualMeasureCalibrator,
    VisualMeasureEvidence,
    semantic_projection_features,
    semantic_features,
)

# The localized candidate is deliberately a fifth independent family.  It runs the
# recognizer on system crops rather than another whole-page preprocessing variant, so
# its evidence must neither be duplicated into a legacy family nor treated as a free
# extra vote.  Keep these constants local to ensemble training: event/context component
# models continue to use their original seven-variant training distributions.
ENSEMBLE_VARIANTS = tuple(BASE_VARIANTS) + (("system_localized", "localization"),)
ENSEMBLE_FAMILIES = tuple(BASE_FAMILIES) + ("localization",)
ENSEMBLE_FAMILY_NAMES = tuple(BASE_FAMILY_NAMES) + ("localization",)
LOCALIZED_INDEX = len(ENSEMBLE_VARIANTS) - 1

SCENARIOS = (
    "clean-majority",
    "clean-agreement",
    "independent-errors",
    "single-family-exact-trap",
    "cross-family-fuzzy-trap",
    "page-score-trap",
    "visual-density-trap",
    "alignment-trap",
    "local-structure-trap",
    "legitimate-transition",
    "invalid-page-trap",
    "missing-visual",
    "evidence-conflict",
    "localized-rescue",
    "localized-isolation-trap",
    "localized-partial-trap",
)


@dataclass(frozen=True)
class EnsembleDataset:
    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    decision_groups: tuple[tuple[int, ...], ...]
    scenarios: tuple[str, ...]


@dataclass(frozen=True)
class FakeCandidate:
    score: float
    calibrated_probability: float
    valid: bool = True


def _ensure_wrong(
    reference: tuple[MeasureIR, MeasureIR, MeasureIR],
    segment: tuple[MeasureIR, MeasureIR, MeasureIR],
    rng: random.Random,
) -> tuple[MeasureIR, MeasureIR, MeasureIR]:
    left, current, right = segment
    for _ in range(10):
        if measure_distance(reference[1], current) > 0.002:
            return left, current, right
        current = event_mutate_once(
            current,
            rng.choice(("step", "octave", "alter", "duration", "onset", "rest", "tie")),
            rng,
        )
    return left, context_mutate(current, rng, ("duration", "step")), right


def _single_event_error(measure: MeasureIR, rng: random.Random, kinds: tuple[str, ...]) -> MeasureIR:
    result = measure
    for kind in kinds:
        result = event_mutate_once(result, kind, rng)
    return result


def _independent_wrong_segment(
    reference: tuple[MeasureIR, MeasureIR, MeasureIR],
    rng: random.Random,
) -> tuple[MeasureIR, MeasureIR, MeasureIR]:
    context_scenario = rng.choice(
        (
            "neighbor-corruption",
            "current-boundary-error",
            "context-neutral-internal-error",
        )
    )
    return _ensure_wrong(reference, corrupt_segment(reference, context_scenario, rng), rng)


def _candidate_segments(
    rng: random.Random,
    scenario: str,
) -> tuple[
    MeasureIR,
    list[MeasureIR],
    list[MeasureIR],
    list[MeasureIR],
    set[int],
]:
    reference = random_segment(
        rng,
        force_transition=scenario == "legitimate-transition",
    )
    if scenario == "localized-rescue":
        correct_index = LOCALIZED_INDEX
    elif scenario in {"localized-isolation-trap", "localized-partial-trap"}:
        correct_index = rng.randrange(LOCALIZED_INDEX)
    else:
        correct_index = rng.randrange(len(ENSEMBLE_VARIANTS))
    correct_indices = {correct_index}
    segments = [_independent_wrong_segment(reference, rng) for _ in ENSEMBLE_VARIANTS]
    segments[correct_index] = reference
    correct_family = ENSEMBLE_FAMILIES[correct_index]

    if scenario == "clean-majority":
        family_choices = [
            next(index for index, candidate_family in enumerate(ENSEMBLE_FAMILIES) if candidate_family == family)
            for family in rng.sample(list(ENSEMBLE_FAMILY_NAMES), k=3)
        ]
        correct_indices = set(family_choices)
        for index in correct_indices:
            segments[index] = reference
    elif scenario == "clean-agreement":
        correct_indices = set(range(len(ENSEMBLE_VARIANTS)))
        segments = [reference for _ in ENSEMBLE_VARIANTS]

    if scenario == "single-family-exact-trap":
        eligible = [family for family in ENSEMBLE_FAMILY_NAMES if family != correct_family and ENSEMBLE_FAMILIES.count(family) > 1]
        family = rng.choice(eligible)
        shared = _independent_wrong_segment(reference, rng)
        for index, candidate_family in enumerate(ENSEMBLE_FAMILIES):
            if candidate_family == family:
                segments[index] = shared

    elif scenario == "cross-family-fuzzy-trap":
        eligible = [family for family in ENSEMBLE_FAMILY_NAMES if family != correct_family]
        selected = set(rng.sample(eligible, k=2))
        shared = _independent_wrong_segment(reference, rng)
        for index, candidate_family in enumerate(ENSEMBLE_FAMILIES):
            if candidate_family not in selected:
                continue
            left, current, right = shared
            # Keep one common semantic error while preventing a strict exact-signature
            # majority from making the ensemble model irrelevant.
            current = _single_event_error(
                current,
                rng,
                (rng.choice(("accidental", "notation", "tie", "slur")),),
            )
            segments[index] = _ensure_wrong(reference, (left, current, right), rng)

    elif scenario == "visual-density-trap":
        trap = rng.choice([index for index in range(len(ENSEMBLE_VARIANTS)) if index not in correct_indices])
        left, current, right = reference
        current = _single_event_error(
            current,
            rng,
            (rng.choice(("step", "octave", "alter", "accidental")),),
        )
        segments[trap] = _ensure_wrong(reference, (left, current, right), rng)

    elif scenario == "local-structure-trap":
        trap = rng.choice([index for index in range(len(ENSEMBLE_VARIANTS)) if index not in correct_indices])
        left, current, right = reference
        current = _single_event_error(
            current,
            rng,
            (rng.choice(("step", "octave", "alter")), rng.choice(("accidental", "notation"))),
        )
        segments[trap] = _ensure_wrong(reference, (left, current, right), rng)

    elif scenario == "legitimate-transition":
        trap = rng.choice([index for index in range(len(ENSEMBLE_VARIANTS)) if index not in correct_indices])
        segments[trap] = _ensure_wrong(
            reference,
            corrupt_segment(reference, "legitimate-transition", rng),
            rng,
        )

    # Guarantee one and only one semantically exact candidate.  Divisions changes are
    # deliberately avoided here because the label is semantic rather than XML-textual.
    for index, segment in enumerate(segments):
        if index in correct_indices:
            segments[index] = reference
            continue
        segments[index] = _ensure_wrong(reference, segment, rng)

    previous = [segment[0] for segment in segments]
    current = [segment[1] for segment in segments]
    following = [segment[2] for segment in segments]
    return reference[1], previous, current, following, correct_indices


def _cluster(
    measures: list[MeasureIR],
    threshold: float = 0.075,
) -> tuple[list[int], int, list[list[float]]]:
    distances = [[measure_distance(left, right) for right in measures] for left in measures]
    choices: list[tuple[int, float, int, list[int]]] = []
    for centre in range(len(measures)):
        members = [index for index, distance in enumerate(distances[centre]) if distance <= threshold]
        mean = sum(distances[centre][index] for index in members) / max(len(members), 1)
        choices.append((len(members), -mean, -centre, members))
    _count, _negative_mean, _negative_centre, best = max(choices)
    medoid = min(
        best,
        key=lambda centre: sum(distances[centre][index] for index in best) / max(len(best), 1),
    )
    return best, medoid, distances


def _margin(values: list[float], index: int) -> float:
    others = [value for other_index, value in enumerate(values) if other_index != index]
    return values[index] - (max(others) if others else 0.5)


def _source_visual_evidence(
    reference: MeasureIR,
    rng: random.Random,
) -> VisualMeasureEvidence:
    (
        anchor,
        pitched,
        rests,
        _chords,
        beam,
        directions,
        articulations,
        accidentals,
        dots,
        open_noteheads,
        _grace,
    ) = semantic_features(reference)
    semantic_complexity = min(
        3.0,
        0.45 * anchor
        + 0.25 * beam
        + 0.15 * directions
        + 0.08 * articulations
        + 0.07 * accidentals,
    )

    def jitter(value: float, deviation: float, ceiling: float = 3.0) -> float:
        return max(0.0, min(ceiling, value + rng.gauss(0.0, deviation)))

    onset_profile, pitch_profile = semantic_projection_features(reference)

    return VisualMeasureEvidence(
        page_index=0,
        system_index=0,
        measure_index=0,
        bbox=(0, 0, 100, 100),
        spacing=10.0,
        ink_density=max(0.0, min(1.0, 0.03 + semantic_complexity * 0.018 + rng.gauss(0.0, 0.003))),
        nonstaff_ink_density=max(0.0, min(1.0, semantic_complexity / 16.0 + rng.gauss(0.0, 0.004))),
        component_density=jitter(max(0.05, rests / 0.35), 0.08),
        notehead_proxy=jitter(anchor, 0.06),
        open_notehead_proxy=jitter(open_noteheads, 0.04),
        stem_proxy=jitter(pitched, 0.06),
        beam_proxy=jitter(beam, 0.05),
        onset_proxy=jitter(anchor, 0.07),
        compact_mark_proxy=jitter(min(3.0, articulations + dots), 0.05),
        accidental_proxy=jitter(accidentals, 0.04),
        above_ink_density=max(0.0, min(1.0, directions / 28.0 + rng.gauss(0.0, 0.003))),
        below_ink_density=max(0.0, min(1.0, directions / 28.0 + rng.gauss(0.0, 0.003))),
        x_ink_profile=tuple(jitter(value, 0.035) for value in onset_profile),
        staff_ink_profile=tuple(jitter(value, 0.035) for value in pitch_profile),
    )



_MODEL_BUNDLE: tuple[MeasureCalibrator, VisualMeasureCalibrator, EventCalibrator, ContextCalibrator] | None = None


def _models() -> tuple[MeasureCalibrator, VisualMeasureCalibrator, EventCalibrator, ContextCalibrator]:
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is None:
        bundle = (
            MeasureCalibrator(),
            VisualMeasureCalibrator(),
            EventCalibrator(),
            ContextCalibrator(),
        )
        if not all(model.enabled for model in bundle):
            raise RuntimeError("current component calibrators must be enabled to build ensemble data")
        _MODEL_BUNDLE = bundle
    return _MODEL_BUNDLE


def _group_seed(seed: int, group: int) -> int:
    # SplitMix-style mixing gives every group an independent deterministic RNG stream.
    value = (int(seed) + 0x9E3779B97F4A7C15 * (int(group) + 1)) & ((1 << 64) - 1)
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return value ^ (value >> 31)


def _build_group_rows(
    task: tuple[int, int],
) -> tuple[int, str, list[list[float]], list[int]]:
    seed, group = task
    rng = random.Random(_group_seed(seed, group))
    scenario = SCENARIOS[group % len(SCENARIOS)]
    measure_model, visual_model, event_model, context_model = _models()
    reference, previous_all, current_all, following_all, correct_indices_all = _candidate_segments(rng, scenario)
    if scenario == "clean-majority":
        candidate_count = rng.choice((3, 4, 5))
    elif scenario.startswith("localized-"):
        candidate_count = rng.choice((5, 6, 7, 8))
    else:
        candidate_count = rng.choice((3, 4, 4, 5, 6, 7, 8))
    required = {0}
    required.add(min(correct_indices_all))
    if scenario.startswith("localized-"):
        required.add(LOCALIZED_INDEX)
    if scenario == "clean-majority":
        required.update(sorted(correct_indices_all)[: min(3, candidate_count)])
    elif scenario == "single-family-exact-trap":
        duplicate_groups: dict[str, list[int]] = {}
        for index, measure in enumerate(current_all):
            if index not in correct_indices_all:
                duplicate_groups.setdefault(measure.fingerprint, []).append(index)
        repeated = [indices for indices in duplicate_groups.values() if len(indices) >= 2]
        if repeated:
            required.update(repeated[0][:2])
    available = [index for index in range(len(ENSEMBLE_VARIANTS)) if index not in required]
    rng.shuffle(available)
    active = sorted(required | set(available[: max(0, candidate_count - len(required))]))
    if len(active) > candidate_count:
        optional_required = [index for index in active if index not in {0, min(correct_indices_all)}]
        while len(active) > candidate_count and optional_required:
            active.remove(optional_required.pop())
    previous = [previous_all[index] for index in active]
    current = [current_all[index] for index in active]
    following = [following_all[index] for index in active]
    families = [ENSEMBLE_FAMILIES[index] for index in active]
    labels_group = [int(index in correct_indices_all) for index in active]
    count = len(current)
    distances_to_reference = [measure_distance(reference, candidate) for candidate in current]
    wrong_indices = [index for index, label in enumerate(labels_group) if not label]
    localized_active_index = active.index(LOCALIZED_INDEX) if LOCALIZED_INDEX in active else -1
    if scenario in {"localized-isolation-trap", "localized-partial-trap"}:
        trap_index = localized_active_index
    else:
        trap_index = rng.choice(wrong_indices) if wrong_indices else -1

    page_scores: list[float] = []
    page_probabilities: list[float] = []
    page_validity: list[bool] = []
    alignments: list[float] = []
    for index, distance in enumerate(distances_to_reference):
        score = 925.0 - 210.0 * min(1.0, distance) + rng.gauss(0.0, 72.0)
        alignment = max(
            0.0,
            min(1.0, 1.0 - 0.72 * min(1.0, distance) + rng.gauss(0.0, 0.065)),
        )
        if scenario == "page-score-trap" and index == trap_index:
            score += 220.0
        if scenario == "alignment-trap" and index == trap_index:
            alignment = min(1.0, alignment + 0.38)
        if scenario == "evidence-conflict" and labels_group[index]:
            score -= 140.0
            alignment = max(0.0, alignment - 0.16)
        if scenario == "localized-rescue" and index == localized_active_index:
            # A stitched candidate may carry a slightly weaker whole-page score even
            # when its local semantic evidence is correct.  Do not teach the model to
            # require the rescue candidate to dominate every page-level signal.
            score -= 70.0
            alignment = max(0.0, alignment - 0.035)
        if scenario in {"localized-isolation-trap", "localized-partial-trap"} and index == trap_index:
            # Local crops can look deceptively clean.  The model must still reject a
            # semantically wrong or incomplete stitch when independent evidence does
            # not corroborate it.
            score += 185.0
            alignment = min(1.0, alignment + 0.30)
        probability = 1.0 / (1.0 + math.exp(-(score - 815.0) / 95.0))
        page_scores.append(score)
        page_probabilities.append(probability)
        invalid_trap = scenario == "invalid-page-trap" and index == trap_index
        incomplete_localized = scenario == "localized-partial-trap" and index == trap_index
        page_validity.append(not (invalid_trap or incomplete_localized))
        alignments.append(alignment)

    evidence = None if scenario == "missing-visual" else _source_visual_evidence(reference, rng)
    signatures: dict[str, list[int]] = {}
    for index, measure in enumerate(current):
        signatures.setdefault(measure.fingerprint, []).append(index)
    exact_indices = max(
        signatures.values(),
        key=lambda indices: (
            len(indices),
            max(page_scores[index] for index in indices),
            -min(indices),
        ),
    )
    initial_cluster, medoid, distance_matrix = _cluster(current)
    template_index = max(range(count), key=lambda index: (page_scores[index], -index))
    missing_count = (
        rng.choice((1, 1, 2))
        if scenario == "alignment-trap" and count < len(ENSEMBLE_VARIANTS)
        else (1 if rng.random() < 0.12 and count < len(ENSEMBLE_VARIANTS) else 0)
    )
    missing_count = min(missing_count, len(ENSEMBLE_VARIANTS) - count)
    eligible_count = count + missing_count
    exact_support_ratio = len(exact_indices) / eligible_count
    semantic_support_ratio = len(initial_cluster) / eligible_count

    event_profiles = agreement_profiles(current, families)
    context_profiles = context_agreement_profiles(previous, current, following, families)
    measure_probabilities: list[float] = []
    visual_probabilities: list[float] = []
    event_probabilities: list[float] = []
    context_probabilities: list[float] = []
    for index, measure in enumerate(current):
        candidate = FakeCandidate(
            score=page_scores[index],
            calibrated_probability=page_probabilities[index],
            valid=page_validity[index],
        )
        measure_probabilities.append(
            measure_model.calibrate(
                MeasureCalibrationInput(
                    candidate=candidate,
                    measure=measure,
                    alignment_similarity=alignments[index],
                    exact_support_ratio=exact_support_ratio,
                    semantic_support_ratio=semantic_support_ratio,
                    missing_ratio=missing_count / eligible_count,
                    distance_to_template=measure_distance(current[template_index], measure),
                    distance_to_medoid=measure_distance(current[medoid], measure),
                    mean_peer_distance=sum(distance_matrix[index]) / count,
                )
            ).probability
        )
        visual_probabilities.append(visual_model.calibrate(evidence, measure).probability)
        event_probabilities.append(event_model.calibrate(event_profiles[index]).probability)
        context_probabilities.append(context_model.calibrate_profile(context_profiles[index]).probability)

    best_page_score = max(page_scores)
    best_alignment = max(alignments)
    full_rows: list[list[float]] = []
    for index, measure in enumerate(current):
        item = EnsembleCalibrationInput(
            page_score=page_scores[index],
            page_probability=page_probabilities[index],
            page_valid=page_validity[index],
            alignment_similarity=alignments[index],
            alignment_margin=alignments[index] - best_alignment,
            exact_support_ratio=exact_support_ratio,
            semantic_support_ratio=semantic_support_ratio,
            signature_support_ratio=len(signatures[measure.fingerprint]) / eligible_count,
            missing_ratio=missing_count / eligible_count,
            distance_to_template=measure_distance(current[template_index], measure),
            distance_to_medoid=measure_distance(current[medoid], measure),
            mean_peer_distance=sum(distance_matrix[index]) / count,
            measure_probability=measure_probabilities[index],
            visual_probability=visual_probabilities[index],
            event_probability=event_probabilities[index],
            context_probability=context_probabilities[index],
            measure_probability_margin=_margin(measure_probabilities, index),
            visual_probability_margin=_margin(visual_probabilities, index),
            event_probability_margin=_margin(event_probabilities, index),
            context_probability_margin=_margin(context_probabilities, index),
            page_score_margin=page_scores[index] - best_page_score,
            candidate_count=count,
            initial_cluster_member=index in initial_cluster,
            exact_signature_member=index in exact_indices,
        )
        full_rows.append(item.feature_vector())
    return group, scenario, full_rows, labels_group


def build_dataset(seed: int, groups: int, workers: int = 1) -> EnsembleDataset:
    tasks = [(seed, group) for group in range(groups)]
    if workers > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        # Use the cross-platform spawn contract explicitly. Forking after NumPy or
        # OpenCV initialises worker threads is unsafe and can make CPU training hang.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            generated = list(executor.map(_build_group_rows, tasks, chunksize=8))
    else:
        generated = [_build_group_rows(task) for task in tasks]
    generated.sort(key=lambda item: item[0])

    features: list[list[float]] = []
    labels: list[int] = []
    group_ids: list[int] = []
    decisions: list[tuple[int, ...]] = []
    scenarios: list[str] = []
    for group, scenario, full_rows, labels_group in generated:
        indices: list[int] = []
        for full, label in zip(full_rows, labels_group, strict=True):
            indices.append(len(labels))
            features.append(full)
            labels.append(label)
            group_ids.append(group)
        decisions.append(tuple(indices))
        scenarios.append(scenario)

    return EnsembleDataset(
        features=np.asarray(features, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int32),
        groups=np.asarray(group_ids, dtype=np.int32),
        decision_groups=tuple(decisions),
        scenarios=tuple(scenarios),
    )
