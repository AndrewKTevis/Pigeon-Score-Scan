from __future__ import annotations

import shutil
import threading
import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

from lxml import etree

from .candidate_calibration import CandidateCalibrator
from .consensus import ConsensusReport, build_measure_consensus, semantic_agreement
from .full_score_consensus import build_full_score_consensus
from .imaging import generate_omr_variants
from .layout import PageLayout
from .localized_recognition import (
    LocalizedSystemResult,
    create_system_crops,
    localized_recognition_eligible,
    merge_localized_system_musicxml,
    validate_localized_system_xml,
)
from .measure_count_resolver import MeasureCountResolution, MeasureCountResolver
from .measure_localized import (
    MeasureLocalizedResult,
    MeasureLocalizedVariantResult,
    choose_measure_localized_variant,
    create_measure_crop,
    create_measure_crop_variants,
    eligible_measure_indices,
    measure_localized_content_signature,
    measure_localized_semantic_signature,
    measure_localized_variant,
    splice_measure_candidate,
    validate_measure_localized_context,
    validate_measure_localized_xml,
)
from .models import PageInfo
from .musicxml import analyze_musicxml, validate_musicxml
from .musicxml_signature import measure_preservation_signatures
from .policy import DEFAULT_POLICY
from .score_ir import audit_production_score, audit_score, score_from_tree
from .omr import EngineResult, HomrRunner
from .tuplet_xml import sanitize_incomplete_implicit_triplets
from .util import atomic_write_json
from .variant_family import variant_family
from .workflow_checkpoint import RecognitionCheckpoint, RecognitionCheckpointKey
from .visual_evidence import (
    VisualMeasureCalibrator,
    VisualMeasureEvidence,
    extract_page_measure_evidence,
    map_evidence_to_measure,
    write_visual_evidence,
)


def _analysis_issues_regressed(
    before: dict[str, object],
    after: dict[str, object],
    key: str,
) -> bool:
    """Return true only when ``after`` introduces an additional issue."""

    def signatures(analysis: dict[str, object]) -> Counter[str]:
        values = analysis.get(key, []) or []
        if not isinstance(values, list):
            return Counter({json.dumps(values, ensure_ascii=False, sort_keys=True, default=str): 1})
        return Counter(
            json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
            for item in values
        )

    before_signatures = signatures(before)
    after_signatures = signatures(after)
    return any(
        count > before_signatures.get(signature, 0)
        for signature, count in after_signatures.items()
    )


@dataclass(frozen=True)
class RecognitionCandidate:
    variant: str
    image_path: str
    xml_path: str | None
    score: float
    valid: bool
    elapsed_seconds: float
    validation_errors: tuple[str, ...] = ()
    measure_count: int = 0
    note_count: int = 0
    rhythm_issue_count: int = 0
    tie_issue_count: int = 0
    slur_issue_count: int = 0
    measure_gap: int | None = None
    expected_measure_count: int | None = None
    measure_gap_penalty: float = 0.0
    agreement_ratio: float = 0.0
    semantic_issue_count: int = 0
    type_duration_mismatch_count: int = 0
    multiple_voice_measure_count: int = 0
    duplicate_direction_count: int = 0
    empty_measure_count: int = 0
    zero_duration_count: int = 0
    chord_duration_mismatch_count: int = 0
    pitch_outlier_count: int = 0
    density_outlier_count: int = 0
    error: str | None = None
    raw_score: float = 0.0
    calibrated_probability: float = 0.5
    calibration_model: str = "disabled"
    visual_compatibility: float = 0.5
    visual_calibration_model: str = "disabled"
    internal_consensus_support: int = 1
    internal_consensus_total: int = 1
    internal_consensus_signature: str | None = None
    part_count: int = 1
    physical_staff_count: int = 1
    expected_physical_staff_count: int | None = None
    staff_count_gap: int | None = None
    staff_count_gap_penalty: float = 0.0
    generalized_score: bool = False
    layout_score_system_count: int | None = None
    layout_physical_staff_appearances: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EnsembleResult:
    selected: RecognitionCandidate | None
    candidates: tuple[RecognitionCandidate, ...]
    consensus: ConsensusReport | None = None
    measure_count_resolution: MeasureCountResolution | None = None
    cancelled: bool = False
    error: str | None = None


def assess_candidate(
    variant: str,
    image_path: Path,
    result: EngineResult,
    expected_measures: int | None,
    expected_physical_staves: int | None = None,
    layout: PageLayout | None = None,
) -> RecognitionCandidate:
    xml_path = result.xml_path
    if xml_path is None or not xml_path.exists():
        return RecognitionCandidate(
            variant=variant,
            image_path=str(image_path),
            xml_path=None,
            score=-10_000.0,
            valid=False,
            raw_score=-10_000.0,
            elapsed_seconds=result.elapsed_seconds,
            error=result.error or "识别引擎未生成 MusicXML",
        )

    validation_errors = tuple(validate_musicxml(xml_path))
    try:
        analysis = analyze_musicxml(xml_path)
    except Exception as exc:
        return RecognitionCandidate(
            variant=variant,
            image_path=str(image_path),
            xml_path=str(xml_path),
            score=-9_000.0,
            valid=False,
            raw_score=-9_000.0,
            elapsed_seconds=result.elapsed_seconds,
            validation_errors=validation_errors,
            error=f"候选 MusicXML 无法分析：{exc}",
        )

    measures = int(analysis.get("measure_count", 0) or 0)
    notes = int(analysis.get("note_count", 0) or 0)
    rhythms = len(analysis.get("rhythm_issues", []) or [])
    ties = len(analysis.get("tie_issues", []) or [])
    slurs = len(analysis.get("slur_issues", []) or [])
    part_count = 1
    physical_staff_count = 1
    generalized_score = False
    try:
        parser = etree.parse(str(xml_path))
        parsed_score = score_from_tree(parser)
        part_count = max(1, len(parsed_score.effective_parts))
        physical_staff_count = max(
            1,
            sum(part.staff_count for part in parsed_score.effective_parts),
        )
        generalized_score = part_count > 1 or physical_staff_count > 1
        semantic_issues = (
            audit_production_score(parsed_score)
            if generalized_score
            else audit_score(parsed_score)
        )
    except Exception:
        semantic_issues = ()
    topology_expectation = (
        layout.expectation_for_staff_topology(physical_staff_count)
        if layout is not None
        else None
    )
    if topology_expectation is not None and topology_expectation.measure_count > 0:
        expected_measures = topology_expectation.measure_count
    gap = abs(measures - expected_measures) if expected_measures and expected_measures > 0 else None
    gap_penalty = min(420.0, gap * 42.0) if gap is not None else 0.0
    type_duration_mismatches = sum(issue.code == "type_duration_mismatch" for issue in semantic_issues)
    multiple_voice_measures = len({issue.measure_index for issue in semantic_issues if issue.code == "multiple_voices"})
    duplicate_directions = sum(issue.code == "duplicate_direction" for issue in semantic_issues)
    empty_measures = sum(issue.code == "empty_measure" for issue in semantic_issues)
    zero_durations = sum(issue.code == "zero_duration" for issue in semantic_issues)
    chord_duration_mismatches = sum(issue.code == "chord_duration_mismatch" for issue in semantic_issues)
    pitch_outliers = sum(issue.code == "pitch_outlier" for issue in semantic_issues)
    density_outliers = sum(issue.code == "density_outlier" for issue in semantic_issues)
    severe_semantic = sum(issue.severity == "error" and issue.code != "zero_duration" for issue in semantic_issues)
    expected_staff_count = int(expected_physical_staves or 0)
    if topology_expectation is not None:
        expected_staff_count = physical_staff_count
        staff_gap = topology_expectation.incomplete_staff_count
    else:
        staff_gap = (
            abs(physical_staff_count - expected_staff_count)
            if expected_staff_count > 0
            else None
        )
    staff_gap_penalty = min(720.0, float(staff_gap * 180)) if staff_gap is not None else 0.0

    # The score intentionally prioritises structurally complete single-voice notation.
    # Semantic agreement with other preprocessing passes is added later.
    score = 1000.0
    score -= 420.0 * len(validation_errors)
    if measures <= 0:
        score -= 700.0
    if notes <= 0:
        score -= 320.0
    score -= 34.0 * rhythms
    score -= 12.0 * ties
    score -= 8.0 * slurs
    score -= 9.0 * type_duration_mismatches
    score -= 55.0 * multiple_voice_measures
    score -= 5.0 * duplicate_directions
    score -= 45.0 * empty_measures
    score -= 120.0 * zero_durations
    score -= 18.0 * chord_duration_mismatches
    score -= 20.0 * pitch_outliers
    score -= 8.0 * density_outliers
    score -= 80.0 * severe_semantic
    score -= staff_gap_penalty
    if gap is not None:
        score -= gap_penalty
    if measures > 0:
        density = notes / max(measures * part_count, 1)
        if density < 0.5:
            score -= 120.0
        elif density > 1.0:
            score += min(35.0, density * 1.5)
    if result.return_code != 0:
        score -= 180.0

    return RecognitionCandidate(
        variant=variant,
        image_path=str(image_path),
        xml_path=str(xml_path),
        score=round(score, 3),
        raw_score=round(score, 3),
        valid=not validation_errors and measures > 0 and notes > 0,
        elapsed_seconds=round(result.elapsed_seconds, 3),
        validation_errors=validation_errors,
        measure_count=measures,
        note_count=notes,
        rhythm_issue_count=rhythms,
        tie_issue_count=ties,
        slur_issue_count=slurs,
        measure_gap=gap,
        expected_measure_count=expected_measures,
        measure_gap_penalty=gap_penalty,
        semantic_issue_count=len(semantic_issues),
        type_duration_mismatch_count=type_duration_mismatches,
        multiple_voice_measure_count=multiple_voice_measures,
        duplicate_direction_count=duplicate_directions,
        empty_measure_count=empty_measures,
        zero_duration_count=zero_durations,
        chord_duration_mismatch_count=chord_duration_mismatches,
        pitch_outlier_count=pitch_outliers,
        density_outlier_count=density_outliers,
        error=result.error,
        part_count=part_count,
        physical_staff_count=physical_staff_count,
        expected_physical_staff_count=expected_staff_count if expected_staff_count > 0 else None,
        staff_count_gap=staff_gap,
        staff_count_gap_penalty=staff_gap_penalty,
        generalized_score=generalized_score,
        layout_score_system_count=(
            topology_expectation.score_system_count
            if topology_expectation is not None
            else None
        ),
        layout_physical_staff_appearances=(
            topology_expectation.physical_staff_appearance_count
            if topology_expectation is not None
            else None
        ),
    )


def _page_preservation_signature(xml_path: Path) -> tuple[str, ...]:
    tree = etree.parse(str(xml_path))
    parts = tree.getroot().xpath("./*[local-name()='part']")
    if not parts:
        raise ValueError("MusicXML has no part")
    tokens: list[str] = [f"parts:{len(parts)}"]
    for part_index, part in enumerate(parts, start=1):
        measures = part.xpath("./*[local-name()='measure']")
        if not measures:
            raise ValueError("MusicXML has no measures")
        staff_count = 1
        for staves in part.xpath("./*[local-name()='measure']/*[local-name()='attributes']/*[local-name()='staves']"):
            try:
                staff_count = max(staff_count, int(staves.text or "1"))
            except ValueError:
                continue
        tokens.append(f"part:{part_index}:staves:{staff_count}:measures:{len(measures)}")
        tokens.extend(
            f"part:{part_index}:measure:{measure_index}:{signature}"
            for measure_index, signature in enumerate(
                measure_preservation_signatures(measures),
                start=1,
            )
        )
    return tuple(tokens)


def _full_score_exact_agreement(
    candidates: list[RecognitionCandidate],
) -> dict[str, float]:
    signatures: dict[str, tuple[str, ...]] = {}
    for candidate in candidates:
        if not candidate.xml_path:
            continue
        try:
            signatures[candidate.variant] = _page_preservation_signature(Path(candidate.xml_path))
        except (OSError, ValueError, etree.XMLSyntaxError):
            continue
    scores = {candidate.variant: 0.0 for candidate in candidates}
    comparisons = {candidate.variant: 0 for candidate in candidates}
    items = list(signatures.items())
    for left_index, (left_variant, left) in enumerate(items):
        for right_variant, right in items[left_index + 1:]:
            left_topology = tuple(token for token in left if ":measure:" not in token)
            right_topology = tuple(token for token in right if ":measure:" not in token)
            if left_topology != right_topology:
                similarity = 0.0
            else:
                left_measures = [token.rsplit(":", 1)[-1] for token in left if ":measure:" in token]
                right_measures = [token.rsplit(":", 1)[-1] for token in right if ":measure:" in token]
                exact = sum(
                    left_hash == right_hash
                    for left_hash, right_hash in zip(left_measures, right_measures, strict=False)
                )
                similarity = exact / max(len(left_measures), len(right_measures), 1)
            scores[left_variant] += similarity
            scores[right_variant] += similarity
            comparisons[left_variant] += 1
            comparisons[right_variant] += 1
    for variant, count in comparisons.items():
        if count:
            scores[variant] /= count
    return scores


def choose_lowres_upscale_internal_candidate(
    internal: list[RecognitionCandidate],
) -> RecognitionCandidate | None:
    """Collapse correlated scale probes into one strict internal-family result."""

    groups: dict[tuple[str, ...], list[RecognitionCandidate]] = {}
    for candidate in internal:
        if not candidate.valid or not candidate.xml_path:
            continue
        try:
            signature = _page_preservation_signature(Path(candidate.xml_path))
        except (OSError, ValueError, etree.XMLSyntaxError):
            continue
        groups.setdefault(signature, []).append(candidate)
    if not groups:
        return None
    signature, members = max(
        groups.items(),
        key=lambda item: (
            len(item[1]),
            max(candidate.raw_score for candidate in item[1]),
            item[0],
        ),
    )
    support = len(members)
    required = max(
        DEFAULT_POLICY.lowres_strip_internal_min_support,
        len(internal) // 2 + 1,
    )
    if support < required:
        return None
    representative = max(
        members,
        key=lambda candidate: (
            candidate.raw_score,
            candidate.score,
            candidate.variant,
        ),
    )
    bonus = DEFAULT_POLICY.lowres_strip_candidate_bonus
    return replace(
        representative,
        variant="upscale",
        score=round(representative.score + bonus, 3),
        raw_score=round(representative.raw_score + bonus, 3),
        elapsed_seconds=round(sum(item.elapsed_seconds for item in internal), 3),
        internal_consensus_support=support,
        internal_consensus_total=len(internal),
        internal_consensus_signature="|".join(signature),
    )


def choose_lowres_upscale_center_fallback(
    internal: list[RecognitionCandidate],
) -> RecognitionCandidate | None:
    """Keep the established centre-scale result when strict consensus abstains.

    A split scale family must not earn a consensus bonus or override a measure-count
    prior.  It also must not turn a valid centre-scale recognition into a page-level
    failure: that centre probe is the same primary upscale used before multi-scale
    consensus was introduced.
    """

    center = next(
        (
            candidate
            for candidate in internal
            if candidate.variant == "upscale"
            and candidate.valid
            and candidate.xml_path
        ),
        None,
    )
    if center is None:
        return None
    signature: tuple[str, ...] = ()
    try:
        signature = _page_preservation_signature(Path(center.xml_path))
    except (OSError, ValueError, etree.XMLSyntaxError):
        pass
    return replace(
        center,
        elapsed_seconds=round(sum(item.elapsed_seconds for item in internal), 3),
        internal_consensus_support=1,
        internal_consensus_total=len(internal),
        internal_consensus_signature="|".join(signature) if signature else None,
    )


def promote_lowres_consensus_measure_count(
    resolution: MeasureCountResolution,
    candidates: list[RecognitionCandidate],
) -> MeasureCountResolution:
    """Let a clean strict multi-scale result escape a weak circular layout prior.

    The override stays deliberately narrow: the page geometry must itself be weak,
    the exact internal scale family must have a strict majority, its structural score
    before the provisional count penalty must exceed every competing candidate by a
    fixed margin, and the chosen count must already be an observed resolver option.
    """

    if resolution.layout_confidence >= DEFAULT_POLICY.risky_layout_confidence:
        return resolution
    eligible = [
        candidate
        for candidate in candidates
        if candidate.variant == "upscale"
        and candidate.valid
        and candidate.internal_consensus_support
        >= DEFAULT_POLICY.lowres_strip_internal_min_support
        and candidate.internal_consensus_support * 2
        > candidate.internal_consensus_total
        and candidate.rhythm_issue_count == 0
        and candidate.tie_issue_count == 0
        and candidate.slur_issue_count == 0
        and candidate.semantic_issue_count == 0
        and candidate.type_duration_mismatch_count == 0
        and candidate.multiple_voice_measure_count == 0
        and candidate.empty_measure_count == 0
        and candidate.zero_duration_count == 0
        and candidate.chord_duration_mismatch_count == 0
        and candidate.pitch_outlier_count == 0
    ]
    if len(eligible) != 1:
        return resolution
    candidate = eligible[0]
    if candidate.measure_count <= 0 or candidate.measure_count == resolution.selected_count:
        return resolution
    candidate_base = candidate.raw_score + candidate.measure_gap_penalty
    competitors = [
        item.raw_score + item.measure_gap_penalty
        for item in candidates
        if item is not candidate and item.valid and item.measure_count > 0
    ]
    if (
        competitors
        and candidate_base - max(competitors)
        < DEFAULT_POLICY.lowres_strip_count_override_score_margin
    ):
        return resolution
    option = next(
        (item for item in resolution.options if item.count == candidate.measure_count),
        None,
    )
    if option is None:
        return resolution
    return replace(
        resolution,
        selected_count=candidate.measure_count,
        probability=option.probability,
        margin=0.0,
        source="lowres_internal_consensus",
    )


def raw_tuplet_candidate_warranted(candidate: RecognitionCandidate) -> bool:
    """Bound the extra homr pass to structurally difficult notation pages."""

    return bool(
        candidate.valid
        and candidate.xml_path
        and candidate.generalized_score
        and candidate.measure_count >= 3
        and candidate.note_count >= 24
        and candidate.rhythm_issue_count * 5 >= candidate.measure_count
    )


def choose_tuplet_cleanup_internal_candidate(
    internal: list[RecognitionCandidate],
) -> RecognitionCandidate | None:
    """Keep one correlated sibling and account for both engine executions."""

    selected = choose_candidate(internal)
    if selected is None:
        return None
    return replace(
        selected,
        elapsed_seconds=round(
            sum(max(0.0, item.elapsed_seconds) for item in internal),
            3,
        ),
        internal_consensus_support=1,
        internal_consensus_total=len(internal),
    )


def inherit_sparse_page_evidence(
    candidate: RecognitionCandidate,
    template: RecognitionCandidate,
) -> RecognitionCandidate:
    """Inherit only non-independent page priors for a sparse local candidate.

    A measure-localised candidate is the complete template with one measure replaced.
    Re-running whole-page agreement scoring would reward the copied measures as if they
    were independent evidence, while leaving the candidate at default probability 0.5
    would unfairly veto every rescue.  Preserve the template's page calibration and
    agreement prior, but never give the sparse candidate a higher page score.  Measure,
    event, visual and context evidence is still recomputed from its own target measure
    inside consensus.
    """
    if not candidate.valid:
        return candidate
    structural_delta = min(0.0, float(candidate.raw_score) - float(template.raw_score))
    return replace(
        candidate,
        score=round(float(template.score) + structural_delta, 3),
        agreement_ratio=float(template.agreement_ratio),
        calibrated_probability=float(template.calibrated_probability),
        calibration_model=str(template.calibration_model),
        visual_compatibility=float(template.visual_compatibility),
        visual_calibration_model=str(template.visual_calibration_model),
    )


def apply_expected_measure_count(
    candidate: RecognitionCandidate,
    expected_measures: int | None,
) -> RecognitionCandidate:
    """Rebase only the provisional measure-count penalty.

    Layout is available before OMR and therefore supplies the first expectation.  Once
    multiple independent candidates exist, the count resolver may produce a better
    estimate.  Recomputing just this bounded penalty avoids rerunning XML analysis and
    prevents the initial layout estimate from becoming a circular prior.
    """
    expected = int(expected_measures or 0)
    gap = abs(candidate.measure_count - expected) if expected > 0 and candidate.measure_count > 0 else None
    penalty = min(420.0, gap * 42.0) if gap is not None else 0.0
    base_without_gap = candidate.raw_score + candidate.measure_gap_penalty
    raw_score = base_without_gap - penalty
    return replace(
        candidate,
        raw_score=round(raw_score, 3),
        score=round(raw_score, 3),
        measure_gap=gap,
        expected_measure_count=expected if expected > 0 else None,
        measure_gap_penalty=penalty,
    )


_MEASURE_COUNT_RESOLVER = MeasureCountResolver()


def reconcile_measure_count_resolution(
    resolution: MeasureCountResolution,
    output_measure_count: int,
    *,
    source: str,
) -> MeasureCountResolution:
    """Bind persisted count evidence to the MusicXML that will actually be returned.

    Consensus is allowed to fail closed and fall back to the strongest complete-page
    candidate.  In that case the previously resolved target count may differ from the
    output tree.  Persisting the unattained target would make quality reporting and
    crash recovery describe a score that was never emitted.
    """
    output_measure_count = max(0, int(output_measure_count))
    if output_measure_count <= 0 or output_measure_count == resolution.selected_count:
        return resolution
    option = next(
        (item for item in resolution.options if item.count == output_measure_count),
        None,
    )
    return replace(
        resolution,
        selected_count=output_measure_count,
        probability=option.probability if option is not None else 0.5,
        margin=0.0,
        source=source,
    )


_CANDIDATE_CALIBRATOR = CandidateCalibrator()


def add_agreement_scores(candidates: list[RecognitionCandidate]) -> list[RecognitionCandidate]:
    generalized_page = any(candidate.generalized_score for candidate in candidates)
    agreements = (
        _full_score_exact_agreement(candidates)
        if generalized_page
        else semantic_agreement(candidates)
    )
    enriched: list[RecognitionCandidate] = []
    for candidate in candidates:
        ratio = float(agreements.get(candidate.variant, 0.0))
        # Cross-variant agreement remains primary probabilistic evidence. The trained
        # calibrator only reorders close candidates and is bounded to ±32 points.
        bonus = min(70.0, ratio * 70.0)
        with_agreement = replace(
            candidate,
            score=round(candidate.raw_score + bonus, 3),
            agreement_ratio=round(ratio, 6),
        )
        if generalized_page:
            # The maintained candidate calibrator is trained on one-staff pages.
            # Its probability is not meaningful for piano/ensemble topology and
            # previously assigned near-certain confidence to a one-part truncation.
            enriched.append(
                replace(
                    with_agreement,
                    calibrated_probability=0.5,
                    calibration_model="disabled-out-of-domain-generalized-score",
                )
            )
            continue
        calibration = _CANDIDATE_CALIBRATOR.calibrate(with_agreement)
        enriched.append(
            replace(
                with_agreement,
                score=round(with_agreement.score + calibration.adjustment, 3),
                calibrated_probability=calibration.probability,
                calibration_model=calibration.model_version,
            )
        )
    return enriched


_VISUAL_CALIBRATOR = VisualMeasureCalibrator()


def add_visual_compatibility(
    candidates: list[RecognitionCandidate],
    evidence: tuple[VisualMeasureEvidence, ...],
) -> list[RecognitionCandidate]:
    if not evidence or not _VISUAL_CALIBRATOR.enabled:
        return candidates
    enriched: list[RecognitionCandidate] = []
    for candidate in candidates:
        if candidate.generalized_score:
            # The current visual calibrator is trained on a single physical
            # staff. Applying it to the first part of a full score would create
            # a spurious ranking signal.
            enriched.append(candidate)
            continue
        probability = 0.5
        if candidate.xml_path:
            try:
                tree = etree.parse(candidate.xml_path)
                score_ir = score_from_tree(tree)
                probabilities = [
                    _VISUAL_CALIBRATOR.predict_probability(
                        map_evidence_to_measure(evidence, index, len(score_ir.measures)),
                        measure,
                    )
                    for index, measure in enumerate(score_ir.measures)
                ]
                if probabilities:
                    # Trim one extreme at either side for long pages so a single badly
                    # estimated barline cannot dominate the page-level evidence.
                    ordered = sorted(probabilities)
                    if len(ordered) >= 8:
                        ordered = ordered[1:-1]
                    probability = sum(ordered) / len(ordered)
            except Exception:
                probability = 0.5
        adjustment = max(-10.0, min(10.0, (probability - 0.5) * 20.0))
        enriched.append(
            replace(
                candidate,
                score=round(candidate.score + adjustment, 3),
                visual_compatibility=round(probability, 6),
                visual_calibration_model=_VISUAL_CALIBRATOR.model_version,
            )
        )
    return enriched


def candidate_is_strong(candidate: RecognitionCandidate, expected_measures: int | None) -> bool:
    allowed_gap = 1 if not expected_measures else max(1, round(expected_measures * DEFAULT_POLICY.strong_measure_gap_ratio))
    return (
        candidate.valid
        and candidate.rhythm_issue_count == 0
        and candidate.type_duration_mismatch_count == 0
        and candidate.multiple_voice_measure_count == 0
        and candidate.empty_measure_count == 0
        and candidate.zero_duration_count == 0
        and candidate.chord_duration_mismatch_count == 0
        and candidate.pitch_outlier_count == 0
        and (candidate.staff_count_gap is None or candidate.staff_count_gap == 0)
        and (candidate.measure_gap is None or candidate.measure_gap <= allowed_gap)
        and candidate.score >= DEFAULT_POLICY.strong_candidate_min_score
    )


def _candidate_rank_key(candidate: RecognitionCandidate) -> tuple[bool, float, float, int]:
    return (
        candidate.valid,
        candidate.score,
        candidate.agreement_ratio,
        candidate.note_count,
    )


def ranked_candidates(candidates: list[RecognitionCandidate]) -> list[RecognitionCandidate]:
    return sorted(
        (candidate for candidate in candidates if candidate.xml_path),
        key=_candidate_rank_key,
        reverse=True,
    )


def choose_candidate(candidates: list[RecognitionCandidate]) -> RecognitionCandidate | None:
    ranked = ranked_candidates(candidates)
    return ranked[0] if ranked else None


def candidate_set_is_ambiguous(
    candidates: list[RecognitionCandidate],
    *,
    agreement_floor: float = DEFAULT_POLICY.disagreement_agreement_floor,
    score_margin: float = DEFAULT_POLICY.disagreement_score_margin,
) -> bool:
    ranked = ranked_candidates(candidates)
    if len(ranked) < 2:
        return True
    best, runner_up = ranked[0], ranked[1]
    if best.measure_count != runner_up.measure_count:
        return True
    if max(item.agreement_ratio for item in ranked) < agreement_floor:
        return True
    return best.score - runner_up.score < score_margin


def should_run_localized_recognition(
    page: PageInfo,
    layout: PageLayout | None,
    candidates: list[RecognitionCandidate],
    resolution: MeasureCountResolution,
) -> tuple[bool, str]:
    eligible, reason = localized_recognition_eligible(layout)
    if not eligible:
        return False, reason
    best = choose_candidate(candidates)
    if best is None:
        return True, "no_complete_page_candidate"
    if best.generalized_score:
        return False, "generalized_score_uses_full_page_recognition"
    if not candidate_is_strong(best, resolution.selected_count):
        return True, "best_complete_page_candidate_not_strong"
    if resolution.probability < DEFAULT_POLICY.measure_count_probability_floor:
        return True, "measure_count_probability_low"
    if resolution.margin < DEFAULT_POLICY.measure_count_margin_floor:
        return True, "measure_count_margin_low"
    if candidate_set_is_ambiguous(
        candidates,
        agreement_floor=DEFAULT_POLICY.localized_trigger_agreement_floor,
        score_margin=DEFAULT_POLICY.localized_trigger_score_margin,
    ):
        return True, "complete_page_candidates_ambiguous"
    if (page.quality_score or 0.0) < DEFAULT_POLICY.risky_page_quality:
        return True, "page_quality_risky"
    return False, "complete_page_candidates_strong"


def _initial_variant_names(variants: list[tuple[str, Path]]) -> list[str]:
    """Choose the smallest deterministic set covering independent families."""

    selected: list[str] = []
    families: set[str] = set()
    for name, _path in variants:
        family = variant_family(name)
        if family in families:
            continue
        selected.append(name)
        families.add(family)
        if len(families) >= DEFAULT_POLICY.minimum_initial_candidate_families:
            break
    if not selected and variants:
        selected.append(variants[0][0])
    return selected


class RecognitionEnsemble:
    """Retry OMR on deterministic scan variants and fuse their MusicXML semantics.

    Whole-page structural ranking chooses a strong template. Equal-length candidates
    are then compared measure by measure; a semantic majority can replace individual
    measures in the template. This avoids discarding a good measure merely because a
    different preprocessing variant was weaker elsewhere on the page.
    """

    def __init__(
        self,
        runner: HomrRunner,
        log: Callable[[str], None],
        variants_root: Path,
    ) -> None:
        self.runner = runner
        self.log = log
        self.variants_root = variants_root

    def run_page(
        self,
        page: PageInfo,
        layout: PageLayout | None,
        cancel_event: threading.Event | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> EnsembleResult:
        last_progress = 0.0
        progress_callback_failed = False

        def report_progress(fraction: float, stage: str) -> None:
            """Report best-effort, monotonic progress for a dynamically-sized page."""

            nonlocal last_progress, progress_callback_failed
            last_progress = min(1.0, max(last_progress, max(0.0, float(fraction))))
            if progress_callback is None or progress_callback_failed:
                return
            try:
                progress_callback(last_progress, stage)
            except Exception as exc:
                # UI persistence must never turn a valid OMR result into a failed
                # conversion. Log once and keep the recognition transaction alive.
                progress_callback_failed = True
                self.log(f"第 {page.index} 页：进度更新失败，识别将继续：{exc}")

        engine_runs_completed = 0

        def report_engine_completion(stage: str) -> None:
            nonlocal engine_runs_completed
            engine_runs_completed += 1
            # Candidate counts are evidence-driven and unknowable in advance. This
            # band advances after real engine work without pretending exact ETA.
            report_progress(
                min(0.68, 0.08 + 0.075 * engine_runs_completed),
                stage,
            )

        report_progress(0.01, "Preparing page recognition")
        page_root = self.variants_root / f"page_{page.index:04d}"
        checkpoint = page_root / "selected.musicxml"
        consensus_json = page_root / "consensus.json"
        count_resolution_json = page_root / "measure_count_resolution.json"
        normalized_page = Path(page.normalized_path or page.image_path)
        layout_path = Path(page.layout_path) if page.layout_path else None
        checkpoint_store = RecognitionCheckpoint(checkpoint)
        checkpoint_key = RecognitionCheckpointKey.for_page(normalized_page, layout_path)
        if checkpoint.exists() and checkpoint.stat().st_size > 300:
            if checkpoint_store.is_valid(checkpoint_key):
                result = EngineResult(0, checkpoint, 0.0)
                candidate = assess_candidate(
                    page.recognition_variant or "checkpoint",
                    normalized_page,
                    result,
                    page.estimated_measure_count or None,
                    None,
                    layout,
                )
                if candidate.valid:
                    report_progress(1.0, "Using validated recognition checkpoint")
                    resolution = MeasureCountResolution(
                        selected_count=max(candidate.measure_count, 1),
                        probability=1.0,
                        margin=1.0,
                        source="checkpoint",
                        model_version=_MEASURE_COUNT_RESOLVER.model_version,
                        model_status=_MEASURE_COUNT_RESOLVER.model_status,
                        layout_count=page.estimated_measure_count,
                        layout_confidence=layout.confidence if layout is not None else 0.0,
                        deterministic_count=max(candidate.measure_count, 1),
                        options=(),
                    )
                    return EnsembleResult(candidate, (candidate,), measure_count_resolution=resolution)
                checkpoint_store.invalidate("semantic-invalid")
                self.log(f"第 {page.index} 页：识别检查点语义校验失败，已自动重新识别")
            else:
                checkpoint_store.invalidate("workflow-mismatch")
                self.log(f"第 {page.index} 页：识别检查点与当前工作流不匹配，已自动重新识别")

        visual_evidence: tuple[VisualMeasureEvidence, ...] = ()
        variants = generate_omr_variants(page, page_root, layout=layout)
        report_progress(0.08, "Recognition candidates prepared")
        by_name = {name: path for name, path in variants}
        expected = page.estimated_measure_count or None
        ordered = _initial_variant_names(variants)
        candidates: list[RecognitionCandidate] = []
        visual_evidence_cache: dict[
            tuple[int, int],
            tuple[VisualMeasureEvidence, ...],
        ] = {}

        def rank_with_count(
            source: list[RecognitionCandidate],
        ) -> tuple[list[RecognitionCandidate], MeasureCountResolution]:
            preliminary = add_agreement_scores(source)
            topology_best = choose_candidate(preliminary)
            layout_count = page.estimated_measure_count
            if (
                topology_best is not None
                and topology_best.expected_measure_count is not None
                and topology_best.expected_measure_count > 0
            ):
                layout_count = topology_best.expected_measure_count
            resolution = _MEASURE_COUNT_RESOLVER.resolve(
                layout_count=layout_count,
                layout_confidence=layout.confidence if layout is not None else 0.0,
                candidates=tuple(preliminary),
            )
            resolution = promote_lowres_consensus_measure_count(
                resolution,
                preliminary,
            )
            rebased = [apply_expected_measure_count(item, resolution.selected_count) for item in source]
            ranked = add_agreement_scores(rebased)
            atomic_write_json(count_resolution_json, resolution.to_dict())
            return ranked, resolution

        def rebuild_visual_evidence(
            target_measure_count: int,
            simultaneous_staff_count: int,
        ) -> tuple[VisualMeasureEvidence, ...]:
            cache_key = (
                max(0, int(target_measure_count or 0)),
                max(1, int(simultaneous_staff_count or 1)),
            )
            cached = visual_evidence_cache.get(cache_key)
            if cached is not None:
                return cached
            if layout is None or not layout.systems:
                visual_evidence_cache[cache_key] = ()
                return ()
            if cache_key[1] != 1:
                # Existing measure evidence is a one-staff model. A full score
                # requires staff-aware evidence and must not duplicate the same
                # semantic measures once per physical staff.
                visual_evidence_cache[cache_key] = ()
                return ()
            evidence = extract_page_measure_evidence(
                Path(page.normalized_path or page.image_path),
                layout,
                page_index=page.index,
                target_measure_count=cache_key[0] or None,
            )
            visual_evidence_cache[cache_key] = evidence
            if evidence:
                write_visual_evidence(page_root / "visual_evidence.json", evidence)
            return evidence

        def run_named(name: str) -> RecognitionCandidate | None:
            path = by_name.get(name)
            if path is None:
                return None
            internal_paths: list[tuple[str, Path]] = [(name, path)]
            if name == "upscale":
                low = path.with_name(f"{path.stem}_low{path.suffix}")
                high = path.with_name(f"{path.stem}_high{path.suffix}")
                internal_paths = [
                    (f"{name}:low", low),
                    (name, path),
                    (f"{name}:high", high),
                ]
                internal_paths = [
                    (variant, candidate_path)
                    for variant, candidate_path in internal_paths
                    if candidate_path.is_file()
                ]

            internal: list[RecognitionCandidate] = []
            for internal_name, internal_path in internal_paths:
                report_progress(last_progress, f"Running candidate {internal_name}")
                self.log(f"第 {page.index} 页：运行 {internal_name} 识别候选")
                engine_result = self.runner.run_page(internal_path, cancel_event)
                if engine_result.cancelled:
                    raise InterruptedError("cancelled")
                report_engine_completion(f"Candidate {internal_name} complete")
                internal.append(
                    assess_candidate(
                        internal_name,
                        internal_path,
                        engine_result,
                        expected,
                        None,
                        layout,
                    )
                )
            if (
                name == "staffnorm"
                and len(internal) == 1
                and raw_tuplet_candidate_warranted(internal[0])
            ):
                raw_path = path.with_name(f"{path.stem}_raw_tuplets{path.suffix}")
                shutil.copy2(path, raw_path)
                raw_name = f"{name}:raw_tuplets"
                report_progress(last_progress, f"Running candidate {raw_name}")
                self.log(f"第 {page.index} 页：运行 {raw_name} 受控内部候选")
                raw_result = self.runner.run_page(
                    raw_path,
                    cancel_event,
                    preserve_raw_tuplets=True,
                )
                if raw_result.cancelled:
                    raise InterruptedError("cancelled")
                if raw_result.xml_path is not None:
                    sanitized_path = raw_result.xml_path.with_name(
                        f"{raw_result.xml_path.stem}_sanitized.musicxml"
                    )
                    try:
                        sanitization = sanitize_incomplete_implicit_triplets(
                            raw_result.xml_path,
                            sanitized_path,
                        )
                    except Exception as exc:
                        self.log(
                            f"第 {page.index} 页：{raw_name} 三连音完整性清理失败，"
                            f"已安全弃用该内部候选：{exc}"
                        )
                        raw_result = replace(
                            raw_result,
                            return_code=1,
                            xml_path=None,
                            error=f"raw tuplet sanitization failed: {exc}",
                        )
                    else:
                        atomic_write_json(
                            page_root / "raw_tuplet_sanitization.json",
                            sanitization,
                        )
                        if sanitization.get("topology_valid") is True:
                            raw_result = replace(
                                raw_result,
                                xml_path=sanitized_path,
                            )
                        else:
                            self.log(
                                f"第 {page.index} 页：{raw_name} 三连音拓扑未闭合，"
                                "已安全弃用该内部候选"
                            )
                            raw_result = replace(
                                raw_result,
                                return_code=1,
                                xml_path=None,
                                error="raw tuplet topology is not closed",
                            )
                report_engine_completion(f"Candidate {raw_name} complete")
                internal.append(
                    assess_candidate(
                        raw_name,
                        raw_path,
                        raw_result,
                        expected,
                        None,
                        layout,
                    )
                )
            if name == "upscale" and len(internal) > 1:
                candidate = choose_lowres_upscale_internal_candidate(internal)
                selection = "strict_consensus"
                if candidate is None:
                    candidate = choose_lowres_upscale_center_fallback(internal)
                    selection = "center_fallback" if candidate is not None else "abstained"
                atomic_write_json(
                    page_root / "lowres_upscale_consensus.json",
                    {
                        "format": 1,
                        "selection": selection,
                        "selected": candidate.to_dict() if candidate is not None else None,
                        "internal_candidates": [item.to_dict() for item in internal],
                    },
                )
                if candidate is None:
                    candidate = RecognitionCandidate(
                        variant=name,
                        image_path=str(path),
                        xml_path=None,
                        score=-10_000.0,
                        raw_score=-10_000.0,
                        valid=False,
                        elapsed_seconds=round(
                            sum(item.elapsed_seconds for item in internal),
                            3,
                        ),
                        internal_consensus_support=0,
                        internal_consensus_total=len(internal),
                        error="低分辨率多尺度候选未形成严格内部多数",
                    )
            elif name == "staffnorm" and len(internal) > 1:
                candidate = choose_tuplet_cleanup_internal_candidate(internal)
                if candidate is None:
                    candidate = internal[0]
                atomic_write_json(
                    page_root / "raw_tuplet_ablation.json",
                    {
                        "format": 1,
                        "selection": candidate.variant,
                        "selected": candidate.to_dict(),
                        "internal_candidates": [item.to_dict() for item in internal],
                    },
                )
            else:
                candidate = internal[0]
            candidates.append(candidate)
            self.log(
                f"第 {page.index} 页：{name} 候选基础得分 {candidate.score:.1f}，"
                f"小节 {candidate.measure_count}，节奏疑点 {candidate.rhythm_issue_count}"
            )
            return candidate

        def run_localized(expected_measures: int, trigger_reason: str) -> RecognitionCandidate:
            variant = "system_localized"
            localized_root = page_root / "localized"
            diagnostics_path = localized_root / "localized_recognition.json"
            total_elapsed = 0.0
            system_results: list[LocalizedSystemResult] = []
            try:
                assert layout is not None
                crops = create_system_crops(normalized_page, layout, localized_root)
                xml_paths: list[Path] = []
                failure: str | None = None
                for crop in crops:
                    crop_path = Path(crop.image_path)
                    report_progress(
                        0.70 + 0.12 * (crop.system_index - 1) / max(len(crops), 1),
                        f"Re-recognizing system {crop.system_index}/{len(crops)}",
                    )
                    self.log(
                        f"第 {page.index} 页：局部识别系统 {crop.system_index}/{len(crops)}"
                    )
                    engine_result = self.runner.run_page(crop_path, cancel_event)
                    total_elapsed += max(0.0, float(engine_result.elapsed_seconds))
                    if engine_result.cancelled:
                        raise InterruptedError("cancelled")
                    report_progress(
                        0.70 + 0.12 * crop.system_index / max(len(crops), 1),
                        f"System {crop.system_index}/{len(crops)} complete",
                    )
                    valid = False
                    observed = 0
                    error = engine_result.error
                    if engine_result.return_code == 0 and engine_result.xml_path is not None:
                        valid, observed, validation_error = validate_localized_system_xml(
                            engine_result.xml_path, crop.expected_measure_count
                        )
                        error = validation_error
                    else:
                        error = error or "system recognition failed"
                    system_results.append(
                        LocalizedSystemResult(
                            system_index=crop.system_index,
                            image_path=crop.image_path,
                            xml_path=str(engine_result.xml_path) if engine_result.xml_path else None,
                            return_code=int(engine_result.return_code),
                            elapsed_seconds=round(float(engine_result.elapsed_seconds), 3),
                            valid=valid,
                            expected_measure_count=crop.expected_measure_count,
                            observed_measure_count=observed,
                            error=error,
                        )
                    )
                    if not valid:
                        failure = f"system {crop.system_index}: {error or 'invalid result'}"
                        break
                    assert engine_result.xml_path is not None
                    xml_paths.append(engine_result.xml_path)

                if failure is None and len(xml_paths) == len(crops):
                    merged = page_root / "system_localized.musicxml"
                    merge_localized_system_musicxml(xml_paths, merged)
                    engine_result = EngineResult(0, merged, total_elapsed)
                else:
                    engine_result = EngineResult(
                        65,
                        None,
                        total_elapsed,
                        error=failure or "localised recognition did not cover every system",
                    )
            except InterruptedError:
                raise
            except Exception as exc:
                engine_result = EngineResult(65, None, total_elapsed, error=str(exc))

            candidate = assess_candidate(
                variant,
                normalized_page,
                engine_result,
                expected_measures or expected,
                None,
                layout,
            )
            candidates.append(candidate)
            atomic_write_json(
                diagnostics_path,
                {
                    "format": 1,
                    "trigger_reason": trigger_reason,
                    "variant": variant,
                    "valid": candidate.valid,
                    "candidate": candidate.to_dict(),
                    "systems": [item.to_dict() for item in system_results],
                },
            )
            if candidate.valid:
                self.log(
                    f"第 {page.index} 页：系统局部化候选完成，小节 {candidate.measure_count}，"
                    f"基础得分 {candidate.score:.1f}"
                )
            else:
                self.log(
                    f"第 {page.index} 页：系统局部化候选未通过完整性安全门，已弃权"
                )
            return candidate

        def run_measure_localized(
            measure_index: int,
            template_path: Path,
            expected_measures: int,
            evidence_set: tuple[VisualMeasureEvidence, ...],
        ) -> tuple[RecognitionCandidate, MeasureLocalizedResult]:
            variant = measure_localized_variant(measure_index)
            root = page_root / "measure_localized"
            crop = None
            local_xml: Path | None = None
            candidate_xml: Path | None = None
            elapsed = 0.0
            observed = 0
            note_count = 0
            local_rhythm_issue_count = 0
            error: str | None = None
            return_code = 65
            internal_results: list[MeasureLocalizedVariantResult] = []
            winning_support = 0
            winning_signature: str | None = None
            try:
                evidence = map_evidence_to_measure(
                    evidence_set,
                    measure_index - 1,
                    expected_measures,
                )
                if evidence is None:
                    raise ValueError("measure visual evidence unavailable")
                crop = create_measure_crop(normalized_page, evidence, root)
                crop_variants = create_measure_crop_variants(crop, root)
                for crop_variant in crop_variants:
                    report_progress(
                        0.88,
                        f"Checking measure {measure_index}: {crop_variant.name}",
                    )
                    local_return_code = 65
                    local_path: Path | None = None
                    local_error: str | None = None
                    local_elapsed = 0.0
                    local_observed = 0
                    local_notes = 0
                    local_rhythm = 0
                    content_signature: str | None = None
                    semantic_signature: str | None = None
                    local_valid = False
                    try:
                        engine_result = self.runner.run_page(Path(crop_variant.image_path), cancel_event)
                        local_elapsed = max(0.0, float(engine_result.elapsed_seconds))
                        elapsed += local_elapsed
                        if engine_result.cancelled:
                            raise InterruptedError("cancelled")
                        local_return_code = int(engine_result.return_code)
                        local_path = engine_result.xml_path
                        local_error = engine_result.error
                        if local_return_code == 0 and local_path is not None:
                            (
                                local_valid,
                                local_observed,
                                local_notes,
                                local_rhythm,
                                validation_error,
                            ) = validate_measure_localized_xml(local_path)
                            local_error = validation_error
                            if local_valid:
                                context_valid, context_error = validate_measure_localized_context(
                                    local_path,
                                    template_path,
                                    measure_index,
                                )
                                if not context_valid:
                                    local_valid = False
                                    local_error = context_error or "measure-localised notation context mismatch"
                            if local_valid:
                                try:
                                    semantic_signature = measure_localized_semantic_signature(local_path)
                                    content_signature = measure_localized_content_signature(local_path)
                                except Exception as exc:
                                    local_valid = False
                                    local_error = f"measure-localised content signature failed: {exc}"
                        else:
                            local_error = local_error or "measure-localised recognition failed"
                    except InterruptedError:
                        raise
                    except Exception as exc:
                        # One optional image treatment may fail independently.  Keep its
                        # abstention in diagnostics and continue; two remaining matching
                        # treatments are still sufficient for the one-family exact gate.
                        local_error = f"measure-localised subvariant failed: {exc}"
                    internal_results.append(
                        MeasureLocalizedVariantResult(
                            name=crop_variant.name,
                            image_path=crop_variant.image_path,
                            xml_path=str(local_path) if local_path is not None else None,
                            return_code=local_return_code,
                            elapsed_seconds=round(local_elapsed, 3),
                            valid=bool(local_valid),
                            observed_measure_count=local_observed,
                            note_count=local_notes,
                            local_rhythm_issue_count=local_rhythm,
                            content_signature=content_signature,
                            semantic_signature=semantic_signature,
                            error=local_error,
                        )
                    )

                representative, winning_support, winning_signature, selection_error = (
                    choose_measure_localized_variant(tuple(internal_results))
                )
                if representative is None or representative.xml_path is None:
                    error = selection_error or "measure-localised internal consensus failed"
                else:
                    local_xml = Path(representative.xml_path)
                    observed = representative.observed_measure_count
                    note_count = representative.note_count
                    local_rhythm_issue_count = representative.local_rhythm_issue_count
                    candidate_xml = root / f"measure_{measure_index:04d}_candidate.musicxml"
                    splice_measure_candidate(
                        template_path,
                        local_xml,
                        measure_index,
                        candidate_xml,
                    )
                    return_code = 0
            except InterruptedError:
                raise
            except Exception as exc:
                error = str(exc)
                return_code = 65
                candidate_xml = None

            engine_result = EngineResult(
                return_code,
                candidate_xml,
                elapsed,
                error=error,
            )
            candidate = assess_candidate(
                variant,
                Path(crop.image_path) if crop is not None else normalized_page,
                engine_result,
                expected_measures,
                None,
                layout,
            )
            candidate = inherit_sparse_page_evidence(candidate, selected)
            if candidate.valid and candidate.rhythm_issue_count > 0:
                candidate = replace(
                    candidate,
                    valid=False,
                    error="measure-localised full-page splice has rhythm issues",
                )
            candidates.append(candidate)
            result = MeasureLocalizedResult(
                measure_index=measure_index,
                variant=variant,
                crop=crop,
                xml_path=str(local_xml) if local_xml is not None else None,
                candidate_xml_path=str(candidate_xml) if candidate_xml is not None else None,
                return_code=return_code,
                elapsed_seconds=round(elapsed, 3),
                valid=bool(candidate.valid),
                observed_measure_count=observed,
                note_count=note_count,
                local_rhythm_issue_count=local_rhythm_issue_count,
                internal_variant_count=len(internal_results),
                internal_valid_count=sum(item.valid for item in internal_results),
                winning_exact_support=winning_support,
                winning_signature=winning_signature,
                internal_variants=tuple(internal_results),
                error=error,
            )
            atomic_write_json(root / f"measure_{measure_index:04d}_result.json", result.to_dict())
            if candidate.valid:
                self.log(
                    f"第 {page.index} 页：第 {measure_index} 小节局部重识别形成内部 "
                    f"{winning_support}/{len(internal_results)} 一致的独立候选"
                )
            else:
                self.log(
                    f"第 {page.index} 页：第 {measure_index} 小节局部重识别未形成内部严格多数，已弃权"
                )
            return candidate, result

        try:
            for name in ordered:
                run_named(name)

            report_progress(0.34, "Comparing initial candidates")
            ranked_now, count_resolution = rank_with_count(candidates)
            topology_best = choose_candidate(ranked_now)
            visual_evidence = rebuild_visual_evidence(
                count_resolution.selected_count,
                topology_best.physical_staff_count if topology_best is not None else 1,
            )
            ranked_now = add_visual_compatibility(ranked_now, visual_evidence)
            best_so_far = choose_candidate(ranked_now)
            page_risky = (
                (page.quality_score or 0.0) < DEFAULT_POLICY.risky_page_quality
                or (layout is not None and layout.confidence < DEFAULT_POLICY.risky_layout_confidence)
            )
            disagreement = candidate_set_is_ambiguous(ranked_now)
            if (
                page_risky
                or best_so_far is None
                or not candidate_is_strong(best_so_far, count_resolution.selected_count)
                or disagreement
            ):
                # Internal siblings such as ``staffnorm:raw_tuplets`` are one
                # preprocessing family.  Treating the selected sibling as a new
                # family would execute the expensive base candidate a second time.
                already = {
                    variant_family(candidate.variant)
                    for candidate in candidates
                }
                for name, _path in variants:
                    family = variant_family(name)
                    if family in already:
                        continue
                    run_named(name)
                    already.add(family)

            report_progress(0.69, "Selecting complete-page candidate")
            ranked_for_localized, localized_resolution = rank_with_count(candidates)
            topology_best = choose_candidate(ranked_for_localized)
            visual_evidence = rebuild_visual_evidence(
                localized_resolution.selected_count,
                topology_best.physical_staff_count if topology_best is not None else 1,
            )
            ranked_for_localized = add_visual_compatibility(ranked_for_localized, visual_evidence)
            run_localized_flag, localized_reason = should_run_localized_recognition(
                page, layout, ranked_for_localized, localized_resolution
            )
            if run_localized_flag:
                run_localized(localized_resolution.selected_count, localized_reason)
        except InterruptedError:
            return EnsembleResult(None, tuple(candidates), cancelled=True, error="任务已取消")

        report_progress(0.84, "Validating structure and measure count")
        candidates, count_resolution = rank_with_count(candidates)
        topology_best = choose_candidate(candidates)
        visual_evidence = rebuild_visual_evidence(
            count_resolution.selected_count,
            topology_best.physical_staff_count if topology_best is not None else 1,
        )
        candidates = add_visual_compatibility(candidates, visual_evidence)
        selected = choose_candidate(candidates)
        if selected is None or not selected.xml_path:
            report_progress(0.96, "No page candidate passed validation")
            return EnsembleResult(None, tuple(candidates), measure_count_resolution=count_resolution, error="所有识别候选均失败")

        page_root.mkdir(parents=True, exist_ok=True)
        report_progress(0.87, "Building measure-level consensus")
        consensus = (
            build_full_score_consensus(
                candidates,
                checkpoint,
                selected.variant,
                target_measure_count=count_resolution.selected_count,
            )
            if selected.generalized_score
            else build_measure_consensus(
                candidates,
                checkpoint,
                selected.variant,
                visual_evidence=visual_evidence,
                target_measure_count=count_resolution.selected_count,
            )
        )
        if consensus is None:
            shutil.copy2(selected.xml_path, checkpoint)
            if selected.generalized_score:
                self.log(
                    f"第 {page.index} 页：完整多谱表候选未形成可用的"
                    "独立家族共识，已保留结构最强的整页候选"
                )
        else:
            consensus_errors = validate_musicxml(checkpoint)
            try:
                consensus_analysis = analyze_musicxml(checkpoint)
            except Exception as exc:
                consensus_analysis = {"rhythm_issues": [str(exc)]}
            try:
                selected_analysis = analyze_musicxml(Path(selected.xml_path))
            except Exception as exc:
                selected_analysis = {"rhythm_issues": [str(exc)]}
            issue_regressed = any(
                _analysis_issues_regressed(
                    selected_analysis,
                    consensus_analysis,
                    key,
                )
                for key in (
                    "rhythm_issues",
                    "tie_issues",
                    "slur_issues",
                    "semantic_issues",
                )
            )
            topology_regressed = (
                int(consensus_analysis.get("part_count", 0) or 0)
                != int(selected_analysis.get("part_count", 0) or 0)
                or list(consensus_analysis.get("part_measure_counts", []) or [])
                != list(selected_analysis.get("part_measure_counts", []) or [])
            )
            if consensus_errors or issue_regressed or topology_regressed:
                # Consensus is an accuracy aid, never a reason to downgrade a valid page.
                # Fall back to the strongest complete candidate if fusion introduces a
                # parse or rhythm problem. Preserve the rejected file for diagnostics.
                rejected = checkpoint.with_name("consensus_rejected.musicxml")
                try:
                    checkpoint.replace(rejected)
                except OSError:
                    pass
                shutil.copy2(selected.xml_path, checkpoint)
                self.log(f"第 {page.index} 页：小节共识未通过结构校验，已回退到最佳整页候选")
                consensus = None
            else:
                retained_rhythm_issues = len(
                    list(consensus_analysis.get("rhythm_issues", []) or [])
                )
                if retained_rhythm_issues:
                    self.log(
                        f"第 {page.index} 页：小节共识未新增节奏问题；"
                        f"保留原候选已有的 {retained_rhythm_issues} 个节奏疑点供复核"
                    )
                # A measure crop is permitted only after ordinary consensus has already
                # identified a two-family near miss.  The local OMR result is one sparse
                # candidate and must still form the normal three-family decision.
                rescue_results: list[MeasureLocalizedResult] = []
                rescue_targets: tuple[int, ...] = ()
                rescue_accepted = False
                if (
                    not selected.generalized_score
                    and
                    layout is not None
                    and layout.confidence >= DEFAULT_POLICY.measure_localized_layout_confidence_floor
                    and visual_evidence
                ):
                    rescue_targets = eligible_measure_indices(consensus)
                if rescue_targets:
                    rescue_root = page_root / "measure_localized"
                    rescue_root.mkdir(parents=True, exist_ok=True)
                    initial_unresolved = tuple(consensus.unresolved_measure_indices)
                    before_rescue = checkpoint.with_name("consensus_before_measure_rescue.musicxml")
                    shutil.copy2(checkpoint, before_rescue)
                    try:
                        successful = 0
                        for rescue_index, measure_index in enumerate(rescue_targets, start=1):
                            candidate, result = run_measure_localized(
                                measure_index,
                                before_rescue,
                                count_resolution.selected_count,
                                visual_evidence,
                            )
                            rescue_results.append(result)
                            successful += int(candidate.valid and bool(candidate.xml_path))
                            report_progress(
                                0.89 + 0.07 * rescue_index / max(len(rescue_targets), 1),
                                f"Measure check {rescue_index}/{len(rescue_targets)} complete",
                            )
                    except InterruptedError:
                        return EnsembleResult(
                            None,
                            tuple(candidates),
                            measure_count_resolution=count_resolution,
                            cancelled=True,
                            error="任务已取消",
                        )
                    if successful:
                        rescue_path = page_root / "selected_measure_rescue.musicxml"
                        rescue_consensus = build_measure_consensus(
                            candidates,
                            rescue_path,
                            selected.variant,
                            visual_evidence=visual_evidence,
                            target_measure_count=count_resolution.selected_count,
                        )
                        rescue_errors = validate_musicxml(rescue_path) if rescue_consensus is not None else ["consensus missing"]
                        try:
                            rescue_analysis = analyze_musicxml(rescue_path) if rescue_consensus is not None else {"rhythm_issues": ["consensus missing"]}
                        except Exception as exc:
                            rescue_analysis = {"rhythm_issues": [str(exc)]}
                        old_unresolved = set(consensus.unresolved_measure_indices)
                        new_unresolved = (
                            set(rescue_consensus.unresolved_measure_indices)
                            if rescue_consensus is not None
                            else old_unresolved
                        )
                        rescue_accepted = bool(
                            rescue_consensus is not None
                            and not rescue_errors
                            and not rescue_analysis.get("rhythm_issues")
                            and rescue_consensus.measure_count == consensus.measure_count
                            and new_unresolved < old_unresolved
                        )
                        if rescue_accepted:
                            shutil.copy2(rescue_path, checkpoint)
                            consensus = rescue_consensus
                            self.log(
                                f"第 {page.index} 页：小节局部重识别减少了 "
                                f"{len(old_unresolved) - len(new_unresolved)} 个未决小节"
                            )
                        else:
                            self.log(
                                f"第 {page.index} 页：小节局部重识别未形成严格净改进，保留原共识"
                            )
                    atomic_write_json(
                        rescue_root / "measure_rescue_summary.json",
                        {
                            "format": 1,
                            "targets": list(rescue_targets),
                            "accepted": rescue_accepted,
                            "before_unresolved": list(initial_unresolved),
                            "after_unresolved": list(consensus.unresolved_measure_indices),
                            "results": [item.to_dict() for item in rescue_results],
                        },
                    )
                atomic_write_json(consensus_json, consensus.to_dict())
                self.log(
                    f"第 {page.index} 页：小节共识 {consensus.unanimous_measure_count}/{consensus.measure_count}，"
                    f"替换 {consensus.replacements} 个小节，分歧 {len(consensus.disagreement_measure_indices)} 个，"
                    f"无严格多数 {len(consensus.unresolved_measure_indices)} 个"
                )
        final_measure_count = consensus.measure_count if consensus is not None else selected.measure_count
        reconciled = reconcile_measure_count_resolution(
            count_resolution,
            final_measure_count,
            source=(
                "consensus_output"
                if consensus is not None
                else "output_candidate_fallback"
            ),
        )
        if reconciled is not count_resolution:
            count_resolution = reconciled
            atomic_write_json(count_resolution_json, count_resolution.to_dict())
            self.log(
                f"第 {page.index} 页：最终输出含 {final_measure_count} 个小节，"
                "已同步小节数审计记录"
            )
        checkpoint_store.commit(
            checkpoint_key,
            selected_variant=selected.variant,
            consensus_applied=consensus is not None,
        )
        selected = replace(
            selected,
            xml_path=str(checkpoint),
            measure_count=max(0, int(final_measure_count)),
        )
        report_progress(1.0, "Page recognition and consensus complete")
        return EnsembleResult(
            selected,
            tuple(candidates),
            consensus=consensus,
            measure_count_resolution=count_resolution,
        )
