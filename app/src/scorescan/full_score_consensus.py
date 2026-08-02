from __future__ import annotations

"""Fail-closed whole-score consensus for piano and ensemble pages.

The established event patcher is deliberately specialised for one physical staff.
Applying it to only the first part of a full score can discard instruments or break
cross-staff relationships.  This module instead votes on one simultaneous measure
transaction: every part/staff at a measure index must come from the same independently
supported candidate family.
"""

import copy
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from lxml import etree

from .consensus import ConsensusReport, MeasureVote
from .musicxml import MUSICXML_DOCTYPE
from .musicxml_signature import measure_preservation_signatures
from .policy import DEFAULT_POLICY
from .util import atomic_write_bytes
from .variant_family import variant_family


class FullScoreCandidate(Protocol):
    variant: str
    xml_path: str | None
    score: float
    valid: bool


@dataclass(frozen=True)
class _CandidateScore:
    candidate: FullScoreCandidate
    tree: etree._ElementTree
    parts: tuple[etree._Element, ...]
    measures: tuple[tuple[etree._Element, ...], ...]
    signatures: tuple[str, ...]
    division_states: tuple[tuple[tuple[int | None, int | None], ...], ...]
    topology: tuple[tuple[int, int], ...]


def _part_staff_count(part: etree._Element) -> int:
    declared = [
        int(text)
        for text in part.xpath("./measure/attributes/staves/text()")
        if str(text).strip().isdigit()
    ]
    observed = [
        int(text)
        for text in part.xpath("./measure/note/staff/text()")
        if str(text).strip().isdigit()
    ]
    return max([1, *declared, *observed])


def _division_states(
    measures: tuple[etree._Element, ...],
) -> tuple[tuple[int | None, int | None], ...]:
    current: int | None = None
    result: list[tuple[int | None, int | None]] = []
    for measure in measures:
        before = current
        for text in measure.xpath("./attributes/divisions/text()"):
            try:
                parsed = int(str(text).strip())
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                current = parsed
        result.append((before, current))
    return tuple(result)


def _parse_candidate(candidate: FullScoreCandidate) -> _CandidateScore | None:
    if not candidate.valid or not candidate.xml_path:
        return None
    path = Path(candidate.xml_path)
    if not path.is_file():
        return None
    try:
        tree = etree.parse(
            str(path),
            etree.XMLParser(remove_blank_text=True, resolve_entities=False, no_network=True),
        )
    except (OSError, etree.XMLSyntaxError):
        return None
    parts = tuple(tree.getroot().findall("part"))
    if len(parts) < 1:
        return None
    measures = tuple(tuple(part.findall("measure")) for part in parts)
    counts = {len(items) for items in measures}
    if len(counts) != 1 or not counts or next(iter(counts)) < 1:
        return None
    topology = tuple(
        (_part_staff_count(part), len(part_measures))
        for part, part_measures in zip(parts, measures)
    )
    per_part_signatures = tuple(
        measure_preservation_signatures(part_measures)
        for part_measures in measures
    )
    measure_count = len(measures[0])
    combined = tuple(
        "|".join(part_signatures[index] for part_signatures in per_part_signatures)
        for index in range(measure_count)
    )
    return _CandidateScore(
        candidate=candidate,
        tree=tree,
        parts=parts,
        measures=measures,
        signatures=combined,
        division_states=tuple(_division_states(items) for items in measures),
        topology=topology,
    )


def _replacement_measure(
    source: etree._Element,
    template: etree._Element,
) -> etree._Element:
    replacement = copy.deepcopy(source)
    for print_element in replacement.findall("print"):
        replacement.remove(print_element)
    template_prints = [copy.deepcopy(item) for item in template.findall("print")]
    for index, print_element in enumerate(template_prints):
        replacement.insert(index, print_element)
    # Page/system layout and public measure identity remain owned by the strongest
    # whole-page template.  Performed content and musical attributes come from the
    # exact-family winner.
    replacement.attrib.clear()
    replacement.attrib.update(template.attrib)
    return replacement


def build_full_score_consensus(
    candidates: list[FullScoreCandidate],
    output_path: Path,
    template_variant: str,
    *,
    target_measure_count: int | None = None,
) -> ConsensusReport | None:
    parsed_by_variant = {
        item.candidate.variant: item
        for candidate in candidates
        if (item := _parse_candidate(candidate)) is not None
    }
    template = parsed_by_variant.get(template_variant)
    if template is None:
        return None
    measure_count = len(template.signatures)
    if target_measure_count and int(target_measure_count) != measure_count:
        return None

    all_families: dict[str, list[FullScoreCandidate]] = {}
    for candidate in candidates:
        all_families.setdefault(variant_family(candidate.variant), []).append(candidate)

    family_members: dict[str, tuple[_CandidateScore, ...]] = {}
    candidate_alignment: dict[str, dict[str, object]] = {}
    for family, members in all_families.items():
        parsed_members: list[_CandidateScore] = []
        complete = True
        for member in members:
            parsed = parsed_by_variant.get(member.variant)
            topology_ok = parsed is not None and parsed.topology == template.topology
            candidate_alignment[member.variant] = {
                "family": family,
                "valid": bool(member.valid),
                "topology_compatible": topology_ok,
                "part_staff_measure_topology": (
                    [list(item) for item in parsed.topology]
                    if parsed is not None
                    else None
                ),
            }
            if not topology_ok:
                complete = False
            elif parsed is not None:
                parsed_members.append(parsed)
        if complete and parsed_members:
            family_members[family] = tuple(parsed_members)

    output_tree = copy.deepcopy(template.tree)
    output_parts = tuple(output_tree.getroot().findall("part"))
    output_measures = tuple(tuple(part.findall("measure")) for part in output_parts)
    votes: list[MeasureVote] = []
    disagreements: list[int] = []
    unresolved: list[int] = []
    resolved_disagreements: list[int] = []
    confidences: list[float] = []
    replacements = 0
    unanimous_count = 0
    majority_count = 0

    for measure_index in range(measure_count):
        signature_families: dict[str, list[str]] = {}
        representatives: dict[str, _CandidateScore] = {}
        abstaining: list[str] = []
        for family, members in family_members.items():
            member_signatures = {member.signatures[measure_index] for member in members}
            if len(member_signatures) != 1:
                abstaining.append(family)
                continue
            signature = next(iter(member_signatures))
            signature_families.setdefault(signature, []).append(family)
            representatives.setdefault(
                signature,
                max(members, key=lambda item: float(item.candidate.score)),
            )

        eligible_family_count = sum(len(items) for items in signature_families.values())
        counts = Counter(
            {
                signature: len(families)
                for signature, families in signature_families.items()
            }
        )
        winning_signature, winning_support = (
            counts.most_common(1)[0] if counts else ("", 0)
        )
        confidence = (
            winning_support / eligible_family_count
            if eligible_family_count
            else 0.0
        )
        confidences.append(confidence)
        is_disagreement = len(signature_families) > 1
        if is_disagreement:
            disagreements.append(measure_index)
        unanimous = (
            eligible_family_count >= DEFAULT_POLICY.minimum_consensus_families
            and len(signature_families) == 1
        )
        strict_majority = (
            eligible_family_count >= DEFAULT_POLICY.minimum_consensus_families
            and winning_support >= DEFAULT_POLICY.minimum_consensus_families
            and winning_support * 2 > eligible_family_count
        )
        if unanimous:
            unanimous_count += 1
        if strict_majority:
            majority_count += 1
            if is_disagreement:
                resolved_disagreements.append(measure_index)
        else:
            unresolved.append(measure_index)

        replaced_template = False
        decision = "kept-template-no-strict-full-score-majority"
        template_signature = template.signatures[measure_index]
        winner = representatives.get(winning_signature)
        if strict_majority and winner is not None:
            if winning_signature == template_signature:
                decision = "kept-template-exact-family-majority"
            else:
                division_compatible = all(
                    winner.division_states[part_index][measure_index]
                    == template.division_states[part_index][measure_index]
                    for part_index in range(len(template.parts))
                )
                if division_compatible:
                    for part_index in range(len(output_parts)):
                        old_measure = output_measures[part_index][measure_index]
                        new_measure = _replacement_measure(
                            winner.measures[part_index][measure_index],
                            old_measure,
                        )
                        output_parts[part_index].replace(old_measure, new_measure)
                    replacements += 1
                    replaced_template = True
                    decision = "replaced-simultaneous-full-score-measure"
                else:
                    strict_majority = False
                    majority_count -= 1
                    if measure_index in resolved_disagreements:
                        resolved_disagreements.remove(measure_index)
                    if measure_index not in unresolved:
                        unresolved.append(measure_index)
                    decision = "abstained-incompatible-divisions-state"

        votes.append(
            MeasureVote(
                measure_index=measure_index,
                selected_variant=(
                    winner.candidate.variant
                    if strict_majority and winner is not None
                    else template_variant
                ),
                selected_support=winning_support,
                eligible_candidates=eligible_family_count,
                unanimous=unanimous,
                strict_majority=strict_majority,
                replaced_template=replaced_template,
                decision=decision,
                exact_family_support=winning_support,
                semantic_family_support=winning_support,
                selected_preservation_family_support=len(
                    signature_families.get(template_signature, ())
                ),
                preservation_gate_required=True,
                preservation_gate_accepted=strict_majority,
                eligible_family_count=eligible_family_count,
                abstaining_family_count=len(abstaining),
                abstaining_families=tuple(sorted(abstaining)),
                signatures={
                    signature: sorted(families)
                    for signature, families in signature_families.items()
                },
                semantic_support_ratio=confidence,
                semantic_confidence=confidence,
            )
        )

    atomic_write_bytes(
        output_path,
        etree.tostring(
            output_tree,
            encoding="UTF-8",
            xml_declaration=True,
            pretty_print=True,
            doctype=MUSICXML_DOCTYPE,
        ),
    )
    average_confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else 0.0
    )
    exact_ratio = unanimous_count / measure_count
    semantic_ratio = majority_count / measure_count
    return ConsensusReport(
        output_path=str(output_path),
        template_variant=template_variant,
        candidate_count=len(candidates),
        eligible_candidate_count=sum(
            len(members) for members in family_members.values()
        ),
        measure_count=measure_count,
        agreement_ratio=round(average_confidence, 6),
        unanimous_measure_count=unanimous_count,
        majority_measure_count=majority_count,
        disagreement_measure_indices=tuple(sorted(disagreements)),
        unresolved_measure_indices=tuple(sorted(unresolved)),
        replacements=replacements,
        votes=tuple(votes),
        exact_agreement_ratio=round(exact_ratio, 6),
        semantic_agreement_ratio=round(semantic_ratio, 6),
        mean_measure_confidence=round(average_confidence, 6),
        preservation_disagreement_measure_indices=tuple(sorted(disagreements)),
        resolved_disagreement_measure_indices=tuple(sorted(resolved_disagreements)),
        candidate_alignment=candidate_alignment,
        mean_selected_measure_probability=round(average_confidence, 6),
        measure_calibration_model="full-score-exact-family-consensus-v1",
        requested_measure_count=int(target_measure_count or measure_count),
        template_measure_count=measure_count,
        template_count_family_support=len(family_members),
        template_count_eligible_family_count=len(family_members),
        mean_ensemble_probability=round(average_confidence, 6),
        ensemble_calibration_model="full-score-exact-family-consensus-v1",
        mean_selection_risk_probability=round(average_confidence, 6),
        selection_risk_model="full-score-exact-family-consensus-v1",
    )
