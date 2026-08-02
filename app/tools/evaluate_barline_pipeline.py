from __future__ import annotations

"""Evaluate the complete local/geometry/sequence barline pipeline on grouped systems."""

import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from barline_training_data import RenderedBarlineSystem, render_groups  # noqa: E402
from scorescan.barline_classifier import (  # noqa: E402
    BarlineClassification,
    BarlineClassifier,
    BarlineFeatures,
)
from scorescan.layout import (  # noqa: E402
    BarlineProposalEvidence,
    StaffSystem,
    _extract_barline_proposal_evidence,
    _measure_count,
    _select_barlines_from_evidence,
)
from scorescan.linear_model import StandardizedLogisticModel  # noqa: E402
from scorescan.util import atomic_write_json  # noqa: E402

LEGACY_FEATURE_NAMES = (
    "band_width_scaled",
    "row_coverage",
    "longest_vertical_run",
    "top_endpoint_ink",
    "bottom_endpoint_ink",
    "staff_line_intersection_ratio",
    "column_peak_ratio",
    "central_density",
    "side_density",
    "side_asymmetry",
    "above_extension",
    "below_extension",
    "mid_horizontal_attachment",
    "local_vertical_dominance",
)


class LegacyClassifier:
    def __init__(self, path: Path) -> None:
        self.model = StandardizedLogisticModel.load(path, "barline_classification", LEGACY_FEATURE_NAMES)

    @property
    def enabled(self) -> bool:
        return self.model.enabled

    @property
    def model_version(self) -> str:
        return self.model.model_version

    def classify(self, features: BarlineFeatures, *, threshold: float | None = None) -> BarlineClassification:
        probability = self.model.predict(features.vector()[: len(LEGACY_FEATURE_NAMES)])
        floor = 0.68 if threshold is None else float(threshold)
        return BarlineClassification(
            probability=probability,
            accepted=probability >= floor,
            model_version=self.model.model_version,
            model_status=self.model.status,
            features=features,
        )


def _replace_probabilities(
    evidence: tuple[BarlineProposalEvidence, ...],
    classifier: LegacyClassifier,
) -> tuple[BarlineProposalEvidence, ...]:
    return tuple(
        replace(
            item,
            probability=float(classifier.classify(item.features).probability),
            model_enabled=classifier.enabled,
        )
        for item in evidence
    )


def _match_boundaries(
    rendered: RenderedBarlineSystem,
    detected: list[int],
) -> tuple[int, int, int]:
    system = rendered.system
    tolerance = max(3, int(round(system.spacing * 0.60)))
    edge_tolerance = max(10, int(round(system.spacing * 3.0)))
    truth = list(rendered.true_barlines[:-1])  # right edge is implicit in measure counting
    candidates = [
        x
        for x in detected
        if abs(x - system.left) > edge_tolerance and abs(x - system.right) > edge_tolerance
    ]
    pairs = sorted(
        (
            (abs(candidate - target), candidate_index, target_index)
            for candidate_index, candidate in enumerate(candidates)
            for target_index, target in enumerate(truth)
            if abs(candidate - target) <= tolerance
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    matched_candidates: set[int] = set()
    matched_truth: set[int] = set()
    for _distance, candidate_index, target_index in pairs:
        if candidate_index in matched_candidates or target_index in matched_truth:
            continue
        matched_candidates.add(candidate_index)
        matched_truth.add(target_index)
    return len(matched_truth), len(truth) - len(matched_truth), len(candidates) - len(matched_candidates)


def _system_result(rendered: RenderedBarlineSystem, detected: list[int]) -> dict[str, object]:
    matched, missed, false = _match_boundaries(rendered, detected)
    system = replace(rendered.system, barlines=list(detected))
    count = _measure_count(system)
    return {
        "matched": matched,
        "missed": missed,
        "false": false,
        "boundary_errors": missed + false,
        "measure_count": count,
        "measure_count_exact": count == rendered.measure_count,
        "detected": list(detected),
    }


def _metrics(results: list[dict[str, object]]) -> dict[str, float | int]:
    matched = sum(int(item["matched"]) for item in results)
    missed = sum(int(item["missed"]) for item in results)
    false = sum(int(item["false"]) for item in results)
    systems = max(len(results), 1)
    exact = sum(bool(item["measure_count_exact"]) for item in results)
    return {
        "systems": len(results),
        "exact_measure_count_systems": exact,
        "exact_measure_count_rate": exact / systems,
        "matched_true_boundaries": matched,
        "missed_true_boundaries": missed,
        "true_boundary_recall": matched / max(matched + missed, 1),
        "false_boundaries": false,
        "false_boundaries_per_system": false / systems,
        "boundary_precision": matched / max(matched + false, 1),
        "total_boundary_errors": missed + false,
    }


def _compare(
    baseline: list[dict[str, object]],
    candidate: list[dict[str, object]],
) -> dict[str, int]:
    fewer = equal = more = helped = harmed = both_exact = 0
    net = 0
    for old, new in zip(baseline, candidate, strict=True):
        old_errors = int(old["boundary_errors"])
        new_errors = int(new["boundary_errors"])
        fewer += int(new_errors < old_errors)
        equal += int(new_errors == old_errors)
        more += int(new_errors > old_errors)
        net += new_errors - old_errors
        old_exact = bool(old["measure_count_exact"])
        new_exact = bool(new["measure_count_exact"])
        helped += int(not old_exact and new_exact)
        harmed += int(old_exact and not new_exact)
        both_exact += int(old_exact and new_exact)
    return {
        "systems_with_fewer_boundary_errors": fewer,
        "systems_with_equal_boundary_errors": equal,
        "systems_with_more_boundary_errors": more,
        "net_boundary_error_change": net,
        "helped_measure_count_systems": helped,
        "harmed_measure_count_systems": harmed,
        "both_measure_counts_exact": both_exact,
    }


def _prepare(
    systems: list[RenderedBarlineSystem],
    new_classifier: BarlineClassifier,
    legacy_classifier: LegacyClassifier,
) -> list[tuple[RenderedBarlineSystem, tuple[BarlineProposalEvidence, ...], tuple[BarlineProposalEvidence, ...]]]:
    prepared = []
    for rendered in systems:
        candidate = _extract_barline_proposal_evidence(
            rendered.binary,
            rendered.system,
            classifier=new_classifier,
        )
        baseline = _replace_probabilities(candidate, legacy_classifier)
        prepared.append((rendered, candidate, baseline))
    return prepared


def _run(
    prepared: list[tuple[RenderedBarlineSystem, tuple[BarlineProposalEvidence, ...], tuple[BarlineProposalEvidence, ...]]],
    *,
    threshold: float,
    baseline: bool,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for rendered, candidate_evidence, baseline_evidence in prepared:
        evidence = baseline_evidence if baseline else candidate_evidence
        barlines, _local, _sequence = _select_barlines_from_evidence(
            rendered.system,
            evidence,
            probability_floor=0.68 if baseline else threshold,
        )
        results.append(_system_result(rendered, barlines))
    return results


def _proposal_recall(systems: list[RenderedBarlineSystem], prepared: list[tuple[RenderedBarlineSystem, tuple[BarlineProposalEvidence, ...], tuple[BarlineProposalEvidence, ...]]]) -> float:
    found = total = 0
    for rendered, evidence, _baseline in prepared:
        tolerance = max(2, int(round(rendered.system.spacing * 0.52)))
        for truth in rendered.true_barlines[:-1]:
            total += 1
            found += int(any(abs(item.x - truth) <= tolerance for item in evidence))
    return found / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=ROOT.parent / "training" / "baselines" / "barline_classifier_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent / "training" / "barline_pipeline_report_v2.json",
    )
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--groups", type=int, default=400)
    args = parser.parse_args()

    new_classifier = BarlineClassifier(args.model)
    legacy_classifier = LegacyClassifier(args.baseline_model)
    if not new_classifier.enabled or not legacy_classifier.enabled:
        raise RuntimeError("barline model could not be loaded")

    calibration_systems = render_groups(args.seed, args.groups, 1)
    test_systems = render_groups(args.seed + 1, args.groups, 1)
    calibration = _prepare(calibration_systems, new_classifier, legacy_classifier)
    frozen = _prepare(test_systems, new_classifier, legacy_classifier)
    baseline_calibration = _run(calibration, threshold=0.68, baseline=True)
    baseline_frozen = _run(frozen, threshold=0.68, baseline=True)

    threshold_rows: list[dict[str, object]] = []
    selected: tuple[tuple[float, float, int, float], float] | None = None
    for threshold in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        candidate = _run(calibration, threshold=threshold, baseline=False)
        metrics = _metrics(candidate)
        comparison = _compare(baseline_calibration, candidate)
        row = {"threshold": threshold, "metrics": metrics, "comparison_to_baseline": comparison}
        threshold_rows.append(row)
        # Prefer exact measure counts, then fewer regressions, then boundary recall and
        # precision. Threshold choice uses calibration systems only.
        key = (
            float(metrics["exact_measure_count_rate"]),
            -float(comparison["systems_with_more_boundary_errors"]),
            float(metrics["true_boundary_recall"]),
            float(metrics["boundary_precision"]),
        )
        if selected is None or key > selected[0]:
            selected = (key, threshold)
    assert selected is not None
    selected_threshold = selected[1]
    candidate_frozen = _run(frozen, threshold=selected_threshold, baseline=False)

    regressions: list[dict[str, object]] = []
    for rendered, old, new in zip(test_systems, baseline_frozen, candidate_frozen, strict=True):
        if int(new["boundary_errors"]) > int(old["boundary_errors"]):
            regressions.append(
                {
                    "group": rendered.group,
                    "variant": rendered.variant,
                    "expected_measure_count": rendered.measure_count,
                    "baseline": old,
                    "candidate": new,
                }
            )

    report = {
        "evaluation": "grouped rendered-system local-barline pipeline benchmark",
        "seed": args.seed,
        "groups": args.groups,
        "split_unit": "rendered staff-system identity; calibration and frozen test use independent groups",
        "proposal_recall": {
            "calibration": _proposal_recall(calibration_systems, calibration),
            "frozen_test": _proposal_recall(test_systems, frozen),
        },
        "baseline": {
            "model_version": legacy_classifier.model_version,
            "threshold": 0.68,
            "calibration": _metrics(baseline_calibration),
            "frozen_test": _metrics(baseline_frozen),
        },
        "candidate": {
            "model_version": new_classifier.model_version,
            "selected_threshold": selected_threshold,
            "frozen_test": _metrics(candidate_frozen),
            "comparison_to_baseline": _compare(baseline_frozen, candidate_frozen),
            "boundary_error_regressions": regressions,
        },
        "threshold_calibration": threshold_rows,
        "scope": "synthetic rendered staff-system layout accuracy; not note recognition or end-to-end OMR accuracy",
        "limitations": [
            "No large frozen real-scan boundary proposal set is bundled.",
            "The benchmark evaluates deskewed single-staff printed systems.",
        ],
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
