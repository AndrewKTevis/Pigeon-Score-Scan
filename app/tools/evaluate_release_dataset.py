from __future__ import annotations

"""Evaluate a frozen set of reference/candidate MusicXML pairs.

The manifest contains only file pairs, split labels, and optional release gates.  It is
suitable both for outputs produced by a full OMR run and for candidate-fusion ablations.
All aggregation uses additive counts; bootstrap intervals resample complete scores so
pages from one score cannot leak across uncertainty estimates.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence


PRODUCTION_SCORE_CONFIGURATIONS = (
    "solo_monophonic",
    "piano",
    "monophonic_ensemble",
    "piano_plus_monophonic_ensemble",
)
PRODUCTION_BOUNDARY_CONTRACT_VERSION = (
    "printed-western-instrumental-scan-boundary@4"
)
PRODUCTION_SCAN_PAGE_SHAPE_CONTRACT = (
    "ordinary-single-page-or-two-page-spread-aspect-ratio@1"
)
PRODUCTION_MAXIMUM_SCAN_PAGE_ASPECT_RATIO = 3.0
PRODUCTION_EVIDENCE_FILE_ROLES = (
    "annotation_adjudication_audit",
    "frozen_benchmark_registration",
    "license_authorization_audit",
    "page_provenance_audit",
    "train_tune_isolation_audit",
)
DIRECTION_METRIC_PREFIXES = (
    "words_direction",
    "dynamic_direction",
    "wedge_direction",
    "crescendo_wedge_start",
    "diminuendo_wedge_start",
    "wedge_stop",
)
PRODUCTION_REFERENCE_FEATURE_MINIMUM_COUNTS = {
    "reference_accidental_marker_count": 1_000,
    "reference_tuplet_event_count": 1_000,
    "reference_tie_endpoint_count": 1_000,
    "reference_slur_endpoint_count": 2_000,
    "reference_beam_marker_count": 10_000,
    "reference_articulation_marker_count": 1_000,
    "reference_ornament_marker_count": 200,
    "reference_grace_event_count": 100,
    "reference_repeat_marker_count": 100,
    "reference_direction_count": 5_000,
    "reference_words_direction_count": 1_000,
    "reference_dynamic_direction_count": 1_000,
    "reference_wedge_direction_count": 500,
    "reference_crescendo_wedge_start_count": 200,
    "reference_diminuendo_wedge_start_count": 200,
    "reference_wedge_stop_count": 400,
}
PRODUCTION_CONFIGURATION_REFERENCE_FEATURE_MINIMUM_COUNTS = {
    "reference_accidental_marker_count": 100,
    "reference_tuplet_event_count": 500,
    "reference_tie_endpoint_count": 500,
    "reference_slur_endpoint_count": 1_000,
    "reference_beam_marker_count": 5_000,
    "reference_articulation_marker_count": 500,
    "reference_ornament_marker_count": 100,
    "reference_grace_event_count": 100,
    "reference_repeat_marker_count": 40,
    "reference_direction_count": 500,
    "reference_words_direction_count": 300,
    "reference_dynamic_direction_count": 500,
    "reference_wedge_direction_count": 200,
    "reference_crescendo_wedge_start_count": 50,
    "reference_diminuendo_wedge_start_count": 50,
    "reference_wedge_stop_count": 100,
}
# Aggregate scores are not sufficient for a public release: a strong result on
# uncomplicated solo pages must never hide errors on piano or mixed ensembles.
# These gates are deliberately score-configuration local.  Coverage gates below
# independently require at least 400 submitted pages in every configuration.
PRODUCTION_CONFIGURATION_MINIMUM_METRICS = {
    "reference_event_count": 5_000,
    "pitch_accuracy_aligned": 0.997,
    "rhythm_accuracy_aligned": 0.997,
    "event_kind_accuracy_aligned": 0.997,
    "chord_topology_accuracy_aligned": 0.997,
    "tuplet_event_f1": 0.995,
    "tie_endpoint_f1": 0.995,
    "slur_endpoint_f1": 0.995,
    "beam_marker_f1": 0.995,
    "articulation_marker_f1": 0.995,
    "ornament_marker_f1": 0.995,
    "accidental_marker_f1": 0.995,
    "time_signature_accuracy": 0.997,
    "key_signature_accuracy": 0.997,
    "clef_accuracy": 0.997,
    "barline_accuracy": 0.997,
    "repeat_marker_f1": 0.995,
    "direction_content_f1": 0.995,
    "direction_anchor_accuracy": 0.997,
    "words_direction_content_f1": 0.995,
    "words_direction_anchor_accuracy": 0.997,
    "dynamic_direction_content_f1": 0.995,
    "dynamic_direction_anchor_accuracy": 0.997,
    "wedge_direction_content_f1": 0.995,
    "wedge_direction_anchor_accuracy": 0.997,
    "crescendo_wedge_start_content_f1": 0.995,
    "crescendo_wedge_start_anchor_accuracy": 0.997,
    "diminuendo_wedge_start_content_f1": 0.995,
    "diminuendo_wedge_start_anchor_accuracy": 0.997,
    "wedge_stop_content_f1": 0.995,
    "wedge_stop_anchor_accuracy": 0.997,
    "expression_marker_f1": 0.995,
    "exact_measure_rate": 0.995,
    "preservation_exact_measure_rate": 0.997,
    "pitch_accuracy_aligned_low_95": 0.990,
    "rhythm_accuracy_aligned_low_95": 0.990,
    "tie_endpoint_f1_low_95": 0.990,
    "slur_endpoint_f1_low_95": 0.990,
    "direction_content_f1_low_95": 0.990,
    "words_direction_content_f1_low_95": 0.990,
    "dynamic_direction_content_f1_low_95": 0.990,
    "wedge_direction_content_f1_low_95": 0.990,
    "crescendo_wedge_start_content_f1_low_95": 0.990,
    "diminuendo_wedge_start_content_f1_low_95": 0.990,
    "wedge_stop_content_f1_low_95": 0.990,
    "exact_measure_rate_low_95": 0.985,
    "preservation_exact_measure_rate_low_95": 0.990,
    **PRODUCTION_CONFIGURATION_REFERENCE_FEATURE_MINIMUM_COUNTS,
}
PRODUCTION_CONFIGURATION_MAXIMUM_METRICS = {
    "event_error_rate": 0.010,
    "deleted_event_rate": 0.003,
    "inserted_event_rate": 0.003,
}
SCORE_CONFIGURATION_BY_SHAPE = {
    "single_staff_solo": "solo_monophonic",
    "keyboard": "piano",
    "single_staff_ensemble": "monophonic_ensemble",
    "keyboard_plus_single_staff_ensemble": (
        "piano_plus_monophonic_ensemble"
    ),
}


STABLE_RELEASE_GATES: dict[str, dict[str, float]] = {
    "minimum": {
        "case_count": 100,
        "reference_event_count": 5_000,
        "pitch_accuracy_aligned": 0.97,
        "rhythm_accuracy_aligned": 0.95,
        "event_kind_accuracy_aligned": 0.985,
        "chord_topology_accuracy_aligned": 0.985,
        "tuplet_topology_accuracy_aligned": 0.99,
        "tuplet_event_precision": 0.97,
        "tuplet_event_recall": 0.97,
        "tuplet_event_f1": 0.97,
        "tie_topology_accuracy_aligned": 0.99,
        "tie_endpoint_precision": 0.97,
        "tie_endpoint_recall": 0.97,
        "tie_endpoint_f1": 0.97,
        "slur_topology_accuracy": 0.99,
        "slur_endpoint_precision": 0.97,
        "slur_endpoint_recall": 0.97,
        "slur_endpoint_f1": 0.97,
        "beam_topology_accuracy_aligned": 0.985,
        "beam_marker_precision": 0.97,
        "beam_marker_recall": 0.97,
        "beam_marker_f1": 0.97,
        "articulation_topology_accuracy_aligned": 0.985,
        "articulation_marker_precision": 0.97,
        "articulation_marker_recall": 0.97,
        "articulation_marker_f1": 0.97,
        "ornament_topology_accuracy_aligned": 0.985,
        "ornament_marker_precision": 0.97,
        "ornament_marker_recall": 0.97,
        "ornament_marker_f1": 0.97,
        "accidental_marker_precision": 0.97,
        "accidental_marker_recall": 0.97,
        "accidental_marker_f1": 0.97,
        "grace_topology_accuracy_aligned": 0.99,
        "grace_event_precision": 0.97,
        "grace_event_recall": 0.97,
        "grace_event_f1": 0.97,
        "lyric_topology_accuracy_aligned": 0.985,
        "lyric_event_precision": 0.97,
        "lyric_event_recall": 0.97,
        "lyric_event_f1": 0.97,
        "cross_tie_boundary_accuracy": 0.99,
        "cross_tie_precision": 0.97,
        "cross_tie_recall": 0.97,
        "cross_tie_f1": 0.97,
        "event_presence_precision": 0.97,
        "event_presence_recall": 0.97,
        "event_presence_f1": 0.97,
        "time_signature_accuracy": 0.99,
        "key_signature_accuracy": 0.99,
        "clef_accuracy": 0.99,
        "barline_accuracy": 0.99,
        "repeat_marker_precision": 0.97,
        "repeat_marker_recall": 0.97,
        "repeat_marker_f1": 0.97,
        "direction_precision": 0.97,
        "direction_recall": 0.97,
        "direction_f1": 0.97,
        "direction_content_f1": 0.98,
        "direction_anchor_accuracy": 0.985,
        "words_direction_f1": 0.97,
        "words_direction_content_f1": 0.98,
        "words_direction_anchor_accuracy": 0.985,
        "dynamic_direction_f1": 0.97,
        "dynamic_direction_content_f1": 0.98,
        "dynamic_direction_anchor_accuracy": 0.985,
        "wedge_direction_f1": 0.97,
        "wedge_direction_content_f1": 0.98,
        "wedge_direction_anchor_accuracy": 0.985,
        "crescendo_wedge_start_f1": 0.97,
        "crescendo_wedge_start_content_f1": 0.98,
        "crescendo_wedge_start_anchor_accuracy": 0.985,
        "diminuendo_wedge_start_f1": 0.97,
        "diminuendo_wedge_start_content_f1": 0.98,
        "diminuendo_wedge_start_anchor_accuracy": 0.985,
        "wedge_stop_f1": 0.97,
        "wedge_stop_content_f1": 0.98,
        "wedge_stop_anchor_accuracy": 0.985,
        "expression_marker_precision": 0.97,
        "expression_marker_recall": 0.97,
        "expression_marker_f1": 0.97,
        "exact_measure_rate": 0.75,
        "preservation_exact_measure_rate": 0.75,
        "pitch_accuracy_low_95": 0.95,
        "rhythm_accuracy_low_95": 0.92,
        "event_kind_accuracy_low_95": 0.97,
        "chord_topology_accuracy_low_95": 0.97,
        "tuplet_event_precision_low_95": 0.95,
        "tuplet_event_recall_low_95": 0.95,
        "tuplet_event_f1_low_95": 0.95,
        "tie_endpoint_precision_low_95": 0.95,
        "tie_endpoint_recall_low_95": 0.95,
        "tie_endpoint_f1_low_95": 0.95,
        "slur_endpoint_precision_low_95": 0.95,
        "slur_endpoint_recall_low_95": 0.95,
        "slur_endpoint_f1_low_95": 0.95,
        "beam_topology_accuracy_aligned_low_95": 0.97,
        "beam_marker_precision_low_95": 0.95,
        "beam_marker_recall_low_95": 0.95,
        "beam_marker_f1_low_95": 0.95,
        "articulation_marker_precision_low_95": 0.95,
        "articulation_marker_recall_low_95": 0.95,
        "articulation_marker_f1_low_95": 0.95,
        "ornament_marker_precision_low_95": 0.95,
        "ornament_marker_recall_low_95": 0.95,
        "ornament_marker_f1_low_95": 0.95,
        "accidental_marker_precision_low_95": 0.95,
        "accidental_marker_recall_low_95": 0.95,
        "accidental_marker_f1_low_95": 0.95,
        "grace_topology_accuracy_aligned_low_95": 0.97,
        "grace_event_precision_low_95": 0.95,
        "grace_event_recall_low_95": 0.95,
        "grace_event_f1_low_95": 0.95,
        "lyric_topology_accuracy_aligned_low_95": 0.97,
        "lyric_event_precision_low_95": 0.95,
        "lyric_event_recall_low_95": 0.95,
        "lyric_event_f1_low_95": 0.95,
        "cross_tie_precision_low_95": 0.95,
        "cross_tie_recall_low_95": 0.95,
        "cross_tie_f1_low_95": 0.95,
        "event_presence_precision_low_95": 0.95,
        "event_presence_recall_low_95": 0.95,
        "event_presence_f1_low_95": 0.95,
        "exact_measure_rate_low_95": 0.65,
        "preservation_exact_measure_rate_low_95": 0.65,
        "time_signature_accuracy_low_95": 0.97,
        "key_signature_accuracy_low_95": 0.97,
        "clef_accuracy_low_95": 0.97,
        "barline_accuracy_low_95": 0.97,
        "repeat_marker_precision_low_95": 0.95,
        "repeat_marker_recall_low_95": 0.95,
        "repeat_marker_f1_low_95": 0.95,
        "direction_precision_low_95": 0.95,
        "direction_recall_low_95": 0.95,
        "direction_f1_low_95": 0.95,
        "direction_content_f1_low_95": 0.96,
        "direction_anchor_accuracy_low_95": 0.97,
        "words_direction_f1_low_95": 0.95,
        "words_direction_content_f1_low_95": 0.96,
        "words_direction_anchor_accuracy_low_95": 0.97,
        "dynamic_direction_f1_low_95": 0.95,
        "dynamic_direction_content_f1_low_95": 0.96,
        "dynamic_direction_anchor_accuracy_low_95": 0.97,
        "wedge_direction_f1_low_95": 0.95,
        "wedge_direction_content_f1_low_95": 0.96,
        "wedge_direction_anchor_accuracy_low_95": 0.97,
        "crescendo_wedge_start_f1_low_95": 0.95,
        "crescendo_wedge_start_content_f1_low_95": 0.96,
        "crescendo_wedge_start_anchor_accuracy_low_95": 0.97,
        "diminuendo_wedge_start_f1_low_95": 0.95,
        "diminuendo_wedge_start_content_f1_low_95": 0.96,
        "diminuendo_wedge_start_anchor_accuracy_low_95": 0.97,
        "wedge_stop_f1_low_95": 0.95,
        "wedge_stop_content_f1_low_95": 0.96,
        "wedge_stop_anchor_accuracy_low_95": 0.97,
        "expression_marker_precision_low_95": 0.95,
        "expression_marker_recall_low_95": 0.95,
        "expression_marker_f1_low_95": 0.95,
    },
    "maximum": {
        "event_error_rate": 0.10,
        "deleted_event_rate": 0.035,
        "inserted_event_rate": 0.035,
    },
}


def _production_release_gates_v2() -> dict[str, dict[str, float]]:
    """Build the near-correction-free product gate without weakening stable-v1."""

    minimum = dict(STABLE_RELEASE_GATES["minimum"])
    # stable-v1 predates the frozen 0.37 product boundary and retains lyric
    # metrics for historical/diagnostic compatibility.  production-v2 accepts
    # only lyric-free pages and must not silently turn lyric transcription into
    # a release promise.  Reports may still expose lyric diagnostics, but they
    # are not production acceptance criteria.
    for metric in tuple(minimum):
        if metric.startswith("lyric_"):
            minimum.pop(metric)
    minimum.update(
        {
            # The product accepts multi-page PDFs, so one frozen case is one
            # submitted document and carries an explicit scan-page count.
            # Source groups bind all documents and derivatives of one work.
            "case_count": 200,
            "source_group_count": 200,
            "submitted_scan_page_count": 2_000,
            "verified_unique_scan_page_count": 2_000,
            "scope_classified_page_count": 2_000,
            "solo_monophonic_page_count": 400,
            "piano_page_count": 400,
            "monophonic_ensemble_page_count": 400,
            "piano_plus_monophonic_ensemble_page_count": 400,
            # Accuracy numbers cannot turn rendered pages, unlicensed material,
            # incomplete labels, or a tuned-on benchmark into release evidence.
            # These metrics are supplied only by validate_production_evidence().
            "production_boundary_contract_evidence": 1,
            "physical_scan_origin_evidence": 1,
            "page_identity_audit_evidence": 1,
            "evaluation_use_authorized_evidence": 1,
            "complete_page_semantics_evidence": 1,
            "instrumental_text_boundary_evidence": 1,
            "double_annotation_evidence": 1,
            "work_isolation_evidence": 1,
            "frozen_before_candidate_evidence": 1,
            "submitted_orientation_preserved_evidence": 1,
            "scan_page_shape_contract_evidence": 1,
            "scan_page_aspect_limit_evidence": 1,
            "ordinary_scan_page_shape_audit_evidence": 1,
            "required_evidence_file_role_count": len(
                PRODUCTION_EVIDENCE_FILE_ROLES
            ),
            "reference_event_count": 50_000,
            "pitch_accuracy_aligned": 0.999,
            "rhythm_accuracy_aligned": 0.999,
            "event_kind_accuracy_aligned": 0.999,
            "chord_topology_accuracy_aligned": 0.999,
            "tuplet_topology_accuracy_aligned": 0.998,
            "tie_topology_accuracy_aligned": 0.998,
            "slur_topology_accuracy": 0.998,
            "beam_topology_accuracy_aligned": 0.998,
            "articulation_topology_accuracy_aligned": 0.998,
            "ornament_topology_accuracy_aligned": 0.998,
            "grace_topology_accuracy_aligned": 0.998,
            "cross_tie_boundary_accuracy": 0.998,
            "time_signature_accuracy": 0.999,
            "key_signature_accuracy": 0.999,
            "clef_accuracy": 0.999,
            "barline_accuracy": 0.999,
            "direction_content_f1": 0.997,
            "direction_anchor_accuracy": 0.998,
            "words_direction_content_f1": 0.997,
            "words_direction_anchor_accuracy": 0.998,
            "dynamic_direction_content_f1": 0.997,
            "dynamic_direction_anchor_accuracy": 0.998,
            "wedge_direction_content_f1": 0.997,
            "wedge_direction_anchor_accuracy": 0.998,
            "crescendo_wedge_start_content_f1": 0.997,
            "crescendo_wedge_start_anchor_accuracy": 0.998,
            "diminuendo_wedge_start_content_f1": 0.997,
            "diminuendo_wedge_start_anchor_accuracy": 0.998,
            "wedge_stop_content_f1": 0.997,
            "wedge_stop_anchor_accuracy": 0.998,
            "exact_measure_rate": 0.995,
            "preservation_exact_measure_rate": 0.998,
            **PRODUCTION_REFERENCE_FEATURE_MINIMUM_COUNTS,
        }
    )
    for key in tuple(minimum):
        if key.endswith("_low_95"):
            if key in {
                "pitch_accuracy_low_95",
                "rhythm_accuracy_low_95",
                "event_kind_accuracy_low_95",
                "chord_topology_accuracy_low_95",
            }:
                minimum[key] = 0.997
            elif key in {
                "exact_measure_rate_low_95",
                "preservation_exact_measure_rate_low_95",
            }:
                minimum[key] = 0.99
            elif key.endswith("_anchor_accuracy_low_95"):
                minimum[key] = 0.995
            else:
                minimum[key] = 0.995
        elif key.endswith(("_precision", "_recall", "_f1")):
            minimum[key] = max(float(minimum[key]), 0.997)
    for configuration in PRODUCTION_SCORE_CONFIGURATIONS:
        for metric, threshold in PRODUCTION_CONFIGURATION_MINIMUM_METRICS.items():
            minimum[f"{configuration}__{metric}"] = threshold
    maximum = {
        "event_error_rate": 0.005,
        "deleted_event_rate": 0.0015,
        "inserted_event_rate": 0.0015,
        "out_of_contract_page_count": 0,
        "duplicate_scan_page_count": 0,
        "unverified_scan_page_identity_count": 0,
    }
    for configuration in PRODUCTION_SCORE_CONFIGURATIONS:
        for metric, threshold in PRODUCTION_CONFIGURATION_MAXIMUM_METRICS.items():
            maximum[f"{configuration}__{metric}"] = threshold
    return {
        "minimum": minimum,
        "maximum": maximum,
    }


PRODUCTION_RELEASE_GATES_V2 = _production_release_gates_v2()
RELEASE_GATE_PROFILES = {
    "stable-v1": STABLE_RELEASE_GATES,
    "production-v2": PRODUCTION_RELEASE_GATES_V2,
}


_BOOTSTRAP_GATE_ALIASES = {
    "pitch_accuracy_low_95": "pitch_accuracy_aligned",
    "rhythm_accuracy_low_95": "rhythm_accuracy_aligned",
    "event_kind_accuracy_low_95": "event_kind_accuracy_aligned",
    "chord_topology_accuracy_low_95": "chord_topology_accuracy_aligned",
}


def _add_bootstrap_gate_metrics(
    gate_metrics: dict[str, object],
    bootstrap: dict[str, dict[str, float]],
) -> None:
    """Populate every stable ``*_low_95`` gate from its bootstrap interval.

    The stable gate is the single source of truth.  Deriving the mapping prevents a new
    confidence requirement from being listed but never supplied to the gate evaluator.
    Only the four historical short aliases differ from the aggregate metric name.
    """

    minimum = STABLE_RELEASE_GATES.get("minimum", {})
    for gate_key in sorted(minimum):
        if not gate_key.endswith("_low_95"):
            continue
        source_key = _BOOTSTRAP_GATE_ALIASES.get(gate_key, gate_key[: -len("_low_95")])
        interval = bootstrap.get(source_key, {})
        gate_metrics[gate_key] = float(interval.get("low_95", 0.0)) if isinstance(interval, dict) else 0.0


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.evaluation import benchmark_fingerprint, compare_musicxml  # noqa: E402
from scorescan.util import atomic_write_json, sha256_file  # noqa: E402


def validate_production_evidence(
    payload: dict[str, object],
    manifest_path: Path,
) -> dict[str, object]:
    """Verify release-evidence declarations and their immutable local audits.

    This does not pretend that a hash proves an audit is correct.  It does make
    the evidence explicit, immutable, reviewable, and impossible to replace
    silently after a benchmark result has been generated.
    """

    raw = payload.get("production_evidence")
    evidence = raw if isinstance(raw, dict) else {}
    errors: list[str] = []
    boolean_fields = (
        "page_identity_audited",
        "evaluation_use_authorized",
        "complete_page_level_semantics",
        "instrumental_lyrics_excluded_or_isolated",
        "independent_double_annotation_adjudicated",
        "work_disjoint_from_training_and_tuning",
        "frozen_before_candidate_evaluation",
        "submitted_orientation_preserved",
        "ordinary_scan_page_shape_audited",
    )
    metrics: dict[str, int] = {
        "production_boundary_contract_evidence": int(
            evidence.get("boundary_contract_version")
            == PRODUCTION_BOUNDARY_CONTRACT_VERSION
        ),
        "physical_scan_origin_evidence": int(
            evidence.get("source_image_origin") == "physical_scan"
        ),
        "scan_page_shape_contract_evidence": int(
            evidence.get("scan_page_shape_contract")
            == PRODUCTION_SCAN_PAGE_SHAPE_CONTRACT
        ),
        "scan_page_aspect_limit_evidence": int(
            evidence.get("maximum_scan_page_aspect_ratio")
            == PRODUCTION_MAXIMUM_SCAN_PAGE_ASPECT_RATIO
        ),
    }
    metric_names = {
        "page_identity_audited": "page_identity_audit_evidence",
        "evaluation_use_authorized": "evaluation_use_authorized_evidence",
        "complete_page_level_semantics": "complete_page_semantics_evidence",
        "instrumental_lyrics_excluded_or_isolated": (
            "instrumental_text_boundary_evidence"
        ),
        "independent_double_annotation_adjudicated": (
            "double_annotation_evidence"
        ),
        "work_disjoint_from_training_and_tuning": "work_isolation_evidence",
        "frozen_before_candidate_evaluation": (
            "frozen_before_candidate_evidence"
        ),
        "submitted_orientation_preserved": (
            "submitted_orientation_preserved_evidence"
        ),
        "ordinary_scan_page_shape_audited": (
            "ordinary_scan_page_shape_audit_evidence"
        ),
    }
    for field in boolean_fields:
        metrics[metric_names[field]] = int(evidence.get(field) is True)

    manifest_directory = manifest_path.parent.resolve()
    raw_files = evidence.get("evidence_files")
    raw_files = raw_files if isinstance(raw_files, list) else []
    seen_roles: set[str] = set()
    verified_files: list[dict[str, object]] = []
    for index, item in enumerate(raw_files):
        if not isinstance(item, dict):
            errors.append(f"evidence_files[{index}] is not an object")
            continue
        role = str(item.get("role", "")).strip()
        relative_path = str(item.get("path", "")).strip()
        expected_sha256 = str(item.get("sha256", "")).strip().lower()
        if role not in PRODUCTION_EVIDENCE_FILE_ROLES:
            errors.append(f"unsupported evidence role: {role or '<empty>'}")
            continue
        if role in seen_roles:
            errors.append(f"duplicate evidence role: {role}")
            continue
        seen_roles.add(role)
        path = (manifest_directory / relative_path).resolve()
        if not path.is_relative_to(manifest_directory):
            errors.append(f"evidence file escapes benchmark directory: {role}")
            continue
        if (
            not relative_path
            or not path.is_file()
            or path.stat().st_size <= 0
        ):
            errors.append(f"missing evidence file: {role}")
            continue
        actual_sha256 = sha256_file(path)
        if (
            len(expected_sha256) != 64
            or actual_sha256 != expected_sha256
        ):
            errors.append(f"evidence hash mismatch: {role}")
            continue
        verified_files.append(
            {
                "role": role,
                "path": relative_path,
                "sha256": actual_sha256,
            }
        )
    missing_roles = sorted(
        set(PRODUCTION_EVIDENCE_FILE_ROLES) - seen_roles
    )
    errors.extend(f"missing evidence role: {role}" for role in missing_roles)
    metrics["required_evidence_file_role_count"] = len(verified_files)
    for metric, value in metrics.items():
        if metric == "required_evidence_file_role_count":
            if value != len(PRODUCTION_EVIDENCE_FILE_ROLES):
                errors.append(
                    "not every required evidence file was verified"
                )
        elif value != 1:
            errors.append(f"production evidence declaration failed: {metric}")
    return {
        "format": 1,
        "passed": not errors,
        "boundary_contract_version": evidence.get(
            "boundary_contract_version"
        ),
        "source_image_origin": evidence.get("source_image_origin"),
        "scan_page_shape_contract": evidence.get(
            "scan_page_shape_contract"
        ),
        "maximum_scan_page_aspect_ratio": evidence.get(
            "maximum_scan_page_aspect_ratio"
        ),
        "metrics": metrics,
        "verified_files": verified_files,
        "errors": errors,
    }


def _rate(numerator: float, denominator: float, *, empty: float = 1.0) -> float:
    return numerator / denominator if denominator else empty


def _marker_f1(matches: float, reference_total: float, candidate_total: float) -> float:
    denominator = reference_total + candidate_total
    return 1.0 if denominator == 0.0 else (2.0 * matches) / denominator


def aggregate_reports(reports: Sequence[dict[str, object]]) -> dict[str, object]:
    totals: dict[str, float] = {}
    for report in reports:
        counts = report.get("counts", {})
        if not isinstance(counts, dict):
            continue
        for key, value in counts.items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0.0) + float(value)

    event_edits = totals.get("substituted_events", 0.0) + totals.get("deleted_events", 0.0) + totals.get("inserted_events", 0.0)
    reference_events = totals.get("reference_events", 0.0)
    candidate_events = totals.get("candidate_events", 0.0)
    deleted_events = totals.get("deleted_events", 0.0)
    inserted_events = totals.get("inserted_events", 0.0)
    event_presence_recall = _rate(reference_events - deleted_events, reference_events)
    event_presence_precision = _rate(candidate_events - inserted_events, candidate_events)
    event_presence_f1 = _rate(
        2.0 * event_presence_precision * event_presence_recall,
        event_presence_precision + event_presence_recall,
    )
    tuplet_matches = totals.get("tuplet_event_matches", 0.0)
    tuplet_precision = _rate(tuplet_matches, totals.get("candidate_tuplet_event_count", 0.0))
    tuplet_recall = _rate(tuplet_matches, totals.get("reference_tuplet_event_count", 0.0))
    tuplet_f1 = _marker_f1(
        tuplet_matches,
        totals.get("reference_tuplet_event_count", 0.0),
        totals.get("candidate_tuplet_event_count", 0.0),
    )
    tie_matches = totals.get("tie_endpoint_matches", 0.0)
    tie_precision = _rate(tie_matches, totals.get("candidate_tie_endpoint_count", 0.0))
    tie_recall = _rate(tie_matches, totals.get("reference_tie_endpoint_count", 0.0))
    tie_f1 = _marker_f1(
        tie_matches,
        totals.get("reference_tie_endpoint_count", 0.0),
        totals.get("candidate_tie_endpoint_count", 0.0),
    )
    slur_matches = totals.get("slur_endpoint_matches", 0.0)
    slur_precision = _rate(slur_matches, totals.get("candidate_slur_endpoint_count", 0.0))
    slur_recall = _rate(slur_matches, totals.get("reference_slur_endpoint_count", 0.0))
    slur_f1 = _marker_f1(
        slur_matches,
        totals.get("reference_slur_endpoint_count", 0.0),
        totals.get("candidate_slur_endpoint_count", 0.0),
    )
    beam_matches = totals.get("beam_marker_matches", 0.0)
    beam_precision = _rate(beam_matches, totals.get("candidate_beam_marker_count", 0.0))
    beam_recall = _rate(beam_matches, totals.get("reference_beam_marker_count", 0.0))
    beam_f1 = _marker_f1(
        beam_matches,
        totals.get("reference_beam_marker_count", 0.0),
        totals.get("candidate_beam_marker_count", 0.0),
    )
    articulation_matches = totals.get("articulation_marker_matches", 0.0)
    articulation_precision = _rate(
        articulation_matches, totals.get("candidate_articulation_marker_count", 0.0)
    )
    articulation_recall = _rate(
        articulation_matches, totals.get("reference_articulation_marker_count", 0.0)
    )
    articulation_f1 = _marker_f1(
        articulation_matches,
        totals.get("reference_articulation_marker_count", 0.0),
        totals.get("candidate_articulation_marker_count", 0.0),
    )
    ornament_matches = totals.get("ornament_marker_matches", 0.0)
    ornament_precision = _rate(
        ornament_matches, totals.get("candidate_ornament_marker_count", 0.0)
    )
    ornament_recall = _rate(
        ornament_matches, totals.get("reference_ornament_marker_count", 0.0)
    )
    ornament_f1 = _marker_f1(
        ornament_matches,
        totals.get("reference_ornament_marker_count", 0.0),
        totals.get("candidate_ornament_marker_count", 0.0),
    )
    accidental_matches = totals.get("accidental_marker_matches", 0.0)
    accidental_precision = _rate(
        accidental_matches,
        totals.get("candidate_accidental_marker_count", 0.0),
    )
    accidental_recall = _rate(
        accidental_matches,
        totals.get("reference_accidental_marker_count", 0.0),
    )
    accidental_f1 = _marker_f1(
        accidental_matches,
        totals.get("reference_accidental_marker_count", 0.0),
        totals.get("candidate_accidental_marker_count", 0.0),
    )
    grace_matches = totals.get("grace_event_matches", 0.0)
    grace_precision = _rate(grace_matches, totals.get("candidate_grace_event_count", 0.0))
    grace_recall = _rate(grace_matches, totals.get("reference_grace_event_count", 0.0))
    grace_f1 = _marker_f1(
        grace_matches,
        totals.get("reference_grace_event_count", 0.0),
        totals.get("candidate_grace_event_count", 0.0),
    )
    lyric_matches = totals.get("lyric_event_matches", 0.0)
    lyric_precision = _rate(lyric_matches, totals.get("candidate_lyric_event_count", 0.0))
    lyric_recall = _rate(lyric_matches, totals.get("reference_lyric_event_count", 0.0))
    lyric_f1 = _marker_f1(
        lyric_matches,
        totals.get("reference_lyric_event_count", 0.0),
        totals.get("candidate_lyric_event_count", 0.0),
    )
    cross_tie_matches = totals.get("cross_tie_matches", 0.0)
    cross_tie_precision = _rate(cross_tie_matches, totals.get("candidate_cross_tie_count", 0.0))
    cross_tie_recall = _rate(cross_tie_matches, totals.get("reference_cross_tie_count", 0.0))
    cross_tie_f1 = _marker_f1(
        cross_tie_matches,
        totals.get("reference_cross_tie_count", 0.0),
        totals.get("candidate_cross_tie_count", 0.0),
    )
    repeat_matches = totals.get("repeat_marker_matches", 0.0)
    repeat_precision = _rate(
        repeat_matches, totals.get("candidate_repeat_marker_count", 0.0)
    )
    repeat_recall = _rate(
        repeat_matches, totals.get("reference_repeat_marker_count", 0.0)
    )
    repeat_f1 = _marker_f1(
        repeat_matches,
        totals.get("reference_repeat_marker_count", 0.0),
        totals.get("candidate_repeat_marker_count", 0.0),
    )
    reference_measures = totals.get("reference_measures", 0.0)
    direction_matches = totals.get("direction_matches", 0.0)
    direction_precision = _rate(direction_matches, totals.get("candidate_direction_count", 0.0))
    direction_recall = _rate(direction_matches, totals.get("reference_direction_count", 0.0))
    direction_content_matches = totals.get("direction_content_matches", 0.0)
    direction_content_precision = _rate(
        direction_content_matches,
        totals.get("candidate_direction_count", 0.0),
    )
    direction_content_recall = _rate(
        direction_content_matches,
        totals.get("reference_direction_count", 0.0),
    )
    direction_diagnostics: dict[str, float] = {}
    for prefix in DIRECTION_METRIC_PREFIXES:
        reference_total = totals.get(f"reference_{prefix}_count", 0.0)
        candidate_total = totals.get(f"candidate_{prefix}_count", 0.0)
        exact_matches = totals.get(f"{prefix}_matches", 0.0)
        content_matches = totals.get(f"{prefix}_content_matches", 0.0)
        direction_diagnostics[f"{prefix}_precision"] = _rate(
            exact_matches, candidate_total
        )
        direction_diagnostics[f"{prefix}_recall"] = _rate(
            exact_matches, reference_total
        )
        direction_diagnostics[f"{prefix}_f1"] = _marker_f1(
            exact_matches, reference_total, candidate_total
        )
        direction_diagnostics[f"{prefix}_content_f1"] = _marker_f1(
            content_matches, reference_total, candidate_total
        )
        direction_diagnostics[f"{prefix}_anchor_accuracy"] = (
            1.0
            if reference_total + candidate_total == 0.0
            else _rate(exact_matches, content_matches, empty=0.0)
        )
    expression_matches = totals.get("expression_marker_matches", 0.0)
    expression_precision = _rate(
        expression_matches, totals.get("candidate_expression_marker_count", 0.0)
    )
    expression_recall = _rate(
        expression_matches, totals.get("reference_expression_marker_count", 0.0)
    )
    expression_f1 = _marker_f1(
        expression_matches,
        totals.get("reference_expression_marker_count", 0.0),
        totals.get("candidate_expression_marker_count", 0.0),
    )
    metrics = {
        "case_count": len(reports),
        "reference_measures": int(totals.get("reference_measures", 0.0)),
        "candidate_measures": int(totals.get("candidate_measures", 0.0)),
        "reference_event_count": int(reference_events),
        "candidate_event_count": int(totals.get("candidate_events", 0.0)),
        "exact_measure_rate": _rate(totals.get("exact_measures", 0.0), totals.get("measure_denominator", 0.0)),
        "preservation_exact_measure_rate": _rate(
            totals.get("preservation_exact_measures", 0.0), totals.get("measure_denominator", 0.0)
        ),
        "event_error_rate": _rate(event_edits, reference_events, empty=0.0),
        "weighted_event_error_rate": _rate(totals.get("weighted_event_cost", 0.0), reference_events, empty=0.0),
        "event_presence_precision": event_presence_precision,
        "event_presence_recall": event_presence_recall,
        "event_presence_f1": event_presence_f1,
        "deleted_event_rate": _rate(deleted_events, reference_events, empty=0.0),
        "inserted_event_rate": _rate(inserted_events, candidate_events, empty=0.0),
        "pitch_accuracy_aligned": _rate(totals.get("pitch_correct", 0.0), totals.get("reference_pitched_events", 0.0)),
        "duration_accuracy_aligned": _rate(totals.get("duration_correct", 0.0), totals.get("reference_rhythm_events", 0.0)),
        "rhythm_accuracy_aligned": _rate(totals.get("rhythm_correct", 0.0), totals.get("reference_rhythm_events", 0.0)),
        "onset_accuracy_aligned": _rate(totals.get("onset_correct", 0.0), totals.get("reference_rhythm_events", 0.0)),
        "chord_topology_accuracy_aligned": _rate(totals.get("chord_correct", 0.0), totals.get("reference_rhythm_events", 0.0)),
        "tuplet_topology_accuracy_aligned": _rate(
            totals.get("tuplet_topology_correct", 0.0), totals.get("reference_rhythm_events", 0.0)
        ),
        "tuplet_event_precision": tuplet_precision,
        "tuplet_event_recall": tuplet_recall,
        "tuplet_event_f1": tuplet_f1,
        "tie_topology_accuracy_aligned": _rate(
            totals.get("tie_topology_correct", 0.0), totals.get("reference_rhythm_events", 0.0)
        ),
        "tie_endpoint_precision": tie_precision,
        "tie_endpoint_recall": tie_recall,
        "tie_endpoint_f1": tie_f1,
        "slur_topology_accuracy": _rate(
            totals.get("slur_topology_correct", 0.0), reference_measures
        ),
        "slur_endpoint_precision": slur_precision,
        "slur_endpoint_recall": slur_recall,
        "slur_endpoint_f1": slur_f1,
        "beam_topology_accuracy_aligned": _rate(
            totals.get("beam_topology_correct", 0.0), totals.get("matched_events", 0.0)
        ),
        "beam_marker_precision": beam_precision,
        "beam_marker_recall": beam_recall,
        "beam_marker_f1": beam_f1,
        "articulation_topology_accuracy_aligned": _rate(
            totals.get("articulation_topology_correct", 0.0), totals.get("matched_events", 0.0)
        ),
        "articulation_marker_precision": articulation_precision,
        "articulation_marker_recall": articulation_recall,
        "articulation_marker_f1": articulation_f1,
        "ornament_topology_accuracy_aligned": _rate(
            totals.get("ornament_topology_correct", 0.0), totals.get("matched_events", 0.0)
        ),
        "ornament_marker_precision": ornament_precision,
        "ornament_marker_recall": ornament_recall,
        "ornament_marker_f1": ornament_f1,
        "accidental_marker_precision": accidental_precision,
        "accidental_marker_recall": accidental_recall,
        "accidental_marker_f1": accidental_f1,
        "grace_topology_accuracy_aligned": _rate(
            totals.get("grace_topology_correct", 0.0), totals.get("matched_events", 0.0)
        ),
        "grace_event_precision": grace_precision,
        "grace_event_recall": grace_recall,
        "grace_event_f1": grace_f1,
        "lyric_topology_accuracy_aligned": _rate(
            totals.get("lyric_topology_correct", 0.0), totals.get("matched_events", 0.0)
        ),
        "lyric_event_precision": lyric_precision,
        "lyric_event_recall": lyric_recall,
        "lyric_event_f1": lyric_f1,
        "cross_tie_boundary_accuracy": _rate(
            totals.get("cross_tie_boundary_correct", 0.0),
            totals.get("reference_cross_tie_boundary_count", 0.0),
        ),
        "cross_tie_precision": cross_tie_precision,
        "cross_tie_recall": cross_tie_recall,
        "cross_tie_f1": cross_tie_f1,
        "event_kind_accuracy_aligned": _rate(totals.get("rest_correct", 0.0), totals.get("matched_events", 0.0)),
        "direction_precision": direction_precision,
        "direction_recall": direction_recall,
        "direction_f1": _marker_f1(
            direction_matches,
            totals.get("reference_direction_count", 0.0),
            totals.get("candidate_direction_count", 0.0),
        ),
        "direction_content_precision": direction_content_precision,
        "direction_content_recall": direction_content_recall,
        "direction_content_f1": _marker_f1(
            direction_content_matches,
            totals.get("reference_direction_count", 0.0),
            totals.get("candidate_direction_count", 0.0),
        ),
        "direction_anchor_accuracy": (
            1.0
            if totals.get("reference_direction_count", 0.0)
            + totals.get("candidate_direction_count", 0.0)
            == 0.0
            else _rate(direction_matches, direction_content_matches, empty=0.0)
        ),
        **direction_diagnostics,
        "expression_marker_precision": expression_precision,
        "expression_marker_recall": expression_recall,
        "expression_marker_f1": expression_f1,
        "time_signature_accuracy": _rate(totals.get("time_signature_correct", 0.0), reference_measures),
        "key_signature_accuracy": _rate(totals.get("key_signature_correct", 0.0), reference_measures),
        "clef_accuracy": _rate(totals.get("clef_correct", 0.0), reference_measures),
        "barline_accuracy": _rate(totals.get("barline_correct", 0.0), reference_measures),
        "repeat_marker_precision": repeat_precision,
        "repeat_marker_recall": repeat_recall,
        "repeat_marker_f1": repeat_f1,
        "deleted_measures": int(totals.get("deleted_measures", 0.0)),
        "inserted_measures": int(totals.get("inserted_measures", 0.0)),
        "deleted_events": int(totals.get("deleted_events", 0.0)),
        "inserted_events": int(totals.get("inserted_events", 0.0)),
        "substituted_events": int(totals.get("substituted_events", 0.0)),
    }
    metrics["utility_score"] = max(
        0.0,
        min(
            1.0,
            0.28 * float(metrics["pitch_accuracy_aligned"])
            + 0.24 * float(metrics["rhythm_accuracy_aligned"])
            + 0.04 * float(metrics["chord_topology_accuracy_aligned"])
            + 0.03 * float(metrics["tuplet_event_f1"])
            + 0.03 * float(metrics["tie_endpoint_f1"])
            + 0.02 * float(metrics["slur_endpoint_f1"])
            + 0.01 * float(metrics["beam_marker_f1"])
            + 0.01 * float(metrics["articulation_marker_f1"])
            + 0.01 * float(metrics["ornament_marker_f1"])
            + 0.02 * float(metrics["grace_event_f1"])
            + 0.02 * float(metrics["lyric_event_f1"])
            + 0.01 * float(metrics["cross_tie_f1"])
            + 0.05 * float(metrics["exact_measure_rate"])
            + 0.05 * float(metrics["preservation_exact_measure_rate"])
            + 0.10 * float(metrics["time_signature_accuracy"])
            + 0.06 * float(metrics["direction_f1"])
            + 0.01 * float(metrics["expression_marker_f1"])
            + 0.01 * float(metrics["repeat_marker_f1"]),
        ),
    )
    metrics["counts"] = totals
    return metrics


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def bootstrap_intervals(
    reports: Sequence[dict[str, object]],
    *,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if not reports or samples <= 0:
        return {}
    keys = (
        "exact_measure_rate",
        "preservation_exact_measure_rate",
        "weighted_event_error_rate",
        "pitch_accuracy_aligned",
        "rhythm_accuracy_aligned",
        "event_kind_accuracy_aligned",
        "chord_topology_accuracy_aligned",
        "tuplet_topology_accuracy_aligned",
        "tuplet_event_precision",
        "tuplet_event_recall",
        "tuplet_event_f1",
        "tie_topology_accuracy_aligned",
        "tie_endpoint_precision",
        "tie_endpoint_recall",
        "tie_endpoint_f1",
        "slur_topology_accuracy",
        "slur_endpoint_precision",
        "slur_endpoint_recall",
        "slur_endpoint_f1",
        "articulation_topology_accuracy_aligned",
        "articulation_marker_precision",
        "articulation_marker_recall",
        "articulation_marker_f1",
        "ornament_topology_accuracy_aligned",
        "ornament_marker_precision",
        "ornament_marker_recall",
        "ornament_marker_f1",
        "accidental_marker_precision",
        "accidental_marker_recall",
        "accidental_marker_f1",
        "grace_topology_accuracy_aligned",
        "grace_event_precision",
        "grace_event_recall",
        "grace_event_f1",
        "lyric_topology_accuracy_aligned",
        "lyric_event_precision",
        "lyric_event_recall",
        "lyric_event_f1",
        "beam_topology_accuracy_aligned",
        "beam_marker_precision",
        "beam_marker_recall",
        "beam_marker_f1",
        "cross_tie_boundary_accuracy",
        "cross_tie_precision",
        "cross_tie_recall",
        "cross_tie_f1",
        "event_presence_precision",
        "event_presence_recall",
        "event_presence_f1",
        "deleted_event_rate",
        "inserted_event_rate",
        "time_signature_accuracy",
        "key_signature_accuracy",
        "clef_accuracy",
        "barline_accuracy",
        "repeat_marker_precision",
        "repeat_marker_recall",
        "repeat_marker_f1",
        "direction_precision",
        "direction_recall",
        "direction_f1",
        "direction_content_f1",
        "direction_anchor_accuracy",
        "words_direction_f1",
        "words_direction_content_f1",
        "words_direction_anchor_accuracy",
        "dynamic_direction_f1",
        "dynamic_direction_content_f1",
        "dynamic_direction_anchor_accuracy",
        "wedge_direction_f1",
        "wedge_direction_content_f1",
        "wedge_direction_anchor_accuracy",
        "crescendo_wedge_start_f1",
        "crescendo_wedge_start_content_f1",
        "crescendo_wedge_start_anchor_accuracy",
        "diminuendo_wedge_start_f1",
        "diminuendo_wedge_start_content_f1",
        "diminuendo_wedge_start_anchor_accuracy",
        "wedge_stop_f1",
        "wedge_stop_content_f1",
        "wedge_stop_anchor_accuracy",
        "expression_marker_precision",
        "expression_marker_recall",
        "expression_marker_f1",
        "utility_score",
    )
    distributions = {key: [] for key in keys}
    rng = random.Random(seed)
    for _ in range(samples):
        sampled = [reports[rng.randrange(len(reports))] for _ in range(len(reports))]
        aggregate = aggregate_reports(sampled)
        for key in keys:
            distributions[key].append(float(aggregate[key]))
    return {
        key: {
            "low_95": _percentile(values, 0.025),
            "median": _percentile(values, 0.5),
            "high_95": _percentile(values, 0.975),
        }
        for key, values in distributions.items()
    }


def _evaluate_gates(metrics: dict[str, object], gates: object) -> dict[str, object]:
    if not isinstance(gates, dict):
        return {"configured": False, "passed": False, "checks": []}
    checks: list[dict[str, object]] = []
    for relation, values in (("minimum", gates.get("minimum")), ("maximum", gates.get("maximum"))):
        if not isinstance(values, dict):
            continue
        for key, threshold in sorted(values.items()):
            actual = metrics.get(str(key))
            ok = isinstance(actual, (int, float)) and isinstance(threshold, (int, float))
            if ok:
                ok = float(actual) >= float(threshold) if relation == "minimum" else float(actual) <= float(threshold)
            checks.append({
                "metric": str(key),
                "relation": relation,
                "threshold": threshold,
                "actual": actual,
                "ok": bool(ok),
            })
    return {
        "configured": bool(checks),
        "passed": bool(checks) and all(bool(item["ok"]) for item in checks),
        "checks": checks,
    }



def _case_strata(raw: dict[str, object]) -> dict[str, str]:
    value = raw.get("strata", {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("case strata must be an object")
    result: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).strip()
        if not key or len(key) > 64:
            raise ValueError("case stratum names must be non-empty and at most 64 characters")
        if not isinstance(raw_value, (str, int, float, bool)):
            raise ValueError(f"case stratum {key} must be a scalar")
        item = str(raw_value).strip()
        if not item or len(item) > 128:
            raise ValueError(f"case stratum {key} values must be non-empty and at most 128 characters")
        result[key] = item
    return dict(sorted(result.items()))


def _aggregate_strata(reports: Sequence[dict[str, object]]) -> dict[str, dict[str, dict[str, object]]]:
    grouped: dict[str, dict[str, list[dict[str, object]]]] = {}
    for report in reports:
        strata = report.get("strata", {})
        if not isinstance(strata, dict):
            continue
        for key, value in strata.items():
            grouped.setdefault(str(key), {}).setdefault(str(value), []).append(report)
    return {
        key: {value: aggregate_reports(rows) for value, rows in sorted(values.items())}
        for key, values in sorted(grouped.items())
    }


def production_scope_coverage(
    reports: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Count document/page/work coverage promised by the public boundary.

    The score-configuration stratum is closed. Unknown or missing values remain
    visible as out-of-contract pages and make production-v2 fail; they may still
    be evaluated diagnostically without diluting an in-boundary denominator.

    A benchmark case is a submitted document, not necessarily one page.  This
    mirrors the real multi-page PDF workflow while keeping every page explicit
    and attributable to one score configuration.  Legacy manifests without a
    page count remain one-page cases.
    """

    counts: Counter[str] = Counter()
    source_groups: set[str] = set()
    scan_page_id_counts: Counter[str] = Counter()
    page_count = 0
    unverified_scan_page_identity_count = 0
    for report in reports:
        source_group = str(report.get("source_group") or "").strip()
        if source_group:
            source_groups.add(source_group)
        submitted_pages = int(report.get("submitted_scan_page_count", 1) or 0)
        if submitted_pages <= 0:
            raise ValueError(
                "submitted scan page counts must be positive integers"
            )
        page_count += submitted_pages
        raw_scan_page_ids = report.get("submitted_scan_page_ids")
        if raw_scan_page_ids is None:
            unverified_scan_page_identity_count += submitted_pages
        elif not isinstance(raw_scan_page_ids, list):
            raise ValueError("submitted scan page ids must be a list")
        else:
            if len(raw_scan_page_ids) != submitted_pages:
                raise ValueError(
                    "submitted scan page id count must match submitted scan page count"
                )
            for raw_page_id in raw_scan_page_ids:
                page_id = str(raw_page_id).strip()
                if not page_id:
                    raise ValueError(
                        "submitted scan page ids must be non-empty strings"
                    )
                scan_page_id_counts[page_id] += 1
        strata = report.get("strata")
        configuration = (
            str(strata.get("score_configuration") or "").strip()
            if isinstance(strata, dict)
            else ""
        )
        counts[configuration] += submitted_pages
    by_configuration = {
        name: int(counts[name]) for name in PRODUCTION_SCORE_CONFIGURATIONS
    }
    classified = sum(by_configuration.values())
    verified_unique_scan_page_count = len(scan_page_id_counts)
    duplicate_scan_page_count = sum(
        count - 1
        for count in scan_page_id_counts.values()
        if count > 1
    )
    return {
        "case_unit": "one_submitted_document",
        "document_count": len(reports),
        "page_count": page_count,
        "verified_unique_scan_page_count": verified_unique_scan_page_count,
        "duplicate_scan_page_count": duplicate_scan_page_count,
        "unverified_scan_page_identity_count": (
            unverified_scan_page_identity_count
        ),
        "source_group_count": len(source_groups),
        "score_configuration_stratum": "score_configuration",
        "pages_by_score_configuration": by_configuration,
        "scope_classified_page_count": classified,
        "out_of_contract_page_count": page_count - classified,
    }


def _add_production_scope_gate_metrics(
    gate_metrics: dict[str, object],
    coverage: dict[str, object],
) -> None:
    gate_metrics["source_group_count"] = coverage["source_group_count"]
    gate_metrics["submitted_scan_page_count"] = coverage["page_count"]
    gate_metrics["verified_unique_scan_page_count"] = coverage[
        "verified_unique_scan_page_count"
    ]
    gate_metrics["duplicate_scan_page_count"] = coverage[
        "duplicate_scan_page_count"
    ]
    gate_metrics["unverified_scan_page_identity_count"] = coverage[
        "unverified_scan_page_identity_count"
    ]
    gate_metrics["scope_classified_page_count"] = coverage[
        "scope_classified_page_count"
    ]
    gate_metrics["out_of_contract_page_count"] = coverage[
        "out_of_contract_page_count"
    ]
    pages = coverage["pages_by_score_configuration"]
    if not isinstance(pages, dict):
        raise ValueError("invalid production score-configuration coverage")
    for name in PRODUCTION_SCORE_CONFIGURATIONS:
        gate_metrics[f"{name}_page_count"] = int(pages.get(name, 0))


def _add_reference_feature_gate_metrics(
    gate_metrics: dict[str, object],
    aggregate: dict[str, object],
) -> dict[str, int]:
    counts = aggregate.get("counts")
    counts = counts if isinstance(counts, dict) else {}
    result: dict[str, int] = {}
    for metric in PRODUCTION_REFERENCE_FEATURE_MINIMUM_COUNTS:
        raw = counts.get(metric, 0)
        value = int(raw) if isinstance(raw, (int, float)) else 0
        result[metric] = value
        gate_metrics[metric] = value
    return result


def _add_production_configuration_quality_gate_metrics(
    gate_metrics: dict[str, object],
    reports: Sequence[dict[str, object]],
    *,
    samples: int,
    seed: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """Flatten per-configuration aggregates and confidence bounds for gates.

    Missing configurations are represented by failing values rather than the
    mathematically convenient empty-set rates returned by ``aggregate_reports``.
    This makes accidental manifest omissions fail closed.
    """

    grouped: dict[str, list[dict[str, object]]] = {
        name: [] for name in PRODUCTION_SCORE_CONFIGURATIONS
    }
    for report in reports:
        strata = report.get("strata")
        configuration = (
            str(strata.get("score_configuration") or "").strip()
            if isinstance(strata, dict)
            else ""
        )
        if configuration in grouped:
            grouped[configuration].append(report)

    configuration_intervals: dict[
        str, dict[str, dict[str, float]]
    ] = {}
    for index, configuration in enumerate(PRODUCTION_SCORE_CONFIGURATIONS):
        rows = grouped[configuration]
        aggregate = aggregate_reports(rows) if rows else {}
        intervals = bootstrap_intervals(
            rows,
            samples=samples,
            seed=seed + (index + 1) * 1009,
        )
        configuration_intervals[configuration] = intervals
        prefix = f"{configuration}__"
        aggregate_counts = aggregate.get("counts")
        aggregate_counts = (
            aggregate_counts if isinstance(aggregate_counts, dict) else {}
        )
        for metric in PRODUCTION_CONFIGURATION_MINIMUM_METRICS:
            if metric.endswith("_low_95"):
                source_metric = metric[: -len("_low_95")]
                interval = intervals.get(source_metric, {})
                value = (
                    float(interval.get("low_95", 0.0))
                    if isinstance(interval, dict)
                    else 0.0
                )
            else:
                value = aggregate.get(
                    metric,
                    aggregate_counts.get(metric, 0.0),
                )
            gate_metrics[f"{prefix}{metric}"] = value
        for metric in PRODUCTION_CONFIGURATION_MAXIMUM_METRICS:
            gate_metrics[f"{prefix}{metric}"] = (
                aggregate.get(metric, 1.0) if rows else 1.0
            )
    return configuration_intervals


def evaluate_manifest(
    manifest_path: Path,
    *,
    split: str | None = None,
    bootstrap_samples_override: int | None = None,
    bootstrap_seed_override: int | None = None,
    enforce_stable_gate: bool = False,
    gate_profile: str | None = None,
) -> dict[str, object]:
    if enforce_stable_gate and gate_profile is not None:
        raise ValueError("choose either enforce_stable_gate or gate_profile")
    if gate_profile is not None and gate_profile not in RELEASE_GATE_PROFILES:
        raise ValueError(f"unknown release gate profile: {gate_profile}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or int(payload.get("format", 0) or 0) != 1:
        raise ValueError("benchmark manifest format must be 1")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("benchmark manifest must contain cases")

    resolved: list[
        tuple[
            str,
            Path,
            Path,
            str,
            str,
            dict[str, str],
            int,
            list[str] | None,
        ]
    ] = []
    seen: set[str] = set()
    source_group_splits: dict[str, set[str]] = {}
    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise ValueError("each benchmark case must be an object")
        case_id = str(raw.get("id", "")).strip()
        case_split = str(raw.get("split", "test")).strip() or "test"
        source_group = str(raw.get("source_group", case_id)).strip() or case_id
        strata = _case_strata(raw)
        raw_submitted_pages = raw.get("submitted_scan_page_count", 1)
        if (
            isinstance(raw_submitted_pages, bool)
            or not isinstance(raw_submitted_pages, int)
            or raw_submitted_pages <= 0
        ):
            raise ValueError(
                "submitted_scan_page_count must be a positive integer"
            )
        submitted_scan_page_count = int(raw_submitted_pages)
        raw_scan_page_ids = raw.get("submitted_scan_page_ids")
        submitted_scan_page_ids: list[str] | None = None
        if raw_scan_page_ids is not None:
            if not isinstance(raw_scan_page_ids, list):
                raise ValueError(
                    "submitted_scan_page_ids must be a list"
                )
            if len(raw_scan_page_ids) != submitted_scan_page_count:
                raise ValueError(
                    "submitted_scan_page_ids length must equal "
                    "submitted_scan_page_count"
                )
            submitted_scan_page_ids = []
            for value in raw_scan_page_ids:
                if not isinstance(value, str):
                    raise ValueError(
                        "submitted_scan_page_ids must contain strings"
                    )
                page_id = value.strip()
                if not page_id or len(page_id) > 256:
                    raise ValueError(
                        "submitted_scan_page_ids must contain non-empty "
                        "strings of at most 256 characters"
                    )
                submitted_scan_page_ids.append(page_id)
        if not case_id or case_id in seen:
            raise ValueError("benchmark case ids must be unique and non-empty")
        seen.add(case_id)
        source_group_splits.setdefault(source_group, set()).add(case_split)
        if split is not None and case_split != split:
            continue
        reference = (manifest_path.parent / str(raw.get("reference", ""))).resolve()
        candidate = (manifest_path.parent / str(raw.get("candidate", ""))).resolve()
        if not reference.is_file() or not candidate.is_file():
            raise FileNotFoundError(f"missing benchmark pair for {case_id}")
        resolved.append(
            (
                case_id,
                reference,
                candidate,
                case_split,
                source_group,
                strata,
                submitted_scan_page_count,
                submitted_scan_page_ids,
            )
        )
    leaking_groups = sorted(group for group, splits in source_group_splits.items() if len(splits) > 1)
    if leaking_groups:
        preview = ", ".join(leaking_groups[:5])
        raise ValueError(f"source groups cross benchmark splits: {preview}")
    if not resolved:
        raise ValueError("no benchmark cases matched the requested split")

    reports: list[dict[str, object]] = []
    for (
        case_id,
        reference,
        candidate,
        case_split,
        source_group,
        strata,
        submitted_scan_page_count,
        submitted_scan_page_ids,
    ) in resolved:
        report = compare_musicxml(reference, candidate)
        report["id"] = case_id
        report["split"] = case_split
        report["source_group"] = source_group
        report["strata"] = strata
        report["submitted_scan_page_count"] = submitted_scan_page_count
        if submitted_scan_page_ids is not None:
            report["submitted_scan_page_ids"] = submitted_scan_page_ids
        reports.append(report)

    aggregate = aggregate_reports(reports)
    stratified = _aggregate_strata(reports)
    production_coverage = production_scope_coverage(reports)
    bootstrap_samples = (
        int(bootstrap_samples_override)
        if bootstrap_samples_override is not None
        else int(payload.get("bootstrap_samples", 1000) or 0)
    )
    seed = (
        int(bootstrap_seed_override)
        if bootstrap_seed_override is not None
        else int(payload.get("bootstrap_seed", 20260719) or 20260719)
    )
    cases_for_hash = [
        (
            f"{case_id}\0{case_split}\0{source_group}\0"
            f"{submitted_scan_page_count}\0"
            + json.dumps(
                submitted_scan_page_ids,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\0"
            + json.dumps(
                strata,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            reference,
            candidate,
        )
        for (
            case_id,
            reference,
            candidate,
            case_split,
            source_group,
            strata,
            submitted_scan_page_count,
            submitted_scan_page_ids,
        ) in resolved
    ]
    bootstrap = bootstrap_intervals(reports, samples=bootstrap_samples, seed=seed)
    gate_metrics = dict(aggregate)
    _add_bootstrap_gate_metrics(gate_metrics, bootstrap)
    _add_production_scope_gate_metrics(gate_metrics, production_coverage)
    reference_feature_coverage = _add_reference_feature_gate_metrics(
        gate_metrics,
        aggregate,
    )
    production_evidence = validate_production_evidence(
        payload,
        manifest_path,
    )
    selected_profile = (
        "stable-v1"
        if enforce_stable_gate
        else gate_profile
        if gate_profile is not None
        else "manifest"
    )
    production_configuration_bootstrap: dict[
        str, dict[str, dict[str, float]]
    ] = {}
    if selected_profile == "production-v2":
        evidence_metrics = production_evidence.get("metrics")
        if not isinstance(evidence_metrics, dict):
            raise ValueError("invalid production evidence metrics")
        gate_metrics.update(evidence_metrics)
        production_configuration_bootstrap = (
            _add_production_configuration_quality_gate_metrics(
                gate_metrics,
                reports,
                samples=bootstrap_samples,
                seed=seed,
            )
        )
    gates = (
        RELEASE_GATE_PROFILES[selected_profile]
        if selected_profile != "manifest"
        else payload.get("gates")
    )
    release_gate = _evaluate_gates(gate_metrics, gates)
    release_gate["profile"] = selected_profile
    return {
        "format": 1,
        "name": str(payload.get("name", manifest_path.stem)),
        "split": split or "all",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "benchmark_fingerprint": benchmark_fingerprint(cases_for_hash),
        "case_count": len(reports),
        "bootstrap_seed": seed,
        "bootstrap_samples": bootstrap_samples,
        "aggregate": aggregate,
        "production_scope_coverage": production_coverage,
        "production_reference_feature_coverage": (
            reference_feature_coverage
        ),
        "production_evidence": production_evidence,
        "stratified": stratified,
        "bootstrap_95": bootstrap,
        "production_configuration_bootstrap_95": (
            production_configuration_bootstrap
        ),
        "release_gate": release_gate,
        "cases": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen MusicXML release dataset manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--split")
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument("--seed", type=int, help="Override the deterministic score-bootstrap seed.")
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Apply the fixed stable-v1 release gate instead of manifest-local gates.",
    )
    parser.add_argument(
        "--gate-profile",
        choices=sorted(RELEASE_GATE_PROFILES),
        help="Apply a named fixed release gate profile.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_manifest(
        args.manifest,
        split=args.split,
        bootstrap_samples_override=args.bootstrap_samples,
        bootstrap_seed_override=args.seed,
        enforce_stable_gate=args.gate,
        gate_profile=args.gate_profile,
    )
    if args.output:
        atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if (args.gate or args.gate_profile) and not bool(
        report["release_gate"]["passed"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
