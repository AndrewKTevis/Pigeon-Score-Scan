from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scorescan.config import APP_VERSION, WORKFLOW_VERSION
from scorescan.model_registry import build_manifest
from scorescan.policy import DEFAULT_POLICY
from scorescan.semantic_detector import semantic_detector_status
from scorescan.state_schema import CURRENT_JOB_SCHEMA
from scorescan.util import read_json, sha256_file


REQUIRED_DOCS = (
    "README_zh-CN.txt",
    "RELEASE_NOTES_zh-CN.txt",
    "ACCURACY_AND_LIMITATIONS_zh-CN.txt",
    "ARCHITECTURE.md",
    "BUILD_AND_TEST.md",
    "PRIVACY_zh-CN.md",
    "SECURITY.md",
    "CONTRIBUTING.md",
    "PUBLIC_RELEASE_CHECKLIST.md",
    "REAL_SCAN_BENCHMARK_PROTOCOL_zh-CN.md",
    "PRODUCTION_EVIDENCE_CONTRACT_zh-CN.md",
    "PRODUCTION_SOAK_PROTOCOL_zh-CN.md",
    "THIRD_PARTY_NOTICES.txt",
    "LICENSE",
)


def find_non_cache_temporary_files(source_root: Path) -> list[str]:
    """Find release-blocking .tmp files without stat-ing every training asset."""

    leftovers: list[str] = []
    for directory, child_directories, filenames in os.walk(source_root):
        child_directories[:] = [
            name for name in child_directories if name != "__pycache__"
        ]
        directory_path = Path(directory)
        for filename in filenames:
            if filename.endswith(".tmp"):
                leftovers.append(
                    (directory_path / filename)
                    .relative_to(source_root)
                    .as_posix()
                )
    return sorted(leftovers)


def check(source_root: Path) -> dict[str, object]:
    app_root = source_root / "app"
    resources = app_root / "src" / "scorescan" / "resources"
    items: list[dict[str, object]] = []

    def add(key: str, ok: bool, message: str, *, stable_blocker: bool = False) -> None:
        items.append({"key": key, "ok": ok, "message": message, "stable_blocker": stable_blocker})

    version = (source_root / "VERSION").read_text(encoding="utf-8").strip() if (source_root / "VERSION").exists() else ""
    add("version:root", version == APP_VERSION, f"VERSION={version}; runtime={APP_VERSION}")
    pyproject = (app_root / "pyproject.toml").read_text(encoding="utf-8") if (app_root / "pyproject.toml").exists() else ""
    pep440_version = APP_VERSION.replace("-dev", ".dev0")
    add(
        "version:pyproject",
        f'version = "{pep440_version}"' in pyproject,
        "pyproject version matches runtime",
    )
    add("workflow", WORKFLOW_VERSION == "printed-full-score-scan@1", WORKFLOW_VERSION)
    add("policy", DEFAULT_POLICY.version == "scorescan-policy-61", DEFAULT_POLICY.version)
    add(
        "policy:lowres-multiscale",
        DEFAULT_POLICY.lowres_strip_low_target_height
        < DEFAULT_POLICY.lowres_strip_target_height
        < DEFAULT_POLICY.lowres_strip_high_target_height
        and DEFAULT_POLICY.lowres_strip_internal_min_support >= 2
        and 0.0 <= DEFAULT_POLICY.lowres_strip_candidate_bonus <= 2.0
        and DEFAULT_POLICY.lowres_strip_count_override_score_margin >= 40.0,
        "bounded three-scale strip probes require strict support and a conservative count-override margin",
    )
    add("schema", CURRENT_JOB_SCHEMA == 24, f"job schema {CURRENT_JOB_SCHEMA}")

    for filename in REQUIRED_DOCS:
        add(f"doc:{filename}", (source_root / filename).is_file(), f"required document {filename}")

    generated = build_manifest(resources)
    committed = read_json(resources / "model_manifest.json", {})
    add(
        "models:manifest",
        generated == committed and len(generated.get("models", [])) == 37,
        "37 in-boundary runtime models match the bundled model bytes",
    )
    positioned_detector = semantic_detector_status(resources)
    add(
        "models:semantic-detector-runtime",
        positioned_detector.enabled,
        (
            "independent-scan and ONNX-parity authorized positioned-symbol "
            f"detector: {positioned_detector.status}"
        ),
        stable_blocker=True,
    )
    transaction_training = read_json(
        source_root / "training" / "patch_transaction_calibrator_report_v1.json", {}
    )
    transaction_selected = (
        transaction_training.get("selected_threshold", {})
        if isinstance(transaction_training, dict)
        else {}
    )
    transaction_test = (
        transaction_training.get("frozen_test", {})
        if isinstance(transaction_training, dict)
        else {}
    )
    transaction_policy = (
        transaction_test.get("policy", {}) if isinstance(transaction_test, dict) else {}
    )
    transaction_baseline = (
        transaction_test.get("accept_all_model_applicable_transactions", {})
        if isinstance(transaction_test, dict)
        else {}
    )
    transaction_confirmation = (
        transaction_training.get("independent_confirmation", {})
        if isinstance(transaction_training, dict)
        else {}
    )
    transaction_confirmation_policy = (
        transaction_confirmation.get("policy", {})
        if isinstance(transaction_confirmation, dict)
        else {}
    )
    transaction_ablation = (
        transaction_training.get("interaction_feature_ablation", {})
        if isinstance(transaction_training, dict)
        else {}
    )
    transaction_ablation_policy = (
        transaction_ablation.get("frozen_test", {}).get("policy", {})
        if isinstance(transaction_ablation, dict)
        else {}
    )
    transaction_deployment = (
        transaction_training.get("deployment_parity", {})
        if isinstance(transaction_training, dict)
        else {}
    )
    add(
        "patch-transaction:cpu-training",
        transaction_training.get("model_version") == "scorescan-patch-transaction-forest-1"
        and transaction_training.get("dataset_kind")
        == "synthetic interaction scenarios derived from ScoreScan patch contracts"
        and isinstance(transaction_selected, dict)
        and int(transaction_selected.get("false_accepts", 1) or 0) == 0
        and float(transaction_selected.get("precision", 0.0) or 0.0) >= 0.9995
        and float(transaction_selected.get("positive_recall", 0.0) or 0.0) >= 0.98
        and float(transaction_selected.get("threshold", 0.0) or 0.0)
        >= DEFAULT_POLICY.patch_transaction_probability_floor
        and isinstance(transaction_policy, dict)
        and int(transaction_policy.get("false_accepts", 1) or 0) == 0
        and float(transaction_policy.get("positive_recall", 0.0) or 0.0) >= 0.98
        and isinstance(transaction_confirmation_policy, dict)
        and int(transaction_confirmation_policy.get("false_accepts", 1) or 0) == 0
        and float(transaction_confirmation_policy.get("positive_recall", 0.0) or 0.0) >= 0.98
        and isinstance(transaction_baseline, dict)
        and int(transaction_baseline.get("false_accepts", 0) or 0) >= 1_000
        and isinstance(transaction_ablation_policy, dict)
        and (
            int(transaction_ablation_policy.get("false_accepts", 0) or 0) >= 1
            or float(transaction_policy.get("positive_recall", 0.0) or 0.0)
            >= float(transaction_ablation_policy.get("positive_recall", 0.0) or 0.0) + 0.01
        )
        and isinstance(transaction_deployment, dict)
        and float(transaction_deployment.get("max_absolute_probability_delta", 1.0) or 1.0) <= 1e-10
        and int(transaction_training.get("model_bytes", 10**9) or 10**9) <= 2_500_000,
        "interaction-aware veto removes synthetic false commits with grouped splits, independent confirmation and ablation evidence",
    )
    imaging_path = app_root / "src" / "scorescan" / "imaging.py"
    imaging_text = imaging_path.read_text(encoding="utf-8") if imaging_path.is_file() else ""
    add(
        "orientation:automatic-rotation-disabled",
        "PageOrientationClassifier" not in imaging_text
        and "rotate_quadrant" not in imaging_text
        and "_rotate_keep_size" not in imaging_text
        and "automatic-rotation-disabled" in imaging_text,
        "production preprocessing preserves submitted page direction and dimensions",
    )
    source_beam_report = read_json(
        source_root
        / "training_data"
        / "diagnostics"
        / "source-beam-restoration-v1"
        / "source-beam-restoration-report.json",
        {},
    )
    source_beam_aggregate = (
        source_beam_report.get("aggregate", {})
        if isinstance(source_beam_report, dict)
        else {}
    )
    source_beam_gate = (
        source_beam_report.get("gate", {})
        if isinstance(source_beam_report, dict)
        else {}
    )
    source_beam_cases = (
        source_beam_report.get("cases", [])
        if isinstance(source_beam_report, dict)
        else []
    )
    add(
        "source-beam:registered-resolver",
        source_beam_report.get("evaluation_kind")
        == "oracle-registered-source-beam-resolver"
        and source_beam_report.get("source_box_provenance")
        == "complete-reference-page-svg-before-tile-clipping"
        and source_beam_report.get("not_a_detector_benchmark") is True
        and isinstance(source_beam_aggregate, dict)
        and int(source_beam_aggregate.get("case_count", 0) or 0) >= 16
        and int(
            source_beam_aggregate.get(
                "reference_beam_marker_count",
                0,
            )
            or 0
        )
        >= 2_500
        and float(
            source_beam_aggregate.get("beam_marker_precision", 0.0)
            or 0.0
        )
        >= 0.999
        and float(
            source_beam_aggregate.get("beam_marker_recall", 0.0)
            or 0.0
        )
        >= 0.99
        and isinstance(source_beam_gate, dict)
        and source_beam_gate.get("passed") is True
        and isinstance(source_beam_cases, list)
        and len(source_beam_cases) >= 16
        and all(
            isinstance(item, dict)
            and float(item.get("beam_marker_precision", 0.0) or 0.0)
            >= 0.995
            and item.get("non_beam_preservation_exact") is True
            and int(item.get("unassigned_box_count", 1) or 0) == 0
            for item in source_beam_cases
        ),
        (
            "registered full-page source resolver: "
            f"{int(source_beam_aggregate.get('beam_marker_matches', 0) or 0)}/"
            f"{int(source_beam_aggregate.get('reference_beam_marker_count', 0) or 0)} "
            "beam markers; detector recall remains a separate gate"
        ),
    )
    count_report = read_json(source_root / "training" / "measure_count_real_scan_eval_v4.json", {})
    add(
        "measure-count:real-layout-regression",
        float(count_report.get("accuracy", 0.0) or 0.0) >= 1.0,
        "real-page layout/count fusion regression resolves all maintained cases",
    )
    count_training = read_json(source_root / "training" / "measure_count_resolver_report_v4.json", {})
    count_frozen = count_training.get("frozen", {}) if isinstance(count_training, dict) else {}
    count_sample = count_frozen.get("sample", {}) if isinstance(count_frozen, dict) else {}
    count_decision = count_frozen.get("decision", {}) if isinstance(count_frozen, dict) else {}
    count_gated = count_frozen.get("gated", {}) if isinstance(count_frozen, dict) else {}
    count_baseline = count_training.get("baseline_v2_same_test", {}) if isinstance(count_training, dict) else {}
    count_baseline_decision = count_baseline.get("decision", {}) if isinstance(count_baseline, dict) else {}
    count_ablation = count_training.get("legacy_feature_forest_ablation", {}) if isinstance(count_training, dict) else {}
    count_ablation_decision = count_ablation.get("decision", {}) if isinstance(count_ablation, dict) else {}
    count_two_family = (
        count_decision.get("by_kind", {}).get("two-family-shared-error", {})
        if isinstance(count_decision, dict)
        else {}
    )
    count_thresholds = count_training.get("selected_policy_thresholds", {}) if isinstance(count_training, dict) else {}
    count_config = count_training.get("selected_configuration", {}) if isinstance(count_training, dict) else {}
    add(
        "measure-count:cpu-training",
        count_training.get("model_version") == "scorescan-measure-count-resolver-4"
        and isinstance(count_sample, dict)
        and float(count_sample.get("roc_auc", 0.0) or 0.0) >= 0.998
        and float(count_sample.get("log_loss", 1.0) or 1.0) <= 0.055
        and isinstance(count_decision, dict)
        and float(count_decision.get("top1", 0.0) or 0.0) >= 0.985
        and isinstance(count_baseline_decision, dict)
        and float(count_decision.get("top1", 0.0) or 0.0)
        >= float(count_baseline_decision.get("top1", 0.0) or 0.0) + 0.08
        and isinstance(count_ablation_decision, dict)
        and float(count_decision.get("top1", 0.0) or 0.0)
        >= float(count_ablation_decision.get("top1", 0.0) or 0.0) + 0.01
        and isinstance(count_two_family, dict)
        and float(count_two_family.get("top1", 0.0) or 0.0) >= 0.80
        and isinstance(count_gated, dict)
        and int(count_gated.get("harms", 1)) == 0
        and int(count_gated.get("improvements", 0) or 0) >= 10
        and float(count_gated.get("override_precision", 0.0) or 0.0) >= 0.999
        and float(count_gated.get("accuracy", 0.0) or 0.0)
        >= float(count_decision.get("deterministic_top1", 0.0) or 0.0) + 0.02
        and float(count_training.get("deployment_max_probability_delta", 1.0) or 1.0) <= 1e-10
        and isinstance(count_thresholds, dict)
        and float(count_thresholds.get("probability_floor", 0.0) or 0.0) == DEFAULT_POLICY.measure_count_probability_floor
        and float(count_thresholds.get("margin_floor", 0.0) or 0.0) == DEFAULT_POLICY.measure_count_margin_floor
        and isinstance(count_config, dict)
        and int(count_config.get("trees", 10**9) or 10**9) <= 96
        and isinstance(count_training.get("independent_confirmation", {}), dict)
        and int(count_training.get("independent_confirmation", {}).get("gated", {}).get("harms", 1)) == 0,
        "family-balanced count resolver improves grouped hard cases with zero harmful audited overrides",
    )

    local_barline = read_json(source_root / "training" / "barline_classifier_report_v2.json", {})
    local_test = local_barline.get("frozen_test", {}) if isinstance(local_barline, dict) else {}
    local_baseline = local_barline.get("baseline_same_frozen_test", {}) if isinstance(local_barline, dict) else {}
    local_baseline_runtime = local_baseline.get("at_rc5_runtime_threshold", {}) if isinstance(local_baseline, dict) else {}
    local_deployment = local_barline.get("deployment_parity", {}) if isinstance(local_barline, dict) else {}
    add(
        "barline-local:cpu-training",
        local_barline.get("model_version") == "scorescan-barline-forest-2"
        and isinstance(local_test, dict)
        and float(local_test.get("roc_auc", 0.0) or 0.0) >= 0.98
        and float(local_test.get("precision", 0.0) or 0.0) >= 0.90
        and float(local_test.get("recall", 0.0) or 0.0) >= 0.92
        and isinstance(local_baseline_runtime, dict)
        and int(local_test.get("false_accepts", 10**9) or 10**9)
        < int(local_baseline_runtime.get("false_accepts", 0) or 0)
        and isinstance(local_deployment, dict)
        and float(local_deployment.get("max_absolute_probability_delta", 1.0) or 1.0) <= 1e-10,
        "local barline forest improves the same proposal holdout and matches deployment inference",
    )

    barline_pipeline = read_json(source_root / "training" / "barline_pipeline_report_v2.json", {})
    pipeline_baseline = barline_pipeline.get("baseline", {}).get("frozen_test", {}) if isinstance(barline_pipeline, dict) else {}
    pipeline_candidate = barline_pipeline.get("candidate", {}).get("frozen_test", {}) if isinstance(barline_pipeline, dict) else {}
    pipeline_comparison = barline_pipeline.get("candidate", {}).get("comparison_to_baseline", {}) if isinstance(barline_pipeline, dict) else {}
    pipeline_proposal = barline_pipeline.get("proposal_recall", {}) if isinstance(barline_pipeline, dict) else {}
    add(
        "barline-local:pipeline-holdout",
        isinstance(pipeline_baseline, dict)
        and isinstance(pipeline_candidate, dict)
        and float(pipeline_candidate.get("exact_measure_count_rate", 0.0) or 0.0)
        >= float(pipeline_baseline.get("exact_measure_count_rate", 0.0) or 0.0) + 0.20
        and float(pipeline_candidate.get("true_boundary_recall", 0.0) or 0.0)
        > float(pipeline_baseline.get("true_boundary_recall", 0.0) or 0.0)
        and float(pipeline_candidate.get("boundary_precision", 0.0) or 0.0)
        > float(pipeline_baseline.get("boundary_precision", 0.0) or 0.0)
        and int(pipeline_candidate.get("total_boundary_errors", 10**9) or 10**9)
        <= int(pipeline_baseline.get("total_boundary_errors", 0) or 0) * 0.50
        and isinstance(pipeline_comparison, dict)
        and int(pipeline_comparison.get("net_boundary_error_change", 0) or 0) < 0
        and int(pipeline_comparison.get("systems_with_more_boundary_errors", 10**9) or 10**9) <= 18
        and isinstance(pipeline_proposal, dict)
        and float(pipeline_proposal.get("frozen_test", 0.0) or 0.0) >= 0.94,
        "local/geometry/sequence pipeline improves grouped frozen layout with bounded regressions",
    )

    barline_training = read_json(source_root / "training" / "barline_sequence_classifier_report_v2.json", {})
    barline_test = barline_training.get("test", {}) if isinstance(barline_training, dict) else {}
    barline_sequence = barline_training.get("sequence_test", {}) if isinstance(barline_training, dict) else {}
    barline_baseline = barline_training.get("baseline_v1_on_same_test", {}) if isinstance(barline_training, dict) else {}
    baseline_sequence = barline_baseline.get("sequence_test", {}) if isinstance(barline_baseline, dict) else {}
    barline_deployment = barline_training.get("deployment_parity", {}) if isinstance(barline_training, dict) else {}
    add(
        "barline-sequence:cpu-training",
        isinstance(barline_test, dict)
        and float(barline_test.get("roc_auc", 0.0) or 0.0) >= 0.94
        and isinstance(barline_deployment, dict)
        and float(barline_deployment.get("max_absolute_probability_delta", 1.0) or 1.0) <= 1e-10
        and isinstance(barline_sequence, dict)
        and isinstance(baseline_sequence, dict)
        and float(barline_sequence.get("refined_exact_measure_count_rate", 0.0) or 0.0)
        > float(baseline_sequence.get("refined_exact_measure_count_rate", 0.0) or 0.0)
        and float(barline_sequence.get("false_candidate_removal", 0.0) or 0.0)
        > float(baseline_sequence.get("false_candidate_removal", 0.0) or 0.0)
        and float(barline_sequence.get("true_boundary_retention", 0.0) or 0.0)
        >= float(baseline_sequence.get("true_boundary_retention", 0.0) or 0.0)
        and int(barline_sequence.get("harmed_systems", 1) or 0) == 0,
        "barline sequence v2 improves the same grouped holdout without harming an exact system",
    )
    barline_real = read_json(source_root / "training" / "barline_sequence_real_scan_eval_v2.json", {})
    runtime = barline_real.get("runtime", {}) if isinstance(barline_real, dict) else {}
    add(
        "barline-sequence:real-regression",
        bool(barline_real.get("removed_false_split_regression"))
        and bool(barline_real.get("page_measure_count_preserved"))
        and isinstance(runtime, dict)
        and runtime.get("model_version") == "scorescan-barline-sequence-gbdt-2"
        and runtime.get("removed_candidates") == [202]
        and runtime.get("retained_candidates") == [324, 539, 754, 964, 1148, 1318],
        "maintained Allegretto trace removes the opening false stem and preserves six boundaries",
    )

    selection_training = read_json(source_root / "training" / "selection_risk_report_v4.json", {})
    selection_frozen = (
        selection_training.get("frozen_test", {})
        if isinstance(selection_training, dict)
        else {}
    )
    selection_policy = (
        selection_frozen.get("v4_policy", {})
        if isinstance(selection_frozen, dict)
        else {}
    )
    selection_baseline = (
        selection_frozen.get("baseline_v3_policy_same_test", {})
        if isinstance(selection_frozen, dict)
        else {}
    )
    selection_confirmation = (
        selection_training.get("independent_confirmation", {})
        if isinstance(selection_training, dict)
        else {}
    )
    confirmation_policy = (
        selection_confirmation.get("v4_policy", {})
        if isinstance(selection_confirmation, dict)
        else {}
    )
    confirmation_baseline = (
        selection_confirmation.get("baseline_v3_policy_same_confirmation", {})
        if isinstance(selection_confirmation, dict)
        else {}
    )
    selection_ablation = (
        selection_training.get("family_feature_ablation", {})
        if isinstance(selection_training, dict)
        else {}
    )
    ablation_policy = (
        selection_ablation.get("policy", {})
        if isinstance(selection_ablation, dict)
        else {}
    )
    selection_deployment = (
        selection_training.get("deployment_parity", {})
        if isinstance(selection_training, dict)
        else {}
    )
    selection_scenarios = (
        selection_frozen.get("by_scenario", {})
        if isinstance(selection_frozen, dict)
        else {}
    )
    localized_clear = selection_scenarios.get("localized-clear-gain", {}) if isinstance(selection_scenarios, dict) else {}
    localized_confident_trap = selection_scenarios.get("localized-confident-trap", {}) if isinstance(selection_scenarios, dict) else {}
    localized_partial_trap = selection_scenarios.get("localized-partial-trap", {}) if isinstance(selection_scenarios, dict) else {}
    add(
        "selection-risk:cpu-training",
        selection_training.get("model_version") == "scorescan-selection-risk-forest-4"
        and isinstance(selection_policy, dict)
        and float(selection_policy.get("precision", 0.0) or 0.0) >= 0.995
        and float(selection_policy.get("coverage", 0.0) or 0.0) >= 0.20
        and int(selection_policy.get("false_accepts", 10**9) or 10**9) <= 1
        and isinstance(selection_baseline, dict)
        and float(selection_policy.get("coverage", 0.0) or 0.0)
        >= float(selection_baseline.get("coverage", 0.0) or 0.0) + 0.04
        and float(selection_policy.get("total_semantic_gain", 0.0) or 0.0)
        >= float(selection_baseline.get("total_semantic_gain", 0.0) or 0.0) + 10.0
        and int(selection_policy.get("false_accepts", 10**9) or 10**9)
        <= int(selection_baseline.get("false_accepts", 10**9) or 10**9)
        and isinstance(confirmation_policy, dict)
        and float(confirmation_policy.get("precision", 0.0) or 0.0) >= 0.999
        and float(confirmation_policy.get("coverage", 0.0) or 0.0) >= 0.20
        and int(confirmation_policy.get("false_accepts", 1) or 0) == 0
        and isinstance(confirmation_baseline, dict)
        and float(confirmation_policy.get("coverage", 0.0) or 0.0)
        >= float(confirmation_baseline.get("coverage", 0.0) or 0.0) + 0.04
        and isinstance(localized_clear, dict)
        and float(localized_clear.get("precision", 0.0) or 0.0) >= 0.999
        and float(localized_clear.get("coverage", 0.0) or 0.0) >= 0.50
        and isinstance(localized_confident_trap, dict)
        and int(localized_confident_trap.get("accepted", 1) or 0) == 0
        and isinstance(localized_partial_trap, dict)
        and int(localized_partial_trap.get("accepted", 1) or 0) == 0
        and isinstance(selection_deployment, dict)
        and float(selection_deployment.get("max_absolute_probability_delta", 1.0) or 1.0) <= 1e-10,
        "replacement verifier v4 expands to five independent families while rejecting confident and partial localized traps",
    )


    selection_policy_audit = read_json(
        source_root / "training" / "selection_risk_policy_audit_v5.json",
        {},
    )
    selection_assertions = (
        selection_policy_audit.get("release_assertions", {})
        if isinstance(selection_policy_audit, dict)
        else {}
    )
    policy_frozen = (
        selection_policy_audit.get("frozen_test", {})
        if isinstance(selection_policy_audit, dict)
        else {}
    )
    policy_safety = (
        selection_policy_audit.get("safety_corpus", {})
        if isinstance(selection_policy_audit, dict)
        else {}
    )
    policy_confirmation = (
        selection_policy_audit.get("independent_confirmation", {})
        if isinstance(selection_policy_audit, dict)
        else {}
    )

    def policy_result(section: object, key: str) -> dict[str, object]:
        if not isinstance(section, dict):
            return {}
        value = section.get(key, {})
        return value if isinstance(value, dict) else {}

    frozen_v23 = policy_result(policy_frozen, "policy_0_23")
    frozen_stored = policy_result(policy_frozen, "stored_exact_threshold_control")
    safety_v23 = policy_result(policy_safety, "policy_0_23")
    safety_stored = policy_result(policy_safety, "stored_exact_threshold_control")
    confirmation_v23 = policy_result(policy_confirmation, "policy_0_23")
    confirmation_stored = policy_result(
        policy_confirmation, "stored_exact_threshold_control"
    )
    add(
        "selection-risk:policy-45-audit",
        selection_policy_audit.get("model_version")
        == "scorescan-selection-risk-forest-4"
        and selection_policy_audit.get("policy_version") == "scorescan-policy-45"
        and selection_policy_audit.get("workflow_version")
        == "single-staff-printed-scan@51"
        and isinstance(selection_assertions, dict)
        and all(bool(value) for value in selection_assertions.values())
        and int(frozen_v23.get("false_accepts", 1) or 0) == 0
        and int(safety_v23.get("false_accepts", 1) or 0) == 0
        and int(confirmation_v23.get("false_accepts", 1) or 0) == 0
        and int(frozen_v23.get("true_accepts", 0) or 0)
        >= int(frozen_stored.get("true_accepts", 0) or 0) + 10
        and int(safety_v23.get("true_accepts", 0) or 0)
        >= int(safety_stored.get("true_accepts", 0) or 0) + 100
        and int(confirmation_v23.get("true_accepts", 0) or 0)
        >= int(confirmation_stored.get("true_accepts", 0) or 0) + 100,
        "policy-45 eliminates audited invalid/cross-family false accepts and recovers exact-majority coverage without harm",
    )

    preservation_audit = read_json(
        source_root / "training" / "measure_preservation_consensus_audit_v1.json",
        {},
    )
    preservation_three_way = (
        preservation_audit.get("three_way_conflict", {})
        if isinstance(preservation_audit, dict)
        else {}
    )
    preservation_support = (
        preservation_audit.get("two_family_preservation_support", {})
        if isinstance(preservation_audit, dict)
        else {}
    )
    add(
        "consensus:whole-measure-preservation-gate",
        preservation_audit.get("version") == APP_VERSION
        and preservation_audit.get("workflow_version") == WORKFLOW_VERSION
        and preservation_audit.get("policy_version") == DEFAULT_POLICY.version
        and preservation_audit.get("passed") is True
        and int(preservation_audit.get("legacy_false_collapse_count", 0) or 0) >= 6
        and isinstance(preservation_three_way, dict)
        and preservation_three_way.get("old_exact_majority_would_form") is True
        and preservation_three_way.get("new_exact_majority_would_form") is False
        and isinstance(preservation_support, dict)
        and preservation_support.get("accepted") is True
        and int(preservation_support.get("support", 0) or 0)
        >= DEFAULT_POLICY.selection_semantic_preservation_minimum_families,
        "whole-measure copy requires independent support for the complete normalized MusicXML surface, not only Score IR semantics",
    )

    preservation_veto_report = read_json(
        source_root / "training" / "experiments" / "rejected_measure_preservation_veto_v1_report.json",
        {},
    )
    preservation_veto_frozen = (
        preservation_veto_report.get("frozen", {})
        if isinstance(preservation_veto_report, dict)
        else {}
    )
    preservation_veto_confirmation = (
        preservation_veto_report.get("confirmation", {})
        if isinstance(preservation_veto_report, dict)
        else {}
    )
    preservation_veto_repro = read_json(
        source_root
        / "training"
        / "experiments"
        / "rejected_measure_preservation_veto_v1_reproducibility.json",
        {},
    )
    add(
        "consensus:preservation-veto-experiment-rejected",
        preservation_veto_report.get("model_version")
        == "scorescan-measure-preservation-veto-experiment-1"
        and preservation_veto_report.get("runtime_deployed") is False
        and isinstance(preservation_veto_frozen, dict)
        and isinstance(preservation_veto_confirmation, dict)
        and int(preservation_veto_frozen.get("error_accepts", 0) or 0) >= 1
        and int(preservation_veto_confirmation.get("error_accepts", 0) or 0) >= 1
        and preservation_veto_report.get("independent_correct_coverage_gain") is False
        and preservation_veto_repro.get("model_byte_identical") is True
        and preservation_veto_repro.get("report_byte_identical") is True
        and preservation_veto_repro.get("passed") is True,
        "CPU preservation-veto experiment is reproducible but rejected because it retains false accepts and adds no independent correct coverage",
    )

    measure_rescue_audit = read_json(
        source_root / "training" / "measure_localized_rescue_policy_audit_v1.json",
        {},
    )
    measure_rescue_assertions = (
        measure_rescue_audit.get("release_assertions", {})
        if isinstance(measure_rescue_audit, dict)
        else {}
    )
    measure_rescue_sections = (
        measure_rescue_audit.get("sections", {})
        if isinstance(measure_rescue_audit, dict)
        else {}
    )
    measure_rescue_confirmation = (
        measure_rescue_sections.get("independent_confirmation", {})
        if isinstance(measure_rescue_sections, dict)
        else {}
    )
    measure_rescue_v4 = (
        measure_rescue_confirmation.get("production_v4", {})
        if isinstance(measure_rescue_confirmation, dict)
        else {}
    )
    measure_rescue_v6 = (
        measure_rescue_confirmation.get("candidate_v6", {})
        if isinstance(measure_rescue_confirmation, dict)
        else {}
    )
    add(
        "measure-localized:policy-audit",
        measure_rescue_audit.get("production_model")
        == "scorescan-selection-risk-forest-4"
        and measure_rescue_audit.get("candidate_model")
        == "scorescan-selection-risk-forest-6"
        and measure_rescue_audit.get("policy_version") == "scorescan-policy-52"
        and measure_rescue_audit.get("workflow_version")
        == "single-staff-printed-scan@58"
        and measure_rescue_audit.get("decision")
        == "retain production v4; reject v6"
        and isinstance(measure_rescue_assertions, dict)
        and all(bool(value) for value in measure_rescue_assertions.values())
        and isinstance(measure_rescue_v4, dict)
        and isinstance(measure_rescue_v6, dict)
        and int(measure_rescue_v4.get("false_accepts", 1) or 0) == 0
        and int(measure_rescue_v4.get("true_accepts", 0) or 0)
        > int(measure_rescue_v6.get("true_accepts", 0) or 0)
        and int(measure_rescue_v6.get("false_accepts", 0) or 0) >= 1,
        "measure-localised third-family evidence is accepted by production v4 with zero audited trap accepts; trained v6 is reproducibly rejected",
    )

    localized_internal_audit = read_json(
        source_root / "training" / "measure_localized_internal_consensus_audit_v1.json",
        {},
    )
    localized_internal_capabilities = (
        localized_internal_audit.get("accepted_capabilities", {})
        if isinstance(localized_internal_audit, dict)
        else {}
    )
    add(
        "measure-localized:internal-exact-consensus",
        localized_internal_audit.get("policy_version") == DEFAULT_POLICY.version
        and int(localized_internal_audit.get("independent_family_count_contributed", 0) or 0) == 1
        and int(localized_internal_audit.get("failed", 1) or 0) == 0
        and int(localized_internal_audit.get("passed", 0) or 0)
        == int(localized_internal_audit.get("scenario_count", -1) or -1)
        and isinstance(localized_internal_capabilities, dict)
        and all(bool(value) for value in localized_internal_capabilities.values())
        and localized_internal_audit.get("end_to_end_accuracy_claim") is False,
        "three related local treatments form one strict internal family and fail closed on splits or insufficient evidence",
    )

    localized_exact_audit = read_json(
        source_root / "training" / "measure_localized_exact_content_audit_v1.json",
        {},
    )
    localized_exact_capabilities = (
        localized_exact_audit.get("accepted_capabilities", {})
        if isinstance(localized_exact_audit, dict)
        else {}
    )
    add(
        "measure-localized:normalized-splice-content",
        localized_exact_audit.get("policy_version") == DEFAULT_POLICY.version
        and localized_exact_audit.get("permission_signature")
        == "normalized-splice-content-c14n-v1"
        and localized_exact_audit.get("diagnostic_signature")
        == "score-ir-semantic-v1"
        and int(localized_exact_audit.get("failed", 1) or 0) == 0
        and int(localized_exact_audit.get("passed", 0) or 0)
        == int(localized_exact_audit.get("scenario_count", -1) or -1)
        and isinstance(localized_exact_capabilities, dict)
        and all(bool(value) for value in localized_exact_capabilities.values())
        and localized_exact_audit.get("end_to_end_accuracy_claim") is False,
        "local-family equality covers actual splice content including beams/stems/unmodelled notation while normalising divisions and layout only",
    )

    localized_context_audit = read_json(
        source_root / "training" / "measure_localized_context_contract_audit_v1.json",
        {},
    )
    localized_context_scenarios = (
        localized_context_audit.get("scenarios", [])
        if isinstance(localized_context_audit, dict)
        else []
    )
    localized_context_by_name = {
        str(item.get("name")): item
        for item in localized_context_scenarios
        if isinstance(item, dict)
    }
    add(
        "measure-localized:notation-context-contract",
        localized_context_audit.get("version") == APP_VERSION
        and localized_context_audit.get("workflow_version") == WORKFLOW_VERSION
        and localized_context_audit.get("policy_version") == DEFAULT_POLICY.version
        and localized_context_audit.get("contract")
        == "measure-localized-notation-context@1"
        and localized_context_audit.get("passed") is True
        and int(localized_context_audit.get("scenario_count", 0) or 0) >= 10
        and int(localized_context_audit.get("passed_count", 0) or 0)
        == int(localized_context_audit.get("scenario_count", -1) or -1)
        and localized_context_by_name.get("implicit-default", {}).get("valid") is True
        and localized_context_by_name.get("matching-nondefault", {}).get("valid") is True
        and localized_context_by_name.get("conflicting-clef", {}).get("valid") is False
        and localized_context_by_name.get("conflicting-key", {}).get("valid") is False
        and localized_context_by_name.get("conflicting-time", {}).get("valid") is False
        and localized_context_by_name.get("conflicting-transpose", {}).get("valid") is False
        and localized_context_by_name.get("missing-nondefault-clef", {}).get("valid") is False
        and localized_context_by_name.get("local-mid-measure-context", {}).get("valid") is False
        and localized_context_by_name.get("template-mid-measure-context", {}).get("valid") is False,
        "crop-local clef/key/time/transpose must match inherited page context before the local family may vote",
    )

    localized_exact_veto_report = read_json(
        source_root
        / "training"
        / "experiments"
        / "rejected_measure_localized_exact_veto_v2_report.json",
        {},
    )
    localized_exact_veto_confirmation = (
        localized_exact_veto_report.get("confirmation", {})
        if isinstance(localized_exact_veto_report, dict)
        else {}
    )
    localized_exact_veto_reproducibility = read_json(
        source_root / "training" / "measure_localized_exact_veto_reproducibility_v2.json",
        {},
    )
    add(
        "measure-localized:exact-veto-v2-rejection",
        localized_exact_veto_report.get("model_version")
        == "scorescan-measure-localized-exact-veto-experiment-2"
        and localized_exact_veto_report.get("runtime_deployed") is False
        and isinstance(localized_exact_veto_confirmation, dict)
        and int(localized_exact_veto_confirmation.get("error_accepts", 0) or 0) >= 1
        and localized_exact_veto_reproducibility.get("model_reproducible") is True
        and localized_exact_veto_reproducibility.get("report_reproducible") is True,
        "CPU exact-content veto experiment is byte-reproducible but rejected because correlated local errors remain and deterministic XML equality is authoritative",
    )

    localized_veto_report = read_json(
        source_root
        / "training"
        / "experiments"
        / "rejected_measure_localized_internal_veto_v1_report.json",
        {},
    )
    localized_veto_confirmation = (
        localized_veto_report.get("confirmation", {})
        if isinstance(localized_veto_report, dict)
        else {}
    )
    localized_veto_reproducibility = read_json(
        source_root / "training" / "measure_localized_internal_veto_reproducibility_v1.json",
        {},
    )
    add(
        "measure-localized:internal-veto-rejection",
        localized_veto_report.get("model_version")
        == "scorescan-measure-localized-internal-veto-experiment-1"
        and localized_veto_report.get("runtime_deployed") is False
        and isinstance(localized_veto_confirmation, dict)
        and int(localized_veto_confirmation.get("error_accepts", 0) or 0) >= 1
        and localized_veto_reproducibility.get("model_reproducible") is True
        and localized_veto_reproducibility.get("report_reproducible") is True
        and localized_veto_reproducibility.get("runtime_deployed") is False,
        "CPU internal-veto experiment is byte-reproducible but rejected because it retains an audited error and overlaps the production verifier",
    )

    rejected_selection_v5 = read_json(
        source_root
        / "training"
        / "experiments"
        / "rejected_selection_risk_v5_runtime_audit.json",
        {},
    )
    rejected_v5_policy = (
        rejected_selection_v5.get("selected_v5_policy", {})
        if isinstance(rejected_selection_v5, dict)
        else {}
    )
    retained_v4_policy = (
        rejected_selection_v5.get("production_v4_same_confirmation", {})
        if isinstance(rejected_selection_v5, dict)
        else {}
    )
    add(
        "selection-risk:v5-rejection-audit",
        rejected_selection_v5.get("decision") == "rejected_from_runtime"
        and isinstance(rejected_v5_policy, dict)
        and isinstance(retained_v4_policy, dict)
        and int(rejected_v5_policy.get("false_accepts", 1) or 0) == 0
        and int(retained_v4_policy.get("false_accepts", 1) or 0) == 0
        and int(retained_v4_policy.get("true_accepts", 0) or 0)
        > int(rejected_v5_policy.get("true_accepts", 0) or 0),
        "trained v5 experiment is retained as a rejected audit because it did not beat the smaller production v4 policy",
    )

    pitch_training = read_json(
        source_root / "training" / "pitch_patch_calibrator_report_v4.json",
        {},
    )
    pitch_frozen = (
        pitch_training.get("frozen_test", {})
        if isinstance(pitch_training, dict)
        else {}
    )
    pitch_no_visual_frozen = (
        pitch_frozen.get("no_visual_policy", {})
        if isinstance(pitch_frozen, dict)
        else {}
    )
    pitch_confirmation = (
        pitch_training.get("independent_confirmation", {})
        if isinstance(pitch_training, dict)
        else {}
    )
    pitch_no_visual_confirmation = (
        pitch_confirmation.get("no_visual_policy", {})
        if isinstance(pitch_confirmation, dict)
        else {}
    )
    pitch_large_no_visual = (
        pitch_training.get("large_no_visual_confirmation", {})
        if isinstance(pitch_training, dict)
        else {}
    )
    pitch_large_no_visual_policy = (
        pitch_large_no_visual.get("policy", {})
        if isinstance(pitch_large_no_visual, dict)
        else {}
    )
    pitch_deployment = (
        pitch_training.get("deployment_parity", {})
        if isinstance(pitch_training, dict)
        else {}
    )
    add(
        "pitch-patch:no-visual-cpu-training",
        pitch_training.get("model_version") == "scorescan-pitch-patch-forest-4"
        and isinstance(pitch_no_visual_frozen, dict)
        and int(pitch_no_visual_frozen.get("false_accepts", 1) or 0) == 0
        and float(pitch_no_visual_frozen.get("positive_recall", 0.0) or 0.0) >= 0.95
        and float(pitch_no_visual_frozen.get("threshold", 0.0) or 0.0)
        >= DEFAULT_POLICY.pitch_patch_no_visual_probability_floor
        and isinstance(pitch_no_visual_confirmation, dict)
        and int(pitch_no_visual_confirmation.get("false_accepts", 1) or 0) == 0
        and float(pitch_no_visual_confirmation.get("positive_recall", 0.0) or 0.0) >= 0.97
        and isinstance(pitch_large_no_visual, dict)
        and int(pitch_large_no_visual.get("eligible_rows", 0) or 0) >= 3_500
        and int(pitch_large_no_visual.get("negative_rows", 0) or 0) >= 1_700
        and isinstance(pitch_large_no_visual_policy, dict)
        and int(pitch_large_no_visual_policy.get("false_accepts", 1) or 0) == 0
        and float(pitch_large_no_visual_policy.get("positive_recall", 0.0) or 0.0) >= 0.97
        and isinstance(pitch_deployment, dict)
        and float(pitch_deployment.get("max_absolute_probability_delta", 1.0) or 1.0) <= 1e-10
        and int(pitch_training.get("model_bytes", 10**9) or 10**9) <= 2_000_000,
        "pitch-only family consensus retains a separately calibrated no-source-crop path with zero maintained false accepts",
    )

    pitch_visual_training = read_json(
        source_root / "training" / "pitch_visual_guard_report_v2.json",
        {},
    )
    pitch_visual_frozen = (
        pitch_visual_training.get("frozen_test", {})
        if isinstance(pitch_visual_training, dict)
        else {}
    )
    pitch_visual_frozen_policy = (
        pitch_visual_frozen.get("policy", {})
        if isinstance(pitch_visual_frozen, dict)
        else {}
    )
    pitch_visual_confirmation = (
        pitch_visual_training.get("independent_confirmation", {})
        if isinstance(pitch_visual_training, dict)
        else {}
    )
    pitch_visual_confirmation_policy = (
        pitch_visual_confirmation.get("policy", {})
        if isinstance(pitch_visual_confirmation, dict)
        else {}
    )
    pitch_visual_deployment = (
        pitch_visual_training.get("deployment_parity", {})
        if isinstance(pitch_visual_training, dict)
        else {}
    )
    pitch_visual_audit = read_json(
        source_root / "training" / "pitch_visual_guard_independent_audit_v2.json",
        {},
    )
    production_pitch_gate = (
        pitch_visual_audit.get("production_and_gate", {})
        if isinstance(pitch_visual_audit, dict)
        else {}
    )
    baseline_pitch_gate = read_json(
        source_root / "training" / "pitch_visual_guard_independent_audit_v1.json",
        {},
    )
    baseline_production_pitch_gate = (
        baseline_pitch_gate.get("production_and_gate", {})
        if isinstance(baseline_pitch_gate, dict)
        else {}
    )
    pitch_single_detector_rejection = read_json(
        source_root / "training" / "experiments" / "rejected_pitch_visual_single_detector_v1.json",
        {},
    )
    pitch_large_tail_rejection = read_json(
        source_root / "training" / "experiments" / "rejected_pitch_visual_large_tail_v2.json",
        {},
    )
    add(
        "pitch-patch:source-crop-visual-gate",
        pitch_visual_training.get("model_version") == "scorescan-pitch-visual-guard-2"
        and isinstance(pitch_visual_frozen_policy, dict)
        and int(pitch_visual_frozen_policy.get("false_accepts", 1) or 0) == 0
        and float(pitch_visual_frozen_policy.get("positive_recall", 0.0) or 0.0) >= 0.10
        and isinstance(pitch_visual_confirmation_policy, dict)
        and int(pitch_visual_confirmation_policy.get("false_accepts", 1) or 0) == 0
        and float(pitch_visual_confirmation_policy.get("positive_recall", 0.0) or 0.0) >= 0.12
        and isinstance(pitch_visual_deployment, dict)
        and float(pitch_visual_deployment.get("max_absolute_probability_delta", 1.0) or 1.0) <= 1e-10
        and pitch_visual_audit.get("pitch_model_version") == "scorescan-pitch-patch-forest-4"
        and pitch_visual_audit.get("visual_model_version") == "scorescan-pitch-visual-guard-2"
        and int(pitch_visual_audit.get("rendered_groups", 0) or 0) >= 800
        and isinstance(production_pitch_gate, dict)
        and int(production_pitch_gate.get("false_accepts", 1) or 0) == 0
        and int(production_pitch_gate.get("true_accepts", 0) or 0) >= 90
        and float(production_pitch_gate.get("positive_recall", 0.0) or 0.0) >= 0.11
        and isinstance(baseline_production_pitch_gate, dict)
        and int(production_pitch_gate.get("true_accepts", 0) or 0)
        >= int(baseline_production_pitch_gate.get("true_accepts", 0) or 0) + 35
        and int(pitch_visual_training.get("model_bytes", 10**9) or 10**9) <= 2_500_000,
        "source-crop pitch repair requires serial high-level and dual-detector visual approval with zero maintained rendered false accepts",
    )
    add(
        "pitch-patch:rejected-visual-experiments",
        pitch_single_detector_rejection.get("decision") == "rejected"
        and pitch_single_detector_rejection.get("runtime_deployed") is False
        and int(pitch_single_detector_rejection.get("observed_false_accepts", 0) or 0) == 3
        and pitch_large_tail_rejection.get("decision") == "rejected"
        and pitch_large_tail_rejection.get("runtime_deployed") is False
        and int(pitch_large_tail_rejection.get("observed_false_accepts", 0) or 0) == 6,
        "single-detector and unstable larger-tail pitch-visual experiments are retained as explicit non-runtime rejection evidence",
    )

    accidental_presence_training = read_json(
        source_root / "training" / "accidental_presence_guard_report_v1.json", {}
    )
    accidental_frozen = (
        accidental_presence_training.get("frozen_test", {})
        if isinstance(accidental_presence_training, dict)
        else {}
    )
    accidental_frozen_policy = (
        accidental_frozen.get("policy", {}) if isinstance(accidental_frozen, dict) else {}
    )
    accidental_safety = (
        accidental_presence_training.get("safety_calibration", {})
        if isinstance(accidental_presence_training, dict)
        else {}
    )
    accidental_safety_policy = (
        accidental_safety.get("policy", {}) if isinstance(accidental_safety, dict) else {}
    )
    accidental_independent = (
        accidental_presence_training.get("independent_test", {})
        if isinstance(accidental_presence_training, dict)
        else {}
    )
    accidental_independent_policy = (
        accidental_independent.get("policy", {})
        if isinstance(accidental_independent, dict)
        else {}
    )
    accidental_independent_sample = (
        accidental_independent.get("sample", {})
        if isinstance(accidental_independent, dict)
        else {}
    )
    accidental_ablation = (
        accidental_presence_training.get("ablation_density_only", {})
        if isinstance(accidental_presence_training, dict)
        else {}
    )
    accidental_ablation_independent = (
        accidental_ablation.get("independent_test_policy", {})
        if isinstance(accidental_ablation, dict)
        else {}
    )
    accidental_deployment = (
        accidental_presence_training.get("deployment_parity", {})
        if isinstance(accidental_presence_training, dict)
        else {}
    )
    accidental_reproducibility = read_json(
        source_root / "training" / "accidental_presence_guard_reproducibility_v1.json", {}
    )
    accidental_model = resources / "accidental_presence_guard.json"
    add(
        "pitch-patch:accidental-presence-gate",
        accidental_presence_training.get("model_version")
        == "scorescan-accidental-presence-forest-1"
        and int(accidental_presence_training.get("groups", 0) or 0) >= 800
        and isinstance(accidental_frozen_policy, dict)
        and int(accidental_frozen_policy.get("false_accepts", 1) or 0) == 0
        and min(
            float(accidental_frozen_policy.get("present_recall", 0.0) or 0.0),
            float(accidental_frozen_policy.get("absent_recall", 0.0) or 0.0),
        ) >= 0.47
        and int(accidental_safety.get("groups", 0) or 0) >= 900
        and isinstance(accidental_safety_policy, dict)
        and int(accidental_safety_policy.get("false_accepts", 1) or 0) == 0
        and min(
            float(accidental_safety_policy.get("present_recall", 0.0) or 0.0),
            float(accidental_safety_policy.get("absent_recall", 0.0) or 0.0),
        ) >= 0.47
        and int(accidental_independent.get("groups", 0) or 0) >= 900
        and isinstance(accidental_independent_policy, dict)
        and int(accidental_independent_policy.get("false_accepts", 1) or 0) == 0
        and min(
            float(accidental_independent_policy.get("present_recall", 0.0) or 0.0),
            float(accidental_independent_policy.get("absent_recall", 0.0) or 0.0),
        ) >= 0.47
        and isinstance(accidental_independent_sample, dict)
        and float(accidental_independent_sample.get("roc_auc", 0.0) or 0.0) >= 0.93
        and isinstance(accidental_ablation_independent, dict)
        and int(accidental_ablation_independent.get("false_accepts", 0) or 0) >= 1
        and isinstance(accidental_deployment, dict)
        and float(accidental_deployment.get("max_absolute_probability_delta", 1.0)) <= 1e-10
        and accidental_reproducibility.get("all_byte_identical") is True
        and accidental_reproducibility.get("dataset_arrays_identical") is True
        and accidental_reproducibility.get("end_to_end_accuracy_claim") is False
        and all(
            isinstance(item, dict) and item.get("byte_identical") is True
            for item in accidental_reproducibility.get("artifacts", {}).values()
        )
        and accidental_model.is_file()
        and accidental_model.stat().st_size <= 1_800_000,
        "same-position chromatic pitch repair is additionally vetoed by a reproducible binary printed-accidental presence model with zero maintained rendered false accepts",
    )

    accidental_rejections = read_json(
        source_root / "training" / "accidental_symbol_rejected_experiments_v1.json", {}
    )
    rejected_accidental_experiments = (
        accidental_rejections.get("experiments", [])
        if isinstance(accidental_rejections, dict)
        else []
    )
    accepted_accidental_replacement = (
        accidental_rejections.get("accepted_replacement", {})
        if isinstance(accidental_rejections, dict)
        else {}
    )
    add(
        "pitch-patch:rejected-accidental-class-experiments",
        accidental_rejections.get("deployed") is False
        and len(rejected_accidental_experiments) >= 2
        and all(
            isinstance(item, dict) and item.get("status") == "rejected"
            for item in rejected_accidental_experiments
        )
        and accepted_accidental_replacement.get("model_version")
        == "scorescan-accidental-presence-forest-1"
        and "class substitution"
        in str(accepted_accidental_replacement.get("excluded", "")),
        "unsafe broad accidental-class models remain rejected and explicit sharp/flat/natural substitutions remain review-only",
    )

    tie_visual_training = read_json(
        source_root / "training" / "tie_visual_guard_report_v1.json", {}
    )
    tie_visual_frozen = (
        tie_visual_training.get("frozen_test", {})
        if isinstance(tie_visual_training, dict)
        else {}
    )
    tie_visual_frozen_policy = (
        tie_visual_frozen.get("policy", {}) if isinstance(tie_visual_frozen, dict) else {}
    )
    tie_visual_safety = (
        tie_visual_training.get("safety_calibration", {})
        if isinstance(tie_visual_training, dict)
        else {}
    )
    tie_visual_safety_policy = (
        tie_visual_safety.get("policy", {}) if isinstance(tie_visual_safety, dict) else {}
    )
    tie_visual_tail = (
        tie_visual_training.get("tail_safety_calibration", {})
        if isinstance(tie_visual_training, dict)
        else {}
    )
    tie_visual_tail_policy = (
        tie_visual_tail.get("policy", {}) if isinstance(tie_visual_tail, dict) else {}
    )
    tie_visual_independent = (
        tie_visual_training.get("independent_test", {})
        if isinstance(tie_visual_training, dict)
        else {}
    )
    tie_visual_independent_policy = (
        tie_visual_independent.get("policy", {})
        if isinstance(tie_visual_independent, dict)
        else {}
    )
    tie_visual_independent_sample = (
        tie_visual_independent.get("sample", {})
        if isinstance(tie_visual_independent, dict)
        else {}
    )
    tie_visual_ablation = (
        tie_visual_training.get("ablation_geometry_only", {})
        if isinstance(tie_visual_training, dict)
        else {}
    )
    tie_visual_ablation_sample = (
        tie_visual_ablation.get("independent_sample", {})
        if isinstance(tie_visual_ablation, dict)
        else {}
    )
    tie_visual_slur_rejection = (
        tie_visual_training.get("rejected_same_endpoint_slur_experiment", {})
        if isinstance(tie_visual_training, dict)
        else {}
    )
    tie_visual_slur_policy = (
        tie_visual_slur_rejection.get("policy_at_deployment_threshold", {})
        if isinstance(tie_visual_slur_rejection, dict)
        else {}
    )
    tie_visual_deployment = (
        tie_visual_training.get("deployment_parity", {})
        if isinstance(tie_visual_training, dict)
        else {}
    )
    tie_visual_reproducibility = read_json(
        source_root / "training" / "tie_visual_guard_reproducibility_v1.json", {}
    )
    tie_visual_rejections = read_json(
        source_root / "training" / "tie_visual_rejected_experiments_v1.json", {}
    )
    tie_visual_model = resources / "tie_visual_guard.json"
    add(
        "tie-patch:source-crop-visual-gate",
        tie_visual_training.get("model_version") == "scorescan-tie-visual-forest-1"
        and int(tie_visual_training.get("groups", 0) or 0) >= 1_600
        and float(tie_visual_training.get("threshold", 0.0) or 0.0)
        == DEFAULT_POLICY.tie_visual_guard_probability_floor
        and isinstance(tie_visual_frozen_policy, dict)
        and int(tie_visual_frozen_policy.get("false_accepts", 1) or 0) == 0
        and float(tie_visual_frozen_policy.get("present_recall", 0.0) or 0.0) >= 0.43
        and int(tie_visual_safety.get("groups", 0) or 0) >= 1_200
        and isinstance(tie_visual_safety_policy, dict)
        and int(tie_visual_safety_policy.get("false_accepts", 1) or 0) == 0
        and float(tie_visual_safety_policy.get("present_recall", 0.0) or 0.0) >= 0.44
        and int(tie_visual_tail.get("groups", 0) or 0) >= 5_200
        and isinstance(tie_visual_tail_policy, dict)
        and int(tie_visual_tail_policy.get("false_accepts", 1) or 0) == 0
        and float(tie_visual_tail_policy.get("present_recall", 0.0) or 0.0) >= 0.45
        and int(tie_visual_independent.get("groups", 0) or 0) >= 3_000
        and isinstance(tie_visual_independent_policy, dict)
        and int(tie_visual_independent_policy.get("false_accepts", 1) or 0) == 0
        and int(tie_visual_independent_policy.get("correct_accepts", 0) or 0) >= 1_300
        and float(tie_visual_independent_policy.get("present_recall", 0.0) or 0.0) >= 0.45
        and isinstance(tie_visual_independent_sample, dict)
        and float(tie_visual_independent_sample.get("roc_auc", 0.0) or 0.0) >= 0.88
        and isinstance(tie_visual_ablation_sample, dict)
        and abs(float(tie_visual_ablation_sample.get("roc_auc", 1.0) or 1.0) - 0.5) <= 1e-12
        and isinstance(tie_visual_slur_rejection, dict)
        and tie_visual_slur_rejection.get("deployed") is False
        and isinstance(tie_visual_slur_policy, dict)
        and int(tie_visual_slur_policy.get("false_accepts", 0) or 0) >= 1
        and isinstance(tie_visual_deployment, dict)
        and float(tie_visual_deployment.get("max_absolute_probability_delta", 1.0)) <= 1e-10
        and tie_visual_reproducibility.get("passed") is True
        and tie_visual_reproducibility.get("training_dataset_array_identical") is True
        and tie_visual_reproducibility.get("model_byte_identical") is True
        and tie_visual_reproducibility.get("report_byte_identical") is True
        and tie_visual_reproducibility.get("end_to_end_accuracy_claim") is False
        and tie_visual_model.is_file()
        and tie_visual_model.stat().st_size <= 4_000_000,
        "source-backed adjacent same-pitch within-measure tie additions require a reproducible local visual veto with zero maintained rendered false accepts",
    )
    rejected_tie_experiments = (
        tie_visual_rejections.get("experiments", [])
        if isinstance(tie_visual_rejections, dict)
        else []
    )
    accepted_tie_replacement = (
        tie_visual_rejections.get("accepted_replacement", {})
        if isinstance(tie_visual_rejections, dict)
        else {}
    )
    add(
        "tie-patch:rejected-broad-curve-experiments",
        tie_visual_rejections.get("deployed") is False
        and len(rejected_tie_experiments) >= 4
        and all(
            isinstance(item, dict)
            and item.get("status") == "rejected"
            and item.get("runtime_deployed") is False
            for item in rejected_tie_experiments
        )
        and any(int(item.get("observed_false_accepts", 0) or 0) >= 1 for item in rejected_tie_experiments)
        and accepted_tie_replacement.get("model_version") == "scorescan-tie-visual-forest-1"
        and "tie removal" in str(accepted_tie_replacement.get("excluded", ""))
        and "slur" in str(accepted_tie_replacement.get("excluded", "")),
        "generic arc, unsafe lower-threshold, tie-removal and same-endpoint slur experiments remain explicit non-runtime rejection evidence",
    )


    accent_visual_training = read_json(
        source_root / "training" / "accent_visual_guard_report_v1.json", {}
    )
    accent_visual_frozen = (
        accent_visual_training.get("frozen_test", {})
        if isinstance(accent_visual_training, dict)
        else {}
    )
    accent_visual_frozen_policy = (
        accent_visual_frozen.get("policy", {}) if isinstance(accent_visual_frozen, dict) else {}
    )
    accent_visual_safety = (
        accent_visual_training.get("safety_calibration", {})
        if isinstance(accent_visual_training, dict)
        else {}
    )
    accent_visual_safety_policy = (
        accent_visual_safety.get("policy", {}) if isinstance(accent_visual_safety, dict) else {}
    )
    accent_visual_independent = (
        accent_visual_training.get("independent_test", {})
        if isinstance(accent_visual_training, dict)
        else {}
    )
    accent_visual_independent_policy = (
        accent_visual_independent.get("policy", {})
        if isinstance(accent_visual_independent, dict)
        else {}
    )
    accent_visual_independent_sample = (
        accent_visual_independent.get("sample", {})
        if isinstance(accent_visual_independent, dict)
        else {}
    )
    accent_visual_ablation = (
        accent_visual_training.get("ablation_without_diagonal_geometry", {})
        if isinstance(accent_visual_training, dict)
        else {}
    )
    accent_visual_ablation_policy = (
        accent_visual_ablation.get("independent_policy", {})
        if isinstance(accent_visual_ablation, dict)
        else {}
    )
    accent_visual_deployment = (
        accent_visual_training.get("deployment_parity", {})
        if isinstance(accent_visual_training, dict)
        else {}
    )
    accent_visual_reproducibility = read_json(
        source_root / "training" / "accent_visual_guard_reproducibility_v1.json", {}
    )
    accent_visual_rejections = read_json(
        source_root / "training" / "accent_visual_rejected_experiments_v1.json", {}
    )
    accent_visual_model = resources / "accent_visual_guard.json"
    add(
        "articulation-patch:accent-source-crop-gate",
        accent_visual_training.get("model_version") == "scorescan-accent-visual-forest-1"
        and int(accent_visual_training.get("groups", 0) or 0) >= 500
        and int(accent_visual_training.get("samples", 0) or 0) >= 4_000
        and float(accent_visual_training.get("threshold", 0.0) or 0.0)
        >= DEFAULT_POLICY.accent_visual_guard_probability_floor
        and isinstance(accent_visual_frozen_policy, dict)
        and int(accent_visual_frozen_policy.get("false_accepts", 1) or 0) == 0
        and float(accent_visual_frozen_policy.get("present_recall", 0.0) or 0.0) >= 0.25
        and int(accent_visual_safety.get("groups", 0) or 0) >= 700
        and isinstance(accent_visual_safety_policy, dict)
        and int(accent_visual_safety_policy.get("false_accepts", 1) or 0) == 0
        and float(accent_visual_safety_policy.get("present_recall", 0.0) or 0.0) >= 0.29
        and int(accent_visual_independent.get("groups", 0) or 0) >= 800
        and isinstance(accent_visual_independent_policy, dict)
        and int(accent_visual_independent_policy.get("false_accepts", 1) or 0) == 0
        and int(accent_visual_independent_policy.get("correct_accepts", 0) or 0) >= 230
        and float(accent_visual_independent_policy.get("present_recall", 0.0) or 0.0) >= 0.29
        and isinstance(accent_visual_independent_sample, dict)
        and float(accent_visual_independent_sample.get("roc_auc", 0.0) or 0.0) >= 0.90
        and isinstance(accent_visual_ablation_policy, dict)
        and float(accent_visual_independent_policy.get("present_recall", 0.0) or 0.0)
        >= float(accent_visual_ablation_policy.get("present_recall", 0.0) or 0.0) + 0.10
        and isinstance(accent_visual_deployment, dict)
        and float(accent_visual_deployment.get("max_absolute_probability_delta", 1.0)) <= 1e-10
        and accent_visual_reproducibility.get("passed") is True
        and accent_visual_reproducibility.get("model_byte_identical") is True
        and accent_visual_reproducibility.get("report_byte_identical") is True
        and accent_visual_reproducibility.get("training_data_generator_deterministic_unit_test") is True
        and accent_visual_reproducibility.get("end_to_end_accuracy_claim") is False
        and accent_visual_model.is_file()
        and accent_visual_model.stat().st_size <= 4_500_000,
        "source-backed empty-to-single-accent additions require a reproducible diagonal-balanced local visual veto with zero maintained rendered false accepts",
    )
    rejected_accent_experiments = (
        accent_visual_rejections.get("experiments", [])
        if isinstance(accent_visual_rejections, dict)
        else []
    )
    accepted_accent_replacement = (
        accent_visual_rejections.get("accepted_replacement", {})
        if isinstance(accent_visual_rejections, dict)
        else {}
    )
    add(
        "articulation-patch:rejected-broad-symbol-experiments",
        accent_visual_rejections.get("deployed") is False
        and len(rejected_accent_experiments) >= 4
        and all(
            isinstance(item, dict)
            and item.get("status") == "rejected"
            and item.get("runtime_deployed") is False
            for item in rejected_accent_experiments
        )
        and any(int(item.get("observed_false_accepts", 0) or 0) >= 1 for item in rejected_accent_experiments)
        and any(float(item.get("best_zero_false_recall_upper_bound", 1.0) or 1.0) <= 0.10 for item in rejected_accent_experiments)
        and accepted_accent_replacement.get("model_version") == "scorescan-accent-visual-forest-1"
        and "staccato" in str(accepted_accent_replacement.get("excluded", ""))
        and "tenuto" in str(accepted_accent_replacement.get("excluded", "")),
        "staccato, tenuto, unbalanced-diagonal, deletion and class-substitution alternatives remain explicit non-runtime rejection evidence",
    )

    event_kind_visual_training = read_json(
        source_root / "training" / "event_kind_visual_guard_report_v1.json", {}
    )
    event_kind_visual_policies = (
        event_kind_visual_training.get("policies", {})
        if isinstance(event_kind_visual_training, dict)
        else {}
    )
    event_kind_visual_frozen = (
        event_kind_visual_policies.get("frozen_test", {})
        if isinstance(event_kind_visual_policies, dict)
        else {}
    )
    event_kind_visual_safety = (
        event_kind_visual_policies.get("safety_calibration", {})
        if isinstance(event_kind_visual_policies, dict)
        else {}
    )
    event_kind_visual_independent = (
        event_kind_visual_policies.get("independent_test", {})
        if isinstance(event_kind_visual_policies, dict)
        else {}
    )
    event_kind_visual_samples = (
        event_kind_visual_training.get("sample_metrics", {})
        if isinstance(event_kind_visual_training, dict)
        else {}
    )
    event_kind_visual_independent_sample = (
        event_kind_visual_samples.get("independent_test", {})
        if isinstance(event_kind_visual_samples, dict)
        else {}
    )
    event_kind_visual_ablation = (
        event_kind_visual_training.get("context_only_ablation", {})
        if isinstance(event_kind_visual_training, dict)
        else {}
    )
    event_kind_visual_dataset = (
        event_kind_visual_training.get("dataset", {})
        if isinstance(event_kind_visual_training, dict)
        else {}
    )
    event_kind_visual_external = read_json(
        source_root / "training" / "event_kind_visual_guard_external_tail_v1.json", {}
    )
    event_kind_visual_external_policy = (
        event_kind_visual_external.get("policy", {})
        if isinstance(event_kind_visual_external, dict)
        else {}
    )
    event_kind_visual_external_sample = (
        event_kind_visual_external.get("sample", {})
        if isinstance(event_kind_visual_external, dict)
        else {}
    )
    event_kind_visual_lower_thresholds = (
        event_kind_visual_external.get("rejected_lower_thresholds", {})
        if isinstance(event_kind_visual_external, dict)
        else {}
    )
    event_kind_visual_reproducibility = read_json(
        source_root / "training" / "event_kind_visual_guard_reproducibility_v1.json", {}
    )
    event_kind_visual_model = resources / "event_kind_visual_guard.json"
    add(
        "event-kind-patch:source-crop-gate",
        event_kind_visual_training.get("model_version")
        == "scorescan-event-kind-visual-forest-1"
        and int(event_kind_visual_training.get("feature_count", 0) or 0) == 448
        and int(event_kind_visual_dataset.get("training_groups", 0) or 0) >= 1_600
        and int(event_kind_visual_dataset.get("safety_groups", 0) or 0) >= 1_600
        and int(event_kind_visual_dataset.get("independent_groups", 0) or 0) >= 3_000
        and isinstance(event_kind_visual_frozen, dict)
        and int(event_kind_visual_frozen.get("false_accepts", 1) or 0) == 0
        and float(event_kind_visual_frozen.get("note_recall", 0.0) or 0.0) >= 0.64
        and float(event_kind_visual_frozen.get("rest_recall", 0.0) or 0.0) >= 0.62
        and isinstance(event_kind_visual_safety, dict)
        and int(event_kind_visual_safety.get("false_accepts", 1) or 0) == 0
        and float(event_kind_visual_safety.get("note_recall", 0.0) or 0.0) >= 0.60
        and float(event_kind_visual_safety.get("rest_recall", 0.0) or 0.0) >= 0.56
        and isinstance(event_kind_visual_independent, dict)
        and int(event_kind_visual_independent.get("false_accepts", 1) or 0) == 0
        and int(event_kind_visual_independent.get("correct_accepts", 0) or 0) >= 1_850
        and float(event_kind_visual_independent.get("note_recall", 0.0) or 0.0) >= 0.62
        and float(event_kind_visual_independent.get("rest_recall", 0.0) or 0.0) >= 0.61
        and float(event_kind_visual_independent.get("auto_patch_threshold", 0.0) or 0.0)
        >= DEFAULT_POLICY.event_kind_visual_guard_probability_floor
        and isinstance(event_kind_visual_independent_sample, dict)
        and float(event_kind_visual_independent_sample.get("roc_auc", 0.0) or 0.0) >= 0.99
        and isinstance(event_kind_visual_ablation, dict)
        and abs(float(event_kind_visual_ablation.get("roc_auc", 1.0) or 1.0) - 0.5) <= 0.02
        and float(event_kind_visual_training.get("deployment_max_probability_delta", 1.0)) <= 1e-10
        and event_kind_visual_reproducibility.get("passed") is True
        and event_kind_visual_reproducibility.get("model_byte_identical") is True
        and event_kind_visual_reproducibility.get("report_byte_identical") is True
        and event_kind_visual_reproducibility.get("end_to_end_accuracy_claim") is False
        and event_kind_visual_model.is_file()
        and event_kind_visual_model.stat().st_size <= 3_000_000,
        "single fixed-duration note/rest replacements require a reproducible source-crop veto with zero maintained rendered false accepts",
    )
    add(
        "event-kind-patch:external-tail",
        event_kind_visual_external.get("model_version")
        == "scorescan-event-kind-visual-forest-1"
        and int(event_kind_visual_external_policy.get("groups", 0) or 0) >= 5_000
        and int(event_kind_visual_external_policy.get("samples", 0) or 0) >= 10_000
        and int(event_kind_visual_external_policy.get("false_accepts", 1) or 0) == 0
        and int(event_kind_visual_external_policy.get("correct_accepts", 0) or 0) >= 3_000
        and float(event_kind_visual_external_policy.get("note_recall", 0.0) or 0.0) >= 0.62
        and float(event_kind_visual_external_policy.get("rest_recall", 0.0) or 0.0) >= 0.59
        and isinstance(event_kind_visual_external_sample, dict)
        and float(event_kind_visual_external_sample.get("roc_auc", 0.0) or 0.0) >= 0.99
        and isinstance(event_kind_visual_lower_thresholds.get("0.9", {}), dict)
        and int(event_kind_visual_lower_thresholds.get("0.9", {}).get("false_accepts", 0) or 0) >= 1
        and event_kind_visual_external.get("end_to_end_accuracy_claim") is False,
        "post-freeze 5,000-group tail remains zero-false at the deployed threshold and documents rejected lower thresholds",
    )

    rhythm_training = read_json(
        source_root / "training" / "rhythm_patch_calibrator_report_v1.json",
        {},
    )
    rhythm_frozen = rhythm_training.get("frozen_test", {}) if isinstance(rhythm_training, dict) else {}
    rhythm_policy = rhythm_frozen.get("policy", {}) if isinstance(rhythm_frozen, dict) else {}
    rhythm_baseline = (
        rhythm_frozen.get("accept_all_deterministic_proposals", {})
        if isinstance(rhythm_frozen, dict)
        else {}
    )
    rhythm_confirmation = (
        rhythm_training.get("independent_confirmation", {})
        if isinstance(rhythm_training, dict)
        else {}
    )
    rhythm_confirmation_policy = (
        rhythm_confirmation.get("policy", {}) if isinstance(rhythm_confirmation, dict) else {}
    )
    rhythm_confirmation_baseline = (
        rhythm_confirmation.get("accept_all_deterministic_proposals", {})
        if isinstance(rhythm_confirmation, dict)
        else {}
    )
    rhythm_deployment = (
        rhythm_training.get("deployment_parity", {})
        if isinstance(rhythm_training, dict)
        else {}
    )
    add(
        "rhythm-patch:cpu-training",
        rhythm_training.get("model_version") == "scorescan-rhythm-patch-forest-1"
        and isinstance(rhythm_policy, dict)
        and float(rhythm_policy.get("precision", 0.0) or 0.0) >= 0.999
        and float(rhythm_policy.get("positive_recall", 0.0) or 0.0) >= 0.95
        and int(rhythm_policy.get("false_accepts", 1) or 0) == 0
        and isinstance(rhythm_baseline, dict)
        and int(rhythm_baseline.get("false_accepts", 0) or 0) >= 250
        and isinstance(rhythm_confirmation_policy, dict)
        and float(rhythm_confirmation_policy.get("precision", 0.0) or 0.0) >= 0.999
        and float(rhythm_confirmation_policy.get("positive_recall", 0.0) or 0.0) >= 0.95
        and int(rhythm_confirmation_policy.get("false_accepts", 1) or 0) == 0
        and isinstance(rhythm_confirmation_baseline, dict)
        and int(rhythm_confirmation_baseline.get("false_accepts", 0) or 0) >= 850
        and isinstance(rhythm_deployment, dict)
        and float(rhythm_deployment.get("max_absolute_probability_delta", 1.0) or 1.0) <= 1e-10
        and int(rhythm_training.get("model_bytes", 10**9) or 10**9) <= 600_000,
        "rhythm-only family consensus repairs meter-complete monophonic events while the CPU veto rejects maintained harmful proposals",
    )

    rhythm_symbol_training = read_json(
        source_root / "training" / "rhythm_symbol_guard_report_v1.json", {}
    )
    rhythm_symbol_frozen = (
        rhythm_symbol_training.get("frozen_test", {})
        if isinstance(rhythm_symbol_training, dict)
        else {}
    )
    rhythm_symbol_frozen_policy = (
        rhythm_symbol_frozen.get("policy", {})
        if isinstance(rhythm_symbol_frozen, dict)
        else {}
    )
    rhythm_symbol_confirmation = (
        rhythm_symbol_training.get("independent_confirmation", {})
        if isinstance(rhythm_symbol_training, dict)
        else {}
    )
    rhythm_symbol_confirmation_policy = (
        rhythm_symbol_confirmation.get("policy", {})
        if isinstance(rhythm_symbol_confirmation, dict)
        else {}
    )
    rhythm_symbol_ablation = (
        rhythm_symbol_training.get("ablation_candidate_signatures_only", {})
        if isinstance(rhythm_symbol_training, dict)
        else {}
    )
    rhythm_symbol_ablation_sample = (
        rhythm_symbol_ablation.get("frozen_test_sample", {})
        if isinstance(rhythm_symbol_ablation, dict)
        else {}
    )
    rhythm_symbol_deployment = (
        rhythm_symbol_training.get("deployment_parity", {})
        if isinstance(rhythm_symbol_training, dict)
        else {}
    )
    rhythm_symbol_safety = read_json(
        source_root / "training" / "rhythm_symbol_guard_safety_calibration_v1.json", {}
    )
    rhythm_symbol_safety_policy = (
        rhythm_symbol_safety.get("selected_policy", {})
        if isinstance(rhythm_symbol_safety, dict)
        else {}
    )
    rhythm_symbol_independent = read_json(
        source_root / "training" / "rhythm_symbol_guard_independent_audit_v1.json", {}
    )
    rhythm_symbol_independent_policy = (
        rhythm_symbol_independent.get("policy", {})
        if isinstance(rhythm_symbol_independent, dict)
        else {}
    )
    rhythm_symbol_independent_sample = (
        rhythm_symbol_independent.get("sample", {})
        if isinstance(rhythm_symbol_independent, dict)
        else {}
    )
    rhythm_symbol_reproducibility = read_json(
        source_root / "training" / "rhythm_symbol_guard_reproducibility_v1.json", {}
    )
    add(
        "rhythm-patch:source-crop-symbol-gate",
        rhythm_symbol_training.get("model_version") == "scorescan-rhythm-symbol-forest-1"
        and isinstance(rhythm_symbol_frozen_policy, dict)
        and int(rhythm_symbol_frozen_policy.get("false_accepts", 1) or 0) == 0
        and float(rhythm_symbol_frozen_policy.get("positive_recall", 0.0) or 0.0) >= 0.68
        and isinstance(rhythm_symbol_confirmation_policy, dict)
        and int(rhythm_symbol_confirmation_policy.get("false_accepts", 1) or 0) == 0
        and float(rhythm_symbol_confirmation_policy.get("positive_recall", 0.0) or 0.0) >= 0.74
        and isinstance(rhythm_symbol_ablation_sample, dict)
        and abs(float(rhythm_symbol_ablation_sample.get("roc_auc", -1.0) or -1.0) - 0.5) <= 1e-12
        and isinstance(rhythm_symbol_deployment, dict)
        and float(rhythm_symbol_deployment.get("max_absolute_probability_delta", 1.0)) <= 1e-10
        and rhythm_symbol_safety.get("model_version") == "scorescan-rhythm-symbol-forest-1"
        and int(rhythm_symbol_safety.get("groups", 0) or 0) >= 1200
        and isinstance(rhythm_symbol_safety_policy, dict)
        and int(rhythm_symbol_safety_policy.get("false_accepts", 1) or 0) == 0
        and float(rhythm_symbol_safety_policy.get("positive_recall", 0.0) or 0.0) >= 0.50
        and abs(float(rhythm_symbol_safety_policy.get("threshold", 0.0) or 0.0) - 0.9875) <= 1e-12
        and rhythm_symbol_independent.get("model_version") == "scorescan-rhythm-symbol-forest-1"
        and int(rhythm_symbol_independent.get("groups", 0) or 0) >= 1200
        and isinstance(rhythm_symbol_independent_policy, dict)
        and int(rhythm_symbol_independent_policy.get("false_accepts", 1) or 0) == 0
        and float(rhythm_symbol_independent_policy.get("positive_recall", 0.0) or 0.0) >= 0.50
        and isinstance(rhythm_symbol_independent_sample, dict)
        and float(rhythm_symbol_independent_sample.get("roc_auc", 0.0) or 0.0) >= 0.996
        and rhythm_symbol_reproducibility.get("all_byte_identical") is True
        and all(
            isinstance(item, dict) and item.get("byte_identical") is True
            for item in rhythm_symbol_reproducibility.get("artifacts", {}).values()
        )
        and int(rhythm_symbol_training.get("model_bytes", 10**9) or 10**9) <= 1_700_000,
        "meter-complete rhythm repair is additionally vetoed by a reproducible source-crop transaction model with zero maintained rendered false accepts",
    )

    rhythm_symbol_rejections = read_json(
        source_root / "training" / "rhythm_symbol_guard_rejected_experiments_v1.json", {}
    )
    rejected_rhythm_experiments = (
        rhythm_symbol_rejections.get("experiments", [])
        if isinstance(rhythm_symbol_rejections, dict)
        else []
    )
    add(
        "rhythm-patch:rejected-symbol-experiments",
        rhythm_symbol_rejections.get("runtime_deployed_model")
        == "scorescan-rhythm-symbol-forest-1"
        and rhythm_symbol_rejections.get("end_to_end_accuracy_claim") is False
        and len(rejected_rhythm_experiments) >= 5
        and all(
            isinstance(item, dict)
            and item.get("decision") == "rejected"
            and item.get("runtime_deployed") is False
            for item in rejected_rhythm_experiments
        ),
        "absolute, unstable-tail, unsafe-threshold, over-quantised, and low-coverage rhythm-symbol alternatives remain explicitly rejected",
    )

    attribute_training = read_json(
        source_root / "training" / "attribute_patch_calibrator_report_v1.json",
        {},
    )
    attribute_frozen = attribute_training.get("frozen_test", {}) if isinstance(attribute_training, dict) else {}
    attribute_policy = attribute_frozen.get("policy", {}) if isinstance(attribute_frozen, dict) else {}
    attribute_baseline = (
        attribute_frozen.get("accept_all_deterministic_proposals", {})
        if isinstance(attribute_frozen, dict)
        else {}
    )
    attribute_confirmation = (
        attribute_training.get("independent_confirmation", {})
        if isinstance(attribute_training, dict)
        else {}
    )
    attribute_confirmation_policy = (
        attribute_confirmation.get("policy", {}) if isinstance(attribute_confirmation, dict) else {}
    )
    attribute_confirmation_baseline = (
        attribute_confirmation.get("accept_all_deterministic_proposals", {})
        if isinstance(attribute_confirmation, dict)
        else {}
    )
    attribute_deployment = (
        attribute_training.get("deployment_parity", {})
        if isinstance(attribute_training, dict)
        else {}
    )
    add(
        "attribute-patch:cpu-training",
        attribute_training.get("model_version") == "scorescan-attribute-patch-forest-1"
        and isinstance(attribute_policy, dict)
        and float(attribute_policy.get("precision", 0.0) or 0.0) >= 0.999
        and float(attribute_policy.get("positive_recall", 0.0) or 0.0) >= 0.95
        and int(attribute_policy.get("false_accepts", 1) or 0) == 0
        and isinstance(attribute_baseline, dict)
        and int(attribute_baseline.get("false_accepts", 0) or 0) >= 250
        and isinstance(attribute_confirmation_policy, dict)
        and float(attribute_confirmation_policy.get("precision", 0.0) or 0.0) >= 0.999
        and float(attribute_confirmation_policy.get("positive_recall", 0.0) or 0.0) >= 0.95
        and int(attribute_confirmation_policy.get("false_accepts", 1) or 0) == 0
        and isinstance(attribute_confirmation_baseline, dict)
        and int(attribute_confirmation_baseline.get("false_accepts", 0) or 0) >= 800
        and isinstance(attribute_deployment, dict)
        and float(attribute_deployment.get("max_absolute_probability_delta", 1.0) or 1.0) <= 1e-10
        and int(attribute_training.get("model_bytes", 10**9) or 10**9) <= 400_000,
        "time/key/clef family consensus is boundary-aware, meter-safe and vetoed by a reproducible CPU model",
    )

    event_kind_training = read_json(
        source_root / "training" / "event_kind_patch_calibrator_report_v1.json",
        {},
    )
    event_kind_frozen = (
        event_kind_training.get("frozen_test", {})
        if isinstance(event_kind_training, dict)
        else {}
    )
    event_kind_policy = (
        event_kind_frozen.get("policy", {}) if isinstance(event_kind_frozen, dict) else {}
    )
    event_kind_baseline = (
        event_kind_frozen.get("accept_all_deterministic_proposals", {})
        if isinstance(event_kind_frozen, dict)
        else {}
    )
    event_kind_confirmation = (
        event_kind_training.get("independent_confirmation", {})
        if isinstance(event_kind_training, dict)
        else {}
    )
    event_kind_confirmation_policy = (
        event_kind_confirmation.get("policy", {})
        if isinstance(event_kind_confirmation, dict)
        else {}
    )
    event_kind_confirmation_baseline = (
        event_kind_confirmation.get("accept_all_deterministic_proposals", {})
        if isinstance(event_kind_confirmation, dict)
        else {}
    )
    event_kind_deployment = (
        event_kind_training.get("deployment_parity", {})
        if isinstance(event_kind_training, dict)
        else {}
    )
    add(
        "event-kind-patch:cpu-training",
        event_kind_training.get("model_version") == "scorescan-event-kind-patch-forest-1"
        and isinstance(event_kind_policy, dict)
        and float(event_kind_policy.get("precision", 0.0) or 0.0) >= 0.999
        and float(event_kind_policy.get("positive_recall", 0.0) or 0.0) >= 0.95
        and int(event_kind_policy.get("false_accepts", 1) or 0) == 0
        and isinstance(event_kind_baseline, dict)
        and int(event_kind_baseline.get("false_accepts", 0) or 0) >= 300
        and isinstance(event_kind_confirmation_policy, dict)
        and float(event_kind_confirmation_policy.get("precision", 0.0) or 0.0) >= 0.999
        and float(event_kind_confirmation_policy.get("positive_recall", 0.0) or 0.0) >= 0.95
        and int(event_kind_confirmation_policy.get("false_accepts", 1) or 0) == 0
        and isinstance(event_kind_confirmation_baseline, dict)
        and int(event_kind_confirmation_baseline.get("false_accepts", 0) or 0) >= 1_000
        and isinstance(event_kind_deployment, dict)
        and float(event_kind_deployment.get("max_absolute_probability_delta", 1.0) or 1.0) <= 1e-10
        and int(event_kind_training.get("model_bytes", 10**9) or 10**9) <= 300_000,
        "rest-versus-pitched-note family consensus preserves the event lattice and fails closed on weak evidence",
    )

    event_presence_training = read_json(
        source_root / "training" / "event_presence_patch_calibrator_report_v1.json",
        {},
    )
    event_presence_frozen = (
        event_presence_training.get("frozen_test", {})
        if isinstance(event_presence_training, dict)
        else {}
    )
    event_presence_policy = (
        event_presence_frozen.get("policy", {})
        if isinstance(event_presence_frozen, dict)
        else {}
    )
    event_presence_baseline = (
        event_presence_frozen.get("accept_all_deterministic_proposals", {})
        if isinstance(event_presence_frozen, dict)
        else {}
    )
    event_presence_confirmation = (
        event_presence_training.get("independent_confirmation", {})
        if isinstance(event_presence_training, dict)
        else {}
    )
    event_presence_confirmation_policy = (
        event_presence_confirmation.get("policy", {})
        if isinstance(event_presence_confirmation, dict)
        else {}
    )
    event_presence_confirmation_baseline = (
        event_presence_confirmation.get("accept_all_deterministic_proposals", {})
        if isinstance(event_presence_confirmation, dict)
        else {}
    )
    event_presence_deployment = (
        event_presence_training.get("deployment_parity", {})
        if isinstance(event_presence_training, dict)
        else {}
    )
    event_presence_leakage = (
        event_presence_training.get("group_leakage_audit", {})
        if isinstance(event_presence_training, dict)
        else {}
    )
    event_presence_overlaps = (
        event_presence_leakage.get("pairwise_group_overlap", {})
        if isinstance(event_presence_leakage, dict)
        else {}
    )
    event_presence_partition_groups = (
        event_presence_leakage.get("groups_per_partition", {})
        if isinstance(event_presence_leakage, dict)
        else {}
    )
    event_presence_partitions = (
        event_presence_training.get("partitions", {})
        if isinstance(event_presence_training, dict)
        else {}
    )
    event_presence_groups = int(event_presence_training.get("groups", 0) or 0) if isinstance(event_presence_training, dict) else 0
    event_presence_samples_per_group = int(event_presence_training.get("samples_per_group", 0) or 0) if isinstance(event_presence_training, dict) else 0
    event_presence_model_path = resources / "event_presence_patch_calibrator.json"
    event_presence_model_hash = sha256_file(event_presence_model_path) if event_presence_model_path.is_file() else ""
    event_presence_dataset_fingerprint = str(event_presence_training.get("dataset_fingerprint", "")) if isinstance(event_presence_training, dict) else ""
    event_presence_confirmation_fingerprint = str(event_presence_confirmation.get("dataset_fingerprint", "")) if isinstance(event_presence_confirmation, dict) else ""
    add(
        "event-presence-patch:cpu-training",
        event_presence_training.get("model_version") == "scorescan-event-presence-patch-forest-1"
        and event_presence_samples_per_group == 3
        and event_presence_groups >= 8_000
        and int(event_presence_training.get("samples", 0) or 0) == event_presence_groups * event_presence_samples_per_group
        and isinstance(event_presence_leakage, dict)
        and not bool(event_presence_leakage.get("leakage_detected", True))
        and isinstance(event_presence_overlaps, dict)
        and len(event_presence_overlaps) == 10
        and all(int(value or 0) == 0 for value in event_presence_overlaps.values())
        and isinstance(event_presence_partition_groups, dict)
        and sum(int(value or 0) for value in event_presence_partition_groups.values()) == event_presence_groups
        and isinstance(event_presence_partitions, dict)
        and all(
            int(event_presence_partitions.get(name, -1) or -1)
            == int(group_count or 0) * event_presence_samples_per_group
            for name, group_count in event_presence_partition_groups.items()
        )
        and len(event_presence_dataset_fingerprint) == 64
        and len(event_presence_confirmation_fingerprint) == 64
        and event_presence_dataset_fingerprint != event_presence_confirmation_fingerprint
        and int(event_presence_confirmation.get("samples", 0) or 0)
        == int(event_presence_confirmation.get("groups", 0) or 0) * event_presence_samples_per_group
        and str(event_presence_training.get("model_sha256", "")) == event_presence_model_hash
        and isinstance(event_presence_policy, dict)
        and float(event_presence_policy.get("precision", 0.0) or 0.0) >= 0.999
        and float(event_presence_policy.get("positive_recall", 0.0) or 0.0) >= 0.98
        and int(event_presence_policy.get("false_accepts", 1) or 0) == 0
        and isinstance(event_presence_baseline, dict)
        and int(event_presence_baseline.get("false_accepts", 0) or 0) >= 1_200
        and isinstance(event_presence_confirmation_policy, dict)
        and float(event_presence_confirmation_policy.get("precision", 0.0) or 0.0) >= 0.999
        and float(event_presence_confirmation_policy.get("positive_recall", 0.0) or 0.0) >= 0.98
        and int(event_presence_confirmation_policy.get("false_accepts", 1) or 0) == 0
        and isinstance(event_presence_confirmation_baseline, dict)
        and int(event_presence_confirmation_baseline.get("false_accepts", 0) or 0) >= 3_700
        and isinstance(event_presence_deployment, dict)
        and float(event_presence_deployment.get("max_absolute_probability_delta", 1.0) or 1.0) <= 1e-10
        and int(event_presence_training.get("model_bytes", 10**9) or 10**9) <= 900_000,
        "single-event presence consensus uses grouped leak-free training and a reproducible CPU veto with no maintained harmful accepts",
    )

    event_presence_visual = read_json(
        source_root / "training" / "event_presence_visual_guard_report_v1.json", {}
    )
    event_presence_visual_external = read_json(
        source_root / "training" / "event_presence_visual_guard_external_tail_v1.json", {}
    )
    event_presence_visual_repro = read_json(
        source_root / "training" / "event_presence_visual_guard_reproducibility_v1.json", {}
    )
    event_presence_visual_rejected = read_json(
        source_root / "training" / "event_presence_visual_rejected_experiments_v1.json", {}
    )
    visual_policies = (
        event_presence_visual.get("policies", {})
        if isinstance(event_presence_visual, dict)
        else {}
    )
    visual_metrics = (
        event_presence_visual.get("sample_metrics", {})
        if isinstance(event_presence_visual, dict)
        else {}
    )
    visual_dataset = (
        event_presence_visual.get("dataset", {})
        if isinstance(event_presence_visual, dict)
        else {}
    )
    visual_contract = (
        event_presence_visual.get("transaction_contract", {})
        if isinstance(event_presence_visual, dict)
        else {}
    )
    visual_ablation = (
        event_presence_visual.get("context_only_ablation", {})
        if isinstance(event_presence_visual, dict)
        else {}
    )
    visual_lower = (
        event_presence_visual.get("rejected_lower_thresholds", {})
        if isinstance(event_presence_visual, dict)
        else {}
    )
    visual_thresholds = (
        event_presence_visual.get("auto_patch_thresholds", {})
        if isinstance(event_presence_visual, dict)
        else {}
    )
    visual_rejected_experiments = (
        event_presence_visual_rejected.get("experiments", {})
        if isinstance(event_presence_visual_rejected, dict)
        else {}
    )
    visual_probability_delta = event_presence_visual.get(
        "deployment_max_probability_delta", None
    ) if isinstance(event_presence_visual, dict) else None
    visual_model_path = resources / "event_presence_visual_guard.json"
    add(
        "event-presence-visual:cpu-training",
        event_presence_visual.get("model_version")
        == "scorescan-event-presence-visual-forest-1"
        and int(event_presence_visual.get("feature_count", 0) or 0) == 640
        and visual_dataset == {
            "training_groups": 1600,
            "training_samples": 6400,
            "safety_groups": 4000,
            "safety_samples": 16000,
            "independent_groups": 3000,
            "independent_samples": 12000,
            "external_groups": 5000,
            "external_samples": 20000,
        }
        and visual_contract.get("operation") == "single_event_insertion_only"
        and visual_contract.get("accepted_suffix_forms")
        == ["unchanged_explicit_gap", "coherent_shift_by_inserted_duration"]
        and visual_contract.get("target_event_kind_sampling") == "balanced_note_rest"
        and visual_contract.get("source_comparison")
        == "proposed_event_vs_displaced_event_plus_complete_before_after_sequence"
        and isinstance(visual_thresholds, dict)
        and all(
            float(visual_thresholds.get(kind, 0.0) or 0.0)
            >= DEFAULT_POLICY.event_presence_visual_guard_probability_floor
            for kind in ("note", "rest")
        )
        and isinstance(visual_policies, dict)
        and set(visual_policies) == {
            "frozen_test", "safety_calibration", "independent_test", "external_tail"
        }
        and all(
            int(policy.get("false_accepts", 1) or 0) == 0
            and float(policy.get("selective_precision", 0.0) or 0.0) >= 0.999999
            and float(policy.get("insert_recall", 0.0) or 0.0) >= 0.38
            and float(policy.get("note_insert_recall", 0.0) or 0.0) >= 0.35
            and float(policy.get("rest_insert_recall", 0.0) or 0.0) >= 0.35
            and int(policy.get("delete_transactions_review_only", 0) or 0) > 0
            for policy in visual_policies.values()
            if isinstance(policy, dict)
        )
        and len(visual_policies) == 4
        and isinstance(visual_metrics, dict)
        and float(visual_metrics.get("independent_test", {}).get("roc_auc", 0.0) or 0.0)
        >= 0.99
        and float(visual_metrics.get("external_tail", {}).get("roc_auc", 0.0) or 0.0)
        >= 0.99
        and isinstance(visual_ablation, dict)
        and 0.49 <= float(visual_ablation.get("roc_auc", -1.0)) <= 0.51
        and visual_probability_delta is not None
        and float(visual_probability_delta) <= 1e-10
        and isinstance(visual_lower, dict)
        and set(visual_lower) == {"0.85", "0.88", "0.90"}
        and all(int(value.get("false_accepts", 0) or 0) > 0 for value in visual_lower.values())
        and event_presence_visual_external.get("model_version")
        == "scorescan-event-presence-visual-forest-1"
        and int(event_presence_visual_external.get("groups", 0) or 0) == 5000
        and int(event_presence_visual_external.get("samples", 0) or 0) == 20000
        and event_presence_visual_external.get("used_for_training_or_threshold_selection") is False
        and event_presence_visual_repro.get("model_version")
        == "scorescan-event-presence-visual-forest-1"
        and event_presence_visual_repro.get("passed") is True
        and event_presence_visual_repro.get("all_byte_identical") is True
        and event_presence_visual_repro.get("dataset_arrays_identical") is True
        and event_presence_visual_repro.get("dataset_npz_byte_identical") is True
        and event_presence_visual_rejected.get("runtime_deployed_scope")
        == "single_event_insertion_confirmation_only"
        and all(
            isinstance(visual_rejected_experiments.get(name), dict)
            and visual_rejected_experiments[name].get("status") == "rejected"
            and visual_rejected_experiments[name].get("runtime_deployed") is False
            for name in (
                "automatic_deletion",
                "gap_only_insertion_dataset",
                "sequence_only_without_competing_event",
                "small_safety_tail",
                "shared_note_rest_target_prior",
                "lower_thresholds",
            )
        )
        and visual_model_path.is_file()
        and visual_model_path.stat().st_size <= 7_000_000,
        "single-event insertion visual guard uses complete transaction evidence, independent-family semantics, four zero-false-accept data gates, exact CPU reproducibility, and review-only deletion",
    )

    chord_training = read_json(
        source_root / "training" / "chord_patch_calibrator_report_v1.json",
        {},
    )
    chord_frozen = chord_training.get("frozen_test", {}) if isinstance(chord_training, dict) else {}
    chord_policy = chord_frozen.get("policy", {}) if isinstance(chord_frozen, dict) else {}
    chord_baseline = (
        chord_frozen.get("accept_all_deterministic_proposals", {})
        if isinstance(chord_frozen, dict)
        else {}
    )
    chord_confirmation = (
        chord_training.get("independent_confirmation", {})
        if isinstance(chord_training, dict)
        else {}
    )
    chord_confirmation_policy = (
        chord_confirmation.get("policy", {})
        if isinstance(chord_confirmation, dict)
        else {}
    )
    chord_confirmation_baseline = (
        chord_confirmation.get("accept_all_deterministic_proposals", {})
        if isinstance(chord_confirmation, dict)
        else {}
    )
    chord_deployment = chord_training.get("deployment_parity", {}) if isinstance(chord_training, dict) else {}
    chord_leakage = chord_training.get("group_leakage_audit", {}) if isinstance(chord_training, dict) else {}
    chord_overlaps = chord_leakage.get("pairwise_group_overlap", {}) if isinstance(chord_leakage, dict) else {}
    chord_partition_groups = chord_leakage.get("groups_per_partition", {}) if isinstance(chord_leakage, dict) else {}
    chord_partitions = chord_training.get("partitions", {}) if isinstance(chord_training, dict) else {}
    chord_groups = int(chord_training.get("groups", 0) or 0) if isinstance(chord_training, dict) else 0
    chord_samples_per_group = int(chord_training.get("samples_per_group", 0) or 0) if isinstance(chord_training, dict) else 0
    chord_model_path = resources / "chord_patch_calibrator.json"
    chord_model_hash = sha256_file(chord_model_path) if chord_model_path.is_file() else ""
    chord_dataset_fingerprint = str(chord_training.get("dataset_fingerprint", "")) if isinstance(chord_training, dict) else ""
    chord_confirmation_fingerprint = str(chord_confirmation.get("dataset_fingerprint", "")) if isinstance(chord_confirmation, dict) else ""
    add(
        "chord-patch:cpu-training",
        chord_training.get("model_version") == "scorescan-chord-patch-forest-1"
        and chord_samples_per_group == 3
        and chord_groups >= 9_000
        and int(chord_training.get("samples", 0) or 0) == chord_groups * chord_samples_per_group
        and isinstance(chord_leakage, dict)
        and not bool(chord_leakage.get("leakage_detected", True))
        and isinstance(chord_overlaps, dict)
        and len(chord_overlaps) == 10
        and all(int(value or 0) == 0 for value in chord_overlaps.values())
        and isinstance(chord_partition_groups, dict)
        and sum(int(value or 0) for value in chord_partition_groups.values()) == chord_groups
        and isinstance(chord_partitions, dict)
        and all(
            int(chord_partitions.get(name, -1) or -1)
            == int(group_count or 0) * chord_samples_per_group
            for name, group_count in chord_partition_groups.items()
        )
        and len(chord_dataset_fingerprint) == 64
        and len(chord_confirmation_fingerprint) == 64
        and chord_dataset_fingerprint != chord_confirmation_fingerprint
        and int(chord_confirmation.get("samples", 0) or 0)
        == int(chord_confirmation.get("groups", 0) or 0) * chord_samples_per_group
        and str(chord_training.get("model_sha256", "")) == chord_model_hash
        and isinstance(chord_policy, dict)
        and float(chord_policy.get("precision", 0.0) or 0.0) >= 0.999
        and float(chord_policy.get("positive_recall", 0.0) or 0.0) >= 0.98
        and int(chord_policy.get("false_accepts", 1) or 0) == 0
        and isinstance(chord_baseline, dict)
        and int(chord_baseline.get("false_accepts", 0) or 0) >= 1_200
        and isinstance(chord_confirmation_policy, dict)
        and float(chord_confirmation_policy.get("precision", 0.0) or 0.0) >= 0.999
        and float(chord_confirmation_policy.get("positive_recall", 0.0) or 0.0) >= 0.98
        and int(chord_confirmation_policy.get("false_accepts", 1) or 0) == 0
        and isinstance(chord_confirmation_baseline, dict)
        and int(chord_confirmation_baseline.get("false_accepts", 0) or 0) >= 4_000
        and isinstance(chord_deployment, dict)
        and float(chord_deployment.get("max_absolute_probability_delta", 1.0) or 1.0) <= 1e-10
        and int(chord_training.get("model_bytes", 10**9) or 10**9) <= 2_000_000,
        "chord-topology consensus uses grouped leak-free training and a reproducible CPU veto with no maintained harmful accepts",
    )

    tuplet_training = read_json(
        source_root / "training" / "tuplet_patch_calibrator_report_v1.json",
        {},
    )
    tuplet_frozen = tuplet_training.get("frozen_test", {}) if isinstance(tuplet_training, dict) else {}
    tuplet_policy = tuplet_frozen.get("policy", {}) if isinstance(tuplet_frozen, dict) else {}
    tuplet_baseline = (
        tuplet_frozen.get("accept_all_deterministic_proposals", {})
        if isinstance(tuplet_frozen, dict)
        else {}
    )
    tuplet_confirmation = (
        tuplet_training.get("independent_confirmation", {})
        if isinstance(tuplet_training, dict)
        else {}
    )
    tuplet_confirmation_policy = (
        tuplet_confirmation.get("policy", {})
        if isinstance(tuplet_confirmation, dict)
        else {}
    )
    tuplet_confirmation_baseline = (
        tuplet_confirmation.get("accept_all_deterministic_proposals", {})
        if isinstance(tuplet_confirmation, dict)
        else {}
    )
    tuplet_deployment = tuplet_training.get("deployment_parity", {}) if isinstance(tuplet_training, dict) else {}
    tuplet_leakage = tuplet_training.get("group_leakage_audit", {}) if isinstance(tuplet_training, dict) else {}
    tuplet_overlaps = tuplet_leakage.get("pairwise_group_overlap", {}) if isinstance(tuplet_leakage, dict) else {}
    tuplet_partition_groups = tuplet_leakage.get("groups_per_partition", {}) if isinstance(tuplet_leakage, dict) else {}
    tuplet_partitions = tuplet_training.get("partitions", {}) if isinstance(tuplet_training, dict) else {}
    tuplet_groups = int(tuplet_training.get("groups", 0) or 0) if isinstance(tuplet_training, dict) else 0
    tuplet_samples_per_group = int(tuplet_training.get("samples_per_group", 0) or 0) if isinstance(tuplet_training, dict) else 0
    tuplet_model_path = resources / "tuplet_patch_calibrator.json"
    tuplet_model_hash = sha256_file(tuplet_model_path) if tuplet_model_path.is_file() else ""
    tuplet_dataset_fingerprint = str(tuplet_training.get("dataset_fingerprint", "")) if isinstance(tuplet_training, dict) else ""
    tuplet_confirmation_fingerprint = str(tuplet_confirmation.get("dataset_fingerprint", "")) if isinstance(tuplet_confirmation, dict) else ""
    add(
        "tuplet-patch:cpu-training",
        tuplet_training.get("model_version") == "scorescan-tuplet-patch-forest-1"
        and tuplet_samples_per_group == 3
        and tuplet_groups >= 9_000
        and int(tuplet_training.get("samples", 0) or 0) == tuplet_groups * tuplet_samples_per_group
        and isinstance(tuplet_leakage, dict)
        and not bool(tuplet_leakage.get("leakage_detected", True))
        and isinstance(tuplet_overlaps, dict)
        and len(tuplet_overlaps) == 10
        and all(int(value or 0) == 0 for value in tuplet_overlaps.values())
        and isinstance(tuplet_partition_groups, dict)
        and sum(int(value or 0) for value in tuplet_partition_groups.values()) == tuplet_groups
        and isinstance(tuplet_partitions, dict)
        and all(
            int(tuplet_partitions.get(name, -1) or -1)
            == int(group_count or 0) * tuplet_samples_per_group
            for name, group_count in tuplet_partition_groups.items()
        )
        and len(tuplet_dataset_fingerprint) == 64
        and len(tuplet_confirmation_fingerprint) == 64
        and tuplet_dataset_fingerprint != tuplet_confirmation_fingerprint
        and int(tuplet_confirmation.get("samples", 0) or 0)
        == int(tuplet_confirmation.get("groups", 0) or 0) * tuplet_samples_per_group
        and str(tuplet_training.get("model_sha256", "")) == tuplet_model_hash
        and isinstance(tuplet_policy, dict)
        and float(tuplet_policy.get("precision", 0.0) or 0.0) >= 0.999
        and float(tuplet_policy.get("positive_recall", 0.0) or 0.0) >= 0.98
        and int(tuplet_policy.get("false_accepts", 1) or 0) == 0
        and isinstance(tuplet_baseline, dict)
        and int(tuplet_baseline.get("false_accepts", 0) or 0) >= 1_300
        and isinstance(tuplet_confirmation_policy, dict)
        and float(tuplet_confirmation_policy.get("precision", 0.0) or 0.0) >= 0.999
        and float(tuplet_confirmation_policy.get("positive_recall", 0.0) or 0.0) >= 0.98
        and int(tuplet_confirmation_policy.get("false_accepts", 1) or 0) == 0
        and isinstance(tuplet_confirmation_baseline, dict)
        and int(tuplet_confirmation_baseline.get("false_accepts", 0) or 0) >= 4_500
        and isinstance(tuplet_deployment, dict)
        and float(tuplet_deployment.get("max_absolute_probability_delta", 1.0) or 1.0) <= 1e-10
        and int(tuplet_training.get("model_bytes", 10**9) or 10**9) <= 2_000_000,
        "simple-triplet topology consensus uses grouped leak-free training and a reproducible CPU veto with no maintained harmful accepts",
    )

    tie_training = read_json(
        source_root / "training" / "tie_patch_calibrator_report_v1.json",
        {},
    )
    tie_frozen = tie_training.get("frozen_test", {}) if isinstance(tie_training, dict) else {}
    tie_policy = tie_frozen.get("policy", {}) if isinstance(tie_frozen, dict) else {}
    tie_baseline = (
        tie_frozen.get("accept_all_deterministic_proposals", {})
        if isinstance(tie_frozen, dict)
        else {}
    )
    tie_confirmation = (
        tie_training.get("independent_confirmation", {})
        if isinstance(tie_training, dict)
        else {}
    )
    tie_confirmation_policy = (
        tie_confirmation.get("policy", {}) if isinstance(tie_confirmation, dict) else {}
    )
    tie_confirmation_baseline = (
        tie_confirmation.get("accept_all_deterministic_proposals", {})
        if isinstance(tie_confirmation, dict)
        else {}
    )
    tie_deployment = tie_training.get("deployment_parity", {}) if isinstance(tie_training, dict) else {}
    tie_leakage = tie_training.get("group_leakage_audit", {}) if isinstance(tie_training, dict) else {}
    tie_overlaps = tie_leakage.get("pairwise_group_overlap", {}) if isinstance(tie_leakage, dict) else {}
    tie_partition_groups = tie_leakage.get("groups_per_partition", {}) if isinstance(tie_leakage, dict) else {}
    tie_partitions = tie_training.get("partitions", {}) if isinstance(tie_training, dict) else {}
    tie_groups = int(tie_training.get("groups", 0) or 0) if isinstance(tie_training, dict) else 0
    tie_samples_per_group = int(tie_training.get("samples_per_group", 0) or 0) if isinstance(tie_training, dict) else 0
    tie_model_path = resources / "tie_patch_calibrator.json"
    tie_model_hash = sha256_file(tie_model_path) if tie_model_path.is_file() else ""
    tie_dataset_fingerprint = str(tie_training.get("dataset_fingerprint", "")) if isinstance(tie_training, dict) else ""
    tie_confirmation_fingerprint = str(tie_confirmation.get("dataset_fingerprint", "")) if isinstance(tie_confirmation, dict) else ""
    add(
        "tie-patch:cpu-training",
        tie_training.get("model_version") == "scorescan-tie-patch-forest-1"
        and tie_samples_per_group == 3
        and tie_groups >= 9_000
        and int(tie_training.get("samples", 0) or 0) == tie_groups * tie_samples_per_group
        and isinstance(tie_leakage, dict)
        and not bool(tie_leakage.get("leakage_detected", True))
        and isinstance(tie_overlaps, dict)
        and len(tie_overlaps) == 10
        and all(int(value or 0) == 0 for value in tie_overlaps.values())
        and isinstance(tie_partition_groups, dict)
        and sum(int(value or 0) for value in tie_partition_groups.values()) == tie_groups
        and isinstance(tie_partitions, dict)
        and all(
            int(tie_partitions.get(name, -1) or -1)
            == int(group_count or 0) * tie_samples_per_group
            for name, group_count in tie_partition_groups.items()
        )
        and len(tie_dataset_fingerprint) == 64
        and len(tie_confirmation_fingerprint) == 64
        and tie_dataset_fingerprint != tie_confirmation_fingerprint
        and int(tie_confirmation.get("samples", 0) or 0)
        == int(tie_confirmation.get("groups", 0) or 0) * tie_samples_per_group
        and str(tie_training.get("model_sha256", "")) == tie_model_hash
        and isinstance(tie_policy, dict)
        and float(tie_policy.get("precision", 0.0) or 0.0) >= 0.999
        and float(tie_policy.get("positive_recall", 0.0) or 0.0) >= 0.98
        and int(tie_policy.get("false_accepts", 1) or 0) == 0
        and isinstance(tie_baseline, dict)
        and int(tie_baseline.get("false_accepts", 0) or 0) >= 1_200
        and isinstance(tie_confirmation_policy, dict)
        and float(tie_confirmation_policy.get("precision", 0.0) or 0.0) >= 0.999
        and float(tie_confirmation_policy.get("positive_recall", 0.0) or 0.0) >= 0.98
        and int(tie_confirmation_policy.get("false_accepts", 1) or 0) == 0
        and isinstance(tie_confirmation_baseline, dict)
        and int(tie_confirmation_baseline.get("false_accepts", 0) or 0) >= 4_000
        and isinstance(tie_deployment, dict)
        and float(tie_deployment.get("max_absolute_probability_delta", 1.0) or 1.0) <= 1e-10
        and int(tie_training.get("model_bytes", 10**9) or 10**9) <= 2_000_000,
        "tie-topology consensus uses grouped leak-free training and a reproducible CPU veto with no maintained harmful accepts",
    )

    slur_training = read_json(
        source_root / "training" / "slur_patch_calibrator_report_v1.json",
        {},
    )
    slur_frozen = slur_training.get("frozen_test", {}) if isinstance(slur_training, dict) else {}
    slur_policy = slur_frozen.get("policy", {}) if isinstance(slur_frozen, dict) else {}
    slur_baseline = (
        slur_frozen.get("accept_all_deterministic_proposals", {})
        if isinstance(slur_frozen, dict)
        else {}
    )
    slur_confirmation = (
        slur_training.get("independent_confirmation", {})
        if isinstance(slur_training, dict)
        else {}
    )
    slur_confirmation_policy = (
        slur_confirmation.get("policy", {}) if isinstance(slur_confirmation, dict) else {}
    )
    slur_confirmation_baseline = (
        slur_confirmation.get("accept_all_deterministic_proposals", {})
        if isinstance(slur_confirmation, dict)
        else {}
    )
    slur_deployment = slur_training.get("deployment_parity", {}) if isinstance(slur_training, dict) else {}
    slur_leakage = slur_training.get("group_leakage_audit", {}) if isinstance(slur_training, dict) else {}
    slur_overlaps = slur_leakage.get("pairwise_group_overlap", {}) if isinstance(slur_leakage, dict) else {}
    slur_partition_groups = slur_leakage.get("groups_per_partition", {}) if isinstance(slur_leakage, dict) else {}
    slur_partitions = slur_training.get("partitions", {}) if isinstance(slur_training, dict) else {}
    slur_groups = int(slur_training.get("groups", 0) or 0) if isinstance(slur_training, dict) else 0
    slur_samples_per_group = int(slur_training.get("samples_per_group", 0) or 0) if isinstance(slur_training, dict) else 0
    slur_model_path = resources / "slur_patch_calibrator.json"
    slur_model_hash = sha256_file(slur_model_path) if slur_model_path.is_file() else ""
    slur_dataset_fingerprint = str(slur_training.get("dataset_fingerprint", "")) if isinstance(slur_training, dict) else ""
    slur_confirmation_fingerprint = str(slur_confirmation.get("dataset_fingerprint", "")) if isinstance(slur_confirmation, dict) else ""
    add(
        "slur-patch:cpu-training",
        slur_training.get("model_version") == "scorescan-slur-patch-forest-1"
        and slur_samples_per_group == 3
        and slur_groups >= 9_000
        and int(slur_training.get("samples", 0) or 0) == slur_groups * slur_samples_per_group
        and isinstance(slur_leakage, dict)
        and not bool(slur_leakage.get("leakage_detected", True))
        and isinstance(slur_overlaps, dict)
        and len(slur_overlaps) == 10
        and all(int(value or 0) == 0 for value in slur_overlaps.values())
        and isinstance(slur_partition_groups, dict)
        and sum(int(value or 0) for value in slur_partition_groups.values()) == slur_groups
        and isinstance(slur_partitions, dict)
        and all(
            int(slur_partitions.get(name, -1) or -1)
            == int(group_count or 0) * slur_samples_per_group
            for name, group_count in slur_partition_groups.items()
        )
        and len(slur_dataset_fingerprint) == 64
        and len(slur_confirmation_fingerprint) == 64
        and slur_dataset_fingerprint != slur_confirmation_fingerprint
        and int(slur_confirmation.get("samples", 0) or 0)
        == int(slur_confirmation.get("groups", 0) or 0) * slur_samples_per_group
        and str(slur_training.get("model_sha256", "")) == slur_model_hash
        and float(slur_training.get("policy_probability_floor", 0.0) or 0.0)
        == DEFAULT_POLICY.slur_patch_probability_floor
        and isinstance(slur_policy, dict)
        and float(slur_policy.get("precision", 0.0) or 0.0) >= 0.999
        and float(slur_policy.get("positive_recall", 0.0) or 0.0) >= 0.98
        and int(slur_policy.get("false_accepts", 1) or 0) == 0
        and isinstance(slur_baseline, dict)
        and int(slur_baseline.get("false_accepts", 0) or 0) >= 1_200
        and isinstance(slur_confirmation_policy, dict)
        and float(slur_confirmation_policy.get("precision", 0.0) or 0.0) >= 0.999
        and float(slur_confirmation_policy.get("positive_recall", 0.0) or 0.0) >= 0.98
        and int(slur_confirmation_policy.get("false_accepts", 1) or 0) == 0
        and isinstance(slur_confirmation_baseline, dict)
        and int(slur_confirmation_baseline.get("false_accepts", 0) or 0) >= 3_900
        and isinstance(slur_deployment, dict)
        and float(slur_deployment.get("max_absolute_probability_delta", 1.0) or 1.0) <= 1e-10
        and int(slur_training.get("model_bytes", 10**9) or 10**9) <= 2_000_000,
        "within-measure slur consensus uses grouped leak-free training and a reproducible CPU veto with zero maintained harmful accepts",
    )

    articulation_training = read_json(
        source_root / "training" / "articulation_patch_calibrator_report_v1.json",
        {},
    )
    articulation_frozen = (
        articulation_training.get("frozen_test", {})
        if isinstance(articulation_training, dict)
        else {}
    )
    articulation_policy = (
        articulation_frozen.get("policy", {})
        if isinstance(articulation_frozen, dict)
        else {}
    )
    articulation_baseline = (
        articulation_frozen.get("accept_all_deterministic_proposals", {})
        if isinstance(articulation_frozen, dict)
        else {}
    )
    articulation_confirmation = (
        articulation_training.get("independent_confirmation", {})
        if isinstance(articulation_training, dict)
        else {}
    )
    articulation_confirmation_policy = (
        articulation_confirmation.get("policy", {})
        if isinstance(articulation_confirmation, dict)
        else {}
    )
    articulation_confirmation_baseline = (
        articulation_confirmation.get("accept_all_deterministic_proposals", {})
        if isinstance(articulation_confirmation, dict)
        else {}
    )
    articulation_deployment = (
        articulation_training.get("deployment_parity", {})
        if isinstance(articulation_training, dict)
        else {}
    )
    articulation_leakage = (
        articulation_training.get("group_leakage_audit", {})
        if isinstance(articulation_training, dict)
        else {}
    )
    articulation_overlaps = (
        articulation_leakage.get("pairwise_group_overlap", {})
        if isinstance(articulation_leakage, dict)
        else {}
    )
    articulation_partition_groups = (
        articulation_leakage.get("groups_per_partition", {})
        if isinstance(articulation_leakage, dict)
        else {}
    )
    articulation_partitions = (
        articulation_training.get("partitions", {})
        if isinstance(articulation_training, dict)
        else {}
    )
    articulation_groups = (
        int(articulation_training.get("groups", 0) or 0)
        if isinstance(articulation_training, dict)
        else 0
    )
    articulation_samples_per_group = (
        int(articulation_training.get("samples_per_group", 0) or 0)
        if isinstance(articulation_training, dict)
        else 0
    )
    articulation_model_path = resources / "articulation_patch_calibrator.json"
    articulation_model_hash = (
        sha256_file(articulation_model_path) if articulation_model_path.is_file() else ""
    )
    articulation_dataset_fingerprint = (
        str(articulation_training.get("dataset_fingerprint", ""))
        if isinstance(articulation_training, dict)
        else ""
    )
    articulation_confirmation_fingerprint = (
        str(articulation_confirmation.get("dataset_fingerprint", ""))
        if isinstance(articulation_confirmation, dict)
        else ""
    )
    add(
        "articulation-patch:cpu-training",
        articulation_training.get("model_version") == "scorescan-articulation-patch-forest-1"
        and articulation_samples_per_group == 3
        and articulation_groups >= 9_000
        and int(articulation_training.get("samples", 0) or 0)
        == articulation_groups * articulation_samples_per_group
        and isinstance(articulation_leakage, dict)
        and not bool(articulation_leakage.get("leakage_detected", True))
        and isinstance(articulation_overlaps, dict)
        and len(articulation_overlaps) == 10
        and all(int(value or 0) == 0 for value in articulation_overlaps.values())
        and isinstance(articulation_partition_groups, dict)
        and sum(int(value or 0) for value in articulation_partition_groups.values())
        == articulation_groups
        and isinstance(articulation_partitions, dict)
        and all(
            int(articulation_partitions.get(name, -1) or -1)
            == int(group_count or 0) * articulation_samples_per_group
            for name, group_count in articulation_partition_groups.items()
        )
        and len(articulation_dataset_fingerprint) == 64
        and len(articulation_confirmation_fingerprint) == 64
        and articulation_dataset_fingerprint != articulation_confirmation_fingerprint
        and int(articulation_confirmation.get("samples", 0) or 0)
        == int(articulation_confirmation.get("groups", 0) or 0)
        * articulation_samples_per_group
        and str(articulation_training.get("model_sha256", ""))
        == articulation_model_hash
        and float(articulation_training.get("policy_probability_floor", 0.0) or 0.0)
        == DEFAULT_POLICY.articulation_patch_probability_floor
        and isinstance(articulation_policy, dict)
        and float(articulation_policy.get("precision", 0.0) or 0.0) >= 0.999
        and float(articulation_policy.get("positive_recall", 0.0) or 0.0) >= 0.90
        and int(articulation_policy.get("false_accepts", 1) or 0) == 0
        and isinstance(articulation_baseline, dict)
        and int(articulation_baseline.get("false_accepts", 0) or 0) >= 1_200
        and isinstance(articulation_confirmation_policy, dict)
        and float(articulation_confirmation_policy.get("precision", 0.0) or 0.0) >= 0.999
        and float(articulation_confirmation_policy.get("positive_recall", 0.0) or 0.0) >= 0.90
        and int(articulation_confirmation_policy.get("false_accepts", 1) or 0) == 0
        and isinstance(articulation_confirmation_baseline, dict)
        and int(articulation_confirmation_baseline.get("false_accepts", 0) or 0) >= 3_900
        and isinstance(articulation_deployment, dict)
        and float(articulation_deployment.get("max_absolute_probability_delta", 1.0) or 1.0)
        <= 1e-10
        and int(articulation_training.get("model_bytes", 10**9) or 10**9) <= 2_000_000,
        "simple articulation consensus uses grouped leak-free training and a reproducible CPU veto with zero maintained harmful accepts",
    )

    direction_training = read_json(
        source_root / "training" / "direction_patch_calibrator_report_v1.json",
        {},
    )
    direction_frozen = direction_training.get("frozen_test", {}) if isinstance(direction_training, dict) else {}
    direction_policy = direction_frozen.get("policy", {}) if isinstance(direction_frozen, dict) else {}
    direction_baseline = (
        direction_frozen.get("accept_all_deterministic_proposals", {})
        if isinstance(direction_frozen, dict)
        else {}
    )
    direction_confirmation = (
        direction_training.get("independent_confirmation", {})
        if isinstance(direction_training, dict)
        else {}
    )
    direction_confirmation_policy = (
        direction_confirmation.get("policy", {}) if isinstance(direction_confirmation, dict) else {}
    )
    direction_confirmation_baseline = (
        direction_confirmation.get("accept_all_deterministic_proposals", {})
        if isinstance(direction_confirmation, dict)
        else {}
    )
    direction_deployment = direction_training.get("deployment_parity", {}) if isinstance(direction_training, dict) else {}
    direction_leakage = direction_training.get("group_leakage_audit", {}) if isinstance(direction_training, dict) else {}
    direction_overlaps = direction_leakage.get("pairwise_group_overlap", {}) if isinstance(direction_leakage, dict) else {}
    direction_partition_groups = direction_leakage.get("groups_per_partition", {}) if isinstance(direction_leakage, dict) else {}
    direction_partitions = direction_training.get("partitions", {}) if isinstance(direction_training, dict) else {}
    direction_groups = int(direction_training.get("groups", 0) or 0) if isinstance(direction_training, dict) else 0
    direction_samples_per_group = int(direction_training.get("samples_per_group", 0) or 0) if isinstance(direction_training, dict) else 0
    direction_model_path = resources / "direction_patch_calibrator.json"
    direction_model_hash = sha256_file(direction_model_path) if direction_model_path.is_file() else ""
    direction_dataset_fingerprint = str(direction_training.get("dataset_fingerprint", "")) if isinstance(direction_training, dict) else ""
    direction_confirmation_fingerprint = str(direction_confirmation.get("dataset_fingerprint", "")) if isinstance(direction_confirmation, dict) else ""
    add(
        "direction-patch:cpu-training",
        direction_training.get("model_version") == "scorescan-direction-patch-forest-1"
        and direction_samples_per_group == 3
        and direction_groups >= 9_000
        and int(direction_training.get("samples", 0) or 0) == direction_groups * direction_samples_per_group
        and isinstance(direction_leakage, dict)
        and not bool(direction_leakage.get("leakage_detected", True))
        and isinstance(direction_overlaps, dict)
        and len(direction_overlaps) == 10
        and all(int(value or 0) == 0 for value in direction_overlaps.values())
        and isinstance(direction_partition_groups, dict)
        and sum(int(value or 0) for value in direction_partition_groups.values()) == direction_groups
        and isinstance(direction_partitions, dict)
        and all(
            int(direction_partitions.get(name, -1) or -1)
            == int(group_count or 0) * direction_samples_per_group
            for name, group_count in direction_partition_groups.items()
        )
        and len(direction_dataset_fingerprint) == 64
        and len(direction_confirmation_fingerprint) == 64
        and direction_dataset_fingerprint != direction_confirmation_fingerprint
        and int(direction_confirmation.get("samples", 0) or 0)
        == int(direction_confirmation.get("groups", 0) or 0) * direction_samples_per_group
        and str(direction_training.get("model_sha256", "")) == direction_model_hash
        and float(direction_training.get("policy_probability_floor", 0.0) or 0.0)
        == DEFAULT_POLICY.direction_patch_probability_floor
        and isinstance(direction_policy, dict)
        and float(direction_policy.get("precision", 0.0) or 0.0) >= 0.999
        and float(direction_policy.get("positive_recall", 0.0) or 0.0) >= 0.99
        and int(direction_policy.get("false_accepts", 1) or 0) == 0
        and isinstance(direction_baseline, dict)
        and int(direction_baseline.get("false_accepts", 0) or 0) >= 1_200
        and isinstance(direction_confirmation_policy, dict)
        and float(direction_confirmation_policy.get("precision", 0.0) or 0.0) >= 0.999
        and float(direction_confirmation_policy.get("positive_recall", 0.0) or 0.0) >= 0.99
        and int(direction_confirmation_policy.get("false_accepts", 1) or 0) == 0
        and isinstance(direction_confirmation_baseline, dict)
        and int(direction_confirmation_baseline.get("false_accepts", 0) or 0) >= 4_000
        and isinstance(direction_deployment, dict)
        and float(direction_deployment.get("max_absolute_probability_delta", 1.0) or 1.0) <= 1e-10
        and int(direction_training.get("model_bytes", 10**9) or 10**9) <= 2_000_000,
        "simple dynamics/metronome consensus uses grouped leak-free CPU training and zero maintained harmful accepts",
    )

    ornament_training = read_json(
        source_root / "training" / "ornament_patch_calibrator_report_v1.json",
        {},
    )
    ornament_frozen = ornament_training.get("frozen_test", {}) if isinstance(ornament_training, dict) else {}
    ornament_policy = ornament_frozen.get("policy", {}) if isinstance(ornament_frozen, dict) else {}
    ornament_baseline = (
        ornament_frozen.get("accept_all_deterministic_proposals", {})
        if isinstance(ornament_frozen, dict)
        else {}
    )
    ornament_confirmation = (
        ornament_training.get("independent_confirmation", {})
        if isinstance(ornament_training, dict)
        else {}
    )
    ornament_confirmation_policy = (
        ornament_confirmation.get("policy", {})
        if isinstance(ornament_confirmation, dict)
        else {}
    )
    ornament_confirmation_baseline = (
        ornament_confirmation.get("accept_all_deterministic_proposals", {})
        if isinstance(ornament_confirmation, dict)
        else {}
    )
    ornament_deployment = ornament_training.get("deployment_parity", {}) if isinstance(ornament_training, dict) else {}
    ornament_leakage = ornament_training.get("group_leakage_audit", {}) if isinstance(ornament_training, dict) else {}
    ornament_overlaps = ornament_leakage.get("pairwise_group_overlap", {}) if isinstance(ornament_leakage, dict) else {}
    ornament_partition_groups = ornament_leakage.get("groups_per_partition", {}) if isinstance(ornament_leakage, dict) else {}
    ornament_partitions = ornament_training.get("partitions", {}) if isinstance(ornament_training, dict) else {}
    ornament_groups = int(ornament_training.get("groups", 0) or 0) if isinstance(ornament_training, dict) else 0
    ornament_samples_per_group = int(ornament_training.get("samples_per_group", 0) or 0) if isinstance(ornament_training, dict) else 0
    ornament_model_path = resources / "ornament_patch_calibrator.json"
    ornament_model_hash = sha256_file(ornament_model_path) if ornament_model_path.is_file() else ""
    ornament_dataset_fingerprint = str(ornament_training.get("dataset_fingerprint", "")) if isinstance(ornament_training, dict) else ""
    ornament_confirmation_fingerprint = str(ornament_confirmation.get("dataset_fingerprint", "")) if isinstance(ornament_confirmation, dict) else ""
    add(
        "ornament-patch:cpu-training",
        ornament_training.get("model_version") == "scorescan-ornament-patch-forest-1"
        and ornament_samples_per_group == 3
        and ornament_groups >= 9_000
        and int(ornament_training.get("samples", 0) or 0) == ornament_groups * ornament_samples_per_group
        and isinstance(ornament_leakage, dict)
        and not bool(ornament_leakage.get("leakage_detected", True))
        and isinstance(ornament_overlaps, dict)
        and len(ornament_overlaps) == 10
        and all(int(value or 0) == 0 for value in ornament_overlaps.values())
        and isinstance(ornament_partition_groups, dict)
        and sum(int(value or 0) for value in ornament_partition_groups.values()) == ornament_groups
        and isinstance(ornament_partitions, dict)
        and all(
            int(ornament_partitions.get(name, -1) or -1)
            == int(group_count or 0) * ornament_samples_per_group
            for name, group_count in ornament_partition_groups.items()
        )
        and len(ornament_dataset_fingerprint) == 64
        and len(ornament_confirmation_fingerprint) == 64
        and ornament_dataset_fingerprint != ornament_confirmation_fingerprint
        and int(ornament_confirmation.get("samples", 0) or 0)
        == int(ornament_confirmation.get("groups", 0) or 0) * ornament_samples_per_group
        and str(ornament_training.get("model_sha256", "")) == ornament_model_hash
        and float(ornament_training.get("policy_probability_floor", 0.0) or 0.0)
        == DEFAULT_POLICY.ornament_patch_probability_floor
        and isinstance(ornament_policy, dict)
        and float(ornament_policy.get("precision", 0.0) or 0.0) >= 0.999
        and float(ornament_policy.get("positive_recall", 0.0) or 0.0) >= 0.74
        and int(ornament_policy.get("false_accepts", 1) or 0) == 0
        and isinstance(ornament_baseline, dict)
        and int(ornament_baseline.get("false_accepts", 0) or 0) >= 1_200
        and isinstance(ornament_confirmation_policy, dict)
        and float(ornament_confirmation_policy.get("precision", 0.0) or 0.0) >= 0.999
        and float(ornament_confirmation_policy.get("positive_recall", 0.0) or 0.0) >= 0.74
        and int(ornament_confirmation_policy.get("false_accepts", 1) or 0) == 0
        and isinstance(ornament_confirmation_baseline, dict)
        and int(ornament_confirmation_baseline.get("false_accepts", 0) or 0) >= 3_900
        and isinstance(ornament_deployment, dict)
        and float(ornament_deployment.get("max_absolute_probability_delta", 1.0) or 1.0) <= 1e-10
        and int(ornament_training.get("model_bytes", 10**9) or 10**9) <= 2_000_000,
        "simple ornament consensus uses grouped leak-free training and a reproducible CPU veto with zero maintained harmful accepts",
    )


    grace_training = read_json(
        source_root / "training" / "grace_patch_calibrator_report_v1.json",
        {},
    )
    grace_frozen = grace_training.get("frozen_test", {}) if isinstance(grace_training, dict) else {}
    grace_policy = grace_frozen.get("policy", {}) if isinstance(grace_frozen, dict) else {}
    grace_baseline = (
        grace_frozen.get("accept_all_deterministic_proposals", {})
        if isinstance(grace_frozen, dict)
        else {}
    )
    grace_confirmation = (
        grace_training.get("independent_confirmation", {})
        if isinstance(grace_training, dict)
        else {}
    )
    grace_confirmation_policy = (
        grace_confirmation.get("policy", {}) if isinstance(grace_confirmation, dict) else {}
    )
    grace_confirmation_baseline = (
        grace_confirmation.get("accept_all_deterministic_proposals", {})
        if isinstance(grace_confirmation, dict)
        else {}
    )
    grace_deployment = grace_training.get("deployment_parity", {}) if isinstance(grace_training, dict) else {}
    grace_leakage = grace_training.get("group_leakage_audit", {}) if isinstance(grace_training, dict) else {}
    grace_overlaps = grace_leakage.get("pairwise_group_overlap", {}) if isinstance(grace_leakage, dict) else {}
    grace_partition_groups = grace_leakage.get("groups_per_partition", {}) if isinstance(grace_leakage, dict) else {}
    grace_partitions = grace_training.get("partitions", {}) if isinstance(grace_training, dict) else {}
    grace_groups = int(grace_training.get("groups", 0) or 0) if isinstance(grace_training, dict) else 0
    grace_samples_per_group = int(grace_training.get("samples_per_group", 0) or 0) if isinstance(grace_training, dict) else 0
    grace_model_path = resources / "grace_patch_calibrator.json"
    grace_model_hash = sha256_file(grace_model_path) if grace_model_path.is_file() else ""
    grace_dataset_fingerprint = str(grace_training.get("dataset_fingerprint", "")) if isinstance(grace_training, dict) else ""
    grace_confirmation_fingerprint = str(grace_confirmation.get("dataset_fingerprint", "")) if isinstance(grace_confirmation, dict) else ""
    add(
        "grace-patch:cpu-training",
        grace_training.get("model_version") == "scorescan-grace-patch-forest-1"
        and grace_samples_per_group == 3
        and grace_groups >= 9_000
        and int(grace_training.get("samples", 0) or 0) == grace_groups * grace_samples_per_group
        and isinstance(grace_leakage, dict)
        and not bool(grace_leakage.get("leakage_detected", True))
        and isinstance(grace_overlaps, dict)
        and len(grace_overlaps) == 10
        and all(int(value or 0) == 0 for value in grace_overlaps.values())
        and isinstance(grace_partition_groups, dict)
        and sum(int(value or 0) for value in grace_partition_groups.values()) == grace_groups
        and isinstance(grace_partitions, dict)
        and all(
            int(grace_partitions.get(name, -1) or -1)
            == int(group_count or 0) * grace_samples_per_group
            for name, group_count in grace_partition_groups.items()
        )
        and len(grace_dataset_fingerprint) == 64
        and len(grace_confirmation_fingerprint) == 64
        and grace_dataset_fingerprint != grace_confirmation_fingerprint
        and int(grace_confirmation.get("samples", 0) or 0)
        == int(grace_confirmation.get("groups", 0) or 0) * grace_samples_per_group
        and str(grace_training.get("model_sha256", "")) == grace_model_hash
        and float(grace_training.get("policy_probability_floor", 0.0) or 0.0)
        == DEFAULT_POLICY.grace_patch_probability_floor
        and isinstance(grace_policy, dict)
        and float(grace_policy.get("precision", 0.0) or 0.0) >= 0.999
        and float(grace_policy.get("positive_recall", 0.0) or 0.0) >= 0.99
        and int(grace_policy.get("false_accepts", 1) or 0) == 0
        and isinstance(grace_baseline, dict)
        and int(grace_baseline.get("false_accepts", 0) or 0) >= 1_200
        and isinstance(grace_confirmation_policy, dict)
        and float(grace_confirmation_policy.get("precision", 0.0) or 0.0) >= 0.999
        and float(grace_confirmation_policy.get("positive_recall", 0.0) or 0.0) >= 0.99
        and int(grace_confirmation_policy.get("false_accepts", 1) or 0) == 0
        and isinstance(grace_confirmation_baseline, dict)
        and int(grace_confirmation_baseline.get("false_accepts", 0) or 0) >= 3_900
        and isinstance(grace_deployment, dict)
        and float(grace_deployment.get("max_absolute_probability_delta", 1.0) or 1.0) <= 1e-10
        and int(grace_training.get("model_bytes", 10**9) or 10**9) <= 700_000,
        "simple grace/regular topology consensus uses grouped leak-free CPU training and zero maintained harmful accepts",
    )

    add(
        "boundary:semantic-lyrics-disabled",
        DEFAULT_POLICY.semantic_lyric_output_enabled is False,
        (
            "lyrics are visually isolated but semantic lyric transcription "
            "and automatic lyric repair remain outside the product boundary"
        ),
    )

    barline_patch_training = read_json(
        source_root / "training" / "barline_patch_calibrator_report_v1.json",
        {},
    )
    barline_patch_frozen = (
        barline_patch_training.get("frozen_test", {})
        if isinstance(barline_patch_training, dict)
        else {}
    )
    barline_patch_policy = (
        barline_patch_frozen.get("policy", {})
        if isinstance(barline_patch_frozen, dict)
        else {}
    )
    barline_patch_baseline = (
        barline_patch_frozen.get("accept_all_deterministic_proposals", {})
        if isinstance(barline_patch_frozen, dict)
        else {}
    )
    barline_patch_confirmation = (
        barline_patch_training.get("independent_confirmation", {})
        if isinstance(barline_patch_training, dict)
        else {}
    )
    barline_patch_confirmation_policy = (
        barline_patch_confirmation.get("policy", {})
        if isinstance(barline_patch_confirmation, dict)
        else {}
    )
    barline_patch_confirmation_baseline = (
        barline_patch_confirmation.get("accept_all_deterministic_proposals", {})
        if isinstance(barline_patch_confirmation, dict)
        else {}
    )
    barline_patch_deployment = (
        barline_patch_training.get("deployment_parity", {})
        if isinstance(barline_patch_training, dict)
        else {}
    )
    barline_patch_leakage = (
        barline_patch_training.get("group_leakage_audit", {})
        if isinstance(barline_patch_training, dict)
        else {}
    )
    barline_patch_overlaps = (
        barline_patch_leakage.get("pairwise_group_overlap", {})
        if isinstance(barline_patch_leakage, dict)
        else {}
    )
    barline_patch_partition_groups = (
        barline_patch_leakage.get("groups_per_partition", {})
        if isinstance(barline_patch_leakage, dict)
        else {}
    )
    barline_patch_partitions = (
        barline_patch_training.get("partitions", {})
        if isinstance(barline_patch_training, dict)
        else {}
    )
    barline_patch_groups = (
        int(barline_patch_training.get("groups", 0) or 0)
        if isinstance(barline_patch_training, dict)
        else 0
    )
    barline_patch_samples_per_group = (
        int(barline_patch_training.get("samples_per_group", 0) or 0)
        if isinstance(barline_patch_training, dict)
        else 0
    )
    barline_patch_model_path = resources / "barline_patch_calibrator.json"
    barline_patch_model_hash = (
        sha256_file(barline_patch_model_path) if barline_patch_model_path.is_file() else ""
    )
    barline_patch_dataset_fingerprint = (
        str(barline_patch_training.get("dataset_fingerprint", ""))
        if isinstance(barline_patch_training, dict)
        else ""
    )
    barline_patch_confirmation_fingerprint = (
        str(barline_patch_confirmation.get("dataset_fingerprint", ""))
        if isinstance(barline_patch_confirmation, dict)
        else ""
    )
    add(
        "barline-patch:cpu-training",
        barline_patch_training.get("model_version") == "scorescan-barline-patch-forest-1"
        and barline_patch_samples_per_group == 3
        and barline_patch_groups >= 9_000
        and int(barline_patch_training.get("samples", 0) or 0)
        == barline_patch_groups * barline_patch_samples_per_group
        and isinstance(barline_patch_leakage, dict)
        and not bool(barline_patch_leakage.get("leakage_detected", True))
        and isinstance(barline_patch_overlaps, dict)
        and len(barline_patch_overlaps) == 10
        and all(int(value or 0) == 0 for value in barline_patch_overlaps.values())
        and isinstance(barline_patch_partition_groups, dict)
        and sum(int(value or 0) for value in barline_patch_partition_groups.values())
        == barline_patch_groups
        and isinstance(barline_patch_partitions, dict)
        and all(
            int(barline_patch_partitions.get(name, -1) or -1)
            == int(group_count or 0) * barline_patch_samples_per_group
            for name, group_count in barline_patch_partition_groups.items()
        )
        and len(barline_patch_dataset_fingerprint) == 64
        and len(barline_patch_confirmation_fingerprint) == 64
        and barline_patch_dataset_fingerprint != barline_patch_confirmation_fingerprint
        and int(barline_patch_confirmation.get("samples", 0) or 0)
        == int(barline_patch_confirmation.get("groups", 0) or 0)
        * barline_patch_samples_per_group
        and str(barline_patch_training.get("model_sha256", ""))
        == barline_patch_model_hash
        and float(barline_patch_training.get("policy_probability_floor", 0.0) or 0.0)
        == DEFAULT_POLICY.barline_patch_probability_floor
        and isinstance(barline_patch_policy, dict)
        and float(barline_patch_policy.get("precision", 0.0) or 0.0) >= 0.999
        and float(barline_patch_policy.get("positive_recall", 0.0) or 0.0) >= 0.90
        and int(barline_patch_policy.get("false_accepts", 1) or 0) == 0
        and isinstance(barline_patch_baseline, dict)
        and int(barline_patch_baseline.get("false_accepts", 0) or 0) >= 1_200
        and isinstance(barline_patch_confirmation_policy, dict)
        and float(barline_patch_confirmation_policy.get("precision", 0.0) or 0.0) >= 0.999
        and float(barline_patch_confirmation_policy.get("positive_recall", 0.0) or 0.0) >= 0.90
        and int(barline_patch_confirmation_policy.get("false_accepts", 1) or 0) == 0
        and isinstance(barline_patch_confirmation_baseline, dict)
        and int(barline_patch_confirmation_baseline.get("false_accepts", 0) or 0) >= 4_000
        and isinstance(barline_patch_deployment, dict)
        and float(barline_patch_deployment.get("max_absolute_probability_delta", 1.0) or 1.0) <= 1e-10
        and int(barline_patch_training.get("model_bytes", 10**9) or 10**9) <= 2_000_000,
        "repeat/barline consensus uses grouped leak-free training and a reproducible CPU veto with zero maintained harmful accepts",
    )

    cross_tie_training = read_json(
        source_root / "training" / "cross_tie_patch_calibrator_report_v2.json",
        {},
    )
    cross_tie_frozen = (
        cross_tie_training.get("frozen_test", {})
        if isinstance(cross_tie_training, dict)
        else {}
    )
    cross_tie_policy = (
        cross_tie_frozen.get("policy", {}) if isinstance(cross_tie_frozen, dict) else {}
    )
    cross_tie_baseline = (
        cross_tie_frozen.get("accept_all_deterministic_proposals", {})
        if isinstance(cross_tie_frozen, dict)
        else {}
    )
    cross_tie_confirmation = (
        cross_tie_training.get("independent_confirmation", {})
        if isinstance(cross_tie_training, dict)
        else {}
    )
    cross_tie_confirmation_policy = (
        cross_tie_confirmation.get("policy", {})
        if isinstance(cross_tie_confirmation, dict)
        else {}
    )
    cross_tie_confirmation_baseline = (
        cross_tie_confirmation.get("accept_all_deterministic_proposals", {})
        if isinstance(cross_tie_confirmation, dict)
        else {}
    )
    cross_tie_safety = (
        cross_tie_training.get("safety_calibration", {})
        if isinstance(cross_tie_training, dict)
        else {}
    )
    cross_tie_safety_policy = (
        cross_tie_safety.get("policy", {})
        if isinstance(cross_tie_safety, dict)
        else {}
    )
    cross_tie_v1_frozen = (
        cross_tie_frozen.get("v1_same_data_policy", {})
        if isinstance(cross_tie_frozen, dict)
        else {}
    )
    cross_tie_v1_confirmation = (
        cross_tie_confirmation.get("v1_same_data_policy", {})
        if isinstance(cross_tie_confirmation, dict)
        else {}
    )
    cross_tie_secondary_confirmation = (
        cross_tie_training.get("secondary_confirmation", {})
        if isinstance(cross_tie_training, dict)
        else {}
    )
    cross_tie_secondary_policy = (
        cross_tie_secondary_confirmation.get("policy", {})
        if isinstance(cross_tie_secondary_confirmation, dict)
        else {}
    )
    cross_tie_v1_secondary = (
        cross_tie_secondary_confirmation.get("v1_same_data_policy", {})
        if isinstance(cross_tie_secondary_confirmation, dict)
        else {}
    )
    cross_tie_deployment = (
        cross_tie_training.get("deployment_parity", {})
        if isinstance(cross_tie_training, dict)
        else {}
    )
    cross_tie_leakage = (
        cross_tie_training.get("group_leakage_audit", {})
        if isinstance(cross_tie_training, dict)
        else {}
    )
    cross_tie_overlaps = (
        cross_tie_leakage.get("pairwise_group_overlap", {})
        if isinstance(cross_tie_leakage, dict)
        else {}
    )
    cross_tie_partition_groups = (
        cross_tie_leakage.get("groups_per_partition", {})
        if isinstance(cross_tie_leakage, dict)
        else {}
    )
    cross_tie_partitions = (
        cross_tie_training.get("partitions", {})
        if isinstance(cross_tie_training, dict)
        else {}
    )
    cross_tie_groups = (
        int(cross_tie_training.get("groups", 0) or 0)
        if isinstance(cross_tie_training, dict)
        else 0
    )
    cross_tie_samples_per_group = (
        int(cross_tie_training.get("samples_per_group", 0) or 0)
        if isinstance(cross_tie_training, dict)
        else 0
    )
    cross_tie_model_path = resources / "cross_tie_patch_calibrator.json"
    cross_tie_model_hash = (
        sha256_file(cross_tie_model_path) if cross_tie_model_path.is_file() else ""
    )
    cross_tie_dataset_fingerprint = (
        str(cross_tie_training.get("dataset_fingerprint", ""))
        if isinstance(cross_tie_training, dict)
        else ""
    )
    cross_tie_confirmation_fingerprint = (
        str(cross_tie_confirmation.get("dataset_fingerprint", ""))
        if isinstance(cross_tie_confirmation, dict)
        else ""
    )
    cross_tie_safety_fingerprint = (
        str(cross_tie_safety.get("dataset_fingerprint", ""))
        if isinstance(cross_tie_safety, dict)
        else ""
    )
    cross_tie_secondary_fingerprint = (
        str(cross_tie_secondary_confirmation.get("dataset_fingerprint", ""))
        if isinstance(cross_tie_secondary_confirmation, dict)
        else ""
    )
    add(
        "cross-tie-patch:cpu-training",
        cross_tie_training.get("model_version") == "scorescan-cross-tie-patch-forest-2"
        and cross_tie_samples_per_group == 3
        and cross_tie_groups >= 9_000
        and int(cross_tie_training.get("samples", 0) or 0)
        == cross_tie_groups * cross_tie_samples_per_group
        and isinstance(cross_tie_leakage, dict)
        and not bool(cross_tie_leakage.get("leakage_detected", True))
        and isinstance(cross_tie_overlaps, dict)
        and len(cross_tie_overlaps) == 10
        and all(int(value or 0) == 0 for value in cross_tie_overlaps.values())
        and isinstance(cross_tie_partition_groups, dict)
        and sum(int(value or 0) for value in cross_tie_partition_groups.values())
        == cross_tie_groups
        and isinstance(cross_tie_partitions, dict)
        and all(
            int(cross_tie_partitions.get(name, -1) or -1)
            == int(group_count or 0) * cross_tie_samples_per_group
            for name, group_count in cross_tie_partition_groups.items()
        )
        and len(cross_tie_dataset_fingerprint) == 64
        and len(cross_tie_safety_fingerprint) == 64
        and len(cross_tie_confirmation_fingerprint) == 64
        and len(cross_tie_secondary_fingerprint) == 64
        and len({
            cross_tie_dataset_fingerprint,
            cross_tie_safety_fingerprint,
            cross_tie_confirmation_fingerprint,
            cross_tie_secondary_fingerprint,
        }) == 4
        and int(cross_tie_safety.get("groups", 0) or 0) >= 4_000
        and int(cross_tie_safety.get("samples", 0) or 0)
        == int(cross_tie_safety.get("groups", 0) or 0) * cross_tie_samples_per_group
        and isinstance(cross_tie_safety_policy, dict)
        and int(cross_tie_safety_policy.get("false_accepts", 1) or 0) == 0
        and float(cross_tie_safety_policy.get("precision", 0.0) or 0.0) == 1.0
        and int(cross_tie_confirmation.get("samples", 0) or 0)
        == int(cross_tie_confirmation.get("groups", 0) or 0)
        * cross_tie_samples_per_group
        and str(cross_tie_training.get("model_sha256", "")) == cross_tie_model_hash
        and float(cross_tie_training.get("policy_probability_floor", 0.0) or 0.0)
        == DEFAULT_POLICY.cross_tie_patch_probability_floor
        and isinstance(cross_tie_policy, dict)
        and float(cross_tie_policy.get("precision", 0.0) or 0.0) == 1.0
        and float(cross_tie_policy.get("positive_recall", 0.0) or 0.0) >= 0.97
        and int(cross_tie_policy.get("false_accepts", 1) or 0) == 0
        and isinstance(cross_tie_v1_frozen, dict)
        and int(cross_tie_v1_frozen.get("false_accepts", 0) or 0) >= 1
        and int(cross_tie_policy.get("true_accepts", 0) or 0)
        > int(cross_tie_v1_frozen.get("true_accepts", 0) or 0)
        and isinstance(cross_tie_baseline, dict)
        and int(cross_tie_baseline.get("false_accepts", 0) or 0) >= 1_300
        and isinstance(cross_tie_confirmation_policy, dict)
        and float(cross_tie_confirmation_policy.get("precision", 0.0) or 0.0) == 1.0
        and float(cross_tie_confirmation_policy.get("positive_recall", 0.0) or 0.0) >= 0.96
        and int(cross_tie_confirmation_policy.get("false_accepts", 1) or 0) == 0
        and isinstance(cross_tie_v1_confirmation, dict)
        and int(cross_tie_v1_confirmation.get("false_accepts", 0) or 0) >= 1
        and int(cross_tie_confirmation_policy.get("true_accepts", 0) or 0)
        > int(cross_tie_v1_confirmation.get("true_accepts", 0) or 0)
        and isinstance(cross_tie_confirmation_baseline, dict)
        and int(cross_tie_confirmation_baseline.get("false_accepts", 0) or 0) >= 4_400
        and int(cross_tie_secondary_confirmation.get("groups", 0) or 0) >= 3_000
        and int(cross_tie_secondary_confirmation.get("samples", 0) or 0)
        == int(cross_tie_secondary_confirmation.get("groups", 0) or 0)
        * cross_tie_samples_per_group
        and isinstance(cross_tie_secondary_policy, dict)
        and float(cross_tie_secondary_policy.get("precision", 0.0) or 0.0) == 1.0
        and float(cross_tie_secondary_policy.get("positive_recall", 0.0) or 0.0) >= 0.96
        and int(cross_tie_secondary_policy.get("false_accepts", 1) or 0) == 0
        and isinstance(cross_tie_v1_secondary, dict)
        and int(cross_tie_v1_secondary.get("false_accepts", 0) or 0) >= 1
        and int(cross_tie_secondary_policy.get("true_accepts", 0) or 0)
        > int(cross_tie_v1_secondary.get("true_accepts", 0) or 0)
        and isinstance(cross_tie_deployment, dict)
        and float(cross_tie_deployment.get("max_absolute_probability_delta", 1.0) or 1.0)
        <= 1e-10
        and int(cross_tie_training.get("model_bytes", 10**9) or 10**9) <= 2_000_000,
        "cross-measure tie consensus uses grouped leak-free training and a reproducible CPU veto with zero maintained harmful accepts",
    )

    context_training = read_json(
        source_root / "training" / "context_calibrator_report_v2.json",
        {},
    )
    context_test = (
        context_training.get("frozen_test", {})
        if isinstance(context_training, dict)
        else {}
    )
    context_sample = (
        context_test.get("sample", {}) if isinstance(context_test, dict) else {}
    )
    context_decision = (
        context_test.get("decision", {}) if isinstance(context_test, dict) else {}
    )
    context_scenarios = (
        context_test.get("by_scenario", {}) if isinstance(context_test, dict) else {}
    )
    context_transition = (
        context_scenarios.get("legitimate-transition", {})
        if isinstance(context_scenarios, dict)
        else {}
    )
    context_cross_family = (
        context_scenarios.get("cross-family-correlated-sequence-error", {})
        if isinstance(context_scenarios, dict)
        else {}
    )
    context_neutral = (
        context_scenarios.get("context-neutral-internal-error", {})
        if isinstance(context_scenarios, dict)
        else {}
    )
    context_baseline = (
        context_training.get("baseline_v1_same_frozen_test", {})
        if isinstance(context_training, dict)
        else {}
    )
    context_baseline_decision = (
        context_baseline.get("decision", {})
        if isinstance(context_baseline, dict)
        else {}
    )
    context_baseline_scenarios = (
        context_baseline.get("by_scenario", {})
        if isinstance(context_baseline, dict)
        else {}
    )
    context_baseline_transition = (
        context_baseline_scenarios.get("legitimate-transition", {})
        if isinstance(context_baseline_scenarios, dict)
        else {}
    )
    context_baseline_cross_family = (
        context_baseline_scenarios.get("cross-family-correlated-sequence-error", {})
        if isinstance(context_baseline_scenarios, dict)
        else {}
    )
    context_baseline_neutral = (
        context_baseline_scenarios.get("context-neutral-internal-error", {})
        if isinstance(context_baseline_scenarios, dict)
        else {}
    )
    context_ablation = (
        context_training.get("family_feature_ablation", {})
        if isinstance(context_training, dict)
        else {}
    )
    context_ablation_decision = (
        context_ablation.get("decision", {})
        if isinstance(context_ablation, dict)
        else {}
    )
    context_deployment = (
        context_training.get("deployment_parity", {})
        if isinstance(context_training, dict)
        else {}
    )
    context_config = (
        context_training.get("selected_config", {})
        if isinstance(context_training, dict)
        else {}
    )
    add(
        "context-family:cpu-training",
        context_training.get("model_version") == "scorescan-context-forest-2"
        and isinstance(context_sample, dict)
        and float(context_sample.get("roc_auc", 0.0) or 0.0) >= 0.83
        and float(context_sample.get("log_loss", 1.0) or 1.0) <= 0.48
        and isinstance(context_decision, dict)
        and float(context_decision.get("top1_accuracy", 0.0) or 0.0) >= 0.84
        and isinstance(context_baseline_decision, dict)
        and float(context_decision.get("top1_accuracy", 0.0) or 0.0)
        >= float(context_baseline_decision.get("top1_accuracy", 0.0) or 0.0) + 0.02
        and isinstance(context_ablation_decision, dict)
        and float(context_decision.get("top1_accuracy", 0.0) or 0.0)
        >= float(context_ablation_decision.get("top1_accuracy", 0.0) or 0.0) + 0.04
        and isinstance(context_transition, dict)
        and isinstance(context_baseline_transition, dict)
        and float(context_transition.get("top1_accuracy", 0.0) or 0.0)
        >= float(context_baseline_transition.get("top1_accuracy", 0.0) or 0.0) + 0.15
        and isinstance(context_cross_family, dict)
        and isinstance(context_baseline_cross_family, dict)
        and float(context_cross_family.get("top1_accuracy", 0.0) or 0.0) >= 0.60
        and float(context_cross_family.get("top1_accuracy", 0.0) or 0.0)
        >= float(context_baseline_cross_family.get("top1_accuracy", 0.0) or 0.0) - 0.10
        and isinstance(context_neutral, dict)
        and isinstance(context_baseline_neutral, dict)
        and float(context_neutral.get("top1_accuracy", 0.0) or 0.0)
        >= float(context_baseline_neutral.get("top1_accuracy", 0.0) or 0.0) - 0.03
        and isinstance(context_deployment, dict)
        and float(context_deployment.get("max_absolute_probability_delta", 1.0) or 1.0) <= 1e-10
        and isinstance(context_config, dict)
        and int(context_config.get("n_estimators", 10**9) or 10**9) <= 24,
        "family-balanced context v2 improves grouped sequence recovery with bounded hard-case regression",
    )

    event_training = read_json(source_root / "training" / "event_calibrator_report_v2.json", {})
    event_test = event_training.get("frozen_test", {}) if isinstance(event_training, dict) else {}
    event_sample = event_test.get("sample", {}) if isinstance(event_test, dict) else {}
    event_decision = event_test.get("decision", {}) if isinstance(event_test, dict) else {}
    event_scenarios = event_test.get("by_scenario", {}) if isinstance(event_test, dict) else {}
    event_cross_family = (
        event_scenarios.get("cross-family-correlated-error", {})
        if isinstance(event_scenarios, dict)
        else {}
    )
    event_baseline = (
        event_training.get("baseline_v1_same_frozen_test", {})
        if isinstance(event_training, dict)
        else {}
    )
    event_baseline_decision = (
        event_baseline.get("decision", {}) if isinstance(event_baseline, dict) else {}
    )
    event_baseline_scenarios = (
        event_baseline.get("by_scenario", {}) if isinstance(event_baseline, dict) else {}
    )
    event_baseline_cross_family = (
        event_baseline_scenarios.get("cross-family-correlated-error", {})
        if isinstance(event_baseline_scenarios, dict)
        else {}
    )
    event_ablation = (
        event_training.get("family_feature_ablation", {})
        if isinstance(event_training, dict)
        else {}
    )
    event_ablation_decision = (
        event_ablation.get("decision", {}) if isinstance(event_ablation, dict) else {}
    )
    event_deployment = (
        event_training.get("deployment_parity", {})
        if isinstance(event_training, dict)
        else {}
    )
    event_config = (
        event_training.get("selected_config", {})
        if isinstance(event_training, dict)
        else {}
    )
    add(
        "event-family:cpu-training",
        event_training.get("model_version") == "scorescan-event-forest-2"
        and isinstance(event_sample, dict)
        and float(event_sample.get("roc_auc", 0.0) or 0.0) >= 0.95
        and float(event_sample.get("log_loss", 1.0) or 1.0) <= 0.30
        and isinstance(event_decision, dict)
        and float(event_decision.get("top1_accuracy", 0.0) or 0.0) >= 0.89
        and isinstance(event_baseline_decision, dict)
        and float(event_decision.get("top1_accuracy", 0.0) or 0.0)
        >= float(event_baseline_decision.get("top1_accuracy", 0.0) or 0.0) + 0.08
        and isinstance(event_ablation_decision, dict)
        and float(event_decision.get("top1_accuracy", 0.0) or 0.0)
        >= float(event_ablation_decision.get("top1_accuracy", 0.0) or 0.0) + 0.08
        and isinstance(event_cross_family, dict)
        and float(event_cross_family.get("top1_accuracy", 0.0) or 0.0) >= 0.75
        and isinstance(event_baseline_cross_family, dict)
        and float(event_cross_family.get("top1_accuracy", 0.0) or 0.0)
        >= float(event_baseline_cross_family.get("top1_accuracy", 0.0) or 0.0) + 0.45
        and isinstance(event_deployment, dict)
        and float(event_deployment.get("max_absolute_probability_delta", 1.0) or 1.0) <= 1e-10
        and isinstance(event_config, dict)
        and int(event_config.get("n_estimators", 10**9) or 10**9) <= 48,
        "family-balanced event v2 improves grouped candidate recovery and deployment parity",
    )

    measure_training = read_json(
        source_root / "training" / "measure_calibrator_report_v3.json",
        {},
    )
    measure_test = (
        measure_training.get("frozen_test", {})
        if isinstance(measure_training, dict)
        else {}
    )
    measure_sample = measure_test.get("sample", {}) if isinstance(measure_test, dict) else {}
    measure_decision = measure_test.get("decision", {}) if isinstance(measure_test, dict) else {}
    measure_scenarios = measure_test.get("by_scenario", {}) if isinstance(measure_test, dict) else {}
    measure_pickup = measure_scenarios.get("pickup_boundary", {}) if isinstance(measure_scenarios, dict) else {}
    measure_final = measure_scenarios.get("final_boundary", {}) if isinstance(measure_scenarios, dict) else {}
    measure_baseline = (
        measure_training.get("baseline_v2_same_frozen_test", {})
        if isinstance(measure_training, dict)
        else {}
    )
    measure_baseline_decision = (
        measure_baseline.get("decision", {}) if isinstance(measure_baseline, dict) else {}
    )
    measure_ablation = (
        measure_training.get("new_feature_ablation", {})
        if isinstance(measure_training, dict)
        else {}
    )
    measure_ablation_decision = (
        measure_ablation.get("decision", {}) if isinstance(measure_ablation, dict) else {}
    )
    measure_deployment = (
        measure_training.get("deployment_parity", {})
        if isinstance(measure_training, dict)
        else {}
    )
    measure_config = (
        measure_training.get("selected_config", {})
        if isinstance(measure_training, dict)
        else {}
    )
    add(
        "measure-structure:cpu-training",
        measure_training.get("model_version") == "scorescan-measure-forest-3"
        and isinstance(measure_sample, dict)
        and float(measure_sample.get("roc_auc", 0.0) or 0.0) >= 0.99
        and float(measure_sample.get("log_loss", 1.0) or 1.0) <= 0.13
        and isinstance(measure_decision, dict)
        and float(measure_decision.get("top1_accuracy", 0.0) or 0.0) >= 0.98
        and isinstance(measure_baseline_decision, dict)
        and float(measure_decision.get("top1_accuracy", 0.0) or 0.0)
        >= float(measure_baseline_decision.get("top1_accuracy", 0.0) or 0.0) + 0.20
        and isinstance(measure_ablation_decision, dict)
        and float(measure_decision.get("top1_accuracy", 0.0) or 0.0)
        >= float(measure_ablation_decision.get("top1_accuracy", 0.0) or 0.0) + 0.04
        and isinstance(measure_pickup, dict)
        and float(measure_pickup.get("top1_accuracy", 0.0) or 0.0) >= 0.99
        and isinstance(measure_final, dict)
        and float(measure_final.get("top1_accuracy", 0.0) or 0.0) >= 0.99
        and isinstance(measure_deployment, dict)
        and float(measure_deployment.get("max_absolute_probability_delta", 1.0) or 1.0) <= 1e-10
        and isinstance(measure_config, dict)
        and int(measure_config.get("n_estimators", 10**9) or 10**9) <= 48
        and 0.50 <= float(measure_training.get("selected_legacy_preservation_floor", -1.0) or -1.0) <= 0.65,
        "measure v3 improves grouped structure decisions while preserving the legacy probability scale",
    )

    ensemble_training = read_json(
        source_root / "training" / "ensemble_calibrator_report_v3.json",
        {},
    )
    ensemble_test = (
        ensemble_training.get("frozen_test", {})
        if isinstance(ensemble_training, dict)
        else {}
    )
    ensemble_sample = (
        ensemble_test.get("sample", {}) if isinstance(ensemble_test, dict) else {}
    )
    ensemble_policy = (
        ensemble_test.get("policy_gate", {}) if isinstance(ensemble_test, dict) else {}
    )
    ensemble_decision = (
        ensemble_test.get("decision", {}) if isinstance(ensemble_test, dict) else {}
    )
    ensemble_scenarios = (
        ensemble_test.get("by_scenario", {}) if isinstance(ensemble_test, dict) else {}
    )
    ensemble_baseline = (
        ensemble_training.get("baseline_v2_same_frozen_test", {})
        if isinstance(ensemble_training, dict)
        else {}
    )
    ensemble_baseline_policy = (
        ensemble_baseline.get("policy_gate", {})
        if isinstance(ensemble_baseline, dict)
        else {}
    )
    ensemble_baseline_decision = (
        ensemble_baseline.get("decision", {})
        if isinstance(ensemble_baseline, dict)
        else {}
    )
    ensemble_linear = (
        ensemble_training.get("linear_model_ablation", {})
        if isinstance(ensemble_training, dict)
        else {}
    )
    ensemble_linear_decision = (
        ensemble_linear.get("decision", {})
        if isinstance(ensemble_linear, dict)
        else {}
    )
    ensemble_deployment = (
        ensemble_training.get("deployment_parity", {})
        if isinstance(ensemble_training, dict)
        else {}
    )
    ensemble_config = (
        ensemble_training.get("selected_config", {})
        if isinstance(ensemble_training, dict)
        else {}
    )
    ensemble_clean_majority = (
        ensemble_scenarios.get("clean-majority", {})
        if isinstance(ensemble_scenarios, dict)
        else {}
    )
    ensemble_clean_agreement = (
        ensemble_scenarios.get("clean-agreement", {})
        if isinstance(ensemble_scenarios, dict)
        else {}
    )
    ensemble_cross_family = (
        ensemble_scenarios.get("cross-family-fuzzy-trap", {})
        if isinstance(ensemble_scenarios, dict)
        else {}
    )
    ensemble_localized_rescue = ensemble_scenarios.get("localized-rescue", {}) if isinstance(ensemble_scenarios, dict) else {}
    ensemble_localized_isolation = ensemble_scenarios.get("localized-isolation-trap", {}) if isinstance(ensemble_scenarios, dict) else {}
    ensemble_localized_partial = ensemble_scenarios.get("localized-partial-trap", {}) if isinstance(ensemble_scenarios, dict) else {}
    add(
        "ensemble-meta:cpu-training",
        ensemble_training.get("model_version") == "scorescan-ensemble-forest-3"
        and isinstance(ensemble_sample, dict)
        and float(ensemble_sample.get("roc_auc", 0.0) or 0.0) >= 0.98
        and float(ensemble_sample.get("log_loss", 1.0) or 1.0) <= 0.17
        and isinstance(ensemble_decision, dict)
        and float(ensemble_decision.get("top1_accuracy", 0.0) or 0.0) >= 0.875
        and isinstance(ensemble_baseline_decision, dict)
        and float(ensemble_decision.get("top1_accuracy", 0.0) or 0.0)
        >= float(ensemble_baseline_decision.get("top1_accuracy", 0.0) or 0.0) + 0.02
        and isinstance(ensemble_linear_decision, dict)
        and float(ensemble_decision.get("top1_accuracy", 0.0) or 0.0)
        >= float(ensemble_linear_decision.get("top1_accuracy", 0.0) or 0.0) + 0.07
        and isinstance(ensemble_policy, dict)
        and isinstance(ensemble_baseline_policy, dict)
        and int(ensemble_policy.get("false_accepts", 10**9) or 10**9)
        <= int(ensemble_baseline_policy.get("false_accepts", 10**9) or 10**9)
        and isinstance(ensemble_clean_majority, dict)
        and float(ensemble_clean_majority.get("top1_accuracy", 0.0) or 0.0) >= 0.99
        and isinstance(ensemble_clean_agreement, dict)
        and float(ensemble_clean_agreement.get("top1_accuracy", 0.0) or 0.0) >= 0.99
        and isinstance(ensemble_cross_family, dict)
        and float(ensemble_cross_family.get("top1_accuracy", 0.0) or 0.0) >= 0.55
        and isinstance(ensemble_localized_rescue, dict)
        and float(ensemble_localized_rescue.get("top1_accuracy", 0.0) or 0.0) >= 0.80
        and isinstance(ensemble_localized_isolation, dict)
        and float(ensemble_localized_isolation.get("top1_accuracy", 0.0) or 0.0) >= 0.80
        and isinstance(ensemble_localized_partial, dict)
        and float(ensemble_localized_partial.get("top1_accuracy", 0.0) or 0.0) >= 0.80
        and isinstance(ensemble_deployment, dict)
        and float(ensemble_deployment.get("max_absolute_probability_delta", 1.0) or 1.0) <= 1e-10
        and isinstance(ensemble_config, dict)
        and int(ensemble_config.get("n_estimators", 10**9) or 10**9) <= 48,
        "ensemble v3 recalibrates difficult 3-8 candidate decisions across five independent families",
    )

    ensemble_compatibility = read_json(
        source_root / "training" / "ensemble_component_compatibility_audit_v3.json",
        {},
    )
    compatibility_checks = (
        ensemble_compatibility.get("checks", {})
        if isinstance(ensemble_compatibility, dict)
        else {}
    )
    compatibility_models = (
        ensemble_compatibility.get("current_models", {})
        if isinstance(ensemble_compatibility, dict)
        else {}
    )
    add(
        "ensemble-meta:component-compatibility",
        ensemble_compatibility.get("audit_version") == "scorescan-ensemble-compatibility-audit-3"
        and bool(ensemble_compatibility.get("passed"))
        and isinstance(compatibility_checks, dict)
        and all(bool(value) for value in compatibility_checks.values())
        and isinstance(compatibility_models, dict)
        and compatibility_models.get("ensemble") == "scorescan-ensemble-forest-3"
        and compatibility_models.get("measure") == "scorescan-measure-forest-3"
        and compatibility_models.get("visual") == "scorescan-visual-measure-calibrator-4",
        "ensemble v3 remains within frozen bounds across whole-page and localized candidate families",
    )

    direction_training = read_json(source_root / "training" / "direction_model_report_v9.json", {})
    direction_pair = direction_training.get("pair_frozen_test", {}) if isinstance(direction_training, dict) else {}
    direction_baseline_pair = (
        direction_training.get("baseline_v8_pair_same_frozen_test", {})
        if isinstance(direction_training, dict)
        else {}
    )
    direction_phrase = (
        direction_training.get("phrase_decoder_frozen_test", {})
        if isinstance(direction_training, dict)
        else {}
    )
    direction_safety_audit = (
        direction_training.get("phrase_safety_audit", {})
        if isinstance(direction_training, dict)
        else {}
    )
    direction_baseline_phrase = (
        direction_training.get("baseline_v8_phrase_decoder_same_test", {})
        if isinstance(direction_training, dict)
        else {}
    )
    direction_compositional = (
        direction_training.get("compositional_frozen_test", {})
        if isinstance(direction_training, dict)
        else {}
    )
    direction_baseline_compositional = (
        direction_training.get("baseline_v8_compositional_same_decoder", {})
        if isinstance(direction_training, dict)
        else {}
    )
    direction_rendered_compositional = (
        direction_training.get("compositional_rendered_ocr_frozen_test", {})
        if isinstance(direction_training, dict)
        else {}
    )
    direction_baseline_rendered_compositional = (
        direction_training.get("baseline_v8_compositional_rendered_ocr_same_decoder", {})
        if isinstance(direction_training, dict)
        else {}
    )
    direction_deployment = (
        direction_training.get("deployment_parity", {})
        if isinstance(direction_training, dict)
        else {}
    )
    add(
        "direction-text:cpu-training",
        direction_training.get("model_version") == "scorescan-direction-logistic-9"
        and isinstance(direction_pair, dict)
        and float(direction_pair.get("roc_auc", 0.0) or 0.0) >= 0.999
        and float(direction_pair.get("log_loss", 1.0) or 1.0) <= 0.04
        and int(direction_pair.get("false_accepts", 10**9) or 10**9)
        < int(direction_baseline_pair.get("false_accepts", 0) or 0)
        and isinstance(direction_deployment, dict)
        and float(direction_deployment.get("max_absolute_probability_delta", 1.0) or 1.0) <= 1e-10
        and isinstance(direction_safety_audit, dict)
        and float(direction_safety_audit.get("autocorrect_precision", 0.0) or 0.0) >= 0.999
        and not direction_safety_audit.get("autocorrect_errors")
        and int(direction_safety_audit.get("autocorrect_count", 0) or 0) >= 35
        and isinstance(direction_phrase, dict)
        and float(direction_phrase.get("top1_accuracy", 0.0) or 0.0)
        >= float(direction_baseline_phrase.get("top1_accuracy", 0.0) or 0.0)
        and float(direction_phrase.get("autocorrect_coverage", 0.0) or 0.0)
        > float(direction_baseline_phrase.get("autocorrect_coverage", 0.0) or 0.0)
        and float(direction_phrase.get("autocorrect_precision", 0.0) or 0.0) >= 0.999
        and not direction_phrase.get("autocorrect_errors")
        and isinstance(direction_compositional, dict)
        and float(direction_compositional.get("top1_accuracy", 0.0) or 0.0) >= 0.89
        and float(direction_compositional.get("autocorrect_coverage", 0.0) or 0.0) >= 0.15
        and float(direction_compositional.get("autocorrect_precision", 0.0) or 0.0) >= 0.999
        and not direction_compositional.get("autocorrect_errors")
        and float(direction_compositional.get("autocorrect_coverage", 0.0) or 0.0)
        > float(direction_baseline_compositional.get("autocorrect_coverage", 0.0) or 0.0)
        and isinstance(direction_rendered_compositional, dict)
        and float(direction_rendered_compositional.get("top1_accuracy", 0.0) or 0.0) >= 0.77
        and float(direction_rendered_compositional.get("autocorrect_precision", 0.0) or 0.0) >= 0.999
        and not direction_rendered_compositional.get("autocorrect_errors")
        and int(direction_rendered_compositional.get("autocorrect_count", 0) or 0)
        > int(direction_baseline_rendered_compositional.get("autocorrect_count", 0) or 0),
        "direction v9 improves grouped matching and safe unseen-phrase OCR correction without unattended errors",
    )

    direction_anchor_training = read_json(
        source_root / "training" / "direction_measure_anchor_report_v1.json", {}
    )
    anchor_audit = (
        direction_anchor_training.get("audit", {})
        if isinstance(direction_anchor_training, dict)
        else {}
    )
    anchor_test = (
        direction_anchor_training.get("test_decisions", {})
        if isinstance(direction_anchor_training, dict)
        else {}
    )
    anchor_config = (
        direction_anchor_training.get("selected_config", {})
        if isinstance(direction_anchor_training, dict)
        else {}
    )
    add(
        "direction-anchor:measure-ownership",
        direction_anchor_training.get("model_version") == "scorescan-direction-anchor-hybrid-2"
        and direction_anchor_training.get("anchor_model_version")
        == "scorescan-direction-measure-anchor-forest-1"
        and isinstance(anchor_audit, dict)
        and float(anchor_audit.get("changed_precision", 0.0) or 0.0) >= 0.999
        and int(anchor_audit.get("changed_errors", 1) or 0) == 0
        and isinstance(anchor_test, dict)
        and float(anchor_test.get("refined_accuracy", 0.0) or 0.0)
        >= float(anchor_test.get("baseline_accuracy", 0.0) or 0.0) + 0.01
        and float(anchor_test.get("changed_precision", 0.0) or 0.0) >= 0.999
        and int(anchor_test.get("changed_errors", 1) or 0) == 0
        and int(anchor_test.get("changed_decisions", 0) or 0) >= 20
        and float(
            direction_anchor_training.get(
                "deployment_max_abs_probability_difference", 1.0
            )
            or 1.0
        )
        <= 1e-10
        and isinstance(anchor_config, dict)
        and int(anchor_config.get("n_estimators", 10**9) or 10**9) <= 32,
        "direction ownership v2 safely corrects count-mismatch measure anchors on grouped holdout",
    )

    visual_training = read_json(source_root / "training" / "visual_measure_calibrator_report_v4.json", {})
    visual_frozen = visual_training.get("frozen_test", {}) if isinstance(visual_training, dict) else {}
    visual_test = visual_frozen.get("v4", {}) if isinstance(visual_frozen, dict) else {}
    visual_sample = visual_test.get("sample", {}) if isinstance(visual_test, dict) else {}
    visual_decision = visual_test.get("decision", {}) if isinstance(visual_test, dict) else {}
    visual_pairwise = (
        visual_decision.get("compatible_over_trap", {})
        if isinstance(visual_decision, dict)
        else {}
    )
    visual_baseline = visual_frozen.get("v3_same_test", {}) if isinstance(visual_frozen, dict) else {}
    visual_baseline_sample = (
        visual_baseline.get("sample", {}) if isinstance(visual_baseline, dict) else {}
    )
    visual_baseline_decision = (
        visual_baseline.get("decision", {}) if isinstance(visual_baseline, dict) else {}
    )
    visual_event_ablation = (
        visual_frozen.get("event_grid_ablation", {}) if isinstance(visual_frozen, dict) else {}
    )
    visual_event_ablation_sample = (
        visual_event_ablation.get("sample", {}) if isinstance(visual_event_ablation, dict) else {}
    )
    visual_baseline_pairwise = (
        visual_baseline_decision.get("compatible_over_trap", {})
        if isinstance(visual_baseline_decision, dict)
        else {}
    )
    local_traps = (
        "pitch-order-trap",
        "accidental-position-trap",
        "compact-position-trap",
        "open-notehead-position-trap",
        "event-kind-position-trap",
        "rhythm-position-trap",
    )
    visual_local_mean = (
        sum(float(visual_pairwise.get(name, 0.0) or 0.0) for name in local_traps) / len(local_traps)
        if isinstance(visual_pairwise, dict)
        else 0.0
    )
    visual_baseline_local_mean = (
        sum(float(visual_baseline_pairwise.get(name, 0.0) or 0.0) for name in local_traps)
        / len(local_traps)
        if isinstance(visual_baseline_pairwise, dict)
        else 0.0
    )
    deployment = visual_training.get("deployment_parity", {}) if isinstance(visual_training, dict) else {}
    add(
        "visual-measure:cpu-training",
        visual_training.get("model_version") == "scorescan-visual-measure-calibrator-4"
        and visual_training.get("baseline_model_version") == "scorescan-visual-measure-calibrator-3"
        and isinstance(visual_sample, dict)
        and isinstance(visual_baseline_sample, dict)
        and float(visual_sample.get("roc_auc", 0.0) or 0.0)
        >= float(visual_baseline_sample.get("roc_auc", 0.0) or 0.0) + 0.02
        and float(visual_sample.get("log_loss", 1.0) or 1.0)
        <= float(visual_baseline_sample.get("log_loss", 1.0) or 1.0) - 0.05
        and int(visual_sample.get("false_accepts", 10**9) or 10**9)
        < int(visual_baseline_sample.get("false_accepts", 0) or 0)
        and isinstance(visual_decision, dict)
        and isinstance(visual_baseline_decision, dict)
        and float(visual_decision.get("compatible_top1", 0.0) or 0.0)
        >= float(visual_baseline_decision.get("compatible_top1", 0.0) or 0.0) + 0.01
        and isinstance(visual_event_ablation_sample, dict)
        and float(visual_sample.get("roc_auc", 0.0) or 0.0)
        >= float(visual_event_ablation_sample.get("roc_auc", 0.0) or 0.0) + 0.025
        and visual_local_mean >= visual_baseline_local_mean + 0.15
        and isinstance(visual_pairwise, dict)
        and float(visual_pairwise.get("pitch-order-trap", 0.0) or 0.0) >= 0.55
        and float(visual_pairwise.get("open-notehead-position-trap", 0.0) or 0.0) >= 0.80
        and float(visual_pairwise.get("event-kind-position-trap", 0.0) or 0.0) >= 0.88
        and isinstance(deployment, dict)
        and float(deployment.get("max_absolute_probability_delta", 1.0) or 1.0) <= 1e-10
        and int(visual_training.get("model_bytes", 10**9) or 10**9) <= 3_200_000,
        "visual v4 adds event-local order and attachment evidence with frozen same-test gain over v3",
    )

    # Importing the checker itself may create __pycache__ entries. Release packaging
    # excludes those deterministically, so only non-cache temporary files are blockers.
    leftovers = find_non_cache_temporary_files(source_root)
    add("tree:clean", not leftovers, f"non-cache temporary artifacts: {len(leftovers)}")
    add("launcher:source", (source_root / "launcher.zig").is_file(), "Windows launcher source present")
    add(
        "launcher:zig-license",
        (source_root / "licenses" / "LICENSE-MIT-Zig.txt").is_file(),
        "Zig launcher-toolchain MIT notice present",
    )
    start_script = source_root / "runtime" / "start.cmd"
    start_text = start_script.read_text(encoding="utf-8", errors="replace") if start_script.is_file() else ""
    add(
        "launcher:start-script",
        start_script.is_file()
        and "uv.sha256" in start_text
        and "Get-FileHash" in start_text
        and "uv-bootstrap.ps1" in start_text
        and (source_root / "runtime" / "uv-bootstrap.ps1").is_file()
        and "--frozen" in start_text
        and "--no-dev" in start_text
        and "--no-lock" not in start_text,
        "portable bootstrap verifies bundled uv.exe or a pinned first-run archive, then enforces the committed lock",
    )
    release_builder = app_root / "tools" / "build_release.py"
    release_builder_text = (
        release_builder.read_text(encoding="utf-8", errors="replace")
        if release_builder.is_file()
        else ""
    )
    add(
        "release:private-state-excluded",
        release_builder.is_file()
        and all(
            name in release_builder_text
            for name in (
                '"workspace"',
                '"development_reports"',
                "RUNTIME_SOURCE_NAMES",
                "GENERATED_RUNTIME_NAMES",
                "return runtime_name in RUNTIME_SOURCE_NAMES",
            )
        ),
        "release archives use a runtime source whitelist and exclude task data, authentication state, logs, caches, probes and virtual environments",
    )
    add("ui:assets", all((app_root / "src" / "scorescan" / "web" / item).is_file() for item in ("index.html", "app.js", "style.css")), "web UI assets present")

    qa = read_json(source_root / "RELEASE_QA.json", {})
    tests = int(qa.get("tests_passed", 0) or 0) if isinstance(qa, dict) else 0
    total_test_functions = int(qa.get("test_functions_total", 0) or 0) if isinstance(qa, dict) else 0
    skipped_tests = int(qa.get("tests_skipped", 0) or 0) if isinstance(qa, dict) else 0
    failed_tests = int(qa.get("tests_failed", 0) or 0) if isinstance(qa, dict) else 0
    non_server_reexecuted = int(qa.get("non_server_tests_reexecuted", 0) or 0) if isinstance(qa, dict) else 0
    server_test_functions = int(qa.get("server_test_functions", 0) or 0) if isinstance(qa, dict) else 0
    server_executed = int(qa.get("server_tests_executed", 0) or 0) if isinstance(qa, dict) else 0
    server_unexecuted = int(qa.get("server_tests_unexecuted", 0) or 0) if isinstance(qa, dict) else 0
    server_gap_reason = str(qa.get("server_test_gap_reason", "")) if isinstance(qa, dict) else ""
    torch_contract = (
        qa.get("detector_isolated_torch_contracts", {})
        if isinstance(qa, dict)
        else {}
    )
    torch_contract_path = (
        source_root / "training/detector_isolated_torch_contracts_v1.json"
    )
    torch_contract_report = read_json(torch_contract_path, {})
    torch_contract_input = (
        torch_contract_report.get("input", {})
        if isinstance(torch_contract_report, dict)
        else {}
    )
    detector_training_source = (
        source_root / "app/tools/train_deepscores_symbol_detector.py"
    )
    isolated_skip_covered = (
        skipped_tests == 0
        or (
            skipped_tests == 1
            and isinstance(torch_contract, dict)
            and isinstance(torch_contract_report, dict)
            and torch_contract_report.get("name")
            == "scorescan-detector-isolated-torch-contracts-v1"
            and torch_contract_report.get("passed") is True
            and torch_contract_report.get("checks", {})
            .get("microbatch_gradient", {})
            .get("passed")
            is True
            and torch_contract_report.get("checks", {})
            .get("legacy_sampler_recovery", {})
            .get("passed")
            is True
            and detector_training_source.is_file()
            and torch_contract_input.get("sha256")
            == sha256_file(detector_training_source)
            and torch_contract_path.is_file()
            and torch_contract.get("sha256")
            == sha256_file(torch_contract_path)
        )
    )
    add(
        "qa:tests",
        total_test_functions >= 300
        and server_test_functions >= 1
        and failed_tests == 0
        and tests + skipped_tests == total_test_functions
        and non_server_reexecuted == tests - server_executed
        and server_executed + server_unexecuted == server_test_functions
        and tests == non_server_reexecuted + server_executed
        and (server_unexecuted == 0 or bool(server_gap_reason.strip()))
        and isolated_skip_covered,
        (
            f"current tests passed/skipped/failed: "
            f"{tests}/{skipped_tests}/{failed_tests}; "
            f"collected nodes: {total_test_functions}; "
            f"non-server re-executed: {non_server_reexecuted}; server executed/unexecuted: "
            f"{server_executed}/{server_unexecuted}"
        ),
    )
    add("qa:compile", bool(qa.get("python_compile")) if isinstance(qa, dict) else False, "Python compile check")
    add("qa:javascript", bool(qa.get("javascript_syntax")) if isinstance(qa, dict) else False, "JavaScript syntax check")

    # These gates require external environments and deliberately keep this build an RC.
    add("external:windows", False, "complete Windows 10/11 non-admin physical-machine matrix pending", stable_blocker=True)
    add(
        "external:musescore",
        False,
        "supported MuseScore-version matrix and manual representative multi-page break acceptance pending",
        stable_blocker=True,
    )
    add(
        "external:soak",
        False,
        "24-hour multi-page cancellation, crash-recovery, disk-pressure and repeated-conversion soak acceptance pending",
        stable_blocker=True,
    )
    real_benchmark = read_json(source_root / "training" / "real_scan_end_to_end_report_v1.json", {})
    real_aggregate = real_benchmark.get("aggregate", {}) if isinstance(real_benchmark, dict) else {}
    real_bootstrap = real_benchmark.get("bootstrap_95", {}) if isinstance(real_benchmark, dict) else {}
    pitch_interval = real_bootstrap.get("pitch_accuracy_aligned", {}) if isinstance(real_bootstrap, dict) else {}
    rhythm_interval = real_bootstrap.get("rhythm_accuracy_aligned", {}) if isinstance(real_bootstrap, dict) else {}
    event_kind_interval = real_bootstrap.get("event_kind_accuracy_aligned", {}) if isinstance(real_bootstrap, dict) else {}
    chord_interval = real_bootstrap.get("chord_topology_accuracy_aligned", {}) if isinstance(real_bootstrap, dict) else {}
    tuplet_precision_interval = real_bootstrap.get("tuplet_event_precision", {}) if isinstance(real_bootstrap, dict) else {}
    tuplet_recall_interval = real_bootstrap.get("tuplet_event_recall", {}) if isinstance(real_bootstrap, dict) else {}
    tuplet_f1_interval = real_bootstrap.get("tuplet_event_f1", {}) if isinstance(real_bootstrap, dict) else {}
    tie_precision_interval = real_bootstrap.get("tie_endpoint_precision", {}) if isinstance(real_bootstrap, dict) else {}
    tie_recall_interval = real_bootstrap.get("tie_endpoint_recall", {}) if isinstance(real_bootstrap, dict) else {}
    tie_f1_interval = real_bootstrap.get("tie_endpoint_f1", {}) if isinstance(real_bootstrap, dict) else {}
    slur_precision_interval = real_bootstrap.get("slur_endpoint_precision", {}) if isinstance(real_bootstrap, dict) else {}
    slur_recall_interval = real_bootstrap.get("slur_endpoint_recall", {}) if isinstance(real_bootstrap, dict) else {}
    slur_f1_interval = real_bootstrap.get("slur_endpoint_f1", {}) if isinstance(real_bootstrap, dict) else {}
    articulation_precision_interval = real_bootstrap.get("articulation_marker_precision", {}) if isinstance(real_bootstrap, dict) else {}
    articulation_recall_interval = real_bootstrap.get("articulation_marker_recall", {}) if isinstance(real_bootstrap, dict) else {}
    articulation_f1_interval = real_bootstrap.get("articulation_marker_f1", {}) if isinstance(real_bootstrap, dict) else {}
    ornament_precision_interval = real_bootstrap.get("ornament_marker_precision", {}) if isinstance(real_bootstrap, dict) else {}
    ornament_recall_interval = real_bootstrap.get("ornament_marker_recall", {}) if isinstance(real_bootstrap, dict) else {}
    ornament_f1_interval = real_bootstrap.get("ornament_marker_f1", {}) if isinstance(real_bootstrap, dict) else {}
    grace_topology_interval = real_bootstrap.get("grace_topology_accuracy_aligned", {}) if isinstance(real_bootstrap, dict) else {}
    beam_topology_interval = real_bootstrap.get("beam_topology_accuracy_aligned", {}) if isinstance(real_bootstrap, dict) else {}
    beam_precision_interval = real_bootstrap.get("beam_marker_precision", {}) if isinstance(real_bootstrap, dict) else {}
    beam_recall_interval = real_bootstrap.get("beam_marker_recall", {}) if isinstance(real_bootstrap, dict) else {}
    beam_f1_interval = real_bootstrap.get("beam_marker_f1", {}) if isinstance(real_bootstrap, dict) else {}
    grace_precision_interval = real_bootstrap.get("grace_event_precision", {}) if isinstance(real_bootstrap, dict) else {}
    grace_recall_interval = real_bootstrap.get("grace_event_recall", {}) if isinstance(real_bootstrap, dict) else {}
    grace_f1_interval = real_bootstrap.get("grace_event_f1", {}) if isinstance(real_bootstrap, dict) else {}
    expression_precision_interval = real_bootstrap.get("expression_marker_precision", {}) if isinstance(real_bootstrap, dict) else {}
    expression_recall_interval = real_bootstrap.get("expression_marker_recall", {}) if isinstance(real_bootstrap, dict) else {}
    expression_f1_interval = real_bootstrap.get("expression_marker_f1", {}) if isinstance(real_bootstrap, dict) else {}
    barline_interval = real_bootstrap.get("barline_accuracy", {}) if isinstance(real_bootstrap, dict) else {}
    repeat_precision_interval = real_bootstrap.get("repeat_marker_precision", {}) if isinstance(real_bootstrap, dict) else {}
    repeat_recall_interval = real_bootstrap.get("repeat_marker_recall", {}) if isinstance(real_bootstrap, dict) else {}
    repeat_f1_interval = real_bootstrap.get("repeat_marker_f1", {}) if isinstance(real_bootstrap, dict) else {}
    cross_tie_precision_interval = real_bootstrap.get("cross_tie_precision", {}) if isinstance(real_bootstrap, dict) else {}
    cross_tie_recall_interval = real_bootstrap.get("cross_tie_recall", {}) if isinstance(real_bootstrap, dict) else {}
    cross_tie_f1_interval = real_bootstrap.get("cross_tie_f1", {}) if isinstance(real_bootstrap, dict) else {}
    presence_precision_interval = real_bootstrap.get("event_presence_precision", {}) if isinstance(real_bootstrap, dict) else {}
    presence_recall_interval = real_bootstrap.get("event_presence_recall", {}) if isinstance(real_bootstrap, dict) else {}
    presence_f1_interval = real_bootstrap.get("event_presence_f1", {}) if isinstance(real_bootstrap, dict) else {}
    time_interval = real_bootstrap.get("time_signature_accuracy", {}) if isinstance(real_bootstrap, dict) else {}
    exact_measure_interval = real_bootstrap.get("exact_measure_rate", {}) if isinstance(real_bootstrap, dict) else {}
    preservation_exact_interval = real_bootstrap.get("preservation_exact_measure_rate", {}) if isinstance(real_bootstrap, dict) else {}
    real_gate = real_benchmark.get("release_gate", {}) if isinstance(real_benchmark, dict) else {}
    real_scope = real_benchmark.get("production_scope_coverage", {}) if isinstance(real_benchmark, dict) else {}
    real_production_evidence = real_benchmark.get("production_evidence", {}) if isinstance(real_benchmark, dict) else {}
    real_evidence_files = (
        real_production_evidence.get("verified_files", [])
        if isinstance(real_production_evidence, dict)
        else []
    )
    add(
        "external:frozen-benchmark",
        isinstance(real_aggregate, dict)
        and int(real_benchmark.get("case_count", 0) or 0) >= 200
        and int(real_aggregate.get("reference_event_count", 0) or 0) >= 50_000
        and isinstance(real_scope, dict)
        and int(real_scope.get("page_count", 0) or 0) >= 2_000
        and int(
            real_scope.get("verified_unique_scan_page_count", 0) or 0
        )
        >= 2_000
        and int(real_scope.get("source_group_count", 0) or 0) >= 200
        and real_scope.get("out_of_contract_page_count") == 0
        and real_scope.get("duplicate_scan_page_count") == 0
        and real_scope.get("unverified_scan_page_identity_count") == 0
        and isinstance(real_production_evidence, dict)
        and real_production_evidence.get("passed") is True
        and real_production_evidence.get("boundary_contract_version")
        == "printed-western-instrumental-scan-boundary@4"
        and real_production_evidence.get("source_image_origin")
        == "physical_scan"
        and real_production_evidence.get("scan_page_shape_contract")
        == "ordinary-single-page-or-two-page-spread-aspect-ratio@1"
        and real_production_evidence.get(
            "maximum_scan_page_aspect_ratio"
        )
        == 3.0
        and isinstance(
            real_production_evidence.get("metrics"),
            dict,
        )
        and real_production_evidence["metrics"].get(
            "ordinary_scan_page_shape_audit_evidence"
        )
        == 1
        and isinstance(real_evidence_files, list)
        and len(real_evidence_files) >= 5
        and float(real_aggregate.get("pitch_accuracy_aligned", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("rhythm_accuracy_aligned", 0.0) or 0.0) >= 0.95
        and float(real_aggregate.get("event_kind_accuracy_aligned", 0.0) or 0.0) >= 0.985
        and float(real_aggregate.get("chord_topology_accuracy_aligned", 0.0) or 0.0) >= 0.985
        and float(real_aggregate.get("tuplet_topology_accuracy_aligned", 0.0) or 0.0) >= 0.99
        and float(real_aggregate.get("tuplet_event_precision", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("tuplet_event_recall", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("tuplet_event_f1", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("tie_topology_accuracy_aligned", 0.0) or 0.0) >= 0.99
        and float(real_aggregate.get("tie_endpoint_precision", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("tie_endpoint_recall", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("tie_endpoint_f1", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("slur_topology_accuracy", 0.0) or 0.0) >= 0.99
        and float(real_aggregate.get("slur_endpoint_precision", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("slur_endpoint_recall", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("slur_endpoint_f1", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("articulation_topology_accuracy_aligned", 0.0) or 0.0) >= 0.985
        and float(real_aggregate.get("articulation_marker_precision", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("articulation_marker_recall", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("articulation_marker_f1", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("ornament_topology_accuracy_aligned", 0.0) or 0.0) >= 0.985
        and float(real_aggregate.get("ornament_marker_precision", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("ornament_marker_recall", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("ornament_marker_f1", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("beam_topology_accuracy_aligned", 0.0) or 0.0) >= 0.985
        and float(real_aggregate.get("beam_marker_precision", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("beam_marker_recall", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("beam_marker_f1", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("grace_topology_accuracy_aligned", 0.0) or 0.0) >= 0.99
        and float(real_aggregate.get("grace_event_precision", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("grace_event_recall", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("grace_event_f1", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("expression_marker_precision", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("expression_marker_recall", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("expression_marker_f1", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("cross_tie_boundary_accuracy", 0.0) or 0.0) >= 0.99
        and float(real_aggregate.get("cross_tie_precision", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("cross_tie_recall", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("cross_tie_f1", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("event_presence_precision", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("event_presence_recall", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("event_presence_f1", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("deleted_event_rate", 1.0) or 1.0) <= 0.035
        and float(real_aggregate.get("inserted_event_rate", 1.0) or 1.0) <= 0.035
        and float(real_aggregate.get("time_signature_accuracy", 0.0) or 0.0) >= 0.99
        and float(real_aggregate.get("key_signature_accuracy", 0.0) or 0.0) >= 0.99
        and float(real_aggregate.get("clef_accuracy", 0.0) or 0.0) >= 0.99
        and float(real_aggregate.get("barline_accuracy", 0.0) or 0.0) >= 0.99
        and float(real_aggregate.get("repeat_marker_precision", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("repeat_marker_recall", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("repeat_marker_f1", 0.0) or 0.0) >= 0.97
        and float(real_aggregate.get("exact_measure_rate", 0.0) or 0.0) >= 0.75
        and float(real_aggregate.get("preservation_exact_measure_rate", 0.0) or 0.0) >= 0.75
        and float(real_aggregate.get("event_error_rate", 1.0) or 1.0) <= 0.10
        and isinstance(exact_measure_interval, dict)
        and float(exact_measure_interval.get("low_95", 0.0) or 0.0) >= 0.65
        and isinstance(preservation_exact_interval, dict)
        and float(preservation_exact_interval.get("low_95", 0.0) or 0.0) >= 0.65
        and isinstance(pitch_interval, dict)
        and float(pitch_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(rhythm_interval, dict)
        and float(rhythm_interval.get("low_95", 0.0) or 0.0) >= 0.92
        and isinstance(event_kind_interval, dict)
        and float(event_kind_interval.get("low_95", 0.0) or 0.0) >= 0.97
        and isinstance(chord_interval, dict)
        and float(chord_interval.get("low_95", 0.0) or 0.0) >= 0.97
        and isinstance(tuplet_precision_interval, dict)
        and float(tuplet_precision_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(tuplet_recall_interval, dict)
        and float(tuplet_recall_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(tuplet_f1_interval, dict)
        and float(tuplet_f1_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(tie_precision_interval, dict)
        and float(tie_precision_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(tie_recall_interval, dict)
        and float(tie_recall_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(tie_f1_interval, dict)
        and float(tie_f1_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(slur_precision_interval, dict)
        and float(slur_precision_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(slur_recall_interval, dict)
        and float(slur_recall_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(slur_f1_interval, dict)
        and float(slur_f1_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(articulation_precision_interval, dict)
        and float(articulation_precision_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(articulation_recall_interval, dict)
        and float(articulation_recall_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(articulation_f1_interval, dict)
        and float(articulation_f1_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(ornament_precision_interval, dict)
        and float(ornament_precision_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(ornament_recall_interval, dict)
        and float(ornament_recall_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(ornament_f1_interval, dict)
        and float(ornament_f1_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(beam_topology_interval, dict)
        and float(beam_topology_interval.get("low_95", 0.0) or 0.0) >= 0.97
        and isinstance(beam_precision_interval, dict)
        and float(beam_precision_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(beam_recall_interval, dict)
        and float(beam_recall_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(beam_f1_interval, dict)
        and float(beam_f1_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(grace_topology_interval, dict)
        and float(grace_topology_interval.get("low_95", 0.0) or 0.0) >= 0.97
        and isinstance(grace_precision_interval, dict)
        and float(grace_precision_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(grace_recall_interval, dict)
        and float(grace_recall_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(grace_f1_interval, dict)
        and float(grace_f1_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(expression_precision_interval, dict)
        and float(expression_precision_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(expression_recall_interval, dict)
        and float(expression_recall_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(expression_f1_interval, dict)
        and float(expression_f1_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(barline_interval, dict)
        and float(barline_interval.get("low_95", 0.0) or 0.0) >= 0.97
        and isinstance(repeat_precision_interval, dict)
        and float(repeat_precision_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(repeat_recall_interval, dict)
        and float(repeat_recall_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(repeat_f1_interval, dict)
        and float(repeat_f1_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(cross_tie_precision_interval, dict)
        and float(cross_tie_precision_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(cross_tie_recall_interval, dict)
        and float(cross_tie_recall_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(cross_tie_f1_interval, dict)
        and float(cross_tie_f1_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(presence_precision_interval, dict)
        and float(presence_precision_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(presence_recall_interval, dict)
        and float(presence_recall_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(presence_f1_interval, dict)
        and float(presence_f1_interval.get("low_95", 0.0) or 0.0) >= 0.95
        and isinstance(time_interval, dict)
        and float(time_interval.get("low_95", 0.0) or 0.0) >= 0.97
        and isinstance(real_gate, dict)
        and real_gate.get("profile") == "production-v2"
        and bool(real_gate.get("passed")),
        "at least 200 independent works and 2,000 unique physical scan pages, backed by hashed provenance/license/annotation/isolation/freeze audits, must pass the production-v2 near-correction-free gate for event presence, pitch, rhythm, chord, tuplet, tie, slur, articulation, ornament, beam, complete-measure preservation, text/directions, score attributes and confidence intervals",
        stable_blocker=True,
    )
    uv_lock = app_root / "uv.lock"
    lock_text = uv_lock.read_text(encoding="utf-8", errors="replace") if uv_lock.is_file() else ""
    add(
        "dependency:transitive-lock",
        uv_lock.is_file()
        and 'requires-python = ">=3.12, <3.14"' in lock_text
        and 'source = { registry = "https://pypi.org/simple" }' in lock_text
        and "files.pythonhosted.org" in lock_text
        and "applied-caas" not in lock_text
        and "reader:" not in lock_text,
        "fully resolved public-PyPI uv.lock is committed and portable bootstrap runs frozen",
        stable_blocker=True,
    )
    add(
        "dependency:single-opencv",
        '\nname = "opencv-python-headless"\n' in lock_text
        and '\nname = "opencv-python"\n' not in lock_text,
        "locked runtime contains only the headless OpenCV distribution",
    )

    internal_failures = [item for item in items if not item["ok"] and not item["stable_blocker"]]
    stable_blockers = [item for item in items if not item["ok"] and item["stable_blocker"]]
    return {
        "version": APP_VERSION,
        "workflow_version": WORKFLOW_VERSION,
        "policy_version": DEFAULT_POLICY.version,
        "rc_ready": not internal_failures,
        "public_test_ready": not internal_failures,
        "stable_ready": not internal_failures and not stable_blockers,
        "internal_failures": len(internal_failures),
        "stable_blockers": len(stable_blockers),
        "items": items,
        "scope": "public RC release-engineering readiness; not end-to-end recognition accuracy",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.source_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "items"}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["rc_ready"] else 1)


if __name__ == "__main__":
    main()
