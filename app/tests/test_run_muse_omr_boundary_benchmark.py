from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from app.tools.run_muse_omr_boundary_benchmark import (
    DEVELOPMENT_BENCHMARK_ROLE,
    RELEASE_BENCHMARK_ROLE,
    _accelerator_execution_evidence,
    _evaluation_case,
    _implementation_fingerprint,
    _materialize_production_evidence,
    _record_is_reusable,
    _unique_work_cases,
    _validate_manifest_role,
    _validate_release_partition_isolation,
)
from app.tools.evaluate_release_dataset import (
    PRODUCTION_EVIDENCE_FILE_ROLES,
)
from app.tools.muse_omr_contract import SCAN_DEGRADED_IMAGE_ORIGIN
from scorescan.product_scope import PRODUCTION_BOUNDARY_CONTRACT_VERSION
from scorescan.util import sha256_file


def test_direct_script_entry_point_loads_project_modules() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "run_muse_omr_boundary_benchmark.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0
    assert "--allow-diagnostic-manifest" in completed.stdout


def test_unique_work_cases_deduplicates_variants_without_reordering() -> None:
    cases = [
        {"id": "muse-1", "work_fingerprint": "a" * 64},
        {"id": "muse-2", "work_fingerprint": "a" * 64},
        {"id": "muse-3", "work_fingerprint": "b" * 64},
    ]

    assert [case["id"] for case in _unique_work_cases(cases)] == [
        "muse-1",
        "muse-3",
    ]


def test_unique_work_cases_fails_closed_without_provenance() -> None:
    with pytest.raises(ValueError, match="valid work fingerprint"):
        _unique_work_cases([{"id": "legacy"}])


def test_release_manifest_role_fails_closed_but_allows_explicit_diagnostic() -> None:
    release = {
        "boundary_contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
        "role": RELEASE_BENCHMARK_ROLE,
        "source_image_origin": "physical_scan",
        "production_evidence_eligible": True,
        "production_evidence": {"source_image_origin": "physical_scan"},
    }
    scan_degraded = {
        "boundary_contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
        "role": DEVELOPMENT_BENCHMARK_ROLE,
        "source_image_origin": SCAN_DEGRADED_IMAGE_ORIGIN,
        "production_evidence_eligible": False,
    }
    diagnostic = {
        "boundary_contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
        "role": (
            "external_real_scan_diagnostic_only_"
            "license_not_release_authorized"
        ),
        "source_image_origin": "physical_scan",
        "production_evidence_eligible": False,
    }
    assert (
        _validate_manifest_role(
            release,
            allow_diagnostic_manifest=False,
        )
        == release["role"]
    )
    with pytest.raises(ValueError, match="not authorized"):
        _validate_manifest_role(
            scan_degraded,
            allow_diagnostic_manifest=False,
        )
    assert (
        _validate_manifest_role(
            scan_degraded,
            allow_diagnostic_manifest=True,
        )
        == scan_degraded["role"]
    )
    with pytest.raises(ValueError, match="invalid origin"):
        _validate_manifest_role(
            {
                key: value
                for key, value in scan_degraded.items()
                if key != "source_image_origin"
            },
            allow_diagnostic_manifest=True,
        )
    with pytest.raises(ValueError, match="not authorized"):
        _validate_manifest_role(
            diagnostic,
            allow_diagnostic_manifest=False,
        )
    assert (
        _validate_manifest_role(
            diagnostic,
            allow_diagnostic_manifest=True,
        )
        == diagnostic["role"]
    )
    with pytest.raises(ValueError, match="invalid origin"):
        _validate_manifest_role(
            {
                **diagnostic,
                "source_image_origin": SCAN_DEGRADED_IMAGE_ORIGIN,
            },
            allow_diagnostic_manifest=True,
        )
    with pytest.raises(ValueError, match="contract is stale"):
        _validate_manifest_role(
            {
                "boundary_contract_version": "obsolete",
                "role": release["role"],
            },
            allow_diagnostic_manifest=False,
        )


def test_release_evidence_is_hash_verified_and_materialized_locally(
    tmp_path: Path,
) -> None:
    evidence_files = []
    for role in sorted(PRODUCTION_EVIDENCE_FILE_ROLES):
        path = tmp_path / "source-audits" / f"{role}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(f'{{"role":"{role}"}}', encoding="utf-8")
        evidence_files.append(
            {
                "role": role,
                "path": path.relative_to(tmp_path).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    source = {
        "production_evidence": {
            "boundary_contract_version": (
                PRODUCTION_BOUNDARY_CONTRACT_VERSION
            ),
            "source_image_origin": "physical_scan",
            "page_identity_audited": True,
            "evaluation_use_authorized": True,
            "complete_page_level_semantics": True,
            "instrumental_lyrics_excluded_or_isolated": True,
            "independent_double_annotation_adjudicated": True,
            "work_disjoint_from_training_and_tuning": True,
            "frozen_before_candidate_evaluation": True,
            "submitted_orientation_preserved": True,
            "scan_page_shape_contract": (
                "ordinary-single-page-or-two-page-spread-aspect-ratio@1"
            ),
            "maximum_scan_page_aspect_ratio": 3.0,
            "ordinary_scan_page_shape_audited": True,
            "evidence_files": evidence_files,
        }
    }

    materialized = _materialize_production_evidence(
        source,
        boundary_manifest=tmp_path / "boundary_manifest.json",
        output_dir=tmp_path / "output",
    )

    assert len(materialized["evidence_files"]) == len(
        PRODUCTION_EVIDENCE_FILE_ROLES
    )
    for item in materialized["evidence_files"]:
        path = tmp_path / "output" / item["path"]
        assert path.is_file()
        assert sha256_file(path) == item["sha256"]


def test_release_evidence_materialization_rejects_tampering(
    tmp_path: Path,
) -> None:
    evidence_files = []
    for role in sorted(PRODUCTION_EVIDENCE_FILE_ROLES):
        path = tmp_path / "source-audits" / f"{role}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(f'{{"role":"{role}"}}', encoding="utf-8")
        evidence_files.append(
            {
                "role": role,
                "path": path.relative_to(tmp_path).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    source = {
        "production_evidence": {
            "boundary_contract_version": (
                PRODUCTION_BOUNDARY_CONTRACT_VERSION
            ),
            "source_image_origin": "physical_scan",
            "page_identity_audited": True,
            "evaluation_use_authorized": True,
            "complete_page_level_semantics": True,
            "instrumental_lyrics_excluded_or_isolated": True,
            "independent_double_annotation_adjudicated": True,
            "work_disjoint_from_training_and_tuning": True,
            "frozen_before_candidate_evaluation": True,
            "submitted_orientation_preserved": True,
            "scan_page_shape_contract": (
                "ordinary-single-page-or-two-page-spread-aspect-ratio@1"
            ),
            "maximum_scan_page_aspect_ratio": 3.0,
            "ordinary_scan_page_shape_audited": True,
            "evidence_files": evidence_files,
        }
    }
    first_path = tmp_path / evidence_files[0]["path"]
    first_path.write_text('{"tampered":true}', encoding="utf-8")

    with pytest.raises(ValueError, match="production evidence is invalid"):
        _materialize_production_evidence(
            source,
            boundary_manifest=tmp_path / "boundary_manifest.json",
            output_dir=tmp_path / "output",
        )
    assert not (tmp_path / "output").exists()


def test_release_partition_evidence_recomputes_work_overlap(
    tmp_path: Path,
) -> None:
    training = tmp_path / "selection.json"
    plan = tmp_path / "plan.json"
    training.write_text(
        (
            '{"role":"external_scan_degraded_training_only",'
            '"selected_work_fingerprints":["'
            + ("b" * 64)
            + '"]}'
        ),
        encoding="utf-8",
    )
    plan.write_text(
        '{"model_outputs_used_for_selection":false}',
        encoding="utf-8",
    )
    source = {
        "selection_used_model_outputs": False,
        "training_evaluation_work_overlap": [],
        "training_selection": str(training),
        "training_selection_sha256": sha256_file(training),
        "partition_plan": str(plan),
        "partition_plan_sha256": sha256_file(plan),
    }
    cases = [{"id": "eval", "work_fingerprint": "a" * 64}]

    evidence = _validate_release_partition_isolation(
        source,
        cases,
        boundary_manifest=tmp_path / "manifest.json",
        manifest_role=(
            RELEASE_BENCHMARK_ROLE
        ),
    )
    assert evidence is not None
    assert evidence["training_evaluation_work_overlap"] == []

    cases[0]["work_fingerprint"] = "b" * 64
    with pytest.raises(ValueError, match="overlaps training"):
        _validate_release_partition_isolation(
            source,
            cases,
            boundary_manifest=tmp_path / "manifest.json",
            manifest_role=(
                RELEASE_BENCHMARK_ROLE
            ),
        )


def test_release_partition_evidence_rejects_accuracy_selected_split(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="model-independent"):
        _validate_release_partition_isolation(
            {
                "selection_used_model_outputs": True,
                "training_evaluation_work_overlap": [],
            },
            [{"id": "eval", "work_fingerprint": "a" * 64}],
            boundary_manifest=tmp_path / "manifest.json",
            manifest_role=(
                RELEASE_BENCHMARK_ROLE
            ),
        )


def test_evaluation_case_exposes_shape_and_complexity_as_strata(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.musicxml"
    reference.write_text("<score-partwise/>", encoding="utf-8")
    candidate = tmp_path / "candidate.musicxml"
    candidate.write_text("<score-partwise/>", encoding="utf-8")
    case = {
        "id": "muse-1",
        "work_fingerprint": "a" * 64,
        "input_pdf_pages": 7,
        "input_pdf_sha256": "b" * 64,
        "_resolved_reference": str(reference),
        "boundary": {
            "score_shape": "keyboard_plus_single_staff_ensemble",
            "counts": {"notes": 100, "slurs": 40},
        },
    }

    result = _evaluation_case(
        case,
        candidate,
        source_image_origin="physical_scan",
    )

    assert result["strata"] == {
        "score_shape": "keyboard_plus_single_staff_ensemble",
        "score_configuration": "piano_plus_monophonic_ensemble",
        "complexity": "high",
    }
    assert result["submitted_scan_page_count"] == 7
    assert result["submitted_scan_page_ids"] == [
        f"{'b' * 64}/page-{page_number:06d}"
        for page_number in range(1, 8)
    ]
    assert result["source"] == "physical_scan"


def test_evaluation_case_rejects_unknown_shape_and_page_count(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.musicxml"
    reference.write_text("<score-partwise/>", encoding="utf-8")
    candidate = tmp_path / "candidate.musicxml"
    candidate.write_text("<score-partwise/>", encoding="utf-8")
    base = {
        "id": "muse-invalid",
        "work_fingerprint": "a" * 64,
        "input_pdf_pages": 1,
        "input_pdf_sha256": "b" * 64,
        "_resolved_reference": str(reference),
        "boundary": {"score_shape": "unknown", "counts": {}},
    }
    with pytest.raises(ValueError, match="unsupported score shape"):
        _evaluation_case(
            base,
            candidate,
            source_image_origin=SCAN_DEGRADED_IMAGE_ORIGIN,
        )
    base["boundary"]["score_shape"] = "keyboard"
    base["input_pdf_pages"] = 0
    with pytest.raises(ValueError, match="positive input PDF page count"):
        _evaluation_case(
            base,
            candidate,
            source_image_origin=SCAN_DEGRADED_IMAGE_ORIGIN,
        )


def test_pipeline_implementation_fingerprint_is_populated_and_stable() -> None:
    first = _implementation_fingerprint()
    second = _implementation_fingerprint()

    assert first == second
    assert first["root"] == "app/src/scorescan"
    assert int(first["file_count"]) > 20
    assert len(str(first["sha256"])) == 64


def test_case_reuse_is_bound_to_pipeline_input_and_audit_report(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.musicxml"
    conversion_report = tmp_path / "conversion_report.json"
    candidate.write_text("<score-partwise/>", encoding="utf-8")
    conversion_report.write_text('{"verified":true}', encoding="utf-8")
    fingerprint = "c" * 64
    input_hash = "d" * 64
    record = {
        "format": 2,
        "status": "completed",
        "candidate_sha256": sha256_file(candidate),
        "conversion_report_sha256": sha256_file(conversion_report),
        "input_pdf_sha256": input_hash,
        "pipeline_evidence_fingerprint": fingerprint,
        "boundary_contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
    }

    assert _record_is_reusable(
        record,
        candidate_path=candidate,
        conversion_report=conversion_report,
        input_pdf_sha256=input_hash,
        pipeline_evidence_fingerprint=fingerprint,
    )
    assert not _record_is_reusable(
        record,
        candidate_path=candidate,
        conversion_report=conversion_report,
        input_pdf_sha256=input_hash,
        pipeline_evidence_fingerprint="e" * 64,
    )
    conversion_report.write_text('{"verified":false}', encoding="utf-8")
    assert not _record_is_reusable(
        record,
        candidate_path=candidate,
        conversion_report=conversion_report,
        input_pdf_sha256=input_hash,
        pipeline_evidence_fingerprint=fingerprint,
    )


def test_accelerator_gate_requires_verified_ocr_and_semantic_pages() -> None:
    report = {
        "accelerator": {"selected": "cpu"},
        "ocr_runtime": {
            "runtime": "cpu",
            "cpu_pages": 3,
            "verified_pages": 3,
            "unverified_pages": 0,
        },
        "semantic_detector_runtime": {
            "authorized_at_job_start": True,
            "model_version": "semantic-v1",
            "enabled_pages": 3,
            "runtime": "cpu",
            "cpu_pages": 3,
            "unverified_enabled_pages": 0,
        },
    }
    evidence = _accelerator_execution_evidence(
        report,
        page_count=3,
        semantic_model_version="semantic-v1",
        required_accelerator="cpu",
    )
    assert evidence["selected"] == "cpu"
    assert evidence["semantic_cpu_pages"] == 3

    report["semantic_detector_runtime"]["runtime"] = "unknown"
    with pytest.raises(ValueError, match="unverified"):
        _accelerator_execution_evidence(
            report,
            page_count=3,
            semantic_model_version="semantic-v1",
            required_accelerator="cpu",
        )


def test_accelerator_gate_rejects_silent_cpu_execution_for_cuda() -> None:
    report = {
        "accelerator": {"selected": "cpu"},
        "ocr_runtime": {
            "gpu_pages": 0,
            "cpu_pages": 1,
            "fallback_pages": 0,
            "unverified_pages": 0,
        },
        "semantic_detector_runtime": {
            "authorized_at_job_start": True,
            "model_version": "semantic-v1",
            "enabled_pages": 1,
            "gpu_pages": 0,
            "cpu_pages": 1,
            "fallback_pages": 0,
            "unverified_enabled_pages": 0,
        },
    }
    with pytest.raises(ValueError, match="must be cpu"):
        _accelerator_execution_evidence(
            report,
            page_count=1,
            semantic_model_version="semantic-v1",
            required_accelerator="cuda",
        )


def test_diagnostic_baseline_records_missing_semantic_detector() -> None:
    report = {
        "accelerator": {"selected": "cpu"},
        "ocr_runtime": {
            "runtime": "cpu",
            "cpu_pages": 2,
            "verified_pages": 2,
            "unverified_pages": 0,
        },
        "semantic_detector_runtime": {
            "authorized_at_job_start": False,
            "model_version": None,
            "enabled_pages": 0,
            "runtime": "cpu",
            "cpu_pages": 0,
            "unverified_enabled_pages": 0,
            "status": "asset_absent",
        },
    }
    evidence = _accelerator_execution_evidence(
        report,
        page_count=2,
        semantic_model_version="asset_absent",
        required_accelerator=None,
        semantic_detector_required=False,
    )
    assert evidence["selected"] == "cpu"
    assert evidence["semantic_detector_required"] is False
    assert evidence["semantic_authorized_at_job_start"] is False
    assert evidence["semantic_status"] == "asset_absent"

    with pytest.raises(ValueError, match="requires the semantic detector"):
        _accelerator_execution_evidence(
            report,
            page_count=2,
            semantic_model_version="asset_absent",
            required_accelerator="cpu",
            semantic_detector_required=False,
        )
