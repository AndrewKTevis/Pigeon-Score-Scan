from __future__ import annotations

"""Run an external boundary manifest through the complete ScoreScan pipeline.

Muse OMR scan-degraded renders are development evidence only.  A release run
must use the separate physical-scan role and its complete hashed production
evidence bundle.
"""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT / "src"))

from scorescan.config import APP_VERSION, WORKFLOW_VERSION, Settings  # noqa: E402
from scorescan.engine_cache import homr_version  # noqa: E402
from scorescan.jobs import JobManager  # noqa: E402
from scorescan.model_registry import audit_model_manifest  # noqa: E402
from scorescan.product_scope import (  # noqa: E402
    PRODUCTION_BOUNDARY_CONTRACT_VERSION,
)
from scorescan.semantic_detector import load_semantic_detector_assets  # noqa: E402
from scorescan.util import atomic_write_bytes, atomic_write_json, read_json, sha256_file, utc_now_iso  # noqa: E402

from app.tools.evaluate_release_dataset import (  # noqa: E402
    PRODUCTION_RELEASE_GATES_V2,
    PRODUCTION_SCORE_CONFIGURATIONS,
    SCORE_CONFIGURATION_BY_SHAPE,
    evaluate_manifest,
    validate_production_evidence,
)
from app.tools.prepare_muse_omr_benchmark import (  # noqa: E402
    production_page_coverage as _production_page_coverage,
    unique_work_cases as _unique_work_cases,
)
from app.tools.muse_omr_contract import (  # noqa: E402
    BENCHMARK_SELECTION_ROLE,
    PHYSICAL_SCAN_RELEASE_BENCHMARK_ROLE,
    SCAN_DEGRADED_IMAGE_ORIGIN,
)


TERMINAL_STATES = {"completed", "failed", "cancelled"}
RELEASE_BENCHMARK_ROLE = PHYSICAL_SCAN_RELEASE_BENCHMARK_ROLE
DEVELOPMENT_BENCHMARK_ROLE = BENCHMARK_SELECTION_ROLE
PIPELINE_EVIDENCE_FORMAT = 1


def _stable_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _optional_file_sha256(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _materialize_production_evidence(
    source: dict[str, object],
    *,
    boundary_manifest: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Copy verified release audits beside the generated evaluation manifest."""

    audit = validate_production_evidence(source, boundary_manifest)
    if audit.get("passed") is not True:
        errors = audit.get("errors")
        detail = ", ".join(str(value) for value in (errors or [])[:5])
        raise ValueError(
            "release benchmark production evidence is invalid: " + detail
        )
    raw_evidence = source.get("production_evidence")
    if not isinstance(raw_evidence, dict):
        raise ValueError("release benchmark production evidence is missing")
    evidence = json.loads(json.dumps(raw_evidence, ensure_ascii=False))
    evidence_root = output_dir / "release_evidence"
    materialized_files: list[dict[str, str]] = []
    for item in audit["verified_files"]:
        role = str(item["role"])
        relative_source = Path(str(item["path"]))
        source_path = (boundary_manifest.parent / relative_source).resolve()
        suffix = source_path.suffix if source_path.suffix else ".bin"
        destination = evidence_root / f"{role}{suffix}"
        content = source_path.read_bytes()
        if destination.is_file() and sha256_file(destination) != item["sha256"]:
            raise ValueError(
                f"existing materialized release evidence is stale: {role}"
            )
        if not destination.is_file():
            atomic_write_bytes(destination, content)
        if sha256_file(destination) != item["sha256"]:
            raise ValueError(
                f"materialized release evidence hash mismatch: {role}"
            )
        materialized_files.append(
            {
                "role": role,
                "path": destination.relative_to(output_dir).as_posix(),
                "sha256": str(item["sha256"]),
            }
        )
    evidence["evidence_files"] = materialized_files
    return evidence


def _implementation_fingerprint() -> dict[str, object]:
    """Bind benchmark reuse to the exact ScoreScan implementation under test."""

    source_root = ROOT / "src" / "scorescan"
    files = sorted(
        (
            path
            for path in source_root.rglob("*.py")
            if path.is_file() and not path.is_symlink()
        ),
        key=lambda path: path.relative_to(source_root).as_posix(),
    )
    rows = [
        {
            "path": path.relative_to(source_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    return {
        "root": "app/src/scorescan",
        "file_count": len(rows),
        "sha256": _stable_digest(rows),
    }


def _pipeline_evidence(
    *,
    product_root: Path,
    resources_dir: Path,
    semantic_detector_required: bool = True,
) -> dict[str, object]:
    """Return immutable evidence for every input that can change a candidate.

    The semantic detector is deliberately allowed to live in an isolated complete
    resources directory.  This lets a release candidate be tested end-to-end
    before it is copied into the canonical application resources.
    """

    resources_dir = resources_dir.resolve()
    model_audit = audit_model_manifest(resources_dir)
    if not model_audit.verified:
        raise ValueError(
            "benchmark resources failed bundled-model integrity audit: "
            + ", ".join(model_audit.errors[:5])
        )
    try:
        semantic_assets = load_semantic_detector_assets(resources_dir)
    except FileNotFoundError:
        if semantic_detector_required:
            raise ValueError(
                "benchmark requires an authorized semantic detector asset"
            ) from None
        semantic_assets = None
    dependency_inputs = (
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    )
    evidence: dict[str, object] = {
        "format": PIPELINE_EVIDENCE_FORMAT,
        "application_version": APP_VERSION,
        "workflow_version": WORKFLOW_VERSION,
        "boundary_contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
        "homr_version": homr_version(),
        "implementation": _implementation_fingerprint(),
        "resources": {
            "directory": str(resources_dir),
            "model_manifest_sha256": sha256_file(
                resources_dir / "model_manifest.json"
            ),
            "verified_model_count": model_audit.verified_count,
            "semantic_detector_manifest_sha256": (
                semantic_assets.manifest_sha256
                if semantic_assets is not None
                else None
            ),
            "semantic_detector_model_version": (
                semantic_assets.model_version
                if semantic_assets is not None
                else None
            ),
            "semantic_detector_model_sha256": (
                sha256_file(semantic_assets.model_path)
                if semantic_assets is not None
                else None
            ),
            "semantic_detector_status": (
                "verified" if semantic_assets is not None else "asset_absent"
            ),
            "semantic_detector_required": semantic_detector_required,
        },
        "dependencies": {
            str(path.relative_to(product_root))
            if path.is_relative_to(product_root)
            else str(path.relative_to(ROOT)): _optional_file_sha256(path)
            for path in dependency_inputs
        },
        "accelerator_request": {
            "runtime": "cpu",
            "runtime_contract_sha256": _optional_file_sha256(ROOT / "uv.lock"),
        },
    }
    fingerprint_payload = json.loads(
        json.dumps(evidence, ensure_ascii=False)
    )
    fingerprint_payload["resources"].pop("directory", None)
    evidence["fingerprint"] = _stable_digest(fingerprint_payload)
    return evidence


def _record_is_reusable(
    existing: object,
    *,
    candidate_path: Path,
    conversion_report: Path,
    input_pdf_sha256: object,
    pipeline_evidence_fingerprint: str,
) -> bool:
    if (
        not candidate_path.is_file()
        or not conversion_report.is_file()
        or not isinstance(existing, dict)
        or existing.get("format") != 2
        or existing.get("status") != "completed"
        or existing.get("input_pdf_sha256") != input_pdf_sha256
        or existing.get("pipeline_evidence_fingerprint")
        != pipeline_evidence_fingerprint
        or existing.get("boundary_contract_version")
        != PRODUCTION_BOUNDARY_CONTRACT_VERSION
    ):
        return False
    try:
        return (
            existing.get("candidate_sha256") == sha256_file(candidate_path)
            and existing.get("conversion_report_sha256")
            == sha256_file(conversion_report)
        )
    except OSError:
        return False


def _accelerator_execution_evidence(
    report: object,
    *,
    page_count: int,
    semantic_model_version: str,
    required_accelerator: str | None,
    semantic_detector_required: bool = True,
) -> dict[str, object]:
    if page_count <= 0:
        raise ValueError("benchmark execution has no pages")
    if required_accelerator not in {None, "cpu"}:
        raise ValueError("required accelerator must be cpu or omitted")
    if required_accelerator is not None and not semantic_detector_required:
        raise ValueError(
            "accelerator verification requires the semantic detector"
        )
    if not isinstance(report, dict):
        raise ValueError("conversion report is missing")
    accelerator = report.get("accelerator")
    ocr = report.get("ocr_runtime")
    semantic = report.get("semantic_detector_runtime")
    if (
        not isinstance(accelerator, dict)
        or not isinstance(ocr, dict)
        or not isinstance(semantic, dict)
    ):
        raise ValueError("conversion accelerator evidence is incomplete")
    if (
        int(ocr.get("unverified_pages", -1)) != 0
        or int(ocr.get("verified_pages", -1)) != page_count
        or ocr.get("runtime") != "cpu"
    ):
        raise ValueError("OCR execution was unverified or fell back")
    if semantic_detector_required and (
        semantic.get("authorized_at_job_start") is not True
        or semantic.get("model_version") != semantic_model_version
        or int(semantic.get("enabled_pages", -1)) != page_count
        or int(semantic.get("unverified_enabled_pages", -1)) != 0
        or semantic.get("runtime") != "cpu"
    ):
        raise ValueError(
            "semantic execution was absent, unverified, or fell back"
        )
    selected = str(accelerator.get("selected", ""))
    if required_accelerator is not None:
        if (
            selected != required_accelerator
            or int(ocr.get("cpu_pages", -1)) != page_count
            or int(semantic.get("cpu_pages", -1)) != page_count
        ):
            raise ValueError(
                f"{required_accelerator} execution was required but not "
                "verified on every page"
            )
    return {
        "required": required_accelerator,
        "selected": selected,
        "page_count": page_count,
        "ocr_cpu_pages": int(ocr.get("cpu_pages", 0)),
        "semantic_cpu_pages": int(semantic.get("cpu_pages", 0)),
        "semantic_model_version": semantic_model_version,
        "semantic_detector_required": semantic_detector_required,
        "semantic_authorized_at_job_start": bool(
            semantic.get("authorized_at_job_start")
        ),
        "semantic_status": str(
            semantic.get("status")
            or semantic.get("reason")
            or "unknown"
        ),
    }


def _validate_manifest_role(
    source: dict[str, object],
    *,
    allow_diagnostic_manifest: bool,
) -> str:
    if (
        source.get("boundary_contract_version")
        != PRODUCTION_BOUNDARY_CONTRACT_VERSION
    ):
        raise ValueError(
            "boundary manifest contract is stale or missing"
        )
    role = str(source.get("role", ""))
    if role == RELEASE_BENCHMARK_ROLE:
        production_evidence = source.get("production_evidence")
        if (
            source.get("source_image_origin") != "physical_scan"
            or source.get("production_evidence_eligible") is not True
            or not isinstance(production_evidence, dict)
            or production_evidence.get("source_image_origin")
            != "physical_scan"
        ):
            raise ValueError(
                "release benchmark has no physical-scan production evidence"
            )
        return role
    if allow_diagnostic_manifest:
        if role == DEVELOPMENT_BENCHMARK_ROLE:
            origin = source.get("source_image_origin")
            if (
                origin != SCAN_DEGRADED_IMAGE_ORIGIN
                or source.get("production_evidence_eligible") is True
            ):
                raise ValueError(
                    "scan-degraded development manifest has invalid origin"
                )
            return role
        if role.startswith("external_real_scan_diagnostic_only_"):
            if (
                source.get("source_image_origin") != "physical_scan"
                or source.get("production_evidence_eligible") is True
            ):
                raise ValueError(
                    "real-scan diagnostic manifest has invalid origin"
                )
            return role
    raise ValueError(
        "boundary manifest role is not authorized for release evaluation"
    )


def _validate_release_partition_isolation(
    source: dict[str, object],
    accepted_cases: list[dict[str, object]],
    *,
    boundary_manifest: Path,
    manifest_role: str,
) -> dict[str, object] | None:
    if manifest_role != RELEASE_BENCHMARK_ROLE:
        return None
    if (
        source.get("selection_used_model_outputs") is not False
        or source.get("training_evaluation_work_overlap") != []
    ):
        raise ValueError(
            "release boundary manifest has no model-independent split evidence"
        )
    raw_training = str(source.get("training_selection", "")).strip()
    raw_plan = str(source.get("partition_plan", "")).strip()
    training_path = Path(raw_training)
    plan_path = Path(raw_plan)
    if not training_path.is_absolute():
        training_path = boundary_manifest.parent / training_path
    if not plan_path.is_absolute():
        plan_path = boundary_manifest.parent / plan_path
    training_path = training_path.resolve(strict=True)
    plan_path = plan_path.resolve(strict=True)
    if (
        source.get("training_selection_sha256")
        != sha256_file(training_path)
        or source.get("partition_plan_sha256") != sha256_file(plan_path)
    ):
        raise ValueError("release partition evidence hash mismatch")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    training_works = (
        training.get("selected_work_fingerprints")
        if isinstance(training, dict)
        else None
    )
    if (
        not isinstance(training, dict)
        or training.get("role")
        != "external_scan_degraded_training_only"
        or not isinstance(training_works, list)
        or not isinstance(plan, dict)
        or plan.get("model_outputs_used_for_selection") is not False
    ):
        raise ValueError("release partition evidence is invalid")
    evaluation_works = {
        str(case.get("work_fingerprint", ""))
        for case in _unique_work_cases(accepted_cases)
    }
    actual_overlap = sorted(
        evaluation_works & {str(value) for value in training_works}
    )
    if actual_overlap:
        raise ValueError(
            "release evaluation overlaps training by independent work"
        )
    return {
        "training_selection": str(training_path),
        "training_selection_sha256": sha256_file(training_path),
        "partition_plan": str(plan_path),
        "partition_plan_sha256": sha256_file(plan_path),
        "training_evaluation_work_overlap": [],
        "selection_used_model_outputs": False,
    }


def _wait(manager: JobManager, job_id: str, timeout_seconds: int):
    deadline = time.monotonic() + max(1, timeout_seconds)
    while time.monotonic() < deadline:
        state = manager.get(job_id)
        if state is None:
            raise RuntimeError("benchmark job disappeared")
        if state.status in TERMINAL_STATES:
            return state
        time.sleep(0.25)
    manager.cancel(job_id)
    raise TimeoutError(f"benchmark case exceeded {timeout_seconds} seconds")


def _evaluation_case(
    case: dict[str, object],
    candidate: Path,
    *,
    source_image_origin: str,
) -> dict[str, object]:
    boundary = case.get("boundary")
    boundary = boundary if isinstance(boundary, dict) else {}
    counts = boundary.get("counts")
    counts = counts if isinstance(counts, dict) else {}
    mark_count = sum(
        int(counts.get(key, 0) or 0)
        for key in ("slurs", "ties", "articulations", "ornaments", "directions", "wedges")
    )
    note_count = max(1, int(counts.get("notes", 0) or 0))
    complexity = (
        "high"
        if mark_count / note_count >= 0.35
        else "medium"
        if mark_count / note_count >= 0.12
        else "low"
    )
    score_shape = str(boundary.get("score_shape", "unknown"))
    try:
        score_configuration = SCORE_CONFIGURATION_BY_SHAPE[score_shape]
    except KeyError as exc:
        raise ValueError(
            f"case {case.get('id')} has unsupported score shape: {score_shape}"
        ) from exc
    submitted_scan_page_count = case.get("input_pdf_pages")
    if (
        isinstance(submitted_scan_page_count, bool)
        or not isinstance(submitted_scan_page_count, int)
        or submitted_scan_page_count <= 0
    ):
        raise ValueError(
            f"case {case.get('id')} has no positive input PDF page count"
        )
    raw_reference = case.get("_resolved_reference") or case.get("reference")
    if not raw_reference:
        raise ValueError(f"case {case.get('id')} has no reference")
    input_pdf_sha256 = str(case.get("input_pdf_sha256", "")).strip().lower()
    if (
        len(input_pdf_sha256) != 64
        or any(character not in "0123456789abcdef" for character in input_pdf_sha256)
    ):
        raise ValueError(
            f"case {case.get('id')} has no valid input PDF SHA-256"
        )
    return {
        "id": str(case["id"]),
        "split": "test",
        "source_group": f"muse-omr-work/{case['work_fingerprint']}",
        "submitted_scan_page_count": submitted_scan_page_count,
        "submitted_scan_page_ids": [
            f"{input_pdf_sha256}/page-{page_number:06d}"
            for page_number in range(1, submitted_scan_page_count + 1)
        ],
        "reference": str(Path(str(raw_reference)).resolve()),
        "candidate": str(candidate.resolve()),
        "score_shape": score_shape,
        "complexity": complexity,
        "strata": {
            "score_shape": score_shape,
            "score_configuration": score_configuration,
            "complexity": complexity,
        },
        "source": source_image_origin,
    }


def _write_evaluation_manifest(
    path: Path,
    completed: list[tuple[dict[str, object], Path]],
    *,
    pipeline_evidence_fingerprint: str,
    source_manifest_sha256: str,
    source_image_origin: str,
    production_evidence: dict[str, object] | None,
) -> dict[str, object]:
    payload = {
        "format": 1,
        "name": "ScoreScan external boundary evaluation",
        "bootstrap_samples": 1000,
        "bootstrap_seed": 20260728,
        "pipeline_evidence_fingerprint": pipeline_evidence_fingerprint,
        "source_manifest_sha256": source_manifest_sha256,
        "source_image_origin": source_image_origin,
        "input_page_count": sum(
            int(case.get("input_pdf_pages", 0) or 0)
            for case, _candidate in completed
        ),
        "cases": [
            _evaluation_case(
                case,
                candidate,
                source_image_origin=source_image_origin,
            )
            for case, candidate in completed
        ],
    }
    if production_evidence is not None:
        payload["production_evidence"] = production_evidence
    atomic_write_json(path, payload)
    return payload


def run_benchmark(
    boundary_manifest: Path,
    product_root: Path,
    output_dir: Path,
    *,
    limit: int | None = None,
    case_ids: set[str] | None = None,
    timeout_seconds: int = 4 * 60 * 60,
    minimum_independent_works: int = 200,
    minimum_input_pages: int = 2_000,
    minimum_score_configuration_pages: int = 400,
    allow_diagnostic_manifest: bool = False,
    resources_dir: Path | None = None,
    required_accelerator: str | None = None,
) -> dict[str, object]:
    boundary_manifest = boundary_manifest.resolve()
    product_root = product_root.resolve()
    output_dir = output_dir.resolve()
    source_manifest_sha256 = sha256_file(boundary_manifest)
    source = json.loads(boundary_manifest.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("boundary manifest must be a JSON object")
    manifest_role = _validate_manifest_role(
        source,
        allow_diagnostic_manifest=allow_diagnostic_manifest,
    )
    source_image_origin = str(source.get("source_image_origin", ""))
    production_evidence = None
    source_cases = source.get("cases")
    if not isinstance(source_cases, list):
        raise ValueError("boundary manifest has no cases")
    accepted = [
        {
            **case,
            "_resolved_reference": str(
                (boundary_manifest.parent / str(case["reference"])).resolve()
            ),
        }
        for case in source_cases
        if isinstance(case, dict)
        and isinstance(case.get("boundary"), dict)
        and case["boundary"].get("accepted") is True
    ]
    stale_cases = [
        str(case.get("id", "<missing>"))
        for case in accepted
        if case["boundary"].get("contract_version")
        != PRODUCTION_BOUNDARY_CONTRACT_VERSION
    ]
    if stale_cases:
        raise ValueError(
            "boundary cases use a stale contract: "
            + ", ".join(stale_cases[:5])
        )
    partition_isolation = _validate_release_partition_isolation(
        source,
        accepted,
        boundary_manifest=boundary_manifest,
        manifest_role=manifest_role,
    )
    if case_ids:
        accepted = [
            case for case in accepted if str(case["id"]) in case_ids
        ]
        missing = sorted(case_ids - {str(case["id"]) for case in accepted})
        if missing:
            raise ValueError(
                "requested cases are absent or outside the boundary: "
                + ", ".join(missing)
            )
    accepted = _unique_work_cases(accepted)
    if limit is not None:
        accepted = accepted[: max(0, int(limit))]
    if (
        minimum_independent_works <= 0
        or minimum_input_pages <= 0
        or minimum_score_configuration_pages < 0
    ):
        raise ValueError("benchmark coverage minimums must be positive")
    (
        input_page_count,
        pages_by_score_configuration,
    ) = _production_page_coverage(accepted)
    if len(accepted) < minimum_independent_works:
        raise ValueError(
            "boundary benchmark has insufficient independent works: "
            f"{len(accepted)} < {minimum_independent_works}"
        )
    if input_page_count < minimum_input_pages:
        raise ValueError(
            "boundary benchmark has insufficient scan pages: "
            f"{input_page_count} < {minimum_input_pages}"
        )
    configuration_gaps = {
        name: minimum_score_configuration_pages - pages
        for name, pages in pages_by_score_configuration.items()
        if pages < minimum_score_configuration_pages
    }
    if configuration_gaps:
        detail = ", ".join(
            f"{name}={pages_by_score_configuration[name]}"
            f"<{minimum_score_configuration_pages}"
            for name in PRODUCTION_SCORE_CONFIGURATIONS
            if name in configuration_gaps
        )
        raise ValueError(
            "boundary benchmark has insufficient per-configuration scan "
            f"coverage: {detail}"
        )
    if manifest_role == RELEASE_BENCHMARK_ROLE:
        production_evidence = _materialize_production_evidence(
            source,
            boundary_manifest=boundary_manifest,
            output_dir=output_dir,
        )
    candidates_dir = output_dir / "candidates"
    records_dir = output_dir / "case_records"
    workspace = output_dir / "pipeline_workspace"
    for path in (candidates_dir, records_dir, workspace):
        path.mkdir(parents=True, exist_ok=True)

    base_settings = Settings.from_root(product_root)
    selected_resources = (
        resources_dir.resolve()
        if resources_dir is not None
        else base_settings.resources.resolve()
    )
    semantic_detector_required = bool(
        manifest_role == RELEASE_BENCHMARK_ROLE
        or resources_dir is not None
        or required_accelerator is not None
    )
    pipeline_evidence = _pipeline_evidence(
        product_root=product_root,
        resources_dir=selected_resources,
        semantic_detector_required=semantic_detector_required,
    )
    pipeline_evidence_fingerprint = str(pipeline_evidence["fingerprint"])
    semantic_model_version = str(
        pipeline_evidence["resources"]["semantic_detector_model_version"]
        or "asset_absent"
    )
    settings = replace(
        base_settings,
        workspace=workspace,
        resources=selected_resources,
        job_retention_days=0,
    )
    manager = JobManager(settings)
    completed: list[tuple[dict[str, object], Path]] = []
    failures: list[dict[str, str]] = []
    for position, case in enumerate(accepted, start=1):
        case_id = str(case["id"])
        candidate_path = candidates_dir / f"{case_id}.musicxml"
        conversion_report = (
            candidates_dir / f"{case_id}.conversion_report.json"
        )
        record_path = records_dir / f"{case_id}.json"
        existing = read_json(record_path, {})
        reusable = _record_is_reusable(
            existing,
            candidate_path=candidate_path,
            conversion_report=conversion_report,
            input_pdf_sha256=case.get("input_pdf_sha256"),
            pipeline_evidence_fingerprint=pipeline_evidence_fingerprint,
        )
        if reusable:
            try:
                _accelerator_execution_evidence(
                    read_json(conversion_report, {}),
                    page_count=int(case["input_pdf_pages"]),
                    semantic_model_version=semantic_model_version,
                    required_accelerator=required_accelerator,
                    semantic_detector_required=(
                        semantic_detector_required
                    ),
                )
            except (TypeError, ValueError, OverflowError):
                reusable = False
        if reusable:
            print(f"[{position}/{len(accepted)}] reuse {case_id}", flush=True)
            completed.append((case, candidate_path))
            continue

        input_pdf = Path(str(case["input_pdf"])).resolve()
        print(f"[{position}/{len(accepted)}] run {case_id}: {input_pdf.name}", flush=True)
        started = time.monotonic()
        job = None
        try:
            job = manager.create_job([input_pdf], [input_pdf.name])
            job = _wait(manager, job.id, timeout_seconds)
            if job.status != "completed" or not job.result_musicxml:
                raise RuntimeError(job.error or f"pipeline ended in {job.status}")
            result_path = Path(job.result_musicxml)
            atomic_write_bytes(candidate_path, result_path.read_bytes())
            if not job.report_path or not Path(job.report_path).is_file():
                raise RuntimeError("completed pipeline did not produce an audit report")
            atomic_write_bytes(
                conversion_report,
                Path(job.report_path).read_bytes(),
            )
            execution_evidence = _accelerator_execution_evidence(
                read_json(conversion_report, {}),
                page_count=int(case["input_pdf_pages"]),
                semantic_model_version=semantic_model_version,
                required_accelerator=required_accelerator,
                semantic_detector_required=semantic_detector_required,
            )
            record = {
                "format": 2,
                "created_at": utc_now_iso(),
                "status": "completed",
                "case_id": case_id,
                "work_fingerprint": case["work_fingerprint"],
                "job_id": job.id,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "quality_state": job.quality_state,
                "quality_score": job.quality_score,
                "warning_count": len(job.warnings),
                "review_issue_count": len(job.review_issues),
                "candidate_sha256": sha256_file(candidate_path),
                "conversion_report_sha256": sha256_file(conversion_report),
                "input_pdf_sha256": case.get("input_pdf_sha256"),
                "application_version": APP_VERSION,
                "workflow_version": WORKFLOW_VERSION,
                "boundary_contract_version": (
                    PRODUCTION_BOUNDARY_CONTRACT_VERSION
                ),
                "pipeline_evidence_fingerprint": (
                    pipeline_evidence_fingerprint
                ),
                "semantic_detector_model_version": pipeline_evidence[
                    "resources"
                ]["semantic_detector_model_version"],
                "semantic_detector_manifest_sha256": pipeline_evidence[
                    "resources"
                ]["semantic_detector_manifest_sha256"],
                "execution_accelerator": execution_evidence,
            }
            atomic_write_json(record_path, record)
            completed.append((case, candidate_path))
        except Exception as exc:
            failure = {
                "case_id": case_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            atomic_write_json(
                record_path,
                {
                    "format": 2,
                    "created_at": utc_now_iso(),
                    "status": "failed",
                    "input_pdf_sha256": case.get("input_pdf_sha256"),
                    "boundary_contract_version": (
                        PRODUCTION_BOUNDARY_CONTRACT_VERSION
                    ),
                    "pipeline_evidence_fingerprint": (
                        pipeline_evidence_fingerprint
                    ),
                    **failure,
                },
            )
        finally:
            if job is not None:
                terminal = manager.get(job.id)
                if terminal is not None and terminal.status in TERMINAL_STATES:
                    manager.remove(job.id)
        _write_evaluation_manifest(
            output_dir / "evaluation_manifest.json",
            completed,
            pipeline_evidence_fingerprint=pipeline_evidence_fingerprint,
            source_manifest_sha256=source_manifest_sha256,
            source_image_origin=source_image_origin,
            production_evidence=production_evidence,
        )

    evaluation = None
    evaluation_manifest = output_dir / "evaluation_manifest.json"
    if completed:
        _write_evaluation_manifest(
            evaluation_manifest,
            completed,
            pipeline_evidence_fingerprint=pipeline_evidence_fingerprint,
            source_manifest_sha256=source_manifest_sha256,
            source_image_origin=source_image_origin,
            production_evidence=production_evidence,
        )
        evaluation = evaluate_manifest(
            evaluation_manifest,
            split="test",
            bootstrap_samples_override=min(1000, max(100, len(completed) * 20)),
            gate_profile="production-v2",
        )
        atomic_write_json(output_dir / "evaluation_report.json", evaluation)
    release_evidence_eligible = manifest_role == RELEASE_BENCHMARK_ROLE
    production_gate_passed = (
        release_evidence_eligible
        and bool(evaluation["release_gate"]["passed"])
        if isinstance(evaluation, dict)
        else False
    )
    report = {
        "format": 2,
        "created_at": utc_now_iso(),
        "source_manifest_sha256": source_manifest_sha256,
        "boundary_contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
        "pipeline_evidence": pipeline_evidence,
        "pipeline_evidence_fingerprint": pipeline_evidence_fingerprint,
        "required_accelerator": required_accelerator,
        "semantic_detector_required": semantic_detector_required,
        "requested_case_count": len(accepted),
        "requested_independent_work_count": len(accepted),
        "requested_input_page_count": input_page_count,
        "requested_input_pages_by_score_configuration": (
            pages_by_score_configuration
        ),
        "completed_case_count": len(completed),
        "completed_independent_work_count": len(completed),
        "failed_case_count": len(failures),
        "failures": failures,
        "production_gate_passed": production_gate_passed,
        "production_evidence_eligible": release_evidence_eligible,
        # Kept for readers of the format-1 report; the authoritative profile is
        # explicit below and is no longer the historical stable-v1 gate.
        "stable_gate_passed": production_gate_passed,
        "release_gate_profile": "production-v2",
        "source_manifest_role": manifest_role,
        "diagnostic_manifest": not release_evidence_eligible,
        "partition_isolation": partition_isolation,
    }
    atomic_write_json(output_dir / "run_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("boundary_manifest", type=Path)
    parser.add_argument("product_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--timeout-seconds", type=int, default=4 * 60 * 60)
    parser.add_argument("--minimum-independent-works", type=int, default=200)
    parser.add_argument("--minimum-input-pages", type=int, default=2_000)
    parser.add_argument(
        "--minimum-score-configuration-pages",
        type=int,
        default=int(
            PRODUCTION_RELEASE_GATES_V2["minimum"][
                "solo_monophonic_page_count"
            ]
        ),
    )
    parser.add_argument("--allow-diagnostic-manifest", action="store_true")
    parser.add_argument(
        "--resources-dir",
        type=Path,
        help=(
            "complete verified resources directory containing the isolated "
            "authorized semantic release candidate"
        ),
    )
    parser.add_argument(
        "--require-accelerator",
        choices=("cpu",),
        help=(
            "fail each case unless OCR and semantic inference are verified on "
            "this accelerator for every page"
        ),
    )
    args = parser.parse_args()
    report = run_benchmark(
        args.boundary_manifest.resolve(),
        args.product_root.resolve(),
        args.output_dir.resolve(),
        limit=args.limit,
        case_ids=set(args.case_ids or ()),
        timeout_seconds=args.timeout_seconds,
        minimum_independent_works=args.minimum_independent_works,
        minimum_input_pages=args.minimum_input_pages,
        minimum_score_configuration_pages=(
            args.minimum_score_configuration_pages
        ),
        allow_diagnostic_manifest=args.allow_diagnostic_manifest,
        resources_dir=(
            args.resources_dir.resolve()
            if args.resources_dir is not None
            else None
        ),
        required_accelerator=args.require_accelerator,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["failed_case_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
