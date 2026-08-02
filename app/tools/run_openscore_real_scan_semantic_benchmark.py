from __future__ import annotations

"""Run split-inherited historical scans through the complete product pipeline.

This is a development benchmark, not a release gate.  Every result retains the
OpenScore work split and is reported separately for train, calibration, and test.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

# This orchestration process already parallelizes at the page/model level.
# Letting numerical backends also allocate one worker per logical CPU can make
# even ``--help`` fail on a memory-constrained training workstation. Explicit
# operator overrides are preserved.
for _thread_limit_variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_limit_variable, "1")

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
sys.path.insert(0, str(APP_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from scorescan.config import APP_VERSION, WORKFLOW_VERSION, Settings  # noqa: E402
from scorescan.evaluation import compare_musicxml  # noqa: E402
from scorescan.jobs import JobManager  # noqa: E402
from scorescan.product_scope import (  # noqa: E402
    PRODUCTION_BOUNDARY_CONTRACT_VERSION,
)
from scorescan.util import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_json,
    read_json,
    sha256_file,
    utc_now_iso,
)

from app.tools.evaluate_release_dataset import aggregate_reports  # noqa: E402
from app.tools.prepare_openscore_real_scan_semantic_corpus import (  # noqa: E402
    ROLE as SEMANTIC_CORPUS_ROLE,
)
from app.tools.prepare_muse_omr_benchmark import (  # noqa: E402
    analyze_reference_boundary,
)
from app.tools.prepare_pdmx_imslp_semantic_corpus import (  # noqa: E402
    ROLE as PDMX_SEMANTIC_CORPUS_ROLE,
)


ROLE = "real_scan_split_inherited_whole_work_semantic_development_run"
TERMINAL_STATES = {"completed", "failed", "cancelled"}
ALLOWED_SPLITS = {"train", "calibration", "test"}
ALLOWED_SEMANTIC_CORPUS_ROLES = {
    SEMANTIC_CORPUS_ROLE,
    PDMX_SEMANTIC_CORPUS_ROLE,
}


def _wait(manager: object, job_id: str, timeout_seconds: int) -> object:
    deadline = time.monotonic() + max(1, timeout_seconds)
    while time.monotonic() < deadline:
        state = manager.get(job_id)  # type: ignore[attr-defined]
        if state is None:
            raise RuntimeError("benchmark job disappeared")
        if state.status in TERMINAL_STATES:
            return state
        time.sleep(0.25)
    manager.cancel(job_id)  # type: ignore[attr-defined]
    raise TimeoutError(f"benchmark case exceeded {timeout_seconds} seconds")


def _validated_cases(
    semantic_manifest: Path,
    *,
    splits: set[str] | None,
    case_ids: set[str] | None,
    limit: int | None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    source = json.loads(semantic_manifest.read_text(encoding="utf-8"))
    if (
        not isinstance(source, dict)
        or source.get("role") not in ALLOWED_SEMANTIC_CORPUS_ROLES
        or source.get("boundary_contract_version")
        != PRODUCTION_BOUNDARY_CONTRACT_VERSION
        or source.get(
            "whole_work_semantic_development_evaluation_authorized"
        )
        is not True
        or source.get("training_authorized") is not False
        or source.get("page_training_labels_authorized") is not False
        or source.get("release_evaluation_authorized") is not False
        or source.get("release_authorized") is not False
        or source.get("independent_holdout") is not False
    ):
        raise ValueError("unexpected semantic development manifest contract")
    if source.get("role") == PDMX_SEMANTIC_CORPUS_ROLE and (
        source.get("page_level_training_authorized") is not False
        or source.get("page_level_release_evaluation_authorized") is not False
        or source.get("independent_release_evaluation_authorized") is not False
    ):
        raise ValueError("PDMX semantic manifest lacks explicit use restrictions")
    raw_cases = source.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("semantic development manifest has no cases")
    selected: list[dict[str, object]] = []
    seen_work_splits: dict[str, str] = {}
    seen_case_ids: set[str] = set()
    seen_pdfs: dict[str, str] = {}
    for row in raw_cases:
        if (
            not isinstance(row, dict)
            or row.get(
                "whole_work_semantic_development_evaluation_authorized"
            )
            is not True
            or row.get("page_training_labels_authorized") is not False
            or row.get("boundary_identity_consistent") is not True
        ):
            raise ValueError("semantic case bypassed its authorization gate")
        if source.get("role") == PDMX_SEMANTIC_CORPUS_ROLE and (
            row.get("page_level_training_authorized") is not False
            or row.get("page_level_release_evaluation_authorized") is not False
            or row.get("independent_release_evaluation_authorized") is not False
        ):
            raise ValueError("PDMX semantic case lacks explicit use restrictions")
        if (
            source.get("role") == PDMX_SEMANTIC_CORPUS_ROLE
            and row.get("exact_scan_to_semantic_alignment_verified") is not True
        ):
            raise ValueError(
                "PDMX semantic case lacks exact scan-to-reference alignment"
            )
        split = str(row.get("semantic_split", ""))
        case_id = str(row.get("id", ""))
        work = str(row.get("work_fingerprint", ""))
        pdf_hash = str(row.get("input_pdf_sha256", "")).casefold()
        if (
            split not in ALLOWED_SPLITS
            or not case_id
            or len(work) != 64
            or len(pdf_hash) != 64
        ):
            raise ValueError("semantic case has invalid identity or split")
        if case_id in seen_case_ids:
            raise ValueError("semantic manifest repeats a case id")
        seen_case_ids.add(case_id)
        previous_work_split = seen_work_splits.setdefault(work, split)
        if previous_work_split != split:
            raise ValueError("semantic work crosses inherited splits")
        previous_pdf_split = seen_pdfs.setdefault(pdf_hash, split)
        if previous_pdf_split != split:
            raise ValueError("semantic scan crosses inherited splits")
        reference = (
            semantic_manifest.parent / str(row.get("reference", ""))
        ).resolve()
        manifest_root = semantic_manifest.parent.resolve()
        if (
            not reference.is_relative_to(manifest_root)
            or not reference.is_file()
            or sha256_file(reference)
            != str(row.get("reference_sha256", "")).casefold()
        ):
            raise ValueError(
                f"{case_id} reference is missing, unsafe, or changed"
            )
        current_boundary = analyze_reference_boundary(reference)
        if (
            current_boundary.get("contract_version")
            != PRODUCTION_BOUNDARY_CONTRACT_VERSION
            or current_boundary.get("accepted") is not True
        ):
            reasons = current_boundary.get("reasons")
            if isinstance(reasons, list):
                reason_text = ",".join(str(item) for item in reasons)
            else:
                reason_text = str(reasons or "unspecified")
            raise ValueError(
                f"{case_id} reference failed current product-boundary "
                f"preflight: {reason_text}"
            )
        if splits is not None and split not in splits:
            continue
        if case_ids is not None and case_id not in case_ids:
            continue
        selected.append(row)
    if case_ids is not None:
        missing = sorted(case_ids - {str(row["id"]) for row in selected})
        if missing:
            raise ValueError(
                "requested semantic cases are absent: " + ", ".join(missing)
            )
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        selected = selected[:limit]
    if not selected:
        raise ValueError("no semantic cases selected")
    return source, selected


def _resolved_case(
    case: dict[str, object],
    semantic_manifest: Path,
) -> tuple[Path, Path]:
    input_pdf = Path(str(case.get("input_pdf", ""))).resolve()
    reference = (
        semantic_manifest.parent / str(case.get("reference", ""))
    ).resolve()
    if (
        not input_pdf.is_file()
        or sha256_file(input_pdf)
        != str(case.get("input_pdf_sha256", "")).casefold()
    ):
        raise ValueError(f"{case.get('id')} scan PDF hash mismatch")
    if (
        not reference.is_file()
        or sha256_file(reference)
        != str(case.get("reference_sha256", "")).casefold()
    ):
        raise ValueError(f"{case.get('id')} reference hash mismatch")
    return input_pdf, reference


def run_benchmark(
    semantic_manifest: Path,
    product_root: Path,
    output_dir: Path,
    *,
    splits: set[str] | None = None,
    case_ids: set[str] | None = None,
    limit: int | None = None,
    timeout_seconds: int = 4 * 60 * 60,
    manager_factory: Callable[[Settings], object] = JobManager,
    comparator: Callable[[Path, Path], dict[str, object]] = (
        compare_musicxml
    ),
) -> dict[str, object]:
    source, selected = _validated_cases(
        semantic_manifest,
        splits=splits,
        case_ids=case_ids,
        limit=limit,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = output_dir / "candidates"
    records_dir = output_dir / "case_records"
    workspace = output_dir / "pipeline_workspace"
    for path in (candidates_dir, records_dir, workspace):
        path.mkdir(parents=True, exist_ok=True)

    settings = replace(
        Settings.from_root(product_root),
        workspace=workspace,
        job_retention_days=0,
    )
    manager = manager_factory(settings)
    reports: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    started = time.monotonic()
    for position, case in enumerate(selected, start=1):
        case_id = str(case["id"])
        input_pdf, reference = _resolved_case(case, semantic_manifest)
        candidate = candidates_dir / f"{case_id}.musicxml"
        conversion_report = (
            candidates_dir / f"{case_id}.conversion_report.json"
        )
        record_path = records_dir / f"{case_id}.json"
        existing = read_json(record_path, {})
        reusable = bool(
            candidate.is_file()
            and isinstance(existing, dict)
            and existing.get("status") == "completed"
            and existing.get("candidate_sha256") == sha256_file(candidate)
            and existing.get("input_pdf_sha256")
            == case.get("input_pdf_sha256")
            and existing.get("reference_sha256")
            == case.get("reference_sha256")
            and existing.get("workflow_version") == WORKFLOW_VERSION
        )
        case_started = time.monotonic()
        job = None
        try:
            if reusable:
                print(
                    f"[{position}/{len(selected)}] reuse {case_id}",
                    flush=True,
                )
            else:
                print(
                    f"[{position}/{len(selected)}] run {case_id}: "
                    f"{input_pdf.name}",
                    flush=True,
                )
                job = manager.create_job(  # type: ignore[attr-defined]
                    [input_pdf],
                    [input_pdf.name],
                )
                job = _wait(manager, job.id, timeout_seconds)
                if job.status != "completed" or not job.result_musicxml:
                    raise RuntimeError(
                        job.error or f"pipeline ended in {job.status}"
                    )
                result_path = Path(job.result_musicxml)
                atomic_write_bytes(candidate, result_path.read_bytes())
                if job.report_path and Path(job.report_path).is_file():
                    atomic_write_bytes(
                        conversion_report,
                        Path(job.report_path).read_bytes(),
                    )
                atomic_write_json(
                    record_path,
                    {
                        "format": 1,
                        "created_at": utc_now_iso(),
                        "status": "completed",
                        "case_id": case_id,
                        "semantic_split": case["semantic_split"],
                        "work_fingerprint": case["work_fingerprint"],
                        "job_id": job.id,
                        "quality_state": job.quality_state,
                        "quality_score": job.quality_score,
                        "warning_count": len(job.warnings),
                        "review_issue_count": len(job.review_issues),
                        "candidate_sha256": sha256_file(candidate),
                        "input_pdf_sha256": case["input_pdf_sha256"],
                        "reference_sha256": case["reference_sha256"],
                        "application_version": APP_VERSION,
                        "workflow_version": WORKFLOW_VERSION,
                    },
                )
            metrics = comparator(reference, candidate)
            current_record = read_json(record_path, {})
            report = {
                "case_id": case_id,
                "semantic_split": case["semantic_split"],
                "split_role": case["split_role"],
                "work_fingerprint": case["work_fingerprint"],
                "input_pdf_pages": case["input_pdf_pages"],
                "input_pdf_sha256": case["input_pdf_sha256"],
                "reference_sha256": case["reference_sha256"],
                "candidate_sha256": sha256_file(candidate),
                "quality_state": current_record.get("quality_state"),
                "quality_score": current_record.get("quality_score"),
                "warning_count": current_record.get("warning_count"),
                "review_issue_count": current_record.get(
                    "review_issue_count"
                ),
                "elapsed_seconds": round(
                    time.monotonic() - case_started,
                    3,
                ),
                "metrics": metrics,
            }
            reports.append(report)
            print(
                f"[{position}/{len(selected)}] evaluated {case_id}: "
                f"utility={float(metrics.get('utility_score', 0.0)):.6f}",
                flush=True,
            )
        except Exception as error:
            failure = {
                "case_id": case_id,
                "semantic_split": case["semantic_split"],
                "error": f"{type(error).__name__}: {error}",
            }
            failures.append(failure)
            atomic_write_json(
                record_path,
                {
                    "format": 1,
                    "created_at": utc_now_iso(),
                    "status": "failed",
                    **failure,
                },
            )
        finally:
            if job is not None:
                terminal = manager.get(job.id)  # type: ignore[attr-defined]
                if (
                    terminal is not None
                    and terminal.status in TERMINAL_STATES
                ):
                    manager.remove(job.id)  # type: ignore[attr-defined]
        _write_report(
            output_dir / "report.json",
            semantic_manifest=semantic_manifest,
            source=source,
            selected=selected,
            reports=reports,
            failures=failures,
            started=started,
        )
    return _write_report(
        output_dir / "report.json",
        semantic_manifest=semantic_manifest,
        source=source,
        selected=selected,
        reports=reports,
        failures=failures,
        started=started,
    )


def _write_report(
    path: Path,
    *,
    semantic_manifest: Path,
    source: dict[str, object],
    selected: list[dict[str, object]],
    reports: list[dict[str, object]],
    failures: list[dict[str, object]],
    started: float,
) -> dict[str, object]:
    metric_reports = [
        row["metrics"]
        for row in reports
        if isinstance(row.get("metrics"), dict)
    ]
    by_split: dict[str, dict[str, object]] = {}
    for split in sorted(ALLOWED_SPLITS):
        rows = [
            row["metrics"]
            for row in reports
            if row.get("semantic_split") == split
            and isinstance(row.get("metrics"), dict)
        ]
        if rows:
            by_split[split] = aggregate_reports(rows)
    payload = {
        "format": 1,
        "created_at": utc_now_iso(),
        "name": "ScoreScan OpenScore/IMSLP real-scan semantic development run",
        "role": ROLE,
        "semantic_manifest_path": str(semantic_manifest),
        "semantic_manifest_sha256": sha256_file(semantic_manifest),
        "source_manifest_role": source.get("role"),
        "boundary_contract_version": source.get(
            "boundary_contract_version"
        ),
        "application_version": APP_VERSION,
        "workflow_version": WORKFLOW_VERSION,
        "requested_case_count": len(selected),
        "requested_work_count": len(
            {str(row["work_fingerprint"]) for row in selected}
        ),
        "requested_page_count": sum(
            int(row["input_pdf_pages"]) for row in selected
        ),
        "completed_case_count": len(reports),
        "failed_case_count": len(failures),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "training_authorized": False,
        "release_evaluation_authorized": False,
        "release_gate_evaluated": False,
        "release_authorized": False,
        "independent_holdout": False,
        "aggregate": (
            aggregate_reports(metric_reports) if metric_reports else {}
        ),
        "aggregate_by_semantic_split": by_split,
        "failures": failures,
        "cases": reports,
    }
    atomic_write_json(path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("semantic_manifest", type=Path)
    parser.add_argument("product_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--split",
        action="append",
        choices=sorted(ALLOWED_SPLITS),
        dest="splits",
    )
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=4 * 60 * 60,
    )
    args = parser.parse_args()
    if not 60 <= args.timeout_seconds <= 24 * 60 * 60:
        raise ValueError("timeout-seconds must be between 60 and 86400")
    report = run_benchmark(
        args.semantic_manifest.resolve(),
        args.product_root.resolve(),
        args.output_dir.resolve(),
        splits=set(args.splits) if args.splits else None,
        case_ids=set(args.case_ids) if args.case_ids else None,
        limit=args.limit,
        timeout_seconds=args.timeout_seconds,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "requested_case_count",
                    "requested_page_count",
                    "completed_case_count",
                    "failed_case_count",
                    "elapsed_seconds",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if report["failed_case_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
