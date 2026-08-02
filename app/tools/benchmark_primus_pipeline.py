from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.config import APP_VERSION, WORKFLOW_VERSION, Settings  # noqa: E402
from scorescan.jobs import JobManager  # noqa: E402
from scorescan.primus_evaluation import (  # noqa: E402
    aggregate_primus_reports,
    compare_primus_semantics,
    parse_musicxml_semantics,
    parse_primus_semantic,
)
from scorescan.util import atomic_write_json, sha256_file, utc_now_iso  # noqa: E402


def _selected_cases(engine_report: Path, limit: int, challenge: bool) -> list[dict[str, object]]:
    payload = json.loads(engine_report.read_text(encoding="utf-8"))
    cases = list(payload.get("cases", []))
    if challenge:
        cases.sort(
            key=lambda item: (
                item.get("error") is None,
                -float(item.get("event_error_rate", 0.0) or 0.0),
                str(item.get("case", "")),
            )
        )
    return cases[:limit]


def _wait_for_job(manager: JobManager, job_id: str, timeout_seconds: int):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        state = manager.get(job_id)
        if state is None:
            raise RuntimeError("benchmark job disappeared")
        if state.status in {"completed", "failed", "cancelled"}:
            return state
        time.sleep(0.25)
    state = manager.get(job_id)
    if state is not None:
        manager.cancel(job_id)
    raise TimeoutError(f"pipeline case exceeded {timeout_seconds} seconds")


def run_benchmark(
    product_root: Path,
    dataset: Path,
    engine_report: Path,
    output: Path,
    limit: int,
    challenge: bool,
    timeout_seconds: int,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    cases_root = output / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)
    base_settings = Settings.from_root(product_root)
    settings = replace(
        base_settings,
        workspace=output / "workspace",
        job_retention_days=0,
    )
    manager = JobManager(settings)
    selected = _selected_cases(engine_report, limit, challenge)
    reports: list[dict[str, object]] = []
    started = time.monotonic()
    metadata: dict[str, object] = {
        "application_version": APP_VERSION,
        "workflow_version": WORKFLOW_VERSION,
        "dataset_name": "PrIMuS",
        "dataset_scope": "synthetic monophonic printed incipits",
        "selection": "engine failures then highest event error" if challenge else "engine-report order",
        "requested_cases": limit,
        "selected_cases": len(selected),
        "engine_report": str(engine_report.resolve()),
        "started_at": utc_now_iso(),
    }

    for index, source_case in enumerate(selected, start=1):
        case_started = time.monotonic()
        case_id = str(source_case["case"])
        semantic_path = dataset / str(source_case["semantic_path"])
        source_image = semantic_path.with_suffix(".png")
        case_root = cases_root / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        persisted_result = case_root / "result.musicxml"
        report: dict[str, object] = {
            "case": case_id,
            "semantic_path": str(semantic_path.relative_to(dataset)),
            "source_sha256": sha256_file(source_image),
            "engine_baseline_error": source_case.get("error"),
            "engine_baseline_event_error_rate": source_case.get("event_error_rate"),
        }
        try:
            if persisted_result.exists():
                candidate_path = persisted_result
                report["resumed"] = True
            else:
                job = manager.create_job([source_image], [source_image.name])
                job = _wait_for_job(manager, job.id, timeout_seconds)
                report["job_id"] = job.id
                report["job_status"] = job.status
                report["quality_state"] = job.quality_state
                report["quality_score"] = job.quality_score
                report["warning_count"] = len(job.warnings)
                report["review_issue_count"] = len(job.review_issues)
                if job.status != "completed" or not job.result_musicxml:
                    raise RuntimeError(job.error or f"pipeline ended with {job.status}")
                candidate_path = Path(job.result_musicxml)
                shutil.copy2(candidate_path, persisted_result)
                if job.report_path:
                    shutil.copy2(job.report_path, case_root / "conversion_report.json")
            reference = parse_primus_semantic(semantic_path)
            candidate = parse_musicxml_semantics(candidate_path)
            report.update(compare_primus_semantics(reference, candidate))
            report["candidate_sha256"] = sha256_file(candidate_path)
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        report["elapsed_seconds"] = round(time.monotonic() - case_started, 3)
        reports.append(report)
        atomic_write_json(
            output / "report.json",
            {
                "metadata": metadata,
                "aggregate": aggregate_primus_reports(reports),
                "cases": reports,
            },
        )
        status = "ERROR" if report.get("error") else (
            "EXACT" if report.get("semantic_exact") else f"EER={float(report['event_error_rate']):.3f}"
        )
        print(f"[{index:03d}/{len(selected):03d}] {case_id}: {status} ({report['elapsed_seconds']}s)", flush=True)

    metadata["completed_at"] = utc_now_iso()
    metadata["elapsed_seconds"] = round(time.monotonic() - started, 3)
    final = {
        "metadata": metadata,
        "aggregate": aggregate_primus_reports(reports),
        "cases": reports,
    }
    atomic_write_json(output / "report.json", final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete ScoreScan pipeline against cases from a PrIMuS engine report."
    )
    parser.add_argument("--product-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--engine-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--challenge",
        action="store_true",
        help="Select engine failures first, then the highest-error successful cases",
    )
    parser.add_argument("--timeout-seconds", type=int, default=20 * 60)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    result = run_benchmark(
        args.product_root,
        args.dataset,
        args.engine_report,
        args.output,
        args.limit,
        args.challenge,
        args.timeout_seconds,
    )
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
