from __future__ import annotations

"""Run one input through the complete ScoreScan job pipeline.

This is intentionally a product-path smoke runner, not a direct homr wrapper.  It
exercises import, normalization, layout, candidate generation, selection, coverage
audit, review generation, preview, MXL packaging and artifact integrity.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.config import APP_VERSION, WORKFLOW_VERSION, Settings  # noqa: E402
from scorescan.jobs import JobManager  # noqa: E402
from scorescan.util import atomic_write_json, sha256_file, utc_now_iso  # noqa: E402


TERMINAL_STATES = {"completed", "failed", "cancelled"}


def _wait(manager: JobManager, job_id: str, timeout_seconds: int):
    deadline = time.monotonic() + timeout_seconds
    last_progress = -1.0
    while time.monotonic() < deadline:
        state = manager.get(job_id)
        if state is None:
            raise RuntimeError("job disappeared from the workspace")
        if state.progress != last_progress:
            print(
                f"{state.progress * 100:6.2f}%  {state.status:10s}  {state.stage}",
                flush=True,
            )
            last_progress = state.progress
        if state.status in TERMINAL_STATES:
            return state
        time.sleep(0.25)
    manager.cancel(job_id)
    raise TimeoutError(f"pipeline exceeded {timeout_seconds} seconds")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one scan through the complete ScoreScan product pipeline.",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--product-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    source = args.input.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    product_root = args.product_root.resolve()
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    summary_path = (args.summary or (workspace / "pipeline_case_summary.json")).resolve()

    settings = replace(
        Settings.from_root(product_root),
        workspace=workspace,
        job_retention_days=0,
    )
    manager = JobManager(settings)
    started = time.monotonic()
    job = manager.create_job([source], [source.name])
    try:
        job = _wait(manager, job.id, max(1, args.timeout_seconds))
    except Exception as exc:
        payload = {
            "application_version": APP_VERSION,
            "workflow_version": WORKFLOW_VERSION,
            "created_at": utc_now_iso(),
            "input_sha256": sha256_file(source),
            "job_id": job.id,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }
        atomic_write_json(summary_path, payload)
        raise

    payload = {
        "application_version": APP_VERSION,
        "workflow_version": WORKFLOW_VERSION,
        "created_at": utc_now_iso(),
        "input_sha256": sha256_file(source),
        "job_id": job.id,
        "status": job.status,
        "stage": job.stage,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "quality_state": job.quality_state,
        "quality_score": job.quality_score,
        "warning_count": len(job.warnings),
        "review_issue_count": len(job.review_issues),
        "result_musicxml": job.result_musicxml,
        "result_mxl": job.result_mxl,
        "conversion_report": job.report_path,
        "artifact_manifest": job.artifact_manifest_path,
        "error": job.error,
    }
    atomic_write_json(summary_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    if job.status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
