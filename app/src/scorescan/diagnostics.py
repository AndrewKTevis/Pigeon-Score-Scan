from __future__ import annotations

"""Build a privacy-safe support bundle without copying user scores or scans."""

import json
import os
import platform
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

from .accelerator import probe_accelerator
from .config import APP_VERSION, WORKFLOW_VERSION, Settings
from .model_registry import model_versions
from .policy import DEFAULT_POLICY
from .semantic_detector import semantic_detector_status
from .self_test import run_system_check
from .state_schema import CURRENT_JOB_SCHEMA
from .util import read_json, replace_file_with_retry


def _safe_job_summary(payload: dict[str, Any]) -> dict[str, Any]:
    pages = payload.get("pages") if isinstance(payload.get("pages"), list) else []
    return {
        "id": payload.get("id"),
        "status": payload.get("status"),
        "stage": payload.get("stage"),
        "progress": payload.get("progress"),
        "total_pages": payload.get("total_pages", len(pages)),
        "current_page": payload.get("current_page"),
        "quality_state": payload.get("quality_state"),
        "quality_score": payload.get("quality_score"),
        "warning_count": len(payload.get("warnings", [])) if isinstance(payload.get("warnings"), list) else 0,
        "error_type": str(payload.get("error", "")).split(":", 1)[0] or None,
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
    }


def create_diagnostics_bundle(settings: Settings) -> Path:
    target_dir = settings.runtime / "diagnostics"
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    output = target_dir / f"ScoreScan-diagnostics-{stamp}.zip"
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=output.name + ".",
        suffix=".tmp",
        dir=target_dir,
    )
    os.close(temporary_fd)
    temporary = Path(temporary_name)

    summaries: list[dict[str, Any]] = []
    # Current JobManager stores jobs directly under workspace/<job-id>.  Older
    # development builds used workspace/jobs/<job-id>; include both layouts so a
    # support bundle remains useful after an in-place upgrade.
    candidates = list(settings.workspace.glob("*/job.json"))
    legacy_jobs_root = settings.workspace / "jobs"
    if legacy_jobs_root.exists():
        candidates.extend(legacy_jobs_root.glob("*/job.json"))
    candidates = sorted(
        set(candidates),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:10]
    for path in candidates:
        payload = read_json(path, {})
        if isinstance(payload, dict):
            summaries.append(_safe_job_summary(payload))

    app_info = {
        "version": APP_VERSION,
        "workflow_version": WORKFLOW_VERSION,
        "policy_version": DEFAULT_POLICY.version,
        "job_schema": CURRENT_JOB_SCHEMA,
        "models": model_versions(settings.resources),
        "semantic_detector": semantic_detector_status(
            settings.resources
        ).to_dict(),
        "accelerator": probe_accelerator().to_dict(),
        "runtime_profile": os.getenv("SCORESCAN_RUNTIME_PROFILE", "development"),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pid": os.getpid(),
        "privacy_note": "本压缩包不包含扫描图片、PDF、MusicXML、OCR 文本或原始文件名。",
    }
    system_check = run_system_check(settings)

    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            archive.writestr("system_check.json", json.dumps(system_check, ensure_ascii=False, indent=2) + "\n")
            archive.writestr("application.json", json.dumps(app_info, ensure_ascii=False, indent=2) + "\n")
            archive.writestr("recent_jobs_redacted.json", json.dumps(summaries, ensure_ascii=False, indent=2) + "\n")
            manifest = settings.resources / "model_manifest.json"
            if manifest.exists():
                archive.write(manifest, "model_manifest.json")
        replace_file_with_retry(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    # Keep only a few support bundles.
    for old in sorted(target_dir.glob("ScoreScan-diagnostics-*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)[5:]:
        old.unlink(missing_ok=True)
    return output
