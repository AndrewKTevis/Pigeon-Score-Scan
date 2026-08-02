from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from .models import JobState
from .state_schema import UnsupportedJobSchema
from .util import atomic_write_json, atomic_write_text, read_json


class JobRepository:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def persist(self, job: JobState) -> None:
        with self._lock:
            payload = job.to_dict(include_logs=True)
            atomic_write_json(job.root / "job.json", payload)
            # Keep a second independently replaced snapshot.  The primary file remains
            # authoritative, while the backup permits deterministic recovery from
            # isolated filesystem corruption or accidental truncation.
            atomic_write_json(job.root / "job.backup.json", payload)
            logs_dir = job.root / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            # Both state and log are atomic.  A hard power loss can leave an old file,
            # but never a truncated UTF-8 document that prevents recovery.
            atomic_write_text(logs_dir / "job.log", "\n".join(job.logs) + ("\n" if job.logs else ""))

    @staticmethod
    def _quarantine(job_file: Path, reason: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        quarantine = job_file.with_name(f"job.corrupt.{timestamp}.json")
        try:
            job_file.replace(quarantine)
            atomic_write_text(quarantine.with_suffix(".reason.txt"), reason)
        except OSError:
            pass

    def load_all(self) -> list[JobState]:
        results: list[JobState] = []
        # Discover a job when either snapshot exists.  A power loss or manual
        # cleanup can remove the primary file while leaving the independently
        # replaced backup intact; recovery must not depend on job.json being
        # present merely to discover the directory.
        job_dirs = {
            path.parent
            for pattern in ("*/job.json", "*/job.backup.json")
            for path in self.workspace.glob(pattern)
        }
        for job_dir in sorted(job_dirs):
            job_file = job_dir / "job.json"
            backup_file = job_dir / "job.backup.json"

            def quarantine_snapshot(reason: str) -> None:
                self._quarantine(job_file if job_file.exists() else backup_file, reason)

            payload = read_json(job_file)
            recovered = False
            if not isinstance(payload, dict) or "id" not in payload:
                backup = read_json(backup_file)
                if isinstance(backup, dict) and "id" in backup:
                    payload = backup
                    recovered = True
                else:
                    quarantine_snapshot("任务状态不是有效的 JSON 对象，或缺少 id")
                    continue
            try:
                state = JobState.from_dict(job_file.parent, payload)
            except (TypeError, ValueError, UnsupportedJobSchema, json.JSONDecodeError) as exc:
                backup = read_json(backup_file) if not recovered else None
                if isinstance(backup, dict) and "id" in backup:
                    try:
                        state = JobState.from_dict(job_file.parent, backup)
                        payload = backup
                        recovered = True
                    except (TypeError, ValueError, UnsupportedJobSchema, json.JSONDecodeError):
                        quarantine_snapshot(f"任务状态无法迁移：{exc}")
                        continue
                else:
                    quarantine_snapshot(f"任务状态无法迁移：{exc}")
                    continue
            if state.id != job_file.parent.name:
                quarantine_snapshot("任务状态 id 与目录名不一致")
                continue
            state.persist_hook = self.persist
            if recovered:
                state.warnings.append("任务状态主文件损坏，已从独立备份恢复")
                atomic_write_json(job_file, state.to_dict(include_logs=True))
            results.append(state)
        return results
