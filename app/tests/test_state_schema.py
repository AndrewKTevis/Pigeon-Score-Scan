import json
from pathlib import Path

import pytest

from scorescan.persistence import JobRepository
from scorescan.state_schema import CURRENT_JOB_SCHEMA, UnsupportedJobSchema, migrate_job_payload


def test_v1_job_state_migrates_forward() -> None:
    migrated = migrate_job_payload({"id": "abc", "mode": "images", "source_files": []})
    assert migrated["schema_version"] == CURRENT_JOB_SCHEMA
    assert migrated["review_issues"] == []
    assert migrated["resumable"] is True
    assert migrated["output_name"] is None
    assert migrated["pdf_dpi"] == 400


def test_future_job_state_is_rejected() -> None:
    with pytest.raises(UnsupportedJobSchema):
        migrate_job_payload({"schema_version": CURRENT_JOB_SCHEMA + 1})


def test_corrupt_job_state_is_quarantined(tmp_path: Path) -> None:
    root = tmp_path / "job-a"
    root.mkdir(parents=True)
    (root / "job.json").write_text("{not json", encoding="utf-8")
    repository = JobRepository(tmp_path)
    assert repository.load_all() == []
    assert not (root / "job.json").exists()
    assert list(root.glob("job.corrupt.*.json"))


def test_v18_job_state_migrates_to_grace_consensus_schema() -> None:
    migrated = migrate_job_payload({"schema_version": 18, "id": "legacy", "pages": []})
    assert migrated["schema_version"] == CURRENT_JOB_SCHEMA


def test_v20_job_state_migrates_to_patch_transaction_schema() -> None:
    migrated = migrate_job_payload({"schema_version": 20, "id": "legacy", "pages": []})
    assert migrated["schema_version"] == CURRENT_JOB_SCHEMA
