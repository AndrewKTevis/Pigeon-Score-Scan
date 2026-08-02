from pathlib import Path

from scorescan.models import JobState
from scorescan.persistence import JobRepository


def test_job_repository_roundtrip(tmp_path: Path) -> None:
    repository = JobRepository(tmp_path)
    job = JobState("abc", tmp_path / "abc", "images", ["1.png"], output_name="Quartet", pdf_dpi=500)
    job.persist_hook = repository.persist
    job.update(stage="test", progress=.4)
    loaded = repository.load_all()
    assert len(loaded) == 1
    assert loaded[0].stage == "test"
    assert loaded[0].progress == .4
    assert loaded[0].output_name == "Quartet"
    assert loaded[0].pdf_dpi == 500


def test_job_repository_recovers_corrupt_primary_from_independent_backup(tmp_path: Path) -> None:
    repository = JobRepository(tmp_path)
    job = JobState("abc", tmp_path / "abc", "images", ["1.png"])
    job.persist_hook = repository.persist
    job.update(stage="recognizing", progress=.6)
    primary = job.root / "job.json"
    backup = job.root / "job.backup.json"
    assert backup.exists()
    primary.write_text('{"broken":', encoding="utf-8")

    loaded = repository.load_all()
    assert len(loaded) == 1
    assert loaded[0].stage == "recognizing"
    assert loaded[0].progress == .6
    assert any("备份恢复" in warning for warning in loaded[0].warnings)
    assert '"id": "abc"' in primary.read_text(encoding="utf-8")


def test_job_repository_rejects_backup_with_wrong_job_identity(tmp_path: Path) -> None:
    root = tmp_path / "abc"
    root.mkdir()
    (root / "job.json").write_text("not-json", encoding="utf-8")
    (root / "job.backup.json").write_text('{"id":"other","root":"x","mode":"images","source_files":[]}', encoding="utf-8")
    assert JobRepository(tmp_path).load_all() == []


def test_repository_recovers_when_primary_snapshot_is_missing(tmp_path: Path) -> None:
    root = tmp_path / "job-only-backup"
    root.mkdir()
    state = JobState(id="job-only-backup", root=root, mode="images", source_files=["page.png"])
    repository = JobRepository(tmp_path)
    repository.persist(state)
    (root / "job.json").unlink()

    loaded = repository.load_all()

    assert len(loaded) == 1
    assert loaded[0].id == "job-only-backup"
    assert (root / "job.json").exists()
    assert any("独立备份恢复" in item for item in loaded[0].warnings)


def test_repository_quarantines_invalid_backup_when_primary_is_absent(tmp_path: Path) -> None:
    root = tmp_path / "backup-only-invalid"
    root.mkdir()
    backup = root / "job.backup.json"
    backup.write_text('{"id":"other","mode":"images","source_files":[]}', encoding="utf-8")

    assert JobRepository(tmp_path).load_all() == []
    assert not backup.exists()
    assert list(root.glob("job.corrupt.*.json"))
