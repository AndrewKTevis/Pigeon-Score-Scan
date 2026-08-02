from __future__ import annotations

import threading
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from lxml import etree
from PIL import Image, ImageDraw

import scorescan.jobs as jobs_module
from scorescan.accelerator import probe_accelerator
from scorescan.config import Settings
from scorescan.jobs import JobManager
from scorescan.models import JobState
from scorescan.persistence import JobRepository
from scorescan.musicxml import MUSICXML_DOCTYPE
from scorescan.omr import EngineResult


def create_fake_musicxml(path: Path) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    for number in range(1, 5):
        measure = etree.SubElement(part, "measure", number=str(number))
        if number == 1:
            attributes = etree.SubElement(measure, "attributes")
            etree.SubElement(attributes, "divisions").text = "1"
            time_node = etree.SubElement(attributes, "time")
            etree.SubElement(time_node, "beats").text = "4"
            etree.SubElement(time_node, "beat-type").text = "4"
            clef = etree.SubElement(attributes, "clef")
            etree.SubElement(clef, "sign").text = "G"
            etree.SubElement(clef, "line").text = "2"
        note = etree.SubElement(measure, "note")
        etree.SubElement(note, "rest", measure="yes")
        etree.SubElement(note, "duration").text = "4"
        etree.SubElement(note, "voice").text = "1"
        etree.SubElement(note, "type").text = "whole"
    etree.ElementTree(root).write(str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE)


class FakeRunner:
    def __init__(self, log, page_timeout_seconds=1, settings=None):
        self.log = log

    @staticmethod
    def available():
        return True

    def initialize_models(self, cancel_event=None):
        return EngineResult(0, None, 0.0)

    def accelerator_status(self):
        return probe_accelerator("cpu")

    def run_page(self, image_path: Path, cancel_event=None):
        output = image_path.with_suffix(".musicxml")
        create_fake_musicxml(output)
        return EngineResult(0, output, 0.01)


class FailingRunner(FakeRunner):
    def run_page(self, image_path: Path, cancel_event=None):
        return EngineResult(1, None, 0.01, error="synthetic OMR failure")


def make_scan(path: Path) -> None:
    image = Image.new("RGB", (1200, 1600), "white")
    draw = ImageDraw.Draw(image)
    for y0 in [250, 750, 1250]:
        for line in range(5): draw.line((80, y0 + line * 14, 1120, y0 + line * 14), fill="black", width=2)
        for x in [80, 340, 600, 860, 1120]: draw.line((x, y0, x, y0 + 56), fill="black", width=3)
    image.save(path)


def test_image_job_produces_musicxml_mxl_and_persistent_state(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(jobs_module, "HomrRunner", FakeRunner)
    monkeypatch.setattr(jobs_module, "enrich_musicxml_with_ocr", lambda image, xml, layout=None: ([], []))
    monkeypatch.setattr(jobs_module, "render_preview", lambda xml, output: (None, []))
    image_a = tmp_path / "10.png"; image_b = tmp_path / "2.png"
    make_scan(image_a); make_scan(image_b)

    manager = JobManager(Settings.from_root(tmp_path / "portable"))
    job = manager.create_job([image_b, image_a], ["2.png", "10.png"])
    # A complete two-page smoke run intentionally includes source-geometry
    # audits and can exceed 20 seconds on a machine concurrently preparing
    # training data.  Keep the assertion bounded without making load a failure.
    deadline = time.time() + 60
    while job.status not in {"completed", "failed"} and time.time() < deadline:
        time.sleep(0.05)

    assert job.status == "completed", job.error
    assert Path(job.result_musicxml).exists()
    assert Path(job.result_mxl).exists()
    assert (job.root / "job.json").exists()
    assert job.source_files == ["2.png", "10.png"]
    assert [p.source_name for p in job.pages] == ["2.png", "10.png"]
    assert job.quality_state in {"verified", "review_recommended"}


def test_job_fails_closed_when_no_page_is_recognized(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(jobs_module, "HomrRunner", FailingRunner)
    monkeypatch.setattr(
        jobs_module,
        "enrich_musicxml_with_ocr",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("OCR must not run without MusicXML")
        ),
    )
    image_path = tmp_path / "page.png"
    make_scan(image_path)
    manager = JobManager(Settings.from_root(tmp_path / "portable"))

    job = manager.create_job([image_path], ["page.png"])
    deadline = time.time() + 60
    while job.status not in {"completed", "failed"} and time.time() < deadline:
        time.sleep(0.05)

    assert job.status == "failed"
    assert job.quality_state == "failed"
    assert job.result_musicxml is None
    assert job.pages[0].omr_status == "fallback"


def test_acknowledgement_only_review_preserves_warning_state(tmp_path: Path, monkeypatch) -> None:
    from scorescan.models import ReviewIssue

    monkeypatch.setattr(jobs_module, "HomrRunner", FakeRunner)
    monkeypatch.setattr(jobs_module, "enrich_musicxml_with_ocr", lambda image, xml, layout=None: ([], []))
    monkeypatch.setattr(jobs_module, "render_preview", lambda xml, output: (None, []))
    image_path = tmp_path / "page.png"
    make_scan(image_path)
    manager = JobManager(Settings.from_root(tmp_path / "portable"))
    job = manager.create_job([image_path], ["page.png"])
    deadline = time.time() + 60
    while job.status not in {"completed", "failed"} and time.time() < deadline:
        time.sleep(0.05)
    assert job.status == "completed", job.error

    issue = ReviewIssue(
        id="measure-risk",
        page_index=1,
        category="measure_consensus",
        title="检查小节",
        message="候选分歧",
        writeback_supported=False,
        requires_value=False,
        risk_preserved=True,
    )
    job.review_issues = [issue]
    job.review_path = str(job.root / "result" / "review_issues.json")
    ok, _ = manager.resolve_review_issue(job.id, issue.id, None, False)
    assert ok
    assert issue.status == "resolved"
    assert job.quality_state == "reviewed_with_warnings"



def test_remove_never_orphans_active_job(tmp_path: Path) -> None:
    manager = JobManager(Settings.from_root(tmp_path / "portable"))
    root = manager.workspace / "active-job"
    root.mkdir(parents=True)
    state = JobState("active-job", root, "images", ["page.png"])
    manager.jobs[state.id] = state

    for status in ("queued", "running", "cancelling", "interrupted"):
        state.status = status
        assert not manager.remove(state.id)
        assert manager.get(state.id) is state
        assert root.exists()

    state.status = "completed"
    assert manager.remove(state.id)
    assert manager.get(state.id) is None
    assert not root.exists()


def test_bounded_fifo_scheduler_cancels_queued_job_before_execution(tmp_path: Path) -> None:
    settings = replace(Settings.from_root(tmp_path / "portable"), max_parallel_jobs=1)
    manager = JobManager(settings)
    first_started = threading.Event()
    release_first = threading.Event()
    execution_order: list[str] = []

    def fake_run(job: JobState) -> None:
        execution_order.append(job.id)
        if job.id == "first":
            first_started.set()
            assert release_first.wait(2)
        job.update(status="completed", stage="转换完成", progress=1.0)

    manager._run_job = fake_run  # type: ignore[method-assign]
    states = []
    for job_id in ("first", "second", "third"):
        root = settings.workspace / job_id
        root.mkdir(parents=True, exist_ok=True)
        state = JobState(job_id, root, "images", [f"{job_id}.png"])
        manager.jobs[job_id] = state
        states.append(state)

    manager._start_worker(states[0])
    assert first_started.wait(2)
    manager._start_worker(states[1])
    manager._start_worker(states[2])
    assert manager.describe(states[1], compact=True)["queue_position"] == 1
    assert manager.describe(states[2], compact=True)["queue_position"] == 2
    assert manager.cancel("second")
    release_first.set()

    deadline = time.time() + 3
    while states[2].status != "completed" and time.time() < deadline:
        time.sleep(0.01)
    assert states[0].status == "completed"
    assert states[1].status == "cancelled"
    assert states[2].status == "completed"
    assert execution_order == ["first", "third"]


def test_startup_removes_only_expired_terminal_jobs(tmp_path: Path) -> None:
    root = tmp_path / "portable"
    settings = replace(Settings.from_root(root), job_retention_days=30)
    repository = JobRepository(settings.workspace)
    old_timestamp = (datetime.now(timezone.utc) - timedelta(days=60)).replace(microsecond=0).isoformat()
    recent_timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    old_root = settings.workspace / "old-completed"
    old_root.mkdir(parents=True)
    old = JobState("old-completed", old_root, "images", ["old.png"], status="completed")
    old.updated_at = old_timestamp
    repository.persist(old)

    recent_root = settings.workspace / "recent-completed"
    recent_root.mkdir(parents=True)
    recent = JobState("recent-completed", recent_root, "images", ["recent.png"], status="completed")
    recent.updated_at = recent_timestamp
    repository.persist(recent)

    incoming = settings.workspace / "incoming" / "stale-upload"
    incoming.mkdir(parents=True)
    stale_file = incoming / "page.png"
    stale_file.write_bytes(b"stale")
    old_epoch = (datetime.now(timezone.utc) - timedelta(days=2)).timestamp()
    import os
    os.utime(incoming, (old_epoch, old_epoch))

    manager = JobManager(settings)

    assert manager.get("old-completed") is None
    assert not old_root.exists()
    assert manager.get("recent-completed") is not None
    assert recent_root.exists()
    assert not incoming.exists()


def test_create_job_rejects_storage_exhaustion_before_creating_state(
    tmp_path: Path, monkeypatch,
) -> None:
    from scorescan.storage import StorageCapacityError

    source = tmp_path / "page.png"
    make_scan(source)
    manager = JobManager(Settings.from_root(tmp_path / "portable"))

    def reject(*_args, **_kwargs):
        raise StorageCapacityError("synthetic storage limit")

    monkeypatch.setattr(jobs_module, "require_workspace_capacity", reject)
    import pytest
    with pytest.raises(StorageCapacityError, match="synthetic storage limit"):
        manager.create_job([source], ["page.png"])

    assert manager.list_recent() == []
    assert not [path for path in manager.workspace.iterdir() if path.name != "incoming"]
