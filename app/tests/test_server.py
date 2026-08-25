from io import BytesIO
import json
from pathlib import Path
import time

from PIL import Image

from scorescan.server import create_app


AUTH = {"X-ScoreScan-Token": "test-token"}


def png_bytes() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (100, 100), "white").save(stream, "PNG")
    return stream.getvalue()


def test_health_and_mixed_input_rejection(tmp_path: Path) -> None:
    app = create_app(tmp_path, access_token="test-token")
    app.testing = True
    client = app.test_client()
    response = client.get("/api/health")
    assert response.status_code == 200
    from scorescan.config import APP_VERSION
    assert response.json["version"] == APP_VERSION
    response = client.post(
        "/api/jobs",
        headers=AUTH,
        data={"files": [(BytesIO(png_bytes()), "1.png"), (BytesIO(b"%PDF-1.4"), "2.pdf")]},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400


def test_local_api_requires_session_token_and_rejects_cross_site_origin(tmp_path: Path) -> None:
    app = create_app(tmp_path, access_token="test-token")
    app.testing = True
    client = app.test_client()

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.headers["Cache-Control"] == "no-store"
    assert health.headers["X-Frame-Options"] == "DENY"
    assert health.headers["Referrer-Policy"] == "no-referrer"

    assert client.get("/api/jobs").status_code == 403
    assert client.get("/api/jobs", headers={"X-ScoreScan-Token": "wrong"}).status_code == 403
    assert client.get("/api/jobs", headers={**AUTH, "Host": "example.invalid"}).status_code == 403
    assert client.get("/api/jobs?token=test-token").status_code == 200
    assert client.get(
        "/api/jobs",
        headers={**AUTH, "Origin": "https://example.invalid"},
    ).status_code == 403
    assert client.get(
        "/api/jobs",
        headers={**AUTH, "Origin": "http://localhost:8765"},
    ).status_code == 200


def test_download_is_guarded_by_bundle_integrity(tmp_path: Path, monkeypatch) -> None:
    from lxml import etree

    from scorescan.integrity import build_bundle_integrity
    from scorescan.models import JobState
    from scorescan.musicxml import MUSICXML_DOCTYPE, package_mxl
    from scorescan.util import atomic_write_json

    app = create_app(tmp_path, access_token="test-token")
    app.testing = True
    manager = app.config["SCORESCAN_JOB_MANAGER"]
    job_root = tmp_path / "workspace" / "jobs" / "integrity-job"
    output_dir = job_root / "result"
    output_dir.mkdir(parents=True, exist_ok=True)
    musicxml = output_dir / "converted_score.musicxml"
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    etree.SubElement(part, "measure", number="1")
    etree.ElementTree(root).write(str(musicxml), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE)
    mxl = output_dir / "converted_score.mxl"
    package_mxl(musicxml, mxl)
    report = output_dir / "conversion_report.json"
    atomic_write_json(report, {"ok": True})
    preview = output_dir / "preview.svg"
    preview.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="20" height="10"></svg>', encoding="utf-8")
    bundle = build_bundle_integrity(output_dir, [("musicxml", musicxml), ("mxl", mxl), ("report", report), ("preview", preview)])

    job = JobState(
        "integrity-job",
        job_root,
        "images",
        ["page.png"],
        status="completed",
        quality_state="verified",
    )
    job.result_musicxml = str(musicxml)
    job.result_mxl = str(mxl)
    job.report_path = str(report)
    job.preview_svg = str(preview)
    job.artifact_manifest_path = bundle.manifest_path
    job.artifact_bundle_id = bundle.bundle_id
    manager.jobs[job.id] = job

    client = app.test_client()
    assert client.get(f"/api/jobs/{job.id}/integrity", headers=AUTH).status_code == 200
    assert client.get(f"/api/jobs/{job.id}/download/musicxml", headers=AUTH).status_code == 200
    preview_response = client.get(f"/api/jobs/{job.id}/preview", headers=AUTH)
    assert preview_response.status_code == 200
    assert preview_response.json == {"page_count": 1}
    preview_page = client.get(f"/api/jobs/{job.id}/preview/1", headers=AUTH)
    assert preview_page.status_code == 200
    assert preview_page.mimetype == "image/svg+xml"
    job.quality_state = "best_effort"
    # Quality evidence is advisory. A complete, integrity-verified result must remain
    # downloadable even when the automatic checks classify it as best effort.
    assert client.get(f"/api/jobs/{job.id}/download/musicxml", headers=AUTH).status_code == 200
    job.quality_state = "verified"

    def reject_open(*_args, **_kwargs):
        raise RuntimeError("private runtime detail")

    monkeypatch.setattr("scorescan.server._open_path", reject_open)
    expected_open_errors = {
        "musicxml": "无法打开 MusicXML 文件。",
        "mxl": "无法打开 MXL 文件。",
        "folder": "无法打开结果文件夹。",
    }
    for target, expected_error in expected_open_errors.items():
        open_response = client.post(f"/api/jobs/{job.id}/open/{target}", headers=AUTH)
        assert open_response.status_code == 500
        assert open_response.json == {"error": expected_error}
        assert "private runtime detail" not in open_response.get_data(as_text=True)

    musicxml.write_text("corrupted", encoding="utf-8")
    integrity_response = client.get(f"/api/jobs/{job.id}/integrity", headers=AUTH)
    assert integrity_response.status_code == 409
    assert integrity_response.json["errors"] == ["结果文件完整性检查未通过"]
    download_response = client.get(f"/api/jobs/{job.id}/download/musicxml", headers=AUTH)
    assert download_response.status_code == 409
    assert download_response.json == {"error": "结果文件完整性检查失败，请重新转换。"}


def test_http_errors_preserve_status_and_completed_result_requires_manifest(tmp_path: Path) -> None:
    from lxml import etree

    from scorescan.models import JobState
    from scorescan.musicxml import MUSICXML_DOCTYPE

    app = create_app(tmp_path, access_token="test-token")
    app.testing = True
    client = app.test_client()
    missing = client.get('/does-not-exist')
    assert missing.status_code == 404
    assert missing.is_json

    manager = app.config['SCORESCAN_JOB_MANAGER']
    job_root = tmp_path / 'workspace' / 'jobs' / 'no-manifest-job'
    result_dir = job_root / 'result'
    result_dir.mkdir(parents=True, exist_ok=True)
    musicxml = result_dir / 'converted_score.musicxml'
    root = etree.Element('score-partwise', version='4.0')
    part_list = etree.SubElement(root, 'part-list')
    score_part = etree.SubElement(part_list, 'score-part', id='P1')
    etree.SubElement(score_part, 'part-name').text = 'Music'
    etree.SubElement(root, 'part', id='P1')
    etree.ElementTree(root).write(str(musicxml), encoding='UTF-8', xml_declaration=True, doctype=MUSICXML_DOCTYPE)
    job = JobState(
        'no-manifest-job',
        job_root,
        'images',
        ['page.png'],
        status='completed',
        quality_state='verified',
    )
    job.result_musicxml = str(musicxml)
    manager.jobs[job.id] = job
    response = client.get(f'/api/jobs/{job.id}/download/musicxml', headers=AUTH)
    assert response.status_code == 409


def test_system_check_and_diagnostics_endpoints(tmp_path: Path) -> None:
    from scorescan.config import APP_VERSION
    import zipfile

    (tmp_path / "VERSION").write_text(APP_VERSION + "\n", encoding="utf-8")
    app = create_app(tmp_path, access_token="test-token")
    app.testing = True
    client = app.test_client()
    response = client.get("/api/system-check", headers=AUTH)
    assert response.status_code in {200, 503}
    assert response.is_json
    assert response.json["version"] == APP_VERSION

    diagnostics = client.get("/api/diagnostics", headers=AUTH)
    assert diagnostics.status_code == 200
    with zipfile.ZipFile(BytesIO(diagnostics.data)) as archive:
        assert "system_check.json" in archive.namelist()
        assert "recent_jobs_redacted.json" in archive.namelist()


def test_runtime_status_reports_bounded_resources(tmp_path: Path) -> None:
    app = create_app(tmp_path, access_token="test-token")
    app.testing = True

    response = app.test_client().get("/api/runtime", headers=AUTH)

    assert response.status_code == 200
    assert set(response.json) == {
        "system_cpu_percent",
        "memory_percent",
        "memory_total_bytes",
        "memory_available_bytes",
        "process_memory_bytes",
        "workspace_free_bytes",
        "service_uptime_seconds",
    }
    if response.json["system_cpu_percent"] is not None:
        assert 0 <= response.json["system_cpu_percent"] <= 100
    if response.json["memory_percent"] is not None:
        assert 0 <= response.json["memory_percent"] <= 100


def test_runtime_preferences_endpoint_is_not_published(tmp_path: Path) -> None:
    app = create_app(tmp_path, access_token="test-token")
    app.testing = True
    client = app.test_client()

    assert client.get("/api/preferences", headers=AUTH).status_code == 404
    assert client.post("/api/preferences", headers=AUTH, json={}).status_code == 404


def test_job_submission_persists_quick_settings(tmp_path: Path, monkeypatch) -> None:
    from scorescan.models import JobState

    app = create_app(tmp_path, access_token="test-token")
    app.testing = True
    manager = app.config["SCORESCAN_JOB_MANAGER"]
    captured = {}

    def create_job(paths, names, **options):
        captured.update(options)
        return JobState(
            "settings-job",
            tmp_path / "workspace" / "settings-job",
            "images",
            names,
            output_name=options.get("output_name"),
            pdf_dpi=options.get("pdf_dpi", 400),
        )

    monkeypatch.setattr(manager, "create_job", create_job)
    response = app.test_client().post(
        "/api/jobs",
        headers=AUTH,
        data={
            "files": [(BytesIO(png_bytes()), "1.png")],
            "output_name": "My Score.mxl",
            "pdf_dpi": "500",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    assert captured["consume_uploads"] is True
    assert captured["output_name"] == "My Score"
    assert captured["pdf_dpi"] == 500
    assert response.json["output_name"] == "My Score"
    assert response.json["pdf_dpi"] == 500


def test_job_submission_rejects_unknown_pdf_detail(tmp_path: Path) -> None:
    app = create_app(tmp_path, access_token="test-token")
    app.testing = True
    response = app.test_client().post(
        "/api/jobs",
        headers=AUTH,
        data={"files": [(BytesIO(png_bytes()), "1.png")], "pdf_dpi": "600"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400


def test_upload_preflight_reports_insufficient_storage(tmp_path: Path, monkeypatch) -> None:
    from scorescan.storage import StorageCapacityError

    app = create_app(tmp_path, access_token="test-token")
    app.testing = True
    client = app.test_client()

    def reject(*_args, **_kwargs):
        raise StorageCapacityError("synthetic disk reserve")

    monkeypatch.setattr("scorescan.server.require_free_space", reject)
    response = client.post(
        "/api/jobs",
        headers=AUTH,
        data={"files": [(BytesIO(png_bytes()), "1.png")]},
        content_type="multipart/form-data",
    )
    assert response.status_code == 507
    assert response.json["error"] == "可用磁盘空间不足，请释放空间后重试。"
    assert "synthetic disk reserve" not in response.get_data(as_text=True)


def test_job_submission_redacts_manager_exceptions(tmp_path: Path, monkeypatch) -> None:
    from scorescan.storage import StorageCapacityError

    app = create_app(tmp_path, access_token="test-token")
    app.testing = True
    client = app.test_client()
    manager = app.config["SCORESCAN_JOB_MANAGER"]

    def submit_with(exception: Exception):
        def reject(*_args, **_kwargs):
            raise exception

        monkeypatch.setattr(manager, "create_job", reject)
        return client.post(
            "/api/jobs",
            headers=AUTH,
            data={"files": [(BytesIO(png_bytes()), "1.png")]},
            content_type="multipart/form-data",
        )

    capacity = submit_with(StorageCapacityError("private workspace path"))
    assert capacity.status_code == 507
    assert capacity.json["error"] == (
        "可用磁盘空间或工作区容量不足，请清理旧任务后重试。"
    )
    assert "private workspace path" not in capacity.get_data(as_text=True)

    invalid = submit_with(ValueError("private parser detail"))
    assert invalid.status_code == 400
    assert invalid.json["error"] == "输入文件或转换设置无效。"
    assert "private parser detail" not in invalid.get_data(as_text=True)


def test_compact_job_status_omits_heavy_page_audits(tmp_path: Path) -> None:
    from scorescan.models import JobState, PageInfo

    app = create_app(tmp_path, access_token="test-token")
    app.testing = True
    manager = app.config["SCORESCAN_JOB_MANAGER"]
    root = tmp_path / "workspace" / "compact-job"
    root.mkdir(parents=True)
    job = JobState("compact-job", root, "images", ["page.png"], status="running")
    job.pages = [PageInfo(
        1,
        "page.png",
        str(root / "page.png"),
        recognition_candidates=[{"large": "x" * 1000} for _ in range(100)],
    )]
    job.update(stage="识别", progress=0.5)
    manager.jobs[job.id] = job

    response = app.test_client().get(
        f"/api/jobs/{job.id}?compact=1&after={job.revision}&wait=0.01",
        headers=AUTH,
    )

    assert response.status_code == 200
    assert response.json["pages"] == []
    assert response.json["page_summary"] == [
        {"index": 1, "omr_status": "pending", "quality_score": None}
    ]
    assert response.json["revision"] == job.revision
    full = app.test_client().get(f"/api/jobs/{job.id}", headers=AUTH)
    assert len(response.data) * 10 < len(full.data)
