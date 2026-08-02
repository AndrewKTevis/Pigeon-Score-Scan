from __future__ import annotations

import hmac
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, Response, jsonify, request, send_file
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.serving import make_server

from .config import APP_NAME, APP_VERSION, Settings
from .jobs import JobManager
from .integrity import verify_bundle_manifest
from .instance_lock import WorkspaceInstanceLock
from .diagnostics import create_diagnostics_bundle
from .self_test import run_system_check
from .storage import StorageCapacityError, require_free_space
from .runtime_metrics import RuntimeMetrics
from .util import path_is_within, safe_filename


def _portable_root() -> Path:
    configured = os.environ.get("SCORESCAN_PORTABLE_ROOT")
    if configured:
        return Path(configured).resolve()
    return Path.cwd().resolve()


def _find_port(start: int = 8765, stop: int = 8795) -> int:
    for port in range(start, stop + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("无法找到可用的本地端口")




def _loopback_hostname(value: str | None) -> bool:
    if not value:
        return False
    try:
        hostname = urlsplit(value if "://" in value else f"//{value}").hostname
    except ValueError:
        return False
    return (hostname or "").casefold() in {"127.0.0.1", "localhost", "::1"}


def _request_token() -> str:
    return request.headers.get("X-ScoreScan-Token", "") or request.args.get("token", "")

def _open_path(path: Path, select: bool = False) -> None:
    if os.name == "nt":
        if select:
            subprocess.Popen(["explorer.exe", "/select,", str(path)])
        else:
            os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        if select:
            subprocess.Popen(["open", "-R", str(path)])
        else:
            subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path.parent if select else path)])


def create_app(root: Path, access_token: str | None = None) -> Flask:
    settings = Settings.from_root(root)
    settings.runtime.mkdir(parents=True, exist_ok=True)
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_bytes
    token = access_token or os.environ.get("SCORESCAN_ACCESS_TOKEN") or secrets.token_urlsafe(32)
    app.config["SCORESCAN_ACCESS_TOKEN"] = token
    manager = JobManager(settings)
    runtime_metrics = RuntimeMetrics(settings.workspace)
    web_dir = Path(__file__).with_name("web")

    @app.before_request
    def protect_local_api():
        if not request.path.startswith("/api/") or request.path == "/api/health":
            return None
        if not _loopback_hostname(request.host):
            return jsonify({"error": "拒绝非本机 API 请求"}), 403
        origin = request.headers.get("Origin")
        if origin and not _loopback_hostname(origin):
            return jsonify({"error": "拒绝跨站 API 请求"}), 403
        supplied = _request_token()
        if not supplied or not hmac.compare_digest(supplied, token):
            return jsonify({"error": "本地会话令牌无效"}), 403
        return None

    @app.after_request
    def local_security_headers(response: Response):
        response.headers["Cache-Control"] = "no-store"
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'",
        )
        return response

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(_: RequestEntityTooLarge):
        return jsonify({"error": "输入文件总大小超过 1 GB"}), 413

    @app.errorhandler(Exception)
    def unhandled_error(exc: Exception):
        # Preserve Flask/Werkzeug status codes.  Treating a normal 404/405 as an
        # internal error makes the local API harder to debug and can hide client bugs.
        if isinstance(exc, HTTPException):
            return jsonify({"error": exc.description}), int(exc.code or 500)
        app.logger.exception("Unhandled ScoreScan error")
        return jsonify({"error": "程序内部错误。详细信息已写入运行日志。"}), 500

    @app.get("/")
    def index() -> Response:
        return send_file(web_dir / "index.html")

    @app.get("/assets/<path:name>")
    def assets(name: str) -> Response:
        path = (web_dir / name).resolve()
        if not path_is_within(path, web_dir):
            return Response(status=404)
        return send_file(path)

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "application": APP_NAME, "version": APP_VERSION})

    @app.get("/api/system-check")
    def system_check():
        result = run_system_check(settings)
        return jsonify(result), (200 if result["ok"] else 503)

    @app.get("/api/runtime")
    def runtime_status():
        return jsonify(runtime_metrics.sample())

    @app.get("/api/diagnostics")
    def diagnostics_bundle():
        path = create_diagnostics_bundle(settings)
        return send_file(path, as_attachment=True, download_name=path.name)

    @app.get("/api/jobs")
    def recent_jobs():
        return jsonify({"jobs": manager.list_recent()})

    @app.post("/api/jobs")
    def create_job_route():
        raw_output_name = request.form.get("output_name", "").strip()
        if len(raw_output_name) > 120:
            return jsonify({"error": "输出文件名不能超过 120 个字符"}), 400
        if raw_output_name.casefold().endswith(".musicxml"):
            raw_output_name = raw_output_name[:-9]
        elif raw_output_name.casefold().endswith(".mxl"):
            raw_output_name = raw_output_name[:-4]
        output_name = safe_filename(Path(raw_output_name).name) if raw_output_name else None
        try:
            pdf_dpi = int(request.form.get("pdf_dpi", str(settings.pdf_dpi)))
        except ValueError:
            return jsonify({"error": "PDF 精度无效"}), 400
        if pdf_dpi not in {300, 400, 500}:
            return jsonify({"error": "PDF 精度必须为 300、400 或 500 DPI"}), 400
        try:
            require_free_space(
                settings.workspace,
                required_bytes=max(0, int(request.content_length or 0)),
                reserve_bytes=settings.minimum_free_space_bytes,
                context="接收上传文件",
            )
        except StorageCapacityError as exc:
            return jsonify({"error": str(exc)}), 507
        files = request.files.getlist("files")
        if not files:
            return jsonify({"error": "没有收到文件"}), 400
        upload_root = settings.workspace / "incoming" / f"upload-{time.time_ns()}"
        upload_root.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        names: list[str] = []
        try:
            for index, file in enumerate(files, start=1):
                safe_name = Path(file.filename or f"file-{index}").name
                path = upload_root / f"{index:04d}_{safe_name}"
                file.save(path)
                saved.append(path)
                names.append(safe_name)
            job = manager.create_job(
                saved,
                names,
                consume_uploads=True,
                output_name=output_name,
                pdf_dpi=pdf_dpi,
            )
            return jsonify(manager.describe(job)), 201
        except StorageCapacityError as exc:
            return jsonify({"error": str(exc)}), 507
        except OSError as exc:
            # ENOSPC and quota failures can still occur between the preflight check
            # and the final write.  Report a capacity error while the finally block
            # removes any partially saved upload.
            if getattr(exc, "errno", None) in {28, 122}:
                return jsonify({"error": "磁盘空间或工作区配额不足，上传未提交。"}), 507
            raise
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        finally:
            for path in saved:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            try:
                upload_root.rmdir()
            except OSError:
                pass

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id: str):
        job = manager.get(job_id)
        if job is None:
            return jsonify({"error": "任务不存在"}), 404
        compact = request.args.get("compact", "0") == "1"
        try:
            after = max(-1, int(request.args.get("after", "-1")))
            wait_seconds = min(20.0, max(0.0, float(request.args.get("wait", "0"))))
        except ValueError:
            return jsonify({"error": "任务状态参数无效"}), 400
        if wait_seconds and after >= 0:
            job.wait_for_revision(after, wait_seconds)
        return jsonify(manager.describe(job, compact=compact, include_logs=True))

    @app.post("/api/jobs/<job_id>/cancel")
    def cancel_job(job_id: str):
        if not manager.cancel(job_id):
            return jsonify({"error": "任务不存在或已经结束"}), 409
        return jsonify({"ok": True})

    def verify_job_bundle(job) -> tuple[bool, list[str]]:
        if not job.artifact_manifest_path:
            # Completed results are never downloadable without a committed bundle
            # manifest.  This prevents a crash between individual file writes from
            # exposing a mixed-revision MusicXML/MXL/report set.
            if job.status == "completed" or any((job.result_musicxml, job.result_mxl, job.report_path)):
                return False, ["输出完整性清单尚未生成"]
            return False, ["结果尚未生成"]
        manifest = Path(job.artifact_manifest_path).resolve()
        output_dir = job.root / "result"
        if not path_is_within(manifest, job.root) or not manifest.exists():
            return False, ["输出完整性清单不存在"]
        return verify_bundle_manifest(output_dir, manifest)

    def result_file(job_id: str, attribute: str):
        job = manager.get(job_id)
        path_value = getattr(job, attribute, None) if job else None
        if job is None or not path_value:
            return None, (jsonify({"error": "结果尚未生成"}), 404)
        valid_bundle, bundle_errors = verify_job_bundle(job)
        if not valid_bundle:
            return None, (jsonify({
                "error": "结果文件完整性检查失败，请重新转换或查看报告",
                "details": bundle_errors,
            }), 409)
        path = Path(path_value).resolve()
        if not path_is_within(path, job.root) or not path.exists():
            return None, (jsonify({"error": "结果文件不存在"}), 404)
        return (job, path), None

    @app.get("/api/jobs/<job_id>/integrity")
    def job_integrity(job_id: str):
        job = manager.get(job_id)
        if job is None:
            return jsonify({"error": "任务不存在"}), 404
        valid, errors = verify_job_bundle(job)
        return jsonify({
            "valid": valid,
            "errors": errors,
            "bundle_id": job.artifact_bundle_id,
            "manifest": bool(job.artifact_manifest_path),
        }), (200 if valid else 409)

    @app.get("/api/jobs/<job_id>/download/musicxml")
    def download_musicxml(job_id: str):
        value, error = result_file(job_id, "result_musicxml")
        if error:
            return error
        job, path = value
        return send_file(path, as_attachment=True, download_name=f"{manager.suggested_download_name(job)}.musicxml")

    @app.get("/api/jobs/<job_id>/download/mxl")
    def download_mxl(job_id: str):
        value, error = result_file(job_id, "result_mxl")
        if error:
            return error
        job, path = value
        return send_file(path, as_attachment=True, download_name=f"{manager.suggested_download_name(job)}.mxl")

    @app.get("/api/jobs/<job_id>/download/report")
    def download_report(job_id: str):
        value, error = result_file(job_id, "report_path")
        if error:
            return error
        _, path = value
        return send_file(path, as_attachment=True, download_name="conversion_report.json")

    def preview_pages(job_id: str):
        value, error = result_file(job_id, "preview_svg")
        if error:
            return None, error
        job, path = value
        if path.suffix.casefold() == ".svg":
            pages = [path]
        else:
            pages = sorted(path.parent.glob("page_*.svg"))
        pages = [page.resolve() for page in pages if page.is_file() and path_is_within(page.resolve(), job.root)]
        if not pages:
            return None, (jsonify({"error": "预览页面不存在"}), 404)
        return pages, None

    @app.get("/api/jobs/<job_id>/preview")
    def preview(job_id: str):
        pages, error = preview_pages(job_id)
        if error:
            return error
        return jsonify({"page_count": len(pages)})

    @app.get("/api/jobs/<job_id>/preview/<int:page_number>")
    def preview_page(job_id: str, page_number: int):
        pages, error = preview_pages(job_id)
        if error:
            return error
        if page_number < 1 or page_number > len(pages):
            return jsonify({"error": "预览页码不存在"}), 404
        return send_file(pages[page_number - 1])

    @app.get("/api/jobs/<job_id>/review")
    def review_issues(job_id: str):
        job = manager.get(job_id)
        if job is None:
            return jsonify({"error": "任务不存在"}), 404
        return jsonify({
            "issues": [item.to_dict() for item in job.review_issues],
            "pending": sum(item.status != "resolved" for item in job.review_issues),
            "resolved": job.review_resolved_count,
        })

    @app.get("/api/jobs/<job_id>/review/<issue_id>/crop")
    def review_crop(job_id: str, issue_id: str):
        job = manager.get(job_id)
        if job is None:
            return jsonify({"error": "任务不存在"}), 404
        issue = next((item for item in job.review_issues if item.id == issue_id), None)
        if issue is None or not issue.crop_path:
            return jsonify({"error": "疑点图片不存在"}), 404
        path = Path(issue.crop_path).resolve()
        if not path_is_within(path, job.root) or not path.exists():
            return jsonify({"error": "疑点图片不存在"}), 404
        return send_file(path)

    @app.post("/api/jobs/<job_id>/review/<issue_id>")
    def resolve_review(job_id: str, issue_id: str):
        payload = request.get_json(silent=True) or {}
        value = payload.get("value")
        ignore = bool(payload.get("ignore", False))
        if value is not None and not isinstance(value, str):
            return jsonify({"error": "文字值格式错误"}), 400
        ok, message = manager.resolve_review_issue(job_id, issue_id, value, ignore)
        if not ok:
            return jsonify({"error": message}), 409
        job = manager.get(job_id)
        return jsonify({
            "ok": True,
            "message": message,
            "job": manager.describe(job) if job else None,
        })

    @app.post("/api/jobs/<job_id>/open/musicxml")
    def open_musicxml(job_id: str):
        value, error = result_file(job_id, "result_musicxml")
        if error:
            return error
        _, path = value
        try:
            _open_path(path)
            return jsonify({"ok": True})
        except Exception as exc:
            return jsonify({"error": f"无法打开文件：{exc}"}), 500

    @app.post("/api/jobs/<job_id>/open/mxl")
    def open_mxl(job_id: str):
        value, error = result_file(job_id, "result_mxl")
        if error:
            return error
        _, path = value
        try:
            _open_path(path)
            return jsonify({"ok": True})
        except Exception as exc:
            return jsonify({"error": f"无法打开文件：{exc}"}), 500

    @app.post("/api/jobs/<job_id>/open/folder")
    def open_result_folder(job_id: str):
        value, error = result_file(job_id, "result_musicxml")
        if error:
            return error
        _, path = value
        try:
            _open_path(path, select=True)
            return jsonify({"ok": True})
        except Exception as exc:
            return jsonify({"error": f"无法打开文件夹：{exc}"}), 500

    shutdown_event = threading.Event()

    @app.post("/api/shutdown")
    def shutdown():
        shutdown_event.set()
        return jsonify({"ok": True})

    app.config["SCORESCAN_SHUTDOWN_EVENT"] = shutdown_event
    app.config["SCORESCAN_JOB_MANAGER"] = manager
    return app


def _run_server_locked(root: Path, settings: Settings) -> None:
    ready_file = settings.runtime / "ready.txt"
    ready_file.unlink(missing_ok=True)

    requested_port = os.environ.get("SCORESCAN_PORT")
    if requested_port:
        try:
            port = int(requested_port)
        except ValueError as exc:
            raise RuntimeError("SCORESCAN_PORT 必须是整数") from exc
        if not 1024 <= port <= 65535:
            raise RuntimeError("SCORESCAN_PORT 必须在 1024–65535 之间")
    else:
        port = _find_port()
    access_token = os.environ.get("SCORESCAN_ACCESS_TOKEN") or secrets.token_urlsafe(32)
    app = create_app(root, access_token=access_token)
    server = make_server("127.0.0.1", port, app, threaded=True)
    url = f"http://127.0.0.1:{port}/?token={access_token}"
    shutdown_event: threading.Event = app.config["SCORESCAN_SHUTDOWN_EVENT"]

    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    ready_file.write_text(f"{url}\n{os.getpid()}\n", encoding="utf-8")

    try:
        desktop_completed = False
        if os.name == "nt" and os.environ.get("SCORESCAN_NO_DESKTOP") != "1":
            try:
                from .desktop import run_desktop

                run_desktop(url, shutdown_event, settings.runtime)
                desktop_completed = True
            except Exception:
                app.logger.exception("ScoreScan desktop shell failed; using browser fallback")
        if not desktop_completed:
            if os.environ.get("SCORESCAN_NO_BROWSER") != "1":
                threading.Timer(0.2, lambda: webbrowser.open(url, new=1)).start()
            while not shutdown_event.wait(0.5):
                if not worker.is_alive():
                    break
    finally:
        shutdown_event.set()
        server.shutdown()
        server.server_close()
        ready_file.unlink(missing_ok=True)


def run_server() -> None:
    root = _portable_root()
    settings = Settings.from_root(root)
    settings.runtime.mkdir(parents=True, exist_ok=True)
    # Acquire before touching ready.txt.  A racing second process must never delete the
    # active instance's discovery file or construct a second JobManager over the same
    # mutable workspace.  The OS lock is released automatically on process failure.
    with WorkspaceInstanceLock(settings.runtime / "server.lock"):
        _run_server_locked(root, settings)
