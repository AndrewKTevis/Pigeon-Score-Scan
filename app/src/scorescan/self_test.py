from __future__ import annotations

"""Deterministic, privacy-safe runtime diagnostics for public releases."""

import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .accelerator import probe_accelerator
from .config import APP_NAME, APP_VERSION, WORKFLOW_VERSION, Settings
from .model_registry import audit_model_manifest, load_verified_json
from .policy import DEFAULT_POLICY
from .preview import _configure_toolkit, _toolkit_log
from .semantic_detector import semantic_detector_status
from .state_schema import CURRENT_JOB_SCHEMA
from .util import read_json, sha256_file


@dataclass(frozen=True)
class CheckResult:
    key: str
    ok: bool
    critical: bool
    message: str
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _module_check(module_name: str, *, critical: bool = True) -> CheckResult:
    try:
        importlib.import_module(module_name)
        distribution_names = {"cv2": "opencv-python-headless", "PIL": "Pillow", "fitz": "PyMuPDF"}
        try:
            version = importlib.metadata.version(distribution_names.get(module_name, module_name))
        except importlib.metadata.PackageNotFoundError:
            version = None
        return CheckResult(
            f"module:{module_name}",
            True,
            critical,
            f"{module_name} 可用",
            {"version": str(version)} if version is not None else None,
        )
    except Exception as exc:  # pragma: no cover - exact dependency errors vary by OS
        return CheckResult(
            f"module:{module_name}",
            False,
            critical,
            f"{module_name} 无法载入",
            {"error": f"{type(exc).__name__}: {exc}"},
        )


def _writable_directory(path: Path, key: str) -> CheckResult:
    try:
        path.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix="scorescan-check-", dir=path)
        os.close(fd)
        Path(temporary).unlink(missing_ok=True)
        return CheckResult(key, True, True, f"{path} 可写")
    except OSError as exc:
        return CheckResult(key, False, True, f"{path} 不可写", {"error": str(exc)})


def _model_checks(resources: Path) -> list[CheckResult]:
    audit = audit_model_manifest(resources)
    results = [
        CheckResult(
            "models:manifest",
            audit.verified,
            True,
            (
                f"模型资源清单完整：{audit.verified_count}/{audit.expected_count}"
                if audit.verified
                else f"模型资源清单不完整：{audit.verified_count}/{audit.expected_count}"
            ),
            {
                "manifest_count": audit.manifest_count,
                "errors": list(audit.errors),
            },
        )
    ]
    for filename, status in sorted(audit.statuses.items()):
        results.append(
            CheckResult(
                f"model:{filename}",
                status == "verified",
                True,
                f"{filename}: {status}",
            )
        )
    return results


def _verovio_smoke_check() -> CheckResult:
    try:
        import verovio  # type: ignore

        xml = """<?xml version='1.0' encoding='UTF-8'?>
<score-partwise version='4.0'><part-list><score-part id='P1'><part-name>Music</part-name></score-part></part-list>
<part id='P1'><measure number='1'><attributes><divisions>1</divisions><key><fifths>0</fifths></key>
<time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>
<note><rest/><duration>4</duration><voice>1</voice><type>whole</type></note></measure></part></score-partwise>"""
        # Use the same explicit packaged-font initialization as the real preview
        # path.  A bare toolkit() can appear to work in a developer checkout
        # while failing in a freshly installed portable environment whose current
        # directory does not happen to contain Verovio's data directory.
        toolkit = _configure_toolkit(verovio)
        loaded = bool(toolkit.loadData(xml))
        rendered = toolkit.renderToSVG(1) if loaded else ""
        ok = loaded and "<svg" in rendered
        details = {"version": str(getattr(verovio, "__version__", "unknown"))}
        if not ok:
            log = _toolkit_log(toolkit)
            if log:
                details["log"] = log
        return CheckResult(
            "render:verovio",
            ok,
            True,
            "Verovio MusicXML 渲染通过" if ok else "Verovio 无法完成最小渲染",
            details,
        )
    except Exception as exc:
        return CheckResult(
            "render:verovio",
            False,
            True,
            "Verovio 自检失败",
            {"error": f"{type(exc).__name__}: {exc}"},
        )



def _bootstrap_integrity(settings: Settings) -> list[CheckResult]:
    uv_path = settings.runtime / "uv.exe"
    hash_path = settings.runtime / "uv.sha256"
    if not uv_path.exists() and not hash_path.exists():
        # Source checkouts do not bundle the Windows bootstrap binary.
        return [CheckResult("bootstrap:uv", True, False, "源码环境未包含 Windows uv.exe，跳过启动器哈希检查")]
    if not uv_path.is_file() or not hash_path.is_file():
        return [CheckResult("bootstrap:uv", False, True, "Windows 启动依赖或哈希文件缺失")]
    try:
        expected = hash_path.read_text(encoding="ascii").strip().casefold()
        actual = sha256_file(uv_path).casefold()
    except OSError as exc:
        return [CheckResult("bootstrap:uv", False, True, "无法验证 uv.exe", {"error": str(exc)})]
    ok = len(expected) == 64 and expected == actual
    return [
        CheckResult(
            "bootstrap:uv",
            ok,
            True,
            "uv.exe SHA-256 验证通过" if ok else "uv.exe SHA-256 不匹配",
            {"expected": expected, "actual": actual},
        )
    ]


def _accelerator_checks() -> list[CheckResult]:
    status = probe_accelerator("cpu")
    package_ok = status.cpu_package_installed and not status.package_conflict
    runtime_ok = status.probe_error is None
    return [
        CheckResult(
            "accelerator:onnxruntime-packages",
            package_ok,
            True,
            "CPU 识别组件已安装"
            if package_ok
            else "CPU 识别组件缺失或冲突",
            status.to_dict(),
        ),
        CheckResult(
            "accelerator:runtime",
            runtime_ok,
            True,
            "CPU 识别环境正常" if runtime_ok else "CPU 识别环境无法载入",
            status.to_dict(),
        ),
    ]


def run_system_check(settings: Settings) -> dict[str, Any]:
    checks: list[CheckResult] = []
    checks.append(_writable_directory(settings.workspace, "filesystem:workspace"))
    checks.append(_writable_directory(settings.runtime, "filesystem:runtime"))
    try:
        usage = shutil.disk_usage(settings.root)
        free_gib = usage.free / (1024**3)
        checks.append(
            CheckResult(
                "filesystem:free-space",
                free_gib >= 1.0,
                True,
                f"可用磁盘空间 {free_gib:.2f} GiB",
                {"free_bytes": usage.free},
            )
        )
    except OSError as exc:
        checks.append(CheckResult("filesystem:free-space", False, True, "无法读取磁盘空间", {"error": str(exc)}))

    for name in ("cv2", "numpy", "lxml", "PIL", "flask", "fitz"):
        checks.append(_module_check(name))
    # homr can be relatively heavy, but a public build must at least import its entrypoint.
    checks.append(_module_check("homr"))
    checks.extend(_accelerator_checks())
    checks.append(_verovio_smoke_check())
    checks.extend(_model_checks(settings.resources))
    positioned_detector = semantic_detector_status(settings.resources)
    checks.append(
        CheckResult(
            "models:semantic-detector",
            positioned_detector.enabled,
            False,
            (
                "语义记号检测器已通过发布门禁"
                if positioned_detector.enabled
                else f"语义记号检测器未启用：{positioned_detector.status}"
            ),
            positioned_detector.to_dict(),
        )
    )
    checks.extend(_bootstrap_integrity(settings))

    version_file = settings.root / "VERSION"
    version_text = version_file.read_text(encoding="utf-8").strip() if version_file.exists() else ""
    checks.append(
        CheckResult(
            "release:version",
            version_text == APP_VERSION,
            True,
            f"版本文件 {version_text or '缺失'}；运行版本 {APP_VERSION}",
        )
    )

    failed_critical = [item for item in checks if item.critical and not item.ok]
    return {
        "application": APP_NAME,
        "version": APP_VERSION,
        "workflow_version": WORKFLOW_VERSION,
        "policy_version": DEFAULT_POLICY.version,
        "job_schema": CURRENT_JOB_SCHEMA,
        "ok": not failed_critical,
        "critical_failures": len(failed_critical),
        "checks": [item.to_dict() for item in checks],
        "environment": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "runtime_profile": os.getenv("SCORESCAN_RUNTIME_PROFILE", "development"),
        },
        "privacy": "自检不读取或上传用户扫描内容",
    }


def dump_system_check(settings: Settings, *, pretty: bool = True) -> str:
    return json.dumps(run_system_check(settings), ensure_ascii=False, indent=2 if pretty else None) + "\n"
