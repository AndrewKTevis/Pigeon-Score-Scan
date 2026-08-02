from __future__ import annotations

import html
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid


PREVIEW_TIMEOUT_SECONDS = 120


def _configure_toolkit(verovio):
    module_path = getattr(verovio, "__file__", None)
    resource_path = Path(module_path).resolve().with_name("data") if module_path else None
    if resource_path is not None and resource_path.is_dir():
        set_default = getattr(verovio, "setDefaultResourcePath", None)
        if callable(set_default):
            set_default(str(resource_path))
    toolkit = verovio.toolkit()
    if resource_path is not None and resource_path.is_dir():
        try:
            if Path(toolkit.getResourcePath()).resolve() != resource_path:
                toolkit.setResourcePath(str(resource_path))
        except Exception:
            # loadFile/loadData below remains the authoritative check and supplies
            # Verovio's own diagnostic log if resource initialization still fails.
            pass
    toolkit.setOptions(
        {
            "inputFrom": "musicxml",
            "adjustPageHeight": False,
            "breaks": "encoded",
            "footer": "none",
            "header": "none",
            "pageWidth": 1680,
            "pageHeight": 2376,
            "scale": 38,
        }
    )
    return toolkit


_PREVIEW_LOCK = threading.RLock()


def _toolkit_log(toolkit) -> str:
    try:
        return " ".join(str(toolkit.getLog()).split())[:500]
    except Exception:
        return ""


def render_preview(musicxml_path: Path, output_dir: Path) -> tuple[Path | None, list[str]]:
    """Render in a disposable process so a native Verovio fault cannot kill the app."""

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f".preview-worker-{uuid.uuid4().hex}.json"
    worker_environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1])
    inherited_pythonpath = worker_environment.get("PYTHONPATH", "")
    pythonpath_entries = [
        source_root,
        *(
            entry
            for entry in inherited_pythonpath.split(os.pathsep)
            if entry and entry != source_root
        ),
    ]
    worker_environment["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    with _PREVIEW_LOCK:
        try:
            completed = subprocess.run(
                [
                    str(Path(sys.executable).resolve()),
                    "-m",
                    "scorescan.preview_worker",
                    str(musicxml_path.resolve()),
                    str(output_dir.resolve()),
                    str(result_path.resolve()),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=PREVIEW_TIMEOUT_SECONDS,
                env=worker_environment,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                ),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None, [
                f"乐谱预览隔离进程超过 {PREVIEW_TIMEOUT_SECONDS} 秒，已安全跳过预览"
            ]
        except Exception as exc:
            return None, [f"无法启动乐谱预览隔离进程：{exc}"]
        try:
            if completed.returncode != 0:
                detail = " ".join((completed.stdout or "").split())[-400:]
                suffix = f"：{detail}" if detail else ""
                return None, [
                    "乐谱预览隔离进程异常退出"
                    f"（代码 {completed.returncode}），转换结果未受影响{suffix}"
                ]
            if not result_path.is_file():
                return None, ["乐谱预览隔离进程未返回结果，已安全跳过预览"]
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            warnings = [
                str(item)
                for item in payload.get("warnings", [])
                if str(item).strip()
            ]
            raw_preview = payload.get("preview_path")
            preview_path = Path(str(raw_preview)) if raw_preview else None
            if preview_path is not None and not preview_path.is_file():
                warnings.append("乐谱预览隔离进程返回的预览文件不存在")
                preview_path = None
            return preview_path, warnings
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return None, [f"乐谱预览隔离进程结果无效：{exc}"]
        finally:
            result_path.unlink(missing_ok=True)


def _render_preview(musicxml_path: Path, output_dir: Path) -> tuple[Path | None, list[str]]:
    warnings: list[str] = []
    try:
        import verovio  # type: ignore
    except Exception:
        return None, ["未安装 Verovio，已跳过乐谱预览渲染"]

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        toolkit = _configure_toolkit(verovio)
        if not toolkit.loadFile(str(musicxml_path)):
            # Python Verovio's file loader can fail transiently on Windows paths.
            # Retrying through an in-memory document also avoids path/encoding
            # differences while preserving the exact bytes already validated.
            first_log = _toolkit_log(toolkit)
            toolkit = _configure_toolkit(verovio)
            if not toolkit.loadData(musicxml_path.read_text(encoding="utf-8")):
                detail = _toolkit_log(toolkit) or first_log
                suffix = f"：{detail}" if detail else ""
                return None, [f"Verovio 无法载入生成的 MusicXML{suffix}"]
        page_count = int(toolkit.getPageCount())
        if page_count <= 0:
            detail = _toolkit_log(toolkit)
            suffix = f"：{detail}" if detail else ""
            return None, [f"Verovio 未生成可预览页面{suffix}"]
        svg_paths: list[Path] = []
        for page_number in range(1, page_count + 1):
            path = output_dir / f"page_{page_number:04d}.svg"
            svg = toolkit.renderToSVG(page_number)
            if "<svg" not in svg:
                return None, [f"Verovio 第 {page_number} 页未生成有效 SVG"]
            path.write_text(svg, encoding="utf-8")
            svg_paths.append(path)
        html_path = output_dir / "preview.html"
        sections = "\n".join(
            f'<section><img src="{html.escape(path.name)}" alt="第 {index} 页"></section>'
            for index, path in enumerate(svg_paths, start=1)
        )
        html_path.write_text(
            "<!doctype html><meta charset='utf-8'><title>Pigeon Score Scan Preview</title>"
            "<style>body{margin:0;background:#ddd;font-family:sans-serif}section{margin:20px auto;"
            "max-width:1100px;background:white;box-shadow:0 2px 14px #777}img{display:block;width:100%}</style>"
            + sections,
            encoding="utf-8",
        )
        return html_path, warnings
    except Exception as exc:
        warnings.append(f"乐谱预览渲染失败：{exc}")
        return None, warnings
