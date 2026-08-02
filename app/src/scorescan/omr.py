from __future__ import annotations

import importlib.util
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .accelerator import AcceleratorStatus, probe_accelerator
from .config import Settings
from .cpu_runtime import configure_cpu_environment
from .engine_cache import ENGINE_PROFILE, EngineCacheKey, EngineResultCache
from .homr_worker import RAW_TUPLET_FLAG
from .process_control import popen_group_options, terminate_process_tree
from .util import atomic_write_json, read_json


_MODEL_INITIALIZATION_LOCK = threading.RLock()
_MODEL_INITIALIZED_PROFILES: set[tuple[str, str]] = set()


@dataclass(frozen=True)
class EngineResult:
    return_code: int
    xml_path: Path | None
    elapsed_seconds: float
    cancelled: bool = False
    timed_out: bool = False
    error: str | None = None


@dataclass(frozen=True)
class OcrWorkerResult:
    return_code: int
    marks: tuple[dict[str, object], ...]
    warnings: tuple[str, ...]
    elapsed_seconds: float
    requested_accelerator: str
    selected_accelerator: str | None = None
    runtime_verified: bool = False
    component_providers: dict[str, list[str]] | None = None
    fallback_reason: str | None = None
    cancelled: bool = False
    timed_out: bool = False
    error: str | None = None


@dataclass(frozen=True)
class SemanticWorkerResult:
    return_code: int
    detections: tuple[dict[str, object], ...]
    elapsed_seconds: float
    requested_accelerator: str
    selected_accelerator: str | None = None
    runtime_verified: bool = False
    providers: tuple[str, ...] = ()
    model_enabled: bool = False
    model_status: str = "unavailable"
    model_version: str | None = None
    scale: float = 1.0
    tile_count: int = 0
    fallback_reason: str | None = None
    cancelled: bool = False
    timed_out: bool = False
    error: str | None = None


class HomrRunner:
    """Crash-contained, page-by-page wrapper around homr.

    Each page is processed in a fresh subprocess. This costs a little startup time,
    but prevents one malformed page or native inference failure from terminating an
    entire multi-page conversion and enables page-level checkpoints.
    """

    def __init__(
        self,
        log: Callable[[str], None],
        page_timeout_seconds: int = 45 * 60,
        max_pending_log_lines: int = 2048,
        max_log_line_chars: int = 8192,
        settings: Settings | None = None,
    ) -> None:
        self.log = log
        self.page_timeout_seconds = page_timeout_seconds
        self.max_pending_log_lines = max(8, int(max_pending_log_lines))
        self.max_log_line_chars = max(256, int(max_log_line_chars))
        self.settings = settings
        self._active_process: subprocess.Popen[str] | None = None
        self._process_lock = threading.RLock()
        self._accelerator_status: AcceleratorStatus | None = None
        self._worker_python = Path(sys.executable).resolve()
        self._profile_prepared = False

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("homr") is not None

    @staticmethod
    def expected_xml(image_path: Path) -> Path:
        return image_path.with_suffix(".musicxml")

    def cancel(self) -> None:
        with self._process_lock:
            process = self._active_process
            if process is not None and process.poll() is None:
                terminate_process_tree(process)

    def accelerator_status(self, *, refresh: bool = False) -> AcceleratorStatus:
        self._prepare_worker_profile()
        if self._accelerator_status is None:
            self._accelerator_status = probe_accelerator("cpu")
        return self._accelerator_status

    def _prepare_worker_profile(self) -> None:
        if self._profile_prepared:
            return
        self._profile_prepared = True
        self._worker_python = Path(sys.executable).resolve()
        self._accelerator_status = probe_accelerator("cpu")
        self.log("识别运行环境已就绪")

    def _homr_command(self, *arguments: str, force_cpu: bool = False) -> list[str]:
        return [
            str(self._worker_python),
            "-m",
            "scorescan.homr_worker",
            *arguments,
            "--gpu",
            "no",
        ]

    def _run_with_cpu_fallback(
        self,
        arguments: list[str],
        cwd: Path | None,
        cancel_event: threading.Event | None,
        timeout_seconds: int,
    ) -> EngineResult:
        return self._run_command(
            self._homr_command(*arguments),
            cwd,
            cancel_event,
            timeout_seconds,
        )

    def _environment(self) -> dict[str, str]:
        env = configure_cpu_environment()
        env.setdefault("PYTHONUTF8", "1")
        package_source = str(Path(__file__).resolve().parents[1])
        inherited_pythonpath = env.get("PYTHONPATH", "")
        pythonpath_entries = [
            item
            for item in inherited_pythonpath.split(os.pathsep)
            if item and Path(item).resolve() != Path(package_source)
        ]
        env["PYTHONPATH"] = os.pathsep.join([package_source, *pythonpath_entries])
        return env

    def initialize_models(self, cancel_event: threading.Event | None = None) -> EngineResult:
        self._prepare_worker_profile()
        if not self.available():
            return EngineResult(127, None, 0.0, error="homr 未安装")
        profile_key = (
            str(self._worker_python),
            self.accelerator_status().selected,
        )
        with _MODEL_INITIALIZATION_LOCK:
            if profile_key in _MODEL_INITIALIZED_PROFILES:
                self.log("识别模型已就绪")
                return EngineResult(0, None, 0.0)
            if cancel_event is not None and cancel_event.is_set():
                return EngineResult(130, None, 0.0, cancelled=True)
            result = self._run_with_cpu_fallback(
                ["--init"],
                None,
                cancel_event,
                max(self.page_timeout_seconds, 60 * 60),
            )
            if result.return_code == 0 and not result.cancelled and not result.timed_out:
                _MODEL_INITIALIZED_PROFILES.add(profile_key)
            return result

    def run_page(
        self,
        image_path: Path,
        cancel_event: threading.Event | None = None,
        *,
        preserve_raw_tuplets: bool = False,
    ) -> EngineResult:
        self._prepare_worker_profile()
        image_path = image_path.resolve()
        xml_path = self.expected_xml(image_path)
        if not self.available():
            return EngineResult(127, None, 0.0, error="homr 未安装")
        cache = EngineResultCache(image_path, xml_path)
        profile = (
            f"{ENGINE_PROFILE}+raw-tuplets"
            if preserve_raw_tuplets
            else ENGINE_PROFILE
        )
        key = EngineCacheKey.for_image(image_path, profile=profile)
        if cache.is_valid(key):
            self.log(f"复用已校验的识别缓存：{image_path.name}")
            return EngineResult(0, xml_path, 0.0)
        if xml_path.exists() or cache.manifest_path.exists():
            cache.invalidate()
            self.log(f"识别输入或引擎版本发生变化，已丢弃旧缓存：{image_path.name}")
        arguments = [str(image_path), "--cache", "--output-large-page"]
        if preserve_raw_tuplets:
            arguments.append(RAW_TUPLET_FLAG)
        result = self._run_with_cpu_fallback(
            arguments,
            image_path.parent,
            cancel_event,
            self.page_timeout_seconds,
        )
        if result.return_code == 0:
            try:
                output_size = xml_path.stat().st_size
            except OSError:
                output_size = 0
            if output_size <= 300:
                if xml_path.exists():
                    cache.invalidate()
                else:
                    cache.manifest_path.unlink(missing_ok=True)
                message = "识别引擎未生成足够大小的 MusicXML"
                self.log(message)
                return EngineResult(65, None, result.elapsed_seconds, error=message)
            try:
                cache.commit(key)
            except ValueError as exc:
                cache.invalidate()
                self.log(f"识别引擎输出结构无效，已隔离该结果：{exc}")
                return EngineResult(65, None, result.elapsed_seconds, error=str(exc))
            except OSError as exc:
                self.log(f"识别缓存清单写入失败，将在下次重新验证：{exc}")
            return EngineResult(0, xml_path, result.elapsed_seconds)
        cache.manifest_path.unlink(missing_ok=True)
        return EngineResult(
            result.return_code,
            xml_path if xml_path.exists() else None,
            result.elapsed_seconds,
            result.cancelled,
            result.timed_out,
            result.error or ("识别引擎未生成 MusicXML" if not xml_path.exists() else None),
        )

    def run_ocr_enrichment(
        self,
        image_path: Path,
        xml_path: Path,
        layout: object,
        cancel_event: threading.Event | None = None,
        semantic_text_regions: tuple[dict[str, object], ...] | None = None,
    ) -> OcrWorkerResult:
        """Run OCR enrichment in the isolated CPU worker."""

        self._prepare_worker_profile()
        requested = "cpu"
        layout_to_dict = getattr(layout, "to_dict", None)
        if not callable(layout_to_dict):
            return OcrWorkerResult(
                64,
                (),
                (),
                0.0,
                requested,
                error="OCR worker received an invalid page layout",
            )
        image_path = image_path.resolve()
        xml_path = xml_path.resolve()
        timeout_seconds = max(180, min(self.page_timeout_seconds, 15 * 60))
        elapsed = 0.0
        fallback_reason: str | None = None

        with tempfile.TemporaryDirectory(
            prefix="scorescan-ocr-request-",
            dir=xml_path.parent,
        ) as temp_dir:
            request_path = Path(temp_dir) / "request.json"
            response_path = request_path.with_suffix(".result.json")
            atomic_write_json(
                request_path,
                {
                    "schema_version": 1,
                    "image_path": str(image_path),
                    "xml_path": str(xml_path),
                    "layout": layout_to_dict(),
                    "semantic_text_regions": list(
                        semantic_text_regions or ()
                    ),
                },
            )
            attempts = ("cpu",)
            last_error: str | None = None
            last_result: EngineResult | None = None
            for attempt_index, accelerator in enumerate(attempts):
                response_path.unlink(missing_ok=True)
                result = self._run_command(
                    [
                        str(self._worker_python),
                        "-m",
                        "scorescan.homr_worker",
                        "--scorescan-ocr-request",
                        str(request_path),
                        "--scorescan-ocr-accelerator",
                        accelerator,
                    ],
                    xml_path.parent,
                    cancel_event,
                    timeout_seconds,
                )
                last_result = result
                elapsed += result.elapsed_seconds
                payload = read_json(response_path, {})
                worker_error = (
                    str(payload.get("error"))
                    if isinstance(payload, dict) and payload.get("error")
                    else result.error
                )
                if (
                    result.return_code == 0
                    and isinstance(payload, dict)
                    and payload.get("ok") is True
                ):
                    runtime = payload.get("runtime")
                    runtime = runtime if isinstance(runtime, dict) else {}
                    selected = str(runtime.get("selected") or "")
                    verified = bool(runtime.get("verified", False))
                    providers_payload = runtime.get("component_providers")
                    providers = (
                        {
                            str(name): [str(value) for value in values]
                            for name, values in providers_payload.items()
                            if isinstance(values, list)
                        }
                        if isinstance(providers_payload, dict)
                        else {}
                    )
                    if selected != accelerator or not verified:
                        worker_error = (
                            f"OCR worker provider verification mismatch: "
                            f"requested={accelerator}, selected={selected or 'unknown'}"
                        )
                    else:
                        raw_marks = payload.get("marks")
                        marks = tuple(
                            dict(item)
                            for item in raw_marks
                            if isinstance(item, dict)
                        ) if isinstance(raw_marks, list) else ()
                        raw_warnings = payload.get("warnings")
                        warnings = tuple(
                            str(item) for item in raw_warnings
                        ) if isinstance(raw_warnings, list) else ()
                        return OcrWorkerResult(
                            0,
                            marks,
                            warnings,
                            elapsed,
                            requested,
                            selected_accelerator=selected,
                            runtime_verified=True,
                            component_providers=providers,
                            fallback_reason=fallback_reason,
                        )
                last_error = worker_error or "OCR worker did not return a valid result"
                if result.cancelled or result.timed_out:
                    break

        return OcrWorkerResult(
            last_result.return_code if last_result is not None else 70,
            (),
            (),
            elapsed,
            requested,
            fallback_reason=fallback_reason,
            cancelled=bool(last_result and last_result.cancelled),
            timed_out=bool(last_result and last_result.timed_out),
            error=last_error,
        )

    def run_semantic_detection(
        self,
        image_path: Path,
        layout: object,
        cancel_event: threading.Event | None = None,
    ) -> SemanticWorkerResult:
        """Run the optional release-gated symbol verifier in the CPU worker."""

        self._prepare_worker_profile()
        requested = "cpu"
        layout_to_dict = getattr(layout, "to_dict", None)
        if not callable(layout_to_dict):
            return SemanticWorkerResult(
                64,
                (),
                0.0,
                requested,
                error="semantic detector worker received an invalid page layout",
            )
        image_path = image_path.resolve()
        timeout_seconds = max(180, min(self.page_timeout_seconds, 15 * 60))
        elapsed = 0.0
        fallback_reason: str | None = None
        with tempfile.TemporaryDirectory(
            prefix="scorescan-semantic-request-",
            dir=image_path.parent,
        ) as temp_dir:
            request_path = Path(temp_dir) / "request.json"
            response_path = request_path.with_suffix(".result.json")
            atomic_write_json(
                request_path,
                {
                    "schema_version": 1,
                    "image_path": str(image_path),
                    "layout": layout_to_dict(),
                },
            )
            attempts = ("cpu",)
            last_result: EngineResult | None = None
            last_error: str | None = None
            for attempt_index, accelerator in enumerate(attempts):
                response_path.unlink(missing_ok=True)
                result = self._run_command(
                    [
                        str(self._worker_python),
                        "-m",
                        "scorescan.homr_worker",
                        "--scorescan-semantic-request",
                        str(request_path),
                        "--scorescan-semantic-accelerator",
                        accelerator,
                    ],
                    image_path.parent,
                    cancel_event,
                    timeout_seconds,
                )
                last_result = result
                elapsed += result.elapsed_seconds
                payload = read_json(response_path, {})
                worker_error = (
                    str(payload.get("error"))
                    if isinstance(payload, dict) and payload.get("error")
                    else result.error
                )
                raw = payload.get("result") if isinstance(payload, dict) else None
                if result.return_code == 0 and isinstance(raw, dict):
                    detector_status = raw.get("status")
                    detector_status = (
                        detector_status if isinstance(detector_status, dict) else {}
                    )
                    enabled = bool(detector_status.get("enabled", False))
                    selected = (
                        str(detector_status.get("selected_accelerator") or "")
                        or None
                    )
                    providers_raw = detector_status.get("providers")
                    providers = (
                        tuple(str(value) for value in providers_raw)
                        if isinstance(providers_raw, list)
                        else ()
                    )
                    verified = bool(
                        enabled
                        and selected == accelerator
                        and providers
                        and providers[0] == "CPUExecutionProvider"
                    )
                    # An intentionally absent or unauthorized optional asset is a
                    # valid no-op, not a reason to rerun the page on CPU.
                    if not enabled:
                        verified = False
                    elif not verified:
                        worker_error = (
                            "semantic detector provider verification mismatch: "
                            f"requested={accelerator}, selected={selected}, "
                            f"providers={list(providers)}"
                        )
                    else:
                        detections_raw = raw.get("detections")
                        detections = (
                            tuple(
                                dict(item)
                                for item in detections_raw
                                if isinstance(item, dict)
                            )
                            if isinstance(detections_raw, list)
                            else ()
                        )
                        return SemanticWorkerResult(
                            0,
                            detections,
                            elapsed,
                            requested,
                            selected,
                            True,
                            providers,
                            True,
                            str(detector_status.get("status") or "verified"),
                            (
                                str(detector_status.get("model_version"))
                                if detector_status.get("model_version")
                                else None
                            ),
                            float(raw.get("scale", 1.0)),
                            int(raw.get("tile_count", 0)),
                            fallback_reason,
                        )
                    if not enabled:
                        return SemanticWorkerResult(
                            0,
                            (),
                            elapsed,
                            requested,
                            model_enabled=False,
                            model_status=str(
                                detector_status.get("status") or "unavailable"
                            ),
                            model_version=(
                                str(detector_status.get("model_version"))
                                if detector_status.get("model_version")
                                else None
                            ),
                        )
                last_error = worker_error or (
                    "semantic detector worker did not return a valid result"
                )
                if result.cancelled or result.timed_out:
                    break
        return SemanticWorkerResult(
            last_result.return_code if last_result is not None else 71,
            (),
            elapsed,
            requested,
            fallback_reason=fallback_reason,
            cancelled=bool(last_result and last_result.cancelled),
            timed_out=bool(last_result and last_result.timed_out),
            error=last_error,
        )

    def _run_command(
        self,
        command: list[str],
        cwd: Path | None,
        cancel_event: threading.Event | None,
        timeout_seconds: int,
    ) -> EngineResult:
        start = time.monotonic()
        group_options = popen_group_options()
        process = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd else None,
            env=self._environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **group_options,
        )
        with self._process_lock:
            self._active_process = process

        lines: queue.Queue[str | None] = queue.Queue(maxsize=self.max_pending_log_lines)
        dropped_lines = [0]

        def reader() -> None:
            def enqueue(line: str) -> None:
                try:
                    lines.put_nowait(line)
                except queue.Full:
                    dropped_lines[0] += 1

            try:
                assert process.stdout is not None
                stream = process.stdout
                read_limit = self.max_log_line_chars + 1
                try:
                    while True:
                        raw = stream.readline(read_limit)
                        if raw == "":
                            break
                        overlong = len(raw) == read_limit and not raw.endswith(("\n", "\r"))
                        if overlong:
                            line = raw[: self.max_log_line_chars].rstrip("\r\n") + " …[truncated]"
                            # Drain the remainder in bounded chunks.  This prevents a hostile
                            # or broken child process from forcing one unbounded Python string.
                            while raw and not raw.endswith(("\n", "\r")):
                                raw = stream.readline(read_limit)
                            enqueue(line)
                        else:
                            enqueue(raw.rstrip("\r\n"))
                except (OSError, ValueError):
                    # The control thread closes stdout after cancellation/timeout to
                    # guarantee reader termination.  A concurrent close is expected.
                    pass
            finally:
                try:
                    lines.put_nowait(None)
                except queue.Full:
                    # The control loop also observes the reader thread state, so a full
                    # queue cannot hide worker completion.
                    pass

        thread = threading.Thread(target=reader, name="scorescan-homr-output", daemon=True)
        thread.start()
        timed_out = False
        cancelled = False
        reader_finished = False
        try:
            while True:
                try:
                    item = lines.get(timeout=0.20)
                    if item is None:
                        reader_finished = True
                    elif item:
                        self.log(item)
                except queue.Empty:
                    pass
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    terminate_process_tree(process)
                    break
                if time.monotonic() - start > timeout_seconds:
                    timed_out = True
                    terminate_process_tree(process)
                    break
                if process.poll() is not None and (reader_finished or (not thread.is_alive() and lines.empty())):
                    break
            try:
                return_code = process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                terminate_process_tree(process, grace_seconds=0.5)
                return_code = process.wait(timeout=8)
            if process.stdout is not None:
                try:
                    process.stdout.close()
                except OSError:
                    pass
            thread.join(timeout=2.0)
            if thread.is_alive():
                self.log("识别引擎输出读取线程未及时退出；已与主任务隔离")
            while True:
                try:
                    item = lines.get_nowait()
                except queue.Empty:
                    break
                if item:
                    self.log(item)
            if dropped_lines[0]:
                self.log(f"识别引擎日志过多，已省略 {dropped_lines[0]} 行")
        finally:
            with self._process_lock:
                self._active_process = None

        elapsed = time.monotonic() - start
        error = None
        if cancelled:
            error = "任务已取消"
        elif timed_out:
            error = f"单页处理超过 {timeout_seconds // 60} 分钟，已终止该页"
        elif return_code != 0:
            error = f"识别引擎退出代码 {return_code}"
        return EngineResult(return_code, None, elapsed, cancelled, timed_out, error)
