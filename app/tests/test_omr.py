from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from scorescan.accelerator import AcceleratorStatus
from scorescan.omr import EngineResult, HomrRunner


def _accelerator(selected: str = "cpu") -> AcceleratorStatus:
    return AcceleratorStatus(
        requested="auto",
        selected=selected,
        onnxruntime_version="test",
        available_providers=("CUDAExecutionProvider", "CPUExecutionProvider")
        if selected == "cuda"
        else ("CPUExecutionProvider",),
        cpu_package_installed=selected == "cpu",
        gpu_package_installed=selected == "cuda",
        package_conflict=False,
    )


def test_homr_output_queue_is_bounded_and_reports_drops() -> None:
    logs: list[str] = []

    def slow_log(message: str) -> None:
        time.sleep(0.001)
        logs.append(message)

    runner = HomrRunner(
        slow_log,
        page_timeout_seconds=10,
        max_pending_log_lines=8,
        max_log_line_chars=128,
    )
    result = runner._run_command(
        [
            sys.executable,
            "-c",
            "for i in range(2000): print(str(i) + ':' + 'x' * 500, flush=True)",
        ],
        None,
        None,
        10,
    )
    assert result.return_code == 0
    assert any("[truncated]" in line for line in logs)
    assert any("已省略" in line for line in logs)
    assert len(logs) < 2000


def test_homr_invalid_musicxml_is_quarantined(tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"image")
    logs: list[str] = []
    runner = HomrRunner(logs.append)
    monkeypatch.setattr(runner, "available", lambda: True)

    def fake_run(*args, **kwargs):
        runner.expected_xml(image).write_bytes(b"<score-partwise>" + b"x" * 400 + b"</score-partwise>")
        return EngineResult(0, None, 0.01)

    monkeypatch.setattr(runner, "_run_command", fake_run)
    result = runner.run_page(image)
    assert result.return_code == 65
    assert result.xml_path is None
    assert result.error is not None and "MusicXML" in result.error
    assert not runner.expected_xml(image).exists()
    assert runner.expected_xml(image).with_name("page.stale.musicxml").exists()
    assert any("结构无效" in line for line in logs)


def test_homr_zero_exit_with_tiny_output_is_failure(tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"image")
    logs: list[str] = []
    runner = HomrRunner(logs.append)
    monkeypatch.setattr(runner, "available", lambda: True)

    def fake_run(*args, **kwargs):
        runner.expected_xml(image).write_bytes(b"<score-partwise/>")
        return EngineResult(0, None, 0.01)

    monkeypatch.setattr(runner, "_run_command", fake_run)
    result = runner.run_page(image)
    assert result.return_code == 65
    assert result.xml_path is None
    assert result.error == "识别引擎未生成足够大小的 MusicXML"
    assert not runner.expected_xml(image).exists()
    assert runner.expected_xml(image).with_name("page.stale.musicxml").exists()


def test_homr_resolves_relative_image_before_changing_worker_directory(
    tmp_path: Path, monkeypatch
) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"image")
    runner = HomrRunner(lambda _message: None)
    monkeypatch.setattr(runner, "available", lambda: True)
    observed: dict[str, object] = {}

    def fake_run(arguments, cwd, *_args, **_kwargs):
        observed["image_argument"] = arguments[0]
        observed["cwd"] = cwd
        return EngineResult(1, None, 0.01, error="fixture")

    monkeypatch.setattr(runner, "_run_with_cpu_fallback", fake_run)
    monkeypatch.chdir(tmp_path)
    runner.run_page(Path("page.png"))

    assert Path(str(observed["image_argument"])).is_absolute()
    assert observed["cwd"] == image.parent


def test_homr_environment_enforces_bounded_shared_thread_budget(monkeypatch) -> None:
    runner = HomrRunner(lambda _message: None)
    monkeypatch.setattr("scorescan.cpu_runtime.os.cpu_count", lambda: 4)
    monkeypatch.setenv("SCORESCAN_ENGINE_THREADS", "999")
    monkeypatch.setenv("OMP_NUM_THREADS", "128")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "64")
    env = runner._environment()
    for key in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
        assert env[key] == "4"

    monkeypatch.setenv("SCORESCAN_ENGINE_THREADS", "2")
    env = runner._environment()
    assert env["OMP_NUM_THREADS"] == "2"
    assert env["OPENBLAS_NUM_THREADS"] == "2"


def test_homr_environment_invalid_budget_falls_back_to_cpu_cap(monkeypatch) -> None:
    runner = HomrRunner(lambda _message: None)
    monkeypatch.setattr("scorescan.cpu_runtime.os.cpu_count", lambda: 16)
    monkeypatch.setenv("SCORESCAN_ENGINE_THREADS", "not-a-number")
    assert runner._environment()["OMP_NUM_THREADS"] == "8"


def test_homr_environment_uses_absolute_scorescan_source_path(monkeypatch) -> None:
    runner = HomrRunner(lambda _message: None)
    monkeypatch.setenv("PYTHONPATH", "relative-entry")
    entries = runner._environment()["PYTHONPATH"].split(";" if sys.platform == "win32" else ":")
    assert Path(entries[0]).is_absolute()
    assert (Path(entries[0]) / "scorescan" / "homr_worker.py").is_file()


def test_homr_passes_explicit_cpu_mode(monkeypatch) -> None:
    runner = HomrRunner(lambda _message: None)
    monkeypatch.setattr(runner, "accelerator_status", lambda **_kwargs: _accelerator("cpu"))
    command = runner._homr_command("page.png")
    assert command[-2:] == ["--gpu", "no"]


def test_raw_tuplet_worker_mode_is_explicit_and_cache_isolated(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"image")
    runner = HomrRunner(lambda _message: None)
    monkeypatch.setattr(runner, "available", lambda: True)
    calls: list[list[str]] = []

    def fake_run(arguments, *_args, **_kwargs):
        calls.append(list(arguments))
        write_score = (
            b"<?xml version='1.0' encoding='UTF-8'?>"
            b"<score-partwise version='4.0'><part-list><score-part id='P1'>"
            b"<part-name>" + b"x" * 160 + b"</part-name></score-part></part-list><part id='P1'>"
            b"<measure number='1'><note><rest/><duration>4</duration><voice>1</voice>"
            b"<type>whole</type></note></measure></part>"
            b"</score-partwise>"
        )
        runner.expected_xml(image).write_bytes(write_score)
        return EngineResult(0, runner.expected_xml(image), 0.01)

    monkeypatch.setattr(runner, "_run_with_cpu_fallback", fake_run)

    raw = runner.run_page(image, preserve_raw_tuplets=True)
    cleaned = runner.run_page(image)

    assert raw.return_code == 0
    assert cleaned.return_code == 0
    assert len(calls) == 2
    assert "--scorescan-preserve-raw-tuplets" in calls[0]
    assert "--scorescan-preserve-raw-tuplets" not in calls[1]
    manifest = json.loads(
        runner.expected_xml(image).with_suffix(".omr-cache.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["profile"] == "homr-large-page-v1"


def test_raw_tuplet_hook_disables_only_the_deletion_stage(monkeypatch) -> None:
    from homr.transformer import vocabulary
    from scorescan.homr_worker import _install_raw_tuplet_preservation

    sentinel = object()
    monkeypatch.setattr(
        vocabulary,
        "_fix_over_eager_tuplets",
        lambda _chords: [sentinel],
    )

    _install_raw_tuplet_preservation()

    chords = [[object()]]
    assert vocabulary._fix_over_eager_tuplets(chords) is chords


def test_homr_ignores_legacy_gpu_markers(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "gpu.enabled").write_text("legacy\n", encoding="utf-8")
    runner = HomrRunner(lambda _message: None)

    command = runner._homr_command("page.png")

    assert Path(command[0]) == Path(sys.executable).resolve()
    assert command[-2:] == ["--gpu", "no"]
    assert runner.accelerator_status().selected == "cpu"


def test_homr_failure_is_not_retried_in_a_second_runtime(monkeypatch) -> None:
    runner = HomrRunner(lambda _message: None)
    commands: list[list[str]] = []

    def fake_run(command, *_args, **_kwargs):
        commands.append(command)
        return EngineResult(1, None, 2.5, error="native failure")

    monkeypatch.setattr(runner, "_run_command", fake_run)
    result = runner._run_with_cpu_fallback(["page.png"], None, None, 60)

    assert result.return_code == 1
    assert len(commands) == 1
    assert commands[0][-2:] == ["--gpu", "no"]


def test_ocr_worker_is_cpu_only_and_provider_verified(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image = tmp_path / "page.png"
    xml = tmp_path / "page.musicxml"
    image.write_bytes(b"image")
    xml.write_bytes(b"<score-partwise/>")
    runner = HomrRunner(lambda _message: None)
    monkeypatch.setattr(runner, "_prepare_worker_profile", lambda: None)
    attempts: list[str] = []

    def fake_run(command, *_args, **_kwargs):
        request_path = Path(command[command.index("--scorescan-ocr-request") + 1])
        accelerator = command[command.index("--scorescan-ocr-accelerator") + 1]
        attempts.append(accelerator)
        response = request_path.with_suffix(".result.json")
        response.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "ok": True,
                    "marks": [{}],
                    "warnings": [],
                    "runtime": {
                        "requested": "cpu",
                        "selected": "cpu",
                        "verified": True,
                        "component_providers": {
                            "detection": ["CPUExecutionProvider"],
                            "classification": ["CPUExecutionProvider"],
                            "recognition": ["CPUExecutionProvider"],
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        return EngineResult(0, None, 2.0)

    monkeypatch.setattr(runner, "_run_command", fake_run)
    result = runner.run_ocr_enrichment(
        image,
        xml,
        SimpleNamespace(to_dict=lambda: {"systems": []}),
    )

    assert attempts == ["cpu"]
    assert result.return_code == 0
    assert result.requested_accelerator == "cpu"
    assert result.selected_accelerator == "cpu"
    assert result.runtime_verified
    assert result.fallback_reason is None
    assert result.elapsed_seconds == 2.0


def test_homr_cancellation_reaps_output_reader_thread() -> None:
    logs: list[str] = []
    runner = HomrRunner(logs.append, page_timeout_seconds=10)
    cancel = threading.Event()
    cancel.set()
    result = runner._run_command(
        [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(30)"],
        None,
        cancel,
        10,
    )
    assert result.cancelled
    assert result.return_code != 0
    assert not any(thread.name == "scorescan-homr-output" and thread.is_alive() for thread in threading.enumerate())
