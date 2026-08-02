from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

from app.tools import build_release


def test_deterministic_zip_accepts_relative_source_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "VERSION").write_text("test\n", encoding="utf-8")
    output = tmp_path / "out" / "source.zip"
    monkeypatch.chdir(tmp_path)

    build_release.deterministic_zip(Path("source"), output, "ScoreScan-test")

    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == ["ScoreScan-test/VERSION"]


def test_release_excludes_out_of_scope_semantic_lyric_model() -> None:
    path = Path(
        "app/src/scorescan/resources/lyric_patch_calibrator.json"
    )
    assert not build_release.include(path)
    assert path.as_posix() in build_release.EXCLUDED_RELEASE_FILES


def test_launcher_and_bootstrap_reject_stale_ready_state() -> None:
    source_root = Path(__file__).resolve().parents[2]
    launcher = (source_root / "launcher.zig").read_text(encoding="utf-8")
    start_script = (source_root / "runtime" / "start.cmd").read_text(encoding="utf-8")

    assert "processIsAlive" in launcher
    assert "GetExitCodeProcess" in launcher
    assert "start.failed" in launcher
    assert "deleteFileAbsolute(init.io, ready)" in launcher
    assert "executableDirPathAlloc(init.io" in launcher
    assert "READY_FILE" in start_script
    assert "FAILED_FILE" in start_script
    assert "--frozen" in start_script
    assert "--no-dev" in start_script
    assert "--no-lock" not in start_script
    assert "uv-bootstrap.ps1" in start_script
    assert "pinned bootstrap" in start_script
    assert "launchHidden" in launcher
    assert "ShellExecuteW" in launcher
    assert '&.{ root, "runtime", "venv-cpu" }' in launcher
    assert "gpu_marker" not in launcher
    assert "if (!exists(init.io, runtime_environment))" in launcher
    assert "openReadyUrl(init.io, allocator, ready, root, true)" in launcher
    assert "openReadyUrl(init.io, allocator, ready, root, false)" in launcher
    assert "if (!allow_browser_fallback) return true;" in launcher


def test_first_run_bootstrap_download_uses_expand_archive_compatible_name() -> None:
    source_root = Path(__file__).resolve().parents[2]
    bootstrap = (source_root / "runtime" / "uv-bootstrap.ps1").read_text(encoding="utf-8")

    assert '"uv-download.zip"' in bootstrap
    assert '"uv-download.tmp"' not in bootstrap


def test_runtime_lock_installs_only_one_opencv_distribution() -> None:
    app_root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((app_root / "pyproject.toml").read_text(encoding="utf-8"))
    lock = (app_root / "uv.lock").read_text(encoding="utf-8")

    assert "opencv-python" in project["tool"]["uv"]["exclude-dependencies"]
    assert "opencv-python-headless==4.14.0.94" in project["project"]["dependencies"]
    assert '\nname = "opencv-python-headless"\n' in lock
    assert '\nname = "opencv-python"\n' not in lock


def test_published_runtime_is_cpu_only_and_frozen() -> None:
    app_root = Path(__file__).resolve().parents[1]
    source_root = app_root.parent
    cpu_project = tomllib.loads((app_root / "pyproject.toml").read_text(encoding="utf-8"))
    cpu_lock = (app_root / "uv.lock").read_text(encoding="utf-8")
    start_script = (source_root / "runtime" / "start.cmd").read_text(encoding="utf-8")

    assert "onnxruntime==1.28.0" in cpu_project["project"]["dependencies"]
    assert not any(
        item.startswith("onnxruntime-gpu")
        for item in cpu_project["project"]["dependencies"]
    )
    assert '\nname = "onnxruntime"\n' in cpu_lock
    assert '\nname = "onnxruntime-gpu"\n' not in cpu_lock
    assert not (source_root / "app-gpu" / "pyproject.toml").exists()
    assert not (source_root / "app-gpu" / "uv.lock").exists()

    assert 'set "PROJECT_ROOT=%ROOT%\\app"' in start_script
    assert 'set "RUNTIME_ENVIRONMENT=%ROOT%\\runtime\\venv-cpu"' in start_script
    assert "gpu.enabled" not in start_script.casefold()
    assert 'set "PROJECT_ROOT=%ROOT%\\app-gpu"' not in start_script
    assert 'set "RUNTIME_ENVIRONMENT=%ROOT%\\runtime\\venv-gpu"' not in start_script
    assert 'set "UV_PROJECT_ENVIRONMENT=%RUNTIME_ENVIRONMENT%"' in start_script
    assert '--frozen --no-dev --project "%PROJECT_ROOT%"' in start_script
    assert not (source_root / "runtime" / "enable-gpu.cmd").exists()
    assert not (source_root / "runtime" / "disable-gpu.cmd").exists()


def test_portable_copy_omits_development_only_roots(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "portable"
    for relative in (
        ".github/workflows/ci.yml",
        "BUILDING.md",
        "launcher.zig",
        "app/src/scorescan/main.py",
        "app/tests/test_main.py",
        "app/tools/benchmark.py",
        "training/queue.json",
        "test_materials/page.png",
        "README_zh-CN.txt",
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")

    build_release.copy_release_tree(source, destination, portable=True)

    assert (destination / "app/src/scorescan/main.py").is_file()
    assert (destination / "README_zh-CN.txt").is_file()
    assert not (destination / ".github").exists()
    assert not (destination / "BUILDING.md").exists()
    assert not (destination / "launcher.zig").exists()
    assert not (destination / "app/tests").exists()
    assert not (destination / "app/tools").exists()
    assert not (destination / "training").exists()
    assert not (destination / "test_materials").exists()


def test_windows_release_contains_verified_bootstrap_hashes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    (source / "runtime").mkdir(parents=True)
    (source / "runtime" / "start.cmd").write_text("@echo off\n", encoding="utf-8")
    (source / "VERSION").write_text("test\n", encoding="utf-8")
    launcher = tmp_path / "ScoreScan.exe"
    uv = tmp_path / "uv.exe"
    launcher.write_bytes(b"launcher")
    uv.write_bytes(b"uv-binary")
    tool = Path(__file__).resolve().parents[1] / "tools" / "build_release.py"
    subprocess.run(
        [
            sys.executable,
            str(tool),
            "--source-root",
            str(source),
            "--output-dir",
            str(output),
            "--version",
            "0.0.0-test",
            "--launcher",
            str(launcher),
            "--uv",
            str(uv),
        ],
        check=True,
    )
    archive_path = output / "Pigeon-Score-Scan-Windows-0.0.0-test.zip"
    with zipfile.ZipFile(archive_path) as archive:
        root = "Pigeon-Score-Scan-0.0.0-test"
        expected = hashlib.sha256(b"uv-binary").hexdigest()
        assert archive.read(f"{root}/runtime/uv.sha256").decode().strip() == expected
        manifest = json.loads(archive.read(f"{root}/runtime/bootstrap_manifest.json"))
        assert manifest["uv_sha256"] == expected
        assert manifest["launcher_sha256"] == hashlib.sha256(b"launcher").hexdigest()


def test_windows_release_can_use_pinned_first_run_uv_bootstrap(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    (source / "runtime").mkdir(parents=True)
    (source / "runtime" / "start.cmd").write_text("@echo off\n", encoding="utf-8")
    (source / "runtime" / "uv-bootstrap.ps1").write_text("# pinned bootstrap\n", encoding="utf-8")
    (source / "VERSION").write_text("test\n", encoding="utf-8")
    launcher = tmp_path / "ScoreScan.exe"
    launcher.write_bytes(b"launcher")
    tool = Path(__file__).resolve().parents[1] / "tools" / "build_release.py"
    subprocess.run(
        [
            sys.executable,
            str(tool),
            "--source-root",
            str(source),
            "--output-dir",
            str(output),
            "--version",
            "0.0.0-test",
            "--launcher",
            str(launcher),
        ],
        check=True,
    )
    archive_path = output / "Pigeon-Score-Scan-Windows-0.0.0-test.zip"
    with zipfile.ZipFile(archive_path) as archive:
        root = "Pigeon-Score-Scan-0.0.0-test"
        manifest = json.loads(archive.read(f"{root}/runtime/bootstrap_manifest.json"))
        assert manifest["format"] == 2
        assert manifest["uv_delivery"] == "pinned-first-run-download"
        assert manifest["uv_version"] == "0.9.26"
        assert len(manifest["uv_archive_sha256"]) == 64
        assert f"{root}/runtime/uv.exe" not in archive.namelist()
        assert f"{root}/runtime/uv-bootstrap.ps1" in archive.namelist()


def test_release_zip_normalises_source_file_permissions(tmp_path: Path) -> None:
    source_a = tmp_path / "source-a"
    source_b = tmp_path / "source-b"
    output_a = tmp_path / "out-a"
    output_b = tmp_path / "out-b"
    source_a.mkdir()
    source_b.mkdir()
    for source, mode in ((source_a, 0o600), (source_b, 0o644)):
        report = source / "report.json"
        report.write_text('{"ok": true}\n', encoding="utf-8")
        report.chmod(mode)
    tool = Path(__file__).resolve().parents[1] / "tools" / "build_release.py"
    for source, output in ((source_a, output_a), (source_b, output_b)):
        subprocess.run(
            [
                sys.executable,
                str(tool),
                "--source-root",
                str(source),
                "--output-dir",
                str(output),
                "--version",
                "0.0.0-test",
            ],
            check=True,
        )
    zip_a = output_a / "Pigeon-Score-Scan-Source-0.0.0-test.zip"
    zip_b = output_b / "Pigeon-Score-Scan-Source-0.0.0-test.zip"
    assert zip_a.read_bytes() == zip_b.read_bytes()
    with zipfile.ZipFile(zip_a) as archive:
        info = archive.getinfo("Pigeon-Score-Scan-Source-0.0.0-test/report.json")
        assert info.external_attr >> 16 == 0o100644


def test_release_archives_exclude_user_runtime_state_and_stale_binaries(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    (source / "runtime" / "venv").mkdir(parents=True)
    (source / "runtime" / "venv-cpu").mkdir()
    (source / "runtime" / "venv-gpu").mkdir()
    (source / "runtime" / "uv-cache").mkdir()
    (source / "runtime" / "python").mkdir()
    (source / "runtime" / "diagnostics").mkdir()
    (source / "runtime" / "preview-stale").mkdir()
    (source / "workspace" / "private-job").mkdir(parents=True)
    (source / "development_reports" / "private-run").mkdir(parents=True)
    (source / "tmp" / "private-run").mkdir(parents=True)
    (source / "training_data" / "external" / "private-corpus").mkdir(parents=True)
    (source / "app-gpu").mkdir()
    (source / "app" / ".venv").mkdir(parents=True)
    (source / "app" / "source.py").write_text("kept = True\n", encoding="utf-8")
    (source / "runtime" / "start.cmd").write_text("@echo off\n", encoding="utf-8")
    (source / "runtime" / "uv-bootstrap.ps1").write_text("# bootstrap\n", encoding="utf-8")
    for path in (
        source / "runtime" / "venv" / "private.bin",
        source / "runtime" / "venv-cpu" / "private.bin",
        source / "runtime" / "venv-gpu" / "private.bin",
        source / "runtime" / "uv-cache" / "private.bin",
        source / "runtime" / "python" / "private.bin",
        source / "runtime" / "diagnostics" / "support.zip",
        source / "runtime" / "preview-stale" / "preview.html",
        source / "runtime" / "ready.txt",
        source / "runtime" / "server.lock",
        source / "runtime" / "launcher.log",
        source / "runtime" / "gpu.enabled",
        source / "runtime" / "uv.exe",
        source / "runtime" / "uv.sha256",
        source / "runtime" / "bootstrap_manifest.json",
        source / "runtime" / "ocr_probe.musicxml",
        source / "runtime" / "wedge_debug.png",
        source / "workspace" / "private-job" / "scan.png",
        source / "development_reports" / "private-run" / "scan.png",
        source / "tmp" / "private-run" / "scan.png",
        source / "training_data" / "external" / "private-corpus" / "page.png",
        source / "app-gpu" / "uv.lock",
        source / "app" / ".venv" / "private.bin",
        source / "ScoreScan.exe",
    ):
        path.write_bytes(b"must-not-ship")
    launcher = tmp_path / "fresh-ScoreScan.exe"
    launcher.write_bytes(b"fresh-launcher")
    tool = Path(__file__).resolve().parents[1] / "tools" / "build_release.py"

    subprocess.run(
        [
            sys.executable,
            str(tool),
            "--source-root",
            str(source),
            "--output-dir",
            str(output),
            "--version",
            "0.0.0-test",
            "--launcher",
            str(launcher),
        ],
        check=True,
    )

    for archive_path in (
        output / "Pigeon-Score-Scan-Source-0.0.0-test.zip",
        output / "Pigeon-Score-Scan-Windows-0.0.0-test.zip",
    ):
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            assert any(name.endswith("/app/source.py") for name in names)
            assert not any("/workspace/" in name for name in names)
            assert not any("/runtime/venv/" in name for name in names)
            assert not any("/runtime/venv-cpu/" in name for name in names)
            assert not any("/runtime/venv-gpu/" in name for name in names)
            assert not any("/runtime/uv-cache/" in name for name in names)
            assert not any("/runtime/python/" in name for name in names)
            assert not any("/runtime/diagnostics/" in name for name in names)
            assert not any("/runtime/preview-" in name for name in names)
            assert not any(
                name.endswith(("/ocr_probe.musicxml", "/wedge_debug.png"))
                for name in names
            )
            assert not any("/development_reports/" in name for name in names)
            assert not any("/tmp/" in name for name in names)
            assert not any("/training_data/" in name for name in names)
            assert not any("/app-gpu/" in name for name in names)
            assert not any("/.venv/" in name for name in names)
            assert not any(
                name.endswith(("/ready.txt", "/server.lock", "/launcher.log", "/gpu.enabled"))
                for name in names
            )
            assert b"must-not-ship" not in b"".join(
                archive.read(name) for name in names if not name.endswith("/")
            )

    with zipfile.ZipFile(output / "Pigeon-Score-Scan-Windows-0.0.0-test.zip") as archive:
        root = "Pigeon-Score-Scan-0.0.0-test"
        assert archive.read(f"{root}/pigeon-score-scan.exe") == b"fresh-launcher"


def test_release_file_walk_prunes_external_training_tree(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    (source / "app").mkdir(parents=True)
    (source / "app" / "kept.py").write_text("kept = True\n", encoding="utf-8")
    private = source / "training_data" / "external" / "private-corpus"
    private.mkdir(parents=True)
    (private / "page.png").write_bytes(b"private")

    visited: list[Path] = []
    original_walk = build_release.os.walk

    def recording_walk(*args, **kwargs):
        for directory, directory_names, file_names in original_walk(*args, **kwargs):
            visited.append(Path(directory))
            yield directory, directory_names, file_names

    monkeypatch.setattr(build_release.os, "walk", recording_walk)
    selected = build_release.release_files(source)

    assert [path.relative_to(source).as_posix() for path in selected] == ["app/kept.py"]
    assert source / "training_data" not in visited


def test_release_file_walk_prunes_output_directories(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    (source / "app").mkdir(parents=True)
    (source / "app" / "kept.py").write_text("kept = True\n", encoding="utf-8")
    for name in ("build", "dist"):
        output = source / name / "nested"
        output.mkdir(parents=True)
        (output / "stale.zip").write_bytes(b"stale")

    visited: list[Path] = []
    original_walk = build_release.os.walk

    def recording_walk(*args, **kwargs):
        for directory, directory_names, file_names in original_walk(*args, **kwargs):
            visited.append(Path(directory))
            yield directory, directory_names, file_names

    monkeypatch.setattr(build_release.os, "walk", recording_walk)
    selected = build_release.release_files(source)

    assert [path.relative_to(source).as_posix() for path in selected] == ["app/kept.py"]
    assert source / "build" not in visited
    assert source / "dist" not in visited
