from pathlib import Path

from app.tools.check_release_readiness import (
    find_non_cache_temporary_files,
    portable_runtime_source_contract,
)


def test_tmp_scan_skips_python_caches_but_covers_training_assets(
    tmp_path: Path,
) -> None:
    training_tmp = tmp_path / "training_data" / "active.tmp"
    source_tmp = tmp_path / "app" / "pending.tmp"
    ignored_tmp = tmp_path / "app" / "__pycache__" / "import.tmp"
    for path in (training_tmp, source_tmp, ignored_tmp):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"tmp")

    assert find_non_cache_temporary_files(tmp_path) == [
        "app/pending.tmp",
        "training_data/active.tmp",
    ]


def test_portable_runtime_source_contract_requires_offline_bundle(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    source = tmp_path / "app" / "src"
    runtime.mkdir(parents=True)
    source.mkdir(parents=True)
    (runtime / "start.cmd").write_text(
        "\n".join(
            (
                'set "PYTHON_EXE=%ROOT%\\runtime\\python\\python.exe"',
                'set "SITE_PACKAGES=%ROOT%\\runtime\\site-packages"',
                'set "SCORESCAN_OFFLINE_RUNTIME=1"',
                '"%PYTHON_EXE%" "%ROOT%\\runtime\\run_scorescan.py"',
            )
        ),
        encoding="utf-8",
    )
    (runtime / "run_scorescan.py").write_text("pass\n", encoding="utf-8")
    (source / "sitecustomize.py").write_text(
        'if event != "socket.connect":\n    pass\nsys.addaudithook(guard)\n',
        encoding="utf-8",
    )

    assert portable_runtime_source_contract(tmp_path)
    (runtime / "uv-bootstrap.ps1").write_text("# download\n", encoding="utf-8")
    assert not portable_runtime_source_contract(tmp_path)
