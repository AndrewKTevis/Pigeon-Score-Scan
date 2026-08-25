from pathlib import Path

from app.tools.check_release_readiness import (
    find_non_cache_temporary_files,
    portable_runtime_source_contract,
    public_pypi_lock_contract,
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


def test_public_pypi_lock_contract_parses_exact_artifact_hosts(tmp_path: Path) -> None:
    lock_path = tmp_path / "uv.lock"
    valid = """\
version = 1
revision = 3
requires-python = ">=3.12, <3.14"

[[package]]
name = "sample"
version = "1.0.0"
source = { registry = "https://pypi.org/simple" }
sdist = { url = "https://files.pythonhosted.org/packages/sample.tar.gz", hash = "sha256:00" }
"""
    lock_path.write_text(valid, encoding="utf-8")
    assert public_pypi_lock_contract(lock_path)

    deceptive_values = (
        valid.replace("files.pythonhosted.org", "files.pythonhosted.org.example.invalid"),
        valid.replace("https://files.pythonhosted.org", "https://user@files.pythonhosted.org"),
        valid.replace("https://pypi.org/simple", "https://pypi.org.example.invalid/simple"),
        valid.replace("/packages/sample.tar.gz", "/other/sample.tar.gz"),
    )
    for value in deceptive_values:
        lock_path.write_text(value, encoding="utf-8")
        assert not public_pypi_lock_contract(lock_path)


def test_checked_in_lock_uses_exact_public_pypi_hosts() -> None:
    app_root = Path(__file__).resolve().parents[1]
    assert public_pypi_lock_contract(app_root / "uv.lock")
