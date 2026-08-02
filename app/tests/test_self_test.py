import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

from scorescan.config import APP_VERSION, Settings
from scorescan.diagnostics import create_diagnostics_bundle
from scorescan.self_test import _verovio_smoke_check, run_system_check


def test_verovio_self_test_uses_packaged_resource_path(tmp_path: Path, monkeypatch) -> None:
    package = tmp_path / "verovio"
    package.mkdir()
    module_file = package / "__init__.py"
    module_file.write_text("", encoding="utf-8")
    resource_path = package / "data"
    resource_path.mkdir()
    observed: dict[str, object] = {}

    class FakeToolkit:
        def __init__(self) -> None:
            self.resource_path = str(tmp_path / "wrong")

        def getResourcePath(self) -> str:
            return self.resource_path

        def setResourcePath(self, value: str) -> None:
            self.resource_path = value
            observed["toolkit_resource"] = value

        def setOptions(self, options: dict[str, object]) -> None:
            observed["options"] = options

        def loadData(self, _xml: str) -> bool:
            return Path(self.resource_path) == resource_path

        def renderToSVG(self, _page: int) -> str:
            return "<svg/>"

    fake_verovio = SimpleNamespace(
        __file__=str(module_file),
        __version__="test",
        setDefaultResourcePath=lambda value: observed.__setitem__("default_resource", value),
        toolkit=FakeToolkit,
    )
    monkeypatch.setitem(sys.modules, "verovio", fake_verovio)

    result = _verovio_smoke_check()

    assert result.ok is True
    assert observed["default_resource"] == str(resource_path)
    assert observed["toolkit_resource"] == str(resource_path)


def test_system_check_and_privacy_safe_diagnostics(tmp_path: Path) -> None:
    (tmp_path / "VERSION").write_text(APP_VERSION + "\n", encoding="utf-8")
    settings = Settings.from_root(tmp_path)
    result = run_system_check(settings)
    assert result["version"] == APP_VERSION
    assert result["environment"]["runtime_profile"] == "development"
    assert any(item["key"] == "models:manifest" and item["ok"] for item in result["checks"])
    assert any(item["key"] == "accelerator:runtime" and item["ok"] for item in result["checks"])

    job_root = settings.workspace / "job-1"
    job_root.mkdir(parents=True)
    secret_name = "private-composer-secret.pdf"
    (job_root / "job.json").write_text(
        json.dumps({"id": "job-1", "status": "failed", "source_files": [secret_name], "error": "RuntimeError: failed"}),
        encoding="utf-8",
    )
    bundle = create_diagnostics_bundle(settings)
    assert bundle.exists()
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
        assert {"system_check.json", "application.json", "recent_jobs_redacted.json", "model_manifest.json"} <= names
        content = b"\n".join(archive.read(name) for name in names)
        assert secret_name.encode() not in content
        recent = json.loads(archive.read("recent_jobs_redacted.json"))
        application = json.loads(archive.read("application.json"))
        assert application["accelerator"]["selected"] in {"cpu", "cuda"}
        assert "available_providers" in application["accelerator"]
        assert application["runtime_profile"] == "development"
        assert recent == [
            {
                "id": "job-1",
                "status": "failed",
                "stage": None,
                "progress": None,
                "total_pages": 0,
                "current_page": None,
                "quality_state": None,
                "quality_score": None,
                "warning_count": 0,
                "error_type": "RuntimeError",
                "created_at": None,
                "updated_at": None,
            }
        ]



def test_cli_self_test_does_not_require_flask_to_start(tmp_path: Path) -> None:
    (tmp_path / "VERSION").write_text(APP_VERSION + "\n", encoding="utf-8")
    app_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(app_root / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scorescan",
            "--self-test",
            "--json",
            "--root",
            str(tmp_path),
        ],
        cwd=app_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode in {0, 1}
    assert "Traceback" not in completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["version"] == APP_VERSION
    assert any(item["key"] == "module:flask" for item in payload["checks"])
