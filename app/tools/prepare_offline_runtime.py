from __future__ import annotations

"""Prepare the pinned Python, dependencies and models for an offline release."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from importlib.metadata import PathDistribution
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[2]
APP_SOURCE = SOURCE_ROOT / "app" / "src"
if str(APP_SOURCE) not in os.sys.path:
    os.sys.path.insert(0, str(APP_SOURCE))

from scorescan.offline_runtime import BUNDLED_MODELS, sha256_file, verify_bundled_models


def _run(arguments: list[str], *, environment: dict[str, str] | None = None) -> None:
    subprocess.run(arguments, check=True, env=environment)


def _remove_bytecode(root: Path) -> None:
    for directory in sorted(root.rglob("__pycache__"), reverse=True):
        shutil.rmtree(directory, ignore_errors=True)
    for suffix in ("*.pyc", "*.pyo"):
        for path in root.rglob(suffix):
            path.unlink(missing_ok=True)


def _distribution_inventory(site_packages: Path) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for metadata_path in sorted(site_packages.glob("*.dist-info"), key=lambda item: item.name.casefold()):
        distribution = PathDistribution(metadata_path)
        name = str(distribution.metadata.get("Name", "")).strip()
        version = str(distribution.version or "").strip()
        if name and version:
            result.append({"name": name, "version": version})
    return result


def _tree_stats(root: Path) -> tuple[int, int]:
    files = [path for path in root.rglob("*") if path.is_file()]
    return len(files), sum(path.stat().st_size for path in files)


def prepare(
    source_root: Path,
    output_root: Path,
    uv_executable: Path,
    python_version: str,
) -> None:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"offline runtime output is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="pigeon-offline-runtime-build-") as temporary_name:
        temporary = Path(temporary_name)
        managed_root = temporary / "managed-python"
        requirements = temporary / "requirements.txt"
        cache = temporary / "uv-cache"
        environment = os.environ.copy()
        environment.update(
            {
                "UV_CACHE_DIR": str(cache),
                "UV_LINK_MODE": "copy",
                "PYTHONUTF8": "1",
            }
        )
        _run(
            [
                str(uv_executable.resolve()),
                "python",
                "install",
                python_version,
                "--install-dir",
                str(managed_root),
                "--no-bin",
                "--no-registry",
            ],
            environment=environment,
        )
        candidates = sorted(managed_root.glob("cpython-*-windows-x86_64-none/python.exe"))
        if len(candidates) != 1:
            raise RuntimeError("expected one pinned x64 Windows Python installation")
        python_source = candidates[0].parent
        python_target = output_root / "python"
        shutil.copytree(python_source, python_target)
        python_executable = python_target / "python.exe"

        _run(
            [
                str(uv_executable.resolve()),
                "export",
                "--project",
                str(source_root / "app"),
                "--frozen",
                "--no-dev",
                "--no-emit-project",
                "--format",
                "requirements.txt",
                "--no-annotate",
                "--no-header",
                "--output-file",
                str(requirements),
            ],
            environment=environment,
        )
        site_packages = output_root / "site-packages"
        _run(
            [
                str(uv_executable.resolve()),
                "pip",
                "install",
                "--python",
                str(python_executable),
                "--target",
                str(site_packages),
                "--requirements",
                str(requirements),
                "--no-deps",
                "--require-hashes",
                "--link-mode",
                "copy",
                "--strict",
            ],
            environment=environment,
        )

        model_environment = environment.copy()
        model_environment.update(
            {
                "PYTHONPATH": str(site_packages),
                "PYTHONNOUSERSITE": "1",
            }
        )
        _run(
            [str(python_executable), "-m", "homr.main", "--init", "--gpu", "no"],
            environment=model_environment,
        )
        _remove_bytecode(site_packages)

        problems = verify_bundled_models(site_packages, verify_hashes=True)
        if problems:
            raise RuntimeError("offline model verification failed: " + "; ".join(problems))

        distributions = _distribution_inventory(site_packages)
        if len(distributions) != 44:
            raise RuntimeError(f"expected 44 locked runtime distributions, found {len(distributions)}")
        file_count, total_bytes = _tree_stats(output_root)
        manifest = {
            "format": 1,
            "delivery": "offline-bundled",
            "network_required": False,
            "python": {
                "version": python_version,
                "executable": "python/python.exe",
                "sha256": sha256_file(python_executable),
            },
            "lock_sha256": sha256_file(source_root / "app" / "uv.lock"),
            "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
            "distributions": distributions,
            "models": [
                {
                    "package": model.package,
                    "path": model.relative_path,
                    "size": model.size,
                    "sha256": model.sha256,
                }
                for model in BUNDLED_MODELS
            ],
            "file_count": file_count,
            "total_bytes": total_bytes,
        }
        (output_root / "offline_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--python-version", default="3.12.10")
    args = parser.parse_args()
    prepare(args.source_root, args.output_root, args.uv, args.python_version)


if __name__ == "__main__":
    main()
