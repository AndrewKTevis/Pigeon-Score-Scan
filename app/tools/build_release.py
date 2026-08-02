from __future__ import annotations

"""Build deterministic source and Windows portable ZIP archives."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)
EXCLUDED_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".git",
    ".venv",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".tmp"}
# `training_data` contains mutable external corpora, checkpoints, logs and
# diagnostics.  It is deliberately outside both source and portable release
# archives: shipping it would leak local research state and can inflate one
# build by tens of gigabytes.
EXCLUDED_ROOT_NAMES = {
    "build",
    "dist",
    "workspace",
    "development_reports",
    "tmp",
    "training_data",
    "app-gpu",
}
PRODUCT_SLUG = "Pigeon-Score-Scan"
PRODUCT_EXE = "pigeon-score-scan.exe"
GENERATED_ROOT_NAMES = {
    "ScoreScan.exe",
    "Pigeon Score Scan.exe",
    PRODUCT_EXE,
    "ScoreScan",
    "launcher.exe",
}
PORTABLE_EXCLUDED_ROOT_NAMES = {
    ".editorconfig",
    ".gitattributes",
    ".github",
    ".gitignore",
    "BUILDING.md",
    "CHANGELOG.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "PUBLIC_RELEASE_CHECKLIST.md",
    "launcher.zig",
    "third_party",
    "training",
    "test_materials",
}
PORTABLE_EXCLUDED_PREFIXES = {("app", "tests"), ("app", "tools")}
RUNTIME_SOURCE_NAMES = {
    "run_scorescan.py",
    "show-window.cmd",
    "start.cmd",
}
GENERATED_RUNTIME_NAMES = {
    "bootstrap_manifest.json",
    "offline_manifest.json",
    "python",
    "site-packages",
}
EXCLUDED_RELEASE_FILES = {
    # Semantic lyric output is outside the frozen product boundary.  Retain
    # the research artifact locally for reproducibility, but never ship it in
    # either the source release or Windows portable package.
    "app/src/scorescan/resources/lyric_patch_calibrator.json",
}


def include(
    path: Path,
    *,
    allow_generated: bool = False,
    portable: bool = False,
) -> bool:
    if not path.parts:
        return True
    if path.as_posix() in EXCLUDED_RELEASE_FILES:
        return False
    if path.parts[0] in EXCLUDED_ROOT_NAMES:
        return False
    if portable and path.parts[0] in PORTABLE_EXCLUDED_ROOT_NAMES:
        return False
    if portable and tuple(path.parts[:2]) in PORTABLE_EXCLUDED_PREFIXES:
        return False
    if path.parts[0] in GENERATED_ROOT_NAMES and not allow_generated:
        return False
    if any(part in EXCLUDED_NAMES for part in path.parts):
        return False
    if path.suffix in EXCLUDED_SUFFIXES:
        return False
    if path.parts[0] == "runtime" and len(path.parts) >= 2:
        runtime_name = path.parts[1]
        if runtime_name in GENERATED_RUNTIME_NAMES:
            return allow_generated
        # Runtime is mutable and may contain user jobs, caches, diagnostics or
        # developer probes with arbitrary names.  A source-name whitelist is
        # safer than trying to enumerate every possible generated artifact.
        return runtime_name in RUNTIME_SOURCE_NAMES
    return True


def copy_release_tree(
    source: Path,
    destination: Path,
    *,
    portable: bool = False,
) -> None:
    source = source.resolve()

    def ignore(directory: str, names: list[str]) -> set[str]:
        relative = Path(directory).resolve().relative_to(source)
        return {
            name
            for name in names
            if not include(relative / name, portable=portable)
        }

    shutil.copytree(source, destination, ignore=ignore)


def release_files(source: Path, *, allow_generated: bool = False) -> list[Path]:
    """Return included files without descending into excluded directory roots."""

    source = source.resolve()
    selected: list[Path] = []
    for directory, directory_names, file_names in os.walk(source, topdown=True):
        relative_directory = Path(directory).resolve().relative_to(source)
        directory_names[:] = sorted(
            (
                name
                for name in directory_names
                if include(relative_directory / name, allow_generated=allow_generated)
            ),
            key=str.casefold,
        )
        for name in file_names:
            path = Path(directory) / name
            relative = path.relative_to(source)
            if include(relative, allow_generated=allow_generated):
                selected.append(path)
    return sorted(
        selected,
        key=lambda path: path.relative_to(source).as_posix().casefold(),
    )


def deterministic_zip(
    source: Path,
    output: Path,
    archive_root: str,
    *,
    allow_generated: bool = False,
) -> None:
    source = source.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in release_files(source, allow_generated=allow_generated):
            relative = path.relative_to(source)
            info = zipfile.ZipInfo(f"{archive_root}/{relative.as_posix()}", FIXED_ZIP_TIME)
            # ZIP permissions are part of the reproducible artifact.  Never
            # inherit umask- or extraction-dependent source modes (for example
            # atomic JSON writes commonly produce 0600 before extraction turns
            # them into 0644).  ScoreScan files are invoked explicitly by their
            # host runtime, so a stable regular-file mode is sufficient on all
            # supported build hosts.
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    os.replace(temporary, output)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_offline_runtime(source: Path, destination: Path) -> dict[str, object]:
    source = source.resolve()
    python_source = source / "python"
    packages_source = source / "site-packages"
    manifest_source = source / "offline_manifest.json"
    if not (python_source / "python.exe").is_file():
        raise FileNotFoundError("offline runtime is missing python/python.exe")
    if not (packages_source / "homr").is_dir():
        raise FileNotFoundError("offline runtime is missing bundled dependencies")
    if not manifest_source.is_file():
        raise FileNotFoundError("offline runtime is missing offline_manifest.json")
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"offline runtime contains a symbolic link: {path}")
    payload = json.loads(manifest_source.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("delivery") != "offline-bundled"
        or payload.get("network_required") is not False
    ):
        raise ValueError("offline runtime manifest has an invalid delivery contract")
    python_payload = payload.get("python")
    if not isinstance(python_payload, dict):
        raise ValueError("offline runtime manifest is missing Python metadata")
    expected_python_hash = str(python_payload.get("sha256", "")).casefold()
    if sha256(python_source / "python.exe") != expected_python_hash:
        raise ValueError("offline Python executable does not match its manifest")
    shutil.copytree(python_source, destination / "python")
    shutil.copytree(packages_source, destination / "site-packages")
    shutil.copy2(manifest_source, destination / "offline_manifest.json")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--launcher", type=Path)
    parser.add_argument("--offline-runtime", type=Path)
    args = parser.parse_args()

    version_label = args.version
    source_name = f"{PRODUCT_SLUG}-Source-{version_label}"
    windows_name = f"{PRODUCT_SLUG}-{version_label}"
    source_zip = args.output_dir / f"{PRODUCT_SLUG}-Source-{version_label}.zip"
    windows_zip = args.output_dir / f"{PRODUCT_SLUG}-Windows-{version_label}.zip"
    deterministic_zip(args.source_root, source_zip, source_name)

    if args.launcher:
        with tempfile.TemporaryDirectory(prefix="pigeon-score-scan-release-") as temp_name:
            windows_root = Path(temp_name) / windows_name
            copy_release_tree(args.source_root, windows_root, portable=True)
            shutil.copy2(args.launcher, windows_root / PRODUCT_EXE)
            runtime = windows_root / "runtime"
            runtime.mkdir(parents=True, exist_ok=True)
            if args.offline_runtime is None:
                raise ValueError("--offline-runtime is required for a Windows release")
            offline_payload = copy_offline_runtime(args.offline_runtime, runtime)
            bootstrap_manifest: dict[str, object] = {
                "format": 2,
                "launcher_sha256": sha256(windows_root / PRODUCT_EXE),
                "start_cmd_sha256": sha256(runtime / "start.cmd"),
                "runner_sha256": sha256(runtime / "run_scorescan.py"),
                "runtime_delivery": "offline-bundled",
                "network_required": False,
                "offline_manifest_sha256": sha256(runtime / "offline_manifest.json"),
                "python_version": offline_payload["python"]["version"],
                "python_sha256": sha256(runtime / "python" / "python.exe"),
            }
            (runtime / "bootstrap_manifest.json").write_text(
                json.dumps(bootstrap_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            deterministic_zip(
                windows_root,
                windows_zip,
                windows_name,
                allow_generated=True,
            )

    hashes = [f"{sha256(source_zip)}  {source_zip.name}"]
    if windows_zip.exists():
        hashes.append(f"{sha256(windows_zip)}  {windows_zip.name}")
    (args.output_dir / f"{PRODUCT_SLUG}-{version_label}-SHA256.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
