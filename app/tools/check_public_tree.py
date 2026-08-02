from __future__ import annotations

"""Reject private, generated or oversized files before publication."""

import argparse
import re
from pathlib import Path


MAX_FILE_BYTES = 20 * 1024 * 1024
FORBIDDEN_ROOTS = {
    ".venv",
    "development_reports",
    "test_materials",
    "tmp",
    "training_data",
    "workspace",
}
FORBIDDEN_DIRECTORIES = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "site-packages",
    "uv-cache",
    "venv",
}
FORBIDDEN_NAMES = {
    ".env",
    "RELEASE_QA.json",
    "RELEASE_READINESS.json",
    "ScoreScan.exe",
    "bootstrap_manifest.json",
    "desktop.active",
    "launcher.exe",
    "launcher.log",
    "offline_manifest.json",
    "pigeon-score-scan.exe",
    "ready.txt",
    "server.lock",
    "start.failed",
    "uv.exe",
    "uv.sha256",
}
TEXT_SUFFIXES = {
    ".cmd",
    ".css",
    ".html",
    ".js",
    ".json",
    ".jsonl",
    ".lock",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".tsv",
    ".xml",
    ".yaml",
    ".yml",
    ".zig",
}
TEXT_NAMES = {"LICENSE", "VERSION"}
PRIVATE_PATTERNS = {
    "Windows user path": re.compile(
        r"[A-Za-z]:(?:\\{1,2})Users(?:\\{1,2})[^\\/\r\n]+",
        re.IGNORECASE,
    ),
    "WSL user UNC path": re.compile(
        r"\\{2,4}wsl(?:\.localhost|\$)\\{1,2}[^\\/\r\n]+"
        r"\\{1,2}home\\{1,2}[^\\/\r\n]+",
        re.IGNORECASE,
    ),
    "WSL user path": re.compile(r"/mnt/[a-z]/Users/[^/\r\n]+", re.IGNORECASE),
    "Unix home path": re.compile(r"/home/[A-Za-z0-9._-]+/"),
    "GitHub token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
}
UNPINNED_ACTION = re.compile(r"^\s*-?\s*uses:\s*[^\s]+@(?:main|master|v\d+(?:\.\d+)*)\s*$", re.MULTILINE)


def audit(root: Path) -> list[str]:
    root = root.resolve()
    problems: list[str] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().casefold()):
        relative = path.relative_to(root)
        if ".git" in relative.parts:
            continue
        if relative.parts and relative.parts[0] in FORBIDDEN_ROOTS:
            problems.append(f"forbidden root: {relative.as_posix()}")
            continue
        if path.is_dir():
            if path.name in FORBIDDEN_DIRECTORIES:
                problems.append(f"generated directory: {relative.as_posix()}")
            continue
        if (
            path.name in FORBIDDEN_NAMES
            or (path.name.startswith(".env.") and path.name != ".env.example")
            or path.suffix.lower() in {".pyc", ".pyo", ".zip", ".7z"}
        ):
            problems.append(f"generated/private file: {relative.as_posix()}")
            continue
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            problems.append(f"file exceeds 20 MiB: {relative.as_posix()} ({size} bytes)")
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in TEXT_NAMES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append(f"text file is not UTF-8: {relative.as_posix()}")
            continue
        for label, pattern in PRIVATE_PATTERNS.items():
            if pattern.search(text):
                problems.append(f"{label}: {relative.as_posix()}")
        if relative.parts[:2] == (".github", "workflows") and UNPINNED_ACTION.search(text):
            problems.append(f"GitHub Action is not pinned to a commit: {relative.as_posix()}")
    return sorted(set(problems))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    problems = audit(args.root)
    if problems:
        for problem in problems:
            print(problem)
        raise SystemExit(1)
    print("Public tree audit passed")


if __name__ == "__main__":
    main()
