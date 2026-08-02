from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CompatibilityResult:
    checked: bool
    success: bool | None
    executable: str | None = None
    message: str | None = None


def find_musescore() -> Path | None:
    candidates: list[Path] = []
    if os.name == "nt":
        for env_name in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
            base_value = os.environ.get(env_name)
            if not base_value:
                continue
            base = Path(base_value)
            candidates.extend(
                [
                    base / "MuseScore 4" / "bin" / "MuseScore4.exe",
                    base / "Programs" / "MuseScore 4" / "bin" / "MuseScore4.exe",
                    base / "MuseScore 3" / "bin" / "MuseScore3.exe",
                ]
            )
    for name in ("MuseScore4.exe", "MuseScore3.exe", "mscore", "musescore", "musescore4"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    return next((path for path in candidates if path.exists()), None)


def validate_with_musescore(musicxml_path: Path, work_dir: Path, timeout_seconds: int = 120) -> CompatibilityResult:
    executable = find_musescore()
    if executable is None:
        return CompatibilityResult(False, None, message="未检测到 MuseScore，已跳过本机导入验证")
    work_dir.mkdir(parents=True, exist_ok=True)
    output = work_dir / "musescore_import_check.pdf"
    try:
        completed = subprocess.run(
            [str(executable), "-o", str(output), str(musicxml_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        success = completed.returncode == 0 and output.exists() and output.stat().st_size > 0
        message = "MuseScore 本机导入验证通过" if success else (
            completed.stderr.strip() or completed.stdout.strip() or f"MuseScore 返回代码 {completed.returncode}"
        )
        return CompatibilityResult(True, success, str(executable), message)
    except Exception as exc:
        return CompatibilityResult(True, False, str(executable), f"MuseScore 验证失败：{exc}")
    finally:
        output.unlink(missing_ok=True)
