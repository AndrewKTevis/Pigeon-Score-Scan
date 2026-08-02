from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from app.tools import acquire_pdmx_mxl_archive as module


def test_direct_script_entry_point_loads_project_modules() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "acquire_pdmx_mxl_archive.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0
    assert "output_dir" in completed.stdout


def test_pinned_mxl_identity_is_bounded_and_exact() -> None:
    assert module.EXPECTED_BYTES < module.MAXIMUM_BYTES
    assert len(module.EXPECTED_MD5) == 32
    assert module.URL.endswith("/mxl.tar.gz/content")
