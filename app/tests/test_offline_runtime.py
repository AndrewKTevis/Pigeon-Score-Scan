from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

from scorescan import offline_runtime


def test_bundled_model_inventory_rejects_missing_and_damaged_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    payload = b"model"
    model = offline_runtime.BundledModel(
        "example",
        "model.onnx",
        len(payload),
        hashlib.sha256(payload).hexdigest(),
    )
    monkeypatch.setattr(offline_runtime, "BUNDLED_MODELS", (model,))

    assert offline_runtime.verify_bundled_models(tmp_path, verify_hashes=True) == [
        "missing: example/model.onnx"
    ]
    path = tmp_path / "example" / "model.onnx"
    path.parent.mkdir()
    path.write_bytes(payload)
    assert offline_runtime.verify_bundled_models(tmp_path, verify_hashes=True) == []
    path.write_bytes(b"other")
    assert offline_runtime.verify_bundled_models(tmp_path, verify_hashes=True) == [
        "SHA-256 mismatch: example/model.onnx"
    ]


def test_published_runtime_blocks_external_socket_connections() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(source_root),
            "SCORESCAN_OFFLINE_RUNTIME": "1",
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import socket; "
                "s=socket.socket(); "
                "s.connect(('192.0.2.1', 9))"
            ),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode != 0
    assert "offline runtime blocked a network connection" in result.stderr
