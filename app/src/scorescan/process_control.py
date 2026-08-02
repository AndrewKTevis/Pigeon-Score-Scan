from __future__ import annotations

"""Cross-platform subprocess containment used by native/model workers."""

import os
import signal
import subprocess
import time


def popen_group_options() -> dict[str, object]:
    if os.name == "nt":
        flags = subprocess.CREATE_NO_WINDOW | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return {"creationflags": flags}
    return {"start_new_session": True}


def terminate_process_tree(process: subprocess.Popen[object], grace_seconds: float = 4.0) -> None:
    """Terminate a worker and its descendants, escalating after a short grace period."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            process.terminate()
        except OSError:
            return
        deadline = time.monotonic() + grace_seconds
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if process.poll() is None:
            # taskkill /T is the only dependency-free way to reliably stop native
            # grandchildren spawned by ONNX/TensorFlow workers on Windows.
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        return

    try:
        group_id = os.getpgid(process.pid)
    except (ProcessLookupError, OSError):
        return
    try:
        os.killpg(group_id, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    deadline = time.monotonic() + grace_seconds
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if process.poll() is None:
        try:
            os.killpg(group_id, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
