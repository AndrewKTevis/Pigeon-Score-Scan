from __future__ import annotations

import threading
import time
from pathlib import Path

from scorescan.desktop import DesktopController


class FakeWindow:
    def __init__(self) -> None:
        self.hidden = 0
        self.shown = 0
        self.restored = 0
        self.destroyed = 0
        self.scripts: list[str] = []

    def hide(self) -> None:
        self.hidden += 1

    def show(self) -> None:
        self.shown += 1

    def restore(self) -> None:
        self.restored += 1

    def destroy(self) -> None:
        self.destroyed += 1

    def evaluate_js(self, script: str) -> None:
        self.scripts.append(script)


def test_window_close_opens_in_app_choice_without_stopping_service() -> None:
    shutdown = threading.Event()
    controller = DesktopController(shutdown)
    window = FakeWindow()
    controller.attach_window(window)

    allow_close = controller.on_window_closing()

    deadline = time.monotonic() + 1.0
    while not window.scripts and time.monotonic() < deadline:
        time.sleep(0.01)

    assert allow_close is False
    assert window.hidden == 0
    assert window.scripts == ["window.PigeonScoreScan?.showCloseDialog?.()"]
    assert not shutdown.is_set()


def test_window_close_can_exit_the_entire_service() -> None:
    shutdown = threading.Event()
    controller = DesktopController(shutdown)
    window = FakeWindow()
    controller.attach_window(window)

    result = controller.exit_completely()

    assert result == {"choice": "exit"}
    assert shutdown.is_set()
    assert controller.exit_requested.is_set()
    assert window.destroyed == 1


def test_web_tray_button_hides_without_stopping_service() -> None:
    shutdown = threading.Event()
    controller = DesktopController(shutdown)
    window = FakeWindow()
    controller.attach_window(window)

    result = controller.minimize_to_tray()

    assert result == {"choice": "tray"}
    assert window.hidden == 1
    assert not shutdown.is_set()


def test_tray_can_restore_window() -> None:
    controller = DesktopController(threading.Event())
    window = FakeWindow()
    controller.attach_window(window)

    controller.show_window()

    assert window.shown == 1
    assert window.restored == 1


def test_second_launch_signal_restores_existing_desktop_window(tmp_path: Path) -> None:
    controller = DesktopController(threading.Event())
    window = FakeWindow()
    controller.attach_window(window)
    controller.start_show_signal_watcher(tmp_path)
    try:
        assert (tmp_path / "desktop.active").is_file()
        (tmp_path / "show-window.signal").write_text("show\n", encoding="utf-8")
        deadline = time.monotonic() + 2.0
        while window.shown == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert window.shown == 1
        assert window.restored == 1
    finally:
        controller.stop_show_signal_watcher(tmp_path)
    assert not (tmp_path / "desktop.active").exists()
