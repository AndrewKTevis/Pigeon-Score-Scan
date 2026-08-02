from __future__ import annotations

import threading
from pathlib import Path

from PIL import Image

from .util import atomic_write_text


def _tray_image() -> Image.Image:
    # The tray API requires pixels. Keep this as a neutral status tile rather
    # than introducing a product-specific logo or music symbol.
    return Image.new("RGBA", (64, 64), (69, 75, 82, 255))


class DesktopController:
    def __init__(
        self,
        shutdown_event: threading.Event,
    ) -> None:
        self.shutdown_event = shutdown_event
        self.window = None
        self.tray = None
        self.exit_requested = threading.Event()
        self._tray_started = False
        self._tray_notice_sent = False
        self._lock = threading.RLock()
        self._signal_stop = threading.Event()
        self._signal_started = False
        self._close_dialog_scheduled = False

    def attach_window(self, window) -> None:
        self.window = window

    def _hide_to_tray(self) -> None:
        if self.window is not None:
            self.window.hide()
        tray = self.tray
        if tray is not None and not self._tray_notice_sent:
            try:
                tray.notify(
                    "Minimized to tray",
                    "Pigeon Score Scan",
                )
            except Exception:
                pass
            self._tray_notice_sent = True

    def _request_full_exit(self) -> None:
        self.exit_requested.set()
        self.shutdown_event.set()
        if self.window is not None:
            self.window.destroy()

    def show_close_dialog(self) -> None:
        window = self.window
        if window is None:
            return
        try:
            window.evaluate_js("window.PigeonScoreScan?.showCloseDialog?.()")
        except Exception:
            # Leave the window open if its page is still loading. Closing it without
            # a visible choice would violate the desktop close contract.
            return

    def schedule_close_dialog(self) -> None:
        """Show the web dialog after the native close callback has returned.

        Edge WebView can stop dispatching pointer events when JavaScript evaluation is
        performed re-entrantly from its native closing callback.  Deferring the call
        keeps the close veto synchronous while the page remains interactive.
        """

        with self._lock:
            if self._close_dialog_scheduled:
                return
            self._close_dialog_scheduled = True

        def show() -> None:
            try:
                self.show_close_dialog()
            finally:
                with self._lock:
                    self._close_dialog_scheduled = False

        timer = threading.Timer(0.01, show)
        timer.daemon = True
        timer.start()

    def minimize_to_tray(self) -> dict[str, str]:
        self._hide_to_tray()
        return {"choice": "tray"}

    def exit_completely(self) -> dict[str, str]:
        self._request_full_exit()
        return {"choice": "exit"}

    def on_window_closing(self, *_args) -> bool:
        if self.exit_requested.is_set():
            return True
        self.schedule_close_dialog()
        return False

    def show_window(self, *_args) -> None:
        window = self.window
        if window is None:
            return
        window.show()
        window.restore()

    def exit_from_tray(self, *_args) -> None:
        self._request_full_exit()

    def start_tray(self) -> None:
        with self._lock:
            if self._tray_started:
                return
            import pystray

            menu = pystray.Menu(
                pystray.MenuItem("Open Pigeon Score Scan", self.show_window, default=True),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Exit completely", self.exit_from_tray),
            )
            self.tray = pystray.Icon(
                "Pigeon Score Scan",
                _tray_image(),
                "Pigeon Score Scan",
                menu,
            )
            self._tray_started = True
            threading.Thread(
                target=self.tray.run,
                name="scorescan-tray",
                daemon=True,
            ).start()

    def stop_tray(self) -> None:
        tray = self.tray
        if tray is not None:
            try:
                tray.stop()
            except Exception:
                pass

    def start_show_signal_watcher(self, runtime: Path) -> None:
        with self._lock:
            if self._signal_started:
                return
            self._signal_started = True
        signal = runtime / "show-window.signal"
        active = runtime / "desktop.active"
        atomic_write_text(active, "Pigeon Score Scan desktop shell\n")

        def watch() -> None:
            while not self._signal_stop.wait(0.2):
                if not signal.exists():
                    continue
                signal.unlink(missing_ok=True)
                self.show_window()

        threading.Thread(
            target=watch,
            name="scorescan-desktop-signal",
            daemon=True,
        ).start()

    def stop_show_signal_watcher(self, runtime: Path) -> None:
        self._signal_stop.set()
        (runtime / "show-window.signal").unlink(missing_ok=True)
        (runtime / "desktop.active").unlink(missing_ok=True)


class DesktopApi:
    """Minimal JS bridge; keep native window and tray objects out of reflection."""

    def __init__(self, controller: DesktopController) -> None:
        self._controller = controller

    def minimize_to_tray(self) -> dict[str, str]:
        return self._controller.minimize_to_tray()

    def exit_completely(self) -> dict[str, str]:
        return self._controller.exit_completely()


def run_desktop(
    url: str,
    shutdown_event: threading.Event,
    runtime: Path,
) -> None:
    """Run the loopback UI in a native Edge WebView window with a tray menu."""

    import webview

    webview.settings["ALLOW_DOWNLOADS"] = True
    webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True
    controller = DesktopController(shutdown_event)
    window = webview.create_window(
        "Pigeon Score Scan",
        url,
        js_api=DesktopApi(controller),
        width=1180,
        height=860,
        min_size=(820, 640),
        background_color="#ececea",
        text_select=True,
    )
    if window is None:
        raise RuntimeError("Could not create the Pigeon Score Scan desktop window")
    controller.attach_window(window)
    window.events.closing += controller.on_window_closing

    def watch_shutdown() -> None:
        shutdown_event.wait()
        if controller.exit_requested.is_set():
            return
        controller.exit_requested.set()
        window.destroy()

    threading.Thread(
        target=watch_shutdown,
        name="scorescan-desktop-shutdown",
        daemon=True,
    ).start()

    def on_shown(*_args) -> None:
        controller.start_tray()
        controller.start_show_signal_watcher(runtime)

    window.events.shown += on_shown
    try:
        webview.start(
            gui="edgechromium",
            private_mode=True,
            storage_path=str(runtime / "webview-data"),
        )
    finally:
        controller.stop_show_signal_watcher(runtime)
        controller.stop_tray()
        shutdown_event.set()
