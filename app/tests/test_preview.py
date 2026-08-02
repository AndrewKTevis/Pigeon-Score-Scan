from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import scorescan.preview as preview_module
from scorescan.preview import _render_preview, render_preview


class _FakeToolkit:
    def __init__(self, *, file_ok: bool, data_ok: bool) -> None:
        self.file_ok = file_ok
        self.data_ok = data_ok

    def setOptions(self, _options) -> None:
        pass

    def loadFile(self, _path: str) -> bool:
        return self.file_ok

    def loadData(self, _xml: str) -> bool:
        return self.data_ok

    def getPageCount(self) -> int:
        return 1

    def renderToSVG(self, _page: int) -> str:
        return "<svg xmlns='http://www.w3.org/2000/svg'/>"

    def getLog(self) -> str:
        return ""


def test_preview_retries_file_load_through_memory(monkeypatch, tmp_path: Path) -> None:
    toolkits = iter(
        [
            _FakeToolkit(file_ok=False, data_ok=False),
            _FakeToolkit(file_ok=False, data_ok=True),
        ]
    )
    fake_verovio = SimpleNamespace(toolkit=lambda: next(toolkits))
    monkeypatch.setitem(__import__("sys").modules, "verovio", fake_verovio)
    musicxml = tmp_path / "score.musicxml"
    musicxml.write_text("<score-partwise version='4.0'/>", encoding="utf-8")

    preview, warnings = _render_preview(musicxml, tmp_path / "preview")

    assert warnings == []
    assert preview is not None
    assert preview.exists()


def test_preview_uses_isolated_worker_result(monkeypatch, tmp_path: Path) -> None:
    musicxml = tmp_path / "score.musicxml"
    musicxml.write_text("<score-partwise version='4.0'/>", encoding="utf-8")
    expected = tmp_path / "preview" / "preview.html"

    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured.update(kwargs)
        result_path = Path(command[-1])
        expected.parent.mkdir(parents=True, exist_ok=True)
        expected.write_text("<!doctype html>", encoding="utf-8")
        result_path.write_text(
            json.dumps(
                {
                    "format": 1,
                    "preview_path": str(expected),
                    "warnings": [],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(preview_module.subprocess, "run", fake_run)

    preview, warnings = render_preview(musicxml, tmp_path / "preview")

    assert preview == expected
    assert warnings == []
    assert str(Path(preview_module.__file__).resolve().parents[1]) in str(
        captured["env"]["PYTHONPATH"]  # type: ignore[index]
    )
    assert not list(expected.parent.glob(".preview-worker-*.json"))


def test_preview_native_worker_crash_is_nonfatal(monkeypatch, tmp_path: Path) -> None:
    musicxml = tmp_path / "score.musicxml"
    musicxml.write_text("<score-partwise version='4.0'/>", encoding="utf-8")
    monkeypatch.setattr(
        preview_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=-1073741819,
            stdout="",
        ),
    )

    preview, warnings = render_preview(musicxml, tmp_path / "preview")

    assert preview is None
    assert len(warnings) == 1
    assert "异常退出" in warnings[0]
    assert "转换结果未受影响" in warnings[0]
