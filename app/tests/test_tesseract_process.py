from __future__ import annotations

import sys
import time

import scorescan.text_enrichment as module


def test_tesseract_tsv_is_spooled_and_returned() -> None:
    payload = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n5\t1\t1\t1\t1\t1\t10\t20\t30\t12\t95\tmf\n"
    result = module._run_tesseract_tsv(
        [sys.executable, "-c", f"import sys; sys.stdout.write({payload!r})"]
    )
    assert result == payload


def test_tesseract_tsv_rejects_oversized_output(monkeypatch) -> None:
    monkeypatch.setattr(module, "_TESSERACT_MAX_TSV_BYTES", 1024)
    result = module._run_tesseract_tsv(
        [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'x' * 4096); sys.stdout.flush()"]
    )
    assert result is None


def test_tesseract_tsv_times_out_and_returns_promptly(monkeypatch) -> None:
    monkeypatch.setattr(module, "_TESSERACT_TIMEOUT_SECONDS", 0.12)
    monkeypatch.setattr(module, "_TESSERACT_POLL_SECONDS", 0.01)
    started = time.monotonic()
    result = module._run_tesseract_tsv(
        [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    elapsed = time.monotonic() - started
    assert result is None
    assert elapsed < 3.0


def test_tesseract_rows_bounds_row_count_and_text(monkeypatch, tmp_path) -> None:
    header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
    rows = [
        f"5\t1\t1\t1\t1\t{i}\t1\t2\t3\t4\t95\tword{i}\n"
        for i in range(8)
    ]
    rows.insert(1, f"5\t1\t1\t1\t1\t99\t1\t2\t3\t4\t95\t{'x' * 40}\n")
    monkeypatch.setattr(module.shutil, "which", lambda _name: "tesseract")
    monkeypatch.setattr(module, "_tesseract_language_spec", lambda: "eng")
    monkeypatch.setattr(module, "_TESSERACT_MAX_ROWS", 5)
    monkeypatch.setattr(module, "_TESSERACT_MAX_TEXT_CHARS", 16)
    monkeypatch.setattr(module, "_run_tesseract_tsv", lambda _command: header + "".join(rows))

    result = module._tesseract_rows(tmp_path / "page.png")
    assert len(result) == 4
    assert all(len(item[0]) <= 16 for item in result)
