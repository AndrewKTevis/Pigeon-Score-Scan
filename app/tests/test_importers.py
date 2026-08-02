from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PIL import Image

from scorescan.config import Settings
from scorescan.importers import classify_mode, copy_uploads, prepare_pages
from scorescan.models import JobState


def test_multiframe_image_is_streamed_into_ordered_pages(tmp_path: Path) -> None:
    source = tmp_path / "scan.tiff"
    frames = [
        Image.new("RGB", (64, 80), "white"),
        Image.new("RGB", (72, 90), "black"),
        Image.new("RGB", (80, 100), "white"),
    ]
    frames[0].save(source, save_all=True, append_images=frames[1:], compression="raw")

    root = tmp_path / "portable"
    settings = replace(Settings.from_root(root), pdf_dpi=72)
    job_root = settings.workspace / "multi-frame"
    copied = copy_uploads([source], job_root / "input")
    job = JobState("multi-frame", job_root, "images", ["scan.tiff"])

    pages = prepare_pages(job, copied, settings)

    assert [page.index for page in pages] == [1, 2, 3]
    assert [page.source_page_number for page in pages] == [1, 2, 3]
    assert [page.source_name for page in pages] == [
        "scan.tiff · 第 1 帧",
        "scan.tiff · 第 2 帧",
        "scan.tiff · 第 3 帧",
    ]
    assert [Path(page.image_path).stat().st_size > 0 for page in pages] == [True, True, True]


def test_webp_contract_is_classified_and_decoded_as_an_image(tmp_path: Path) -> None:
    source = tmp_path / "scan.WEBP"
    Image.new("RGB", (80, 60), "white").save(source, "WEBP", lossless=True)

    assert classify_mode([source]) == "images"

    root = tmp_path / "portable"
    settings = Settings.from_root(root)
    job_root = settings.workspace / "webp"
    copied = copy_uploads([source], job_root / "input")
    job = JobState("webp", job_root, "images", [source.name])

    pages = prepare_pages(job, copied, settings)

    assert len(pages) == 1
    assert pages[0].width == 80
    assert pages[0].height == 60
    assert Path(pages[0].image_path).is_file()


def test_copy_uploads_can_consume_temporary_uploads(tmp_path: Path) -> None:
    source = tmp_path / "incoming.png"
    source.write_bytes(b"payload")

    copied = copy_uploads([source], tmp_path / "job" / "input", move=True)

    assert not source.exists()
    assert copied[0].read_bytes() == b"payload"


def test_multiframe_image_page_budget_fails_before_writing_partial_pages(tmp_path: Path) -> None:
    source = tmp_path / "many-pages.tiff"
    frames = [Image.new("RGB", (32, 40), "white") for _ in range(3)]
    frames[0].save(source, save_all=True, append_images=frames[1:], compression="raw")

    root = tmp_path / "portable"
    settings = replace(Settings.from_root(root), max_pages_per_job=2)
    job_root = settings.workspace / "page-budget"
    copied = copy_uploads([source], job_root / "input")
    job = JobState("page-budget", job_root, "images", ["many-pages.tiff"])

    import pytest

    with pytest.raises(ValueError, match="超过上限 2 页"):
        prepare_pages(job, copied, settings)

    assert not list((job_root / "pages" / "source").glob("page_*.png"))


def test_image_pixel_budget_is_not_converted_to_placeholder(tmp_path: Path) -> None:
    source = tmp_path / "oversized.png"
    Image.new("RGB", (101, 100), "white").save(source)

    root = tmp_path / "portable"
    settings = replace(Settings.from_root(root), max_image_pixels_per_page=10_000)
    job_root = settings.workspace / "pixel-budget"
    copied = copy_uploads([source], job_root / "input")
    job = JobState("pixel-budget", job_root, "images", ["oversized.png"])

    import pytest

    with pytest.raises(ValueError, match="10,100 像素"):
        prepare_pages(job, copied, settings)

    assert not list((job_root / "pages" / "source").glob("page_*.png"))


def test_pdf_render_pixel_budget_fails_before_pixmap_allocation(tmp_path: Path) -> None:
    import fitz
    import pytest

    source = tmp_path / "large-page.pdf"
    document = fitz.open()
    document.new_page(width=1_000, height=1_000)
    document.save(source)
    document.close()

    root = tmp_path / "portable"
    settings = replace(
        Settings.from_root(root),
        pdf_dpi=72,
        max_pdf_render_pixels_per_page=999_999,
    )
    job_root = settings.workspace / "pdf-pixel-budget"
    copied = copy_uploads([source], job_root / "input")
    job = JobState("pdf-pixel-budget", job_root, "pdf", ["large-page.pdf"])

    with pytest.raises(ValueError, match="1,000,000 像素"):
        prepare_pages(job, copied, settings)

    assert not list((job_root / "pages" / "source").glob("page_*.png"))


def test_pdf_render_adapts_dpi_within_safe_quality_floor(tmp_path: Path) -> None:
    import fitz

    source = tmp_path / "large-page.pdf"
    document = fitz.open()
    document.new_page(width=1_000, height=1_000)
    document.save(source)
    document.close()

    root = tmp_path / "portable"
    settings = replace(
        Settings.from_root(root),
        pdf_dpi=72,
        minimum_pdf_dpi=60,
        max_pdf_render_pixels_per_page=900_000,
        minimum_free_space_bytes=0,
    )
    job_root = settings.workspace / "adaptive-pdf-dpi"
    copied = copy_uploads([source], job_root / "input")
    job = JobState("adaptive-pdf-dpi", job_root, "pdf", ["large-page.pdf"])

    pages = prepare_pages(job, copied, settings)

    assert len(pages) == 1
    assert pages[0].width * pages[0].height <= 900_000
    assert pages[0].render_dpi is not None
    assert 60 <= pages[0].render_dpi < 72
    assert any("自适应" in note for note in pages[0].quality_notes)


def test_pillow_decompression_bomb_is_reported_as_resource_limit(
    tmp_path: Path, monkeypatch,
) -> None:
    import pytest

    source = tmp_path / "bomb.png"
    source.write_bytes(b"not-decoded")

    def reject(_path):
        raise Image.DecompressionBombError("synthetic bomb")

    monkeypatch.setattr("scorescan.importers.Image.open", reject)
    root = tmp_path / "portable"
    settings = Settings.from_root(root)
    job_root = settings.workspace / "bomb"
    copied = copy_uploads([source], job_root / "input")
    job = JobState("bomb", job_root, "images", ["bomb.png"])

    with pytest.raises(ValueError, match="安全解码像素上限"):
        prepare_pages(job, copied, settings)

    assert not list((job_root / "pages" / "source").glob("page_*.png"))


def test_page_expansion_is_atomic_across_multiple_input_files(tmp_path: Path) -> None:
    import pytest

    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (50, 50), "white").save(first)
    Image.new("RGB", (101, 100), "white").save(second)

    root = tmp_path / "portable"
    settings = replace(
        Settings.from_root(root),
        max_image_pixels_per_page=10_000,
        minimum_free_space_bytes=0,
    )
    job_root = settings.workspace / "atomic-pages"
    copied = copy_uploads([first, second], job_root / "input")
    job = JobState("atomic-pages", job_root, "images", ["first.png", "second.png"])

    with pytest.raises(ValueError, match="10,100 像素"):
        prepare_pages(job, copied, settings)

    assert not (job_root / "pages" / "source").exists()
    assert not list((job_root / "pages").glob(".source-*.staging"))


def test_page_commit_restores_previous_pages_when_directory_swap_fails(
    tmp_path: Path, monkeypatch,
) -> None:
    import os
    import pytest

    source = tmp_path / "replacement.png"
    Image.new("RGB", (40, 50), "white").save(source)
    root = tmp_path / "portable"
    settings = replace(Settings.from_root(root), minimum_free_space_bytes=0)
    job_root = settings.workspace / "atomic-swap"
    copied = copy_uploads([source], job_root / "input")
    job = JobState("atomic-swap", job_root, "images", ["replacement.png"])
    previous = job_root / "pages" / "source"
    previous.mkdir(parents=True)
    (previous / "sentinel.txt").write_text("previous", encoding="utf-8")

    real_replace = os.replace
    calls = 0

    def fail_staging_commit(source_path, destination_path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic directory commit failure")
        return real_replace(source_path, destination_path)

    monkeypatch.setattr("scorescan.importers.os.replace", fail_staging_commit)
    with pytest.raises(OSError, match="synthetic directory commit failure"):
        prepare_pages(job, copied, settings)

    assert (previous / "sentinel.txt").read_text(encoding="utf-8") == "previous"
    assert not list((job_root / "pages").glob(".source-*.staging"))
    assert not list((job_root / "pages").glob(".source-*.backup"))
