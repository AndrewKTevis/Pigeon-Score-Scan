from __future__ import annotations

import math
import os
import shutil
import uuid
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageDraw, ImageOps, ImageSequence

from .config import Settings
from .models import JobState, PageInfo
from .storage import require_workspace_capacity
from .util import sha256_file

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_EXTENSIONS = {".pdf"}


class InputResourceLimitError(ValueError):
    """Raised before decoding when an input exceeds the configured resource budget."""


def _require_page_budget(*, current_pages: int, additional_pages: int, settings: Settings) -> None:
    if additional_pages < 0:
        raise InputResourceLimitError("输入页数无效。")
    total_pages = current_pages + additional_pages
    if total_pages > settings.max_pages_per_job:
        raise InputResourceLimitError(
            f"任务展开后共有 {total_pages} 页，超过上限 {settings.max_pages_per_job} 页。"
        )


def _require_pixel_budget(
    *, width: int, height: int, limit: int, label: str
) -> None:
    if width <= 0 or height <= 0:
        raise InputResourceLimitError(f"{label} 的页面尺寸无效：{width}×{height}。")
    pixels = width * height
    if pixels > limit:
        raise InputResourceLimitError(
            f"{label} 展开后为 {width}×{height}（{pixels:,} 像素），"
            f"超过单页上限 {limit:,} 像素。"
        )


def _pdf_render_plan(
    *,
    page_width_points: float,
    page_height_points: float,
    requested_dpi: float,
    minimum_dpi: float,
    pixel_limit: int,
    label: str,
) -> tuple[float, float, int, int]:
    """Choose the highest safe PDF DPI without silently producing a tiny page."""

    if (
        page_width_points <= 0
        or page_height_points <= 0
        or requested_dpi <= 0
        or minimum_dpi <= 0
        or pixel_limit <= 0
    ):
        raise InputResourceLimitError(f"{label} 的 PDF 页面或渲染设置无效。")
    effective_minimum = min(float(requested_dpi), float(minimum_dpi))

    def dimensions(dpi: float) -> tuple[float, int, int]:
        scale = dpi / 72.0
        return (
            scale,
            max(1, math.ceil(page_width_points * scale)),
            max(1, math.ceil(page_height_points * scale)),
        )

    requested_scale, requested_width, requested_height = dimensions(requested_dpi)
    if requested_width * requested_height <= pixel_limit:
        return float(requested_dpi), requested_scale, requested_width, requested_height

    # The 0.999 factor leaves room for the two independent ceil operations.
    safe_scale = math.sqrt(
        pixel_limit / (page_width_points * page_height_points)
    ) * 0.999
    safe_dpi = min(float(requested_dpi), math.floor(safe_scale * 720.0) / 10.0)
    while safe_dpi >= effective_minimum:
        scale, width, height = dimensions(safe_dpi)
        if width * height <= pixel_limit:
            return safe_dpi, scale, width, height
        safe_dpi = round(safe_dpi - 0.1, 1)

    _require_pixel_budget(
        width=requested_width,
        height=requested_height,
        limit=pixel_limit,
        label=label,
    )
    raise AssertionError("unreachable")


def classify_mode(paths: list[Path]) -> str:
    suffixes = {path.suffix.casefold() for path in paths}
    if suffixes and suffixes <= IMAGE_EXTENSIONS:
        return "images"
    if suffixes and suffixes <= PDF_EXTENSIONS:
        return "pdf"
    raise ValueError("一次任务只能导入图片，或只能导入 PDF，不能混合。")


def _placeholder_page(output: Path, label: str) -> tuple[int, int]:
    width, height = 2480, 3508
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 80, width - 80, height - 80), outline="black", width=4)
    draw.text((160, 180), label, fill="black")
    image.save(output, "PNG", optimize=False)
    return width, height


def copy_uploads(paths: list[Path], target_dir: Path, *, move: bool = False) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for index, source in enumerate(paths, start=1):
        destination = target_dir / f"{index:04d}_{Path(source.name).name}"
        if move:
            shutil.move(str(source), str(destination))
        else:
            shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def list_copied_inputs(job_root: Path) -> list[Path]:
    return sorted((job_root / "input").glob("*"), key=lambda path: path.name)


def prepare_pages(job: JobState, uploaded_paths: list[Path], settings: Settings) -> list[PageInfo]:
    """Expand inputs transactionally so failed decoding never exposes partial pages."""

    final_dir = job.root / "pages" / "source"
    staging_dir = job.root / "pages" / f".source-{uuid.uuid4().hex}.staging"
    staging_dir.mkdir(parents=True, exist_ok=False)
    try:
        pages = _prepare_pages_into(job, uploaded_paths, settings, staging_dir)
        backup_dir = job.root / "pages" / f".source-{uuid.uuid4().hex}.backup"
        had_previous = final_dir.exists()
        if had_previous:
            os.replace(final_dir, backup_dir)
        try:
            os.replace(staging_dir, final_dir)
        except Exception:
            # Directory replacement is atomic on the supported local filesystems, but
            # fail closed even if a third-party filter unexpectedly materialises the
            # destination before the rename returns.  The last committed page set wins.
            if final_dir.exists():
                shutil.rmtree(final_dir, ignore_errors=True)
            if had_previous and backup_dir.exists():
                os.replace(backup_dir, final_dir)
            raise
        else:
            shutil.rmtree(backup_dir, ignore_errors=True)
        for page in pages:
            page.image_path = str(final_dir / Path(page.image_path).name)
        return pages
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def _prepare_pages_into(
    job: JobState, uploaded_paths: list[Path], settings: Settings, pages_dir: Path
) -> list[PageInfo]:
    pages: list[PageInfo] = []
    page_index = 1

    if job.mode == "images":
        for file_index, source in enumerate(uploaded_paths, start=1):
            try:
                with Image.open(source) as image:
                    frame_count = max(1, int(getattr(image, "n_frames", 1)))
                    _require_page_budget(
                        current_pages=page_index - 1,
                        additional_pages=frame_count,
                        settings=settings,
                    )
                    frame_sizes: list[tuple[int, int]] = []
                    for frame_offset in range(frame_count):
                        image.seek(frame_offset)
                        width, height = tuple(map(int, image.size))
                        _require_pixel_budget(
                            width=width,
                            height=height,
                            limit=settings.max_image_pixels_per_page,
                            label=f"图像 {source.name} 第 {frame_offset + 1} 帧",
                        )
                        frame_sizes.append((width, height))
                    required_bytes = sum(
                        width * height * settings.page_spool_bytes_per_pixel
                        for width, height in frame_sizes
                    )
                    require_workspace_capacity(
                        settings,
                        additional_bytes=required_bytes,
                        context=f"展开图像 {source.name}",
                    )
                    image.seek(0)
                    frame_iterator = ImageSequence.Iterator(image)
                    for frame_number, frame in enumerate(frame_iterator, start=1):
                        width, height = frame.size
                        _require_pixel_budget(
                            width=width,
                            height=height,
                            limit=settings.max_image_pixels_per_page,
                            label=f"图像 {source.name} 第 {frame_number} 帧",
                        )
                        output = pages_dir / f"page_{page_index:04d}.png"
                        notes: list[str] = []
                        normalized = ImageOps.exif_transpose(frame).convert("RGB")
                        normalized.save(output, "PNG", optimize=False)
                        width, height = normalized.size
                        label = job.source_files[file_index - 1] if file_index <= len(job.source_files) else source.name
                        if frame_count > 1:
                            label += f" · 第 {frame_number} 帧"
                        pages.append(
                            PageInfo(
                                index=page_index,
                                source_name=label,
                                image_path=str(output),
                                source_file_index=file_index,
                                source_page_number=frame_number,
                                width=width,
                                height=height,
                                sha256=sha256_file(output),
                                quality_notes=notes,
                            )
                        )
                        page_index += 1
            except Image.DecompressionBombError as exc:
                raise InputResourceLimitError(
                    f"图像 {source.name} 超过 Pillow 的安全解码像素上限。"
                ) from exc
            except InputResourceLimitError:
                raise
            except Exception as exc:
                output = pages_dir / f"page_{page_index:04d}.png"
                width, height = _placeholder_page(output, f"Unreadable image: {source.name}")
                pages.append(
                    PageInfo(
                        index=page_index,
                        source_name=source.name,
                        image_path=str(output),
                        source_file_index=file_index,
                        width=width,
                        height=height,
                        sha256=sha256_file(output),
                        quality_notes=[f"图像无法正常解码：{exc}；已继续处理并保留页面"],
                    )
                )
                page_index += 1
        return pages

    for file_index, source in enumerate(uploaded_paths, start=1):
        source_label = job.source_files[file_index - 1] if file_index <= len(job.source_files) else source.name
        try:
            document = fitz.open(source)
        except Exception as exc:
            output = pages_dir / f"page_{page_index:04d}.png"
            width, height = _placeholder_page(output, f"Unreadable PDF: {source.name}")
            pages.append(
                PageInfo(
                    index=page_index,
                    source_name=source_label,
                    image_path=str(output),
                    source_file_index=file_index,
                    width=width,
                    height=height,
                    sha256=sha256_file(output),
                    quality_notes=[f"PDF 无法正常打开：{exc}；已继续处理并保留页面"],
                )
            )
            page_index += 1
            continue

        try:
            expanded_page_count = max(1, int(document.page_count))
            _require_page_budget(
                current_pages=page_index - 1,
                additional_pages=expanded_page_count,
                settings=settings,
            )
            render_plans: list[tuple[float, float, int, int]] = []
            for estimate_page_number, estimate_page in enumerate(document, start=1):
                render_plans.append(
                    _pdf_render_plan(
                        page_width_points=float(estimate_page.rect.width),
                        page_height_points=float(estimate_page.rect.height),
                        requested_dpi=float(settings.pdf_dpi),
                        minimum_dpi=float(settings.minimum_pdf_dpi),
                        pixel_limit=settings.max_pdf_render_pixels_per_page,
                        label=f"PDF {source.name} 第 {estimate_page_number} 页",
                    )
                )
            require_workspace_capacity(
                settings,
                additional_bytes=sum(
                    width * height * settings.page_spool_bytes_per_pixel
                    for _dpi, _scale, width, height in render_plans
                ),
                context=f"展开 PDF {source.name}",
            )
            if document.page_count == 0:
                output = pages_dir / f"page_{page_index:04d}.png"
                width, height = _placeholder_page(output, f"Empty PDF: {source.name}")
                pages.append(
                    PageInfo(
                        index=page_index,
                        source_name=source_label,
                        image_path=str(output),
                        source_file_index=file_index,
                        width=width,
                        height=height,
                        sha256=sha256_file(output),
                        quality_notes=["PDF 没有页面；已保留一个占位页面"],
                    )
                )
                page_index += 1
            for pdf_page_number, page in enumerate(document, start=1):
                output = pages_dir / f"page_{page_index:04d}.png"
                notes: list[str] = []
                render_dpi, scale, _estimated_width, _estimated_height = render_plans[
                    pdf_page_number - 1
                ]
                if render_dpi < float(settings.pdf_dpi):
                    notes.append(
                        f"PDF 第 {pdf_page_number} 页按安全像素上限自适应以 "
                        f"{render_dpi:.1f} DPI 渲染（设定 {settings.pdf_dpi} DPI）"
                    )
                try:
                    pixmap = page.get_pixmap(
                        matrix=fitz.Matrix(scale, scale), alpha=False, colorspace=fitz.csRGB
                    )
                    pixmap.save(output)
                    width, height = pixmap.width, pixmap.height
                except Exception as exc:
                    width, height = _placeholder_page(output, f"Unreadable PDF page {pdf_page_number}")
                    notes.append(f"PDF 第 {pdf_page_number} 页无法渲染：{exc}；已继续处理并保留页面")
                pages.append(
                    PageInfo(
                        index=page_index,
                        source_name=f"{source_label} · 第 {pdf_page_number} 页",
                        image_path=str(output),
                        source_file_index=file_index,
                        source_page_number=pdf_page_number,
                        width=width,
                        height=height,
                        render_dpi=render_dpi,
                        sha256=sha256_file(output),
                        quality_notes=notes,
                    )
                )
                page_index += 1
        finally:
            document.close()
    return pages
