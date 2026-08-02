from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

APP_NAME = "Pigeon Score Scan"
APP_VERSION = "0.37.0-dev"
WORKFLOW_VERSION = "printed-full-score-scan@1"


@dataclass(frozen=True)
class Settings:
    root: Path
    workspace: Path
    runtime: Path
    resources: Path
    max_upload_bytes: int = 1024 * 1024 * 1024  # 1 GiB per task
    pdf_dpi: int = 400
    minimum_pdf_dpi: int = 240
    page_timeout_seconds: int = 45 * 60
    job_retention_days: int = 30
    max_parallel_jobs: int = 1
    max_pages_per_job: int = 500
    max_image_pixels_per_page: int = 100_000_000
    max_pdf_render_pixels_per_page: int = 100_000_000
    minimum_free_space_bytes: int = 2 * 1024 * 1024 * 1024
    max_workspace_bytes: int = 50 * 1024 * 1024 * 1024
    page_spool_bytes_per_pixel: int = 4

    @classmethod
    def from_root(cls, root: Path) -> "Settings":
        package_dir = Path(__file__).resolve().parent
        return cls(
            root=root,
            workspace=root / "workspace",
            runtime=root / "runtime",
            resources=package_dir / "resources",
        )
