from __future__ import annotations

"""CPU runtime status retained for report and diagnostic compatibility."""

import importlib.metadata
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AcceleratorStatus:
    requested: str
    selected: str
    onnxruntime_version: str | None
    available_providers: tuple[str, ...]
    cpu_package_installed: bool
    gpu_package_installed: bool = False
    package_conflict: bool = False
    fallback_reason: str | None = None
    probe_error: str | None = None
    invalid_request: str | None = None
    cuda_native_load_ok: bool = False
    cuda_probe_error: str | None = None

    @property
    def cuda_available(self) -> bool:
        return False

    @property
    def cuda_provider_listed(self) -> bool:
        return False

    @property
    def homr_gpu_argument(self) -> str:
        return "no"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            cuda_provider_listed=False,
            cuda_available=False,
            homr_gpu_argument="no",
        )
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AcceleratorStatus":
        return cls(
            requested=str(payload.get("requested", "cpu")),
            selected="cpu",
            onnxruntime_version=(str(payload["onnxruntime_version"]) if payload.get("onnxruntime_version") is not None else None),
            available_providers=tuple(str(item) for item in payload.get("available_providers", ())),
            cpu_package_installed=bool(payload.get("cpu_package_installed", False)),
            probe_error=(str(payload["probe_error"]) if payload.get("probe_error") is not None else None),
            invalid_request=(str(payload["invalid_request"]) if payload.get("invalid_request") is not None else None),
        )


def _distribution_installed(name: str) -> bool:
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def preload_onnxruntime_cuda_dlls() -> None:
    """Compatibility no-op; the published runtime does not load CUDA libraries."""


def probe_accelerator(requested: str | None = None) -> AcceleratorStatus:
    raw = str(requested or "cpu").strip().casefold()
    invalid = raw if raw not in {"cpu", "auto"} else None
    version: str | None = None
    providers: tuple[str, ...] = ()
    error: str | None = None
    try:
        import onnxruntime as ort  # type: ignore

        version = str(getattr(ort, "__version__", "") or "") or None
        providers = tuple(
            provider
            for provider in (str(item) for item in ort.get_available_providers())
            if provider == "CPUExecutionProvider"
        )
    except Exception as exc:  # pragma: no cover - native loader errors vary by host
        error = f"{type(exc).__name__}: {exc}"
    return AcceleratorStatus(
        requested="cpu",
        selected="cpu",
        onnxruntime_version=version,
        available_providers=providers,
        cpu_package_installed=_distribution_installed("onnxruntime"),
        probe_error=error,
        invalid_request=invalid,
    )
