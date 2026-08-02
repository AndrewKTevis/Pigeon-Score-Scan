from __future__ import annotations

"""Shared CPU inference scheduling for the desktop runtime."""

import os

ENGINE_THREADS_ENVIRONMENT_VARIABLE = "SCORESCAN_ENGINE_THREADS"
_THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
)


def cpu_thread_budget(logical_cpus: int | None = None, requested: str | None = None) -> int:
    """Return a bounded budget for one inference worker.

    Recognition pages run in isolated subprocesses.  Giving one ONNX session every
    logical core causes nested libraries to compete with preprocessing and the UI.
    The default uses roughly half the machine, with a useful floor and a hard cap.
    An explicit environment override remains bounded by the available CPU count.
    """

    available = max(1, int(logical_cpus or os.cpu_count() or 1))
    default = min(12, max(1, (available + 1) // 2))
    raw = requested
    if raw is None:
        raw = os.environ.get(ENGINE_THREADS_ENVIRONMENT_VARIABLE)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError, OverflowError):
        return default
    return max(1, min(available, 16, value))


def configure_cpu_environment(environment: dict[str, str] | None = None) -> dict[str, str]:
    """Apply one consistent thread budget across native numerical libraries."""

    env = dict(os.environ if environment is None else environment)
    value = str(cpu_thread_budget(requested=env.get(ENGINE_THREADS_ENVIRONMENT_VARIABLE)))
    for name in _THREAD_ENVIRONMENT_VARIABLES:
        env[name] = value
    env.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")
    return env


def rapidocr_cpu_parameters() -> dict[str, object]:
    """Keep RapidOCR's three ONNX sessions from oversubscribing one another."""

    return {
        "EngineConfig.onnxruntime.use_cuda": False,
        "EngineConfig.onnxruntime.intra_op_num_threads": cpu_thread_budget(),
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
    }
