from __future__ import annotations

from scorescan import cpu_runtime
from scorescan.cpu_runtime import (
    configure_cpu_environment,
    cpu_thread_budget,
    rapidocr_cpu_parameters,
)


def test_default_thread_budget_uses_half_the_machine_with_cap() -> None:
    assert cpu_thread_budget(1, None) == 1
    assert cpu_thread_budget(4, None) == 2
    assert cpu_thread_budget(22, None) == 11
    assert cpu_thread_budget(64, None) == 12


def test_explicit_thread_budget_is_validated_and_bounded() -> None:
    assert cpu_thread_budget(8, "4") == 4
    assert cpu_thread_budget(8, "999") == 8
    assert cpu_thread_budget(32, "999") == 16
    assert cpu_thread_budget(8, "0") == 1
    assert cpu_thread_budget(8, "invalid") == 4


def test_native_libraries_share_one_cpu_budget() -> None:
    env = configure_cpu_environment({"SCORESCAN_ENGINE_THREADS": "3", "OMP_NUM_THREADS": "99"})
    for name in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
    ):
        assert env[name] == "3"


def test_rapidocr_is_explicitly_cpu_and_avoids_interop_oversubscription(monkeypatch) -> None:
    monkeypatch.setattr(cpu_runtime.os, "cpu_count", lambda: 8)
    monkeypatch.setenv("SCORESCAN_ENGINE_THREADS", "5")
    parameters = rapidocr_cpu_parameters()
    assert parameters["EngineConfig.onnxruntime.use_cuda"] is False
    assert parameters["EngineConfig.onnxruntime.intra_op_num_threads"] == 5
    assert parameters["EngineConfig.onnxruntime.inter_op_num_threads"] == 1
