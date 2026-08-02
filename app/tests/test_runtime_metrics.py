from __future__ import annotations

import time

from scorescan.runtime_metrics import RuntimeMetrics


def test_runtime_metrics_are_bounded_and_dependency_free(tmp_path) -> None:
    sampler = RuntimeMetrics(tmp_path)
    time.sleep(0.01)
    payload = sampler.sample()

    assert set(payload) == {
        "system_cpu_percent",
        "memory_percent",
        "memory_total_bytes",
        "memory_available_bytes",
        "process_memory_bytes",
        "workspace_free_bytes",
        "service_uptime_seconds",
    }
    if payload["system_cpu_percent"] is not None:
        assert 0 <= payload["system_cpu_percent"] <= 100
    if payload["memory_percent"] is not None:
        assert 0 <= payload["memory_percent"] <= 100
    assert payload["process_memory_bytes"] is None or payload["process_memory_bytes"] > 0
    assert payload["workspace_free_bytes"] is not None
    assert payload["workspace_free_bytes"] > 0
    assert payload["service_uptime_seconds"] >= 0
