from __future__ import annotations

import sys
from types import SimpleNamespace

from scorescan.accelerator import probe_accelerator


def test_runtime_selects_cpu_even_if_cuda_provider_is_visible(monkeypatch) -> None:
    fake_ort = SimpleNamespace(
        __version__="1.test",
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(
        "scorescan.accelerator._distribution_installed",
        lambda name: name == "onnxruntime",
    )

    status = probe_accelerator("auto")

    assert status.requested == "cpu"
    assert status.selected == "cpu"
    assert status.available_providers == ("CPUExecutionProvider",)
    assert status.homr_gpu_argument == "no"
    assert not status.cuda_available


def test_removed_cuda_request_is_observable_and_stays_on_cpu(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            __version__="1.test",
            get_available_providers=lambda: ["CPUExecutionProvider"],
        ),
    )
    monkeypatch.setattr(
        "scorescan.accelerator._distribution_installed",
        lambda name: name == "onnxruntime",
    )

    status = probe_accelerator("cuda")

    assert status.invalid_request == "cuda"
    assert status.requested == "cpu"
    assert status.selected == "cpu"


def test_cpu_runtime_loader_failure_is_reported(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    monkeypatch.setattr(
        "scorescan.accelerator._distribution_installed",
        lambda name: name == "onnxruntime",
    )

    status = probe_accelerator("cpu")

    assert status.selected == "cpu"
    assert status.probe_error is not None
