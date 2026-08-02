from __future__ import annotations

import ctypes
import os
import shutil
import sys
import threading
import time
from pathlib import Path


class _FileTime(ctypes.Structure):
    _fields_ = [("low", ctypes.c_ulong), ("high", ctypes.c_ulong)]

    @property
    def ticks(self) -> int:
        return (int(self.high) << 32) | int(self.low)


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ulong),
        ("memory_load", ctypes.c_ulong),
        ("total_physical", ctypes.c_ulonglong),
        ("available_physical", ctypes.c_ulonglong),
        ("total_page_file", ctypes.c_ulonglong),
        ("available_page_file", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("available_virtual", ctypes.c_ulonglong),
        ("available_extended_virtual", ctypes.c_ulonglong),
    ]


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("page_fault_count", ctypes.c_ulong),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
        ("quota_non_paged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
    ]


class RuntimeMetrics:
    """Small, dependency-free sampler for the local status display."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.started_at = time.monotonic()
        self._lock = threading.Lock()
        self._previous_cpu: tuple[int, int] | None = None
        self._last_cpu_percent: float | None = None
        self._prime_cpu_counter()

    def _windows_cpu_counter(self) -> tuple[int, int] | None:
        if os.name != "nt" or not hasattr(ctypes, "windll"):
            return None
        idle = _FileTime()
        kernel = _FileTime()
        user = _FileTime()
        if not ctypes.windll.kernel32.GetSystemTimes(  # type: ignore[attr-defined]
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        ):
            return None
        return idle.ticks, kernel.ticks + user.ticks

    def _proc_cpu_counter(self) -> tuple[int, int] | None:
        if not sys.platform.startswith("linux"):
            return None
        try:
            values = Path("/proc/stat").read_text(encoding="ascii").splitlines()[0].split()[1:]
            ticks = [int(value) for value in values]
        except (OSError, ValueError, IndexError):
            return None
        idle = ticks[3] + (ticks[4] if len(ticks) > 4 else 0)
        return idle, sum(ticks)

    def _cpu_counter(self) -> tuple[int, int] | None:
        return self._windows_cpu_counter() or self._proc_cpu_counter()

    def _prime_cpu_counter(self) -> None:
        self._previous_cpu = self._cpu_counter()

    def _sample_cpu(self) -> float | None:
        current = self._cpu_counter()
        previous = self._previous_cpu
        self._previous_cpu = current
        if current is None or previous is None:
            return self._last_cpu_percent
        idle_delta = current[0] - previous[0]
        total_delta = current[1] - previous[1]
        if total_delta <= 0:
            return self._last_cpu_percent
        self._last_cpu_percent = round(max(0.0, min(100.0, 100.0 * (total_delta - idle_delta) / total_delta)), 1)
        return self._last_cpu_percent

    @staticmethod
    def _windows_memory() -> tuple[int | None, int | None, float | None]:
        if os.name != "nt" or not hasattr(ctypes, "windll"):
            return None, None, None
        status = _MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return None, None, None
        return int(status.total_physical), int(status.available_physical), float(status.memory_load)

    @staticmethod
    def _linux_memory() -> tuple[int | None, int | None, float | None]:
        if not sys.platform.startswith("linux"):
            return None, None, None
        try:
            rows = {}
            for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
                name, value = line.split(":", 1)
                rows[name] = int(value.strip().split()[0]) * 1024
            total = rows["MemTotal"]
            available = rows.get("MemAvailable", rows.get("MemFree", 0))
        except (OSError, ValueError, KeyError):
            return None, None, None
        percent = 100.0 * (total - available) / total if total else None
        return total, available, round(percent, 1) if percent is not None else None

    @staticmethod
    def _process_memory() -> int | None:
        if os.name == "nt" and hasattr(ctypes, "windll"):
            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            process = kernel32.GetCurrentProcess()
            psapi.GetProcessMemoryInfo.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_ProcessMemoryCounters),
                ctypes.c_ulong,
            ]
            psapi.GetProcessMemoryInfo.restype = ctypes.c_int
            if psapi.GetProcessMemoryInfo(
                process, ctypes.byref(counters), counters.cb
            ):
                return int(counters.working_set_size)
            return None
        if sys.platform.startswith("linux"):
            try:
                resident_pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
                return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
            except (OSError, ValueError, IndexError):
                return None
        return None

    def sample(self) -> dict[str, int | float | None]:
        with self._lock:
            total_memory, available_memory, memory_percent = self._windows_memory()
            if total_memory is None:
                total_memory, available_memory, memory_percent = self._linux_memory()
            try:
                disk = shutil.disk_usage(self.workspace)
                disk_free = int(disk.free)
            except OSError:
                disk_free = None
            return {
                "system_cpu_percent": self._sample_cpu(),
                "memory_percent": memory_percent,
                "memory_total_bytes": total_memory,
                "memory_available_bytes": available_memory,
                "process_memory_bytes": self._process_memory(),
                "workspace_free_bytes": disk_free,
                "service_uptime_seconds": round(time.monotonic() - self.started_at, 1),
            }
