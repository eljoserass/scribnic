"""Sample application and owned runtime processes with portable psutil APIs."""

import os
import threading
from time import monotonic
from typing import Callable

import psutil
from opentelemetry import context

from .observability import meter


class ResourceMonitor:
    def __init__(
        self,
        run_id: str,
        runtime_pids: Callable[[], tuple[int, ...]],
        interval: float = 0.25,
    ):
        self.run_id = run_id
        self.runtime_pids = runtime_pids
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self._entries: dict[int, dict] = {}
        self._context = context.get_current()
        metrics = meter()
        self._rss = metrics.create_gauge("scribnic.process.memory.usage", unit="By")
        self._cpu = metrics.create_gauge("scribnic.process.cpu.cores", unit="1")

    def __enter__(self) -> "ResourceMonitor":
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._sample()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def _sample(self) -> None:
        for pid in {os.getpid(), *self.runtime_pids()}:
            try:
                self._sample_process(pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    def _sample_process(self, pid: int) -> None:
        if pid not in self._entries:
            process = psutil.Process(pid)
            times = process.cpu_times()
            process.cpu_percent()
            self._entries[pid] = {
                "process": process,
                "started": monotonic(),
                "initial_cpu": times.user + times.system,
                "peak_rss": 0,
                "role": "application" if pid == os.getpid() else "runtime",
            }
        entry = self._entries[pid]
        process = entry["process"]
        rss = process.memory_info().rss
        cpu = process.cpu_percent() / 100
        times = process.cpu_times()
        entry["peak_rss"] = max(entry["peak_rss"], rss)
        entry["cpu_seconds"] = times.user + times.system - entry["initial_cpu"]
        entry["elapsed_seconds"] = monotonic() - entry["started"]
        attributes = {
            "scribnic.run.id": self.run_id,
            "process.pid": pid,
            "role": entry["role"],
        }
        self._rss.set(rss, attributes, self._context)
        self._cpu.set(cpu, attributes, self._context)

    def summary(self) -> dict[str, object]:
        processes = []
        for pid, entry in self._entries.items():
            elapsed = entry.get("elapsed_seconds", 0)
            seconds = entry.get("cpu_seconds", 0)
            processes.append(
                {
                    "pid": pid,
                    "role": entry["role"],
                    "peak_sampled_rss_bytes": entry["peak_rss"],
                    "cpu_seconds": round(seconds, 4),
                    "average_cpu_cores": round(seconds / elapsed, 3) if elapsed else 0,
                }
            )
        return {
            "sample_interval_seconds": self.interval,
            "scope": "application and owned local runtime processes",
            "logical_cpus": psutil.cpu_count(),
            "system_memory_bytes": psutil.virtual_memory().total,
            "processes": processes,
        }
