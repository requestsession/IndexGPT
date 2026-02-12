from dataclasses import dataclass
from typing import Callable, Optional
import subprocess
import shutil

try:
    import torch
except Exception:  # pragma: no cover - environment dependent
    torch = None

try:
    import psutil
except Exception:  # pragma: no cover - environment dependent
    psutil = None


@dataclass
class ResourceSnapshot:
    cpu_percent: Optional[float] = None
    system_memory_percent: Optional[float] = None
    gpu_util_percent: Optional[float] = None


class ResourceMonitor:
    def __init__(self):
        if psutil:
            # Prime system CPU measurement to avoid an always-0 first sample.
            psutil.cpu_percent(interval=None)

    def collect(self) -> ResourceSnapshot:
        snapshot = ResourceSnapshot()

        if psutil:
            # Use system-wide CPU utilization so displayed CPU stays in [0, 100].
            snapshot.cpu_percent = psutil.cpu_percent(interval=None)
            snapshot.system_memory_percent = psutil.virtual_memory().percent

        snapshot.gpu_util_percent = self._collect_gpu_util_percent()

        return snapshot

    def _collect_gpu_util_percent(self) -> Optional[float]:
        if shutil.which("nvidia-smi") is None:
            return None
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            values = []
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    values.append(float(line))
                except ValueError:
                    continue
            if not values:
                return None
            # Show the busiest GPU utilization when multiple GPUs are visible.
            return max(values)
        except Exception:
            return None


class ResourceUsageFormatter:
    @staticmethod
    def _pct_or_na(value: Optional[float]) -> str:
        return f"{value:.1f}%" if value is not None else "N/A"

    @staticmethod
    def _gb_or_na(value: Optional[float]) -> str:
        return f"{value:.2f}GB" if value is not None else "N/A"

    @staticmethod
    def to_line(step: int, snapshot: ResourceSnapshot) -> str:
        return (
            f"[Resource] step={step} | "
            f"CPU {ResourceUsageFormatter._pct_or_na(snapshot.cpu_percent)} | "
            f"RAM {ResourceUsageFormatter._pct_or_na(snapshot.system_memory_percent)} | "
            f"GPU {ResourceUsageFormatter._pct_or_na(snapshot.gpu_util_percent)}"
        )


class PeakTracker:
    def __init__(self):
        self.peak_gpu_util_percent: Optional[float] = None

    def update(self, snapshot: ResourceSnapshot) -> None:
        if snapshot.gpu_util_percent is not None:
            if (
                self.peak_gpu_util_percent is None
                or snapshot.gpu_util_percent > self.peak_gpu_util_percent
            ):
                self.peak_gpu_util_percent = snapshot.gpu_util_percent

    def summary_line(self) -> str:
        gpu = ResourceUsageFormatter._pct_or_na(self.peak_gpu_util_percent)
        return f"[ResourceSummary] PeakGPUUtil {gpu}"


class ResourceProgressLogger:
    def __init__(
        self,
        sample_fn: Optional[Callable[[], ResourceSnapshot]] = None,
        sink: Callable[[str], None] = print,
        every_n_steps: int = 10,
    ):
        self._sample_fn = sample_fn or ResourceMonitor().collect
        self._sink = sink
        self._every_n_steps = max(1, every_n_steps)
        self._peaks = PeakTracker()

    def maybe_log(self, step: int, force: bool = False) -> None:
        if not force and step % self._every_n_steps != 0:
            return
        snapshot = self._sample_fn()
        self._peaks.update(snapshot)
        self._sink(ResourceUsageFormatter.to_line(step, snapshot))

    def log_summary(self) -> None:
        self._sink(self._peaks.summary_line())
