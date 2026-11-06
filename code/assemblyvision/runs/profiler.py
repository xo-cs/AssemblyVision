"""Stage profiler (Week 10): wall time per stage and GPU samples, for experiments.

GpuSampler streams `nvidia-smi --query-gpu=... -lms <ms>` in a background
thread (nothing is installed; nvidia-smi comes with the driver). Profiler marks
the stage boundaries, so each sample belongs to a stage, and summary() gives per
stage seconds, mean and max GPU utilisation and peak GPU memory. The inner
breakdowns (pose: decode, preprocess, inference, postprocess, extract, writing;
overlay: render, encode; shots: cuts, motion) are kept by the stages themselves
in their manifest sections and copied in by summary().

nvidia-smi numbers its GPUs in PCI bus order, like CUDA_DEVICE_ORDER=PCI_BUS_ID,
so the run's --gpu index is also the nvidia-smi index.
"""
from __future__ import annotations

import subprocess
import threading
import time
from contextlib import contextmanager

_QUERY = "utilization.gpu,memory.used,memory.total"


def sampler_command(index: int, every_s: float) -> list[str]:
    return ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits", "-i", str(index),
            f"--loop-ms={max(50, round(every_s * 1000))}"]


def parse_sample(line: str) -> tuple[float, float] | None:
    """'37, 5120, 24564' -> (utilisation %, used MB); None for anything else."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


class GpuSampler:
    """Samples (t, util %, used MB) from a streaming command until stop()."""

    def __init__(self, command: list[str]):
        self.command = command
        self.samples: list[tuple[float, float, float]] = []
        self.error: str | None = None
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> "GpuSampler":
        try:
            self._proc = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                          stdin=subprocess.DEVNULL, text=True)
        except OSError as e:
            self.error = f"{type(e).__name__}: {e}"
            return self
        self._thread = threading.Thread(target=self._read, name="gpu-sampler", daemon=True)
        self._thread.start()
        return self

    def _read(self) -> None:
        for line in self._proc.stdout:
            s = parse_sample(line)
            if s is not None:
                self.samples.append((time.monotonic(), *s))

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        if self._thread is not None:
            self._thread.join(timeout=5)


def _stats(samples: list[tuple[float, float, float]]) -> dict:
    if not samples:
        return {"samples": 0, "gpu_util_mean": None, "gpu_util_max": None, "gpu_mem_peak_mb": None}
    util = [s[1] for s in samples]
    return {"samples": len(samples), "gpu_util_mean": round(sum(util) / len(util), 1),
            "gpu_util_max": max(util), "gpu_mem_peak_mb": max(s[2] for s in samples)}


class Profiler:
    """Stage marks plus GPU sampling. Disabled (or on the CPU) it only keeps the marks."""

    def __init__(self, enabled: bool, gpu_index: int | None = None, every_s: float = 0.5,
                 command: list[str] | None = None):
        self.enabled = enabled
        self.marks: list[tuple[str, float, float]] = []
        self.t0 = time.monotonic()
        self.sampler = None
        if enabled and (command is not None or (gpu_index is not None and gpu_index >= 0)):
            self.sampler = GpuSampler(command or sampler_command(gpu_index, every_s))

    def start(self) -> "Profiler":
        self.t0 = time.monotonic()
        if self.sampler is not None:
            self.sampler.start()
        return self

    @contextmanager
    def stage(self, name: str):
        a = time.monotonic()
        try:
            yield
        finally:
            self.marks.append((name, a, time.monotonic()))

    def stop(self) -> None:
        if self.sampler is not None:
            self.sampler.stop()

    def summary(self, manifest: dict | None = None) -> dict:
        """manifest["profile"]: stages in order with seconds and GPU stats, the
        whole run, and the inner breakdowns the stages stored in the manifest."""
        samples = self.sampler.samples if self.sampler is not None else []
        end = max([m[2] for m in self.marks] + [time.monotonic()])
        stages = {}
        for name, a, b in self.marks:
            rec = {"seconds": round(b - a, 3), **_stats([s for s in samples if a <= s[0] <= b])}
            stages[name] = rec
        out = {"version": 1, "wall_s": round(end - self.t0, 3), "stages": stages,
               "run": _stats(samples), "gpu_sampler": None if self.sampler is None else
               {"command": " ".join(self.sampler.command), "error": self.sampler.error},
               "timeline": [[round(s[0] - self.t0, 2), s[1], s[2]] for s in samples]}
        m = manifest or {}
        out["breakdown"] = {k: m[k]["breakdown_s"] for k in ("shots", "pose_body", "overlay")
                            if isinstance(m.get(k), dict) and "breakdown_s" in m[k]}
        return out


# GPU identity and other processes (experiments refuse a shared GPU) ---------------------

def gpu_info(index: int, run=subprocess.run) -> dict:
    """{name, bus_id, driver, memory_total_mb} of one GPU, {} if nvidia-smi fails."""
    try:
        p = run(["nvidia-smi", "--query-gpu=name,pci.bus_id,driver_version,memory.total",
                 "--format=csv,noheader,nounits", "-i", str(index)], capture_output=True, text=True, timeout=20)
        name, bus, driver, total = [x.strip() for x in p.stdout.strip().splitlines()[0].split(",")]
        return {"name": name, "bus_id": bus, "driver": driver, "memory_total_mb": float(total)}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return {}


def gpu_processes(index: int, run=subprocess.run) -> list[dict]:
    """Compute processes on one GPU: [{pid, name, used_mb}]. Raises OSError if
    nvidia-smi cannot be asked (an experiment then does not start)."""
    info = gpu_info(index, run)
    if not info:
        raise OSError(f"nvidia-smi could not report GPU {index}")
    p = run(["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    out = []
    for line in p.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 4 or parts[0].lower() != info["bus_id"].lower():
            continue
        try:
            out.append({"pid": int(parts[1]), "name": parts[2], "used_mb": float(parts[3])})
        except ValueError:
            out.append({"pid": None, "name": parts[2], "used_mb": None})
    return out
