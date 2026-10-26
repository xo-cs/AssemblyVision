"""RunManager (LLD 3.1), first part: create a run and start its worker.

The interface never runs analysis itself (NFR-PERF-04). create_run makes the
run folder with a queued manifest; start spawns the batch command
(tools/run_pose with --run-id) as a separate process in its own session, with
its output in runs/<run_id>/log/worker.out, writes its PID to
log/worker.pid and returns at once. The worker takes the GPU lock itself, so a
second run waits as queued. The prepare and analyse phases with the subject
click (LLD 3.1) come with the database; until then a run picks its subjects
automatically.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ..config import VARIANTS, _toml_value
from ..errors import AvError
from ..storage.files import (SOURCE_TYPE, RunPaths, check_video_source, new_run_id, utc_now,
                             write_atomic)
from ..storage.runstore import RunStore, worker_alive

CODE_DIR = Path(__file__).resolve().parents[2]


class RunManager:
    def __init__(self, store: RunStore, cfg, worker_cmd: list[str] | None = None):
        self.store = store
        self.cfg = cfg
        # the worker command without the run arguments (tests pass a fake worker)
        self.worker_cmd = worker_cmd or [sys.executable, "-m", "assemblyvision.tools.run_pose"]
        self._procs: list[subprocess.Popen] = []

    def create_run(self, video: str | Path, source: str, variant: str, engine: str = "A",
                   overrides: dict | None = None) -> tuple[str, list]:
        """New queued run folder; returns (run_id, notices). AV-RUN-001 if the
        video is not in the data area of source, or source is not open.
        overrides {section: {key: value}} are passed to the worker as --set."""
        notices = []
        if source not in self.store.roots:
            raise AvError("AV-RUN-001", name=Path(video).name, detail=f"source {source} is not open here")
        real = check_video_source(Path(video), source, self.cfg["paths"], self.store.repo_root)
        if variant not in VARIANTS:
            notices.append(AvError("AV-APP-006", what="Model variant", value=variant,
                                   default=self.cfg["pose"]["variant"]))
            variant = self.cfg["pose"]["variant"]
        if engine != "A":
            notices.append(AvError("AV-SEG-001"))
            engine = "A"
        while True:
            run_id = new_run_id()
            paths = RunPaths(run_id, self.store.roots[source] / "runs" / run_id)
            try:
                paths.create()
                break
            except FileExistsError:
                continue
        manifest = {"run_id": run_id, "source": source, "source_type": SOURCE_TYPE[source], "status": "queued",
                    "created_utc": utc_now(), "video": str(real), "variant": variant, "engine": engine,
                    "requested_by": "interface", "stages": {}, "overrides": overrides or {},
                    "notices": [{"stage": "interface", **n.as_record()} for n in notices]}
        write_atomic(paths.manifest, json.dumps(manifest, indent=2).encode("utf-8"), paths)
        return run_id, notices

    def start(self, run_id: str) -> int:
        """Spawn the worker of a queued run and return its PID at once."""
        source, paths = self.store._locate(run_id)
        m = self.store.get_run(run_id)
        try:
            pid = int(paths.worker_pid.read_text())
        except (OSError, ValueError):
            pid = None
        if m.get("status") != "queued" or worker_alive(pid, run_id):
            raise AvError("AV-RUN-007")
        cmd = [*self.worker_cmd, "--source", source, "--video", m["video"], "--variant", m["variant"],
               "--gpu", str(self.cfg["gpu"]["pose"]), "--run-id", run_id]
        for section, values in (m.get("overrides") or {}).items():
            for key, value in values.items():
                cmd += ["--set", f"{section}.{key}={_toml_value(value)}"]
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", TMPDIR=str(paths.tmp_dir),
                   PYTHONPATH=str(CODE_DIR) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        paths.log_dir.mkdir(exist_ok=True)
        with open(paths.worker_out, "ab") as out:
            proc = subprocess.Popen(cmd, cwd=str(CODE_DIR), env=env, stdin=subprocess.DEVNULL, stdout=out,
                                    stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        write_atomic(paths.worker_pid, str(proc.pid).encode(), paths)
        self._procs.append(proc)
        return proc.pid

    def reap(self) -> None:
        """Collect finished workers, so they do not stay as zombie processes."""
        self._procs = [p for p in self._procs if p.poll() is None]
