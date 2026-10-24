"""Test helpers. Every test works in a fake repo root under ~/AssemblyVision/tmp,
never in the real data/ folders."""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from assemblyvision.config import Config
from assemblyvision.storage.files import REPO_ROOT, RunContext, RunPaths, new_run_id

TMP_BASE = REPO_ROOT / "tmp" / "tests"


def fake_root() -> tempfile.TemporaryDirectory:
    TMP_BASE.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=TMP_BASE)


def make_ctx(root: Path, source: str = "public") -> tuple[Config, RunContext]:
    cfg = Config.load()
    for s in ("qisens", "public"):
        (root / cfg["paths"][f"{s}_root"]).mkdir(parents=True, exist_ok=True)
    run_id = new_run_id()
    paths = RunPaths.for_source(source, run_id, cfg["paths"], repo_root=root)
    paths.create()
    log = logging.getLogger(f"test.{run_id}")
    log.addHandler(logging.NullHandler())
    log.propagate = False
    return cfg, RunContext(run_id, source, paths, log)


def ffmpeg_clip(out: Path, *args: str, seconds: float = 3, rate: int = 25, size: str = "320x240") -> Path:
    """Test pattern clip; extra args go before the output path."""
    cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi",
           "-i", f"testsrc=duration={seconds}:size={size}:rate={rate}", *args, str(out)]
    subprocess.run(cmd, check=True)
    return out
