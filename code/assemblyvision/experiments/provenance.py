"""What an experiment result depends on: code, model file, GPU, library versions, data."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from ..storage.files import REPO_ROOT


def git_state(repo: Path = REPO_ROOT, run=subprocess.run) -> dict:
    """{commit, branch, dirty}: dirty lists changed tracked files (untracked ones are ignored)."""
    def git(*args: str) -> str:
        p = run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30)
        return p.stdout.rstrip("\n") if p.returncode == 0 else ""
    changed = [line[3:] for line in git("status", "--porcelain", "--untracked-files=no").splitlines() if line.strip()]
    return {"commit": git("rev-parse", "HEAD").strip() or None,
            "branch": git("rev-parse", "--abbrev-ref", "HEAD").strip() or None,
            "dirty": changed}


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def model_checksum(weights: Path, sums_file: Path | None = None) -> dict:
    """sha256 of a weight file and whether it matches models/SHA256SUMS_yolo26_pose.txt."""
    digest = sha256_file(weights)
    sums_file = sums_file or weights.parent / "SHA256SUMS_yolo26_pose.txt"
    expected = None
    if sums_file.is_file():
        for line in sums_file.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].lstrip("*") == weights.name:
                expected = parts[0]
    return {"file": weights.name, "sha256": digest, "sums_file": sums_file.name if sums_file.is_file() else None,
            "matches_sums": None if expected is None else expected == digest}


def library_versions() -> dict:
    """The versions a run records (run_pose._versions), plus the CUDA build of torch."""
    from ..tools.run_pose import _versions
    return _versions()
