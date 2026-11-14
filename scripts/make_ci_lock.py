"""Write code/requirements-ci.lock from code/requirements-pinned.txt (Week 11).

    python scripts/make_ci_lock.py

The CI runner has no GPU: the CUDA packages are left out and torch and torchvision use
their CPU builds of the same version; every other package keeps its exact pin.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEAD = ["# CI lockfile (Week 11): every package of the av env at its exact version (code/requirements-pinned.txt),",
        "# for a CPU only runner: without the CUDA packages, with the CPU builds of torch and torchvision.",
        "# Regenerate after code/requirements-pinned.txt changes (python scripts/make_ci_lock.py).",
        "--extra-index-url https://download.pytorch.org/whl/cpu"]
DROP = re.compile(r"^(cuda-|nvidia-.*-cu12==|triton==)")
EXTRA = ["setuptools==84.0.0"]      # left out by pip freeze, needed by torch

pins = [x for x in (ROOT / "code/requirements-pinned.txt").read_text().splitlines() if x and not x.startswith("#")]
lines = HEAD + [x.replace("+cu126", "+cpu") for x in pins if not DROP.match(x)] + EXTRA
(ROOT / "code/requirements-ci.lock").write_text("\n".join(lines) + "\n")
print(f"{len(lines) - len(HEAD)} packages")
