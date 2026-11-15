#!/bin/bash
# Make the two conda envs of this copy from the exact specs in scripts/ (no env is copied: conda envs
# hold absolute paths). env: Python 3.11 and every pip package of code/requirements-pinned.txt
# (torch 2.14.1+cu126 from the PyTorch cu126 index). envs/llamabuild: gcc 12, nvcc 12.6, cmake, ninja.
# Downloads about 5 GB; all caches stay inside this copy.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA="${CONDA:-$(command -v conda)}"
export CONDA_PKGS_DIRS="$ROOT/.cache/conda_pkgs" PIP_CACHE_DIR="$ROOT/.cache/pip"
mkdir -p "$CONDA_PKGS_DIRS"
[ -d "$ROOT/env" ] || "$CONDA" create -y -q -p "$ROOT/env" --file "$ROOT/scripts/env.conda-spec.txt"
"$ROOT/env/bin/python" -m pip install -q -r "$ROOT/code/requirements-pinned.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu126
[ -d "$ROOT/envs/llamabuild" ] || "$CONDA" create -y -q -p "$ROOT/envs/llamabuild" --file "$ROOT/scripts/llamabuild.conda-spec.txt"
"$ROOT/env/bin/python" -c "import torch, ultralytics, transformers, gradio; print('env ok: torch', torch.__version__, 'cuda', torch.cuda.is_available())"
