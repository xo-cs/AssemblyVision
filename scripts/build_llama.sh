#!/bin/bash
# Build llama-server (llama.cpp tag b12005, CUDA, sm 86 for the RTX A5000 and A4000) inside this copy.
# Needs the toolchain env envs/llamabuild (scripts/setup_env.sh makes it from scripts/llamabuild.conda-spec.txt).
# The binary finds its libraries through an RPATH into this copy, so it is built per copy, never copied.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TC="$ROOT/envs/llamabuild"
SRC="$ROOT/third_party/llama.cpp"
test "$(git -C "$SRC" rev-parse HEAD)" = "e0f142d5ddc1461ce721644d92af33db23545105" || { echo "llama.cpp is not at tag b12005"; exit 1; }
rm -rf "$SRC/build"
# activate the toolchain env: its activation scripts set the compiler and linker flags (rpath-link into envs/llamabuild/lib)
CONDA_BASE="$(conda info --base)"
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$TC"
set -u
export CMAKE_PREFIX_PATH="$TC"
"$TC/bin/cmake" -S "$SRC" -B "$SRC/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DCMAKE_C_COMPILER="$TC/bin/x86_64-conda-linux-gnu-cc" -DCMAKE_CXX_COMPILER="$TC/bin/x86_64-conda-linux-gnu-c++" \
  -DCMAKE_CUDA_COMPILER="$TC/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER="$TC/bin/x86_64-conda-linux-gnu-c++" \
  -DCMAKE_BUILD_RPATH="$TC/lib" -DCMAKE_INSTALL_RPATH="$TC/lib"
"$TC/bin/cmake" --build "$SRC/build" --target llama-server -j "$(nproc)"
"$SRC/build/bin/llama-server" --version
