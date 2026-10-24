# AssemblyVision session setup. Use:  source ~/AssemblyVision/av.sh
conda activate ~/AssemblyVision/env
export PIP_CACHE_DIR=~/AssemblyVision/.cache/pip
export TORCH_HOME=~/AssemblyVision/.cache/torch
export XDG_CACHE_HOME=~/AssemblyVision/.cache
export HF_HOME=~/AssemblyVision/.cache/huggingface
export YOLO_CONFIG_DIR=~/AssemblyVision/.config/ultralytics
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd ~/AssemblyVision

# Node 22 and Claude Code, all inside ~/AssemblyVision
export npm_config_prefix=$HOME/AssemblyVision/tools/npm-global
export npm_config_cache=$HOME/AssemblyVision/.cache/npm
export PATH=$HOME/AssemblyVision/tools/npm-global/bin:$HOME/AssemblyVision/tools/node/bin:$PATH
export CLAUDE_CONFIG_DIR=$HOME/AssemblyVision/.claude-home
export CLAUDE_CODE_TMPDIR=$HOME/AssemblyVision/tmp
export DISABLE_AUTOUPDATER=1
