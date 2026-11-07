#!/bin/bash
# Build 1Cat-vLLM upstream main (SM70 only) into a wheel, low priority.
set -euo pipefail
B=/opt/1cat/vllm-main-20261106
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$B/.venv/bin:$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=7.0
export CMAKE_CUDA_ARCHITECTURES=70
export FLASH_ATTN_V100_CUDA_ARCH_LIST=7.0
export MAX_JOBS=${MAX_JOBS:-6}
export NVCC_THREADS=1
export VLLM_TARGET_DEVICE=cuda
export TMPDIR=$B/tmp
export CUDA_VISIBLE_DEVICES=""
mkdir -p $TMPDIR $B/dist
cd $B/src
echo "START $(date -Is) commit $(git rev-parse --short HEAD)"
nice -n 19 $B/.venv/bin/python -m pip wheel --no-build-isolation --no-deps -v -w $B/dist .
echo "END $(date -Is) rc=$?"
