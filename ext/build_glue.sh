#!/bin/bash
# Build TC-GLUE for sm_70. CPU-only (never touches a GPU). Output: /opt/1cat/tcflash-trial/ext/build/tc_glue_C.so
# Never overwrite a .so a running server has loaded: set GLUE_OUT=<name> for a differently named copy.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
export TORCH_CUDA_ARCH_LIST="7.0"
export MAX_JOBS=4
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
D=/opt/1cat/tcflash-trial/ext
OUT=${GLUE_OUT:-tc_glue_C.so}
mkdir -p $D/build/jit
cd /tmp
nice -n 19 /opt/1cat/vllm-main-20261025/.venv/bin/python - "$D" "$OUT" <<'PY'
import os, shutil, sys, torch
from torch.utils.cpp_extension import load
D, OUT = sys.argv[1], sys.argv[2]
bd = f"{D}/build/jit"
load(
    name="tc_glue_C",
    sources=[f"{D}/tc_glue_src/tc_glue.cu"],
    extra_cflags=["-O3", "-std=c++17"],
    # NO --use_fast_math; --fmad=false: no implicit contraction, every FFMA is explicit (__fmaf_rn) exactly where the
    # production SASS (Triton _hc_combine_norm_kernel, ptxas 12.8.93 = this CUDA's ptxas) has one.
    extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_70,code=sm_70", "-lineinfo", "-Xptxas=-v",
                       "--expt-relaxed-constexpr", "--fmad=false"],
    build_directory=bd,
    is_python_module=False,
    verbose=False,
)
shutil.copy2(f"{bd}/tc_glue_C.so", f"{D}/build/{OUT}.tmp")
os.replace(f"{D}/build/{OUT}.tmp", f"{D}/build/{OUT}")
print("BUILT", f"{D}/build/{OUT}", "torch", torch.__version__)
PY
