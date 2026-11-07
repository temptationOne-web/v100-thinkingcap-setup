#!/bin/bash
# Engine 20261106 (1Cat main e098ee11b): install wheel + 11 ports (2026-11-06; 9 old + ReVolta #74 + 1Cat-#567 offload fix).
# Ports = 3-way merges (git merge-file, 0 conflicts) of the 20261025 live edits onto the new source: ports/stage/<path>.
#   packed-only (gemv, hc, + sm70_packed_only.py), PCIe P2P as full (custom_all_reduce), adaptive MTP (arg_utils,
#   async_scheduler, cudagraph_utils, compilation, speculator), CT NVFP4 -> SM70 ModelOpt route (w4a4_nvfp4 moe).
# Dropped: PLE spin-wait + PLE FP8 lane (env never set in prod; spin-wait measured no gain). PR#1232 is upstream now
#   (b5b7707a1) -> run with TC_UPFIX_PR1232=0.
# Each changed file keeps <file>.orig. Undo ports: for f in $(find vllm -name '*.py.orig'); do cp -p $f ${f%.orig}; done;
#   rm vllm/models/qwen4_exp/nvidia/sm70_packed_only.py
set -euo pipefail
N=/opt/1cat/vllm-main-20261106; ST=$N/ports/stage; SP=$N/.venv/lib/python3.12/site-packages
W=$(ls $N/dist/*.whl | tail -1)
[ "${SKIP_WHEEL:-0}" = 1 ] || uv pip install -q --no-deps --reinstall --python $N/.venv/bin/python "$W"
cd $SP
for s in $(cd $ST && find vllm -name '*.py' ! -name '*.base' ); do
  case $s in *sm70_packed_only.py) continue;; esac
  [ -e "$s.orig" ] && { echo "already ported ($s.orig)"; exit 1; }
  cmp -s "$s" "$ST/$s.new" || { echo "installed $s != staged upstream copy; refusing"; exit 1; }
  cp -p "$s" "$s.orig"; cp "$ST/$s" "$s"; echo "ported $s"
done
cp $ST/vllm/models/qwen4_exp/nvidia/sm70_packed_only.py vllm/models/qwen4_exp/nvidia/sm70_packed_only.py
$N/.venv/bin/python -m py_compile $(cd $SP && find vllm -name '*.py.orig' | sed 's/\.orig$//') vllm/models/qwen4_exp/nvidia/sm70_packed_only.py && echo "py_compile OK"
