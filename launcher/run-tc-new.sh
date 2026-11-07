#!/bin/bash
# THINKINGCAP trial (2026-10-25, owner "test it straight away"): bottlecapai/ThinkingCap-Qwen3.8-Flash-Next-NVFP4 on the
# live 1Cat engine (vllm-main-20261008), TP4, TEST port :8030 (front :8021 + priority-client lane :8006 forward here), served under
# the same names as Ornith so every harness works. Settings = Swift-1.5's tuned prod recipe (same Flash-Next arch + same
# NVFP4/FP8-PLE format): /opt/1cat/swiftfn-trial/run-swift-prod.sh + serve-swift-prod.inc. NOT prod.
# run-tc-new.sh (2026-10-25): same recipe on the NEW engine /opt/1cat/vllm-main-20261025 (upstream main 6a143197e + ports, see
# /opt/1cat/vllm-main-20261025/PORTS.txt). Own compile cache (cache-new). SB-FUSE/SB-P5 point to compat copies in
# /opt/1cat/vllm-main-20261025/ext-compat (ContextVar fix for F1; prepare_mtp_qpn8_head signature fix for dvocab).
# ALLOW_MM is dropped: upstream covers Qwen4ExpForConditionalGeneration natively. NOT STARTED automatically.
set -u
M=${MODEL:-/opt/1cat/models-tcflash/ThinkingCap-Qwen3.8-Flash-Next-NVFP4}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3
# TC_ENGINE (2026-10-30): engine dir; default = live engine. Next engine trial: TC_ENGINE=/opt/1cat/vllm-main-20261030
TC_ENGINE=${TC_ENGINE:-/opt/1cat/vllm-main-20261025}
export CUDA_HOME=/usr/local/cuda-12.8 PATH=$TC_ENGINE/.venv/bin:/usr/local/cuda-12.8/bin:$PATH
export NCCL_IB_DISABLE=1 VLLM_CACHE_ROOT=/opt/1cat/tcflash-trial/cache-new VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=1
# PLE_MODE=cpu (default, Swift-era: whole 47.7 GiB n-gram table in the CPU offload worker, every decode step waits on an IPC
# round trip) | cascade (upstream default placement: rows resident in device/pinned-host tiers, gathered inside the graph;
# PLE_HOST_GIB = pinned GiB per rank, PLE_VRAM_GIB = device reserve GiB per rank). 2026-10-26 why-slow.
if [ "${PLE_MODE:-cpu}" = cpu ]; then export VLLM_PLE_CPU_OFFLOAD=1 VLLM_PLE_DISK_OFFLOAD=${PLE_DISK:-1}
else unset VLLM_PLE_CPU_OFFLOAD VLLM_PLE_DISK_OFFLOAD; export VLLM_QWEN4EXP_PLE_HOST_GIB=${PLE_HOST_GIB:-auto}; [ -n "${PLE_VRAM_GIB:-}" ] && export VLLM_QWEN4EXP_PLE_VRAM_RESERVE_GIB=$PLE_VRAM_GIB; fi
# CT_TM=1 -> compressed-tensors NVFP4 experts on the SM70 ModelOpt TurboMind/QPN route instead of Marlin (PORTS.txt #8)
export VLLM_SM70_CT_NVFP4_TM=${CT_TM:-0}
export VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT=0
export VLLM_SM70_QWEN38_FP16_GEMV=1 VLLM_SM70_QWEN38_FUSED_HC_FP16=1 VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16=1
export VLLM_SM70_QWEN38_PACKED_ONLY=${PACKED_ONLY:-1} VLLM_SM70_MTP_ROUTER_BATCH=${MTP_RB:-0} VLLM_SM70_MTP_SHARED_BATCH=${MTP_SB:-0}
if grep -qwE 'iommu=pt|iommu=off|intel_iommu=off' /proc/cmdline; then
  export VLLM_SM70_PCIE_P2P_AS_FULL=1 VLLM_SM70_HC_FORCE_REPLICATED=1
fi
export PYTHONPATH=/opt/1cat/tcflash-trial/ext:/opt/1cat/swiftfn-trial${PYTHONPATH:+:$PYTHONPATH}   # ext/tc_ext.py = SBP5Ext + optional ARX (WEXT=tc_ext.TCExt SB_ORN_ARX=1)
export MEMGUARD_LOG=/opt/1cat/tcflash-trial/logs/memguard.log MEMGUARD_METRICS=http://127.0.0.1:8030/metrics MEMGUARD_HEALTH=http://127.0.0.1:8030/health
export SB_FUSE_PKG=/opt/1cat/vllm-main-20261025/ext-compat/sbfuse SB_FUSE_LIB=/opt/1cat/vllm-main-20261025/ext-compat/sbfuse/sb_fuse_C.so SB_FUSE_MTP_MOE=${SB_FUSE:-1} SB_FUSE_MTP_MOE_MAX_M=16
export SB_CAR_PKG=/opt/1cat/swiftfn-trial/sbcar SB_CAR=${SB_CAR:-1} SB_CAR_MODE=a2a SB_CAR_BLOCK=32 SB_CAR_MIN_BYTES=3145728
export SB_P5_PKG=/opt/1cat/vllm-main-20261025/ext-compat/sbp5 SB_P5_LIB_DIR=/opt/1cat/vllm-main-20261025/ext-compat/sbp5 SB_P5_CONF=/opt/1cat/vllm-main-20261025/ext-compat/sbp5/p5.conf SB_P5_BASE=p4 SB_P5_DVOCAB_RANKING=/opt/1cat/vllm-main-20261025/ext-compat/sbp5/swift_draft_ranking.pt
export SB_P5_ROUTER=${SB_P5_ROUTER:-${SB_P5:-1}} SB_P5_ROUTER_MODE=fused2 SB_P5_ROUTER_MS=2,4,5,8,10,12,16 SB_P5_DVOCAB=${SB_P5_DVOCAB:-${SB_P5:-1}} SB_P5_DVOCAB_N=98304
# ADAPTIVE=0 -> plain k=MTP_K (upstream now admits concurrent MTP4 rows 5..40 natively; A/B both)
[ "${ADAPTIVE:-1}" = 1 ] && export VLLM_SB_ADAPTIVE_MTP_DRAFTS=3 VLLM_SB_ADAPTIVE_MTP_MAX_CTX=24000
export VLLM_V2_CUDAGRAPH_MEM_MIB=460
K=${MTP_K:-4}
DSM=${DRAFT_SAMPLE:-greedy}
LAR=""; [ "$DSM" = probabilistic ] && LAR='"use_local_argmax_reduction":false,'   # engine refuses probabilistic + local argmax
# TC_NOBLOCKDROP=1 (2026-10-30): #53388 opt-in, keep the trailing EAGLE/MTP block so a follow-up reuses it (PR #1276)
NBD=""; [ "${TC_NOBLOCKDROP:-0}" = 1 ] && NBD='"disable_eagle_block_drop":true,'
# TC_MOE_OVERLAP=0 (2026-10-30): turn off the auto-enabled shared-expert two-stream overlap (runtime_defaults.py:111)
[ "${TC_MOE_OVERLAP:-1}" = 0 ] && export VLLM_QWEN3NEXT_ENABLE_SHARED_MOE_OVERLAP=0 VLLM_SM70_DISABLE_QWEN3NEXT_SHARED_MOE_OVERLAP=1
SPEC=(); [ "$K" != 0 ] && SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$K,\"draft_sample_method\":\"$DSM\",$LAR$NBD\"attention_backend\":\"FLASH_ATTN_V100\"}")
EXT=${WEXT:-sb_p5_ext.SBP5Ext}
echo "$(date '+%F %T') TC-NEW run engine=$(basename $TC_ENGINE) partial=${TC_PARTIAL:-0} nbd=${TC_NOBLOCKDROP:-0} moeov=${TC_MOE_OVERLAP:-1} K=$K ext=$EXT util=${GPU_UTIL:-0.905} len=${MAX_MODEL_LEN:-262144} kvoff=${KVOFF_GB:-48}" >> /opt/1cat/tcflash-trial/logs/runs.log
KVO=(); [ "${KVOFF_GB:-48}" != 0 ] && KVO=(--kv-offloading-size ${KVOFF_GB:-48} --kv-offloading-backend native \
  --kv-transfer-config '{"kv_connector_extra_config":{"mamba_state_slots_reference_tokens":50000}}')
# PROF=1: torch profiler endpoints (diagnostic only; RAM chat cache must be OFF -> KVOFF_GB=0)
# HOTVOCAB=N -> upstream FR-Spec draft head: MTP drafts choose among the N most frequent token ids (ranking file from sb_p5), verification unchanged
# TC_PARTIAL=1 (2026-10-30, engine 20261030+ only): upstream #1297 partial prefix-cache hits (resume at the end of the last message)
PPC=(); [ "${TC_PARTIAL:-0}" = 1 ] && PPC=(--partial-prefix-cache)
HOT=(); [ "${HOTVOCAB:-0}" != 0 ] && HOT=(--kernel-config.sm70_draft_hot_vocab=$HOTVOCAB --kernel-config.sm70_draft_hot_vocab_ids=${HOTVOCAB_IDS:-/opt/1cat/tcflash-trial/bench/draft_hot_ids.npy})
PRF=(); [ "${PROF:-0}" = 1 ] && PRF=(--profiler-config '{"profiler":"torch","torch_profiler_dir":"/opt/1cat/ornith122-trial/barrier/logs/prof","torch_profiler_with_stack":false}')
cd /tmp
exec vllm serve "$M" \
  --served-model-name ornith-122b qwen38-27b qwen36-27b --trust-remote-code --dtype half --kv-cache-dtype auto \
  --attention-backend FLASH_ATTN_V100 \
  --host 127.0.0.1 --port 8030 --tensor-parallel-size 4 --gpu-memory-utilization ${GPU_UTIL:-0.905} \
  --max-model-len ${MAX_MODEL_LEN:-262144} --max-num-seqs 4 --max-num-batched-tokens 4096 \
  --enable-prefix-caching --mamba-cache-mode align \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  "${SPEC[@]}" --default-chat-template-kwargs '{"enable_thinking": true}' \
  --worker-extension-cls $EXT --scheduling-policy priority \
  "${KVO[@]}" "${PRF[@]}" "${HOT[@]}" "${PPC[@]}" ${TC_EXTRA:-} --async-scheduling
