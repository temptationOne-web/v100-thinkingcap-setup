#!/bin/bash
# ThinkingCap best-known launcher (2026-10-26 why-slow investigation). Wraps run-tc-new.sh (NEW engine
# /opt/1cat/vllm-main-20261025) with the settings that measured best today. Any K=V on the environment overrides.
#   CT_TM=1   compressed-tensors NVFP4 experts -> engine's SM70 ModelOpt TurboMind/QPN route (PORTS.txt #8); was Marlin.
#             1 user 34.9 -> 27.7 ms/step (code prompt 101 -> 134 tok/s, story 66 -> 84), hard questions 7/7.
#   VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES=0   TurboMind autotune of the small batched expert GEMM shapes crashes at warmup
#             (M=50 slot shape, ec=-1) on this engine; default kernels are used for those shapes (decode uses the QPN
#             direct kernels, not these).
#   VLLM_SM70_NVFP4_QWEN38_MOE_QPN_DYNAMIC_DECODE=1   QPN batch expert route for 2..16 verify tokens (2-3 users),
#             not only {2,4,8,16}. Neutral at 1u/4u.
#   VLLM_SM70_QWEN38_HYBRID_PLE=1 PLE_DISK=1   upstream "hybrid" PLE placement: each rank keeps its 11.9 GiB n-gram shard
#             in pinned host RAM and gathers it inside the decode graph; prefill uses the CPU worker over the mapped
#             checkpoint (no 47.7 GiB anonymous copy). 1u unchanged (27.6 ms), 2u 34.5 -> 32.8, 4u ~65 -> 45-55 ms/step.
#             RAM: ~63 GiB RSS at load (pinned shards + loader pages); the RAM chat cache must stay 0 (or <=4 GiB) under the 95G cap.
#   WEXT=tc_ext.TCExt SB_ORN_ARX=1   Ornith's ARX two-shot decode all-reduce (bit-identical, half the PCIe bytes): 27.6 -> 26.6 ms/step.
#   MTP_RB=1 MTP_SB=1   upstream's exact M5/M10 router + shared-expert batch kernels (~1%).
#   SB_P5=0   the sb_p5 router hook breaks a fresh torch.compile; dvocab gave no gain.
#   MTP_K=4 DRAFT_SAMPLE=probabilistic + adaptive 3 drafts at >=2 users (VLLM_SB_ADAPTIVE_MTP_DRAFTS=3 via ADAPTIVE=1).
# Memory: this route needs ~1.5 GiB/rank more than Marlin: KV 377k tokens at GPU_UTIL 0.905 / 262k ctx (was 487k).
set -u
export CT_TM=${CT_TM:-1}
export VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES=${VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES:-0}
export VLLM_SM70_NVFP4_QWEN38_MOE_QPN_DYNAMIC_DECODE=${VLLM_SM70_NVFP4_QWEN38_MOE_QPN_DYNAMIC_DECODE:-1}
export PLE_MODE=${PLE_MODE:-cpu} PLE_DISK=${PLE_DISK:-1} VLLM_SM70_QWEN38_HYBRID_PLE=${VLLM_SM70_QWEN38_HYBRID_PLE:-1}
export SB_P5=${SB_P5:-0} MTP_K=${MTP_K:-4} DRAFT_SAMPLE=${DRAFT_SAMPLE:-probabilistic}
# 2026-10-26 21:xx priority-client-exactness stack (gates: inv_probe 4/4, priority-client_stress 2xABCD one fp, deadlock PASS; see WHY-SLOW-REPORT.md 3c):
#   --scheduler-cls orn_solo_sched_v4.OrnSoloStepScheduler  Ornith's no-mixing scheduler (decode-only steps vs single-prefill
#        steps; priority-client lane (priority<0) never drafted and decodes alone; structured outputs never drafted). Read-only import
#        from /opt/1cat/ornith122-trial/{mtp2,pkg}; its kvfix2/prodchunk parts stay off (no RAM KV tier here).
#   VLLM_FLASH_V100_DECODE_PARTITION_SIZE=256  pins the FA_V100 decode partition (was chosen from the batch max seq len).
#   ORN_V3_PRIOCLIENT_NOCACHE=1  priority-client never reads the prefix cache (cold = reproducible; the warm first-call fp differed).
export PYTHONPATH=/opt/1cat/ornith122-trial/mtp2:/opt/1cat/ornith122-trial/pkg${PYTHONPATH:+:$PYTHONPATH}
export VLLM_FLASH_V100_DECODE_PARTITION_SIZE=${VLLM_FLASH_V100_DECODE_PARTITION_SIZE:-256}
export ORN_V4_SO_NODRAFT=${ORN_V4_SO_NODRAFT:-0} ORN_V3_PRIOCLIENT_SOLO=${ORN_V3_PRIOCLIENT_SOLO:-1} ORN_V3_PRIOCLIENT_NOCACHE=${ORN_V3_PRIOCLIENT_NOCACHE:-1}
# 2026-10-27: tc_solo_sched = orn v4 minus R1/R5/R9 (priority-client + JSON draft again; R2 solo + R7 nocache kept)
export TC_EXTRA="${TC_EXTRA:---mixed-prefill-step-latency-ms 0} --scheduler-cls ${SCHED_CLS:-tc_solo_sched.TCSoloStepScheduler}"
export WEXT=${WEXT:-tc_ext.TCExt} SB_ORN_ARX=${SB_ORN_ARX:-1} MTP_RB=${MTP_RB:-1} MTP_SB=${MTP_SB:-1}
export KVOFF_GB=${KVOFF_GB:-0}   # 14:49: hybrid pinned PLE (~63 GiB RSS at load) + 16 GiB RAM chat cache hit the 95G cgroup cap -> killed; 0 (or <=4) with hybrid PLE
exec bash /opt/1cat/tcflash-trial/run-tc-new.sh
