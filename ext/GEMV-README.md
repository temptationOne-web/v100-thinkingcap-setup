# tc_gemv - small-M FP16 dense GEMV for ThinkingCap's remaining cuBLAS projections (2026-10-26, CODE+CPU night)

Status: BUILT (sm_70 DSO) + CPU-tested. NOT yet run on a GPU, NOT live. Default OFF (`TC_GEMV=1` enables).
GPU check + A/B belong to the 06:55 window (`tests/tc_gemv_gpu_check.py`, then an A/B boot).

## 1. Findings (why the engine still runs dense GEMMs on cuBLAS at M=5)

Measured on the best stack (mp0 profile, rank 0, 1 user, 34.9 ms/round profiled = 26.2 unprofiled), the dense
cuBLAS/cutlass family costs **2.73 ms/round in 248 calls** (the "4 ms / 417 calls" figure in WHY-SLOW-REPORT.md was
the pre-bridge profile with 4 drafts). Per projection (TP4 per-rank shape, calls/round, us/call):

| projection | (N,K)/rank | M | calls | us each | route today | reachable by |
|---|---|---|---|---|---|---|
| self_attn.qkv_proj | 3584x2560 | 5 | 13 | 40.8 (cutlass 16x16 splitK8) | cuBLAS | tc_gemv |
| mlp.shared_expert.down_proj | 2560x160 | 5 | 49 | 8.7 (cublas gemmSN) | cuBLAS | tc_gemv + role table |
| mlp.shared_expert_gate | 1x2560 | 5 | 49 | 3.6+3.9 (cublas dot + reduce, 2 launches) | cuBLAS | tc_gemv + role table |
| self_attn.indexer.index_qk_proj | 640x2560 | 5 | 13 (+4) | 15.1+3.4 (cutlass 32x32 splitK4 + reduce) | cuBLAS | tc_gemv |
| final hyper_connection_mixer up/down (nn.Linear) | 10240x320 / 320x10240 | 5,1 | 3 + 2 | 41 / 17.5 | cuBLAS | TC_GEMV_MIXER=1 |
| lm_head verify | 62080x2560 | 5 | 1 | 483 (eager cutlass, 660 GB/s = near bandwidth) | cuBLAS | not worth it |
| mtp fc_embedding/fc_hidden | 640x2560 | 5/20, 1/4 | ~2 | 17 + batched gemv | cuBLAS | TC_GEMV_FC=1 |
| out_proj / o_proj | 2560x1536 | >8 (3-4 users only) | 49 | ~25 | cuBLAS | tc_gemv ("engine" rule) |
| drafts M=1 (qkv/o_proj/router/index) | | 1 | 12 | 7-26 (Triton row GEMV) | engine lane | unchanged |

Everything else at M=5 (HC 98x, GDN input 36x, out/o_proj 49x, router 49x, shared gate_up 49x) is already on the
engine's exact batch kernels and is left alone.

Why the existing lane does not cover them (`sm70_fp16_gemv.py`, live copy in `.venv/.../vllm/models/qwen4_exp/nvidia/`):
1. `_runtime_ok()` admits **x.shape[0] == 1 only**; the Triton row GEMV has no weight reuse across tokens.
2. M>1 is served by hand-written MMA kernels for fixed shapes only: `dense_batch_kernel` (router 512x2560 M<=4,
   out/o_proj 2560x1536 M<=8), `router_batch` / `shared_up_batch` (rows % 5 == 0), `gdn_input_batch` (2..16), HC batch.
3. The two shared-expert layers (`.mlp.shared_expert.down_proj`, `.mlp.shared_expert_gate`) are **not in `_ROLE_PLANS`**,
   so even their M=1 draft calls go to cuBLAS. Adding them to the table (hook 1 below) only moves those M=1 calls
   (~0.05 ms/round) - **extending the role table alone does NOT touch the M=5 work.** That is why the kernel route.
4. `VLLM_SM70_QWEN4_EXP_ONLINE_QPN8` never engages because `sm70_online_qpn8._exact_runtime_contract()` requires
   `speculative_config is None` (and admits 6 roles only); with MTP it can never run. It also requantizes to FP8
   (not exact) - Ornith's ~0 ms dense lane comes from FP8 weights in its checkpoint.
5. The packed-only port (`sm70_packed_only.py`) is live in the venv but only for hc_down / hc_up / gdn_qkvz; none of
   the hooked roles lose their row-major weight.

## 2. What is built

* `ext/build/tc_gemv.cu` -> `ext/build/tc_gemv_C.so` (nvcc 12.8, `-gencode arch=compute_70,code=sm_70`, torch 2.10+cu128).
  `torch.ops.tc_gemv.gemv_out(out, x, w, rows=0)`: out[M,N] = x[M,K] * w[N,K]^T, FP16 in/out, FP32 accumulate,
  1 <= M <= 20 (tiles 1,2,4,5,8,10,16,20), K % 32 == 0, contiguous + 16 B aligned.
  Block = 4 warps = R weight rows (R = 4 if K <= 512, 2 if N >= 2048, else 1); warp w owns K-quarter w; lanes stride
  one uint4 (8 halves); weights streamed once (`ld.global.cs`), x re-read from L1/L2; per-lane fmaf chain ->
  xor-butterfly (lane-0 value) -> the 4 quarter partials summed in order 0..3 -> `__float2half_rn`.
  **Deterministic** (no atomics/split-K workspace; the order depends only on K and lane id) and **batch-invariant**
  (a row's bits do not depend on M or on other rows). **Not bit-identical to cuBLAS** (different FP32 tree): expect
  <= 1 FP16 ulp differences on a few % of elements. ptxas: no errors; 3 of 24 variants (MT=16/20 with R=4, i.e. the
  2560x160 shared down at M>=16) spill 4-24 B - fine, M=20 happens only at 4 users.
* `ext/tc_gemv.py` - the runtime hook (no engine file edited):
  1. role table: adds `.mlp.shared_expert.down_proj` (2560,160) and `.mlp.shared_expert_gate` (1,2560) to
     `_ROLE_PLANS`, so those layers get `Qwen38SM70FP16LinearMethod` (engine Triton row GEMV at M=1; hook 2 at M>1).
  2. `Qwen38SM70FP16LinearMethod.apply` -> opaque op `torch.ops.tc_gemv_py.proj`: admitted (role, M) -> kernel,
     everything else -> `torch.ops.vllm.qwen38_sm70_fp16_gemv` unchanged. Packed-only layers, bias, non-decode
     semantics -> original apply. out_proj/o_proj are taken only when the engine's own `_can_use_dense_batch` declines.
  3. optional `TC_GEMV_MIXER=1` (shims `F.linear` inside `common/hyperconnection.py` only) and `TC_GEMV_FC=1`
     (`UnquantizedLinearMethod.apply` for `.fc_embedding/.fc_hidden` only).
  Hooks arm from a post-import finder (same as tc_ext/sb_p5), so no CUDA context before the worker picks its device.
* `ext/tc_gemv_ext.py` - `TCGemvExt(TCExt)` worker extension (tc_ext.py untouched).
* `tests/tc_gemv_cpu_test.py` - CPU: float32 emulation of the kernel's summation order vs FP64 on the real shapes,
  admission/role logic, role-table + apply hook against a stub of the engine module, fallbacks.
* `tests/tc_gemv_gpu_check.py` - tomorrow: correctness vs FP64 and cuBLAS (ulps, bit-mismatch %), 20x eager + 20x
  CUDA-graph determinism, batch invariance (M=20 rows == M=1/5 rows), microbench vs cuBLAS per shape/M, rows-per-block
  sweep, projected ms/round saved. `--model` loads real checkpoint rows.

## 3. Expected gain (from the profile; to be measured)

Assuming the kernel reaches ~700 GB/s on the big shapes and ~3 us on the tiny ones (rank 0, 1 user, profiled us):
qkv 13 x (41 -> ~27) = 0.18, shared down 49 x (8.7 -> ~3.5) = 0.25, shared gate 49 x (7.5 -> ~3) = 0.22,
index 13 x (18.5 -> ~6) = 0.16, drafts M=1 shared down/gate via the role table ~0.03 => **~0.85 ms profiled,
~0.7 ms unprofiled per round: 26.2 -> ~25.5 ms/step (-2.5..3%, ~+3% tok/s at 1 user)**. With TC_GEMV_MIXER=1 another
~0.15 ms. At 4 users (M=20) out/o_proj (49x) and qkv at M=20 join: cuBLAS was 5.7 ms/round there, expect -1.5..2 ms of
~44 ms (-3..4%). This is the honest ceiling: the remaining big items (LM head 0.48 ms at 660 GB/s, HC/all-reduce,
drafts, PLE stall) are not dense-GEMM problems. The M=1 path (TC_GEMV_MIN_M=1) is worth an A/B: the Triton row GEMV
does qkv at 26 us (700 GB/s) so little is expected there.

## 4. Tomorrow's procedure (06:55 window, JARVIS may be down)

1. Kernel check on one idle GPU (no server on it):
   `CUDA_VISIBLE_DEVICES=0 /opt/1cat/vllm-main-20261025/.venv/bin/python /opt/1cat/tcflash-trial/tests/tc_gemv_gpu_check.py --model /opt/1cat/models-tcflash/ThinkingCap-Qwen3.8-Flash-Next-NVFP4 --json /opt/1cat/tcflash-trial/logs/tc_gemv_gpu_check.json`
   Must print `RESULT: PASS` (ulps <= cuBLAS+1, deterministic, batch-invariant) and a positive "projected saved".
   If a shape is slower than cuBLAS, drop it from the role list (`TC_GEMV_ROLES`), do not ship it.
2. A/B boot (same harness as variants5: roundtime 1u/2u/4u + hard 7 + greedy.py hashes):
   baseline:  `bash /opt/1cat/tcflash-trial/run-tc-best.sh`
   candidate: `TC_GEMV=1 WEXT=tc_gemv_ext.TCGemvExt VLLM_CACHE_ROOT=/opt/1cat/tcflash-trial/cache-gemv bash /opt/1cat/tcflash-trial/run-tc-best.sh`
   (run-tc-new.sh hard-codes VLLM_CACHE_ROOT=cache-new; the candidate's traced graph contains a new op, so use the
   run-tc-prod.sh-style override or copy the launcher if the override is ignored - never share a compile cache
   between the two). Worker log must show `TC-GEMV ... Qwen38SM70FP16LinearMethod.apply hooked` and later
   `TC-GEMV ... active: self_attn.qkv_proj M=5 ...` lines (one per role/M). `collective_rpc("tc_gemv_status")` works too.
3. Gates (feedback rule: soak + json_schema test + prod log watch): greedy hashes WILL differ from baseline by design
   (cuBLAS tree != ours) - judge by hard 7/7, LCB sample, priority-client x10 identical under load (batch invariance), not by
   hash equality with the old stack. Run-to-run hashes with TC_GEMV=1 must be identical.
4. Options: `TC_GEMV_MIN_M=1` (also replace the Triton M=1 row GEMV), `TC_GEMV_MIXER=1`, `TC_GEMV_FC=1`,
   `TC_GEMV_ROLES=".self_attn.qkv_proj:2-20,.mlp.shared_expert_gate"` (explicit allowlist), `TC_GEMV_VERBOSE=0`.

## 5. Not done / follow-ups

* shared_expert.gate_up_proj at draft M=2..4 (multi-user drafts) goes through its own op (`qwen38_sm70_shared_up`),
  not `apply`; ~49 x 7 us at 2-4 users. Could be hooked the same way.
* LM head verify (M=5, 318 MB/rank) is already at ~660 GB/s on cutlass; only an FP8 head (numerics change) would help.
* PLE key_proj/value_proj (1x/round, eager) and the engine's `DenseLinearState` lane (`VLLM_SM70_ENABLE_DENSE_F16_FASTPATH`,
  `VLLM_SM70_F16_DENSE_ALLOWLIST`) were not evaluated; the latter is upstream's generic small-M FP16 provider and is a
  candidate for a second A/B if tc_gemv under-delivers.
* Lower-register variant for MT=16/20 x R=4 (spills) if the 4-user microbench shows the 2560x160 shape regressing.
