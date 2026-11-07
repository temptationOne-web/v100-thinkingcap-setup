# tc_gdnglue - ThinkingCap GDN verify glue fusion (port of Ornith ports/gdnglue)

Built 2026-10-26 evening, CPU only (no GPU was used). Target: the new engine `/opt/1cat/vllm-main-20261025`
(ThinkingCap / Qwen3.8 Flash-Next / Qwen4Exp, TP4, MTP k=4). Nothing in the engine, `tc_ext.py`, Ornith or Swift
trees was edited. Default **off**.

## What it does

Every MTP verify round runs ThinkingCap's 36 Gated-DeltaNet layers once (the draft steps have no GDN layers). Per
layer the live verify route (`qwen_gdn_full_forward` op -> `forward_cuda` -> `qwen_gdn_attention_core_standard_spec`
-> `_forward_core`) launches, after the input GEMV:

| kernel (profile prof-mp0 1 user / prof-bridge4 4 users, median per layer) | 1u us | 4u us |
|---|---:|---:|
| `FillFunctor` (torch.zeros core_attn_out) | 2.4 | 1.9 |
| `_causal_conv1d_update_kernel` (spec, in place) | 4.9 | 4.8 |
| `fused_sigmoid_gating_delta_rule_update_kernel` (recurrence) | 19.2 | 26.6 |
| `rmsnorm_gated_exact_kernel<false>` (vllm._C, sigmoid gate) | 3.8 | 3.9 |
| span from fill start to norm end (incl. 3 launch gaps) | 33.2 | 43.6 |

`tc_gdnglue` replaces those four launches by **one** Triton kernel (`tc_gdnglue_kernels.tc_gdnglue_verify_kernel`):

1. conv prologue: each recurrence program (request, v-head, 32-wide V block) convolves the q/k channels of its k-head
   and its own v slice (copied stock conv code) into a private fp16 scratch (mixed_qkv is not modified);
2. conv-state roll: a program commits its own v slice right away; the q/k slice of a k-head is read by 12 programs,
   the last of them (self-resetting atomic counter) commits it - pure data movement copied from the stock kernel;
3. recurrence: the stock fused-sigmoid kernel body copied verbatim (same constexpr route), reading q/k/v from the
   scratch; PAD requests write zeros (stock leaves the zero fill);
4. gated RMSNorm epilogue: the last of the 4 V-block programs of a (request, head) norms that head's rows with the
   CUDA kernel's exact arithmetic (inline PTX taken from nvcc 12.8 -O3 output of `rmsnorm_gated_exact_sm70.cu`:
   mul.rn/add.rn, its lane/shuffle summation tree, rsqrt.approx, the CUDA expf expansion, rcp.rn / div.rn);
   graph-padding rows (zero core rows) are normed by one program per head, as stock does.

Then the stock out_proj and `output[:n] = proj` run unchanged. The input projection GEMV (`gdn_input_batch_kernel`,
36 us/layer = 1.3 ms of the "2.2 ms GDN" in WHY-SLOW-REPORT.md) and the recurrence itself are **not** changed - the GEMV
is bandwidth bound and the recurrence is exactly what we must reproduce bit for bit.

## Expected gain (from WHY-SLOW-REPORT.md / the profiles above; to be confirmed by the GPU bench + server A/B)

- removed: fill + conv + norm launches and 3 inter-kernel gaps = 11.1 us + ~3 us (1u), 10.6 + ~6 us (4u) per layer
- added inside the fused kernel: conv prologue ~1-2 us, last-arriver norm tail ~1.5 us, counters/fences ~1 us
- **net ~6-10 us per layer x 36 = ~0.2-0.4 ms per verify round** -> 1 user 26.2 ms/step -> ~25.9 ms (about 1-1.5%),
  4 users ~0.3-0.45 ms of ~44 ms (~1%). Modest: the bulk of ThinkingCap's GDN time is the GEMV + recurrence.
- `tests/tc_gdnglue_gpu_check.py` prints the per-layer saving measured in CUDA graphs of 36 layers (stock vs fused).

## Exactness (target: bit-identical to the unfused route)

Done tonight (CPU only, logs in `logs/`):
- `tests/test_tc_gdnglue_cpu.py` (Triton interpreter, USE_ASM=False path): stock sequence (zero fill + vLLM
  `causal_conv1d_update` + vLLM `fused_sigmoid_gating_delta_rule_update_mixed_qkv` + numpy emulation of the CUDA norm)
  vs fused kernel, **bitwise** on normed output, core output, ssm state cache, conv state cache, mixed_qkv untouched,
  counters reset: **12/12 PASS** (`logs/gdnglue-cpu.log`; 1-4 requests, T 4/5, PAD request, graph-padding rows,
  DS/SD conv layout, BV 32 and 16, sigmoid and silu gate). Negative controls fire (`logs/gdnglue-cpu-negctl.log`):
  sequential instead of shuffle tree -> normed mismatch; stale counters -> core/ssm mismatch.
- `tests/gdnglue_offline_check.py`: both kernels compiled for sm_70 with `/usr/local/cuda-12.8` ptxas (no GPU):
  fused kernel 127 regs, no spills (stock 117); the four recurrence reductions (sum q^2, sum k^2, sum h*k, sum h*q)
  have **identical shape/axis/layout** in the fused kernel and in the stock kernel (layout = reduction tree); inline
  PTX assembles. Artifacts: `build/tc_gdnglue_bv32.{cubin,sass,ttgir,ptx}`, `build/stock_sigmoid.*`.
- `tests/test_tc_gdnglue_install.py`: patch installs in both import orders; every engine name used exists
  (`logs/gdnglue-install.log`).

Tomorrow on GPU (the only proof of the transcendental bits):
```
CUDA_VISIBLE_DEVICES=<free gpu> PYTHONPATH=/opt/1cat/tcflash-trial/ext \
  /opt/1cat/vllm-main-20261025/.venv/bin/python /opt/1cat/tcflash-trial/ext/tests/tc_gdnglue_gpu_check.py \
  2>&1 | tee /opt/1cat/tcflash-trial/ext/logs/gdnglue-gpu.log
```
Optional extra (`--bv 16`): offline, the stock kernel compiled at BV=16 reduces each row over K with the SAME layout
as at BV=32 (`sizePerThread [1,4], threadsPerWarp [1,32], warpsPerCTA [4,1]`; `logs/gdnglue-offline-bv16.log`); only
the number of rows per CTA changes. So a 16-wide V block should also be bit-identical to the live BV=32 route while
doubling the CTAs of the latency-bound recurrence (48 -> 96 at 1 user, 80 SMs; the 1-user recurrence is 19 us for
5 sequential tokens). If `tc_gdnglue_gpu_check.py --bv 16` passes and its bench is faster, set `TC_GDNGLUE_BV=16`
(possible extra ~0.1-0.25 ms/round, unmeasured).

200 random cases vs the REAL stock kernels incl. `torch.ops._C.sm70_rmsnorm_gated_exact_out` (4 input regimes, PAD,
padding rows, both conv layouts, silu+sigmoid), 16 chained rounds, CUDA-graph capture + 10 replays, a negative control
(USE_ASM=False must differ) and the microbench. Refuses to run 23:40-06:50 unless `--force`. Optional
`--bv 16` checks whether a 16-wide V block is also exact (more CTAs at 1 user; knob `TC_GDNGLUE_BV`).

## How the investigator enables it

The module self-installs when imported with `TC_GDNGLUE=1`; it only has to be imported in the worker processes. Two
ways (PYTHONPATH already contains `/opt/1cat/tcflash-trial/ext` via run-tc-new.sh):

A. worker-extension chain (no edit anywhere):
```
TC_GDNGLUE=1 WEXT=tc_gdnglue.TCGdnGlueExt TC_GDNGLUE_BASE_EXT=tc_ext.TCExt bash /opt/1cat/tcflash-trial/run-tc-best.sh
# with the sampler too:
TC_GDNGLUE=1 TC_TPSAMPLE=1 WEXT=tc_gdnglue.TCGdnGlueExt TC_GDNGLUE_BASE_EXT=tc_sampler.TCSamplerExt \
  TC_TPSAMPLE_BASE_EXT=tc_ext.TCExt bash /opt/1cat/tcflash-trial/run-tc-best.sh
# prod launcher: TC_BASE_EXT=tc_gdnglue.TCGdnGlueExt (tc_prod_ext wraps it)
```
B. or add `import tc_gdnglue  # noqa` to tc_ext.py (owner of that file: the investigator); it is a no-op unless
   `TC_GDNGLUE=1`.

No torch.compile change: the patched code runs eagerly inside the opaque `vllm::qwen_gdn_full_forward` op (and is
captured by the FULL decode CUDA graphs at boot), so the existing compile cache stays valid.

Verify after boot (serve log, every worker): `TC-GDNGLUE pid N: installed (QwenGatedDeltaNetAttention.forward_cuda
patched)` and, at graph capture, `TC-GDNGLUE pid N: fused GDN verify active (BV=32 warps=4 stages=3 N=1 rows=5/5)`.
`collective_rpc("tc_gdnglue_status")` -> fused_calls (eager + captured calls; graph replays are not counted), fallback
reasons (prefill / mixed batches are expected), errors 0. Then: greedy.py hashes must equal the baseline
(the port is meant to be bit-identical, so even sampled outputs with fixed seeds must match), 7/7 hard questions,
roundtime.py ms/step 1/2/4 users.

Knobs (none are VLLM_*): `TC_GDNGLUE=1` enable; `TC_GDNGLUE_MAXREQ=64` counter buffer (requests per launch);
`TC_GDNGLUE_BV` V-block override (only if the GPU check says exact); `TC_GDNGLUE_LOG=1` log each fallback reason once.
Rollback: unset `TC_GDNGLUE` (or the WEXT change) and restart.

## Admission (anything else -> the original forward_cuda, untouched)

Static per layer: 4 k-heads / 12 v-heads per rank, head dims 128, conv width 4 without bias, SiLU conv, no spec
replay, no native CUDA verifier (`kernel_config.sm70_gdn_verify` off - default), no DFlash2 paths, mixed-QKV decode
on, full-forward auto route on and no 0.0.3/spec-core boundary, norm = `RMSNormGated.forward_native` with
`sm70_rmsnorm_gated_exact` (128 cols, fp16 weight, no group, norm before gate).
Per call: active spec batch with only spec rows (no prefill / plain decode in the batch), no DDTree, mixed_qkv fp16
row-major (contiguous or the stock 5/20-row strided case), 2..16 or 20 verify rows (the stock mixed-QKV verifier
range), fp32 ssm state, fp16 conv state, z contiguous, <= 64 rows, <= 16 tokens per request, <= TC_GDNGLUE_MAXREQ
requests. The stock projection (Part 1 of forward_cuda) is copied verbatim and always runs first; if the per-call
check fails after it, the stock Part 2/3 (also copied verbatim) finish the call.

Differences that are never read: mixed_qkv keeps the pre-conv values (stock overwrites it in place; nothing reads it
after the core), and when the stock verify takes its non-direct output path (verify rows not 5/20) the rows of PAD
requests are zeros here instead of uninitialized memory (padding requests are never sampled).

## Files

- `tc_gdnglue.py` - hook, admission, launcher, worker-extension class `TCGdnGlueExt`, `status()`
- `tc_gdnglue_kernels.py` - the Triton kernel (conv prologue, recurrence copy, exact norm epilogue)
- `tests/test_tc_gdnglue_cpu.py`, `tests/gdnglue_offline_check.py`, `tests/test_tc_gdnglue_install.py`,
  `tests/tc_gdnglue_gpu_check.py`; logs in `logs/gdnglue-*.log`; compiled artifacts in `build/`
