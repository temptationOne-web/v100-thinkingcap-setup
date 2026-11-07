# TC-GLUE: ThinkingCap sync glue (2026-10-26, CPU-only build night; GPU A/B 06:55 10-27)

Status: kernel built for sm_70, port module + CPU tests DONE, GPU test + A/B line READY. Nothing is live; default OFF.
No engine file was edited (runtime hooks only; /opt/1cat/vllm-main-20261025/PORTS.txt unchanged). No GPU was used.

## What is fused
Per hyper-connection boundary at decode (96 per target forward: 48 after the attention out_proj = "A", 48 after the
MoE = "B"; plus 2 per MTP draft step), ONE kernel replaces:

| live today (per boundary)                                                                | fused |
|------------------------------------------------------------------------------------------|---|
| A: `sm70_cross_device_reduce_1stage_push` / ARX two-shot (28 us at M=5 profiled) + Triton `_hc_combine_norm_kernel` (4.5 us) | `tc_glue_c.fused` mode 2 |
| B: `sm70_cross_device_reduce_sum2_1stage_push` ONE-SHOT (32 us at M=5; ARX never replaced it) + `_hc_combine_norm_kernel` | `tc_glue_c.fused` mode 2 (half add fused) |
| PLE layer combine + final mixers: sum2 one-shot AR, then live combine / combine+norm                              | `tc_glue_c.fused` mode 0 (two-shot AR), live epilogue |

The kernel (`tc_glue_src/tc_glue.cu`, one CTA of 320 threads per row, thread t owns 16-byte pack t): ARX two-shot
all-reduce (reduce-scatter to the quarter owner, owner reduces in rank order like the live kernels, all-gather of the
owner's bits; 0x7f7f sentinel protocol, 2 epochs, graph-replay safe) + HC combine `f16(fma(2*sigmoid(inj/4), ar, res))`
for the 4 streams + grouped Gemma RMSNorm with the production Triton kernel's exact reduction tree (read from the cached
PTX/SASS in cache-new: 8x512 tile tree for M != 1, 4x1024 tree for M == 1; `FFMA(sum, 0x39cccccd, 1e-6)` + `MUFU.RSQ`;
sigmoid/combine/affine ops as in the SASS). Built with `--fmad=false`, every FFMA explicit. Same ptxas (12.8.93) as
Triton's. The HC down/silu/up/gate-mix projections (weight-bandwidth bound, ~30 us/boundary) are NOT touched: they
stay on the stock fp16 fused route via `module._project(xn)`.

Hooks (`tc_glue.py`, loaded by `tc_ext.TCExt` when `TC_GLUE=1`): upstream's own HCX plumbing is wired without its
NVLink runtime (out_proj/o_proj `reduce_results=False` + `qwen38_sm70_hcx_output_projection`, MoE runner
`output_transform -> qwen38_sm70_hcx_moe_output` two-plane payload, consumer HC modules `_partial_inputs`), and every
consumer's `_hcx` runtime is this port's `_Runtime` (`torch.ops.tc_glue.ar_combine_norm` then `_project`).
`GatedResidual._reduce_partial` -> `torch.ops.tc_glue.ar2` for the PLE layer / final mixers. Target wired after
`load_weights` (wrapper of `model._maybe_enable_sm70_peer_paths`), MTP draft after its `load_weights` (`TC_GLUE_DRAFT=1`).
Dispatch inside the ops: capturing a CUDA graph and 1 <= M <= `TC_GLUE_MAXM` -> fused; else live-equivalent fallback
(`tensor_model_parallel_all_reduce(_sum2)` + `torch.ops.vllm.qwen4_exp_hc_combine_norm`). Prefill (M > 40) keeps the
upstream large-M path (all-reduced projection / reduced MoE payload). Why not upstream HCX itself: its exchange is
NVLink-sized (fp32+tag words one-shot to 3 peers + lora/hidden exchanges, ~1.2 MB through the PCIe root complex per
boundary at M=5 = ~100 us); it is also admitted only on NVLink (`requires_two_direct_nvlink_peers_per_rank`).

## Expected gain (from logs/prof-bridge1, 1 user, M=5 verify + 4 M=1 draft steps per round, 26.2 ms/round)
- B boundaries: one-shot sum2 32 us -> two-shot ~14-16 us (ARX measured 24 KB: 27 -> 13.5 us): ~15 us x 48 = ~0.7 ms
- A boundaries: ARX AR + separate combine_norm (4.5 us + launch gap) -> fused: ~3-4 us x 48 = ~0.17 ms
- draft steps (M=1, 8 boundaries/round): ~0.06 ms; minus the new `torch.cat` payload copy (~2 us x 48 = ~0.1 ms)
=> ~0.8-0.9 ms/round at 1 user (~3%, 26.2 -> ~25.3 ms; 141 -> ~146 tok/s code), more at 4 users (M=20: the live
sum2 is ~100 us/call and fused rows are independent; set TC_GLUE_MAXM=20..40 to cover it; ~2 ms of 45 ms = ~5%).
Measured numbers: tests/glue_gpu_check.py perf (isolated chain) and the variants5 A/B (in-engine).

## Enable / disable
    TC_GLUE=1 bash /opt/1cat/tcflash-trial/run-tc-best.sh          # (WEXT=tc_ext.TCExt is the default there)
    TC_GLUE=0 (or unset) = today's stack, bit for bit (install() is a no-op).
Optional: TC_GLUE_MAXM=16 (1..40), TC_GLUE_TREE=auto|512|1024, TC_GLUE_DRAFT=1, TC_GLUE_EAGER=1 (tests), TC_GLUE_LIB.
Verify after boot (serve.log, every rank): "TC-GLUE install: installed", "TC-GLUE fused all-reduce+combine+norm ready on
rank r", "TC-GLUE linked 48 target layers", "TC-GLUE linked 1 mtp.layers". Live counters: collective_rpc
`tc_glue_status` (worker extension) -> fused_norm / fused_ar > 0 after capture, fallback_eager = warmup/prefill only.
Build: `bash ext/build_glue.sh` (CPU only, ~1 min) -> ext/build/tc_glue_C.so. Never overwrite a .so a running server loaded (GLUE_OUT=).

## Tomorrow (06:55+, GPUs free, speed investigator)
1. `cd /opt/1cat/tcflash-trial && CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 600 /opt/1cat/vllm-main-20261025/.venv/bin/python tests/glue_gpu_check.py all`
   Expect "RESULT: ALL BITWISE" (fused == live for M=1..40 in graphs + eager fallback + batch invariance) and the perf
   lines (us/boundary live vs fused). Any mismatch: STOP, report the line (M, kind, which output), do not A/B.
2. A/B (one boot each, fresh compile on the first TC_GLUE=1 boot = new cache key factor "tc_glue-v1", ~10 min):
   `VARIANTS="glue1:TC_GLUE=1 base:TC_GLUE=0" LAUNCHER=run-tc-best.sh bash variants5.sh`   (-> logs/variants5.tsv)
   Then with 4 users covered: `VARIANTS="glue40:TC_GLUE=1,TC_GLUE_MAXM=40" LAUNCHER=run-tc-best.sh bash variants5.sh`
   and the priority-client-exactness gate of the investigator's stack on top (`run-tc-inv.sh` + `TC_GLUE=1`): greedy hashes
   (greedy.py) must be identical to TC_GLUE=0 for the same prompts/batch composition.
3. Keep if: hard questions unchanged, greedy identical, 1u ms/step down. Snapshot the compile cache after a good boot.

## CPU results tonight (ext/tests/glue_cpu_check.py, new venv, no GPU)
- math: Triton-layout model (tiles/warps/lanes/smem stages/masked tiles from the cached PTX) vs the fused kernel's
  warp model: 30/30 rows bitwise for both trees; negative control (different association) differs in 28/30
  (the 2 agreeing rows are sparse "zeros" rows); sum2 + combine model identical.
- install: TC_GLUE=1 install with the engine modules imported on CPU: all 5 patch targets present, ops registered,
  fake impls right, idempotent, mock 2-layer model links with the right attributes, ineligible layer refused. PASS.
- SASS (cuobjdump of the built .so vs the production cubin): `MOV 0x39cccccd; FFMA(sum, rcp, 1e-6); MUFU.RSQ`, combine
  `FFMA(g, ar, res)`, chains `FMUL e1*e1; FFMA x7`, butterflies, `FFMA(x,-0.25,RZ); FMUL log2e; FSETP -126 / MUFU.EX2;
  FADD 1; FSETP 8.5e37 / FMUL 0.25 / MUFU.RCP / FSEL / FMUL; FADD s+s` all present. Only difference: ptxas kept the
  general denormal/abs guards of `div.full` in our build (`|x| < 1.17e-38 -> *2^24`, never taken: the denominator is
  ex2()+1 >= 1 or inf/NaN) -> same value. The approximate ops (ex2.approx, div.full, rsqrt.approx.ftz) are the same
  SASS opcodes; their bitwise equality is confirmed only by tomorrow's GPU test.
- Build: ext/build/tc_glue_C.so (nvcc 12.8, sm_70, --fmad=false); 7 kernel instantiations.

## Risks / notes
- Compile cache: the traced graph changes (upstream HCX ops instead of in-projection all-reduces) and a new AOT key
  factor is appended, so the first TC_GLUE=1 boot compiles fresh (cache-new keeps the TC_GLUE=0 entries). Snapshot
  after a good boot; a fresh compile re-autotunes the other inductor pieces (the usual lottery).
- Determinism: rows are independent, fixed trees, rank-order all-reduce -> deterministic and batch invariant for a
  given tree. The LIVE kernel is itself not invariant between M == 1 (1024 tree) and M >= 2 (512 tree); the port
  reproduces that (bit-identical to live). TC_GLUE_TREE=512 makes M=1 use the M>=2 tree (invariant across M, but not
  bit-identical to live at M=1). The priority-client solo scheduler / pins being ported in parallel are untouched; no new
  nondeterminism source (no atomics on data, no owner-order dependence beyond the live kernels' rank order).
- IPC: 800 KB buffer per GPU, collective setup in CustomAllreduce.__init__ (all 4 ranks or none); chains with ARX.
- Fallback paths are live-equivalent (same ops), so a non-admitted communicator or M > MAXM never changes numerics.
- Upstream's `qwen38_sm70_hcx_moe_output` builds the two-plane payload with one `torch.cat` (~2 us) per MoE layer;
  a tuple-free variant would remove it (follow-up).
- Not fused: the HC down/up GEMVs (bandwidth bound; rank-sharding them needs an extra exchange = HCX's design),
  the final mixer's norm, prefill.
Files: ext/tc_glue.py, ext/tc_glue_src/tc_glue.cu, ext/build_glue.sh, ext/build/{tc_glue_C.so,hc_norm_weights.pt},
ext/tests/glue_cpu_check.py, tests/glue_gpu_check.py, ext/tc_ext.py (TC_GLUE hook; .pre-glue-20261026 backup).
