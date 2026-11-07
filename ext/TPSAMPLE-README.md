# tc_sampler - TP-local exact MTP sampler for ThinkingCap ("each card picks its own best words")

Built 2026-10-26 evening, CPU only (no GPU used). Target engine `/opt/1cat/vllm-main-20261025`, ThinkingCap TP4,
MTP k=4, `draft_sample_method=probabilistic` (run-tc-best.sh). Nothing in the engine, `tc_ext.py`, Ornith or Swift
trees was edited. Default **off** (`TC_TPSAMPLE=1` enables).

Lineage: Ornith's `/opt/1cat/ornith122-trial/ports/tpsamp` (orn_tpsamp.py, exactness suite suite*.log) did TP-local
sampling for the NON-spec sampler and was rejected on 10-23 ("no gain; the earlier 4% was my bench's min_tokens"):
Ornith ran k=1 with a single 1-row sampler call per step. ThinkingCap is different: every MTP round all-gathers the
full 248,320-token logits **five times** (target verify M=5 rows + 4 draft steps). This port keeps Ornith's idea and
exactness discipline but targets the spec path (draft Gumbel sampling + rejection sampler), which Ornith's port did
not cover.

## Why (measured in the existing profiles, rank 1, median per round)

| per MTP round | 1 user (prof-mp0) | 4 users (prof-bridge4) |
|---|---:|---:|
| NCCL all-gather of logits (target ~500 us at 1u + 4 x ~125 us draft) | 1.05 ms | 2.74 ms |
| other small all-gathers (unchanged) | 0.10 ms | 0.08 ms |
| vocab-wide sampling kernels (temperature, topk, block stats, rejection, resample, gumbel) | 0.28 ms | 0.38 ms |

The all-gather durations are the same on all 4 ranks (not skew): NCCL RING_LL moves ~124 KB/row/rank at ~1 GB/s on
this PCIe box; a tiny all-gather costs ~23 us.

## What it does

Draft steps (`EagleSpeculator._sample_draft`; runs inside the draft CUDA graphs):
- each rank computes its 62,080-token shard of the draft logits (same `head.quant_method.apply` call the stock
  `compute_logits` makes before gathering), runs the stock Gumbel arithmetic on the shard (`tc_draft_gumbel_kernel`:
  same temperature division, same `gumbel_noised_argmax`, Philox keys = **global** int32 ids, exactly the stock
  `block` offsets), stores the temperature-applied shard in a local draft buffer and returns its best (value, id);
- one all-gather of `[rows, 2]` per step; global winner = max value, lowest id (= stock per-block argmax + torch.argmax
  first-occurrence order).
Target verify (`GPUModelRunner.sample`, eager):
- each rank: fp32 copy of its logits shard, the stock temperature kernel, local top-128 (value, global id), the
  matching draft logits (draft step = row position), the draft value at the proposed token, and its shard's draft
  (max, sum-exp) -> one ~1.5 KB/row packet -> ONE all-gather;
- every rank then computes the same thing from the same gathered bytes (all ops deterministic, no atomics): global
  top-128 -> the stock SM70 compact top-k/top-p routine (`_apply_top_k_top_p_compact`, lines copied verbatim, incl.
  `sort_topk_with_vocab_ties`, softmax, cumsum, its "reference row" detection) -> rejection loop, residual resample
  and bonus token on the kept set (`tc_target_reject_kernel`, expressions copied from `rejection_sampler_utils.py`).

## Exactness - what is identical and what is "distribution-exact"

Bit-identical to stock (given identical logits):
- every **draft token** (any temperature) and every value of the draft logits buffer (shards == stock row slices);
- the **kept set and kept values** of the target top-k/top-p processing (the stock routine runs on the same 128-entry
  shortlist; entries outside it are -inf in stock as well);
- **greedy requests** (temperature 0): target argmax over the processed row (max value, lowest id), acceptance,
  bonus -> bit-identical sampled tokens. The only greedy fallback: an ambiguous row whose maximum is tied;
- the **bonus / prefill-row sample** and every Gumbel noise value (same Philox seed/offset/key).

Distribution-exact (probabilistic requests): the rejection test `log p(x) > log u + log q(x)` and the residual
`max(p - q, 0)` need the target and draft log-normalisers (logsumexp). Stock sums them over fixed 8,192-token vocab
blocks of the full row; here the target one is summed over the kept slots grouped by the same 8,192-blocks and
combined with the stock `_compute_global_lse` formula (bit-identical whenever no vocab block holds 3+ kept tokens), and
the draft one per rank shard (the 62,080 shards are not aligned to 8,192 blocks) and combined across ranks with the same
formula. Mathematically the identical rule; numerically a different fp32 summation order, so an accept/reject or
residual decision can differ from stock only when the compared log-probabilities are within a few fp32 ulps (~1e-7
relative) - a probability-~1e-7 event per decision, with no bias in either direction. Seeds, Philox keys and the
candidate set are the same, so the two samplers draw from the same distribution with the same random numbers.

Stock "reference rows" (a top-k cutoff tied past the 128-shortlist / non-finite values -> always fall back; a top-p
cumulative within 8*128*eps of 1-p -> stock re-decides with a full-row sort, i.e. the same rule with another rounding):
by default the margin case keeps the compact decision (distribution-exact); `TC_TPSAMPLE_STRICT=1` sends those steps
to the stock path too.

Fallback = the stock sampler (full logits all-gather) for the whole verify step, after re-assembling the stock draft
logits of the batch's requests from the local shards (one all-gather, ~1 MB per request per step), for: grammar /
structured output, logprobs or logprob token ids, logit bias / allowed ids / min_tokens, presence / frequency /
repetition penalties, bad words, min_p, NaN counting, synthetic acceptance, top_k missing or >= 128, top_p without
top_k, the cases above, and when stock's own `sm70_greedy_verify` route would take the step. JARVIS defaults
(generation_config top_k 20, top_p 0.95, no penalties, tc_front sets no presence penalty) are on the fast path. If a
traffic mix often hits a fallback reason (e.g. presence_penalty), leave the port off: every such step pays the
re-assembly on top of the stock path.

No rank-local error handling around collectives: the two paths issue different collectives, so a "catch and fall
back" on one rank would hang NCCL; every branch is a function of replicated host state or the gathered bytes.

## Tests done tonight (CPU only)

- `tests/test_tc_sampler_cpu.py` (Triton interpreter, simulated TP=4, stock `gumbel_sample`,
  `_apply_top_k_top_p_compact`, `rejection_sample` as reference; vocab 4 x 16,384 and the real 4 x 62,080):
  56 cases, 175 draft steps, 4 drafts each: **draft tokens and draft-logit shards bit-identical in all cases**;
  **greedy requests 27/27 identical**; probabilistic requests 98/99 identical (1 resample-token difference, case 19 of
  `logs/tpsample-cpu.log`, same accept count - attributed to the different lse summation, NOT yet proven: the float64
  replay `tests/debug_tpsample_case.py --seed 3 --case 19` was cut off, re-run it); 3 whole-step fallbacks in the
  strict-semantics run, 0 in default mode. Logs: `tpsample-cpu.log` (39 cases, 4x16384, written by the earlier strict
  build), `tpsample-cpu-default.log` (14 cases, default mode), `tpsample-cpu-realvocab.log` (3 cases, real
  248,320 vocab). Several runs were killed externally at ~22:15 (logs end without a TOTAL line); counts above are
  tallied from the per-case lines.
  Negative controls: local ids as Philox keys -> 4/4 cases draft mismatch (`tpsample-cpu-negctl-keys.log`); draft
  normaliser +0.5 -> 2/10 probabilistic mismatches (`tpsample-cpu-negctl.log`).
  Note: the interpreter has no libdevice; the test maps `log1p` to `log(1+x)` for BOTH stock and port.
- `tests/tpsample_offline_check.py`: all kernels compiled for sm_70 (fp32 and fp64 Gumbel variants) with no spills
  (`build/tc_sampler_*.cubin/.sass`); hooks install in both import orders; every engine name used exists
  (`logs/tpsample-offline.log`).

Still to run on CPU (after 00:00, rig power budget: `nice -n 19`, one process at a time):
```
cd /opt/1cat/tcflash-trial/ext && CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 PYTHONPATH=$PWD nice -n 19 \
  /opt/1cat/vllm-main-20261025/.venv/bin/python tests/debug_tpsample_case.py --seed 3 --case 19 > logs/tpsample-debug-case19.log
```
(prints, for each mismatching request, the float64 noised values of the stock-chosen and port-chosen tokens; nearly
equal values = rounding flip, a large gap = a bug).

## GPU checks for tomorrow (06:55+; refuse to run in the busy window unless --force)

1. One free GPU, simulated TP4 vs the real CUDA stock kernels (bit checks of drafts / greedy, counts for sampled):
```
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=/opt/1cat/tcflash-trial/ext /opt/1cat/vllm-main-20261025/.venv/bin/python \
  /opt/1cat/tcflash-trial/ext/tests/tc_sampler_gpu_check.py --cases 300 2>&1 | tee /opt/1cat/tcflash-trial/ext/logs/tpsample-gpu.log
```
2. All 4 GPUs (engine stopped), real NCCL: no rank divergence + the all-gather times removed/added on this box:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/opt/1cat/tcflash-trial/ext /opt/1cat/vllm-main-20261025/.venv/bin/torchrun \
  --nproc_per_node 4 /opt/1cat/tcflash-trial/ext/tests/tc_sampler_gpu_check.py --tp4 --cases 100 \
  2>&1 | tee /opt/1cat/tcflash-trial/ext/logs/tpsample-tp4.log
```

## Expected gain

- NCCL: 1.05 ms -> 5 x ~23-30 us = ~0.13 ms per round at 1 user (-0.9 ms); 2.74 ms -> ~0.15 ms at 4 users (-2.6 ms).
- vocab-wide kernels work on 1/4 of the vocab, the full-row top-k/top-p rewrite and full-row block statistics go away:
  -0.1 to -0.2 ms; extra small launches (shard top-k, packet assembly, 512-candidate merge) add back ~0.1 ms of eager
  launch time in the target stage.
- **Estimate: 1 user ~0.7-1.0 ms of 26.2 ms/step (3-4%), 4 users ~2-2.6 ms of 44.2 ms (5-6%).** The tp4 GPU check
  prints the real per-step all-gather times; the server A/B (roundtime.py, 1/2/4 users) is the judge. Ornith's
  version gave nothing because it had no 5x logits gathers - check before believing.

## How the investigator enables it

Self-installs on import with `TC_TPSAMPLE=1`; needs to be imported in the workers:
```
TC_TPSAMPLE=1 WEXT=tc_sampler.TCSamplerExt TC_TPSAMPLE_BASE_EXT=tc_ext.TCExt bash /opt/1cat/tcflash-trial/run-tc-best.sh
# both ports: see GDNGLUE-README.md (WEXT=tc_gdnglue.TCGdnGlueExt TC_GDNGLUE_BASE_EXT=tc_sampler.TCSamplerExt ...)
# or add `import tc_sampler  # noqa` to tc_ext.py (no-op unless TC_TPSAMPLE=1)
```
Requires DRAFT_SAMPLE=probabilistic (default in run-tc-best.sh); with greedy drafts the draft hook stays on the stock
path and the target hook falls back ("drafts_not_local"). No torch.compile / cache change (sampling is outside the
compiled graphs; the draft graphs are re-captured at every boot).

Verify after boot: serve log `TC-TPSAMPLE pid N: installed EagleSpeculator hooks`, `... GPUModelRunner.sample hook`,
`... draft steps TP-local: True (tp=4)`; `collective_rpc("tc_tpsample_status")` -> target_local grows, fallback reasons
(expect few: tie rows / prefill-only steps are not counted), errors 0. Then: greedy.py hashes identical to baseline
(greedy must be bit-identical), 7/7 hard questions, roundtime.py 1/2/4 users, acceptance rate (draft acc/round) within
noise of the baseline (distribution-exact => same acceptance statistics).
Knobs: `TC_TPSAMPLE_STRICT=1`, `TC_TPSAMPLE_TARGET=0` (drafts only; target stock + re-assembly - debugging),
`TC_TPSAMPLE_LOG=1`. Rollback: unset `TC_TPSAMPLE` and restart.

## Files

- `tc_sampler.py` - hooks, admission, rank-local / replicated building blocks (also used by the tests),
  `TCSamplerExt`, `status()`
- `tc_sampler_kernels.py` - draft Gumbel on the shard, draft shard statistics, replicated rejection + resample kernel
- `tests/_tpsample_cases.py` (shared case generator), `tests/test_tc_sampler_cpu.py`, `tests/tpsample_offline_check.py`,
  `tests/tc_sampler_gpu_check.py`; logs `logs/tpsample-*.log`; artifacts `build/tc_sampler_*`
