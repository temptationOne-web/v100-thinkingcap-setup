# tc_draft (item 2): draft-step / host-gap work, kept lean on owner order (2026-10-26)

## Finding: the "~2.4 ms PLE-layer wait" is graph-launch starvation, not PLE
From the prof-mp0 rank0 trace (best stack, 1 user):
- The target verify graph has 1480 nodes. Its `cudaGraphLaunch` takes **2.69 ms of host time** (1.8 us/node). The four
  draft graphs are 60-70 nodes at 130-200 us each.
- The GPU executes the graph's first part while the host is still submitting it, then idles. The gap ends about 90 us after
  `cudaGraphLaunch` returns (2 of 2 samples: return +1217 -> gap end +1306; return +2418 -> gap end +2509).
- It always lands at the same node: a GDN layer's `down_replicated_reduce` -> `hc_up_batch`. No PLE kernel runs there.
  That explains why the spin-wait port (#11) and the FP8 lane (#10) changed nothing.
- CUPTI (the profiler) inflates per-node launch cost. So the unprofiled size of this stall is unknown. It is bounded by the
  ~3-5 ms of unprofiled idle per round (26.2 ms step vs 21-23 ms GPU busy).
- Host metadata gaps before the target graph (pageable HtoD + QSA/GDN metadata kernels) total ~1-1.5 ms per round in the
  trace (individual gaps of 60-390 us between _build_qsa_metadata_kernel, _dflash2_gdn_group_metadata_kernel and pageable
  HtoD copies). These are also host-bound, not GPU work.

## What is built
- `ext/tc_draft.py`: `WEXT=tc_draft.TCDraftExt TC_DRAFT=1` wraps tc_ext.TCExt and times every CUDA graph replay on the host
  (no sync, no numerics change, passthrough when off). It logs p50/p90 host us per graph. RPC: `tc_draft_stats()`.
- `ext/tests/test_tc_draft_cpu.py`: CPU test, PASS.
- `ext/tests/gpu_check_tc_draft.sh`: tomorrow's steps.

## Expected gain / next step (not implemented, by order)
If the unprofiled target-graph launch is still >~1 ms, the fix is structural, not a kernel: fewer graph nodes, or
launching the target graph before host metadata prep finishes. Up to ~2 ms/round (~7%) is at stake at most; the realistic
gain is unknown until the timer runs. Draft steps: 4 x 130-200 us host launch, which mostly overlaps GPU work. No code-level
draft change was attempted.
