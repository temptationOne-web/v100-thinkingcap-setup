"""TC_NOSYNC (staged 2026-10-30 night, NOT live): remove the one host<->GPU sync per MTP round in the sampled path.

WHY (evidence: next-tps/tools/trace_phases.py + trace_syncs.py on logs/prof-mp0):
  Every MTP round with a sampled request (temperature > 0, top_k 20, top_p 0.95 = ThinkingCap generation_config, which
  JARVIS chats use because Hermes sends no temperature) runs the SM70 compact top-k/top-p routine
  vllm/v1/sample/ops/topk_topp_triton.py::_apply_top_k_top_p_compact. It ends with

      if reference_rows.any():          # <- aten::is_nonzero -> cudaStreamSynchronize

  which blocks the worker's host thread until the whole target verify graph has finished on the GPU. From then on the
  host is behind the GPU for the rest of the round: the eager sampler/rejection kernels, the 4 draft-graph launches with
  their eager glue, the next round's input prep and the 1,480-node target-graph launch all run while the GPU waits.
  prof-mp0 (1 user, temp 0.6, 23 rounds, rank 0): round 34.7 ms profiled, GPU busy 20.4 ms; GPU idle 3.2 ms between the
  sync and the first draft launch and 7.2 ms from the drafts to the next target launch. That is also why sync vs async
  scheduling measured identical on 10-26: this sync defeats async scheduling.
  Greedy requests (temperature 0: priority-client, onetps.py, thinkgate) never reach this routine (vLLM resets top_k/top_p for
  greedy), which is why onetps.py (temp 0) does not see the stall.

WHAT this changes (only when TC_NOSYNC=1, only on the eager compact route, never inside CUDA graphs):
  * The host decides admission from the HOST copies of top_k/top_p (SamplingStates.top_k.np / top_p.np, numpy, no sync).
    If any row needs the full-vocabulary route for a structural reason (top_k <= 0 or >= 128 with top_p < 1), the stock
    function runs unchanged (with its sync).
  * Otherwise the data-dependent boundary rows (a top-p cumulative within 8*128*eps of the boundary, more than 128 ties
    at the top-k cutoff, a non-finite tail of the shortlist = grammar-masked rows) are decided on the 128-candidate
    shortlist instead of re-running the full-vocabulary sort after a sync. This is the same rule upstream ships as
    kernel_config.sm70_sparse_target_reference_check=False ("no host synchronization; those rare boundary cases are
    decided on the compact candidates ..., still a valid top-k/top-p sample").
  * Rows that were NOT flagged (the normal case) get bit-identical processed logits to stock.
  * Greedy rows (temperature 0) are never affected: they are passthrough rows (k disabled, p = 1), and even when a
    greedy row sits in a mixed batch, top-k/top-p filtering never removes the argmax.
  * Telemetry without a sync: the per-call flagged-row count is copied non_blocking into pinned memory and read a few
    calls later (event.query()). status(): calls, nosync_calls, stock_calls (+reason), flagged_rows_seen.

Install: imported by tc_nosync_ext.TCNoSyncExt (worker extension), self-installs when TC_NOSYNC=1. Off = stock bit for
bit (install() not called). No torch.compile / cache-key change: the sampler is eager and nothing here is traced.
"""
import os

_ON = os.environ.get("TC_NOSYNC", "0") == "1"
_LOG_EVERY = int(os.environ.get("TC_NOSYNC_LOG_EVERY", "20000") or 0)
_ALLOW_CPU = os.environ.get("TC_NOSYNC_TEST_CPU", "0") == "1"   # CPU tests only (TRITON_INTERPRET=1)
_CAPACITY = 128
_stats = {"installed": False, "calls": 0, "nosync_calls": 0, "stock_calls": 0, "stock_reason": {},
          "flagged_rows_seen": 0, "flag_reads": 0, "errors": 0}
_host_ok = [None]   # set by the SamplingStates wrapper for the duration of one apply_top_k_top_p call


def _say(msg):
    print(f"TC-NOSYNC pid {os.getpid()}: {msg}", flush=True)


def status():
    return dict(_stats)


class _FlagTelemetry:
    """Non-blocking per-call count of rows the stock code would have sent to the full-vocabulary route."""

    def __init__(self, torch, slots=64):
        self.torch = torch
        self.slots = slots
        self.buf = None
        self.events = [None] * slots
        self.i = 0

    def record(self, reference_rows):
        torch = self.torch
        if self.buf is None:
            self.buf = torch.zeros(self.slots, dtype=torch.int64, pin_memory=True)
        s = self.i % self.slots
        ev = self.events[s]
        if ev is not None:
            if ev.query():                      # finished long ago in steady state; never blocks
                _stats["flagged_rows_seen"] += int(self.buf[s])
                _stats["flag_reads"] += 1
            # if not finished yet: drop this sample (telemetry only)
        self.buf[s : s + 1].copy_(reference_rows.sum().view(1), non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self.events[s] = ev
        self.i += 1


def install() -> bool:
    import numpy as np
    import torch
    from vllm.v1.sample.ops import topk_topp_triton as T
    from vllm.v1.worker.gpu.sample import states as S

    if getattr(T, "_tc_nosync", False):
        return True
    orig_compact = T._apply_top_k_top_p_compact
    orig_states_apply = S.SamplingStates.apply_top_k_top_p
    tele = _FlagTelemetry(torch)

    def apply_top_k_top_p(self, logits, expanded_idx_mapping, idx_mapping_np):
        # Host-side admission from numpy copies (no device read).
        ks = self.top_k.np[idx_mapping_np]
        ps = self.top_p.np[idx_mapping_np]
        no_top_p = ps >= 1.0
        passthrough = ((ks <= 0) | (ks >= self.vocab_size)) & no_top_p
        admitted = (ks > 0) & (ks < _CAPACITY)
        _host_ok[0] = bool(np.all(admitted | passthrough))
        try:
            return orig_states_apply(self, logits, expanded_idx_mapping, idx_mapping_np)
        finally:
            _host_ok[0] = None

    def compact(logits, k, p, mask_value):
        _stats["calls"] += 1
        ok = _host_ok[0]
        if ok is not True or (not logits.is_cuda and not _ALLOW_CPU) or (logits.is_cuda and torch.cuda.is_current_stream_capturing()):
            reason = "no_host_info" if ok is None else ("not_admitted" if ok is False else "capture_or_cpu")
            _stats["stock_calls"] += 1
            _stats["stock_reason"][reason] = _stats["stock_reason"].get(reason, 0) + 1
            return orig_compact(logits, k, p, mask_value)
        _stats["nosync_calls"] += 1
        # ---- verbatim from the engine's _apply_top_k_top_p_compact (vllm-main-20261030, topk_topp_triton.py:1130-1229)
        # ---- up to the reference_rows decision; only the fallback block is replaced. Keep in sync on engine upgrades.
        capacity = _CAPACITY
        vocab_size = logits.shape[1]
        no_top_p = p >= 1.0
        passthrough = ((k <= 0) | (k >= vocab_size)) & no_top_p
        admitted = (k > 0) & (k < capacity)
        shortlist_k = torch.where(admitted, k, torch.ones_like(k))

        values, indices = logits.topk(capacity, dim=-1, sorted=False)
        values, indices = T.sort_topk_with_vocab_ties(values, indices, vocab_size=vocab_size)
        cutoff = values.gather(1, (capacity - shortlist_k.long()).unsqueeze(1))
        reference_rows = (values[:, 0] >= cutoff[:, 0]) | ~torch.isfinite(values[:, -1])
        reference_rows |= torch.isnan(values[:, 0])
        values.masked_fill_(values < cutoff, -float("inf"))
        probabilities = values.softmax(dim=-1)
        cumulative = probabilities.cumsum(dim=-1)
        boundary = 1 - p.unsqueeze(1)
        margin = 8 * capacity * torch.finfo(torch.float32).eps
        reference_rows |= ~no_top_p & ((cumulative - boundary).abs() <= margin).any(dim=-1)
        reference_rows |= ~admitted
        reference_rows &= ~passthrough
        # CHANGED: stock keeps reference rows untouched here and re-computes them after `if reference_rows.any()`
        # (a device->host sync). We rewrite every non-passthrough row from the shortlist instead.
        keep_rows = passthrough
        remove = (cumulative <= boundary) & ~no_top_p.unsqueeze(1)
        remove[:, -1] = False
        values.masked_fill_(remove, -float("inf"))
        result = logits
        T._clear_compact_rows_kernel[(logits.shape[0], T.triton.cdiv(logits.shape[1], 4096))](
            result, keep_rows, result.stride(0), result.shape[1], BLOCK=4096,
        )
        T._scatter_compact_rows_kernel[(logits.shape[0],)](
            result, keep_rows, values, indices, result.stride(0), values.stride(0), indices.stride(0),
            CAPACITY=capacity, num_warps=4,
        )
        try:
            tele.record(reference_rows)
        except Exception:  # noqa: BLE001 telemetry must never break sampling
            _stats["errors"] += 1
        if mask_value != float("-inf"):
            result.masked_fill_(torch.isneginf(result), mask_value)
        if _LOG_EVERY and _stats["calls"] % _LOG_EVERY == 0:
            _say(f"status {status()}")
        return result

    T._apply_top_k_top_p_compact = compact
    S.SamplingStates.apply_top_k_top_p = apply_top_k_top_p
    T._tc_nosync = True
    _stats["installed"] = True
    return True


if _ON:
    try:
        _say(f"installed={install()}")
    except Exception as exc:  # noqa: BLE001 never take the worker down; the gate checks the log line
        _say(f"install FAILED {exc!r}")
