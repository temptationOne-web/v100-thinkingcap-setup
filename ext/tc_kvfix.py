"""ThinkingCap Mamba-align KV fix (2026-10-27): "AssertionError: num_required_blocks N < len(req_blocks) N+1".

Crash (EngineCore dead, ~8 min outage): 07:13 "8 < 9" and 15:10 "15 < 16", both in
  orn_solo_sched_v4._solo_step -> Scheduler.schedule -> KVCacheManager.allocate_slots -> MambaManager.allocate_new_blocks
  (vllm-main-20261025 single_type_kv_cache_manager.py, align branch: assert num_required_blocks > len(req_blocks)).

Root cause. In mamba_cache_mode=align the GDN block list of a request is sized for
  T = num_tokens_main_model = num_computed_tokens + num_new_tokens   (optimistic: includes every in-flight draft)
  num_required_blocks = cdiv(T, 816) + num_speculative_blocks
and the align branch ASSUMES T never decreases between two schedules of the same request. With async scheduling + MTP,
T at a request's schedule is  committed + 1 + sum(1 + S_j over its in-flight steps) + S_next, so between two schedules
  T2 - T1 = 1 + S_next - (sum of tokens REJECTED in the request's steps that finished in between).
Stock vLLM schedules a decoding request on every step, so exactly one step finishes in between and T2 - T1 >= 1.
Our no-mixing solo scheduler (orn_solo_sched_v2/v4 -> TCSoloStepScheduler) HOLDS every decoder on prefill steps (and
on priority-client-solo steps). While a decoder is held, TWO of its verify steps finish (the one in flight at its last schedule
and that schedule's own step): with K=4 drafts both fully rejected, T2 - T1 = 1 + S_next - 8 = -3 (-4 with adaptive
depth 3). Rejection rollbacks of that size are routine (acceptance ~0.66/0.44/0.35/0); the engine dies only when the
drop crosses an 816-token GDN block boundary (~4/816 per such event) -> intermittent, more likely under many prefills
(JSON soak with churn = a prefill step every other step). The TC MTP sodraft fix is NOT involved: it only fills
scheduler_output.scheduled_spec_decode_tokens (padded back to the scheduled count) and never touches
request.spec_token_ids, num_computed_tokens or num_output_placeholders.

Why returning no new blocks is the CORRECT layout (not just silencing the assert): the align layout after the
allocation for T_prev (cdiv(T_prev) = m + 1) is
  [.. , m-1: previous running state (queued in last_state_block_idx, freed only once processed tokens > m*816),
        m: running state, m+1 .. m+B: speculative blocks]
The worker (vllm/v1/worker/gpu/mamba_align.py) writes the new state to column cdiv(T)-1 = m-1 and the draft states to
m-1+j (j <= S <= B), copying from its last column (m + bias). All those columns are real, non-null blocks owned by
the request: m-1 cannot have been freed (processed <= T-1 < m*816) nor prefix-cached (only committed full blocks are
cached). So the high-water list already covers the smaller T, exactly like the base manager's "rejected drafts ->
num_required_blocks may be smaller than num_req_blocks" rule. When T grows again, cdiv(T)+B == len(req_blocks) and the
stock code returns [] (no new allocation, no second last_state record), the worker crosses m-1 -> m as usual.
The scheduler side (get_num_blocks_to_allocate) already returns 0 for this case (num_new_blocks <= 0), so the
admission count and the allocation now agree. Real inconsistencies are still fatal: a drop of more than one block, or
any required column being a null block, raises with the full numbers (never silently compute on a wrong state).

Install: imported by tc_solo_sched.py / tc_so_nodraft_sched.py (EngineCore process). TC_KVFIX=0 disables (A/B only).
Stats via status(): shrinks (T went down, any size), crossings (the old crash case, handled).
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("vllm.tc_kvfix")
_ON = os.environ.get("TC_KVFIX", "1") == "1"
_stats = {"installed": False, "calls": 0, "shrinks": 0, "max_shrink": 0, "crossings": 0, "error": ""}


def status():
    return dict(_stats)


def install() -> bool:
    if not _ON:
        return False
    from vllm.utils.math_utils import cdiv  # noqa: F401  (import check: same helper the manager uses)
    from vllm.v1.core import single_type_kv_cache_manager as M

    MM = M.MambaManager
    if getattr(MM, "_tc_kvfix", False):
        return True
    orig_alloc = MM.allocate_new_blocks
    orig_free = MM.free

    def allocate_new_blocks(self, request_id, num_tokens, num_tokens_main_model):
        if self.mamba_cache_mode != "align":
            return orig_alloc(self, request_id, num_tokens, num_tokens_main_model)
        _stats["calls"] += 1
        hw = getattr(self, "_tc_hw", None)
        if hw is None:
            hw = self._tc_hw = {}
        prev_t = hw.get(request_id)
        t = num_tokens_main_model
        if prev_t is not None and t < prev_t:
            _stats["shrinks"] += 1
            if prev_t - t > _stats["max_shrink"]:
                _stats["max_shrink"] = prev_t - t
            n = _stats["shrinks"]
            if n in (1, 10, 100) or n % 1000 == 0:
                log.warning("TC-KVFIX: GDN align T shrank (rejected drafts of a held request) #%d "
                            "max=%d crossings=%d", n, _stats["max_shrink"], _stats["crossings"])
        hw[request_id] = t if prev_t is None else max(prev_t, t)
        req_blocks = self.req_to_blocks[request_id]
        bs = self.block_size
        nspec = self.num_speculative_blocks
        need = -(-t // bs) + nspec
        have = len(req_blocks)
        if need >= have:
            return orig_alloc(self, request_id, num_tokens, num_tokens_main_model)
        # T went back across a GDN block boundary (rejected drafts of a held request): keep the high-water layout.
        first = need - nspec - 1   # running-state column the worker will write = cdiv(T) - 1
        bad = [i for i in range(max(first, 0), need) if req_blocks[i].is_null]
        if have - need > 1 or bad or request_id not in self._allocated_block_reqs:
            msg = (f"tc_kvfix: inconsistent mamba align layout req={request_id} T={t} prevT={prev_t} bs={bs} "
                   f"need={need} have={have} null_cols={bad} allocated={request_id in self._allocated_block_reqs}")
            _stats["error"] = msg
            raise AssertionError(msg)
        _stats["crossings"] += 1
        if _stats["crossings"] <= 20 or _stats["crossings"] % 100 == 0:
            log.warning("TC-KVFIX: GDN align T shrank across a block boundary (req=%s T %s -> %d, blocks need %d have %d)"
                        " -> keep high-water layout (#%d, shrinks=%d max=%d)", request_id, prev_t, t, need, have,
                        _stats["crossings"], _stats["shrinks"], _stats["max_shrink"])
        return []

    def free(self, request_id):
        hw = getattr(self, "_tc_hw", None)
        if hw is not None:
            hw.pop(request_id, None)
        return orig_free(self, request_id)

    MM.allocate_new_blocks = allocate_new_blocks
    MM.free = free
    MM._tc_kvfix = True
    _stats["installed"] = True
    log.warning("TC-KVFIX installed: MambaManager align allocate keeps the high-water layout when T shrinks")
    return True
