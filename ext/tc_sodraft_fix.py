"""ThinkingCap structured-output draft fix (2026-10-27, JOB 1: JSON HTTP 500s with MTP on).

Root cause (V2 model runner + async scheduling + our solo scheduler):
  The V2 worker keeps every request's MTP drafts in req_states.draft_tokens (GPU, per request) and verifies THOSE on
  the request's next scheduled step, even when that step is not the very next engine step. For grammar requests the
  scheduler must build the bitmask over the same drafts (EngineCore._fill_structured_output_drafts /
  update_draft_token_ids_in_output -> grammar.validate_tokens). It gets them from take_draft_token_ids(), but stock
  DraftTokensHandler only returns the drafts of the LAST drafting step's batch (and all -1 if that step had no grammar
  row). The solo scheduler often holds a JSON row for a step (prefill step, priority-client-solo step), so the row is missing,
  its spec slots stay -1, xgrammar treats every row after the first -1 as unconstrained (and cannot see a </think>
  inside the window), while the GPU accepts its real drafts -> grammar-invalid tokens -> "Failed to advance FSM" ->
  "grammar rejected tokens" -> HTTP 500.

Fix (worker process, no engine file edited): DraftTokensHandler keeps a small per-request cache of the drafts it last
copied back (full proposal width = exactly what req_states.draft_tokens holds and the verifier reads), and
get_draft_tokens() returns the latest row of EVERY cached request. The scheduler then validates the drafts the worker
will really verify, so the bitmask always matches. Only steps that already copied drafts (batch has a grammar row)
do any work; non-JSON traffic is untouched. A request's row only changes in steps that contain it, and such a step
has a grammar row, so the cache is always current for grammar requests.
Env: TC_SODRAFT_FIX=0 disables (A/B only). Stats via status().
"""
from __future__ import annotations

import collections
import os

_ON = os.environ.get("TC_SODRAFT_FIX", "1") == "1"
_CAP = 256
_stats = {"installed": False, "copies": 0, "gets": 0, "rows_returned": 0, "extra_rows": 0, "error": ""}


def status():
    return dict(_stats)


def install() -> bool:
    if not _ON:
        return False
    import torch
    from vllm.v1.outputs import DraftTokenIds
    from vllm.v1.worker.gpu.async_utils import async_copy_to_np
    from vllm.v1.worker.gpu.spec_decode import utils as U

    H = U.DraftTokensHandler
    if getattr(H, "_tc_sodraft_fix", False):
        return True

    def _flush(self):
        # Materialise the pending device->host copy into the per-request cache.
        pend = getattr(self, "_tc_pending", None)
        if pend is None:
            return
        self._tc_pending = None
        req_ids, arr, ev = pend
        ev.synchronize()
        cache = self._tc_cache
        rows = arr.tolist()
        for rid, row in zip(req_ids, rows):
            if rid is None:
                continue
            cache[rid] = row
            cache.move_to_end(rid)
        while len(cache) > _CAP:
            cache.popitem(last=False)

    def set_draft_tokens(self, input_batch, draft_tokens, num_draft_tokens=None):
        if not hasattr(self, "_tc_cache"):
            self._tc_cache = collections.OrderedDict()
            self._tc_pending = None
        self.req_ids = input_batch.req_ids
        if num_draft_tokens is None:
            num_draft_tokens = draft_tokens.shape[1]
        if not 0 <= num_draft_tokens <= draft_tokens.shape[1]:
            raise ValueError("num_draft_tokens must be within the proposal tensor width")
        self.num_draft_tokens = num_draft_tokens
        if not input_batch.has_structured_output_reqs:
            self.draft_tokens_np = None
            return
        _flush(self)   # previous step's copy (long complete) -> cache
        # Full width: the verifier reads req_states.draft_tokens[:, :n_scheduled], n_scheduled may exceed
        # num_draft_tokens (scheduler picks the placeholder count, the speculator picks its own depth).
        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.copy_stream):
            arr = async_copy_to_np(draft_tokens)
            self.copy_event.record()
        self.draft_tokens_np = arr
        self._tc_pending = (list(self.req_ids), arr, self.copy_event)
        _stats["copies"] += 1

    def get_draft_tokens(self):
        if not hasattr(self, "_tc_cache"):
            self._tc_cache = collections.OrderedDict()
            self._tc_pending = None
        _flush(self)
        _stats["gets"] += 1
        if not self._tc_cache:
            # Stock fallback (no grammar step seen yet): placeholders for the last batch.
            return DraftTokenIds(self.req_ids, [[-1] * self.num_draft_tokens for _ in self.req_ids])
        last = set(r for r in self.req_ids if r is not None)
        ids = list(self._tc_cache.keys())
        _stats["rows_returned"] += len(ids)
        _stats["extra_rows"] += sum(1 for r in ids if r not in last)
        return DraftTokenIds(ids, [list(self._tc_cache[r]) for r in ids])

    H.set_draft_tokens = set_draft_tokens
    H.get_draft_tokens = get_draft_tokens
    H._tc_sodraft_fix = True
    _stats["installed"] = True
    return True
