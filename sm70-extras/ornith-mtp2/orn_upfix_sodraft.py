"""Upstream backport (2026-10-25, owner "do the small fix"): fill structured-output drafts before the grammar bitmask.

Upstream 1CatAI/1Cat-vLLM a411e330f + ee71284bc + 00d8dfaef (EngineCore._fill_structured_output_drafts). Under async
scheduling the spec-token rows of a scheduled structured-output request are -1 placeholders; they are only replaced by
the worker's real drafts on the DEFERRED path (a step of the same request still in flight). Our no-mixing solo
scheduler often schedules a request with no step in flight -> the bitmask was built from placeholders while the worker
verified real drafts -> grammar-invalid bonus token -> xgrammar "Failed to advance FSM" -> HTTP 500 (10-24, 329 errors).
v4 works around it by never drafting structured requests (ORN_V4_SO_NODRAFT=1); this fix lets them draft again.

Engine-core process only. No vLLM file is edited: OrnUpfixScheduler (mixed into the scheduler class by
orn_solo_sched_v4cp1) calls fill() at the top of get_grammar_bitmask, i.e. exactly where upstream calls it on the
non-deferred path. On the deferred path EngineCore has already called update_draft_token_ids_in_output for that
SchedulerOutput; we mark those outputs and skip them, so the deferred path is unchanged. Nothing here is traced by
torch.compile (engine-core process, no model code), so compile caches are unaffected.
"""
from __future__ import annotations

import inspect
import logging
import os

log = logging.getLogger("vllm.orn_upfix_sodraft")

ENABLED = os.environ.get("ORN_UPFIX_SODRAFT", "1") == "1"
STATS = {"fill_calls": 0, "filled": 0, "no_drafts": 0, "skipped_deferred": 0}


def needs_drafts(scheduler, scheduler_output) -> bool:
    """Upstream condition, verbatim logic."""
    if not (scheduler_output.has_structured_output_requests and scheduler_output.scheduled_spec_decode_tokens):
        return False
    requests = getattr(scheduler, "requests", None)
    if requests is None:   # 00d8dfaef: custom schedulers without a request registry keep their path
        return False
    for req_id, spec in scheduler_output.scheduled_spec_decode_tokens.items():
        request = requests.get(req_id)
        if (request is not None and request.use_structured_output
                and request.num_output_placeholders <= len(spec) + 1 and any(t == -1 for t in spec)):
            return True
    return False


def find_engine_core():
    """The EngineCore whose __init__ is constructing the scheduler (walk the caller frames)."""
    from vllm.v1.engine.core import EngineCore
    f = inspect.currentframe()
    try:
        while f is not None:
            s = f.f_locals.get("self")
            if isinstance(s, EngineCore):
                return s
            f = f.f_back
    finally:
        del f
    return None


class OrnUpfixScheduler:
    """Mixin: put it BEFORE the real scheduler class in the bases."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._orn_core = find_engine_core() if ENABLED else None
        self._orn_draft_filled = set()
        log.warning("ORN upfix sodraft (upstream a411e330f): enabled=%s core=%s", ENABLED,
                    type(self._orn_core).__name__ if self._orn_core is not None else None)

    def update_draft_token_ids_in_output(self, draft_token_ids, scheduler_output):
        self._orn_draft_filled.add(id(scheduler_output))
        return super().update_draft_token_ids_in_output(draft_token_ids, scheduler_output)

    def get_grammar_bitmask(self, scheduler_output):
        core = self._orn_core
        if core is not None and core.use_spec_decode:
            if id(scheduler_output) in self._orn_draft_filled:
                STATS["skipped_deferred"] += 1
            else:
                STATS["fill_calls"] += 1
                if needs_drafts(self, scheduler_output):
                    d = core.model_executor.take_draft_token_ids()
                    if d is None:
                        STATS["no_drafts"] += 1
                    else:
                        self.update_draft_token_ids_in_output(d, scheduler_output)
                        STATS["filled"] += 1
        self._orn_draft_filled.discard(id(scheduler_output))
        return super().get_grammar_bitmask(scheduler_output)
