"""ORNITH batch-invariance scheduler (2026-10-22, main): every engine step is EITHER a decode-only step (all requests
that are writing, one token each) OR a prefill step for exactly ONE request. Nothing is ever mixed.

Why: on this engine the maths for a request depends on what shares its step. When a prefill shares a step with decodes,
the decode rows take prefill-type kernels (GDN chunk instead of recurrent, MoE path by call size, attention path by
max_query_len) and two prefills in one step change chunk sizes / MoE call sizes / all-reduce message layout. With
"no mixing", a request's prefill chunks depend only on that request, and decode-only steps use kernels whose per-row
results were measured batch-invariant (F1 MoE, router, QPN8 dense, 1-stage all-reduce; attention needs the decode
partition size pinned: VLLM_FLASH_V100_DECODE_PARTITION_SIZE).

How: before the stock (async) scheduler runs, requests that must not run this step get next_decode_eligible_step pushed
past the current step (the stock running loop skips them), and the waiting loop is capped through max_num_running_reqs.
Everything is restored right after. When both kinds are pending, steps alternate decode / prefill, so decodes are never
starved and prompt reading keeps ~half the steps (decode steps are ~10-17 ms against ~1.7 s prefill chunks).
Priority: while a request with a better priority than every pending prefill is writing, steps stay decode-only\n(priority-client's short replies finish at full speed). Prefill choice: the best (priority, arrival) among running partial prefills and the head of the waiting queue, so
priority-client's priority lane still goes first. Off switch: drop --scheduler-cls (env ORN_SOLO_SCHED=0 in run-ornith-test.sh).
"""
from __future__ import annotations

import logging

from vllm.v1.core.sched.async_scheduler import AsyncScheduler

log = logging.getLogger("orn_solo_sched")


def _prefill_remaining(r) -> int:
    # same definition as the stock scheduler (replay after preemption counts as prefill too)
    return max(0, max(r.num_prompt_tokens, r.num_tokens - 1) - r.num_computed_tokens)


def _key(r):
    return (r.priority, r.arrival_time)


class OrnSoloStepScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._solo_last = "prefill"
        self._solo_stats = {"decode": 0, "prefill": 0, "idle": 0}
        log.warning("ORN-SOLO scheduler active: decode-only or single-prefill steps (no mixing)")

    def _waiting_head(self):
        for q in (self.waiting, self.skipped_waiting):
            if q:
                try:
                    return q.peek_request()
                except Exception:
                    continue
        return None

    def schedule(self):
        step = self.current_step + 1  # the stock schedule() increments first
        prefills = [r for r in self.running if _prefill_remaining(r) > 0]
        decoders = [r for r in self.running if _prefill_remaining(r) == 0 and step >= r.next_decode_eligible_step]
        head = self._waiting_head()
        want_prefill = bool(prefills) or head is not None

        if decoders and want_prefill:
            cands = list(prefills) + ([head] if head is not None else [])
            if min(r.priority for r in decoders) < min(r.priority for r in cands):
                kind = "decode"   # a higher-priority request (priority-client's lane) is writing: let it finish first
            else:
                kind = "prefill" if self._solo_last == "decode" else "decode"
        elif decoders:
            kind = "decode"
        elif want_prefill:
            kind = "prefill"
        else:
            self._solo_stats["idle"] += 1
            return super().schedule()

        saved = {}
        orig_max = self.max_num_running_reqs

        def hold(r):
            if r.request_id not in saved:
                saved[r.request_id] = (r, r.next_decode_eligible_step)
            r.next_decode_eligible_step = max(r.next_decode_eligible_step, step + 1)

        try:
            if kind == "decode":
                for r in prefills:
                    hold(r)
                self.max_num_running_reqs = min(orig_max, len(self.running))  # admit nobody new
            else:
                cands = list(prefills) + ([head] if head is not None else [])
                pick = min(cands, key=_key)
                for r in self.running:
                    if r is not pick:
                        hold(r)
                if pick in prefills:
                    self.max_num_running_reqs = min(orig_max, len(self.running))
                else:  # admit exactly one waiting request (the queue head)
                    self.max_num_running_reqs = min(orig_max, len(self.running) + 1)
            out = super().schedule()
        finally:
            self.max_num_running_reqs = orig_max
            for r, v in saved.values():
                r.next_decode_eligible_step = v
        if out.num_scheduled_tokens:
            self._solo_last = kind
            self._solo_stats[kind] += 1
        return out
