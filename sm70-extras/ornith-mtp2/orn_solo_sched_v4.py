"""ORNITH solo scheduler v4 (2026-10-24) = v3 below + two fixes for the faults of the first MTP go-live (24 Oct):
  F1 orn_mtp_kvfix2 instead of orn_mtp_kvfix: the RAM-tier eagle cap is applied INSIDE the connector lookup, so the GDN
     snapshot at the capped end is verified (v1 cut the hit after the lookup -> prepare_load asked for a snapshot block
     that was never stored -> "Block ... not found in cache", EngineCore dead at 12:12 and 16:42).
  F2 R9: structured-output requests (response_format json_schema / regex / choice) never draft, exactly like priority-client's
     R1/R5. Under async scheduling the draft is a -1 placeholder when the grammar masks are built, so vLLM fills the
     bonus position with the mask of the position BEFORE the draft; whenever the draft is accepted the bonus token can
     break the grammar ("grammar rejected tokens", request killed: 329 x 500 on 24 Oct). With no draft row the single
     sampled token gets the right mask. Unlike priority-client they still share decode steps with JARVIS rows (no solo hold).
---- v3 ----
ORNITH solo scheduler v3 (2026-10-23 PROPOSAL, mtp2): v2 (pkg/orn_solo_sched_v2.py, imported unchanged) + MTP-aware
priority-client lane. Goal: with MTP on, priority-client's priority lane (request.priority < 0, set by the :8006 router) must compute
EXACTLY what it computes today: one token per step, never a draft row, never sharing a step with draft/verify rows.

Rules added on top of v2 (v2's decode/prefill alternation, priority choice and deadlock fixes are unchanged):
  R1 no drafts for priority-client: after every schedule, a priority-client-lane request's spec_token_ids (the async scheduler's draft
     placeholders for the NEXT step) are cleared, so the scheduler never schedules a draft for it. The worker's drafter
     may still propose one on the GPU; nothing consumes it (scheduled_spec_decode_tokens has no entry for priority-client).
  R2 priority-client decodes alone: on a decode step where any priority-client-lane request is writing, every other decoder is held for
     that step (same mechanism v2 uses for prefills: next_decode_eligible_step pushed past the step). The step is a
     pure q=1 batch of priority-client rows -> the plain (non-spec) decode kernels, exactly today's path (batch-invariant
     kernels: priority-client alone == priority-client in a batch). JARVIS requests are never forced through a spec->plain->spec state
     transition (their GDN state slots stay on the spec contract), they just pause for priority-client's short reply.
  R3 optional starvation cap (OFF by default since v3b, owner rule "priority-client never waits"): ORN_V3_PRIOCLIENT_SOLO_MAX=N>0 ->
     after N consecutive priority-client-only decode steps with others waiting, every 2nd decode step is others-only.
  R6 (v3c) importing this module installs orn_mtp_kvfix (RAM KV tier capped at the MTP eagle-dropped page = the
     root cause of priority-client's run-to-run fingerprints) and v2's stock safety net is suppressed while priority-client can decode.
  R7 (v3d, ORN_V3_PRIOCLIENT_NOCACHE=1) priority-client-lane requests never READ the prefix cache (GPU or RAM tier) under MTP.
  R8 (v3e, ORN_V3_PRIOCLIENT_PRODCHUNK=1) priority-client's prompt is chunked exactly like prod: under MTP the scheduler stops prefill
     chunks at the EAGLE replay boundary (round_down(L-1,1056)-1056) and shifts last_cache_position one page earlier,
     e.g. a 4,392-token screenshot prompt runs as 3,168 + 1,224 instead of prod's 4,224 + 167 + 1 -> different MoE
     (PF2/PF1 by call size), prefix-attention and GDN chunk calls -> different bits (6/7 every run in qt-sr-mtp2c-nokv).
     For priority<0 the non-eagle replay boundaries and last_cache_position are used (she never reads the cache: R7).
  R5 (v3b) priority-client-lane requests are tagged sampling_params.extra_args["orn_nodraft"]=True at arrival, so the worker
     port mtp2/ports/orn_mtp_skip.py can skip the MTP drafter for her prefill chunks and decode steps.
  R4 if v2's stock safety-net step runs (3 empty steps), priority-client's row may share a step with verify rows (no draft for
     priority-client itself, R1 still holds): counted in stats['priority-client_mixed'] and logged (expected 0).
Without MTP (num_spec_tokens == 0) R1 is a no-op and R2 only changes batching of priority-client decode steps (still exact).
Off switch: ORN_V3_PRIOCLIENT_SOLO=0 keeps R1 only (priority-client rides along in q=1-for-priority-client mixed steps -> NOT recommended).
Use: PYTHONPATH += /opt/1cat/ornith122-trial/mtp2 ; --scheduler-cls orn_solo_sched_v3.OrnSoloStepScheduler
"""
from __future__ import annotations

import logging
import os

from orn_solo_sched_v2 import OrnSoloStepScheduler as _V2
from orn_solo_sched_v2 import _key, _prefill_remaining

log = logging.getLogger("orn_solo_sched_v3")

# v3c: engine-core-side fix for priority-client's run-to-run bits under MTP (RAM-tier serving the eagle-dropped page), see
# orn_mtp_kvfix.py. Installed at import = before the scheduler builds its KV connector.
try:
    import orn_mtp_kvfix2 as _kvfix   # v4: F1 fix (cap inside the lookup)

    _KVFIX = _kvfix.install()
except Exception as _exc:  # never block the scheduler
    _KVFIX = f"failed {_exc!r}"


def _priority-client(r) -> bool:
    return r.priority < 0


def _structured(r) -> bool:
    return getattr(r, "structured_output_request", None) is not None


_SO_NODRAFT = os.environ.get("ORN_V4_SO_NODRAFT", "1") == "1"   # 0 = v3 behaviour (only for the before/after test)


def _nodraft(r) -> bool:   # v4 R9: priority-client's lane or a structured-output request
    return r.priority < 0 or (_SO_NODRAFT and _structured(r))


class OrnSoloStepScheduler(_V2):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._v3_solo = os.environ.get("ORN_V3_PRIOCLIENT_SOLO", "1") == "1"
        self._v3_max = max(0, int(os.environ.get("ORN_V3_PRIOCLIENT_SOLO_MAX", "0") or 0))
        self._v3_run = 0          # consecutive priority-client-only decode steps with others waiting
        self._v3_flip = False
        self._solo_stats.update({"priority-client_solo": 0, "others_only": 0, "priority-client_mixed": 0, "priority-client_drafts_cleared": 0,
                                 "so_tagged": 0, "so_drafts_cleared": 0})
        self._v3_prodchunk = bool(self.num_spec_tokens) and os.environ.get("ORN_V3_PRIOCLIENT_PRODCHUNK", "0") == "1"
        if self._v3_prodchunk:
            try:
                coord = self.kv_cache_manager.coordinator
                orig_rb = coord.get_replay_boundaries

                def get_replay_boundaries(request, alignment, _orig=orig_rb):
                    if _priority-client(request):  # prod (no-spec) replay boundaries for priority-client: same chunks, same retention
                        return (request.num_tokens - 1, request.num_tokens)
                    return _orig(request, alignment)

                coord.get_replay_boundaries = get_replay_boundaries
            except Exception as exc:
                self._v3_prodchunk = False
                log.warning("ORN-SOLO v3e: prod-chunk patch failed %r -> off", exc)
        log.warning("ORN-SOLO v4 scheduler active (v3 + kvfix2=%s + structured outputs never drafted=%s): v2 + priority-client lane never drafted, priority-client decodes alone (spec=%d, "
                    "solo=%s, cap=%d, v3b tag, v3c kvfix=%s, v3e prodchunk=%s)", _KVFIX, _SO_NODRAFT, self.num_spec_tokens, self._v3_solo,
                    self._v3_max, _KVFIX, self._v3_prodchunk)

    # R5 ----------------------------------------------------------------------------------------------------------------
    def add_request(self, request) -> None:
        try:
            sp = getattr(request, "sampling_params", None)
            if _nodraft(request) and sp is not None:
                ea = dict(sp.extra_args or {})
                ea["orn_nodraft"] = True
                sp.extra_args = ea
                k = "priority-client_tagged" if _priority-client(request) else "so_tagged"
                self._solo_stats[k] = self._solo_stats.get(k, 0) + 1
            if _priority-client(request) and self.num_spec_tokens and os.environ.get("ORN_V3_PRIOCLIENT_NOCACHE", "0") == "1":
                # v3d: under MTP priority-client's warm prefix-cache hits are not reproducible (cold runs are: FP e89bb76d on
                # two boots); she always prefills cold. Her blocks are still WRITTEN to the cache (JARVIS unaffected).
                request.skip_reading_prefix_cache = True
                self._solo_stats["priority-client_nocache"] = self._solo_stats.get("priority-client_nocache", 0) + 1
        except Exception as exc:
            log.warning("ORN-SOLO v3: priority-client tag failed %r", exc)
        super().add_request(request)

    # R8 (v3e) -----------------------------------------------------------------------------------------------------------
    def _mamba_block_aligned_split(self, request, *args, **kwargs):
        if not (self._v3_prodchunk and _priority-client(request)):
            return super()._mamba_block_aligned_split(request, *args, **kwargs)
        saved = self.use_eagle
        self.use_eagle = False   # prod's last_cache_position (no eagle page shift) for priority-client's prompt chunking
        try:
            n = super()._mamba_block_aligned_split(request, *args, **kwargs)
        finally:
            self.use_eagle = saved
        self._solo_stats["priority-client_prodchunk"] = self._solo_stats.get("priority-client_prodchunk", 0) + 1
        return n

    # R1 ------------------------------------------------------------------------------------------------------------
    def _update_after_schedule(self, scheduler_output) -> None:
        super()._update_after_schedule(scheduler_output)
        if not self.num_spec_tokens:
            return
        for req_id in scheduler_output.num_scheduled_tokens:
            r = self.requests.get(req_id)
            if r is not None and _nodraft(r) and r.spec_token_ids:
                r.spec_token_ids = []
                self._solo_stats["priority-client_drafts_cleared" if _priority-client(r) else "so_drafts_cleared"] += 1

    # R4 accounting on the stock safety-net step --------------------------------------------------------------------
    def schedule(self):
        if self.num_spec_tokens:  # R1 belt and braces (e.g. placeholders set before v3 saw the request)
            for r in self.running:
                if _nodraft(r) and r.spec_token_ids:
                    r.spec_token_ids = []
                    self._solo_stats["priority-client_drafts_cleared" if _priority-client(r) else "so_drafts_cleared"] += 1
        # v3c R4 fix: never let v2's stock safety-net step run while a priority-client row can decode (it could mix her with
        # verify rows); a priority-client-only decode step is always schedulable, so the constrained step is non-empty.
        if self._solo_empty >= 3 and any(_priority-client(r) and _prefill_remaining(r) == 0 for r in self.running):
            self._solo_empty = 0
            self._solo_stats["priority-client_netskip"] = self._solo_stats.get("priority-client_netskip", 0) + 1
        out = super().schedule()
        try:
            if self.num_spec_tokens and out.scheduled_spec_decode_tokens:
                ids = out.num_scheduled_tokens
                if any(_priority-client(self.requests[i]) for i in ids if i in self.requests):
                    self._solo_stats["priority-client_mixed"] += 1
                    if self._solo_stats["priority-client_mixed"] in (1, 10, 100, 1000):
                        log.warning("ORN-SOLO v3: priority-client row shared a step with verify rows (#%d)",
                                    self._solo_stats["priority-client_mixed"])
        except Exception:
            pass
        return out

    # R2 / R3: v2's _solo_step with the priority-client-solo hold for decode steps ---------------------------------------------
    def _solo_step(self, kind, step, prefills, head):
        saved = {}
        orig_max = self.max_num_running_reqs

        def hold(r):
            if r.request_id not in saved:
                saved[r.request_id] = (r, r.next_decode_eligible_step)
            r.next_decode_eligible_step = max(r.next_decode_eligible_step, step + 1)

        sub = None
        try:
            if kind == "decode":
                for r in prefills:
                    hold(r)
                self.max_num_running_reqs = min(orig_max, len(self.running))  # admit nobody new
                if self._v3_solo:
                    dec = [r for r in self.running if _prefill_remaining(r) == 0
                           and step >= r.next_decode_eligible_step]
                    bk = [r for r in dec if _priority-client(r)]
                    others = [r for r in dec if not _priority-client(r)]
                    if bk and others:
                        capped = self._v3_max > 0 and self._v3_run >= self._v3_max
                        if capped:
                            self._v3_flip = not self._v3_flip
                        if capped and self._v3_flip:
                            sub = "others_only"
                            for r in bk:
                                hold(r)
                        else:
                            sub = "priority-client_solo"
                            self._v3_run += 1
                            for r in others:
                                hold(r)
                    elif bk:
                        sub = "priority-client_solo"
                        self._v3_run = 0
                    else:
                        self._v3_run = 0
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
            out = super(_V2, self).schedule()   # the stock async scheduler (exactly one call per engine step)
        finally:
            self.max_num_running_reqs = orig_max
            for r, v in saved.values():
                r.next_decode_eligible_step = v
        if out.num_scheduled_tokens:
            self._solo_last = kind
            self._solo_stats[kind] += 1
            if sub:
                self._solo_stats[sub] += 1
        return out
