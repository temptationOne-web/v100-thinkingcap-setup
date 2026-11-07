"""ThinkingCap R9 port (2026-10-26, PREPARED, OFF by default): structured-output requests never draft.

Ornith's scheduler v4 (/opt/1cat/ornith122-trial/mtp2/orn_solo_sched_v4.py, ORN_V4_SO_NODRAFT=1) cleared the MTP draft
placeholders of every request with response_format json_schema / regex / choice: under async scheduling the draft is a
-1 placeholder when the grammar bitmask is built, so an accepted draft could let the bonus token break the grammar
("grammar rejected tokens" -> HTTP 500; 329 of them on Ornith's first MTP go-live, 24 Oct). Upstream's structured-draft
fill (a411e330f / ee71284bc / 00d8dfaef) IS in the new engine vllm-main-20261025 (pending_structured_output_tokens in
async_scheduler.py), but on Ornith that fill alone still failed (97 grammar 500s, arm B 10-25). So the JSON gate decides:
  - JSON gate clean with this OFF  -> leave it off (structured requests keep MTP speed).
  - any "grammar rejected" / 500   -> turn it on: TC_SO_NODRAFT=1 for run-tc-prod.sh (adds
    --scheduler-cls tc_so_nodraft_sched.TCSONoDraftScheduler) and re-run the JSON gate.

Only R9 is ported (no priority-client solo lane, no Ornith solo-step rules). Subclass of the engine's AsyncScheduler (prod runs
--async-scheduling); the scheduler class is not part of the compile hash. Runs in the EngineCore process (PYTHONPATH
from run-tc-new.sh). TC_SO_NODRAFT_OFF=1 makes the class a pure pass-through (for an A/B without changing args).
Log line at start: "TC-SO-NODRAFT scheduler active"; counter in self.tc_so_stats."""
from __future__ import annotations

import logging
import os

from vllm.v1.core.sched.async_scheduler import AsyncScheduler

import tc_kvfix as _kvfix  # 10-27: GDN align high-water layout (fixes "num_required_blocks N < len(req_blocks) N+1")

try:
    _KVFIX_OK = _kvfix.install()
except Exception as _exc:  # noqa: BLE001
    _KVFIX_OK = f"failed {_exc!r}"

log = logging.getLogger("vllm.tc_so_nodraft")
_ON = os.environ.get("TC_SO_NODRAFT_OFF", "0") != "1"


def _structured(r) -> bool:
    return bool(getattr(r, "use_structured_output", False)) or getattr(r, "structured_output_request", None) is not None


class TCSONoDraftScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tc_so_stats = {"so_drafts_cleared": 0}
        log.warning("TC-SO-NODRAFT scheduler active (structured outputs never drafted=%s, spec=%d)", _ON,
                    self.num_spec_tokens)

    def _clear(self, r) -> None:
        if r is not None and r.spec_token_ids and _structured(r):
            r.spec_token_ids = []
            self.tc_so_stats["so_drafts_cleared"] += 1

    def _update_after_schedule(self, scheduler_output) -> None:
        super()._update_after_schedule(scheduler_output)
        if not (_ON and self.num_spec_tokens):
            return
        for req_id in scheduler_output.num_scheduled_tokens:
            self._clear(self.requests.get(req_id))

    def update_draft_token_ids(self, draft_token_ids) -> None:   # sync-path drafts (belt and braces)
        super().update_draft_token_ids(draft_token_ids)
        if not (_ON and self.num_spec_tokens):
            return
        for req_id in draft_token_ids.req_ids:
            self._clear(self.requests.get(req_id))

    def schedule(self):
        if _ON and self.num_spec_tokens:   # placeholders set before this class saw the request
            for r in self.running:
                self._clear(r)
        return super().schedule()
