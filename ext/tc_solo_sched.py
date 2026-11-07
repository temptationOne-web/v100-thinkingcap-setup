"""ThinkingCap scheduler (2026-10-27 00:xx, owner "fix it now"): Ornith's orn_solo_sched_v4 with R2 (priority-client decodes
alone -> exact) and R7 (priority-client never reads prefix cache) KEPT, but R1/R5 (priority-client never drafts) and R9 (structured outputs
never draft) REMOVED. On ThinkingCap the no-draft path is the plain q=1 decode = ~12 tok/s (normal MTP lane ~99 tok/s).
The new engine carries upstream's structured-output draft-fill fix (JSON gate passed 10-26 with drafting on)."""
import orn_solo_sched_v4 as _m

import tc_kvfix as _kvfix  # 10-27: GDN align high-water layout (fixes "num_required_blocks N < len(req_blocks) N+1")

try:
    _KVFIX_OK = _kvfix.install()
except Exception as _exc:  # noqa: BLE001
    _KVFIX_OK = f"failed {_exc!r}"

_m._nodraft = lambda r: False   # R1/R5/R9 off: every request may draft


class TCSoloStepScheduler(_m.OrnSoloStepScheduler):
    # 11-06 priority-client sparse split (TC_PRIOCLIENT_SPARSE=1): engine 20261106 splits a prefill at every Mamba state block while
    # other requests run (mamba_dense_boundaries_on_contention), and at sparse replay boundaries when alone. priority-client's
    # prompt was therefore read in different chunks idle vs busy -> different bits (fp 150e idle / 0a7d busy). priority-client
    # (priority < 0) always takes the lone-request split; with R7 (ORN_V3_PRIOCLIENT_NOCACHE=1) she never reads the cache
    # either, so her replies are identical under any load. Other requests are unchanged.
    # RESULT 11-06 14:06: NO-GO, left OFF. Even with R7 + this, priority-client fps still varied (idle 150e/150e/0404, busy 0a7d x3,
    # stress 3 fps; answers always 7/7+22/22) -> the variance is not the cache or the split (suspect: priority-client drafts).
    def _mamba_block_aligned_split(self, request, *args, **kwargs):
        if not (_PRIOCLIENT_SPARSE and request.priority < 0 and getattr(self, "mamba_dense_boundaries_on_contention", False)):
            return super()._mamba_block_aligned_split(request, *args, **kwargs)
        self.mamba_dense_boundaries_on_contention = False
        try:
            return super()._mamba_block_aligned_split(request, *args, **kwargs)
        finally:
            self.mamba_dense_boundaries_on_contention = True


import os as _os

_PRIOCLIENT_SPARSE = _os.environ.get("TC_PRIOCLIENT_SPARSE", "0") == "1"


# 11-05 short-first (TC_SHORTFIRST=1): among prefills of the SAME priority, a prompt with <= TC_SHORTFIRST_TOKENS left
# to read goes before a longer one (before: oldest first, so "hi" waited ~15 s behind a 60k read). Only the ORDER of
# single-request prefill steps changes; no step mixes requests, so every request's maths is unchanged. priority-client
# (priority < 0) still always goes first.
import os as _os

if _os.environ.get("TC_SHORTFIRST", "0") == "1":
    _SF_TOK = int(_os.environ.get("TC_SHORTFIRST_TOKENS", "8192"))

    def _sf_key(r):
        return (r.priority, _m._prefill_remaining(r) > _SF_TOK, r.arrival_time)

    _m._key = _sf_key
