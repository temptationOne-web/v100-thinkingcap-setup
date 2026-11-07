"""mtp2 ORN-MTP-KVFIX2 (2026-10-24): orn_mtp_kvfix with the cap applied INSIDE the RAM-tier lookup.

Prod fault 24 Oct 12:12 + 16:42 (sr-mtp3e, the first MTP go-live): EngineCore died with
  AssertionError: Block ... not found in cache   (vllm/v1/kv_offload/cpu/manager.py prepare_load)
ROOT CAUSE: orn_mtp_kvfix (v1) ran the connector's normal lookup and THEN cut the served tokens back to the
eagle-dropped boundary cap = round_down(num_tokens - 1, align) - align. The lookup verifies the full-attention blocks
as a prefix (anything shorter is verified too) but the Mamba/GDN groups only as a SUFFIX: the one recurrent-snapshot
block at the END of the hit. After the cut the hit ends one page earlier, where the GDN snapshot was never checked:
the RAM tier only holds snapshots at boundaries some earlier request stopped at (prefill chunk ends, decode pages),
so a RAM hit that reaches >= 2 pages past the GPU hit (GPU blocks evicted: long chats, wiki/diary builds) and is cut
to a boundary with no stored snapshot makes update_state_after_alloc ask prepare_load for a missing key -> assert.
v1's own log only ever showed the 1056 -> 0 case (cut to nothing = no load, harmless); the crash is the partial cut.
FIX: run the lookup itself with the cap as its upper bound, so every group (prefix AND GDN suffix) is verified at the
boundary that will really be loaded; a missing snapshot there makes the suffix scan fall back to an earlier stored one
(or a miss), exactly as the stock connector does without MTP. Done by handing _lookup a view of the request whose
num_tokens is one align lower: _lookup's own bound round_down(n - 1, align) then equals cap exactly.
Everything else is v1: priority-client (skip_reading_prefix_cache) never reads the RAM tier; no speculative config -> no cap.
ORN_KVFIX_MODE=v1diag reproduces v1 (post-hoc cut) and only COUNTS/logs cuts whose GDN end block is absent
("ORN-MTP-KVFIX2 v1diag: WOULD-CRASH") - for proving the soak workload hits the fault; never use it live.
Runs in the ENGINE-CORE (scheduler) process: imported by orn_solo_sched_v4. Never raises (stock result on error).
"""
import logging
import os

log = logging.getLogger("orn_mtp_kvfix2")
_state = {"installed": False, "mode": None, "calls": 0, "capped": 0, "zeroed": 0, "would_crash": 0, "errors": 0}


def _cap_tokens(num_tokens: int, align: int) -> int:
    top = (max(num_tokens - 1, 0) // align) * align
    return max(top - align, 0)


class _ReqView:
    """The request as _lookup sees it, with num_tokens lowered so its bound lands on the cap."""

    __slots__ = ("_r", "_n")

    def __init__(self, req, num_tokens):
        object.__setattr__(self, "_r", req)
        object.__setattr__(self, "_n", num_tokens)

    @property
    def num_tokens(self):
        return self._n

    def __getattr__(self, name):
        return getattr(self._r, name)


class _StatusView:
    """req_status with .req replaced; every other read AND write goes to the real object."""

    __slots__ = ("_s", "_req")

    def __init__(self, status, req_view):
        object.__setattr__(self, "_s", status)
        object.__setattr__(self, "_req", req_view)

    @property
    def req(self):
        return self._req

    def __getattr__(self, name):
        return getattr(self._s, name)

    def __setattr__(self, name, value):
        setattr(self._s, name, value)


def _gdn_end_missing(self, request, served):
    """v1diag: True if a sliding/GDN group's block that ends at `served` is not in the RAM tier (read-only check)."""
    st = self._req_status[request.request_id]
    for gc in self.config.kv_group_configs:
        if gc.sliding_window_size_in_blocks is None:
            continue
        keys = st.group_states[gc.group_idx].offload_keys
        idx = -(-served // gc.offloaded_block_size) - 1
        if 0 <= idx < len(keys):
            b = _blk(self, keys[idx])
            if b is None or not b.is_ready:
                return True
    return False


_TRACE = os.environ.get("ORN_KVFIX_TRACE", "0") == "1"


def _blk(self, key):
    """Read-only RAM-tier block for a key (the CPU tier is a GroupedCPUOffloadingManager: one pool per KV group)."""
    m = self.manager
    if hasattr(m, "managers"):
        m = m.managers[int.from_bytes(bytes(key)[-4:], "big")]
    return m._policy.get(key)


def _check_load(self, request, blocks, num_external_tokens):
    """Recompute the keys update_state_after_alloc will hand prepare_load (same arithmetic) and report absent ones."""
    st = self._req_status[request.request_id]
    ncached = st.num_locally_computed_tokens + num_external_tokens
    miss, info = [], []
    for gc in self.config.kv_group_configs:
        gs = st.group_states[gc.group_idx]
        gb = blocks.blocks[gc.group_idx]
        ngpu = -(-ncached // gc.gpu_block_size)
        nlcg = ngpu
        for i, b in enumerate(gb[:ngpu]):
            if not b.is_null and b.block_hash is None:
                nlcg = i
                break
        nb = -(-ncached // gc.offloaded_block_size)
        start = nlcg // self.config.block_size_factor
        present = "".join("1" if _blk(self, k) is not None else "0" for k in gs.offload_keys[max(start - 3, 0):nb])
        info.append((gc.group_idx, gc.sliding_window_size_in_blocks, gc.gpu_block_size, gc.offloaded_block_size, nlcg,
                     start, nb, present))
        if ngpu - nlcg:
            for j, k in enumerate(gs.offload_keys[start:nb]):
                if _blk(self, k) is None:
                    miss.append((gc.group_idx, start + j))
    return miss, info


def install() -> str:
    if _state["installed"]:
        return "already"
    if os.environ.get("ORN_MTP_KVFIX", "1") != "1":
        return "off"
    mode = os.environ.get("ORN_KVFIX_MODE", "lookup")
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading import scheduler as S
    except Exception as exc:
        log.warning("ORN-MTP-KVFIX2: offloading scheduler not importable (%r) -> not installed", exc)
        return "unavailable"
    C = S.OffloadingConnectorScheduler
    if getattr(C.get_num_new_matched_tokens, "_orn_kvfix_orig", None) is not None:
        _state["installed"] = True
        return "already (another kvfix)"
    orig_init = C.__init__
    orig_get = C.get_num_new_matched_tokens
    orig_lookup = C._lookup

    def __init__(self, spec, *a, **kw):
        orig_init(self, spec, *a, **kw)
        self._orn_cap_n = None
        self._orn_rec = {}
        try:
            sc = getattr(spec.vllm_config, "speculative_config", None)
            self._orn_kvfix_on = bool(sc is not None and getattr(self, "_mamba_align_size", None))
            log.warning("ORN-MTP-KVFIX2 installed (mode=%s): spec=%s mamba_align=%s -> RAM-tier hits bounded at the "
                        "eagle-dropped boundary inside the lookup: %s", mode, getattr(sc, "method", None),
                        getattr(self, "_mamba_align_size", None), self._orn_kvfix_on)
            if _TRACE:
                log.warning("ORN-MTP-KVFIX2 TRACE on: groups (idx, sliding_window_blocks, gpu_bs, off_bs, hash_factor) = %s, "
                            "block_size_factor=%s", [(g.group_idx, g.sliding_window_size_in_blocks, g.gpu_block_size,
                                                      g.offloaded_block_size, getattr(g, "hash_block_size_factor", None))
                                                     for g in self.config.kv_group_configs],
                            getattr(self.config, "block_size_factor", None))
        except Exception as exc:
            self._orn_kvfix_on = False
            log.warning("ORN-MTP-KVFIX2: init check failed %r -> off", exc)

    def _lookup(self, req_status):
        n = getattr(self, "_orn_cap_n", None)
        if n is None:
            return orig_lookup(self, req_status)
        return orig_lookup(self, _StatusView(req_status, _ReqView(req_status.req, n)))

    def _note(kind, num_hit, new_hit, request, local):
        _state[kind] += 1
        tot = _state["capped"] + _state["zeroed"]
        if tot in (1, 10, 100, 1000, 10000):
            log.warning("ORN-MTP-KVFIX2: RAM-tier hit bounded %s -> %d tokens (num_tokens %d, local %d; #%d)",
                        num_hit, new_hit, int(request.num_tokens), local, tot)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        on = getattr(self, "_orn_kvfix_on", False) and not getattr(request, "skip_reading_prefix_cache", False)
        cap = None
        if on and mode == "lookup":
            try:
                A = int(self._mamba_align_size)
                cap = _cap_tokens(int(request.num_tokens), A)
                # _lookup bound = round_down(n' - 1, A); n' = num_tokens - A gives exactly cap
                self._orn_cap_n = max(int(request.num_tokens) - A, 0)
                _state["calls"] += 1
            except Exception as exc:
                self._orn_cap_n = None
                _state["errors"] += 1
                if _state["errors"] <= 3:
                    log.warning("ORN-MTP-KVFIX2: cap setup error %r -> stock lookup", exc)
        try:
            res = orig_get(self, request, num_computed_tokens)
        finally:
            self._orn_cap_n = None
        if _TRACE:
            try:
                self._orn_rec[request.request_id] = (int(request.num_tokens), num_computed_tokens, res[0], mode,
                                                     cap, bool(getattr(request, "skip_reading_prefix_cache", False)))
            except Exception:
                pass
        try:
            if getattr(request, "skip_reading_prefix_cache", False) and res[0]:
                # priority-client (v3d): a request that must not read the GPU prefix cache must not read the RAM tier either
                self._req_status[request.request_id].update_num_hit_blocks(num_computed_tokens)
                _state["zeroed"] += 1
                return 0, False
            if not on or mode == "lookup":
                if on and res[0] and num_computed_tokens + res[0] == cap:
                    _note("capped", "(at cap)", res[0], request, num_computed_tokens)
                return res
            # ---- v1diag: the old post-hoc cut, plus the would-crash check (diagnostic builds only) ----
            _state["calls"] += 1
            num_hit, is_async = res
            if not num_hit:
                return res
            cap = _cap_tokens(int(request.num_tokens), int(self._mamba_align_size))
            served = num_computed_tokens + num_hit
            if served <= cap:
                return res
            new_hit = max(cap - num_computed_tokens, 0)
            min_blk = min(g.offloaded_block_size for g in self.config.kv_group_configs)
            if new_hit < min_blk:
                new_hit = 0
            self._req_status[request.request_id].update_num_hit_blocks(num_computed_tokens + new_hit)
            if new_hit:
                miss = _gdn_end_missing(self, request, num_computed_tokens + new_hit)
                if miss:
                    _state["would_crash"] += 1
                    log.warning("ORN-MTP-KVFIX2 v1diag: WOULD-CRASH #%d: cut %d -> %d tokens ends where the GDN "
                                "snapshot is not in the RAM tier (num_tokens %d, local %d)", _state["would_crash"],
                                num_hit, new_hit, int(request.num_tokens), num_computed_tokens)
                _note("capped", num_hit, new_hit, request, num_computed_tokens)
            else:
                _note("zeroed", num_hit, 0, request, num_computed_tokens)
            return new_hit, bool(new_hit)
        except Exception as exc:
            _state["errors"] += 1
            if _state["errors"] <= 3:
                log.warning("ORN-MTP-KVFIX2: error %r -> stock result", exc)
            return res

    orig_after = C.update_state_after_alloc

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if _TRACE and num_external_tokens:
            try:
                miss, info = _check_load(self, request, blocks, num_external_tokens)
                if miss:
                    _state["trace_miss"] = _state.get("trace_miss", 0) + 1
                    log.warning("ORN-MTP-KVFIX2 TRACE MISS #%d req=%s lookup(num_tokens,local,hit,mode,cap,skip)=%s "
                                "num_external=%d groups(idx,sw,gpu_bs,off_bs,local_gpu_blocks,start,end,present)=%s "
                                "missing(group,idx)=%s", _state["trace_miss"], request.request_id[-8:],
                                getattr(self, "_orn_rec", {}).get(request.request_id), num_external_tokens, info, miss)
            except Exception as exc:
                log.warning("ORN-MTP-KVFIX2 TRACE error %r", exc)
        return orig_after(self, request, blocks, num_external_tokens)

    update_state_after_alloc._orn_kvfix_orig = orig_after
    C.update_state_after_alloc = update_state_after_alloc
    __init__._orn_kvfix_orig = orig_init
    get_num_new_matched_tokens._orn_kvfix_orig = orig_get
    _lookup._orn_kvfix_orig = orig_lookup
    C.__init__ = __init__
    C.get_num_new_matched_tokens = get_num_new_matched_tokens
    C._lookup = _lookup
    _state["installed"] = True
    _state["mode"] = mode
    return f"installed ({mode})"


def status() -> dict:
    return dict(_state)
