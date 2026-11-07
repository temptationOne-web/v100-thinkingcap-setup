"""Sound-barrier phase 3 (2026-10-14): SM70 decode fusions for the Swift-1.5 Flash-Next brain, installed as
RUNTIME HOOKS from a worker extension -- no file of the vLLM engine is edited.

Why hooks instead of a venv patch:
  * vLLM's torch.compile cache key hashes (a) every *traced* source file and (b) EVERY VLLM_* variable present in
    os.environ (envs.compile_factors(), "Out-of-tree switches ... Hash whatever is set"). Editing a traced file
    (model.py, hyperconnection.py, sm70_fp16_hc.py ...) or exporting a VLLM_* switch forces a fresh inductor compile,
    which on 10-14 21:42 alone changed greedy output (10/18 vs greedy-fixed). These hooks patch module attributes
    that are only looked up at RUN time inside opaque custom ops (fused_moe dispatch, vllm._custom_ops), so the
    compiled graphs and their cache key stay byte-identical (cache 74b3eb7076 is reused).
  * Switches: SB_FUSE_* (NOT hashed -> LKG compile cache reused; recommended) or VLLM_SB_FUSE_* (hashed -> fresh
    compile, new numerics reference). Both are read with os.environ directly; nothing is registered in vllm/envs.py.

Features (all default OFF):
  SB_FUSE_MTP_MOE=1        F1: exact SM70 FP16 MTP-MoE kernel for drafter batches M=2..SB_FUSE_MTP_MOE_MAX_M
                           (default 16; M=1/5 keep the stock path). Stock code only admits M in {1,5} (k=4 widths),
                           so at k=3 the drafter's M=4 first pass (and every 2u/4u draft pass) runs the generic Triton
                           fused_moe_kernel (~740+389 us per step at 1 user in the 10-10 trace).
                           Drafter-only: target-model numerics are untouched.
  SB_FUSE_HC=1             F2: replicated batched HyperConnection (M=2..20) in 2 launches instead of 3 (split-K
                           reduction fused into the down-partials kernel). Bitwise identical to stock.
  SB_FUSE_HC_UNROLL_DOWN   4|8|16|32 (default 4 = stock schedule)   -- bitwise identical for every value
  SB_FUSE_HC_UNROLL_UP     4|5|10|20 (default 4 = stock schedule)   -- bitwise identical for every value
  SB_FUSE_LIB              path of sb_fuse_C.so (default /opt/1cat/soundbarrier/p3/build/sb_fuse_C.so)
"""

from __future__ import annotations

import inspect
import os
import threading

_LIB_DEFAULT = "/opt/1cat/soundbarrier/p3/build/sb_fuse_C.so"
_installed = False
_lock = threading.Lock()
_log_lines: list[str] = []


_CONF_DEFAULT = "/opt/1cat/soundbarrier/p3/fuse.conf"
_conf_cache: dict | None = None


def conf() -> dict:
    """FILE switch (2026-10-15): KEY=VALUE lines (KEY = SB_FUSE_<name>), '#' comments. Path SB_FUSE_CONF or the default.
    Read once per process. Absent file = {} (everything OFF unless an env var says otherwise). A file switch does not
    depend on the env reaching the spawned TP workers and is not part of vLLM's compile key."""
    global _conf_cache
    if _conf_cache is None:
        d: dict = {}
        try:
            with open(os.environ.get("SB_FUSE_CONF", _CONF_DEFAULT)) as f:
                for line in f:
                    line = line.split("#", 1)[0].strip()
                    if "=" in line:
                        k, v = line.split("=", 1)
                        d[k.strip()] = v.strip()
        except OSError:
            pass
        _conf_cache = d
    return _conf_cache


def flag_source(name: str) -> str:
    if "SB_FUSE_" + name in os.environ:
        return "env"
    if "VLLM_SB_FUSE_" + name in os.environ:
        return "env-vllm"
    if "SB_FUSE_" + name in conf():
        return "file"
    return "default"


def flag(name: str, default: str = "0") -> str:
    """Read SB_FUSE_<name>, then VLLM_SB_FUSE_<name>, then the fuse.conf file, then the default."""
    v = os.environ.get("SB_FUSE_" + name, os.environ.get("VLLM_SB_FUSE_" + name))
    if v is None:
        v = conf().get("SB_FUSE_" + name, default)
    return v


def _logger():
    try:
        from vllm.logger import init_logger

        return init_logger("vllm.sb_fuse")  # under "vllm" so it reaches brain.log
    except Exception:  # pragma: no cover
        import logging

        return logging.getLogger("sb_fuse")


def load_library(path: str | None = None) -> None:
    import torch

    path = path or flag("LIB", _LIB_DEFAULT)
    if not hasattr(torch.ops, "sb_fuse") or not hasattr(torch.ops.sb_fuse, "hc_replicated_fused"):
        torch.ops.load_library(path)


# --------------------------------------------------------------------------------------------------------------
# F1: MTP MoE M=2..MAX_M
# --------------------------------------------------------------------------------------------------------------
_BM2_CONFIG = {
    "BLOCK_SIZE_M": 2,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "SPLIT_K": 1,
    "num_warps": 4,
    "num_stages": 3,
}


def _mtp_shape_ok(a, b, c, weights, ids, top_k, weighted, max_m) -> bool:
    import torch

    if weights is None or weights.ndim != 2 or weights.shape[1] != 10:
        return False
    m = weights.shape[0]
    if not (2 <= m <= max_m) or m == 5:  # M1/M5 stay on the stock exact op
        return False
    n, k = (2560, 160) if weighted else (320, 2560)
    return bool(
        top_k == (1 if weighted else 10)
        and tuple(a.shape) == ((m * 10, k) if weighted else (m, k))
        and tuple(b.shape) == (512, n, k)
        and tuple(c.shape) == (m, 10, n)
        and tuple(ids.shape) == (m * 10,)
        and a.dtype == b.dtype == c.dtype == torch.float16
        and weights.dtype == torch.float32
        and ids.dtype == torch.int32
        and all(t.device == a.device and t.is_contiguous() for t in (a, b, c, weights, ids))
    )



def _legacy_forced(fm) -> bool:
    """2026-10-25 compat: fused_moe._force_sm70_mtp_moe_legacy_config was a module bool (engine 10-08) and is a
    ContextVar in upstream 6a143197e+. A ContextVar object is always truthy, so the old `not fm._force...` test
    silently disabled F1 on the new engine."""
    v = fm._force_sm70_mtp_moe_legacy_config
    get = getattr(v, "get", None)
    return bool(get()) if callable(get) else bool(v)

def install_mtp_moe(log) -> bool:
    import torch

    import vllm.envs as envs
    import vllm.model_executor.layers.fused_moe.fused_moe as fm
    from vllm.platforms import current_platform
    from vllm.triton_utils import tl

    import sys

    tm = sys.modules.get("vllm.model_executor.layers.fused_moe.experts.triton_moe")
    if tm is None:  # (when installed from the end of triton_moe.py the module is already in sys.modules)
        try:
            import vllm.model_executor.layers.fused_moe.experts.triton_moe as tm
        except Exception:  # pragma: no cover
            tm = None

    if getattr(fm.dispatch_fused_moe_kernel, "_sb_fuse_orig", None) is not None:
        return True  # already installed in this process
    max_m = int(flag("MTP_MOE_MAX_M", "16"))
    max_m = max(2, min(max_m, 32))
    orig_cfg = fm._get_sm70_mtp_moe_decode_config
    orig_prep = fm._prepare_expert_assignment
    orig_dispatch = fm.dispatch_fused_moe_kernel
    sig = inspect.signature(orig_dispatch)

    def sb_cfg(M, E, N, K, topk):
        config = orig_cfg(M, E, N, K, topk)
        if (
            config is None
            and (E, N, K, topk) == (512, 160, 2560, 10)
            and 2 <= M <= max_m
            and not _legacy_forced(fm)
            and envs.VLLM_SM70_MTP_MOE_TUNED_CONFIG
        ):
            return dict(_BM2_CONFIG)
        return config

    def sb_prep(topk_ids, config, num_tokens, top_k_num, global_num_experts, expert_map, **kw):
        # Stock naive assignment needs M*topk*4 <= E (M <= 12 here). The route-parallel exact kernel does not
        # care, so also use naive routes for 12 < M <= max_m of this exact MTP shape (4 users x 4 tokens = 16).
        if (
            expert_map is None
            and global_num_experts == 512
            and top_k_num == 10
            and 12 < num_tokens <= max_m
            and config.get("BLOCK_SIZE_M") == 2
            and config.get("BLOCK_SIZE_N") == 128
            and not kw.get("use_int8_w8a16")
            and not kw.get("use_int4_w4a16")
        ):
            return (
                None,
                topk_ids.view(-1),
                torch.full((1,), topk_ids.numel() * 2, dtype=torch.int32, device=topk_ids.device),
            )
        return orig_prep(topk_ids, config, num_tokens, top_k_num, global_num_experts, expert_map, **kw)

    def sb_dispatch(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        p = bound.arguments
        A, B, C = p["A"], p["B"], p["C"]
        cfg = p["config"]
        if (
            envs.VLLM_SM70_MTP_MOE_FP16_EXACT
            and not envs.VLLM_BATCH_INVARIANT
            and not _legacy_forced(fm)
            and envs.VLLM_SM70_MTP_MOE_TUNED_CONFIG
            and p["sorted_token_ids"] is None
            and not (p["use_fp8_w8a8"] or p["use_int8_w8a8"] or p["use_int8_w8a16"] or p["use_int4_w4a16"])
            and p["A_scale"] is None
            and p["B_scale"] is None
            and p["B_zp"] is None
            and p["B_bias"] is None
            and p["block_shape"] is None
            and p["compute_type"] == tl.float16
            and cfg.get("BLOCK_SIZE_M") == 2
            and cfg.get("BLOCK_SIZE_N") == 128
            and cfg.get("BLOCK_SIZE_K") == 64
            and cfg.get("num_warps") == 4
            and A.is_cuda
            and current_platform.is_device_capability(70)
            and _mtp_shape_ok(
                A, B, C, p["topk_weights"], p["expert_ids"], p["top_k"], p["mul_routed_weight"], max_m
            )
            and B.data_ptr() % 16 == 0
            and p["num_tokens_post_padded"].device == A.device
            and p["num_tokens_post_padded"].dtype == torch.int32
            and p["num_tokens_post_padded"].numel() == 1
        ):
            torch.ops.sb_fuse.mtp_moe_fp16_out(
                C, A, B, p["expert_ids"], p["topk_weights"], p["num_tokens_post_padded"], p["mul_routed_weight"]
            )
            if not getattr(sb_dispatch, "_announced", False):
                sb_dispatch._announced = True
                log.info(
                    "SB-FUSE F1 active: exact SM70 FP16 MTP MoE, first M=%d (range 2..%d)",
                    p["topk_weights"].shape[0],
                    max_m,
                )
            return None
        return orig_dispatch(*args, **kwargs)

    sb_dispatch._sb_fuse_orig = orig_dispatch
    sb_prep._sb_fuse_orig = orig_prep
    sb_cfg._sb_fuse_orig = orig_cfg
    fm._get_sm70_mtp_moe_decode_config = sb_cfg
    fm._prepare_expert_assignment = sb_prep
    fm.dispatch_fused_moe_kernel = sb_dispatch
    if tm is not None:
        if getattr(tm, "_prepare_expert_assignment", None) is orig_prep:
            tm._prepare_expert_assignment = sb_prep
        if getattr(tm, "dispatch_fused_moe_kernel", None) is orig_dispatch:
            tm.dispatch_fused_moe_kernel = sb_dispatch
    return True


# --------------------------------------------------------------------------------------------------------------
# F2: HC replicated, 3 -> 2 launches, bitwise identical
# --------------------------------------------------------------------------------------------------------------
_SLOTS = 256
_SLOT_STRIDE = 64  # >= 3 groups x 11 tiles
_counter_buffers: dict = {}
_slot_of: dict = {}


def _counters(device, key: int):
    """Per-call-site slice of a zeroed int32 buffer (self-resetting in-kernel); None while capturing w/o buffer."""
    import torch

    buf = _counter_buffers.get(device)
    if buf is None:
        if torch.cuda.is_current_stream_capturing():
            return None  # never allocate inside a CUDA-graph capture; stock path for this call
        buf = torch.zeros((_SLOTS * _SLOT_STRIDE,), dtype=torch.int32, device=device)
        _counter_buffers[device] = buf
    slot = _slot_of.get((device, key))
    if slot is None:
        slot = len([k for k in _slot_of if k[0] == device])
        if slot >= _SLOTS:
            return None
        _slot_of[(device, key)] = slot
    return buf[slot * _SLOT_STRIDE : (slot + 1) * _SLOT_STRIDE]


def install_hc(log) -> bool:
    import torch

    import vllm._custom_ops as ops

    ud = int(flag("HC_UNROLL_DOWN", "4"))
    uu = int(flag("HC_UNROLL_UP", "4"))
    if ud not in (4, 8, 16, 32) or uu not in (4, 5, 10, 20):
        log.warning("SB-FUSE F2 not installed: bad unroll (down=%s up=%s)", ud, uu)
        return False
    if getattr(ops.sm70_qwen38_hc_replicated, "_sb_fuse_orig", None) is not None:
        return True  # already installed in this process
    orig = ops.sm70_qwen38_hc_replicated
    warned = {"capture": False, "announced": False}

    def sb_hc_replicated(x, down, up, partials, lora, output, injection) -> None:
        m = x.shape[0]
        if 2 <= m <= 20 and tuple(partials.shape) == (20, m, 352):
            ctr = _counters(x.device, down.data_ptr())
            if ctr is not None:
                torch.ops.sb_fuse.hc_replicated_fused(x, down, up, partials, lora, output, injection, ctr, ud, uu)
                if not warned["announced"]:
                    warned["announced"] = True
                    log.info("SB-FUSE F2 active: HC replicated 2-launch (unroll down=%d up=%d), first M=%d", ud, uu, m)
                return
            if not warned["capture"]:
                warned["capture"] = True
                log.warning("SB-FUSE F2: a call during graph capture had no counter buffer -> stock path there")
        orig(x, down, up, partials, lora, output, injection)

    sb_hc_replicated._sb_fuse_orig = orig
    ops.sm70_qwen38_hc_replicated = sb_hc_replicated
    return True


def uninstall() -> None:
    """Restore every patched attribute (used by the unit test)."""
    import vllm._custom_ops as ops
    import vllm.model_executor.layers.fused_moe.fused_moe as fm

    try:
        import vllm.model_executor.layers.fused_moe.experts.triton_moe as tm
    except Exception:  # pragma: no cover
        tm = None
    global _installed
    for mod in (fm, tm):
        if mod is None:
            continue
        for name in ("dispatch_fused_moe_kernel", "_prepare_expert_assignment", "_get_sm70_mtp_moe_decode_config"):
            fn = getattr(mod, name, None)
            if fn is not None and getattr(fn, "_sb_fuse_orig", None) is not None:
                setattr(mod, name, fn._sb_fuse_orig)
    fn = ops.sm70_qwen38_hc_replicated
    if getattr(fn, "_sb_fuse_orig", None) is not None:
        ops.sm70_qwen38_hc_replicated = fn._sb_fuse_orig
    _installed = False
    _log_lines.clear()


def install() -> list[str]:
    """Install the enabled hooks once per process. Returns the list of active features."""
    global _installed
    with _lock:
        if _installed:
            return list(_log_lines)
        _installed = True
        log = _logger()
        want_mtp = flag("MTP_MOE") == "1"
        want_hc = flag("HC") == "1"
        if not (want_mtp or want_hc):
            return []
        try:
            load_library()
        except Exception as exc:
            log.warning("SB-FUSE: cannot load %s (%s); all fusions OFF", flag("LIB", _LIB_DEFAULT), exc)
            return []
        if want_mtp:
            try:
                if install_mtp_moe(log):
                    _log_lines.append("mtp_moe(M2..%s)" % flag("MTP_MOE_MAX_M", "16"))
            except Exception as exc:
                log.warning("SB-FUSE F1 install failed (%s); stock MTP MoE kept", exc)
        if want_hc:
            try:
                if install_hc(log):
                    _log_lines.append(
                        "hc_fused(ud=%s,uu=%s)" % (flag("HC_UNROLL_DOWN", "4"), flag("HC_UNROLL_UP", "4"))
                    )
            except Exception as exc:
                log.warning("SB-FUSE F2 install failed (%s); stock HC kept", exc)
        log.info(
            "SB-FUSE installed in pid %d: %s [switch MTP_MOE=%s HC=%s; env probe SB_FUSE_ENV_PROBE=%s PYTHONPATH=%s]",
            os.getpid(),
            ", ".join(_log_lines) or "nothing",
            flag_source("MTP_MOE"),
            flag_source("HC"),
            os.environ.get("SB_FUSE_ENV_PROBE", "unset"),
            "set" if os.environ.get("PYTHONPATH") else "unset",
        )
        return list(_log_lines)
