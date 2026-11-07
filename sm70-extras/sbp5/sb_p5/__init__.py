"""Sound-barrier phase 5 (2026-10-15): small decode fusions for the Swift-1.5 Flash-Next brain, installed as RUNTIME
HOOKS from a worker extension (same style as P3 F1 / sb_fuse): no vLLM engine/venv file is edited, no VLLM_* variable
is added (vLLM hashes EVERY VLLM_* var that is set + the source of traced files -> fresh compile -> new numerics).

Features (all default OFF; each has its own switch, so they can be screened alone or stacked):
  SB_P5_GDN=1        (a) GDN layer tail, sub-switches:
                         SB_P5_GDN_MODE=trim   exact: drop the core_attn_out zero-fill + the verify-output memcpy
                                               (delta kernel writes every row straight into core_attn_out) -> bitwise
                         SB_P5_GDN_MODE=fused  one CUDA kernel = conv1d_update + l2norm + sigmoid gating + delta rule
                                               + direct output (replaces Fill/conv/delta/memcpy: 4 launches -> 1);
                                               fp32 state, tolerance-level (target numerics change -> full suite)
                         SB_P5_GDN_NORM=1      (fused only) also fold the gated RMSNorm into the kernel epilogue
                                               (5 launches -> 1)
  SB_P5_ROUTER=1     (b) MoE router GEMV + top-k/softmax/renorm in one kernel for decode M=1..20 (see router.py)
  SB_P5_DVOCAB=1     (c) V2-runner static draft vocab for the MTP drafter (see dvocab.py); SB_P5_DVOCAB_N=<rows>
  SB_P5_LIB_DIR      dir holding the built .so files (default /opt/1cat/soundbarrier/p5/build)
Switch sources (first hit wins): env SB_P5_<NAME> > file $SB_P5_CONF (default /opt/1cat/soundbarrier/p5/p5.conf).
"""

from __future__ import annotations

import os
import threading

_CONF_DEFAULT = "/opt/1cat/soundbarrier/p5/p5.conf"
_LIB_DIR_DEFAULT = "/opt/1cat/soundbarrier/p5/build"
_conf_cache: dict | None = None
_lock = threading.Lock()
_done: dict[str, str] = {}  # feature -> status line


def conf() -> dict:
    global _conf_cache
    if _conf_cache is None:
        d: dict = {}
        try:
            with open(os.environ.get("SB_P5_CONF", _CONF_DEFAULT)) as f:
                for line in f:
                    line = line.split("#", 1)[0].strip()
                    if "=" in line:
                        k, v = line.split("=", 1)
                        d[k.strip()] = v.strip()
        except OSError:
            pass
        _conf_cache = d
    return _conf_cache


def flag(name: str, default: str = "0") -> str:
    v = os.environ.get("SB_P5_" + name)
    if v is None:
        v = conf().get("SB_P5_" + name, default)
    return v


def flag_source(name: str) -> str:
    if "SB_P5_" + name in os.environ:
        return "env"
    if "SB_P5_" + name in conf():
        return "file"
    return "default"


def lib_path(stem: str) -> str:
    return os.path.join(flag("LIB_DIR", _LIB_DIR_DEFAULT), stem + ".so")


def load_lib(stem: str, probe_ns: str, probe_op: str) -> None:
    import torch

    ns = getattr(torch.ops, probe_ns, None)
    if ns is None or not hasattr(ns, probe_op):
        torch.ops.load_library(lib_path(stem))


def logger():
    try:
        from vllm.logger import init_logger

        return init_logger("vllm.sb_p5")  # under "vllm" so it reaches brain.log
    except Exception:  # pragma: no cover
        import logging

        return logging.getLogger("sb_p5")


# feature name -> (switch, module that must be imported first, installer "module:function")
FEATURES = {
    "gdn": ("GDN", "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn", "sb_p5.gdn:install"),
    "router": ("ROUTER", "vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv", "sb_p5.router:install"),
    "dvocab": ("DVOCAB", "vllm.models.qwen4_exp.nvidia.sm70_mtp_head", "sb_p5.dvocab:install"),
}


def wanted() -> dict[str, str]:
    """feature -> target module, for every feature whose switch is 1."""
    return {f: tgt for f, (sw, tgt, _) in FEATURES.items() if flag(sw) == "1"}


def install_feature(feature: str) -> str:
    """Install one feature once per process (idempotent). Never raises: an optional speed hook must not break a worker."""
    import importlib

    with _lock:
        if feature in _done:
            return _done[feature]
        log = logger()
        sw, _tgt, entry = FEATURES[feature]
        mod_name, fn_name = entry.split(":")
        try:
            status = getattr(importlib.import_module(mod_name), fn_name)(log) or "installed"
        except Exception as exc:  # stock path kept
            status = f"FAILED ({exc!r}); stock kept"
            log.warning("SB-P5 %s install failed in pid %d: %r; stock path kept", feature, os.getpid(), exc)
        _done[feature] = status
        log.info("SB-P5 %s in pid %d: %s [switch %s=%s]", feature, os.getpid(), status, sw, flag_source(sw))
        return status


def status() -> dict[str, str]:
    return dict(_done)


def uninstall_all() -> None:
    """Restore every patched attribute (unit tests)."""
    import importlib

    for feature, (_sw, _tgt, entry) in FEATURES.items():
        mod_name = entry.split(":")[0]
        try:
            m = importlib.import_module(mod_name)
            if hasattr(m, "uninstall"):
                m.uninstall()
        except Exception:
            pass
        _done.pop(feature, None)
