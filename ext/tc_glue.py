"""TC-GLUE (2026-10-26): ThinkingCap (Qwen4Exp / Qwen3.8 Flash-Next, TP4 fp16 SM70) sync glue, ported from Ornith's
arnorm/ARX ports. Each hyper-connection boundary at decode becomes ONE kernel: two-shot all-reduce (ARX protocol, half the
PCIe wire bytes of the live one-shot push / sum2 kernels) + HC combine + grouped Gemma RMSNorm, bit-identical to the live
path (see tc_glue_src/tc_glue.cu and GLUE-README.md).

What changes when TC_GLUE=1 (runtime hooks only; the engine files are untouched):
  * upstream's own HCX plumbing is wired WITHOUT its NVLink runtime: attention out_proj/o_proj.reduce_results=False +
    sm70_hcx output-projection op (partials for M <= 40, all-reduced for larger M), MoE runner output_transform ->
    sm70_hcx_moe_output (two-plane [routed; shared] payload for M <= 40, reduced payload otherwise), consumer HC
    modules get _partial_inputs (= the graph upstream compiles for HCX; new torch.compile cache key factor "tc_glue-v1").
  * GatedResidual._hcx (the "runtime" upstream's qwen38_sm70_hcx_combine_and_mix op calls for M <= 40) = _Runtime:
        res_out, xn = torch.ops.tc_glue.ar_combine_norm(partial, secondary, hidden, inj, hc_norm.weight)
        block, inj = module._project(xn)        (the stock fp16 fused HC down/silu/up/gate-mix, unchanged)
  * GatedResidual._reduce_partial (used by the PLE layer's combine and the final mixers) -> torch.ops.tc_glue.ar2
    (two-shot AR with the fused shared+routed half add; norm stays live there).
  Inside the ops: capturing a CUDA graph and 1 <= M <= TC_GLUE_MAXM -> fused kernel (tree 1024 for M == 1, 512 otherwise
  = the production Triton kernel's trees); otherwise the live-equivalent fallback: tensor_model_parallel_all_reduce(_sum2)
  + torch.ops.vllm.qwen4_exp_hc_combine_norm. Prefill and eager calls take the fallback.

Env (none are VLLM_*):
  TC_GLUE=1               enable (default 0: install() is a no-op)
  TC_GLUE_MAXM=16         largest M on the fused kernel (1..40)
  TC_GLUE_TREE=auto       auto | 512 | 1024 (force one reduction tree: batch invariant across M, NOT bit-identical to live at M==1)
  TC_GLUE_DRAFT=1         also wire the MTP draft model (its decoder layer + final mixer)
  TC_GLUE_EAGER=1         use the fused kernel outside graph capture too (tests only)
  TC_GLUE_LIB             path of tc_glue_C.so (default <this dir>/build/tc_glue_C.so)
Memory: 800 KB IPC buffer per GPU (+ IPC mappings of the 3 peers).
"""

import importlib.abc
import importlib.util
import logging
import os
import sys
import threading
from functools import partial

_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_DEFAULT = f"{_DIR}/build/tc_glue_C.so"
_CA_MOD = "vllm.distributed.device_communicators.custom_all_reduce"
_MODEL_MOD = "vllm.models.qwen4_exp.nvidia.model"
_MTP_MOD = "vllm.models.qwen4_exp.nvidia.mtp"
_HC_MOD = "vllm.models.qwen4_exp.nvidia.hyperconnection"
_CACHING_MOD = "vllm.compilation.caching"
CACHE_FACTOR = "tc_glue-v1"
H, HC, KD = 2560, 4, 10240
EPS = 1e-6
HCX_BATCH_M = 40

log = logging.getLogger("tc_glue")
_lock = threading.RLock()
_state = {"ca": False, "model": False, "mtp": False, "hc": False, "caching": False, "ops": False, "lib": None,
          "status": "not installed"}
_stats = {"setup_ok": 0, "setup_skipped": 0, "last_reason": None, "linked_target": 0, "linked_draft": 0,
          "link_reason": None, "fused_norm": 0, "fused_ar": 0, "fallback_norm": 0, "fallback_ar": 0, "fallback_eager": 0}
_G = {}


def enabled() -> bool:
    return os.environ.get("TC_GLUE", "0") == "1"


def _int(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _say(msg):
    try:
        from vllm.logger import init_logger

        init_logger("tc_glue").warning("TC-GLUE " + msg)
    except Exception:  # noqa: BLE001
        log.warning("TC-GLUE %s", msg)


def _load_lib() -> bool:
    if _state["lib"] is None:
        try:
            import torch

            if not hasattr(torch.ops, "tc_glue_c") or not hasattr(torch.ops.tc_glue_c, "fused"):
                torch.ops.load_library(os.environ.get("TC_GLUE_LIB", _LIB_DEFAULT))
            _state["lib"] = hasattr(torch.ops.tc_glue_c, "fused")
        except Exception as exc:  # noqa: BLE001
            _say(f"library load failed {exc!r}")
            _state["lib"] = False
    return _state["lib"]


# ----------------------------------------------------------------------------------------------- communicator setup
def _setup(ca) -> None:
    """Collective over ca.group (all 4 ranks agree or none uses the fused kernel). Never raises."""
    import torch
    import torch.distributed as dist

    ca._tc_glue = None
    if getattr(ca, "disabled", True) or getattr(ca, "world_size", 0) != 4:
        return
    reason = None
    try:
        if torch.cuda.get_device_capability() != (7, 0):
            reason = "capability"
        elif not getattr(ca, "fully_connected", False):
            reason = "not fully connected"
        elif not _load_lib():
            reason = "library"
    except Exception as exc:  # noqa: BLE001
        reason = repr(exc)
    reasons = [None] * 4
    dist.all_gather_object(reasons, reason, group=ca.group)
    bad = next((r for r in reasons if r is not None), None)
    if bad is not None:
        _stats["setup_skipped"] += 1
        _stats["last_reason"] = bad
        _say(f"fused kernel not used on this communicator: {bad} (live-equivalent fallback only)")
        return
    ok, mine, opened, hx = True, None, [], None
    try:
        with torch.cuda.device(ca.device):
            p, hd = torch.ops.tc_glue_c.alloc()
            mine = (p, bytes(hd.numpy().tobytes()))
    except Exception as exc:  # noqa: BLE001
        ok = False
        _say(f"alloc failed {exc!r}")
    handles = [None] * 4
    dist.all_gather_object(handles, mine if ok else None, group=ca.group)
    if all(h is not None for h in handles):
        try:
            with torch.cuda.device(ca.device):
                ptrs = []
                for i, h in enumerate(handles):
                    if i == ca.rank:
                        ptrs.append(mine[0])
                    else:
                        q = torch.ops.tc_glue_c.open(torch.frombuffer(bytearray(h[1]), dtype=torch.uint8))
                        opened.append(q)
                        ptrs.append(q)
                hx = torch.ops.tc_glue_c.init(ptrs, ca.rank)
        except Exception as exc:  # noqa: BLE001
            ok = False
            _say(f"open failed {exc!r}")
    else:
        ok = False
    oks = [None] * 4
    dist.all_gather_object(oks, ok, group=ca.group)
    if not all(oks):
        _stats["setup_skipped"] += 1
        _stats["last_reason"] = "setup failed on a rank"
        for q in opened:
            try:
                torch.ops.tc_glue_c.close_handle(q)
            except Exception:  # noqa: BLE001
                pass
        if mine is not None:
            try:
                torch.ops.tc_glue_c.free(mine[0])
            except Exception:  # noqa: BLE001
                pass
        return
    ca._tc_glue = {"h": hx, "mine": mine[0], "opened": opened}
    _stats["setup_ok"] += 1
    _say(f"fused all-reduce+combine+norm ready on rank {ca.rank} (maxM {_maxm()}, tree {os.environ.get('TC_GLUE_TREE', 'auto')})")


def _install_ca(m) -> None:
    cls = m.CustomAllreduce
    if getattr(cls.__init__, "_tc_glue", False):
        return
    orig_init, orig_close = cls.__init__, cls.close

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        try:
            _setup(self)
        except Exception as exc:  # noqa: BLE001
            self._tc_glue = None
            _say(f"setup error {exc!r}")

    def close(self):
        st = getattr(self, "_tc_glue", None)
        if st is not None:
            self._tc_glue = None
            try:
                import torch

                torch.cuda.synchronize()
                for q in st["opened"]:
                    torch.ops.tc_glue_c.close_handle(q)
                torch.ops.tc_glue_c.free(st["mine"])
            except Exception:  # noqa: BLE001
                pass
        return orig_close(self)

    __init__._tc_glue = True
    cls.__init__ = __init__
    cls.close = close
    _state["ca"] = True


def _maxm() -> int:
    return max(1, min(_int("TC_GLUE_MAXM", 16), HCX_BATCH_M))


def _tree(M: int) -> int:
    t = os.environ.get("TC_GLUE_TREE", "auto")
    if t in ("512", "1024"):
        return int(t)
    return 1024 if M == 1 else 512  # the live Triton kernel: BLOCK_SIZE 1024 iff N == 1 on SM70


def _comm():
    try:
        from vllm.distributed.parallel_state import get_tp_group

        dc = get_tp_group().device_communicator
        ca = getattr(dc, "ca_comm", None)
        if ca is None or getattr(ca, "disabled", True):
            return None
        return getattr(ca, "_tc_glue", None)
    except Exception:  # noqa: BLE001
        return None


def _use_fused(x):
    import torch

    M = x.shape[0]
    if not (1 <= M <= _maxm()) or x.dtype != torch.float16 or not x.is_cuda:
        return None
    st = _comm()
    if st is None:
        return None
    if torch.cuda.is_current_stream_capturing() or os.environ.get("TC_GLUE_EAGER", "0") == "1":
        return st
    return None


def _live_reduce(p, second):
    from vllm.distributed import tensor_model_parallel_all_reduce, tensor_model_parallel_all_reduce_sum2

    if second is not None:
        return tensor_model_parallel_all_reduce_sum2(p, second)
    return tensor_model_parallel_all_reduce(p)


# ----------------------------------------------------------------------------------------------- custom ops
def _register_ops() -> None:
    if _state["ops"]:
        return
    import torch

    @torch.library.custom_op("tc_glue::ar_combine_norm", mutates_args=())
    def ar_combine_norm(p: torch.Tensor, second: torch.Tensor | None, res: torch.Tensor, inj: torch.Tensor,
                        w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ar = AR(p) (or AR(half(p + second)));  res_out = f16(res + ar * 2*sigmoid(inj/4)) per HC stream;
        xn = grouped Gemma RMSNorm(res_out, w). p/second: [M, 2560] fp16 TP partials; res: [M, 10240]; inj: [M, 4]."""
        _load_lib()
        p = p.contiguous()
        second = None if second is None else second.contiguous()
        res = res.contiguous()
        inj = inj if inj.stride(1) == 1 else inj.contiguous()
        M = p.shape[0]
        st = _use_fused(p)
        if st is not None:
            res_out = torch.empty((M, KD), dtype=torch.float16, device=p.device)
            xn = torch.empty((M, KD), dtype=torch.float16, device=p.device)
            torch.ops.tc_glue_c.fused(st["h"], 2, _tree(M), p, second, res, inj, w, None, res_out, xn)
            _stats["fused_norm"] += 1
            return res_out, xn
        reduced = _live_reduce(p, second)
        res_out, xn = torch.ops.vllm.qwen4_exp_hc_combine_norm(res, reduced, inj, w, EPS, HC)
        _stats["fallback_norm"] += 1
        if not torch.cuda.is_current_stream_capturing():
            _stats["fallback_eager"] += 1
        return res_out, xn

    @ar_combine_norm.register_fake
    def _(p, second, res, inj, w):
        return res.new_empty(res.shape), res.new_empty(res.shape)

    @torch.library.custom_op("tc_glue::ar2", mutates_args=())
    def ar2(p: torch.Tensor, second: torch.Tensor | None) -> torch.Tensor:
        """AR(p) or AR(half(p + second)) with the two-shot kernel (fp16 [M, 2560]); live all-reduce otherwise."""
        _load_lib()
        p = p.contiguous()
        second = None if second is None else second.contiguous()
        M = p.shape[0]
        st = _use_fused(p)
        if st is not None:
            out = torch.empty_like(p)
            torch.ops.tc_glue_c.fused(st["h"], 0, 512, p, second, None, None, None, out, None, None)
            _stats["fused_ar"] += 1
            return out
        _stats["fallback_ar"] += 1
        if not torch.cuda.is_current_stream_capturing():
            _stats["fallback_eager"] += 1
        return _live_reduce(p, second)

    @ar2.register_fake
    def _(p, second):
        return p.new_empty(p.shape)

    _state["ops"] = True


# ----------------------------------------------------------------------------------------------- model wiring
class _Runtime:
    """Stands in for sm70_hcx.Sm70HcxRuntime on one GatedResidual (what qwen38_sm70_hcx_combine_and_mix calls)."""

    diagnostic = False
    local_schedule = False
    enabled = True

    def __init__(self, module):
        self.module = module

    def run(self, partial, hidden, injection, norm_weight, eps, packed_down, packed_up, oproj=None, secondary=None,
            snapshot_name=None, outputs=None):
        import torch

        res_out, xn = torch.ops.tc_glue.ar_combine_norm(partial, secondary, hidden, injection, norm_weight)
        block, inj = self.module._project(xn)
        return res_out, block, inj


def _eligible_layer(layer) -> str | None:
    try:
        from vllm.distributed import get_tensor_model_parallel_world_size

        if get_tensor_model_parallel_world_size() != 4:
            return "tp != 4"
        for nm in ("attn_hyper_connection", "mlp_hyper_connection"):
            r = _eligible_hc(getattr(layer, nm), True)
            if r is not None:
                return f"{nm}: {r}"
        if layer.layer_type == "linear_attention":
            proj = layer.linear_attn.out_proj
            if getattr(layer.linear_attn, "output_projection_override", None) is not None:
                return "GDN output_projection_override already set"
            # (the GDN full-forward opaque route still reaches _compute_output_projection, which honours the override)
        elif layer.layer_type == "full_attention":
            proj = layer.self_attn.o_proj
            if getattr(layer.self_attn, "sm70_hcx_projection_name", None) is not None:
                return "attention projection already HCX"
            if not hasattr(layer.self_attn, "_run_qsa"):
                return "full attention is not QSA (only QSA honours sm70_hcx_projection_name)"
        else:
            return "layer type"
        if not proj.reduce_results or getattr(proj, "bias", None) is not None:
            return "out projection (reduce_results off or bias)"
        mlp = layer.mlp
        experts = getattr(mlp, "experts", None)
        runner = getattr(experts, "runner", None)
        if runner is None:
            return "mlp without a MoE runner"
        if getattr(mlp, "is_sequence_parallel", False) or getattr(mlp, "sm70_hcx_packed_outputs", False):
            return "mlp sequence parallel / already packed"
        if runner._fused_output_is_reduced:
            return "MoE kernel output already reduced"
        if type(getattr(runner, "router", None)).__name__ == "ZeroExpertRouter":
            return "zero expert router"
        if getattr(runner, "output_transform", None) is not None:
            return "runner output_transform already set"
    except Exception as exc:  # noqa: BLE001
        return repr(exc)
    return None


def _eligible_hc(hc, need_combine) -> str | None:
    import torch

    if hc is None:
        return "missing"
    if getattr(hc, "_partial_inputs", False):
        return "already partial"
    if hc.hc_count != HC or hc.hidden_size != H:
        return "hc shape"
    if abs(float(hc.config.rms_norm_eps) - EPS) > 0:
        return "eps"
    w = hc.hc_norm.weight
    if w.dtype != torch.float16 or w.numel() != KD:
        return "norm weight (needs per-branch fp16 [10240])"
    if need_combine and not hc.use_combine:
        return "use_combine"
    return None


def _link(layers, final_mixer, tag) -> int:
    """Wire upstream's HCX plumbing (no NVLink runtime) + this port's runtime on every layer, or nothing."""
    import torch
    from vllm.models.qwen4_exp.nvidia import sm70_hcx as X

    layers = list(layers)
    if not layers:
        _stats["link_reason"] = f"{tag}no layers"
        return 0
    for i, layer in enumerate(layers):
        r = _eligible_layer(layer)
        if r is not None:
            _stats["link_reason"] = f"{tag}layer {i}: {r}"
            _say(f"not linked ({tag}layer {i}: {r}); whole model stays live")
            return 0
    if final_mixer is not None:
        r = _eligible_hc(final_mixer, False)
        if r is not None:
            _stats["link_reason"] = f"{tag}final mixer: {r}"
            _say(f"not linked ({tag}final mixer: {r}); whole model stays live")
            return 0
    n = 0
    for i, layer in enumerate(layers):
        prefix = f"{tag}layer{layer.layer_idx}"
        if layer.layer_type == "linear_attention":
            attention, attention_module = layer.linear_attn.out_proj, layer.linear_attn
        else:
            attention, attention_module = layer.self_attn.o_proj, layer.self_attn
        attention.reduce_results = False
        X.register_output_projection(prefix, attention, False)
        attention_module.output_projection_override = partial(X.partial_projection, prefix)
        attention_module.sm70_hcx_projection_name = prefix
        runner = layer.mlp.experts.runner
        runner.output_transform = partial(X.partial_moe_output, prefix)
        layer.mlp.sm70_hcx_packed_outputs = True
        X.register_moe_runner(prefix, runner)
        # consumer of this layer's MoE payload: the next layer's attn HC (target) or the final mixer (draft / last layer)
        consumer = layers[i + 1].attn_hyper_connection if (tag == "" and i + 1 < len(layers)) else final_mixer
        if consumer is not None and not getattr(consumer, "_partial_inputs", False):
            consumer._hcx_moe_payload = True
            consumer.enable_partial_inputs(prefix + ".moe")
            consumer._tc_glue = True
            if consumer.use_combine:
                consumer._hcx_down = consumer._hcx_up = None
                consumer._hcx = _Runtime(consumer)
        mlp_hc = layer.mlp_hyper_connection
        mlp_hc.enable_partial_inputs(prefix + ".attn")
        mlp_hc._tc_glue = True
        mlp_hc._hcx_down = mlp_hc._hcx_up = None
        mlp_hc._hcx = _Runtime(mlp_hc)
        n += 1
    _say(f"linked {n} {tag or 'target '}layers (upstream HCX plumbing + tc_glue runtime; maxM {_maxm()})")
    return n


def _install_model(m) -> None:
    if getattr(m._maybe_enable_sm70_peer_paths, "_tc_glue", False):
        return
    orig = m._maybe_enable_sm70_peer_paths

    def _maybe_enable_sm70_peer_paths(vllm_config, model):
        orig(vllm_config, model)
        try:
            st = getattr(model, "sm70_hcx_status", None)
            if st is not None and st.get("enabled"):
                _say("upstream HCX is enabled; tc_glue not linked")
                return
            layers = [l for l in model.layers if isinstance(l, m.Qwen4ExpDecoderLayer)]
            _stats["linked_target"] = _link(layers, getattr(model, "hyper_connection_mixer", None), "")
            _G["target"] = model
        except Exception as exc:  # noqa: BLE001
            _stats["link_reason"] = repr(exc)
            _say(f"target link failed {exc!r} (live path)")

    _maybe_enable_sm70_peer_paths._tc_glue = True
    _maybe_enable_sm70_peer_paths._tc_glue_orig = orig
    m._maybe_enable_sm70_peer_paths = _maybe_enable_sm70_peer_paths
    _state["model"] = True


def _install_mtp(m) -> None:
    cls = m.Qwen4ExpMultiTokenPredictor
    if getattr(cls.load_weights, "_tc_glue", False):
        return
    orig = cls.load_weights

    def load_weights(self, weights):
        out = orig(self, weights)
        if os.environ.get("TC_GLUE_DRAFT", "1") == "1":
            try:
                _stats["linked_draft"] = _link(list(self.layers), getattr(self, "hyper_connection_mixer", None), "mtp.")
                _G["draft"] = self
            except Exception as exc:  # noqa: BLE001
                _stats["link_reason"] = repr(exc)
                _say(f"draft link failed {exc!r} (live path)")
        return out

    load_weights._tc_glue = True
    load_weights._tc_glue_orig = orig
    cls.load_weights = load_weights
    _state["mtp"] = True


def _install_hc(m) -> None:
    import torch

    cls = m.GatedResidual
    if getattr(cls._reduce_partial, "_tc_glue", False):
        return
    orig = cls._reduce_partial

    def _reduce_partial(self, block_output):
        if not getattr(self, "_partial_inputs", False) or not getattr(self, "_tc_glue", False):
            return orig(self, block_output)
        first, second = self._partial_pair(block_output)
        if first.shape[0] > HCX_BATCH_M:
            return first
        return torch.ops.tc_glue.ar2(first, second)

    _reduce_partial._tc_glue = True
    _reduce_partial._tc_glue_orig = orig
    cls._reduce_partial = _reduce_partial
    _state["hc"] = True


def _install_caching(m) -> None:
    """An AOT compile artifact made without the port must never be loaded with it (and vice versa)."""
    if getattr(m.aot_compile_hash_factors, "_tc_glue", False):
        return
    orig = m.aot_compile_hash_factors

    def aot_compile_hash_factors(*a, **kw):
        f = list(orig(*a, **kw))
        f.append(CACHE_FACTOR)
        return f

    aot_compile_hash_factors._tc_glue = True
    aot_compile_hash_factors._tc_glue_orig = orig
    m.aot_compile_hash_factors = aot_compile_hash_factors
    _state["caching"] = True


_TARGETS = {_CA_MOD: ("ca", _install_ca), _MODEL_MOD: ("model", _install_model), _MTP_MOD: ("mtp", _install_mtp),
            _HC_MOD: ("hc", _install_hc), _CACHING_MOD: ("caching", _install_caching)}


def _all_installed() -> bool:
    return all(_state[k] for k, _ in _TARGETS.values())


def _try_install_targets():
    with _lock:
        for mod, (key, fn) in _TARGETS.items():
            if not _state[key] and mod in sys.modules:
                try:
                    if key in ("model", "mtp", "hc"):
                        _register_ops()
                    fn(sys.modules[mod])
                except Exception as exc:  # noqa: BLE001
                    _say(f"patch of {mod} failed {exc!r}")
        _state["status"] = "installed" if _all_installed() else "armed (%s)" % ",".join(
            f"{k}={int(_state[k])}" for k, _ in _TARGETS.values())


class _PostImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in _TARGETS:
            return None
        if self in sys.meta_path:
            sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            if not _all_installed() and self not in sys.meta_path:
                sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        real_exec = spec.loader.exec_module

        def exec_module(module):
            real_exec(module)
            _try_install_targets()
            if _all_installed() and self in sys.meta_path:
                sys.meta_path.remove(self)

        spec.loader.exec_module = exec_module
        return spec


def install() -> None:
    """Idempotent, never raises, no CUDA work. No-op unless TC_GLUE=1."""
    try:
        if not enabled():
            _state["status"] = "off (TC_GLUE!=1)"
            return
        with _lock:
            _register_ops()
        _try_install_targets()
        if not _all_installed() and not any(isinstance(f, _PostImport) for f in sys.meta_path):
            sys.meta_path.insert(0, _PostImport())
        _say(f"install: {_state['status']}")
    except Exception as exc:  # noqa: BLE001
        log.warning("TC-GLUE install() error %r", exc)


def status() -> dict:
    return {"status": _state["status"], "maxm": _maxm(), "tree": os.environ.get("TC_GLUE_TREE", "auto"), **dict(_stats)}
