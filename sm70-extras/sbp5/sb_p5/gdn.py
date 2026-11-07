"""P5 (a): GDN layer tail for the MTP verify step (36 GDN layers/step) -- runtime hooks, no engine file edited.

Prod route (brain.log 10-15 + qwen_gdn_linear_attn.py): forward() [traced] -> vllm::qwen_gdn_full_forward [opaque op]
-> self._full_forward -> self._forward_method (= forward_cuda, bound in __init__) -> qwen38 fused input projection
(gdn_input_batch_kernel) -> torch.zeros(core_attn_out) [Fill<Half>] -> vllm::qwen_gdn_attention_core_standard_spec
-> self._forward_core -> causal_conv1d_update [Triton _causal_conv1d_update_kernel]
-> fused_sigmoid_gating_delta_rule_update_mixed_qkv [Triton, BV32, 4 warps] -> core_attn_out[:n] = out [memcpy32_post,
only when M not in (5, 20)] -> self._output_projection -> self._compute_output_projection -> self.norm
[rmsnorm_gated_exact_kernel] -> out_proj [dense_batch_kernel<1536>] -> push all-reduce.
Everything after forward() is eager Python inside opaque custom ops, captured into the FULL CUDA graph, so class
attributes patched before capture take effect and the compile key is untouched.

Modes (SB_P5_GDN=1 plus):
  SB_P5_GDN_MODE=trim  : exact (bitwise). The stock delta kernel writes straight into core_attn_out for every M (stock
                         does that only for M=5/20) -> the verify memcpy disappears. Nothing else changes.
  SB_P5_GDN_MODE=fused : torch.ops.sb_p5_gdn.verify_fused = conv1d_update + l2norm + gating + delta rule (+ norm) in ONE
                         kernel (src/sb_p5_gdn.cu). Sub-switches:
                           SB_P5_GDN_BV=16|32|64|128 (v columns per block, default 32; NORM forces 128)
                           SB_P5_GDN_CONV=1|0  (1 = conv fused in the kernel; 0 = stock conv kernel, fused delta)
                           SB_P5_GDN_NORM=1|0  (1 = also the exact gated RMSNorm in the epilogue: the core call is
                                                deferred to _compute_output_projection, where z is known)
Only a pure speculative-verify batch (no prefill/plain-decode rows, no DFlash2/DDTree) with the prod shapes takes the
fast path, the same admission as the stock mixed-QKV verifier route (1 < M <= 16 or M == 20); everything else is stock.
"""

from __future__ import annotations

import os
import threading

import sb_p5

_TARGET_CLS = "QwenGatedDeltaNetAttention"
_tls = threading.local()
_pending: dict = {}  # id(layer) -> launch params (NORM deferral)
_state = {"announced": False, "norm_broken": False, "warned": set(), "hits": 0, "mode": None}
_bufs: dict = {}  # device -> (scratch, sync, stream_ptr)
_dtb: dict = {}  # id(layer) -> fp32 dt_bias copy
_orig: dict = {}  # name -> original function

SCRATCH_FLOATS = 32 * 4 * 8 * 2 * 128
SYNC_INTS = 2 * 32 * 4 + 2


def _warn_once(log, key, msg, *args):
    if key not in _state["warned"]:
        _state["warned"].add(key)
        log.warning(msg, *args)


def _capturing() -> bool:
    import torch

    return torch.cuda.is_current_stream_capturing()


def _buffers(device):
    """(scratch, sync) per device, allocated once OUTSIDE graph capture (first eager / profile call)."""
    import torch

    b = _bufs.get(device)
    if b is None:
        if _capturing():
            return None
        scratch = torch.empty((SCRATCH_FLOATS,), dtype=torch.float32, device=device)
        sync = torch.zeros((SYNC_INTS,), dtype=torch.int32, device=device)
        # one set per device: the GDN cores of a forward run serially on one stream (eager warm-up, the vLLM
        # graph-capture stream, and graph replays never overlap), so the self-resetting counters are never shared
        # by two concurrent launches. (The V2 runner warms every graph eagerly before capture -> allocation happens
        # outside capture.)
        b = (scratch, sync)
        _bufs[device] = b
    return b


def _dt_bias32(layer):
    import torch

    t = _dtb.get(id(layer))
    if t is None:
        if _capturing():
            return None
        t = layer.dt_bias.detach().to(torch.float32).contiguous()
        _dtb[id(layer)] = t
    return t


def _fast_params(layer, mixed_qkv, b, a, core_attn_out, kv_cache, log):
    """Launch parameters when this call is a pure spec-verify batch with prod shapes, else None (-> stock)."""
    import torch

    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as q
    from vllm.forward_context import get_forward_context

    md_raw = get_forward_context().attn_metadata
    if not isinstance(md_raw, dict):
        return None
    md = md_raw.get(layer.prefix)
    if not isinstance(md, q.GDNAttentionMetadata):
        return None
    if md.spec_sequence_masks is None or md.num_prefills != 0 or md.num_decodes != 0 or md.num_spec_decodes <= 0:
        return None
    if getattr(md, "ddtree_parent_ids", None) is not None:
        return None
    n_act = int(md.num_actual_tokens)
    if not (1 < n_act <= 16 or n_act == 20):  # stock mixed-QKV verifier admission
        return None
    if not getattr(layer, "enable_sm70_fused_sigmoid_mixed_qkv", False):
        return None
    if (layer.num_k_heads // layer.tp_size, layer.num_v_heads // layer.tp_size, layer.head_k_dim, layer.head_v_dim) != (
        4,
        12,
        128,
        128,
    ):
        return None
    if not (mixed_qkv.is_cuda and mixed_qkv.dtype == torch.float16 and mixed_qkv.dim() == 2):
        return None
    if mixed_qkv.shape[1] != 2560 or mixed_qkv.stride(1) != 1 or mixed_qkv.shape[0] < n_act:
        return None
    if not (mixed_qkv.is_contiguous() or (mixed_qkv.shape[0] in (5, 20) and mixed_qkv.stride(0) >= 2560)):
        return None
    conv_state = kv_cache[0] if q.is_conv_state_dim_first() else kv_cache[0].transpose(-1, -2)
    ssm = kv_cache[1]
    if conv_state.dtype != torch.float16 or conv_state.dim() != 3 or conv_state.shape[1] != 2560:
        return None
    if ssm.dtype != torch.float32 or ssm.dim() != 4 or tuple(ssm.shape[1:]) != (12, 128, 128):
        return None
    if ssm.stride(3) != 1 or ssm.stride(2) != 128 or ssm.stride(1) != 128 * 128:
        return None
    if layer._can_use_sm70_gdn_preprocess(mixed_qkv, a, b, conv_state, md):
        return None  # DFlash2 conv/gating/zero route
    mq, a_, b_ = mixed_qkv[:n_act], a[:n_act], b[:n_act]
    if layer._can_use_dflash2_packed_gdn_verify(
        mixed_qkv=mq, a=a_, b=b_, core_attn_out=core_attn_out, ssm_state=ssm, attn_metadata=md
    ):
        return None
    sidx = md.spec_state_indices_tensor
    sel = md.spec_state_slot_selectors if md.spec_state_slot_selectors is not None else md.num_accepted_tokens
    qsl = md.spec_query_start_loc
    N = int(md.num_spec_decodes)
    if sidx is None or sel is None or qsl is None:
        return None
    if sidx.dtype != torch.int32 or sidx.dim() != 2 or sidx.stride(1) != 1 or not (1 <= sidx.shape[1] <= 8):
        return None
    if sel.dtype != torch.int32 or qsl.dtype != torch.int32 or not (sel.is_contiguous() and qsl.is_contiguous()):
        return None
    if not (1 <= N <= 32) or sidx.shape[0] < N or qsl.numel() < N + 1 or sel.numel() < N:
        return None
    if conv_state.shape[2] < 3 + sidx.shape[1] - 1:
        return None
    if layer.activation not in ("silu", "swish"):
        return None
    w = layer.conv1d.weight
    if w.dtype != torch.float16 or w.dim() != 3 or w.shape[0] != 2560 or w.shape[2] != 4:
        return None
    conv_w = w.view(w.size(0), w.size(2))
    bias = layer.conv1d.bias
    if bias is not None and bias.dtype != torch.float16:
        return None
    if layer.A_log.dtype != torch.float32 or not layer.A_log.is_contiguous() or layer.A_log.numel() != 12:
        return None
    if a_.dtype != torch.float16 or b_.dtype != torch.float16 or a_.stride(1) != 1 or b_.stride(1) != 1:
        return None
    if not (
        core_attn_out.dtype == torch.float16
        and core_attn_out.is_contiguous()
        and core_attn_out.dim() == 3
        and tuple(core_attn_out.shape[1:]) == (12, 128)
        and core_attn_out.shape[0] >= n_act
    ):
        return None
    bufs = _buffers(mixed_qkv.device)
    dtb = _dt_bias32(layer)
    if bufs is None or dtb is None:
        _warn_once(log, "nobuf", "SB-P5 GDN: no persistent buffer at this call (first fast-path call was inside "
                   "graph capture) -> stock path for it")
        return None
    return dict(
        layer=layer, mq=mq, a=a_, b=b_, conv_state=conv_state, conv_w=conv_w, bias=bias, ssm=ssm, sidx=sidx,
        sel=sel, qsl=qsl, N=N, core=core_attn_out, n_act=n_act, scratch=bufs[0], sync=bufs[1], dtb=dtb,
        scale=layer.head_k_dim ** -0.5, md=md,
    )


def _launch(prm, out, z=None, norm_w=None, eps=0.0, rows=None):
    import torch

    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as q

    layer = prm["layer"]
    fuse_conv = sb_p5.flag("GDN_CONV", "1") == "1"
    bv = 128 if z is not None else int(sb_p5.flag("GDN_BV", "32"))
    x = prm["mq"]
    if not fuse_conv:
        # stock conv kernel (in place on x), then the fused delta rule reads the convolved rows
        x = q.causal_conv1d_update(
            x,
            prm["conv_state"],
            prm["conv_w"],
            prm["bias"],
            layer.activation,
            conv_state_indices=prm["sidx"][:, 0][: prm["N"]],
            num_accepted_tokens=prm["sel"],
            query_start_loc=prm["qsl"],
            max_query_len=prm["sidx"].size(-1),
            validate_data=False,
        )
    torch.ops.sb_p5_gdn.verify_fused(
        x, prm["a"], prm["b"], layer.A_log, prm["dtb"], prm["conv_state"], prm["conv_w"], prm["bias"], prm["ssm"],
        prm["sidx"], prm["sel"], prm["qsl"], out, z, norm_w, float(eps), prm["scratch"], prm["sync"], prm["N"],
        int(rows if rows is not None else prm["n_act"]), bv, fuse_conv, float(prm["scale"]),
    )


def _norm_ok(layer) -> bool:
    import torch

    import vllm.envs as envs

    n = getattr(layer, "norm", None)
    return bool(
        n is not None
        and getattr(n, "_sm70_rmsnorm_gated_exact", False)
        and not envs.VLLM_BATCH_INVARIANT
        and n.group_size is None
        and n.norm_before_gate
        and n.activation in ("silu", "swish")
        and n.weight.dtype == torch.float16
        and n.weight.shape == (128,)
        and n.weight.is_contiguous()
        and not getattr(layer, "enable_sm70_dflash2_fused_gdn_norm", False)
        and not getattr(layer, "enable_sm70_gdn_rmsnorm_onepass", False)
    )


def install(log) -> str:
    import vllm.envs as envs
    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as q

    C = getattr(q, _TARGET_CLS)
    if getattr(C._forward_core, "_sb_p5_orig", None) is not None:
        return "already installed"
    mode = sb_p5.flag("GDN_MODE", "fused")
    if mode not in ("trim", "fused"):
        return f"not installed: bad SB_P5_GDN_MODE={mode}"
    # forward_cuda's default branch must be the one this hook understands
    for v in (
        "VLLM_SM70_QWEN_GDN_CONTEXT_CORE",
        "VLLM_SM70_QWEN_GDN_INPUT_CORE_OP",
        "VLLM_SM70_QWEN_GDN_INPUT_PROJECTION_OP",
        "VLLM_SM70_QWEN_GDN_OUTPUT_PROJECTION_OP",
    ):
        if getattr(envs, v, False):
            return f"not installed: {v}=1 (route not covered)"
    _state["mode"] = mode
    orig_core = C._forward_core

    if mode == "trim":
        orig_mq = q.fused_sigmoid_gating_delta_rule_update_mixed_qkv

        def sb_mixed_qkv(*args, **kw):
            tgt = getattr(_tls, "trim_out", None)
            if tgt is not None and kw.get("out") is None and "mixed_qkv" in kw:
                T = kw["mixed_qkv"].shape[0]
                cand = tgt[:T].unsqueeze(0)
                if (
                    tuple(cand.shape) == (1, T, kw.get("num_v_heads"), kw.get("head_v_dim"))
                    and cand.is_contiguous()
                    and cand.dtype == kw["mixed_qkv"].dtype
                ):
                    kw["out"] = cand
                    if not _state["announced"]:
                        _state["announced"] = True
                        log.info("SB-P5 GDN trim active: verify delta writes core_attn_out directly, first M=%d", T)
            return orig_mq(*args, **kw)

        def sb_core_trim(self, mixed_qkv, b, a, core_attn_out, kv_cache):
            from vllm.forward_context import get_forward_context

            md_raw = get_forward_context().attn_metadata
            md = md_raw.get(self.prefix) if isinstance(md_raw, dict) else None
            pure = bool(
                isinstance(md, q.GDNAttentionMetadata)
                and md.spec_sequence_masks is not None
                and md.num_prefills == 0
                and md.num_decodes == 0
                and core_attn_out.is_contiguous()
            )
            _tls.trim_out = core_attn_out if pure else None
            try:
                return orig_core(self, mixed_qkv, b, a, core_attn_out, kv_cache)
            finally:
                _tls.trim_out = None

        sb_mixed_qkv._sb_p5_orig = orig_mq
        sb_core_trim._sb_p5_orig = orig_core
        q.fused_sigmoid_gating_delta_rule_update_mixed_qkv = sb_mixed_qkv
        C._forward_core = sb_core_trim
        _orig.update(core=orig_core, mq=orig_mq)
        return "trim (exact: verify memcpy removed)"

    sb_p5.load_lib("sb_p5_gdn_C", "sb_p5_gdn", "verify_fused")
    want_norm = sb_p5.flag("GDN_NORM", "0") == "1"
    orig_fwd = C.forward_cuda
    orig_cop = C._compute_output_projection

    def sb_core(self, mixed_qkv, b, a, core_attn_out, kv_cache):
        prm = _fast_params(self, mixed_qkv, b, a, core_attn_out, kv_cache, log)
        if prm is None:
            return orig_core(self, mixed_qkv, b, a, core_attn_out, kv_cache)
        _state["hits"] += 1
        if not _state["announced"]:
            _state["announced"] = True
            log.info(
                "SB-P5 GDN fused active: first M=%d N=%d S=%d (conv=%s bv=%s norm=%s)",
                prm["n_act"], prm["N"], prm["sidx"].shape[1], sb_p5.flag("GDN_CONV", "1"),
                "128" if want_norm else sb_p5.flag("GDN_BV", "32"), want_norm,
            )
        if want_norm and not _state["norm_broken"] and getattr(_tls, "armed", None) == id(self) and _norm_ok(self):
            _pending[id(self)] = prm  # launched by sb_cop with z (same eager call chain, before any reader)
            return None
        _launch(prm, out=core_attn_out)
        return None

    def sb_cop(self, core_attn_out, z, num_tokens, output=None):
        import torch

        prm = _pending.pop(id(self), None)
        if prm is None:
            return orig_cop(self, core_attn_out, z, num_tokens, output)
        ok = bool(
            output is None
            and core_attn_out.data_ptr() == prm["core"].data_ptr()
            and z.dtype == torch.float16
            and z.dim() == 3
            and tuple(z.shape[1:]) == (12, 128)
            and z.stride(2) == 1
            and z.shape[0] == num_tokens == core_attn_out.shape[0]
        )
        if not ok:
            _launch(prm, out=prm["core"])  # fallback: core now, stock norm
            return orig_cop(self, core_attn_out, z, num_tokens, output)
        normed = torch.empty((num_tokens, 12, 128), dtype=torch.float16, device=core_attn_out.device)
        _launch(prm, out=normed, z=z, norm_w=self.norm.weight, eps=self.norm.eps, rows=num_tokens)
        proj_out, _ = self.out_proj(normed.reshape(num_tokens, -1))
        return proj_out

    def sb_fwd(self, hidden_states, output):
        prev = getattr(_tls, "armed", None)
        _tls.armed = id(self)
        try:
            return orig_fwd(self, hidden_states, output)
        finally:
            _tls.armed = prev
            prm = _pending.pop(id(self), None)
            if prm is not None:  # deferred core never consumed: never leave the state un-updated
                _state["norm_broken"] = True
                log.error("SB-P5 GDN: deferred core not consumed in %s -> launched late, NORM fusion disabled",
                          self.prefix)
                _launch(prm, out=prm["core"])

    sb_core._sb_p5_orig = orig_core
    C._forward_core = sb_core
    _orig.update(core=orig_core)
    if want_norm:
        sb_cop._sb_p5_orig = orig_cop
        sb_fwd._sb_p5_orig = orig_fwd
        C._compute_output_projection = sb_cop
        C.forward_cuda = sb_fwd  # bound into _forward_method at layer __init__ -> install before model build
        _orig.update(cop=orig_cop, fwd=orig_fwd)
    return "fused (conv=%s bv=%s norm=%s)" % (
        sb_p5.flag("GDN_CONV", "1"), "128" if want_norm else sb_p5.flag("GDN_BV", "32"), want_norm)


def uninstall() -> None:
    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as q

    C = getattr(q, _TARGET_CLS)
    for attr, key in (("_forward_core", "core"), ("_compute_output_projection", "cop"), ("forward_cuda", "fwd")):
        fn = getattr(C, attr, None)
        if fn is not None and getattr(fn, "_sb_p5_orig", None) is not None:
            setattr(C, attr, fn._sb_p5_orig)
    fn = q.fused_sigmoid_gating_delta_rule_update_mixed_qkv
    if getattr(fn, "_sb_p5_orig", None) is not None:
        q.fused_sigmoid_gating_delta_rule_update_mixed_qkv = fn._sb_p5_orig
    _orig.clear()
    _pending.clear()
    _state.update(announced=False, norm_broken=False, hits=0, mode=None)
    _state["warned"] = set()
