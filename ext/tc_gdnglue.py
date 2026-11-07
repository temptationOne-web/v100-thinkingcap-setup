"""ThinkingCap GDN verify glue fusion (2026-10-26, port of Ornith ports/gdnglue to Qwen4Exp / Flash-Next). README:
/opt/1cat/tcflash-trial/ext/GDNGLUE-README.md.

Enable:  TC_GDNGLUE=1  (default off; install() is a no-op otherwise). Nothing here is a VLLM_* env.
Knobs:   TC_GDNGLUE_MAXREQ=64   largest number of verify requests per launch (counter buffer size)
         TC_GDNGLUE_BV=<n>      override the recurrence V block (default: the stock sigmoid schedule, 32). Changing it
                                is only exact if the GPU check says so (tests/tc_gdnglue_gpu_check.py --bv n).
         TC_GDNGLUE_LOG=1       log every fallback reason once
Hook:    QwenGatedDeltaNetAttention.forward_cuda (runs EAGERLY inside the opaque vllm::qwen_gdn_full_forward op, and is
         captured by the FULL decode CUDA graphs). Nothing is traced by torch.compile -> no compile-cache change.

What it does on an admitted call (an MTP target-verify batch: only spec rows, no prefill / plain decode):
  stock forward_cuda Part 1 (projection) is run unchanged (copied below, every branch), then ONE Triton launch
  (tc_gdnglue_kernels.tc_gdnglue_verify_kernel) replaces zero fill + causal_conv1d_update + fused sigmoid-gating
  delta-rule recurrence + rmsnorm_gated_exact, then the stock out_proj / output write. Everything else -> the
  original forward_cuda, untouched.
Differences vs stock that are never read: mixed_qkv keeps the pre-conv values (stock overwrites it in place with the
conv output; nothing reads it after the core), and when the stock verify takes its non-direct output path
(num_actual_tokens not 5/20) the rows of PAD requests are 0 here instead of uninitialized memory.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import logging
import os
import sys

TARGET = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
log = logging.getLogger("tc_gdnglue")

_stats = {
    "installed": False, "fused_calls": 0, "fused_capture_calls": 0, "fallbacks": 0, "fallback_reasons": {},
    "errors": 0, "last_error": "",
}
_STATIC = {}      # id(layer) -> (ok, reason)
_CNT = {}         # device index -> int32 counters
_ORIG = {}


def _say(msg):
    print(f"TC-GDNGLUE pid {os.getpid()}: {msg}", flush=True)


def enabled() -> bool:
    return os.environ.get("TC_GDNGLUE", "0") == "1"


def _env_int(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _fb(reason: str) -> None:
    _stats["fallbacks"] += 1
    d = _stats["fallback_reasons"]
    first = reason not in d
    d[reason] = d.get(reason, 0) + 1
    if first and os.environ.get("TC_GDNGLUE_LOG", "0") == "1":
        _say(f"fallback: {reason}")


def status() -> dict:
    out = dict(_stats)
    out["fallback_reasons"] = dict(_stats["fallback_reasons"])
    out["layers_static_ok"] = sum(1 for ok, _ in _STATIC.values() if ok)
    out["layers_static_rejected"] = {r: sum(1 for ok, rr in _STATIC.values() if not ok and rr == r)
                                     for r in {rr for ok, rr in _STATIC.values() if not ok}}
    return out


# ----------------------------------------------------------------------------------------------------------------------
# admission
# ----------------------------------------------------------------------------------------------------------------------
def _static_check(self):
    """Layer-level conditions under which stock forward_cuda takes, for an active-spec batch, the route
    core_standard_spec -> _forward_core -> convolve_decode(spec) -> fused_sigmoid_gating_delta_rule_update_mixed_qkv
    (Triton) -> RMSNormGated.forward_native -> sm70_rmsnorm_gated_exact."""
    import torch

    gm = sys.modules[TARGET]
    from vllm import envs
    from vllm.model_executor.layers.layernorm import RMSNormGated

    pol = self.gdn_policy.projection
    checks = (
        ("tp_heads", self.num_k_heads // self.tp_size == 4 and self.num_v_heads // self.tp_size == 12),
        ("head_dims", self.head_k_dim == 128 and self.head_v_dim == 128),
        ("gqa_interleaved", not self.gqa_interleaved_layout),
        ("conv_bias", self.conv1d.bias is None),
        ("conv_act", self.activation in ("silu", "swish")),
        ("conv_weight", tuple(self.conv1d.weight.shape) == (2560, 1, 4)
         and self.conv1d.weight.dtype == torch.float16),
        ("spec_replay", not self._spec_replay),
        ("native_verify", self.verification_update is None),
        ("dflash2_verify", not self.enable_sm70_dflash2_fused_gdn_verify),
        ("dflash2_norm", not self.enable_sm70_dflash2_fused_gdn_norm),
        ("mixed_qkv_decode", bool(self.enable_sm70_fused_sigmoid_mixed_qkv)),
        ("spec_core_003", not getattr(self, "auto_sm70_qwen_gdn_003_spec_core", False)),
        ("spec_core", not getattr(self, "auto_sm70_qwen_gdn_spec_core", False)),
        ("full_forward_auto", bool(getattr(self, "auto_sm70_qwen_gdn_full_forward", False))),
        ("context_core", not pol.context_core),
        ("input_core", not gm._sm70_qwen_gdn_input_core_boundary_enabled(pol)),
        ("input_projection", not pol.input_projection),
        ("norm_exact", bool(getattr(self.norm, "_sm70_rmsnorm_gated_exact", False))
         and not envs.VLLM_BATCH_INVARIANT),
        ("norm_shape", self.norm.group_size is None and self.norm.norm_before_gate
         and self.norm.activation in ("sigmoid", "silu", "swish")
         and tuple(self.norm.weight.shape) == (128,) and self.norm.weight.dtype == torch.float16),
        ("norm_dispatch", getattr(getattr(self.norm, "_forward_method", None), "__func__", None)
         is RMSNormGated.forward_native),
    )
    for name, ok in checks:
        if not ok:
            return False, name
    return True, ""


def _counters(device):
    import torch

    key = device.index
    t = _CNT.get(key)
    if t is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        maxreq = _env_int("TC_GDNGLUE_MAXREQ", 64)
        t = torch.zeros(maxreq * (4 + 12), dtype=torch.int32, device=device)
        _CNT[key] = t
    return t


# ----------------------------------------------------------------------------------------------------------------------
# launcher
# ----------------------------------------------------------------------------------------------------------------------
def launch_fused(*, mixed_qkv, a, b, z, core_out, nout, conv_state, conv_w2d, conv_idx, ssm_state, cu_seqlens,
                 ssm_state_indices, num_accepted_tokens, A_log, dt_bias, norm_weight, eps, silu_norm, num_reqs,
                 num_actual_tokens, tmax, counters, BV, num_warps, num_stages, use_asm=True, scale=None):
    """Launch tc_gdnglue_verify_kernel. Shapes/strides follow the stock calls (see tests for a CPU example).
    mixed_qkv [T, 2560] fp16 (stride(1) == 1, not modified); a, b [T, 12] contiguous; z [M, 12, 128] contiguous;
    core_out / nout [M, 12, 128] contiguous (written); conv_state = the (lines, dim, state_len) VIEW stock passes to
    the conv kernel; conv_w2d [2560, 4]; conv_idx = spec_state_indices[:, 0] (any 1-D stride); ssm_state [lines, 12,
    128, 128] fp32; cu_seqlens [N+1]; ssm_state_indices [N, tmax]; num_accepted_tokens [N] (spec_state_slot_selectors)."""
    import torch
    import triton

    from tc_gdnglue_kernels import tc_gdnglue_verify_kernel

    H, HV, K, V = 4, 12, 128, 128
    T = num_actual_tokens
    M = core_out.shape[0]
    N = num_reqs
    NV = V // BV
    SW = 2 * K + BV
    scr = torch.empty((N * HV * NV, tmax, SW), dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    width = conv_w2d.shape[1]
    state_len = width - 1 + (tmax - 1)
    if scale is None:
        scale = K ** -0.5
    if ssm_state_indices.ndim == 1:
        s_seq, s_tok = ssm_state_indices.stride(0), 1
    else:
        s_seq, s_tok = ssm_state_indices.stride()
    tc_gdnglue_verify_kernel[(1, NV, N * HV)](
        mixed_qkv, conv_w2d, conv_state, conv_idx,
        A_log, a, b, dt_bias, 1.0, 20.0,
        scr, core_out, ssm_state, ssm_state, cu_seqlens, ssm_state_indices, num_accepted_tokens, scale,
        z, norm_weight, nout, float(eps),
        counters,
        N, T, M, conv_state.size(0),
        mixed_qkv.stride(0),
        B=1, H=H, HV=HV, K=K, V=V, BK=triton.next_power_of_2(K), BV=BV,
        Q_OFFSET=0, K_OFFSET=H * K, V_OFFSET=2 * H * K, DIM=2 * H * K + HV * V,
        SW=SW, TMAX=tmax,
        stride_init_state_token=ssm_state.stride(0), stride_final_state_token=ssm_state.stride(0),
        stride_indices_seq=s_seq, stride_indices_tok=s_tok,
        stride_ci=conv_idx.stride(0),
        stride_cs_seq=conv_state.stride(0), stride_cs_dim=conv_state.stride(1), stride_cs_tok=conv_state.stride(2),
        stride_w_dim=conv_w2d.stride(0), stride_w_width=conv_w2d.stride(1),
        stride_z_tok=z.stride(0),
        STATE_LEN=state_len, NP2_STATELEN=triton.next_power_of_2(state_len),
        CNT_NORM_BASE=(counters.numel() // 16) * 4,
        SILU_NORM=bool(silu_norm), USE_ASM=bool(use_asm),
        num_warps=num_warps, num_stages=num_stages,
    )
    return nout


def _schedule(self, device, N):
    from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
        _select_fused_sigmoid_schedule,
        _use_sm70_fused_sigmoid_schedule,
    )

    sched = self.gdn_policy.schedule
    if _use_sm70_fused_sigmoid_schedule(device, schedule=sched):
        BV, nw, ns = _select_fused_sigmoid_schedule(128, N, 12, device, schedule=sched)
    else:
        BV, nw, ns = 32, 4, 3
    bv = _env_int("TC_GDNGLUE_BV", 0)
    if bv in (16, 32, 64, 128):
        BV = bv
    return BV, nw, ns


# ----------------------------------------------------------------------------------------------------------------------
# patched forward_cuda
# ----------------------------------------------------------------------------------------------------------------------
def _projection_part1(self, gm, hidden_states, layer_name, num_tokens):
    """Verbatim copy of the stock QwenGatedDeltaNetAttention.forward_cuda Part 1 (engine 6a143197e), all branches."""
    import torch

    gm._sm70_gdn_graph_buffer_copy("gdn_hidden_states", layer_name, hidden_states, "proj")
    hidden_states = gm._sm70_dump_gdn_projection_tensor("gdn_hidden_states", layer_name, hidden_states)
    use_qwen38_fused_input = bool(
        getattr(self, "sm70_qwen38_fp16_fused_input", False)
        and isinstance(self.in_proj_qkvz, gm.LinearBase)
        and isinstance(self.in_proj_ba, gm.LinearBase)
        and gm.use_sm70_decode_graph_semantics(self._execution_graph_policy)
        and not gm._sm70_gdn_projection_dump_requested(layer_name)
    )
    from vllm.model_executor.layers.quantization.sm70_gdn_ba_verify import apply_gdn_ba_verify

    fused_verify_projection = apply_gdn_ba_verify(self, hidden_states)
    if fused_verify_projection is not None:
        mixed_qkv, z, b, a = fused_verify_projection
        z = z.reshape(z.size(0), -1, self.head_v_dim)
    elif use_qwen38_fused_input:
        assert self.in_proj_ba is not None
        mixed_qkv, z, b, a = torch.ops.vllm.qwen38_sm70_fp16_gdn_input(
            hidden_states,
            self.in_proj_qkvz.weight,
            self.in_proj_ba.weight,
            getattr(self.in_proj_qkvz, "_sm70_qwen38_gdn_packed", None),
            getattr(self.in_proj_ba, "_sm70_qwen38_gdn_packed", None),
        )
        z = z.reshape(z.size(0), -1, self.head_v_dim)
    else:
        side = getattr(self, "input_projection_override", None)
        if side is not None and 1 <= hidden_states.shape[0] <= 8:
            mixed_qkvz, ba = side(hidden_states)
        else:
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
        mixed_qkvz = gm._sm70_dump_gdn_projection_tensor("in_proj_qkvz", layer_name, mixed_qkvz)
        ba = gm._sm70_dump_gdn_projection_tensor("in_proj_ba", layer_name, ba)

        from vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv import (
            _can_fuse_gdn_projection_split,
            _split_gdn_projection_tails,
        )

        if (
            not self.gqa_interleaved_layout
            and not self.disable_tp_for_ba_proj
            and mixed_qkvz.shape[0] in (5, 20)
            and (self.key_dim * 2 + self.value_dim) // self.tp_size == 2560
            and self.value_dim // self.tp_size == 1536
            and _can_fuse_gdn_projection_split(mixed_qkvz, ba)
        ):
            z = mixed_qkvz.new_empty((num_tokens, 12, 128))
            mixed_qkv, b, a = _split_gdn_projection_tails(mixed_qkvz, ba, z)
            if self.gdn_policy.projection.mixed_qkv_contiguous:
                mixed_qkv = mixed_qkv.contiguous()
            gm._log_runtime_route_once("SM70 default GDN projection tail-copy route hit.")
        elif self.gqa_interleaved_layout:
            from einops import rearrange

            query, key, value, z, b, a = self.fix_query_key_value_ordering(mixed_qkvz, ba)
            query, key, value = map(lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value))
            mixed_qkv = torch.cat((query, key, value), dim=-1)
        else:
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv = mixed_qkvz[..., :qkv_size]
            mixed_qkv = gm._sm70_dump_gdn_projection_tensor("split_mixed_qkv", layer_name, mixed_qkv)
            if self.gdn_policy.projection.mixed_qkv_contiguous:
                mixed_qkv = mixed_qkv.contiguous()
            z = gm._sm70_compile_graph_slice_dim(mixed_qkvz, -1, qkv_size, z_size)
            z = gm._sm70_dump_gdn_projection_tensor("split_z", layer_name, z)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            ba_size = ba.shape[-1] // 2
            b = ba[..., :ba_size]
            a = gm._sm70_compile_graph_slice_dim(ba, -1, ba_size, ba_size)
            if self.disable_tp_for_ba_proj and self.tp_size > 1:
                ba_chunk = self.num_v_heads // self.tp_size
                ba_start = self.tp_rank * ba_chunk
                b = b[:, ba_start : ba_start + ba_chunk]
                a = a[:, ba_start : ba_start + ba_chunk]
            b = b.contiguous()
            a = a.contiguous()

    if self.gdn_policy.projection.z_contiguous and gm.current_platform.is_device_capability(70):
        z = z.contiguous()
    return mixed_qkv, z, b, a


def _stock_tail(self, gm, mixed_qkv, z, b, a, hidden_states, output, layer_name, num_tokens):
    """Verbatim copy of stock forward_cuda Part 2 + Part 3 (used when the dynamic admission fails after Part 1)."""
    import torch

    core_attn_out = torch.zeros(
        (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    conv_state_cache, ssm_state_cache = gm._resolve_qwen_gdn_kv_cache_args(layer_name, core_attn_out)
    core_attn_out = gm._qwen_gdn_run_recurrent_core(
        self,
        mixed_qkv=mixed_qkv,
        b=b,
        a=a,
        core_attn_out=core_attn_out,
        layer_name=layer_name,
        conv_state_cache=conv_state_cache,
        ssm_state_cache=ssm_state_cache,
    )
    if self.gdn_policy.projection.output_projection:
        torch.ops.vllm.qwen_gdn_output_projection(core_attn_out, z, output, num_tokens, layer_name)
    else:
        return self._output_projection(core_attn_out, z, output, num_tokens)
    return None


def _project_out(self, normed, z, output, num_tokens):
    """Stock _output_projection / _compute_output_projection after the norm (direct_write is never admitted)."""
    core = normed.reshape(z.shape).flatten(-2)
    projection_override = getattr(self, "output_projection_override", None)
    if projection_override is not None:
        proj_out = projection_override(core)
    else:
        proj_out, _ = self.out_proj(core)
    if output is not None:
        output[:num_tokens] = proj_out
    return proj_out


def _dynamic_check(self, gm, md, mixed_qkv, a, b, z, conv_state, ssm_state, num_tokens):
    """The stock _forward_core conditions for convolve_decode(spec) + use_sm70_mixed_qkv_verify (Triton verifier)."""
    import torch

    if md.spec_sequence_masks is None or md.num_spec_decodes <= 0:
        return "no_active_spec"
    if md.num_prefills != 0 or md.num_decodes != 0:
        return "mixed_batch"
    if getattr(md, "ddtree_parent_ids", None) is not None:
        return "ddtree"
    if md.spec_query_start_loc is None or md.spec_state_indices_tensor is None:
        return "spec_metadata_missing"
    n = int(md.num_actual_tokens)
    mq = mixed_qkv[:n]
    if not (mq.is_cuda and mq.dtype == torch.float16 and mq.ndim == 2 and mq.shape[1] == 2560):
        return "mixed_qkv_type"
    full = mixed_qkv            # mixed_qkv_decode_requested is evaluated on the unsliced tensor
    requested = full.is_contiguous() or (
        full.shape[0] in (5, 20) and full.stride(1) == 1 and full.stride(0) >= full.shape[1])
    verify = mq.is_contiguous() or (mq.shape[0] in (5, 20) and mq.stride(1) == 1 and mq.stride(0) >= mq.shape[1])
    if not (requested and verify and mq.stride(1) == 1):
        return "mixed_qkv_layout"
    if not (1 < n <= 16 or n == 20):
        return "rows_not_mixed_qkv_verify"
    if ssm_state.dtype != torch.float32 or conv_state.dtype != torch.float16:
        return "state_dtype"
    if not (a.dtype == b.dtype == torch.float16 and a.shape[-1] == 12 and b.shape[-1] == 12):
        return "ab_type"
    if num_tokens > 64:
        return "norm_rows_gt64"
    if not (z.is_contiguous() and z.dtype == torch.float16 and tuple(z.shape[1:]) == (12, 128)
            and z.shape[0] == num_tokens):
        return "z_layout"
    idx = md.spec_state_indices_tensor
    if idx.ndim != 2 or idx.size(-1) > 16 or idx.dtype not in (torch.int32, torch.int64):
        return "state_indices"
    if md.num_spec_decodes > _env_int("TC_GDNGLUE_MAXREQ", 64):
        return "too_many_requests"
    return ""


def _tc_forward_cuda(self, hidden_states, output):
    import torch

    gm = sys.modules[TARGET]
    orig = _ORIG["forward_cuda"]
    st = _STATIC.get(id(self))
    if st is None:
        try:
            st = _static_check(self)
        except Exception as exc:  # noqa: BLE001
            st = (False, f"static_check_error:{type(exc).__name__}")
        _STATIC[id(self)] = st
    if not st[0]:
        return orig(self, hidden_states, output)
    if not torch.cuda.is_current_stream_capturing():
        _counters(hidden_states.device)                 # allocate outside graph capture (first eager call)
    layer_name = gm._encode_layer_name(self.prefix)
    if not gm._sm70_qwen_gdn_has_active_spec_decode(layer_name):
        return orig(self, hidden_states, output)        # prefill / plain decode / warmup: stock, not counted
    num_tokens = hidden_states.size(0)
    try:
        mixed_qkv, z, b, a = _projection_part1(self, gm, hidden_states, layer_name, num_tokens)
    except Exception as exc:  # noqa: BLE001
        _stats["errors"] += 1
        _stats["last_error"] = f"part1: {exc!r}"
        raise
    fwd = gm.get_forward_context()
    md = fwd.attn_metadata.get(self.prefix) if isinstance(fwd.attn_metadata, dict) else None
    conv_cache, ssm_cache = gm._resolve_qwen_gdn_kv_cache_args(layer_name, z)
    reason = ""
    if not isinstance(md, gm.GDNAttentionMetadata):
        reason = "no_gdn_metadata"
    elif conv_cache.numel() == 0 or ssm_cache.numel() == 0:
        reason = "no_kv_cache"
    counters = None
    if not reason:
        counters = _counters(z.device)
        if counters is None:
            reason = "counters_unallocated_in_capture"
    if reason:
        _fb(reason)
        return _stock_tail(self, gm, mixed_qkv, z, b, a, hidden_states, output, layer_name, num_tokens)

    tensors = gm._qwen_gdn_metadata_tensors(layer_name, z.device)
    with gm.GdnMetadataOverride(md, skip_empty=True) as view:
        names = ("non_spec_query_start_loc", "non_spec_state_indices_tensor", "spec_query_start_loc",
                 "spec_state_indices_tensor", "spec_token_indx", "non_spec_token_indx", "spec_sequence_masks",
                 "num_accepted_tokens", "spec_state_slot_selectors")
        for name, value in zip(names, tensors):
            if name == "spec_state_slot_selectors" and value.numel() == 0:
                value = tensors[7]
            view.set(name, value)
        conv_state = conv_cache if gm.is_conv_state_dim_first() else conv_cache.transpose(-1, -2)
        reason = _dynamic_check(self, gm, md, mixed_qkv, a, b, z, conv_state, ssm_cache, num_tokens)
        if reason:
            _fb(reason)
        else:
            N = int(md.num_spec_decodes)
            n = int(md.num_actual_tokens)
            selectors = (md.spec_state_slot_selectors if md.spec_state_slot_selectors is not None
                         else md.num_accepted_tokens)
            idx = md.spec_state_indices_tensor
            BV, nw, ns = _schedule(self, z.device, N)
            core_out = torch.empty((num_tokens, 12, 128), dtype=hidden_states.dtype, device=hidden_states.device)
            nout = torch.empty_like(core_out)
            conv_w2d = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            launch_fused(
                mixed_qkv=mixed_qkv[:n], a=a[:n].contiguous(), b=b[:n].contiguous(), z=z, core_out=core_out,
                nout=nout, conv_state=conv_state, conv_w2d=conv_w2d, conv_idx=idx[:N, 0], ssm_state=ssm_cache,
                cu_seqlens=md.spec_query_start_loc[: N + 1], ssm_state_indices=idx, num_accepted_tokens=selectors,
                A_log=self.A_log, dt_bias=self.dt_bias, norm_weight=self.norm.weight, eps=self.norm.eps,
                silu_norm=self.norm.activation != "sigmoid", num_reqs=N, num_actual_tokens=n,
                tmax=idx.size(-1), counters=counters, BV=BV, num_warps=nw, num_stages=ns,
                scale=self.head_k_dim ** -0.5,
            )
            _stats["fused_calls"] += 1
            if torch.cuda.is_current_stream_capturing():
                _stats["fused_capture_calls"] += 1
            if _stats["fused_calls"] == 1:
                _say(f"fused GDN verify active (BV={BV} warps={nw} stages={ns} N={N} rows={n}/{num_tokens})")
    if reason:
        return _stock_tail(self, gm, mixed_qkv, z, b, a, hidden_states, output, layer_name, num_tokens)
    proj_out = _project_out(self, nout, z, output, num_tokens)
    return None if self.gdn_policy.projection.output_projection else proj_out


# ----------------------------------------------------------------------------------------------------------------------
# install
# ----------------------------------------------------------------------------------------------------------------------
def _install_now():
    if _stats["installed"]:
        return True
    gm = sys.modules[TARGET]
    cls = gm.QwenGatedDeltaNetAttention
    if getattr(cls.forward_cuda, "_tc_gdnglue", False):
        _stats["installed"] = True
        return True
    _ORIG["forward_cuda"] = cls.forward_cuda
    _tc_forward_cuda._tc_gdnglue = True
    cls.forward_cuda = _tc_forward_cuda
    # QwenGatedDeltaNetAttention.__init__ binds self._forward_method = self.forward_cuda, so every layer built after
    # this point (the model is built after the worker extension is imported) dispatches here.
    _stats["installed"] = True
    _say("installed (QwenGatedDeltaNetAttention.forward_cuda patched)")
    return True


class _AfterImport(importlib.abc.MetaPathFinder):
    def __init__(self, name, fn):
        self.name, self.fn, self.done = name, fn, False

    def find_spec(self, fullname, path, target=None):
        if fullname != self.name or self.done:
            return None
        self.done = True
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            try:
                self.fn()
            except Exception as exc:  # noqa: BLE001
                _say(f"post-import install failed: {exc!r}")

        loader.exec_module = exec_module
        return spec


def install() -> bool:
    """Idempotent, never raises. Patches now if the GDN module is imported, else arms a post-import hook."""
    if not enabled():
        return False
    try:
        if TARGET in sys.modules:
            return _install_now()
        sys.meta_path.insert(0, _AfterImport(TARGET, _install_now))
        _say("armed (post-import hook)")
        return True
    except Exception as exc:  # noqa: BLE001
        _say(f"install failed: {exc!r}")
        return False


install()


def _base_ext():
    import importlib

    name = os.environ.get("TC_GDNGLUE_BASE_EXT", "tc_ext.TCExt") or "tc_ext.TCExt"
    if name.startswith("tc_gdnglue."):
        name = "tc_ext.TCExt"
    mod, _, cls = name.rpartition(".")
    return getattr(importlib.import_module(mod), cls)


_EXT_CLS = None


def __getattr__(name):
    """TCGdnGlueExt is built on first access (PEP 562), i.e. when vLLM resolves --worker-extension-cls
    tc_gdnglue.TCGdnGlueExt, so tc_ext.py may also `import tc_gdnglue` (enable route B) without a circular import.
    Base class = $TC_GDNGLUE_BASE_EXT (default tc_ext.TCExt)."""
    global _EXT_CLS
    if name != "TCGdnGlueExt":
        raise AttributeError(name)
    if _EXT_CLS is None:
        try:
            base = _base_ext()
        except Exception as exc:  # noqa: BLE001  (CPU tests run without the prod packages on sys.path)
            _say(f"base worker extension unavailable ({exc!r}); TCGdnGlueExt derives from object")
            base = object

        def tc_gdnglue_status(self):
            return status()

        _EXT_CLS = type("TCGdnGlueExt", (base,), {"tc_gdnglue_status": tc_gdnglue_status, "__module__": __name__})
    return _EXT_CLS
