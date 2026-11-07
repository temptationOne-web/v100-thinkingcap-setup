"""Triton kernel for tc_gdnglue (ThinkingCap GDN verify glue fusion). See GDNGLUE-README.md.

ONE launch per GDN layer per MTP verify round replaces four:
    FillFunctor (torch.zeros core_attn_out)            -> every row is written by the kernel (zeros where stock has 0)
    _causal_conv1d_update_kernel (spec, in place)      -> conv prologue per program, outputs to a private scratch
    fused_sigmoid_gating_delta_rule_update_kernel      -> body copied verbatim (route constexprs folded)
    rmsnorm_gated_exact_kernel<Silu> (CUDA, vllm._C)   -> epilogue run by the last of the NV programs of a (request, head)

Bit-exactness strategy
  * conv + recurrence: the stock Triton expressions are copied token for token -> same TTIR/LLVM -> same rounding and
    FMA decisions. The recurrence reads q/k/v from the scratch through pointers with the same contiguity/divisibility
    class as the stock mixed_qkv pointers (tests/gdnglue_offline_check.py diffs the recurrence TTGIR + SASS).
  * gated RMSNorm: inline PTX copied from nvcc 12.8 -O3 output of csrc/sm70_turbomind/ops/rmsnorm_gated_exact_sm70.cu
    (mul.rn/add.rn, rsqrt.approx.f32, the CUDA expf expansion, rcp.rn / div.rn) with the kernel's reduction tree
    (lane L sums cols 4L..4L+3 left to right from 0.0f, then shfl_down 16,8,4,2,1, lane 0 broadcast).
  * conv state: the v-channel slice of a program is read only by that program -> it commits that slice itself; the
    q/k slice of a k-head is read by (HV/H)*NV programs -> the last of them (atomic counter, self-resetting) commits it.
    Every commit is the stock STEP 1/2 data movement (no arithmetic) from the unmodified mixed_qkv.
USE_ASM=False (CPU interpreter tests only) swaps the PTX for plain float32 ops with the same operation order.
"""
from __future__ import annotations

import triton
import triton.language as tl

PAD_SLOT_ID = -1


# ----------------------------------------------------------------------------------------------------------------------
# float32 helpers: explicit rounding (never contracted into FMA) on GPU, plain numpy-float32 ops in the interpreter
# ----------------------------------------------------------------------------------------------------------------------
@triton.jit
def _mul_rn(a, b, USE_ASM: tl.constexpr):
    if USE_ASM:
        return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32,
                                         is_pure=True, pack=1)
    else:
        return a * b


@triton.jit
def _add_rn(a, b, USE_ASM: tl.constexpr):
    if USE_ASM:
        return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32,
                                         is_pure=True, pack=1)
    else:
        return a + b


@triton.jit
def _rsqrt_cuda(x, USE_ASM: tl.constexpr):
    # rsqrtf() without fast-math = rsqrt.approx.f32 (MUFU.RSQ + non-ftz denormal fix-up)
    if USE_ASM:
        return tl.inline_asm_elementwise("rsqrt.approx.f32 $0, $1;", "=f,f", [x], dtype=tl.float32,
                                         is_pure=True, pack=1)
    else:
        return 1.0 / tl.sqrt(x)


# nvcc 12.8 -O3 sm_70 PTX of `__fadd_rn(1.0f, expf(-g))` followed by `1.0f / d` (Silu=false: rcp.rn) or `g / d`
# (Silu=true: div.rn), copied from the compiled rmsnorm_gated_exact kernel. The two plain mul.f32/add.f32 of that PTX
# are written with an explicit .rn (the reference SASS has them unfused: FMUL / FADD), so ptxas cannot contract them.
_GATE_SIGMOID_PTX = """{
.reg .f32 gn, gt, gj, gk, gr, gs, ge, gd;
.reg .b32 gji;
neg.f32 gn, $1;
fma.rn.f32 gt, gn, 0f3BBB989D, 0f3F000000;
cvt.sat.f32.f32 gt, gt;
fma.rm.f32 gj, gt, 0f437C0000, 0f4B400001;
add.rn.f32 gk, gj, 0fCB40007F;
neg.f32 gk, gk;
fma.rn.f32 gr, gn, 0f3FB8AA3B, gk;
fma.rn.f32 gr, gn, 0f32A57060, gr;
mov.b32 gji, gj;
shl.b32 gji, gji, 23;
mov.b32 gs, gji;
ex2.approx.ftz.f32 ge, gr;
mul.rn.f32 ge, ge, gs;
add.rn.f32 gd, 0f3F800000, ge;
rcp.rn.f32 $0, gd;
}"""
_GATE_SILU_PTX = tl.constexpr(_GATE_SIGMOID_PTX.replace("rcp.rn.f32 $0, gd;", "div.rn.f32 $0, $1, gd;"))
_GATE_SIGMOID_PTX = tl.constexpr(_GATE_SIGMOID_PTX)


@triton.jit
def _gate_cuda(g, SILU: tl.constexpr, USE_ASM: tl.constexpr):
    if USE_ASM:
        if SILU:
            return tl.inline_asm_elementwise(_GATE_SILU_PTX, "=f,f", [g], dtype=tl.float32, is_pure=True, pack=1)
        else:
            return tl.inline_asm_elementwise(_GATE_SIGMOID_PTX, "=f,f", [g], dtype=tl.float32, is_pure=True,
                                             pack=1)
    else:
        d = 1.0 + tl.exp(-g)
        if SILU:
            return g / d
        else:
            return 1.0 / d


@triton.jit
def _fence_gpu(x, USE_ASM: tl.constexpr):
    """fence.acq_rel.gpu in every thread (CUDA __threadfence()); returns x unchanged. No-op in the interpreter."""
    if USE_ASM:
        return tl.inline_asm_elementwise("fence.acq_rel.gpu;\nmov.b32 $0, $1;", "=r,r", [x], dtype=tl.int32,
                                         is_pure=False, pack=1)
    else:
        return x


@triton.jit
def _norm_vals(v, g, w, eps, V: tl.constexpr, SILU: tl.constexpr, USE_ASM: tl.constexpr):
    """rmsnorm_gated_exact_kernel arithmetic for one 128-col row (v, g, w fp32 [128]) -> fp32 [128] (before cvt.rn)."""
    sq = _mul_rn(v, v, USE_ASM)                                   # __fmul_rn(values[i], values[i])
    q4 = tl.reshape(sq, (32, 2, 2))                               # [lane, i // 2, i % 2]
    ev, od = tl.split(q4)                                         # ev[L, a] = sq[4L+2a], od[L, a] = sq[4L+2a+1]
    c0, c2 = tl.split(ev)
    c1, c3 = tl.split(od)
    s = _add_rn(tl.zeros((32,), tl.float32), c0, USE_ASM)         # sum = 0.0f; sum = __fadd_rn(sum, sq_i), i = 0..3
    s = _add_rn(s, c1, USE_ASM)
    s = _add_rn(s, c2, USE_ASM)
    s = _add_rn(s, c3, USE_ASM)
    # shfl_down 16, 8, 4, 2, 1: lane L <- s[L] + s[L + off]; lane 0's value is broadcast (shfl idx 0)
    lo, hi = tl.split(tl.permute(tl.reshape(s, (2, 16)), (1, 0)))
    s = _add_rn(lo, hi, USE_ASM)
    lo, hi = tl.split(tl.permute(tl.reshape(s, (2, 8)), (1, 0)))
    s = _add_rn(lo, hi, USE_ASM)
    lo, hi = tl.split(tl.permute(tl.reshape(s, (2, 4)), (1, 0)))
    s = _add_rn(lo, hi, USE_ASM)
    lo, hi = tl.split(tl.permute(tl.reshape(s, (2, 2)), (1, 0)))
    s = _add_rn(lo, hi, USE_ASM)
    lo, hi = tl.split(tl.permute(tl.reshape(s, (2, 1)), (1, 0)))
    s = _add_rn(lo, hi, USE_ASM)                                  # [1]
    mean = _mul_rn(s, tl.full((1,), 0.0078125, tl.float32), USE_ASM)          # __fmul_rn(sum, 1.0f / 128)
    inv = _rsqrt_cuda(_add_rn(mean, tl.full((1,), eps, tl.float32), USE_ASM), USE_ASM)
    inv_b = tl.broadcast_to(inv, (V,))
    normalized = _mul_rn(_mul_rn(v, inv_b, USE_ASM), w, USE_ASM)
    act = _gate_cuda(g, SILU, USE_ASM)
    return _mul_rn(normalized, act, USE_ASM)


@triton.jit
def _norm_row(o, z_ptr, nw_ptr, nout_ptr, row, hv, stride_z_tok, eps, HV: tl.constexpr, V: tl.constexpr,
              IS_ZERO: tl.constexpr, SILU: tl.constexpr, USE_ASM: tl.constexpr):
    cols = tl.arange(0, V)
    base = row * (HV * V) + hv * V
    if IS_ZERO:
        v = tl.zeros((V,), tl.float32)
    else:
        # .cg: other CTAs wrote these rows; read them from L2 (L1 is not coherent across SMs)
        v = tl.load(o + base + cols, cache_modifier=".cg").to(tl.float32)
    g = tl.load(z_ptr + row * stride_z_tok + hv * V + cols).to(tl.float32)
    w = tl.load(nw_ptr + cols).to(tl.float32)
    out = _norm_vals(v, g, w, eps, V, SILU, USE_ASM)
    tl.store(nout_ptr + base + cols, out.to(nout_ptr.dtype.element_ty))


# ----------------------------------------------------------------------------------------------------------------------
# conv pieces (copied from vllm/model_executor/layers/mamba/ops/causal_conv1d.py::_causal_conv1d_update_kernel)
# ----------------------------------------------------------------------------------------------------------------------
@triton.jit
def _conv_seg(x_row0, w_ptr, cs_ptr, conv_idx, tok_off, chan0, T, dst, stride_x_token, stride_w_dim,
              stride_w_width, stride_cs_seq, stride_cs_dim, stride_cs_tok,
              DIM: tl.constexpr, CB: tl.constexpr, SW: tl.constexpr, CONV_OK):
    """KERNEL_WIDTH 4, no bias, SiLU, IS_SPEC + IS_VARLEN, for CB channels from chan0, tokens 0..T-1 of one request.
    Writes the fp16 outputs to dst[t * SW + c] instead of in place. CONV_OK False = the stock conv returned early (PAD
    slot): the recurrence then reads raw x, so raw x is copied."""
    idx_feats = chan0 + tl.arange(0, CB)
    mask_w = idx_feats < DIM
    x_base_1d = x_row0 + idx_feats            # stride_x_dim == 1 (admission)
    out_c = tl.arange(0, CB)
    if CONV_OK:
        prior_tokens = cs_ptr + conv_idx * stride_cs_seq + idx_feats * stride_cs_dim + tok_off * stride_cs_tok
        col0 = tl.load(prior_tokens, mask_w, 0.0)
        col1 = tl.load(prior_tokens + 1 * stride_cs_tok, mask_w, 0.0)
        col2 = tl.load(prior_tokens + 2 * stride_cs_tok, mask_w, 0.0)
        acc_preload = tl.zeros((CB,), dtype=tl.float32)
        w_base = w_ptr + (idx_feats * stride_w_dim)
        w_col0 = tl.load(w_base + (0 * stride_w_width), mask_w, other=0.0)
        w_col1 = tl.load(w_base + (1 * stride_w_width), mask_w, other=0.0)
        w_col2 = tl.load(w_base + (2 * stride_w_width), mask_w, other=0.0)
        w_col3 = tl.load(w_base + (3 * stride_w_width), mask_w, other=0.0)
        for idx_token in tl.range(T):
            acc = acc_preload
            matrix_w = w_col0
            matrix_x = col0
            acc += matrix_x * matrix_w
            matrix_w = w_col1
            matrix_x = col1
            acc += matrix_x * matrix_w
            matrix_w = w_col2
            matrix_x = col2
            acc += matrix_x * matrix_w
            matrix_w = w_col3
            x_ptrs_1d = x_base_1d + idx_token * stride_x_token
            matrix_x = tl.load(x_ptrs_1d, mask=mask_w)
            acc += matrix_x * matrix_w
            col0 = col1
            col1 = col2
            col2 = matrix_x
            acc = acc / (1 + tl.exp(-acc))
            tl.store(dst + idx_token * SW + out_c, acc, mask=mask_w)
    else:
        for idx_token in tl.range(T):
            xv = tl.load(x_base_1d + idx_token * stride_x_token, mask=mask_w)
            tl.store(dst + idx_token * SW + out_c, xv, mask=mask_w)


@triton.jit
def _conv_commit(x_row0, cs_ptr, conv_idx, tok_off, chan0, T, num_cache_lines, stride_x_token, stride_cs_seq,
                 stride_cs_dim, stride_cs_tok, DIM: tl.constexpr, CB: tl.constexpr, STATE_LEN: tl.constexpr,
                 MAXQ: tl.constexpr, NP2_STATELEN: tl.constexpr):
    """Stock STEP 1/2 conv-state roll for CB channels (IS_SPEC, IS_VARLEN, APC off: write slot == read slot)."""
    idx_feats = chan0 + tl.arange(0, CB)
    seqlen = T
    state_len = STATE_LEN - (MAXQ - T)
    idx_tokens = tl.arange(0, NP2_STATELEN)
    conv_state_ptrs_source = (
        cs_ptr
        + (conv_idx * stride_cs_seq)
        + tok_off * stride_cs_tok
        + (idx_feats * stride_cs_dim)[None, :]
        + ((idx_tokens + 1) * stride_cs_tok)[:, None]
    )
    mask = (
        (conv_idx < num_cache_lines)
        & ((idx_tokens + seqlen) < state_len)[:, None]
        & (idx_feats < DIM)[None, :]
    )
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)
    VAL = state_len - seqlen
    x_base = x_row0 + idx_feats
    x_ptrs = x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
    mask_x = (
        (idx_tokens - VAL >= 0)[:, None]
        & (idx_tokens - VAL < seqlen)[:, None]
        & (idx_feats < DIM)[None, :]
    )
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)
    tl.debug_barrier()
    new_conv_state = tl.where(mask, conv_state, loaded_x)
    conv_state_ptrs_target = (
        cs_ptr + (conv_idx * stride_cs_seq) + (idx_feats * stride_cs_dim)
    )[None, :] + (idx_tokens * stride_cs_tok)[:, None]
    mask = (idx_tokens < state_len)[:, None] & (idx_feats < DIM)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask)


# ----------------------------------------------------------------------------------------------------------------------
# the fused kernel. Grid (1, NV, N * HV), num_warps / num_stages = the stock sigmoid schedule.
# ----------------------------------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=["N", "T", "M_rows"])
def tc_gdnglue_verify_kernel(
    # conv
    x_ptr, conv_w_ptr, conv_state_ptr, conv_idx_ptr,
    # recurrence (stock argument names)
    A_log, a, b, dt_bias, beta, threshold,
    scr, o, h0, ht, cu_seqlens, ssm_state_indices, num_accepted_tokens, scale,
    # norm
    z_ptr, nw_ptr, nout_ptr, eps,
    cnt_ptr,
    N, T, M_rows, num_cache_lines,
    stride_x_token: tl.int64,
    B: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
    Q_OFFSET: tl.constexpr, K_OFFSET: tl.constexpr, V_OFFSET: tl.constexpr, DIM: tl.constexpr,
    SW: tl.constexpr, TMAX: tl.constexpr,
    stride_init_state_token: tl.constexpr, stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr, stride_indices_tok: tl.constexpr,
    stride_ci: tl.constexpr,
    stride_cs_seq: tl.constexpr, stride_cs_dim: tl.constexpr, stride_cs_tok: tl.constexpr,
    stride_w_dim: tl.constexpr, stride_w_width: tl.constexpr,
    stride_z_tok: tl.constexpr,
    STATE_LEN: tl.constexpr, NP2_STATELEN: tl.constexpr,
    CNT_NORM_BASE: tl.constexpr,
    SILU_NORM: tl.constexpr, USE_ASM: tl.constexpr,
):
    NV: tl.constexpr = V // BV
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    # ---- rows outside every request (graph padding): stock = zero-filled core rows, gated-normed with their z ----
    if i_n == 0:
        if i_v == 0:
            first = tl.load(cu_seqlens).to(tl.int64)
            last = tl.load(cu_seqlens + N).to(tl.int64)
            zrow = tl.zeros((V,), dtype=o.dtype.element_ty)
            for r in range(0, first):
                tl.store(o + r * (HV * V) + i_hv * V + tl.arange(0, V), zrow)
                _norm_row(o, z_ptr, nw_ptr, nout_ptr, r, i_hv, stride_z_tok, eps, HV, V, True, SILU_NORM, USE_ASM)
            for r in range(last, M_rows):
                tl.store(o + r * (HV * V) + i_hv * V + tl.arange(0, V), zrow)
                _norm_row(o, z_ptr, nw_ptr, nout_ptr, r, i_hv, stride_z_tok, eps, HV, V, True, SILU_NORM, USE_ASM)

    bos, eos = (
        tl.load(cu_seqlens + i_n).to(tl.int64),
        tl.load(cu_seqlens + i_n + 1).to(tl.int64),
    )
    all = T
    T = eos - bos
    if T == 0:
        # no tokens to process for this sequence (stock conv and recurrence both return)
        return

    # ---- 1. convolution prologue (per program: q + k of its k-head, its BV slice of v) ----
    conv_idx = tl.load(conv_idx_ptr + i_n * stride_ci).to(tl.int64)
    conv_ok = conv_idx != -1
    tok_off = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1    # conv_state_token_offset
    p_prog = i_nh * NV + i_v
    scr_base = scr + p_prog * (TMAX * SW)
    x_row0 = x_ptr + bos * stride_x_token
    _conv_seg(x_row0, conv_w_ptr, conv_state_ptr, conv_idx, tok_off, Q_OFFSET + i_h * K, T, scr_base,
              stride_x_token, stride_w_dim, stride_w_width, stride_cs_seq, stride_cs_dim, stride_cs_tok,
              DIM, K, SW, conv_ok)
    _conv_seg(x_row0, conv_w_ptr, conv_state_ptr, conv_idx, tok_off, K_OFFSET + i_h * K, T, scr_base + K,
              stride_x_token, stride_w_dim, stride_w_width, stride_cs_seq, stride_cs_dim, stride_cs_tok,
              DIM, K, SW, conv_ok)
    _conv_seg(x_row0, conv_w_ptr, conv_state_ptr, conv_idx, tok_off, V_OFFSET + i_hv * V + i_v * BV, T,
              scr_base + 2 * K, stride_x_token, stride_w_dim, stride_w_width, stride_cs_seq, stride_cs_dim,
              stride_cs_tok, DIM, BV, SW, conv_ok)
    tl.debug_barrier()

    if conv_ok:
        # own v slice: no other program reads it -> commit now (stock order: loads, barrier, store)
        _conv_commit(x_row0, conv_state_ptr, conv_idx, tok_off, V_OFFSET + i_hv * V + i_v * BV, T, num_cache_lines,
                     stride_x_token, stride_cs_seq, stride_cs_dim, stride_cs_tok, DIM, BV, STATE_LEN, TMAX,
                     NP2_STATELEN)
        # q/k slice of k-head i_h: read by (HV // H) * NV programs; the last reader commits it
        fz = _fence_gpu(tl.zeros((32,), tl.int32), USE_ASM)
        tl.debug_barrier()
        c_ptr = cnt_ptr + i_n * H + i_h + tl.sum(fz, axis=0)
        old = tl.atomic_add(c_ptr, 1, sem="acq_rel", scope="gpu")
        if old == (HV // H) * NV - 1:
            tl.atomic_xchg(c_ptr, 0, sem="relaxed", scope="gpu")
            _conv_commit(x_row0, conv_state_ptr, conv_idx, tok_off, Q_OFFSET + i_h * K, T, num_cache_lines,
                         stride_x_token, stride_cs_seq, stride_cs_dim, stride_cs_tok, DIM, K, STATE_LEN, TMAX,
                         NP2_STATELEN)
            _conv_commit(x_row0, conv_state_ptr, conv_idx, tok_off, K_OFFSET + i_h * K, T, num_cache_lines,
                         stride_x_token, stride_cs_seq, stride_cs_dim, stride_cs_tok, DIM, K, STATE_LEN, TMAX,
                         NP2_STATELEN)

    # ---- 2. recurrence: fused_sigmoid_gating_delta_rule_update_kernel body (MIXED_QKV, IS_VARLEN, continuous
    #         batching, IS_SPEC_DECODING, INPLACE_FINAL_STATE, USE_QK_L2NORM_IN_KERNEL, no KDA/DDTREE/quantize) ----
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = scr_base + o_k
    p_k = scr_base + K + o_k
    p_v = scr_base + 2 * K + tl.arange(0, BV)

    p_A_log = A_log + i_hv
    p_a = a + bos * HV + i_hv
    p_dt_bias = dt_bias + i_hv

    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    i_t0 = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t0).to(tl.int64)
    if state_idx >= 0:
        b_h = tl.zeros([BV, BK], dtype=tl.float32)
        p_h0 = h0 + state_idx * stride_init_state_token
        p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

        for i_t in range(0, T):
            b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
            b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
            b_b = tl.load(p_b).to(tl.float32)

            # If the model is loaded in fp16, without the .float() here, A
            # might be -inf.
            x = tl.load(p_a).to(tl.float32) + tl.load(p_dt_bias).to(tl.float32)
            softplus_x = tl.where(
                beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
            )
            b_g = -tl.exp(tl.load(p_A_log).to(tl.float32)) * softplus_x

            # compute beta_output = sigmoid(b)
            b_beta = tl.sigmoid(b_b.to(tl.float32))

            b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))
            b_q = b_q * scale
            # [BV, BK]
            b_h *= tl.exp(b_g)
            # [BV]
            b_v -= tl.sum(b_h * b_k[None, :], 1)
            b_v *= b_beta
            # [BV, BK]
            b_h += b_v[:, None] * b_k[None, :]
            # [BV]
            b_o = tl.sum(b_h * b_q[None, :], 1)
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

            # keep the states for multi-query tokens
            final_state_idx = tl.load(
                ssm_state_indices + i_n * stride_indices_seq + i_t
            ).to(tl.int64)
            if final_state_idx >= 0:
                p_ht = ht + final_state_idx * stride_final_state_token
                p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

            p_q += SW
            p_k += SW
            p_v += SW
            p_o += HV * V
            p_b += HV
            p_a += HV
    else:
        # stock: the recurrence program returns -> these rows keep the zero fill
        zv = tl.zeros((BV,), dtype=o.dtype.element_ty)
        for i_t in range(0, T):
            tl.store(p_o, zv, mask=mask_v)
            p_o += HV * V

    # ---- 3. gated RMSNorm epilogue: the last of the NV programs of (request, v-head) norms the head's rows ----
    fz2 = _fence_gpu(tl.zeros((32,), tl.int32), USE_ASM)
    tl.debug_barrier()
    n_ptr = cnt_ptr + CNT_NORM_BASE + i_n * HV + i_hv + tl.sum(fz2, axis=0)
    old2 = tl.atomic_add(n_ptr, 1, sem="acq_rel", scope="gpu")
    if old2 == NV - 1:
        tl.atomic_xchg(n_ptr, 0, sem="relaxed", scope="gpu")
        fz3 = _fence_gpu(tl.zeros((32,), tl.int32), USE_ASM)
        for r in range(bos + tl.sum(fz3, axis=0), eos):
            _norm_row(o, z_ptr, nw_ptr, nout_ptr, r, i_hv, stride_z_tok, eps, HV, V, False, SILU_NORM, USE_ASM)
