"""Triton kernels for tc_sampler (ThinkingCap TP-local exact MTP sampler). See TPSAMPLE-README.md.

All noise/acceptance expressions are copied from vllm/v1/worker/gpu/sample/gumbel.py and
vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py (engine 6a143197e); token ids used as Philox keys are int32
GLOBAL vocabulary ids exactly like the stock `block` offsets, so every per-token noise value is bit-identical.
"""
from __future__ import annotations

import triton
import triton.language as tl

from vllm.triton_utils import tldevice
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax, tl_rand32, tl_rand64

_DRAFT_SALT = tl.constexpr(1 << 30)     # gumbel._DRAFT_NOISE_SALT (only used through gumbel_noised_argmax)


# ----------------------------------------------------------------------------------------------------------------------
# draft step: _gumbel_sample_kernel on the local vocab shard
# ----------------------------------------------------------------------------------------------------------------------
@triton.jit
def tc_draft_gumbel_kernel(
    local_argmax_ptr, local_argmax_stride, local_max_ptr, local_max_stride,
    dl_ptr, dl_stride0, dl_stride1, dl_col_ptr,
    logits_ptr, logits_stride, idx_mapping_ptr, seeds_ptr, pos_ptr, temp_ptr,
    vocab_local, vstart,
    BLOCK_SIZE: tl.constexpr, USE_FP64: tl.constexpr,
):
    """Per (row, 1024-block of the local shard). Same as stock gumbel_block_argmax(APPLY_TEMPERATURE=True,
    IS_DRAFTING=True) except: the processed (temperature-applied) logits go to the LOCAL draft buffer column `block`,
    the Philox keys are the GLOBAL ids vstart + block."""
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_local
    logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask, other=float("-inf"))
    logits = logits.to(tl.float32)

    req_state_idx = tl.load(idx_mapping_ptr + token_idx).to(tl.int64)
    is_valid_req = req_state_idx >= 0
    temp = tl.load(temp_ptr + req_state_idx, mask=is_valid_req, other=0.0).to(tl.float32)
    if temp != 0.0:
        logits = logits / temp
    col = tl.load(dl_col_ptr)
    tl.store(dl_ptr + req_state_idx * dl_stride0 + col * dl_stride1 + block, logits, mask=mask & is_valid_req)
    seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid_req, other=0)
    pos = tl.load(pos_ptr + token_idx)
    keys = vstart + block
    value, idx = gumbel_noised_argmax(logits, keys, mask & is_valid_req, seed, pos, temp, IS_DRAFTING=True,
                                      USE_FP64=USE_FP64, APPLY_TEMPERATURE=False)
    token_id = vstart + block_idx * BLOCK_SIZE + idx
    tl.store(local_argmax_ptr + token_idx * local_argmax_stride + block_idx, token_id)
    tl.store(local_max_ptr + token_idx * local_max_stride + block_idx, value)


# ----------------------------------------------------------------------------------------------------------------------
# target step, rank-local: (max, sumexp) of the row's draft logits over the local shard (stock 8192-block statistics)
# ----------------------------------------------------------------------------------------------------------------------
@triton.jit
def tc_draft_stats_kernel(out_max_ptr, out_sum_ptr, dl_ptr, dl_stride0, dl_stride1, exp_idx_ptr, exp_pos_ptr,
                          num_steps, vocab_local, NBLK: tl.constexpr, NBLK_P2: tl.constexpr,
                          BLOCK_SIZE: tl.constexpr):
    r = tl.program_id(0)
    step = tl.load(exp_pos_ptr + r)
    if step >= num_steps:
        tl.store(out_max_ptr + r, float("-inf"))
        tl.store(out_sum_ptr + r, 0.0)
        return
    req = tl.load(exp_idx_ptr + r).to(tl.int64)
    base = dl_ptr + req * dl_stride0 + step * dl_stride1
    maxes = tl.full((NBLK_P2,), float("-inf"), tl.float32)
    sums = tl.zeros((NBLK_P2,), tl.float32)
    bl = tl.arange(0, NBLK_P2)
    for b in tl.static_range(NBLK):
        offs = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(base + offs, mask=offs < vocab_local, other=float("-inf")).to(tl.float32)
        bmax = tl.max(x, axis=0)
        bsum = tl.where(bmax > float("-inf"), tl.sum(tl.exp(x - bmax)), 0.0)
        maxes = tl.where(bl == b, bmax, maxes)
        sums = tl.where(bl == b, bsum, sums)
    gmax = tl.max(maxes, axis=0)
    tot = tl.sum(sums * tl.exp(maxes - gmax))
    tl.store(out_max_ptr + r, gmax)
    tl.store(out_sum_ptr + r, tot)


# ----------------------------------------------------------------------------------------------------------------------
# target step, replicated on every rank: rejection + resample + insert on the compact candidate set
# ----------------------------------------------------------------------------------------------------------------------
@triton.jit
def _target_lse(kv, ids, VOCAB_BLOCK: tl.constexpr, NVB: tl.constexpr, NVB_P2: tl.constexpr):
    """Stock _compute_block_max_and_sumexp per 8192 vocab block + _compute_global_lse, evaluated on the kept slots
    (every other token of the processed row is -inf and contributes exactly 0)."""
    blk = ids // VOCAB_BLOCK
    bl = tl.arange(0, NVB_P2)
    maxes = tl.full((NVB_P2,), float("-inf"), tl.float32)
    sums = tl.zeros((NVB_P2,), tl.float32)
    for b in tl.static_range(NVB):
        x = tl.where(blk == b, kv, float("-inf"))
        bmax = tl.max(x, axis=0)
        bsum = tl.where(bmax > float("-inf"), tl.sum(tl.exp(x - bmax)), 0.0)
        maxes = tl.where(bl == b, bmax, maxes)
        sums = tl.where(bl == b, bsum, sums)
    gmax = tl.max(maxes, axis=0)
    return gmax + tl.log(tl.sum(sums * tl.exp(maxes - gmax)))


@triton.jit
def tc_target_reject_kernel(
    sampled_ptr, sampled_stride, num_sampled_ptr,
    kept_ptr, ids_ptr, dl_ptr, cand_stride, tgt_argmax_ptr,
    dmax_ptr, dsum_ptr, dlx_ptr, TP: tl.constexpr,
    draft_sampled_ptr, cu_num_logits_ptr, idx_mapping_ptr, expanded_idx_mapping_ptr,
    temp_ptr, seed_ptr, pos_ptr,
    C: tl.constexpr, VOCAB_BLOCK: tl.constexpr, NVB: tl.constexpr, NVB_P2: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr, USE_FP64: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx
    seed = tl.load(seed_ptr + req_state_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    slots = tl.arange(0, C)
    ranks = tl.arange(0, TP)

    rejected_step = 0
    target_lse = 0.0
    draft_lse = 0.0
    accepted = True
    for i in range(num_tokens - 1):
        if accepted:
            logit_idx = start_idx + i
            draft_sampled = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
            if temp == 0.0:
                target_argmax = tl.load(tgt_argmax_ptr + logit_idx).to(tl.int64)
                accepted &= target_argmax == draft_sampled
                tl.store(sampled_ptr + req_idx * sampled_stride + i,
                         draft_sampled if accepted else target_argmax)
            else:
                kv = tl.load(kept_ptr + logit_idx * cand_stride + slots)
                ids = tl.load(ids_ptr + logit_idx * cand_stride + slots)
                target_logit = tl.max(tl.where(ids.to(tl.int64) == draft_sampled, kv, float("-inf")), axis=0)
                target_lse = _target_lse(kv, ids, VOCAB_BLOCK, NVB, NVB_P2)
                target_log_prob = target_logit - target_lse
                pos = tl.load(pos_ptr + logit_idx)
                u = tl_rand32(seed, pos, includes_zero=False)
                if HAS_DRAFT_LOGITS:
                    draft_logit = tl.max(tl.load(dlx_ptr + logit_idx * TP + ranks), axis=0)
                    dmax = tl.load(dmax_ptr + logit_idx * TP + ranks)
                    dsum = tl.load(dsum_ptr + logit_idx * TP + ranks)
                    gmax = tl.max(dmax, axis=0)
                    draft_lse = gmax + tl.log(tl.sum(dsum * tl.exp(dmax - gmax)))
                    draft_log_prob = draft_logit - draft_lse
                else:
                    draft_log_prob = 0
                accepted &= target_log_prob > tl.log(u) + draft_log_prob
                tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)
            rejected_step += accepted

    # ---- resample the rejected / bonus token (stock _resample_kernel + _insert_resampled_kernel) ----
    resample_token_idx = start_idx + rejected_step
    rs_state = tl.load(expanded_idx_mapping_ptr + resample_token_idx)
    rtemp = tl.load(temp_ptr + rs_state).to(tl.float32)
    is_bonus = resample_token_idx == end_idx - 1
    tl.store(num_sampled_ptr + req_idx, rejected_step + 1)
    if rtemp == 0.0 and not is_bonus:
        return
    kv = tl.load(kept_ptr + resample_token_idx * cand_stride + slots)
    ids = tl.load(ids_ptr + resample_token_idx * cand_stride + slots)
    if is_bonus:
        residual_logits = kv
    elif HAS_DRAFT_LOGITS:
        dlv = tl.load(dl_ptr + resample_token_idx * cand_stride + slots)
        target_log_probs = kv - target_lse
        draft_log_probs = dlv - draft_lse
        ratio = tl.exp(draft_log_probs - target_log_probs)
        residual_logits = tl.where(
            ratio < 1.0,
            target_log_probs + tldevice.log1p(-ratio),
            float("-inf"),
        ).to(tl.float32)
    else:
        rejected_draft_token = tl.load(draft_sampled_ptr + resample_token_idx + 1)
        residual_logits = tl.where(ids != rejected_draft_token, kv, float("-inf")).to(tl.float32)
    # gumbel_noised_argmax(APPLY_TEMPERATURE=False, IS_DRAFTING=False) with Philox keys = global ids; ties -> lowest id
    rseed = tl.load(seed_ptr + rs_state)
    rpos = tl.load(pos_ptr + resample_token_idx)
    logits = residual_logits
    if USE_FP64:
        logits = logits.to(tl.float64)
    if rtemp != 0.0:
        gumbel_seed = tl.randint(rseed, rpos)
        if USE_FP64:
            uu = tl_rand64(gumbel_seed, ids, includes_zero=False)
            gumbel_noise = -tl.log(-tl.log(uu))
        else:
            uu = tl_rand32(gumbel_seed, ids, includes_zero=False)
            gumbel_noise = -tl.log(-tldevice.log1p(-uu))
        logits = logits + gumbel_noise
    best = tl.max(logits, axis=0)
    tok = tl.min(tl.where(logits == best, ids, 2147483647), axis=0)
    tok = tl.where(best == float("-inf"), 0, tok)        # stock all -inf row -> block 0, index 0
    tl.store(sampled_ptr + req_idx * sampled_stride + rejected_step, tok.to(tl.int64))
