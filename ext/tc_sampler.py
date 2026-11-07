"""ThinkingCap TP-local exact MTP sampler (2026-10-26). README: /opt/1cat/tcflash-trial/ext/TPSAMPLE-README.md.

Enable:  TC_TPSAMPLE=1   (default off; install() is a no-op otherwise). Nothing here is a VLLM_* env and nothing is
         traced by torch.compile. Needs draft_sample_method=probabilistic (the run-tc-best.sh default); with greedy
         drafts it stays out of the way (stock path).
Knobs:   TC_TPSAMPLE_LOG=1     log each fallback reason once
         TC_TPSAMPLE_STRICT=1  also fall back on stock "margin" reference rows (a top-p cumulative within 1.2e-4 of
                               1-p, ~a few % of verify steps); default 0 keeps the compact decision there
         TC_TPSAMPLE_TARGET=0  only the draft steps go TP-local (target verify on the stock path, draft logits are
                               then re-assembled by one all-gather per verify step) - A/B / debugging only.

Draft steps (EagleSpeculator._sample_draft, captured in the draft CUDA graphs): each rank runs the stock Gumbel kernel
  on its 62,080-token logits shard (Philox keys = global ids) and keeps the temperature-applied shard in a local draft
  buffer; one tiny all-gather of (value, id) per row picks the global winner (max value, lowest id = stock order).
  -> the draft token is bit-identical to stock; the 124 KB/row logits all-gather is gone.
Target verify (GPUModelRunner.sample, eager): each rank computes its shard's processed logits (fp32 copy + the stock
  temperature kernel), its local top-128 (value, id), the row's draft logits at those ids, the draft value at the
  proposed token and its shard's draft (max, sumexp). ONE all-gather of that ~1.5 KB/row packet; then every rank
  computes the same result from the same gathered bytes: global top-128 -> the stock SM70 compact top-k/top-p
  routine on that shortlist (same function, same input -> same kept set and values) -> rejection / residual resample /
  bonus on the kept set. Exactness: see README (greedy bit-identical; probabilistic = same distribution, the target /
  draft log-normalisers are summed in a different order -> accept/resample decisions can differ only within fp32
  rounding of a log-probability).
Falls back to the stock sampler (with the draft logits re-assembled first) for: grammar / structured output, logprobs,
  logit bias / allowed ids / min_tokens, penalties, bad words, min_p, NaN counting, synthetic acceptance, top_k
  missing or >= 128, top_p without top_k, a top-k cutoff tied past the 128-shortlist or non-finite logits, a greedy row
  whose maximum is tied inside an ambiguous row, (TC_TPSAMPLE_STRICT=1) any stock "reference row", vocab padding,
  tp == 1.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys

TARGET_RUNNER = "vllm.v1.worker.gpu.model_runner"
TARGET_SPEC = "vllm.v1.worker.gpu.spec_decode.eagle.speculator"
CAPACITY = 128            # stock _apply_top_k_top_p_compact shortlist
VOCAB_BLOCK = 8192        # stock rejection block statistics

_stats = {"installed": [], "draft_local_calls": 0, "draft_stock_calls": 0, "target_local": 0,
          "target_fallbacks": 0, "fallback_reasons": {}, "draft_reassembled": 0, "errors": 0, "last_error": ""}
_ORIG = {}


def _say(msg):
    print(f"TC-TPSAMPLE pid {os.getpid()}: {msg}", flush=True)


def enabled() -> bool:
    return os.environ.get("TC_TPSAMPLE", "0") == "1"


def _fb(reason):
    _stats["target_fallbacks"] += 1
    d = _stats["fallback_reasons"]
    first = reason not in d
    d[reason] = d.get(reason, 0) + 1
    if first and os.environ.get("TC_TPSAMPLE_LOG", "0") == "1":
        _say(f"target fallback: {reason}")


def status() -> dict:
    out = dict(_stats)
    out["fallback_reasons"] = dict(_stats["fallback_reasons"])
    return out


# ----------------------------------------------------------------------------------------------------------------------
# pure building blocks (device-agnostic; the CPU test calls them per simulated rank)
# ----------------------------------------------------------------------------------------------------------------------
def draft_local(local_logits, vstart, idx_mapping, temperature, seeds, pos_plus1, dl_col, dl_local, use_fp64):
    """Rank-local half of the draft sample: writes dl_local[req_state, col, :] and returns [rows, 2] (value, id) in
    the gumbel value dtype (fp32 or fp64; ids < 2^24 are exact in both)."""
    import torch
    import triton

    from tc_sampler_kernels import tc_draft_gumbel_kernel

    rows, vloc = local_logits.shape
    BLOCK = 1024
    nb = triton.cdiv(vloc, BLOCK)
    la = local_logits.new_empty(rows, nb, dtype=torch.int64)
    lm = local_logits.new_empty(rows, nb, dtype=torch.float64 if use_fp64 else torch.float32)
    tc_draft_gumbel_kernel[(rows, nb)](
        la, la.stride(0), lm, lm.stride(0),
        dl_local, dl_local.stride(0), dl_local.stride(1), dl_col,
        local_logits, local_logits.stride(0), idx_mapping, seeds, pos_plus1, temperature,
        vloc, vstart, BLOCK_SIZE=BLOCK, USE_FP64=use_fp64,
    )
    best = lm.argmax(dim=-1, keepdim=True)                       # first max = lowest id (as stock torch.argmax)
    val = lm.gather(1, best)
    tok = la.gather(1, best).to(val.dtype)
    return torch.cat([val, tok], dim=-1)


def draft_combine(gathered_pairs, tp):
    """[rows, tp*2] (all_gather dim=-1 of draft_local pairs) -> [rows] int64 token ids (first rank with the max)."""
    g = gathered_pairs.view(gathered_pairs.shape[0], tp, 2)
    r = g[:, :, 0].argmax(dim=-1, keepdim=True)
    return g[:, :, 1].gather(1, r).squeeze(-1).to(__import__("torch").int64)


def target_local_packet(local_logits, vstart, exp_idx, exp_pos, temperature_gpu, temperature_np, idx_np,
                        draft_sampled, dl_local, num_steps):
    """Rank-local half of the verify: [M, 3*C + 3] int32 packet (fp32 bits): local top-C processed values, their
    global ids, the row's draft logits at those ids, then draft (max, sumexp) over the shard and the draft logit at the
    proposed token (or -inf off-shard)."""
    import numpy as np
    import torch
    import triton

    from tc_sampler_kernels import tc_draft_stats_kernel
    from vllm.v1.worker.gpu.sample.gumbel import apply_temperature

    M, vloc = local_logits.shape
    proc = torch.empty_like(local_logits, dtype=torch.float32).copy_(local_logits)
    temp_np = temperature_np[idx_np]
    if not np.all((temp_np == 0.0) | (temp_np == 1.0)):          # stock SamplingStates.apply_temperature skip rule
        apply_temperature(proc, exp_idx, temperature_gpu)
    vals, lidx = proc.topk(CAPACITY, dim=-1, sorted=False)
    gids = (lidx + vstart).to(torch.int32)
    steps = exp_pos.to(torch.int64)
    has = steps < num_steps
    stepc = steps.clamp(max=num_steps - 1)
    reqs = exp_idx.to(torch.int64)
    dl_rows = dl_local[reqs, stepc]                               # [M, vloc]
    dl_at = torch.where(has[:, None], dl_rows.gather(1, lidx), torch.full_like(vals, float("-inf")))
    dmax = torch.empty(M, dtype=torch.float32, device=proc.device)
    dsum = torch.empty_like(dmax)
    nblk = triton.cdiv(vloc, VOCAB_BLOCK)
    tc_draft_stats_kernel[(M,)](dmax, dsum, dl_local, dl_local.stride(0), dl_local.stride(1), exp_idx, exp_pos,
                                num_steps, vloc, NBLK=nblk, NBLK_P2=triton.next_power_of_2(nblk),
                                BLOCK_SIZE=VOCAB_BLOCK)
    nxt = torch.cat([draft_sampled[1:], draft_sampled[:1]]).to(torch.int64)    # proposed token of row r = ids[r+1]
    loc = nxt - vstart
    on = (loc >= 0) & (loc < vloc) & has
    dlx = torch.where(on, dl_rows.gather(1, loc.clamp(0, vloc - 1)[:, None]).squeeze(1),
                      torch.full_like(dmax, float("-inf")))
    packet = torch.cat([vals, gids.view(torch.float32), dl_at, dmax[:, None], dsum[:, None], dlx[:, None]], dim=-1)
    return packet.contiguous().view(torch.int32)


def target_finish(gathered, tp, vocab, k_rows, p_rows, do_top_p, input_rows, use_fp64, has_draft_logits=True,
                  strict=None):
    """Replicated half: `gathered` = [tp, M, 3*C+3] int32 (identical bytes on every rank). input_rows = dict with
    draft_sampled, cu_num_logits, idx_mapping, expanded_idx_mapping, temperature, seeds, pos, num_reqs, num_steps.
    Returns (sampled [R, S+1] int64, num_sampled [R] int32, needs_fallback scalar bool tensor)."""
    import torch

    from tc_sampler_kernels import tc_target_reject_kernel
    from vllm.v1.sample.ops.topk_topp_triton import sort_topk_with_vocab_ties

    C = CAPACITY
    g = gathered.view(torch.float32)
    M = g.shape[1]
    cvals = g[:, :, 0:C].permute(1, 0, 2).reshape(M, tp * C)
    cids = g[:, :, C:2 * C].contiguous().view(torch.int32).permute(1, 0, 2).reshape(M, tp * C).to(torch.int64)
    cdl = g[:, :, 2 * C:3 * C].permute(1, 0, 2).reshape(M, tp * C)
    dmax = g[:, :, 3 * C].t().contiguous()                      # [M, tp]
    dsum = g[:, :, 3 * C + 1].t().contiguous()
    dlx = g[:, :, 3 * C + 2].t().contiguous()

    # global shortlist (any tie rule at the C-th value: ties there are either masked below or flagged as reference)
    tv, tpos = cvals.topk(C, dim=-1, sorted=False)
    ti = cids.gather(1, tpos)
    values, indices = sort_topk_with_vocab_ties(tv.contiguous(), ti.contiguous(), vocab_size=vocab)
    dl_s = torch.where(indices[:, :, None] == cids[:, None, :], cdl[:, None, :],
                       torch.full_like(cdl[:, None, :], float("-inf"))).amax(-1)

    # --- stock vllm/v1/sample/ops/topk_topp_triton.py::_apply_top_k_top_p_compact on the shortlist (verbatim) ---
    k = k_rows
    cutoff = values.gather(1, (C - k.long()).unsqueeze(1))
    reference_rows = (values[:, 0] >= cutoff[:, 0]) | ~torch.isfinite(values[:, -1])
    reference_rows |= torch.isnan(values[:, 0])
    tie_rows = reference_rows.clone()          # cutoff ties may extend past the shortlist / non-finite -> must fall back
    values = values.masked_fill(values < cutoff, -float("inf"))
    if do_top_p:
        p = p_rows
        probabilities = values.softmax(dim=-1)
        cumulative = probabilities.cumsum(dim=-1)
        boundary = 1 - p.unsqueeze(1)
        margin = 8 * C * torch.finfo(torch.float32).eps
        reference_rows |= ((cumulative - boundary).abs() <= margin).any(dim=-1)
        remove = cumulative <= boundary
        remove[:, -1] = False
        values = values.masked_fill(remove, -float("inf"))
    kept = values.contiguous()

    # greedy target argmax over the processed row: max value, lowest id
    rowmax = kept.amax(-1, keepdim=True)
    big = torch.iinfo(torch.int64).max
    tgt_argmax = torch.where(kept == rowmax, indices, torch.full_like(indices, big)).amin(-1)
    tie_at_max = (kept == rowmax).sum(-1) > 1
    temp_rows = input_rows["temperature"][input_rows["expanded_idx_mapping"].long()]
    greedy = temp_rows == 0
    # greedy rows: the processed argmax only depends on the kept set when the maximum is tied.
    # sampled rows: tie rows always fall back; margin rows (a cumulative probability within 8*128*eps of 1-p: the
    # stock route re-decides them with a full-row reduction, i.e. a different fp32 rounding of the SAME rule) fall back
    # only in strict mode (TC_TPSAMPLE_STRICT=1); otherwise the compact decision is used (distribution-exact).
    if strict is None:
        strict = os.environ.get("TC_TPSAMPLE_STRICT", "0") == "1"
    sampled_need = reference_rows if strict else tie_rows
    needs = torch.where(greedy, reference_rows & tie_at_max, sampled_need).any()

    R = input_rows["num_reqs"]
    S = input_rows["num_steps"]
    sampled = kept.new_empty((R, S + 1), dtype=torch.int64)
    num_sampled = torch.empty(R, dtype=torch.int32, device=kept.device)
    ids32 = indices.to(torch.int32).contiguous()
    nvb = (vocab + VOCAB_BLOCK - 1) // VOCAB_BLOCK
    tc_target_reject_kernel[(R,)](
        sampled, sampled.stride(0), num_sampled,
        kept, ids32, dl_s.contiguous(), C, tgt_argmax.contiguous(),
        dmax, dsum, dlx, tp,
        input_rows["draft_sampled"], input_rows["cu_num_logits"], input_rows["idx_mapping"],
        input_rows["expanded_idx_mapping"], input_rows["temperature"], input_rows["seeds"], input_rows["pos"],
        C=C, VOCAB_BLOCK=VOCAB_BLOCK, NVB=nvb, NVB_P2=1 << (nvb - 1).bit_length(),
        HAS_DRAFT_LOGITS=has_draft_logits, USE_FP64=use_fp64, num_warps=1,
    )
    return sampled, num_sampled, needs


# ----------------------------------------------------------------------------------------------------------------------
# engine hooks
# ----------------------------------------------------------------------------------------------------------------------
def _tp():
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

    return get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()


def _lm_owner(model):
    """The module that owns lm_head + logits_processor (unwraps ...ForConditionalGeneration.language_model and simple
    runtime wrappers)."""
    m = model
    for _ in range(6):
        if m is None:
            return None
        if hasattr(m, "lm_head") and hasattr(m, "logits_processor"):
            return m
        m = getattr(m, "language_model", None) or getattr(m, "runnable", None) or getattr(m, "module", None)
    return None


def _draft_head(owner):
    head = getattr(owner, "_sm70_draft_head", None)
    return head if head is not None else owner.lm_head


def _head_ok(head, lp, tp):
    si = getattr(head, "shard_indices", None)
    if si is None or tp <= 1:
        return False
    if getattr(si, "num_org_vocab_padding", 0) or getattr(si, "num_added_elements", 0):
        return False
    if lp is None or lp.scale != 1.0 or lp.soft_cap is not None or lp.logits_as_input:
        return False
    return True


def _spec_init_wrapper(orig_init):
    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self._tc_dl_local = None
        self._tc_local_drafts = False
        try:
            import torch

            tp, _ = _tp()
            dl = self.draft_logits
            if dl is not None and tp > 1 and dl.shape[-1] % tp == 0:
                self._tc_dl_local = torch.zeros(dl.shape[0], dl.shape[1], dl.shape[-1] // tp, dtype=dl.dtype,
                                                device=dl.device)
        except Exception as exc:  # noqa: BLE001
            _say(f"speculator init hook: {exc!r} (TP-local drafts off)")
    return __init__


def _draft_ready(spec):
    """Static per speculator: decided once (the draft graphs bake the choice)."""
    ok = getattr(spec, "_tc_local_drafts_checked", None)
    if ok is not None:
        return ok
    tp, _ = _tp()
    owner = _lm_owner(getattr(spec, "model", None))
    head = _draft_head(owner) if owner is not None else None
    ok = bool(
        spec.draft_logits is not None and getattr(spec, "_tc_dl_local", None) is not None
        and head is not None and _head_ok(head, getattr(owner, "logits_processor", None), tp)
        and spec._tc_dl_local.shape[-1] * tp == spec.draft_logits.shape[-1]
        and head.shard_indices.org_vocab_end_index - head.shard_indices.org_vocab_start_index
        == spec._tc_dl_local.shape[-1]
    )
    spec._tc_local_drafts_checked = ok
    spec._tc_local_drafts = ok
    _say(f"draft steps TP-local: {ok} (tp={tp})")
    return ok


def _tc_sample_draft(self, hidden_states, idx_mapping, pos, draft_step, draft_logits):
    orig = _ORIG["_sample_draft"]
    if draft_logits is None or not _draft_ready(self):
        _stats["draft_stock_calls"] += 1
        return orig(self, hidden_states, idx_mapping, pos, draft_step, draft_logits)
    from vllm.distributed import tensor_model_parallel_all_gather

    tp, _ = _tp()
    head = _draft_head(_lm_owner(self.model))
    local = head.quant_method.apply(head, hidden_states, bias=None)
    vstart = int(head.shard_indices.org_vocab_start_index)
    pairs = draft_local(local, vstart, idx_mapping, self.temperature, self.seeds, pos + 1, draft_step,
                        self._tc_dl_local, bool(self.use_fp64_gumbel))
    gathered = tensor_model_parallel_all_gather(pairs, dim=-1)
    _stats["draft_local_calls"] += 1
    return draft_combine(gathered, tp)


def _reassemble_draft_logits(spec, input_batch):
    """Fallback: rebuild the stock full-vocab draft logits of this batch's requests from the local shards."""
    import torch

    from vllm.distributed import tensor_model_parallel_all_gather

    reqs = torch.as_tensor(input_batch.idx_mapping_np, device=spec._tc_dl_local.device, dtype=torch.long)
    part = spec._tc_dl_local.index_select(0, reqs)                  # [n, S, vloc]
    full = tensor_model_parallel_all_gather(part.contiguous(), dim=-1)
    spec.draft_logits.index_copy_(0, reqs, full)
    _stats["draft_reassembled"] += 1


def _target_admit(runner, input_batch, grammar_output):
    import numpy as np

    from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS

    rs = runner.rejection_sampler
    spec = runner.speculator
    if rs is None or spec is None or not getattr(spec, "_tc_local_drafts", False):
        return "drafts_not_local"
    if os.environ.get("TC_TPSAMPLE_TARGET", "1") != "1":
        return "target_disabled"
    if grammar_output is not None or input_batch.has_structured_output_reqs:
        return "grammar"
    sampler = rs.sampler
    st = sampler.sampling_states
    idx = input_batch.idx_mapping_np
    if sampler.compute_nans:
        return "compute_nans"
    if rs.synthetic_conditional_rates is not None:
        return "synthetic"
    if st.max_num_logprobs(idx) != NO_LOGPROBS or sampler.logprob_token_ids_state.max_num_token_ids(idx) != 0:
        return "logprobs"
    if np.any(sampler.logit_bias_state.use_logit_bias[idx]):
        return "logit_bias"
    if np.any(sampler.penalties_state.use_penalty[idx]):
        return "penalties"
    if np.any(sampler.bad_words_state.num_bad_words.np[idx]):
        return "bad_words"
    if np.any(st.min_p.np[idx] != 0.0):
        return "min_p"
    k = st.top_k.np[idx]
    if not np.all((k > 0) & (k < CAPACITY)):
        return "top_k_missing_or_large"
    kc = runner.vllm_config.kernel_config
    if getattr(kc, "sm70_greedy_verify", False) and runner._sm70_greedy_capability and \
            sampler.can_use_sm70_greedy_token_fastpath(input_batch):
        return "stock_sm70_greedy_verify"          # stock's own TP-local greedy route (needs no draft logits)
    owner = _lm_owner(runner.model)
    tp, _ = _tp()
    if owner is None or not _head_ok(owner.lm_head, owner.logits_processor, tp):
        return "target_head"
    if input_batch.cu_num_logits_np[-1] < 2:
        return "single_row"
    return ""


def _tc_sample(self, hidden_states, input_batch, grammar_output):
    orig = _ORIG["sample"]
    if input_batch.num_draft_tokens <= 0:
        return orig(self, hidden_states, input_batch, grammar_output)
    spec = self.speculator
    reason = _target_admit(self, input_batch, grammar_output)
    result = None
    if not reason:
        # No try/except here on purpose: the TP-local path and the stock path issue DIFFERENT collectives, so a
        # rank-local "catch and fall back" would desynchronise NCCL (hang). Errors surface like any engine error.
        # Every branch decision below is a function of replicated host state or of the gathered bytes -> identical on
        # all ranks.
        result = _target_tp_local(self, hidden_states, input_batch)
        if result is None:
            reason = "ambiguous_cutoff"
    if result is not None:
        _stats["target_local"] += 1
        return result
    _fb(reason)
    if spec is not None and getattr(spec, "_tc_local_drafts", False):
        _reassemble_draft_logits(spec, input_batch)
    return orig(self, hidden_states, input_batch, grammar_output)


def _target_tp_local(runner, hidden_states, input_batch):
    import numpy as np
    import torch

    from vllm.distributed import get_tp_group
    from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    rs = runner.rejection_sampler
    spec = runner.speculator
    st = rs.sampler.sampling_states
    tp, _ = _tp()
    idx = input_batch.idx_mapping_np
    sample_hidden_states = hidden_states[input_batch.logits_indices]
    head = _lm_owner(runner.model).lm_head
    local = head.quant_method.apply(head, sample_hidden_states, bias=None)
    vstart = int(head.shard_indices.org_vocab_start_index)
    vocab = int(local.shape[-1]) * tp
    draft_sampled = input_batch.input_ids[input_batch.logits_indices]
    pos = input_batch.positions[input_batch.logits_indices]
    S = rs.num_speculative_steps
    packet = target_local_packet(local, vstart, input_batch.expanded_idx_mapping, input_batch.expanded_local_pos,
                                 st.temperature.gpu, st.temperature.np, idx, draft_sampled, spec._tc_dl_local, S)
    M = packet.shape[0]
    gathered = get_tp_group().all_gather(packet, dim=0).view(tp, M, packet.shape[1])
    do_top_p = bool(np.any(st.top_p.np[idx] != 1.0))
    k_rows = st.top_k.gpu[input_batch.expanded_idx_mapping]
    p_rows = st.top_p.gpu[input_batch.expanded_idx_mapping] if do_top_p else None
    rows = dict(draft_sampled=draft_sampled, cu_num_logits=input_batch.cu_num_logits,
                idx_mapping=input_batch.idx_mapping, expanded_idx_mapping=input_batch.expanded_idx_mapping,
                temperature=st.temperature.gpu, seeds=st.seeds.gpu, pos=pos,
                num_reqs=input_batch.cu_num_logits.shape[0] - 1, num_steps=S)
    sampled, num_sampled, needs = target_finish(gathered, tp, vocab, k_rows, p_rows, do_top_p, rows,
                                                bool(rs.sampler.use_fp64_gumbel))
    if bool(needs):            # host sync, like the stock compact route's reference_rows.any()
        return None
    so = SamplerOutput(sampled_token_ids=sampled, logprobs_tensors=None, num_nans=None, num_sampled=num_sampled)
    n_s, n_r = get_num_sampled_and_rejected(so.num_sampled, input_batch.seq_lens, input_batch.cu_num_logits,
                                            input_batch.idx_mapping, runner.req_states.prefill_len.gpu)
    import logging

    spec.trace_target_output(input_batch, so, n_s, n_r, sample_hidden_states, logging.getLogger("tc_sampler"))
    return so, n_s, n_r


# ----------------------------------------------------------------------------------------------------------------------
# install
# ----------------------------------------------------------------------------------------------------------------------
def _install_spec():
    mod = sys.modules[TARGET_SPEC]
    cls = mod.EagleSpeculator
    if getattr(cls._sample_draft, "_tc_tpsample", False):
        return
    _ORIG["_sample_draft"] = cls._sample_draft
    _tc_sample_draft._tc_tpsample = True
    cls._sample_draft = _tc_sample_draft
    cls.__init__ = _spec_init_wrapper(cls.__init__)
    _stats["installed"].append("EagleSpeculator._sample_draft/__init__")
    _say("installed EagleSpeculator hooks")


def _install_runner():
    mod = sys.modules[TARGET_RUNNER]
    cls = mod.GPUModelRunner
    if getattr(cls.sample, "_tc_tpsample", False):
        return
    _ORIG["sample"] = cls.sample
    _tc_sample._tc_tpsample = True
    cls.sample = _tc_sample
    _stats["installed"].append("GPUModelRunner.sample")
    _say("installed GPUModelRunner.sample hook")


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
                _say(f"post-import install failed for {fullname}: {exc!r}")

        loader.exec_module = exec_module
        return spec


def install() -> bool:
    if not enabled():
        return False
    try:
        for name, fn in ((TARGET_SPEC, _install_spec), (TARGET_RUNNER, _install_runner)):
            if name in sys.modules:
                fn()
            else:
                sys.meta_path.insert(0, _AfterImport(name, fn))
                _say(f"armed post-import hook for {name}")
        return True
    except Exception as exc:  # noqa: BLE001
        _say(f"install failed: {exc!r}")
        return False


install()


def _base_ext():
    import importlib

    name = os.environ.get("TC_TPSAMPLE_BASE_EXT", "tc_ext.TCExt") or "tc_ext.TCExt"
    if name.startswith("tc_sampler."):
        name = "tc_ext.TCExt"
    mod, _, cls = name.rpartition(".")
    return getattr(importlib.import_module(mod), cls)


_EXT_CLS = None


def __getattr__(name):
    """TCSamplerExt is built on first access (PEP 562), i.e. when vLLM resolves --worker-extension-cls
    tc_sampler.TCSamplerExt, so tc_ext.py may also `import tc_sampler` (enable route B) without a circular import.
    Base class = $TC_TPSAMPLE_BASE_EXT (default tc_ext.TCExt)."""
    global _EXT_CLS
    if name != "TCSamplerExt":
        raise AttributeError(name)
    if _EXT_CLS is None:
        try:
            base = _base_ext()
        except Exception as exc:  # noqa: BLE001  (CPU tests run without the prod packages on sys.path)
            _say(f"base worker extension unavailable ({exc!r}); TCSamplerExt derives from object")
            base = object

        def tc_tpsample_status(self):
            return status()

        _EXT_CLS = type("TCSamplerExt", (base,), {"tc_tpsample_status": tc_tpsample_status, "__module__": __name__})
    return _EXT_CLS
