"""SB-P5 (c): static draft vocabulary for the V2-runner MTP drafter (Swift-1.5 Flash-Next, TP4, SM70).

PROD DRAFT PATH (V2 runner, method mtp, draft_sample_method greedy, use_local_argmax_reduction auto-on for Qwen4ExpMTP):
  EagleSpeculator._sample_draft -> Qwen4ExpMTP.get_top_tokens -> LogitsProcessor.get_top_tokens(head=_sm70_draft_head)
  -> MTPQPN8Head.maybe_get_sm70_lm_head_top1_pair: fp8_qpn8_gemm over the rank's FULL 62,080-row shard (~192 us per
  draft pass) + qwen38_mtp_local_top1 -> (logit, global id) pair -> custom top1 all-reduce -> global argmax.
  MTPQPN8Head is a DRAFT-ONLY channel-QPN8 copy (159 MB/rank) of the shared lm_head; the target keeps its own fp16 head.

WHY NOT "top-N rows of each local shard": Swift's output tokens are 96-98 % in rank 0's shard (ids 0..62,079 = the
  most frequent BPE merges), so a per-shard subset only shortens ranks 1-3 while rank 0 (the slowest rank gates the
  top1 all-reduce) still needs most of its rows (offline: rank-0 top-16k still misses 6-17 % of tokens).
THIS HOOK = RE-PARTITIONED DRAFT HEAD: the draft-only pack is rebuilt so that each rank holds R = N/4 rows of the
  GLOBAL top-N (interleaved: rank r gets ranked ids r, r+4, r+8, ...). Rows come from an all-gather of the fp16 lm_head
  shards at load time (inside prepare_mtp_qpn8_head, called once on every rank from the Eagle loader after checkpoint
  load + head sharing, BEFORE KV sizing and CUDA-graph capture). Per-row channel quantization uses the stock arithmetic
  (row-independent) -> every kept row has bit-identical codes/scale -> identical logit for the same hidden state.
  The draft GEMM then runs on R rows per rank (balanced), local top1 -> global id via a table -> stock cross-rank reduce.
  VRAM: the 62,080-row pack is replaced by an R-row pack -> SAVES (62,080-R)*2,560 B per rank (R=24,576: -96 MB/rank,
  ~ +6k KV tokens). Load-time transient ~+0.5 GB (freed + empty_cache before KV sizing).
  Full-logits consumers (probabilistic drafts, probes, rows > 8, any non-pair fallback) use the TARGET fp16 head
  (self.head.quant_method.apply) in the original column order -> always correct, only slower; prod greedy never uses it.
OUTPUT NEUTRALITY: prod drafts are greedy one-hot (draft_logits=None). The V2 rejection sampler accepts a greedy
  request's draft iff target_argmax == draft and otherwise emits target_argmax; for sampled requests it accepts with
  p(x) (q one-hot) and resamples from p with x removed. Either way the emitted distribution is the target's, whatever
  the draft is. Only acceptance changes, and only when the full-vocab draft argmax is outside the subset.

Switches (not VLLM_*): SB_P5_DVOCAB=1, SB_P5_DVOCAB_N=<total rows, default 98304 -> 24,576/rank (multiple of 128)>,
  SB_P5_DVOCAB_RANKING=<.pt with "global_ranking" int32[248320]> (default p5/build/swift_draft_ranking.pt from
  tests/build_swift_draft_ranking.py = Swift's own output-token counts, then lowest id; missing file -> lowest-id order).
"""

from __future__ import annotations

import os
import types

from . import flag

_TARGET_MOD = "vllm.models.qwen4_exp.nvidia.sm70_mtp_head"
_RANKING_DEFAULT = "/opt/1cat/soundbarrier/p5/build/swift_draft_ranking.pt"
VOCAB = 248320
TP = 4
SHARD = 62080
HIDDEN = 2560
ADDED = (248044, 248077)  # Swift/Qwen3.5 added special tokens (<|endoftext|> .. tokenizer end): always in the subset
_state: dict = {}


def rows_per_rank() -> int:
    n_total = int(flag("DVOCAB_N", "98304"))
    r = -(-max(n_total, 128) // TP)
    r = -(-r // 32) * 32  # QPN8 GEMM needs N % 32 == 0
    return min(r, SHARD)


def load_global_ranking():
    """CPU int64 [VOCAB] ranking (best first). Fallback: lowest id first (BPE merge order ~ corpus frequency)."""
    import torch

    path = flag("DVOCAB_RANKING", _RANKING_DEFAULT)
    if os.path.exists(path):
        d = torch.load(path, map_location="cpu", weights_only=False)
        g = (d["global_ranking"] if isinstance(d, dict) else d).to(torch.int64).flatten()
        if g.numel() != VOCAB or torch.unique(g).numel() != VOCAB:
            raise ValueError(f"bad ranking {path}: need a permutation of {VOCAB} ids")
        return g, path
    return torch.arange(VOCAB, dtype=torch.int64), "lowest-id (no ranking file)"


def assign(ranking, r_rows: int):
    """CPU: per-rank sorted global ids [TP, r_rows]. Subset = ADDED + top ranked ids; rank r gets entries r::TP."""
    import torch

    forced = torch.arange(*ADDED, dtype=torch.int64)
    seen = torch.zeros(VOCAB, dtype=torch.bool)
    seen[forced] = True
    rest = ranking[~seen[ranking]]
    sub = torch.cat([forced, rest])[: TP * r_rows]
    per = torch.stack([torch.sort(sub[r::TP]).values for r in range(TP)])  # sorted -> coalesced gather order
    assert per.shape == (TP, r_rows) and torch.unique(per).numel() == TP * r_rows
    return per


def _quantize_rows_like_stock(w16):
    """Same arithmetic as sm70_online_qpn8.prepare_channel_qpn8_weight (per-row amax/448 scale, fp8 e4m3 codes);
    amax is exact and the rest elementwise -> bit-identical per-row codes/scales regardless of chunking/row order."""
    import torch

    from vllm import _sm70_ops as sm70_ops

    rows = w16.shape[0]
    qweight = torch.empty_like(w16, dtype=torch.float8_e4m3fn)
    scales = torch.empty((rows, 1), dtype=torch.float32, device=w16.device)
    for b in range(0, rows, 4096):
        e = min(b + 4096, rows)
        values = w16[b:e].float()
        scale = values.abs().amax(dim=1, keepdim=True) / 448
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        qweight[b:e].copy_((values / scale).to(torch.float8_e4m3fn))
        scales[b:e].copy_(scale)
        del values
    codes, packed = sm70_ops.fp8_qpn8_prepare_sm70(qweight.contiguous(), scales.contiguous())
    return codes, packed


def gather_rows(local_w16, my_ids, tp_rank: int, group, chunk: int = 2048):
    """All-gather the fp16 head shards chunk by chunk (collective: every rank calls it identically) and keep the rows
    whose global id is in my_ids (CPU int64 sorted). Returns [len(my_ids), HIDDEN] fp16 on the device."""
    import torch
    import torch.distributed as dist

    dev = local_w16.device
    out = torch.empty((my_ids.numel(), HIDDEN), dtype=torch.float16, device=dev)
    shard_of = my_ids // SHARD
    off = my_ids % SHARD
    for c0 in range(0, SHARD, chunk):
        c1 = min(c0 + chunk, SHARD)
        ch = c1 - c0
        buf = torch.empty((TP * ch, HIDDEN), dtype=torch.float16, device=dev)
        dist.all_gather_into_tensor(buf, local_w16[c0:c1].contiguous(), group=group)
        sel = ((off >= c0) & (off < c1)).nonzero().flatten()
        if sel.numel():
            src = (shard_of[sel] * ch + (off[sel] - c0)).to(dev)
            out.index_copy_(0, sel.to(dev), buf.index_select(0, src))
        del buf
    return out


def _sb_apply(self, layer, x, bias=None):
    """Full logits (original order) for every non-pair consumer: the target fp16 head path (exact, slower)."""
    return self.head.quant_method.apply(self.head, x, bias)


def _sb_top1_pair(self, hidden_states, bias=None):
    """Greedy draft: (logit, global id) of this rank's best token among its R subset rows."""
    import torch

    from vllm import _sm70_ops as ops

    if (
        hidden_states.ndim != 2
        or hidden_states.dtype != torch.float16
        or not hidden_states.is_cuda
        or not 1 <= hidden_states.shape[0] <= 8
        or bias is not None
        or not hasattr(torch.ops._C, "qwen38_mtp_local_top1_sm70_out")
    ):
        return None  # -> get_top_tokens falls back to apply() above (full target-head logits, correct)
    rows = hidden_states.shape[0]
    r = self._sb_rows
    logits = hidden_states.new_empty(rows, r)
    ops.fp8_qpn8_gemm_sm70_out(logits, hidden_states.contiguous(), self.codes, self.scales, 8, 2, True, False)
    pairs = logits.new_empty((rows, 2), dtype=torch.float32)
    partial = logits.new_empty((rows, (r + 511) // 512, 2), dtype=torch.float32)
    torch.ops._C.qwen38_mtp_local_top1_sm70_out(pairs, partial, logits, r, 0)
    pairs[:, 1] = self._sb_ids_f32.index_select(0, pairs[:, 1].to(torch.int64))
    return pairs


def convert_view(view, ranking, tp_rank: int, group) -> str:
    """Collective on the TP group. Rebuild one MTPQPN8Head as the re-partitioned R-row subset pack (all-or-nothing)."""
    import torch

    r_rows = rows_per_rank()
    per = assign(ranking, r_rows)
    my_ids = per[tp_rank]
    w_sub = gather_rows(view.weight, my_ids, tp_rank, group)
    codes, scales = _quantize_rows_like_stock(w_sub)
    del w_sub
    ids_f32 = my_ids.to(torch.float32).to(view.weight.device)  # float32 exact for ids < 2^24
    # all computed -> switch the instance in one go
    view.codes = codes  # buffer re-assignment frees the 62,080-row stock pack
    view.scales = scales
    view._sb_rows, view._sb_ids_f32, view._sb_ids_cpu = r_rows, ids_f32, my_ids
    view.apply = types.MethodType(_sb_apply, view)
    view.maybe_get_sm70_lm_head_top1_pair = types.MethodType(_sb_top1_pair, view)
    torch.cuda.empty_cache()  # hand the transient back before KV sizing
    _state.update(rows=r_rows, rank=tp_rank)
    return (f"active: re-partitioned draft vocab {r_rows} rows/rank (total {TP * r_rows} of {VOCAB}); "
            f"draft pack {SHARD}->{r_rows} rows = -{(SHARD - r_rows) * HIDDEN / 2**20:.0f} MiB/rank")


def install(log) -> str:
    import importlib

    mod = importlib.import_module(_TARGET_MOD)
    if getattr(mod.prepare_mtp_qpn8_head, "_sb_p5_orig", None) is not None:
        return "already installed"
    ranking, src = load_global_ranking()  # CPU; identical on every rank
    orig = mod.prepare_mtp_qpn8_head

    def sb_prepare_mtp_qpn8_head(head, *args, **kwargs):
        # 2026-10-25 compat: upstream 6a143197e calls prepare_mtp_qpn8_head(head, hot, ranking_path) (native FR-Spec
        # hot vocab). Pass the extras through; with hot=0 (default) the stock view is full-vocab and we convert it.
        if args and int(args[0] or 0) > 0 or int(kwargs.get("hot", 0) or 0) > 0:
            log.warning("SB-P5 dvocab: native sm70_draft_hot_vocab is set -> SB-P5 dvocab skipped (stock hot vocab)")
            return orig(head, *args, **kwargs)
        view = orig(head, *args, **kwargs)
        if view is None:  # stock gate (shape/arch/TP) - same decision on every rank
            log.warning("SB-P5 dvocab: stock draft QPN8 head not prepared -> stock path")
            return view
        from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size, get_tp_group

        si = view.shard_indices
        if (
            get_tensor_model_parallel_world_size() != TP
            or getattr(si, "num_added_elements", 0)
            or getattr(si, "num_org_vocab_padding", 0)
            or int(si.org_vocab_end_index) - int(si.org_vocab_start_index) != SHARD
            or int(si.org_vocab_start_index) != get_tensor_model_parallel_rank() * SHARD
        ):
            log.warning("SB-P5 dvocab: shard layout is not 4 x 62,080 -> stock path")
            return view
        # From here every rank runs the same collectives. A failure on one rank would leave the others waiting in
        # all_gather, so everything that can fail deterministically was checked above / at install time.
        status = convert_view(view, ranking, get_tensor_model_parallel_rank(), get_tp_group().device_group)
        log.info("SB-P5 dvocab %s [ranking %s]", status, src)
        return view

    sb_prepare_mtp_qpn8_head._sb_p5_orig = orig
    mod.prepare_mtp_qpn8_head = sb_prepare_mtp_qpn8_head
    return f"hooked prepare_mtp_qpn8_head (N={TP * rows_per_rank()} total, {rows_per_rank()}/rank, ranking {src})"


def uninstall() -> None:
    import importlib

    mod = importlib.import_module(_TARGET_MOD)
    orig = getattr(mod.prepare_mtp_qpn8_head, "_sb_p5_orig", None)
    if orig is not None:
        mod.prepare_mtp_qpn8_head = orig
