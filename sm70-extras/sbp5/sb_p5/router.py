"""SB-P5 (b): MoE router GEMV + top-k for the Swift/Qwen3.8 E512 x K2560 FP16 router, decode M=2..16 (runtime hook).

Prod route being replaced (both calls run EAGERLY inside the opaque vllm::moe_forward(_shared) custom op, i.e.
FusedMoE runner._forward_impl -> self.gate(...) -> router.select_experts -> fused_topk; nothing here is Dynamo-traced,
so patching run-time-looked-up attributes changes neither the graphs nor the compile key):
  GEMV : Qwen38SM70FP16LinearMethod.apply -> torch.ops.vllm.qwen38_sm70_fp16_gemv
           M=1 Triton row GEMV (kept: different reduction tree), M=2..4 dense_batch_kernel<2560> (exact FP32 4xK640),
           M>=5 torch F.linear (cuBLAS; VLLM_SM70_MTP_ROUTER_BATCH=0 in prod)
  top-k: fused_topk_router.fused_topk -> Triton _sm70_qwen38_router_topk_kernel (M<=16, packed fp16 keys)
Our kernels (sb_p5_router_C.so) reproduce dense_batch_kernel's arithmetic and the Triton top-k's ordering, exp2,
reduction tree and division bit for bit (see src/sb_p5_router.cu). Whether cuBLAS (M>=5) is ALSO bit-identical to
the dense_batch arithmetic is measured by tests/test_p5_router.py -> restrict SB_P5_ROUTER_MS to the bitwise Ms.

Switches (sb_p5.flag):
  SB_P5_ROUTER=1             enable
  SB_P5_ROUTER_MODE          fused2 (default): wide GEMV kernel + CUDA top-k kernel (2 launches, was 2-3)
                             fused : ONE launch, top-k in a last-block epilogue (self-resetting counter)
                             gemv  : wide GEMV only, stock Triton top-k
                             topk  : stock GEMV, CUDA top-k only
  SB_P5_ROUTER_MS            token counts to take over, e.g. "2-16" (default) or "4,5,8,16"
Hand-off GEMV -> top-k: the hook computes top-k right after the logits (same stream) and stashes the result keyed by
the logits tensor object; the fused_topk hook returns it only if it is called with that very tensor (identity) and
the stock SM70 contract (topk=10, renormalize, softmax, int32 ids, VLLM_SM70_QWEN38_ROUTER_TOPK=1). Otherwise stock.
"""

from __future__ import annotations

import threading

_state: dict = {}
_lock = threading.Lock()

_ROUTER_SUFFIX = ".mlp.gate"
_SLOTS = 512


def _parse_ms(spec: str) -> frozenset:
    out = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return frozenset(m for m in out if 2 <= m <= 16)


def install(log) -> str:
    import torch

    import sb_p5
    import vllm.envs as envs
    import vllm.model_executor.layers.fused_moe.router.fused_topk_router as ftr
    import vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv as g
    from vllm.platforms import current_platform

    with _lock:
        if _state.get("installed"):
            return _state["status"]
        mode = sb_p5.flag("ROUTER_MODE", "fused2")
        if mode not in ("fused2", "fused", "gemv", "topk"):
            raise ValueError(f"bad SB_P5_ROUTER_MODE={mode!r}")
        ms = _parse_ms(sb_p5.flag("ROUTER_MS", "2-16"))
        sb_p5.load_lib("sb_p5_router_C", "sb_p5_router", "gemv_out")
        ops = torch.ops.sb_p5_router
        cls = g.Qwen38SM70FP16LinearMethod
        orig_apply = cls.apply
        orig_topk = ftr.fused_topk
        stash: dict = {"logits": None}
        counters: dict = {}  # device -> int32[_SLOTS] (allocated outside graph capture)
        slot_of: dict = {}
        announced: set = set()

        def _counter(device, key):
            buf = counters.get(device)
            if buf is None:
                if torch.cuda.is_current_stream_capturing():
                    return None, 0
                buf = torch.zeros((_SLOTS,), dtype=torch.int32, device=device)
                counters[device] = buf
            s = slot_of.get((device, key))
            if s is None:
                s = sum(1 for k in slot_of if k[0] == device)
                if s >= _SLOTS:
                    return None, 0
                slot_of[(device, key)] = s
            return buf, s

        def _eligible(layer, x) -> bool:
            w = getattr(layer, "weight", None)
            return bool(
                w is not None
                and x.ndim == 2
                and x.shape[0] in ms
                and x.shape[1] == 2560
                and tuple(w.shape) == (512, 2560)
                and x.dtype == w.dtype == torch.float16
                and x.is_cuda
                and w.device == x.device
                and x.is_contiguous()
                and w.is_contiguous()
                and x.data_ptr() % 16 == 0
                and w.data_ptr() % 16 == 0
                and str(getattr(layer, "prefix", "")).endswith(_ROUTER_SUFFIX)
                and getattr(layer, "_sm70_packed_only_kind", None) is None
                and not envs.VLLM_BATCH_INVARIANT
                and not torch.backends.cuda.matmul.allow_fp16_accumulation
                and g.use_sm70_decode_graph_semantics()
                and current_platform.is_device_capability(70)
            )

        def sb_apply(self, layer, x, bias=None):
            # allocate the fused-mode counter on the first eager call (profile run) so it is never made in capture
            if mode == "fused" and not counters and x.is_cuda and not torch.cuda.is_current_stream_capturing():
                _counter(x.device, -1)
            if bias is not None or not _eligible(layer, x):
                return orig_apply(self, layer, x, bias)
            m = x.shape[0]
            w = layer.weight
            if mode == "topk":
                logits = orig_apply(self, layer, x, bias)
                if not (logits.is_contiguous() and logits.dtype == torch.float16 and logits.data_ptr() % 16 == 0):
                    return logits
            else:
                logits = x.new_empty((m, 512))
            if mode == "gemv":
                ops.gemv_out(logits, x, w)
                stash["logits"] = None
            else:
                tw = torch.empty((m, 10), dtype=torch.float32, device=x.device)
                ti = torch.empty((m, 10), dtype=torch.int32, device=x.device)
                tei = torch.empty((m, 10), dtype=torch.int32, device=x.device)
                used = mode
                if mode == "fused":
                    buf, s = _counter(x.device, w.data_ptr())
                    if buf is None:
                        used = "fused2"
                    else:
                        ops.fused_out(logits, tw, ti, tei, x, w, buf, s)
                if used == "fused2":
                    ops.gemv_out(logits, x, w)
                    ops.topk_out(tw, ti, tei, logits)
                elif used == "topk":
                    ops.topk_out(tw, ti, tei, logits)
                stash["logits"] = logits
                stash["out"] = (tw, ti, tei)
            if m not in announced:
                announced.add(m)
                log.info("SB-P5 router active: mode=%s M=%d (layer %s)", mode, m, getattr(layer, "prefix", "?"))
            return logits

        def sb_fused_topk(hidden_states, gating_output, topk, renormalize, indices_type=None, scoring_func="softmax"):
            held = stash.get("logits")
            if held is not None:
                stash["logits"] = None
                if (
                    gating_output is held
                    and topk == 10
                    and renormalize
                    and scoring_func == "softmax"
                    and indices_type in (None, torch.int32)
                    and envs.VLLM_SM70_QWEN38_ROUTER_TOPK
                    and hidden_states.size(0) == gating_output.size(0)
                ):
                    if "hit" not in announced:
                        announced.add("hit")
                        log.info("SB-P5 router top-k hand-off active (first M=%d)", gating_output.size(0))
                    return stash.pop("out")
                if "miss" not in announced:  # diagnostic: computed a top-k nobody consumed -> stock path used
                    announced.add("miss")
                    log.warning("SB-P5 router top-k hand-off MISSED (tensor identity/contract) -> stock top-k")
            return orig_topk(hidden_states, gating_output, topk, renormalize, indices_type, scoring_func)

        sb_apply._sb_p5_orig = orig_apply
        sb_fused_topk._sb_p5_orig = orig_topk
        cls.apply = sb_apply
        ftr.fused_topk = sb_fused_topk
        _state.update(installed=True, cls=cls, ftr=ftr, mode=mode, ms=ms, counters=counters)
        lo, hi = (min(ms), max(ms)) if ms else (0, 0)
        _state["status"] = f"router(mode={mode}, M={lo}..{hi} n={len(ms)})"
        return _state["status"]


def uninstall() -> None:
    with _lock:
        if not _state.get("installed"):
            return
        cls, ftr = _state["cls"], _state["ftr"]
        if getattr(cls.apply, "_sb_p5_orig", None) is not None:
            cls.apply = cls.apply._sb_p5_orig
        if getattr(ftr.fused_topk, "_sb_p5_orig", None) is not None:
            ftr.fused_topk = ftr.fused_topk._sb_p5_orig
        _state.clear()
