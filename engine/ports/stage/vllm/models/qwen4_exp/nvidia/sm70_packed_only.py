# SPDX-License-Identifier: Apache-2.0
"""Local 2026-10-10 patch: keep ONE copy of the batch-packed FP16 projections.

The Qwen3.8 SM70 FP16 decode lane packs HC (replicated) and GDN in_proj_qkvz
weights into MMA tile order for M2..16, and keeps the row-major original for
M1 and prefill. On a 4x PCIe TP4 box that duplicate costs ~2 GiB per rank.

With VLLM_SM70_QWEN38_PACKED_ONLY=1 the row-major original is freed. Each layer's
``weight`` becomes a view of one shared per-device scratch buffer, and every
consumer that needs row-major weights unpacks the layer's packed copy into the
scratch immediately before use (inside the same opaque op, same stream). The
packing is a pure permutation, so unpacking is exact; this is verified per layer
at load time against the original before it is freed.
"""

import os

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

ENABLED = os.getenv("VLLM_SM70_QWEN38_PACKED_ONLY") == "1"

# kind -> (exposed weight shape, padded rows, row-major view shape, pack permutation)
# Must mirror _pack_hc_batch_weight (rank=None, replicated) and
# _pack_gdn_input_weight exactly; the load-time self-test enforces this.
_KINDS = {
    "hc_down": ((336, 10240), 352, (11, 32, 640, 2, 8), (0, 2, 3, 1, 4)),
    "hc_up": ((10240, 320), 10240, (4, 320, 8, 20, 2, 8), (1, 3, 4, 0, 2, 5)),
    "gdn_qkvz": ((4096, 2560), 4096, (128, 32, 160, 2, 8), (0, 2, 3, 1, 4)),
}
_PACKED_ATTR = {
    "hc_down": "_sm70_qwen38_hc_batch_packed",
    "hc_up": "_sm70_qwen38_hc_batch_packed",
    "gdn_qkvz": "_sm70_qwen38_gdn_packed",
}
_SCRATCH: dict[tuple[str, torch.device], torch.Tensor] = {}
_SCRATCH_KIND: dict[int, str] = {}  # data_ptr of the exposed weight view -> kind
_STATS = {"layers": 0, "bytes": 0}


def _inverse(perm: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(range(len(perm)), key=lambda i: perm[i]))


def _unpack(kind: str, packed: torch.Tensor, scratch: torch.Tensor) -> None:
    _, _, view_shape, perm = _KINDS[kind]
    scratch.view(view_shape).copy_(packed.permute(_inverse(perm)))


def make_packed_only(layer: nn.Module, kind: str) -> None:
    """Free ``layer.weight`` after proving the packed copy reproduces it."""
    if not ENABLED:
        return
    prefix = str(getattr(layer, "prefix", ""))
    if "mtp" in prefix:
        return  # draft layers run M=1 every step; keep their row-major copy
    if getattr(layer, "bias", None) is not None:
        return  # audit engine-patches-8: packed_only_apply is bias-free; never strip a layer that has a bias
    weight = getattr(layer, "weight", None)
    packed = getattr(layer, _PACKED_ATTR[kind], None)
    shape, padded_rows, _, _ = _KINDS[kind]
    if (
        weight is None
        or packed is None
        or tuple(weight.shape) != shape
        or weight.dtype != torch.float16
        or not weight.is_cuda
        or packed.numel() != padded_rows * shape[1]
    ):
        return
    key = (kind, weight.device)
    scratch = _SCRATCH.get(key)
    if scratch is None:
        scratch = torch.empty(
            (padded_rows, shape[1]), dtype=weight.dtype, device=weight.device
        )
        _SCRATCH[key] = scratch
    exposed = scratch[: shape[0]]
    # 2026-10-26 (GGUF arm): the packed copy can still be in flight on another stream when this check runs (flaky
    # FAILED on the same layer, passing on other boots). Wait for ALL streams first; a real mismatch still refuses.
    torch.cuda.synchronize(weight.device)
    _unpack(kind, packed, scratch)
    torch.cuda.synchronize(weight.device)
    if not torch.equal(exposed, weight.data):
        raise RuntimeError(
            f"SM70 packed-only self-test FAILED for {prefix} ({kind}); refusing to "
            "free the original weight. Unset VLLM_SM70_QWEN38_PACKED_ONLY."
        )
    _STATS["layers"] += 1
    _STATS["bytes"] += weight.numel() * weight.element_size()
    layer.weight = nn.Parameter(exposed, requires_grad=False)
    layer._sm70_packed_only_kind = kind
    _SCRATCH_KIND[exposed.data_ptr()] = kind
    if _STATS["layers"] in (1, 48, 96, 132) or _STATS["layers"] % 50 == 0:
        logger.info(
            "SM70 packed-only: %d projections now keep one copy (%.2f GiB freed "
            "per rank so far; last %s).",
            _STATS["layers"],
            _STATS["bytes"] / 2**30,
            prefix,
        )


def ensure_unpacked(weight: torch.Tensor, packed: torch.Tensor | None) -> None:
    """Refill the shared scratch before a row-major consumer reads ``weight``."""
    if not _SCRATCH_KIND:
        return
    kind = _SCRATCH_KIND.get(weight.data_ptr())
    if kind is None:
        return
    if packed is None:
        raise RuntimeError("SM70 packed-only weight read without its packed copy")
    _unpack(kind, packed, _SCRATCH[(kind, weight.device)])


def _packed_only_linear(
    x: torch.Tensor, weight: torch.Tensor, packed: torch.Tensor, kind: str
) -> torch.Tensor:
    _unpack(kind, packed, _SCRATCH[(kind, weight.device)])
    return torch.nn.functional.linear(x, weight)


def _packed_only_linear_fake(
    x: torch.Tensor, weight: torch.Tensor, packed: torch.Tensor, kind: str
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="qwen38_sm70_packed_only_linear",
    op_func=_packed_only_linear,
    fake_impl=_packed_only_linear_fake,
)


def packed_only_apply(layer: nn.Module, x: torch.Tensor) -> torch.Tensor | None:
    kind = getattr(layer, "_sm70_packed_only_kind", None)
    if kind is None:
        return None
    return torch.ops.vllm.qwen38_sm70_packed_only_linear(
        x, layer.weight, getattr(layer, _PACKED_ATTR[kind]), kind
    )
