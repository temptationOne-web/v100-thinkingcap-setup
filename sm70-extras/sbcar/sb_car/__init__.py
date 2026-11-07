"""sb_car (variant X: lazy triton import + instance-level hook on the TP communicator only)
 — sound-barrier P4: compressed (int8 + per-block fp16 scale) TP all-reduce for PREFILL-size messages.

Algorithm (world size W=4, x = fp16 [N] flattened, split into W contiguous chunks of C = N/W elements):
  mode "a2a"  (2x fewer bytes than fp16):
    1. quantize x -> packed buffer (per chunk: C int8 | C/B fp16 scales)          [1 kernel]
    2. all-to-all: chunk p -> rank p (NCCL grouped send/recv)                     [3/4 * N bytes out per rank]
    3. owner r: acc = x_own_fp16 + sum_p deq(chunk r from p) (fp32), quantize      [1 kernel]
    4. all-gather the quantized reduced chunks (NCCL, in-place)                     [3/4 * N bytes out per rank]
    5. every rank dequantizes all W chunks -> fp16 out (bitwise identical on all ranks)  [1 kernel]
  mode "rs16" (fp16 reduce-scatter + int8 all-gather = 0.75x bytes, ONE quantization of the final sum)
  mode "fp16" (fp16 reduce-scatter + fp16 all-gather; pattern check only, numerically = NCCL ring order may differ)

install(): wraps CudaCommunicator.all_reduce at CLASS level (runtime Python behind the opaque vllm::all_reduce custom op;
no traced file edited, no VLLM_* var -> torch.compile cache key unchanged). Only for the TP group of world size 4, fp16,
contiguous, numel*2 >= SB_CAR_MIN_BYTES, and NEVER while a CUDA graph is being captured (decode graphs <= 20 tokens = 100 KiB,
far below the threshold; decode keeps the custom push AR untouched).

Switches (env, not VLLM_*):  SB_CAR=1 (on) | SB_CAR_MODE=a2a|rs16|fp16 | SB_CAR_MIN_BYTES (default 4 MiB) | SB_CAR_BLOCK=64
  SB_CAR_DUMP_DIR + flag file <dir>/dump.flag -> save the next SB_CAR_DUMP_N large AR inputs (CPU copies) per rank.
"""
import os
import sys
import time

import torch


def flag(name, default=""):
    return os.environ.get("SB_CAR_" + name, default)


# --- kernels live in _kern.py, imported on the first compressed call (triton is NOT imported at hook-install time)
_K = {}


def _kernels():
    if not _K:
        from . import _kern

        _K.update(quant=_kern._quant_pack_kernel, reduce=_kern._reduce_pack_kernel, dequant=_kern._dequant_kernel)
    return _K


_G = 16


def _grid(nblocks):
    return ((nblocks + _G - 1) // _G,)


class CompressedAR:
    """Compressed all-reduce over a vLLM PyNcclCommunicator (all ops on the current stream)."""

    def __init__(self, pynccl, rank, world, block=64, mode="a2a"):
        self.nccl = pynccl
        self.rank = rank
        self.world = world
        self.B = block
        self.mode = mode
        # v2 (17:35): PERSISTENT send/recv buffers sized by the FIRST call (= vLLM's profile_run dummy batch at
        # max_num_batched_tokens, so the memory profiler counts them and the KV pool shrinks accordingly). v1 allocated
        # them per call: allocator fragmentation + a KV pool sized ~0.4 GB too optimistic on GPU0 -> slow eager steps
        # (4-user TTFT 2.3 s after long prefills). Larger later calls (rare) fall back to transient buffers.
        self._buf = None
        self._rbuf = None
        self._stream = None  # persistent buffers are only reused on the stream that created them

    def layout(self, n):
        W, B = self.world, self.B
        if n % (W * B):
            return None
        C = n // W
        nb = C // B
        stride = (C + 2 * nb + 255) // 256 * 256
        return C, nb, stride

    def supports(self, x):
        return x.dtype == torch.float16 and x.is_contiguous() and self.layout(x.numel()) is not None

    def all_reduce(self, x):
        K = _kernels()
        _quant_pack_kernel, _reduce_pack_kernel, _dequant_kernel = K['quant'], K['reduce'], K['dequant']
        n = x.numel()
        C, nb, S = self.layout(n)
        W, B, r = self.world, self.B, self.rank
        xf = x.view(-1)
        need = W * S
        cs = torch.cuda.current_stream(x.device).cuda_stream
        if self._buf is None:
            self._buf = torch.empty(need, dtype=torch.uint8, device=x.device)
            self._rbuf = torch.empty(need, dtype=torch.uint8, device=x.device) if self.mode == "a2a" else None
            self._stream = cs
        if need <= self._buf.numel() and cs == self._stream:
            buf = self._buf[:need]
            rbuf = self._rbuf[:need] if self._rbuf is not None else None
        else:
            if not getattr(self, "_warned", False):
                self._warned = True
                _log(f"transient buffers used (need {need} B vs persistent {self._buf.numel()} B, stream "
                     f"{'same' if cs == self._stream else 'OTHER'}); shape={tuple(x.shape)} (logged once)")
            buf = torch.empty(need, dtype=torch.uint8, device=x.device)
            rbuf = torch.empty(need, dtype=torch.uint8, device=x.device) if self.mode == "a2a" else None
        bq, bs = buf.view(torch.int8), buf.view(torch.float16)
        if self.mode == "a2a":
            _quant_pack_kernel[_grid(W * nb)](xf, bq, bs, W * nb, nb, C, S, 0, B=B, G=_G)
            nc = self.nccl
            nc.group_start()
            for p in range(W):
                if p != r:
                    nc.send(buf[p * S:(p + 1) * S], p)
                    nc.recv(rbuf[p * S:(p + 1) * S], p)
            nc.group_end()
            _reduce_pack_kernel[_grid(nb)](xf, rbuf.view(torch.int8), rbuf.view(torch.float16), bq, bs, r, nb, C, S,
                                           W=W, B=B, G=_G)
        elif self.mode == "rs16":
            part = torch.empty(C, dtype=torch.float16, device=x.device)
            self.nccl.reduce_scatter(part, xf)
            _quant_pack_kernel[_grid(nb)](part, bq, bs, nb, nb, C, S, r, B=B, G=_G)
        else:  # fp16 pattern check
            part = torch.empty(C, dtype=torch.float16, device=x.device)
            self.nccl.reduce_scatter(part, xf)
            out = torch.empty_like(xf)
            self.nccl.all_gather(out, part)
            return out.view_as(x)
        self.nccl.all_gather(buf, buf[r * S:(r + 1) * S])
        y = torch.empty_like(xf)
        _dequant_kernel[_grid(W * nb)](bq, bs, y, W * nb, nb, C, S, B=B, G=_G)
        return y.view_as(x)


# ----------------------------------------------------------------------------------------------------------- hook
_STATE = {"installed": False, "calls": 0, "bytes": 0, "logged": set(), "dumped": 0}


def _log(msg):
    print(f"SB-CAR pid {os.getpid()}: {msg}", file=sys.stderr, flush=True)


def install():
    if _STATE["installed"]:
        return _STATE
    from vllm.distributed.device_communicators import cuda_communicator as cc

    min_bytes = int(flag("MIN_BYTES", str(4 << 20)))
    block = int(flag("BLOCK", "64"))
    mode = flag("MODE", "a2a")
    dump_dir = flag("DUMP_DIR", "")
    dump_n = int(flag("DUMP_N", "8"))
    dump_every = int(flag("DUMP_EVERY", "12"))
    on = os.environ.get("SB_CAR", "0") == "1"
    trace = os.environ.get("SB_CAR_TRACE", "0") == "1"  # diagnosis only: log every compressed call (shape, time)
    orig = cc.CudaCommunicator.all_reduce
    if getattr(orig, "_sb_car", False):
        return _STATE

    def _dump(self, x):
        flagf = os.path.join(dump_dir, "dump.flag")
        if not os.path.exists(flagf):
            return
        k = _STATE["dumped"]
        if k >= dump_n:
            return
        _STATE["seen"] = _STATE.get("seen", 0) + 1
        if (_STATE["seen"] - 1) % dump_every:
            return
        r = self.rank_in_group if hasattr(self, "rank_in_group") else torch.distributed.get_rank(self.cpu_group)
        torch.save(x.detach().to("cpu", copy=True), os.path.join(dump_dir, f"ar_r{r}_{k:03d}.pt"))
        _STATE["dumped"] = k + 1

    def all_reduce(self, input_):
        nbytes = input_.numel() * input_.element_size()
        if (nbytes >= min_bytes and self.world_size == 4 and "tp" in self.unique_name
                and input_.dtype == torch.float16 and not torch.cuda.is_current_stream_capturing()):
            if dump_dir:
                _dump(self, input_)
            if on:
                car = getattr(self, "_sb_car_obj", None)
                if car is None:
                    pc = self.pynccl_comm
                    if pc is None or pc.disabled:
                        self._sb_car_obj = False
                        _log("no pynccl comm -> stock path")
                    else:
                        car = CompressedAR(pc, pc.rank, pc.world_size, block=block, mode=mode)
                        self._sb_car_obj = car
                        _log(f"active group={self.unique_name} rank={pc.rank} mode={mode} block={block} "
                             f"min_bytes={min_bytes} first shape={tuple(input_.shape)}")
                if car and car.supports(input_):
                    _STATE["calls"] += 1
                    _STATE["bytes"] += nbytes
                    if trace and car.rank == 0:
                        _log(f"TRACE call {_STATE['calls']} shape={tuple(input_.shape)} t={time.time():.3f}")
                    return car.all_reduce(input_)
        return orig(self, input_)

    all_reduce._sb_car = True
    import types
    orig_init = cc.CudaCommunicator.__init__

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        if "tp" in getattr(self, "unique_name", "") and getattr(self, "world_size", 0) == 4:
            self.all_reduce = types.MethodType(all_reduce, self)
            _log(f"hooked TP communicator {self.unique_name} (instance-level)")

    cc.CudaCommunicator.__init__ = __init__
    _STATE["installed"] = True
    _log(f"installed: SB_CAR={'1' if on else '0'} mode={mode} block={block} min_bytes={min_bytes} dump_dir={dump_dir or '-'}")
    return _STATE
