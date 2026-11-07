"""sb_car Triton kernels (variant X: imported lazily on the first compressed call)."""
import triton
import triton.language as tl


@triton.jit
def _rint(v):
    return tl.where(v >= 0, tl.floor(v + 0.5), tl.ceil(v - 0.5))


@triton.jit(do_not_specialize=["n_blocks", "NB_CHUNK", "C", "STRIDE", "CHUNK0"])
def _quant_pack_kernel(x_ptr, q_ptr, s_ptr, n_blocks, NB_CHUNK, C, STRIDE, CHUNK0,
                       B: tl.constexpr, G: tl.constexpr):
    pid = tl.program_id(0)
    blk = pid * G + tl.arange(0, G)
    bm = blk < n_blocks
    cols = tl.arange(0, B)
    x = tl.load(x_ptr + blk[:, None] * B + cols[None, :], mask=bm[:, None], other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    s16 = (amax * (1.0 / 127.0)).to(tl.float16)
    s = s16.to(tl.float32)
    inv = tl.where(s > 0, 1.0 / s, 0.0)
    q = _rint(x * inv[:, None])
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
    chunk = CHUNK0 + blk // NB_CHUNK
    lb = blk % NB_CHUNK
    qoff = chunk.to(tl.int64) * STRIDE + lb * B
    tl.store(q_ptr + qoff[:, None] + cols[None, :], q, mask=bm[:, None])
    soff = (chunk.to(tl.int64) * STRIDE + C) // 2 + lb
    tl.store(s_ptr + soff, s16, mask=bm)


@triton.jit(do_not_specialize=["rank", "NB_CHUNK", "C", "STRIDE"])
def _reduce_pack_kernel(x_ptr, rq_ptr, rs_ptr, q_ptr, s_ptr, rank, NB_CHUNK, C, STRIDE,
                        W: tl.constexpr, B: tl.constexpr, G: tl.constexpr):
    pid = tl.program_id(0)
    lb = pid * G + tl.arange(0, G)
    bm = lb < NB_CHUNK
    cols = tl.arange(0, B)
    acc = tl.load(x_ptr + rank.to(tl.int64) * C + lb[:, None] * B + cols[None, :], mask=bm[:, None],
                  other=0.0).to(tl.float32)
    for p in tl.static_range(W):
        base = p * STRIDE
        qp = tl.load(rq_ptr + base + lb[:, None] * B + cols[None, :], mask=bm[:, None], other=0).to(tl.float32)
        sp = tl.load(rs_ptr + (base + C) // 2 + lb, mask=bm, other=0.0).to(tl.float32)
        acc += tl.where(p != rank, qp * sp[:, None], 0.0)
    amax = tl.max(tl.abs(acc), axis=1)
    s16 = (amax * (1.0 / 127.0)).to(tl.float16)
    s = s16.to(tl.float32)
    inv = tl.where(s > 0, 1.0 / s, 0.0)
    q = _rint(acc * inv[:, None])
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
    qoff = rank.to(tl.int64) * STRIDE + lb * B
    tl.store(q_ptr + qoff[:, None] + cols[None, :], q, mask=bm[:, None])
    tl.store(s_ptr + (rank.to(tl.int64) * STRIDE + C) // 2 + lb, s16, mask=bm)


@triton.jit(do_not_specialize=["n_blocks", "NB_CHUNK", "C", "STRIDE"])
def _dequant_kernel(q_ptr, s_ptr, y_ptr, n_blocks, NB_CHUNK, C, STRIDE, B: tl.constexpr, G: tl.constexpr):
    pid = tl.program_id(0)
    blk = pid * G + tl.arange(0, G)
    bm = blk < n_blocks
    cols = tl.arange(0, B)
    chunk = blk // NB_CHUNK
    lb = blk % NB_CHUNK
    q = tl.load(q_ptr + chunk.to(tl.int64)[:, None] * STRIDE + lb[:, None] * B + cols[None, :], mask=bm[:, None],
                other=0).to(tl.float32)
    s = tl.load(s_ptr + (chunk.to(tl.int64) * STRIDE + C) // 2 + lb, mask=bm, other=0.0).to(tl.float32)
    tl.store(y_ptr + blk[:, None] * B + cols[None, :], (q * s[:, None]).to(tl.float16), mask=bm[:, None])


