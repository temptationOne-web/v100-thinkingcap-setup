// TC-GLUE (2026-10-26): ThinkingCap (Qwen3.8 Flash-Next / Qwen4Exp, TP4, fp16, SM70) sync glue.
// One kernel per hyper-connection boundary at decode = two-shot all-reduce (ARX protocol) + HC combine + grouped Gemma
// RMSNorm, BIT-IDENTICAL to the live path:
//   A boundary (after attention out_proj):  ar = AR(P)                         [live: push / ARX kernel]
//   B boundary (after the MoE):             ar = AR(half(shared + routed))     [live: sm70_cross_device_reduce_sum2_1stage_push]
//   then, per HC stream b (4 streams of 2560):                                  [live: Triton _hc_combine_norm_kernel]
//       g_b   = 2*sigmoid(f32(inj_b)/4)            (SASS: FFMA(x,-0.25,0); FMUL log2e; MUFU.EX2; FADD 1; div.full -> MUFU.RCP; FADD s+s)
//       o     = f16(fma(g_b, f32(ar), f32(res_b)))  -> res_out (fp16 multi-stream residual)
//       ss    = sum of squares of f32(o) in the Triton reduction tree of the production kernel (per thread: acc = e1*e1,
//               fma(e0), fma(e2..e7); warp butterfly xor 16,8,4,2,1; tile/warp sums in the order of the cached PTX)
//       rstd  = rsqrt.approx.ftz(FFMA(ss, 0x39cccccd (= rcp 2560, ptxas-contracted with the eps add), 1e-6))
//       xn    = f16(fma(y, f32(w), y)) with y = rstd*o         -> normed multi-stream input of the HC projection
//   TREE 512 : the production kernel for M != 1 (BLOCK_SIZE 512, 8x512 tile, warpsPerCTA [2,2]):
//              T_k = W[2k] + W[2k+1] (k = 0..4, W = warp sums over packs 32w..32w+31), total = ((T0+T2)+T4) + (T1+T3)
//   TREE 1024: the production kernel for M == 1 (BLOCK_SIZE 1024, 4x1024 tile, warpsPerCTA [1,4]):
//              T0 = (W0+W2)+(W1+W3), T1 = (W4+W6)+(W5+W7), T2 = W8+W9, total = (T0+T1)+T2
//   (masked tiles of the Triton kernel contribute exact +0 and are omitted.)
// All-reduce arithmetic = live: half(a+b) per rank (sum2), then the owner computes half(((f32 x0 + x1) + x2) + x3) in
// rank order; 16-byte packs with the fp16 NaN 0x7f7f empty sentinel (real 0x7f7f escaped to 0x7e00), reduce-scatter to
// the quarter owner, all-gather of the owner's bits (half the PCIe wire bytes of the one-shot push kernels).
// One CTA (320 threads = 10 warps) per row, thread t owns 16-byte pack t of the 2560-wide row in every phase.
// Built with --fmad=false: every FMA is explicit (__fmaf_rn) exactly where the production SASS has an FFMA.
// Rows are independent (one CTA per row, fixed tree) -> deterministic and batch invariant for a given tree.
// MODE 0: all-reduce only (arout). MODE 1: + combine (res_out). MODE 2: + norm (res_out, xn).
#include <torch/all.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <mutex>
#include <vector>

#define TCG_CHECK(cmd)                                                                           \
  do {                                                                                           \
    cudaError_t e_ = (cmd);                                                                      \
    TORCH_CHECK(e_ == cudaSuccess, "tc_glue CUDA error ", cudaGetErrorString(e_), " at ", __FILE__, \
                ":", __LINE__);                                                                  \
  } while (0)

namespace tcg {

constexpr int NR = 4;
constexpr int H = 2560;           // block / HC stream width
constexpr int HC = 4;             // streams
constexpr int KD = HC * H;        // 10240 multi-stream width
constexpr int PACKS = H / 8;      // 320 16-byte packs per row
constexpr int QP = PACKS / NR;    // 80 packs per quarter
constexpr int NT = PACKS;         // threads per CTA = 10 warps
constexpr int NW = NT / 32;       // 10
constexpr int MAXM = 40;          // rows per call (HCX_BATCH_M)
constexpr uint16_t SENT = 0x7f7f;
constexpr int SENT_BYTE = 0x7f;
constexpr int64_t HDR = 256;
constexpr int64_t SLOT = (int64_t)MAXM * QP * 16;     // bytes per (epoch, peer) slot set = 51200
constexpr int64_t RS_OFF = HDR;                       // [2][4][SLOT]
constexpr int64_t AG_OFF = RS_OFF + 2 * NR * SLOT;    // [2][4][SLOT]
constexpr int64_t BUF_BYTES = AG_OFF + 2 * NR * SLOT; // 819456
constexpr uint32_t RCP_H = 0x39cccccdu;               // production SASS: MOV R, 0x39cccccd (rcp of 2560)
constexpr float EPS = 1e-06f;                          // 0x358637BD in the production SASS

struct Ptrs {
  char* p[NR];
};

__device__ __forceinline__ uint4 ldv(const void* a) {
  uint4 b;
  asm volatile("ld.volatile.global.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w)
               : "l"(a));
  return b;
}
__device__ __forceinline__ void stv(void* a, uint4 b) {
  asm volatile("st.volatile.global.v4.b32 [%4], {%0,%1,%2,%3};" ::"r"(b.x), "r"(b.y), "r"(b.z), "r"(b.w), "l"(a)
               : "memory");
}
__device__ __forceinline__ bool has_sent(uint4 b) {
  uint32_t w[4] = {b.x, b.y, b.z, b.w};
  bool r = false;
#pragma unroll
  for (int i = 0; i < 4; ++i) r |= ((w[i] & 0xffffu) == SENT) | ((w[i] >> 16) == SENT);
  return r;
}
__device__ __forceinline__ uint4 escape(uint4 b) {
  uint32_t w[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint32_t lo = w[i] & 0xffffu, hi = w[i] >> 16;
    if (lo == SENT) lo = 0x7e00u;
    if (hi == SENT) hi = 0x7e00u;
    w[i] = lo | (hi << 16);
  }
  return make_uint4(w[0], w[1], w[2], w[3]);
}
__device__ __forceinline__ uint4 sent4() { return make_uint4(0x7f7f7f7fu, 0x7f7f7f7fu, 0x7f7f7f7fu, 0x7f7f7f7fu); }

// live push / sum2 / ARX arithmetic: A = f32(v0); A += f32(v1); A += f32(v2); A += f32(v3); half(A)
__device__ __forceinline__ uint4 reduce4(const uint4 (&v)[NR]) {
  float acc[8];
  {
    const __half* h = reinterpret_cast<const __half*>(&v[0]);
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = __half2float(h[i]);
  }
#pragma unroll
  for (int r = 1; r < NR; ++r) {
    const __half* h = reinterpret_cast<const __half*>(&v[r]);
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = __fadd_rn(acc[i], __half2float(h[i]));
  }
  uint4 o;
  __half* oh = reinterpret_cast<__half*>(&o);
#pragma unroll
  for (int i = 0; i < 8; ++i) oh[i] = __float2half_rn(acc[i]);
  return o;
}

// sum2 kernel: packed_assign_add(value_a, value_b) = __hadd per element (== half(f32(a)+f32(b)))
__device__ __forceinline__ uint4 add_pack(uint4 a, uint4 b) {
  const __half* ha = reinterpret_cast<const __half*>(&a);
  const __half* hb = reinterpret_cast<const __half*>(&b);
  uint4 o;
  __half* oh = reinterpret_cast<__half*>(&o);
#pragma unroll
  for (int i = 0; i < 8; ++i) oh[i] = __hadd(ha[i], hb[i]);
  return o;
}

__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
// Triton's `div.full.f32 r, 1.0, x` with the IMMEDIATE numerator: ptxas lowers it to the reciprocal special case
// (FSETP.GT 8.5e37; @P FMUL 0.25; MUFU.RCP; FSEL; FMUL), not the general a/b path.
__device__ __forceinline__ float rcp_full(float x) {
  float r;
  asm("div.full.f32 %0, 0f3F800000, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ float ex2_approx(float a) {
  float r;
  asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ float rsqrt_approx_ftz(float a) {
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ float shfl_xor(float v, int o) { return __shfl_xor_sync(0xffffffffu, v, o, 32); }

// Triton: inj = 2.0 * sigmoid(f32(inj) / 4)  (production SASS: FFMA(x,-0.25,RZ); FMUL log2e; [ex2 lowering] MUFU.EX2;
// FADD 1; [div.full lowering] MUFU.RCP ...; FADD s+s)
__device__ __forceinline__ float gate2(float x) {
  // The production PTX verbatim (one asm block, so ptxas sees the same instruction sequence and lowers it the same way:
  // FFMA(x,-0.25,RZ); FMUL log2e; FSETP/-126 scaling + MUFU.EX2; FADD 1; FSETP 8.5e37 + MUFU.RCP + FSEL/FMUL; FADD s+s).
  float g;
  asm("{\n\t.reg .f32 a, b, c, d, e, f;\n\t"
      "div.full.f32 a, %1, 0f40800000;\n\t"
      "sub.f32 b, 0f00000000, a;\n\t"
      "mul.f32 c, b, 0f3FB8AA3B;\n\t"
      "ex2.approx.f32 d, c;\n\t"
      "add.f32 e, d, 0f3F800000;\n\t"
      "div.full.f32 f, 0f3F800000, e;\n\t"
      "add.f32 %0, f, f;\n\t}"
      : "=f"(g)
      : "f"(x));
  return g;
}

template <int TREE>
__device__ __forceinline__ float tree_total(const float* W) {
  if constexpr (TREE == 512) {
    const float T0 = __fadd_rn(W[0], W[1]), T1 = __fadd_rn(W[2], W[3]), T2 = __fadd_rn(W[4], W[5]),
                T3 = __fadd_rn(W[6], W[7]), T4 = __fadd_rn(W[8], W[9]);
    const float even = __fadd_rn(__fadd_rn(T0, T2), T4);  // warp-row 0: ((T0 + T2) + T4) + 0
    const float odd = __fadd_rn(T1, T3);                  // warp-row 1: (T1 + T3) + 0 + 0
    return __fadd_rn(even, odd);
  } else {
    const float T0 = __fadd_rn(__fadd_rn(W[0], W[2]), __fadd_rn(W[1], W[3]));
    const float T1 = __fadd_rn(__fadd_rn(W[4], W[6]), __fadd_rn(W[5], W[7]));
    const float T2 = __fadd_rn(W[8], W[9]);  // (W8 + 0) + (W9 + 0)
    return __fadd_rn(__fadd_rn(T0, T1), T2);
  }
}

// Combine (+ norm) of one row given this thread's all-reduced pack `val` (elements 8t..8t+7 of the block row).
// res/res_out/xn point at the row (KD wide), w at the full [KD] weight, g = the 4 gate values.
template <int MODE, int TREE>
__device__ __forceinline__ void epilogue(uint4 val, const uint4 (&resv)[HC], const float (&g)[HC], const __half* __restrict__ w,
                                         __half* __restrict__ res_out, __half* __restrict__ xn, float (*red)[NW],
                                         float* rstd_s) {
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const __half* vh = reinterpret_cast<const __half*>(&val);
  float B[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) B[j] = __half2float(vh[j]);
  float of[HC][8];
#pragma unroll
  for (int b = 0; b < HC; ++b) {
    const __half* rh = reinterpret_cast<const __half*>(&resv[b]);
    uint4 o;
    __half* oh = reinterpret_cast<__half*>(&o);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const __half h = __float2half_rn(__fmaf_rn(g[b], B[j], __half2float(rh[j])));
      oh[j] = h;
      of[b][j] = __half2float(h);
    }
    *reinterpret_cast<uint4*>(res_out + b * H + 8 * t) = o;
  }
  if constexpr (MODE == 1) return;
  float acc[HC];
#pragma unroll
  for (int b = 0; b < HC; ++b) {
    acc[b] = __fmul_rn(of[b][1], of[b][1]);
    acc[b] = __fmaf_rn(of[b][0], of[b][0], acc[b]);
#pragma unroll
    for (int j = 2; j < 8; ++j) acc[b] = __fmaf_rn(of[b][j], of[b][j], acc[b]);
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) acc[b] = __fadd_rn(acc[b], shfl_xor(acc[b], o));
  }
  if (lane == 0) {
#pragma unroll
    for (int b = 0; b < HC; ++b) red[b][warp] = acc[b];
  }
  __syncthreads();
  if (t < HC) {
    const float total = tree_total<TREE>(red[t]);
    rstd_s[t] = rsqrt_approx_ftz(__fmaf_rn(total, __int_as_float(RCP_H), EPS));
  }
  __syncthreads();
#pragma unroll
  for (int b = 0; b < HC; ++b) {
    const float rstd = rstd_s[b];
    const uint4 wv = *reinterpret_cast<const uint4*>(w + b * H + 8 * t);
    const __half* wh = reinterpret_cast<const __half*>(&wv);
    uint4 o;
    __half* oh = reinterpret_cast<__half*>(&o);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float y = __fmul_rn(rstd, of[b][j]);
      oh[j] = __float2half_rn(__fmaf_rn(y, __half2float(wh[j]), y));
    }
    *reinterpret_cast<uint4*>(xn + b * H + 8 * t) = o;
  }
}

__device__ __forceinline__ void finish_epoch(uint32_t* ctrl, uint32_t e) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (gridDim.x == 1) {
      ctrl[0] = e ^ 1u;
    } else {
      __threadfence();
      uint32_t prev = atomicAdd(&ctrl[1], 1u);
      if (prev == gridDim.x - 1) {
        ctrl[1] = 0u;
        ctrl[0] = e ^ 1u;
      }
    }
  }
}

__device__ __forceinline__ void load_gates(const __half* __restrict__ inj, int64_t inj_stride, int m, float (&g)[HC]) {
#pragma unroll
  for (int b = 0; b < HC; ++b) g[b] = gate2(__half2float(inj[(int64_t)m * inj_stride + b]));
}

// in1 == nullptr: A (single partial). in1 != nullptr: B (half(in0 + in1) first, like the sum2 kernel).
template <int MODE, int TREE>
__global__ void __launch_bounds__(NT) fused_kernel(Ptrs P, int rank, const uint4* __restrict__ in0,
                                                   const uint4* __restrict__ in1, const __half* __restrict__ res,
                                                   const __half* __restrict__ inj, int64_t inj_stride,
                                                   const __half* __restrict__ w, uint4* __restrict__ arout,
                                                   __half* __restrict__ res_out, __half* __restrict__ xn) {
  __shared__ float red[HC][NW];
  __shared__ float rstd_s[HC];
  const int m = blockIdx.x, t = threadIdx.x;
  char* L = P.p[rank];
  uint32_t* ctrl = reinterpret_cast<uint32_t*>(L);
  const uint32_t e = *reinterpret_cast<volatile uint32_t*>(ctrl);
  const int64_t rowpk = (int64_t)m * QP;
  const int q = t / QP, off = t - q * QP;
  uint4 v = in0[(int64_t)m * PACKS + t];
  if (in1 != nullptr) v = add_pack(v, in1[(int64_t)m * PACKS + t]);
  stv(P.p[q] + RS_OFF + (int64_t)(e * NR + rank) * SLOT + (rowpk + off) * 16, escape(v));
  // prefetch the local operands while the network works
  uint4 resv[HC];
  float g[HC];
  if constexpr (MODE >= 1) {
#pragma unroll
    for (int b = 0; b < HC; ++b) resv[b] = *reinterpret_cast<const uint4*>(res + (int64_t)m * KD + b * H + 8 * t);
    load_gates(inj, inj_stride, m, g);
  }
  uint4 val;
  if (q == rank) {
    uint4 vv[NR];
    char* base = L + RS_OFF + (int64_t)e * NR * SLOT + (rowpk + off) * 16;
    while (true) {
      bool empty = false;
#pragma unroll
      for (int s = 0; s < NR; ++s) {
        vv[s] = ldv(base + (int64_t)s * SLOT);
        empty |= has_sent(vv[s]);
      }
      if (!empty) break;
    }
#pragma unroll
    for (int s = 0; s < NR; ++s) stv(base + (int64_t)s * SLOT, sent4());
    val = reduce4(vv);
    const uint4 rp = escape(val);
#pragma unroll
    for (int k = 1; k < NR; ++k) {
      const int d = (rank + k) & (NR - 1);
      stv(P.p[d] + AG_OFF + (int64_t)(e * NR + rank) * SLOT + (rowpk + off) * 16, rp);
    }
  } else {
    char* a = L + AG_OFF + (int64_t)(e * NR + q) * SLOT + (rowpk + off) * 16;
    do {
      val = ldv(a);
    } while (has_sent(val));
    stv(a, sent4());
  }
  if constexpr (MODE == 0) {
    arout[(int64_t)m * PACKS + t] = val;
  } else {
    epilogue<MODE, TREE>(val, resv, g, w, res_out + (int64_t)m * KD, xn == nullptr ? nullptr : xn + (int64_t)m * KD, red,
                         rstd_s);
  }
  finish_epoch(ctrl, e);
}

// Local (no communication): ar = already all-reduced fp16 [M, H]. Fallback (any M) and tests.
template <int MODE, int TREE>
__global__ void __launch_bounds__(NT) local_kernel(const uint4* __restrict__ ar, const __half* __restrict__ res,
                                                   const __half* __restrict__ inj, int64_t inj_stride,
                                                   const __half* __restrict__ w, __half* __restrict__ res_out,
                                                   __half* __restrict__ xn, float* __restrict__ dbg) {
  __shared__ float red[HC][NW];
  __shared__ float rstd_s[HC];
  const int m = blockIdx.x, t = threadIdx.x;
  uint4 resv[HC];
  float g[HC];
#pragma unroll
  for (int b = 0; b < HC; ++b) resv[b] = *reinterpret_cast<const uint4*>(res + (int64_t)m * KD + b * H + 8 * t);
  load_gates(inj, inj_stride, m, g);
  const uint4 val = ar[(int64_t)m * PACKS + t];
  epilogue<MODE, TREE>(val, resv, g, w, res_out + (int64_t)m * KD, xn == nullptr ? nullptr : xn + (int64_t)m * KD, red,
                       rstd_s);
  if (dbg != nullptr && t < HC) {
    dbg[m * 8 + t] = g[t];
    if constexpr (MODE == 2) dbg[m * 8 + 4 + t] = rstd_s[t];
  }
}

struct State {
  Ptrs ptrs;
  int rank;
  int device;
};
static std::mutex g_mu;
static std::vector<State*> g_states;

}  // namespace tcg

using namespace tcg;

static std::tuple<int64_t, at::Tensor> tcg_alloc() {
  void* p = nullptr;
  TCG_CHECK(cudaMalloc(&p, BUF_BYTES));
  TCG_CHECK(cudaMemset(p, 0, HDR));
  TCG_CHECK(cudaMemset(static_cast<char*>(p) + HDR, SENT_BYTE, BUF_BYTES - HDR));
  TCG_CHECK(cudaDeviceSynchronize());
  cudaIpcMemHandle_t h;
  TCG_CHECK(cudaIpcGetMemHandle(&h, p));
  auto t = at::empty({(int64_t)sizeof(h)}, at::kByte);
  memcpy(t.data_ptr(), &h, sizeof(h));
  return {reinterpret_cast<int64_t>(p), t};
}

static int64_t tcg_open(at::Tensor handle) {
  TORCH_CHECK(handle.numel() == sizeof(cudaIpcMemHandle_t));
  cudaIpcMemHandle_t h;
  memcpy(&h, handle.data_ptr(), sizeof(h));
  void* p = nullptr;
  TCG_CHECK(cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<int64_t>(p);
}

static void tcg_close(int64_t p) { cudaIpcCloseMemHandle(reinterpret_cast<void*>(p)); }
static void tcg_free(int64_t p) { cudaFree(reinterpret_cast<void*>(p)); }

static int64_t tcg_init(std::vector<int64_t> ptrs, int64_t rank) {
  TORCH_CHECK(ptrs.size() == NR, "tc_glue: world size must be 4");
  auto* s = new State();
  for (int i = 0; i < NR; ++i) s->ptrs.p[i] = reinterpret_cast<char*>(ptrs[i]);
  s->rank = (int)rank;
  TCG_CHECK(cudaGetDevice(&s->device));
  std::lock_guard<std::mutex> lk(g_mu);
  g_states.push_back(s);
  return reinterpret_cast<int64_t>(s);
}

static int64_t tcg_max_m() { return MAXM; }
static int64_t tcg_buf_bytes() { return BUF_BYTES; }

static void check_rows(const at::Tensor& t, int64_t M, int64_t cols, const char* name) {
  TORCH_CHECK(t.scalar_type() == at::kHalf, "tc_glue: ", name, " must be fp16");
  TORCH_CHECK(t.is_contiguous() && t.dim() == 2 && t.size(0) == M && t.size(1) == cols, "tc_glue: ", name, " shape");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(t.data_ptr()) & 15) == 0, "tc_glue: ", name, " alignment");
}

static void check_inj(const at::Tensor& inj, int64_t M) {
  TORCH_CHECK(inj.scalar_type() == at::kHalf && inj.dim() == 2 && inj.size(0) == M && inj.size(1) == HC &&
                  inj.stride(1) == 1,
              "tc_glue: inj must be fp16 [M, 4] with unit inner stride");
}

#define TCG_PTR(T, x) reinterpret_cast<T*>((x).data_ptr())

// mode 0: AR only (arout). 1: AR + combine (res_out). 2: AR + combine + norm (res_out, xn). tree 512|1024.
static void tcg_fused(int64_t h, int64_t mode, int64_t tree, at::Tensor in0, std::optional<at::Tensor> in1,
                      std::optional<at::Tensor> res, std::optional<at::Tensor> inj, std::optional<at::Tensor> w,
                      std::optional<at::Tensor> arout, std::optional<at::Tensor> res_out, std::optional<at::Tensor> xn) {
  auto* s = reinterpret_cast<State*>(h);
  const int64_t M = in0.size(0);
  TORCH_CHECK(M >= 1 && M <= MAXM, "tc_glue: M out of range ", M);
  TORCH_CHECK(tree == 512 || tree == 1024, "tc_glue: tree");
  TORCH_CHECK(mode >= 0 && mode <= 2, "tc_glue: mode");
  check_rows(in0, M, H, "in0");
  if (in1.has_value()) check_rows(*in1, M, H, "in1");
  const uint4* i0 = TCG_PTR(const uint4, in0);
  const uint4* i1 = in1.has_value() ? TCG_PTR(const uint4, *in1) : nullptr;
  const __half* rp = nullptr;
  const __half* ip = nullptr;
  const __half* wp = nullptr;
  uint4* ao = nullptr;
  __half* ro = nullptr;
  __half* xo = nullptr;
  int64_t istride = 0;
  if (mode == 0) {
    TORCH_CHECK(arout.has_value(), "tc_glue: mode 0 needs arout");
    check_rows(*arout, M, H, "arout");
    ao = TCG_PTR(uint4, *arout);
  } else {
    TORCH_CHECK(res.has_value() && inj.has_value() && res_out.has_value(), "tc_glue: mode>=1 needs res, inj, res_out");
    check_rows(*res, M, KD, "res");
    check_rows(*res_out, M, KD, "res_out");
    check_inj(*inj, M);
    rp = TCG_PTR(const __half, *res);
    ip = TCG_PTR(const __half, *inj);
    istride = inj->stride(0);
    ro = TCG_PTR(__half, *res_out);
    if (mode == 2) {
      TORCH_CHECK(w.has_value() && xn.has_value(), "tc_glue: mode 2 needs w, xn");
      TORCH_CHECK(w->scalar_type() == at::kHalf && w->is_contiguous() && w->numel() == KD, "tc_glue: w must be fp16 [10240]");
      check_rows(*xn, M, KD, "xn");
      wp = TCG_PTR(const __half, *w);
      xo = TCG_PTR(__half, *xn);
    }
  }
  const at::cuda::CUDAGuard guard(in0.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 g((unsigned)M), b(NT);
#define TCG_LAUNCH(MD, TR) \
  fused_kernel<MD, TR><<<g, b, 0, stream>>>(s->ptrs, s->rank, i0, i1, rp, ip, istride, wp, ao, ro, xo)
  if (mode == 0) TCG_LAUNCH(0, 512);
  else if (mode == 1) TCG_LAUNCH(1, 512);
  else if (tree == 512) TCG_LAUNCH(2, 512);
  else TCG_LAUNCH(2, 1024);
#undef TCG_LAUNCH
  TCG_CHECK(cudaGetLastError());
}

// Local epilogue on an already-reduced ar [M, H]. mode 1|2. dbg: fp32 [M, 8] = (g0..g3, rstd0..rstd3).
static void tcg_local(int64_t mode, int64_t tree, at::Tensor ar, at::Tensor res, at::Tensor inj, at::Tensor w,
                      at::Tensor res_out, std::optional<at::Tensor> xn, std::optional<at::Tensor> dbg) {
  const int64_t M = ar.size(0);
  if (M == 0) return;
  TORCH_CHECK(mode == 1 || mode == 2, "tc_glue: local mode");
  TORCH_CHECK(tree == 512 || tree == 1024, "tc_glue: tree");
  check_rows(ar, M, H, "ar");
  check_rows(res, M, KD, "res");
  check_rows(res_out, M, KD, "res_out");
  check_inj(inj, M);
  TORCH_CHECK(w.scalar_type() == at::kHalf && w.is_contiguous() && w.numel() == KD, "tc_glue: w");
  __half* xo = nullptr;
  if (mode == 2) {
    TORCH_CHECK(xn.has_value(), "tc_glue: mode 2 needs xn");
    check_rows(*xn, M, KD, "xn");
    xo = TCG_PTR(__half, *xn);
  }
  float* dp = dbg.has_value() ? dbg->data_ptr<float>() : nullptr;
  const at::cuda::CUDAGuard guard(ar.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 g((unsigned)M), b(NT);
#define TCG_LLAUNCH(MD, TR)                                                                                          \
  local_kernel<MD, TR><<<g, b, 0, stream>>>(TCG_PTR(const uint4, ar), TCG_PTR(const __half, res),                   \
                                            TCG_PTR(const __half, inj), inj.stride(0), TCG_PTR(const __half, w), \
                                            TCG_PTR(__half, res_out), xo, dp)
  if (mode == 1) TCG_LLAUNCH(1, 512);
  else if (tree == 512) TCG_LLAUNCH(2, 512);
  else TCG_LLAUNCH(2, 1024);
#undef TCG_LLAUNCH
  TCG_CHECK(cudaGetLastError());
}

TORCH_LIBRARY(tc_glue_c, m) {
  m.def("alloc() -> (int, Tensor)");
  m.impl("alloc", &tcg_alloc);
  m.def("open(Tensor handle) -> int");
  m.impl("open", torch::kCPU, &tcg_open);
  m.def("close_handle(int p) -> ()");
  m.impl("close_handle", &tcg_close);
  m.def("free(int p) -> ()");
  m.impl("free", &tcg_free);
  m.def("init(int[] ptrs, int rank) -> int");
  m.impl("init", &tcg_init);
  m.def("max_m() -> int");
  m.impl("max_m", &tcg_max_m);
  m.def("buf_bytes() -> int");
  m.impl("buf_bytes", &tcg_buf_bytes);
  m.def("fused(int h, int mode, int tree, Tensor in0, Tensor? in1, Tensor? res, Tensor? inj, Tensor? w, "
        "Tensor(a!)? arout, Tensor(b!)? res_out, Tensor(c!)? xn) -> ()");
  m.impl("fused", torch::kCUDA, &tcg_fused);
  m.def("local(int mode, int tree, Tensor ar, Tensor res, Tensor inj, Tensor w, Tensor(a!) res_out, Tensor(b!)? xn, "
        "Tensor(c!)? dbg=None) -> ()");
  m.impl("local", torch::kCUDA, &tcg_local);
}
