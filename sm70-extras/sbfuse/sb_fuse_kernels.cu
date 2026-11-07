// SPDX-License-Identifier: Apache-2.0
// Sound-barrier phase 3 (2026-10-14) — SM70 decode fusions for Swift-1.5 Flash-Next (qwen4_exp, TP4).
// Built as a stand-alone torch extension (namespace torch.ops.sb_fuse); it does NOT touch vLLM's _C.
//
// F1  mtp_moe_fp16_out          : copy of 1Cat's exact SM70 MTP-MoE kernels (csrc/sm70_turbomind/ops/
//                                  mtp_moe_fp16_sm70.cu @5d02c1bd1) with the M gate widened from {1,5} to 1..32.
//                                  Same arithmetic per route; drafter-only (target numerics unchanged).
// F2  hc_replicated_fused       : the replicated batched HyperConnection decode route in 2 launches instead of 3.
//                                  hc_down_partials + down_replicated_reduce are merged with a serial split-K
//                                  ("last block reduces") epilogue; hc_up_batch unchanged. The FP32 summation order
//                                  (split 0..19, __fadd_rn) and every FP16 boundary are identical to the original, so
//                                  the result is BITWISE identical to sm70_qwen38_hc_replicated. Unroll factors are
//                                  template knobs (they reorder nothing, they only change load scheduling).
//     hc_replicated_orig        : verbatim copy of the 3-launch original, for bitwise A/B in the unit test.
//
// Kernel bodies below are copied from csrc/sm70_qwen38_hc_batch.cuh and mtp_moe_fp16_sm70.cu (1Cat vLLM
// @5d02c1bd1, Apache-2.0) unless marked "SB:".
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/library.h>
#include <torch/types.h>

#include <cstdint>

namespace sb_fuse {

// ----------------------------------------------------------------------------------------------------------------
// HC helpers (verbatim)
// ----------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void mma(float (&d)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0,%1,%2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ float sigmoid(float x) {
  float e, d;
  asm("mul.f32 %0,%1,0fBFB8AA3B;" : "=f"(e) : "f"(x));
  asm("ex2.approx.f32 %0,%1;" : "=f"(e) : "f"(e));
  asm("add.f32 %0,%1,0f3F800000;" : "=f"(d) : "f"(e));
  return div_full(1.0f, d);
}

// hc_up_batch_body<PairRows=false, Warps=1, Unroll, FuseMix=true, FullOutput=false> (verbatim, PairRows/FuseMix fixed)
template <int Unroll>
__device__ __forceinline__ void hc_up_body(const half* __restrict__ lora, const half* __restrict__ packed,
                                           const half* __restrict__ branches, half* __restrict__ out, int rows,
                                           int hidden, int hidden_offset, int block_x, int block_y) {
  constexpr int Warps = 1;
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int tile = block_x * Warps + warp;
  if (tile >= hidden / 8) return;
  const int group = block_y;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int branch = (lane >> 2) & 3;
  const int col = branch * 8 + r;
  const half* w = packed + static_cast<size_t>(tile) * 320 * 32;
  float accum[8] = {};
#pragma unroll Unroll
  for (int g = 0; g < 20; ++g) {
    const uint4 lo = *reinterpret_cast<const uint4*>(w + (g * 64 + col) * 8);
    const uint4 hi = *reinterpret_cast<const uint4*>(w + (g * 64 + 32 + col) * 8);
    const int row = group * 8 + r;
    uint4 a = make_uint4(0, 0, 0, 0), b = a;
    if (row < rows) {
      const half* x = lora + row * 320 + g * 16;
      a = *reinterpret_cast<const uint4*>(x);
      b = *reinterpret_cast<const uint4*>(x + 8);
    }
    mma(accum, a.x, a.y, lo.x, lo.y);
    mma(accum, a.z, a.w, lo.z, lo.w);
    mma(accum, b.x, b.y, hi.x, hi.y);
    mma(accum, b.z, b.w, hi.z, hi.w);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int row = group * 8 + ((i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1));
    const int h = tile * 8 + ((i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
    const half gate = __float2half_rn(accum[i]);
    const float gate_f = __half2float(gate);
    float v = 0.0f;
    if (row < rows) {
      v = __half2float(branches[row * 10240 + branch * 2560 + hidden_offset + h]);
    }
    const float s = sigmoid(gate_f);
    float mixed = 0.0f;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const int src = (lane & ~12) | (b << 2);
      const float gs = __shfl_sync(0xffffffff, s, src);
      const float xx = __shfl_sync(0xffffffff, v, src);
      mixed = fmaf(gs, xx, mixed);
    }
    if (branch == 0 && row < rows) {
      out[row * hidden + h] = __float2half_rn(div_full(mixed, 4.0f));
    }
  }
}

template <int Unroll>
__global__ __launch_bounds__(32, 4) void hc_up_kernel(const half* lora, const half* packed, const half* branches,
                                                      half* out, int rows, int hidden, int hidden_offset) {
  hc_up_body<Unroll>(lora, packed, branches, out, rows, hidden, hidden_offset, blockIdx.x, blockIdx.y);
}

// hc_down_partials_body<PairRows=false, Warps=1, WarpM16=false, N=352, RoundPartials=false, Unroll> (verbatim)
template <int Unroll>
__device__ __forceinline__ void hc_down_body(const half* __restrict__ x, const half* __restrict__ packed,
                                             float* __restrict__ partials, int rows, int block_x, int block_y,
                                             int block_z) {
  constexpr int N = 352;
  constexpr int TileN = 32;
  constexpr int Warps = 1;
  const int lane = threadIdx.x % 32;
  const int tile = block_x * Warps + threadIdx.x / 32;
  if (tile >= N / TileN) return;
  const int group = block_y;
  const int split = block_z;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int output_quad = quad;
  const int col = output_quad * 8 + r;
  const int row_base = group * 8;
  const half* w = packed + static_cast<size_t>(tile) * 10240 * TileN;
  float accum[8] = {};
#pragma unroll Unroll
  for (int g = 0; g < 32; ++g) {
    const int kg = split * 32 + g;
    const uint4 lo = *reinterpret_cast<const uint4*>(w + (kg * 2 * TileN + col) * 8);
    const uint4 hi = *reinterpret_cast<const uint4*>(w + (kg * 2 * TileN + TileN + col) * 8);
    const int row = row_base + r;
    uint4 a = make_uint4(0, 0, 0, 0), b = a;
    if (row < rows) {
      const half* input = x + row * 10240 + kg * 16;
      a = *reinterpret_cast<const uint4*>(input);
      b = *reinterpret_cast<const uint4*>(input + 8);
    }
    mma(accum, a.x, a.y, lo.x, lo.y);
    mma(accum, a.z, a.w, lo.z, lo.w);
    mma(accum, b.x, b.y, hi.x, hi.y);
    mma(accum, b.z, b.w, hi.z, hi.w);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int row = row_base + ((i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1));
    const int n = tile * TileN + output_quad * 8 + ((i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
    if (row < rows) {
      partials[(split * rows + row) * N + n] = accum[i];
    }
  }
}

template <int Unroll>
__global__ __launch_bounds__(32, 4) void hc_down_kernel(const half* x, const half* packed, float* partials,
                                                        int rows) {
  hc_down_body<Unroll>(x, packed, partials, rows, blockIdx.x, blockIdx.y, blockIdx.z);
}

// verbatim down_replicated_reduce
__global__ void down_replicated_reduce(const float* partials, half* lora, half* injection, int rows) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * 324) return;
  const int row = index / 324, col = index % 324;
  float acc = 0.0f;
#pragma unroll
  for (int split = 0; split < 20; ++split) acc = __fadd_rn(acc, partials[(split * rows + row) * 352 + col]);
  const half value = __float2half_rn(acc);
  if (col < 320) {
    const float xx = div_full(__half2float(value), 4.0f);
    lora[row * 320 + col] = __float2half_rn(__fmul_rn(xx, sigmoid(xx)));
  } else {
    injection[row * 4 + col - 320] = value;
  }
}

// SB: down partials + serial split-K reduction. Every (tile, group) has gridDim.z = 20 split blocks; the block
// that arrives last reduces that tile's 32 output columns for the group's 8 rows, with exactly the arithmetic of
// down_replicated_reduce (sequential __fadd_rn over split 0..19 from 0.0f, FP16 rounding, SiLU on value/4).
// counters: int32[groups * 11] for this call site, zero on entry; the reducing block restores 0 (graph-replay safe).
template <int Unroll>
__global__ __launch_bounds__(32, 4) void hc_down_fused_reduce_kernel(const half* x, const half* packed,
                                                                     float* partials, half* lora, half* injection,
                                                                     int* counters, int rows) {
  hc_down_body<Unroll>(x, packed, partials, rows, blockIdx.x, blockIdx.y, blockIdx.z);
  __threadfence();  // make this block's partials visible device-wide before signalling
  const int lane = threadIdx.x;
  int* counter = counters + blockIdx.y * 11 + blockIdx.x;
  int previous = 0;
  if (lane == 0) previous = atomicAdd(counter, 1);
  previous = __shfl_sync(0xffffffff, previous, 0);
  if (previous != static_cast<int>(gridDim.z) - 1) return;
  __threadfence();  // acquire side: all 20 producers fenced before their atomicAdd
  // Lane owns column tile*32+lane (coalesced), iterates the group's 8 rows. All loads of two rows are issued
  // before their (ordered) additions so the tail costs ~1 L2 round trip per row pair, not per load.
  const int tile = blockIdx.x, group = blockIdx.y;
  const int col = tile * 32 + lane;
  const bool col_ok = col < 324;
#pragma unroll
  for (int j0 = 0; j0 < 8; j0 += 2) {
    float v[2][20];
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      const int row = group * 8 + j0 + jj;
      const bool ok = col_ok && row < rows;
#pragma unroll
      for (int split = 0; split < 20; ++split)
        v[jj][split] = ok ? __ldcg(partials + (split * rows + row) * 352 + col) : 0.0f;
    }
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      const int row = group * 8 + j0 + jj;
      if (!col_ok || row >= rows) continue;
      float acc = 0.0f;
#pragma unroll
      for (int split = 0; split < 20; ++split) acc = __fadd_rn(acc, v[jj][split]);
      const half value = __float2half_rn(acc);
      if (col < 320) {
        const float xx = div_full(__half2float(value), 4.0f);
        lora[row * 320 + col] = __float2half_rn(__fmul_rn(xx, sigmoid(xx)));
      } else {
        injection[row * 4 + col - 320] = value;
      }
    }
  }
  if (lane == 0) atomicExch(counter, 0);
}

// ----------------------------------------------------------------------------------------------------------------
// Host wrappers — HC
// ----------------------------------------------------------------------------------------------------------------
static void check_hc(const torch::Tensor& input, const torch::Tensor& packed_down, const torch::Tensor& packed_up,
                     const torch::Tensor& partials, const torch::Tensor& lora, const torch::Tensor& output,
                     const torch::Tensor& injection) {
  TORCH_CHECK(input.is_cuda() && input.dim() == 2 && input.size(0) >= 2 && input.size(0) <= 20 &&
                  input.size(1) == 10240,
              "sb_fuse HC requires CUDA [2..20, 10240] input");
  const int64_t m = input.size(0);
  for (const auto& t : {input, packed_down, packed_up, lora, output, injection})
    TORCH_CHECK(t.is_cuda() && t.device() == input.device() && t.scalar_type() == at::kHalf && t.is_contiguous() &&
                    reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                "sb_fuse HC requires aligned contiguous FP16 tensors");
  TORCH_CHECK(packed_down.sizes() == at::IntArrayRef({11, 640, 2, 32, 8}) &&
                  packed_up.sizes() == at::IntArrayRef({320, 20, 2, 4, 8, 8}) &&
                  lora.sizes() == at::IntArrayRef({m, 320}) && output.sizes() == at::IntArrayRef({m, 2560}) &&
                  injection.sizes() == at::IntArrayRef({m, 4}) && partials.is_cuda() &&
                  partials.device() == input.device() && partials.scalar_type() == at::kFloat &&
                  partials.is_contiguous() && partials.sizes() == at::IntArrayRef({20, m, 352}),
              "sb_fuse HC: invalid replicated geometry");
}

template <int U>
static void launch_up(cudaStream_t s, const torch::Tensor& lora, const torch::Tensor& packed_up,
                      const torch::Tensor& input, torch::Tensor& output, int m) {
  hc_up_kernel<U><<<dim3(320, (m + 7) / 8), 32, 0, s>>>(
      reinterpret_cast<const half*>(lora.data_ptr()), reinterpret_cast<const half*>(packed_up.data_ptr()),
      reinterpret_cast<const half*>(input.data_ptr()), reinterpret_cast<half*>(output.data_ptr()), m, 2560, 0);
}

static void dispatch_up(int unroll, cudaStream_t s, const torch::Tensor& lora, const torch::Tensor& packed_up,
                        const torch::Tensor& input, torch::Tensor& output, int m) {
  switch (unroll) {
    case 4: launch_up<4>(s, lora, packed_up, input, output, m); break;
    case 5: launch_up<5>(s, lora, packed_up, input, output, m); break;
    case 10: launch_up<10>(s, lora, packed_up, input, output, m); break;
    case 20: launch_up<20>(s, lora, packed_up, input, output, m); break;
    default: TORCH_CHECK(false, "sb_fuse HC: unroll_up must be 4, 5, 10 or 20");
  }
}

void hc_replicated_orig(torch::Tensor input, torch::Tensor packed_down, torch::Tensor packed_up,
                        torch::Tensor partials, torch::Tensor lora, torch::Tensor output, torch::Tensor injection) {
  check_hc(input, packed_down, packed_up, partials, lora, output, injection);
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  const int m = input.size(0);
  const auto s = at::cuda::getCurrentCUDAStream().stream();
  hc_down_kernel<4><<<dim3(11, (m + 7) / 8, 20), 32, 0, s>>>(reinterpret_cast<const half*>(input.data_ptr()),
                                                             reinterpret_cast<const half*>(packed_down.data_ptr()),
                                                             partials.data_ptr<float>(), m);
  down_replicated_reduce<<<(m * 324 + 127) / 128, 128, 0, s>>>(partials.data_ptr<float>(),
                                                                reinterpret_cast<half*>(lora.data_ptr()),
                                                                reinterpret_cast<half*>(injection.data_ptr()), m);
  launch_up<4>(s, lora, packed_up, input, output, m);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int UD>
static void launch_down_fused(cudaStream_t s, const torch::Tensor& input, const torch::Tensor& packed_down,
                              torch::Tensor& partials, torch::Tensor& lora, torch::Tensor& injection,
                              torch::Tensor& counters, int m) {
  hc_down_fused_reduce_kernel<UD><<<dim3(11, (m + 7) / 8, 20), 32, 0, s>>>(
      reinterpret_cast<const half*>(input.data_ptr()), reinterpret_cast<const half*>(packed_down.data_ptr()),
      partials.data_ptr<float>(), reinterpret_cast<half*>(lora.data_ptr()),
      reinterpret_cast<half*>(injection.data_ptr()), counters.data_ptr<int>(), m);
}

void hc_replicated_fused(torch::Tensor input, torch::Tensor packed_down, torch::Tensor packed_up,
                         torch::Tensor partials, torch::Tensor lora, torch::Tensor output, torch::Tensor injection,
                         torch::Tensor counters, int64_t unroll_down, int64_t unroll_up) {
  check_hc(input, packed_down, packed_up, partials, lora, output, injection);
  const int m = input.size(0);
  TORCH_CHECK(counters.is_cuda() && counters.device() == input.device() && counters.scalar_type() == at::kInt &&
                  counters.is_contiguous() && counters.numel() >= ((m + 7) / 8) * 11,
              "sb_fuse HC: counters must be a zeroed int32 CUDA buffer of >= groups*11 entries");
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  const auto s = at::cuda::getCurrentCUDAStream().stream();
  switch (unroll_down) {
    case 4: launch_down_fused<4>(s, input, packed_down, partials, lora, injection, counters, m); break;
    case 8: launch_down_fused<8>(s, input, packed_down, partials, lora, injection, counters, m); break;
    case 16: launch_down_fused<16>(s, input, packed_down, partials, lora, injection, counters, m); break;
    case 32: launch_down_fused<32>(s, input, packed_down, partials, lora, injection, counters, m); break;
    default: TORCH_CHECK(false, "sb_fuse HC: unroll_down must be 4, 8, 16 or 32");
  }
  dispatch_up(static_cast<int>(unroll_up), s, lora, packed_up, input, output, m);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Same as hc_replicated_orig but with tunable unroll and 3 launches (isolates the unroll effect in the bench).
void hc_replicated_unfused(torch::Tensor input, torch::Tensor packed_down, torch::Tensor packed_up,
                           torch::Tensor partials, torch::Tensor lora, torch::Tensor output, torch::Tensor injection,
                           int64_t unroll_down, int64_t unroll_up) {
  check_hc(input, packed_down, packed_up, partials, lora, output, injection);
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  const int m = input.size(0);
  const auto s = at::cuda::getCurrentCUDAStream().stream();
  const dim3 grid(11, (m + 7) / 8, 20);
  const auto* xp = reinterpret_cast<const half*>(input.data_ptr());
  const auto* dp = reinterpret_cast<const half*>(packed_down.data_ptr());
  float* pp = partials.data_ptr<float>();
  switch (unroll_down) {
    case 4: hc_down_kernel<4><<<grid, 32, 0, s>>>(xp, dp, pp, m); break;
    case 8: hc_down_kernel<8><<<grid, 32, 0, s>>>(xp, dp, pp, m); break;
    case 16: hc_down_kernel<16><<<grid, 32, 0, s>>>(xp, dp, pp, m); break;
    case 32: hc_down_kernel<32><<<grid, 32, 0, s>>>(xp, dp, pp, m); break;
    default: TORCH_CHECK(false, "sb_fuse HC: unroll_down must be 4, 8, 16 or 32");
  }
  down_replicated_reduce<<<(m * 324 + 127) / 128, 128, 0, s>>>(pp, reinterpret_cast<half*>(lora.data_ptr()),
                                                                reinterpret_cast<half*>(injection.data_ptr()), m);
  dispatch_up(static_cast<int>(unroll_up), s, lora, packed_up, input, output, m);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ----------------------------------------------------------------------------------------------------------------
// F1: MTP MoE (verbatim kernels; host gate widened)
// ----------------------------------------------------------------------------------------------------------------
template <int N, int K, int BN, int BK, bool Weighted>
__global__ void mtp_moe_fp16_tile_kernel(const half* x, const half* w, const int32_t* ids, const float* weights,
                                         const int32_t* padded, half* y) {
  __shared__ half tile[BN][BK + 2];
  __shared__ float input[BK];
  const int t = threadIdx.x, route = blockIdx.y;
  if (route * 2 >= *padded) return;
  const int row = Weighted ? route : route / 10;
  const int expert = ids[route], col = blockIdx.x * BN + t;
  if (expert < 0 || expert >= 512) {
    if (t < BN && col < N) y[route * N + col] = __float2half(0.0f);
    return;
  }
  float acc = 0.0f;
  for (int base = 0; base < K; base += BK) {
    for (int i = t; i < BK; i += 128) input[i] = base + i < K ? __half2float(x[row * K + base + i]) : 0.0f;
    for (int i = t; i < BN * BK / 8; i += 128) {
      const int n = i / (BK / 8), k = i % (BK / 8) * 8;
      union {
        uint4 v;
        half h[8];
      } data;
      data.v = make_uint4(0, 0, 0, 0);
      if (blockIdx.x * BN + n < N && base + k < K)
        data.v = *reinterpret_cast<const uint4*>(w + (int64_t(expert) * N + blockIdx.x * BN + n) * K + base + k);
#pragma unroll
      for (int j = 0; j < 8; ++j) tile[n][k + j] = data.h[j];
    }
    __syncthreads();
    if (t < BN) {
#pragma unroll
      for (int k = 0; k < BK; ++k) acc = __fmaf_rn(input[k], __half2float(tile[t][k]), acc);
    }
    __syncthreads();
  }
  if (t < BN && col < N) {
    if (Weighted) acc = __fmul_rn(acc, weights[route]);
    y[route * N + col] = __float2half_rn(acc);
  }
}

__global__ void mtp_moe_fp16_m1_w13_kernel(const half* x, const half* w, const int32_t* ids, const int32_t* padded,
                                           half* y) {
  constexpr int kN = 320, kK = 2560, kThreads = 64;
  __shared__ float input[kK];
  const int t = threadIdx.x, route = blockIdx.y;
  if (route * 2 >= *padded) return;
  const int expert = ids[route], col = blockIdx.x * kThreads + t;
  for (int k = t; k < kK; k += kThreads) input[k] = __half2float(x[k]);
  __syncthreads();
  float acc = 0.0f;
  if (expert >= 0 && expert < 512) {
    const half* p = w + (int64_t(expert) * kN + col) * kK;
    for (int k = 0; k < kK; k += 8) {
      union {
        uint4 v;
        half h[8];
      } data;
      data.v = *reinterpret_cast<const uint4*>(p + k);
#pragma unroll
      for (int j = 0; j < 8; ++j) acc = __fmaf_rn(input[k + j], __half2float(data.h[j]), acc);
    }
  }
  y[route * kN + col] = __float2half_rn(acc);
}

void mtp_moe_fp16_out(torch::Tensor out, torch::Tensor x, torch::Tensor w, torch::Tensor ids, torch::Tensor weights,
                      torch::Tensor padded, bool down) {
  TORCH_CHECK(x.is_cuda(), "sb_fuse MTP MoE requires CUDA tensors");
  const c10::cuda::CUDAGuard guard(x.device());
  for (const auto& tensor : {out, x, w, ids, weights, padded})
    TORCH_CHECK(tensor.device() == x.device() && tensor.is_contiguous(),
                "sb_fuse MTP MoE requires same-device contiguous tensors");
  TORCH_CHECK(out.scalar_type() == at::kHalf && x.scalar_type() == at::kHalf && w.scalar_type() == at::kHalf &&
                  ids.scalar_type() == at::kInt && weights.scalar_type() == at::kFloat &&
                  padded.scalar_type() == at::kInt,
              "sb_fuse MTP MoE: invalid dtypes");
  // SB: M gate widened from {1,5} to 1..32 (route-parallel kernels are M-generic).
  TORCH_CHECK(weights.dim() == 2 && weights.size(1) == 10 && weights.size(0) >= 1 && weights.size(0) <= 32,
              "sb_fuse MTP MoE requires M1..32, topk=10");
  const int m = weights.size(0), n = down ? 2560 : 320, k = down ? 160 : 2560;
  TORCH_CHECK(w.sizes() == at::IntArrayRef({512, n, k}) && x.sizes() == at::IntArrayRef({down ? m * 10 : m, k}) &&
                  out.sizes() == at::IntArrayRef({m, 10, n}) && ids.sizes() == at::IntArrayRef({m * 10}) &&
                  padded.numel() == 1,
              "sb_fuse MTP MoE: invalid geometry");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0, "sb_fuse MTP MoE requires 16-byte aligned weights");
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto* xp = reinterpret_cast<const half*>(x.data_ptr());
  const auto* wp = reinterpret_cast<const half*>(w.data_ptr());
  const auto* ip = ids.data_ptr<int32_t>();
  const auto* tp = weights.data_ptr<float>();
  const auto* pp = padded.data_ptr<int32_t>();
  auto* op = reinterpret_cast<half*>(out.data_ptr());
  if (down) {
    mtp_moe_fp16_tile_kernel<2560, 160, 64, 64, true><<<dim3(40, m * 10), 128, 0, stream>>>(xp, wp, ip, tp, pp, op);
  } else if (m == 1) {
    mtp_moe_fp16_m1_w13_kernel<<<dim3(5, 10), 64, 0, stream>>>(xp, wp, ip, pp, op);
  } else {
    mtp_moe_fp16_tile_kernel<320, 2560, 32, 128, false><<<dim3(10, m * 10), 128, 0, stream>>>(xp, wp, ip, tp, pp,
                                                                                               op);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace sb_fuse

TORCH_LIBRARY(sb_fuse, m) {
  m.def(
      "mtp_moe_fp16_out(Tensor(a!) out, Tensor x, Tensor w, Tensor ids, Tensor weights, Tensor padded, bool down) "
      "-> ()");
  m.def(
      "hc_replicated_orig(Tensor input, Tensor packed_down, Tensor packed_up, Tensor(a!) partials, "
      "Tensor(b!) lora, Tensor(c!) output, Tensor(d!) injection) -> ()");
  m.def(
      "hc_replicated_unfused(Tensor input, Tensor packed_down, Tensor packed_up, Tensor(a!) partials, "
      "Tensor(b!) lora, Tensor(c!) output, Tensor(d!) injection, int unroll_down, int unroll_up) -> ()");
  m.def(
      "hc_replicated_fused(Tensor input, Tensor packed_down, Tensor packed_up, Tensor(a!) partials, "
      "Tensor(b!) lora, Tensor(c!) output, Tensor(d!) injection, Tensor(e!) counters, int unroll_down, "
      "int unroll_up) -> ()");
}

TORCH_LIBRARY_IMPL(sb_fuse, CUDA, m) {
  m.impl("mtp_moe_fp16_out", &sb_fuse::mtp_moe_fp16_out);
  m.impl("hc_replicated_orig", &sb_fuse::hc_replicated_orig);
  m.impl("hc_replicated_unfused", &sb_fuse::hc_replicated_unfused);
  m.impl("hc_replicated_fused", &sb_fuse::hc_replicated_fused);
}
