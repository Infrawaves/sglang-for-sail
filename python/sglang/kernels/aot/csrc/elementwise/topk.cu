/**
 * @NOTE: This file is adapted from
 * https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_v32/topk_selector.py
 * We:
 * 1. adapt from tilelang to pure cuda
 * 2. optimize the performance a little
 * 3. fix the potential illegal memory access
 */
#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <optional>

namespace {

constexpr int TopK = 2048;
constexpr int kThreadsPerBlock = 1024;

#ifdef USE_ROCM
// On ROCm, the per-workgroup LDS budget depends on the target arch, so we inject a
// per-arch value from `setup_rocm.py` via `-DSGL_TOPK_DYNAMIC_SMEM_BYTES=...`.
#ifdef SGL_TOPK_DYNAMIC_SMEM_BYTES
constexpr size_t kSmem = static_cast<size_t>(SGL_TOPK_DYNAMIC_SMEM_BYTES);
#else
constexpr size_t kSmem = 48 * 1024;  // bytes
#endif
#else
// Reduced from 128KB to 32KB to improve occupancy.
// Each radix pass needs at most ~TopK candidates in the threshold bin,
// so 4K entries per round (2 rounds = 8K entries = 32KB) is sufficient.
constexpr size_t kSmem = 8 * 1024 * sizeof(uint32_t);  // 32KB (bytes)
#endif

struct FastTopKParams {
  const float* __restrict__ input;         // [B, input_stride]
  const int32_t* __restrict__ row_starts;  // [B]
  int32_t* __restrict__ indices;           // [B, TopK]
  int32_t* __restrict__ lengths;           // [B]
  int64_t input_stride;
};

// when length <= TopK, we can directly write the indices
__device__ void naive_topk_cuda(const float* __restrict__ score, int32_t* __restrict__ indice, int32_t length) {
  const auto tid = threadIdx.x;
  for (int i = tid; i < TopK; i += kThreadsPerBlock) {
    indice[i] = (i < length) ? i : -1;
  }
}

// keep the first `length` entries, set others to -1
__device__ void naive_topk_transform(
    const float* __restrict__ score,
    int32_t length,
    int32_t* __restrict__ dst_page_table,
    const int32_t* __restrict__ src_page_table) {
  const auto tid = threadIdx.x;
  for (auto i = tid; i < TopK; i += kThreadsPerBlock) {
    dst_page_table[i] = (i < length) ? src_page_table[i] : -1;
  }
}

// keep the first `length` entries, set others to -1
__device__ void naive_topk_transform_ragged(
    const float* __restrict__ score, int32_t length, int32_t* __restrict__ topk_indices_ragged, int32_t offset) {
  const auto tid = threadIdx.x;
  for (auto i = tid; i < TopK; i += kThreadsPerBlock) {
    topk_indices_ragged[i] = (i < length) ? static_cast<int32_t>(i) + offset : -1;
  }
}

__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ void fast_topk_cuda_tl(const float* __restrict__ input, int* __restrict__ index, int row_start, int length) {
  // An optimized topk kernel copied from tilelang kernel
  // We assume length > TopK here, or it will crash
  int topk = TopK;
  constexpr auto BLOCK_SIZE = 1024;
  constexpr auto RADIX = 256;
  constexpr auto SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));

  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];

  auto& s_histogram = s_histogram_buf[0];
  // allocate for two rounds
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  const int tx = threadIdx.x;

  // stage 1: 8bit coarse histogram
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();

  for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
    const auto bin = convert_to_uint8(input[idx + row_start]);
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      static_assert(1 << 8 == RADIX);
      if (C10_LIKELY(tx < RADIX)) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += s_histogram_buf[k][tx + j];
        }
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  topk -= s_histogram[threshold_bin + 1];

  if (topk == 0) {
    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto bin = static_cast<int>(convert_to_uint8(input[idx + row_start]));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      }
    }
    __syncthreads();
    return;
  } else {
    __syncthreads();
    if (tx < RADIX + 1) {
      s_histogram[tx] = 0;
    }
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto raw_input = input[idx + row_start];
      const auto bin = static_cast<int>(convert_to_uint8(raw_input));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      } else if (bin == threshold_bin) {
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        /// NOTE: (dark) fuse the histogram computation here
        if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
          s_input_idx[0][pos] = idx;
          const auto bin = convert_to_uint32(raw_input);
          const auto sub_bin = (bin >> 24) & 0xFF;
          ::atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    }
    __syncthreads();
  }

  // stage 2: refine with 8bit radix passes
#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    __shared__ int s_last_remain;
    const auto r_idx = round % 2;

    // clip here to prevent overflow
    const auto _raw_num_input = s_num_input[r_idx];
    const auto num_input = (_raw_num_input < int(SMEM_INPUT_SIZE)) ? _raw_num_input : int(SMEM_INPUT_SIZE);

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
      s_threshold_bin_id = tx;
      s_num_input[r_idx ^ 1] = 0;
      s_last_remain = topk - s_histogram[tx + 1];
    }
    __syncthreads();

    const auto threshold_bin = s_threshold_bin_id;
    topk -= s_histogram[threshold_bin + 1];

    if (topk == 0) {
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(input[idx + row_start]) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        }
      }
      __syncthreads();
      break;
    } else {
      __syncthreads();
      if (tx < RADIX + 1) {
        s_histogram[tx] = 0;
      }
      __syncthreads();
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto raw_input = input[idx + row_start];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        } else if (bin == threshold_bin) {
          if (round == 3) {
            const auto pos = ::atomicAdd(&s_last_remain, -1);
            if (pos > 0) {
              index[TopK - pos] = idx;
            }
          } else {
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
              /// NOTE: (dark) fuse the histogram computation here
              s_input_idx[r_idx ^ 1][pos] = idx;
              const auto bin = convert_to_uint32(raw_input);
              const auto sub_bin = (bin >> (offset - 8)) & 0xFF;
              ::atomicAdd(&s_histogram[sub_bin], 1);
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // topk
    void topk_kernel(const FastTopKParams params) {
  const auto& [input, row_starts, indices, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto length = lengths[bid];
  const auto indice = indices + bid * TopK;
  const auto score = input + bid * input_stride;
  if (length <= TopK) {
    return naive_topk_cuda(score, indice, length);
  } else {
    return fast_topk_cuda_tl(score, indice, row_start, length);
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // decode
    void topk_transform_decode_kernel(
        const FastTopKParams params,
        int32_t* __restrict__ dst_page_table,
        const int32_t* __restrict__ src_page_table,
        const int64_t src_stride) {
  const auto& [input, _1, _2, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto row_start = 0;
  const auto length = lengths[bid];
  const auto src_page_entry = src_page_table + bid * src_stride;
  const auto dst_page_entry = dst_page_table + bid * TopK;
  const auto score = input + bid * input_stride;
  if (length <= TopK) {
    return naive_topk_transform(score, length, dst_page_entry, src_page_entry);
  } else {
    __shared__ int s_indices[TopK];
    fast_topk_cuda_tl(score, s_indices, row_start, length);
    // copy src[s_indices] to dst, we manually unroll here
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_page_entry[idx_0] = src_page_entry[pos_0];
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_page_entry[idx_1] = src_page_entry[pos_1];
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // prefill
    void topk_transform_prefill_kernel(
        const FastTopKParams params,
        int32_t* __restrict__ dst_page_table,
        const int32_t* __restrict__ src_page_table,
        const int64_t src_stride,
        const int32_t* __restrict__ cu_seqlens_q,
        const int64_t prefill_bs) {
  const auto& [input, row_starts, _, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto length = lengths[bid];
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto dst_page_entry = dst_page_table + bid * TopK;
  const auto score = input + bid * input_stride;

  /// NOTE: prefill bs is usually small, we can just use a simple loop here
  /// We ensure that last cu_seqlens is equal to number of blocks launched
  __shared__ const int32_t* s_src_page_entry;
  if (C10_LIKELY(prefill_bs <= kThreadsPerBlock)) {
    if (tid < prefill_bs) {
      if (bid >= cu_seqlens_q[tid] && bid < cu_seqlens_q[tid + 1]) {
        s_src_page_entry = src_page_table + tid * src_stride;
      }
    }
  } else {
    for (int64_t i = tid; i < prefill_bs; i += kThreadsPerBlock) {
      if (bid >= cu_seqlens_q[i] && bid < cu_seqlens_q[i + 1]) {
        s_src_page_entry = src_page_table + i * src_stride;
      }
    }
  }
  __syncthreads();
  const auto src_page_entry = s_src_page_entry;

  if (length <= TopK) {
    return naive_topk_transform(score, length, dst_page_entry, src_page_entry);
  } else {
    __shared__ int s_indices[TopK];
    fast_topk_cuda_tl(score, s_indices, row_start, length);
    // copy src[s_indices] to dst, we manually unroll here
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_page_entry[idx_0] = src_page_entry[pos_0];
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_page_entry[idx_1] = src_page_entry[pos_1];
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // prefill, ragged kv
    void topk_transform_prefill_ragged_kernel(
        const FastTopKParams params,
        int32_t* __restrict__ topk_indices_ragged,
        const int32_t* __restrict__ topk_indices_offset) {
  const auto& [input, row_starts, _, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto length = lengths[bid];
  const auto dst_indices_entry = topk_indices_ragged + bid * TopK;
  const auto score = input + bid * input_stride;
  const auto offset = topk_indices_offset[bid];

  if (length <= TopK) {
    return naive_topk_transform_ragged(score, length, dst_indices_entry, offset);
  } else {
    __shared__ int s_indices[TopK];
    fast_topk_cuda_tl(score, s_indices, row_start, length);
    // copy src[s_indices] to dst, we manually unroll here
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_indices_entry[idx_0] = pos_0 + offset;
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_indices_entry[idx_1] = pos_1 + offset;
  }
}

#if defined(__HGGCCC__)
// Exact ragged top-2048 for PPU prefill. A sampled threshold collects ~kTarget candidates in
// one read of the row; if it misses, the key window is narrowed over the row until it fits.
// Measured on PPU 890, K3.1 CP=8 rank shape: 655 -> 351 us at 16K, 2990 -> 727 us at 48K.
namespace ppu_topk {

constexpr int kThreads = 512;
constexpr int kWarps = kThreads / 32;
constexpr int kBins = 2048;
// Tuned on PPU 890; kCap and kTarget bound how often the sampled threshold falls back to
// narrowing the row, kUnroll is the float4 loads in flight per thread.
constexpr int kCap = 6144;
constexpr int kTarget = 3072;
constexpr int kSegs = 32;
constexpr int kNs = kSegs * 32;
constexpr int kUnroll = 2;
constexpr uint32_t kMaxKey = 0xFFFFFFFFu;

struct Smem {
  uint32_t ckey[kCap];
  int cidx[kCap];
  int hist[kBins];
  uint32_t samp[kNs];
  int warp_sum[32];
  uint32_t red_min[32], red_max[32];
  int n, n_out, thr, above, tie_left;
};

// Keys between lo and hi (inclusive) are still undecided; `above` keys are > hi.
struct Window {
  uint32_t lo, hi;
  int above, win;
};

__device__ __forceinline__ int shift_for(uint32_t range) {
  return range < kBins ? 0 : 32 - __clz(range) - 11;
}

// Upper edge of the bin starting at lo, clamped to hi without 32-bit overflow.
__device__ __forceinline__ uint32_t bin_hi(uint32_t lo, int shift, uint32_t hi) {
  const uint64_t end = static_cast<uint64_t>(lo) + ((1ull << shift) - 1);
  return end < hi ? static_cast<uint32_t>(end) : hi;
}

// Sets s.thr to the bin holding the k-th largest count of s.hist and s.above to the count
// in higher bins.
__device__ void find_thr(Smem& s, int k) {
  constexpr int per = kBins / kThreads;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  int local[per];
  int sum = 0;
#pragma unroll
  for (int i = 0; i < per; ++i) {
    local[i] = s.hist[kBins - 1 - (tid * per + i)];
    sum += local[i];
  }
  int incl = sum;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += v;
  }
  if (lane == 31) s.warp_sum[warp] = incl;
  __syncthreads();
  if (warp == 0) {
    const int w = lane < kWarps ? s.warp_sum[lane] : 0;
    int wi = w;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, wi, o);
      if (lane >= o) wi += v;
    }
    if (lane < kWarps) s.warp_sum[lane] = wi - w;
  }
  __syncthreads();
  int run = s.warp_sum[warp] + incl - sum;
  if (run < k && run + sum >= k) {
#pragma unroll
    for (int i = 0; i < per; ++i) {
      if (run < k && run + local[i] >= k) {
        s.thr = kBins - 1 - (tid * per + i);
        s.above = run;
      }
      run += local[i];
    }
  }
  __syncthreads();
}

// Finds the k-th largest of keys[0, n), 1 <= k <= n. Exact mode (coarse_max = 0) returns that
// key and how many copies of it to take. Coarse mode stops once the threshold bin holds
// <= coarse_max keys and returns the bin's lower edge, so at least k keys are >= thr.
__device__ void select_kth(Smem& s, const uint32_t* keys, int n, int k, int coarse_max, uint32_t& thr, int& take_eq) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  uint32_t mn = kMaxKey, mx = 0;
  for (int i = tid; i < n; i += kThreads) {
    const uint32_t v = keys[i];
    mn = min(mn, v);
    mx = max(mx, v);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    mn = min(mn, __shfl_xor_sync(0xffffffffu, mn, o));
    mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  }
  if (lane == 0) {
    s.red_min[warp] = mn;
    s.red_max[warp] = mx;
  }
  __syncthreads();
  uint32_t lo = s.red_min[0], hi = s.red_max[0];
#pragma unroll
  for (int w = 1; w < kWarps; ++w) {
    lo = min(lo, s.red_min[w]);
    hi = max(hi, s.red_max[w]);
  }
  while (true) {
    const int shift = shift_for(hi - lo);
    for (int i = tid; i < kBins; i += kThreads)
      s.hist[i] = 0;
    __syncthreads();
    for (int i = tid; i < n; i += kThreads) {
      const uint32_t v = keys[i];
      if (v >= lo && v <= hi) atomicAdd(&s.hist[(v - lo) >> shift], 1);
    }
    __syncthreads();
    find_thr(s, k);
    const int b = s.thr;
    k -= s.above;
    const int bin_cnt = s.hist[b];
    __syncthreads();  // thr, above and hist are rewritten by the next round
    const uint32_t bin_lo = lo + (static_cast<uint32_t>(b) << shift);
    if (shift == 0) {
      thr = bin_lo;
      take_eq = k;
      return;
    }
    if (bin_cnt <= coarse_max) {
      thr = bin_lo;
      take_eq = -1;
      return;
    }
    hi = bin_hi(bin_lo, shift, hi);
    lo = bin_lo;
  }
}

// Calls f(v, idx, ok) for every element of row[0, len). Every lane of a warp makes the same
// sequence of calls, so f may use full-mask warp primitives.
template <typename F>
__device__ __forceinline__ void for_row(const float* __restrict__ row, int len, F&& f) {
  const int tid = threadIdx.x;
  const int mis = static_cast<int>((reinterpret_cast<uintptr_t>(row) >> 2) & 3);
  const int head = min(len, (4 - mis) & 3);
  const int n4 = (len - head) >> 2;
  const float4* body = reinterpret_cast<const float4*>(row + head);
  const int iters = (n4 + kUnroll * kThreads - 1) / (kUnroll * kThreads);
  for (int it = 0; it < iters; ++it) {
    float4 q[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int i = (it * kUnroll + u) * kThreads + tid;
      q[u] = i < n4 ? __ldg(body + i) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int i = (it * kUnroll + u) * kThreads + tid;
      const bool ok = i < n4;
      const int idx = head + 4 * i;
      f(q[u].x, idx + 0, ok);
      f(q[u].y, idx + 1, ok);
      f(q[u].z, idx + 2, ok);
      f(q[u].w, idx + 3, ok);
    }
  }
  if (tid < 32) {  // < 4 head and < 4 tail elements, handled by warp 0
    const int i = tid < 4 ? tid : head + 4 * n4 + tid - 4;
    const bool ok = tid < 4 ? tid < head : i < len;
    f(ok ? row[i] : 0.f, i, ok);
  }
}

// Warp-aggregated append through a shared counter; returns the slot, or -1 when !take.
// Every lane of the warp must call it.
__device__ __forceinline__ int warp_append(int* counter, bool take) {
  const unsigned m = __ballot_sync(0xffffffffu, take);
  if (m == 0) return -1;
  const int lane = threadIdx.x & 31;
  const int leader = __ffs(m) - 1;
  int base = 0;
  if (lane == leader) base = atomicAdd(counter, __popc(m));
  base = __shfl_sync(0xffffffffu, base, leader);
  return take ? base + __popc(m & ((1u << lane) - 1)) : -1;
}

__device__ __forceinline__ void push_cand(Smem& s, int pos, uint32_t key, int idx) {
  if (pos >= 0 && pos < kCap) {
    s.ckey[pos] = key;
    s.cidx[pos] = idx;
  }
}

// Appends the n valid keys of v that are >= lo to the candidates with one atomic per warp.
// Every lane of the warp must call it.
template <int N>
__device__ __forceinline__ void filter_group(Smem& s, const float (&v)[N], int idx0, int valid, uint32_t lo) {
  uint32_t key[N];
  int c = 0;
#pragma unroll
  for (int j = 0; j < N; ++j) {
    key[j] = convert_to_uint32(v[j]);
    c += (j < valid) & (key[j] >= lo);
  }
  if (!__any_sync(0xffffffffu, c)) return;
  const int lane = threadIdx.x & 31;
  int incl = c;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int t = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += t;
  }
  int base = 0;
  if (lane == 31) base = atomicAdd(&s.n, incl);
  int pos = __shfl_sync(0xffffffffu, base, 31) + incl - c;
#pragma unroll
  for (int j = 0; j < N; ++j) {
    if ((j < valid) & (key[j] >= lo)) {
      if (pos < kCap) {
        s.ckey[pos] = key[j];
        s.cidx[pos] = idx0 + j;
      }
      ++pos;
    }
  }
}

// One read of the row collecting every key >= lo into the candidates (s.n also counts past kCap).
__device__ void filter_row(Smem& s, const float* __restrict__ row, int len, uint32_t lo) {
  const int tid = threadIdx.x;
  const int mis = static_cast<int>((reinterpret_cast<uintptr_t>(row) >> 2) & 3);
  const int head = min(len, (4 - mis) & 3);
  const int n4 = (len - head) >> 2;
  const float4* body = reinterpret_cast<const float4*>(row + head);
  const int iters = (n4 + kUnroll * kThreads - 1) / (kUnroll * kThreads);
  for (int it = 0; it < iters; ++it) {
    float4 q[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int i = (it * kUnroll + u) * kThreads + tid;
      q[u] = i < n4 ? __ldg(body + i) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int i = (it * kUnroll + u) * kThreads + tid;
      const float v[4] = {q[u].x, q[u].y, q[u].z, q[u].w};
      filter_group<4>(s, v, head + 4 * i, i < n4 ? 4 : 0, lo);
    }
  }
  if (tid < 32) {  // < 4 head and < 4 tail elements, handled by warp 0
    const int i = tid < 4 ? tid : head + 4 * n4 + tid - 4;
    const bool ok = tid < 4 ? tid < head : i < len;
    const float v[1] = {ok ? row[i] : 0.f};
    filter_group<1>(s, v, i, ok ? 1 : 0, lo);
  }
}

// One read of the row: keys in [w.lo, w.hi] go to the candidates and keys > w.hi go to out.
__device__ void collect_row(Smem& s, const float* __restrict__ row, int len, const Window& w, int* out, int offset) {
  const uint32_t lo = w.lo, hi = w.hi;
  for_row(row, len, [&](float v, int idx, bool ok) {
    const uint32_t key = convert_to_uint32(v);
    const bool cand = ok && key >= lo && key <= hi;
    if (__any_sync(0xffffffffu, cand)) push_cand(s, warp_append(&s.n, cand), key, idx);
    const int pos = warp_append(&s.n_out, ok && key > hi);
    if (pos >= 0) out[pos] = idx + offset;
  });
}

// One histogram round over the row restricted to the window; keeps the bin holding the
// TopK-th largest key.
__device__ void narrow_row(Smem& s, const float* __restrict__ row, int len, Window& w) {
  const uint32_t lo = w.lo, hi = w.hi;
  const int shift = shift_for(hi - lo);
  for (int i = threadIdx.x; i < kBins; i += kThreads)
    s.hist[i] = 0;
  __syncthreads();
  for_row(row, len, [&](float v, int, bool ok) {
    const uint32_t key = convert_to_uint32(v);
    if (ok && key >= lo && key <= hi) atomicAdd(&s.hist[(key - lo) >> shift], 1);
  });
  __syncthreads();
  find_thr(s, TopK - w.above);
  const int b = s.thr;
  w.above += s.above;
  w.win = s.hist[b];
  __syncthreads();
  w.lo = lo + (static_cast<uint32_t>(b) << shift);
  w.hi = bin_hi(w.lo, shift, hi);
}

// The window is a single key with more than kCap copies: emit keys above it and the first
// TopK - above copies that arrive.
__device__ void take_ties(Smem& s, const float* __restrict__ row, int len, const Window& w, int* out, int offset) {
  if (threadIdx.x == 0) {
    s.n_out = 0;
    s.tie_left = TopK - w.above;
  }
  __syncthreads();
  const uint32_t hi = w.hi;
  for_row(row, len, [&](float v, int idx, bool ok) {
    const uint32_t key = convert_to_uint32(v);
    bool take = ok && key > hi;
    if (ok && key == hi) take = atomicSub(&s.tie_left, 1) > 0;
    const int pos = warp_append(&s.n_out, take);
    if (pos >= 0) out[pos] = idx + offset;
  });
}

// Appends the k largest of the n candidates to out, after the s.n_out entries already there.
__device__ void finish(Smem& s, int n, int k, int* out, int offset) {
  uint32_t thr;
  int take_eq;
  select_kth(s, s.ckey, n, k, 0, thr, take_eq);
  if (threadIdx.x == 0) s.tie_left = take_eq;
  __syncthreads();
  for (int base = 0; base < n; base += kThreads) {
    const int i = base + threadIdx.x;
    const bool ok = i < n;
    const uint32_t key = ok ? s.ckey[i] : 0u;
    bool take = ok && key > thr;
    if (ok && key == thr) take = atomicSub(&s.tie_left, 1) > 0;
    const int pos = warp_append(&s.n_out, take);
    if (pos >= 0) out[pos] = s.cidx[i] + offset;
  }
}

// Lower edge of a key window that should hold about kTarget of the row's keys, estimated
// from kSegs evenly spaced 128-byte segments.
__device__ uint32_t sample_threshold(Smem& s, const float* __restrict__ row, int len) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int seg_stride = (len >> 5) / kSegs;  // len > kCap keeps this >= 1
  for (int j = warp; j < kSegs; j += kWarps) {
    s.samp[j * 32 + lane] = convert_to_uint32(__ldg(row + j * seg_stride * 32 + lane));
  }
  __syncthreads();
  const int m = min(kNs, (kNs * kTarget + len - 1) / len);
  uint32_t thr;
  int take_eq;
  select_kth(s, s.samp, kNs, m, max(4, m / 8), thr, take_eq);
  return thr;
}

}  // namespace ppu_topk

__global__ __launch_bounds__(ppu_topk::kThreads, 2048 / ppu_topk::kThreads)  // prefill, ragged kv, PPU
    void topk_transform_prefill_ragged_kernel_ppu(
        const FastTopKParams params,
        int32_t* __restrict__ topk_indices_ragged,
        const int32_t* __restrict__ topk_indices_offset) {
  using namespace ppu_topk;
  extern __shared__ __align__(16) unsigned char ppu_smem_raw[];
  auto& s = *reinterpret_cast<Smem*>(ppu_smem_raw);
  const auto& [input, row_starts, _, lengths, input_stride] = params;
  const int row_id = gridDim.x - 1 - blockIdx.x;  // rows grow with the index; run long ones first
  const int tid = threadIdx.x;
  const int len = lengths[row_id];
  const int offset = topk_indices_offset[row_id];
  int* out = topk_indices_ragged + static_cast<int64_t>(row_id) * TopK;
  if (len <= TopK) {
    for (int i = tid; i < TopK; i += kThreads)
      out[i] = i < len ? i + offset : -1;
    return;
  }
  const float* row = input + static_cast<int64_t>(row_id) * input_stride + (row_starts ? row_starts[row_id] : 0);
  if (len <= kCap) {
    for (int i = tid; i < len; i += kThreads) {
      s.ckey[i] = convert_to_uint32(__ldg(row + i));
      s.cidx[i] = i;
    }
    if (tid == 0) s.n_out = 0;
    __syncthreads();
    finish(s, len, TopK, out, offset);
    return;
  }

  const uint32_t t_lo = sample_threshold(s, row, len);
  if (tid == 0) s.n = s.n_out = 0;
  __syncthreads();
  filter_row(s, row, len, t_lo);
  __syncthreads();
  const int n = s.n;
  if (n >= TopK && n <= kCap) {
    finish(s, n, TopK, out, offset);
    return;
  }

  // The sample missed: n < TopK means every key >= t_lo is selected and t_lo > 0.
  Window w = n < TopK ? Window{0u, t_lo - 1, n, len - n} : Window{t_lo, kMaxKey, 0, n};
  while (w.win > kCap && w.lo < w.hi)
    narrow_row(s, row, len, w);
  if (w.win > kCap) {
    take_ties(s, row, len, w, out, offset);
    return;
  }
  __syncthreads();
  if (tid == 0) s.n = s.n_out = 0;
  __syncthreads();
  collect_row(s, row, len, w, out, offset);
  __syncthreads();
  finish(s, s.n, TopK - w.above, out, offset);
}
#endif  // __HGGCCC__

auto get_params(
    const at::Tensor& score,
    const at::Tensor& lengths,
    std::optional<at::Tensor> row_starts_opt = std::nullopt,
    std::optional<at::Tensor> indices_opt = std::nullopt) -> FastTopKParams {
  const auto B = score.size(0);
  TORCH_CHECK(score.dim() == 2 && score.stride(1) == 1);
  if (row_starts_opt.has_value()) {
    const auto& row_starts = row_starts_opt.value();
    TORCH_CHECK(row_starts.dim() == 1);
    TORCH_CHECK(row_starts.size(0) == B);
  }
  TORCH_CHECK(lengths.dim() == 1 && lengths.is_contiguous());
  TORCH_CHECK(lengths.size(0) == B);
  int32_t* indices_data_ptr = nullptr;
  if (indices_opt.has_value()) {
    const auto& indices = indices_opt.value();
    TORCH_CHECK(indices.dim() == 2 && indices.is_contiguous());
    TORCH_CHECK(indices.size(0) == B);
    TORCH_CHECK(indices.size(1) == TopK);
    indices_data_ptr = indices.data_ptr<int32_t>();
  }

  return FastTopKParams{
      .input = score.data_ptr<float>(),
      .row_starts = row_starts_opt.has_value() ? row_starts_opt->data_ptr<int32_t>() : nullptr,
      .indices = indices_data_ptr,
      .lengths = lengths.data_ptr<int32_t>(),
      .input_stride = score.stride(0),
  };
}

template <auto* f, size_t max_dynamic_smem>
void setup_kernel_smem_once() {
  [[maybe_unused]]
  static const auto result = [] {
#ifdef USE_ROCM
    // hipify will turn cudaFuncSetAttribute -> hipFuncSetAttribute. On ROCm,
    // hipFuncSetAttribute expects `const void*` and hipcc does not accept passing
    // a function pointer directly, so cast explicitly.
    return ::cudaFuncSetAttribute(
        reinterpret_cast<const void*>(f), ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#else
    // CUDA: keep original behavior (no cast needed).
    return ::cudaFuncSetAttribute(f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#endif
  }();
  TORCH_CHECK(result == cudaSuccess, "set_up_kernel_once failed:", ::cudaGetErrorString(result));
}

}  // namespace

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

void fast_topk_interface(
    const at::Tensor& score, at::Tensor& indices, const at::Tensor& lengths, std::optional<at::Tensor> row_starts_opt) {
  CHECK_CUDA(score);
  CHECK_CUDA(indices);
  if (row_starts_opt.has_value()) {
    CHECK_CUDA(row_starts_opt.value());
  }
  CHECK_CUDA(lengths);
  const auto params = get_params(score, lengths, row_starts_opt, indices);
  const auto B = score.size(0);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};
  setup_kernel_smem_once<topk_kernel, kSmem>();
  topk_kernel<<<grid, block, kSmem, stream>>>(params);
  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}

void fast_topk_transform_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& dst_page_table,
    const at::Tensor& src_page_table,
    const at::Tensor& cu_seqlens_q,
    std::optional<at::Tensor> row_starts_opt) {
  CHECK_CUDA(score);
  CHECK_CUDA(lengths);
  CHECK_CUDA(dst_page_table);
  CHECK_CUDA(src_page_table);
  CHECK_CUDA(cu_seqlens_q);
  if (row_starts_opt.has_value()) {
    CHECK_CUDA(row_starts_opt.value());
  }
  const auto params = get_params(score, lengths, row_starts_opt);
  const auto B = score.size(0);
  TORCH_CHECK(dst_page_table.dim() == 2 && dst_page_table.is_contiguous());
  TORCH_CHECK(src_page_table.dim() == 2 && src_page_table.stride(1) == 1);
  TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_q.is_contiguous());
  const auto prefill_bs = cu_seqlens_q.size(0) - 1;
  TORCH_CHECK(dst_page_table.size(0) == B);
  TORCH_CHECK(dst_page_table.size(1) == TopK);
  TORCH_CHECK(src_page_table.size(0) == prefill_bs);
  TORCH_CHECK(prefill_bs <= B);  // prefill_bs should be smaller than expanded bs

  // launch kernel
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};
  const auto src_stride = src_page_table.stride(0);

  // dispatch to decode or prefill
  // extend and draft extend: row_starts_opt is not null, invokes the prefill kernel
  // decode: row_starts_opt is null, invokes the decode kernel
  // target verify: row_starts_opt is null, invokes the prefill kernel
  const auto is_decode = !row_starts_opt.has_value() && prefill_bs == B;
  if (is_decode) {
    setup_kernel_smem_once<topk_transform_decode_kernel, kSmem>();
    topk_transform_decode_kernel<<<grid, block, kSmem, stream>>>(
        params, dst_page_table.data_ptr<int32_t>(), src_page_table.data_ptr<int32_t>(), src_stride);
  } else {
    setup_kernel_smem_once<topk_transform_prefill_kernel, kSmem>();
    topk_transform_prefill_kernel<<<grid, block, kSmem, stream>>>(
        params,
        dst_page_table.data_ptr<int32_t>(),
        src_page_table.data_ptr<int32_t>(),
        src_stride,
        cu_seqlens_q.data_ptr<int32_t>(),
        prefill_bs);
  }

  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}

void fast_topk_transform_ragged_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& topk_indices_ragged,
    const at::Tensor& topk_indices_offset,
    std::optional<at::Tensor> row_starts_opt) {
  CHECK_CUDA(score);
  CHECK_CUDA(lengths);
  CHECK_CUDA(topk_indices_ragged);
  CHECK_CUDA(topk_indices_offset);
  if (row_starts_opt.has_value()) {
    CHECK_CUDA(row_starts_opt.value());
  }

  const auto params = get_params(score, lengths, row_starts_opt);
  const auto B = score.size(0);
  TORCH_CHECK(topk_indices_ragged.dim() == 2 && topk_indices_ragged.is_contiguous());
  TORCH_CHECK(topk_indices_offset.dim() == 1);

  TORCH_CHECK(topk_indices_ragged.size(0) == B);
  TORCH_CHECK(topk_indices_ragged.size(1) == TopK);
  TORCH_CHECK(topk_indices_offset.size(0) == B);

  // launch kernel
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
#if defined(__HGGCCC__)
  constexpr auto kPpuSmem = sizeof(ppu_topk::Smem);
  setup_kernel_smem_once<topk_transform_prefill_ragged_kernel_ppu, kPpuSmem>();
  topk_transform_prefill_ragged_kernel_ppu<<<grid, dim3{ppu_topk::kThreads}, kPpuSmem, stream>>>(
      params, topk_indices_ragged.data_ptr<int32_t>(), topk_indices_offset.data_ptr<int32_t>());
#else
  const auto block = dim3{kThreadsPerBlock};

  setup_kernel_smem_once<topk_transform_prefill_ragged_kernel, kSmem>();
  topk_transform_prefill_ragged_kernel<<<grid, block, kSmem, stream>>>(
      params, topk_indices_ragged.data_ptr<int32_t>(), topk_indices_offset.data_ptr<int32_t>());
#endif

  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}
