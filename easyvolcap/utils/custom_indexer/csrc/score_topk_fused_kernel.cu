#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cub/block/block_radix_sort.cuh>

#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int kMaxHeads = 16;
constexpr int kMaxQElems = 4096;

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t v) {
    return static_cast<float>(v);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(float v) {
    return static_cast<scalar_t>(v);
}

template <typename scalar_t>
__device__ __forceinline__ float compute_score_cached_q(
    const float* __restrict__ q_cache,
    const scalar_t* __restrict__ k_row,
    const scalar_t* __restrict__ w_row,
    int n_heads,
    int head_dim,
    float softmax_scale) {
    float sum = 0.0f;
    for (int h = 0; h < n_heads; ++h) {
        float dot = 0.0f;
        int off = h * head_dim;
#pragma unroll 8
        for (int d = 0; d < head_dim; ++d) {
            dot = fmaf(q_cache[off + d], to_float<scalar_t>(k_row[off + d]), dot);
        }
        dot *= softmax_scale;
        if (dot > 0.0f) {
            sum = fmaf(dot, to_float<scalar_t>(w_row[h]), sum);
        }
    }
    return sum;
}

template <typename scalar_t, int TOPK, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void score_topk_fused_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ w,
    scalar_t* __restrict__ out_scores,
    int32_t* __restrict__ out_pos,
    int tgt_len,
    int src_len,
    int n_heads,
    int head_dim,
    float softmax_scale) {
    constexpr int MERGE_ITEMS = TOPK * 2;
    static_assert(BLOCK_THREADS * ITEMS_PER_THREAD == MERGE_ITEMS, "THREADS*ITEMS must equal 2*TOPK");

    int row = blockIdx.x;
    int rows = gridDim.x;
    if (row >= rows) {
        return;
    }

    int b = row / tgt_len;
    int t = row - b * tgt_len;

    using BlockSort = cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD, int32_t>;
    __shared__ typename BlockSort::TempStorage temp_storage;
    __shared__ float topk_keys[TOPK];
    __shared__ int32_t topk_vals[TOPK];
    __shared__ float q_cache[kMaxQElems];
    constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

    int tid = threadIdx.x;
    for (int i = tid; i < TOPK; i += BLOCK_THREADS) {
        topk_keys[i] = NEG_INF;
        topk_vals[i] = 0;
    }
    __syncthreads();

    float keys[ITEMS_PER_THREAD];
    int32_t vals[ITEMS_PER_THREAD];

    int q_row_off = ((b * tgt_len + t) * n_heads) * head_dim;
    const scalar_t* q_row = q + q_row_off;
    int q_elems = n_heads * head_dim;
    for (int i = tid; i < q_elems; i += BLOCK_THREADS) {
        q_cache[i] = to_float<scalar_t>(q_row[i]);
    }
    __syncthreads();

    for (int chunk_start = 0; chunk_start < src_len; chunk_start += TOPK) {
#pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            int pos = tid + i * BLOCK_THREADS;
            if (pos < TOPK) {
                keys[i] = topk_keys[pos];
                vals[i] = topk_vals[pos];
            } else {
                int rel = pos - TOPK;
                int s = chunk_start + rel;
                if (s < src_len) {
                    int k_row_off = ((b * src_len + s) * n_heads) * head_dim;
                    int w_row_off = (b * src_len + s) * n_heads;
                    const scalar_t* k_row = k + k_row_off;
                    const scalar_t* w_row = w + w_row_off;
                    keys[i] = compute_score_cached_q<scalar_t>(q_cache, k_row, w_row, n_heads, head_dim, softmax_scale);
                    vals[i] = static_cast<int32_t>(s);
                } else {
                    keys[i] = NEG_INF;
                    vals[i] = 0;
                }
            }
        }

        BlockSort(temp_storage).SortDescending(keys, vals);
        __syncthreads();

#pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            int rank = tid * ITEMS_PER_THREAD + i;
            if (rank < TOPK) {
                topk_keys[rank] = keys[i];
                topk_vals[rank] = vals[i];
            }
        }
        __syncthreads();
    }

    int out_off = row * TOPK;
    for (int i = tid; i < TOPK; i += BLOCK_THREADS) {
        out_scores[out_off + i] = from_float<scalar_t>(topk_keys[i]);
        out_pos[out_off + i] = topk_vals[i];
    }
}

}  // namespace

void score_topk_fused_cuda_launcher(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& w,
    torch::Tensor& out_scores,
    torch::Tensor& out_pos,
    double softmax_scale) {
    auto qsz = q.sizes();
    int bsz = static_cast<int>(qsz[0]);
    int tgt_len = static_cast<int>(qsz[1]);
    int n_heads = static_cast<int>(qsz[2]);
    int head_dim = static_cast<int>(qsz[3]);
    int src_len = static_cast<int>(k.sizes()[1]);
    TORCH_CHECK(n_heads > 0 && n_heads <= kMaxHeads, "score_topk_fused supports 1..", kMaxHeads, " heads");
    TORCH_CHECK(
        n_heads * head_dim <= kMaxQElems,
        "score_topk_fused supports n_heads * head_dim <= ",
        kMaxQElems
    );

    constexpr int TOPK = 512;
    int rows = bsz * tgt_len;

    int threads = 128;
    if (const char* env = std::getenv("VGGT_INDEXER_SCORE_TOPK_FUSED_THREADS")) {
        int parsed = std::atoi(env);
        if (parsed == 64 || parsed == 128 || parsed == 256) {
            threads = parsed;
        }
    }

    auto stream = at::cuda::getDefaultCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        q.scalar_type(),
        "score_topk_fused_cuda_launcher",
        [&] {
            if (threads == 64) {
                score_topk_fused_kernel<scalar_t, TOPK, 64, 16><<<rows, 64, 0, stream>>>(
                    q.data_ptr<scalar_t>(),
                    k.data_ptr<scalar_t>(),
                    w.data_ptr<scalar_t>(),
                    out_scores.data_ptr<scalar_t>(),
                    out_pos.data_ptr<int32_t>(),
                    tgt_len,
                    src_len,
                    n_heads,
                    head_dim,
                    static_cast<float>(softmax_scale));
            } else if (threads == 256) {
                score_topk_fused_kernel<scalar_t, TOPK, 256, 4><<<rows, 256, 0, stream>>>(
                    q.data_ptr<scalar_t>(),
                    k.data_ptr<scalar_t>(),
                    w.data_ptr<scalar_t>(),
                    out_scores.data_ptr<scalar_t>(),
                    out_pos.data_ptr<int32_t>(),
                    tgt_len,
                    src_len,
                    n_heads,
                    head_dim,
                    static_cast<float>(softmax_scale));
            } else {
                score_topk_fused_kernel<scalar_t, TOPK, 128, 8><<<rows, 128, 0, stream>>>(
                    q.data_ptr<scalar_t>(),
                    k.data_ptr<scalar_t>(),
                    w.data_ptr<scalar_t>(),
                    out_scores.data_ptr<scalar_t>(),
                    out_pos.data_ptr<int32_t>(),
                    tgt_len,
                    src_len,
                    n_heads,
                    head_dim,
                    static_cast<float>(softmax_scale));
            }
        });
}
