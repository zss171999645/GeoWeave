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

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t v) {
    return static_cast<float>(v);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(float v) {
    return static_cast<scalar_t>(v);
}

template <typename scalar_t, int TOPK, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void topk_exact_kernel(
    const scalar_t* __restrict__ scores,
    scalar_t* __restrict__ out_scores,
    int64_t* __restrict__ out_pos,
    int rows,
    int cols) {
    constexpr int MERGE_ITEMS = TOPK * 2;
    static_assert(BLOCK_THREADS * ITEMS_PER_THREAD == MERGE_ITEMS, "THREADS*ITEMS must be 2*TOPK");

    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    using BlockSort = cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD, int32_t>;
    __shared__ typename BlockSort::TempStorage temp_storage;
    __shared__ float topk_keys[TOPK];
    __shared__ int32_t topk_vals[TOPK];
    constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

    int tid = threadIdx.x;
    for (int i = tid; i < TOPK; i += BLOCK_THREADS) {
        topk_keys[i] = NEG_INF;
        topk_vals[i] = 0;
    }
    __syncthreads();

    float keys[ITEMS_PER_THREAD];
    int32_t vals[ITEMS_PER_THREAD];
    int row_off = row * cols;

    for (int chunk_start = 0; chunk_start < cols; chunk_start += TOPK) {
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            int pos = tid + i * BLOCK_THREADS;
            if (pos < TOPK) {
                keys[i] = topk_keys[pos];
                vals[i] = topk_vals[pos];
            } else {
                int rel = pos - TOPK;
                int col = chunk_start + rel;
                if (col < cols) {
                    keys[i] = to_float<scalar_t>(scores[row_off + col]);
                    vals[i] = static_cast<int32_t>(col);
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
        out_pos[out_off + i] = static_cast<int64_t>(topk_vals[i]);
    }
}

}  // namespace

void topk_exact_cuda_launcher(
    const torch::Tensor& scores,
    torch::Tensor& out_scores,
    torch::Tensor& out_pos) {
    auto s = scores.sizes();
    int bsz = static_cast<int>(s[0]);
    int rows = static_cast<int>(s[1]);
    int cols = static_cast<int>(s[2]);
    int total_rows = bsz * rows;

    constexpr int TOPK = 512;
    int threads = 128;
    if (const char* env = std::getenv("VGGT_INDEXER_TOPK_EXACT_THREADS")) {
        int parsed = std::atoi(env);
        if (parsed == 64 || parsed == 128 || parsed == 256) {
            threads = parsed;
        }
    }

    auto stream = at::cuda::getDefaultCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        scores.scalar_type(),
        "topk_exact_cuda_launcher",
        [&] {
            if (threads == 64) {
                topk_exact_kernel<scalar_t, TOPK, 64, 16><<<total_rows, 64, 0, stream>>>(
                    scores.data_ptr<scalar_t>(),
                    out_scores.data_ptr<scalar_t>(),
                    out_pos.data_ptr<int64_t>(),
                    total_rows,
                    cols);
            } else if (threads == 256) {
                topk_exact_kernel<scalar_t, TOPK, 256, 4><<<total_rows, 256, 0, stream>>>(
                    scores.data_ptr<scalar_t>(),
                    out_scores.data_ptr<scalar_t>(),
                    out_pos.data_ptr<int64_t>(),
                    total_rows,
                    cols);
            } else {
                topk_exact_kernel<scalar_t, TOPK, 128, 8><<<total_rows, 128, 0, stream>>>(
                    scores.data_ptr<scalar_t>(),
                    out_scores.data_ptr<scalar_t>(),
                    out_pos.data_ptr<int64_t>(),
                    total_rows,
                    cols);
            }
        });
}
