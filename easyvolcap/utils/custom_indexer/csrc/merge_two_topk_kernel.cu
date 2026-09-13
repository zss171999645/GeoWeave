#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cub/block/block_radix_sort.cuh>

#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

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
__global__ void merge_two_topk_kernel(
    const scalar_t* __restrict__ topk_scores,
    const int32_t* __restrict__ topk_indices,
    const scalar_t* __restrict__ pending_scores,
    const int64_t* __restrict__ pending_pos,
    scalar_t* __restrict__ out_scores,
    int32_t* __restrict__ out_indices,
    int rows,
    int pending_start) {
    constexpr int N = TOPK * 2;
    static_assert(BLOCK_THREADS * ITEMS_PER_THREAD == N, "BLOCK_THREADS * ITEMS_PER_THREAD must be 2*TOPK");

    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    using BlockSort = cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD, int32_t>;
    __shared__ typename BlockSort::TempStorage temp_storage;

    float keys[ITEMS_PER_THREAD];
    int32_t vals[ITEMS_PER_THREAD];

    int topk_offset = row * TOPK;
    int tid = threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        int pos = tid + i * BLOCK_THREADS;
        if (pos < TOPK) {
            keys[i] = to_float<scalar_t>(topk_scores[topk_offset + pos]);
            vals[i] = topk_indices[topk_offset + pos];
        } else {
            int rel = pos - TOPK;
            keys[i] = to_float<scalar_t>(pending_scores[topk_offset + rel]);
            vals[i] = static_cast<int32_t>(pending_pos[topk_offset + rel]) + pending_start;
        }
    }

    BlockSort(temp_storage).SortDescending(keys, vals);

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        int rank = tid * ITEMS_PER_THREAD + i;
        if (rank < TOPK) {
            out_scores[topk_offset + rank] = from_float<scalar_t>(keys[i]);
            out_indices[topk_offset + rank] = vals[i];
        }
    }
}

}  // namespace

void merge_two_topk_cuda_launcher(
    const torch::Tensor& topk_scores,
    const torch::Tensor& topk_indices,
    const torch::Tensor& pending_scores,
    const torch::Tensor& pending_pos,
    torch::Tensor& out_scores,
    torch::Tensor& out_indices,
    int64_t pending_start) {
    auto sizes = topk_scores.sizes();
    int64_t bsz = sizes[0];
    int64_t tgt = sizes[1];
    int64_t topk = sizes[2];
    TORCH_CHECK(topk == 512, "merge_two_topk_cuda currently supports topk=512 only");

    int rows = static_cast<int>(bsz * tgt);
    constexpr int TOPK = 512;
    auto stream = at::cuda::getDefaultCUDAStream();
    // 128 threads (8 items/thread) is fastest on 4090 for TOPK=512 merge.
    int threads = 128;
    if (const char* env = std::getenv("VGGT_INDEXER_MERGE_TWO_THREADS")) {
        int parsed = std::atoi(env);
        if (parsed == 128 || parsed == 256 || parsed == 512) {
            threads = parsed;
        }
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        topk_scores.scalar_type(),
        "merge_two_topk_cuda_launcher",
        [&] {
            if (threads == 128) {
                merge_two_topk_kernel<scalar_t, TOPK, 128, 8><<<rows, 128, 0, stream>>>(
                    topk_scores.data_ptr<scalar_t>(),
                    topk_indices.data_ptr<int32_t>(),
                    pending_scores.data_ptr<scalar_t>(),
                    pending_pos.data_ptr<int64_t>(),
                    out_scores.data_ptr<scalar_t>(),
                    out_indices.data_ptr<int32_t>(),
                    rows,
                    static_cast<int>(pending_start));
            } else if (threads == 512) {
                merge_two_topk_kernel<scalar_t, TOPK, 512, 2><<<rows, 512, 0, stream>>>(
                    topk_scores.data_ptr<scalar_t>(),
                    topk_indices.data_ptr<int32_t>(),
                    pending_scores.data_ptr<scalar_t>(),
                    pending_pos.data_ptr<int64_t>(),
                    out_scores.data_ptr<scalar_t>(),
                    out_indices.data_ptr<int32_t>(),
                    rows,
                    static_cast<int>(pending_start));
            } else {
                merge_two_topk_kernel<scalar_t, TOPK, 256, 4><<<rows, 256, 0, stream>>>(
                    topk_scores.data_ptr<scalar_t>(),
                    topk_indices.data_ptr<int32_t>(),
                    pending_scores.data_ptr<scalar_t>(),
                    pending_pos.data_ptr<int64_t>(),
                    out_scores.data_ptr<scalar_t>(),
                    out_indices.data_ptr<int32_t>(),
                    rows,
                    static_cast<int>(pending_start));
            }
        });
}
