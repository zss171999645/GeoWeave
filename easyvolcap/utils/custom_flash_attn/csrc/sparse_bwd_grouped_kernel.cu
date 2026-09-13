#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cub/block/block_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int kStage1Threads = 128;
constexpr int kFillThreads = 256;
constexpr int kReduceThreads = 64;
constexpr int64_t kInvalidKey = std::numeric_limits<int64_t>::max();

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t v) {
    return static_cast<float>(v);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(float v) {
    return static_cast<scalar_t>(v);
}

template <typename scalar_t>
__device__ __forceinline__ float load4d(
    const scalar_t* __restrict__ ptr,
    int b,
    int h,
    int t,
    int d,
    int q_layout,
    int64_t s0,
    int64_t s1,
    int64_t s2,
    int64_t s3) {
    int64_t off = 0;
    if (q_layout == 0) {
        // bthd
        off = static_cast<int64_t>(b) * s0 + static_cast<int64_t>(t) * s1 + static_cast<int64_t>(h) * s2 + static_cast<int64_t>(d) * s3;
    } else {
        // bhtd
        off = static_cast<int64_t>(b) * s0 + static_cast<int64_t>(h) * s1 + static_cast<int64_t>(t) * s2 + static_cast<int64_t>(d) * s3;
    }
    return to_float<scalar_t>(ptr[off]);
}

template <typename scalar_t>
__device__ __forceinline__ void store4d(
    scalar_t* __restrict__ ptr,
    int b,
    int h,
    int t,
    int d,
    float v,
    int q_layout,
    int64_t s0,
    int64_t s1,
    int64_t s2,
    int64_t s3) {
    int64_t off = 0;
    if (q_layout == 0) {
        off = static_cast<int64_t>(b) * s0 + static_cast<int64_t>(t) * s1 + static_cast<int64_t>(h) * s2 + static_cast<int64_t>(d) * s3;
    } else {
        off = static_cast<int64_t>(b) * s0 + static_cast<int64_t>(h) * s1 + static_cast<int64_t>(t) * s2 + static_cast<int64_t>(d) * s3;
    }
    ptr[off] = from_float<scalar_t>(v);
}

template <typename scalar_t>
__global__ void stage1_ds_dq_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int32_t* __restrict__ kv_pos,
    const scalar_t* __restrict__ do_tensor,
    const float* __restrict__ delta,
    const scalar_t* __restrict__ prob,
    const scalar_t* __restrict__ dattn,
    float* __restrict__ ds_out,
    float* __restrict__ dq,
    int bsz,
    int nheads,
    int seqlen_q,
    int seqlen_k,
    int head_dim,
    int topk,
    int chunk_start,
    int q_chunk,
    float softmax_scale,
    bool has_dattn,
    int q_layout,
    int64_t q_s0,
    int64_t q_s1,
    int64_t q_s2,
    int64_t q_s3,
    int64_t k_s0,
    int64_t k_s1,
    int64_t k_s2,
    int64_t k_s3,
    int64_t v_s0,
    int64_t v_s1,
    int64_t v_s2,
    int64_t v_s3,
    int64_t do_s0,
    int64_t do_s1,
    int64_t do_s2,
    int64_t do_s3,
    int64_t dq_s0,
    int64_t dq_s1,
    int64_t dq_s2,
    int64_t dq_s3,
    int64_t kv_s0,
    int64_t kv_s1,
    int64_t kv_s2,
    int64_t delta_s0,
    int64_t delta_s1,
    int64_t delta_s2,
    int64_t prob_s0,
    int64_t prob_s1,
    int64_t prob_s2,
    int64_t prob_s3,
    int64_t dattn_s0,
    int64_t dattn_s1,
    int64_t dattn_s2) {
    int row = blockIdx.x;
    int rows = bsz * nheads * q_chunk;
    if (row >= rows) {
        return;
    }

    int bh = row / q_chunk;
    int q_local = row - bh * q_chunk;
    int b = bh / nheads;
    int h = bh % nheads;
    int t = chunk_start + q_local;
    if (t >= seqlen_q) {
        return;
    }

    using BlockReduce = cub::BlockReduce<float, kStage1Threads>;
    __shared__ typename BlockReduce::TempStorage reduce_storage;

    extern __shared__ unsigned char smem[];
    float* ds_shared = reinterpret_cast<float*>(smem);
    int32_t* key_shared = reinterpret_cast<int32_t*>(ds_shared + topk);
    float* do_shared = reinterpret_cast<float*>(key_shared + topk);

    int tid = threadIdx.x;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        do_shared[d] = load4d<scalar_t>(do_tensor, b, h, t, d, q_layout, do_s0, do_s1, do_s2, do_s3);
    }
    __syncthreads();

    float sum_dattn_p_local = 0.0f;
    if (has_dattn) {
        for (int i = tid; i < topk; i += blockDim.x) {
            int64_t kv_off = static_cast<int64_t>(b) * kv_s0 + static_cast<int64_t>(t) * kv_s1 + static_cast<int64_t>(i) * kv_s2;
            int key = kv_pos[kv_off];
            bool valid = (key >= 0) && (key < seqlen_k);
            if (valid) {
                int64_t p_off =
                    static_cast<int64_t>(b) * prob_s0 + static_cast<int64_t>(h) * prob_s1 + static_cast<int64_t>(t) * prob_s2 + static_cast<int64_t>(i) * prob_s3;
                int64_t da_off =
                    static_cast<int64_t>(b) * dattn_s0 + static_cast<int64_t>(t) * dattn_s1 + static_cast<int64_t>(i) * dattn_s2;
                float p = to_float<scalar_t>(prob[p_off]);
                float da = to_float<scalar_t>(dattn[da_off]);
                sum_dattn_p_local += da * p;
            }
        }
    }
    float sum_dattn_p = BlockReduce(reduce_storage).Sum(sum_dattn_p_local);
    __syncthreads();

    float delta_val = delta[
        static_cast<int64_t>(b) * delta_s0 + static_cast<int64_t>(h) * delta_s1 + static_cast<int64_t>(t) * delta_s2
    ];
    float total = delta_val + (has_dattn ? sum_dattn_p : 0.0f);

    int64_t ds_row_off = static_cast<int64_t>(row) * topk;
    for (int i = tid; i < topk; i += blockDim.x) {
        int64_t kv_off = static_cast<int64_t>(b) * kv_s0 + static_cast<int64_t>(t) * kv_s1 + static_cast<int64_t>(i) * kv_s2;
        int key = kv_pos[kv_off];
        bool valid = (key >= 0) && (key < seqlen_k);
        key_shared[i] = key;

        float p = 0.0f;
        float dp = 0.0f;
        if (valid) {
            int64_t p_off =
                static_cast<int64_t>(b) * prob_s0 + static_cast<int64_t>(h) * prob_s1 + static_cast<int64_t>(t) * prob_s2 + static_cast<int64_t>(i) * prob_s3;
            p = to_float<scalar_t>(prob[p_off]);
            for (int d = 0; d < head_dim; ++d) {
                float vv = load4d<scalar_t>(v, b, h, key, d, q_layout, v_s0, v_s1, v_s2, v_s3);
                dp = fmaf(vv, do_shared[d], dp);
            }
            if (has_dattn) {
                int64_t da_off =
                    static_cast<int64_t>(b) * dattn_s0 + static_cast<int64_t>(t) * dattn_s1 + static_cast<int64_t>(i) * dattn_s2;
                dp += to_float<scalar_t>(dattn[da_off]);
            }
        }
        float ds = valid ? ((dp - total) * p) : 0.0f;
        ds_shared[i] = ds;
        ds_out[ds_row_off + i] = ds;
    }
    __syncthreads();

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float dq_acc = 0.0f;
        for (int i = 0; i < topk; ++i) {
            int key = key_shared[i];
            if (key >= 0 && key < seqlen_k) {
                float ds = ds_shared[i];
                float kv = load4d<scalar_t>(k, b, h, key, d, q_layout, k_s0, k_s1, k_s2, k_s3);
                dq_acc = fmaf(ds, kv, dq_acc);
            }
        }
        dq_acc *= softmax_scale;
        store4d<float>(dq, b, h, t, d, dq_acc, q_layout, dq_s0, dq_s1, dq_s2, dq_s3);
    }
}

__global__ void fill_keys_vals_kernel(
    const int32_t* __restrict__ kv_pos,
    int64_t* __restrict__ keys,
    int32_t* __restrict__ vals,
    int bsz,
    int nheads,
    int seqlen_k,
    int topk,
    int chunk_start,
    int q_chunk,
    int64_t kv_s0,
    int64_t kv_s1,
    int64_t kv_s2,
    int entries) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= entries) {
        return;
    }

    int tmp = e;
    int i = tmp % topk;
    tmp /= topk;
    int q_local = tmp % q_chunk;
    tmp /= q_chunk;
    int h = tmp % nheads;
    int b = tmp / nheads;
    int t = chunk_start + q_local;

    int64_t kv_off = static_cast<int64_t>(b) * kv_s0 + static_cast<int64_t>(t) * kv_s1 + static_cast<int64_t>(i) * kv_s2;
    int key = kv_pos[kv_off];
    if (key >= 0 && key < seqlen_k) {
        int64_t packed = (static_cast<int64_t>(b) * nheads + h) * static_cast<int64_t>(seqlen_k) + key;
        keys[e] = packed;
    } else {
        keys[e] = kInvalidKey;
    }
    vals[e] = e;
}

template <typename scalar_t>
__global__ void reduce_runs_kernel(
    const int64_t* __restrict__ unique_keys,
    const int32_t* __restrict__ run_lengths,
    const int32_t* __restrict__ run_offsets,
    const int32_t* __restrict__ sorted_vals,
    const float* __restrict__ ds_out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ do_tensor,
    const scalar_t* __restrict__ prob,
    float* __restrict__ dk,
    float* __restrict__ dv,
    int bsz,
    int nheads,
    int seqlen_q,
    int seqlen_k,
    int head_dim,
    int topk,
    int chunk_start,
    int q_chunk,
    float softmax_scale,
    int q_layout,
    int64_t q_s0,
    int64_t q_s1,
    int64_t q_s2,
    int64_t q_s3,
    int64_t do_s0,
    int64_t do_s1,
    int64_t do_s2,
    int64_t do_s3,
    int64_t prob_s0,
    int64_t prob_s1,
    int64_t prob_s2,
    int64_t prob_s3,
    int64_t dk_s0,
    int64_t dk_s1,
    int64_t dk_s2,
    int64_t dk_s3,
    int64_t dv_s0,
    int64_t dv_s1,
    int64_t dv_s2,
    int64_t dv_s3,
    int num_runs) {
    int run_id = blockIdx.x;
    if (run_id >= num_runs) {
        return;
    }
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= head_dim) {
        return;
    }

    int64_t packed = unique_keys[run_id];
    if (packed == kInvalidKey) {
        return;
    }

    int bh = static_cast<int>(packed / seqlen_k);
    int kv = static_cast<int>(packed - static_cast<int64_t>(bh) * seqlen_k);
    int b = bh / nheads;
    int h = bh % nheads;

    int start = run_offsets[run_id];
    int len = run_lengths[run_id];

    float acc_dk = 0.0f;
    float acc_dv = 0.0f;
    for (int j = 0; j < len; ++j) {
        int e = sorted_vals[start + j];
        int tmp = e;
        int i = tmp % topk;
        tmp /= topk;
        int q_local = tmp % q_chunk;
        tmp /= q_chunk;
        int h2 = tmp % nheads;
        int b2 = tmp / nheads;
        int t = chunk_start + q_local;

        if (t >= seqlen_q || b2 < 0 || b2 >= bsz || h2 < 0 || h2 >= nheads) {
            continue;
        }

        float ds = ds_out[e];
        int64_t p_off =
            static_cast<int64_t>(b2) * prob_s0 + static_cast<int64_t>(h2) * prob_s1 + static_cast<int64_t>(t) * prob_s2 + static_cast<int64_t>(i) * prob_s3;
        float p = to_float<scalar_t>(prob[p_off]);
        float qv = load4d<scalar_t>(q, b2, h2, t, d, q_layout, q_s0, q_s1, q_s2, q_s3);
        float dov = load4d<scalar_t>(do_tensor, b2, h2, t, d, q_layout, do_s0, do_s1, do_s2, do_s3);
        acc_dk = fmaf(ds, qv * softmax_scale, acc_dk);
        acc_dv = fmaf(p, dov, acc_dv);
    }

    int64_t dk_off = 0;
    int64_t dv_off = 0;
    if (q_layout == 0) {
        dk_off = static_cast<int64_t>(b) * dk_s0 + static_cast<int64_t>(kv) * dk_s1 + static_cast<int64_t>(h) * dk_s2 + static_cast<int64_t>(d) * dk_s3;
        dv_off = static_cast<int64_t>(b) * dv_s0 + static_cast<int64_t>(kv) * dv_s1 + static_cast<int64_t>(h) * dv_s2 + static_cast<int64_t>(d) * dv_s3;
    } else {
        dk_off = static_cast<int64_t>(b) * dk_s0 + static_cast<int64_t>(h) * dk_s1 + static_cast<int64_t>(kv) * dk_s2 + static_cast<int64_t>(d) * dk_s3;
        dv_off = static_cast<int64_t>(b) * dv_s0 + static_cast<int64_t>(h) * dv_s1 + static_cast<int64_t>(kv) * dv_s2 + static_cast<int64_t>(d) * dv_s3;
    }

    dk[dk_off] += acc_dk;
    dv[dv_off] += acc_dv;
}

}  // namespace

void sparse_prob_bwd_grouped_cuda_launcher(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& kv_pos,
    const torch::Tensor& do_tensor,
    const torch::Tensor& delta,
    const torch::Tensor& prob,
    const torch::Tensor& dattn,
    torch::Tensor& dq,
    torch::Tensor& dk,
    torch::Tensor& dv,
    double softmax_scale,
    int64_t q_layout,
    int64_t query_chunk) {
    auto q_sizes = q.sizes();
    int bsz = static_cast<int>(q_sizes[0]);
    int nheads = static_cast<int>((q_layout == 0) ? q_sizes[2] : q_sizes[1]);
    int seqlen_q = static_cast<int>((q_layout == 0) ? q_sizes[1] : q_sizes[2]);
    int head_dim = static_cast<int>(q_sizes[3]);
    int seqlen_k = static_cast<int>((q_layout == 0) ? k.sizes()[1] : k.sizes()[2]);
    int topk = static_cast<int>(kv_pos.sizes()[2]);

    int q_chunk = static_cast<int>(query_chunk);
    if (q_chunk > seqlen_q) {
        q_chunk = seqlen_q;
    }

    auto opts_f32 = q.options().dtype(torch::kFloat);
    auto opts_i64 = q.options().dtype(torch::kLong);
    auto opts_i32 = q.options().dtype(torch::kInt);
    auto opts_u8 = q.options().dtype(torch::kByte);

    int64_t max_rows = static_cast<int64_t>(bsz) * nheads * q_chunk;
    int64_t max_entries = max_rows * topk;
    TORCH_CHECK(max_entries <= static_cast<int64_t>(std::numeric_limits<int>::max()), "max_entries overflow int");
    int max_entries_i = static_cast<int>(max_entries);

    auto ds_buf = torch::empty({max_entries}, opts_f32);
    auto keys_in = torch::empty({max_entries}, opts_i64);
    auto keys_out = torch::empty({max_entries}, opts_i64);
    auto vals_in = torch::empty({max_entries}, opts_i32);
    auto vals_out = torch::empty({max_entries}, opts_i32);
    auto uniq_keys = torch::empty({max_entries}, opts_i64);
    auto run_lengths = torch::empty({max_entries}, opts_i32);
    auto run_offsets = torch::empty({max_entries}, opts_i32);
    auto num_runs_dev = torch::zeros({1}, opts_i32);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    size_t sort_temp_bytes = 0;
    size_t rle_temp_bytes = 0;
    size_t scan_temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr,
        sort_temp_bytes,
        keys_in.data_ptr<int64_t>(),
        keys_out.data_ptr<int64_t>(),
        vals_in.data_ptr<int32_t>(),
        vals_out.data_ptr<int32_t>(),
        max_entries_i,
        0,
        8 * sizeof(int64_t),
        stream);
    cub::DeviceRunLengthEncode::Encode(
        nullptr,
        rle_temp_bytes,
        keys_out.data_ptr<int64_t>(),
        uniq_keys.data_ptr<int64_t>(),
        run_lengths.data_ptr<int32_t>(),
        num_runs_dev.data_ptr<int32_t>(),
        max_entries_i,
        stream);
    cub::DeviceScan::ExclusiveSum(
        nullptr,
        scan_temp_bytes,
        run_lengths.data_ptr<int32_t>(),
        run_offsets.data_ptr<int32_t>(),
        max_entries_i,
        stream);

    auto sort_temp = torch::empty({static_cast<int64_t>(sort_temp_bytes > 0 ? sort_temp_bytes : 1)}, opts_u8);
    auto rle_temp = torch::empty({static_cast<int64_t>(rle_temp_bytes > 0 ? rle_temp_bytes : 1)}, opts_u8);
    auto scan_temp = torch::empty({static_cast<int64_t>(scan_temp_bytes > 0 ? scan_temp_bytes : 1)}, opts_u8);

    int num_chunks = (seqlen_q + q_chunk - 1) / q_chunk;
    int32_t num_runs_host = 0;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        q.scalar_type(),
        "sparse_prob_bwd_grouped_cuda",
        [&] {
            for (int c = 0; c < num_chunks; ++c) {
                int chunk_start = c * q_chunk;
                int q_chunk_cur = q_chunk;
                if (chunk_start + q_chunk_cur > seqlen_q) {
                    q_chunk_cur = seqlen_q - chunk_start;
                }
                int rows = bsz * nheads * q_chunk_cur;
                int entries = rows * topk;
                if (rows <= 0 || entries <= 0) {
                    continue;
                }

                size_t smem_bytes = static_cast<size_t>(topk) * (sizeof(float) + sizeof(int32_t)) + static_cast<size_t>(head_dim) * sizeof(float);
                stage1_ds_dq_kernel<scalar_t><<<rows, kStage1Threads, smem_bytes, stream>>>(
                    q.data_ptr<scalar_t>(),
                    k.data_ptr<scalar_t>(),
                    v.data_ptr<scalar_t>(),
                    kv_pos.data_ptr<int32_t>(),
                    do_tensor.data_ptr<scalar_t>(),
                    delta.data_ptr<float>(),
                    prob.data_ptr<scalar_t>(),
                    (dattn.numel() > 0) ? dattn.data_ptr<scalar_t>() : nullptr,
                    ds_buf.data_ptr<float>(),
                    dq.data_ptr<float>(),
                    bsz,
                    nheads,
                    seqlen_q,
                    seqlen_k,
                    head_dim,
                    topk,
                    chunk_start,
                    q_chunk_cur,
                    static_cast<float>(softmax_scale),
                    dattn.numel() > 0,
                    static_cast<int>(q_layout),
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    do_tensor.stride(0), do_tensor.stride(1), do_tensor.stride(2), do_tensor.stride(3),
                    dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                    kv_pos.stride(0), kv_pos.stride(1), kv_pos.stride(2),
                    delta.stride(0), delta.stride(1), delta.stride(2),
                    prob.stride(0), prob.stride(1), prob.stride(2), prob.stride(3),
                    (dattn.numel() > 0) ? dattn.stride(0) : 0,
                    (dattn.numel() > 0) ? dattn.stride(1) : 0,
                    (dattn.numel() > 0) ? dattn.stride(2) : 0);

                int fill_blocks = (entries + kFillThreads - 1) / kFillThreads;
                fill_keys_vals_kernel<<<fill_blocks, kFillThreads, 0, stream>>>(
                    kv_pos.data_ptr<int32_t>(),
                    keys_in.data_ptr<int64_t>(),
                    vals_in.data_ptr<int32_t>(),
                    bsz,
                    nheads,
                    seqlen_k,
                    topk,
                    chunk_start,
                    q_chunk_cur,
                    kv_pos.stride(0),
                    kv_pos.stride(1),
                    kv_pos.stride(2),
                    entries);

                cub::DeviceRadixSort::SortPairs(
                    sort_temp.data_ptr<uint8_t>(),
                    sort_temp_bytes,
                    keys_in.data_ptr<int64_t>(),
                    keys_out.data_ptr<int64_t>(),
                    vals_in.data_ptr<int32_t>(),
                    vals_out.data_ptr<int32_t>(),
                    entries,
                    0,
                    8 * sizeof(int64_t),
                    stream);

                cub::DeviceRunLengthEncode::Encode(
                    rle_temp.data_ptr<uint8_t>(),
                    rle_temp_bytes,
                    keys_out.data_ptr<int64_t>(),
                    uniq_keys.data_ptr<int64_t>(),
                    run_lengths.data_ptr<int32_t>(),
                    num_runs_dev.data_ptr<int32_t>(),
                    entries,
                    stream);

                cudaMemcpyAsync(&num_runs_host, num_runs_dev.data_ptr<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                if (num_runs_host <= 0) {
                    continue;
                }

                cub::DeviceScan::ExclusiveSum(
                    scan_temp.data_ptr<uint8_t>(),
                    scan_temp_bytes,
                    run_lengths.data_ptr<int32_t>(),
                    run_offsets.data_ptr<int32_t>(),
                    num_runs_host,
                    stream);

                dim3 block(kReduceThreads);
                dim3 grid(num_runs_host, (head_dim + kReduceThreads - 1) / kReduceThreads);
                reduce_runs_kernel<scalar_t><<<grid, block, 0, stream>>>(
                    uniq_keys.data_ptr<int64_t>(),
                    run_lengths.data_ptr<int32_t>(),
                    run_offsets.data_ptr<int32_t>(),
                    vals_out.data_ptr<int32_t>(),
                    ds_buf.data_ptr<float>(),
                    q.data_ptr<scalar_t>(),
                    do_tensor.data_ptr<scalar_t>(),
                    prob.data_ptr<scalar_t>(),
                    dk.data_ptr<float>(),
                    dv.data_ptr<float>(),
                    bsz,
                    nheads,
                    seqlen_q,
                    seqlen_k,
                    head_dim,
                    topk,
                    chunk_start,
                    q_chunk_cur,
                    static_cast<float>(softmax_scale),
                    static_cast<int>(q_layout),
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    do_tensor.stride(0), do_tensor.stride(1), do_tensor.stride(2), do_tensor.stride(3),
                    prob.stride(0), prob.stride(1), prob.stride(2), prob.stride(3),
                    dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                    dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                    num_runs_host);
            }
        });
}
