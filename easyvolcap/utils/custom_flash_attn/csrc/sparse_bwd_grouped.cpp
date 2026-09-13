#include <torch/extension.h>

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
    int64_t query_chunk);

void sparse_prob_bwd_grouped_cuda(
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
    TORCH_CHECK(q.is_cuda(), "q must be CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be CUDA tensor");
    TORCH_CHECK(kv_pos.is_cuda(), "kv_pos must be CUDA tensor");
    TORCH_CHECK(do_tensor.is_cuda(), "do must be CUDA tensor");
    TORCH_CHECK(delta.is_cuda(), "delta must be CUDA tensor");
    TORCH_CHECK(prob.is_cuda(), "prob must be CUDA tensor");
    TORCH_CHECK(dq.is_cuda(), "dq must be CUDA tensor");
    TORCH_CHECK(dk.is_cuda(), "dk must be CUDA tensor");
    TORCH_CHECK(dv.is_cuda(), "dv must be CUDA tensor");
    TORCH_CHECK(dattn.is_cuda(), "dattn must be CUDA tensor");

    TORCH_CHECK(q_layout == 0 || q_layout == 1, "q_layout must be 0(bthd) or 1(bhtd)");
    TORCH_CHECK(query_chunk > 0, "query_chunk must be > 0");

    TORCH_CHECK(q.dim() == 4, "q must be 4D");
    TORCH_CHECK(k.dim() == 4, "k must be 4D");
    TORCH_CHECK(v.dim() == 4, "v must be 4D");
    TORCH_CHECK(do_tensor.dim() == 4, "do must be 4D");
    TORCH_CHECK(dq.dim() == 4, "dq must be 4D");
    TORCH_CHECK(dk.dim() == 4, "dk must be 4D");
    TORCH_CHECK(dv.dim() == 4, "dv must be 4D");
    TORCH_CHECK(kv_pos.dim() == 3, "kv_pos must be [B,T,K]");
    TORCH_CHECK(prob.dim() == 4, "prob must be [B,H,T,K]");
    TORCH_CHECK(delta.dim() == 3, "delta must be [B,H,T_round]");

    TORCH_CHECK(kv_pos.scalar_type() == torch::kInt, "kv_pos must be int32");
    TORCH_CHECK(delta.scalar_type() == torch::kFloat, "delta must be float32");
    TORCH_CHECK(prob.scalar_type() == q.scalar_type(), "prob dtype must match q dtype");
    TORCH_CHECK(k.scalar_type() == q.scalar_type(), "k dtype must match q dtype");
    TORCH_CHECK(v.scalar_type() == q.scalar_type(), "v dtype must match q dtype");
    TORCH_CHECK(do_tensor.scalar_type() == q.scalar_type(), "do dtype must match q dtype");
    TORCH_CHECK(dq.scalar_type() == torch::kFloat, "dq must be float32 for grouped backward");
    TORCH_CHECK(dk.scalar_type() == torch::kFloat, "dk must be float32 for grouped backward");
    TORCH_CHECK(dv.scalar_type() == torch::kFloat, "dv must be float32 for grouped backward");

    if (dattn.numel() > 0) {
        TORCH_CHECK(dattn.dim() == 3, "dattn must be [B,T,K] when provided");
        TORCH_CHECK(dattn.scalar_type() == q.scalar_type(), "dattn dtype must match q dtype");
    }

    auto q_sizes = q.sizes();
    int64_t b = q_sizes[0];
    int64_t h = (q_layout == 0) ? q_sizes[2] : q_sizes[1];
    int64_t t = (q_layout == 0) ? q_sizes[1] : q_sizes[2];
    int64_t d = q_sizes[3];

    TORCH_CHECK(d <= 128, "grouped backward supports head_dim <= 128");

    TORCH_CHECK(kv_pos.sizes()[0] == b && kv_pos.sizes()[1] == t, "kv_pos shape mismatch");
    TORCH_CHECK(prob.sizes()[0] == b && prob.sizes()[1] == h && prob.sizes()[2] == t, "prob shape mismatch");
    TORCH_CHECK(delta.sizes()[0] == b && delta.sizes()[1] == h && delta.sizes()[2] >= t, "delta shape mismatch");

    TORCH_CHECK(k.sizes() == q.sizes(), "k shape must match q shape in current grouped backward");
    TORCH_CHECK(v.sizes() == q.sizes(), "v shape must match q shape in current grouped backward");
    TORCH_CHECK(do_tensor.sizes() == q.sizes(), "do shape must match q shape");
    TORCH_CHECK(dq.sizes() == q.sizes(), "dq shape must match q shape");
    TORCH_CHECK(dk.sizes() == k.sizes(), "dk shape must match k shape");
    TORCH_CHECK(dv.sizes() == v.sizes(), "dv shape must match v shape");

    sparse_prob_bwd_grouped_cuda_launcher(
        q,
        k,
        v,
        kv_pos,
        do_tensor,
        delta,
        prob,
        dattn,
        dq,
        dk,
        dv,
        softmax_scale,
        q_layout,
        query_chunk);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_prob_bwd_grouped_cuda", &sparse_prob_bwd_grouped_cuda, "Sparse prob-cache grouped backward CUDA");
}
