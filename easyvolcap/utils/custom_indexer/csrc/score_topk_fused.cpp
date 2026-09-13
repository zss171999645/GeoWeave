#include <torch/extension.h>

void score_topk_fused_cuda_launcher(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& w,
    torch::Tensor& out_scores,
    torch::Tensor& out_pos,
    double softmax_scale);

void score_topk_fused_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& w,
    torch::Tensor& out_scores,
    torch::Tensor& out_pos,
    double softmax_scale) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be CUDA tensor");
    TORCH_CHECK(out_scores.is_cuda(), "out_scores must be CUDA tensor");
    TORCH_CHECK(out_pos.is_cuda(), "out_pos must be CUDA tensor");

    TORCH_CHECK(q.dim() == 4, "q must be [B,T,H,D]");
    TORCH_CHECK(k.dim() == 4, "k must be [B,S,H,D]");
    TORCH_CHECK(w.dim() == 3, "w must be [B,S,H]");
    TORCH_CHECK(out_scores.dim() == 3, "out_scores must be [B,T,K]");
    TORCH_CHECK(out_pos.dim() == 3, "out_pos must be [B,T,K]");

    TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q/k dtype mismatch");
    TORCH_CHECK(q.scalar_type() == w.scalar_type(), "q/w dtype mismatch");
    TORCH_CHECK(q.scalar_type() == out_scores.scalar_type(), "q/out_scores dtype mismatch");
    TORCH_CHECK(out_pos.scalar_type() == torch::kInt, "out_pos must be int32");

    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(w.is_contiguous(), "w must be contiguous");
    TORCH_CHECK(out_scores.is_contiguous(), "out_scores must be contiguous");
    TORCH_CHECK(out_pos.is_contiguous(), "out_pos must be contiguous");

    auto qsz = q.sizes();
    auto ksz = k.sizes();
    auto wsz = w.sizes();
    auto osz = out_scores.sizes();
    auto psz = out_pos.sizes();

    TORCH_CHECK(qsz[0] == ksz[0] && qsz[0] == wsz[0], "batch mismatch");
    TORCH_CHECK(ksz[1] == wsz[1], "src_len mismatch between k and w");
    TORCH_CHECK(qsz[2] == ksz[2] && qsz[2] == wsz[2], "n_heads mismatch");
    TORCH_CHECK(qsz[3] == ksz[3], "head_dim mismatch");
    TORCH_CHECK(osz[0] == qsz[0] && osz[1] == qsz[1], "out_scores shape mismatch");
    TORCH_CHECK(psz[0] == qsz[0] && psz[1] == qsz[1], "out_pos shape mismatch");
    TORCH_CHECK(osz[2] == 512 && psz[2] == 512, "score_topk_fused supports topk=512 only");

    score_topk_fused_cuda_launcher(q, k, w, out_scores, out_pos, softmax_scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("score_topk_fused_cuda", &score_topk_fused_cuda, "Fused score+topk CUDA");
}
