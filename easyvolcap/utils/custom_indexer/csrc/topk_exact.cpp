#include <torch/extension.h>

void topk_exact_cuda_launcher(
    const torch::Tensor& scores,
    torch::Tensor& out_scores,
    torch::Tensor& out_pos);

void topk_exact_cuda(
    const torch::Tensor& scores,
    torch::Tensor& out_scores,
    torch::Tensor& out_pos) {
    TORCH_CHECK(scores.is_cuda(), "scores must be CUDA tensor");
    TORCH_CHECK(out_scores.is_cuda(), "out_scores must be CUDA tensor");
    TORCH_CHECK(out_pos.is_cuda(), "out_pos must be CUDA tensor");

    TORCH_CHECK(scores.dim() == 3, "scores must be [B, T, C]");
    TORCH_CHECK(out_scores.dim() == 3, "out_scores must be [B, T, K]");
    TORCH_CHECK(out_pos.dim() == 3, "out_pos must be [B, T, K]");

    TORCH_CHECK(scores.scalar_type() == out_scores.scalar_type(), "scores/out_scores dtype mismatch");
    TORCH_CHECK(out_pos.scalar_type() == torch::kLong, "out_pos must be int64");

    TORCH_CHECK(scores.is_contiguous(), "scores must be contiguous");
    TORCH_CHECK(out_scores.is_contiguous(), "out_scores must be contiguous");
    TORCH_CHECK(out_pos.is_contiguous(), "out_pos must be contiguous");

    auto s = scores.sizes();
    auto o = out_scores.sizes();
    auto p = out_pos.sizes();
    TORCH_CHECK(s[0] == o[0] && s[1] == o[1], "batch/row mismatch");
    TORCH_CHECK(s[0] == p[0] && s[1] == p[1], "batch/row mismatch");
    TORCH_CHECK(o[2] == p[2], "topk mismatch");
    TORCH_CHECK(o[2] == 512, "topk_exact_cuda currently supports topk=512 only");
    TORCH_CHECK(s[2] >= o[2], "scores last dim must be >= topk");

    topk_exact_cuda_launcher(scores, out_scores, out_pos);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk_exact_cuda", &topk_exact_cuda, "Exact topk select CUDA");
}

