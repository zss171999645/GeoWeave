#include <torch/extension.h>

void merge_two_topk_cuda_launcher(
    const torch::Tensor& topk_scores,
    const torch::Tensor& topk_indices,
    const torch::Tensor& pending_scores,
    const torch::Tensor& pending_pos,
    torch::Tensor& out_scores,
    torch::Tensor& out_indices,
    int64_t pending_start);

void merge_two_topk_cuda(
    const torch::Tensor& topk_scores,
    const torch::Tensor& topk_indices,
    const torch::Tensor& pending_scores,
    const torch::Tensor& pending_pos,
    torch::Tensor& out_scores,
    torch::Tensor& out_indices,
    int64_t pending_start) {
    TORCH_CHECK(topk_scores.is_cuda(), "topk_scores must be CUDA tensor");
    TORCH_CHECK(topk_indices.is_cuda(), "topk_indices must be CUDA tensor");
    TORCH_CHECK(pending_scores.is_cuda(), "pending_scores must be CUDA tensor");
    TORCH_CHECK(pending_pos.is_cuda(), "pending_pos must be CUDA tensor");
    TORCH_CHECK(out_scores.is_cuda(), "out_scores must be CUDA tensor");
    TORCH_CHECK(out_indices.is_cuda(), "out_indices must be CUDA tensor");

    TORCH_CHECK(topk_scores.dim() == 3, "topk_scores must be [B, T, K]");
    TORCH_CHECK(topk_indices.dim() == 3, "topk_indices must be [B, T, K]");
    TORCH_CHECK(pending_scores.dim() == 3, "pending_scores must be [B, T, K]");
    TORCH_CHECK(pending_pos.dim() == 3, "pending_pos must be [B, T, K]");
    TORCH_CHECK(out_scores.dim() == 3, "out_scores must be [B, T, K]");
    TORCH_CHECK(out_indices.dim() == 3, "out_indices must be [B, T, K]");

    TORCH_CHECK(topk_scores.scalar_type() == pending_scores.scalar_type(), "scores dtype mismatch");
    TORCH_CHECK(topk_scores.scalar_type() == out_scores.scalar_type(), "scores output dtype mismatch");

    TORCH_CHECK(topk_indices.scalar_type() == torch::kInt32, "topk_indices must be int32");
    TORCH_CHECK(out_indices.scalar_type() == torch::kInt32, "out_indices must be int32");
    TORCH_CHECK(pending_pos.scalar_type() == torch::kLong, "pending_pos must be int64");

    TORCH_CHECK(topk_scores.sizes() == topk_indices.sizes(), "topk score/index shape mismatch");
    TORCH_CHECK(topk_scores.sizes() == pending_scores.sizes(), "topk/pending score shape mismatch");
    TORCH_CHECK(topk_scores.sizes() == pending_pos.sizes(), "topk/pending pos shape mismatch");
    TORCH_CHECK(topk_scores.sizes() == out_scores.sizes(), "topk/out score shape mismatch");
    TORCH_CHECK(topk_scores.sizes() == out_indices.sizes(), "topk/out index shape mismatch");

    TORCH_CHECK(topk_scores.is_contiguous(), "topk_scores must be contiguous");
    TORCH_CHECK(topk_indices.is_contiguous(), "topk_indices must be contiguous");
    TORCH_CHECK(pending_scores.is_contiguous(), "pending_scores must be contiguous");
    TORCH_CHECK(pending_pos.is_contiguous(), "pending_pos must be contiguous");
    TORCH_CHECK(out_scores.is_contiguous(), "out_scores must be contiguous");
    TORCH_CHECK(out_indices.is_contiguous(), "out_indices must be contiguous");

    merge_two_topk_cuda_launcher(
        topk_scores,
        topk_indices,
        pending_scores,
        pending_pos,
        out_scores,
        out_indices,
        pending_start);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("merge_two_topk_cuda", &merge_two_topk_cuda, "Merge two top-k lists CUDA");
}
