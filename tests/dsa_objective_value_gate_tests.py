import unittest
from unittest import mock


class DSAObjectiveValueGateTests(unittest.TestCase):
    def _build_sparse_attention(self, *, value_gate_enabled: bool):
        import torch

        from easyvolcap.official_vggt.layers.dsa_attention import DSAAttention

        torch.manual_seed(17)
        return DSAAttention(
            dim=8,
            num_heads=2,
            qkv_bias=True,
            proj_bias=True,
            qk_norm=False,
            fused_attn=False,
            indexer_cfg={
                "enabled": True,
                "n_heads": 2,
                "head_dim": 4,
                "topk": 2,
                "score_head_chunk_size": 1,
                "score_key_chunk_size": 4,
                "use_sparse_flash_attn": False,
                "objective_value_gate_enabled": value_gate_enabled,
                "objective_value_gate_scale": 0.2,
                "objective_value_gate_tau": 1.0,
                "objective_value_gate_query_chunk_size": 2,
                "objective_value_gate_value_chunk_size": 1,
            },
        )

    def _set_sparse_state(self, attn):
        from easyvolcap.utils.base_utils import dotdict

        attn.train()
        attn.indexer_state = dotdict(
            enabled=True,
            warmup=False,
            sparse=True,
            compute_loss=False,
            detach_input=True,
            topk=2,
            use_sparse_flash_attn=False,
            loss_weight=1.0,
            eps=1e-6,
            layer_idx=0,
        )

    def test_objective_value_gate_starts_as_identity_for_flat_scores(self):
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch runtime unavailable: {exc}")

        attn = self._build_sparse_attention(value_gate_enabled=True)
        state = attn._state()
        flat_scores = torch.zeros(1, 3, 2)

        gate = attn._build_objective_value_gate(flat_scores, state, target_dtype=torch.float32)

        self.assertEqual(tuple(gate.shape), (1, 1, 3, 2))
        self.assertTrue(torch.allclose(gate, torch.ones_like(gate)))

    def test_apply_attention_to_values_matches_broadcast_sum(self):
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch runtime unavailable: {exc}")

        attn = self._build_sparse_attention(value_gate_enabled=True)
        probs = torch.randn(1, 2, 3, 4).softmax(dim=-1)
        values = torch.randn(1, 2, 3, 4, 5)

        expected = (probs.unsqueeze(-1) * values).sum(dim=3)
        actual = attn._apply_attention_to_values(probs, values)

        self.assertTrue(torch.allclose(expected, actual, atol=1e-6, rtol=1e-6))

    def test_apply_attention_to_values_key_chunk_matches_full(self):
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch runtime unavailable: {exc}")

        attn = self._build_sparse_attention(value_gate_enabled=True)
        probs = torch.randn(1, 2, 3, 5).softmax(dim=-1)
        values = torch.randn(1, 2, 3, 5, 4)

        full = attn._apply_attention_to_values(probs, values)
        chunked = attn._apply_attention_to_values(probs, values, key_chunk_size=2)

        self.assertTrue(torch.allclose(full, chunked, atol=1e-6, rtol=1e-6))

    def test_apply_attention_to_values_key_chunk_backward_matches_full(self):
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch runtime unavailable: {exc}")

        attn = self._build_sparse_attention(value_gate_enabled=True)
        probs_base = torch.randn(1, 2, 3, 5).softmax(dim=-1)
        values_base = torch.randn(1, 2, 3, 5, 4)

        probs_full = probs_base.clone().requires_grad_(True)
        values_full = values_base.clone().requires_grad_(True)
        full = attn._apply_attention_to_values(probs_full, values_full)
        full.square().sum().backward()
        probs_full_grad = probs_full.grad.detach().clone()
        values_full_grad = values_full.grad.detach().clone()

        probs_chunk = probs_base.clone().requires_grad_(True)
        values_chunk = values_base.clone().requires_grad_(True)
        chunked = attn._apply_attention_to_values(probs_chunk, values_chunk, key_chunk_size=2)
        chunked.square().sum().backward()

        self.assertTrue(torch.allclose(full.detach(), chunked.detach(), atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(probs_full_grad, probs_chunk.grad, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(values_full_grad, values_chunk.grad, atol=1e-6, rtol=1e-6))

    def test_apply_attention_to_values_key_chunk_equal_topk_skips_matmul(self):
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch runtime unavailable: {exc}")

        attn = self._build_sparse_attention(value_gate_enabled=True)
        probs = torch.randn(1, 2, 3, 4).softmax(dim=-1)
        values = torch.randn(1, 2, 3, 4, 5)
        expected = (probs.unsqueeze(-1) * values).sum(dim=3)

        with mock.patch("torch.matmul", side_effect=RuntimeError("matmul should not be used")):
            actual = attn._apply_attention_to_values(probs, values, key_chunk_size=4)

        self.assertTrue(torch.allclose(expected, actual, atol=1e-6, rtol=1e-6))

    def test_compute_sparse_scores_matches_broadcast_sum(self):
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch runtime unavailable: {exc}")

        attn = self._build_sparse_attention(value_gate_enabled=True)
        q = torch.randn(1, 2, 3, 5)
        k_sel = torch.randn(1, 2, 3, 4, 5)
        scale = 0.25

        expected = (q.unsqueeze(3) * k_sel).sum(dim=-1) * scale
        actual = attn._compute_sparse_scores(q, k_sel, scale)
        chunked = attn._compute_sparse_scores(q, k_sel, scale, key_chunk_size=2)

        self.assertTrue(torch.allclose(expected, actual, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(expected, chunked, atol=1e-6, rtol=1e-6))

    def test_compute_sparse_scores_key_chunk_equal_topk_skips_matmul(self):
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch runtime unavailable: {exc}")

        attn = self._build_sparse_attention(value_gate_enabled=True)
        q = torch.randn(1, 2, 3, 5)
        k_sel = torch.randn(1, 2, 3, 4, 5)
        scale = 0.25
        expected = (q.unsqueeze(3) * k_sel).sum(dim=-1) * scale

        with mock.patch("torch.matmul", side_effect=RuntimeError("matmul should not be used")):
            actual = attn._compute_sparse_scores(q, k_sel, scale, key_chunk_size=4)

        self.assertTrue(torch.allclose(expected, actual, atol=1e-6, rtol=1e-6))

    def test_compute_sparse_scores_key_chunk_backward_matches_full(self):
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch runtime unavailable: {exc}")

        attn = self._build_sparse_attention(value_gate_enabled=True)
        q_base = torch.randn(1, 2, 3, 5)
        k_sel_base = torch.randn(1, 2, 3, 4, 5)
        scale = 0.25

        q_full = q_base.clone().requires_grad_(True)
        k_sel_full = k_sel_base.clone().requires_grad_(True)
        full = attn._compute_sparse_scores(q_full, k_sel_full, scale)
        full.square().sum().backward()
        q_full_grad = q_full.grad.detach().clone()
        k_sel_full_grad = k_sel_full.grad.detach().clone()

        q_chunk = q_base.clone().requires_grad_(True)
        k_sel_chunk = k_sel_base.clone().requires_grad_(True)
        chunked = attn._compute_sparse_scores(q_chunk, k_sel_chunk, scale, key_chunk_size=2)
        chunked.square().sum().backward()

        self.assertTrue(torch.allclose(full.detach(), chunked.detach(), atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(q_full_grad, q_chunk.grad, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(k_sel_full_grad, k_sel_chunk.grad, atol=1e-6, rtol=1e-6))

    def test_gather_selected_matches_gather_kv(self):
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch runtime unavailable: {exc}")

        attn = self._build_sparse_attention(value_gate_enabled=True)
        k = torch.randn(1, 2, 5, 4)
        v = torch.randn(1, 2, 5, 4)
        idx = torch.tensor([[[0, 2], [1, 4], [0, 3]]], dtype=torch.long)

        k_sel, v_sel = attn._gather_kv(k, v, idx)
        self.assertTrue(torch.allclose(k_sel, attn._gather_selected(k, idx), atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(v_sel, attn._gather_selected(v, idx), atol=1e-6, rtol=1e-6))

    def test_gather_selected_avoids_broadcast_arange_indexing(self):
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch runtime unavailable: {exc}")

        attn = self._build_sparse_attention(value_gate_enabled=True)
        x = torch.randn(1, 2, 5, 4)
        idx = torch.tensor([[[0, 2], [1, 4], [0, 3]]], dtype=torch.long)
        expected = attn._gather_selected(x, idx)

        with mock.patch("torch.arange", side_effect=RuntimeError("broadcast arange should not be used")):
            actual = attn._gather_selected(x, idx)

        self.assertTrue(torch.allclose(expected, actual, atol=1e-6, rtol=1e-6))

    def test_objective_value_gate_lets_main_loss_backpropagate_to_indexer(self):
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch runtime unavailable: {exc}")

        x = torch.randn(1, 4, 8)

        attn_no_gate = self._build_sparse_attention(value_gate_enabled=False)
        self._set_sparse_state(attn_no_gate)
        out_no_gate, _ = attn_no_gate(x.clone())
        out_no_gate.square().sum().backward()
        grad_no_gate = sum(
            float(param.grad.detach().abs().sum().item())
            for param in attn_no_gate.indexer.parameters()
            if param.grad is not None
        )

        attn_gate = self._build_sparse_attention(value_gate_enabled=True)
        attn_gate.load_state_dict(attn_no_gate.state_dict(), strict=False)
        self._set_sparse_state(attn_gate)
        out_gate, _ = attn_gate(x.clone())
        out_gate.square().sum().backward()
        grad_gate = sum(
            float(param.grad.detach().abs().sum().item())
            for param in attn_gate.indexer.parameters()
            if param.grad is not None
        )

        self.assertEqual(grad_no_gate, 0.0)
        self.assertGreater(grad_gate, 0.0)

    def test_objective_value_gate_head_chunk_matches_full_heads(self):
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch runtime unavailable: {exc}")

        from easyvolcap.utils.base_utils import dotdict

        attn = self._build_sparse_attention(value_gate_enabled=True)
        attn.eval()
        q = torch.randn(1, 2, 3, 4)
        k = torch.randn(1, 2, 5, 4)
        v = torch.randn(1, 2, 5, 4)
        topk_indices = torch.tensor([[[0, 2], [1, 4], [0, 3]]], dtype=torch.long)
        topk_scores = torch.tensor([[[0.4, -0.1], [0.8, 0.2], [-0.5, 0.3]]], dtype=torch.float32)

        state_full = dotdict(attn._state())
        state_full.compute_loss = True
        state_full.head_chunk_size = 0
        out_full, p_full = attn._sparse_attention_with_objective_value_gate(
            q, k, v, topk_indices, topk_scores, None, state_full
        )

        state_chunk = dotdict(state_full)
        state_chunk.head_chunk_size = 1
        out_chunk, p_chunk = attn._sparse_attention_with_objective_value_gate(
            q, k, v, topk_indices, topk_scores, None, state_chunk
        )

        self.assertTrue(torch.allclose(out_full, out_chunk, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(p_full, p_chunk, atol=1e-6, rtol=1e-6))

    def test_objective_value_gate_streams_value_gather_in_smaller_chunks(self):
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch runtime unavailable: {exc}")

        from easyvolcap.utils.base_utils import dotdict

        attn = self._build_sparse_attention(value_gate_enabled=True)
        attn.eval()
        q = torch.randn(1, 2, 3, 4)
        k = torch.randn(1, 2, 5, 4)
        v = torch.randn(1, 2, 5, 4)
        topk_indices = torch.tensor([[[0, 2], [1, 4], [0, 3]]], dtype=torch.long)
        topk_scores = torch.tensor([[[0.4, -0.1], [0.8, 0.2], [-0.5, 0.3]]], dtype=torch.float32)

        state = dotdict(attn._state())
        state.compute_loss = False
        state.head_chunk_size = 1
        state.objective_value_gate_value_chunk_size = 2
        state.objective_value_gate_gather_chunk_size = 1

        original_gather = attn._gather_selected
        v_storage_ptr = v.untyped_storage().data_ptr()
        max_value_gather = 0

        def guarded_gather(x, idx):
            nonlocal max_value_gather
            if x.untyped_storage().data_ptr() == v_storage_ptr:
                max_value_gather = max(max_value_gather, int(idx.shape[-1]))
            return original_gather(x, idx)

        attn._gather_selected = guarded_gather
        try:
            attn._sparse_attention_with_objective_value_gate(
                q,
                k,
                v,
                topk_indices,
                topk_scores,
                None,
                state,
            )
        finally:
            attn._gather_selected = original_gather

        self.assertEqual(max_value_gather, 1)

    def test_sparse_flash_value_gate_matches_python_reference(self):
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch runtime unavailable: {exc}")
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")

        from easyvolcap.utils.base_utils import dotdict
        from easyvolcap.utils.custom_flash_attn.sparse_index_flash_attn import (
            sparse_index_flash_attn_value_gate_bhtd_func,
        )

        if sparse_index_flash_attn_value_gate_bhtd_func is None:
            self.skipTest("sparse flash value-gate kernel unavailable")

        device = torch.device("cuda")
        dtype = torch.float16
        attn = self._build_sparse_attention(value_gate_enabled=True).to(device=device, dtype=dtype)
        attn.eval()
        state = dotdict(attn._state())
        state.compute_loss = True

        q_base = torch.randn(1, 2, 4, 4, device=device, dtype=dtype)
        k_base = torch.randn(1, 2, 6, 4, device=device, dtype=dtype)
        v_base = torch.randn(1, 2, 6, 4, device=device, dtype=dtype)
        topk_indices = torch.tensor(
            [[[0, 2, 5], [1, 4, 3], [0, 3, 5], [2, 4, 1]]],
            device=device,
            dtype=torch.long,
        )
        topk_scores_base = torch.tensor(
            [[[0.4, -0.1, 0.2], [0.8, 0.2, -0.3], [-0.5, 0.3, 0.1], [0.6, -0.2, 0.4]]],
            device=device,
            dtype=torch.float32,
        )

        q_ref = q_base.clone().requires_grad_(True)
        k_ref = k_base.clone().requires_grad_(True)
        v_ref = v_base.clone().requires_grad_(True)
        topk_scores_ref = topk_scores_base.clone().requires_grad_(True)
        out_ref, p_ref = attn._sparse_attention_with_objective_value_gate(
            q_ref,
            k_ref,
            v_ref,
            topk_indices,
            topk_scores_ref,
            None,
            state,
        )
        loss_ref = out_ref.float().square().sum() + 0.1 * p_ref.float().square().sum()
        loss_ref.backward()

        q_kernel = q_base.clone().requires_grad_(True)
        k_kernel = k_base.clone().requires_grad_(True)
        v_kernel = v_base.clone().requires_grad_(True)
        topk_scores_kernel = topk_scores_base.clone().requires_grad_(True)
        gate = attn._build_objective_value_gate(topk_scores_kernel, state, target_dtype=dtype)
        out_kernel, p_kernel = sparse_index_flash_attn_value_gate_bhtd_func(
            q_kernel,
            k_kernel,
            v_kernel,
            topk_indices.to(torch.int32),
            gate,
            float(attn.scale),
            True,
        )
        loss_kernel = out_kernel.float().square().sum() + 0.1 * p_kernel.float().square().sum()
        loss_kernel.backward()

        self.assertTrue(torch.allclose(out_ref.float(), out_kernel.float(), atol=3e-2, rtol=3e-2))
        self.assertTrue(torch.allclose(p_ref.float(), p_kernel.float(), atol=3e-2, rtol=3e-2))
        self.assertTrue(torch.allclose(q_ref.grad.float(), q_kernel.grad.float(), atol=3e-2, rtol=3e-2))
        self.assertTrue(torch.allclose(k_ref.grad.float(), k_kernel.grad.float(), atol=3e-2, rtol=3e-2))
        self.assertTrue(torch.allclose(v_ref.grad.float(), v_kernel.grad.float(), atol=3e-2, rtol=3e-2))
        self.assertTrue(
            torch.allclose(topk_scores_ref.grad.float(), topk_scores_kernel.grad.float(), atol=4e-2, rtol=4e-2)
        )


if __name__ == "__main__":
    unittest.main()
