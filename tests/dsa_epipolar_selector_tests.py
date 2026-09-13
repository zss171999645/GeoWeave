import unittest

import torch

from easyvolcap.utils.base_utils import dotdict

try:
    import easyvolcap.official_vggt.layers.dsa_attention as dsa_attention_module
    from easyvolcap.official_vggt.layers.dsa_attention import DSAAttention
except Exception as exc:  # pragma: no cover - local env fallback
    dsa_attention_module = None
    DSAAttention = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _rectified_stereo_extrinsics() -> torch.Tensor:
    extrinsics = torch.zeros(1, 2, 3, 4, dtype=torch.float32)
    extrinsics[:, :, :3, :3] = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3)
    extrinsics[:, 1, 0, 3] = -1.0
    return extrinsics


def _identity_intrinsics() -> torch.Tensor:
    return torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 2, 1, 1)


def _identity_extrinsics() -> torch.Tensor:
    extrinsics = torch.zeros(1, 2, 3, 4, dtype=torch.float32)
    extrinsics[:, :, :3, :3] = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3)
    return extrinsics


class DSAEpipolarSelectorTests(unittest.TestCase):
    def test_compute_epipolar_selector_band_loss_can_checkpoint_blocks(self):
        if DSAAttention is None or dsa_attention_module is None:
            self.skipTest(f"DSAAttention import unavailable in local env: {_IMPORT_ERROR}")

        common_indexer_cfg = dict(
            enabled=True,
            n_heads=1,
            head_dim=4,
            use_topk_kernel=False,
            use_sparse_flash_attn=False,
            epipolar_selector_band_enabled=True,
            epipolar_selector_band_px=0.75,
            epipolar_selector_query_stride=1,
        )
        attention = DSAAttention(
            dim=4,
            num_heads=1,
            indexer_cfg=dotdict(common_indexer_cfg),
        )

        state = dotdict(
            enabled=True,
            sparse=True,
            compute_loss=True,
            topk=2,
            loss_weight=1.0,
            eps=1e-6,
            tokens_per_view=10,
            num_views=2,
            patch_start_idx=1,
            patch_grid_height=3,
            patch_grid_width=3,
            image_height=6,
            image_width=6,
            extrinsics=_rectified_stereo_extrinsics(),
            intrinsics=_identity_intrinsics(),
            epipolar_selector_band_enabled=True,
            epipolar_selector_band_px=0.75,
            epipolar_selector_query_stride=1,
            epipolar_selector_checkpoint_blocks=True,
        )

        projected_indexer = (
            torch.randn(1, 20, 1, 4, dtype=torch.float32, requires_grad=True),
            torch.randn(1, 20, 1, 4, dtype=torch.float32, requires_grad=True),
            torch.randn(1, 20, 1, dtype=torch.float32, requires_grad=True),
        )

        original_checkpoint = dsa_attention_module.activation_checkpoint
        checkpoint_calls = []

        def fake_checkpoint(function, *args, **kwargs):
            checkpoint_calls.append((args, kwargs))
            return function(*args)

        dsa_attention_module.activation_checkpoint = fake_checkpoint
        try:
            loss = attention._compute_epipolar_selector_band_loss(
                projected_indexer=projected_indexer,
                indexer_input=torch.zeros(1, 20, 4, dtype=torch.float32),
                pos=None,
                state=state,
                index_mask=None,
                view_bias_data=None,
            )
        finally:
            dsa_attention_module.activation_checkpoint = original_checkpoint

        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(len(checkpoint_calls), 0)

    def test_compute_epipolar_selector_band_loss_recomputes_geometry_inside_checkpoint(self):
        if DSAAttention is None or dsa_attention_module is None:
            self.skipTest(f"DSAAttention import unavailable in local env: {_IMPORT_ERROR}")

        attention = DSAAttention(
            dim=4,
            num_heads=1,
            indexer_cfg=dotdict(
                enabled=True,
                n_heads=1,
                head_dim=4,
                use_topk_kernel=False,
                use_sparse_flash_attn=False,
                epipolar_selector_band_enabled=True,
                epipolar_selector_band_px=0.75,
                epipolar_selector_query_stride=1,
            ),
        )

        state = dotdict(
            enabled=True,
            sparse=True,
            compute_loss=True,
            topk=2,
            loss_weight=1.0,
            eps=1e-6,
            tokens_per_view=10,
            num_views=2,
            patch_start_idx=1,
            patch_grid_height=3,
            patch_grid_width=3,
            image_height=6,
            image_width=6,
            extrinsics=_rectified_stereo_extrinsics(),
            intrinsics=_identity_intrinsics(),
            epipolar_selector_band_enabled=True,
            epipolar_selector_band_px=0.75,
            epipolar_selector_query_stride=1,
            epipolar_selector_checkpoint_blocks=True,
        )

        projected_indexer = (
            torch.randn(1, 20, 1, 4, dtype=torch.float32, requires_grad=True),
            torch.randn(1, 20, 1, 4, dtype=torch.float32, requires_grad=True),
            torch.randn(1, 20, 1, dtype=torch.float32, requires_grad=True),
        )

        original_checkpoint = dsa_attention_module.activation_checkpoint
        original_build_band_mask = dsa_attention_module.build_epipolar_band_mask
        checkpoint_calls = []
        geometry_calls = []

        def fake_checkpoint(function, *args, **kwargs):
            checkpoint_calls.append((args, kwargs))
            first = function(*args)
            second = function(*args)
            for a, b in zip(first, second):
                torch.testing.assert_close(a, b)
            return first

        def counted_build_band_mask(*args, **kwargs):
            geometry_calls.append((args, kwargs))
            return original_build_band_mask(*args, **kwargs)

        dsa_attention_module.activation_checkpoint = fake_checkpoint
        dsa_attention_module.build_epipolar_band_mask = counted_build_band_mask
        try:
            loss = attention._compute_epipolar_selector_band_loss(
                projected_indexer=projected_indexer,
                indexer_input=torch.zeros(1, 20, 4, dtype=torch.float32),
                pos=None,
                state=state,
                index_mask=None,
                view_bias_data=None,
            )
        finally:
            dsa_attention_module.activation_checkpoint = original_checkpoint
            dsa_attention_module.build_epipolar_band_mask = original_build_band_mask

        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(len(checkpoint_calls), 0)
        self.assertEqual(len(geometry_calls), len(checkpoint_calls) * 3)

    def test_compute_epipolar_selector_band_loss_checkpoint_backward_binds_query_chunk(self):
        if DSAAttention is None:
            self.skipTest(f"DSAAttention import unavailable in local env: {_IMPORT_ERROR}")

        attention = DSAAttention(
            dim=4,
            num_heads=1,
            indexer_cfg=dotdict(
                enabled=True,
                n_heads=1,
                head_dim=4,
                use_topk_kernel=False,
                use_sparse_flash_attn=False,
                epipolar_selector_band_enabled=True,
                epipolar_selector_band_px=0.75,
                epipolar_selector_query_stride=1,
            ),
        )

        state = dotdict(
            enabled=True,
            sparse=True,
            compute_loss=True,
            topk=2,
            loss_weight=1.0,
            eps=1e-6,
            tokens_per_view=10,
            num_views=2,
            patch_start_idx=1,
            patch_grid_height=3,
            patch_grid_width=3,
            image_height=6,
            image_width=6,
            extrinsics=_rectified_stereo_extrinsics(),
            intrinsics=_identity_intrinsics(),
            epipolar_selector_band_enabled=True,
            epipolar_selector_band_px=0.75,
            epipolar_selector_query_stride=1,
            epipolar_selector_query_chunk_size=8,
            epipolar_selector_checkpoint_blocks=True,
        )

        projected_indexer = (
            torch.randn(1, 20, 1, 4, dtype=torch.float32, requires_grad=True),
            torch.randn(1, 20, 1, 4, dtype=torch.float32, requires_grad=True),
            torch.randn(1, 20, 1, dtype=torch.float32, requires_grad=True),
        )
        loss = attention._compute_epipolar_selector_band_loss(
            projected_indexer=projected_indexer,
            indexer_input=torch.zeros(1, 20, 4, dtype=torch.float32),
            pos=None,
            state=state,
            index_mask=None,
            view_bias_data=None,
        )

        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(projected_indexer[0].grad)

    def test_compute_epipolar_selector_band_loss_respects_source_view_chunk_size(self):
        if DSAAttention is None:
            self.skipTest(f"DSAAttention import unavailable in local env: {_IMPORT_ERROR}")

        common_indexer_cfg = dict(
            enabled=True,
            n_heads=1,
            head_dim=4,
            use_topk_kernel=False,
            use_sparse_flash_attn=False,
            epipolar_selector_band_enabled=True,
            epipolar_selector_band_px=0.75,
            epipolar_selector_query_stride=1,
        )
        attention = DSAAttention(
            dim=4,
            num_heads=1,
            indexer_cfg=dotdict(common_indexer_cfg),
        )

        state = dotdict(
            enabled=True,
            sparse=True,
            compute_loss=True,
            topk=2,
            loss_weight=1.0,
            eps=1e-6,
            tokens_per_view=10,
            num_views=2,
            patch_start_idx=1,
            patch_grid_height=3,
            patch_grid_width=3,
            image_height=6,
            image_width=6,
            extrinsics=_rectified_stereo_extrinsics(),
            intrinsics=_identity_intrinsics(),
            epipolar_selector_band_enabled=True,
            epipolar_selector_band_px=0.75,
            epipolar_selector_query_stride=1,
            epipolar_selector_source_view_chunk_size=1,
        )

        projected_indexer = (
            torch.randn(1, 20, 1, 4, dtype=torch.float32),
            torch.randn(1, 20, 1, 4, dtype=torch.float32),
            torch.randn(1, 20, 1, dtype=torch.float32),
        )

        patch_token_count = state.tokens_per_view - state.patch_start_idx
        original_select = attention.indexer.select_topk_projected

        def guarded_select(q, k, w, **kwargs):
            self.assertLessEqual(k.shape[1], patch_token_count)
            self.assertLessEqual(w.shape[1], patch_token_count)
            return original_select(q, k, w, **kwargs)

        attention.indexer.select_topk_projected = guarded_select
        loss = attention._compute_epipolar_selector_band_loss(
            projected_indexer=projected_indexer,
            indexer_input=torch.zeros(1, 20, 4, dtype=torch.float32),
            pos=None,
            state=state,
            index_mask=None,
            view_bias_data=None,
        )

        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))

    def test_compute_epipolar_selector_band_loss_can_downsample_loss_grid(self):
        if DSAAttention is None:
            self.skipTest(f"DSAAttention import unavailable in local env: {_IMPORT_ERROR}")

        attention = DSAAttention(
            dim=4,
            num_heads=1,
            indexer_cfg=dotdict(
                enabled=True,
                n_heads=1,
                head_dim=4,
                use_topk_kernel=False,
                use_sparse_flash_attn=False,
                epipolar_selector_band_enabled=True,
                epipolar_selector_band_px=100.0,
                epipolar_selector_query_stride=1,
            ),
        )

        state = dotdict(
            enabled=True,
            sparse=True,
            compute_loss=True,
            topk=2,
            loss_weight=1.0,
            eps=1e-6,
            tokens_per_view=17,
            num_views=2,
            patch_start_idx=1,
            patch_grid_height=4,
            patch_grid_width=4,
            image_height=8,
            image_width=8,
            extrinsics=_rectified_stereo_extrinsics(),
            intrinsics=_identity_intrinsics(),
            epipolar_selector_band_enabled=True,
            epipolar_selector_band_px=100.0,
            epipolar_selector_query_stride=1,
            epipolar_selector_query_chunk_size=64,
            epipolar_selector_source_view_chunk_size=2,
            epipolar_selector_loss_downsample_factor=2,
        )

        total_tokens = state.tokens_per_view * state.num_views
        projected_indexer = (
            torch.randn(1, total_tokens, 1, 4, dtype=torch.float32),
            torch.randn(1, total_tokens, 1, 4, dtype=torch.float32),
            torch.randn(1, total_tokens, 1, dtype=torch.float32),
        )

        original_select = attention.indexer.select_topk_projected
        score_shapes = []

        def tracked_select(q, k, w, **kwargs):
            score_shapes.append((q.shape[1], k.shape[1], w.shape[1]))
            return original_select(q, k, w, **kwargs)

        attention.indexer.select_topk_projected = tracked_select
        loss = attention._compute_epipolar_selector_band_loss(
            projected_indexer=projected_indexer,
            indexer_input=torch.zeros(1, total_tokens, 4, dtype=torch.float32),
            pos=None,
            state=state,
            index_mask=None,
            view_bias_data=None,
        )

        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(score_shapes, [(8, 8, 8)])

    def test_compute_epipolar_selector_band_loss_empty_support_keeps_graph(self):
        if DSAAttention is None:
            self.skipTest(f"DSAAttention import unavailable in local env: {_IMPORT_ERROR}")

        attention = DSAAttention(
            dim=4,
            num_heads=1,
            indexer_cfg=dotdict(
                enabled=True,
                n_heads=1,
                head_dim=4,
                use_topk_kernel=False,
                use_sparse_flash_attn=False,
                epipolar_selector_band_enabled=True,
                epipolar_selector_band_px=0.75,
                epipolar_selector_query_stride=1,
            ),
        )

        state = dotdict(
            enabled=True,
            sparse=True,
            compute_loss=True,
            topk=2,
            loss_weight=1.0,
            eps=1e-6,
            tokens_per_view=10,
            num_views=2,
            patch_start_idx=1,
            patch_grid_height=3,
            patch_grid_width=3,
            image_height=6,
            image_width=6,
            extrinsics=_rectified_stereo_extrinsics(),
            intrinsics=_identity_intrinsics(),
            epipolar_selector_band_enabled=True,
            epipolar_selector_band_px=0.75,
            epipolar_selector_query_stride=1,
            epipolar_selector_source_view_chunk_size=2,
            epipolar_selector_checkpoint_blocks=False,
        )

        projected_indexer = (
            torch.randn(1, 20, 1, 4, dtype=torch.float32, requires_grad=True),
            torch.randn(1, 20, 1, 4, dtype=torch.float32, requires_grad=True),
            torch.randn(1, 20, 1, dtype=torch.float32, requires_grad=True),
        )
        index_mask = torch.zeros(1, 20, 20, dtype=torch.bool)
        loss = attention._compute_epipolar_selector_band_loss(
            projected_indexer=projected_indexer,
            indexer_input=torch.zeros(1, 20, 4, dtype=torch.float32),
            pos=None,
            state=state,
            index_mask=index_mask,
            view_bias_data=None,
        )

        self.assertIsNotNone(loss)
        self.assertTrue(loss.requires_grad)
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        for tensor in projected_indexer:
            self.assertIsNotNone(tensor.grad)

    def test_topk_anchor_geometry_loss_empty_support_keeps_graph(self):
        if DSAAttention is None:
            self.skipTest(f"DSAAttention import unavailable in local env: {_IMPORT_ERROR}")

        attention = DSAAttention(
            dim=4,
            num_heads=1,
            indexer_cfg=dotdict(
                enabled=True,
                n_heads=1,
                head_dim=4,
                use_topk_kernel=False,
                use_sparse_flash_attn=False,
                epipolar_selector_band_enabled=True,
                epipolar_selector_loss_mode="topk_anchor",
                epipolar_selector_band_px=0.75,
                epipolar_selector_query_stride=1,
            ),
        )

        state = dotdict(
            enabled=True,
            sparse=True,
            compute_loss=True,
            topk=2,
            loss_weight=1.0,
            eps=1e-6,
            tokens_per_view=10,
            num_views=2,
            patch_start_idx=1,
            patch_grid_height=3,
            patch_grid_width=3,
            image_height=6,
            image_width=6,
            extrinsics=_rectified_stereo_extrinsics(),
            intrinsics=_identity_intrinsics(),
            epipolar_selector_band_enabled=True,
            epipolar_selector_loss_mode="topk_anchor",
            epipolar_selector_band_px=0.75,
            epipolar_selector_query_stride=1,
            epipolar_selector_query_chunk_size=4,
        )

        projected_indexer = (
            torch.randn(1, 20, 1, 4, dtype=torch.float32, requires_grad=True),
            torch.randn(1, 20, 1, 4, dtype=torch.float32, requires_grad=True),
            torch.randn(1, 20, 1, dtype=torch.float32, requires_grad=True),
        )
        topk_indices = torch.full((1, 20, 2), state.tokens_per_view + state.patch_start_idx, dtype=torch.long)
        topk_scores = torch.randn(1, 20, 2, dtype=torch.float32, requires_grad=True)
        index_mask = torch.zeros(1, 20, 20, dtype=torch.bool)
        loss = attention._compute_epipolar_selector_band_loss(
            projected_indexer=projected_indexer,
            indexer_input=torch.zeros(1, 20, 4, dtype=torch.float32),
            pos=None,
            state=state,
            index_mask=index_mask,
            view_bias_data=None,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
        )

        self.assertIsNotNone(loss)
        self.assertTrue(loss.requires_grad)
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        for tensor in projected_indexer:
            self.assertIsNotNone(tensor.grad)

    def test_topk_geometry_loss_skips_anchor_path_and_records_valid_query_ratio(self):
        if DSAAttention is None:
            self.skipTest(f"DSAAttention import unavailable in local env: {_IMPORT_ERROR}")

        attention = DSAAttention(
            dim=4,
            num_heads=1,
            indexer_cfg=dotdict(
                enabled=True,
                n_heads=1,
                head_dim=4,
                use_topk_kernel=False,
                use_sparse_flash_attn=False,
                epipolar_selector_band_enabled=True,
                epipolar_selector_loss_mode="topk",
                epipolar_selector_band_px=3.0,
                epipolar_selector_query_stride=1,
            ),
        )

        state = dotdict(
            enabled=True,
            sparse=True,
            compute_loss=True,
            topk=3,
            loss_weight=1.0,
            eps=1e-6,
            tokens_per_view=10,
            num_views=2,
            patch_start_idx=1,
            patch_grid_height=3,
            patch_grid_width=3,
            image_height=6,
            image_width=6,
            extrinsics=_rectified_stereo_extrinsics(),
            intrinsics=_identity_intrinsics(),
            epipolar_selector_band_enabled=True,
            epipolar_selector_loss_mode="topk",
            epipolar_selector_band_px=3.0,
            epipolar_selector_query_stride=1,
            epipolar_selector_query_chunk_size=2,
        )

        total_tokens = state.tokens_per_view * state.num_views
        projected_indexer = (
            torch.randn(1, total_tokens, 1, 4, dtype=torch.float32, requires_grad=True),
            torch.randn(1, total_tokens, 1, 4, dtype=torch.float32, requires_grad=True),
            torch.randn(1, total_tokens, 1, dtype=torch.float32, requires_grad=True),
        )
        topk_indices = torch.full((1, total_tokens, 3), state.tokens_per_view + state.patch_start_idx, dtype=torch.long)
        topk_indices[..., 1] = state.tokens_per_view + state.patch_start_idx + 3
        topk_indices[..., 2] = state.tokens_per_view + state.patch_start_idx + 6
        topk_scores = torch.randn(1, total_tokens, 3, dtype=torch.float32, requires_grad=True)

        def fail_anchor_path(*args, **kwargs):
            raise AssertionError("topk mode should not build geometry anchors")

        attention._build_geometry_anchor_indices = fail_anchor_path
        loss = attention._compute_epipolar_selector_band_loss(
            projected_indexer=projected_indexer,
            indexer_input=torch.zeros(1, total_tokens, 4, dtype=torch.float32),
            pos=None,
            state=state,
            index_mask=None,
            view_bias_data=None,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
        )

        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        stats = attention.last_epipolar_selector_stats
        self.assertIsNotNone(stats)
        self.assertIn("epipolar_selector_valid_query_ratio", stats)
        ratio = float(stats.epipolar_selector_valid_query_ratio)
        self.assertGreaterEqual(ratio, 0.0)
        self.assertLessEqual(ratio, 1.0)
        self.assertGreater(float(stats.epipolar_selector_candidate_queries), 0.0)

    def test_geometry_support_auto_uses_depth_when_query_depth_is_valid(self):
        if DSAAttention is None or dsa_attention_module is None:
            self.skipTest(f"DSAAttention import unavailable in local env: {_IMPORT_ERROR}")

        attention = DSAAttention(
            dim=4,
            num_heads=1,
            indexer_cfg=dotdict(
                enabled=True,
                n_heads=1,
                head_dim=4,
                use_topk_kernel=False,
                use_sparse_flash_attn=False,
                geometry_support_enabled=True,
                geometry_support_type="auto",
                geometry_support_depth_radius_px=0.75,
                epipolar_selector_query_stride=1,
            ),
        )

        depths = torch.zeros(1, 2, 6, 6, dtype=torch.float32)
        depths[:, :, 3, 3] = 2.0
        state = dotdict(
            enabled=True,
            sparse=True,
            compute_loss=True,
            topk=2,
            loss_weight=1.0,
            eps=1e-6,
            tokens_per_view=10,
            num_views=2,
            patch_start_idx=1,
            patch_grid_height=3,
            patch_grid_width=3,
            image_height=6,
            image_width=6,
            extrinsics=_rectified_stereo_extrinsics(),
            intrinsics=_identity_intrinsics(),
            depths=depths,
            point_masks=depths > 0,
            geometry_support_enabled=True,
            geometry_support_type="auto",
            geometry_support_depth_radius_px=0.75,
            epipolar_selector_band_px=0.75,
            epipolar_selector_query_stride=1,
            epipolar_selector_query_chunk_size=64,
            epipolar_selector_source_view_chunk_size=2,
            epipolar_selector_loss_downsample_factor=3,
        )

        projected_indexer = (
            torch.randn(1, 20, 1, 4, dtype=torch.float32),
            torch.randn(1, 20, 1, 4, dtype=torch.float32),
            torch.randn(1, 20, 1, dtype=torch.float32),
        )

        original_build_epipolar = dsa_attention_module.build_epipolar_band_mask
        epipolar_calls = []

        def counted_epipolar(*args, **kwargs):
            epipolar_calls.append((args, kwargs))
            return original_build_epipolar(*args, **kwargs)

        dsa_attention_module.build_epipolar_band_mask = counted_epipolar
        try:
            loss = attention._compute_epipolar_selector_band_loss(
                projected_indexer=projected_indexer,
                indexer_input=torch.zeros(1, 20, 4, dtype=torch.float32),
                pos=None,
                state=state,
                index_mask=None,
                view_bias_data=None,
            )
        finally:
            dsa_attention_module.build_epipolar_band_mask = original_build_epipolar

        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(epipolar_calls, [])

    def test_geometry_support_auto_falls_back_only_for_depth_invalid_queries(self):
        if DSAAttention is None or dsa_attention_module is None:
            self.skipTest(f"DSAAttention import unavailable in local env: {_IMPORT_ERROR}")

        attention = DSAAttention(
            dim=4,
            num_heads=1,
            indexer_cfg=dotdict(
                enabled=True,
                n_heads=1,
                head_dim=4,
                use_topk_kernel=False,
                use_sparse_flash_attn=False,
                geometry_support_enabled=True,
                geometry_support_type="auto",
                geometry_support_depth_radius_px=0.75,
                epipolar_selector_query_stride=1,
            ),
        )

        depths = torch.zeros(1, 2, 6, 6, dtype=torch.float32)
        depths[:, 0, 3, 3] = 2.0
        state = dotdict(
            enabled=True,
            sparse=True,
            compute_loss=True,
            topk=2,
            loss_weight=1.0,
            eps=1e-6,
            tokens_per_view=10,
            num_views=2,
            patch_start_idx=1,
            patch_grid_height=3,
            patch_grid_width=3,
            image_height=6,
            image_width=6,
            extrinsics=_rectified_stereo_extrinsics(),
            intrinsics=_identity_intrinsics(),
            depths=depths,
            point_masks=depths > 0,
            geometry_support_enabled=True,
            geometry_support_type="auto",
            geometry_support_depth_radius_px=0.75,
            epipolar_selector_band_px=0.75,
            epipolar_selector_query_stride=1,
            epipolar_selector_query_chunk_size=64,
            epipolar_selector_source_view_chunk_size=2,
            epipolar_selector_loss_downsample_factor=3,
        )

        projected_indexer = (
            torch.randn(1, 20, 1, 4, dtype=torch.float32),
            torch.randn(1, 20, 1, 4, dtype=torch.float32),
            torch.randn(1, 20, 1, dtype=torch.float32),
        )

        original_build_epipolar = dsa_attention_module.build_epipolar_band_mask
        epipolar_query_indices = []

        def counted_epipolar(*args, **kwargs):
            epipolar_query_indices.append(kwargs["query_token_indices"].clone())
            return original_build_epipolar(*args, **kwargs)

        dsa_attention_module.build_epipolar_band_mask = counted_epipolar
        try:
            loss = attention._compute_epipolar_selector_band_loss(
                projected_indexer=projected_indexer,
                indexer_input=torch.zeros(1, 20, 4, dtype=torch.float32),
                pos=None,
                state=state,
                index_mask=None,
                view_bias_data=None,
            )
        finally:
            dsa_attention_module.build_epipolar_band_mask = original_build_epipolar

        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(len(epipolar_query_indices), 1)
        self.assertEqual(epipolar_query_indices[0].tolist(), [[15]])

    def test_geometry_support_weight_overrides_epipolar_band_weight(self):
        if DSAAttention is None:
            self.skipTest(f"DSAAttention import unavailable in local env: {_IMPORT_ERROR}")

        attention = DSAAttention(dim=4, num_heads=1, indexer_cfg=dotdict(enabled=False))

        self.assertAlmostEqual(
            attention._epipolar_selector_loss_weight(
                dotdict(
                    geometry_support_enabled=True,
                    geometry_support_weight=0.2,
                    epipolar_selector_band_weight=0.9,
                )
            ),
            0.2,
        )
        self.assertAlmostEqual(
            attention._epipolar_selector_loss_weight(
                dotdict(
                    geometry_support_enabled=False,
                    epipolar_selector_band_weight=0.9,
                )
            ),
            0.9,
        )

    def test_compute_epipolar_selector_band_loss_respects_query_chunk_size(self):
        if DSAAttention is None:
            self.skipTest(f"DSAAttention import unavailable in local env: {_IMPORT_ERROR}")

        common_indexer_cfg = dict(
            enabled=True,
            n_heads=1,
            head_dim=4,
            use_topk_kernel=False,
            use_sparse_flash_attn=False,
            epipolar_selector_band_enabled=True,
            epipolar_selector_band_px=0.75,
            epipolar_selector_query_stride=1,
        )
        chunked_attention = DSAAttention(
            dim=4,
            num_heads=1,
            indexer_cfg=dotdict(common_indexer_cfg),
        )

        state = dotdict(
            enabled=True,
            sparse=True,
            compute_loss=True,
            topk=2,
            loss_weight=1.0,
            eps=1e-6,
            tokens_per_view=10,
            num_views=2,
            patch_start_idx=1,
            patch_grid_height=3,
            patch_grid_width=3,
            image_height=6,
            image_width=6,
            extrinsics=_rectified_stereo_extrinsics(),
            intrinsics=_identity_intrinsics(),
            epipolar_selector_band_enabled=True,
            epipolar_selector_band_px=0.75,
            epipolar_selector_query_stride=1,
        )
        chunked_state = dotdict(state)
        chunked_state.epipolar_selector_query_chunk_size = 2

        projected_indexer = (
            torch.randn(1, 20, 1, 4, dtype=torch.float32),
            torch.randn(1, 20, 1, 4, dtype=torch.float32),
            torch.randn(1, 20, 1, dtype=torch.float32),
        )

        original_select = chunked_attention.indexer.select_topk_projected

        def guarded_select(q, k, w, **kwargs):
            self.assertLessEqual(q.shape[1], 2)
            return original_select(q, k, w, **kwargs)

        chunked_attention.indexer.select_topk_projected = guarded_select
        loss = chunked_attention._compute_epipolar_selector_band_loss(
            projected_indexer=projected_indexer,
            indexer_input=torch.zeros(1, 20, 4, dtype=torch.float32),
            pos=None,
            state=chunked_state,
            index_mask=None,
            view_bias_data=None,
        )

        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))

    def test_compute_epipolar_selector_band_loss_reuses_source_chunks_across_query_chunks(self):
        if DSAAttention is None:
            self.skipTest(f"DSAAttention import unavailable in local env: {_IMPORT_ERROR}")

        attention = DSAAttention(
            dim=4,
            num_heads=1,
            indexer_cfg=dotdict(
                enabled=True,
                n_heads=1,
                head_dim=4,
                use_topk_kernel=False,
                use_sparse_flash_attn=False,
                epipolar_selector_band_enabled=True,
                epipolar_selector_band_px=0.75,
                epipolar_selector_query_stride=1,
            ),
        )

        state = dotdict(
            enabled=True,
            sparse=True,
            compute_loss=True,
            topk=2,
            loss_weight=1.0,
            eps=1e-6,
            tokens_per_view=10,
            num_views=2,
            patch_start_idx=1,
            patch_grid_height=3,
            patch_grid_width=3,
            image_height=6,
            image_width=6,
            extrinsics=_rectified_stereo_extrinsics(),
            intrinsics=_identity_intrinsics(),
            epipolar_selector_band_enabled=True,
            epipolar_selector_band_px=0.75,
            epipolar_selector_query_stride=1,
            epipolar_selector_query_chunk_size=2,
            epipolar_selector_source_view_chunk_size=1,
            epipolar_selector_checkpoint_blocks=False,
        )

        total_tokens = state.tokens_per_view * state.num_views
        q = torch.randn(1, total_tokens, 1, 4, dtype=torch.float32)
        k = torch.zeros(1, total_tokens, 1, 4, dtype=torch.float32)
        w = torch.randn(1, total_tokens, 1, dtype=torch.float32)
        k[:, state.patch_start_idx : state.tokens_per_view] = 10.0
        k[:, state.tokens_per_view + state.patch_start_idx :] = 20.0
        projected_indexer = (q, k, w)

        original_select = attention.indexer.select_topk_projected
        source_chunks = {}
        call_order = []

        def tracked_select(q, k, w, **kwargs):
            key = float(k[0, 0, 0, 0].item())
            call_order.append(key)
            source_chunks.setdefault(key, []).append(k)
            return original_select(q, k, w, **kwargs)

        attention.indexer.select_topk_projected = tracked_select
        loss = attention._compute_epipolar_selector_band_loss(
            projected_indexer=projected_indexer,
            indexer_input=torch.zeros(1, total_tokens, 4, dtype=torch.float32),
            pos=None,
            state=state,
            index_mask=None,
            view_bias_data=None,
        )

        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(sorted(source_chunks), [10.0, 20.0])
        self.assertTrue(
            all(len({id(tensor) for tensor in tensors}) == 1 for tensors in source_chunks.values())
        )
        first_second_source = call_order.index(20.0)
        self.assertTrue(all(key == 10.0 for key in call_order[:first_second_source]))
        self.assertTrue(all(key == 20.0 for key in call_order[first_second_source:]))

    def test_topk_anchor_geometry_loss_uses_topk_scores_without_full_source_scores(self):
        if DSAAttention is None:
            self.skipTest(f"DSAAttention import unavailable in local env: {_IMPORT_ERROR}")

        attention = DSAAttention(
            dim=4,
            num_heads=1,
            indexer_cfg=dotdict(
                enabled=True,
                n_heads=1,
                head_dim=4,
                use_topk_kernel=False,
                use_sparse_flash_attn=False,
                epipolar_selector_band_enabled=True,
                epipolar_selector_loss_mode="topk_anchor",
                epipolar_selector_band_px=3.0,
                epipolar_selector_query_stride=1,
            ),
        )

        state = dotdict(
            enabled=True,
            sparse=True,
            compute_loss=True,
            topk=3,
            loss_weight=1.0,
            eps=1e-6,
            tokens_per_view=10,
            num_views=2,
            patch_start_idx=1,
            patch_grid_height=3,
            patch_grid_width=3,
            image_height=6,
            image_width=6,
            extrinsics=_rectified_stereo_extrinsics(),
            intrinsics=_identity_intrinsics(),
            epipolar_selector_band_enabled=True,
            epipolar_selector_loss_mode="topk_anchor",
            epipolar_selector_band_px=3.0,
            epipolar_selector_query_stride=1,
            epipolar_selector_query_chunk_size=2,
        )

        total_tokens = state.tokens_per_view * state.num_views
        projected_indexer = (
            torch.randn(1, total_tokens, 1, 4, dtype=torch.float32, requires_grad=True),
            torch.randn(1, total_tokens, 1, 4, dtype=torch.float32, requires_grad=True),
            torch.randn(1, total_tokens, 1, dtype=torch.float32, requires_grad=True),
        )
        topk_indices = torch.full((1, total_tokens, 3), state.tokens_per_view + state.patch_start_idx, dtype=torch.long)
        topk_indices[..., 1] = state.tokens_per_view + state.patch_start_idx + 3
        topk_indices[..., 2] = state.tokens_per_view + state.patch_start_idx + 6
        topk_scores = torch.randn(1, total_tokens, 3, dtype=torch.float32, requires_grad=True)

        def fail_full_score(*args, **kwargs):
            raise AssertionError("topk_anchor mode should not materialize full source scores")

        attention.indexer.select_topk_projected = fail_full_score
        loss = attention._compute_epipolar_selector_band_loss(
            projected_indexer=projected_indexer,
            indexer_input=torch.zeros(1, total_tokens, 4, dtype=torch.float32),
            pos=None,
            state=state,
            index_mask=None,
            view_bias_data=None,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
        )

        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(topk_scores.grad)

    def test_topk_anchor_geometry_loss_adds_depth_anchor_when_topk_misses_support(self):
        if DSAAttention is None:
            self.skipTest(f"DSAAttention import unavailable in local env: {_IMPORT_ERROR}")

        attention = DSAAttention(
            dim=4,
            num_heads=1,
            indexer_cfg=dotdict(
                enabled=True,
                n_heads=1,
                head_dim=4,
                use_topk_kernel=False,
                use_sparse_flash_attn=False,
                geometry_support_enabled=True,
                geometry_support_type="depth_reprojection",
                epipolar_selector_loss_mode="topk_anchor",
                geometry_support_anchor_patch_radius=0,
            ),
        )

        state = dotdict(
            enabled=True,
            sparse=True,
            compute_loss=True,
            topk=1,
            loss_weight=1.0,
            eps=1e-6,
            tokens_per_view=10,
            num_views=2,
            patch_start_idx=1,
            patch_grid_height=3,
            patch_grid_width=3,
            image_height=6,
            image_width=6,
            extrinsics=_identity_extrinsics(),
            intrinsics=_identity_intrinsics(),
            depths=torch.ones(1, 2, 6, 6, dtype=torch.float32),
            point_masks=torch.ones(1, 2, 6, 6, dtype=torch.bool),
            geometry_support_enabled=True,
            geometry_support_type="depth_reprojection",
            geometry_support_depth_radius_px=1.5,
            geometry_support_min_depth=1e-4,
            epipolar_selector_loss_mode="topk_anchor",
            epipolar_selector_query_stride=1,
            epipolar_selector_query_chunk_size=2,
            geometry_support_anchor_patch_radius=0,
        )

        total_tokens = state.tokens_per_view * state.num_views
        projected_indexer = (
            torch.randn(1, total_tokens, 1, 4, dtype=torch.float32, requires_grad=True),
            torch.randn(1, total_tokens, 1, 4, dtype=torch.float32, requires_grad=True),
            torch.randn(1, total_tokens, 1, dtype=torch.float32, requires_grad=True),
        )
        topk_indices = torch.full(
            (1, total_tokens, 1),
            state.tokens_per_view + state.patch_start_idx + 8,
            dtype=torch.long,
        )
        topk_scores = torch.full((1, total_tokens, 1), 4.0, dtype=torch.float32, requires_grad=True)

        loss = attention._compute_epipolar_selector_band_loss(
            projected_indexer=projected_indexer,
            indexer_input=torch.zeros(1, total_tokens, 4, dtype=torch.float32),
            pos=None,
            state=state,
            index_mask=None,
            view_bias_data=None,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
        )

        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(topk_scores.grad)

    def test_compute_epipolar_selector_band_loss_returns_finite_scalar(self):
        if DSAAttention is None:
            self.skipTest(f"DSAAttention import unavailable in local env: {_IMPORT_ERROR}")
        attention = DSAAttention(
            dim=4,
            num_heads=1,
            indexer_cfg=dotdict(
                enabled=True,
                n_heads=1,
                head_dim=4,
                use_topk_kernel=False,
                use_sparse_flash_attn=False,
                epipolar_selector_band_enabled=True,
                epipolar_selector_band_px=0.75,
                epipolar_selector_query_stride=1,
            ),
        )

        state = dotdict(
            enabled=True,
            sparse=True,
            compute_loss=True,
            topk=2,
            loss_weight=1.0,
            eps=1e-6,
            tokens_per_view=10,
            num_views=2,
            patch_start_idx=1,
            patch_grid_height=3,
            patch_grid_width=3,
            image_height=6,
            image_width=6,
            extrinsics=_rectified_stereo_extrinsics(),
            intrinsics=_identity_intrinsics(),
            epipolar_selector_band_enabled=True,
            epipolar_selector_band_px=0.75,
            epipolar_selector_query_stride=1,
        )

        projected_indexer = (
            torch.zeros(1, 20, 1, 4, dtype=torch.float32),
            torch.zeros(1, 20, 1, 4, dtype=torch.float32),
            torch.ones(1, 20, 1, dtype=torch.float32),
        )
        loss = attention._compute_epipolar_selector_band_loss(
            projected_indexer=projected_indexer,
            indexer_input=torch.zeros(1, 20, 4, dtype=torch.float32),
            pos=None,
            state=state,
            index_mask=None,
            view_bias_data=None,
        )

        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
