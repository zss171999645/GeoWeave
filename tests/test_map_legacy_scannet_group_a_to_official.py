#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "map_legacy_scannet_group_a_to_official.py"


def load_module():
    spec = importlib.util.spec_from_file_location("map_legacy_scannet_group_a_to_official", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalized_thumbnail_is_robust_to_affine_brightness() -> None:
    module = load_module()
    x = np.linspace(0, 180, 64 * 48 * 3, dtype=np.float32).reshape(48, 64, 3)
    brighter = np.clip(x * 1.2 + 20.0, 0, 255)

    a = module.normalized_thumbnail(Image.fromarray(x.astype(np.uint8)))
    b = module.normalized_thumbnail(Image.fromarray(brighter.astype(np.uint8)))

    assert float(np.dot(a, b)) > 0.99


def test_rank_candidates_returns_correct_match_and_margin() -> None:
    module = load_module()
    query = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    candidates = np.asarray(
        [
            [0.1, 0.9, 0.0],
            [0.95, 0.05, 0.0],
            [0.8, 0.2, 0.0],
        ],
        dtype=np.float32,
    )
    candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)

    ranked = module.rank_candidates(query, candidates, ["wrong", "correct", "runner_up"], top_k=3)

    assert ranked[0]["candidate"] == "correct"
    assert ranked[0]["similarity"] > ranked[1]["similarity"]
    assert np.isclose(module.top_match_margin(ranked), ranked[0]["similarity"] - ranked[1]["similarity"])


def test_pairwise_center_scale_residual_is_similarity_invariant() -> None:
    module = load_module()
    official = np.asarray([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.float64)
    rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    legacy = 2.5 * (official @ rotation.T) + np.asarray([4.0, -2.0, 1.0])

    result = module.pairwise_center_scale_residual(official, legacy)

    assert np.isclose(result["scale"], 2.5)
    assert result["normalized_rmse"] < 1.0e-10


def test_estimate_similarity_transform_recovers_scale_rotation_and_translation() -> None:
    module = load_module()
    source = np.asarray([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.float64)
    rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    target = 1.7 * (source @ rotation.T) + np.asarray([3.0, -4.0, 2.0])

    transform = module.estimate_similarity_transform(source, target)
    aligned = module.apply_similarity_transform(source, transform)

    assert np.isclose(transform["scale"], 1.7)
    np.testing.assert_allclose(transform["rotation"], rotation, atol=1.0e-10)
    np.testing.assert_allclose(aligned, target, atol=1.0e-10)


def test_ransac_top1_similarity_ignores_one_wrong_correspondence() -> None:
    module = load_module()
    official = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3], [5, 5, 5], [9, -4, 2]],
        dtype=np.float64,
    )
    rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    legacy = 2.0 * (official[:5] @ rotation.T) + np.asarray([4.0, 2.0, -1.0])
    top1_indices = [0, 1, 2, 3, 5]
    official_rotations = np.repeat(np.eye(3, dtype=np.float64)[None], len(official), axis=0)
    official_rotations[5] = np.diag([-1.0, -1.0, 1.0])
    legacy_rotations = np.repeat(rotation[None], len(legacy), axis=0)

    result = module.ransac_similarity_from_top1(
        official_centers=official,
        legacy_centers=legacy,
        top1_indices=top1_indices,
        normalized_threshold=0.02,
        official_rotations=official_rotations,
        legacy_rotations=legacy_rotations,
        rotation_threshold_degrees=2.0,
    )

    assert result["num_inliers"] == 4
    np.testing.assert_array_equal(result["inlier_mask"], [True, True, True, True, False])


def test_pose_guided_assignment_recovers_image_ambiguous_frame() -> None:
    module = load_module()
    official_centers = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 2, 0], [3, 3, 0], [0, 0, 4], [5, 1, 2]],
        dtype=np.float64,
    )
    official_rotations = np.repeat(np.eye(3, dtype=np.float64)[None], len(official_centers), axis=0)
    true_indices = np.asarray([0, 2, 4, 5], dtype=int)
    global_rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    legacy_centers = 1.5 * (official_centers[true_indices] @ global_rotation.T) + np.asarray([2.0, -3.0, 1.0])
    legacy_rotations = np.repeat(global_rotation[None], len(true_indices), axis=0)
    similarities = np.zeros((len(true_indices), len(official_centers)), dtype=np.float64)
    similarities[0, 0] = 1.0
    similarities[1, 2] = 1.0
    similarities[2, 3] = 0.95
    similarities[2, 4] = 0.90
    similarities[3, 5] = 1.0
    initial_top1 = np.argmax(similarities, axis=1).tolist()

    result = module.pose_guided_assignment(
        official_centers=official_centers,
        official_rotations=official_rotations,
        legacy_centers=legacy_centers,
        legacy_rotations=legacy_rotations,
        image_similarities=similarities,
        initial_top1_indices=initial_top1,
        ransac_threshold=0.03,
    )

    np.testing.assert_array_equal(result["assigned_indices"], true_indices)
    assert result["center_normalized_rmse"] < 1.0e-10
    assert result["rotation_mean_degrees"] < 1.0e-8


def test_sift_match_stats_separates_shared_local_descriptors() -> None:
    module = load_module()
    query = np.asarray(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        dtype=np.float32,
    )
    correct = np.asarray(
        [[0.01, 0.0], [1.01, 1.0], [2.01, 2.0], [3.01, 3.0], [10.0, 10.0]],
        dtype=np.float32,
    )
    wrong = np.asarray(
        [[5.0, 0.0], [0.0, 5.0], [8.0, 1.0], [1.0, 8.0], [10.0, 10.0]],
        dtype=np.float32,
    )

    correct_stats = module.sift_match_stats(query, correct, ratio=0.75)
    wrong_stats = module.sift_match_stats(query, wrong, ratio=0.75)

    assert correct_stats["good_matches"] == 4
    assert correct_stats["good_matches"] > wrong_stats["good_matches"]


def test_one_to_one_image_assignment_resolves_top1_collision() -> None:
    module = load_module()
    scores = np.asarray(
        [
            [0.90, 0.89, 0.10],
            [0.95, 0.80, 0.20],
        ],
        dtype=np.float64,
    )

    assigned = module.one_to_one_image_assignment(scores)

    np.testing.assert_array_equal(assigned, [1, 0])


if __name__ == "__main__":
    test_normalized_thumbnail_is_robust_to_affine_brightness()
    test_rank_candidates_returns_correct_match_and_margin()
    test_pairwise_center_scale_residual_is_similarity_invariant()
    test_estimate_similarity_transform_recovers_scale_rotation_and_translation()
    test_ransac_top1_similarity_ignores_one_wrong_correspondence()
    test_pose_guided_assignment_recovers_image_ambiguous_frame()
    test_sift_match_stats_separates_shared_local_descriptors()
    test_one_to_one_image_assignment_resolves_top1_collision()
    print("ok")
