#!/usr/bin/env python3
"""Map legacy ScanNet++ Group A images back to official DSLR frames."""

from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import cv2
from PIL import Image
from scipy.optimize import linear_sum_assignment


def normalized_thumbnail(image: Image.Image, size: tuple[int, int] = (64, 48)) -> np.ndarray:
    array = np.asarray(image.convert("RGB").resize(size, Image.Resampling.LANCZOS), dtype=np.float32)
    array = (array - array.mean(axis=(0, 1), keepdims=True)) / (array.std(axis=(0, 1), keepdims=True) + 1.0e-6)
    feature = array.reshape(-1)
    norm = float(np.linalg.norm(feature))
    if norm <= 0.0:
        raise ValueError("Image has a zero-norm normalized thumbnail")
    return (feature / norm).astype(np.float32)


def feature_from_path(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return normalized_thumbnail(image)


def sift_descriptors_from_path(path: Path, max_width: int = 700, nfeatures: int = 800) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    scale = min(1.0, float(max_width) / max(float(image.shape[1]), 1.0))
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _keypoints, descriptors = cv2.SIFT_create(nfeatures=int(nfeatures)).detectAndCompute(image, None)
    return None if descriptors is None else np.asarray(descriptors, dtype=np.float32)


def sift_match_stats(
    query_descriptors: np.ndarray | None,
    candidate_descriptors: np.ndarray | None,
    ratio: float = 0.72,
) -> dict[str, float | int]:
    if query_descriptors is None or candidate_descriptors is None:
        return {"good_matches": 0, "mean_distance": float("inf")}
    query = np.asarray(query_descriptors, dtype=np.float32)
    candidate = np.asarray(candidate_descriptors, dtype=np.float32)
    if len(query) == 0 or len(candidate) < 2:
        return {"good_matches": 0, "mean_distance": float("inf")}
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(query, candidate, k=2)
    good = [first for first, second in matches if first.distance < float(ratio) * second.distance]
    return {
        "good_matches": int(len(good)),
        "mean_distance": float(np.mean([match.distance for match in good])) if good else float("inf"),
    }


def rank_candidates(
    query_feature: np.ndarray,
    candidate_features: np.ndarray,
    candidate_names: Sequence[str],
    top_k: int,
) -> list[dict[str, Any]]:
    query = np.asarray(query_feature, dtype=np.float32).reshape(-1)
    candidates = np.asarray(candidate_features, dtype=np.float32)
    if candidates.ndim != 2 or candidates.shape[1] != query.shape[0]:
        raise ValueError(f"Feature shape mismatch: query={query.shape}, candidates={candidates.shape}")
    if candidates.shape[0] != len(candidate_names):
        raise ValueError("candidate_names length does not match candidate feature rows")
    similarities = candidates @ query
    order = np.argsort(-similarities, kind="stable")[: int(top_k)]
    return [
        {
            "rank": int(rank + 1),
            "candidate": str(candidate_names[index]),
            "candidate_index": int(index),
            "similarity": float(similarities[index]),
        }
        for rank, index in enumerate(order)
    ]


def top_match_margin(ranked: Sequence[Mapping[str, Any]]) -> float:
    if len(ranked) < 2:
        return float("inf")
    return float(ranked[0]["similarity"]) - float(ranked[1]["similarity"])


def one_to_one_image_assignment(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] > scores.shape[1]:
        raise ValueError(f"Expected a query-by-candidate score matrix with Q<=C, got {scores.shape}")
    query_indices, candidate_indices = linear_sum_assignment(-scores)
    assigned = np.empty(scores.shape[0], dtype=int)
    assigned[query_indices] = candidate_indices
    return assigned


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    diff = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def pairwise_center_scale_residual(official: np.ndarray, legacy: np.ndarray) -> dict[str, float]:
    official_dist = pairwise_distances(official)
    legacy_dist = pairwise_distances(legacy)
    mask = np.triu(np.ones_like(official_dist, dtype=bool), k=1)
    x = official_dist[mask]
    y = legacy_dist[mask]
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        raise ValueError("Official camera centers are degenerate")
    scale = float(np.dot(x, y) / denom)
    residual = y - scale * x
    normalization = float(np.median(y[y > 0.0])) if np.any(y > 0.0) else 1.0
    return {
        "scale": scale,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "normalized_rmse": float(np.sqrt(np.mean(residual**2)) / max(normalization, 1.0e-12)),
        "max_abs": float(np.max(np.abs(residual))),
    }


def estimate_similarity_transform(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or source.shape[0] < 3:
        raise ValueError(f"Expected matched Nx3 arrays with N>=3, got {source.shape} and {target.shape}")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (target_centered.T @ source_centered) / float(source.shape[0])
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    correction[-1, -1] = 1.0 if np.linalg.det(u @ vt) >= 0.0 else -1.0
    rotation = u @ correction @ vt
    variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if variance <= 0.0:
        raise ValueError("Source points are degenerate")
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return {
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
    }


def apply_similarity_transform(points: np.ndarray, transform: Mapping[str, Any]) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    rotation = np.asarray(transform["rotation"], dtype=np.float64)
    translation = np.asarray(transform["translation"], dtype=np.float64)
    return float(transform["scale"]) * (points @ rotation.T) + translation


def ransac_similarity_from_top1(
    official_centers: np.ndarray,
    legacy_centers: np.ndarray,
    top1_indices: Sequence[int],
    normalized_threshold: float,
    official_rotations: np.ndarray | None = None,
    legacy_rotations: np.ndarray | None = None,
    rotation_threshold_degrees: float = 5.0,
) -> dict[str, Any]:
    official_centers = np.asarray(official_centers, dtype=np.float64)
    legacy_centers = np.asarray(legacy_centers, dtype=np.float64)
    top1_indices = [int(item) for item in top1_indices]
    if legacy_centers.shape != (len(top1_indices), 3):
        raise ValueError("legacy_centers and top1_indices length mismatch")
    normalization_values = pairwise_distances(legacy_centers)
    positive = normalization_values[normalization_values > 0.0]
    normalization = float(np.median(positive)) if positive.size else 1.0
    use_rotations = official_rotations is not None or legacy_rotations is not None
    if use_rotations:
        if official_rotations is None or legacy_rotations is None:
            raise ValueError("official_rotations and legacy_rotations must be provided together")
        official_rotations = np.asarray(official_rotations, dtype=np.float64)
        legacy_rotations = np.asarray(legacy_rotations, dtype=np.float64)
        if legacy_rotations.shape != (len(top1_indices), 3, 3):
            raise ValueError("legacy_rotations shape does not match top1_indices")
    best: dict[str, Any] | None = None
    query_indices = range(len(top1_indices))
    for sample in combinations(query_indices, 3):
        candidate_ids = [top1_indices[index] for index in sample]
        if len(set(candidate_ids)) < 3:
            continue
        try:
            transform = estimate_similarity_transform(official_centers[candidate_ids], legacy_centers[list(sample)])
        except ValueError:
            continue
        predicted = apply_similarity_transform(official_centers[top1_indices], transform)
        normalized_residual = np.linalg.norm(predicted - legacy_centers, axis=1) / max(normalization, 1.0e-12)
        inlier_mask = normalized_residual <= float(normalized_threshold)
        rotation_residual = np.zeros(len(top1_indices), dtype=np.float64)
        if use_rotations:
            global_rotation = np.asarray(transform["rotation"], dtype=np.float64)
            rotation_residual = np.asarray(
                [
                    rotation_angle_degrees(
                        legacy_rotations[index].T
                        @ (global_rotation @ official_rotations[int(candidate_index)])
                    )
                    for index, candidate_index in enumerate(top1_indices)
                ],
                dtype=np.float64,
            )
            inlier_mask &= rotation_residual <= float(rotation_threshold_degrees)
        num_inliers = int(inlier_mask.sum())
        median_inlier = float(np.median(normalized_residual[inlier_mask])) if num_inliers else float("inf")
        median_rotation = float(np.median(rotation_residual[inlier_mask])) if num_inliers else float("inf")
        score = (num_inliers, -median_inlier, -median_rotation)
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "transform": transform,
                "inlier_mask": inlier_mask,
                "normalized_residual": normalized_residual,
                "rotation_residual": rotation_residual,
            }
    if best is None or int(np.asarray(best["inlier_mask"]).sum()) < 3:
        raise RuntimeError("Could not estimate a similarity transform from top-1 image matches")
    inliers = np.asarray(best["inlier_mask"], dtype=bool)
    refined = estimate_similarity_transform(
        official_centers[np.asarray(top1_indices, dtype=int)[inliers]],
        legacy_centers[inliers],
    )
    predicted = apply_similarity_transform(official_centers[top1_indices], refined)
    normalized_residual = np.linalg.norm(predicted - legacy_centers, axis=1) / max(normalization, 1.0e-12)
    inlier_mask = normalized_residual <= float(normalized_threshold)
    rotation_residual = np.zeros(len(top1_indices), dtype=np.float64)
    if use_rotations:
        global_rotation = np.asarray(refined["rotation"], dtype=np.float64)
        rotation_residual = np.asarray(
            [
                rotation_angle_degrees(
                    legacy_rotations[index].T @ (global_rotation @ official_rotations[int(candidate_index)])
                )
                for index, candidate_index in enumerate(top1_indices)
            ],
            dtype=np.float64,
        )
        inlier_mask &= rotation_residual <= float(rotation_threshold_degrees)
    return {
        "transform": refined,
        "inlier_mask": inlier_mask,
        "normalized_residual": normalized_residual,
        "rotation_residual_degrees": rotation_residual,
        "num_inliers": int(inlier_mask.sum()),
        "normalization": normalization,
    }


def pose_guided_assignment(
    official_centers: np.ndarray,
    official_rotations: np.ndarray,
    legacy_centers: np.ndarray,
    legacy_rotations: np.ndarray,
    image_similarities: np.ndarray,
    initial_top1_indices: Sequence[int],
    ransac_threshold: float,
) -> dict[str, Any]:
    official_centers = np.asarray(official_centers, dtype=np.float64)
    official_rotations = np.asarray(official_rotations, dtype=np.float64)
    legacy_centers = np.asarray(legacy_centers, dtype=np.float64)
    legacy_rotations = np.asarray(legacy_rotations, dtype=np.float64)
    image_similarities = np.asarray(image_similarities, dtype=np.float64)
    if image_similarities.shape != (len(legacy_centers), len(official_centers)):
        raise ValueError("image_similarities shape does not match pose arrays")

    ransac = ransac_similarity_from_top1(
        official_centers=official_centers,
        legacy_centers=legacy_centers,
        top1_indices=initial_top1_indices,
        normalized_threshold=float(ransac_threshold),
        official_rotations=official_rotations,
        legacy_rotations=legacy_rotations,
        rotation_threshold_degrees=5.0,
    )
    transform = ransac["transform"]
    normalization = float(ransac["normalization"])
    assigned = np.asarray(initial_top1_indices, dtype=int)

    for _ in range(6):
        transformed_centers = apply_similarity_transform(official_centers, transform)
        center_distance = np.linalg.norm(
            legacy_centers[:, None, :] - transformed_centers[None, :, :],
            axis=-1,
        ) / max(normalization, 1.0e-12)
        global_rotation = np.asarray(transform["rotation"], dtype=np.float64)
        predicted_rotations = np.einsum("ij,mjk->mik", global_rotation, official_rotations)
        rotation_error = np.empty_like(center_distance)
        for query_index in range(len(legacy_centers)):
            for candidate_index in range(len(official_centers)):
                rotation_error[query_index, candidate_index] = rotation_angle_degrees(
                    legacy_rotations[query_index].T @ predicted_rotations[candidate_index]
                )
        cost = (
            (center_distance / 0.02) ** 2
            + (rotation_error / 5.0) ** 2
            + 0.25 * np.maximum(1.0 - image_similarities, 0.0)
        )
        query_indices, candidate_indices = linear_sum_assignment(cost)
        next_assigned = np.empty(len(legacy_centers), dtype=int)
        next_assigned[query_indices] = candidate_indices
        assigned_centers = apply_similarity_transform(official_centers[next_assigned], transform)
        assigned_center_residual = np.linalg.norm(assigned_centers - legacy_centers, axis=1) / max(
            normalization, 1.0e-12
        )
        global_rotation = np.asarray(transform["rotation"], dtype=np.float64)
        assigned_rotation_residual = np.asarray(
            [
                rotation_angle_degrees(
                    legacy_rotations[index].T @ (global_rotation @ official_rotations[int(candidate_index)])
                )
                for index, candidate_index in enumerate(next_assigned)
            ],
            dtype=np.float64,
        )
        refit_mask = (assigned_center_residual <= 0.04) & (assigned_rotation_residual <= 10.0)
        next_transform = (
            estimate_similarity_transform(official_centers[next_assigned[refit_mask]], legacy_centers[refit_mask])
            if int(refit_mask.sum()) >= 3
            else transform
        )
        stable = np.array_equal(next_assigned, assigned)
        assigned = next_assigned
        transform = next_transform
        if stable:
            break

    transformed_assigned = apply_similarity_transform(official_centers[assigned], transform)
    center_residual = np.linalg.norm(transformed_assigned - legacy_centers, axis=1) / max(normalization, 1.0e-12)
    global_rotation = np.asarray(transform["rotation"], dtype=np.float64)
    rotation_residual = np.asarray(
        [
            rotation_angle_degrees(
                legacy_rotations[index].T @ (global_rotation @ official_rotations[int(candidate_index)])
            )
            for index, candidate_index in enumerate(assigned)
        ],
        dtype=np.float64,
    )
    return {
        "assigned_indices": assigned,
        "transform": transform,
        "initial_ransac_inliers": int(ransac["num_inliers"]),
        "center_normalized_residual": center_residual,
        "center_normalized_rmse": float(np.sqrt(np.mean(center_residual**2))),
        "rotation_residual_degrees": rotation_residual,
        "rotation_mean_degrees": float(rotation_residual.mean()),
    }


def rotation_angle_degrees(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def pairwise_rotation_angle_residual(official: np.ndarray, legacy: np.ndarray) -> dict[str, float]:
    official = np.asarray(official, dtype=np.float64)
    legacy = np.asarray(legacy, dtype=np.float64)
    errors: list[float] = []
    for lhs in range(len(official)):
        for rhs in range(lhs + 1, len(official)):
            official_angle = rotation_angle_degrees(official[lhs].T @ official[rhs])
            legacy_angle = rotation_angle_degrees(legacy[lhs].T @ legacy[rhs])
            errors.append(abs(official_angle - legacy_angle))
    values = np.asarray(errors, dtype=np.float64)
    return {
        "mean_abs_degrees": float(values.mean()),
        "median_abs_degrees": float(np.median(values)),
        "max_abs_degrees": float(values.max()),
    }


def nerfstudio_c2w_to_opencv(c2w_gl: np.ndarray) -> np.ndarray:
    c2w_gl = np.asarray(c2w_gl, dtype=np.float64)
    return c2w_gl @ np.diag([1.0, -1.0, -1.0, 1.0])


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def scene_id_from_meta(meta: Mapping[str, Any]) -> str:
    if "source_scene_root" in meta:
        return Path(str(meta["source_scene_root"])).name
    return str(meta["scene_name"]).split("-")[-1]


def collect_legacy_group_a(legacy_root: Path) -> dict[str, list[dict[str, Any]]]:
    rows_by_scene: dict[str, list[dict[str, Any]]] = {}
    meta_paths = sorted(Path(legacy_root).glob("scannetpp_slight5x2_smoke/*/tuple_meta.json"))
    if len(meta_paths) != 12:
        raise ValueError(f"Expected 12 legacy tuple metadata files, found {len(meta_paths)} under {legacy_root}")
    for meta_path in meta_paths:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        scene_id = scene_id_from_meta(meta)
        tuple_root = meta_path.parent
        poses = np.loadtxt(tuple_root / "pose_90.txt", dtype=np.float64).reshape(-1, 4, 4)
        frames = list(meta["frames"])
        for out_index in range(5):
            frame = frames[out_index]
            rows_by_scene.setdefault(scene_id, []).append(
                {
                    "scene_id": scene_id,
                    "legacy_tuple_name": str(meta["tuple_name"]),
                    "legacy_tuple_root": str(tuple_root),
                    "legacy_out_index": int(out_index),
                    "legacy_frame_id": int(frame["frame_id"]),
                    "legacy_image_path": str(tuple_root / "color_90" / f"frame_{out_index:04d}.jpg"),
                    "legacy_pose": poses[out_index],
                    "legacy_anchor_a": int(meta["anchor_a"]),
                }
            )
    return rows_by_scene


def load_official_scene(official_root: Path, scene_id: str, workers: int) -> dict[str, Any]:
    scene_root = Path(official_root) / scene_id
    transforms_path = scene_root / "dslr" / "nerfstudio" / "transforms.json"
    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for frame in [*list(transforms.get("frames", [])), *list(transforms.get("test_frames", []))]:
        image_name = Path(str(frame["file_path"])).name
        if image_name in seen_names:
            continue
        seen_names.add(image_name)
        frames.append(frame)
    image_root = scene_root / "dslr" / "resized_images"
    image_paths = [image_root / Path(str(frame["file_path"])).name for frame in frames]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing official image for {scene_id}: {missing[0]}")
    def load_features(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
        return feature_from_path(path), sift_descriptors_from_path(path)

    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        loaded = list(executor.map(load_features, image_paths))
    return {
        "scene_root": scene_root,
        "frames": frames,
        "image_paths": image_paths,
        "features": np.stack([item[0] for item in loaded], axis=0),
        "sift_descriptors": [item[1] for item in loaded],
    }


def map_scene(
    scene_id: str,
    legacy_rows: Sequence[Mapping[str, Any]],
    official_scene: Mapping[str, Any],
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    unique_rows: list[Mapping[str, Any]] = []
    unique_index_by_frame: dict[int, int] = {}
    for row in legacy_rows:
        frame_id = int(row["legacy_frame_id"])
        if frame_id not in unique_index_by_frame:
            unique_index_by_frame[frame_id] = len(unique_rows)
            unique_rows.append(row)
    query_features = np.stack([feature_from_path(Path(str(row["legacy_image_path"]))) for row in unique_rows], axis=0)
    query_sift = [sift_descriptors_from_path(Path(str(row["legacy_image_path"]))) for row in unique_rows]
    candidate_features = np.asarray(official_scene["features"], dtype=np.float32)
    thumbnail_similarity = query_features @ candidate_features.T
    candidate_sift = list(official_scene["sift_descriptors"])
    sift_counts = np.zeros((len(unique_rows), len(candidate_features)), dtype=np.float64)
    sift_mean_distance = np.full_like(sift_counts, np.inf)
    for query_index, descriptors in enumerate(query_sift):
        for candidate_index, candidate_descriptors in enumerate(candidate_sift):
            stats = sift_match_stats(descriptors, candidate_descriptors)
            sift_counts[query_index, candidate_index] = int(stats["good_matches"])
            sift_mean_distance[query_index, candidate_index] = float(stats["mean_distance"])
    per_query_max = np.maximum(sift_counts.max(axis=1, keepdims=True), 1.0)
    image_similarity = 0.9 * (sift_counts / per_query_max) + 0.1 * np.clip(
        (thumbnail_similarity + 1.0) * 0.5,
        0.0,
        1.0,
    )
    initial_top1 = np.argmax(image_similarity, axis=1).astype(int).tolist()
    image_assignment = one_to_one_image_assignment(image_similarity)
    official_poses = np.stack(
        [
            nerfstudio_c2w_to_opencv(np.asarray(frame["transform_matrix"], dtype=np.float64))
            for frame in official_scene["frames"]
        ],
        axis=0,
    )
    legacy_poses_unique = np.stack(
        [np.asarray(row["legacy_pose"], dtype=np.float64) for row in unique_rows],
        axis=0,
    )
    pose_assignment = pose_guided_assignment(
        official_centers=official_poses[:, :3, 3],
        official_rotations=official_poses[:, :3, :3],
        legacy_centers=legacy_poses_unique[:, :3, 3],
        legacy_rotations=legacy_poses_unique[:, :3, :3],
        image_similarities=image_similarity,
        initial_top1_indices=initial_top1,
        ransac_threshold=0.04,
    )
    use_pose_assignment = bool(
        float(pose_assignment["center_normalized_rmse"]) < 0.01
        and float(pose_assignment["rotation_mean_degrees"]) < 1.0
    )
    assigned_unique = (
        np.asarray(pose_assignment["assigned_indices"], dtype=int)
        if use_pose_assignment
        else image_assignment
    )
    candidate_names = [path.name for path in official_scene["image_paths"]]
    mapped_rows: list[dict[str, Any]] = []

    for source in legacy_rows:
        query_index = unique_index_by_frame[int(source["legacy_frame_id"])]
        order = sorted(
            range(len(candidate_names)),
            key=lambda index: (
                -int(sift_counts[query_index, index]),
                float(sift_mean_distance[query_index, index]),
                -float(thumbnail_similarity[query_index, index]),
                candidate_names[index],
            ),
        )
        ranked = [
            {
                "rank": int(rank + 1),
                "candidate": candidate_names[index],
                "candidate_index": int(index),
                "sift_good_matches": int(sift_counts[query_index, index]),
                "sift_mean_distance": (
                    float(sift_mean_distance[query_index, index])
                    if np.isfinite(sift_mean_distance[query_index, index])
                    else None
                ),
                "thumbnail_similarity": float(thumbnail_similarity[query_index, index]),
            }
            for rank, index in enumerate(order[: int(top_k)])
        ]
        chosen_index = int(assigned_unique[query_index])
        chosen_name = candidate_names[chosen_index]
        independent_rank = int(order.index(chosen_index)) + 1
        sift_margin = int(ranked[0]["sift_good_matches"]) - int(ranked[1]["sift_good_matches"])
        frame = official_scene["frames"][chosen_index]
        mapped_rows.append(
            {
                **{key: value for key, value in source.items() if key != "legacy_pose"},
                "official_frame_index": int(chosen_index),
                "official_image_name": chosen_name,
                "official_image_path": str(official_scene["image_paths"][chosen_index]),
                "mapping_mode": "pose_guided" if use_pose_assignment else "sift",
                "similarity": float(thumbnail_similarity[query_index, chosen_index]),
                "top1_similarity": float(ranked[0]["thumbnail_similarity"]),
                "top1_margin": float(sift_margin),
                "sift_good_matches": int(sift_counts[query_index, chosen_index]),
                "sift_top1_good_matches": int(ranked[0]["sift_good_matches"]),
                "sift_top1_margin": int(sift_margin),
                "assigned_independent_rank": int(independent_rank),
                "top_candidates": ranked,
                "pose_center_normalized_residual": float(
                    np.asarray(pose_assignment["center_normalized_residual"])[query_index]
                ),
                "pose_rotation_residual_degrees": float(
                    np.asarray(pose_assignment["rotation_residual_degrees"])[query_index]
                ),
                "legacy_pose": np.asarray(source["legacy_pose"], dtype=np.float64),
                "official_pose": nerfstudio_c2w_to_opencv(np.asarray(frame["transform_matrix"], dtype=np.float64)),
            }
        )

    mapped_unique = [
        next(row for row in mapped_rows if int(row["legacy_frame_id"]) == int(source["legacy_frame_id"]))
        for source in unique_rows
    ]
    official_poses_mapped = np.stack([row["official_pose"] for row in mapped_unique], axis=0)
    legacy_poses_mapped = np.stack([row["legacy_pose"] for row in mapped_unique], axis=0)
    geometry = {
        "center_distance": pairwise_center_scale_residual(
            official_poses_mapped[:, :3, 3], legacy_poses_mapped[:, :3, 3]
        ),
        "rotation_angle": pairwise_rotation_angle_residual(
            official_poses_mapped[:, :3, :3], legacy_poses_mapped[:, :3, :3]
        ),
        "pose_guided": {
            "selected": use_pose_assignment,
            "mapping_mode": "pose_guided" if use_pose_assignment else "sift",
            "num_unique_queries": len(unique_rows),
            "initial_ransac_inliers": int(pose_assignment["initial_ransac_inliers"]),
            "center_normalized_rmse": float(pose_assignment["center_normalized_rmse"]),
            "rotation_mean_degrees": float(pose_assignment["rotation_mean_degrees"]),
        },
    }
    return mapped_rows, geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", required=True)
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-similarity", type=float, default=0.75)
    parser.add_argument("--min-margin", type=float, default=0.005)
    parser.add_argument("--min-sift-matches", type=int, default=20)
    parser.add_argument("--max-sift-rank", type=int, default=5)
    parser.add_argument("--max-center-nrmse", type=float, default=0.08)
    parser.add_argument("--max-rotation-mae-degrees", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    legacy_root = Path(args.legacy_root).expanduser().resolve()
    official_root = Path(args.official_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    rows_by_scene = collect_legacy_group_a(legacy_root)
    all_rows: list[dict[str, Any]] = []
    scene_summaries: list[dict[str, Any]] = []

    for scene_id in sorted(rows_by_scene):
        official_scene = load_official_scene(official_root, scene_id, workers=int(args.workers))
        mapped_rows, geometry = map_scene(
            scene_id,
            rows_by_scene[scene_id],
            official_scene,
            top_k=int(args.top_k),
        )
        all_rows.extend(mapped_rows)
        scene_summary = {
            "scene_id": scene_id,
            "num_queries": len(mapped_rows),
            "num_official_frames": len(official_scene["frames"]),
            "min_similarity": min(float(row["similarity"]) for row in mapped_rows),
            "min_margin": min(float(row["top1_margin"]) for row in mapped_rows),
            "max_assigned_rank": max(int(row["assigned_independent_rank"]) for row in mapped_rows),
            **geometry,
        }
        scene_summaries.append(scene_summary)
        print(
            f"[legacy-map] scene={scene_id} min_sim={scene_summary['min_similarity']:.4f} "
            f"min_margin={scene_summary['min_margin']:.4f} center_nrmse={geometry['center_distance']['normalized_rmse']:.4f} "
            f"rot_mae={geometry['rotation_angle']['mean_abs_degrees']:.3f}",
            flush=True,
        )

    serializable_rows = [
        {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in row.items()
        }
        for row in all_rows
    ]
    anchors: list[dict[str, Any]] = []
    for tuple_name in sorted({str(row["legacy_tuple_name"]) for row in all_rows}):
        tuple_rows = sorted(
            (row for row in all_rows if str(row["legacy_tuple_name"]) == tuple_name),
            key=lambda row: int(row["legacy_out_index"]),
        )
        anchors.append(
            {
                "scene_name": str(tuple_rows[0]["scene_id"]),
                "anchor_name": f"{tuple_rows[0]['scene_id']}__legacy_a{int(tuple_rows[0]['legacy_anchor_a']):06d}",
                "legacy_tuple_name": tuple_name,
                "legacy_anchor_a": int(tuple_rows[0]["legacy_anchor_a"]),
                "legacy_group_a_ids": [int(row["legacy_frame_id"]) for row in tuple_rows],
                "group_a_ids": [int(row["official_frame_index"]) for row in tuple_rows],
                "official_image_names": [str(row["official_image_name"]) for row in tuple_rows],
            }
        )

    report = {
        "legacy_root": str(legacy_root),
        "official_root": str(official_root),
        "num_scenes": len(rows_by_scene),
        "num_anchors": len(anchors),
        "num_mapped_images": len(all_rows),
        "thresholds": {
            "min_similarity": float(args.min_similarity),
            "min_margin": float(args.min_margin),
            "min_sift_matches": int(args.min_sift_matches),
            "max_sift_rank": int(args.max_sift_rank),
            "max_center_nrmse": float(args.max_center_nrmse),
            "max_rotation_mae_degrees": float(args.max_rotation_mae_degrees),
        },
        "scenes": scene_summaries,
        "rows": serializable_rows,
    }
    write_json(output_root / "mapping_report.json", report)
    write_csv(
        output_root / "mapping_rows.csv",
        [
            {key: value for key, value in row.items() if key not in {"legacy_pose", "official_pose"}}
            for row in serializable_rows
        ],
    )
    write_json(
        output_root / "fixed_group_a_manifest.json",
        {
            "protocol": "original_group_a_viewpoint_mapping_v1",
            "legacy_root": str(legacy_root),
            "dataset_root": str(official_root),
            "anchors": anchors,
        },
    )

    failures: list[str] = []
    for row in all_rows:
        if str(row["mapping_mode"]) == "pose_guided":
            if float(row["pose_center_normalized_residual"]) > 0.01:
                failures.append(
                    f"pose center residual {row['legacy_tuple_name']} view {row['legacy_out_index']}: "
                    f"{row['pose_center_normalized_residual']}"
                )
            if float(row["pose_rotation_residual_degrees"]) > 1.0:
                failures.append(
                    f"pose rotation residual {row['legacy_tuple_name']} view {row['legacy_out_index']}: "
                    f"{row['pose_rotation_residual_degrees']}"
                )
        else:
            if int(row["sift_good_matches"]) < int(args.min_sift_matches):
                failures.append(
                    f"low SIFT support {row['legacy_tuple_name']} view {row['legacy_out_index']}: "
                    f"{row['sift_good_matches']}"
                )
            if int(row["assigned_independent_rank"]) > int(args.max_sift_rank):
                failures.append(
                    f"low SIFT rank {row['legacy_tuple_name']} view {row['legacy_out_index']}: "
                    f"{row['assigned_independent_rank']}"
                )
    for scene in scene_summaries:
        if str(scene["pose_guided"]["mapping_mode"]) == "pose_guided":
            if float(scene["center_distance"]["normalized_rmse"]) > float(args.max_center_nrmse):
                failures.append(f"center residual {scene['scene_id']}: {scene['center_distance']['normalized_rmse']}")
            if float(scene["rotation_angle"]["mean_abs_degrees"]) > float(args.max_rotation_mae_degrees):
                failures.append(f"rotation residual {scene['scene_id']}: {scene['rotation_angle']['mean_abs_degrees']}")
    if failures:
        write_json(output_root / "mapping_failures.json", failures)
        raise RuntimeError(f"Mapping validation failed with {len(failures)} issues; see {output_root / 'mapping_failures.json'}")
    print(f"[legacy-map] mapped={len(all_rows)} anchors={len(anchors)} scenes={len(rows_by_scene)} output={output_root}")


if __name__ == "__main__":
    main()
