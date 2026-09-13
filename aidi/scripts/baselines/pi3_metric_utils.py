from __future__ import annotations

import numpy as np

try:
    from scipy.spatial import cKDTree as KDTree
except ModuleNotFoundError:  # pragma: no cover - lightweight local env fallback
    KDTree = None


def _query_nearest_neighbors(reference: np.ndarray, query: np.ndarray):
    if KDTree is not None:
        tree = KDTree(reference)
        return tree.query(query, workers=-1)

    try:  # pragma: no cover - exercised in dev env if scipy is absent but open3d exists
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(reference, dtype=np.float64))
        tree = o3d.geometry.KDTreeFlann(pcd)
        distances = np.empty((len(query),), dtype=np.float64)
        indices = np.empty((len(query),), dtype=np.int64)
        for i, point in enumerate(np.asarray(query, dtype=np.float64)):
            _, idx, dist2 = tree.search_knn_vector_3d(point, 1)
            indices[i] = int(idx[0])
            distances[i] = float(np.sqrt(dist2[0]))
        return distances, indices
    except ModuleNotFoundError:
        pass

    reference = np.asarray(reference, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    distances = np.empty((len(query),), dtype=np.float64)
    indices = np.empty((len(query),), dtype=np.int64)
    for i, point in enumerate(query):
        diff = reference - point[None]
        dist2 = np.sum(diff * diff, axis=-1)
        idx = int(np.argmin(dist2))
        indices[i] = idx
        distances[i] = float(np.sqrt(dist2[idx]))
    return distances, indices


def umeyama(X: np.ndarray, Y: np.ndarray):
    """Estimate Sim(3) that maps X to Y.

    Mirrors the PI3 evaluation branch implementation in `mv_recon/utils.py`.
    Inputs follow shape (m, n), where m is point dimension and n is point count.
    """

    mu_x = X.mean(axis=1).reshape(-1, 1)
    mu_y = Y.mean(axis=1).reshape(-1, 1)
    var_x = np.square(X - mu_x).sum(axis=0).mean()
    cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
    U, D, VH = np.linalg.svd(cov_xy)
    S = np.eye(X.shape[0], dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[-1, -1] = -1
    c = np.trace(np.diag(D) @ S) / var_x
    R = U @ S @ VH
    t = mu_y - c * R @ mu_x
    return c, R, t


def completion_ratio(gt_points: np.ndarray, rec_points: np.ndarray, dist_th: float = 0.05) -> float:
    distances, _ = _query_nearest_neighbors(rec_points, gt_points)
    return float(np.mean((distances < dist_th).astype(np.float32)))


def accuracy(
    gt_points: np.ndarray,
    rec_points: np.ndarray,
    gt_normals: np.ndarray | None = None,
    rec_normals: np.ndarray | None = None,
):
    distances, idx = _query_nearest_neighbors(gt_points, rec_points)
    acc = float(np.mean(distances))
    acc_median = float(np.median(distances))
    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals[idx] * rec_normals, axis=-1)
        normal_dot = np.abs(normal_dot)
        return acc, acc_median, float(np.mean(normal_dot)), float(np.median(normal_dot))
    return acc, acc_median


def completion(
    gt_points: np.ndarray,
    rec_points: np.ndarray,
    gt_normals: np.ndarray | None = None,
    rec_normals: np.ndarray | None = None,
):
    distances, idx = _query_nearest_neighbors(rec_points, gt_points)
    comp = float(np.mean(distances))
    comp_median = float(np.median(distances))
    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals * rec_normals[idx], axis=-1)
        normal_dot = np.abs(normal_dot)
        return comp, comp_median, float(np.mean(normal_dot)), float(np.median(normal_dot))
    return comp, comp_median


def compute_iou(pred_vox, target_vox) -> float:
    pred_indices = [voxel.grid_index for voxel in pred_vox.get_voxels()]
    target_indices = [voxel.grid_index for voxel in target_vox.get_voxels()]
    pred_filled = set(tuple(np.round(x, 4)) for x in pred_indices)
    target_filled = set(tuple(np.round(x, 4)) for x in target_indices)
    intersection = pred_filled & target_filled
    union = pred_filled | target_filled
    return float(len(intersection) / len(union))
