import numpy as np
import open3d as o3d
import os
from scipy.linalg import svd

class PointCloudAligner:
    def __init__(self, target_points, source_points):
        self.target_points = target_points
        self.source_points = source_points
        assert self.source_points.shape == self.target_points.shape, "Point clouds must be point-to-point aligned."

    def align_se3(self):
        src = self.source_points['xyz']
        tgt = self.target_points['xyz']

        centroid_src = np.mean(src, axis=0)
        centroid_tgt = np.mean(tgt, axis=0)
        src_centered = src - centroid_src
        tgt_centered = tgt - centroid_tgt

        H = src_centered.T @ tgt_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        t = centroid_tgt - R @ centroid_src

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def align_sim3(self):
        src = self.source_points[:, :3]
        tgt = self.target_points[:, :3]

        centroid_src = np.mean(src, axis=0)
        centroid_tgt = np.mean(tgt, axis=0)
        src_centered = src - centroid_src
        tgt_centered = tgt - centroid_tgt

        H = src_centered.T @ tgt_centered / src_centered.shape[0]
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        var_src = np.var(src_centered, axis=0).sum()
        scale = np.trace(np.diag(S)) / var_src
        print(scale)

        t = centroid_tgt - scale * R @ centroid_src

        T = np.eye(4)
        T[:3, :3] = scale * R
        T[:3, 3] = t
        return T

    def apply_transform_and_save(self, T, src, output_path):
        src_h = np.hstack([src[:, :3], np.ones((src.shape[0], 1))])
        aligned = (T @ src_h.T).T
        aligned /= aligned[:, 3:4]
        aligned_xyz = aligned[:, :3]
        self._save_ply(np.hstack([aligned_xyz, src[:, 3:]]), output_path)
        return aligned_xyz
