from __future__ import annotations
import os
import cv2
import json
import random
import torch
import numpy as np
from enum import Enum, auto
import torch.nn.functional as F
from typing import Union, List, Tuple

from scipy import interpolate
from scipy.spatial.transform import Rotation

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.chunk_utils import multi_gather
from easyvolcap.utils.data_utils import get_rays, get_near_far
from easyvolcap.utils.math_utils import affine_inverse, affine_padding


def compute_camera_similarity(tar_c2ws: torch.Tensor, src_c2ws: torch.Tensor):
    # Get the camera centers
    centers_target = tar_c2ws[..., :3, 3]  # Nt, L, 3
    centers_source = src_c2ws[..., :3, 3]  # Ns, L, 3

    # Using distance between centers for camera selection
    sims: torch.Tensor = 1 / (centers_source[None] - centers_target[:, None]).norm(dim=-1)  # Nt, Ns, L,

    # Source view index and there similarity
    src_sims, src_inds = sims.sort(dim=1, descending=True)  # similarity to source views # Target, Source, Latent
    return src_sims, src_inds  # Nt, Ns, L


def compute_camera_dist_rot_similarity(
    tar_c2ws: torch.Tensor,
    src_c2ws: torch.Tensor,
    weight: float = 1.0,
    normalize: bool = True,
):
    # Add `.clone()` to avoid in-place operations
    tar_c2ws = tar_c2ws.clone()
    src_c2ws = src_c2ws.clone()

    # Normalization if needed
    if normalize:
        # Compute the average translation
        t = torch.cat([tar_c2ws[..., :3, 3], src_c2ws[..., :3, 3]], dim=0)  # (Nt + Ns, L, 3)
        s = torch.mean(torch.norm(t, dim=-1))  # scalar
        # Normalize the translation
        tar_c2ws[..., :3, 3] = tar_c2ws[..., :3, 3] / s
        src_c2ws[..., :3, 3] = src_c2ws[..., :3, 3] / s

    # Get the camera centers
    centers_target = tar_c2ws[..., :3, 3]  # Nt, L, 3
    centers_source = src_c2ws[..., :3, 3]  # Ns, L, 3
    # Compute the distance between the centers
    dis_sims = 1 / (centers_source[None] - centers_target[:, None]).norm(dim=-1)  # Nt, Ns, L,

    # Get the camera rotations
    rotates_target = tar_c2ws[..., :3, :3]  # Nt, L, 3, 3
    rotates_source = src_c2ws[..., :3, :3]  # Ns, L, 3, 3
    # Compute the dot product between the rotations
    rot_prod = (rotates_source[None] @ rotates_target[:, None].transpose(-2, -1))  # Nt, Ns, L, 3, 3
    # Compute the trace for each dot product
    rot_trac = rot_prod[..., 0, 0] + rot_prod[..., 1, 1] + rot_prod[..., 2, 2]  # Nt, Ns, L,
    rot_sims = ((rot_trac - 1.) / 2.).clamp(-1., 1.)  # Nt, Ns, L,
    rot_sims = torch.acos(rot_sims) * (180 / np.pi) / 180  # Nt, Ns, L,

    # Combine the distance and rotation similarity
    sims = rot_sims + dis_sims * weight  # Nt, Ns, L,

    # Source view index and there similarity
    src_sims, src_inds = sims.sort(dim=1, descending=True)  # similarity to source views # Target, Source, Latent
    return src_sims, src_inds  # Nt, Ns, L


def compute_camera_sequential_similarity(tar_c2ws: torch.Tensor, src_c2ws: torch.Tensor):
    # This function assumes that target views and source views are the same
    assert (tar_c2ws[..., :3, :] - src_c2ws[..., :3, :]).abs().max() < 1e-6, 'Target and source views must be the same'

    # Get the number of views, and latent dimension
    N = tar_c2ws.shape[0]  # target frame number
    L = tar_c2ws.shape[1]  # latent dimension
    device = tar_c2ws.device

    src_inds = []
    for i in range(N):
        indices = [i]  # current frame
        indices.extend(range(i + 1, N))  # next frames
        indices.extend(range(i - 1, -1, -1))  # previous frames (reversed)
        src_inds.append(indices)
    src_inds = torch.tensor(src_inds, dtype=torch.long, device=device)  # (N, N)

    # Generate pseudo-similarity: the current frame is the maximum, and the subsequent frames decrease
    src_sims = torch.zeros((N, N), device=device)
    for i in range(N):
        src_sims[i] = torch.tensor([1e10] + [1.0 / (j + 1) for j in range(N - 1)], device=device)

    # Expand the similarity and indices to the latent dimension
    src_inds = src_inds.unsqueeze(-1).expand(-1, -1, L)
    src_sims = src_sims.unsqueeze(-1).expand(-1, -1, L)

    return src_sims, src_inds


def compute_camera_zigzag_similarity(tar_c2ws: torch.Tensor, src_c2ws: torch.Tensor):
    # Get the camera centers
    centers_target = tar_c2ws[..., :3, 3]  # (Vt, F, 3)
    centers_source = src_c2ws[..., :3, 3]  # (Vs, F, 3)

    # Compute the distance between the centers
    sims: torch.Tensor = 1 / (centers_source[None] - centers_target[:, None]).norm(dim=-1)  # (Vt, Vs, F)
    # Source view index and there similarity
    src_sims, src_inds = sims.sort(dim=1, descending=True)  # (Vt, Vs, F), (Vt, Vs, F)

    # Select the closest source view as the reference view for each target view
    ref_view = multi_gather(centers_source.permute(1, 0, 2), src_inds.permute(2, 0, 1)[..., 0]).permute(1, 0, 2)  # (Vt, F, 3)

    # Compute the cross product between the reference view and target view, and the cross product between the source views and the target view
    ref_cross = torch.cross(ref_view, centers_target, dim=-1)  # (Vt, F, 3)
    src_cross = torch.cross(centers_source[None], centers_target[:, None], dim=-1)  # (Vt, Vs, F, 3)

    # Compute the inner product between the cross products to determine the zigzag placing
    zigzag = (ref_cross[:, None] * src_cross).sum(dim=-1)  # (Vt, Vs, F)

    zigzag_src_sims, zigzag_src_inds = src_sims.clone(), src_inds.clone()
    # Re-indexing the similarity and indices
    for v in range(len(zigzag)):
        # Get the sorted zig and zag similarity and indices respectively
        zig_msk = torch.sum(torch.eq(torch.arange(len(centers_source))[zigzag[v, :, 0] > 0][:, None], src_inds[v, :, 0]), dim=0).bool()
        zig_src_sims, zig_src_inds = src_sims[v][zig_msk], src_inds[v][zig_msk]  # (L, F), (L, F)
        zag_msk = torch.sum(torch.eq(torch.arange(len(centers_source))[zigzag[v, :, 0] < 0][:, None], src_inds[v, :, 0]), dim=0).bool()
        zag_src_sims, zag_src_inds = src_sims[v][zag_msk], src_inds[v][zag_msk]  # (R, F), (R, F)

        # Concatenate the zig and zag similarity and indices in order zig-zag-zig-zag-...
        size = min(len(zig_src_sims), len(zag_src_sims))
        zigzag_src_sims[v, 0:size * 2:2], zigzag_src_sims[v, 1:size * 2:2] = zig_src_sims[:size], zag_src_sims[:size]  # (S*2, F), (S*2, F)
        zigzag_src_inds[v, 0:size * 2:2], zigzag_src_inds[v, 1:size * 2:2] = zig_src_inds[:size], zag_src_inds[:size]  # (S*2, F), (S*2, F)

        # Concatenate the remaining similarity and indices
        if len(zig_src_sims) > len(zag_src_sims): zigzag_src_sims[v, size * 2:], zigzag_src_inds[v, size * 2:] = zig_src_sims[size:], zig_src_inds[size:]
        else: zigzag_src_sims[v, size * 2:], zigzag_src_inds[v, size * 2:] = zag_src_sims[size:], zag_src_inds[size:]

    # Return the zigzag similarity and indices
    return zigzag_src_sims, zigzag_src_inds


class Sourcing(Enum):
    # Type of source indexing
    DISTANCE = auto()  # the default source indexing
    DISTANCEROT = auto()  # consider both distance and rotation
    ZIGZAG = auto()  # will index the source view in zigzag order
    SEQUENTIAL = auto()  # will index the source view in sequential order
    RANDOM = auto()  # will index the source view in random order
    MULTIVIEWSEQ = auto()  # will index the source view in a multi-view sequential order
    MULTIVIEWSEQV2 = auto()  # will index the source view in a multi-view sequential order, every other views have a ref 0 view, eg. [0, 1, 2, 0, 3, 4, 0, 5, 2, 0, ...]
    MULTIVIEWSEQLC = auto()  # will index the source view in a multi-view sequential order, plus loop closure view

class Interpolation(Enum):
    # Type of interpolation to use
    CUBIC = auto()  # the default interpolation
    ORBIT = auto()  # will find a full circle around the cameras, the default orbit path
    SPIRAL = auto()  # will perform spiral motion around the cameras
    SECTOR = auto()  # will find a circular sector around the cameras
    NONE = auto()  # used as is


def normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-13)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec0_avg = up
    vec1 = normalize(np.cross(vec2, vec0_avg))
    vec0 = normalize(np.cross(vec1, vec2))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


# From https://github.com/NVLabs/instant-ngp


def compute_center_of_attention(c2ws: np.ndarray):
    # TODO: Should vectorize this to make it faster, this is not very tom94
    totw = 0.0
    totp = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    for mf in c2ws:
        for mg in c2ws:
            p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
            if w > 0.01:
                totp += p * w
                totw += w
    totp /= totw
    return totp[..., None]  # 3, 1


def closest_point_2_lines(oa: np.ndarray, da: np.ndarray, ob: np.ndarray, db: np.ndarray):
    # Returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    return (oa + ta * da + ob + tb * db) * 0.5, denom


# From: https://github.com/sarafridov/K-Planes/blob/main/plenoxels/datasets/ray_utils.py


def average_c2ws(c2ws: np.ndarray, align_cameras: bool = True, look_at_center: bool = True) -> np.ndarray:
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """

    if align_cameras:
        # 1. Compute the center
        center = compute_center_of_attention(c2ws)[..., 0]  # (3)
        # 2. Compute the z axis
        z = -normalize(c2ws[..., 1].mean(0))  # (3) # FIXME: WHY?
        # 3. Compute axis y' (no need to normalize as it's not the final output)
        y_ = c2ws[..., 2].mean(0)  # (3)
        # 4. Compute the x axis
        x = -normalize(np.cross(z, y_))  # (3)
        # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
        y = -np.cross(x, z)  # (3)

    else:
        # 1. Compute the center
        center = c2ws[..., 3].mean(0)  # (3)
        # 2. Compute the z axis
        if look_at_center:
            look = compute_center_of_attention(c2ws)[..., 0]  # (3)
            z = normalize(look - center)
        else:
            z = normalize(c2ws[..., 2].mean(0))  # (3)
        # 3. Compute axis y' (no need to normalize as it's not the final output)
        y_ = c2ws[..., 1].mean(0)  # (3)
        # 4. Compute the x axis
        x = -normalize(np.cross(z, y_))  # (3)
        # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
        y = -np.cross(x, z)  # (3)

    c2w_avg = np.stack([x, y, z, center], 1)  # (3, 4)
    return c2w_avg


def align_c2ws(c2ws: np.ndarray, c2w_avg: Union[np.ndarray, None] = None) -> np.ndarray:
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    c2w_avg = c2w_avg if c2w_avg is not None else average_c2ws(c2ws)  # (3, 4)
    c2w_avg_homo = np.eye(4, dtype=c2ws.dtype)
    c2w_avg_homo[:3] = c2w_avg  # convert to homogeneous coordinate for faster computation

    last_row = np.tile(np.asarray([0, 0, 0, 1], dtype=np.float32), (len(c2ws), 1, 1))  # (N_images, 1, 4)
    c2ws_homo = np.concatenate([c2ws, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    c2ws_centered = np.linalg.inv(c2w_avg_homo) @ c2ws_homo  # (N_images, 4, 4)
    c2ws_centered = c2ws_centered[:, :3]  # (N_images, 3, 4)

    return c2ws_centered


def average_w2cs(w2cs: np.ndarray) -> np.ndarray:
    # Transform the world2camera extrinsic from matrix representation to vector representation
    rvecs = np.array([cv2.Rodrigues(w2c[:3, :3])[0] for w2c in w2cs], dtype=np.float32)  # (V, 3, 1)
    tvecs = w2cs[:, :3, 3:]  # (V, 3, 1)

    # Compute the average view direction and center in vector mode
    rvec_avg = rvecs.mean(axis=0)  # (3, 1)
    tvec_avg = tvecs.mean(axis=0)  # (3, 1)

    # Back to matrix representation
    w2c_avg = np.concatenate([cv2.Rodrigues(rvec_avg)[0], tvec_avg], axis=1)
    return w2c_avg


def gen_cam_interp_func_bspline(c2ws: np.ndarray, smoothing_term=1.0, per: int = 0):
    center_t, center_u, front_t, front_u, up_t, up_u = gen_cam_interp_params_bspline(c2ws, smoothing_term, per)

    def f(us: np.ndarray):
        if isinstance(us, int) or isinstance(us, float): us = np.asarray([us])
        if isinstance(us, list): us = np.asarray(us)

        # The interpolation t
        center = np.asarray(interpolate.splev(us, center_t)).T.astype(c2ws.dtype)
        v_front = np.asarray(interpolate.splev(us, front_t)).T.astype(c2ws.dtype)
        v_up = np.asarray(interpolate.splev(us, up_t)).T.astype(c2ws.dtype)

        # Normalization
        v_front = normalize(v_front)
        v_up = normalize(v_up)
        v_right = normalize(np.cross(v_front, v_up))
        v_down = np.cross(v_front, v_right)

        # Combination
        render_c2ws = np.stack([v_right, v_down, v_front, center], axis=-1)
        return render_c2ws
    return f


def gen_cam_interp_params_bspline(c2ws: np.ndarray, smoothing_term=1.0, per: int = 0):
    """Return B-spline interpolation parameters for the camera # MARK: Quite easy to error out
    Actually this should be implemented as a general interpolation function
    Reference get_camera_up_front_center for the definition of worldup, front, center
    Args:
        smoothing_term(float): degree of smoothing to apply on the camera path interpolation
    """
    centers = c2ws[..., :3, 3]
    fronts = c2ws[..., :3, 2]
    ups = -c2ws[..., :3, 1]

    center_t, center_u = interpolate.splprep(centers.T, s=smoothing_term, per=per)  # array of u corresponds to parameters of specific camera points
    front_t, front_u = interpolate.splprep(fronts.T, s=smoothing_term, per=per)  # array of u corresponds to parameters of specific camera points
    up_t, up_u = interpolate.splprep(ups.T, s=smoothing_term, per=per)  # array of u corresponds to parameters of specific camera points
    return center_t, center_u, front_t, front_u, up_t, up_u


def cubic_spline(us: np.ndarray, N: int):
    if isinstance(us, int) or isinstance(us, float): us = np.asarray([us])
    if isinstance(us, list): us = np.asarray(us)

    # Preparation
    t = (N - 1) * us  # expanded to the length of the sequence
    i0 = np.floor(t).astype(np.int32) - 1
    i0 = np.where(us != 1.0, i0, i0 - 1)  # remove end point (nans for 1s)
    i1 = i0 + 1
    i2 = i0 + 2
    i3 = i0 + 3
    i0, i1, i2, i3 = np.clip(i0, 0, N - 1), np.clip(i1, 0, N - 1), np.clip(i2, 0, N - 1), np.clip(i3, 0, N - 1)
    t0, t1, t2, t3 = i0 / (N - 1), i1 / (N - 1), i2 / (N - 1), i3 / (N - 1)
    t = (t - i1)  # normalize to the start?
    t = t.astype(np.float32)  # avoid fp64 problems

    # Compute coeffs
    tt = t * t
    ttt = tt * t
    a = (1 - t) * (1 - t) * (1 - t) * (1. / 6.)
    b = (3. * ttt - 6. * tt + 4.) * (1. / 6.)
    c = (-3. * ttt + 3. * tt + 3. * t + 1.) * (1. / 6.)
    d = ttt * (1. / 6.)

    t0, t1, t2, t3 = t0.astype(np.float32), t1.astype(np.float32), t2.astype(np.float32), t3.astype(np.float32)
    a, b, c, d = a.astype(np.float32), b.astype(np.float32), c.astype(np.float32), d.astype(np.float32)

    return t, (i0, i1, i2, i3), (t0, t1, t2, t3), (a, b, c, d)


class InterpolatingExtrinsics:
    def __init__(self, c2w: np.ndarray) -> None:
        self.Q = Rotation.from_matrix(c2w[..., :3, :3]).as_quat()
        self.T = c2w[..., :3, 3]

    def __add__(lhs, rhs: InterpolatingExtrinsics):  # FIXME: Dangerous
        Ql, Qr = lhs.Q, rhs.Q
        Qr = np.where((Ql * Qr).sum(axis=-1, keepdims=True) < 0, -Qr, Qr)
        lhs.Q = Ql + Qr
        lhs.T = lhs.T + rhs.T
        return lhs

    def __radd__(rhs, lhs: InterpolatingExtrinsics):
        return rhs.__add__(lhs)

    def __mul__(lhs, rhs: np.ndarray):
        lhs.Q = rhs[..., None] * lhs.Q
        lhs.T = rhs[..., None] * lhs.T
        return lhs  # inplace modification

    def __rmul__(rhs, lhs: np.ndarray):
        return rhs.__mul__(lhs)

    def numpy(self):
        return np.concatenate([Rotation.from_quat(self.Q).as_matrix(), self.T[..., None]], axis=-1).astype(np.float32)


def gen_cubic_spline_interp_func(c2ws: np.ndarray, smoothing_term=10.0, *args, **kwargs):
    # Split interpolation
    N = len(c2ws)
    assert N > 3, 'Cubic Spline interpolation requires at least four inputs'
    if smoothing_term == 0:
        low = -2  # when we view index as from 0 to n, should remove first two segments
        high = N - 1 + 4 - 2  # should remove last one segment, please just work...
        c2ws = np.concatenate([c2ws[-2:], c2ws, c2ws[:2]])

    def lf(us: np.ndarray):
        N = len(c2ws)  # should this be recomputed?
        t, (i0, i1, i2, i3), (t0, t1, t2, t3), (a, b, c, d) = cubic_spline(us, N)

        # Extra inter target
        c0, c1, c2, c3 = InterpolatingExtrinsics(c2ws[i0]), InterpolatingExtrinsics(c2ws[i1]), InterpolatingExtrinsics(c2ws[i2]), InterpolatingExtrinsics(c2ws[i3])
        c = c0 * a + c1 * b + c2 * c + c3 * d  # to utilize operator overloading
        c = c.numpy()  # from InterpExt to numpy
        if isinstance(us, int) or isinstance(us, float): c = c[0]  # remove extra dim
        return c

    if smoothing_term == 0:
        def pf(us): return lf((us * N - low) / (high - low))  # periodic function will call the linear function
        f = pf  # periodic function
    else:
        f = lf  # linear function
    return f


def gen_linear_interp_func(lins: np.ndarray, smoothing_term=10.0):  # smoothing_term <= will loop the interpolation
    if smoothing_term == 0:
        n = len(lins)
        low = -2  # when we view index as from 0 to n, should remove first two segments
        high = n - 1 + 4 - 2  # should remove last one segment, please just work...
        lins = np.concatenate([lins[-2:], lins, lins[:2]])

    lf = interpolate.interp1d(np.linspace(0, 1, len(lins), dtype=np.float32), lins, axis=-2)  # repeat

    if smoothing_term == 0:
        def pf(us): return lf((us * n - low) / (high - low))  # periodic function will call the linear function
        f = pf  # periodic function
    else:
        f = lf  # linear function
    return f


def interpolate_camera_path(c2ws: np.ndarray, n_render_views=50, smoothing_term=10.0, **kwargs):
    # Store interpolation parameters
    f = gen_cubic_spline_interp_func(c2ws, smoothing_term)

    # The interpolation t
    us = np.linspace(0, 1, n_render_views, dtype=c2ws.dtype)
    return f(us)


def interpolate_camera_lins(lins: np.ndarray, n_render_views=50, smoothing_term=10.0, **kwargs):
    # Store interpolation parameters
    f = gen_linear_interp_func(lins, smoothing_term)

    # The interpolation t
    us = np.linspace(0, 1, n_render_views, dtype=lins.dtype)
    return f(us)


def generate_spiral_path(c2ws: np.ndarray,
                         n_render_views=300,
                         n_rots=2,
                         zrate=0.5,
                         percentile=70,

                         focal_offset=0.0,
                         radius_ratio=1.0,
                         xyz_ratio=[1.0, 1.0, 0.25],
                         xyz_offset=[0.0, 0.0, 0.0],
                         **kwargs) -> np.ndarray:
    """Calculates a forward facing spiral path for rendering.
    From https://github.com/google-research/google-research/blob/342bfc150ef1155c5254c1e6bd0c912893273e8d/regnerf/internal/datasets.py
    and https://github.com/apchenstu/TensoRF/blob/main/dataLoader/llff.py
    """
    # Prepare input data
    c2ws = c2ws[..., :3, :4]

    # Center pose
    c2w_avg = average_c2ws(c2ws, align_cameras=False, look_at_center=True)  # [3, 4]

    # Get average pose
    v_up = -normalize(c2ws[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset as a weighted average
    # of near and far bounds in disparity space.
    focal = focal_offset + np.linalg.norm(compute_center_of_attention(c2ws)[..., 0] - c2w_avg[..., 3])  # (3)

    # Get radii for spiral path using 70th percentile of camera origins.
    radii = np.percentile(np.abs(c2ws[:, :3, 3] - c2w_avg[..., 3]), percentile, 0) * radius_ratio  # N, 3
    radii = np.concatenate([xyz_ratio * radii, [1.]])  # 4,

    # Generate c2ws for spiral path.
    render_c2ws = []
    for theta in np.linspace(0., 2. * np.pi * n_rots, n_render_views, endpoint=False):
        t = radii * [np.cos(theta), np.sin(theta), np.sin(theta * zrate), 1.] + \
            np.concatenate([xyz_offset, [0.]])

        center = c2w_avg @ t
        center = center.astype(c2ws.dtype)
        lookat = c2w_avg @ np.array([0, 0, focal, 1.0], dtype=c2ws.dtype)

        v_front = -normalize(center - lookat)
        v_right = normalize(np.cross(v_front, v_up))
        v_down = np.cross(v_front, v_right)
        c2w = np.stack([v_right, v_down, v_front, center], axis=-1)  # 3, 4
        render_c2ws.append(c2w)

    render_c2ws = np.stack(render_c2ws, axis=0)  # N, 3, 4
    return render_c2ws


def generate_hemispherical_orbit(c2ws: np.ndarray,
                                 n_render_views=50,
                                 orbit_height=0.,
                                 orbit_radius=-1,
                                 radius_ratio=1.0,
                                 **kwargs):
    """Calculates a render path which orbits around the z-axis.
    Based on https://github.com/google-research/google-research/blob/342bfc150ef1155c5254c1e6bd0c912893273e8d/regnerf/internal/datasets.py
    TODO: Implement this for non-centered camera paths
    """
    # Center pose
    c2w_avg = average_c2ws(c2ws)  # [3, 4]

    # Find the origin and radius for the orbit
    origins = c2ws[:, :3, 3]
    radius = (np.sqrt(np.mean(np.sum(origins ** 2, axis=-1))) * radius_ratio) if orbit_radius <= 0 else orbit_radius

    # Get average pose
    v_up = -normalize(c2ws[:, :3, 1].sum(0))

    # Assume that z-axis points up towards approximate camera hemispherical
    sin_phi = np.mean(origins[:, 2], axis=0) / radius
    cos_phi = np.sqrt(1 - sin_phi ** 2)
    render_c2ws = []

    for theta in np.linspace(0., 2. * np.pi, n_render_views, endpoint=False, dtype=c2ws.dtype):
        center = radius * np.asarray([cos_phi * np.cos(theta), cos_phi * np.sin(theta), sin_phi], dtype=c2ws.dtype)
        center[2] += orbit_height
        v_front = -normalize(center)
        center += c2w_avg[..., :3, -1]  # last dim, center of avg
        v_right = normalize(np.cross(v_front, v_up))
        v_down = np.cross(v_front, v_right)
        c2w = np.stack([v_right, v_down, v_front, center], axis=-1)  # 3, 4
        render_c2ws.append(c2w)

    render_c2ws = np.stack(render_c2ws, axis=0)  # N, 3, 4
    return render_c2ws


# Adapted from VGGT
# https://github.com/facebookresearch/vggt/blob/main/vggt/utils/rotation.py
def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    """ Quaternion Order: XYZW or say ijkr, scalar-last.
        Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """ Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
        Quaternion Order: XYZW or say ijkr, scalar-last
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # We produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # If not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))

    # Convert from rijk to ijkr
    out = out[..., [1, 2, 3, 0]]

    out = standardize_quaternion(out)
    return out


def encode_camera_params(
    ext: torch.Tensor,
    ixt: torch.Tensor,
    H: Union[int, torch.Tensor, List],
    W: Union[int, torch.Tensor, List],
    type: str = 'abs_quat_fov',
):
    if type == 'abs_quat_fov':
        T = ext[..., :3, 3]  # (..., 3)
        Q = mat_to_quat(ext[..., :3, :3])  # (..., 4)
        F = torch.stack([
            2 * torch.atan(H / (2 * ixt[..., 1, 1])),
            2 * torch.atan(W / (2 * ixt[..., 0, 0])),
        ], dim=-1)  # (..., 2)
        cam = torch.cat([T, Q, F], dim=-1).float()  # (..., 9)
    else:
        raise ValueError(f'Unknown camera parameter encoding type: {type}')
    return cam


def decode_camera_params(
    cam: torch.Tensor,
    H: Union[int, torch.Tensor, List] = None,
    W: Union[int, torch.Tensor, List] = None,
    type: str = 'abs_quat_fov',
):
    if type == 'abs_quat_fov':
        T = cam[..., 0:3]  # (..., 3)
        Q = cam[..., 3:7]  # (..., 4)
        F = cam[..., 7:9]  # (..., 2)
        ext = torch.cat([quat_to_mat(Q), T[..., None]], dim=-1)  # (..., 3, 4)
        ixt = torch.zeros(cam.shape[:-1] + (3, 3), device=cam.device)
        ixt[..., 0, 0] = W / 2 / torch.tan(F[..., 1] / 2)
        ixt[..., 1, 1] = H / 2 / torch.tan(F[..., 0] / 2)
        ixt[..., 0, 2] = W / 2
        ixt[..., 1, 2] = H / 2
        ixt[..., 2, 2] = 1.0
    else:
        raise ValueError(f'Unknown camera parameter encoding type: {type}')
    return ext, ixt


def camera_to_relative_degree(
    target: torch.Tensor,
    prediction: torch.Tensor,
):
    """ Compute the relative pose of input camera extrinsic,
        both the prediction and the ground truth.
        NOTE: it seems that VGGT uses w2c as input instead of c2w.

    Args:
        target (torch.Tensor), (B, S, 3, 4): ground truth camera extrinsic
        prediction (torch.Tensor), (B, S, 3, 4): predicted camera extrinsic

    Returns:
        rrd (torch.Tensor), (B, P): relative rotation degree
        rtd (torch.Tensor), (B, P): relative translation degree
    """
    # Deal with nasty shapes
    if target.ndim < 4:
        target = target[None]  # (1, S, 3, 4)
        prediction = prediction[None]  # (1, S, 3, 4)

    # Generate pairwise indices to compute relative poses
    B, S, _, _ = prediction.shape
    i, j = get_pairs(S)
    i = i.to(target.device, non_blocking=True)
    j = j.to(target.device, non_blocking=True)

    with torch.no_grad():
        # Iterate through the batch dimension
        rotation = []
        translation = []

        for b in range(B):
            # Compute relative camera poses between pairs
            relative_target = affine_padding(target[b][i]).bmm(
                affine_padding(affine_inverse(target[b][j]))
            )  # (P, 4, 4)
            relative_prediction = affine_padding(prediction[b][i]).bmm(
                affine_padding(affine_inverse(prediction[b][j]))
            )  # (P, 4, 4)

            # Compute the difference in rotation and translation
            rotation.append(compute_rotation_angle(
                relative_target[:, :3, :3], relative_prediction[:, :3, :3]
            ))  # (P,)
            translation.append(compute_translation_angle(
                relative_target[:, :3, 3], relative_prediction[:, :3, 3]
            ))  # (P,)

        # Stack the results
        rotation = torch.stack(rotation, dim=0)  # (B, P)
        translation = torch.stack(translation, dim=0)  # (B, P)

    return rotation, translation


def camera_to_relative_degree_centers(
    target: torch.Tensor,
    prediction: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Alternative relative pose error computation used for VGGT paper-style AUC.

    - Rotation error: compare relative rotations R_i * R_j^T.
    - Translation error: compare *world-frame* translation directions given by
      camera center differences (C_i - C_j), which is invariant to the choice of
      per-camera coordinate frames (unlike using the translation part of the
      relative transform directly).
    """
    # Deal with nasty shapes
    if target.ndim < 4:
        target = target[None]  # (1, S, 3, 4)
        prediction = prediction[None]  # (1, S, 3, 4)

    B, S, _, _ = prediction.shape
    i, j = get_pairs(S)
    i = i.to(target.device, non_blocking=True)
    j = j.to(target.device, non_blocking=True)
    P = int(i.numel())

    Rt = target[..., :3, :3]
    tt = target[..., :3, 3]
    Rp = prediction[..., :3, :3]
    tp = prediction[..., :3, 3]

    # Relative rotation matrices (B, P, 3, 3)
    rel_Rt = Rt[:, i] @ Rt[:, j].transpose(-1, -2)
    rel_Rp = Rp[:, i] @ Rp[:, j].transpose(-1, -2)

    # Camera centers in world: C = -R^T t (B, S, 3)
    Ct = -(Rt.transpose(-1, -2) @ tt[..., None]).squeeze(-1)
    Cp = -(Rp.transpose(-1, -2) @ tp[..., None]).squeeze(-1)

    # Translation directions in world (B, P, 3)
    dt = Ct[:, i] - Ct[:, j]
    dp = Cp[:, i] - Cp[:, j]

    # Flatten pairs then reshape back via batch_size.
    rotation = compute_rotation_angle(rel_Rt.reshape(B * P, 3, 3), rel_Rp.reshape(B * P, 3, 3), batch_size=B)
    translation = compute_translation_angle(dt.reshape(B * P, 3), dp.reshape(B * P, 3), batch_size=B)
    return rotation, translation


def get_pairs(S: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # Pairing protocol for pose AUC evaluation.
    # Default ("all"): use all unordered pairs C(S, 2) as in the original implementation.
    # Optional ("anchor"): only compare each view against the first view (0,k), k=1..S-1.
    # This matches some multi-view pose benchmarks that treat the first view as the canonical frame.
    mode = os.environ.get("EVC_CAM_AUC_PAIRING", "all").strip().lower()
    if mode in {"temporal", "time", "sequential", "consecutive", "adjacent"}:
        if S <= 1:
            return (
                torch.empty((0,), dtype=torch.long),
                torch.empty((0,), dtype=torch.long),
            )
        i = torch.arange(0, S - 1, dtype=torch.long)
        j = torch.arange(1, S, dtype=torch.long)
        return i, j
    if mode in {"anchor", "ref", "first", "first_view"}:
        if S <= 1:
            return (
                torch.empty((0,), dtype=torch.long),
                torch.empty((0,), dtype=torch.long),
            )
        i = torch.zeros((S - 1,), dtype=torch.long)
        j = torch.arange(1, S, dtype=torch.long)
        return i, j

    i, j = torch.combinations(torch.arange(S), 2, with_replacement=False).unbind(-1)
    # Keep backward-compatible shape logic.
    i, j = [ (k[None] + torch.arange(1)[:, None] * S).reshape(-1) for k in (i, j) ]
    return i, j


def compute_rotation_angle(
    target: torch.Tensor,
    prediction: torch.Tensor,
    batch_size: Optional[int] = None,
    eps: float = 1e-15,
):
    # Lazy import
    from easyvolcap.utils.cam_utils import mat_to_quat

    # Convert rotation matrices to quaternions
    qt = mat_to_quat(target)  # (P, 4)
    qp = mat_to_quat(prediction)  # (P, 4)

    # Compute the relative rotation angle
    angle = torch.arccos(
        1 - 2 * (
            1 - (qt * qp).sum(dim=-1) ** 2
        ).clamp(min=eps)
    )  # (P,)
    degree = angle * 180 / np.pi

    if batch_size is not None:
        degree = degree.reshape(batch_size, -1)
    return degree


def compute_translation_angle(
    target: torch.Tensor,
    prediction: torch.Tensor,
    batch_size: Optional[int] = None,
    ambiguity=True,
):
    # Compute the relative translation with angle
    angle = compare_translation_by_angle(target, prediction)
    degree = angle * 180.0 / np.pi

    if ambiguity:
        degree = torch.min(degree, (180 - degree).abs())

    if batch_size is not None:
        degree = degree.reshape(batch_size, -1)
    return degree


def compare_translation_by_angle(
    target,
    prediction,
    eps=1e-15,
    default_angle=1e6
):
    """ Normalize the translation vectors and compute the angle between them. """
    # Normalize the target
    target = target / (
        torch.norm(target, dim=-1, keepdim=True) + eps
    )  # (S, 3)
    # Normalize the prediction
    prediction = prediction / (
        torch.norm(prediction, dim=-1, keepdim=True) + eps
    )  # (S, 3)

    # 1 - cos^2(\theta) = sin^2(\theta)
    sinsq = torch.clamp_min(
        1.0 - torch.sum(target * prediction, dim=-1) ** 2,
        eps
    )  # (S,)
    angle = torch.acos(torch.sqrt(1 - sinsq))  # (S,)

    # Fill the invalid with default value
    angle[torch.isnan(angle) | torch.isinf(angle)] = default_angle
    return angle


def calculate_auc(error_r, error_t, threshold=30):
    """ Calculate the Area Under the Curve (AUC) for the given error arrays.

    Args:
        error_r (torch.Tensor), (*, P): representing R error values (Degree).
        error_t (torch.Tensor), (*, P): representing T error values (Degree).
        threshold: maximum threshold value for binning the histogram.
    Returns:
        histogram (torch.Tensor), (*, threshold + 1): normalized histogram of maximum error values.
    """
    # Deal with nasty shapes
    ndim = error_r.ndim
    if ndim < 2:
        # Always process the error tensors in the batch dimension
        error_r = error_r[None]  # (1, P)
        error_t = error_t[None]  # (1, P)

    # Concatenate the error tensors along a new axis
    error_mat = torch.stack([error_r, error_t], dim=-1)  # (*, P, 2)
    # Compute the maximum error value for each pair
    error_max, _ = torch.max(error_mat, dim=-1)  # (*, P)

    # Calculate histogram of maximum error values
    histogram = [
        torch.histc(error_max[b], bins=threshold + 1, min=0, max=threshold)
            for b in range(error_max.shape[0])
    ]
    histogram = torch.stack(histogram, dim=0)  # (B, threshold + 1)
    # Normalize the histogram by number of samples
    histogram = histogram / error_max.shape[-1]  # (B, threshold + 1)

    if ndim < 2:
        histogram = histogram[0]  # (threshold + 1,)

    # Return the normalized histogram
    return histogram


def calculate_auc_np(error_r, error_t, threshold=30, combine: str = "min"):
    """Calculate the pose AUC used in VGGT / PoseDiff-style evaluation.

    VGGT paper definition (AUC@30):
    - For each threshold t, compute RRA(t) = P(error_r < t), RTA(t) = P(error_t < t).
    - Accuracy(t) = min(RRA(t), RTA(t)).
    - AUC is the area under Accuracy(t) over thresholds (discretized to 1..threshold).

    Args:
        error_r: (B, P) rotation angular errors in degrees.
        error_t: (B, P) translation angular errors in degrees.
        threshold: maximum threshold (degrees).
        combine:
            - "min" (default): paper definition, use min(RRA, RTA) curve.
            - "max"/"legacy": legacy implementation, compute AUC over max(error_r, error_t).
    """
    # Deal with nasty shapes
    if error_r.ndim < 2:
        error_r = error_r[None]  # (1, P)
        error_t = error_t[None]  # (1, P)

    combine = (combine or "min").strip().lower()
    if combine in {"max", "legacy", "legacy_max", "max_err"}:
        # Legacy behavior: accuracy(t) = P(max(error_r, error_t) < t)
        error_max = np.maximum(error_r, error_t)  # (B, P)
        auc = 0.0
        for b in range(error_max.shape[0]):
            bins = np.arange(threshold + 1)
            histogram, _ = np.histogram(error_max[b], bins=bins)
            histogram_norm = histogram.astype(float) / float(len(error_max[b]))
            auc += float(np.mean(np.cumsum(histogram_norm)))
        return auc / float(error_max.shape[0])

    if combine not in {"min", "paper", "min_rra_rta", "min_rra"}:
        raise ValueError(f"Unknown combine={combine!r}, expected min|max.")

    # Paper behavior: accuracy(t) = min(P(error_r < t), P(error_t < t))
    ths = np.arange(1, threshold + 1, dtype=float)  # (T,)
    auc = 0.0
    for b in range(error_r.shape[0]):
        rra = np.mean(error_r[b][:, None] < ths[None, :], axis=0)  # (T,)
        rta = np.mean(error_t[b][:, None] < ths[None, :], axis=0)  # (T,)
        acc = np.minimum(rra, rta)  # (T,)
        auc += float(np.mean(acc))
    return auc / float(error_r.shape[0])


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """ Returns torch.sqrt(torch.max(0, x))
        but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """ Convert a unit quaternion to a standard form: one in which the
        real part is non negative.

    Args:
        quaternions: Quaternions with real part last, as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def add_noise_to_poses(poses: torch.Tensor, trans_sigma=0.01, rot_sigma_deg=1.0):
    """
    给 N×4×4 的 pose tensor 加噪声
    poses: (N, 4, 4) 的张量
    trans_sigma: 平移噪声 (米)
    rot_sigma_deg: 旋转噪声 (度)
    """
    device = poses.device
    N = poses.shape[0]

    # === 平移噪声 ===
    trans_noise = torch.randn(N, 3, device=device) * trans_sigma
    poses_noisy = poses.clone()
    poses_noisy[:, :3, 3] += trans_noise
    

    # === 旋转噪声 ===
    # 随机旋转轴
    axis = torch.randn(N, 3, device=device)
    axis = axis / axis.norm(dim=1, keepdim=True)

    # 随机旋转角度 (弧度)
    angle = torch.randn(N, device=device) * (rot_sigma_deg * torch.pi / 180.0)
    # print(angle*180/np.pi)

    # Rodrigues 公式生成旋转扰动
    K = torch.zeros(N, 3, 3, device=device)
    K[:, 0, 1], K[:, 1, 0] = -axis[:, 2], axis[:, 2]
    K[:, 0, 2], K[:, 2, 0] =  axis[:, 1], -axis[:, 1]
    K[:, 1, 2], K[:, 2, 1] = -axis[:, 0], axis[:, 0]

    I = torch.eye(3, device=device).expand(N, 3, 3)
    angle = angle.view(N, 1, 1)
    R_delta = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)

    # 应用扰动: R' = R_delta @ R
    R = poses_noisy[:, :3, :3]
    poses_noisy[:, :3, :3] = torch.matmul(R_delta, R)

    return poses_noisy
