from os.path import join,exists
import numpy as np
from scipy.spatial.transform import Rotation
from evo.core.trajectory import PosePath3D
from evo.core.metrics import PoseRelation, RPE
from evo.core.units import Unit
from evo.core.metrics import id_pairs_from_delta
from horizon_driving_dataset import PoseTransformer, PoseEvaluator
from saturnv_eval_tools.utils.align import robust_pose_alignment
from easyvolcap.utils.metric_utils import align_sim3, align_se3

def interpolate(traget_ts, ref_tum, pt = None, interpolate=True):
    """
    Get the pose of ts images, interpolate colmap poses
        to match the timestamps of the images
    """
    if pt is None:
        pt = PoseTransformer()
        pt.loadarray(ref_tum)
    poses = []
    for timestamp in traget_ts:
        try:
            interpolated_transform = pt.seek_by_timestamp(
                timestamp, 1.5, interpolate=interpolate
            )
            tvec = interpolated_transform[:3, 3]
            qvec = Rotation.from_matrix(
                interpolated_transform[:3, :3]).as_quat()
        except RuntimeError as e:
            print(f"{timestamp}: {e} use the nearest one, ts diff {min(np.abs(timestamp - ref_tum[:, 0]))}")
            all_ts = ref_tum[:, 0]
            idx = np.argmin(np.abs(all_ts - timestamp))
            target_tum = ref_tum[idx]
            tvec = target_tum[1:4]
            qvec = target_tum[4:]

        # x y z qx qy qz qw
        pose = [float(i) for i in [timestamp, tvec[0], tvec[1], tvec[2], qvec[0], qvec[1], qvec[2], qvec[3]]]  # noqa: E501
        poses.append(pose)
    poses = np.asarray(poses)
    return poses

def tum2T44(pose_tum):
    ts = pose_tum[0]
    t = pose_tum[1:4]
    q = pose_tum[4:]

    rot = Rotation.from_quat(q)
    R_mat = rot.as_matrix()  # (N, 3, 3)

    # 构造正向 T
    T44 = np.eye(4)
    T44[:3, :3] = R_mat
    T44[:3, 3] = t
    
    return T44, ts

def T442tum(T44, ts):
    T44 = T44 / T44[3, 3]
    R = T44[:3, :3]
    scale = np.cbrt(np.linalg.det(R))  # 取立方根作为尺度
    R_unit = R / scale
    t = T44[:3, 3]
    q = Rotation.from_matrix(R_unit).as_quat()
    ts = np.array([ts])
    tum_pose = np.concatenate([ts, t, q], axis=0)
    return tum_pose

def transform_pose(pose_tum, type="matrix"):
    """
    输入：
        pose_tum: shape (N, 8) 或 (8,) 的 ndarray，格式 [ts, tx, ty, tz, qx, qy, qz, qw]
        type: 'matrix' | 'tum'

    输出：
        - type='matrix': 返回 (N, 4, 4) 或 (4, 4) 的逆变换矩阵
        - type='tum': 返回 (N, 8) 或 (8,) 的逆变换 TUM pose
    """
    pose_tum = np.asarray(pose_tum)
    is_single = False

    if pose_tum.ndim == 1:
        pose_tum = pose_tum[None, ...]  # 转为 (1, 8)
        is_single = True

    assert pose_tum.shape[1] == 8, "Expected shape (N, 8) or (8,)"

    ts = pose_tum[:, 0]
    t = pose_tum[:, 1:4]
    q = pose_tum[:, 4:]

    rot = Rotation.from_quat(q)
    R_mat = rot.as_matrix()  # (N, 3, 3)

    # 构造正向 T
    T = np.eye(4)[None, ...].repeat(len(pose_tum), axis=0)  # (N, 4, 4)
    T[:, :3, :3] = R_mat
    T[:, :3, 3] = t

    # 求逆
    T_inv = np.eye(4)[None, ...].repeat(len(pose_tum), axis=0)
    T_inv[:, :3, :3] = np.transpose(R_mat, (0, 2, 1))  # R.T
    T_inv[:, :3, 3] = -np.matmul(T_inv[:, :3, :3], t[..., None]).squeeze(-1)

    if type == "matrix":
        return T_inv[0] if is_single else T_inv
    elif type == "tum":
        t_inv = T_inv[:, :3, 3]
        R_inv = Rotation.from_matrix(T_inv[:, :3, :3])
        q_inv = R_inv.as_quat()
        out = np.concatenate([ts[:, None], t_inv, q_inv], axis=1)
        return out[0] if is_single else out
    else:
        raise ValueError(f"Unknown output type: {type}")
    


def tum_to_matrix(tum_line):
    tx, ty, tz = tum_line[1:4]
    qx, qy, qz, qw = tum_line[4:8]
    rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = [tx, ty, tz]
    return mat

def load_tum_pose_dict(txt_file):
    arr = np.loadtxt(txt_file).reshape(-1, 8)
    poses = {}
    for i in range(len(arr)):
        poses[i] = tum_to_matrix(arr[i])
    return poses


def compute_rpe(gt_dict, pred_dict):
    traj_gt = PosePath3D(poses_se3=[gt_dict[i] for i in sorted(gt_dict.keys())])
    traj_pred = PosePath3D(poses_se3=[pred_dict[i] for i in sorted(pred_dict.keys())])

    # Translational error
    rpe_trans_metric = RPE(
        pose_relation=PoseRelation.translation_part,
        delta=5, delta_unit=Unit.frames, pairs_from_reference=True
    )
    rpe_trans_metric.process_data((traj_gt, traj_pred))
    
    # 获取 GT pair 索引 (i, j)
    id_pairs = id_pairs_from_delta(
        traj_gt.poses_se3 if rpe_trans_metric.pairs_from_reference else traj_pred.poses_se3,
        delta=rpe_trans_metric.delta,
        delta_unit=rpe_trans_metric.delta_unit,
        rel_tol=rpe_trans_metric.rel_delta_tol, 
        all_pairs=rpe_trans_metric.all_pairs
    )
    # 用 GT 的位置信息计算每对的实际距离
    positions_gt = traj_gt.positions_xyz
    gt_dists = np.array([
        np.linalg.norm(positions_gt[j] - positions_gt[i])
        for i, j in id_pairs
    ])
    errors = np.array(rpe_trans_metric.error)
    trans_err_per_meter = errors / gt_dists
    
    # Rotational error in degrees
    rpe_rot_metric = RPE(
        pose_relation=PoseRelation.rotation_angle_deg,
        delta=5, delta_unit=Unit.frames, pairs_from_reference=True
    )
    rpe_rot_metric.process_data((traj_gt, traj_pred))
    rot_errors = np.array(rpe_rot_metric.error)
    rot_error_per_meter = rot_errors / gt_dists  # 与 trans 用同一 id_pairs
    
    mean_trans_err = np.mean(trans_err_per_meter)
    mean_rot_err = np.mean(rot_error_per_meter)
    
    return mean_trans_err, mean_rot_err

def evo_rpe(gt_file, pred_file, align_mode="sim3"):
    if not exists(gt_file) or not exists(pred_file):
        return 0, 0
    gt_dict = load_tum_pose_dict(gt_file)
    pred_dict = load_tum_pose_dict(pred_file)
    if len(gt_dict) == 0 or len(pred_dict) == 0:
        return 0, 0
    try:
        if align_mode == "sim3":
            scale, traj_aligned = align_sim3(gt_dict, pred_dict)
        elif align_mode == "se3":
            scale, traj_aligned = align_se3(gt_dict, pred_dict)
        else:
            raise ValueError(f"Invalid align_mode: {align_mode}")
        rpe_trans, rpe_rot = compute_rpe(gt_dict, traj_aligned)
    except:
        print(f"Failed to align {gt_file} and {pred_file}")
        rpe_trans, rpe_rot = 0, 0
    return rpe_trans, rpe_rot

def eval_smalltraj_ate(gt_file, pred_file, align_mode="sim3"):
    # gt/pred_file: cam2world matrix
    if not exists(gt_file) or not exists(pred_file):
        return 0, 0
    gt_dict = load_tum_pose_dict(gt_file)
    pred_dict = load_tum_pose_dict(pred_file)
    if len(gt_dict) == 0 or len(pred_dict) == 0:
        return 0, 0
    if align_mode == "sim3":
        scale, traj_aligned = align_sim3(gt_dict, pred_dict)
    elif align_mode == "se3":
        scale, traj_aligned = align_se3(gt_dict, pred_dict)
    else:
        raise ValueError(f"Invalid align_mode: {align_mode}")
    rel_results = []
    for key in gt_dict.keys():
        rel_pose = np.linalg.inv(gt_dict[key]) @ traj_aligned[key]
        rel_angle = np.arccos((np.trace(rel_pose[:3, :3]) - 1) / 2) * 180 / np.pi
        rel_dist = np.linalg.norm(rel_pose[:3, 3])
        rel_results.append([rel_angle, rel_dist])
    rel_results = np.array(rel_results)
    ate_angle = np.mean(rel_results[:, 0])
    ate_dist = np.mean(rel_results[:, 1])
    # print("framewise dist", rel_results[:, 1])
    # print("framewise angle", rel_results[:, 0])
    return ate_dist, ate_angle


# sync timestamps
def get_nearest_timestamp(timestamp, timestamps, delta_ms, t_max_diff):
    """
    Input:
        timestamp: timestamp
        timestamps: timestamps
        delta_ms: delta time in ms
    Return:
        nearest timestamp
    """
    timestamp_offset = timestamp + delta_ms
    right_idx = np.searchsorted(
        timestamps, timestamp_offset, side="left"
    )
    left_idx = right_idx - 1
    left_time_diff = (
        timestamp_offset - timestamps[left_idx]
        if left_idx >= 0 else float("inf")
    )
    right_time_diff = (
        timestamps[right_idx] - timestamp_offset
        if right_idx < len(timestamps) else float("inf")
    )
    time_diff = min(left_time_diff, right_time_diff)
    if time_diff > t_max_diff:
        return None
    else:
        query_index = (
            left_idx
            if left_time_diff < right_time_diff else right_idx
        )
        return query_index

def get_sync_timestamps(ref_timestamps, timestamps_list, delta_ms, t_max_diff):
    """
    Get synchronized timestamps
    Input:
        ref_timestamps: reference timestamps, N
        timestamps_list: list of timestamps, X * M
        delta_ms: delta time in ms
        t_max_diff: max time difference
    Return:
        sync_timestamps: synchronized timestamps
    """
    sync_timestamps = []
    sync_idxs = []
    for idx, timestamp in enumerate(ref_timestamps):
        per_sync_timestamps = []
        per_sync_timestamps.append(timestamp)
        valid = True
        for timestamps in timestamps_list:
            query_index = get_nearest_timestamp(timestamp, timestamps, delta_ms, t_max_diff)
            if query_index is None:
                valid = False
                break
            per_sync_timestamps.append(timestamps[query_index])
        if valid:
            sync_timestamps.append(per_sync_timestamps)
            sync_idxs.append(idx)
    sync_timestamps = np.array(sync_timestamps, dtype=np.int64)
    sync_idxs = np.array(sync_idxs, dtype=np.int64)
    return sync_timestamps, sync_idxs

def align_pose(tar_pose, src_pose, scale=None, src2tar=None):
    # align pose to target
    if scale is None or src2tar is None:
        r, t, scale, residuals = robust_pose_alignment(src_pose, tar_pose)
        # logger.info(f"mean residual: {np.mean(residuals, axis=0)}")
        # logger.info(f"r: {r}, t: {t}, scale: {scale}")
        
        # align pose
        src2tar = np.eye(4)
        src2tar[:3, :3] = r # vggtworld2odoworld
        src2tar[:3, 3] = t

    pt = PoseTransformer()
    src_pose[:, 1:4] *= scale
    pt.loadarray(src_pose)
    pt.left_rotate(src2tar)
    poses_src_tarcoor = pt.dumparray()

    return scale, src2tar, poses_src_tarcoor


def cal_scale(tar_pose, src_pose):
    avg_scale_from = np.mean(np.linalg.norm(src_pose[1:, 1:4] - src_pose[0, 1:4], axis=1))
    if avg_scale_from <= 0.01:
        return src_pose, 1.0
    avg_scale_to = np.mean(np.linalg.norm(tar_pose[1:, 1:4] - tar_pose[0, 1:4], axis=1))
    scale_factor = avg_scale_to / avg_scale_from
    assert np.isfinite(scale_factor), f"scale_factor is not finite, {scale_factor}, c2ws_from: {src_pose[:, :3, 3]}"
    return scale_factor

def cam_pose_eval(clip_infos, camera_name, cam_posefile, pgo_filename):

    camera_cam2world = None
    if exists(join(cam_posefile, f"{camera_name}.txt")):
        camera_cam2world = np.loadtxt(join(cam_posefile, f"{camera_name}.txt"))

        timestamps = camera_cam2world[:, 0]

        sorted_indices = np.argsort(timestamps)
        camera_cam2world = camera_cam2world[sorted_indices]

        _, unique_indices = np.unique(camera_cam2world[:, 0], return_index=True)
        camera_cam2world = camera_cam2world[np.sort(unique_indices)]

    camera_cam2world_pgo = np.loadtxt(join(cam_posefile, f"{camera_name}{pgo_filename}.txt"))

    timestamps = camera_cam2world_pgo[:, 0]

    sorted_indices = np.argsort(timestamps)
    camera_cam2world_pgo = camera_cam2world_pgo[sorted_indices]

    _, unique_indices = np.unique(camera_cam2world_pgo[:, 0], return_index=True)
    camera_cam2world_pgo = camera_cam2world_pgo[np.sort(unique_indices)]



    gt_tum = clip_infos[camera_name]["pandar_cam2w"]
    sfm_tum = clip_infos[camera_name]["sfm_cam2w"]
    odo_tum = clip_infos[camera_name]["odo_cam2w"]
    matched_sfm_pose = sfm_tum[np.isin(sfm_tum[:, 0], camera_cam2world_pgo[:, 0])]
    res = {}
    debug_pose = {}
    for posetype, pre_tum in zip(["vggt", "odo", "sfm", "vggt_pgo"], [camera_cam2world, odo_tum, matched_sfm_pose, camera_cam2world_pgo]):
        if pre_tum is None:
            continue
        pe = PoseEvaluator(alignment="7dof")
        eval_result = pe.eval(gt_tum, pre_tum)
        rte = eval_result["RTE"]
        ate = eval_result["ATE"]
        rre = eval_result["RRE"]
        scale = eval_result["scale"]

        print(
            f"{posetype}-gt scale: {scale:.3f}, ATE: {ate:.3f} meters, RTE: {rte:.3f} %, RRE: {rre:.3f} deg/100m"  # noqa: E501
        )
        res[f"{posetype}_ate"] = ate
        res[f"{posetype}_rte"] = rte
        res[f"{posetype}_rre"] = rre
        
        # save pose
        vggt_pose = [pose[:3, 3] for pose in pe.poses_gt.values()]
        pred_pose = [pose[:3, 3] for pose in pe.poses_pred.values()]
        debug_pose[posetype] = pred_pose
        debug_pose[f"gt"] = vggt_pose
    
    return res, debug_pose