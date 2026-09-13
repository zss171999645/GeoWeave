import json
import math
from glob import glob
from os.path import basename, dirname, join

import numpy as np
from horizon_driving_dataset import DatasetReader, PoseTransformer

# from bokeh.models import ColumnDataSource
# from bokeh.plotting import figure, output_file, save, show
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


# 读取第一组位姿数据
def read_poses(file_path):
    poses = []
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("#"):
                continue
            elements = line.strip().split()
            timestamp = float(elements[0])
            x, y, z = (
                float(elements[1]),
                float(elements[2]),
                float(elements[3]),
            )
            qx, qy, qz, qw = (
                float(elements[4]),
                float(elements[5]),
                float(elements[6]),
                float(elements[7]),
            )
            pose = {
                "timestamp": timestamp,
                "position": np.array([x, y, z]),
                "quaternion": np.array([qx, qy, qz, qw]),
            }
            poses.append(pose)
    return poses


def transform_frame_to_poses(frames):
    pose_list = []

    for row in frames:
        timestamp = row[0]
        x, y, z = row[1:4]
        yaw = row[4]
        state = row[5]

        # 将yaw角度转换为四元数表示
        qx = 0
        qy = 0
        qz = math.sin(yaw / 2)
        qw = math.cos(yaw / 2)

        pose = {
            "timestamp": timestamp,
            "position": np.array([x, y, z]),
            "quaternion": np.array([qx, qy, qz, qw]),
            "state": state,
        }

        pose_list.append(pose)
    times = np.array([pose["timestamp"] for pose in pose_list])
    position = np.array([pose["position"] for pose in pose_list])
    quaternion = np.array([pose["quaternion"] for pose in pose_list])
    state = np.array([pose["state"] for pose in pose_list])

    poses = np.column_stack((times, position, quaternion, state))
    return poses


# 线性插值
# poses1: estimated
# poses2: GT
def interpolate_poses(poses2):
    # todo use slerp for rotation
    times2 = poses2[:, 0]
    position2 = poses2[:, 1:4]
    quaternion2 = poses2[:, 4:8]
    # state2 = poses2[:, -1]
    interpolator_trans = interp1d(
        times2, position2, axis=0, kind="linear", fill_value="extrapolate"
    )
    # interpolator_rot = interp1d(
    #     times2, quaternion2, axis=0, kind="linear", fill_value="extrapolate"
    # )
    rot2 = Rotation.from_quat(quaternion2)
    interpolator_rot = Slerp(times2, rot2)

    return interpolator_trans, interpolator_rot


# poses1: estimated
# poses2: GT
def calculate_errors(
    position1, quaternion1, interpolated_trans2, interpolated_quat2
):
    # 将每组位姿转换为齐次变换矩阵
    transformations1 = []
    transformations2 = []
    for quat, trans in zip(quaternion1, position1):
        R = Rotation.from_quat(quat)
        T = np.eye(4)
        T[:3, :3] = R.as_matrix()
        T[:3, 3] = trans[:3]
        transformations1.append(T)

    for quat, trans in zip(interpolated_quat2, interpolated_trans2):
        R = Rotation.from_quat(quat)
        T = np.eye(4)
        T[:3, :3] = R.as_matrix()
        T[:3, 3] = trans[:3]
        transformations2.append(T)

    # 计算第一组位姿相对于第二组位姿的相对变换矩阵
    relative_transformations = []
    for T1, T2 in zip(transformations1, transformations2):
        relative_transform = np.linalg.inv(T2).dot(T1)
        relative_transformations.append(relative_transform)

    long_error = [abs(t[0, 3]) for t in relative_transformations]
    lat_error = [abs(t[1, 3]) for t in relative_transformations]
    norm_error = [
        np.sqrt(t[0, 3] * t[0, 3] + t[1, 3] * t[1, 3])
        for t in relative_transformations
    ]
    return long_error, lat_error, norm_error


def read_ins_data(ins_path):
    with open(ins_path, "r") as f:
        data = json.load(f)
        ins_frames = []
        for key, value in data.items():
            ts = float(value["timeStamp"]) / 1e3
            x = value["x"]
            y = value["y"]
            z = value["z"]
            roll = value["roll"]
            pitch = value["pitch"]
            yaw = value["yaw"]
            r = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=False)
            qx, qy, qz, qw = r.as_quat()
            ins_frame = [ts, x, y, z, qx, qy, qz, qw]
            ins_frames.append(ins_frame)

    ins_frames = np.array(ins_frames)
    return ins_frames


def read_loc_data(sfm_path):
    sfm_pose = np.loadtxt(sfm_path)
    loc_frames = []
    for pose in sfm_pose:
        ts, x, y, z, qx, qy, qz, qw = pose[:]

        loc_frame = [ts, x, y, z, qx, qy, qz, qw]

        loc_frames.append(loc_frame)
    loc_frames = np.array(loc_frames)
    return loc_frames


def filter_loc_poses(loc_poses):
    state = loc_poses[:, -1]
    loc_poses = loc_poses[state == 2, :]
    return loc_poses


def keyframepose2gnss(sfm_path, gnss_path):
    sfm_pose = np.loadtxt(sfm_path)
    gnss_pose = {}

    for pose in sfm_pose:
        ts, x, y, z, qx, qy, qz, qw = pose[:]
        ts = int(ts * 1000)
        gnss_frame = {}
        gnss_frame["header"] = {}
        gnss_frame["header"]["stamp"] = str(ts)
        gnss_frame["header"]["frameId"] = "ub482"
        gnss_frame["sampleStamp"] = str(ts)
        gnss_frame["position"] = {}
        gnss_frame["position"]["x"] = x
        gnss_frame["position"]["y"] = y
        gnss_frame["position"]["z"] = z
        gnss_frame["orientation"] = {}
        gnss_frame["orientation"]["x"] = qx
        gnss_frame["orientation"]["y"] = qy
        gnss_frame["orientation"]["z"] = qz
        gnss_frame["orientation"]["w"] = qw
        gnss_frame["positionStdDev"] = {}
        gnss_frame["positionStdDev"]["x"] = 0
        gnss_frame["positionStdDev"]["y"] = 0
        gnss_frame["positionStdDev"]["z"] = 0

        gnss_pose[ts] = gnss_frame

    with open(gnss_path, "w") as f:
        json.dump(gnss_pose, f, indent=4)


def read_timestamp(image_stamp_path):
    with open(image_stamp_path, "r") as f:
        data = json.load(f)
        stamps = data["sync"]["camera_front"]
        timestamp = []
        for value in stamps:
            ts = value / 1000
            timestamp.append(ts)
    return timestamp


def transformation_from_trans_quat(trans, quat, scale_first=False):
    if scale_first:
        quat_scale_first = np.array([quat[1], quat[2], quat[3], quat[0]])
        R = Rotation.from_quat(quat_scale_first.T)
    else:
        R = Rotation.from_quat(quat)

    T = np.eye(4)
    T[:3, :3] = R.as_matrix()
    T[:3, 3] = trans[:3]
    return T


def generate_high_freq_pose(
    image_timestamps, odo_poses, keyframe_poses, cam_T_vcs, cam_T_lidar=None
):
    image_timestamps.sort()
    (
        interpolated_trans_ins,
        interpolated_quat_ins,
    ) = interpolate_poses(odo_poses)
    image_poses_tum = []
    for i in range(0, len(image_timestamps)):
        try:
            ts = image_timestamps[i]
            if ts < np.min(odo_poses[:, 0]) or ts > np.max(odo_poses[:, 0]):
                continue
            image_trans_ins = interpolated_trans_ins(ts)
            image_quat_ins = interpolated_quat_ins(ts).as_quat()
            # find the nearest keyframe pose by ts
            keyframe_ts = keyframe_poses[:, 0]
            keyframe_ts = keyframe_ts - ts
            keyframe_ts = np.abs(keyframe_ts)
            keyframe_idx = np.argmin(keyframe_ts)
            keyframe_ts = keyframe_poses[keyframe_idx][0]
            if np.abs(keyframe_ts - ts) > 1.0:
                continue
            keyframe_trans_odo = interpolated_trans_ins(keyframe_ts)
            keyframe_quat_odo = interpolated_quat_ins(keyframe_ts).as_quat()
            image_pose_odo = transformation_from_trans_quat(
                image_trans_ins, image_quat_ins
            ).dot(np.linalg.inv(cam_T_vcs))
            keyframe_pose_odo = transformation_from_trans_quat(
                keyframe_trans_odo, keyframe_quat_odo
            ).dot(np.linalg.inv(cam_T_vcs))
            keyframe_pose = transformation_from_trans_quat(
                keyframe_poses[keyframe_idx][1:4],
                keyframe_poses[keyframe_idx][4:8],
            )

            image_pose = keyframe_pose.dot(
                np.linalg.inv(keyframe_pose_odo)
            ).dot(image_pose_odo)
            if cam_T_lidar is not None:
                image_pose = image_pose.dot(cam_T_lidar)
            # transform image_pose to tum format
            image_pose_tum = np.zeros(8)
            image_pose_tum[0] = ts
            image_pose_tum[1:4] = image_pose[:3, 3]
            r = Rotation.from_matrix(image_pose[:3, :3])
            if r.as_quat()[-1] < 0:
                image_pose_tum[4:8] = r.as_quat() * -1
            else:
                image_pose_tum[4:8] = r.as_quat()
            image_poses_tum.append(image_pose_tum)
        except Exception as e:
            print(e)
            continue

    image_poses_tum = np.array(image_poses_tum)
    return image_poses_tum


def read_cam_poses_byclipdir(odometry_dir, clips_name=[]):
    clip_dirs = []
    for clip in list(glob((odometry_dir) + "/DZ*_202*_*")):
        clip_dirs.append(
            join(
                dirname(dirname(dirname(odometry_dir))),
                "Raw_data",
                basename(clip),
            )
        )
    clip_dirs.sort()
    if clips_name == []:
        clip_names = [basename(clip_dir) for clip_dir in clip_dirs]
    else:
        clip_names = clips_name

    clip_name2dir = dict(zip(clip_names, clip_dirs))

    camera_names = [
        "camera_front",
        "camera_front_left",
        "camera_front_right",
        "camera_rear",
        "camera_rear_left",
        "camera_rear_right",
    ]

    cam_poses_lidar_frame = {}
    cam_poses = {}

    for clip_name in clip_names:
        input_clip_dir = clip_name2dir[clip_name]
        cam_poses_lidar_frame_one_clip = {}
        cam_poses_one_clip = {}

        for camera_name in camera_names:
            colmap_tum_cam = np.loadtxt(
                join(odometry_dir, clip_name, f"{camera_name}.txt")
            )
            dr = DatasetReader(input_clip_dir)
            lidar2camera = dr.get_extrinsic("chassis", camera_name)
            pt = PoseTransformer()
            pt.loadarray(colmap_tum_cam)
            pt.right_rotate(lidar2camera)
            colmap_lidar_top = pt.dumparray()
            cam_poses_lidar_frame_one_clip[camera_name] = colmap_lidar_top
            cam_poses_one_clip[camera_name] = colmap_tum_cam

        cam_poses_lidar_frame[clip_name] = cam_poses_lidar_frame_one_clip
        cam_poses[clip_name] = cam_poses_one_clip

    return cam_poses_lidar_frame, cam_poses


def read_lidar_data(lidar_path):
    with open(lidar_path, "r") as f:
        data = json.load(f)
        lidar_poses = {}
        for value in data["reconstruct_clips"]:
            clip_dir = value["clip_dir"]
            clip_name = basename(clip_dir)
            poses = value["pose"]
            lidar_frames_one_clip = []
            for pose in poses:
                x = pose[0]
                y = pose[1]
                z = pose[2]
                # id = pose[3]
                roll = pose[4]
                pitch = pose[5]
                yaw = pose[6]
                ts = pose[7]
                r = Rotation.from_euler(
                    "xyz", [roll, pitch, yaw], degrees=False
                )
                qx, qy, qz, qw = r.as_quat()
                lidar_frame = [ts, x, y, z, qx, qy, qz, qw]
                lidar_frames_one_clip.append(lidar_frame)
            lidar_frames_one_clip = np.array(lidar_frames_one_clip)
            lidar_poses[clip_name] = lidar_frames_one_clip
    return lidar_poses
