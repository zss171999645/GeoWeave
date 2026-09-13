import os
import cv2
import numpy as np
from horizon_driving_dataset import PoseTransformer, DatasetReader
from os.path import join
from concurrent.futures import ThreadPoolExecutor
from saturnv_eval_tools.utils.pose_tools import interpolate, transform_pose

import open3d as o3d

def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a depth map to camera coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: Camera coordinates (H, W, 3)
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    return cam_coords


def depth_to_world_coords_points(
    seg_map: np.ndarray,
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    eps=1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a depth map to world coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).
        extrinsic (np.ndarray): Camera extrinsic matrix of shape (4, 4). OpenCV camera coordinate convention, cam to world.

    Returns:
        tuple[np.ndarray, np.ndarray]: World coordinates (H, W, 3) and valid depth mask (H, W).
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask
    point_mask = depth_map > eps
    seg_mask = ~((seg_map == 20) | (seg_map == 2) | (seg_map == 38))
    point_mask = point_mask & seg_mask
    if np.sum(point_mask) == 0:
        return None, None, None
    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # Multiply with the inverse of extrinsic matrix to transform to world coordinates
    # extrinsic_inv is 4x4 (note closed_form_inverse_OpenCV is batched, the output is (N, 4, 4))
    cam_to_world_extrinsic = extrinsic

    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world  # HxWx3, 3x3 -> HxWx3
    # world_coords_points = np.einsum("ij,hwj->hwi", R_cam_to_world, cam_coords_points) + t_cam_to_world

    return world_coords_points, cam_coords_points, point_mask


def project_depth_to_pointcloud(rgb_path, seg_path, depth_path, c2w, K):
    if not os.path.exists(rgb_path) or not os.path.exists(seg_path) or not os.path.exists(depth_path):
        return np.zeros((0, 6))
    rgb = (cv2.imread(rgb_path) / 255.0)[:, :, ::-1]
    seg = cv2.imread(seg_path, -1)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH).astype(np.float32) / 256.0
    world_coords_points, _, point_mask = depth_to_world_coords_points(seg, depth, c2w, K)
    if world_coords_points is None:
        return np.zeros((0, 6))
    world_coords_points = world_coords_points[point_mask]
    rgb = rgb[point_mask]
    points = np.concatenate([world_coords_points, rgb], axis=1)
    # x = np.arange(0, width)
    # y = np.arange(0, height)
    # x, y = np.meshgrid(x, y)
    # x, y = x.reshape(-1), y.reshape(-1)
    # z = depth.reshape(-1)
    # rgb = rgb.reshape(-1, 3)
    # seg = seg.reshape(-1)
    # depth = depth.reshape(-1)
    # depth_mask = (z > 0.5) & (z < 20)
    # seg_mask = ~((seg == 20) | (seg == 2) | (seg == 38))
    # mask = depth_mask & seg_mask
    # if np.sum(mask) == 0:
    #     return []
    # x, y, z, rgb, seg, depth = x[mask], y[mask], z[mask], rgb[mask], seg[mask], depth[mask]
    # points = np.stack([x, y, np.ones_like(x)], axis=1)
    # points = (np.linalg.inv(K) @ points.T).T
    # points = points * z[:, np.newaxis]
    # points = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    # points = (c2w @ points.T).T
    # points = points[:, :3]
    # points = np.concatenate([points, rgb], axis=1)
    return points

if __name__ == "__main__":
    # clip_path = "/horizon-bucket/perception-dataprocess/4dlabel/static_visual/000_mvs_data/parking_data/20250630_mechanical_at128_eval/DZ298/20241105_D/garage__1730775214572__0/Debug/VisualSFM/Vision_Result/20241105-110834_572"
    # # odo_path = "/horizon-bucket/perception-dataprocess/4dlabel/static_visual/users/qingfeng.li/visual_test/pack_list_mode/output/park_mechanical_20250719_135933/DZ298/20241105_D/garage__1730775214572__10/4DLabel_output/VisualSFM_3DModel_0/odometry/20241105-110834_572"
    # odo_path = "/horizon-bucket/perception-dataprocess/4dlabel/static_visual/000_mvs_data/parking_data/20250630_mechanical_at128_eval/DZ298/20241105_D/garage__1730775214572__0/4DLabel_output/VisualSFM_3DModel_0/odometry/20241105-110834_572"
    
    clip_path = "/horizon-bucket/perception-dataprocess/4dlabel/static_visual/000_mvs_data/parking_data/20250307/DZ689/20240810_D/garage__576__35/Debug/VisualSFM/Vision_Result/20240810-104414_576"
    odo_path = "/horizon-bucket/perception-dataprocess/4dlabel/static_visual/000_mvs_data/parking_data/20250307/DZ689/20240810_D/garage__576__35/4DLabel_output/VisualSFM_3DModel_1/odometry/20240810-104414_576"
    camera_names = [
                    # "camera_front", 
                    # "camera_front_left", 
                    # "camera_front_right",
                    # "camera_rear", 
                    # "camera_rear_left", 
                    # "camera_rear_right"
                    "fisheye_front",
                    "fisheye_left",
                    "fisheye_right",
                    "fisheye_rear"
                ]
    
    # load_K
    clip_infos = {}
    for camera_name in camera_names:
        camera_info = {}
        camera_param_path = join(clip_path, camera_name, f"{camera_name}.params")
        with open(camera_param_path, "r") as f:
            camera_param = f.read()
            camera_param = camera_param.split(" ")
            K = np.eye(3)
            K[0, 0] = camera_param[0]
            K[1, 1] = camera_param[0]
            K[0, 2] = camera_param[1]
            K[1, 2] = camera_param[2]
            width = int(camera_param[3])
            height = int(camera_param[4])
            camera_info["K"] = K
            camera_info["width"] = width
            camera_info["height"] = height
        clip_infos[camera_name] = camera_info

    # load paths
    image_paths = {}
    seg_paths = {}
    depth_paths = {}
    for camera_name in camera_names:
        image_dir = join(clip_path, camera_name, "keyframe")
        ims = sorted(os.listdir(image_dir))
        ims = [join(image_dir, im) for im in ims]
        segs =[im.replace("keyframe", "seg").replace(".jpg", ".png") for im in ims]
        dpts = [im.replace(f"/{camera_name}/", f"/fixed_mixed_depth_{camera_name}/").replace(".jpg", ".png").replace("keyframe/", "") for im in ims]
        image_paths[camera_name] = ims
        seg_paths[camera_name] = segs
        depth_paths[camera_name] = dpts
    
    # get args
    task_args = []
    for camera_name in camera_names:
        camera_pose_path = join(odo_path, f"{camera_name}.txt")
        pt = PoseTransformer()
        pt.reset()
        pt.loadarray(np.loadtxt(camera_pose_path))
        # camera_tum = camera_tums[camera_name]
        K = clip_infos[camera_name]["K"]
        for im, seg, dpt in zip(image_paths[camera_name], seg_paths[camera_name], depth_paths[camera_name]):
            timestamp = int(im.split("/")[-1].split(".")[0]) / 1000.0
            # print(timestamp)
            try:
                c2w = pt.seek_by_timestamp(timestamp, 0.1, interpolate=False)
            except:
                print("error", timestamp)
                continue
            # tum_array = interpolate([timestamp], camera_tum)
            # w2c = transform_pose(tum_array)[0]
            # c2w = np.linalg.inv(w2c)
            # c2w_ = pt.seek_by_timestamp(timestamp, 0.1, interpolate=False)
            # print(np.abs(c2w - c2w_).max())
            task_args.append((im, seg, dpt, c2w, K))
        del pt

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(project_depth_to_pointcloud, *args) for args in task_args]
        results = [future.result() for future in futures]
    results = np.concatenate(results, axis=0)
    points = results[:, :3]
    colors = results[:, 3:]
    ply = o3d.geometry.PointCloud()
    ply.points = o3d.utility.Vector3dVector(points)
    ply.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud("data/20240810-104414_576.ply", ply)
