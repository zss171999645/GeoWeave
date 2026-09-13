import open3d as o3d
import numpy as np
import glob
import os
import gc
from numba import njit, prange
from multiprocessing import Pool
from plyfile import PlyData, PlyElement
from tqdm import tqdm
from os.path import join, exists, dirname, basename
from collections import defaultdict
import psutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import torch

ROAD_LABELS = [0, 1, 7, 8, 24, 25, 26, 27, 28, 29, 30, 31, 32, 100]

def voxel_downsample(xyzs, rgbs, labels, block_ids, voxel_size):
    """
    xyzs: N×3 点坐标
    rgbs: N×3 RGB 颜色 [0,1]
    labels: N×1 标签
    block_ids: N×1 block_id
    voxel_size: [vx, vy, vz] 体素尺寸
    """
    assert len(voxel_size) == 3, "voxel_size must be a 3-element list"
    voxel_size = np.array(voxel_size)

    # 将点映射到体素格子索引
    voxel_indices = np.floor(xyzs / voxel_size).astype(np.int64)

    # 用 defaultdict 存索引
    voxel_dict = defaultdict(list)
    for i, index in tqdm(enumerate(voxel_indices)):
        voxel_dict[tuple(index)].append(i)

    downsampled_xyzs = []
    downsampled_rgbs = []
    downsampled_labels = []
    downsampled_blockids = []

    for idxs in tqdm(voxel_dict.values()):
        xyzs_ = xyzs[idxs]
        rgbs_ = rgbs[idxs]
        labels_ = labels[idxs]
        block_ids_ = block_ids[idxs]

        # 均值位置
        xyz = xyzs_.mean(axis=0)

        # RGB 平均
        rgb = rgbs_.mean(axis=0)

        # Label 按出现次数最多
        label = np.bincount(labels_.flatten()).argmax()

        # Block_id 按出现次数最多
        block_id = np.bincount(block_ids_.astype(np.int64).flatten()).argmax()

        downsampled_xyzs.append(xyz)
        downsampled_rgbs.append(rgb)
        downsampled_labels.append(label)
        downsampled_blockids.append(block_id)

    return (
        np.vstack(downsampled_xyzs),
        np.vstack(downsampled_rgbs),
        np.array(downsampled_labels)[:, None],
        np.array(downsampled_blockids)[:, None]
    )


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)



def get_voxel_size(sence_mode, label):
    road_labels = ROAD_LABELS
    if label == 2 or label == 18:
        return 0.02 if sence_mode == "packlist" else 0.05
    elif label in road_labels:
        return 0.01 if sence_mode == "packlist" else 0.02
    else:
        return 0.02


def process_grid_and_save(sence_mode, grid_index, grid_indices_x, grid_indices_y,
                          points, colors, normals, alphas, has_num_observations, num_observations, tmpdir):
    mask = (grid_indices_x == grid_index[0]) & (grid_indices_y == grid_index[1])
    batch_points = points[mask]
    batch_colors = colors[mask]
    batch_normals = normals[mask]
    batch_alphas = alphas[mask]

    if not len(batch_points):
        return []  # 空列表表示无输出文件

    unique_labels = np.unique(batch_alphas)
    results = []

    for label in unique_labels:
        label_mask = (batch_alphas == label)
        masked_points = batch_points[label_mask]
        masked_colors = batch_colors[label_mask]
        masked_normals = batch_normals[label_mask]

        if len(masked_points) == 0:
            continue

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(masked_points)
        pcd.colors = o3d.utility.Vector3dVector(masked_colors)
        pcd.normals = o3d.utility.Vector3dVector(masked_normals)

        voxel_size = get_voxel_size(sence_mode, label)
        downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        down_points = np.asarray(downpcd.points)
        down_colors = np.asarray(downpcd.colors)
        down_normals = np.asarray(downpcd.normals)
        down_alphas = np.full((len(down_points),), label)

        if has_num_observations:
            masked_num_obs = num_observations[mask][label_mask]
            pcd.colors = o3d.utility.Vector3dVector(np.column_stack((
                masked_num_obs,
                np.zeros_like(masked_num_obs),
                np.zeros_like(masked_num_obs))))
            downpcd_num_obs = pcd.voxel_down_sample(voxel_size=voxel_size)
            down_num_observations = np.asarray(downpcd_num_obs.colors)[:, 0]
            batch_result = np.column_stack((down_points, down_normals, down_colors, down_alphas, down_num_observations))
            # batch_result = np.column_stack((down_points, down_colors, down_alphas, down_num_observations))
        else:
            batch_result = np.column_stack((down_points, down_normals, down_colors, down_alphas))
            # batch_result = np.column_stack((down_points, down_colors, down_alphas))
            

        temp_file = os.path.join(tmpdir, f"temp_{int(time.time() * 1000)}.ply")
        write_ply_files(batch_result, temp_file, True)
        results.append(temp_file)

    return results  # 返回多个文件路径（每个 label 一个）或空列表

def merge_temp_files(temp_files, output_path):
    if not temp_files:
        print("No temporary files to merge.")
        return

    # 只保留真实存在的 .ply 文件路径
    valid_files = [f for f in temp_files if os.path.isfile(f)]
    if not valid_files:
        print("All temp files are invalid.")
        return

    # 读取第一个文件 header 作为模板
    first_data = PlyData.read(valid_files[0])
    vertex_element_template = first_data['vertex']
    vertex_dtype = vertex_element_template.data.dtype
    total_size = sum(PlyData.read(f)['vertex'].data.size for f in valid_files)

    merged_data = np.empty(total_size, dtype=vertex_dtype)
    current_idx = 0

    for temp_file in valid_files:
        data = PlyData.read(temp_file)['vertex'].data
        merged_data[current_idx:current_idx + len(data)] = data
        current_idx += len(data)

    final_vertex = PlyElement.describe(merged_data, 'vertex')
    PlyData([final_vertex]).write(output_path)
    

def merge_block_ply(block_savedir, save_ply_path):
    
    clips = os.listdir(block_savedir)
    clips = sorted(clips)
    
    all_points = []
    all_colors = []
    all_alphas = []
    all_blockids = []

    for clip in clips:
        input_folder = join(block_savedir, clip)
        ply_files = sorted(glob.glob(os.path.join(input_folder, "*.ply")))

        for block_id, ply_file in tqdm(enumerate(ply_files)):
            points, rgbs, alpha = parse_mvs_pc(ply_file)

            all_points.append(points)             # Nx3
            all_colors.append(rgbs)               # Nx3
            all_alphas.append(alpha.reshape(-1, 1)) # Nx1
            all_blockids.append(np.full((points.shape[0], 1), block_id, dtype=np.int32)) # Nx1

    # 合并
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)
    all_alphas = np.vstack(all_alphas)
    all_blockids = np.vstack(all_blockids)

    # 合成 Nx8
    merged_points = np.hstack([all_points, all_colors, all_alphas, all_blockids])

    # 保存
    write_ply_files(merged_points, save_ply_path)
    print(f"合并点云已保存到: {save_ply_path}")
    

def downsample_points_batch(sence_mode, block_savedir, save_ply_path, grid_size=100, use_parallel=True):
    print(f"Memory before reading PLY file: {get_memory_usage():.2f} MB")
    
    clips = os.listdir(block_savedir)
    clips = sorted(clips)
    
    all_points = []
    all_colors = []
    all_alphas = []
    all_normals = []
    for clip in clips:
        input_folder = join(block_savedir, clip)
        if os.path.exists(os.path.join(input_folder, "bilateral_filter")):
            input_folder = os.path.join(input_folder, "bilateral_filter")
            print(f"Using bilateral filter folder: {input_folder}")
        ply_files = sorted(glob.glob(os.path.join(input_folder, "*.ply")))


        for block_id, ply_file in tqdm(enumerate(ply_files)):
            points, rgbs, alpha, normals = parse_mvs_pc(ply_file, True)
            all_points.append(points)
            all_colors.append(rgbs)
            all_alphas.append(alpha.reshape(-1, 1))
            all_normals.append(normals)
    print(f"Memory after reading PLY file: {get_memory_usage():.2f} MB")
    
    points = np.vstack(all_points)
    colors = np.vstack(all_colors)
    alphas = np.vstack(all_alphas).reshape(-1)
    normals = np.vstack(all_normals)
    print(f"before del: {get_memory_usage():.2f} MB")
    del all_points, all_colors, all_alphas, all_normals
    gc.collect()
    print(f"Memory after stacking arrays: {get_memory_usage():.2f} MB")
    
    has_num_observations = False
    num_observations = None

    print(f"Total points: {len(points)}")
    print(f"Memory after loading data: {get_memory_usage():.2f} MB")

    grid_indices_x = (points[:, 0] // grid_size).astype(int)
    grid_indices_y = (points[:, 1] // grid_size).astype(int)
    unique_grids = np.unique(np.vstack((grid_indices_x, grid_indices_y)).T, axis=0)

    print(f"Unique grids: {len(unique_grids)}")
    print(f"Memory before processing grids: {get_memory_usage():.2f} MB")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_files = []

        if use_parallel and len(unique_grids) > 10:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(
                        process_grid_and_save,
                        sence_mode,
                        grid_index,
                        grid_indices_x,
                        grid_indices_y,
                        points,
                        colors,
                        normals,
                        alphas,
                        has_num_observations,
                        num_observations,
                        tmpdir
                    )
                    for grid_index in unique_grids
                ]
                for future in futures:
                    temp_files.extend(future.result())
        else:
            for grid_index in unique_grids:
                temp_files.extend(process_grid_and_save(
                    sence_mode, grid_index, grid_indices_x, grid_indices_y,
                    points, colors, normals, alphas,
                    has_num_observations, num_observations, tmpdir
                ))
        merge_temp_files(temp_files, save_ply_path)

    print(f"Memory after writing PLY file: {get_memory_usage():.2f} MB")
    print(f"Saved to: {save_ply_path}")
    
    
    
def write_ply_files(points_plane, path, include_normals=False):
    if include_normals:
        all_fields = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('alpha', 'u1'), ('block_id', 'i4')
        ]
    else:
        all_fields = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('alpha', 'u1'), ('block_id', 'i4')
        ]
    num_columns = points_plane.shape[1]
    fields = all_fields[:num_columns]

    structured_array = np.empty(points_plane.shape[0], dtype=np.dtype(fields))
    for i, field in enumerate(structured_array.dtype.names):
        structured_array[field] = points_plane[:, i]

    PlyData([PlyElement.describe(structured_array, 'vertex')]).write(path)


SIMPLE_FIELDS = [
    ("x", "f4"),
    ("y", "f4"),
    ("z", "f4"),
    ("red", "u1"),
    ("green", "u1"),
    ("blue", "u1"),
    ("alpha", "u1"),
]

def write_simple_ply_files(points_plane, output_path, all_fields=SIMPLE_FIELDS, print_info=True):
    num_columns = points_plane.shape[1]
    fields = all_fields[:num_columns]

    structured_array = np.empty(points_plane.shape[0], dtype=np.dtype(fields))
    for i, field in enumerate(structured_array.dtype.names):
        structured_array[field] = points_plane[:, i]

    PlyData([PlyElement.describe(structured_array, "vertex")]).write(
        output_path
    )
    if print_info is True:
        print(
            f"Saved {structured_array.shape[0]} points to {output_path}"
        )

def write_simple_bin_files(points, output_path):
    """
    保存点云数据为二进制文件
    输入点云应为7维：x, y, z, r, g, b, seg
    输出点云为6维：x, y, z, 0, 0, 0
    
    要求：
    1. 点云数据类型为float32
    2. 点云数据维度为6
       - dim0～2：xyz坐标（需要转换到vcs坐标系）
       - dim3：反射率强度（置0）
       - dim4～5：无效信息（置0）
    """
    # 确保输入点云是7维
    assert points.shape[1] == 7, "输入点云应为7维数据"
    
    # 创建6维输出数组
    output_points = np.zeros((points.shape[0], 6), dtype=np.float32)
    
    # 复制前3维（xyz坐标）
    output_points[:, :3] = points[:, :3]
    
    # 第4-6维已初始化为0
    # 不需要转换到vcs坐标系，因为输入已经是vcs坐标系
    
    # 保存为二进制文件
    output_points.tofile(output_path)

def parse_mvs_pc(mvs_ply, include_normals=False):
    """
    Parse mvs point cloud with labels
    """
    # parse road points
    ply_data = PlyData.read(mvs_ply)
    vertex_data = ply_data['vertex'].data
    x = np.array(vertex_data['x'])
    y = np.array(vertex_data['y'])
    z = np.array(vertex_data['z'])
    r = np.array(vertex_data['red'])
    g = np.array(vertex_data['green'])
    b = np.array(vertex_data['blue'])
    alpha = np.array(vertex_data['alpha'])
    xyzs = np.stack((x, y, z), axis=1)
    rgbs = np.stack((r, g, b), axis=1)
    if include_normals:
        if 'nx' in vertex_data.dtype.names:
            nx = np.array(vertex_data['nx'])
            ny = np.array(vertex_data['ny'])
            nz = np.array(vertex_data['nz'])
            normals = np.stack((nx, ny, nz), axis=1)
            return xyzs, rgbs, alpha, normals
        else:
            return xyzs, rgbs, alpha, None
    return xyzs, rgbs, alpha

def block_downsample(block_ply, save_ply_path, block_id, voxel_size=0.01):
    
    points, colors, alphas, normals = parse_mvs_pc(block_ply, True)
    alphas = alphas.reshape(-1, 1)
    block_ids = np.full((len(points), 1), block_id, dtype=np.float32)
    
    
    pcd_color = o3d.geometry.PointCloud()
    pcd_color.points = o3d.utility.Vector3dVector(points)
    pcd_color.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd_color.normals = o3d.utility.Vector3dVector(normals)
    pcd_block = o3d.geometry.PointCloud()
    pcd_block.points = o3d.utility.Vector3dVector(points)
    pcd_block.colors = o3d.utility.Vector3dVector(np.hstack([alphas, block_ids, np.zeros_like(block_ids)]))
    

    down_color = pcd_color.voxel_down_sample(voxel_size=voxel_size)
    down_block = pcd_block.voxel_down_sample(voxel_size=voxel_size)

    down_points = np.asarray(down_color.points)
    down_colors = np.asarray(down_color.colors)
    if normals is not None:
        down_normals = np.asarray(down_color.normals)
    down_alphas = np.round(np.asarray(down_block.colors)[:, 0]).astype(np.int32)
    down_blockid = np.round(np.asarray(down_block.colors)[:, 1]).astype(np.int32)

    if normals is not None:
        downsample_ply = np.column_stack((down_points, down_normals, down_colors, down_alphas, down_blockid))
    else:
        downsample_ply = np.column_stack((down_points, down_colors, down_alphas, down_blockid))
    
    print(f"downsample ratio: {len(down_points)/len(points)}")
    if normals is not None:
        write_ply_files(downsample_ply, save_ply_path, True)
    else:
        write_ply_files(downsample_ply, save_ply_path, False)


def process_one_file(args):
    clip, ply_file, block_id, save_path, site_dir = args
    save_ply_path = f"{save_path}/{clip}/{block_id}.ply"
    os.makedirs(os.path.dirname(save_ply_path), exist_ok=True)
    block_downsample(ply_file, save_ply_path, block_id)
    # block_downsample_torch(ply_file, save_ply_path, block_id)



def block_downsample_torch(block_ply, save_ply_path, block_id, voxel_size=0.01):
    # 读取数据
    points, colors, alphas = parse_mvs_pc(block_ply)
    alphas = alphas.reshape(-1, 1)
    block_ids = np.full((len(points), 1), block_id, dtype=np.int32)

    # 拼成一个完整属性矩阵
    data = np.hstack([points, colors, alphas, block_ids])  # [x,y,z,r,g,b,a,bid]
    
    # 转torch
    data_torch = torch.from_numpy(data).to(device="cuda")
    coords = torch.floor(data_torch[:, :3] / voxel_size)

    # 找到每个voxel第一个出现的点
    _, unique_indices = torch.unique(coords, dim=0, return_inverse=True, return_counts=False)
    unique_indices = unique_indices.sort()[0]  # 保证顺序

    # 取出降采样结果
    downsampled = data_torch[unique_indices].cpu().numpy()

    print(f"downsample ratio: {len(downsampled) / len(points):.4f}")

    write_ply_files(downsampled, save_ply_path)

    

def downsample_all_block(clips, save_path, site_dir, num_workers=8):
    tasks = []
    
    for clip in clips:
        input_folder = join(site_dir, clip, "before_optimization")
        ply_files = sorted(glob.glob(os.path.join(input_folder, "*.ply")))
        for block_id, ply_file in enumerate(ply_files):
            tasks.append((clip, ply_file, block_id, save_path, site_dir))
    
    # for t in tqdm(tasks):
    #     process_one_file(t)
    with Pool(processes=num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_one_file, tasks), total=len(tasks)):
            pass


@njit(parallel=True)
def bilateral_filter_numba(points, normals, indices, distances, sigma_space=0.1, sigma_normal=0.1):
    """
    用Numba加速的双边滤波
    indices: 每个点的邻域索引 (N, k)
    distances: 每个点到邻域的距离 (N, k)
    """
    n_points = points.shape[0]
    k = indices.shape[1]
    filtered_points = np.zeros_like(points, dtype=np.float32)
    
    for i in prange(n_points):
        # 获取邻域点和法向量
        neighbors = points[indices[i]]
        neighbor_normals = normals[indices[i]]
        center_normal = normals[i]
        
        # 计算空间权重
        spatial_weights = np.exp(-0.5 * (distances[i] **2) / (sigma_space** 2))
        
        # 计算法向量权重（点积 -> 角度差异）
        normal_dots = np.dot(neighbor_normals, center_normal)
        angle_diffs = 1 - normal_dots  # 转换为角度差异
        normal_weights = np.exp(-0.5 * (angle_diffs **2) / (sigma_normal** 2))
        
        # 组合权重并归一化
        combined = spatial_weights * normal_weights
        combined /= np.sum(combined)
        
        # 加权平均
        filtered_points[i] = np.sum(neighbors * combined.reshape(-1, 1), axis=0)        
    
    return filtered_points


def bilateral_filter_one(args):
    src_ply_file, dst_ply_file, block_id, k, sigma_space, sigma_normal = args
    from pykdtree.kdtree import KDTree
    points, colors, alphas, normals = parse_mvs_pc(src_ply_file, True)
    tree = KDTree(points.astype(np.float32))
    distances, indices = tree.query(points, k=k)
    filtered_points = bilateral_filter_numba(points, normals, indices, distances, sigma_space, sigma_normal)
    del tree, points, distances, indices
    # 重新计算法向量
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    o3d_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    normals = np.asarray(o3d_pcd.normals)

    alphas = alphas.reshape(-1, 1)
    block_ids = np.full((len(filtered_points), 1), block_id, dtype=np.float32)
    filtered_ply = np.column_stack((filtered_points, normals, colors, alphas, block_ids))
    write_ply_files(filtered_ply, dst_ply_file, True)


def bilateral_filter_all_clips(clips, site_dir, k=100, sigma_space=0.1, sigma_normal=0.1, num_workers=4):
    tasks = []
    for clip in clips:
        input_folder = join(site_dir, clip)
        ply_files = sorted(glob.glob(os.path.join(input_folder, "*.ply")))
        for block_id, ply_file in enumerate(ply_files):
            dst_ply_file = os.path.join(os.path.dirname(ply_file), "bilateral_filter", f"{block_id}.ply")
            os.makedirs(os.path.dirname(dst_ply_file), exist_ok=True)
            tasks.append((ply_file, dst_ply_file, block_id, k, sigma_space, sigma_normal))
    for t in tqdm(tasks):
        bilateral_filter_one(t)
    # with Pool(processes=num_workers) as pool:  # 会卡死，像是资源不够
    #     for _ in tqdm(pool.imap_unordered(bilateral_filter_one, tasks), total=len(tasks)):
    #         pass


if __name__ == "__main__":
    # 参数
    site_dir = "/home/users/yingfeng.cai/dev/meshx/data/park_mechanical_20250801_143848/mv_wfixscale/DZ298/20241105_D/garage__1730775214572__10"
    save_path = "/home/users/yingfeng.cai/dev/meshx/data"
    # site_dir = "/home/users/yingfeng.cai/dev/meshx/data/park_mechanical_20250801_143848/mv_fixscale_conf5/DZ298/20241105_D/garage__1730775214572__10"
    # save_path = "/home/users/yingfeng.cai/dev/meshx/data/garage__1730775214572__10_downsample_conf5"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path) 
    
    sence_mode="packlist"

    clips = os.listdir(site_dir)
    clips = sorted(clips)
    
    downsample_all_block(clips, save_path, site_dir, num_workers=16)
    
    site_ply_path = join(save_path, f"{basename(site_dir)}2.ply")
    # downsample_points_batch(sence_mode, save_path, site_ply_path)
    
    # merge_block_ply(save_path, site_ply_path)
    