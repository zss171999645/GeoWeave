"""
This file may save mesh, point cloud, volume bit mask or other volumetric representation to disk
"""
import os
import glob
import torch
import mcubes
import numpy as np
from torch import nn
from os.path import join
from functools import partial
from multiprocessing.pool import ThreadPool
from typing import List, Dict, Tuple, Union, Type, Callable

from easyvolcap.engine import cfg
from easyvolcap.engine import VISUALIZERS
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.data_utils import Visualization, export_pts, export_mesh
from easyvolcap.utils.chunk_utils import multi_gather, multi_scatter_
from easyvolcap.utils.dist_utils import get_rank, get_distributed


@VISUALIZERS.register_module()
class GeometryVisualizer:
    def __init__(self,
                 result_dir: str = f'data/geometry',
                 save_tag: str = '',
                 types: List[str] = [
                     Visualization.MESH.name,
                 ],
                 exts: Dict[str, str] = {
                     Visualization.MESH.name: ".ply",
                     Visualization.POINT.name: ".ply",
                 },
                 exports: Dict[str, Callable] = {
                     Visualization.MESH.name: (lambda mesh, filename: export_mesh(**mesh, filename=filename)),
                     Visualization.POINT.name: (lambda mesh, filename: export_pts(**mesh, filename=filename)),
                 },
                 verbose: bool = True,
                 pool_limit: int = 1,  # maximum number of pending tasks in the thread pool

                 occ_thresh: float = 0.5,
                 sdf_thresh: float = 0.0,
                 pts_thresh: int = 2073600,
                 dtu_official_ply_export: bool = False,
                 # True: 跨整个 test 按 scene 累积点云再写 scan*.ply（对齐 DTU Table2 / MASt3R 多视图融合评测）
                 dtu_official_ply_fuse: bool = True,
                 dtu_fuse_max_points: int = 6_000_000,
                 **kwargs,
                 ):
        self.occ_thresh = occ_thresh
        self.sdf_thresh = sdf_thresh
        self.pts_thresh = pts_thresh
        self.dtu_official_ply_export = dtu_official_ply_export
        self.dtu_official_ply_fuse = dtu_official_ply_fuse
        self.dtu_fuse_max_points = int(dtu_fuse_max_points)
        self._dtu_fuse_buffer: Dict[str, Dict[str, List[Union[torch.Tensor, Tuple[int, int]]]]] = {}

        result_dir = join(result_dir, cfg.exp_name)  # MARK: global configuration
        result_dir = join(result_dir, save_tag) if save_tag != '' else result_dir
        self.result_dir = result_dir
        self.types = [Visualization[t] for t in types]  # types of visualization
        self.exts = exts  # file extensions for each type of visualization
        self.exports = exports
        self.verbose = verbose

        self.thread_pools: List[ThreadPool] = []
        self.pool_limit = pool_limit
        self.geo_pattern = f'{{type}}/frame{{frame:04d}}_camera{{camera:04d}}{{ext}}'

        if verbose:
            log(f'Visualization output: {yellow(join(result_dir, dirname(self.geo_pattern)))}')  # use yellow for output path
            log(f'Visualization types:', line(types))

    def generate_type(self, output: dotdict, batch: dotdict, type: Visualization = Visualization.MESH):
        # Extract the renderable image from output and batch
        mesh: dotdict = None
        mesh_gt: Union[dotdict, None] = None
        mesh_dpt: Union[dotdict, None] = None

        if type == Visualization.MESH:
            if 'sdf' in output:
                occ = -output.sdf + 0.5
                self.occ_thresh = -self.sdf_thresh + 0.5
            else:
                occ = output.occ
            voxel_size = batch.meta.voxel_size
            W, H, D = batch.meta.W[0].item(), batch.meta.H[0].item(), batch.meta.D[0].item()  # !: BATCH
            cube = torch.full((np.prod(batch.valid.shape),), -10.0, dtype=occ.dtype, device='cpu')[None]  # 1, WHD
            cube = multi_scatter_(cube[..., None], batch.inds.cpu(), occ.cpu())  # dim = -2 # B, WHD, 1 assigned B, P, 1
            cube = cube.view(-1, W, H, D)  # B, W, H, D

            # We leave the results on CPU but as tensors instead of numpy arrays
            torch.cuda.synchronize()  # some of the batched data are asynchronously moved to the cpu
            verts, faces = mcubes.marching_cubes(cube.float().numpy()[0], self.occ_thresh)
            verts = torch.as_tensor(verts, dtype=torch.float)[None]
            faces = torch.as_tensor(faces.astype(np.int32), dtype=torch.int)[None]
            verts = verts * voxel_size.to(verts.dtype) + batch.meta.bounds[:, 0].to(verts.dtype)  # !: BATCH

            mesh = dotdict()
            mesh.verts = verts
            mesh.faces = faces

        elif type == Visualization.POINT:
            # DTU 官方评测需要与数据集/校准一致的坐标；decoder 的 xyz_map 可能在归一化/其它坐标系。
            # 反投影点 xyz_bcd（深度×位姿）与 GT 深度同一尺度，更接近 MVS 世界坐标。
            if self.dtu_official_ply_export and "xyz_bcd" in output:
                src = output.xyz_bcd
            else:
                src = output.xyz_map
            B = src.shape[0]
            xyz = src.clone().reshape(B, -1, 3)  # (B, P, 3), maybe multiple frames
            if 'rgb' in batch: rgb = batch.rgb.reshape(B, -1, 3)  # (B, P, 3)
            else: rgb = torch.zeros_like(xyz)

            idx = torch.arange(xyz.shape[1])
            # Downsample the point cloud
            if xyz.shape[1] > self.pts_thresh:
                idx = torch.randperm(xyz.shape[1])[:self.pts_thresh]
                xyz = xyz[:, idx]
                rgb = rgb[:, idx]

            mesh = dotdict()
            mesh.pts = xyz
            mesh.color = rgb

            if 'xyz' in batch:
                xyz_gt = batch.xyz.clone().reshape(B, -1, 3)
                mesh_gt = dotdict()
                mesh_gt.pts = xyz_gt[:, idx]
                mesh_gt.color = rgb

            if "xyz_bcd" in output:
                xyz_bcd = output.xyz_bcd.clone().reshape(B, -1, 3)
                mesh_dpt = dotdict()
                # dtu 主输出已用 xyz_bcd 时，_dpt 另存 point head 便于对比
                if self.dtu_official_ply_export:
                    mesh_dpt.pts = output.xyz_map.clone().reshape(B, -1, 3)[:, idx]
                else:
                    mesh_dpt.pts = xyz_bcd[:, idx]
                mesh_dpt.color = rgb

        else:
            raise NotImplementedError(f'Unimplemented visualization type: {type}')
        return mesh, mesh_gt, mesh_dpt

    def visualize_type(self, output: dotdict, batch: dotdict, type: Visualization = Visualization.MESH):
        geos, geos_gt, geos_dpt = self.generate_type(output, batch, type)  # can be batched

        geo_stats = dotdict()
        camera_index: torch.Tensor = batch.meta.camera_index
        frame_index: torch.Tensor = batch.meta.frame_index
        geo_paths = []
        geo_arrays = []

        for i in range(len(frame_index)):
            frame = frame_index[i].item()
            camera = camera_index[i].item()

            # For shared values
            geo_path = self.geo_pattern.format(type=type.name, camera=camera, frame=frame, ext=self.exts[type.name])
            geo_gt_path = geo_path.replace(self.exts[type.name], f'_gt{self.exts[type.name]}')
            geo_dpt_path = geo_path.replace(self.exts[type.name], f'_dpt{self.exts[type.name]}')

            # Geometries
            geo = dotdict({k: geos[k][i] for k in geos.keys()})
            if geos_gt is not None: geo_gt = dotdict({k: geos_gt[k][i] for k in geos_gt.keys()})
            if geos_dpt is not None: geo_dpt = dotdict({k: geos_dpt[k][i] for k in geos_dpt.keys()})

            # For recorder
            geo_stats[geo_path] = geo
            if geos_gt is not None: geo_stats[geo_gt_path] = geo_gt
            if geos_dpt is not None: geo_stats[geo_dpt_path] = geo_dpt

            # Saving images to disk
            geo_paths.append(join(self.result_dir, f'{batch.meta.iter:08d}', geo_path))
            geo_arrays.append(geo)
            if geos_gt is not None:
                geo_paths.append(join(self.result_dir, f'{batch.meta.iter:08d}', geo_gt_path))
                geo_arrays.append(geo_gt)
            if geos_dpt is not None:
                geo_paths.append(join(self.result_dir, f'{batch.meta.iter:08d}', geo_dpt_path))
                geo_arrays.append(geo_dpt)

            if (
                self.dtu_official_ply_export
                and type == Visualization.POINT
                and hasattr(batch.meta, "data_root")
                and batch.meta.data_root is not None
            ):
                dr = batch.meta.data_root
                if isinstance(dr, (list, tuple)) or (
                    hasattr(dr, "__getitem__") and not isinstance(dr, str) and hasattr(dr, "__len__")
                ):
                    try:
                        root = dr[i] if i < len(dr) else dr[0]
                    except Exception:
                        root = dr[0]
                else:
                    root = dr
                scene = os.path.basename(os.path.normpath(str(root)))
                if self.dtu_official_ply_fuse:
                    buf = self._dtu_fuse_buffer.setdefault(scene, {"pts": [], "color": [], "sample_key": []})
                    buf["pts"].append(geo.pts.detach().cpu().reshape(-1, 3))
                    buf["color"].append(geo.color.detach().cpu().reshape(-1, 3))
                    buf["sample_key"].append((frame, camera))
                else:
                    dtu_dir = join(self.result_dir, "dtu_official_ply")
                    os.makedirs(dtu_dir, exist_ok=True)
                    dtu_path = join(dtu_dir, f"{scene}.ply")
                    geo_paths.append(dtu_path)
                    geo_arrays.append(geo)

        pool = parallel_execution(geo_arrays, geo_paths, action=self.exports[type.name], async_return=True, num_workers=3)  # actual writing to disk (async)
        self.thread_pools.append(pool)
        self.limit_thread_pools()  # maybe clear some of the taskes in the thread pool
        return geo_stats

    def visualize(self, output: dotdict, batch: dotdict):
        geo_stats = dotdict()
        for type in self.types:
            geo_stats.update(self.visualize_type(output, batch, type))
        return geo_stats

    def limit_thread_pools(self):
        if len(self.thread_pools) > self.pool_limit:
            for pool in self.thread_pools[:self.pool_limit]:
                pool.close()
                pool.join()
            self.thread_pools = self.thread_pools[self.pool_limit:]

    @staticmethod
    def _deterministic_downsample_indices(num_points: int, max_points: int):
        if num_points <= max_points:
            return None
        steps = torch.arange(max_points, dtype=torch.float64)
        sel = torch.floor((steps + 0.5) * (float(num_points) / float(max_points))).to(torch.long)
        return sel.clamp_max(num_points - 1)

    def synchronize(self):
        if not (
            self.dtu_official_ply_export
            and self.dtu_official_ply_fuse
            and get_distributed()
        ):
            return

        part_dir = join(self.result_dir, "dtu_official_ply_parts")
        os.makedirs(part_dir, exist_ok=True)
        rank = get_rank()
        for scene, buf in self._dtu_fuse_buffer.items():
            part_path = join(part_dir, f"{scene}.rank{rank:02d}.pt")
            torch.save({
                "sample_key": list(buf.get("sample_key", [])),
                "pts": list(buf["pts"]),
                "color": list(buf["color"]),
            }, part_path)
            if self.verbose:
                log(f"[dtu-fuse] rank={rank} cached {scene} -> {blue(part_path)}")
        self._dtu_fuse_buffer.clear()
        torch.distributed.barrier()

    def _load_dtu_fuse_entries(self):
        scene_entries: Dict[str, Dict[Tuple[int, int], Tuple[torch.Tensor, Union[torch.Tensor, None]]]] = {}

        def append_entries(scene: str,
                           sample_keys: List[Tuple[int, int]],
                           pts_list: List[torch.Tensor],
                           color_list: List[torch.Tensor]):
            entries = scene_entries.setdefault(scene, {})
            for idx, sample_key in enumerate(sample_keys):
                key = tuple(sample_key)
                if key in entries:
                    continue
                color = color_list[idx] if idx < len(color_list) else None
                entries[key] = (pts_list[idx], color)

        for scene, buf in self._dtu_fuse_buffer.items():
            append_entries(scene, list(buf.get("sample_key", [])), list(buf["pts"]), list(buf["color"]))

        part_dir = join(self.result_dir, "dtu_official_ply_parts")
        if os.path.isdir(part_dir):
            for part_path in sorted(glob.glob(join(part_dir, "*.rank*.pt"))):
                payload = torch.load(part_path, map_location="cpu")
                scene = os.path.basename(part_path).split(".rank", 1)[0]
                append_entries(scene, list(payload.get("sample_key", [])), list(payload.get("pts", [])), list(payload.get("color", [])))

        return scene_entries

    def summarize(self):
        for pool in self.thread_pools:  # finish all pending taskes before generating videos
            pool.close()
            pool.join()
        self.thread_pools.clear()  # remove all pools for this evaluation
        if (
            self.dtu_official_ply_export
            and self.dtu_official_ply_fuse
        ):
            scene_entries = self._load_dtu_fuse_entries()
            if not scene_entries:
                return dotdict()
            dtu_dir = join(self.result_dir, "dtu_official_ply")
            os.makedirs(dtu_dir, exist_ok=True)
            for scene, entries in scene_entries.items():
                ordered = [entries[key] for key in sorted(entries.keys())]
                pts = torch.cat([item[0] for item in ordered], dim=0)
                if all(item[1] is not None for item in ordered):
                    col = torch.cat([item[1] for item in ordered], dim=0)
                else:
                    col = None
                n = pts.shape[0]
                if n > self.dtu_fuse_max_points:
                    sel = self._deterministic_downsample_indices(n, self.dtu_fuse_max_points)
                    pts = pts[sel]
                    col = col[sel] if col is not None else None
                mesh = dotdict()
                mesh.pts = pts[None]
                mesh.color = col[None] if col is not None else torch.zeros_like(mesh.pts)
                out_path = join(dtu_dir, f"{scene}.ply")
                self.exports[Visualization.POINT.name](mesh, out_path)
                if self.verbose:
                    log(f"[dtu-fuse] {scene}.ply points={pts.shape[0]} -> {blue(out_path)}")
            self._dtu_fuse_buffer.clear()
            part_dir = join(self.result_dir, "dtu_official_ply_parts")
            if os.path.isdir(part_dir):
                for part_path in glob.glob(join(part_dir, "*.rank*.pt")):
                    try:
                        os.remove(part_path)
                    except OSError:
                        pass
        return dotdict()
