import numpy as np
# import open3d as o3d
from plyfile import PlyData, PlyElement
from map_optimizer.utils import SerializableMixin
import os

class PoseGraphOptimizer(SerializableMixin):
    def __init__(self):
        self.submaps = {}
        self.sim3_constraints = []

    def add_submap(self, submap):
        self.submaps[submap.submap_id] = submap

    def get_submap(self, submap_id):
        return self.submaps[submap_id]

    def add_sim3_constraint(self, constraint):
        self.sim3_constraints.append(constraint)

    # def save_ply(self, points, output_path):
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    #     pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6])
    #     labels = points[:, 6].astype(np.float32)
    #     pcd.point["label"] = o3d.utility.DoubleVector(labels)
    #     o3d.io.write_point_cloud(output_path, pcd)
    
    def save_ply(self,points, output_path):
        """
        points: N×7 数组, [x, y, z, r, g, b, label]
        r, g, b 范围 [0,1] 或 [0,255]
        """
        xyz = points["xyz"]
        rgb = points["rgb"]
        normals = points["normals"]
        label = points["seg"]
        if label.shape[-1] == 1:
            label = label.squeeze(-1)

        # 如果颜色是 0~1，转成 0~255 uint8
        if rgb.max() <= 1.0:
            rgb = (rgb * 255).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8)

        # 创建结构化数组
        vertex = np.empty(len(points), dtype=[
            ('x', 'f4'),
            ('y', 'f4'),
            ('z', 'f4'),
            ('nx', 'f4'),
            ('ny', 'f4'),
            ('nz', 'f4'),
            ('red', 'u1'),
            ('green', 'u1'),
            ('blue', 'u1'),
            ('alpha', 'i4')
        ])
        vertex['x'] = xyz[:, 0]
        vertex['y'] = xyz[:, 1]
        vertex['z'] = xyz[:, 2]
        vertex['nx'] = normals[:, 0]
        vertex['ny'] = normals[:, 1]
        vertex['nz'] = normals[:, 2]
        vertex['red'] = rgb[:, 0]
        vertex['green'] = rgb[:, 1]
        vertex['blue'] = rgb[:, 2]
        vertex['alpha'] = label

        # 写入 PLY
        ply_data = PlyData([PlyElement.describe(vertex, 'vertex')], text=False)
        ply_data.write(output_path)

    def save_pointclouds(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        global_cloudpoints_list = []
        for key, submap in self.submaps.items():
            global_cloudpoints = submap.get_global_submap()
            print(f"save {output_path}/{key}.ply pointcloud")
            self.save_ply(global_cloudpoints, f"{output_path}/{key}.ply")
            global_cloudpoints_list.append(global_cloudpoints)
            
        # global_cloudpoints_np = np.concatenate(global_cloudpoints_list, axis=0)
        # self.save_ply(global_cloudpoints_np, f"{output_path}/global_cloudpoints.ply")
    def save_poses(self, output_path):
        poses_list = {}
        for submapid, submap in self.submaps.items():
            if submapid == 0:
                submap.aligned_poses = submap.poses
            for cam, poses in submap.aligned_poses.items():
                if cam not in poses_list:
                    poses_list[cam] = []
                poses_list[cam].extend(poses)
                
        for cam, poses in poses_list.items():
            poses_arr = np.array(poses)
            poses_arr = poses_arr[np.argsort(poses_arr[:, 0])]
            np.savetxt(f"{output_path}/{cam}.txt", poses_arr)
            
    def optimize_pypose(self, iters=100, lr=1e-2, anchor_weight=100.0):
        import pypose as pp
        import pypose.optim.solver as ppos
        import pypose.optim.kernel as ppok
        import pypose.optim.corrector as ppoc
        import pypose.optim.strategy as ppost
        from pypose.optim.scheduler import StopOnPlateau
        import torch
        from torch import nn
        
        class PoseGraph(nn.Module):
            def __init__(self, nodes):
                super().__init__()
                self.nodes = pp.Parameter(nodes)

            def forward(self, edges, poses):
                node1 = self.nodes[edges[..., 0]]
                node2 = self.nodes[edges[..., 1]]
                error = poses.Inv() @ node1.Inv() @ node2
                return error.Log().tensor()
        
        poses = torch.stack([
            pp.mat2Sim3(submap.global_pose) for submap in self.submaps.values()
        ])
        graph = PoseGraph(poses)

        solver = ppos.Cholesky()
        strategy = ppost.TrustRegion(radius=10000)
        optimizer = pp.optim.LM(graph, solver=solver, strategy=strategy, min=1e-6, vectorize=False)
        scheduler = StopOnPlateau(optimizer, steps=iters, patience=3, decreasing=1e-3, verbose=True)

        edges = []
        meas = []
        for c in self.sim3_constraints:
            edges.append([c.submap_id1, c.submap_id2])
            meas.append(pp.mat2Sim3(c.sim3))
        edges = np.array(edges)
        meas = torch.stack(meas)
        infos = torch.ones(meas.shape[0], meas.shape[0])

        scheduler.optimize(input=(edges, meas), weight=None)

        for key, submap in self.submaps.items():
            submap.global_pose = poses[key].matrix().cpu().numpy()

    def optimize_gtsam(self):
        import gtsam
        from gtsam import (
            NonlinearFactorGraph, Values, Similarity3, Rot3, Point3,
            PriorFactorSimilarity3, BetweenFactorSimilarity3, noiseModel
        )
        from map_optimizer.utils import matrix_to_similarity3

        graph = NonlinearFactorGraph()
        initial = Values()

        # Noise model: [rot(3), trans(3), scale(1)]
        sim3_noise = noiseModel.Diagonal.Sigmas(np.array([0]*7))
        for i, submap in self.submaps.items():
            sim3 = matrix_to_similarity3(submap.global_pose)
            initial.insert(i, sim3)
            if i == 0:
                graph.add(PriorFactorSimilarity3(i, sim3, sim3_noise))

        for c in self.sim3_constraints:
            sim3_noise = noiseModel.Diagonal.Sigmas(np.array([0.01]*3 + [0.1]*3 + [0.3]))
            sim3 = matrix_to_similarity3(c.sim3)
            graph.add(BetweenFactorSimilarity3(c.submap_id1, c.submap_id2, sim3, sim3_noise))

        parameters = gtsam.LevenbergMarquardtParams()
        parameters.setMaxIterations(100)
        parameters.setVerbosityLM("SUMMARY")

        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, parameters)
        result = optimizer.optimize()
        
        for key, submap in self.submaps.items():
            submap.global_pose = result.atSimilarity3(key).matrix()

    def optimize_gtsam_pointclouds(self):
        import gtsam
        from map_optimizer.utils import matrix_to_similarity3
        from functools import partial
        from map_optimizer.sim3_pointcloud_err import Sim3PointCloudFactor
        
        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()

        # sim3_noise = noiseModel.Diagonal.Sigmas(np.array([10]*6+[0.3]*1))
        sim3_noise_zero = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6]*7))
        for i, submap in self.submaps.items():
            sim3 = matrix_to_similarity3(submap.global_pose)
            initial.insert(i, sim3)
            if i == 0:
                graph.add(gtsam.PriorFactorSimilarity3(i, sim3, sim3_noise_zero))
            # else:
            #     graph.add(gtsam.PriorFactorSimilarity3(i, sim3, sim3_noise))

        pointcloud_factor = Sim3PointCloudFactor()
        for c in self.sim3_constraints:
            noise_model = gtsam.noiseModel.Isotropic.Sigma(len(c.overlap1)*3, 10)
            gf = gtsam.CustomFactor(noise_model, [c.submap_id1, c.submap_id2], partial(pointcloud_factor.error_func, c.overlap1, c.overlap2))
            graph.add(gf)

        parameters = gtsam.LevenbergMarquardtParams()
        parameters.setMaxIterations(100)
        parameters.setVerbosityLM("SUMMARY")

        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, parameters)
        result = optimizer.optimize()

        for key, submap in self.submaps.items():
            submap.global_pose = result.atSimilarity3(key).matrix()


if __name__ == "__main__":
    optimizer = PoseGraphOptimizer()
    exp_name = "data_0707_overlap10_2_loops"
    root_path = f"/home/users/mengqi.wu/work/reconstruction/meshx/{exp_name}/20240525-173358_334/camera_rear"
    pickle_path = f"{root_path}/optimizer.pickle"

    optimizer.load(pickle_path)
    print("pose before optimization")
    for key, submap in optimizer.submaps.items():
        print(submap.global_pose)

    optimizer.optimize_pypose()
    optimizer.save_pointclouds(f"{root_path}/optimized_result_by_pypose")
    print("pose optimized by pypose")
    for key, submap in optimizer.submaps.items():
        print(submap.global_pose)

    optimizer.load(pickle_path)

    print("pose optimized by gtsam")
    optimizer.optimize_gtsam()
    optimizer.save_pointclouds(f"{root_path}/optimized_result_by_gtsam_small_rot_large_scale")
    for key, submap in optimizer.submaps.items():
        print(submap.global_pose)
