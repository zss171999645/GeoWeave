from typing import Literal
from map_optimizer.submap import Submap
from map_optimizer.optimizer import PoseGraphOptimizer
from map_optimizer.point_cloud_aligner import PointCloudAligner
from map_optimizer.constraints import Sim3Constraint
import numpy as np
import os
from saturnv_eval_tools.utils.pose_tools import T442tum, tum2T44

class MapProcessor:
    def __init__(self, align_mode: Literal["sim3", "se3", "none"] = "none"):
        self.submap_id = 0
        self.optimizer = PoseGraphOptimizer()
        self.align_mode = align_mode
        
    def add_constraints(self, pre_submap, current_submap, update_pose = True):
        final_last_points_np, final_new_points_np = pre_submap.find_overlap(current_submap)
        
        if final_last_points_np is None or final_new_points_np is None:
            # raise ValueError("No overlap found")
            # T_current_to_pre = np.eye(4)
            return
        else:
            pointcloud_aligner = PointCloudAligner(final_last_points_np, final_new_points_np)
            if self.align_mode == "sim3":
                T_current_to_pre = pointcloud_aligner.align_sim3()
                print("align sim3", T_current_to_pre)
            elif self.align_mode == "se3":
                T_current_to_pre = pointcloud_aligner.align_se3()
                print("align se3", T_current_to_pre)
            else:
                T_current_to_pre = np.eye(4)
                print("align with identity matrix")

        if self.align_mode != "none":
            self.optimizer.add_sim3_constraint(
                Sim3Constraint(pre_submap.submap_id, current_submap.submap_id, T_current_to_pre, final_last_points_np, final_new_points_np)
            )
        
        if update_pose:
            current_pose = pre_submap.global_pose @ T_current_to_pre # cursub2world = pre2world @ cur2pre
            current_submap.global_pose = current_pose
            print("set current submap global pose", current_pose)

            for cam, poses in current_submap.poses.items():
                current_submap.aligned_poses[cam] = []
                for pose in poses:
                    pose44, ts = tum2T44(pose) # se3
                    current_submap.aligned_poses[cam].append(T442tum(current_pose @ pose44, ts))
                current_submap.aligned_poses[cam] = np.array(current_submap.aligned_poses[cam])

    def add_submap(self, xyz, depth, cnf, mask_imgs, img_names, poses, scale_factor, save_path):
        current_submap = Submap(self.submap_id, xyz, depth, cnf, mask_imgs, img_names, poses, scale_factor, save_path)
        self.optimizer.add_submap(current_submap)

        if self.submap_id > 0:
            pre_submap = self.optimizer.get_submap(self.submap_id - 1)
            self.add_constraints(pre_submap, current_submap)

        self.submap_id += 1

    def add_loop_closure(self, pre_submap_id, current_submap_id):
        pre_submap = self.optimizer.get_submap(pre_submap_id)
        current_submap = self.optimizer.get_submap(current_submap_id)
        self.add_constraints(pre_submap, current_submap, update_pose = False)

    def run_optimization(self, with_gtsam = False):
        if with_gtsam:
            self.optimizer.optimize_gtsam()
        else:
            self.optimizer.optimize_pypose()

    def save_pointclouds(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        self.optimizer.save_pointclouds(output_path)
    
    def save_pose(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        self.optimizer.save_poses(output_path)