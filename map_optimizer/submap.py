import numpy as np
import os
import cv2
import open3d as o3d
from map_optimizer.utils import SerializableMixin

class Submap(SerializableMixin):
    def __init__(self, submap_id, xyz, depth, cnf, mask_imgs, img_names, poses, scale_factor, save_path, conf_threshold=0.6, normal_threshold=45):
        self.submap_id = submap_id
        self.xyz = xyz
        # self.depth = depth
        # self.cnf = cnf
        # self.mask_imgs = mask_imgs
        self.img_names = img_names
        self.save_path = save_path
        self.scale_factor = scale_factor
        self.normal_threshold = normal_threshold
        self.global_pose = np.eye(4)
        self.aligned_poses = {}
        self.keep_mask = self.get_mask(xyz, depth, cnf, mask_imgs, conf_threshold)
        self.current_all_points_np = self.get_local_submap()
        self.poses = {}
        for pose, imgname in zip(poses, img_names):
            cam_name = imgname.split("/")[-3]
            if cam_name not in self.poses:
                self.poses[cam_name] = []
            self.poses[cam_name].append(pose)

    def filter_by_conf_mask_depth(self, depth, xyz_cnf, mask_img, conf_threshold):
        H, W = xyz_cnf.shape[:2]
        conf = xyz_cnf
        flat_conf = conf.flatten()
        top_k = int(flat_conf.size * conf_threshold)
        threshold_value = np.partition(flat_conf, -top_k)[-top_k]
        conf_mask = conf >= threshold_value
        # conf_mask = conf > 15
        # mask upper part of the image
        # conf_mask[:H//3, :] = False

        if isinstance(mask_img, str):
            mask = cv2.imread(mask_img, cv2.IMREAD_GRAYSCALE)
            mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask_img.reshape((H, W))
        
        # sem_mask = ~(((mask_resized >= 9) & (mask_resized <= 17)) | 
        #             (mask_resized == 20) | (mask_resized == 2) | (mask_resized == 38))

        sem_mask = ~((mask_resized == 20) | (mask_resized == 2) | (mask_resized == 38))

        min_depth = 0.5 / self.scale_factor
        max_depth = 20 / self.scale_factor
        
        depth_mask = (depth >= min_depth) & (depth <= max_depth)

        keep_mask = conf_mask & sem_mask & depth_mask
        return keep_mask
    
    def filter_by_normal(self, normal, angle_threshold=30):
        normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)

        def compute_mask(normal1, normal2):
            dot = np.sum(normal1 * normal2, axis=-1)
            angles = np.arccos(dot) * 180 / np.pi
            return ((angles < angle_threshold) & (angles >= 0)).astype(np.uint8)

        keep_mask = compute_mask(normal[..., 1:-1, 1:-1, :], normal[..., 1:-1, 2:, :])
        keep_mask += compute_mask(normal[..., 1:-1, 1:-1, :], normal[..., 1:-1, :-2, :])
        keep_mask += compute_mask(normal[..., 1:-1, 1:-1, :], normal[..., 2:, 1:-1, :])
        keep_mask += compute_mask(normal[..., 1:-1, 1:-1, :], normal[..., :-2, 1:-1, :])
        keep_mask = keep_mask >= 2
        keep_mask = np.pad(keep_mask, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        # print("normal masking", keep_mask.mean())
        return keep_mask

    def get_mask(self, xyz, depth, cnf, mask_imgs, conf_threshold):
        keep_mask_list = []
        for i in range(self.xyz.shape[0]):
            keep_mask = self.filter_by_conf_mask_depth(
                depth[i, :, :, 0], cnf[i, :, :, 0], mask_imgs[i], conf_threshold)
            keep_mask &= self.filter_by_normal(xyz["normals"][i], self.normal_threshold)
            keep_mask_list.append(keep_mask)
        keep_mask_np = np.stack(keep_mask_list, axis=0)
        return keep_mask_np

    def get_local_submap(self):
        current_all_points_list = []
        for i in range(self.xyz.shape[0]):
            current_mask = self.keep_mask[i]
            current_all_points = self.xyz[i][current_mask]
            current_all_points_list.append(current_all_points)
            
        current_all_points_np = np.concatenate(current_all_points_list, axis=0)
        
        return current_all_points_np

    def get_global_submap(self):
        local_xyz = self.current_all_points_np["xyz"]
        src_h = np.hstack([local_xyz, np.ones((local_xyz.shape[0], 1))])
        aligned = (self.global_pose @ src_h.T).T
        aligned /= aligned[:, 3:4]
        aligned_xyz = aligned[:, :3]

        global_cloudpoints = self.current_all_points_np.copy()
        global_cloudpoints["xyz"] = aligned_xyz
        return global_cloudpoints

    def find_overlap(self, current_submap):
        common_names = []
        prev_indices = []
        curr_indices = []

        for i, name in enumerate(current_submap.img_names):
            if name in self.img_names:
                j = self.img_names.index(name)
                common_names.append(name)
                prev_indices.append(j)
                curr_indices.append(i)
                
        if len(common_names) == 0:
            return None, None

        print(common_names)
        print(f"find overlap: {self.submap_id}->{current_submap.submap_id} {prev_indices} {curr_indices}")

        final_last_points_list = []
        final_new_points_list = []

        for i, j in zip(prev_indices, curr_indices):
            last_keep_mask = self.keep_mask[i]
            new_keep_mask = current_submap.keep_mask[j]
            final_keep_mask = last_keep_mask & new_keep_mask
            final_last_points = self.xyz[i][final_keep_mask]
            final_new_points = current_submap.xyz[j][final_keep_mask]
            final_last_points_list.append(final_last_points)
            final_new_points_list.append(final_new_points)
            
        final_last_points_np = np.concatenate(final_last_points_list, axis=0)
        final_new_points_np = np.concatenate(final_new_points_list, axis=0)

        return final_last_points_np, final_new_points_np

