from dataclasses import dataclass
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import gc
import yaml
import glob
import time
from typing import List, Literal
import sys
sys.path.append(".")
import argparse
import torch
import numpy as np
import json
from copy import deepcopy
from os.path import join, exists, basename
from collections import defaultdict
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.models.samplers.vggt_sampler import VGGTSampler

from easyvolcap.utils.math_utils import affine_inverse, affine_padding

from easyvolcap.utils.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from easyvolcap.utils.vggt.utils.geometry import average_rgb_by_knn, unproject_depth_map_to_point_map, world_coordinate_map_to_normal_map
from easyvolcap.utils.cam_utils import encode_camera_params

from utils.pose_tools import interpolate, transform_pose, evo_rpe, eval_smalltraj_ate, get_sync_timestamps, align_pose, cam_pose_eval, cal_scale
from utils.pointcloud import downsample_all_block, bilateral_filter_all_clips, bilateral_filter_one, merge_block_ply, downsample_points_batch, write_simple_ply_files, write_simple_bin_files
from utils.load_fn import load_and_preprocess_images_K, load_and_preprocess_images_K_parallel
from horizon_driving_dataset import PoseTransformer, DatasetReader
from scipy.spatial.transform import Rotation
from easyvolcap.utils.prof_utils import print_profile_stats_if_slow
from map_optimizer.map_processor import MapProcessor

from concurrent.futures import ThreadPoolExecutor
import subprocess


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s", force=True)


# workflow:
# 1. align depth scale (fixed scale, no align / sfm scale / odom scale)
# 2. replace in-block extrinsics (pandar / sfm / odom / keep-vggt)
# 3. replace intrinsics (gt / vggt)
# 4. optimize (sim3 / se3 / none)
# 5. save pointcloud
@dataclass
class ClipInferConfig:
    window_size: int = 10
    overlap_size: int = 5
    scale: float = 15.0
    fixed_scale: bool = True
    inblock_scale_mode: Literal["noscale", "pandarscale", "sfmscale", "odomscale"] = "noscale"
    no_align_depth_scale: bool = True
    inblock_extri_mode: Literal["pandar", "sfm", "odom", "vggt"] = "vggt"
    inblock_intri_mode: Literal["gt", "vggt"] = "gt"
    optimization_mode: Literal["sim3", "se3", "none"] = "se3"
    do_verify: bool = True
    eval_mode: Literal["sim3", "se3"] = "se3"
    
    test_with_gt_cam = False # whether to use gt pose as input

    def __post_init__(self):
        if not self.do_verify:
            return
        
        if self.inblock_extri_mode == "pandar":
            assert self.optimization_mode == "none"
            assert self.inblock_scale_mode in ["noscale", "pandarscale"]
        elif self.inblock_extri_mode == "sfm":
            assert self.optimization_mode in ["none"]
            assert self.inblock_scale_mode in ["noscale", "sfmscale"]
        elif self.inblock_extri_mode == "odom":
            assert self.optimization_mode in ["se3", "none"]
            assert self.inblock_scale_mode in ["noscale", "odomscale"]
        elif self.inblock_extri_mode == "vggt":
            assert self.optimization_mode in ["sim3", "se3", "none"]
        else:
            raise ValueError(f"Invalid extri_mode: {self.inblock_extri_mode}")

class ClipInfer:
    def __init__(self, 
                 config: ClipInferConfig,
                 site_path: str,
                 save_root: str, 
                 weight_path: str, 
                 config_path: str,
                 pandar_site_path: str, 
                 clip_name: str,
                 camera_names: List[str],
                 gt_type: str = "pandar",
                 mvseq: bool = False,
                 use_cam_emb: bool = False,
                 save_pointcloud: bool = True,
                 average_rgb_by_knn: bool = False,
                 low_vram: bool = False):
        self.config = config
        self.camera_names = camera_names
        
        self.save_root = save_root
        self.weight_path = weight_path
        self.config_path = config_path
        self.pandar_site_path = pandar_site_path
        self.clip_name = clip_name
        
        self.site_rawdata_path = join(site_path, "Raw_data")
        self.site_debug_path = join(site_path, "Debug")
        self.site_output_path = join(site_path, "4DLabel_output")
    
        self.window_size = config.window_size
        self.overlap_size = config.overlap_size
        self.scale = config.scale
        
        self.eval_res = {}
        self.gt_type = gt_type
        
        self.mvseq = mvseq
        self.use_cam_emb = use_cam_emb
        self.segment_pointcloud_name = "static_merged.ply"
        self.save_pointcloud = save_pointcloud
        self.average_rgb_by_knn = average_rgb_by_knn
        self.low_vram = low_vram
        
        # self.use_3ddr = config.use_3ddr
        self.test_with_gt_cam = config.test_with_gt_cam
        
        # self.cam_embed_cfg = config.cam_embed_cfg
        
        self.map_processor = MapProcessor(align_mode=config.optimization_mode)

    def load_model(self):
        
        model_cfg = {}
        if self.config_path is not None:
            with open(self.config_path, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            model_cfg = config["model_cfg"]["sampler_cfg"]
            del model_cfg["use_cam_emb"]
            del model_cfg["dinov2_ckpt"]
            model_cfg["load_pretrained"] = False
            
        model_cfg["test_with_gt_cam"] = self.test_with_gt_cam

        self.model = VGGTSampler(dinov2_ckpt=None, use_cam_emb=self.use_cam_emb, network=None, low_vram=self.low_vram, **model_cfg)

        print("loading model from ", self.weight_path)
        pt_dict = torch.load(self.weight_path, map_location=torch.device('cpu'))
        pt_dict = pt_dict["model"]
        pt_dict = {k.replace('sampler.', ''): v for k, v in pt_dict.items()}
        self.model.load_state_dict(pt_dict)
        self.model.eval()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}, load model from {self.weight_path}")
        self.model = self.model.to(device)
        self.model.eval()

    def get_pandar_pose(self):
        
        # GPS mode
        pandarpose_file = join(self.pandar_site_path, "4DLabel_output/LidarSLAM/odometry", self.clip_name, "lio_offline_100HZ_by_odom.txt")
        if exists(pandarpose_file): 
            pandar_chassis2w = np.loadtxt(pandarpose_file)
        else:   
            # packlist mode
            pandarpose_file = join(self.pandar_site_path, "4DLabel_output/LidarSLAM_3DModel_0/odometry", "lio_offline_100HZ_by_odom.txt")
            if exists(pandarpose_file): 
                pandar_chassis2w = np.loadtxt(pandarpose_file)
            else:
                logger.info(f"not find {pandarpose_file}, pls check, maybe use sfm pose as gt")
                exit()

        pandar_chassis2w = np.array(sorted(pandar_chassis2w, key=lambda x: x[0]))

        return pandar_chassis2w


    def get_clip_pose(self):
        
        visual_3d_model_jpath = join(self.site_rawdata_path, "visual_3d_model.json")
        if self.gt_type is not None:
            with open(visual_3d_model_jpath, "r") as f:
                visual_3d_model = json.load(f)

            clip2model = {}
            for modelid, modelinfo in visual_3d_model.items():
                for clipname in modelinfo["clip_names"]:
                    if not clipname in clip2model:
                        clip2model[clipname] = []
                    clip2model[clipname].append(modelid)

            clips_pose = {}
            for camera_name in self.camera_names:
                clips_pose[camera_name] = {}
                modelids = clip2model[self.clip_name]
                sfm_cam2w = []
                for modelid in modelids:
                    sfmpose_file = join(self.site_output_path, f"VisualSFM_{modelid}", "odometry_by_odom_100Hz", self.clip_name, f"{camera_name}.txt")
                    sfm_cam2w.extend(np.loadtxt(sfmpose_file).tolist())
                sfm_cam2w = np.array(sorted(sfm_cam2w, key=lambda x: x[0]))
                clips_pose[camera_name]["sfm_cam2w"] = sfm_cam2w

                odopose_file = join(self.site_debug_path, "VisualSFM", "Vision_Result", self.clip_name, camera_name, f"{camera_name}_wigo_offset.txt")
                odo_cam2w = np.loadtxt(odopose_file)
                odo_cam2w = np.array(sorted(odo_cam2w, key=lambda x: x[0]))
                clips_pose[camera_name]["odo_cam2w"] = odo_cam2w

                if self.gt_type == "pandar":
                
                    pandar_chassis2w = self.get_pandar_pose()
                    
                    # 转到各个相机坐标系
                    clip_dir = join(self.site_rawdata_path, self.clip_name)
                    dr = DatasetReader(clip_dir)
                    cam2chassis = dr.get_extrinsic(camera_name, "chassis")
                    pt = PoseTransformer()
                    pt.loadarray(pandar_chassis2w)
                    pt.right_rotate(cam2chassis)
                    clips_pose[camera_name]["pandar_cam2w"] = pt.dumparray()
                    
                elif self.gt_type == "sfm":
                    logger.info("use sfm pose as gt")
                    clips_pose[camera_name]["pandar_cam2w"] = sfm_cam2w
                else:
                    logger.info(f"not support gt type {self.gt_type}")
        else:
            logger.info(f"no sfm result, only infer vggt")
            visual_3d_model_jpath = join(self.site_rawdata_path, "sence_level_infos.json")
            with open(visual_3d_model_jpath, "r") as f:
                visual_3d_model = json.load(f)

            clip2model = {}
            for modelid, modelinfo in visual_3d_model.items():
                if len(modelinfo) == 0:
                    continue
                for clipname in modelinfo:
                    if not clipname in clip2model:
                        clip2model[clipname] = []
                    clip2model[clipname].append(modelid)

            clips_pose = {}
            for camera_name in self.camera_names:
                clips_pose[camera_name] = {}
                modelids = clip2model[self.clip_name]

                odopose_file = join(self.site_debug_path, "VisualSFM", "Vision_Result", self.clip_name, camera_name, f"{camera_name}_wigo_offset.txt")
                odo_cam2w = np.loadtxt(odopose_file)
                odo_cam2w = np.array(sorted(odo_cam2w, key=lambda x: x[0]))
                clips_pose[camera_name]["odo_cam2w"] = odo_cam2w
                clips_pose[camera_name]["sfm_cam2w"] = None
                clips_pose[camera_name]["pandar_cam2w"] = None

        return clips_pose


    def prepare_clipdata(self):
        clip_infos = {}
        clip_pose = self.get_clip_pose()
        
        for camera_name in self.camera_names:
            camera_info = {}
            image_dir = join(self.site_debug_path, "VisualSFM", "Vision_Result", self.clip_name, camera_name, "keyframe")
            ims = sorted(os.listdir(image_dir))
            ims = [join(image_dir, im) for im in ims]
            camera_info["ims"] = ims
            camera_info["sfm_cam2w"] = clip_pose[camera_name]["sfm_cam2w"]
            camera_info["odo_cam2w"] = clip_pose[camera_name]["odo_cam2w"]
            camera_info["pandar_cam2w"] = clip_pose[camera_name]["pandar_cam2w"]
            camera_param_path = join(self.site_debug_path, "VisualSFM", "Vision_Result", self.clip_name, camera_name, f"{camera_name}.params")
            clip_attribute_path = join(self.site_rawdata_path, self.clip_name, "attribute.json")
            with open(clip_attribute_path, 'r') as f:
                clip_attribute = json.load(f)
            T_camera_2_vcs = np.array(clip_attribute["calibration"][f"{camera_name}_2_chassis"])
            camera_info["cam2vcs"] = T_camera_2_vcs
            
            with open(camera_param_path, "r") as f:
                camera_param = f.read()
                camera_param = camera_param.split(" ")
                K = np.eye(3)
                K[0, 0] = camera_param[0]
                K[1, 1] = camera_param[0]
                K[0, 2] = camera_param[1]
                K[1, 2] = camera_param[2]
                camera_info["K"] = K
            clip_infos[camera_name] = camera_info
            
        return clip_infos


    def prepare_segment(self, clip_infos):
        self.segments = {}
        
        for camera_name, camera_info in clip_infos.items():
            ims = camera_info["ims"]
            sfm_cam2w = camera_info["sfm_cam2w"]
            odo_cam2w = camera_info["odo_cam2w"]
            pandar_cam2w = camera_info["pandar_cam2w"]
            K = camera_info["K"]
            self.segments[camera_name] = []
            sidx = 0
            for sidx in range(0, len(ims), self.window_size - self.overlap_size):
                eidx = min(sidx + self.window_size, len(ims))
                if eidx - sidx < 3:
                    continue
                imgs = ims[sidx:eidx]
                
                imgs = center_outward_sort(imgs)
                imgs_ts = [int(im.split("/")[-1].split(".")[0])/1000.0 for im in imgs]
                
                segs =[im.replace("keyframe", "seg").replace(".jpg", ".png") for im in imgs]
                
                segment_sfm_cam2w = interpolate(imgs_ts, sfm_cam2w)
                segment_odo_cam2w = interpolate(imgs_ts, odo_cam2w)
                segment_pandar_cam2w = interpolate(imgs_ts, pandar_cam2w)
                
                segment_pose = {}
                segment_pose["sfm_cam2w"] = segment_sfm_cam2w
                segment_pose["odo_cam2w"] = segment_odo_cam2w
                segment_pose["pandar_cam2w"] = segment_pandar_cam2w
                Ks = [K for _ in range(len(imgs))]
                segment_save_path = join(self.save_root, camera_name, f"{sidx}-{eidx}")
                self.segments[camera_name].append(
                    {"ims": imgs, 
                     "segs":segs, 
                     "imgs_ts":imgs_ts, 
                     "sidx":sidx,
                     "eidx":eidx,
                     "Ks":Ks,
                     "save_path": segment_save_path ,
                     "pose": segment_pose}) 
                
    def prepare_segment_mvseq(self, clip_infos):
        self.segments = {}

        # sync_ts
        imgs_ts_list = []
        max_len = 0
        for camera_name, camera_info in clip_infos.items():
            ims = camera_info["ims"]
            imgs_ts_list.append([int(im.split("/")[-1].split(".")[0]) for im in ims])
            max_len = max(max_len, len(ims))
        sync_imgs_ts_list, _ = get_sync_timestamps(imgs_ts_list[0], imgs_ts_list, 0, 10)
        sync_imgs_ts_list = sync_imgs_ts_list[:, 1:]
        sync_imgs_ts_list = np.swapaxes(sync_imgs_ts_list, 0, 1) 
        logger.info(f"sync_imgs_ts_list shape: {sync_imgs_ts_list.shape} of {max_len}")
        sync_imgs_ts_list = sync_imgs_ts_list.tolist()

        # sync clip infos
        sync_clip_infos = {}
        for cammera_idx, (camera_name, camera_info) in enumerate(clip_infos.items()):
            ims = camera_info["ims"]
            K = camera_info["K"]
            sync_ims = []
            for im in ims:
                im_ts = int(im.split("/")[-1].split(".")[0])
                if im_ts in sync_imgs_ts_list[cammera_idx]:
                    sync_ims.append(im)
            assert len(sync_ims) == len(sync_imgs_ts_list[cammera_idx]), f"len(sync_ims) != len(sync_imgs_ts_list[cammera_idx])"
            Ks = [K for _ in range(len(sync_ims))]
            sfm_cam2w = camera_info["sfm_cam2w"]
            odo_cam2w = camera_info["odo_cam2w"]
            pandar_cam2w = camera_info["pandar_cam2w"]
            sync_clip_infos[camera_name] = {"ims": sync_ims, 
                                            "Ks": Ks, 
                                            "sfm_cam2w": sfm_cam2w, 
                                            "odo_cam2w": odo_cam2w, 
                                            "pandar_cam2w": pandar_cam2w}
        
        # get segment idx
        length = len(sync_imgs_ts_list[0])
        segment_idx = []
        batch_idx = []
        i = 0
        while i < length:
            batch_idx.append(i)
            if i == length - 1:
                segment_idx.append(batch_idx)
                break
            if len(batch_idx) % (self.window_size) == 0:
                batch_idx_mid = batch_idx[(len(batch_idx) - 1) // 2]
                batch_idx.remove(batch_idx_mid)
                batch_idx = [batch_idx_mid] + batch_idx  # 中间作为ref
                segment_idx.append(batch_idx)
                batch_idx = []
                i -= self.overlap_size
            i += 1

        # split according to segment idx
        for batch_idx in segment_idx:
            sidx = int(np.array(batch_idx).min())
            eidx = int(np.array(batch_idx).max()) + 1
            self.segments[f"{sidx}-{eidx}"] = []
            batch_camera_names = []
            batch_imgs = []
            batch_segs = []
            batch_imgs_ts = []
            batch_Ks = []
            batch_sfm_cam2w = []
            batch_odo_cam2w = []
            batch_pandar_cam2w = []
            batch_save_path = []
            for camera_name, camera_info in sync_clip_infos.items():
                ims = camera_info["ims"]
                Ks = camera_info["Ks"]
                sfm_cam2w = camera_info["sfm_cam2w"]
                odo_cam2w = camera_info["odo_cam2w"]
                pandar_cam2w = camera_info["pandar_cam2w"]
                imgs = [ims[i] for i in batch_idx]
                imgs_ts = [int(im.split("/")[-1].split(".")[0])/1000.0 for im in imgs]
                segs =[im.replace("keyframe", "seg").replace(".jpg", ".png") for im in imgs]

                segment_sfm_cam2w = interpolate(imgs_ts, sfm_cam2w) if sfm_cam2w is not None else None
                segment_odo_cam2w = interpolate(imgs_ts, odo_cam2w)
                segment_pandar_cam2w = interpolate(imgs_ts, pandar_cam2w) if pandar_cam2w is not None else None

                batch_imgs += imgs
                batch_segs += segs
                batch_imgs_ts += imgs_ts
                batch_Ks += [Ks[i] for i in batch_idx]
                batch_sfm_cam2w.append(segment_sfm_cam2w)
                batch_odo_cam2w.append(segment_odo_cam2w)
                batch_pandar_cam2w.append(segment_pandar_cam2w)
                batch_camera_names += [camera_name] * len(imgs)
                batch_save_path += [join(self.save_root, camera_name, f"{sidx}-{eidx}")] * len(imgs)
            batch_sfm_cam2w = np.concatenate(batch_sfm_cam2w, axis=0) if batch_sfm_cam2w[0] is not None else None
            batch_odo_cam2w = np.concatenate(batch_odo_cam2w, axis=0)
            batch_pandar_cam2w = np.concatenate(batch_pandar_cam2w, axis=0) if batch_pandar_cam2w[0] is not None else None
            batch_segment_pose = {}
            batch_segment_pose["sfm_cam2w"] = batch_sfm_cam2w
            batch_segment_pose["odo_cam2w"] = batch_odo_cam2w

            batch_segment_pose["pandar_cam2w"] = batch_pandar_cam2w
            self.segments[f"{sidx}-{eidx}"].append(
                {"camera_names": batch_camera_names,
                 "Ks": batch_Ks,
                 "ims": batch_imgs,
                 "segs": batch_segs,
                 "imgs_ts": batch_imgs_ts,
                 "sidx": sidx,
                 "eidx": eidx,
                 "save_path": batch_save_path,
                 "pose": batch_segment_pose}) 


    def segment_infer(self, segment):

        imgs = segment["ims"]
        segs = segment["segs"]
        imgs_ts = segment["imgs_ts"]
        Ks = segment["Ks"]
        logger.info(f"loading images, {len(imgs)}")
        # rgb, ixts = load_and_preprocess_images_K(imgs, Ks)
        rgb, ixts = load_and_preprocess_images_K_parallel(imgs, Ks)
        logger.info(f"loading segs, {len(segs)}")
        # seg, _ = load_and_preprocess_images_K(segs, Ks, True)  # need modify segs the same as images
        seg, _ = load_and_preprocess_images_K_parallel(segs, Ks, is_seg=True)
        logger.info(f"loading images done, {len(imgs)}")
        segment["ixts"] = ixts

        N, C, H, W = rgb.shape
        batch = dotdict(meta=dotdict())
        batch.rgb = rgb.permute(0, 2, 3, 1).reshape(1, N, -1, C).cuda()  # (1, N, H*W, C)
        batch.meta.H, batch.meta.W = torch.tensor(H).unsqueeze(0), torch.tensor(W).unsqueeze(0)
        batch.scale = torch.tensor(self.scale)
        if self.use_cam_emb:
            pose = segment["pose"]
            if pose["sfm_cam2w"] is not None:
                sfm_cam2w = pose["sfm_cam2w"]
                c2ws = np.zeros((len(imgs), 4, 4))
                c2ws[:, 3, 3] = 1
                c2ws[:, :3, :3] = Rotation.from_quat(sfm_cam2w[:, 4:8]).as_matrix()
                c2ws[:, :3, 3] = sfm_cam2w[:, 1:4]
                c2ws = torch.from_numpy(c2ws).float().cuda()
                w2cs = affine_inverse(c2ws)
                c2ws = w2cs[:1] @ c2ws  # (S, 4, 4)
                c2ws[..., :3, 3:] = c2ws[..., :3, 3:] / self.scale
                w2cs = affine_inverse(c2ws)  # (N, 4, 4)
                ixts = ixts.float().cuda()
                cam = encode_camera_params(w2cs, ixts, H, W)
                cam = cam.unsqueeze(0)
                batch.cam = cam
            
            # prepar cam_3ddr
            odo_cam2w = pose["odo_cam2w"]
            c2ws = np.zeros((len(imgs), 4, 4))
            c2ws[:, 3, 3] = 1
            c2ws[:, :3, :3] = Rotation.from_quat(odo_cam2w[:, 4:8]).as_matrix()
            c2ws[:, :3, 3] = odo_cam2w[:, 1:4]
            c2ws = torch.from_numpy(c2ws).float().cuda()
            w2cs = affine_inverse(c2ws)
            c2ws = w2cs[:1] @ c2ws  # (S, 4, 4)
            c2ws[..., :3, 3:] = c2ws[..., :3, 3:] / self.scale
            w2cs = affine_inverse(c2ws)  # (N, 4, 4)
            ixts = ixts.float().cuda()
            cam = encode_camera_params(w2cs, ixts, H, W)
            cam = cam.unsqueeze(0)
            batch.cam_3ddr = cam


        res = {}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.inference_mode():
                logger.info("infering batch")
                start_time = time.time()
                self.model(batch)
                mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                logger.info(f"infer done, time: {time.time() - start_time}, mem: {mem_gb:.2f} GB")
                
                # Deal with the output
                w2c, ixt = pose_encoding_to_extri_intri(batch.output.cam_map, image_size_hw=(H, W))
                w2c, ixt = affine_padding(w2c)[0].cpu(), ixt[0].cpu()
                c2w = affine_inverse(w2c) # N, 4, 4
                ixt = ixt.numpy()
                c2w = c2w.numpy()
                rgb_map = batch.rgb.reshape(N, H, W, 3).cpu().numpy()  # (N, H, W, 3)
                # xyz_map = batch.output.xyz_map.reshape(N, H, W, 3).cpu().numpy()  # (N, H, W, 3)
                # xyz_bcd = batch.output.xyz_bcd.reshape(N, H, W, 3).cpu().numpy()  # (N, H, W, 3)
                # xyz_cnf = batch.output.xyz_cnf.reshape(N, H, W, 1).cpu().numpy()
                dpt_map = batch.output.dpt_map.reshape(N, H, W, 1).cpu().numpy()  # (N, H, W, 1)
                dpt_cnf = batch.output.dpt_cnf.reshape(N, H, W, 1).cpu().numpy()
                # save_path = join(save_root, camera_name, f"{sidx}-{eidx}")
                seg_map = seg.reshape(N, H, W, 1).cpu().numpy().astype(np.uint8)
                
                if self.use_cam_emb:
                    w2c[:, :3, 3] *= self.scale
                    c2w[:, :3, 3] *= self.scale
                    dpt_map *= self.scale
                    # xyz_map *= self.scale

                res["ixt"] = ixt
                res["w2c"] = w2c
                res["c2w"] = c2w
                res["rgb_map"] = rgb_map
                res["seg_map"] = seg_map
                res["dpt_map"] = dpt_map
                res["dpt_cnf"] = dpt_cnf
                # res["xyz_cnf"] = xyz_cnf
                # res["xyz_map"] = xyz_map
                
                # pose加到segment
                vggt_cam2w_tum = []
                # 转成ts tx ty tz qx qy qz qw
                for imgname, pose_c2w in zip(imgs_ts, c2w):
                    tvec = pose_c2w[:3, 3]
                    qvec = Rotation.from_matrix(pose_c2w[:3, :3]).as_quat()
                    vggt_cam2w_tum.append([float(i) for i in [imgname, tvec[0], tvec[1], tvec[2], qvec[0], qvec[1], qvec[2], qvec[3]]])
                segment["pose"]["vggt_cam2w"] = np.asarray(vggt_cam2w_tum)
            return segment, res


    def align_toodo(self, segment, align_choice="odo_cam2w", scale=None, vggt2tar=None):
        # align_choice: odo_cam2w, pandar_cam2w, sfm_cam2w
        
        target_pose = segment["pose"][align_choice]
        vggt_pose = segment["pose"]["vggt_cam2w"]
        
        scale, vggt2tar, vggt_pose_aligned = align_pose(target_pose, vggt_pose, scale, vggt2tar)
        
        segment["pose"]["vggt_cam2w_aligned"] = vggt_pose_aligned
        segment["scale"] = scale
        segment["vggt2tar"] = vggt2tar
        
        return segment
        
    def segment_fusion(self, segment, vggt_res):
        scale_mode = self.config.inblock_scale_mode
        intri_mode = self.config.inblock_intri_mode
        extri_mode = self.config.inblock_extri_mode
        
        # 拼点云
        pandar_cam2w = segment["pose"]["pandar_cam2w"]
        odo_cam2w = segment["pose"]["odo_cam2w"]
        sfm_cam2w = segment["pose"]["sfm_cam2w"]
        vggt_cam2w = segment["pose"]["vggt_cam2w"]
        depth_map = vggt_res["dpt_map"]
        
        # 内参
        if intri_mode == "gt":
            intrinsics = segment['ixts']
        elif intri_mode == "vggt":
            intrinsics = vggt_res["ixt"]
        else:
            raise ValueError(f"Invalid intrin_source: {intri_mode}")
        # print(f"intri_mode: {intri_mode}, intrinsics: {intrinsics.shape}")
        
        # 外参
        # extrinsics: 用于映射点云
        # segment_pose: 用于优化
        if scale_mode == "noscale":
            scale_factor = 1.0
        elif scale_mode == "pandarscale":
            scale_factor = cal_scale(pandar_cam2w, vggt_cam2w)
        elif scale_mode == "sfmscale":
            scale_factor = cal_scale(sfm_cam2w, vggt_cam2w)
        elif scale_mode == "odomscale":
            scale_factor = cal_scale(odo_cam2w, vggt_cam2w)
        else:
            raise ValueError(f"Invalid scale_mode: {scale_mode}")
        vggt_cam2w[:, 1:4] = vggt_cam2w[:, 1:4] * scale_factor
        if not self.config.no_align_depth_scale:
            depth_map *= scale_factor
        logger.info(f"scale_mode: {scale_mode}, scale_factor: {scale_factor}")
        
        if extri_mode == "vggt":
            segment_cam2w = vggt_cam2w
        elif extri_mode == "pandar":
            segment_cam2w = pandar_cam2w
        elif extri_mode == "sfm":
            segment_cam2w = sfm_cam2w
        elif extri_mode == "odom":
            segment_cam2w = odo_cam2w
        else:
            raise ValueError(f"Invalid extri_mode: {extri_mode}")
        logger.info(f"extri_mode: {extri_mode}, scale_factor: {scale_factor}")
        
        # depth_map (np.ndarray): Batch of depth maps of shape (S, H, W, 1) or (S, H, W)
        # extrinsics_cam (np.ndarray): Batch of camera extrinsic matrices of shape (S, 3, 4)
        # intrinsics_cam (np.ndarray): Batch of camera intrinsic matrices of shape (S, 3, 3)
        extrinsics_w2c = transform_pose(segment_cam2w, type="matrix") # (S, 4, 4)
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_w2c, intrinsics)
        world_normals = world_coordinate_map_to_normal_map(world_points)
        
        if self.average_rgb_by_knn:
            colors = average_rgb_by_knn(world_points, vggt_res["rgb_map"] * 255.0, k=10)
        else:
            colors = vggt_res["rgb_map"] * 255.0

        dtype = [("xyz", "float32", (3,)), ("normals", "float32", (3,)), ("rgb", "uint8", (3,)), ("seg", "uint8", (1,)), ("depth", "float32", (1,)), ("confidence", "float32", (1,))]
        vertices = np.zeros(world_points.shape[:3], dtype=dtype)
        vertices["xyz"] = world_points
        vertices["normals"] = world_normals
        vertices["rgb"] = colors
        vertices["seg"] = vggt_res["seg_map"]
        vertices["depth"] = depth_map
        vertices["confidence"] = vggt_res["dpt_cnf"]
        self.map_processor.add_submap(vertices, depth_map, vggt_res["dpt_cnf"], vggt_res["seg_map"], segment["ims"], segment_cam2w, scale_factor, segment["save_path"])
    
    def save_pose(self, segment):
        # save_pose
        sfm_cam2w = sorted(segment["pose"]["sfm_cam2w"], key=lambda x: x[0]) if segment["pose"]["sfm_cam2w"] is not None else None
        vggt_cam2w = sorted(segment["pose"]["vggt_cam2w"], key=lambda x: x[0])
        odo_cam2w = sorted(segment["pose"]["odo_cam2w"], key=lambda x: x[0])
        pandar_cam2w = sorted(segment["pose"]["pandar_cam2w"], key=lambda x: x[0]) if segment["pose"]["pandar_cam2w"] is not None else None
        sfm_w2c = transform_pose(sfm_cam2w, type="tum") if sfm_cam2w is not None else None # (S, 4, 4)
        vggt_w2c = transform_pose(vggt_cam2w, type="tum") # (S, 4, 4)
        odo_w2c = transform_pose(odo_cam2w, type="tum") # (S, 4, 4)
        pandar_w2c = transform_pose(pandar_cam2w, type="tum") if pandar_cam2w is not None else None # (S, 4, 4)
        if sfm_cam2w is not None:
            np.savetxt(join(segment["save_path"], "sfm_cam2w.txt"), sfm_cam2w)
        if vggt_cam2w is not None:
            np.savetxt(join(segment["save_path"], "vggt_cam2w.txt"), vggt_cam2w)
        if odo_cam2w is not None:
            np.savetxt(join(segment["save_path"], "odo_cam2w.txt"), odo_cam2w)
        if pandar_cam2w is not None:
            np.savetxt(join(segment["save_path"], "pandar_cam2w.txt"), pandar_cam2w)
        if sfm_w2c is not None:
            np.savetxt(join(segment["save_path"], "sfm_w2c.txt"), sfm_w2c)
        if vggt_w2c is not None:
            np.savetxt(join(segment["save_path"], "vggt_w2c.txt"), vggt_w2c)
        if odo_w2c is not None:
            np.savetxt(join(segment["save_path"], "odo_w2c.txt"), odo_w2c)
        if pandar_w2c is not None:
            np.savetxt(join(segment["save_path"], "pandar_w2c.txt"), pandar_w2c)
        
        
    def pose_eval(self, camera_name, segment):

        posefile_map = {
            "vggt": join(segment["save_path"], "vggt_cam2w.txt"),
            "wigo": join(segment["save_path"], "odo_cam2w.txt"),
            "pandar": join(segment["save_path"], "pandar_cam2w.txt"),
            "sfm": join(segment["save_path"], "sfm_cam2w.txt"),
        }
        evo_list = [
                    ("vggt", "pandar"),
                    ("wigo", "pandar"),
                    ("sfm", "pandar"),
                    ("vggt", "sfm"),
                    ("wigo", "sfm"),
                ]

        if camera_name not in self.eval_res:
            self.eval_res[camera_name] = {}
            for p1, p2 in evo_list:
                self.eval_res[camera_name][f"{p1}-{p2}"] = []
        
        segment_res = {}
        for p1, p2 in evo_list:
            rpe_trans, rpe_rot = evo_rpe(posefile_map[p2], posefile_map[p1], align_mode=self.config.eval_mode)
            ate_trans, ate_rot = eval_smalltraj_ate(posefile_map[p2], posefile_map[p1], align_mode=self.config.eval_mode)
            metrics = {"rpe_trans": rpe_trans, "rpe_rot": rpe_rot, "ate_trans": ate_trans, "ate_rot": ate_rot}
            sidx = segment["sidx"]
            eidx = segment["eidx"]
            logger.info(f"[{camera_name}] {sidx}-{eidx} {p1}-{p2} rpe_trans: {rpe_trans:.4f}, rpe_rot: {rpe_rot:.4f}, ate_trans: {ate_trans:.4f}, ate_rot: {ate_rot:.4f}")
            self.eval_res[camera_name][f"{p1}-{p2}"].append(metrics)
            segment_res[f"{p1}-{p2}"] = metrics

        segment_res["starttime"] = min(segment["imgs_ts"])
        segment_res["endtime"] = max(segment["imgs_ts"])
        with open(join(segment["save_path"], "pose_eval.json"), "w") as f:
            json.dump(segment_res, f, indent=4)


    def split_seg_res_by_idx(self, segment, res, sdix, edix):
        segment_split = {}
        for k, v in segment.items():
            if k == "pose":
                segment_split[k] = {}
                for k2, v2 in v.items():
                    segment_split[k][k2] = v2[sdix:edix] if v2 is not None else None
            elif k == "save_path":
                segment_split[k] = v[sdix]
            elif (k == "sidx") or (k == "eidx"):
                segment_split[k] = v
            else:
                segment_split[k] = v[sdix:edix]
        res_split = {}
        for k, v in res.items():
            res_split[k] = v[sdix:edix]
        return segment_split, res_split


    def save_sync_pointcloud(self, clip_infos, segment, res, time_tolerance_ms=5, conf_threshold=0.6):
        imgs_ts = np.array(segment["imgs_ts"])
        camera_names = np.array(segment["camera_names"])
        Ks = np.array(res["ixt"], dtype=object)
        seg_maps = np.array(res["seg_map"], dtype=object)
        rgb_maps = np.array(res["rgb_map"], dtype=object)
        dpt_maps = np.array(res["dpt_map"], dtype=object)
        conf_maps = np.array(res["dpt_cnf"], dtype=object)
        # --- 1. 按 camera_name 分组并按时间排序 ---
        grouped = defaultdict(list)
        for i, cam in enumerate(camera_names):
            grouped[cam].append((imgs_ts[i], i))
        # for cam in grouped:
        #     grouped[cam] = sorted(grouped[cam], key=lambda x: x[0])
        cams = sorted(grouped.keys())
        logger.info(f"Found {len(cams)} cameras: {cams}")

        # --- 2. 以第一个相机为时间基准 ---
        ref_cam = cams[0]
        ref_frames = grouped[ref_cam]

        # --- 3. 同步帧匹配 ---
        sync_indices = []  # 每项是 [idx_cam0, idx_cam1, ..., idx_camN]
        for ref_ts, ref_idx in ref_frames:
            matched_indices = [ref_idx]
            valid = True
            for cam in cams[1:]:
                cam_ts_list, cam_idx_list = zip(*grouped[cam])
                cam_ts_array = np.array(cam_ts_list)
                dt = np.abs(cam_ts_array - ref_ts)
                nearest_idx = np.argmin(dt)
                if dt[nearest_idx] <= time_tolerance_ms:
                    matched_indices.append(grouped[cam][nearest_idx][1])
                else:
                    valid = False
                    break
            if valid:
                sync_indices.append((ref_ts*1000, matched_indices))
                break
        logger.info(f"Found {len(sync_indices)} synchronized frames (within {time_tolerance_ms} ms)")

        # --- 4. 生成并保存点云 ---
        dynamic_classes = [9, 10, 11, 12, 13, 14, 15, 16, 17]

        for ts, indices in sync_indices:
            total_points = []

            for idx in indices:
                cam_name = camera_names[idx]
                intrinsic = Ks[idx]
                depth = np.squeeze(dpt_maps[idx])
                conf = np.squeeze(conf_maps[idx])
                img = rgb_maps[idx] * 255
                seg = seg_maps[idx]

                T_camera_2_vcs = np.array(clip_infos[cam_name]["cam2vcs"])

                # 内参
                fx, fy = intrinsic[0, 0], intrinsic[1, 1]
                cx, cy = intrinsic[0, 2], intrinsic[1, 2]

                H, W = depth.shape
                u, v = np.meshgrid(np.arange(W), np.arange(H))

                # --- 相机坐标点云 ---
                Z = depth
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy
                points_cam = np.stack((X, Y, Z), axis=-1)  # [H, W, 3]

                # --- 转换到 VCS 坐标系 ---
                points_vcs = points_cam @ T_camera_2_vcs[:3, :3].T + T_camera_2_vcs[:3, 3]

                # --- 拼接颜色与语义 ---
                rgb_flat = img.reshape(-1, 3)
                seg_flat = seg.reshape(-1, 1)
                points_flat = points_vcs.reshape(-1, 3)
                combined = np.concatenate([points_flat, rgb_flat, seg_flat], axis=1)

                # --- 去掉无效深度 ---
                valid_mask = (Z.flatten() > 0) & (Z.flatten() >= 2.0) & (Z.flatten() <= 60.0)
                combined_valid = combined[valid_mask]

                flat_conf = conf.flatten()
                top_k = int(flat_conf.size * conf_threshold)
                threshold_value = np.partition(flat_conf, -top_k)[-top_k]
                conf_mask = flat_conf >= threshold_value
                conf_points = combined[conf_mask]
                # --- 提取动态类 ---
                # dynamic_mask = np.isin(combined_valid[:, -1], dynamic_classes)
                # dynamic_points = combined_valid[dynamic_mask]
                total_points.append(conf_points)

            if len(total_points) == 0:
                continue

            total_points = np.concatenate(total_points, axis=0)
            logger.info(f"ts={ts}: dynamic_points num={len(total_points)}")

            # --- 保存为 PLY ---
            output_path = os.path.join(self.site_output_path, "VisualSFM/dynamic_ply_vggt", self.clip_name, f"{int(ts)}.ply")
            output_path_bin = os.path.join(self.site_output_path, "VisualSFM/dynamic_ply_vggt", self.clip_name, f"{int(ts)}.bin")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            write_simple_ply_files(total_points, output_path)
            write_simple_bin_files(total_points, output_path_bin)
            logger.info(f"finish saved to {output_path}")

    
    def clip_infer(self):
        
        clip_infos = self.prepare_clipdata()
        if not self.mvseq:
            self.prepare_segment(clip_infos)
            self.load_model()
            for camera_name, segments in self.segments.items():
                self.map_processor = MapProcessor(align_mode=self.config.optimization_mode)
                for segment in segments:
                    if not exists(segment["save_path"]):
                        os.makedirs(segment["save_path"])
                        
                    logger.info(f"start infer {camera_name} {segment['sidx']}-{segment['eidx']}")
                    segment, res = self.segment_infer(segment)
                    logger.info(f"finish infer {camera_name} {segment['sidx']}-{segment['eidx']}")
                    
                    logger.info(f"start save pose {camera_name} {segment['sidx']}-{segment['eidx']}")
                    self.save_pose(segment)
                    logger.info(f"finish save pose {camera_name} {segment['sidx']}-{segment['eidx']}")
                    
                    logger.info(f"start fusion {camera_name} {segment['sidx']}-{segment['eidx']}")
                    if self.save_pointcloud:
                        self.segment_fusion(segment, res)
                    logger.info(f"finish fusion {camera_name} {segment['sidx']}-{segment['eidx']}")
                    
                    # pose eval
                    logger.info(f"start pose eval{camera_name} {segment['sidx']}-{segment['eidx']}")
                    self.pose_eval(camera_name, segment)
                    logger.info(f"finish pose eval{camera_name} {segment['sidx']}-{segment['eidx']}")
                    
                    logger.info(f"finish {camera_name} {segment['sidx']}-{segment['eidx']}")

                cam_posefile = join(self.save_root, camera_name, "before_optimization")
                self.map_processor.save_pose(cam_posefile)
                # # clip pose eval
                trajres, debug_pose = cam_pose_eval(clip_infos, camera_name, cam_posefile)

                pose_save_path = join(self.save_root, f"{self.clip_name}_eval")
                if not exists(pose_save_path):
                    os.makedirs(pose_save_path)
                        
                for posetype, pose in debug_pose.items():
                    np.savetxt(join(pose_save_path, f"{camera_name}-{posetype}.txt"), pose)
                    
                with open(join(pose_save_path, f"{self.clip_name}-{camera_name}.json"), "w") as f:
                    json.dump(trajres, f, indent=4)
                if self.save_pointcloud:
                    self.map_processor.save_pointclouds(os.path.join(segment["save_path"], "../before_optimization"))
        else:
            self.prepare_segment_mvseq(clip_infos)
            self.load_model()
            segments_dict = {}
            for camera_name in self.camera_names:
                segments_dict[camera_name] = []
            self.map_processor = MapProcessor(align_mode=self.config.optimization_mode)
            for i, (segment_name, segment) in enumerate(self.segments.items()):
                segment = segment[0]
                for save_path in segment["save_path"]:
                    if not exists(save_path):
                        os.makedirs(save_path)
                logger.info(f"start infer {segment['sidx']}-{segment['eidx']}")
                segment, res = self.segment_infer(segment)
                logger.info(f"finish infer {segment['sidx']}-{segment['eidx']}")
                # self.save_sync_pointcloud(clip_infos, segment, res)
                # need split by camera
                logger.info(f"start save pose {segment['sidx']}-{segment['eidx']}")
                batch_len = len(segment["camera_names"]) / len(self.camera_names)
                for camera_idx, camera_name in enumerate(self.camera_names):
                    camera_sidx = int(camera_idx * batch_len)
                    camera_eidx = int((camera_idx + 1) * batch_len)
                    segment_camera, res_camera = self.split_seg_res_by_idx(segment, res, camera_sidx, camera_eidx)
                    self.save_pose(segment_camera)
                    # pose eval
                    self.pose_eval(camera_name, segment_camera)
                    logger.info(f"finish {camera_name} {segment['sidx']}-{segment['eidx']}")
                    segments_dict[camera_name].append(segment_camera)
                logger.info(f"finish save pose {segment['sidx']}-{segment['eidx']}")
                logger.info(f"start fusion {segment['sidx']}-{segment['eidx']}")
                if self.save_pointcloud:
                    self.segment_fusion(segment, res)
                logger.info(f"finish fusion {segment['sidx']}-{segment['eidx']}")
            
            # save results before optimization
            if self.save_pointcloud:
                save_path = segments_dict[self.camera_names[0]][0]["save_path"]
                save_path = os.path.dirname(os.path.dirname(save_path))
                self.map_processor.save_pointclouds(os.path.join(save_path, "before_optimization"))
            self.segments = segments_dict  #`update segments`
            cam_posefile = join(self.save_root, "before_optimization")
            self.map_processor.save_pose(cam_posefile)
            pgo_kFront = self.camera_names[0]
            exe_path = "./saturnv_eval_tools/pgo/build/pgo"
            input_args = self.save_root
            cmd = [
                exe_path,
                input_args,  # 数据目录
                *self.camera_names,  # 展开相机列表
                f"--front={pgo_kFront}",  # 前置相机
            ]
            
            print("运行命令:", " ".join(cmd))
            return_code = subprocess.call(cmd)
            
            if return_code == 0:
                print("程序执行成功")
            else:
                print(f"程序执行失败，返回码: {return_code}")
            if gt_type is not None:
                for camera_name in self.camera_names:
                    trajres, debug_pose = cam_pose_eval(clip_infos, camera_name, cam_posefile, "")
                    pose_save_path = join(self.save_root, f"{self.clip_name}_eval_7dof")
                    if not exists(pose_save_path):
                        os.makedirs(pose_save_path)
                    for posetype, pose in debug_pose.items():
                        np.savetxt(join(pose_save_path, f"{camera_name}-{posetype}.txt"), pose)
                    with open(join(pose_save_path, f"{self.clip_name}-{camera_name}.json"), "w") as f:
                        json.dump(trajres, f, indent=4)
        if gt_type is not None:
            cal_metrics(self.eval_res, self.save_root)
        logger.info(f"finish {self.clip_name}")


def center_outward_sort(ims):
    N = len(ims)
    center = N // 2
    order = []

    # 从中心开始扩散
    for offset in range(N):
        left = center - offset
        right = center + offset
        if left >= 0:
            order.append(left)
        if right < N and right != left:
            order.append(right)

    return [ims[i] for i in order]


def cal_metrics(eval_res, save_root):
    summary_list = []
    pose_type_agg = defaultdict(lambda: defaultdict(list))  # pose_type -> metric_name -> list

    for camera_name, pose_dict in eval_res.items():
        for pose_type, values in pose_dict.items():
            if not values:
                continue
            metric_names = values[0].keys()
            for metric in metric_names:
                v = np.array([v[metric] for v in values], dtype=np.float32)
                pose_type_agg[pose_type][metric].extend(v.tolist())

                summary_list.append({
                    'Camera': camera_name,
                    'PoseType': pose_type,
                    'Metric': metric,
                    'Mean': float(np.mean(v)),
                    'Median': float(np.median(v)),
                    'Sigma 68%': float(np.percentile(v, 68)),
                    'Sigma 95%': float(np.percentile(v, 95)),
                    'Sigma 99.6%': float(np.percentile(v, 99.6))
                })

    for pose_type, metric_dict in pose_type_agg.items():
        for metric, all_v in metric_dict.items():
            v = np.array(all_v, dtype=np.float32)
            summary_list.append({
                'Camera': 'ALL',
                'PoseType': pose_type,
                'Metric': metric,
                'Mean': float(np.mean(v)),
                'Median': float(np.median(v)),
                'Sigma 68%': float(np.percentile(v, 68)),
                'Sigma 95%': float(np.percentile(v, 95)),
                'Sigma 99.6%': float(np.percentile(v, 99.6))
            })

    summary_path = join(save_root, "summary.json")
    # print(json.dumps(summary_list, indent=4))
    with open(summary_path, "w") as f:
        json.dump(summary_list, f, indent=4)
    logger.info(f"save summary to {summary_path}")
    return summary_list

def auto_search_config_yaml(weight_path):
    folder_path = os.path.dirname(weight_path)
    assert "trained_model" in folder_path, "trained_model not in folder_path"
    folder_path = folder_path.replace("trained_model/", "record/")
    config_files = glob.glob(os.path.join(folder_path, "*.yaml"))
    config_files = sorted(config_files)
    if len(config_files) == 0:
        raise ValueError(f"No config file found in {folder_path}")
    if len(config_files) > 1:
        for config_file in config_files:
            print(f"config_file: {config_file}")
        return config_files[-1]
        
    else:
        print(f"use config file: {config_files[0]}")
        return config_files[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Run clip-level inference for a site")

    parser.add_argument(
        "--save_root",
        type=str,
        default="/home/users/yingfeng.cai/dev/meshx/data",
        help="Root directory to save inference results"
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        default="/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/vggt/trained_model/exp/basemodel/0927/8x8_1e-4_stage2_resumebase_No2/latest.pt",
        help="Path to the trained model weight (.pt)"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to the trained model config (.yaml)"
    )
    parser.add_argument(
        "--pandar_gtpath",
        type=str,
        default="/horizon-bucket/saturn_v_4dlabel/006_4dlabel_hde/000_static_visual/release_eval/visual_eval_gt/pandar_gt_v8.3.0_driving/pandar_20250414_173757",
        help="Path to the GT folder of pandar"
    )
    parser.add_argument(
        "--site_rootpath",
        type=str,
        default="/horizon-bucket/perception-dataprocess/4dlabel/static_visual/000_mvs_data/parking_data/20250630_mechanical_at128_eval",
        help="Path to site root folder"
    )
    parser.add_argument(
        "--site",
        type=str,
        default="DZ298/20241105_D/garage__1730775214572__0",
        help="Site folder name (e.g. Cloudy/Site_xxx)"
    )
    parser.add_argument(
        "--gt_type",
        type=str,
        default="sfm",
        help="choice: pandar,sfm"
    )
    parser.add_argument(
        "--mvseq",
        type=lambda x: x.lower() == 'true', 
        default=False,
        help="whether to use mvseq"
    )
    parser.add_argument(
        "--use_cam_emb",
        type=lambda x: x.lower() == 'true', 
        default=False,
        help="whether to use use_cam_emb"
    )
    parser.add_argument(
        '--save_pointcloud', 
        type=lambda x: x.lower() == 'true', 
        default=True,
        help="save pointcloud or not"
        )
    parser.add_argument(
        "--debug",
        type=lambda x: x.lower() == 'true',
        default=False,
        help="whether to use debugpy"
    )
    parser.add_argument(
        "--sensors",
        type=str,
        choices=["4v", "6v", "10v"],
        default="6v",
        help="choice: 4v, 6v, 10v"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sitepair = args.site.split(",")
    
    if len(sitepair) == 1:
        sitesfm = sitepair[0]
        sitepandar = sitepair[0]
    elif len(sitepair) == 2:
        sitesfm = sitepair[0]
        sitepandar = sitepair[1]
    else:
        print("site name error")
        exit()

    args.debug = 1
    if args.debug:
        # import debugpy
        # debugpy.listen(15678)

        # args.save_root = "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yukai.lin/vggt_test/1020/stage2_fixmaxdepth_epoch30_3ddrinput/parking/6v"
        args.save_root = "/home/users/yukai.lin/dev/vggt/meshx/data/6v"
        args.weight_path = "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/vggt/trained_model/exp/basemodel/0927/8x8_1e-4_stage2_resumebase_No2/latest.pt"
        args.use_cam_emb = True
        args.mvseq = True
        args.sensors = "6v"
        args.gt_type = None
        
        args.site_rootpath = "/horizon-bucket/perception-dataprocess/4dlabel/static_visual/users/qingfeng.li/visual_test/pack_list_mode/output/driving_parking_20251021_215545/"
        sitesfm = "DZ689/20241216_D/garage__1734305707311__39"
        sitepandar = sitesfm

        # args.pandar_gtpath = "/horizon-bucket/perception-dataprocess/4dlabel/static_visual/users/qingfeng.li/visual_test/pack_list_mode/output/driving_parking_20250922_160909"
        # args.site_rootpath = "/horizon-bucket/perception-dataprocess/4dlabel/static_visual/users/qingfeng.li/visual_test/pack_list_mode/output/driving_parking_20250922_160909"
        # sitesfm = "DZ035/20240814_D/garage__1723606330000__0"
        # sitepandar = "DZ035/20240814_D/garage__1723606330000__0"
        args.config_path = auto_search_config_yaml(args.weight_path)
        
        
    if args.sensors == "4v":
        camera_names = [
            "fisheye_front",
            "fisheye_left",
            "fisheye_right",
            "fisheye_rear"
        ]
    elif args.sensors == "6v":
        camera_names = [
            "camera_front", 
            "camera_front_left", 
            "camera_front_right",
            "camera_rear", 
            "camera_rear_left", 
            "camera_rear_right"
        ]
    elif args.sensors == "10v":
        camera_names = [
            "camera_front", 
            "camera_front_left", 
            "camera_front_right",
            "camera_rear", 
            "camera_rear_left", 
            "camera_rear_right",
            "fisheye_front",
            "fisheye_left",
            "fisheye_right",
            "fisheye_rear"
        ]
        
    else:
        raise ValueError(f"Invalid sensors: {args.sensors}")
            
    site_save_root = join(args.save_root, sitesfm)
    site_path = join(args.site_rootpath, sitesfm)
    pandar_site_path = join(args.pandar_gtpath, sitepandar)
    site_rawdata_path = join(site_path, "Raw_data")
    site_debug_path = join(site_path, "Debug")
    weight_path = args.weight_path
    config_path = args.config_path
    gt_type = args.gt_type
    mvseq = args.mvseq
    use_cam_emb = args.use_cam_emb
    save_pointcloud = args.save_pointcloud
    clip_names = [clipname for clipname in os.listdir(site_rawdata_path) if clipname.startswith("DZ") or clipname.startswith("202") or clipname.startswith("BSJ") or clipname.startswith("QR")]

    config = ClipInferConfig()

    if args.debug:
        clip_names = [clip_names[0]]

    for clip_name in clip_names:
        print(f"clip_name: {clip_name}")
        clip_save_root = join(site_save_root, clip_name)
        CI = ClipInfer(config, site_path, clip_save_root, weight_path, config_path, pandar_site_path, clip_name, camera_names, gt_type, mvseq, use_cam_emb, save_pointcloud,low_vram=True)
        CI.use_window = True
        CI.clip_infer()
    # downsample_all_block(clip_names, site_save_root, site_save_root, num_workers=16)
    # bilateral_filter_all_clips(clip_names, site_save_root, num_workers=4)  # block 内部 bilateral filter
    
    
    # for var in ["CI"]:
    #     if var in locals():
    #         del globals()[var]
    # gc.collect()
    

    # site_ply_path = join(site_save_root, f"{basename(site_path)}.ply")
    # sence_mode = "packlist"
    # # downsample_points_batch(sence_mode, site_save_root, site_ply_path)
    
    # max_retries = 5
    # retry_delay = 60  # 秒
    # for attempt in range(1, max_retries + 1):
    #     try:
    #         downsample_points_batch(sence_mode, site_save_root, site_ply_path)
    #         break  # 成功就跳出循环
    #     except Exception as e:
    #         logger.info(f"[Attempt {attempt}/{max_retries}] downsample_points_batch error: {e}")
    #         if attempt < max_retries:
    #             logger.info(f"Retrying after {retry_delay} seconds...")
    #             time.sleep(retry_delay)
    #         else:
    #             logger.error("All retries failed, giving up.")
    #             raise
        

    # bilateral_filter_one((site_ply_path, site_ply_path.replace(".ply", "_bf.ply"), 0, 100, 0.1, 0.1))  # 整个场景 bilateral filter
    # merge_block_ply(site_save_root, site_ply_path)
