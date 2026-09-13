from dataclasses import dataclass
import os
import time
from typing import List, Literal
import sys
sys.path.append(".")
import argparse
import torch
import numpy as np
from os.path import join, exists, basename
from scipy.spatial.transform import Rotation

from easyvolcap.utils.base_utils import dotdict
from easyvolcap.models.samplers.vggt_sampler import VGGTSampler
from easyvolcap.utils.math_utils import affine_inverse, affine_padding
from easyvolcap.utils.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from easyvolcap.utils.cam_utils import encode_camera_params
from utils.pose_tools import interpolate, get_sync_timestamps
from utils.load_fn import load_and_preprocess_images_K



import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s", force=True)


@dataclass
class ClipInferConfig:
    window_size: int = 10
    overlap_size: int = 5
    scale: float = 15.0
    fixed_scale: bool = True
    
    use_3ddr: bool = True
    cam_embed_cfg =dotdict()
    cam_embed_cfg.type = "CamMlp"
    cam_embed_cfg.in_features = 10
    cam_embed_cfg.hidden_features = 1024
    cam_embed_cfg.out_features = 4096


class ClipInfer:
    def __init__(self, 
                 config: ClipInferConfig,
                 save_root: str, 
                 weight_path: str, 
                 img_folder: str, 
                 odo_folder: str,
                 intri_folder: str,
                 camera_names: List[str],
                 mvseq: bool = False,
                 use_cam_emb: bool = False,
                 low_vram: bool = True):
        
        self.config = config
        self.camera_names = camera_names
        
        self.save_root = save_root
        self.weight_path = weight_path
        self.img_folder = img_folder
        self.odo_folder = odo_folder
        self.intri_folder = intri_folder

        self.window_size = config.window_size
        self.overlap_size = config.overlap_size
        self.scale = config.scale
        self.mvseq = mvseq
        self.use_cam_emb = use_cam_emb
        self.low_vram = low_vram
        
        self.use_3ddr = config.use_3ddr
        self.cam_embed_cfg = config.cam_embed_cfg
    

    def load_model(self):

        self.model = VGGTSampler(dinov2_ckpt=None, use_cam_emb=self.use_cam_emb, network=None, low_vram=self.low_vram, use_3ddr=self.use_3ddr, cam_embed_cfg=self.cam_embed_cfg)

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


    def prepare_clipdata(self):
        clip_infos = {}

        for camera_name in self.camera_names:
            camera_info = {}
            image_dir = join(self.img_folder, camera_name, "keyframe")
            ims = sorted(os.listdir(image_dir))
            ims = [join(image_dir, im) for im in ims]
            camera_info["ims"] = ims
            camera_info["odo_cam2w"] = np.loadtxt(join(self.odo_folder, camera_name, f"{camera_name}_wigo_offset.txt"))
            
            camera_param_path = join(self.intri_folder, camera_name, f"{camera_name}.params")
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
            odo_cam2w = camera_info["odo_cam2w"]
            sync_clip_infos[camera_name] = {"ims": sync_ims, 
                                            "Ks": Ks, 
                                            "odo_cam2w": odo_cam2w}

        # get segment idx
        length = len(sync_imgs_ts_list[0])
        segment_idx = []
        batch_idx = []
        i = 0
        while i < length:
            batch_idx.append(i)
            if i == length - 1:
                segment_idx.append(batch_idx)
                # print(batch_idx)
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
            batch_odo_cam2w = []
            batch_save_path = []
            for camera_name, camera_info in sync_clip_infos.items():
                ims = camera_info["ims"]
                Ks = camera_info["Ks"]
                odo_cam2w = camera_info["odo_cam2w"]
                imgs = [ims[i] for i in batch_idx]
                imgs_ts = [int(im.split("/")[-1].split(".")[0])/1000.0 for im in imgs]
                segs =[im.replace("keyframe", "seg").replace(".jpg", ".png") for im in imgs]

                segment_odo_cam2w = interpolate(imgs_ts, odo_cam2w)

                batch_imgs += imgs
                batch_segs += segs
                batch_imgs_ts += imgs_ts
                batch_Ks += [Ks[i] for i in batch_idx]
                batch_odo_cam2w.append(segment_odo_cam2w)
                batch_camera_names += [camera_name] * len(imgs)
                batch_save_path += [join(self.save_root, camera_name, f"{sidx}-{eidx}")] * len(imgs)
            batch_odo_cam2w = np.concatenate(batch_odo_cam2w, axis=0)
            batch_segment_pose = {}
            batch_segment_pose["odo_cam2w"] = batch_odo_cam2w
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


    # @print_profile_stats_if_slow("segment_infer", 5)
    def segment_infer(self, segment):

        imgs = segment["ims"]
        imgs_ts = segment["imgs_ts"]
        Ks = segment["Ks"]
        print(f"loading images, {len(imgs)}")
        rgb, ixts = load_and_preprocess_images_K(imgs, Ks)
        segment["ixts"] = ixts

        N, C, H, W = rgb.shape
        batch = dotdict(meta=dotdict())
        batch.rgb = rgb.permute(0, 2, 3, 1).reshape(1, N, -1, C).cuda()  # (1, N, H*W, C)
        batch.meta.H, batch.meta.W = torch.tensor(H).unsqueeze(0), torch.tensor(W).unsqueeze(0)
        batch.scale = torch.tensor(self.scale)
        if self.use_cam_emb:
            pose = segment["pose"]
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
        # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.inference_mode():
            print("infering batch")
            start_time = time.time()
            self.model(batch)
            mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"infer done, time: {time.time() - start_time}, mem: {mem_gb:.2f} GB")
            
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
            
            if self.use_cam_emb:
                w2c[:, :3, 3] *= self.scale
                c2w[:, :3, 3] *= self.scale
                dpt_map *= self.scale
                # xyz_map *= self.scale

            res["ixt"] = ixt
            res["w2c"] = w2c
            res["c2w"] = c2w
            res["rgb_map"] = rgb_map
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


    def clip_infer(self):
        clip_infos = self.prepare_clipdata()
        
        self.prepare_segment_mvseq(clip_infos)
        self.load_model()

        for i, (segment_name, segment) in enumerate(self.segments.items()):
            segment = segment[0]
            for save_path in segment["save_path"]:
                if not exists(save_path):
                    os.makedirs(save_path)
            segment, res = self.segment_infer(segment)
            
                    

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
        default="/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/vggt/trained_model/exp/basemodel/0828/8x8_1e-4_stage2_basemodel_3ddrratio0.7/159.pt",
        help="Path to the trained model weight (.pt)"
    )
    parser.add_argument(
        "--img_folder",
        type=str,
        default="/horizon-bucket/perception-dataprocess/4dlabel/static_visual/000_mvs_data/parking_data/20250630_mechanical_at128_eval/DZ298/20241105_D/garage__1730775214572__0/Debug/VisualSFM/Vision_Result/20241105-105334_572",
        help="Path to image folder"
    )
    parser.add_argument(
        "--odo_folder",
        type=str,
        default="/horizon-bucket/perception-dataprocess/4dlabel/static_visual/000_mvs_data/parking_data/20250630_mechanical_at128_eval/DZ298/20241105_D/garage__1730775214572__0/Debug/VisualSFM/Vision_Result/20241105-105334_572",
        help="Path to site root folder"
    )
    parser.add_argument(
        "--intri_folder",
        type=str,
        default="/horizon-bucket/perception-dataprocess/4dlabel/static_visual/000_mvs_data/parking_data/20250630_mechanical_at128_eval/DZ298/20241105_D/garage__1730775214572__0/Debug/VisualSFM/Vision_Result/20241105-105334_572",
        help="Path to site root folder"
    )
    parser.add_argument(
        "--camera_names",
        type=list,
        default=[
            "camera_front", 
            "camera_front_left", 
            "camera_front_right",
            "camera_rear", 
            "camera_rear_left", 
            "camera_rear_right"
            ]
    )
    parser.add_argument(
        "--mvseq",
        type=lambda x: x.lower() == 'true', 
        default=True,
        help="whether to use mvseq"
    )
    parser.add_argument(
        "--use_cam_emb",
        type=lambda x: x.lower() == 'true', 
        default=True,
        help="whether to use use_cam_emb"
    )
    parse_args
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    import pickle
    with open("/home/users/yingfeng.cai/dev/meshx/data/consi/batch.pkl", "rb") as f:
        batch_clipinfer = pickle.load(f)
    
    debug = False
    # debug = True
    if debug:

        args.save_root = "data/0825/evaltest/stage2_3v"
        args.weight_path = "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/vggt/trained_model/exp/basemodel/0828/8x8_1e-4_stage2_basemodel_3ddrratio0.7/159.pt"

        args.odo_folder = "/horizon-bucket/perception-dataprocess/4dlabel/static_visual/000_mvs_data/parking_data/20250630_mechanical_at128_eval/DZ298/20241105_D/garage__1730775214572__0/Debug/VisualSFM/Vision_Result/20241105-105334_572"
        args.img_folder = "/horizon-bucket/perception-dataprocess/4dlabel/static_visual/000_mvs_data/parking_data/20250630_mechanical_at128_eval/DZ298/20241105_D/garage__1730775214572__0/Debug/VisualSFM/Vision_Result/20241105-105334_572"
        args.intri_folder = "/horizon-bucket/perception-dataprocess/4dlabel/static_visual/000_mvs_data/parking_data/20250630_mechanical_at128_eval/DZ298/20241105_D/garage__1730775214572__0/Debug/VisualSFM/Vision_Result/20241105-105334_572"
        
    save_root = args.save_root
    weight_path = args.weight_path
    odo_folder = args.odo_folder
    img_folder = args.img_folder
    intri_folder = args.intri_folder
    camera_names = args.camera_names
    mvseq = args.mvseq
    use_cam_emb = args.use_cam_emb

    config = ClipInferConfig()
   
    CI = ClipInfer(config, save_root, weight_path, img_folder, odo_folder, intri_folder, camera_names, mvseq=mvseq, use_cam_emb=use_cam_emb)
    CI.clip_infer()
