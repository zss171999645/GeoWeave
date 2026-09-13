import os
from datetime import datetime
import yaml
import subprocess
from os.path import dirname, abspath, join

CLUSTER = "project-intel-l20-4dlabel-release2-tcloud"
# DEFAULT_WEIGHT = "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/vggt/trained_model/exp/basemodel/extrapool/8x4_2e-5_stage2_resumeno3_chunkdptbp_src63_extrapool31"
# WEIGHT_CONFIG = "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/vggt/finetune_model/basemodel/extrapool/8x4_2e-5_stage2_resumeno3_chunkdptbp_src63_extrapool31/record/1763925201.yaml"
DEFAULT_WEIGHT = "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/vggt/finetune_model/smallmodel/1025/L20_8x8_1e-4_vits_woxyzhead_bz24x2_80k_resumefromscratch"
WEIGHT_CONFIG = "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/vggt/finetune_model/smallmodel/1025/L20_8x8_1e-4_vits_woxyzhead_bz24x2_80k_resumefromscratch/record/1761403247.yaml"
EXPNAME = "basemodel/extrapool/8x4_2e-5_stage2_resumeno3_chunkdptbp_src63_extrapool31_testpyv2"

DEFAULT_EPOCH = -1
DOWNSAMPLE_RATIO = 10

TASKS = {
    "driving": {
        "config": "vggt/saturnv/evaluation/vggt_eval_driving",
        "expname": f"test/driving/{EXPNAME}/6v_3ddrinput",
        "jobname": "test_driving6v_3ddrinput",
    },
    "parking": {
        "config": "vggt/saturnv/evaluation/vggt_eval_parking",
        "expname": f"test/parking/{EXPNAME}/6v_3ddrinput",
        "jobname": "test_parking6v_3ddrinput",
    },
    "machanical4v": {
        "config": "vggt/saturnv/evaluation/vggt_eval_machanical_fisheye",
        "expname": f"test/machnical/{EXPNAME}/4v_3ddrinput",
        "jobname": "test_machnical4v_3ddrinput",
    },
    "driving_sfminput": {
        "config": "vggt/saturnv/evaluation/vggt_eval_driving",
        "expname": f"test/driving/{EXPNAME}/6v_sfminput",
        "jobname": "test_driving6v_sfminput",
        "test_with_gt_cam": True
    },
    "parking_sfminput": {
        "config": "vggt/saturnv/evaluation/vggt_eval_parking",
        "expname": f"test/parking/{EXPNAME}/6v_sfminput",
        "jobname": "test_parking6v_sfminput",
        "test_with_gt_cam": True
    },
    "machanical4v_sfminput": {
        "config": "vggt/saturnv/evaluation/vggt_eval_machanical_fisheye",
        "expname": f"test/driving/{EXPNAME}/6v_sfminput",
        "jobname": "test_machnical4v_sfminput",
        "test_with_gt_cam": True
    }
}


def submit():
    for task_name, task in TASKS.items():
        eval_config_path = task["config"]
        expname = task["expname"]
        jobname = task["jobname"]
        epoch = task.get("epoch", DEFAULT_EPOCH)
        weight = task.get("weight", DEFAULT_WEIGHT)
        test_with_gt_cam = task.get("test_with_gt_cam", False)
        downsample_ratio = task.get("downsample_ratio", DOWNSAMPLE_RATIO)
        
        with open(WEIGHT_CONFIG, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            

        repodir = dirname(dirname(abspath(__file__)))
        with open(join(repodir, "configs", "exps", eval_config_path+".yaml"), "r") as f:
            eval_config = yaml.load(f, Loader=yaml.FullLoader)
        eval_config["model_cfg"]["sampler_cfg"] = config["model_cfg"]["sampler_cfg"]
        
        eval_config["val_dataloader_cfg"]["sampler_cfg"]["frame_sample"] = [0, None, downsample_ratio]

        curtime = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_config_tmp_path = f"{eval_config_path}_{curtime}_tmp"
        with open(join(repodir, "configs", "exps", eval_config_tmp_path+".yaml"), "w") as f:
            yaml.dump(eval_config, f)
    
        cmd = f'''
            python aidi/submit.py \
            --job_name {jobname} \
            --cluster_name {CLUSTER} \
            --gpu 8 \
            --node 1 \
            --priority 5 \
            --trial 1 \
            "aidi/run.sh \
                --config {eval_config_tmp_path} \
                --exp_name {expname} \
                --max_retry 0 \
                --test \
                --dist \
                --dist_args \\"--rdzv_endpoint=127.0.0.1:29501 --rdzv_backend=c10d --nproc_per_node=auto\\" \
                --test_args \\" \
                runner_cfg.resume=True \
                runner_cfg.test_use_amp=True \
                runner_cfg.load_epoch={epoch} \
                runner_cfg.trained_model={weight} \
                model_cfg.sampler_cfg.use_checkpoint=True \
                model_cfg.sampler_cfg.use_cam_emb=True \
                model_cfg.sampler_cfg.test_with_gt_cam={test_with_gt_cam} \
                model_cfg.sampler_cfg.load_pretrained=False\\""
            '''

        print("============ FINAL COMMAND ============")
        print(cmd)

        subprocess.run(cmd, shell=True)
        
        os.remove(join(repodir, "configs", "exps", eval_config_tmp_path+".yaml"))


if __name__ == "__main__":
    submit()