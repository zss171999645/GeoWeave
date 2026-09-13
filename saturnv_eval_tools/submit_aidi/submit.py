import os
import subprocess
from datetime import datetime
import time
import yaml
import shutil

def read_config(config_file):
    try:
        with open(config_file, "r") as f:
            content = yaml.safe_load(f)
    except Exception:
        with open(config_file, "r") as f:
            content = yaml.unsafe_load(f)
    return content

class Config:
    def __init__(
        self,
        cfg_dict: dict = None,
    ):
        for k, v in cfg_dict.items():
            setattr(self, k, v)
    
    def __iter__(self):
        for k, v in self.__dict__.items():
            yield k

def generate_job_yaml(cfg):
    yaml_name = "%s.yaml" % cfg.job_name
    with open(yaml_name, "w") as fn:
        fn.write("REQUIRED:\n")
        fn.write('  JOB_NAME: "%s"\n' % cfg.job_name)
        fn.write('  JOB_PASSWD: "%s"\n' % cfg.job_password)
        fn.write('  UPLOAD_DIR: "%s"\n' % cfg.upload_folder)
        fn.write('  PROJECT_ID: "%s"\n' % cfg.project_id)
        fn.write("  WORKER_MIN_NUM: %d\n" % 1)
        fn.write("  WORKER_MAX_NUM: %d\n" % 1)
        fn.write("  GPU_PER_WORKER: %d\n" % 1)
        fn.write('  RUN_SCRIPTS: '+cfg.run_cmd+"\n")
        fn.write("OPTIONAL:\n")
        fn.write("  PRIORITY: %s\n" % cfg.priority)
        fn.write('  DOCKER_IMAGE: "%s"\n' % cfg.docker_image)
        fn.write("  WALL_TIME: %d\n" % cfg.max_jobtime)
        fn.write("  CPU_PER_WORKER: %d\n" % 8)
        fn.write("  CPU_MEM_RATIO: %d\n" % 8)
        fn.write('  JOB_TYPE: "%s"\n' % "Filter")
        # set bucket
        if hasattr(cfg, "input_bucket"):
            fn.write("  DATA_SPACE:\n")
            fn.write('    DATA_TYPE: "dmp"\n')
            if type(cfg.input_bucket) == list:
                fn.write('    INPUT: "%s"\n' % (','.join(cfg.input_bucket)))
            else:
                fn.write('    INPUT: "%s"\n' % cfg.input_bucket)
        if hasattr(cfg, "output_bucket") and cfg.output_bucket:
            if type(cfg.output_bucket) == list:
                fn.write('    OUTPUT: "%s"\n' % (','.join(cfg.output_bucket)))
            else:
                fn.write('    OUTPUT: "%s"\n' % cfg.output_bucket)
    return yaml_name


if __name__ == "__main__":
    cur_file = os.path.abspath(__file__)
    repo_fold_dir = "/".join(cur_file.split("/")[:-3])
    
    config_file = f"{repo_fold_dir}/saturnv_eval_tools/configs/evalset_old_parking.yaml"
    config = read_config(config_file)
    infer_sets = config["infer_sets"]
    cluster = config["cluster"]

    
    for infer_set in infer_sets:
    
        config_infer_set = config[infer_set]
        
        save_root = config_infer_set["save_root"]
        sites = config_infer_set["sites"]
        site_rootpath = config_infer_set["site_rootpath"]
        pandar_gtpath = config_infer_set.get("pandar_gtpath", "None")
        weight_path = config_infer_set["weight_path"]
        config_path = config_infer_set["config_path"]
        weight_path_lc = config_infer_set["weight_path_lc"]
        config_path_lc = config_infer_set["config_path_lc"]
        gt_type = config_infer_set.get("gt_type", "pandar")
        mvseq = config_infer_set.get("mvseq", False)
        use_cam_emb = config_infer_set.get("use_cam_emb", False)
        save_pointcloud = config_infer_set.get("save_pointcloud", True)
        sensors = config_infer_set.get("sensors", "fisheye_4v")
        add_loop_closure = config_infer_set.get("add_loop_closure", False)
        
        
        for site in sites:
            
            tmp_path = f"{repo_fold_dir}/aidi/.tmp/infer_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            os.makedirs(tmp_path, exist_ok=True)
            
            src1 = os.path.join(repo_fold_dir, "saturnv_eval_tools")
            dst1 = os.path.join(tmp_path, "saturnv_eval_tools")

            src2 = os.path.join(repo_fold_dir, "easyvolcap")
            dst2 = os.path.join(tmp_path, "easyvolcap")

            src3 = os.path.join(repo_fold_dir, "map_optimizer")
            dst3 = os.path.join(tmp_path, "map_optimizer")

            for src, dst in [(src1, dst1), (src2, dst2), (src3, dst3)]:
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
    
        
            run_cmd = f"sh ${{WORKING_PATH}}/saturnv_eval_tools/packlist_infer.sh " \
                    f"{save_root} " \
                    f"{weight_path} " \
                    f"{config_path} " \
                    f"{weight_path_lc} " \
                    f"{config_path_lc} " \
                    f"{pandar_gtpath} " \
                    f"{site_rootpath} " \
                    f"{site} " \
                    f"{gt_type} " \
                    f"{mvseq} " \
                    f"{use_cam_emb} " \
                    f"{save_pointcloud} " \
                    f"{sensors} " \
                    f"{add_loop_closure}"
                    
            k8s_config = dict(
                job_name=f"site_infer_{site.replace('/', '_').replace(',', '_')}",
                job_password=os.environ["AIDI_JOB_PASSWORD"],
                num_machines=2,
                project_id="GA20230001",
                input_bucket=['saturn_v_release'],
                output_bucket=["saturn_v_dev", "saturn_v_4dlabel", "perception-dataprocess"],
                priority=5,
                docker_image = "docker.hobot.cc/imagesys/4dlabel:vggt_pgo_v1",
                max_jobtime=20160,
                upload_folder=tmp_path,
                cluster=cluster,
                run_cmd=run_cmd
            )

            cfg = Config(k8s_config)
            yaml_name = generate_job_yaml(cfg)

            cmd = ["aidi-inf-cli", "job", "submit", "-f", yaml_name, "--queue_name", cfg.cluster]
            print(k8s_config)
            print(cmd)

            try:
                subprocess.check_call(cmd)
                print("Submit job successfully.")
            except Exception as e:
                print(e)
                raise Exception("error")
            finally:
                if os.path.exists(tmp_path):
                    subprocess.check_call(["rm", "-rf", tmp_path])
                if os.path.exists(yaml_name):
                    subprocess.check_call(["rm", yaml_name])

            time.sleep(0.5)