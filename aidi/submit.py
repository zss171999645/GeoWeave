"""
Usage:
    python3 aidi/submit.py --query
    python3 aidi/submit.py --job_mode train_b48 --cluster_name a800-2
    python3 aidi/submit.py --job_mode train_b24 --cluster_name a800-2
    python3 aidi/submit.py --job_mode train_pretrained --cluster_name a800-2
    python3 aidi/submit.py --job_mode train_resume_8x8_b24 --cluster_name a800-2 -n 16 --trial 5
    python3 aidi/submit.py --cluster_name a800-2 --job_name b24_ddp_ga2_mix5_xiaoyang_nocentercrop -n 16 -- ./aidi/run.sh --dist --config vggt/vggt_resume --exp_name vggt/reproduce/a800_bcloud/torchrun/8x8_b24_ddp_ga2_mix5_xiaoyang_nocentercrop --train_args \"  \"
    python3 aidi/submit.py --cluster_name a800 --job_name b24_ddp_ga2_mix5_resume -n 64 -- ./aidi/run.sh --dist --config vggt/vggt_resume --exp_name vggt/reproduce/a800_bcloud/torchrun/8x8_b24_ddp_ga2_mix5_resume

resume using v2 config:
    python3 aidi/submit.py --cluster_name a800 --job_name b24_ddp_ga2_mix5_resume -n 64 -- ./aidi/run.sh --dist --config vggt/experimental/vggt_v2_resume --exp_name vggt/reproduce/a800_bcloud/torchrun/8x8_b24_ddp_ga2_mix5_resume
due to slow bucket speed, we recomment using /job_tboard to monitor the job, pls set SAVE_TO_AIDI=1 to save the tensorboard to /job_tboard
    python3 aidi/submit.py --cluster_name 4090 --job_name test_exiting -n 16 -- SAVE_TO_AIDI=1 ./aidi/run.sh --dist --config vggt/experimental/vggt_v2_resume --exp_name vggt/reproduce/a800_bcloud/torchrun/8x8_b24_ddp_ga2_mix5_resume    
"""

import os
import subprocess
import argparse
from time import sleep
import yaml
import shutil
import re

from datetime import datetime


user = os.getenv("USER")

if not user:
    raise ValueError("cannot find user from environment variable")
else:
    print(f"using user: {user}")

# Define cluster names and job commands
CLUSTER_NAMES = {
    "a800": "project-a800-4dlabel-perception-bcloud",
    "a800-2": "project-a800-4dlabel-perception2-bcloud",
    "a800-share": "share-a800-small-bcloud",
    "l20-tcloud": "project-l20-4dlabel-perception-tcloud",
    "h20-model": "project-h20-saturnv-model-vcloud",
    "location": "project-l20-perception-location-tcloud",
    "5090": "project-5090-4dlabel-perception-acloud-langfang",
    "l20-release2": "project-intel-l20-4dlabel-release2-tcloud",
    "3090-release": "project-3090-4dlabel-release2-bcloud-bj",
    "4090-release": "project-4090-4dlabel-release-bcloud",
    "5090-release": "project-5090-4dlabel-release-bcloud-bj",
    "5090-release2": "project-5090-4dlabel-release2-bcloud-bj:v2",
}

RUN_COMMANDS = {
    "argv": "--- use sys.argv[1:] --- ",
    "train_b24": f"./aidi/run.sh --dist --config vggt/vggt --exp_name vggt/b24 --train_args \" \" ",
    "train_b48": f"./aidi/run.sh --dist --config vggt/vggt_batch48 --exp_name vggt/b48 --train_args \" \"",
    "train_pretrained": f"./aidi/run.sh --dist --config vggt/vggt_pretrained --exp_name vggt/b24_pt --train_args \" \" ",
    "train_resume_8x8_b24": f"./aidi/run.sh --dist --config vggt/vggt_resume --exp_name vggt/reproduce/a800_bcloud/torchrun/8x8_b24_ddp_ga2_mix5_xiaoyang --train_args \" \" ",
}

DOCKER_IMAGE = "docker.hobot.cc/imagesys/base:centos7.6-gcc11.4-py3.11-cu12.4-rdma-torch2.6.0-fa3-spattn"
DOCKER_IMAGE_5090 = "docker.hobot.cc/imagesys/4dlabel:ubuntu2204-gcc11.4-cu128-nccl2277-torch271-erd-vggt-aidisdk"
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXCLUDE_PATH = [
    ".git",
    "aidi_uploads",
    ".tmp",
    "tmp",
    ".venv",
    ".venvs",
    ".worktrees",
    ".autonomous",
    ".research",
    ".codex",
    ".tb_cache",
    ".DS_Store",
    "docker",
    # Anchor the repository-level data directory only. A bare "data" pattern
    # would also exclude config groups such as pi3_training/configs/data.
    "/data",
    # Keep this anchored: a bare "core*" also filters Pi3 Hydra configs such as
    # aidi/third_party/pi3_training/configs/extras/core4_main_val.yaml.
    "/core*",
    "aidi/.tmp",
    "debug_data_loading*",
    "logs",
    "tensorboard_logs",
    "experiment_records",
    "outputs",
    "weights",
    "pytorch3d.zip",
]

date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
TMP_DIR = os.path.join(os.path.dirname(__file__), f".tmp/upload_{date_str}_{os.getpid()}")



def run_command(command, show_output=False):
    try:
        print("call", ' '.join(command) if isinstance(command, list) else command)
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE if not show_output else None,
            stderr=subprocess.PIPE if not show_output else None,
            text=True if not show_output else None,       # 以字符串形式返回输出
            check=True,       # 如果命令返回非零退出状态，将引发CalledProcessError
            shell=True if isinstance(command, str) else False
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Failed to run command: {e.stderr}")
        return None


def parse_table_with_regex(output):
    """
    使用正则表达式解析表格输出，并返回包含字典的列表。
    
    :param output: 命令输出的表格字符串
    :return: 字典列表，每个字典对应表格中的一行数据
    """
    if not output:
        return []
    lines = output.strip().splitlines()
    data = []
    headers = []
    
    # 正则表达式匹配以 '|' 开头和结尾的行，并捕获中间的字段
    pattern = re.compile(r'^\|(.+)\|$')
    
    for line in lines:
        match = pattern.match(line)
        if not match:
            continue  # 跳过不符合的行（如分隔线）
        
        # 使用 re.split 分割字段，并去除多余的空格
        fields = [field.strip() for field in match.group(1).split('|')]
        
        if not headers:
            headers = fields  # 第一行符合的是表头
            continue
        
        if len(fields) != len(headers):
            print("字段数与标题数不匹配，跳过该行")
            continue
        
        # 创建字典，键为表头，值为对应字段
        entry = dict(zip(headers, fields))
        data.append(entry)
    
    return data


def query_gpu_queue():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("`--query` 依赖 pandas，请先安装后重试") from exc

    entries = []
    for tag, cluster_name in CLUSTER_NAMES.items():
        cmd = [
            "aidi-inf-cli",
            "job",
            "quota",
            "--queue_name",
            cluster_name.replace(":v2", ""),
        ]
        output = run_command(cmd)
        data = parse_table_with_regex(output)
        for entry in data:
            entry["cluster_name"] = tag
            entry["is_free"] = int(entry["QUEUING JOB"]) == 0 and int(entry["FREE QUOTA"]) > 0
            entry["is_free"] = str(entry["is_free"]) + (" (release)" if "release" in cluster_name else "")
        entries.extend(data)
    df = pd.DataFrame(entries)
    print(df)
    # sorted by int(QUEUING JOB)
    df = df.sort_values(by="QUEUING JOB", key=lambda x: x.astype(int), ascending=True)
    return entries


def main():
    parser = argparse.ArgumentParser(description="Simplified job submission script")
    parser.add_argument('--job_name', type=str, default='_', help="Job name")
    parser.add_argument('--job_type', type=str, default='train', help="Job type")
    parser.add_argument('--docker_image', type=str, default=DOCKER_IMAGE, help="Docker image")
    parser.add_argument('--cluster_name', type=str, default="a800", help=f"Specify cluster name or use short names {list(CLUSTER_NAMES.keys())}")
    parser.add_argument('--job_mode', type=str, default="")
    parser.add_argument('--cpu', type=int, default=2, help="Number of CPUs per machine")
    parser.add_argument(
        '--cpu_mem_ratio',
        type=int,
        default=int(os.environ.get("AIDI_CPU_MEM_RATIO", "4")),
        help="CPU memory ratio per worker. Defaults to AIDI_CPU_MEM_RATIO or 4.",
    )
    parser.add_argument('--gpu', type=int, default=8, help="Number of GPUs per machine")
    parser.add_argument('--node', type=int, default=2, help="Number of machines")
    parser.add_argument('-n', '--total_gpu', type=int, default=None, help="Number of total GPUs")
    parser.add_argument('--port', type=str, default='29500', help="Port")
    parser.add_argument('--sleep', action='store_true', help="Use sleep command for on-machine debug")
    parser.add_argument('--local', action='store_true', help="Use local machine to run the command")
    parser.add_argument('--debug', action='store_true', help="Use debug cluster for debug")
    parser.add_argument('--dry_run', action='store_true', help="Dry run")
    parser.add_argument('--trial', type=int, default=5, help="Number of trials to restart the program if failed")
    parser.add_argument('--priority', type=int, default=None, help="Priority [None, 1, 2, 3, 4, 5]", choices=[None, 1, 2, 3, 4, 5])
    parser.add_argument('cmd', nargs='*', help='Command and its arguments')
    parser.add_argument(
        "--query", 
        action="store_true",
        help="Query job status",
    )

    args = parser.parse_args()

    # Query gpu queue
    if args.query:
        query_gpu_queue()
        return

    # Get cluster name
    docker_image = args.docker_image if "/" in args.docker_image else f"docker.hobot.cc/imagesys/sysimage/{args.docker_image}"
    cluster_name = args.cluster_name if args.cluster_name not in CLUSTER_NAMES else CLUSTER_NAMES[args.cluster_name]
    is_v2 = cluster_name.endswith(":v2")
    cluster_name = cluster_name.replace(":v2", "")

    # Check if the cluster name is `share-h20-small-tcloud`
    if 'h20' in cluster_name and 'tcloud' in cluster_name:
        docker_image = "docker.hobot.cc/imagesys/base:centos7.6-gcc11.4-py3.11-cu12.4-rdma-torch2.6.0-fa3-tcloud"
    if "5090" in cluster_name:
        docker_image = DOCKER_IMAGE_5090

    # Check if the total gpu is specified
    if args.total_gpu is not None:
        assert args.total_gpu % 8 == 0 or args.total_gpu < 8
        args.gpu = min(8, args.total_gpu)
        args.node = max(1, args.total_gpu // args.gpu)
        print("total gpu:", args.total_gpu, "node:", args.node, "gpu:", args.gpu)

    cpu_num = args.cpu
    cpu_mem_ratio = args.cpu_mem_ratio
    gpu_num = args.gpu
    node_num = args.node
    port = args.port
    max_jobtime = 20160  # 20160 minutes = 14 days

    # If the job mode is not specified, read the command from argv
    if args.job_mode is None or args.job_mode == "":
        print("reading cmd from argv")
        if len(args.cmd) <= 0:
            raise ValueError("pls specify command to be executed")
        run_cmd = ' '.join(args.cmd)
    else:
        run_cmd = RUN_COMMANDS[args.job_mode]
    
    print("run_cmd:", run_cmd)
    sleep(2)

    # If the local flag is set, run the command on local machine
    if args.local:
        run_command(run_cmd, show_output=True)
        return

    # Debug-related configs
    if args.debug:  # test before submitting
        cluster_name = CLUSTER_NAMES["debug"]
        gpu_num = min(gpu_num, 4)
        node_num = min(node_num, 2)
        max_jobtime = min(max_jobtime, 120)
        args.job_mode = "CrashTest_" + args.job_mode
    if args.sleep:
        try:
            hours = int(os.environ.get("AIDI_SLEEP_HOURS", "10"))
        except ValueError:
            hours = 10
        if hours <= 0:
            hours = 10
        run_cmd = f"sleep {3600 * hours} "
        max_jobtime = min(max_jobtime, 60 * hours)
        args.job_mode = "On-cluster-debug_" + args.job_mode

    if node_num <= 1 and args.gpu < 8:
        # Cluster code package commonly extracts as ${WORKING_PATH}/<repo_name>.
        # Probe and jump to real repo root so relative paths like aidi/run.sh work.
        repo_probe = (
            "if [ ! -f aidi/submit.py ]; then "
            "FOUND=$(find . -maxdepth 4 -type f -path '*/aidi/submit.py' -print -quit); "
            "if [ -n \"$FOUND\" ]; then "
            "REPO_ROOT=$(dirname \"$(dirname \"$FOUND\")\"); "
            "cd \"$REPO_ROOT\"; "
            "fi; "
            "fi; "
        )
        run_cmd = f"echo ${{WORKING_PATH}} && cd ${{WORKING_PATH}} && {repo_probe}NUM_NODES=1 GPU_PER_NODE={args.gpu} NODE_RANK=0 HOST_NODE_ADDR=127.0.0.1 MASTER_PORT={port} USER={user} {run_cmd}"
    else:
        # use ssh_launcher.py to run the command on multiple nodes, just like how mpirun works
        repo_probe = (
            "if [ ! -f aidi/ssh_launcher.py ]; then "
            "FOUND=$(find . -maxdepth 4 -type f -path '*/aidi/ssh_launcher.py' -print -quit); "
            "if [ -n \"$FOUND\" ]; then "
            "REPO_ROOT=$(dirname \"$(dirname \"$FOUND\")\"); "
            "cd \"$REPO_ROOT\"; "
            "fi; "
            "fi; "
        )
        # Use "--" to terminate ssh_launcher option parsing, so downstream command
        # tokens like "--config/--train_args" are treated as payload, not launcher args.
        run_cmd = f"echo ${{WORKING_PATH}} && cd ${{WORKING_PATH}} && {repo_probe}python3 aidi/ssh_launcher.py -n {node_num * gpu_num} -g {gpu_num} -p {port} -t {args.trial} -- USER={user} {run_cmd}"

    # Configure parameters
    if args.job_mode:
        args.job_name = f"{args.job_mode.upper()}_{args.job_name}"
    if args.cluster_name:
        gpu_names = [name for name in ["l20", "4090", "3090", "h20", "a800", "cpu", "5090"] if name in cluster_name]
        assert len(gpu_names) == 1, f"cluster name {cluster_name} not recognized"
        total_gpu = gpu_num * node_num
        args.job_name = f"{gpu_names[0]}x{total_gpu}_{args.job_name}"
    print("job name:", args.job_name)
    # input("Press Enter to continue...")
    config = {
        "job_name": f"{args.job_name}",
        "job_type": f"{args.job_type}",
        "job_password": "password1234",
        "num_machines": node_num,
        "num_cpus": cpu_num,
        "cpu_mem_ratio": cpu_mem_ratio,
        "num_gpus": gpu_num,
        "project_id": "GA20230001",
        # "input_bucket": ['saturn_v_release', "saturn_v_dev", "saturn_v_4dlabel"],
        # "output_bucket": ["saturn_v_4dlabel"],
        "input_bucket": ['saturn_v_release', "saturn_v_dev", "saturn_v_4dlabel", "perception-dataprocess", "perception-train"],
        "output_bucket": ["saturn_v_dev", "saturn_v_4dlabel", "perception-dataprocess"],
        "priority": args.priority if args.priority is not None else (4 if "project" in cluster_name else 5),
        "docker_image": docker_image,
        "max_jobtime": max_jobtime,
        "upload_folder_name": TMP_DIR,
        "cluster": cluster_name,
        "run_cmd": run_cmd,
    }

    print("================================")
    print(f"CPU_NUM={cpu_num}")
    print(f"CPU_MEM_RATIO={cpu_mem_ratio}")
    print(f"GPU_NUM={gpu_num}")
    print(f"NODE_NUM={node_num}")
    print(f"CLUSTER={config['cluster']}")
    print(f"RUN={run_cmd}")
    print("================================")

    # Job yaml
    copy_code_to_target_dir(PROJECT_DIR, config["upload_folder_name"])
    yaml_file = os.path.join(config["upload_folder_name"], "job.yaml")
    print("yaml:", yaml_file)
    generate_yaml(config, run_cmd, yaml_file)

    # Submit
    if not is_v2:
        execute(config["cluster"], yaml_file, dry_run=args.dry_run)
    else:
        execute_v2(config["cluster"], config, dry_run=args.dry_run)


def copy_code_to_target_dir(code_dir, target_dir):
    # Sync folders
    if os.path.exists(target_dir):
        y_or_n = input(f"delete {target_dir}? y or n? ")
        if y_or_n.lower() == "y":
            shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    repo_name = os.path.basename(os.path.normpath(code_dir))
    repo_target_dir = os.path.join(target_dir, repo_name)
    os.makedirs(repo_target_dir, exist_ok=True)
    exclude_args = [f"--exclude={p}" for p in EXCLUDE_PATH]
    cmd = ["rsync", *exclude_args, "-aL", os.path.join(code_dir, ""), repo_target_dir]
    print("call:", ' '.join(cmd))
    subprocess.check_call(cmd)
    print(f"Copied {code_dir} to {target_dir}")
    # get all file size of target_dir, if > 100MB, raise error
    print(f"checking folder size...")
    folder_size = 0
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            folder_size += os.path.getsize(os.path.join(root, file))
    folder_size = folder_size / 1024 / 1024
    if folder_size > 100:
        subprocess.check_call(["du", "-sh", target_dir])
        raise ValueError(f"folder size {folder_size} MB is too large, pls delete some files")


def generate_yaml(config, run_cmd, yaml_file):
    # Generate YAML data
    yaml_data = {
        "REQUIRED": {
            "JOB_NAME": config["job_name"],
            "JOB_PASSWD": config["job_password"],
            "UPLOAD_DIR": config["upload_folder_name"],
            "PROJECT_ID": config["project_id"],
            "WORKER_MIN_NUM": config["num_machines"],
            "WORKER_MAX_NUM": config["num_machines"],
            "GPU_PER_WORKER": config["num_gpus"],
            "RUN_SCRIPTS": run_cmd,
        },
        "OPTIONAL": {
            "PRIORITY": config["priority"],
            "DOCKER_IMAGE": config["docker_image"],
            "WALL_TIME": config["max_jobtime"],
            "CPU_PER_WORKER": config["num_cpus"],
            "CPU_MEM_RATIO": config["cpu_mem_ratio"],
            "DATA_SPACE": {
                "DATA_TYPE": "dmp",
                "INPUT": ",".join(config["input_bucket"]),
                "OUTPUT": ",".join(config["output_bucket"]),
            },
            "FRAMEWORK": "pytorch",
            "JOB_TYPE": config["job_type"],
            "SCHEDULE_TYPE": "topo",
        },
    }

    print("================================")
    print("Submit config:")
    print(yaml.dump(yaml_data))
    print("================================")

    with open(yaml_file, "w") as file:
        yaml.dump(yaml_data, file, default_flow_style=False, sort_keys=False, width=10000)
    assert os.path.exists(yaml_file), "Failed to generate YAML file."

def execute(cluster_name, yaml_file, dry_run=False):
    print("execute", yaml_file)
    proj_dir = os.path.dirname(yaml_file)
    submit_cmd = [
        "aidi-inf-cli", "job", "submit",
        "-f", yaml_file,
        "--queue_name", cluster_name,
        "-t", proj_dir,
    ]
    if dry_run:
        print("dry run:", ' '.join(submit_cmd))
        return
    print("call:", ' '.join(submit_cmd))
    try:
        subprocess.check_call(submit_cmd)
        print("done:", submit_cmd)
        print("Job submitted successfully.")
    except Exception as e:
        # shutil.rmtree(proj_dir)
        raise e


def execute_v2(cluster_name, config, dry_run=False):
    from uuid import uuid1
    from aidisdk.compute.job_abstract import (
        JobMountType,
        JobType,
        MountItem,
        MountMode,
        RunningResourceConfig,
        StartUpConfig,
    )
    from aidisdk.compute.package_abstract import (
        CodePackageConfig,
        LocalPackageItem,
    )
    from aidisdk import AIDIClient
    import aidisdk 
    from packaging.version import Version

    version = aidisdk.__version__
    if Version(version) < Version("0.20.15b20251127203044"):
        raise ImportError(f"using aidisdk version: {version}, pls run pip3 install aidisdk>=0.20.15b20251127203044")
    else:
        print(f"using aidisdk version: {version}")

    client = AIDIClient()
    print("cluster_name:", cluster_name)

    job = client.single_job.create(
        job_name=config["job_name"].replace("-", "_"),
        job_type=JobType.TRAIN,
        ipd_number=config["project_id"],
        queue_name=cluster_name,
        running_resource=RunningResourceConfig(
            docker_image=config["docker_image"],
            instance=config["num_machines"],
            cpu=config["num_cpus"],
            gpu=config["num_gpus"],
            cpu_mem_ratio=config["cpu_mem_ratio"],
            walltime=config["max_jobtime"],
        ),
        mount=[
            *[
                MountItem(
                    mount_type=JobMountType.BUCKET,
                    name=bucket,
                    mode=MountMode.READ_ONLY,
                )
                for bucket in config["input_bucket"]
            ],
            *[
                MountItem(
                    mount_type=JobMountType.BUCKET,
                    name=bucket,
                    mode=MountMode.READ_AND_WRITE,
                )
                for bucket in config["output_bucket"]
            ],
        ],
        startup=StartUpConfig(
            command=config["run_cmd"],  # noqa
            startup_dir="/",
        ),
        code_package=CodePackageConfig(
            raw_package=LocalPackageItem(
                lpath=config["upload_folder_name"],
                encrypt_passwd="12345",
                follow_softlink=True,
            ).set_as_startup_dir(),
        ),
        # subscribers=["dan.song", "shulan.shen"],
        desc="training job. ",
    )
    print("submitted to cluster:", cluster_name)
    print("job:", job)


if __name__ == "__main__":
    main()
