""" This is a simplified ssh launcher for launching multiple processes on multiple nodes.
    It is similar to the functionality of mpirun.

Usage:
    1. Run `python ssh_launcher.py "python3 your_script.py"`, will launch a process on each node.
"""

import os
import sys
import time
import socket
import logging
import argparse
import subprocess
import shlex
from threading import Thread
from functools import partial


logger = logging.getLogger("ssh_launcher")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

LAUNCH_MODE = "per_node"
assert LAUNCH_MODE in ["per_node", "per_worker"]


def run(cmd, node, exit_if_error: bool = False):
    logger.info(f"launching on node {node}: {cmd}")
    try:
        subprocess.check_call(
            cmd,
            shell=True,
        )
    except subprocess.CalledProcessError as e:
        logger.warning(f"failed on node {node}: failed({e.returncode})!")
        if exit_if_error:
            os._exit(e.returncode)
        else:
            raise e


def get_env(pass_envs):
    envs = []
    for k, v in list(pass_envs.items()):
        if "()" in str(k) or "()" in str(v):
            continue
        envs.append(f"export {k}={shlex.quote(str(v))};")
    return " ".join(envs)


class SafeThread(Thread):
    def __init__(self, target, args=(), kwargs=None, error_callback=None):
        super().__init__()
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.error_callback = error_callback
        self.exception = None
        self.result = None
    
    def run(self):
        try:
            self.result = self.target(*self.args, **self.kwargs)
        except Exception as e:
            self.exception = e
            if self.error_callback:
                self.error_callback(e, self.args, self.kwargs)


def error_handler(error, args, kwargs):
    logger.error(f"ssh command failed: {args}: {error}")
    sys.stdout.flush()
    sys.stderr.flush()


def apply_on_multi_node_by_ssh(
    nworker,
    ngpu,
    port,
    prog_func,
    hosts,
    cmd,
    local_dir,
    pass_envs,
):
    pass_envs = pass_envs.copy()
    assert nworker == len(hosts) * ngpu

    thread_list = []
    for i in range(nworker if LAUNCH_MODE == "per_worker" else len(hosts)):
        if LAUNCH_MODE == "per_worker":
            host = hosts[i // ngpu]
            node_rank = i // ngpu
        elif LAUNCH_MODE == "per_node":
            host = hosts[i]
            node_rank = i
        else:
            raise ValueError(f"Unknown launch mode: {LAUNCH_MODE}")

        pass_envs["WORLD_SIZE"] = str(nworker)
        pass_envs["MASTER_ADDR"] = hosts[0]
        pass_envs["MASTER_PORT"] = port
        pass_envs["NODE_RANK"] = str(node_rank)
        pass_envs["RANK"] = str(node_rank * ngpu) if LAUNCH_MODE == "per_node" else str(i)
        pass_envs["LOCAL_RANK"] = 0 if LAUNCH_MODE == "per_node" else str(i % ngpu)
        pass_envs["GPU_PER_NODE"] = str(ngpu)
        pass_envs["NUM_NODES"] = str(len(hosts))
        # pass_envs["CUDA_VISIBLE_DEVICES"] = str(i % ngpu)

        remote_cmd = f"{get_env(pass_envs)} cd {shlex.quote(local_dir)} && {cmd}"
        prog = f"ssh -o StrictHostKeyChecking=no {host} {shlex.quote(remote_cmd)}"
        logger.info(f"starting thread for {host}")

        thread = SafeThread(target=prog_func, args=(prog, host), error_callback=error_handler)
        thread.daemon = True
        thread.start()
        thread_list.append(thread)

    task_failed = False
    for i, t in enumerate(thread_list):
        t.join()
        if t.exception:
            logger.error(f"thread {i} failed: {t.exception}")
            task_failed = True
        else:
            logger.info(f"thread {i} join success")
    
    if task_failed:
        raise RuntimeError("ssh launcher task failed")


def submit(nworker, ngpu, port, cmd):
    logger.info(f"submitting job with {nworker} workers and {ngpu} gpus")

    nodes, node_rank = parse_aidi_nodes()
    pass_keys = ["PYTHONPATH", "CUDA", "LIBRARY_PATH", "PATH", "NCCL_SOCKET_IFNAME", "LD_LIBRARY_PATH", "GPU_STR", "WORKING_PATH"]
    pass_keys += list(key for key in os.environ.keys() if "NCCL" in key or "IB" in key)
    pass_envs = {}
    skip_envs = {}
    for k, v in os.environ.items():
        if k in pass_keys:
            pass_envs[k] = v
        else:
            skip_envs[k] = v
    if "HOST_NODE_ADDR" not in os.environ:
        pass_envs["HOST_NODE_ADDR"] = nodes[0]
    logger.info(f"\tpass_envs: {pass_envs}")
    logger.info(f"\tskip_envs: {skip_envs}\n")

    apply_on_multi_node_by_ssh(
        nworker,
        ngpu,
        port,
        prog_func=run,
        hosts=nodes,
        cmd=cmd,
        local_dir=os.getcwd(),
        pass_envs=pass_envs,
    )


def parse_aidi_nodes():
    nodes = []
    node_rank = None
    with open('/job_data/hosts', 'r') as f:
        hostnames = [line.strip() for line in f.readlines() if line.strip()]
        hostnames = sorted(hostnames)
        trials = 100
        for hostname in hostnames:
            if hostname.startswith(socket.gethostname()):
                node_rank = len(nodes)
            for i in range(trials):
                try:
                    ip = socket.gethostbyname(hostname)
                    if ip is not None:
                        nodes.append(ip)
                        break
                except Exception as e:
                    logger.warning(f"{i}-th attempt to get hostip by hostname failed!!! {e}")
                    time.sleep(1)
                    ip = None
            if ip is None:
                raise ConnectionError(f"{hostname} to IP ERROR !")
    if len(nodes) < 1 or node_rank is None:
        raise ValueError(f"Failed to parse nodes: {nodes}, {node_rank}")
    return nodes, node_rank


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--nworker",
        type=int,
        required=True,
        help="number of worker process to be launched",
    )
    parser.add_argument(
        "-g",
        "--ngpus",
        type=int,
        required=True,
        help="number of gpus on per node",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=str,
        default="29500",
        help="Port for master node",
    )
    parser.add_argument(
        "-t",
        "--trial",
        type=int,
        default=3,
        help="number of trials to restart the program if failed",
    )
    parser.add_argument(
        "command", nargs="+", help="command for plugin program"
    )
    args = parser.parse_args()
    cmd = " ".join(args.command)

    success = False
    for i in range(args.trial):
        logger.info(f"executing trial {i+1} of {args.trial}")
        try:
            submit(
                nworker=args.nworker,
                ngpu=args.ngpus,
                port=args.port,
                cmd=cmd,
            )
            success = True
            logger.info(f"trial {i+1} of {args.trial} succeeded")
            break
        except Exception as e:
            for _ in range(10):
                logger.error(f"trial {i+1} of {args.trial} failed: {e}")
                time.sleep(1)
            continue

    if not success:
        logger.error(f"all {args.trial} trials failed, exiting with code 1")
        sys.exit(1)
