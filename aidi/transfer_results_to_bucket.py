#!/usr/bin/env python3
import argparse
import subprocess
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', required=True)
parser.add_argument('--src_tb_root', required=True)
parser.add_argument('--src_vis_root', required=True)
parser.add_argument('--dst_root', required=True)
parser.add_argument('--interval', type=int, default=600)
args = parser.parse_args()

print(f"[INFO] Transfer daemon started for exp: {args.exp_name}")
print(f"[INFO] TB: {args.src_tb_root}/record/{args.exp_name} -> {args.dst_root}/record/{args.exp_name}")
print(f"[INFO] VIS: {args.src_vis_root}/result/{args.exp_name} -> {args.dst_root}/result/{args.exp_name}")

while True:
    try:
        # Sync tensorboard logs for specific experiment
        tb_path = f'{args.src_tb_root}/record/{args.exp_name}'
        if os.path.isdir(tb_path):
            dst_tb_path = f'{args.dst_root}/record/{args.exp_name}'
            if not os.path.exists(os.path.dirname(dst_tb_path)):
                os.makedirs(os.path.dirname(dst_tb_path), exist_ok=True)
            cmd = f'rsync -avz --update {tb_path}/ {dst_tb_path}/'
            subprocess.run(cmd, shell=True)
            print(f"[INFO] TB synced at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"[WARN] TB folder not found: {tb_path}")

        # Sync visualization results for specific experiment
        vis_path = f'{args.src_vis_root}/result/{args.exp_name}'
        if os.path.isdir(vis_path):
            dst_vis_path = f'{args.dst_root}/result/{args.exp_name}'
            if not os.path.exists(os.path.dirname(dst_vis_path)):
                os.makedirs(os.path.dirname(dst_vis_path), exist_ok=True)
            cmd = f'rsync -avz --update {vis_path}/ {dst_vis_path}/'
            subprocess.run(cmd, shell=True)
            print(f"[INFO] VIS synced at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"[WARN] VIS folder not found: {vis_path}")
    except Exception as e:
        print(f"[ERROR] {e}")

    time.sleep(args.interval)  # 10 minutes
