import json
import os
from os.path import join, basename
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from tabulate import tabulate


def check_turn(ts, turn_list):
    for turn_start, turn_end in turn_list:
        if turn_start <= ts <= turn_end:
            return True
    return False

def compute_mean_metrics(metric_list):
    if not metric_list:
        return {}
    keys = metric_list[0].keys()
    result = {}
    for key in keys:
        values = [m[key] for m in metric_list if isinstance(m[key], (int, float))]
        result[f"max_{key}"] = round(np.max(values), 6)
        result[f"min_{key}"] = round(np.min(values), 6)
        result[f"mean_{key}"] = round(np.mean(values), 6)
        result[f"median_{key}"] = round(np.median(values), 6)
        result[f"1sigma_{key}"] = round(np.percentile(values, 67), 6)
        result[f"2sigma_{key}"] = round(np.percentile(values, 95), 6)
        result[f"3sigma_{key}"] = round(np.percentile(values, 99.7), 6)
    return result

def process_metric(metric):
    imgpath = os.path.realpath(join(metric["path"], "images", f"{metric['camera']:02d}", f"{metric['frame']:05d}.jpg"))
    clip = basename(metric["path"])
    img_ts = float(basename(imgpath).split(".")[0]) / 1000.0

    clip_turns_info = turns.get(clip, [])
    clip_turn_timestamp = [(turn_info["start_ts"]/1000.0, turn_info["end_ts"]/1000.0) for turn_info in clip_turns_info]

    segment_metrics = {
        "cam:RPE_trans": metric["cam:RPE_trans"],
        "cam:RPE_rot": metric["cam:RPE_rot"],
        "dpt:absrel_error_ratio": metric["dpt:absrel_error_ratio"],
        "dpt:depth_error>thres0.2m_ratio(2-20m)": metric["dpt:depth_error>thres0.2m_ratio(2-20m)"],
        "cam:translation_scale": metric["cam:translation_scale"],
    }

    is_turn = check_turn(img_ts, clip_turn_timestamp)
    return segment_metrics, is_turn


        
if __name__ == "__main__":
    metric_jpath = "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/vggt/result/test/parking/0815/bussiness_parking_mechanical-mvseq-w_cam_emb-fisheye-fixscale_15/00039999/metrics.json"
    with open(metric_jpath, "r") as f:
        metrics = json.load(f)["metrics"]

    turns_info = "/horizon-bucket/perception-dataprocess/4dlabel/static_visual/002_vggt_data/evc_data/business_data/driving_data/20250522/turn_timestamps_eval100site.json"
    with open(turns_info, "r") as f:
        turns = json.load(f)

    all_metrics = []
    turn_metrics = []
    other_metrics = []
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_metric, metrics), total=len(metrics)))

    for segment_metrics, is_turn in results:
        if is_turn:
            turn_metrics.append(segment_metrics)
        else:
            other_metrics.append(segment_metrics)
        all_metrics.append(segment_metrics)
        
        
    def get_named_metrics(name, metrics_list):
        return name, compute_mean_metrics(metrics_list)

    # 获取每组的统计结果
    turn_name, turn_stats = get_named_metrics("Turn", turn_metrics)
    other_name, other_stats = get_named_metrics("Other", other_metrics)
    all_name, all_stats = get_named_metrics("All", all_metrics)

    # 统一指标名（所有组可能有些缺失）
    all_keys = sorted(set(turn_stats.keys()) | set(other_stats.keys()) | set(all_stats.keys()))

    # 构造表格：每行是一个指标
    rows = []
    for key in all_keys:
        row = [key,
            round(turn_stats.get(key, float("nan")), 6),
            round(other_stats.get(key, float("nan")), 6),
            round(all_stats.get(key, float("nan")), 6)]
        rows.append(row)

    # 打印表格
    print(tabulate(rows, headers=["Metric", f"Turn {len(turn_metrics)}", f"Other {len(other_metrics)}", f"All {len(all_metrics)}"], tablefmt="pretty"))
    
    
    save_root = "data/turn_metric"
    save_root = os.path.dirname(metric_jpath)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    # print("============== turn metric ===============")
    # print(len(turn_metrics))
    mean_metrics = compute_mean_metrics(turn_metrics)
    turn_metrics.insert(0, mean_metrics)
    with open(join(save_root, "turn_metrics.json"), "w") as f:
        json.dump(turn_metrics, f, indent=2)
    
    # print("============== other metric ===============")
    # print(len(other_metrics))
    mean_metrics = compute_mean_metrics(other_metrics)
    other_metrics.insert(0, mean_metrics)
    with open(join(save_root, "other_metrics.json"), "w") as f:
        json.dump(other_metrics, f, indent=2)
    
    # print("============== all metric ===============")
    # print(len(all_metrics))
    mean_metrics = compute_mean_metrics(all_metrics)
    
