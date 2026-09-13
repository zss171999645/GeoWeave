# 分析各项外界条件与指标的相关度
# 横坐标：小段id
# 纵坐标：vggt pose；3ddr pose；天气；速度；时间（白天晚上）；动态物体数量；转弯直行

import os
import json
import numpy as np
from glob import glob
from os.path import join, basename
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2


def load_avg_speed_from_tum(tum_path):
    try:
        data = np.loadtxt(tum_path)
    except Exception:
        return None
    data = np.atleast_2d(data)
    if data.shape[0] < 2 or data.shape[1] < 4:
        return None
    t = data[:, 0].astype(float)
    p = data[:, 1:4].astype(float)
    dt = np.diff(t)
    dp = np.linalg.norm(np.diff(p, axis=0), axis=1)
    valid = dt > 0
    if not np.any(valid):
        return None
    return float(np.mean(dp[valid] / dt[valid]))


def detect_turn_from_tum(tum_path, angvel_thresh_deg_per_s=5):
    """
    根据位姿文件检测 segment 是否转弯
    输入:
        tum_path: TUM 格式 pose 文件
        angvel_thresh_deg_per_s: 阈值，角速度超过这个认为转弯 (度/秒)
    输出:
        turn: True/False
        mean_ang_vel_deg: 平均角速度 (度/秒)
    """
    import numpy as np

    try:
        data = np.loadtxt(tum_path)
    except Exception:
        return False, 0.0

    if data.shape[0] < 3:
        return False, 0.0  # 数据太少，认为直行

    # 取 xy 平面
    p = data[:, 1:3]
    t = data[:, 0]

    # 计算每帧位移向量和时间差
    v = np.diff(p, axis=0)
    dt = np.diff(t)
    valid = np.linalg.norm(v, axis=1) > 0
    v = v[valid]
    dt = dt[valid]

    if len(v) < 2:
        return False, 0.0

    # 航向角
    heading = np.arctan2(v[:, 1], v[:, 0])
    dtheta = np.diff(heading)
    dtheta = (dtheta + np.pi) % (2*np.pi) - np.pi  # 归一化 [-π, π]

    # 对应时间间隔
    dt_theta = dt[1:]
    ang_vel = dtheta / dt_theta  # rad/s
    mean_ang_vel_deg = np.mean(np.abs(ang_vel)) * 180 / np.pi

    turn = mean_ang_vel_deg >= angvel_thresh_deg_per_s
    # if turn:
    #     print(f"turn: {mean_ang_vel_deg:.2f} deg/s {tum_path}")
    return turn, mean_ang_vel_deg


def load_dynamic_info(tum_path, cam, raw_data_path="/horizon-bucket/perception-dataprocess/4dlabel/static_visual/000_mvs_data/driving_data/20250506"):
    try:
        data = np.loadtxt(tum_path)
    except Exception:
        return None
    
    raw_clip_path = "/".join(tum_path.replace(exp_root, raw_data_path).split("/")[:-3])
    clipname = raw_clip_path.split("/")[-1]
    site_path = os.path.dirname(raw_clip_path)
    dynamic_path = join(site_path, "Debug/VisualSFM/Vision_Result", clipname, cam, "dynamic_mask")
    all_img_ts = data[:, 0]
    img_names = [f"{int(img_ts * 1000)}.png" for img_ts in all_img_ts]
    
    dynamic_ratios = []
    for img_name in img_names:
        dynamic_mask_path = os.path.join(dynamic_path, img_name.replace(".jpg", ".png"))
        
        if os.path.exists(dynamic_mask_path):
            dynamic_mask = cv2.imread(dynamic_mask_path, cv2.IMREAD_UNCHANGED)
            # 统计255占比
            dynamic_ratio = np.sum(dynamic_mask == 255) / (dynamic_mask.shape[0] * dynamic_mask.shape[1])
            # print(f"dynamic_ratio: {dynamic_ratio}")
            dynamic_ratios.append(dynamic_ratio)
        else:
            dynamic_ratios.append(0.0)
    dynamic_ratios = np.array(dynamic_ratios) 

    return np.mean(dynamic_ratios)


def load_pose_eval(json_path):
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def collect_segment_from_camera(seg_path, cam):
    result = {}
    tum_path = join(seg_path, "sfm_cam2w.txt")
    speed = load_avg_speed_from_tum(tum_path)
    if speed is not None:
        result.setdefault("avg_speed", []).append(speed)
    
    turn, mean_ang_vel_deg = detect_turn_from_tum(tum_path)
    result.setdefault("ang_vel_deg", []).append(mean_ang_vel_deg)

    pose_eval = join(seg_path, "pose_eval.json")
    if os.path.exists(pose_eval):
        jd = load_pose_eval(pose_eval)
        for k, v in jd.items():
            if isinstance(v, dict):
                for m, mv in v.items():
                    if isinstance(mv, (int, float)):
                        result.setdefault((k, m), []).append(float(mv))
            elif isinstance(v, (int, float)):
                result.setdefault(k, []).append(float(v))
                
    # dynamic mask 
    dynamic_ratio = load_dynamic_info(tum_path, cam)
    result.setdefault("dynamic_ratio", []).append(dynamic_ratio)
    
    return result


def merge_camera_results(results_list):
    merged = {}
    for res in results_list:
        for k, vals in res.items():
            merged.setdefault(k, []).extend(vals)
    averaged = {}
    for k, vals in merged.items():
        if not vals:
            continue
        if isinstance(k, tuple):
            alg, metric = k
            averaged.setdefault(alg, {})[metric] = float(np.mean(vals))
        else:
            averaged[k] = float(np.mean(vals))
    return averaged


def process_segment(clipname, segment, camera_names, clip):
    per_camera = []
    for cam in camera_names:
        seg_path = join(clip, cam, segment)
        if os.path.isdir(seg_path):
            per_camera.append(collect_segment_from_camera(seg_path, cam))
    if not per_camera:
        return None
    # vel_degs = [seg["ang_vel_deg"] for seg in per_camera if "ang_vel_deg" in seg]
    # print(max(vel_degs), min(vel_degs))
    merged = merge_camera_results(per_camera)
    merged["clip"] = clip
    merged["clipname"] = clipname
    merged["segment"] = segment
    merged["weather"] = clip.split("/")[-3]
    merged["time"] = clipname.split("_")[-1]
    return merged


def load_segment_info(exp_root, camera_names, num_workers=32):

# 'avg_speed':7.592611833175162
# 'vggt-pandar':{'rpe_trans': 0.006407411450472994, 'rpe_rot': 0.028014123546891357, 'ate_trans': 0.020759618270030873, 'ate_rot': 14.230414818454284}
# 'wigo-pandar':{'rpe_trans': 0.0015360240472250847, 'rpe_rot': 0.004149505481849637, 'ate_trans': 0.007647325766343842, 'ate_rot': 2.7899427671584722}
# 'sfm-pandar':{'rpe_trans': 0.0020648141565725014, 'rpe_rot': 0.00905403272717459, 'ate_trans': 0.005766533622321296, 'ate_rot': 1.643699916696229}
# 'vggt-sfm':{'rpe_trans': 0.006486893948996771, 'rpe_rot': 0.03053451073966923, 'ate_trans': 0.01625975788005468, 'ate_rot': 15.105084023241908}
# 'wigo-sfm':{'rpe_trans': 0.003085799162280489, 'rpe_rot': 0.009737717105052456, 'ate_trans': 0.004502312783199398, 'ate_rot': 1.8230799763415462}
# 'clip':'/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/tmpdata/0917/stage2_fixmaxdepth_epoch30_3ddrinput/driving/6v/Cloudy/Site_113_17429_23_06844_0/DZ878_20240822_013134'
# 'segment':'175-185'
# 'weather':'Cloudy'
# 'time':'013134'

    import re
    def natural_key(s):
        # 提取字符串中的所有数字并转换成整数，返回 tuple 作为排序 key
        return tuple(int(x) for x in re.findall(r'\d+', s))
        
    results = []
    tasks = []
    # 先统计总任务数
    all_segments = []
    clips = sorted(glob(join(exp_root, "*/*/*")))
    for clip in clips:
        clipname = basename(clip)
        cam0_dir = join(clip, camera_names[0])
        if not os.path.isdir(cam0_dir):
            continue
        
        segments_sorted = sorted(os.listdir(cam0_dir), key=natural_key)

        for segment in segments_sorted:
            all_segments.append((clipname, segment, clip))

    all_segments = all_segments[::10]
    
    results = [None] * len(all_segments)

    # for clipname, segment, clip in tqdm(all_segments, desc="Loading segments"):
    #     process_segment(clipname, segment, camera_names, clip)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(process_segment, clipname, segment, camera_names, clip): i
            for i, (clipname, segment, clip) in enumerate(all_segments)
        }

        for fut in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Processing segments"):
            idx = future_to_idx[fut]
            r = fut.result()
            results[idx] = r  # 按原始顺序存放
    
    return results


def _parse_time_to_hour(t):
    if t is None:
        return np.nan
    s = str(t)
    s = "".join(ch for ch in s if ch.isdigit())
    if len(s) >= 6:
        s6 = s[-6:]
        try:
            hh = int(s6[0:2]); mm = int(s6[2:4]); ss = int(s6[4:6])
            return hh + mm / 60.0 + ss / 3600.0
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def _seg_start_key(seg_name, fallback_idx):
    if not seg_name:
        return fallback_idx
    try:
        return int(seg_name.split('-')[0])
    except Exception:
        return fallback_idx

def _detect_algorithms(results):
    algos = set()
    for r in results:
        for k, v in r.items():
            if isinstance(v, dict):
                algos.add(k)
    return sorted(algos)

def plot_factors_vs_pose(
    results,
    external_factors=("avg_speed", "ang_vel_deg", "time", "weather", "dynamic_ratio"),
    algos=["vggt-pandar", "wigo-pandar", "sfm-pandar"],
    metrics=("rpe_trans", "rpe_rot"),
    figsize=None,
    save_path=None,
    xtick_max=25,
    sort_by_factor=True,  # <--- 新增参数
):
    if not results:
        raise ValueError("results is empty")

    results_sorted = results
    # results_sorted = sorted(
    #     results,
    #     key=lambda r, idx=0: _seg_start_key(r.get("segment", None), idx),
    # )

    segments = [r.get("segment", "") for r in results_sorted]
    x = np.arange(len(segments))

    if algos is None:
        algos = _detect_algorithms(results_sorted)
    if not algos:
        raise ValueError("no algorithms detected in results; please pass algos parameter")

    n_factors = len(external_factors)
    if figsize is None:
        figsize = (32, 3.8 * max(1, n_factors))

    fig, axes = plt.subplots(n_factors, 1, figsize=figsize, sharex=True)
    if n_factors == 1:
        axes = [axes]

    base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_map = {alg: base_colors[i % len(base_colors)] for i, alg in enumerate(algos)}

    for ax, factor in zip(axes, external_factors):
        # 提取 factor 数值
        raw_vals = []
        for r in results_sorted:
            if factor not in r:
                raw_vals.append(None)
            else:
                raw_vals.append(_parse_time_to_hour(r.get("time")) if factor=="time" else r.get(factor))

        is_categorical = False
        raw_vals_clean = [v if v is not None else "" for v in raw_vals]
        if raw_vals_clean and not all(isinstance(v, (int, float)) for v in raw_vals_clean):
            is_categorical = True
            cats = sorted(set(raw_vals_clean))
            cat_map = {cat: i for i, cat in enumerate(cats)}
            factor_numeric = np.array([cat_map.get(v, np.nan) for v in raw_vals_clean], dtype=float)
        else:
            factor_numeric = np.array([float(v) if (v is not None and not (isinstance(v, float) and np.isnan(v))) else np.nan for v in raw_vals], dtype=float)

        # 根据参数决定是否排序
        if sort_by_factor:
            sort_idx = np.argsort(factor_numeric)
        else:
            sort_idx = np.arange(len(results_sorted))  # 不排序

        # 绘制 pose metrics
        for alg in algos:
            for metric, ls in zip(metrics, ['-', '--']):
                y = np.array([ (results_sorted[i].get(alg, {}) or {}).get(metric, np.nan) for i in sort_idx ], dtype=float)
                ax.plot(y, linestyle=ls, marker='o', markersize=3, label=f"{alg}.{metric}", color=color_map[alg], linewidth=1.2)

        ax.set_ylabel("pose metrics")
        # ax.set_yscale("log")
        # ax.set_yticks([0.01, 0.05, 0.1, 0.2, 0.5])
        ax.set_ylim(0, 0.1)
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(axis='y', alpha=0.3)

        # right axis
        axr = ax.twinx()
        axr.set_xticks([])
        # axr.set_xlabel("")
        axr.set_xticklabels([])
        y_factor_sorted = factor_numeric[sort_idx]
        if is_categorical:
            axr.plot(y_factor_sorted, linestyle=':', marker='s', markersize=4, label=factor, color='k', linewidth=1.5)
            axr.set_yticks(list(cat_map.values()))
            axr.set_yticklabels(list(cat_map.keys()))
        else:
            axr.plot(y_factor_sorted, linestyle=':', marker='s', markersize=4, label=factor, color='k', linewidth=1.5)
            axr.set_ylabel(factor)

        # legends
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = axr.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize='small', ncol=2)

        ax.set_title(f"External factor: {factor}  —  Pose metrics per algorithm")
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_xticks([])
        # ax.set_xlabel("")
        ax.set_xticklabels([])

    # x ticks
    if len(segments) <= xtick_max:
        xticks = x
        xlabels = segments
    else:
        step = max(1, len(segments) // xtick_max)
        xticks = x[::step]
        xlabels = [segments[i] for i in xticks]
    plt.xticks(xticks, xlabels, rotation=90)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')

    return fig, axes


if __name__ == '__main__':
    save_path = "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/tmpdata/0917/driving/6v"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    experiment_roots = [
        "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/tmpdata/0917/stage2_fixmaxdepth_epoch30_3ddrinput/driving/6v",
        # "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/tmpdata/0917/stage2_fixmaxdepth_epoch30_sfminput/driving/6v"
    ]
    experiment_names = [
        "stage2_3ddr_input", 
        # "stage2_sfm_input"
        ]
    
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


    all_exp_clip_metrics = []
    all_exp_means = []
    
    for exp_root, exp_name in zip(experiment_roots, experiment_names):
        segments_info = load_segment_info(exp_root, camera_names)
        save_path = "/home/users/yingfeng.cai/dev/meshx/data"
        # plot_factors_vs_pose(segments_info, save_path=save_path)
        # 按右轴排序
        fig1, axes1 = plot_factors_vs_pose(segments_info, sort_by_factor=True, save_path=join(save_path, f"{exp_name}_sorted_sample_0.1.png"))

        # 不排序
        fig2, axes2 = plot_factors_vs_pose(segments_info, sort_by_factor=False, save_path=join(save_path, f"{exp_name}_unsorted_sample_0.1.png"))
                

        