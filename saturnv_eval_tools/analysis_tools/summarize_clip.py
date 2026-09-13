import matplotlib.pyplot as plt
from glob import glob
from os.path import join, basename
import os, json
from collections import defaultdict

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

metric_types = ["ate", "rte", "rre"]
methods_to_plot = ["vggt", "vggt_pgo"]
common_methods = ["odo", "sfm"]

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

# --- 函数：读取实验下所有clip的指标 ---
def load_clip_metrics(exp_root, camera_names):
    clips = [clip for clip in sorted(glob(exp_root + "/*/*/*")) if "202" in basename(clip)]
    if len(clips) == 0: 
        clips = [clip for clip in sorted(glob(exp_root + "/*/*/*/*")) if "202" in basename(clip)]
    
    clip_metrics = defaultdict(lambda: defaultdict(list))
    for clip in clips:
        clipname = basename(clip)
        if "202" not in clipname:
            continue
        for camera_name in camera_names:
            res_json = join(exp_root, clip, f"{clipname}_eval", f"{clipname}-{camera_name}.json")
            if not os.path.exists(res_json):
                res_json = join(exp_root, clip, f"{clipname}_eval_7dof", f"{clipname}-{camera_name}.json")
            
            if not os.path.exists(res_json):
                continue
            with open(res_json, "r") as f:
                json_data = json.load(f)
            for k, v in json_data.items():
                if isinstance(v, (int, float)):
                    clip_metrics[clip][k].append(v)
    return clip_metrics

# --- 函数：计算过滤后均值 ---
def compute_filtered_means(clip_metrics, methods, metric_types):
    means = defaultdict(dict)
    stats = defaultdict(lambda: defaultdict(dict))  # 统计信息

    for method in methods:
        for mtype in metric_types:
            key = f"{method}_{mtype}"
            all_values = []

            for clip, metrics in clip_metrics.items():
                if key in metrics:
                    all_values.extend(metrics[key])

            # 分类计数
            count_zero = sum(1 for v in all_values if v == 0)
            count_gt1 = sum(1 for v in all_values if v > 1)
            count_total = len(all_values)

            # 只取 (0,1] 的值计算均值
            valid_values = [v for v in all_values if 0 < v <= 1]
            mean_val = sum(valid_values) / len(valid_values) if valid_values else None

            means[method][mtype] = mean_val
            stats[method][mtype] = {
                "count_=0": count_zero,
                "count_>1": count_gt1,
                "count_total": count_total,
            }

    return means, stats

# --- 函数：绘图 ---
def plot_metrics(all_exp_clip_metrics, experiment_names, save_path):
    fig, axes = plt.subplots(3, 1, figsize=(24, 12))  # 横轴拉长

    all_lines = []
    all_labels = []

    for exp_idx, clip_metrics in enumerate(all_exp_clip_metrics):
        x = list(clip_metrics.keys())
        color = colors[exp_idx % len(colors)]

        for i, mtype in enumerate(metric_types):
            ax = axes[i]

            # vggt / vggt_pgo
            for method, linestyle in zip(methods_to_plot, ["-", "--"]):
                y = [
                    sum(clip_metrics[c].get(f"{method}_{mtype}", [0])) / len(clip_metrics[c].get(f"{method}_{mtype}", [1]))
                    if len(clip_metrics[c].get(f"{method}_{mtype}", [])) > 0 else None
                    for c in x
                ]
                line, = ax.plot(range(len(x)), y, marker="o", linestyle=linestyle, color=color)
                if i == 0:
                    all_lines.append(line)
                    all_labels.append(f"{method} {experiment_names[exp_idx]}")

            # odo / sfm 只画一次
            if exp_idx == 0:
                for method in common_methods:
                    y = [
                        sum(clip_metrics[c].get(f"{method}_{mtype}", [0])) / len(clip_metrics[c].get(f"{method}_{mtype}", [1]))
                        if len(clip_metrics[c].get(f"{method}_{mtype}", [])) > 0 else None
                        for c in x
                    ]
                    if method == "odo":
                        linestyle = "--"
                    else:
                        linestyle = "-"
                    line, = ax.plot(range(len(x)), y, marker="x", linestyle=linestyle, color="gray")
                    if i == 0:
                        all_lines.append(line)
                        all_labels.append(method)

            ax.set_ylabel(mtype.upper())
            ax.set_ylim(0, 1)
            ax.set_xticks([])

    axes[-1].set_xlabel("Clip Index")
    axes[-1].set_xticks([])

    # 统一图例放右上角
    fig.legend(all_lines, all_labels, loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"图像已保存到 {save_path}")

# --- 主函数 ---
def main():
    save_path = "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/tmpdata/1009/stage2_no1_driving/6v"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    experiment_roots = [
        # "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/tmpdata/0906/gtpose_scaleup_4v_olddata_10frame_overlap5_pgo",
        # "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/tmpdata/0906/stage2_4v_olddata_10frame_overlap5_pgo"
        "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/tmpdata/1009/stage2_no1_3ddrinput/driving/6v",
        "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/tmpdata/1009/stage2_no1_sfminput/driving/6v",
    ]
    experiment_names = [
        # "stage1_sfm_input", 
        "stage2_3ddr_input", 
        "stage2_sfm_input"
                        ]

    all_exp_clip_metrics = []
    all_exp_means = []

    for exp_root, exp_name in zip(experiment_roots, experiment_names):
        clip_metrics = load_clip_metrics(exp_root, camera_names)
        all_exp_clip_metrics.append(clip_metrics)

        # 计算均值
        means, stats = compute_filtered_means(clip_metrics, methods_to_plot + common_methods, metric_types)
        all_exp_means.append((exp_name, means, stats))

    # 输出均值
    with open(join(save_path, "means.txt"), "w") as f:
        for exp_name, means, stats in all_exp_means:
            print(f"\nExperiment: {exp_name}")
            f.write(f"\nExperiment: {exp_name}\n")
            for method, mvals in means.items():
                print(f"  {method}: ", end="")
                f.write(f"  {method}: ")
                for mtype, v in mvals.items():
                    st = stats[method][mtype]
                    if v is None:
                        print(f"{mtype}=None ({st['count_=0']}/{st['count_>1']}/{st['count_total']}) ", end="")
                        f.write(f"{mtype}=None ({st['count_=0']}/{st['count_>1']}/{st['count_total']}) ")
                    else:
                        print(f"{mtype}={v:.4f} ({st['count_=0']}/{st['count_>1']}/{st['count_total']}) ", end="")
                        f.write(f"{mtype}={v:.4f} ({st['count_=0']}/{st['count_>1']}/{st['count_total']}) ")
                print()
                f.write("\n")

    with open(join(save_path, "metrics.json"), "w") as f:
        json.dump(all_exp_clip_metrics, f)
        
    # 绘图
    import pandas as pd
    with open(join(save_path, "metrics.json"), "r") as f:
        all_exp_clip_metrics = json.load(f)
        print(f"加载 {join(save_path, 'metrics.json')}")
    
    plot_metrics(all_exp_clip_metrics, experiment_names, join(save_path, "metrics.png"))
    
    scene_counter = defaultdict(int)
    rows = {}
    scene_groups = defaultdict(list)  # 每个场景的 clip 行名

    # 遍历 list
    for item in all_exp_clip_metrics:
        for clip, metrics in item.items():
            scene = clip.split("/")[-3]
            scene_counter[scene] += 1
            scene_name = f"{scene}{scene_counter[scene]}"  # 带编号

            # clip 的平均值
            row = {metric: sum(vals) / len(vals) for metric, vals in metrics.items()}
            rows[scene_name] = row
            scene_groups[scene].append(scene_name)

    # 转成 DataFrame
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "weather/scene"

    # 先生成每个场景的 mean 行
    mean_rows = []
    for scene, clip_names in scene_groups.items():
        mean_row = df.loc[clip_names].mean()
        mean_rows.append(pd.DataFrame([mean_row], index=[f"{scene}_mean"]))

    df_means = pd.concat(mean_rows)

    # 拼接最终 DataFrame：均值行在前，clip 行在后
    df_final = pd.concat([df_means, df])

    print(df_final.round(4))

    # 保存到 txt
    with open(join(save_path, "metrics_per_clip.txt"), "w") as f:
        f.write(df_final.round(4).to_string())
        
                

if __name__ == "__main__":
    main()