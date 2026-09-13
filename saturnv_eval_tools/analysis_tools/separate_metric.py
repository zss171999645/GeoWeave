import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def average_metrics(metrics_list):
    """对一个 metrics 列表取平均"""
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    avg = {}
    for k in keys:
        values = [m[k] for m in metrics_list if isinstance(m[k], (int, float))]
        if values:
            avg[k] = float(np.mean(values))
    return avg


def split_metrics(metrics, driving6v_key, machanical_key):
    """按场景和相机id拆分 metrics"""
    driving6v, machanical4v, machanical10v = [], [], []
    for metric in metrics:
        img_path = metric["path"]
        camid = metric["camera"]
        if driving6v_key in img_path:
            driving6v.append(metric)
        elif machanical_key in img_path:
            if camid == 0:
                machanical10v.append(metric)
            elif camid == 6:
                machanical4v.append(metric)
    return driving6v, machanical4v, machanical10v


def load_baseline(baseline_path, min_iter=None, max_iter=None):
    """读取 baseline 路径下所有迭代的 summary"""
    summaries = []
    iter_nums = []
    for iter in sorted(os.listdir(baseline_path), key=lambda x: int(x)):
        iter_num = int(iter)
        if max_iter is not None and iter_num > max_iter:
            continue
        if min_iter is not None and iter_num < min_iter:
            continue
        metric_jpath = os.path.join(baseline_path, iter, "metrics.json")
        if not os.path.exists(metric_jpath):
            continue
        with open(metric_jpath, "r") as f:
            metrics = json.load(f)["metrics"]
        avg_summary = average_metrics(metrics)
        summaries.append(avg_summary)
        iter_nums.append(iter_num)
    return iter_nums, summaries


def collect_experiment_summaries(test_root, driving6v_key, machanical_key, min_iter=None, max_iter=None):
    """收集实验路径下所有迭代的 summary (6v, 4v, 10v)"""
    summaries_6v, summaries_4v, summaries_10v = [], [], []
    iter_nums = []
    for iter in sorted(os.listdir(test_root)):
        if min_iter is not None and int(iter) < min_iter:
            continue
        if max_iter is not None and int(iter) > max_iter:
            continue
        
        metric_jpath = os.path.join(test_root, iter, "metrics.json")
        if not os.path.exists(metric_jpath):
            continue
        with open(metric_jpath, "r") as f:
            metrics = json.load(f)["metrics"]

        driving6v, machanical4v, machanical10v = split_metrics(
            metrics, driving6v_key, machanical_key
        )
        summaries_6v.append(average_metrics(driving6v))
        summaries_4v.append(average_metrics(machanical4v))
        summaries_10v.append(average_metrics(machanical10v))
        iter_nums.append(int(iter))
    return iter_nums, summaries_6v, summaries_4v, summaries_10v


def collect_dr_summaries(dr_path, driving6v_key, machanical_key):
    """读取一个静态 metrics.json，返回三个 summary dict (6v, 4v, 10v)"""
    with open(dr_path, "r") as f:
        metrics = json.load(f)["metrics"]
    driving6v, machanical4v, machanical10v = split_metrics(metrics, driving6v_key, machanical_key)
    summary_6v = average_metrics(driving6v)
    summary_4v = average_metrics(machanical4v)
    summary_10v = average_metrics(machanical10v)
    return {"6v": summary_6v, "4v": summary_4v, "10v": summary_10v}


def group_metric_keys(*summaries):
    """把所有 summary 里的 key 按 prefix 分组"""
    all_keys = set()
    for s in sum(summaries, []):
        all_keys.update(s.keys())

    grouped_keys = defaultdict(list)
    for key in sorted(all_keys):
        prefix, suffix = key.split(":", 1) if ":" in key else ("other", key)
        grouped_keys[prefix].append((key, suffix))
        
    filter_dpt_metric = []
    for (key, suffix) in grouped_keys["dpt"]:
        if "depth" in suffix or "absrel_error_ratio" in suffix:
            filter_dpt_metric.append((key, suffix))
    grouped_keys["dpt"] = filter_dpt_metric
    
    return grouped_keys


def plot_group(prefix, key_list, iter_nums,
               summaries_6v, summaries_4v, summaries_10v,
               baselines, out_dir, dr_metrics=None):
    if prefix == "dpt":
        dr_metrics = None
    """画出某个前缀的所有指标曲线"""
    n = len(key_list)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    styles = {
        "driving6v": {"color": "tab:blue", "marker": "o"},
        "machanical4v": {"color": "tab:orange", "marker": "o"},
        "machanical10v": {"color": "tab:green", "marker": "o"},
        "stage2": {"color": "tab:gray", "linestyle": "--", "marker": "^"},
        "business_fisheye_mv": {"color": "tab:orange", "linestyle": "--", "marker": "^"},
        "business_driving_sv": {"color": "tab:blue", "linestyle": "--", "marker": "^"},
    }

    for ax, (key, suffix) in zip(axes, key_list):
        y6 = [s.get(key, np.nan) for s in summaries_6v]
        y4 = [s.get(key, np.nan) for s in summaries_4v]
        y10 = [s.get(key, np.nan) for s in summaries_10v]

        ax.plot(iter_nums, y6, **styles["driving6v"])
        ax.plot(iter_nums, y4, **styles["machanical4v"])
        ax.plot(iter_nums, y10, **styles["machanical10v"])

        for bname, (iters_b, summaries_b) in baselines.items():
            yb = [s.get(key, np.nan) for s in summaries_b]
            ax.plot(iters_b, yb, label=bname, **styles[bname])

        # 🔹 加 dr 参考线 (三类)
        if dr_metrics:
            if key in dr_metrics["6v"]:
                ax.hlines(dr_metrics["6v"][key], xmin=min(iter_nums), xmax=max(iter_nums),
                          colors=styles["driving6v"]["color"], linestyles="--", label="dr-6v")
            if key in dr_metrics["4v"]:
                ax.hlines(dr_metrics["4v"][key], xmin=min(iter_nums), xmax=max(iter_nums),
                          colors=styles["machanical4v"]["color"], linestyles="--", label="dr-4v")
            if key in dr_metrics["10v"]:
                ax.hlines(dr_metrics["10v"][key], xmin=min(iter_nums), xmax=max(iter_nums),
                          colors=styles["machanical10v"]["color"], linestyles="--", label="dr-10v")

        ax.set_title(suffix)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        ax.grid(True)

    # 多余 subplot 去掉
    for ax in axes[len(key_list):]:
        ax.axis("off")

    handles = [
        plt.Line2D([0], [0], **styles["driving6v"]),
        plt.Line2D([0], [0], **styles["machanical4v"]),
        plt.Line2D([0], [0], **styles["machanical10v"]),
    ]
    labels = ["driving6v", "machanical4v", "machanical10v"]

    for bname in baselines.keys():
        handles.append(plt.Line2D([0], [0], **styles[bname]))
        labels.append(bname)

    if dr_metrics:
        handles.extend([
            plt.Line2D([0], [0], color=styles["driving6v"]["color"], linestyle="--"),
            plt.Line2D([0], [0], color=styles["machanical4v"]["color"], linestyle="--"),
            plt.Line2D([0], [0], color=styles["machanical10v"]["color"], linestyle="--"),
        ])
        labels.extend(["dr-6v", "dr-4v", "dr-10v"])

    fig.legend(handles, labels, loc="upper right", fontsize=10)
    fig.suptitle(f"{prefix} metrics", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(out_dir, f"{prefix}_metrics.png"))
    plt.close()


if __name__ == "__main__":
    # =====================
    # 配置路径
    # =====================
    test_root = "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/vggt/result/exp/basemodel/0927/8x8_1e-4_stage2_resumebase_No2"
    baseline_roots = {
        # "stage2": "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/vggt/result/exp/basemodel/0828/8x8_1e-4_stage2_basemodel_3ddrratio0.7",
        "business_fisheye_mv": "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/vggt/result/exp/business_parking/0812/8x8_1e-4_bussiness_parking_mechanical_fisheye_mv",
        "business_driving_sv": "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/vggt/result/exp/business_driving/0713/3ksite_5v_4x8_lr2e-5_fovaug_imgaug_1kiter"
    }
    dr_path = "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/vggt/result/exp/basemodel/0828/3ddr/metrics.json"

    driving6v_key = "driving_data"
    machanical_key = "20250630_mechanical_at128_eval"
    out_dir = "/home/users/yingfeng.cai/dev/meshx/data/stage2model"
    os.makedirs(out_dir, exist_ok=True)

    # =====================
    # 加载实验 & baseline & dr
    # =====================
    iter_nums, summaries_6v, summaries_4v, summaries_10v = collect_experiment_summaries(
        test_root, driving6v_key, machanical_key, min_iter=-1*1000
    )
    baselines = {name: load_baseline(path, min_iter=0*1000, max_iter=160*1000) for name, path in baseline_roots.items()}
    dr_metrics = collect_dr_summaries(dr_path, driving6v_key, machanical_key)

    # =====================
    # 分组绘图
    # =====================
    grouped_keys = group_metric_keys(
        summaries_6v, summaries_4v, summaries_10v, *[v[1] for v in baselines.values()]
    )
    for prefix, key_list in grouped_keys.items():
        plot_group(prefix, key_list, iter_nums, summaries_6v, summaries_4v, summaries_10v,
                   baselines, out_dir, dr_metrics=dr_metrics)