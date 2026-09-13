
import json
import os
import yaml
import numpy as np
from os.path import join, exists

def collect_all_metrics(summary_path):
    with open(summary_path, "r") as f:
        summary = json.load(f)

    results = {}
    for item in summary:
        if item.get("Camera") != "ALL":
            continue
        pose_type = item.get("PoseType")
        metric = item.get("Metric")
        key = f"{pose_type}-{metric}"

        results[key] = {
            "Mean": item.get("Mean"),
            "Median": item.get("Median"),
            "Sigma 68%": item.get("Sigma 68%"),
            "Sigma 95%": item.get("Sigma 95%"),
            "Sigma 99.6%": item.get("Sigma 99.6%")
        }
    return results


def extract_mean(summary_list):
    means, sigma1s, sigma2s, sigma3s = [], [], [], []
    for line in summary_list:
        try:
            parts = line.split(",")
            mean = float(parts[1].split("mean=")[1])
            sigma1 = float(parts[2].split("=")[1])
            sigma2 = float(parts[3].split("=")[1])
            sigma3 = float(parts[4].split("=")[1])
            means.append(mean)
            sigma1s.append(sigma1)
            sigma2s.append(sigma2)
            sigma3s.append(sigma3)
        except Exception:
            continue

    count = len(means)
    if count > 0:
        return (
            f"Total clips: {count}, "
            f"Average mean: {sum(means)/count:.4f}, "
            f"1sigma: {sum(sigma1s)/count:.4f}, "
            f"2sigma: {sum(sigma2s)/count:.4f}, "
            f"3sigma: {sum(sigma3s)/count:.4f}"
        )
    else:
        return "Total clips: 0, Average mean: N/A, 1sigma: N/A, 2sigma: N/A, 3sigma: N/A"
    
def compute_mean_metrics(metric_list):
    if not metric_list:
        return {}

    keys = metric_list[0].keys()
    result = {}
    for key in keys:
        values = [m[key] for m in metric_list if isinstance(m[key], (int, float))]
        result[key] = float(np.mean(values)) if values else None
    return result

def read_config(config_file):
    try:
        with open(config_file, "r") as f:
            content = yaml.safe_load(f)
    except Exception:
        with open(config_file, "r") as f:
            content = yaml.unsafe_load(f)
    return content

if __name__ == "__main__":
    camera_names = [
                "camera_front", 
                "camera_front_left", 
                "camera_front_right",
                "camera_rear", 
                "camera_rear_left", 
                "camera_rear_right"
                    ]
    
    config_file = f"saturnv_eval_tools/configs/evalset.yaml"
    config = read_config(config_file)

    infer_sets = config["infer_sets"]

    for infer_set in infer_sets:
        summary_metrics = {}
        config_infer_set = config[infer_set]
        save_root = config_infer_set["save_root"]
        sites = config_infer_set["sites"]
        for site in sites:
            site_split = site.split(",")
            site = site_split[0]
            save_dir = join(save_root, site)
            clips = os.listdir(save_dir)

            for clip in clips:
                summary_jpath = join(save_dir, clip, "summary.json")
                if not os.path.exists(summary_jpath):
                    continue

                clip_metrics = collect_all_metrics(summary_jpath)

                for key, values in clip_metrics.items():
                    line = f"{site},{clip}: mean={values['Mean']}, 1sigma={values['Sigma 68%']}, 2sigma={values['Sigma 95%']}, 3sigma={values['Sigma 99.6%']}"
                    summary_metrics.setdefault(key, []).append(line)

        for metric in summary_metrics.keys():
            with open(join(save_root, f"summary_{metric}.txt"), "w") as f:
                f.write(extract_mean(summary_metrics[metric]) + "\n")
                f.write("\n".join(summary_metrics[metric]))
                print(f"Saved {join(save_root, f'summary_{metric}.txt')}")
            
        

        
    
    
    

