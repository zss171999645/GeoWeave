#!/usr/bin/env python3
"""Join pose/structure metrics and summarize distractor-ratio trends."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


LOWER_IS_BETTER = (
    "ATE",
    "RPE trans",
    "RPE rot",
    "acc_mean_norm",
    "comp_mean_norm",
    "depth_abs_rel",
    "depth_rmse_norm",
)
HIGHER_IS_BETTER = ("nc_mean", "depth_delta1")
ALL_METRICS = LOWER_IS_BETTER + HIGHER_IS_BETTER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-summaries", nargs="+", required=True)
    parser.add_argument("--structure-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def variant_to_context_ratio(variant: str) -> float:
    if not str(variant).startswith("noise"):
        raise ValueError(f"Unsupported distractor variant: {variant}")
    count = int(str(variant).removeprefix("noise"))
    if count < 0 or count > 4:
        raise ValueError(f"Distractor count outside [0,4]: {count}")
    return float(count / 4.0)


def compute_clean_relative_rows(
    rows: list[dict[str, Any]],
    *,
    lower_is_better: Iterable[str],
    higher_is_better: Iterable[str],
) -> list[dict[str, Any]]:
    lower = tuple(lower_is_better)
    higher = tuple(higher_is_better)
    clean_by_model = {str(row["model"]): row for row in rows if row["variant"] == "noise0"}
    output: list[dict[str, Any]] = []
    for row in rows:
        model = str(row["model"])
        clean = clean_by_model.get(model)
        if clean is None:
            raise KeyError(f"Missing noise0 row for {model}")
        payload: dict[str, Any] = {
            "model": model,
            "variant": str(row["variant"]),
            "context_distractor_ratio": variant_to_context_ratio(str(row["variant"])),
        }
        for key in lower:
            payload[f"degradation_{key}"] = float(row[key]) - float(clean[key])
        for key in higher:
            payload[f"degradation_{key}"] = float(clean[key]) - float(row[key])
        output.append(payload)
    return output


def join_aggregates(pose_summaries: list[dict[str, Any]], structure_summary: dict[str, Any]) -> list[dict[str, Any]]:
    pose_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for summary in pose_summaries:
        model = str(summary["model"])
        for variant, metrics in summary["variant_summary"].items():
            pose_by_key[(model, str(variant))] = dict(metrics)
    structure_by_key = {
        (str(row["model"]), str(row["variant"])): row
        for row in structure_summary["aggregate"]
    }
    keys = sorted(
        set(pose_by_key) & set(structure_by_key),
        key=lambda item: (item[0], int(item[1].removeprefix("noise"))),
    )
    rows: list[dict[str, Any]] = []
    for model, variant in keys:
        pose = pose_by_key[(model, variant)]
        structure = structure_by_key[(model, variant)]
        row = {
            "model": model,
            "variant": variant,
            "distractor_count": int(variant.removeprefix("noise")),
            "context_distractor_ratio": variant_to_context_ratio(variant),
            "num_sequences": int(pose["num_sequences"]),
        }
        for key in ALL_METRICS:
            row[key] = float(pose[key] if key in pose else structure[key])
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(rows: list[dict[str, Any]], deltas: list[dict[str, Any]]) -> str:
    lines = [
        "# Distractor-Ratio Sweep",
        "",
        "All metrics evaluate the fixed six-view clean prefix. Context distractor ratio varies over 0/25/50/75/100%.",
        "",
        "| Model | Ratio | ATE ↓ | RPE-t ↓ | Acc ↓ | Comp ↓ | NC ↑ | Depth AbsRel ↓ | Depth delta1 ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['context_distractor_ratio']:.0%} | {row['ATE']:.6f} | "
            f"{row['RPE trans']:.6f} | {row['acc_mean_norm']:.6f} | {row['comp_mean_norm']:.6f} | "
            f"{row['nc_mean']:.6f} | {row['depth_abs_rel']:.6f} | {row['depth_delta1']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Positive degradation means worse than the same model at 0% distractors.",
            "",
            "| Model | Ratio | ΔATE | ΔAcc | ΔComp | NC degradation | Depth AbsRel degradation | delta1 degradation |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in deltas:
        lines.append(
            f"| {row['model']} | {row['context_distractor_ratio']:.0%} | {row['degradation_ATE']:.6f} | "
            f"{row['degradation_acc_mean_norm']:.6f} | {row['degradation_comp_mean_norm']:.6f} | "
            f"{row['degradation_nc_mean']:.6f} | {row['degradation_depth_abs_rel']:.6f} | "
            f"{row['degradation_depth_delta1']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def render_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (
        ("ATE", "ATE ↓"),
        ("RPE trans", "RPE-t ↓"),
        ("acc_mean_norm", "Point Acc ↓"),
        ("comp_mean_norm", "Point Comp ↓"),
        ("nc_mean", "Normal consistency ↑"),
        ("depth_abs_rel", "Depth AbsRel ↓"),
        ("depth_rmse_norm", "Depth RMSE norm ↓"),
        ("depth_delta1", "Depth delta1 ↑"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), constrained_layout=True)
    models = sorted({str(row["model"]) for row in rows})
    for axis, (key, title) in zip(axes.flat, metrics):
        for model in models:
            subset = sorted(
                (row for row in rows if row["model"] == model),
                key=lambda row: float(row["context_distractor_ratio"]),
            )
            axis.plot(
                [100.0 * float(row["context_distractor_ratio"]) for row in subset],
                [float(row[key]) for row in subset],
                marker="o",
                label=model,
            )
        axis.set_title(title)
        axis.set_xlabel("Context distractor ratio (%)")
        axis.grid(alpha=0.25)
    axes.flat[0].legend()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    pose_summaries = [load_json(path) for path in args.pose_summaries]
    structure_summary = load_json(args.structure_summary)
    rows = join_aggregates(pose_summaries, structure_summary)
    deltas = compute_clean_relative_rows(
        rows,
        lower_is_better=LOWER_IS_BETTER,
        higher_is_better=HIGHER_IS_BETTER,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "distractor_ratio_absolute.csv", rows)
    write_csv(output_dir / "distractor_ratio_degradation.csv", deltas)
    (output_dir / "distractor_ratio_summary.json").write_text(
        json.dumps({"absolute": rows, "degradation": deltas}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "distractor_ratio_summary.md").write_text(render_markdown(rows, deltas), encoding="utf-8")
    render_plot(output_dir / "distractor_ratio_trends.png", rows)
    print(f"[distractor-ratio-summary] wrote {output_dir}")


if __name__ == "__main__":
    main()
