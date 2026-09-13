#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
MODE_TO_SCRIPT = {
    "monodepth": SCRIPT_DIR / "eval_pi3_monodepth_protocol.py",
    "videodepth": SCRIPT_DIR / "eval_pi3_videodepth_protocol.py",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified PI3-aligned depth benchmark entrypoint (Sintel, KITTI, Bonn, ETH3D, 7scenes, BlendedMVS, MVS Synth, ScanNet++, VKITTI2, CO3Dv2, DIODE). "
            "Use --mode to switch between monocular depth and video depth."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python aidi/scripts/baselines/eval_pi3_depth_protocol.py \\\n"
            "    --mode monodepth --dataset sintel --model-family pi3 --data-root /path/to/sintel/training\n\n"
            "  python aidi/scripts/baselines/eval_pi3_depth_protocol.py \\\n"
            "    --mode videodepth --dataset kitti --model-family vggt --vggt-model-tag pt34 \\\n"
            "    --data-root /path/to/kitti/depth_selection/val_selection_cropped --eval-device cpu\n\n"
            "Pass all other arguments exactly as you would to the underlying mode-specific script."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_TO_SCRIPT.keys()),
        required=True,
        help="Depth evaluation mode: `monodepth` for PI3 Table 6, `videodepth` for PI3 Table 4.",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the resolved underlying command before execution.",
    )
    parser.add_argument(
        "--mode-help",
        action="store_true",
        help="Show the original help message of the selected mode-specific script and exit.",
    )
    return parser


def forward_args(argv: List[str]) -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args(argv)
    target = MODE_TO_SCRIPT[args.mode]
    if not target.is_file():
        raise FileNotFoundError(f"Target protocol script not found for mode={args.mode}: {target}")

    env = dict(os.environ)
    pythonpath_parts = [str(REPO_ROOT)]
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    if args.mode_help:
        cmd = [sys.executable, str(target), "--help"]
        return int(subprocess.run(cmd, env=env, cwd=str(REPO_ROOT)).returncode)

    cmd = [sys.executable, str(target), *passthrough]
    if args.print_command:
        print("Resolved command:")
        print(" ".join(cmd))

    completed = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
    return int(completed.returncode)


def main() -> None:
    raise SystemExit(forward_args(sys.argv[1:]))


if __name__ == "__main__":
    main()
