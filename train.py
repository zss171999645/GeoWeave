#!/usr/bin/env python3
"""Launch the original GeoWeave trainers locally, without a cluster account."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['pi3', 'vggt'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', type=int, default=1, help='Number of local GPU processes')
    parser.add_argument('--stage', choices=['warmup', 'sparse'], default='sparse', help='Pi3 training stage')
    parser.add_argument('--data-root', type=Path, help='Pi3: processed TarTanAir root')
    parser.add_argument('--config', type=Path, help='VGGT: complete training YAML with your data paths')
    parser.add_argument('--print-command', action='store_true')
    parser.add_argument('overrides', nargs=argparse.REMAINDER, help='Trainer config overrides after --')
    args = parser.parse_args()
    if args.gpus < 1:
        parser.error('--gpus must be positive')
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint not found: {args.checkpoint}')
    extra = args.overrides[1:] if args.overrides[:1] == ['--'] else args.overrides
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.print_command:
        parser.error('Use a new, empty --output directory for a fresh training run')
    checkpoint = args.checkpoint.resolve()
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    env.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
    if args.model == 'pi3':
        if args.config:
            parser.error('--config is for VGGT; use Hydra overrides after -- for Pi3')
        if args.data_root is None or not args.data_root.is_dir():
            parser.error('Pi3 requires --data-root pointing to processed TarTanAir data')
        cwd = ROOT / 'aidi/third_party/pi3_training'
        env['TARTANAIR_ROOT'] = str(args.data_root.resolve())
        command = [sys.executable, '-m', 'accelerate.commands.launch', '--num_processes', str(args.gpus),
                   '--num_machines', '1', '--mixed_precision', 'bf16', '--dynamo_backend', 'no']
        if args.gpus > 1:
            command += ['--multi_gpu']
        command += ['scripts/train_pi3.py', f'train=train_pi3_lowres_indexer_{args.stage}',
                    'data=public_tartanair', 'model.load_vggt=false', f'model.ckpt={checkpoint}',
                    f'work_dir={output}', f'name=geoweave_pi3_{args.stage}', 'log.use_wandb=false',
                    'train.auto_resume=false', *extra]
    else:
        if args.config is None or not args.config.is_file():
            parser.error('VGGT requires --config with local data paths; see docs/TRAINING.md')
        if args.data_root:
            parser.error('Set VGGT data paths in --config, not --data-root')
        cwd = ROOT
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   '--nproc_per_node', str(args.gpus), '-m', 'easyvolcap.scripts.main',
                   '-t', 'train', '-c', str(args.config.resolve()),
                   f'runner_cfg.trained_model={output}/checkpoints',
                   f'runner_cfg.recorder_cfg.record_dir={output}/logs',
                   f'runner_cfg.pretrained_model={checkpoint}',
                   'runner_cfg.resume=False', 'runner_cfg.pretrained_load_training_state=False', *extra,
                   f'distributed={args.gpus > 1}', 'accelerated=False']
    print('Working directory:', cwd, flush=True)
    print(shlex.join(command), flush=True)
    if not args.print_command:
        output.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, cwd=cwd, env=env, check=True)


if __name__ == '__main__':
    main()
