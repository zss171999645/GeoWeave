"""
This file may compare metrics on mesh, point and volume data
"""
import math
import torch
import numpy as np

from easyvolcap.engine import cfg
from easyvolcap.engine import EVALUATORS
from easyvolcap.runners.evaluators.volumetric_video_evaluator import VolumetricVideoEvaluator, _filter_nonfinite_metrics

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.json_utils import serialize
from easyvolcap.utils.metric_utils import (
    Metrics,
    chamfer_distance,
    distance,
    distance_accuracy,
    camera_accuracy_auc,
)
from easyvolcap.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri
from easyvolcap.utils.vggt.utils.geometry import unproject_depth_map_to_point_map


@EVALUATORS.register_module()
class GeometryEvaluator:
    def __init__(self,
                 skip_time_in_summary: int = 0,  # skip first 5 image in summary
                 result_dir: str = cfg.runner_cfg.visualizer_cfg.result_dir,  # MARK: GLOBAL
                 metrics_file: str = 'metrics.json',
                 compute_cam_metrics: List[str] = ['CAM_ACC_AUC'],
                 compute_dpt_metrics: List[str] = ['DPT'],
                 compute_xyz_metrics: List[str] = ['XYZ'],
                 force_xyz_from_depth: bool = False,
                 fixscale: float = None,
                 add_loss_to_metrics: bool = False,
                 print_summary: bool = True,
                 print_save_path: bool = True,
                 ) -> None:
        self.skip_time_in_summary = skip_time_in_summary
        result_dir = join(result_dir, cfg.exp_name)  # MARK: global configuration
        os.makedirs(result_dir, exist_ok=True)
        self.result_dir = result_dir
        self.metrics = []
        self.metrics_file = metrics_file
        self.compute_xyz_metrics = [getattr(Metrics, m) for m in compute_xyz_metrics]
        self.compute_dpt_metrics = [getattr(Metrics, m) for m in compute_dpt_metrics]
        self.compute_cam_metrics = [getattr(Metrics, m) for m in compute_cam_metrics]
        self.force_xyz_from_depth = force_xyz_from_depth
        self.fixscale = fixscale
        self.add_loss_to_metrics = add_loss_to_metrics
        self.print_summary = print_summary
        self.print_save_path = print_save_path
        self._warned_xyz_from_depth = False

    def _maybe_build_xyz_from_depth(self, output: dotdict, batch: dotdict) -> None:
        if 'xyz_map' in output:
            return
        if not self.compute_xyz_metrics:
            return
        if 'dpt_map' not in output:
            return
        if not hasattr(batch, 'meta') or not hasattr(batch.meta, 'H') or not hasattr(batch.meta, 'W'):
            return

        try:
            H_meta = int(batch.meta.H[0].item())
            W_meta = int(batch.meta.W[0].item())
        except Exception:
            return

        depth = output.dpt_map
        if depth is None or depth.dim() != 4:
            return

        B, S, P, C = depth.shape
        if C != 1:
            return
        # 与 OfficialVGGTModel 一致：深度/相机分辨率以 dpt 展平长度为准（常为 518×518），
        # 勿用数据集原始分辨率解码 pose，否则与 depth 网格错位。
        if H_meta * W_meta == P:
            H, W = H_meta, W_meta
        else:
            s = int(round(math.sqrt(P)))
            if s * s != P:
                return
            H, W = s, s

        if 'cam_map' in output:
            pred_w2c, pred_ixt = pose_encoding_to_extri_intri(
                output.cam_map, image_size_hw=(H, W)
            )
            extrinsics = pred_w2c
            intrinsics = pred_ixt
        else:
            if hasattr(batch, 'w2cs'):
                extrinsics = batch.w2cs[..., :3, :4]
            elif hasattr(batch, 'R') and hasattr(batch, 'T'):
                extrinsics = torch.cat([batch.R, batch.T], dim=-1)
            else:
                return
            if hasattr(batch, 'ixts'):
                intrinsics = batch.ixts
            elif hasattr(batch, 'K'):
                intrinsics = batch.K
            else:
                return

        depth = depth.reshape(B, S, H, W, 1)
        world_points = []
        for b in range(B):
            pts = unproject_depth_map_to_point_map(depth[b], extrinsics[b], intrinsics[b])
            world_points.append(pts)
        world_points = np.stack(world_points, axis=0)
        xyz_map = torch.from_numpy(world_points).to(device=depth.device, dtype=depth.dtype)
        output.xyz_map = xyz_map.reshape(B, S, -1, 3)

        if not self._warned_xyz_from_depth:
            log(yellow('GeometryEvaluator: xyz_map missing, built from depth+camera for point metrics.'))
            self._warned_xyz_from_depth = True

    def evaluate(self, output: dotdict, batch: dotdict, printlog: bool = False):
        metrics = dotdict()
        def _to_scalar(value):
            if isinstance(value, torch.Tensor):
                return value.mean().item()
            if isinstance(value, (int, float)):
                return float(value)
            if hasattr(value, "mean"):
                try:
                    return float(value.mean())
                except Exception:
                    pass
            return float(value)

        def record_metric(compute, m, prefix: str = ''):
            if isinstance(m, torch.Tensor):
                metrics[f'{prefix}:{compute.__name__}'] = m.mean().item()
            elif isinstance(m, dict):
                for k, v in m.items():
                    metrics[f'{prefix}:{k}'] = _to_scalar(v)

        if 'cam_map' in output and 'cam' in batch and batch.msk.sum() > 0:
            cam = batch.cam.clone()  # (B, S, 9)
            cam_map = output.cam_map.clone()  # (B, S, 9)
            cam_map = torch.nan_to_num(cam_map, nan=0.0, posinf=0.0, neginf=0.0)  # avoid error

            # Compute the metrics
            for compute in self.compute_cam_metrics:
                m = compute(cam, cam_map, batch)
                record_metric(compute, m, 'cam')

        if 'dpt_map' in output and 'dpt' in batch and batch.msk.sum() > 0:
            dpt = batch.dpt.clone()  # (B, S, P, 1)
            dpt_map = output.dpt_map.clone()  # (B, S, P, 1)
            dpt_map = torch.nan_to_num(dpt_map, nan=0.0, posinf=0.0, neginf=0.0)  # avoid error
            msk = batch.msk.clone()  # (B, S, P, 1)

            # NOTE: for evaluation, there should not be a batch dimension
            assert dpt.shape[0] == 1 and dpt_map.shape[0] == 1 and msk.shape[0] == 1, (
                'For evaluation, there should not be a batch dimension'
            )
            dpt = dpt[0]
            dpt_map = dpt_map[0]
            msk = msk[0]

            # Compute the metrics
            for compute in self.compute_dpt_metrics:
                m = compute(dpt_map, dpt, msk, batch, self.fixscale)
                record_metric(compute, m, 'dpt')

        # Process the output and batch
        if self.force_xyz_from_depth and 'xyz_map' in output:
            output = dotdict(output.copy())
            output.pop('xyz_map', None)
        if 'xyz_map' not in output:
            self._maybe_build_xyz_from_depth(output, batch)

        if 'xyz_map' in output and 'xyz' in batch and batch.msk.sum() > 0:
            # Get the ground truth xyz and the predicted xyz
            xyz = batch.xyz.clone()  # (B, S, P, 3)
            xyz_map = output.xyz_map.clone()  # (B, S, P, 3)
            xyz_map = torch.nan_to_num(xyz_map, nan=0.0, posinf=0.0, neginf=0.0)  # avoid error
            msk = batch.msk.clone()  # (B, S, P, 1)

            # NOTE: for evaluation, there should not be a batch dimension
            assert xyz.shape[0] == 1 and xyz_map.shape[0] == 1 and msk.shape[0] == 1, (
                'For evaluation, there should not be a batch dimension'
            )
            xyz = xyz[0]
            xyz_map = xyz_map[0]
            msk = msk[0]

            # Compute the metrics
            for compute in self.compute_xyz_metrics:
                m = compute(xyz_map, xyz, msk, batch)
                record_metric(compute, m, 'xyz')

        if self.add_loss_to_metrics and 'scalar_stats' in output:
            for k, v in output.scalar_stats.items():
                metrics[f'training:{k}'] = _to_scalar(v)

        if len(metrics):
            # Record the iteration for logging
            self.iter = batch.meta.iter
            self.metrics.append(metrics)

            # For recording
            c = batch.meta.camera_index.item()
            f = batch.meta.frame_index.item()
            if printlog: log(f'camera: {c}', f'frame: {f}', metrics)
            metrics.camera = c
            metrics.frame = f
            # Record the scaling factor if available
            if 'scale' in batch:
                metrics.scale = batch.scale.item()
            metrics.path = batch.meta.data_root[0]
        elif self.compute_cam_metrics or self.compute_xyz_metrics or self.compute_dpt_metrics:
            # 个别 batch 无有效 mask / 缺输出时原逻辑不 append，DDP 下各 rank 条数不一致，synchronize 后少 1 条
            metrics = dotdict(_eval_skipped_no_valid_pixels=1.0)
            self.iter = batch.meta.iter
            self.metrics.append(metrics)
            c = batch.meta.camera_index.item()
            f = batch.meta.frame_index.item()
            metrics.camera = c
            metrics.frame = f
            if 'scale' in batch:
                metrics.scale = batch.scale.item()
            metrics.path = batch.meta.data_root[0]

        scalar_stats = dotdict({f'{k}_frame{f:04d}_cam{c:04d}': v for k, v in metrics.items()})
        return scalar_stats

    def synchronize_metrics(self, max_size: Optional[int] = None):
        VolumetricVideoEvaluator.synchronize_metrics(self, max_size=max_size)

    def presummary_log(self):
        """在 summarize 之前调用：打印各帧是否算出了 VGGT 相关 cam/dpt/xyz 指标，便于排查「只有 scale」的情况。"""
        if not self.metrics:
            log(yellow("[vggt-eval] presummary: metrics 为空"))
            return
        n = len(self.metrics)
        has_cam = sum(1 for m in self.metrics if any(str(k).startswith("cam:") for k in m))
        has_dpt = sum(1 for m in self.metrics if any(str(k).startswith("dpt:") for k in m))
        has_xyz = sum(1 for m in self.metrics if any(str(k).startswith("xyz:") for k in m))
        log(
            yellow("[vggt-eval] presummary: "),
            f"frames with cam/dpt/xyz metrics = {has_cam}/{has_dpt}/{has_xyz} (total {n})",
        )
        sample = None
        for m in self.metrics:
            if any(str(k).startswith("cam:") for k in m):
                sample = m
                break
        if sample is not None:
            ck = [k for k in sample.keys() if str(k).startswith("cam:")]
            log(yellow("[vggt-eval] one frame cam keys (sample): "), f"{ck[:16]}{'...' if len(ck) > 16 else ''}")
        else:
            log(
                yellow("[vggt-eval] "),
                "本 split 未写入 cam:*（需 batch.cam、output.cam_map 且 batch.msk 有效；Table2-only 常关 dpt/xyz）",
            )

    def summarize(self):
        summary = dotdict()
        dropped = []
        metrics = self.metrics
        n_metrics_raw = len(metrics)
        if len(metrics):
            metrics, dropped = _filter_nonfinite_metrics(metrics)
            if len(dropped) and self.print_summary:
                log(
                    yellow("[vggt-eval] nonfinite filter: "),
                    f"dropped {len(dropped)} / {n_metrics_raw} frames (fields in metrics json: dropped_nonfinite)",
                )
        if len(metrics):
            keys = set()
            for m in metrics:
                keys.update(m.keys())
            for key in keys:
                if key in metrics[0] and isinstance(metrics[0][key], list):
                    continue
                if key.startswith('_eval_'):
                    continue
                values = [m[key] for m in metrics if key in m]
                if not values:
                    continue
                if key == 'time':
                    if np.sum(values) == 0: continue  # timer has not been enabled
                    values = values[self.skip_time_in_summary:]
                    summary[f'{key}{self.skip_time_in_summary:}+_mean'] = np.mean(values).astype(float).item()
                    summary[f'{key}{self.skip_time_in_summary:}+_std'] = np.std(values).astype(float).item()
                elif key == 'camera':
                    pass
                elif key == 'frame':
                    pass
                elif key == 'path':
                    pass
                else:
                    summary[f'{key}_mean'] = np.mean(values).astype(float).item()
                    summary[f'{key}_std'] = np.std(values).astype(float).item()

        if len(summary) and self.print_summary:
            self._log_vggt_eval_digest(summary)
            log(summary)

        if len(metrics) or len(dropped):
            metric = dotdict()
            metric.summary = summary
            metric.metrics = metrics
            if len(dropped):
                metric.dropped_nonfinite = dropped
            metric_path = join(self.result_dir, f'{self.iter:08d}', self.metrics_file)
            try:
                if not exists(dirname(metric_path)):
                    os.makedirs(dirname(metric_path), exist_ok=True)
                with open(metric_path, 'w') as f:
                    # TODO: After finding out the offending object, we can remove the try-except block and serialize call
                    json.dump(serialize(metric), f, indent=4)
                if self.print_save_path:
                    log(yellow(f'Evaluation metrics saved to {blue(metric_path)}'))
            except Exception as e:
                log(red(f'Error in dumping evaluation metrics to {blue(metric_path)}: {e}'))

            self.metrics.clear()  # clear mean after extracting summary
        return summary

    def _log_vggt_eval_digest(self, summary: dotdict) -> None:
        """汇总结束后打印一行可读的 pose AUC / depth / point 摘要（与 metrics json 一致）。"""
        if not summary:
            return
        pose_aucs = []
        rot_acc = []
        tra_acc = []
        dpt_keys = []
        xyz_keys = []
        saco_inline = []
        ts_mean = None
        for k, v in summary.items():
            if not k.endswith("_mean"):
                continue
            base = k[: -len("_mean")]
            # 键形如 cam:pose_auc_30_mean → base 为 cam:pose_auc_30（无 CAM_ACC_AUC 段）
            if base.startswith("cam:") and "pose_auc_" in base:
                pose_aucs.append((base.split(":")[-1], float(v)))
            elif base.startswith("cam:") and "rotation_accuracy_" in base:
                rot_acc.append((base.split(":")[-1], float(v)))
            elif base.startswith("cam:") and "translation_accuracy_" in base:
                tra_acc.append((base.split(":")[-1], float(v)))
            elif base.startswith("cam:") and base.rsplit(":", 1)[-1] == "translation_scale":
                ts_mean = float(v)
            elif str(base).startswith("dpt:"):
                dpt_keys.append((base, float(v)))
            elif str(base).startswith("xyz:"):
                xyz_keys.append((base, float(v)))
                tail = base.rsplit(":", 1)[-1]
                if tail in ("accuracy", "completion", "overall"):
                    saco_inline.append((tail, float(v)))
        pose_aucs.sort(key=lambda x: x[0])
        if pose_aucs:
            auc_str = " ".join(f"{a}={b:.4f}" for a, b in pose_aucs)
            log(yellow("[vggt-eval] pose_auc (mean over frames): "), auc_str)
        if rot_acc:
            log(yellow("[vggt-eval] rotation_accuracy: "), " ".join(f"{a}={b:.4f}" for a, b in sorted(rot_acc)))
        if tra_acc:
            log(yellow("[vggt-eval] translation_accuracy: "), " ".join(f"{a}={b:.4f}" for a, b in sorted(tra_acc)))
        if ts_mean is not None:
            log(yellow("[vggt-eval] translation_scale_mean: "), f"{ts_mean:.6g}")
        if dpt_keys:
            show = dpt_keys[:6]
            log(
                yellow("[vggt-eval] dpt (sample): "),
                " ".join(f"{a.split(':')[-1]}={b:.6g}" for a, b in show),
            )
        if saco_inline:
            saco_inline.sort(key=lambda x: x[0])
            log(
                yellow("[vggt-eval] inline Acc/Comp/Overall (XYZ_SACO_NA，内联；离线毫米见 dtu_mast3r_chamfer_mm.json 或 sampleset 时 dtu_table2_mm.json): "),
                " ".join(f"{a}={b:.6g}" for a, b in saco_inline),
            )
        elif xyz_keys:
            show = xyz_keys[:8]
            log(
                yellow("[vggt-eval] xyz (sample): "),
                " ".join(f"{a}={b:.6g}" for a, b in show),
            )
        if not pose_aucs and not dpt_keys and not xyz_keys:
            log(
                yellow("[vggt-eval] digest: "),
                "无 cam pose_auc / dpt / xyz 的 mean（若仅有 scale_mean，请检查 batch.cam 与 compute_*_metrics）",
            )
