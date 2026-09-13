import unittest
import json
from pathlib import Path

import numpy as np
import yaml

from aidi.scripts.baselines.pi3_metric_utils import accuracy, completion, umeyama
from aidi.utils.vggt_core4_main_val import (
    CO3DV2_RAW_ROOT_DEFAULT,
    extract_core4_tb_scalars,
    _merge_co3dv2_payloads,
    _merge_mv_recon_payloads,
    _merge_re10k_payloads,
    _shard_by_rank,
    normalize_core4_main_val_cfg,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CFG = REPO_ROOT / "configs/exps/vggt/vggt_official_finetune_5090.yaml"
EXPECTED_RE10K_ROOT = "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/datasets/re10k/processed_pose1800_clusterfix/test"
EXPECTED_CO3DV2_IMAGE_ROOT = CO3DV2_RAW_ROOT_DEFAULT
LEGACY_CO3DV2_SETLIST_MIRROR_ROOT = (
    "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/"
    "eval_datasets/co3d_official_image_mirror_co3d_setlists_seen41_test_full"
)
EXPECTED_CO3DV2_SETLIST_ROOT = (
    "/horizon-bucket/saturn_v_4dlabel/009_geo/002_data/dust3r_datasets/dust3r_extracted/co3dv2"
)
EXPECTED_DTU_ROOT = (
    "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/"
    "eval_datasets/dtu_test_mvsnet_release_full"
)
EXPECTED_ETH3D_ROOT = (
    "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/"
    "eval_datasets/eth3d_pi3_style_root"
)
EXPECTED_ETH3D_SEQMAP = (
    "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/"
    "eval_datasets/eth3d_pi3_style_seqmap.json"
)
DTU_SEQ_ASSET = REPO_ROOT / "aidi/assets/pi3_seq_id_maps/DTU_mv-recon_seq-id-map-kf5.json"
ETH3D_SEQ_ASSET = REPO_ROOT / "aidi/assets/pi3_seq_id_maps/ETH3D_mv-recon_seq-id-map-kf5.json"


def load_yaml(path: Path):
    text = path.read_text(encoding="utf-8").replace("{{fileBasenameNoExtension}}", '"fileBasenameNoExtension"')
    return yaml.safe_load(text)


class VggtCore4MainValTests(unittest.TestCase):
    def test_5090_train_config_uses_core4_main_val_cfg(self):
        config = load_yaml(TRAIN_CFG)
        runner_cfg = config["runner_cfg"]
        self.assertNotIn("main_val_cfgs", runner_cfg)

        core4_cfg = runner_cfg["main_val_core4_cfg"]
        datasets = core4_cfg["datasets"]

        self.assertEqual(core4_cfg["run"]["name"], "vggt_official_train_core4_main_val")
        self.assertEqual(list(datasets.keys()), ["re10k", "co3dv2", "dtu", "eth3d"])
        self.assertTrue(datasets["re10k"]["enabled"])
        self.assertEqual(datasets["re10k"]["root"], EXPECTED_RE10K_ROOT)
        self.assertEqual(datasets["re10k"]["sample_stride"], 10)
        self.assertTrue(datasets["co3dv2"]["enabled"])
        self.assertEqual(datasets["co3dv2"]["image_root"], EXPECTED_CO3DV2_IMAGE_ROOT)
        self.assertEqual(datasets["co3dv2"]["setlist_root"], EXPECTED_CO3DV2_SETLIST_ROOT)
        self.assertEqual(datasets["co3dv2"]["selection_source"], "co3d_setlists")
        self.assertEqual(datasets["co3dv2"]["subset"], "seen41")
        self.assertEqual(datasets["co3dv2"]["sample_stride"], 10)
        self.assertTrue(datasets["dtu"]["enabled"])
        self.assertEqual(datasets["dtu"]["dataset_root"], EXPECTED_DTU_ROOT)
        self.assertTrue(datasets["eth3d"]["enabled"])
        self.assertEqual(datasets["eth3d"]["dataset_root"], EXPECTED_ETH3D_ROOT)
        self.assertEqual(datasets["eth3d"]["seq_map"], EXPECTED_ETH3D_SEQMAP)
        self.assertEqual(runner_cfg["extra_eval_cfgs"], [])
        self.assertEqual(runner_cfg["extra_eval_every"], 0)

    def test_normalize_core4_main_val_cfg_sets_stride_defaults_for_pose_datasets(self):
        normalized = normalize_core4_main_val_cfg(
            {
                "run": {"name": "unit"},
                "datasets": {
                    "re10k": {"enabled": True, "root": EXPECTED_RE10K_ROOT},
                    "co3dv2": {
                        "enabled": True,
                        "image_root": EXPECTED_CO3DV2_IMAGE_ROOT,
                        "setlist_root": EXPECTED_CO3DV2_SETLIST_ROOT,
                    },
                },
            },
            repo_root=REPO_ROOT,
        )

        self.assertEqual(normalized["datasets"]["re10k"]["sample_stride"], 1)
        self.assertEqual(normalized["datasets"]["co3dv2"]["sample_stride"], 1)

    def test_normalize_core4_main_val_cfg_switches_hf_jgz_to_raw_co3d_root(self):
        normalized = normalize_core4_main_val_cfg(
            {
                "run": {"name": "unit"},
                "datasets": {
                    "co3dv2": {
                        "enabled": True,
                        "image_root": LEGACY_CO3DV2_SETLIST_MIRROR_ROOT,
                        "selection_source": "hf_jgz",
                        "anno_dir": "/tmp/fake_hf_jgz_annos",
                    },
                },
            },
            repo_root=REPO_ROOT,
        )

        self.assertEqual(normalized["datasets"]["co3dv2"]["image_root"], CO3DV2_RAW_ROOT_DEFAULT)

    def test_normalize_core4_main_val_cfg_switches_setlist_mirror_to_raw_co3d_root(self):
        normalized = normalize_core4_main_val_cfg(
            {
                "run": {"name": "unit"},
                "datasets": {
                    "co3dv2": {
                        "enabled": True,
                        "image_root": LEGACY_CO3DV2_SETLIST_MIRROR_ROOT,
                        "selection_source": "co3d_setlists",
                        "setlist_root": EXPECTED_CO3DV2_SETLIST_ROOT,
                    },
                },
            },
            repo_root=REPO_ROOT,
        )

        self.assertEqual(normalized["datasets"]["co3dv2"]["image_root"], CO3DV2_RAW_ROOT_DEFAULT)

    def test_normalize_core4_main_val_cfg_switches_annotation_mirror_to_raw_co3d_root(self):
        normalized = normalize_core4_main_val_cfg(
            {
                "run": {"name": "unit"},
                "datasets": {
                    "co3dv2": {
                        "enabled": True,
                        "image_root": LEGACY_CO3DV2_SETLIST_MIRROR_ROOT,
                        "selection_source": "co3d_annotations",
                        "anno_dir": f"{LEGACY_CO3DV2_SETLIST_MIRROR_ROOT}/_meta/annos",
                    },
                },
            },
            repo_root=REPO_ROOT,
        )

        self.assertEqual(normalized["datasets"]["co3dv2"]["image_root"], CO3DV2_RAW_ROOT_DEFAULT)

    def test_extract_core4_tb_scalars_handles_pose_and_mv_payloads(self):
        pose_scalars = extract_core4_tb_scalars(
            "re10k",
            {
                "summary": {
                    "pose_auc": 0.42,
                    "num_sequences": 12,
                    "note": "ignored",
                }
            },
        )
        mv_scalars = extract_core4_tb_scalars(
            "dtu",
            {
                "protocols": {
                    "dtu-kf5": {
                        "metrics": {
                            "acc": 1.2,
                            "comp": 2.3,
                            "status": "ignored",
                        }
                    }
                }
            },
        )

        self.assertEqual(pose_scalars, {"pose_auc": 0.42, "num_sequences": 12.0})
        self.assertEqual(mv_scalars, {"dtu-kf5/acc": 1.2, "dtu-kf5/comp": 2.3})

    def test_shard_by_rank_partitions_without_overlap(self):
        items = list(range(10))
        shards = [_shard_by_rank(items, rank=rank, world_size=3) for rank in range(3)]

        self.assertEqual(shards[0], [0, 3, 6, 9])
        self.assertEqual(shards[1], [1, 4, 7])
        self.assertEqual(shards[2], [2, 5, 8])
        self.assertEqual(sorted(item for shard in shards for item in shard), items)

    def test_merge_re10k_payloads_recomputes_summary_from_all_rank_metrics(self):
        payload = _merge_re10k_payloads(
            [
                {
                    "model_tag": "train_main_val",
                    "metrics": [
                        {"cam:pose_auc_10": 0.1, "cam:pose_auc_20": 0.2, "cam:pose_auc_30": 0.3},
                    ],
                    "failures": [{"path": "a", "error": "boom"}],
                },
                {
                    "model_tag": "train_main_val",
                    "metrics": [
                        {"cam:pose_auc_10": 0.3, "cam:pose_auc_20": 0.4, "cam:pose_auc_30": 0.5},
                    ],
                    "failures": [],
                },
            ]
        )

        self.assertEqual(payload["summary"]["metrics_count"], 2)
        self.assertAlmostEqual(payload["summary"]["cam:pose_auc_10_mean"], 0.2)
        self.assertAlmostEqual(payload["summary"]["cam:pose_auc_20_mean"], 0.3)
        self.assertAlmostEqual(payload["summary"]["cam:pose_auc_30_mean"], 0.4)
        self.assertEqual(len(payload["metrics"]), 2)
        self.assertEqual(len(payload["failures"]), 1)

    def test_merge_co3dv2_payloads_recomputes_category_and_overall_metrics(self):
        seq_a = {
            "category": "apple",
            "sequence_name": "seq_a",
            "r_error": [1.0, 2.0],
            "t_error": [1.0, 3.0],
        }
        seq_b = {
            "category": "apple",
            "sequence_name": "seq_b",
            "r_error": [4.0, 6.0],
            "t_error": [5.0, 7.0],
        }

        payload = _merge_co3dv2_payloads(
            [
                {
                    "implementation": "vendored_upstream_test_co3d",
                    "categories": ["apple"],
                    "metrics": [seq_a],
                    "per_sequence_results": {"apple": [seq_a]},
                    "failures": [],
                },
                {
                    "implementation": "vendored_upstream_test_co3d",
                    "categories": ["apple"],
                    "metrics": [seq_b],
                    "per_sequence_results": {"apple": [seq_b]},
                    "failures": [{"path": "apple/bad", "error": "bad"}],
                },
            ]
        )

        self.assertEqual(payload["summary"]["num_categories"], 1)
        self.assertEqual(payload["summary"]["num_sequences"], 2)
        self.assertEqual(payload["per_category_results"]["apple"]["num_sequences"], 2)
        self.assertEqual(len(payload["metrics"]), 2)
        self.assertEqual(len(payload["failures"]), 1)

    def test_merge_mv_recon_payloads_weight_averages_rank_metrics(self):
        payload = _merge_mv_recon_payloads(
            "dtu",
            [
                {
                    "dataset": "dtu",
                    "protocols": {
                        "DTU": {
                            "metrics": {"Acc-mean": 1.0, "Comp-mean": 2.0, "num_sequences": 1, "num_skipped": 1},
                            "num_seq_in_map": 2,
                            "skipped": [{"seq": "scan1", "reason": "missing"}],
                        }
                    },
                },
                {
                    "dataset": "dtu",
                    "protocols": {
                        "DTU": {
                            "metrics": {"Acc-mean": 3.0, "Comp-mean": 4.0, "num_sequences": 3, "num_skipped": 0},
                            "num_seq_in_map": 3,
                            "skipped": [],
                        }
                    },
                },
            ],
        )

        metrics = payload["protocols"]["DTU"]["metrics"]
        self.assertEqual(metrics["num_sequences"], 4)
        self.assertEqual(metrics["num_skipped"], 1)
        self.assertAlmostEqual(metrics["Acc-mean"], 2.5)
        self.assertAlmostEqual(metrics["Comp-mean"], 3.5)
        self.assertEqual(payload["protocols"]["DTU"]["num_seq_in_map"], 5)
        self.assertEqual(len(payload["protocols"]["DTU"]["skipped"]), 1)

    def test_pi3_metric_utils_matches_expected_shapes(self):
        src = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ).T
        scale = 2.0
        rotation = np.eye(3, dtype=np.float64)
        translation = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        target = scale * rotation @ src + translation

        c, R, t = umeyama(src, target)
        self.assertAlmostEqual(c, scale)
        np.testing.assert_allclose(R, rotation, atol=1e-6)
        np.testing.assert_allclose(t, translation, atol=1e-6)

        gt_points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
        rec_points = gt_points.copy()
        gt_normals = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        rec_normals = gt_normals.copy()

        acc, acc_median, nc1, nc1_median = accuracy(gt_points, rec_points, gt_normals, rec_normals)
        comp, comp_median, nc2, nc2_median = completion(gt_points, rec_points, gt_normals, rec_normals)
        self.assertEqual((acc, acc_median, comp, comp_median), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual((nc1, nc1_median, nc2, nc2_median), (1.0, 1.0, 1.0, 1.0))

    def test_mv_recon_seq_map_assets_exist(self):
        dtu = json.loads(DTU_SEQ_ASSET.read_text(encoding="utf-8"))
        eth3d = json.loads(ETH3D_SEQ_ASSET.read_text(encoding="utf-8"))
        self.assertIn("scan1", dtu)
        self.assertEqual(dtu["scan1"][:3], [0, 5, 10])
        self.assertIn("courtyard", eth3d)
        self.assertEqual(eth3d["courtyard"][:3], [0, 5, 10])


if __name__ == "__main__":
    unittest.main()
