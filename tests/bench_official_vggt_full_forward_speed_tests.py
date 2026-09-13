from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / 'aidi/scripts/vggt/bench_official_vggt_full_forward_speed.py'


def load_module():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = spec_from_file_location('bench_official_vggt_full_forward_speed', SCRIPT_PATH)
    assert spec is not None and spec.loader is not None, 'failed to load benchmark module spec'
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_build_model_cfg_qk_all_layers():
    module = load_module()
    cfg = module.build_model_cfg(
        str(REPO_ROOT / 'aidi/configs/vggt/finetune_5090_sparse_topk2048_l9_19_p34_record.yaml'),
        mode='qk',
        topk=2048,
        all_layers=True,
    )
    indexer_cfg = cfg['vggt_cfg']['indexer_cfg']
    assert indexer_cfg['enabled'] is True
    assert indexer_cfg['enable_sparse'] is True
    assert indexer_cfg['inference_sparse'] is True
    assert indexer_cfg['topk'] == 2048
    assert indexer_cfg['indexer_layers'] is None
    assert indexer_cfg['source_downsample_enabled'] is True
    assert indexer_cfg['source_downsample_factor'] == 2
    assert indexer_cfg['source_downsample_strategy'] == 'qk_sym_x2_broadcast'


def test_build_model_cfg_dense_disables_sparse():
    module = load_module()
    cfg = module.build_model_cfg(
        str(REPO_ROOT / 'aidi/configs/vggt/finetune_5090_sparse_topk2048_l9_19_p34_record.yaml'),
        mode='dense',
        topk=2048,
        all_layers=False,
    )
    indexer_cfg = cfg['vggt_cfg']['indexer_cfg']
    assert indexer_cfg['enabled'] is False
    assert indexer_cfg['enable_sparse'] is False
    assert indexer_cfg['inference_sparse'] is False


if __name__ == '__main__':
    test_build_model_cfg_qk_all_layers()
    test_build_model_cfg_dense_disables_sparse()
    print('bench_official_vggt_full_forward_speed_tests passed')
