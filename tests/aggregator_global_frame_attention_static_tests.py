import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AGGREGATOR_PATH = REPO_ROOT / "easyvolcap/official_vggt/models/aggregator.py"
GLOBAL_FRAME_EVAL_CONFIG = (
    REPO_ROOT
    / "aidi/configs/vggt/finetune_5090_sparse_20260329_resumept30_oldlogic_keepdecay_ep80_2x8_chunk2_topk1024_tbfix_v1_globalframe0_8_eval_record.yaml"
)
GLOBAL_FRAME_0_8_20_23_EVAL_CONFIG = (
    REPO_ROOT
    / "aidi/configs/vggt/finetune_5090_sparse_20260329_resumept30_oldlogic_keepdecay_ep80_2x8_chunk2_topk1024_tbfix_v1_globalframe0_8_20_23_eval_record.yaml"
)
OFFICIAL_GLOBAL_FRAME_0_8_EVAL_CONFIG = (
    REPO_ROOT / "aidi/configs/vggt/vggt_official_eval_paper_globalframe0_8_eval_record.yaml"
)


def _aggregator_ast():
    return ast.parse(AGGREGATOR_PATH.read_text(encoding="utf-8"))


def _find_method(tree, class_name, method_name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"{class_name}.{method_name} not found")


def _load_static_method(method_name):
    method_node = _find_method(_aggregator_ast(), "Aggregator", method_name)
    method_node.decorator_list = []
    module = ast.Module(body=[method_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(AGGREGATOR_PATH), "exec"), namespace)
    return namespace[method_name]


def test_aggregator_exposes_global_frame_attention_layer_config():
    init_node = _find_method(_aggregator_ast(), "Aggregator", "__init__")
    arg_names = [arg.arg for arg in init_node.args.args]
    assert "global_frame_attention_layers" in arg_names


def test_process_global_attention_has_frame_local_branch():
    process_node = _find_method(_aggregator_ast(), "Aggregator", "_process_global_attention")
    names = {node.id for node in ast.walk(process_node) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(process_node) if isinstance(node, ast.Attribute)}
    assert "use_frame_range" in names
    assert "global_frame_attention_layers" in attrs


def test_global_frame_eval_config_uses_runtime_base_key():
    text = GLOBAL_FRAME_EVAL_CONFIG.read_text(encoding="utf-8")
    assert "_base_" not in text
    assert "configs:" in text
    assert "finetune_5090_sparse_topk2048_l9_19_p34_record.yaml" in text
    assert 'global_frame_attention_layers: "0-8"' in text


def test_global_frame_eval_config_can_move_final_global_layers_to_frame_attention():
    text = GLOBAL_FRAME_0_8_20_23_EVAL_CONFIG.read_text(encoding="utf-8")
    assert "_base_" not in text
    assert "configs:" in text
    assert "finetune_5090_sparse_topk2048_l9_19_p34_record.yaml" in text
    assert 'global_frame_attention_layers: "0-8,20-23"' in text


def test_official_global_frame_eval_config_keeps_official_base_without_sparse_indexer():
    text = OFFICIAL_GLOBAL_FRAME_0_8_EVAL_CONFIG.read_text(encoding="utf-8")
    assert "_base_" not in text
    assert "configs:" in text
    assert "configs/exps/vggt/vggt_official_eval_paper.yaml" in text
    assert "finetune_5090_sparse_topk2048_l9_19_p34_record.yaml" not in text
    assert "indexer_cfg" not in text
    assert 'global_frame_attention_layers: "0-8"' in text


def test_indexer_layer_parser_accepts_comma_separated_ranges():
    parse_indexer_layers = _load_static_method("_parse_indexer_layers")
    assert parse_indexer_layers("0-8,20-23") == {
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        20,
        21,
        22,
        23,
    }
