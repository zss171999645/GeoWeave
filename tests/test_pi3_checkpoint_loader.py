from pathlib import Path

from aidi.scripts.baselines.pi3_checkpoint_loader import (
    build_native_pi3_kwargs,
    find_hydra_config_for_checkpoint,
    resolve_pi3_eval_options,
)


def test_resolves_native_config_from_checkpoint_tree(tmp_path: Path):
    output_root = tmp_path / "outputs" / "run_a"
    hydra_dir = output_root / ".hydra"
    ckpt_dir = output_root / "ckpts" / "checkpoint_44"
    hydra_dir.mkdir(parents=True)
    ckpt_dir.mkdir(parents=True)
    config_path = hydra_dir / "config.yaml"
    ckpt_path = ckpt_dir / "pytorch_model.bin"
    config_path.write_text(
        "model:\n"
        "  _target_: pi3.models.pi3_training.Pi3\n"
        "  decoder_size: large\n"
        "  ckpt: old.pt\n"
        "  indexer_cfg:\n"
        "    enabled: true\n"
        "    indexer_layers: 9-17\n",
        encoding="utf-8",
    )
    ckpt_path.write_bytes(b"placeholder")

    assert find_hydra_config_for_checkpoint(ckpt_path) == config_path
    impl, resolved_config = resolve_pi3_eval_options(str(ckpt_path))

    assert impl == "native_sparse"
    assert resolved_config == config_path


def test_native_kwargs_drop_training_load_checkpoint(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model:\n"
        "  _target_: pi3.models.pi3_training.Pi3\n"
        "  decoder_size: large\n"
        "  load_vggt: false\n"
        "  ckpt: warmup.pt\n"
        "  indexer_init_ckpt: indexer.pt\n"
        "  head_view_chunk_size: 4\n"
        "  indexer_cfg:\n"
        "    enabled: true\n"
        "    topk: 1024\n",
        encoding="utf-8",
    )

    kwargs = build_native_pi3_kwargs(config_path)

    assert kwargs["decoder_size"] == "large"
    assert kwargs["head_view_chunk_size"] == 4
    assert kwargs["indexer_cfg"]["topk"] == 1024
    assert "ckpt" not in kwargs
    assert "indexer_init_ckpt" not in kwargs
