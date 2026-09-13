import torch
from torch import nn

from easyvolcap.models.official_vggt_model import OfficialVGGTModel


class DummyAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Linear(3, 2)
        self.soft_view_bias_q_proj = nn.Linear(3, 1)
        self.soft_view_bias_k_proj = nn.Linear(3, 1)
        self.soft_view_bias_scale = nn.Parameter(torch.tensor(0.0))


def test_load_module_ckpt_allows_missing_soft_view_bias_params_for_legacy_aggregator(tmp_path):
    module = DummyAggregator()
    legacy = DummyAggregator()
    ckpt_path = tmp_path / "legacy_aggregator.pt"
    torch.save({"model": {"core.weight": legacy.core.weight.detach().clone(),
                          "core.bias": legacy.core.bias.detach().clone()}}, ckpt_path)

    OfficialVGGTModel._load_module_ckpt(module, str(ckpt_path), strict=True, name="aggregator")

    assert torch.allclose(module.core.weight, legacy.core.weight)
    assert torch.allclose(module.core.bias, legacy.core.bias)


def test_load_module_ckpt_still_rejects_missing_non_soft_view_bias_params_for_aggregator(tmp_path):
    module = DummyAggregator()
    ckpt_path = tmp_path / "broken_aggregator.pt"
    torch.save({"model": {}}, ckpt_path)

    try:
        OfficialVGGTModel._load_module_ckpt(module, str(ckpt_path), strict=True, name="aggregator")
    except RuntimeError as exc:
        assert "core.weight" in str(exc)
    else:
        raise AssertionError("expected missing non-soft-view-bias parameters to remain fatal")


def test_remap_special_token_keys_keeps_new_format_keys_unchanged():
    state_dict = {
        "vggt.aggregator.special_tokens.camera_token": torch.randn(1, 2, 1, 4),
        "vggt.aggregator.special_tokens.register_token": torch.randn(1, 2, 4, 4),
        "vggt.aggregator.patch_embed.special_tokens.register_tokens": torch.randn(1, 4, 4),
    }

    remapped, changed = OfficialVGGTModel._remap_special_token_keys(state_dict)

    assert changed is False
    assert "vggt.aggregator.special_tokens.camera_token" in remapped
    assert "vggt.aggregator.special_tokens.register_token" in remapped
    assert "vggt.aggregator.patch_embed.special_tokens.register_tokens" in remapped
    assert "vggt.aggregator.special_tokens.special_tokens.camera_token" not in remapped
    assert "vggt.aggregator.special_tokens.special_tokens.register_token" not in remapped


def test_remap_special_token_keys_maps_legacy_suffixes_once():
    state_dict = {
        "vggt.aggregator.camera_token": torch.randn(1, 2, 1, 4),
        "vggt.aggregator.register_token": torch.randn(1, 2, 4, 4),
        "vggt.aggregator.patch_embed.register_tokens": torch.randn(1, 4, 4),
    }

    remapped, changed = OfficialVGGTModel._remap_special_token_keys(state_dict)

    assert changed is True
    assert "vggt.aggregator.special_tokens.camera_token" in remapped
    assert "vggt.aggregator.special_tokens.register_token" in remapped
    assert "vggt.aggregator.patch_embed.special_tokens.register_tokens" in remapped
    assert "vggt.aggregator.camera_token" not in remapped
    assert "vggt.aggregator.register_token" not in remapped
    assert "vggt.aggregator.patch_embed.register_tokens" not in remapped
