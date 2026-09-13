#!/usr/bin/env python3
"""Tests for the synthetic Pi3 forward/backward benchmark helpers."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

THIS_DIR = Path(__file__).resolve().parent
for candidate in (THIS_DIR, THIS_DIR.parent / "tools"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

import synthetic_pi3_training_time_benchmark as training_benchmark


def test_proxy_training_loss_backpropagates_through_float_outputs() -> None:
    points = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    camera_poses = torch.tensor([0.5, 1.5], requires_grad=True)
    output = {
        "points": points,
        "camera_poses": camera_poses,
        "integer_metadata": torch.tensor([1, 2], dtype=torch.int64),
    }

    loss, keys = training_benchmark.proxy_training_loss(output)
    loss.backward()

    assert loss.ndim == 0
    assert keys == ["camera_poses", "points"]
    assert points.grad is not None and torch.count_nonzero(points.grad).item() == points.numel()
    assert camera_poses.grad is not None and torch.count_nonzero(camera_poses.grad).item() == camera_poses.numel()


def test_summarize_timings_reports_mean_and_median() -> None:
    summary = training_benchmark.summarize_timings_ms([1.0, 2.0, 6.0])

    assert summary == {
        "mean_ms": 3.0,
        "median_ms": 2.0,
        "min_ms": 1.0,
        "max_ms": 6.0,
    }


def test_make_native_base_kwargs_disables_indexer_without_mutating_source() -> None:
    source = {
        "freeze_encoder": True,
        "indexer_cfg": {"enabled": True, "topk": 1024},
    }

    result = training_benchmark.make_native_base_kwargs(source)

    assert result["freeze_encoder"] is True
    assert result["indexer_cfg"]["enabled"] is False
    assert result["indexer_cfg"]["topk"] == 1024
    assert source["indexer_cfg"]["enabled"] is True


def test_count_trainable_parameters_counts_only_requires_grad() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    for parameter in model[1].parameters():
        parameter.requires_grad = False

    assert training_benchmark.count_trainable_parameters(model) == 9


if __name__ == "__main__":
    test_proxy_training_loss_backpropagates_through_float_outputs()
    test_summarize_timings_reports_mean_and_median()
    test_make_native_base_kwargs_disables_indexer_without_mutating_source()
    test_count_trainable_parameters_counts_only_requires_grad()
    print("ok")
