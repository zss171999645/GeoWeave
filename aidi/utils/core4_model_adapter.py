from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PreparedCore4Model:
    live_model: Any
    inference_model: Any
    was_training: bool
    model_family: str


def unwrap_training_model(model: Any) -> Any:
    wrapped = model
    while hasattr(wrapped, "module"):
        wrapped = wrapped.module
    return wrapped


def looks_like_native_pi3_model(model: Any) -> bool:
    return (
        not hasattr(model, "vggt")
        and hasattr(model, "encoder")
        and hasattr(model, "decoder")
        and hasattr(model, "point_head")
        and hasattr(model, "camera_head")
    )


def unwrap_official_vggt_model(model: Any) -> Any:
    wrapped = unwrap_training_model(model)
    if not hasattr(wrapped, "vggt"):
        raise TypeError(f"Expected OfficialVGGTModel-like module, got {type(wrapped)!r}")
    return wrapped


def prepare_live_model_for_core4_eval(model: Any, global_step: int) -> PreparedCore4Model:
    live_model = unwrap_training_model(model)
    was_training = bool(live_model.training)
    live_model.eval()

    if hasattr(live_model, "vggt"):
        if hasattr(live_model, "_indexer_state_from_step") and hasattr(live_model.vggt, "aggregator"):
            state = live_model._indexer_state_from_step(global_step)
            live_model.vggt.aggregator.set_indexer_state(state)
        return PreparedCore4Model(
            live_model=live_model,
            inference_model=live_model.vggt,
            was_training=was_training,
            model_family="vggt",
        )

    if looks_like_native_pi3_model(live_model):
        if hasattr(live_model, "set_indexer_state_by_step"):
            live_model.set_indexer_state_by_step(global_step, training=False)
        return PreparedCore4Model(
            live_model=live_model,
            inference_model=live_model,
            was_training=was_training,
            model_family="pi3",
        )

    raise TypeError(f"Expected OfficialVGGTModel-like or native Pi3 module, got {type(live_model)!r}")


def restore_live_model_after_core4_eval(prepared: PreparedCore4Model, global_step: int) -> None:
    live_model = prepared.live_model
    live_model.train(prepared.was_training)
    if (
        prepared.model_family == "vggt"
        and hasattr(live_model, "_indexer_state_from_step")
        and hasattr(live_model.vggt, "aggregator")
    ):
        state = live_model._indexer_state_from_step(global_step)
        live_model.vggt.aggregator.set_indexer_state(state)
    elif prepared.model_family == "pi3" and hasattr(live_model, "set_indexer_state_by_step"):
        live_model.set_indexer_state_by_step(global_step, training=prepared.was_training)
