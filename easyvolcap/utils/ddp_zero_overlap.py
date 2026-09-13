import weakref
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import ddp_zero_hook, default_hooks
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.optim.zero_redundancy_optimizer import _OverlapStatus
from torch.nn.parallel import DistributedDataParallel


def _collect_local_bucket_grads(
    bucket: dist.GradBucket,
    zero: ZeroRedundancyOptimizer,
    rank: int,
) -> List[torch.Tensor]:
    overlap_info = zero._overlap_info
    bucket_index = bucket.index()
    assert bucket_index in overlap_info.offsets, f"Bucket index {bucket_index} was not assigned to rank {rank}"
    gradients_offset = overlap_info.offsets[bucket_index]
    bucket_assignment = zero._bucket_assignments_per_rank[rank][bucket_index]
    bucket_offset = bucket_assignment.offset
    length = len(bucket_assignment.parameters)
    bucket_gradients = bucket.gradients()[bucket_offset: bucket_offset + length]
    return [bucket_gradients[i] for i in range(len(bucket_gradients))]


def _unscale_local_bucket_grads(
    bucket_grads: List[torch.Tensor],
    inv_scale: torch.Tensor,
    found_inf: torch.Tensor,
) -> None:
    per_device_and_dtype_grads: Dict[torch.device, Dict[torch.dtype, List[torch.Tensor]]] = {}
    for grad in bucket_grads:
        if grad is None:
            continue
        per_dtype_grads = per_device_and_dtype_grads.setdefault(grad.device, {})
        per_dtype_grads.setdefault(grad.dtype, []).append(grad)

    if not per_device_and_dtype_grads:
        return

    with torch.no_grad():
        for device, per_dtype_grads in per_device_and_dtype_grads.items():
            device_inv_scale = inv_scale.to(device=device, non_blocking=True)
            device_found_inf = found_inf.to(device=device, non_blocking=True)
            for grads in per_dtype_grads.values():
                torch._amp_foreach_non_finite_check_and_unscale_(
                    grads,
                    device_found_inf,
                    device_inv_scale,
                )
            if device_found_inf.data_ptr() != found_inf.data_ptr():
                found_inf.copy_(device_found_inf.to(device=found_inf.device, non_blocking=True))


def hook_with_zero_step_grad_scaler(
    hook: Callable[[Any, dist.GradBucket], torch.futures.Future],
    ddp: DistributedDataParallel,
    zero: ZeroRedundancyOptimizer,
    shard_buckets: bool = False,
) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    if not zero._overlap_with_ddp:
        raise ValueError("ZeroRedundancyOptimizer must be constructed with overlap_with_ddp=True")

    ddp_ref = weakref.ref(ddp)
    pg = dist.get_backend(ddp_ref().process_group)  # type: ignore[union-attr]
    if (pg != dist.Backend.NCCL) and (pg != "hccl"):
        raise RuntimeError("Overlapping DDP with ZeRO currently requires NCCL/HCCL backend")

    if shard_buckets:
        zero._overlap_info.shard_buckets = True
        zero._overlap_info.total_size = 0

    def hook_with_zero_amp_fn(
        state: Any,
        bucket: dist.GradBucket,
    ) -> torch.futures.Future[torch.Tensor]:
        fut = hook(state, bucket)
        ddp_zero_hook._hook_with_zero_step_setup(ddp_ref, zero, bucket)
        if zero._overlap_info.status != _OverlapStatus.INITIALIZED:
            return fut

        overlap_info = zero._overlap_info
        bucket_index = bucket.index()
        rank = zero.global_rank

        assert overlap_info.status == _OverlapStatus.INITIALIZED
        assert len(overlap_info.assigned_ranks_per_bucket) > bucket_index, "`assigned_ranks_per_bucket` is not fully constructed"
        assigned_to_bucket = rank in overlap_info.assigned_ranks_per_bucket[bucket_index]

        if assigned_to_bucket:
            overlap_info.bucket_index_to_bucket[bucket_index] = bucket
            overlap_info.bucket_index_to_future[bucket_index] = fut

        if len(overlap_info.bucket_indices_seen) > 0:
            assert overlap_info.bucket_indices_seen[-1] == bucket_index - 1, "Bucket indices are not in incremental order"
        else:
            assert bucket_index == 0, "Bucket indices do not start from 0"
        overlap_info.bucket_indices_seen.append(bucket_index)

        num_buckets = len(overlap_info.params_per_bucket)
        if bucket_index != num_buckets - 1:
            return fut

        amp_scale = getattr(zero, "_codex_overlap_amp_scale", None)
        found_inf = None
        if amp_scale is not None:
            found_inf = torch.zeros((), dtype=torch.float32, device=amp_scale.device)
            inv_scale = amp_scale.double().reciprocal().float()
        else:
            inv_scale = None

        local_bucket_grads: Dict[int, List[torch.Tensor]] = {}
        for local_bucket_index in range(num_buckets):
            assigned_ranks = overlap_info.assigned_ranks_per_bucket[local_bucket_index]
            if rank not in assigned_ranks:
                continue

            assert local_bucket_index in overlap_info.bucket_index_to_future, (
                f"All-reduce future for bucket {local_bucket_index} not saved on rank {rank}"
            )
            overlap_info.bucket_index_to_future[local_bucket_index].wait()
            curr_bucket = overlap_info.bucket_index_to_bucket[local_bucket_index]
            if inv_scale is not None:
                local_bucket_grads[local_bucket_index] = _collect_local_bucket_grads(curr_bucket, zero, rank)

        if inv_scale is not None and found_inf is not None:
            for grads in local_bucket_grads.values():
                _unscale_local_bucket_grads(grads, inv_scale, found_inf)

        if hasattr(zero, "codex_apply_overlap_grad_transforms"):
            zero.codex_apply_overlap_grad_transforms()

        should_step = True
        if found_inf is not None:
            should_step = float(found_inf.item()) == 0.0

        if should_step:
            for local_bucket_index in range(num_buckets):
                assigned_ranks = overlap_info.assigned_ranks_per_bucket[local_bucket_index]
                if rank in assigned_ranks:
                    curr_bucket = overlap_info.bucket_index_to_bucket[local_bucket_index]
                    ddp_zero_hook._perform_local_step(curr_bucket, zero, rank)
                ddp_zero_hook._broadcast_bucket(local_bucket_index, zero)
            overlap_info.wait_for_broadcasts()

        overlap_info.clear_per_iter_info()
        if hasattr(zero, "_codex_mark_overlap_step_ready"):
            zero._codex_mark_overlap_step_ready(found_inf)
        return fut

    return hook_with_zero_amp_fn


def register_zero_overlap_comm_hook(
    ddp_model: DistributedDataParallel,
    zero: ZeroRedundancyOptimizer,
    use_grad_scaler_aware: bool = True,
    shard_buckets: bool = False,
) -> None:
    if use_grad_scaler_aware:
        hook = hook_with_zero_step_grad_scaler(
            default_hooks.allreduce_hook,
            ddp_model,
            zero,
            shard_buckets=shard_buckets,
        )
    else:
        hook = ddp_zero_hook.hook_with_zero_step(
            default_hooks.allreduce_hook,
            ddp_model,
            zero,
            shard_buckets=shard_buckets,
        )
    ddp_model.register_comm_hook(state=None, hook=hook)
