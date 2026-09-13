import torch
import torch.distributed as dist

from datetime import timedelta
from typing import no_type_check, Optional, Any

_LOCAL_PROCESS_GROUP = None


def get_distributed():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert _LOCAL_PROCESS_GROUP is not None
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def is_main_process() -> bool:
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    if dist.get_backend() == "nccl" and torch.cuda.is_available():
        try:
            dist.barrier(device_ids=[torch.cuda.current_device()])
            return
        except TypeError:
            pass
    dist.barrier()


@no_type_check
def init_intra_node_process_group(
    num_devices_per_node: int,
    timeout: timedelta = None,
    pg_options: Optional[Any] = None,
) -> dist.ProcessGroup:
    """
    Return a process group across the current node.
    This is a parameterized version of `_init_intra_node_process_group()` in `torch.distributed.fsdp._init_utils.py`

    For example, given each row is a distinct node:
    0  1  2  3  4  5  6  7
    8  9 10 11 12 13 14 15
    This API would return an intra-node subgroup across
    [0, 1, ..., 7] or [8, 9, ..., 15] depending on the process's rank.
    For example, rank 3 would get [0, 1, ..., 7].
    """
    intra_node_subgroup, _ = dist.new_subgroups(num_devices_per_node, timeout=timeout, pg_options=pg_options)
    return intra_node_subgroup


@no_type_check
def init_inter_node_process_group(
    global_process_group: dist.ProcessGroup,
    num_devices_per_node: int,
    timeout: timedelta = None,
    pg_options: Optional[Any] = None,
) -> dist.ProcessGroup:
    """
    Return an inter-node process group where each contained rank has the same local rank.
    This is a parameterized version of `_init_inter_node_process_group()` in `torch.distributed.fsdp._init_utils.py`

    For example, given each row is a distinct node:
    0  1  2  3  4  5  6  7
    8  9 10 11 12 13 14 15
    This API would return inter-node process group [0, 8], [1, 9], [2, 10], and so forth
    depending on the process's rank. For example, rank 1 would get [1, 9], rank 5
    would get [5, 13].
    """
    # the inter-node pg that is returned
    inter_node_pg = None
    sharding_backend = dist.get_backend(global_process_group)
    world_size = dist.get_world_size(global_process_group)
    # Assuming fully homogeneous setup
    num_nodes = world_size // num_devices_per_node
    my_local_rank = dist.get_rank(global_process_group) % num_devices_per_node
    for local_rank in range(num_devices_per_node):
        ranks_for_inter_group = [
            local_rank + (i * num_devices_per_node) for i in range(num_nodes)
        ]
        # every rank always needs to call dist.new_group
        grp = dist.new_group(ranks=ranks_for_inter_group, backend=sharding_backend, timeout=timeout, pg_options=pg_options)
        if local_rank == my_local_rank:
            inter_node_pg = grp

    assert (
        inter_node_pg is not None
    ), f"{my_local_rank} expected to assign inter-node pg, but did not"
    return inter_node_pg
