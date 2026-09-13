# import torch
# import torch.distributed as dist
# from accelerate import Accelerator

# def main():
#     # 初始化 Accelerator（自动处理多机多卡配置）
#     accelerator = Accelerator()
    
#     # 获取当前进程信息
#     rank = accelerator.process_index
#     world_size = accelerator.num_processes
#     device = accelerator.device

#     print(f"Rank {rank}/{world_size} | Device: {device} | NCCL Version: {torch.cuda.nccl.version()}")

#     for i in range(10000000):
#         # 测试张量同步
#         tensor = torch.tensor([rank], dtype=torch.float32).to(device)
#         dist.all_reduce(tensor, op=dist.ReduceOp.SUM)  # 所有节点求和

#         # 验证结果
#         expected_sum = sum(range(world_size))
#         assert tensor.item() == expected_sum, f"Sync failed! Got {tensor.item()}, expected {expected_sum}"

#     if rank == 0:
#         print("\n✅ All nodes synchronized successfully!")
#         print(f"Sum of ranks: {tensor.item()} (expected: {expected_sum})")

# if __name__ == "__main__":
#     main()

import torch
import torch.distributed as dist
from accelerate import Accelerator
import time

def main():
    accelerator = Accelerator()
    rank = accelerator.process_index
    device = accelerator.device
    print('device:', device)

    # 大张量测试（128MB）
    tensor_size = 32 * 1024 * 1024  # 128MB (float32)
    tensor = torch.rand(tensor_size, dtype=torch.float32).to(device)

    # 预热
    for _ in range(10000):
        dist.all_reduce(tensor)

    # 正式测试
    start = time.time()
    for _ in range(100):
        dist.all_reduce(tensor)
    elapsed = time.time() - start

    # 计算带宽
    bandwidth = (100 * tensor_size * 4 * 2) / (elapsed * 1e9)  # GB/s (float32=4Bytes, 2x for send+recv)
    if rank == 0:
        print(f"RDMA Bandwidth: {bandwidth:.2f} GB/s")

if __name__ == "__main__":
    main()
