import torch
import accelerate


def get_accelerator_dtype(
    accelerator: accelerate.Accelerator,
    dtype: torch.dtype | torch.Tensor | str = torch.float32,
) -> torch.dtype:
    if accelerator is not None:
        if accelerator.mixed_precision == "fp16":
            dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            dtype = torch.bfloat16
        elif accelerator.mixed_precision == "fp32":
            dtype = torch.float32
        elif accelerator.mixed_precision == "no":
            dtype = torch.float32
    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    elif isinstance(dtype, torch.Tensor):
        dtype = dtype.dtype
    elif not isinstance(dtype, torch.dtype):
        raise ValueError(f"Invalid dtype: {dtype}")
    return dtype
