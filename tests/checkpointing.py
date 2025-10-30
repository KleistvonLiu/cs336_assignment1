# checkpointing.py
from __future__ import annotations
from typing import BinaryIO, IO
import os
import torch

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """
    Serialize model/optimizer state + iteration into `out` (path or file-like).
    """
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": int(iteration),
    }
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Load checkpoint from `src` (path or file-like), restore model/optimizer state,
    and return the saved iteration number.
    """
    # 映射到 CPU 更通用；随后把 optimizer.state 迁移到各参数的 device
    ckpt = torch.load(src, map_location="cpu")

    # 恢复模型与优化器
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    # ☆ 可选的稳健性增强：把优化器里每个参数关联的 state 张量迁移到该参数所在设备
    # 这样即使保存时在CPU、加载后模型在CUDA，也能一致。
    for p in model.parameters():
        state = optimizer.state.get(p, None)
        if state is None:
            continue
        p_device = p.device
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(p_device)

    return int(ckpt.get("iteration", 0))
